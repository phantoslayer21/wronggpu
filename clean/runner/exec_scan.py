# SPDX-License-Identifier: MIT
"""Matrix and lane-exchange instructions under a lane-divergent exec mask, in gfx11 code objects.

On gfx11 (RDNA 3) v_wmma_* and v_permlanex16_b32 need every lane of the wave executing. The compiler may move such an
instruction into a branch that only some lanes take (a store behind `if (token < ntok)`, a gather behind a bounds test): on a
partly valid tile the instruction then runs with EXEC partly off and gives NaN or wrong values for the lanes that are on, while
the lane emulator (which always runs every lane) never shows it.

This pass disassembles a code object (llvm-objdump) and checks the control-flow graph of every function: a forward data-flow
over the basic blocks tracks whether EXEC may be a partial mask at the start of each block. A mask becomes partial at a
`s_and_saveexec`, at a write to EXEC by `s_and`, `s_and_not1`, `s_xor` and the like, and at `v_cmpx`; it is restored at
`s_or exec, exec, S` or `s_mov exec, S` where S was saved from EXEC earlier, and at `s_mov exec, -1`. Paths merge by "partial if
any path is partial", so a loop body with a divergent exit is partial. Uniform control flow (`s_cbranch_scc*`, `s_cbranch_vcc*`
over a wave-uniform value) does not change the mask and is not flagged. The check is conservative about lane-divergent masks
that happen to be full at run time: a flagged site is a hazard until the source makes the mask full by construction.

    python clean/runner/exec_scan.py <code object or build folder> ... [--objdump <llvm-objdump>] [--mcpu gfx1100]

Exit status 1 when any instruction is flagged. A flagged site prints the function, the address, the instruction and the address
of the instruction that made the mask partial.
"""
import argparse
import os
import re
import subprocess
import sys

NEEDS_FULL_EXEC = re.compile(r"^(v_wmma\w*|v_permlanex16\w*|v_permlane16\w*)$")
INSTR = re.compile(r"^\t(\S+)\s*([^/\n]*?)\s*// ([0-9A-Fa-f]+): ([0-9A-Fa-f]{8})")
FUNC = re.compile(r"^[0-9a-f]+ <(.+)>:$")
EXEC = ("exec_lo", "exec")


def _ops(text):
    """Operands split at top-level commas ('s[4:5]' stays whole)."""
    return [o.strip() for o in text.split(",")] if text.strip() else []


def parse(text):
    """{function: [(addr, mnemonic, operands, first dword)]} from llvm-objdump -d output."""
    funcs, cur = {}, None
    for line in text.splitlines():
        m = FUNC.match(line)
        if m:
            cur = funcs.setdefault(m.group(1), [])
            continue
        m = INSTR.match(line)
        if m and cur is not None:
            cur.append((int(m.group(3), 16), m.group(1), _ops(m.group(2)), int(m.group(4), 16)))
    return funcs


def _branch_target(addr, dword):
    """Target of s_branch / s_cbranch_*: simm16 words from the next instruction (the low half of the first dword)."""
    simm = dword & 0xFFFF
    if simm & 0x8000:
        simm -= 0x10000
    return addr + 4 + 4 * simm


def _is_branch(mn):
    return mn.startswith("s_branch") or mn.startswith("s_cbranch")


def _ends_block(mn):
    return mn in ("s_endpgm", "s_setpc_b64", "s_branch") or mn.startswith("s_cbranch")


def _blocks(instrs):
    """Basic blocks as lists of instruction indices, and the successor block numbers of each."""
    addr_ix = {a: i for i, (a, *_rest) in enumerate(instrs)}
    leaders = {0}
    for i, (a, mn, _ops_, dw) in enumerate(instrs):
        if _is_branch(mn):
            t = addr_ix.get(_branch_target(a, dw))
            if t is not None:
                leaders.add(t)
        if _ends_block(mn) and i + 1 < len(instrs):
            leaders.add(i + 1)
    starts = sorted(leaders)
    block_of = {}
    blocks = []
    for n, s in enumerate(starts):
        e = starts[n + 1] if n + 1 < len(starts) else len(instrs)
        blocks.append(list(range(s, e)))
        for i in range(s, e):
            block_of[i] = n
    succ = []
    for n, ixs in enumerate(blocks):
        a, mn, _o, dw = instrs[ixs[-1]]
        out = []
        if _is_branch(mn):
            t = addr_ix.get(_branch_target(a, dw))
            if t is not None:
                out.append(block_of[t])
        if mn not in ("s_endpgm", "s_setpc_b64", "s_branch") and n + 1 < len(blocks):
            out.append(n + 1)
        succ.append(out)
    return blocks, succ


def _first(a, b):
    """The first of two mask origins that is not None (an origin is an address, and 0 is one)."""
    return a if a is not None else b


def _step(instr, state, loop_restore):
    """Transfer function of one instruction. state = (origin address of the partial mask or None, {saved register: origin}).
    loop_restore[(address, register)] is the mask a `s_or exec, exec, R` after a divergent loop gives back."""
    addr, mn, ops, _dw = instr
    cur, saved = state
    if mn.startswith("v_cmpx"):
        return (_first(cur, addr), saved)
    if not ops:
        return state
    dst = ops[0]
    if "saveexec" in mn:
        if len(ops) > 1 and ops[1] == dst and dst in saved:  # `s_and_not1_saveexec S, S` reuses S: S keeps its first mask
            pass
        elif len(ops) > 1 and mn.startswith("s_and_not1_saveexec") and ops[1] in saved:
            # the else side of an if / else: the join (`s_or exec, exec, D`) gives back the mask the if was entered with
            saved = {**saved, dst: saved[ops[1]]}
        else:
            saved = {**saved, dst: cur}
        if mn.startswith("s_or_saveexec"):
            return (cur, saved)
        return (_first(cur, addr), saved)
    if mn.startswith("s_mov") and len(ops) == 2 and ops[1] in EXEC and dst not in EXEC:
        return (cur, {**saved, dst: cur})
    if dst in EXEC and mn.startswith("s_"):
        if mn.startswith("s_mov"):
            src = ops[1]
            if src in ("-1", "0xffffffff"):
                return (None, saved)
            return (saved.get(src, addr), saved)
        if mn.startswith("s_or_") and len(ops) == 3 and ops[1] in EXEC:
            if ops[2] in saved:
                return (saved[ops[2]], saved)
            return (loop_restore.get((addr, ops[2]), addr), saved)
        return (_first(cur, addr), saved)
    return state


def _join(a, b):
    cur = _first(a[0], b[0])
    saved = dict(a[1])
    for k, v in b[1].items():
        saved[k] = _first(saved[k], v) if k in saved else v
    return (cur, saved)


def _flow(instrs, blocks, succ, loop_restore, forward_only, hits):
    """Data-flow to a fixed point; calls hits(instruction index, state before it) for every instruction."""
    state_in = [None] * len(blocks)
    state_in[0] = (None, {})
    work = [0]
    while work:
        n = work.pop()
        st = state_in[n]
        for i in blocks[n]:
            hits(i, st)
            st = _step(instrs[i], st, loop_restore)
        for s in succ[n]:
            if forward_only and s <= n:
                continue
            new = st if state_in[s] is None else _join(state_in[s], st)
            if state_in[s] is None or new != state_in[s]:
                state_in[s] = new
                work.append(s)


def scan_function(instrs):
    """[(address, mnemonic, origin address)] of matrix / exchange instructions that may run under a partial mask.

    A divergent loop is `s_and_not1 exec, exec, R` (R collects the lanes that left) with a branch back, and `s_or exec, exec, R`
    after it: R was never saved from EXEC, so the mask after the loop is the one the loop was entered with, taken from a first
    pass that follows forward edges only."""
    blocks, succ = _blocks(instrs)
    narrow = {}

    def note(i, st):
        a, mn, ops, _dw = instrs[i]
        if mn.startswith("s_and_not1") and len(ops) == 3 and ops[0] in EXEC and ops[1] in EXEC:
            narrow[(a, ops[2])] = st[0]

    _flow(instrs, blocks, succ, {}, True, note)
    loop_restore = {}
    for i, (a, mn, ops, _dw) in enumerate(instrs):
        if mn.startswith("s_or_") and len(ops) == 3 and ops[0] in EXEC and ops[1] in EXEC:
            before = [(na, r) for (na, r) in narrow if r == ops[2] and na < a]
            if before:
                loop_restore[(a, ops[2])] = narrow[max(before)]
    found = {}

    def check(i, st):
        a, mn, _o, _dw = instrs[i]
        if NEEDS_FULL_EXEC.match(mn) and st[0] is not None:
            found[a] = (a, mn, st[0])

    _flow(instrs, blocks, succ, loop_restore, False, check)
    return sorted(found.values())


def scan_text(text):
    """{function: findings} for the functions of a disassembly that have any, plus the matrix counts per function."""
    out, counts = {}, {}
    for name, instrs in parse(text).items():
        counts[name] = sum(1 for ins in instrs if NEEDS_FULL_EXEC.match(ins[1]))
        f = scan_function(instrs) if counts[name] else []
        if f:
            out[name] = f
    return out, counts


def disassemble(co, objdump=None, mcpu="gfx1100"):
    objdump = objdump or os.path.join(os.environ.get("ROCM_PATH", "/usr/lib64/rocm"), "llvm", "bin", "llvm-objdump")
    return subprocess.run([objdump, "-d", f"--mcpu={mcpu}", co], check=True, capture_output=True, text=True).stdout


def scan_object(co, objdump=None, mcpu="gfx1100"):
    return scan_text(disassemble(co, objdump, mcpu))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("paths", nargs="+", help="code objects, or build folders (every *.co in them)")
    ap.add_argument("--objdump")
    ap.add_argument("--mcpu", default="gfx1100")
    a = ap.parse_args(argv)
    files = []
    for p in a.paths:
        if os.path.isdir(p):
            files += sorted(os.path.join(p, f) for f in os.listdir(p) if f.endswith(".co"))
        else:
            files.append(p)
    bad = 0
    for co in files:
        found, counts = scan_object(co, a.objdump, a.mcpu)
        total = sum(counts.values())
        n = sum(len(v) for v in found.values())
        bad += n
        print(f"{os.path.basename(co)}: {total} matrix/exchange instructions in {sum(1 for c in counts.values() if c)} "
              f"functions, {n} under a partial exec mask")
        for name, items in found.items():
            for addr, mn, origin in items:
                print(f"  {name}: {mn} at {addr:#x}, mask made partial at {origin:#x}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
