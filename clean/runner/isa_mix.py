# SPDX-License-Identifier: MIT
"""Dynamic instruction mix per window of the Swin block kernels, from the compiled ISA (no GPU needed).

    python clean/runner/isa_mix.py --co build/swin.co [--llvm-objdump llvm-objdump]

For every block kernel: instruction counts per window and wave (straight-line code plus loop bodies times their trip
counts: head loop = number of heads, MLP unit loop = number of staged MLP units, tile loop = hidden tiles per unit),
split into matrix (WMMA), vector ALU, FP8/FP16 conversions, transcendentals, LDS, global memory. The model:
  matrix cycles  = 8 per FP8 WMMA (4096 MACs at the measured 407 TFLOPS peak = 512 MAC/clk/SIMD)
  vector slots   = valu + cvt instructions
  If matrix and vector work do NOT overlap on a SIMD the time per window is about matrix + vector (+ transcendentals);
  if they overlap fully it is about the larger of the two. overlap_probe.py measures which one is true.
It is a model of issue slots, not a prediction of time: latency, barriers, memory and occupancy are not in it.
"""
import argparse
import collections
import os
import re
import subprocess
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from clean.runner.packing import block_layout  # noqa: E402

KERNELS = {"swin_block_c64": (64, 2, 256), "swin_block_c64_add": (64, 2, 256), "swin_block_c96": (96, 4, 384),
           "swin_block_c96_add": (96, 4, 384), "swin_block_c128": (128, 4, 512)}
KERNELS.update({n + "_nc": v for n, v in list(KERNELS.items())})  # the clamp-free kernels the runner launches when it can
WMMA_CYCLES = 8


def classify(op):
    if "wmma" in op:
        return "wmma"
    if re.match(r"v_(exp|rcp|rsq)_f32", op):
        return "trans"
    if op.startswith("v_cvt"):
        return "cvt"
    if op.startswith("v_"):
        return "valu"
    if op.startswith("ds_"):
        return "ds"
    if op.startswith("global_load") or op.startswith("global_store"):
        return "gmem"
    if "barrier" in op:
        return "bar"
    if "wait" in op or op.startswith("s_delay"):
        return "wait"
    return "salu"


def disassemble(co, objdump="llvm-objdump"):
    out = subprocess.run([objdump, "-d", "--mcpu=gfx1201", "--no-show-raw-insn", co], capture_output=True, text=True,
                         check=True).stdout
    funcs, cur, start = {}, None, 0
    for line in out.split("\n"):
        m = re.match(r"([0-9a-f]+) <(\w+)>:$", line.strip())
        if m:
            cur, start = m.group(2), int(m.group(1), 16)
            funcs[cur] = []
            continue
        if cur:
            m = re.match(r"\s+(\S+)(.*?)//\s*([0-9A-F]+):.*?(?:<\w+\+0x([0-9a-f]+)>)?\s*$", line)
            if m:
                funcs[cur].append((int(m.group(3), 16) - start, m.group(1), int(m.group(4), 16) if m.group(4) else None))
    return funcs


def dynamic_mix(lines, heads, nmlp, tiles):
    addr = {a: i for i, (a, _, _) in enumerate(lines)}
    by_start = {}
    for i, (a, op, t) in enumerate(lines):
        if (op.startswith("s_cbranch") or op == "s_branch") and t is not None and t < a and t in addr:
            by_start[addr[t]] = max(by_start.get(addr[t], i), i)
    loops = sorted(by_start.items())

    def depth(l):
        return sum(1 for o in loops if o != l and o[0] <= l[0] and o[1] >= l[1])

    d0 = [l for l in loops if depth(l) == 0]
    d1 = [l for l in loops if depth(l) == 1]
    trip = {}
    if len(d0) >= 2:
        trip[d0[0]], trip[d0[1]] = heads, nmlp
    for l in d1:
        trip[l] = tiles
    mix = collections.Counter()
    for i, (_, op, _) in enumerate(lines):
        w = 1
        for (s, e), t in trip.items():
            if s <= i <= e:
                w *= t
        mix[classify(op)] += w
    return mix


def dynamic_ops(lines, heads, nmlp, tiles):
    """Per-window dynamic counts of every vector opcode, by region of the block: before the head loop (load, norm 1),
    head loop (QKV, attention, Wp), between the loops (norm 2), MLP loop, after it (store)."""
    addr = {a: i for i, (a, _, _) in enumerate(lines)}
    by_start = {}
    for i, (a, op, t) in enumerate(lines):
        if (op.startswith("s_cbranch") or op == "s_branch") and t is not None and t < a and t in addr:
            by_start[addr[t]] = max(by_start.get(addr[t], i), i)
    loops = sorted(by_start.items())

    def depth(l):
        return sum(1 for o in loops if o != l and o[0] <= l[0] and o[1] >= l[1])

    d0 = [l for l in loops if depth(l) == 0]
    d1 = [l for l in loops if depth(l) == 1]
    trip = {}
    if len(d0) >= 2:
        trip[d0[0]], trip[d0[1]] = heads, nmlp
    for l in d1:
        trip[l] = tiles
    h0, m0 = (d0[0], d0[1]) if len(d0) >= 2 else (None, None)
    regions = collections.defaultdict(collections.Counter)
    for i, (_, op, _) in enumerate(lines):
        if classify(op) not in ("valu", "cvt", "trans"):
            continue
        w = 1
        for (s_, e), t in trip.items():
            if s_ <= i <= e:
                w *= t
        if h0 is None:
            reg = "all"
        elif i < h0[0]:
            reg = "pre (load, norm 1)"
        elif i <= h0[1]:
            reg = "head loop (qkv, attention, wp)"
        elif i < m0[0]:
            reg = "between (norm 2)"
        elif i <= m0[1]:
            reg = "mlp loop"
        else:
            reg = "post (store)"
        regions[reg][re.sub(r"_e(32|64)$", "", op)] += w
    return regions


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--co", required=True)
    ap.add_argument("--llvm-objdump", default="llvm-objdump")
    ap.add_argument("--ops", metavar="KERNEL", help="print the dynamic count of every vector opcode of this kernel "
                    "(e.g. swin_block_c64), by region of the block, instead of the summary")
    ap.add_argument("--top", type=int, default=14, help="opcodes per region with --ops")
    args = ap.parse_args(argv)
    funcs = disassemble(args.co, args.llvm_objdump)
    if args.ops:
        C, heads, mlp = KERNELS[args.ops]
        lay = block_layout(C, heads, mlp)
        for reg, cnt in dynamic_ops(funcs[args.ops], heads, lay["NMLP"], lay["HPU"]).items():
            print(f"{reg}: {sum(cnt.values())} vector instructions")
            for op, n in cnt.most_common(args.top):
                print(f"    {op:28s} {n:5d}")
        return
    for name, (C, heads, mlp) in KERNELS.items():
        if name not in funcs:
            continue
        lay = block_layout(C, heads, mlp)
        # the MLP tile loop runs HPU tiles per unit; the pipelined variant peels two of them (approximation)
        mix = dynamic_mix(funcs[name], heads, lay["NMLP"], lay["HPU"])
        m, v = mix["wmma"] * WMMA_CYCLES, mix["valu"] + mix["cvt"]
        print(f"{name:20s} wmma {mix['wmma']:4d}  valu {mix['valu']:5d}  cvt {mix['cvt']:4d}  trans {mix['trans']:4d}  "
              f"ds {mix['ds']:4d}  gmem {mix['gmem']:3d} | matrix cycles {m:5d}  vector slots {v:5d}  "
              f"vector/matrix {v / max(m, 1):.2f}  sum {m + v}  max {max(m, v)}")


if __name__ == "__main__":
    main()
