# SPDX-License-Identifier: GPL-3.0-only
"""ISA rules of the hybrid kernels' D3D12 code (the hybrid check design: the byte legs and these rules catch a driver that
contracts an FP16 product into an add, or folds the FP16 rounding before an FP8 point into the FP8 conversion).

1. Fused squares. The hybrid form's FP16 sums follow the original's order with every square rounded first
   (q = RN16(x x)); the driver fuses a square feeding an add into one instruction when it can see the pattern, even under
   `precise` (seen on gfx1201: v_pk_fma_f16 v11, v17, v17, v11 in the select-free norm tree). The kernels hide the
   pattern with an OR against a zero the compiler cannot prove (hlsl/hyb_block.hlsli HYB_OPAQUE0); a future driver that
   sees through it would bring the fusion back. The rule fails on any fused multiply-add whose two multiplicands hold
   the same value half (a square):
     v_pk_fma_f16, v_fma_f16, v_fmac_f16, v_pk_fmac_f16, v_fmaak_f16, v_fma_mix{lo,hi}_f16, v_fma_mix_f32, v_fma_f32,
     v_fmac_f32, v_fmaak_f32, v_fma_dx9_zero_f32, v_fmac_dx9_zero_f32, and the dual-issue (VOPD) halves v_dual_fmac_f32,
     v_dual_fmaak_f32
   "The same value": the same register, or registers that a copy (v_mov_b32 / v_mov_b16 / v_dual_mov_b32 of a VGPR)
   made equal on every path to the instruction (must-copies, through the basic blocks: a compiler that copies x to a
   second register and fuses x * copy is caught); the same op_sel / op_sel_hi bits for both sources; neg / abs
   modifiers ignored (-x x is a square too). FP16 forms fail with any addend. The float32 / mixed forms fail when the
   addend is a VGPR (a running sum, as in the norm tree; v_fmac always adds its destination): the softmax's float32
   steps e(u') compute -u u + constant with a literal / SGPR addend, exact by design (hlsl/wg_hyb.hlsli
   hyb_softmax_e8), and are not flagged.
2. Conversions before FP8 points. Every FP8 point of the hybrid form is FP8(RN16(float32 value)): two roundings by
   design (wg_hyb.hlsli hyb_q8_of_chain). In the ISA, both sources of every v_cvt_pk_fp8_f32 must be FP16 values: written
   by v_cvt_f32_f16 (the exact widening after the RN16), or by the saturating pack's steps on such values only
   (v_mov_b32, v_cndmask_b32, v_max / v_min / v_max3 / v_min3 / v_maxmin / v_minmax / v_med3 (_num)_f32 with constants:
   the ReLU and the clamps; the NaN quieting v_or_b32 with 0x400000), on every path to the conversion (a dataflow over
   the basic blocks). A constant is a literal, or a scalar register that s_mov_b32 / s_movk_i32 set to a literal on
   every path (any other write makes it unknown); an unknown scalar register is not an FP16 value. Anything else
   (a WMMA result, an FMA, a load, an instruction the rule does not know) fails: a driver that drops the RN16 or folds
   it into the FP8 conversion leaves the float32 value as the source.

Each rule has its negative control: every model hybrid kernel's mutant (*_scanmut: no contraction barrier, no RN16
before the FP8 pack, HYB_SCAN_MUTANT=1; build_dx12.py) must FAIL rule 2, and rule 1 too unless the kernel has no norm
tree; tests/test_hyb_isa_scan.py checks both rules on constructed ISA (a copied-register square, a removed widening).

    python clean/dx12/hyb_isa_scan.py <isa.txt or dir> ...      exit 1 when a rule fails
build_dx12.py runs both rules on every *_hyb* kernel it builds (no skip switch).
"""
import os
import re
import sys

FMA_OPS = {"v_pk_fma_f16", "v_fma_f16", "v_fmac_f16", "v_pk_fmac_f16", "v_fmaak_f16", "v_fma_mixlo_f16",
           "v_fma_mixhi_f16", "v_fma_mix_f32", "v_fma_f32", "v_fmac_f32", "v_fmaak_f32", "v_fma_dx9_zero_f32",
           "v_fmac_dx9_zero_f32"}
F16_FMA = {"v_pk_fma_f16", "v_fma_f16", "v_fmac_f16", "v_pk_fmac_f16", "v_fmaak_f16"}
FP8_PACK = {"v_cvt_pk_fp8_f32", "v_cvt_sr_fp8_f32"}
WIDEN = {"v_cvt_f32_f16"}
# ops that keep a value an FP16 value when every VGPR source is one (constants: the pack's bounds)
KEEP = {"v_mov_b32", "v_cndmask_b32", "v_max_num_f32", "v_min_num_f32", "v_maxmin_num_f32", "v_minmax_num_f32",
        "v_med3_num_f32", "v_max3_num_f32", "v_min3_num_f32", "v_max_f32", "v_min_f32", "v_maxmin_f32",
        "v_minmax_f32", "v_med3_f32", "v_max3_f32", "v_min3_f32"}
QUIET_NAN = "0x400000"
NO_DEST = re.compile(r"^(s_|v_cmp_|v_cmpx_|v_readlane|v_readfirstlane|.*_store|ds_write|exp\b|v_nop)")
VREG = re.compile(r"^v(\d+)(?:\.(l|h))?$")
SREG = re.compile(r"^s(\d+)$")
SRANGE = re.compile(r"^s\[(\d+):(\d+)\]$")
LITERAL = re.compile(r"^-?(0x[0-9a-fA-F]+|\d+(\.\d+)?(e[-+]?\d+)?)$")
# scalar instructions whose first operand is read, not written
S_NO_DEST = re.compile(r"^(s_cmp|s_bitcmp|s_cbranch|s_branch|s_wait|s_delay_alu|s_nop|s_barrier|s_endpgm|s_sendmsg|"
                       r"s_setprio|s_sleep|s_clause|s_set_inst_prefetch|s_trap|s_setpc|.*_store)")
VRANGE = re.compile(r"^v\[(\d+):(\d+)\]$")
MOD = re.compile(r"\b[a-z_]+:(?:\[[^\]]*\]|[^\s,]+)")
LABEL = re.compile(r"^\s*([A-Za-z_][\w.$]*):\s*(?://.*)?$")


class Ins:
    __slots__ = ("line", "op", "ops", "mods", "text")

    def __init__(self, line, op, ops, mods, text):
        self.line, self.op, self.ops, self.mods, self.text = line, op, ops, mods, text


def _norm_op(op):
    op = re.sub(r"_e(32|64|64_dpp|32_dpp)$", "", op)
    return op.replace("v_dual_", "v_", 1) if op.startswith("v_dual_") else op


def _split_one(line, text):
    text = text.strip()
    if not text:
        return None
    head, _, rest = text.partition(" ")
    # key:value modifiers (op_sel:[0,1,0], quad_perm:[...], offset:16; their brackets hold commas) first
    mods = [m.group(0) for m in MOD.finditer(rest)]
    rest = MOD.sub(" ", rest)
    parts = [p.strip() for p in rest.split(",")] if rest.strip() else []
    ops = []
    for p in parts:
        toks = p.split()
        if not toks:
            continue
        ops.append(toks[0])
        mods += toks[1:]   # bare modifiers (clamp, ...)
    return Ins(line, _norm_op(head), ops, " ".join(mods), text)


def parse(text):
    """[(label or None, Ins or None)] in text order (labels as their own entries); VOPD lines give two Ins."""
    out = []
    for n, raw in enumerate(text.splitlines(), 1):
        m = LABEL.match(raw)
        if m:
            out.append((m.group(1), None))
            continue
        code = raw.split("//", 1)[0].strip()
        if not code or code.startswith((";", ".")):
            continue
        for half in code.split("::"):
            ins = _split_one(n, half)
            if ins:
                out.append((None, ins))
    return out


def _reg(op):
    return op.strip().lstrip("-").strip("|").strip()


def _vregs(tok):
    """32-bit VGPR numbers a token names (a register, a half, a range); [] for anything else."""
    t = _reg(tok)
    m = VREG.match(t)
    if m:
        return [int(m.group(1))]
    m = VRANGE.match(t)
    if m:
        return list(range(int(m.group(1)), int(m.group(2)) + 1))
    return []


def _dest(ins):
    if NO_DEST.match(ins.op) or not ins.ops:
        return []
    return _vregs(ins.ops[0])


def blocks(items):
    """Basic blocks [(label set, [Ins], successor labels, falls through)] of a parsed program."""
    bl, cur, labels = [], [], []

    def close(succ, fall):
        bl.append((labels[:], cur[:], succ, fall))

    for lab, ins in items:
        if lab is not None:
            if cur:
                close([], True)
                cur.clear()
                labels.clear()
            labels.append(lab)
            continue
        cur.append(ins)
        if ins.op == "s_branch" and ins.ops:
            close([ins.ops[0]], False)
        elif ins.op.startswith("s_cbranch") and ins.ops:
            close([ins.ops[0]], True)
        elif ins.op in ("s_endpgm", "s_setpc_b64", "s_trap"):
            close([], False)
        else:
            continue
        cur.clear()
        labels.clear()
    if cur or labels:
        close([], False)
    return bl


def _solve(bl, entry, transfer, meet):
    """Forward must-analysis over the blocks: in-state of each block (None = not reached yet)."""
    idx = {}
    for i, (labs, _, _, _) in enumerate(bl):
        for lab in labs:
            idx[lab] = i
    preds = [[] for _ in bl]
    for i, (_, _, succ, fall) in enumerate(bl):
        for s in succ:
            if s in idx:
                preds[idx[s]].append(i)
        if fall and i + 1 < len(bl):
            preds[i + 1].append(i)
    ins_state = [None] * len(bl)
    out_state = [None] * len(bl)
    changed = True
    while changed:
        changed = False
        for i, (_, body, _, _) in enumerate(bl):
            ps = [out_state[p] for p in preds[i] if out_state[p] is not None]
            st = entry() if i == 0 else (meet(ps) if ps else None)
            if st is None:
                continue
            if ins_state[i] != st or out_state[i] is None:
                ins_state[i] = st
                o = st
                for ins in body:
                    o = transfer(o, ins, None)
                if o != out_state[i]:
                    out_state[i] = o
                    changed = True
    return ins_state


# ---- rule 1: fused squares (with must-copies) ----

def _alias_kill(al, d):
    al = dict(al)
    for k in [k for k in al if k == d or k.startswith(d + ".")]:
        del al[k]
    members = sorted(k for k, r in al.items() if r == d or r.startswith(d + "."))
    if members:
        for k in members:
            r = al[k]
            al[k] = members[0] + r[len(d):] if r != d else members[0]
    return al


def _root(al, tok):
    t = _reg(tok)
    if t in al:
        return al[t]
    base, dot, half = t.partition(".")
    if dot and base in al:
        return al[base] + "." + half
    return t


def _alias_transfer(al, ins, hits):
    if hits is not None and ins.op in FMA_OPS and len(ins.ops) >= 3:
        _check_fma(al, ins, hits)
    dregs = _dest(ins)
    if not dregs:
        return al
    tok = _reg(ins.ops[0])
    src = _reg(ins.ops[1]) if len(ins.ops) > 1 else ""
    al = dict(al)
    for r in dregs:
        al = _alias_kill(al, "v%d" % r)
    if ins.op in ("v_mov_b32", "v_mov_b16") and len(ins.ops) == 2 and (VREG.match(src)) and not ins.mods:
        if src.split(".")[0] != tok.split(".")[0]:
            al[tok] = _root(al, src)
    return al


def _alias_meet(states):
    first = states[0]
    return {k: v for k, v in first.items() if all(s.get(k) == v for s in states[1:])}


def _bits(mods, key):
    m = re.search(key + r":\[([01,\s]+)\]", mods)
    return [int(b) for b in m.group(1).replace(" ", "").split(",")] if m else None


def _check_fma(al, ins, hits):
    ops = ins.ops
    a, b = _reg(ops[1]), _reg(ops[2])
    if not (a.startswith("v") and b.startswith("v")):
        return
    if a != b and _root(al, a) != _root(al, b):
        return
    sel, hi = _bits(ins.mods, "op_sel"), _bits(ins.mods, "op_sel_hi")
    if sel and len(sel) > 1 and sel[0] != sel[1]:
        return
    if hi and len(hi) > 1 and hi[0] != hi[1]:
        return
    if ins.op not in F16_FMA:
        if "fmaak" in ins.op:
            return   # float32: a literal addend
        addend = _reg(ops[0]) if "fmac" in ins.op else (_reg(ops[3]) if len(ops) > 3 else "")
        if not re.match(r"v[0-9]|v\[", addend):
            return
    hits.append((ins.line, ins.text))


def fused_squares(text):
    """[(line number, instruction text)] of fused multiply-adds whose two multiplicands are one value half."""
    bl = blocks(parse(text))
    if not bl:
        return []
    states = _solve(bl, dict, _alias_transfer, _alias_meet)
    hits = []
    for st, (_, body, _, _) in zip(states, bl):
        if st is None:
            st = {}
        for ins in body:
            st = _alias_transfer(st, ins, hits)
    return sorted(set(hits))


# ---- rule 2: FP16 values into the FP8 conversions ----

ALL = None   # the meet's top: every register an FP16 value (blocks not reached yet)


def _sdest(ins):
    """Scalar registers an instruction writes ("s<n>" names): the first operand of a scalar instruction (but compares,
    branches, waits, stores) or of a vector instruction that writes a scalar (compares, readlane)."""
    if not ins.ops or (ins.op.startswith("s_") and S_NO_DEST.match(ins.op)):
        return []
    if not (ins.op.startswith("s_") or ins.op.startswith(("v_cmp_", "v_readlane", "v_readfirstlane"))):
        return []
    t = _reg(ins.ops[0])
    m = SREG.match(t)
    if m:
        return ["s" + m.group(1)]
    m = SRANGE.match(t)
    return ["s%d" % i for i in range(int(m.group(1)), int(m.group(2)) + 1)] if m else []


def _value_ok(tok, ex):
    """A source that is an FP16 value: a VGPR the dataflow says is one, a literal, or a constant scalar register."""
    t = _reg(tok)
    rs = _vregs(t)
    if rs:
        return all(r in ex for r in rs)
    if LITERAL.match(t):
        return True
    m = SREG.match(t)
    return bool(m) and ("s" + m.group(1)) in ex


def _exact_transfer(ex, ins, bad):
    srcs = ins.ops[1:]
    if bad is not None and ins.op in FP8_PACK:
        for s in srcs[:2]:
            if not _value_ok(s, ex):
                bad.append((ins.line, ins.text))
                break
    sd = _sdest(ins)
    if sd:   # scalar provenance: a constant only from a literal (or a constant) moved in on every path
        ex = set(ex)
        const = ins.op in ("s_mov_b32", "s_movk_i32") and len(sd) == 1 and len(srcs) == 1 and _value_ok(srcs[0], ex) \
            and not _vregs(srcs[0])
        ex.difference_update(sd)
        if const:
            ex.add(sd[0])
        ex = frozenset(ex)
    dregs = _dest(ins)
    if not dregs:
        return ex
    vsrc = [r for s in srcs for r in _vregs(s)]
    if ins.op in WIDEN:
        keep = True
    elif ins.op in KEEP:
        vals = srcs[:2] if ins.op == "v_cndmask_b32" else srcs   # (cndmask's third operand: the lane mask)
        keep = all(_value_ok(s, ex) for s in vals) and not ("v_cndmask" in ins.op and ins.mods and "neg" in ins.mods)
    elif ins.op == "v_or_b32":
        lits = [_reg(s) for s in srcs if not _vregs(s)]
        keep = len(vsrc) == 1 and lits == [QUIET_NAN] and vsrc[0] in ex
    else:
        keep = False
    if any(t in ins.mods for t in ("clamp", "mul:", "div:")):
        keep = False
    ex = set(ex)
    full = VREG.match(_reg(ins.ops[0]))
    if keep and len(dregs) == 1 and full and not full.group(2):   # a whole 32-bit register (a half write: not)
        ex.add(dregs[0])
    else:
        ex.difference_update(dregs)
    return frozenset(ex)


def _exact_meet(states):
    out = states[0]
    for s in states[1:]:
        out = out & s
    return out


def fp8_sources_not_fp16(text):
    """[(line number, instruction text)] of FP8 conversions with a source that is not an FP16 value on every path."""
    bl = blocks(parse(text))
    if not bl:
        return []
    states = _solve(bl, frozenset, _exact_transfer, _exact_meet)
    bad = []
    for st, (_, body, _, _) in zip(states, bl):
        if st is None:
            continue   # unreachable
        for ins in body:
            st = _exact_transfer(st, ins, bad)
    return sorted(set(bad))


def fp8_conversions(text):
    return sum(1 for _, ins in parse(text) if ins is not None and ins.op in FP8_PACK)


def main(argv):
    files = []
    for p in argv:
        files += [os.path.join(p, f) for f in sorted(os.listdir(p)) if f.endswith(".txt")] if os.path.isdir(p) else [p]
    bad = 0
    for f in files:
        text = open(f, encoding="utf-8", errors="replace").read()
        hits, conv = fused_squares(text), fp8_sources_not_fp16(text)
        print(f"{os.path.basename(f)}: {len(hits)} fused square(s), {len(conv)} of {fp8_conversions(text)} FP8 "
              "conversion(s) with a source that is not an FP16 value")
        for ln, ins in (hits[:4] + conv[:4]):
            print(f"  line {ln}: {ins}")
        bad += len(hits) + len(conv)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
