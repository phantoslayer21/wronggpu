# SPDX-License-Identifier: MIT
"""GELU chains of a HIP ISA listing (llvm-objdump -d): from every fmaak with the literal 0xc013b8ed, the producer of its
VGPR operand and the first reader of each result, as one signature line per chain. The oracle's code object must show
exactly the chain kinds of the runtime's kernels (run.py checks it).

    python isa_chain.py <listing.s> [kernel symbol ...]
"""
import collections
import re
import sys

_INS = re.compile(r"^\s+(v_[a-z0-9_]+)\s*(.*?)\s*(//.*)?$")


def parse(lines):
    out = []
    for ln in lines:
        m = _INS.match(ln)
        if not m:
            continue
        for part in (ln.split("//")[0]).split("::"):
            part = part.strip()
            if not part.startswith("v_"):
                continue
            op, _, rest = part.partition(" ")
            ops = [o.strip() for o in rest.split(",") if o.strip()]
            ops = [o.split()[0] for o in ops]   # drop op_sel etc.
            out.append((op.replace("v_dual_", "v_"), ops[0] if ops else None, ops[1:]))
    return out


def norm(op):
    return re.sub(r"_e(32|64)$", "", op)


def chains(ins):
    sig = collections.Counter()
    for i, (op, dst, src) in enumerate(ins):
        if norm(op) not in ("v_fmaak_f32",) or "0xc013b8ed" not in src:
            continue
        t = [s for s in src if s.startswith("v")]
        x = None
        steps = []
        if t:
            for j in range(i - 1, -1, -1):
                if ins[j][1] == t[0]:
                    o, _, s = ins[j]
                    steps.append(norm(o) + ("(x,x)" if len(s) >= 2 and s[0] == s[1] else "(?)"))
                    x = s[0] if len(s) >= 2 and s[0] == s[1] else None
                    break
        steps.append("fmaak(s," + ",".join("K" if v == "0xc013b8ed" else ("t" if v == t[0] else v[0]) for v in src) + ")")
        r = dst
        k = i
        for _ in range(6):
            nxt = None
            for j in range(k + 1, len(ins)):
                if r in ins[j][2]:
                    nxt = j
                    break
                if ins[j][1] == r:
                    break
            if nxt is None:
                steps.append("?")
                break
            o, d, s = ins[nxt]
            desc = norm(o)
            if desc == "v_mul_f32":
                desc += "(x)" if x in s else "(?)"
            elif desc == "v_add_f32":
                desc += "(1.0)" if "1.0" in s else "(?)"
            steps.append(desc)
            if desc.startswith("v_cvt_pk_fp8"):
                break
            r, k = d, nxt
        sig[" ".join(steps)] += 1
    return sig


def kernel_lines(text, names):
    if not names:
        return text.splitlines()
    out, keep = [], False
    for ln in text.splitlines():
        m = re.match(r"^[0-9a-f]+ <(.+)>:", ln)
        if m:
            keep = m.group(1) in names
        elif keep:
            out.append(ln)
    return out


def main():
    text = open(sys.argv[1], encoding="utf-8", errors="replace").read()
    for s, n in chains(parse(kernel_lines(text, sys.argv[2:]))).items():
        print(f"{n:5d}  {s}")


if __name__ == "__main__":
    main()
