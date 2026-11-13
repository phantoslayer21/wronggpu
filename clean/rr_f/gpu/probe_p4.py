# SPDX-License-Identifier: MIT
"""Probes P2-P4 (DESIGN-51.md section 9; spec tests 51.8 items 2 and 3): the kernels' Q8, attention weight,
activation and r = R(1 / Z) for every f16 input on the GPU, against the reference tables.

    python -m clean.rr_f.gpu.probe_p4 [--nolock]

Attention weight and activation for the three constant sets of the vectors (A, B, zovf), and both f16 halves of the
packed helpers. Q8: exact codes (the zero codes NOT counted equal: 51.8 item 3). Reciprocal: for Z >= 2^-8 (51.3 step 4)
and +inf; outside that range r is never formed."""
import argparse
import sys

import numpy as np

from clean.rr_f.gpu import build, stage
from clean.rr_f.reference.fmt import ALL_F16, q8, to_f16_bits
from clean.rr_f.reference.scalar import activation, attention_weight, recip_table
from clean.rr_f.tests.vectors_io import load_manifest, window_consts


def prepare(ar):
    """Adds P2-P4 to an arena: returns (kernels, dispatches, analyze(img) -> ok)."""
    man = load_manifest()
    dx = build.compile_kernel("rr_probe.hlsl", "rr_p4_q8", {})
    n = 65536
    runs = []
    for flav in ("A", "B", "zovf"):
        wc = window_consts(man, flav)
        inp = np.concatenate([np.arange(n, dtype=np.uint32), np.array(stage.wc_dwords(wc), np.uint32)])
        pin = ar.add(inp)
        pout = ar.reserve(n * 16)
        runs.append((flav, wc, pin, pout))
    return ({"p4": (dx, 3)}, [("p4", n // 128, [pin, pout, n]) for _, _, pin, pout in runs],
            lambda img: _analyze(img, runs, n))


def _analyze(img, runs, n):
    bad = 0
    exp_q = q8(ALL_F16)
    rec = to_f16_bits(recip_table())
    zr = (ALL_F16 >= 2.0 ** -8) | (ALL_F16 == np.inf)
    for flav, wc, pin, pout in runs:
        o = img[pout:pout + n * 16].copy().view(np.uint32).reshape(n, 4)
        dq = int((o[:, 0] != exp_q).sum())
        with np.errstate(all="ignore"):
            w = attention_weight(ALL_F16, wc, check=False)
        okw = ~np.isnan(w)                       # the range of 51.3 step 3 (the converter rejects the rest)
        wb = to_f16_bits(np.where(okw, w, 0.0)).astype(np.uint32)
        gw = o[:, 1]
        dw = int(((gw & 0xffff) != wb)[okw].sum() + ((gw >> 16) != wb)[okw].sum())
        act = activation(ALL_F16, wc)
        ab = to_f16_bits(act).astype(np.uint32)
        ga = o[:, 2]
        an = np.isnan(act)
        lo, hi = ga & 0xffff, ga >> 16
        isn = lambda b: ((b & 0x7c00) == 0x7c00) & ((b & 0x3ff) != 0)
        da = int(((lo != ab) & ~(an & isn(lo))).sum() + ((hi != ab) & ~(an & isn(hi))).sum())
        dr = int(((o[:, 3] & 0xffff) != rec)[zr].sum())
        print(f"P4 set {flav:5s}: Q8 differ {dq}, attention weight differ {dw} (of {2 * int(okw.sum())} in range), "
              f"activation differ {da} (NaN = NaN), R(1/Z) differ {dr} (of {int(zr.sum())})")
        bad += dq + dw + da + dr
        if dq:
            i = np.nonzero(o[:, 0] != exp_q)[0][:5]
            print("   Q8 e.g.", [(hex(int(k)), hex(int(o[k, 0])), hex(int(exp_q[k]))) for k in i])
        if da:
            i = np.nonzero((lo != ab) & ~(an & isn(lo)))[0][:5]
            print("   act e.g.", [(hex(int(k)), hex(int(lo[k])), hex(int(ab[k]))) for k in i])
        if dw:
            i = np.nonzero(((gw & 0xffff) != wb) & okw)[0][:5]
            print("   w e.g.", [(hex(int(k)), hex(int(gw[k] & 0xffff)), hex(int(wb[k]))) for k in i])
    print("P2-P4:", "all identical" if bad == 0 else f"{bad} differences")
    return bad == 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--nolock", action="store_true")
    a = ap.parse_args(argv)
    ar = build.Arena()
    k, d, an = prepare(ar)
    img, _ = build.run(ar, k, d, lock=not a.nolock)
    return 0 if an(img) else 1


if __name__ == "__main__":
    sys.exit(main())
