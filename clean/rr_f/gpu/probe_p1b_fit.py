# SPDX-License-Identifier: MIT
"""Offline analysis of probe P1b (probe_p1b.py's npz): accuracy of v_wmma_f32_16x16x16_fp8_fp8 and
v_wmma_f32_16x16x16_f16 against the exact sum of the 16 products and C.

    python -m clean.rr_f.gpu.probe_p1b_fit [npz]

Prints per data mode: the largest |D - exact| relative to the largest |term| (terms = the 16 products and C), the
share of results that differ from the round-to-nearest f32 value of the exact sum (all lanes, and lanes with C = 0),
and how many results an "align and truncate" model explains (every term floored to a multiple of 2^(E - F), E = the
exponent of the largest term, the sum then rounded to nearest f32)."""
import math
import os
import sys
from fractions import Fraction

import numpy as np

from clean.rr_f.gpu import build, probe_p1


def to_f32(x):
    """Round a Fraction to the nearest f32 (ties to even); normal range only (enough for these magnitudes)."""
    if x == 0:
        return 0.0
    f = float(x)                         # nearest double, then nearest f32 with the exact remainder as tie-breaker
    r = np.float32(f)
    r64 = float(r)
    if Fraction(r64) != x:
        other = float(np.nextafter(r, np.float32(np.inf) if Fraction(r64) < x else np.float32(-np.inf)))
        d1, d2 = abs(Fraction(r64) - x), abs(Fraction(other) - x)
        if d2 < d1 or (d2 == d1 and (np.float32(other).view(np.uint32) & 1) == 0):
            r64 = other
    return r64


def floor_model(terms, F):
    terms = [Fraction(float(t)) for t in terms if t != 0]
    if not terms:
        return 0.0
    m = max(abs(t) for t in terms)
    e = math.frexp(float(m))[1] - 1
    q = Fraction(2) ** (e - F)
    return to_f32(sum(((t / q).numerator // (t / q).denominator) * q for t in terms))


def main(argv=None):
    argv = argv if argv is not None else sys.argv[1:]
    path = argv[0] if argv else os.path.join(build.BUILD, "p1b.npz")
    z = np.load(path)
    A, B, C, modes = z["A"], z["B"], z["C"].astype(np.float64), z["modes"]
    A1, B1 = probe_p1.frag_mats(A, B)
    P = (A1[:, :, :, None] * B1[:, None, :, :]).transpose(0, 1, 3, 2)          # [w, r, c, k]
    Pk = np.stack([probe_p1.mat_to_acc(P[..., k]) for k in range(16)], axis=-1)  # [w, lane, v, k]
    rng = np.random.default_rng(0)
    for name, D in (("fp8", z["D8"].astype(np.float64)), ("f16", z["D16"].astype(np.float64))):
        print(f"== v_wmma_f32_16x16x16_{name}")
        for md in sorted(set(modes.tolist())):
            sel = np.argwhere((modes[:, None, None] == md) & np.isfinite(C))
            pick = sel[rng.choice(len(sel), min(500, len(sel)), replace=False)]
            worst, nonrn, nonrn0, n0, fit = 0.0, 0, 0, 0, 0
            for w, l, v in pick:
                terms = list(Pk[w, l, v]) + [C[w, l, v]]
                ex = sum((Fraction(float(t)) for t in terms), Fraction(0))
                mt = max(abs(float(t)) for t in terms)
                if mt > 0:
                    worst = max(worst, abs(float(Fraction(D[w, l, v]) - ex)) / mt)
                bad = to_f32(ex) != D[w, l, v]
                nonrn += bad
                if C[w, l, v] == 0:
                    n0 += 1
                    nonrn0 += bad
                fit += floor_model(terms, 25) == D[w, l, v]
            print(f"mode {md}: max|err|/max|term| = 2^{math.log2(worst) if worst else -math.inf:.2f}, "
                  f"non-RN {nonrn}/{len(pick)} (C = 0: {nonrn0}/{n0}), floor model F=25 explains {fit}/{len(pick)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
