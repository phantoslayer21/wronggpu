# SPDX-License-Identifier: MIT
"""Exhaustively check the production CPU table for the tiled input's packed FP16 transfer.

Run as a module from the repository root with the path written by ki_s_breakpoints_cpu.cpp.
"""

import argparse
from decimal import Decimal, localcontext
from pathlib import Path

import numpy as np

from clean.k.reference.fp import fma32

FIRST = 0x3B4D2E1D
LAST = 0x3F800000
SHIFT = 12
BUCKET = 1 << SHIFT
BASE = FIRST & ~(BUCKET - 1)
INVALID = 0x80000000


def f32(word):
    return np.uint32(word).view(np.float32)


def round_with_bounds(v, err, x, kind):
    """Resolve FP32 midpoints with Decimal only when the fast interval straddles one."""
    mid = v.astype(np.float32)
    lo = (v - err).astype(np.float32)
    hi = (v + err).astype(np.float32)
    uncertain = np.flatnonzero(lo != hi)
    if len(uncertain):
        with localcontext() as ctx:
            ctx.prec = 80
            ln2 = Decimal(2).ln()
            for i in uncertain:
                d = Decimal.from_float(float(x[i]))
                if kind == "log":
                    exact = d.ln() / ln2
                else:
                    exact = (d * ln2).exp()
                # Pick the FP32 neighbour at the smaller Decimal distance. The
                # wider Decimal error bound must separate the two distances.
                choices = [mid[i], np.nextafter(mid[i], np.float32(-np.inf)),
                           np.nextafter(mid[i], np.float32(np.inf))]
                closest = sorted((abs(exact - Decimal.from_float(float(c))), c)
                                 for c in choices)
                assert closest[1][0] - closest[0][0] > Decimal("1e-74")
                mid[i] = closest[0][1]
    return mid


def log32(x):
    # m in [sqrt(0.5), sqrt(2)], so |t| < 0.172. The omitted terms of
    # 2*atanh(t) are bounded by 2*0.172**25 / (25*(1-0.172**2)).
    # Binary64 unit roundoff is 2**-53. Near m=1 the subtraction is exact
    # (Sterbenz), and a 60*2**-53*|t| bound covers the subsequent operations.
    # The exponent and final addition contribute at most 2**-49 here.
    m, e = np.frexp(x.astype(np.float64))
    low = m < 0.7071067811865476
    m = np.where(low, 2.0 * m, m)
    e = e - low
    t = (m - 1.0) / (m + 1.0)
    t2 = t * t
    poly = 1.0 / 23.0
    for n in range(21, 0, -2):
        poly = poly * t2 + 1.0 / n
    v = e + (2.0 / 0.6931471805599453) * t * poly
    err = 2.0**-46
    return round_with_bounds(v, err, x, "log")


def exp32(x):
    # |t| <= ln(2)/2. The omitted exponential series tail is below
    # 1.42*(ln(2)/2)**15/15!. Horner's method uses 28 binary64 rounded
    # operations, with absolute intermediates below 1.5. Scaling is exact.
    y = x.astype(np.float64)
    n = np.rint(y).astype(np.int32)
    t = (y - n) * 0.6931471805599453
    v = 1.0
    for k in range(14, 0, -1):
        v = 1.0 + (t / k) * v
    return round_with_bounds(np.ldexp(v, n), 2.0**-46, x, "exp")


def reference(zp):
    y = log32(zp) * f32(0x3ED55555)
    return fma32(exp32(y), f32(0x3F870A3D), f32(0xBD6147AE)).astype(np.float16).view(np.uint16)


def decode(entry, low):
    return (entry & 0xffff) + (low >= ((entry >> 16) & 0x1fff))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("production_table", type=Path)
    args = parser.parse_args()
    entries = []
    first_cut = None
    n_codes = 0
    for start in range(BASE, LAST + 1, BUCKET * 128):
        stop = min(start + BUCKET * 128, (LAST | (BUCKET - 1)) + 1)
        words = np.arange(start, stop, dtype=np.uint32).reshape(-1, BUCKET)
        reachable = (words >= FIRST) & (words <= LAST)
        want = reference(np.clip(words, FIRST, LAST).ravel().view(np.float32)).reshape(words.shape)
        for row, ramp, used in zip(words, want, reachable):
            change = np.flatnonzero(ramp != ramp[0])
            initial = int(ramp[0])
            cut = int(change[0]) if len(change) else BUCKET
            entry = initial | (cut << 16)
            low = row & (BUCKET - 1)
            if not np.array_equal(decode(entry, low)[used], ramp[used]):
                entry = INVALID
            else:
                if first_cut is None and cut < BUCKET and used[cut]:
                    mutant = initial | ((cut + 1) << 16)
                    assert decode(mutant, low)[cut] != ramp[cut]
                    first_cut = int(row[cut])
            entries.append(entry)
        n_codes += int(reachable.sum())
    assert n_codes == LAST - FIRST + 1
    assert first_cut is not None
    assert INVALID not in entries, "shader requires all buckets to be encodable"
    # All half inputs include both signed zeros, denormals, infinities and NaNs.
    # The shader retains the original clamp and linear branch verbatim.
    with np.errstate(invalid="ignore"):
        half = np.arange(65536, dtype=np.uint16).view(np.float16).astype(np.float32)
        clamped = np.fmin(np.fmax(half, np.float32(0)), np.float32(1))
    nonlinear = clamped > f32(FIRST - 1)
    hwords = clamped[nonlinear].view(np.uint32)
    enc = np.asarray(entries, dtype=np.uint32)[(hwords - BASE) >> SHIFT]
    got = np.empty(65536, dtype=np.uint16)
    want = np.empty_like(got)
    got[nonlinear] = decode(enc, hwords & (BUCKET - 1))
    want[nonlinear] = reference(clamped[nonlinear])
    linear = clamped[~nonlinear]
    linear32 = linear * f32(0x414EB852)
    # wg_f2h uses a biased FP32 addition for FP16 subnormals.
    words = linear32.view(np.uint32)
    mag = words & 0x7fffffff
    normal = (mag + np.uint32(0xc8000fff) + ((mag >> 13) & 1)) >> 13
    subnormal = (mag.view(np.float32) + np.float32(0.5)).view(np.uint32) - 0x3f000000
    got[~nonlinear] = (words >> 16 & 0x8000) | np.where(mag < 0x38800000, subnormal, normal)
    want[~nonlinear] = linear32.astype(np.float16).view(np.uint16)
    assert np.array_equal(got, want)
    stored = np.fromfile(args.production_table, dtype="<u4")
    np.testing.assert_array_equal(stored, np.asarray(entries, dtype=np.uint32))
    print(f"{n_codes:,} nonlinear FP32 inputs and 65,536 half inputs checked; "
          f"{len(entries):,} buckets, {entries.count(INVALID)} fallback buckets; "
          f"cut+1 mutant caught at 0x{first_cut:08x}")


if __name__ == "__main__":
    main()
