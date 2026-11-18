# SPDX-License-Identifier: MIT
"""Enumerate every finite FP32 log2 argument in [1, +inf) for tiled HDR phi.

Pass the generated dx12/gen/k_fp_tables.hlsli as the only argument. Baseline
FP32 operations mirror kf2_log2. The baseline binary64 branch is enclosed
using the independent series below. Exceptional inputs use the same kf_log2.
"""
import re
import sys
from pathlib import Path

import numpy as np


F = np.float32
D = np.float64
U = np.uint32
MANTISSA_COUNT = 1 << 23


def fma(a, b, c):
    # A binary32 product is exact in binary64. Recover the lost addend at
    # binary32 midpoints to eliminate the possibility of double rounding.
    p = np.asarray(a, dtype=D) * np.asarray(b, dtype=D)
    c = np.asarray(c, dtype=D)
    s = p + c
    r = s.astype(F)
    diff = s - r
    midpoint = (diff != 0) & (np.abs(diff) == np.ldexp(D(1), np.frexp(s)[1] - 25))
    # A downward step from a power of two has half the usual spacing.
    # frexp(s) samples the midpoint's exponent rather than r's exponent.
    if np.any(midpoint):
        tail = s[midpoint] - p[midpoint]
        missing = (p[midpoint] - (s[midpoint] - tail)) + (np.broadcast_to(c, s.shape)[midpoint] - tail)
        change = midpoint.copy()
        change[midpoint] = np.sign(missing) == np.sign(diff[midpoint])
        target = np.copysign(F(np.inf), diff[change])
        r[change] = np.nextafter(r[change], target)
    return r


def fast_sum(a, b):
    s = a + b
    return s, b - (s - a)


def two_sum(a, b):
    s = a + b
    bb = s - a
    return s, (a - (s - bb)) + (b - bb)


def prod(a, b):
    p = a * b
    return p, fma(a, b, -p)


def add(a, b):
    h, l = two_sum(a[0], b[0])
    return fast_sum(h, l + (a[1] + b[1]))


def mul(a, b):
    h, l = prod(a[0], b[0])
    return fast_sum(h, fma(a[0], b[1], fma(a[1], b[0], l)))


def rounding_gap(h, l):
    bits = h.view(U)
    exponent = (bits >> 23) & 255
    toward_zero = ((l.view(U) ^ bits) & ((bits & 0x7fffff) - 1)) >> 31
    return ((exponent - 24 - toward_zero) << 23).view(F)


def oracle_fraction(m):
    # log2(m) = 2/log(2) * (s + s^3/3 + ...), 0 <= s < 1/3.
    # Neglected tail after 36 terms is below 2^-111 relative; all retained
    # terms are positive, with at most 150 binary64 roundings on any term.
    # Sterbenz makes m-1 exact and m+1 is exact for binary32 m.
    s = (m - 1) / (m + 1)
    z = s * s
    p = np.full_like(s, 1 / 71)
    for k in range(34, -1, -1):
        p = p * z + 1 / (2 * k + 1)
    return (2 / 0.69314718055994530942) * s * p


def table(name, source):
    line = re.search(r"\b" + name + r"\[\d+\] = \{([^}]+)\}", source)[1]
    return np.array([int(x, 16) for x in re.findall(r"0x[0-9A-F]+", line)], dtype=U).view(F)


def const(name, source):
    return U(int(re.search(r"#define " + name + r" asfloat\((0x[0-9A-F]+)u\)", source)[1], 16)).view(F)


def verify(source, limit):
    invs, highs, lows = (table(f"KF_LOG_{n}", source) for n in ("INV", "H", "L"))
    ih, il, th, tl, sixth = (const(n, source) for n in
                             ("KF2_INV_LN2_H", "KF2_INV_LN2_L", "KF_THIRD_H", "KF_THIRD_L", "KF_SIXTH"))
    counts = dict(finite=0, candidate_fallback=0, baseline_fallback_on_accept=0, mutant_wrong=0)
    first_mutant = None
    max_ratio = 0.0
    for start in range(0, limit, 1 << 17):
        mantissa = np.arange(start, min(limit, start + (1 << 17)), dtype=U)
        m0 = (mantissa | 0x3f800000).view(F)
        fold = m0 >= F(1.5)
        m = np.where(fold, m0 * F(0.5), m0)
        j = np.rint((m - F(1)) * F(128)).astype(np.int32) + 32
        inv, hi, lo = invs[j], highs[j], lows[j]
        p, dt = prod(m, inv)
        t = p - F(1)
        lead, lead_tail = prod(t, ih)
        q = fma(t, fma(t, fma(t, F(0.2885390082), F(-0.3606737602)), F(0.4808983470)), F(-0.7213475204))
        dt_scaled = fma(dt, fma(-t, ih, ih), fma(t, il, lead_tail))
        corr = fma(t * t, q, dt_scaled + lo)
        vh, vl = fast_sum(hi, lead)
        vtail = vl + corr

        # Baseline kf2_log2, through the last exponent addition.
        t0 = two_sum(t, dt)
        x = t0[0]
        poly = -fma(x, fma(x, sixth, F(-0.2)), F(0.25))
        pp = add((th, tl), prod(x, poly))
        pp = add((F(-0.5), F(0)), mul(t0, pp))
        pp = add((F(1), F(0)), mul(t0, pp))
        baseline = add(mul(mul(t0, pp), (ih, il)), (hi, lo))

        fraction = oracle_fraction(m0.astype(D))
        # The series and final e addition are enclosed by 2^-45 * |log2 x|.
        # The same bound encloses kd_log2: reduction keeps |s| <= .172,
        # and its polynomial tail is below 2^-64 relative. A reciprocal
        # seed within 2 FP32 ulps gains two Newton steps; rounding through
        # the double evaluation costs less than 2^-48 relative. For the
        # folded case, the final nonnegative e+log2(m) uses at most a 3x
        # relative error amplification. Use 2^-44 for the two enclosures.
        for exponent in range(128):
            e = F(exponent) + fold.astype(F)
            sh, sl = fast_sum(e, vh)
            h, l = fast_sum(sh, vtail + sl)
            exact_center = h.astype(D) + l.astype(D)
            oracle = fraction + exponent
            positive = h != 0
            if not np.array_equal(positive, oracle != 0):
                raise AssertionError("zero shortcut outside x=1")
            radius = abs(vh) * F(2 ** -29) + h * F(2 ** -43)
            needed = abs(exact_center - oracle) + abs(oracle) * D(2 ** -44)
            # Extra margin covers rounding of h+l in binary64 and gives
            # slack for computing the FP32 guard; the true bound is smaller.
            if np.any(needed[positive] > radius[positive].astype(D) * 0.875):
                i = int(np.argmax(needed - radius.astype(D) * 0.875))
                raise AssertionError((hex(int(mantissa[i])), exponent, needed[i], radius[i]))
            max_ratio = max(max_ratio, float(np.max(needed[positive] / radius[positive], initial=0)))
            gap = rounding_gap(h, l)
            accepted = ~positive | (abs(l) + radius < gap)
            if np.any(radius[positive] >= h[positive] * F(2 ** -26)):
                raise AssertionError("error can reach the opposite midpoint of a power of two")

            bh, bl = add((e, F(0)), baseline)
            bpower = (bh.view(U) >> 23) & U(255)
            base_accept = ((bpower >= 40) & (bpower <= 254)
                           & ((abs(bl) + abs(bh) * F(2 ** -38)) < rounding_gap(bh, bl)))
            # For baseline's double fallback, exact_center's certified
            # interval encloses the double result before RN32. Thus accepted
            # results agree even if an arbitrary library log2 rounds a tie
            # differently. For baseline's FP32 branch, compare the words.
            bad = accepted & base_accept & (h.view(U) != bh.view(U))
            if np.any(bad):
                i = int(np.flatnonzero(bad)[0])
                raise AssertionError((hex(int(mantissa[i]) | ((127 + exponent) << 23)), h[i], bh[i]))
            oracle_lo = (oracle - abs(oracle) * D(2 ** -44)).astype(F)
            oracle_hi = (oracle + abs(oracle) * D(2 ** -44)).astype(F)
            mutant = (base_accept & (h.view(U) != bh.view(U))) | (
                ~base_accept & (h < oracle_lo) & (h < oracle_hi)) | (
                ~base_accept & (h > oracle_lo) & (h > oracle_hi))
            if first_mutant is None and np.any(mutant):
                i = int(np.flatnonzero(mutant)[0])
                first_mutant = f"0x{int(mantissa[i]) | ((127 + exponent) << 23):08x}"
            counts["mutant_wrong"] += int(np.count_nonzero(mutant))
            counts["finite"] += len(mantissa)
            counts["candidate_fallback"] += int(np.count_nonzero(~accepted))
            counts["baseline_fallback_on_accept"] += int(np.count_nonzero(accepted & ~base_accept))
    # +inf is the only reachable non-finite argument and takes kf_log2
    # without evaluating the guard. Negative, NaN and subnormal arguments
    # cannot result from maxNum(Y,0) followed by the positive fused affine map.
    assert counts["finite"] == 128 * limit
    assert counts["mutant_wrong"] > 0
    print(dict(counts, exceptional_fallback=1, first_mutant=first_mutant, max_radius_fraction=max_ratio))


if __name__ == "__main__":
    verify(Path(sys.argv[1]).read_text(), int(sys.argv[2]) if len(sys.argv) == 3 else MANTISSA_COUNT)
