# SPDX-License-Identifier: MIT
"""Enumerate every finite FP32 log2 argument in [1, +inf) for tiled HDR phi.

Pass the generated dx12/gen/k_fp_tables.hlsli as the only argument. Baseline
FP32 operations mirror kf2_log2. The baseline binary64 branch is enclosed
using the independent series below. Exceptional inputs use the same kf_log2.
"""
from fractions import Fraction
from functools import lru_cache
import math
import re
import sys
from pathlib import Path

import numpy as np


F = np.float32
D = np.float64
U = np.uint32
MANTISSA_COUNT = 1 << 23
ORACLE_REL_ERROR = 2 ** -45
UNIT_ROUNDOFF = 2 ** -53


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
        target = np.copysign(F(np.inf), diff[change].astype(F))
        r[change] = np.nextafter(r[change], target)
    return r


def exact_fma(a, b, c):
    return Fraction(a) * Fraction(b) + Fraction(c)


def fma64(a, b, c):
    if hasattr(math, "fma"):
        return math.fma(a, b, c)
    # Integer true division rounds the exact rational directly to binary64.
    an, ad = a.as_integer_ratio()
    bn, bd = b.as_integer_ratio()
    cn, cd = c.as_integer_ratio()
    return (an * bn * cd + cn * ad * bd) / (ad * bd * cd)


def verify_fma():
    rng = np.random.default_rng(287)
    sample = np.ldexp(rng.uniform(-1, 1, (4096, 3)).astype(F),
                      rng.integers(-30, 31, (4096, 3)))
    # A product at a binary32 midpoint with a tail that binary64 rounds away.
    midpoint = np.array([[F(1.5), np.nextafter(F(1), F(2)), F(t)]
                         for t in (-2 ** -70, 0, 2 ** -70)], dtype=F)
    sample = np.concatenate((sample, midpoint))
    got = fma(*sample.T)
    for i, (a, b, c) in enumerate(sample):
        args = tuple(map(float, (a, b, c)))
        rounded64 = math.fma(*args) if hasattr(math, "fma") else float(exact_fma(*args))
        near = F(rounded64)
        step = np.nextafter(near, F(np.inf if rounded64 > near else -np.inf))
        if not hasattr(math, "fma") or abs(rounded64 - float(near)) == abs(float(step) - float(near)) * 0.5:
            exact = exact_fma(*args)
            adjacent = np.nextafter(near, F(np.inf if exact > float(near) else -np.inf))
            near = min((near, adjacent), key=lambda x: (abs(exact - Fraction(float(x))), int(x.view(U)) & 1))
        if got[i].view(U) != near.view(U):
            raise AssertionError(("fma", args, got[i], near))
    return len(sample)


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


def double_from_words(lo, hi):
    return float(np.uint64(int(lo, 16) | (int(hi, 16) << 32)).view(D))


def kd_constants(source):
    def dconst(name):
        words = re.search(r"#define " + name + r" asdouble\((0x[0-9A-F]+)u, (0x[0-9A-F]+)u\)", source)
        return double_from_words(*words.groups())

    odd = re.search(r"KD_INV_ODD\[11\] = \{([^}]+)\}", source)[1]
    inv_odd = tuple(double_from_words(*words)
                    for words in re.findall(r"uint2\((0x[0-9A-F]+)u, (0x[0-9A-F]+)u\)", odd))
    return dconst("KD_SQRT2"), dconst("KD_INV_LN2"), dconst("KD_INV23"), inv_odd


def kd_log_core(m, sqrt2, inv23, inv_odd):
    fold = m > sqrt2
    if fold:
        m *= 0.5
    d = m + 1.0
    # Use RN32 reciprocal as the CPU seed. The enclosure covers the GPU estimate.
    q = float(F(1.0 / float(F(d))))
    for _ in range(2):
        q = fma64(q, fma64(-d, q, 1.0), q)
    s = (m - 1.0) * q
    z = s * s
    p = inv23
    for coefficient in reversed(inv_odd):
        p = fma64(p, z, coefficient)
    return 2.0 * s * p, int(fold)


def kd_log2_relative_error():
    u = UNIT_ROUNDOFF
    # The 2 FP32 ulp reciprocal seed error and RN32(d) fit within 2^-21.
    # A fused Newton step squares the error and incurs at most 3u.
    q_error = 2 ** -21
    for _ in range(2):
        q_error = q_error * q_error + 3 * u
    s_error = q_error + u * (1 + q_error)
    z_error = 2 * s_error + s_error ** 2 + u * (1 + s_error) ** 2

    # The Horner sum has positive terms of degree <= 11. Bound the
    # perturbation from z and the coefficient and FMA rounding separately.
    z_effect = 11 * z_error * (1 + z_error) ** 10
    horner_error = 12 * u * (1 + u) ** 11
    max_s = 0.172
    series_tail = max_s ** 24 / (25 * (1 - max_s ** 2))
    errors = (s_error, z_effect, horner_error, series_tail, u, u)
    fraction_error = sum(errors) * (1 + max(errors)) ** (len(errors) - 1)
    # The exact fractional log has magnitude <= the full log on [1, +inf).
    # Include the last fused exponent addition and slack for the bound arithmetic.
    return fraction_error + u * (1 + fraction_error) + 2 * u


def verify(source, limit):
    invs, highs, lows = (table(f"KF_LOG_{n}", source) for n in ("INV", "H", "L"))
    ih, il, th, tl, sixth = (const(n, source) for n in
                             ("KF2_INV_LN2_H", "KF2_INV_LN2_L", "KF_THIRD_H", "KF_THIRD_L", "KF_SIXTH"))
    sqrt2, inv_ln2, inv23, inv_odd = kd_constants(source)
    kd_rel_error = kd_log2_relative_error()
    kd_oracle_rel_error = (ORACLE_REL_ERROR + kd_rel_error) / (1 - ORACLE_REL_ERROR)
    fma_samples = verify_fma()
    counts = dict(finite=0, candidate_fallback=0, baseline_fallback_on_accept=0, kd_checks=0, mutant_wrong=0)
    max_kd_ratio = 0.0
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
        # Cache the emulated reduced logarithm for probe lanes sharing a mantissa.
        @lru_cache(maxsize=None)
        def reduced(i):
            return kd_log_core(float(m0[i]), sqrt2, inv23, inv_odd)

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
            oracle_error = abs(oracle) * kd_oracle_rel_error
            needed = abs(exact_center - oracle) + oracle_error
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
            probes = np.flatnonzero(~accepted | ~base_accept)
            if probes.size:
                kd = np.empty(probes.size, dtype=D)
                for n, i in enumerate(probes):
                    ln_m, kd_fold = reduced(int(i))
                    kd[n] = fma64(ln_m, inv_ln2, float(exponent + kd_fold))
                error = abs(kd - oracle[probes])
                bound = abs(oracle[probes]) * kd_oracle_rel_error
                if np.any(error > bound):
                    i = int(np.flatnonzero(error > bound)[0])
                    raise AssertionError(("kd_log2 enclosure", int(mantissa[probes[i]]), exponent, error[i], bound[i]))
                relative = np.divide(error, bound, out=np.zeros_like(error), where=bound != 0)
                max_kd_ratio = max(max_kd_ratio, float(np.max(relative)))
                use_kd = accepted[probes] & ~base_accept[probes]
                if np.any(kd[use_kd].astype(F).view(U) != h[probes[use_kd]].view(U)):
                    raise AssertionError(("accepted kd_log2 mismatch", exponent, int(mantissa[0])))
                counts["kd_checks"] += probes.size

            oracle_lo = (oracle - oracle_error).astype(F)
            oracle_hi = (oracle + oracle_error).astype(F)
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
        reduced.cache_clear()
    # +inf is the only reachable non-finite argument and takes kf_log2
    # without evaluating the guard. Negative, NaN and subnormal arguments
    # cannot result from maxNum(Y,0) followed by the positive fused affine map.
    assert counts["finite"] == 128 * limit
    assert counts["mutant_wrong"] > 0
    assert counts["kd_checks"] == counts["candidate_fallback"] + counts["baseline_fallback_on_accept"]
    print(dict(counts, exceptional_fallback=1, first_mutant=first_mutant, max_radius_fraction=max_ratio,
               kd_oracle_rel_error=kd_oracle_rel_error, max_kd_bound_fraction=max_kd_ratio, fma_samples=fma_samples))


if __name__ == "__main__":
    verify(Path(sys.argv[1]).read_text(), int(sys.argv[2]) if len(sys.argv) == 3 else MANTISSA_COUNT)
