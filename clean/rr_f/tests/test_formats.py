# SPDX-License-Identifier: MIT
"""Spec 51.1: e4m3 decode (all 256 codes), Q8 (all 65536 f16 inputs and random / edge f32 inputs), f16 rounding,
fma16, the matrix step.

Run: python -m unittest discover -s clean/rr_f/tests -t ."""
from fractions import Fraction
import math
import struct
import unittest

import numpy as np

from clean.rr_f.reference import exact
from clean.rr_f.reference.fmt import (ALL_F16, D8_TABLE, add16, d8, fma16, from_f16_bits, mm, mul16, q8,
                                      round_f16, to_f16_bits)
from clean.rr_f.tests import slow


def brute_q8(h):
    """Q8 by exhaustive search over the codes of h's sign (vectorised over a 1-D array h of f16/f32 values)."""
    h = np.asarray(h, dtype=np.float64)
    neg = np.signbit(h)
    pos_codes = np.arange(0x7F)                         # 0x00..0x7E (0x7F is NaN)
    vals = D8_TABLE[pos_codes]                          # 0 .. 448 ascending
    a = np.abs(h)
    out = np.empty(h.shape, dtype=np.int64)
    for s in range(0, len(h), 4096):
        aa = a[s:s + 4096]
        fin = np.isfinite(aa)
        aa_f = np.where(fin, aa, 0.0)
        d = np.abs(aa_f[:, None] - vals[None, :])       # exact: both on fine binary grids, magnitudes < 2^17
        best = d.min(axis=1, keepdims=True)
        cand = d == best
        # ties: even code
        even = cand & (pos_codes[None, :] % 2 == 0)
        pick = np.where(even.any(axis=1), np.argmax(even, axis=1), np.argmax(cand, axis=1))
        pick = np.where(np.isinf(aa), 0x7E, pick)
        pick = np.where(np.isnan(aa), 0x7F, pick)
        out[s:s + 4096] = pick
    return (out | (neg.astype(np.int64) << 7)).astype(np.uint8)


def f32_from_bits(b):
    return np.asarray(b, dtype=np.uint32).view(np.float32).astype(np.float64)


class TestE4M3(unittest.TestCase):
    def test_d8_all_codes(self):
        for code in range(256):
            s, e, m = code >> 7, (code >> 3) & 15, code & 7
            got = D8_TABLE[code]
            if e == 15 and m == 7:
                self.assertTrue(math.isnan(got), code)
                continue
            if e > 0:
                v = Fraction(2) ** (e - 7) * (1 + Fraction(m, 8))
            else:
                v = Fraction(1, 64) * Fraction(m, 8)
            v = -v if s else v
            self.assertEqual(Fraction(got), v, code)
            self.assertEqual(math.copysign(1, got) < 0, bool(s), code)    # -0 for 0x80
            # exact in f16
            self.assertEqual(float(np.float16(got)), got)
        self.assertEqual(np.nanmax(D8_TABLE), 448.0)
        self.assertEqual(np.nanmin(D8_TABLE), -448.0)

    def test_q8_all_f16(self):
        got = q8(ALL_F16)
        ref = brute_q8(ALL_F16)
        bad = np.nonzero(got != ref)[0]
        self.assertEqual(len(bad), 0, [(hex(b), ALL_F16[b], got[b], ref[b]) for b in bad[:10]])

    def test_q8_rules(self):
        cases = [
            (448.0, 0x7E), (449.0, 0x7E), (464.0, 0x7E), (65504.0, 0x7E), (np.inf, 0x7E), (-np.inf, 0xFE),
            (-1000.0, 0xFE), (0.0, 0x00), (-0.0, 0x80), (2.0 ** -10, 0x00), (-(2.0 ** -10), 0x80),
            (2.0 ** -10 + 2.0 ** -20, 0x01), (-(2.0 ** -24), 0x80), (1.0, 0x38), (1.0625, 0x38), (1.1875, 0x3A),
            (240.0, 0x77), (232.0, 0x76),
        ]
        for h, code in cases:
            self.assertEqual(int(q8(np.array([h]))[0]), code, h)
        self.assertEqual(int(q8(np.array([np.nan]))[0]), 0x7F)
        self.assertEqual(int(q8(np.array([-np.nan]))[0]), 0xFF)

    def test_q8_roundtrip(self):
        codes = np.array([c for c in range(256) if c not in (0x7F, 0xFF)], dtype=np.uint8)
        np.testing.assert_array_equal(q8(d8(codes)), codes)

    def test_q8_f32(self):
        rng = np.random.default_rng(51)
        # random f32 bit patterns of magnitude below 2^10 and above 2^-12, plus every e4m3 midpoint +- 1 f32 ulp
        bits = rng.integers(0, 2 ** 32, size=200000, dtype=np.uint64).astype(np.uint32)
        exp = (bits >> 23) & 0xFF
        keep = (exp > 127 - 13) & (exp < 127 + 10)
        x = f32_from_bits(bits[keep])
        vals = D8_TABLE[:0x7F]
        mids = (vals[1:] + vals[:-1]) / 2
        mids = np.concatenate([mids, [464.0, 2.0 ** -10]])
        m32 = mids.astype(np.float32)
        up = np.nextafter(m32, np.float32(np.inf)).astype(np.float64)
        dn = np.nextafter(m32, np.float32(0)).astype(np.float64)
        x = np.concatenate([x, mids, up, dn, -mids, -up, -dn])
        np.testing.assert_array_equal(q8(x), brute_q8(x))
        # and against the rule as written, element by element (exact rational rounding)
        for v in x[:3000]:
            self.assertEqual(float(d8(q8(np.array([v])))[0]), slow.q8_value(float(v)))


class TestF16(unittest.TestCase):
    def test_round_vs_numpy_cast(self):
        rng = np.random.default_rng(1)
        x = rng.normal(0, 1, 100000) * np.exp2(rng.integers(-30, 17, 100000))
        x = x[np.abs(x) < 65519]
        np.testing.assert_array_equal(to_f16_bits(round_f16(x)), x.astype(np.float16).view(np.uint16))

    def test_midpoints(self):
        fin = ALL_F16[np.isfinite(ALL_F16) & (ALL_F16 >= 0)]
        fin = np.unique(fin)
        mids = (fin[1:] + fin[:-1]) / 2                 # exact in float64
        lo, hi = fin[:-1], fin[1:]
        lo_bits = to_f16_bits(lo)
        tie = np.where(lo_bits % 2 == 0, lo, hi)
        np.testing.assert_array_equal(round_f16(mids), tie)
        tiny = np.ldexp(1.0, -60)
        np.testing.assert_array_equal(round_f16(mids, tiny), hi)
        np.testing.assert_array_equal(round_f16(mids, -tiny), lo)
        np.testing.assert_array_equal(round_f16(-mids, tiny), -lo)
        np.testing.assert_array_equal(round_f16(-mids, -tiny), -hi)
        # overflow threshold: from 65520 up -> inf
        self.assertEqual(round_f16(np.array(65520.0)), np.inf)
        self.assertEqual(round_f16(np.array(65520.0), -tiny), 65504.0)
        self.assertEqual(round_f16(np.array(-65520.0)), -np.inf)
        self.assertEqual(round_f16(np.array(65519.996)), 65504.0)
        # subnormals and signed zeros
        self.assertEqual(round_f16(np.array(2.0 ** -25)), 0.0)
        self.assertTrue(np.signbit(round_f16(np.array(-(2.0 ** -25)))))
        self.assertEqual(round_f16(np.array(3 * 2.0 ** -26)), 2.0 ** -24)

    def test_fma16_random(self):
        rng = np.random.default_rng(2)
        bits = rng.integers(0, 65536, size=(3, 40000)).astype(np.uint16)
        a, b, c = (from_f16_bits(x) for x in bits)
        got = fma16(a, b, c)
        for i in range(40000):
            ref = slow.fma(slow.SV.of(a[i]), slow.SV.of(b[i]), slow.SV.of(c[i])).float()
            g = got[i]
            if math.isnan(ref):
                self.assertTrue(math.isnan(g), (a[i], b[i], c[i]))
            else:
                self.assertEqual(g, ref, (a[i], b[i], c[i]))
                self.assertEqual(math.copysign(1, g), math.copysign(1, ref), (a[i], b[i], c[i]))

    def test_fma16_constructed_midpoints(self):
        # c large, a*b tiny: the exact sum is close to (but not on) a midpoint of c's binade
        rng = np.random.default_rng(3)
        for _ in range(20000):
            c = float(np.float16(rng.uniform(1, 2000)))
            ulp = 2.0 ** (math.floor(math.log2(abs(c))) - 10)
            a = float(np.float16(ulp / 2 * rng.choice([1, -1]) * (1 + rng.integers(-3, 4) / 1024)))
            b = float(np.float16(1 + rng.integers(-3, 4) / 1024))
            ref = slow.fma(slow.SV.of(a), slow.SV.of(b), slow.SV.of(c)).float()
            self.assertEqual(float(fma16(a, b, c)), ref, (a, b, c))

    def test_add_mul(self):
        rng = np.random.default_rng(4)
        bits = rng.integers(0, 65536, size=(2, 20000)).astype(np.uint16)
        a, b = (from_f16_bits(x) for x in bits)
        ga, gm = add16(a, b), mul16(a, b)
        for i in range(20000):
            ra = slow.s_round(slow.s_add(slow.SV.of(a[i]), slow.SV.of(b[i]))).float()
            rm = slow.s_round(slow.s_mul(slow.SV.of(a[i]), slow.SV.of(b[i]))).float()
            for g, r in ((ga[i], ra), (gm[i], rm)):
                if math.isnan(r):
                    self.assertTrue(math.isnan(g))
                else:
                    self.assertEqual(to_f16_bits(np.array([g]))[0], to_f16_bits(np.array([r]))[0], (a[i], b[i]))


class TestMatrixStep(unittest.TestCase):
    def test_mm_vs_slow(self):
        rng = np.random.default_rng(5)
        for K, N, sig, bsig in ((32, 7, 1.0, 1.0), (64, 5, 4.0, 100.0), (160, 6, 0.3, 0.01), (640, 3, 8.0, 1000.0)):
            x = q8(rng.normal(0, sig, (9, K)))
            W = q8(rng.normal(0, sig, (K, N)))
            c0 = round_f16(rng.normal(0, bsig, N))
            got = mm(d8(x), d8(W), c0)
            xl = d8(x).tolist()
            Wl = d8(W).tolist()
            for r in range(9):
                for n in range(N):
                    ref = slow.mm_step_chain(xl[r], [Wl[k][n] for k in range(K)], float(c0[n]))
                    self.assertEqual(got[r, n], ref, (K, r, n))

    def test_mm_one_rounding_per_step(self):
        # 32 products summing to 1 + 2^-11 + 2^-14: one rounding gives 1 + 2^-10; two separate halves would not
        x = np.zeros((1, 64))
        W = np.zeros((64, 1))
        x[0, 0], W[0, 0] = 1.0, 1.0
        x[0, 1], W[1, 0] = 2.0 ** -6, 2.0 ** -5              # 2^-11
        x[0, 2], W[2, 0] = 2.0 ** -7, 2.0 ** -7              # 2^-14
        self.assertEqual(mm(x, W)[0, 0], 1.0 + 2.0 ** -10)
        # overflow to inf, and the next step starts from inf
        x = np.full((1, 64), 448.0)
        W = np.full((64, 1), 448.0)
        self.assertEqual(mm(x, W)[0, 0], np.inf)
        W[32:] = -448.0
        self.assertEqual(mm(x, W)[0, 0], np.inf)        # inf + finite = inf


if __name__ == "__main__":
    unittest.main()
