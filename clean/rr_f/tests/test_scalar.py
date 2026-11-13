# SPDX-License-Identifier: MIT
"""Spec 51 scalar rules over every f16 input: attention weight (51.3 step 3), activation (step 8), reciprocal of the
row sum (step 4), quarter of the merged hidden (51.4), sigmoid (51.4), and the converter checks of the attention
constants."""
from fractions import Fraction
import decimal
import math
import unittest

import numpy as np

from clean.rr_f.reference import exact
from clean.rr_f.reference.fmt import ALL_F16, mul16, round_f16, to_f16_bits
from clean.rr_f.reference.scalar import (AttentionRangeError, WindowConsts, activation, attention_p,
                                         attention_weight, check_attention_consts, recip_table, sig_tables)
from clean.rr_f.tests import slow, synth


def bits_equal(test, got, ref, what):
    gb = to_f16_bits(np.asarray(got))
    rb = to_f16_bits(np.asarray(ref))
    bad = np.nonzero(gb != rb)[0]
    test.assertEqual(len(bad), 0, "%s: %d mismatches, first %s" % (
        what, len(bad), [(hex(int(b)), got[b], ref[b]) for b in bad[:5]]))


def const_sets():
    rng = np.random.default_rng(510)
    return [synth.window_consts(rng, f)[0] for f in ("A", "B", "zovf")]


class TestAttentionWeight(unittest.TestCase):
    def test_all_f16_scores(self):
        for wc in const_sets():
            got = attention_weight(ALL_F16, wc)
            ref = np.array([slow.attention_weight(float(s), wc) for s in ALL_F16])
            bits_equal(self, got, ref, "attention weight %r" % wc)
            fin = np.isfinite(got)
            self.assertTrue(fin.all())
            self.assertTrue(((got >= 2.0 ** -14) & (got < 2.0 ** 16)).all())

    def test_formula_points(self):
        # exact at integer y: p = 1 + n/32 -> w = 2^(n-15); midway: 2^(n-15) * 1.5
        wc = const_sets()[0]
        for n in range(1, 31):
            p = 1 + n / 32
            y = 32 * (p - 1)
            nn = math.floor(y)
            self.assertEqual(2.0 ** (nn - 15) * (1 + y - nn), 2.0 ** (n - 15))

    def test_nan_and_inf_scores(self):
        wc = const_sets()[0]
        s = np.array([np.nan, np.inf, -np.inf])
        p = attention_p(s, wc)
        lo = attention_p(np.array([wc.A_CLAMPn / wc.A_SCALE * 4]), wc)    # a score far below the lower clamp
        hi = attention_p(np.array([wc.A_CLAMP / wc.A_SCALE * 4]), wc)
        self.assertEqual(p[0], lo[0])        # NaN -> lower clamp bound
        self.assertEqual(p[1], hi[0])
        self.assertEqual(p[2], lo[0])

    def test_range_rejection(self):
        wc = const_sets()[0]
        bad = WindowConsts(**{**wc.as_dict(), "A_P0": 1.0})
        self.assertFalse(check_attention_consts(bad)["ok"])
        with self.assertRaises(AttentionRangeError):
            attention_weight(ALL_F16, bad)
        for wc in const_sets():
            chk = check_attention_consts(wc)
            self.assertTrue(chk["ok"])
        self.assertTrue(check_attention_consts(const_sets()[2])["z_may_overflow"])
        self.assertFalse(check_attention_consts(const_sets()[0])["z_may_overflow"])


class TestActivation(unittest.TestCase):
    def test_all_f16(self):
        for wc in const_sets()[:2]:
            got = activation(ALL_F16, wc)
            ref = np.array([slow.activation(float(x), wc) for x in ALL_F16])
            nan = np.isnan(ref)
            np.testing.assert_array_equal(np.isnan(got), nan)
            bits_equal(self, got[~nan], ref[~nan], "activation")


class TestReciprocal(unittest.TestCase):
    def test_all_f16(self):
        r = recip_table()
        z = ALL_F16
        with np.errstate(divide="ignore", invalid="ignore"):
            alt = round_f16(1.0 / z)              # float64 quotient (correctly rounded) then f16: innocuous here
        nan = np.isnan(z)
        bits_equal(self, r[~nan], alt[~nan], "reciprocal")
        self.assertEqual(r[to_f16_bits(np.array([np.inf]))[0]], 0.0)
        self.assertFalse(np.signbit(r[to_f16_bits(np.array([np.inf]))[0]]))
        # Z >= 64 * 2^-14 keeps r finite
        self.assertTrue(np.isfinite(r[to_f16_bits(np.array([64 * 2.0 ** -14]))[0]]))


class TestQuarter(unittest.TestCase):
    def test_all_f16(self):
        got = mul16(ALL_F16, 0.25)
        ref = np.array([slow.s_round(slow.s_mul(slow.SV.of(x), slow.SV.of(0.25))).float() for x in ALL_F16])
        nan = np.isnan(ref)
        bits_equal(self, got[~nan], ref[~nan], "h/4")


class TestSigmoid(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sig, cls.s32, cls.near = sig_tables()

    def test_specials(self):
        def at(x):
            return self.sig[to_f16_bits(np.array([x]))[0]]
        self.assertEqual(at(0.0), 0.5)
        self.assertEqual(at(-0.0), 0.5)
        self.assertEqual(at(np.inf), 1.0)
        self.assertEqual(at(-np.inf), 0.0)
        self.assertEqual(at(65504.0), 1.0)
        self.assertEqual(at(-65504.0), 0.0)
        self.assertEqual(at(-200.0), 0.0)        # q = +inf -> s = +0
        self.assertTrue(math.isnan(at(np.nan)))
        self.assertEqual(exact.log2e_f32(), float(np.float32(1.4426950408889634)))

    def test_monotone(self):
        fin = np.isfinite(ALL_F16)
        x = ALL_F16[fin]
        s = self.sig[fin]
        o = np.argsort(x, kind="stable")
        self.assertTrue(np.all(np.diff(s[o]) >= 0))

    def test_independent_float64_evaluation(self):
        diffs = []
        for b in range(65536):
            x = float(ALL_F16[b])
            if math.isnan(x):
                continue
            alt = slow.sigmoid_f64(x)
            if alt != self.sig[b]:
                diffs.append(b)
        # the float64 helpers are within an ulp of correct rounding, so differences can only occur where the
        # reference f32 value lies close to an f16 rounding midpoint (51.4 required accuracy)
        for b in diffs:
            self.assertTrue(self.near[b], (hex(b), ALL_F16[b]))
            step = abs(to_f16_bits(np.array([slow.sigmoid_f64(float(ALL_F16[b]))]))[0].astype(int)
                       - to_f16_bits(np.array([self.sig[b]]))[0].astype(int))
            self.assertLessEqual(step, 1)

    def test_vs_correctly_rounded_exact_sigmoid(self):
        """51.4: the correctly rounded f16 of the exact sigmoid differs from the reference by one f16 step for
        7 of the 63,488 finite f16 inputs."""
        n_diff = 0
        n_fin = 0
        with decimal.localcontext() as ctx:
            ctx.prec = 60
            ctx.Emax = 999999
            ctx.Emin = -999999
            for b in range(65536):
                x = float(ALL_F16[b])
                if not math.isfinite(x):
                    continue
                n_fin += 1
                if x > 30:
                    ex = 1.0
                elif x < -30:
                    ex = exact.round_frac(Fraction(1 / (1 + decimal.Decimal(-x).exp())), exact.F16)
                else:
                    ex = exact.round_frac(Fraction(1 / (1 + decimal.Decimal(-x).exp())), exact.F16)
                if ex != self.sig[b]:
                    n_diff += 1
                    d = abs(int(to_f16_bits(np.array([ex]))[0]) - int(to_f16_bits(np.array([self.sig[b]]))[0]))
                    self.assertEqual(d, 1)
        self.assertEqual(n_fin, 63488)
        self.assertEqual(n_diff, 7)


if __name__ == "__main__":
    unittest.main()
