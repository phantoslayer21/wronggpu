# SPDX-License-Identifier: MIT
"""Output filter with the weight set "original" (spec 06 draft 4, steps 2 to 5): clean/reference/post.py
h_fma, filter_pass_original, encoded_filter_original, output_filter_original.

Checked against a second, literal implementation of the spec's sentences in exact rational arithmetic (fractions),
and by properties: constant image, clamp, mirrored phases, edge clamping, weights rounded once.

All weights here are made up (spec 06's analytic set, perturbations of it, hand-written numbers).

Run: python -m unittest discover -s clean/tests -t .
"""
import unittest
from fractions import Fraction

import numpy as np

from clean.reference import post as P
from clean.runner import synthetic_model as SM

TWO = Fraction(2)


def h_exact(v, negative_zero=False):
    """Round the rational v to FP16, nearest with ties to even (spec 06: h). Returns a Python float (an FP16 value,
    +-inf on overflow); a zero result takes the sign of v, or `negative_zero` when v is exactly 0."""
    if v == 0:
        return -0.0 if negative_zero else 0.0
    sign = -1.0 if v < 0 else 1.0
    m = abs(v)
    e = 0
    while TWO ** e > m:
        e -= 1
    while TWO ** (e + 1) <= m:
        e += 1
    q = TWO ** (max(e, -14) - 10)             # the FP16 step at m (subnormals below 2^-14: 2^-24)
    n = m / q
    lo = n.numerator // n.denominator
    r = n - lo
    if r > Fraction(1, 2) or (r == Fraction(1, 2) and lo % 2 == 1):
        lo += 1
    r16 = lo * q
    if r16 >= 65520:                            # past the largest finite FP16 value's rounding range
        return sign * float("inf")
    return sign * float(r16)


def fma_exact(w, t, a):
    """h(fma(w, t, a)) for FP16 values given as floats: the exact w t + a, rounded once."""
    v = Fraction(float(w)) * Fraction(float(t)) + Fraction(float(a))
    # exact zero: -0 only when product and addend are both -0 (IEEE 754, round to nearest)
    neg_p = (np.signbit(w) != np.signbit(t))
    return h_exact(v, negative_zero=bool(neg_p and np.signbit(a) and float(a) == 0.0 and float(w) * float(t) == 0.0))


def pass_literal(values, n_out, taps):
    """Spec 06 draft 4 step 2 for a 1-D list of FP16 values, one output at a time, in exact arithmetic."""
    w16 = [float(np.float16(w)) for w in taps]
    n_in = len(values)
    out = []
    for x in range(n_out):
        c = int(np.floor((x + 0.5) * 1.5))
        t = [float(values[min(max(c + k, 0), n_in - 1)]) for k in range(-2, 3)]
        a = 0.0
        for k in range(-2, 3):
            wk = w16[k + 2] if x % 2 == 0 else w16[-k + 2]   # odd index: tap k uses the even weight of -k
            a = fma_exact(wk, t[k + 2], a)
        out.append(min(max(a, min(t)), max(t)))
    return out


def bits16(a):
    return np.asarray(a, np.float64).astype(np.float16).view(np.uint16)


def random_f16(rng, n, lo=-14, hi=3):
    """FP16 values with random signs, exponents 2^lo ... 2^hi and random significands (some subnormal, some zero)."""
    v = rng.integers(1024, 2048, n) / 1024.0 * 2.0 ** rng.integers(lo, hi + 1, n) * rng.choice([-1.0, 1.0], n)
    v[rng.random(n) < 0.03] = 0.0
    sub = rng.random(n) < 0.05
    v[sub] = rng.integers(0, 1024, int(sub.sum())) * 2.0 ** -24
    return P.f16(v).astype(np.float64)


class HFma(unittest.TestCase):
    def test_against_exact_arithmetic(self):
        rng = np.random.default_rng(11)
        n = 4000
        w, t, a = random_f16(rng, n, -6, 1), random_f16(rng, n), random_f16(rng, n)
        got = P.h_fma(w, t, a)
        want = np.array([fma_exact(*v) for v in zip(w, t, a)])
        np.testing.assert_array_equal(bits16(got), bits16(want))

    def test_wide_exponent_gaps(self):
        # addend and product up to 2^30 apart: the float64 sum is inexact, the rounding must still be single
        rng = np.random.default_rng(12)
        n = 3000
        w, t, a = random_f16(rng, n, -14, 0), random_f16(rng, n, -14, -8), random_f16(rng, n, 8, 15)
        want = np.array([fma_exact(*v) for v in zip(w, t, a)])
        np.testing.assert_array_equal(bits16(P.h_fma(w, t, a)), bits16(want))
        np.testing.assert_array_equal(bits16(P.h_fma(w, a, t)), bits16([fma_exact(*v) for v in zip(w, a, t)]))

    def test_ties_go_to_even(self):
        # a = 1 (FP16 step 2^-10), product k1 k2 / 4096: exactly half a step whenever k1 k2 = 2 mod 4
        k1, k2 = np.meshgrid(np.arange(1, 64), np.arange(1, 64))
        w, t = (k1 / 64.0).ravel(), (k2 / 64.0).ravel()
        for a0 in (1.0, 1.0 + 2.0 ** -10, -1.0, 1.5):
            a = np.full(w.shape, a0)
            want = np.array([fma_exact(*v) for v in zip(w, t, a)])
            np.testing.assert_array_equal(bits16(P.h_fma(w, t, a)), bits16(want))
        self.assertEqual(float(P.h_fma(0.5, 2.0 ** -10, 1.0)), 1.0)                          # tie -> even (down)
        self.assertEqual(float(P.h_fma(0.5, 2.0 ** -10, 1.0 + 2.0 ** -10)), 1.0 + 2.0 ** -9)   # tie -> even (up)

    def test_one_rounding_not_two(self):
        # exact value just below the middle of two FP16 values: FP32 would round it onto the middle, and the second
        # rounding (ties to even) would then go up. h(fma) rounds the exact value: down.
        w, t, a = 8.0 * (1 + 2.0 ** -10), 1 - 2.0 ** -10, 16400.0
        for v in (w, t, a):
            self.assertEqual(float(np.float16(v)), v)
        self.assertEqual(float(P.h_fma(w, t, a)), 16400.0)
        self.assertEqual(fma_exact(w, t, a), 16400.0)
        twice = np.float16(np.float32(np.float32(w) * np.float32(t)) + np.float32(a))   # the product is exact in FP32
        self.assertEqual(float(twice), 16416.0)

    def test_zero_signs_and_shapes(self):
        z = P.h_fma(-0.0, 1.0, -0.0)
        self.assertTrue(z == 0.0 and np.signbit(z))
        for w, t, a in ((0.0, 1.0, -0.0), (-0.0, 1.0, 0.0), (1.0, 1.0, -1.0), (0.5, 0.0, 0.0)):
            z = P.h_fma(w, t, a)
            self.assertTrue(z == 0.0 and not np.signbit(z), (w, t, a))
        tiny = 2.0 ** -24
        self.assertTrue(np.signbit(P.h_fma(-tiny, tiny, 0.0)))        # underflow from below: -0
        self.assertEqual(P.h_fma(np.ones((3, 2)), np.ones(2), 0.0).shape, (3, 2))
        self.assertEqual(float(P.h_fma(2.0, 3.0, 1.0)), 7.0)


class OriginalFilter(unittest.TestCase):
    TAPS = [SM.output_taps("analytic"), SM.output_taps("random", 1), SM.output_taps("random", 2),
            np.array([0.05, 0.1, 0.5, 0.25, 0.1], np.float32), np.array([-0.2, 0.3, 0.9, 0.4, -0.3], np.float32)]

    def test_center(self):
        x = np.arange(200)
        np.testing.assert_array_equal(P.filter_center(x), np.floor((x + 0.5) * 1.5).astype(np.int64))
        np.testing.assert_array_equal(P.filter_center(x), P.filter_weights(x)[0])   # the analytic filter's centres

    def test_pass_is_the_spec_chain(self):
        # FP16 inputs, every output index of several sizes (odd and even, edges included), bit for bit
        rng = np.random.default_rng(21)
        for taps in self.TAPS:
            for n_out in (1, 2, 3, 6, 7, 16, 33):
                n_in = P.recon_size_for(n_out)
                v = P.f16(rng.uniform(0.0, 2.45, n_in) * (rng.random(n_in) > 0.1)).astype(np.float64)
                got = P.filter_pass_original(v, n_out, taps)
                np.testing.assert_array_equal(bits16(got), bits16(pass_literal(v, n_out, taps)), err_msg=str(n_out))

    def test_two_passes_order_and_vertical_clamp(self):
        # horizontal first, then vertical on its results; the vertical clamp uses the horizontally filtered values
        rng = np.random.default_rng(22)
        taps = self.TAPS[4]
        ow, oh = 7, 5
        dw, dh = P.recon_size_for(ow), P.recon_size_for(oh)
        disp = P.f16(rng.uniform(0.0, 2.45, (dh, dw, 3)))
        got = P.encoded_filter_original(disp, (ow, oh), taps)
        self.assertEqual(got.dtype, np.float16)
        self.assertEqual(got.shape, (oh, ow, 3))
        for c in range(3):
            hor = np.array([pass_literal(disp[r, :, c], ow, taps) for r in range(dh)])           # [dh, ow]
            want = np.array([pass_literal(hor[:, x], oh, taps) for x in range(ow)]).T            # [oh, ow]
            np.testing.assert_array_equal(got[..., c].view(np.uint16), bits16(want))
        # not the other order
        other = P.filter_pass_original(P.filter_pass_original(disp, oh, taps).transpose(1, 0, 2), ow,
                                       taps).transpose(1, 0, 2)
        self.assertFalse(np.array_equal(bits16(other), got.view(np.uint16)))

    def test_constant_image(self):
        # the clamp makes a constant image exact for any weights, also when they do not sum to 1
        for taps in self.TAPS + [np.array([0.3, 0.3, 0.3, 0.3, 0.3], np.float32)]:
            for value in (0.0, 0.7001953125, 2.44921875):
                disp = np.full((P.recon_size_for(9), P.recon_size_for(7), 3), value)
                out = P.encoded_filter_original(disp, (7, 9), taps)
                np.testing.assert_array_equal(out.astype(np.float64), value)

    def test_clamp(self):
        rng = np.random.default_rng(23)
        taps = self.TAPS[4]                                  # negative lobes: overshoots at edges
        v = P.f16(rng.uniform(0.2, 0.9, P.recon_size_for(40))).astype(np.float64)
        out = P.filter_pass_original(v, 40, taps)
        idx = np.clip(P.filter_center(np.arange(40))[:, None] + np.arange(-2, 3)[None, :], 0, len(v) - 1)
        self.assertTrue((out >= v[idx].min(axis=1)).all() and (out <= v[idx].max(axis=1)).all())
        # a step edge: without the clamp the chain overshoots; the result is the largest / smallest tap
        step = np.array([0.0] * 9 + [1.0] * 9)
        out = P.filter_pass_original(step, 12, taps)
        self.assertEqual(float(out.min()), 0.0)
        self.assertEqual(float(out.max()), 1.0)
        w16 = P.filter_taps_f16(taps)
        x = 5                                                # centre 8: taps 6 ... 10 = 0 0 0 1 1, odd index (mirrored)
        raw = P.h_fma(w16[0], 1.0, P.h_fma(w16[1], 1.0, 0.0))  # the two taps on the bright side: k = 1, 2
        self.assertLess(float(raw), 1.0)
        self.assertEqual(float(out[x]), float(raw))
        x = 6                                                # centre 9: taps 7 ... 11 = 0 0 1 1 1: sum above 1
        self.assertEqual(float(out[x]), 1.0)
        img = rng.uniform(0.2, 0.9, (P.recon_size_for(10), P.recon_size_for(12), 3))
        q = P.encoded_filter_original(P.f16(img), (12, 10), taps).astype(np.float64)
        self.assertGreaterEqual(q.min(), P.f16(img).min())
        self.assertLessEqual(q.max(), P.f16(img).max())

    def test_mirrored_phases(self):
        # impulse response: one input 1, the rest 0. Even output index: the weight of tap k = j - c; odd index: the
        # even weight of -k. (Weights within 0 ... 1 pass the clamp unchanged.)
        taps = self.TAPS[3]
        w16 = P.filter_taps_f16(taps)
        n_out = 12
        n_in = P.recon_size_for(n_out)
        for j in range(3, n_in - 3):                         # away from the edges (no duplicated taps)
            v = np.zeros(n_in)
            v[j] = 1.0
            out = P.filter_pass_original(v, n_out, taps)
            for x in range(n_out):
                k = j - int(P.filter_center(x))
                want = 0.0 if abs(k) > 2 else (w16[k + 2] if x % 2 == 0 else w16[2 - k])
                self.assertEqual(float(out[x]), float(want), (j, x))
        # the same in two dimensions: horizontal weight times vertical weight, rounded once
        ow, oh = 6, 6
        disp = np.zeros((P.recon_size_for(oh), P.recon_size_for(ow), 3))
        disp[4, 5, :] = 1.0
        out = P.encoded_filter_original(disp, (ow, oh), taps).astype(np.float64)
        for y in range(oh):
            for x in range(ow):
                kx, ky = 5 - int(P.filter_center(x)), 4 - int(P.filter_center(y))
                wx = 0.0 if abs(kx) > 2 else (w16[kx + 2] if x % 2 == 0 else w16[2 - kx])
                wy = 0.0 if abs(ky) > 2 else (w16[ky + 2] if y % 2 == 0 else w16[2 - ky])
                self.assertEqual(out[y, x, 0], float(np.float16(wy * wx)), (y, x))

    def test_edges_clamp_the_index(self):
        # x = 0: centre 0, taps -2, -1 read input 0; last x of an even size: centre n_in - 1, taps +1, +2 read it too
        taps = self.TAPS[3]
        w16 = P.filter_taps_f16(taps)
        n_out = 8
        n_in = P.recon_size_for(n_out)
        v = np.zeros(n_in)
        v[0] = 1.0
        out = P.filter_pass_original(v, n_out, taps)
        chain = 0.0
        for k in range(3):                                   # taps -2, -1, 0 all read input 0 (even index)
            chain = float(P.h_fma(w16[k], 1.0, chain))
        self.assertEqual(float(out[0]), chain)
        v = np.zeros(n_in)
        v[-1] = 1.0
        out = P.filter_pass_original(v, n_out, taps)
        chain = 0.0
        for k in range(2, 5):                                # x = 7 (odd): centre 11 = n_in - 1; taps 0, 1, 2
            chain = float(P.h_fma(w16[4 - k], 1.0, chain))
        self.assertEqual(int(P.filter_center(n_out - 1)), n_in - 1)
        self.assertEqual(float(out[-1]), chain)

    def test_weights_are_rounded_once(self):
        rng = np.random.default_rng(24)
        taps = SM.output_taps("random", 9)                   # float32, not FP16 values
        self.assertFalse(np.array_equal(P.f16(taps), taps))
        v = P.f16(rng.uniform(0.0, 2.45, P.recon_size_for(30)))
        a = P.filter_pass_original(v, 30, taps)
        b = P.filter_pass_original(v, 30, P.f16(taps))
        np.testing.assert_array_equal(bits16(a), bits16(b))
        np.testing.assert_array_equal(P.filter_taps_f16(taps), P.f16(taps).astype(np.float64))
        for bad in ([1.0, 2.0, 3.0], [0, 0, np.inf, 0, 0]):
            with self.assertRaises(ValueError):
                P.filter_taps_f16(bad)

    def test_output_filter(self):
        rng = np.random.default_rng(25)
        taps = self.TAPS[1]
        ow, oh = 9, 6
        disp = P.f16(rng.uniform(0.0, 2.45, (P.recon_size_for(oh), P.recon_size_for(ow), 3)))
        q = P.encoded_filter_original(disp, (ow, oh), taps)
        out = P.output_filter_original(disp, 1.75, (ow, oh), taps)
        self.assertEqual(out.dtype, np.float32)
        np.testing.assert_array_equal(out, (P.dec(q.astype(np.float64)) / 1.75).astype(np.float32))
        np.testing.assert_array_equal(P.output_filter_original(disp, 0.0, (ow, oh), taps),
                                      P.output_filter_original(disp, 1.0, (ow, oh), taps))   # exposure 0 counts as 1
        with self.assertRaises(ValueError):
            P.output_filter_original(disp, 1.0, (ow + 1, oh), taps)

    def test_close_to_the_analytic_filter_with_its_weights(self):
        # same weights (rounded to FP16) and FP16 steps instead of float64: a few FP16 steps apart at most
        rng = np.random.default_rng(26)
        ow, oh = 20, 14
        disp = P.f16(rng.uniform(0.3, 2.0, (P.recon_size_for(oh), P.recon_size_for(ow), 3)))
        q = P.encoded_filter_original(disp, (ow, oh), SM.output_taps("analytic")).astype(np.float64)
        h = P.filter_pass(disp.astype(np.float64).transpose(1, 0, 2), ow).transpose(1, 0, 2)
        ref = P.filter_pass(h, oh)
        self.assertLess(float(np.abs(q - ref).max()), 6 * 2.0 ** -10)
        self.assertGreater(float(np.abs(q - ref).max()), 0.0)


if __name__ == "__main__":
    unittest.main()
