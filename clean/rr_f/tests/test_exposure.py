# SPDX-License-Identifier: MIT
"""Synthetic spec-58 tests; scalar oracle shares no exposure arithmetic helpers."""
from dataclasses import replace
from decimal import Decimal, localcontext
from fractions import Fraction
import math
import unittest

import numpy as np

from clean.rr_f.reference.exposure import (
    ExposureSymbols, auto_exposure, _exact_mean, _ftz, _log2, _rcp,
)


def symbols(**changes):
    # Deliberately invented, simple synthetic values, not build/model constants.
    values = dict(X_LWrb=0.25, X_LWg=0.5, X_LMAX=65504.0,
                  X_KEY=0.125, X_MFL=0.25, X_EFL=2.0 ** -20,
                  X_CL=8.0, X_CLn=-8.0, X_NRM=0.125)
    values.update(changes)
    return ExposureSymbols(**{k: np.float32(v) for k, v in values.items()})


def bits(value):
    return np.asarray(value, dtype=np.float32).view(np.uint32).item()


def flush(value):
    return math.copysign(0.0, value) if abs(value) < 2.0 ** -126 else value


def rounded(value):
    """Independent nearest-neighbour rational oracle, with f32 FTZ."""
    if not isinstance(value, Fraction):
        return flush(float(np.float32(value)))
    sign = -1 if value < 0 else 1
    magnitude = abs(value)
    overflow = Fraction(2) ** 128 - Fraction(2) ** 103
    if magnitude >= overflow:
        return math.copysign(math.inf, sign)
    with np.errstate(all="ignore"):
        initial = np.float32(float(magnitude))
    if np.isinf(initial):
        initial = np.finfo(np.float32).max
    neighbours = [initial, np.nextafter(initial, np.float32(0)),
                  np.nextafter(initial, np.float32(np.inf))]
    finite = [float(x) for x in neighbours if np.isfinite(x)]
    closest = min(finite, key=lambda x: (abs(Fraction(x) - magnitude), bits(x) & 1))
    return flush(math.copysign(closest, sign))


def op(a, b, multiply=False):
    a, b = flush(float(a)), flush(float(b))
    if not (math.isfinite(a) and math.isfinite(b)):
        return rounded(a * b if multiply else a + b)
    value = Fraction(a) * Fraction(b) if multiply else Fraction(a) + Fraction(b)
    return rounded(value)


def max_num(a, b):
    if math.isnan(a):
        return b
    if math.isnan(b):
        return a
    return max(a, b)


def min_num(a, b):
    return -max_num(-a, -b)


def reciprocal(a):
    a = flush(float(a))
    if math.isnan(a):
        return math.nan
    if math.isinf(a):
        return math.copysign(0.0, a)
    if a == 0:
        return math.copysign(math.inf, a)
    return rounded(1 / Fraction(a))


def logarithm(a):
    if a == 0:
        return -math.inf
    if a < 0 or math.isnan(a):
        return math.nan
    if a == math.inf:
        return a
    with localcontext() as ctx:
        ctx.prec = 150
        return rounded(Fraction(Decimal(float(a)).ln() / Decimal(2).ln()))


def scalar_oracle(colour, s, render_size, previous, reset=False, origin=(0, 0)):
    width, height = render_size
    ox, oy = origin
    weights = (s.X_LWrb, s.X_LWg, s.X_LWrb)
    total = Fraction(0)
    for y in range(oy, oy + height):
        for x in range(ox, ox + width):
            p = [op(float(colour[y, x, k]), float(weights[k]), True) for k in range(3)]
            w = min_num(max_num(op(op(p[0], p[1]), p[2]), 0.0), float(s.X_LMAX))
            total += Fraction(w)
    mean = rounded(total / (width * height))
    raw = op(s.X_KEY, reciprocal(max_num(mean, float(s.X_MFL))), True)
    with np.errstate(all="ignore"):
        stored = np.float16(raw)
    exposure = float(stored) if stored != 0 else 1.0
    previous = 1.0 if reset or previous == 0 else float(previous)
    inverse = reciprocal(exposure)
    change = 0.0
    if not reset:
        ratio = op(max_num(exposure, float(s.X_EFL)),
                   reciprocal(max_num(flush(previous), float(s.X_EFL))), True)
        change = op(min_num(max_num(logarithm(ratio), float(s.X_CLn)), float(s.X_CL)),
                    s.X_NRM, True)
    return dict(E=exposure, Ep=previous, invE=inverse, X3=change,
                E_raw=raw, mean=mean, stored=stored)


class Spec58Exposure(unittest.TestCase):
    def compare_oracle(self, colour, s, size, previous, reset=False, origin=(0, 0)):
        actual = auto_exposure(colour, symbols=s, render_size=size,
                               Eprev=previous, reset=reset, origin=origin)
        expected = scalar_oracle(colour, s, size, previous, reset, origin)
        for name, value in expected.items():
            result = getattr(actual, name)
            if name == "stored":
                self.assertIsInstance(result, np.float16)
                self.assertEqual(result.view(np.uint16), value.view(np.uint16), name)
            else:
                self.assertIsInstance(result, np.float32, name)
                if math.isnan(value):
                    self.assertTrue(math.isnan(result), name)
                else:
                    self.assertEqual(bits(result), bits(value), name)
        return actual

    def test_random_scalar_oracle(self):
        rng = np.random.default_rng(5802)
        for index in range(72):
            width, height = (int(x) for x in rng.integers(1, 9, size=2))
            colour = rng.normal(2, 5, (height + 3, width + 4, 4)).astype(np.float32)
            s = symbols(X_LWrb=(-0.25 if index % 3 == 0 else 0.25),
                        X_NRM=(-0.125 if index % 5 == 0 else 0.125))
            previous = np.float32(rng.choice([0, 1, 0.001, 0.3, 2.5, 1e9, -0.5]))
            self.compare_oracle(colour, s, (width, height), previous,
                                reset=index % 7 == 0, origin=(2, 1))

    def test_crop_origin_and_alpha_do_not_participate(self):
        colour = np.full((5, 7, 4), np.nan, dtype=np.float32)
        colour[1:4, 2:6, :3] = 2
        before = colour.copy()
        result = self.compare_oracle(colour, symbols(), (4, 3), 1, origin=(2, 1))
        self.assertEqual(result.mean, 2)
        np.testing.assert_array_equal(colour, before)
        tight = auto_exposure(colour[1:4, 2:6, :], symbols=symbols(),
                              render_size=(4, 3), Eprev=1, reset=False)
        self.assertEqual(result, tight)

    def test_nonfinite_channels_and_clamp(self):
        c = np.array([[[np.nan, 2, 3], [np.inf, 2, 3], [-np.inf, 2, 3],
                       [np.inf, -np.inf, 3], [-10, -10, -10]]], dtype=np.float32)
        result = self.compare_oracle(c, symbols(X_LMAX=10), (5, 1), 1)
        self.assertEqual(result.mean, 2)
        for previous in [np.nan, np.inf, -np.inf, -2, -0.0, 0, 2.0 ** -149]:
            with self.subTest(previous=previous):
                self.compare_oracle(c, symbols(), (5, 1), np.float32(previous))

    def test_reset_overrides_previous_and_change(self):
        c = np.full((2, 3, 3), 7, dtype=np.float32)
        for previous in [np.nan, np.inf, -4, 0, 2]:
            result = self.compare_oracle(c, symbols(), (3, 2), previous, reset=True)
            self.assertEqual(result.Ep, 1)
            self.assertEqual(bits(result.X3), 0)

    def test_binary16_subnormal_storage_and_next_frame(self):
        c = np.full((2, 2, 3), 65504, dtype=np.float32)
        first = self.compare_oracle(c, symbols(), (2, 2), 1, reset=True)
        self.assertGreater(first.stored, 0)
        self.assertLess(first.stored, np.finfo(np.float16).tiny)
        second = self.compare_oracle(c, symbols(), (2, 2), np.float32(first.stored))
        self.assertEqual(second.Ep, first.E)
        self.assertEqual(second.X3, 0)

    def test_f16_ties_round_to_even(self):
        c = np.zeros((1, 1, 3), dtype=np.float32)
        for raw, wanted in [(1 + 2.0 ** -11, 1), (1 + 3 * 2.0 ** -11, 1 + 2.0 ** -9),
                            (2.0 ** -25, 0), (3 * 2.0 ** -25, 2.0 ** -23)]:
            result = self.compare_oracle(c, symbols(X_KEY=raw, X_MFL=1), (1, 1), 1)
            self.assertEqual(result.stored, wanted)
            self.assertEqual(result.E, wanted if wanted else 1)

    def test_zero_storage_is_distinct_from_exposure(self):
        c = np.full((1, 1, 3), 4, dtype=np.float32)
        s = symbols(X_KEY=2.0 ** -126, X_MFL=1)
        first = self.compare_oracle(c, s, (1, 1), 1)
        self.assertEqual(first.stored, 0)
        self.assertEqual(first.E, 1)
        second = self.compare_oracle(c, s, (1, 1), np.float32(first.stored))
        self.assertEqual(second.Ep, 1)

    def test_ftz_input_product_sum_and_output_signs(self):
        tiny = np.float32(2.0 ** -126)
        sub = np.nextafter(tiny, np.float32(0))
        c = np.array([[[sub, -sub, sub], [tiny, tiny, tiny]]], dtype=np.float32)
        self.compare_oracle(c, symbols(), (2, 1), 1)
        self.assertEqual(bits(_ftz(-sub)), 0x80000000)
        self.assertEqual(bits(_rcp(-np.finfo(np.float32).max)), 0x80000000)
        c = np.array([[[tiny, np.nextafter(-tiny, np.float32(-np.inf)), 0]]], dtype=np.float32)
        self.assertEqual(self.compare_oracle(c, symbols(X_LWrb=1, X_LWg=1), (1, 1), 1).mean, 0)

    def test_final_change_product_ftz_keeps_negative_zero(self):
        c = np.zeros((1, 1, 3), dtype=np.float32)
        result = self.compare_oracle(c, symbols(X_NRM=2.0 ** -126), (1, 1), np.float32(0.75))
        self.assertEqual(bits(result.X3), 0x80000000)

    def test_exact_mean_adversarial_midpoint(self):
        # An ordinary f64 sum loses tiny terms. The exact mean lies just above
        # an f32 midpoint; rounding a double approximation would choose below.
        v = np.array([1, 1 + 2.0 ** -23, 2.0 ** -126, 2.0 ** -126], dtype=np.float32)
        expected = rounded(sum((Fraction(float(x)) for x in v), Fraction()) / len(v))
        self.assertEqual(bits(_exact_mean(v)), bits(expected))
        self.assertEqual(_exact_mean(v), np.nextafter(np.float32(0.5), np.float32(np.inf)))
        self.assertNotEqual(_exact_mean(v), np.float32(v.sum(dtype=np.float64) / 4))
        c = np.zeros((1, 4, 3), dtype=np.float32)
        c[0, :, 0] = v
        result = self.compare_oracle(c, symbols(X_LWrb=1, X_LWg=0), (4, 1), 1)
        self.assertEqual(result.mean, _exact_mean(v))

    def test_exact_mean_many_exponents_and_block_boundary(self):
        rng = np.random.default_rng(5858)
        raw = rng.integers(1 << 23, 255 << 23, size=(1 << 16) + 37, dtype=np.uint32)
        values = raw.view(np.float32)
        total = sum((Fraction(float(x)) for x in values), Fraction())
        expected = rounded(total / len(values))
        for arrangement in (values, values[::-1], np.sort(values)):
            self.assertEqual(bits(_exact_mean(arrangement)), bits(expected))
        # A low-precision accumulator loses every unit following the highlight.
        values = np.ones(131073, dtype=np.float32)
        values[0] = 2.0 ** 32
        self.assertEqual(bits(_exact_mean(values)), bits(rounded(Fraction(2 ** 32 + 131072, 131073))))

    def test_correct_scalar_reciprocal_log_and_actual_f32_outputs(self):
        for x in [2.0 ** -126, 0.3, 1, 1.0000001192092896, 3, 65504, 2.0 ** 127]:
            value = np.float32(x)
            self.assertEqual(bits(_rcp(value)), bits(reciprocal(value)))
            self.assertEqual(bits(_log2(value)), bits(logarithm(float(value))))
        c = np.full((1, 1, 3), 3, dtype=np.float32)
        result = self.compare_oracle(c, symbols(), (1, 1), np.float32(0.3))
        self.assertNotEqual(float(result.invE), 1 / float(result.E))

    def test_mean_result_subnormal_is_flushed(self):
        values = np.array([2.0 ** -126, 0], dtype=np.float32)
        self.assertEqual(bits(_exact_mean(values)), 0)
        c = np.zeros((1, 2, 3), dtype=np.float32)
        c[0, :, 0] = values
        result = self.compare_oracle(c, symbols(X_LWrb=1, X_LWg=0), (2, 1), 1)
        self.assertEqual(bits(result.mean), 0)

    def test_separate_reciprocal_product_and_floor(self):
        c = np.zeros((1, 1, 3), dtype=np.float32)
        self.compare_oracle(c, symbols(X_KEY=0.3, X_MFL=0.7), (1, 1), np.float32(0.3))

    def test_spec_range_counterexamples_preserve_literal_arithmetic(self):
        # These are invented symbol values satisfying 58.1, not observations
        # about either supported build. They expose gaps in the finite-output
        # assertions, without silently strengthening the published validation.
        mfl = np.uint32(0x3f8005a9).view(np.float32)
        key = np.uint32(0x477ffb51).view(np.float32)
        self.assertLess(Fraction(float(key)) / Fraction(float(mfl)), 65520)
        c = np.zeros((1, 1, 3), dtype=np.float32)
        result = self.compare_oracle(c, symbols(X_KEY=key, X_MFL=mfl), (1, 1), 1)
        self.assertEqual(result.E_raw, 65520)
        self.assertTrue(np.isinf(result.stored))
        result = self.compare_oracle(c, symbols(X_KEY=4, X_MFL=1, X_EFL=1,
                                    X_CL=2, X_CLn=-2, X_NRM=np.finfo(np.float32).max),
                                    (1, 1), 1)
        self.assertTrue(np.isinf(result.X3))

    def test_symbol_rejection_and_stated_ranges(self):
        valid = symbols()
        for name in valid.__dataclass_fields__:
            for bad in [np.nan, np.inf, -np.inf, 2.0 ** -149, -2.0 ** -149, [1], True, 0.1]:
                with self.subTest(name=name, bad=bad), self.assertRaises(ValueError):
                    replace(valid, **{name: bad})
        for name in ["X_KEY", "X_MFL", "X_EFL"]:
            for bad in [0, -1]:
                with self.assertRaises(ValueError):
                    replace(valid, **{name: bad})
        for changes in [dict(X_LMAX=-1), dict(X_CLn=9),
                        dict(X_KEY=65520, X_MFL=1), dict(X_KEY=65536, X_MFL=1)]:
            with self.assertRaises(ValueError):
                replace(valid, **changes)
        self.assertEqual(replace(valid, X_KEY=65504, X_MFL=1).X_KEY, 65504)
        # No undocumented positivity requirements for weights or normaliser.
        replace(valid, X_LWrb=-1, X_LWg=0, X_NRM=-1, X_LMAX=0)

    def test_shape_bounds_scalar_and_zero_size_rejection(self):
        c = np.ones((2, 3, 4), dtype=np.float32)
        kwargs = dict(symbols=symbols(), render_size=(3, 2), Eprev=1, reset=False)
        for size in [(0, 2), (3, 0), (-1, 2), (4, 2), (3, 3), (1.5, 1), (True, 1), (1,)]:
            with self.subTest(size=size), self.assertRaises(ValueError):
                auto_exposure(c, **{**kwargs, "render_size": size})
        for origin in [(-1, 0), (0, -1), (1, 0), (0, 1), (0.5, 0)]:
            with self.subTest(origin=origin), self.assertRaises(ValueError):
                auto_exposure(c, **kwargs, origin=origin)
        for image in [c.astype(np.float64), c.astype(np.uint8), c[:, :, :2], c[:, :, 0]]:
            with self.assertRaises(ValueError):
                auto_exposure(image, **kwargs)
        for changes in [dict(Eprev=[1]), dict(Eprev=0.1), dict(reset=1)]:
            with self.assertRaises(ValueError):
                auto_exposure(c, **{**kwargs, **changes})
        with self.assertRaises(TypeError):
            auto_exposure(c, **{**kwargs, "symbols": {}})


if __name__ == "__main__":
    unittest.main()
