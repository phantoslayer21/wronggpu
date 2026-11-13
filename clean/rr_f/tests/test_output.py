# SPDX-License-Identifier: MIT
"""Synthetic, independently derived checks of published page 54 only."""
from dataclasses import replace
import itertools
import math
import unittest
from unittest.mock import patch

import numpy as np

from clean.rr_f.reference import output as out


def symbols(**changes):
    """Arbitrary test symbols; not model constants or page-57 substitutions."""
    table = np.zeros((513, 4), np.float32)
    table[:, 1] = 1
    values = dict(K1=1 / 16, K2n=-1, K2x2n=-2, K3=0.5, K4=1,
                  K5=0, K7=1, K8=1, K10=10, K11=0.25, K12=1,
                  K13=2**-20, TR=table)
    values.update(changes)
    return out.OutputConstants(**values)


def inputs(render=(2, 2), output=(2, 2)):
    wr, hr = render
    wo, ho = output
    wh, hh = 3 * (wo + wo % 2) // 2, 3 * (ho + ho % 2) // 2
    color = np.zeros((((hr + 31) // 32) * 32, ((wr + 31) // 32) * 32, 4))
    color[:hr, :wr, :3] = [1, 2, 3]
    return dict(Cnet=color, G=np.ones((hr, wr, 3)), P=np.zeros((hr, wr, 3)),
                W4=out.decode_unorm8(np.full((hr, wr, 4), 64, np.uint8)), PM=np.zeros((hr, wr, 4)),
                Pe=np.zeros((hr, wr, 2)), Pn=np.zeros((hr, wr, 2)),
                Hprev=np.zeros((hh, wh, 4)), output_size=output, jitter=(0, 0),
                E=1, Ep=1, invE=1, reset=False, constants=symbols())


def scalar_compress(v, c):
    # Test values have K4=1 and need no fused cancellation; Python float
    # computes the exact addition, then explicit np.float32 marks each step.
    value = np.float32(max(np.float32(float(v) * float(c.K4) + 1), c.K5))
    return np.float32(np.float32(np.float32(math.log2(value)) * np.float32(math.log10(2))) * c.K7)


def scalar_decompress(v, c):
    argument = max(min(float(v), float(c.K10)), float(c.K5))
    exponent = np.float32(np.float32(np.float32(argument) * c.K8) * np.float32(math.log2(10)))
    return np.float32(np.float32(2.0 ** float(exponent)) - np.float32(1))


class NumericTests(unittest.TestCase):
    def test_reciprocal_signed_zeros_and_flushed_inputs_in_both_call_orders(self):
        for values in ([0.0, -0.0], [-0.0, 0.0], [2**-149, -2**-149], [-2**-149, 2**-149]):
            out._elementary.cache_clear()
            for value in values:
                actual = out.rcp(value)
                self.assertTrue(np.isinf(actual))
                self.assertEqual(bool(np.signbit(actual)), bool(np.signbit(value)))
            vector = out.rcp(values)
            np.testing.assert_array_equal(np.signbit(vector), np.signbit(values))

    def test_f32_fused_cancellation_and_flush(self):
        # (1+2^-23)*(1-2^-23)-1 = -2^-46; separate f32 mul loses it.
        self.assertEqual(float(out.fma(1 + 2**-23, 1 - 2**-23, -1)), -2**-46)
        self.assertEqual(float(out.add(2**-149, 2**-126)), 2**-126)
        self.assertEqual(float(out.mul(2**-126, 0.5)), 0)
        self.assertTrue(np.signbit(out.mul(-2**-126, 0.5)))
        self.assertEqual(float(out.rcp(np.float32(2**-149))), math.inf)
        self.assertEqual(float(out.log2(np.float32(2**-149))), -math.inf)
        self.assertEqual(float(out.exp2(-127)), 0)

    def test_encoding_nan_branch_and_signed_zero(self):
        c = symbols(K5=0.25)
        expected = np.float32(np.float32(-2 * np.float32(math.log10(2))) * c.K7)
        self.assertEqual(float(out.compress(np.nan, c)), float(expected))
        self.assertTrue(np.isnan(out.decompress(np.nan, c)))
        self.assertEqual(float(out.decompress(-100, c)), float(out.decompress(0.25, c)))
        self.assertTrue(np.signbit(out.fma(-0.0, 1, -0.0)))
        self.assertFalse(np.signbit(out.fma(-0.0, 1, 0.0)))

    def test_unorm_readback_and_rgb10a2_store(self):
        codes = np.arange(256, dtype=np.uint8)
        np.testing.assert_array_equal(out.decode_unorm8(codes), (codes.astype(np.float64) / 255).astype(np.float16))
        self.assertNotEqual(float(out.decode_unorm8(np.array(128, np.uint8))), float(np.float32(128 / 255)))
        np.testing.assert_array_equal(out.unorm_codes([np.nan, -np.inf, 0, 0.5, 1, np.inf], 10),
                                      [0, 0, 0, 512, 1023, 1023])
        np.testing.assert_array_equal(out.unorm_codes([0, 0.5, 1], 2), [0, 2, 3])
        # Values immediately around the only exact f32 interior half-code tie.
        values = [np.nextafter(np.float32(0.5), np.float32(0)), 0.5,
                  np.nextafter(np.float32(0.5), np.float32(1))]
        np.testing.assert_array_equal(out.unorm_codes(values, 10), [511, 512, 512])

    def test_constants_are_explicit_and_only_stated_ranges_checked(self):
        self.assertRaises(TypeError, out.OutputConstants)
        for name in ("K2n", "K2x2n"):
            with self.assertRaises(ValueError):
                symbols(**{name: 0})
        with self.assertRaises(ValueError):
            symbols(TR=np.zeros((512, 4)))
        with self.assertRaises(ValueError):
            symbols(K1=0.1)  # not an exact f32 supplied symbol
        self.assertEqual(float(symbols(K3=-1).K3), -1)  # no invented K3 range


class CandidateTests(unittest.TestCase):
    def test_distinct_permutations(self):
        for values in itertools.permutations((0.1, 0.2, 0.3, 0.4)):
            expected = sorted(range(4), key=lambda i: values[i], reverse=True)
            np.testing.assert_array_equal(out.candidate_order(values), expected)

    def test_hand_derived_tie_orders(self):
        cases = [
            ([1, 1, 1, 1], [0, 1, 2, 3]),
            ([1, 1, 1, 0], [0, 1, 2, 3]),
            ([1, 1, 0, 1], [0, 1, 3, 2]),
            ([1, 0, 1, 1], [0, 3, 2, 1]),
            ([0, 1, 1, 1], [1, 3, 2, 0]),
            ([0, 0, 0, 1], [3, 1, 0, 2]),
            ([0, 0, 1, 0], [2, 1, 0, 3]),
            ([0, 1, 0, 0], [1, 0, 2, 3]),
            ([1, 0, 0, 0], [0, 1, 2, 3]),
            ([1, 1, 0, 0], [0, 1, 2, 3]),
            ([1, 0, 1, 0], [0, 2, 1, 3]),
            ([1, 0, 0, 1], [0, 3, 1, 2]),
            ([0, 1, 1, 0], [1, 2, 0, 3]),
            ([0, 1, 0, 1], [1, 3, 0, 2]),
            ([0, 0, 1, 1], [2, 3, 0, 1]),
            ([2, 1, 1, 0], [0, 1, 2, 3]),
            ([1, 2, 1, 0], [1, 0, 2, 3]),
            ([1, 0, 2, 1], [2, 3, 0, 1]),
        ]
        for values, expected in cases:
            with self.subTest(values=values):
                np.testing.assert_array_equal(out.candidate_order(values), expected)

    def test_nan_order_before_saturation(self):
        for values, expected in [([np.nan, 4, 3, 2], [0, 1, 2, 3]),
                                 ([4, np.nan, 3, 2], [0, 1, 2, 3]),
                                 ([4, 3, np.nan, 2], [0, 1, 2, 3]),
                                 ([4, 3, 2, np.nan], [0, 1, 2, 3]),
                                 ([np.nan, 1, 2, 3], [0, 3, 2, 1])]:
            np.testing.assert_array_equal(out.candidate_order(values), expected)
        order, norm = out.candidate_weights([1.5, 2, -1, np.nan], symbols())
        np.testing.assert_array_equal(order, [1, 0, 2, 3])
        np.testing.assert_array_equal(norm, [0.5, 0.5, 0, 0])

    def test_strict_threshold_and_fourth_weight_only_in_denominator(self):
        history = np.zeros((3, 3, 4))
        motions = np.array([[i, 0] for i in range(4)])
        calls = []

        def sampled(_history, _x, _y, motion, *_args):
            calls.append(int(motion[0]))
            return np.full(3, (int(motion[0]) + 1) / 8, np.float32)

        with patch.object(out, "reconstruct_history", side_effect=sampled):
            out.history_estimate(history, 0, 0, [0.25] * 4, motions,
                                 np.ones(3), 1, 1, 1, False, symbols(K3=0.5))
            self.assertEqual(calls, [0, 1])  # equality at 0.5 stops before p2
            calls.clear()
            actual = out.history_estimate(history, 0, 0, [0.25] * 4, motions,
                                          np.ones(3), 1, 1, 1, False, symbols(K3=1))
            self.assertEqual(calls, [0, 1, 2])  # p3 never fetched
        # Three equal compressed samples average to 1/4 exactly.
        np.testing.assert_array_equal(actual, np.full(3, scalar_decompress(0.25, symbols())))


class SamplingAndGuardTests(unittest.TestCase):
    def test_bilinear_clamp_nan_and_zero_integer_fetch(self):
        image = np.array([[[1], [3]], [[5], [7]]], np.float16)
        self.assertEqual(float(out.bilinear(image, 0.5, 0.5)[0]), 4)
        self.assertEqual(float(out.bilinear(image, -2, 0.25)[0]), 1)
        self.assertEqual(float(out.bilinear(image, 2, 0.75)[0]), 7)
        self.assertEqual(float(out.bilinear(image, np.nan, np.nan)[0]), 1)
        np.testing.assert_array_equal(out.integer_fetch(image, -1, 0), [0])
        np.testing.assert_array_equal(out.integer_fetch(image, 2, 0), [0])

    def test_reprojection_bounds_are_strict_and_nan_motion_is_not_fallback(self):
        image = np.zeros((3, 3, 4))
        image[..., :3] = 0.25
        fallback = np.full(3, 9)
        c = symbols()
        for motion in [(-0.5, 0), (2.5, 0), (0, -0.5), (0, 2.5)]:
            np.testing.assert_array_equal(out.reconstruct_history(image, 0, 0, motion, fallback, False, c), [0.25] * 3)
        for motion in [(-0.5001, 0), (2.5001, 0), (0, -0.5001), (0, 2.5001), (np.inf, 0), (-np.inf, 0)]:
            np.testing.assert_array_equal(out.reconstruct_history(image, 0, 0, motion, fallback, False, c), fallback)
        image[:, 0, :3] = 0.5
        # floor(NaN) -> NaN anchor; sat(NaN) -> TR[0]; normalized NaN
        # replaced by 0 in 54.N. No fabricated integer coordinate is used.
        np.testing.assert_array_equal(out.reconstruct_history(image, 0, 0, (np.nan, 0), fallback, False, c), [0.5] * 3)
        np.testing.assert_array_equal(out.reconstruct_history(image, 0, 0, (np.nan, 0), fallback, True, c), fallback)

    def test_reconstruction_table_rounding_indices(self):
        c = symbols()
        table = np.array(c.TR)
        table[:, 3] = 0
        table[0, 3], table[1, 3], table[512, 3] = 0, 1, 2
        image = np.zeros((3, 3, 4))
        image[..., :3] = np.arange(3)[None, :, None]
        c = replace(c, TR=table)
        # ix=trunc(fx*512+0.5): just below 1/1024 -> 0, at tie -> 1.
        below = np.nextafter(np.float32(0.5 + 1 / 1024), np.float32(0.5)) - np.float32(0.5)
        a = out.reconstruct_history(image, 0, 0, (below, 0), [99]*3, False, c)
        b = out.reconstruct_history(image, 0, 0, (1 / 1024, 0), [99]*3, False, c)
        np.testing.assert_array_equal(a, [0]*3)
        np.testing.assert_array_equal(b, [1]*3)
        last = out.reconstruct_history(image, 0, 0, (1023 / 1024, 0), [99]*3, False, c)
        np.testing.assert_array_equal(last, [2]*3)

    def test_five_tap_weights_and_nested_sum(self):
        table = np.tile([1, 2, 3, 0.25], (513, 1))
        c = symbols(TR=table)
        yy, xx = np.mgrid[:8, :8]
        image = np.zeros((8, 8, 4))
        image[..., :3] = (10 * yy + xx * xx)[..., None]
        actual = out.reconstruct_history(image, 2, 2, [0, 0], [99]*3, False, c)
        # U,L,C,D,R = 15.25,23.5,27.75,45.25,38.5, weights 2,2,4,6,6.
        expected = np.float32(691 * np.float32(1 / 20))
        np.testing.assert_array_equal(actual, [expected]*3)

    def test_nan_guard_replaces_all_channels_and_is_gate_local(self):
        current = np.array([1, 2, 3], np.float32)
        history = np.array([np.nan, 6, 7], np.float32)
        np.testing.assert_array_equal(out.blend_guard(current, history, 0.25, symbols()), current)
        finite = out.blend_guard(current, [5, 6, 7], 0.25, symbols())
        np.testing.assert_array_equal(finite, [4, 5, 6])
        # Infinity alone in the sum is not a reason to invoke the guard.
        infinite = out.blend_guard([np.inf, 2, 3], [0, 0, 0], 1, symbols())
        self.assertTrue(np.isposinf(infinite[0]))
        np.testing.assert_array_equal(infinite[1:], [2, 3])
        # Opposite infinities make the sum NaN and restore every channel.
        mixed = out.blend_guard([np.inf, -np.inf, 3], [0, 0, 0], 0.5, symbols())
        self.assertEqual(mixed[2], 3)


class WholeStageTests(unittest.TestCase):
    def test_uniform_hand_calculation_and_formats(self):
        args = inputs()
        result = out.output_stage(**args)
        c = args["constants"]
        expected = []
        for color in (1, 2, 3):
            compressed = np.float16(scalar_compress(color, c))
            total = np.float16(0)
            for _ in range(16):
                total = np.float16(float(total) + float(compressed))
            normalized = np.float32(float(total) / 16)
            current = scalar_decompress(normalized, c)
            expected.append(np.float16(scalar_compress(current, c)))
        np.testing.assert_array_equal(result.CH[..., :3], np.broadcast_to(expected, (3, 3, 3)))
        np.testing.assert_array_equal(result.Q, result.CH[..., :3])
        np.testing.assert_array_equal(result.CH[..., 3], 0)
        self.assertEqual(result.CH.dtype, np.float16)
        self.assertEqual(result.Q.dtype, np.float16)
        self.assertEqual(result.PH1.dtype, np.uint16)
        self.assertEqual(result.PH1.shape, (2, 2, 4))
        np.testing.assert_array_equal(result.PH1[..., 3], 3)

    def test_reset_history_independence_with_nan_motion(self):
        args = inputs(render=(2, 1), output=(3, 1))
        args["reset"] = True
        args["G"][:] = out.decode_unorm8(np.array([0, 64, 191], np.uint8))
        args["P"][:] = [1, 1, 0]
        args["PM"][..., 0] = np.nan
        first = out.output_stage(**args)
        args["Hprev"][:] = [np.nan, np.inf, -np.inf, 4]
        second = out.output_stage(**args)
        for name in ("CH", "Q", "PH1"):
            np.testing.assert_array_equal(getattr(first, name), getattr(second, name))

    def test_distinct_internal_and_output_gates(self):
        args = inputs()
        args["G"][:] = [0, 0, 1]
        args["Hprev"][..., :3] = 0.125
        result = out.output_stage(**args)
        self.assertFalse(np.array_equal(result.CH[..., :3], result.Q))
        self.assertTrue(np.all(result.CH[..., :3] < result.Q))

    def test_zero_candidate_weights_trigger_guard_to_current(self):
        args = inputs()
        args["W4"][:] = 0
        args["G"][:] = [0, 0, 0]
        zero = out.output_stage(**args)
        args["W4"][:] = out.decode_unorm8(np.array(64, np.uint8))
        args["G"][:] = 1
        fresh = out.output_stage(**args)
        np.testing.assert_array_equal(zero.CH, fresh.CH)
        np.testing.assert_array_equal(zero.Q, fresh.Q)

    def test_padding_never_read_and_input_validation(self):
        args = inputs()
        original = out.output_stage(**args)
        args["Cnet"][2:, :, :] = np.nan
        args["Cnet"][:, 2:, :] = np.inf
        actual = out.output_stage(**args)
        np.testing.assert_array_equal(actual.CH, original.CH)
        args["jitter"] = (0.5, 0)
        with self.assertRaisesRegex(ValueError, "54.8"):
            out.output_stage(**args)
        args = inputs()
        args["G"][0, 0, 0] = 0.1
        with self.assertRaisesRegex(ValueError, "binary16"):
            out.output_stage(**args)

    def test_odd_pooling_valid_count_and_linear_before_compression(self):
        c = symbols()
        linear = np.repeat(np.array([[0, 3, 15], [7, 15, 31], [63, 127, 255]], np.float32)[..., None], 3, axis=2)
        pooled = out.pool_history(linear, c)
        # Explicit valid-pixel means in the four blocks: 6.25, 23, 95, 255.
        means = np.array([[6.25, 23], [95, 255]])
        expected = np.array([[round(float(scalar_compress(v, c)) * 0.25 * 1023) for v in row] for row in means])
        for channel in range(3):
            np.testing.assert_array_equal(pooled[..., channel], expected)
        # Averaging compressed pixels would produce a different result.
        wrong = round(sum(float(scalar_compress(v, c)) for v in [0, 3, 7, 15]) / 4 * 0.25 * 1023)
        self.assertNotEqual(int(pooled[0, 0, 0]), wrong)

    def test_full_stage_rejects_prestore_unorm_and_vector_exposure(self):
        for name, bad in (("G", 0.5), ("W4", 0.25), ("G", np.nan), ("W4", np.inf)):
            args = inputs()
            args[name][0, 0, 0] = bad
            with self.subTest(name=name, bad=bad):
                with self.assertRaisesRegex(ValueError, "UNORM8 read-back"):
                    out.output_stage(**args)
        for name in ("E", "Ep", "invE"):
            for bad in ([1, 2, 3], [1], 0.1):
                args = inputs()
                args[name] = bad
                with self.subTest(name=name, bad=bad):
                    with self.assertRaisesRegex(ValueError, "f32"):
                        out.output_stage(**args)
        args = inputs()
        args["reset"] = np.array([True, False])
        with self.assertRaisesRegex(ValueError, "boolean"):
            out.output_stage(**args)

    def test_all_unorm_codes_are_valid_full_stage_inputs(self):
        # Validate all byte values at the same boundary used by output_stage.
        codes = np.arange(256, dtype=np.uint8).reshape(16, 16, 1)
        decoded = out.decode_unorm8(codes)
        np.testing.assert_array_equal(out._unorm_texture(decoded, (16, 16, 1), "test"), decoded)

    def test_pooling_uses_specified_column_pair_sum_order(self):
        linear = np.repeat(np.array([[2**24, 1], [-2**24, 1]], np.float32)[..., None], 3, axis=2)
        result = out.pool_history(linear, symbols())
        # (2^24 + -2^24) + (1+1), then /4 = 0.5; row-major gives 0.25.
        expected = round(float(scalar_compress(0.5, symbols())) * 0.25 * 1023)
        np.testing.assert_array_equal(result[0, 0], [expected, expected, expected, 3])

    def test_vertical_pair_gather_matches_independent_scalar_calculation(self):
        args = inputs(render=(3, 2), output=(3, 2))
        args["Cnet"][0, :3, :3] = 1
        args["Cnet"][1, :3, :3] = 8
        args["P"][:] = [1, 1, 0]
        args["constants"] = symbols(K1=16)
        actual = out.output_stage(**args)
        c = args["constants"]

        def expected_pixel(x, y, shared):
            # Independent scalar formula, ordinary Python math and NumPy
            # casts; no implementation arithmetic/gather/sampling helpers.
            sx, sy = np.float32(3 / 6), np.float32(2 / 3)
            u, v = np.float32((x + 0.5) * float(sx)), np.float32((y + 0.5) * float(sy))
            va = np.float32(((y // 2) * 2 + 0.5) * float(sy)) if shared else v
            cx, cy, cya = math.floor(float(np.float32(u - 0.5))), math.floor(float(np.float32(v - 0.5))), math.floor(float(np.float32(va - 0.5)))
            hx, hy = np.float16(cx + 0.5 - float(u)), np.float16(cy + 0.5 - float(v))
            if shared and cy != cya:
                hy = np.float16(float(hy) - 1)
            total, accum = np.float32(0), np.float16(0)
            for k in range(4):
                for m in range(4):
                    ox, oy = np.float16(float(hx) + k - 1), np.float16(float(hy) + m - 1)
                    exponent = np.float16(-float(np.float16(float(ox)**2)) - float(np.float16(float(oy)**2)))
                    weight = np.float32(2.0 ** float(exponent))
                    tap = 1 if min(max(cya - 1 + m, 0), 1) == 0 else 8
                    compressed = np.float16(scalar_compress(tap, c))
                    accum = np.float16(float(np.float16(weight)) * float(compressed) + float(accum))
                    total = np.float32(total + weight)
            normalized = np.float32(float(accum) * float(np.float32(1 / float(total))))
            return np.float16(scalar_compress(scalar_decompress(normalized, c), c))

        expected = np.array([[expected_pixel(x, y, True) for x in range(6)] for y in range(3)])
        np.testing.assert_array_equal(actual.CH[..., 0], expected)
        centered = np.array([expected_pixel(x, 1, False) for x in range(6)])
        self.assertTrue(np.any(expected[1] != centered))


if __name__ == "__main__":
    unittest.main()
