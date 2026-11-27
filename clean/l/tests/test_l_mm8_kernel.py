# SPDX-License-Identifier: GPL-3.0-only
"""The FP16-form kernels' arithmetic of one FP8 matrix op (clean/l/runner/l_mm8_f32_check.py op_kernel) against the
oracle's exact model, on adversarial ops: heavy cancellation, wide dynamic range, sums at FP16 ties, a large start whose
float32 addition lands on an FP16 midpoint, subnormal operands, the overflow boundary and non-finite inputs. The plain
float32-sequential sum (op_f32seq) is shown to fail where the kernel's corrections are needed, and each correction,
removed, fails these tests.

    python -m unittest clean.l.tests.test_l_mm8_kernel
"""
import unittest

import numpy as np

from clean.l.runner import l_mm8_f32_check as M


def _op(start, prods):
    """One op from (a, w) pairs at positions 0, 1, ...: start [1], a [32], w [1, 32]."""
    a, w = np.zeros(32), np.zeros((1, 32))
    for k, (x, y) in enumerate(prods):
        a[k], w[0, k] = x, y
    return np.array([float(start)]), a, w


# Constructed start-midpoint ops (synthetic: chosen from the property, no data behind them). The start is an FP16
# number whose float32 sum with one product lands exactly on an FP16 midpoint, and a second, tiny product decides the
# rounding: the exact sum lies just below (or beyond) the midpoint, but in float32 the tiny product is half an ulp, the
# sum ties back to the midpoint, and RN16's ties-to-even then picks the other neighbour (the lower neighbour's FP16
# mantissa is odd). Every operand is an FP8 value (0.125, 0.5, 1, 2^-7, 2^-8).
#   300.25 (FP16 spacing 0.25 in [256, 512); 300.25 / 0.25 = 1201, odd) + 0.125 - 2^-16: exact -> 300.25,
#   float32-sequential -> the midpoint 300.375 -> ties to even -> 300.5.
#   -1500 (spacing 1 in [1024, 2048)) - 0.5 - 2^-14: exact -> -1501, float32-sequential -> -1500.5 -> -1500.
MIDPOINT_CASES = (
    (300.25, ((0.125, 1.0), (2.0 ** -8, -(2.0 ** -8))), 300.25),
    (-1500.0, ((-0.5, 1.0), (2.0 ** -7, -(2.0 ** -7))), -1501.0),
)


class KernelOpTest(unittest.TestCase):
    N = 2000

    def _family(self, name, op):
        return M.compare_family(name, np.random.default_rng([11, M.FAMILIES.index(name)]), self.N, op=op)

    def test_kernel_equals_exact_on_every_family(self):
        for name in M.FAMILIES:
            with self.subTest(family=name):
                r = self._family(name, M.op_kernel)
                self.assertEqual(r["differ"], 0, r["examples"][:3])

    def test_plain_float32_fails_where_the_corrections_are_needed(self):
        self.assertGreater(self._family("cancellation", M.op_f32seq)["differ"], 0)
        self.assertGreater(self._family("start_midpoint", M.op_f32seq)["differ"], 0)

    def test_constructed_midpoint_cases(self):
        for start, prods, want in MIDPOINT_CASES:
            st, a, w = _op(start, prods)
            self.assertEqual(M.op_exact(st, a, w)[0], want)
            self.assertEqual(M.op_kernel(st, a, w)[0], want)
            self.assertNotEqual(M.op_f32seq(st, a, w)[0], want)

    def test_without_the_slow_path_cancellation_fails(self):
        old = M.ABS_FAST
        M.ABS_FAST = np.float32(1e30)
        try:
            self.assertGreater(self._family("cancellation", M.op_kernel)["differ"], 0)
        finally:
            M.ABS_FAST = old

    def test_without_the_midpoint_fix_the_start_cases_fail(self):
        def no_fix(start, a8, W8):
            # the start added with one float32 addition and converted (the float32 product sum is exact here)
            return M.op_f32seq(start, a8, W8)
        self.assertGreater(self._family("start_midpoint", no_fix)["differ"], 0)
        for start, prods, want in MIDPOINT_CASES:
            st, a, w = _op(start, prods)
            self.assertNotEqual(no_fix(st, a, w)[0], want)

    def test_overflow_boundary_and_specials(self):
        cases = [(65504.0, ((4.0, 4.0),)), (65504.0, ((4.0, 4.0), (2.0 ** -9, -(2.0 ** -9)))),
                 (-65504.0, ((-4.0, 4.0),)), (65504.0, ((2.0, 4.0),)), (0.0, ((448.0, 448.0), (448.0, 448.0))),
                 (np.inf, ((1.0, 1.0),)), (1.0, ((np.nan, 1.0),)), (-np.inf, ((448.0, 448.0),))]
        for start, prods in cases:
            st, a, w = _op(start, prods)
            want, got = M.op_exact(st, a, w)[0], M.op_kernel(st, a, w)[0]
            with self.subTest(start=start, prods=prods):
                self.assertTrue(want == got or (np.isnan(want) and np.isnan(got)), (want, got))


if __name__ == "__main__":
    unittest.main()
