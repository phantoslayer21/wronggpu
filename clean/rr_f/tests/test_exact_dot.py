# SPDX-License-Identifier: MIT
"""Independent scalar regressions for the standalone exact spec51 dot baseline."""
import unittest
from unittest import mock

import numpy as np

from clean.rr_f.gpu import exact_dot
from clean.rr_f.reference import fmt


class ExactDotTests(unittest.TestCase):
    def test_integer_oracle_against_own_mm(self):
        rng = np.random.default_rng(5101)
        finite = np.array([c for c in range(256) if c & 127 != 127], np.uint8)
        a = rng.choice(finite, (2048, 32))
        b = rng.choice(finite, (2048, 32))
        c = rng.integers(0, 65536, 2048, dtype=np.uint16)
        result = fmt.mm(fmt.d8(a)[:, None, :], fmt.d8(b)[:, :, None],
                        fmt.from_f16_bits(c)[:, None, None])[:, 0, 0]
        expected = fmt.to_f16_bits(result)
        actual = np.array([exact_dot.integer_oracle(x, y, z) for x, y, z in zip(a, b, c)], np.uint16)
        self.assertTrue(np.all(exact_dot.same_bits(actual, expected)))

    def test_double_rounding_counterexample(self):
        a = np.zeros((1, 32), np.uint8)
        b = a.copy()
        a[0, :2] = [0x38, 0x10]  # 1, 1/32
        b[0, :2] = [0x38, 0x08]  # 1, 1/64
        c = np.array([1], np.uint16)  # smallest positive f16 subnormal
        result = exact_dot.integer_oracle(a[0], b[0], c[0])
        self.assertEqual(result, 0x3c01)
        total = 1.0 + 2.0**-11 + 2.0**-24
        self.assertEqual(np.float16(np.float32(total)).view(np.uint16), 0x3c00)

    def test_zero_and_nonfinite_contract(self):
        a = np.zeros(32, np.uint8)
        b = a.copy()
        self.assertEqual(exact_dot.integer_oracle(a, b, 0x8000), 0)
        for c in (0x7c00, 0xfc00):
            self.assertEqual(exact_dot.integer_oracle(a, b, c), c)
        a[31] = 0x7f
        self.assertEqual(exact_dot.integer_oracle(a, b, 0x7c00), 0x7e00)

    def test_range_proof_and_half_boundaries(self):
        self.assertLess((32 * 448**2 + 65504) * 2**24, 2**47)
        self.assertLess(2**47, 2**53)
        values = exact_dot.rounding_cases()
        self.assertEqual(len(values), 256012)
        with np.errstate(over="ignore", invalid="ignore"):
            expected = values.astype(np.float16).view(np.uint16)
        actual = fmt.to_f16_bits(fmt.round_f16(values))
        self.assertTrue(np.all(exact_dot.same_bits(actual, expected)))

    @mock.patch.object(exact_dot, "_dispatch")
    def test_invalid_shapes_do_not_dispatch(self, dispatch):
        a = np.zeros((2, 32), np.uint8)
        c = np.zeros(2, np.uint16)
        for aa, bb, cc in ((a.astype(np.int32), a, c), (a, a[:1], c),
                           (a, a, c.astype(np.uint32)), (a, a, c[:, None]),
                           (a[:, :31], a[:, :31], c)):
            with self.assertRaises(ValueError):
                exact_dot.dot32(aa, bb, cc)
        dispatch.assert_not_called()


if __name__ == "__main__":
    unittest.main()
