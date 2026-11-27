# SPDX-License-Identifier: GPL-3.0-only
"""Model K's softmax rounding probe cases (clean/k/dx12/k_mix_cases.py) and the runtime's table (clean/k/runtime/
k_mix_probe.h): CPU only.
- the committed header is exactly gen_k_mix_probe.py's output;
- the exact oracle (Fraction arithmetic) agrees with an independent conversion: numpy's binary64 -> binary16 rounding of
  the products, which are exact in binary64 (two 24-bit significands);
- the cases cover every kind (FP16-midpoint products with residuals of both signs, normal and subnormal FP16, exact
  ties, ordinary products) and a second rounding through FP32 differs from the exact result in many of them, so the
  probe catches a missing fold and the residual-dropped control.
Run: python -m unittest clean.k.tests.test_k_mix_probe_table
"""
import os
import unittest

import numpy as np

from clean.k.dx12 import gen_k_mix_probe, k_mix_cases

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


class TestKMixProbeTable(unittest.TestCase):
    def test_header_is_generated(self):
        with open(os.path.join(ROOT, "clean", "k", "runtime", "k_mix_probe.h"), newline="") as f:
            self.assertEqual(f.read(), gen_k_mix_probe.header(), "regenerate: python clean/k/dx12/gen_k_mix_probe.py")

    def test_oracle_matches_numpy(self):
        for seed in (7, 2000, 2001):
            rec, want, _ = k_mix_cases.cases(seed, 64)
            prod = rec[:, 1:9].astype(np.float64) * rec[:, :1].astype(np.float64)
            self.assertTrue(np.array_equal(prod.astype(np.float16).view(np.uint16), want), seed)
        rng = np.random.default_rng(5)
        x = (2.0 ** rng.uniform(-26, 0, 4000)).astype(np.float32).astype(np.float64)
        got = np.array([k_mix_cases.rn16_bits(float(v)) for v in x], np.uint16)
        self.assertTrue(np.array_equal(x.astype(np.float16).view(np.uint16), got))

    def test_rn16_corners(self):
        from fractions import Fraction as F
        self.assertEqual(k_mix_cases.rn16_bits(1), 0x3C00)
        self.assertEqual(k_mix_cases.rn16_bits(F(1, 2 ** 24)), 0x0001)            # smallest subnormal
        self.assertEqual(k_mix_cases.rn16_bits(F(1, 2 ** 25)), 0x0000 + 0)        # tie between 0 and 2^-24: even (0)
        self.assertEqual(k_mix_cases.rn16_bits(F(3, 2 ** 25)), 0x0002)            # tie 1.5 ulp: even (2)
        self.assertEqual(k_mix_cases.rn16_bits(F(2047, 2 ** 25)), 0x0400)         # rounds up into the normal range
        self.assertEqual(k_mix_cases.rn16_bits(F(2049, 2 ** 11)), 0x3C00)         # 1 + 2^-11: tie, even (1.0)
        self.assertEqual(k_mix_cases.rn16_bits(F(2051, 2 ** 11)), 0x3C02)         # 1 + 3 2^-11: tie, even
        self.assertEqual(k_mix_cases.rn16_bits(65504), 0x7BFF)
        self.assertEqual(k_mix_cases.rn16_bits(65520), 0x7C00)

    def test_cases_separate_single_and_double_rounding(self):
        rec, want, kinds = k_mix_cases.cases(gen_k_mix_probe.SEED, gen_k_mix_probe.LANES)
        kinds = np.array(kinds)
        dbl = k_mix_cases.double_rounding_bits(rec)
        for k in k_mix_cases.KINDS:
            self.assertGreater(int((kinds == k).sum()), 40, k)
        for k in ("mid+", "mid-", "sub+", "sub-"):
            self.assertGreater(int(((dbl != want) & (kinds == k)).sum()), 10, k)
        self.assertEqual(int(((dbl != want) & np.isin(kinds, ("tie", "plain"))).sum()), 0)
        prod = rec[:, 1:9].astype(np.float64) * rec[:, :1].astype(np.float64)
        self.assertTrue(((prod >= 2.0 ** -26) & (prod <= 1.0)).all())
        self.assertTrue((rec[:, 9:] == 0).all())


if __name__ == "__main__":
    unittest.main()
