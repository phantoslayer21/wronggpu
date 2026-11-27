# SPDX-License-Identifier: GPL-3.0-only
"""The exhaustive facts behind the fast hybrid kernels' element-wise steps (clean/l/reference/hyb_tables.py), each over
every possible FP16 input, plus mutants each check catches."""
import unittest

import numpy as np

from clean.l.reference import hyb_tables as T
from clean.l.reference import oracle as O


class Facts(unittest.TestCase):
    def test_all_facts(self):
        for name, ok in T.facts().items():
            self.assertTrue(ok, name)

    def test_gelu_table_is_the_oracle(self):
        g = T.gelu16_q8()
        self.assertEqual(g.shape, (65536,))
        # spot values: GELU16(2) = 2 (FP8 0x40), GELU16(-2) = -0 or 0, GELU16(0) = 0
        self.assertEqual(int(g[0x4000]), 0x40)
        self.assertIn(int(g[0xC000]), (0x00, 0x80))
        self.assertEqual(int(g[0x0000]), 0x00)


class Mutants(unittest.TestCase):
    """Each exhaustive comparison must fail for a deliberately wrong form."""

    def test_softmax_two_roundings_inside_fma_is_caught(self):
        u = T.clamped_u()
        wrong_t = T.f16(T.f16(u * -u) + O.Q7_POLY_T)          # product rounded to FP16 first: not FMA16
        p = T.f16(u * wrong_t + O.Q7_POLY_P)
        e = ((p.astype(np.float16).view(np.uint16) & 0x03FF) << 5).astype(np.uint16)
        self.assertFalse(np.array_equal(e, T.softmax_e_ref(u)))

    def test_rsq_table_off_by_one_is_caught(self):
        tab = T.rsq16_table().copy()
        tab[0, 512] += 1
        sb = T.ALL16[(T.H >= 2.0 ** -13) & np.isfinite(T.H)]
        ref = T._rsq_ref(sb.view(np.float16).astype(np.float64)).astype(np.float16).view(np.uint16)
        self.assertFalse(np.array_equal(T.rsq16_lookup(sb, tab), ref))

    def test_gelu_fused_mutant_is_caught(self):
        h = T.H[np.isfinite(T.H)]
        y = np.fmin(np.fmax(h, -2.0), 2.0)
        fused = T.f16(h * T.f16(0.5 + T.f16(y * (O.Q7_GELU_A - O.Q7_GELU_B * np.abs(y)))))   # inner mul+sub fused
        self.assertFalse(np.array_equal(T.f16(fused), O.gelu_q7(h)))


if __name__ == "__main__":
    unittest.main()
