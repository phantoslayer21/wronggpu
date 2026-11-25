# SPDX-License-Identifier: MIT
"""HYB_GELU_POLY (Model L wave 3; hlsl/hyb_block.hlsli hyb_gelu_poly8): q8(GELU16(h)) of the MLP by packed FP16 arithmetic,
exact against the 65,536-byte table of the oracle (hyb_tables.gelu16_q8) for EVERY FP16 bit pattern, and the mutant that
shows why the kernels fence both multiplies (the contracted form differs). No GPU."""
import os
import unittest

import numpy as np

from clean.l.reference import hyb_tables as T

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


class GeluPoly(unittest.TestCase):
    def test_equals_the_table_for_every_fp16_input(self):
        got, want = T.gelu_poly_q8(), T.gelu16_q8()
        self.assertEqual(got.shape, (65536,))
        self.assertEqual(int(np.count_nonzero(got != want)), 0)

    def test_special_values(self):
        got = T.gelu_poly_q8()
        for name, bits, byte in (("+inf", 0x7C00, 0x7E), ("-inf", 0xFC00, 0xFF), ("qNaN", 0x7E00, 0xFF),
                                 ("sNaN", 0x7C01, 0xFF), ("-NaN", 0xFE00, 0xFF), ("-0", 0x8000, 0x80), ("+0", 0x0000, 0x00)):
            self.assertEqual(int(got[bits]), byte, name)

    def test_contracted_form_is_caught(self):
        """The mutant: the add of 1/2 contracted with the product (one fma): the comparison must fail (37 inputs)."""
        got, want = T.gelu_poly_q8(fused=True), T.gelu16_q8()
        self.assertGreater(int(np.count_nonzero(got != want)), 0)

    def test_the_kernel_text_fences_both_multiplies(self):
        """hyb_gelu_poly8 passes both multiplies that feed an add through the contraction barrier (and nothing else
        changes the FP16 steps): a source edit that drops one is caught here, the ISA by isa_checks.json."""
        with open(os.path.join(REPO, "clean", "dx12", "hlsl", "hyb_block.hlsli"), encoding="utf-8") as f:
            text = f.read()
        body = text[text.index("v8f hyb_gelu_poly8(uint4 h)"):]
        self.assertEqual(body.count("hyb_opaque_h2("), 2)
        self.assertIn("const h2 p2 = ca - p1b;", body)
        self.assertIn("const h2 p4 = chf + p3b;", body)


if __name__ == "__main__":
    unittest.main()
