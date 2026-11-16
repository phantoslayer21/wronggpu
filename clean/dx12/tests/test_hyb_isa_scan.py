# SPDX-License-Identifier: MIT
"""The hybrid kernels' ISA rules (clean/dx12/hyb_isa_scan.py) on constructed gfx12 ISA: each rule finds what it must
(the negative controls: a square fused through a copied register, an FP8 conversion of a float32 value whose RN16 was
removed) and passes what it must not flag. No GPU, no RGA.
Run: python clean/dx12/tests/test_hyb_isa_scan.py
"""
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import hyb_isa_scan as S  # noqa: E402


def isa(*lines):
    return "\n".join("\t" + x if not x.endswith(":") else x for x in lines) + "\n\ts_endpgm\n"


class FusedSquares(unittest.TestCase):
    def found(self, *lines):
        return len(S.fused_squares(isa(*lines)))

    def test_square(self):
        self.assertEqual(self.found("v_pk_fma_f16 v1, v4, v4, v1"), 1)
        self.assertEqual(self.found("v_fmac_f16_e32 v1, -v4, v4"), 1)
        self.assertEqual(self.found("v_fma_f32 v1, v4, v4, v2"), 1)
        self.assertEqual(self.found("v_pk_fma_f16 v1, v4, v6, v1"), 0)

    def test_copied_register(self):
        """The negative control the review asked for: x copied to a second register, then x * copy fused."""
        self.assertEqual(self.found("v_mov_b32_e32 v5, v4", "v_pk_fma_f16 v1, v4, v5, v1"), 1)
        self.assertEqual(self.found("v_mov_b32_e32 v5, v4", "v_mov_b32_e32 v6, v5", "v_pk_fma_f16 v1, v6, v4, v1"), 1)
        self.assertEqual(self.found("v_mov_b16_e32 v5.l, v4.h", "v_fma_f16 v1.l, v4.h, v5.l, v1.l"), 1)
        # the copy overwritten, or its source overwritten: not one value any more
        self.assertEqual(self.found("v_mov_b32_e32 v5, v4", "v_add_f32_e32 v5, v6, v7", "v_pk_fma_f16 v1, v4, v5, v1"), 0)
        self.assertEqual(self.found("v_mov_b32_e32 v5, v4", "v_add_f32_e32 v4, v6, v7", "v_pk_fma_f16 v1, v4, v5, v1"), 0)
        # a lane move (DPP) is not a copy of the lane's value
        self.assertEqual(self.found("v_mov_b32_dpp v5, v4 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf",
                                    "v_pk_fma_f16 v1, v4, v5, v1"), 0)

    def test_copies_through_blocks(self):
        both = isa("s_cbranch_scc0 L1", "v_mov_b32_e32 v5, v4", "s_branch L2", "L1:", "v_mov_b32_e32 v5, v4", "L2:",
                   "v_pk_fma_f16 v1, v4, v5, v1")
        one = isa("s_cbranch_scc0 L1", "v_mov_b32_e32 v5, v4", "L1:", "v_pk_fma_f16 v1, v4, v5, v1")
        self.assertEqual(len(S.fused_squares(both)), 1)
        self.assertEqual(len(S.fused_squares(one)), 0)

    def test_dual_issue(self):
        self.assertEqual(self.found("v_dual_mov_b32 v5, v4 :: v_dual_fmac_f32 v1, v6, v6"), 1)
        self.assertEqual(self.found("v_mov_b32_e32 v5, v4", "v_dual_add_f32 v8, v2, v3 :: v_dual_fmac_f32 v1, v4, v5"), 1)

    def test_not_flagged(self):
        self.assertEqual(self.found("v_pk_fma_f16 v1, v4, v4, v1 op_sel:[0,1,0] op_sel_hi:[1,1,1]"), 0)  # two halves
        self.assertEqual(self.found("v_fma_f32 v1, -v4, v4, 0x3f6d6000"), 0)   # float32, literal addend (softmax)
        self.assertEqual(self.found("v_fma_f32 v1, -v4, v4, s3"), 0)
        self.assertEqual(self.found("v_fmaak_f32 v1, v4, v4, 0x3f800000"), 0)


class Fp8Sources(unittest.TestCase):
    def bad(self, *lines):
        return len(S.fp8_sources_not_fp16(isa(*lines)))

    WIDEN = ("v_cvt_f16_f32_e32 v2.l, v1", "v_cvt_f16_f32_e32 v2.h, v9", "v_cvt_f32_f16_e32 v3, v2.l",
             "v_cvt_f32_f16_e32 v4, v2.h")

    def test_fp16_sources(self):
        self.assertEqual(self.bad(*self.WIDEN, "v_cvt_pk_fp8_f32 v5.l, v3, v4"), 0)
        # the saturating pack's steps on FP16 values: clamp, NaN quieting, select
        self.assertEqual(self.bad(*self.WIDEN, "v_maxmin_num_f32 v6, s6, v3, 0x43e00000", "v_or_b32_e32 v7, 0x400000, v3",
                                  "v_cndmask_b32_e64 v8, v6, v7, s4", "v_max3_num_f32 v9, 0, v4, 0xc3e00000",
                                  "v_cvt_pk_fp8_f32 v5.h, v8, v9 op_sel:[0,0,1]"), 0)
        self.assertEqual(self.bad("v_mov_b32_e32 v3, 0", "v_cvt_pk_fp8_f32 v5.l, v3, 0"), 0)   # constants

    def test_removed_widening(self):
        """The negative control the review asked for: the RN16 removed, the FP8 pack takes the float32 value."""
        self.assertEqual(self.bad("v_fma_f32 v1, v2, v3, v4", "v_cvt_pk_fp8_f32 v5.l, v1, v1"), 1)
        self.assertEqual(self.bad("v_fma_f32 v1, v2, v3, v4", "v_maxmin_num_f32 v6, s6, v1, 0x43e00000",
                                  "v_cvt_pk_fp8_f32 v5.l, v6, v6"), 1)
        self.assertEqual(self.bad(*self.WIDEN, "v_or_b32_e32 v7, 0x1, v3", "v_cvt_pk_fp8_f32 v5.l, v7, v4"), 1)
        # one source FP16, the other not
        self.assertEqual(self.bad(*self.WIDEN, "v_add_f32_e32 v6, v3, v4", "v_cvt_pk_fp8_f32 v5.l, v3, v6"), 1)
        # a register range written over the widened value (a WMMA result), an op the rule does not know
        self.assertEqual(self.bad(*self.WIDEN, "v_wmma_f32_16x16x16_fp8_fp8 v[0:7], v[8:9], v[10:11], v[0:7]",
                                  "v_cvt_pk_fp8_f32 v5.l, v3, v4"), 1)
        self.assertEqual(self.bad(*self.WIDEN, "v_permlanex16_b32 v6, v3, s0, s1", "v_cvt_pk_fp8_f32 v5.l, v6, v4"), 1)
        # loads are not FP16 values
        self.assertEqual(self.bad("global_load_b32 v3, v1, s[0:1]", "v_cvt_pk_fp8_f32 v5.l, v3, v3"), 1)

    def test_paths(self):
        both = isa("s_cbranch_scc0 L1", *self.WIDEN, "s_branch L2", "L1:", *self.WIDEN, "L2:",
                   "v_cvt_pk_fp8_f32 v5.l, v3, v4")
        one = isa("s_cbranch_scc0 L1", *self.WIDEN, "L1:", "v_cvt_pk_fp8_f32 v5.l, v3, v4")
        loop = isa(*self.WIDEN, "L0:", "v_cvt_pk_fp8_f32 v5.l, v3, v4", "v_add_f32_e32 v3, v3, v3", "s_cbranch_scc0 L0")
        self.assertEqual(len(S.fp8_sources_not_fp16(both)), 0)
        self.assertEqual(len(S.fp8_sources_not_fp16(one)), 1)
        self.assertEqual(len(S.fp8_sources_not_fp16(loop)), 1)   # the back edge brings a float32 v3


if __name__ == "__main__":
    unittest.main()
