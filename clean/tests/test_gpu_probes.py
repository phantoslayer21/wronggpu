# SPDX-License-Identifier: MIT
"""The matrix wrappers and the half-wave exchange of kernels/wg_platform.h on a gfx11 (RDNA 3) GPU, through
runner/selftest.py: the layout probes (accumulator rows 2v + h), random operands against float64 sums, which lanes feed
which half of the matrix instruction's result, and the exchange in divergent code.

Skipped unless WG_GPU_TEST=1. Needs a build for the card with probe.co:

    python clean/runner/build.py --arch gfx1100 --rocm <ROCm> --out <build> --no-runtime
    WG_GPU_TEST=1 WG_BUILD=<build> WG_ROCM=<ROCm> python -m unittest clean.tests.test_gpu_probes
"""
import os
import unittest

from clean.runner import family


class TestGfx11Probes(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if os.environ.get("WG_GPU_TEST") != "1":
            raise unittest.SkipTest("set WG_GPU_TEST=1 (and WG_BUILD, WG_ROCM) to run on the GPU")
        build = os.environ.get("WG_BUILD")
        if not build:
            raise unittest.SkipTest("set WG_BUILD to a build folder made for the GPU")
        if family.build_family(build) != "gfx11":
            raise unittest.SkipTest("the build is not for a gfx11 target")
        from clean.runner import selftest
        cls.report = selftest.selftest(build, os.environ.get("WG_ROCM", ""))

    def test_the_whole_selftest_passes(self):
        self.assertTrue(self.report["pass"], self.report)

    def test_layout_through_the_wrappers(self):
        for kind in ("wmma_fp8", "wmma_f16"):
            with self.subTest(kind=kind):
                r = self.report[kind]
                self.assertEqual(r["d_row"], [[0, 2, 4, 6, 8, 10, 12, 14], [1, 3, 5, 7, 9, 11, 13, 15]])
                self.assertEqual(r["d_row_name"], "gfx11")
                self.assertTrue(r["a_row_is_lane_mod_16"] and r["b_col_is_lane_mod_16"] and r["k_slots_symmetric"], r)
                self.assertEqual(r["one_times_one"], 1.0)

    def test_arithmetic_is_close_to_the_exact_sum(self):
        for kind in ("wmma_fp8_arithmetic", "wmma_f16_arithmetic"):
            with self.subTest(kind=kind):
                r = self.report[kind]
                self.assertTrue(r["pass"], r)
                self.assertEqual(r["nonfinite"], 0)
                self.assertLessEqual(r["max_units"], 64)

    def test_each_half_of_the_result_reads_its_own_operand_lanes(self):
        """Measured on an RX 7900 XT: the odd rows of D (lanes 16-31) take A and B from lanes 16-31, the even rows from
        lanes 0-15, so lanes 16-31 are not ignored: the wrappers give both halves the same operands."""
        r = self.report["wmma16_lanes"]
        self.assertEqual(r["output_half_reads_operand_halves"], {0: (0, 0), 1: (1, 1)})
        self.assertTrue(r["lanes_16_31_matter"])

    def test_exchange_in_divergent_code(self):
        r = self.report["exchange_divergent"]
        self.assertTrue(r["pass"], r)
        self.assertEqual(len(r["modes"]), 5)


if __name__ == "__main__":
    unittest.main()
