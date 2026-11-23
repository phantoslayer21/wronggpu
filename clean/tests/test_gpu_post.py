# SPDX-License-Identifier: MIT
"""The post_recon2 .. post_fused3 reconstruction kernels (kernels/post.hip) on a gfx11 (RDNA 3) GPU.

Their reduced-history step takes the partner column from the other half of the wave with v_permlanex16 (wg_xor16), inside the
`if (active && has1 && wl < 16 && x + 1 < dw)` store. The scan of test_exec_scan sees every such exchange under a lane mask.
It is harmless here, and this test is the evidence: the partner lane is off in that branch (the stores run in lanes 0-15 only,
their partners are lanes 16-31), but v_permlanex16 reads the partner's register all the same, and hsum was summed by every
lane before the branch (selftest probe_xchg16_div shows the same on raw registers). The emulator test classes of the post
kernels run here on the card with the same cases (odd D, partial 32-column tiles, one D row or column, 3x ratio, NaN/inf
inputs), so each kernel is compared bit for bit with post_recon, which has no exchange, and with the float64 reference where
the emulator test does.

Skipped unless WG_GPU_TEST=1. Needs a build for the card (post.co):

    python clean/runner/build.py --arch gfx1100 --rocm <ROCm> --out <build> --no-runtime
    WG_GPU_TEST=1 WG_BUILD=<build> WG_ROCM=<ROCm> python -m unittest clean.tests.test_gpu_post
"""
import os
import unittest

from clean.runner import post as PS
from clean.tests import test_post_band_emu as TB
from clean.tests import test_post_fused3_emu as TF3
from clean.tests import test_post_fused_emu as TF
from clean.tests import test_post_out_emu as TO
from clean.tests import test_post_recon2_emu as TR2
from clean.tests import test_post_recon4_emu as TR4
from clean.tests.test_gpu_stages import GpuCase


LAUNCHED = set()
_function = PS._function


def setUpModule():
    """Note every kernel the classes below launch (post.py looks each one up by name)."""
    def noting(runner, name, code_object=None):
        LAUNCHED.add(name)
        return _function(runner, name, code_object)
    PS._function = noting


def tearDownModule():
    PS._function = _function


class TestRecon2Gpu(GpuCase, TR2.TestRecon2Emu):
    pass


class TestRecon4Gpu(GpuCase, TR4.TestRecon4Emu):
    pass


class TestBandGpu(GpuCase, TB.TestBandEmu):
    pass


class TestOutGpu(GpuCase, TO.TestOutEmu):
    pass


class TestFusedGpu(GpuCase, TF.TestFusedEmu):
    pass


class TestFused3Gpu(GpuCase, TF3.TestFused3Emu):
    pass


# Not launched here: post_recon2_r_out_diet is post_recon2_r_out with other arguments (div_magic, the ratio read from the exposure
# pair) and is only reachable through the C++ runtime's option post_diet=1; the stage runner has no launch plan for it. Its body
# is post_recon2_r_out's, so the exchange is the same code (test_exec_scan: three v_permlanex16 like every other kernel).
NOT_RUN_BY_THE_STAGE_RUNNER = {"post_recon2_r_out_diet"}


class TestZEveryExchangeKernelRan(unittest.TestCase):
    """Every kernel of post.co that holds the reduced-history exchange (test_exec_scan.POST_EXCHANGE_KERNELS) was launched by a
    class above, except those named in NOT_RUN_BY_THE_STAGE_RUNNER. Named to sort last: unittest runs the classes of a module in name order, in one process."""

    def test_every_kernel_with_the_exchange_ran_on_the_card(self):
        if os.environ.get("WG_GPU_TEST") != "1":
            self.skipTest("set WG_GPU_TEST=1")
        from clean.tests.test_exec_scan import POST_EXCHANGE_KERNELS
        if not LAUNCHED:
            self.skipTest("the card classes did not run in this process")
        self.assertEqual(sorted(POST_EXCHANGE_KERNELS - NOT_RUN_BY_THE_STAGE_RUNNER - LAUNCHED), [])


if __name__ == "__main__":
    unittest.main()
