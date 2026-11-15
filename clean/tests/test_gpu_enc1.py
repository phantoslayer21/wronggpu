# SPDX-License-Identifier: MIT
"""Model M stage enc1 (swin_block_c64, then patch_merge_c64) on a gfx11 (RDNA 3) GPU: the FP16 route, with the matrix
wrappers of kernels/wg_platform.h. Against the float64 reference (spec 09 limits; the emulator tests' stricter limits as
an expected failure, see there), the distance to the gfx11 emulator in one-code flips, and the stage time.

Skipped unless WG_GPU_TEST=1. Needs a build for the card with swin.co:

    python clean/runner/build.py --arch gfx1100 --rocm <ROCm> --out <build> --no-runtime
    WG_GPU_TEST=1 WG_BUILD=<build> WG_ROCM=<ROCm> python -m unittest clean.tests.test_gpu_enc1
"""
import os
import unittest

import numpy as np

from clean.reference import swin as S
from clean.runner import family
from clean.tests import test_kernels_emu as TK
from clean.tests import test_kernels_emu_gfx11 as TG
from clean.tests.test_gpu_stages import GpuCase

# enc1 at the in-game sizes (runner/sweep.py GAME_SIZES: 1440p Quality, render 1706x960; the same stage for a 2560x1440 render)
TIMED_SIZES = {"1706x960": (480, 864), "2560x1440": (720, 1280)}


class Enc1Gpu(GpuCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        if family.build_family(os.environ["WG_BUILD"]) != "gfx11":
            raise unittest.SkipTest("the build is not for a gfx11 target")
        if not os.path.exists(os.path.join(os.environ["WG_BUILD"], "swin.co")):
            raise unittest.SkipTest("the build has no swin.co")
        from clean.runner.stage_runner import Runner
        cls.Runner = Runner


class TestEnc1Gpu(Enc1Gpu, TK.EmuTest):
    """The enc1 cases of the emulator tests (test_kernels_emu.TestStagesEmu.CASES), on the card."""

    CASES = [c for c in TK.TestStagesEmu.CASES if c[0] == "enc1"]

    def cases(self):
        """(stage, H, W, offset, card output, reference output) of every case, as run_case and the reference give them."""
        for i, (stage, H, W, offset) in enumerate(self.CASES):
            got, (p, x, skip) = self.run_case(stage, H, W, offset, seed=100 + i)
            yield stage, H, W, offset, got, S.run_stage(stage, p, x, offset, skip, gelu="sigmoid")

    def test_row_table_of_the_build_is_gfx11(self):
        self.assertEqual(self.runner.d_row, "gfx11")

    def test_spec09_limits_against_the_reference(self):
        """Spec 09 per case and output (skip, merged): no non-finite value, at most 2 % of the values more than one FP8
        step off, mean absolute error at most 1 % of the reference RMS."""
        for stage, H, W, offset, got, ref in self.cases():
            for k in ref:
                with self.subTest(stage=stage, H=H, W=W, output=k):
                    self.assertEqual(got[k].shape, ref[k].shape)
                    m = TK.spec09_metrics(got[k], ref[k])
                    print(f"enc1 {H}x{W} {k}: {m['frac_over_1_step'] * 100:.3f} % more than one step off, mean abs error "
                          f"{m['mean_abs_err'] / m['ref_rms'] * 100:.4f} % of RMS, {m['nonfinite']} non-finite")
                    self.assertEqual(m["nonfinite"], 0)
                    self.assertLessEqual(m["frac_over_1_step"], 0.02, m)
                    self.assertLessEqual(m["mean_abs_err"], 0.01 * m["ref_rms"], m)

    @unittest.expectedFailure
    def test_stricter_limits_of_the_emulator_tests(self):
        """The emulator tests also ask for < 5 % of the values different per case and < 1 % over all cases (the emulator
        gives 0: its sums are exact). On the card the four enc1 cases give 1.008 % over all cases (the cases alone,
        0.2 to 3.4 %), because RDNA 3's F16 matrix instruction loses about a unit in the last place with mixed signs.
        Whether that is within the bar of a gfx11 stage (the maintainer's question 7 of the plan) is open: expected to
        fail until it is decided, and an unexpected success says the number moved."""
        total = differ = 0
        for stage, H, W, offset, got, ref in self.cases():
            for k in ref:
                u = TK.ulp_distance(got[k], ref[k])
                self.assertLess(float(np.mean(u > 0)), 0.05, k)
                total += u.size
                differ += int(np.sum(u > 0))
        print(f"enc1 all cases: {differ / total * 100:.4f} % of the values differ from the reference ({total} values)")
        self.assertLess(differ / total, 0.01)

    def test_the_row_table_matters(self):
        """Sanity of the comparison: weights packed for the gfx12 accumulator rows give another result on this build."""
        a, _ = self.run_case("enc1", 8, 12, (2, 2), seed=9)
        wrong = self.Runner(os.environ["WG_BUILD"], os.environ.get("WG_ROCM", ""), d_row="gfx12")
        b, _ = self.run_case("enc1", 8, 12, (2, 2), seed=9, runner=wrong)
        self.assertGreater(float(np.mean(a["skip"] != b["skip"])), 0.2)

    def test_one_code_flips_against_the_gfx11_emulator(self):
        """The FP16 route on the card is not the emulator's exact arithmetic (RDNA 3's F16 matrix instruction loses about
        one unit in the last place with mixed signs), so some values land on the other side of an FP8 rounding boundary.
        Reported per case and output (the analysis expects about 450 per million on realistic data; these random networks
        amplify differences: 2 000 to 34 000 per million here, 1.008 % over all cases, the same number as against the
        reference because the emulator equals the reference). Bounded per case and output at 5 %, the emulator tests'
        per-case limit; the 1 % over all cases is test_stricter_limits_of_the_emulator_tests."""
        emu = self.Runner(TG.build_dir(), "", backend="emu", arch="gfx11")
        total = flips = over = 0
        for i, (stage, H, W, offset) in enumerate(self.CASES):
            with self.subTest(stage=stage, H=H, W=W, offset=offset):
                got, (p, x, skip) = self.run_case(stage, H, W, offset, seed=100 + i)
                want = emu.run(stage, emu.pack_weights(stage, p), x, offset, skip)
                self.assertEqual(set(want), set(got))
                for k in want:
                    u = TK.ulp_distance(got[k], want[k])
                    total += u.size
                    flips += int(np.sum(u > 0))
                    over += int(np.sum(u > 1))
                    print(f"enc1 {H}x{W} {k}: {np.mean(u > 0) * 1e6:.0f} per million one-code flips, "
                          f"{np.mean(u > 1) * 1e6:.0f} per million more than one step (card vs gfx11 emulator)")
                    self.assertLess(float(np.mean(u > 0)), 0.05, k)
        print(f"enc1 all cases: {flips / total * 1e6:.0f} per million flips, {over / total * 1e6:.0f} per million "
              f"more than one step, {total} values")

    def test_stage_time(self):
        """Median of 50 launches at the in-game sizes (inputs already on the card); printed, and finite."""
        r = self.runner
        for name, (H, W) in TIMED_SIZES.items():
            with self.subTest(size=name):
                rng = np.random.default_rng(5)
                p = S.random_params("enc1", rng)
                x, skip = S.random_inputs("enc1", H, W, rng)
                ms = r.time("enc1", r.pack_weights("enc1", p), x, S.window_offset("enc1", 0), skip, repeats=50)
                print(f"enc1 {name} render ({H}x{W} tokens): median of 50 launches {ms:.3f} ms on {r.dev.device_name}")
                self.assertTrue(0.0 < ms < 1000.0)


if __name__ == "__main__":
    unittest.main()
