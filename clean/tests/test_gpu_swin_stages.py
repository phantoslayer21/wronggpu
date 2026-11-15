# SPDX-License-Identifier: MIT
"""The other Swin block shapes of Model M on a gfx11 (RDNA 3) GPU, the FP16 route, as test_gpu_enc1 does for enc1: c96 (enc2,
dec2), c128 (b0 to b5), the `_add` form (dec1, dec2) and the weight-resident `_r` kernels (enc1, dec1). Each against the float64
reference (clean/reference/swin.py) with the acceptance of test_gpu_enc1: spec 09 per case and output, no more than 5 % of the
values different per case; the stricter "< 1 % over all cases" of the emulator tests is an expected failure where it misses
(the output standard of an RDNA 3 build is the maintainer's open decision, plan question 7). Printed: the numbers, the distance
to the gfx11 emulator in one-code flips, the stage time at the in-game sizes.

Skipped unless WG_GPU_TEST=1. Needs a build for the card with swin.co:

    python clean/runner/build.py --arch gfx1100 --rocm <ROCm> --out <build> --no-runtime
    WG_GPU_TEST=1 WG_BUILD=<build> WG_ROCM=<ROCm> python -m unittest clean.tests.test_gpu_swin_stages

WG_VARIANT=<name> runs a variant of the build (build.py --variants).
"""
import os
import unittest

import numpy as np

from clean.reference import swin as S
from clean.tests import test_gpu_enc1 as T1
from clean.tests import test_kernels_emu as TK
from clean.tests import test_kernels_emu_gfx11 as TG

# (rows, columns) in tokens of each stage at the render sizes 1706x960 and 2560x1440 (runner/gfx11_decision.py RENDER_SIZES):
# the block of enc1 and dec1 runs at half the render size, enc2 and dec2 at a quarter, the bottleneck at an eighth
SIZES = {"enc2": {"1706x960": (240, 432), "2560x1440": (360, 640)},
         "bneck": {"1706x960": (120, 216), "2560x1440": (180, 320)},
         "dec2": {"1706x960": (240, 432), "2560x1440": (360, 640)},
         "dec1": {"1706x960": (480, 864), "2560x1440": (720, 1280)}}
CASES = TK.TestStagesEmu.CASES


def cases_of(*stages, extra=()):
    return [c for c in CASES if c[0] in stages] + list(extra)


class StagesGpu:
    """Mixin for a group of stages: CASES (stage, H, W, offset), the stages timed (TIMED) and the key of SIZES for their sizes."""

    CASES = ()
    TIMED = ()
    ROWS = "gfx11"
    OVERALL_MISS = False   # set where the "< 1 % over all cases" bound misses on the card (see check_overall)

    def cases(self):
        """(stage, H, W, offset, card output, reference output) of every case."""
        for i, (stage, H, W, offset) in enumerate(self.CASES):
            got, (p, x, skip) = self.run_case(stage, H, W, offset, seed=100 + i)
            yield stage, H, W, offset, got, S.run_stage(stage, p, x, offset, skip, gelu="sigmoid")

    def test_row_table_of_the_build_is_gfx11(self):
        self.assertEqual(self.runner.d_row, self.ROWS)

    def test_spec09_limits_and_the_per_case_bound_against_the_reference(self):
        """Spec 09 per case and output (no non-finite value, at most 2 % of the values more than one FP8 step off, mean absolute
        error at most 1 % of the reference RMS), and fewer than 5 % of the values different per case and output."""
        for stage, H, W, offset, got, ref in self.cases():
            for k in ref:
                with self.subTest(stage=stage, H=H, W=W, output=k):
                    self.assertEqual(got[k].shape, ref[k].shape)
                    m = TK.spec09_metrics(got[k], ref[k])
                    differ = float(np.mean(TK.ulp_distance(got[k], ref[k]) > 0))
                    print(f"{stage} {H}x{W} {k}: {m['frac_over_1_step'] * 100:.3f} % more than one step off, mean abs error "
                          f"{m['mean_abs_err'] / m['ref_rms'] * 100:.4f} % of RMS, {differ * 100:.3f} % different, "
                          f"{m['nonfinite']} non-finite")
                    self.assertEqual(m["nonfinite"], 0)
                    self.assertLessEqual(m["frac_over_1_step"], 0.02, m)
                    self.assertLessEqual(m["mean_abs_err"], 0.01 * m["ref_rms"], m)
                    self.assertLess(differ, 0.05)

    def check_overall(self):
        """The emulator tests' "< 1 % of the values different over all cases" (the emulator gives 0: its sums are exact)."""
        total = differ = 0
        for stage, H, W, offset, got, ref in self.cases():
            for k in ref:
                u = TK.ulp_distance(got[k], ref[k])
                total += u.size
                differ += int(np.sum(u > 0))
        print(f"{self.CASES[0][0]}.. all cases: {differ / total * 100:.4f} % of the values differ from the reference "
              f"({total} values)")
        self.assertLess(differ / total, 0.01)

    def test_the_row_table_matters(self):
        """Sanity of the comparison: weights packed for the gfx12 accumulator rows give another result on this build."""
        stage, H, W, offset = self.CASES[0]
        a, _ = self.run_case(stage, H, W, offset, seed=9)
        wrong = self.Runner(os.environ["WG_BUILD"], os.environ.get("WG_ROCM", ""), variant=self.variant, d_row="gfx12")
        b, _ = self.run_case(stage, H, W, offset, seed=9, runner=wrong)
        k = sorted(a)[-1]
        self.assertGreater(float(np.mean(a[k] != b[k])), 0.2)

    def test_one_code_flips_against_the_gfx11_emulator(self):
        """Reported per case and output (the card's F16 matrix instruction is not the emulator's exact sums, see
        test_gpu_enc1), bounded per case at 5 %."""
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
                    print(f"{stage} {H}x{W} {k}: {np.mean(u > 0) * 1e6:.0f} per million one-code flips, "
                          f"{np.mean(u > 1) * 1e6:.0f} per million more than one step (card vs gfx11 emulator)")
                    self.assertLess(float(np.mean(u > 0)), 0.05, k)
        print(f"{self.CASES[0][0]}.. all cases: {flips / total * 1e6:.0f} per million flips, {over / total * 1e6:.0f} per "
              f"million more than one step, {total} values")

    def test_stage_time(self):
        """Median of 50 launches at the in-game sizes (inputs already on the card); printed, and finite."""
        r = self.runner
        for stage in self.TIMED:
            for name, (H, W) in SIZES[self.SIZE_KEY].items():
                with self.subTest(stage=stage, size=name):
                    rng = np.random.default_rng(5)
                    p = S.random_params(stage, rng)
                    x, skip = S.random_inputs(stage, H, W, rng)
                    ms = r.time(stage, r.pack_weights(stage, p), x, S.window_offset(stage, 0), skip, repeats=50)
                    print(f"{stage} {name} render ({H}x{W} tokens): median of 50 launches {ms:.3f} ms on {r.dev.device_name}")
                    self.assertTrue(0.0 < ms < 1000.0)


class TestEnc2Gpu(StagesGpu, T1.Enc1Gpu, TK.EmuTest):
    """swin_block_c96 and patch_merge_c96."""

    CASES = cases_of("enc2", extra=[("enc2", 36, 32, (2, 2))])
    TIMED = ("enc2",)
    SIZE_KEY = "enc2"

    def test_stricter_overall_bound_of_the_emulator_tests(self):
        self.check_overall()


class TestBottleneckGpu(StagesGpu, T1.Enc1Gpu, TK.EmuTest):
    """swin_block_c128: all six blocks (b0, b2, b4 with the offsets (2, 2) and (0, 2), b1, b3, b5 with (0, 0) and (2, 0))."""

    CASES = cases_of("b0", "b1", "b3", extra=[("b2", 12, 8, (0, 2)), ("b4", 8, 12, (2, 2)), ("b5", 12, 12, (0, 0)),
                                              ("b0", 36, 32, (2, 2)), ("b1", 36, 32, (2, 0))])
    TIMED = ("b0", "b1")
    SIZE_KEY = "bneck"

    def test_stricter_overall_bound_of_the_emulator_tests(self):
        self.check_overall()


class TestDec2Gpu(StagesGpu, T1.Enc1Gpu, TK.EmuTest):
    """expand_c128 and swin_block_c96_add."""

    CASES = cases_of("dec2", extra=[("dec2", 36, 32, (2, 0))])
    TIMED = ("dec2",)
    SIZE_KEY = "dec2"

    def test_stricter_overall_bound_of_the_emulator_tests(self):
        self.check_overall()


class TestDec1Gpu(StagesGpu, T1.Enc1Gpu, TK.EmuTest):
    """expand_c96 and swin_block_c64_add. The case ("dec1", 12, 20, (2, 0)) of the emulator tests is not here: see
    test_expand_c96_with_a_partly_valid_second_tile."""

    CASES = [("dec1", 8, 8, (0, 0)), ("dec1", 16, 20, (2, 0)), ("dec1", 36, 32, (2, 0)), ("dec1", 36, 32, (0, 0))]
    TIMED = ("dec1",)
    SIZE_KEY = "dec1"

    def test_stricter_overall_bound_of_the_emulator_tests(self):
        self.check_overall()

    @unittest.expectedFailure
    def test_expand_c96_with_a_partly_valid_second_tile(self):
        """Finding, not a block-kernel result: expand_c96 (107 spilled registers in the gfx11 builds, slice E2) writes NaN codes
        for valid tokens when a wave's second 16-token tile is only partly valid (60, 84 or 120 low-resolution tokens; 36, 40,
        100 and 288 are fine: no partial second tile). The emulator tests' case dec1 12x20 gives 5120 non-finite values here
        and none in either emulator. Expected to fail until expand_c96 is fixed; an unexpected success says it is."""
        stage, H, W, offset = "dec1", 12, 20, (2, 0)
        got, (p, x, skip) = self.run_case(stage, H, W, offset, seed=110)
        self.assertEqual(int(np.sum(~np.isfinite(got["out"]))), 0)


class TestResidentGpu(T1.Enc1Gpu, TK.EmuTest):
    """The weight-resident persistent kernels swin_block_c64_r and swin_block_c64_add_r (Runner.resident = N workgroups): the
    results of the plain kernels bit for bit (the kernels' own contract), and the reference limits of test_gpu_enc1."""

    CASES = [("enc1", 20, 24, (2, 2)), ("enc1", 36, 32, (2, 2)), ("enc1", 36, 32, (0, 2)),
             ("dec1", 16, 20, (2, 0)), ("dec1", 36, 32, (2, 0)), ("dec1", 36, 32, (0, 0))]
    WORKGROUPS = (1, 2, 5)   # fewer workgroups than windows: each wave takes several windows

    def resident_runner(self, n):
        r = self.Runner(os.environ["WG_BUILD"], os.environ.get("WG_ROCM", ""), variant=self.variant)
        r.resident = n
        return r

    def test_the_resident_kernel_is_launched(self):
        for stage, kernel in (("enc1", "swin_block_c64_r"), ("dec1", "swin_block_c64_add_r")):
            with self.subTest(stage=stage):
                rng = np.random.default_rng(1)
                p = S.random_params(stage, rng)
                x, skip = S.random_inputs(stage, 20, 24, rng)
                r = self.resident_runner(2)
                plan = r._prepare(stage, r.pack_weights(stage, p), x, S.window_offset(stage, 0), skip)
                try:
                    names = [launch[0] for launch in plan["launches"]]
                    self.assertTrue(any(n.startswith(kernel) for n in names), names)
                finally:
                    r._release(plan)
                plain = self.runner
                plan = plain._prepare(stage, plain.pack_weights(stage, p), x, S.window_offset(stage, 0), skip)
                try:
                    self.assertFalse(any(n.startswith(kernel) for n in [launch[0] for launch in plan["launches"]]))
                finally:
                    plain._release(plan)

    def test_bit_identical_to_the_plain_kernels_and_within_the_reference_limits(self):
        for i, (stage, H, W, offset) in enumerate(self.CASES):
            plain, (p, x, skip) = self.run_case(stage, H, W, offset, seed=400 + i)
            ref = S.run_stage(stage, p, x, offset, skip, gelu="sigmoid")
            for n in self.WORKGROUPS:
                with self.subTest(stage=stage, H=H, W=W, offset=offset, workgroups=n):
                    r = self.resident_runner(n)
                    got = r.run(stage, r.pack_weights(stage, p), x, offset, skip)
                    for k in plain:
                        np.testing.assert_array_equal(got[k], plain[k])
                    for k in ref:
                        m = TK.spec09_metrics(got[k], ref[k])
                        differ = float(np.mean(TK.ulp_distance(got[k], ref[k]) > 0))
                        if n == self.WORKGROUPS[0]:
                            print(f"{stage}_r {H}x{W} {k}: {m['frac_over_1_step'] * 100:.3f} % more than one step off, mean abs "
                                  f"error {m['mean_abs_err'] / m['ref_rms'] * 100:.4f} % of RMS, {differ * 100:.3f} % different")
                        self.assertEqual(m["nonfinite"], 0)
                        self.assertLessEqual(m["frac_over_1_step"], 0.02, m)
                        self.assertLessEqual(m["mean_abs_err"], 0.01 * m["ref_rms"], m)
                        self.assertLess(differ, 0.05)

    def test_stage_time(self):
        """Median of 50 launches of the plain and the resident kernels (84 workgroups, about 2 per WGP) at the in-game sizes."""
        for stage, sizes in (("enc1", T1.TIMED_SIZES), ("dec1", SIZES["dec1"])):
            for name, (H, W) in sizes.items():
                rng = np.random.default_rng(5)
                p = S.random_params(stage, rng)
                x, skip = S.random_inputs(stage, H, W, rng)
                ms = {}
                for label, r in (("plain", self.runner), ("resident", self.resident_runner(84))):
                    ms[label] = r.time(stage, r.pack_weights(stage, p), x, S.window_offset(stage, 0), skip, repeats=50)
                print(f"{stage} {name} render ({H}x{W} tokens): plain {ms['plain']:.3f} ms, resident {ms['resident']:.3f} ms")
                self.assertTrue(all(0.0 < v < 1000.0 for v in ms.values()))


if __name__ == "__main__":
    unittest.main()
