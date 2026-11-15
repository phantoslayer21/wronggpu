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
import json
import os
import unittest

import numpy as np

from clean.reference import swin as S
from clean.reference.fp8 import fp8_decode, fp8_encode, q8
from clean.runner import variants as V
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


class PartialTileGpu:
    """The merge and expand kernels alone, with a token count that is not a multiple of 16 (PARTIAL: (stage, H, W); the
    low-resolution tokens of a decoder stage are H/2 x W/2, the merged tokens of an encoder stage the same): one 16-token tile of
    a wave is only partly valid. v_wmma needs every lane of the wave executing; a store under `if (token is valid)` let the
    compiler move the matrix chain of the result into that branch, so it ran with only the valid lanes: NaN codes or wrong values
    for valid tokens (and 75 to 112 spilled registers). The kernels clamp the token and store from every lane instead.
    Both runners get the same inputs, so the one launch is one matrix product and a q8 pack: at most one code apart."""

    PARTIAL = ()

    @staticmethod
    def one_kernel(r, stage, w, x, skip, offset, s_map):
        """Output of the merge (encoder: input map s_map, as the block would leave it) or the expand (decoder) kernel alone."""
        plan = r._prepare(stage, w, x, offset, skip)
        try:
            if s_map is not None:
                r.dev.upload(plan["bufs"]["skip"], fp8_encode(s_map).tobytes())
            name = "merged" if s_map is not None else "e"
            launch = plan["launches"][-1 if s_map is not None else 0]
            r.dev.memset(plan["bufs"][name], 0x01, plan["sizes"][name])   # a token the kernel does not write stays visible
            r.dev.launch(r._func(launch[0]), launch[1], launch[2], launch[3])
            r.dev.sync()
            n = plan["sizes"][name]
            return fp8_decode(np.frombuffer(r.dev.download(plan["bufs"][name], n), np.uint8))
        finally:
            r._release(plan)

    def test_partly_valid_tiles_against_the_gfx11_emulator(self):
        emu = self.Runner(TG.build_dir(), "", backend="emu", arch="gfx11")
        for stage, H, W in self.PARTIAL:
            with self.subTest(stage=stage, H=H, W=W):
                rng = np.random.default_rng(110)
                p = S.random_params(stage, rng)
                x, skip = S.random_inputs(stage, H, W, rng)
                s_map = q8(rng.standard_normal(x.shape)).astype(np.float32) if skip is None else None
                offset = S.window_offset(stage, 0)
                got = self.one_kernel(self.runner, stage, self.runner.pack_weights(stage, p), x, skip, offset, s_map)
                want = self.one_kernel(emu, stage, emu.pack_weights(stage, p), x, skip, offset, s_map)
                self.assertEqual(int(np.sum(~np.isfinite(got))), 0)
                u = TK.ulp_distance(got, want)
                print(f"{stage} {H}x{W}: {np.mean(u > 0) * 1e6:.0f} per million one-code flips, {int(np.sum(u > 1))} more than one "
                      f"step (the kernel alone, card vs gfx11 emulator)")
                self.assertLess(float(np.mean(u > 0)), 0.01)
                self.assertEqual(int(np.sum(u > 1)), 0)


class TestEnc1PartialTilesGpu(PartialTileGpu, T1.Enc1Gpu, TK.EmuTest):
    """patch_merge_c64 (the enc1 block shapes are in test_gpu_enc1)."""

    PARTIAL = [("enc1", 12, 20), ("enc1", 36, 28)]


class TestEnc2Gpu(PartialTileGpu, StagesGpu, T1.Enc1Gpu, TK.EmuTest):
    """swin_block_c96 and patch_merge_c96."""

    CASES = cases_of("enc2", extra=[("enc2", 36, 32, (2, 2))])
    PARTIAL = [("enc2", 12, 20), ("enc2", 20, 28)]
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


class TestDec2Gpu(PartialTileGpu, StagesGpu, T1.Enc1Gpu, TK.EmuTest):
    """expand_c128 and swin_block_c96_add."""

    CASES = cases_of("dec2", extra=[("dec2", 36, 32, (2, 0))])
    PARTIAL = [("dec2", 12, 20), ("dec2", 20, 28)]
    TIMED = ("dec2",)
    SIZE_KEY = "dec2"

    def test_stricter_overall_bound_of_the_emulator_tests(self):
        self.check_overall()


class TestDec1Gpu(PartialTileGpu, StagesGpu, T1.Enc1Gpu, TK.EmuTest):
    """expand_c96 and swin_block_c64_add."""

    CASES = [("dec1", 8, 8, (0, 0)), ("dec1", 12, 20, (2, 0)), ("dec1", 16, 20, (2, 0)), ("dec1", 36, 32, (2, 0)),
             ("dec1", 36, 32, (0, 0))]
    PARTIAL = [("dec1", 12, 20), ("dec1", 12, 40), ("dec1", 20, 28), ("dec1", 24, 20), ("dec1", 28, 36)]
    TIMED = ("dec1",)
    SIZE_KEY = "dec1"

    def test_stricter_overall_bound_of_the_emulator_tests(self):
        self.check_overall()


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


class TestVariantsGpu(T1.Enc1Gpu, TK.EmuTest):
    """The gfx11 variants of the build (variants.FAMILY_VARIANTS, built with build.py --variants): the bytes of the default gfx11
    build on the card, for every block shape. They differ in speed only (the emulator says the same, test_kernels_emu_gfx11);
    the F16 operands are exact and the matrix instruction and its K order are the same, so the card's rounding is too."""

    CASES = [("enc1", 20, 24, (2, 2)), ("b1", 12, 16, (0, 0)), ("enc2", 12, 16, (0, 2)), ("dec1", 16, 20, (2, 0)),
             ("dec2", 12, 16, (2, 0)), ("dec1", 36, 32, (2, 0)), ("b0", 36, 32, (2, 2))]

    def built_variants(self):
        manifest = os.path.join(os.environ["WG_BUILD"], V.MANIFEST)
        if not os.path.isfile(manifest):  # built without --variants
            return []
        with open(manifest) as f:
            built = json.load(f)
        return [n for n in V.FAMILY_VARIANTS["gfx11"] if n in built]

    def test_variants_bit_identical_to_base_on_the_card(self):
        names = self.built_variants()
        if not names:
            self.skipTest("the build has no gfx11 variants (build.py --variants)")
        base = self.Runner(os.environ["WG_BUILD"], os.environ.get("WG_ROCM", ""))
        for name in names:
            variant = self.Runner(os.environ["WG_BUILD"], os.environ.get("WG_ROCM", ""), variant=name)
            for i, (stage, H, W, offset) in enumerate(self.CASES):
                with self.subTest(variant=name, stage=stage, H=H, W=W):
                    rng = np.random.default_rng(300 + i)
                    p = S.random_params(stage, rng)
                    x, skip = S.random_inputs(stage, H, W, rng)
                    want = base.run(stage, base.pack_weights(stage, p), x, offset, skip)
                    got = variant.run(stage, base.pack_weights(stage, p), x, offset, skip)
                    for k in want:
                        np.testing.assert_array_equal(got[k], want[k])


if __name__ == "__main__":
    unittest.main()
