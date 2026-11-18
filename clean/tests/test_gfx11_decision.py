# SPDX-License-Identifier: MIT
"""runner/gfx11_decision.py without a GPU: the mapping from numbers to the verdict word and the spec 09 pass, the
per-frame arithmetic, the resource-report reading and the structure of the markdown sheet (built from invented numbers).
The card and emulator measurements are in test_gpu_enc1.py and the sheet's own run."""
import unittest

import numpy as np

from clean.reference import fp8
from clean.runner import gfx11_decision as D
from clean.runner import variants as V


def codes_to_values(codes):
    return fp8.fp8_decode(codes.astype(np.uint8)).astype(np.float32)


class TestVerdictMapping(unittest.TestCase):
    """equivalence_verdict on FP8 maps with a known share of one-step differences (random sign, fixed seed)."""

    def maps(self, share, step=1, seed=1):
        rng = np.random.default_rng(seed)
        mag = rng.integers(8, 100, size=(64, 64, 8))
        sign = rng.integers(0, 2, size=mag.shape) * 0x80
        ref = (mag | sign).astype(np.int64)
        got = ref.copy()
        hit = rng.random(mag.shape) < share
        move = rng.choice([-step, step], size=mag.shape)
        got[hit] = ((mag + move)[hit]) | sign[hit]
        return codes_to_values(got), codes_to_values(ref)

    def test_identical_is_matches(self):
        got, ref = self.maps(0.0)
        v = D.equivalence_verdict(got, ref)
        self.assertEqual(v["word"], "MATCHES")
        self.assertEqual((v["equal"], v["max_step"], v["over_band"], v["over_16"]), (1.0, 0, 0.0, 0.0))

    def test_half_a_percent_one_step_unbiased_is_close(self):
        got, ref = self.maps(0.005)
        self.assertEqual(D.equivalence_verdict(got, ref)["word"], "CLOSE")

    def test_five_percent_different_is_different(self):
        got, ref = self.maps(0.05)
        v = D.equivalence_verdict(got, ref)
        self.assertEqual(v["word"], "DIFFERENT")
        self.assertIn("under 99 % equal", v["reasons"])
        self.assertEqual(v["size"], "minor")
        self.assertAlmostEqual(v["equal"], 0.95, delta=0.01)
        self.assertEqual(v["max_step"], 1)

    def test_a_three_step_difference_is_different(self):
        got, ref = self.maps(0.001, step=3)
        v = D.equivalence_verdict(got, ref)
        self.assertEqual(v["word"], "DIFFERENT")
        self.assertIn("differences over 2 steps", v["reasons"])
        self.assertEqual(v["max_step"], 3)
        self.assertGreater(v["over_band"], 0.0)

    def test_shape_mismatch_is_error(self):
        got, ref = self.maps(0.0)
        self.assertEqual(D.equivalence_verdict(got[:-1], ref)["word"], "ERROR")

    def test_window_arguments_are_accepted(self):
        got, ref = self.maps(0.005)
        self.assertEqual(D.equivalence_verdict(got, ref, window=(4, 4), shift=(-2, -2))["word"], "CLOSE")


class TestSpec09(unittest.TestCase):
    base = {"nonfinite": 0, "frac_over_1_step": 0.02, "mean_abs_err": 0.01, "ref_rms": 1.0}

    def test_limits_are_inclusive(self):
        self.assertTrue(D.spec09_pass(self.base))

    def test_each_limit_fails_on_its_own(self):
        for k, v in (("nonfinite", 1), ("frac_over_1_step", 0.0201), ("mean_abs_err", 0.0101)):
            with self.subTest(k=k):
                self.assertFalse(D.spec09_pass({**self.base, k: v}))

    def test_mean_error_is_relative_to_the_rms(self):
        self.assertTrue(D.spec09_pass({**self.base, "mean_abs_err": 0.05, "ref_rms": 5.0}))


class TestExtrapolation(unittest.TestCase):
    def test_stage_time_times_launches_per_frame(self):
        f = D.extrapolate({"enc1": 2.0, "b0": 0.5})
        self.assertAlmostEqual(f["measured"], 2.0 * 1 + 0.5 * 6)
        self.assertAlmostEqual(f["crude_all"], f["measured"] + 3 * 2.0)

    def test_launch_counts_are_the_stage_table(self):
        self.assertEqual(D.STAGES_PER_FRAME["b0"], sum(1 for s in D.S.STAGES if s.startswith("b")))
        self.assertEqual(sorted(D.UNMEASURED + tuple(D.STAGES_PER_FRAME) + ("b1", "b2", "b3", "b4", "b5")),
                         sorted(D.S.STAGES))


REPORT = """\
f.hip:1:1: remark: Function Name: swin_block_c64 [-Rpass-analysis=kernel-resource-usage]
f.hip:1:1: remark:     VGPRs: 256 [-Rpass-analysis=kernel-resource-usage]
f.hip:1:1: remark:     ScratchSize [bytes/lane]: 80 [-Rpass-analysis=kernel-resource-usage]
f.hip:1:1: remark:     Occupancy [waves/SIMD]: 5 [-Rpass-analysis=kernel-resource-usage]
f.hip:1:1: remark:     VGPRs Spill: 27 [-Rpass-analysis=kernel-resource-usage]
f.hip:1:1: remark:     LDS Size [bytes/block]: 32768 [-Rpass-analysis=kernel-resource-usage]
f.hip:2:1: remark: Function Name: other [-Rpass-analysis=kernel-resource-usage]
f.hip:2:1: remark:     VGPRs: 9 [-Rpass-analysis=kernel-resource-usage]
"""


class TestResources(unittest.TestCase):
    def test_fields_of_the_named_kernel(self):
        r = D.kernel_resources(REPORT, ["swin_block_c64"])["swin_block_c64"]
        self.assertEqual(r, {"vgprs": 256, "spills": 27, "waves": 5, "lds": 32768, "scratch": 80})

    def test_a_missing_kernel_is_an_error(self):
        with self.assertRaises(KeyError):
            D.kernel_resources(REPORT, ["swin_block_c128"])


def invented(spec09_b0=True, verdict="CLOSE"):
    """A result dict of the shape measure() returns, with invented numbers."""
    def case(ok):
        m = {"nonfinite": 0, "frac_over_1_step": 0.001 if ok else 0.05, "mean_abs_err": 0.001, "ref_rms": 1.0,
             "max_abs_err": 1.0}
        return {"H": 8, "W": 8, "output": "out", "values": 512, "metrics": m, "spec09": ok, "verdict": verdict,
                "reasons": ["differences over 2 steps"], "size": "minor", "equal": 0.995, "max_step": 3,
                "over_band": 0.0002, "over_16": 0.0, "flip_per_million": 1234.0}

    builds = {}
    for n in ["base", *V.FAMILY_VARIANTS["gfx11"]]:
        stages = {}
        for s in D.STAGES:
            ok = spec09_b0 or s != "b0"
            stages[s] = {"resources": {k: {"vgprs": 200, "spills": 3, "waves": 5, "lds": 1024, "scratch": 12}
                                       for k in D.STAGE_KERNELS[s]},
                         "correctness": {"cases": [case(ok)], "spec09": ok, "verdict": verdict, "flip_per_million": 1234.0,
                                         "over1_per_million": 5.0, "values": 512},
                         "time_ms": {z: 1.0 for z in D.RENDER_SIZES}}
        builds[n] = {"stages": stages,
                     "frame_ms": {z: D.extrapolate({s: 1.0 for s in D.STAGES}) for z in D.RENDER_SIZES}}
    return {"device": "Test card", "arch": "gfx1100", "repeats": 50, "builds": builds}


class TestReport(unittest.TestCase):
    def test_every_stage_and_build_has_a_row_in_each_table(self):
        text = D.render_report(invented())
        for s in D.STAGES:
            for n in ["default", *V.FAMILY_VARIANTS["gfx11"]]:
                with self.subTest(stage=s, build=n):
                    self.assertIn(f"| {s} | {n} | 0 |", text)           # correctness table
                    self.assertIn(f"| {s} | {n} | 1.000 | 1.000 |", text)  # time table
                    self.assertIn(f"### {s}, {n}", text)
            for k in D.STAGE_KERNELS[s]:
                self.assertIn(f"| {s} | {k} | default | 200 | 3 |", text)

    def test_sections_and_labels(self):
        text = D.render_report(invented())
        for h in ("## Correctness", "## Resources", "## Time", "## Swin time per frame, extrapolated (INFERRED)",
                  "## Cases", "## Notes"):
            self.assertIn(h, text)
        self.assertIn("applicability INFERRED", text)
        self.assertIn("a different card only", text)
        self.assertIn("0.942", text)
        self.assertIn("maintainer's decision", text)

    def test_extrapolated_numbers_are_in_the_sheet(self):
        text = D.render_report(invented())
        self.assertIn("| default | 7.00 | 10.00 | 7.00 | 10.00 |", text)

    def test_a_failing_spec09_is_recorded_and_named(self):
        text = D.render_report(invented(spec09_b0=False))
        self.assertIn("**Spec 09 is not met on the card by:** b0 (default)", text)
        self.assertIn("**FAIL**", text)
        self.assertNotIn("enc1 (default)", text.split("**Spec 09 is not met on the card by:**")[1].split("\n")[0])

    def test_case_rows_carry_the_verdict_numbers_and_reasons(self):
        text = D.render_report(invented())
        self.assertIn("| 8x8 | out | 512 | 0.100 % | 0.1000 % | 0 | pass | CLOSE (minor) | 99.500 % | 3 | 0.0200 % | 0.0000 % | 1234 |", text)
        self.assertIn("Reasons the tool gives: differences over 2 steps.", text)

    def test_the_sheet_says_the_variants_leave_c128_alone(self):
        self.assertIn("leave the C = 128 kernel unchanged", D.render_report(invented()))

    def test_a_passing_sheet_says_so_and_has_no_failure_line(self):
        text = D.render_report(invented())
        self.assertIn("Spec 09 is met by every listed stage and build", text)
        self.assertNotIn("is not met", text)


if __name__ == "__main__":
    unittest.main()
