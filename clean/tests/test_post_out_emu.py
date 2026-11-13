# SPDX-License-Identifier: MIT
"""Output mode D = O (runtime option recon = output; kernels/post.hip: post_recon2_r_out, post_decode_o16) in the host
lane emulator: post_recon2_r_out is bit-identical to post_recon2_r + post_decode_o16 (reconstruction at D = O, then the
identity output filter) on the cases of test_post_emu (taken with O = D), and the output is the decode of the
reference reconstruction within the FP16 rounding of the display image and of the output.

Run: python -m unittest discover -s clean/tests -t .
"""
import numpy as np

from clean.reference import post as P
from clean.runner import post as PS
from clean.tests.test_kernels_emu import EmuTest
from clean.tests import test_post_emu as PE


class TestOutEmu(EmuTest):
    def run_pair(self, w, x):
        """post_recon2_r + post_decode_o16."""
        a = PS.run_plan(self.runner, PS.prepare_post(self.runner.dev, w, x, "post_recon2_r"))
        out = PS.run_plan(self.runner, PS.prepare_decode(self.runner.dev, {"display": a["display"],
                                                                           "exposure": x["exposure"]}))["out"]
        return a, out

    def test_identical_to_recon_and_identity_filter(self):
        maker = PE.TestPostEmu("test_matches_reference")
        maker.runner = self.runner
        checked = 0
        for i, (R, D, xo, special) in enumerate(PE.TestPostEmu.CASES):
            x, s = maker.make_case(i, R, D, xo, special)
            rh, rw = x["color"].shape[:2]
            dw, dh = x["recon_size"]
            if not PS.recon2_supported(rw, rh, dw, dh):
                continue
            w = self.runner.pack_weights("post", {"s": np.array(s)})
            with self.subTest(case=i, special=special, D=(dw, dh)):
                a, out = self.run_pair(w, x)
                g = PS.run_plan(self.runner, PS.prepare_post(self.runner.dev, w, x, PS.KERNEL_RECON_OUT))
                np.testing.assert_array_equal(g["history"], a["history"])
                np.testing.assert_array_equal(g["reduced"], a["reduced"])
                np.testing.assert_array_equal(g["out"], out)
                self.assertTrue(np.all(np.isfinite(g["out"])))
                checked += 1
        self.assertGreaterEqual(checked, 8)

    def test_decode_matches_reference(self):
        """post_decode_o16 = FP16(Dec(clamp(E, 1e-4, 3)) * 0.01 / e) within one FP16 step (v_exp is not correctly
        rounded), alpha 1."""
        rng = np.random.default_rng(12)
        disp = rng.uniform(0.0, 2.45, (17, 29, 3)).astype(np.float16).astype(np.float32)
        disp[0, :5] = np.array([0.0, 1e-4, 0.03, 0.0547, 2.45], np.float16).astype(np.float32)[:, None]  # edge values
        e = 1.7
        got = PS.run_plan(self.runner, PS.prepare_decode(self.runner.dev, {"display": disp, "exposure": e}))["out"]
        q = np.clip(disp.astype(np.float64), 1e-4, 3.0)
        want = ((10.0 ** (q / 0.36) - 1.0) / 100.0 / e).astype(np.float16)
        self.assertLessEqual(int(PE.f16_steps(got, want.astype(np.float32)).max()), 1)

    def test_against_reference(self):
        """Against the float64 reference reconstruction at D = O, decoded: spec 09 float32 rule (1e-3 relative for
        99.9 % of the values, 1e-2 for all). The kernel's display value is rounded to FP16 before the decode (as the
        filter reads it); measured 99.9 % quantile about 5e-4, maximum about 3.5e-3."""
        for i, (R, D) in enumerate([((40, 24), (60, 36)), ((26, 15), (39, 23)), ((70, 40), (105, 60))]):
            rng = np.random.default_rng(810 + i)
            x = P.random_inputs(rng, R[0], R[1], dw=D[0], dh=D[1])
            s = float(rng.uniform(0.2, 2.0))
            ref = P.reconstruct(x, s)
            e = P.exposure_or_one(x["exposure"])
            want = (10.0 ** (np.clip(ref["display"], 1e-4, 3.0) / 0.36) - 1.0) / 100.0 / e
            g = PS.run_plan(self.runner, PS.prepare_post(self.runner.dev, self.runner.pack_weights(
                "post", {"s": np.array(s)}), x, PS.KERNEL_RECON_OUT))
            with self.subTest(case=i):
                m = PE.spec09_fp16(g["history"], ref["history"])
                self.assertLessEqual(m["frac_over_1"], 1e-4, m)
                rel = np.abs(g["out"].astype(np.float64) - want) / np.maximum(np.abs(want), 1e-6)
                self.assertTrue(np.all(np.isfinite(g["out"])))
                self.assertLessEqual(float(np.mean(rel > 1e-3)), 1e-3)   # spec 09 float32 rule
                self.assertLessEqual(float(rel.max()), 1e-2)
