# SPDX-License-Identifier: MIT
"""post_recon2* (drop-in replacements of post_recon, kernels/post.hip) in the host lane emulator: bit-identical to
post_recon on every case of test_post_emu (NaN/inf inputs, reset, reprojection on the borders, highlights, ties, odd
and tiny sizes, 3x ratio).

Run: python -m unittest discover -s clean/tests -t .
"""
import numpy as np

from clean.reference import post as P
from clean.runner import post as PS
from clean.tests.test_kernels_emu import EmuTest
from clean.tests import test_post_emu as PE


class TestRecon2Emu(EmuTest):
    def test_identical_to_post_recon(self):
        maker = PE.TestPostEmu("test_matches_reference")
        maker.runner = self.runner
        checked = 0
        for i, (R, D, xo, special) in enumerate(PE.TestPostEmu.CASES):
            x, s = maker.make_case(i, R, D, xo, special)
            rh, rw = x["color"].shape[:2]
            dw, dh = x["recon_size"]
            w = self.runner.pack_weights("post", {"s": np.array(s)})
            base = PS.run_plan(self.runner, PS.prepare_post(self.runner.dev, w, x))
            for k in PS.RECON2_VARIANTS:
                with self.subTest(case=i, R=R, D=(dw, dh), special=special, kernel=k):
                    if not PS.recon2_supported(rw, rh, dw, dh):
                        with self.assertRaises(ValueError):
                            PS.prepare_post(self.runner.dev, w, x, k)
                        continue
                    got = PS.run_plan(self.runner, PS.prepare_post(self.runner.dev, w, x, k))
                    for name in ("history", "reduced", "display"):
                        np.testing.assert_array_equal(got[name], base[name], err_msg=name)
                    checked += 1
        self.assertGreaterEqual(checked, 4 * 10)

    def test_recon3(self):
        """post_recon3 and post_recon5 (pre-pass + reconstruction): bit-identical to post_recon; post_recon3_h (packed-FP16 tap weights): spec 09 FP16 measures
        against the reference, at most one FP16 step from post_recon."""
        maker = PE.TestPostEmu("test_matches_reference")
        maker.runner = self.runner
        for i, (R, D, xo, special) in enumerate(PE.TestPostEmu.CASES):
            x, s = maker.make_case(i, R, D, xo, special)
            rh, rw = x["color"].shape[:2]
            dw, dh = x["recon_size"]
            if not PS.recon2_supported(rw, rh, dw, dh) or dw < 4:
                continue
            w = self.runner.pack_weights("post", {"s": np.array(s)})
            base = PS.run_plan(self.runner, PS.prepare_post(self.runner.dev, w, x))
            ref = P.reconstruct(x, s)
            for k, exact in {**PS.RECON3_VARIANTS, "post_recon5": True}.items():
                with self.subTest(case=i, special=special, kernel=k):
                    got = PS.run_plan(self.runner, PS.prepare_post(self.runner.dev, w, x, k))
                    for n in ("history", "reduced", "display"):
                        if exact:
                            np.testing.assert_array_equal(got[n], base[n], err_msg=n)
                        elif n != "reduced":
                            m = PE.spec09_fp16(got[n], ref[n])
                            self.assertEqual(m["nonfinite"], 0)
                            self.assertLessEqual(m["frac_over_1"], 1e-4, (n, m))
                            self.assertLessEqual(int(PE.f16_steps(got[n], base[n]).max()), 1, n)

    def test_recon6(self):
        """post_recon6 (pre-pass + reconstruction with the cheaper head): bit-identical to post_recon wherever
        recon6_supported (R < D per axis); refused (ValueError) otherwise."""
        maker = PE.TestPostEmu("test_matches_reference")
        maker.runner = self.runner
        checked = 0
        for i, (R, D, xo, special) in enumerate(PE.TestPostEmu.CASES):
            x, s = maker.make_case(i, R, D, xo, special)
            rh, rw = x["color"].shape[:2]
            dw, dh = x["recon_size"]
            if not PS.recon2_supported(rw, rh, dw, dh) or dw < 4:
                continue
            w = self.runner.pack_weights("post", {"s": np.array(s)})
            with self.subTest(case=i, special=special):
                if not PS.recon6_supported(rw, rh, dw, dh):
                    with self.assertRaises(ValueError):
                        PS.prepare_post(self.runner.dev, w, x, "post_recon6")
                    continue
                base = PS.run_plan(self.runner, PS.prepare_post(self.runner.dev, w, x))
                got = PS.run_plan(self.runner, PS.prepare_post(self.runner.dev, w, x, "post_recon6"))
                for n in ("history", "reduced", "display"):
                    np.testing.assert_array_equal(got[n], base[n], err_msg=n)
                checked += 1
        self.assertGreaterEqual(checked, 4)

    def test_filter_o16(self):
        """post_filter_c1_s4_o16: the FP16 rounding (nearest even) of post_filter's output, alpha 1, on random, odd and
        edge-heavy display images."""
        rng = np.random.default_rng(31)
        for O in ((40, 24), (37, 19), (130, 9), (1, 1)):
            D = (P.recon_size_for(O[0]), P.recon_size_for(O[1]))
            disp = rng.uniform(0.0, 2.45, (D[1], D[0], 3)).astype(np.float16).astype(np.float32)
            disp[0, :, 1] = 0.0
            x = {"display": disp, "exposure": 1.3, "output_size": O}
            with self.subTest(O=O):
                base = PS.run_plan(self.runner, PS.prepare_filter(self.runner.dev, b"", x))["out"]
                plan = PS.prepare_filter(self.runner.dev, b"", x, "post_filter_c1_s4_o16")
                plan["decoders"]["out"] = lambda raw, shape: np.frombuffer(raw, np.float16).reshape(
                    shape[0], shape[1], 4).astype(np.float32)
                got = PS.run_plan(self.runner, plan)["out"]
                np.testing.assert_array_equal(got[..., :3], base.astype(np.float16).astype(np.float32))
                np.testing.assert_array_equal(got[..., 3], 1.0)
