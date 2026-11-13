# SPDX-License-Identifier: MIT
"""post_recon4_* (SPEED APPROXIMATIONS of post_recon, kernels/post.hip) in the host lane emulator: finite outputs on
the post test cases (NaN/inf inputs included) and a single-frame PSNR floor against the exact path on synthetic data
(random history: harder for the bilinear variants than real images). The whole-pipeline rule of spec 09 draft 2 is
checked by the spec team on recorded sequences.

Run: python -m unittest discover -s clean/tests -t .
"""
import numpy as np

from clean.reference import post as P
from clean.runner import post as PS
from clean.tests import test_post_emu as PE
from clean.tests.test_kernels_emu import EmuTest

# Single-frame PSNR floors (dB) of the display image against post_recon on the synthetic cases below.
FLOORS = {"post_recon4_bil": 50, "post_recon4_crbil": 55, "post_recon4_skip16": 75, "post_recon4_skip4": 58,
          "post_recon4_f16": 75, "post_recon4_f16x": 68, "post_recon4_fast": 55, "post_recon4_fastest": 50,
          "post_recon5_bil": 50, "post_recon6_bil": 50, "post_recon6_eb": 46, "post_recon6_eb_bil": 46}


def psnr(a, b, peak):
    m = float(np.mean((np.asarray(a, np.float64) - b) ** 2))
    return 99.0 if m == 0 else 10 * np.log10(peak * peak / m)


class TestRecon4Emu(EmuTest):
    def test_finite_on_edge_cases(self):
        maker = PE.TestPostEmu("test_matches_reference")
        maker.runner = self.runner
        for i, (R, D, xo, special) in enumerate(PE.TestPostEmu.CASES):
            x, s = maker.make_case(i, R, D, xo, special)
            rh, rw = x["color"].shape[:2]
            dw, dh = x["recon_size"]
            if not PS.recon2_supported(rw, rh, dw, dh) or dw < 4:
                continue
            w = self.runner.pack_weights("post", {"s": np.array(s)})
            approx6 = [k for k, exact in PS.RECON6_VARIANTS.items() if not exact]
            for k in list(PS.RECON4_VARIANTS) + ["post_recon5_bil"] + approx6:
                if k in approx6 and not PS.recon6_supported(rw, rh, dw, dh):
                    continue
                with self.subTest(case=i, special=special, kernel=k):
                    got = PS.run_plan(self.runner, PS.prepare_post(self.runner.dev, w, x, k))
                    for n, v in got.items():
                        self.assertTrue(np.all(np.isfinite(v)), n)

    def test_psnr_floor(self):
        for seed in range(2):
            rng = np.random.default_rng(60 + seed)
            x = P.random_inputs(rng, 70, 40, motion_scale=2.0)
            w = self.runner.pack_weights("post", {"s": np.array(0.8)})
            base = PS.run_plan(self.runner, PS.prepare_post(self.runner.dev, w, x))
            for k, floor in FLOORS.items():
                with self.subTest(seed=seed, kernel=k):
                    got = PS.run_plan(self.runner, PS.prepare_post(self.runner.dev, w, x, k))
                    self.assertGreaterEqual(psnr(got["display"], base["display"], 2.45), floor)
