# SPDX-License-Identifier: MIT
"""post_fused3* (fused reconstruction + output filter scheduled for latency, kernels/post.hip) in the host lane
emulator: the exact variants are bit-identical to the two-kernel path (post_recon + post_filter; the _o16 output is its
FP16 rounding) on the cases of test_post_emu; the approximations give finite outputs on every case and stay above
single-frame PSNR floors against the exact path; the compact 4-byte history round-trips through the host codec.

Run: python -m unittest discover -s clean/tests -t .
"""
import numpy as np

from clean.reference import post as P
from clean.runner import post as PS
from clean.tests.test_kernels_emu import EmuTest
from clean.tests import test_post_emu as PE

# Single-frame PSNR floors (dB) of the output against the exact two-kernel path (synthetic data, random history).
FLOORS = {"post_fused3_fast_h6_o16": 45, "post_fused3_h4_o16": 50, "post_fused3_fast_h4_o16": 45}


def psnr(a, b, peak):
    m = float(np.mean((np.asarray(a, np.float64) - b) ** 2))
    return 99.0 if m == 0 else 10 * np.log10(peak * peak / m)


class TestFused3Emu(EmuTest):
    def two_kernel(self, w, x):
        a = self.runner.run("post", w, x)
        O = PS.output_size_for(*x["recon_size"])
        f = self.runner.run("filter", b"", {"display": a["display"], "exposure": x["exposure"], "output_size": O})
        return a, f["out"]

    def cases(self):
        maker = PE.TestPostEmu("test_matches_reference")
        maker.runner = self.runner
        for i, (R, D, xo, special) in enumerate(PE.TestPostEmu.CASES):
            x, s = maker.make_case(i, R, D, xo, special)
            dw, dh = x["recon_size"]
            try:
                PS.output_size_for(dw, dh)
            except ValueError:
                continue
            rh, rw = x["color"].shape[:2]
            if not PS.fused_supported(rw, rh, dw, dh) or dw < 4:
                continue
            yield i, special, x, s

    def test_exact_variants(self):
        checked = 0
        for i, special, x, s in self.cases():
            w = self.runner.pack_weights("post", {"s": np.array(s)})
            a, out = self.two_kernel(w, x)
            for k, (_, o16, exact) in PS.FUSED3_VARIANTS.items():
                if exact is not True:
                    continue
                with self.subTest(case=i, special=special, kernel=k):
                    g = PS.run_fused(self.runner, w, x, k)
                    np.testing.assert_array_equal(g["history"], a["history"])
                    np.testing.assert_array_equal(g["reduced"], a["reduced"])
                    np.testing.assert_array_equal(g["out"], out.astype(np.float16).astype(np.float32) if o16 else out)
                    checked += 1
        self.assertGreaterEqual(checked, 2 * 8)

    def test_approximations_finite(self):
        for i, special, x, s in self.cases():
            w = self.runner.pack_weights("post", {"s": np.array(s)})
            for k, (_, _, exact) in PS.FUSED3_VARIANTS.items():
                if exact is True:
                    continue
                with self.subTest(case=i, special=special, kernel=k):
                    g = PS.run_fused(self.runner, w, x, k)
                    for n, v in g.items():
                        self.assertTrue(np.all(np.isfinite(v)), n)

    def test_psnr_floor(self):
        for seed in range(2):
            rng = np.random.default_rng(70 + seed)
            x = P.random_inputs(rng, 70, 40, motion_scale=2.0)
            w = self.runner.pack_weights("post", {"s": np.array(0.8)})
            _, out = self.two_kernel(w, x)
            for k, floor in FLOORS.items():
                with self.subTest(seed=seed, kernel=k):
                    g = PS.run_fused(self.runner, w, x, k)
                    self.assertGreaterEqual(psnr(g["out"], out, float(out.max())), floor)

    def test_h4_codec(self):
        """The host encoder matches h4_pack (the kernel's history output re-encodes to the same bytes) and the
        quantization error is at most half a step."""
        rng = np.random.default_rng(3)
        x = P.random_inputs(rng, 40, 24)
        g = PS.run_fused(self.runner, self.runner.pack_weights("post", {"s": np.array(0.7)}), x, "post_fused3_h4_o16")
        h = g["history"]
        np.testing.assert_array_equal(PS.decode_history4(PS.encode_history4(h), h.shape), h)
        v = rng.uniform(-0.5, 3.5, (8, 9, 3)).astype(np.float32)
        d = PS.decode_history4(PS.encode_history4(v), v.shape)
        step = np.array([3 / 2047, 3 / 2047, 3 / 1023], np.float32)
        self.assertTrue(np.all(np.abs(d - np.clip(v, 0, 3)) <= step / 2 + 1e-6))
