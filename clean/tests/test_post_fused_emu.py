# SPDX-License-Identifier: MIT
"""Fused reconstruction + output filter (post_fused*, kernels/post.hip) in the host lane emulator: bit-identical to the
two-kernel path (post_recon + post_filter) on the cases of test_post_emu, and within spec 09 of the reference.

Run: python -m unittest discover -s clean/tests -t .
"""
import numpy as np

from clean.reference import post as P
from clean.runner import post as PS
from clean.tests.test_kernels_emu import EmuTest
from clean.tests import test_post_emu as PE


class TestFusedEmu(EmuTest):
    def two_kernel(self, w, x):
        a = self.runner.run("post", w, x)
        O = PS.output_size_for(*x["recon_size"])
        f = self.runner.run("filter", b"", {"display": a["display"], "exposure": x["exposure"], "output_size": O})
        return a, f["out"]

    def test_identical_to_two_kernel_path(self):
        maker = PE.TestPostEmu("test_matches_reference")
        maker.runner = self.runner
        checked = 0
        for i, (R, D, xo, special) in enumerate(PE.TestPostEmu.CASES):
            x, s = maker.make_case(i, R, D, xo, special)
            dw, dh = x["recon_size"]
            try:
                PS.output_size_for(dw, dh)
            except ValueError:
                continue   # no output size gives this D: not a fused case
            rh, rw = x["color"].shape[:2]
            w = self.runner.pack_weights("post", {"s": np.array(s)})
            if not PS.fused_supported(rw, rh, dw, dh):
                with self.subTest(case=i, special=special, expect="refused"):
                    with self.assertRaises(ValueError):
                        PS.run_fused(self.runner, w, x)
                continue
            a, out = self.two_kernel(w, x)
            for k, (_, o16) in PS.FUSED_VARIANTS.items():
                with self.subTest(case=i, R=R, D=(dw, dh), special=special, kernel=k):
                    g = PS.run_fused(self.runner, w, x, k)
                    np.testing.assert_array_equal(g["history"], a["history"])
                    np.testing.assert_array_equal(g["reduced"], a["reduced"])
                    np.testing.assert_array_equal(g["out"], out.astype(np.float16).astype(np.float32) if o16 else out)
                    checked += 1
        self.assertGreaterEqual(checked, 4 * 9)

    def test_against_reference(self):
        """Spec 09: FP16 history within one step or 0.001; float32 output within 1e-3 for 99.9 % of the values (pooled
        over the cases: single-step FP16 differences of the display image, FP32 vs FP64, become up to 6.4 x 2^-11 of
        relative output error after Dec) and within 1e-2 for all."""
        rels = []
        for i, (R, D) in enumerate([((40, 24), None), ((26, 15), (59, 33)), ((90, 40), None), ((70, 40), None)]):
            rng = np.random.default_rng(800 + i)
            x = P.random_inputs(rng, R[0], R[1], dw=D and D[0], dh=D and D[1])
            s = float(rng.uniform(0.2, 2.0))
            ref = P.reconstruct(x, s)
            ref_out = P.output_filter(ref["display"], x["exposure"], PS.output_size_for(*x["recon_size"]))
            g = PS.run_fused(self.runner, self.runner.pack_weights("post", {"s": np.array(s)}), x)
            with self.subTest(case=i):
                m = PE.spec09_fp16(g["history"], ref["history"])
                self.assertEqual(m["nonfinite"], 0)
                self.assertLessEqual(m["frac_over_1"], 1e-4, m)
                rel = np.abs(g["out"].astype(np.float64) - ref_out) / np.abs(ref_out)
                self.assertTrue(np.all(np.isfinite(g["out"])))
                self.assertLessEqual(float(rel.max()), 1e-2)
                rels.append(rel.ravel())
        self.assertLessEqual(float(np.mean(np.concatenate(rels) > 1e-3)), 1e-3)

    def test_history_formats(self):
        """The 6-byte history read back by the next frame gives the same result as the 8-byte one."""
        rng = np.random.default_rng(900)
        x = P.random_inputs(rng, 40, 24)
        w = self.runner.pack_weights("post", {"s": np.array(0.7)})
        g6 = PS.run_fused(self.runner, w, x, "post_fused")
        g8 = PS.run_fused(self.runner, w, x, "post_fused_h8")
        x2 = dict(x, history=g6["history"], jitter=np.array([-0.2, 0.3]))
        np.testing.assert_array_equal(PS.run_fused(self.runner, w, x2, "post_fused")["out"],
                                      PS.run_fused(self.runner, w, dict(x2, history=g8["history"]),
                                                   "post_fused_h8")["out"])
        raw = PS.encode_history(g6["history"], True)
        self.assertEqual(len(raw), PS.history_bytes(*x["recon_size"], True))
        np.testing.assert_array_equal(PS.decode_history_fmt(raw, g6["history"].shape, True), g6["history"])
