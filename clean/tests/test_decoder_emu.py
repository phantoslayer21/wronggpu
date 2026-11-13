# SPDX-License-Identifier: MIT
"""Stage dec0 (spec 05): the real kernel source (kernels/decoder.hip) run in the host lane emulator through the spec 90
runner, compared with the NumPy reference (reference/decoder.py). Synthetic data only.

Run: python -m unittest discover -s clean/tests -t .
"""
import unittest

import numpy as np

from clean.reference import decoder as Dc
from clean.reference import fp8
from clean.runner import dec0
from clean.tests.test_decoder_reference import border_case
from clean.tests.test_encoder_emu import f16_ulps
from clean.tests.test_kernels_emu import EmuTest, build_dir


def code_dist(a, b, scale):
    return np.abs(np.rint(np.asarray(a, np.float64) * scale) - np.rint(np.asarray(b, np.float64) * scale))


class TestDecoderEmu(EmuTest):
    CASES = [  # (render size, padded size or None for ceil32, input overrides, param overrides)
        ((40, 24), None, {}, {}),
        ((40, 24), None, {"reset": True}, {}),
        ((22, 18), (26, 20), {}, {}),                  # partial wave and workgroup, odd render width
        ((70, 40), None, {"motion_scale": 20.0}, {}),  # 10 workgroups, many invalid positions
        ((40, 24), None, {}, {"w_scale": 1.0}),        # saturated logits
    ]

    def check(self, got, ref):
        """spec 09 (8-bit and FP16 outputs), and the stricter: >= 99.9 % of all values identical. The kernel follows
        the reference operation by operation, so differences can only come from the projection's FP32 accumulation
        (exact in the reference) and, on the GPU, the exp in the softmax."""
        self.assertEqual(set(got), {"A", "B", "C", "latent"})
        for k, scale in (("A", 255), ("B", 255), ("latent", 127)):
            self.assertEqual(got[k].shape, ref[k].shape, k)
            d = code_dist(got[k], ref[k], scale)
            self.assertEqual(int(np.sum(~np.isfinite(got[k]))), 0, k)
            self.assertLessEqual(float(np.mean(d > 1)), 1e-4, k)
            self.assertLessEqual(float(np.mean(d == 1)), 0.01, k)
            self.assertGreaterEqual(float(np.mean(d == 0)), 0.999, k)
        u = f16_ulps(got["C"], ref["C"])
        self.assertEqual(int(np.sum(~np.isfinite(got["C"]))), 0)
        self.assertLessEqual(float(np.mean(u > 1)), 1e-4)
        self.assertGreaterEqual(float(np.mean(u == 0)), 0.999)
        rms = float(np.sqrt(np.mean(ref["C"].astype(np.float64) ** 2)))
        self.assertLessEqual(float(np.mean(np.abs(got["C"] - ref["C"]))), 1e-3 * rms)

    def run_dec0(self, x, p, runner=None):
        r = runner or self.runner
        return r.run("dec0", r.pack_weights("dec0", p), x, None)

    def test_matches_reference(self):
        for i, (R, F, xo, po) in enumerate(self.CASES):
            with self.subTest(case=i, R=R, F=F, x=xo, p=po):
                rng = np.random.default_rng(500 + i)
                fw, fh = F if F else (None, None)
                x = Dc.random_inputs(rng, rw=R[0], rh=R[1], fw=fw, fh=fh, **xo)
                p = Dc.random_params(rng, **po)
                self.check(self.run_dec0(x, p), Dc.decode(x, p))

    def test_reset_05_T2(self):
        """On reset the previous latent must not be read: garbage there changes nothing."""
        rng = np.random.default_rng(12)
        x = Dc.random_inputs(rng, reset=True)
        p = Dc.random_params(rng)
        a = self.run_dec0(x, p)
        b = self.run_dec0(dict(x, latent_prev=-x["latent_prev"]), p)
        for k in a:
            np.testing.assert_array_equal(a[k], b[k])
        np.testing.assert_array_equal(a["latent"], Dc.decode(x, p)["latent"])

    def test_border_05_T3(self):
        x, p, expect = border_case()
        got, ref = self.run_dec0(x, p), Dc.decode(x, p)
        np.testing.assert_array_equal(got["latent"], ref["latent"])
        # where valid, the stored latent differs from the reset value (p = 0)
        nores = Dc.decode(dict(x, reset=True), p)["latent"]
        for (yy, xx), ok in expect.items():
            self.assertEqual(bool(np.any(got["latent"][yy, xx] != nores[yy, xx])), ok, (yy, xx))


class TestDecoderLayout(EmuTest):
    def run_once(self, runner=None, seed=31):
        rng = np.random.default_rng(seed)
        x = Dc.random_inputs(rng)
        p = Dc.random_params(rng)
        r = runner or self.runner
        return r.run("dec0", r.pack_weights("dec0", p), x, None)

    def test_k_slot_permutation_does_not_matter(self):
        a = self.run_once()
        self.runner.dev.set_layout(1, 0)
        b = self.run_once()
        for k in a:
            np.testing.assert_array_equal(a[k], b[k])

    def test_accumulator_layout_is_absorbed_by_packing(self):
        a = self.run_once()
        self.runner.dev.set_layout(1, 1)
        b = self.run_once(runner=self.Runner(build_dir(), "", backend="emu", d_row="gfx11"))
        for k in a:
            np.testing.assert_array_equal(a[k], b[k])
        c = self.run_once()  # packing for the wrong accumulator layout must break the result
        self.assertGreater(float(np.mean(a["A"] != c["A"])), 0.2)


class TestDecoderPacking(unittest.TestCase):
    def test_orders(self):
        K = dec0.kord_decoder()
        self.assertEqual(sorted(K.tolist()), list(range(64)))
        self.assertTrue(np.all(K[:32] < 32))                     # blocks 0-1: channels 0..31 only
        for name in ("gfx12", "gfx11"):
            Rw = dec0.rows_decoder(dec0.resolve_d_row(name))
            used = Rw[Rw != dec0.PAD]
            self.assertEqual(sorted(used.tolist()), list(range(68)))

    def test_blob_and_tables(self):
        rng = np.random.default_rng(3)
        p = Dc.random_params(rng)
        blob = dec0.pack(p)
        self.assertEqual(len(blob), dec0.BLOB_BYTES)
        tab = np.frombuffer(blob, np.uint32, 5 * 256, dec0.OFF_TABLES).reshape(5, 256)
        f32 = tab.view(np.float32)
        h16 = lambda v: (v & 0xFFFF).astype(np.uint16).view(np.float16).astype(np.float32)  # noqa: E731
        a = np.nan_to_num(fp8.DECODE, nan=0.0).astype(np.float32)
        g = Dc.sigmoid16(a)
        S = np.float32(p["S"])
        np.testing.assert_array_equal(tab[0] & 0xFF, Dc.unorm8_code(g))                       # A code
        np.testing.assert_array_equal(h16(tab[0] >> 16), Dc.rn16((S * g) * (S * g)))           # C0
        np.testing.assert_array_equal(f32[1], S * g)                                          # sx
        np.testing.assert_array_equal(f32[2], Dc.rho32(a, p["T"]))                            # rho
        np.testing.assert_array_equal(h16(tab[3]), Dc.rn16(np.float32(0.5) + np.float32(0.5) * Dc.tanh16(a / 2)))
        np.testing.assert_array_equal(h16(tab[3] >> 16), Dc.tanh16(a))                         # g
        self.assertEqual(f32[4, 0x81], np.float32(-127 / 127))  # s8 code -127
        self.assertEqual(f32[4, 0x80], np.float32(-1.0))        # -128 -> -1
        self.assertEqual(tab[0, 0] & 0xFF, 128)                  # A code of sigma(0) = 0.5
        self.assertTrue(np.all(np.isfinite(f32[[1, 2, 4]])))

if __name__ == "__main__":
    unittest.main()
