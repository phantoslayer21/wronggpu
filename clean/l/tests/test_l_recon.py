# SPDX-License-Identifier: MIT
"""Model L's reconstruction oracle (clean/l/reference/recon.py) and its variant rule, CPU only, synthetic data.

Run: python -m unittest discover -s clean/l/tests -t . -p test_l_recon.py

The strongest check reduces Model L to Model M through the table (3 x 3 window, encoded-domain combination and
fallback) and requires Model M's oracle (clean/reference/post.py) byte for byte.
"""
import unittest

import numpy as np

from clean.l.reference import oracle as L
from clean.l.reference import recon as R
from clean.l.reference import table as T
from clean.reference import motion_hi as MH
from clean.reference import post as MP


def _comb32(w1, w2, h1, h2, e):
    """Model L's combination in float32 (the spec side's order): f32(fused(w1 h1 + f32(w2 h2)) e), w1 h1 alone when
    w2 = 0, w1 and w2 rounded to float32 first."""
    w1, w2 = R._f32(w1)[..., None], R._f32(w2)[..., None]
    c = np.where(w2 == 0, R._f32(w1 * h1), np.asarray(MH.fma32(w1, h1, R._f32(w2 * h2)), np.float64))
    return R._f32(c * e)


def _same(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return a.shape == b.shape and a.dtype == b.dtype and a.tobytes() == b.tobytes()


def _f16_steps(a, b):
    """Distance in FP16 steps (ordered bit patterns) of FP16-exact arrays."""
    ia, ib = (np.asarray(v, np.float32).astype(np.float16).view(np.int16).astype(np.int64) for v in (a, b))
    ia = np.where(ia < 0, -(ia & 0x7FFF), ia)
    ib = np.where(ib < 0, -(ib & 0x7FFF), ib)
    return np.abs(ia - ib)


def _inputs(rng, rw=23, rh=15, reset=False):
    x = MP.random_inputs(rng, rw, rh, reset=reset)
    x["exposure"], x["exposure_prev"] = np.float32(1.7), np.float32(0.6)
    return x


class VariantTest(unittest.TestCase):
    def test_modes(self):
        t = T.TABLE
        out = (3840, 2160)
        self.assertEqual(T.recon_size(*out), (5760, 3240))
        self.assertEqual(T.recon_variant(t, 1920, 1080, *out), "3/1")          # Performance: exactly 3.0
        self.assertEqual(T.recon_variant(t, 1280, 720, *out), "3/1")           # Ultra Performance: 4.5
        self.assertEqual(T.recon_variant(t, 2560, 1440, *out), "3/2")          # Quality: 2.25
        self.assertEqual(T.recon_variant(t, 2227, 1253, *out), "3/2")          # Balanced: about 2.6
        self.assertEqual(T.recon_variant(t, 1921, 1080, *out), "3/2")          # just below 3.0 on one axis
        self.assertEqual(T.recon_variant(t, 1920, 1081, *out), "3/2")

    def test_ratio_is_d_over_render(self):
        """The ratio is D / render, so output = 2 x render takes 3/1 while output / render is 2, not 3."""
        self.assertEqual(T.recon_variant(T.TABLE, 1000, 600, 2000, 1200), "3/1")
        self.assertEqual(T.recon_variant(T.TABLE, 1000, 600, 1999, 1200), "3/2")


class ReconTest(unittest.TestCase):
    def test_model_m_entry_is_model_m(self):
        """With Model M's recon entry the L oracle is Model M's oracle, byte for byte (also with reset)."""
        t = T.table(recon=dict(T.M_RECON))
        for seed, reset in ((1, False), (2, True), (3, False)):
            rng = np.random.default_rng(seed)
            x = _inputs(rng, reset=reset)
            s = 2.9796233
            want = MP.reconstruct(x, s)
            got = R.reconstruct(x, s, t)
            for k in ("history", "reduced", "display"):
                self.assertTrue(_same(got[k], want[k]), (k, seed))

    def test_gather_window(self):
        """The 4 x 4 gather against a tap-by-tap loop over columns ax - 1 ... ax + 2 and rows b - 1 ... b + 2 (b of the
        even row of the pair)."""
        rng = np.random.default_rng(4)
        x = _inputs(rng)
        color = np.asarray(x["color"], np.float64)
        rh, rw = color.shape[:2]
        dw, dh = x["recon_size"]
        px, py = MP.target_positions(rw, rh, dw, dh, *np.asarray(x["jitter"], np.float64))
        feat = MP.render_features(color, x["A"], x["B"], 1.0)
        W, fbar = R.gather(feat, x["C"], px, py, rw, rh, 4)
        C = np.asarray(x["C"], np.float64)
        for (yy, xx) in ((0, 0), (5, 7), (dh - 1, dw - 1), (6, 3), (7, 3)):
            ax = int(np.floor(px[xx] - 0.5))
            b = int(np.floor(py[yy & ~1] - 0.5))
            w_sum, acc = 0.0, np.zeros(9)
            for i in (-1, 0, 1, 2):
                for j in (-1, 0, 1, 2):
                    sx, sy = min(max(ax + i, 0), rw - 1), min(max(b + j, 0), rh - 1)
                    u, v = ax + i + 0.5 - px[xx], b + j + 0.5 - py[yy]
                    a_, b_, c_ = C[sy, sx]
                    w = np.exp(-0.5 * (a_ * u * u + b_ * v * v + 2 * c_ * u * v))
                    w_sum += w
                    acc += w * feat[sy, sx]
            np.testing.assert_allclose(W[yy, xx], w_sum, rtol=1e-12)
            np.testing.assert_allclose(fbar[yy, xx], acc / w_sum, rtol=1e-12)

    def test_reset_uses_the_linear_colour(self):
        """On reset every sample falls back to the render pixel's linear colour (no exposure), combined in linear
        colour and times the current exposure: H_lin = (w1 + w2) colour(floor p) e."""
        rng = np.random.default_rng(5)
        x = _inputs(rng, reset=True)
        out = R.reconstruct(x, 2.0)
        lin = out["_lin"]
        color = np.asarray(x["color"], np.float64)
        rh, rw = color.shape[:2]
        dw, dh = x["recon_size"]
        px, py = R.target_positions_f32(rw, rh, dw, dh, *np.asarray(x["jitter"], np.float64))
        fy, fx = MP.point_index(px, py, rw, rh)
        H_lin = _comb32(lin["w1"], lin["w2"], color[fy, fx], color[fy, fx], float(x["exposure"]))
        self.assertTrue(np.isfinite(lin["H_hat"]).all())
        m = R.reconstruct(x, 2.0, T.table(recon=dict(T.M_RECON)))
        self.assertFalse(np.allclose(m["_lin"]["H_hat"], lin["H_hat"]))
        # alpha = 1 everywhere (huge s) gives beta' = 1, so H^ = C_lin and O^ = C_lin: the history term drops out.
        big = R.reconstruct(x, 1e30)
        np.testing.assert_allclose(big["_lin"]["H_hat"], big["_lin"]["O_hat"])
        # s = 0 gives alpha = 0: H^ = H_lin + (C_lin - H_lin) beta_h, linear in H_lin; with beta_h = 0 it is H_lin.
        x0 = dict(x, A=np.zeros_like(np.asarray(x["A"])))           # beta_h = beta_o = 0 (and g = 0)
        z = R.reconstruct(x0, 0.0)
        H_lin = _comb32(z["_lin"]["w1"], z["_lin"]["w2"], color[fy, fx], color[fy, fx], float(x["exposure"]))
        self.assertTrue(np.array_equal(z["_lin"]["H_hat"], H_lin))   # beta' = 0 exactly: H^ = Hl

    def test_linear_combination(self):
        """A valid sample is decoded and divided by e_prev before the two candidates are combined; the result is
        then multiplied by e. Checked against a direct recomputation with s = 0 and beta = 0 (O^ = H^ = H_lin)."""
        rng = np.random.default_rng(6)
        x = _inputs(rng)
        x0 = dict(x, A=np.zeros_like(np.asarray(x["A"])))
        out = R.reconstruct(x0, 0.0)
        lin = out["_lin"]
        color = np.asarray(x["color"], np.float64)
        rh, rw = color.shape[:2]
        dw, dh = x["recon_size"]
        e, ep = float(x["exposure"]), float(x["exposure_prev"])
        px, py = R.target_positions_f32(rw, rh, dw, dh, *np.asarray(x["jitter"], np.float64))
        fy, fx = MP.point_index(px, py, rw, rh)
        X = np.arange(dw)[None, :] + 0.5
        Y = np.arange(dh)[:, None] + 0.5
        m = [np.asarray(x["m_near"], np.float64)[fy, fx], MP.bilinear(x["m_center"], px, py),
             np.asarray(x["m_far"], np.float64)[fy, fx]]
        cands = []
        hist = np.asarray(x["history"], np.float64)
        for mk in m:
            qx, qy = X + mk[..., 0], Y + mk[..., 1]
            valid = (qx >= 0) & (qx <= dw) & (qy >= 0) & (qy <= dh)
            hs = R._f32(R._f32(MP.dec(MP.history_sample(hist, np.where(valid, qx, 0.5), np.where(valid, qy, 0.5))))
                        / ep)
            cands.append(np.where(valid[..., None], hs, color[fy, fx]))
        Hk = np.stack(cands, -1)
        h1 = np.take_along_axis(Hk, lin["first"][..., None, None], -1)[..., 0]
        h2 = np.take_along_axis(Hk, lin["second"][..., None, None], -1)[..., 0]
        self.assertTrue(np.array_equal(lin["H_hat"], _comb32(lin["w1"], lin["w2"], h1, h2, e)))
        # The same with the blend in float64 (Model M's precision): the float64 combination.
        lin64 = R.reconstruct(x0, 0.0, T.table(recon=dict(T.TABLE["recon"], blend="float64")))["_lin"]
        w1, w2 = lin64["w1"][..., None], lin64["w2"][..., None]
        h1d = np.take_along_axis(Hk, lin64["first"][..., None, None], -1)[..., 0]
        h2d = np.take_along_axis(Hk, lin64["second"][..., None, None], -1)[..., 0]
        np.testing.assert_allclose(lin64["H_hat"], np.where(w2 == 0, w1 * h1d, w1 * h1d + w2 * h2d) * e, rtol=1e-6)

    def test_ldr_reads_no_exposure(self):
        rng = np.random.default_rng(7)
        x = _inputs(rng)
        a = R.reconstruct(x, 2.9796233, ldr=True)
        b = R.reconstruct(dict(x, exposure=np.float32(5.0), exposure_prev=np.float32(0.1)), 2.9796233, ldr=True)
        for k in ("history", "reduced", "display"):
            self.assertTrue(_same(a[k], b[k]), k)
        self.assertFalse(_same(a["history"], R.reconstruct(x, 2.9796233)["history"]))

    def test_zero_weight_known_answer(self):
        """a, b = 65504, c = 0 everywhere: almost every D pixel has total gather weight 0. Confirmed results there:
        HDR history h(E(NaN)) = -1.4404296875 per channel, display 0, reduced code 0; LDR history NaN, display 0,
        reduced code 0 (also for any 2 x 2 block holding such a pixel)."""
        rng = np.random.default_rng(9)
        x = _inputs(rng, rw=17, rh=11)
        x["C"] = np.broadcast_to(np.array([65504.0, 65504.0, 0.0], np.float32), np.shape(x["C"])).copy()
        for ldr in (False, True):
            out = R.reconstruct(x, 2.9796233, ldr=ldr)
            zero = out["_lin"]["W"] == 0
            self.assertGreater(zero.mean(), 0.5)
            hist = out["history"][zero]
            if ldr:
                self.assertTrue(np.isnan(hist).all())
                self.assertTrue((out["display_codes"][zero] == 0).all())
            else:
                self.assertTrue((hist == np.float32(-1.4404296875)).all())
            self.assertTrue((out["display"][zero] == 0).all())
            dh, dw = zero.shape
            blk = zero[:dh // 2 * 2, :dw // 2 * 2].reshape(dh // 2, 2, dw // 2, 2).any(axis=(1, 3))
            self.assertTrue(blk.any())
            self.assertTrue((out["reduced"][blk] == 0).all())

    def test_ldr_constant_image_end_to_end(self):
        """LDR, constant colour 0.5 and history 0.5: O^ = 0.5, display codes R = G = floor(0.5 2047 + 1/2) = 1024,
        B = floor(0.5 1023 + 1/2) = 512; the output stage unpacks code RN32(1/N) to FP16 and filters a constant image
        to itself, with no decode and no exposure."""
        rng = np.random.default_rng(10)
        x = _inputs(rng, rw=20, rh=12)
        x["color"] = np.full(np.shape(x["color"]), 0.5, np.float32)
        x["history"] = np.full(np.shape(x["history"]), 0.5, np.float32)
        x["C"] = np.zeros(np.shape(x["C"]), np.float32)          # every weight exactly 1: the half gather is exact
        dw, dh = x["recon_size"]
        ow, oh = (2 * dw) // 3, (2 * dh) // 3
        self.assertEqual(T.recon_size(ow, oh), (dw, dh))
        want = np.array([np.float16(np.float32(1024 * np.float32(1 / 2047))),
                         np.float16(np.float32(1024 * np.float32(1 / 2047))),
                         np.float16(np.float32(512 * np.float32(1 / 1023)))], np.float32)
        for expo in (1.0, 5.0):
            out = R.reconstruct(dict(x, exposure=np.float32(expo)), 2.9796233, ldr=True)
            np.testing.assert_allclose(out["_lin"]["O_hat"], 0.5, rtol=1e-12)
            self.assertTrue((out["display_codes"] == np.array([1024, 1024, 512])).all())
            img = R.output_filter(out["display"], np.float32(expo), (ow, oh), ldr=True)
            self.assertEqual(img.shape, (oh, ow, 3))
            self.assertTrue((img == want).all())
        self.assertEqual(np.float32(1 / 2047).view(np.uint32), 0x3A001002)
        self.assertEqual(np.float32(1 / 1023).view(np.uint32), 0x3A802008)

    def _extreme(self, offset, reset):
        """D = R, every target `offset` render pixels right of a texel centre (jitter), C = (65504, 65504, 0)."""
        rng = np.random.default_rng(12)
        x = MP.random_inputs(rng, 16, 12, 16, 12, reset=reset)
        x["jitter"] = np.array([offset, 0.0], np.float32)
        x["exposure"], x["exposure_prev"] = np.float32(1.5), np.float32(0.75)
        x["C"] = np.broadcast_to(np.array([65504.0, 65504.0, 0.0], np.float32), np.shape(x["C"])).copy()
        for k in ("m_near", "m_center", "m_far"):
            x[k] = np.zeros(np.shape(x[k]), np.float32)
        x["history"] = np.full(np.shape(x["history"]), 0.75, np.float32)  # E-encoded, constant
        return x

    def test_gather_precision_constants(self):
        """A = B = h(f32(65504 -0.721325)) = -47264; at offset 0.1: u = h(0.1) = 0.0999755859375, h(u u) =
        0.0099945068359375, eps = -472.5."""
        self.assertEqual(R._h(R._f32(65504.0 * R._f32(-0.721325))), -47264.0)
        u = R._h(0.1)
        self.assertEqual(u, 0.0999755859375)
        self.assertEqual(R._h(u * u), 0.0099945068359375)
        self.assertEqual(R._h(R._h(u * u) * -47264.0), -472.5)

    def test_offset_005_is_the_candidate_history(self):
        """Offset 0.05 (between the h(w) = 0 and w = 0 thresholds): W > 0, every FP16 weight 0, all features 0, so the
        output is candidate 1's history: with reset the linear colour times e; without, D(history) / e_prev times e
        (the history is constant and every position valid)."""
        for reset in (True, False):
            x = self._extreme(0.05, reset)
            out = R.reconstruct(x, 2.9796233)
            W = out["_lin"]["W"]
            self.assertTrue((W > 0).all() and np.isfinite(W).all())
            e, ep = 1.5, 0.75
            if reset:
                Hl = R._f32(np.asarray(x["color"], np.float64) * e)        # D = R, floor p = the pixel itself; w1 = 1
            else:
                Hl = np.full(out["history"].shape, R._f32(R._f32(R._f32(MP.dec(0.75)) / ep) * e))
            self.assertTrue(np.array_equal(out["_lin"]["H_hat"], Hl))     # alpha = 0, beta-bar = 0: H^ = Hl
            self.assertTrue(np.array_equal(out["_lin"]["O_hat"], Hl))

    def test_offset_01_is_zero_weight(self):
        """Offset 0.1 (beyond 0.0516): W = 0 everywhere, the confirmed zero-weight values."""
        x = self._extreme(0.1, False)
        out = R.reconstruct(x, 2.9796233)
        self.assertTrue((out["_lin"]["W"] == 0).all())
        self.assertTrue((out["history"] == np.float32(-1.4404296875)).all())
        self.assertTrue((out["display"] == 0).all())
        self.assertTrue((out["reduced"] == 0).all())

    def test_blend_in_float32_with_large_colours(self):
        """Steps 4 to 6 in float32 with the fused forms (the spec side's order): H^ = fused(f32(C - Hl) beta' + Hl)
        exactly, also where the fallback colour is large (1e3 ... 1e6, reset: every sample is the fallback). There
        H^ comes from cancelling large terms, so the float64 form lands several FP16 steps away: the case of the
        frame check's special colours."""
        rng = np.random.default_rng(14)
        x = _inputs(rng, reset=True)
        col = np.asarray(x["color"], np.float32).copy()
        big = rng.random(col.shape[:2]) < 0.1
        col[big] = rng.uniform(1e3, 1e6, (int(big.sum()), 3))
        x["color"] = col
        lin = R.reconstruct(x, 2.9796233)["_lin"]
        C, Hl, bh, bo = lin["C_lin"], lin["H_lin"], lin["beta_h"][..., None], lin["beta_o"][..., None]
        self.assertTrue(np.array_equal(lin["H_hat"], np.asarray(MH.fma32(R._f32(C - Hl), bh, Hl), np.float64)))
        self.assertTrue(np.array_equal(lin["O_hat"], np.asarray(MH.fma32(R._f32(C - Hl), bo, Hl), np.float64)))
        out64 = R.reconstruct(x, 2.9796233, T.table(recon=dict(T.TABLE["recon"], blend="float64")))
        st = _f16_steps(R.reconstruct(x, 2.9796233)["history"], out64["history"])
        self.assertGreater(int(st.max()), 1)

    def test_history_of_another_size_needs_reset(self):
        rng = np.random.default_rng(11)
        x = _inputs(rng)
        small = np.asarray(x["history"])[:-2, :-2]
        with self.assertRaises(ValueError):
            R.reconstruct(dict(x, history=small), 2.0)
        R.reconstruct(dict(x, history=small, reset=True), 2.0)          # reset: the history is not read

    def test_oracle_entry_uses_the_post_scale(self):
        rng = np.random.default_rng(8)
        x = _inputs(rng)
        params = {"post": {"s": np.float32(2.9796233)}}
        self.assertTrue(_same(L.reconstruct(x, params)["display"], R.reconstruct(x, 2.9796233)["display"]))
        self.assertEqual(T.PROVISIONAL, {})


if __name__ == "__main__":
    unittest.main()
