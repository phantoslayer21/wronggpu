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
from clean.reference import post as MP


def _same(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return a.shape == b.shape and a.dtype == b.dtype and a.tobytes() == b.tobytes()


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
        px, py = MP.target_positions(rw, rh, dw, dh, *np.asarray(x["jitter"], np.float64))
        fy, fx = MP.point_index(px, py, rw, rh)
        w = lin["w1"] + np.where(lin["w2"] == 0, 0.0, lin["w2"])
        H_lin = w[..., None] * color[fy, fx] * float(x["exposure"])
        self.assertTrue(np.isfinite(lin["H_hat"]).all())
        m = R.reconstruct(x, 2.0, T.table(recon=dict(T.M_RECON)))
        self.assertFalse(np.allclose(m["_lin"]["H_hat"], lin["H_hat"]))
        # alpha = 1 everywhere (huge s) gives beta' = 1, so H^ = C_lin and O^ = C_lin: the history term drops out.
        big = R.reconstruct(x, 1e30)
        np.testing.assert_allclose(big["_lin"]["H_hat"], big["_lin"]["O_hat"])
        # s = 0 gives alpha = 0: H^ = H_lin + (C_lin - H_lin) beta_h, linear in H_lin; with beta_h = 0 it is H_lin.
        x0 = dict(x, A=np.zeros_like(np.asarray(x["A"])))           # beta_h = beta_o = 0 (and g = 0)
        z = R.reconstruct(x0, 0.0)
        np.testing.assert_allclose(z["_lin"]["H_hat"], H_lin, rtol=1e-12)

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
        px, py = MP.target_positions(rw, rh, dw, dh, *np.asarray(x["jitter"], np.float64))
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
            hs = MP.dec(MP.history_sample(hist, np.where(valid, qx, 0.5), np.where(valid, qy, 0.5))) / ep
            cands.append(np.where(valid[..., None], hs, color[fy, fx]))
        Hk = np.stack(cands, -1)
        h1 = np.take_along_axis(Hk, lin["first"][..., None, None], -1)[..., 0]
        h2 = np.take_along_axis(Hk, lin["second"][..., None, None], -1)[..., 0]
        w1, w2 = lin["w1"][..., None], lin["w2"][..., None]
        H_lin = np.where(w2 == 0, w1 * h1, w1 * h1 + w2 * h2) * e
        np.testing.assert_allclose(lin["H_hat"], H_lin, rtol=1e-12)

    def test_ldr_reads_no_exposure(self):
        rng = np.random.default_rng(7)
        x = _inputs(rng)
        a = R.reconstruct(x, 2.9796233, ldr=True)
        b = R.reconstruct(dict(x, exposure=np.float32(5.0), exposure_prev=np.float32(0.1)), 2.9796233, ldr=True)
        for k in ("history", "reduced", "display"):
            self.assertTrue(_same(a[k], b[k]), k)
        self.assertFalse(_same(a["history"], R.reconstruct(x, 2.9796233)["history"]))

    def test_oracle_entry_uses_the_post_scale(self):
        rng = np.random.default_rng(8)
        x = _inputs(rng)
        params = {"post": {"s": np.float32(2.9796233)}}
        self.assertTrue(_same(L.reconstruct(x, params)["display"], R.reconstruct(x, 2.9796233)["display"]))
        self.assertEqual(T.PROVISIONAL, {})


if __name__ == "__main__":
    unittest.main()
