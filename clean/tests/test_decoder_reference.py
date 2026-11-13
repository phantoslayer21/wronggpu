# SPDX-License-Identifier: MIT
"""Spec 05 reference (clean/reference/decoder.py) vs a slow per-pixel version written separately from the spec text,
plus 05-T2 (reset) and 05-T3 (reprojection at the borders). Synthetic data only.

Run: python -m unittest discover -s clean/tests -t .
"""
import math
import unittest

import numpy as np

from clean.reference import decoder as Dc
from clean.reference import fp8

F = np.float32


def rn16(v):
    return F(np.float16(F(v)))


def q8(v):
    return float(fp8.q8(v))


def slow_decode(x, prm):
    """Pixel by pixel, NumPy float32 scalars for every single-precision step."""
    rh, rw = np.shape(x["motion"])[:2]
    dw, dh = x["recon_size"]
    Z = np.asarray(x["features"], np.float64)
    W = np.asarray(prm["W"], np.float64)
    b = np.asarray(prm["b"], np.float64)
    S, T = F(prm["S"]), F(prm["T"])
    kx, ky = F(dw / rw), F(dh / rh)
    jx, jy = (F(v) for v in x["jitter"])
    pjx, pjy = (F(v) for v in x["jitter_prev"])
    H = [[[F(max(round(float(v) * 127) / 127, -1.0)) for v in px] for px in row] for row in x["latent_prev"]]
    out = {k: np.zeros((rh, rw, n), np.float64) for k, n in (("A", 3), ("B", 3), ("C", 3), ("latent", 4))}
    for y in range(rh):
        for xx in range(rw):
            z = Z[y // 2, xx // 2]
            s = (xx % 2) + 2 * (y % 2)
            a = []
            for k in range(17):
                n = 17 * s + k
                u = float(np.float16(b[n] + sum(W[n, c] * z[c] for c in range(32))))
                v = float(np.float16(u + sum(W[n, c] * z[c] for c in range(32, 64))))
                a.append(F(q8(v)))

            def sig(t):
                return rn16(F(1) / (F(1) + F(np.exp(F(-t)))))

            A = [sig(a[i]) for i in range(3)]
            m = max(a[3], a[4], a[5])
            e = [F(np.exp(F(a[i] - m))) for i in (3, 4, 5)]
            ssum = F(F(e[0] + e[1]) + e[2])
            B = [F(ei / ssum) for ei in e]
            sx, sy = F(S * sig(a[6])), F(S * sig(a[7]))
            rho = F(math.tanh(float(F(T * a[8]))))
            C = [rn16(F(sx * sx)), rn16(F(sy * sy)), rn16(F(F(sx * sy) * rho))]

            M = x["motion"][y][xx]
            hx = F(F(F(F(xx) + F(0.5)) - jx) * kx) + F(M[0])
            hy = F(F(F(F(y) + F(0.5)) - jy) * ky) + F(M[1])
            valid = (not x["reset"]) and 0 <= hx <= dw and 0 <= hy <= dh
            p = [F(0)] * 4
            if valid:
                X, Y = F(F(hx / kx) + pjx), F(F(hy / ky) + pjy)
                tx, ty = F(X - F(0.5)), F(Y - F(0.5))
                x0, y0 = math.floor(tx), math.floor(ty)
                fx, fy = F(tx - F(x0)), F(ty - F(y0))

                def tex(i, k):
                    return H[min(max(k, 0), rh - 1)][min(max(i, 0), rw - 1)]

                w00, w10 = F(F(1 - fx) * F(1 - fy)), F(fx * F(1 - fy))
                w01, w11 = F(F(1 - fx) * fy), F(fx * fy)
                for c in range(4):
                    P = F(F(F(F(w00 * tex(x0, y0)[c]) + F(w10 * tex(x0 + 1, y0)[c])) + F(w01 * tex(x0, y0 + 1)[c]))
                          + F(w11 * tex(x0 + 1, y0 + 1)[c]))
                    p[c] = rn16(P)
            L = []
            for c in range(4):
                zg = rn16(F(F(0.5) + F(F(0.5) * rn16(math.tanh(float(a[13 + c]) / 2)))))
                g = rn16(math.tanh(float(a[9 + c])))
                w = rn16(F(1) - zg)
                L.append(rn16(F(rn16(F(w * p[c])) + rn16(F(zg * g)))))
            out["A"][y, xx] = [round(float(F(min(max(v, 0), 1) * F(255)))) / 255 for v in A]
            out["B"][y, xx] = [np.rint(F(min(max(v, F(0)), F(1)) * F(255))) / 255 for v in B]
            out["C"][y, xx] = C
            out["latent"][y, xx] = [np.rint(F(min(max(v, F(-1)), F(1)) * F(127))) / 127 for v in L]
    return {k: v.astype(np.float32) for k, v in out.items()}


class DecoderReferenceTest(unittest.TestCase):
    def test_matches_slow_version(self):
        rng = np.random.default_rng(1)
        for reset in (False, True):
            x = Dc.random_inputs(rng, rw=14, rh=10, fw=16, fh=12, reset=reset)
            prm = Dc.random_params(rng)
            ref = Dc.decode(x, prm)
            slow = slow_decode(x, prm)
            for k in ("A", "B", "C", "latent"):
                np.testing.assert_array_equal(ref[k], slow[k], err_msg=k)
            if not reset:
                self.assertGreater(float(np.mean(ref["p"] != 0)), 0.5)       # history path exercised

    def test_reset_05_T2(self):
        rng = np.random.default_rng(2)
        x = Dc.random_inputs(rng, reset=True)
        prm = Dc.random_params(rng)
        ref = Dc.decode(x, prm)
        self.assertTrue(np.all(ref["p"] == 0))
        a = Dc.pixel_values(ref["h"], 40, 24).astype(np.float32)
        z = Dc.rn16(np.float32(0.5) + np.float32(0.5) * Dc.tanh16(a[..., 13:17] / 2))
        L = Dc.rn16(z * Dc.tanh16(a[..., 9:13]))
        np.testing.assert_array_equal(ref["codes_L"], Dc.snorm8_code(L))
        x2 = dict(x, latent_prev=-x["latent_prev"])                              # history must not matter
        np.testing.assert_array_equal(Dc.decode(x2, prm)["latent"], ref["latent"])

    def test_border_05_T3(self):
        x, prm, expect = border_case()
        p = Dc.decode(x, prm)["p"]
        for (yy, xx), ok in expect.items():
            self.assertEqual(bool(np.any(p[yy, xx] != 0)), ok, (yy, xx))

    def test_tables_are_unambiguous(self):
        """RN16(sigma), th(a), th(a/2) over all FP8 inputs: float32 evaluation equals correctly rounded float64."""
        a = fp8.DECODE[np.isfinite(fp8.DECODE)]
        exact = lambda v: np.asarray(v, np.float64).astype(np.float16).astype(np.float32)  # noqa: E731
        np.testing.assert_array_equal(Dc.sigmoid16(a), exact(1 / (1 + np.exp(-a))))
        np.testing.assert_array_equal(Dc.tanh16(a), exact(np.tanh(a.astype(np.float32).astype(np.float64))))


def border_case():
    """D = 2R (k = 2 exactly), jitter (0.5, 0.5): hx = 2x + Mx. Pixels at the four borders get motion that puts hx or
    hy exactly on 0 / D (valid) or one FP16 step outside (invalid). Returns (x, params, {(y, x): valid})."""
    rng = np.random.default_rng(9)
    rw, rh = 40, 24
    x = Dc.random_inputs(rng, rw=rw, rh=rh, dw=2 * rw, dh=2 * rh, motion_scale=0.0)
    x["latent_prev"] = np.full((rh, rw, 4), 100 / 127, np.float32)            # any valid sample is non-zero
    x["jitter"] = np.array([0.5, 0.5], np.float32)
    M = np.zeros((rh, rw, 2), np.float32)
    expect = {}
    eps = 2.0 ** -9
    # left / top edge: hx = Mx (x = 0); right / bottom: hx = 2 (rw - 1) + Mx, D = 2 rw -> Mx = 2 is exactly on D
    cases = [((5, 0), (0.0, 0.0), True), ((6, 0), (-eps, 0.0), False),
             ((0, 7), (0.0, 0.0), True), ((0, 8), (0.0, -eps), False),
             ((5, rw - 1), (2.0, 0.0), True), ((6, rw - 1), (2.0 + 2 * eps, 0.0), False),
             ((rh - 1, 7), (0.0, 2.0), True), ((rh - 1, 8), (0.0, 2.0 + 2 * eps), False)]
    for (yy, xx), (mx, my), ok in cases:
        M[yy, xx] = (mx, my)
        expect[(yy, xx)] = ok
    x["motion"] = M
    return x, Dc.random_params(rng), expect


if __name__ == "__main__":
    unittest.main()
