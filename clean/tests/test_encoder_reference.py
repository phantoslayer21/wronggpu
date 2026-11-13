# SPDX-License-Identifier: MIT
"""Spec 03 reference (clean/reference/encoder.py) vs a slow scalar version written separately from the spec text,
plus the reset test 03-T2 and properties of the dithered quantization. Synthetic data only.

Run: python -m unittest discover -s clean/tests -t .
"""
import math
import unittest

import numpy as np

from clean.reference import encoder as E
from clean.reference import fp8

FP8_VALUES = fp8.DECODE[np.isfinite(fp8.DECODE)]
FP8_CODES = np.nonzero(np.isfinite(fp8.DECODE))[0]


def q8_scalar(x):
    """Nearest E4M3FN value by search; ties to the even code; saturating."""
    if math.isnan(x):
        return math.nan
    x = max(-448.0, min(448.0, x))
    d = np.abs(FP8_VALUES - x)
    best = np.nonzero(d == d.min())[0]
    if len(best) > 1:
        best = [i for i in best if FP8_CODES[i] % 2 == 0]
    v = float(FP8_VALUES[best[0]])
    return math.copysign(v, x) if v == 0.0 else v


def f16_scalar(x):
    return float(np.float16(x))


def dither_scalar(v, t):
    """Spec 03 draft 2, "Dithered FP8 quantization", steps 1-5, one value at a time."""
    hv = float(np.float16(v))
    if math.isfinite(hv):
        sign = 0.0 if hv == 0.0 else math.copysign(1.0, hv)
        if abs(hv) >= 2.0 ** -14:
            E = math.floor(math.log2(abs(hv)))
            u = sign * 2.0 ** E
        else:
            E, u = -14, 0.0
        delta = sign * 2.0 ** E * t / 1024.0
        s = float(np.float16(hv + u + delta))   # Python float sum is exact here; one rounding to FP16
        hv = s - u
    bits = int(np.array(hv, dtype=np.float16).view(np.uint16)) & 0xFF80
    return q8_scalar(float(np.array(bits, dtype=np.uint16).view(np.float16)))


class SlowEncoder:
    """Pixel-by-pixel implementation of spec 03 with Python floats."""

    def __init__(self, x, params):
        self.x, self.p = x, params
        self.rh, self.rw = np.shape(x["color"])[:2]
        self.fw, self.fh = x["feature_size"]
        self.dw, self.dh = x["recon_size"]
        e = float(x["exposure"])
        self.e = 1.0 if e == 0 else e
        ep = float(x["exposure_prev"])
        self.ep = 1.0 if ep == 0 else ep
        self.kx, self.ky = self.dw / self.rw, self.dh / self.rh

    def refl(self, p, n):
        if p < 0:
            p = -p
        if p >= n:
            p = 2 * n - p - 2
        return p

    def m(self, x, y):
        mv = self.x["motion"][y][x]
        sx = float(self.x["mv_scale"][0]) * self.kx
        sy = float(self.x["mv_scale"][1]) * self.ky
        return (sx * (float(self.x["mv_bias"][0]) + float(mv[0])), sy * (float(self.x["mv_bias"][1]) + float(mv[1])))

    def near_far(self, x, y):
        order = [(0, 0), (-1, -1), (0, -1), (-1, 0), (1, -1), (1, 0), (-1, 1), (0, 1), (1, 1)]
        cands = [(min(max(x + dx, 0), self.rw - 1), min(max(y + dy, 0), self.rh - 1)) for dx, dy in order]
        z = [float(self.x["depth"][cy][cx]) for cx, cy in cands]
        lo = hi = 0
        for i in range(1, 9):
            if z[i] < z[lo]:
                lo = i
            if z[i] > z[hi]:
                hi = i
        if self.x["depth_inverted"]:
            return cands[hi], cands[lo]
        return cands[lo], cands[hi]

    @staticmethod
    def sample(tex, u, v):
        h, w = len(tex), len(tex[0])
        tx, ty = u * w - 0.5, v * h - 0.5
        x0, y0 = math.floor(tx), math.floor(ty)
        fx, fy = tx - x0, ty - y0

        def at(i, k):
            return np.asarray(tex[min(max(k, 0), h - 1)][min(max(i, 0), w - 1)], dtype=np.float64)

        top = at(x0, y0) * (1 - fx) + at(x0 + 1, y0) * fx
        bot = at(x0, y0 + 1) * (1 - fx) + at(x0 + 1, y0 + 1) * fx
        return top * (1 - fy) + bot * fy

    def values(self, X, Y):
        x, y = self.refl(X, self.rw), self.refl(Y, self.rh)
        reset = bool(self.x["reset"])
        (nx, ny), (fx_, fy_) = self.near_far(x, y)
        mn, mf, mc = self.m(nx, ny), self.m(fx_, fy_), self.m(x, y)
        jx, jy = (float(v) for v in self.x["jitter"])
        jpx, jpy = (float(v) for v in self.x["jitter_prev"])
        bx, by = x + 0.5 - jx, y + 0.5 - jy
        pos = {"near": (bx * self.kx + mn[0], by * self.ky + mn[1]),
               "center": (bx * self.kx + mc[0], by * self.ky + mc[1]),
               "far": (bx * self.kx + mf[0], by * self.ky + mf[1]),
               "unjit": ((x + 0.5) * self.kx + mc[0], (y + 0.5) * self.ky + mc[1])}

        def valid(p):
            return (not reset) and 0 <= p[0] <= self.dw and 0 <= p[1] <= self.dh

        out = []
        c = [float(v) for v in self.x["color"][y][x]]
        for ch in c:
            prod = self.e * ch
            if math.isnan(prod):
                out.append(0.0)
            else:
                out.append(0.36 * math.log10(max(1 + 100 * min(prod, 32752.0), 1e-4)))
        for name in ("near", "center", "far", "unjit"):
            p = pos[name]
            if valid(p):
                s = self.sample(self.x["reduced_history"], p[0] / self.dw, p[1] / self.dh)
                hs = []
                for ch in s:
                    if math.isnan(self.ep) or self.ep < 0:
                        hs.append(2.45 * ch)
                    else:
                        q = min(max(2.45 * ch, 1e-4), 3.0)
                        hs.append((10 ** (q / 0.36) - 1) / 100 / self.ep)
            else:
                hs = c
            for h in hs:
                arg = 1 + 100 * self.e * h
                out.append(math.nan if math.isnan(arg) else 0.36 * math.log10(max(arg, 1e-4)) if arg != math.inf
                           else math.inf)
        if reset:
            out.append(0.0)
        else:
            ux = min(max((bx + jpx + mn[0] / self.kx) / self.rw, 0.0), 1.0)
            uy = min(max((by + jpy + mn[1] / self.ky) / self.rh, 0.0), 1.0)
            if math.isnan(ux) or math.isnan(uy):
                out.append(math.nan)
            else:
                mo = self.sample(self.x["motion_history"], ux, uy)
                s1, s2 = float(self.p["sigma1"]), float(self.p["sigma2"])
                vx = s1 * (540.0 / self.rh) * (mn[0] - mo[0]) / self.kx
                vy = s1 * (540.0 / self.rh) * (mn[1] - mo[1]) / self.ky
                n = math.hypot(vx, vy)
                out.append(0.0 if n == 0 else s2 * math.log(1 + n))
        if valid(pos["near"]):
            lu = (pos["near"][0] / self.kx + jpx) / self.rw
            lv = (pos["near"][1] / self.ky + jpy) / self.rh
            out += [float(v) for v in self.sample(self.x["latent"], lu, lv)]
        else:
            out += [0.0] * 4
        return [min(0.0 if math.isnan(v) else v, 1024.0) for v in out]

    def run(self):
        fw, fh = self.fw, self.fh
        vals = np.zeros((fh, fw, 20))
        qv = np.zeros((fh, fw, 20))
        T = self.p["T"][int(self.x["frame"])]
        for Y in range(fh):
            for X in range(fw):
                vals[Y, X] = self.values(X, Y)
                t = int(T[abs(Y) % 16][abs(X) % 16])
                qv[Y, X] = [dither_scalar(v, t) for v in vals[Y, X]]
        W, b = np.asarray(self.p["W"], np.float64), np.asarray(self.p["b"], np.float64)
        feats = np.zeros((fh // 2, fw // 2, 64))
        for j in range(fh // 2):
            for i in range(fw // 2):
                for f in range(64):
                    s = float(b[f])
                    for p in range(4):
                        dx, dy = p % 2, p // 2
                        a = qv[2 * j + dy, 2 * i + dx]
                        s = f16_scalar(s + sum(float(W[f, 20 * p + n]) * a[n] for n in range(20)))
                    feats[j, i, f] = q8_scalar(max(-6.28125, min(6.28125, s)))
        return vals, qv, feats


class EncoderReferenceTest(unittest.TestCase):
    def check_against_slow(self, x, params):
        ref = E.encode(x, params)
        vals, qv, feats = SlowEncoder(x, params).run()
        np.testing.assert_allclose(ref["values"], vals, rtol=1e-12, atol=1e-12)
        np.testing.assert_array_equal(ref["qvalues"], qv)
        np.testing.assert_array_equal(ref["features"], feats.astype(np.float32))
        return ref

    def test_matches_slow_version(self):
        rng = np.random.default_rng(3)
        for inv in (False, True):
            x = E.random_inputs(rng, rw=20, rh=18, fw=24, fh=20, depth_inverted=inv)
            p = E.random_params(rng)
            ref = self.check_against_slow(x, p)
            self.assertGreater(np.mean(ref["values"][..., E.V_LATENT] != 0), 0.5)   # latent path exercised
            self.assertGreater(np.mean(ref["values"][..., E.V_DISAGREE] != 0), 0.5)

    def test_exposure_edge_cases(self):
        rng = np.random.default_rng(4)
        for e, ep in ((0.0, 0.0), (1.7, -1.0), (0.5, float("nan"))):
            x = E.random_inputs(rng, rw=18, rh=18, fw=20, fh=18)
            x["exposure"], x["exposure_prev"] = e, ep
            self.check_against_slow(x, E.random_params(rng))

    def test_reset_03_T2(self):
        rng = np.random.default_rng(5)
        x = E.random_inputs(rng, rw=40, rh=24, reset=True)
        out = E.encode(x, E.random_params(rng))
        v = out["values"]
        c = v[..., 0:3]
        for off in (E.V_NEAR, E.V_CENTER, E.V_FAR, E.V_UNJIT):
            # Equal to the colour encoding wherever the colour cap 32752 and NaN -> 0 do not differ.
            hv = v[..., off:off + 3]
            Y, X = np.mgrid[0:32, 0:64]
            colour = np.asarray(x["color"], np.float64)[E.reflect(Y, 24), E.reflect(X, 40)]
            ok = np.isfinite(colour) & (E.exposure_or_one(x["exposure"]) * colour < 32752)
            np.testing.assert_allclose(hv[ok], c[ok], rtol=1e-12, atol=1e-12)
        self.assertTrue(np.all(v[..., E.V_DISAGREE] == 0))
        self.assertTrue(np.all(v[..., E.V_LATENT:] == 0))

    def test_motion_surfaces(self):
        rng = np.random.default_rng(6)
        x = E.random_inputs(rng, rw=20, rh=18, fw=24, fh=20, special=False)
        out = E.encode(x, E.random_params(rng))
        slow = SlowEncoder(x, E.random_params(rng))
        for yy in range(18):
            for xx in range(20):
                n, f = slow.near_far(xx, yy)
                for key, (px, py) in (("m_near", n), ("m_far", f), ("m_center", (xx, yy))):
                    want = [f16_scalar(v) for v in slow.m(px, py)]
                    self.assertEqual(list(out[key][yy, xx]), want)

    def test_dither_is_stochastic_rounding(self):
        # Averaged over all 128 dither values, the result is the FP16 input within a small fraction of an FP8 step
        # (the dither has steps of 1/128 of an FP8 step; step 3 rounds on a grid of 1/64 step, ties to even).
        rng = np.random.default_rng(7)
        v = rng.uniform(0.02, 400.0, size=2000) * rng.choice([-1, 1], size=2000)
        t = np.arange(128)
        q = E.dither_fp8(v[:, None], t[None, :])
        mean = q.mean(axis=1)
        step = 2.0 ** (np.floor(np.log2(np.abs(v))) - 3)
        self.assertTrue(np.all(np.abs(mean - E.f16(v)) <= step / 64 + 1e-12))
        self.assertTrue(fp8.is_fp8_exact(q))
        for vi, ti in zip(rng.uniform(-500, 500, 300), rng.integers(0, 128, 300)):
            self.assertEqual(E.dither_fp8(vi, ti), dither_scalar(vi, int(ti)))


if __name__ == "__main__":
    unittest.main()
