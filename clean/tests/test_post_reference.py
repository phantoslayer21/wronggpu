# SPDX-License-Identifier: MIT
"""Spec 06 reference (reference/post.py) against slow per-pixel loop versions written independently from the spec
text, plus checks of the stated edge cases. Synthetic data only.

Run: python -m unittest discover -s clean/tests -t .
"""
import math
import unittest

import numpy as np

from clean.reference import post as P


def fmax(a, b):
    return b if math.isnan(a) else (a if math.isnan(b) else max(a, b))


def fmin(a, b):
    return b if math.isnan(a) else (a if math.isnan(b) else min(a, b))


def E(v):
    t = 1.0 + 100.0 * v if not math.isnan(v) else float("nan")
    return 0.36 * math.log10(fmax(t, 1e-4))


def E_in(v):
    return 0.0 if math.isnan(v) else E(fmin(v, 32752.0))


def Dec(q):
    q = fmin(fmax(q, 1e-4), 3.0)
    return (10.0 ** (q / 0.36) - 1.0) / 100.0


def clampi(v, lo, hi):
    return min(max(v, lo), hi)


def cr(t):
    t = abs(t)
    if t < 1:
        return 1.5 * t ** 3 - 2.5 * t ** 2 + 1
    if t < 2:
        return -0.5 * t ** 3 + 2.5 * t ** 2 - 4 * t + 2
    return 0.0


def slow_reconstruct(x, s):
    """Spec 06 "Reconstruction", one D pixel at a time."""
    color, A, B, C = (np.asarray(x[k], np.float64) for k in ("color", "A", "B", "C"))
    hist = np.asarray(x["history"], np.float64)
    rh, rw = color.shape[:2]
    dw, dh = x["recon_size"]
    jx, jy = (float(v) for v in x["jitter"])
    e = x["exposure"] or 1.0
    ep = x["exposure_prev"] or 1.0
    rx, ry = rw / dw, rh / dh
    Hh = np.zeros((dh, dw, 3))
    Oh = np.zeros((dh, dw, 3))
    for y in range(dh):
        for xx in range(dw):
            px = (xx + 0.5) * rx + jx
            py = (y + 0.5) * ry + jy
            ax = math.floor(px - 0.5)
            ye = y - (y % 2)
            b = math.floor((ye + 0.5) * ry + jy - 0.5)
            W = 0.0
            f = np.zeros(9)
            for j in (-1, 0, 1):
                for i in (-1, 0, 1):
                    sx, sy = clampi(ax + i, 0, rw - 1), clampi(b + j, 0, rh - 1)
                    u, v = (ax + i + 0.5) - px, (b + j + 0.5) - py
                    a_, b_, c_ = C[sy, sx]
                    w = math.exp(-0.5 * (a_ * u * u + b_ * v * v + 2 * c_ * u * v))
                    feat = [E_in(e * color[sy, sx, k]) for k in range(3)] + list(B[sy, sx]) + list(A[sy, sx])
                    W += w
                    f += w * np.array(feat)
            f /= W
            cbar, sig, g, bh, bo = f[0:3], f[3:6], f[6], f[7], f[8]
            fx, fy = clampi(math.floor(px), 0, rw - 1), clampi(math.floor(py), 0, rh - 1)
            m1 = x["m_near"][fy, fx].astype(np.float64)
            m3 = x["m_far"][fy, fx].astype(np.float64)
            tx, ty = px - 0.5, py - 0.5
            x0, y0 = math.floor(tx), math.floor(ty)
            m2 = np.zeros(2)
            for (cx, wxx) in ((x0, 1 - (tx - x0)), (x0 + 1, tx - x0)):
                for (cy, wyy) in ((y0, 1 - (ty - y0)), (y0 + 1, ty - y0)):
                    m2 += wxx * wyy * x["m_center"][clampi(cy, 0, rh - 1), clampi(cx, 0, rw - 1)]
            cands = [(sig[0], 0, m1), (sig[1], 1, m2), (sig[2], 2, m3)]
            # highest score first; ties: lower index first (no NaN scores in these tests)
            order = sorted(cands, key=lambda c: (-c[0], c[1]))
            q1 = min(max(1.1 * order[0][0] - 0.05, 0.0), 1.0)
            q2 = min(max(1.1 * order[1][0] - 0.05, 0.0), 1.0)
            w2 = q2 / max(q1 + q2, 1e-10)
            if w2 < 1 / 255:
                w1, w2 = 1.0, 0.0
            else:
                w1 = q1 / max(q1 + q2, 1e-10)
            Hs = []
            for _, _, m in order[:2]:
                qx, qy = xx + 0.5 + m[0], y + 0.5 + m[1]
                if (not x["reset"]) and 0 <= qx <= dw and 0 <= qy <= dh:
                    acc, ws = np.zeros(3), 0.0
                    cx0, cy0 = math.floor(qx - 0.5), math.floor(qy - 0.5)
                    for jj in range(-1, 3):
                        for ii in range(-1, 3):
                            if ii in (-1, 2) and jj in (-1, 2):
                                continue
                            tcx, tcy = cx0 + ii, cy0 + jj
                            w = cr((tcx + 0.5) - qx) * cr((tcy + 0.5) - qy)
                            acc += w * hist[clampi(tcy, 0, dh - 1), clampi(tcx, 0, dw - 1)]
                            ws += w
                    Hs.append(acc / ws)
                else:
                    Hs.append(np.array([E(ep * color[fy, fx, k]) for k in range(3)]))
            H = w1 * Hs[0] + (w2 * Hs[1] if w2 != 0 else 0.0)
            Cl = np.array([Dec(v) for v in cbar])
            Hl = np.array([Dec(v) for v in H]) / ep * e
            alpha = fmin(fmax(s * W * g, 0.0), 1.0)
            bh2, bo2 = alpha * (1 - bh) + bh, alpha * (1 - bo) + bo
            hh = Hl + (Cl - Hl) * bh2
            oo = Hl + (Cl - Hl) * bo2
            Hh[y, xx] = Cl if math.isnan(hh.sum()) else hh
            Oh[y, xx] = Cl if math.isnan(oo.sum()) else oo
    red = np.zeros((dh // 2, dw // 2, 3))
    for Y in range(dh // 2):
        for X in range(dw // 2):
            for k in range(3):
                M = (Hh[2 * Y, 2 * X, k] + Hh[2 * Y, 2 * X + 1, k] + Hh[2 * Y + 1, 2 * X, k]
                     + Hh[2 * Y + 1, 2 * X + 1, k]) / 4
                red[Y, X, k] = round(fmin(fmax(E(M) / 2.45, 0.0), 1.0) * 1023) / 1023
    enc = np.vectorize(E)
    disp = np.vectorize(lambda v: fmin(fmax(E(v), 0.0), 2.45))   # spec 06 draft 2: clamp(E(O^), 0, 2.45)
    return {"history": enc(Hh).astype(np.float16).astype(np.float32), "reduced": red.astype(np.float32),
            "display": disp(Oh).astype(np.float16).astype(np.float32)}


def bright(x):
    """Highlights above the display limit. The current colour alone cannot get there (E_in caps it at
    E(32752) ~ 2.34); a bright history scaled up by e / e_prev = 16 can (E(Dec(3) * 16) ~ 3.5)."""
    x = dict(x)
    x["color"] = (np.asarray(x["color"]) * 1e6).astype(np.float32)
    x["history"] = np.minimum(np.asarray(x["history"]) + 2.0, 3.0).astype(np.float32)
    x["exposure"], x["exposure_prev"] = 8.0, 0.5
    return x


def lanczos(t):
    """Spec 06 draft 3: Lanczos-3, evaluated at (k - delta) / 1.5 by the caller."""
    if abs(t) >= 3:
        return 0.0
    s = lambda z: 1.0 if z == 0 else math.sin(math.pi * z) / (math.pi * z)  # noqa: E731
    return s(t) * s(t / 3)


def slow_pass(row, n_out):
    out = []
    for xx in range(n_out):
        c = math.floor((xx + 0.5) * 1.5)
        d = (xx + 0.5) * 1.5 - (c + 0.5)
        L = [lanczos((k - d) / 1.5) for k in range(-2, 3)]
        vals = [row[clampi(c + k, 0, len(row) - 1)] for k in range(-2, 3)]
        v = sum(l * t for l, t in zip(L, vals)) / sum(L)
        out.append(min(max(v, min(vals)), max(vals)))
    return out


def slow_filter(display, e, O):
    ow, oh = O
    display = np.asarray(display, np.float64)
    dh, dw = display.shape[:2]
    h = np.array([[slow_pass(display[r, :, ch], ow) for ch in range(3)] for r in range(dh)])   # [dh, 3, ow]
    v = np.array([[slow_pass(h[:, ch, c], oh) for ch in range(3)] for c in range(ow)])       # [ow, 3, oh]
    out = np.vectorize(Dec)(v.transpose(2, 0, 1)) / (e or 1.0)
    return out.astype(np.float32)


class TestPostReference(unittest.TestCase):
    CASES = [
        dict(R=(12, 8), D=None, reset=False),
        dict(R=(12, 8), D=None, reset=True),
        dict(R=(10, 7), D=(23, 15), reset=False),     # odd D
        dict(R=(9, 6), D=(9, 6), reset=False),        # D = R
        dict(R=(8, 6), D=(30, 21), reset=False, motion=12.0),  # far reprojections, many invalid
        dict(R=(10, 8), D=None, reset=False, bright=True),    # highlights: display clamped at 2.45
    ]

    def test_against_loops(self):
        for i, c in enumerate(self.CASES):
            with self.subTest(case=i, **{k: v for k, v in c.items()}):
                rng = np.random.default_rng(10 + i)
                D = c["D"] or (None, None)
                x = P.random_inputs(rng, *c["R"], dw=D[0], dh=D[1], reset=c["reset"],
                                    motion_scale=c.get("motion", 3.0))
                if c.get("bright"):
                    x = bright(x)
                ref = P.reconstruct(x, 0.8)
                slow = slow_reconstruct(x, 0.8)
                for k in ("history", "display", "reduced"):
                    np.testing.assert_allclose(ref[k], slow[k], rtol=2e-3, atol=1e-6, err_msg=k)
                    # identical except where float rounding of an intermediate lands on an FP16 rounding midpoint
                    self.assertGreaterEqual(np.mean(ref[k] == slow[k]), 0.995, k)

    def test_filter_against_loops(self):
        rng = np.random.default_rng(5)
        for O in ((7, 5), (8, 6), (1, 1), (2, 3)):
            with self.subTest(O=O):
                dw, dh = P.recon_size_for(O[0]), P.recon_size_for(O[1])
                disp = P.f16(rng.uniform(0, 2.5, (dh, dw, 3)))
                np.testing.assert_allclose(P.output_filter(disp, 1.7, O), slow_filter(disp, 1.7, O), rtol=1e-6)

    def test_filter_phases(self):
        c, w = P.filter_weights(np.arange(6))
        np.testing.assert_array_equal(c, [0, 2, 3, 5, 6, 8])
        np.testing.assert_allclose(w[0], w[1][::-1])
        # spec 06 draft 3 weights (even outputs), as published with the draft
        np.testing.assert_allclose(w[0], [-0.090933, 0.112848, 0.639512, 0.409200, -0.070627], atol=6e-7)

    def test_filter_constant_and_clamp(self):
        disp = np.full((P.recon_size_for(9), P.recon_size_for(7), 3), 0.75, np.float32)
        out = P.output_filter(disp, 2.0, (7, 9))
        np.testing.assert_allclose(out, P.dec(0.75) / 2.0, rtol=1e-7)  # float32 output
        # an impulse: no ringing below the local minimum (0 here, then Dec clamps to 1e-4)
        img = np.zeros((P.recon_size_for(6), P.recon_size_for(6), 3))
        img[4, 4] = 2.0
        out = P.output_filter(img, 1.0, (6, 6))
        self.assertEqual(out.min(), np.float32(P.dec(0.0)))

    def test_display_limit(self):
        """Spec 06 draft 2: display = clamp(E(O^), 0, 2.45) before the FP16 store; the full history is not limited."""
        rng = np.random.default_rng(21)
        out = P.reconstruct(bright(P.random_inputs(rng, 12, 8)), 0.8)
        top = np.float32(np.float16(2.45))
        self.assertEqual(float(out["display"].max()), float(top))
        self.assertGreater(float(np.mean(out["display"] == top)), 0.1)
        self.assertGreater(float(out["history"].max()), float(top))       # history keeps the highlight
        # the lower limit 0 is a guard: Dec floors its argument at 1e-4, so O^ > 0 and E(O^) > 0 on every normal path
        self.assertGreaterEqual(float(out["display"].min()), 0.0)

    def test_edge_cases(self):
        rng = np.random.default_rng(3)
        x = P.random_inputs(rng, 12, 8)
        dw, dh = x["recon_size"]
        # reset: the history input is not read
        xr = dict(x, reset=True)
        a = P.reconstruct(xr, 0.5)
        b = P.reconstruct(dict(xr, history=np.full_like(x["history"], np.nan)), 0.5)
        for k in ("history", "display", "reduced"):
            np.testing.assert_array_equal(a[k], b[k])
        # NaN everywhere in colour, C and the history: outputs stay finite
        xn = dict(x, color=np.where(rng.random(x["color"].shape) < 0.2, np.nan, x["color"]),
                  C=np.where(rng.random(x["C"].shape) < 0.05, np.nan, x["C"]),
                  history=np.where(rng.random(x["history"].shape) < 0.05, np.nan, x["history"]))
        out = P.reconstruct(xn, 0.5)
        for k in ("history", "display", "reduced"):
            self.assertTrue(np.all(np.isfinite(out[k])), k)
        # infinite colour: E_in caps at 32752
        self.assertAlmostEqual(float(P.enc_in(np.inf)), float(P.enc(32752.0)))
        self.assertEqual(float(P.enc_in(np.nan)), 0.0)
        self.assertAlmostEqual(float(P.enc(np.nan)), 0.36 * math.log10(1e-4))
        self.assertAlmostEqual(float(P.dec(np.nan)), Dec(1e-4))
        # reprojection exactly on 0 and on D is valid
        xe = dict(x, m_near=np.zeros_like(x["m_near"]), m_center=np.zeros_like(x["m_center"]),
                  m_far=np.zeros_like(x["m_far"]))
        for mv in (-0.5, 0.5):
            q = P.history_sample(x["history"], np.array([0.5 + mv]), np.array([0.5]))
            self.assertTrue(np.all(np.isfinite(q)))
        self.assertTrue(np.all(np.isfinite(P.reconstruct(xe, 0.5)["history"])))

    def test_select_two(self):
        # rows: (score near, score center, score far) of one pixel -> (first, second)
        cases = [((0.5, 0.5, 0.5), (0, 1)), ((0.5, 0.2, 0.5), (0, 2)), ((0.2, 0.5, 0.5), (1, 2)),
                 ((0.1, 0.3, 0.2), (1, 2)), ((0.3, 0.1, 0.2), (0, 2)), ((0.1, 0.2, 0.3), (2, 1)),
                 ((np.nan,) * 3, (0, 1))]
        s = np.array([c[0] for c in cases], np.float64)
        f, g = P.select_two(s[:, 0], s[:, 1], s[:, 2])
        np.testing.assert_array_equal(f, [c[1][0] for c in cases])
        np.testing.assert_array_equal(g, [c[1][1] for c in cases])

if __name__ == "__main__":
    unittest.main()
