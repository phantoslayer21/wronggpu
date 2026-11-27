# SPDX-License-Identifier: GPL-3.0-only
"""Frame generation (#96): the shared non-network stages of the light path (clean/fg/reference/stages_common.py)
against plain scalar loop versions written separately here, on small synthetic grids; the menu-detection thresholds at
their exact boundaries; the packed-depth round trip; the hole rule against brute force.

    python -m unittest clean.fg.tests.test_fg_common_stages
"""
import math
import unittest
from fractions import Fraction

import numpy as np

from clean.fg.reference import pipeline as P
from clean.fg.reference import stages_common as S
from clean.reference.motion_hi import fma32

f = np.float32


def fma(a, b, c):
    return f(fma32(f(a), f(b), f(c)))


def rcp(x):
    with np.errstate(divide="ignore"):
        return f(f(1.0) / f(x))


def adiv(a, b):
    return f(f(a) * rcp(b))


def fmaxs(a, b):
    return f(np.fmax(f(a), f(b)))


def fmins(a, b):
    return f(np.fmin(f(a), f(b)))


def sat(x):
    return fmins(fmaxs(x, 0.0), 1.0)


def centre(i, n):
    return sat(f(f(i + 0.5) * rcp(n)))


def ftrunc(x):
    x = float(x)
    return 0 if math.isnan(x) else int(max(-2 ** 31, min(2 ** 31 - 1, math.trunc(max(-1e300, min(1e300, x))))))


def h(x):
    return np.asarray(np.asarray(x, np.float32).astype(np.float16)).astype(np.float32)[()]


def hb(x):
    """float16 array (for bit comparisons)."""
    return np.asarray(np.asarray(x, np.float32).astype(np.float16))


def tex_nearest(img, X, Y):
    hh, ww = img.shape[:2]
    ix = min(max(ftrunc(math.floor(f(f(X) * f(ww)))), 0), ww - 1)
    iy = min(max(ftrunc(math.floor(f(f(Y) * f(hh)))), 0), hh - 1)
    return img[iy, ix].astype(np.float32)


def tex_linear(img, X, Y):
    hh, ww = img.shape[:2]
    px, py = fma(X, ww, -0.5), fma(Y, hh, -0.5)
    fx, fy = f(math.floor(px)), f(math.floor(py))
    ax, ay = f(px - fx), f(py - fy)
    rx, ry = min(max(ftrunc(fx), -1), ww - 1), min(max(ftrunc(fy), -1), hh - 1)
    x0, x1, y0, y1 = max(rx, 0), min(rx + 1, ww - 1), max(ry, 0), min(ry + 1, hh - 1)
    out = []
    for c in range(img.shape[2]):
        t00, t10, t01, t11 = (f(img[y, x, c]) for y, x in ((y0, x0), (y0, x1), (y1, x0), (y1, x1)))
        r0 = fma(ax, f(t10 - t00), t00)
        r1 = fma(ax, f(t11 - t01), t01)
        out.append(fma(ay, f(r1 - r0), r0))
    return np.array(out, np.float32)


def key(z, s_k=1.0, inverted=False):
    d = f(z) if inverted else f(f(1.0) - f(z))
    return f(f(s_k) * rcp(fmaxs(d, f(1e-7))))


def sigma(a, b):
    z = f(f(a) - f(b))
    e = S.ex2(f(abs(z) * S.NEG_LOG2E))[()]
    return adiv(f(1.0) if a > b else e, f(e + f(1.0)))


def bits(x):
    return int(np.array(x, np.float32).view(np.uint32))


def rng_img(rng, shape, lo=0.0, hi=1.0, dtype=np.float16):
    return rng.uniform(lo, hi, shape).astype(dtype)


# ---- helpers --------------------------------------------------------------------------------------------------------

class Helpers(unittest.TestCase):
    def test_rcp_is_correctly_rounded(self):
        rng = np.random.default_rng(1)
        x = rng.uniform(0.001, 5000, 2000).astype(np.float32)
        r = S.rcp(x)
        for a, b in zip(x[:300], r[:300]):
            q = Fraction(1) / Fraction(float(a))
            lo, hi = np.nextafter(b, f(0)), np.nextafter(b, f(np.inf))
            self.assertLessEqual(abs(Fraction(float(b)) - q), abs(Fraction(float(lo)) - q))
            self.assertLessEqual(abs(Fraction(float(b)) - q), abs(Fraction(float(hi)) - q))

    def test_ex2(self):
        y = np.linspace(-30, 0, 20001).astype(np.float32)
        e = S.ex2(y)
        rel = np.abs(e.astype(np.float64) / np.exp2(y.astype(np.float64)) - 1)
        self.assertLess(rel.max(), 4e-7)
        self.assertEqual(S.ex2(f(0.0))[()], f(1.0))
        self.assertEqual(S.ex2(f(-3.0))[()], f(0.125))
        self.assertEqual(S.ex2(f(-127.5))[()], f(0.0))
        self.assertEqual(S.ex2(f(-np.inf))[()], f(0.0))
        self.assertTrue(np.isnan(S.ex2(f(np.nan))[()]))
        self.assertEqual(S.ex2(f(200.0))[()], f(np.inf))
        self.assertTrue(np.all(np.diff(e) >= 0))                    # monotone on this range

    def test_sigmoid(self):
        a = np.linspace(-20, 20, 401).astype(np.float32)
        s = S.sigmoid_pair(a, np.zeros_like(a))
        self.assertLess(np.abs(s - 1 / (1 + np.exp(-a.astype(np.float64)))).max(), 2e-7)
        self.assertEqual(S.sigmoid_pair(f(-1000), f(3))[()], f(0.0))
        self.assertEqual(S.sigmoid_pair(f(1000), f(3))[()], f(1.0))
        self.assertEqual(S.sigmoid_pair(f(2), f(2))[()], f(0.5))
        for x, y in ((0.3, -1.25), (-2.0, 0.5)):
            self.assertEqual(S.sigmoid_pair(f(x), f(y))[()], sigma(x, y))

    def test_sat_min_max_follow_minnum(self):
        self.assertEqual(S.sat(f(np.nan))[()], f(0.0))
        self.assertEqual(S.fmax(f(np.nan), f(2))[()], f(2.0))
        self.assertEqual(S.f2i(np.array([np.nan, 3.9, -3.9, 1e20, -np.inf]))[:].tolist(), [0, 3, -3, 2 ** 31 - 1, -2 ** 31])
        self.assertEqual(S.f2u(np.array([np.nan, -2.0, 7.9, 1e20])).tolist(), [0, 0, 7, 2 ** 32 - 1])

    def test_region_bounds(self):
        r = S.Region(37, 23, 3, 2, 30, 19)
        for lo, hi, b, e, rx in ((r.lox, r.hix, 3, 30, r.rx), (r.loy, r.hiy, 2, 19, r.ry)):
            ql, qh = Fraction(b * 2 + 1, 2) * Fraction(float(rx)), Fraction(2 * (b + e) - 1, 2) * Fraction(float(rx))
            self.assertGreaterEqual(Fraction(float(lo)), ql)
            self.assertLess(Fraction(float(np.nextafter(lo, f(0)))), ql)
            self.assertLessEqual(Fraction(float(hi)), qh)
            self.assertGreater(Fraction(float(np.nextafter(hi, f(1)))), qh)
        X, Y = r.coords(np.array([0.0, 1.0], np.float32), np.array([0.0, 1.0], np.float32))
        self.assertEqual((X[0], X[1], Y[0], Y[1]), (r.lox, r.hix, r.loy, r.hiy))
        u = f(0.37)
        Xb, _ = r.coords(np.array([u]), np.array([u]), bias=True)
        self.assertEqual(Xb[0], S.med3(fma(r.rx, fma(u, 30, 3), -2.0 ** -23), r.lox, r.hix))

    def test_texture_reads(self):
        rng = np.random.default_rng(2)
        img = rng_img(rng, (5, 7, 3), -1, 1)
        X = rng.uniform(-0.2, 1.2, 50).astype(np.float32)
        Y = rng.uniform(-0.2, 1.2, 50).astype(np.float32)
        lin = S.read_linear(img, X, Y)
        near = S.read_nearest(img, X, Y)
        for k in range(50):
            np.testing.assert_array_equal(lin[k], tex_linear(img, X[k], Y[k]))
            np.testing.assert_array_equal(near[k], tex_nearest(img, X[k], Y[k]))


# ---- menu detection -------------------------------------------------------------------------------------------------

def grid_points(reg):
    X, Y = S.detect_points(reg.bx, reg.by, reg.ew, reg.eh)
    return X.reshape(-1), Y.reshape(-1)


class MenuDetection(unittest.TestCase):
    def setUp(self):
        self.reg = S.Region(130, 66, 1, 1, 128, 64)      # 128 x 64: all 2048 points distinct
        X, Y = grid_points(self.reg)
        self.assertEqual(len(set(zip(X.tolist(), Y.tolist()))), 2048)
        self.X, self.Y = X, Y

    def alpha_with(self, k, channels=1):
        a = np.full((66, 130, channels), np.float16(0.69970703), np.float16)
        a[self.Y[:k], self.X[:k], channels - 1] = np.float16(0.75)
        return a

    def test_points(self):
        X, Y = S.detect_points(10, 20, 100, 50)
        self.assertEqual(X.shape, (32, 64))
        self.assertEqual((X[0, 0], X[0, -1], Y[0, 0], Y[-1, 0]), (10, 109, 20, 69))
        self.assertEqual(X[0, 1], 10 + 99 // 63)

    def test_alpha_threshold_boundary(self):
        for k, want in ((1433, False), (1434, True)):
            for ch in (1, 4):
                cnt = S.detect_count_alpha(self.alpha_with(k, ch), self.reg, ch - 1)
                self.assertEqual(cnt, k)
                self.assertEqual(S.detect_decide("alpha", cnt), want)

    def test_alpha_strict_comparison(self):
        a = np.full((66, 130, 1), np.float16(0.7), np.float16)      # 0.70019531 in float16
        thr = f(np.float16(0.7))
        self.assertEqual(S.detect_count_alpha(a, self.reg, 0, thr), 0)
        self.assertEqual(S.detect_count_alpha(a, self.reg, 0, f(0.7)), 2048)

    def test_hudless_threshold_boundary(self):
        hud = np.full((66, 130, 4), np.float16(0.5), np.float16)
        for k, want in ((614, False), (615, True)):
            col = hud.copy()
            col[..., :3] = np.float16(0.2)                             # opacity 0.6
            col[self.Y[:k], self.X[:k], :3] = np.float16(0.1)          # opacity 0.8
            cnt = S.detect_count_hudless(col, self.reg, hud, self.reg)
            self.assertEqual(cnt, k)
            self.assertEqual(S.detect_decide("hudless", cnt), want)
        self.assertEqual(float(f(0.3) * f(2048)), 614.4000244140625)

    def test_hudless_offset_read(self):
        hud = np.full((66, 130, 4), np.float16(0.5), np.float16)
        col = hud.copy()
        col[..., :3] = np.float16(0.1)
        hreg = S.Region(130, 66, 0, 0, 128, 64)                       # other base: bilinear region read
        self.assertEqual(S.detect_count_hudless(col, self.reg, hud, hreg), 2048)

    def test_opacity_formula(self):
        c, hh = np.array([[0.1, 0.9, 0.5]], np.float32), np.array([[0.5, 0.5, 0.5]], np.float32)
        d = S.hudless_opacity(c, hh)[0]
        want = max(max(adiv(f(0.5 - f(c[0, i])), f(f(0.5) + f(1e-10))),
                       adiv(f(0.5 - f(c[0, i])), f(f(f(0.5) - f(1.0)) - f(1e-10)))) for i in range(3))
        self.assertEqual(d, want)

    def test_latch_and_hysteresis(self):
        d = np.zeros(4, np.uint32)
        d = S.detect_update(d, "alpha", True, True)
        self.assertEqual(d.tolist(), [0xFFFFFFFF, 1, 0, 0])            # hysteresis off: [2] = 0
        d = S.detect_update(d, "alpha", True, False)
        self.assertEqual(d.tolist(), [0xFFFFFFFF, 1, 1, 0])
        d = S.detect_update(d, "alpha", False, False)
        self.assertEqual(d.tolist(), [0xFFFFFFFF, 0, 0, 0])            # latched one more game frame
        d = S.detect_update(d, "alpha", False, False)
        self.assertEqual(d.tolist(), [0, 0, 0, 0])
        d = S.detect_update(np.array([0, 1, 0, 0], np.uint32), "hudless", False, True)
        self.assertEqual(d.tolist(), [0, 0, 0, 0])                     # hysteresis off: no latch
        d = S.detect_update(np.array([0, 1, 0, 0], np.uint32), "hudless", True, False)
        self.assertEqual(d.tolist(), [0xFFFFFFFF, 1, 0, 0])


# ---- R3 -------------------------------------------------------------------------------------------------------------

def r3_scalar(colour, creg, hud, hreg, alpha, areg, ach, cp, fl, d, wd, hd):
    ho, wo = cp.shape[:2]
    case = S.r3_case(fl)
    L1, L2 = fl["L1"], fl["L2"]
    cc = np.zeros((ho, wo, 4), np.float16)
    uc = np.zeros((ho, wo, 4), np.float16)
    lc = np.zeros((ho, wo, 4), np.float16)
    s = np.zeros((ho, wo, 3), np.float32)
    diff = np.zeros((ho, wo), np.float32)
    flag = np.zeros((ho, wo), bool)
    for y in range(ho):
        for x in range(wo):
            u, v = centre(x, wo), centre(y, ho)
            c = colour[creg.by + y, creg.bx + x].astype(np.float32)
            hc = al = None
            if case in ("U", "H") or (case == "A" and L2):
                X, Y = hreg.coords(np.array([u]), np.array([v]))
                hc = tex_linear(hud, X[0], Y[0]) if L2 else tex_nearest(hud, X[0], Y[0])
            if case in ("U", "A"):
                X, Y = areg.coords(np.array([u]), np.array([v]))
                al = tex_nearest(alpha, X[0], Y[0])[ach]
            if case == "U":
                ap = (sat(al) if math.isfinite(al) else f(1.0)) if L1 else al
                U = [fma(hc[i], f(ap - f(1.0)), c[i]) for i in range(3)]
                nf = not all(math.isfinite(q) for q in U)
                bad = ap == 1 or nf or (min(U) <= -1) or any(U[i] < f(c[i] - f(hc[i] + hc[i])) for i in range(3))
                if L1 and ap == 0:
                    uc[y, x] = 0
                elif bad:
                    uc[y, x] = (c[0], c[1], c[2], 1)
                else:
                    uc[y, x] = (U[0], U[1], U[2], ap)
                cv = (hc[0], hc[1], hc[2], ap)
                s[y, x] = hc[:3]
            elif case == "A":
                fg = al > S.UI_ALPHA_THRESHOLD
                cv = (c[0], c[1], c[2], (1.0 if fg else 0.0) if L2 else al)
                if L2:
                    lc[y, x] = (hc[0], hc[1], hc[2], 0)
                flag[y, x] = fg
                s[y, x] = c[:3]
            elif case == "H":
                fg = max(abs(f(c[i] - hc[i])) for i in range(3)) > S.UI_DIFF_THRESHOLD
                cv = (c[0], c[1], c[2], (1.0 if fg else 0.0) if L2 else 0.0)
                if L2:
                    lc[y, x] = (hc[0], hc[1], hc[2], 0)
                flag[y, x] = fg
                s[y, x] = c[:3]
            else:
                cv = (c[0], c[1], c[2], 0.0)
                s[y, x] = c[:3]
            cc[y, x] = cv
            c16 = cc[y, x].astype(np.float32)
            diff[y, x] = abs(f(S.luma(c16)[()] - S.luma(cp[y, x].astype(np.float32))[()]))
    dc = np.zeros((hd, wd, 4), np.float16)
    ld = np.zeros((hd, wd, 1), np.float16)
    um = np.zeros((hd, wd, 1), np.float16)

    def at(a, x, y, zero):
        return a[y, x] if (x < wo and y < ho) else zero
    for Y in range(hd):
        for X in range(wd):
            sv = lambda i, j: at(s, d * X + i, d * Y + j, np.zeros(3, np.float32))
            dv = lambda i, j: at(diff, d * X + i, d * Y + j, f(0))
            if d == 2:
                tot = ((sv(0, 0) + sv(1, 0)).astype(np.float32) + (sv(0, 1) + sv(1, 1)).astype(np.float32))
                dcv = (tot * f(0.25)).astype(np.float32)
                ldv = fmins(fmins(dv(0, 0), dv(1, 0)), fmins(dv(0, 1), dv(1, 1)))
            else:
                a = {(i, j): (sv(i, j) + sv(i + 1, j)).astype(np.float32) for i in (0, 2) for j in range(4)}
                b = {(i, j): (a[i, j] + a[i, j + 1]).astype(np.float32) for i in (0, 2) for j in (0, 2)}
                e = ((b[0, 0] + b[2, 0]).astype(np.float32) + (b[0, 2] + b[2, 2]).astype(np.float32))
                dcv = (e.astype(np.float32) * f(1 / 16)).astype(np.float32)
                m = {(i, j): fmins(fmins(dv(i, j), dv(i + 1, j)), fmins(dv(i, j + 1), dv(i + 1, j + 1)))
                     for i in (0, 2) for j in (0, 2)}
                ldv = f(f(f(m[0, 0] + m[2, 0]) + f(m[0, 2] + m[2, 2])) * f(0.25))
            dc[Y, X] = (dcv[0], dcv[1], dcv[2], 0)
            ld[Y, X] = ldv
            um[Y, X] = 1.0 if any(at(flag, d * X + i, d * Y + j, False) for i in range(d) for j in range(d)) else 0.0
    return case, cc, uc, lc, dc, ld, um


class Downsample(unittest.TestCase):
    def run_case(self, fl, d, wo, ho, seed):
        rng = np.random.default_rng(seed)
        pad = 1
        tw, th = wo + 2 * pad, ho + 2 * pad
        reg = S.Region(tw, th, pad, pad, wo, ho)
        colour = rng_img(rng, (th, tw, 4))
        hud = rng_img(rng, (th, tw, 4))
        hud[::3] = colour[::3]                                         # some pixels without UI
        alpha = rng_img(rng, (th, tw, 1))
        alpha[0, :4, 0] = np.float16(np.inf)                           # a non-finite alpha (L1 clamp path)
        cp = rng_img(rng, (ho, wo, 4))
        wd, hd = wo // d, ho // d
        out = {k: np.full(s, np.float16(203.25), np.float16) for k, s in
               (("Cc", (ho, wo, 4)), ("Uc", (ho, wo, 4)), ("Lc", (ho, wo, 4)), ("Dc", (hd, wd, 4)),
                ("Ld", (hd, wd, 1)), ("Um", (hd, wd, 1)))}
        case = S.downsample_ui(colour, reg, hud, reg, alpha, reg, 0, cp, fl, d, wd, hd, out)
        ref = r3_scalar(colour, reg, hud, reg, alpha, reg, 0, cp, fl, d, wd, hd)
        self.assertEqual(case, ref[0])
        np.testing.assert_array_equal(out["Cc"].view(np.uint16), ref[1].view(np.uint16))
        if case == "U":
            np.testing.assert_array_equal(out["Uc"].view(np.uint16), ref[2].view(np.uint16))
        else:
            self.assertTrue(np.all(out["Uc"] == np.float16(203.25)))     # untouched
        if fl["L2"] and case in ("A", "H"):
            np.testing.assert_array_equal(out["Lc"].view(np.uint16), ref[3].view(np.uint16))
        else:
            self.assertTrue(np.all(out["Lc"] == np.float16(203.25)))
        np.testing.assert_array_equal(out["Dc"].view(np.uint16), ref[4].view(np.uint16))
        np.testing.assert_array_equal(out["Ld"].view(np.uint16), ref[5].view(np.uint16))
        if case in ("A", "H"):
            np.testing.assert_array_equal(out["Um"], ref[6])
        else:
            self.assertTrue(np.all(out["Um"] == np.float16(203.25)))
        return out

    def flags(self, **kw):
        fl = {"F1": False, "A": False, "Hd": False, "L1": False, "L2": False}
        fl.update(kw)
        return fl

    def test_cases_d2(self):
        self.run_case(self.flags(), 2, 9, 7, 1)
        self.run_case(self.flags(Hd=True), 2, 9, 7, 2)
        self.run_case(self.flags(Hd=True, L2=True), 2, 9, 7, 3)
        self.run_case(self.flags(A=True, Hd=True), 2, 9, 7, 4)
        o = self.run_case(self.flags(A=True, Hd=True, L2=True), 2, 9, 7, 5)
        self.assertTrue(set(np.unique(o["Cc"][..., 3]).tolist()) <= {0.0, 1.0})
        self.run_case(self.flags(F1=True, A=True, Hd=True), 2, 9, 7, 6)
        self.run_case(self.flags(F1=True, A=True, Hd=True, L1=True), 2, 9, 7, 7)

    def test_cases_d4(self):
        self.run_case(self.flags(Hd=True, L2=True), 4, 18, 13, 8)
        self.run_case(self.flags(F1=True, A=True, Hd=True, L1=True), 4, 18, 13, 9)
        self.run_case(self.flags(A=True, Hd=True), 4, 18, 13, 10)

    def test_frame_not_multiple_of_d(self):
        g = P.Geometry(18, 13, d=4)
        self.assertEqual((g.wd, g.hd), (4, 3))


# ---- R4 and the packed depth ----------------------------------------------------------------------------------------

class PackedDepth(unittest.TestCase):
    def test_round_trip(self):
        rng = np.random.default_rng(3)
        z = np.concatenate([rng.uniform(0, 1, 400), [0.0, 1.0, 2.0 ** -64, 3e-30, 2.0 ** -63, 0.5]]).astype(np.float32)
        for centre_ in (True, False):
            for oxp in (False, True):
                for oyp in (False, True):
                    n = z.size
                    Z = S.pack_depth(z, np.full(n, oxp), np.full(n, oyp), np.full(n, centre_))
                    zz, ox, oy = S.unpack_depth(Z, f(0.25), f(0.125))
                    b = S.bits32(z)
                    low29 = (b >> 29) & 1 == 0
                    if centre_:
                        np.testing.assert_array_equal(S.bits32(zz), b)
                        self.assertTrue(np.all(ox == 0) and np.all(oy == 0))
                    else:
                        exact = ~low29
                        np.testing.assert_array_equal(S.bits32(zz)[exact], b[exact])
                        # bit 29 clear: only the last bit may differ (it carries the x sign)
                        np.testing.assert_array_equal(S.bits32(zz)[low29] >> 1, b[low29] >> 1)
                        self.assertTrue(np.all(ox == (0.25 if oxp else -0.25)))
                        self.assertTrue(np.all(oy == (0.125 if oyp else -0.125)))

    def test_layout_bits(self):
        z = np.array([1.0], np.float32)                                 # 0x3F800000, bit 29 set
        self.assertEqual(int(S.pack_depth(z, np.array([True]), np.array([False]), np.array([True]))[0]), 0x9F800000)
        self.assertEqual(int(S.pack_depth(z, np.array([True]), np.array([True]), np.array([False]))[0]),
                         0x80000000 | 0x40000000 | (0x3F800000 & 0x3FFFFFFF))
        z = np.array([3e-30], np.float32)                               # bit 29 clear
        w = int(S.pack_depth(z, np.array([True]), np.array([False]), np.array([False]))[0])
        self.assertEqual(w & 0xC0000001, 0x40000001)


def mvec_scalar(depth, dreg, motion, mreg, w1, h1, inv, s_k, sx, sy):
    Z = np.zeros((h1, w1), np.uint32)
    M = np.zeros((h1, w1, 2), np.float32)
    dx, dy = rcp(dreg.ew), rcp(dreg.eh)
    offs = [(f(0), f(0)), (-dx, -dy), (dx, -dy), (-dx, dy), (dx, dy)]
    for y in range(h1):
        for x in range(w1):
            u, v = centre(x, w1), centre(y, h1)
            best, kb, sel = None, None, 0
            for i, (ox, oy) in enumerate(offs):
                X, Y = dreg.coords(np.array([f(u + ox)]), np.array([f(v + oy)]), bias=True)
                di = tex_nearest(depth, X[0], Y[0])[0]
                ki = key(di, s_k, inv)
                if i == 0 or ki < kb:
                    best, kb, sel = di, ki, i
            z = sat(best)
            ox, oy = offs[sel]
            Z[y, x] = S.pack_depth(np.array([z]), np.array([ox > 0]), np.array([oy > 0]), np.array([sel == 0]))[0]
            X, Y = mreg.coords(np.array([f(u + ox)]), np.array([f(v + oy)]), bias=True)
            m = tex_nearest(motion, X[0], Y[0])
            M[y, x] = (f(m[0] * f(f(sx) / f(mreg.ew))), f(m[1] * f(f(sy) / f(mreg.eh))))
    return Z, M


class MotionDepth(unittest.TestCase):
    def test_against_scalar(self):
        rng = np.random.default_rng(4)
        for inv in (False, True):
            depth = rng.uniform(0.0, 1.0, (14, 18, 1)).astype(np.float32)
            depth[3:6, 4:9] = 0.995 if not inv else 0.005               # a near block
            motion = rng.uniform(-0.05, 0.05, (14, 18, 2)).astype(np.float32)
            reg = S.Region(18, 14, 1, 1, 16, 12)
            out = {"Zc": np.zeros((6, 8), np.uint32), "Mv": np.zeros((6, 8, 2), np.float32)}
            S.mvec_process(depth, reg, motion, reg, 8, 6, inv, 1.0, 1.5, 0.75, out)
            Z, M = mvec_scalar(depth, reg, motion, reg, 8, 6, inv, 1.0, 1.5, 0.75)
            np.testing.assert_array_equal(out["Zc"], Z)
            np.testing.assert_array_equal(out["Mv"], M)
            _, ox, _ = S.unpack_depth(out["Zc"], f(1 / 16), f(1 / 12))
            self.assertTrue(np.any(ox != 0))                            # some neighbour samples chosen


# ---- R6 -------------------------------------------------------------------------------------------------------------

def camera_scalar(u, v, z, M, w1, h1):
    M = [f(m) for m in M]
    nx = fma(u, 2, -1)
    ny = f(f(1) - f(v + v))
    xp = f(M[12] + fma(M[8], z, fma(ny, M[4], f(M[0] * nx))))
    yp = f(M[13] + fma(M[9], z, fma(ny, M[5], f(M[1] * nx))))
    w = f(M[15] + fma(M[11], z, fma(ny, M[7], f(M[3] * nx))))
    if not w > 0:
        return f(10000), f(10000), w
    cx = f(f(nx - adiv(xp, abs(w))) * f(f(w1) * f(-0.5)))
    cy = f(f(ny - adiv(yp, abs(w))) * f(f(h1) * f(0.5)))
    return cx, cy, w


def sq(x, y):
    return fma(x, x, f(f(y) * f(y)))


def scatter_scalar(Zc, Mv, M, ho, steps):
    h1, w1 = Zc.shape
    T1, T2 = S.thresholds(ho)
    Sb = np.full(h1 * w1, S.EMPTY64, np.int64)
    K = np.zeros((h1, w1, 1), np.float16)
    for y in range(h1):
        for x in range(w1):
            u, v = centre(x, w1), centre(y, h1)
            z, ox, oy = (f(np.asarray(a).reshape(-1)[0]) for a in S.unpack_depth(np.array(Zc[y, x]), *steps))
            kap = key(z)
            cx, cy, w = camera_scalar(f(u + ox), f(v + oy), z, M, w1, h1)
            mx, my = f(Mv[y, x, 0] * f(w1)), f(Mv[y, x, 1] * f(h1))
            m2 = sq(mx, my)
            cons = (not w <= 0) and sq(f(mx - cx), f(my - cy)) <= fmaxs(adiv(m2, T1), 1)
            K[y, x] = 0 if cons else 1
            kk = f(kap + f(f(kap * w) - kap)) if cons else kap
            hi = (bits(kk) & 0xFFFFFFFE) | int(cons)
            lo = (int(np.float16(mx).view(np.uint16)) << 16) | int(np.float16(my).view(np.uint16))
            pay = np.array((hi << 32) | lo, np.uint64).view(np.int64)
            sx, sy = (1 if mx > 0 else -1), (1 if my > 0 else -1)
            bx, by = x + ftrunc(mx), y + ftrunc(my)
            for ex, ey in ((0, 0), (0, 1), (1, 0), (1, 1)):
                tx, ty = bx + ex * sx, by + ey * sy
                if not (0 <= tx < w1 and 0 <= ty < h1):
                    continue
                if (ex, ey) != (0, 0):
                    nx_, ny_ = min(max(x + ex * sx, 0), w1 - 1), min(max(y + ey * sy, 0), h1 - 1)
                    nb = Mv[ny_, nx_]
                    if not sq(f(f(nb[0] * f(w1)) - mx), f(f(nb[1] * f(h1)) - my)) < fmaxs(adiv(m2, T2), 1):
                        continue
                Sb[ty * w1 + tx] = min(Sb[ty * w1 + tx], pay)
    return Sb, K


def unpack_scalar(Sb, Zp, Kp, Mv, M2, steps, kmax=f(600), tau=f(40)):
    h1, w1 = Zp.shape
    F = np.zeros((h1, w1, 2), np.float16)
    E = np.zeros((h1, w1, 1), np.float16)
    Fl = np.zeros((h1, w1), np.uint32)
    for y in range(h1):
        for x in range(w1):
            u, v = centre(x, w1), centre(y, h1)
            e = int(Sb[y * w1 + x])
            empty = e == int(S.EMPTY64)
            ue = e & 0xFFFFFFFFFFFFFFFF
            hi, lo = ue >> 32, ue & 0xFFFFFFFF
            sx = 0.0 if empty else f(np.array(lo >> 16, np.uint16).view(np.float16))
            sy = 0.0 if empty else f(np.array(lo & 0xFFFF, np.uint16).view(np.float16))
            keyE = np.array(hi & 0xFFFFFFFE, np.uint32).view(np.float32)[()]
            z, ox, oy = (f(np.asarray(a).reshape(-1)[0]) for a in S.unpack_depth(np.array(Zp[y, x]), *steps))
            kp = f(Kp[y, x, 0])
            cx, cy, w = camera_scalar(f(u + ox), f(v + oy), z, M2, w1, h1)
            kap = key(z)
            if kp == 0:
                kap = f(kap * w)
            kE = f(0) if keyE <= 0 else (kmax if kmax <= keyE else keyE)
            kP = f(0) if kap <= 0 else (kmax if kap >= kmax else kap)
            mism = empty or abs(f(kE - kP)) > tau
            g = tex_nearest(Mv, f(u + adiv(cx, w1)), f(v + adiv(cy, h1)))
            rx, ry = fma(g[0], w1, cx), fma(g[1], h1, cy)
            dyn = not kp == 0
            if dyn:
                vx, vy = -f(sx), -f(sy)
            elif (hi & 1) or mism or sq(rx, ry) < 0.25:
                vx, vy = cx, cy
            else:
                vx, vy = -f(sx), -f(sy)
            F[y, x] = (vx, vy)
            E[y, x] = 1.0 if empty else 0.0
            Fl[y, x] = int(dyn) | (int(dyn and mism) << 1) | (int(empty) << 2)
    return F, E, Fl


def p2c_inputs(seed, w1=12, h1=9, ho=90):
    rng = np.random.default_rng(seed)
    z = rng.uniform(0.3, 0.7, (h1, w1)).astype(np.float32)
    z[2:5, 3:7] = 0.985                                                  # near block (keys ~ 67)
    oxp, oyp = rng.random((h1, w1)) < 0.5, rng.random((h1, w1)) < 0.5
    cen = rng.random((h1, w1)) < 0.5
    Zc = S.pack_depth(z, oxp, oyp, cen)
    tx = 0.25
    Mv = np.zeros((h1, w1, 2), np.float32)
    Mv[..., 0] = tx / 2
    Mv[2:5, 3:7] = (-2.5 / w1, 1.5 / h1)
    Mv += rng.uniform(-0.3, 0.3, Mv.shape).astype(np.float32) / np.array([w1, h1], np.float32)
    a = np.zeros(16, np.float32)
    a[0] = a[5] = a[15] = 1
    a[11] = 0.0625
    b = a.copy()
    a[12], b[12] = tx, -tx
    return Zc, Mv, a, b, ho


class PrevToCurrent(unittest.TestCase):
    def test_scatter(self):
        Zc, Mv, M, M2, ho = p2c_inputs(5)
        steps = (f(1 / 24), f(1 / 18))
        Sb = np.full(Zc.size, S.EMPTY64, np.int64)
        K = np.zeros(Zc.shape + (1,), np.float16)
        S.p2c_scatter(Zc, Mv, M, ho, 1.0, False, steps, Sb, K)
        Sr, Kr = scatter_scalar(Zc, Mv, M, ho, steps)
        np.testing.assert_array_equal(Sb, Sr)
        np.testing.assert_array_equal(K, Kr)
        self.assertTrue(np.any(K == 0) and np.any(K == 1))
        self.assertTrue(np.any(Sb == S.EMPTY64) and np.any(Sb != S.EMPTY64))

    def test_clear_value_is_the_largest_key(self):
        self.assertEqual(int(S.EMPTY64), 0x7FFFFFFF7FFFFFFF)
        pay = np.array((0x3F800000 << 32) | 0x1234, np.uint64).view(np.int64)
        self.assertLess(int(pay), int(S.EMPTY64))

    def test_unpack(self):
        Zc, Mv, M, M2, ho = p2c_inputs(6)
        steps = (f(1 / 24), f(1 / 18))
        Sb = np.full(Zc.size, S.EMPTY64, np.int64)
        K = np.zeros(Zc.shape + (1,), np.float16)
        S.p2c_scatter(Zc, Mv, M, ho, 1.0, False, steps, Sb, K)
        Zp, _, _, _, _ = p2c_inputs(7)
        rng = np.random.default_rng(8)
        Kp = (rng.random(Zc.shape + (1,)) < 0.4).astype(np.float16)
        h1, w1 = Zc.shape
        out = {"F": np.zeros((h1, w1, 2), np.float16), "E": np.zeros((h1, w1, 1), np.float16),
               "Fl": np.zeros((h1, w1), np.uint32), "Vf": np.zeros((h1, w1, 2), np.float32)}
        S.p2c_unpack(Sb, Zp, Kp, Mv, M2, 1.0, False, steps, 600.0, 40.0, out)
        F, E, Fl = unpack_scalar(Sb, Zp, Kp, Mv, M2, steps)
        np.testing.assert_array_equal(out["F"].view(np.uint16), F.view(np.uint16))
        np.testing.assert_array_equal(out["E"], E)
        np.testing.assert_array_equal(out["Fl"], Fl)
        np.testing.assert_array_equal(hb(out["Vf"]).view(np.uint16), F.view(np.uint16))
        self.assertTrue(len(set(out["Fl"].reshape(-1).tolist())) >= 3)

    def test_hole_rule_brute_force(self):
        rng = np.random.default_rng(9)
        for trial in range(6):
            h1, w1 = 9, 11
            Fl = (rng.random((h1, w1)) < 0.5).astype(np.uint32) | ((rng.random((h1, w1)) < (0.2 + 0.1 * trial))
                                                                 .astype(np.uint32) << 1)
            G = np.zeros((h1, w1), np.uint32)
            S.p2c_inpaint(Fl, G)
            for y in range(h1):
                for x in range(w1):
                    def b1(px, py):
                        return 0 <= px < w1 and 0 <= py < h1 and bool(Fl[py, px] & 2)
                    inp = all(any(b1(x + a + i, y + b + j) for i in (-1, 0, 1) for j in (-1, 0, 1))
                              for a in (-1, 0, 1) for b in (-1, 0, 1))
                    w = bool(Fl[y, x] & 1) and not inp
                    self.assertEqual(int(G[y, x]), (2 if w else 0) | (1 if inp else 0))

    def test_hole_rule_needs_all_nine_windows(self):
        # one bit-1 pixel per corner of p's 3 x 3 neighbourhood minus the last: eight windows hold one, the ninth not
        Fl = np.zeros((7, 7), np.uint32)
        for (x, y) in ((2, 2), (4, 2), (2, 4)):
            Fl[y, x] = 2
        G = np.zeros((7, 7), np.uint32)
        S.p2c_inpaint(Fl, G)
        self.assertEqual(int(G[3, 3]) & 1, 0)
        Fl[4, 4] = 2
        S.p2c_inpaint(Fl, G)
        self.assertEqual(int(G[3, 3]) & 1, 1)

    def test_pyramid_levels(self):
        self.assertEqual(len(S.pyramid_levels(960, 540)), 11)
        self.assertEqual(len(S.pyramid_levels(1280, 720)), 12)
        lv = S.pyramid_levels(8, 6)
        self.assertEqual(len(lv), 9)
        self.assertEqual(lv[-1], (1, 1))
        self.assertEqual(S.pull_top(S.pyramid_levels(300, 50)), 8)
        self.assertEqual(S.pull_top(S.pyramid_levels(64, 36)), 4)

    def pull_scalar(self, G, Vf, top):
        h1, w1 = G.shape
        w = ((G >> 1) & 1).astype(np.float32)
        base = np.stack([(w * Vf[..., 0]).astype(np.float32), (w * Vf[..., 1]).astype(np.float32), w], -1)

        def tree(a):
            hh, ww = a.shape[:2]
            out = np.zeros(((hh + 1) // 2, (ww + 1) // 2, 3), np.float32)
            for Y in range(out.shape[0]):
                for X in range(out.shape[1]):
                    def c(i, j):
                        x, y = 2 * X + i, 2 * Y + j
                        return a[y, x] if (x < ww and y < hh) else np.zeros(3, np.float32)
                    out[Y, X] = ((c(0, 0) + c(1, 0)).astype(np.float32) + (c(0, 1) + c(1, 1)).astype(np.float32))
            return out

        def elem(s, n):
            den = fmaxs(s[..., 2], f(1e-12))
            return hb(np.stack([s[..., 0] * rcp(den), s[..., 1] * rcp(den), s[..., 2] * rcp(n)], -1))
        lv, cur = {}, base
        for l in range(1, min(top, 4) + 1):
            cur = tree(cur)
            lv[l] = elem(cur, 4 ** l)
        if top > 4:
            e = lv[4].astype(np.float32)
            cur = np.stack([(e[..., 0] * e[..., 2]).astype(np.float32), (e[..., 1] * e[..., 2]).astype(np.float32),
                            e[..., 2]], -1)
            for l in range(5, top + 1):
                cur = tree(cur)
                lv[l] = elem(cur, 4 ** (l - 4))
        return lv

    def test_pull_and_push(self):
        rng = np.random.default_rng(10)
        for (w1, h1) in ((13, 9), (300, 7)):
            top = S.pull_top(S.pyramid_levels(w1, h1))
            G = rng.integers(0, 4, (h1, w1)).astype(np.uint32)
            G[:, : w1 // 3] &= 1                                            # a region without weight
            Vf = rng.uniform(-6, 6, (h1, w1, 2)).astype(np.float32)
            lv = S.p2c_pull(G, Vf, top)
            ref = self.pull_scalar(G, Vf, top)
            self.assertEqual(sorted(lv), list(range(1, top + 1)))
            for l in lv:
                np.testing.assert_array_equal(lv[l].view(np.uint16), ref[l].view(np.uint16))
            # push: scalar version of one upsample
            filled = {l: a.copy() for l, a in lv.items()}
            F = hb(rng.uniform(-3, 3, (h1, w1, 2)))
            F0 = F.copy()
            S.p2c_push(filled, top, G, F)
            for l in range(top - 1, 0, -1):
                par = filled[l + 1].astype(np.float32)
                hp, wp = par.shape[:2]
                for (qy, qx) in ((0, 0), (lv[l].shape[0] - 1, lv[l].shape[1] - 1)):
                    own = lv[l][qy, qx]
                    px, py = qx // 2, qy // 2
                    nx = min(max(px + (1 if qx & 1 else -1), 0), wp - 1)
                    ny = min(max(py + (1 if qy & 1 else -1), 0), hp - 1)
                    taps = ((par[py, px], 0.5625), (par[py, nx], 0.1875), (par[ny, px], 0.1875), (par[ny, nx], 0.0625))
                    ws, v0, v1 = f(0), f(0), f(0)
                    for t, k in taps:
                        ws = fma(t[2], k, ws)
                        v0 = fma(f(t[0] * t[2]), k, v0)
                        v1 = fma(f(t[1] * t[2]), k, v1)
                    den = fmins(fmaxs(ws, f(1e-12)), f(1))
                    want = own if not f(own[2]) <= 0 else hb(np.array([v0 * rcp(den), v1 * rcp(den), ws], np.float32))
                    np.testing.assert_array_equal(filled[l][qy, qx].view(np.uint16), np.asarray(want).view(np.uint16))
            inp = (G & 1).astype(bool)
            np.testing.assert_array_equal(F[~inp].view(np.uint16), F0[~inp].view(np.uint16))
            if inp.any():
                vu, _ = S.push_upsample(filled[1], h1, w1)
                np.testing.assert_array_equal(F[inp].view(np.uint16), hb(vu)[inp].view(np.uint16))


# ---- R7 / R8 / R9 ----------------------------------------------------------------------------------------------------

class HoleMaskQuality(unittest.TestCase):
    def test_hole_mask(self):
        rng = np.random.default_rng(11)
        F = hb(rng.uniform(-4, 4, (9, 12, 2)))
        Hc = np.zeros((9, 12, 1), np.float16)
        S.hole_mask(F, 12, 9, Hc)
        want = np.ones((9, 12), np.float16)
        for y in range(9):
            for x in range(12):
                m = tex_nearest(F, centre(x, 12), centre(y, 9))
                mx, my = f(adiv(m[0], 12) * f(12)), f(adiv(m[1], 9) * f(9))
                sx, sy = (1 if mx > 0 else -1), (1 if my > 0 else -1)
                for ex, ey in ((0, 0), (0, 1), (1, 0), (1, 1)):
                    tx, ty = x + ftrunc(mx) + ex * sx, y + ftrunc(my) + ey * sy
                    if 0 <= tx < 12 and 0 <= ty < 9:
                        want[ty, tx] = 0
        np.testing.assert_array_equal(Hc[..., 0], want)
        self.assertTrue(np.any(want == 1) and np.any(want == 0))

    def test_quality_init(self):
        rng = np.random.default_rng(12)
        Dc = rng_img(rng, (4, 5, 4), 0, 1.5)
        Q = np.zeros((4, 5, 2), np.float16)
        S.quality_init(Dc, Q)
        for y in range(4):
            for x in range(5):
                c = Dc[y, x].astype(np.float32)
                self.assertEqual(Q[y, x, 0], h(sat(adiv(f(f(c[0] + c[1]) + c[2]), 3))))
                self.assertEqual(Q[y, x, 1], 0)


class Net1InputOutput(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(13)
        self.hd, self.wd, self.h1, self.w1 = 10, 14, 10, 14
        hd, wd, h1, w1 = self.hd, self.wd, self.h1, self.w1
        self.Ld = rng_img(rng, (hd, wd, 1))
        self.Dc, self.Dp = rng_img(rng, (hd, wd, 4)), rng_img(rng, (hd, wd, 4))
        self.F = hb(rng.uniform(-3, 3, (h1, w1, 2)))
        self.Mv = rng.uniform(-0.1, 0.1, (h1, w1, 2)).astype(np.float32)
        self.E = (rng.random((h1, w1, 1)) < 0.3).astype(np.float16)
        self.Hc = (rng.random((hd, wd, 1)) < 0.3).astype(np.float16)
        self.Hp = (rng.random((hd, wd, 1)) < 0.5).astype(np.float16)
        self.Qp = rng_img(rng, (hd, wd, 2))
        self.m = rng_img(rng, (hd, wd, 1))

    def test_values(self):
        vals, qc, om = S.net1_values(self.Ld, self.Dc, self.Dp, self.F, self.Mv, self.E, self.Hc, self.Hp, self.Qp,
                                     self.m)
        for (y, x) in ((0, 0), (3, 7), (9, 13), (5, 1)):
            u, v = centre(x, self.wd), centre(y, self.hd)
            dc, dp = self.Dc[y, x].astype(np.float32), self.Dp[y, x].astype(np.float32)
            fv = tex_nearest(self.F, u, v)
            a = tex_linear(self.Dc, f(u + adiv(fv[0], self.w1)), f(v + adiv(fv[1], self.h1)))
            d = np.abs((a[:3] - dp[:3]).astype(np.float32))
            v2 = sat(adiv(f(d[2] + f(d[0] + d[1])), 3))
            mv = tex_nearest(self.Mv, u, v)
            b = tex_linear(self.Dp, f(u + mv[0]), f(v + mv[1]))
            d = np.abs((b[:3] - dc[:3]).astype(np.float32))
            v6 = sat(adiv(f(d[2] + f(d[0] + d[1])), 3))
            e = tex_nearest(self.E, u, v)[0]
            hc = tex_nearest(self.Hc, u, v)[0]
            mm = f(self.m[y, x, 0])
            want = [f(self.Ld[y, x, 0]), S.luma(dp)[()], v2, f(self.Qp[y, x, 0]), e, S.luma(dc)[()], v6, hc, mm,
                    f(1 - mm)]
            np.testing.assert_array_equal(vals[:, y, x], np.array(want, np.float32))
            np.testing.assert_array_equal(qc[y, x], [v6, v2])
            self.assertEqual(om[y, x], sat(f(f(e * tex_nearest(self.Hp, u, v)[0]) * hc)))

    def test_pack_matches_fold(self):
        rng = np.random.default_rng(14)
        vals = rng.uniform(0, 1, (10, 12, 16)).astype(np.float32)
        cells = S.net1_pack(vals, 8, 6, 0, 0)
        ref = np.asarray(__import__("clean.fg.reference.net1", fromlist=["fold"]).fold(vals.astype(np.float64)))
        np.testing.assert_array_equal(cells.transpose(2, 0, 1)[:, :6, :8].astype(np.float64), ref)
        padded = S.net1_pack(vals, 9, 8, 1, 2)                              # centring pad in cells
        np.testing.assert_array_equal(padded[2:8, 1:9], cells)
        self.assertTrue(np.all(padded[:2] == 0) and np.all(padded[:, :1] == 0))
        self.assertTrue(np.all(cells[..., 40:] == 0))

    def test_output(self):
        rng = np.random.default_rng(15)
        W0, H0, pl, pt, wd, hd = 9, 7, 1, 1, 13, 11                       # odd image: the last pixel clamps
        o16 = hb(rng.uniform(-4, 4, (H0, W0, 16)))
        Om = (rng.random((hd, wd, 1)) < 0.3).astype(np.float16)
        Um = (rng.random((hd, wd, 1)) < 0.3).astype(np.float16)
        Ump = (rng.random((hd, wd, 1)) < 0.3).astype(np.float16)
        for forcing in (False, True):
            out = {"NA": np.zeros((hd, wd, 2), np.float16), "NB": np.zeros((hd, wd, 2), np.float16),
                   "m": np.zeros((hd, wd, 1), np.float16)}
            S.net1_output(o16, Om, Um, Ump, wd, hd, pl, pt, forcing, out)
            for y in range(hd):
                for x in range(wd):
                    cx, cy = pl + min(x >> 1, wd // 2 - 1), pt + min(y >> 1, hd // 2 - 1)
                    q = 2 * (y & 1) + (x & 1)
                    o = [f(o16[cy, cx, 4 * i + q]) for i in range(4)]
                    om = f(Om[y, x, 0])
                    o[0] = fma(f(S.PUSH - o[0]), om, o[0])
                    o[2] = fma(f(S.PUSH - o[2]), om, o[2])
                    mm = sigma(o[2], o[3])
                    if forcing and f(Ump[y, x, 0]) > f(0.99):
                        o[0], o[1] = f(o[0] - f(1000)), f(o[1] + f(1000))
                    if forcing and f(Um[y, x, 0]) > f(0.99):
                        o[2], o[3] = f(o[2] - f(1000)), f(o[3] + f(1000))
                    np.testing.assert_array_equal(out["NA"][y, x].view(np.uint16), hb(np.array(o[:2])).view(np.uint16))
                    np.testing.assert_array_equal(out["NB"][y, x].view(np.uint16), hb(np.array(o[2:])).view(np.uint16))
                    self.assertEqual(out["m"][y, x, 0].view(np.uint16), hb(mm).view(np.uint16))


# ---- G2 / G3 --------------------------------------------------------------------------------------------------------

def interm_scalar(dxy, keys, t2, tau, depth_test):
    h1, w1 = keys.shape
    out = np.full(h1 * w1, S.EMPTY64, np.int64)

    def rd(a, x, y):
        return a[min(max(y, 0), h1 - 1), min(max(x, 0), w1 - 1)]
    for y in range(h1):
        for x in range(w1):
            kp, dp = f(keys[y, x]), dxy[y, x]
            ks = [f(rd(keys, x + i, y + j)) for j in (-1, 0, 1) for i in (-1, 0, 1)]
            qx = [f(f(f(x + i) + f(0.5)) + rd(dxy, x + i, y + j)[0]) for j in (-1, 0, 1) for i in (-1, 0, 1)]
            qy = [f(f(f(y + j) + f(0.5)) + rd(dxy, x + i, y + j)[1]) for j in (-1, 0, 1) for i in (-1, 0, 1)]
            qpx, qpy = f(f(f(x) + f(0.5)) + dp[0]), f(f(f(y) + f(0.5)) + dp[1])
            if f(max(ks) - kp) > tau or f(kp - min(ks)) > tau:
                rx = ry = 2
            else:
                def rad(q, qp):
                    a = max(abs(f(max(q + [f(-10000)]) - qp)), abs(f(min(q + [f(10000)]) - qp)))
                    return int(min(max(math.ceil(f(a + f(0.5))), 2), 5))
                rx, ry = rad(qx, qpx), rad(qy, qpy)
            hi = bits(kp)
            lo = (int(np.float16(adiv(-dp[0], w1)).view(np.uint16)) << 16) | int(np.float16(adiv(-dp[1], h1))
                                                                                .view(np.uint16))
            pay = int(np.array((hi << 32) | lo, np.uint64).view(np.int64))
            thr = fmaxs(adiv(sq(dp[0], dp[1]), t2), 1)
            sx, sy = (1 if dp[0] > 0 else -1), (1 if dp[1] > 0 else -1)
            bx, by = x + ftrunc(dp[0]), y + ftrunc(dp[1])
            for j in range(ry):
                for i in range(rx):
                    tx, ty = bx + i * sx, by + j * sy
                    if not (0 <= tx < w1 and 0 <= ty < h1):
                        continue
                    nk, nd = f(rd(keys, x + i * sx, y + j * sy)), rd(dxy, x + i * sx, y + j * sy)
                    dd = sq(f(nd[0] - dp[0]), f(nd[1] - dp[1]))
                    if depth_test:
                        kt = f(min(max(int(nk), 0), 2 ** 32 - 1))
                        if not (abs(f(kt - kp)) < 3) and not (dd < thr):
                            continue
                    elif not dd < thr:
                        continue
                    out[ty * w1 + tx] = min(out[ty * w1 + tx], pay)
    return out


class IntermediateTime(unittest.TestCase):
    def test_both_passes(self):
        Zc, Mv, _, _, ho = p2c_inputs(16)
        Zp, _, _, _, _ = p2c_inputs(17)
        rng = np.random.default_rng(18)
        F = hb(rng.uniform(-4, 4, Zc.shape + (2,)))
        _, T2 = S.thresholds(ho)
        for t in (P.CFG.intermediate_time(1, 1), P.CFG.intermediate_time(5, 1), P.CFG.intermediate_time(5, 5)):
            da, ka, db, kb = S.interm_inputs(Mv, Zc, F, Zp, t, 1.0, False)
            np.testing.assert_array_equal(da[..., 0], (f(1 - t) * (Mv[..., 0] * f(Zc.shape[1])).astype(np.float32))
                                          .astype(np.float32))
            np.testing.assert_array_equal(db[..., 1], (f(t) * F[..., 1].astype(np.float32)).astype(np.float32))
            for dxy, keys, dt in ((da, ka, True), (db, kb, False)):
                out = np.full(keys.size, S.EMPTY64, np.int64)
                S.interm_scatter(dxy, keys, T2, f(40), dt, out)
                np.testing.assert_array_equal(out, interm_scalar(dxy, keys, T2, f(40), dt))
                self.assertTrue(np.any(out != S.EMPTY64))

    def test_footprint_grows_with_spread(self):
        keys = np.ones((12, 12), np.float32)
        d = np.zeros((12, 12, 2), np.float32)
        d[:, 6:, 0] = 4.0                                                   # a spread of 4 pixels to the right
        out = np.full(144, S.EMPTY64, np.int64)
        S.interm_scatter(d, keys, f(1.0), f(40), False, out)
        self.assertGreater(int(np.sum(out != S.EMPTY64)), 0)


class BlendWeights(unittest.TestCase):
    def test_against_scalar(self):
        rng = np.random.default_rng(19)
        h1, w1, hd, wd = 6, 8, 6, 8
        NA, NB = hb(rng.uniform(-3, 3, (hd, wd, 2))), hb(rng.uniform(-3, 3, (hd, wd, 2)))
        MA = np.full(h1 * w1, S.EMPTY64, np.int64)
        MB = MA.copy()
        for buf in (MA, MB):
            for k in rng.choice(h1 * w1, 30, replace=False):
                vx, vy = np.float16(rng.uniform(-0.2, 0.2)), np.float16(rng.uniform(-0.2, 0.2))
                lo = (int(vx.view(np.uint16)) << 16) | int(vy.view(np.uint16))
                buf[k] = (0x3F800000 << 32) | lo
        W = np.zeros((h1, w1, 2), np.float16)
        S.blend_weights(MA, MB, NA, NB, w1, h1, W)
        for y in range(h1):
            for x in range(w1):
                u, v = centre(x, w1), centre(y, h1)
                a, ea = S.vec_lookup(MA, w1, h1, np.array([u]), np.array([v]))
                b, eb = S.vec_lookup(MB, w1, h1, np.array([u]), np.array([v]))
                self.assertEqual(bool(ea[0]), MA[y * w1 + x] == S.EMPTY64)
                A1 = tex_linear(NA, f(u + b[0, 0]), f(v + b[0, 1]))[0]
                B1 = tex_linear(NB, f(u + a[0, 0]), f(v + a[0, 1]))[0]
                self.assertEqual(W[y, x, 0], h(sigma(A1, f(NA[y, x, 1]))))
                self.assertEqual(W[y, x, 1], h(sigma(B1, f(NB[y, x, 1]))))


if __name__ == "__main__":
    unittest.main()
