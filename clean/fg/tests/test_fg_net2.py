# SPDX-License-Identifier: MIT
"""Frame generation (#96): net2 with the delivered wiring and arithmetic (clean/fg/reference/net2.py), on synthetic
weights and inputs, against versions written separately here: exact rationals for the float16 operations, one
convolution and the warp sample; scalar loops for the upsample and the pool; a hand-composed 28-launch network whose
convolutions use fp16acc.acc_dot16 (not the matrix-product path) and whose fused multiply-adds use exact integer
accumulation (not the two-sum path).

    python -m unittest clean.fg.tests.test_fg_net2
"""
import unittest
from fractions import Fraction

import numpy as np

from clean.fg import layout as L
from clean.fg.reference import fp16acc
from clean.fg.reference import net2 as N
from clean.fg.tests.test_fg_fp16acc import rn16_exact

S = float(np.uint16(0x211F).view(np.float16))


def f16(x):
    return np.asarray(x, dtype=np.float64).astype(np.float16).astype(np.float64)


def bits(x):
    return np.asarray(x, np.float64).astype(np.float16).view(np.uint16)


def all_f16():
    return np.arange(65536, dtype=np.uint32).astype(np.uint16).view(np.float16).astype(np.float64)


def synthetic_params(rng, wscale=2.0, bscale=0.5):
    out = {}
    for l in L.NET2:
        s = wscale / np.sqrt(9 * l.cin_used)
        out[l.name] = (f16(rng.uniform(-s, s, (l.out, 3, 3, l.cin))), f16(rng.uniform(-bscale, bscale, l.out)))
    return out


def fma_exact(a, b, c):
    """RN16(a b + c) by rationals, with IEEE's zero sign (an exact zero is -0 only when a b and c are both -0)."""
    q = Fraction(a) * Fraction(b) + Fraction(c)
    if q == 0:
        neg = (np.signbit(a) ^ np.signbit(b)) and np.signbit(c) and (a == 0 or b == 0) and c == 0
        return -0.0 if neg else 0.0
    return rn16_exact(q)


def fma_int(a, b, c):
    """RN16(a b + c) by fp16acc's integer accumulation (one-term chunk) with IEEE's zero sign: an independent version
    of net2.fma16."""
    a, b, c = np.broadcast_arrays(*(np.asarray(v, np.float64) for v in (a, b, c)))
    r = fp16acc.acc_dot16(c, a[..., None], b[..., None])
    zero_terms = ((a == 0) | (b == 0)) & (c == 0)
    neg = np.signbit(a) ^ np.signbit(b)
    neg &= np.signbit(c)
    return np.where(zero_terms & (r == 0), np.where(neg, -0.0, 0.0), r)


def axis_scalar(n_in):
    out = []
    for o in range(2 * n_in):
        s = np.float32((np.float32(o) - np.float32(0.5)) * np.float32(0.5))
        i0, i1 = int(np.trunc(s)), int(np.trunc(np.float32(s + np.float32(1.0))))
        if i1 >= n_in:
            i0 = i1 = n_in - 1
        if s < 0:
            i0 = i1 = 0
        out.append((i0, i1, float(f16(np.float32((np.float32(i0) - s) + np.float32(1.0)))),
                    float(f16(np.float32(s - np.float32(i0))))))
    return out


def up2_scalar(p):
    """The 2x half-pixel rule one value at a time with exact rationals."""
    p = f16(p)
    hi, wi, C = p.shape
    ax, ay = axis_scalar(wi), axis_scalar(hi)
    out = np.zeros((2 * hi, 2 * wi, C))
    for oy, (y0, y1, wy0, wy1) in enumerate(ay):
        for ox, (x0, x1, wx0, wx1) in enumerate(ax):
            for c in range(C):
                top = fma_exact(wx1, p[y0, x1, c], float(f16(wx0 * p[y0, x0, c])))
                bot = fma_exact(wx1, p[y1, x1, c], float(f16(wx0 * p[y1, x0, c])))
                out[oy, ox, c] = fma_exact(wy1, bot, float(f16(wy0 * top)))
    return out


def conv_exact(x, w, b, oy, ox, o):
    """One convolution output by rationals: chunks per tap (row-major), 16-channel blocks inside a tap."""
    H, W, cin = x.shape
    cw = w.shape[3]
    acc = float(f16(b[o]))
    for ky in range(3):
        for kx in range(3):
            sy, sx = oy + ky - 1, ox + kx - 1
            for c0 in range(0, cw, 16):
                q = Fraction(acc)
                for c in range(c0, min(cw, c0 + 16)):
                    xv = x[sy, sx, c] if (0 <= sy < H and 0 <= sx < W and c < cin) else 0.0
                    q += Fraction(float(w[o, ky, kx, c])) * Fraction(float(xv))
                acc = rn16_exact(q)
    return acc


def conv_dot(x, w, b):
    """The convolution with acc_dot16 per chunk (tap-major), broadcasting over every pixel and output channel."""
    x, w = f16(x), f16(w)
    H, W, cin = x.shape
    cout, _, _, cw = w.shape
    xp = np.zeros((H + 2, W + 2, cw))
    xp[1:H + 1, 1:W + 1, :cin] = x
    acc = np.broadcast_to(f16(b), (H, W, cout)).copy()
    for ky in range(3):
        for kx in range(3):
            for c0 in range(0, cw, 16):
                c1 = min(cw, c0 + 16)
                xs = xp[ky:ky + H, kx:kx + W, c0:c1][:, :, None, :]
                acc = fp16acc.acc_dot16(acc, xs, w[None, None, :, ky, kx, c0:c1], axis=-1)
    return acc


# ---- grid -------------------------------------------------------------------------------------------------------

class Grid(unittest.TestCase):
    TABLE = [((1280, 720), 2, (640, 368)), ((1920, 1080), 4, (480, 272)), ((2560, 1440), 4, (640, 368)),
             ((3440, 1440), 4, (864, 368)), ((3840, 2160), 8, (480, 272)), ((5120, 2880), 8, (640, 368)),
             ((7680, 4320), 16, (480, 272))]

    def test_table(self):
        for (wo, ho), k, (w2, h2) in self.TABLE:
            self.assertEqual(N.grid(wo, ho), (k, w2, h2), (wo, ho))

    def test_rule_brute(self):
        rng = np.random.default_rng(3)
        for wo, ho in zip(rng.integers(4, 8193, 400), rng.integers(4, 7681, 400)):
            k, w2, h2 = N.grid(int(wo), int(ho))
            cands = [kk for kk in (2 ** i for i in range(14)) if ho // kk <= 480]
            self.assertEqual(k, cands[0])
            self.assertEqual((w2 % 16, h2 % 16), (0, 0))
            self.assertTrue(w2 * k >= wo > w2 * k - 16 * k and h2 * k >= ho > h2 * k - 16 * k)
            r, bt = N.padding(int(wo), int(ho))
            self.assertEqual((wo + r, ho + bt), (w2 * k, h2 * k))

    def test_range(self):
        for wo, ho in ((3, 100), (8193, 100), (100, 3), (100, 7681)):
            with self.assertRaises(ValueError):
                N.grid(wo, ho)
        N.grid(4, 4)
        N.grid(8192, 7680)

    def test_extents(self):
        e = N.extents(480, 272)
        self.assertEqual(e["r0"], (34, 60))
        self.assertEqual(e["g0"], (68, 120))
        self.assertEqual(e["A4"], (136, 240))
        self.assertEqual(e["out"], (272, 480))
        with self.assertRaises(ValueError):
            N.extents(40, 32)


# ---- float16 operations -------------------------------------------------------------------------------------------

class Ops(unittest.TestCase):
    def test_leaky_clamp_all_values(self):
        v = all_f16()
        fin = np.isfinite(v)
        got_l, got_a, got_c = N.leaky(v), N.act(v), N.clamp6(v)
        for i in np.flatnonzero(fin):
            x = float(v[i])
            t = rn16_exact(Fraction(S) * Fraction(x)) if x != 0 else x
            if t == 0 and x < 0:
                t = -0.0
            want = max(t, x)
            self.assertEqual(bits(got_l[i]), bits(want), x)
            self.assertEqual(bits(got_a[i]), bits(min(max(want, -6.0), 6.0)), x)
            self.assertEqual(bits(got_c[i]), bits(min(max(x, -6.0), 6.0)), x)
        self.assertTrue(np.all(np.isnan(got_a[np.isnan(v)])))
        self.assertEqual(float(N.leaky(np.inf)), np.inf)
        self.assertEqual(float(N.act(-np.inf)), -6.0)
        self.assertEqual(int(bits(N.LEAKY_SLOPE)), 0x211F)

    def test_fma_random_and_midpoints(self):
        rng = np.random.default_rng(5)
        n = 4000
        a = f16(rng.uniform(-1, 1, n) * 2.0 ** rng.integers(-24, 16, n))
        b = f16(rng.uniform(-1, 1, n) * 2.0 ** rng.integers(-24, 16, n))
        c = f16(rng.uniform(-1, 1, n) * 2.0 ** rng.integers(-24, 16, n))
        # constructed: products exactly at a float16 midpoint, plus the smallest nudges either way
        a2 = f16(np.full(200, 683.0))
        b2 = f16(np.full(200, 3.0))                          # a b = 2049 (between 2048 and 2050)
        c2 = np.tile(f16([2.0 ** -24, -2.0 ** -24, 0.0, -0.0]), 50)
        a, b, c = (np.concatenate(v) for v in ((a, a2), (b, b2), (c, c2)))
        got = N.fma16(a, b, c)
        for i in range(a.size):
            self.assertEqual(bits(got[i]), bits(fma_exact(float(a[i]), float(b[i]), float(c[i]))),
                             (a[i], b[i], c[i]))
        self.assertTrue(np.array_equal(bits(got), bits(fma_int(a, b, c))))
        # one rounding, not two: 2049 + 2^-24 rounds up, h(2049) + 2^-24 rounds to 2048
        self.assertEqual(float(N.fma16(683.0, 3.0, 2.0 ** -24)), 2050.0)
        self.assertEqual(float(f16(f16(2049.0) + 2.0 ** -24)), 2048.0)

    def test_signed_zeros(self):
        z = [0.0, -0.0]
        for a in z + [1.0, -1.0]:
            for b in z:
                for c in z:
                    self.assertEqual(bits(N.fma16(a, b, c)), bits(fma_exact(a, b, c)), (a, b, c))
        self.assertEqual(bits(N.fma16(1.0, 1.0, -1.0)), 0)               # exact cancellation: +0
        self.assertEqual(bits(N.mul16(-2.0 ** -24, 2.0 ** -12)), 0x8000)  # underflow keeps the sign
        self.assertEqual(bits(N.add16(-0.0, -0.0)), 0x8000)
        self.assertEqual(bits(N.plus_zero(-0.0)), 0)
        self.assertEqual(bits(N.plus_zero(-1.5)), bits(-1.5))

    def test_acc_matmul16(self):
        rng = np.random.default_rng(9)
        for k in (16, 8, 2, 1):
            a = f16(rng.uniform(-1, 1, (300, k)) * 2.0 ** rng.integers(-24, 16, (300, k)))
            b = f16(rng.uniform(-1, 1, (k, 7)) * 2.0 ** rng.integers(-24, 16, (k, 7)))
            acc = f16(rng.uniform(-1, 1, (300, 7)) * 2.0 ** rng.integers(-24, 16, (300, 7)))
            want = fp16acc.acc_dot16(acc, a[:, None, :], b.T[None, :, :], axis=-1)
            self.assertTrue(np.array_equal(bits(fp16acc.acc_matmul16(acc, a, b)), bits(want)))
        a = f16(np.full((4, 16), 65504.0))
        a[1, 3] = np.inf
        a[2, 0] = np.nan
        b = f16(np.full((16, 3), -65504.0))
        b[5, 2] = 0.0
        acc = f16(np.zeros((4, 3)))
        acc[3, 1] = -np.inf
        got = fp16acc.acc_matmul16(acc, a, b)
        want = fp16acc.acc_dot16(acc, a[:, None, :], b.T[None, :, :], axis=-1)
        self.assertTrue(np.array_equal(np.isnan(got), np.isnan(want)))
        self.assertTrue(np.array_equal(bits(got[~np.isnan(got)]), bits(want[~np.isnan(want)])))
        with self.assertRaises(ValueError):
            fp16acc.acc_matmul16(np.zeros((1, 1)), np.zeros((1, 17)), np.zeros((17, 1)))


# ---- layers -------------------------------------------------------------------------------------------------------

class Conv(unittest.TestCase):
    def test_against_rationals(self):
        rng = np.random.default_rng(11)
        for cin, cw, cout in ((18, 18, 3), (10, 16, 2), (32, 32, 2), (16, 16, 2)):
            x = f16(rng.uniform(-8, 8, (3, 4, cin)))
            w = f16(rng.uniform(-1, 1, (cout, 3, 3, cw)))
            b = f16(rng.uniform(-1, 1, cout))
            got = N.conv3(x, w, b)
            for oy in range(3):
                for ox in range(4):
                    for o in range(cout):
                        self.assertEqual(bits(got[oy, ox, o]), bits(conv_exact(x, w, b, oy, ox, o)), (cin, oy, ox, o))
            self.assertTrue(np.array_equal(bits(got), bits(conv_dot(x, w, b))))

    def test_tap_major_order(self):
        """Taps outermost: (0,0) block 0 = 1, (0,0) block 1 = 1, (0,1) block 0 = 2048 gives 2050; a channel-block-outer
        order would add 2048 before the second 1 and lose both ones (2048)."""
        x = np.ones((3, 3, 32))
        w = np.zeros((1, 3, 3, 32))
        w[0, 0, 0, 0] = 1.0
        w[0, 0, 0, 16] = 1.0
        w[0, 0, 1, 0] = 2048.0
        self.assertEqual(float(N.conv3(x, w, np.zeros(1))[1, 1, 0]), 2050.0)

    def test_l14_chunks_16_then_2(self):
        """18 channels: one chunk of 16, then one of the last 2 (exact sums): 2048 + 1 from block 0 (rounds to 2048),
        then 1 + 1 from channels 16, 17 gives 2050; one 18-term chunk would give 2051 -> 2052."""
        x = np.zeros((1, 1, 18))
        x[0, 0, :] = 1.0
        w = np.zeros((1, 3, 3, 18))
        w[0, 1, 1, 0], w[0, 1, 1, 1], w[0, 1, 1, 16], w[0, 1, 1, 17] = 2048.0, 1.0, 1.0, 1.0
        self.assertEqual(float(N.conv3(x, w, np.zeros(1))[0, 0, 0]), 2050.0)

    def test_padded_input_channels_read_zero(self):
        rng = np.random.default_rng(2)
        x = f16(rng.uniform(-2, 2, (4, 5, 10)))
        w = f16(rng.uniform(-1, 1, (3, 3, 3, 16)))
        w0 = w.copy()
        w0[:, :, :, 10:] = 0
        b = f16(rng.uniform(-1, 1, 3))
        self.assertTrue(np.array_equal(bits(N.conv3(x, w, b)), bits(N.conv3(x, w0, b))))

    def test_zero_border(self):
        x = np.ones((2, 2, 16))
        w = np.ones((1, 3, 3, 16))
        self.assertTrue(np.array_equal(N.conv3(x, w, np.zeros(1))[:, :, 0], np.full((2, 2), 64.0)))


class Pool(unittest.TestCase):
    def test_order_and_quarter(self):
        a = np.zeros((2, 2, 1))
        a[0, 0, 0], a[0, 1, 0], a[1, 0, 0] = 2048.0, 1.0, 1.0      # the exact mean 512.5; the order gives 512
        self.assertEqual(float(N.pool2(a)[0, 0, 0]), 512.0)
        a = np.full((2, 2, 1), 2.0 ** -24)
        self.assertEqual(float(N.pool2(a)[0, 0, 0]), 2.0 ** -24)    # 4 * 2^-24 * 0.25
        a = np.full((2, 2, 1), 3 * 2.0 ** -24)                      # h(12 * 2^-24 / 4) = 3 * 2^-24
        self.assertEqual(float(N.pool2(a)[0, 0, 0]), 3 * 2.0 ** -24)
        self.assertEqual(bits(N.pool2(np.full((2, 2, 1), -0.0))[0, 0, 0]), 0x8000)

    def test_random_against_scalar(self):
        rng = np.random.default_rng(4)
        a = f16(rng.uniform(-60000, 60000, (6, 8, 3)) * 2.0 ** rng.integers(-20, 1, (6, 8, 3)))
        got = N.pool2(a)
        for y in range(3):
            for x in range(4):
                for c in range(3):
                    p = f16(f16(f16(a[2 * y, 2 * x, c] + a[2 * y, 2 * x + 1, c]) + a[2 * y + 1, 2 * x, c])
                            + a[2 * y + 1, 2 * x + 1, c])
                    self.assertEqual(bits(got[y, x, c]), bits(f16(p * 0.25)))


class Upsample(unittest.TestCase):
    def test_axis_edges(self):
        for n in (1, 2, 5, 34):
            i0, i1, w0, w1 = N.up_axis(n)
            self.assertEqual((i0[0], i1[0], w0[0], w1[0]), (0, 0, 1.25, -0.25))
            self.assertEqual((i0[-1], i1[-1], w0[-1], w1[-1]), (n - 1, n - 1, 0.75, 0.25))
            for o in range(1, 2 * n - 1):
                self.assertEqual((w0[o], w1[o]), (0.75, 0.25) if o % 2 else (0.25, 0.75))
                self.assertEqual((i0[o], i1[o]), ((o - 1) // 2, (o - 1) // 2 + 1))
            self.assertEqual(list(zip(i0, i1, w0, w1)), axis_scalar(n))

    def test_against_scalar(self):
        rng = np.random.default_rng(6)
        for hi, wi, C in ((1, 1, 2), (3, 5, 2), (2, 7, 1)):
            p = f16(rng.uniform(-40, 40, (hi, wi, C)) * 2.0 ** rng.integers(-14, 10, (hi, wi, C)))
            self.assertTrue(np.array_equal(bits(N.up2(p)), bits(up2_scalar(p))), (hi, wi))

    def test_fused_pairs_matter(self):
        """Rounding every product gives other bits than the fused pairs on ordinary data (the rationals above decide
        which form this is)."""
        rng = np.random.default_rng(8)
        p = f16(rng.uniform(-30, 30, (6, 9, 3)))
        got = N.up2(p)
        x0, x1, wx0, wx1 = N.up_axis(9)
        y0, y1, wy0, wy1 = N.up_axis(6)
        rr = lambda wa, a, wb, b: f16(f16(wa * a) + f16(wb * b))
        top = rr(wx0[None, :, None], p[y0][:, x0], wx1[None, :, None], p[y0][:, x1])
        bot = rr(wx0[None, :, None], p[y1][:, x0], wx1[None, :, None], p[y1][:, x1])
        every = rr(wy0[:, None, None], top, wy1[:, None, None], bot)
        self.assertGreater(int(np.sum(bits(got) != bits(every))), 0)


class Merge(unittest.TestCase):
    def test_input_merge(self):
        rng = np.random.default_rng(12)
        ib, ic, ia = (f16(rng.uniform(-9, 9, (16, 16, c))) for c in (4, 4, 1))
        for f, hf in ((0.37, float(np.float16(np.float32(0.37)))), (1 + 2.0 ** -11, 1.0), (1 + 3 * 2.0 ** -11, 1.001953125),
                      (65520.0, np.inf), (1e-9, 0.0), (-2.5, -2.5)):
            x = N.merge_in(ib, ic, ia, np.float32(f))
            self.assertEqual(x.shape, (16, 16, 10))
            self.assertTrue(np.array_equal(x[:, :, 0:4], ib) and np.array_equal(x[:, :, 4:8], ic)
                            and np.array_equal(x[:, :, 8:9], ia))
            self.assertTrue(np.all(x[:, :, 9] == hf), f)

    def test_warp_against_rationals(self):
        rng = np.random.default_rng(13)
        H, W = 5, 6
        img = f16(rng.uniform(-9, 9, (H, W, 2)))
        ux = f16(rng.uniform(-4, 4, (H, W)))
        uy = f16(rng.uniform(-4, 4, (H, W)))
        ux[0, 0], uy[0, 0] = -0.2998046875, -0.1          # small negative: fx rounds
        got = N.warp_sample(img, ux, uy)
        for y in range(H):
            for x in range(W):
                px = fma_exact(2.0, float(ux[y, x]), float(x))
                py = fma_exact(2.0, float(uy[y, x]), float(y))
                X0, Y0 = int(np.floor(px)), int(np.floor(py))
                fx, fy = rn16_exact(Fraction(px) - X0), rn16_exact(Fraction(py) - Y0)
                x0, x1 = min(max(X0, 0), W - 1), min(max(X0 + 1, 0), W - 1)
                y0, y1 = min(max(Y0, 0), H - 1), min(max(Y0 + 1, 0), H - 1)
                gx, gy = rn16_exact(1 - Fraction(fx)), rn16_exact(1 - Fraction(fy))
                w00, w10 = rn16_exact(Fraction(gx) * Fraction(gy)), rn16_exact(Fraction(fx) * Fraction(gy))
                w01, w11 = rn16_exact(Fraction(gx) * Fraction(fy)), rn16_exact(Fraction(fx) * Fraction(fy))
                for c in range(2):
                    acc = fma_exact(w10, img[y0, x1, c], rn16_exact(Fraction(w00) * Fraction(float(img[y0, x0, c]))))
                    acc = fma_exact(w01, img[y1, x0, c], acc)
                    want = fma_exact(w11, img[y1, x1, c], acc)
                    self.assertEqual(bits(got[y, x, c]), bits(want), (y, x, c))

    def test_warp_whole_pixels_and_edges(self):
        rng = np.random.default_rng(14)
        img = f16(rng.uniform(-9, 9, (4, 5, 3)))
        u = np.full((4, 5), 0.5)                                  # 2 u = one pixel right / down, clamped at the edge
        got = N.warp_sample(img, u, -u)
        for y in range(4):
            for x in range(5):
                self.assertTrue(np.array_equal(got[y, x], img[max(y - 1, 0), min(x + 1, 4)]))
        got = N.warp_sample(img, np.full((4, 5), 100.0), np.full((4, 5), -100.0))
        self.assertTrue(np.array_equal(got, np.broadcast_to(img[0, 4], got.shape)))

    def test_warp_merge_layout(self):
        rng = np.random.default_rng(15)
        ib, ic, ia = (f16(rng.uniform(-9, 9, (16, 16, c))) for c in (4, 4, 1))
        u4, u1, u3 = (f16(rng.uniform(-3, 3, (16, 16, c))) for c in (4, 1, 3))
        y = N.warp_merge(ib, ic, ia, u4, u1, u3, np.float32(0.7))
        self.assertEqual(y.shape, (16, 16, 18))
        self.assertTrue(np.array_equal(y[:, :, 0:4], N.warp_sample(ib, u4[:, :, 0], u4[:, :, 1])))
        self.assertTrue(np.array_equal(y[:, :, 4:8], N.warp_sample(ic, u4[:, :, 2], u4[:, :, 3])))
        self.assertTrue(np.array_equal(y[:, :, 8:12], u4))         # not doubled
        self.assertTrue(np.array_equal(y[:, :, 12:13], u1) and np.array_equal(y[:, :, 13:14], ia))
        self.assertTrue(np.array_equal(y[:, :, 14:17], u3))
        self.assertTrue(np.all(y[:, :, 17] == float(np.float16(np.float32(0.7)))))


# ---- launches: activations, clamps, residuals ---------------------------------------------------------------------

def bias_params(layer, bias, cin=None):
    l = {x.name: x for x in L.NET2}[layer]
    return {layer: (np.zeros((l.out, 3, 3, cin or l.cin)), f16(bias))}


class Launches(unittest.TestCase):
    """Zero weights make a convolution's value its bias, so each post-op is seen on values above 6 and below -6."""

    def step(self, n):
        return N.NET2_LAUNCHES[n - 1]

    def test_list(self):
        self.assertEqual(len(N.NET2_LAUNCHES), 28)
        self.assertTrue(N.check_launch_layers())
        ops = [s["op"] for s in N.NET2_LAUNCHES]
        self.assertEqual(ops[0], "merge")
        self.assertEqual(ops[14:16], ["up", "warp"])
        res = [s.get("res") for s in N.NET2_LAUNCHES[4:12]]
        self.assertEqual(res, ["zero", "r0", "zero", "r1", "zero", "r2", "zero", "r3"])

    def test_l12_split_clamps(self):
        b = np.concatenate([np.full(32, 10.0), np.full(32, 10.0), np.full(32, -10.0)])
        buf = {"r4": np.zeros((2, 2, 64))}
        out = N.run_launch(self.step(13), buf, bias_params("net2_12", b), {})
        self.assertEqual(out["g0"].shape, (4, 4, 32))
        self.assertTrue(np.all(out["g0"] == 10.0))                  # leaky only
        self.assertTrue(np.all(out["g1"] == 6.0))                   # leaky + clamp
        self.assertTrue(np.all(out["g2"] == float(f16(S * -10.0))))
        b[:32] = -10.0
        out = N.run_launch(self.step(13), buf, bias_params("net2_12", b), {})
        self.assertTrue(np.all(out["g0"] == float(f16(S * -10.0))))

    def test_l13_heads(self):
        b = np.array([-10.0, 10.0, 7.0, -7.0, -10.0, 10.0, -10.0, 3.0])
        buf = {"g%d" % j: np.zeros((2, 3, 32)) for j in range(3)}
        out = N.run_launch(self.step(14), buf, bias_params("net2_13", b), {})
        self.assertEqual(out["A4"].shape, (4, 6, 4))
        self.assertTrue(np.all(out["A4"] == b[:4]))                 # raw: no leaky, no clamp
        self.assertTrue(np.all(out["A1"] == -6.0))                  # clamp only (no leaky)
        self.assertTrue(np.all(out["A3"] == np.array([6.0, -6.0, 3.0])))

    def test_l24_no_clamp(self):
        b = np.concatenate([np.full(16, 10.0), np.full(16, -10.0), np.full(16, 300.0)])
        out = N.run_launch(self.step(27), {"s4": np.zeros((2, 2, 32))}, bias_params("net2_24", b), {})
        self.assertTrue(np.all(out["e0"] == 10.0) and np.all(out["e2"] == 300.0))
        self.assertTrue(np.all(out["e1"] == float(f16(S * -10.0))))

    def test_l25_output(self):
        rng = np.random.default_rng(16)
        b = np.array([10.0, -10.0, 0.5, 2.0, -10.0, 10.0, -9.0, 1.0])
        buf = {"e%d" % j: np.zeros((2, 2, 16)) for j in range(3)}
        buf.update({"U4": f16(rng.uniform(-3, 3, (4, 4, 4))), "U1": f16(rng.uniform(-3, 3, (4, 4, 1))),
                    "U3": f16(rng.uniform(-3, 3, (4, 4, 3)))})
        out = N.run_launch(self.step(28), buf, bias_params("net2_25", b), {})["out"]
        self.assertEqual(out.shape, (4, 4, 8))
        self.assertTrue(np.array_equal(out[:, :, 0:4], f16(b[:4] + 2.0 * buf["U4"])))
        self.assertTrue(np.array_equal(out[:, :, 4:5], f16(-6.0 + buf["U1"])))
        self.assertTrue(np.array_equal(out[:, :, 5:8], f16(np.array([6.0, -6.0, 1.0]) + buf["U3"])))
        # the doubled vector and the add are one rounding: 2 * 40000 alone would overflow, -29984 + 80000 does not
        b2 = b.copy()
        b2[0] = -29984.0
        buf["U4"][:] = 40000.0
        out = N.run_launch(self.step(28), buf, bias_params("net2_25", b2), {})["out"]
        self.assertEqual(float(out[0, 0, 0]), 50016.0)

    def test_residual_unit(self):
        """Flag 0: h(act + 0) (-0 -> +0); flag 1: h(act + skip), clamp before the add, nothing after."""
        r0 = np.full((2, 2, 64), 5.0)
        r0[0, 0, 0] = -0.0
        p = bias_params("net2_04", np.full(64, -0.0))
        t1 = N.run_launch(self.step(5), {"r0": r0}, p, {})["t1"]
        self.assertTrue(np.all(bits(t1) == 0))                      # conv = bias -0 is +0 (exact chunks), stays +0
        p = bias_params("net2_05", np.full(64, 10.0))
        r1 = N.run_launch(self.step(6), {"t1": t1, "r0": r0}, p, {})["r1"]
        self.assertTrue(np.all(r1[r0 == 5.0] == 11.0))              # 6 + 5: above 6, not clamped after the add
        self.assertEqual(float(r1[0, 0, 0]), 6.0)
        # flag 0 on a value that is -0 after the activation: leaky(-2^-24) = -0, h(-0 + 0) = +0
        p = bias_params("net2_04", np.full(64, -2.0 ** -24))
        t1 = N.run_launch(self.step(5), {"r0": r0}, p, {})["t1"]
        self.assertTrue(np.all(bits(t1) == 0))
        self.assertEqual(bits(N.act(-2.0 ** -24)), 0x8000)

    def test_pool_layer(self):
        b = np.full(32, 8.0)
        b[1] = -8.0
        out = N.run_launch(self.step(2), {"x": np.zeros((4, 4, 10))}, bias_params("net2_01", b), {})["a1"]
        self.assertEqual(out.shape, (2, 2, 32))
        v = f16(S * -8.0)
        pv = f16(f16(f16(f16(v + v) + v) + v) * 0.25)
        self.assertTrue(np.all(out[:, :, 0] == 6.0) and np.all(out[:, :, 1] == pv))


# ---- the whole network --------------------------------------------------------------------------------------------

def up2_int(p):
    """up2 with the test's own axis tables and fma_int."""
    p = f16(p)
    hi, wi, _ = p.shape
    ax, ay = np.array(axis_scalar(wi)), np.array(axis_scalar(hi))
    x0, x1, wx0, wx1 = ax[:, 0].astype(int), ax[:, 1].astype(int), ax[:, 2], ax[:, 3]
    y0, y1, wy0, wy1 = ay[:, 0].astype(int), ay[:, 1].astype(int), ay[:, 2], ay[:, 3]

    def row(r):
        return fma_int(wx1[None, :, None], r[:, x1], f16(wx0[None, :, None] * r[:, x0]))
    return fma_int(wy1[:, None, None], row(p[y1]), f16(wy0[:, None, None] * row(p[y0])))


def act_c(v):
    lk = np.maximum(f16(S * v), v)
    return np.clip(lk, -6.0, 6.0)


def warp_int(img, ux, uy):
    H, W, _ = img.shape
    gy_, gx_ = np.meshgrid(np.arange(H, dtype=np.float64), np.arange(W, dtype=np.float64), indexing="ij")
    px, py = fma_int(2.0, ux, gx_), fma_int(2.0, uy, gy_)
    X0, Y0 = np.floor(px), np.floor(py)
    fx, fy = f16(px - X0), f16(py - Y0)
    X0, Y0 = X0.astype(int), Y0.astype(int)
    x0, x1, y0, y1 = np.clip(X0, 0, W - 1), np.clip(X0 + 1, 0, W - 1), np.clip(Y0, 0, H - 1), np.clip(Y0 + 1, 0, H - 1)
    gx, gy = f16(1 - fx), f16(1 - fy)
    w = [f16(a * c)[:, :, None] for a, c in ((gx, gy), (fx, gy), (gx, fy), (fx, fy))]
    acc = fma_int(w[1], img[y0, x1], f16(w[0] * img[y0, x0]))
    acc = fma_int(w[2], img[y1, x0], acc)
    return fma_int(w[3], img[y1, x1], acc)


def pool_c(a):
    p = f16(f16(f16(a[0::2, 0::2] + a[0::2, 1::2]) + a[1::2, 0::2]) + a[1::2, 1::2])
    return f16(p * 0.25)


def compose(ib, ic, ia, f, P):
    """net2 composed by hand from the test's own operations."""
    hf = float(np.float16(np.float32(f)))
    H, W, _ = ib.shape
    b = {}
    b["x"] = np.concatenate([ib, ic, ia, np.full((H, W, 1), hf)], 2)
    cv = lambda name, v: conv_dot(v, *P[name])
    b["a1"] = pool_c(act_c(cv("net2_01", b["x"])))
    b["a2"] = pool_c(act_c(cv("net2_02", b["a1"])))
    x = b["r0"] = pool_c(act_c(cv("net2_03", b["a2"])))
    for u in range(4):
        t = f16(act_c(cv("net2_%02d" % (4 + 2 * u), x)) + 0.0)
        x = f16(act_c(cv("net2_%02d" % (5 + 2 * u), t)) + x)
        b["t%d" % (u + 1)], b["r%d" % (u + 1)] = t, x
    c = cv("net2_12", up2_int(x))
    b["g0"] = np.maximum(f16(S * c[:, :, :32]), c[:, :, :32])
    b["g1"], b["g2"] = act_c(c[:, :, 32:64]), act_c(c[:, :, 64:])
    b["A4"] = cv("net2_13", up2_int(b["g0"]))[:, :, 0:4]
    b["A1"] = np.clip(cv("net2_13", up2_int(b["g1"]))[:, :, 4:5], -6, 6)
    b["A3"] = np.clip(cv("net2_13", up2_int(b["g2"]))[:, :, 5:8], -6, 6)
    for k in ("4", "1", "3"):
        b["U" + k] = up2_int(b["A" + k])
    u4 = b["U4"]
    b["y"] = np.concatenate([warp_int(ib, u4[:, :, 0], u4[:, :, 1]), warp_int(ic, u4[:, :, 2], u4[:, :, 3]), u4,
                             b["U1"], ia, b["U3"], np.full((H, W, 1), hf)], 2)
    b["b1"] = pool_c(act_c(cv("net2_14", b["y"])))
    x = b["s0"] = pool_c(act_c(cv("net2_15", b["b1"])))
    for u in range(4):
        t = f16(act_c(cv("net2_%02d" % (16 + 2 * u), x)) + 0.0)
        x = f16(act_c(cv("net2_%02d" % (17 + 2 * u), t)) + x)
        b["tb%d" % (u + 1)], b["s%d" % (u + 1)] = t, x
    c = cv("net2_24", up2_int(x))
    c = np.maximum(f16(S * c), c)
    b["e0"], b["e1"], b["e2"] = c[:, :, :16], c[:, :, 16:32], c[:, :, 32:]
    o0 = fma_int(2.0, b["U4"], cv("net2_25", up2_int(b["e0"]))[:, :, 0:4])
    o1 = f16(np.clip(cv("net2_25", up2_int(b["e1"]))[:, :, 4:5], -6, 6) + b["U1"])
    o2 = f16(np.clip(cv("net2_25", up2_int(b["e2"]))[:, :, 5:8], -6, 6) + b["U3"])
    b["out"] = np.concatenate([o0, o1, o2], 2)
    return b


class Network(unittest.TestCase):
    def test_chain_against_composed(self):
        rng = np.random.default_rng(20261121)
        H, W = 32, 48
        P = synthetic_params(rng, 2.5, 1.0)
        ib, ic, ia = (f16(rng.uniform(-8, 8, (H, W, c))) for c in (4, 4, 1))
        got = N.run(ib, ic, ia, np.float32(0.4375), P)
        want = compose(ib, ic, ia, np.float32(0.4375), P)
        ext = N.extents(W, H)
        for name, v in want.items():
            self.assertEqual(got[name].shape[:2], ext[name], name)
            self.assertTrue(np.array_equal(bits(got[name]), bits(v)), name)
        # the clamp exceptions are exercised: values above 6 where nothing clamps, none where something does
        for name in ("r4", "g0", "A4", "s4", "e0", "out"):
            self.assertGreater(float(np.max(np.abs(got[name]))), 6.0, name)
        for name in ("a1", "t1", "g1", "g2", "A1", "A3", "b1", "tb4"):
            self.assertLessEqual(float(np.max(np.abs(got[name]))), 6.0, name)
        self.assertTrue(np.all(np.isfinite(got["out"])))

    def test_input_checks(self):
        P = synthetic_params(np.random.default_rng(1))
        with self.assertRaises(ValueError):
            N.run(np.zeros((16, 24, 4)), np.zeros((16, 24, 4)), np.zeros((16, 24, 1)), 0.0, P)
        with self.assertRaises(ValueError):
            N.run(np.zeros((16, 16, 3)), np.zeros((16, 16, 4)), np.zeros((16, 16, 1)), 0.0, P)


if __name__ == "__main__":
    unittest.main()
