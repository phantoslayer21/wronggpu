# SPDX-License-Identifier: GPL-3.0-only
"""Frame generation (issue #96): the Full path's per-call stages (clean/fg/reference/stages_full.py) on synthetic data:
the Full blend with presets A and B and net2's input tensors, the output pull / pull-middle / push / push-fine pyramid,
the hole rule, NaN and special values, and the distortion stages. Scalar re-computations (plain loops) check the
vectorised indexing; properties check the behaviour."""
import itertools
import unittest

import numpy as np

from clean.fg.reference import stages_full as S
from clean.rr_f.reference import f32 as F


def f16(a):
    return np.asarray(a, np.float32).astype(np.float16).astype(np.float32)


def rand(rng, shape, lo, hi):
    return f16(rng.uniform(lo, hi, shape))


def landed(rng, shape, amp, empty_frac=0.0):
    words = S.pack_mv(rng.uniform(1, 500, shape).astype(np.float32), rng.uniform(-amp, amp, shape),
                      rng.uniform(-amp, amp, shape))
    if empty_frac:
        words = np.where(rng.random(shape) < empty_frac, np.uint64(S.EMPTY), words)
    return words.astype(np.uint64)


def blend_inputs(rng, Wo=21, Ho=13, W1=None, H1=None, Wd=None, Hd=None, amp=0.03, empty=0.3):
    W1, H1 = W1 or Wo, H1 or Ho
    Wd, Hd = Wd or Wo // 2, Hd or Ho // 2
    return dict(Cp=rand(rng, (Ho, Wo, 4), 0, 1), Cc=rand(rng, (Ho, Wo, 4), 0, 1), Wimg=rand(rng, (H1, W1, 2), 0, 1),
                Qc=rand(rng, (Hd, Wd, 2), 0, 1), MA=landed(rng, (H1, W1), amp, empty), MB=landed(rng, (H1, W1), amp, empty))


def ui_inputs(rng, Wo, Ho, frac=0.4):
    out = []
    for _ in range(2):
        a = np.where(rng.random((Ho, Wo)) < frac, rng.uniform(0.05, 1, (Ho, Wo)), 0.0)
        rgb = rng.uniform(0, 1, (Ho, Wo, 3)) * a[..., None]
        out.append(f16(np.concatenate([rgb, a[..., None]], -1)))
    return tuple(out)


# ---- scalar re-computations ----------------------------------------------------------------------------------------

def s_bil(img, tx, ty):
    H, W = img.shape[:2]
    tx, ty = np.float32(tx), np.float32(ty)
    x0f, y0f = np.floor(tx), np.floor(ty)
    fx, fy = F.sub(tx, x0f), F.sub(ty, y0f)
    xi = 0 if np.isnan(x0f) else int(max(min(x0f, 2 ** 31 - 1), -2 ** 31))
    yi = 0 if np.isnan(y0f) else int(max(min(y0f, 2 ** 31 - 1), -2 ** 31))
    x0, x1 = min(max(xi, 0), W - 1), min(max(xi, -1), W - 2) + 1
    y0, y1 = min(max(yi, 0), H - 1), min(max(yi, -1), H - 2) + 1
    gx, gy = F.sub(1, fx), F.sub(1, fy)
    out = []
    for c in range(img.shape[2]):
        v = F.mul(F.mul(fx, gy), img[y0, x1, c])
        v = F.fma(F.mul(gx, gy), img[y0, x0, c], v)
        v = F.fma(F.mul(fy, gx), img[y1, x0, c], v)
        v = F.fma(F.mul(fx, fy), img[y1, x1, c], v)
        out.append(np.float32(v))
    return np.array(out, np.float32)


def s_texl(img, u, v):
    H, W = img.shape[:2]
    return s_bil(img, F.fma(u, np.float32(W), np.float32(-0.5)), F.fma(v, np.float32(H), np.float32(-0.5)))


def s_blend_pixel(d, x, y, ui=None):
    Cp, Cc, Wimg, Qc, MA, MB = (d[k] for k in ("Cp", "Cc", "Wimg", "Qc", "MA", "MB"))
    Ho, Wo = Cp.shape[:2]
    H1, W1 = MA.shape
    u = S.sat(F.div(np.float32(x + 0.5), np.float32(Wo)))
    v = S.sat(F.div(np.float32(y + 0.5), np.float32(Ho)))
    ix = int(np.rint(F.add(F.fma(u, np.float32(W1), np.float32(-0.5)), S.NEG_TIE)))
    iy = int(np.rint(F.add(F.fma(v, np.float32(H1), np.float32(-0.5)), S.NEG_TIE)))
    ix, iy = min(max(ix, 0), W1 - 1), min(max(iy, 0), H1 - 1)

    def vec(word):
        word = int(word)
        if word == S.EMPTY:
            return 0.0, 0.0, 1.0
        bx = np.uint16((word >> 16) & 0xFFFF).view(np.float16)
        by = np.uint16(word & 0xFFFF).view(np.float16)
        return np.float32(bx), np.float32(by), 0.0
    ax, ay, eA = vec(MA[iy, ix])
    bx, by, eB = vec(MB[iy, ix])
    cP = s_texl(Cp, F.add(u, bx), F.add(v, by))[:3]
    cQ = s_texl(Cc, F.add(u, ax), F.add(v, ay))[:3]
    w = s_texl(Wimg, u, v)
    qu, qb, qa = s_texl(Qc, u, v), s_texl(Qc, F.add(u, bx), F.add(v, by)), s_texl(Qc, F.add(u, ax), F.add(v, ay))
    x0 = np.zeros(4, np.float32)
    x1 = np.zeros(4, np.float32)
    for c in range(3):
        x0[c] = F.fma(w[0], F.sub(F.fma(np.float32(eB), F.sub(cQ[c], cP[c]), cP[c]), Cp[y, x, c]), Cp[y, x, c])
        x1[c] = F.fma(w[1], F.sub(F.fma(np.float32(eA), F.sub(cP[c], cQ[c]), cQ[c]), Cc[y, x, c]), Cc[y, x, c])
    x0[3] = F.fma(w[0], F.sub(qb[0], qu[0]), qu[0])
    x1[3] = F.fma(w[1], F.sub(qa[1], qu[1]), qu[1])
    hole = bool(eA and eB and (not (w[0] < 0.5) or not (w[1] < 0.5)))
    if ui is not None:
        Up, Uc = ui[0][y, x], ui[1][y, x]
        for c in range(3):
            x0[c] = F.fma(x0[c], F.sub(1, Up[3]), Up[c])
            x1[c] = F.fma(x1[c], F.sub(1, Uc[3]), Uc[c])
        ad = [np.float32(abs(F.sub(Up[c], Uc[c]))) for c in range(3)]
        e = S.sat(F.div(F.add(ad[2], F.add(ad[0], ad[1])), np.float32(3)))
        if Up[3] > S.UI_EPS:
            x0[3] = e
        if Uc[3] > S.UI_EPS:
            x1[3] = e
        hole = hole and Up[3] < S.UI_EPS and Uc[3] < S.UI_EPS
    return x0, x1, hole


def s_tree(block):
    """Recursive pairwise tree sum of a 2^n x 2^n block (x before y)."""
    n = block.shape[0]
    if n == 1:
        return np.float32(block[0, 0])
    m = n // 2
    a, b = s_tree(block[:m, :m]), s_tree(block[:m, m:])
    c, d = s_tree(block[m:, :m]), s_tree(block[m:, m:])
    return np.float32(F.add(F.add(a, b), F.add(c, d)))


def bits_equal(a, b):
    """float16 bit equality with NaN equal to NaN."""
    ab, bb = S.h_bits(a), S.h_bits(b)
    an = np.isnan(np.asarray(a, np.float32))
    bn = np.isnan(np.asarray(b, np.float32))
    return bool(np.all((ab == bb) | (an & bn)))


# ---- tests -----------------------------------------------------------------------------------------------------------

class TestHelpers(unittest.TestCase):
    def test_constants(self):
        self.assertEqual(int(S.NEG_LOG2E.view(np.uint32)), 0xBFB8AA3B)
        self.assertEqual(int(S.UI_EPS.view(np.uint32)), 0x34000000)
        self.assertEqual(int(S.NEG_TIE.view(np.uint32)), 0xB4000000)

    def test_levels_and_grids(self):
        self.assertEqual(S.n_levels(1920, 1080), 12)
        self.assertEqual(S.n_levels(2560, 1440), 12)
        self.assertEqual(S.n_levels(256, 256), 9)
        self.assertEqual(S.n_levels(300, 200), 10)
        self.assertEqual(S.n_levels(16, 16), 9)
        self.assertEqual(S.coarsest_level(1920, 1080), 8)
        self.assertEqual(S.coarsest_level(48, 40), 4)
        self.assertEqual(S.padded(1920, 1080, 4), (1920, 1088))
        self.assertEqual(S.padded(2560, 1440, 4), (2560, 1472))
        self.assertEqual(S.padded(21, 13, 8), (128, 128))
        self.assertEqual((S.net2_k(1080), S.net2_k(1440), S.net2_k(2160), S.net2_k(480)), (4, 4, 8, 1))
        with self.assertRaises(ValueError):
            S.padded(64, 64, 3)

    def test_block_sums_tree(self):
        rng = np.random.default_rng(1)
        for k in (2, 4, 8, 16):
            a = rng.uniform(-1, 1, (2 * k, 3 * k)).astype(np.float32) * 2.0 ** rng.integers(-20, 20, (2 * k, 3 * k))
            a = a.astype(np.float32)
            got = S.block_sums(a, k)
            for Y in range(2):
                for X in range(3):
                    self.assertEqual(got[Y, X], s_tree(a[Y * k:(Y + 1) * k, X * k:(X + 1) * k]))

    def test_block_sums_order_matters(self):
        # x pairs first: (2^24 + 1 -> 2^24) + (-2^24 + 1) = 1; a column-first running sum gives 2
        a = np.array([[2.0 ** 24, 1.0], [-2.0 ** 24, 1.0]], np.float32)
        tree = S.block_sums(a, 2)[0, 0]
        run = np.float32(np.float32(np.float32(a[0, 0] + a[1, 0]) + a[0, 1]) + a[1, 1])
        self.assertEqual((float(tree), float(run)), (1.0, 2.0))

    def test_bilinear_against_scalar(self):
        rng = np.random.default_rng(2)
        img = rand(rng, (7, 9, 3), -4, 4)
        tx = rng.uniform(-3, 12, 200).astype(np.float32)
        ty = rng.uniform(-3, 10, 200).astype(np.float32)
        tx[:4], ty[:4] = [np.nan, np.inf, -np.inf, 3.0], [1.0, 2.0, 2.5, np.nan]
        got = S.bilinear_px(img, tx, ty)
        for i in range(200):
            want = s_bil(img, tx[i], ty[i])
            self.assertTrue(np.array_equal(got[i], want, equal_nan=True), (i, got[i], want))
        self.assertTrue(np.all(np.isnan(got[:4])))

    def test_bilinear_integer_positions_exact(self):
        rng = np.random.default_rng(3)
        img = rand(rng, (5, 6, 2), -100, 100)
        ys, xs = np.meshgrid(np.arange(5), np.arange(6), indexing="ij")
        got = S.bilinear_px(img, xs.astype(np.float32), ys.astype(np.float32))
        self.assertTrue(np.array_equal(got, img))

    def test_nearest(self):
        u = S.centres(10)
        self.assertTrue(np.array_equal(S.nearest_index(u, 10), np.arange(10)))
        self.assertEqual(int(S.nearest_index(np.float32(np.nan), 10)), 0)
        self.assertEqual(int(S.nearest_index(np.float32(np.inf), 10)), 9)
        self.assertEqual(int(S.nearest_index(np.float32(-5), 10)), 0)

    def test_mv_words(self):
        w = S.pack_mv(np.float32(3.5), np.float32(0.25), np.float32(-0.125))
        vec, empty = S.unpack_mv(np.array([w, np.uint64(S.EMPTY)], np.uint64))
        self.assertEqual(vec.tolist(), [[0.25, -0.125], [0.0, 0.0]])
        self.assertEqual(empty.tolist(), [False, True])
        self.assertEqual(int(w) >> 32, int(np.float32(3.5).view(np.uint32)))

    def test_directed_products(self):
        r = S.Region(7, 5, 1, 2, 4, 3)
        lo, hi = 1.5 * np.float64(r.rx), 4.5 * np.float64(r.rx)       # exact products
        self.assertTrue(np.float64(r.lox) >= lo > np.float64(np.nextafter(r.lox, np.float32(-1))))
        self.assertTrue(np.float64(r.hix) <= hi < np.float64(np.nextafter(r.hix, np.float32(1))))
        self.assertEqual(S.mul_up(np.float32(0.5), np.float32(2.0)), 1.0)


class TestFullBlend(unittest.TestCase):
    def test_against_scalar_preset_a_and_b(self):
        rng = np.random.default_rng(10)
        for (Wo, Ho, W1, H1, Wd, Hd) in ((21, 13, 21, 13, 10, 6), (24, 16, 12, 8, 12, 8), (17, 9, 9, 5, 8, 4)):
            d = blend_inputs(rng, Wo, Ho, W1, H1, Wd, Hd, amp=0.2, empty=0.4)
            d["Wimg"][0, 0] = [np.nan, 0.75]
            ui = ui_inputs(rng, Wo, Ho)
            for preset_ui in (None, ui):
                r = S.g4_full_blend(**d, k=2, ui=preset_ui)
                for y in range(Ho):
                    for x in range(Wo):
                        x0, x1, hole = s_blend_pixel(d, x, y, preset_ui)
                        self.assertTrue(np.array_equal(r["x0"][y, x], x0, equal_nan=True), (x, y, r["x0"][y, x], x0))
                        self.assertTrue(np.array_equal(r["x1"][y, x], x1, equal_nan=True))
                        self.assertEqual(bool(r["hole"][y, x]), hole)
                self.assertTrue(bits_equal(r["X0"], f16(r["x0"])))
                self.assertTrue(np.array_equal(r["Xw"], r["hole"].astype(np.float32)))

    def test_zero_weights_keep_colocated(self):
        rng = np.random.default_rng(11)
        d = blend_inputs(rng)
        d["Wimg"][:] = 0.0
        r = S.g4_full_blend(**d, k=1)
        self.assertTrue(np.array_equal(r["X0"][..., :3], d["Cp"][..., :3]))
        self.assertTrue(np.array_equal(r["X1"][..., :3], d["Cc"][..., :3]))
        self.assertFalse(r["Xw"].any())

    def test_empty_scatter_swaps_source(self):
        # MB empty, MA landed: X0 takes the current frame's sample cQ (weight 1), X1 its own sample
        rng = np.random.default_rng(12)
        Wo, Ho = 16, 8
        d = blend_inputs(rng, Wo, Ho, amp=0.0, empty=0.0)
        d["MB"][:] = S.EMPTY
        d["Wimg"][:] = 1.0
        d["MA"] = S.pack_mv(np.ones((Ho, Wo), np.float32), np.zeros((Ho, Wo)), np.zeros((Ho, Wo)))
        r = S.g4_full_blend(**d, k=1)
        u, v = S.uv_grid(Wo, Ho)
        cQ = S.tex_l(d["Cc"], u, v)[..., :3]
        self.assertTrue(np.array_equal(r["x0"][..., :3], cQ))
        self.assertTrue(np.array_equal(r["x1"][..., :3], cQ))
        self.assertFalse(r["Xw"].any())                  # MA landed: never a hole
        np.testing.assert_allclose(cQ, d["Cc"][..., :3], atol=2e-6)

    def test_hole_rule_and_unordered_compare(self):
        rng = np.random.default_rng(13)
        Wo, Ho = 8, 4
        d = blend_inputs(rng, Wo, Ho, empty=0.0)
        d["MA"][:] = S.EMPTY
        d["MB"][:] = S.EMPTY
        d["Wimg"][:] = [0.25, 0.25]
        self.assertFalse(S.g4_full_blend(**d, k=1)["Xw"].any())
        d["Wimg"][:] = [0.25, 0.5]
        self.assertTrue(S.g4_full_blend(**d, k=1)["Xw"].all())
        d["Wimg"][:] = [np.nan, 0.0]                      # NaN counts as >= 0.5
        self.assertTrue(S.g4_full_blend(**d, k=1)["Xw"].all())
        d["MA"][0, 0] = S.pack_mv(np.float32(1), 0.0, 0.0)
        self.assertFalse(S.g4_full_blend(**d, k=1)["Xw"][0, 0])

    def test_preset_b_overlays_ui(self):
        rng = np.random.default_rng(14)
        Wo, Ho = 20, 12
        d = blend_inputs(rng, Wo, Ho)
        d["MA"][:4] = S.EMPTY
        d["MB"][:4] = S.EMPTY
        d["Wimg"][:4] = 0.75
        Up, Uc = ui_inputs(rng, Wo, Ho, 0.5)
        a = S.g4_full_blend(**d, k=1)
        b = S.g4_full_blend(**d, k=1, ui=(Up, Uc))
        want0 = F.fma(a["x0"][..., :3], F.sub(1, Up[..., 3:4]), Up[..., :3])
        want1 = F.fma(a["x1"][..., :3], F.sub(1, Uc[..., 3:4]), Uc[..., :3])
        self.assertTrue(np.array_equal(b["x0"][..., :3], want0))
        self.assertTrue(np.array_equal(b["x1"][..., :3], want1))
        e = S.sat(F.div(np.abs(Up[..., 2] - Uc[..., 2]) + (np.abs(Up[..., 0] - Uc[..., 0]) + np.abs(Up[..., 1] - Uc[..., 1])),
                        np.float32(3)))
        on0, on1 = Up[..., 3] > S.UI_EPS, Uc[..., 3] > S.UI_EPS
        self.assertTrue(np.array_equal(b["x0"][..., 3][on0], e[on0]))
        self.assertTrue(np.array_equal(b["x1"][..., 3][on1], e[on1]))
        self.assertTrue(np.array_equal(b["x0"][..., 3][~on0], a["x0"][..., 3][~on0]))
        # UI pixels are never holes; UI-free pixels keep preset A's decision
        self.assertTrue(a["Xw"][:4].any())
        self.assertFalse(b["Xw"][on0 | on1].any())
        self.assertTrue(np.array_equal(b["Xw"][~(on0 | on1)], a["Xw"][~(on0 | on1)]))
        self.assertFalse(bits_equal(a["X0"], b["X0"]))

    def test_preset_b_without_ui_equals_a(self):
        rng = np.random.default_rng(15)
        d = blend_inputs(rng, 18, 10)
        z = np.zeros((10, 18, 4), np.float32)
        a = S.g4_full_blend(**d, k=2)
        b = S.g4_full_blend(**d, k=2, ui=(z, z))
        for key in ("X0", "X1", "Xw", "Ib", "Ic", "Ia"):
            self.assertTrue(bits_equal(np.abs(a[key]), np.abs(b[key])), key)    # -0 + 0 = +0 only

    def test_net2_inputs_k(self):
        rng = np.random.default_rng(16)
        Wo, Ho = 37, 21
        d = blend_inputs(rng, Wo, Ho, empty=0.6)
        d["Wimg"][:] = rand(rng, d["Wimg"].shape, 0.3, 1.0)
        r1 = S.g4_full_blend(**d, k=1)
        Wp1, Hp1 = S.padded(Wo, Ho, 1)
        self.assertEqual(r1["Ib"].shape, (Hp1, Wp1, 4))
        self.assertTrue(bits_equal(r1["Ib"][:Ho, :Wo], r1["X0"]))
        self.assertTrue(bits_equal(r1["Ic"][:Ho, :Wo], r1["X1"]))
        self.assertTrue(np.array_equal(r1["Ia"][:Ho, :Wo], r1["Xw"]))
        self.assertFalse(r1["Ib"][Ho:].any() or r1["Ib"][:, Wo:].any() or r1["Ia"][Ho:].any())
        for k in (2, 4, 8, 16):
            r = S.g4_full_blend(**d, k=k)
            Wp, Hp = S.padded(Wo, Ho, k)
            self.assertEqual(r["Ib"].shape, (Hp // k, Wp // k, 4))
            self.assertEqual(r["Ia"].shape, (Hp // k, Wp // k))
            vals = np.zeros((Hp, Wp, 9), np.float32)
            vals[:Ho, :Wo, :4], vals[:Ho, :Wo, 4:8], vals[:Ho, :Wo, 8] = r["x0"], r["x1"], r["hole"]
            for Y in range(Hp // k):
                for X in range(Wp // k):
                    blk = vals[Y * k:(Y + 1) * k, X * k:(X + 1) * k]
                    for c in range(4):
                        self.assertEqual(r["Ib"][Y, X, c], f16(F.div(s_tree(blk[..., c]), np.float32(k * k))))
                        self.assertEqual(r["Ic"][Y, X, c], f16(F.div(s_tree(blk[..., 4 + c]), np.float32(k * k))))
                    self.assertEqual(r["Ia"][Y, X], float(blk[..., 8].any()))
            # the mean is close to the float64 mean of the frame's values (zeros outside the frame)
            m = vals[..., :4].reshape(Hp // k, k, Wp // k, k, 4).astype(np.float64).mean((1, 3))
            np.testing.assert_allclose(r["Ib"], m, rtol=2e-3, atol=1e-4)

    def test_special_values(self):
        rng = np.random.default_rng(17)
        d = blend_inputs(rng, 16, 16, amp=0.1)
        sp = np.array([np.nan, np.inf, -np.inf, 65504, -65504, 2.0 ** -24, -0.0], np.float32)
        d["Cp"].reshape(-1)[::17] = np.resize(sp, d["Cp"].reshape(-1)[::17].shape)
        d["Cc"].reshape(-1)[::13] = np.resize(sp, d["Cc"].reshape(-1)[::13].shape)
        d["MA"][3, 3] = S.pack_mv(np.float32(2), np.float32(np.nan), np.float32(0))
        d["MB"][4, 4] = S.pack_mv(np.float32(2), np.float32(np.inf), np.float32(-np.inf))
        with np.errstate(all="ignore"):
            r = S.g4_full_blend(**d, k=4)
            for y, x in ((3, 3), (4, 4), (0, 0), (15, 15)):
                x0, x1, hole = s_blend_pixel(d, x, y)
                self.assertTrue(np.array_equal(r["x0"][y, x], x0, equal_nan=True))
                self.assertTrue(np.array_equal(r["x1"][y, x], x1, equal_nan=True))
        self.assertTrue(np.isnan(r["X1"][3, 3, :3]).all())       # a NaN vector: NaN sample


class TestOutputPull(unittest.TestCase):
    def test_hole_rule_brute_force(self):
        rng = np.random.default_rng(20)
        for _ in range(300):
            H, W = rng.integers(1, 9, 2)
            need = rng.random((H, W)) < rng.uniform(0.1, 0.9)
            got = S.hole_rule(need)
            for y, x in itertools.product(range(H), range(W)):
                want = True
                for cy, cx in itertools.product((y - 1, y, y + 1), (x - 1, x, x + 1)):
                    anyn = False
                    for qy, qx in itertools.product((cy - 1, cy, cy + 1), (cx - 1, cx, cx + 1)):
                        anyn |= not (0 <= qy < H and 0 <= qx < W) or bool(need[qy, qx])
                    want &= anyn
                self.assertEqual(bool(got[y, x]), want)

    def test_isolated_need_is_not_a_hole(self):
        # a need pixel is always a hole (every window containing it holds it); a neighbour is one only when every
        # window around it reaches a need pixel
        need = np.zeros((9, 9), bool)
        need[4, 4] = True
        self.assertTrue(np.array_equal(S.hole_rule(need), need))
        need[3:6, 3:6] = True
        self.assertTrue(np.array_equal(S.hole_rule(need), need))
        cb = (np.add.outer(np.arange(9), np.arange(9)) % 2 == 0)
        self.assertTrue(S.hole_rule(cb).all())
        stripes = np.zeros((9, 9), bool)
        stripes[:, ::3] = True
        self.assertTrue(S.hole_rule(stripes).all())
        stripes[:, ::4] = False
        stripes[:, 0] = False
        self.assertFalse(S.hole_rule(stripes)[:, 1].any())
        self.assertTrue(S.hole_rule(np.zeros((1, 1), bool))[0, 0])   # every window reaches outside

    def test_sig(self):
        t = S.sig_table()
        z = S.from_bits16(np.arange(65536, dtype=np.uint32).astype(np.uint16))
        fin = np.isfinite(z)
        with np.errstate(over="ignore"):
            want = 1.0 / (1.0 + np.exp(-z[fin].astype(np.float64)))
        np.testing.assert_allclose(t[fin], want, rtol=2e-5, atol=1e-37)
        self.assertEqual(t[0x3C00], S.sig(np.float32(1.0)))
        self.assertEqual((t[0x7C00], t[0xFC00]), (1.0, 0.0))
        self.assertTrue(np.isnan(t[0x7E00]))
        self.assertEqual(t[0x0000], 0.5)
        self.assertEqual(F.SLOW_PATH_COUNT["exp2"] >= 0, True)

    def test_n2_sample_grid_points_exact(self):
        rng = np.random.default_rng(21)
        for k in (1, 2, 4, 8):
            Wo, Ho = 9 * k + 3, 5 * k + 1
            Wp, Hp = S.padded(Wo, Ho, k)
            N2 = rand(rng, (Hp // k, Wp // k, 8), -3, 3)
            sigma, vA, vB = S.n2_sample(N2, Wo, Ho, k)
            t = S.sig_table()
            for y in range(0, Ho, k):
                for x in range(0, Wo, k):
                    e = N2[y // k, x // k]
                    self.assertEqual(vA[y, x].tolist(), e[0:2].tolist())
                    self.assertEqual(vB[y, x].tolist(), e[2:4].tolist())
                    self.assertEqual(sigma[y, x], t[S.h_bits(e[4])])
            if k > 1:      # between grid points: a bilinear mix of the logistic values, not the logistic of a mix
                y, x = 0, k // 2
                lo, hi = sorted((t[S.h_bits(N2[0, 0, 4])], t[S.h_bits(N2[0, 1, 4])]))
                self.assertTrue(lo - 1e-7 <= sigma[y, x] <= hi + 1e-7)

    def test_pull_colour_mix_and_holes(self):
        rng = np.random.default_rng(22)
        Wo, Ho, k = 24, 16, 2
        X0 = rand(rng, (Ho, Wo, 4), 0, 1)
        X1 = rand(rng, (Ho, Wo, 4), 0, 1)
        Xw = np.zeros((Ho, Wo), np.float32)
        Xw[5:10, 6:12] = 1.0
        N2 = np.zeros((Ho // k, Wo // k, 8), np.float32)
        N2[..., 4] = rand(rng, (Ho // k, Wo // k), -2, 2)
        p = S.g6_output_pull(X0, X1, Xw, N2, k=k)
        hole = S.hole_rule(Xw > 0)
        self.assertTrue(np.array_equal(hole, Xw > 0))
        sigma, _, _ = S.n2_sample(N2, Wo, Ho, k)
        sg = sigma[..., None]
        want = S.sat(F.fma(sg, X0[..., :3], F.mul(F.sub(1, sg), X1[..., :3])))
        want[hole] = 0
        self.assertTrue(np.array_equal(p["c"], want))
        self.assertTrue(np.array_equal(p["out"][..., 3], np.ones((Ho, Wo), np.float32)))
        self.assertTrue(np.array_equal(p["fine"], hole.astype(np.float32)))
        self.assertTrue(np.array_equal(p["out"][hole][:, :3], np.zeros((int(hole.sum()), 3), np.float32)))

    def test_pull_vectors_shift_sample(self):
        # vA = (1, 0) on the net2 grid moves the X0 sample k pixels to the right
        Wo, Ho, k = 32, 8, 4
        X0 = np.zeros((Ho, Wo, 4), np.float32)
        X0[..., 0] = np.arange(Wo, dtype=np.float32)[None, :] / 64
        X0 = f16(X0)
        N2 = np.zeros((Ho // k, Wo // k, 8), np.float32)
        N2[..., 0] = 1.0
        N2[..., 4] = 65504.0                              # sigma = 1: only X0
        p = S.g6_output_pull(X0, X0, np.zeros((Ho, Wo), np.float32), N2, k=k)
        want = np.minimum(np.arange(Wo) + k, Wo - 1) / 64
        np.testing.assert_array_equal(p["c"][0, :, 0], want.astype(np.float32))

    def test_nan_inputs_never_reach_output(self):
        rng = np.random.default_rng(23)
        Wo, Ho, k = 16, 16, 2
        X0 = rand(rng, (Ho, Wo, 4), 0, 1)
        X1 = rand(rng, (Ho, Wo, 4), 0, 1)
        X0[3, 3, 0] = np.nan
        X1[8, 8, :3] = np.inf
        N2 = rand(rng, (8, 8, 8), -0.5, 0.5)
        N2[2, 2, 4] = np.nan
        N2[5, 5, 0] = np.inf
        with np.errstate(all="ignore"):
            f = S.g6_output(X0, X1, np.zeros((Ho, Wo), np.float32), N2, k=k)["frame"]
        self.assertFalse(np.isnan(f).any())
        self.assertTrue(((f >= 0) & (f <= 1)).all())

    def test_levels(self):
        rng = np.random.default_rng(24)
        Wo, Ho = 37, 23
        X0 = rand(rng, (Ho, Wo, 4), 0, 1)
        Xw = (rng.random((Ho, Wo)) < 0.3).astype(np.float32)
        N2 = rand(rng, (Ho, Wo, 8), -0.2, 0.2)
        p = S.g6_output_pull(X0, X0, Xw, N2, k=1)
        base = np.zeros((48, 48, 4), np.float32)
        base[:Ho, :Wo, :3], base[:Ho, :Wo, 3] = p["c"], p["w"]
        for l in range(1, 5):
            n = 1 << l
            lw, lh = S.level_size(Wo, Ho, l)
            self.assertEqual(p["levels"][l].shape, (lh, lw, 4))
            for Y in range(lh):
                for X in range(lw):
                    blk = base[Y * n:(Y + 1) * n, X * n:(X + 1) * n]
                    s = [s_tree(blk[..., c]) for c in range(4)]
                    den = max(s[3], S.FLOOR)
                    want = [f16(F.div(s[c], den)) for c in range(3)] + [f16(F.div(s[3], np.float32(n * n)))]
                    self.assertEqual(p["levels"][l][Y, X].tolist(), [float(w) for w in want])
        self.assertTrue(np.all(p["levels"][1][..., 3] <= 1.0))

    def test_pull_mid(self):
        rng = np.random.default_rng(25)
        self.assertEqual(S.g6_output_pull_mid(np.zeros((3, 3, 4), np.float32), 48, 40), {})
        Wo, Ho = 600, 70
        self.assertGreater(S.n_levels(Wo, Ho), 9)
        lw, lh = S.level_size(Wo, Ho, 4)
        L4 = rand(rng, (lh, lw, 4), 0, 1)
        L4[..., 3] = np.where(rng.random((lh, lw)) < 0.3, 0.0, L4[..., 3])
        mid = S.g6_output_pull_mid(L4, Wo, Ho)
        self.assertEqual(sorted(mid), [5, 6, 7, 8])
        base = np.zeros((16 * -(-lh // 16), 16 * -(-lw // 16), 4), np.float32)
        base[:lh, :lw, :3], base[:lh, :lw, 3] = F.mul(L4[..., :3], L4[..., 3:4]), L4[..., 3]
        for l in range(5, 9):
            n = 1 << (l - 4)
            sw, sh = S.level_size(Wo, Ho, l)
            self.assertEqual(mid[l].shape, (sh, sw, 4))
            for Y in range(sh):
                for X in range(sw):
                    blk = base[Y * n:(Y + 1) * n, X * n:(X + 1) * n]
                    s = [s_tree(blk[..., c]) for c in range(4)]
                    want = [f16(F.div(s[c], max(s[3], S.FLOOR))) for c in range(3)] + \
                        [f16(F.div(s[3], np.float32(n * n)))]
                    self.assertEqual(mid[l][Y, X].tolist(), [float(w) for w in want])


class TestOutputPush(unittest.TestCase):
    def test_upsample_taps(self):
        coarse = np.zeros((3, 3, 4), np.float32)
        coarse[1, 1] = [0.5, 0.25, 1.0, 1.0]
        coarse[1, 2] = [1.0, 1.0, 1.0, 0.5]
        up = S.upsample(coarse, 6, 6)
        # fine (3, 3): parent (1, 1) (0.5625), x-neighbour (1, 2) (0.1875), y-neighbour (2, 1), diagonal (2, 2)
        ws = 0.5625 * 1.0 + 0.1875 * 0.5
        self.assertAlmostEqual(float(up[3, 3, 3]), ws, places=7)
        self.assertAlmostEqual(float(up[3, 3, 0]), (0.5625 * 0.5 + 0.1875 * 0.5) / ws, places=6)
        # fine (2, 2): neighbours at -1 -> (0, *), all empty but the parent
        self.assertAlmostEqual(float(up[2, 2, 3]), 0.5625, places=7)
        self.assertEqual(float(up[2, 2, 1]), 0.25)
        # an empty neighbourhood: v = 0 / 1e-12 = 0, w = 0
        self.assertEqual(up[0, 5].tolist(), [0.0, 0.0, 0.0, 0.0])
        # weights above 1 are clamped in the normalisation only
        big = np.full((2, 2, 4), 2.0, np.float32)
        u2 = S.upsample(big, 4, 4)
        self.assertEqual(float(u2[1, 1, 3]), 2.0)
        self.assertEqual(float(u2[1, 1, 0]), 4.0)

    def test_push_keeps_own_and_fills_empty(self):
        rng = np.random.default_rng(30)
        Wo, Ho = 48, 40
        levels = {}
        for l in range(1, 5):
            lw, lh = S.level_size(Wo, Ho, l)
            levels[l] = rand(rng, (lh, lw, 4), 0.1, 1)
        levels[1][2:5, 3:7, 3] = 0.0
        levels[1][0, 0, 3] = np.nan
        filled = S.g6_output_push(levels, Wo, Ho)
        self.assertEqual(sorted(filled), [1, 2, 3, 4])
        self.assertTrue(np.array_equal(filled[4], levels[4]))
        keep = ~(levels[1][..., 3] <= 0)
        self.assertTrue(np.array_equal(filled[1][keep], levels[1][keep], equal_nan=True))
        up = f16(S.upsample(filled[2], *S.level_size(Wo, Ho, 1)))
        self.assertTrue(np.array_equal(filled[1][~keep], up[~keep]))
        self.assertTrue(np.isnan(filled[1][0, 0, 3]))     # a NaN weight counts as > 0: kept

    def test_inpaint_small_frame(self):
        # uniform colour around a 12 x 12 hole: after the pyramid every hole pixel has that colour
        Wo, Ho, k = 48, 40, 2
        col = np.float32([0.25, 0.5, 0.75])
        X = np.zeros((Ho, Wo, 4), np.float32)
        X[..., :3] = col
        Xw = np.zeros((Ho, Wo), np.float32)
        Xw[14:26, 18:30] = 1.0
        N2 = np.zeros((Ho // k, Wo // k, 8), np.float32)
        out = S.g6_output(X, X, Xw, N2, k=k)
        hole = out["pull"]["fine"] > 0
        self.assertEqual(int(hole.sum()), 144)
        np.testing.assert_allclose(out["frame"][hole][:, :3], np.broadcast_to(col, (144, 3)), atol=1e-3)
        self.assertTrue(np.array_equal(out["frame"][~hole], out["pull"]["out"][~hole]))

    def test_inpaint_multi_level(self):
        # a frame with more than 9 levels and a hole wider than level 4's reach: levels 5..8 are needed
        Wo, Ho, k = 320, 40, 4
        self.assertEqual(S.n_levels(Wo, Ho), 10)
        X0 = np.zeros((Ho, Wo, 4), np.float32)
        X0[..., 0] = f16(np.linspace(0, 1, Wo))[None, :]
        X0[..., 1] = 0.5
        Xw = np.zeros((Ho, Wo), np.float32)
        Xw[:, 100:260] = 1.0                              # a full-height band: 158 columns of holes
        N2 = np.zeros((Ho // k, Wo // k, 8), np.float32)
        N2[..., 4] = 65504.0
        out = S.g6_output(X0, X0, Xw, N2, k=k)
        self.assertEqual(sorted(out["levels"]), [1, 2, 3, 4, 5, 6, 7, 8])
        self.assertEqual(sorted(out["filled"]), [1, 2, 3, 4, 5, 6, 7, 8])
        self.assertTrue(np.array_equal(out["filled"][8], out["levels"][8]))
        hole = out["pull"]["fine"] > 0
        self.assertTrue(hole[:, 102:258].all())
        f = out["frame"]
        self.assertTrue(np.isfinite(f).all())
        self.assertTrue((f[hole][:, 1] > 0.45).all() and (f[hole][:, 1] < 0.55).all())
        # the middle of the band takes colour from both sides: between the edge colours
        mid = f[20, 180, 0]
        self.assertTrue(X0[0, 99, 0] - 0.05 < mid < X0[0, 260, 0] + 0.05)
        # without levels 5..8 the centre would stay empty: level 4 alone does not reach it
        lv = dict(out["levels"])
        L4w = lv[4][..., 3]
        self.assertEqual(float(L4w[:, 8:15].max()), 0.0)

    def test_push_fine_only_holes(self):
        rng = np.random.default_rng(31)
        Wo, Ho = 20, 12
        out = rand(rng, (Ho, Wo, 4), 0, 1)
        fine = np.zeros((Ho, Wo), np.float32)
        fine[3, 4] = 1.0
        filled1 = rand(rng, (6, 10, 4), 0.2, 1)
        r = S.g6_output_push_fine(out, fine, filled1)
        m = fine > 0
        self.assertTrue(np.array_equal(r[~m], out[~m]))
        up = S.upsample(filled1, Wo, Ho)
        self.assertEqual(r[3, 4].tolist(), [float(f16(up[3, 4, c])) for c in range(3)] + [1.0])


class TestDistortion(unittest.TestCase):
    def test_downsample_region_and_scale(self):
        rng = np.random.default_rng(40)
        tw, th = 30, 20
        field = rng.uniform(-1, 1, (th, tw, 4)).astype(np.float32)
        reg = S.Region(tw, th, 3, 2, 16, 10)
        d = S.r5_downsample(field, reg, 16, 10)
        self.assertTrue(bits_equal(d, field[2:12, 3:19]))
        d2 = S.r5_downsample(field, reg, 16, 10, lowp=(0.5, -2.0))
        self.assertTrue(bits_equal(d2, F.fma(np.float32(-2), field[2:12, 3:19], np.float32(0.5))))
        dh = S.r5_downsample(field, reg, 8, 5)          # half the resolution: the texel right of / below each centre
        self.assertTrue(bits_equal(dh, field[3:13:2, 4:20:2]))
        reg2 = S.Region(tw, th, 0, 0, 4, 4)
        self.assertTrue(bits_equal(S.r5_downsample(field, reg2, 2, 2), field[1:4:2, 1:4:2]))
        # upsampling a 2 x 2 region to 8 x 8: positions outside the texel centres clamp to the edge texels
        up = S.r5_downsample(field, S.Region(tw, th, 5, 5, 2, 2), 8, 8)
        self.assertTrue(bits_equal(up[:4, :4], np.broadcast_to(field[5, 5], (4, 4, 4))))
        self.assertTrue(bits_equal(up[4:, 4:], np.broadcast_to(field[6, 6], (4, 4, 4))))

    def test_downsample_negative_zero(self):
        field = np.full((4, 4, 4), -0.0, np.float32)
        d = S.r5_downsample(field, S.Region(4, 4), 4, 4)
        self.assertTrue((S.h_bits(d) == 0).all())          # fma(1, -0, +0) = +0

    def test_pair_clears_previous(self):
        d = np.ones((2, 2, 4), np.float32)
        p = np.full((2, 2, 4), 3.0, np.float32)
        self.assertEqual(S.r5_pair(d, p, False)[1].tolist(), np.zeros((2, 2, 4)).tolist())
        self.assertTrue(np.array_equal(S.r5_pair(d, p, True)[1], p))
        self.assertEqual(S.r5_pair(d, None, True)[1].sum(), 0)

    def test_identity_fields(self):
        rng = np.random.default_rng(41)
        W1, H1 = 24, 14
        Z = rng.integers(0, 2 ** 32, (H1, W1), dtype=np.uint64).astype(np.uint32)
        zero4 = np.zeros((H1, W1, 4), np.float32)
        zero2 = np.zeros((H1, W1, 2), np.float32)
        r = S.r5_fields(zero4, zero4, Z, zero2, zero2)
        self.assertTrue(np.array_equal(r["Zd"], Z))
        self.assertFalse(r["Mcp"].any() or r["Fpc"].any())

    def test_translation(self):
        rng = np.random.default_rng(42)
        W1, H1 = 32, 16
        Z = rng.integers(0, 2 ** 32, (H1, W1), dtype=np.uint64).astype(np.uint32)
        Dc = np.zeros((H1, W1, 4), np.float32)
        Dc[..., 0] = 2.0 / W1                           # distorted -> undistorted: two pixels right
        Dp = np.zeros((H1, W1, 4), np.float32)
        Mv = np.zeros((H1, W1, 2), np.float32)
        Mv[..., 1] = 1.0 / H1
        Fpc = np.zeros((H1, W1, 2), np.float32)
        Fpc[..., 0] = 3.0
        r = S.r5_fields(f16(Dc), Dp, Z, Mv, Fpc)
        self.assertTrue(np.array_equal(r["Zd"][:, :W1 - 2], Z[:, 2:]))
        self.assertTrue(np.array_equal(r["Zd"][:, W1 - 2:], Z[:, W1 - 1:].repeat(2, 1)))
        np.testing.assert_allclose(r["Mcp"][..., 0], f16(2.0 / W1), atol=1e-3)
        np.testing.assert_allclose(r["Mcp"][..., 1], 1.0 / H1, atol=1e-3)
        np.testing.assert_allclose(r["Fpc"][..., 0], 3.0, atol=2e-3)
        np.testing.assert_allclose(r["Fpc"][..., 1], 0.0, atol=2e-3)

    def test_against_scalar(self):
        rng = np.random.default_rng(43)
        W1, H1 = 13, 9
        Dc = rand(rng, (H1, W1, 4), -0.1, 0.1)
        Dp = rand(rng, (H1, W1, 4), -0.1, 0.1)
        Z = rng.integers(0, 2 ** 32, (H1, W1), dtype=np.uint64).astype(np.uint32)
        Mv = rng.uniform(-0.1, 0.1, (H1, W1, 2)).astype(np.float32)
        Fp = rand(rng, (H1, W1, 2), -3, 3)
        Dc[0, 0, 0] = np.nan
        r = S.r5_fields(Dc, Dp, Z, Mv, Fp)
        for y in range(H1):
            for x in range(W1):
                u = S.sat(F.div(np.float32(x + 0.5), np.float32(W1)))
                v = S.sat(F.div(np.float32(y + 0.5), np.float32(H1)))
                dcu = s_texl(Dc, u, v)
                p1 = (F.add(u, dcu[0]), F.add(v, dcu[1]))
                ix, iy = int(S.nearest_index(p1[0], W1)), int(S.nearest_index(p1[1], H1))
                self.assertEqual(r["Zd"][y, x], Z[iy, ix])
                p2 = (F.add(p1[0], Mv[iy, ix, 0]), F.add(p1[1], Mv[iy, ix, 1]))
                q = s_texl(Dp, *p2)
                want = f16([F.sub(F.add(p2[0], q[2]), u), F.sub(F.add(p2[1], q[3]), v)])
                self.assertTrue(np.array_equal(r["Mcp"][y, x], want, equal_nan=True))
                dpu = s_texl(Dp, u, v)
                rr = (F.add(u, dpu[0]), F.add(v, dpu[1]))
                f = Fp[int(S.nearest_index(rr[1], H1)), int(S.nearest_index(rr[0], W1))]
                s = (F.add(rr[0], F.div(f[0], np.float32(W1))), F.add(rr[1], F.div(f[1], np.float32(H1))))
                dcs = s_texl(Dc, *s)
                want = f16([F.mul(F.sub(F.add(s[0], dcs[2]), u), np.float32(W1)),
                            F.mul(F.sub(F.add(s[1], dcs[3]), v), np.float32(H1))])
                self.assertTrue(np.array_equal(r["Fpc"][y, x], want, equal_nan=True))


if __name__ == "__main__":
    unittest.main()
