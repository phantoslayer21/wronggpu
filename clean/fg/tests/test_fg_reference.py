# SPDX-License-Identifier: MIT
"""Frame generation (#96): self-checks of the CPU reference's building blocks (clean/fg/reference/ops.py) on
synthetic data, against plain loop implementations written separately here.

    python -m unittest clean.fg.tests.test_fg_reference
"""
import unittest

import numpy as np

from clean.fg import layout as L
from clean.fg.reference import ops


def f16(x):
    return np.asarray(x, dtype=np.float64).astype(np.float16).astype(np.float64)


def halves(rng, shape, scale=1.0):
    return f16(rng.uniform(-scale, scale, size=shape))


def conv_loops(x, w, b, pad):
    """A plain loop convolution: float64 sums, one float16 rounding at the end."""
    cout, k, _, cin = w.shape
    _, H, W = x.shape
    p = k // 2
    out = np.zeros((cout, H, W))
    for o in range(cout):
        for yy in range(H):
            for xx in range(W):
                s = 0.0
                for ky in range(k):
                    for kx in range(k):
                        sy, sx = yy + ky - p, xx + kx - p
                        if pad == "zero" and not (0 <= sy < H and 0 <= sx < W):
                            continue
                        if pad == "replicate":
                            sy, sx = min(max(sy, 0), H - 1), min(max(sx, 0), W - 1)
                        if pad == "mirror":
                            sy = -sy if sy < 0 else (2 * (H - 1) - sy if sy >= H else sy)
                            sx = -sx if sx < 0 else (2 * (W - 1) - sx if sx >= W else sx)
                        for c in range(cin):
                            s += w[o, ky, kx, c] * x[c, sy, sx]
                out[o, yy, xx] = s + b[o]
    return f16(out)


class Conv(unittest.TestCase):
    def test_against_loops(self):
        rng = np.random.default_rng(1)
        for k in (1, 3):
            for pad in ops.PAD_MODES:
                x = halves(rng, (5, 6, 7))
                w = halves(rng, (4, k, k, 5), 0.5)
                b = halves(rng, (4,), 0.25)
                want = conv_loops(x, w, b, pad)
                got = ops.conv2d(x, w, b, pad=pad, relu_after=False)
                np.testing.assert_array_equal(got, want, err_msg="k=%d pad=%s" % (k, pad))
                np.testing.assert_array_equal(ops.conv2d(x, w, b, pad=pad, relu_after=True), np.maximum(want, 0.0))

    def test_output_is_float16(self):
        rng = np.random.default_rng(2)
        y = ops.conv2d(halves(rng, (3, 4, 4)), halves(rng, (2, 3, 3, 3)), halves(rng, (2,)), pad="zero", relu_after=False)
        np.testing.assert_array_equal(y, f16(y))

    def test_open_items_have_no_default(self):
        rng = np.random.default_rng(3)
        x, w, b = halves(rng, (2, 4, 4)), halves(rng, (2, 3, 3, 2)), halves(rng, (2,))
        with self.assertRaises(TypeError):
            ops.conv2d(x, w, b, relu_after=False)          # pad is required
        with self.assertRaises(TypeError):
            ops.conv2d(x, w, b, pad="zero")                # relu_after is required
        with self.assertRaises(ValueError):
            ops.conv2d(x, w, b, pad="wrap", relu_after=False)
        with self.assertRaises(ops.OpenItem):
            ops.conv2d(x, w, b, pad="zero", relu_after=False, acc="f16")

    def test_shape_checks(self):
        rng = np.random.default_rng(4)
        with self.assertRaises(ValueError):
            ops.conv2d(halves(rng, (3, 4, 4)), halves(rng, (2, 3, 3, 2)), halves(rng, (2,)), pad="zero", relu_after=False)
        with self.assertRaises(ValueError):
            ops.conv2d(halves(rng, (2, 4, 4)), halves(rng, (2, 2, 2, 2)), halves(rng, (2,)), pad="zero", relu_after=False)


class PoolUpscale(unittest.TestCase):
    def test_pool(self):
        x = f16(np.arange(2 * 4 * 6).reshape(2, 4, 6) * 0.25 - 3.0)
        mx = ops.pool2x(x, kind="max")
        av = ops.pool2x(x, kind="avg")
        for c in range(2):
            for i in range(2):
                for j in range(3):
                    blk = x[c, 2 * i:2 * i + 2, 2 * j:2 * j + 2]
                    self.assertEqual(mx[c, i, j], blk.max())
                    self.assertEqual(av[c, i, j], f16(blk.mean()))
        with self.assertRaises(TypeError):
            ops.pool2x(x)
        with self.assertRaises(ValueError):
            ops.pool2x(x, kind="min")

    def test_upscale_align_corners(self):
        rng = np.random.default_rng(5)
        x = halves(rng, (3, 5, 4))
        y = ops.upscale2x_bilinear(x)
        self.assertEqual(y.shape, (3, 10, 8))
        # corners copied exactly
        for (i, j), (p, q) in (((0, 0), (0, 0)), ((0, -1), (0, -1)), ((-1, 0), (-1, 0)), ((-1, -1), (-1, -1))):
            np.testing.assert_array_equal(y[:, i, j], x[:, p, q])
        # a linear ramp stays linear: value at output i is the ramp at i * (n - 1) / (2n - 1)
        ramp = f16(np.tile(np.arange(4.0)[None, None, :], (1, 3, 1)))
        yr = ops.upscale2x_bilinear(ramp)
        np.testing.assert_array_equal(yr[0, 0], f16(np.arange(8) * 3.0 / 7.0))
        # one sample against a hand formula
        fy, fx = 3 * 4 / 9, 5 * 3 / 7
        y0, x0 = int(fy), int(fx)
        ay, ax = fy - y0, fx - x0
        want = ((1 - ay) * ((1 - ax) * x[1, y0, x0] + ax * x[1, y0, x0 + 1]) +
                ay * ((1 - ax) * x[1, y0 + 1, x0] + ax * x[1, y0 + 1, x0 + 1]))
        self.assertAlmostEqual(y[1, 3, 5], f16(want), delta=abs(want) * 2 ** -10 + 1e-7)

    def test_upscale_single_row(self):
        x = f16(np.array([[[1.5, -2.0]]]))
        y = ops.upscale2x_bilinear(x)
        self.assertEqual(y.shape, (1, 2, 4))
        np.testing.assert_array_equal(y[0, 0], y[0, 1])


class Features(unittest.TestCase):
    def test_luma_and_error(self):
        rng = np.random.default_rng(6)
        a = rng.uniform(0, 1, (3, 4, 4))
        b = rng.uniform(0, 3, (3, 4, 4))
        np.testing.assert_array_equal(ops.luma(a), f16(0.299 * a[0] + 0.587 * a[1] + 0.114 * a[2]))
        e = ops.warp_error(a, b)
        self.assertTrue(np.all((e >= 0) & (e <= 1)))
        np.testing.assert_array_equal(e, f16(np.minimum(np.abs(a - b).mean(axis=0), 1.0)))
        m, mc = ops.mask_pair(np.array([0.0, 0.25, 1.0]))
        np.testing.assert_array_equal(mc, [1.0, 0.75, 0.0])

    def test_net1_features_and_fold(self):
        rng = np.random.default_rng(7)
        names = ["luma0", "luma1", "err0", "err1", "in0", "in1", "in2", "in3", "mask", "mask_c"]
        feats = {n: halves(rng, (6, 8)) for n in names}
        with self.assertRaises(TypeError):
            ops.net1_features(feats)
        with self.assertRaises(ValueError):
            ops.net1_features(feats, order=names[:-1] + ["luma0"])
        x = ops.net1_features(feats, order=names)
        self.assertEqual(x.shape, (ops.NET1_FEATURES, 6, 8))
        for order in ("dy dx c", "c dy dx", "dx c dy"):
            cells = ops.fold2x2(x, order=order)
            self.assertEqual(cells.shape, (ops.NET1_CELL, 3, 4))
            np.testing.assert_array_equal(cells[40:], 0.0)
            np.testing.assert_array_equal(ops.unfold2x2(cells, ops.NET1_FEATURES, order=order), x)
        cells = ops.fold2x2(x, order="dy dx c")
        # value (dy, dx, c) of cell (Y, X) is pixel (2Y + dy, 2X + dx) channel c
        self.assertEqual(cells[(1 * 2 + 0) * 10 + 7, 2, 3], x[7, 5, 6])
        with self.assertRaises(TypeError):
            ops.fold2x2(x)


class LayoutShapes(unittest.TestCase):
    def test_weight_shapes(self):
        self.assertEqual(ops.weight_shape(L.NET1[0]), (32, 3, 3, 48))
        self.assertEqual(ops.weight_shape(L.NET1[5]), (512, 1, 1, 256))
        with self.assertRaises(ops.OpenItem):
            ops.weight_shape(L.NET1[1])
        for l in L.NET2:
            s = ops.weight_shape(l)
            self.assertEqual(s, (l.out, 3, 3, l.cin))
            self.assertEqual(int(np.prod(s)), l.weight_elements)
        for l in L.NET1:
            if l.taps is not None:
                self.assertEqual(int(np.prod(ops.weight_shape(l))), l.weight_elements)

    def test_every_layer_runs(self):
        """Each layer of both tables runs as one conv on a tiny grid with random weights (shapes fit)."""
        rng = np.random.default_rng(8)
        for l in [l for l in L.NET1 if l.taps is not None] + list(L.NET2):
            shape = ops.weight_shape(l)
            y = ops.conv2d(halves(rng, (shape[3], 2, 2)), halves(rng, shape, 0.05), halves(rng, (l.out,)),
                           pad="zero", relu_after=True)
            self.assertEqual(y.shape, (l.out, 2, 2))
            self.assertTrue(np.all(np.isfinite(y)) and np.all(y >= 0))

    def test_networks_open(self):
        with self.assertRaises(ops.OpenItem):
            ops.net1_forward()
        with self.assertRaises(ops.OpenItem):
            ops.net2_forward()


if __name__ == "__main__":
    unittest.main()
