# SPDX-License-Identifier: MIT
"""Frame generation (#96): the forward-pass reference (clean/fg/reference/forward.py) on synthetic weights, against
direct NumPy versions written separately here. The wirings used are stand-ins (the real one is not delivered); they
only show that a wiring runs the layers unchanged.

    python -m unittest clean.fg.tests.test_fg_forward
"""
import os
import shutil
import tempfile
import unittest

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from clean.fg import layout as L
from clean.fg.reference import blobs as B
from clean.fg.reference import forward as F
from clean.fg.reference import ops

ORDER = "out in ky kx"   # the stand-in storage order of the synthetic weights


def f16(x):
    return np.asarray(x, dtype=np.float64).astype(np.float16).astype(np.float64)


def natural_shape(l):
    return (l.out, 1, 1, l.k) if l.name == "net1_02" else ops.weight_shape(l)


def synthetic_params(rng, scale=0.05):
    """{layer: (storage weights, bias)} with natural weights stored in ORDER; also returns the natural weights."""
    params, natural = {}, {}
    for name, l in F.LAYERS.items():
        cout, k, _, cin = natural_shape(l)
        nat = rng.uniform(-scale, scale, (cout, k, k, cin)).astype(np.float16)
        stored = nat.transpose(0, 3, 1, 2).reshape(-1)                      # out, in, ky, kx
        params[name] = (stored, rng.uniform(-0.1, 0.1, cout).astype(np.float16))
        natural[name] = nat.astype(np.float64)
    return params, natural


def direct_conv(x, w, b, pad, relu):
    """Direct NumPy convolution: sliding windows and one tensordot (float64 sums), then one float16 rounding."""
    k = w.shape[1]
    p = k // 2
    mode = {"zero": "constant", "replicate": "edge", "mirror": "reflect"}[pad]
    xp = np.pad(f16(x), ((0, 0), (p, p), (p, p)), mode=mode) if p else f16(x)
    win = sliding_window_view(xp, (k, k), axis=(1, 2))                     # cin, H, W, ky, kx
    y = np.tensordot(f16(w), win, axes=([3, 1, 2], [0, 3, 4])) + f16(b)[:, None, None]
    y = f16(y)
    return np.maximum(y, 0.0) if relu else y


VIEW = F.view_by_axes(ORDER)


class Layers(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.params, cls.natural = synthetic_params(np.random.default_rng(1))

    def test_every_layer_against_direct(self):
        rng = np.random.default_rng(2)
        for name, l in F.LAYERS.items():
            cin = natural_shape(l)[3]
            x = f16(rng.uniform(-1, 1, (cin, 5, 4)))
            for pad in ("zero", "mirror"):
                got = F.run_layer(name, x, self.params, weight_view=VIEW, pad=pad, relu_after=True, blob2="1x1")
                want = direct_conv(x, self.natural[name], self.params[name][1], pad, True)
                np.testing.assert_array_equal(got, want, err_msg="%s pad=%s" % (name, pad))
            got = F.run_layer(name, x, self.params, weight_view=VIEW, pad="replicate", relu_after=False, blob2="1x1")
            np.testing.assert_array_equal(got, direct_conv(x, self.natural[name], self.params[name][1], "replicate", False))

    def test_blob2_is_required(self):
        x = f16(np.ones((32, 2, 2)))
        with self.assertRaises(ValueError):
            F.run_layer("net1_02", x, self.params, weight_view=VIEW, pad="zero", relu_after=True)
        with self.assertRaises(ops.OpenItem):
            F.run_layer("net1_02", x, self.params, weight_view=VIEW, pad="zero", relu_after=True, blob2="2x2")
        # other layers do not need it
        F.run_layer("net1_16", x, self.params, weight_view=VIEW, pad="zero", relu_after=True)

    def test_required_arguments(self):
        x = f16(np.ones((32, 2, 2)))
        for missing in ("weight_view", "pad", "relu_after"):
            kw = {"weight_view": VIEW, "pad": "zero", "relu_after": True}
            del kw[missing]
            with self.assertRaises(TypeError, msg=missing):
                F.run_layer("net1_16", x, self.params, **kw)

    def test_weight_view_checks(self):
        x = f16(np.ones((32, 2, 2)))
        bad = lambda w, l: np.zeros((1, 1, 1, 1))   # noqa: E731
        with self.assertRaises(ValueError):
            F.run_layer("net1_16", x, self.params, weight_view=bad, pad="zero", relu_after=True)
        with self.assertRaises(ValueError):
            F.view_by_axes("out in ky")
        p = dict(self.params)
        p["net1_16"] = (p["net1_16"][0][:-1], p["net1_16"][1])
        with self.assertRaises(ValueError):
            F.run_layer("net1_16", x, p, weight_view=VIEW, pad="zero", relu_after=True)

    def test_view_by_axes_orders(self):
        rng = np.random.default_rng(3)
        l = F.LAYERS["net2_14"]
        nat = rng.standard_normal(ops.weight_shape(l))
        for order in ("out ky kx in", "out in ky kx", "in ky kx out", "ky kx in out"):
            axes = order.split()
            stored = nat.transpose([("out", "ky", "kx", "in").index(a) for a in axes]).reshape(-1)
            np.testing.assert_array_equal(F.view_by_axes(order)(stored, l), nat, err_msg=order)


class GraphSteps(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(4)
        self.params, _ = synthetic_params(np.random.default_rng(5))

    def run_one(self, step, inputs):
        return F.run_graph([step], inputs, self.params, weight_view=VIEW, pad="zero")[step["out"]]

    def test_pool_upscale_add_concat_slice(self):
        a = f16(self.rng.uniform(-2, 2, (3, 4, 6)))
        b = f16(self.rng.uniform(-2, 2, (3, 4, 6)))
        mx = a.reshape(3, 2, 2, 3, 2).max(axis=(2, 4))
        av = f16(a.reshape(3, 2, 2, 3, 2).mean(axis=(2, 4)))
        np.testing.assert_array_equal(self.run_one({"op": "pool", "in": "a", "out": "y", "kind": "max"}, {"a": a}), mx)
        np.testing.assert_array_equal(self.run_one({"op": "pool", "in": "a", "out": "y", "kind": "avg"}, {"a": a}), av)
        # bilinear, align corners: separable linear interpolation at i * (n - 1) / (2n - 1)
        def interp(v, n, axis):
            pos = np.arange(2 * n) * (n - 1) / (2 * n - 1)
            return np.apply_along_axis(lambda r: np.interp(pos, np.arange(n), r), axis, v)
        up = f16(interp(interp(a, 4, 1), 6, 2))
        got = self.run_one({"op": "upscale", "in": "a", "out": "y"}, {"a": a})
        np.testing.assert_allclose(got, up, rtol=2 ** -10, atol=1e-6)
        np.testing.assert_array_equal(self.run_one({"op": "add", "in": ["a", "b"], "out": "y"}, {"a": a, "b": b}),
                                      f16(a + b))
        np.testing.assert_array_equal(self.run_one({"op": "concat", "in": ["a", "b"], "out": "y"}, {"a": a, "b": b}),
                                      np.concatenate([a, b]))
        np.testing.assert_array_equal(self.run_one({"op": "slice", "in": "a", "out": "y", "channels": [1, 3]}, {"a": a}),
                                      a[1:3])

    def test_wiring_checks(self):
        x = {"x": f16(np.ones((32, 2, 2)))}
        conv = {"op": "conv", "layer": "net1_16", "in": "x", "out": "y", "relu": True}
        bad = [
            [dict(conv, op="dense")],
            [{k: v for k, v in conv.items() if k != "relu"}],                         # relu is required
            [dict(conv, relu=1)],
            [dict(conv, extra=1)],
            [dict(conv, **{"in": "nope"})],
            [dict(conv, layer="net9_01")],
            [conv, dict(conv, out="z", **{"in": "y"})],                              # a layer used twice
            [conv, dict(conv, layer="net1_15", out="y")],                            # a buffer written twice
            [{"op": "pool", "in": "x", "out": "p"}],                                 # pool kind is required
            [{"op": "add", "in": ["x"], "out": "s"}],
            [{"op": "upscale", "in": ["x", "x"], "out": "u"}],
        ]
        for w in bad:
            with self.assertRaises(ValueError, msg=w):
                F.run_graph(w, x, self.params, weight_view=VIEW, pad="zero")
        with self.assertRaises(ValueError):
            F.run_graph([{"op": "slice", "in": "x", "out": "s", "channels": [30, 40]}], x, self.params,
                        weight_view=VIEW, pad="zero")


class Networks(unittest.TestCase):
    """Stand-in wirings over the real layer shapes: each network result equals the same layers composed by hand with
    the direct reference. Two different wirings reuse the same parameters and layers unchanged."""

    @classmethod
    def setUpClass(cls):
        cls.params, cls.natural = synthetic_params(np.random.default_rng(6))

    def direct(self, name, x, pad, relu):
        return direct_conv(x, self.natural[name], self.params[name][1], pad, relu)

    def test_net1_stand_in(self):
        rng = np.random.default_rng(7)
        feats = f16(rng.uniform(0, 1, (10, 8, 12)))
        wiring = [
            {"op": "conv", "layer": "net1_01", "in": "cells", "out": "e0", "relu": True},     # 48 -> 32
            {"op": "conv", "layer": "net1_02", "in": "e0", "out": "e1", "relu": True},        # 32 -> 64 (1x1)
            {"op": "pool", "in": "e1", "out": "e1p", "kind": "max"},
            {"op": "conv", "layer": "net1_14", "in": "e1p", "out": "d1", "relu": True},       # 64 -> 64
            {"op": "upscale", "in": "d1", "out": "d1u"},
            {"op": "add", "in": ["d1u", "e1"], "out": "s1"},
            {"op": "conv", "layer": "net1_15", "in": "s1", "out": "d0", "relu": True},        # 64 -> 32
            {"op": "conv", "layer": "net1_16", "in": "d0", "out": "d0b", "relu": True},       # 32 -> 32
            {"op": "conv", "layer": "net1_17", "in": "d0b", "out": "out", "relu": False},     # 32 -> 16
        ]
        for pad in ("zero", "replicate"):
            got = F.net1_forward(feats, self.params, wiring=wiring, weight_view=VIEW, pad=pad, blob2="1x1",
                                 fold_order="dy dx c", output="out")
            cells = ops.fold2x2(feats, order="dy dx c")
            e0 = self.direct("net1_01", cells, pad, True)
            e1 = self.direct("net1_02", e0, pad, True)
            d1 = self.direct("net1_14", ops.pool2x(e1, kind="max"), pad, True)
            s1 = f16(ops.upscale2x_bilinear(d1) + e1)
            d0 = self.direct("net1_16", self.direct("net1_15", s1, pad, True), pad, True)
            out = self.direct("net1_17", d0, pad, False)
            want = ops.unfold2x2(out, 4, order="dy dx c")
            self.assertEqual(got.shape, (4, 8, 12))
            np.testing.assert_array_equal(got, want, err_msg=pad)
        for missing in ("wiring", "weight_view", "pad", "blob2", "fold_order", "output"):
            kw = dict(wiring=wiring, weight_view=VIEW, pad="zero", blob2="1x1", fold_order="dy dx c", output="out")
            del kw[missing]
            with self.assertRaises(TypeError, msg=missing):
                F.net1_forward(feats, self.params, **kw)
        with self.assertRaises(ValueError):    # the output must have 16 values per cell
            F.net1_forward(feats, self.params, wiring=wiring, weight_view=VIEW, pad="zero", blob2="1x1",
                           fold_order="dy dx c", output="d0")

    def test_net2_stand_in(self):
        rng = np.random.default_rng(8)
        x18 = f16(rng.uniform(-1, 1, (18, 8, 8)))
        wiring = [
            {"op": "conv", "layer": "net2_14", "in": "x18", "out": "h", "relu": True},        # 18 -> 16
            {"op": "pool", "in": "h", "out": "hp", "kind": "avg"},
            {"op": "conv", "layer": "net2_15", "in": "hp", "out": "q", "relu": True},         # 16 -> 32
            {"op": "conv", "layer": "net2_16", "in": "q", "out": "q1", "relu": True},         # 32 -> 32
            {"op": "conv", "layer": "net2_24", "in": "q1", "out": "g", "relu": False},        # 32 -> 48
            {"op": "upscale", "in": "g", "out": "gu"},
            {"op": "slice", "in": "gu", "out": "g0", "channels": [0, 16]},
            {"op": "slice", "in": "gu", "out": "g1", "channels": [16, 32]},
            {"op": "add", "in": ["g0", "g1", "h"], "out": "s"},
            {"op": "conv", "layer": "net2_25", "in": "s", "out": "y", "relu": False},         # 16 -> 8
        ]
        got = F.net2_forward({"x18": x18}, self.params, wiring=wiring, weight_view=VIEW, pad="mirror", outputs=["y", "q"])
        h = self.direct("net2_14", x18, "mirror", True)
        q = self.direct("net2_15", ops.pool2x(h, kind="avg"), "mirror", True)
        g = self.direct("net2_24", self.direct("net2_16", q, "mirror", True), "mirror", False)
        gu = ops.upscale2x_bilinear(g)
        s = f16(f16(gu[:16] + gu[16:32]) + h)
        np.testing.assert_array_equal(got["y"], self.direct("net2_25", s, "mirror", False))
        np.testing.assert_array_equal(got["q"], q)

    def test_same_layers_under_two_wirings(self):
        rng = np.random.default_rng(9)
        x = f16(rng.uniform(-1, 1, (32, 4, 4)))
        w1 = [{"op": "conv", "layer": "net1_16", "in": "x", "out": "a", "relu": True}]
        w2 = [{"op": "concat", "in": ["x", "x"], "out": "xx"},
              {"op": "slice", "in": "xx", "out": "x2", "channels": [32, 64]},
              {"op": "conv", "layer": "net1_16", "in": "x2", "out": "a", "relu": True}]
        a1 = F.run_graph(w1, {"x": x}, self.params, weight_view=VIEW, pad="zero")["a"]
        a2 = F.run_graph(w2, {"x": x}, self.params, weight_view=VIEW, pad="zero")["a"]
        np.testing.assert_array_equal(a1, a2)


class Params(unittest.TestCase):
    def test_from_blobs_and_raw(self):
        rng = np.random.default_rng(10)
        blobs = {}
        for l in L.NET1:
            blobs[l.name] = B.build_net1(l, rng.uniform(-1, 1, l.weight_elements).astype("<f2"),
                                         rng.uniform(-1, 1, l.out).astype("<f2"))
        for l in L.NET2:
            blobs[l.name + "_w"] = rng.uniform(-1, 1, l.weight_elements).astype("<f2").tobytes()
            blobs[l.name + "_b"] = rng.uniform(-1, 1, l.out).astype("<f2").tobytes()
        p = F.params_from_blobs(blobs)
        tmp = tempfile.mkdtemp(prefix="fg-raw-test-")
        try:
            for name, l in F.LAYERS.items():
                st = os.path.join(tmp, F.map_stage_name(l))
                os.makedirs(st)
                p[name][0].astype("<f4").tofile(os.path.join(st, "W_storage.bin"))
                p[name][1].astype("<f4").tofile(os.path.join(st, "b.bin" if name.startswith("net1") else "b_storage.bin"))
            r = F.params_from_raw(tmp)
            for name in F.LAYERS:
                np.testing.assert_array_equal(r[name][0].view("<u2"), p[name][0].view("<u2"))
                np.testing.assert_array_equal(r[name][1].view("<u2"), p[name][1].view("<u2"))
            self.assertEqual(F.map_stage_name(F.LAYERS["net1_01"]), "dl1.l01")
            self.assertEqual(F.map_stage_name(F.LAYERS["net2_25"]), "dl2.l25")
            # a value that is not exact in float16 is refused
            st = os.path.join(tmp, "dl2.l03")
            np.full(F.LAYERS["net2_03"].out, 0.1, dtype="<f4").tofile(os.path.join(st, "b_storage.bin"))
            with self.assertRaises(ValueError):
                F.params_from_raw(tmp)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
