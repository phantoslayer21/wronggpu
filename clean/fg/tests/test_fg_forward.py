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

from clean.fg.reference import forward as F
from clean.fg.reference import natural as N
from clean.fg.reference import ops


def f16(x):
    return np.asarray(x, dtype=np.float64).astype(np.float16).astype(np.float64)


def synthetic_params(rng, scale=0.05):
    """{layer: (canonical weights [out, k, k, in] float16, bias float16)} of every layer."""
    return {name: (rng.uniform(-scale, scale, ops.weight_shape(l)).astype(np.float16),
                   rng.uniform(-0.1, 0.1, l.out).astype(np.float16)) for name, l in F.LAYERS.items()}


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


class Layers(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.params = synthetic_params(np.random.default_rng(1))

    def test_every_layer_against_direct(self):
        rng = np.random.default_rng(2)
        for name, l in F.LAYERS.items():
            w, b = self.params[name]
            x = f16(rng.uniform(-1, 1, (w.shape[3], 5, 4)))
            for pad in ("zero", "mirror"):
                got = F.run_layer(name, x, self.params, pad=pad, relu_after=True)
                np.testing.assert_array_equal(got, direct_conv(x, w, b, pad, True), err_msg="%s pad=%s" % (name, pad))
            got = F.run_layer(name, x, self.params, pad="replicate", relu_after=False)
            np.testing.assert_array_equal(got, direct_conv(x, w, b, "replicate", False))

    def test_shapes(self):
        self.assertEqual(self.params["net1_02"][0].shape, (64, 1, 1, 32))   # blob 2: 1 x 1
        self.assertEqual(self.params["net2_01"][0].shape, (32, 3, 3, 10))   # 10 input channels carry data
        self.assertEqual(self.params["net2_14"][0].shape, (16, 3, 3, 18))

    def test_required_arguments(self):
        x = f16(np.ones((32, 2, 2)))
        for missing in ("pad", "relu_after"):
            kw = {"pad": "zero", "relu_after": True}
            del kw[missing]
            with self.assertRaises(TypeError, msg=missing):
                F.run_layer("net1_16", x, self.params, **kw)

    def test_parameter_checks(self):
        x = f16(np.ones((32, 2, 2)))
        p = dict(self.params)
        p["net1_16"] = (p["net1_16"][0][:, :, :, :-1], p["net1_16"][1])
        with self.assertRaises(ValueError):
            F.run_layer("net1_16", x, p, pad="zero", relu_after=True)
        with self.assertRaises(KeyError):
            F.run_layer("net9_01", x, self.params, pad="zero", relu_after=True)


class GraphSteps(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(4)
        self.params = synthetic_params(np.random.default_rng(5))

    def run_one(self, step, inputs):
        return F.run_graph([step], inputs, self.params, pad="zero")[step["out"]]

    def test_pool_upscale_add_concat_slice(self):
        a = f16(self.rng.uniform(-2, 2, (3, 4, 6)))
        b = f16(self.rng.uniform(-2, 2, (3, 4, 6)))
        mx = a.reshape(3, 2, 2, 3, 2).max(axis=(2, 4))
        av = f16(a.reshape(3, 2, 2, 3, 2).mean(axis=(2, 4)))
        np.testing.assert_array_equal(self.run_one({"op": "pool", "in": "a", "out": "y", "kind": "max"}, {"a": a}), mx)
        np.testing.assert_array_equal(self.run_one({"op": "pool", "in": "a", "out": "y", "kind": "avg"}, {"a": a}), av)

        def interp(v, n, axis):   # bilinear, align corners: linear interpolation at i * (n - 1) / (2n - 1)
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
                F.run_graph(w, x, self.params, pad="zero")
        with self.assertRaises(ValueError):
            F.run_graph([{"op": "slice", "in": "x", "out": "s", "channels": [30, 40]}], x, self.params, pad="zero")


class Networks(unittest.TestCase):
    """Stand-in wirings over the real layer shapes: each network result equals the same layers composed by hand with
    the direct reference. Two different wirings reuse the same parameters and layers unchanged."""

    @classmethod
    def setUpClass(cls):
        cls.params = synthetic_params(np.random.default_rng(6))

    def direct(self, name, x, pad, relu):
        w, b = self.params[name]
        return direct_conv(x, w, b, pad, relu)

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
        # the generic engine (kept for net2) still runs any wiring over net1's layers; net1 itself has its delivered
        # wiring (clean/fg/reference/net1.py, tested in test_fg_net1.py)
        cells = ops.fold2x2(feats, order="c dy dx")
        for pad in ("zero", "replicate"):
            got = F.run_graph(wiring, {"cells": cells}, self.params, pad=pad)["out"]
            e1 = self.direct("net1_02", self.direct("net1_01", cells, pad, True), pad, True)
            d1 = self.direct("net1_14", ops.pool2x(e1, kind="max"), pad, True)
            s1 = f16(ops.upscale2x_bilinear(d1) + e1)
            d0 = self.direct("net1_16", self.direct("net1_15", s1, pad, True), pad, True)
            np.testing.assert_array_equal(got, self.direct("net1_17", d0, pad, False), err_msg=pad)
        with self.assertRaises(TypeError):     # net1_forward takes no wiring: it has its own
            F.net1_forward(feats, self.params, wiring=wiring)

    def test_net2_stand_in(self):
        rng = np.random.default_rng(8)
        x10 = f16(rng.uniform(-1, 1, (10, 8, 8)))
        x18 = f16(rng.uniform(-1, 1, (18, 8, 8)))
        wiring = [
            {"op": "conv", "layer": "net2_01", "in": "x10", "out": "a", "relu": True},        # 10 -> 32
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
        got = F.net2_forward({"x10": x10, "x18": x18}, self.params, wiring=wiring, pad="mirror", outputs=["y", "q", "a"])
        h = self.direct("net2_14", x18, "mirror", True)
        q = self.direct("net2_15", ops.pool2x(h, kind="avg"), "mirror", True)
        g = self.direct("net2_24", self.direct("net2_16", q, "mirror", True), "mirror", False)
        gu = ops.upscale2x_bilinear(g)
        s = f16(f16(gu[:16] + gu[16:32]) + h)
        np.testing.assert_array_equal(got["y"], self.direct("net2_25", s, "mirror", False))
        np.testing.assert_array_equal(got["q"], q)
        np.testing.assert_array_equal(got["a"], self.direct("net2_01", x10, "mirror", True))

    def test_same_layers_under_two_wirings(self):
        rng = np.random.default_rng(9)
        x = f16(rng.uniform(-1, 1, (32, 4, 4)))
        w1 = [{"op": "conv", "layer": "net1_16", "in": "x", "out": "a", "relu": True}]
        w2 = [{"op": "concat", "in": ["x", "x"], "out": "xx"},
              {"op": "slice", "in": "xx", "out": "x2", "channels": [32, 64]},
              {"op": "conv", "layer": "net1_16", "in": "x2", "out": "a", "relu": True}]
        a1 = F.run_graph(w1, {"x": x}, self.params, pad="zero")["a"]
        a2 = F.run_graph(w2, {"x": x}, self.params, pad="zero")["a"]
        np.testing.assert_array_equal(a1, a2)


class RawParams(unittest.TestCase):
    def test_from_raw(self):
        m = N.load_map()
        p = synthetic_params(np.random.default_rng(10))
        tmp = tempfile.mkdtemp(prefix="fg-raw-test-")
        try:
            for name, l in F.LAYERS.items():
                st = os.path.join(tmp, F.map_stage_name(l))
                os.makedirs(st)
                w = p[name][0]
                nat = w.transpose(0, 3, 1, 2) if w.shape[1] == 3 else w[:, 0, 0, :]   # canonical -> map shape
                self.assertEqual(nat.shape, N.weight_shape(l, m))
                nat.astype("<f4").tofile(os.path.join(st, "W.bin"))
                p[name][1].astype("<f4").tofile(os.path.join(st, "b.bin"))
            r = F.params_from_raw(tmp, m)
            for name in F.LAYERS:
                np.testing.assert_array_equal(r[name][0].view("<u2"), p[name][0].view("<u2"))
                np.testing.assert_array_equal(r[name][1].view("<u2"), p[name][1].view("<u2"))
            st = os.path.join(tmp, "dl2.l03")
            np.full(F.LAYERS["net2_03"].out, 0.1, dtype="<f4").tofile(os.path.join(st, "b.bin"))
            with self.assertRaises(ValueError):      # not exact in float16
                F.params_from_raw(tmp, m)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
