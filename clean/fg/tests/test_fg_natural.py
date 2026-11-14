# SPDX-License-Identifier: MIT
"""Frame generation (#96): the natural weight order from the converter map (clean/fg/reference/natural.py), on
synthetic blobs: natural weights scattered into stored blobs through the map's own gather indices come back unchanged
through blobs.split_all and the natural view, for every layer.

    python -m unittest clean.fg.tests.test_fg_natural
"""
import unittest

import numpy as np

from clean.fg import layout as L
from clean.fg.reference import blobs as B
from clean.fg.reference import forward as F
from clean.fg.reference import natural as N


class NaturalView(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.m = N.load_map()

    def test_gathers_cover_the_weight_spans(self):
        for l in list(L.NET1) + list(L.NET2):
            idx = N.gather_index([t for t in N._stage(self.m, l)["tensors"] if t["name"] == "W"][0]).reshape(-1)
            self.assertEqual(len(np.unique(idx)), idx.size, l.name)          # no element read twice
            self.assertTrue(idx.min() >= 0 and idx.max() < l.weight_elements, l.name)
            if l.name == "net2_01":
                self.assertEqual(idx.size, 9 * l.cin_used * l.out)          # 10 of 16 stored inputs are read
            else:
                self.assertEqual(idx.size, l.weight_elements, l.name)       # a permutation of the span

    def test_shapes(self):
        for l in L.NET1:
            want = (l.out, l.cin, 3, 3) if l.taps == 9 else (l.out, l.cin)
            self.assertEqual(N.weight_shape(l, self.m), want, l.name)
        for l in L.NET2:
            self.assertEqual(N.weight_shape(l, self.m), (l.out, l.cin_used, 3, 3), l.name)

    def test_bias_indices(self):
        for l in L.NET1:
            np.testing.assert_array_equal(N.natural_bias_index(l, self.m),
                                          [l.bias_offset + L.net1_bias_index(c) for c in range(l.out)])
        for l in L.NET2:
            np.testing.assert_array_equal(N.natural_bias_index(l, self.m), np.arange(l.out))

    def test_round_trip_through_blobs(self):
        rng = np.random.default_rng(1)
        blobs, want = {}, {}
        for l in list(L.NET1) + list(L.NET2):
            nat = rng.uniform(-1, 1, N.weight_shape(l, self.m)).astype("<f2")
            bias = rng.uniform(-1, 1, l.out).astype("<f2")
            stored = np.zeros(l.weight_elements, dtype="<f2")
            t = [t for t in N._stage(self.m, l)["tensors"] if t["name"] == "W"][0]
            stored[N.gather_index(t)] = nat
            if l in L.NET1:
                blobs[l.name] = B.build_net1(l, stored, bias)
            else:
                blobs[l.name + "_w"] = stored.tobytes()
                blobs[l.name + "_b"] = bias.tobytes()
            want[l.name] = (N.to_canonical(nat), bias)
        got = F.params_from_blobs(blobs, self.m)
        for name, (w, b) in want.items():
            np.testing.assert_array_equal(got[name][0].view("<u2"), w.view("<u2"), err_msg=name)
            np.testing.assert_array_equal(got[name][1].view("<u2"), b.view("<u2"), err_msg=name)
            self.assertEqual(got[name][0].shape, w.shape)

    def test_to_canonical(self):
        a = np.arange(2 * 3 * 3 * 3).reshape(2, 3, 3, 3)          # out, in, ky, kx
        c = N.to_canonical(a)
        self.assertEqual(c.shape, (2, 3, 3, 3))
        self.assertEqual(c[1, 2, 0, 1], a[1, 1, 2, 0])            # [o, ky, kx, i] = natural [o, i, ky, kx]
        self.assertEqual(N.to_canonical(np.zeros((4, 5))).shape, (4, 1, 1, 5))
        with self.assertRaises(ValueError):
            N.to_canonical(np.zeros(3))


if __name__ == "__main__":
    unittest.main()
