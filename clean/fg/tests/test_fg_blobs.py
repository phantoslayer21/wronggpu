# SPDX-License-Identifier: MIT
"""Frame generation (#96): splitting weight blobs (clean/fg/reference/blobs.py), on synthetic blobs only.

    python -m unittest clean.fg.tests.test_fg_blobs
"""
import struct
import unittest

import numpy as np

from clean.fg import layout as L
from clean.fg.reference import blobs as B


def rand_halves(rng, n):
    return rng.uniform(-2, 2, n).astype("<f2")


def synthetic_blobs(rng):
    out = {}
    for l in L.NET1:
        out[l.name] = B.build_net1(l, rand_halves(rng, l.weight_elements), rand_halves(rng, l.out))
    for l in L.NET2:
        out[l.name + "_w"] = rand_halves(rng, l.weight_elements).tobytes()
        out[l.name + "_b"] = rand_halves(rng, l.out).tobytes()
    return out


class SplitNet1(unittest.TestCase):
    def test_round_trip_and_layout(self):
        rng = np.random.default_rng(1)
        l = L.NET1[0]
        w, b = rand_halves(rng, l.weight_elements), rand_halves(rng, l.out)
        blob = B.build_net1(l, w, b)
        self.assertEqual(len(blob), l.blob_bytes)
        gw, gb = B.split_net1(l, blob)
        np.testing.assert_array_equal(gw.view("<u2"), w.view("<u2"))
        np.testing.assert_array_equal(gb.view("<u2"), b.view("<u2"))
        # the bias of channel c is at element bias_offset + net1_bias_index(c), read here byte by byte
        for c in (0, 7, 8, l.out - 1):
            e = l.bias_offset + L.net1_bias_index(c)
            self.assertEqual(blob[2 * e:2 * e + 2], struct.pack("<e", float(b[c])))

    def test_checks(self):
        rng = np.random.default_rng(2)
        l = L.NET1[16]
        blob = bytearray(B.build_net1(l, rand_halves(rng, l.weight_elements), rand_halves(rng, l.out)))
        # -0 in the zero area is still zero
        neg = bytearray(blob)
        neg[2 * l.zero_offset + 1] = 0x80
        B.split_net1(l, bytes(neg))
        bad_zero = bytearray(blob)
        bad_zero[2 * (l.bias_offset - 1)] = 0x01
        with self.assertRaises(B.BlobError):
            B.split_net1(l, bytes(bad_zero))
        B.split_net1(l, bytes(bad_zero), check=False)
        bad_copy = bytearray(blob)
        e = l.bias_offset + L.net1_bias_index(9) + 5 * L.NET1_BIAS_GROUP   # copy 5 of channel 9
        bad_copy[2 * e] ^= 0x01
        with self.assertRaises(B.BlobError):
            B.split_net1(l, bytes(bad_copy))
        _, gb = B.split_net1(l, bytes(bad_copy), check=False)   # copy 0 is still taken
        _, want = B.split_net1(l, bytes(blob))
        np.testing.assert_array_equal(gb.view("<u2"), want.view("<u2"))
        with self.assertRaises(B.BlobError):
            B.split_net1(l, bytes(blob[:-2]))


class SplitAll(unittest.TestCase):
    def test_every_layer(self):
        rng = np.random.default_rng(3)
        blobs = synthetic_blobs(rng)
        self.assertEqual(sum(len(v) for v in blobs.values()), L.totals()["bytes"])
        parts = B.split_all(blobs)
        self.assertEqual(len(parts), len(L.NET1) + len(L.NET2))
        for l in L.NET1:
            w, b = parts[l.name]
            self.assertEqual((w.size, b.size, w.dtype.str, b.dtype.str), (l.weight_elements, l.out, "<f2", "<f2"))
            self.assertEqual(B.build_net1(l, w, b), blobs[l.name])
        for l in L.NET2:
            w, b = parts[l.name]
            self.assertEqual(w.tobytes(), blobs[l.name + "_w"])
            self.assertEqual(b.tobytes(), blobs[l.name + "_b"])
        self.assertEqual(sum(w.size + b.size for w, b in parts.values()),
                         L.totals()["net1_params"] + L.totals()["net2_params"])

    def test_blob_set_and_sizes(self):
        rng = np.random.default_rng(4)
        blobs = synthetic_blobs(rng)
        less = dict(blobs)
        del less["net2_25_b"]
        with self.assertRaises(B.BlobError):
            B.split_all(less)
        with self.assertRaises(B.BlobError):
            B.split_all(dict(blobs, extra=b"\0\0"))
        l = L.NET2[0]
        with self.assertRaises(B.BlobError):
            B.split_net2(l, blobs[l.name + "_w"][:-2], blobs[l.name + "_b"])
        with self.assertRaises(B.BlobError):
            B.split_net2(l, blobs[l.name + "_w"], blobs[l.name + "_b"] + b"\0\0")


if __name__ == "__main__":
    unittest.main()
