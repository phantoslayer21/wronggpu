# SPDX-License-Identifier: GPL-3.0-only
"""The bias split of the exact step Y (clean/rr_f/gpu/layout.py bias_split and pack_weights' Tb split): the same bytes as
the runtime's host split (runtime/src/rr_pack.cpp rne_int / split3), so tools/pack_model.py and the runtime pack a model
identically (runtime/tests/test_model_pack.py). The constructed values cover the cases where numpy's round and the
runtime's floor-based rule give different zeros: negatives in [-0.5, 0) (cb = +0.0), -0.0 (cb = -0.0), and ties.
CPU only; constructed values. Run: python -m unittest clean.rr_f.tests.test_layout_bias_split
"""
import math
import os
import sys
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from clean.rr_f.gpu import layout  # noqa: E402

# f16 values (exact in f16 and f32)
VALUES = [-0.3, -0.5, -0.25, -0.0, 0.0, 0.25, 0.5, 1.5, -1.5, 2.5, -2.5, -0.75, 0.75, -3.0, 3.5, -1024.5, 65504.0,
          -65504.0, 2 ** -24, -(2 ** -24)]


def rne_int_cpp(x):
    """rr_pack.cpp rne_int, line for line."""
    if not math.isfinite(x):
        return x
    r = float(np.floor(x))   # C's floor keeps -0.0 (Python's math.floor would return the integer 0)
    d = x - r
    if d > 0.5 or (d == 0.5 and math.fmod(r, 2.0) != 0.0):
        r += 1.0
    return r


def split3_cpp(v):
    """rr_pack.cpp split3 on f32 values (the exact widening of the f16 bias)."""
    v = [float(np.float32(x)) for x in v]
    cb = [rne_int_cpp(x) for x in v]
    return np.array(v + cb + [x - c for x, c in zip(v, cb)], np.float32)


def bits(a):
    return np.asarray(a, np.float32).view(np.uint32)


class BiasSplit(unittest.TestCase):
    def test_bytes_equal_the_runtime_split(self):
        v = np.array(VALUES, np.float16).astype(np.float64)
        got = layout.bias_split(v)
        np.testing.assert_array_equal(bits(got), bits(split3_cpp(v)))

    def test_zero_signs(self):
        n = len(VALUES)
        got = layout.bias_split(np.array(VALUES, np.float16).astype(np.float64))
        cb = got[n:2 * n]
        for x, c in zip(VALUES, cb):
            if -0.5 <= x < 0:   # negative, integer part zero: +0.0 as the runtime (numpy's round: -0.0)
                self.assertEqual(int(bits(c)), 0, x)
            if x == 0 and math.copysign(1.0, x) < 0:   # -0.0 stays -0.0 (floor keeps it)
                self.assertEqual(int(bits(c)), 0x80000000)

    def test_split_is_exact_and_ties_to_even(self):
        v = np.array(VALUES, np.float16).astype(np.float64)
        n = v.size
        got = layout.bias_split(v).astype(np.float64)
        vals, cb, cs = got[:n], got[n:2 * n], got[2 * n:]
        np.testing.assert_array_equal(vals, v)
        np.testing.assert_array_equal(cb + cs, v)
        self.assertTrue(np.all(np.abs(cs) <= 0.5))
        np.testing.assert_array_equal(cb, np.round(v))   # the same integers as numpy (only zero signs differ)

    def test_non_finite_values_pass_through(self):
        got = layout.bias_split(np.array([np.inf, -np.inf, np.nan]))
        self.assertTrue(np.isinf(got[3]) and np.isinf(got[4]) and np.isnan(got[5]))

    def test_score_bias_split(self):
        """pack_weights' Tb split uses the same rule: a constructed stage with one negative small score bias."""
        C, H = 32, 1
        W = {"Wq": np.zeros((H, C, 32), np.uint8), "Wv": np.zeros((H, C, 32), np.uint8),
             "Tb": np.zeros((H, 12, 16, 16)), "Wo": np.zeros((H, 32, C), np.uint8), "bo": np.zeros(C),
             "W1": np.zeros((C, 4 * C), np.uint8), "b1": np.zeros(4 * C), "W2": np.zeros((4 * C, C), np.uint8),
             "b2": np.zeros(C)}
        W["Tb"][0, 0, 0, 0] = -0.3
        W["Tb"][0, 0, 1, 2] = -0.5
        tb = layout.pack_weights(W, C, H)["Tb"]
        n = tb.size // 3
        vals = tb[:n].astype(np.float64)
        np.testing.assert_array_equal(bits(tb), bits(split3_cpp(vals)))
        self.assertTrue(np.any(vals == np.float32(-0.3)) and np.all(bits(tb[n:2 * n][vals < 0]) == 0))


if __name__ == "__main__":
    unittest.main()
