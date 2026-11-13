# SPDX-License-Identifier: MIT
"""The float-float log2 / exp2 model (gpu/ffmath.py) of the input kernel's fast path: unflagged results equal the
correctly rounded reference, and the pair's relative error stays below the flag threshold EPS_REL."""
import unittest

import numpy as np

import clean.rr_f.gpu.ffmath as FF
from clean.rr_f.reference import f32 as F


class TestFFMath(unittest.TestCase):
    def capture(self, fn, x):
        cap = {}
        orig = FF.near_mid

        def nm(H, L):
            cap["H"], cap["L"] = H, L
            return orig(H, L)
        FF.near_mid = nm
        try:
            out, flag = fn(x)
        finally:
            FF.near_mid = orig
        return out, flag, cap["H"].astype(np.float64) + cap["L"].astype(np.float64)

    def test_log2(self):
        rng = np.random.default_rng(5600)
        m1 = (np.arange(1, 1 << 23, 13, dtype=np.uint32) | 0x3F800000).view(np.float32)
        m2 = (np.arange(1, 1 << 23, 13, dtype=np.uint32) | 0x3F000000).view(np.float32)
        r = (np.abs(rng.normal(0, 1, 400000)) * np.exp2(rng.integers(-120, 120, 400000))).astype(np.float32)
        x = np.concatenate([m1, m2, r])
        x = x[((x.view(np.uint32) & 0x7FFFFF) != 0) & (x >= np.float32(2.0 ** -126))]
        H, flag, pair = self.capture(FF.log2_ff, x)
        ref = F.log2(x)
        self.assertEqual(int(((H != ref) & ~flag).sum()), 0)
        rel = np.abs(pair - np.log2(x.astype(np.float64))) / np.abs(np.log2(x.astype(np.float64)))
        self.assertLess(rel.max(), FF.EPS_REL / 4)
        self.assertLess(flag.mean(), 1e-4)

    def test_exp2(self):
        rng = np.random.default_rng(5601)
        t = np.concatenate([rng.uniform(-125.9, 127.9, 600000), rng.uniform(-1, 1, 400000),
                            rng.uniform(-0.01, 0.01, 100000)]).astype(np.float32)
        t = t[t != np.floor(t)]
        H, flag, pair = self.capture(FF.exp2_ff, t)
        ref = F.exp2(t)
        self.assertEqual(int(((H != ref) & ~flag).sum()), 0)
        n = np.floor(t).astype(np.float64)
        n = np.where(t.astype(np.float64) - n > 0.5, n + 1, n)
        ex = np.exp2(t.astype(np.float64) - n)
        self.assertLess((np.abs(pair - ex) / ex).max(), FF.EPS_REL / 4)
        self.assertLess(flag.mean(), 1e-4)


if __name__ == "__main__":
    unittest.main()
