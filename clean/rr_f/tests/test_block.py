# SPDX-License-Identifier: MIT
"""Spec 51.3 window block against the slow oracle, and the 51.6 grid / border rule."""
import unittest

import numpy as np

from clean.rr_f.reference.block import Grid, reflect, score_bias, window_block
from clean.rr_f.reference.fmt import d8, q8
from clean.rr_f.tests import slow, synth
from clean.rr_f.tests.slow_stage import _decode_w


class TestBorders(unittest.TestCase):
    def test_reflect_exhaustive(self):
        for N in range(1, 41):
            v = np.arange(-200, 241)
            got = reflect(v, N)
            ref = np.array([slow.reflect(int(x), N) for x in v])
            np.testing.assert_array_equal(got, ref, N)
            self.assertTrue(((got >= 0) & (got < N)).all())
        np.testing.assert_array_equal(reflect(np.array([-1, 10]), 10), [1, 8])     # -1 -> 1, N -> N-2

    def test_single_reflection_range(self):
        # 51.6 minimum size: for -(N-1) <= v <= 2(N-1) the rule is the single reflection
        for N in range(2, 30):
            for v in range(-(N - 1), 2 * (N - 1) + 1):
                single = -v if v < 0 else (2 * (N - 1) - v if v >= N else v)
                self.assertEqual(int(reflect(np.array([v]), N)[0]), single)

    def test_read_reach(self):
        # with even shifts below 8 the reads of a stage reach at most 6 below 0 and 7 beyond N-1 (own level),
        # and the coarse reads of a fusion half at most 4 below 0 / 4 beyond M-1
        for N in range(1, 40):
            for s in (0, 2, 4, 6):
                g = Grid(N, 1, s, 0)
                X = g.X[:, :8]
                self.assertGreaterEqual(X.min(), -6)
                self.assertLessEqual(X.max(), N - 1 + 7)
                if N % 2 == 0:
                    Xc = np.floor_divide(X, 2)
                    self.assertGreaterEqual(Xc.min(), -4)
                    self.assertLessEqual(Xc.max(), N // 2 - 1 + 4)

    def test_grid_covers_each_token_once(self):
        for Nx, Ny in ((1, 1), (2, 3), (8, 8), (9, 17), (32, 32)):
            for sx in (0, 2, 4, 6):
                for sy in (0, 6):
                    g = Grid(Nx, Ny, sx, sy)
                    vals = np.zeros((g.nw, 64, 1))
                    g.scatter(vals, 0)        # asserts once-only coverage
                    self.assertEqual(int(g.real.sum()), Nx * Ny)
                    # every window holds at least one real token
                    self.assertTrue(g.real.any(axis=1).all())

    def test_odd_shift_rejected(self):
        with self.assertRaises(ValueError):
            Grid(8, 8, 1, 0)
        with self.assertRaises(ValueError):
            Grid(8, 8, 8, 0)


class TestWindowBlock(unittest.TestCase):
    def _run(self, name_like, wflavour, wq_scale=1.0, seed=7):
        rng = np.random.default_rng(seed)
        wc, _ = synth.window_consts(rng, wflavour)
        W = synth.stage_weights(rng, name_like)
        if wq_scale != 1.0:
            W["Wq"] = q8(d8(W["Wq"]) * wq_scale)
        C = W["W2"].shape[1]
        x = synth.e4m3(rng, (1, 64, C), 1.0, 0.02, 0.001)
        xr = d8(x)
        consts = synth.stage_consts(rng)
        y2, info = window_block(x, xr, W, consts["ra"], consts["m"], wc, trace=True)
        ys, Zs = slow.window_block(d8(x[0]).tolist(), xr[0].tolist(), _decode_w(W), consts["ra"], consts["m"], wc)
        got = d8(y2[0])
        ref = np.array(ys)
        same = (got == ref) | (np.isnan(got) & np.isnan(ref))
        self.assertTrue(same.all(), np.argwhere(~same)[:5])
        np.testing.assert_array_equal(info["Z"][0], np.array(Zs))
        return info

    def test_vs_slow_c32(self):
        self._run("IN0", "A")

    def test_vs_slow_c64_buildB(self):
        self._run("D1-R", "B", seed=8)

    def test_z_overflow_rows(self):
        info = self._run("IN0", "zovf", wq_scale=6.0, seed=9)
        Z = info["Z"]
        self.assertTrue(np.isinf(Z).any(), "test needs rows with Z = inf")
        P = info["P"]
        rows = np.isinf(Z)
        self.assertTrue((d8(P)[rows] == 0).all())

    def test_z_override_identity(self):
        rng = np.random.default_rng(11)
        wc, _ = synth.window_consts(rng, "A")
        W = synth.stage_weights(rng, "E2")
        x = synth.e4m3(rng, (3, 64, 96), 1.0)
        y2, info = window_block(x, d8(x), W, 1.0, 1.0, wc)
        y2b, _ = window_block(x, d8(x), W, 1.0, 1.0, wc, Z_override=info["Z"])
        np.testing.assert_array_equal(y2, y2b)

    def test_score_bias(self):
        rng = np.random.default_rng(12)
        Tb = rng.normal(size=(12, 16, 16))
        B = score_bias(Tb)
        for a in range(0, 64, 3):
            for b in range(0, 64, 5):
                self.assertEqual(B[a, b], slow.bias(Tb, a, b))
        # the transposed blocks: B(a, b) for (I, J) = (2, 0) is block (0, 2) transposed
        self.assertEqual(B[33, 2], Tb[4][2][1])


if __name__ == "__main__":
    unittest.main()
