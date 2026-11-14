# SPDX-License-Identifier: MIT
"""net1's matrix fast path as planned on the CPU (clean/fg/runner/fg_net1_mma.py): the weight fragments, the fused
plan's buffers and launches."""
import os
import sys
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "clean", "fg", "runner"))

import fg_net1_gpu as G  # noqa: E402
import fg_net1_mma as M  # noqa: E402
from clean.fg.reference import net1 as N  # noqa: E402


def chunk_tap(q, order, k, nb):
    if order == "ascending":
        return q, 0, 0
    if order == "block_kx_ky":
        cb, t = divmod(q, k * k)
        kx, ky = divmod(t, k)
        return cb, ky, kx
    kx, r = divmod(q, nb * k)
    cb, ky = divmod(r, k)
    return cb, ky, kx


class SwizzleTest(unittest.TestCase):
    def test_every_fragment_element(self):
        rng = np.random.default_rng(2)
        for cout, k, cin, order in ((32, 3, 48, "block_kx_ky"), (256, 3, 128, "kx_block_ky"), (16, 1, 32, "ascending")):
            w = rng.uniform(-1, 1, (cout, k, k, cin)).astype(np.float16)
            nb, nq = cin // 16, (cin // 16) * k * k
            f = np.frombuffer(M.swizzle(w, order), np.float16).reshape(cout // 16, nq, 32, 8)
            for _ in range(500):
                mt, q, lane, j = rng.integers(cout // 16), rng.integers(nq), rng.integers(32), rng.integers(8)
                cb, ky, kx = chunk_tap(int(q), order, k, nb)
                o = 16 * mt + lane % 16
                c = 16 * cb + 8 * (lane // 16) + j
                self.assertEqual(f[mt, q, lane, j], w[o, ky, kx, c])


class PlanTest(unittest.TestCase):
    def test_buffers_and_launches(self):
        params = G.synthetic_params(np.random.default_rng(3))
        for gh, gw in ((64, 64), (70, 90)):
            cells = N.fold(np.zeros((10, gh, gw)))
            H0, W0 = cells.shape[1:]
            ref = N.run(cells, params)
            data, acts, steps = M.plan(H0, W0, params)
            self.assertEqual(len(steps), 27)                        # 32 minus the five fused pools
            self.assertEqual(sum(s[4] == "conv" for s in steps), 17)
            self.assertEqual(set(acts), set(ref) - {"cells"})
            for out, (shape, nbytes) in acts.items():
                self.assertEqual(tuple(np.array(ref[out].shape)[[1, 2, 0]]), shape, out)   # HWC
                self.assertEqual(nbytes, 2 * int(np.prod(shape)))
            for kernel, grid, args, outs, op in steps:
                self.assertIn(kernel, M.KERNELS)
                if op == "conv":
                    cout, H, W, mt = args[6], args[7], args[8], int(kernel[8])
                    self.assertEqual(grid, -(-W // 16) * -(-H // 16) * (cout // (16 * mt)))
                    self.assertEqual(len(args), 15)
            pooled = [s[3] for s in steps if len(s[3]) == 2]
            self.assertEqual(pooled, [["a0", "p1"], ["a1", "p2"], ["a2", "p3"], ["a3", "p4"], ["a4", "p5"]])


if __name__ == "__main__":
    unittest.main()
