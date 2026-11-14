# SPDX-License-Identifier: MIT
"""The GPU chain of net1 (clean/fg/runner/fg_net1_gpu.py) as planned on the CPU: buffers, launches and chain lines."""
import os
import sys
import tempfile
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "clean", "fg", "runner"))

import fg_net1_gpu as G  # noqa: E402
from clean.fg.reference import net1 as N  # noqa: E402

ARITY = {"fg_conv_f16": 11, "fg_maxpool2_f16": 5, "fg_upscale_f16": 9, "fg_add_f16": 4}


class PlanTest(unittest.TestCase):
    def setUp(self):
        self.params = G.synthetic_params(np.random.default_rng(3))

    def test_every_step_output_has_the_reference_shape(self):
        for gh, gw in ((64, 64), (70, 90), (65, 129)):
            cells = N.fold(np.zeros((10, gh, gw)))
            data, acts, steps = G.plan(cells.shape[1], cells.shape[2], self.params)
            self.assertEqual(len(steps), 32)
            self.assertEqual([s[3] for s in steps], [s["out"] for s in N.NET1_WIRING])
            ref = N.run(cells, self.params)
            for out, (shape, nbytes) in acts.items():
                self.assertEqual(ref[out].shape, shape, out)
                self.assertEqual(nbytes, 2 * int(np.prod(shape)))

    def test_launch_arguments(self):
        data, acts, steps = G.plan(32, 32, self.params)
        ids = set(data) | set(acts) | {"cells"}
        for kernel, grid, args, out, op in steps:
            self.assertEqual(len(args), ARITY[kernel], kernel)
            self.assertEqual(grid, -(-int(np.prod(acts[out][0])) // 256))
            for a in args:
                if isinstance(a, str):
                    self.assertIn(a[1:], ids)
                else:
                    self.assertTrue(0 <= a < 2 ** 32)
        self.assertEqual(sum(s[0] == "fg_conv_f16" for s in steps), 17)
        self.assertEqual(sum(s[0] == "fg_add_f16" for s in steps), 5)

    def test_weights_are_natural_out_in_ky_kx(self):
        data, _, _ = G.plan(32, 32, self.params)
        w, _ = self.params["net1_03"]                        # canonical [out, k, k, in]
        nat = np.frombuffer(data["w_net1_03"], np.float16).reshape(w.shape[0], w.shape[3], 3, 3)
        self.assertEqual(nat[5, 7, 2, 1], w[5, 2, 1, 7])

    def test_chain_lines(self):
        data, acts, steps = G.plan(32, 32, self.params)
        with tempfile.TemporaryDirectory() as d:
            files = G.write_files(d, "t", {"cells": b"\0" * 8, **data})
            full = G.chain_lines("t", files, acts, steps)
            neg = G.chain_lines("n", files, acts, steps, half=2, stop=2, saves=[steps[2][3]], time_runs=3)
        self.assertEqual(sum(l.startswith("step ") for l in full), 32)
        self.assertEqual(sum(l.startswith("save ") for l in full), 32)
        nsteps = [l for l in neg if l.startswith("step ")]
        self.assertEqual(len(nsteps), 3)
        self.assertEqual(int(nsteps[2].split()[2]), max(1, steps[2][1] // 2))
        self.assertIn("time 3", neg)


if __name__ == "__main__":
    unittest.main()
