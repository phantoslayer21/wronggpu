# SPDX-License-Identifier: MIT
"""Frame generation (#96): net2's D3D12 check (clean/fg/runner/fg_net2_d3d12_check.py) as planned on the CPU: the chain's
launches against the kernels' declarations, every buffer reference, the output sizes against the reference's buffers,
the workgroup counts, and file names that stay distinct on a case-insensitive file system. No GPU.

    python -m unittest clean.fg.tests.test_fg_net2_gpu_plan
"""
import os
import re
import sys
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, os.path.join(ROOT, "clean", "fg", "runner"))

import fg_net2_d3d12_check as K  # noqa: E402

KDIR = os.path.join(ROOT, "clean", "dx12", "kernels")


def declarations():
    out = {}
    for k in K.KERNELS:
        with open(os.path.join(KDIR, k + ".hlsl"), encoding="utf-8") as f:
            m = re.search(r"@wg-kernel\s+(\S+)\s+co=(\S+).*args=(\S+)", f.read())
        out[m.group(1)] = (m.group(2), m.group(3).split(","))
    return out


class Plan(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(7)
        cls.p = K.synthetic_params(rng)
        cls.ib, cls.ic, cls.ia = (K._rand(rng, (32, 48, c), 8.0) for c in (4, 4, 1))
        cls.case = K.chain_case("plan 48x32", 48, 32, cls.p, cls.ib, cls.ic, cls.ia, np.float32(0.5))
        cls.decl = declarations()

    def test_declarations(self):
        self.assertEqual(sorted(self.decl), sorted(K.KERNELS))
        self.assertTrue(all(co == K.CO for co, _ in self.decl.values()))

    def check_steps(self, case):
        for kernel, grid, args in case.steps:
            _, kinds = self.decl[kernel]
            self.assertEqual(len(args), len(kinds), kernel)
            for a, kind in zip(args, kinds):
                if kind == "p":
                    self.assertTrue(a == "null" or a[1:] in case.data or a[1:] in case.outs, (kernel, a))
                else:
                    self.assertTrue(re.fullmatch(r"-?\d+", a), (kernel, a))
                    self.assertLess(int(a), 1 << 32)
            self.assertGreaterEqual(grid, 1)

    def test_chain(self):
        c = self.case
        self.assertEqual(len(c.steps), 38)
        self.check_steps(c)
        for i, n in c.outs.items():
            self.assertEqual(n, 2 * c.expect[i].size, i)
        lower = [i.lower() for i in list(c.outs) + list(c.data)]
        self.assertEqual(len(lower), len(set(lower)), "buffer ids must differ ignoring case (they name files)")
        # the output covers every element: the three L25 launches write channels 0-3, 4, 5-7 of each pixel
        tail = [s for s in c.steps if "@out" in s[2]]
        self.assertEqual([(int(s[2][9]), int(s[2][10]), int(s[2][12])) for s in tail], [(0, 4, 0), (4, 1, 4), (5, 3, 5)])
        for kernel, grid, args in c.steps:     # enough workgroups for every output element
            if kernel == "fg_net2_conv":
                co_n, H, W, pool = int(args[10]), int(args[13]), int(args[14]), int(args[15])
                n = (H // 2 if pool else H) * (W // 2 if pool else W) * co_n
            elif kernel == "fg_net2_up2":
                n = 4 * int(args[2]) * int(args[3]) * int(args[4])
            elif kernel == "fg_net2_merge":
                n = 10 * int(args[4])
            else:
                n = 18 * int(args[7]) * int(args[8])
            self.assertEqual(grid, -(-n // 256), kernel)

    def test_mutants(self):
        for m, (target, before) in K.MUTANT_TARGETS.items():
            c = K.chain_case("m " + m, 48, 32, self.p, self.ib, self.ic, self.ia, np.float32(0.5), mutant=m)
            self.assertIn(target, c.outs)
            for b in before or []:
                self.assertIn(b, c.outs)
            diff = [j for j, (s0, s1) in enumerate(zip(self.case.steps, c.steps)) if s0 != s1]
            self.assertEqual(len(diff), 1, m)

    def test_kernel_cases(self):
        rng = np.random.default_rng(3)
        cases = K.conv_cases(rng) + [K.probe_case(rng)] + K.up_cases(rng) + K.merge_cases(rng) + K.warp_cases(rng)
        kinds = {c.kernel for c in cases}
        self.assertEqual(kinds, set(K.KERNELS))
        for c in cases:
            self.check_steps(c)
            for i, n in c.outs.items():
                self.assertEqual(n, 2 * np.asarray(c.expect[i]).size, (c.name, i))
            ids = [i.lower() for i in list(c.outs) + list(c.data)]
            self.assertEqual(len(ids), len(set(ids)), c.name)
        probe = [c for c in cases if c.alt]
        self.assertEqual(len(probe), 1)
        n = sum(int(np.sum(np.asarray(probe[0].expect[i], np.float16).view(np.uint16)
                           != np.asarray(probe[0].alt[i], np.float16).view(np.uint16))) for i in probe[0].alt)
        self.assertGreater(n, 0, "the probe tells one rounding from two")


if __name__ == "__main__":
    unittest.main()
