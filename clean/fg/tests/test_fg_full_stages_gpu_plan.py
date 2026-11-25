# SPDX-License-Identifier: MIT
"""Frame generation (issue #96): the Full path stage check (clean/fg/runner/fg_full_stages_d3d12_check.py) as planned on
the CPU: every launch's arguments against its kernel's declaration, grids against the outputs, the chain lines, the
probe's mutants against its expected buffer, and that the CPU reference does not depend on its row bands. No GPU."""
import os
import re
import unittest

import numpy as np

from clean.fg.reference import stages_full as S
from clean.fg.runner import fg_full_stages_d3d12_check as R

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
SOURCES = ["fg_full_blend.hlsl", "fg_out_pull.hlsl", "fg_out_push.hlsl", "fg_distortion.hlsl"]


def declarations():
    out = {}
    for fn in SOURCES:
        with open(os.path.join(ROOT, "clean", "dx12", "kernels", fn), encoding="utf-8") as f:
            for line in f:
                m = re.match(r"^//\s*@wg-kernel\s+(\S+)(.*)$", line)
                if m:
                    kv = dict(t.split("=", 1) for t in m.group(2).split())
                    # co=- : D3D12-only variants (the executor's packed-slot output kernels), not stage-check kernels
                    if kv.get("co") != "-":
                        out[m.group(1)] = kv
    return out


class TestPlan(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(3)
        cls.cases = R.kernel_cases(rng) + R.chain_cases(rng) + R.mutant_cases(rng) + [R.floor_case()]
        cls.decl = declarations()

    def test_declarations(self):
        self.assertEqual(sorted(self.decl), sorted(R.KERNELS))
        for name, kv in self.decl.items():
            self.assertEqual(kv["co"], R.CO, name)
            self.assertEqual((kv["block"], kv["wave"]), ("256", "32"), name)

    def test_launch_arguments(self):
        seen = set()
        for c in self.cases:
            ids = set(c.data) | set(c.outs)
            for kernel, groups, args in c.steps:
                seen.add(kernel)
                kinds = self.decl[kernel]["args"].split(",")
                self.assertEqual(len(args), len(kinds), (c.name, kernel))
                self.assertGreaterEqual(groups, 1)
                for a, k in zip(args, kinds):
                    if k == "p":
                        self.assertTrue(a == "null" or (a.startswith("@") and a[1:] in ids), (c.name, kernel, a))
                    else:
                        v = int(a)
                        self.assertTrue(-2 ** 31 <= v < 2 ** 32, (c.name, kernel, a))
        self.assertEqual(seen, set(R.KERNELS))

    def test_grids_cover_outputs(self):
        for c in self.cases:
            for kernel, groups, args in c.steps:
                if kernel == "fg_full_blend":
                    Wo, Ho, k, tx = int(args[14]), int(args[15]), int(args[20]), int(args[21])
                    Wp, Hp = S.padded(Wo, Ho, k)
                    self.assertEqual((tx, groups), (Wp // 16, (Wp // 16) * (Hp // 16)), c.name)
                    self.assertEqual(c.outs[args[11][1:]], 8 * (Wp // k) * (Hp // k), c.name)
                    self.assertEqual(c.outs[args[8][1:]], 8 * Wo * Ho, c.name)
                elif kernel == "fg_out_pull":
                    Wo, Ho = int(args[11]), int(args[12])
                    self.assertEqual(groups, R.tiles(Wo) * R.tiles(Ho))
                    for l in range(1, 5):
                        lw, lh = S.level_size(Wo, Ho, l)
                        self.assertEqual(c.outs[args[6 + l][1:]], 8 * lw * lh, (c.name, l))
                elif kernel == "fg_out_push":
                    wf, hf = int(args[3]), int(args[4])
                    self.assertGreaterEqual(groups * 256, wf * hf)
                    self.assertEqual(c.outs[args[2][1:]], 8 * wf * hf, c.name)
                elif kernel in ("fg_dist_down", "fg_dist_fields", "fg_out_push_fine"):
                    n = int(args[2]) * int(args[3]) if kernel == "fg_dist_down" else (
                        int(args[8]) * int(args[9]) if kernel == "fg_dist_fields" else int(args[3]) * int(args[4]))
                    self.assertGreaterEqual(groups * 256, n, c.name)

    def test_expected_buffers(self):
        for c in self.cases:
            for i, n in c.outs.items():
                self.assertEqual(c.expect[i].nbytes, n, (c.name, i))
                if i in c.init:
                    self.assertEqual(len(c.init[i]), n, (c.name, i))

    def test_chain_lines(self):
        c = next(c for c in self.cases if c.name.startswith("pull offset"))
        m = {"id": R._id(c.name), "files": {i: "f." + i for i in c.data}, "sizes": {i: len(b) for i, b in c.data.items()},
             "outs": c.outs, "inits": {i: "init." + i for i in c.init}, "steps": c.steps}
        lines = R.chain_lines(m, 0xA5, "t", half=0)
        self.assertEqual(lines[1], "module fg_full.co")
        out_line = next(ln for ln in lines if ln.startswith("buf out "))
        self.assertEqual(out_line.split()[2:], [str(c.outs["out"] + R.SLACK), "init.out", str(0xA5)])
        step = next(ln for ln in lines if ln.startswith("step "))
        self.assertEqual(int(step.split()[2]), max(1, c.steps[0][1] // 2))

    def test_probe_mutants_differ(self):
        c = R.probe_case()
        exp = c.expect["out"]
        for name, alt in c.alt["out"].items():
            a = np.ascontiguousarray(np.asarray(alt, np.float32).astype("<f2")).view("<u2").reshape(-1)
            self.assertGreater(int(np.sum(a != exp)), 0, name)
        # the fused case: 1 + 2^-10 from one rounding, 1 from two
        self.assertEqual(float(np.frombuffer(exp[:1].tobytes(), "<f2")[0]), 1.0 + 2.0 ** -10)

    def test_presets_differ_in_mutants(self):
        a, b = [c for c in self.cases if c.name.startswith("mutant")]
        self.assertEqual([s[2][-1] for s in a.steps], ["0"])
        self.assertEqual([s[2][-1] for s in b.steps], ["1"])

    def test_bands_do_not_change_results(self):
        rng = np.random.default_rng(9)
        d, ui = R.blend_data(rng, 41, 37, 20, 18, 20, 18, ui=True)
        X0, X1, Xw, N2, _, _ = R.pull_data(rng, 41, 37, 2)
        old = S.BAND
        try:
            S.BAND = 64
            r1 = S.g4_full_blend(**d, k=2, ui=ui)
            o1 = S.g6_output(X0, X1, Xw, N2, k=2)
            S.BAND = 5
            r2 = S.g4_full_blend(**d, k=2, ui=ui)
            o2 = S.g6_output(X0, X1, Xw, N2, k=2)
        finally:
            S.BAND = old
        for key in ("X0", "X1", "Xw", "Ib", "Ic", "Ia"):
            self.assertTrue(np.array_equal(r1[key], r2[key], equal_nan=True), key)
        self.assertTrue(np.array_equal(o1["frame"], o2["frame"], equal_nan=True))


if __name__ == "__main__":
    unittest.main()
