# SPDX-License-Identifier: MIT
"""Whole stage calls of spec 51 (fronts, block, merge, hidden-state rules, borders) against the slow per-token
implementation, for every stage kind, on small tensors (one window, borders reflected, also repeated reflection)."""
import os
import unittest

import numpy as np

from clean.rr_f.reference.fmt import d8
from clean.rr_f.reference.stages import STAGES, run_stage, stage_info
from clean.rr_f.tests import slow_stage, synth

# (Nx, Ny, shift row): small extents; level-size rules are not needed by the logic under test
CASES = [((6, 4), (2, 4)), ((2, 2), (6, 6))]
FAST = os.environ.get("RR_SLOW_ALL", "1") != "1"


def _eq(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return (a == b) | (np.isnan(a) & np.isnan(b))


class TestStagesVsSlow(unittest.TestCase):
    def check(self, name, ext, shift, seed, hot=False):
        rng = np.random.default_rng(seed)
        wc, _ = synth.window_consts(rng, "zovf" if hot else "AB"[seed % 2])
        W = synth.stage_weights(rng, name, hot=hot)
        consts = synth.stage_consts(rng, hot=hot)
        Nx, Ny = ext
        if name == "E5":
            Nx, Ny = Nx - 1, Ny - 1 if Ny > 1 else 1     # odd extents where no merge is needed
        inputs = synth.stage_inputs(rng, name, Nx, Ny, hot=hot)
        out, aux = run_stage(name, inputs, W, consts, wc, shift)
        ref = slow_stage.run_stage(name, inputs, W, consts, wc, shift)
        info = stage_info(name)
        for (Y, X), vals in ref["y2"].items():
            self.assertTrue(_eq(d8(out["y2"][Y, X]), vals).all(), (name, "y2", Y, X))
        self.assertEqual(len(ref["y2"]), Nx * Ny)
        for (j, i), Z in ref["Z"].items():
            w = j * aux["grid"][0] + i
            self.assertTrue(_eq(aux["Z"][w], Z).all(), (name, "Z", j, i))
        if info["Cn"]:
            self.assertEqual(len(ref["merged"]), (Nx // 2) * (Ny // 2))
            for (Y, X), vals in ref["merged"].items():
                self.assertTrue(_eq(d8(out["merged"][Y, X]), vals).all(), (name, "merged", Y, X))
        if info["hidden"]:
            for (Y, X), vals in ref["hnew"].items():
                self.assertTrue(_eq(out["hnew"][Y, X], vals).all(), (name, "hnew", Y, X, out["hnew"][Y, X], vals))
        if info["merged_hidden"]:
            for (Y, X), vals in ref["hmerged"].items():
                self.assertTrue(_eq(out["hmerged"][Y, X], vals).all(), (name, "hmerged", Y, X))


def _make(name, k, ext, shift, hot=False):
    def t(self):
        self.check(name, ext, shift, seed=100 + 7 * k + list(STAGES).index(name) + (1000 if hot else 0), hot=hot)
    return t


for _name in ("IN0", "E1", "D0-F", "D1-R"):     # large weights and inputs, Z overflow constants: inf / NaN paths
    setattr(TestStagesVsSlow, "test_hot_%s" % _name.replace("-", "_"), _make(_name, 0, (6, 4), (4, 2), hot=True))


for _name in STAGES:
    for _k, (_ext, _shift) in enumerate(CASES):
        if FAST and _k == 1 and stage_info(_name)["C"] > 64:
            continue
        setattr(TestStagesVsSlow, "test_%s_%d" % (_name.replace("-", "_"), _k), _make(_name, _k, _ext, _shift))


class TestStageRules(unittest.TestCase):
    def test_recurrent_shift(self):
        from clean.rr_f.reference.stages import effective_shift
        self.assertEqual(effective_shift("D2-R", (0, 6)), (4, 2))
        self.assertEqual(effective_shift("D2-F", (0, 6)), (0, 6))
        self.assertEqual(effective_shift("E3", (2, 4)), (2, 4))

    def test_fusion_reads_are_per_read(self):
        """51.6: at X = -1 the skip read uses fine token 1 but the up-projection read uses coarse token
        reflect(-1, M) = 1 with child dx = 1 -- not the parts of fine token 1 (coarse 0, dx = 1)."""
        from clean.rr_f.reference.block import reflect
        X = -1
        self.assertEqual(int(reflect(np.array([X]), 8)[0]), 1)
        self.assertEqual(int(reflect(np.array([X // 2]), 4)[0]), 1)
        self.assertEqual(X % 2, 1)


if __name__ == "__main__":
    unittest.main()
