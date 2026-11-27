# SPDX-License-Identifier: GPL-3.0-only
"""Model K host rules in front of the stages, CPU only (no device, no emulator, no network weights):

- clean/k/reference/exposure.py: which exposure value is which (E_cur the game's exposure or 1, E_aux the auto-exposure
  value when it runs, LDR everything 1, e1 = e2 the game's positive finite scale else 1);
- clean/k/runner/k_frame.py reference_frame: the rule is applied, E_prev is the previous frame's E_cur (E_cur on a
  reset), and a reset is the game's flag or the first frame only: a render-size change keeps the history and the
  frame index.

The stages themselves are replaced by recorders; their own tests are the other modules of this folder.

Run: python -m unittest clean.k.tests.test_k_exposure_rule
"""
import math
import unittest
from unittest import mock

import numpy as np

from clean.k.reference import exposure
from clean.k.runner import k_frame


class TestExposureRule(unittest.TestCase):
    def test_scale(self):
        self.assertEqual(exposure.exposure_scale(2.5), 2.5)
        for bad in (None, "x", 0.0, -1.0, math.inf, -math.inf, math.nan):
            self.assertEqual(exposure.exposure_scale(bad), 1.0, bad)

    def test_ldr_is_all_ones(self):
        v = exposure.exposure_values(False, game_exposure=3.0, measured=4.0, scale=5.0)
        self.assertEqual(v, {"E_cur": 1.0, "E_aux": 1.0, "e1": 1.0, "e2": 1.0})

    def test_hdr_cases(self):
        # no texture, no auto-exposure: all 1
        self.assertEqual(exposure.exposure_values(True), {"E_cur": 1.0, "E_aux": 1.0, "e1": 1.0, "e2": 1.0})
        # no texture + auto-exposure: E_cur 1, E_aux the measured value (it acts through lambda only)
        v = exposure.exposure_values(True, measured=0.5)
        self.assertEqual((v["E_cur"], v["E_aux"]), (1.0, 0.5))
        # texture, no auto-exposure: E_aux = E_cur
        v = exposure.exposure_values(True, game_exposure=2.0)
        self.assertEqual((v["E_cur"], v["E_aux"]), (2.0, 2.0))
        # texture + auto-exposure: the measured value never takes E_cur's place
        v = exposure.exposure_values(True, game_exposure=2.0, measured=0.5, scale=3.0)
        self.assertEqual((v["E_cur"], v["E_aux"], v["e1"], v["e2"]), (2.0, 0.5, 3.0, 3.0))

    def test_apply_rule(self):
        given = {"E_cur": 0.75, "E_aux": 1.5, "e1": 1.0, "e2": 1.0}
        self.assertIs(exposure.apply_rule(given, True), given)  # values given directly are left alone
        d = exposure.apply_rule({"game_exposure": 2.0, "measured": 0.25, "exposure_scale": 4.0}, True)
        self.assertEqual((d["E_cur"], d["E_aux"], d["e1"], d["e2"]), (2.0, 0.25, 4.0, 4.0))
        self.assertEqual(exposure.apply_rule({}, True)["E_cur"], 1.0)
        self.assertEqual(exposure.apply_rule({"game_exposure": 9.0}, False)["E_cur"], 1.0)


class TestReferenceFrameRules(unittest.TestCase):
    """reference_frame with recorder stages."""

    W, H, BASE = 40, 24, (32, 16)

    def run_sequence(self, hdr, frames):
        state = k_frame.new_reference_state((self.W, self.H), self.BASE)
        seen = []

        def fake_in(d):
            seen.append(dict(d))
            return np.zeros((self.BASE[1], self.BASE[0], 16), np.float16)

        def fake_net(x0, params, f):
            seen[-1]["f"] = f
            return {"d0": {"head": np.zeros((self.BASE[1], self.BASE[0], 40), np.float16)}}

        def fake_out(d):
            return {"out": np.zeros((self.H, self.W, 4), np.float32), "L": d["L"], "Hc": d["Hc"],
                    "Hh": np.zeros((self.BASE[1], self.BASE[0], 4), np.float16)}

        results = []
        with mock.patch.object(k_frame.instage, "input_stage", fake_in), \
                mock.patch.object(k_frame.network, "run_network", fake_net), \
                mock.patch.object(k_frame.outstage, "output_stage", fake_out):
            for x in frames:
                results.append(k_frame.reference_frame(state, x, None, {"hdr": hdr}))
        return results, seen

    def frame(self, size, reset=False, game=None, measured=None):
        return {"render_size": size, "reset": reset, "game_exposure": game, "measured": measured}

    def test_reset_is_the_flag_or_the_first_frame_only(self):
        frames = [self.frame((27, 16)), self.frame((27, 16)), self.frame((30, 20)), self.frame((30, 20), reset=True),
                  self.frame((30, 20))]
        results, _ = self.run_sequence(True, frames)
        self.assertEqual([r["reset"] for r in results], [True, False, False, True, False])
        # the size change of frame 2 keeps the frame index running
        self.assertEqual([r["f"] for r in results], [0, 1, 2, 0, 1])

    def test_e_prev_is_the_previous_e_cur(self):
        frames = [self.frame((27, 16), game=0.75), self.frame((27, 16), game=1.25, measured=4.0),
                  self.frame((30, 20), game=None), self.frame((30, 20), game=2.0, reset=True)]
        _, seen = self.run_sequence(True, frames)
        self.assertEqual([s["E_cur"] for s in seen], [0.75, 1.25, 1.0, 2.0])
        self.assertEqual([s["E_aux"] for s in seen], [0.75, 4.0, 1.0, 2.0])
        # first frame and reset frame: E_prev = E_cur; otherwise the previous frame's E_cur (not E_aux)
        self.assertEqual([s["E_prev"] for s in seen], [0.75, 0.75, 1.25, 2.0])

    def test_ldr_values_are_one(self):
        frames = [self.frame((27, 16), game=3.0, measured=2.0), self.frame((27, 16), game=5.0)]
        _, seen = self.run_sequence(False, frames)
        for s in seen:
            self.assertEqual((s["E_cur"], s["E_aux"], s["E_prev"], s["e1"], s["e2"]), (1.0,) * 5)


if __name__ == "__main__":
    unittest.main()
