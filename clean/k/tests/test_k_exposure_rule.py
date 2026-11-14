# SPDX-License-Identifier: MIT
"""Model K's exposure rule, CPU only (no device, no emulator, no weights): clean/k/reference/exposure.py, which exposure
value is which. HDR: E_cur is the game's exposure or 1, E_aux the auto-exposure value when it runs (else E_cur), the
measured value never takes E_cur's place; LDR: every value and factor 1; e1 = e2 is the game's positive finite scale,
else 1.

Run: python -m unittest clean.k.tests.test_k_exposure_rule
"""
import math
import unittest

from clean.k.reference import exposure


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
        # the input is not modified
        x = {"game_exposure": 2.0}
        exposure.apply_rule(x, True)
        self.assertEqual(x, {"game_exposure": 2.0})


if __name__ == "__main__":
    unittest.main()
