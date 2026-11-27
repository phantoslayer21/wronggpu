# SPDX-License-Identifier: GPL-3.0-only
"""Frame generation (#96): the configuration reference (clean/fg/reference/config.py).

    python -m unittest clean.fg.tests.test_fg_config
"""
import struct
import unittest
from fractions import Fraction

import numpy as np

from clean.fg.reference import config as C


def rn32(q):
    """A rational rounded to the nearest binary32 (ties to even), written separately: via struct of a float64 that is
    itself correctly rounded from the exact quotient (innocuous double rounding: 53 >= 2 * 24 + 2)."""
    return struct.unpack("<f", struct.pack("<f", q.numerator / q.denominator))[0]


class MultiFrame(unittest.TestCase):
    def test_times(self):
        self.assertEqual(C.intermediate_time(1, 1), np.float32(0.5))
        self.assertEqual([float(C.intermediate_time(3, i)) for i in (1, 2, 3)], [0.25, 0.5, 0.75])
        t = C.intermediate_time(2, 1)
        self.assertEqual(t.dtype, np.float32)
        self.assertEqual(float(t), rn32(Fraction(1, 3)))
        self.assertNotEqual(float(t), 1.0 / 3.0)   # single, not double precision

    def test_times_against_exact_rounding(self):
        for n in range(1, 64):
            for i in range(1, n + 2):
                self.assertEqual(float(C.intermediate_time(n, i)), rn32(Fraction(i, n + 1)), (n, i))

    def test_flags(self):
        self.assertFalse(C.multi_frame_active(1))
        self.assertTrue(C.multi_frame_active(2))
        self.assertTrue(C.last_index(3, 3))
        self.assertFalse(C.last_index(2, 3))

    def test_validation(self):
        C.check_multi_frame(1, 1, n_max=1)
        C.check_multi_frame(3, 3, n_max=4)
        C.check_multi_frame(3, 4, n_max=4, extra_index=True)
        for n, i, n_max in ((0, 1, 4), (5, 1, 4), (3, 0, 4), (3, 4, 4), (1, 1, 0)):
            with self.assertRaises(ValueError, msg=(n, i, n_max)):
                C.check_multi_frame(n, i, n_max=n_max)
        with self.assertRaises(ValueError):
            C.check_multi_frame(3, 5, n_max=4, extra_index=True)
        with self.assertRaises(TypeError):
            C.check_multi_frame(2, 1)            # the cap is OPEN: required
        with self.assertRaises(TypeError):
            C.check_multi_frame(2.0, 1, n_max=4)
        with self.assertRaises(TypeError):
            C.check_multi_frame(True, 1, n_max=4)


class UiRecomposition(unittest.TestCase):
    def test_profile_values(self):
        for app in (False, True):
            self.assertFalse(C.ui_recomposition(app, 1))
            self.assertTrue(C.ui_recomposition(app, 2))
            self.assertFalse(C.ui_recomposition(app, 0xFFFFFE))
            self.assertTrue(C.ui_recomposition(app, 0xFFFFFF))
            for other in (None, 0, 3, 0xFFFFFD, 0x1000000, -1):
                self.assertEqual(C.ui_recomposition(app, other), app, other)

    def test_presets(self):
        self.assertEqual(C.preset(False), "A")
        self.assertEqual(C.preset(True), "B")
        self.assertEqual(C.preset(C.ui_recomposition(False, 0xFFFFFF)), "B")
        with self.assertRaises(TypeError):
            C.ui_recomposition(1)


if __name__ == "__main__":
    unittest.main()
