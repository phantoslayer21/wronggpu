# SPDX-License-Identifier: MIT
"""Model M output-filter profiles (clean/reference/output_quality.py): weights and the generated kernel header."""
import os
import unittest

import numpy as np

from clean.reference import output_quality as Q
from clean.reference import post as P

HEADER = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "kernels", "output_quality5.h")


def pair_error(w):
    """Largest deviation from 1/4 D pixel of where an even output pixel samples a sine of the picture, over the
    frequencies below a third of the output's Nyquist frequency (phase delay of the taps)."""
    half = len(w) // 2
    k = np.arange(-half, half + 1)
    f = np.linspace(0.01, 1 / 9, 50)
    h = np.exp(-2j * np.pi * f[:, None] * k[None, :]) @ w
    return float(np.abs(-np.angle(h) / (2 * np.pi * f) - 0.25).max())


class OutputQuality(unittest.TestCase):
    def test_weights(self):
        for name, n in (("quality9", 9), ("quality5", 5)):
            w = Q.PROFILES[name]()
            self.assertEqual(len(w), n)
            self.assertAlmostEqual(float(w.sum()), 1.0, places=12)
        k9 = np.arange(-4, 5)
        self.assertAlmostEqual(float(Q.weights() @ k9), 0.25, places=12)

    def test_phase_closer_than_spec(self):
        _, w = P.filter_weights(np.arange(2))
        spec = pair_error(w[0])
        self.assertGreater(spec, 0.08)                      # spec 06's 5 taps: about 0.087 D pixel off
        self.assertLess(pair_error(Q.weights5()), spec / 3)
        self.assertLess(pair_error(Q.weights()), spec / 20)

    def test_header_is_generated(self):
        with open(HEADER, newline="") as f:
            self.assertEqual(f.read().replace("\r\n", "\n"), Q.header5())

    def test_filter_constant_and_range(self):
        rng = np.random.default_rng(3)
        for name in Q.PROFILES:
            disp = np.full((P.recon_size_for(9), P.recon_size_for(7), 3), 0.75)
            np.testing.assert_allclose(Q.encoded_filter(disp, (7, 9), name), 0.75, rtol=1e-12)
            disp = rng.uniform(0.2, 0.9, (P.recon_size_for(10), P.recon_size_for(12), 3))
            out = Q.encoded_filter(disp, (12, 10), name)
            self.assertGreaterEqual(out.min(), disp.min())  # the local min/max clamp: no overshoot
            self.assertLessEqual(out.max(), disp.max())


if __name__ == "__main__":
    unittest.main()
