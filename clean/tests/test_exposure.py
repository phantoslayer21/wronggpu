# SPDX-License-Identifier: MIT
"""Stage exposure (spec 07): reference vs a slow scalar version and the spec's examples; the real kernels
(kernels/exposure.hip) in the host lane emulator vs the reference (07-T1 cases: black, uniform, HDR, NaN/Inf,
negative values, odd sizes). Synthetic data only.

Run: python -m unittest discover -s clean/tests -t .
"""
import math
import unittest
from fractions import Fraction

import numpy as np

from clean.reference import exposure as X
from clean.tests.test_kernels_emu import EmuTest


def slow_exposure(color, pi=1.0, alpha=0.18):
    rh, rw = len(color), len(color[0])
    nx, ny = max(256, rw // 2), max(256, rh // 2)
    total = 0.0
    for j in range(ny):
        Y = math.floor(Fraction(j * (rh - 1), ny - 1) + Fraction(1, 2))
        for i in range(nx):
            Xi = math.floor(Fraction(i * (rw - 1), nx - 1) + Fraction(1, 2))
            r, g, b = (float(v) for v in color[Y][Xi])
            with np.errstate(all="ignore"):
                L = (0.25 * r + 0.5 * g + 0.25 * b) / pi
            total += math.log(L + 1e-6) if math.isfinite(L) and L > 0 else 0.0
    m = total / (nx * ny)
    return float(np.float16(alpha / (1 - alpha) * math.exp(-m)))


def fp16_steps(a, b):
    return abs(int(np.float16(a).view(np.int16)) - int(np.float16(b).view(np.int16)))


class ExposureReferenceTest(unittest.TestCase):
    def test_spec_examples(self):
        for v, want in ((0.0, 0.2195), (0.25, 0.878), (0.05, 4.39)):
            e = X.exposure(np.full((20, 30, 3), v, np.float32))
            self.assertAlmostEqual(e, want, delta=0.002 * want)
        self.assertEqual(X.exposure(np.full((9, 7, 3), np.nan, np.float32)), float(np.float16(0.18 / 0.82)))
        self.assertEqual(X.exposure(np.full((9, 7, 3), 1e-9, np.float32)), 65504.0)   # saturated, not +inf

    def test_sample_grid(self):
        for rw in (1, 2, 255, 256, 257, 511, 512, 513, 1706):
            nx, _, Xs, _ = X.sample_grid(rw, 4)
            self.assertEqual(nx, max(256, rw // 2))
            want = [math.floor(Fraction(i * (rw - 1), nx - 1) + Fraction(1, 2)) for i in range(nx)]
            self.assertEqual(Xs.tolist(), want)
            self.assertEqual((Xs[0], Xs[-1]), (0, rw - 1))

    def test_matches_slow_version(self):
        rng = np.random.default_rng(4)
        for rw, rh, pi, al in ((13, 11, 1.0, 0.18), (300, 9, 2.0, 0.25)):
            c = X.random_color(rng, rw, rh)
            self.assertEqual(fp16_steps(X.exposure(c, pi, al), slow_exposure(c, pi, al)), 0)


class ExposureEmuTest(EmuTest):
    def test_matches_reference(self):
        rng = np.random.default_rng(8)
        cases = [((40, 24), 1.0, 0.18, "hdr"), ((601, 523), 1.0, 0.18, "hdr"), ((1030, 517), 2.5, 0.3, "hdr"),
                 ((7, 5), 1.0, 0.18, "hdr"), ((1, 1), 1.0, 0.18, "hdr"), ((64, 48), 1.0, 0.18, "black"),
                 ((64, 48), 1.0, 0.18, "nan"), ((64, 48), 0.5, 0.18, "uniform"), ((64, 48), 1.0, 0.18, "dark"),
                 ((64, 48), 1.0, 0.18, "nearly_black")]
        for (rw, rh), pi, al, kind in cases:
            with self.subTest(size=(rw, rh), pi=pi, alpha=al, kind=kind):
                if kind == "hdr":
                    c = X.random_color(rng, rw, rh)
                elif kind == "black":
                    c = np.zeros((rh, rw, 3), np.float32)
                elif kind == "nan":
                    c = np.full((rh, rw, 3), np.nan, np.float32)
                elif kind == "uniform":
                    c = np.full((rh, rw, 3), 0.25, np.float32)
                elif kind == "dark":
                    c = np.full((rh, rw, 3), 1e-3, np.float32)
                else:  # m = ln(2e-8 + 1e-6) < -13.7: K exp(-m) > 65504 -> saturated
                    c = np.full((rh, rw, 3), 2e-8, np.float32)
                x = {"color": c, "pre_exposure": pi, "alpha": al}
                got = self.runner.run("exposure", b"", x)
                self.assertEqual(set(got), {"e"})
                ref = X.exposure(c, pi, al)
                self.assertTrue(math.isfinite(got["e"]))
                self.assertLessEqual(fp16_steps(got["e"], ref), 1, (got["e"], ref))

    def test_pack_weights_is_empty(self):
        self.assertEqual(self.runner.pack_weights("exposure", {}), b"")


if __name__ == "__main__":
    unittest.main()
