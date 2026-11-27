# SPDX-License-Identifier: GPL-3.0-only
"""Frame generation (#96): exact float16 chunk accumulation (clean/fg/reference/fp16acc.py) against an exact rational
reference written here (fractions.Fraction, nearest-even rounding to float16 by comparing neighbours).

    python -m unittest clean.fg.tests.test_fg_fp16acc
"""
import unittest
from fractions import Fraction

import numpy as np

from clean.fg.reference import fp16acc as A


def rn16_exact(q):
    """Fraction -> nearest float16 (ties to even), as a Python float; overflow to +-inf at 65520."""
    if q >= 65520:
        return float("inf")
    if q <= -65520:
        return float("-inf")
    c = np.float16(float(q))                           # within one float16 step of the answer
    cands = {float(c), float(np.nextafter(c, np.float16(np.inf))), float(np.nextafter(c, np.float16(-np.inf)))}
    cands = [x for x in cands if np.isfinite(x)]
    best = min(cands, key=lambda x: (abs(Fraction(x) - q),
                                     int(np.float16(x).view(np.uint16)) & 1))   # nearest, then even mantissa
    if best == 0.0 and q < 0:
        return -0.0
    return best


def exact_acc(acc, a, b):
    return rn16_exact(Fraction(float(acc)) + sum(Fraction(float(x)) * Fraction(float(y)) for x, y in zip(a, b)))


def f16_values(rng, n, kind):
    if kind == "small":
        return rng.uniform(-1, 1, n).astype(np.float16)
    if kind == "wide":     # every exponent, both signs
        bits = rng.integers(0, 0x7C00, n, dtype=np.uint16) | (rng.integers(0, 2, n, dtype=np.uint16) << 15)
        return bits.view(np.float16)
    if kind == "subnormal":
        bits = rng.integers(0, 0x400, n, dtype=np.uint16) | (rng.integers(0, 2, n, dtype=np.uint16) << 15)
        return bits.view(np.float16)
    if kind == "large":
        return rng.uniform(-65504, 65504, n).astype(np.float16)
    raise ValueError(kind)


class Exactness(unittest.TestCase):
    def check(self, acc, a, b):
        got = A.acc_dot16(acc.astype(np.float64), a.astype(np.float64), b.astype(np.float64))
        for i in range(len(acc)):
            want = exact_acc(acc[i], a[i], b[i])
            g = float(got[i])
            ok = (g == want and np.signbit(g) == np.signbit(want)) or (np.isnan(g) and np.isnan(want))
            self.assertTrue(ok, "row %d: got %r want %r (acc %r, a %r, b %r)" % (i, g, want, acc[i], a[i], b[i]))

    def test_random_kinds(self):
        rng = np.random.default_rng(1)
        for ka, kb in (("small", "small"), ("wide", "small"), ("wide", "wide"), ("subnormal", "wide"),
                       ("subnormal", "subnormal"), ("large", "small"), ("large", "large")):
            n = 300
            acc = f16_values(rng, n, "wide" if ka == "wide" else "small")
            a = f16_values(rng, n * 16, ka).reshape(n, 16)
            b = f16_values(rng, n * 16, kb).reshape(n, 16)
            with self.subTest(a=ka, b=kb):
                self.check(acc, a, b)

    def test_ties_and_cancellation(self):
        # acc + exact half-ulp sums: ties to even, and a sticky bit far below breaking the tie
        one = np.float16(1.0)
        ulp = float(np.nextafter(one, np.float16(2)) - one)       # 2^-10
        rows_acc, rows_a, rows_b = [], [], []
        for extra in (0.0, 2.0 ** -40, -2.0 ** -40):
            a = np.zeros(16, dtype=np.float16)
            b = np.zeros(16, dtype=np.float16)
            a[0], b[0] = np.float16(ulp / 2), np.float16(1)        # + half an ulp: a tie
            if extra:
                a[1], b[1] = np.float16(np.sign(extra) * 2.0 ** -20), np.float16(2.0 ** -20)
            for start in (1.0, 1.0 + ulp):
                rows_acc.append(np.float16(start))
                rows_a.append(a.copy())
                rows_b.append(b.copy())
        # exact cancellation to zero, and to a tiny negative value (-0 after rounding)
        a = np.zeros(16, dtype=np.float16); b = np.zeros(16, dtype=np.float16)
        a[0], b[0], a[1], b[1] = 3, 5, -3, 5
        rows_acc.append(np.float16(0)); rows_a.append(a.copy()); rows_b.append(b.copy())
        a[2], b[2] = np.float16(-2.0 ** -24), np.float16(2.0 ** -10)
        rows_acc.append(np.float16(0)); rows_a.append(a.copy()); rows_b.append(b.copy())
        self.check(np.array(rows_acc), np.array(rows_a), np.array(rows_b))

    def test_overflow(self):
        a = np.full((3, 16), 255.0, dtype=np.float16)
        b = np.full((3, 16), 255.0, dtype=np.float16)
        b[1] *= -1
        acc = np.array([0, 0, 65504], dtype=np.float16)
        got = A.acc_dot16(acc.astype(np.float64), a.astype(np.float64), b.astype(np.float64))
        self.assertEqual(list(got[:2]), [np.inf, -np.inf])
        # 65504 + 16 * 255^2 overflows; just below the threshold stays finite
        self.assertEqual(got[2], np.inf)
        z = np.zeros((1, 16)); z[0, 0] = 15.0
        self.assertEqual(float(A.acc_dot16(np.array([65504.0]), z, np.ones((1, 16)))[0]), 65504.0)   # 65519 -> 65504
        z[0, 0] = 16.0
        self.assertEqual(float(A.acc_dot16(np.array([65504.0]), z, np.ones((1, 16)))[0]), np.inf)    # 65520 -> inf

    def test_non_finite(self):
        a = np.zeros((4, 16)); b = np.ones((4, 16))
        a[0, 0] = np.inf
        a[1, 0], a[1, 1] = np.inf, -np.inf
        a[2, 0], b[2, 0] = np.inf, 0.0
        a[3, 0] = np.nan
        got = A.acc_dot16(np.zeros(4), a, b)
        self.assertEqual(got[0], np.inf)
        self.assertTrue(np.isnan(got[1]) and np.isnan(got[2]) and np.isnan(got[3]))
        self.assertTrue(np.isnan(A.acc_dot16(np.array([np.nan]), np.zeros((1, 16)), np.zeros((1, 16)))[0]))

    def test_shape_rules(self):
        with self.assertRaises(ValueError):
            A.acc_dot16(np.zeros(2), np.zeros((2, 17)), np.zeros((2, 17)))
        out = A.acc_dot16(np.zeros((3, 1, 1)), np.ones((3, 16, 1, 1)), np.ones((16, 4, 5)), axis=1)
        self.assertEqual(out.shape, (3, 4, 5))
        self.assertTrue(np.all(out == 16.0))


if __name__ == "__main__":
    unittest.main()
