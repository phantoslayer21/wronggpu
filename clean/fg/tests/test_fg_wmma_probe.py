# SPDX-License-Identifier: GPL-3.0-only
"""The matrix probe's CPU parts (clean/fg/runner/fg_wmma_probe.py): the float16-midpoint test, the RN32 oracle and the
case classes."""
import os
import sys
import unittest
from fractions import Fraction

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "clean", "fg", "runner"))

import fg_wmma_probe as P  # noqa: E402
from clean.fg.reference import fp16acc as FA  # noqa: E402

F16_VALUES = np.arange(0, 0x7C00, dtype=np.uint16).view(np.float16).astype(np.float64)   # every finite value >= 0


def is_midpoint(v):
    """Brute force: v (float64, exact float32 value) lies exactly halfway between two adjacent float16 values (65520,
    halfway between 65504 and the next step, included)."""
    v = abs(v)
    if v == 0 or not np.isfinite(v) or v >= 65536:
        return False
    i = np.searchsorted(F16_VALUES, v)
    if i < len(F16_VALUES) and F16_VALUES[i] == v:
        return False
    lo = F16_VALUES[i - 1]
    hi = F16_VALUES[i] if i < len(F16_VALUES) else 65536.0
    return v - lo == hi - v


class MidpointTest(unittest.TestCase):
    def test_against_brute_force(self):
        rng = np.random.default_rng(5)
        vals = list(rng.integers(0, 0x7F800000, 20000, dtype=np.uint32))
        for h in rng.integers(0, 0x7BFF, 3000):                     # near float16 values and their midpoints
            f = float(np.uint16(h).view(np.float16))
            g = float(np.uint16(h + 1).view(np.float16))
            for x in (f, g, (f + g) / 2, np.nextafter(np.float32((f + g) / 2), np.float32(0)),
                      np.nextafter(np.float32((f + g) / 2), np.float32(1e9))):
                vals.append(int(np.array([x], np.float32).view(np.uint32)[0]))
        vals += [int(np.array([x], np.float32).view(np.uint32)[0]) for x in (65520.0, 65504.0, 2.0 ** -25, 3 * 2.0 ** -25,
                                                                              2.0 ** -24, 2.0 ** -14 - 2.0 ** -25)]
        for b in vals:
            for s in (0, 0x80000000):
                x = int(b) | s
                v = float(np.array([x], np.uint32).view(np.float32)[0])
                self.assertEqual(P.mid16_py(x), int(is_midpoint(v)), hex(x))


class Rn32Test(unittest.TestCase):
    def test_against_fractions(self):
        rng = np.random.default_rng(6)
        for _ in range(300):
            a = P.rbits(rng, (16,), 20).view(np.uint16)
            b = P.rbits(rng, (16,), 20).view(np.uint16)
            c = int(P.rbits(rng, (1,), 20).view(np.uint16)[0])
            exact = Fraction(float(np.uint16(c).view(np.float16)))
            for x, y in zip(a, b):
                exact += Fraction(float(x.view(np.float16))) * Fraction(float(y.view(np.float16)))
            want = np.float32(0) if exact == 0 else None
            got = np.array([P.rn32_py(c, [int(x) for x in a], [int(y) for y in b])], np.uint32).view(np.float32)[0]
            if want is not None:
                self.assertEqual(float(got), 0.0)
                continue
            # the float32 nearest to exact: no other float32 is closer, and ties go to the even one
            g = Fraction(float(got))
            up, dn = Fraction(float(np.nextafter(got, np.float32(np.inf)))), Fraction(float(np.nextafter(got, np.float32(-np.inf))))
            self.assertLessEqual(abs(g - exact), abs(up - exact))
            self.assertLessEqual(abs(g - exact), abs(dn - exact))


class ClassesTest(unittest.TestCase):
    def test_shapes_and_exact_constructions(self):
        rng = np.random.default_rng(7)
        for name, gen, _ in P.SINGLE:
            A, B, C = gen(rng, 4)
            self.assertEqual((A.shape, B.shape, C.shape), ((4, 1, 16, 16), (4, 1, 16, 16), (4, 16, 16)), name)
            self.assertEqual((A.dtype, B.dtype, C.dtype), (np.float16,) * 3, name)
        A, B, C = P.c_cancel_acc(rng, 8)                 # C + (-1) * V is exactly zero
        np.testing.assert_array_equal(C, B[:, 0, 0, :][:, None, :].repeat(16, 1))
        A, B, C = P.c_signed_zero(rng, 8)                # every sum is an exact zero
        acc = FA.acc_dot16(C.astype(np.float64), A.astype(np.float64)[:, 0, :, None, :],
                           B.astype(np.float64)[:, 0].transpose(0, 2, 1)[:, None, :, :], axis=-1)
        self.assertTrue(np.all(acc == 0))

    def test_chain_classes_cover_every_layer(self):
        cls = P.chain_classes(0.25)
        self.assertEqual([c[0] for c in cls], ["chain_" + l.name for l in P.LY.NET1])
        self.assertEqual(dict((c[0], c[4]) for c in cls)["chain_net1_05"], 72)
        A, B, C = cls[0][1](np.random.default_rng(1), 2)
        self.assertEqual(A.shape, (2, 27, 16, 16))
        self.assertTrue(np.all(B[:, 18:, 8:, :] == 0))


if __name__ == "__main__":
    unittest.main()
