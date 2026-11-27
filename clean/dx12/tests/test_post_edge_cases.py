# SPDX-License-Identifier: GPL-3.0-only
"""CPU checks of Model L's post history fast path (clean/dx12/kernels/l_post.hlsl history_cr) and of the inputs of
its GPU edge check (clean/l/runner/l_post_edge_check.py):
- the fast path's predicate as written (unsigned tests with the size guard) holds exactly when the 4 x 4 footprint
  lies inside the image, at every size from 1 x 1; without the guard (the old predicate, mutant LP_MUT 9) it also
  holds outside an image narrower or shorter than 4;
- each edge-check kind puts the history positions where it says (inside the footprint, outside it, or both in one
  wave), so the counter expectations of the GPU check follow."""
import os
import sys
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, ROOT)

from clean.l.runner import l_post_edge_check as E  # noqa: E402


def u32(v):
    return np.int64(v) & 0xFFFFFFFF


def predicate(ix, iy, dw, dh, guard=True):
    """l_post.hlsl's interior predicate of one lane (int32 arithmetic, unsigned compares)."""
    p = u32(ix - 1) <= u32(dw - 4) and u32(iy - 1) <= u32(dh - 4)
    return bool(p and (not guard or (dw >= 4 and dh >= 4)))


def inside(ix, iy, dw, dh):
    """Rows iy - 1 ... iy + 2 and columns ix - 1 ... ix + 2 all in the image."""
    return ix - 1 >= 0 and ix + 2 <= dw - 1 and iy - 1 >= 0 and iy + 2 <= dh - 1


class PredicateTest(unittest.TestCase):
    def test_guarded_predicate_is_the_footprint(self):
        for dw in range(1, 10):
            for dh in range(1, 10):
                for ix in range(-3, 12):
                    for iy in range(-3, 12):
                        self.assertEqual(predicate(ix, iy, dw, dh), inside(ix, iy, dw, dh), (dw, dh, ix, iy))

    def test_old_predicate_reads_outside(self):
        bad = [(dw, dh, ix, iy) for dw in range(1, 10) for dh in range(1, 10) for ix in range(-3, 12)
               for iy in range(-3, 12) if predicate(ix, iy, dw, dh, guard=False) and not inside(ix, iy, dw, dh)]
        self.assertIn((5, 3, 2, 1), bad)   # height 3: the span reads row 3
        self.assertTrue(all(dw < 4 or dh < 4 for dw, dh, _, _ in bad))


class EdgeCasesTest(unittest.TestCase):
    def footprints(self, x, dw, dh):
        X, Y = np.meshgrid(np.arange(dw) + 0.5, np.arange(dh) + 0.5)
        qx = X + x["m_near"][..., 0].astype(np.float64)
        qy = Y + x["m_near"][..., 1].astype(np.float64)
        ix, iy = np.floor(qx - 0.5).astype(int), np.floor(qy - 0.5).astype(int)
        return np.vectorize(lambda a, b: inside(a, b, dw, dh))(ix, iy)

    def test_kinds(self):
        rng = np.random.default_rng(1)
        mixed = set()
        for dw, dh in E.SIZES:
            for kind in E.KINDS:
                if kind == "random":
                    continue
                x = E.edge_case(rng, dw, dh, kind)
                for k in ("m_center", "m_far"):
                    self.assertTrue(np.array_equal(x[k], x["m_near"]))
                case = {"x": x, "dw": dw, "dh": dh, "kind": kind}
                want = E.counter_expected(case)
                rows = E.wave_rows(x, dw, dh)
                if dw < 4 or dh < 4:
                    self.assertIs(want, False)
                elif kind in ("center", "in_lo", "in_hi"):
                    self.assertTrue(want, (dw, dh, kind))
                    self.assertTrue(self.footprints(x, dw, dh).all(), (dw, dh, kind))
                elif kind in ("left", "right", "out_lo", "out_hi"):
                    self.assertIs(want, False, (dw, dh, kind))
                    self.assertFalse(self.footprints(x, dw, dh).any(), (dw, dh, kind))
                elif kind in ("split", "split_half"):
                    if any(i > 0 and o > 0 for i, o in rows):
                        mixed.add((dw, dh, kind))
        # the mixed waves (both inside and outside lanes) appear at every size of at least 4 x 4, for both split kinds
        for dw, dh in E.SIZES:
            if dw >= 4 and dh >= 4:
                self.assertIn((dw, dh, "split"), mixed)
                self.assertIn((dw, dh, "split_half"), mixed)


if __name__ == "__main__":
    unittest.main()
