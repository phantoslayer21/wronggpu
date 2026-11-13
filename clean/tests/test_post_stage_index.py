# SPDX-License-Identifier: MIT
"""Exactness checks for the 32-bit / 24-bit index arithmetic of post_recon2_body (kernels/post.hip):

1. Staging row r = (int)((i + 0.5f) * (1.0f / ncol)) equals i / ncol (integer division) for every ncol the kernel can
   see and every staging index i < kR2Cap, even if 1 / ncol comes out up to 2 ulp off (the compiler may use v_rcp_f32,
   1 ulp) and the product up to 1 ulp off. (i + 0.5) / ncol is at least 0.5 / ncol away from an integer, far more than
   these errors.
2. runner/post.py recon2_supported rejects every size where the 32-bit byte offsets (mul24) of the full history, the
   display planes, the reduced history or the fallback colour could reach 2^32, or a mul24 factor 2^24.

NumPy only (no GPU, no emulator). Run: python -m unittest discover -s clean/tests -t .
"""
import unittest

import numpy as np

from clean.runner import post as PS

K_R2_CAP = 336  # kR2Cap


def _ulps(x, n):
    """x and its neighbours up to n ulp away (float32)."""
    out = {np.float32(x)}
    lo = hi = np.float32(x)
    for _ in range(n):
        lo = np.nextafter(lo, np.float32(-np.inf), dtype=np.float32)
        hi = np.nextafter(hi, np.float32(np.inf), dtype=np.float32)
        out.update((lo, hi))
    return sorted(out)


class TestStageRow(unittest.TestCase):
    def test_reciprocal_row(self):
        i = np.arange(K_R2_CAP, dtype=np.int64)
        fi = i.astype(np.float32) + np.float32(0.5)      # exact (i < 2^23)
        for ncol in range(1, 4097):
            want = i // ncol
            exact = np.float32(1.0) / np.float32(ncol)   # IEEE division, correctly rounded
            for inv in _ulps(exact, 2):
                prod = fi * inv                           # float32 multiply, round to nearest even
                for p in (prod, np.nextafter(prod, np.float32(-np.inf), dtype=np.float32),
                          np.nextafter(prod, np.float32(np.inf), dtype=np.float32)):
                    r = p.astype(np.int64)                # v_cvt_i32_f32 truncates; p >= 0
                    bad = np.nonzero(r != want)[0]
                    self.assertEqual(bad.size, 0, f"ncol {ncol} inv {inv!r} i {i[bad[:4]]}")
                    c = i - r * ncol                      # the column, i - mul_i24(r, ncol)
                    self.assertTrue(np.all((c >= 0) & (c < ncol)))


class TestOffsetGuard(unittest.TestCase):
    def test_guard(self):
        two32, two24 = 1 << 32, 1 << 24
        cases = [(1920, 1080, 2880, 1620), (2560, 1440, 3840, 2160), (3840, 2160, 5760, 3240),
                 (7680, 4320, 11520, 6480), (10000, 10000, 15000, 15000), (16384, 16384, 24576, 24576),
                 (8000, 8, 23000, 12), (11000, 11000, 16500, 16500), (40, 40, 60, 60),
                 (1100000, 1, 1650000, 2), (2000, 2000, 1500000, 3)]
        for rw, rh, dw, dh in cases:
            pitch = (dw + 63) & ~63
            fits = (8 * dw * dh <= two32 and 6 * pitch * dh <= two32 and 16 * rw * rh <= two32 and
                    16 * pitch < two24 and 16 * rw < two24)
            with self.subTest(R=(rw, rh), D=(dw, dh)):
                self.assertEqual(PS.offsets32_ok(rw, rh, dw, dh), fits)
                if not fits:
                    self.assertFalse(PS.recon2_supported(rw, rh, dw, dh))
        self.assertTrue(PS.recon2_supported(7680, 4320, 11520, 6480))   # 8K output (DLAA) still takes post_recon2
        self.assertFalse(PS.recon2_supported(16384, 16384, 24576, 24576))


if __name__ == "__main__":
    unittest.main()
