# SPDX-License-Identifier: GPL-3.0-only
"""Model L: the q7 checks' NaN-sign exception (clean/l/runner/l_q7_check.py mask_nan) masks ONLY a NaN's sign bit.

Run: python -m unittest discover -s clean/l/tests -t .
"""
import unittest

import numpy as np

from clean.l.runner.l_q7_check import mask_control, mask_nan


class NanSignMask(unittest.TestCase):
    def test_control_passes(self):
        r = mask_control()
        self.assertTrue(r.pop("pass"))
        self.assertTrue(all(r.values()), r)

    def test_non_nan_bytes_unchanged(self):
        b8 = bytes(c for c in range(256) if (c & 0x7F) != 0x7F)
        self.assertEqual(mask_nan(b8, "fp8"), b8)
        h = np.array([v for v in range(65536) if not ((v & 0x7C00) == 0x7C00 and (v & 0x3FF))], np.uint16).tobytes()
        self.assertEqual(mask_nan(h, "fp16"), h)

    def test_nan_only_sign_cleared(self):
        self.assertEqual(mask_nan(bytes([0xFF, 0x7F]), "fp8"), bytes([0x7F, 0x7F]))
        h = np.array([0xFE01, 0xFC01, 0x7E02], np.uint16).tobytes()
        self.assertEqual(np.frombuffer(mask_nan(h, "fp16"), np.uint16).tolist(), [0x7E01, 0x7C01, 0x7E02])

    def test_mutant_full_mask_is_caught(self):
        """A mask that also cleared non-NaN signs would make the control fail."""
        import clean.l.runner.l_q7_check as Q
        real = Q.mask_nan
        Q.mask_nan = lambda body, kind: (bytes(b & 0x7F for b in body) if kind == "fp8" else
                                         (np.frombuffer(body, np.uint16) & 0x7FFF).astype(np.uint16).tobytes())
        try:
            self.assertFalse(Q.mask_control()["pass"])
        finally:
            Q.mask_nan = real


if __name__ == "__main__":
    unittest.main()
