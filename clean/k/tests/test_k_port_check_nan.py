# SPDX-License-Identifier: MIT
"""The port check's NaN rule (clean/k/dx12/k_port_check.py nan_equivalent_differ): the only accepted difference between
the D3D12 port and HIP is the SIGN of a NaN; the exponent, the quiet bit and the payload must be equal, and every other
differing element counts as a difference. CPU only.

Run: python -m unittest clean.k.tests.test_k_port_check_nan
"""
import unittest

import numpy as np

from clean.k.dx12.k_port_check import nan_equivalent_differ as differ


def run(got, want, dtype, width):
    g, w = np.array(got, dtype).tobytes(), np.array(want, dtype).tobytes()
    return differ(g, w, len(g), width)


class TestNanRule(unittest.TestCase):
    def test_identical(self):
        for v in (0x7E00, 0x1234, 0xFE00):
            self.assertEqual(run([v], [v], np.uint16, 2), (0, 0))

    def test_sign_only_fp16(self):
        self.assertEqual(run([0x7E00, 0xFE01], [0xFE00, 0x7E01], np.uint16, 2), (0, 2))

    def test_sign_only_fp32(self):
        self.assertEqual(run([0x7FC00000, 0xFFC12345], [0xFFC00000, 0x7FC12345], np.uint32, 4), (0, 2))

    def test_payload_differs_is_outside(self):
        self.assertEqual(run([0x7E01], [0x7E02], np.uint16, 2), (1, 0))
        self.assertEqual(run([0x7E01], [0xFE02], np.uint16, 2), (1, 0))   # the sign and the payload differ
        self.assertEqual(run([0x7FC00001], [0x7FC00002], np.uint32, 4), (1, 0))

    def test_quiet_bit_differs_is_outside(self):
        self.assertEqual(run([0x7E00], [0x7C01], np.uint16, 2), (1, 0))   # quiet and signalling NaN
        self.assertEqual(run([0x7FC00000], [0x7F800001], np.uint32, 4), (1, 0))

    def test_nan_against_number_is_outside(self):
        self.assertEqual(run([0x7E00], [0x3C00], np.uint16, 2), (1, 0))
        self.assertEqual(run([0x3C00], [0x7E00], np.uint16, 2), (1, 0))

    def test_negative_zero_and_numbers_are_outside(self):
        self.assertEqual(run([0x0000], [0x8000], np.uint16, 2), (1, 0))   # +0 against -0 is not a NaN sign
        self.assertEqual(run([0x1234], [0x1235], np.uint16, 2), (1, 0))


if __name__ == "__main__":
    unittest.main()
