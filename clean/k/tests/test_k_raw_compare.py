# SPDX-License-Identifier: MIT
"""CPU test of the raw-byte comparison with distinct sentinels (clean/k/tests/raw_compare.py): it must pass equal data
and fail a dropped write, a write only D3D12 made, a changed bit, a NaN payload and a negative zero."""
import unittest

import numpy as np

from clean.k.tests.raw_compare import D3D12_SENTINEL, HIP_SENTINEL, compare_raw, sentinel_word


class TestRawCompare(unittest.TestCase):
    def data(self):
        rng = np.random.default_rng(5)
        h = rng.integers(0, 2**32, 64, dtype=np.uint64).astype(np.uint32)
        h[(h == sentinel_word(HIP_SENTINEL)) | (h == sentinel_word(D3D12_SENTINEL))] = 1
        return h, h.copy()

    def test_equal(self):
        h, d = self.data()
        self.assertEqual(compare_raw("equal", h.tobytes(), d.tobytes()), 0)

    def test_dropped_write_fails(self):
        """D3D12 left its sentinel where HIP stored a value."""
        h, d = self.data()
        d[7] = sentinel_word(D3D12_SENTINEL)
        with self.assertRaisesRegex(AssertionError, "differ"):
            compare_raw("dropped", h.tobytes(), d.tobytes())

    def test_write_only_on_d3d12_fails(self):
        h, d = self.data()
        h[9] = sentinel_word(HIP_SENTINEL)
        with self.assertRaisesRegex(AssertionError, "differ"):
            compare_raw("extra", h.tobytes(), d.tobytes())

    def test_one_bit_fails(self):
        h, d = self.data()
        d[3] ^= np.uint32(1 << 31)
        with self.assertRaises(AssertionError):
            compare_raw("bit", h.tobytes(), d.tobytes())

    def test_nan_payload_and_negative_zero_fail(self):
        h, d = self.data()
        h[1], d[1] = np.uint32(0x7FC00000), np.uint32(0x7FC00001)
        with self.assertRaises(AssertionError):
            compare_raw("nan", h.tobytes(), d.tobytes())
        h, d = self.data()
        h[2], d[2] = np.uint32(0), np.uint32(0x80000000)
        with self.assertRaises(AssertionError):
            compare_raw("negzero", h.tobytes(), d.tobytes())

    def test_untouched_words_counted_and_limited(self):
        """A word neither side wrote (padding) is counted; a caller that knows the buffer is full allows none."""
        h, d = self.data()
        h[5], d[5] = sentinel_word(HIP_SENTINEL), sentinel_word(D3D12_SENTINEL)
        self.assertEqual(compare_raw("pad", h.tobytes(), d.tobytes(), allow_untouched=1), 1)
        with self.assertRaisesRegex(AssertionError, "neither"):
            compare_raw("pad", h.tobytes(), d.tobytes())

    def test_size_mismatch_fails(self):
        h, d = self.data()
        with self.assertRaises(AssertionError):
            compare_raw("size", h.tobytes(), d[:-1].tobytes())


if __name__ == "__main__":
    unittest.main()
