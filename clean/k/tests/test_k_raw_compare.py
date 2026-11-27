# SPDX-License-Identifier: GPL-3.0-only
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

    def odd_bytes(self, n, seed=7):
        """Ordinary bytes of length n (no sentinel byte, so every byte is data)."""
        rng = np.random.default_rng(seed)
        b = rng.integers(0, 256, n, dtype=np.uint8)
        b[(b == HIP_SENTINEL) | (b == D3D12_SENTINEL)] = 1
        return b

    def test_odd_length_equal_passes(self):
        """A buffer whose size is not a multiple of 4 compares byte for byte: 2 x 131 x 79 = 20698 bytes (ctapT at the
        odd render size) and the other trailing-byte lengths 1, 2 and 3 mod 4."""
        for n in (1, 2, 3, 6, 4 * 5 + 1, 4 * 5 + 3, 2 * 131 * 79):
            with self.subTest(n=n):
                b = self.odd_bytes(n)
                self.assertEqual(compare_raw("odd", b.tobytes(), b.copy().tobytes()), 0)

    def test_odd_length_last_byte_differs_fails(self):
        """A change in the last byte of an odd-length buffer (the trailing bytes are compared, not skipped)."""
        for n in (4 * 5 + 1, 4 * 5 + 2, 4 * 5 + 3, 2 * 131 * 79):
            with self.subTest(n=n):
                h = self.odd_bytes(n)
                d = h.copy()
                d[-1] ^= 1
                with self.assertRaisesRegex(AssertionError, f"differ; first at byte {n - 1}: HIP"):
                    compare_raw("odd-last", h.tobytes(), d.tobytes())

    def test_odd_length_first_trailing_byte_differs_fails(self):
        """The first trailing byte (the first byte after the last whole word) is compared too."""
        n = 2 * 131 * 79
        h = self.odd_bytes(n)
        d = h.copy()
        d[(n // 4) * 4] ^= 0x80
        with self.assertRaisesRegex(AssertionError, f"first at byte {(n // 4) * 4}"):
            compare_raw("odd-tail0", h.tobytes(), d.tobytes())

    def test_odd_length_untouched_trailing_byte_counted(self):
        """A trailing byte neither side wrote (HIP's sentinel on one side, D3D12's on the other) is counted like a word;
        a full buffer allows none."""
        n = 2 * 131 * 79
        h = self.odd_bytes(n)
        d = h.copy()
        h[-1], d[-1] = HIP_SENTINEL, D3D12_SENTINEL
        self.assertEqual(compare_raw("odd-pad", h.tobytes(), d.tobytes(), allow_untouched=1), 1)
        with self.assertRaisesRegex(AssertionError, "neither"):
            compare_raw("odd-pad", h.tobytes(), d.tobytes())

    def test_odd_length_write_only_on_d3d12_fails(self):
        """A trailing byte D3D12 wrote where HIP left its sentinel fails (the sentinel pair is not the only way to pass)."""
        n = 2 * 131 * 79
        h = self.odd_bytes(n)
        d = h.copy()
        h[-2] = HIP_SENTINEL
        d[-2] = 9
        with self.assertRaisesRegex(AssertionError, "differ"):
            compare_raw("odd-extra", h.tobytes(), d.tobytes())

    def test_odd_length_size_mismatch_fails(self):
        n = 2 * 131 * 79
        h = self.odd_bytes(n)
        with self.assertRaisesRegex(AssertionError, "sizes"):
            compare_raw("odd-size", h.tobytes(), h[:-1].tobytes())


if __name__ == "__main__":
    unittest.main()
