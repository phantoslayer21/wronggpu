# SPDX-License-Identifier: MIT
"""The integer form of the dithered FP8 quantization used by kernels/encoder.hip (dither_q) against the step-by-step
definition of spec 03 draft 2 (reference/encoder.py, dither_fp8).

dither_q works on the FP32 bit pattern b of the value v, clamped to +-447.75:
    tt = 0xfff + ((t + 1) << 13);   d = (b + bit13(b) + tt) & 0xfff00000;   code = E4M3(d) (nearest even)
The test covers every finite FP16 value together with its FP32 neighbours and the FP16 rounding midpoints next to it
(the inputs where RN16 matters), values beyond the clamp, and every dither value t = 0..127. (A C version of this
check over all FP32 values in [-2048, 2048] for eight dither values, 6.7e9 inputs, found no difference either.)

Run: python -m unittest clean.tests.test_encoder_dither
"""
import unittest

import numpy as np

from clean.reference.encoder import dither_fp8
from clean.reference.fp8 import q8


def dither_q(v, t):
    """NumPy model of the kernel's dither_q (FP32 bit arithmetic) followed by the E4M3 conversion."""
    c = np.clip(np.asarray(v, np.float32), np.float32(-447.75), np.float32(447.75))
    b = c.view(np.uint32).astype(np.uint64)
    tt = np.uint64(0xFFF) + (np.asarray(t, np.uint64) + np.uint64(1) << np.uint64(13))
    d = ((b + ((b >> np.uint64(13)) & np.uint64(1)) + tt) & np.uint64(0xFFF00000)).astype(np.uint32)
    return q8(d.view(np.float32).astype(np.float64))


def inputs():
    h = np.arange(65536, dtype=np.uint32).astype(np.uint16).view(np.float16).astype(np.float32)
    h = h[np.isfinite(h)]
    b = h.view(np.uint32).astype(np.int64)
    parts = [b + d for d in (-2, -1, 0, 1, 2)] + [b + 0x1000 + d for d in (-1, 0, 1)]   # midpoints next to h
    v = np.concatenate(parts).astype(np.uint32).view(np.float32)
    v = v[np.isfinite(v)]
    extra = np.array([447.75, 447.8, 448.0, 449.0, 500.0, 1023.9, 1024.0, 5000.0, 65504.0, 1e30], np.float32)
    return np.concatenate([v, extra, -extra])


class TestDitherIntegerForm(unittest.TestCase):
    def test_every_fp16_neighbourhood_and_dither(self):
        v = inputs()
        for t in range(128):
            ref = dither_fp8(np.minimum(v.astype(np.float64), 1024.0), t)   # the spec's cap at 1024
            got = dither_q(v, t)
            bad = ~((ref == got) | (np.isnan(ref) & np.isnan(got)))
            self.assertEqual(int(bad.sum()), 0, f"t={t}: first differences at {v[bad][:5]}")


if __name__ == "__main__":
    unittest.main()
