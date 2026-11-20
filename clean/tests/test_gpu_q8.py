# SPDX-License-Identifier: MIT
"""The FP32 -> E4M3 conversions of the gfx11 kernels (kernels/wg_platform.h) against a reference written from the
definition of the format (runner/q8_exhaustive.py).

TestReference needs no GPU: the reference itself is checked against reference/fp8.py. TestGpuSweep runs the conversions on
a gfx11 GPU over a set of 2^24-pattern chunks that cover every class of input (zero, FP32 denormals, the E4M3 subnormal
range, every binade up to 448, the saturation edge, infinity, NaN, both signs); it is skipped unless WG_GPU_TEST=1:

    python clean/runner/build.py --arch gfx1100 --rocm <ROCm> --out <build> --no-runtime
    WG_GPU_TEST=1 WG_BUILD=<build> WG_ROCM=<ROCm> python -m unittest clean.tests.test_gpu_q8

All 256 chunks, 2^32 patterns: python clean/runner/q8_exhaustive.py --build <build> --rocm <ROCm>
"""
import os
import unittest

import numpy as np

from clean.reference.fp8 import fp8_encode, q8
from clean.runner import q8_exhaustive as Q


def sample_bits():
    rng = np.random.default_rng(11)
    values = Q.VALUES
    mids = (values[:-1] + values[1:]) / 2                      # the ties between neighbouring E4M3 values
    near = []
    for v in np.concatenate([values, mids, [448.0, 464.0, 480.0, 2.0 ** -10, 2.0 ** -6]]):
        u = int(np.float32(v).view(np.uint32))
        near += [(u + d) & 0xFFFFFFFF for d in range(-2, 3)]
    near = np.array(near, np.uint32)
    special = np.array([0, 0x80000000, 0x7F800000, 0xFF800000, 0x7FC00000, 0xFFC00000, 0x7F800001, 0xFFFFFFFF,
                        0x00000001, 0x007FFFFF, 0x80000001, 0x807FFFFF], np.uint32)
    patterns = np.concatenate([near, near | 0x80000000, special, rng.integers(0, 2 ** 32, 1_000_000, dtype=np.uint32)])
    return patterns


class TestReference(unittest.TestCase):
    def test_codes_equal_fp8_py(self):
        bits = sample_bits()
        with np.errstate(all="ignore"):
            x = bits.view(np.float32).astype(np.float64)
        nan = np.isnan(x)
        got = Q.reference_codes(bits)
        want = fp8_encode(x)
        np.testing.assert_array_equal(got[~nan], want[~nan])
        np.testing.assert_array_equal(got[nan] & 0x7F, np.full(int(nan.sum()), 0x7F))
        np.testing.assert_array_equal(got[nan] >> 7, (bits[nan] >> 31).astype(np.uint8))

    def test_values_equal_q8(self):
        bits = sample_bits()
        with np.errstate(all="ignore"):
            x = bits.view(np.float32).astype(np.float64)
        finite = np.isfinite(x)
        decoded = np.where(Q.reference_codes(bits) & 0x80, -1.0, 1.0) * np.concatenate([Q.VALUES, [np.nan]])[
            Q.reference_codes(bits) & 0x7F]
        np.testing.assert_array_equal(np.signbit(decoded[finite]), np.signbit(q8(x[finite])))
        np.testing.assert_array_equal(decoded[finite], q8(x[finite]))

    def test_halves_are_the_values(self):
        half = Q.HALVES.view(np.float16).astype(np.float64)
        for code in range(0x7F):
            self.assertEqual(half[code], Q.VALUES[code])
            self.assertEqual(half[0x80 | code], -Q.VALUES[code])
        self.assertTrue(np.isnan(half[0x7F]) and np.isnan(half[0xFF]))
        self.assertEqual(int(Q.HALVES[0x80]), 0x8000)   # -0 keeps its sign

    def test_the_check_can_fail(self):
        """A conversion with the tie rule of the opposite kind (ties up instead of to even) is caught."""
        bits = np.arange(0x3F000000, 0x3F000000 + (1 << 20), dtype=np.uint32)
        code = Q.reference_codes(bits)
        x = bits.view(np.float32)
        up = np.floor(x.astype(np.float64) * 16 + 0.5) / 16     # 1 / 16 grid in [0.5, 1): spacing 2^-4 is E4M3's there
        wrong = fp8_encode(up).astype(np.uint8)
        half = Q.reference_halves(code)
        got = {f: code if "pack" in f or f == "wg_f32_to_e4m3" else half for f in Q.FORMS}
        self.assertFalse(any(b.any() for b in Q.mismatches(bits, got).values()))
        self.assertTrue(Q.mismatches(bits, {**got, "wg_f32_to_e4m3": wrong})["wg_f32_to_e4m3"].any())
        self.assertTrue(Q.mismatches(bits, {**got, "wg_q8_to_f16": Q.reference_halves(wrong)})["wg_q8_to_f16"].any())


# top bytes of the patterns: 0x00 zero / denormals / 2^-126.., 0x3b..0x3d around 2^-9..2^-6 (subnormal step, smallest
# normals), 0x3f one, 0x43 the saturation edge (448 = 0x43e00000), 0x7f inf and NaN; the same with the sign set
CHUNKS = (0x00, 0x3B, 0x3C, 0x3D, 0x3F, 0x43, 0x7F, 0x80, 0xBC, 0xC3, 0xFF)


class TestGpuSweep(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if os.environ.get("WG_GPU_TEST") != "1":
            raise unittest.SkipTest("set WG_GPU_TEST=1 (and WG_BUILD, WG_ROCM) to run on the GPU")
        cls.build = os.environ.get("WG_BUILD")
        if not cls.build:
            raise unittest.SkipTest("set WG_BUILD to a gfx11 build folder")

    def test_chunks(self):
        total, first, done = Q.check(self.build, os.environ.get("WG_ROCM", ""), CHUNKS, log=lambda *_: None)
        self.assertEqual(done, len(CHUNKS) * Q.CHUNK)
        for form in Q.FORMS:
            with self.subTest(form=form):
                self.assertEqual(int(total[form].sum()), 0, (dict(zip(Q.CLASSES, total[form].tolist())), first.get(form)))


if __name__ == "__main__":
    unittest.main()
