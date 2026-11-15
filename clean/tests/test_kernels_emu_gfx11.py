# SPDX-License-Identifier: MIT
"""The Swin kernels with the gfx11 (RDNA 3) matrix wrappers of kernels/wg_platform.h, run in the host lane emulator in its
gfx11 mode (-DWG_EMU_GFX11: the wrappers are the real source, the matrix instruction is the F16 one of gfx11), with the
gfx11 accumulator row table. Expected values come from the float64 reference (clean/reference/swin.py), never from the
kernels. Same host requirements as test_kernels_emu.py.
"""
import os
import tempfile
import unittest

import numpy as np

from clean.reference import fp8
from clean.reference import swin as S
from clean.runner import build
from clean.tests import test_kernels_emu as TK

_BUILD = None


def build_dir():
    """A folder with wg_emu__gfx11.so, built once (the file name Runner(..., arch="gfx11") loads)."""
    global _BUILD
    if _BUILD is None:
        cxx = TK.host_cxx()
        d = tempfile.mkdtemp(prefix="wg_emu_gfx11_")
        r = TK.subprocess.run(build.emu_command(cxx, os.path.join(d, "wg_emu__gfx11.so"), {"WG_EMU_GFX11": 1}),
                              capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError("gfx11 emulator build failed:\n" + r.stderr)
        _BUILD = d
    return _BUILD


class Gfx11Case:
    """Mixin for the emulator test classes: the gfx11 emulator library and its row table."""

    @classmethod
    def setUpClass(cls):
        from clean.runner.stage_runner import Runner
        cls.Runner = Runner
        cls.runner = Runner(build_dir(), "", backend="emu", arch="gfx11")


class TestRunnerGfx11(Gfx11Case, unittest.TestCase):
    def test_runner_takes_the_gfx11_row_table(self):
        self.assertEqual(self.runner.d_row, "gfx11")

    def test_only_the_emulator_takes_an_arch(self):
        with self.assertRaises(ValueError):
            self.Runner(build_dir(), "", backend="hip", arch="gfx11")
        with self.assertRaises(ValueError):
            self.Runner(build_dir(), "", backend="emu", arch="gfx12")


class TestEnc1Gfx11(Gfx11Case, TK.TestStagesEmu):
    """enc1 (swin_block_c64 and patch_merge_c64) and the other stages of the emulator tests, through the wrappers."""

    CASES = [c for c in TK.TestStagesEmu.CASES if c[0] == "enc1"]

    def test_the_row_table_matters(self):
        """Sanity of the comparison: weights packed for the gfx12 accumulator rows give another result."""
        a, _ = self.run_case("enc1", 8, 12, (2, 2), seed=9)
        wrong = self.Runner(build_dir(), "", backend="emu", arch="gfx11", d_row="gfx12")
        b, _ = self.run_case("enc1", 8, 12, (2, 2), seed=9, runner=wrong)
        self.assertGreater(float(np.mean(a["skip"] != b["skip"])), 0.2)


def f16_bits(x):
    return np.asarray(x, np.float16).view(np.uint16)


class TestWrappers(Gfx11Case, unittest.TestCase):
    """wg_wmma_fp8 and wg_wmma_f16 (probe kernels) against exact arithmetic: lane 16h + r holds row r (A) or column r (B),
    slot (h, j) is K index 8h + j, and element v of D in lane 16h + n is row 2v + h of column n. Sums of 16 products of
    E4M3 (or F16 with 4 bits) values are exact in float64, so the emulator's one rounding to FP32 is reproduced bit for bit."""

    def call(self, kernel, a, b):
        from clean.runner import selftest
        dev = selftest._Dev("emu", build_dir(), "", emu_library="wg_emu__gfx11.so")
        n = a.shape[0]
        (d,) = dev.call(kernel, [a.tobytes(), b.tobytes()], [n * 32 * 32], threads=n * 32)
        return np.frombuffer(d, np.float32).reshape(n, 32, 8)

    @staticmethod
    def exact(A, B):
        """A [n, 32, 8], B [n, 32, 8] decoded values -> D [n, 32, 8] as the instruction lays it out."""
        n = A.shape[0]
        am = np.zeros((n, 16, 16))   # [row, k]
        bm = np.zeros((n, 16, 16))   # [k, column]
        for l in range(32):
            h, r = divmod(l, 16)
            am[:, r, 8 * h:8 * h + 8] = A[:, l]
            bm[:, 8 * h:8 * h + 8, r] = B[:, l]
        d = np.einsum("nrk,nkc->nrc", am, bm)
        out = np.zeros((n, 32, 8))
        for l in range(32):
            for v in range(8):
                out[:, l, v] = d[:, 2 * v + l // 16, l % 16]
        return out.astype(np.float32)

    def test_fp8_wrapper_is_exact_arithmetic(self):
        rng = np.random.default_rng(11)
        n = 64
        a = rng.integers(0, 256, (n, 32, 8), dtype=np.uint8)
        b = rng.integers(0, 256, (n, 32, 8), dtype=np.uint8)
        a[a & 0x7F == 0x7F] = 0x38   # no NaN codes here
        b[b & 0x7F == 0x7F] = 0x38
        got = self.call("probe_wmma_fp8", a, b)
        want = self.exact(fp8.DECODE[a].astype(np.float64), fp8.DECODE[b].astype(np.float64))
        np.testing.assert_array_equal(got, want)

    def test_every_code_is_widened_exactly(self):
        """One-hot A with a 1.0 and B holding every code in turn: the product is the code's value."""
        a = np.zeros((256, 32, 8), np.uint8)
        b = np.zeros((256, 32, 8), np.uint8)
        a[:, 0, 0] = 0x38   # K index 0 of row 0 (1.0)
        for c in range(256):
            b[c, 0, 0] = c   # K index 0 of column 0
        got = self.call("probe_wmma_fp8", a, b)
        # row 0 is element 0 of lane 0 (column 0)
        want = fp8.DECODE.astype(np.float32)
        have = got[:, 0, 0]
        np.testing.assert_array_equal(np.where(np.isnan(want), 0, have), np.where(np.isnan(want), 0, want))
        self.assertTrue(np.all(np.isnan(have[np.isnan(want)])), "a NaN code (0x7f, 0xff) must give a NaN")

    def test_f16_wrapper_is_exact_arithmetic(self):
        rng = np.random.default_rng(12)
        n = 64
        a = (rng.integers(-16, 17, (n, 32, 8)) / 16.0).astype(np.float16)
        b = (rng.integers(-16, 17, (n, 32, 8)) / 16.0).astype(np.float16)
        got = self.call("probe_wmma_f16", f16_bits(a), f16_bits(b))
        np.testing.assert_array_equal(got, self.exact(a.astype(np.float64), b.astype(np.float64)))

    def test_the_instruction_reads_each_half_for_its_own_rows(self):
        """As measured on the card: with lanes 16-31 of A different from lanes 0-15, only the odd rows change."""
        rng = np.random.default_rng(13)
        a = (rng.integers(-8, 9, (8, 32, 16))).astype(np.float16)
        b = (rng.integers(-8, 9, (8, 32, 16))).astype(np.float16)
        a[:, 16:] = a[:, :16]
        b[:, 16:] = b[:, :16]
        base = self.call("probe_wmma16", f16_bits(a), f16_bits(b))
        a2 = a.copy()
        a2[:, 16:] = rng.integers(-8, 9, (8, 16, 16)).astype(np.float16)
        got = self.call("probe_wmma16", f16_bits(a2), f16_bits(b))
        np.testing.assert_array_equal(got[:, :16], base[:, :16])
        self.assertTrue(np.any(got[:, 16:] != base[:, 16:]))


if __name__ == "__main__":
    unittest.main()
