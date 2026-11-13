# SPDX-License-Identifier: MIT
"""Every half normalization input: exact non-NaN result bits and NaN classes.

NaN payload/sign bits are intentionally reported, not required to match. Run
under the shared GPU lock with WG_K_GPU_BUILD selecting the output directory.
"""
import ctypes
import os
from pathlib import Path
import subprocess
import unittest

import numpy as np

from clean.runner.build import device_command, find_clang
from clean.runner.hip import Hip


class TestNormR16(unittest.TestCase):
    def test_all_half_inputs(self):
        build = os.environ.get("WG_K_GPU_BUILD")
        if not build:
            self.skipTest("GPU only: set WG_K_GPU_BUILD and use shared GPU lock")
        root = Path(__file__).resolve().parents[3]
        sdk = os.environ.get("WG_K_TEST_ROCM") or os.environ.get("ROCM_PATH") or r"C:\Program Files\AMD\ROCm\7.2"
        obj = Path(build).resolve() / "k_norm_r16_probe.co"
        cmd = device_command(find_clang(sdk), sdk,
                             str(Path(__file__).with_name("k_norm_r16_probe.hip")), str(obj),
                             extra=["-I", str(root / "clean/kernels")])
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode:
            raise RuntimeError(result.stdout + result.stderr)
        dev = Hip(os.environ.get("ROCM_PATH", ""))
        mod = dev.load_module(obj.read_bytes())
        original = dev.get_function(mod, "k_norm_r16_original")
        for name in ("k_norm_r16_candidate", "k_norm_r16_candidate_select"):
            with self.subTest(candidate=name):
                self._check(dev, original, dev.get_function(mod, name), name)

    def _check(self, dev, original, candidate, name):
        a, b, marker = dev.malloc(65536 * 2), dev.malloc(65536 * 2), dev.malloc(4)
        try:
            dev.memset(a, 0xa5, 65536 * 2)
            # Negative finite sentinels cannot be valid reciprocal-sqrt outputs.
            dev.memset(b, 0xb6, 65536 * 2)
            dev.memset(marker, 0xcc, 4)
            dev.launch(original, 512, 128, [ctypes.c_void_p(a)])
            dev.launch(candidate, 512, 128, [ctypes.c_void_p(b), ctypes.c_void_p(marker)])
            dev.sync()
            self.assertEqual(int(np.frombuffer(dev.download(marker, 4), np.uint32)[0]), 0x1201)
            aa = np.frombuffer(dev.download(a, 65536 * 2), np.uint16)
            bb = np.frombuffer(dev.download(b, 65536 * 2), np.uint16)
            self.assertFalse(np.any(aa == 0xa5a5), "original left an output unwritten")
            self.assertFalse(np.any(bb == 0xb6b6), "candidate left an output unwritten")
            nan_a, nan_b = (aa & 0x7fff) > 0x7c00, (bb & 0x7fff) > 0x7c00
            np.testing.assert_array_equal(nan_a, nan_b, err_msg="NaN class differs")
            np.testing.assert_array_equal(aa[~nan_a], bb[~nan_a],
                                          err_msg="finite/Inf/signed-zero result bits differ")
            for bits in (0, 0x8000, 0x7c00):
                self.assertEqual(int(aa[bits]), 0)
                self.assertEqual(int(bb[bits]), 0)
            self.assertTrue(bool(nan_a[0xbc00]))  # negative finite input stays NaN
            payloads = int(np.count_nonzero((aa != bb) & nan_a))
            print(f"normalization r16 {name}: 65536 inputs, no non-NaN bit/class differences; "
                  f"{payloads} NaN payload/sign differences", flush=True)
        finally:
            for ptr in (a, b, marker):
                dev.free(ptr)


if __name__ == "__main__":
    unittest.main()
