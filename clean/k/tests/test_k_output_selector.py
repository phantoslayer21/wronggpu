# SPDX-License-Identifier: MIT
"""Exhaustive GPU equality of production selector indices for every half input."""
import ctypes
import os
from pathlib import Path
import subprocess
import unittest

import numpy as np

from clean.runner.build import device_command, find_clang
from clean.runner.hip import Hip


class TestOutputSelector(unittest.TestCase):
    def test_all_half_inputs(self):
        build = os.environ.get("WG_K_GPU_BUILD")
        if not build:
            self.skipTest("GPU only: set WG_K_GPU_BUILD and use shared GPU lock")
        root = Path(__file__).resolve().parents[3]
        sdk = os.environ.get("WG_K_TEST_ROCM") or os.environ.get("ROCM_PATH") or r"C:\Program Files\AMD\ROCm\7.2"
        obj = Path(build).resolve() / "k_output_selector_probe.co"
        cmd = device_command(find_clang(sdk), sdk,
                             str(Path(__file__).with_name("k_output_selector_probe.hip")), str(obj),
                             extra=["-I", str(root / "clean/kernels"), "-mwavefrontsize64"])
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode:
            raise RuntimeError(result.stdout + result.stderr)
        dev = Hip(os.environ.get("ROCM_PATH", ""))
        mod = dev.load_module(obj.read_bytes())
        native = dev.get_function(mod, "k_selector_native")
        candidate = dev.get_function(mod, "k_selector_fast")
        a, b = dev.malloc(65536 * 8), dev.malloc(65536 * 8)
        try:
            dev.memset(a, 0xa5, 65536 * 8)
            dev.memset(b, 0x5a, 65536 * 8)
            dev.launch(native, 512, 128, [ctypes.c_void_p(a)])
            dev.launch(candidate, 512, 128, [ctypes.c_void_p(b)])
            aa = np.frombuffer(dev.download(a, 65536 * 8), np.uint32).reshape(-1, 2)
            bb = np.frombuffer(dev.download(b, 65536 * 8), np.uint32).reshape(-1, 2)
            self.assertFalse(np.any(aa == 0xa5a5a5a5))
            self.assertFalse(np.any(bb == 0x5a5a5a5a))
            np.testing.assert_array_equal(aa[:, 1], bb[:, 1])
            both_nan = ((aa[:, 0] & 0x7fff) > 0x7c00) & ((bb[:, 0] & 0x7fff) > 0x7c00)
            self.assertFalse(np.any((aa[:, 0] != bb[:, 0]) & ~both_nan))
            print("selector: 65536 inputs, identical indices and finite quotient bits", flush=True)
        finally:
            dev.free(a)
            dev.free(b)


if __name__ == "__main__":
    unittest.main()
