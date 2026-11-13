# SPDX-License-Identifier: MIT
"""Exhaustive GPU validation of the production output reciprocal/cubic-beta helpers.

WG_K_GPU_BUILD selects output directory and enables tests. Run under the shared
GPU lock. WG_K_TEST_ROCM may select a compiler SDK independently of the runtime.
Three separate kernels prevent native/candidate expression sharing; a 64-bit
mismatch counter cannot wrap when checking all 2**32 input patterns.
"""
import ctypes
import os
from pathlib import Path
import struct
import subprocess
import unittest

from clean.runner.build import device_command, find_clang
from clean.runner.hip import Hip


class TestOutputArithmetic(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        build = os.environ.get("WG_K_GPU_BUILD")
        if not build:
            raise unittest.SkipTest("GPU only: set WG_K_GPU_BUILD and use shared GPU lock")
        root = Path(__file__).resolve().parents[3]
        sdk = os.environ.get("WG_K_TEST_ROCM") or os.environ.get("ROCM_PATH") or r"C:\Program Files\AMD\ROCm\7.2"
        obj = Path(build).resolve() / "k_output_arithmetic_probe.co"
        cmd = device_command(find_clang(sdk), sdk, str(Path(__file__).with_name("k_output_arithmetic_probe.hip")),
                             str(obj), extra=["-I", str(root/"clean/kernels"), "-mwavefrontsize64"])
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode:
            raise RuntimeError(proc.stdout + proc.stderr)
        cls.dev = Hip(os.environ.get("ROCM_PATH", ""))
        mod = cls.dev.load_module(obj.read_bytes())
        cls.fn = {name: cls.dev.get_function(mod, "k_probe_"+name)
                  for name in ("native_rcp", "candidate_rcp", "native_axis", "candidate_axis", "candidate_bounded", "compare")}

    def compare_range(self, kind, first, end):
        dev, chunk = self.dev, 1 << 22
        a, b, result = dev.malloc(chunk*4), dev.malloc(chunk*4), dev.malloc(16)
        try:
            dev.upload(result, struct.pack("<QII", 0, 0xffffffff, 0))
            for lo in range(first, end, chunk):
                count = min(chunk, end-lo)
                # Different sentinels make unwritten elements disagree instead of
                # accidentally comparing two untouched allocation contents equal.
                dev.memset(a, 0xa5, count*4)
                dev.memset(b, 0x5a, count*4)
                args = [ctypes.c_uint(lo), ctypes.c_uint(count)]
                dev.launch(self.fn["native_rcp" if kind == "bounded" else "native_"+kind], 4096, 128, [ctypes.c_void_p(a)]+args)
                dev.launch(self.fn["candidate_"+kind], 4096, 128, [ctypes.c_void_p(b)]+args)
                dev.launch(self.fn["compare"], 4096, 128,
                           [ctypes.c_void_p(a), ctypes.c_void_p(b), ctypes.c_void_p(result),
                            ctypes.c_void_p(result+8)]+args)
            dev.sync()
            self.assertNotEqual(dev.download(a, 4), bytes([0xa5])*4)
            self.assertNotEqual(dev.download(b, 4), bytes([0x5a])*4)
            errors, lowest, _ = struct.unpack("<QII", dev.download(result, 16))
            self.assertEqual(errors, 0, f"{kind}: {errors} differences; first input bits {lowest:#x}")
            self.assertEqual(lowest, 0xffffffff)
            print(f"{kind}: {end-first} patterns [{first:#x},{end-1:#x}], 0 mismatches", flush=True)
        finally:
            for p in (a,b,result):
                dev.free(p)

    def test_reciprocal_all_float_bits(self):
        self.compare_range("rcp", 0, 1 << 32)

    def test_bounded_reciprocal_complete_domain(self):
        self.compare_range("bounded", 0x2f800000, 0x4f800001)  # [2**-32, 2**32]

    def test_axis_complete_fraction_domain(self):
        self.compare_range("axis", 0, 0x3f800001)
        self.compare_range("axis", 0x80000000, 0x80000001)  # negative zero


if __name__ == "__main__":
    unittest.main()
