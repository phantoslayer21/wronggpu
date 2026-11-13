# SPDX-License-Identifier: MIT
"""GPU audit of the actual guarded hidden helper; run under the shared GPU lock."""
import ctypes as ct
import os
from pathlib import Path
import subprocess
import unittest

import numpy as np

from clean.runner.build import device_command, find_clang
from clean.runner.hip import Hip


class TestHiddenGuard(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        build = os.environ.get("WG_K_GPU_BUILD")
        if not build:
            raise unittest.SkipTest("GPU only: set WG_K_GPU_BUILD and use shared GPU lock")
        root = Path(__file__).resolve().parents[3]
        sdk = os.environ.get("WG_K_TEST_ROCM") or os.environ.get("ROCM_PATH") or r"C:\Program Files\AMD\ROCm\7.2"
        obj = Path(build).resolve() / "k_hidden_guard_probe.co"
        cmd = device_command(find_clang(sdk), sdk,
                             str(Path(__file__).with_name("k_hidden_guard_probe.hip")), str(obj),
                             extra=["-I", str(root / "clean/kernels"),
                                    "-I", str(root / "clean/k/kernels")])
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode:
            raise RuntimeError(result.stdout + result.stderr)
        cls.dev = Hip(os.environ.get("ROCM_PATH", ""))
        cls.mod = cls.dev.load_module(obj.read_bytes())

    def launch(self, name, args):
        self.dev.launch(self.dev.get_function(self.mod, name), 4096, 128, args)

    def test_hardware_bound(self):
        """Every positive FP32 input from 1 through 2**34, including endpoints."""
        dev = self.dev
        chunk = 1 << 22
        a, b, r = dev.malloc(4 * chunk), dev.malloc(4 * chunk), dev.malloc(36)
        totals = [0] * 5
        try:
            for first in range(0x3f800000, 0x50800001, chunk):
                n = min(chunk, 0x50800001 - first)
                dev.memset(a, 0xa5, 4 * n)
                dev.memset(b, 0x5a, 4 * n)
                dev.memset(r, 0, 36)
                dev.memset(r + 32, 0xff, 4)
                self.launch("hidden_rsq_native", [ct.c_void_p(a), ct.c_uint(first), ct.c_uint(n)])
                self.launch("hidden_rsq_estimate", [ct.c_void_p(b), ct.c_uint(first), ct.c_uint(n)])
                self.launch("hidden_rsq_compare", [ct.c_void_p(a), ct.c_void_p(b), ct.c_void_p(r), ct.c_uint(first), ct.c_uint(n)])
                dev.sync()
                row = list(map(int, np.frombuffer(dev.download(r, 36), np.uint32)))
                self.assertEqual(sum(row[:5]), n)
                self.assertEqual(row[2:5], [0, 0, 0])
                self.assertLessEqual(row[5], 1)
                totals = [v + w for v, w in zip(totals, row[:5])]
                if first == 0x3f800000:
                    self.assertEqual(dev.download(a, 4), bytes.fromhex("0000803f"))
                    self.assertEqual(dev.download(b, 4), bytes.fromhex("0000803f"))
                if first + n == 0x50800001:
                    self.assertEqual(dev.download(a + 4 * (n - 1), 4), bytes.fromhex("00000037"))
                    self.assertEqual(dev.download(b + 4 * (n - 1), 4), bytes.fromhex("00000037"))
            self.assertEqual(sum(totals), 285212673)
            print("hidden rsq exhaustive buckets:", totals, flush=True)
        finally:
            for ptr in (a, b, r):
                dev.free(ptr)

    def test_actual_guarded_outputs(self):
        dev = self.dev
        n = 1 << 22
        h, a, b = dev.malloc(n * 16), dev.malloc(n * 8), dev.malloc(n * 8)
        flags, r = dev.malloc(n * 4), dev.malloc(16)
        try:
            dev.memset(r, 0xa5, 16)
            self.launch("hidden_marker", [ct.c_void_p(r)])
            self.assertEqual(int(np.frombuffer(dev.download(r, 4), np.uint32)[0]), 0x1201)
            total_fast = total_fallback = 0
            for mode in range(5):
                dev.memset(a, 0x59, n * 8)
                dev.memset(b, 0xa6, n * 8)
                dev.memset(flags, 0xcc, n * 4)
                dev.memset(r, 0, 16)
                self.launch("make_h", [ct.c_void_p(h), ct.c_uint(n), ct.c_uint(mode)])
                self.launch("hidden_reference", [ct.c_void_p(h), ct.c_void_p(a), ct.c_uint(n)])
                self.launch("hidden_candidate", [ct.c_void_p(h), ct.c_void_p(b), ct.c_void_p(flags), ct.c_uint(n)])
                self.launch("compare_hidden", [ct.c_void_p(a), ct.c_void_p(b), ct.c_void_p(flags), ct.c_void_p(r), ct.c_uint(n)])
                dev.sync()
                bad, fast, fallback, invalid = map(int, np.frombuffer(dev.download(r, 16), np.uint32))
                self.assertEqual(bad, 0, (mode, bad))
                self.assertEqual(invalid, 0)
                self.assertEqual(fast + fallback, n)
                total_fast += fast
                total_fallback += fallback
                print("hidden actual branch:", mode, fast, fallback, flush=True)
            self.assertGreater(total_fast, 0, "compiled candidate must execute its fast path")
            self.assertGreater(total_fallback, 0, "fallback must also be exercised")
        finally:
            for ptr in (h, a, b, flags, r):
                dev.free(ptr)

    def test_rounding_interval(self):
        dev = self.dev
        r = dev.malloc(8)
        try:
            dev.memset(r, 0, 8)
            first, count = 0x38800000, 0x477fe000 - 0x38800000 + 1
            self.launch("half_interval", [ct.c_void_p(r), ct.c_uint(first), ct.c_uint(count)])
            accepted, bad = map(int, np.frombuffer(dev.download(r, 8), np.uint32))
            self.assertGreater(accepted, count * 0.99)
            self.assertEqual(bad, 0)
            print("hidden interval:", count, accepted, bad, flush=True)
        finally:
            dev.free(r)


if __name__ == "__main__":
    unittest.main()
