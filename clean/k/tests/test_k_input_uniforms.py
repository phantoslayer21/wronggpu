# SPDX-License-Identifier: MIT
"""Native KiHost precompute versus the actual GPU helper, with CPU-mode fallbacks.

Requires test-only k_input_uniforms_probe.co and input_uniforms_probe.dll in
WG_K_GPU_BUILD. Run under the shared GPU lock. No CPU mode escapes the probe call.
"""
import ctypes
import json
import os
from pathlib import Path
import unittest

import numpy as np

from clean.k.runner.k_io import KFrameIn


class TestInputUniforms(unittest.TestCase):
    def test_native_rounding_guard_and_gpu_constants(self):
        build = Path(os.environ.get("WG_K_GPU_BUILD", ""))
        if not os.environ.get("WG_K_GPU_BUILD") or not (build / "input_uniforms_probe.dll").is_file():
            self.skipTest("GPU and test-only CPU/GPU uniform probes required")
        from clean.runner.hip import Hip
        dev = Hip("")
        fn = dev.get_function(dev.load_module((build / "k_input_uniforms_probe.co").read_bytes()), "ki_host_audit")
        lib = ctypes.CDLL(str((build / "input_uniforms_probe.dll").resolve()))
        cpu = lib.test_input_uniforms
        cpu.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint, ctypes.c_int]
        cpu.restype = ctypes.c_int
        n = 65536
        rng = np.random.default_rng(101016)
        frames = np.zeros((n, ctypes.sizeof(KFrameIn)), np.uint8)
        ints = frames.view(np.int32).reshape(n, -1)
        floats = frames.view(np.float32).reshape(n, -1)
        W, H = rng.integers(2, 16385, (2, n))
        W[:16383] = np.arange(2, 16385)
        sx, sy = rng.integers(0, 2**32, (2, n), dtype=np.uint32).view(np.float32)
        # Derive IEEE-754 boundaries as raw integer encodings so DAZ cannot
        # erase the subnormal inputs before the native guard sees them.
        sign = 1 << 31
        minimum_normal = 1 << 23
        safe_floor = 1 << 24
        infinity = 255 << 23
        positive_edges = [0, 1, minimum_normal-1, minimum_normal,
                          safe_floor-1, safe_floor, safe_floor+1, 127 << 23,
                          infinity-1, infinity]
        edges = np.array([bits for magnitude in positive_edges
                          for bits in (magnitude, magnitude | sign)] +
                         [infinity | (1 << 22), infinity | 1], np.uint32)
        sx[:4096] = np.resize(edges.view(np.float32), 4096)
        sy[:4096] = np.resize(np.roll(edges.view(np.float32), 7), 4096)
        values = dict(W=W, H=H, rw=np.maximum(1, (W*.667).astype(np.int32)),
                      rh=np.maximum(1, (H*.667).astype(np.int32)), Bw=rng.integers(1, 4097, n),
                      Bh=rng.integers(1, 4097, n), flags=(np.arange(n)%2)*2, s=.5, sx=sx, sy=sy)
        for key, value in values.items():
            target = floats if key in ('s', 'sx', 'sy') else ints
            target[:, getattr(KFrameIn, key).offset//4] = value
        pointers = [dev.malloc(frames.nbytes), dev.malloc(n*48), dev.malloc(n*4)]
        try:
            dev.upload(pointers[0], frames.tobytes())
            dev.memset(pointers[1], 0xcd, n*48)
            dev.memset(pointers[2], 0xcd, n*4)
            dev.launch(fn, (n+127)//128, 128, [ctypes.c_void_p(p) for p in pointers] + [ctypes.c_uint(n)])
            dev.sync()
            gpu = np.frombuffer(dev.download(pointers[1], n*48), np.float32).reshape(n, 12)
            self.assertTrue(np.all(np.frombuffer(dev.download(pointers[2], n*4), np.uint32) == 0x4b49484f))
            records = []
            normal_mask = None
            for mode in range(5):
                host = np.full((n, 12), np.nan, np.float32)
                accepted = np.full(n, 0xffffffff, np.uint32)
                self.assertEqual(cpu(frames.ctypes.data, host.ctypes.data, accepted.ctypes.data, n, mode), 1)
                if mode >= 2:
                    self.assertFalse(accepted.any(), "non-nearest rounding must retain GPU uniforms")
                else:
                    expected = np.ones(n, bool)
                    for scale in (sx, sy):
                        bits = scale.view(np.uint32) & 0x7fffffff
                        expected &= (bits == 0) | ((bits >= 0x01000000) & (bits < 0x7f800000))
                    expected &= np.isfinite(host[:, 8]) & np.isfinite(host[:, 9])
                    np.testing.assert_array_equal(accepted.astype(bool), expected)
                    if mode == 0:
                        normal_mask = accepted.copy()
                    else:
                        np.testing.assert_array_equal(accepted, normal_mask)
                    selected = accepted.astype(bool)
                    np.testing.assert_array_equal(host[selected].view(np.uint32), gpu[selected].view(np.uint32))
                records.append(dict(cpu_mode=mode, rows=n, accepted_rows=int(accepted.sum()),
                                    accepted_values=int(accepted.sum())*12, mismatches=0))
            target = os.environ.get("WG_K_UNIFORM_AUDIT_JSON")
            if target:
                Path(target).write_text(json.dumps(dict(records=records, cpu_state_restored=True), indent=2))
        finally:
            for p in pointers:
                dev.free(p)
