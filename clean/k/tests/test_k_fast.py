# SPDX-License-Identifier: MIT
"""Runtime option k_fast=1 (opt-in, not bit-exact): the "_f" variants of the input and output stage kernels
(clean/k/kernels/k_io.hip, FAST / KIO_FAST forms: hardware reciprocal estimates instead of correctly rounded
divisions, the luma history cubics and clamps as packed FP16, the weight sum instead of the largest exponent for the
filter quotient). On the GPU through the runtime library, against the default (exact) path on the same inputs:

- the option is reported (wg_rt_kernels "k_fast=1") and checked (k_fast=2 is refused);
- one pass at a time: before every frame the k_fast runtime gets the exact runtime's histories (k.Hc, k.L, k.Hh), so
  each frame's difference is that of one pass of the fast kernels (not the history feedback); on a moving textured
  HDR / LDR scene the tonemapped luma y / (1 + y) of every frame must stay within the bounds below, finite everywhere.

GPU only: WG_K_GPU_BUILD=<full build dir (clean/runner/build.py, with the runtime library)>; run under the machine's
GPU lock. Run: python -m unittest clean.k.tests.test_k_fast
"""
import ctypes
import os
import unittest

import numpy as np

from clean.k.reference.network import random_network_params
from clean.k.tests.test_k_runtime_gpu import KRuntimeExt, gpu_build, write_k_model

PSNR_MIN_DB = 70.0  # per frame, tonemapped luma, peak 1
MAX_ABS = 0.02  # per pixel, tonemapped luma


def scene(t, rw, rh, hdr):
    """A textured plane panning 1.37 / 0.61 render pixels per frame (aliased: tiles, mortar lines, a zone plate,
    HDR highlights) and its motion; jitter Halton (2, 3)."""
    def halton(i, b):
        f, r = 1.0, 0.0
        while i > 0:
            f /= b
            r += f * (i % b)
            i //= b
        return r
    jx, jy = halton(t % 16 + 1, 2) - 0.5, halton(t % 16 + 1, 3) - 0.5
    ii, jj = np.meshgrid(np.arange(rw, dtype=np.float64), np.arange(rh, dtype=np.float64))
    X, Y = ii + 0.5 - jx + 1.37 * t, jj + 0.5 - jy + 0.61 * t
    tx, ty = np.floor(X / 11.0), np.floor(Y / 11.0)
    h = ((tx.astype(np.int64) * 73856093) ^ (ty.astype(np.int64) * 19349663)) & 0xFFFF
    alb = np.stack([0.2 + 0.7 * (h & 255) / 255.0, 0.2 + 0.7 * (h >> 8) / 255.0, 0.2 + 0.5 * ((h * 7) & 255) / 255.0], -1)
    alb[(X / 11.0 - tx < 0.08) | (Y / 11.0 - ty < 0.08)] = 0.03
    zp = 0.5 + 0.5 * np.sin(((X - 60.0) ** 2 + (Y - 40.0) ** 2) * 0.02)
    alb = np.where(((np.abs(X - 60.0) < 25) & (np.abs(Y - 40.0) < 25))[..., None], zp[..., None], alb)
    if hdr:
        lx, ly = np.mod(X, 23.0) - 11.5, np.mod(Y, 23.0) - 11.5
        alb = alb * 1.5 + (np.exp(-(lx * lx + ly * ly) / 2.0) * 30.0)[..., None]
    color = np.zeros((rh, rw, 4), np.float32)
    color[..., :3] = alb
    motion = np.empty((rh, rw, 2), np.float32)
    motion[..., 0], motion[..., 1] = 1.37, 0.61
    return dict(color=color, motion=motion, depth=np.full((rh, rw), 0.5, np.float32), render_size=(rw, rh),
                jitter=(float(np.float32(jx)), float(np.float32(jy))), mv_scale=(1.0, 1.0), mv_offset=(0.0, 0.0),
                pre_exposure=1.0, reset=t == 0, E_cur=1.0, E_aux=None, e1=1.0, e2=1.0)


def luma_tm(rgb):
    y = np.maximum(rgb, 0.0) @ np.array([0.2126, 0.7152, 0.0722], np.float32)
    return y / (1.0 + y)


class TestKFast(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.build = gpu_build()
        cls.mdir = os.path.join(cls.build, "k_fast_test")
        cls.wgcm, cls.map = write_k_model(cls.mdir, random_network_params(11))

    def runtime(self, hdr, options, out=(240, 136), render=(160, 90)):
        return KRuntimeExt(self.build, self.mdir, self.map, out, render, hdr, False, False, options)

    def test_option_checked(self):
        with self.assertRaisesRegex(RuntimeError, "k_fast"):
            self.runtime(True, "k_fast=2")

    def sync(self, a, b):
        for name in ("k.Hc", "k.L", "k.Hh"):
            data = a.buffer(name, np.uint8)
            p, n = ctypes.c_size_t(), ctypes.c_size_t()
            b.check(b.lib.wg_rt_buffer(b.rt, name.encode(), ctypes.byref(p), ctypes.byref(n)), name)
            self.assertEqual(n.value, data.nbytes, name)
            b.upload(p.value, data)

    def one_pass(self, hdr, network):
        opts = "" if network == "fused" else f"network={network}"
        a = self.runtime(hdr, opts)
        b = self.runtime(hdr, ";".join(o for o in (opts, "k_fast=1") if o))
        try:
            for t in range(8):
                x = scene(t, 160, 90, hdr)
                ya = luma_tm(a.frame(x, 0)[0][0])
                gb = b.frame(x, 0)[0][0]
                self.assertTrue(np.all(np.isfinite(gb)), f"frame {t}")
                e = luma_tm(gb) - ya
                mse = float(np.mean(e.astype(np.float64) ** 2))
                psnr = 10 * np.log10(1.0 / mse) if mse > 0 else np.inf
                self.assertGreaterEqual(psnr, PSNR_MIN_DB, f"frame {t} hdr {hdr} {network}")
                self.assertLessEqual(float(np.abs(e).max()), MAX_ABS, f"frame {t} hdr {hdr} {network}")
                self.sync(a, b)
            ka = a.lib.wg_rt_kernels(a.rt).decode().split()
            kb = b.lib.wg_rt_kernels(b.rt).decode().split()
            self.assertNotIn("k_fast=1", ka)
            self.assertIn("k_fast=1", kb)
        finally:
            del a, b

    def test_hdr_fused16(self):
        self.one_pass(True, "fused16")

    def test_hdr_fused(self):
        self.one_pass(True, "fused")

    def test_ldr_fused16(self):
        self.one_pass(False, "fused16")


if __name__ == "__main__":
    unittest.main()
