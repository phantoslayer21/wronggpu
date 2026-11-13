# SPDX-License-Identifier: MIT
"""network=fused8 (clean/k/kernels/k_block8.h: FP8 forms of e0, e1, d1, d0; opt-in, not bit-exact) in the runtime
library on the GPU, on the synthetic model K of test_k_runtime_gpu and a short moving sequence:

- the runtime reports the variant and the four FP8 stages; an unknown fp8_stages entry is refused;
- fp8_stages=none runs the "fused16" kernels: every frame bit for bit network=fused16 (the selection adds nothing else);
- the FP8 output is finite and close to the exact output (fused): PSNR of tonemapped luma above 30 dB in every frame
  (measured about 40 dB closed loop on the 1440p sequence of the 2026-10-14 FP8 study; fused16 about 55 dB);
- fused8 is deterministic (the same frames twice give the same bits).

GPU only: WG_K_GPU_BUILD=<full build dir with the runtime library>; run under the machine's GPU lock.
Run: python -m unittest clean.k.tests.test_k_fused8
"""
import os
import unittest

import numpy as np

from clean.k.reference.network import random_network_params
from clean.k.tests.test_k_runtime_gpu import KRuntimeExt, gpu_build, write_k_model

OUT, REN = (480, 272), (320, 180)


def scene(k):
    """A panning textured frame with a moving disk (motion = previous position - current, render pixels)."""
    rw, rh = REN
    jx, jy = (((k * 7) % 16) + 0.5) / 16 - 0.5, (((k * 11) % 16) + 0.5) / 16 - 0.5
    ys, xs = np.mgrid[0:rh, 0:rw].astype(np.float64)
    X, Y = xs + 0.5 - jx + 1.5 * k, ys + 0.5 - jy + 0.5 * k
    col = np.zeros((rh, rw, 4), np.float32)
    col[..., 0] = 0.4 + 0.3 * np.sin(0.05 * X) * np.cos(0.04 * Y) + 0.1 * np.sin(1.2 * X + 0.5 * Y)
    col[..., 1] = 0.35 + 0.25 * ((np.floor(X / 23) + np.floor(Y / 17)) % 2)
    col[..., 2] = 0.3 + 0.2 * np.cos(0.03 * Y - 0.02 * X) + 3.0 * ((np.sin(0.37 * X) * np.sin(0.41 * Y)) > 0.995)
    mot = np.empty((rh, rw, 2), np.float32)
    mot[..., 0], mot[..., 1] = 1.5, 0.5
    depth = np.full((rh, rw), 0.8, np.float32)
    inside = (xs - 80 - 2.5 * k) ** 2 + (ys - 90) ** 2 < 40 ** 2
    col[inside, :3] = [0.9, 0.5, 0.2]
    mot[inside] = [-2.5, 0.0]
    depth[inside] = 0.3
    return dict(color=col, motion=mot, depth=depth, render_size=REN, jitter=(float(np.float32(jx)), float(np.float32(jy))),
                mv_scale=(1.0, 1.0), mv_offset=(0.0, 0.0), pre_exposure=1.0, reset=k == 0, E_cur=1.0, E_aux=None,
                e1=1.0, e2=1.0)


def tm_luma(rgb):
    y = np.maximum(0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2], 0.0).astype(np.float64)
    return y / (1.0 + y)


class TestKFused8(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.build = gpu_build()
        cls.mdir = os.path.join(cls.build, "k_fused8_test")
        _, cls.map = write_k_model(cls.mdir, random_network_params(11))

    def rt(self, options):
        return KRuntimeExt(self.build, self.mdir, self.map, OUT, REN, True, False, False, options)

    def frames(self, rt, n):
        return [rt.frame(scene(k), 0)[0][0] for k in range(n)]

    def test_kernels_text(self):
        rt = self.rt("network=fused8")
        rt.frame(scene(0), 0)
        text = rt.lib.wg_rt_kernels(rt.rt).decode()
        self.assertIn("network=fused8 fp8_stages=4 ", text)
        self.assertNotIn("fp8_kept", text)

    def test_unknown_fp8_stage(self):
        with self.assertRaisesRegex(RuntimeError, "fp8_stages: e2 is not one of"):
            self.rt("network=fused8;fp8_stages=e0,e2")

    def test_none_is_fused16(self):
        a = self.frames(self.rt("network=fused16"), 6)
        b = self.frames(self.rt("network=fused8;fp8_stages=none"), 6)
        for k, (x, y) in enumerate(zip(a, b)):
            self.assertTrue(np.array_equal(x.view(np.uint32), y.view(np.uint32)), f"frame {k}")

    def test_error_and_determinism(self):
        ref = self.frames(self.rt(""), 12)
        f8 = self.frames(self.rt("network=fused8"), 12)
        again = self.frames(self.rt("network=fused8"), 12)
        for k, (r, o, o2) in enumerate(zip(ref, f8, again)):
            self.assertTrue(np.all(np.isfinite(o)), f"frame {k}: non-finite output")
            self.assertTrue(np.array_equal(o.view(np.uint32), o2.view(np.uint32)), f"frame {k}: not deterministic")
            d = tm_luma(o) - tm_luma(r)
            psnr = 10 * np.log10(1.0 / max(float(np.mean(d * d)), 1e-20))
            self.assertGreater(psnr, 30.0, f"frame {k}: PSNR {psnr:.2f} dB")
            self.assertGreater(float(np.mean(d * d)), 0.0, f"frame {k}: FP8 output equals the exact one")


if __name__ == "__main__":
    unittest.main()
