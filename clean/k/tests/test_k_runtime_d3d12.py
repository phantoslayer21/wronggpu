# SPDX-License-Identifier: MIT
"""Model K's per-frame runtime on the D3D12 devices against the HIP backend, frame by frame, byte for byte (issue #15).

The same frames (a synthetic model K, random FP16 parameters; clean/k/tests/test_k_runtime_gpu.py) go through two
runtimes of the same build: HIP (backend 0) and a D3D12 device, and after every frame the game output (RGBA16F rows or
dense RGB float32, and the alpha) and the K buffers (exposure.pair, exposure.measured, k.scalars, k.x0, the compact
pixel head k.Np, the histories k.Hc / k.L / k.Hh) must be identical, compared as raw bit patterns:
- the hybrid device (backend 2: the ports of dx12/manifest.json run on D3D12, any other kernel on HIP, one arena):
  every kernel the frame launches must have run on D3D12 (the launch trace), so that an unported kernel cannot hide
  behind HIP running on both sides;
- the native device (backend 3, no HIP).
Sequences: a reset frame, steady frames with a pan, a game reset, a render-size change (the history is kept); HDR with a
value, with the game's exposure texel, and with auto-exposure, and LDR; output rows RGBA16F and dense RGB.

GPU only: WG_K_GPU_BUILD=<full build dir: clean/runner/build.py with the runtime library, and clean/dx12/build_dx12.py
--model-k (or --only the K kernels and exposure_auto_diet) into the same folder>, ROCM_PATH; run each test under the
machine's GPU lock.
Run: python -m unittest clean.k.tests.test_k_runtime_d3d12
"""
import os
import sys
import unittest

import numpy as np

# read once per process, at the first launch of the first runtime
TRACE = os.path.join(os.environ.get("WG_K_GPU_BUILD", "."), "k_d3d12_launch_trace.txt")
os.environ.setdefault("WG_DX12_KERNELS", "all")
os.environ.setdefault("WG_LAUNCH_TRACE", TRACE)

from clean.k.reference.network import random_network_params  # noqa: E402
from clean.k.tests.test_k_runtime_gpu import KRuntimeExt, gpu_build, inputs, write_k_model  # noqa: E402

OUT, MAX_RENDER = (200, 120), (130, 78)
BUFFERS = ["exposure.pair", "exposure.measured", "k.scalars", "k.x0", "k.Np", "k.Hc", "k.L", "k.Hh"]
# kernels a HDR / LDR frame of the default path launches (the D3D12 ports must run every one of them in the hybrid device)
STAGES = ["k_stage2_e0", "k_stage2_e1", "k_stage_e2", "k_stagew2_e3", "k_stagew4_e4", "k_stagew4_m", "k_stagew4_d4",
          "k_stagew2_d3", "k_stage_d2", "k_stage2_d1", "k_stage2_d0_hp"]
ALWAYS = ["k_expo_scalars", "k_out_pixel_c"]
INPUTS = ["kx/k_in_tiled", "kx/k_in_tiled_uniform"]


def trace_since(pos):
    if not os.path.exists(TRACE):
        return [], pos
    with open(TRACE, "rb") as f:
        f.seek(pos)
        data = f.read()
    return data.decode(errors="replace").splitlines(), pos + len(data)


class TestKRuntimeD3D12(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.build = gpu_build()
        if not os.path.exists(os.path.join(cls.build, "dx12", "manifest.json")):
            raise unittest.SkipTest("no dx12/manifest.json in WG_K_GPU_BUILD (build_dx12.py --model-k)")
        cls.params = random_network_params(11)
        cls.mdir = os.path.join(cls.build, "k_runtime_d3d12_test")
        cls.wgcm, cls.map = write_k_model(cls.mdir, cls.params)

    def sequence(self, backend, hdr, mode, half, frames, options="", seed=0, out=OUT, max_render=MAX_RENDER):
        """The frames [(rw, rh, reset, E, pi)] through HIP and `backend`; raises on the first difference."""
        auto = mode == 2
        hip = KRuntimeExt(self.build, self.mdir, self.map, out, max_render, hdr, False, auto, options, backend=0)
        dx = KRuntimeExt(self.build, self.mdir, self.map, out, max_render, hdr, False, auto, options, backend=backend)
        pos = os.path.getsize(TRACE) if os.path.exists(TRACE) else 0
        on_dx, on_hip = {}, {}
        try:
            for i, (rw, rh, reset, E, pi) in enumerate(frames):
                x = inputs(np.random.default_rng(seed * 100 + i), rw, rh, hdr, reset)
                x["pre_exposure"], x["E_cur"] = pi, float(np.float32(E))
                if i == 2:
                    x["mv_scale"], x["mv_offset"] = (0.75, 1.25), (0.125, -0.25)
                (o_h, a_h), _ = hip.frame(x, mode, half=half)
                _, pos = trace_since(pos)
                (o_d, a_d), _ = dx.frame(x, mode, half=half)
                lines, pos = trace_since(pos)
                for ln in lines:
                    parts = ln.split()
                    if parts:
                        tgt = on_dx if parts[-1] == "dx12" else on_hip
                        tgt[parts[0]] = tgt.get(parts[0], 0) + 1
                np.testing.assert_array_equal(o_d.view(np.uint32), o_h.view(np.uint32), err_msg=f"frame {i}: output")
                if half:
                    np.testing.assert_array_equal(a_d.view(np.uint32), a_h.view(np.uint32), err_msg=f"frame {i}: alpha")
                for name in BUFFERS:
                    try:
                        want = hip.buffer(name, np.uint8)
                    except RuntimeError:
                        continue   # not in this configuration
                    np.testing.assert_array_equal(dx.buffer(name, np.uint8), want, err_msg=f"frame {i}: {name}")
        finally:
            del hip, dx
        return on_dx, on_hip

    FRAMES = [(130, 78, True, 1.0, 1.0), (130, 78, False, 1.0, 1.0), (130, 78, False, 1.25, 0.5),
              (100, 60, False, 1.0, 1.0), (100, 60, True, 0.75, 1.0)]

    def check_hybrid(self, hdr, mode, half, options="", frames=None, **size):
        on_dx, on_hip = self.sequence(2, hdr, mode, half, frames or self.FRAMES, options, **size)
        want = STAGES + ALWAYS + ["k_out_color"] + (["exposure_auto_diet"] if mode == 2 and hdr else [])
        if size.get("out", OUT)[0] >= 3840:   # above 24 windows e4 and d4 take their NG = 2 kernels
            want = [{"k_stagew4_e4": "k_stagew2_e4", "k_stagew4_d4": "k_stagew2_d4"}.get(k, k) for k in want]
        if "k_head=full" in options:   # the head map and k_out_pixel instead of the compact pixel head
            want = [{"k_stage2_d0_hp": "k_stage2_d0_hidden", "k_out_pixel_c": "k_out_pixel"}.get(k, k) for k in want]
        never = [k for k in want if not on_dx.get(k)]
        self.assertFalse(never, f"never ran on D3D12 in the hybrid device: {never}; D3D12: {sorted(on_dx)}")
        # the tiled input stage: the form with the host's frame values (k_in_uniforms default), else the GPU's own
        self.assertTrue(any(on_dx.get(k) for k in INPUTS), f"no tiled input kernel ran on D3D12: {sorted(on_dx)}")
        k_on_hip = sorted(k for k in on_hip if k.startswith(("k_", "kx/")))
        self.assertFalse(k_on_hip, f"model K kernels that ran on HIP in the hybrid device: {k_on_hip}")

    def test_hybrid_hdr_value_rows(self):
        self.check_hybrid(True, 0, True)

    def test_hybrid_hdr_texel_dense(self):
        self.check_hybrid(True, 1, False)

    def test_hybrid_hdr_auto_exposure(self):
        self.check_hybrid(True, 2, True)

    def test_hybrid_ldr_rows(self):
        self.check_hybrid(False, 0, True)

    def test_hybrid_head_full(self):
        self.check_hybrid(True, 0, True, "k_head=full")

    def test_hybrid_input_without_host_uniforms(self):
        """k_in_uniforms=0: the tiled input kernel that computes its frame values on the GPU."""
        on_dx, _ = self.sequence(2, True, 0, True, self.FRAMES, "k_in_uniforms=0")
        self.assertTrue(on_dx.get("kx/k_in_tiled"), sorted(on_dx))
        self.assertFalse(on_dx.get("kx/k_in_tiled_uniform"))

    # the sizes of a game: 1440p Quality (2560 x 1440 from 1707 x 960) and 4K Performance (3840 x 2160 from 1920 x 1080,
    # where e4 / d4 run their NG = 2 kernels); a reset frame, a steady frame with a pan, and a smaller render size
    QUALITY = dict(out=(2560, 1440), max_render=(1707, 960))
    FRAMES_Q = [(1707, 960, True, 1.0, 1.0), (1707, 960, False, 1.0, 1.0), (1500, 840, False, 1.0, 1.0)]

    def test_hybrid_1440p_quality(self):
        self.check_hybrid(True, 0, True, frames=self.FRAMES_Q, **self.QUALITY)

    def test_hybrid_4k_performance(self):
        self.check_hybrid(True, 0, True, frames=[(1920, 1080, True, 1.0, 1.0), (1920, 1080, False, 1.0, 1.0)],
                          out=(3840, 2160), max_render=(1920, 1080))

    def test_native_1440p_quality(self):
        self.sequence(3, True, 0, True, self.FRAMES_Q, **self.QUALITY)

    def test_native_hdr_value_rows(self):
        self.sequence(3, True, 0, True, self.FRAMES)

    def test_native_hdr_auto_exposure_dense(self):
        self.sequence(3, True, 2, False, self.FRAMES)

    def test_native_ldr_rows(self):
        self.sequence(3, False, 0, True, self.FRAMES)


if __name__ == "__main__":
    unittest.main()
