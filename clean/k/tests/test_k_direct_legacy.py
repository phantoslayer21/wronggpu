# SPDX-License-Identifier: MIT
"""GPU/D3D12: Model K on the native direct route, frames on the legacy input path, byte for byte against the exact
reference.

The legacy input kernels (kx/k_in_color_direct, then kx/k_in_reset or kx/k_in_base_direct) run on a frame outside the
tiled guard: a render size above 0.68 x the output (DLAA), a jitter outside [-0.5, 0.5], or input_tiled=0. They write
and read the intermediates cpre / cpre2, which a direct context must allocate like a buffer context (a null pointer is
packed as arena offset 0, clean/runtime/struct_args.h, and the frame would overwrite the arena's first allocation).
test_interop_d3d12.exe dx12same runs the same 6 frames (resets at frames 1 and 5) through executor=hip (the exact
reference on the buffer route) and executor=dx12 (the native direct route: "inputs read directly 1") and compares the
raw output texture bytes: every frame must be identical. The tiled Quality case is the control. With k_approx=a1 on the
direct context, the legacy frames (DLAA, a jitter outside the guard) run the exact input stage: still identical.

Set WG_K_GPU_BUILD (a developer runtime build: executor=hip is a diagnostic key) and WG_K_INTEROP_EXE
(test_interop_d3d12.exe); run under the shared GPU lock. Only synthetic parameters (random_network_params).

Run: python -m unittest clean.k.tests.test_k_direct_legacy
"""
import os
from pathlib import Path
import re
import subprocess
import tempfile
import unittest

from clean.k.reference.network import random_network_params
from clean.k.tests.test_k_runtime_gpu import gpu_build, write_k_model

FRAMES = 6
CASES = [  # (label, render w, h, output w, h, extra options, extra environment)
    ("DLAA", 1280, 720, 1280, 720, "", {}),
    ("jitter outside the guard, Quality", 853, 480, 1280, 720, "", {"WG_TEST_JITTER_OUT": "1"}),
    ("input_tiled=0, Quality", 853, 480, 1280, 720, "input_tiled=0", {}),
    ("control: tiled Quality", 853, 480, 1280, 720, "", {}),
    # k_approx=a1 on the direct context only (WG_TEST_DX12_OPTIONS: the HIP reference refuses k_approx): a legacy frame
    # runs the EXACT input stage (a1 applies to the tiled input stage only) and a1 changes nothing else, so every output
    # byte still equals the exact reference
    ("k_approx=a1, DLAA", 1280, 720, 1280, 720, "", {"WG_TEST_DX12_OPTIONS": "k_approx=a1"}),
    ("k_approx=a1, jitter outside the guard", 853, 480, 1280, 720, "", {"WG_TEST_JITTER_OUT": "1",
                                                                       "WG_TEST_DX12_OPTIONS": "k_approx=a1"}),
]
ROW = re.compile(r"frame (\d+): (\d+) of (\d+) bytes differ")


class TestKDirectLegacy(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.exe = os.environ.get("WG_K_INTEROP_EXE")
        if not cls.exe:
            raise unittest.SkipTest("set WG_K_INTEROP_EXE to the clean native interop test")
        cls.build = Path(gpu_build()).resolve()
        cls.tmp = tempfile.TemporaryDirectory(prefix="k-direct-legacy-")
        cls.model, cls.map = write_k_model(os.path.join(cls.tmp.name, "model"), random_network_params(11))

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def run_case(self, rw, rh, ow, oh, extra, env_extra):
        opts = ";".join(["model=K", f"kernels_dir={self.build}", f"model_dir={self.tmp.name}",
                         f"model_k_dir={Path(self.model).parent}", f"model_k_map={self.map}"] + ([extra] if extra else []))
        env = {k: v for k, v in os.environ.items() if not k.startswith("WG_") or k == "WG_K_INTEROP_EXE"}
        env.update(WG_TEST_RENDER_WIDTH=str(rw), WG_TEST_RENDER_HEIGHT=str(rh), WG_TEST_OUTPUT_WIDTH=str(ow),
                   WG_TEST_OUTPUT_HEIGHT=str(oh), **env_extra)
        p = subprocess.run([self.exe, str(self.build / "wronggpu_runtime.dll"), "dx12same", opts, str(FRAMES)],
                           capture_output=True, text=True, timeout=170, env=env)
        out = p.stdout + p.stderr
        if "executor=hip ignored: diagnostic key, release build" in out:
            self.skipTest("the exact reference (executor=hip) needs a developer build")
        return p.returncode, out

    def test_direct_legacy_frames_match_the_reference(self):
        for label, rw, rh, ow, oh, extra, env_extra in CASES:
            with self.subTest(case=label):
                rc, out = self.run_case(rw, rh, ow, oh, extra, env_extra)
                self.assertIn("inputs read directly 1", out, out[-2000:])   # the native direct route
                rows = [(int(f), int(d), int(n)) for f, d, n in ROW.findall(out)]
                print("%s: %s" % (label, ", ".join("frame %d %d/%d" % r for r in rows)))
                self.assertEqual([f for f, _, _ in rows], list(range(1, FRAMES + 1)), out[-2000:])
                self.assertEqual([d for _, d, _ in rows], [0] * FRAMES, "%s: output bytes differ from the exact reference"
                                 % label)
                self.assertNotIn("null cpre", out)
                if "k_approx=a1" in env_extra.get("WG_TEST_DX12_OPTIONS", ""):   # the option took effect: its one line
                    self.assertEqual(out.count("k_approx=a1 applies to the tiled input stage only"), 1, out[-2000:])
                self.assertEqual(rc, 0, out[-2000:])


if __name__ == "__main__":
    unittest.main()
