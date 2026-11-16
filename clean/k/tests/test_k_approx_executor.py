# SPDX-License-Identifier: MIT
"""GPU/D3D12: the executor prepares only the kernels model K selected, so an opt-in form that is not selected cannot
affect a default context.

A copy of the runtime build with one byte changed in k_out_pixel_c__a2.dxil and k_out_pixel_c_direct__a2.dxil (the
k_approx=a2 output form; the executor runs the direct one) runs
test_interop_d3d12.exe's `matrix` case at 1080p (every usual upscaler-mode ratio, HDR and LDR, both declared maximum
sizes): with nothing set every context is taken by the D3D12 executor and its frame runs (the executor's warm-up does not
compile the unselected form) and its mode line names the exact path; with k_approx=a2 every context is refused, naming
the shader. The unchanged build passes with k_approx=a2 (control: the refusal comes from the changed byte) and its mode
line names k_approx=a2 and an approximation.

Set WG_K_GPU_BUILD (a runtime build with dx12/) and WG_K_INTEROP_EXE (test_interop_d3d12.exe); run under the shared GPU
lock. Only synthetic parameters (random_network_params).

Run: python -m unittest clean.k.tests.test_k_approx_executor
"""
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

from clean.k.reference.network import random_network_params
from clean.k.tests.test_k_runtime_gpu import gpu_build, write_k_model

# the k_approx=a2 output form: the buffer form and the direct form the executor runs (textures)
VICTIMS = ("k_out_pixel_c__a2", "k_out_pixel_c_direct__a2")


def copy_build(src, dst):
    """The runtime build's files and its dx12 folder (not its test model folders)."""
    for f in os.listdir(src):
        p = os.path.join(src, f)
        if os.path.isfile(p):
            shutil.copy2(p, dst)
    shutil.copytree(os.path.join(src, "dx12"), os.path.join(dst, "dx12"))


def mode_lines(out):
    return [ln for ln in out.splitlines() if "network model K mode:" in ln]


class TestKApproxExecutor(unittest.TestCase):
    def setUp(self):
        self.exe = os.environ.get("WG_K_INTEROP_EXE")
        if not self.exe:
            self.skipTest("set WG_K_INTEROP_EXE to the clean native interop test")
        self.build = Path(gpu_build()).resolve()
        for v in VICTIMS:
            if not (self.build / "dx12" / (v + ".dxil")).exists():
                self.skipTest("no %s.dxil in WG_K_GPU_BUILD (build_dx12.py --model-k)" % v)

    def matrix(self, build, model_dir, model, modelmap, extra):
        opts = ";".join(["model=K", f"kernels_dir={build}", f"model_dir={model_dir}",
                         f"model_k_dir={Path(model).parent}", f"model_k_map={modelmap}"] + extra)
        env = os.environ.copy()
        env.pop("WG_RUNTIME_OPTIONS", None)
        p = subprocess.run([self.exe, str(Path(build) / "wronggpu_runtime.dll"), "matrix", opts, "1080"],
                           capture_output=True, text=True, timeout=900, env=env)
        return p.returncode, p.stdout + p.stderr

    def run_case(self, corrupt, extra):
        """One matrix run on the build (corrupt: on a copy with one byte of the victim shader changed)."""
        with tempfile.TemporaryDirectory(prefix="k-approx-exe-") as td:
            model, modelmap = write_k_model(os.path.join(td, "model"), random_network_params(11))
            build = str(self.build)
            if corrupt:
                build = os.path.join(td, "build")
                os.makedirs(build)
                copy_build(str(self.build), build)
                for v in VICTIMS:
                    path = os.path.join(build, "dx12", v + ".dxil")
                    data = bytearray(Path(path).read_bytes())
                    data[len(data) // 2] ^= 0x40
                    Path(path).write_bytes(bytes(data))
            return self.matrix(build, td, model, modelmap, extra)

    def test_selected_form_runs(self):
        """Control: the unchanged build takes k_approx=a2 in every context; the mode log names the approximation."""
        rc, out = self.run_case(False, ["k_approx=a2"])
        self.assertEqual(rc, 0, out)
        self.assertTrue(out.rstrip().endswith("PASS"), out)
        modes = mode_lines(out)
        self.assertEqual(len(modes), 20, modes)
        self.assertTrue(all(";k_approx=a2;" in m and "an approximation (not bit-exact)" in m for m in modes), modes)

    def test_unselected_form_is_not_prepared(self):
        """The changed shader, nothing set: every context runs on the exact path (the warm-up skips the form)."""
        rc, out = self.run_case(True, [])
        self.assertEqual(rc, 0, out)
        self.assertTrue(out.rstrip().endswith("PASS"), out)
        self.assertNotIn("integrity mismatch", out)
        modes = mode_lines(out)
        self.assertEqual(len(modes), 20, modes)
        self.assertTrue(all("k_approx" not in m and "the exact path" in m for m in modes), modes)

    def test_selected_changed_form_is_refused(self):
        """The changed shader, k_approx=a2: every context is refused, naming the shader."""
        rc, out = self.run_case(True, ["k_approx=a2"])
        self.assertNotEqual(rc, 0, out)
        self.assertEqual(out.count("could not take this context"), 20, out[-3000:])
        self.assertIn("integrity mismatch: " + VICTIMS[1], out)
        self.assertNotIn("ready dx12 1", out)

if __name__ == "__main__":
    unittest.main()
