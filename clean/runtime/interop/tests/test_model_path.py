# SPDX-License-Identifier: MIT
"""The model file each model resolves to on every executor (model_path_check.cpp, model_select.h model_wgcm_path):
model=L -> <model_dir>\\L\\model.wgcm, model=M unchanged (<model_dir>\\model.wgcm). CPU only."""
import os
import subprocess
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
sys.path.insert(0, os.path.join(ROOT, "clean", "tests"))
import host_cxx  # noqa: E402


class TestModelPath(unittest.TestCase):
    def test_paths(self):
        exe = host_cxx.build("model_path_check", ["clean/runtime/interop/tests/model_path_check.cpp"])
        r = subprocess.run([exe], capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn("pass model=L: C:\\Games\\X\\bin\\x64\\WrongGPU\\model\\L\\model.wgcm", r.stdout)
        self.assertIn("pass model=M: C:\\Games\\X\\bin\\x64\\WrongGPU\\model\\model.wgcm", r.stdout)


if __name__ == "__main__":
    unittest.main()
