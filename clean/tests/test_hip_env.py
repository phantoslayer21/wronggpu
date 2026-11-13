# SPDX-License-Identifier: MIT
"""Environment cleanup of clean/runner/hip.py for HIP / clang processes (no GPU)."""
import os
import unittest
from unittest import mock

from clean.runner import hip


class HipEnvironmentTest(unittest.TestCase):
    ENV = {"PATH": "p", "TMP": "t", "LLVM_PATH": "x", "HIP_PATH": "x", "HIP_PATH_72": "keep", "HIP_PLATFORM": "amd",
           "ROCM_PATH": "x", "HSA_OVERRIDE_GFX_VERSION": "x", "HIP_DEVICE_LIB_PATH": "x", "HIP_CLANG_PATH": "x",
           "my_llvm_dir": "x", "Rocm_Home": "x", "SHAPE": "s"}

    def test_child_environment(self):
        out, removed = hip.hip_child_environment(self.ENV, "sdk-dir")
        self.assertEqual(out, {"PATH": "p", "TMP": "t", "HIP_PATH_72": "keep", "HIP_PLATFORM": "amd", "SHAPE": "s",
                               "ROCM_PATH": "sdk-dir"})
        self.assertEqual(set(removed), {"LLVM_PATH", "HIP_PATH", "ROCM_PATH", "HSA_OVERRIDE_GFX_VERSION",
                                        "HIP_DEVICE_LIB_PATH", "HIP_CLANG_PATH", "my_llvm_dir", "Rocm_Home"})

    def test_default_sdk(self):
        out, _ = hip.hip_child_environment({}, None)
        self.assertEqual(out, {"ROCM_PATH": hip.DEFAULT_ROCM})
        self.assertTrue(hip.DEFAULT_ROCM.endswith(os.path.join("ROCm", "7.2")) or hip.DEFAULT_ROCM.endswith("7.2"))

    def test_process_environment(self):
        with mock.patch.dict(os.environ, self.ENV, clear=True):
            removed = hip.clean_process_environment("sdk-dir")
            self.assertNotIn("LLVM_PATH", os.environ)
            self.assertNotIn("HSA_OVERRIDE_GFX_VERSION", os.environ)
            self.assertEqual(os.environ["ROCM_PATH"], "sdk-dir")
            self.assertEqual(os.environ["HIP_PATH_72"], "keep")
            self.assertEqual(os.environ["PATH"], "p")
            self.assertIn("LLVM_PATH", removed)


if __name__ == "__main__":
    unittest.main()
