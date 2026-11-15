# SPDX-License-Identifier: MIT
"""The HIP SDK of the builds (clean/runner/sdk.py): AMD's official SDK folder, never one from the environment. No
compiler and no GPU: argument parsing, the refusal of another --rocm, clang lookup without PATH, the build log banner,
and a scan that no Python file of clean/ takes an SDK from ROCM_PATH."""
import argparse
import os
import re
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)

from clean.runner import build, sdk  # noqa: E402


class SdkTest(unittest.TestCase):
    def parse(self, argv):
        ap = argparse.ArgumentParser()
        sdk.add_arguments(ap)
        return ap.parse_args(argv)

    def test_default_is_the_official_sdk_whatever_the_environment(self):
        with patch.dict(os.environ, {"ROCM_PATH": "C:\\elsewhere", "HIP_PATH": "C:\\elsewhere",
                                     "LLVM_PATH": "C:\\elsewhere", "HIP_DEVICE_LIB_PATH": "C:\\elsewhere"}):
            a = self.parse([])
            self.assertEqual(sdk.resolve(a.rocm, a.force_sdk), sdk.OFFICIAL_ROCM)

    def test_another_sdk_needs_force(self):
        a = self.parse(["--rocm", "C:\\elsewhere"])
        with self.assertRaises(SystemExit):
            sdk.resolve(a.rocm, a.force_sdk)
        a = self.parse(["--rocm", "C:\\elsewhere", "--force-sdk"])
        self.assertEqual(sdk.resolve(a.rocm, a.force_sdk), "C:\\elsewhere")
        self.assertEqual(sdk.resolve(sdk.OFFICIAL_ROCM.upper() if sys.platform == "win32" else sdk.OFFICIAL_ROCM),
                         sdk.OFFICIAL_ROCM.upper() if sys.platform == "win32" else sdk.OFFICIAL_ROCM)

    def test_clang_never_from_path(self):
        with tempfile.TemporaryDirectory() as empty, patch.object(build.shutil, "which", lambda *a: "C:\\elsewhere\\clang"):
            with self.assertRaises(SystemExit):
                build.find_clang(empty)
            if sys.platform == "win32":
                with self.assertRaises(SystemExit):
                    build.find_host_cxx(empty)

    def test_clang_from_the_sdk(self):
        with tempfile.TemporaryDirectory() as root:
            exe = "clang.exe" if sys.platform == "win32" else "clang"
            os.makedirs(os.path.join(root, "bin"))
            open(os.path.join(root, "bin", exe), "w").close()
            self.assertEqual(build.find_clang(root), os.path.join(root, "bin", exe))

    def test_banner(self):
        line = sdk.banner(sdk.OFFICIAL_ROCM, clang="c", device_libs=True)
        self.assertIn(f"HIP SDK: {sdk.OFFICIAL_ROCM} (official)", line)
        self.assertIn("device libs: " + os.path.join(sdk.OFFICIAL_ROCM, "amdgcn", "bitcode"), line)
        self.assertIn("NOT the official SDK", sdk.banner("C:\\elsewhere"))

    def test_build_py_ignores_rocm_path(self):
        """build.py with no --rocm compiles with the official SDK's device libraries, ROCM_PATH set elsewhere."""
        cmds = []
        with tempfile.TemporaryDirectory() as out, patch.dict(os.environ, {"ROCM_PATH": "C:\\elsewhere"}), \
                patch.object(build, "run", lambda cmd, *a, **k: cmds.append(cmd)), \
                patch.object(build, "find_clang", lambda rocm=None: "clang"), \
                patch.object(build, "build_runtime", lambda *a, **k: None), \
                patch.object(build, "run_post_w64", lambda cmd, *a, **k: cmds.append(cmd)), \
                patch.object(build, "find_host_cxx", lambda rocm=None: "cxx"):
            try:
                build.main(["--out", out, "--no-model-k", "--no-resource-gate"])
            except (SystemExit, Exception):
                pass
        dev = [c for c in cmds if any(str(a).startswith("--rocm-device-lib-path=") for a in c)]
        self.assertTrue(dev, "no device compile recorded")
        for c in dev:
            self.assertIn("--rocm-device-lib-path=" + sdk.device_lib_dir(sdk.OFFICIAL_ROCM), c)
            self.assertFalse(any("elsewhere" in str(a) for a in c))

    def test_no_rocm_path_defaults_left(self):
        files = subprocess.run(["git", "ls-files", "clean/*.py", "clean/**/*.py"], cwd=ROOT, capture_output=True,
                               text=True).stdout.split()
        pat = re.compile(r"os\.environ\.get\(\s*[\"']ROCM_PATH[\"']")
        hits = []
        for f in files:
            with open(os.path.join(ROOT, f), encoding="utf-8") as fh:
                if pat.search(fh.read()):
                    hits.append(f)
        self.assertEqual(hits, [])


if __name__ == "__main__":
    unittest.main()
