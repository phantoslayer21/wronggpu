# SPDX-License-Identifier: MIT
"""Bounds checks of the RR runtime's model loaders (src/rr_model.cpp: read_wgcm for model.wgcm, load_own for WGRRMOD1).
No GPU and no model data: clean/rr_f/runtime/tests/test_model_bounds.cpp is built with MSVC cl (on PATH, e.g. a developer
prompt or CI, or through vcvars64.bat; skipped without one) together with src/rr_model.cpp (a stub stands in for the
packer of src/rr_pack.cpp, which needs the generated tables), and loads
crafted files whose whole-file SHA-256 is correct but whose offsets, sizes and shapes wrap around 2^64. Each case runs in
its own process, since a loader that reads through a wrapped range may crash.

Run: python -m unittest clean.rr_f.runtime.tests.test_model_bounds
"""
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
RT = os.path.join(ROOT, "clean", "rr_f", "runtime")
SOURCES = [os.path.join(RT, "tests", "test_model_bounds.cpp"), os.path.join(RT, "src", "rr_model.cpp")]
_EXE = None


def _vcvars64():
    """vcvars64.bat of the newest Visual Studio (vswhere), or None."""
    vswhere = os.path.join(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)"),
                           "Microsoft Visual Studio", "Installer", "vswhere.exe")
    if not os.path.exists(vswhere):
        return None
    r = subprocess.run([vswhere, "-latest", "-products", "*", "-property", "installationPath"], capture_output=True,
                       text=True)
    vcvars = os.path.join(r.stdout.strip(), "VC", "Auxiliary", "Build", "vcvars64.bat")
    return vcvars if r.returncode == 0 and r.stdout.strip() and os.path.exists(vcvars) else None


def bounds_exe():
    global _EXE
    if _EXE is None:
        if os.name != "nt":
            raise unittest.SkipTest("the loaders use Windows (bcrypt): Windows only")
        vcvars = None if shutil.which("cl") else _vcvars64()
        if not shutil.which("cl") and not vcvars:
            raise unittest.SkipTest("no MSVC cl (not on PATH, no Visual Studio found)")
        work = tempfile.mkdtemp(prefix="wg_mb_")
        exe = os.path.join(work, "test_model_bounds.exe")
        cl = ('cl /nologo /std:c++17 /O1 /EHsc /utf-8 /D_CRT_SECURE_NO_WARNINGS %s /Fo"%s\\\\" /Fe"%s"'
              % (" ".join('"%s"' % s for s in SOURCES), work, exe))
        bat = os.path.join(work, "build.bat")
        with open(bat, "w") as f:
            f.write("@echo off\r\n")
            if vcvars:
                f.write('call "%s" >nul || exit /b 1\r\n' % vcvars)
            f.write(cl + " || exit /b 1\r\n")
        r = subprocess.run(["cmd", "/d", "/c", bat], capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError("build of test_model_bounds failed:\n" + r.stdout + r.stderr)
        _EXE = exe
    return _EXE


class ModelBounds(unittest.TestCase):
    def run_case(self, name):
        scratch = tempfile.mkdtemp(prefix="wg_mb_")
        r = subprocess.run([bounds_exe(), name, scratch], capture_output=True, text=True, timeout=120)
        self.assertEqual(r.returncode, 0, "case %s ended abnormally (%s): %s" % (name, r.returncode, r.stdout + r.stderr))
        self.assertEqual(r.stdout.strip(), "OK", "case %s: %s" % (name, r.stdout.strip()))

    # model.wgcm (read_wgcm)
    def test_wgcm_valid_file_still_loads(self):
        self.run_case("wgcm_valid")

    def test_wgcm_offset_plus_size_wraps(self):
        self.run_case("wgcm_offset_wraps")

    def test_wgcm_size_from_a_huge_shape_and_offset_above_2_33(self):
        self.run_case("wgcm_size_past_end_after_wrap")

    def test_wgcm_shape_element_count_wraps(self):
        self.run_case("wgcm_shape_wraps")

    # WGRRMOD1 (load_own, through load_model)
    def test_own_valid_file_still_loads(self):
        self.run_case("own_valid")

    def test_own_blob_range_wraps(self):
        self.run_case("own_blob_wraps")

    def test_own_image_start_behind_the_end_of_the_file(self):
        self.run_case("own_image_start_past_end")


if __name__ == "__main__":
    unittest.main()
