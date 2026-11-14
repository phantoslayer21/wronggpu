# SPDX-License-Identifier: MIT
"""CPU test of the RR runtime's retire list (clean/rr_f/runtime/src/rr_retire.h, issue #63): an object that a frame
recorded into the game's command list may use (buffers, the shader-visible descriptor heap, the timestamp query heap)
is released only when the frame marker proves that frame complete, never because a wait ran out; device drop releases
the rest. rr_retire_test.cpp drives the list with fake objects that count Release and a fake marker value.

The program is compiled with the first available C++17 compiler: $CXX, clang++ / g++ on PATH, MSVC cl on PATH (a
developer prompt), then MSVC through vswhere and vcvars64.bat. Skipped when there is none.

Run: python -m unittest clean.rr_f.runtime.tests.test_rr_retire
"""
import os
import shutil
import subprocess
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(os.path.dirname(HERE), "src")
TEST_CPP = os.path.join(HERE, "rr_retire_test.cpp")
EXE = ".exe" if os.name == "nt" else ""
NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


def _run(cmd, **kw):
    return subprocess.run(cmd, capture_output=True, text=True, creationflags=NO_WINDOW, **kw)


def _gcc_style(cxx):
    def build(out):
        r = _run([cxx, "-std=c++17", "-O2", "-Wall", "-I" + SRC, TEST_CPP, "-o", out])
        if r.returncode != 0:
            raise AssertionError("compile failed:\n" + r.stdout + r.stderr)
    return build


def _cl(prefix):
    def build(out):
        d = os.path.dirname(out)
        cmd = 'cl /nologo /std:c++17 /O2 /EHsc /W4 /I"%s" "%s" /Fo"%s\\\\" /Fe"%s"' % (SRC, TEST_CPP, d, out)
        bat = os.path.join(d, "build.bat")
        with open(bat, "w") as f:
            f.write("@echo off\r\n" + (prefix or "") + cmd + " || exit /b 1\r\n")
        r = _run(["cmd", "/c", bat])
        if r.returncode != 0:
            raise AssertionError("compile failed:\n" + r.stdout + r.stderr)
    return build


def find_compiler():
    cxx = os.environ.get("CXX")
    if cxx:
        return _gcc_style(cxx)
    for name in ("clang++", "g++"):
        p = shutil.which(name)
        if p:
            return _gcc_style(p)
    if os.name == "nt":
        if shutil.which("cl"):
            return _cl("")
        vswhere = r"C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe"
        if os.path.exists(vswhere):
            r = _run([vswhere, "-latest", "-property", "installationPath"])
            vcvars = os.path.join(r.stdout.strip(), "VC", "Auxiliary", "Build", "vcvars64.bat")
            if r.returncode == 0 and r.stdout.strip() and os.path.exists(vcvars):
                return _cl('call "%s" >nul || exit /b 1\r\n' % vcvars)
    return None


class TestRrRetire(unittest.TestCase):
    def test_retire_until_complete(self):
        build = find_compiler()
        if build is None:
            self.skipTest("no C++ compiler (clang++, g++ or MSVC)")
        with tempfile.TemporaryDirectory(prefix="wg_rr_retire_") as d:
            exe = os.path.join(d, "rr_retire_test" + EXE)
            build(exe)
            r = _run([exe])
            self.assertEqual(r.returncode, 0, "rr_retire_test failed:\n" + r.stdout + r.stderr)
            self.assertIn("passed", r.stdout)


if __name__ == "__main__":
    unittest.main()
