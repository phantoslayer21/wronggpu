# SPDX-License-Identifier: MIT
"""The runtime's model selection (#89): clean/runtime/model_select.h and model_from_options in both runtime builds,
with model K (clean/k/runtime/model_k.cpp) and without (clean/runtime/no_model_k.cpp, -DWG_NO_MODEL_K).

No GPU: clean/tests/test_model_select.cpp is built twice with the host C++ compiler (CXX, clang++ or g++; on Windows
MSVC cl after them, from PATH or through vcvars64.bat; skipped without one) and run. It checks the model table and the
exact messages of the runtime option model=<name> (Model M and model K as before; model=L refused by name).

Run: python -m unittest clean.tests.test_model_select
"""
import os
import shutil
import subprocess
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TEST = os.path.join("clean", "tests", "test_model_select.cpp")
BUILDS = {"with model K": ([TEST, os.path.join("clean", "k", "runtime", "model_k.cpp")], []),
          "no model K": ([TEST, os.path.join("clean", "runtime", "no_model_k.cpp")], ["WG_NO_MODEL_K"])}


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


def _build(exe, sources, defines):
    cxx = os.environ.get("CXX") or shutil.which("clang++") or shutil.which("g++")
    if cxx:
        cmd = [cxx, "-std=c++17", "-O1"] + (["-D_CRT_SECURE_NO_WARNINGS"] if os.name == "nt" else [])
        cmd += ["-D" + d for d in defines] + [os.path.join(ROOT, s) for s in sources] + ["-o", exe]
        return subprocess.run(cmd, capture_output=True, text=True)
    vcvars = None if shutil.which("cl") else _vcvars64()
    if os.name != "nt" or not (shutil.which("cl") or vcvars):
        raise unittest.SkipTest("no host C++ compiler (CXX, clang++, g++ or MSVC cl)")
    objdir = os.path.join(os.path.dirname(exe), "obj_" + os.path.splitext(os.path.basename(exe))[0])
    os.makedirs(objdir, exist_ok=True)
    cl = ('cl /nologo /std:c++17 /O1 /EHsc /utf-8 /D_CRT_SECURE_NO_WARNINGS %s %s /Fo"%s\\\\" /Fe"%s"'
          % (" ".join("/D" + d for d in defines), " ".join('"%s"' % os.path.join(ROOT, s) for s in sources), objdir,
             exe))
    bat = os.path.join(os.path.dirname(exe), os.path.basename(exe) + ".bat")
    with open(bat, "w") as f:
        f.write("@echo off\r\n")
        if vcvars:
            f.write('call "%s" >nul || exit /b 1\r\n' % vcvars)
        f.write(cl + " || exit /b 1\r\n")
    return subprocess.run(["cmd", "/d", "/c", bat], capture_output=True, text=True)


class ModelSelect(unittest.TestCase):
    def test_both_builds(self):
        tmp = tempfile.mkdtemp(prefix="wg_ms_")
        for i, (label, (sources, defines)) in enumerate(BUILDS.items()):
            with self.subTest(build=label):
                exe = os.path.join(tmp, "test_model_select_%d%s" % (i, ".exe" if os.name == "nt" else ""))
                r = _build(exe, sources, defines)
                self.assertEqual(r.returncode, 0, "build failed:\n" + r.stdout + r.stderr)
                env = dict(os.environ)
                env.pop("WG_RUNTIME_OPTIONS", None)
                r = subprocess.run([exe], capture_output=True, text=True, env=env)
                self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
                self.assertIn("test_model_select (%s): 0 failure(s)" % label, r.stdout)


if __name__ == "__main__":
    unittest.main()
