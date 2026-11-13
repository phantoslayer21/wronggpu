# SPDX-License-Identifier: MIT
"""Build a small host program from repository sources for a CPU test: the host C++ compiler (CXX, clang++ or g++; on
Windows MSVC cl after them, from PATH or through vcvars64.bat). Raises unittest.SkipTest without a compiler."""
import os
import shutil
import subprocess
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _vcvars64():
    vswhere = os.path.join(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)"),
                           "Microsoft Visual Studio", "Installer", "vswhere.exe")
    if not os.path.exists(vswhere):
        return None
    r = subprocess.run([vswhere, "-latest", "-products", "*", "-property", "installationPath"], capture_output=True,
                       text=True)
    vcvars = os.path.join(r.stdout.strip(), "VC", "Auxiliary", "Build", "vcvars64.bat")
    return vcvars if r.returncode == 0 and r.stdout.strip() and os.path.exists(vcvars) else None


def build(name, sources, defines=()):
    """sources: paths relative to the repository root; defines: NAME or NAME=VALUE. Returns the program's path."""
    tmp = tempfile.mkdtemp(prefix="wg_%s_" % name)
    exe = os.path.join(tmp, name + (".exe" if os.name == "nt" else ""))
    srcs = [os.path.join(ROOT, s) for s in sources]
    cxx = os.environ.get("CXX") or shutil.which("clang++") or shutil.which("g++")
    if cxx:
        cmd = [cxx, "-std=c++17", "-O1"] + ["-D" + d for d in defines]
        cmd += (["-D_CRT_SECURE_NO_WARNINGS"] if os.name == "nt" else []) + srcs + ["-o", exe]
        r = subprocess.run(cmd + ([] if os.name == "nt" else ["-ldl", "-pthread"]), capture_output=True, text=True)
    elif os.name == "nt" and (shutil.which("cl") or _vcvars64()):
        vcvars = None if shutil.which("cl") else _vcvars64()
        objdir = os.path.join(tmp, "obj")
        os.makedirs(objdir)
        cl = ('cl /nologo /std:c++17 /O1 /EHsc /utf-8 /D_CRT_SECURE_NO_WARNINGS %s %s /Fo"%s\\\\" /Fe"%s"'
              % (" ".join("/D" + d for d in defines), " ".join('"%s"' % s for s in srcs), objdir, exe))
        bat = os.path.join(tmp, "build.bat")
        with open(bat, "w") as f:
            f.write("@echo off\r\n" + ('call "%s" >nul || exit /b 1\r\n' % vcvars if vcvars else "") +
                    cl + " || exit /b 1\r\n")
        r = subprocess.run(["cmd", "/d", "/c", bat], capture_output=True, text=True)
    else:
        raise unittest.SkipTest("no host C++ compiler (CXX, clang++, g++ or MSVC cl)")
    if r.returncode != 0:
        raise RuntimeError("build of %s failed:\n%s%s" % (name, r.stdout, r.stderr))
    return exe
