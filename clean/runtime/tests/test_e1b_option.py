# SPDX-License-Identifier: MIT
"""The runtime option e1b. An invalid value must fall back to off with one log line, never fail create (a typo in a
player's ini must not lose the upscaler). clean/runtime/tests/test_e1b_option.cpp is built with MSVC cl (on PATH, e.g. a
developer prompt or CI, or through vcvars64.bat; skipped without one) and checks the value parse, the options scan and
the report item. A source check confirms the wiring: the option is read and resolved at create, the kernel report has
no side effects, and the default warm-up never compiles an opt-in form.

Run: python -m unittest clean.runtime.tests.test_e1b_option
"""
import os
import re
import shutil
import subprocess
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
RT = os.path.join(ROOT, "clean", "runtime")


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


def _read(name):
    with open(os.path.join(RT, name), encoding="utf-8") as f:
        return f.read()


def _block(src, head):
    """The text from `head` to the end of that member function (the next line that is just two spaces and a brace)."""
    start = src.find(head)
    if start < 0:
        return ""
    end = src.find("\n  }\n", start)
    return src[start:end if end >= 0 else len(src)]


class E1bOption(unittest.TestCase):
    def test_parse_scan_report(self):
        if os.name != "nt":
            raise unittest.SkipTest("built with MSVC cl: Windows only")
        vcvars = None if shutil.which("cl") else _vcvars64()
        if not shutil.which("cl") and not vcvars:
            raise unittest.SkipTest("no MSVC cl (not on PATH, no Visual Studio found)")
        work = tempfile.mkdtemp(prefix="wg_e1b_")
        exe = os.path.join(work, "test_e1b_option.exe")
        src = os.path.join(RT, "tests", "test_e1b_option.cpp")
        bat = os.path.join(work, "build.bat")
        with open(bat, "w") as f:
            f.write("@echo off\r\n")
            if vcvars:
                f.write('call "%s" >nul || exit /b 1\r\n' % vcvars)
            f.write('cl /nologo /std:c++17 /O1 /EHsc /utf-8 /W4 /WX "%s" /Fo"%s\\\\" /Fe"%s" || exit /b 1\r\n'
                    % (src, work, exe))
        r = subprocess.run(["cmd", "/d", "/c", bat], capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, "build failed:\n" + r.stdout + r.stderr)
        r = subprocess.run([exe], capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn("e1b option: 0 failure(s)", r.stdout)
        shutil.rmtree(work, ignore_errors=True)

    def test_runtime_wiring(self):
        src = _read("runtime.cpp")
        self.assertIn('#include "e1b_option.h"', src)
        self.assertNotIn("e1b must be", src, "an invalid e1b value must not fail create")
        # read before Model K's create returns, so every model validates and reports it
        scan, k_create = src.find("e1b_scan("), src.find("return create_k(c, std::move(given));")
        self.assertTrue(0 <= scan < k_create, "e1b_scan must come before Model K's create")
        # resolved at create: create() calls e1b_resolve(), which prepares the route's kernel and alone sets e1b_on
        create = src[src.find("  bool create(const WgRtConfig& c"):src.find("  bool create_k(")]
        self.assertIn("e1b_resolve();", create)
        resolve = _block(src, "  void e1b_resolve() {")
        self.assertIn("dev->prepare(k)", resolve)
        assign = re.compile(r"^\s+e1b_on = [^;]*;", re.M)  # statements (not the declaration or a comment)
        self.assertEqual(len(assign.findall(src)), 2, "e1b_on is set only in e1b_resolve")
        self.assertEqual(len(assign.findall(resolve)), 2)
        # the report is side-effect free: kernels() reads the route resolved at create and changes nothing
        body = _block(src, "  const char* kernels() {")
        self.assertIn("e1b_report(e1b_on, e1b_reason)", body)
        for forbidden in ("e1b_on =", "e1b_reason =", "e1b_opt =", "prepare(", "manifest_kernels()"):
            self.assertNotIn(forbidden, body, "kernels() must not contain " + forbidden)
        # the launch follows the create-time route
        self.assertIn('launch(e1b_on ? (native_direct_io ? "e1b/input_encoder_direct" : "e1b/input_encoder")', src)

    def test_default_warm_up_skips_opt_in_forms(self):
        dev = _read("device_d3d12.cpp")
        warm = _block(dev, "  bool warm_up(std::string* error) override {")
        self.assertIn("optional_kernel(k.name)", warm, "the default warm-up must skip the opt-in forms")
        self.assertIn('name.compare(0, 4, "e1b/") == 0', dev)


if __name__ == "__main__":
    unittest.main()
