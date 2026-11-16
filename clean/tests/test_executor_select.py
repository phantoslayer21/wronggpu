# SPDX-License-Identifier: MIT
"""R1: which executor runs a context, and who is an RDNA 4 GPU (clean/runtime/interop/executor_select.h, clean/runtime/rdna4.h).

No GPU: clean/runtime/interop/tests/test_executor_select.cpp is built with the host C++ compiler (CXX, clang++ or g++; on
Windows MSVC cl after them, from PATH or through vcvars64.bat; skipped without one) and run. It covers the identity rule (the
kernel-mode driver's vendor and device id, nothing else, failing closed), the lookup with injectable sources (the game
device's LUID selects the adapter among several, a spoofed name on an RX 9000 id, a supported-looking name on another id,
a failing driver query or DXGI), the executor choice for every model, the diagnostics, the sparkle guard and the refusal
paths.

The create path itself needs a D3D12 device, so its ORDER is pinned from the source here: Context::create answers
WG_ERROR_UNSUPPORTED for an adapter that is not RDNA 4 (or not identified) before any converter, kernel, HIP or D3D12
executor is made, a D3D12 refusal of any model is the same answer and never builds the HIP executor, and the RR runtime
and the native device identify the adapter by the shared id rule before anything is created.

Run: python -m unittest clean.tests.test_executor_select
"""
import os
import re
import shutil
import subprocess
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TEST = os.path.join("clean", "runtime", "interop", "tests", "test_executor_select.cpp")


def read(*p):
    with open(os.path.join(ROOT, *p), encoding="utf-8") as f:
        return f.read()


def _vcvars64():
    vswhere = os.path.join(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)"),
                           "Microsoft Visual Studio", "Installer", "vswhere.exe")
    if not os.path.exists(vswhere):
        return None
    r = subprocess.run([vswhere, "-latest", "-products", "*", "-property", "installationPath"], capture_output=True,
                       text=True)
    vcvars = os.path.join(r.stdout.strip(), "VC", "Auxiliary", "Build", "vcvars64.bat")
    return vcvars if r.returncode == 0 and r.stdout.strip() and os.path.exists(vcvars) else None


def _build(exe, defines=()):
    cxx = os.environ.get("CXX") or shutil.which("clang++") or shutil.which("g++")
    if cxx:
        cmd = [cxx, "-std=c++17", "-O1"] + ["-D" + d for d in defines] + [os.path.join(ROOT, TEST), "-o", exe]
        return subprocess.run(cmd, capture_output=True, text=True)
    vcvars = None if shutil.which("cl") else _vcvars64()
    if os.name != "nt" or not (shutil.which("cl") or vcvars):
        raise unittest.SkipTest("no host C++ compiler (CXX, clang++, g++ or MSVC cl)")
    objdir = os.path.join(os.path.dirname(exe), "obj")
    os.makedirs(objdir, exist_ok=True)
    cl = 'cl /nologo /std:c++17 /permissive- /O1 /EHsc /utf-8 /D_CRT_SECURE_NO_WARNINGS %s "%s" /Fo"%s\\\\" /Fe"%s"' % (
        " ".join("/D" + d for d in defines), os.path.join(ROOT, TEST), objdir, exe)
    bat = exe + ".bat"
    with open(bat, "w") as f:
        f.write("@echo off\r\n")
        if vcvars:
            f.write('call "%s" >nul || exit /b 1\r\n' % vcvars)
        f.write(cl + " || exit /b 1\r\n")
    return subprocess.run(["cmd", "/d", "/c", bat], capture_output=True, text=True)


class ExecutorSelect(unittest.TestCase):
    def _run(self, defines, expect_line):
        tmp = tempfile.mkdtemp(prefix="wg_es_")
        exe = os.path.join(tmp, "test_executor_select" + (".exe" if os.name == "nt" else ""))
        r = _build(exe, defines)
        self.assertEqual(r.returncode, 0, "build failed:\n" + r.stdout + r.stderr)
        r = subprocess.run([exe], capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, "\n".join(l for l in r.stdout.splitlines() if l.startswith("FAIL")) + r.stderr)
        self.assertIn("ALL TESTS PASSED", r.stdout)
        self.assertIn(expect_line, r.stdout)
        self.assertGreater(r.stdout.count("pass "), 100)

    def test_cpp_cases_release_build(self):
        # the shipped configuration (no WG_DEV_DIAGNOSTICS); both configurations of the selector are checked in the binary
        self._run((), "this test binary is a release build: kDevDiagnostics is off")

    def test_cpp_cases_developer_build(self):
        # the same cases with the developer define on: the build's own default is then the developer one
        self._run(("WG_DEV_DIAGNOSTICS=1",), "this test binary is a developer build: kDevDiagnostics is on")


class CreateOrder(unittest.TestCase):
    """The order of Context::create (backend.cpp): a pin on the source, as the create needs a D3D12 device."""

    def setUp(self):
        self.b = read("clean", "runtime", "interop", "backend.cpp")

    def at(self, text, start=0):
        i = self.b.find(text, start)
        self.assertGreaterEqual(i, 0, text)
        return i

    def test_unsupported_adapter_before_anything_is_made(self):
        refuse = self.at('if (ch.executor == "fsr") return WG_ERROR_UNSUPPORTED;')
        for later in ("conv_.init(", "configure_hip(", "make_hip_executor(", "make_d3d12_executor(", "make_shared_objects(",
                      "make_test_executor("):
            self.assertLess(refuse, self.at(later, refuse), later + " comes before the unsupported answer")
        # the decision itself is made from the game device's LUID: the system DXGI and the driver's ids (all three models:
        # the choice does not look at the model, only at the identity)
        self.assertIn("adapter_info(device_, &q.adapter, &q.vendor_id, &q.software, &ident)", self.b)
        self.assertIn("wgrt::read_adapter_identity(device->GetAdapterLuid())", self.b)

    def test_d3d12_refusal_is_the_front_end_fallback_for_every_model(self):
        self.assertIn("return fail(WG_ERROR_UNSUPPORTED, dx12_refused_line(dx12_default_, err));", self.b)
        self.assertNotIn("dx12_refused_line_k", self.b)
        self.assertNotIn("the HIP executor runs it", read("clean", "runtime", "interop", "executor_select.h"))
        # the HIP executor is made in one place, at create, for the explicit diagnostics (executor=hip, the frame capture); there is
        # no switch to it mid-session (no switch_to_hip, no hip_switch policy: the option is ignored with one line)
        self.assertNotIn("switch_to_hip", self.b)
        self.assertNotIn("refused_frame_policy", self.b)
        self.assertEqual(self.b.count("make_hip_executor("), 1)
        branch = self.at("if (!dx12 && !exec_) {")
        first = self.at("make_hip_executor(")
        self.assertLess(branch, first)
        refusal = self.at("dx12_refused_line(dx12_default_, err)")
        self.assertLess(refusal, branch)
        # the refusal returns: nothing between it and the HIP branch resets the D3D12 executor to fall through
        self.assertNotIn("exec_.reset();", self.b[refusal:branch])

    def test_hip_switch_is_ignored_once_for_every_model(self):
        # the option reaches the selector, its one line is logged once at create, and nothing in the create path is
        # conditional on the model (M, L and K are treated alike); no code reads the option to move a context to HIP
        self.assertEqual(self.b.count("q.hip_switch = o.hip_switch;"), 1)
        self.assertEqual(self.b.count("say(WG_LOG_INFO, ch.hip_note)"), 1)
        note = self.at("say(WG_LOG_INFO, ch.hip_note)")
        window = self.b[max(0, note - 200):note + 60]
        self.assertNotIn("model_k", window)
        self.assertNotIn("model_k", self.b[self.at("q.hip_switch = o.hip_switch;"):note])
        self.assertEqual(self.b.count("o.hip_switch"), 1 + 1)  # parse_options' assignment and the request
        self.assertNotIn("hip_switch_", self.b)
        sel = read("clean", "runtime", "interop", "executor_select.h")
        self.assertNotIn("kSwitchToHip", sel)

    def test_release_ignores_hip_and_capture_and_never_makes_a_substitute_image(self):
        sel = read("clean", "runtime", "interop", "executor_select.h")
        # the selector returns "hip" only for a developer build; a release build ignores executor=hip and capture_dir
        self.assertIn("kDevDiagnostics", sel)
        self.assertIn("ignored: diagnostic key, release build", sel)
        self.assertIn("bool devRequested = kDevDiagnostics", sel)
        self.assertIn("const bool dev = kDevDiagnostics && devRequested;", sel)
        # the developer texts exist only behind WG_DEV_DIAGNOSTICS (the package gate scans a release library for them)
        self.assertEqual(sel.count("#ifdef WG_DEV_DIAGNOSTICS"), 4)  # the constant, the capture text, the two routes
        # the backend logs the ignored keys and drops capture_dir in a release build, before anything is made
        drop = self.at("if (!kDevDiagnostics) o.capture_dir.clear();")
        self.assertIn("for (const std::string& line : ch.ignored) say(WG_LOG_WARN, line);", self.b[drop - 200:drop])
        self.assertGreater(drop, self.at("choose_executor(q)"))
        self.assertLess(drop, self.at("make_hip_executor("))
        # a refused frame (every model, forced dx12 too) is answered WG_ERROR_UNSUPPORTED: no policy function, no bilinear branch
        self.assertNotIn("refused_frame_action", self.b + sel)
        self.assertNotIn("kBilinear", self.b + sel)
        refused = self.at("if (refused) {")
        self.assertIn("return WG_ERROR_UNSUPPORTED;", self.b[refused:refused + 400])

    def test_hard_rule_the_hip_route_exists_only_in_a_developer_build(self):
        # R1 hard rule: a refused or unwanted D3D12 route never falls back to HIP or a reference path in a release build. The
        # selector's only way to the HIP executor ('c.executor = "hip"') is inside a WG_DEV_DIAGNOSTICS block (a release
        # library does not even contain it; the package gate scans for its texts), and the backend builds the HIP executor in
        # one place, behind that choice.
        sel = read("clean", "runtime", "interop", "executor_select.h").split("\n")
        depth, at = 0, []
        for line in sel:
            t = line.strip()
            if t.startswith("#ifdef WG_DEV_DIAGNOSTICS"):
                depth += 1
            elif t.startswith("#if") and depth:
                depth += 1
            elif t.startswith("#endif") and depth:
                depth -= 1
            if 'c.executor = "hip"' in line or 'if (e == "hip") {  // developer build only' in line:
                at.append(depth)
        self.assertEqual(len(at), 2)  # the executor=hip route (its test) and the capture route
        self.assertTrue(all(d >= 1 for d in at), "a route to HIP outside WG_DEV_DIAGNOSTICS")
        self.assertEqual(self.b.count("make_hip_executor("), 1)
        # the create refusal and the refused frame build no executor of their own and fall back to nothing but the front end
        refusal = self.at("dx12_refused_line(dx12_default_, err)")
        self.assertNotIn("make_hip_executor", self.b[refusal - 300:refusal + 300])
        self.assertNotIn("bilinear", read("clean", "runtime", "interop", "executor_select.h").lower().replace("no bilinear", ""))

    def test_sparkle_guard_is_off_and_hip_only_filters_fall_back_for_every_model(self):
        # R1-a / H4: the choice turns the guard off before anything is made, whatever the model
        off = self.at("if (ch.sparkle_off) o.sparkle_guard = 0.0f;")
        self.assertGreater(off, self.at("choose_executor(q)"))
        self.assertNotIn("model_k", self.b[self.at("choose_executor(q)"):off])
        # H3: the HIP-only filters are dropped for a context on the D3D12 executor only, before the runtime reads them
        drop = self.at("o.runtime = drop_hip_only_filter(o.runtime, &fnote);")
        self.assertEqual(self.b.count("drop_hip_only_filter("), 1)
        self.assertIn('if (ch.executor == "dx12")', self.b[drop - 200:drop])
        self.assertLess(drop, self.at("n_rtopts_ = o.runtime;"))

    def test_rr_and_native_device_use_the_shared_id_rule(self):
        rr = read("clean", "rr_f", "runtime", "src", "rr_runtime.cpp")
        self.assertIn('#include "../../../runtime/rdna4_win.h"', rr)
        self.assertNotIn("bool rdna4_adapter(", rr)
        self.assertNotIn("kmt_device_ids", rr)
        self.assertNotIn('find("RX 90', rr)
        # set-device: the identity check comes before the compute queue (nothing is created for another GPU)
        start = rr.index("rt.device_failed = true;\n  std::string err;")
        check = rr.index("wgrt::rdna4_identity(id)", start)
        self.assertLess(check, rr.index("CreateCommandQueue(&qd", start))
        self.assertIn("wgrt::needs_rx9000_sentence()", rr[check:check + 400])
        dev = read("clean", "runtime", "device_d3d12.cpp")
        self.assertNotIn("rdna4_adapter(", dev)
        self.assertEqual(dev.count("wgrt::rdna4_identity("), 2)
        self.assertNotIn('LoadLibraryExW(L"dxgi.dll"', dev)

    def test_no_name_rule_left(self):
        h = read("clean", "runtime", "rdna4.h")
        self.assertNotIn("RX 9070", re.sub(r"//[^\n]*", "", h))
        self.assertNotIn("find(", re.sub(r"//[^\n]*", "", h))
        w = read("clean", "runtime", "rdna4_win.h")
        self.assertIn("GetSystemDirectoryW", w)
        self.assertNotIn('LoadLibraryW(L"dxgi.dll")', w)


if __name__ == "__main__":
    unittest.main()
