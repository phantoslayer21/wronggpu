# SPDX-License-Identifier: MIT
"""The developer-only shortcut of the WG_DEV_GFX11 opening (clean/runtime/rdna4.h): a gfx1100 board runs the UNCHANGED gfx12
kernels through dxil-spirv's FP8 emulation. The mechanism (the environment variable WG_DEV_GFX11_SHORTCUT=1, read only in a build
made with the macro) is a placeholder for the maintainer's choice.

No GPU: clean/runtime/tests/test_shortcut_probe.cpp is built twice with the host C++ compiler (CXX, clang++ or g++; skipped without
one), without and with -DWG_DEV_GFX11, and both builds must pass (the variable, the probe tolerance, the rounding probe). The
device's wiring cannot run on a host and is pinned from the source: the variable never reaches a normal build, the rounding probe
runs at create in shortcut mode, its kernel is made only by the gfx11 build (so the gfx12 kernel set keeps its 46 files), and no
release path mentions the variable.

Run: python -m unittest clean.tests.test_dev_gfx11_shortcut
"""
import os
import re
import shutil
import subprocess
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TEST = os.path.join(ROOT, "clean", "runtime", "tests", "test_shortcut_probe.cpp")


def read(*p):
    with open(os.path.join(ROOT, *p), encoding="utf-8") as f:
        return f.read()


class HostBuilds(unittest.TestCase):
    def _run(self, defines):
        cxx = os.environ.get("CXX") or shutil.which("clang++") or shutil.which("g++")
        if not cxx:
            self.skipTest("no host C++ compiler (CXX, clang++ or g++)")
        with tempfile.TemporaryDirectory(prefix="wg_sc_") as tmp:
            exe = os.path.join(tmp, "test_shortcut_probe")
            r = subprocess.run([cxx, "-std=c++17", "-O1", *["-D" + d for d in defines], TEST, "-o", exe],
                               capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, "build failed:\n" + r.stdout + r.stderr)
            r = subprocess.run([exe], capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, "\n".join(l for l in r.stdout.splitlines() if l.startswith("FAIL")) + r.stderr)
            self.assertIn("ALL TESTS PASSED", r.stdout)
            self.assertEqual(sum(l.startswith("pass") for l in r.stdout.splitlines()), 27)

    def test_a_normal_build(self):
        self._run([])

    def test_the_opening(self):
        self._run(["WG_DEV_GFX11"])


class SourcePins(unittest.TestCase):
    def test_the_variable_is_read_only_in_a_build_with_the_macro(self):
        h = read("clean", "runtime", "rdna4.h")
        on = h.index("#ifdef WG_DEV_GFX11\ninline bool dev_gfx11_shortcut() {")
        off = h.index("#else\ninline bool dev_gfx11_shortcut() { return false; }\n#endif")
        self.assertLess(on, off)
        self.assertIn('std::getenv("WG_DEV_GFX11_SHORTCUT")', h[on:off])
        self.assertEqual(h.count("WG_DEV_GFX11_SHORTCUT\")"), 1)
        for name in ("device_d3d12.cpp", "runtime.cpp", "interop/executor_select.h"):
            self.assertNotIn("getenv(\"WG_DEV_GFX11_SHORTCUT", read("clean", "runtime", *name.split("/")))

    def test_the_device_takes_the_shortcut_only_on_a_board_of_the_opening(self):
        dev = read("clean", "runtime", "device_d3d12.cpp")
        self.assertEqual(dev.count("gfx11_ = gfx11_card && !wgrt::dev_gfx11_shortcut();"), 2)  # borrowed and own device
        self.assertEqual(dev.count("shortcut_ = gfx11_card && wgrt::dev_gfx11_shortcut();"), 2)
        # no target in shortcut mode: arch() is "" (the gfx12 folder dx12 and the gfx12 accumulator table, as on RDNA 4)
        self.assertIn("std::string arch() const override { return gfx11_ ? wgrt::dev_gfx11_arch() : std::string(); }", dev)

    def test_the_rounding_probe_runs_at_create_in_shortcut_mode_and_only_there(self):
        dev = read("clean", "runtime", "device_d3d12.cpp")
        self.assertIn("(!borrowed_ || driver_probe()) && (!shortcut_ || rounding_probe())", dev)
        body = dev[dev.index("bool rounding_probe() {"):dev.index("bool create_device() {")]
        self.assertIn("round16::first_mismatch(values)", body)
        self.assertIn("shortcut refused: the driver does not round F32 -> F16 -> F32", body)
        self.assertIn("return fail(\"native D3D12 shortcut requires native_round_probe", body)

    def test_the_rounding_kernel_is_in_the_gfx11_set_only(self):
        src = read("clean", "dx12", "kernels", "native_game_helpers.hlsl")
        line = [l for l in src.splitlines() if l.startswith("// @wg-kernel native_round_probe ")]
        self.assertEqual(len(line), 1)
        self.assertTrue(line[0].endswith(" gfx=11"), line[0])   # a gfx12 build (the default) does not make it: the 46 files stay

    def test_no_release_path_mentions_the_variable(self):
        offenders = []
        for top in ("clean/runner", "clean/installer/tools", "tools", ".github"):
            base = os.path.join(ROOT, *top.split("/"))
            for dirpath, _, files in os.walk(base):
                for name in files:
                    path = os.path.join(dirpath, name)
                    try:
                        with open(path, encoding="utf-8") as f:
                            if "WG_DEV_GFX11_SHORTCUT" in f.read():
                                offenders.append(os.path.relpath(path, ROOT))
                    except (UnicodeDecodeError, OSError):
                        pass
        self.assertEqual(offenders, [], ", ".join(offenders))


if __name__ == "__main__":
    unittest.main()
