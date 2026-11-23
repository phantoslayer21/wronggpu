# SPDX-License-Identifier: MIT
"""The developer-only opening of RDNA 3 (WG_DEV_GFX11, clean/runtime/rdna4.h): a normal build is unchanged, the opening takes
the RX 7900 series board id (1002:744C) and nothing else.

No GPU: clean/runtime/interop/tests/test_dev_gfx11.cpp is built twice with the host C++ compiler (CXX, clang++ or g++; skipped
without one), without and with -DWG_DEV_GFX11, and both builds must pass. The places that cannot run on a host (the native
device, the runtime's create) are pinned from the source: every identity check of the device asks the opening next to the
RDNA 4 rule, the Ray Reconstruction runtime never does, the manifest of a build is read through one function, and no build
script, CI file or release script defines the macro (a library made with it is never a package).

Run: python -m unittest clean.tests.test_dev_gfx11
"""
import os
import re
import shutil
import subprocess
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TEST = os.path.join(ROOT, "clean", "runtime", "interop", "tests", "test_dev_gfx11.cpp")


def read(*p):
    with open(os.path.join(ROOT, *p), encoding="utf-8") as f:
        return f.read()


class HostBuilds(unittest.TestCase):
    def _run(self, defines, expect_passes):
        cxx = os.environ.get("CXX") or shutil.which("clang++") or shutil.which("g++")
        if not cxx:
            self.skipTest("no host C++ compiler (CXX, clang++ or g++)")
        with tempfile.TemporaryDirectory(prefix="wg_g11_") as tmp:
            exe = os.path.join(tmp, "test_dev_gfx11")
            r = subprocess.run([cxx, "-std=c++17", "-O1", *["-D" + d for d in defines], TEST, "-o", exe],
                               capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, "build failed:\n" + r.stdout + r.stderr)
            r = subprocess.run([exe], capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, "\n".join(l for l in r.stdout.splitlines() if l.startswith("FAIL")) + r.stderr)
            self.assertIn("ALL TESTS PASSED", r.stdout)
            self.assertEqual(sum(l.startswith("pass") for l in r.stdout.splitlines()), expect_passes)

    def test_a_normal_build_is_unchanged(self):
        self._run([], 28)

    def test_the_opening_takes_the_rx_7900_id(self):
        self._run(["WG_DEV_GFX11"], 29)


class SourcePins(unittest.TestCase):
    def test_every_identity_check_of_the_device_asks_the_opening_too(self):
        dev = read("clean", "runtime", "device_d3d12.cpp")
        calls = [m.start() for m in re.finditer(r"wgrt::rdna4_identity\(", dev)]
        self.assertEqual(len(calls), 2)  # (test_executor_select.py pins the same count: the shared rule, twice)
        self.assertIn("if (!wgrt::rdna4_identity(identity) && !gfx11_card)", dev)  # the borrowed device refuses everything else
        self.assertIn("if (wgrt::rdna4_identity(identity) || gfx11_card) {", dev)  # the own device selects the same boards
        self.assertEqual(dev.count("const bool gfx11_card = wgrt::dev_gfx11_identity(identity);"), 2)
        sel = read("clean", "runtime", "interop", "executor_select.h")
        self.assertEqual(sel.count("wgrt::rdna4_identity("), 1)  # the executor choice's one identity condition
        self.assertEqual(sel.count("wgrt::dev_gfx11_identity("), 1)

    def test_the_ray_reconstruction_runtime_and_the_rule_itself_stay_closed(self):
        rr = read("clean", "rr_f", "runtime", "src", "rr_runtime.cpp")
        self.assertNotIn("gfx11", rr)
        self.assertNotIn("dev_gfx11", rr)
        h = read("clean", "runtime", "rdna4.h")
        body = h[h.index("inline bool rdna4_identity("):]
        body = body[:body.index("\n}\n")]
        self.assertNotIn("gfx11", body)
        self.assertNotIn("kDevGfx11", body)

    def test_the_manifest_of_a_build_is_read_through_one_function(self):
        rt = read("clean", "runtime", "runtime.cpp")
        code = "\n".join(l for l in rt.splitlines() if not l.lstrip().startswith("//"))
        self.assertNotIn('"/dx12/manifest.json"', code)
        self.assertEqual(code.count("dx12_manifest()"), 2 + 1)  # two readers and the definition
        dev = read("clean", "runtime", "device_d3d12.cpp")
        self.assertIn('wgrt::dx12_folder(arch())', dev)
        self.assertEqual(dev.count('"/dx12/manifest.json"'), 1)  # the original reader, kept for every other build

    def test_the_runtime_has_the_device_before_it_reads_a_manifest_and_takes_its_layout_from_the_device(self):
        rt = read("clean", "runtime", "runtime.cpp")
        early = rt.index("given = make_d3d12_device(build_dir, &dev_err);")
        self.assertIn("if (given) dx12_arch = given->arch();", rt[early:early + 200])
        self.assertLess(early, rt.index("const std::string missing = l_native_missing();"))
        self.assertLess(early, rt.index("if (!choose_kernels(c)) return false;"))
        self.assertIn("wgrt::kDevGfx11 && c.backend == 3", rt[early - 200:early])  # the other builds never get here
        self.assertIn("drow != (native_gfx11 ? drow_gfx11() : drow_gfx12())", rt)
        self.assertIn('const bool native_gfx11 = wgrt::kDevGfx11 && dev && target_family(dev->arch()) == "gfx11";', rt)
        dev = read("clean", "runtime", "device_d3d12.cpp")
        self.assertIn("std::string arch() const override { return gfx11_ ? wgrt::dev_gfx11_arch() : std::string(); }", dev)
        # the manifest of a normal build is read before the device, as it always was; the opening reads it after
        first = dev.index('if (!wgrt::kDevGfx11 && !read_manifest(build + "/dx12/manifest.json")) return false;')
        self.assertLess(first, dev.index("if (!borrow_device(borrowed_device, borrowed_queue)) return false;"))
        self.assertGreater(dev.index("if (wgrt::kDevGfx11 && !read_manifest("), dev.index("} else if (!create_device()) return false;"))

    def test_no_build_script_ci_or_release_file_defines_the_macro(self):
        offenders = []
        for top in ("clean/runner", "clean/installer/tools", "tools", ".github"):
            base = os.path.join(ROOT, *top.split("/"))
            for dirpath, _, files in os.walk(base):
                for name in files:
                    path = os.path.join(dirpath, name)
                    try:
                        with open(path, encoding="utf-8") as f:
                            if "WG_DEV_GFX11" in f.read():
                                offenders.append(os.path.relpath(path, ROOT))
                    except (UnicodeDecodeError, OSError):
                        pass
        self.assertEqual(offenders, [], "a release path mentions WG_DEV_GFX11: " + ", ".join(offenders))


if __name__ == "__main__":
    unittest.main()
