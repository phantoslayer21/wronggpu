# SPDX-License-Identifier: MIT
"""The developer-only native Model K choice of the WG_DEV_GFX11 opening (clean/runtime/rdna4.h): the runtime launches the
native kernel <name>__g11 where the dx12-gfx11 manifest has it, only in a build made with the macro, only when
WG_DEV_GFX11_NATIVE_K=1 is set on top of the shortcut.

No GPU: clean/runtime/tests/test_native_k_name.cpp is built twice with the host C++ compiler (CXX, clang++ or g++; skipped
without one), without and with -DWG_DEV_GFX11, and both builds must pass (the lookup picks __g11 only in the second, only when
the manifest has it). The device's wiring cannot run on a host and is pinned from the source: the manifest filter, the one
lookup point, the explicit refusal without the shortcut, no release path mentions the variable, and every native kernel
declaration has the same launch interface (block, wave, arguments, code object) as the RDNA 4 kernel it replaces.

Run: python -m unittest clean.tests.test_dev_gfx11_native_k
"""
import os
import re
import shutil
import subprocess
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TEST = os.path.join(ROOT, "clean", "runtime", "tests", "test_native_k_name.cpp")
DECL = re.compile(r"^//\s*@wg-kernel\s+(\S+)\s+(.*)$", re.M)


def read(*p):
    with open(os.path.join(ROOT, *p), encoding="utf-8") as f:
        return f.read()


def declarations(*p):
    out = {}
    for name, rest in DECL.findall(read(*p)):
        out[name] = dict(tok.split("=", 1) for tok in rest.split() if "=" in tok)
    return out


class HostBuilds(unittest.TestCase):
    def _run(self, defines, header="rdna4.h", mutate=None):
        cxx = os.environ.get("CXX") or shutil.which("clang++") or shutil.which("g++")
        if not cxx:
            self.skipTest("no host C++ compiler (CXX, clang++ or g++)")
        with tempfile.TemporaryDirectory(prefix="wg_nk_") as tmp:
            src = TEST
            if mutate:  # a copy of the runtime folder's two files with one deliberate break (negative control)
                os.makedirs(os.path.join(tmp, "tests"))
                shutil.copy(TEST, os.path.join(tmp, "tests", "test_native_k_name.cpp"))
                with open(os.path.join(tmp, "rdna4.h"), "w", encoding="utf-8") as f:
                    f.write(mutate(read("clean", "runtime", "rdna4.h")))
                src = os.path.join(tmp, "tests", "test_native_k_name.cpp")
            exe = os.path.join(tmp, "t")
            r = subprocess.run([cxx, "-std=c++17", "-O1", *["-D" + d for d in defines], src, "-o", exe],
                               capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, "build failed:\n" + r.stdout + r.stderr)
            return subprocess.run([exe], capture_output=True, text=True)

    def _ok(self, defines):
        r = self._run(defines)
        self.assertEqual(r.returncode, 0, "\n".join(l for l in r.stdout.splitlines() if l.startswith("FAIL")) + r.stderr)
        self.assertIn("ALL TESTS PASSED", r.stdout)
        self.assertEqual(sum(l.startswith("pass") for l in r.stdout.splitlines()), 16)

    def test_a_normal_build_never_picks_a_native_kernel(self):
        self._ok([])

    def test_the_opening_picks_the_native_kernel_when_the_manifest_has_it(self):
        self._ok(["WG_DEV_GFX11"])

    def test_negative_control_a_broken_lookup_is_caught(self):
        r = self._run(["WG_DEV_GFX11"], mutate=lambda h: h.replace('name + "__g11"', 'name + "__g12"'))
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("FAIL on: native form only in a WG_DEV_GFX11 build", r.stdout)

    def test_negative_control_picking_without_asking_the_manifest_is_caught(self):
        r = self._run(["WG_DEV_GFX11"], mutate=lambda h: h.replace("return has(g11) ? g11 : name;", "return g11;"))
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("FAIL on: a kernel the manifest has no __g11 for keeps its name", r.stdout)


class SourcePins(unittest.TestCase):
    def test_the_variable_is_read_only_in_a_build_with_the_macro(self):
        h = read("clean", "runtime", "rdna4.h")
        on = h.index("#ifdef WG_DEV_GFX11\ninline bool dev_gfx11_native_k() {")
        off = h.index("#else\ninline bool dev_gfx11_native_k() { return false; }")
        self.assertLess(on, off)
        self.assertIn('std::getenv("WG_DEV_GFX11_NATIVE_K")', h[on:off])
        self.assertNotIn('"__g11"', h[:on] + h[off:])  # no such string literal in the normal build's half
        self.assertEqual(h[on:off].count('"__g11"'), 2)  # the suffix test and the name; none in the normal build's half
        self.assertEqual(h.count("WG_DEV_GFX11_NATIVE_K\")"), 1)

    def test_the_device_has_one_lookup_point_and_filters_the_manifest(self):
        dev = read("clean", "runtime", "device_d3d12.cpp")
        self.assertIn("const std::string name = native_kernel_name(requested);", dev)
        self.assertEqual(dev.count("native_kernel_name("), 2)  # definition and find_port, which prepare and launch both use
        self.assertEqual(dev.count("find_port("), 3)           # definition, prepare, launch
        self.assertIn("g11_only_ = true;", dev)
        self.assertIn("needs WG_DEV_GFX11_SHORTCUT=1", dev)
        # the manifest read, the variable and the refusal are inside the macro
        i = dev.index("wgrt::dev_gfx11_native_k()")
        self.assertEqual(dev[:i].count("#ifdef WG_DEV_GFX11"), dev[:i].count("#endif") + 1)

    def test_no_release_path_mentions_the_variable(self):
        offenders = []
        for top in ("clean/runner", "clean/installer/tools", "tools", ".github"):
            base = os.path.join(ROOT, *top.split("/"))
            for dirpath, _, files in os.walk(base):
                for name in files:
                    path = os.path.join(dirpath, name)
                    try:
                        with open(path, encoding="utf-8") as f:
                            if "WG_DEV_GFX11_NATIVE_K" in f.read():
                                offenders.append(os.path.relpath(path, ROOT))
                    except (UnicodeDecodeError, OSError):
                        pass
        self.assertEqual(offenders, [], ", ".join(offenders))

    def test_the_smx_hook_is_outside_the_self_check_digest(self):
        rt = read("clean", "k", "runtime", "runtime_k.cpp")
        for m in re.finditer(r"// SMX-SELFCHECK-SOURCE BEGIN.*?// SMX-SELFCHECK-SOURCE END", rt, re.S):
            self.assertNotIn("mix_unchecked", m.group(0))
        self.assertIn('hook == "mix_unchecked"', rt)

    def test_every_native_declaration_has_the_launch_interface_of_its_port(self):
        ports = declarations("clean", "k", "dx12", "kernels", "k_block.hlsl")
        native = {n: d for n, d in declarations("clean", "k", "dx12", "kernels", "k_block_g11.hlsl").items() if n.endswith("__g11")}
        self.assertEqual(len(native), 13)
        for name, d in native.items():
            port = ports[name[:-5]]  # the manifest's runtime-visible fields: what launch() packs and checks
            for key in ("co", "block", "wave", "args"):
                self.assertEqual(d.get(key), port.get(key), name + " " + key)
            self.assertEqual(d.get("gfx"), "11", name)
            self.assertNotIn("gfx", port)  # the RDNA 4 kernel is the default set: it stays


if __name__ == "__main__":
    unittest.main()
