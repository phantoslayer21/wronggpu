# SPDX-License-Identifier: MIT
"""R1: WG_DEV_DIAGNOSTICS (executor=hip and the frame capture in a game) is off in every release build and package.

No GPU, no compiler needed: the build script's runtime command, the CI and release files, and the library check are read.
Run: python -m unittest clean.tests.test_dev_diagnostics_off
"""
import importlib.util
import os
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
import check_no_dev_diagnostics as chk  # noqa: E402


def read(*p):
    with open(os.path.join(ROOT, *p), encoding="utf-8", errors="replace") as f:
        return f.read()


def load_build():
    sys.path.insert(0, ROOT)
    spec = importlib.util.spec_from_file_location("wg_build_for_test", os.path.join(ROOT, "clean", "runner", "build.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class DevDiagnosticsOff(unittest.TestCase):
    def test_the_runtime_command_has_the_define_only_on_request(self):
        b = load_build()
        self.assertFalse(b.DEV_DIAGNOSTICS)
        self.assertFalse([a for a in b.runtime_command("clang++", "out.dll") if "WG_DEV_DIAGNOSTICS" in a])
        b.DEV_DIAGNOSTICS = True
        try:
            self.assertTrue([a for a in b.runtime_command("clang++", "out.dll") if a.startswith("-DWG_DEV_DIAGNOSTICS")])
        finally:
            b.DEV_DIAGNOSTICS = False

    def test_no_release_path_turns_it_on(self):
        # the define and the build flag appear in the source of the build script and the backend only; no CI workflow,
        # CMake file, release script or packaging file asks for it
        offenders = []
        for base in (".github", os.path.join("tools", "release"), os.path.join("clean", "installer")):
            full = os.path.join(ROOT, base)
            if not os.path.isdir(full):
                continue
            for dp, _, names in os.walk(full):
                for n in names:
                    if n.endswith((".yml", ".yaml", ".txt", ".cmake", ".py", ".sh", ".bat", ".ps1", ".cpp", ".h", ".json")):
                        p = os.path.join(dp, n)
                        with open(p, encoding="utf-8", errors="replace") as f:
                            t = f.read()
                        if "WG_DEV_DIAGNOSTICS" in t or "--dev-diagnostics" in t:
                            offenders.append(os.path.relpath(p, ROOT))
        self.assertEqual(offenders, [], "a CI, release or installer file turns the developer diagnostics on")

    def test_the_library_marker_and_its_check(self):
        backend = read("clean", "runtime", "interop", "backend.cpp")
        self.assertIn('"wg-build dev-diagnostics=0"', backend)
        self.assertIn('"wg-build dev-diagnostics=1"', backend)
        self.assertIn("#ifdef WG_DEV_DIAGNOSTICS", backend)
        ok, _ = chk.check_bytes(b"MZ\x00..wg-build dev-diagnostics=0\x00..")
        self.assertTrue(ok)
        for bad in (b"wg-build dev-diagnostics=1", b"no marker here", b"wg-build dev-diagnostics=0 wg-build dev-diagnostics=1"):
            self.assertFalse(chk.check_bytes(bad)[0], bad)
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(chk.main([d]), 1, "a folder with no library must not pass")
            good = os.path.join(d, "wronggpu_runtime.dll")
            with open(good, "wb") as f:
                f.write(b"x" * 100 + b"wg-build dev-diagnostics=0" + b"y")
            self.assertEqual(chk.main([d]), 0)
            with open(os.path.join(d, "wronggpu_runtime_dev.dll"), "wb") as f:
                f.write(b"wg-build dev-diagnostics=1")
            self.assertEqual(chk.main([d]), 1, "one developer library among the packaged ones fails the folder")


if __name__ == "__main__":
    unittest.main()
