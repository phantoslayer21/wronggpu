# SPDX-License-Identifier: MIT
"""R1: WG_DEV_DIAGNOSTICS (executor=hip and the frame capture in a game) is off in every release build and package.

No GPU, no compiler needed: the build script's runtime command, the CI and release files, and the library check are read.
Run: python -m unittest clean.tests.test_dev_diagnostics_off
"""
import importlib.util
import os
import subprocess
import sys
import tempfile
import unittest
import zipfile

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
                    # (build and packaging scripts and configuration; the packager and the checks only NAME the flag to refuse it)
                    if n.endswith((".yml", ".yaml", ".txt", ".cmake", ".sh", ".bat", ".ps1", ".json")) or (
                            n.endswith(".py") and "release" in dp and n != "check_package_diagnostics.py"):
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


MARK0 = b"wg-build dev-diagnostics=0"
MARK1 = b"wg-build dev-diagnostics=1"
PACKAGER = os.path.join(ROOT, "clean", "installer", "tools", "make_payload.py")
RELEASE_CHECK = os.path.join(ROOT, "tools", "release", "check_package_diagnostics.py")


class PackagerRefusesDeveloperRuntime(unittest.TestCase):
    """The payload packager and the release check look at the BYTES of the runtime that would ship."""

    def _pack(self, files, extra_args=()):
        """Runs make_payload.py on a payload folder holding `files` ({relative name: bytes}) next to a front end."""
        d = tempfile.mkdtemp(prefix="wg_pl_")
        payload = os.path.join(d, "payload")
        os.makedirs(payload)
        for rel, data in {"amd_fidelityfx_dx12.dll": b"front end", **files}.items():
            p = os.path.join(payload, *rel.split("/"))
            os.makedirs(os.path.dirname(p), exist_ok=True)
            with open(p, "wb") as f:
                f.write(data)
        with open(os.path.join(d, "map.json"), "wb") as f:
            f.write(b"{}")
        out = os.path.join(d, "out")
        r = subprocess.run([sys.executable, PACKAGER, "--payload", payload, "--map", os.path.join(d, "map.json"), "--version", "0.0.1",
                            "--out", out, *extra_args], capture_output=True, text=True)
        return r, out

    def test_a_release_runtime_is_packaged(self):
        r, out = self._pack({"wronggpu_runtime.dll": b"MZ" + b"x" * 64 + MARK0})
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertTrue(os.path.isfile(os.path.join(out, "payload_table.h")))

    def test_a_developer_runtime_refuses_the_payload(self):
        r, out = self._pack({"wronggpu_runtime.dll": b"MZ" + MARK1})
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("dev-diagnostics=1", r.stderr + r.stdout)
        self.assertFalse(os.path.isfile(os.path.join(out, "payload_table.h")), "no payload table is written for a refused payload")

    def test_a_runtime_without_the_marker_refuses_the_payload(self):  # fails closed: a build older than R1
        r, _ = self._pack({"wronggpu_runtime.dll": b"MZ" + b"x" * 64})
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("no 'wg-build dev-diagnostics=' marker", r.stderr + r.stdout)

    def test_every_runtime_library_is_checked_wherever_it_is(self):
        r, _ = self._pack({"wronggpu_runtime.dll": MARK0, "sub/wronggpu_runtime_extra.dll": MARK1})
        self.assertNotEqual(r.returncode, 0, "a second library that is a developer build refuses the payload")
        r, _ = self._pack({"wronggpu_runtime.dll": MARK0, "sub/other.dll": b"carries a developer runtime " + MARK1})
        self.assertNotEqual(r.returncode, 0, "any other embedded file that says dev-diagnostics=1 refuses it too")
        r, _ = self._pack({"wronggpu_runtime.dll": MARK0, "sub/other.dll": b"plain"})
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)

    def test_the_final_package_check(self):
        with tempfile.TemporaryDirectory() as d:
            good_zip, bad_zip = os.path.join(d, "good.zip"), os.path.join(d, "bad.zip")
            with zipfile.ZipFile(good_zip, "w") as z:
                z.writestr("WrongGPU/wronggpu_runtime.dll", b"MZ" + MARK0)
                z.writestr("WrongGPU/readme.txt", b"text")
            with zipfile.ZipFile(bad_zip, "w") as z:
                z.writestr("WrongGPU/wronggpu_runtime.dll", b"MZ" + MARK1)
            empty_zip = os.path.join(d, "empty.zip")
            with zipfile.ZipFile(empty_zip, "w") as z:
                z.writestr("readme.txt", b"no runtime in here")
            exe_good, exe_bad = os.path.join(d, "WrongGPU.exe"), os.path.join(d, "WrongGPU-dev.exe")
            with open(exe_good, "wb") as f:
                f.write(b"MZ" + b"payload " + MARK0 + b" more")
            with open(exe_bad, "wb") as f:
                f.write(b"MZ" + b"payload " + MARK1)

            def run(path):
                return subprocess.run([sys.executable, RELEASE_CHECK, path], capture_output=True, text=True).returncode

            self.assertEqual(run(good_zip), 0)
            self.assertEqual(run(bad_zip), 1, "a developer runtime inside the package zip fails")
            self.assertEqual(run(empty_zip), 1, "a package with no runtime found must not pass")
            self.assertEqual(run(exe_good), 0, "the launcher embeds the payload's runtime as it is: its marker is found")
            self.assertEqual(run(exe_bad), 1)
            self.assertEqual(run(d), 1, "a folder holding any developer package fails")


if __name__ == "__main__":
    unittest.main()
