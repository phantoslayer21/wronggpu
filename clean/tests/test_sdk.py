# SPDX-License-Identifier: MIT
"""The HIP SDK policy (clean/runner/sdk.py) and its boundary. No compiler and no GPU: fake SDK folders, stubbed
subprocesses.

- discovery: the newest compatible official install by numeric version, 7.2 missing, several versions, none;
- the compiler: bare and absolute names canonicalised to the SDK's own, anything else only with --force-sdk;
- the boundary: device_command / find_clang refuse an unapproved SDK or compiler, so the entry points post_ablate,
  l_c32_check, k_port_check and fg_d3d12_check (and build.py, k_build.py) cannot compile with one;
- the HIP runtime: only the checked SDK's library, by full path, never a bare name; fails closed;
- Linux: the newest /opt/rocm-<version> wins over a /opt/rocm that points to an older one; /opt/rocm carries the version
  it points to; a standalone /opt/rocm counts (its .info/version, or ranked last without one);
- --force-sdk: post_ablate accepts a forced SDK end to end; every tracked CLI with --rocm offers --force-sdk and resolves
  --rocm through the policy (scan);
- logs: custom folders appear by id only (banner, refusal, command echo, compiler output of run / run_post_w64 /
  post_ablate, the post_w64 report), in every spelling (relative, either separator, case, 8.3, under the home folder);
- build.py: with ROCM_PATH and friends pointing elsewhere, every compile it starts uses the SDK's clang and its device
  libraries, and the build completes;
- a scan of clean/ and tools/: no Python file reads an SDK variable (ROCM_PATH, HIP_PATH, LLVM_PATH,
  HIP_DEVICE_LIB_PATH, HIP_CLANG_PATH, HIPFIRE_ROCM_PATH, ROCM_HOME) by os.environ[...], os.getenv, .get() or an
  alias; the scanner's own negative controls; a git failure fails the test."""
import contextlib
import io
import os
import re
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)

from clean.runner import build, hip, sdk  # noqa: E402

EXE = ".exe" if sys.platform == "win32" else ""


def fake_sdk(path, runtime=True):
    """A folder that looks like a HIP SDK install: bin/clang, bin/clang++, device libraries, the runtime library."""
    os.makedirs(os.path.join(path, "bin"), exist_ok=True)
    os.makedirs(os.path.join(path, "amdgcn", "bitcode"), exist_ok=True)
    for n in ("clang", "clang++"):
        open(os.path.join(path, "bin", n + EXE), "w").close()
    if runtime:
        if sys.platform == "win32":
            open(os.path.join(path, "bin", "amdhip64_7.dll"), "w").close()
        else:
            os.makedirs(os.path.join(path, "lib"), exist_ok=True)
            open(os.path.join(path, "lib", "libamdhip64.so"), "w").close()
    return path


class Env(unittest.TestCase):
    """Each test: a fake official root with versions, the policy's process state reset."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = os.path.join(self.tmp.name, "AMD", "ROCm")
        self.custom = fake_sdk(os.path.join(self.tmp.name, "custom-sdk"))
        self.patches = [patch.object(sdk, "WINDOWS_ROOT", self.root), patch.object(sdk, "LINUX_ROOT", self.root),
                        patch.object(sdk, "_approved", set()), patch.object(sdk, "_bannered", set())]
        for p in self.patches:
            p.start()

    def tearDown(self):
        for p in self.patches:
            p.stop()
        self.tmp.cleanup()

    def install(self, *versions):
        out = {}
        for v in versions:
            name = v if sys.platform == "win32" else "rocm-" + v
            out[v] = fake_sdk(os.path.join(self.root, name))
        return out


class Discovery(Env):
    def test_newest_compatible_by_numeric_version(self):
        got = self.install("7.2", "7.10", "7.0.1", "6.4", "8.0")
        os.makedirs(os.path.join(self.root, "7.9" if sys.platform == "win32" else "rocm-7.9"))   # incomplete: no clang
        self.assertEqual(sdk.default_rocm(), got["7.10"])
        self.assertEqual([v for v, _ in sdk.official_installs()], [(7, 10, 0), (7, 2, 0), (7, 0, 1)])
        self.assertTrue(sdk.is_official(got["7.2"]))
        self.assertFalse(sdk.is_official(got["6.4"]) or sdk.is_official(got["8.0"]))

    def test_7_2_missing(self):
        got = self.install("7.1")
        self.assertEqual(sdk.resolve(""), got["7.1"])
        self.assertEqual(sdk.resolve(got["7.1"]), got["7.1"])   # an official one needs no force

    def test_none_installed(self):
        self.assertIsNone(sdk.default_rocm())
        with self.assertRaises(SystemExit):
            sdk.resolve("")

    def test_environment_is_ignored(self):
        got = self.install("7.2")
        with patch.dict(os.environ, {k: self.custom for k in ("ROCM_PATH", "HIP_PATH", "LLVM_PATH",
                                                              "HIP_DEVICE_LIB_PATH", "HIP_CLANG_PATH")}):
            self.assertEqual(sdk.resolve(""), got["7.2"])


class Compiler(Env):
    def test_bare_and_absolute_names(self):
        s = self.install("7.2")["7.2"]
        own = os.path.join(s, "bin", "clang" + EXE)
        self.assertEqual(sdk.compiler(s, "clang"), own)
        self.assertEqual(sdk.compiler(s, "clang" + EXE), own)
        self.assertEqual(sdk.compiler(s, own), own)
        self.assertEqual(sdk.compiler(s, None), own)

    def test_other_compilers_need_force(self):
        s = self.install("7.2")["7.2"]
        other = os.path.join(self.custom, "bin", "clang" + EXE)
        with patch("shutil.which", lambda *a: other):
            with self.assertRaises(SystemExit):
                sdk.compiler(s, other)
            with self.assertRaises(SystemExit):
                sdk.compiler(s, "gcc")            # bare, not in the SDK (PATH would have one)
            self.assertEqual(sdk.compiler(s, other, force=True), other)
        sdk.check(s, other)                       # approved now

    def test_custom_sdk_needs_force(self):
        with self.assertRaises(SystemExit):
            sdk.resolve(self.custom)
        self.assertEqual(sdk.resolve(self.custom, force=True), self.custom)
        sdk.check_rocm(self.custom)


class Boundary(Env):
    def no_subprocess(self):
        def fail(*a, **k):
            raise AssertionError("a compiler was started")
        return contextlib.ExitStack(), fail

    def assert_refused(self, fn):
        with patch.object(build.subprocess, "run", side_effect=AssertionError("compiler started")), \
                patch.object(subprocess, "run", side_effect=AssertionError("compiler started")), \
                patch.dict(os.environ, {}), contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(SystemExit):
                fn()

    def test_device_command_and_find_clang(self):
        s = self.install("7.2")["7.2"]
        own = os.path.join(s, "bin", "clang" + EXE)
        with contextlib.redirect_stdout(io.StringIO()) as log:
            cmd = build.device_command(own, s, "k.hip", "k.co")
        self.assertIn("--rocm-device-lib-path=" + sdk.device_lib_dir(s), cmd)
        self.assertIn("HIP SDK: " + s, log.getvalue())
        for bad in (lambda: build.device_command(own, self.custom, "k.hip", "k.co"),
                    lambda: build.device_command(os.path.join(self.custom, "bin", "clang" + EXE), s, "k.hip", "k.co"),
                    lambda: build.find_clang(self.custom)):
            with self.assertRaises(SystemExit):
                bad()

    def test_entry_points_refuse_a_custom_sdk(self):
        self.install("7.2")
        out = os.path.join(self.tmp.name, "out")
        from clean.runner import post_ablate
        from clean.l.runner import l_c32_check
        from clean.k.dx12 import k_port_check
        from clean.fg.runner import fg_d3d12_check
        clang = os.path.join(self.custom, "bin", "clang" + EXE)
        self.assert_refused(lambda: post_ablate.build_all(out, self.custom, None, ["base"]))
        self.assert_refused(lambda: post_ablate.build_all(out, self.custom, clang, ["base"]))
        self.assert_refused(lambda: l_c32_check.compile_kernels(out, self.custom))
        self.assert_refused(lambda: l_c32_check.compile_kernels(out, self.custom, clang=clang))
        self.assert_refused(lambda: k_port_check.prepare(out, self.custom, list(k_port_check.FAMILY_KERNELS)[:1]))
        self.assert_refused(lambda: fg_d3d12_check.prepare(out, self.custom, []))

    def test_build_and_k_build_refuse_without_force(self):
        self.install("7.2")
        from clean.k.runner import k_build
        out = os.path.join(self.tmp.name, "out")
        self.assert_refused(lambda: build.main(["--out", out, "--rocm", self.custom]))
        self.assert_refused(lambda: build.main(["--out", out, "--clang", os.path.join(self.custom, "bin", "clang" + EXE)]))
        self.assert_refused(lambda: k_build.main(["--out", out, "--rocm", self.custom]))
        self.assert_refused(lambda: k_build.main(["--out", out, "--clang", os.path.join(self.custom, "bin", "clang" + EXE)]))


class Runtime(Env):
    def test_only_the_sdk_library(self):
        s = self.install("7.2")["7.2"]
        loaded = []
        with patch.object(hip.ctypes, "CDLL", lambda p: loaded.append(p) or object()), \
                patch.object(hip.os, "add_dll_directory", lambda p: None, create=True):
            hip._find_library(s)
        self.assertEqual(len(loaded), 1)
        self.assertTrue(os.path.isabs(loaded[0]) and sdk._inside(loaded[0], s))

    def test_fails_closed(self):
        s = fake_sdk(os.path.join(self.root, "7.3" if sys.platform == "win32" else "rocm-7.3"), runtime=False)
        loaded = []
        with patch.object(hip.ctypes, "CDLL", lambda p: loaded.append(p) or object()):
            with self.assertRaises(hip.HipError):
                hip._find_library(s)            # official, but no runtime library: no bare-name fallback
            with self.assertRaises(hip.HipError):
                hip._find_library(self.custom)  # not approved
            with self.assertRaises(hip.HipError):
                hip.Hip(self.custom)
        self.assertEqual(loaded, [])


class Logs(Env):
    def test_custom_paths_by_id_only(self):
        self.install("7.2")
        with self.assertRaises(SystemExit) as e:
            sdk.resolve(self.custom)
        self.assertNotIn(self.custom, str(e.exception))
        sdk.resolve(self.custom, force=True)
        clang = os.path.join(self.custom, "bin", "clang" + EXE)
        line = sdk.banner(self.custom, clang=clang)
        self.assertNotIn(os.path.normcase(self.custom), os.path.normcase(line))
        self.assertIn("custom (forced, id ", line)
        with contextlib.redirect_stdout(io.StringIO()) as log, patch.object(build.subprocess, "run",
                                                                            return_value=subprocess.CompletedProcess([], 0)):
            build.run(build.device_command(clang, self.custom, "k.hip", "k.co"))
        self.assertNotIn(os.path.normcase(self.custom), os.path.normcase(log.getvalue()))
        self.assertNotIn(os.path.normcase(self.custom).replace("\\", "/"), os.path.normcase(log.getvalue()))


class BuildPy(Env):
    def test_every_compile_uses_the_sdk(self):
        """build.py (no --rocm, SDK variables elsewhere): every device compile it starts uses the SDK's clang and both
        SDK flags, every Model M source is compiled, and the build completes."""
        s = self.install("7.2")["7.2"]
        own = os.path.join(s, "bin", "clang" + EXE)
        cmds = []
        env = {k: self.custom for k in ("ROCM_PATH", "HIP_PATH", "LLVM_PATH", "HIP_DEVICE_LIB_PATH", "HIP_CLANG_PATH")}
        with tempfile.TemporaryDirectory() as out, patch.dict(os.environ, env), \
                patch.object(build, "run", lambda cmd, *a, **k: cmds.append(list(cmd))), \
                patch.object(build, "run_post_w64", lambda cmd, *a, **k: cmds.append(list(cmd))), \
                patch.object(build, "build_runtime", lambda *a, **k: None), \
                contextlib.redirect_stdout(io.StringIO()):
            build.main(["--out", out, "--no-model-k", "--no-resource-gate", "--no-runtime"])
        self.assertTrue(cmds)
        for c in cmds:
            self.assertEqual(c[0], own, c[:2])
            self.assertIn(f"--rocm-path={s}", c)
            self.assertIn("--rocm-device-lib-path=" + sdk.device_lib_dir(s), c)
            self.assertFalse(any(os.path.normcase(self.custom) in os.path.normcase(str(a)) for a in c))
        outs = {os.path.basename(c[c.index("-o") + 1]) for c in cmds}
        for src in build.SOURCES:
            self.assertIn(os.path.splitext(src)[0] + ".co", outs)


class LinuxDiscovery(Env):
    """/opt/rocm next to /opt/rocm-<version> (official_installs(windows=False), on any test platform)."""

    def linux(self, *versions):
        got = {v: fake_sdk(os.path.join(self.root, "rocm-" + v)) for v in versions}
        for p in got.values():   # the Linux spelling of the compiler too
            open(os.path.join(p, "bin", "clang"), "w").close()
        return got

    def standalone(self, version_text=None):
        link = fake_sdk(os.path.join(self.root, "rocm"))
        open(os.path.join(link, "bin", "clang"), "w").close()
        if version_text is not None:
            os.makedirs(os.path.join(link, ".info"), exist_ok=True)
            with open(os.path.join(link, ".info", "version"), "w") as f:
                f.write(version_text)
        return link

    def found(self):
        return sdk.official_installs(linux_root=self.root, windows=False)

    def points_to(self, link, target):
        """/opt/rocm resolving to `target` (a real symlink where the platform allows, else _norm patched)."""
        orig = sdk._norm
        return patch.object(sdk, "_norm", lambda p: orig(target) if orig(p) == orig(link) else orig(p))

    def test_newest_wins_over_the_link(self):
        got = self.linux("7.1", "7.2")
        link = self.standalone()      # stands in for the link: resolves to rocm-7.1
        with self.points_to(link, got["7.1"]):
            found = self.found()
        self.assertEqual(found[0], ((7, 2, 0), got["7.2"]))                 # the newest, not the link's
        self.assertIn(((7, 1, 0), link), found)                             # the link with the version it points to
        self.assertLess(found.index(((7, 1, 0), link)), found.index(((7, 1, 0), got["7.1"])))

    def test_link_to_the_newest(self):
        got = self.linux("7.1", "7.2")
        link = self.standalone()
        with self.points_to(link, got["7.2"]):
            found = self.found()
        self.assertEqual(found[0], ((7, 2, 0), link))
        self.assertEqual(found[1], ((7, 2, 0), got["7.2"]))

    def test_real_symlink(self):
        got = self.linux("7.1", "7.2")
        link = os.path.join(self.root, "rocm")
        try:
            os.symlink(got["7.1"], link, target_is_directory=True)
        except (OSError, NotImplementedError):
            self.skipTest("no symlinks on this system")
        found = self.found()
        self.assertEqual(found[0], ((7, 2, 0), got["7.2"]))
        self.assertIn(((7, 1, 0), link), found)

    def test_standalone_counts(self):
        link = self.standalone("7.2.0-43\n")
        self.assertEqual(self.found(), [((7, 2, 0), link)])
        got = self.linux("7.1")
        self.assertEqual(self.found(), [((7, 2, 0), link), ((7, 1, 0), got["7.1"])])

    def test_standalone_without_version_ranks_last(self):
        link = self.standalone()
        got = self.linux("7.0")
        self.assertEqual(self.found(), [((7, 0, 0), got["7.0"]), ((-1,), link)])

    def test_standalone_incompatible_or_incomplete(self):
        self.standalone("6.4.1")
        self.assertEqual(self.found(), [])
        os.makedirs(os.path.join(self.tmp.name, "other"), exist_ok=True)
        self.assertEqual(sdk.official_installs(linux_root=os.path.join(self.tmp.name, "other"), windows=False), [])


class ForceCli(Env):
    """--force-sdk on the CLIs with --rocm: post_ablate end to end, and a scan of every tracked CLI."""

    def test_post_ablate_accepts_a_forced_sdk(self):
        self.install("7.2")
        from clean.runner import post_ablate
        out = os.path.join(self.tmp.name, "out")
        seen = []

        class Stop(Exception):
            pass

        def build_all(build_dir, rocm, clang, names):
            seen.append((rocm, clang))
            raise Stop()
        with patch.object(post_ablate, "build_all", build_all), contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(SystemExit):   # refused without the override
                post_ablate.main(["--build", out, "--build-only", "--rocm", self.custom])
            self.assertEqual(seen, [])
            with self.assertRaises(Stop):
                post_ablate.main(["--build", out, "--build-only", "--rocm", self.custom, "--force-sdk"])
        self.assertEqual(seen, [(os.path.abspath(self.custom), None)])
        sdk.check_rocm(self.custom)                # approved for this process

    def test_post_ablate_forced_build_output_is_redacted(self):
        from clean.runner import post_ablate
        clang = os.path.join(self.custom, "bin", "clang" + EXE)
        sdk.resolve(self.custom, force=True)
        err = f"clang: error: {clang}: bad\nIn file included from {self.custom.replace(os.sep, '/')}/include/x.h"
        with patch.object(build, "run_captured", lambda cmd: subprocess.CompletedProcess(cmd, 1, "", err)), \
                contextlib.redirect_stdout(io.StringIO()) as log:
            with self.assertRaises(SystemExit) as e:
                post_ablate.build_all(os.path.join(self.tmp.name, "out"), self.custom, clang, ["base"])
        for text in (str(e.exception), log.getvalue()):
            self.assertNotIn("custom-sdk", text)
            self.assertIn("custom (forced, id ", text)

    def test_every_rocm_cli_has_the_override(self):
        """Every tracked Python file with a --rocm option also offers --force-sdk and resolves --rocm through the
        policy after parsing (sdk.add_arguments + sdk.resolve, or sdk.add_force_argument + sdk.apply)."""
        r = subprocess.run(["git", "ls-files", "--", "clean", "tools"], cwd=ROOT, capture_output=True, text=True,
                           check=True)
        files = [f for f in r.stdout.split() if f.endswith(".py")]
        self.assertGreater(len(files), 200)
        clis, missing = 0, []
        for f in files:
            with open(os.path.join(ROOT, f), encoding="utf-8") as fh:
                s = fh.read()
            if not re.search(r"""add_argument\(\s*["']--rocm["']""", s) or f == "clean/runner/sdk.py":
                continue
            clis += 1
            if not (("sdk.add_force_argument(" in s and "sdk.apply(" in s) or
                    ("sdk.add_arguments(" in s and "sdk.resolve(" in s)):
                missing.append(f)
        self.assertGreater(clis, 40)
        self.assertEqual(missing, [])
        with open(os.path.join(ROOT, "clean", "runner", "build.py"), encoding="utf-8") as fh:
            self.assertIn("sdk.add_arguments(ap)", fh.read())


class Redaction(Env):
    """Custom (forced) paths in logs: every spelling, and build.py's post_w64 path (echo, stderr, report)."""

    def test_spellings(self):
        sdk.resolve(self.custom, force=True)
        c = self.custom
        texts = [c, c.replace("\\", "/"), c.upper(), c.replace(os.sep, os.sep * 2), "-I" + c + os.sep + "include",
                 "--rocm-path=" + c, '"' + os.path.join(c, "bin", "clang" + EXE) + '"']
        for t in texts:
            out = sdk.redact(t)
            self.assertNotIn("custom-sdk", out.lower(), t)
            self.assertIn("custom (forced, id ", out, t)
        for t in (c + "2", os.path.join(self.tmp.name, "custom-sdk-old"), "custom-sdk.txt"):
            self.assertEqual(sdk.redact(t), t)      # another name stays

    def test_relative_and_home_spellings(self):
        cwd = os.getcwd()
        os.chdir(self.tmp.name)
        try:
            with patch.object(sdk.os.path, "expanduser", lambda p: self.tmp.name if p == "~" else p):
                sdk.resolve("custom-sdk", force=True)
                for t in ("custom-sdk", "custom-sdk/bin/clang", "-Icustom-sdk\\include", "." + os.sep + "custom-sdk",
                          os.path.join("~", "custom-sdk", "bin")):
                    out = sdk.redact(t)
                    self.assertNotIn("custom-sdk", out, t)
                for t in ("notcustom-sdk", "custom-sdk2", "custom-sdk.py"):
                    self.assertEqual(sdk.redact(t), t)
        finally:
            os.chdir(cwd)

    def test_short_name(self):
        short = sdk._short_name(self.custom)
        if not short or os.path.normcase(short) == os.path.normcase(self.custom):
            self.skipTest("no distinct 8.3 name here")
        sdk.resolve(self.custom, force=True)
        self.assertNotIn(short.lower(), sdk.redact(short).lower())

    def test_run_post_w64(self):
        sdk.resolve(self.custom, force=True)
        clang = sdk.compiler(self.custom, os.path.join(self.custom, "bin", "clang" + EXE), force=True)
        cmd = [clang, f"--rocm-path={self.custom}", f"--rocm-device-lib-path={sdk.device_lib_dir(self.custom)}",
               "k.hip"]
        err = (f"remark: {self.custom.replace(os.sep, '/')}/amdgcn/bitcode/ocml.bc\n"
               f"{os.path.relpath(self.custom)}{os.sep}include: warning\n")
        out = os.path.join(self.tmp.name, "out")
        os.makedirs(out)
        for rc in (1, 0):
            with patch.object(build, "run_captured", lambda c, rc=rc: subprocess.CompletedProcess(c, rc, "", err)), \
                    contextlib.redirect_stdout(io.StringIO()) as log, contextlib.redirect_stderr(io.StringIO()) as elog:
                with contextlib.suppress(SystemExit):
                    build.run_post_w64(cmd, out, gates={})
            with open(os.path.join(out, build.POST_W64_REPORT)) as f:
                report = f.read()
            for text in (log.getvalue(), elog.getvalue(), report):
                self.assertNotIn("custom-sdk", text)
            self.assertIn("custom (forced, id ", log.getvalue())
            self.assertIn("custom (forced, id ", report)

    def test_run(self):
        sdk.resolve(self.custom, force=True)
        err = f"In file included from {self.custom}{os.sep}include{os.sep}hip.h\n"
        with patch.object(build, "run_captured", lambda c: subprocess.CompletedProcess(c, 1, err, err)), \
                contextlib.redirect_stdout(io.StringIO()) as log, contextlib.redirect_stderr(io.StringIO()) as elog:
            with self.assertRaises(SystemExit):
                build.run([os.path.join(self.custom, "bin", "clang" + EXE), "-c", "k.hip"])
        self.assertNotIn("custom-sdk", log.getvalue() + elog.getvalue())


SDK_VARS = ("ROCM_PATH", "HIP_PATH", "LLVM_PATH", "HIP_DEVICE_LIB_PATH", "HIP_CLANG_PATH", "HIPFIRE_ROCM_PATH",
            "ROCM_HOME")
_V = "|".join(SDK_VARS)
READ = [re.compile(rf"""getenv\(\s*["']({_V})["']"""),                        # os.getenv("X"), getenv("X")
        re.compile(rf"""\.get\(\s*["']({_V})["']"""),                          # os.environ.get / env.get / alias.get
        re.compile(rf"""\b\w+\[\s*["']({_V})["']\s*\](?!\s*=[^=])""")]  # environ["X"] / alias["X"], not stores
# files whose reads are of a dictionary the code itself built or checks (no environment value used as an SDK):
# hip.py's cleaned copy, and test_hip_env.py, which reads back the value the cleanup set
ALLOWED = {"clean/runner/hip.py", "clean/tests/test_hip_env.py"}


def scan_text(text):
    """The SDK-variable reads in Python source (lines that delete a variable are not reads)."""
    return [m.group(0) for line in text.splitlines() if not line.lstrip().startswith("del ")
            for p in READ for m in p.finditer(line)]


class Scan(unittest.TestCase):
    def test_scanner_negative_controls(self):
        for src in ('os.environ["ROCM_PATH"]', "os.getenv('HIP_PATH')", 'env = os.environ\nx = env.get("LLVM_PATH")',
                    'from os import environ\ny = environ["HIP_DEVICE_LIB_PATH"]', 'getenv("HIPFIRE_ROCM_PATH", "")',
                    'e.get("ROCM_HOME")'):
            self.assertTrue(scan_text(src), src)
        for src in ('os.environ["ROCM_PATH"] = sdk', 'del os.environ["HIP_PATH"]', 'for k in ("HIP_PATH",): env.pop(k)'):
            self.assertFalse(scan_text(src), src)

    def test_no_sdk_variable_read(self):
        r = subprocess.run(["git", "ls-files", "--", "clean", "tools"], cwd=ROOT, capture_output=True, text=True,
                           check=True)
        files = [f for f in r.stdout.split() if f.endswith(".py")]
        self.assertGreater(len(files), 200)       # a scan of nothing is not a pass
        hits = []
        for f in files:
            if f in ALLOWED or f == "clean/tests/test_sdk.py":
                continue
            with open(os.path.join(ROOT, f), encoding="utf-8") as fh:
                hits += [f"{f}: {h}" for h in scan_text(fh.read())]
        self.assertEqual(hits, [])


if __name__ == "__main__":
    unittest.main()
