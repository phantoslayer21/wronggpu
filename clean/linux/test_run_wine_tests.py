# SPDX-License-Identifier: GPL-3.0-only
"""CPU-only tests of run_wine_tests.py's hang probe and wine_env's --wine override, with fake wine, wineserver, compiler
and xvfb-run scripts (no Wine, no GPU, no MinGW):  python3 -I clean/linux/test_run_wine_tests.py"""
import contextlib
import io
import os
import re
import signal
import stat
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_wine_tests as rwt  # noqa: E402
import wine_env  # noqa: E402

FAKE_WINE = """#!/bin/sh
echo "wine $* DISPLAY=${DISPLAY-unset} WINEPREFIX=${WINEPREFIX-}" >> "$FAKE_LOG"
if [ "$1" = wineboot ]; then
  mkdir -p "$WINEPREFIX/drive_c/windows/system32"; : > "$WINEPREFIX/system.reg"; exit 0
fi
case "${FAKE_MODE-}" in
  hang) sleep 300 & echo $! >> "$FAKE_PIDS"; echo $$ >> "$FAKE_PIDS"; exec sleep 300 ;;
  needs_display) [ -n "${DISPLAY-}" ] || { echo $$ >> "$FAKE_PIDS"; exec sleep 300; } ;;
  exit1) echo "fake wine: probe failed" >&2; exit 1 ;;
  summary) echo "wine ran: $1"; echo "5 checks passed"; exit 0 ;;
esac
exit 0
"""
FAKE_WINESERVER = """#!/bin/sh
echo "wineserver $* WINEPREFIX=${WINEPREFIX-}" >> "$FAKE_LOG"
if [ "$1" = -k ]; then   # only the clean-up call misbehaves, so a prefix can be set up first
  case "${FAKE_WS-}" in
    fail) echo "wineserver: cannot kill" >&2; exit 3 ;;
    slow) sleep 30 ;;
  esac
fi
exit 0
"""
FAKE_XVFB = """#!/bin/sh
echo "xvfb-run $*" >> "$FAKE_LOG"
if [ -n "${FAKE_XVFB_FAIL-}" ]; then echo "xvfb-run: error: xauth command not found" >&2; exit 3; fi
[ "$1" = -a ] && shift
: > "$FAKE_XAUTH"
trap 'echo xvfb-cleanup >> "$FAKE_LOG"; rm -f "$FAKE_XAUTH"' EXIT
[ -n "${FAKE_XVFB_STUBBORN-}" ] && trap '' TERM
DISPLAY=:99 "$@" &
child=$!
echo $$ >> "$FAKE_PIDS"
wait $child
rc=$?
[ -n "${FAKE_XVFB_STUBBORN-}" ] && while :; do sleep 1; done
exit $rc
"""
FAKE_CXX = """#!/bin/sh
echo "cxx $*" >> "$FAKE_LOG"
if [ "$1" = --version ]; then echo fake-cxx; exit 0; fi
[ -n "${FAKE_CXX_FAIL-}" ] && { echo "fake compiler: cannot find -ldxgi" >&2; exit 1; }
while [ $# -gt 0 ]; do [ "$1" = -o ] && { : > "$2"; break; }; shift; done
exit 0
"""


def script(path, body):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body)
    path.chmod(path.stat().st_mode | stat.S_IEXEC)
    return path


LINUX_ONLY = "needs Linux: the fake wine, wineserver and xvfb-run are shell scripts, and the clean-up uses /proc and process groups"


class Fixture(unittest.TestCase):
    @classmethod
    def setUpClass(cls):   # at run time, not import time: a Windows shard must still discover (and skip) the module
        if not sys.platform.startswith("linux"):
            raise unittest.SkipTest(LINUX_ONLY)

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = Path(self.tmp.name).resolve()
        self.log = self.dir / "fake.log"
        self.pids = self.dir / "fake.pids"
        self.log.write_text("")
        self.addCleanup(self.stop_pids)
        self.xauth = self.dir / "fake.xauth"
        base = {"FAKE_LOG": str(self.log), "FAKE_PIDS": str(self.pids), "FAKE_XAUTH": str(self.xauth)}
        p = mock.patch.dict(os.environ, base)
        p.start()
        self.addCleanup(p.stop)
        # xvfb-run is found in the fixture's own bin folder only, never in a host install
        os.environ["PATH"] = os.pathsep.join(d for d in os.environ["PATH"].split(os.pathsep)
                                             if not os.access(os.path.join(d, "xvfb-run"), os.X_OK))
        for k in ("DISPLAY", "WAYLAND_DISPLAY", "FAKE_MODE", "FAKE_WS", "FAKE_CXX_FAIL", "FAKE_XVFB_FAIL", "FAKE_XVFB_STUBBORN", "MINGW_CXX", "WG_WINE_BIN", "WG_DXC"):
            os.environ.pop(k, None)
        # a Proton install with the DLLs the runner copies, and a fake wine + wineserver in it
        self.proton = self.dir / "proton"
        for rel in ("vkd3d-proton/x86_64-windows/d3d12.dll", "vkd3d-proton/x86_64-windows/d3d12core.dll",
                    "dxvk/x86_64-windows/dxgi.dll", "vkd3d/x86_64-windows/libvkd3d-utils-1.dll",
                    "vkd3d/x86_64-windows/libvkd3d-1.dll", "vkd3d/x86_64-windows/libvkd3d-shader-1.dll"):
            f = self.proton / "files" / "lib" / "wine" / rel
            f.parent.mkdir(parents=True, exist_ok=True)
            f.write_bytes(b"x")
        self.pwine = script(self.proton / "files" / "bin" / "wine", FAKE_WINE)
        script(self.proton / "files" / "bin" / "wineserver", FAKE_WINESERVER)
        self.prefix = self.dir / "prefix"

    def stop_pids(self):
        if self.pids.exists():
            for line in self.pids.read_text().split():
                with contextlib.suppress(ProcessLookupError):
                    os.kill(int(line), signal.SIGTERM)

    def alive_pids(self):
        alive = []
        for line in (self.pids.read_text().split() if self.pids.exists() else []):
            try:
                os.kill(int(line), 0)
                alive.append(int(line))
            except ProcessLookupError:
                pass
        return alive

    def use_xvfb(self):
        script(self.dir / "bin" / "xvfb-run", FAKE_XVFB)
        os.environ["PATH"] = f"{self.dir / 'bin'}{os.pathsep}{os.environ['PATH']}"

    def calls(self, prefix=""):
        return [l for l in self.log.read_text().splitlines() if l.startswith(prefix)]

    def env_for(self, wine=None, **extra):
        env = wine_env.prepare(self.proton, self.prefix, log=lambda m: None, wine=wine)
        env.update(extra)
        return env

    def run_main(self, *args, stubs=None):
        out = io.StringIO()
        saved = dict(rwt.TESTS)
        try:
            for k, v in (stubs or {}).items():
                rwt.TESTS[k] = v
            with contextlib.redirect_stdout(out):
                rc = rwt.main(["--proton", str(self.proton), "--work", str(self.dir / "work"), "--probe-timeout", "2", *args])
        finally:
            rwt.TESTS.clear()
            rwt.TESTS.update(saved)
        return rc, out.getvalue()


class ProbeHang(Fixture):
    def test_finished_probe_is_not_a_hang(self):
        self.assertFalse(rwt.probe_hang(self.env_for(), "x.exe", self.dir, 5))

    def test_stuck_probe_is_a_hang_even_when_a_child_keeps_the_output_open(self):
        os.environ["FAKE_MODE"] = "hang"
        self.assertTrue(rwt.probe_hang(self.env_for(), "x.exe", self.dir, 2))

    # finding 1: the probe gets interop's display setup (xvfb-run when DISPLAY is unset and it is installed)
    def test_probe_runs_under_xvfb_run_when_there_is_no_display(self):
        script(self.dir / "bin" / "xvfb-run", FAKE_XVFB)
        os.environ["PATH"] = f"{self.dir / 'bin'}{os.pathsep}{os.environ['PATH']}"
        os.environ["FAKE_MODE"] = "needs_display"
        self.assertFalse(rwt.probe_hang(self.env_for(), "x.exe", self.dir, 3))
        self.assertTrue(self.calls("xvfb-run"))

    def test_test_programs_run_under_xvfb_run_too(self):
        script(self.dir / "bin" / "xvfb-run", FAKE_XVFB)
        os.environ["PATH"] = f"{self.dir / 'bin'}{os.pathsep}{os.environ['PATH']}"
        os.environ["FAKE_MODE"] = "needs_display"
        rc, _ = rwt.run_wine(self.env_for(), ["x.exe"], self.dir, 5)
        self.assertEqual(rc, 0)

    # finding 3: the wineserver that belongs to the wine, used for cleanup, scoped to the prepared prefix
    def test_timeout_cleanup_uses_the_resolved_wineserver_and_the_prefix(self):
        os.environ["FAKE_MODE"] = "hang"
        env = self.env_for()
        self.assertTrue(rwt.probe_hang(env, "x.exe", self.dir, 2))
        self.assertEqual(self.calls("wineserver -k"), [f"wineserver -k WINEPREFIX={self.prefix}"])

    def test_cleanup_uses_wg_wineserver_not_a_sibling_guess(self):
        # a wine whose wineserver is not beside it (a symlink in a user bin directory)
        os.environ["FAKE_MODE"] = "hang"
        ws = script(self.dir / "elsewhere" / "wineserver", FAKE_WINESERVER)
        env = self.env_for()
        env["WG_WINE"] = str(script(self.dir / "userbin" / "wine", FAKE_WINE))
        env["WG_WINESERVER"] = str(ws)
        self.assertTrue(rwt.probe_hang(env, "x.exe", self.dir, 2))
        self.assertEqual(len(self.calls("wineserver -k")), 1)

    # finding 4: a failing or stuck cleanup is reported, never raised
    def test_cleanup_failure_is_reported_not_discarded(self):
        os.environ["FAKE_MODE"] = "hang"
        os.environ["FAKE_WS"] = "fail"
        env = self.env_for()
        notes = []
        self.assertTrue(rwt.probe_hang(env, "x.exe", self.dir, 2, notes=notes))
        self.assertIn("cannot kill", " ".join(notes))

    def test_cleanup_timeout_is_reported_not_raised(self):
        os.environ["FAKE_MODE"] = "hang"
        os.environ["FAKE_WS"] = "slow"
        env = self.env_for()
        notes = []
        with mock.patch.object(rwt, "KILL_TIMEOUT", 1):
            self.assertTrue(rwt.probe_hang(env, "x.exe", self.dir, 2, notes=notes))
        self.assertTrue(notes)

    # round 2, item 1: a timeout under a wrapper that stays alive around its child (xvfb-run)
    def test_probe_timeout_under_xvfb_run_lets_the_wrapper_clean_up_and_stops_the_client(self):
        self.use_xvfb()
        os.environ["FAKE_MODE"] = "hang"
        env = self.env_for()
        self.assertTrue(rwt.probe_hang(env, "x.exe", self.dir, 2))
        self.assertEqual(self.alive_pids(), [])
        self.assertFalse(self.xauth.exists(), "the wrapper's own clean-up (X authority file) did not run")
        self.assertEqual(self.calls("xvfb-cleanup"), ["xvfb-cleanup"])
        self.assertEqual(self.calls("wineserver -k"), [f"wineserver -k WINEPREFIX={self.prefix}"])

    def test_test_program_timeout_under_xvfb_run_stops_the_client_and_cleans_up(self):
        self.use_xvfb()
        os.environ["FAKE_MODE"] = "hang"
        env = self.env_for()
        with self.assertRaises(Exception) as c:
            rwt.run_wine(env, ["x.exe"], self.dir, 2)
        self.assertIn("timed out", str(c.exception))
        self.assertEqual(self.alive_pids(), [])
        self.assertFalse(self.xauth.exists())
        self.assertEqual(self.calls("wineserver -k"), [f"wineserver -k WINEPREFIX={self.prefix}"])

    def test_test_program_timeout_without_xvfb_stops_the_client(self):
        os.environ["FAKE_MODE"] = "hang"
        env = self.env_for()
        with self.assertRaises(Exception):
            rwt.run_wine(env, ["x.exe"], self.dir, 2)
        self.assertEqual(self.alive_pids(), [])
        self.assertEqual(len(self.calls("wineserver -k")), 1)

    def test_a_wrapper_that_ignores_term_is_escalated_within_a_bound(self):
        self.use_xvfb()
        os.environ["FAKE_MODE"] = "hang"
        os.environ["FAKE_XVFB_STUBBORN"] = "1"
        env = self.env_for()
        notes = []
        t = time.monotonic()
        with mock.patch.object(rwt, "KILL_TIMEOUT", 1):
            self.assertTrue(rwt.probe_hang(env, "x.exe", self.dir, 2, notes=notes))
        self.assertLess(time.monotonic() - t, 20)
        self.assertEqual(self.alive_pids(), [])
        self.assertEqual(len(self.calls("wineserver -k")), 1)

    # item 3: a non-zero probe or wrapper exit is a failure with its log, not a finished probe
    def test_wrapper_setup_error_is_a_probe_failure_with_its_log(self):
        self.use_xvfb()
        os.environ["FAKE_XVFB_FAIL"] = "1"
        with self.assertRaises(Exception) as c:
            rwt.probe_hang(self.env_for(), "x.exe", self.dir, 5)
        self.assertIn("xauth command not found", str(c.exception))

    def test_probe_that_exits_non_zero_is_a_probe_failure(self):
        os.environ["FAKE_MODE"] = "exit1"
        with self.assertRaises(Exception) as c:
            rwt.probe_hang(self.env_for(), "x.exe", self.dir, 5)
        self.assertIn("probe failed", str(c.exception))

    # finding 6
    def test_message_names_the_cause_the_override_and_only_the_observed_case(self):
        msg = rwt.hang_message({"WG_WINE": "/p/files/bin/wine"}, 90)
        self.assertIn("--wine", msg)
        self.assertIn("GE-Proton 11-11", msg)
        self.assertNotIn("11-12", msg)
        self.assertIn("90 s", msg)

    def test_readme_names_only_the_observed_case(self):
        text = (Path(__file__).resolve().parent / "README.md").read_text()
        self.assertNotIn("11-12", text)

    def test_readme_says_which_tests_fail_and_that_ineligible_ones_skip(self):
        text = " ".join((Path(__file__).resolve().parent / "README.md").read_text().replace("`", "").split())
        self.assertIn("A test that would skip (rr-leak without --runtime, ags-layout without dxc) skips as before", text)


class WineOverride(Fixture):
    def test_missing_wine_is_refused(self):
        with self.assertRaises(SystemExit) as c:
            wine_env.prepare(self.proton, self.prefix, wine="/nonexistent/wine")
        self.assertIn("--wine", str(c.exception))

    # finding 2
    def test_relative_wine_becomes_absolute(self):
        script(self.dir / "rel" / "bin" / "wine", FAKE_WINE)
        script(self.dir / "rel" / "bin" / "wineserver", FAKE_WINESERVER)
        cwd = os.getcwd()
        os.chdir(self.dir / "rel")
        self.addCleanup(os.chdir, cwd)
        env = self.env_for(wine="./bin/wine")
        self.assertTrue(os.path.isabs(env["WG_WINE"]), env["WG_WINE"])
        os.chdir(self.dir)   # a child process runs from another folder (the --work folder)
        self.assertFalse(rwt.probe_hang(env, "x.exe", self.dir, 5))

    def test_wine_from_path_name_becomes_absolute(self):
        script(self.dir / "pbin" / "wine", FAKE_WINE)
        script(self.dir / "pbin" / "wineserver", FAKE_WINESERVER)
        os.environ["PATH"] = f"{self.dir / 'pbin'}{os.pathsep}{os.environ['PATH']}"
        self.assertTrue(os.path.isabs(self.env_for(wine="wine")["WG_WINE"]))

    # finding 3
    def test_wine_with_its_wineserver_elsewhere_prepares_a_fresh_prefix(self):
        real = script(self.dir / "real" / "bin" / "wine", FAKE_WINE)
        (self.dir / "userbin").mkdir()
        (self.dir / "userbin" / "wine").symlink_to(real)
        script(self.dir / "wsdir" / "wineserver", FAKE_WINESERVER)
        os.environ["PATH"] = f"{self.dir / 'wsdir'}{os.pathsep}{os.environ['PATH']}"
        env = self.env_for(wine=str(self.dir / "userbin" / "wine"))
        self.assertEqual(Path(env["WG_WINESERVER"]), self.dir / "wsdir" / "wineserver")
        self.assertEqual(self.calls("wineserver -w"), [f"wineserver -w WINEPREFIX={self.prefix}"])

    # round 2, item 2: a symlinked wine uses the wineserver beside its target, not an unrelated one beside the link
    def test_symlinked_wine_prefers_the_wineserver_beside_its_target(self):
        real = script(self.dir / "real" / "bin" / "wine", FAKE_WINE)
        script(self.dir / "real" / "bin" / "wineserver", FAKE_WINESERVER)
        (self.dir / "userbin").mkdir()
        (self.dir / "userbin" / "wine").symlink_to(real)
        script(self.dir / "userbin" / "wineserver", FAKE_WINESERVER.replace("echo ", "echo WRONG ", 1))
        env = self.env_for(wine=str(self.dir / "userbin" / "wine"))
        self.assertEqual(Path(env["WG_WINESERVER"]), self.dir / "real" / "bin" / "wineserver")
        self.assertFalse([l for l in self.calls() if "WRONG" in l])

    def test_wine_without_any_wineserver_is_refused_with_a_message(self):
        script(self.dir / "lonely" / "wine", FAKE_WINE)
        os.environ["PATH"] = os.pathsep.join(p for p in os.environ["PATH"].split(os.pathsep) if not (Path(p) / "wineserver").exists())
        with self.assertRaises(SystemExit) as c:
            self.env_for(wine=str(self.dir / "lonely" / "wine"))
        self.assertIn("wineserver", str(c.exception))

    def test_new_prefix_log_names_the_wine_that_made_it(self):
        script(self.dir / "sys" / "wine", FAKE_WINE)
        script(self.dir / "sys" / "wineserver", FAKE_WINESERVER)
        msgs = []
        wine_env.prepare(self.proton, self.prefix, log=msgs.append, wine=str(self.dir / "sys" / "wine"))
        self.assertIn(str(self.dir / "sys" / "wine"), msgs[0])


class MainFlow(Fixture):
    def setUp(self):
        super().setUp()
        self.cxx = script(self.dir / "tools" / "x86_64-w64-mingw32-g++", FAKE_CXX)
        os.environ["MINGW_CXX"] = str(self.cxx)
        self.called = []

        def stub(name, result):
            def f(a, env, work):
                self.called.append(name)
                return result
            return f
        self.stub = stub

    # finding 4
    def test_probe_build_failure_is_a_per_test_result_and_launcher_still_runs(self):
        os.environ["FAKE_CXX_FAIL"] = "1"
        rc, out = self.run_main("--tests", "interop,launcher", stubs={"launcher": self.stub("launcher", ("PASS", "ok"))})
        self.assertEqual(rc, 1)
        self.assertIn("FAIL interop", out)
        self.assertIn("-ldxgi", out)
        self.assertIn("PASS launcher", out)
        self.assertIn("== summary", out)
        self.assertEqual(self.called, ["launcher"])

    def test_probe_start_failure_is_a_per_test_result(self):
        script(self.pwine, FAKE_WINE.replace("exit 0\n", "exit 0\n", 1))
        with mock.patch.object(rwt, "probe_hang", side_effect=OSError("cannot start wine")):
            rc, out = self.run_main("--tests", "interop,launcher", stubs={"launcher": self.stub("launcher", ("PASS", "ok"))})
        self.assertIn("cannot start wine", out)
        self.assertIn("PASS launcher", out)

    def test_hang_with_cleanup_error_puts_the_error_in_the_message(self):
        os.environ["FAKE_MODE"] = "hang"
        os.environ["FAKE_WS"] = "fail"
        rc, out = self.run_main("--tests", "interop,launcher", stubs={"launcher": self.stub("launcher", ("PASS", "ok"))})
        self.assertIn("FAIL interop", out)
        self.assertIn("cannot kill", out)
        self.assertIn("PASS launcher", out)

    # finding 5
    def test_rr_leak_without_runtime_still_skips_without_probe_or_test_program(self):
        os.environ["FAKE_MODE"] = "hang"
        rc, out = self.run_main("--tests", "rr-leak")
        self.assertIn("SKIP rr-leak", out)
        self.assertEqual(rc, 0)
        # the prefix is still set up (wineboot, as on main); no adapter probe and no test program start
        self.assertEqual([l for l in self.calls("wine ") if "wineboot" not in l], [])
        self.assertFalse([l for l in self.calls("cxx") if "dxgi_probe" in l])

    def test_ags_layout_without_dxc_still_skips_without_probe_or_test_program(self):
        os.environ["FAKE_MODE"] = "hang"
        os.environ["WG_DXC"] = str(self.dir / "no-such-dxc")
        rc, out = self.run_main("--tests", "ags-layout")
        self.assertIn("SKIP ags-layout", out)
        self.assertEqual(rc, 0)
        self.assertEqual([l for l in self.calls("wine ") if "wineboot" not in l], [])   # wineboot: prefix setup only

    def test_wrapper_setup_error_fails_the_tests_that_need_the_probe_with_its_log(self):
        self.use_xvfb()
        os.environ["FAKE_XVFB_FAIL"] = "1"
        rc, out = self.run_main("--tests", "interop,launcher", stubs={"launcher": self.stub("launcher", ("PASS", "ok"))})
        self.assertEqual(rc, 1)
        self.assertIn("FAIL interop", out)
        self.assertIn("xauth command not found", out)
        self.assertIn("PASS launcher", out)

    def test_eligible_test_still_fails_when_wine_hangs(self):
        os.environ["FAKE_MODE"] = "hang"
        rc, out = self.run_main("--tests", "interop", stubs={"interop": self.stub("interop", ("PASS", "ok"))})
        self.assertEqual(rc, 1)
        self.assertIn("did not list the GPU adapters", out)
        self.assertEqual(self.called, [])


class NonLinux(unittest.TestCase):
    """What a Windows CI shard sees: the module must be discoverable and skip, not fail."""

    def test_fixture_tests_skip_when_the_platform_is_not_linux(self):
        code = ("import sys, unittest\nsys.path.insert(0, sys.argv[1])\n"
                "import test_run_wine_tests as t\nsys.platform = 'win32'\n"
                "suite = unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromTestCase(c) for c in vars(t).values()\n"
                "    if isinstance(c, type) and issubclass(c, t.Fixture) and c is not t.Fixture)\n"
                "r = unittest.TextTestRunner(verbosity=0).run(suite)\n"
                "print('SKIPPED', len(r.skipped), 'RAN', r.testsRun, 'FAILED', len(r.failures) + len(r.errors))\n"
                "print('REASON', sorted({x[1] for x in r.skipped})[0] if r.skipped else None)\n")
        r = subprocess.run([sys.executable, "-I", "-c", code, str(Path(__file__).resolve().parent)],
                           capture_output=True, text=True, timeout=120)
        self.assertEqual(r.returncode, 0, r.stderr)
        m = re.search(r"SKIPPED (\d+) RAN (\d+) FAILED (\d+)", r.stdout)
        self.assertTrue(m, r.stdout + r.stderr)
        skipped, ran, failed = map(int, m.groups())
        self.assertEqual(failed, 0)
        self.assertGreater(skipped, 0)
        self.assertEqual(ran, 0)   # no fixture test body ran
        self.assertIn("needs Linux", r.stdout)


class LauncherWithoutDisplay(Fixture):
    """test_launcher runs its programs directly with Wine, as before xvfb-run was added: a broken xvfb-run
    (here: xauth missing) must not stop them."""

    def test_launcher_programs_run_directly_when_xvfb_run_fails(self):
        self.use_xvfb()
        os.environ["FAKE_XVFB_FAIL"] = "1"
        os.environ["FAKE_MODE"] = "summary"
        tools = self.dir / "tools"
        script(tools / "x86_64-w64-mingw32-g++", FAKE_CXX)
        script(tools / "x86_64-w64-mingw32-windres", "#!/bin/sh\nexit 0\n")
        for t in ("cmake", "ninja"):
            script(tools / t, "#!/bin/sh\nexit 0\n")
        os.environ["MINGW_CXX"] = str(tools / "x86_64-w64-mingw32-g++")
        os.environ["PATH"] = f"{tools}{os.pathsep}{os.environ['PATH']}"
        work = self.dir / "work"
        (work / "launcher-build").mkdir(parents=True)
        real_run = subprocess.run

        def fake_run(argv, *a, **k):   # the installer's --prepare-only and the cmake steps: nothing to build here
            if argv[0] == sys.executable or Path(argv[0]).name in ("cmake", "ninja"):
                return subprocess.CompletedProcess(argv, 0, "", "")
            return real_run(argv, *a, **k)

        env = self.env_for()
        a = mock.Mock(runtime=None)
        with mock.patch.object(rwt.subprocess, "run", fake_run):
            status, why = rwt.test_launcher(a, env, work)
        self.assertEqual(status, "PASS", why)
        self.assertFalse(self.calls("xvfb-run"), "the launcher programs went through xvfb-run")
        self.assertEqual(len([l for l in self.calls("wine ") if ".exe" in l or "launcher-build" in l]), 4)


if __name__ == "__main__":
    unittest.main()
