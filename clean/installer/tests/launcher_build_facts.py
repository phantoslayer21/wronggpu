# SPDX-License-Identifier: MIT
"""What the launcher under test was built with, for run_launcher_tests.py: checks that follow the build instead of assuming
a release build (a fresh clone's suite build differs from a release in its version, its payload and its direct route).
Kept out of run_launcher_tests.py, which stays below the 256 KB source limit. Written for this project."""
import os
import re
import shutil

from run_installer_tests import read, write, STAND_IN_RUNTIME


def restorer_ok(un, exe):
    """uninstall.exe is the restorer, not the launcher: byte for byte the restorer built next to the launcher
    (WrongGPU-Restore.exe) when that is there, else at least smaller than the launcher and not its bytes."""
    if not os.path.isfile(un):
        return False
    built = os.path.join(os.path.dirname(exe), "WrongGPU-Restore.exe")
    if os.path.isfile(built):
        return read(un) == read(built)
    return os.path.getsize(un) < os.path.getsize(exe) and read(un) != read(exe)


def older_version(version):
    """A version below the launcher's own ("0.3.0-test" -> "0.2.9"; "0.6.5" -> "0.6.4"; "1.0.0" -> "0.9.9"), for a faked
    record of an earlier install. None when there is none (0.0.0)."""
    m = re.match(r"(\d+)\.(\d+)\.(\d+)", version or "")
    if not m:
        return None
    a, b, c = (int(x) for x in m.groups())
    if c:
        return "%d.%d.%d" % (a, b, c - 1)
    if b:
        return "%d.%d.9" % (a, b - 1)
    if a:
        return "%d.9.9" % (a - 1)
    return None


def stand_in_runtime(folder):
    """Whether the WrongGPU runtime installed in folder is the suite's own stand-in (run_installer_tests payload), which
    has no kernels and cannot prepare anything on a GPU."""
    p = os.path.join(folder, "wronggpu_runtime.dll")
    return os.path.isfile(p) and read(p) == STAND_IN_RUNTIME


# The test isolation hooks a WRONGGPU_TESTING build compiles in (each string is only in such a build, behind #ifdef): its
# store libraries from the test folder only and every store access logged (#243), the edition download from the test server
# only, and an Apps & features root of its own per test run. A release launcher has none of them: run by the suite, it could
# reach the real download page, the PC's own stores and the real uninstall registry root before any check sees it.
TEST_HOOKS = ("WRONGGPU_TEST_LIBRARIES", "WRONGGPU_TEST_ACCESS_LOG", "WRONGGPU_TEST_EDITION", "WrongGPU-CandidateTests")
_hooks_ok = {}


def missing_test_hooks(path):
    """The test isolation hooks the program at path lacks (UTF-16 strings in its bytes); empty: a testing build with all."""
    try:
        data = read(path)
    except OSError:
        return list(TEST_HOOKS)
    return [m for m in TEST_HOOKS if m.encode("utf-16-le") not in data]


def require_test_hooks(path, what="the program"):
    """Fails closed BEFORE a program runs: the suite runs only WRONGGPU_TESTING builds with the test isolation hooks (every
    launcher it is given, their copies and the uninstall.exe they write). Checked once per file (path, size, time)."""
    try:
        st = os.stat(path)
        key = (os.path.abspath(path).lower(), st.st_size, st.st_mtime_ns)
    except OSError:
        raise SystemExit("%s %s is not there: not run" % (what, path))
    if key not in _hooks_ok:
        missing = missing_test_hooks(path)
        if missing:
            raise SystemExit("%s %s is not a WRONGGPU_TESTING build with the test isolation hooks (missing: %s): the suite never "
                             "runs it (build it with -DWRONGGPU_TESTING=ON)" % (what, path, ", ".join(missing)))
        _hooks_ok[key] = True


def rr_free(folder, record):
    """No RR settings file in an RR-free install: neither WrongGPU\\wronggpu.ini on disk (case-insensitively: on Windows the
    payload's own lower-case wronggpu\\ folder is the same folder, and its kernels are allowed) nor its record line."""
    return not os.path.isfile(os.path.join(folder, "WrongGPU", "wronggpu.ini")) and "wronggpu\\wronggpu.ini" not in record.lower()


def self_checks(t):
    """The suite's own guards on synthetic files, before any launcher call: a program without the test isolation hooks is
    refused before it runs (nothing started, nothing logged), the launcher under test has the hooks, and the RR-free check
    finds an RR settings file that landed in the payload's lower-case wronggpu\\ folder."""
    print("\n[self-checks] the suite's guards: only testing builds run; the RR-free check")
    d = os.path.join(t.work, "self-checks")
    shutil.rmtree(d, ignore_errors=True)
    fake = os.path.join(d, "release", "WrongGPU.exe")
    write(fake, STAND_IN_RUNTIME)  # a program without the hooks, as a release launcher is
    size = lambda: os.path.getsize(t.log) if os.path.isfile(t.log) else 0
    before, why = size(), ""
    try:
        t.run("version", exe=fake)
    except SystemExit as e:
        why = str(e)
    t.check("preflight: a program without the test isolation hooks is refused before it runs (a release launcher never reaches the "
            "real download page, the PC's stores or the real uninstall registry root)",
            "not a WRONGGPU_TESTING build" in why and size() == before and not os.path.exists(t.out), why)
    t.check("preflight: the launcher under test is a WRONGGPU_TESTING build with every isolation hook", not missing_test_hooks(t.exe),
            missing_test_hooks(t.exe))
    b = os.path.join(d, "Game", "bin")
    write(os.path.join(b, "wronggpu", "kernels", "enc.hsaco"), b"kernel")
    ok_kernels = rr_free(b, "file wronggpu\\kernels\\enc.hsaco")
    write(os.path.join(b, "wronggpu", "wronggpu.ini"), b"[wronggpu]\r\n")  # WrongGPU\wronggpu.ini, in the lower-case folder
    t.check("RR-free check: the payload's wronggpu\\kernels alone is RR-free; WrongGPU\\wronggpu.ini in that lower-case folder, or "
            "its record line, is not", ok_kernels and not rr_free(b, "") and not rr_free(os.path.join(d, "empty"), "file WrongGPU\\wronggpu.ini"),
            ok_kernels)
    shutil.rmtree(d, ignore_errors=True)
