# SPDX-License-Identifier: MIT
"""What the launcher under test was built with, for run_launcher_tests.py: checks that follow the build instead of assuming
a release build (a fresh clone's suite build differs from a release in its version, its payload and its direct route).
Kept out of run_launcher_tests.py, which stays below the 256 KB source limit. Written for this project."""
import os
import re

from run_installer_tests import read, STAND_IN_RUNTIME


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


def has_exact_name(folder, name):
    """Whether folder holds an entry spelled exactly name (Windows names are case-insensitive: os.path.exists would also
    match a payload's lower-case wronggpu\\ folder for WrongGPU\\)."""
    return os.path.isdir(folder) and name in os.listdir(folder)
