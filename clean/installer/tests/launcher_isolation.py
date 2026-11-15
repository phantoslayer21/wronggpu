# SPDX-License-Identifier: MIT
"""The launcher test runners' guard against the PC's own game folders (run_launcher_tests.py and the coexistence, Model L / K
and convert-ignore runners use it): a log line of the launcher's library searches that names a path outside the run's work
folder stops the whole run. Its own check (test_isolation_guard) runs first in every launcher suite run. Written for this
project."""
import os
import re

from run_installer_tests import write


def check_isolation(log, work, start, args):
    """The launcher's searches of the store libraries (the AMD FSR 4 search, the model search, a store library path) must stay
    inside the run's work folder: any such line of the log (from byte `start` on) naming a path outside it stops the whole run
    here, loudly (never a passing suite that read the PC's own game folders). Used by every launcher test runner."""
    try:
        with open(log, "rb") as f:
            f.seek(start)
            text = f.read().decode("utf-8", "replace")
    except OSError:
        return
    root = os.path.normcase(os.path.abspath(work))
    searches = ("amd fsr 4", "converter: file:", "model: using", "model: rejected", "\\steamapps\\", "\\steamlibrary\\", "\\epic games\\",
                "\\gog galaxy\\", "\\gog games\\")
    for line in text.splitlines():
        low = line.lower()
        if not any(s in low for s in searches):
            continue
        for m in re.finditer(r"(?i)\b[a-z]:\\[^\r\n]*?(?=\s\(|\)|;|,\s|\s+not used|$)", line):
            path = os.path.normcase(os.path.abspath(m.group(0).strip()))
            if not (path == root or path.startswith(root + os.sep)):
                raise SystemExit("TEST ISOLATION FAILED: the launcher (%s) searched outside the test folder:\n  %s\n  path: %s\n"
                                 "A test build must take its store libraries from WRONGGPU_TEST_LIBRARIES only." % (" ".join(args[:2]), line.strip(), m.group(0)))


def test_isolation_guard(t):
    """check_isolation itself (no launcher run): a search line naming a path outside the work folder stops the run, one inside
    it (or a line that is no search) does not."""
    print("\n[isolation] the guard against the PC's own game folders: an outside search path stops the run")
    log = os.path.join(t.work, "isolation-guard.log")
    outside = "Q:\\NotTheTestFolder\\Store\\Game"
    inside = os.path.join(t.work, "libraries", "Game")
    for kind, line in (("the model search", "model: using %s\\model-file.dll (another game (Game))"),
                       ("the AMD FSR 4 search", "AMD FSR 4 search: candidate %s\\fsr4.dll (4.1.0.0, Game)"),
                       ("the converter", "converter: file: %s\\model-file.dll")):
        for where, path, stops in (("outside", outside, True), ("inside", inside, False)):
            write(log, ("[00:00:00] " + line % path + "\r\n").encode("utf-8"))
            try:
                check_isolation(log, t.work, 0, ("install",))
                stopped = False
            except SystemExit as e:
                stopped = "TEST ISOLATION FAILED" in str(e)
            t.check("guard: %s naming a path %s the work folder %s the run" % (kind, where, "stops" if stops else "does not stop"), stopped == stops, line % path)
    write(log, b"[00:00:00] model: reusing Q:\\NotTheTestFolder\\model.wgcm (no search)\r\n")
    try:
        check_isolation(log, t.work, 0, ("install",))
        t.check("guard: a line that is no search (an outside path in it) does not stop the run", True)
    except SystemExit:
        t.check("guard: a line that is no search (an outside path in it) does not stop the run", False)
    os.remove(log)
