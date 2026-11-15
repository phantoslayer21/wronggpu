# SPDX-License-Identifier: MIT
"""The launcher test runners' guard against the PC's own game folders (#243). A WRONGGPU_TESTING launcher logs every store
access (util.h LibraryAccess: "library access: <path>", "library access refused: <path>") to WRONGGPU_TEST_ACCESS_LOG
before the read; check_isolation parses that log after every launcher or installer call and stops the whole run, loudly, on
a read outside the run's work folder (also \\\\?\\ and UNC forms), a store registry read, a missing or unreadable access log,
a log not of this run (its first line "run <WRONGGPU_TEST_RUN_ID>"), a call that logged nothing (every start of ours logs a
"process" line), or no access at all where the call was expected to scan. The runners never create the log: the launcher
does, and refuses a read whose line it cannot write. self_test (no launcher) runs first in every runner's main and
aborts the run when the guard itself is broken. run_verbs: the launcher's verbs with and without --noscan against a
non-empty test library. Written for this project."""
import os
import re
import uuid

from run_installer_tests import write

ACCESS_LOG = "library-access.log"  # in the work folder (each runner sets WRONGGPU_TEST_ACCESS_LOG to it)
_RUN_IDS = {}  # work folder -> this run's id (WRONGGPU_TEST_RUN_ID): the access log's first line must name it


def test_env(env, work):
    """The environment every runner gives the launcher: its store libraries from <work>\\libraries only (empty: none), every
    store access logged to <work>\\library-access.log by the launcher itself (never created here: a log the launcher could not
    write stays missing, and the guard stops the run), this run's id for its first line."""
    libs = os.path.join(work, "libraries")
    os.makedirs(libs, exist_ok=True)
    env["WRONGGPU_TEST_LIBRARIES"] = libs
    env["WRONGGPU_TEST_ACCESS_LOG"] = os.path.join(work, ACCESS_LOG)
    env["WRONGGPU_TEST_RUN_ID"] = new_run(work)
    return env


def new_run(work):
    """This run's id (the access log's first line must name it); an earlier run's log removed, never counted as this run's."""
    p = os.path.join(work, ACCESS_LOG)
    if os.path.exists(p):
        os.remove(p)
    rid = _RUN_IDS[os.path.normcase(os.path.abspath(work))] = uuid.uuid4().hex
    return rid



def access_size(work):
    p = os.path.join(work, ACCESS_LOG)
    return os.path.getsize(p) if os.path.isfile(p) else 0


def _scope(p):
    """A path in one form for the scope test (as util.cpp ScopeForm): \\\\?\\C:\\x as C:\\x, \\\\?\\UNC\\s\\x as \\\\s\\x."""
    p = p.strip()
    if p[:8].upper() == "\\\\?\\UNC\\":
        p = "\\\\" + p[8:]
    elif p[:4] in ("\\\\?\\", "\\??\\"):
        p = p[4:]
    if p.startswith("\\\\"):
        return os.path.normcase(p)  # UNC: never inside a local work folder
    return os.path.normcase(os.path.abspath(p))


def _inside(path, root):
    p = _scope(path)
    return p == root or p.startswith(root.rstrip("\\") + "\\")


def _stop(args, what, line):
    raise SystemExit("TEST ISOLATION FAILED: the launcher (%s) %s:\n  %s\nA test build must take its store libraries from "
                     "WRONGGPU_TEST_LIBRARIES only (util.h LibraryAccess)." % (" ".join(str(a) for a in args[:2]), what, line.strip()))


def check_isolation(log, work, start, args, access_start=0, expect_scan=False, logged=True):
    """After a launcher or installer call: its new access-log lines (from byte access_start) and its new main-log lines
    (from byte start; log None: none) must name nothing outside the work folder. The access log must be there, written by
    this run (its first line "run <this run's id>"), and (logged: a program of ours) have new lines from this call (every run
    of ours writes its start line). expect_scan: the call was meant to search the test libraries, so at least one access
    inside <work>\\libraries must be there (the guard's sensitivity)."""
    root = os.path.normcase(os.path.abspath(work))
    access = os.path.join(work, ACCESS_LOG)
    try:
        with open(access, "rb") as f:
            first = f.readline().decode("utf-8", "replace").strip()
            f.seek(access_start)
            lines = f.read().decode("utf-8", "replace").splitlines()
    except OSError as e:
        _stop(args, "left no readable store access log (the launcher writes it; a missing one means it could not)", "%s (%s)" % (access, e))
    want = _RUN_IDS.get(root)
    if want is None or first != "run " + want:
        _stop(args, "left an access log that is not this run's", "first line %r, this run is %r" % (first, want))
    if logged and not lines:
        _stop(args, "logged nothing (a program of ours writes its start line every run: its access lines are missing)", access)
    scanned = False
    libs = os.path.normcase(os.path.join(root, "libraries"))
    for line in lines:
        if line.startswith("library access refused: "):
            continue  # refused: nothing was read
        if not line.startswith("library access: "):
            continue
        what = line[len("library access: "):]
        if what.startswith("registry "):
            _stop(args, "read a store's registry key (the PC's own stores)", line)
        if not _inside(what, root):
            _stop(args, "read outside the test folder", line)
        scanned = scanned or _inside(what, libs)
    if expect_scan and not scanned:
        _stop(args, "was expected to search the test libraries but logged no access there (a guard that sees nothing proves nothing)",
              "%d access line(s) in %s" % (len(lines), access))
    if log:  # the main log: the searches' own lines (a second net)
        try:
            with open(log, "rb") as f:
                f.seek(start)
                text = f.read().decode("utf-8", "replace")
        except OSError as e:
            _stop(args, "left no readable log", "%s (%s)" % (log, e))
        searches = ("amd fsr 4", "converter: file:", "model: using", "model: rejected", "\\steamapps\\", "\\steamlibrary\\",
                    "\\epic games\\", "\\gog galaxy\\", "\\gog games\\")
        for line in text.splitlines():
            if not any(s in line.lower() for s in searches):
                continue
            for m in re.finditer(r"(?i)(\\\\[?]\\[^\r\n]*?|\\\\[^\\\s][^\r\n]*?|\b[a-z]:\\[^\r\n]*?)(?=\s\(|\)|;|,\s|\s+not used|$)", line):
                if not _inside(m.group(0), root):
                    _stop(args, "searched outside the test folder", line)


def self_test(work):
    """The guard's own check, before any launcher call of a runner: every case must hold, else the run stops here."""
    print("\n[isolation] the guard against the PC's own game folders (before any launcher call)")
    work = os.path.abspath(work)
    os.makedirs(os.path.join(work, "libraries"), exist_ok=True)
    access, log = os.path.join(work, ACCESS_LOG), os.path.join(work, "isolation-self-test.log")
    inside = os.path.join(work, "libraries", "Game")
    cases = [
        ("an access outside the work folder stops the run", "library access: Q:\\NotTheTestFolder\\Store\\Game", "", False, True),
        ("an access as a UNC path stops the run", "library access: \\\\server\\share\\Game", "", False, True),
        ("an access as \\\\?\\UNC stops the run", "library access: \\\\?\\UNC\\server\\share\\Game", "", False, True),
        ("a store registry read stops the run", "library access: registry GOG.com Games", "", False, True),
        ("an access inside the work folder passes", "library access: " + inside, "", True, False),
        ("an access inside, as \\\\?\\, passes", "library access: \\\\?\\" + inside, "", True, False),
        ("a refused access (nothing read) passes", "library access refused: Q:\\NotTheTestFolder\\Store", "", False, False),
        ("no access where a scan was expected stops the run", "", "", True, True),
        ("a model search line outside the work folder stops the run", "", "model: using Q:\\NotTheTestFolder\\x\\model-file.dll (another game (G))", False, True),
        ("an AMD FSR 4 search line outside stops the run", "", "AMD FSR 4 search: candidate Q:\\NotTheTestFolder\\fsr4.dll (4.1.0.0, G)", False, True),
        ("a converter line outside stops the run", "", "converter: file: Q:\\NotTheTestFolder\\model-file.dll", False, True),
        ("a line that is no search passes", "", "model: reusing Q:\\NotTheTestFolder\\model.wgcm (no search)", False, False),
    ]
    key = os.path.normcase(os.path.abspath(work))
    saved = _RUN_IDS.get(key)
    _RUN_IDS[key] = "self-test-run"
    head = b"run self-test-run\r\nprocess 1 WrongGPU start\r\n"
    cases += [("an access log of another run stops the run", None, "", False, True),
              ("a call that logged nothing stops the run", "", "", False, True)]
    bad = []
    for name, aline, mline, expect, stops in cases:
        if aline is None:  # another run's log
            write(access, b"run another-run\r\nprocess 1 WrongGPU start\r\n")
            start = 0
        elif name.startswith("a call that logged nothing"):
            write(access, head)
            start = len(head)  # (nothing after this call's start)
        else:
            write(access, head + ((aline + "\r\n").encode("utf-8") if aline else b""))
            start = 0
        write(log, ("[00:00:00] " + mline + "\r\n").encode("utf-8") if mline else b"")
        try:
            check_isolation(log, work, 0, ("self-test",), start, expect)
            stopped = False
        except SystemExit as e:
            stopped = "TEST ISOLATION FAILED" in str(e)
        print("  %s  guard: %s" % ("PASS " if stopped == stops else "FAIL ", name))
        if stopped != stops:
            bad.append(name)
    os.remove(access)
    try:
        check_isolation(None, work, 0, ("self-test",))
        missing = False
    except SystemExit:
        missing = True
    print("  %s  guard: a missing store access log stops the run" % ("PASS " if missing else "FAIL "))
    if not missing:
        bad.append("missing access log")
    for p in (log, access):
        if os.path.exists(p):
            os.remove(p)
    if saved is None:
        _RUN_IDS.pop(key, None)
    else:
        _RUN_IDS[key] = saved
    if bad:
        raise SystemExit("TEST ISOLATION SELF-TEST FAILED (%s): no launcher call is made with a broken guard." % "; ".join(bad))


def run_verbs(t, opti, m):
    """Every verb with --noscan searches nothing (no access in the test libraries), against a NON-EMPTY test library (a game
    with an AMD FSR 4 file name and a model file name, synthetic bytes), and without --noscan it searches them (the guard's
    sensitivity): install, repair, update, uninstall, the window's background refresh (ui-dump), the other model of a
    Preset K build, an apply batch with a noscan job then a normal one, and /NOSCAN with /FSR4 and /LIST."""
    print("\n[isolation-verbs] --noscan for every verb, against a non-empty test library")
    libgame = os.path.join(t.work, "libraries", "Library Game")
    write(os.path.join(libgame, "amd_fidelityfx_upscaler_dx12.dll"), b"MZ synthetic, not AMD's")
    write(os.path.join(libgame, "nvngx_dlss.dll"), t.model.file)
    libs = os.path.normcase(os.path.join(os.path.abspath(t.work), "libraries"))

    def touched(start):
        with open(os.path.join(t.work, ACCESS_LOG), "rb") as f:
            f.seek(start)
            return [l for l in f.read().decode("utf-8", "replace").splitlines()
                    if l.startswith("library access: ") and _inside(l[len("library access: "):], libs)]

    root, b, exe = m.game(t, "Isolation Game")
    inst = ("--opti=" + opti, "--force", "--experimental")
    for verb in ("install", "repair", "update", "uninstall"):
        a0 = access_size(t.work)
        args = (verb, exe, "--noscan") + (() if verb == "uninstall" else inst)
        rc = t.run(*args)
        t.check("%s --noscan: exit 0, nothing of the test libraries read (no access there)" % verb, rc == 0 and not touched(a0), (rc, touched(a0)[:3], t.j.get("message")))
    a0 = access_size(t.work)
    l0 = os.path.getsize(t.log) if os.path.isfile(t.log) else 0
    t.expect_scan = True  # the guard's sensitivity: this call must log accesses in the test libraries
    rc = t.run("install", exe, *inst)
    t.expect_scan = False
    got = touched(a0)
    t.check("install without --noscan: the test libraries searched (accesses logged there; the AMD FSR 4 search and the model step)",
            rc == 0 and got and any("Library Game" in l for l in got), (rc, got[:3]))
    logt = m.read(t.log)[l0:].decode("utf-8", "replace") if os.path.isfile(t.log) else ""  # (this call's lines only)
    t.check("...the AMD FSR 4 search saw the test library's file (not AMD's: not used); the PC's stores never (the guard)",
            "AMD FSR 4 search:" in logt and "Library Game" in logt, t.logtail(800))
    if "(Preset K)" in logt or "Preset K model caches are prepared" in logt:  # a Preset K build: the other model's search
        t.check("...the other model's search (Preset K build) looked in the test library (its model file), nowhere else",
                any(("converter: file:" in l or "model: " in l) and "Library Game" in l for l in logt.splitlines()), t.logtail(800))
    else:
        print("  skipped: the other model's search (this launcher is built without Preset K)")
    for verb in ("repair", "update"):  # the guard's sensitivity for the other verbs that search
        a0 = access_size(t.work)
        t.expect_scan = True
        rc = t.run(verb, exe, *inst)
        t.expect_scan = False
        t.check("%s without --noscan: the test libraries searched (accesses logged there)" % verb, rc == 0 and touched(a0), (rc, touched(a0)[:3], t.j.get("message")))
    dump = os.path.join(t.work, "ui-dump-isolation.json")
    for noscan in (True, False):
        a0 = access_size(t.work)
        t.expect_scan = not noscan
        rc = t.run("ui-dump", "--dump=" + dump, *(("--noscan",) if noscan else ()))
        t.expect_scan = False
        t.check("the window's background refresh (ui-dump) %s: %s" % ("--noscan" if noscan else "without --noscan",
                "nothing of the test libraries read" if noscan else "the test libraries searched"),
                rc == 0 and (not touched(a0)) == noscan, (rc, touched(a0)[:3]))
    # an apply batch: a noscan job, then a normal one: the second one searches (the no-scan is per job)
    jobs = [os.path.join(t.work, "iso-job-%d.txt" % i) for i in (1, 2)]
    results = [os.path.join(t.work, "iso-result-%d.txt" % i) for i in (1, 2)]
    lad = t.env["LOCALAPPDATA"]
    for j, ns in zip(jobs, ("1", "0")):
        write(j, "op=repair\r\nexe=%s\r\ndir=%s\r\nlocalappdata=%s\r\nopti=%s\r\nforce=1\r\nexperimental=1\r\nnoscan=%s\r\n" % (exe, b, lad, opti, ns))
    batch = os.path.join(t.work, "iso-batch.txt")
    write(batch, "".join("job=%s\r\nresult=%s\r\n" % (j, r) for j, r in zip(jobs, results)))
    a0 = access_size(t.work)
    t.expect_scan = True
    rc = t.run("apply", "--batch=" + batch)
    t.expect_scan = False
    t.check("apply batch: a noscan job then a normal one: the normal one searches the test libraries (the no-scan never sticks)",
            rc == 0 and touched(a0), (rc, touched(a0)[:3], [m.read(r).decode("utf-8", "replace") if os.path.isfile(r) else "" for r in results]))
    a0 = access_size(t.work)
    write(jobs[0], "op=repair\r\nexe=%s\r\ndir=%s\r\nlocalappdata=%s\r\nopti=%s\r\nforce=1\r\nexperimental=1\r\nnoscan=0\r\n" % (exe, b, lad, opti))
    rc = t.run("apply", "--job=" + jobs[0], "--result=" + results[0], "--noscan")
    t.check("apply --noscan: the job searches nothing (its noscan set by the command line)", rc == 0 and not touched(a0), (rc, touched(a0)[:3]))
    # /NOSCAN with /FSR4 and /LIST: no search, the report says so
    for flag in ("/FSR4", "/LIST"):
        a0 = access_size(t.work)
        l0 = os.path.getsize(t.log) if os.path.isfile(t.log) else 0
        rc = t.raw([t.exe, "/S", "/NOSCAN", flag, "/LOG=" + t.log])
        with open(t.log, "rb") as f:
            f.seek(l0)
            said = f.read().decode("utf-8", "replace")
        t.check("/S /NOSCAN %s: no search (nothing of the test libraries read), the report says \"libraries not searched\"" % flag,
                rc == 0 and not touched(a0) and "libraries not searched (/NOSCAN)" in said, (rc, touched(a0)[:3], said[-400:]))
    rc = t.run("uninstall", exe, "--noscan")
    m.shutil.rmtree(root, ignore_errors=True)
    m.shutil.rmtree(libgame, ignore_errors=True)
