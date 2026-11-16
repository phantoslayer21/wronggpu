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


IDENTITY_NAME = "WRONGGPU_TESTING_BUILD"  # the named RCDATA resource only a WRONGGPU_TESTING build links in (CMakeLists.txt)
IDENTITY_TEXT = b"wg-testing-build 1: store libraries, access log, edition download, uninstall root"


def pe_checksum(data, at):
    """The PE image checksum of data (the field at offset at counted as 0): what the linker (/RELEASE) writes."""
    import array
    body = bytearray(data[:at]) + b"\0\0\0\0" + data[at + 4:]
    if len(body) % 2:
        body += b"\0"
    total = sum(array.array("H", bytes(body)))
    s = total % 0xFFFF
    if s == 0 and total:
        s = 0xFFFF
    return (s + len(data)) & 0xFFFFFFFF


def testing_identity(data):
    """Why data is not a WRONGGPU_TESTING build's program, read from its structure ('' when it is one): a valid PE whose
    checksum matches (nothing changed or appended after its link), the identity resource in its resource directory, and the
    test hooks' strings inside its sections (not in data appended to it)."""
    import struct
    try:
        if data[:2] != b"MZ":
            return "not a program (no MZ header)"
        pe = struct.unpack_from("<I", data, 0x3C)[0]
        if data[pe:pe + 4] != b"PE\0\0":
            return "not a PE file"
        nsec, optsize = struct.unpack_from("<H", data, pe + 6)[0], struct.unpack_from("<H", data, pe + 20)[0]
        opt = pe + 24
        magic = struct.unpack_from("<H", data, opt)[0]
        if magic not in (0x10B, 0x20B):
            return "not a PE image"
        csum_at = opt + 64
        csum = struct.unpack_from("<I", data, csum_at)[0]
        if csum == 0 or csum != pe_checksum(data, csum_at):
            return "its PE checksum is missing or does not match (changed after its link, or not a testing build)"
        dirs = opt + (96 if magic == 0x10B else 112)
        rsrc_rva, rsrc_size = struct.unpack_from("<II", data, dirs + 2 * 8)
        sections = []
        for i in range(nsec):
            s = opt + optsize + 40 * i
            vsize, va, rsize, raw = struct.unpack_from("<IIII", data, s + 8)
            sections.append((va, max(vsize, rsize), raw, rsize))
        end = max((raw + rsize for _, _, raw, rsize in sections), default=0)
        image = data[:end]
        if [m for m in TEST_HOOKS if m.encode("utf-16-le") not in image]:
            return "the test hooks' strings are not in its sections"

        def off(rva):
            for va, size, raw, rsize in sections:
                if va <= rva < va + size and rva - va < rsize:
                    return raw + rva - va
            raise ValueError("rva %x outside the sections" % rva)

        if not rsrc_rva:
            return "no resources"
        base = off(rsrc_rva)

        def entries(at):
            named, ids = struct.unpack_from("<HH", data, at + 12)
            for k in range(named + ids):
                name, target = struct.unpack_from("<II", data, at + 16 + 8 * k)
                yield name, target

        for name, target in entries(base):
            if name != 10 or not target & 0x80000000:  # RT_RCDATA, a sub-directory
                continue
            for name2, target2 in entries(base + (target & 0x7FFFFFFF)):
                if not name2 & 0x80000000:
                    continue
                n_at = base + (name2 & 0x7FFFFFFF)
                length = struct.unpack_from("<H", data, n_at)[0]
                if data[n_at + 2:n_at + 2 + 2 * length].decode("utf-16-le") != IDENTITY_NAME or not target2 & 0x80000000:
                    continue
                for _, leaf in entries(base + (target2 & 0x7FFFFFFF)):
                    rva, size = struct.unpack_from("<II", data, base + (leaf & 0x7FFFFFFF))
                    if data[off(rva):off(rva) + size].startswith(IDENTITY_TEXT):
                        return ""
        return "no %s resource (not a WRONGGPU_TESTING build)" % IDENTITY_NAME
    except (struct.error, ValueError, UnicodeDecodeError) as e:
        return "not a readable PE file (%s)" % e


def missing_test_hooks(path):
    """Why the program at path is not a WRONGGPU_TESTING build with the test isolation hooks, as a list (empty: it is one)."""
    try:
        data = read(path)
    except OSError as e:
        return ["not readable (%s)" % e]
    why = testing_identity(data)
    return [why] if why else []


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
            raise SystemExit("%s %s is not a WRONGGPU_TESTING build with the test isolation hooks (%s): the suite never "
                             "runs it (build it with -DWRONGGPU_TESTING=ON)" % (what, path, "; ".join(missing)))
        _hooks_ok[key] = True


def reparse_in_path(path):
    """The first folder of path (path itself and every folder above it that exists) that is a junction, a symbolic link or
    another reparse point; None when there is none."""
    p = os.path.abspath(path)
    while True:
        if os.path.lexists(p):
            st = os.lstat(p)
            if os.path.islink(p) or getattr(st, "st_file_attributes", 0) & 0x400:  # FILE_ATTRIBUTE_REPARSE_POINT
                return p
        up = os.path.dirname(p)
        if up == p:
            return None
        p = up


def fresh_temp(work):
    """<work>\\tmp, the launcher's TEMP and TMP: refused when it or a folder above it is a reparse point (it could lead
    outside the work folder), else removed and made anew, empty."""
    tmp = os.path.join(os.path.abspath(work), "tmp")
    bad = reparse_in_path(tmp)
    if bad:
        raise SystemExit("the suite's temporary folder %s is not a plain folder (%s is a junction or link): not run" % (tmp, bad))
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(tmp)
    return tmp


CANARIES = ("wg-test-canary.exe", "wg-test-canary-strings.exe")
BUILD_LIST = "wg-test-build.sha256"  # written by the build next to the launcher: SHA-256 of WrongGPU.exe and the canaries


def verified_canaries(exe):
    """The canaries built in THIS build: their names in <launcher dir>\\wg-test-build.sha256 (written by the testing build
    after it links WrongGPU.exe, CMakeLists.txt), whose WrongGPU.exe line is this launcher's SHA-256 and whose canary lines
    match the files. Returns ({name: bytes}, why not); a file that is not verified is never started."""
    import hashlib
    d = os.path.dirname(os.path.abspath(exe))
    try:
        lines = read(os.path.join(d, BUILD_LIST)).decode("ascii").split()
    except (OSError, UnicodeDecodeError) as e:
        return {}, "no build list %s (%s)" % (BUILD_LIST, e)
    listed = dict(zip(lines[1::2], lines[0::2]))
    if listed.get("WrongGPU.exe") != hashlib.sha256(read(exe)).hexdigest():
        return {}, "the build list is not this launcher's (another build)"
    out = {}
    for name in CANARIES:
        p = os.path.join(d, name)
        data = read(p) if os.path.isfile(p) else b""
        if not data or listed.get(name) != hashlib.sha256(data).hexdigest():
            return {}, "%s is missing or not the one this build made" % name
        out[name] = data
    return out, ""


def junction_call(t, exe, verb, work):
    """The data-root junction test's launcher call without --log (its log goes to TEMP\\WrongGPU-junction-<pid>.log, TEMP being
    the work folder's tmp): the preflight first, as for every launcher call. Returns (exit code, the log's text)."""
    import subprocess
    import launcher_isolation as iso
    out = os.path.join(work, verb + "-result.json")
    astart = iso.access_size(t.work)
    require_test_hooks(exe)  # a testing build only, before it starts
    proc = subprocess.Popen([exe, verb, "--out=" + out], env=t.env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    proc.communicate(timeout=180)
    log = os.path.join(t.env["TEMP"], "WrongGPU-junction-%d.log" % proc.pid)
    assert os.path.abspath(log).lower().startswith(os.path.abspath(t.work).lower() + os.sep), log
    iso.check_isolation(log if os.path.isfile(log) else None, t.work, 0, (verb,), astart)  # (#243: every launcher call)
    text = read(log).decode("utf-8", "replace") if os.path.isfile(log) else ""
    if os.path.isfile(log):
        os.remove(log)
    return proc.returncode, text


def rr_free(folder, record):
    """No RR settings file in an RR-free install: neither WrongGPU\\wronggpu.ini on disk (case-insensitively: on Windows the
    payload's own lower-case wronggpu\\ folder is the same folder, and its kernels are allowed) nor its record line."""
    return not os.path.isfile(os.path.join(folder, "WrongGPU", "wronggpu.ini")) and "wronggpu\\wronggpu.ini" not in record.lower()


def self_checks(t):
    """The suite's own guards on synthetic files, before any launcher call: a program without the test isolation hooks is
    refused before it runs (nothing started, nothing logged), the launcher under test has the hooks, and the RR-free check
    finds an RR settings file that landed in the payload's lower-case wronggpu\\ folder."""
    print("\n[self-checks] the suite's guards: only testing builds run; the RR-free check")
    import subprocess
    import sys
    d = os.path.join(t.work, "self-checks")
    shutil.rmtree(d, ignore_errors=True)
    os.makedirs(d)
    t.check("preflight: the launcher under test is a WRONGGPU_TESTING build (its identity resource, a matching PE checksum, the hooks)",
            not missing_test_hooks(t.exe), missing_test_hooks(t.exe))
    # programs that are not testing builds, each leaving <program>.started when it starts (tests/canary.cpp; built with the
    # launcher in a testing build): one plain, one carrying the hooks' strings. Only the canaries THIS build made are ever
    # started (their SHA-256 in the build list next to the launcher, with this launcher's), by their full path, with the
    # suite's own child environment: first directly, to show they do start
    import hashlib
    size = lambda: os.path.getsize(t.log) if os.path.isfile(t.log) else 0
    canaries, why_not = verified_canaries(t.exe)
    t.check("negative controls: the canaries are this build's (wg-test-build.sha256 next to the launcher names this launcher "
            "and the canaries' bytes)", bool(canaries), why_not)
    for name, data in canaries.items():
        canary = os.path.join(d, name)
        write(canary, data)
        started = canary + ".started"
        if hashlib.sha256(read(canary)).digest() != hashlib.sha256(data).digest():
            t.check("negative control %s: the copy is the verified file" % name, False)
            continue
        subprocess.run([canary], env=t.env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=60)
        runs = os.path.isfile(started)
        if runs:
            os.remove(started)
        before, why, jwhy, rwhy = size(), "", "", ""
        try:
            t.run("version", exe=canary)
        except SystemExit as e:
            why = str(e)
        try:
            junction_call(t, canary, "remove-data", d)  # the data-root junction test's path
        except SystemExit as e:
            jwhy = str(e)
        try:
            t.raw([canary, "/S"])  # a program run as it is (the restorer's path)
        except SystemExit as e:
            rwhy = str(e)
        entry = []
        suite = os.path.join(os.path.dirname(os.path.abspath(__file__)), "run_launcher_tests.py")
        for args in (["--launcher", canary], ["--launcher", t.exe, "--no-direct-launcher", canary]):  # the suite's entry points
            p = subprocess.run([sys.executable, suite] + args + ["--work", os.path.join(d, "entry")], env=dict(os.environ),
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=300)
            entry.append(p.returncode != 0 and b"not a WRONGGPU_TESTING build" in p.stderr + p.stdout)
        t.check("negative control %s: it starts when run directly (its start marker); the suite never starts it (a launcher call, "
                "the junction test's call, a program run as it is, --launcher, --no-direct-launcher): refused before it runs, no start "
                "marker, nothing logged" % name,
                runs and all("not a WRONGGPU_TESTING build" in w for w in (why, jwhy, rwhy)) and entry == [True, True] and
                not os.path.exists(started) and size() == before, (runs, why, jwhy, rwhy, entry))
    # the real launcher changed after its link: one byte, or data appended (the hooks' strings and the identity text)
    data = read(t.exe)
    mid = len(data) // 2
    mixed = {"one byte changed": data[:mid] + bytes([data[mid] ^ 0xFF]) + data[mid + 1:],
             "data appended": data + b"".join(m.encode("utf-16-le") for m in TEST_HOOKS) + IDENTITY_TEXT}
    if "wg-test-canary-strings.exe" in canaries:  # the hooks' strings and a valid checksum, but no identity resource
        import struct
        c = bytearray(canaries["wg-test-canary-strings.exe"])
        at = struct.unpack_from("<I", c, 0x3C)[0] + 24 + 64
        struct.pack_into("<I", c, at, pe_checksum(bytes(c), at))
        mixed["the hooks' strings and a valid checksum, no identity resource"] = bytes(c)
    refused = {k: bool(testing_identity(v)) for k, v in mixed.items()}
    t.check("preflight: a launcher changed after its link (a byte, or the hooks' strings and the identity appended), or a program "
            "with the strings and a valid checksum but no identity resource, is refused",
            all(refused.values()), refused)
    # WRONGGPU_HIP_RUNTIME only from --hip: an inherited value (another HIP library) is never the launcher's
    conflict = os.path.join(d, "conflict", "amdhip64_7.dll")
    env = t.child_env(dict(os.environ, WRONGGPU_HIP_RUNTIME=conflict))
    saved = t.env
    t.env = dict(env, **{k: v for k, v in saved.items() if k in ("WRONGGPU_TEST_LIBRARIES", "WRONGGPU_TEST_ACCESS_LOG", "WRONGGPU_TEST_RUN_ID")})
    try:
        t.run("hipcheck")
    finally:
        t.env = saved
    dll = t.j.get("dll", "")
    t.check("HIP runtime: an inherited WRONGGPU_HIP_RUNTIME is dropped, only --hip sets it (hipcheck never uses the inherited one)",
            env.get("WRONGGPU_HIP_RUNTIME") == t.hip and conflict.lower() not in dll.lower() and (not t.hip or dll.lower() == t.hip.lower()),
            (env.get("WRONGGPU_HIP_RUNTIME"), t.hip, t.j))
    # the temporary folder: a junction in its place is refused (fresh_temp), never followed
    jw = os.path.join(d, "junction-work")
    os.makedirs(os.path.join(d, "elsewhere"))
    os.makedirs(jw)
    linked = subprocess.run(["cmd.exe", "/d", "/c", "mklink", "/J", os.path.join(jw, "tmp"), os.path.join(d, "elsewhere")],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE).returncode == 0
    jwhy = ""
    try:
        fresh_temp(jw)
    except SystemExit as e:
        jwhy = str(e)
    t.check("the suite's temporary folder: a junction in its place (or above it) is refused, never followed or emptied",
            linked and "junction or link" in jwhy and os.path.isdir(os.path.join(d, "elsewhere")), (linked, jwhy))
    if os.path.lexists(os.path.join(jw, "tmp")):
        os.rmdir(os.path.join(jw, "tmp"))  # (removes the junction only)
    b = os.path.join(d, "Game", "bin")
    write(os.path.join(b, "wronggpu", "kernels", "enc.hsaco"), b"kernel")
    ok_kernels = rr_free(b, "file wronggpu\\kernels\\enc.hsaco")
    write(os.path.join(b, "wronggpu", "wronggpu.ini"), b"[wronggpu]\r\n")  # WrongGPU\wronggpu.ini, in the lower-case folder
    t.check("RR-free check: the payload's wronggpu\\kernels alone is RR-free; WrongGPU\\wronggpu.ini in that lower-case folder, or "
            "its record line, is not", ok_kernels and not rr_free(b, "") and not rr_free(os.path.join(d, "empty"), "file WrongGPU\\wronggpu.ini"),
            ok_kernels)
    shutil.rmtree(d, ignore_errors=True)
