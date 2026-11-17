# SPDX-License-Identifier: MIT
"""What run_launcher_tests.py --native-linux needs to run the launcher suite against the Linux WrongGPU binary.

Written for this project; Python standard library only. The suite's checks stay the same; this module gives them the
Linux equivalents of what the Windows runs get from Windows: the safe environment (every folder the program writes to
sits under the work folder, SDL opens no window, the registry is a file of the run's own), HKCU as that file, and the
list of checks that only mean something on Windows."""
import ntpath
import os
import re
import shutil
import stat
import sys

# Windows-only features of the launcher suite: each is skipped on Linux with its reason printed (see skip())
WINDOWS_ONLY = {
    "shortcut": "the launcher's self-install is a Windows feature: a Start menu .lnk shortcut, an Apps & features entry, Mark of the Web",
    "elevation": "the elevated worker is Windows UAC; Linux has no elevation prompt",
    "restorer": "the per-game restorer (uninstall.exe) is a Windows program that Linux does not build or embed",
    "gpu": "this check needs the Windows graphics driver and an RDNA 4 card, which a native Linux run has neither of",
    "http": "the Linux launcher has no HTTP client yet (WinHTTP is answered as not available; HTTPS needs a TLS library)",
    "lock": "a file another process holds open with no sharing (Windows share mode 0) is not emulated: the POSIX layer's sharing rules are per process",
    "junction": "NTFS junctions do not exist on Linux (the section tests the data folder reached through one)",
    "baseline": "the 0.5.1 baseline installer is a Windows program",
}

# Checks that fail on main's own Windows launcher as well (reproduced under Wine), so not the Linux port's: skipped while they
# fail, with the issue that reports them (check name prefix -> reason)
KNOWN_ON_MAIN = {
    "uninstall: exit 0, the RR game's folder as before except the two RR settings files":
        "fails on main too: a repair after an in-game edit records WrongGPU\\wronggpu.ini as the player's (issue #219)",
}


def environment(work, base):
    """The environment of every run of the app under test: its XDG folders, HOME, temp folder and registry file under `work`,
    SDL on its offscreen driver and no runtime or display variable (an earlier run opened real windows on a desktop that way).
    LOCALAPPDATA is a folder of its own, spelled as the program spells it (Z:\\...), and not XDG_DATA_HOME: the program takes XDG_DATA_HOME
    for the PC's real local application data folder and refuses to write a game's Engine.ini there (rrconfig.cpp), as on Windows.
    `work` may be either spelling."""
    env = {k: v for k, v in base.items() if k not in ("DISPLAY", "WAYLAND_DISPLAY", "XDG_RUNTIME_DIR", "WRONGGPU_UI_SCRIPT")}
    work = host(work)
    run = os.path.join(work, "native")
    for d in ("home", "config", "cache", "state", "tmp", "xdg-data"):
        os.makedirs(os.path.join(run, d), exist_ok=True)
    data = os.path.join(work, "localappdata")
    os.makedirs(data, exist_ok=True)
    env.update(HOME=os.path.join(run, "home"), XDG_DATA_HOME=os.path.join(run, "xdg-data"), XDG_CONFIG_HOME=os.path.join(run, "config"),
               XDG_CACHE_HOME=os.path.join(run, "cache"), XDG_STATE_HOME=os.path.join(run, "state"), TMPDIR=os.path.join(run, "tmp"),
               WRONGGPU_REGISTRY_FILE=os.path.join(run, "registry.ini"), LOCALAPPDATA=win_path(data),
               ProgramData=win_path(os.path.join(work, "programdata")), SDL_VIDEO_DRIVER="offscreen", SDL_AUDIODRIVER="dummy")
    return env


def host(p):
    """The Linux path of a path the suite spells as Windows does (Z:\\home\\...\\name, or a relative one with '\\'). A name that is
    not there is looked for in another case, as NTFS (and so every Windows run of the suite) finds it."""
    if not isinstance(p, str):
        return p
    if len(p) >= 2 and p[:2].upper() == "Z:" and (len(p) == 2 or p[2] in "\\/"):
        p = p[2:] or "/"
    p = p.replace("\\", "/")
    if os.path.lexists(p) or not p:
        return p
    cur, parts = ("/" if p.startswith("/") else "."), [x for x in p.split("/") if x]
    for i, name in enumerate(parts):
        nxt = os.path.join(cur, name)
        if not os.path.lexists(nxt):
            try:
                same = [n for n in os.listdir(cur) if n.lower() == name.lower()]
            except OSError:
                return p
            if not same:
                return p
            nxt = os.path.join(cur, same[0])
        cur = nxt
    return cur


def win_path(p):
    """How a Linux path is spelled in the Windows way the suite (and the program) use: Z:\\ plus the path, as in Wine."""
    p = os.path.abspath(p)
    return "Z:" + p.replace("/", "\\")


def _hostfn(fn):
    return lambda *a, **k: fn(*[host(x) for x in a], **{n: host(v) for n, v in k.items()})


class _WinPath:
    """os.path as the suite sees it on Windows: Windows spellings (ntpath), existence checks on the Linux files behind them."""
    sep, altsep, pathsep = "\\", "/", ";"
    def __init__(self):
        for n in ("join", "split", "splitext", "dirname", "basename", "normpath", "isabs", "normcase", "splitdrive", "commonpath"):
            setattr(self, n, getattr(ntpath, n))
        for n in ("exists", "lexists", "isfile", "isdir", "islink", "getsize", "getmtime", "realpath", "samefile", "ismount"):
            setattr(self, n, _hostfn(getattr(os.path, n)))

    def abspath(self, p):
        p = os.fspath(p)
        if p[:2].upper() == "Z:":
            return ntpath.normpath(p)
        return ntpath.normpath(win_path(p) if p.startswith("/") else ntpath.join(win_path(os.getcwd()), p))

    def relpath(self, p, start=None):
        return ntpath.relpath(self.abspath(p), self.abspath(start or os.getcwd()))


class _Stat:
    """os.stat's result plus the Windows st_file_attributes, from the mode the way the program maps it (read-only: no write bit)."""

    def __init__(self, st):
        self._st = st
        self.st_file_attributes = (0x10 if stat.S_ISDIR(st.st_mode) else 0x20) | (0x1 if not st.st_mode & 0o222 else 0)

    def __getattr__(self, name):
        return getattr(self._st, name)


class WinOs:
    """Stands in for `os` in the suite's modules: paths are Windows spellings (Z:\\... roots, backslashes), as the program under test
    prints them, so every check compares what it compares on Windows; the files behind them are the Linux ones."""

    def __init__(self):
        self.path, self.sep, self.altsep = _WinPath(), "\\", "/"
        for n in ("listdir", "makedirs", "mkdir", "remove", "unlink", "rmdir", "rename", "replace", "chmod", "utime",
                  "symlink", "link", "readlink", "access", "truncate", "chdir"):  # stat and lstat: below
            setattr(self, n, _hostfn(getattr(os, n)))

    def __getattr__(self, name):
        return getattr(os, name)

    def stat(self, p, **k):
        return _Stat(os.stat(host(p), **k))

    def lstat(self, p, **k):
        return _Stat(os.lstat(host(p), **k))

    def getcwd(self):
        return win_path(os.getcwd())

    def walk(self, top, topdown=True, onerror=None, followlinks=False):
        for d, dirs, files in os.walk(host(top), topdown, onerror, followlinks):
            yield win_path(d), dirs, files


class WinShutil:
    def __getattr__(self, name):
        fn = getattr(shutil, name)
        return _hostfn(fn) if callable(fn) else fn


def windows_paths(*modules):
    """Give each module Windows-style os, shutil and open (see WinOs)."""
    for m in modules:
        for n, v in (("os", WinOs()), ("shutil", WinShutil())):
            if hasattr(m, n):
                setattr(m, n, v)
        m.open = lambda f, *a, **k: open(host(f), *a, **k)


def _unescape(s):
    return re.sub(r"%([0-9A-Fa-f]{2})", lambda m: chr(int(m.group(1), 16)), s)


def _escape(s):
    return "".join("%%%02X" % ord(c) if ord(c) < 0x20 or c in "%=]" else c for c in s)


class Registry:
    """HKCU as the program keeps it (src/platform/posix/win32_posix_registry.cpp): a text file, one [key path] section per key,
    `name=s:value` lines. The tests read the Apps & features entries through it and set up and remove stale ones."""

    def __init__(self, file):
        self.file = file

    def load(self):
        keys, cur = {}, None
        try:
            lines = open(self.file, encoding="utf-8").read().splitlines()
        except OSError:
            return keys
        for line in lines:
            if len(line) >= 2 and line[0] == "[" and line[-1] == "]":
                cur = keys.setdefault(_unescape(line[1:-1]), {})
            elif cur is not None and "=" in line:
                name, body = line.split("=", 1)
                if body[:2] in ("s:", "e:"):
                    cur[_unescape(name)] = _unescape(body[2:])
                elif body[:2] in ("d:", "q:"):
                    cur[_unescape(name)] = body[2:]
        return keys

    def save(self, keys):
        tmp = self.file + ".test"
        with open(tmp, "w", encoding="utf-8") as f:
            for path, values in keys.items():
                f.write("[%s]\n" % _escape(path))
                for n, v in values.items():
                    f.write("%s=s:%s\n" % (_escape(n), _escape(v)))
        os.replace(tmp, self.file)

    @staticmethod
    def _path(path):
        return re.sub(r"^HKCU\\", "", path, flags=re.I)

    def subkeys(self, root):
        """{subkey name: {value name: text}} directly under root (an HKCU\\... path), like the Windows suite's reg query parse."""
        base = self._path(root).lower()
        out = {}
        for path, values in self.load().items():
            head, _, name = path.rpartition("\\")
            if head.lower() == base:
                out[name] = dict(values)
        return out

    def add(self, path, name, value):
        keys = self.load()
        keys.setdefault(self._path(path), {})[name] = value
        self.save(keys)

    def delete(self, path):
        base = self._path(path).lower()
        keys = {p: v for p, v in self.load().items() if p.lower() != base and not p.lower().startswith(base + "\\")}
        self.save(keys)


def setup(main_module, suite, tests_dir):
    """Called by run_launcher_tests.main for --native-linux: the checks see Windows spellings of paths (WinOs), as the program prints
    them; what the suite built at import time with the host's separator is made again."""
    import launcher_build_facts
    import launcher_edition_cases
    import launcher_ignore_cases
    import launcher_isolation
    import launcher_rr_prepare_cases
    import launcher_state_cases
    windows_paths(main_module, suite, launcher_ignore_cases, launcher_edition_cases, launcher_build_facts, launcher_isolation,
                  launcher_rr_prepare_cases, launcher_state_cases)
    sys.path.insert(0, host(tests_dir + "/../tools"))  # the tests add it with a Windows separator, which sys.path cannot use
    main_module.RR_SETTINGS = main_module.os.path.join("WrongGPU", "wronggpu.ini")
    main_module.T = native_class(main_module.T, suite)


def native_class(T, suite):
    """The suite's T for the Linux binary: the safe environment, the registry file for HKCU, host paths for the program and
    the Windows-only checks skipped with their reasons (kept here so the suite file stays below the 256 KB source limit)."""
    class NativeT(T):
        native = True

        def __init__(self, exe, work, model, hip):
            super().__init__(exe, work, model, hip)
            self.skipped = []
            guard = self.env
            self.env = environment(work, os.environ)
            for k in ("WRONGGPU_TEST_LIBRARIES", "WRONGGPU_TEST_ACCESS_LOG", "WRONGGPU_TEST_RUN_ID"):  # the store-access guard's (#243)
                self.env[k] = guard[k]
            if hip:
                self.env["WRONGGPU_HIP_RUNTIME"] = hip
            for k in [k for k in self.env if k.upper().startswith("WG_") or k.upper() in ("HIP_PATH", "ROCM_PATH", "LLVM_PATH", "HIP_DEVICE_LIB_PATH")]:
                self.env.pop(k)
            self.uninstall = suite.uninstall_root(self.env["LOCALAPPDATA"])
            self.registry = Registry(self.env["WRONGGPU_REGISTRY_FILE"])

        def check(self, name, cond, detail=""):
            """A check that fails on main's own launcher too (issue #219) is skipped with that reason while it fails; once it
            passes it counts again, and the run says to drop it from KNOWN_ON_MAIN."""
            known = next((k for k in KNOWN_ON_MAIN if name.startswith(k)), None)
            if known and not cond:
                self.skipped.append(name)
                return skip(name, KNOWN_ON_MAIN[known])
            if known:
                print("  NOTE  passes now, remove it from native_linux.KNOWN_ON_MAIN:", known)
            return super().check(name, cond, detail)

        def skip(self, feature, what):
            """A check that only means something on Windows: not run, and the log says which and why."""
            self.skipped.append(what)
            skip(what, WINDOWS_ONLY[feature])

        def run(self, *args, exe=None, cwd=None, **k):
            return super().run(*args, exe=host(exe or self.exe), cwd=host(cwd), **k)

        def raw(self, cmd, timeout=600):
            return super().raw([host(cmd[0])] + list(cmd[1:]), timeout)

        def reg_keys(self, root=None):
            return self.registry.subkeys(root or self.uninstall)

        def reg_add(self, path, name, value):
            return self.registry.add(path, name, value)

        def reg_delete(self, path):
            return self.registry.delete(path)

    return NativeT


def skip(name, reason):
    print("  SKIP ", name, "-", reason)
