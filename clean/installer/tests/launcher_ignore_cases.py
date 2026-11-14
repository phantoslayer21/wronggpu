# SPDX-License-Identifier: MIT
"""The ignore lists (src/wg_core/convert_ignore.h) on the install path, for run_launcher_tests.py (which calls run() after
its rc6 cases; this file keeps that suite below the 256 KB source limit). Needs installs, so it runs with the launcher
suite in the GPU window; the CPU-only cases are tests/run_convert_ignore_tests.py and ctest convert-ignore.

Runs in a work folder of its own (its own %LOCALAPPDATA%, so its own test registry root and managed list): the model data
of the suite's other games is never touched, and its last uninstall really is the last game's. Generated fixture games
and synthetic model files only. Written for this project.

    an install converts from the game's own file; that file listed: a repair does not reuse the model data and, with no
    other allowed file, fails (exit 5) with the reason; a second game's install converts again from its own file (the
    listed conversion replaced, its SHA-256 recorded); the well-known list held open: a repair exits 10, nothing changed;
    the last game's uninstall keeps ignore-files.txt byte for byte and says so in its result
"""
import os
import shutil
import time

KEPT = "kept ignore-files.txt (your/integrators' ignore list)"


def _hold(path):
    """The file opened with no sharing (an installer writing it); returns a function that lets it go."""
    import ctypes
    from ctypes import wintypes
    k = ctypes.WinDLL("kernel32", use_last_error=True)
    k.CreateFileW.restype = wintypes.HANDLE
    k.CreateFileW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, wintypes.LPVOID, wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE]
    k.CloseHandle.argtypes = [wintypes.HANDLE]
    h = k.CreateFileW(path, 0x80000000 | 0x40000000, 0, None, 3, 0x80, None)
    if h in (None, wintypes.HANDLE(-1).value):
        raise OSError("could not hold %s (error %d)" % (path, ctypes.get_last_error()))
    return lambda: k.CloseHandle(h)


def run(T, base, opti, game):
    """T: the suite's test class; base: its instance (checks are counted there); game: its fixture-game maker."""
    print("\n[convert-ignore] the ignore lists on the install path: a listed model source, a list held open, the last uninstall")
    import hashlib
    from run_installer_tests import read, write
    work = os.path.join(base.work, "convert-ignore")
    shutil.rmtree(work, ignore_errors=True)
    t = T(base.exe, work, base.model, base.env.get("WRONGGPU_HIP_RUNTIME"))
    t.check = base.check  # one count, one list of failures
    shared = os.path.join(t.env["LOCALAPPDATA"], "WrongGPU")
    lst = os.path.join(shared, "ignore-files.txt")
    src_txt = os.path.join(shared, "model", "model-source.txt")
    flags = ("--opti=" + opti, "--noscan", "--force", "--experimental")

    def said():
        return " ".join([t.j.get("message", ""), t.j.get("error", "")] + list(t.j.get("warnings", [])) + list(t.j.get("notes", [])))

    root_a, b_a, exe_a = game(t, "Ignore A")
    dll_a = os.path.join(b_a, "nvngx_dlss.dll")
    rc = t.run("install", exe_a, *flags)
    src = read(src_txt).decode("utf-8", "replace") if os.path.isfile(src_txt) else ""
    t.check("convert-ignore: an install converts from the game's own file and records its SHA-256",
            rc == 0 and ("source\t" + dll_a).lower() in src.lower() and ("sha256\t" + hashlib.sha256(read(dll_a)).hexdigest()) in src, (rc, said(), src))
    if rc != 0:
        return
    model_before = read(os.path.join(shared, "model", "model.wgcm"))
    write(lst, ("wronggpu-ignore-files 1\r\n# placed by another program's installer (test)\r\n%s\r\n" % dll_a).encode())
    rc = t.run("repair", exe_a, *flags)
    t.check("convert-ignore: its source listed, no other allowed file: the repair does not reuse the model data, exit 5, the reason names the list and line 3",
            rc == 5 and ("is not used: it was converted from " + dll_a).lower() in said().lower() and (lst + ", line 3").lower() in said().lower() and
            "choose the nvngx_dlss.dll that came with one of your own games" in said() and
            read(os.path.join(shared, "model", "model.wgcm")) == model_before, (rc, said()))
    root_b, b_b, exe_b = game(t, "Ignore B", exe="Second.exe", extra={"bin/x64/nvngx_dlss.dll": base.model.file + b" synthetic tail of game B's own copy"})
    dll_b = os.path.join(b_b, "nvngx_dlss.dll")
    rc = t.run("install", exe_b, *flags)
    src = read(src_txt).decode("utf-8", "replace") if os.path.isfile(src_txt) else ""
    t.check("convert-ignore: a second game's install converts again from its own (allowed) file; the listed conversion is replaced, the new SHA-256 recorded",
            rc == 0 and ("source\t" + dll_b).lower() in src.lower() and ("sha256\t" + hashlib.sha256(read(dll_b)).hexdigest()) in src, (rc, said(), src))
    let_go = _hold(lst)
    try:
        t0 = time.time()
        rc = t.run("repair", exe_b, *flags)
        took = time.time() - t0
    finally:
        let_go()
    t.check("convert-ignore: the well-known list held open: the repair exits 10 after the retries, the file and the reason named",
            rc == 10 and lst.lower() in said().lower() and "another program holds it open" in said() and took >= 0.75, (rc, round(took, 2), said()))
    list_bytes = read(lst)
    rc_a = t.run("uninstall", exe_a)
    rc_b = t.run("uninstall", exe_b)
    for _ in range(100):  # the hidden cleanup deletes the moved-aside folder
        if not [n for n in os.listdir(t.env["LOCALAPPDATA"]) if ".cleanup-" in n]:
            break
        time.sleep(0.2)
    t.check("convert-ignore: the last game's uninstall keeps ignore-files.txt byte for byte (the rest of the shared folder goes) and says so",
            rc_a == 0 and rc_b == 0 and os.path.isfile(lst) and read(lst) == list_bytes and sorted(os.listdir(shared)) == ["ignore-files.txt"] and
            any(w.startswith(KEPT) for w in t.j.get("warnings", [])), (rc_a, rc_b, sorted(os.listdir(shared)) if os.path.isdir(shared) else None, t.j.get("warnings")))
    os.remove(lst)
    shutil.rmtree(work, ignore_errors=True)
