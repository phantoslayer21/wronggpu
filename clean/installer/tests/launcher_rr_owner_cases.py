# SPDX-License-Identifier: MIT
"""Who owns the RR plugin's settings file WrongGPU\\wronggpu.ini (#219), for run_launcher_tests.py (test_rr calls run() at its
end; this file keeps that suite below the 256 KB source limit, as launcher_ignore_cases.py does). WrongGPU's RR plugin saves
the mode the player picks in OptiScaler's menu there (rr_panel.cpp: one "rr_mode=<mode>" line, CRLF): the file stays
WrongGPU's, so a repair does not take it for the player's and an uninstall removes it, with nothing kept in
WrongGPU-originals. A hand edit makes it the player's: it stays, and a repair keeps it as the player's original, which the
uninstall puts back. Generated fixture games only. Written for this project."""
import os
import shutil

from run_installer_tests import read, write, snapshot, diff, sha, kept_runtime, unrecorded_wronggpu_files

SETTINGS = os.path.join("WrongGPU", "wronggpu.ini")
RR_EXTRA = {"bin/x64/nvngx_dlssd.dll": b"MZ synthetic stand-in, not a usable RR model file " * 64,
            "bin/x64/sl.dlss_d.dll": b"MZ synthetic stand-in"}  # the DLSS RR plugin stand-in (test_rr's layout)


def plugin_save(path, mode):
    """What WrongGPU's RR plugin writes when the player picks `mode` in OptiScaler's menu (rr_panel.cpp choose(), through
    wg_settings_reader.h ini_set): the [wronggpu] rr_mode line becomes "rr_mode=<mode>", every line ends with CRLF."""
    lines = read(path).decode("utf-8").replace("\r\n", "\n").split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    sec = ""
    for i, line in enumerate(lines):
        s = line.strip()
        if s.startswith("["):
            sec = s[1:s.find("]")].strip().lower() if "]" in s else s[1:].lower()
        elif sec == "wronggpu" and "=" in s and not s.startswith((";", "#")) and s.split("=", 1)[0].strip().lower() == "rr_mode":
            lines[i] = "rr_mode=" + mode
    write(path, "".join(line + "\r\n" for line in lines).encode("utf-8"))


def record_entry(b, rel):
    """The record's copy entry for rel: (existed, original hash, installed hash), or None."""
    for line in read(os.path.join(b, "wronggpu-setup", "install.record")).decode("utf-8", "replace").replace("\r", "").split("\n"):
        f = line.split("\t")
        if len(f) == 5 and f[0] == "copy" and f[4].lower() == rel.lower():
            return f[1], f[2], f[3]
    return None


def kept_dirs(b):
    return os.path.isdir(os.path.join(b, "WrongGPU-originals"))


def run(t, opti, game):
    print("\n[rr-owner] WrongGPU\\wronggpu.ini: the RR plugin's own mode save stays WrongGPU's (removed at uninstall); a hand edit stays the player's (#219)")
    args = ("--opti=" + opti, "--noscan", "--force", "--experimental")
    # 1. the plugin saves the mode twice (before and after a repair): the file stays WrongGPU's, the uninstall removes it
    root, b, exe = game(t, "Rr Owner Plugin", extra=RR_EXTRA)
    before = snapshot(root)
    rc = t.run("install", exe, *args)
    p = os.path.join(b, SETTINGS)
    if rc != 0 or not os.path.isfile(p):
        t.check("rr-owner: the RR game installs with WrongGPU\\wronggpu.ini", False, (rc, t.j.get("message"), t.logtail()))
        return
    plugin_save(p, "accurate")
    rc = t.run("status", exe)
    t.check("plugin save: status shows the mode picked in the game", rc == 0 and t.j.get("game", {}).get("rr_mode") == "accurate", t.j.get("game"))
    rc = t.run("repair", exe, *args)
    e = record_entry(b, SETTINGS)
    t.check("plugin save, then repair: the file is still WrongGPU's (not backed up as the player's), the mode stays, the record follows, nothing kept",
            rc == 0 and e is not None and e[0] == "0" and e[2] == sha(read(p)) and b"rr_mode=accurate\r\n" in read(p) and not kept_dirs(b),
            (rc, t.j.get("message"), e, read(p), t.j.get("warnings")))
    plugin_save(p, "fast")
    left = kept_runtime(before, root, *unrecorded_wronggpu_files(root))
    rc = t.run("uninstall", exe)
    after = snapshot(root)
    t.check("plugin save, then uninstall: exit 0, not partly, WrongGPU\\wronggpu.ini removed, no WrongGPU-originals, the folder as before",
            rc == 0 and "partly" not in t.j.get("message", "") and not os.path.exists(p) and not kept_dirs(b) and after == left and
            not any(SETTINGS in w for w in t.j.get("warnings", [])), (rc, t.j.get("message"), diff(left, after), t.j.get("warnings")))
    shutil.rmtree(root, ignore_errors=True)

    # 2. a hand edit (spaces around "=", a key of the player's own): the player's file; the uninstall leaves it as it is
    root, b, exe = game(t, "Rr Owner Hand", extra=RR_EXTRA)
    before = snapshot(root)
    rc = t.run("install", exe, *args)
    p = os.path.join(b, SETTINGS)
    hand = read(p).replace(b"rr_mode=fast", b"rr_mode = accurate\r\nmy_note=1") if rc == 0 and os.path.isfile(p) else b""
    if hand:
        write(p, hand)
    left = kept_runtime(before, root, "bin/x64/" + SETTINGS.replace("\\", "/"), *unrecorded_wronggpu_files(root))
    rc = t.run("uninstall", exe)
    after = snapshot(root)
    t.check("hand edit, then uninstall: exit 0, not partly, the file stays with the player's bytes, the warning says changed since install",
            bool(hand) and rc == 0 and "partly" not in t.j.get("message", "") and os.path.isfile(p) and read(p) == hand and after == left and
            "Left in place (changed since install): " + SETTINGS in t.j.get("warnings", []), (rc, t.j.get("message"), diff(left, after), t.j.get("warnings")))
    shutil.rmtree(root, ignore_errors=True)

    # 3. a hand edit, then a repair: kept as the player's original; the uninstall puts it back, byte for byte
    root, b, exe = game(t, "Rr Owner Hand Repair", extra=RR_EXTRA)
    before = snapshot(root)
    rc = t.run("install", exe, *args)
    p = os.path.join(b, SETTINGS)
    hand = read(p) + b"my_note=1\r\n" if rc == 0 and os.path.isfile(p) else b""
    if hand:
        write(p, hand)
    rc = t.run("repair", exe, *args)
    e = record_entry(b, SETTINGS)
    t.check("hand edit, then repair: the edited file is the player's original (existed, its hash), WrongGPU's file in its place",
            bool(hand) and rc == 0 and e is not None and e[0] == "1" and e[1] == sha(hand) and e[2] == sha(read(p)), (rc, t.j.get("message"), e))
    left = kept_runtime(before, root, "bin/x64/" + SETTINGS.replace("\\", "/"), *unrecorded_wronggpu_files(root))
    rc = t.run("uninstall", exe)
    after = snapshot(root)
    left["bin/x64/wronggpu/wronggpu.ini"] = sha(hand)
    t.check("hand edit, repair, uninstall: exit 0, not partly, the player's edited file back byte for byte, no WrongGPU-originals",
            rc == 0 and "partly" not in t.j.get("message", "") and os.path.isfile(p) and read(p) == hand and not kept_dirs(b) and after == left,
            (rc, t.j.get("message"), diff(left, after), t.j.get("warnings")))
    shutil.rmtree(root, ignore_errors=True)
