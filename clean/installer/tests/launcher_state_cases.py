# SPDX-License-Identifier: MIT
"""#219: the game's state folder (wronggpu-setup\\settings.ini, wg_state.h) for run_launcher_tests.py, which calls run() in
its place in the suite (this file keeps that suite below the 256 KB source limit, as launcher_edition_cases.py does).
wronggpu.ini and WrongGPU\\wronggpu.ini hold the install's defaults and are the player's: no launcher operation rewrites them
after the install. What the launcher and the in-game menu save goes into the state folder, which decides over them, which is
WrongGPU's by its bytes (WrongGPU's stamp: uninstall removes a stamped file, keeps an edited or unknown one), and which
uninstall, repair and update only change while no other program has WrongGPU's files open. The in-game menu's save is written
here the way the RR plugin writes it (wg_state.h SetSettings: the key set, the other lines kept, the stamp kept on a file that
was WrongGPU's: menu_save); the plugin's own save is tested with the real plugin in clean/rr_f/proxy/tests/test_hook_dll.cpp.
The install's seed files (wronggpu-asi.ini, WrongGPU\\wronggpu.ini) are created only where nothing is: a file that appears
at either boundary (after the check, after the record: WRONGGPU_TEST_CREATE_AT) is the player's. Written for this project."""
import os
import shutil
import subprocess
import sys

from run_installer_tests import read, sha, snapshot, diff, write, kept_runtime, unrecorded_wronggpu_files

HOLD = r"""
import ctypes, sys
k = ctypes.WinDLL("kernel32", use_last_error=True)
k.CreateFileW.restype = ctypes.c_void_p
k.LoadLibraryExW.restype = ctypes.c_void_p
path, how = sys.argv[1], sys.argv[2]
if how == "open":  # another program (antivirus, a backup tool) reads the file without sharing writes
    h = k.CreateFileW(path, 0x80000000, 1, None, 3, 0, None)
    ok = h not in (None, ctypes.c_void_p(-1).value)
else:  # the game has it loaded (mapped as an image; nothing of it runs)
    h = k.LoadLibraryExW(path, None, 0x1) or k.LoadLibraryExW(path, None, 0x20)
    ok = bool(h)
print("held" if ok else "failed %d" % ctypes.get_last_error(), flush=True)
sys.stdin.read()
"""


STAMP = b"; wronggpu-written "


def stamp_check(body, name, install_id):
    """wg_state.h StampCheck: FNV-1a 64 of the bytes, then of "\\n<name, lower case>\\n<install id>"; 16 lower-case hex digits."""
    h = 1469598103934665603
    for c in body + b"\n" + name.lower().encode() + b"\n" + install_id:
        h = ((h ^ c) * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return b"%016x" % h


def install_id(state_dir):
    """The record's install-id (wg_state.h InstallIdOf); b"" when none."""
    rp = os.path.join(state_dir, "install.record")
    for line in (read(rp).split(b"\r\n") if os.path.isfile(rp) else []):
        if line.startswith(b"install-id\t"):
            return line[len(b"install-id\t"):]
    return b""


def stamped_by_wronggpu(data, name, iid):
    if not iid:
        return False
    t = data[:-1] if data.endswith(b"\n") else data
    t = t[:-1] if t.endswith(b"\r") else t
    start = t.rfind(b"\n") + 1
    last = t[start:]
    return last.startswith(STAMP) and len(last) == len(STAMP) + 16 and last[len(STAMP):] == stamp_check(data[:start], name, iid)


def menu_save(path, k, v):
    """The in-game menu's save as wg_state.h SetSettings makes it: k=v set (its line rewritten, or added at the end), every
    other line kept with its own line end, only the valid stamp taken off, put back only on a file that was WrongGPU's (or new)
    of an install with an id."""
    name, iid = os.path.basename(path), install_id(os.path.dirname(path))
    data = read(path) if os.path.isfile(path) else None
    ours = bool(iid) and (data is None or stamped_by_wronggpu(data, name, iid))
    text = (data or b"").decode("utf-8")
    if data is not None and stamped_by_wronggpu(data, name, iid):
        text = text[:text.rstrip("\r\n").rfind("\n") + 1]  # (the valid stamp line off)
    lines = text.splitlines(keepends=True)
    keys = [l.split("=", 1)[0].strip() if "=" in l and not l.lstrip().startswith((";", "#", "[")) else None for l in lines]
    if k in keys:
        i = keys.index(k)
        lines[i] = k + "=" + v + lines[i][len(lines[i].rstrip("\r\n")):]
    else:
        if lines and not lines[-1].endswith("\n"):
            lines[-1] += "\r\n"
        lines.append(k + "=" + v + "\r\n")
    body = "".join(lines).encode("utf-8")
    write(path, body + STAMP + stamp_check(body, name, iid) + b"\r\n" if ours else body)


def seed_appears(t, m, name, rel, extra=None):
    """A seed file that appears during the install (WRONGGPU_TEST_CREATE_AT: after its check, after the record is written): kept
    byte for byte, not recorded, said; the uninstall keeps it and lists it."""
    inst = ("--opti=" + t.opti, "--noscan", "--force", "--experimental")
    made = b"; made by the player during the install (test)\r\n"  # (record.cpp RestoreStep writes these bytes)
    for step in ("seed-planned", "seed-recorded"):
        root, b, exe = m.game(t, "State Seed %s %s" % (name, step), extra=extra)
        before = snapshot(root)
        f = os.path.join(b, rel)
        t.env["WRONGGPU_TEST_CREATE_AT"], t.env["WRONGGPU_TEST_CREATE_FILE"] = step, f
        try:
            rc = t.run("install", exe, *inst)
        finally:
            t.env.pop("WRONGGPU_TEST_CREATE_AT", None)
            t.env.pop("WRONGGPU_TEST_CREATE_FILE", None)
        t.check("%s appears during the install (%s): the install keeps it byte for byte, does not record it, and says so" % (rel, step),
                rc == 0 and os.path.isfile(f) and read(f) == made and m.record_entry_sha(b, rel) is None and
                any(w.startswith("Left in place (appeared during the install") and rel in w for w in t.j.get("warnings", [])),
                (rc, read(f) if os.path.isfile(f) else None, m.record_entry_sha(b, rel), t.j.get("warnings"), t.logtail(800)))
        left = kept_runtime(before, root, os.path.join("bin", "x64", rel), *unrecorded_wronggpu_files(root))
        rc = t.run("uninstall", exe)
        t.check("...and the uninstall keeps it (the player's) byte for byte; everything else as before",
                rc == 0 and read(f) == made and snapshot(root) == left, (rc, diff(left, snapshot(root)), t.j.get("warnings")))
        shutil.rmtree(root, ignore_errors=True)


def record_060(t, m, b):
    """The install record as 0.6.0 wrote it for this install, built from this install's record in 0.6.0's format: the
    WrongGPU 0.6.0 source (commit f3aa109, clean/installer/src/wg_core/record.cpp SaveRecord, line 141) writes no install-id
    line, and 0.6.0 seeded wronggpu-asi.ini without recording it (clean/installer/src/wg_core/optiscaler.h, lines 77-87): so
    version 0.6.0, no install-id, no wronggpu-asi.ini entry; and 0.6.0 had no state files (settings.ini goes)."""
    rp = os.path.join(b, "wronggpu-setup", "install.record")
    lines = read(rp).decode("utf-8").split("\r\n")
    lines = ["version\t0.6.0" if l.startswith("version\t") else l for l in lines
             if not l.endswith("\twronggpu-asi.ini") and not l.startswith("install-id\t")]
    write(rp, "\r\n".join(lines).encode("utf-8"))
    for f in ("settings.ini", "model-choice.ini"):
        if os.path.isfile(os.path.join(b, "wronggpu-setup", f)):
            os.remove(os.path.join(b, "wronggpu-setup", f))


def failed_runs(t, m):
    """A repair or install that fails takes back only what it made (the run journal): the player's files in the state folder
    (a note, a folder, a kept temporary file, a copy of the settings file) stay byte for byte, whichever way the run fails."""
    inst = ("--opti=" + t.opti, "--noscan", "--force", "--experimental")
    root, b, exe = m.game(t, "State Failed Runs")
    st = os.path.join(b, "wronggpu-setup")
    rc = t.run("install", exe, *inst)
    mine = {"notes.txt": b"the player's notes", os.path.join("mine", "x.txt"): b"a folder of the player's",
            "settings.ini.wgtmp-player": b"player_key=1\r\n"}
    for rel, data in mine.items():
        write(os.path.join(st, rel), data)
    if os.path.isfile(os.path.join(st, "settings.ini")):
        mine["settings-copy.ini"] = read(os.path.join(st, "settings.ini"))  # (a copy of WrongGPU's file: the player's)
        write(os.path.join(st, "settings-copy.ini"), mine["settings-copy.ini"])

    def same():
        return all(os.path.isfile(os.path.join(st, r)) and read(os.path.join(st, r)) == d for r, d in mine.items())
    for how, step in (("a repair that fails after everything was written", "installed"),
                      ("a repair that fails before its record (the earlier install already removed)", "seed-planned")):
        t.env["WRONGGPU_TEST_FAIL_AT"] = step
        try:
            rc2 = t.run("repair", exe, *inst)
        finally:
            t.env.pop("WRONGGPU_TEST_FAIL_AT", None)
        t.check("%s: refused, the player's files in the state folder stay byte for byte" % how,
                rc == 0 and rc2 != 0 and same(), (rc, rc2, [r for r, d in mine.items() if not os.path.isfile(os.path.join(st, r))], t.logtail(1200)))
        rc = t.run("install", exe, *inst)  # (installed again for the next case)
    rc = t.run("uninstall", exe)
    t.check("...the uninstall keeps them too (listed)", rc == 0 and same(), (rc, t.j.get("warnings")))
    t.env["WRONGGPU_TEST_FAIL_AT"] = "seed-planned"
    try:
        rc2 = t.run("install", exe, *inst)  # a new install into the state folder those files are in
    finally:
        t.env.pop("WRONGGPU_TEST_FAIL_AT", None)
    t.check("an install that fails where an earlier uninstall kept the player's files: they stay byte for byte (never a removal of the folder)",
            rc2 != 0 and same() and not os.path.isfile(os.path.join(st, "install.record")), (rc2, os.listdir(st) if os.path.isdir(st) else None))
    shutil.rmtree(root, ignore_errors=True)


def hold(path, how):
    p = subprocess.Popen([sys.executable, "-c", HOLD, path, how], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                         creationflags=0x08000000)
    return p, p.stdout.readline().decode().strip()


def release(p):
    p.stdin.close()
    p.wait(30)


def asi_seed_cases(t, opti, m):
    """The OptiScaler plugin's starting entry (wronggpu-asi.ini, written by a Preset K build on the hosted route): one of the
    install's recorded files, removed by an uninstall while unchanged, kept and listed once the plugin's menu changed it."""
    inst = ("--opti=" + opti, "--noscan", "--force", "--experimental")
    root, b, exe = m.game(t, "State Asi")
    before = snapshot(root)
    asi = os.path.join(b, "wronggpu-asi.ini")
    rc = t.run("install", exe, *inst)
    if rc != 0 or not os.path.isfile(asi):
        t.check("install (hosted route)", rc == 0, (rc, t.j.get("message")))
        print("  skipped: the plugin's wronggpu-asi.ini (a build without Preset K writes none)")
        return
    seeded = read(asi)
    t.check("Preset K build: the plugin's wronggpu-asi.ini is one of the install's recorded files, with its hash",
            m.record_entry_sha(b, "wronggpu-asi.ini") == sha(seeded) and b"selection=wronggpu" in seeded, (m.record_entry_sha(b, "wronggpu-asi.ini"), seeded))
    left = kept_runtime(before, root, *unrecorded_wronggpu_files(root))
    rc = t.run("uninstall", exe)
    t.check("wronggpu-asi.ini unchanged: the uninstall removes it, the folder byte for byte as before, nothing listed for it",
            rc == 0 and not os.path.exists(asi) and snapshot(root) == left and not any("wronggpu-asi.ini" in w for w in t.j.get("warnings", [])),
            (rc, diff(left, snapshot(root)), t.j.get("warnings")))
    rel = os.path.join("bin", "x64", "wronggpu-asi.ini")
    rc = t.run("install", exe, *inst)
    menu = b"; WrongGPU OptiScaler plugin\r\nselection=amd\r\n"  # the plugin's menu: AMD chosen
    write(asi, menu)
    left = kept_runtime(before, root, rel, *unrecorded_wronggpu_files(root))
    rc2 = t.run("uninstall", exe)
    t.check("wronggpu-asi.ini changed by the plugin's menu: the uninstall keeps it as it is and lists it (changed since install); everything else as before",
            rc == 0 and rc2 == 0 and read(asi) == menu and snapshot(root) == left and
            any(w.startswith("Left in place (changed since install)") and "wronggpu-asi.ini" in w for w in t.j.get("warnings", [])),
            (rc, rc2, diff(left, snapshot(root)), t.j.get("warnings")))
    # the player's file there at the next install: kept as it is, not recorded (a repair keeps it too); the uninstall lists it
    rc = t.run("install", exe, *inst)
    rc2 = t.run("repair", exe, *inst)
    t.check("the player's wronggpu-asi.ini at an install and a repair: kept byte for byte, not in the record",
            rc == 0 and rc2 == 0 and read(asi) == menu and m.record_entry_sha(b, "wronggpu-asi.ini") is None, (rc, rc2, read(asi)))
    left = kept_runtime(before, root, rel, *unrecorded_wronggpu_files(root))
    rc = t.run("uninstall", exe)
    t.check("...and the uninstall keeps it and lists it; everything else as before",
            rc == 0 and read(asi) == menu and snapshot(root) == left and any("wronggpu-asi.ini" in w and w.startswith("Left in place") for w in t.j.get("warnings", [])),
            (rc, diff(left, snapshot(root)), t.j.get("warnings")))
    # an update over a 0.6.0 install: its record has no wronggpu-asi.ini entry; the file (0.6.0's seed, or the plugin's menu
    # save since) is not WrongGPU's: the update keeps it byte for byte and does not record it, the uninstall keeps and lists it
    for label, asi_bytes in (("0.6.0's seed bytes", seeded), ("changed by the plugin's menu", menu)):
        rc = t.run("install", exe, *inst)
        write(asi, asi_bytes)
        record_060(t, m, b)
        rc2 = t.run("update", exe, *inst)
        t.check("update over a 0.6.0 record without the seed entry (wronggpu-asi.ini %s): kept byte for byte, not recorded" % label,
                rc == 0 and rc2 == 0 and read(asi) == asi_bytes and m.record_entry_sha(b, "wronggpu-asi.ini") is None,
                (rc, rc2, read(asi), m.record_entry_sha(b, "wronggpu-asi.ini"), t.j.get("message")))
        rc3 = t.run("status", exe)
        said = str(t.j.get("game", {})) + str(t.j.get("warnings", []))
        t.check("...a status (what the launcher shows at every start) does not list it", rc3 == 0 and "wronggpu-asi.ini" not in said, said[-600:])
        left = kept_runtime(before, root, rel, *unrecorded_wronggpu_files(root))
        rc = t.run("uninstall", exe)
        t.check("...the uninstall keeps it and lists it once; everything else as before",
                rc == 0 and read(asi) == asi_bytes and snapshot(root) == left and sum("wronggpu-asi.ini" in w for w in t.j.get("warnings", [])) == 1,
                (rc, diff(left, snapshot(root)), t.j.get("warnings")))
        os.remove(asi)
    shutil.rmtree(root, ignore_errors=True)  # (the later sections, the window's library among them, see only their own games)
    seed_appears(t, m, "Asi", "wronggpu-asi.ini")


def run(t, opti, m):
    """m: the suite's module (its helpers: game, ini, state_kv, state_set, rr_ini, record_entry_sha, RR_SETTINGS, STATE)."""
    print("\n[state] #219: the saved settings in the state folder, wronggpu.ini never rewritten, uninstall/repair/update refused while in use")
    t.opti = opti
    asi_seed_cases(t, opti, m)
    rr_files = {"bin/x64/nvngx_dlssd.dll": b"MZ synthetic stand-in, not a usable RR model file " * 64,
                "bin/x64/sl.dlss_d.dll": b"MZ synthetic stand-in"}  # the DLSS RR plugin stand-in
    seed_appears(t, m, "RR", m.RR_SETTINGS, extra=rr_files)
    failed_runs(t, m)
    inst = ("--opti=" + opti, "--noscan", "--force", "--experimental")
    root, b, exe = m.game(t, "State Game", extra=rr_files)
    before = snapshot(root)
    rc = t.run("install", exe, *inst)
    rrs = os.path.join(b, m.RR_SETTINGS)
    t.check("install of an RR game", rc == 0 and os.path.isfile(rrs), (rc, t.j.get("message"), t.logtail()))
    if rc != 0 or not os.path.isfile(rrs):
        return
    wini0, rr0 = read(os.path.join(b, "wronggpu.ini")), read(rrs)

    # 1. the in-game menu saves Fast, then uninstall: nothing of WrongGPU is left (the #219 case: the save made the repair and
    #    the uninstall take WrongGPU\wronggpu.ini for the player's)
    m.state_set(b, "rr_mode", "fast")
    rc = t.run("status", exe)
    t.check("in-game save (state rr_mode=fast): status shows Fast, WrongGPU\\wronggpu.ini as installed",
            rc == 0 and t.j.get("game", {}).get("rr_mode") == "fast" and read(rrs) == rr0, t.j.get("game"))
    rc = t.run("repair", exe, *inst)
    t.check("repair after the in-game save: Fast kept (state), both wronggpu.ini files as installed, nothing 'left in place'",
            rc == 0 and m.rr_ini(b).get("rr_mode") == "fast" and read(rrs) == rr0 and read(os.path.join(b, "wronggpu.ini")) == wini0 and
            not any("Left in place" in w for w in t.j.get("warnings", [])),
            (rc, m.rr_ini(b), t.j.get("warnings"), read(rrs) == rr0, read(os.path.join(b, "wronggpu.ini")) == wini0))
    left = kept_runtime(before, root, *unrecorded_wronggpu_files(root))
    rc = t.run("uninstall", exe)
    t.check("uninstall after the in-game save: the folder byte for byte as before (the state folder gone, no settings file left)",
            rc == 0 and snapshot(root) == left and not os.path.exists(os.path.join(b, "wronggpu-setup")), (rc, diff(left, snapshot(root)), t.j.get("warnings")))

    # 2. a player's edit of wronggpu.ini, then the launcher's Apply: the edit stays, the launcher's change goes into the state
    rc = t.run("install", exe, *inst)
    t.check("install again", rc == 0, (rc, t.j.get("message")))
    wp = os.path.join(b, "wronggpu.ini")
    edited = read(wp) + b"; the player's note\r\n"
    write(wp, edited)
    rc = t.run("set", exe, "output_filter=quality5")
    t.check("a player's edit of wronggpu.ini, then set: wronggpu.ini kept byte for byte, output_filter=quality5 in the state",
            rc == 0 and read(wp) == edited and m.state_kv(b).get("output_filter") == "quality5" and "output_filter=quality5" in (m.options_line(m.eff_ini(b)) or ""),
            (rc, m.state_kv(b), t.j.get("message")))
    rc = t.run("set", exe, "rr_mode=accurate")
    t.check("set rr_mode=accurate: in the state, WrongGPU\\wronggpu.ini as installed", rc == 0 and m.state_kv(b).get("rr_mode") == "accurate" and read(rrs) == rr0,
            (rc, m.state_kv(b)))

    # 3. the player edits WrongGPU\wronggpu.ini by hand, then the in-game menu saves: the save decides, the hand edit stays
    hand = rr0.replace(b"rr_mode=fast", b"rr_mode=exact") + b"; mine\r\n"
    write(rrs, hand)
    m.state_set(b, "rr_mode", "fast")
    rc = t.run("status", exe)
    t.check("hand edit then in-game save: the save decides (Fast), the hand-edited file kept", rc == 0 and t.j.get("game", {}).get("rr_mode") == "fast" and read(rrs) == hand,
            t.j.get("game"))

    # 4. uninstall, repair and update refused while another program has a WrongGPU file open or the game has it loaded
    for rel, how in (("plugins\\WrongGPU-RR.asi", "open"), ("wronggpu_runtime.dll", "load")):
        p, state = hold(os.path.join(b, rel), how)
        try:
            if state != "held":
                print("  skipped: %s %s (%s)" % (how, rel, state))
                continue
            snap = snapshot(root)
            for op in ("uninstall", "repair", "update"):
                rc = t.run(op, exe, *(() if op == "uninstall" else inst))
                msg = t.j.get("message", "") + t.j.get("error", "")
                t.check("%s while %s is %s: refused (exit 2), nothing changed, the file named" % (op, rel, "open in another program" if how == "open" else "loaded"),
                        rc == 2 and "in use by the game or another program" in msg and os.path.basename(rel) in msg and snapshot(root) == snap,
                        (rc, msg, diff(snap, snapshot(root))))
        finally:
            release(p)

    # 5. an older install (before #219): its WrongGPU\wronggpu.ini rewritten by the older plugin's save; the next repair
    #    carries the mode into the state and keeps the file; an unchanged file carries nothing
    os.remove(os.path.join(b, m.STATE))
    write(rrs, rr0)
    rc = t.run("repair", exe, *inst)
    t.check("upgrade, WrongGPU\\wronggpu.ini unchanged: nothing carried over (no rr_mode from the file in the state), the file as installed",
            rc == 0 and read(rrs) == rr0 and m.record_entry_sha(b, m.RR_SETTINGS) == sha(rr0), (rc, m.state_kv(b)))
    os.remove(os.path.join(b, m.STATE))
    older = rr0.replace(b"\r\n", b"\n").replace(b"rr_mode=fast", b"rr_mode = exact\nrr_overlay=1")  # an older plugin's save (LF)
    write(rrs, older)
    rc = t.run("repair", exe, *inst)
    sk = m.state_kv(b)
    t.check("upgrade, WrongGPU\\wronggpu.ini saved by an older plugin (LF): rr_mode and rr_overlay carried into the state, the file kept",
            rc == 0 and sk.get("rr_mode") == "exact" and sk.get("rr_overlay") == "1" and read(rrs) == older, (rc, sk))
    left = kept_runtime(before, root, *unrecorded_wronggpu_files(root))  # the unrecorded files, with their bytes now
    rc = t.run("uninstall", exe)
    # the player's two files: WrongGPU\wronggpu.ini kept through the repairs as it is; wronggpu.ini (the install wrote WrongGPU's
    # version of it and kept the player's in the backup) back as the player saved it
    left = kept_runtime(left, root, os.path.join("bin", "x64", m.RR_SETTINGS), os.path.join("bin", "x64", "wronggpu.ini"))
    t.check("uninstall: the player's WrongGPU\\wronggpu.ini and wronggpu.ini stay as the player saved them; everything else as before, the state folder gone",
            rc == 0 and snapshot(root) == left and read(rrs) == older and read(wp) == edited, (rc, diff(left, snapshot(root)), t.j.get("warnings")))
    shutil.rmtree(root, ignore_errors=True)
