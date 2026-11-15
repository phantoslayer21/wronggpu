# SPDX-License-Identifier: MIT
"""#219: the game's state folder (wronggpu-setup\\settings.ini, wg_state.h) for run_launcher_tests.py, which calls run() in
its place in the suite (this file keeps that suite below the 256 KB source limit, as launcher_edition_cases.py does).
wronggpu.ini and WrongGPU\\wronggpu.ini hold the install's defaults and are the player's: no launcher operation rewrites them
after the install. What the launcher and the in-game menu save goes into the state folder, which decides over them, which is
always WrongGPU's (uninstall removes it, repair and update keep it), and which uninstall, repair and update only change while
no other program has WrongGPU's files open. The in-game menu's save is written here the way the RR plugin writes it (one
rr_mode line in settings.ini); the plugin's own save is tested with the real plugin in clean/rr_f/proxy/tests/test_hook_dll.cpp.
Written for this project."""
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
    shutil.rmtree(root, ignore_errors=True)  # (the later sections, the window's library among them, see only their own games)


def run(t, opti, m):
    """m: the suite's module (its helpers: game, ini, state_kv, state_set, rr_ini, record_entry_sha, RR_SETTINGS, STATE)."""
    print("\n[state] #219: the saved settings in the state folder, wronggpu.ini never rewritten, uninstall/repair/update refused while in use")
    asi_seed_cases(t, opti, m)
    rr_files = {"bin/x64/nvngx_dlssd.dll": b"MZ synthetic stand-in, not a usable RR model file " * 64,
                "bin/x64/sl.dlss_d.dll": b"MZ synthetic stand-in"}  # the DLSS RR plugin stand-in
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
