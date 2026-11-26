#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Frame generation's files with an install (wg_core/fg.h, 0.7.0): a WRONGGPU_TESTING launcher built with WRONGGPU_FG and the
generated frame generation stand-ins (tests/make_mock_fg.py) installs WrongGPU-FG.asi beside WrongGPU's plugin and its kernels in
plugins\\wronggpu-setup\\fg-dxil (where the plugin reads them) only into a game with a frame generation library, each file an
entry of the install record; a repair keeps them; the uninstall and the restorer leave the game folder byte for byte as it was.
WrongGPU-FG.asi never keeps WrongGPU's Ray Reconstruction out (recorded, or dropped in by hand before the install, which then
backs it up and puts it back). The four settings round-trip through `set`, profile.ini and the game's settings.ini (fg off there
while the data is not converted; an outside change kept), and the game page shows its Frame Generation section with them.

Runs the launcher against GENERATED fixture game folders only (run_launcher_tests.py's harness: its isolation, its test
registry root); never a real game, never a GPU program. Written for this project; Python standard library only. Run by ctest
("fg-install").

    python run_fg_install_tests.py --launcher <WrongGPU.exe> --work <dir> [--fg-dir <the build's frame generation folder>]
"""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time

sys.dont_write_bytecode = True
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
os.environ["WRONGGPU_TESTING"] = "1"
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import run_launcher_tests as lt  # noqa: E402  (the launcher suite's harness, fixtures and isolation)
from run_installer_tests import read, snapshot, diff, write, kept_runtime, unrecorded_wronggpu_files  # noqa: E402

FG_LIB = b"MZ synthetic DLSS Frame Generation library stand-in (never loaded) " * 16
RR_FILES = {"bin/x64/nvngx_dlssd.dll": b"MZ synthetic stand-in, not a usable RR model file " * 64, "bin/x64/sl.dlss_d.dll": b"MZ synthetic stand-in"}
PLUGIN = "plugins\\WrongGPU-FG.asi"
DXIL = "plugins\\wronggpu-setup\\fg-dxil\\"


def sha_file(p):
    with open(p, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def rec_copies(b):
    """The install record's copy entries: lower-case path -> installed hash."""
    p = os.path.join(b, "wronggpu-setup", "install.record")
    out = {}
    if os.path.isfile(p):
        for line in lt.record_text(b).replace("\r", "").split("\n"):
            f = line.split("\t")
            if len(f) == 5 and f[0] == "copy":
                out[f[4].lower()] = f[3]
    return out


def fg_expected(fg_dir):
    """What the build embeds (its frame generation folder): installed path (lower case) -> SHA-256."""
    exp = {}
    for dirpath, _, filenames in os.walk(fg_dir):
        for fn in filenames:
            rel = os.path.relpath(os.path.join(dirpath, fn), fg_dir).replace("/", "\\").lower()
            if rel == "plugins\\wronggpu-fg.asi":
                exp[PLUGIN.lower()] = sha_file(os.path.join(dirpath, fn))
            elif rel.endswith(".dxil"):
                exp[(DXIL + fn).lower()] = sha_file(os.path.join(dirpath, fn))
    return exp


def fg_on_disk(b):
    out = {}
    for rel in [PLUGIN] + [DXIL + n for n in (os.listdir(os.path.join(b, DXIL)) if os.path.isdir(os.path.join(b, DXIL)) else [])]:
        p = os.path.join(b, rel)
        if os.path.isfile(p):
            out[rel.lower()] = sha_file(p)
    return out


def fg_record(b):
    return {k: v for k, v in rec_copies(b).items() if k == PLUGIN.lower() or k.startswith(DXIL.lower())}


def install(t, exe, opti, *more):
    return t.run("install", exe, "--opti=" + opti, "--noscan", "--force", "--experimental", *more)


def drop_entries(t, under):
    for k, v in t.reg_keys().items():
        if k.startswith("WrongGPU-") and v.get("InstallLocation", "").lower().startswith(under.lower()):
            subprocess.run(["reg", "delete", t.uninstall + "\\" + k, "/f"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def page_of(d, name):
    pages = [gp for gp in d.get("games", []) if gp.get("name") == name]
    return pages[0] if pages else {}


def test_no_fg_game(t, opti):
    print("\n[fg] a game without a frame generation library: nothing of frame generation")
    root, b, exe = lt.game(t, "Fg None")
    before = snapshot(root)
    rc = t.run("status", exe, "--noscan")
    g = t.j.get("game", {})
    t.check("status: fg_uses false, no fg_installed", rc == 0 and g.get("fg_uses") is False and "fg_installed" not in g, g)
    rc = install(t, exe, opti)
    rec = lt.record_text(b) if rc == 0 else ""
    t.check("install: no WrongGPU-FG.asi, no kernel folder, no record line of either", rc == 0 and "wronggpu-fg.asi" not in rec.lower() and
            "fg-dxil" not in rec.lower() and not os.path.exists(os.path.join(b, PLUGIN)) and not os.path.exists(os.path.join(b, "plugins", "wronggpu-setup")),
            (rc, t.j.get("message"), rec[:600]))
    t.check("install: the log says frame generation's files were not placed (no frame generation library)",
            "Frame generation: not placed" in read(t.log).decode("utf-8", "replace"), t.logtail())
    rc = t.run("uninstall", exe)
    t.check("uninstall: the game folder as before", rc == 0 and snapshot(root) == before, diff(before, snapshot(root)))
    shutil.rmtree(root, ignore_errors=True)


def test_fg_game(t, opti, expected):
    print("\n[fg] a game with a frame generation library: the plugin and its kernels with the install, settings, the page, uninstall")
    root, b, exe = lt.game(t, "Fg Game", validated=True, extra={"bin/x64/nvngx_dlssg.dll": FG_LIB})
    before = snapshot(root)
    rc = t.run("status", exe, "--noscan")
    g = t.j.get("game", {})
    t.check("status: fg_uses true (the game's nvngx_dlssg.dll), not installed yet", rc == 0 and g.get("fg_uses") is True and g.get("fg_installed") is False, g)
    rc = install(t, exe, opti)
    on_disk, recd = fg_on_disk(b), fg_record(b)
    t.check("install: WrongGPU-FG.asi beside WrongGPU.asi, every kernel in plugins\\wronggpu-setup\\fg-dxil, byte for byte the build's",
            rc == 0 and os.path.isfile(os.path.join(b, "plugins", "WrongGPU.asi")) and on_disk == expected and len(expected) >= 2,
            (rc, t.j.get("message"), sorted(on_disk), sorted(expected), t.logtail()))
    t.check("install: each of them an entry of the install record with its hash (backed up, rolled back, removed with it)", recd == expected,
            (sorted(recd), sorted(expected)))
    t.check("install: the kernel build's list (manifest.json) is not shipped",
            not os.path.exists(os.path.join(b, DXIL, "manifest.json")) and "manifest.json" not in lt.record_text(b).lower())
    t.check("install: the game's own frame generation library untouched", read(os.path.join(b, "nvngx_dlssg.dll")) == FG_LIB)
    rc = t.run("status", exe, "--noscan")
    g = t.j.get("game", {})
    t.check("status: fg_installed true", rc == 0 and g.get("fg_uses") is True and g.get("fg_installed") is True, g)
    # the settings: off by default, nothing in settings.ini until the player opts in
    t.check("install: settings.ini holds no frame generation key (never opted in)", not any(k.startswith("fg") for k in lt.state_kv(b)), lt.state_kv(b))
    prof_path = os.path.join(b, "wronggpu-setup", "profile.ini")
    rc = t.run("set", exe, "fg_preset=B", "fg_multiplier=3", "fg_presenter=dxgi_tearing")
    kv = lt.state_kv(b)
    prof = read(prof_path).decode("utf-8", "replace") if os.path.isfile(prof_path) else ""
    t.check("set fg_preset=B fg_multiplier=3 fg_presenter=dxgi_tearing: profile.ini has the three; settings.ini has all four (fg off)",
            rc == 0 and all(x in prof for x in ("fg_preset=B", "fg_multiplier=3", "fg_presenter=dxgi_tearing")) and kv.get("fg") == "off" and
            kv.get("fg_preset") == "B" and kv.get("fg_multiplier") == "3" and kv.get("fg_presenter") == "dxgi_tearing", (rc, t.j.get("message"), kv, prof))
    # fg=on with `set` needs its data: this PC has none and the game's library is a stand-in that can't be converted (refused, as before)
    rc = t.run("set", exe, "fg=on")
    t.check("set fg=on without convertible data: refused (exit 5), saying why; nothing changed", rc == 5 and "Frame generation was not turned on" in
            (t.j.get("error") or "") and lt.state_kv(b).get("fg") == "off" and "fg=on" not in read(prof_path).decode("utf-8", "replace"), (rc, t.j.get("error")))
    rc = t.run("set", exe, "fg_preset=Q")
    t.check("set fg_preset=Q: refused (exact spellings), nothing changed", rc != 0 and lt.state_kv(b).get("fg_preset") == "B", (rc, t.j.get("message")))
    # an outside change (the game's state folder) is kept by the next launcher write
    lt.state_set(b, "fg_preset", "C")
    rc = t.run("set", exe, "fg_multiplier=4")
    kv = lt.state_kv(b)
    t.check("an outside change (fg_preset=C in settings.ini) stays through the next set; fg_multiplier=4 written", rc == 0 and kv.get("fg_preset") == "C" and
            kv.get("fg_multiplier") == "4" and kv.get("fg_presenter") == "dxgi_tearing", (rc, t.j.get("message"), kv))
    # frame generation turned on outside the launcher (the game's state folder): the page and the next write keep it on, the
    # game gets fg=off while the data is not converted, and the result says why
    lt.state_set(b, "fg", "on")
    oini = os.path.join(b, "OptiScaler.ini")
    oorig = read(oini)
    otext = oorig + b"\r\n[FrameGen]\r\nFGInput=dlssg\r\n"
    write(oini, otext)
    rc = t.run("set", exe, "fg_multiplier=3")
    said = json.dumps(t.j)
    kv = lt.state_kv(b)
    t.check("fg=on changed outside: kept in the profile; the game's settings.ini gets fg=off while the data is not converted, the result says why",
            rc == 0 and "fg=on" in read(prof_path).decode("utf-8", "replace") and kv.get("fg") == "off" and kv.get("fg_multiplier") == "3" and
            "Frame generation's data on this PC is not ready" in said, (rc, kv, said[:1500]))
    t.check("OptiScaler's own frame generation set (FGInput=dlssg): the result says so; OptiScaler.ini left as it is",
            "FGInput=dlssg" in said and read(oini) == otext, said[:1500])
    write(oini, oorig)  # (the install's own file again: the uninstall removes it by its record)
    # the game page: its Frame Generation section with the page's values (the dump forces the data ready for the controls)
    dump = os.path.join(t.work, "ui-dump.json")
    rc = t.run("ui-dump", "--dump=" + dump, "--noscan", "--library=" + os.path.join(t.work, "games"), "--shots=" + os.path.join(t.work, "ui-shots"))
    try:
        d = json.loads(read(dump).decode("utf-8"))
    except Exception as e:  # noqa: BLE001
        d = {"error": str(e)}
    page = page_of(d, "Cyberpunk 2077")
    texts = [c.get("text", "") for c in page.get("controls", [])]
    t.check("game page: the Frame Generation section (heading), its data not converted yet: Convert", rc == 0 and "Frame Generation" in texts and
            "Convert the frame generation data" in texts, (rc, [x for x in texts if "rame" in x]))
    fc = {c.get("text", ""): c for c in page.get("fg_controls", [])}
    sw = fc.get("Use WrongGPU (Frame generation)", {})
    t.check("game page with the data ready: Use WrongGPU (Frame generation) on", sw.get("checked") is True and sw.get("enabled") is True, sw)
    segs = [c for c in page.get("fg_controls", []) if 3810 <= c.get("id", 0) < 3815]
    pres = [c for c in page.get("fg_controls", []) if 3820 <= c.get("id", 0) < 3823]
    t.check("game page: the multiplier 2x..6x with 3x chosen", [c.get("text") for c in segs][:1] and len(segs) == 5 and
            [c.get("checked") for c in segs] == [False, True, False, False, False], [(c.get("text"), c.get("checked")) for c in segs])
    t.check("game page: the preset A, B, C with C chosen (the outside change)", len(pres) == 3 and [c.get("checked") for c in pres] == [False, False, True],
            [(c.get("text"), c.get("checked")) for c in pres])
    sel = [c for c in page.get("fg_controls", []) if c.get("id") == 3803]
    t.check("game page: the presenter shows DXGI Tearing", sel and "DXGI Tearing" in (sel[0].get("value", "") + sel[0].get("text", "")),
            sel)
    st = [c for c in d.get("settings", {}).get("controls", []) if "Frame generation data" in c.get("text", "")]
    t.check("Settings > Model and runtime: the Frame generation data row (not converted yet)", st and "Not converted yet" in st[0].get("text", ""),
            [c.get("text", "")[:120] for c in st])
    # a repair keeps one copy of each, the same entries
    rc = t.run("repair", exe, "--opti=" + opti, "--noscan", "--force")
    t.check("repair: the same frame generation files and record entries", rc == 0 and fg_on_disk(b) == expected and fg_record(b) == expected,
            (rc, t.j.get("message"), sorted(fg_record(b))))
    rc = t.run("status", exe, "--noscan")
    t.check("repair: the settings stay (fg_preset=C, fg_multiplier=3)", lt.state_kv(b).get("fg_preset") == "C" and lt.state_kv(b).get("fg_multiplier") == "3",
            lt.state_kv(b))
    # the plugin's log as the game would write it: unrecorded, left in place by the uninstall
    write(os.path.join(b, "wronggpu-fg.log"), "fg: frame generation is off (fg=off or not set); no hooks installed\r\n")
    left = kept_runtime(before, root, *unrecorded_wronggpu_files(root))
    rc = t.run("uninstall", exe)
    after = snapshot(root)
    t.check("uninstall: the game folder as before (the plugin, every kernel and the plugins\\wronggpu-setup folders gone), only the unrecorded log left",
            rc == 0 and after == left and not os.path.exists(os.path.join(b, "plugins", "wronggpu-setup")), (rc, t.j.get("message"), diff(left, after)))
    # again, uninstalled by the restorer in the game folder
    os.remove(os.path.join(b, "wronggpu-fg.log"))
    rc = install(t, exe, opti)
    t.check("install again: the frame generation files", rc == 0 and fg_on_disk(b) == expected, (rc, t.j.get("message")))
    rc2 = t.raw([os.path.join(b, "wronggpu-setup", "uninstall.exe"), "/S", "/LOG=" + t.log])
    lt.wait_gone(os.path.join(b, "wronggpu-setup"))
    t.check("the restorer (uninstall.exe /S): the game folder as before", rc2 == 0 and snapshot(root) == before, (rc2, diff(before, snapshot(root))))
    shutil.rmtree(root, ignore_errors=True)


def test_rr_and_fg(t, opti, expected):
    print("\n[fg] a game with Ray Reconstruction and frame generation: WrongGPU-FG.asi never keeps WrongGPU's RR out")
    root, b, exe = lt.game(t, "Fg Rr Game", extra=dict(RR_FILES, **{"bin/x64/nvngx_dlssg.dll": FG_LIB}))
    before = snapshot(root)
    rc = install(t, exe, opti)
    rec = rec_copies(b)
    warn = " ".join(t.j.get("warnings", []))
    t.check("install: WrongGPU's RR plugin and WrongGPU-FG.asi side by side; no \"other OptiScaler plugins\" warning",
            rc == 0 and "plugins\\wronggpu-rr.asi" in rec and fg_record(b) == expected and "Other OptiScaler plugins" not in warn,
            (rc, t.j.get("message"), sorted(rec), warn))
    rc = t.run("repair", exe, "--opti=" + opti, "--noscan", "--force")
    rec = rec_copies(b)
    t.check("repair: both stay (the earlier install's WrongGPU-FG.asi is not another setup's plugin)",
            rc == 0 and "plugins\\wronggpu-rr.asi" in rec and fg_record(b) == expected and "Other OptiScaler plugins" not in " ".join(t.j.get("warnings", [])),
            (rc, t.j.get("message"), t.j.get("warnings")))
    rc = t.run("uninstall", exe)
    t.check("uninstall: the game folder as before", rc == 0 and snapshot(root) == before, diff(before, snapshot(root)))
    # a WrongGPU-FG.asi dropped in by hand before the install (a release's files copied over): not another setup's plugin
    hand = b"MZ a frame generation plugin copied in by hand (synthetic)"
    write(os.path.join(b, "plugins", "WrongGPU-FG.asi"), hand)
    before2 = snapshot(root)
    rc = install(t, exe, opti)
    rec = rec_copies(b)
    t.check("a hand-placed WrongGPU-FG.asi: WrongGPU's RR still installed (no question, no warning), the plugin replaced by the build's",
            rc == 0 and "plugins\\wronggpu-rr.asi" in rec and fg_on_disk(b) == expected and "Other OptiScaler plugins" not in " ".join(t.j.get("warnings", [])),
            (rc, t.j.get("message"), t.j.get("warnings")))
    rc = t.run("uninstall", exe)
    t.check("uninstall: the hand-placed WrongGPU-FG.asi back byte for byte, the folder as before the install", rc == 0 and snapshot(root) == before2 and
            read(os.path.join(b, "plugins", "WrongGPU-FG.asi")) == hand, (rc, diff(before2, snapshot(root))))
    shutil.rmtree(root, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--launcher", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--fg-dir", help="the frame generation folder the launcher was built with (default: mock-fg beside its build folder)")
    a = ap.parse_args()
    launcher = os.path.abspath(a.launcher)
    lt.facts.require_test_hooks(launcher, "--launcher")
    fg_dir = os.path.abspath(a.fg_dir) if a.fg_dir else next(
        (p for p in (os.path.join(os.path.dirname(launcher), "mock-fg"), os.path.join(os.path.dirname(os.path.dirname(launcher)), "mock-fg")) if os.path.isdir(p)), "")
    if not os.path.isdir(fg_dir):
        raise SystemExit("no frame generation folder (--fg-dir): not run")
    expected = fg_expected(fg_dir)
    os.makedirs(os.path.abspath(a.work), exist_ok=True)
    work = lt.facts.new_run_folder(os.path.abspath(a.work))
    lt.iso.self_test(work)  # (#243) first: no launcher call with a broken guard
    model = lt.suite.cases.f_mini_case(present=0)
    t = lt.T(launcher, work, model, None)
    drop_entries(t, work)
    opti = lt.opti_package(t)
    t0 = time.time()
    test_no_fg_game(t, opti)
    test_fg_game(t, opti, expected)
    test_rr_and_fg(t, opti, expected)
    drop_entries(t, work)  # nothing of this run stays in the test registry
    print("\n%d checks passed, %d failed (%.0f s)" % (t.passed, len(t.failures), time.time() - t0))
    for f in t.failures:
        print("  FAILED:", f)
    if not t.failures:
        shutil.rmtree(work, ignore_errors=True)
    return 1 if t.failures else 0


if __name__ == "__main__":
    sys.exit(main())
