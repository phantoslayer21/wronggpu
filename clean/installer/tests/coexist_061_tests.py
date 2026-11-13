# SPDX-License-Identifier: MIT
"""0.6.1 coexistence fixes (another setup next to WrongGPU's: an OptiScaler that names WrongGPU, other plugins, a changed
OptiScaler.ini, WrongGPU's files in another setup) and interrupted operations, run by run_coexist_tests.py.

Written for this project; Python standard library only. Every file of "another setup" is a synthetic stand-in made here
(coexist_fixtures.py / mockpe.py) or a copy of WrongGPU's own files from a test install; no third-party file is read. The
running game is a stand-in program built here from two lines of C (only when cl is on PATH; skipped otherwise).
"""
import json
import os
import shutil
import subprocess

import coexist_fixtures as fx
import run_launcher_tests as lt
from run_installer_tests import diff, read, sha, snapshot, write


def install(t, exe, opti, *more):
    return t.run("install", exe, "--opti=" + opti, "--noscan", "--force", "--experimental", *more)


def repair(t, exe, opti, *more):
    return t.run("repair", exe, "--opti=" + opti, "--noscan", "--force", *more)


def rec_lines(b):
    p = os.path.join(b, "wronggpu-setup", "install.record")
    return lt.record_text(b).replace("\r", "").split("\n") if os.path.isfile(p) else []


def by_tool(t):
    by = {}
    for f in t.j.get("coexistence", []):
        by.setdefault(f["tool"], []).append(f)
    return by


def proxy_of(b):
    return next((l.split("\t")[1] for l in rec_lines(b) if l.startswith("proxy\t")), "dxgi.dll")


def with_files(before, root, files):
    """`before` (a snapshot) with these files (relative to root) as they are now."""
    want = dict(before)
    now = snapshot(root)
    for rel in files:
        k = rel.replace("\\", "/").lower()
        want[k] = now.get(k)
    return want


# ---------------------------------------------------------------- an OptiScaler that names WrongGPU

def test_optiscaler_named_wronggpu(t, opti):
    print("\n[0.6.1 coexist] an OptiScaler whose version resource names WrongGPU is an OptiScaler; ours only by the record's name and hash")
    d = os.path.join(t.work, "fp061")
    shutil.rmtree(d, ignore_errors=True)
    files = {"edition-a.dll": (fx.optiscaler_naming_wronggpu(), "optiscaler"), "edition-b.dll": (fx.optiscaler_naming_wronggpu(product_first=False), "optiscaler"),
             "plugin.asi": (fx.wronggpu_menu_plugin(), "wronggpu"), "runtime.dll": (fx.wronggpu_runtime_like(), "wronggpu")}
    for n, (data, _) in files.items():
        write(os.path.join(d, n), data)
    rc = t.run("fingerprint", *[os.path.join(d, n) for n in files])
    got = {os.path.basename(f["file"]): f.get("tool") for f in t.j.get("files", [])}
    t.check("fingerprint: an OptiScaler naming WrongGPU (either field) is optiscaler; WrongGPU's own plugin and runtime stay wronggpu",
            rc == 0 and all(got.get(n) == k for n, (_, k) in files.items()), got)
    shutil.rmtree(d, ignore_errors=True)

    root, b, exe = lt.game(t, "Edition Next Game", validated=True)
    before = snapshot(root)
    rc = t.run("install", exe, "--opti=" + opti, "--noscan", "--force")
    proxy = proxy_of(b)
    other = "winmm.dll" if proxy.lower() != "winmm.dll" else "version.dll"
    edition = fx.optiscaler_naming_wronggpu()
    write(os.path.join(b, other), edition)  # another setup's OptiScaler (it names WrongGPU) beside WrongGPU's
    rc2 = t.run("coexist", exe)
    by = by_tool(t)
    t.check("coexist: the other OptiScaler (names WrongGPU) is not WrongGPU's: ask to switch, WrongGPU's setup still managed",
            rc == 0 and rc2 == 0 and any(f["decision"] == "ask-switch" and other in f.get("files", []) for f in by.get("optiscaler", [])) and
            any(f["decision"] == "managed" for f in by.get("optiscaler", [])), by.get("optiscaler"))
    state = snapshot(root)
    rc = repair(t, exe, opti)
    t.check("repair without the player's OK: exit 6 (switch asked), nothing changed (never a second OptiScaler beside it)",
            rc == 6 and snapshot(root) == state, (rc, t.j.get("message"), diff(state, snapshot(root))))
    rc = repair(t, exe, opti, "--switch")
    lines = rec_lines(b) if rc == 0 else []
    t.check("repair --switch: the other OptiScaler moved aside (owner moved, in the backup)", rc == 0 and ("owner\tdisable\t%s\tmoved" % other) in lines and
            not os.path.exists(os.path.join(b, other)) and read(os.path.join(b, "wronggpu-setup", "backup", other)) == edition, (rc, t.j.get("message"), lines[:16]))
    rc = t.run("uninstall", exe)
    want = dict(before)
    want[os.path.relpath(os.path.join(b, other), root).replace("\\", "/").lower()] = sha(edition)
    t.check("uninstall: the game as before WrongGPU, the other OptiScaler back byte for byte", rc == 0 and snapshot(root) == want, diff(want, snapshot(root)))

    # the same OptiScaler build as WrongGPU's setup (the package's hash) under a name the record does not have: not ours
    os.remove(os.path.join(b, other))
    rc = t.run("install", exe, "--opti=" + opti, "--noscan", "--force")
    proxy = proxy_of(b)
    same = "version.dll" if proxy.lower() != "version.dll" else "wininet.dll"
    shutil.copyfile(os.path.join(b, proxy), os.path.join(b, same))
    rc2 = t.run("coexist", exe)
    by = by_tool(t)
    t.check("coexist: the package's own OptiScaler build under another name (" + same + ") is another setup's: ask to switch",
            rc == 0 and rc2 == 0 and any(f["decision"] == "ask-switch" and same in f.get("files", []) for f in by.get("optiscaler", [])), by.get("optiscaler"))
    os.remove(os.path.join(b, same))
    t.run("uninstall", exe)
    shutil.rmtree(root, ignore_errors=True)


# ---------------------------------------------------------------- another setup's OptiScaler over WrongGPU's proxy

def older_record(b, version="0.5.1"):
    p = os.path.join(b, "wronggpu-setup", "install.record")
    write(p, "\r\n".join(("version\t" + version) if l.startswith("version\t") else l for l in read(p).decode("utf-8").split("\r\n")))


def test_proxy_replaced(t, opti):
    print("\n[0.6.1 coexist] another setup's OptiScaler replaced WrongGPU's proxy: Repair / Update ask the switch question first")
    root, b, exe = lt.game(t, "Proxy Replaced Game", validated=True)
    before = snapshot(root)
    rc = t.run("install", exe, "--opti=" + opti, "--noscan", "--force")
    proxy = proxy_of(b)
    theirs = fx.optiscaler()
    write(os.path.join(b, proxy), theirs)  # another setup put its OptiScaler under WrongGPU's proxy name
    key = os.path.relpath(os.path.join(b, proxy), root).replace("\\", "/").lower()
    rc2 = t.run("status", exe)
    g = t.j.get("game", {})
    t.check("status: the other OptiScaler under WrongGPU's proxy name is \"Asks first\" (ask-switch)", rc == 0 and rc2 == 0 and
            any(f.get("decision") == "ask-switch" and proxy in f.get("files", []) for f in g.get("coexistence", [])), g.get("coexistence"))
    t.check("status: that file is listed by one finding only (WrongGPU's setup says its proxy was replaced)",
            sum(1 for f in g.get("coexistence", []) if proxy in f.get("files", [])) == 1 and
            any(f.get("decision") == "managed" and "was replaced" in f.get("what", "") for f in g.get("coexistence", [])), g.get("coexistence"))
    state = snapshot(root)
    rc = repair(t, exe, opti)
    t.check("repair without the player's OK: exit 6 (the switch question), nothing changed, the other OptiScaler untouched",
            rc == 6 and snapshot(root) == state and "Switch to WrongGPU's OptiScaler setup" in (t.j.get("message") or ""), (rc, t.j.get("message"), diff(state, snapshot(root))))
    rc = repair(t, exe, opti, "--switch")
    lines = rec_lines(b) if rc == 0 else []
    warns = t.j.get("warnings", [])
    t.check("repair --switch: the other OptiScaler moved aside (owner moved, in the backup), WrongGPU's under a free name",
            rc == 0 and ("owner\tdisable\t%s\tmoved" % proxy) in lines and read(os.path.join(b, "wronggpu-setup", "backup", proxy)) == theirs and
            proxy_of(b).lower() != proxy.lower(), (rc, t.j.get("message"), lines[:16]))
    t.check("repair --switch: the result says it was moved aside with the OK (no \"Left in place\" for it)",
            any(w.startswith("Moved aside with your OK: " + proxy) for w in warns) and not any("Left in place" in w and proxy in w for w in warns), warns)
    rc = t.run("uninstall", exe)
    want = dict(before)
    want[key] = sha(theirs)
    t.check("uninstall: the game as before WrongGPU, with the other setup's OptiScaler back byte for byte", rc == 0 and snapshot(root) == want, diff(want, snapshot(root)))

    # Update: the same question; an uninstall leaves the other OptiScaler and says what it is
    os.remove(os.path.join(b, proxy))
    rc = t.run("install", exe, "--opti=" + opti, "--noscan", "--force")
    proxy = proxy_of(b)
    write(os.path.join(b, proxy), theirs)
    older_record(b)
    state = snapshot(root)
    rc2 = t.run("update", exe, "--opti=" + opti, "--noscan", "--force")
    t.check("update without the player's OK: exit 6, nothing changed", rc == 0 and rc2 == 6 and snapshot(root) == state, (rc, rc2, t.j.get("message")))
    rc = t.run("uninstall", exe)
    warns = t.j.get("warnings", [])
    t.check("uninstall: the other setup's OptiScaler left in place, named as such", rc == 0 and read(os.path.join(b, proxy)) == theirs and
            any(("Left in place: %s is now another setup's OptiScaler" % proxy) in w for w in warns), warns)
    shutil.rmtree(root, ignore_errors=True)


# ---------------------------------------------------------------- OptiScaler.ini: the install's beside its OptiScaler

OTHER_SETUP_INI = ("[Upscalers]\r\nDx12Upscaler=xess\r\n[Libraries]\r\nFfxDx12SRPath=C:\\OtherSetup\\upscaler.dll\r\n"
                   "; another setup's OptiScaler.ini (synthetic)\r\n")


def test_opti_ini_ownership(t, opti):
    print("\n[0.6.1 coexist] OptiScaler.ini is WrongGPU's beside WrongGPU's OptiScaler (whatever its menu saved); another setup's with that setup's OptiScaler")
    root, b, exe = lt.game(t, "Ini Owner Game", validated=True)
    before = snapshot(root)
    ini_path = os.path.join(b, "OptiScaler.ini")
    key = os.path.relpath(ini_path, root).replace("\\", "/").lower()
    # OptiScaler's menu saves the ini (WrongGPU's keys kept): still WrongGPU's, removed by the uninstall
    rc = t.run("install", exe, "--opti=" + opti, "--noscan", "--force")
    write(ini_path, read(ini_path) + b"; saved by OptiScaler's menu\r\n")
    rc2 = t.run("uninstall", exe)
    t.check("menu-saved OptiScaler.ini (WrongGPU's keys kept): still WrongGPU's, removed with the uninstall", rc == 0 and rc2 == 0 and snapshot(root) == before,
            diff(before, snapshot(root)))
    # the player picked another upscaler in OptiScaler's menu (its keys point elsewhere now), WrongGPU's OptiScaler still in
    # place: still WrongGPU's (never "another setup's"), removed by the uninstall
    rc = t.run("install", exe, "--opti=" + opti, "--noscan", "--force")
    write(ini_path, OTHER_SETUP_INI)
    rc2 = t.run("status", exe)
    g = t.j.get("game", {})
    t.check("status: an OptiScaler.ini the menu pointed elsewhere, beside WrongGPU's OptiScaler, is WrongGPU's (no question for it)", rc == 0 and rc2 == 0 and
            not any(f.get("decision") == "ask-switch" for f in g.get("coexistence", [])), g.get("coexistence"))
    rc = t.run("uninstall", exe)
    warns = t.j.get("warnings", [])
    t.check("uninstall: that ini goes with the rest, no \"another setup changed it\" warning", rc == 0 and snapshot(root) == before and
            not any("OptiScaler.ini" in w for w in warns), (warns, diff(before, snapshot(root))))
    # another setup replaced WrongGPU's OptiScaler with its own, and its ini: asked first, backed up, left alone without the OK
    rc = t.run("install", exe, "--opti=" + opti, "--noscan", "--force")
    proxy = proxy_of(b)
    theirs = fx.optiscaler()
    pkey = os.path.relpath(os.path.join(b, proxy), root).replace("\\", "/").lower()
    write(os.path.join(b, proxy), theirs)
    write(ini_path, OTHER_SETUP_INI)
    rc2 = t.run("status", exe)
    g = t.j.get("game", {})
    t.check("status: with another setup's OptiScaler over WrongGPU's, its OptiScaler.ini is another setup's too (ask-switch)", rc == 0 and rc2 == 0 and
            any(f.get("decision") == "ask-switch" and "OptiScaler.ini" in f.get("files", []) for f in g.get("coexistence", [])), g.get("coexistence"))
    state = snapshot(root)
    rc = repair(t, exe, opti)
    t.check("repair without the player's OK: exit 6, nothing changed (the other setup's files kept)", rc == 6 and snapshot(root) == state, (rc, t.j.get("message")))
    rc = repair(t, exe, opti, "--switch")
    lines = rec_lines(b) if rc == 0 else []
    warns = t.j.get("warnings", [])
    t.check("repair --switch: the other setup's ini backed up (an entry that existed), WrongGPU's in place, said as replaced with the OK",
            rc == 0 and any(l.startswith("copy\t1\t") and l.endswith("\tOptiScaler.ini") for l in lines) and
            read(os.path.join(b, "wronggpu-setup", "backup", "OptiScaler.ini")) == OTHER_SETUP_INI.encode() and
            any(w.startswith("Replaced with your OK: OptiScaler.ini") for w in warns), (rc, t.j.get("message"), warns))
    rc = t.run("uninstall", exe)
    want = dict(before)
    want[key] = sha(OTHER_SETUP_INI.encode())
    want[pkey] = sha(theirs)
    t.check("uninstall: the other setup's OptiScaler and OptiScaler.ini back byte for byte", rc == 0 and snapshot(root) == want, diff(want, snapshot(root)))
    os.remove(ini_path)
    os.remove(os.path.join(b, proxy))
    # uninstall with both replaced: left in place, with warnings
    rc = t.run("install", exe, "--opti=" + opti, "--noscan", "--force")
    proxy = proxy_of(b)
    write(os.path.join(b, proxy), theirs)
    write(ini_path, OTHER_SETUP_INI)
    rc2 = t.run("uninstall", exe)
    warns = t.j.get("warnings", [])
    t.check("uninstall: another setup's OptiScaler.ini (beside its OptiScaler) is left in place with a warning", rc == 0 and rc2 == 0 and
            read(ini_path) == OTHER_SETUP_INI.encode() and any("OptiScaler.ini no longer points at WrongGPU's files" in w for w in warns), warns)
    shutil.rmtree(root, ignore_errors=True)


def test_players_ini_edited_in_menu(t, opti):
    print("\n[0.6.1 coexist] the player's own OptiScaler.ini (backed up by a switch), then changed in OptiScaler's menu: never lost")
    theirs = fx.optiscaler()
    players_ini = "[Menu]\r\nShowFps=true\r\n; the player's own OptiScaler.ini (synthetic)\r\n"
    root, b, exe = lt.game(t, "Menu Ini Game", validated=True, extra={"bin/x64/winmm.dll": theirs, "bin/x64/OptiScaler.ini": players_ini})
    before = snapshot(root)
    ini_path = os.path.join(b, "OptiScaler.ini")
    # uninstall after the menu changed the upscaler (keys no longer WrongGPU's): the player's own ini comes back
    rc = install(t, exe, opti, "--switch")
    rec = "\n".join(rec_lines(b))
    write(ini_path, OTHER_SETUP_INI)
    rc2 = t.run("uninstall", exe)
    t.check("uninstall after the menu changed the player's switched-over OptiScaler.ini: their own ini and OptiScaler back byte for byte",
            rc == 0 and "\tOptiScaler.ini" in rec and rc2 == 0 and snapshot(root) == before, (rc, rc2, t.j.get("message"), t.j.get("warnings"), diff(before, snapshot(root))))
    # Repair after the same menu change: the player's own ini is kept in the new install's backup, and comes back on uninstall
    rc = install(t, exe, opti, "--switch")
    write(ini_path, OTHER_SETUP_INI)
    rc2 = repair(t, exe, opti)
    backup = os.path.join(b, "wronggpu-setup", "backup", "OptiScaler.ini")
    t.check("repair after that menu change: done, the player's own ini in the new install's backup", rc == 0 and rc2 == 0 and
            os.path.isfile(backup) and read(backup) == players_ini.encode(), (rc, rc2, t.j.get("message")))
    rc = t.run("uninstall", exe)
    t.check("...uninstall: the player's own ini and OptiScaler back byte for byte", rc == 0 and snapshot(root) == before, diff(before, snapshot(root)))
    # another setup replaced WrongGPU's OptiScaler and that ini since: the backup of the player's own ini is never deleted.
    # Repair refuses before anything changes; the uninstall leaves the other setup's files and keeps the backup folder
    rc = install(t, exe, opti, "--switch")
    proxy = proxy_of(b)
    other = fx.optiscaler_naming_wronggpu()
    write(os.path.join(b, proxy), other)
    write(ini_path, OTHER_SETUP_INI)
    state = snapshot(root)
    rc2 = repair(t, exe, opti, "--switch")
    t.check("repair --switch with another setup's files over a backed-up original: refused (exit 2, \"Nothing was changed\"), nothing changed",
            rc == 0 and rc2 == 2 and "Nothing was changed" in (t.j.get("message") or "") and snapshot(root) == state, (rc, rc2, t.j.get("message"), diff(state, snapshot(root))))
    rc = t.run("uninstall", exe)
    warns = t.j.get("warnings", [])
    t.check("uninstall: the other setup's files stay, the player's own ini stays in the kept backup folder (said)", rc == 0 and read(ini_path) == OTHER_SETUP_INI.encode() and
            read(os.path.join(b, proxy)) == other and os.path.isfile(backup) and read(backup) == players_ini.encode() and
            any(w.startswith("The backup folder was kept") for w in warns), (rc, t.j.get("message"), warns))
    shutil.rmtree(root, ignore_errors=True)


# ---------------------------------------------------------------- what the uninstall removes goes by hash

def test_uninstall_by_hash(t, opti):
    print("\n[0.6.1 coexist] the uninstall removes a changed file only when its hash is one of WrongGPU's releases")
    root, b, exe = lt.game(t, "Uninstall Hash Game", validated=True)
    rc = t.run("install", exe, "--opti=" + opti, "--noscan", "--force")
    rt, fe = os.path.join(b, "wronggpu_runtime.dll"), os.path.join(b, "wronggpu_ffx_dx12.dll")
    named = fx.wronggpu_runtime_like()  # another setup's build that names WrongGPU in its version resource (synthetic)
    write(rt, named)
    rc2 = t.run("uninstall", exe)
    warns = t.j.get("warnings", [])
    t.check("uninstall: a changed file whose version resource names WrongGPU (not a release's hash) is left in place with a warning",
            rc == 0 and rc2 == 0 and read(rt) == named and not os.path.exists(fe) and any("changed since install" in w and "wronggpu_runtime.dll" in w for w in warns), warns)
    os.remove(rt)
    # a release's file under another of the record's names: WrongGPU's, removed
    rc = t.run("install", exe, "--opti=" + opti, "--noscan", "--force")
    shutil.copyfile(fe, rt)  # the payload's front end (a hash WrongGPU knows) where the runtime was
    rc2 = t.run("uninstall", exe)
    t.check("uninstall: a changed file whose hash is a WrongGPU build's is removed with the rest", rc == 0 and rc2 == 0 and not os.path.exists(rt),
            (rc2, t.j.get("warnings")))
    # another setup's OptiScaler.ini that names WrongGPU's files stays and is warned about
    rc = t.run("install", exe, "--opti=" + opti, "--noscan", "--force")
    ini = "[Libraries]\r\nFfxDx12SRPath=%s\r\n" % os.path.join(b, "OtherSetup", "wronggpu_ffx_dx12.dll")
    write(os.path.join(b, proxy_of(b)), fx.optiscaler())  # (that setup's OptiScaler replaced WrongGPU's: the ini is that setup's)
    write(os.path.join(b, "OptiScaler.ini"), ini)
    rc2 = t.run("uninstall", exe)
    warns = t.j.get("warnings", [])
    t.check("uninstall: another setup's OptiScaler.ini that names WrongGPU's files gets a warning (that setup may need them)",
            rc == 0 and rc2 == 0 and any("still names WrongGPU's files" in w for w in warns), warns)
    shutil.rmtree(root, ignore_errors=True)


# ---------------------------------------------------------------- other setups' plugins beside OptiScaler

# an RR game: the RR model file stand-in alone (a game that creates RR itself, as run_launcher_tests.py's [direct-rr] game)
RR_FILES = {"bin/x64/nvngx_dlssd.dll": b"MZ synthetic stand-in, not a usable RR model file " * 64}


def test_other_plugins(t, opti):
    print("\n[0.6.1 coexist] other setups' OptiScaler plugins: WrongGPU's RR stays out by default; moved aside only with the player's OK")
    other = fx.unknown("another setup's OptiScaler plugin (synthetic .asi)")
    extra = dict(RR_FILES)
    extra["bin/x64/plugins/OtherSetup.asi"] = other
    root, b, exe = lt.game(t, "Other Plugin Rr Game", extra=extra)
    before = snapshot(root)
    rc = t.run("coexist", exe)
    by = by_tool(t)
    t.check("coexist: the other plugin is found (warn: it may hook RR too)", rc == 0 and
            any(f["decision"] == "warn" and f.get("files") == ["plugins\\OtherSetup.asi"] for f in by.get("optiscaler-plugin", [])), by.get("optiscaler-plugin"))
    rc = install(t, exe, opti)
    rec = lt.record_text(b) if rc == 0 else ""
    warns = t.j.get("warnings", [])
    t.check("install (default): WrongGPU's RR left out with a warning, the other plugin untouched",
            rc == 0 and "WrongGPU-RR.asi" not in rec and read(os.path.join(b, "plugins", "OtherSetup.asi")) == other and
            any(w.startswith("Other OptiScaler plugins are in this game (plugins\\OtherSetup.asi)") for w in warns), (rc, t.j.get("message"), warns))
    rc = t.run("uninstall", exe)
    t.check("...uninstall byte for byte", rc == 0 and snapshot(root) == before, diff(before, snapshot(root)))
    rc = install(t, exe, opti, "--switch-plugins")
    lines = rec_lines(b) if rc == 0 else []
    warns = t.j.get("warnings", [])
    t.check("install --switch-plugins: the other plugin moved aside (owner moved, in the backup), WrongGPU's RR plugin in",
            rc == 0 and any(l.endswith("\\WrongGPU-RR.asi") and l.startswith("copy\t") for l in lines) and "owner\tdisable\tplugins\\OtherSetup.asi\tmoved" in lines and
            not os.path.exists(os.path.join(b, "plugins", "OtherSetup.asi")) and read(os.path.join(b, "wronggpu-setup", "backup", "plugins", "OtherSetup.asi")) == other and
            any(w.startswith("Moved aside with your OK: plugins\\OtherSetup.asi") for w in warns), (rc, t.j.get("message"), lines[:20], warns))
    # Repair and Update after "Move them aside": the plugin moved with the OK stays aside, WrongGPU's RR stays in (the
    # earlier install's restore brings it back; byte for byte that file goes aside again on that OK)
    for verb, more in (("repair", ()), ("update", ())):
        if verb == "update":
            older_record(b)
        rc = repair(t, exe, opti, *more) if verb == "repair" else t.run("update", exe, "--opti=" + opti, "--noscan", "--force")
        lines = rec_lines(b) if rc == 0 else []
        t.check(verb + " after \"Move them aside\": the other plugin still aside (recorded, in the backup), WrongGPU's RR plugin still in",
                rc == 0 and "owner\tdisable\tplugins\\OtherSetup.asi\tmoved" in lines and not os.path.exists(os.path.join(b, "plugins", "OtherSetup.asi")) and
                os.path.isfile(os.path.join(b, "plugins", "WrongGPU-RR.asi")) and read(os.path.join(b, "wronggpu-setup", "backup", "plugins", "OtherSetup.asi")) == other,
                (rc, t.j.get("message"), t.j.get("warnings"), lines[:20]))
    rc = t.run("uninstall", exe)
    t.check("...uninstall: the other plugin back byte for byte", rc == 0 and snapshot(root) == before, diff(before, snapshot(root)))
    # a different plugin under the moved name since: never moved without the OK; WrongGPU's RR goes out instead
    rc = install(t, exe, opti, "--switch-plugins")
    changed = fx.unknown("another plugin put under the moved name since (synthetic .asi)")
    write(os.path.join(b, "plugins", "OtherSetup.asi"), changed)
    state = snapshot(root)
    rc2 = repair(t, exe, opti, "--switch-plugins")
    t.check("repair with a different file under a moved plugin's name: refused before anything changes (its moved original's backup is kept)",
            rc == 0 and rc2 == 2 and "Nothing was changed" in (t.j.get("message") or "") and snapshot(root) == state, (rc, rc2, t.j.get("message")))
    os.remove(os.path.join(b, "plugins", "OtherSetup.asi"))
    rc = t.run("uninstall", exe)
    t.check("...uninstall after it was taken away again: the other plugin back byte for byte", rc == 0 and snapshot(root) == before, diff(before, snapshot(root)))
    # OptiScaler's own plugin folder counts too; a game without RR keeps every plugin beside WrongGPU's (inform only)
    write(os.path.join(b, "OptiScaler", "plugins", "Second.asi"), other)
    rc = t.run("coexist", exe)
    by = by_tool(t)
    t.check("coexist: a plugin in OptiScaler\\plugins\\ counts too", rc == 0 and
            any("OptiScaler\\plugins\\Second.asi" in f.get("files", []) for f in by.get("optiscaler-plugin", [])), by.get("optiscaler-plugin"))
    shutil.rmtree(root, ignore_errors=True)
    root, b, exe = lt.game(t, "Other Plugin Plain Game", extra={"bin/x64/plugins/OtherSetup.asi": other})
    before = snapshot(root)
    rc = t.run("coexist", exe)
    by = by_tool(t)
    rc2 = install(t, exe, opti)
    t.check("a game without RR: the other plugin is only shown (inform) and stays; WrongGPU's plugin beside it", rc == 0 and rc2 == 0 and
            any(f["decision"] == "inform" for f in by.get("optiscaler-plugin", [])) and read(os.path.join(b, "plugins", "OtherSetup.asi")) == other and
            os.path.isfile(os.path.join(b, "plugins", "WrongGPU.asi")), (rc2, by.get("optiscaler-plugin"), t.j.get("message")))
    rc = t.run("uninstall", exe)
    t.check("...uninstall byte for byte", rc == 0 and snapshot(root) == before, diff(before, snapshot(root)))
    shutil.rmtree(root, ignore_errors=True)


# ---------------------------------------------------------------- WrongGPU's files in another mod's setup

def our_files(t, opti):
    """WrongGPU's runtime and front end as a hosted install writes them (from a throwaway game), and its licence file when
    the install writes one (any package that ships WrongGPU's files carries that text)."""
    root, b, exe = lt.game(t, "Kit Source")
    install(t, exe, opti)
    files = {n: read(os.path.join(b, n)) for n in ("wronggpu_runtime.dll", "wronggpu_ffx_dx12.dll", "wronggpu-license.txt") if os.path.isfile(os.path.join(b, n))}
    t.run("uninstall", exe)
    shutil.rmtree(root, ignore_errors=True)
    return files


def test_leftovers_other_setup(t, opti):
    print("\n[0.6.1 coexist] WrongGPU's files beside an OptiScaler WrongGPU didn't install: used by another mod's setup, left alone")
    kit = our_files(t, opti)
    extra = {"bin/x64/dxgi.dll": fx.optiscaler(), "bin/x64/OptiScaler.ini": "[Menu]\r\n; another mod's setup (synthetic)\r\n"}
    extra.update({"bin/x64/" + n: d for n, d in kit.items()})
    root, b, exe = lt.game(t, "Other Setup Kit Game", extra=extra)
    state = snapshot(root)
    rc = t.run("status", exe, "--noscan")
    g = t.j.get("game", {})
    left = {x["path"] for x in g.get("leftovers", [])}
    t.check("status: not \"leftovers\" (the other mod's OptiScaler is asked about); the files listed as used by another mod's setup",
            rc == 0 and g.get("status") == "players-optiscaler" and g.get("leftovers_other_setup") is True and set(kit) <= left, (g.get("status"), g.get("leftovers_other_setup"), left))
    for verb in ("clean-leftovers", "adopt-leftovers"):
        rc = t.run(verb, exe)
        t.check(verb + ": refused without confirming (exit 2), nothing changed, said why", rc == 2 and snapshot(root) == state and
                "used by another mod's setup" in (t.j.get("message") or ""), (rc, t.j.get("message")))
    rc = install(t, exe, opti, "--switch")
    notes = " ".join(t.j.get("notes", []))
    t.check("install --switch: the files are not removed first (left alone, said in the notes); installed", rc == 0 and "used by another mod's setup" in notes,
            (rc, t.j.get("message"), notes))
    rc = t.run("uninstall", exe)
    t.check("uninstall: the other mod's setup back byte for byte (its OptiScaler and WrongGPU's files it used)", rc == 0 and snapshot(root) == state,
            diff(state, snapshot(root)))
    rc = t.run("clean-leftovers", exe, "--force")
    t.check("clean-leftovers --force: still refused (--force is only the GPU/HIP check's), nothing changed", rc == 2 and snapshot(root) == state, (rc, t.j.get("message")))
    rc = t.run("clean-leftovers", exe, "--confirm-other-setup")
    t.check("clean-leftovers --confirm-other-setup (confirmed): the files go", rc == 0 and not any(os.path.exists(os.path.join(b, n)) for n in kit), (rc, t.j.get("message")))
    shutil.rmtree(root, ignore_errors=True)
    # the licence file alone never makes them WrongGPU's own setup (any package that ships WrongGPU's files carries it);
    # WrongGPU's menu plugin by its hash does
    print("  (the kit above %s WrongGPU's licence file)" % ("held" if "wronggpu-license.txt" in kit else "had no"))
    root, b, exe = lt.game(t, "Own Setup Lost Record")
    rc = install(t, exe, opti)
    menu = os.path.join(b, "plugins", "WrongGPU.asi")
    has_menu = os.path.isfile(menu)
    shutil.rmtree(os.path.join(b, "wronggpu-setup"))  # the record is lost
    for k, v in t.reg_keys().items():  # ... and its Apps & features entry (test root only)
        if k.startswith("WrongGPU-") and v.get("InstallLocation", "").lower().startswith(root.lower()):
            subprocess.run(["reg", "delete", t.uninstall + "\\" + k, "/f"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    rc2 = t.run("status", exe, "--noscan")
    g = t.j.get("game", {})
    t.check("WrongGPU's own setup that lost its record (its menu plugin there, by its hash): leftovers to clean up, not another mod's setup",
            rc == 0 and rc2 == 0 and has_menu and g.get("status") == "leftovers" and not g.get("leftovers_other_setup"),
            (rc, has_menu, g.get("status"), g.get("leftovers_other_setup"), [x["path"] for x in g.get("leftovers", [])]))
    shutil.rmtree(root, ignore_errors=True)


# ---------------------------------------------------------------- WrongGPU's RR plugin is a known leftover

def test_rr_plugin_leftover(t, opti):
    print("\n[0.6.1 coexist] leftovers: WrongGPU's RR plugin is recognised (by the embedded payload's and the releases' hashes)")
    root, b, exe = lt.game(t, "Rr Leftover Game", extra=dict(RR_FILES))
    rc = install(t, exe, opti)
    has_rr = os.path.isfile(os.path.join(b, "plugins", "WrongGPU-RR.asi"))
    shutil.rmtree(os.path.join(b, "wronggpu-setup"))  # the record is lost
    for k, v in t.reg_keys().items():  # ... and its Apps & features entry (test root only)
        if k.startswith("WrongGPU-") and v.get("InstallLocation", "").lower().startswith(root.lower()):
            subprocess.run(["reg", "delete", t.uninstall + "\\" + k, "/f"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    rc2 = t.run("status", exe, "--noscan")
    g = t.j.get("game", {})
    left = {x["path"] for x in g.get("leftovers", [])}
    t.check("status: leftovers include plugins\\WrongGPU-RR.asi", rc == 0 and has_rr and rc2 == 0 and g.get("status") == "leftovers" and
            "plugins\\WrongGPU-RR.asi" in left, (rc, has_rr, g.get("status"), sorted(left)))
    rc = t.run("clean-leftovers", exe)
    t.check("clean up: the RR plugin goes with the other leftovers", rc == 0 and not os.path.exists(os.path.join(b, "plugins", "WrongGPU-RR.asi")), (rc, t.j.get("message")))
    shutil.rmtree(root, ignore_errors=True)


# ---------------------------------------------------------------- the "stays out" card of another upscaler mod

def test_stays_out_card(t, opti):
    print("\n[0.6.1 coexist] the card of a game with another upscaler mod (never named) reads as a sentence")
    root, b, exe = lt.game(t, "Other Mod Card Game", extra={"bin/x64/" + lt.other_mod_keys()[1][0]: "[mock]\r\n"})  # its marker file
    dump = os.path.join(t.work, "ui-dump-061.json")
    rc = t.run("ui-dump", "--dump=" + dump, "--noscan", "--library=" + os.path.join(t.work, "games"))
    try:
        d = json.loads(read(dump).decode("utf-8"))
    except Exception as e:  # noqa: BLE001
        d = {"error": str(e)}
    page = next((p for p in d.get("games", []) if p.get("name") == "Other Mod Card Game"), {})
    texts = [c.get("text", "") for c in page.get("controls", [])]
    t.check("card: \"Another upscaler mod is in this game. WrongGPU stays out ...\", never \"Not touched is in this game\"",
            rc == 0 and any(x.startswith("Another upscaler mod is in this game. WrongGPU stays out") for x in texts) and
            not any("ot touched is in this game" in x for x in texts) and not lt.names_other_mod(json.dumps(page)), (rc, texts[:12]))
    shutil.rmtree(root, ignore_errors=True)


# ---------------------------------------------------------------- the switch's OK stands for the same files

def test_repair_after_switch(t, opti):
    print("\n[0.6.1 coexist] Repair after a switch: the files moved aside with the OK go aside again; a different file is asked about")
    theirs, ini = fx.optiscaler(), "[Menu]\r\nShowFps=true\r\n; the player's\r\n"
    root, b, exe = lt.game(t, "Switch Repair Game", validated=True, extra={"bin/x64/winmm.dll": theirs, "bin/x64/OptiScaler.ini": ini})
    before = snapshot(root)
    rc = t.run("install", exe, "--opti=" + opti, "--noscan", "--force", "--switch")
    rc2 = repair(t, exe, opti)
    lines = rec_lines(b) if rc2 == 0 else []
    t.check("repair without --switch after a switch: done, the player's OptiScaler moved aside again on the earlier OK",
            rc == 0 and rc2 == 0 and "owner\tdisable\twinmm.dll\tmoved" in lines and not os.path.exists(os.path.join(b, "winmm.dll")) and
            read(os.path.join(b, "wronggpu-setup", "backup", "winmm.dll")) == theirs, (rc, rc2, t.j.get("message"), lines[:16]))
    rc = t.run("uninstall", exe)
    t.check("...uninstall: the player's OptiScaler and ini back byte for byte", rc == 0 and snapshot(root) == before, diff(before, snapshot(root)))
    # the switch moved winmm.dll aside; then another file is put under that name: that one is asked about
    rc = t.run("install", exe, "--opti=" + opti, "--noscan", "--force", "--switch")
    other = fx.optiscaler_naming_wronggpu()
    write(os.path.join(b, "winmm.dll"), other)
    state = snapshot(root)
    rc2 = repair(t, exe, opti)
    t.check("repair without --switch, a different file under the moved name: exit 6 (asked), nothing changed", rc == 0 and rc2 == 6 and snapshot(root) == state,
            (rc, rc2, t.j.get("message"), diff(state, snapshot(root))))
    os.remove(os.path.join(b, "winmm.dll"))
    t.run("uninstall", exe)
    shutil.rmtree(root, ignore_errors=True)


# ---------------------------------------------------------------- backups that must be kept; the settings file

def test_backups_kept(t, opti):
    print("\n[0.6.1 coexist] Update, Repair and the move out of a wrong folder never discard a backup of the player's files")
    theirs, log_a, log_b = fx.optiscaler(), b"the player's own OptiScaler.log (synthetic)\r\n", b"a log WrongGPU's OptiScaler wrote (synthetic)\r\n"
    root, b, exe = lt.game(t, "Kept Log Game", validated=True, extra={"bin/x64/winmm.dll": theirs, "bin/x64/OptiScaler.log": log_a})
    before = snapshot(root)
    backup = os.path.join(b, "wronggpu-setup", "backup", "OptiScaler.log")
    rc = install(t, exe, opti, "--switch")
    t.check("install --switch: the player's OptiScaler.log moved aside with their OptiScaler (in the backup)", rc == 0 and os.path.isfile(backup) and read(backup) == log_a,
            (rc, t.j.get("message")))
    # WrongGPU's OptiScaler writes its own OptiScaler.log while the game runs; Update and Repair keep the player's in the backup
    for verb in ("update", "repair"):
        write(os.path.join(b, "OptiScaler.log"), log_b)
        if verb == "update":
            older_record(b)
            rc = t.run("update", exe, "--opti=" + opti, "--noscan", "--force")
        else:
            rc = repair(t, exe, opti)
        t.check(verb + " after WrongGPU's OptiScaler wrote its log: done, the player's OptiScaler.log still in the backup byte for byte",
                rc == 0 and os.path.isfile(backup) and read(backup) == log_a, (rc, t.j.get("message"), t.j.get("warnings")))
    write(os.path.join(b, "OptiScaler.log"), log_b)
    rc = t.run("uninstall", exe)
    t.check("...uninstall: the player's OptiScaler, and their OptiScaler.log, back byte for byte", rc == 0 and snapshot(root) == before, diff(before, snapshot(root)))
    shutil.rmtree(root, ignore_errors=True)

    # the move out of a wrong folder (an install next to a launcher stub, moved to the game by Repair): when a different file
    # is back where the install there moved the player's file aside, the move stops before anything changes
    root, b, exe = lt.game(t, "Kept Wrong Folder Game", validated=True, extra={"winmm.dll": theirs})
    stub, starter = os.path.join(root, "REDprelauncher.exe"), os.path.join(root, "Starter.exe")
    write(stub, "a launcher stub " * 50)
    write(starter, "mock starter exe " * 200)
    rc = t.run("install", starter, "--opti=" + opti, "--noscan", "--force", "--experimental", "--switch")
    rp = os.path.join(root, "wronggpu-setup", "install.record")
    ok = rc == 0 and os.path.isfile(rp)
    if ok:
        write(rp, read(rp).decode("utf-8").replace("exe\t" + starter + "\r\n", "exe\t" + stub + "\r\n"))
        os.remove(starter)
        other = fx.optiscaler_naming_wronggpu()
        write(os.path.join(root, "winmm.dll"), other)  # the player put another OptiScaler there since
        state = snapshot(root)
        rc2 = t.run("repair", root, "--opti=" + opti, "--noscan", "--force", "--switch")
        t.check("repair that moves an install out of a wrong folder, with a different file where the player's was moved aside: refused, nothing changed, backup kept",
                rc2 == 2 and "Nothing was changed" in (t.j.get("message") or "") and snapshot(root) == state and
                read(os.path.join(root, "wronggpu-setup", "backup", "winmm.dll")) == theirs, (rc2, t.j.get("message"), diff(state, snapshot(root))))
        os.remove(os.path.join(root, "winmm.dll"))
        rc3 = t.run("uninstall", root)
        t.check("...uninstall there once that file is away: the player's OptiScaler back byte for byte", rc3 == 0 and read(os.path.join(root, "winmm.dll")) == theirs,
                (rc3, t.j.get("message")))
    else:
        t.check("wrong-folder setup: an install in the game's root", False, (rc, t.j.get("message")))
    drop_test_entries(t, root)
    shutil.rmtree(root, ignore_errors=True)


def drop_test_entries(t, root):
    for k, v in t.reg_keys().items():  # Apps & features entries of games below root (test root only)
        if k.startswith("WrongGPU-") and v.get("InstallLocation", "").lower().startswith(root.lower()):
            subprocess.run(["reg", "delete", t.uninstall + "\\" + k, "/f"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def test_settings_file(t, opti):
    print("\n[0.6.1 coexist] wronggpu.ini (WrongGPU's settings) changed by the launcher or by hand: Repair keeps it; uninstall removes it and says so")
    root, b, exe = lt.game(t, "Settings File Game", validated=True)
    before = snapshot(root)
    ini = os.path.join(b, "wronggpu.ini")
    rc = install(t, exe, opti)
    rc2 = t.run("set", exe, "output_filter=quality9")
    write(ini, read(ini) + b"; edited by hand\r\n")
    rc3 = repair(t, exe, opti)
    t.check("repair after the settings changed (launcher and by hand): done, the hand edit kept", rc == 0 and rc2 == 0 and rc3 == 0 and b"edited by hand" in read(ini),
            (rc, rc2, rc3, t.j.get("message")))
    lines = rec_lines(b)
    existed = any(l.startswith("copy\t1\t") and l.endswith("\twronggpu.ini") for l in lines)
    write(ini, read(ini) + b"; edited again\r\n")
    rc = repair(t, exe, opti)
    t.check("repair again (wronggpu.ini now over an earlier one, changed again): done, never refused", rc == 0 and b"edited again" in read(ini), (existed, rc, t.j.get("message")))
    write(ini, read(ini) + b"; edited before the uninstall\r\n")
    rc = t.run("uninstall", exe)
    warns = t.j.get("warnings", [])
    t.check("uninstall with a hand-edited wronggpu.ini: removed (the game as before WrongGPU); the result says removed, never \"left in place\"",
            rc == 0 and snapshot(root) == before and any(w.startswith("wronggpu.ini had changed") and "removed with the rest" in w for w in warns) and
            not any("Left in place" in w and "wronggpu.ini" in w for w in warns), (warns, diff(before, snapshot(root))))
    shutil.rmtree(root, ignore_errors=True)


# ---------------------------------------------------------------- interrupted operations

def _lock(path):
    """Opens `path` with no sharing (as a running game holds its DLLs): (handle, close)."""
    import ctypes
    from ctypes import wintypes
    k = ctypes.WinDLL("kernel32", use_last_error=True)
    k.CreateFileW.restype = wintypes.HANDLE
    k.CreateFileW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, wintypes.LPVOID, wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE]
    h = k.CreateFileW(path, 0x80000000, 0, None, 3, 0x80, None)  # GENERIC_READ, no sharing, OPEN_EXISTING
    ok = h not in (None, wintypes.HANDLE(-1).value)
    return ok, (lambda: k.CloseHandle(h)) if ok else (lambda: None)


def _sleeper(t):
    """A tiny program that only waits (built here with cl, when it is on PATH), to stand in for a running game; None without cl."""
    cl = shutil.which("cl")
    if not cl:
        return None
    d = os.path.join(t.work, "sleeper")
    os.makedirs(d, exist_ok=True)
    src, exe = os.path.join(d, "sleeper.c"), os.path.join(d, "sleeper.exe")
    if not os.path.isfile(exe):
        write(src, "#include <windows.h>\nint main(void) { Sleep(120000); return 0; }\n")
        subprocess.run([cl, "/nologo", "/O1", src, "/Fe:" + exe, "/Fo:" + d + "\\"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                       cwd=d, creationflags=0x08000000 | 0x00004000)
    return exe if os.path.isfile(exe) else None


def test_interrupted(t, opti):
    print("\n[0.6.1 coexist] interrupted operations: a locked or read-only file, the game running, a stop part-way: nothing is lost")
    other = fx.unknown("another setup's OptiScaler plugin (synthetic .asi)")
    extra = dict(RR_FILES)
    extra["bin/x64/plugins/OtherSetup.asi"] = other
    root, b, exe = lt.game(t, "Interrupted Game", extra=extra)
    before = snapshot(root)
    rt = os.path.join(b, "wronggpu_runtime.dll")
    # a locked file (another program holds it open without sharing): the uninstall stops with the record and backup kept;
    # once it is free, the uninstall runs again and the game is as before WrongGPU
    rc = install(t, exe, opti, "--switch-plugins")
    state = snapshot(root)
    ok, unlock = _lock(rt)
    try:
        rc2 = t.run("uninstall", exe)
    finally:
        unlock()
    t.check("uninstall with a locked WrongGPU file: refused before anything changes (exit 2, \"Nothing was changed\"), record and backup kept",
            rc == 0 and ok and rc2 == 2 and "Nothing was changed" in (t.j.get("message") or "") and snapshot(root) == state, (rc, ok, rc2, t.j.get("message")))
    rc = t.run("uninstall", exe)
    t.check("...uninstall again once the file is free: the game as before WrongGPU, the other plugin back byte for byte", rc == 0 and snapshot(root) == before,
            (rc, t.j.get("message"), diff(before, snapshot(root))))
    # read-only files (WrongGPU's runtime, OptiScaler's ini): the uninstall removes them all the same
    rc = install(t, exe, opti, "--switch-plugins")
    for p in (rt, os.path.join(b, "OptiScaler.ini")):
        os.chmod(p, 0o444)
    rc2 = t.run("uninstall", exe)
    t.check("uninstall with read-only WrongGPU files: done, the game as before WrongGPU", rc == 0 and rc2 == 0 and snapshot(root) == before,
            (rc2, t.j.get("message"), diff(before, snapshot(root))))
    # a stop part-way through a Repair (after the earlier install's restore, the test build's hook): the earlier install is
    # undone completely, the moved plugin is back byte for byte, nothing half-installed
    rc = install(t, exe, opti, "--switch-plugins")
    t.env["WRONGGPU_TEST_FAIL_AFTER_RESTORE"] = "1"
    try:
        rc2 = repair(t, exe, opti)
    finally:
        t.env.pop("WRONGGPU_TEST_FAIL_AFTER_RESTORE", None)
    now = {k: v for k, v in snapshot(root).items() if "/wronggpu-setup/" not in k}
    want = {k: v for k, v in before.items() if "/wronggpu-setup/" not in k}
    t.check("repair stopped right after the earlier install's restore: the game's files as before WrongGPU (the other plugin back), no record",
            rc == 0 and rc2 != 0 and now == want and not os.path.isfile(os.path.join(b, "wronggpu-setup", "install.record")), (rc2, t.j.get("message"), diff(want, now)))
    for f in [os.path.join(dp, n) for dp, _, ns in os.walk(os.path.join(b, "wronggpu-setup")) for n in ns]:
        os.remove(f)  # (the settings the failed repair kept for a next try)
    # an uninstall killed part-way (simulated: the files it removes first are gone, the record and backup are still there):
    # run again, it finishes, and the game is as before WrongGPU
    rc = install(t, exe, opti, "--switch-plugins")
    lines = rec_lines(b)
    gone = [l.split("\t")[-1] for l in lines if l.startswith("copy\t0\t")][-2:]
    for rel in gone:
        if os.path.isfile(os.path.join(b, rel)):
            os.remove(os.path.join(b, rel))
    rc2 = t.run("uninstall", exe)
    t.check("uninstall after one killed part-way (%s already gone): finishes, the game as before WrongGPU" % ", ".join(gone), rc == 0 and len(gone) == 2 and rc2 == 0 and
            snapshot(root) == before, (rc2, t.j.get("message"), diff(before, snapshot(root))))
    # the game running: uninstall and repair refuse before anything changes; once it is closed, the uninstall runs
    sleeper = _sleeper(t)
    if not sleeper:
        print("  (skipped: the game-running cases need cl on PATH to build a stand-in program)")
    else:
        game_exe = read(exe)
        rc = install(t, exe, opti, "--switch-plugins")
        shutil.copyfile(sleeper, exe)
        state = snapshot(root)
        p = subprocess.Popen([exe], creationflags=0x08000000 | 0x00004000)
        try:
            rc2 = t.run("uninstall", exe)
            m2 = t.j.get("message") or ""
            rc3 = repair(t, exe, opti)
            m3 = t.j.get("message") or ""
            same = snapshot(root) == state
        finally:
            p.kill()
            p.wait()
        t.check("game running: uninstall and repair refused (exit 2, \"is running\"), nothing changed", rc == 0 and rc2 == 2 and rc3 == 2 and
                "is running" in m2 and "is running" in m3 and same, (rc2, m2, rc3, m3))
        write(exe, game_exe)
        rc = t.run("uninstall", exe)
        t.check("...closed: the uninstall runs, the game as before WrongGPU", rc == 0 and snapshot(root) == before, diff(before, snapshot(root)))
    print("  (a full disk is not simulated: it cannot be done safely here without filling a real drive)")
    shutil.rmtree(root, ignore_errors=True)


def run_all(t, opti, full):
    test_optiscaler_named_wronggpu(t, opti)
    test_proxy_replaced(t, opti)
    test_repair_after_switch(t, opti)
    test_opti_ini_ownership(t, opti)
    test_players_ini_edited_in_menu(t, opti)
    test_uninstall_by_hash(t, opti)
    test_other_plugins(t, opti)
    test_leftovers_other_setup(t, opti)
    test_rr_plugin_leftover(t, opti)
    test_stays_out_card(t, opti)
    test_backups_kept(t, opti)
    test_settings_file(t, opti)
    test_interrupted(t, opti)
