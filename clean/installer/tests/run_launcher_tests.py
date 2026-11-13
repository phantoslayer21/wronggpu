#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Tests of the WrongGPU launcher's verbs (build/launcher-design.md section 7.2 and 7.3).

Written for this project; Python standard library only. Runs WrongGPU.exe (a WRONGGPU_TESTING build made with the
installer suite's test map: registry writes stay in HKCU\\Software\\WrongGPU-CandidateTests) and the restorer it writes
into each game against GENERATED fixture game folders only (names and synthetic contents; never a real game, never a
real third-party file). Every install is undone and the folder compared byte for byte with its snapshot.

    python run_launcher_tests.py --launcher <WrongGPU.exe> --work <dir> [--baseline <0.5.1 WrongGPU-Setup.exe>]
                                 [--hip <mock amdhip64_7.dll>] [--synth-f <folder of synth-common-v0.bin>]
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time

os.environ["WRONGGPU_TESTING"] = "1"
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import mockpe  # noqa: E402  (synthetic PE files with a version resource)
import run_installer_tests as suite  # noqa: E402  (mock content, snapshots, the synthetic model case)
from run_installer_tests import read, sha, snapshot, diff, write  # noqa: E402


def sha256_of(path):
    """A job or batch file's SHA-256, as the launcher passes it to the elevated worker (apply --sha256=)."""
    return sha(read(path))


OPTI_INI = ("[Upscalers]\nDx12Upscaler=auto\n[Libraries]\nFfxDx12SRPath=auto\n[FSR]\nFsr4Update=auto\nUpscalerIndex=auto\n"
            "[Menu]\nShortcutKey=auto\nShowFps=auto\n")


class T:
    def __init__(self, exe, work, model, hip):
        self.exe, self.work, self.model = exe, work, model
        self.passed, self.failures = 0, []
        self.data = os.path.join(work, "data")
        self.log = os.path.join(work, "launcher-test.log")
        self.out = os.path.join(work, "out.json")
        self.env = dict(os.environ, LOCALAPPDATA=os.path.join(work, "localappdata"), ProgramData=os.path.join(work, "programdata"))
        if hip:
            self.env["WRONGGPU_HIP_RUNTIME"] = hip
        for k in [k for k in self.env if k.upper().startswith("WG_") or k.upper() in ("HIP_PATH", "ROCM_PATH", "LLVM_PATH", "HIP_DEVICE_LIB_PATH")]:
            self.env.pop(k)
        os.makedirs(self.env["LOCALAPPDATA"], exist_ok=True)
        self.uninstall = suite.uninstall_root(self.env["LOCALAPPDATA"])  # this run's Apps & features root (the build's own)
        self.j = {}

    def check(self, name, cond, detail=""):
        if cond:
            self.passed += 1
            print("  PASS ", name)
        else:
            self.failures.append(name + (": " + str(detail)[:1500] if detail else ""))
            print("  FAIL ", name, str(detail)[:1500])

    def run(self, *args, exe=None, data=True, timeout=600, cwd=None):
        """WrongGPU.exe <args> --data --log --out; the JSON result in self.j; returns the exit code. cwd: its current directory."""
        if os.path.exists(self.out):
            os.remove(self.out)
        cmd = [exe or self.exe] + list(args) + ["--log=" + self.log, "--out=" + self.out] + (["--data=" + self.data] if data else [])
        p = subprocess.run(cmd, env=self.env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout, cwd=cwd)
        try:
            with open(self.out, encoding="utf-8") as f:
                self.j = json.load(f)
        except (OSError, ValueError):
            self.j = {}
        return p.returncode

    def raw(self, cmd, timeout=600):
        return subprocess.run(cmd, env=self.env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout).returncode

    def logtail(self, n=1500):
        try:
            return read(self.log).decode("utf-8", "replace")[-n:]
        except OSError:
            return ""

    def reg_keys(self, root=None):
        root = root or self.uninstall
        p = subprocess.run(["reg", "query", root, "/s"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        keys, cur = {}, None
        for line in p.stdout.decode("utf-8", "replace").splitlines():
            if line.upper().startswith("HKEY_"):
                name = line.strip().split("\\")[-1]
                cur = keys.setdefault(name, {}) if line.strip().lower() != root.lower().replace("hkcu", "hkey_current_user") else None
            elif cur is not None and line.startswith("    "):
                parts = [x for x in line.strip().split("    ") if x]
                if len(parts) >= 3:
                    cur[parts[0]] = parts[-1].strip()
        return keys

    def entries_for(self, folder):
        return {k: v for k, v in self.reg_keys().items() if v.get("InstallLocation", "").lower() == folder.lower()}


def wait_gone(path, seconds=20):
    for _ in range(seconds * 5):
        if not os.path.exists(path):
            return True
        time.sleep(0.2)
    return not os.path.exists(path)


def removed_folders(t):
    """What a launcher uninstall moved aside in the user folder (WrongGPU.removed-*): deleted once nothing runs from it."""
    lad = t.env["LOCALAPPDATA"]
    return [n for n in os.listdir(lad) if n.startswith("WrongGPU.removed-")] if os.path.isdir(lad) else []


def wait_removed(t, seconds=20):
    for _ in range(seconds * 5):
        if not removed_folders(t):
            return True
        time.sleep(0.2)
    return not removed_folders(t)


# ---------------------------------------------------------------- fixtures (generated: names and synthetic bytes only)

def game(t, name, exe="Game.exe", validated=False, extra=None):
    """A D3D12 game folder: bin\\x64\\<exe>, AMD's FSR 3.1 DLL (mock), the synthetic DLSS model file, game files."""
    root = os.path.join(t.work, "games", name)
    shutil.rmtree(root, ignore_errors=True)
    b = os.path.join(root, "bin", "x64")
    write(os.path.join(b, "Cyberpunk2077.exe" if validated else exe), "mock game exe " * 200 + name)
    write(os.path.join(b, "amd_fidelityfx_dx12.dll"), suite.AMD_V1)
    write(os.path.join(b, "nvngx_dlss.dll"), t.model.file)
    write(os.path.join(b, "steam_api64.dll"), "untouched game file")
    write(os.path.join(root, "launcher.exe"), "a store launcher")
    for rel, data in (extra or {}).items():
        write(os.path.join(root, rel), data)
    return root, b, os.path.join(b, "Cyberpunk2077.exe" if validated else exe)


def opti_package(t, name="opti-package"):
    p = os.path.join(t.work, name)
    shutil.rmtree(p, ignore_errors=True)
    write(os.path.join(p, "OptiScaler.dll"), b"MZ mock OptiScaler (synthetic)")
    write(os.path.join(p, "OptiScaler.ini"), OPTI_INI)
    return p


def record_text(b):
    return read(os.path.join(b, "wronggpu-setup", "install.record")).decode("utf-8", "replace")


def ini(b, name="wronggpu.ini"):
    p = os.path.join(b, name)
    return read(p).decode("utf-8", "replace") if os.path.isfile(p) else ""


def options_line(text):
    for line in text.replace("\r", "").split("\n"):
        s = line.strip()
        if s.lower().startswith("runtime_options") and "=" in s:
            return s.split("=", 1)[1].strip()
    return None


def managed_list(t):
    p = os.path.join(t.env["LOCALAPPDATA"], "WrongGPU", "managed-games.txt")
    return read(p).decode("utf-8", "replace").lower() if os.path.isfile(p) else ""


def record_entry_sha(b, rel):
    for line in record_text(b).replace("\r", "").split("\n"):
        f = line.split("\t")
        if len(f) == 5 and f[0] == "copy" and f[4].lower() == rel.lower():
            return f[3]
    return None


# ---------------------------------------------------------------- tests

def test_basics(t):
    print("\n[basics] version, hipcheck, update check off, portable paths")
    rc = t.run("version")
    t.check("version: exit 0 and a version", rc == 0 and t.j.get("version"), t.j)
    # rc5: the link time that tells builds of one version apart is the build's real time (no /Brepro hash): within a day of
    # the exe's file time, and the same moment in words
    built = t.j.get("built_unix", 0)
    t.check("version: built_unix is the build's link time (within a day of the exe's file time) and 'built' says it in UTC",
            rc == 0 and built and abs(built - os.path.getmtime(t.exe)) < 86400 and t.j.get("built", "").endswith(" UTC"), (built, t.j.get("built"), os.path.getmtime(t.exe)))
    emb = t.j.get("embedded", [])
    t.check("no OptiScaler embedded (owner 2026-11-02: the launcher downloads WrongGPU's edition of OptiScaler, never ships it)",
            rc == 0 and emb and not [n for n in emb if "optiscaler" in n.lower()], emb)
    rc = t.run("hipcheck")
    t.check("hipcheck: exit 0 whatever the HIP runtime (it is optional now)", rc == 0 and "note" in t.j, t.j)
    rc = t.run("update-check")
    t.check("update check: off by default, no request", rc == 0 and not t.j.get("checked") and "off" in t.j.get("message", ""), t.j)
    rc = t.run("set", "--global", "update_check=1")
    rc2 = t.run("update-check")
    t.check("update check on without a URL: no request", rc == 0 and rc2 == 0 and not t.j.get("checked") and "no release address" in t.j.get("message", ""), t.j)
    rc = t.run("set", "--global", "update_url=http://example.invalid/x")
    t.check("update URL must be https (exit 10)", rc == 10, t.j)
    rc = t.run("set", "--global", "update_check=0")
    rc = t.run("get", "--global")
    g = t.j.get("global", {})
    t.check("get --global: update check off again, no URL, Play button on", rc == 0 and g.get("update_check") is False and g.get("update_url") == "" and g.get("play_button") is True, t.j)
    rc = t.run("frobnicate")
    t.check("an unknown verb: exit 10", rc == 10, t.j)
    # portable mode: a marker next to the exe, data next to it
    port = os.path.join(t.work, "portable")
    shutil.rmtree(port, ignore_errors=True)
    os.makedirs(port)
    shutil.copy2(t.exe, os.path.join(port, "WrongGPU.exe"))
    write(os.path.join(port, "WrongGPU.portable"), "portable\r\n")
    rc = t.run("get", "--global", exe=os.path.join(port, "WrongGPU.exe"), data=False)
    g = t.j.get("global", {})
    t.check("portable: data next to the exe (WrongGPU-data)", rc == 0 and g.get("portable") is True and
            g.get("data_dir", "").lower() == os.path.join(port, "WrongGPU-data").lower(), t.j)
    shutil.rmtree(port, ignore_errors=True)


def other_mod_keys():
    """The other upscaler mod's detection keys, read from their one source line (src/wg_core/constants.h kOtherMods):
    (signature, [file names], the words of its signature a player text must never contain). Brief 17."""
    text = read(os.path.join(HERE, "..", "src", "wg_core", "constants.h")).decode("utf-8")
    m = re.search(r'\{L"Another upscaler mod", "([^"]+)", \{L"([^"]+)", L"([^"]+)"', text)
    sig, files = m.group(1), [m.group(2), m.group(3)]
    words = {sig, sig.replace("_", " "), sig.replace("_", "-"), sig[4:].replace("_", " "), sig[4:].replace("_", "-")}
    return sig, files, sorted(words)


def names_other_mod(text):
    """True when a text players read names the other upscaler mod or its marker files (Brief 17)."""
    low = text.lower()
    return any(w in low for w in other_mod_keys()[2] + [f.lower() for f in other_mod_keys()[1]])


def test_scan_and_refusals(t, opti):
    print("\n[scan] statuses of generated games; refusals change nothing")
    game(t, "Eligible Game")
    ac_root, _, ac_exe = game(t, "Anticheat Game", extra={"bin/x64/EasyAntiCheat/EasyAntiCheat_x64.dll": "mock anti-cheat"})
    nr_root, _, nr_exe = game(t, "Other Mod Game", extra={"bin/x64/" + other_mod_keys()[1][0]: "[mock]\n"})  # another upscaler mod's marker file
    po_root, _, _ = game(t, "Player Opti Game", extra={"bin/x64/winmm.dll": b"MZ OptiScaler (the player's, synthetic)", "bin/x64/OptiScaler.ini": "[Menu]\nShowFps=true\n"})
    # another game's BattlEye in the same library, and one in the library folder itself: not the other games' anti-cheat
    write(os.path.join(t.work, "games", "Shooter With BattlEye", "BattlEye", "BEClient_x64.dll"), "mock anti-cheat")
    write(os.path.join(t.work, "games", "BattlEye", "BEClient_x64.dll"), "mock anti-cheat")
    rc = t.run("scan", "--noscan", "--library=" + os.path.join(t.work, "games"))
    games = {g["name"]: g for g in t.j.get("games", [])}
    t.check("scan: exit 0, the generated games listed", rc == 0 and {"Eligible Game", "Anticheat Game", "Other Mod Game", "Player Opti Game"} <= set(games), list(games))
    t.check("scan: statuses (eligible, refused, foreign-mod, players-optiscaler)",
            games.get("Eligible Game", {}).get("status") == "eligible" and games.get("Anticheat Game", {}).get("status") == "refused" and
            games.get("Other Mod Game", {}).get("status") == "foreign-mod" and games.get("Player Opti Game", {}).get("status") == "players-optiscaler",
            {k: v.get("status") for k, v in games.items()})
    # the marker's full path: shortened in the middle in the status line, whole in the finding and the refusal
    marker = os.path.join(ac_root, "bin", "x64", "EasyAntiCheat")
    st = games.get("Anticheat Game", {}).get("status_text", "")
    inner = st[len("Refused: anti-cheat ("):-1] if st.startswith("Refused: anti-cheat (") and st.endswith(")") else ""
    t.check("scan: the anti-cheat status names the marker's path (shortened in the middle when long)",
            inner and (inner.lower() == marker.lower() if len(marker) <= 60 else len(inner) <= 60 and "…" in inner and
                       marker.lower().startswith(inner.split("…")[0].lower()) and marker.lower().endswith(inner.split("…")[1].lower())), (st, marker))
    acf = [f for f in games.get("Anticheat Game", {}).get("coexistence", []) if f.get("tool") == "anti-cheat"]
    t.check("scan: the anti-cheat finding has the full path (evidence and message)",
            len(acf) == 1 and acf[0].get("evidence", "").lower() == marker.lower() and marker.lower() in acf[0].get("message", "").lower(), acf)
    before_ac, before_nr = snapshot(ac_root), snapshot(nr_root)
    rc = t.run("install", ac_exe, "--opti=" + opti, "--noscan", "--experimental", "--force")
    t.check("anti-cheat game: refused (exit 4), nothing changed", rc == 4 and snapshot(ac_root) == before_ac, (rc, t.j.get("message")))
    t.check("anti-cheat refusal: 'Anti-cheat found: <full path> - games with anti-cheat are never supported'",
            t.j.get("message", "").lower() == ("Anti-cheat found: " + marker + " - games with anti-cheat are never supported [WG-SETUP-4]").lower(), t.j.get("message"))
    rc = t.run("install", nr_exe, "--opti=" + opti, "--noscan", "--experimental", "--force", "--switch")
    t.check("other-mod game: refused even with --switch (exit 6), nothing changed", rc == 6 and t.j.get("message", "").startswith("Another upscaler mod") and
            snapshot(nr_root) == before_nr, (rc, t.j.get("message")))
    # owner 2026-11-04 (Brief 17): it is found by its file names, and nothing players read names it (nor its files)
    nr = games.get("Other Mod Game", {})
    t.check("other-mod game: the status says 'Other upscaler mod: not touched'", nr.get("status_text") == "Other upscaler mod: not touched", nr.get("status_text"))
    t.check("other-mod game: no text names the other mod or its files (status, findings, refusal)",
            not names_other_mod(json.dumps(nr)) and not names_other_mod(t.j.get("message", "")), (nr, t.j.get("message")))
    for r in (ac_root, nr_root, po_root, *(os.path.join(t.work, "games", n) for n in ("Eligible Game", "Shooter With BattlEye", "BattlEye"))):
        shutil.rmtree(r, ignore_errors=True)


def test_install_settings_uninstall(t, opti):
    print("\n[install] install, record v2, owned-files.json, restorer, settings, log tail, update, roll back, uninstall")
    root, b, exe = game(t, "Main Game", validated=True)
    before = snapshot(root)
    rc = t.run("status", exe, "--noscan")
    t.check("status before install: eligible", rc == 0 and t.j.get("game", {}).get("status") == "eligible", t.j)
    rc = t.run("install", exe, "--opti=" + opti, "--noscan", "--force")
    t.check("install: exit 0", rc == 0, (t.j.get("message"), t.logtail()))
    if rc != 0:
        return
    rec = record_text(b)
    t.check("record v2: header, payload, proxy and owner lines",
            rec.startswith("WRONGGPU-RECORD 2\r\n") and "\npayload\t" in rec and "\nproxy\tdxgi.dll\t" in rec and "\nowner\tcopy\twronggpu_runtime.dll\twg" in rec and
            "\nowner\tcopy\tdxgi.dll\twg-optibundle" in rec, rec[:1500])
    try:
        owned = json.loads(read(os.path.join(b, "wronggpu-setup", "owned-files.json")).decode("utf-8"))
    except Exception as e:  # noqa: BLE001
        owned = {"error": str(e)}
    t.check("owned-files.json: valid JSON, every recorded file with its owner", owned.get("format") == "wronggpu-owned-files/1" and
            any(f["path"] == "wronggpu_runtime.dll" and f["owner"] == "wg" for f in owned.get("files", [])), owned)
    un = os.path.join(b, "wronggpu-setup", "uninstall.exe")
    t.check("uninstall.exe is the small restorer, not the launcher", os.path.isfile(un) and os.path.getsize(un) < os.path.getsize(t.exe) // 4,
            (os.path.getsize(un) if os.path.isfile(un) else None, os.path.getsize(t.exe)))
    t.check("profile.ini written", os.path.isfile(os.path.join(b, "wronggpu-setup", "profile.ini")))
    t.check("the game is on the managed list", b.lower() in managed_list(t), managed_list(t))
    t.check("launcher not installed: the game keeps its own Apps & features entry (uninstallable)", len(t.entries_for(b)) == 1, t.entries_for(b))
    rc = t.run("status", exe)
    g = t.j.get("game", {})
    t.check("status: installed, record v2, not run yet", rc == 0 and g.get("status") == "installed" and g.get("record_v1") is False and
            g.get("last_run", {}).get("found") is False, g)
    # settings
    rc = t.run("set", exe, "output_filter=quality9", "executor=dx12", "sparkle_guard=0.4")
    text = ini(b)
    t.check("set: exit 0; runtime_options one line without spaces, managed items appended", rc == 0 and
            (options_line(text) or "").endswith("executor=dx12;output_filter=quality9;sparkle_guard=0.4") and " " not in (options_line(text) or " "),
            (rc, options_line(text), t.j.get("message")))
    t.check("set: the record knows the new wronggpu.ini (uninstall can prove it is ours)", record_entry_sha(b, "wronggpu.ini") == sha(read(os.path.join(b, "wronggpu.ini"))))
    before_bad = ini(b)
    rc = t.run("set", exe, "executor=warp")
    t.check("set a bad value: exit 10, the file unchanged", rc == 10 and ini(b) == before_bad, (rc, t.j))
    rc = t.run("set", exe, "nosuch=1")
    t.check("set an unknown key: exit 10", rc == 10, t.j)
    rc = t.run("set", exe, "fps_overlay=on")
    opt = ini(b, "OptiScaler.ini")
    t.check("fps_overlay=on: ShowFps=true in our OptiScaler.ini, record updated", rc == 0 and "ShowFps=true" in opt and
            record_entry_sha(b, "OptiScaler.ini") == sha(read(os.path.join(b, "OptiScaler.ini"))), (rc, opt))
    rc = t.run("get", exe)
    s = t.j.get("settings", {})
    t.check("get: profile and file agree", rc == 0 and s.get("output_filter", {}).get("profile") == "quality9" and s.get("output_filter", {}).get("file") == "quality9" and
            s.get("fps_overlay", {}).get("file") == "on", s)
    # an edit outside the launcher
    write(os.path.join(b, "wronggpu.ini"), ini(b).replace("sparkle_guard=0.4", "sparkle_guard=0.4;pre_exposure=original"))
    rc = t.run("get", exe)
    t.check("an outside edit is reported (pre_exposure=original)", rc == 0 and any(x["key"] == "pre_exposure" and x["file"] == "original" for x in t.j.get("outside_edits", [])), t.j.get("outside_edits"))
    rc = t.run("set", exe, "pre_exposure=scale")  # revert: the profile wins
    t.check("revert: the file follows the profile again", rc == 0 and "pre_exposure" not in (options_line(ini(b)) or ""), options_line(ini(b)))
    # the log tail
    write(os.path.join(b, "wronggpu-fsr.log"),
          "[11:00:00.000 9] [amd_fidelityfx_dx12] ==== session start: Cyberpunk2077.exe (process 9) ====\r\n"
          "[11:00:00.100 9] [amd_fidelityfx_dx12] settings: upscaler=network runtime_dll=wronggpu_runtime.dll\r\n"
          "[11:00:01.000 9] [info] network runtime ready: executor dx12, sync inline, max render 1280x720\r\n")
    rc = t.run("status", exe)
    lr = t.j.get("game", {}).get("last_run", {})
    t.check("status: last run from wronggpu-fsr.log (Model M on dx12)", rc == 0 and lr.get("found") and lr.get("executor") == "dx12" and lr.get("runtime_ready"), lr)
    # rc5: at the same version the payload line decides (another build's files are an update), never the version string;
    # status only reads (no write before the player's Update / Repair)
    rec_path = os.path.join(b, "wronggpu-setup", "install.record")
    rec_same = read(rec_path)
    g0 = t.j.get("game", {})
    t.check("same build (the payload line is this launcher's): installed, healthy, no update, other_build false",
            g0.get("status") == "installed" and g0.get("other_build") is False, (g0.get("status"), g0.get("other_build"), g0.get("status_text")))
    rec_lines = rec_same.decode("utf-8").split("\r\n")
    pay = [l for l in rec_lines if l.startswith("payload\t")]
    t.check("the record has a payload line (version, runtime, front end, plugin)", len(pay) == 1 and len(pay[0].split("\t")) == 5, pay)
    if pay:
        f = pay[0].split("\t")
        other = "\t".join([f[0], f[1], "0" * 64, f[3], f[4]])
        write(rec_path, "\r\n".join(other if l == pay[0] else l for l in rec_lines))
        rc = t.run("status", exe)
        g1 = t.j.get("game", {})
        t.check("same version, another build's runtime in the payload line: update offered (\"Installed V (another build) -> update V\"), status writes nothing",
                rc == 0 and g1.get("status") == "update-available" and g1.get("other_build") is True and "(another build)" in g1.get("status_text", "") and
                read(rec_path) != rec_same and "0" * 64 in read(rec_path).decode("utf-8"), (rc, g1.get("status"), g1.get("other_build"), g1.get("status_text")))
        write(rec_path, "\r\n".join(l for l in rec_lines if not l.startswith("payload\t")))
        rc = t.run("status", exe)
        g2 = t.j.get("game", {})
        t.check("same version, a record with no payload line: another build, update offered",
                rc == 0 and g2.get("status") == "update-available" and g2.get("other_build") is True, (rc, g2.get("status"), g2.get("other_build"), g2.get("status_text")))
        rc = t.run("update", exe, "--opti=" + opti, "--noscan", "--force")
        rc2 = t.run("status", exe)
        t.check("the player's Update at the same version: exit 0, this launcher's payload line written, installed again",
                rc == 0 and rc2 == 0 and pay[0] in read(rec_path).decode("utf-8").split("\r\n") and t.j.get("game", {}).get("status") == "installed",
                (rc, t.j.get("game", {}).get("status"), t.logtail(1500)))
        write(rec_path, rec_same)
        rc = t.run("status", exe)
        t.check("the record back as installed: installed again", rc == 0 and t.j.get("game", {}).get("status") == "installed", t.j.get("game", {}).get("status"))
    # update from an older version, then roll back and forward
    lines = read(rec_path).decode("utf-8").split("\r\n")
    lines = ["version\t0.5.1" if l.startswith("version\t") else l for l in lines]
    write(rec_path, "\r\n".join(lines))
    rc = t.run("status", exe)
    t.check("an older record: update available", rc == 0 and t.j.get("game", {}).get("status") == "update-available", t.j.get("game", {}).get("status"))
    t.check("an older record: an older version, not \"another build\" (other_build false, no such words)",
            t.j.get("game", {}).get("other_build") is False and "another build" not in t.j.get("game", {}).get("status_text", ""), t.j.get("game", {}))
    # a build without Preset K: an older install set up to start on Preset K (the plugin's choice) moves to Model M, said in the log
    no_k = "Preset K isn't in this version".encode("utf-16-le") in read(t.exe)
    if no_k:
        write(os.path.join(b, "wronggpu-asi.ini"), "; WrongGPU OptiScaler plugin\r\nselection=wronggpu-k\r\n")
    rc = t.run("update", exe, "--opti=" + opti, "--noscan", "--force")
    if no_k:
        asi_ini = os.path.join(b, "wronggpu-asi.ini")
        t.check("update of a Preset K install (no Preset K in this build): starts on Model M, the log says so",
                rc == 0 and "now starts on Model M" in t.logtail(20000) and "model=M" in (options_line(ini(b)) or "") and
                (not os.path.isfile(asi_ini) or "selection=wronggpu\r\n" in read(asi_ini).decode()),
                (rc, options_line(ini(b)), t.logtail(1500)))
    prev_txt = os.path.join(b, "wronggpu-setup", "previous", "previous.txt")
    t.check("update: exit 0; the earlier version kept in previous\\", rc == 0 and os.path.isfile(prev_txt) and "version=0.5.1" in read(prev_txt).decode(), (rc, t.j.get("message"), t.logtail()))
    rec = record_text(b)
    t.check("update: the settings survive (profile re-applied)", "output_filter=quality9" in (options_line(ini(b)) or "") and "ShowFps=true" in ini(b, "OptiScaler.ini"),
            (options_line(ini(b)), ini(b, "OptiScaler.ini")))
    t.check("update: record at this version", "\nversion\t0.5.1" not in rec, rec[:300])
    rc = t.run("rollback", exe)
    rec = record_text(b)
    t.check("roll back: exit 0, record says 0.5.1, the newer version kept for later", rc == 0 and "\nversion\t0.5.1\r\n" in rec and
            "version=" in read(prev_txt).decode() and "version=0.5.1" not in read(prev_txt).decode(), (rc, t.j.get("message")))
    rc = t.run("rollback", exe)
    t.check("roll back again: back to the newer version", rc == 0 and "\nversion\t0.5.1\r\n" not in record_text(b), t.j.get("message"))
    rc = t.run("repair", exe, "--opti=" + opti, "--noscan", "--force")
    t.check("repair: exit 0, settings kept", rc == 0 and "output_filter=quality9" in (options_line(ini(b)) or ""), (rc, t.j.get("message")))
    # a missing settings file is one problem, said once (the record lists wronggpu.ini as a file of the install too)
    os.remove(os.path.join(b, "wronggpu.ini"))
    rc = t.run("status", exe)
    g = t.j.get("game", {})
    t.check("status: a missing wronggpu.ini is one problem, said once (needs repair)", rc == 0 and g.get("status") == "needs-repair" and
            g.get("problems") == ["wronggpu.ini is missing"], g.get("problems"))
    # Repair on an older WrongGPU puts this release's files back: an update too, and its message says so
    lines = read(rec_path).decode("utf-8").split("\r\n")
    write(rec_path, "\r\n".join("version\t0.5.1" if l.startswith("version\t") else l for l in lines))
    rc = t.run("repair", exe, "--opti=" + opti, "--noscan", "--force")
    t.check("repair of an older version: wronggpu.ini back with the settings, the message says it updated from 0.5.1",
            rc == 0 and "output_filter=quality9" in (options_line(ini(b)) or "") and "updated from 0.5.1 to" in t.j.get("message", ""), (rc, t.j.get("message")))
    # the elevated worker's job (run here without elevation: the same code path). --sha256: the job file's; the worker
    # writes its result next to it, result-<name>.txt for job-<name>.txt
    job = os.path.join(t.work, "job-main.txt")
    result = os.path.join(t.work, "result-main.txt")
    write(job, "op=apply\r\ndir=%s\r\nlocalappdata=%s\r\nprofile.output_filter=analytic\r\nprofile.fps_overlay=off\r\n" % (b, t.env["LOCALAPPDATA"]))
    rc = t.run("apply", "--job=" + job, "--sha256=" + sha256_of(job))
    res = read(result).decode("utf-8", "replace") if os.path.isfile(result) else ""
    t.check("apply --job: the worker writes the result; the job's settings are in the files", rc == 0 and "code=0" in res and
            "output_filter=analytic" in (options_line(ini(b)) or "") and "ShowFps=false" in ini(b, "OptiScaler.ini"), (rc, res))
    write(job, "op=apply\r\ndir=relative\\folder\r\nlocalappdata=%s\r\n" % t.env["LOCALAPPDATA"])
    rc = t.run("apply", "--job=" + job, "--sha256=" + sha256_of(job))
    t.check("apply --job with a relative path: refused (exit 10)", rc == 10 and "code=10" in read(result).decode(), rc)
    os.remove(result)
    rc = t.run("apply", "--job=" + job, "--sha256=" + "0" * 64)
    t.check("apply --job with another SHA-256: refused (exit 10), nothing written", rc == 10 and not os.path.exists(result), rc)
    # several jobs in one worker run (the window's one permission prompt for every protected folder): each writes its result, in order
    lad = t.env["LOCALAPPDATA"]
    jobs = [os.path.join(t.work, "batch-job-%d.txt" % i) for i in (1, 2)]
    batch = os.path.join(t.work, "batch-B.txt")
    results = [os.path.join(t.work, "result-B-%d.txt" % i) for i in (1, 2)]   # the worker's own names
    for f in results:
        if os.path.exists(f):
            os.remove(f)
    write(jobs[0], "op=apply\r\ndir=%s\r\nlocalappdata=%s\r\nprofile.output_filter=quality5\r\n" % (b, lad))
    write(jobs[1], "op=apply\r\ndir=%s\r\nlocalappdata=%s\r\nprofile.output_filter=analytic\r\nprofile.fps_overlay=off\r\n" % (b, lad))
    write(batch, "stop=1\r\n" + "".join("job=%s\r\nsha256=%s\r\n" % (j, sha256_of(j)) for j in jobs))
    rc = t.run("apply", "--batch=" + batch, "--sha256=" + sha256_of(batch))
    res = [read(f).decode("utf-8", "replace") if os.path.isfile(f) else "" for f in results]
    t.check("apply --batch: both jobs run in order, each writes its result (the last one's settings are in the files)",
            rc == 0 and all("code=0" in r for r in res) and t.j.get("ran") == 2 and "output_filter=analytic" in (options_line(ini(b)) or ""), (rc, res, t.j))
    for f in results:
        if os.path.exists(f):
            os.remove(f)
    write(jobs[0], "op=apply\r\ndir=relative\\folder\r\nlocalappdata=%s\r\n" % lad)
    write(jobs[1], "op=apply\r\ndir=%s\r\nlocalappdata=%s\r\nprofile.output_filter=quality5\r\n" % (b, lad))
    write(batch, "stop=1\r\n" + "".join("job=%s\r\nsha256=%s\r\n" % (j, sha256_of(j)) for j in jobs))
    rc = t.run("apply", "--batch=" + batch, "--sha256=" + sha256_of(batch))
    res = [read(f).decode("utf-8", "replace") if os.path.isfile(f) else "" for f in results]
    t.check("apply --batch with stop=1: the first fails (exit 10), the second is not run and its result says so (code -1, the files unchanged)",
            rc == 10 and "code=10" in res[0] and "code=-1" in res[1] and t.j.get("ran") == 1 and "output_filter=analytic" in (options_line(ini(b)) or ""), (rc, res, t.j))
    # collect logs
    zpath = os.path.join(t.work, "support.zip")
    rc = t.run("collect", exe, "--zip=" + zpath)
    t.check("collect: a support zip", rc == 0 and os.path.isfile(zpath), t.j)
    # uninstall through the launcher
    rc = t.run("uninstall", exe)
    after = snapshot(root)
    t.check("uninstall: exit 0, the game folder byte for byte as before", rc == 0 and after == before, (rc, t.j.get("message"), diff(before, after)))
    t.check("uninstall: off the managed list, its entry gone", b.lower() not in managed_list(t)
            and not t.entries_for(b))
    # again, uninstalled by the restorer in the game folder
    rc = t.run("install", exe, "--opti=" + opti, "--noscan", "--force")
    rc2 = t.raw([os.path.join(b, "wronggpu-setup", "uninstall.exe"), "/S", "/LOG=" + t.log])
    wait_gone(os.path.join(b, "wronggpu-setup"))
    after = snapshot(root)
    t.check("the restorer (uninstall.exe /S): exit 0, byte for byte as before", rc == 0 and rc2 == 0 and after == before, (rc, rc2, diff(before, after)))
    shutil.rmtree(root, ignore_errors=True)


def test_slot(t):
    """OptiScaler 10 (a package with an OptiScaler\\ folder, as the WrongGPU edition): WrongGPU's front end in OptiScaler's
    custom upscaler slot, selected by its own index; no menu plugin, no FidelityFX library route (slot fix 2026-11-03)."""
    print("\n[slot] OptiScaler 10: the front end in OptiScaler\\custom\\WrongGPU\\ffx, Dx12Upscaler=WrongGPU, [WrongGPU] UpscalerIndex=0")
    pkg = os.path.join(t.work, "opti10-package")
    shutil.rmtree(pkg, ignore_errors=True)
    write(os.path.join(pkg, "OptiScaler.dll"), b"MZ mock OptiScaler 10 (synthetic)")
    write(os.path.join(pkg, "OptiScaler.ini"), OPTI_INI)
    write(os.path.join(pkg, "OptiScaler", "libxess.dll"), b"MZ mock OptiScaler 10 folder file (synthetic)")
    write(os.path.join(pkg, "amd_fidelityfx_upscaler_dx12.dll"), b"MZ mock FSR 4 of the package (synthetic)")
    root, b, exe = game(t, "Slot Game", validated=True)
    before = snapshot(root)
    mark = os.path.getsize(t.log) if os.path.isfile(t.log) else 0
    rc = t.run("install", exe, "--opti=" + pkg, "--noscan", "--force")
    inst_log = read(t.log)[mark:].decode("utf-8", "replace") if os.path.isfile(t.log) else ""
    t.check("install: exit 0", rc == 0, (t.j.get("message"), t.logtail()))
    if rc != 0:
        return
    slot = os.path.join(b, "OptiScaler", "custom", "WrongGPU", "ffx", "amd_fidelityfx_upscaler_dx12.dll")
    t.check("the front end is in the custom slot; no wronggpu_ffx_dx12.dll, no plugins\\WrongGPU.asi, no wronggpu-asi.ini",
            os.path.isfile(slot) and not os.path.exists(os.path.join(b, "wronggpu_ffx_dx12.dll")) and
            not os.path.exists(os.path.join(b, "plugins", "WrongGPU.asi")) and not os.path.exists(os.path.join(b, "wronggpu-asi.ini")),
            sorted(os.listdir(b)))
    opt = ini(b, "OptiScaler.ini")
    t.check("OptiScaler.ini: Dx12Upscaler=WrongGPU, FfxDx12SRPath=auto, [WrongGPU] UpscalerIndex=0, [FSR] UpscalerIndex untouched",
            "Dx12Upscaler=WrongGPU" in opt and "FfxDx12SRPath=auto" in opt and "[WrongGPU]\r\nUpscalerIndex=0" in opt.replace("\r\n", "\n").replace("\n", "\r\n") and
            "[FSR]\r\nFsr4Update=false\r\nUpscalerIndex=auto" in opt.replace("\r\n", "\n").replace("\n", "\r\n"), opt)
    t.check("the install says WrongGPU is OptiScaler's own upscaler entry (no FSR wording)",
            "WrongGPU is its own upscaler there, selected" in inst_log and "default FSR upscaler" not in inst_log and
            "AMD's FSR versions in OptiScaler's list" not in inst_log, inst_log[-2000:])
    t.check("no AMD DLL copied (OptiScaler 10 lists AMD's FSR itself): no wronggpu-amd-compare.dll, the package's FSR 4 not placed",
            not os.path.exists(os.path.join(b, "wronggpu-amd-compare.dll")) and not os.path.exists(os.path.join(b, "amd_fidelityfx_upscaler_dx12.dll")) and
            "AMD FSR 4 search:" not in inst_log and "for comparisons" not in inst_log and "no AMD FSR 4 search, no AMD provider copy" in inst_log, sorted(os.listdir(b)))
    t.check("wronggpu.ini and wronggpu_runtime.dll next to the exe (the front end's game-folder layout)",
            os.path.isfile(os.path.join(b, "wronggpu.ini")) and os.path.isfile(os.path.join(b, "wronggpu_runtime.dll")))
    t.check("the record holds the slot's front end", record_entry_sha(b, "OptiScaler\\custom\\WrongGPU\\ffx\\amd_fidelityfx_upscaler_dx12.dll") == sha(read(slot)),
            record_text(b)[-1500:])
    rc = t.run("status", exe)
    g = t.j.get("game", {})
    t.check("status: installed (no repair asked for the missing menu plugin)", rc == 0 and g.get("status") == "installed", g)
    write(os.path.join(os.path.dirname(slot), "wronggpu-fsr.log"), b"[00:00:01.000 1] [amd_fidelityfx_dx12] ==== session start: Slot.exe (process 1) ====\r\n")
    rc = t.run("status", exe)
    t.check("Last run reads the front end's log in the slot", rc == 0 and t.j.get("game", {}).get("last_run", {}).get("found") is True, t.j.get("game", {}).get("last_run"))
    # another build of the same version (the 0.6.0 beta, old route): the record's front end differs -> an update (rc5: was a
    # repair problem; the agreed wording "Installed V (another build) -> update V")
    recp = os.path.join(b, "wronggpu-setup", "install.record")
    rec0 = read(recp)
    lines = rec0.decode("utf-8").split("\r\n")
    for i, line in enumerate(lines):
        f = line.split("\t")
        if f[0] == "payload" and len(f) == 5:
            f[3] = "0" * 64
            lines[i] = "\t".join(f)
    write(recp, "\r\n".join(lines).encode("utf-8"))
    rc = t.run("status", exe)
    g = t.j.get("game", {})
    t.check("another build of this version (beta, its front end in the payload line): update offered, says so, no repair problem",
            rc == 0 and g.get("status") == "update-available" and g.get("other_build") is True and "(another build)" in g.get("status_text", "") and
            not g.get("problems"), g)
    write(recp, rec0)
    # Roll back across the route change: an earlier version kept under the old names can't go back into the slot layout
    prevd = os.path.join(b, "wronggpu-setup", "previous")
    old = b"MZ old front end (synthetic)"
    write(os.path.join(prevd, "wronggpu_ffx_dx12.dll"), old)
    write(os.path.join(prevd, "previous.txt"), ("version=0.5.2\r\nfile=" + sha(old) + "|wronggpu_ffx_dx12.dll\r\n").encode())
    rc = t.run("status", exe)
    t.check("roll back across the route change: not offered", rc == 0 and t.j.get("game", {}).get("can_roll_back") is False, t.j.get("game"))
    s0 = snapshot(root)
    rc = t.run("rollback", exe)
    t.check("roll back across the route change: refused, nothing changed", rc != 0 and "different OptiScaler setup" in t.j.get("message", "") and
            snapshot(root) == s0, (rc, t.j.get("message"), diff(s0, snapshot(root))))
    os.remove(os.path.join(prevd, "wronggpu_ffx_dx12.dll"))
    os.remove(os.path.join(prevd, "previous.txt"))
    os.rmdir(prevd)
    rc = t.run("uninstall", exe)
    after = snapshot(root)
    t.check("uninstall: the folder byte for byte as before (the slot's log removed with the slot)", rc == 0 and after == before, (rc, diff(before, after)))
    shutil.rmtree(root, ignore_errors=True)
    # an OptiScaler before 10 that has an OptiScaler\ folder too: the FidelityFX library route, not the custom slot
    old = os.path.join(t.work, "opti09-folder-package")
    shutil.rmtree(old, ignore_errors=True)
    write(os.path.join(old, "OptiScaler.dll"), mockpe.make_dll({"ProductName": "OptiScaler"}, (0, 9, 4, 0), dll_name="OptiScaler.dll"))
    write(os.path.join(old, "OptiScaler.ini"), OPTI_INI)
    write(os.path.join(old, "OptiScaler", "libxess.dll"), b"MZ mock OptiScaler 0.9 folder file (synthetic)")
    root, b, exe = game(t, "Slot Old Game", validated=True)
    before = snapshot(root)
    rc = t.run("install", exe, "--opti=" + old, "--noscan", "--force")
    opt = ini(b, "OptiScaler.ini") if rc == 0 else ""
    t.check("OptiScaler 0.9 with an OptiScaler\\ folder: the FidelityFX library route (wronggpu_ffx_dx12.dll, FfxDx12SRPath), no custom slot",
            rc == 0 and os.path.isfile(os.path.join(b, "wronggpu_ffx_dx12.dll")) and
            not os.path.exists(os.path.join(b, "OptiScaler", "custom", "WrongGPU", "ffx", "amd_fidelityfx_upscaler_dx12.dll")) and
            "Dx12Upscaler=WrongGPU" not in opt and "wronggpu_ffx_dx12.dll" in opt, (rc, t.j.get("message"), sorted(os.listdir(b)) if os.path.isdir(b) else None))
    rc = t.run("uninstall", exe)
    t.check("OptiScaler 0.9 with a folder: uninstall byte for byte", rc == 0 and snapshot(root) == before, (rc, diff(before, snapshot(root))))
    shutil.rmtree(root, ignore_errors=True)


def test_players_optiscaler(t, opti):
    print("\n[player's OptiScaler] ask to switch; their files backed up as moved and put back byte for byte")
    root, b, exe = game(t, "Switch Game", extra={"bin/x64/winmm.dll": b"MZ OptiScaler (the player's, synthetic) optiscaler",
                                                   "bin/x64/OptiScaler.ini": "[Menu]\nShowFps=true\n; the player's\n"})
    before = snapshot(root)
    rc = t.run("install", exe, "--opti=" + opti, "--noscan", "--force", "--experimental")
    t.check("without the player's OK: exit 6 (ask to switch), nothing changed", rc == 6 and "Switch" in t.j.get("message", "") and snapshot(root) == before, (rc, t.j.get("message")))
    rc = t.run("install", exe, "--opti=" + opti, "--noscan", "--force", "--experimental", "--switch")
    rec = record_text(b) if rc == 0 else ""
    t.check("with --switch: installed; the player's proxy recorded as moved", rc == 0 and "\nowner\tdisable\twinmm.dll\tmoved" in rec, (rc, t.j.get("message"), rec[-800:]))
    t.check("the player's files are in the backup", os.path.isfile(os.path.join(b, "wronggpu-setup", "backup", "winmm.dll")) and
            read(os.path.join(b, "wronggpu-setup", "backup", "OptiScaler.ini")) == b"[Menu]\nShowFps=true\n; the player's\n")
    rc = t.run("uninstall", exe)
    after = snapshot(root)
    t.check("uninstall: the player's OptiScaler is back byte for byte", rc == 0 and after == before, (rc, diff(before, after)))
    shutil.rmtree(root, ignore_errors=True)


def test_experimental_and_shared_model(t, opti):
    print("\n[two games] experimental confirmation; the model stays while another managed game uses it")
    r1, b1, e1 = game(t, "Exp One", exe="One.exe")
    r2, b2, e2 = game(t, "Exp Two", exe="Two.exe")
    s1, s2 = snapshot(r1), snapshot(r2)
    rc = t.run("install", e1, "--opti=" + opti, "--noscan", "--force")
    t.check("an untested game without --experimental: exit 3", rc == 3 and snapshot(r1) == s1, (rc, t.j.get("message")))
    rc1 = t.run("install", e1, "--opti=" + opti, "--noscan", "--force", "--experimental")
    rc2 = t.run("install", e2, "--opti=" + opti, "--noscan", "--force", "--experimental")
    model = os.path.join(t.env["LOCALAPPDATA"], "WrongGPU", "model", "model.wgcm")
    t.check("two installs", rc1 == 0 and rc2 == 0 and os.path.isfile(model), (rc1, rc2))
    rc = t.run("uninstall", e1)
    t.check("first uninstalled: the model stays (the other game is on the managed list)", rc == 0 and os.path.isfile(model) and snapshot(r1) == s1, diff(s1, snapshot(r1)))
    # a stale Apps & features entry (its game folder deleted by hand, so no record) is no game that still uses the data
    stale = t.uninstall + "\\WrongGPU-00000000"
    subprocess.run(["reg", "add", stale, "/v", "InstallLocation", "/t", "REG_SZ", "/d", os.path.join(t.work, "games", "Deleted Game", "bin", "x64"), "/f"],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    rc = t.run("uninstall", e2)
    subprocess.run(["reg", "delete", stale, "/f"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    t.check("last uninstalled (a stale entry of a deleted game left): byte for byte; the shared model data goes", rc == 0 and snapshot(r2) == s2 and not os.path.isfile(model),
            (rc, diff(s2, snapshot(r2))))
    shutil.rmtree(r1, ignore_errors=True)
    shutil.rmtree(r2, ignore_errors=True)


def test_orphan(t, opti):
    print("\n[orphan] the game's exe is gone: status orphaned, uninstall by its folder")
    root, b, exe = game(t, "Gone Game", exe="Gone.exe")
    os.remove(exe)
    before = snapshot(root)
    write(exe, "mock game exe " * 200 + "Gone Game")
    rc = t.run("install", exe, "--opti=" + opti, "--noscan", "--force", "--experimental")
    os.remove(exe)
    rc2 = t.run("status", b)
    g = t.j.get("game", {})
    t.check("orphaned: status from the install folder", rc == 0 and rc2 == 0 and g.get("status") == "orphaned" and g.get("installed_version"), g)
    rc = t.run("uninstall", b)
    after = snapshot(root)
    t.check("orphaned: uninstall by the folder, byte for byte (without the exe)", rc == 0 and after == before, (rc, t.j.get("message"), diff(before, after)))
    shutil.rmtree(root, ignore_errors=True)


def test_rc6_hunt(t, opti):
    """rc6 launcher hunt: an install next to a launcher stub moved to the game by Repair (L2); a failed update keeps the settings (L4); OptiScaler.ini saved by its menu (L5)."""
    print("\n[rc6 hunt] an install in the wrong place")
    # L2: an install made next to REDprelauncher.exe before rc5 (made here next to a starter, then its record names the stub)
    root, b, exe = game(t, "Stub Game", validated=True)
    stub = os.path.join(root, "REDprelauncher.exe")
    starter = os.path.join(root, "Starter.exe")
    write(stub, "a launcher stub " * 50)
    write(starter, "mock starter exe " * 200)
    rc = t.run("install", starter, "--opti=" + opti, "--noscan", "--force", "--experimental")
    t.check("L2 setup: an install in the game's root", rc == 0 and os.path.isfile(os.path.join(root, "wronggpu-setup", "install.record")), (rc, t.j.get("message"), t.logtail()))
    if rc != 0:
        return
    rp = os.path.join(root, "wronggpu-setup", "install.record")
    write(rp, read(rp).decode("utf-8").replace("exe\t" + starter + "\r\n", "exe\t" + stub + "\r\n"))
    os.remove(starter)
    # (the CLI names it by its folder, where the record is: the stub's path itself is read as the game it starts)
    rc = t.run("set", root, "output_filter=quality9")
    rc2 = t.run("status", root, "--noscan")
    g = t.j.get("game", {})
    probs = g.get("problems", [])
    t.check("L2: an install next to the stub is shown in the wrong place (needs repair, the game's program named)",
            rc == 0 and rc2 == 0 and g.get("status") == "needs-repair" and probs and probs[0].startswith("WrongGPU is next to REDprelauncher.exe") and
            "bin\\x64\\Cyberpunk2077.exe" in probs[0], (rc, g))
    rc = t.run("repair", root, "--opti=" + opti, "--noscan", "--force")
    t.check("L2: Repair moves it to the game's program", rc == 0 and "moved to the game's program" in t.j.get("message", "") and
            os.path.isfile(os.path.join(b, "wronggpu-setup", "install.record")) and not os.path.isdir(os.path.join(root, "wronggpu-setup")) and
            not os.path.isfile(os.path.join(root, "dxgi.dll")) and not os.path.isfile(os.path.join(root, "wronggpu_runtime.dll")), (rc, t.j.get("message"), t.logtail()))
    t.check("L2: the record names the game's program; the managed list has the game, not the root",
            "\nexe\t" + exe + "\r\n" in record_text(b) and b.lower() in managed_list(t) and (root.lower() + "\r\n") not in managed_list(t), (record_text(b)[:400], managed_list(t)))
    rc = t.run("get", exe)
    t.check("L2: the settings went over to the game", rc == 0 and t.j.get("settings", {}).get("output_filter", {}).get("profile") == "quality9", t.j.get("settings"))
    rc = t.run("uninstall", b)
    rc2 = t.run("install", stub, "--opti=" + opti, "--noscan", "--force")
    t.check("L2: never installed next to a stub: Install of the stub installs at the game", rc == 0 and rc2 == 0 and
            os.path.isfile(os.path.join(b, "wronggpu-setup", "install.record")) and not os.path.isdir(os.path.join(root, "wronggpu-setup")), (rc, rc2, t.j.get("message")))
    t.run("uninstall", b)
    shutil.rmtree(root, ignore_errors=True)

    # L4: an update that fails after the earlier install was removed keeps the settings (profile.ini, previous\ for Roll back)
    print("\n[rc6 hunt] a failed update keeps the settings")
    root, b, exe = game(t, "Failed Update Game", validated=True)
    rc = t.run("install", exe, "--opti=" + opti, "--noscan", "--force")
    rc2 = t.run("set", exe, "output_filter=quality9")
    rp = os.path.join(b, "wronggpu-setup", "install.record")
    write(rp, b"\r\n".join(b"version\t0.5.9" if l.startswith(b"version\t") else l for l in read(rp).split(b"\r\n")))  # an earlier version (Roll back keeps it)
    t.env["WRONGGPU_TEST_FAIL_AFTER_RESTORE"] = "1"
    try:
        rc3 = t.run("update", exe, "--opti=" + opti, "--noscan", "--force")
    finally:
        t.env.pop("WRONGGPU_TEST_FAIL_AFTER_RESTORE", None)
    state = os.path.join(b, "wronggpu-setup")
    t.check("L4: the update failed after the earlier install was removed: no record, the game's files back",
            rc == 0 and rc2 == 0 and rc3 != 0 and not os.path.isfile(rp) and not os.path.isfile(os.path.join(b, "wronggpu_runtime.dll")), (rc, rc2, rc3, t.j.get("message")))
    t.check("L4: profile.ini and previous\\ (Roll back) are kept; the uninstaller without a record is gone",
            os.path.isfile(os.path.join(state, "profile.ini")) and os.path.isfile(os.path.join(state, "previous", "previous.txt")) and
            not os.path.isfile(os.path.join(state, "uninstall.exe")), sorted(os.listdir(state)) if os.path.isdir(state) else None)
    rc = t.run("install", exe, "--opti=" + opti, "--noscan", "--force")
    rc2 = t.run("get", exe)
    t.check("L4: the next install starts with the kept settings", rc == 0 and rc2 == 0 and
            t.j.get("settings", {}).get("output_filter", {}).get("profile") == "quality9", (rc, t.j.get("settings")))
    t.run("uninstall", b)
    shutil.rmtree(root, ignore_errors=True)

    # L5: OptiScaler.ini saved by OptiScaler's own menu is the install's: carried over by Update, removed by Uninstall
    print("\n[rc6 hunt] OptiScaler.ini saved by OptiScaler's menu")
    root, b, exe = game(t, "Menu Save Game", validated=True)
    before = snapshot(root)
    oini = os.path.join(b, "OptiScaler.ini")
    rc = t.run("install", exe, "--opti=" + opti, "--noscan", "--force")
    rc2 = t.run("set", exe, "opti.Menu.ShortcutKey=0x24")
    menu = lambda v: write(oini, re.sub(r"ShortcutKey=[^\r\n]*", "ShortcutKey=" + v, ini(b, "OptiScaler.ini")) + "; saved by the menu\r\n")
    menu("0x2E")  # the menu saves its ini (its own value over the launcher's)
    rc3 = t.run("update", exe, "--opti=" + opti, "--noscan", "--force")
    rec = record_text(b)
    t.check("L5: Update carries the menu's value over and records OptiScaler.ini as the install's (no backup of it)",
            rc == 0 and rc2 == 0 and rc3 == 0 and "ShortcutKey=0x2E" in ini(b, "OptiScaler.ini") and "\ncopy\t0\t-\t" in rec and
            re.search(r"\ncopy\t0\t-\t[0-9a-f]+\tOptiScaler\.ini\r\n", rec) and not os.path.isfile(os.path.join(b, "wronggpu-setup", "backup", "OptiScaler.ini")),
            (rc, rc2, rc3, t.j.get("message"), ini(b, "OptiScaler.ini")[-300:]))
    menu("0x2F")
    rc = t.run("uninstall", b)
    after = snapshot(root)
    t.check("L5: Uninstall removes the menu-saved OptiScaler.ini with the rest (the game byte for byte)", rc == 0 and after == before and not os.path.isfile(oini),
            (rc, t.j.get("warnings"), diff(before, after)))
    rc = t.run("status", exe, "--noscan")
    t.check("L5: after the uninstall the game is eligible (not \"your own OptiScaler\")", rc == 0 and t.j.get("game", {}).get("status") == "eligible", t.j.get("game"))
    shutil.rmtree(root, ignore_errors=True)


def test_adopt_and_self_install(t, opti, baseline):
    print("\n[adoption] a 0.5.1 install (record v1) adopted; the launcher installed for this user; one Apps & features entry")
    if not baseline or not os.path.isfile(baseline):
        print("  (no 0.5.1 installer given: skipped)")
        return
    root, b, exe = game(t, "Old Game", exe="Old.exe")
    before = snapshot(root)
    rc = t.raw([baseline, "/S", "/LOG=" + t.log, "/NOSCAN", "/FORCE", "/EXPERIMENTAL", "/DIR=" + root, "/OPTISCALER=" + opti])
    # the 0.5.1 test build wrote its entry under the shared test root of its time; released builds all use the one Apps &
    # features root, so the entry moves to this run's root (where this build looks), as if both had written the same one
    old_root = r"HKCU\Software\WrongGPU-CandidateTests\Uninstall"
    for k, v in t.reg_keys(old_root).items():
        if k.startswith("WrongGPU-") and v.get("InstallLocation", "").lower() == b.lower():
            subprocess.run(["reg", "copy", old_root + "\\" + k, t.uninstall + "\\" + k, "/s", "/f"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            subprocess.run(["reg", "delete", old_root + "\\" + k, "/f"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    t.check("0.5.1 installer: installed (record v1, its own entry)", rc == 0 and record_text(b).startswith("WRONGGPU-RECORD 1") and len(t.entries_for(b)) == 1, (rc, t.logtail()))
    if rc != 0:
        return
    # a player's setting plus developer / test keys (as on the owner's PC: timing_log=1, model_dir=..., log_dispatches = 1)
    dev = ini(b).replace("runtime_options=model=M", "runtime_options=model=M;output_filter=quality5;timing_log=1;watchdog=1;model_dir=C:\\models;executor=d3d12-test")
    dev = dev.replace("log_dispatches = 0", "log_dispatches = 1") if "log_dispatches = 0" in dev else dev + "\r\nlog_dispatches = 1\r\n"
    write(os.path.join(b, "wronggpu.ini"), dev)
    old_un = read(os.path.join(b, "wronggpu-setup", "uninstall.exe"))
    rc = t.run("status", exe)
    t.check("status: installed by 0.5.1, adoptable", rc == 0 and t.j.get("game", {}).get("record_v1") is True and t.j.get("game", {}).get("adoptable") is True, t.j.get("game"))
    rc = t.run("adopt", exe)
    rec = record_text(b)
    t.check("adopt: record v2, the v1 file kept", rc == 0 and rec.startswith("WRONGGPU-RECORD 2") and os.path.isfile(os.path.join(b, "wronggpu-setup", "install.record.v1")), (rc, t.j))
    prof = read(os.path.join(b, "wronggpu-setup", "profile.ini")).decode() if os.path.isfile(os.path.join(b, "wronggpu-setup", "profile.ini")) else ""
    t.check("adopt: the player's setting imported into profile.ini", "output_filter=quality5" in prof, prof)
    notes = t.j.get("notes", [])
    ignored = sorted(n.split(" (")[0][len("ignored: "):] for n in notes if n.startswith("ignored: "))
    t.check("adopt: developer/test keys never become profile settings (only the schema's keys in profile.ini; executor=d3d12-test not imported)",
            not any(k in prof for k in ("timing_log", "watchdog", "model_dir", "log_dispatches", "d3d12-test")) and "executor=auto" in prof, prof)
    t.check("adopt: the ignored keys listed in the result and in setup.log (timing_log, watchdog, model_dir, log_dispatches; model= is the installer's)",
            ignored == sorted(["timing_log=1", "watchdog=1", "model_dir=C:\\models", "log_dispatches=1"]) and any(n.startswith("not imported:") and "d3d12-test" in n for n in notes) and
            "ignored: timing_log=1" in read(t.log).decode("utf-8", "replace"), (ignored, notes))
    t.check("adopt: wronggpu.ini left as it was (the keys stay until the player removes them)", ini(b) == dev, ini(b)[-300:])
    t.check("adopt: uninstall.exe is now the restorer", read(os.path.join(b, "wronggpu-setup", "uninstall.exe")) != old_un)
    t.check("adopt (launcher not installed): the game's own entry stays", len(t.entries_for(b)) == 1)
    programs, menu = os.path.join(t.work, "programs", "WrongGPU"), os.path.join(t.work, "startmenu")
    shutil.rmtree(programs, ignore_errors=True)
    shutil.rmtree(menu, ignore_errors=True)
    rc = t.run("self-install", "--dir=" + programs, "--startmenu=" + menu)
    keys = t.reg_keys()
    t.check("self-install: the exe copied, a Start-menu shortcut, one Apps & features entry \"WrongGPU\"", rc == 0 and os.path.isfile(os.path.join(programs, "WrongGPU.exe")) and
            os.path.isfile(os.path.join(menu, "WrongGPU.lnk")) and keys.get("WrongGPU", {}).get("DisplayName") == "WrongGPU", (rc, t.j, list(keys)))
    t.check("self-install: the adopted game's own entry is gone (one entry for all)", not t.entries_for(b) and "\nregkey\t\r\n" in record_text(b), t.entries_for(b))
    rc = t.run("install", exe, "--opti=" + opti, "--noscan", "--force", "--experimental")
    t.check("an install with the launcher installed: no entry of its own", rc == 0 and not t.entries_for(b), t.entries_for(b))
    model_dir = os.path.join(t.env["LOCALAPPDATA"], "WrongGPU", "model")
    had_model = os.path.isdir(model_dir)
    rc = t.raw([os.path.join(b, "wronggpu-setup", "uninstall.exe"), "/S", "/LOG=" + t.log])
    wait_gone(os.path.join(b, "wronggpu-setup"))
    after = snapshot(root)
    t.check("the restorer: the game as before the 0.5.1 install, byte for byte", rc == 0 and after == before, (rc, diff(before, after)))
    t.check("the last game uninstalled while the launcher is installed: the shared model data stays (the launcher shows it as its own)",
            had_model and os.path.isdir(model_dir), (had_model, t.logtail()))
    rc = t.run("self-uninstall", "--keep-games", "--startmenu=" + menu)
    t.check("self-uninstall: entry, shortcut and exe removed", rc == 0 and "WrongGPU" not in t.reg_keys() and not os.path.isfile(os.path.join(menu, "WrongGPU.lnk")) and
            not os.path.isfile(os.path.join(programs, "WrongGPU.exe")), (rc, t.j))
    t.check("...and with no game left, the launcher's uninstall takes the shared model data with it", wait_removed(t) and not os.path.isdir(model_dir),
            (removed_folders(t), os.listdir(model_dir) if os.path.isdir(model_dir) else None))
    shutil.rmtree(root, ignore_errors=True)


def test_launcher_uninstall(t, opti):
    print("\n[launcher] installed at its default place from a portable copy; uninstalled by its own copy: games kept, then games too; nothing left")
    lad = t.env["LOCALAPPDATA"]
    programs, shared = os.path.join(lad, "Programs", "WrongGPU"), os.path.join(lad, "WrongGPU")
    installed = os.path.join(programs, "WrongGPU.exe")
    menu = os.path.join(t.work, "startmenu-default")
    port = os.path.join(t.work, "portable-copy")
    for d in (programs, menu, port):
        shutil.rmtree(d, ignore_errors=True)
    try:  # %LOCALAPPDATA%\Programs as on a fresh profile (only when empty: never another program's)
        os.rmdir(os.path.dirname(programs))
    except OSError:
        pass
    programs_made = not os.path.exists(os.path.dirname(programs))  # the install makes it, so its uninstall removes it
    os.makedirs(port)
    pexe = os.path.join(port, "WrongGPU.exe")
    shutil.copy2(t.exe, pexe)
    write(os.path.join(port, "WrongGPU.portable"), "portable\r\n")
    try:  # the portable copy as a browser saves a download: with its Mark of the Web (an NTFS stream)
        with open(pexe + ":Zone.Identifier", "w") as f:
            f.write("[ZoneTransfer]\nZoneId=3\n")
        motw = True
    except OSError:
        motw = False
    root, b, exe = game(t, "Kept Game", exe="Kept.exe")
    before = snapshot(root)
    rc = t.run("install", exe, "--opti=" + opti, "--noscan", "--force", "--experimental")
    t.check("a game installed before the launcher: its own entry", rc == 0 and len(t.entries_for(b)) == 1, (rc, t.j.get("message"), t.entries_for(b)))
    rc = t.run("set", "--global", "play_button=0", exe=pexe, data=False)
    rc2 = t.run("self-install", "--startmenu=" + menu, exe=pexe, data=False)
    entry = t.reg_keys().get("WrongGPU", {})
    t.check("self-install (default place): the exe, the shortcut, one entry whose Uninstall opens the launcher's uninstall, an install date",
            rc == 0 and rc2 == 0 and os.path.isfile(installed) and os.path.isfile(os.path.join(menu, "WrongGPU.lnk")) and
            entry.get("UninstallString", "").lower() == ('"%s" uninstall-launcher' % installed).lower() and re.fullmatch(r"\d{8}", entry.get("InstallDate", "")) is not None,
            (rc, rc2, t.j, entry))
    t.check("self-install: the game's own entry folded into the launcher's", not t.entries_for(b) and "\nregkey\t\r\n" in record_text(b), t.entries_for(b))
    if motw:
        try:
            with open(installed + ":Zone.Identifier", "rb"):
                kept_motw = True
        except OSError:
            kept_motw = False
        t.check("self-install of a downloaded copy: the installed exe has no Mark of the Web (Zone.Identifier)", os.path.isfile(installed) and not kept_motw)
    else:
        print("  skipped: the Mark of the Web check (this folder's file system has no streams)")
    li = os.path.join(shared, "launcher.ini")
    t.check("self-install from a portable copy: its settings go with it (launcher.ini of the installed copy)", os.path.isfile(li) and "play_button=0" in read(li).decode(),
            read(li).decode() if os.path.isfile(li) else "(none)")
    # uninstalled by the installed copy itself (as its window and Apps & features do): its running exe is moved out at once.
    # Started as its Start-menu shortcut starts it: its install folder is its current directory
    rc = t.run("self-uninstall", "--keep-games", "--startmenu=" + menu, exe=installed, data=False, cwd=programs)
    t.check("self-uninstall by the installed copy (current directory: its install folder, as from the Start menu): its folder gone at once, entry and shortcut gone",
            rc == 0 and not os.path.exists(programs) and
            "WrongGPU" not in t.reg_keys() and not os.path.isfile(os.path.join(menu, "WrongGPU.lnk")), (rc, t.j, os.listdir(programs) if os.path.isdir(programs) else None))
    t.check("keep games: the game keeps WrongGPU with its own entry again; the launcher's settings gone; the shared data stays (the game uses it)",
            len(t.entries_for(b)) == 1 and "\nregkey\tWrongGPU-" in record_text(b) and not os.path.isfile(li) and os.path.isdir(os.path.join(shared, "model")),
            (t.entries_for(b), os.listdir(shared) if os.path.isdir(shared) else None))
    t.check("the moved exe is deleted once it no longer runs", wait_removed(t), removed_folders(t))
    # a game that keeps its own entry while the launcher is installed (as a 0.5 install not adopted, or an adoption declined
    # at the permission prompt): installed before the launcher, then left off the launcher's list
    root2, b2, exe2 = game(t, "Own Entry Game", exe="Own.exe")
    before2 = snapshot(root2)
    rc = t.run("install", exe2, "--opti=" + opti, "--noscan", "--force", "--experimental")
    listf = os.path.join(shared, "managed-games.txt")
    if os.path.isfile(listf):
        write(listf, "".join(l + "\r\n" for l in read(listf).decode("utf-8").splitlines() if l.strip().lower() != b2.lower()))
    rc2 = t.run("self-install", "--startmenu=" + menu, exe=pexe, data=False)
    t.check("installed again: the game's entry folded in again", rc2 == 0 and not t.entries_for(b), (rc2, t.entries_for(b)))
    t.check("...a game left off the launcher's list keeps its own entry", rc == 0 and len(t.entries_for(b2)) == 1, (rc, t.entries_for(b2)))
    # uninstalled from a portable copy that stays open (its window after Installed apps' Uninstall), which then repairs a game in
    # the same process: the game keeps its own entry (the process started while the launcher was installed)
    rc = t.run("self-uninstall", "--keep-games", "--startmenu=" + menu, "--repair-after=" + exe, "--opti=" + opti, "--noscan", "--force", "--experimental",
               exe=pexe, data=False)
    t.check("uninstalled by a portable copy that then repairs a game in the same process: the launcher gone, the game keeps its own entry",
            rc == 0 and t.j.get("repair_code") == 0 and "WrongGPU" not in t.reg_keys() and len(t.entries_for(b)) == 1 and "\nregkey\tWrongGPU-" in record_text(b),
            (rc, t.j, t.entries_for(b)))
    t.check("...the moved exe is deleted once it no longer runs", wait_removed(t), removed_folders(t))
    rc2 = t.run("self-install", "--startmenu=" + menu, exe=pexe, data=False)
    t.check("installed again from the portable copy: the game's entry folded in once more", rc2 == 0 and not t.entries_for(b), (rc2, t.entries_for(b)))
    rc = t.run("self-uninstall", "--startmenu=" + menu, exe=installed, data=False)
    after = snapshot(root)
    t.check("self-uninstall with the games: the game byte for byte as before, the launcher gone", rc == 0 and after == before and not os.path.exists(programs) and
            "WrongGPU" not in t.reg_keys() and not t.entries_for(b), (rc, t.j, diff(before, after)))
    t.check("...and the game with its own entry too: byte for byte, its entry gone", snapshot(root2) == before2 and not t.entries_for(b2),
            (diff(before2, snapshot(root2)), t.entries_for(b2)))
    t.check("nothing left in the user folder: %LOCALAPPDATA%\\WrongGPU (model data, list, logs) and the moved exe gone", wait_removed(t) and not os.path.exists(shared),
            (removed_folders(t), os.listdir(shared) if os.path.isdir(shared) else None))
    t.check("...and %LOCALAPPDATA%\\Programs, which the install made on this fresh profile, is gone too",
            not programs_made or not os.path.exists(os.path.dirname(programs)),
            os.listdir(os.path.dirname(programs)) if os.path.isdir(os.path.dirname(programs)) else None)
    # the Start-menu shortcut: a replace (no folder named) keeps it where the entry has it; test builds never use the
    # player's Start menu (their default is a folder in LOCALAPPDATA)
    tmenu = os.path.join(lad, "WrongGPU-TestStartMenu")
    shutil.rmtree(tmenu, ignore_errors=True)
    rc = t.run("self-install", "--startmenu=" + menu, exe=pexe, data=False)
    rc2 = t.run("self-install", exe=pexe, data=False)
    entry = t.reg_keys().get("WrongGPU", {})
    t.check("a replace without --startmenu keeps the custom Start-menu folder (no second shortcut, the entry still names it)",
            rc == 0 and rc2 == 0 and os.path.isfile(os.path.join(menu, "WrongGPU.lnk")) and not os.path.exists(os.path.join(tmenu, "WrongGPU.lnk")) and
            entry.get("StartMenuShortcut", "").lower() == os.path.join(menu, "WrongGPU.lnk").lower(), (rc, rc2, entry))
    # the launcher installed, its only game uninstalled: the shared model data stays (Settings shows it as the launcher's);
    # the launcher's own uninstall below takes it ("nothing left in the user folder")
    model_dir = os.path.join(shared, "model")
    rc = t.run("install", exe, "--opti=" + opti, "--noscan", "--force", "--experimental")
    had = rc == 0 and os.path.isdir(model_dir)
    rc = t.run("uninstall", exe)
    t.check("the last game uninstalled while the launcher is installed: the game byte for byte, the shared model data stays",
            had and rc == 0 and snapshot(root) == before and os.path.isdir(model_dir), (had, rc, diff(before, snapshot(root))))
    rc = t.run("self-uninstall", exe=installed, data=False)
    rc2 = t.run("self-install", exe=pexe, data=False)
    t.check("test build, no folder named: the shortcut goes to the test folder in LOCALAPPDATA, never the player's Start menu; the custom one went with the uninstall",
            rc == 0 and rc2 == 0 and os.path.isfile(os.path.join(tmenu, "WrongGPU.lnk")) and not os.path.isfile(os.path.join(menu, "WrongGPU.lnk")),
            (rc, rc2, t.reg_keys().get("WrongGPU", {})))
    rc = t.run("self-uninstall", exe=installed, data=False)
    t.check("and uninstalled again: that shortcut gone, nothing left in the user folder", rc == 0 and not os.path.isfile(os.path.join(tmenu, "WrongGPU.lnk")) and
            wait_removed(t) and not os.path.exists(shared) and not os.path.exists(programs), (rc, removed_folders(t)))
    # a managed game on a drive that is not connected (a letter no drive has): left as it is, on the list, with the shared data
    free = next((c for c in "QPOZYXWVUTSRNMLKJIH" if not os.path.exists(c + ":" + os.sep)), None)
    if free:
        offline = os.path.join(free + ":" + os.sep, "Games", "Offline Game", "bin", "x64")
        rc = t.run("self-install", exe=pexe, data=False)
        write(os.path.join(shared, "managed-games.txt"), "; test\r\n" + offline + "\r\n")
        rc2 = t.run("self-uninstall", exe=installed, data=False)
        t.check("a managed game on a drive that is not connected: the launcher uninstalled, the game named, kept on the list, the shared data kept",
                rc == 0 and rc2 == 0 and [x.lower() for x in t.j.get("unreachable", [])] == [offline.lower()] and offline.lower() in managed_list(t) and
                not os.path.exists(programs) and not os.path.isfile(li) and "WrongGPU" not in t.reg_keys(), (rc, rc2, t.j, managed_list(t)))
        own = t.entries_for(offline)
        t.check("...it gets its own Installed apps entry (its uninstaller in its wronggpu-setup folder, named by its folder), so it can be removed with no launcher",
                len(own) == 1 and list(own.values())[0].get("UninstallString", "").lower() == ('"%s"' % os.path.join(offline, "wronggpu-setup", "uninstall.exe")).lower() and
                list(own.values())[0].get("DisplayName") == "WrongGPU - Offline Game", own)
        # another game installed and uninstalled meanwhile: the game on the drive that isn't connected still uses the shared data
        rc = t.run("install", exe, "--opti=" + opti, "--noscan", "--force", "--experimental")
        had = rc == 0 and os.path.isdir(model_dir)
        rc2 = t.run("uninstall", exe)
        t.check("...while it is on the list, the last reachable game's own uninstall keeps the shared model data (the game byte for byte)",
                had and rc2 == 0 and snapshot(root) == before and os.path.isdir(model_dir), (had, rc2, diff(before, snapshot(root)), t.logtail()))
        for k in own:
            subprocess.run(["reg", "delete", t.uninstall + "\\" + k, "/f"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        wait_removed(t)
        shutil.rmtree(shared, ignore_errors=True)
    else:
        print("  skipped: the drive that is not connected (every drive letter is in use)")
    # a portable launcher that used the shared data (it notes itself there): the last game's own uninstall keeps that data while the
    # copy is there, and removes it once the copy is gone
    rc = t.run("install", exe, "--opti=" + opti, "--noscan", "--force", "--experimental")
    write(os.path.join(shared, "portable-launchers.txt"), "; test\r\n" + pexe + "\r\n")
    rc2 = t.run("uninstall", exe)
    t.check("a portable launcher noted in the shared folder: the last game's uninstall keeps the model data (the game byte for byte)",
            rc == 0 and rc2 == 0 and snapshot(root) == before and os.path.isdir(model_dir), (rc, rc2, diff(before, snapshot(root))))
    # ...and the launcher's own uninstall with its games keeps it for that copy too (its question says the data stays)
    rc = t.run("install", exe, "--opti=" + opti, "--noscan", "--force", "--experimental")
    rc2 = t.run("self-install", exe=pexe, data=False)
    rc3 = t.run("self-uninstall", exe=installed, data=False)
    t.check("...the launcher's own uninstall with its games keeps the shared model data for that copy (the game byte for byte, the launcher gone)",
            rc == 0 and rc2 == 0 and rc3 == 0 and snapshot(root) == before and os.path.isdir(model_dir) and "WrongGPU" not in t.reg_keys() and wait_removed(t),
            (rc, rc2, rc3, t.j, diff(before, snapshot(root)), removed_folders(t)))
    # that copy has no uninstall: its Settings > Remove data... (remove-data) is the way out for the shared data. Refused while a
    # game has WrongGPU (nothing removed); then the whole shared folder goes, the copy's own folder stays
    rc = t.run("install", exe, "--opti=" + opti, "--noscan", "--force", "--experimental")
    rc2 = t.run("remove-data", exe=pexe, data=False)
    refused = rc == 0 and rc2 == 2 and os.path.isdir(model_dir) and "game" in t.j.get("error", "")
    rc3 = t.run("uninstall", exe)
    t.check("a portable copy's remove-data: refused while a game has WrongGPU, the shared model data kept", refused and rc3 == 0 and os.path.isdir(model_dir),
            (rc, rc2, rc3, t.j))
    rc = t.run("remove-data", exe=pexe, data=False)
    t.check("...no game left: the shared folder removed (nothing moved aside left), the portable copy kept",
            rc == 0 and t.j.get("removed") is True and not os.path.exists(shared) and wait_removed(t) and os.path.isfile(pexe),
            (rc, t.j, os.listdir(shared) if os.path.isdir(shared) else None, removed_folders(t)))
    shutil.rmtree(port, ignore_errors=True)
    rc = t.run("install", exe, "--opti=" + opti, "--noscan", "--force", "--experimental")
    rc2 = t.run("uninstall", exe)
    t.check("...that copy deleted: the last game's uninstall removes the shared data", rc == 0 and rc2 == 0 and snapshot(root) == before and wait_gone(shared),
            (rc, rc2, os.listdir(shared) if os.path.isdir(shared) else None))
    shutil.rmtree(tmenu, ignore_errors=True)
    shutil.rmtree(root, ignore_errors=True)
    shutil.rmtree(root2, ignore_errors=True)
    shutil.rmtree(port, ignore_errors=True)


def synth_f_dll(folder):
    """The synthetic preset-F source of the test build's F map (spec data rr-f-synth, run 0, variant 0): a stand-in for
    a game's RR model file, made of synthetic bytes only."""
    import struct
    n = 6386372 // 2
    tags = [struct.pack("<e", float(v)) for v in range(16)]
    return b"".join(tags[e & 15] for e in range(n)) + read(os.path.join(folder, "synth-common-v0.bin"))


RR_SETTINGS = os.path.join("WrongGPU", "wronggpu.ini")


def rr_ini(b):
    """The RR plugin's settings file ([wronggpu] key=value) as a dict, lower-cased keys."""
    out, sec = {}, ""
    for line in ini(b, RR_SETTINGS).replace("\r", "").split("\n"):
        s = line.strip()
        if s.startswith("["):
            sec = s.lower()
        elif sec == "[wronggpu]" and "=" in s and not s.startswith(("#", ";")):
            k, v = s.split("=", 1)
            out[k.strip().lower()] = v.strip()
    return out


def test_rr(t, opti, synth_f):
    print("\n[rr] Ray Reconstruction: which games, the RR files with the install, the RR data, settings, the RR log, uninstall")
    lad = t.env["LOCALAPPDATA"]
    fdata = os.path.join(lad, "WrongGPU", "model", "F", "model.wgcm")
    # a game without Ray Reconstruction: nothing of it anywhere
    rootN, bN, exeN = game(t, "Rr None")
    beforeN = snapshot(rootN)
    rc = t.run("status", exeN, "--noscan")
    t.check("a game without RR: rr_uses false, no other RR fields", rc == 0 and t.j.get("game", {}).get("rr_uses") is False and "rr_installed" not in t.j.get("game", {}), t.j.get("game"))
    rc = t.run("install", exeN, "--opti=" + opti, "--noscan", "--force", "--experimental")
    rec = record_text(bN) if rc == 0 else ""
    t.check("a game without RR: installed without any RR file or record line", rc == 0 and "WrongGPU-RR.asi" not in rec and "wronggpu-rr.ini" not in rec and
            not os.path.exists(os.path.join(bN, "wronggpu_rr_runtime.dll")) and not os.path.exists(os.path.join(bN, "WrongGPU")) and
            not os.path.exists(os.path.join(bN, "wronggpu-rr.ini")) and
            not os.path.exists(os.path.join(bN, "plugins", "WrongGPU-RR.asi")), (rc, t.j.get("message"), rec[:800]))
    # a game that ships the RR model file (a stand-in WrongGPU cannot convert: the RR data is not ready yet)
    # with the game interface's RR plugin beside the RR model file (Cyberpunk's layout): a game that takes RR through the game
    # interface (the RR model file alone makes a game that creates RR itself: [direct-rr])
    root, b, exe = game(t, "Rr Game", extra={"bin/x64/nvngx_dlssd.dll": b"MZ synthetic stand-in, not a usable RR model file " * 64,
                                            "bin/x64/sl.dlss_d.dll": b"MZ synthetic stand-in"})  # the DLSS RR plugin stand-in
    before = snapshot(root)
    dll_before = read(os.path.join(b, "nvngx_dlssd.dll"))
    rc = t.run("status", exe, "--noscan")
    g = t.j.get("game", {})
    t.check("status: a game with the RR files uses RR (why: the game interface's RR plugin), through the game interface (not direct), not installed yet",
            rc == 0 and g.get("rr_uses") is True and "sl.dlss_d.dll" in g.get("rr_why", "") and g.get("rr_direct") is False and  # DLSS RR plugin
            g.get("rr_installed") is False, g)
    rc = t.run("rr-data", "status")
    rd = t.j.get("rr_data", {})
    t.check("rr-data status: not converted yet, this build has the RR files", rc == 0 and rd.get("valid") is False and rd.get("payload") is True and
            not os.path.exists(fdata), rd)
    rc = t.run("install", exe, "--opti=" + opti, "--noscan", "--force", "--experimental")
    rec = record_text(b) if rc == 0 else ""
    files = {"plugin": os.path.join(b, "plugins", "WrongGPU-RR.asi"), "runtime": os.path.join(b, "wronggpu_runtime.dll"),
             "ini": os.path.join(b, "wronggpu-rr.ini"), "settings": os.path.join(b, RR_SETTINGS)}
    t.check("install: the RR plugin beside WrongGPU's plugin, the runtime (RR built in), no RR runtime of its own, wronggpu-rr.ini and WrongGPU\\wronggpu.ini",
            not os.path.exists(os.path.join(b, "wronggpu_rr_runtime.dll")) and
            rc == 0 and all(os.path.isfile(p) for p in files.values()) and os.path.isfile(os.path.join(b, "plugins", "WrongGPU.asi")), (rc, t.j.get("message"), t.logtail()))
    if rc != 0:
        return
    t.check("install: every RR file is an entry of the record (copy, its hash)", all(record_entry_sha(b, rel) for rel in
            ("plugins\\WrongGPU-RR.asi", "wronggpu_runtime.dll", "wronggpu-rr.ini", RR_SETTINGS)) and "wronggpu_rr_runtime.dll" not in rec, rec[:2500])
    t.check("install: wronggpu-rr.ini names the shared RR data file", ("rr_model=" + fdata).lower() in ini(b, "wronggpu-rr.ini").lower(), ini(b, "wronggpu-rr.ini"))
    s = rr_ini(b)
    t.check("install without RR data: rr_enabled=0, Fast (the default since rc5), Exact not allowed", s.get("rr_enabled") == "0" and s.get("rr_mode") == "fast" and
            s.get("rr_exact_allowed") == "0", s)
    t.check("install: the game's RR model file untouched, nothing of RR in the upscaler's wronggpu.ini",
            read(os.path.join(b, "nvngx_dlssd.dll")) == dll_before and "rr_" not in ini(b), ini(b))
    t.check("install: the log says the RR data is missing and RR stays off", "Ray Reconstruction data isn't converted yet" in read(t.log).decode("utf-8", "replace"), t.logtail())
    t.check("install: the result's warnings say it too (the window's note after the success: nothing fails silently)",
            any("Ray Reconstruction data isn't converted yet" in w and "stays off in this game" in w for w in t.j.get("warnings", [])), t.j.get("warnings"))
    rc = t.run("status", exe)
    g = t.j.get("game", {})
    t.check("status: RR installed, off until the data is converted", rc == 0 and g.get("rr_installed") is True and g.get("rr_enabled") is False, g)
    # the RR data: a file that is not usable is refused; the synthetic RR model file converts
    rc = t.run("rr-data", "convert", "--dll=" + os.path.join(b, "nvngx_dlssd.dll"), "--noscan")
    t.check("rr-data convert of an unusable file: exit 5, no RR data written", rc == 5 and not os.path.exists(fdata), (rc, t.j))
    if synth_f and os.path.isfile(os.path.join(synth_f, "synth-common-v0.bin")):
        src = os.path.join(t.work, "rr-source", "nvngx_dlssd.dll")
        write(src, synth_f_dll(synth_f))
        src_bytes = read(src)
        rc = t.run("rr-data", "convert", "--dll=" + src, "--noscan")
        rd = t.j.get("rr_data", {})
        t.check("rr-data convert: the RR data written and valid, its source named, the source file untouched",
                rc == 0 and rd.get("valid") is True and os.path.isfile(fdata) and rd.get("source", "").lower() == src.lower() and read(src) == src_bytes, (rc, t.j, t.logtail()))
        s = rr_ini(b)
        t.check("after the conversion: the installed RR game's settings file turns RR on, the record follows",
                s.get("rr_enabled") == "1" and record_entry_sha(b, RR_SETTINGS) == sha(read(files["settings"])), (s, record_entry_sha(b, RR_SETTINGS)))
        rc = t.run("status", exe)
        t.check("status: RR on", rc == 0 and t.j.get("game", {}).get("rr_enabled") is True, t.j.get("game"))
        rc = t.run("rr-data", "status")
        t.check("rr-data status: valid", rc == 0 and t.j.get("rr_data", {}).get("valid") is True, t.j)
        # the data gone again: a repair turns RR off in the game and says so; then another RR game's install converts the data
        # from its own file, and the game installed before it gets RR too (no Repair or Apply on it needed)
        shutil.rmtree(os.path.join(lad, "WrongGPU", "model", "F"), ignore_errors=True)
        rc = t.run("repair", exe, "--opti=" + opti, "--noscan", "--force", "--experimental")
        t.check("RR data gone: a repair writes rr_enabled=0 and its result says RR stays off",
                rc == 0 and rr_ini(b).get("rr_enabled") == "0" and any("Ray Reconstruction data isn't converted yet" in w for w in t.j.get("warnings", [])),
                (rc, rr_ini(b), t.j.get("warnings")))
        root2, b2, exe2 = game(t, "Rr Second", exe="Second.exe", extra={"bin/x64/nvngx_dlssd.dll": src_bytes, "bin/x64/sl.dlss_d.dll": b"MZ synthetic stand-in"})  # the DLSS RR plugin stand-in
        before2 = snapshot(root2)
        rc = t.run("install", exe2, "--opti=" + opti, "--noscan", "--force", "--experimental")
        warn2 = t.j.get("warnings", [])
        s1 = rr_ini(b)
        t.check("a second RR game's install converts the data from its own file; the RR game installed before it gets RR too (its file and record follow)",
                rc == 0 and os.path.isfile(fdata) and rr_ini(b2).get("rr_enabled") == "1" and s1.get("rr_enabled") == "1" and
                record_entry_sha(b, RR_SETTINGS) == sha(read(files["settings"])) and not any("Ray Reconstruction" in w for w in warn2),
                (rc, s1, rr_ini(b2), warn2, t.logtail()))
        rc = t.run("status", exe)
        t.check("status of the game installed before: RR on", rc == 0 and t.j.get("game", {}).get("rr_enabled") is True, t.j.get("game"))
        rc = t.run("rr-data", "convert", "--dll=" + os.path.join(b, "nvngx_dlssd.dll"), "--noscan")
        rc2 = t.run("rr-data", "status")
        rd = t.j.get("rr_data", {})
        t.check("rr-data convert of a chosen file that doesn't convert: exit 5; no other game's file is converted instead (the data in use and its source stay)",
                rc == 5 and rc2 == 0 and rd.get("valid") is True and rd.get("source", "").lower() == os.path.join(b2, "nvngx_dlssd.dll").lower(), (rc, rd))
        rc = t.run("uninstall", exe2)
        t.check("uninstall of the second RR game: byte for byte; the RR data stays for the other RR game", rc == 0 and snapshot(root2) == before2 and os.path.isfile(fdata),
                (rc, diff(before2, snapshot(root2))))
        shutil.rmtree(root2, ignore_errors=True)
    else:
        print("  skipped: the RR data conversion (no synthetic preset-F folder: --synth-f)")
    data_on = os.path.isfile(fdata)
    # settings: the profile's rr_enabled and rr_mode into WrongGPU\wronggpu.ini (rr_exact_allowed only with exact)
    rc = t.run("set", exe, "rr_mode=fast")
    s = rr_ini(b)
    t.check("set rr_mode=fast: written, Exact not allowed, the record follows", rc == 0 and s.get("rr_mode") == "fast" and s.get("rr_exact_allowed") == "0" and
            record_entry_sha(b, RR_SETTINGS) == sha(read(files["settings"])), (rc, s, t.j.get("message")))
    rc = t.run("set", exe, "rr_mode=exact")
    s = rr_ini(b)
    t.check("set rr_mode=exact: rr_exact_allowed=1 with it", rc == 0 and s.get("rr_mode") == "exact" and s.get("rr_exact_allowed") == "1", (rc, s))
    rc = t.run("set", exe, "rr_enabled=off")
    s = rr_ini(b)
    t.check("set rr_enabled=off: rr_enabled=0", rc == 0 and s.get("rr_enabled") == "0" and s.get("rr_mode") == "exact", (rc, s))
    rc = t.run("set", exe, "rr_mode=turbo")
    t.check("set rr_mode=turbo: refused (exit 10), the file unchanged", rc == 10 and rr_ini(b) == s, (rc, rr_ini(b)))
    rc = t.run("set", exe, "rr_enabled=on", "rr_mode=accurate")
    s = rr_ini(b)
    t.check("set back: on (with the data), Accurate, Exact not allowed", rc == 0 and s.get("rr_enabled") == ("1" if data_on else "0") and s.get("rr_mode") == "accurate" and
            s.get("rr_exact_allowed") == "0" and "rr_" not in ini(b), (rc, s))
    rc = t.run("get", exe)
    st = t.j.get("settings", {})
    t.check("get: the RR settings, profile and file agree; no outside edits", rc == 0 and st.get("rr_mode", {}).get("profile") == "accurate" and
            st.get("rr_mode", {}).get("file") == "accurate" and st.get("rr_enabled", {}).get("profile") == "on" and not t.j.get("outside_edits"), (st, t.j.get("outside_edits")))
    # rc5 (owner 2026-11-05: Fast is the default): a player's rr_mode=accurate survives Repair and Update, never rewritten
    rc = t.run("repair", exe, "--opti=" + opti, "--noscan", "--force", "--experimental")
    rcg = t.run("get", exe)
    t.check("repair (rc5, Fast the default): an existing rr_mode=accurate stays, in the profile and the file",
            rc == 0 and rcg == 0 and t.j.get("settings", {}).get("rr_mode", {}).get("profile") == "accurate" and rr_ini(b).get("rr_mode") == "accurate",
            (rc, t.j.get("settings", {}).get("rr_mode"), rr_ini(b)))
    recp = os.path.join(b, "wronggpu-setup", "install.record")
    write(recp, b"\r\n".join(b"version\t0.5.9" if l.startswith(b"version\t") else l for l in read(recp).split(b"\r\n")))
    rc = t.run("update", exe, "--opti=" + opti, "--noscan", "--force", "--experimental")
    rcg = t.run("get", exe)
    t.check("update from an older version (rc5): an existing rr_mode=accurate stays, in the profile and the file",
            rc == 0 and rcg == 0 and t.j.get("settings", {}).get("rr_mode", {}).get("profile") == "accurate" and rr_ini(b).get("rr_mode") == "accurate",
            (rc, t.j.get("settings", {}).get("rr_mode"), rr_ini(b), t.logtail(800)))
    # the player switches it in the game (OptiScaler's menu): the plugin saves rr_mode and its own key; the next scan shows it,
    # and an update or repair keeps it
    write(files["settings"], read(files["settings"]).replace(b"rr_mode=accurate", b"rr_mode = fast\r\nrr_overlay=1"))
    rc = t.run("status", exe)
    t.check("switched in the game: status shows rr_mode fast", rc == 0 and t.j.get("game", {}).get("rr_mode") == "fast", t.j.get("game"))
    rc = t.run("repair", exe, "--opti=" + opti, "--noscan", "--force", "--experimental")
    rcg = t.run("get", exe)
    t.check("repair: the mode chosen in the game becomes the profile's and stays in the file",
            rc == 0 and rcg == 0 and t.j.get("settings", {}).get("rr_mode", {}).get("profile") == "fast" and rr_ini(b).get("rr_mode") == "fast", (rc, t.j.get("settings"), rr_ini(b)))
    # the last run's RR log
    write(os.path.join(b, "wronggpu-rr.log"),
          "[19:19:00] load: process 4242 attached\r\n[19:19:01] rr_mode=fast\r\n[19:19:02] runtime: create instance 2560x1440\r\n"
          "[19:20:00] timing, 600 frames: GPU 6.81 ms per frame\r\n")
    rc = t.run("status", exe)
    lr = t.j.get("game", {}).get("last_run", {}).get("rr", {})
    t.check("status: Last run's RR from wronggpu-rr.log (used, Fast, 6.8 ms a frame)", rc == 0 and lr.get("found") is True and lr.get("used") is True and
            lr.get("mode") == "fast" and lr.get("ms") == "6.8", lr)
    # the window: the RR section on the RR game only
    dump = os.path.join(t.work, "ui-dump-rr.json")
    rc2 = t.run("ui-dump", "--dump=" + dump, "--noscan", "--library=" + os.path.join(t.work, "games"))
    try:
        d = json.loads(read(dump).decode("utf-8"))
    except Exception as e:  # noqa: BLE001
        d = {"error": str(e)}
    pages = {gp.get("name"): {c["text"]: c for c in gp.get("controls", [])} for gp in d.get("games", [])}
    pr, pn = pages.get("Rr Game", {}), pages.get("Rr None", {})
    if data_on:
        t.check("window: the RR game's Ray Reconstruction section (switch on, Accurate | Fast, no Exact without Settings' switch)",
                rc2 == 0 and pr.get("Use WrongGPU (Ray Reconstruction)", {}).get("checked") is True and "Ray Reconstruction quality: Fast" in pr and
                "Ray Reconstruction quality: Accurate" in pr and "Ray Reconstruction quality: Exact" not in pr, (rc2, list(pr)))
        t.check("window: the default Quality is Fast (rc5): its segment is the chosen one, Accurate isn't",
                pr.get("Ray Reconstruction quality: Fast", {}).get("checked") is True and pr.get("Ray Reconstruction quality: Accurate", {}).get("checked") is not True,
                (pr.get("Ray Reconstruction quality: Fast"), pr.get("Ray Reconstruction quality: Accurate")))
    else:
        t.check("window: the RR game without RR data shows Convert", rc2 == 0 and "Convert the Ray Reconstruction data" in pr, (rc2, list(pr)))
    t.check("window: the game without RR shows nothing of it", pn and not any("Ray Reconstruction" in k or "Denoiser" in k for k in pn), list(pn))
    st = [c["text"] for c in d.get("settings", {}).get("controls", [])]
    t.check("window: Settings has the Ray Reconstruction data row and the Exact mode switch (off)",
            any(x.startswith("Ray Reconstruction data") for x in st) and any(c["text"] == "Exact mode" and c.get("checked") is False for c in d.get("settings", {}).get("controls", [])), st)
    # release 0.6.0 (owner: RR's kernels pre-compiled inside the launcher, rrprep.h): rr-prepare runs the launcher as a helper
    # under the game's program name, which loads the installed wronggpu_runtime.dll (the real runtime of this payload, RR
    # built in) and creates RR's pipelines on the GPU; the driver keeps them per program name, so the game finds them ready
    prep_dir = os.path.join(t.data, "rr-prepare")  # the launcher data folder (the suite runs with --data)
    rc = t.run("rr-prepare", exe, "--check")
    t.check("rr-prepare --check before: not prepared yet", rc == 0 and t.j.get("rr_prepare", {}).get("ready") is False and
            "not prepared yet" in t.j.get("rr_prepare", {}).get("not_ready_why", ""), t.j)
    # release 0.6.0 polish: the window's start prepares again only games prepared before (never prepared: Play's job)
    rc = t.run("rr-recheck", exe)
    rq = t.j.get("rr_recheck", {})
    t.check("rr-recheck before any prepare: not stale (state not-prepared), the start leaves it to Play",
            rc == 0 and rq.get("stale") is False and rq.get("state_before") == "not-prepared", t.j)
    glog = os.path.join(b, "wronggpu-rr.log")  # the RR log an earlier case wrote (Last run): prepare must not touch it
    glog_before = read(glog) if os.path.exists(glog) else None
    rc = t.run("rr-prepare", exe, "--passes=1")
    rp = t.j.get("rr_prepare", {})
    gpu = rp.get("result") != 2  # 2: no RDNA 4 graphics card on this PC (nothing to prepare)
    t.check("rr-prepare: the helper prepared the default mode's kernels (result 0, ready, every kernel reported), or no RDNA 4 card",
            rc == 0 and (not gpu or (rp.get("result") == 0 and rp.get("ready") is True and rp.get("kernels", 0) >= 50)), rp)
    t.check("rr-prepare: the helper copy is gone, nothing new in the game's folder (the runtime log went to the data folder)",
            not os.listdir(os.path.join(prep_dir, "run")) and (read(glog) if os.path.exists(glog) else None) == glog_before and
            (not gpu or os.path.isfile(os.path.join(prep_dir, "prepare.log"))), (os.listdir(prep_dir), os.listdir(b)))
    if gpu:
        ptxt = read(os.path.join(prep_dir, "prepared.txt")).decode("utf-8", "replace")
        t.check("rr-prepare: prepared.txt keeps the game's program name, the driver's version, the runtime's SHA-256, the passes",
                ptxt.lower().startswith(os.path.basename(exe).lower() + "\t" + rp.get("driver", "?").lower() + "\t") and ptxt.rstrip().endswith("\t1"), ptxt)
        plog = read(os.path.join(prep_dir, "prepare.log")).decode("utf-8", "replace")
        t.check("rr-prepare: the runtime ran as the game's program name (the driver's cache key)",
                ("as " + os.path.join(prep_dir, "run", os.path.basename(exe))).lower() in plog.lower() and "prepare: done" in plog, plog[-1500:])
        rc = t.run("rr-prepare", exe, "--passes=1")
        rp2 = t.j.get("rr_prepare", {})
        t.check("rr-prepare again: the driver had every kernel (quick: under 15 s, was %s ms)" % rp.get("ms"),
                rc == 0 and rp2.get("result") == 0 and rp2.get("ms", 99999) < 15000, rp2)
        rc = t.run("rr-prepare", exe, "--check", "--passes=2")
        t.check("rr-prepare --check for the other modes (pass 2): not prepared (default mode only)", rc == 0 and t.j.get("rr_prepare", {}).get("ready") is False, t.j)
        write(os.path.join(prep_dir, "prepared.txt"), ptxt.replace("\t" + rp.get("driver", "?") + "\t", "\t1.2.3.4\t", 1).encode("utf-8"))
        rc = t.run("rr-prepare", exe, "--check")
        t.check("rr-prepare --check after a driver change: not prepared, the reason in words", rc == 0 and t.j.get("rr_prepare", {}).get("ready") is False and
                "driver changed" in t.j.get("rr_prepare", {}).get("not_ready_why", ""), t.j)
        # release 0.6.0 polish (owner 2026-11-02): the launcher's start prepares again after a driver / WrongGPU update
        rc = t.run("rr-recheck", exe)
        rq = t.j.get("rr_recheck", {})
        t.check("rr-recheck after a driver change: stale (driver-changed), the reason names the old driver, nothing run without --run",
                rc == 0 and rq.get("stale") is True and rq.get("state_before") == "driver-changed" and "1.2.3.4" in rq.get("why", "") and rq.get("result") == -1, t.j)
        rc = t.run("rr-recheck", exe, "--run")
        rq = t.j.get("rr_recheck", {})
        p2 = read(os.path.join(prep_dir, "prepared.txt")).decode("utf-8", "replace")
        t.check("rr-recheck --run: prepared again (result 0), ready, prepared.txt names this driver again",
                rc == 0 and rq.get("result") == 0 and rq.get("state") == "ready" and
                p2.lower().startswith(os.path.basename(exe).lower() + "\t" + rp.get("driver", "?").lower() + "\t"), (t.j, p2))
        cols = p2.splitlines()[0].split("\t")
        if len(cols) >= 4:
            cols[2] = "0" * 64  # another WrongGPU runtime (an update): its SHA-256 differs
            write(os.path.join(prep_dir, "prepared.txt"), ("\t".join(cols) + "\r\n").encode("utf-8"))
        rc = t.run("rr-recheck", exe)
        rq = t.j.get("rr_recheck", {})
        t.check("rr-recheck after a WrongGPU runtime change: stale (runtime-changed)",
                rc == 0 and rq.get("stale") is True and rq.get("state_before") == "runtime-changed", t.j)
        write(os.path.join(prep_dir, "prepared.txt"), p2.encode("utf-8"))
        rc = t.run("rr-recheck", exe)
        rq = t.j.get("rr_recheck", {})
        t.check("rr-recheck with the same driver and runtime: not stale (ready), nothing run",
                rc == 0 and rq.get("stale") is False and rq.get("state_before") == "ready" and rq.get("result") == -1, t.j)
    # the plugin rewrote both settings files while the game ran; uninstall still leaves the folder as it was
    write(files["ini"], read(files["ini"]) + b"rr_plugin_key=1\r\n")
    write(files["settings"], read(files["settings"]) + b"rr_last_seen=1\r\n")
    rc = t.run("uninstall", exe)
    after = snapshot(root)
    t.check("uninstall: exit 0, the RR game's folder byte for byte as before (the rewritten RR settings files go too)", rc == 0 and after == before, (rc, t.j.get("message"), diff(before, after)))
    t.check("uninstall: the shared RR data stays for the other games", os.path.isfile(fdata) == data_on)
    rc = t.run("uninstall", exeN)
    t.check("uninstall of the game without RR: byte for byte as before", rc == 0 and snapshot(rootN) == beforeN, (rc, diff(beforeN, snapshot(rootN))))
    shutil.rmtree(root, ignore_errors=True)
    shutil.rmtree(rootN, ignore_errors=True)


RE9_IFACE = "wg_test_interface.dll"  # the game interface file the test build's RR folder names (make_mock_rr.py)


def mock_rr(t, rel):
    """A file of the test build's RR stand-ins (tests/make_mock_rr.py writes them next to the launcher it builds)."""
    return read(os.path.join(os.path.dirname(t.exe), "mock-rr", rel))


def add_refw(root):
    """REFramework's loader stand-in (its own words in the version resource) and its folder: synthetic bytes only."""
    write(os.path.join(root, "dinput8.dll"), mockpe.make_dll({"ProductName": "REFramework (test stand-in)"}, (1, 5, 9, 1),
                                                              extra=b"loader stand-in (synthetic, never loaded)", dll_name="dinput8.dll"))
    write(os.path.join(root, "reframework", "config.txt"), "synthetic settings")


def re9_game(t, name, refw=True, storage=True):
    """An RE Requiem layout: re9.exe in the folder's root, its game interface file (version 2.8.0), the RR plugin file that
    makes it an RR game, the synthetic model file; REFramework's loader and its mirror folder (_storage_) when asked."""
    root = os.path.join(t.work, "games", name)
    shutil.rmtree(root, ignore_errors=True)
    write(os.path.join(root, "re9.exe"), "mock game exe " * 200 + name)
    write(os.path.join(root, "nvngx_dlss.dll"), t.model.file)
    write(os.path.join(root, "sl.dlss_d.dll"), b"MZ synthetic stand-in")  # DLSS Ray Reconstruction's plugin (an RR game)
    write(os.path.join(root, RE9_IFACE), mockpe.make_dll({"ProductName": "game interface (test stand-in)"}, (2, 8, 0, 0),
                                                         extra=b"synthetic, never loaded", dll_name=RE9_IFACE))
    write(os.path.join(root, "steam_api64.dll"), "untouched game file")
    if refw:
        add_refw(root)
    if storage:  # REFramework's mirror of the root's DLLs
        write(os.path.join(root, "_storage_", RE9_IFACE), read(os.path.join(root, RE9_IFACE)))
    return root, os.path.join(root, "re9.exe")


def edit_record(root, rel, drop=False, new_sha=None):
    """The install record as an earlier build wrote it: `rel`'s lines taken out (drop), or its copy line's installed hash
    set to new_sha."""
    p = os.path.join(root, "wronggpu-setup", "install.record")
    out = []
    for x in read(p).decode("utf-8").splitlines(True):
        f = x.rstrip("\r\n").split("\t")
        if rel in f:
            if drop:
                continue
            if new_sha and len(f) == 5 and f[0] == "copy" and f[4] == rel:
                f[3] = new_sha
                x = "\t".join(f) + x[len(x.rstrip("\r\n")):]
        out.append(x)
    write(p, "".join(out).encode("utf-8"))


def edition_package(t, name="edition-package"):
    """A stand-in of WrongGPU's edition of OptiScaler (never embedded: the launcher downloads the real one): an OptiScaler
    package whose OptiScaler.dll the test build takes as the pinned edition DLL (WRONGGPU_TEST_EDITION_DLL / _EDITION)."""
    p = os.path.join(t.work, name)
    shutil.rmtree(p, ignore_errors=True)
    dll = mockpe.make_dll({"ProductName": "OptiScaler", "FileDescription": "OptiScaler WrongGPU edition, plugin interface 5 (test stand-in)"}, (10, 0, 0, 5),
                          extra=b"OptiScaler edition stand-in (synthetic, never loaded)", dll_name="OptiScaler.dll")
    write(os.path.join(p, "OptiScaler.dll"), dll)
    write(os.path.join(p, "OptiScaler.ini"), OPTI_INI)
    return p, dll


def test_re9(t, opti, synth_f):
    print("\n[re9] RE Requiem: REFramework first, the plugin build for its game interface, OptiScaler's edition, REFramework's mirror")
    plugin28 = mock_rr(t, os.path.join("plugins", "WrongGPU-RR.asi"))  # one build for interfaces 2.7 / 2.8 / 2.14 since 0.6.0
    t.check("the test build embeds no OptiScaler (no edition stand-in among its RR files)",
            not os.path.exists(os.path.join(os.path.dirname(t.exe), "mock-rr", "optiscaler-edition", "OptiScaler.dll")))
    ed_pkg, edition = edition_package(t)
    t.env["WRONGGPU_TEST_EDITION_DLL"] = sha(edition)
    plain = opti  # another OptiScaler (the player's own download): RE Requiem's RR is not offered with it
    opti = ed_pkg
    rootP, exeP = re9_game(t, "Re9 Plain")
    beforeP = snapshot(rootP)
    rc = t.run("install", exeP, "--opti=" + plain, "--noscan", "--force", "--experimental")
    logp = read(t.log).decode("utf-8", "replace")
    t.check("another OptiScaler (not the edition): installed, RR not offered with the plain warning (no OptiScaler edition with plugin interface 5)",
            rc == 0 and "is no OptiScaler edition with plugin interface 5" in logp and
            any("no OptiScaler edition" in w for w in t.j.get("warnings", [])), (rc, t.j.get("message"), t.j.get("warnings")))
    rc = t.run("uninstall", exeP)
    t.check("another OptiScaler: uninstall back byte for byte", rc == 0 and snapshot(rootP) == beforeP, (rc, diff(beforeP, snapshot(rootP))))
    shutil.rmtree(rootP, ignore_errors=True)
    # without REFramework: the status says what it needs, Install is refused and nothing changes
    rootB, exeB = re9_game(t, "Re9 Bare", refw=False)
    beforeB = snapshot(rootB)
    rc = t.run("status", exeB, "--noscan")
    g = t.j.get("game", {})
    t.check("no REFramework: status needs-prerequisite, needs REFramework (not found, its page), an RR game with path tracing only",
            rc == 0 and g.get("status") == "needs-prerequisite" and g.get("needs", {}).get("tool") == "REFramework" and g.get("needs", {}).get("found") is False and
            "REFramework" in g.get("needs", {}).get("page", "") and g.get("rr_uses") is True and g.get("rr_path_tracing_only") is True, g)
    rc = t.run("install", exeB, "--opti=" + opti, "--noscan", "--force", "--experimental")
    t.check("no REFramework: install refused (exit 2) with the plain reason, the folder unchanged",
            rc == 2 and "needs REFramework installed first" in t.j.get("message", "") and snapshot(rootB) == beforeB, (rc, t.j.get("message"), diff(beforeB, snapshot(rootB))))
    # REFramework there, with a copy in its mirror that is not WrongGPU's (backed up, put back)
    root, exe = re9_game(t, "Re9 Game")
    write(os.path.join(root, "_storage_", "dxgi.dll"), b"MZ another dxgi.dll's old copy (synthetic)")
    before = snapshot(root)
    refw_bytes = read(os.path.join(root, "dinput8.dll"))
    rc = t.run("status", exe, "--noscan")
    g = t.j.get("game", {})
    co = [c for c in g.get("coexistence", []) if c.get("tool") == "reframework"]
    t.check("REFramework found: eligible, needs found, REFramework an information finding that names the game's need",
            rc == 0 and g.get("status") == "eligible" and g.get("needs", {}).get("found") is True and co and co[0].get("decision") == "inform" and
            "needs it" in co[0].get("message", "") and g.get("proxy_plan", {}).get("name", "").lower() == "dxgi.dll", g)
    rc = t.run("install", exe, "--opti=" + opti, "--noscan", "--force", "--experimental")
    rec = record_text(root) if rc == 0 else ""
    t.check("install: exit 0; OptiScaler's DLL is WrongGPU's edition (plugin interface 5), the RR plugin the one build (serves interface 2.8)",
            rc == 0 and read(os.path.join(root, "dxgi.dll")) == edition and read(os.path.join(root, "plugins", "WrongGPU-RR.asi")) == plugin28,
            (rc, t.j.get("message"), t.logtail()))
    if rc != 0:
        return
    log = read(t.log).decode("utf-8", "replace")
    t.check("install: the log names the game interface's version and the plugin build it chose",
            "the one build for game interfaces 2.7, 2.8 and 2.14" in log and "2.8.0.0" in log, t.logtail(3000))
    # release 0.6.0 polish: RE Requiem keeps its driver shader cache under its own AGS application name, so the launcher's
    # helper can't prepare its kernels; it prepares them at its own first start and no helper runs (rrprep.h rrKernelsInGame)
    run_dir = os.path.join(t.data, "rr-prepare", "run")
    run_before = sorted(os.listdir(run_dir)) if os.path.isdir(run_dir) else []
    rc = t.run("rr-prepare", exe, "--passes=1")
    rp = t.j.get("rr_prepare", {})
    t.check("RE Requiem rr-prepare: result 4 (prepared by the game at its first start), exit 0, state in-game, no helper run",
            rc == 0 and rp.get("result") == 4 and rp.get("state") == "in-game" and
            (sorted(os.listdir(run_dir)) if os.path.isdir(run_dir) else []) == run_before and "prepares its kernels at its first start" in t.logtail(3000), (t.j, t.logtail(1500)))
    rc = t.run("rr-recheck", exe)
    rq = t.j.get("rr_recheck", {})
    t.check("RE Requiem rr-recheck: never stale (state in-game): the launcher's start leaves it to the game",
            rc == 0 and rq.get("stale") is False and rq.get("state_before") == "in-game", t.j)
    t.check("install: REFramework's mirror gets OptiScaler's DLL and the other root DLLs, record entries with the old copy backed up",
            read(os.path.join(root, "_storage_", "dxgi.dll")) == edition and
            read(os.path.join(root, "_storage_", "wronggpu_runtime.dll")) == read(os.path.join(root, "wronggpu_runtime.dll")) and
            "copy\t1\t" in "\n".join(x for x in rec.splitlines() if x.endswith("\t_storage_\\dxgi.dll")) and record_entry_sha(root, "_storage_\\dxgi.dll") == sha(edition),
            rec[-2500:])
    oini = ini(root, "OptiScaler.ini")
    t.check("install: REFramework untouched, nothing RE Requiem-specific in OptiScaler.ini (ExtendedStateRestore left as it was)",
            read(os.path.join(root, "dinput8.dll")) == refw_bytes and os.path.isfile(os.path.join(root, "reframework", "config.txt")) and
            "ExtendedStateRestore" not in oini and "RestoreComputeSignature" not in oini, oini[:600])
    s = rr_ini(root)
    t.check("install: Ray Reconstruction set up, Fast by default (rc5)", os.path.isfile(os.path.join(root, RR_SETTINGS)) and s.get("rr_mode") == "fast", s)
    # the game runs the interface from REFramework's mirror, so the RR log lands there; Last run reads it
    write(os.path.join(root, "_storage_", "wronggpu-rr.log"),
          "[05:00:00] load: process 9 attached\r\n[05:00:01] runtime: create instance 1706x960\r\n[05:01:00] timing, 300 frames: GPU 7.08 ms per frame\r\n")
    rc = t.run("status", exe)
    g = t.j.get("game", {})
    lr = g.get("last_run", {}).get("rr", {})
    t.check("status: installed, REFramework found; Last run's RR from the mirror's log (7.1 ms)", rc == 0 and g.get("status") in ("installed", "turned-off") and
            g.get("needs", {}).get("found") is True and lr.get("used") is True and lr.get("ms") == "7.1", g)
    # the window: the RR switch says what off means in this game (with the RR data: the synthetic preset-F file converted)
    data_on = False
    if synth_f and os.path.isfile(os.path.join(synth_f, "synth-common-v0.bin")):
        src = os.path.join(t.work, "rr-source-re9", "nvngx_dlssd.dll")
        write(src, synth_f_dll(synth_f))
        data_on = t.run("rr-data", "convert", "--dll=" + src, "--noscan") == 0
    rootN, exeN = re9_game(t, "Re9 Needs", refw=False)
    dump = os.path.join(t.work, "ui-dump-re9.json")
    rc2 = t.run("ui-dump", "--dump=" + dump, "--noscan", "--library=" + os.path.join(t.work, "games"))
    try:
        d = json.loads(read(dump).decode("utf-8"))
    except Exception as e:  # noqa: BLE001
        d = {"error": str(e)}
    pages = {gp.get("name"): gp.get("controls", []) for gp in d.get("games", [])}
    pg = {c["text"]: c for c in pages.get("Re9 Game", pages.get("re9", []))}  # installed from the command line: named after its exe
    sw = pg.get("Use WrongGPU (Ray Reconstruction)", {})
    if data_on:
        t.check("window: RE Requiem's RR switch: off means no path tracing, only the game's own ray tracing levels",
                rc2 == 0 and sw and "offers no path tracing, only its own ray tracing levels" in sw.get("help", "") and "Ray Reconstruction quality: Accurate" in pg,
                (rc2, sw, list(pg)[:40]))
    else:
        t.check("window (no RR data): RE Requiem's Convert row says the game offers no path tracing until then",
                rc2 == 0 and any("offers no path tracing" in k or "offers no path tracing" in c.get("help", "") for k, c in pg.items()), (rc2, list(pg)[:40]))
    pn = {c["text"]: c for c in pages.get("Re9 Needs", [])}
    # the header says what the game needs and why; the callout says what to do, with its two actions (Learn more, Check again)
    t.check("window: without REFramework, Install is held back (disabled), the status says why, the callout has Learn more and Check again",
            pn.get("Install", {}).get("enabled") is False and pn.get("Learn more about REFramework", {}).get("enabled") is True and
            pn.get("Check again for REFramework", {}).get("enabled") is True and
            any("Needs REFramework" in k and "can't load in the game without it" in k for k in pn) and
            any("Add REFramework to the game first" in k for k in pn), (list(pn)[:40]))
    rows = d.get("rows", [])
    t.check("window: the game list's row says what it needs", any(r.startswith("Re9 Needs - Needs REFramework") for r in rows), rows)
    shutil.rmtree(rootN, ignore_errors=True)
    # an update over an install made before the mirror was handled: REFramework's copy of the earlier OptiScaler DLL (ours by its
    # hash) is replaced with this one; a player's file REFramework mirrored stays
    rootO, exeO = re9_game(t, "Re9 Older")
    beforeO = snapshot(rootO)
    rc = t.run("install", exeO, "--opti=" + opti, "--noscan", "--force", "--experimental")
    old = b"MZ an earlier OptiScaler build (synthetic)"
    if rc == 0:
        edit_record(rootO, "_storage_\\dxgi.dll", drop=True)
        write(os.path.join(rootO, "dxgi.dll"), old)
        write(os.path.join(rootO, "_storage_", "dxgi.dll"), old)
        edit_record(rootO, "dxgi.dll", new_sha=sha(old))
        write(os.path.join(rootO, "_storage_", "steam_api64.dll"), read(os.path.join(rootO, "steam_api64.dll")))  # REFramework's copy of the game's file
    rc2 = t.run("repair", exeO, "--opti=" + opti, "--noscan", "--force", "--experimental")
    t.check("update over an earlier install: the mirror's copy of the earlier OptiScaler DLL (ours by its hash) replaced with this one, recorded",
            rc == 0 and rc2 == 0 and read(os.path.join(rootO, "dxgi.dll")) == edition and read(os.path.join(rootO, "_storage_", "dxgi.dll")) == edition and
            record_entry_sha(rootO, "_storage_\\dxgi.dll") == sha(edition), (rc, rc2, t.j.get("message"), t.logtail()))
    # REFramework removed after the install: the page says so (with Learn more), a repair still runs with a warning
    os.remove(os.path.join(rootO, "dinput8.dll"))
    rc = t.run("status", exeO)
    g = t.j.get("game", {})
    t.check("REFramework gone after the install: still installed, needs not found, its status says so (never installed and healthy)",
            rc == 0 and g.get("installed_version") and g.get("needs", {}).get("found") is False and g.get("status") == "needs-prerequisite", g)
    rc = t.run("repair", exeO, "--opti=" + opti, "--noscan", "--force", "--experimental")
    t.check("REFramework gone: a repair runs, its warning says the game blocks OptiScaler until REFramework is back",
            rc == 0 and any("REFramework isn't in this game any more" in w for w in t.j.get("warnings", [])), (rc, t.j.get("warnings"), t.j.get("message")))
    add_refw(rootO)
    rc = t.run("uninstall", exeO)
    afterO = snapshot(rootO)
    mirrored = afterO.pop("_storage_/steam_api64.dll", None)
    t.check("uninstall after the update: everything of WrongGPU's gone from the folder and the mirror; the player's mirrored file stays",
            rc == 0 and mirrored is not None and afterO == beforeO, (rc, t.j.get("message"), diff(beforeO, afterO)))
    # uninstall: the game folder byte for byte as before, the mirror's earlier copy put back, REFramework kept
    write(os.path.join(root, "_storage_", "wronggpu-rr.log"), "written again by the game")
    rc = t.run("uninstall", exe)
    after = snapshot(root)
    t.check("uninstall: byte for byte as before (the mirror's other copy back, REFramework kept, the mirror's log gone)",
            rc == 0 and after == before, (rc, t.j.get("message"), diff(before, after)))
    for r in (root, rootB, rootO):
        shutil.rmtree(r, ignore_errors=True)
    t.env.pop("WRONGGPU_TEST_EDITION_DLL", None)


def direct_game(t, name):
    """The Unreal Engine plugin layout of a game that creates Ray Reconstruction itself (Hogwarts Legacy, LEGO Batman,
    Borderlands 4): <root>\\<Project>\\Binaries\\Win64\\<Project>-Win64-Shipping.exe, AMD's FSR 3.1 DLL and the synthetic DLSS
    model file next to it, and the RR library (version 310.5, a synthetic stand-in that no converter takes) deep in Engine\\,
    without the game interface's RR plugin anywhere."""
    root = os.path.join(t.work, "games", name)
    shutil.rmtree(root, ignore_errors=True)
    b = os.path.join(root, "Brick", "Binaries", "Win64")
    exe = os.path.join(b, "Brick-Win64-Shipping.exe")
    write(exe, "mock game exe " * 200 + name)
    write(os.path.join(b, "amd_fidelityfx_dx12.dll"), suite.AMD_V1)
    write(os.path.join(b, "nvngx_dlss.dll"), t.model.file)
    write(os.path.join(b, "steam_api64.dll"), "untouched game file")
    write(os.path.join(root, "Brick.exe"), "the game's small starter (Unreal)")
    lib = os.path.join(root, "Engine", "Plugins", "Runtime", "Vendor", "RR", "Binaries", "ThirdParty", "Win64", "nvngx_dlssd.dll")
    write(lib, mockpe.make_dll({"ProductName": "RR library (test stand-in)"}, (310, 5, 0, 0), extra=b"synthetic, never loaded", dll_name="nvngx_dlssd.dll"))
    return root, b, exe


def test_direct_rr(t, opti, synth_f):
    """Direct-RR games (2026-11-04): a game that ships the RR library but takes no RR through the game interface creates RR itself.
    It counts as an RR game on the edition route (WrongGPU's edition of OptiScaler hands its RR calls to the RR plugin, plugin
    interface 4 or newer): the RR plugin is installed and the RR card works as in the other RR games. With another
    OptiScaler (no plugin interface 4) the files go in all the same, the result says RR isn't offered, and the card says RR is
    unavailable with the plain reason; a repair with the edition makes it available. The game-interface games stay as
    they were (the RR tests and the RE Requiem tests above)."""
    print("\n[direct-rr] a game that creates Ray Reconstruction itself (the Unreal plugin layout): RR on the edition route only")
    plugin = mock_rr(t, os.path.join("plugins", "WrongGPU-RR.asi"))
    ed_pkg, edition = edition_package(t, "edition-direct")
    t.env["WRONGGPU_TEST_EDITION_DLL"] = sha(edition)
    lad = t.env["LOCALAPPDATA"]
    fdata = os.path.join(lad, "WrongGPU", "model", "F", "model.wgcm")
    root, b, exe = direct_game(t, "Direct Brick")
    before = snapshot(root)
    rc = t.run("status", exe, "--noscan")
    g = t.j.get("game", {})
    t.check("status: the RR library without the game interface's RR plugin: uses RR, direct, the library named (deep in Engine), not installed",
            rc == 0 and g.get("rr_uses") is True and g.get("rr_direct") is True and "nvngx_dlssd.dll" in g.get("rr_why", "") and
            g.get("rr_why", "").startswith("Engine") and g.get("rr_installed") is False and g.get("rr_available") is True, g)
    # another OptiScaler (the player's own download, used when the edition can't be downloaded): no plugin interface 4
    rc = t.run("install", exe, "--opti=" + opti, "--noscan", "--force", "--experimental")
    log = read(t.log).decode("utf-8", "replace")
    warn = t.j.get("warnings", [])
    t.check("another OptiScaler: installed (RR files too), the result's warning says RR isn't offered (no OptiScaler edition with plugin interface 4)",
            rc == 0 and read(os.path.join(b, "plugins", "WrongGPU-RR.asi")) == plugin and record_entry_sha(b, "wronggpu-rr.ini") and
            any("no OptiScaler edition with plugin interface 4" in w for w in warn) and
            "the game creates Ray Reconstruction itself" in log, (rc, t.j.get("message"), warn, t.logtail()))
    rc = t.run("status", exe)
    g = t.j.get("game", {})
    t.check("status: installed with another OptiScaler: RR installed, direct, not the edition, RR not available with the plain reason",
            rc == 0 and g.get("rr_installed") is True and g.get("rr_direct") is True and g.get("optiscaler_edition") is False and
            g.get("rr_available") is False and "creates Ray Reconstruction itself" in g.get("rr_unavailable_why", ""), g)
    dump = os.path.join(t.work, "ui-dump-direct.json")

    def page():
        rc_ = t.run("ui-dump", "--dump=" + dump, "--noscan", "--library=" + os.path.join(t.work, "games"))
        try:
            d = json.loads(read(dump).decode("utf-8"))
        except Exception as e:  # noqa: BLE001
            d = {"error": str(e)}
        for gp in d.get("games", []):
            if (gp.get("exe") or "").lower() == exe.lower() or gp.get("name") in ("Direct Brick", "Brick", "Brick-Win64-Shipping"):
                return rc_, {c["text"]: c for c in gp.get("controls", [])}, gp.get("name")
        return rc_, {}, [gp.get("name") for gp in d.get("games", [])]

    rc2, pg, nm = page()
    unavailable = [k for k in pg if "isn't available here with this game's OptiScaler" in k]
    t.check("window: the RR card says RR is unavailable with this game's OptiScaler, in plain words (what it needs, Repair), no switch",
            rc2 == 0 and unavailable and "WrongGPU edition of OptiScaler" in unavailable[0] and "Repair installs the edition" in unavailable[0] and
            "Use WrongGPU (Ray Reconstruction)" not in pg and "Convert the Ray Reconstruction data" not in pg, (rc2, nm, list(pg)[:50]))
    t.check("window: the card's words make no sameness claim (vendor names: the wording ctest)",
            unavailable and not re.search(r"native|identical|bit-for-bit|same as", unavailable[0], re.I), unavailable)
    # a repair with WrongGPU's edition: RR available
    rc = t.run("repair", exe, "--opti=" + ed_pkg, "--noscan", "--force", "--experimental")
    log = read(t.log).decode("utf-8", "replace")
    proxy = t.j.get("proxy") or "dxgi.dll"
    t.check("repair with the edition: OptiScaler's DLL is the edition, the RR plugin stays, the log says the edition hands the game's RR to the plugin, no RR warning",
            rc == 0 and read(os.path.join(b, "dxgi.dll")) == edition and read(os.path.join(b, "plugins", "WrongGPU-RR.asi")) == plugin and
            "hands it to WrongGPU's RR plugin" in log and not any("no OptiScaler edition" in w for w in t.j.get("warnings", [])),
            (rc, proxy, t.j.get("message"), t.j.get("warnings"), t.logtail()))
    rc = t.run("status", exe)
    g = t.j.get("game", {})
    t.check("status after the repair: the edition, RR available, no reason", rc == 0 and g.get("optiscaler_edition") is True and
            g.get("rr_available") is True and "rr_unavailable_why" not in g and g.get("rr_direct") is True, g)
    # the RR data as for the game-interface games: this game's own library (310.5 stand-in) doesn't convert; the data comes from
    # another game's file (shared on the PC), and the game's settings file turns RR on
    if synth_f and os.path.isfile(os.path.join(synth_f, "synth-common-v0.bin")):
        if not os.path.isfile(fdata):
            src = os.path.join(t.work, "rr-source-ngx", "nvngx_dlssd.dll")
            write(src, synth_f_dll(synth_f))
            t.run("rr-data", "convert", "--dll=" + src, "--noscan")
        rc = t.run("status", exe)
        g = t.j.get("game", {})
        t.check("RR data from another game's file: valid, the game's RR on (its own older library not used)",
                os.path.isfile(fdata) and rr_ini(b).get("rr_enabled") == "1" and g.get("rr_enabled") is True, (rr_ini(b), g))
        rc2, pg, nm = page()
        sw = pg.get("Use WrongGPU (Ray Reconstruction)", {})
        t.check("window: with the edition and the data, the RR card has the switch (on) and Accurate, as in the other RR games",
                rc2 == 0 and sw.get("checked") is True and "Ray Reconstruction quality: Accurate" in pg and
                not any("isn't available here" in k for k in pg), (rc2, nm, list(pg)[:50]))
    else:
        rc2, pg, nm = page()
        t.check("window (no RR data): with the edition the RR card shows Convert, not the unavailable line",
                rc2 == 0 and "Convert the Ray Reconstruction data" in pg and not any("isn't available here" in k for k in pg), (rc2, nm, list(pg)[:50]))
    rc = t.run("uninstall", exe)
    t.check("uninstall: byte for byte as before (the RR library untouched)", rc == 0 and snapshot(root) == before, (rc, t.j.get("message"), diff(before, snapshot(root))))
    shutil.rmtree(root, ignore_errors=True)
    # rc5 (gamerules.h rrNotOffered): LEGO Batman has the same Unreal plugin layout, but its menu has no RR option: no RR
    # here (no RR files, no RR card but one line), and an older install's RR files go with a repair
    lego = "LEGOBatmanLotDK-Win64-Shipping.exe"
    nooffer = "LEGO Batman's menu has no Ray Reconstruction option (it would need a separate game mod), so WrongGPU's Ray Reconstruction isn't offered here. Model M works."
    lroot = os.path.join(t.work, "games", "Lego Brick")

    def lego_game(exe_name):
        shutil.rmtree(lroot, ignore_errors=True)
        lb = os.path.join(lroot, "LEGOBatmanLotDK", "Binaries", "Win64")
        write(os.path.join(lb, exe_name), "mock game exe " * 200 + "Lego Brick")
        write(os.path.join(lb, "amd_fidelityfx_dx12.dll"), suite.AMD_V1)
        write(os.path.join(lb, "nvngx_dlss.dll"), t.model.file)
        write(os.path.join(lroot, "Engine", "Plugins", "Runtime", "Vendor", "RR", "Binaries", "ThirdParty", "Win64", "nvngx_dlssd.dll"),
              mockpe.make_dll({"ProductName": "RR library (test stand-in)"}, (310, 5, 0, 0), extra=b"synthetic, never loaded", dll_name="nvngx_dlssd.dll"))
        return lb, os.path.join(lb, exe_name)

    def lego_page(lexe):
        rc_ = t.run("ui-dump", "--dump=" + dump, "--noscan", "--library=" + os.path.join(t.work, "games"))
        try:
            d = json.loads(read(dump).decode("utf-8"))
        except Exception as e:  # noqa: BLE001
            d = {"error": str(e)}
        for gp in d.get("games", []):
            if gp.get("name") == "Lego Brick":
                return rc_, {c["text"]: c for c in gp.get("controls", [])}
        return rc_, {}

    lb, lexe = lego_game(lego)
    lbefore = snapshot(lroot)
    rc = t.run("status", lexe, "--noscan")
    g = t.j.get("game", {})
    t.check("LEGO Batman (its Unreal plugin layout): not an RR game", rc == 0 and g.get("rr_uses") is False, g)
    rc = t.run("install", lexe, "--opti=" + ed_pkg, "--noscan", "--force", "--experimental")
    t.check("LEGO Batman install (with the edition): no RR plugin, no RR settings file, no RR warning",
            rc == 0 and not os.path.exists(os.path.join(lb, "plugins", "WrongGPU-RR.asi")) and not record_entry_sha(lb, "wronggpu-rr.ini") and
            not any("Ray Reconstruction" in w for w in t.j.get("warnings", [])), (rc, t.j.get("message"), t.j.get("warnings")))
    rc = t.run("repair", lexe, "--opti=" + ed_pkg, "--noscan", "--force", "--experimental")
    t.check("LEGO Batman repair: still no RR files", rc == 0 and not os.path.exists(os.path.join(lb, "plugins", "WrongGPU-RR.asi")), (rc, t.j.get("message")))
    rc2, pg = lego_page(lexe)
    t.check("window, LEGO Batman: the RR card is the one line, no switch, nothing to convert or prepare, no Engine.ini question",
            rc2 == 0 and nooffer in pg and "Use WrongGPU (Ray Reconstruction)" not in pg and "Convert the Ray Reconstruction data" not in pg and
            "Prepare Ray Reconstruction" not in pg and not any("Engine.ini" in k for k in pg), (rc2, list(pg)[:60]))
    rc = t.run("uninstall", lexe)
    t.check("LEGO Batman uninstall: byte for byte as before", rc == 0 and snapshot(lroot) == lbefore, (rc, diff(lbefore, snapshot(lroot))))
    # an older install with the RR files (made here under another exe name, which the plugin layout makes an RR game; then the
    # exe and its record line renamed to LEGO Batman's): status says no RR, a repair takes the RR files out by the record
    lb, oexe = lego_game("Older-Win64-Shipping.exe")
    rc = t.run("install", oexe, "--opti=" + ed_pkg, "--noscan", "--force", "--experimental")
    had = os.path.exists(os.path.join(lb, "plugins", "WrongGPU-RR.asi"))
    lexe = os.path.join(lb, lego)
    os.rename(oexe, lexe)
    recp = os.path.join(lb, "wronggpu-setup", "install.record")
    write(recp, read(recp).replace(b"Older-Win64-Shipping.exe", lego.encode()))
    rc2 = t.run("status", lexe, "--noscan")
    g = t.j.get("game", {})
    t.check("LEGO Batman with an older install's RR files: status says no RR (the RR files still in, as recorded)",
            rc == 0 and had and rc2 == 0 and g.get("rr_uses") is False and record_entry_sha(lb, "wronggpu-rr.ini"), (rc, had, rc2, g.get("rr_uses")))
    rc = t.run("repair", lexe, "--opti=" + ed_pkg, "--noscan", "--force", "--experimental")
    rc2 = t.run("status", lexe, "--noscan")
    g = t.j.get("game", {})
    t.check("LEGO Batman repair of that install: the RR plugin and RR settings file are gone (record-based), status no RR",
            rc == 0 and rc2 == 0 and not os.path.exists(os.path.join(lb, "plugins", "WrongGPU-RR.asi")) and not record_entry_sha(lb, "wronggpu-rr.ini") and
            g.get("rr_uses") is False, (rc, os.path.exists(os.path.join(lb, "plugins", "WrongGPU-RR.asi")), record_entry_sha(lb, "wronggpu-rr.ini"), t.logtail(1200)))
    t.run("uninstall", lexe)
    shutil.rmtree(lroot, ignore_errors=True)
    t.env.pop("WRONGGPU_TEST_EDITION_DLL", None)


def test_sm2_settings(t, opti):
    """Marvel's Spider-Man 2 (Brief 13; opt-in since 2026-11-04): an install writes none of the OptiScaler.ini keys of its
    experimental Ray Reconstruction and says nothing about it (Model M only). `set <game> sm2_rr_experimental=on` writes
    them (GPU-name spoof on, device id 0x2684, fakenvapi off in the section where the package's ini has it; no GPU name) and
    says the player's steps; off puts the install's values back; a repair keeps the player's choice; the uninstall takes
    the keys away either way. Another game refuses the setting."""
    print("\n[sm2] Marvel's Spider-Man 2: experimental Ray Reconstruction is an opt-in (off by default)")
    pkg = os.path.join(t.work, "opti-sm2")
    shutil.rmtree(pkg, ignore_errors=True)
    write(os.path.join(pkg, "OptiScaler.dll"), b"MZ mock OptiScaler (synthetic)")
    write(os.path.join(pkg, "OptiScaler.ini"), OPTI_INI + "[Spoofing]\nDxgi=auto\nSpoofedDeviceId=auto\n[HostSection]\nUseFakenvapi=auto\n")
    keys = lambda text: {m.group(1).lower(): m.group(2) for m in re.finditer(r"(?m)^\s*(\w+)\s*=\s*(\S*)\s*$", text)}
    oini = lambda b: read(os.path.join(b, "OptiScaler.ini")).decode("utf-8", "replace")
    steps = lambda msgs: any("turn on OptiScaler's spoofing for the game interface" in w and "keep ray-traced reflections off" in w for w in msgs)
    root, b, exe = game(t, "Spider-Man Two", exe="Spider-Man2.exe")
    before = snapshot(root)
    rc = t.run("install", exe, "--opti=" + pkg, "--noscan", "--force", "--experimental")
    installed = oini(b) if rc == 0 else ""
    k = keys(installed)
    msgs = t.j.get("warnings", []) + t.j.get("notes", [])
    t.check("Spider-Man 2 default install: OptiScaler.ini keeps Dxgi, SpoofedDeviceId and UseFakenvapi as the package has them (Model M only)",
            rc == 0 and k.get("dxgi") == "auto" and k.get("spoofeddeviceid") == "auto" and k.get("usefakenvapi") == "auto", (rc, t.j.get("message"), k))
    t.check("Spider-Man 2 default install: no Ray Reconstruction steps in the result", rc == 0 and not steps(msgs), msgs)
    rc = t.run("get", exe, "sm2_rr_experimental")
    t.check("Spider-Man 2: sm2_rr_experimental is off by default",
            rc == 0 and t.j.get("settings", {}).get("sm2_rr_experimental", {}).get("profile") == "off", (rc, t.j.get("settings")))
    rc = t.run("set", exe, "sm2_rr_experimental=on")
    on = oini(b) if rc == 0 else ""
    k = keys(on)
    msgs = t.j.get("warnings", []) + t.j.get("notes", [])
    t.check("sm2_rr_experimental=on: OptiScaler.ini gets Dxgi=true, SpoofedDeviceId=0x2684 and UseFakenvapi=false (in the section that has it)",
            rc == 0 and k.get("dxgi") == "true" and k.get("spoofeddeviceid") == "0x2684" and
            re.search(r"(?ms)^\[HostSection\]\s*^UseFakenvapi\s*=\s*false", on) is not None, (rc, t.j.get("error"), on[-600:]))
    t.check("sm2_rr_experimental=on: no GPU name is written", rc == 0 and "spoofedgpuname" not in k, sorted(k))
    t.check("sm2_rr_experimental=on: the result says the player's steps and that it is not recommended",
            steps(msgs) and any("not recommended" in w for w in msgs), msgs)
    rc = t.run("get", exe, "sm2_rr_experimental")
    st = t.j.get("settings", {}).get("sm2_rr_experimental", {})
    t.check("sm2_rr_experimental=on: get shows it on in the profile and in the file", rc == 0 and st.get("profile") == "on" and st.get("file") == "on", (rc, st))
    rc = t.run("set", exe, "sm2_rr_experimental=off")
    t.check("sm2_rr_experimental=off: OptiScaler.ini is the installed one again, byte for byte", rc == 0 and oini(b) == installed,
            (rc, t.j.get("error"), oini(b)[-600:]))
    rc = t.run("set", exe, "sm2_rr_experimental=on")
    rc2 = t.run("install", exe, "--opti=" + pkg, "--noscan", "--force", "--experimental")
    k = keys(oini(b)) if rc == 0 and rc2 == 0 else {}
    t.check("sm2_rr_experimental=on: a repair keeps the player's choice (the keys are written again)",
            rc == 0 and rc2 == 0 and k.get("dxgi") == "true" and k.get("spoofeddeviceid") == "0x2684" and k.get("usefakenvapi") == "false", (rc, rc2, k))
    rc = t.run("set", exe, "sm2_rr_experimental=off")
    t.check("sm2_rr_experimental=off after a repair: the installed values are back", rc == 0 and oini(b) == installed, (rc, oini(b)[-600:]))
    rc = t.run("set", exe, "sm2_rr_experimental=on")
    rc2 = t.run("uninstall", exe)
    t.check("Spider-Man 2 with sm2_rr_experimental=on: uninstall leaves the game folder as before (the keys went with the installed OptiScaler.ini)",
            rc == 0 and rc2 == 0 and snapshot(root) == before, (rc, rc2, diff(before, snapshot(root))))
    shutil.rmtree(root, ignore_errors=True)
    # a package ini without UseFakenvapi: nothing invented for it
    write(os.path.join(pkg, "OptiScaler.ini"), OPTI_INI + "[Spoofing]\nDxgi=auto\nSpoofedDeviceId=auto\n")
    root, b, exe = game(t, "Spider-Man Two Plain", exe="Spider-Man2.exe")
    rc = t.run("install", exe, "--opti=" + pkg, "--noscan", "--force", "--experimental")
    rc2 = t.run("set", exe, "sm2_rr_experimental=on") if rc == 0 else -1
    k = keys(oini(b)) if rc2 == 0 else {}
    t.check("sm2_rr_experimental=on with a package ini that has no UseFakenvapi key: none written, the other keys are",
            rc2 == 0 and "usefakenvapi" not in k and k.get("dxgi") == "true" and k.get("spoofeddeviceid") == "0x2684", (rc, rc2, k))
    t.run("uninstall", exe)
    shutil.rmtree(root, ignore_errors=True)
    # another game: none of these keys, no notice, and the setting is refused
    write(os.path.join(pkg, "OptiScaler.ini"), OPTI_INI + "[Spoofing]\nDxgi=auto\nSpoofedDeviceId=auto\n[HostSection]\nUseFakenvapi=auto\n")
    root, b, exe = game(t, "Not Spider-Man")
    rc = t.run("install", exe, "--opti=" + pkg, "--noscan", "--force", "--experimental")
    k = keys(oini(b)) if rc == 0 else {}
    t.check("another game: none of Spider-Man 2's keys, no notice", rc == 0 and k.get("dxgi") == "auto" and k.get("spoofeddeviceid") == "auto" and
            k.get("usefakenvapi") == "auto" and not any("Spider-Man 2" in w for w in t.j.get("warnings", []) + t.j.get("notes", [])), (rc, k))
    rc = t.run("set", exe, "sm2_rr_experimental=on")
    k = keys(oini(b))
    t.check("another game: set sm2_rr_experimental=on is refused and changes nothing",
            rc != 0 and "Spider-Man2.exe only" in str(t.j.get("error", "")) and k.get("dxgi") == "auto", (rc, t.j.get("error"), k))
    t.run("uninstall", exe)
    shutil.rmtree(root, ignore_errors=True)
    # rc5 (gamerules.h): the window's RR control in Spider-Man 2 (an RR game: DLSS Ray Reconstruction's plugin next to the
    # exe). Off (the default): RR off and not switchable, one plain line; on (command line): labelled Experimental with the
    # warning, and the switch turns the opt-in off
    root, b, exe = game(t, "Spider-Man Two RR", exe="Spider-Man2.exe")
    write(os.path.join(b, "sl.dlss_d.dll"), b"MZ synthetic stand-in")  # the DLSS RR plugin stand-in (an RR game)
    dump = os.path.join(t.work, "ui-dump-sm2.json")

    def sm2_page():
        rc_ = t.run("ui-dump", "--dump=" + dump, "--noscan", "--library=" + os.path.join(t.work, "games"))
        try:
            d = json.loads(read(dump).decode("utf-8"))
        except Exception as e:  # noqa: BLE001
            d = {"error": str(e)}
        for gp in d.get("games", []):
            if gp.get("name") == "Spider-Man Two RR":
                return rc_, {c["text"]: c for c in gp.get("controls", [])}
        return rc_, {}

    held = "Ray Reconstruction isn't stable in Spider-Man 2 yet, so it's off. Model M works. (Experimental opt-in: see the release notes.)"
    rc = t.run("install", exe, "--opti=" + pkg, "--noscan", "--force", "--experimental")
    rc2 = t.run("status", exe)
    g = t.j.get("game", {})
    t.check("Spider-Man 2 with its RR plugin: installed as an RR game (RR files in)", rc == 0 and rc2 == 0 and g.get("rr_uses") is True and g.get("rr_installed") is True,
            (rc, t.j.get("message"), g.get("rr_uses"), g.get("rr_installed")))
    rc, pg = sm2_page()
    t.check("window, sm2_rr_experimental off: the RR card says it's off with the one plain line, no RR switch, nothing to convert, set up or prepare",
            rc == 0 and held in pg and "Use WrongGPU (Ray Reconstruction)" not in pg and "Use WrongGPU (Ray Reconstruction, experimental)" not in pg and
            "Convert the Ray Reconstruction data" not in pg and "Prepare Ray Reconstruction" not in pg and "Set up Ray Reconstruction" not in pg, (rc, list(pg)[:60]))
    rc = t.run("set", exe, "sm2_rr_experimental=on")
    rc2, pg = sm2_page()
    data_ready = os.path.isfile(os.path.join(t.env["LOCALAPPDATA"], "WrongGPU", "model", "F", "model.wgcm"))
    sw = pg.get("Use WrongGPU (Ray Reconstruction, experimental)", {})
    t.check("window, sm2_rr_experimental on (command line): the held-off line is gone; with the RR data the switch is on, labelled Experimental, with the freeze / crash warning",
            rc == 0 and rc2 == 0 and held not in pg and
            ((sw.get("checked") is True and "may freeze or crash" in json.dumps(pg) and "Use WrongGPU (Ray Reconstruction)" not in pg) if data_ready
             else "Convert the Ray Reconstruction data" in pg), (rc, rc2, data_ready, list(pg)[:60]))
    rc = t.run("set", exe, "sm2_rr_experimental=off")
    rc2, pg = sm2_page()
    t.check("window, sm2_rr_experimental off again: the held-off line is back", rc == 0 and rc2 == 0 and held in pg, (rc, rc2, list(pg)[:40]))
    t.run("uninstall", exe)
    shutil.rmtree(root, ignore_errors=True)


def rr_config_line():
    """The one Engine.ini line, read from its one constant (src/wg_core/rrconfig_line.h): this file never spells it."""
    text = read(os.path.join(HERE, "..", "src", "wg_core", "rrconfig_line.h")).decode("utf-8")
    m = re.search(r'kRrConfigLine\[\] = "([^"]+)"', text)
    return m.group(1).encode("ascii")


def read_only(p):
    import stat
    return bool(os.stat(p).st_file_attributes & stat.FILE_ATTRIBUTE_READONLY)


def set_writable(root):
    import stat
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            os.chmod(os.path.join(dirpath, fn), stat.S_IWRITE | stat.S_IREAD)


def outside_pre_exposure(b):
    # a value changed outside the launcher: pre_exposure=original added to wronggpu.ini's runtime_options line (r3 #1)
    out = []
    for line in ini(b).split("\n"):
        if line.strip().lower().startswith("runtime_options") and "=" in line:
            cr = line.endswith("\r")
            line = line.rstrip("\r")
            line += ("" if line.rstrip().endswith("=") else ";") + "pre_exposure=original" + ("\r" if cr else "")
        out.append(line)
    write(os.path.join(b, "wronggpu.ini"), "\n".join(out))


def changed_keys(before, after):
    return sorted(k for k in set(before) | set(after) if before.get(k) != after.get(k))


def test_rr_config_fix(t, opti):
    """Brief 16 (owner 2026-11-04): Borderlands 4 and Hogwarts Legacy need one line in the player's own Engine.ini for Ray
    Reconstruction, and Borderlands 4 deletes that file unless it is read-only. An install asks (the result's
    rr_config_question: the notice, the steps with the line, how to answer); rr_config_fix=on (set, or --rr-config-fix=on)
    backs the file up or notes it absent, adds the line only where it is missing (under an existing [SystemSettings], a
    key with another value set to 1), makes Borderlands 4's file read-only and records it; repair applies it again; =off
    and the uninstall undo it byte for byte. Test roots only: WRONGGPU_TEST_DOCUMENTS and this run's LOCALAPPDATA (the
    build refuses the PC's real folders). Other games refuse the setting."""
    print("\n[rr-config] Borderlands 4 / Hogwarts Legacy: the Engine.ini setting for Ray Reconstruction")
    sys.path.insert(0, os.path.join(HERE, "..", "tools"))
    import check_wording  # noqa: E402
    line = rr_config_line()
    new_file = b"[SystemSettings]\r\n" + line + b"\r\n"
    t.check("the line's constant gives the 45-byte file of the in-game test", len(new_file) == 45 and line.count(b"=") == 1, (len(new_file), line))
    docs = os.path.join(t.work, "documents")
    shutil.rmtree(docs, ignore_errors=True)
    os.makedirs(docs)
    t.env["WRONGGPU_TEST_DOCUMENTS"] = docs
    bl4_cfg = os.path.join(docs, "My Games", "Borderlands 4", "Saved", "Config")
    bl4_ini = os.path.join(bl4_cfg, "Windows", "Engine.ini")
    get_fix = lambda exe: (t.run("get", exe, "rr_config_fix"), t.j.get("settings", {}).get("rr_config_fix", {}), t.j.get("rr_config", {}))
    msgs = lambda: t.j.get("warnings", []) + t.j.get("notes", [])
    waits = lambda: any("will be added after you start the game once" in n for n in t.j.get("notes", []))

    def row(pg, start):
        return [k for k in pg if k.startswith(start)]
    try:
        def rr_page(tag, env=None, extra=()):
            dmp = os.path.join(t.work, "ui-dump-rrconfig-%s.json" % tag)
            keep = t.env
            if env is not None:
                t.env = env
            try:
                rcx = t.run("ui-dump", "--dump=" + dmp, "--noscan", "--library=" + os.path.join(t.work, "games"), *extra)
            finally:
                t.env = keep
            try:
                dd = json.loads(read(dmp).decode("utf-8"))
            except Exception as e:  # noqa: BLE001
                dd = {"error": str(e)}
            for gp in dd.get("games", []):
                if gp.get("rrconfig"):
                    gp["rrconfig"]["due"] = gp.get("rrconfig_due", {})
                    gp["rrconfig"]["show_me_how_answer"] = gp.get("show_me_how_answer", {})
                    gp["rrconfig"]["fix_for_me_answer"] = gp.get("fix_for_me_answer", {})
                    return rcx, {c["text"]: c for c in gp.get("controls", [])}, gp["rrconfig"]
            return rcx, {}, dd.get("error", [gp.get("name") for gp in dd.get("games", [])])
        # 1. no Engine.ini yet: the install asks and changes nothing; on waits for the game's settings folder, then creates the 45-byte file, read-only
        root, b, exe = game(t, "Borderlands Four", exe="Borderlands4.exe")
        # r7 #5: the game ships the RR library (Unreal's plugin layout, as Borderlands 4: a synthetic stand-in), so WrongGPU's RR files go
        # in and the window's own apply below runs on a real RR install (without them it never adds the line by itself: checked below)
        rr_lib = os.path.join(root, "Engine", "Plugins", "Runtime", "Vendor", "RR", "Binaries", "ThirdParty", "Win64", "nvngx_dlssd.dll")
        write(rr_lib, mockpe.make_dll({"ProductName": "RR library (test stand-in)"}, (310, 5, 0, 0), extra=b"synthetic, never loaded", dll_name="nvngx_dlssd.dll"))
        before = snapshot(root)
        rc = t.run("install", exe, "--opti=" + opti, "--noscan", "--force", "--experimental")
        q = t.j.get("rr_config_question", {})
        text = q.get("text", "")
        # the game named in the title, the recommended way first, every choice changeable on the game's page (UI review #2)
        t.check("Borderlands 4 install: the result asks (the game named, what the line is for, deleted on every start, WrongGPU can add it, changeable on the game's page)",
                rc == 0 and text.startswith("Borderlands 4 needs one more setting for Ray Reconstruction") and "deletes its Engine.ini on every start" in text and
                "Ray Reconstruction in Borderlands 4 needs one line in Engine.ini, because the game's own menu can't turn it on." in text and
                "WrongGPU can add the line" in text and "on the game's page at any time" in text and "uninstalling WrongGPU takes it out" in text,
                (rc, t.j.get("message"), q))
        t.check("the question's steps have the line, the game's folder and read-only; the fix and myself commands name the setting",
                line.decode() in q.get("steps", "") and os.path.dirname(bl4_ini) in q.get("steps", "") and "Read-only" in q.get("steps", "") and
                q.get("fix", "").endswith("rr_config_fix=on") and q.get("myself", "").endswith("rr_config_fix=off"), q)
        bad = [p for p in check_wording.PHRASES if p in (text + q.get("steps", "")).lower()]
        # the line's own words (its key's parts) appear nowhere but in the line itself
        words = [w for w in line.split(b"=")[0].decode().split(".") if len(w) >= 3]
        rest = (text + q.get("steps", "").replace(line.decode(), "")).lower()
        brands = [w for w in words if w.lower() in rest]
        t.check("the notice and steps pass the wording check, and the line's words appear only in the line", words and not bad and not brands, (bad, brands))
        t.check("...and nothing was written into the Documents test root", rc == 0 and not os.listdir(docs), os.listdir(docs))
        rc, st, info = get_fix(exe)
        t.check("rr_config_fix is ask (not decided) after the install", rc == 0 and st.get("profile") == "ask" and info.get("fixed") is False, (rc, st, info))
        # before the game's first start (no sub-folder holds GameUserSettings.ini: Windows or WindowsNoEditor is a guess until
        # then): the choice is kept, nothing is created, the result says when it is added (UI review r2b #1)
        rc = t.run("set", exe, "rr_config_fix=on")
        t.check("set rr_config_fix=on before the game's first start: nothing created, no record line, the result says it is added after the first start (r2b #1)",
                rc == 0 and not os.listdir(docs) and "rrconfig\t" not in record_text(b) and waits() and not t.j.get("warnings"), (rc, os.listdir(docs), msgs()))
        rc, st, info = get_fix(exe)
        t.check("...get: on in the profile, not added, waiting for the game's settings folder (-1)",
                rc == 0 and st.get("profile") == "on" and info.get("fixed") is False and info.get("waiting") == -1, (st, info))
        rx, pg, rcf = rr_page("waiting")
        t.check("the Engine.ini question in a folder that needs no permission: no permission sentence (r4 #9)",
                rx == 0 and rcf.get("question_text", "").startswith("Borderlands 4 needs one more setting") and "Windows asks" not in rcf.get("question_text", ""),
                rcf.get("question_text"))
        t.check("an install with WrongGPU's RR files: the window may ask the Engine.ini question after it; the card doesn't say RR is missing (r7 #5)",
                rx == 0 and "WrongGPU-RR.asi" in record_text(b) and rcf.get("rr_installed") is True and rcf.get("askable") is True and
                not any("Ray Reconstruction isn't installed for this game" in c.get("help", "") for c in pg.values()), (rcf.get("rr_installed"), rcf.get("askable")))
        t.check("window, WrongGPU's choice waiting: the card says it will be added after the first start, with Check again; nothing made (r2b #1)",
                rx == 0 and row(pg, "Will be added after you start the game once. Borderlands 4 makes its settings folder at its first start") and
                "Check Engine.ini again" in pg and "Fix Engine.ini again" not in pg and rcf.get("state") == -1 and
                rcf.get("due", {}).get("due") is False and rcf.get("due", {}).get("fixed") is False and
                "WrongGPU adds the line after you start the game once" in rcf.get("check_toast", "") and not os.listdir(docs), (rx, rcf, [k for k in pg if "ngine" in k or "added" in k]))
        rc = t.run("set", exe, "rr_config_fix=off")
        t.check("=off while it waits: nothing created or taken out, no record line, off in the profile (r2b #1)",
                rc == 0 and not os.listdir(docs) and "rrconfig\t" not in record_text(b) and get_fix(exe)[1].get("profile") == "off", (rc, os.listdir(docs), msgs()))
        rc = t.run("set", exe, "rr_config_fix=on")
        # the game's first start makes its settings folder (GameUserSettings.ini in Windows): the choice is due, and the window
        # applies it when it opens or shows the page (ui-dump --apply=rrdue: as the window does it)
        bl4_gus = os.path.join(os.path.dirname(bl4_ini), "GameUserSettings.ini")
        write(bl4_gus, b"[ScalabilityGroups]\r\n")
        rc, st, info = get_fix(exe)
        t.check("the game made its settings folder (Windows): get says the choice is due (waiting 1), nothing written yet",
                rc == 0 and info.get("waiting") == 1 and not os.path.exists(bl4_ini), (st, info))
        # every permission prompt the window would raise is listed (a test build: cli.cpp StartElevated); none for a game whose
        # folder needs none (r2c)
        ask_log = os.path.join(t.work, "elevation-asks.txt")
        if os.path.exists(ask_log):
            os.remove(ask_log)
        asks = lambda: read(ask_log).decode("utf-8", "replace").splitlines() if os.path.exists(ask_log) else []
        # r3 #1: a value changed outside the launcher before the window's own apply (wronggpu.ini's pre_exposure): only Engine.ini,
        # its backup and the install record are written, so it stays (its own Keep / Use buttons decide), and so do the page's
        # unsaved changes and every other file of the game
        outside_pre_exposure(b)
        rc = t.run("get", exe)
        t.check("(r3 #1) the outside change is reported before the window's own apply",
                rc == 0 and any(x["key"] == "pre_exposure" and x["file"] == "original" for x in t.j.get("outside_edits", [])), t.j.get("outside_edits"))
        game_before = snapshot(root)
        rx, pg, rcf = rr_page("due", dict(t.env, WRONGGPU_TEST_ELEVATION_LOG=ask_log), extra=("--apply=rrdue",))
        du = rcf.get("due", {}) if isinstance(rcf, dict) else {}
        ch = changed_keys(game_before, snapshot(root))
        t.check("...the window's own apply on the game's page writes Engine.ini alone: only the install record changed in the game folder, "
                "wronggpu.ini byte for byte with the outside pre_exposure, the page's unsaved change still unsaved (r3 #1)",
                ch and all(k.endswith("wronggpu-setup/install.record") for k in ch) and "pre_exposure=original" in (options_line(ini(b)) or "") and
                du.get("from_library") is False and du.get("edit_kept") is True, (ch, options_line(ini(b)), du.get("edit_kept")))
        t.check("window opens with the folder there: the choice applied once (Borderlands 4: Windows\\Engine.ini, the 45-byte file, read-only), the toast says so (r2b #1)",
                rx == 0 and du.get("due") is True and du.get("entry_due") is True and du.get("started") is True and du.get("applied") is True and
                du.get("state") == 1 and du.get("fixed_after") is True and "has the Ray Reconstruction setting now" in du.get("toast", "") and du.get("again") is False and
                read(bl4_ini) == new_file and read_only(bl4_ini), (rx, du))
        t.check("...a game whose folder needs no permission: applied by itself, no permission prompt (r2c)",
                du.get("protected") is False and du.get("asks") == 0 and not asks(), (du, asks()))
        t.check("...an apply that couldn't start first (the last session's list shown) wasn't marked as tried: it started once it could (r3 #7)",
                du.get("blocked_started") is False and du.get("started") is True and du.get("applied") is True, du)
        rec = record_text(b)
        t.check("...recorded in the install record (created, read-only set, key absent before, no folder made)",
                re.search(r"(?m)^rrconfig\t" + re.escape(bl4_ini) + r"\tcr\tabsent\t-\r?$", rec) is not None,
                [l for l in rec.splitlines() if l.startswith("rrconfig")])
        rc, st, info = get_fix(exe)
        t.check("get: rr_config_fix on in the profile and in the file", rc == 0 and st.get("profile") == "on" and st.get("file") == "on" and info.get("fixed") is True, (st, info))
        rc = t.run("install", exe, "--opti=" + opti, "--noscan", "--force", "--experimental")
        t.check("repair: the file is applied again (one line, read-only), the record keeps it, no question", rc == 0 and read(bl4_ini) == new_file and
                read_only(bl4_ini) and "rrconfig\t" in record_text(b) and "rr_config_question" not in t.j, (rc, t.j.get("message"), msgs()))
        rc = t.run("set", exe, "rr_config_fix=off")
        t.check("set rr_config_fix=off: the file it created is gone, the game's settings folder stays", rc == 0 and not os.path.exists(bl4_ini) and
                os.listdir(os.path.dirname(bl4_ini)) == ["GameUserSettings.ini"] and "rrconfig\t" not in record_text(b), (rc, os.listdir(docs)))
        # r3 #1: the same from the Library (the launcher opening, the game's page not shown), with an outside change again; then
        # the uninstall takes the line out and leaves the game folder as it was before the install
        shutil.rmtree(os.path.join(docs, "My Games"))
        rc = t.run("set", exe, "rr_config_fix=on")
        write(bl4_gus, b"[ScalabilityGroups]\r\n")
        outside_pre_exposure(b)
        game_before = snapshot(root)
        rx, pg, rcf = rr_page("due-library", dict(t.env, WRONGGPU_TEST_ELEVATION_LOG=ask_log), extra=("--apply=rrdue-library",))
        du = rcf.get("due", {}) if isinstance(rcf, dict) else {}
        ch = changed_keys(game_before, snapshot(root))
        t.check("window, the Library shown: the choice applied by itself (the 45-byte read-only file, recorded), no prompt; only the install record "
                "changed in the game folder, the outside pre_exposure kept, the page's unsaved change kept (r3 #1)",
                rx == 0 and du.get("from_library") is True and du.get("started") is True and du.get("applied") is True and du.get("state") == 1 and
                du.get("asks") == 0 and du.get("edit_kept") is True and read(bl4_ini) == new_file and read_only(bl4_ini) and
                re.search(r"(?m)^rrconfig\t" + re.escape(bl4_ini) + r"\tcr\tabsent\t-\r?$", record_text(b)) is not None and
                ch and all(k.endswith("wronggpu-setup/install.record") for k in ch) and "pre_exposure=original" in (options_line(ini(b)) or ""),
                (rx, du, ch, options_line(ini(b))))
        rc = t.run("uninstall", exe)
        t.check("...uninstall takes that line out: Engine.ini gone, the game's settings folder stays, the game folder as before the install (r3 #1)",
                rc == 0 and not os.path.exists(bl4_ini) and os.listdir(os.path.dirname(bl4_ini)) == ["GameUserSettings.ini"] and snapshot(root) == before,
                (rc, msgs(), diff(before, snapshot(root))))
        os.remove(rr_lib)  # r6 #5 / r7 #5: the game without the RR library now, so WrongGPU's RR files don't go in
        rc = t.run("install", exe, "--opti=" + opti, "--noscan", "--force", "--experimental")
        t.check("...installed again for the checks below (without WrongGPU's RR files)", rc == 0 and "WrongGPU-RR.asi" not in record_text(b), (rc, t.j.get("message")))
        # r6 #5: an install without WrongGPU's RR files (offline, or not the WrongGPU edition: this game ships no RR library) has no
        # RR of WrongGPU's for the line to serve: in a protected folder with the choice on and the settings folder there, nothing
        # about it needs the player: not in Needs attention, no Fix it now, no toast, no prompt
        keep_env0 = t.env
        t.env = {k: v for k, v in t.env.items() if k != "WRONGGPU_TEST_DOCUMENTS"}  # (the CLI's own apply can't add the line meanwhile)
        try:
            rc = t.run("set", exe, "rr_config_fix=on")
        finally:
            t.env = keep_env0
        n_asks = len(asks())
        rx, pg, rcf = rr_page("protected-library-norr", dict(t.env, WRONGGPU_TEST_PROTECTED=root, WRONGGPU_TEST_ELEVATION_LOG=ask_log),
                              extra=("--apply=rrdue-library",))
        du = rcf.get("due", {}) if isinstance(rcf, dict) else {}
        lb = du.get("library", {})
        t.check("an install without WrongGPU's RR files, protected, the choice on, the settings folder there: not in Needs attention (not ready, no Fix "
                "it now pill, no line on its tile), no toast, no prompt, nothing written (r6 #5)",
                rc == 0 and rx == 0 and "WrongGPU-RR.asi" not in record_text(b) and du.get("due") is True and du.get("protected") is True and
                lb.get("ready") is False and not lb.get("pill", "").startswith("Fix Engine.ini now") and "ready to add" not in lb.get("tile", "") and
                "ready to add" not in du.get("toast", "") and len(asks()) == n_asks and not os.path.exists(bl4_ini) and "rrconfig\t" not in record_text(b),
                (rc, rx, du, asks()))
        # r7 #5: ...and in a folder that needs no permission: the window's own apply adds nothing by itself (no line, no record entry,
        # no prompt); the window asks no Engine.ini question for it; the card says WrongGPU's Ray Reconstruction isn't installed here
        rx, pg, rcf = rr_page("due-norr", dict(t.env, WRONGGPU_TEST_ELEVATION_LOG=ask_log), extra=("--apply=rrdue",))
        du = rcf.get("due", {}) if isinstance(rcf, dict) else {}
        t.check("an install without WrongGPU's RR files, a folder that needs no permission, the choice on, the settings folder there: the window's "
                "own apply doesn't start, nothing written, no prompt (r7 #5)",
                rx == 0 and du.get("due") is True and du.get("entry_due") is False and du.get("protected") is False and du.get("started") is False and
                not os.path.exists(bl4_ini) and "rrconfig\t" not in record_text(b) and len(asks()) == n_asks, (rx, du, asks()))
        t.check("...no Engine.ini question after an install there, and the card says WrongGPU's Ray Reconstruction isn't installed for this game (r7 #5)",
                rcf.get("rr_installed") is False and rcf.get("askable") is False and
                any("WrongGPU's Ray Reconstruction isn't installed for this game, so this line does nothing for now." in c.get("help", "") for c in pg.values()),
                (rcf.get("askable"), [c.get("help") for k, c in pg.items() if k.startswith("Engine.ini setting")][:1]))
        # ...then the game ships the RR library (Unreal's plugin layout, as Borderlands 4): installed again, WrongGPU's RR files go in,
        # so the checks below exercise the line of a real RR install
        rc = t.run("uninstall", exe)
        write(os.path.join(root, "Engine", "Plugins", "Runtime", "Vendor", "RR", "Binaries", "ThirdParty", "Win64", "nvngx_dlssd.dll"),
              mockpe.make_dll({"ProductName": "RR library (test stand-in)"}, (310, 5, 0, 0), extra=b"synthetic, never loaded", dll_name="nvngx_dlssd.dll"))
        before = snapshot(root)  # (the game folder before the install now has the RR library: the uninstall checks below compare with it)
        rc2 = t.run("install", exe, "--opti=" + opti, "--noscan", "--force", "--experimental")
        t.check("...installed again with WrongGPU's RR files (the game ships the RR library) (r6 #5)",
                rc == 0 and rc2 == 0 and "WrongGPU-RR.asi" in record_text(b) and os.path.isfile(os.path.join(b, RR_SETTINGS)), (rc, rc2, t.j.get("message")))

        def rr_data_ready():
            """RR on in the game's RR settings file, as the RR data's conversion leaves an RR install (RefreshRrSettings): this run has
            no RR data, so the install wrote rr_enabled=0 (r6 #5)"""
            p = os.path.join(b, RR_SETTINGS)
            write(p, re.sub(rb"(?m)^(rr_enabled[ \t]*=[ \t]*)0", rb"\g<1>1", read(p)))
            return rr_ini(b).get("rr_enabled") == "1"
        # r2c: the game in a protected folder (WRONGGPU_TEST_PROTECTED: its folder counts as needing administrator permission).
        # The window never adds the line by itself there: opening the launcher or the page, or Check again, asks Windows for
        # nothing; the card says "Ready to add" with Fix it now, and that click brings the prompt (the test build runs the elevated
        # worker without one: cli.cpp). The CLI below runs as an elevated run would (the folder isn't protected for it).
        prot_env = dict(t.env, WRONGGPU_TEST_PROTECTED=root, WRONGGPU_TEST_ELEVATION_LOG=ask_log)
        shutil.rmtree(os.path.join(docs, "My Games"))  # before the game's first start again
        # r6 #7: the question's action sequence in a protected folder before the game's first start, as its text says it: Fix it for
        # me saves the choice at once through the elevated worker (Windows asks now: one prompt), and nothing goes into Engine.ini (no
        # settings folder yet); after the game's first start the window adds nothing by itself (no prompt), the card says Ready to
        # add, and Fix it now adds the line (Windows asks again: a second prompt)
        rc = t.run("set", exe, "rr_config_fix=ask")
        if os.path.exists(ask_log):
            os.remove(ask_log)
        rx, pg, rcf = rr_page("protected-fixforme-first", prot_env, extra=("--apply=rrfixforme",))
        ff = rcf.get("fix_for_me_answer", {}) if isinstance(rcf, dict) else {}
        qt = rcf.get("question_text", "") if isinstance(rcf, dict) else ""
        # r7 #7: three sentences under the title, nothing said twice: what the line is for, that WrongGPU adds it after the first start
        # (read-only, so the game keeps it), and when Windows asks (twice: Fix it for me, then Fix it now after that start)
        body = qt.split("\n\n", 1)[-1]
        t.check("protected folder before the game's first start: the question says, in three sentences without repeats, that the line goes in after "
                "the first start and that Windows asks twice: at Fix it for me and at Fix it now after that start (r6 #7, r7 #7)",
                rc == 0 and rx == 0 and body == "Ray Reconstruction in Borderlands 4 needs one line in Engine.ini, because the game's own menu can't turn "
                "it on. WrongGPU can add it after the game's first start and keep the file read-only, so Borderlands 4 doesn't delete it. Borderlands 4 "
                "is in a protected folder, so Windows asks for permission twice: when you click Fix it for me, and when you click Fix it now on the "
                "game's page after that start." and body.count("deletes") + body.count("delete") == 1 and "You can change" not in body,
                qt)
        t.check("...Fix it for me: the elevated worker saves the choice at once (one permission prompt, its apply job), on in the profile, nothing in "
                "Engine.ini or the record, the card waits for the first start (r6 #7)",
                ff.get("busy_started") is True and ff.get("done") is True and ff.get("saved") == "on" and ff.get("state") == -1 and
                ff.get("notice") == "" and len(asks()) == 1 and "apply" in asks()[0] and "--job=" in asks()[0] and get_fix(exe)[1].get("profile") == "on" and
                not os.listdir(docs) and "rrconfig\t" not in record_text(b), (ff, asks(), os.listdir(docs)))
        write(bl4_gus, b"[ScalabilityGroups]\r\n")  # the game's first start
        rx, pg, rcf = rr_page("protected-fixforme-then", prot_env, extra=("--apply=rrdue",))
        du = rcf.get("due", {}) if isinstance(rcf, dict) else {}
        fx = du.get("fix_now", {})
        t.check("...after the first start: the window adds nothing by itself (no new prompt), the card says Ready to add; Fix it now adds the line "
                "and Windows asks again (the second prompt) (r6 #7)",
                rx == 0 and du.get("started") is False and du.get("asks") == 1 and du.get("ready_text", "").startswith("Ready to add") and
                fx.get("done") is True and fx.get("asks") == 2 and len(asks()) == 2 and fx.get("fixed") is True and read(bl4_ini) == new_file,
                (du, asks()))
        rc = t.run("set", exe, "rr_config_fix=off")
        shutil.rmtree(os.path.join(docs, "My Games"))  # before the game's first start again, for the checks below
        if os.path.exists(ask_log):
            os.remove(ask_log)
        t.check("...off takes that line out again; the settings folder gone again for the checks below", rc == 0 and "rrconfig\t" not in record_text(b) and
                not os.listdir(docs), (rc, msgs(), os.listdir(docs)))
        rc = t.run("set", exe, "rr_config_fix=on")
        t.check("protected folder: set on before the game's first start waits as before (nothing created, no record line) (r2c)",
                rc == 0 and waits() and not os.listdir(docs) and "rrconfig\t" not in record_text(b), (rc, msgs(), os.listdir(docs)))
        rx, pg, rcf = rr_page("protected-waiting", prot_env)
        # r4 #9: in a protected folder the question says Fix it for me brings Windows' permission prompt; r6 #7: before the game's
        # first start (here) it says when Windows asks: at Fix it for me (to save the choice), and again at Fix it now after that start
        t.check("protected folder before the game's first start: the Engine.ini question says Windows asks when Fix it for me is clicked and again at "
                "Fix it now after the game's first start (r4 #9, r6 #7)",
                rx == 0 and "Windows asks for permission twice: when you click Fix it for me, and when you click Fix it now on the game's page after that "
                "start." in rcf.get("question_text", "") and
                "asks for permission first" not in rcf.get("question_text", ""),
                rcf.get("question_text"))
        # r3 #4: it waits for the game's first start and then for the player's click, not for itself: the title and Check
        # again's toast say so (no "will be added")
        t.check("protected folder, waiting: the card says it waits for the game's first start, then Fix it now adds it (Windows asks first); "
                "Check again's toast says to click Fix it now; no prompt raised (r2c, r3 #4)",
                rx == 0 and row(pg, "Waiting for the game's first start. Borderlands 4 makes its settings folder at its first start. "
                                    "Then click Fix it now here to add the line. Windows asks for permission first, because Borderlands 4 is installed in a protected folder.") and
                not row(pg, "Will be added") and "then click Fix it now here to add the line" in rcf.get("check_toast", "") and
                "WrongGPU adds the line" not in rcf.get("check_toast", "") and
                rcf.get("due", {}).get("protected") is True and rcf.get("due", {}).get("due") is False and not asks() and not os.listdir(docs),
                (rx, rcf, [k for k in pg if "ngine" in k or "added" in k], asks()))
        write(bl4_gus, b"[ScalabilityGroups]\r\n")  # the game's first start
        rx, pg, rcf = rr_page("protected-open", prot_env)
        # r4 #9 / r5 #9: the settings folder there now: Fix it for me itself brings the prompt, and the question says so
        t.check("protected folder after the game's first start: the Engine.ini question says Windows asks for permission first (r4 #9, r5 #9)",
                rx == 0 and "(Windows asks for permission first, because Borderlands 4 is installed in a protected folder)" in rcf.get("question_text", ""),
                rcf.get("question_text"))
        du = rcf.get("due", {}) if isinstance(rcf, dict) else {}
        t.check("protected folder, the folder there: the launcher and the page open, Check again: nothing written, no prompt; the card says Ready to add with Fix it now (r2c)",
                rx == 0 and du.get("due") is True and du.get("entry_due") is True and du.get("protected") is True and
                row(pg, "Ready to add. Borderlands 4 has made its settings folder. Click Fix it now to add the line. Windows asks for permission first") and
                pg.get("Fix Engine.ini now", {}).get("enabled") is True and "Fix Engine.ini again" not in pg and
                "Click Fix it now to add the line" in rcf.get("check_toast", "") and
                not asks() and not os.path.exists(bl4_ini) and "rrconfig\t" not in record_text(b), (rx, rcf, [k for k in pg if "ngine" in k or "add" in k], asks()))
        # r3 #10: the player's own line in a file that isn't read-only (WrongGPU never added it: no record line): "Fix it now",
        # never "Fix it again"
        write(bl4_ini, new_file)
        rx, pg, rcf = rr_page("protected-handline", prot_env)
        t.check("the player's own line, not read-only, WrongGPU's line never added: the row says Fix it now, not Fix it again (r3 #10)",
                rx == 0 and rcf.get("state") == 2 and row(pg, "Not read-only. Borderlands 4 will remove it at its next start. Fix it now makes the file read-only") and
                "Fix Engine.ini now" in pg and "Fix Engine.ini again" not in pg and not asks(), (rx, rcf, [k for k in pg if "ngine" in k or "read-only" in k]))
        os.remove(bl4_ini)
        # r3 #5: the Library shows it too ("Needs attention", Fix it now), and the launcher opening says it once in a toast
        t.check("(RR on in the game: its RR data as if converted) (r6 #5)", rr_data_ready(), rr_ini(b))
        rx, pg, rcf = rr_page("protected-library", prot_env, extra=("--apply=rrdue-library",))
        du = rcf.get("due", {}) if isinstance(rcf, dict) else {}
        lb = du.get("library", {})
        t.check("protected folder, the folder there, the Library: the game is in Needs attention with Fix it now; the launcher opening says once that "
                "its Ray Reconstruction setting is ready to add; nothing written, no prompt (r3 #5)",
                rx == 0 and lb.get("attention", 0) >= 1 and lb.get("pill", "").startswith("Fix Engine.ini now in ") and "ready to add" in lb.get("tile", "") and
                du.get("started") is False and "Ray Reconstruction setting is ready to add" in du.get("toast", "") and "Fix it now" in du.get("toast", "") and
                not asks() and not os.path.exists(bl4_ini) and "rrconfig\t" not in record_text(b), (rx, du, asks()))
        # r4 #8: an Engine.ini line ready to add is routine (Fix it now on the accent's tint): it never makes the tab's count red
        t.check("...that game alone doesn't make the Library tab's count red (r4 #8)", lb.get("ready") is True and lb.get("red") is False, lb)
        t.check("...the shelf's caption counts it and says what Fix it now does, not Adopt's sentence (r4 #8)",
                "has its Ray Reconstruction setting ready to add" in lb.get("caption", "") and "Adopt keeps" not in lb.get("caption", ""), lb.get("caption"))
        # r5 #5: the line is Ray Reconstruction's, so RR's own switch decides, not Use WrongGPU (the upscaler). Each set without the
        # Documents test root, so the CLI's own apply can't add the line meanwhile
        no_docs = {k: v for k, v in t.env.items() if k != "WRONGGPU_TEST_DOCUMENTS"}
        keep_env = t.env
        def set_no_docs(*kv):
            t.env = no_docs
            try:
                return t.run("set", exe, *kv)
            finally:
                t.env = keep_env
        # the upscaler turned off (Use WrongGPU off: AMD FSR upscales), RR on: RR still runs, so its line is listed
        rc = set_no_docs("wronggpu=off")
        t.check("(RR on in the game again: set wrote its settings without RR data) (r6 #5)", rr_data_ready(), rr_ini(b))
        rx, pg, rcf = rr_page("protected-library-off", prot_env, extra=("--apply=rrdue-library",))
        du = rcf.get("due", {}) if isinstance(rcf, dict) else {}
        lb = du.get("library", {})
        t.check("upscaler turned off, RR on: its line ready to add is listed in the Library: ready, the Fix it now pill, the line on its tile, the "
                "toast; nothing written, no prompt (r5 #5)",
                rc == 0 and rx == 0 and du.get("due") is True and lb.get("ready") is True and lb.get("attention", 0) >= 1 and
                lb.get("pill", "").startswith("Fix Engine.ini now in ") and "ready to add" in lb.get("tile", "") and
                "Ray Reconstruction setting is ready to add" in du.get("toast", "") and not asks() and not os.path.exists(bl4_ini),
                (rc, rx, du, asks()))
        # RR turned off, the upscaler on: nothing needs the player's attention there
        rc = set_no_docs("wronggpu=on")
        rc2 = set_no_docs("rr_enabled=off")
        rx, pg, rcf = rr_page("protected-library-rroff", prot_env, extra=("--apply=rrdue-library",))
        du = rcf.get("due", {}) if isinstance(rcf, dict) else {}
        lb = du.get("library", {})
        t.check("RR turned off, the upscaler on: its line ready to add is left out of the Library: not ready, no Fix it now pill, no line on its tile, "
                "no toast; nothing written, no prompt (r5 #5)",
                rc == 0 and rc2 == 0 and rx == 0 and du.get("due") is True and lb.get("ready") is False and
                not lb.get("pill", "").startswith("Fix Engine.ini now") and "ready to add" not in lb.get("tile", "") and
                "ready to add" not in du.get("toast", "") and not asks() and not os.path.exists(bl4_ini),
                (rc, rc2, rx, du, asks()))
        rc = set_no_docs("rr_enabled=on")
        t.check("...RR on again (no line written meanwhile)", rc == 0 and not os.path.exists(bl4_ini) and "rrconfig\t" not in record_text(b), (rc, msgs()))
        rx, pg, rcf = rr_page("protected-due", prot_env, extra=("--apply=rrdue",))
        du = rcf.get("due", {}) if isinstance(rcf, dict) else {}
        gd, fx = du.get("guard", {}), du.get("fix_now", {})
        t.check("protected folder, the window's own apply (as when it opens or shows the page): not started, no prompt, Ready to add with Fix it now (r2c)",
                rx == 0 and du.get("due") is True and du.get("protected") is True and du.get("started") is False and du.get("again") is False and
                du.get("asks") == 0 and du.get("ready_text", "").startswith("Ready to add. Borderlands 4 has made its settings folder") and
                du.get("fix_button") == "Fix Engine.ini now" and du.get("fixed_after") is False and du.get("toast") == "", du)
        t.check("...the worker's own check (the entry's flag cleared as if stale): the Apply starts, runs nothing, raises no prompt, no toast or note; Ready to add again (r2c)",
                gd.get("started") is True and gd.get("done") is True and gd.get("asks") == 0 and gd.get("protected_after") is True and gd.get("fixed") is False and
                gd.get("toast") == "" and gd.get("notice") == "" and gd.get("saved") == "on" and gd.get("ready_text", "").startswith("Ready to add") and
                gd.get("fix_button") == "Fix Engine.ini now", gd)
        rec = record_text(b)
        t.check("...Fix it now, the player's click: one permission prompt (the elevated worker's job), the line added (45-byte file, read-only), recorded, the toast says so (r2c)",
                fx.get("done") is True and fx.get("asks") == 1 and len(asks()) == 1 and "apply" in asks()[0] and "--job=" in asks()[0] and
                fx.get("state") == 1 and fx.get("fixed") is True and "has the Ray Reconstruction setting now" in fx.get("toast", "") and
                read(bl4_ini) == new_file and read_only(bl4_ini) and
                re.search(r"(?m)^rrconfig\t" + re.escape(bl4_ini) + r"\tcr\tabsent\t-\r?$", rec) is not None,
                (fx, asks(), [l for l in rec.splitlines() if l.startswith("rrconfig")]))
        rc = t.run("set", exe, "rr_config_fix=off")
        t.check("...off takes the line added that way out again (the file it created gone, the game's settings folder stays, no record line) (r2c)",
                rc == 0 and not os.path.exists(bl4_ini) and os.listdir(os.path.dirname(bl4_ini)) == ["GameUserSettings.ini"] and
                "rrconfig\t" not in record_text(b), (rc, msgs(), os.listdir(os.path.dirname(bl4_ini))))
        # r3 #6: the question answered with Show me how: in the protected folder the steps show at once (Myself on the page,
        # unsaved: Apply saves it, Windows asks then), with no permission prompt and no failure note; where no permission is
        # needed the choice is saved as well
        # r4 #10: the write test in the game's folder (is it protected?) runs only where WrongGPU may add the line (on, or ask: the
        # question); with Myself chosen the scan never tests the folder, so the entry isn't marked protected
        rx, pg, rcf = rr_page("protected-myself", prot_env)
        t.check("protected folder, Myself chosen: no write test for it at the scan, so the entry isn't marked protected (r4 #10)",
                rx == 0 and get_fix(exe)[1].get("profile") == "off" and rcf.get("due", {}).get("protected") is False, (rx, rcf.get("due")))
        rc = t.run("set", exe, "rr_config_fix=ask")
        rx, pg, rcf = rr_page("protected-ask", prot_env)
        t.check("...not decided yet (ask: the question can offer Fix it for me): tested, protected (r4 #10)",
                rc == 0 and rx == 0 and rcf.get("due", {}).get("protected") is True, (rx, rcf.get("due")))
        n0 = len(asks())  # (Fix it now's one prompt above)
        rx, pg, rcf = rr_page("protected-showme", prot_env, extra=("--apply=rrshowme",))
        sa = rcf.get("show_me_how_answer", {}) if isinstance(rcf, dict) else {}
        t.check("protected folder, Show me how: the steps show at once (Myself on the page, unsaved), the keyboard on Open folder; nothing run, "
                "no permission prompt, no note; still ask in the profile (r3 #6)",
                rc == 0 and rx == 0 and sa.get("busy_started") is False and sa.get("edit_off_at_once") is True and sa.get("steps") is True and
                sa.get("edit") == "off" and sa.get("saved") == "ask" and sa.get("dirty") is True and sa.get("focus_open") is True and
                "steps" in sa.get("toast", "") and sa.get("notice") == "" and sa.get("asks") == n0 and len(asks()) == n0 and
                get_fix(exe)[1].get("profile") == "ask", (rc, rx, sa, asks()))
        # r4 #4: the choice saved alone (rr_config_fix into profile.ini), never a full Apply of the profile captured when the question
        # was asked: a value changed outside the launcher since (wronggpu.ini's pre_exposure) stays, every other file byte for byte
        outside_pre_exposure(b)
        prof = os.path.join(b, "wronggpu-setup", "profile.ini")
        prof_before, wini_before = read(prof), read(os.path.join(b, "wronggpu.ini"))
        game_before = snapshot(root)
        rx, pg, rcf = rr_page("showme", dict(t.env, WRONGGPU_TEST_ELEVATION_LOG=ask_log), extra=("--apply=rrshowme",))
        sa = rcf.get("show_me_how_answer", {}) if isinstance(rcf, dict) else {}
        t.check("no permission needed, Show me how: the steps show at once and the choice is saved (Myself, nothing unsaved), no prompt (r3 #6)",
                rx == 0 and sa.get("busy_started") is True and sa.get("edit_off_at_once") is True and sa.get("steps") is True and
                sa.get("saved") == "off" and sa.get("dirty") is False and sa.get("asks") == n0 and len(asks()) == n0 and
                get_fix(exe)[1].get("profile") == "off", (rx, sa, asks()))
        ch = changed_keys(game_before, snapshot(root))
        pdiff = sorted(set(read(prof).decode("utf-8", "replace").splitlines()) ^ set(prof_before.decode("utf-8", "replace").splitlines()))
        t.check("...Show me how saves rr_config_fix alone: only profile.ini changed in the game folder, and only its rr_config_fix line; wronggpu.ini "
                "byte for byte with the outside pre_exposure; the toast says it is saved (r4 #4)",
                ch == [k for k in ch if k.endswith("wronggpu-setup/profile.ini")] and len(ch) == 1 and
                all(l.strip().startswith("rr_config_fix") for l in pdiff) and len(pdiff) == 2 and read(os.path.join(b, "wronggpu.ini")) == wini_before and
                "pre_exposure=original" in (options_line(ini(b)) or "") and "Saved: you add" in sa.get("toast", ""), (ch, pdiff, sa.get("toast")))
        # r4 #4: Fix it for me, answered from a profile captured before a change (its sparkle_guard differs from the saved one): the
        # line added, rr_config_fix=on saved alone; the saved sparkle_guard, the outside pre_exposure and every other file stay
        rc = t.run("set", exe, "rr_config_fix=ask")
        outside_pre_exposure(b)  # (set wrote wronggpu.ini again: the outside change made again, after it)
        prof_before, wini_before = read(prof), read(os.path.join(b, "wronggpu.ini"))
        game_before = snapshot(root)
        rx, pg, rcf = rr_page("fixforme", dict(t.env, WRONGGPU_TEST_ELEVATION_LOG=ask_log), extra=("--apply=rrfixforme",))
        ff = rcf.get("fix_for_me_answer", {}) if isinstance(rcf, dict) else {}
        ch = changed_keys(game_before, snapshot(root))
        pdiff = sorted(set(read(prof).decode("utf-8", "replace").splitlines()) ^ set(prof_before.decode("utf-8", "replace").splitlines()))
        t.check("no permission needed, Fix it for me: the line added (45-byte read-only file, recorded), on saved, no prompt; the toast says so (r4 #4)",
                rc == 0 and rx == 0 and ff.get("busy_started") is True and ff.get("done") is True and ff.get("saved") == "on" and ff.get("state") == 1 and
                ff.get("fixed") is True and "has the Ray Reconstruction setting now" in ff.get("toast", "") and ff.get("notice") == "" and
                read(bl4_ini) == new_file and read_only(bl4_ini) and len(asks()) == n0 and "rrconfig	" in record_text(b), (rc, rx, ff, asks()))
        t.check("...only rr_config_fix saved: the profile's own sparkle_guard kept (not the captured one); only wronggpu-setup/profile.ini (its rr_config_fix "
                "line alone) and install.record changed in the game folder; wronggpu.ini byte for byte with the outside pre_exposure (r4 #4)",
                ch and all(k.endswith("wronggpu-setup/profile.ini") or k.endswith("wronggpu-setup/install.record") for k in ch) and
                all(l.strip().startswith("rr_config_fix") for l in pdiff) and len(pdiff) == 2 and read(os.path.join(b, "wronggpu.ini")) == wini_before and
                "pre_exposure=original" in (options_line(ini(b)) or "") and ff.get("sparkle_guard") != ff.get("stale_sparkle_guard"), (ch, pdiff, ff))
        rc = t.run("set", exe, "rr_config_fix=off")
        t.check("...off takes that line out again (Engine.ini gone, no record line)", rc == 0 and not os.path.exists(bl4_ini) and "rrconfig	" not in record_text(b),
                (rc, msgs()))
        shutil.rmtree(os.path.join(docs, "My Games"))  # before the game's first start again, for the window's checks below
        # the game page's Engine.ini setting (UI review #2): Myself chosen, the steps with Open folder, Copy (the line) and the
        # file check (no line now: the off above took it out), nothing written by drawing them
        dump = os.path.join(t.work, "ui-dump-rrconfig.json")
        rc2 = t.run("ui-dump", "--dump=" + dump, "--noscan", "--library=" + os.path.join(t.work, "games"))
        try:
            d = json.loads(read(dump).decode("utf-8"))
        except Exception as e:  # noqa: BLE001
            d = {"error": str(e)}
        pg = [{c["text"]: c for c in gp.get("controls", [])} for gp in d.get("games", [])]
        pg = [p for p in pg if any(k.startswith("Engine.ini setting") for k in p)]
        p0 = pg[0] if pg else {}
        t.check("window: the Engine.ini setting row with Myself chosen, the steps (Open folder, the line with Copy) and the file check",
                rc2 == 0 and len(pg) == 1 and p0.get("Engine.ini setting: Myself", {}).get("checked") is True and "Copy the line" in p0 and
                any(k.startswith("Open the settings folder of") for k in p0) and "Check Engine.ini again" in p0 and
                any(k.startswith("Step 4: Paste this line under [SystemSettings]") and line.decode() in k for k in p0) and
                # every step spelled out, read-only last (UI review r2 #5)
                any(k.startswith("Step 2: Create or open Engine.ini there with Notepad") for k in p0) and
                any(k.startswith("Step 3: If the file has no [SystemSettings] line, add it at the end") for k in p0) and
                any(k.startswith("Step 5: Save the file") for k in p0) and
                any(k.startswith("Step 6: Right-click Engine.ini, choose Properties, tick Read-only and click OK") for k in p0) and
                any(k.startswith("The game's settings folder isn't there yet") for k in p0) and not os.listdir(docs), (rc2, [k for k in p0 if "ngine" in k or "Step" in k]))
        rg = [gp.get("rrconfig") for gp in d.get("games", []) if gp.get("rrconfig")]
        # the game never started (no settings folder): the check says so (-1), Open folder makes nothing and says to start the
        # game first, and a guessed folder never counts (UI review r2 #7)
        r0 = rg[0] if rg else {}
        t.check("window, Myself: no settings folder yet: the check says so (-1), Open folder makes nothing and says start the game once (UI review r2 #7)",
                r0.get("state") == -1 and "Start the game once, then come back" in r0.get("open_toast", "") and r0.get("open_made_folder") is False and
                any(k.startswith("Step 1: Start the game once, then come back") for k in p0) and not os.listdir(docs), r0)
        sm = rg[0].get("show_me_how", {}) if rg else {}
        t.check("window: Show me how opens the game's page scrolled to the Engine.ini card, the keyboard on its Open folder (UI review r2 #1)",
                sm.get("scroll", 0) > 0 and sm.get("focus_open") is True and sm.get("open_shown") is True and sm.get("pending") is False and
                "Engine.ini" in sm.get("toast", ""), sm)
        # the choice made on the page and applied (ui-dump --apply=rrconfig): WrongGPU writes the 45-byte read-only file and the
        # check says it is there; Myself takes it out again (the folders it made too), the check says no folder (UI review r2 #14)
        dump_ap = os.path.join(t.work, "ui-dump-rrconfig-apply.json")
        rc_ap = t.run("ui-dump", "--dump=" + dump_ap, "--noscan", "--library=" + os.path.join(t.work, "games"), "--apply=rrconfig")
        try:
            d_ap = json.loads(read(dump_ap).decode("utf-8"))
        except Exception as e:  # noqa: BLE001
            d_ap = {"error": str(e)}
        ap = [gp.get("rrconfig_apply") for gp in d_ap.get("games", []) if gp.get("rrconfig")]
        ap = ap[0] if ap and ap[0] else [{}, {}]
        a_on, a_off = (ap + [{}, {}])[:2]
        t.check("window: Engine.ini setting WrongGPU, then Apply before the game's first start: saved on, nothing written, waiting, Check again (r2 #14, r2b #1)",
                rc_ap == 0 and a_on.get("chose") == "on" and a_on.get("dirty_before") is True and a_on.get("applied") is True and a_on.get("saved") == "on" and
                a_on.get("state") == -1 and a_on.get("fixed") is False and a_on.get("dirty_after") is False and a_on.get("check_button") is True and
                a_on.get("fix_button") is False and "will be added after you start the game once" in a_on.get("toast", "") and not os.listdir(docs), (rc_ap, ap, os.listdir(docs)))
        t.check("window: then Myself and Apply: saved off, nothing there, the check says no folder yet (UI review r2 #14)",
                a_off.get("chose") == "off" and a_off.get("dirty_before") is True and a_off.get("applied") is True and a_off.get("saved") == "off" and
                a_off.get("state") == -1 and a_off.get("dirty_after") is False and not os.listdir(docs) and "rrconfig\t" not in record_text(b), (ap, os.listdir(docs)))
        write(bl4_gus, b"[ScalabilityGroups]\r\n")  # the game's settings folder, as its first start makes it
        rc = t.run("set", exe, "rr_config_fix=on")
        # the page with WrongGPU's choice saved: the file check's row in each state, with what puts it right (UI review r2 #6):
        # there (Check again), not read-only and gone (Fix it again), unknown (no Documents folder known: Check again)
        import stat
        on_states = {}
        on_states["there"] = rr_page("on-there")
        os.chmod(bl4_ini, stat.S_IREAD | stat.S_IWRITE)
        on_states["notro"] = rr_page("on-notro")
        write(bl4_ini, b"[SystemSettings]\r\n")
        on_states["missing"] = rr_page("on-missing")
        write(bl4_ini, bytes([0xFF, 0xFE]) + "[SystemSettings]\r\n".encode("utf-16-le"))  # r3 #8: saved as UTF-16 (not read)
        on_states["utf16"] = rr_page("on-utf16")
        write(bl4_ini, b"[SystemSettings]\r\n")  # as the missing state left it
        on_states["unknown"] = rr_page("on-unknown", {k: v for k, v in t.env.items() if k != "WRONGGPU_TEST_DOCUMENTS"})
        rx, pg, rcf = on_states["there"]
        t.check("window, WrongGPU's choice: the line there and read-only says so, with Check again (state 1)",
                rx == 0 and row(pg, "Engine.ini has the line. The file is read-only") and "Check Engine.ini again" in pg and
                "Fix Engine.ini again" not in pg and rcf.get("state") == 1 and pg.get("Engine.ini setting: WrongGPU", {}).get("checked") is True, (rx, rcf, [k for k in pg if "ngine" in k]))
        rx, pg, rcf = on_states["notro"]
        t.check("window, WrongGPU's choice: not read-only says the game will remove it, with Fix it again (state 2)",
                rx == 0 and row(pg, "Not read-only. Borderlands 4 will remove it at its next start") and "Fix Engine.ini again" in pg and
                pg["Fix Engine.ini again"].get("enabled") is True and rcf.get("state") == 2 and "isn't read-only" in rcf.get("check_toast", ""), (rx, rcf, [k for k in pg if "ngine" in k or "read-only" in k]))
        rx, pg, rcf = on_states["missing"]
        t.check("window, WrongGPU's choice: the line gone says so, with Fix it again (state 0)",
                rx == 0 and row(pg, "Not in Engine.ini now") and "Fix Engine.ini again" in pg and rcf.get("state") == 0, (rx, rcf, [k for k in pg if "ngine" in k]))
        rx, pg, rcf = on_states["utf16"]
        t.check("window, WrongGPU's choice: an Engine.ini saved as UTF-16 says Couldn't read Engine.ini with why and what to do, Check again, "
                "not Fix it again; Check again's toast says so (state 3, r3 #8)",
                rx == 0 and row(pg, "Couldn't read Engine.ini. WrongGPU couldn't read it: it is saved as UTF-16") and "Check Engine.ini again" in pg and
                "Fix Engine.ini again" not in pg and rcf.get("state") == 3 and "couldn't be read" in rcf.get("check_toast", "") and
                "UTF-16" in rcf.get("check_toast", ""), (rx, rcf, [k for k in pg if "ngine" in k]))
        rx, pg, rcf = on_states["unknown"]
        t.check("window, WrongGPU's choice: a folder it can't check says so (not Added by WrongGPU), with Check again (state -1)",
                rx == 0 and row(pg, "Couldn't check. Start the game once, then check again") and "Check Engine.ini again" in pg and
                not row(pg, "Added by WrongGPU") and rcf.get("state") == -1, (rx, rcf, [k for k in pg if "ngine" in k or "check" in k]))
        # Myself, the game's folder there: the file check in each state (UI review r2 #6, #7): no line (0), the line in a file that
        # isn't read-only (2), read-only (1); the folder made as the game makes it (GameUserSettings.ini in it)
        rc = t.run("set", exe, "rr_config_fix=off")
        my_cfg = os.path.join(docs, "My Games", "Borderlands 4", "Saved", "Config", "Windows")
        write(os.path.join(my_cfg, "GameUserSettings.ini"), b"[ScalabilityGroups]" + bytes([13, 10]))
        my_states = {"missing": rr_page("off-missing")}
        write(bl4_ini, new_file)
        my_states["notro"] = rr_page("off-notro")
        os.chmod(bl4_ini, stat.S_IREAD)
        my_states["there"] = rr_page("off-there")
        for tag, want, title in (("missing", 0, "Not in Engine.ini yet"), ("notro", 2, "The line is there, but the file isn't read-only"),
                                 ("there", 1, "Engine.ini has the line")):
            rx, pg, rcf = my_states[tag]
            t.check("window, Myself: the file check, %s (state %d), with Check again" % (tag, want),
                    rx == 0 and rcf.get("state") == want and row(pg, title) and "Check Engine.ini again" in pg and
                    pg.get("Engine.ini setting: Myself", {}).get("checked") is True and "open_toast" not in rcf, (rx, rcf, [k for k in pg if "ngine" in k]))
        os.chmod(bl4_ini, stat.S_IREAD | stat.S_IWRITE)
        # r4 #6: Myself, the player's own Engine.ini saved as UTF-16 (the game may read it fine): only WrongGPU's check can't read it,
        # and how to check it by hand; never a request to save the player's file another way
        write(bl4_ini, bytes([0xFF, 0xFE]) + ("[SystemSettings]\r\n" + line.decode() + "\r\n").encode("utf-16-le"))
        rx, pg, rcf = rr_page("off-utf16")
        cr = row(pg, "Couldn't read Engine.ini")
        t.check("window, Myself, an Engine.ini saved as UTF-16: WrongGPU's check couldn't read it, check it by hand in Notepad (the line, read-only); "
                "no Save as / Encoding step; Check again; its toast says the same (r4 #6)",
                rx == 0 and rcf.get("state") == 3 and cr and "WrongGPU's check couldn't read it: it is saved as UTF-16" in cr[0] and
                "open Engine.ini in Notepad and look for the line under [SystemSettings]" in cr[0] and "Read-only is ticked" in cr[0] and
                "Save as" not in cr[0] and "Encoding" not in cr[0] and "Check Engine.ini again" in pg and
                "WrongGPU's check couldn't read Engine.ini" in rcf.get("check_toast", "") and "by hand" in rcf.get("check_toast", ""), (rx, rcf, cr))
        os.remove(bl4_ini)  # the player's own file gone again, as before these checks (the game's settings folder stays)
        rc = t.run("set", exe, "rr_config_fix=on")  # the line and read-only back (applied again), as before the checks
        rc2 = t.run("uninstall", exe)
        t.check("on, then uninstall: the game folder as before, Engine.ini gone, the game's settings folder as the game left it", rc == 0 and rc2 == 0 and
                snapshot(root) == before and not os.path.exists(bl4_ini) and os.listdir(my_cfg) == ["GameUserSettings.ini"], (rc, rc2, diff(before, snapshot(root)), os.listdir(my_cfg)))
        shutil.rmtree(os.path.join(docs, "My Games"))
        # 2. Engine.ini of the player's (GameUserSettings.ini beside it), no [SystemSettings]: the section added at the end
        os.makedirs(os.path.dirname(bl4_ini))
        write(os.path.join(os.path.dirname(bl4_ini), "GameUserSettings.ini"), b"[ScalabilityGroups]\r\nsg.ResolutionQuality=100\r\n")
        own = b"[Core.Log]\r\nLogNet=Log\r\n"
        write(bl4_ini, own)
        cfg_before = snapshot(bl4_cfg)
        root, b, exe = game(t, "Borderlands Four", exe="Borderlands4.exe")
        rc = t.run("install", exe, "--opti=" + opti, "--noscan", "--force", "--experimental", "--rr-config-fix=on")
        t.check("install --rr-config-fix=on, Engine.ini without the section: [SystemSettings] and the line added after the player's lines, read-only",
                rc == 0 and read(bl4_ini) == own + b"\r\n" + new_file and read_only(bl4_ini) and "rr_config_question" not in t.j, (rc, read(bl4_ini), msgs()))
        t.check("...the player's file is backed up in the game's wronggpu-setup folder",
                read(os.path.join(b, "wronggpu-setup", "rr-config", "Engine.ini")) == own, os.listdir(os.path.join(b, "wronggpu-setup")))
        rc = t.run("uninstall", exe)
        t.check("uninstall: Engine.ini byte for byte the player's, not read-only, the config folder as before", rc == 0 and read(bl4_ini) == own and
                not read_only(bl4_ini) and snapshot(bl4_cfg) == cfg_before, (rc, read(bl4_ini), diff(cfg_before, snapshot(bl4_cfg))))
        # 3. [SystemSettings] there with another key: the line under it, no second section; =off puts the bytes back
        own = b"[SystemSettings]\r\nr.Other=2\r\n\r\n[Core.Log]\r\nLogNet=Log\r\n"
        write(bl4_ini, own)
        root, b, exe = game(t, "Borderlands Four", exe="Borderlands4.exe")
        t.run("install", exe, "--opti=" + opti, "--noscan", "--force", "--experimental")
        rc = t.run("set", exe, "rr_config_fix=on")
        t.check("[SystemSettings] there: the line added under it, one section only",
                rc == 0 and read(bl4_ini) == b"[SystemSettings]\r\nr.Other=2\r\n" + line + b"\r\n\r\n[Core.Log]\r\nLogNet=Log\r\n", (rc, read(bl4_ini)))
        rc = t.run("set", exe, "rr_config_fix=off")
        t.check("=off: Engine.ini byte for byte as before, not read-only", rc == 0 and read(bl4_ini) == own and not read_only(bl4_ini), (rc, read(bl4_ini)))
        # 4. the key with another value: set to the line's; off puts the old value back
        key, value = line.split(b"=")
        own = b"[SystemSettings]\r\n" + key + b"=0\r\n"
        write(bl4_ini, own)
        rc = t.run("set", exe, "rr_config_fix=on")
        t.check("the key with another value: set to " + value.decode() + " (no second line)", rc == 0 and read(bl4_ini) == new_file, (rc, read(bl4_ini)))
        rc = t.run("set", exe, "rr_config_fix=off")
        t.check("=off: the key's value before is back", rc == 0 and read(bl4_ini) == own, (rc, read(bl4_ini)))
        # 5. the line there already, the file read-only already (the player did it): nothing changes, and off leaves it so
        write(bl4_ini, new_file)
        os.chmod(bl4_ini, 0o444)
        rc = t.run("set", exe, "rr_config_fix=on")
        t.check("line there and read-only already: the file unchanged", rc == 0 and read(bl4_ini) == new_file and read_only(bl4_ini), (rc, msgs()))
        rc = t.run("set", exe, "rr_config_fix=off")
        t.check("=off on a file the player had made read-only with the line: it stays as the player had it", rc == 0 and read(bl4_ini) == new_file and
                read_only(bl4_ini), (rc, read(bl4_ini)))
        os.chmod(bl4_ini, 0o666)
        # r4 #5: saved as UTF-8 with a byte order mark (Notepad's "UTF-8 with BOM"), a [SystemSettings] right after it: the line goes
        # into that section (no second one), the BOM stays; off puts the bytes back as they were
        own = b"\xef\xbb\xbf[SystemSettings]\r\nr.Other=1\r\n"
        write(bl4_ini, own)
        rc = t.run("set", exe, "rr_config_fix=on")
        got = read(bl4_ini)
        t.check("an Engine.ini with a UTF-8 BOM: the line added into its [SystemSettings] (one section), the BOM kept (r4 #5)",
                rc == 0 and got == own + line + b"\r\n" and got.count(b"[SystemSettings]") == 1 and read_only(bl4_ini), (rc, got, msgs()))
        rc = t.run("get", exe, "rr_config_fix")
        t.check("...get: the line counts as there", rc == 0 and t.j.get("rr_config", {}).get("fixed") is True, t.j.get("rr_config"))
        rc = t.run("set", exe, "rr_config_fix=off")
        t.check("...=off: byte for byte as before, BOM and all, not read-only (r4 #5)", rc == 0 and read(bl4_ini) == own and not read_only(bl4_ini), (rc, read(bl4_ini)))
        # 6. a read-only file without the line: written all the same, and read-only again after off (as the player had it)
        own = b"[Core.Log]\r\nLogNet=Log\r\n"
        write(bl4_ini, own)
        os.chmod(bl4_ini, 0o444)
        rc = t.run("set", exe, "rr_config_fix=on")
        t.check("a read-only Engine.ini without the line: the line added, read-only", rc == 0 and read(bl4_ini) == own + b"\r\n" + new_file and read_only(bl4_ini), (rc, read(bl4_ini), msgs()))
        rc = t.run("uninstall", exe)
        t.check("uninstall: the player's bytes back, read-only as before", rc == 0 and read(bl4_ini) == own and read_only(bl4_ini), (rc, read(bl4_ini)))
        os.chmod(bl4_ini, 0o666)
        shutil.rmtree(os.path.join(docs, "My Games"))
        # 7. Hogwarts Legacy: %LOCALAPPDATA%, the folder that holds GameUserSettings.ini (WindowsNoEditor), not read-only
        hog = os.path.join(t.env["LOCALAPPDATA"], "Hogwarts Legacy", "Saved", "Config")
        hog_root = os.path.join(t.env["LOCALAPPDATA"], "Hogwarts Legacy")
        shutil.rmtree(hog_root, ignore_errors=True)
        root, b, exe = game(t, "Hogwarts Test", exe="HogwartsLegacy.exe")
        rc = t.run("install", exe, "--opti=" + opti, "--noscan", "--force", "--experimental")
        q = t.j.get("rr_config_question", {})
        t.check("Hogwarts Legacy install: the question (the game named, its menu can't turn it on; no read-only step)",
                rc == 0 and q.get("text", "").startswith("Hogwarts Legacy needs one more setting for Ray Reconstruction") and
                "needs one line in Engine.ini, because the game's own menu can't turn it on" in q.get("text", "") and "read-only" not in q.get("text", "") and "Read-only" not in q.get("steps", ""),
                (rc, t.j.get("message"), q))
        hog_ini = os.path.join(hog, "WindowsNoEditor", "Engine.ini")
        # before the game's first start: no guessed Windows folder (the game uses WindowsNoEditor): nothing made, waits (r2b #1)
        rc = t.run("set", exe, "rr_config_fix=on")
        t.check("Hogwarts rr_config_fix=on before the game's first start: nothing made in LOCALAPPDATA (no guessed Windows folder), no record line, it waits (r2b #1)",
                rc == 0 and not os.path.exists(hog_root) and "rrconfig\t" not in record_text(b) and waits(), (rc, msgs()))
        write(os.path.join(hog, "WindowsNoEditor", "GameUserSettings.ini"), b"[ScalabilityGroups]\r\n")  # the game's first start
        hog_before = snapshot(hog)
        rc = t.run("install", exe, "--opti=" + opti, "--noscan", "--force", "--experimental")
        t.check("Hogwarts repair once the game made WindowsNoEditor: the 45-byte Engine.ini beside GameUserSettings.ini, not read-only, no Windows folder (r2b #1)",
                rc == 0 and os.path.isfile(hog_ini) and read(hog_ini) == new_file and not read_only(hog_ini) and not os.path.exists(os.path.join(hog, "Windows")) and
                re.search(r"(?m)^rrconfig\t" + re.escape(hog_ini) + r"\tc\tabsent\t-\r?$", record_text(b)) is not None and "rr_config_question" not in t.j,
                (rc, msgs(), snapshot(hog)))
        rc = t.run("uninstall", exe)
        t.check("Hogwarts uninstall: Engine.ini gone, the folder as before", rc == 0 and snapshot(hog) == hog_before, (rc, diff(hog_before, snapshot(hog))))
        # an earlier build's fix before the first start: Engine.ini made in the guessed Windows folder (and the folders up to
        # Hogwarts Legacy), recorded; once the game made WindowsNoEditor, the next Apply moves the line there (r2b #1)
        shutil.rmtree(hog_root, ignore_errors=True)
        root, b, exe = game(t, "Hogwarts Test", exe="HogwartsLegacy.exe")
        t.run("install", exe, "--opti=" + opti, "--noscan", "--force", "--experimental", "--rr-config-fix=on")
        old_ini = os.path.join(hog, "Windows", "Engine.ini")
        write(old_ini, new_file)
        with open(os.path.join(b, "wronggpu-setup", "install.record"), "ab") as f:
            f.write(("rrconfig\t%s\tc\tabsent\t%s\r\n" % (old_ini, hog_root)).encode("utf-8"))
        write(os.path.join(hog, "WindowsNoEditor", "GameUserSettings.ini"), b"[ScalabilityGroups]\r\n")
        rc = t.run("set", exe, "rr_config_fix=on")
        t.check("Hogwarts, an earlier build's line in the guessed Windows folder: moved to WindowsNoEditor, the Windows folder gone, the record names the new file (r2b #1)",
                rc == 0 and read(hog_ini) == new_file and not os.path.exists(os.path.join(hog, "Windows")) and
                re.search(r"(?m)^rrconfig\t" + re.escape(hog_ini) + r"\t", record_text(b)) is not None and old_ini not in record_text(b), (rc, msgs(), snapshot(hog)))
        rc = t.run("uninstall", exe)
        t.check("...and the uninstall takes it out there: only the game's GameUserSettings.ini left", rc == 0 and snapshot(hog) == hog_before, (rc, diff(hog_before, snapshot(hog))))
        shutil.rmtree(root, ignore_errors=True)  # no fixture game left for the later tests' scans (the window's Library)
        # 7b. rc6: Hogwarts Legacy since its November 2026 update (its game-interface integration: the frame-generation plugin
        # below in the game's folder): WrongGPU's RR is not offered there (rr.cpp): no question, no line added, the page's one
        # line instead of the RR card; an install made before the update keeps its line as chosen (off and the uninstall still
        # take it out). The cases above (no such file) are the older builds, unchanged
        hog_new = "Since the game's November 2026 update, Hogwarts Legacy doesn't offer Ray Reconstruction with WrongGPU yet"
        framegen = {os.path.join("Engine", "Plugins", "Runtime", "Vendor", "GameInterface", "Binaries", "ThirdParty", "Win64",
                                 "sl.dlss_g.dll"): b"game interface DLSS Frame Generation plugin (test stand-in, never loaded)"}
        shutil.rmtree(hog_root, ignore_errors=True)
        root, b, exe = game(t, "Hogwarts Test", exe="HogwartsLegacy.exe", extra=framegen)
        write(os.path.join(hog, "WindowsNoEditor", "GameUserSettings.ini"), b"[ScalabilityGroups]\r\n")
        hog_before = snapshot(hog)
        gbefore = snapshot(root)
        rc0 = t.run("status", exe, "--noscan")
        st = t.j.get("game", {})
        rc = t.run("install", exe, "--opti=" + opti, "--noscan", "--force", "--experimental")
        t.check("updated Hogwarts (its game-interface integration): not an RR game, the install asks no Engine.ini question (rc6)",
                rc0 == 0 and st.get("rr_uses") is False and rc == 0 and "rr_config_question" not in t.j and
                not os.path.exists(os.path.join(b, "plugins", "WrongGPU-RR.asi")), (rc0, st.get("rr_uses"), rc, t.j.get("rr_config_question")))
        rc = t.run("set", exe, "rr_config_fix=on")
        t.check("updated Hogwarts: rr_config_fix=on adds no line (Engine.ini left as it is), the settings folder as before (rc6)",
                rc == 0 and not os.path.exists(hog_ini) and snapshot(hog) == hog_before and "rrconfig\t" not in record_text(b) and
                any("Engine.ini left as it is" in m for m in msgs()), (rc, msgs(), diff(hog_before, snapshot(hog))))
        dump = os.path.join(t.work, "ui-dump-hogwarts-new.json")
        rc2 = t.run("ui-dump", "--dump=" + dump, "--noscan", "--library=" + os.path.join(t.work, "games"))
        try:
            d = json.loads(read(dump).decode("utf-8"))
        except Exception as e:  # noqa: BLE001
            d = {"error": str(e)}
        hp = {}
        for gp in d.get("games", []):
            if gp.get("name") == "Hogwarts Test":
                hp = {c["text"]: c for c in gp.get("controls", [])}
        t.check("window, updated Hogwarts: the RR section is the one line, no switch, no Engine.ini setting, nothing to convert or prepare (rc6)",
                rc2 == 0 and any(k.startswith(hog_new) for k in hp) and "Use WrongGPU (Ray Reconstruction)" not in hp and
                not any(k.startswith("Engine.ini setting") for k in hp) and "Convert the Ray Reconstruction data" not in hp and
                "Prepare Ray Reconstruction" not in hp, (rc2, list(hp)[:60]))
        rc = t.run("uninstall", exe)
        t.check("updated Hogwarts uninstall: the game folder and the settings folder as before", rc == 0 and snapshot(root) == gbefore and
                snapshot(hog) == hog_before, (rc, diff(gbefore, snapshot(root))))
        # an install made before the update, with the line (the older build: no such file), then the game updated
        root, b, exe = game(t, "Hogwarts Test", exe="HogwartsLegacy.exe")
        rc = t.run("install", exe, "--opti=" + opti, "--noscan", "--force", "--experimental", "--rr-config-fix=on")
        had = os.path.isfile(hog_ini) and read(hog_ini) == new_file
        for rel, data in framegen.items():
            write(os.path.join(root, rel), data)  # the game's update
        rc1 = t.run("set", exe, "rr_config_fix=on")
        rc2 = t.run("status", exe, "--noscan")
        st = t.j.get("game", {})
        t.check("Hogwarts updated after an install with the line: the line stays as chosen, status says no RR (rc6)",
                rc == 0 and had and rc1 == 0 and read(hog_ini) == new_file and rc2 == 0 and st.get("rr_uses") is False, (rc, had, rc1, rc2, st.get("rr_uses")))
        rc = t.run("set", exe, "rr_config_fix=off")
        t.check("...off still takes it out (Engine.ini gone, no record line) (rc6)", rc == 0 and not os.path.exists(hog_ini) and "rrconfig\t" not in record_text(b),
                (rc, msgs()))
        t.run("uninstall", exe)
        shutil.rmtree(root, ignore_errors=True)
        shutil.rmtree(hog_root, ignore_errors=True)
        # 8. another game: the setting refused, no question
        root, b, exe = game(t, "Not Borderlands")
        rc = t.run("install", exe, "--opti=" + opti, "--noscan", "--force", "--experimental")
        t.check("another game: no question in the install result", rc == 0 and "rr_config_question" not in t.j, t.j.get("rr_config_question"))
        rc = t.run("set", exe, "rr_config_fix=on")
        t.check("another game: set rr_config_fix=on is refused", rc != 0 and "rr_config_fix is a setting of Borderlands4.exe and HogwartsLegacy.exe only" in
                str(t.j.get("error", "")), (rc, t.j.get("error")))
        t.run("uninstall", exe)
        shutil.rmtree(root, ignore_errors=True)
        # 9. the elevated worker's job names the player's Documents: that folder, not this process's
        docs2 = os.path.join(t.work, "documents-job")
        shutil.rmtree(docs2, ignore_errors=True)
        os.makedirs(docs2)
        root, b, exe = game(t, "Borderlands Four", exe="Borderlands4.exe")
        t.run("install", exe, "--opti=" + opti, "--noscan", "--force", "--experimental")
        job_cfg = os.path.join(docs2, "My Games", "Borderlands 4", "Saved", "Config", "Windows")
        write(os.path.join(job_cfg, "GameUserSettings.ini"), b"[ScalabilityGroups]\r\n")  # the game's settings folder there
        job, res = os.path.join(t.work, "job-rrconfig.txt"), os.path.join(t.work, "result-rrconfig.txt")
        write(job, "op=apply\r\ndir=%s\r\nlocalappdata=%s\r\ndocuments=%s\r\nprofile.rr_config_fix=on\r\n" % (b, t.env["LOCALAPPDATA"], docs2))
        rc = t.run("apply", "--job=" + job, "--sha256=" + sha256_of(job))
        job_ini = os.path.join(docs2, "My Games", "Borderlands 4", "Saved", "Config", "Windows", "Engine.ini")
        t.check("elevated job with documents=: Engine.ini in the job's Documents folder, none in this process's", rc == 0 and os.path.isfile(job_ini) and
                read(job_ini) == new_file and not os.listdir(docs), (rc, read(res) if os.path.exists(res) else b"", os.listdir(docs)))
        rc = t.run("uninstall", exe)
        t.check("...and the uninstall takes it out there (the record's path)", rc == 0 and os.listdir(job_cfg) == ["GameUserSettings.ini"], (rc, os.listdir(job_cfg)))
        # 10. a test build without a Documents test root refuses (never this PC's Documents)
        t.env.pop("WRONGGPU_TEST_DOCUMENTS")
        root, b, exe = game(t, "Borderlands Four", exe="Borderlands4.exe")
        t.run("install", exe, "--opti=" + opti, "--noscan", "--force", "--experimental")
        rc = t.run("set", exe, "rr_config_fix=on")
        t.check("no Documents test root: nothing written, the result says the setting wasn't added",
                rc == 0 and "rrconfig\t" not in record_text(b) and any("Ray Reconstruction setting not added for Borderlands 4" in w for w in t.j.get("warnings", [])),
                (rc, msgs()))
        t.run("uninstall", exe)
        shutil.rmtree(root, ignore_errors=True)
    finally:
        t.env.pop("WRONGGPU_TEST_DOCUMENTS", None)
        # this test's fixture games and test roots go, even after a failure part way (a later test scans the games folder)
        games = [os.path.join(t.work, "games", n) for n in ("Borderlands Four", "Hogwarts Test", "Not Borderlands")]
        for d in [docs, os.path.join(t.work, "documents-job"), os.path.join(t.env["LOCALAPPDATA"], "Hogwarts Legacy")] + games:
            if os.path.isdir(d):
                set_writable(d)
                shutil.rmtree(d, ignore_errors=True)


def test_edition_download(t, opti):
    """WrongGPU's edition of OptiScaler, downloaded by the launcher (optiedition.h): pinned size + SHA-256, a mismatch refused,
    the cache reused, offline falls back to the player's own download. A local test server (127.0.0.1) stands in for the
    release page; the test build never downloads without WRONGGPU_TEST_EDITION."""
    print("\n[edition] the launcher downloads WrongGPU's edition of OptiScaler: pinned hash, mismatch refused, cache, offline fallback")
    import http.server
    import threading
    import zipfile
    ed_pkg, dll = edition_package(t, "edition-src")
    srv_dir = os.path.join(t.work, "edition-server")
    shutil.rmtree(srv_dir, ignore_errors=True)
    os.makedirs(srv_dir)
    zpath = os.path.join(srv_dir, "edition.zip")
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("OptiScaler.dll", dll)
        z.writestr("OptiScaler.ini", OPTI_INI)
        z.writestr("LICENSE-OptiScaler.txt", "GPL-3.0 (test stand-in)")
    zbytes = read(zpath)
    hits = []

    class H(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *a, **k):
            super().__init__(*a, directory=srv_dir, **k)

        def do_GET(self):
            hits.append(self.path)
            return super().do_GET()

        def log_message(self, *a):
            pass

    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
    port = httpd.server_address[1]
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    cache = os.path.join(t.data, "optiscaler", "edition-test.zip")
    try:
        pin = lambda url, s=None: "%s|%d|%s|%s" % (url, len(zbytes), s or sha(zbytes), sha(dll))
        t.env["WRONGGPU_TEST_EDITION"] = pin("http://127.0.0.1:%d/edition.zip" % port)
        if os.path.exists(cache):
            os.remove(cache)
        root, b, exe = game(t, "Edition Game", validated=True)
        before = snapshot(root)
        rc = t.run("install", exe, "--noscan", "--force")
        log = read(t.log).decode("utf-8", "replace")
        proxy = [n for n in ("dxgi.dll", "winmm.dll", "version.dll") if os.path.isfile(os.path.join(b, n))]
        t.check("no OptiScaler chosen: the launcher downloads its edition (one GET), checks it (sha256), caches it in its data folder and installs it",
                rc == 0 and len(hits) == 1 and read(cache) == zbytes and "downloaded and checked" in log and proxy and read(os.path.join(b, proxy[0])) == dll and
                "(WrongGPU edition, plugin interface 5)" in log, (rc, t.j.get("message"), hits, proxy, t.logtail()))
        rc = t.run("repair", exe, "--noscan", "--force")
        log = read(t.log).decode("utf-8", "replace")
        t.check("the next install / repair reuses the cached, checked copy (no second download)",
                rc == 0 and len(hits) == 1 and "from the launcher's cache" in log, (rc, hits, t.logtail()))
        # the cached copy changed on disk: not used, downloaded again
        write(cache, zbytes[:-1] + b"X")
        rc = t.run("repair", exe, "--noscan", "--force")
        t.check("a cached copy that doesn't match is never used: downloaded again and checked", rc == 0 and len(hits) == 2 and read(cache) == zbytes, (rc, hits))
        # a download that isn't the pinned file: refused, deleted, never used; the player's own download (Settings) instead
        os.remove(cache)
        t.env["WRONGGPU_TEST_EDITION"] = pin("http://127.0.0.1:%d/edition.zip" % port, "0" * 64)
        rc = t.run("repair", exe, "--noscan", "--force")
        log = read(t.log).decode("utf-8", "replace")
        t.check("hash mismatch: refused with the plain words, nothing cached, no OptiScaler without the player's own download (exit 2)",
                rc == 2 and "download refused" in log and not os.path.exists(cache) and not os.path.exists(cache + ".part"), (rc, t.j.get("message"), t.logtail()))
        rc = t.run("set", "--global", "optiscaler=" + opti)
        rc = t.run("repair", exe, "--noscan", "--force")
        log = read(t.log).decode("utf-8", "replace")
        t.check("hash mismatch with the player's own OptiScaler download set: that one is used", rc == 0 and "mock OptiScaler" in read(os.path.join(b, proxy[0])).decode("latin-1") if proxy else False,
                (rc, t.j.get("message"), t.logtail()))
        # offline (nothing answers): the player's own download
        t.env["WRONGGPU_TEST_EDITION"] = pin("http://127.0.0.1:9/edition.zip")
        rc = t.run("repair", exe, "--noscan", "--force")
        log = read(t.log).decode("utf-8", "replace")
        t.check("offline: the player's own OptiScaler download is used (fallback), nothing cached", rc == 0 and "download failed" in log and not os.path.exists(cache),
                (rc, t.j.get("message"), t.logtail()))
        t.run("set", "--global", "optiscaler=")
        rc = t.run("repair", exe, "--noscan", "--force")
        t.check("offline without the player's own download: refused (exit 2), said in words", rc == 2 and "OptiScaler" in t.j.get("message", ""), (rc, t.j.get("message")))
        # only https:// outside 127.0.0.1 (a test build allows the local test server only)
        t.env["WRONGGPU_TEST_EDITION"] = pin("http://example.invalid/edition.zip")
        rc = t.run("repair", exe, "--noscan", "--force")
        t.check("plain http:// to another host is never used", rc == 2 and "only https://" in read(t.log).decode("utf-8", "replace"), (rc, t.logtail()))
        t.env["WRONGGPU_TEST_EDITION"] = pin("http://127.0.0.1:%d/edition.zip" % port)
        rc = t.run("uninstall", exe)
        t.check("uninstall: the game folder back byte for byte",
                rc == 0 and snapshot(root) == before, (rc, diff(before, snapshot(root))))
        shutil.rmtree(root, ignore_errors=True)
    finally:
        httpd.shutdown()
        t.env.pop("WRONGGPU_TEST_EDITION", None)


def test_ui_dump(t, opti):
    print("\n[window] built hidden: pages, controls and names (nothing shown, nothing changed)")
    root, b, exe = game(t, "Ui Game", validated=True)
    rc = t.run("install", exe, "--opti=" + opti, "--noscan", "--force")
    # a number setting (Menu scale) in the game's OptiScaler.ini: the page gets a native text field
    oini = os.path.join(b, "OptiScaler.ini")
    if os.path.isfile(oini):
        write(oini, re.sub(rb"\[Menu\](\r?\n)", rb"[Menu]\1Scale=auto\1", read(oini), count=1))  # the file mixes line ends
    dump = os.path.join(t.work, "ui-dump.json")
    rc2 = t.run("ui-dump", "--dump=" + dump, "--noscan", "--library=" + os.path.join(t.work, "games"), "--shots=" + os.path.join(t.work, "ui-shots"))
    try:
        d = json.loads(read(dump).decode("utf-8"))
    except Exception as e:  # noqa: BLE001
        d = {"error": str(e)}
    t.check("ui-dump: exit 0 and every page", rc == 0 and rc2 == 0 and all(k in d for k in ("first_run", "list", "games", "settings")), (rc, rc2, list(d)))
    texts = [c["text"] for c in d.get("list", {}).get("controls", [])]
    t.check("library: search, shelves, rescan, add game, a lockup per game", "Rescan for games" in texts and "Add game…" in texts and "Search games" in texts and
            # the shelves follow the games (no heading for an empty shelf; "Needs attention" only while something needs it)
            "With WrongGPU, 1 game" in texts and not any(x.startswith("Needs attention") or x.startswith("Available") for x in texts) and
            # a lockup's accessible name is its name and its state; its action is a button named after the game
            any(x.startswith("Cyberpunk 2077 - ") and " installed." in x for x in texts) and "Play Cyberpunk 2077" in texts, texts)
    bn = d.get("banner", {})
    t.check("library: the featured banner is 240 high in the default window, 300 in a taller one (UI review #10)",
            bn.get("default_h") == 240 and bn.get("tall_h") == 300, bn)
    # the keyboard: Continue has it once the first run's checks are done, the Library's first stop once the sheet closes;
    # Down from a lockup reaches its action (Play, Update, ...) and Up comes back; the toolbar's two tabs are one Tab stop
    kb = d.get("keyboard", {})
    t.check("keyboard: first run Continue, then the Library's first stop; a lockup's action by Down and back by Up; the tabs one stop",
            "ontinue" in kb.get("first_run_focus", "") and kb.get("after_first_run") == "Search games" and kb.get("tile", 0) > 0 and
            kb.get("down_action") is True and kb.get("up_lockup") is True and kb.get("nav_stops") == 1, kb)
    cs = d.get("clear_search", {})
    t.check("keyboard: a search without a match, its Clear search puts the focus in the search field (not on a game's Play)",
            cs.get("button") is True and cs.get("cleared") is True and cs.get("on_search") is True, cs)
    fb = d.get("featured_back", {})
    t.check("keyboard: back from the featured banner's game page, the banner's Game settings has the focus (its game has no lockup there)",
            fb.get("feat", -1) >= 0 and fb.get("focus_on_banner") is True, fb)
    # the keyboard follows the game it pressed when the operation's result moves the Library under it (made-up games): the
    # banner that takes another game after its own Repair (the focus goes to the repaired game's Play, not "Update Bravo"),
    # a lockup's Update that makes its game the banner's (the banner's Play), a lockup that moves to a shelf out of view
    pf = d.get("place_focus", {})
    bc, bb, ms = pf.get("banner_changed", {}), pf.get("became_banner", {}), pf.get("moved_shelf", {})
    t.check("keyboard: the featured banner takes another game after its own Repair; the focus stays with the repaired game",
            bc.get("feat_before") == 0 and bc.get("feat_after") == 1 and "Alpha" in bc.get("before", "") and bc.get("on_alpha") is True and
            "Alpha" in bc.get("after", "") and "Bravo" not in bc.get("after", ""), bc)
    t.check("keyboard: a lockup's Update makes its game the banner's; the banner's Play has the focus, not the next game's pill",
            bb.get("feat_before") == 0 and bb.get("feat_after") == 1 and bb.get("before") == "Update Xray" and bb.get("on_banner_action") is True and
            "Xray" in bb.get("after", ""), bb)
    t.check("keyboard: a lockup that moves to another shelf out of view after its operation: the page scrolls to it",
            ms.get("reveal") is True and ms.get("same_game") is True and ms.get("shown_after") is True and ms.get("scroll", 0) > 0, ms)
    ik = d.get("index_keys", {})
    t.check("keyboard: Settings' section index is one Tab stop; Down and Up move between its entries, not past its ends",
            ik.get("stops") == 1 and ik.get("down_next") is True and ik.get("stop_follows") is True and ik.get("up_back") is True and ik.get("up_stays") is True, ik)
    t.check("keyboard: Return on Settings' Launcher entry puts the focus on Appearance's chosen segment, as Tab does",
            ik.get("launcher_chosen_segment") is True, ik)
    frc = d.get("first_run", {}).get("controls", [])
    t.check("first run: the sheet is a dialog named after its title", any(c["class"] == "Dialog" and c["text"] == "Welcome to WrongGPU" for c in frc),
            [(c["class"], c["text"]) for c in frc][:6])
    rows = d.get("rows", [])
    t.check("game list rows: name, status and last run", any(r.startswith("Cyberpunk 2077 - Installed") for r in rows), rows)
    pages = [gp for gp in d.get("games", []) if gp.get("name") == "Cyberpunk 2077"]
    gtexts = {c["text"]: c for c in (pages[0]["controls"] if pages else [])}
    # installed, not played yet: "Turn it on in the game" leads the page, step 1 done, step 2 (choose DLSS in the game) open,
    # step 3 offers OptiScaler's FPS overlay (UI review #1)
    t.check("game page: \"Turn it on in the game\" until Model M has run: installed done, choose DLSS open, the FPS overlay offered (UI review #1)",
            "Turn it on in the game, 1 of 3 done" in gtexts and any(k.startswith("Step 1, done: Installed") for k in gtexts) and
            any(k.startswith("Step 2: Choose DLSS as the game's upscaler") for k in gtexts) and any(k.startswith("Step 3: See it running") for k in gtexts) and
            "Turn on the FPS overlay" in gtexts, [k for k in gtexts if k.startswith(("Step", "Turn"))])
    adv = pages[0].get("advanced", {}) if pages else {}
    t.check("game page: Runs on, Output filter, Pre-exposure and Sparkle guard folded under Advanced settings at their defaults; the toggle shows them (UI review #5)",
            adv.get("open_at_start") is False and adv.get("link") is True and adv.get("runs_on_at_start") is False and adv.get("open_after_toggle") is True and
            "Advanced settings (4)" not in gtexts and "Fewer settings" in gtexts, adv)
    # the mark may flip to RightGPU on a healthy game's page, but no words on the page name it (UI review, 0.6.0)
    t.check("game page: the status words say WrongGPU (RightGPU is only the mark)",
            pages and not [k for k in gtexts if "RightGPU" in k and k != "RightGPU"] and
            any(k.startswith("WrongGPU") and ("installed" in k or "Worked" in k) for k in gtexts), [k for k in gtexts if "GPU" in k])
    # the page is dumped with its "more" menu open: Repair and Uninstall are its items
    t.check("game page: actions, the Basic settings", all(k in gtexts for k in ("Repair", "Use WrongGPU (Model M)", "Runs on: Auto", "Output filter", "Pre-exposure: Scale",
                                                                              "Sparkle guard", "FPS overlay: Default")) and
            any(k.startswith("Uninstall WrongGPU") and c.get("visible") for k, c in gtexts.items()) and "Install" not in gtexts and
            gtexts.get("Use WrongGPU (Model M)", {}).get("checked") is True, list(gtexts))
    t.check("game page: Apply only while settings differ", "Apply" not in gtexts, list(gtexts))
    # scrolled down with an unsaved change: nothing under the pinned plate or the unsaved bar takes a click, no text field
    # shows there, and Tab to a control hidden under the plate scrolls it out from under it
    gp = d.get("game_pinned", {})
    frames = gp.get("frames", [])
    pinned = [f for f in frames if f["pin_h"] > 0]
    bad = [(f["scroll"], c["text"], c["y"], c["b"]) for f in frames for c in f["controls"] if c["y"] < f["pin_h"] - 1 or c["b"] > f["bar_y"] + 1]
    t.check("game page pinned, unsaved: the plate pins, the bar docks, no click goes to what is under them",
            gp.get("dirty") is True and pinned and all(f["bar_y"] < gp.get("height", 0) for f in frames) and not bad, (bad, [(f["scroll"], f["pin_h"], f["bar_y"]) for f in frames]))
    fields = [(f["scroll"], x) for f in frames for x in f["fields"]]
    badf = [(sc, x) for sc, x in fields if x["shown"] and [f for f in frames if f["scroll"] == sc and (x["y"] < f["pin_h"] - 1 or x["b"] > f["bar_y"] + 1)]]
    t.check("game page pinned, unsaved: the Menu scale text field shows only between the plate and the bar",
            any(x["shown"] for _, x in fields) and any(not x["shown"] for _, x in fields) and not badf, (badf, fields))
    tab = gp.get("tab_under_plate", {})
    t.check("game page pinned: Tab to a control under the plate scrolls it into view below the plate",
            tab.get("id", 0) > 0 and tab.get("shown") is True and tab.get("y", -1) >= tab.get("pin_h", 0) - 1, tab)
    rcol = gp.get("tab_right_column", {})
    t.check("game page pinned: Tab into the right column (it stops under the header) brings the control out from under the header",
            rcol.get("found") is True and rcol.get("shown") is True and rcol.get("y", -1) >= rcol.get("pin_h", 0) - 1, rcol)
    cf = gp.get("capsule_focus", {})
    t.check("game page: Discard from the keyboard closes the capsule and the focus goes back to the control that changed the setting",
            cf.get("dirty_before") is True and cf.get("dirty_after") is False and cf.get("focus") == cf.get("want"), cf)
    ks = gp.get("keys", {})
    t.check("keyboard: Space acts as it comes up on the control it went down on (Tab away cancels); a held arrow steps a segmented control once",
            ks.get("space_down_waits") is True and ks.get("space_up_acts") is True and ks.get("space_moved_cancels") is True and
            ks.get("segment", 0) > 0 and ks.get("held_arrow_once") is True and ks.get("arrow_steps") is True, ks)
    t.check("keyboard: Apply refusing an OptiScaler value puts the focus on that value's field (shown), nothing written",
            ks.get("opti_row", -1) >= 0 and ks.get("apply_refused") is True and ks.get("bad_field_focus") is True, ks)
    t.check("keyboard: a menu takes the focus to its control; a lone Alt opens no system-menu mode, Alt+Space still does; Browser Back goes back",
            ks.get("menu_focus") is True and ks.get("lone_alt") is True and ks.get("alt_space") is True and ks.get("browser_back") is True, ks)
    # a run where the game asked for no upscaler, on both routes (made up in the ui-dump: the direct route isn't in this build):
    # OptiScaler's route says DLSS, the direct one AMD FSR, as the checklist's step 2 says (UI review r2 #3)
    ps = d.get("page_states", {})
    nh, nd = ps.get("notused_hosted", {}), ps.get("notused_direct", {})
    t.check("game page: a not-used run says what to choose by route: DLSS through OptiScaler, AMD FSR on the direct route (UI review r2 #3)",
            "choose DLSS" in nh.get("warning", "") and "choose AMD FSR" in nd.get("warning", "") and "DLSS" not in nd.get("warning", "") and
            "Choose FSR" in nd.get("step2", "") and "Choose DLSS" in nh.get("step2", ""), (nh, nd))
    # the Upscaler's Use WrongGPU says what off does on the game's route: OptiScaler hands upscaling back to AMD FSR; on the
    # direct route (no OptiScaler) WrongGPU passes it to the game's own AMD FSR, and OptiScaler isn't named (UI review r2b #3)
    t.check("game page: Use WrongGPU's sentence by route: OptiScaler hands upscaling back to AMD FSR; the direct route names no OptiScaler (UI review r2b #3)",
            "Off: OptiScaler hands upscaling back to AMD FSR" in nh.get("use_wronggpu", "") and
            "Off: WrongGPU passes upscaling to the game's own AMD FSR" in nd.get("use_wronggpu", "") and "OptiScaler" not in nd.get("use_wronggpu", "x OptiScaler"),
            (nh.get("use_wronggpu"), nd.get("use_wronggpu")))
    # the Upscaler's Advanced rows open by themselves for a value off its default, an unsaved one and one changed outside the
    # launcher, and stay folded at the defaults (UI review #5, r2 #14)
    av = ps.get("advanced", {})
    t.check("game page: Advanced settings open by themselves for a saved value off its default, an unsaved change and an outside edit (UI review r2 #14)",
            av.get("defaults") is False and av.get("saved_off_default") is True and av.get("unsaved") is True and av.get("outside") is True, av)
    # the checklist goes once Model M has upscaled a frame, not when the runtime is only ready (UI review r2 #10)
    gd = ps.get("guide", {})
    t.check("game page: the checklist stays after a run where the runtime was only ready, and goes after one with an upscaled frame (UI review r2 #10)",
            gd.get("ready_only", {}).get("shown") is True and gd.get("ready_only", {}).get("heading") is True and
            gd.get("upscaled", {}).get("parsed_upscaled") is True and gd.get("upscaled", {}).get("shown") is False and
            gd.get("upscaled", {}).get("heading") is False, gd)
    # the header says "Worked last time you played" only after the same test, a frame upscaled; a run that ended at "ready"
    # gets the plain installed line, so it never says worked beside a checklist that says not yet (UI review r2b #2)
    ro, up = gd.get("ready_only", {}), gd.get("upscaled", {})
    t.check("game page header: no \"Worked last time you played\" after a run where the runtime was only ready (WrongGPU <version> is installed), "
            "\"Worked\" after one with an upscaled frame (UI review r2b #2)",
            ro.get("parsed_ready") is True and ro.get("parsed_upscaled") is False and ro.get("header_worked") == "" and
            ro.get("header_installed", "").startswith("WrongGPU ") and "Worked" not in ro.get("header_installed", "") and
            up.get("header_worked", "").startswith("WrongGPU") and up.get("header_installed") == "", gd)
    # r3 #3: Last run after a run that only started says so under Model M; step 3 follows the OptiScaler (WrongGPU's edition
    # names WrongGPU; another shows AMD FSR and points at Last run)
    t.check("game page: Last run after a run where the runtime only started says \"Started; no upscaled frame recorded in the log\", that the log records "
            "only the first frames, so it can't confirm the frames after them; never that a right-looking game means Model M ran; not after an "
            "upscaled frame (r3 #3, r4 #2, r5 #6)",
            "The log records only the first few frames" in ro.get("last_run_sub", "") and "can't confirm whether Model M upscaled" in ro.get("last_run_sub", "") and
            "looked right" not in ro.get("last_run_sub", "") and
            up.get("last_run_sub") == "", (ro, up))
    # r4 #1: a run of Ray Reconstruction alone (the RR games: no front-end log) counts as a run: no "Not played yet", the
    # Denoiser row with its cost, the size from RR's timing, no upscaler row, the checklist gone, the header's Worked
    ro4 = ps.get("rr_only", {})
    t.check("game page: an RR-only run counts as a run: Denoiser row (Ray Reconstruction, ms a frame) instead of Not played yet, its size, "
            "no Upscaler row, checklist gone, header says Worked (r4 #1)",
            ro4.get("counted") is True and ro4.get("rr_only") is True and ro4.get("upscaled") is True and ro4.get("frames") == 36044 and
            ro4.get("not_played") == "" and ro4.get("denoiser", "").startswith("Denoiser: Ray Reconstruction") and "1.6 ms a frame" in ro4.get("denoiser", "") and
            ro4.get("size", "").startswith("Size: 642") and ro4.get("upscaler_row") == "" and ro4.get("guide") is False and ro4.get("step2") == "" and
            ro4.get("header_worked", "").startswith("WrongGPU") and ro4.get("right_gpu") is True, ro4)
    t.check("game page: after an RR-only run (RR on) the header's second line says Ray Reconstruction, not Model M and Ray Reconstruction (r5 #4)",
            ro4.get("header_worked", "").endswith("Worked last time you played. Ray Reconstruction.") and "Model M" not in ro4.get("header_worked", ""), ro4)
    t.check("library: the featured banner after a run of Ray Reconstruction alone says WrongGPU is on, Ray Reconstruction, as the page header "
            "does, not Model M and Ray Reconstruction (r6 #6)",
            "WrongGPU is on. Ray Reconstruction." in ro4.get("banner", "") and "Model M" not in ro4.get("banner", "x"), ro4.get("banner"))
    # r5 #3: one game session in both logs: RR's frames are never Model M's evidence (upscaled stays false). r6 #1: where Model M
    # upscaled none, RR did the upscaling (it denoises and upscales in one pass): the Upscaler row names it, no "wasn't used" warning
    # or "choose DLSS" advice, no late-start caveat; the run counts as working (no checklist, the header's Worked, Ray Reconstruction)
    mu, ml = ps.get("rr_mixed_unused", {}), ps.get("rr_mixed_late", {})
    for case, m in (("A: Model M asked for nothing", mu), ("B: a late start, ready, no frame of Model M's own", ml)):
        t.check("game page, one session with RR frames and no Model M frame (%s): Model M's evidence unchanged (not upscaled), RR did the "
                "upscaling: the Upscaler row says Ray Reconstruction upscales while it's on; no \"wasn't used\", no \"choose DLSS\", no late-start "
                "caveat, no \"didn't start\"; the Denoiser row has RR; no checklist; the header says Worked and Ray Reconstruction (r6 #1)" % case,
                m.get("counted") is True and m.get("rr_only") is False and m.get("upscaled") is False and m.get("rr_upscaled") is True and
                m.get("upscaler_row", "").startswith("Upscaler: Ray Reconstruction") and "upscales in one pass while it's on" in m.get("upscaler_row", "") and
                m.get("not_used") == "" and m.get("choose_advice") == "" and m.get("not_asked") == "" and m.get("late_sub") == "" and
                m.get("cant_confirm") == "" and m.get("not_started") == "" and m.get("denoiser", "").startswith("Denoiser: Ray Reconstruction") and
                m.get("guide") is False and m.get("header_worked", "").endswith("Worked last time you played. Ray Reconstruction.") and
                "Model M" not in m.get("header_worked", "") and m.get("log_row", "").startswith("Log: No warnings") and
                m.get("summary", "").startswith("Ray Reconstruction (3.6 ms a frame)"), m)
    t.check("...case A keeps the front end's own evidence: not created (r5 #3); case B: created", mu.get("created") is False and ml.get("created") is True, (mu, ml))
    # r7 #6: the Library tile's words for UI Automation follow the page: RR did the upscaling, so "with Ray Reconstruction", never "on Direct3D 12"
    for case, m in (("A", mu), ("B: ready, a late start", ml)):
        t.check("library tile, one session where RR did the upscaling (%s): its UI Automation text says Last run with Ray Reconstruction, not on "
                "Direct3D 12 (r7 #6)" % case,
                "with Ray Reconstruction." in m.get("tile", "") and "Direct3D" not in m.get("tile", "x"), m.get("tile"))
    # r6 #2: an RR attempt without a successful frame: a run that says so with the log's reason, never "Not played yet"; no Worked,
    # the checklist stays. In one session beside a Model M that wasn't used, the warning names Model M and gives no DLSS advice (r6 #1)
    fo, fm = ps.get("rr_failed_only", {}), ps.get("rr_failed_mixed", {})
    t.check("game page, an RR attempt alone without a successful frame: a run (not \"Not played yet\"), the Denoiser row says no successful frame "
            "was recorded, the log's reason and the pass-through; no Worked, the checklist stays, no RightGPU (r6 #2)",
            fo.get("counted") is True and fo.get("found") is True and fo.get("rr_only") is True and fo.get("upscaled") is False and
            fo.get("not_played") == "" and fo.get("denoiser", "").startswith("Denoiser: Ray Reconstruction") and
            "No successful frame recorded in the log." in fo.get("denoiser", "") and "The log says: runtime not ready." in fo.get("denoiser", "") and
            "passed through" in fo.get("denoiser", "") and fo.get("header_worked") == "" and fo.get("header_installed") != "" and
            fo.get("guide") is True and fo.get("right_gpu") is False and fo.get("summary", "").startswith("Ray Reconstruction: no successful frame"), fo)
    # r7 #1: there the game asked for Ray Reconstruction (it needs DLSS on already): the Upscaler row says so, no amber "wasn't used" line,
    # checklist step 2 done as on the RR-only page, the summary in RR's words
    t.check("game page, one session: Model M asked for nothing, RR attempted without a frame: not RR's upscaling; the Denoiser row qualified; "
            "the Upscaler row says Not used, Ray Reconstruction was asked for instead; no \"wasn't used\" line, no DLSS advice; step 2 done; "
            "no Worked; the summary says no successful RR frame (r6 #1, r6 #2, r7 #1)",
            fm.get("counted") is True and fm.get("rr_only") is False and fm.get("rr_upscaled") is False and fm.get("upscaled") is False and
            fm.get("upscaler_row", "").startswith("Upscaler: Not used") and "Ray Reconstruction was asked for instead" in fm.get("upscaler_row", "") and
            "No successful frame recorded" in fm.get("denoiser", "") and
            fm.get("model_m_not_used") == "" and fm.get("wronggpu_not_used") == "" and fm.get("choose_advice") == "" and
            fm.get("header_worked") == "" and fm.get("guide") is True and fm.get("step2", "").startswith("Step 2, done") and
            fm.get("summary", "").startswith("Ray Reconstruction: no successful frame"), fm)
    t.check("...the RR attempt alone: step 2 done too, as before (r7 #1)", fo.get("step2", "").startswith("Step 2, done"), fo.get("step2"))
    # r7 #2: a failed RR attempt has one amber row with the next step (with the pass-through when the log has it), never a green "No
    # warnings": alone, beside a Model M that wasn't asked, and beside a Model M that upscaled frames (Model M's run, RR failed)
    fk = ps.get("rr_failed_mixed_ok", {})
    for case, m in (("alone", fo), ("Model M not asked", fm), ("Model M upscaled", fk)):
        t.check("game page, a failed RR attempt (%s): the amber row says Ray Reconstruction didn't finish a frame, the game showed its image "
                "without it, start again and use Collect logs if it happens again; no \"No warnings\" (r7 #2)" % case,
                m.get("rr_warn", "").startswith("Ray Reconstruction didn't finish a frame in this run, so the game showed its image without it.") and
                "Start the game again; if it happens again, use Collect logs." in m.get("rr_warn", "") and "No warnings" not in m.get("log_row", ""), m)
    # r7 #6: the tile's words for a failed RR attempt: no successful RR frame (alone, beside a Model M not asked); beside Model M's own
    # frames, Model M's run on Direct3D 12 and the RR failure
    t.check("library tile, a failed RR attempt alone and beside a Model M the game didn't ask: Last run ...: no successful Ray Reconstruction "
            "frame recorded (r7 #6)",
            all(": no successful Ray Reconstruction frame recorded." in m.get("tile", "") and "Direct3D" not in m.get("tile", "x") for m in (fo, fm)),
            (fo.get("tile"), fm.get("tile")))
    t.check("library tile, a failed RR attempt beside Model M's upscaled frames: Last run ... on Direct3D 12; no successful Ray Reconstruction frame "
            "recorded (r7 #6)",
            ("on Direct3D" + chr(0xa0) + "12; no successful Ray Reconstruction frame recorded.") in fk.get("tile", ""), fk.get("tile"))
    t.check("...beside Model M's upscaled frames: Model M's run (upscaled, its Worked), RR's failure still said (r7 #2)",
            fk.get("upscaled") is True and fk.get("rr_upscaled") is False and fk.get("upscaler_row", "").startswith("Upscaler: Model M") and
            "No successful frame recorded" in fk.get("denoiser", ""), fk)
    se = ps.get("step3_edition", {})
    t.check("game page: step 3 with WrongGPU's edition of OptiScaler says DLSS -> WrongGPU Model M; with another, AMD FSR and Last run (r3 #3)",
            "DLSS → WrongGPU" in se.get("edition", "") and "AMD FSR" not in se.get("edition", "x") and
            "shows AMD FSR" in se.get("other", "") and "Last run" in se.get("other", "") and "DLSS →" not in se.get("other", "x"), se)
    t.check("game page, WrongGPU's RR on, before the first run: step 3 names the RR form too (the overlay's DLSS -> WrongGPU RR, Last run's Ray "
            "Reconstruction) beside Model M's (r6 #4)",
            se.get("rr_on") is True and "DLSS → WrongGPU Model" in se.get("edition_rr", "") and "DLSS → WrongGPU RR" in se.get("edition_rr", "") and
            "Ray Reconstruction when it did the upscaling" in se.get("edition_rr", "") and "WrongGPU RR" not in se.get("edition", "x"), se)
    t.check("game page: another OptiScaler after a late start: \"doesn't list WrongGPU by name\" is said once (Last run's foot), step 3 still says "
            "the overlay shows AMD FSR (r5 #9)",
            se.get("by_name_count") == 1 and "shows AMD FSR" in se.get("other_after_run", "") and "by name" not in se.get("other_after_run", "x") and
            "by name" in se.get("other", ""), se)
    # the checklist's FPS overlay button (UI review r2 #9): an unsaved On says it turns on with Apply, Discard puts it back,
    # saved says it is on; an OptiScaler without the overlay's key gets no promise and no button
    fp = ps.get("fps", {})
    t.check("game page: Turn on the FPS overlay makes fps_overlay=on unsaved, step 3 says it turns on when applied; Discard puts it back (UI review r2 #9)",
            fp.get("guide") is True and fp.get("button") is True and "(it is on)" not in fp.get("step3", "") and fp.get("after_press") == "on" and
            fp.get("dirty") is True and "turns on when you apply" in fp.get("step3_pending", "") and "(it is on)" not in fp.get("step3_pending", "") and
            fp.get("button_pending") is False and fp.get("after_discard") == fp.get("saved") and fp.get("dirty_after_discard") is False and
            fp.get("button_after_discard") is True, fp)
    t.check("game page: the FPS overlay saved on: step 3 says it is on, no button; without OptiScaler's overlay key: no claim, no button (UI review r2 #9)",
            "(it is on)" in fp.get("step3_saved", "") and fp.get("button_saved") is False and "FPS overlay shows" not in fp.get("step3_nokey", "x") and
            "no FPS overlay setting" in fp.get("step3_nokey", "") and fp.get("button_nokey") is False, fp)
    t.check("settings page: update check off by default", any(c["text"] == "Check for new releases" and c.get("checked") is False for c in d.get("settings", {}).get("controls", [])))
    bf = d.get("busy_focus", {})
    t.check("settings page: a button that goes busy keeps the focus, shown off with its busy name; Return does nothing, Tab goes on from it",
            bf.get("id", 0) > 0 and bf.get("kept") is True and bf.get("enabled") is False and "Converting" in bf.get("name", "") and
            bf.get("return_inert") is True and bf.get("tab_on") is True, bf)
    # a run where the game never asked for an upscaler (TAA in its settings): the front end's log has its session and settings
    # lines only; Last run says WrongGPU wasn't used and what to choose in the game, and step 2 of the checklist stays open
    # (UI review #9, #1). A dump of its own: a played game leads the Library's banner, which the checks above don't expect
    write(os.path.join(b, "wronggpu-fsr.log"), "[1] [x] ==== session start: Cyberpunk2077.exe (process 1) ====\r\n[1] [x] settings: upscaler=network\r\n")
    dump2 = os.path.join(t.work, "ui-dump-notused.json")
    rc3 = t.run("ui-dump", "--dump=" + dump2, "--noscan", "--library=" + os.path.join(t.work, "games"))
    try:
        d2 = json.loads(read(dump2).decode("utf-8"))
    except Exception as e:  # noqa: BLE001
        d2 = {"error": str(e)}
    p2 = [gp for gp in d2.get("games", []) if gp.get("name") == "Cyberpunk 2077"]
    g2 = {c["text"]: c for c in (p2[0]["controls"] if p2 else [])}
    t.check("game page: a run where the game asked for no upscaler says WrongGPU wasn't used and to choose DLSS in the game (UI review #9)",
            rc3 == 0 and any(k.startswith("Upscaler: Not used") for k in g2) and any("wasn't used in this run" in k and "choose DLSS" in k for k in g2) and
            any(k.startswith("Step 2: Choose DLSS as the game's upscaler") for k in g2), [k for k in g2 if "used" in k or k.startswith("Step")])
    # ...and applied for real (ui-dump --apply: the button, then the capsule's Apply): ShowFps=true in the game's OptiScaler.ini,
    # the page read again says it is on (UI review r2 #9, #14)
    dump3 = os.path.join(t.work, "ui-dump-fps-apply.json")
    rc4 = t.run("ui-dump", "--dump=" + dump3, "--noscan", "--library=" + os.path.join(t.work, "games"), "--apply=fps")
    try:
        d3 = json.loads(read(dump3).decode("utf-8"))
    except Exception as e:  # noqa: BLE001
        d3 = {"error": str(e)}
    fa = d3.get("page_states", {}).get("fps", {})
    opt_ini = read(os.path.join(b, "OptiScaler.ini")).decode("utf-8", "replace") if os.path.isfile(os.path.join(b, "OptiScaler.ini")) else ""
    t.check("game page: the FPS button then Apply writes ShowFps=true; the page read again says it is on, nothing unsaved (UI review r2 #9)",
            rc4 == 0 and fa.get("applied") is True and fa.get("saved_after_apply") == "on" and fa.get("dirty_after_apply") is False and
            "(it is on)" in fa.get("step3_applied", "") and fa.get("button_applied") is False and re.search(r"(?m)^ShowFps\s*=\s*true", opt_ini) is not None,
            (rc4, fa, re.findall(r"(?m)^ShowFps.*$", opt_ini)))
    # r6 #2: a failed RR attempt is a run that says so, read the way the launcher reads a game's folder (ReadLastRun), from the
    # synthetic fixtures tests/logs/rr-failed-only.log and rr-passthrough-only.log: alone (no front-end log), and newer than the
    # front end's successful run (tests/logs/cyberpunk-quality-fsr.log, a real run): never "Not played yet", never the older
    # success's "Worked"; the Denoiser row says no successful frame was recorded, with the log's reason
    def rr_dump(tag):
        dmp = os.path.join(t.work, "ui-dump-rr-%s.json" % tag)
        rcx = t.run("ui-dump", "--dump=" + dmp, "--noscan", "--library=" + os.path.join(t.work, "games"))
        try:
            dd = json.loads(read(dmp).decode("utf-8"))
        except Exception as e:  # noqa: BLE001
            dd = {"error": str(e)}
        pg = [gp for gp in dd.get("games", []) if gp.get("name") == "Cyberpunk 2077"]
        return rcx, [c["text"] for c in (pg[0]["controls"] if pg else [])], [r for r in dd.get("rows", []) if r.startswith("Cyberpunk 2077")], (pg[0] if pg else {})
    # (a day in November: the launcher shows file times with the bias of today, as Windows converts them)
    stamp = lambda p, h, mi: os.utime(p, (time.mktime((2026, 11, 2, h, mi, 0, 0, 0, -1)),) * 2)
    for fx, reason in (("rr-failed-only.log", "The log says: frame kernels still being prepared (first start with a cold driver cache)."),
                       ("rr-passthrough-only.log", "The log says: runtime not ready."),
                       # r7 #3: the plugin's direct route refusing RR before any instance
                       ("rr-refused-only.log", "The log says: the game's request was refused (WrongGPU's runtime or model is not available).")):
        for k in os.listdir(b):
            if k.startswith("wronggpu-") and k.endswith(".log"):
                os.remove(os.path.join(b, k))
        write(os.path.join(b, "wronggpu-rr.log"), read(os.path.join(HERE, "logs", fx)))
        rcx, ctl, rows, gp = rr_dump(fx[:-4])
        den = [x for x in ctl if x.startswith("Denoiser: Ray Reconstruction")]
        # r7 #2: the amber row's next step: the first start's kernels (the cold-cache fixture) or start again and Collect logs
        nxt = "The first start prepares its kernels. Play again;" if "kernels" in reason else "Start the game again; if it happens again, use Collect logs."
        t.check("game page from %s alone: one amber row says Ray Reconstruction didn't finish a frame, with its next step (r7 #2)" % fx,
                rcx == 0 and len([x for x in ctl if x.startswith("Ray Reconstruction didn't finish a frame in this run") and nxt in x]) == 1,
                [x for x in ctl if "didn't finish" in x])
        t.check("game page from %s alone: a run (not \"Not played yet\"), the Denoiser row says no successful frame was recorded and the log's "
                "reason; no Worked; the list's row says no successful frame (r6 #2)" % fx,
                rcx == 0 and not any("Not played yet" in x for x in ctl) and den and "No successful frame recorded in the log." in den[0] and
                reason in den[0] and not any("Worked last time you played" in x for x in ctl) and
                any("Ray Reconstruction: no successful frame" in r for r in rows) and gp.get("open_log", "").lower().endswith("wronggpu-rr.log"),
                (rcx, den, rows, gp.get("open_log"), [x for x in ctl if "Not played" in x or "Worked" in x]))
    # newer than the front end's successful run (another process, later that day)
    write(os.path.join(b, "wronggpu-fsr.log"), read(os.path.join(HERE, "logs", "cyberpunk-quality-fsr.log")))  # (process 1000, 01:00 to 01:11)
    write(os.path.join(b, "wronggpu-rr.log"), read(os.path.join(HERE, "logs", "rr-failed-only.log")))         # (process 4100, 20:00 to 20:01)
    stamp(os.path.join(b, "wronggpu-fsr.log"), 1, 12)
    stamp(os.path.join(b, "wronggpu-rr.log"), 20, 2)
    rcx, ctl, rows, gp = rr_dump("newer-failed")
    den = [x for x in ctl if x.startswith("Denoiser: Ray Reconstruction")]
    t.check("game page: a failed RR attempt newer than the front end's successful run: that attempt is Last run (its time, no successful frame), "
            "not the older success; no Worked, no Model M row (r6 #2)",
            rcx == 0 and den and "No successful frame recorded in the log." in den[0] and ("Last run: Nov" + chr(0xa0) + "2 at" + chr(0xa0) + "20:02") in ctl and
            not any("Worked last time you played" in x for x in ctl) and not any(x.startswith("Upscaler: Model M") for x in ctl),
            (rcx, den, [x for x in ctl if x.startswith("Last run") or "Worked" in x or x.startswith("Upscaler")]))
    # r7 #4: a short direct-route run that ended after its successful first evaluate (no timing or destroy line): a run that worked,
    # no "no successful frame", no amber row
    for k in os.listdir(b):
        if k.startswith("wronggpu-") and k.endswith(".log"):
            os.remove(os.path.join(b, k))
    write(os.path.join(b, "wronggpu-rr.log"), read(os.path.join(HERE, "logs", "rr-direct-short.log")))
    rcx3, ctl3, rows3, gp3 = rr_dump("direct-short")
    t.check("game page from rr-direct-short.log alone (the direct route's successful first evaluate, nothing after it): Ray Reconstruction ran, "
            "no \"No successful frame\", no amber row; the list's row says Ray Reconstruction (r7 #4)",
            rcx3 == 0 and any(x.startswith("Denoiser: Ray Reconstruction") for x in ctl3) and not any("No successful frame" in x for x in ctl3) and
            not any("didn't finish a frame" in x for x in ctl3) and any("Ray Reconstruction" in r and "no successful" not in r for r in rows3),
            (rcx3, [x for x in ctl3 if "Ray Reconstruction" in x], rows3))
    write(os.path.join(b, "wronggpu-fsr.log"), read(os.path.join(HERE, "logs", "cyberpunk-quality-fsr.log")))  # (back: the checks below)
    stamp(os.path.join(b, "wronggpu-fsr.log"), 1, 12)
    # r7 #3: the direct route's refusal newer than the front end's success: the refusal is Last run, not the older success
    write(os.path.join(b, "wronggpu-rr.log"), read(os.path.join(HERE, "logs", "rr-refused-only.log")))  # (process 5200, 19:30 to 19:35)
    stamp(os.path.join(b, "wronggpu-rr.log"), 19, 36)
    rcx2, ctl2, rows2, gp2 = rr_dump("newer-refused")
    den2 = [x for x in ctl2 if x.startswith("Denoiser: Ray Reconstruction")]
    t.check("game page: a direct-route refusal of RR newer than the front end's successful run: that attempt is Last run (its time, no successful frame, "
            "the refusal's reason), not the older success; no Worked, no Model M row (r7 #3)",
            rcx2 == 0 and den2 and "No successful frame recorded in the log." in den2[0] and "refused" in den2[0] and
            ("Last run: Nov" + chr(0xa0) + "2 at" + chr(0xa0) + "19:36") in ctl2 and not any("Worked last time you played" in x for x in ctl2) and
            not any(x.startswith("Upscaler: Model M") for x in ctl2) and any("Ray Reconstruction: no successful frame" in r for r in rows2),
            (rcx2, den2, rows2, [x for x in ctl2 if x.startswith("Last run") or "Worked" in x or x.startswith("Upscaler")]))
    write(os.path.join(b, "wronggpu-rr.log"), read(os.path.join(HERE, "logs", "rr-failed-only.log")))  # (back: the checks below)
    stamp(os.path.join(b, "wronggpu-rr.log"), 20, 2)
    # r6 #3: Open log opens the log of the run Last run shows: the RR plugin's here, not the front end's older one beside it
    t.check("game page: Open log opens the RR plugin's log when Last run shows the newer RR-only session, not the front end's older log (r6 #3)",
            os.path.normcase(gp.get("open_log", "")) == os.path.normcase(os.path.join(b, "wronggpu-rr.log")), gp.get("open_log"))
    # ...and the front end's log when the front end's run is the newer one
    stamp(os.path.join(b, "wronggpu-rr.log"), 0, 30)  # (the RR session now the day before: its times of day are later than the write)
    rcx, ctl, rows, gp = rr_dump("front-newer")
    t.check("game page: Open log opens the front end's log when Last run shows the front end's newer run (r6 #3)",
            rcx == 0 and os.path.normcase(gp.get("open_log", "")) == os.path.normcase(os.path.join(b, "wronggpu-fsr.log")) and
            any(x.startswith("Upscaler: Model M") for x in ctl), (rcx, gp.get("open_log"), [x for x in ctl if x.startswith("Upscaler")]))
    for k in ("wronggpu-fsr.log", "wronggpu-rr.log"):
        os.remove(os.path.join(b, k))
    t.run("uninstall", exe)
    shutil.rmtree(root, ignore_errors=True)



def test_data_root_junctions(parent):
    """A junction at the data root must be unlinked without modifying anything in its target."""
    work = os.path.join(parent.work, "junction-regression")
    os.makedirs(work, exist_ok=True)
    t = T(parent.exe, work, parent.model, None)
    shared = os.path.join(t.env["LOCALAPPDATA"], "WrongGPU")
    target = os.path.join(work, "junction-target")
    os.makedirs(target, exist_ok=True)
    write(os.path.join(target, "canary.txt"), "canary")
    write(os.path.join(target, "launcher.log"), "do not change")
    write(os.path.join(target, "managed-games.txt"), os.path.join(work, "missing-game") + "\n")
    before = snapshot(target)
    program = os.path.join(work, "installed")
    rc = t.run("self-install", "--dir=" + program, data=False)
    if os.path.isdir(shared):
        os.replace(shared, os.path.join(work, "parked-data"))
    def link():
        return subprocess.run(["cmd.exe", "/d", "/c", "mklink", "/J", shared, target],
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE).returncode == 0
    def invoke_without_log(exe, verb):
        out = os.path.join(work, verb + "-result.json")
        proc = subprocess.Popen([exe, verb, "--out=" + out], env=t.env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        proc.communicate(timeout=180)
        log = os.path.join(t.env.get("TEMP", ""), "WrongGPU-junction-%d.log" % proc.pid)
        text = read(log).decode("utf-8", "replace") if os.path.isfile(log) else ""
        if os.path.isfile(log): os.remove(log)
        return proc.returncode, text
    linked = link()
    rc2, log = invoke_without_log(os.path.join(program, "WrongGPU.exe"), "self-uninstall") if linked else (-1, "")
    parent.check("self-uninstall: data-root junction unlinked, target unchanged, and link removal logged",
                 rc == 0 and linked and rc2 == 0 and not os.path.lexists(shared) and snapshot(target) == before and "target was kept" in log,
                 (rc, linked, rc2, log[-500:], diff(before, snapshot(target))))
    if os.path.lexists(shared): os.rmdir(shared)
    linked = link()
    rc3, log = invoke_without_log(t.exe, "remove-data") if linked else (-1, "")
    parent.check("remove-data: data-root junction unlinked, target unchanged, and link removal logged",
                 linked and rc3 == 0 and not os.path.lexists(shared) and snapshot(target) == before and "target was kept" in log,
                 (linked, rc3, log[-500:], diff(before, snapshot(target))))
    if os.path.lexists(shared): os.rmdir(shared)
    shutil.rmtree(work, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--launcher", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--baseline", help="the 0.5.1 installer (WRONGGPU_TESTING build, same test map) for the adoption cases")
    ap.add_argument("--hip", help="mock amdhip64_7.dll (WRONGGPU_HIP_RUNTIME)")
    ap.add_argument("--only", nargs="*", default=[], help="development: only these tests (names without test_) after the basics")
    ap.add_argument("--synth-f", default=os.environ.get("WGC_SYNTH_BINS", ""),
                    help="folder with the spec team's synthetic preset-F file synth-common-v0.bin (the test build's F map is synth-r0)")
    a = ap.parse_args()
    work = os.path.abspath(a.work)
    for d in ("games", "data", "localappdata", "programs", "startmenu"):
        shutil.rmtree(os.path.join(work, d), ignore_errors=True)
    os.makedirs(work, exist_ok=True)
    model = suite.cases.f_mini_case(present=0)  # the synthetic source of the test map the launcher was built with
    t = T(os.path.abspath(a.launcher), work, model, a.hip)
    # registry entries of an earlier run (test root only)
    for k, v in t.reg_keys().items():
        if (k.startswith("WrongGPU-") and v.get("InstallLocation", "").lower().startswith(work.lower())) or (k == "WrongGPU" and work.lower() in v.get("InstallLocation", "").lower()):
            subprocess.run(["reg", "delete", t.uninstall + "\\" + k, "/f"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    opti = opti_package(t)
    t0 = time.time()
    test_basics(t)
    if a.only:  # development: only the tests named (after the basics)
        for name in a.only:
            fn = globals()["test_" + name]
            fn(t, opti) if name != "rr" else fn(t, opti, a.synth_f)
        print("\n%d checks passed, %d failed (%.0f s)" % (t.passed, len(t.failures), time.time() - t0))
        for f in t.failures:
            print("  FAILED:", f)
        return 1 if t.failures else 0
    test_scan_and_refusals(t, opti)
    test_install_settings_uninstall(t, opti)
    test_slot(t)
    test_rr(t, opti, a.synth_f)
    test_re9(t, opti, a.synth_f)
    test_direct_rr(t, opti, a.synth_f)
    test_sm2_settings(t, opti)
    test_rr_config_fix(t, opti)
    test_edition_download(t, opti)
    test_players_optiscaler(t, opti)
    test_experimental_and_shared_model(t, opti)
    test_orphan(t, opti)
    test_rc6_hunt(t, opti)
    test_adopt_and_self_install(t, opti, a.baseline)
    test_launcher_uninstall(t, opti)
    test_data_root_junctions(t)
    test_ui_dump(t, opti)
    print("\n%d checks passed, %d failed (%.0f s)" % (t.passed, len(t.failures), time.time() - t0))
    for f in t.failures:
        print("  FAILED:", f)
    return 1 if t.failures else 0


if __name__ == "__main__":
    sys.exit(main())
