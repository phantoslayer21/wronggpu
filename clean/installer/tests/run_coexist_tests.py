#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Tests of the launcher's coexistence rules (build/launcher-design.md sections 5, 6 and 7, milestone M2).

Written for this project; Python standard library only. Runs WrongGPU.exe (a WRONGGPU_TESTING build, as for
run_launcher_tests.py) against GENERATED fixture game folders only: every other tool's file is a synthetic stand-in
from coexist_fixtures.py (a version resource, exports, imports or a few words of text), never a real third-party file.
One case per rule of the rules table: fingerprints (never names), the proxy picker, ReShade / Special K (chain),
DXVK / vkd3d-proton (refuse), Ultimate ASI Loader (the plugin's own folder), CET / RED4ext / UE4SS (names kept), HIP
tools (warn, no version.dll), DLSS-NR on AMD (refuse, untouched), loose nvapi64 / nvngx (warn), overlays (inform),
anti-cheat (refuse), not D3D12 (refuse), older WrongGPU (migrate), two OptiScaler copies (refuse), the player's
OptiScaler (ask to switch, backed up as moved, put back byte for byte), the OptiScaler tab (locked and editable keys,
the inikey journal), leftovers (hash-matched only: clean up / adopt), orphans, warnings from the logs. Every install
is undone and the folder compared byte for byte with its snapshot.

    python run_coexist_tests.py --launcher <WrongGPU.exe> --work <dir>
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time

os.environ["WRONGGPU_TESTING"] = "1"
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import coexist_fixtures as fx  # noqa: E402
import elevated_worker as ew  # noqa: E402  (the elevated worker run directly: job files, batch, reply pipe)
import run_installer_tests as suite  # noqa: E402
import run_launcher_tests as lt  # noqa: E402
from run_installer_tests import diff, read, sha, snapshot, write  # noqa: E402

OPTI_INI_FULL = ("[Upscalers]\r\nDx12Upscaler=auto\r\n[Libraries]\r\nFfxDx12SRPath=auto\r\n[FSR]\r\nFsr4Update=auto\r\nUpscalerIndex=auto\r\n"
                 "[Menu]\r\nShortcutKey = auto\r\nShowFps=auto\r\nFpsOverlayType=auto\r\n[Spoofing]\r\nDxgiFactoryWrapping=auto\r\n"
                 "[Inputs]\r\nEnableXeSSInputs=auto\r\n[Plugins]\r\nPath=auto\r\nLoadAsiPlugins=false\r\n[Hotfix]\r\nMipmapBiasOverride=auto\r\n")


def full_package(t):
    p = os.path.join(t.work, "opti-package-full")
    shutil.rmtree(p, ignore_errors=True)
    write(os.path.join(p, "OptiScaler.dll"), fx.optiscaler())
    write(os.path.join(p, "OptiScaler.ini"), OPTI_INI_FULL)
    return p


def coexist(t, exe):
    """The rules table for one game: (exit code, {tool: [finding, ...]}, the whole JSON)."""
    rc = t.run("coexist", exe)
    by = {}
    for f in t.j.get("coexistence", []):
        by.setdefault(f["tool"], []).append(f)
    return rc, by, dict(t.j)


def one(by, tool, decision):
    return any(f["decision"] == decision for f in by.get(tool, []))


def install(t, exe, opti, *more):
    return t.run("install", exe, "--opti=" + opti, "--noscan", "--force", "--experimental", *more)


def rec_lines(b):
    return lt.record_text(b).replace("\r", "").split("\n")


# ---------------------------------------------------------------- fingerprints

def test_fingerprints(t):
    print("\n[fingerprints] a file is what it contains, never what it is called")
    d = os.path.join(t.work, "fp")
    shutil.rmtree(d, ignore_errors=True)
    files = {"winmm.dll": (fx.reshade(), "reshade"), "version.dll": (fx.optiscaler(), "optiscaler"), "ReShade64.dll": (fx.unknown(), "unknown"),
             "dxgi.dll": (fx.dxvk(), "dxvk"), "d3d12.dll": (fx.vkd3d(), "vkd3d"), "dinput8.dll": (fx.asi_loader(), "asi-loader"),
             "dbghelp.dll": (fx.microsoft(), "microsoft"), "amdhip64_7.dll": (fx.hip_runtime(), "hip-tool"), "x.dll": (fx.specialk(), "specialk"),
             "OptiScaler.dll": (fx.reshade(), "reshade"), "y.dll": (fx.cet(), "cet"), "z.dll": (fx.red4ext(), "red4ext"), "w.dll": (fx.ue4ss(), "ue4ss"),
             "nvapi64.dll": (fx.fakenvapi(), "fakenvapi"), "text.dll": (fx.optiscaler_text(), "optiscaler"),
             # 2026-10-17 regressions: DXVK / vkd3d-proton only by their own markers and only under their DLL names
             "d3d11.dll": (fx.dxvk(), "dxvk"), "q.dll": (fx.dxvk(), "unknown"), "d3d10core.dll": (fx.stray_dxvk_text(), "unknown"),
             "d3d12core.dll": (fx.vkd3d(), "vkd3d"), "d3d9.dll": (fx.optiscaler_mentions_dxvk(), "optiscaler"),
             "nvngx.dll": (fx.vendor_copy(), "nvidia"), "signed.dll": (fx.vendor_copy_signed_only(), "nvidia")}
    for n, (data, _) in files.items():
        write(os.path.join(d, n), data)
    rc = t.run("fingerprint", *[os.path.join(d, n) for n in files])
    got = {os.path.basename(f["file"]): f for f in t.j.get("files", [])}
    t.check("fingerprint: exit 0, every file identified by its content", rc == 0 and all(got.get(n, {}).get("tool") == k for n, (_, k) in files.items()),
            {n: got.get(n, {}).get("tool") for n in files})
    t.check("fingerprint: versions from the version resource (ReShade 6.3.3.0, OptiScaler 0.9.5.4.. as version.dll)",
            got.get("winmm.dll", {}).get("version") == "6.3.3.0" and got.get("version.dll", {}).get("version") == "0.9.5.4" and got.get("winmm.dll", {}).get("pe"), got.get("winmm.dll"))
    shutil.rmtree(d, ignore_errors=True)


# ---------------------------------------------------------------- the rules table

def test_reshade_chain(t, opti):
    print("\n[ReShade / Special K] they stay; OptiScaler takes another free name (untested combination)")
    for label, data, tool in (("ReShade", fx.reshade(), "reshade"), ("Special K", fx.specialk(), "specialk")):
        root, b, exe = lt.game(t, label + " Game", extra={"bin/x64/dxgi.dll": data, "bin/x64/ReShade.ini": "[GENERAL]\r\n"})
        before = snapshot(root)
        rc, by, j = coexist(t, exe)
        t.check(label + ": chain; OptiScaler's name winmm.dll, an untested combination", rc == 0 and one(by, tool, "chain") and j.get("proxy", {}).get("name") == "winmm.dll" and
                j.get("proxy", {}).get("untested") is True and "dxgi.dll (" + ("ReShade" if tool == "reshade" else "Special K") + ")" in j.get("proxy", {}).get("reason", ""), j)
        rc = install(t, exe, opti)
        lines = rec_lines(b) if rc == 0 else []
        t.check(label + ": installed as winmm.dll; record: proxy line with the reason, a coexist line", rc == 0 and any(l.startswith("proxy\twinmm.dll\t") and "dxgi.dll" in l for l in lines) and
                any(l.startswith("coexist\t" + tool + "\t") and l.endswith("\tchain") for l in lines), (rc, t.j.get("message"), lines[:14]))
        t.check(label + ": its dxgi.dll untouched; the warning names the untested combination",
                read(os.path.join(b, "dxgi.dll")) == data and any("untested" in w for w in t.j.get("warnings", [])), t.j.get("warnings"))
        rc = t.run("uninstall", exe)
        t.check(label + ": uninstall byte for byte", rc == 0 and snapshot(root) == before, diff(before, snapshot(root)))
        shutil.rmtree(root, ignore_errors=True)


def test_refusals(t, opti):
    print("\n[refusals] DXVK, vkd3d-proton, another upscaler mod, anti-cheat, two OptiScaler copies, not D3D12: nothing changed, no override")
    cases = [
        ("DXVK", {"bin/x64/dxgi.dll": fx.dxvk()}, "dxvk", 2),
        ("vkd3d", {"bin/x64/d3d12.dll": fx.vkd3d()}, "vkd3d", 2),
        ("Other Mod", {"bin/x64/" + lt.other_mod_keys()[1][0]: "[mock]\r\n"}, "other-upscaler-mod", 6),  # another upscaler mod's marker file
        ("Anticheat", {"EasyAntiCheat/EasyAntiCheat_x64.dll": "mock"}, "anti-cheat", 4),
        ("Two Opti", {"bin/x64/dxgi.dll": fx.optiscaler(), "bin/x64/winmm.dll": fx.optiscaler_text(), "bin/x64/OptiScaler.ini": "[Menu]\r\n"}, "optiscaler", 6),
    ]
    for label, extra, tool, code in cases:
        root, b, exe = lt.game(t, label + " Refusal", extra=extra)
        before = snapshot(root)
        rc, by, j = coexist(t, exe)
        t.check(label + ": refused in the rules table (" + tool + ")", rc == 0 and one(by, tool, "refuse") and j.get("refused"), by)
        rc = install(t, exe, opti, "--switch", "--removeold")
        t.check(label + ": install refused even with every confirmation (exit %d), nothing changed" % code, rc == code and snapshot(root) == before, (rc, t.j.get("message")))
        rc = t.run("status", exe)
        st = t.j.get("game", {}).get("status")
        t.check(label + ": status " + ("foreign-mod" if tool == "other-upscaler-mod" else "refused"), st == ("foreign-mod" if tool == "other-upscaler-mod" else "refused"), st)
        if tool == "other-upscaler-mod":  # Brief 17: found by its files, never named (rules table, status, refusal)
            t.check(label + ": nothing names the other mod or its files", not lt.names_other_mod(json.dumps(by) + json.dumps(t.j)), by)
        shutil.rmtree(root, ignore_errors=True)
    # not a D3D12 game: the exe's imports
    root, b, exe = lt.game(t, "Dx11 Game")
    write(exe, fx.exe(["kernel32.dll", "d3d11.dll", "dxgi.dll"]))
    os.remove(os.path.join(b, "amd_fidelityfx_dx12.dll"))  # AMD's DX12 FidelityFX DLL would be D3D12 evidence
    before = snapshot(root)
    rc, by, j = coexist(t, exe)
    t.check("not D3D12 (imports d3d11.dll): refused, the API named", rc == 0 and one(by, "graphics-api", "refuse") and "Direct3D 11" in json.dumps(by) and
            j.get("graphics", {}).get("pe") is True and j.get("graphics", {}).get("d3d12") is False, (by, j.get("graphics")))
    rc = install(t, exe, opti)
    t.check("not D3D12: install exit 2, nothing changed", rc == 2 and snapshot(root) == before and "Direct3D 12" in t.j.get("message", ""), (rc, t.j.get("message")))
    for label, data, more in (("imports d3d12.dll", fx.exe(["kernel32.dll", "d3d12.dll"]), {}),
                              ("a Direct3D 11 exe with AMD's DX12 FidelityFX DLL", fx.exe(["d3d11.dll"]), {"amd_fidelityfx_dx12.dll": suite.AMD_V1}),
                              ("loads d3d12.dll at run time", fx.exe(["d3d11.dll"], extra=b"LoadLibrary d3d12.dll"), {}),
                              ("Vulkan exe with the Agility SDK folder", fx.exe(["vulkan-1.dll"]), {"D3D12/D3D12Core.dll": b"agility (synthetic)"})):
        write(exe, data)
        for rel, d in more.items():
            write(os.path.join(b, rel), d)
        rc, by, j = coexist(t, exe)
        t.check("D3D12 evidence (" + label + "): not refused", rc == 0 and not one(by, "graphics-api", "refuse"), (by.get("graphics-api"), j.get("graphics")))
    shutil.rmtree(root, ignore_errors=True)


def test_loaders_and_tools(t, opti, full):
    print("\n[tools] Ultimate ASI Loader, CET, RED4ext, UE4SS, HIP tools, loose nvapi64 / nvngx, overlays, older WrongGPU")
    # Ultimate ASI Loader: the plugin goes to wronggpu-plugins\ (OptiScaler's [Plugins] Path)
    root, b, exe = lt.game(t, "Asi Game", extra={"bin/x64/dinput8.dll": fx.asi_loader(), "bin/x64/scripts/mod.asi": fx.unknown("mod")})
    before = snapshot(root)
    rc, by, j = coexist(t, exe)
    t.check("ASI loader: warned, the plugin gets its own folder", rc == 0 and one(by, "asi-loader", "warn") and j.get("asi_loader") is True, by)
    rc = install(t, exe, full)
    ini = lt.ini(b, "OptiScaler.ini")
    t.check("ASI loader: plugin in wronggpu-plugins\\, not plugins\\; OptiScaler's Path names that folder",
            rc == 0 and os.path.isfile(os.path.join(b, "wronggpu-plugins", "WrongGPU.asi")) and not os.path.exists(os.path.join(b, "plugins", "WrongGPU.asi")) and
            ("Path=" + os.path.join(b, "wronggpu-plugins")).lower() in ini.lower(), (rc, t.j.get("message"), ini[-300:]))
    write(os.path.join(b, "wronggpu-plugins", "wronggpu-asi.log"), "[1] [WrongGPU.asi] WrongGPU.asi loaded\r\n")  # what the plugin writes next to itself
    rc = t.run("uninstall", exe)
    t.check("ASI loader: uninstall byte for byte (the plugin's folder and its log gone)", rc == 0 and snapshot(root) == before, diff(before, snapshot(root)))
    rc = install(t, exe, opti)  # a package without [Plugins] Path: the plugin stays in plugins\, with a warning
    t.check("ASI loader, OptiScaler without a Path setting: installed with a warning (may load twice)",
            rc == 0 and any("a second time" in w for w in t.j.get("warnings", [])) and os.path.isfile(os.path.join(b, "plugins", "WrongGPU.asi")), (rc, t.j.get("warnings")))
    t.run("uninstall", exe)
    t.check("...uninstall byte for byte", snapshot(root) == before, diff(before, snapshot(root)))
    shutil.rmtree(root, ignore_errors=True)

    # CET (version.dll), RED4ext (winmm.dll), ReShade (dxgi.dll): OptiScaler takes wininet.dll
    extra = {"bin/x64/version.dll": fx.cet(), "bin/x64/plugins/cyber_engine_tweaks/mod.lua": "-- synthetic", "bin/x64/winmm.dll": fx.red4ext(),
             "bin/x64/red4ext/RED4ext.dll": fx.red4ext(), "bin/x64/dxgi.dll": fx.reshade()}
    root, b, exe = lt.game(t, "Cet Game", extra=extra)
    before = snapshot(root)
    rc, by, j = coexist(t, exe)
    t.check("CET and RED4ext: chain, their names kept; OptiScaler as wininet.dll", rc == 0 and one(by, "cet", "chain") and one(by, "red4ext", "chain") and
            j.get("proxy", {}).get("name") == "wininet.dll", (by.keys(), j.get("proxy")))
    rc = install(t, exe, opti)
    t.check("CET and RED4ext: installed as wininet.dll; their files untouched", rc == 0 and any(l.startswith("proxy\twininet.dll") for l in rec_lines(b)) and
            all(read(os.path.join(root, k)) == (v if isinstance(v, bytes) else v.encode()) for k, v in extra.items()), (rc, t.j.get("message")))
    rc = t.run("uninstall", exe)
    t.check("CET and RED4ext: uninstall byte for byte", rc == 0 and snapshot(root) == before, diff(before, snapshot(root)))
    shutil.rmtree(root, ignore_errors=True)

    # UE4SS; loose nvapi64 / nvngx; a Windows dbghelp
    root, b, exe = lt.game(t, "Ue Game", extra={"bin/x64/dwmapi.dll": fx.ue4ss(), "bin/x64/ue4ss/UE4SS.dll": fx.ue4ss(), "bin/x64/nvapi64.dll": fx.fakenvapi(),
                                                 "bin/x64/nvngx.dll": fx.unknown("loose"), "bin/x64/dbghelp.dll": fx.microsoft()})
    rc, by, j = coexist(t, exe)
    t.check("UE4SS: chain; loose nvapi64.dll (fakenvapi) and nvngx.dll: warnings; OptiScaler as dxgi.dll", rc == 0 and one(by, "ue4ss", "chain") and
            one(by, "fakenvapi", "warn") and any("nvngx.dll" in f["what"] and f["decision"] == "warn" for f in by.get("unknown", [])) and
            j.get("proxy", {}).get("name") == "dxgi.dll", (by, j.get("proxy")))
    shutil.rmtree(root, ignore_errors=True)

    # a HIP-based tool: version.dll is never taken
    root, b, exe = lt.game(t, "Hip Game", extra={"bin/x64/neuralhub.ini": "[hub]\r\n", "bin/x64/amdhip64_7.dll": fx.hip_runtime(), "bin/x64/dxgi.dll": fx.reshade(),
                                                  "bin/x64/winmm.dll": fx.unknown()})
    rc, by, j = coexist(t, exe)
    t.check("HIP tool: warned; version.dll skipped, OptiScaler as wininet.dll", rc == 0 and one(by, "hip-tool", "warn") and j.get("hip_tool") is True and
            j.get("proxy", {}).get("name") == "wininet.dll" and "HIP" in j.get("proxy", {}).get("reason", ""), (by.get("hip-tool"), j.get("proxy")))
    shutil.rmtree(root, ignore_errors=True)

    # no free name: refused before anything changes
    root, b, exe = lt.game(t, "Full Game", extra={"bin/x64/" + n: fx.unknown(n) for n in ("dxgi.dll", "winmm.dll", "version.dll", "wininet.dll", "winhttp.dll")})
    write(os.path.join(b, "dbghelp.dll"), fx.microsoft())
    before = snapshot(root)
    rc = install(t, exe, opti)
    t.check("every proxy name taken (the game's own Windows dbghelp counts): exit 6, nothing changed", rc == 6 and snapshot(root) == before and "no free name" in t.j.get("message", ""),
            (rc, t.j.get("message")))
    shutil.rmtree(root, ignore_errors=True)

    # overlays and injectors that run (test builds see only WRONGGPU_TEST_PROCESSES)
    root, b, exe = lt.game(t, "Overlay Game")
    t.env["WRONGGPU_TEST_PROCESSES"] = "RTSS.exe;LosslessScaling.exe;SKIF.exe;notepad.exe"
    rc, by, j = coexist(t, exe)
    t.env.pop("WRONGGPU_TEST_PROCESSES")
    ov = by.get("overlay", [])
    t.check("overlays: RTSS and Lossless Scaling inform (upscaling twice is pointless); Special K's injector warns",
            rc == 0 and len(ov) == 2 and all(f["decision"] == "inform" for f in ov) and any("pointless" in f["message"] for f in ov) and one(by, "specialk", "warn"), by)
    rc, by, j = coexist(t, exe)
    t.check("overlays: none when none runs", rc == 0 and "overlay" not in by, by)
    # an older WrongGPU (DLSS 4.5 AMD): migrate
    write(os.path.join(b, "version.dll"), fx.legacy_loader())
    write(os.path.join(b, "dlss45-amd-runtime.dll"), b"old runtime")
    rc, by, j = coexist(t, exe)
    t.check("an older WrongGPU (DLSS 4.5 AMD): migrate", rc == 0 and one(by, "wronggpu-legacy", "migrate"), by)
    shutil.rmtree(root, ignore_errors=True)


def test_old_layout_vendor_copy(t, opti):
    print("\n[2026-10-17 regression] an older OptiScaler layout: OptiScaler as dxgi.dll naming DXVK, the DLSS DLL copied as nvngx.dll")
    root, b, exe = lt.game(t, "Old Layout Game", extra={"bin/x64/dxgi.dll": fx.optiscaler_mentions_dxvk(), "bin/x64/nvngx.dll": fx.vendor_copy(),
                                                         "bin/x64/OptiScaler.ini": "[Upscalers]\r\n"})
    before = snapshot(root)
    rc, by, j = coexist(t, exe)
    t.check("not refused: no DXVK (the DLSS copy's stray letters and OptiScaler's DXVK text are not DXVK)", rc == 0 and not j.get("refused") and "dxvk" not in by and
            "vkd3d" not in by, by)
    t.check("nvngx.dll = the DLSS DLL copy: a warning naming the old OptiScaler layout (not a refusal, not \"unknown\")",
            any(f["decision"] == "warn" and "Old OptiScaler layout (nvngx.dll = DLSS DLL copy)" in f["what"] for f in by.get("nvidia", [])) and
            not any("nvngx.dll" in f["what"] for f in by.get("unknown", [])), by)
    t.check("the OptiScaler as dxgi.dll that names DXVK is still OptiScaler: ask to switch", one(by, "optiscaler", "ask-switch") and j.get("players_optiscaler_proxy") == "dxgi.dll", by)
    rc = t.run("status", exe)
    st = t.j.get("game", {}).get("status")
    t.check("status: the player's OptiScaler, not refused", rc == 0 and st == "players-optiscaler", (st, t.j.get("game", {}).get("status_text")))
    rc = install(t, exe, opti, "--switch")
    lines = rec_lines(b) if rc == 0 else []
    t.check("--switch: installed; the DLSS copy left alone (never written, never moved), the warning repeated",
            rc == 0 and read(os.path.join(b, "nvngx.dll")) == fx.vendor_copy() and not any(l.endswith("\tnvngx.dll") for l in lines if l.startswith(("copy\t", "disable\t"))) and
            any("nvngx.dll" in w for w in t.j.get("warnings", [])), (rc, t.j.get("message"), t.j.get("warnings")))
    rc = t.run("uninstall", exe)
    t.check("uninstall byte for byte", rc == 0 and snapshot(root) == before, diff(before, snapshot(root)))
    shutil.rmtree(root, ignore_errors=True)


def test_own_optiscaler_not_players(t, opti):
    print("\n[2026-10-17 regression] WrongGPU's own OptiScaler setup after a game run is never \"your OptiScaler\"")
    root, b, exe = lt.game(t, "Own Opti Game", validated=True)
    before = snapshot(root)
    rc = t.run("install", exe, "--opti=" + opti, "--noscan", "--force")
    # what a game run leaves: OptiScaler's log, OptiScaler.ini saved by its menu
    write(os.path.join(b, "OptiScaler.log"), "[info] OptiScaler (synthetic log of our setup)\r\n")
    installed_ini = read(os.path.join(b, "OptiScaler.ini"))
    write(os.path.join(b, "OptiScaler.ini"), installed_ini + b"; saved by OptiScaler's menu\r\n")
    rc2, by, j = coexist(t, exe)
    t.check("coexist: WrongGPU's setup managed, no \"Your OptiScaler\" (ask-switch), not two copies", rc == 0 and rc2 == 0 and one(by, "optiscaler", "managed") and
            not one(by, "optiscaler", "ask-switch") and not one(by, "optiscaler", "refuse"), by.get("optiscaler"))
    rc = t.run("status", exe)
    g = t.j.get("game", {})
    write(os.path.join(b, "OptiScaler.ini"), installed_ini)  # (a settings file changed outside is left in place on uninstall, by design)
    t.check("status: installed; the page lists no \"Asks first\"", g.get("status") == "installed" and not any(f.get("decision") == "ask-switch" for f in g.get("coexistence", [])),
            (g.get("status"), g.get("coexistence")))
    proxy = next((l.split("\t")[1] for l in rec_lines(b) if l.startswith("proxy\t")), "dxgi.dll")
    rc = t.run("repair", exe, "--opti=" + opti, "--noscan", "--force")
    t.check("repair over it: no switch asked, OptiScaler.log is not moved as a player's file", rc == 0 and not any(l.endswith("\tOptiScaler.log\tmoved") for l in rec_lines(b)),
            (rc, t.j.get("message")))
    rc = t.run("uninstall", exe)
    t.check("uninstall byte for byte (" + proxy + ", the log and the ini ours)", rc == 0 and snapshot(root) == before, diff(before, snapshot(root)))
    shutil.rmtree(root, ignore_errors=True)


def test_players_optiscaler(t, opti):
    print("\n[the player's OptiScaler] ask to switch (owner decision); backed up as moved; put back byte for byte")
    # a recognised build as winmm.dll: the switch reuses its name
    root, b, exe = lt.game(t, "Opti Pe Game", extra={"bin/x64/winmm.dll": fx.optiscaler(), "bin/x64/OptiScaler.ini": "[Menu]\r\nShowFps=true\r\n; the player's\r\n"})
    before = snapshot(root)
    rc, by, j = coexist(t, exe)
    t.check("your OptiScaler 0.9.5.4 as winmm.dll: ask to switch; OptiScaler's name the first free one (dxgi.dll)", rc == 0 and one(by, "optiscaler", "ask-switch") and
            j.get("players_optiscaler_proxy") == "winmm.dll" and j.get("proxy", {}).get("name") == "dxgi.dll", (by, j.get("proxy")))
    rc = install(t, exe, opti)
    t.check("without the player's OK: exit 6, nothing changed", rc == 6 and snapshot(root) == before, rc)
    rc = install(t, exe, opti, "--switch")
    lines = rec_lines(b) if rc == 0 else []
    t.check("--switch: installed as dxgi.dll; the player's winmm.dll moved aside (owner moved), in the backup", rc == 0 and any(l.startswith("proxy\tdxgi.dll\t") for l in lines) and
            "owner\tdisable\twinmm.dll\tmoved" in lines and not os.path.exists(os.path.join(b, "winmm.dll")) and
            read(os.path.join(b, "wronggpu-setup", "backup", "winmm.dll")) == fx.optiscaler(), (rc, t.j.get("message"), lines[:12]))
    rc = t.run("status", exe)
    t.check("after the switch: installed (the moved OptiScaler is not \"your OptiScaler\" any more)", t.j.get("game", {}).get("status") == "installed", t.j.get("game", {}).get("status"))
    rc = t.run("uninstall", exe)
    t.check("uninstall: the player's OptiScaler back byte for byte", rc == 0 and snapshot(root) == before, diff(before, snapshot(root)))
    shutil.rmtree(root, ignore_errors=True)
    # an older layout: OptiScaler as nvngx.dll (moved aside on a switch; WrongGPU never writes that name for itself)
    root, b, exe = lt.game(t, "Opti Old Game", extra={"bin/x64/nvngx.dll": fx.optiscaler((0, 7, 7, 0)), "bin/x64/OptiScaler.ini": "[Upscalers]\r\n"})
    before = snapshot(root)
    rc, by, j = coexist(t, exe)
    t.check("older layout (nvngx.dll, 0.7.7): ask to switch, named as the older layout; OptiScaler as dxgi.dll", rc == 0 and
            any(f["decision"] == "ask-switch" and "older OptiScaler layout" in f["message"] for f in by.get("optiscaler", [])) and j.get("proxy", {}).get("name") == "dxgi.dll", by)
    rc = install(t, exe, opti, "--switch")
    lines = rec_lines(b) if rc == 0 else []
    t.check("older layout, --switch: nvngx.dll moved aside (owner moved), OptiScaler as dxgi.dll", rc == 0 and "owner\tdisable\tnvngx.dll\tmoved" in lines and
            not os.path.exists(os.path.join(b, "nvngx.dll")) and os.path.isfile(os.path.join(b, "dxgi.dll")), (rc, t.j.get("message")))
    rc = t.run("uninstall", exe)
    t.check("older layout: uninstall byte for byte", rc == 0 and snapshot(root) == before, diff(before, snapshot(root)))
    shutil.rmtree(root, ignore_errors=True)


def test_opti_tab(t, full):
    print("\n[OptiScaler tab] locked and editable keys; byte-exact edits; the inikey journal; carried over by an update")
    root, b, exe = lt.game(t, "Tab Game", validated=True)
    before = snapshot(root)
    rc = t.run("install", exe, "--opti=" + full, "--noscan", "--force")
    t.check("install with an OptiScaler build that has more keys", rc == 0, t.j.get("message"))
    installed = read(os.path.join(b, "OptiScaler.ini"))
    rc = t.run("opti", exe)
    locked = {(x["section"], x["key"]) for x in t.j.get("locked", [])}
    editable = {(x["section"], x["key"]) for x in t.j.get("editable", [])}
    t.check("tab: locked keys (WrongGPU's) shown with why", rc == 0 and {("Upscalers", "Dx12Upscaler"), ("Plugins", "Path"), ("Menu", "ShowFps"), ("Hotfix", "MipmapBiasOverride")} <= locked and
            all(x["why"] for x in t.j.get("locked", [])), sorted(locked))
    t.check("tab: editable keys are only those this build has", editable == {("Menu", "ShortcutKey"), ("Menu", "FpsOverlayType"), ("Spoofing", "DxgiFactoryWrapping"),
                                                                              ("Inputs", "EnableXeSSInputs")}, sorted(editable))
    rc = t.run("set", exe, "opti.Menu.ShortcutKey=0x2d")
    after = read(os.path.join(b, "OptiScaler.ini"))
    t.check("edit: only the value's bytes change (\"ShortcutKey = 0x2D\")", rc == 0 and after == installed.replace(b"ShortcutKey = auto", b"ShortcutKey = 0x2D"), (rc, t.j.get("message")))
    t.check("edit: an inikey line in the record; the record knows the new file", any(l.startswith("inikey\tOptiScaler.ini\tMenu\tShortcutKey\t=auto\t=0x2D") for l in rec_lines(b)) and
            lt.record_entry_sha(b, "OptiScaler.ini") == sha(after), [l for l in rec_lines(b) if l.startswith("inikey")])
    for bad, why in (("opti.Upscalers.Dx12Upscaler=xess", "a locked key"), ("opti.Menu.Scale=1.5", "a key this build lacks"),
                     ("opti.Inputs.EnableXeSSInputs=maybe", "a bad value"), ("opti.Menu.NoSuch=1", "an unknown key")):
        rc = t.run("set", exe, bad)
        t.check("refused (exit 10), file unchanged: " + why, rc == 10 and read(os.path.join(b, "OptiScaler.ini")) == after, (rc, t.j.get("message")))
    # OptiScaler's menu saves its ini: an outside edit; Keep takes it
    write(os.path.join(b, "OptiScaler.ini"), after.replace(b"0x2D", b"0x2E"))
    rc = t.run("opti", exe)
    t.check("outside edit (0x2E saved over ours) shown", rc == 0 and any(x["key"] == "ShortcutKey" and x["file"] == "0x2E" and x["ours"] == "0x2D" for x in t.j.get("outside_edits", [])),
            t.j.get("outside_edits"))
    # ...and one in WrongGPU's own settings file at the same time: not the tab's, so --keep leaves it as it is (it has its own
    # Keep / Use on the game page)
    wpath = os.path.join(b, "wronggpu.ini")
    wini = read(wpath)
    wlines = wini.split(b"\n")
    at = next((i for i, l in enumerate(wlines) if l.replace(b" ", b"").startswith(b"runtime_options=")), -1)
    if at >= 0:
        cr = b"\r" if wlines[at].endswith(b"\r") else b""
        wlines[at] = wlines[at].rstrip(b"\r") + b";pre_exposure=original" + cr
    edited = b"\n".join(wlines)
    write(wpath, edited)
    rc = t.run("opti", exe, "--keep")
    t.check("--keep: the file's value becomes ours, no outside edit left", rc == 0 and not t.j.get("outside_edits") and
            any(l.startswith("inikey\tOptiScaler.ini\tMenu\tShortcutKey\t=auto\t=0x2E") for l in rec_lines(b)), t.j.get("outside_edits"))
    t.check("--keep: WrongGPU's own settings file keeps its outside change (the tab only)", at >= 0 and read(wpath) == edited, read(wpath)[-300:])
    write(wpath, wini)
    rc = t.run("set", exe, "opti.Inputs.EnableXeSSInputs=false", "opti.Spoofing.DxgiFactoryWrapping=true")
    t.check("two edits in one set", rc == 0 and b"EnableXeSSInputs=false" in read(os.path.join(b, "OptiScaler.ini")), t.j.get("message"))
    # carried over by an update (the record made older)
    rec_path = os.path.join(b, "wronggpu-setup", "install.record")
    write(rec_path, "\r\n".join("version\t0.5.1" if l.startswith("version\t") else l for l in read(rec_path).decode("utf-8").split("\r\n")))
    rc = t.run("update", exe, "--opti=" + full, "--noscan", "--force")
    now = read(os.path.join(b, "OptiScaler.ini"))
    t.check("update: the OptiScaler tab's values carried over to the new ini, the journal kept", rc == 0 and b"ShortcutKey = 0x2E" in now and b"EnableXeSSInputs=false" in now and
            b"DxgiFactoryWrapping=true" in now and sum(1 for l in rec_lines(b) if l.startswith("inikey\t")) == 3, (rc, t.j.get("message"), now[-400:]))
    # the elevated worker's job carries OptiScaler edits too
    rc, rep = ew.run_worker(t.run, t.work, ["op=apply\r\ndir=%s\r\nlocalappdata=%s\r\noptikey.Menu.FpsOverlayType=2\r\n" % (b, t.env["LOCALAPPDATA"])])
    t.check("apply (the elevated worker): optikey lines applied", rc == 0 and b"FpsOverlayType=2" in read(os.path.join(b, "OptiScaler.ini")), (rc, vars(rep)))
    rc = t.run("opti", exe, "--reset")
    t.check("--reset: the editable keys back byte for byte to what the install wrote", rc == 0 and read(os.path.join(b, "OptiScaler.ini")) == installed and
            not any(l.startswith("inikey\t") and "\tMenu\tShowFps\t" not in l for l in rec_lines(b)), (rc, t.j.get("message")))
    rc = t.run("uninstall", exe)
    t.check("uninstall: byte for byte", rc == 0 and snapshot(root) == before, diff(before, snapshot(root)))
    shutil.rmtree(root, ignore_errors=True)


def test_log_warnings(t, opti):
    print("\n[warnings from the logs] foreign tables, XeSS without depth, plugin not hooked")
    root, b, exe = lt.game(t, "Log Game", validated=True)
    rc = t.run("install", exe, "--opti=" + opti, "--noscan", "--force")
    write(os.path.join(b, "wronggpu-fsr.log"),
          "[11:00:00.000 9] [x] ==== session start: Cyberpunk2077.exe (process 9) ====\r\n"
          "[11:00:00.100 9] [x] settings: upscaler=network\r\n"
          "[11:00:01.000 9] [info] network runtime ready: executor dx12, sync inline\r\n"
          "[11:00:02.000 9] [hooks] own table: list 01 records into its own vtable 02 (a), which WrongGPU does not take over: ReShade\r\n"
          "[11:00:03.000 9] [fe] [up] WrongGPU frame: 1280x720 notes: no depth\r\n")
    write(os.path.join(b, "OptiScaler.log"), "[info] Inputs: XeSS\r\n")
    write(os.path.join(b, "wronggpu-asi.log"), "[1] [WrongGPU.asi] WrongGPU.asi loaded\r\n[2] [WrongGPU.asi] not hooked: no Selectable call site could be patched\r\n")
    rc2 = t.run("status", exe)
    lr = t.j.get("game", {}).get("last_run", {})
    kinds = {w["kind"] for w in lr.get("warnings", [])}
    t.check("status: warnings foreign-vtable, xess-no-depth, plugin-not-hooked", rc == 0 and rc2 == 0 and kinds == {"foreign-vtable", "xess-no-depth", "plugin-not-hooked"} and
            lr.get("plugin") == "not-hooked", lr)
    t.run("uninstall", exe)
    shutil.rmtree(root, ignore_errors=True)


def test_leftovers_and_orphans(t, opti):
    print("\n[leftovers / orphans] WrongGPU files without a record: hash-matched only; clean up or adopt, never automatic")
    root, b, exe = lt.game(t, "Left Game", validated=True)
    pristine = snapshot(root)
    rc = t.run("install", exe, "--opti=" + opti, "--noscan", "--force")
    ours = {n: read(os.path.join(b, n)) for n in ("wronggpu_runtime.dll", "wronggpu_ffx_dx12.dll")}
    shutil.rmtree(os.path.join(b, "wronggpu-setup"))  # the record is lost (a hand-deleted folder, an old tool)
    drop_entries(t, b)  # ... and its Apps & features entry with it
    rc2 = t.run("status", exe)
    g = t.j.get("game", {})
    left = {x["path"] for x in g.get("leftovers", [])}
    t.check("status: leftovers, only files whose hashes are WrongGPU's", rc == 0 and rc2 == 0 and g.get("status") == "leftovers" and
            {"wronggpu_runtime.dll", "wronggpu_ffx_dx12.dll", "plugins\\WrongGPU.asi"} <= left and "OptiScaler.ini" not in left and "wronggpu.ini" not in left, (g.get("status"), left))
    state = snapshot(root)
    rc = t.run("install", exe, "--opti=" + opti, "--noscan", "--force")
    t.check("install over leftovers without the player's OK: exit 7, nothing changed", rc == 7 and snapshot(root) == state, (rc, t.j.get("message")))
    rc = t.run("clean-leftovers", exe)
    after = snapshot(root)
    keys = {suite.key(os.path.relpath(os.path.join(b, p), root)) for p in left}
    t.check("clean up: exactly the hash-matched files removed, nothing else", rc == 0 and len(keys) == len(left) and all(k not in after for k in keys) and
            all(after.get(k) == v for k, v in state.items() if k not in keys and not k.endswith("/")), diff(state, after))
    rc = t.run("clean-leftovers", exe)
    t.check("clean up again: nothing left to clean (exit 9)", rc == 9, rc)
    shutil.rmtree(root, ignore_errors=True)
    # a direct install's leftovers: our DLL under the game's name, the game's kept as .wronggpu-original.dll
    for how in ("clean-leftovers", "adopt-leftovers"):
        root, b, exe = lt.game(t, "Direct Left " + how)
        pristine = snapshot(root)
        os.replace(os.path.join(b, "amd_fidelityfx_dx12.dll"), os.path.join(b, "amd_fidelityfx_dx12.wronggpu-original.dll"))
        write(os.path.join(b, "amd_fidelityfx_dx12.dll"), ours["wronggpu_ffx_dx12.dll"])
        write(os.path.join(b, "wronggpu_runtime.dll"), ours["wronggpu_runtime.dll"])
        rc = t.run(how, exe)
        if how == "clean-leftovers":
            t.check("direct leftovers, clean up: our files gone, the game's DLL back under its own name", rc == 0 and snapshot(root) == pristine, (rc, diff(pristine, snapshot(root))))
        else:
            lines = rec_lines(b) if rc == 0 else []
            t.check("direct leftovers, adopt: a record (copy + rename entries); then a managed install", rc == 0 and any(l.startswith("rename\t") for l in lines) and
                    any(l.startswith("copy\t0\t-\t") and l.endswith("wronggpu_runtime.dll") for l in lines) and b.lower() in lt.managed_list(t), (rc, t.j.get("message"), lines))
            rc = t.run("status", exe)
            t.check("adopted: status installed or needs repair (no wronggpu.ini)", t.j.get("game", {}).get("status") in ("installed", "needs-repair", "update-available"), t.j.get("game", {}).get("status"))
            rc = t.run("uninstall", exe)
            t.check("adopted, uninstall: the game as it was before our files, byte for byte", rc == 0 and snapshot(root) == pristine, diff(pristine, snapshot(root)))
        shutil.rmtree(root, ignore_errors=True)
    # an orphan with a file changed since the install: left alone (not provably ours)
    root, b, exe = lt.game(t, "Orphan Changed", exe="Orph.exe")
    rc = install(t, exe, opti)
    rt = os.path.join(b, "wronggpu_runtime.dll")
    write(rt, read(rt) + b"changed by someone")
    os.remove(exe)
    rc2 = t.run("uninstall", b)
    t.check("orphan: uninstall by its folder; a changed file is left in place with a warning", rc == 0 and rc2 == 0 and os.path.isfile(rt) and
            any("changed since install" in w for w in t.j.get("warnings", [])), (rc, rc2, t.j.get("warnings")))
    shutil.rmtree(root, ignore_errors=True)


SIGNEDBIN = os.path.join("Engine", "Plugins", "FSR", "Source", "fidelityfx-sdk", "Kits", "FidelityFX", "signedbin")


COMPANIONS = {"wronggpu.ini": "[upscaler]\r\nmode=network\r\n", "wronggpu-fsr.log": "an older direct build's log\r\n",
              "wronggpu-fsr.log.old": "the log before that\r\n", "wronggpu-license.txt": "license text (synthetic)\r\n",
              "wronggpu-notices.txt": "notices text (synthetic)\r\n"}


def old_direct_runtime():
    """An older direct build's runtime: not one of our releases' hashes, but WrongGPU's version resource (synthetic)."""
    return fx.mockpe.make_dll({"ProductName": "WrongGPU", "FileDescription": "WrongGPU runtime (synthetic, an older direct build)"},
                              version=(0, 2, 6, 0))


def test_engine_direct_leftovers(t, opti):
    print("\n[engine leftovers] an older direct install in an Unreal game's engine plugin folder (LEGO Batman): found, migrated with --removeold")
    # our files as the hosted route installs them (hash-matched to the embedded payload), from a throwaway game
    root, b, exe = lt.game(t, "Engine Source")
    rc = install(t, exe, opti)
    ours = read(os.path.join(b, "wronggpu_ffx_dx12.dll"))
    t.run("uninstall", exe)
    shutil.rmtree(root, ignore_errors=True)
    t.check("engine leftovers: our FidelityFX DLL from a hosted install", rc == 0 and len(ours) > 0, rc)

    def planted(name, keep_original=True):
        root, b, exe = lt.direct_game(t, name)
        sb = os.path.join(root, SIGNEDBIN)
        write(os.path.join(sb, "amd_fidelityfx_dx12.dll"), suite.AMD_V1)          # the engine plugin's own FSR DLL
        write(os.path.join(sb, "amd_fidelityfx_vk.dll"), "the engine plugin's Vulkan DLL (untouched)")
        pristine = snapshot(root)
        # what the old direct route left: ours under the game's name, the game's kept beside it, our runtime, ini, log
        if keep_original:
            os.replace(os.path.join(sb, "amd_fidelityfx_dx12.dll"), os.path.join(sb, "amd_fidelityfx_dx12.wronggpu-original.dll"))
        write(os.path.join(sb, "amd_fidelityfx_dx12.dll"), ours)
        write(os.path.join(sb, "wronggpu_runtime.dll"), old_direct_runtime())
        for n, text in COMPANIONS.items():
            write(os.path.join(sb, n), text)
        return root, b, exe, sb, pristine

    root, b, exe, sb, pristine = planted("Engine Left")
    rel_sb = SIGNEDBIN
    rc = t.run("status", exe, "--noscan")
    g = t.j.get("game", {})
    left = {x["path"]: x for x in g.get("leftovers", [])}
    dlls = {os.path.join(rel_sb, "amd_fidelityfx_dx12.dll"), os.path.join(rel_sb, "wronggpu_runtime.dll")}
    comp = {os.path.join(rel_sb, n) for n in COMPANIONS}
    t.check("status: the old direct install in the engine plugin folder is found (by hash, and by WrongGPU's version resource)",
            rc == 0 and g.get("status") == "leftovers" and dlls <= set(left) and
            "older direct install" in left.get(os.path.join(rel_sb, "wronggpu_runtime.dll"), {}).get("version", ""), (g.get("status"), left))
    t.check("status: the old install's ini, logs, license and notices count too (beside a DLL of ours), nothing else",
            set(left) == dlls | comp and os.path.join(rel_sb, "amd_fidelityfx_vk.dll") not in left, sorted(left))
    state = snapshot(root)
    rc = install(t, exe, opti)
    t.check("install without the player's OK: exit 7 (an older install), nothing changed", rc == 7 and snapshot(root) == state, (rc, t.j.get("message")))
    rc = t.run("adopt-leftovers", exe)
    t.check("adopt: refused for the engine folder (Clean up only), nothing changed", rc == 2 and snapshot(root) == state and
            "engine folder" in (t.j.get("message") or ""), (rc, t.j.get("message")))
    rc = install(t, exe, opti, "--removeold")
    after = snapshot(root)
    k = lambda n: suite.key(os.path.join(rel_sb, n))
    t.check("install --removeold: our files gone from the engine folder, the game's FSR DLL back under its own name",
            rc == 0 and after.get(k("amd_fidelityfx_dx12.dll")) == pristine.get(k("amd_fidelityfx_dx12.dll")) and
            k("amd_fidelityfx_dx12.wronggpu-original.dll") not in after and k("wronggpu_runtime.dll") not in after and
            not any(k(n) in after for n in COMPANIONS) and
            after.get(k("amd_fidelityfx_vk.dll")) == pristine.get(k("amd_fidelityfx_vk.dll")), (rc, t.j.get("message"), diff(pristine, after)))
    t.check("install --removeold: then the hosted install beside the game's exe", rc == 0 and os.path.isfile(os.path.join(b, "wronggpu_runtime.dll")) and
            os.path.isfile(os.path.join(b, "wronggpu-setup", "install.record")), rc)
    rc = t.run("uninstall", exe)
    after = snapshot(root)
    rest = {kk for kk in set(after) ^ set(pristine)}
    t.check("uninstall: the game exactly as before the old install (its ini, logs and text files gone too)",
            rc == 0 and not rest and all(after[x] == pristine[x] for x in pristine), (rc, sorted(rest)))
    shutil.rmtree(root, ignore_errors=True)

    # the same names in a signedbin folder with no DLL of ours: the game's (or another tool's) files, never counted
    root, b, exe = lt.direct_game(t, "Engine Names Only")
    sb = os.path.join(root, SIGNEDBIN)
    write(os.path.join(sb, "amd_fidelityfx_dx12.dll"), suite.AMD_V1)
    for n, text in COMPANIONS.items():
        write(os.path.join(sb, n), text)
    state = snapshot(root)
    rc = t.run("status", exe, "--noscan")
    g = t.j.get("game", {})
    t.check("names only (no DLL of ours in that signedbin): not leftovers, nothing counted",
            rc == 0 and g.get("status") != "leftovers" and not g.get("leftovers"), (g.get("status"), g.get("leftovers")))
    rc = install(t, exe, opti)
    t.check("names only: the install proceeds (no exit 7) and leaves those files alone",
            rc == 0 and all(snapshot(root).get(x) == state[x] for x in state), (rc, t.j.get("message")))
    t.run("uninstall", exe)
    shutil.rmtree(root, ignore_errors=True)

    # the game's own DLL is not there to put back: ours stays (the engine plugin needs a DLL of that name)
    root, b, exe, sb, pristine = planted("Engine Left No Original", keep_original=False)
    rc = t.run("clean-leftovers", exe)
    notes = " ".join(t.j.get("notes", []))
    t.check("clean up without the game's own DLL: our FidelityFX DLL stays (with a note), the provable runtime goes",
            rc == 0 and read(os.path.join(sb, "amd_fidelityfx_dx12.dll")) == ours and not os.path.isfile(os.path.join(sb, "wronggpu_runtime.dll")) and
            "not there to put back" in notes, (rc, notes))
    t.check("clean up without the game's own DLL: the old ini, logs and text files go",
            not any(os.path.isfile(os.path.join(sb, n)) for n in COMPANIONS), sorted(os.listdir(sb)))
    shutil.rmtree(root, ignore_errors=True)


def test_status_lists_findings(t, opti):
    print("\n[status / scan] the rules table in status --json and the window's game page")
    root, b, exe = lt.game(t, "Listed Game", validated=True, extra={"bin/x64/dxgi.dll": fx.reshade()})
    rc = t.run("status", exe)
    g = t.j.get("game", {})
    t.check("status: coexistence findings and the proxy plan", rc == 0 and any(f["tool"] == "reshade" and f["decision"] == "chain" for f in g.get("coexistence", [])) and
            g.get("proxy_plan", {}).get("name") == "winmm.dll" and "ReShade" in " ".join(g.get("found", [])), g)
    rc = t.run("install", exe, "--opti=" + opti, "--noscan", "--force")
    dump = os.path.join(t.work, "ui-dump-m2.json")
    rc2 = t.run("ui-dump", "--dump=" + dump, "--noscan", "--library=" + os.path.join(t.work, "games"))
    try:
        d = json.loads(read(dump).decode("utf-8"))
    except Exception as e:  # noqa: BLE001
        d = {"error": str(e)}
    pages = [p for p in d.get("games", []) if p.get("name") == "Cyberpunk 2077"]
    page = pages[0] if pages else {}
    texts = {c["text"]: c for c in page.get("opti_controls", [])}
    t.check("window: the game page says what else is there (ReShade works next to it)", rc == 0 and rc2 == 0 and "ReShade" in page.get("info", "") and "Works next to it" in page.get("info", ""),
            page.get("info"))
    t.check("window: an OptiScaler card with its settings and the keys WrongGPU manages", "FPS overlay: Default" in texts and
            any(k.startswith("Managed by WrongGPU") for k in texts) and "Managed by WrongGPU" in page.get("opti_info", ""),
            (list(texts), page.get("opti_info")))
    t.run("uninstall", exe)
    shutil.rmtree(root, ignore_errors=True)


def drop_entries(t, under):
    """Removes the test registry's Apps & features entries of games below `under` (test root only)."""
    for k, v in t.reg_keys().items():
        if k.startswith("WrongGPU-") and v.get("InstallLocation", "").lower().startswith(under.lower()):
            subprocess.run(["reg", "delete", t.uninstall + "\\" + k, "/f"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--launcher", required=True)
    ap.add_argument("--work", required=True)
    a = ap.parse_args()
    work = os.path.abspath(a.work)
    for d in ("games", "data", "localappdata"):
        shutil.rmtree(os.path.join(work, d), ignore_errors=True)
    os.makedirs(work, exist_ok=True)
    model = suite.cases.f_mini_case(present=0)
    t = lt.T(os.path.abspath(a.launcher), work, model, None)
    t.env.pop("WRONGGPU_TEST_PROCESSES", None)
    drop_entries(t, work)  # entries of an earlier run
    opti = lt.opti_package(t)
    full = full_package(t)
    t0 = time.time()
    test_fingerprints(t)
    test_reshade_chain(t, opti)
    test_refusals(t, opti)
    test_loaders_and_tools(t, opti, full)
    test_players_optiscaler(t, opti)
    test_old_layout_vendor_copy(t, opti)
    test_own_optiscaler_not_players(t, opti)
    test_opti_tab(t, full)
    test_log_warnings(t, opti)
    test_leftovers_and_orphans(t, opti)
    test_engine_direct_leftovers(t, opti)
    test_status_lists_findings(t, opti)
    drop_entries(t, work)  # nothing of this run stays in the test registry (the launcher tests share it)
    print("\n%d checks passed, %d failed (%.0f s)" % (t.passed, len(t.failures), time.time() - t0))
    for f in t.failures:
        print("  FAILED:", f)
    return 1 if t.failures else 0


if __name__ == "__main__":
    sys.exit(main())
