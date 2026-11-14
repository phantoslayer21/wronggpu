#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""End-to-end tests of the installer (spec 30: 30-T1 and 30-T2 with synthetic model files).

Written for this project; Python standard library only. Runs WrongGPU-Setup.exe in silent mode against THROWAWAY mock
game folders, mock store libraries and synthetic DLSS model files only (fake source blocks built by the converter's
test generator; never a real model file). On Linux it uses Wine (a dedicated prefix) and builds the test installer
with MinGW; on Windows pass --setup with an installer built from the files that --prepare-only writes.

    python run_installer_tests.py [--work <dir>] [--setup <exe>] [--prepare-only]
"""
import argparse
import hashlib
import json
import os
import random
import re
import shutil
import struct
import subprocess
import sys
import time
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
INSTALLER = os.path.dirname(HERE)
CONVERTER_TESTS = os.path.join(INSTALLER, "..", "converter", "tests")
sys.path.insert(0, HERE)
sys.path.insert(0, CONVERTER_TESTS)

import mockpe  # noqa: E402
import cases  # noqa: E402  (converter test cases: synthetic model files)
from run_tests import read_container  # noqa: E402



def uninstall_root(localappdata):
    """The Apps & features root the build under test writes to (wg_core/constants.h UninstallRoot): a WRONGGPU_TESTING
    build has a test root of its own per %LOCALAPPDATA% (FNV-1a of the folder: backslashes, no trailing one, ASCII lower
    case, over its UTF-16 code units), so a run never counts another run's or a sandbox's games as still installed."""
    if os.environ.get("WRONGGPU_TESTING") != "1":
        return r"HKCU\Software\Microsoft\Windows\CurrentVersion\Uninstall"
    p = "".join(chr(ord(c) + 32) if "A" <= c <= "Z" else c for c in localappdata.replace("/", "\\")).rstrip("\\")
    h = 2166136261
    for (u,) in struct.iter_unpack("<H", p.encode("utf-16-le")):
        h = ((h ^ u) * 16777619) & 0xFFFFFFFF
    return r"HKCU\Software\WrongGPU-CandidateTests\%08X\Uninstall" % h


EXIT = {"ok": 0, "failed": 1, "refused": 2, "experimental": 3, "anticheat": 4, "no_model": 5, "other_mods": 6,
        "old_install": 7, "damaged": 8, "not_installed": 9, "usage": 10}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def write(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(data if isinstance(data, bytes) else data.encode())


def read(path):
    with open(path, "rb") as f:
        return f.read()


def key(rel):
    """Snapshot key: lower case with forward slashes on every OS ("bin/x64/game.exe")."""
    return rel.replace("\\", "/").lower()


def snapshot(root):
    snap = {}
    for dirpath, dirnames, filenames in os.walk(root):
        for d in dirnames:
            snap[key(os.path.relpath(os.path.join(dirpath, d), root)) + "/"] = "dir"
        for fn in filenames:
            p = os.path.join(dirpath, fn)
            snap[key(os.path.relpath(p, root))] = sha(read(p))
    return snap


def diff(a, b):
    out = []
    for k in sorted(set(a) | set(b)):
        if a.get(k) != b.get(k):
            out.append("%s: %s -> %s" % (k, "absent" if k not in a else a[k][:8], "absent" if k not in b else b[k][:8]))
    return "; ".join(out[:8])


class Env:
    """Runs Windows programs natively or through Wine and maps paths."""

    def __init__(self, work):
        self.work = work
        self.native = os.name == "nt"
        if not self.native:
            self.prefix = os.path.join(work, "wineprefix")
            self.env = dict(os.environ, WINEPREFIX=self.prefix, WINEDEBUG="-all")
            self.wine = shutil.which("wine64") or "/usr/lib/wine/wine64"
            self.wineserver = shutil.which("wineserver") or "/usr/lib/wine/wineserver"
            subprocess.run([self.wine, "wineboot", "-i"], env=self.env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            subprocess.Popen([self.wineserver, "-p"], env=self.env).wait()
        else:
            # Native Windows: never touch the tester's own WrongGPU data, store registry keys or libraries.
            self.env = dict(os.environ, LOCALAPPDATA=os.path.join(work, "localappdata"), ProgramData=os.path.join(work, "programdata"))
            os.makedirs(self.env["LOCALAPPDATA"], exist_ok=True)

    def close(self):
        if not self.native:
            subprocess.run([self.wineserver, "-k"], env=self.env)

    def win(self, path):
        return path if self.native else "Z:" + os.path.abspath(path).replace("/", "\\")

    def run(self, args, timeout=300, extra_env=None):
        cmd = list(args) if self.native else [self.wine] + list(args)
        env = dict(self.env, **(extra_env or {}))
        p = subprocess.run(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout)
        return p.returncode

    def localappdata(self):
        if self.native:
            return self.env["LOCALAPPDATA"]
        return os.path.join(self.prefix, "drive_c", "users", os.environ.get("USER", "root"), "AppData", "Local")

    def programdata(self):
        return self.env["ProgramData"] if self.native else os.path.join(self.prefix, "drive_c", "ProgramData")

    def reg(self, *args):
        cmd = ["reg"] + list(args) if self.native else [self.wine, "reg"] + list(args)
        p = subprocess.run(cmd, env=self.env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return p.returncode, p.stdout.decode("utf-8", "replace")

    def uninstall(self):
        """This environment's Apps & features root (uninstall_root of its %LOCALAPPDATA%, as the program sees it)."""
        return uninstall_root(self.env["LOCALAPPDATA"] if self.native else "C:\\users\\player\\AppData\\Local")

    def reg_keys(self, root=None):
        """{subkey name: {value: data}} under root (default: this environment's Apps & features root)."""
        root = root or self.uninstall()
        code, out = self.reg("query", root, "/s")
        keys, cur = {}, None
        for line in out.splitlines():
            if line.upper().startswith("HKEY_"):
                name = line.strip().split("\\")[-1]
                cur = keys.setdefault(name, {}) if line.strip().lower() != root.lower().replace("hkcu", "hkey_current_user") else None
            elif cur is not None and line.startswith("    "):
                parts = re.split(r"\s{4}", line.strip(), maxsplit=2)
                if len(parts) == 3:
                    cur[parts[0]] = parts[2]
        return keys


class Tests:
    def __init__(self, env, setup, work, model):
        self.env, self.setup, self.work, self.model = env, setup, work, model
        self.failures, self.passed = [], 0
        self.log = os.path.join(work, "setup.log")
        # What the cases that are not about the install route add to an install: nothing, or /OPTISCALER=<a mock
        # package> for an installer without the direct FSR route (main).
        self.route = ()

    def check(self, name, cond, detail=""):
        if cond:
            self.passed += 1
            print("  PASS ", name)
        else:
            self.failures.append(name + (": " + detail if detail else ""))
            print("  FAIL ", name, detail)

    def run(self, *args, exe=None, extra_env=None):
        if os.path.exists(self.log):
            os.remove(self.log)
        args = [a if not a.startswith(("/DIR=", "/EXE=", "/MODEL=", "/OUT=")) else a.split("=", 1)[0] + "=" + self.env.win(a.split("=", 1)[1]) for a in args]
        if self.env.native and "/NOSCAN" not in args:
            args.append("/NOSCAN")  # never scan the tester's real store libraries
        return self.env.run([exe or self.setup, "/S", "/LOG=" + self.env.win(self.log)] + list(args), extra_env=extra_env)

    def logtext(self):
        try:
            return read(self.log).decode("utf-8", "replace")
        except OSError:
            return ""

    def uninstall(self, install_dir, *extra):
        state = os.path.join(install_dir, "wronggpu-setup")
        rc = self.run(*extra, exe=os.path.join(state, "uninstall.exe"))
        for _ in range(100):  # the uninstaller deletes itself through a short-lived cmd.exe
            if not os.path.exists(state):
                break
            time.sleep(0.2)
        return rc

    def fresh(self, name):
        root = os.path.join(self.work, "games", name)
        if os.path.exists(root):
            shutil.rmtree(root)
        os.makedirs(root)
        return root

    def appdata(self):
        return os.path.join(self.env.localappdata(), "WrongGPU")

    def model_file(self):
        return os.path.join(self.appdata(), "model", "model.wgcm")

    def clear_model(self):
        shutil.rmtree(os.path.join(self.appdata(), "model"), ignore_errors=True)

    def keys_for(self, install_dir):
        loc = self.env.win(install_dir)
        return {k: v for k, v in self.env.reg_keys().items() if v.get("InstallLocation", "").lower() == loc.lower()}

    def model_ok(self, case=None):
        """The model file on this PC holds the reference tensors of `case` (default: the suite's synthetic source, which
        has the output-filter block): the key, the stages in map order, each digest, every tensor's bytes. The optional
        group `filter` is expected exactly when the case's source has the block (spec 08 section 7.2)."""
        case = case or self.model
        try:
            key, _, stages = read_container(self.model_file())
        except Exception:  # noqa: BLE001
            return False
        want = [st for st in case.map["stages"] if st["stage"] in case.expected]
        if key != case.key or [s[0] for s in stages] != [st["stage"] for st in want]:
            return False
        for (name, digest, tensors), st in zip(stages, want):
            expected = st["digest"] if isinstance(st["digest"], str) else st["digest"][case.found[name]]
            if digest.hex() != expected or [t[3] for t in tensors] != [b for _, b in case.expected[name]]:
                return False
        return True


# ---------------------------------------------------------------- mock content

SLOTS = ["amd_fidelityfx_dx12.dll", "ffx_fsr3_x64.dll", "ffx_fsr3upscaler_x64.dll"]
API_ONLY = SLOTS[:1]
COMPONENTS = SLOTS[1:]
AMD_V1 = mockpe.make_dll(mockpe.AMD, (1, 1, 4, 0), extra=b"amd fidelityfx v1" * 10, exports=mockpe.FFX_API_EXPORTS, dll_name=SLOTS[0])
AMD_V2 = mockpe.make_dll(mockpe.AMD, (1, 1, 5, 0), extra=b"amd fidelityfx v2 (game update)" * 10, exports=mockpe.FFX_API_EXPORTS, dll_name=SLOTS[0])
FOREIGN = mockpe.make_dll({"CompanyName": "Some Modder", "ProductName": "Another upscaler"}, (0, 9, 0, 0))
# The FSR 3.0 component DLLs a game developer builds: no version resource, recognised by their exports.
FSR3 = mockpe.make_dll(None, extra=b"game-built fsr3 v1" * 10, exports=mockpe.FSR3_EXPORTS, dll_name=SLOTS[1])
FSR3UP = mockpe.make_dll(None, extra=b"game-built upscaler v1" * 10, exports=mockpe.FSR3UPSCALER_EXPORTS, dll_name=SLOTS[2])
FSR3UP_V2 = mockpe.make_dll(None, extra=b"game-built upscaler v2 (game update)" * 10, exports=mockpe.FSR3UPSCALER_EXPORTS, dll_name=SLOTS[2])
FOREIGN_FSR3 = mockpe.make_dll(None, extra=b"some other mod", exports=["ModInit", "ModHook"], dll_name=SLOTS[1])
OLD_DLSS = mockpe.make_dll({"CompanyName": "Mock", "ProductName": "Mock DLSS"}, (310, 1, 0, 0), extra=random.Random(5).randbytes(200_000))


def payload(root):
    exports = {SLOTS[0]: mockpe.FFX_API_EXPORTS, SLOTS[1]: mockpe.FSR3_EXPORTS, SLOTS[2]: mockpe.FSR3UPSCALER_EXPORTS}
    for name in SLOTS:  # WrongGPU's three FSR DLLs (front end, spec 20)
        write(os.path.join(root, name), mockpe.make_dll(mockpe.FRONT_END, (0, 3, 0, 0), extra=("front end " + name).encode(), exports=exports[name], dll_name=name))
    write(os.path.join(root, "wronggpu_runtime.dll"), mockpe.make_dll({"ProductName": "WrongGPU", "FileDescription": "WrongGPU runtime"}, (0, 3, 0, 0)))
    rng = random.Random(11)
    for k in ("enc", "swin", "dec"):
        write(os.path.join(root, "wronggpu", "kernels", k + ".hsaco"), rng.randbytes(3000))
    write(os.path.join(root, "wronggpu-notices.txt"), "mock notices\r\n")
    write(os.path.join(root, "plugins", "WrongGPU.asi"), mockpe.make_dll({"ProductName": "WrongGPU", "FileDescription": "WrongGPU OptiScaler plugin"}, (0, 4, 5, 1)))


def payload_installed(t, install_dir, payload_dir, slots=SLOTS):
    """Every payload file is in place, the FSR DLLs only for `slots` (the ones the game has), the others untouched."""
    for dirpath, _, files in os.walk(payload_dir):
        for fn in files:
            src = os.path.join(dirpath, fn)
            rel = os.path.relpath(src, payload_dir)
            dst = os.path.join(install_dir, rel)
            if rel in SLOTS and rel not in slots:
                if os.path.exists(dst) and read(dst) == read(src):
                    return False
                continue
            if not os.path.isfile(dst) or read(dst) != read(src):
                return False
    return True


def originals_kept(install_dir, expected):
    """{slot name: game's bytes}: each is now <name>.wronggpu-original.dll."""
    return all(read(os.path.join(install_dir, n[:-4] + ".wronggpu-original.dll")) == data for n, data in expected.items())


def cyberpunk(t, name, model=None, ffx=AMD_V1, fsr3=FSR3, fsr3up=FSR3UP):
    root = t.fresh(name)
    b = os.path.join(root, "bin", "x64")
    write(os.path.join(b, "Cyberpunk2077.exe"), "mock game exe" * 100)
    for n, data in zip(SLOTS, (ffx, fsr3, fsr3up)):
        if data is not None:
            write(os.path.join(b, n), data)
    write(os.path.join(b, "ffx_backend_dx12_x64.dll"), "fsr 3.0 backend (untouched)")
    write(os.path.join(b, "steam_api64.dll"), "untouched game file")
    write(os.path.join(root, "REDprelauncher.exe"), "launcher")
    if model is not None:
        write(os.path.join(b, "nvngx_dlss.dll"), model)
    return root, b


def unreal(t, name, model=None, ffx_in_plugin=False):
    root = t.fresh(name)
    win64 = os.path.join(root, "MyGame", "Binaries", "Win64")
    write(os.path.join(root, "MyGame.exe"), "unreal bootstrap")
    write(os.path.join(win64, "MyGame-Win64-Shipping.exe"), "shipping exe" * 1000)
    write(os.path.join(win64, "steam_api64.dll"), "untouched game file")
    ffx_dir = os.path.join(root, "Engine", "Plugins", "FSR3", "Binaries", "Win64") if ffx_in_plugin else win64
    write(os.path.join(ffx_dir, "amd_fidelityfx_dx12.dll"), AMD_V1)
    write(os.path.join(root, "Engine", "Binaries", "Win64", "CrashReportClient.exe"), "crash reporter")
    if model is not None:
        write(os.path.join(root, "Engine", "Plugins", "DLSS", "Binaries", "Win64", "nvngx_dlss.dll"), model)
    return root, win64, ffx_dir


def legacy_install(t, exe_dir):
    """A mock WrongGPU 0.2.x ("DLSS 4.5 AMD") install with its record, backup, files and registry entry."""
    state = os.path.join(exe_dir, "dlss45-amd")
    loader = b"mock 0.2 loader " + "dlss45-loader.log".encode("utf-16-le")
    winmm_orig = read(os.path.join(exe_dir, "winmm.dll"))
    # Paths in the old record use both separators (the installer must accept either).
    files = {"version.dll": loader, "dlss45-amd-runtime.dll": b"old runtime", "dlss45-amd.pak": b"old pack", "winmm.dll": b"replaced by the old install",
             "dlss45-model/w.bin": b"old weights", "dlss45-model\\gpu-build\\k.hsaco": b"old kernel"}
    lines = ["DLSS45AMD-MANIFEST 2", "version\t0.2.0", "game\t" + t.env.win(exe_dir), "exe\tCyberpunk2077.exe", "slot\tversion.dll",
             "profile\tx", "status\tvalidated", "regkey\tDLSS45AMD-0000ABCD"]
    for rel, data in files.items():
        existed = rel == "winmm.dll"
        lines.append("copy\t%d\t%s\t%s\t%s" % (existed, sha(winmm_orig) if existed else "-", sha(data), rel))
        if existed:
            write(os.path.join(state, "backup", rel), winmm_orig)
        write(os.path.join(exe_dir, *re.split(r"[\\/]", rel)), data)
    write(os.path.join(state, "install.manifest"), "\r\n".join(lines) + "\r\n")
    write(os.path.join(state, "uninstall.exe"), "old uninstaller")
    write(os.path.join(exe_dir, "dlss45-loader.log"), "old log")
    key = t.env.uninstall() + r"\DLSS45AMD-0000ABCD"
    t.env.reg("add", key, "/v", "InstallLocation", "/t", "REG_SZ", "/d", t.env.win(exe_dir), "/f")
    t.env.reg("add", key, "/v", "DisplayName", "/t", "REG_SZ", "/d", "DLSS 4.5 AMD - Cyberpunk 2077", "/f")


def steam_library(t, games):
    """A mock Steam install with one library holding the given {installdir: (name, builder)} games. Under Wine only: it writes
    HKCU\\Software\\Valve\\Steam SteamPath, which on native Windows would be the tester's own Steam (the native run skips the
    library search tests before this is reached; the guard keeps it that way)."""
    if t.env.native:
        raise RuntimeError("steam_library writes the Steam registry key: Wine only, never on native Windows")
    steam = os.path.join(t.work, "steam")
    shutil.rmtree(steam, ignore_errors=True)
    write(os.path.join(steam, "steamapps", "libraryfolders.vdf"), '"libraryfolders"\n{\n "0"\n {\n  "path"  "%s"\n }\n}\n' % t.env.win(steam).replace("\\", "\\\\"))
    for i, (installdir, name) in enumerate(games.items()):
        write(os.path.join(steam, "steamapps", "appmanifest_%d.acf" % (1000 + i)), '"AppState"\n{\n "name" "%s"\n "installdir" "%s"\n}\n' % (name, installdir))
    t.env.reg("add", r"HKCU\Software\Valve\Steam", "/v", "SteamPath", "/t", "REG_SZ", "/d", t.env.win(steam).replace("\\", "/"), "/f")
    return os.path.join(steam, "steamapps", "common")


# ---------------------------------------------------------------- the tests

def run_all(t, payload_dir):
    GOOD = t.model.file

    print("\n[1] Cyberpunk layout (validated): install, collect logs, uninstall, exact restore")
    t.clear_model()
    root, b = cyberpunk(t, "cp", model=GOOD)
    before = snapshot(root)
    rc = t.run("/DIR=" + root, "/FORCE")
    log = t.logtext()
    t.check("install exit 0", rc == 0, "exit %d; %s" % (rc, log[-600:]))
    t.check("payload installed: all three FSR DLLs replaced by WrongGPU's", payload_installed(t, b, payload_dir))
    t.check("the game's three FSR DLLs kept as *.wronggpu-original.dll", originals_kept(b, {SLOTS[0]: AMD_V1, SLOTS[1]: FSR3, SLOTS[2]: FSR3UP}))
    t.check("FSR 3.0 component DLLs recognised by their exports", "ffx_fsr3_x64.dll: exports the FSR 3.0 functions" in log and
            "ffx_fsr3upscaler_x64.dll: exports the FSR 3.0 upscaler functions" in log, log[-1500:])
    t.check("Authenticode result of the FSR DLLs logged (not required)", "amd_fidelityfx_dx12.dll: Advanced Micro Devices, Inc., version 1.1.4.0; " in log and "Authenticode" in log, log[-1500:])
    t.check("FSR DLL recognised as AMD's from its version resource", "AMD FSR found: amd_fidelityfx_dx12.dll (AMD, version 1.1.4.0), ffx_fsr3_x64.dll (the game's FSR 3.0 build)" in log, log[-800:])
    t.check("model converted into %LOCALAPPDATA%\\WrongGPU\\model, bit-exact with the reference", t.model_ok())
    src = read(os.path.join(t.appdata(), "model", "model-source.txt")).decode()
    t.check("model source recorded (this game's nvngx_dlss.dll)", "origin\tthis game" in src and "nvngx_dlss.dll" in src, src)
    t.check("the DLSS file is untouched", read(os.path.join(b, "nvngx_dlss.dll")) == GOOD)
    t.check("other FidelityFX DLLs untouched", read(os.path.join(b, "ffx_backend_dx12_x64.dll")) == b"fsr 3.0 backend (untouched)")
    rec = read(os.path.join(b, "wronggpu-setup", "install.record")).decode()
    t.check("install record: validated, three rename + replaced copy entries", "status\tvalidated" in rec and rec.count("\nrename\t") == 3 and rec.count("\ncopy\t1\t") == 3, rec[:600])
    keys = t.keys_for(b)
    t.check("one Apps & features entry", len(keys) == 1, str(t.env.reg_keys()))
    if len(keys) == 1:
        v = list(keys.values())[0]
        t.check("entry name and Modify = collect logs", v.get("DisplayName") == "WrongGPU - Cyberpunk 2077" and v.get("ModifyPath", "").endswith('uninstall.exe" /COLLECT'), str(v))
    # Pretend the game ran: logs written by WrongGPU (wronggpu-*), plus the player's own settings file.
    for n in ("wronggpu-fsr.log", "wronggpu-fsr.ini", "wronggpu-runtime.log", "wronggpu-report.json", "wronggpu-crash-20261005-1.txt"):
        write(os.path.join(b, n), "mock " + n)
    write(os.path.join(root, "wronggpu-root.log"), "log in the game root")
    diagnostic = {"schema": 1, "session": "test-session", "module": "amd_fidelityfx_dx12", "event": "failure",
                  "data": {"code": "WG-HIP-719", "incident": "test-incident", "observed_at": "input_wait",
                           "numeric_code": 719, "root_cause_proven": False}}
    write(os.path.join(b, "wronggpu-diagnostics.jsonl"), json.dumps(diagnostic) + "\n")
    zpath = os.path.join(t.work, "cp-logs.zip")
    rc = t.run("/COLLECT", "/DIR=" + b, "/OUT=" + zpath, exe=os.path.join(b, "wronggpu-setup", "uninstall.exe"))
    t.check("collect logs through uninstall.exe /COLLECT", rc == 0 and os.path.isfile(zpath), "exit %d" % rc)
    if os.path.isfile(zpath):
        z = zipfile.ZipFile(zpath)
        names = z.namelist()
        info = z.read("system-info.txt").decode("utf-8", "replace")
        t.check("bundle: system info, logs, record, model source, events", all(n in names for n in (
            "system-info.txt", "windows-events.txt", "wronggpu-fsr.log", "wronggpu-fsr.ini", "wronggpu-runtime.log", "wronggpu-report.json",
            "crash/wronggpu-crash-20261005-1.txt", "install/install.record", "install/model-source.txt", "game-root/wronggpu-root.log")), str(names))
        t.check("system info: status, FSR DLL, model, GPU", "Game status: Validated" in info and "amd_fidelityfx_dx12.dll: WrongGPU" in info and "ffx_fsr3upscaler_x64.wronggpu-original.dll: the game's" in info and
                "Model data: present and matching" in info and "GPU:" in info, info[:1500])
        t.check("structured diagnostics and support manifest included", "wronggpu-diagnostics.jsonl" in names and "support-manifest.json" in names, str(names))
        if "support-manifest.json" in names:
            support = json.loads(z.read("support-manifest.json"))
            t.check("support report links failure stage/session without asserting cause", support["schema"] == 1 and
                    any(e["session"] == "test-session" and e["data"]["code"] == "WG-HIP-719" and
                        e["data"]["root_cause_proven"] is False for e in support["failures"]), str(support)[:800])
            t.check("support artifacts match their actual bundled hashes", all(hashlib.sha256(z.read(a["name"])).hexdigest() == a["sha256"]
                    for a in support["artifacts"] if a["status"] == "included"))
    os.remove(os.path.join(b, "wronggpu-diagnostics.jsonl"))  # Keep the pre-existing uninstall fixture scope.
    t.check("collect changed nothing", os.path.isfile(os.path.join(b, "wronggpu-setup", "uninstall.exe")))
    write(os.path.join(b, "wronggpu.ini"), "[settings]")  # the player's settings: removed by a full uninstall
    rc = t.uninstall(b)
    t.check("uninstall exit 0", rc == 0, "exit %d; %s" % (rc, t.logtext()[-500:]))
    t.check("uninstall removes the player's wronggpu.ini", not os.path.exists(os.path.join(b, "wronggpu.ini")))
    t.check("folder restored byte for byte (logs and settings removed, state folder gone)", snapshot(root) == before, diff(before, snapshot(root)))
    t.check("Apps & features entry removed", not t.keys_for(b))
    t.check("model data removed with the last game", not os.path.exists(os.path.join(t.appdata(), "model")))
    app = t.appdata()
    for _ in range(50):  # the hidden final cleanup removes %LOCALAPPDATA%\WrongGPU a few seconds after the uninstaller exits
        if not os.path.exists(app):
            break
        time.sleep(0.2)
    t.check("%LOCALAPPDATA%\\WrongGPU removed with the last game", not os.path.exists(app), app)

    print("\n[2] Unreal layout (experimental) and an FSR DLL in a plugin folder")
    root, win64, ffx_dir = unreal(t, "ue", model=GOOD)
    before = snapshot(root)
    rc = t.run("/DIR=" + root, "/FORCE")
    t.check("experimental game refused without /EXPERIMENTAL (exit 3)", rc == 3, "exit %d" % rc)
    t.check("nothing changed", snapshot(root) == before)
    t.check("no model written before the install is confirmed", not os.path.exists(t.model_file()))
    rc = t.run("/EXE=" + os.path.join(root, "MyGame.exe"), "/EXPERIMENTAL", "/FORCE")
    t.check("installed with /EXPERIMENTAL from the bootstrap exe", rc == 0 and "MyGame-Win64-Shipping.exe" in t.logtext(), "exit %d" % rc)
    t.check("payload in Binaries\\Win64 (API DLL only: the game has no FSR 3.0 components)", payload_installed(t, win64, payload_dir, API_ONLY))
    t.check("record says experimental", "status\texperimental" in read(os.path.join(win64, "wronggpu-setup", "install.record")).decode())
    rc = t.uninstall(win64)
    t.check("uninstall restores the Unreal folder byte for byte", rc == 0 and snapshot(root) == before, diff(before, snapshot(root)))
    root, win64, ffx_dir = unreal(t, "ue-plugin", model=GOOD, ffx_in_plugin=True)
    before = snapshot(root)
    rc = t.run("/DIR=" + root, "/EXPERIMENTAL", "/FORCE")
    t.check("FSR DLL outside the exe folder: installed where the DLL is", rc == 0 and payload_installed(t, ffx_dir, payload_dir, API_ONLY) and
            not os.path.exists(os.path.join(win64, "wronggpu-setup")), "exit %d" % rc)
    rc = t.run("/UNINSTALL", "/DIR=" + root)
    for _ in range(50):
        if not os.path.exists(os.path.join(ffx_dir, "wronggpu-setup")):
            break
        time.sleep(0.2)
    t.check("/UNINSTALL /DIR=<game root> finds the install folder and restores it", rc == 0 and snapshot(root) == before, "exit %d; %s" % (rc, diff(before, snapshot(root))))

    print("\n[3] requirement checks: no FSR 3.1, foreign FSR DLL, no GPU, empty folder, bad option")
    root, b = cyberpunk(t, "nofsr", model=GOOD, ffx=None, fsr3=None, fsr3up=None)
    before = snapshot(root)
    rc = t.run("/DIR=" + root, "/FORCE")
    t.check("no FSR DLL at all: refused (exit 2), nothing changed", rc == 2 and snapshot(root) == before and "No AMD FSR 3 DLL found" in t.logtext() and "OptiScaler" in t.logtext(), "exit %d" % rc)
    # Update from an older WrongGPU: settings still at that version's own outdated defaults are brought up to date, the
    # player's own changes stay; a newer old version keeps everything.
    def fake_old_version(b, version, ini):
        rec = os.path.join(b, "wronggpu-setup", "install.record")
        text = read(rec).decode("utf-8")
        write(rec, "\n".join(("version\t" + version) if l.startswith("version\t") else l for l in text.split("\n")).encode("utf-8"))
        write(os.path.join(b, "wronggpu.ini"), ini.encode("utf-8"))
    root, b = cyberpunk(t, "update-030", model=GOOD)
    t.check("update test: first install", t.run("/DIR=" + root, "/FORCE") == 0)
    fake_old_version(b, "0.3.0", "# WrongGPU settings\r\n[fsr]\r\nupscaler = standin\r\nview = depth\r\nlog_dispatches = 1\r\n")
    rc = t.run("/DIR=" + root, "/FORCE")
    ini = read(os.path.join(b, "wronggpu.ini")).decode("utf-8").replace(" ", "")
    t.check("update from 0.3.0: outdated defaults fixed (upscaler, log_dispatches), the player's own view kept",
            rc == 0 and "upscaler=network" in ini and "log_dispatches=0" in ini and "view=depth" in ini and "standin" not in ini, ini)
    fake_old_version(b, "0.3.1", "[fsr]\r\nupscaler = standin\r\n")
    rc = t.run("/DIR=" + root, "/FORCE")
    t.check("update from 0.3.1: a stand-in the player chose is kept", rc == 0 and "upscaler = standin" in read(os.path.join(b, "wronggpu.ini")).decode("utf-8"))
    t.check("...and uninstalled", t.uninstall(b) == 0 and not os.path.exists(os.path.join(b, "wronggpu.ini")))

    # OptiScaler mode: a game without FSR (DLSS only) and the player's own OptiScaler download (a mock of its release)
    opti = os.path.join(t.work, "optiscaler-download")
    write(os.path.join(opti, "OptiScaler.dll"), b"MZ mock OptiScaler")
    write(os.path.join(opti, "OptiScaler.ini"), "[Upscalers]\nDx12Upscaler=auto\n[Libraries]\nFfxDx12SRPath=auto\n"
                                                "[FSR]\nFsr4Update=auto\nUpscalerIndex=auto\n[Menu]\nShortcutKey=auto\n")
    write(os.path.join(opti, "amd_fidelityfx_upscaler_dx12.dll"), b"the player's AMD FSR 4 DLL from OptiScaler")
    write(os.path.join(opti, "setup_windows.bat"), "OptiScaler's own setup script (not placed)")
    root, b = cyberpunk(t, "opti-dlss", model=GOOD, ffx=None, fsr3=None, fsr3up=None)
    before = snapshot(root)
    rc = t.run("/DIR=" + root, "/FORCE", "/EXPERIMENTAL", "/OPTISCALER=" + t.env.win(opti))
    ini = read(os.path.join(b, "OptiScaler.ini")).decode("utf-8", "replace").lower() if os.path.isfile(os.path.join(b, "OptiScaler.ini")) else ""
    t.check("OptiScaler mode: installed (DLSS-only game)", rc == 0, "exit %d; %s" % (rc, t.logtext()[-500:]))
    t.check("...AMD comparison DLL uses the unredirected alias, byte-exact",
            read(os.path.join(b, "wronggpu-amd-compare.dll")) == read(os.path.join(opti, "amd_fidelityfx_upscaler_dx12.dll")) and
            not os.path.exists(os.path.join(b, "amd_fidelityfx_upscaler_dx12.dll")))
    t.check("...OptiScaler.dll as the proxy, WrongGPU's DLL under its own name, the game's files untouched",
            os.path.isfile(os.path.join(b, "dxgi.dll")) and read(os.path.join(b, "dxgi.dll")) == b"MZ mock OptiScaler" and
            read(os.path.join(b, "wronggpu_ffx_dx12.dll")) == read(os.path.join(payload_dir, "amd_fidelityfx_dx12.dll")) and
            not os.path.exists(os.path.join(b, "amd_fidelityfx_dx12.dll")) and not os.path.exists(os.path.join(b, "setup_windows.bat")))
    t.check("...OptiScaler.ini set up for WrongGPU (FSR upscaler = WrongGPU's DLL, its first version, no FSR 4 patch, menu key untouched)",
            "dx12upscaler=fsr31" in ini and "upscalerindex=0" in ini and "fsr4update=false" in ini and "shortcutkey=auto" in ini and
            ("ffxdx12srpath=" + os.path.join(t.env.win(b), "wronggpu_ffx_dx12.dll").lower()) in ini, ini[:400])
    rc = t.uninstall(b)
    t.check("...uninstall removes OptiScaler and WrongGPU byte for byte", rc == 0 and snapshot(root) == before, diff(before, snapshot(root)))
    root, b = cyberpunk(t, "fsr30-only", model=GOOD, ffx=None)
    before = snapshot(root)
    rc = t.run("/DIR=" + root, "/FORCE")
    t.check("FSR 3.0 components only (no FFX API DLL): both replaced", rc == 0 and payload_installed(t, b, payload_dir, COMPONENTS) and
            originals_kept(b, {SLOTS[1]: FSR3, SLOTS[2]: FSR3UP}), "exit %d; %s" % (rc, t.logtext()[-500:]))
    t.check("...and uninstalled byte for byte", t.uninstall(b) == 0 and snapshot(root) == before, diff(before, snapshot(root)))
    root, b = cyberpunk(t, "fsr30-foreign", model=GOOD, fsr3=FOREIGN_FSR3)
    before = snapshot(root)
    rc = t.run("/DIR=" + root, "/FORCE", "/MOVEMODS")
    t.check("ffx_fsr3_x64.dll without the FSR 3.0 exports (another mod): exit 6, nothing changed", rc == 6 and snapshot(root) == before and
            "ffx_fsr3_x64.dll (does not export the FSR 3.0 functions" in t.logtext(), "exit %d" % rc)
    root, b = cyberpunk(t, "foreign", model=GOOD, ffx=FOREIGN)
    before = snapshot(root)
    rc = t.run("/DIR=" + root, "/FORCE", "/MOVEMODS")
    t.check("FSR DLL replaced by another mod: refused (exit 6), nothing changed", rc == 6 and snapshot(root) == before and "amd_fidelityfx_dx12.dll (Some Modder, version 0.9.0.0) is not the game's FSR DLL" in t.logtext(), "exit %d" % rc)
    root, b = cyberpunk(t, "nogpu", model=GOOD)
    before = snapshot(root)
    rc = t.run("/DIR=" + root)
    log = t.logtext()
    if rc == 0 and "(gfx1201)" in log and "AMD HIP runtime works" in log:
        # This PC has a supported card and a working HIP runtime (native run on the test machine): the real checks passed.
        t.check("real GPU and HIP checks pass on this PC (gfx1201)", True)
        t.check("...and the install is removed again", t.uninstall(b) == 0 and snapshot(root) == before)
    else:
        t.check("no RX 9000-series GPU: refused with the spec message (exit 2)", rc == 2 and snapshot(root) == before and
                "WrongGPU needs an AMD Radeon RX 9000-series graphics card." in log, "exit %d; %s" % (rc, log[-600:]))
    root, b = cyberpunk(t, "script-install", model=GOOD)
    write(os.path.join(b, "wronggpu-fsr-install.json"), '{"format": 2}')
    before = snapshot(root)
    rc = t.run("/DIR=" + root, "/FORCE")
    t.check("a test install made with install-wronggpu-fsr.ps1: refused with the way out, nothing changed", rc == 2 and snapshot(root) == before and
            "install-wronggpu-fsr.ps1 -Action Uninstall" in t.logtext(), "exit %d" % rc)
    empty = t.fresh("empty")
    t.check("empty folder: refused (exit 2)", t.run("/DIR=" + empty, "/FORCE") == 2)
    t.check("unknown option: exit 10", t.run("/BOGUS") == 10)
    t.check("uninstall without a record: exit 9", t.run("/UNINSTALL", "/DIR=" + empty) == 9)

    print("\n[4] anti-cheat: refused even with every confirmation")
    root, win64, _ = unreal(t, "ue-eac", model=GOOD)
    write(os.path.join(root, "EasyAntiCheat", "EasyAntiCheat_EOS_Setup.exe"), "eac")
    before = snapshot(root)
    rc = t.run("/DIR=" + root, "/EXPERIMENTAL", "/MOVEMODS", "/REMOVEOLD", "/FORCE")
    t.check("EasyAntiCheat folder: exit 4, nothing changed", rc == 4 and snapshot(root) == before and "Anti-cheat found" in t.logtext(), "exit %d" % rc)
    root, win64, _ = unreal(t, "ue-be", model=GOOD)
    write(os.path.join(win64, "BEService_x64.exe"), "battleye")
    before = snapshot(root)
    t.check("BattlEye next to the exe: exit 4, nothing changed", t.run("/DIR=" + root, "/EXPERIMENTAL", "/FORCE") == 4 and snapshot(root) == before)
    root = t.fresh("Call of Duty Mock")
    write(os.path.join(root, "BlackOpsColdWar.exe"), "cod")
    write(os.path.join(root, "amd_fidelityfx_dx12.dll"), AMD_V1)
    before = snapshot(root)
    t.check("built-in anti-cheat game by name: exit 4", t.run("/DIR=" + root, "/EXPERIMENTAL", "/FORCE") == 4 and snapshot(root) == before)

    print("\n[5] other upscaler mod: list and stop, or move aside with a backup")
    root, b = cyberpunk(t, "opti", model=GOOD)
    write(os.path.join(b, "OptiScaler.ini"), "[Upscalers]")
    write(os.path.join(b, "OptiScaler.log"), "log")
    write(os.path.join(b, "dxgi.dll"), b"proxy of an OptiScaler build")
    write(os.path.join(b, "winmm.dll"), b"unrelated proxy of another tool")
    before = snapshot(root)
    rc = t.run("/DIR=" + root, "/FORCE")
    t.check("without /MOVEMODS: exit 6, files listed, nothing changed", rc == 6 and snapshot(root) == before and "OptiScaler.ini" in t.logtext() and "dxgi.dll" in t.logtext(), "exit %d" % rc)
    rc = t.run("/DIR=" + root, "/FORCE", "/MOVEMODS")
    t.check("with /MOVEMODS: installed", rc == 0, "exit %d; %s" % (rc, t.logtext()[-400:]))
    t.check("mod files moved into the backup, unrelated proxy left alone",
            all(not os.path.exists(os.path.join(b, n)) and read(os.path.join(b, "wronggpu-setup", "backup", n)) == read_snapshot_file(before, n, root, b) for n in ("OptiScaler.ini", "OptiScaler.log", "dxgi.dll"))
            and os.path.isfile(os.path.join(b, "winmm.dll")))
    rc = t.run("/DIR=" + root, "/FORCE")
    t.check("re-install keeps the original files in the backup", rc == 0 and os.path.isfile(os.path.join(b, "wronggpu-setup", "backup", "dxgi.dll")), "exit %d" % rc)
    rc = t.uninstall(b)
    t.check("uninstall puts the mod back byte for byte", rc == 0 and snapshot(root) == before, diff(before, snapshot(root)))

    print("\n[6] older WrongGPU / DLSS 4.5 AMD installs")
    root, b = cyberpunk(t, "legacy", model=GOOD)
    write(os.path.join(b, "winmm.dll"), b"a game file the old install replaced")
    pristine = snapshot(root)
    legacy_install(t, b)
    with_old = snapshot(root)
    rc = t.run("/DIR=" + root, "/FORCE")
    t.check("without /REMOVEOLD: exit 7, nothing changed", rc == 7 and snapshot(root) == with_old, "exit %d" % rc)
    rc = t.run("/DIR=" + root, "/FORCE", "/REMOVEOLD")
    t.check("with /REMOVEOLD: installed", rc == 0, "exit %d; %s" % (rc, t.logtext()[-500:]))
    t.check("old install removed byte-exactly (files, record, log, registry entry)",
            read(os.path.join(b, "winmm.dll")) == b"a game file the old install replaced" and not os.path.exists(os.path.join(b, "dlss45-amd")) and
            not os.path.exists(os.path.join(b, "version.dll")) and not os.path.exists(os.path.join(b, "dlss45-loader.log")) and
            "DLSS45AMD-0000ABCD" not in t.env.reg_keys())
    rc = t.uninstall(b)
    t.check("uninstall leaves the original game (as before the old install)", rc == 0 and snapshot(root) == pristine, diff(pristine, snapshot(root)))
    root, b = cyberpunk(t, "legacy-loose", model=GOOD)
    write(os.path.join(b, "version.dll"), b"manual old loader " + "dlss45-loader.log".encode("utf-16-le"))
    write(os.path.join(b, "dlss45-amd-runtime.dll"), b"old runtime")
    write(os.path.join(b, "dlss45-model", "w.bin"), b"old weights")
    before = snapshot(root)
    t.check("loose old files without a record: exit 7", t.run("/DIR=" + root, "/FORCE") == 7 and snapshot(root) == before)
    rc = t.run("/DIR=" + root, "/FORCE", "/REMOVEOLD")
    t.check("moved into the backup with /REMOVEOLD", rc == 0 and not os.path.exists(os.path.join(b, "version.dll")) and
            os.path.isfile(os.path.join(b, "wronggpu-setup", "backup", "dlss45-model", "w.bin")), "exit %d" % rc)
    rc = t.uninstall(b)
    t.check("uninstall restores them byte for byte", rc == 0 and snapshot(root) == before, diff(before, snapshot(root)))

    print("\n[7] a failure half-way through rolls back completely")
    root, b = cyberpunk(t, "rollback", model=GOOD)
    write(os.path.join(b, "wronggpu_runtime.dll.wronggpu-tmp", "keep.txt"), "in the way")
    before = snapshot(root)
    rc = t.run("/DIR=" + root, "/FORCE")
    t.check("install failed (exit 1)", rc == 1, "exit %d" % rc)
    t.check("game folder rolled back exactly", snapshot(root) == before, diff(before, snapshot(root)))
    t.check("no Apps & features entry left", not t.keys_for(b))

    print("\n[8] game update replaces amd_fidelityfx_dx12.dll")
    root, b = cyberpunk(t, "update", model=GOOD)
    before = snapshot(root)
    t.check("install", t.run("/DIR=" + root, "/FORCE") == 0)
    write(os.path.join(b, "amd_fidelityfx_dx12.dll"), AMD_V2)  # the game's update overwrites WrongGPU's DLL
    rc = t.uninstall(b)
    expect = dict(before)
    expect[key("bin/x64/amd_fidelityfx_dx12.dll")] = sha(AMD_V2)
    t.check("uninstall keeps the game's new DLL, removes everything else", rc == 0 and snapshot(root) == expect, diff(expect, snapshot(root)))
    t.check("the log explains it", "its new file was kept" in t.logtext())
    t.check("re-install over the updated game", t.run("/DIR=" + root, "/FORCE") == 0 and
            read(os.path.join(b, "amd_fidelityfx_dx12.wronggpu-original.dll")) == AMD_V2)
    write(os.path.join(b, "amd_fidelityfx_dx12.dll"), AMD_V1)  # an update after install that happens to restore v1
    t.check("install over an update while installed (record present, AMD DLL in place)", t.run("/DIR=" + root, "/FORCE") == 0 and
            read(os.path.join(b, "amd_fidelityfx_dx12.wronggpu-original.dll")) == AMD_V1)
    rc = t.uninstall(b)
    expect[key("bin/x64/amd_fidelityfx_dx12.dll")] = sha(AMD_V1)
    t.check("and uninstall restores that state", rc == 0 and snapshot(root) == expect, diff(expect, snapshot(root)))
    root, b = cyberpunk(t, "update-fsr30", model=GOOD)
    before = snapshot(root)
    t.check("install (three DLLs)", t.run("/DIR=" + root, "/FORCE") == 0)
    write(os.path.join(b, "ffx_fsr3upscaler_x64.dll"), FSR3UP_V2)  # a game update brings a new FSR 3.0 upscaler build
    rc = t.uninstall(b)
    expect = dict(before)
    expect[key("bin/x64/ffx_fsr3upscaler_x64.dll")] = sha(FSR3UP_V2)
    t.check("game update of ffx_fsr3upscaler_x64.dll: the new file is kept, the other two restored", rc == 0 and snapshot(root) == expect, diff(expect, snapshot(root)))

    print("\n[9] 30-T2: model file search and converter integration (synthetic model files)")
    t.clear_model()
    root, b = cyberpunk(t, "m-none", model=OLD_DLSS)
    before = snapshot(root)
    rc = t.run("/DIR=" + root, "/FORCE", "/NOSCAN")
    log = t.logtext()
    t.check("only an old DLSS file: exit 5, plain explanation, nothing written", rc == 5 and snapshot(root) == before and not os.path.exists(t.model_file()) and
            "version 310.5 or newer" in log and "This file is version 310.1.0.0" in log, "exit %d; %s" % (rc, log[-600:]))
    outside = os.path.join(t.work, "elsewhere", "nvngx_dlss.dll")
    write(outside, GOOD)
    rc = t.run("/DIR=" + root, "/FORCE", "/NOSCAN", "/MODEL=" + outside)
    t.check("a file the player picks: used first", rc == 0 and t.model_ok() and "origin\tchosen by you" in read(os.path.join(t.appdata(), "model", "model-source.txt")).decode(), "exit %d" % rc)
    t.check("the picked file is not copied or changed", read(outside) == GOOD and not any(f.endswith("nvngx_dlss.dll") for f in snapshot(root) if f not in before))
    t.uninstall(b)
    if t.env.native:
        print("  (library search tests skipped on native Windows: they need mock Steam and GOG registry keys; run them under Wine)")
        t.clear_model()
        return run_rest(t, payload_dir, GOOD)
    # Library search: the selected game has only an old file; a Steam library game has a good one; Epic and GOG mocks too.
    t.clear_model()
    common = steam_library(t, {"Cyberpunk 2077": "Cyberpunk 2077", "Other Game": "Other Game"})
    lib_cp = os.path.join(common, "Cyberpunk 2077")
    shutil.rmtree(lib_cp, ignore_errors=True)
    shutil.copytree(cyberpunk(t, "lib-src", model=None)[0], lib_cp)
    write(os.path.join(common, "Other Game", "Game.exe"), "x")
    write(os.path.join(common, "Other Game", "sub", "nvngx_dlss.dll"), GOOD)
    epic_game = os.path.join(t.work, "epic", "EpicGame")
    write(os.path.join(epic_game, "EpicGame.exe"), "epic exe" * 10)
    write(os.path.join(epic_game, "amd_fidelityfx_dx12.dll"), AMD_V1)
    write(os.path.join(t.env.programdata(), "Epic", "EpicGamesLauncher", "Data", "Manifests", "x.item"),
          json.dumps({"DisplayName": "Epic Mock", "InstallLocation": t.env.win(epic_game), "LaunchExecutable": "EpicGame.exe"}, indent=1))
    gog_game = os.path.join(t.work, "gog", "GogGame")
    write(os.path.join(gog_game, "GogGame.exe"), "gog exe" * 10)
    write(os.path.join(gog_game, "amd_fidelityfx_dx12.dll"), AMD_V1)
    t.env.reg("add", r"HKLM\SOFTWARE\GOG.com\Games\1234", "/v", "gameName", "/t", "REG_SZ", "/d", "GOG Mock", "/f", "/reg:32")
    t.env.reg("add", r"HKLM\SOFTWARE\GOG.com\Games\1234", "/v", "path", "/t", "REG_SZ", "/d", t.env.win(gog_game), "/f", "/reg:32")
    rc = t.run("/LIST")
    log = t.logtext()
    t.check("/LIST finds the FSR 3.1 games of Steam, Epic and GOG (not the one without FSR)",
            rc == 0 and "game Cyberpunk 2077 [Steam] VALIDATED" in log and "game Epic Mock [Epic]" in log and "game GOG Mock [GOG]" in log and "Other Game" not in log, log[-1500:])
    root, b = cyberpunk(t, "m-lib", model=OLD_DLSS)
    rc = t.run("/DIR=" + root, "/FORCE")
    src = read(os.path.join(t.appdata(), "model", "model-source.txt")).decode() if os.path.exists(t.model_file()) else ""
    log = t.logtext()
    t.check("model found in another library game after the game's own old file was rejected",
            rc == 0 and t.model_ok() and "origin\tanother game (Other Game)" in src and "rejected" in log and "nvngx_dlss.dll" in log, "exit %d; %s" % (rc, src))
    # Reuse: a second game without any candidate uses the converted model.
    root2, b2 = cyberpunk(t, "m-reuse", model=None)
    rc = t.run("/DIR=" + root2, "/FORCE", "/NOSCAN")
    t.check("second game reuses the model on this PC (no candidate needed)", rc == 0 and "model: reusing" in t.logtext(), "exit %d" % rc)
    rc = t.uninstall(b)
    t.check("uninstalling one game keeps the shared model", rc == 0 and t.model_ok())
    # A damaged model file is not reused.
    data = bytearray(read(t.model_file()))
    data[200] ^= 1
    write(t.model_file(), bytes(data))
    root3, b3 = cyberpunk(t, "m-damaged", model=None)
    rc = t.run("/DIR=" + root3, "/FORCE", "/NOSCAN")
    t.check("damaged model file and no candidate: exit 5", rc == 5, "exit %d" % rc)
    root4, b4 = cyberpunk(t, "m-reconvert", model=GOOD)
    rc = t.run("/DIR=" + root4, "/FORCE", "/NOSCAN")
    t.check("damaged model file is converted again from a candidate", rc == 0 and t.model_ok(), "exit %d" % rc)
    rc = t.uninstall(b2)
    rc4 = t.uninstall(b4)
    t.check("model removed when the last game is uninstalled", rc == 0 and rc4 == 0 and not os.path.exists(t.model_file()))
    t.env.reg("delete", r"HKCU\Software\Valve\Steam", "/f")
    run_rest(t, payload_dir, GOOD)


def run_rest(t, payload_dir, GOOD):  # noqa: N803
    print("\n[10] two games side by side")
    a_root, a = cyberpunk(t, "two-a", model=GOOD)
    b_root, bw, _ = unreal(t, "two-b")
    sa, sb = snapshot(a_root), snapshot(b_root)
    ra = t.run("/DIR=" + a_root, "/FORCE", "/NOSCAN")
    rb = t.run("/DIR=" + b_root, "/FORCE", "/NOSCAN", "/EXPERIMENTAL")
    ka, kb = t.keys_for(a), t.keys_for(bw)
    t.check("both installed, one entry each", ra == 0 and rb == 0 and len(ka) == 1 and len(kb) == 1 and set(ka) != set(kb), "%d %d" % (ra, rb))
    t.check("game A restored, B still installed", t.uninstall(a) == 0 and snapshot(a_root) == sa and payload_installed(t, bw, payload_dir, API_ONLY))
    t.check("game B restored", t.uninstall(bw) == 0 and snapshot(b_root) == sb)

    print("\n[11] uninstall with /OUT collects the logs first")
    root, b = cyberpunk(t, "collect-first", model=GOOD)
    before = snapshot(root)
    t.run("/DIR=" + root, "/FORCE", "/NOSCAN")
    write(os.path.join(b, "wronggpu-frontend.log"), "a log")
    zpath = os.path.join(t.work, "before-uninstall.zip")
    rc = t.uninstall(b, "/OUT=" + zpath)
    t.check("logs zipped, then uninstalled", rc == 0 and os.path.isfile(zpath) and "wronggpu-frontend.log" in zipfile.ZipFile(zpath).namelist() and snapshot(root) == before)


def test_hip_probe(t):
    """The HIP runtime check against mock HIP runtimes built with MinGW: the child-process probe, the architecture
    decision (gfx1201 and gfx1200), LLVM_PATH cleared for the child, and the search order (newest ROCm folder first)."""
    print("\n[12] HIP runtime check")
    mingw = shutil.which("x86_64-w64-mingw32-gcc") or (shutil.which("gcc") if t.env.native else None)
    msvc = shutil.which("cl") if t.env.native else None
    if not mingw and not msvc:
        print("  (skipped: no C compiler for the mock HIP runtime: MinGW gcc, or MSVC cl from a developer prompt)")
        return
    out = os.path.join(t.work, "hip")
    mocks = {"rx9070": ("AMD Radeon RX 9070 XT", "gfx1201"), "rx9060": ("AMD Radeon RX 9060 XT", "gfx1200"), "rx7900": ("AMD Radeon RX 7900 XTX", "gfx1100")}
    mocks.update({"unknown": ("AMD Radeon RX 9070 XT", ""), "unsupported": ("AMD Radeon RX 9070 XT", "gfx1299")})
    for name, (device, arch) in mocks.items():
        d = os.path.join(out, name)
        os.makedirs(d, exist_ok=True)
        src = os.path.join(d, "mock.c")  # the variant's settings, then the mock itself
        write(src, '#define MOCK_DEVICE_NAME "%s"\n#define MOCK_ARCH "%s"\n#include "%s"\n' % (device, arch, os.path.join(HERE, "mock_hip.c").replace("\\", "/")))
        dll_path = os.path.join(d, "amdhip64_7.dll")
        if mingw:
            subprocess.run([mingw, "-shared", "-O2", src, "-o", dll_path], check=True)
        else:
            subprocess.run([msvc, "/nologo", "/LD", "/O2", src, "/Fe" + dll_path, "/Fo" + d + os.sep], check=True, cwd=d)

    def dll(name):
        return os.path.join(out, name, "amdhip64_7.dll")

    def probe(path, **env):
        env.setdefault("LLVM_PATH", "")
        return t.env.run([t.setup, "/HIPPROBE=" + t.env.win(path), "/LOG=" + t.env.win(t.log)], extra_env=env)

    t.check("probe: RX 9070 XT, gfx1201, context created (0)", probe(dll("rx9070")) == 0)
    t.check("probe: RX 9060 XT, gfx1200, context created (0)", probe(dll("rx9060")) == 0)
    t.check("probe: RX 7900 XTX, gfx1100 (14)", probe(dll("rx7900")) == 14)
    t.check("probe: runtime missing (11)", probe(os.path.join(out, "missing", "amdhip64_7.dll")) == 11)
    t.check("probe: a DLL without the HIP functions (11)", probe(t.setup) == 11)
    t.check("mock check: with LLVM_PATH set the mock runtime fails hipInit (12)", probe(dll("rx9070"), LLVM_PATH="C:\\llvm-elsewhere") == 12)
    if os.path.exists(t.log):
        os.remove(t.log)
    rc = t.env.run([t.setup, "/S", "/HIPCHECK", "/LOG=" + t.env.win(t.log)], extra_env={"WRONGGPU_HIP_RUNTIME": t.env.win(dll("rx9070")), "LLVM_PATH": "C:\\llvm-elsewhere"})
    log = t.logtext()
    t.check("/HIPCHECK clears LLVM_PATH for the child: OK, device and gfx1201 reported", rc == 0 and "OK - AMD Radeon RX 9070 XT (gfx1201)" in log, "exit %d; %s" % (rc, log[-600:]))
    if os.environ.get("WRONGGPU_TESTING") == "1":
        for first in ("rx7900", "unknown", "unsupported"):
            if os.path.exists(t.log):
                os.remove(t.log)
            candidates = ";".join(t.env.win(dll(n)) for n in (first, "rx9060"))
            rc = t.env.run([t.setup, "/S", "/HIPCHECK", "/LOG=" + t.env.win(t.log)], extra_env={"WRONGGPU_TEST_HIP_CANDIDATES": candidates})
            log = t.logtext()
            t.check("candidate retry after " + first, rc == 0 and "OK - AMD Radeon RX 9060 XT (gfx1200)" in log and dll(first) in log)
        rc = t.env.run([t.setup, "/S", "/HIPCHECK", "/LOG=" + t.env.win(t.log)], extra_env={"WRONGGPU_TEST_HIP_CANDIDATES": t.env.win(dll("unsupported"))})
        t.check("all candidates unsupported: still refused", rc == 2)
    if t.env.native:
        print("  (ROCm search-order test skipped on native Windows: it writes to C:\\Program Files)")
        return
    rocm = os.path.join(t.env.prefix, "drive_c", "Program Files", "AMD", "ROCm")
    shutil.rmtree(rocm, ignore_errors=True)
    for ver, name in (("6.4", "rx9060"), ("7.2", "rx7900"), ("7.10", "rx9070")):
        write(os.path.join(rocm, ver, "bin", "amdhip64_7.dll"), read(dll(name)))
    if os.path.exists(t.log):
        os.remove(t.log)
    rc = t.env.run([t.setup, "/S", "/HIPCHECK", "/LOG=" + t.env.win(t.log)])
    log = t.logtext()
    t.check("search order: the newest ROCm folder (7.10, not 7.2) is tried first", rc == 0 and "ROCm\\7.10\\bin\\amdhip64_7.dll" in log and "ROCm\\7.2\\" not in log,
            "exit %d; %s" % (rc, log[-600:]))
    shutil.rmtree(os.path.join(rocm, "7.10"))
    if os.path.exists(t.log):
        os.remove(t.log)
    rc = t.env.run([t.setup, "/S", "/HIPCHECK", "/LOG=" + t.env.win(t.log)])
    log = t.logtext()
    t.check("an older Radeon through the HIP runtime: refused, the card named", rc == 2 and "AMD Radeon RX 7900 XTX (gfx1100) is not supported" in log, "exit %d; %s" % (rc, log[-600:]))
    shutil.rmtree(rocm)


FILTER_NOTE = "The output filter of this DLSS build is not known to this version; WrongGPU's own filter will be used."
DTYPE_CODE = {"float32": 0, "uint8": 1, "int32": 2}


def build_container(case, map_sha):
    """A model file written by this test from clean/converter/OUTPUT-FORMAT.md and the reference tensors of a converter
    test case (independent of the converter's own writer). The optional group is in it exactly when the case's source
    has the block. map_sha: the header's hash of the map file (any 32 bytes for "written with another map")."""
    stages = [st for st in case.map["stages"] if st["stage"] in case.expected]
    datas = [data for st in stages for _, data in case.expected[st["stage"]]]

    def index(offsets):
        out, i = bytearray(), 0
        for st in stages:
            name = st["stage"].encode()
            digest = st["digest"] if isinstance(st["digest"], str) else st["digest"][case.found[st["stage"]]]
            out += struct.pack("<H", len(name)) + name + bytes.fromhex(digest) + struct.pack("<I", len(st["tensors"]))
            for mt, (tname, data) in zip(st["tensors"], case.expected[st["stage"]]):
                n = tname.encode()
                out += struct.pack("<H", len(n)) + n + bytes([DTYPE_CODE[mt["dtype"]], len(mt["shape"])])
                out += struct.pack("<%dI" % len(mt["shape"]), *mt["shape"]) + struct.pack("<QQ", offsets[i], len(data))
                i += 1
        return bytes(out)

    def align(n):
        return (n + 63) // 64 * 64

    index_size = len(index([0] * len(datas)))
    pos, offsets = align(88 + index_size), []
    for data in datas:
        offsets.append(pos)
        pos = align(pos + len(data))
    body = bytearray(b"WGCMODEL" + struct.pack("<II", 1, len(stages)) + case.key + map_sha + struct.pack("<Q", index_size) + index(offsets))
    for off, data in zip(offsets, datas):
        body += bytes(off - len(body)) + data
    body += bytes(pos - len(body))
    return bytes(body) + hashlib.sha256(body).digest()


def test_filter_group(t):
    """Spec 08 draft 2: the optional output-filter weights (source F, group `filter` of the model M file), with the
    version-4 test map this installer was built with. Every source is synthetic and made by the converter's test code:
    the same fake parameter data with the block of variant 1 (the suite's default source), of variant 2, or without a
    block. An "earlier" model file is written by the test itself with the version-1 map of the same data."""
    print("\n[13] output-filter weights: fresh installs, reuse, converting an earlier model file again, several DLLs")
    with_f, with_f2, none = t.model, cases.f_mini_case(present=1), cases.f_mini_case(present=None)
    assert with_f.map == with_f2.map == none.map and "filter" in with_f.expected and "filter" not in none.expected
    none2 = none.file + b"another synthetic file without the block"
    v1 = cases.mini_case()  # the version-1 map of the same data: no source F, no group
    map_sha = hashlib.sha256(read(os.path.join(t.work, "prepared", "test-map.json"))).digest()
    earlier = build_container(v1, hashlib.sha256(json.dumps(v1.map, indent=1).encode()).digest())
    ref_with, ref_with2, ref_none = (build_container(c, map_sha) for c in (with_f, with_f2, none))
    other_map_with = build_container(with_f, hashlib.sha256(b"another version-4 map").digest())
    assert len({earlier, ref_with, ref_with2, ref_none, other_map_with}) == 5
    mdir = os.path.join(t.appdata(), "model")
    root, b = cyberpunk(t, "filter")
    game_dll = os.path.join(b, "nvngx_dlss.dll")
    picked = os.path.join(t.work, "elsewhere", "filter", "nvngx_dlss.dll")
    earlier_source = "source\tC:\\earlier\\nvngx_dlss.dll\r\nversion\t310.5.0.0\r\norigin\tthis game\r\n"
    state = {}

    def install(label, existing, in_game, pick=None, env=None, block=False):
        """One silent install into the same mock game: `existing` is the model file on the PC before it (None: none),
        `in_game` the game's own DLSS file (None: none), `pick` the file given with /MODEL=. block: a file sits where
        the installer would stage a replacement for the model file."""
        t.clear_model()
        if existing is not None:
            write(t.model_file(), existing)
            write(os.path.join(mdir, "model-source.txt"), earlier_source)
        if block:
            write(os.path.join(mdir, "model-new"), "a file in the way")
        if os.path.exists(game_dll):
            os.remove(game_dll)
        if in_game is not None:
            write(game_dll, in_game)
        args = ["/DIR=" + root, "/FORCE", "/NOSCAN", *t.route]
        if pick is not None:
            write(picked, pick)
            args.append("/MODEL=" + picked)
        rc = t.run(*args, extra_env=env)
        log = t.logtext()
        rec = read(os.path.join(b, "wronggpu-setup", "install.record")).decode() if rc == 0 else ""
        state.update(log=log, case=next((l.split("\t", 1)[1] for l in rec.splitlines() if l.startswith("model-m\t")), ""),
                     source=read(os.path.join(mdir, "model-source.txt")).decode() if os.path.isfile(os.path.join(mdir, "model-source.txt")) else "",
                     model=read(t.model_file()) if os.path.isfile(t.model_file()) else b"",
                     beside=sorted(n for n in os.listdir(mdir) if n not in ("model.wgcm", "model-source.txt", "K")) if os.path.isdir(mdir) else [])
        t.check(label + ": the install succeeds", rc == 0, "exit %d; %s" % (rc, log[-900:]))
        return state

    def says(text, path=None):
        """`text` (with the Windows form of `path` for %s) is in the last install's log."""
        return (text % t.env.win(path) if path else text).lower() in state["log"].lower()

    def note_shown():
        return "warning: " + FILTER_NOTE in state["log"]

    def conversions():  # model M conversions that passed every digest (preset K has no source in these files)
        return state["log"].count("stage digests match")

    def detail():
        return "case: %s | beside: %s | %s" % (state["case"], state["beside"], state["log"][-1200:])

    def collect(name):
        zpath = os.path.join(t.work, name)
        rc = t.run("/COLLECT", "/DIR=" + b, "/OUT=" + zpath)
        return zipfile.ZipFile(zpath).read("system-info.txt").decode("utf-8", "replace") if rc == 0 and os.path.isfile(zpath) else ""

    # Fresh installs: the group is there when the source has the block, with the digest of the variant found.
    s = install("fresh install, source with the block", None, with_f.file)
    t.check("...the model file has the group `filter`, byte for byte this test's reference container", s["model"] == ref_with, detail())
    taps = [st for st in read_container(t.model_file())[2] if st[0] == "filter"] if s["model"] else []
    t.check("...output_taps is five float32 with the bits of the source's five fields",
            len(taps) == 1 and taps[0][2] == [("output_taps", 0, [5], struct.pack("<5I", *with_f.variants[0]["elements"]))])
    t.check("...no note, and neither the log nor the record shows a weight",
            not note_shown() and FILTER_NOTE not in s["log"] and not any(("%08x" % x) in (s["log"] + s["case"] + s["source"]).lower() for x in with_f.variants[0]["elements"]), detail())
    t.check("...install record and model-source.txt say: converted, output filter from the DLSS file",
            s["case"].lower() == ("converted from %s; output filter: from the DLSS file" % t.env.win(game_dll)).lower() and "filter\tfrom this file" in s["source"], detail())
    s = install("existing model with the group, written with this map", ref_with, None)
    t.check("...reused without a conversion", says("model: reusing") and conversions() == 0 and s["model"] == ref_with and
            s["case"] == "reused; output filter: from the DLSS file" and s["source"] == earlier_source, detail())
    info = collect("filter-reused.zip")
    t.check("...support report: where the output filter comes from and what the last install did",
            "Model M output filter: from the DLSS file (weights in the Model M file); file written with this installer's converter data" in info and
            "Model M file at the last install: reused; output filter: from the DLSS file" in info, info[:1500])
    s = install("fresh install, source with the block of variant 2", None, with_f2.file)
    t.check("...the group has variant 2's digest and numbers", s["model"] == ref_with2 and t.model_ok(with_f2) and not note_shown(), detail())
    s = install("fresh install, source without the block", None, none.file)
    t.check("...the model file has no group and is this test's reference container", s["model"] == ref_none and t.model_ok(none), detail())
    t.check("...the player's line is shown once in the result (not only by the converter's log)", note_shown() and s["log"].count("warning: " + FILTER_NOTE) == 1, detail())
    t.check("...install record and model-source.txt say so",
            s["case"].lower() == ("converted from %s; output filter: WrongGPU's own (not known for this DLSS build)" % t.env.win(game_dll)).lower() and
            "filter\tnot known for this file" in s["source"], detail())

    # An earlier model file (version-1 map, no group) is valid but is converted again where that can add the group.
    s = install("earlier model file + source with the block", earlier, with_f.file)
    t.check("...converted again: the model file now has the group (reference container)", s["model"] == ref_with and conversions() == 1, detail())
    checked, replaced = s["log"].find("passed its check"), s["log"].find("replaced by the new file")
    t.check("...the earlier file was replaced only after the new file passed its check, and nothing is left beside it",
            0 <= checked < replaced and s["beside"] == [], detail())
    t.check("...install record: replaced, and why; model-source.txt names the new source; no note",
            s["case"].lower() == ("replaced: the file has no output-filter weights; converted again from %s; output filter: from the DLSS file" % t.env.win(game_dll)).lower() and
            ("source\t%s\r\n" % t.env.win(game_dll)).lower() in s["source"].lower() and "filter\tfrom this file" in s["source"] and not note_shown(), detail())
    s = install("earlier model file + source without the block", earlier, none.file)
    t.check("...the earlier file is kept byte for byte, with its model-source.txt", s["model"] == earlier and s["source"] == earlier_source and s["beside"] == [], detail())
    t.check("...the player's line is shown", note_shown(), detail())
    t.check("...install record: kept, and why",
            s["case"].lower() == ("kept: the file has no output-filter weights; the DLSS file found (%s) converts without output-filter weights; output filter: WrongGPU's own" % t.env.win(game_dll)).lower(), detail())
    info = collect("filter-kept.zip")
    t.check("...support report says the same",
            "Model M output filter: WrongGPU's own (no weights in the Model M file); file written with other converter data" in info and
            "Model M file at the last install: kept: the file has no output-filter weights" in info, info[:1500])
    s = install("earlier model file + no source", earlier, None)
    t.check("...the earlier file is kept; no note (no DLSS build was looked at)", s["model"] == earlier and not note_shown() and FILTER_NOTE not in s["log"] and
            s["case"] == "kept: the file has no output-filter weights; no usable DLSS file was found to convert it again; output filter: WrongGPU's own", detail())
    s = install("earlier model file + only a DLSS file without the model", earlier, OLD_DLSS)
    t.check("...the earlier file is kept, the rejection is logged", s["model"] == earlier and says("model: rejected") and s["case"].startswith("kept: ") and not note_shown(), detail())

    # A new conversion that cannot be saved, or that is damaged on disk, never costs the player the working model.
    for label, env, block, why in (
            ("earlier model file + a new conversion that cannot be saved beside it", None, True, "it could not be saved next to the model data in use"),
            ("earlier model file + a new conversion damaged before its check", {"WRONGGPU_TEST_DAMAGE_NEW_MODEL": "1"}, False, "the new file failed its check")):
        if env and os.environ.get("WRONGGPU_TESTING") != "1":
            print("  (skipped: %s needs an installer built with WRONGGPU_TESTING)" % label)
            continue
        s = install(label, earlier, with_f.file, env=env, block=block)
        t.check("...the earlier file is intact, with its model-source.txt",
                s["model"] == earlier and s["source"] == earlier_source and conversions() == 1 and "replaced by the new file" not in s["log"], detail())
        t.check("...install record: kept, and why; nothing of the attempt is left",
                s["case"].startswith("kept: the new conversion from ") and why in s["case"] and s["case"].endswith("output filter: WrongGPU's own") and
                s["beside"] == (["model-new"] if block else []), detail())

    # A file that was written from a source without the block with this map, and one with the group from another map.
    s = install("model file of this map without the group + source without the block", ref_none, none.file)
    t.check("...kept, the player's line again", s["model"] == ref_none and note_shown() and s["case"].startswith("kept: the file has no output-filter weights; the DLSS file found ("), detail())
    s = install("model file of this map without the group + source with the block", ref_none, with_f.file)
    t.check("...converted again and replaced", s["model"] == ref_with and s["case"].startswith("replaced: the file has no output-filter weights; converted again from ") and not note_shown(), detail())
    s = install("model file with the group from other converter data + source without the block", other_map_with, none.file)
    t.check("...kept, and no note: the file has the weights",
            s["model"] == other_map_with and not note_shown() and s["case"].startswith("kept: the file was written with other converter data; ") and
            s["case"].endswith("output filter: from the DLSS file"), detail())
    s = install("model file with the group from other converter data + source with the block", other_map_with, with_f.file)
    t.check("...converted again and replaced (the header now names this map)", s["model"] == ref_with and
            s["case"].startswith("replaced: the file was written with other converter data; converted again from "), detail())

    # Several DLLs (spec 08 section 8): the first that converts with the group; if none has it, the first that converts.
    s = install("[picked file without the block, game's file with it]", None, with_f.file, pick=none.file)
    t.check("...the second is used: group present, no note", s["model"] == ref_with and conversions() == 2 and not note_shown(), detail())
    t.check("...which file was used is in the log, the install record and model-source.txt (with the file passed over)",
            says("model: using %s", game_dll) and s["case"].lower().startswith(("converted from %s;" % t.env.win(game_dll)).lower()) and
            ("source\t%s\r\n" % t.env.win(game_dll)).lower() in s["source"].lower() and "origin\tthis game" in s["source"] and
            ("not-used\t%s (converts, without output-filter weights)" % t.env.win(picked)).lower() in s["source"].lower(), detail() + s["source"])
    s = install("[picked file with the block, game's file without it]", None, none.file, pick=with_f.file)
    t.check("...the first is used and the second is not converted", s["model"] == ref_with and conversions() == 1 and not note_shown() and
            ("source\t%s\r\n" % t.env.win(picked)).lower() in s["source"].lower() and "origin\tchosen by you" in s["source"] and "not-used" not in s["source"], detail() + s["source"])
    s = install("[two files without the block]", None, none2, pick=none.file)
    t.check("...both are tried, the first is used, the player's line is shown", s["model"] == ref_none and conversions() == 2 and note_shown() and
            says("model: using %s", picked) and ("source\t%s\r\n" % t.env.win(picked)).lower() in s["source"].lower() and
            ("not-used\t%s (converts, without output-filter weights)" % t.env.win(game_dll)).lower() in s["source"].lower(), detail() + s["source"])
    s = install("earlier model file + [picked file without the block, game's file with it]", earlier, with_f.file, pick=none.file)
    t.check("...the earlier file is replaced from the file that has the block", s["model"] == ref_with and
            s["case"].lower().startswith(("replaced: the file has no output-filter weights; converted again from %s;" % t.env.win(game_dll)).lower()), detail())
    t.check("the DLSS files were only read", read(game_dll) == with_f.file and read(picked) == none.file)
    rc = t.uninstall(b)
    t.check("uninstall after these installs; the model data goes with the last game", rc == 0 and not os.path.exists(mdir), "exit %d" % rc)


def test_upgrade_paths(t, payload_dir):
    """Setup run over an earlier install (README "Install", "Uninstall"): WrongGPU's own files replaced by hand since
    that install (a newer build dropped into the folder), a game update, an upgrade that fails, and WrongGPU's own
    earlier OptiScaler-hosted install with the logs its OptiScaler wrote."""
    print("\n[14] setup over an earlier install: WrongGPU's files replaced by hand, game update, failures, OptiScaler's logs")
    GOOD = t.model.file  # noqa: N806
    t.clear_model()
    game = {SLOTS[0]: AMD_V1, SLOTS[1]: FSR3, SLOTS[2]: FSR3UP}
    # A newer WrongGPU build placed by hand: WrongGPU's version resource, but not the files the install record names. Under
    # the game's FSR names (the original comes back in their place). 0.6.1: a hand-placed file under another name counts as
    # WrongGPU's only by a release's hash (the hand-placed-runtime case below: left in place).
    hot = {SLOTS[0]: mockpe.make_dll(mockpe.FRONT_END, (0, 5, 0, 0), extra=b"hand-placed front end", exports=mockpe.FFX_API_EXPORTS, dll_name=SLOTS[0]),
           SLOTS[2]: mockpe.make_dll(mockpe.FRONT_END, (0, 5, 0, 0), extra=b"hand-placed forwarder", exports=mockpe.FSR3UPSCALER_EXPORTS, dll_name=SLOTS[2])}

    def same(path, data):
        return os.path.isfile(path) and read(path) == data

    def kept(b, expected):  # originals_kept, false instead of an error when a file is gone
        return all(same(os.path.join(b, n[:-4] + ".wronggpu-original.dll"), data) for n, data in expected.items())

    def installed(name, hand_placed=False):
        root, b = cyberpunk(t, name, model=GOOD)
        before = snapshot(root)
        rc = t.run("/DIR=" + root, "/FORCE", "/NOSCAN")
        t.check(name + ": first install", rc == 0, "exit %d; %s" % (rc, t.logtext()[-500:]))
        for n, data in hot.items() if hand_placed else ():
            write(os.path.join(b, n), data)
        return root, b, before

    def setup_again(root, env=None):
        rc = t.run("/DIR=" + root, "/FORCE", "/NOSCAN", extra_env=env)
        return rc, t.logtext()

    def tail(rc, log, root=None, want=None):
        return "exit %d; %s%s" % (rc, diff(want, snapshot(root)) + "; " if root else "", log[-700:])

    # Defect of 0.4.5-rc: the restore kept the hand-placed DLL, removed the game's renamed original, then refused.
    root, b, before = installed("hotfix-upgrade", hand_placed=True)
    rc, log = setup_again(root)
    t.check("setup over hand-placed WrongGPU DLLs (front end, a forwarder): installed", rc == 0, tail(rc, log))
    t.check("...the new payload is in place and the game's three FSR DLLs are kept as *.wronggpu-original.dll", payload_installed(t, b, payload_dir) and kept(b, game), log[-700:])
    rec_path = os.path.join(b, "wronggpu-setup", "install.record")
    rec = read(rec_path).decode() if os.path.isfile(rec_path) else ""
    t.check("...the hand-placed files are not taken for the game's: the runtime is recorded as new, the backup holds AMD's DLL",
            any(l.startswith("copy\t0\t-\t") and l.endswith("\twronggpu_runtime.dll") for l in rec.splitlines()) and
            same(os.path.join(b, "wronggpu-setup", "backup", SLOTS[0]), AMD_V1) and not os.path.exists(os.path.join(b, "wronggpu-setup", "backup", "wronggpu_runtime.dll")), rec[:900])
    rc = t.uninstall(b) if rec else -1
    t.check("...and uninstall restores the game byte for byte, nothing of WrongGPU left", rc == 0 and snapshot(root) == before, tail(rc, t.logtext(), root, before))
    root, b, before = installed("hotfix-uninstall", hand_placed=True)
    rc = t.uninstall(b)
    t.check("uninstall with hand-placed WrongGPU DLLs: the game's originals are back, nothing of WrongGPU left", rc == 0 and snapshot(root) == before, tail(rc, t.logtext(), root, before))
    # 0.6.1: a hand-placed runtime that names WrongGPU but has no release's hash is left in place with a warning
    root, b, before = installed("hand-placed-runtime")
    hand_runtime = mockpe.make_dll({"ProductName": "WrongGPU", "FileDescription": "WrongGPU runtime"}, (0, 5, 0, 0), extra=b"hand-placed runtime")
    write(os.path.join(b, "wronggpu_runtime.dll"), hand_runtime)
    rc = t.uninstall(b)
    expect = dict(before)
    expect[key("bin/x64/wronggpu_runtime.dll")] = sha(hand_runtime)
    t.check("uninstall with a hand-placed runtime (WrongGPU's version resource, not a release's hash): left in place, the rest as before",
            rc == 0 and snapshot(root) == expect and "changed since install" in t.logtext(), tail(rc, t.logtext(), root, expect))
    # The game's original is never removed while WrongGPU's DLL is under the game's name: without a backup, and with a
    # record that has no entry for WrongGPU's DLL, it comes back from its renamed copy (hash-checked against the record).
    root, b, before = installed("hotfix-no-backup", hand_placed=True)
    shutil.rmtree(os.path.join(b, "wronggpu-setup", "backup"))
    rc = t.uninstall(b)
    t.check("...the same without the backup folder: the originals come back from their renamed copies", rc == 0 and snapshot(root) == before, tail(rc, t.logtext(), root, before))
    root, b, before = installed("record-without-entry")
    rec_path = os.path.join(b, "wronggpu-setup", "install.record")
    write(rec_path, "\r\n".join(l for l in read(rec_path).decode().split("\r\n") if not (l.startswith("copy\t") and l.endswith("\t" + SLOTS[0]))))
    rc = t.uninstall(b)
    t.check("a record without the entry for WrongGPU's amd_fidelityfx_dx12.dll: uninstall puts the game's DLL back from its renamed copy",
            rc == 0 and snapshot(root) == before, tail(rc, t.logtext(), root, before))

    # Control: files that are not WrongGPU's keep the rule for game updates (the new file stays; 0.6.1: the old copy is kept
    # in WrongGPU-originals, never deleted).
    root, b, before = installed("update-upgrade")
    write(os.path.join(b, SLOTS[0]), AMD_V2)
    write(os.path.join(b, SLOTS[2]), FSR3UP_V2)
    rc, log = setup_again(root)
    t.check("control - a game update replaced two FSR DLLs, then setup again: installed over the game's new files",
            rc == 0 and log.count("its new file was kept") == 2 and payload_installed(t, b, payload_dir) and kept(b, {SLOTS[0]: AMD_V2, SLOTS[1]: FSR3, SLOTS[2]: FSR3UP_V2}), tail(rc, log))
    expect = dict(before)
    expect[key("bin/x64/" + SLOTS[0])], expect[key("bin/x64/" + SLOTS[2])] = sha(AMD_V2), sha(FSR3UP_V2)
    rc = t.uninstall(b)
    outside = {k: v for k, v in snapshot(root).items() if "wronggpu-originals" not in k}
    kept_old = sorted(v for k, v in snapshot(root).items() if "/wronggpu-originals/" in k and not k.endswith("/"))
    t.check("...and uninstall leaves the updated game; the two DLLs from before WrongGPU are kept in WrongGPU-originals",
            rc == 0 and outside == expect and kept_old == sorted([sha(AMD_V1), sha(FSR3UP)]), tail(rc, t.logtext(), root, expect))
    root, b, before = installed("foreign-upgrade")
    write(os.path.join(b, SLOTS[0]), FOREIGN)
    state = snapshot(root)
    rc = t.run("/DIR=" + root, "/FORCE", "/NOSCAN", "/MOVEMODS")
    t.check("control - another mod replaced WrongGPU's DLL after the install: setup refused (exit 6), nothing changed", rc == 6 and snapshot(root) == state, tail(rc, t.logtext(), root, state))
    write(os.path.join(b, SLOTS[0]), read(os.path.join(payload_dir, SLOTS[0])))
    t.check("...with WrongGPU's DLL back the install is removed as usual", t.uninstall(b) == 0 and snapshot(root) == before, diff(before, snapshot(root)))

    # A failed upgrade leaves the game's original files (the earlier install is gone, nothing of the new one stays).
    root, b, before = installed("hotfix-upgrade-fails", hand_placed=True)
    write(os.path.join(b, "wronggpu_runtime.dll.wronggpu-tmp", "keep.txt"), "in the way")  # the new runtime cannot be written
    expect = dict(before)
    expect[key("bin/x64/wronggpu_runtime.dll.wronggpu-tmp") + "/"] = "dir"
    expect[key("bin/x64/wronggpu_runtime.dll.wronggpu-tmp/keep.txt")] = sha(b"in the way")
    rc, log = setup_again(root)
    t.check("an upgrade that fails while the new files are written (exit 1): the game's original files, nothing of WrongGPU, no entry",
            rc == 1 and snapshot(root) == expect and not t.keys_for(b), tail(rc, log, root, expect))
    if os.environ.get("WRONGGPU_TESTING") == "1":
        root, b, before = installed("hotfix-upgrade-stops", hand_placed=True)
        rc, log = setup_again(root, {"WRONGGPU_TEST_FAIL_AFTER_RESTORE": "1"})
        t.check("an upgrade that stops right after the earlier install was removed (exit 1): the game's original files, no state folder, no entry",
                rc == 1 and snapshot(root) == before and not t.keys_for(b), tail(rc, log, root, before))
    else:
        print("  (skipped: the upgrade that stops after the earlier install was removed needs an installer built with WRONGGPU_TESTING)")
    if t.env.native:  # a file of the earlier install that can be read but not replaced (open elsewhere)
        import ctypes
        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        k32.CreateFileW.restype = ctypes.c_void_p
        k32.CreateFileW.argtypes = [ctypes.c_wchar_p, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_void_p]
        k32.CloseHandle.argtypes = [ctypes.c_void_p]
        root, b, before = installed("hotfix-upgrade-in-use", hand_placed=True)
        handle = k32.CreateFileW(os.path.join(b, SLOTS[0]), 0x80000000, 1, None, 3, 0, None)  # GENERIC_READ, FILE_SHARE_READ, OPEN_EXISTING
        try:
            rc, log = setup_again(root)
        finally:
            k32.CloseHandle(handle)
        t.check("a file of the earlier install is in use (exit 1): no original of the game is lost, the record and the backup stay",
                handle not in (None, ctypes.c_void_p(-1).value) and rc == 1 and all(same(os.path.join(b, n), d) or kept(b, {n: d}) for n, d in game.items()) and
                os.path.isfile(os.path.join(b, "wronggpu-setup", "install.record")) and same(os.path.join(b, "wronggpu-setup", "backup", SLOTS[0]), AMD_V1), tail(rc, log))
        rc, log = setup_again(root)
        t.check("...setup run again finishes the upgrade", rc == 0 and payload_installed(t, b, payload_dir) and kept(b, game), tail(rc, log))
        rc = t.uninstall(b) if rc == 0 else -1
        t.check("...and uninstall restores the game byte for byte", rc == 0 and snapshot(root) == before, tail(rc, t.logtext(), root, before))

    # WrongGPU's own OptiScaler-hosted install: the logs its OptiScaler wrote are not "another mod".
    opti = os.path.join(t.work, "optiscaler-download")
    write(os.path.join(opti, "OptiScaler.dll"), b"MZ mock OptiScaler")
    write(os.path.join(opti, "OptiScaler.ini"), "[Upscalers]\nDx12Upscaler=auto\n[Libraries]\nFfxDx12SRPath=auto\n[FSR]\nFsr4Update=auto\nUpscalerIndex=auto\n")
    root, b = cyberpunk(t, "opti-upgrade", model=GOOD, ffx=None, fsr3=None, fsr3up=None)
    before = snapshot(root)
    hosted = ("/DIR=" + root, "/FORCE", "/NOSCAN", "/EXPERIMENTAL", "/OPTISCALER=" + t.env.win(opti))
    rc = t.run(*hosted)
    t.check("OptiScaler-hosted install", rc == 0, "exit %d; %s" % (rc, t.logtext()[-500:]))
    run_time = ("OptiScaler.log", "OptiScaler.previous.log", "fakenvapi.log", "wronggpu-fsr.log")
    for n in run_time:
        write(os.path.join(b, n), "written while the game ran")
    rc = t.run(*hosted)
    log = t.logtext()
    t.check("setup over WrongGPU's own OptiScaler-hosted install: no /MOVEMODS needed, no warning about another mod",
            rc == 0 and "No other upscaler mod or older install found" in log and "needs your OK" not in log and "Other upscaler mods" not in log, tail(rc, log))
    rec = read(os.path.join(b, "wronggpu-setup", "install.record")).decode()
    t.check("...nothing was moved aside as another mod's file, and OptiScaler's logs of the earlier install are gone",
            "\ndisable\t" not in rec and not any(os.path.exists(os.path.join(b, n)) for n in run_time), rec[:900])
    rc = t.uninstall(b)
    t.check("...and uninstall restores the game byte for byte", rc == 0 and snapshot(root) == before, tail(rc, t.logtext(), root, before))
    root, b, before = installed("foreign-opti-upgrade")
    foreign = {"OptiScaler.ini": b"[Upscalers]", "OptiScaler.log": b"log", "dxgi.dll": b"proxy of an OptiScaler build"}
    for n, data in foreign.items():
        write(os.path.join(b, n), data)
    state = snapshot(root)
    rc, log = setup_again(root)
    t.check("control - an OptiScaler that is not in WrongGPU's install record: exit 6 without /MOVEMODS as before, nothing changed",
            rc == 6 and snapshot(root) == state and "OptiScaler.log" in log, tail(rc, log, root, state))
    for n in foreign:
        os.remove(os.path.join(b, n))
    t.check("...and without it the install is removed as usual", t.uninstall(b) == 0 and snapshot(root) == before, diff(before, snapshot(root)))
    t.clear_model()


def read_snapshot_file(_before, name, _root, b):
    return {"OptiScaler.ini": b"[Upscalers]", "OptiScaler.log": b"log", "dxgi.dll": b"proxy of an OptiScaler build"}[name]


PRESET_K_MARK = "Start with Preset K"  # the wizard's second model choice: only in an installer with Preset K


def has_preset_k(setup):
    """Whether the installer was built with Preset K (CMake WRONGGPU_PRESET_K, default ON)."""
    return PRESET_K_MARK.encode("utf-16-le") in read(setup)


def test_without_preset_k(t, payload_dir):
    """An installer built with -DWRONGGPU_PRESET_K=OFF (a release that ships Model M only): nothing of Preset K in it
    or written by it, /PRESET=K is a bad command line, UpscalerIndex is 0, an earlier build's model\\K folder is left
    alone by an install and goes with the rest at the last uninstall."""
    print("\n[no Preset K] installer without Preset K")
    data = read(t.setup)
    for folder, _, names in os.walk(payload_dir):  # the installer's own code and resources: not the payload files in it
        for n in names:
            blob = read(os.path.join(folder, n))
            if len(blob) >= 64:
                data = data.replace(blob, b"")
    # 0.6.0: the only Preset K words are the switch of a game an earlier build set up on Preset K (the question before an
    # update or repair, PresetKText, and its log line): they say Preset K is not in this build and offer Model M
    switch = [" starts on Preset K, which WrongGPU ", " as it is, on Preset K.", "Preset K isn't in this version of WrongGPU: "]
    t.check("the switch to Model M for a game set up on Preset K is there (question and log line)",
            all(s.encode("utf-16-le") in data for s in switch), [s for s in switch if s.encode("utf-16-le") not in data])
    for s in switch:
        data = data.replace(s.encode("utf-16-le"), b"")
    marks = [PRESET_K_MARK, "Preset K", "Model K", "model=K", "/PRESET=K", "preset=k", "Installed startup preset"]
    found = [m for m in marks if m.encode() in data or m.encode("utf-16-le") in data]
    t.check("no other Preset K text in the installer (wizard, command line, report, options)", not found, str(found))
    t.clear_model()
    kfile = os.path.join(t.appdata(), "model", "K", "model.wgcm")
    kdata = b"the Preset K cache of an earlier test build"
    write(kfile, kdata)
    root, b = cyberpunk(t, "no-preset-k", model=t.model.file)
    before = snapshot(root)
    rc = t.run("/DIR=" + root, "/PRESET=K")
    t.check("/PRESET=K: bad command line (exit 10), nothing changed",
            rc == 10 and snapshot(root) == before and "unknown command-line option" in t.logtext() and not os.path.exists(t.model_file()),
            "exit %d; %s" % (rc, t.logtext()[-300:]))
    rc = t.run("/DIR=" + root, "/PRESET=M", *t.route)
    log = t.logtext()
    t.check("/PRESET=M is accepted and installs Model M", rc == 0 and t.model_ok(), "exit %d; %s" % (rc, log[-500:]))
    t.check("no second model cache is prepared or mentioned",
            not [m for m in ("Preset K", "Model K", "other in-game provider", "Only the selected model", "\\K\\") if m in log], log[-900:])
    t.check("the model\\K folder of an earlier build is left alone by the install",
            os.listdir(os.path.dirname(kfile)) == ["model.wgcm"] and read(kfile) == kdata)
    ini_path = os.path.join(b, "wronggpu.ini")
    ini = read(ini_path).decode()
    t.check("wronggpu.ini names Model M", re.search(r"(?m)^runtime_options\s*=\s*model=M\s*$", ini) is not None, ini[-300:])
    rec = read(os.path.join(b, "wronggpu-setup", "install.record")).decode("utf-8", "replace")
    t.check("the install record names Model M's file only", "\\model\\model.wgcm" in rec and "\\K\\" not in rec, rec[:600])
    bundle = os.path.join(t.work, "no-preset-k-support.zip")
    rc = t.run("/COLLECT", "/DIR=" + b, "/OUT=" + bundle)
    info = ""
    if rc == 0 and os.path.isfile(bundle):
        import zipfile
        with zipfile.ZipFile(bundle) as z:
            info = z.read("system-info.txt").decode("utf-8", "replace")
    t.check("support report: Model M's data present and matching, no preset line, nothing of Preset K",
            "present and matching" in info and "startup preset" not in info and "Preset K" not in info and "\\K\\" not in info, "exit %d; %s" % (rc, info[:600]))
    # An install made by a build with Preset K left model=K in the settings: the one model is named again.
    write(ini_path, ini.replace("model=M", "timing_log=1;model=" + "K"))
    earlier = read(ini_path)
    rc = t.run("/DIR=" + root, "/PRESET=M", *t.route)
    ini = read(ini_path).decode()
    t.check("install over settings of a build with Preset K: Model M again, the player's other options kept",
            rc == 0 and re.search(r"(?m)^runtime_options\s*=\s*timing_log=1;model=M\s*$", ini) is not None and "model=" + "K" not in ini, "exit %d; %s" % (rc, ini[-300:]))
    rc = t.uninstall(b)
    # (0.6.1, the ownership rule: that wronggpu.ini was not as WrongGPU last wrote it, so the install backed it up and the
    # uninstall puts it back; never deleted by its name)
    want = dict(before)
    want[key(os.path.relpath(ini_path, root))] = sha(earlier)
    t.check("uninstall restores the game byte for byte (the earlier settings file back as it was)", rc == 0 and snapshot(root) == want, diff(want, snapshot(root)))
    t.check("...and the model folder goes with the earlier build's model\\K in it", not os.path.exists(os.path.join(t.appdata(), "model")))
    # OptiScaler route: UpscalerIndex 0 whatever the package's or an earlier install's file says (index 1 was Model K;
    # without it index 1 is AMD's first version).
    opti = os.path.join(t.work, "optiscaler-no-preset-k")
    write(os.path.join(opti, "OptiScaler.dll"), b"MZ mock OptiScaler")
    write(os.path.join(opti, "OptiScaler.ini"), "[Upscalers]\nDx12Upscaler=auto\n[Libraries]\nFfxDx12SRPath=auto\n[FSR]\nFsr4Update=auto\nUpscalerIndex=1\n")
    root, b = cyberpunk(t, "no-preset-k-opti", model=t.model.file, ffx=None, fsr3=None, fsr3up=None)
    before = snapshot(root)
    hosted = ("/DIR=" + root, "/FORCE", "/NOSCAN", "/EXPERIMENTAL", "/PRESET=M", "/OPTISCALER=" + t.env.win(opti))
    index = lambda: re.findall(r"(?m)^UpscalerIndex\s*=\s*(\S+)\s*$", read(os.path.join(b, "OptiScaler.ini")).decode("utf-8", "replace"))
    rc = t.run(*hosted)
    t.check("OptiScaler route: UpscalerIndex=0 (the package's file said 1)", rc == 0 and index() == ["0"] and "Preset K" not in t.logtext(),
            "exit %d; %s; %s" % (rc, index() if rc == 0 else "", t.logtext()[-500:]))
    if rc == 0:
        oini = os.path.join(b, "OptiScaler.ini")
        write(oini, re.sub(r"(?m)^UpscalerIndex\s*=\s*\S+", "UpscalerIndex=1", read(oini).decode("utf-8", "replace")))
        t.check("control: the installed file now says 1, as after an install with Preset K selected", index() == ["1"])
        rc = t.run(*hosted)
        t.check("...install over it: UpscalerIndex=0 again", rc == 0 and index() == ["0"], "exit %d; %s" % (rc, t.logtext()[-500:]))
        rc = t.uninstall(b)
        after = snapshot(root)
        changed = {k for k in set(before) | set(after) if before.get(k) != after.get(k)}
        # (the file changed after the first install is left to the player, as any file changed since install)
        t.check("...and uninstall restores the game; only the OptiScaler.ini that was changed after the first install is left",
                rc == 0 and changed <= {key(os.path.join("bin", "x64", "OptiScaler.ini"))}, diff(before, after))
    t.clear_model()


def test_game_exe(t):
    """Which executable below a folder is the game (src/setup_game.h, FindGameExe). Found with The Witcher 3 (5.0):
    /DIR=<game>\\bin\\x64_dx12 took D3D12_0\\D3D12StateObjectCompiler.exe, a program of the DirectX Agility SDK the
    game ships, for the game, because of the "d3d12" in its name; the install went into D3D12_0. The fixture has that
    shape. The runs use a mock OptiScaler package (the hosted route is in every build) and stop before anything is
    installed, except the one install."""
    print("\n[15] which exe is the game: the DirectX Agility SDK's programs, DX12 / DX11 names, upscaler files next to the exe")
    GOOD = t.model.file  # noqa: N806
    t.clear_model()
    hosted = ("/FORCE", "/NOSCAN", "/OPTISCALER=" + t.env.win(optiscaler_package(t, "optiscaler-game-exe")))

    def chosen(folder, want):
        """A silent run for /DIR=folder names `want` as the game; it changes nothing. Returns (ok, detail)."""
        before = snapshot(folder)
        rc = t.run("/DIR=" + folder, *hosted)
        log = t.logtext()
        m = re.search(r"(?m)\] game: (.*?) \((?:validated|experimental)\), install folder ", log)
        exe = m.group(1) if m else "(none)"
        return exe.lower() == t.env.win(want).lower() and snapshot(folder) == before, "exit %d, game: %s; %s" % (rc, exe, log[-400:])

    root = t.fresh("witcher-shape")
    dx12 = os.path.join(root, "bin", "x64_dx12")
    sdk = os.path.join(dx12, "D3D12_0")
    game_exe = b"mock game exe " * 300000  # 4.2 MB
    write(os.path.join(dx12, "witcher3.exe"), game_exe)
    write(os.path.join(root, "bin", "x64", "witcher3.exe"), game_exe)  # the DX11 build: same name and size, no upscaler files
    write(os.path.join(dx12, "nvngx_dlss.dll"), GOOD)
    for n in ("libxess.dll", "amd_fidelityfx_upscaler_dx12.dll", "amd_fidelityfx_loader_dx12.dll"):
        write(os.path.join(dx12, n), "mock " + n)
    for n in ("D3D12Core.dll", "d3d12SDKLayers.dll", "D3D12StateObjectCompiler.dll"):
        write(os.path.join(sdk, n), "mock " + n)
    write(os.path.join(sdk, "D3D12StateObjectCompiler.exe"), b"agility sdk program " * 12000)  # 240 KB
    write(os.path.join(sdk, "d3dconfig.exe"), b"agility sdk program " * 4000)
    before = snapshot(root)
    want = os.path.join(dx12, "witcher3.exe")
    ok, detail = chosen(dx12, want)
    t.check("/DIR=<the game's bin\\x64_dx12>: the game exe, not a program of the Agility SDK in D3D12_0", ok, detail)
    ok, detail = chosen(root, want)
    t.check("/DIR=<the game root> (what the library scan hands over): the same exe, next to the DLSS, XeSS and FSR 4 files, not the DX11 build in bin\\x64", ok, detail)
    rc = t.run("/DIR=" + dx12, *hosted, "/EXPERIMENTAL")
    log = t.logtext()
    t.check("...installed next to the game exe, the game's DLSS / XeSS inputs found, D3D12_0 untouched",
            rc == 0 and os.path.isfile(os.path.join(dx12, "wronggpu-setup", "install.record")) and os.path.isfile(os.path.join(dx12, "wronggpu_ffx_dx12.dll")) and
            "hands the game's DLSS / XeSS inputs to WrongGPU" in log and "no DLSS, XeSS or FSR files found" not in log and
            sorted(os.listdir(sdk)) == sorted(["D3D12Core.dll", "d3d12SDKLayers.dll", "D3D12StateObjectCompiler.dll", "D3D12StateObjectCompiler.exe", "d3dconfig.exe"]),
            "exit %d; %s; %s" % (rc, sorted(os.listdir(sdk)), log[-600:]))
    rc = t.uninstall(dx12) if rc == 0 else -1
    t.check("...and uninstall restores the game byte for byte", rc == 0 and snapshot(root) == before, "exit %d; %s" % (rc, diff(before, snapshot(root))))
    if not t.env.native:  # the library scan itself (on Windows the suite never scans the tester's libraries)
        item = os.path.join(t.env.programdata(), "Epic", "EpicGamesLauncher", "Data", "Manifests", "witcher-shape.item")
        write(item, json.dumps({"DisplayName": "Witcher Shape", "InstallLocation": t.env.win(root)}, indent=1))
        rc = t.run("/LIST")
        log = t.logtext()
        t.check("/LIST with the game in an Epic library: the same exe", rc == 0 and ("game Witcher Shape [Epic] experimental -> " + t.env.win(want)).lower() in log.lower(), log[-1500:])
        os.remove(item)

    root = t.fresh("game-exe-control")
    write(os.path.join(root, "Control_DX11.exe"), b"x" * 600000)
    write(os.path.join(root, "Control_DX12.exe"), b"x" * 500000)
    ok, detail = chosen(root, os.path.join(root, "Control_DX12.exe"))
    t.check("Control_DX12.exe next to a larger Control_DX11.exe: the DX12 build", ok, detail)
    root = t.fresh("game-exe-small-tool")
    write(os.path.join(root, "BigGame.exe"), b"x" * 2000000)
    write(os.path.join(root, "tools", "shadercache_dx12.exe"), b"x" * 100000)
    ok, detail = chosen(root, os.path.join(root, "BigGame.exe"))
    t.check("a small program with a DX12 name does not beat the game exe", ok, detail)
    root = t.fresh("game-exe-dx11-only")
    write(os.path.join(root, "OnlyGame_DX11.exe"), b"x" * 500000)
    ok, detail = chosen(root, os.path.join(root, "OnlyGame_DX11.exe"))
    t.check("a game with a DX11 exe only: that exe (the checks decide about the game, not the name)", ok, detail)
    root = t.fresh("game-exe-sdk-folder")
    write(os.path.join(root, "PlainGame.exe"), b"x" * 500000)
    write(os.path.join(root, "D3D12", "D3D12Core.dll"), "mock D3D12Core.dll")
    write(os.path.join(root, "D3D12", "sdktool_d3d12.exe"), b"x" * 2000000)
    ok, detail = chosen(root, os.path.join(root, "PlainGame.exe"))
    t.check("an unknown program in the folder that holds D3D12Core.dll, larger and with a DX12 name: not the game", ok, detail)
    t.clear_model()


def test_anticheat_scope(t):
    """Anti-cheat markers count only in the game's own folders (wg_core/checks.cpp: GameTopFolder, FindAntiCheat).
    User reports (2026-10-20): GTA Online's BattlEye refused The Witcher 3, Cyberpunk 2077, RE Requiem and Star Wars
    Outlaws in the same library. (a) a sibling game's BattlEye / EasyAntiCheat and (b) markers in steamapps\\common,
    in the Steam library root or in the games folder itself do not count: the install goes through (RE Requiem with
    its REFramework stand-in, which it needs); (c) a marker in
    the game's top folder, next to the exe or two levels below the game root refuses it (exit 4) with the marker's
    full path in the check line and in the error; (d) an exe several levels deep (bin\\x64_dx12, Binaries\\Win64):
    the walk reaches the game's top folder and no further. Generated folders only; each install is removed again."""
    print("\n[16] anti-cheat only in the game's own folders: sibling games and library folders do not count")
    GOOD = t.model.file  # noqa: N806
    t.clear_model()
    hosted = ("/FORCE", "/NOSCAN", "/EXPERIMENTAL", "/OPTISCALER=" + t.env.win(optiscaler_package(t, "optiscaler-anticheat")))
    base = os.path.join(t.work, "ac-scope")
    shutil.rmtree(base, ignore_errors=True)

    def game_at(exe_dir, exe):
        write(os.path.join(exe_dir, exe), "mock game exe " * 100 + exe)
        write(os.path.join(exe_dir, "amd_fidelityfx_dx12.dll"), AMD_V1)
        write(os.path.join(exe_dir, "nvngx_dlss.dll"), GOOD)
        write(os.path.join(exe_dir, "steam_api64.dll"), "untouched game file")
        return os.path.join(exe_dir, exe)

    def refw(exe):
        # RE Requiem runs OptiScaler only with the player's REFramework (wg_core/gamerules.cpp): its loader stand-in
        # (synthetic bytes, its own words in the version resource) and folder, which the install leaves where they are.
        d = os.path.dirname(exe)
        write(os.path.join(d, "dinput8.dll"), mockpe.make_dll({"ProductName": "REFramework (test stand-in)"}, (1, 5, 9, 1),
                                                           extra=b"loader stand-in (synthetic, never loaded)", dll_name="dinput8.dll"))
        write(os.path.join(d, "reframework", "config.txt"), "synthetic settings")
        return exe

    def mark(path, folder=True):
        write(os.path.join(path, "marker.dll") if folder else path, "mock anti-cheat")
        return path

    def installs(label, root, exe):
        before = snapshot(root)
        rc = t.run("/EXE=" + exe, *hosted)
        log = t.logtext()
        d = os.path.dirname(exe)
        ok = rc == 0 and os.path.isfile(os.path.join(d, "wronggpu-setup", "install.record")) and "No anti-cheat found" in log
        rc2 = t.uninstall(d) if rc == 0 else -1
        t.check(label, ok and rc2 == 0 and snapshot(root) == before, "exit %d, uninstall %d; %s" % (rc, rc2, log[-700:]))

    def refused(label, root, exe, marker):
        before = snapshot(root)
        rc = t.run("/EXE=" + exe, *hosted)
        log = t.logtext().lower()
        want = ("Anti-cheat found: %s - games with anti-cheat are never supported" % t.env.win(marker)).lower()
        t.check(label, rc == 4 and snapshot(root) == before and ("fail  " + want) in log and ("error: " + want) in log,
                "exit %d; want %r; %s" % (rc, want, log[-900:]))

    # A games folder that is no store library (a launcher's own folder): GTA V with BattlEye, another game with
    # EasyAntiCheat, and a BattlEye folder in the games folder itself.
    lib = os.path.join(base, "OpenNR Games")
    mark(os.path.join(lib, "Grand Theft Auto V", "BattlEye"))
    mark(os.path.join(lib, "Grand Theft Auto V", "BEService_x64.exe"), folder=False)
    write(os.path.join(lib, "Grand Theft Auto V", "GTA5.exe"), "gta")
    mark(os.path.join(lib, "Online Shooter", "EasyAntiCheat"))
    mark(os.path.join(lib, "Online Shooter", "start_protected_game.exe"), folder=False)
    mark(os.path.join(lib, "BattlEye"))
    w3 = game_at(os.path.join(lib, "The Witcher 3", "bin", "x64_dx12"), "witcher3.exe")
    cp = game_at(os.path.join(lib, "Cyberpunk 2077", "bin", "x64"), "Cyberpunk2077.exe")
    re9 = refw(game_at(os.path.join(lib, "RE Requiem"), "re9.exe"))
    swo = game_at(os.path.join(lib, "Star Wars Outlaws"), "Outlaws.exe")
    ue = game_at(os.path.join(lib, "Unreal Game", "Binaries", "Win64"), "UnrealGame-Win64-Shipping.exe")  # no project folder
    installs("(a) sibling GTA V's BattlEye: The Witcher 3 (bin\\x64_dx12) installs", os.path.join(lib, "The Witcher 3"), w3)
    installs("(a) ...Cyberpunk 2077 (bin\\x64) installs", os.path.join(lib, "Cyberpunk 2077"), cp)
    installs("(a) ...RE Requiem (exe in the game folder) installs", os.path.join(lib, "RE Requiem"), re9)
    installs("(a) ...Star Wars Outlaws installs", os.path.join(lib, "Star Wars Outlaws"), swo)
    installs("(a) ...an Unreal game without a project folder (Binaries\\Win64 below the game folder) installs", os.path.join(lib, "Unreal Game"), ue)

    # A Steam library: markers in steamapps\common and in the library root.
    steam = os.path.join(base, "SteamLibrary")
    common = os.path.join(steam, "steamapps", "common")
    write(os.path.join(steam, "libraryfolder.vdf"), '"libraryfolder"\n{\n}\n')
    mark(os.path.join(common, "BattlEye"))
    mark(os.path.join(common, "EasyAntiCheat"))
    mark(os.path.join(steam, "BattlEye"))
    mark(os.path.join(steam, "EasyAntiCheat"))
    mark(os.path.join(common, "GTAV", "BattlEye"))
    scp = game_at(os.path.join(common, "Cyberpunk 2077", "bin", "x64"), "Cyberpunk2077.exe")
    sre = refw(game_at(os.path.join(common, "RE Requiem"), "re9.exe"))
    installs("(b) BattlEye / EasyAntiCheat in steamapps\\common and the library root: Cyberpunk 2077 (bin\\x64) installs", os.path.join(common, "Cyberpunk 2077"), scp)
    installs("(b) ...RE Requiem (exe in the game folder, 3 folders below the library root) installs", os.path.join(common, "RE Requiem"), sre)

    # (c) the game's own markers, in the same games folder (the siblings' markers are still there): refused, full path.
    root = os.path.join(lib, "Own Top")
    exe = game_at(os.path.join(root, "bin", "x64"), "OwnTop.exe")
    m = mark(os.path.join(root, "BattlEye"))
    refused("(c) BattlEye in the game's top folder (exe in bin\\x64): exit 4, the full path in the log", root, exe, m)
    root = os.path.join(lib, "Next To Exe")
    exe = game_at(os.path.join(root, "bin", "x64"), "NextToExe.exe")
    m = mark(os.path.join(root, "bin", "x64", "BEService_x64.exe"), folder=False)
    refused("(c) BEService_x64.exe next to the exe: exit 4, the full path in the log", root, exe, m)
    root = os.path.join(lib, "Two Below")
    exe = game_at(root, "TwoBelow.exe")
    m = mark(os.path.join(root, "Support", "EasyAntiCheat"))
    refused("(c) EasyAntiCheat two levels below the game root: exit 4, the full path in the log", root, exe, m)
    root = os.path.join(common, "Steam Own")
    exe = game_at(os.path.join(root, "bin", "x64"), "SteamOwn.exe")
    m = mark(os.path.join(root, "EasyAntiCheat"))
    refused("(c) EasyAntiCheat in a Steam game's own folder (the library's markers too): exit 4, its own path", root, exe, m)

    # (d) exe several levels deep; a BattlEye folder in the games folder just above the game's top folder.
    deep = os.path.join(base, "Deep Library")
    mark(os.path.join(deep, "BattlEye"))
    root = os.path.join(deep, "Deep Bin")
    exe = game_at(os.path.join(root, "bin", "x64_dx12"), "DeepBin.exe")
    installs("(d) bin\\x64_dx12: the walk stops at the game's top folder (BattlEye one above it ignored)", root, exe)
    m = mark(os.path.join(root, "EasyAntiCheat"))
    refused("(d) ...and reaches it: EasyAntiCheat in the top folder refuses (exit 4)", root, exe, m)
    root = os.path.join(deep, "Deep UE")
    exe = game_at(os.path.join(root, "Proj", "Binaries", "Win64"), "Proj-Win64-Shipping.exe")
    write(os.path.join(root, "Proj.exe"), "unreal bootstrap")
    write(os.path.join(root, "Engine", "Binaries", "Win64", "CrashReportClient.exe"), "crash reporter")
    installs("(d) Unreal <root>\\<Project>\\Binaries\\Win64: the walk stops at <root> (BattlEye one above it ignored)", root, exe)
    m = mark(os.path.join(root, "EasyAntiCheat"))
    refused("(d) ...and reaches it: EasyAntiCheat in <root> refuses (exit 4)", root, exe, m)
    t.clear_model()
    shutil.rmtree(base, ignore_errors=True)


DIRECT_ROUTE_MARK = "Use direct FSR installation instead (advanced)"  # the wizard's advanced choice: only with the direct route
OPTI_REQUIRED = "OptiScaler is required: pass /OPTISCALER=<its release archive, folder or OptiScaler.dll>"


def has_direct_route(setup):
    """Whether the installer was built with the direct FSR route (CMake WRONGGPU_DIRECT_ROUTE, default ON)."""
    return DIRECT_ROUTE_MARK.encode("utf-16-le") in read(setup)


def optiscaler_package(t, name):
    """A mock of the player's OptiScaler download (its unpacked release folder)."""
    opti = os.path.join(t.work, name)
    shutil.rmtree(opti, ignore_errors=True)
    write(os.path.join(opti, "OptiScaler.dll"), b"MZ mock OptiScaler")
    write(os.path.join(opti, "OptiScaler.ini"), "[Upscalers]\nDx12Upscaler=auto\n[Libraries]\nFfxDx12SRPath=auto\n"
                                                "[FSR]\nFsr4Update=auto\nUpscalerIndex=auto\n[Menu]\nShortcutKey=auto\n")
    return opti


def test_amd_provider(t):
    """0.4.5.1: the hosted front end lists AMD's FSR versions after WrongGPU's only when it can load an AMD DLL past
    OptiScaler's redirection (0.4.5 player report: FSR 4 gone from OptiScaler's list and the game's menu). The install
    sets one up under the private name wronggpu-amd-compare.dll (the package's FSR 4 DLL, else the FSR 4 DLL found on
    this PC, else a copy of the game's own amd_fidelityfx_dx12.dll), says which in its log, and never deletes, renames,
    moves aside or overwrites the game's own AMD FidelityFX DLLs. Mock files only; /NOSCAN, so no FSR 4 on this PC."""
    print("\n[AMD provider] AMD's versions for the hosted front end; the game's AMD DLLs left as they are")
    compare = "wronggpu-amd-compare.dll"
    t.clear_model()

    def package(name, fsr4=None, extra=()):
        opti = optiscaler_package(t, name)
        if fsr4 is not None:
            write(os.path.join(opti, "amd_fidelityfx_upscaler_dx12.dll"), fsr4)
        for n, data in extra:
            write(os.path.join(opti, n), data)
        return opti

    def install(root, opti):
        return t.run("/DIR=" + root, "/FORCE", "/NOSCAN", "/EXPERIMENTAL", "/OPTISCALER=" + t.env.win(opti))

    # 1. An OptiScaler package without amd_fidelityfx_upscaler_dx12.dll, no FSR 4 on this PC, a DLSS-only game.
    opti = package("optiscaler-no-fsr4")
    root, b = cyberpunk(t, "amd-none", model=t.model.file, ffx=None, fsr3=None, fsr3up=None)
    before = snapshot(root)
    rc = install(root, opti)
    log = t.logtext()
    t.check("package without FSR 4, no FSR 4 on this PC, no AMD DLL in the game: installed, no AMD copy, the log says FSR 4 / AMD versions are not available",
            rc == 0 and not os.path.exists(os.path.join(b, compare)) and "AMD provider: FSR 4 / AMD versions not available" in log, "exit %d; %s" % (rc, log[-700:]))
    rc = t.uninstall(b)
    t.check("...uninstall restores the game byte for byte", rc == 0 and snapshot(root) == before, diff(before, snapshot(root)))

    # 2. The same package; a game with only amd_fidelityfx_dx12.dll (AMD's FSR 3.1 DLL that the driver upgrades to FSR 4).
    root, b = cyberpunk(t, "amd-game-ffx-only", model=t.model.file, fsr3=None, fsr3up=None)
    before = snapshot(root)
    rc = install(root, opti)
    log = t.logtext()
    t.check("game with only amd_fidelityfx_dx12.dll: installed; its private copy is AMD's provider, byte-exact",
            rc == 0 and os.path.isfile(os.path.join(b, compare)) and read(os.path.join(b, compare)) == AMD_V1, "exit %d; %s" % (rc, log[-700:]))
    t.check("...the game's amd_fidelityfx_dx12.dll stays where it was, byte for byte, nothing renamed",
            read(os.path.join(b, SLOTS[0])) == AMD_V1 and not os.path.exists(os.path.join(b, "amd_fidelityfx_dx12.wronggpu-original.dll")))
    t.check("...the install log names the provider", "AMD provider: the game's own amd_fidelityfx_dx12.dll, version 1.1.4.0" in log and
            "copied as " + compare in log, log[-900:])
    rec = read(os.path.join(b, "wronggpu-setup", "install.record")).decode("utf-8", "replace")
    t.check("...the copy is in the install record (removed with WrongGPU)", ("\t" + compare + "\t") in rec or ("\t" + compare + "\n") in rec or compare in rec, rec[:600])
    rc = install(root, opti)
    t.check("...install again over it: the same provider, the game's DLL untouched",
            rc == 0 and read(os.path.join(b, compare)) == AMD_V1 and read(os.path.join(b, SLOTS[0])) == AMD_V1, "exit %d; %s" % (rc, t.logtext()[-500:]))
    rc = t.uninstall(b)
    t.check("...uninstall removes the copy and restores the game byte for byte", rc == 0 and snapshot(root) == before, diff(before, snapshot(root)))

    # 3. The package's FSR 4 DLL comes first; a package with AMD's other DLLs never replaces the game's own (FSR 4 SDK
    # game: amd_fidelityfx_loader_dx12.dll and amd_fidelityfx_upscaler_dx12.dll next to amd_fidelityfx_dx12.dll).
    pkg_fsr4 = b"the AMD FSR 4 DLL of the player's OptiScaler package"
    opti3 = package("optiscaler-fsr4-and-amd-dlls", fsr4=pkg_fsr4,
                    extra=(("amd_fidelityfx_dx12.dll", b"OptiScaler package's amd_fidelityfx_dx12.dll"),
                           ("amd_fidelityfx_loader_dx12.dll", b"OptiScaler package's loader"),
                           ("amd_fidelityfx_vk.dll", b"OptiScaler package's Vulkan DLL")))
    root, b = cyberpunk(t, "amd-sdk-game", model=t.model.file)
    game_amd = {SLOTS[0]: AMD_V1, "amd_fidelityfx_loader_dx12.dll": b"the game's own FSR 4 SDK loader",
                "amd_fidelityfx_upscaler_dx12.dll": b"the game's own FSR 4 SDK upscaler"}
    for n, data in game_amd.items():
        write(os.path.join(b, n), data)
    before = snapshot(root)
    rc = install(root, opti3)
    log = t.logtext()
    t.check("package with FSR 4: installed; the package's FSR 4 DLL is the provider", rc == 0 and read(os.path.join(b, compare)) == pkg_fsr4 and
            "AMD provider: AMD FSR" in log and "of the OptiScaler package" in log, "exit %d; %s" % (rc, log[-700:]))
    t.check("...the game's amd_fidelityfx_dx12.dll, loader and upscaler DLLs byte for byte, none moved aside or renamed",
            all(os.path.isfile(os.path.join(b, n)) and read(os.path.join(b, n)) == d for n, d in game_amd.items()) and
            not [n for n in os.listdir(b) if "wronggpu-original" in n], str(sorted(os.listdir(b))))
    t.check("...the package's AMD DLLs the game has are not placed; one the game lacks is",
            "OptiScaler's amd_fidelityfx_dx12.dll not placed" in log and "OptiScaler's amd_fidelityfx_loader_dx12.dll not placed" in log and
            read(os.path.join(b, "amd_fidelityfx_vk.dll")) == b"OptiScaler package's Vulkan DLL", log[-900:])
    rc = t.uninstall(b)
    t.check("...uninstall restores the game byte for byte", rc == 0 and snapshot(root) == before, diff(before, snapshot(root)))
    t.clear_model()


def test_asi_plugin(t, payload_dir):
    """0.4.5.1: WrongGPU's OptiScaler plugin (plugins\\WrongGPU.asi, its own payload file) is installed with OptiScaler
    for every game, and OptiScaler.ini loads it ([Plugins] LoadAsiPlugins=true, Path=auto), also when the player's
    OptiScaler.ini has no [Plugins] section; WrongGPU stays OptiScaler's default (Dx12Upscaler=fsr31, its FFX DLL, its
    first version). Update from 0.4.5 (no plugin) adds it; uninstall removes the plugin, its folder and its own files
    (wronggpu-asi.log, wronggpu-asi.ini in the OptiScaler folder) and puts the player's OptiScaler.ini back."""
    print("\n[OptiScaler plugin] WrongGPU.asi installed by default; ini set up; update from 0.4.5; uninstall")
    plugin = os.path.join("plugins", "WrongGPU.asi")
    src = os.path.join(payload_dir, plugin)
    if not os.path.isfile(src):
        t.check("the payload has plugins\\WrongGPU.asi", False, str(sorted(os.listdir(payload_dir))))
        return
    t.clear_model()

    def package(name, ini):
        opti = optiscaler_package(t, name)
        write(os.path.join(opti, "OptiScaler.ini"), ini)
        return opti

    def install(root, opti):
        return t.run("/DIR=" + root, "/FORCE", "/NOSCAN", "/EXPERIMENTAL", "/OPTISCALER=" + t.env.win(opti))

    def ini_of(b):
        p = os.path.join(b, "OptiScaler.ini")
        return read(p).decode("utf-8", "replace") if os.path.isfile(p) else ""

    def section(text, name):
        out, cur = {}, None
        for line in text.replace("\r", "").split("\n"):
            s = line.strip()
            if s.startswith("["):
                cur = s[1:-1].strip().lower()
            elif cur == name.lower() and "=" in s:
                k, v = s.split("=", 1)
                out[k.strip().lower()] = v.strip()
        return out

    def record(b):
        return read(os.path.join(b, "wronggpu-setup", "install.record")).decode("utf-8", "replace")

    # 1. A package whose OptiScaler.ini has no [Plugins] section; a DLSS-only game.
    no_plugins = "[Upscalers]\nDx12Upscaler=auto\n[Libraries]\nFfxDx12SRPath=auto\n[FSR]\nFsr4Update=auto\nUpscalerIndex=auto\n[Menu]\nShortcutKey=auto\n"
    opti = package("optiscaler-no-plugins-section", no_plugins)
    root, b = cyberpunk(t, "asi-default", model=t.model.file, ffx=None, fsr3=None, fsr3up=None)
    before = snapshot(root)
    rc = install(root, opti)
    ini = ini_of(b)
    t.check("install: plugins\\WrongGPU.asi installed next to OptiScaler, byte-exact the payload's",
            rc == 0 and os.path.isfile(os.path.join(b, plugin)) and read(os.path.join(b, plugin)) == read(src), "exit %d; %s" % (rc, t.logtext()[-700:]))
    t.check("...OptiScaler.ini without [Plugins]: the section is added with LoadAsiPlugins=true and Path=auto",
            section(ini, "Plugins") == {"loadasiplugins": "true", "path": "auto"}, ini[-400:])
    t.check("...WrongGPU stays OptiScaler's default (Dx12Upscaler=fsr31, FfxDx12SRPath = WrongGPU's DLL, UpscalerIndex=0)",
            section(ini, "Upscalers").get("dx12upscaler") == "fsr31" and section(ini, "FSR").get("upscalerindex") == "0" and
            section(ini, "Libraries").get("ffxdx12srpath", "").lower().endswith("wronggpu_ffx_dx12.dll"), ini[:500])
    rec = record(b)
    t.check("...the install record has the plugin (copy) and its new folder (dir)",
            "\tplugins\\WrongGPU.asi\r\n" in rec and "\r\ndir\t0\t-\t-\tplugins\r\n" in rec, rec[-900:])
    # What the plugin writes while the game runs (its log and the entry last selected) into the OptiScaler folder.
    write(os.path.join(b, "wronggpu-asi.log"), "[12:00:00.000 1] [WrongGPU.asi] hooked\r\n")
    write(os.path.join(b, "wronggpu-asi.ini"), "selection=amd\r\n")
    rc = install(root, opti)
    t.check("...install again over it: the plugin and the ini are the same", rc == 0 and read(os.path.join(b, plugin)) == read(src) and
            section(ini_of(b), "Plugins").get("loadasiplugins") == "true", "exit %d" % rc)
    write(os.path.join(b, "wronggpu-asi.log"), "[12:00:01.000 1] [WrongGPU.asi] hooked\r\n")
    write(os.path.join(b, "wronggpu-asi.ini"), "selection=amd\r\n")
    rc = t.uninstall(b)
    t.check("...uninstall removes the plugin, its folder and its files, and restores the game byte for byte",
            rc == 0 and snapshot(root) == before and not os.path.exists(os.path.join(b, "plugins")), diff(before, snapshot(root)))

    # 2. The player's OptiScaler.ini has [Plugins] with plugins off and its own path: the install loads WrongGPU's
    # plugin from OptiScaler's own folder; the player's files come back on uninstall.
    custom = no_plugins + "[Plugins]\nPath=G:\\elsewhere\nLoadAsiPlugins=false\nLoadReShade=auto\n"
    opti2 = package("optiscaler-plugins-off", custom)
    root, b = cyberpunk(t, "asi-plugins-off", model=t.model.file, ffx=None, fsr3=None, fsr3up=None)
    write(os.path.join(b, "plugins", "SomeoneElse.asi"), b"another plugin the player had")
    before = snapshot(root)
    rc = install(root, opti2)
    ini = ini_of(b)
    t.check("[Plugins] with LoadAsiPlugins=false and another path: set to true and auto, the other keys kept",
            rc == 0 and section(ini, "Plugins") == {"path": "auto", "loadasiplugins": "true", "loadreshade": "auto"}, "exit %d; %s" % (rc, ini[-300:]))
    t.check("...an existing plugins folder: WrongGPU.asi added, the player's plugin left as it is, the folder not recorded as new",
            rc == 0 and read(os.path.join(b, plugin)) == read(src) and read(os.path.join(b, "plugins", "SomeoneElse.asi")) == b"another plugin the player had" and
            "\r\ndir\t0\t-\t-\tplugins\r\n" not in record(b))
    rc = t.uninstall(b)
    t.check("...uninstall: the player's plugins folder back byte for byte, WrongGPU.asi gone",
            rc == 0 and snapshot(root) == before, diff(before, snapshot(root)))

    # 3. Update from 0.4.5: an install of that version (no plugin, no [Plugins] keys), made here by turning this
    # version's install into one: plugin and folder gone from disk and record, OptiScaler.ini as 0.4.5 wrote it.
    root, b = cyberpunk(t, "asi-update-045", model=t.model.file, ffx=None, fsr3=None, fsr3up=None)
    before = snapshot(root)
    rc = install(root, opti)
    t.check("update from 0.4.5: first install", rc == 0, "exit %d; %s" % (rc, t.logtext()[-500:]))
    shutil.rmtree(os.path.join(b, "plugins"))
    old_ini = "".join(l for l in ini_of(b).splitlines(True) if not l.strip().lower().startswith(("[plugins]", "loadasiplugins=", "path=")))
    write(os.path.join(b, "OptiScaler.ini"), old_ini.encode("utf-8"))
    lines = []
    for l in record(b).split("\r\n"):
        f = l.split("\t")
        if l.startswith("version\t"):
            l = "version\t0.4.5"
        elif len(f) == 5 and f[4].lower().startswith("plugins"):
            continue
        elif len(f) == 5 and f[4] == "OptiScaler.ini":
            l = "\t".join(f[:3] + [sha(old_ini.encode("utf-8")).lower()] + f[4:])
        lines.append(l)
    write(os.path.join(b, "wronggpu-setup", "install.record"), "\r\n".join(lines).encode("utf-8"))
    t.check("...it looks like 0.4.5's (no plugin, no [Plugins] keys)", not os.path.exists(os.path.join(b, "plugins")) and "[Plugins]" not in ini_of(b))
    rc = install(root, opti)
    ini = ini_of(b)
    t.check("...update: the plugin is installed and OptiScaler.ini loads it",
            rc == 0 and read(os.path.join(b, plugin)) == read(src) and section(ini, "Plugins").get("loadasiplugins") == "true" and
            "\tplugins\\WrongGPU.asi\r\n" in record(b) and "earlier installation" in t.logtext(), "exit %d; %s" % (rc, t.logtext()[-900:]))
    rc = t.uninstall(b)
    t.check("...uninstall after the update restores the game byte for byte", rc == 0 and snapshot(root) == before, diff(before, snapshot(root)))
    t.clear_model()


def test_without_direct_route(t, payload_dir, direct_setup):
    """An installer built with -DWRONGGPU_DIRECT_ROUTE=OFF (a release that requires OptiScaler): nothing of the direct
    route in it, /DIRECTFSR is a bad command line, a silent install without /OPTISCALER= is refused before anything is
    written, a game needs none of the three FidelityFX DLLs, the FSR 3.0 component DLLs are not built in, and an
    earlier direct install is removed (the game's FSR DLLs back byte for byte) before the hosted one is made.
    direct_setup: an installer WITH the direct route, built from the same sources and test map; it makes the earlier
    direct installs."""
    print("\n[no direct route] installer that requires OptiScaler")
    GOOD = t.model.file  # noqa: N806
    game = {SLOTS[0]: AMD_V1, SLOTS[1]: FSR3, SLOTS[2]: FSR3UP}
    t.clear_model()
    opti = optiscaler_package(t, "optiscaler-required")

    def hosted(root, *more):
        return ("/DIR=" + root, "/FORCE", "/NOSCAN", "/OPTISCALER=" + t.env.win(opti)) + more

    def same(path, data):
        return os.path.isfile(path) and read(path) == data

    def record(b):
        p = os.path.join(b, "wronggpu-setup", "install.record")
        return read(p).decode("utf-8", "replace") if os.path.isfile(p) else ""

    def renamed(b):
        return sorted(n for n in os.listdir(b) if n.lower().endswith(".wronggpu-original.dll"))

    def hosted_ok(b):
        """A normal hosted install in the exe folder: OptiScaler as the proxy, WrongGPU's FFX DLL under its own name,
        the runtime; the record has OptiScaler's files and nothing of the direct route."""
        rec = record(b)
        return (same(os.path.join(b, "dxgi.dll"), b"MZ mock OptiScaler") and os.path.isfile(os.path.join(b, "OptiScaler.ini")) and
                same(os.path.join(b, "wronggpu_ffx_dx12.dll"), read(os.path.join(payload_dir, SLOTS[0]))) and
                same(os.path.join(b, "wronggpu_runtime.dll"), read(os.path.join(payload_dir, "wronggpu_runtime.dll"))) and
                "\tOptiScaler.ini" in rec and "\twronggpu_ffx_dx12.dll" in rec and "\nrename\t" not in rec and
                not any("\t" + n in rec for n in SLOTS) and not renamed(b) and len(t.keys_for(b)) == 1)

    def tail(rc, log, root=None, want=None):
        return "exit %d; %s%s" % (rc, diff(want, snapshot(root)) + "; " if root else "", log[-700:])

    # Point 1 (wizard) and the command line: the texts of the direct route are not in the program.
    data = read(t.setup)
    for folder, _, names in os.walk(payload_dir):  # the installer's own code and resources: not the payload files in it
        for n in names:
            blob = read(os.path.join(folder, n))
            if len(blob) >= 64:
                data = data.replace(blob, b"")
    marks = [DIRECT_ROUTE_MARK, "Direct FSR", "direct FSR", "DIRECTFSR", "/directfsr", "Advanced direct installation", "advanced option", "(advanced)",
             "without OptiScaler", "Replaces supported FSR 3 DLLs", "(recommended)", "RECOMMENDED", "Install method",
             "AMD FSR found", "No AMD FSR 3 DLL found", "Renamed:  ", "Choose AMD FSR 3"]
    found = [m for m in marks if m.encode() in data or m.encode("utf-16-le") in data]
    t.check("no text of the direct FSR route or of a method choice in the installer (wizard pages, review, finish page, command line)", not found, str(found))
    t.check("the requirement message is in it", OPTI_REQUIRED.encode("utf-16-le") in data)

    # Point 5: the FSR 3.0 component DLLs are not shipped; a payload folder that has them all the same.
    t.check("no FSR 3.0 component DLL of the payload folder is built into the installer",
            all(not os.path.isfile(os.path.join(payload_dir, n)) or read(os.path.join(payload_dir, n)) not in read(t.setup) for n in COMPONENTS))
    full = os.path.join(t.work, "payload-with-fsr30")
    shutil.rmtree(full, ignore_errors=True)
    for n in SLOTS + ["wronggpu_runtime.dll"]:
        write(os.path.join(full, n), mockpe.make_dll(mockpe.FRONT_END, (0, 4, 5, 0), extra=n.encode()))
    tables = {}
    for flag in ("", "--hosted-only"):
        out = os.path.join(t.work, "payload-table" + flag)
        shutil.rmtree(out, ignore_errors=True)
        p = subprocess.run([sys.executable, os.path.join(INSTALLER, "tools", "make_payload.py"), "--payload", full, "--map", os.path.join(t.work, "prepared", "test-map.json"),
                            "--version", "0.0.0", "--out", out] + ([flag] if flag else []), stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        tables[flag] = (read(os.path.join(out, "payload_table.h")) + read(os.path.join(out, "resources.rc"))).decode() if p.returncode == 0 else ""
    t.check("make_payload.py --hosted-only (what this build passes): the FSR 3.0 DLLs of a payload folder are left out, the rest is built in",
            all(n in tables[""] for n in SLOTS) and SLOTS[0] in tables["--hosted-only"] and "wronggpu_runtime.dll" in tables["--hosted-only"] and
            not any(n in tables["--hosted-only"] for n in COMPONENTS), tables["--hosted-only"][:600])

    # Point 2: the command line.
    root, b = cyberpunk(t, "no-direct", model=GOOD)
    before = snapshot(root)
    rc = t.run("/DIR=" + root, "/FORCE", "/NOSCAN", "/DIRECTFSR")
    log = t.logtext()
    rc2 = t.run(*hosted(root, "/DIRECTFSR"))
    t.check("/DIRECTFSR: bad command line (exit 10), with a package too; nothing changed",
            rc == 10 and rc2 == 10 and snapshot(root) == before and "unknown command-line option" in log and not os.path.exists(t.model_file()), "exit %d, %d; %s" % (rc, rc2, log[-300:]))
    for label, more in (("", ()), (", also with every confirmation", ("/EXPERIMENTAL", "/MOVEMODS", "/REMOVEOLD"))):
        rc = t.run("/DIR=" + root, "/FORCE", "/NOSCAN", *more)
        log = t.logtext()
        t.check("a silent install without /OPTISCALER=: refused (exit 2) with the requirement message" + label + "; nothing written",
                rc == 2 and OPTI_REQUIRED in log and snapshot(root) == before and not os.path.exists(os.path.join(t.appdata(), "model")) and not t.keys_for(b), tail(rc, log, root, before))
    not_opti = os.path.join(t.work, "not-optiscaler")
    write(os.path.join(not_opti, "readme.txt"), "not an OptiScaler release")
    rc = t.run("/DIR=" + root, "/FORCE", "/NOSCAN", "/OPTISCALER=" + t.env.win(not_opti))
    log = t.logtext()
    t.check("a package that is not OptiScaler's release: refused (exit 2), nothing written",
            rc == 2 and "OptiScaler: No OptiScaler.dll and OptiScaler.ini" in log and snapshot(root) == before and not os.path.exists(os.path.join(t.appdata(), "model")), tail(rc, log, root, before))

    # Point 3: what qualifies. A game with all three FidelityFX DLLs (they stay the game's), one with none of them, one
    # with no upscaler file at all; validated and experimental as before.
    rc = t.run(*hosted(root))
    log = t.logtext()
    t.check("hosted install into a game with the three FidelityFX DLLs: installed, the game's DLLs untouched",
            rc == 0 and hosted_ok(b) and all(same(os.path.join(b, n), d) for n, d in game.items()) and t.model_ok() and "status\tvalidated" in record(b), tail(rc, log))
    t.check("...none of WrongGPU's FSR 3.0 DLLs is in the game folder or the record", not any(n in record(b) for n in COMPONENTS) and
            sorted(n for n in os.listdir(b) if n.lower().startswith("ffx_fsr3")) == sorted(COMPONENTS))
    zpath = os.path.join(t.work, "no-direct-logs.zip")
    rc = t.run("/COLLECT", "/DIR=" + b, "/OUT=" + zpath)
    t.check("collect logs needs no package", rc == 0 and os.path.isfile(zpath) and "install/install.record" in zipfile.ZipFile(zpath).namelist(), "exit %d" % rc)
    rc = t.run(*hosted(root))
    t.check("setup again over the hosted install", rc == 0 and hosted_ok(b) and all(same(os.path.join(b, n), d) for n, d in game.items()), tail(rc, t.logtext()))
    rc = t.run("/UNINSTALL", "/DIR=" + root)
    t.check("/UNINSTALL needs no package: the game is restored byte for byte", rc == 0 and snapshot(root) == before and not t.keys_for(b), tail(rc, t.logtext(), root, before))
    root, b = cyberpunk(t, "no-direct-no-ffx", model=GOOD, ffx=None, fsr3=None, fsr3up=None)
    before = snapshot(root)
    rc = t.run(*hosted(root))
    t.check("hosted install into a game folder with none of the three FidelityFX DLLs (validated game): installed",
            rc == 0 and hosted_ok(b) and not any(os.path.exists(os.path.join(b, n)) for n in SLOTS) and "status\tvalidated" in record(b), tail(rc, t.logtext()))
    rc = t.uninstall(b)
    t.check("...and uninstall restores it byte for byte", rc == 0 and snapshot(root) == before, tail(rc, t.logtext(), root, before))
    root = t.fresh("no-direct-plain")
    write(os.path.join(root, "PlainGame.exe"), "plain game exe" * 100)
    picked = os.path.join(t.work, "elsewhere", "no-direct", "nvngx_dlss.dll")
    write(picked, GOOD)
    before = snapshot(root)
    rc = t.run(*hosted(root, "/MODEL=" + picked))
    t.check("a game that is not validated: refused without /EXPERIMENTAL (exit 3) as before, nothing changed", rc == 3 and snapshot(root) == before, tail(rc, t.logtext(), root, before))
    rc = t.run(*hosted(root, "/MODEL=" + picked, "/EXPERIMENTAL"))
    t.check("...with /EXPERIMENTAL: installed (no FSR, DLSS or XeSS file in the game), the record says experimental",
            rc == 0 and hosted_ok(root) and "status\texperimental" in record(root), tail(rc, t.logtext()))
    rc = t.uninstall(root)
    t.check("...and uninstall restores it byte for byte", rc == 0 and snapshot(root) == before, tail(rc, t.logtext(), root, before))
    root = t.fresh("no-direct-007")
    write(os.path.join(root, "007FirstLight.exe"), "mock game exe" * 100)
    write(os.path.join(root, "nvngx_dlss.dll"), GOOD)
    before = snapshot(root)
    rc = t.run(*hosted(root, "/EXPERIMENTAL"))
    oini = read(os.path.join(root, "OptiScaler.ini")).decode("utf-8", "replace") if rc == 0 else ""
    t.check("007 First Light: OptiScaler's DxgiFactoryWrapping=true as before", rc == 0 and re.search(r"(?m)^DxgiFactoryWrapping\s*=\s*true\s*$", oini) is not None, tail(rc, t.logtext()))
    t.check("...and uninstalled", t.uninstall(root) == 0 and snapshot(root) == before, diff(before, snapshot(root)))

    # Point 4: setup over an earlier direct install.
    if not direct_setup or not os.path.isfile(direct_setup) or not has_direct_route(direct_setup):
        t.check("--direct-setup names an installer with the direct route (it makes the earlier direct installs)", False, str(direct_setup))
        t.clear_model()
        return
    # A newer WrongGPU build placed by hand over the direct install (the states of test_upgrade_paths).
    hot = {SLOTS[0]: mockpe.make_dll(mockpe.FRONT_END, (0, 5, 0, 0), extra=b"hand-placed front end", exports=mockpe.FFX_API_EXPORTS, dll_name=SLOTS[0]),
           SLOTS[2]: mockpe.make_dll(mockpe.FRONT_END, (0, 5, 0, 0), extra=b"hand-placed forwarder", exports=mockpe.FSR3UPSCALER_EXPORTS, dll_name=SLOTS[2])}  # (0.6.1: no hand-placed runtime, see hand-placed-runtime)

    def direct_install(name, version=None, hand_placed=False):
        root, b = cyberpunk(t, name, model=GOOD)
        before = snapshot(root)
        rc = t.run("/DIR=" + root, "/FORCE", "/NOSCAN", exe=direct_setup)
        rec = record(b)
        t.check(name + ": an earlier direct install (the game's three FSR DLLs renamed, WrongGPU's under their names)",
                rc == 0 and rec.count("\nrename\t") == 3 and "\tOptiScaler.ini" not in rec and renamed(b) == sorted(n[:-4] + ".wronggpu-original.dll" for n in SLOTS) and
                all(same(os.path.join(b, n[:-4] + ".wronggpu-original.dll"), d) for n, d in game.items()) and len(t.keys_for(b)) == 1, tail(rc, t.logtext()))
        if version:
            write(os.path.join(b, "wronggpu-setup", "install.record"), "\n".join(("version\t" + version + ("\r" if l.endswith("\r") else "")) if l.startswith("version\t") else l
                                                                                 for l in rec.split("\n")).encode("utf-8"))
        for n, d in hot.items() if hand_placed else ():
            write(os.path.join(b, n), d)
        return root, b, before

    def originals_back(b, expected=None):
        return all(same(os.path.join(b, n), d) for n, d in (expected or game).items()) and not renamed(b)

    root, b, before = direct_install("direct-to-hosted", version="0.4.1")
    ini_path = os.path.join(b, "wronggpu.ini")
    # The player's own settings; the last two only make sense on the direct route (WrongGPU's own panel and compare key).
    write(ini_path, read(ini_path).decode("utf-8").rstrip("\r\n") + "\r\nview = depth\r\noverlay = 1\r\ncompare_key = home\r\n")
    old_ini = read(ini_path).decode("utf-8")
    state = snapshot(root)
    rc = t.run("/DIR=" + root, "/FORCE", "/NOSCAN")
    log = t.logtext()
    t.check("setup over a direct install without /OPTISCALER=: refused (exit 2), the direct install is left as it is",
            rc == 2 and OPTI_REQUIRED in log and snapshot(root) == state and len(t.keys_for(b)) == 1, tail(rc, log, root, state))
    rc = t.run(*hosted(root))
    log = t.logtext()
    t.check("setup over a direct install (of 0.4.1) with /OPTISCALER=: installed", rc == 0, tail(rc, log))
    t.check("...the direct install was removed first: the game's three FSR DLLs are back under their names byte for byte, no renamed copies", originals_back(b), str(sorted(os.listdir(b))))
    t.check("...and the install is a normal hosted one (OptiScaler as the proxy, WrongGPU's DLL under its own name, the record, one Apps & features entry)",
            hosted_ok(b) and "version\t0.4.1" not in record(b), record(b)[:900])
    new_ini = read(ini_path).decode("utf-8") if os.path.isfile(ini_path) else ""
    mine = lambda text: [l for l in text.splitlines() if not l.replace(" ", "").startswith(("upscaler=", "runtime_dll=", "runtime_options="))]  # noqa: E731
    t.check("...the player's wronggpu.ini is kept, the settings of the direct route as they are (only what a hosted install sets is written)",
            mine(new_ini) == mine(old_ini) and "overlay = 1" in new_ini and "compare_key = home" in new_ini and "view = depth" in new_ini, new_ini)
    rc = t.uninstall(b)
    # (0.6.1, the ownership rule: the player's edited wronggpu.ini was backed up by the hosted install and comes back)
    want = dict(before)
    want[key(os.path.relpath(ini_path, root))] = sha(old_ini.encode("utf-8"))
    t.check("...uninstall then removes everything of WrongGPU: the game as it was before WrongGPU, the player's edited wronggpu.ini back as it was",
            rc == 0 and snapshot(root) == want and not t.keys_for(b), tail(rc, t.logtext(), root, want))

    root, b, before = direct_install("direct-hand-placed-to-hosted", hand_placed=True)
    rc = t.run(*hosted(root))
    log = t.logtext()
    t.check("setup over a direct install with hand-placed WrongGPU DLLs (front end, a forwarder): the originals are back byte for byte, hosted install",
            rc == 0 and originals_back(b) and hosted_ok(b), tail(rc, log))
    rc = t.uninstall(b)
    t.check("...and uninstall leaves the game as it was before WrongGPU", rc == 0 and snapshot(root) == before, tail(rc, t.logtext(), root, before))
    root, b, before = direct_install("direct-no-backup-to-hosted", hand_placed=True)
    shutil.rmtree(os.path.join(b, "wronggpu-setup", "backup"))
    rc = t.run(*hosted(root))
    log = t.logtext()
    t.check("...the same without the backup folder: the originals come back from their renamed copies, hosted install", rc == 0 and originals_back(b) and hosted_ok(b), tail(rc, log))
    rc = t.uninstall(b)
    t.check("...and uninstall leaves the game as it was before WrongGPU", rc == 0 and snapshot(root) == before, tail(rc, t.logtext(), root, before))
    root, b, before = direct_install("direct-game-update-to-hosted")
    write(os.path.join(b, SLOTS[0]), AMD_V2)  # a game update replaced WrongGPU's DLL after the direct install
    expect = dict(before)
    expect[key("bin/x64/" + SLOTS[0])] = sha(AMD_V2)
    rc = t.run(*hosted(root))
    log = t.logtext()
    t.check("setup over a direct install after a game update of amd_fidelityfx_dx12.dll: the game's new file stays, the other two are back, hosted install",
            rc == 0 and originals_back(b, {SLOTS[0]: AMD_V2, SLOTS[1]: FSR3, SLOTS[2]: FSR3UP}) and hosted_ok(b), tail(rc, log))
    rc = t.uninstall(b)
    # (0.6.1, the ownership rule: the game's FSR DLL from before WrongGPU, which the game's update replaced, is kept in
    # WrongGPU-originals, never deleted)
    kept_old = [v for k, v in snapshot(root).items() if "/wronggpu-originals/" in k and not k.endswith("/")]
    outside = {k: v for k, v in snapshot(root).items() if "wronggpu-originals" not in k}
    t.check("...and uninstall leaves the updated game; the game's FSR DLL from before WrongGPU is kept in WrongGPU-originals",
            rc == 0 and outside == expect and kept_old == [sha(AMD_V1)], tail(rc, t.logtext(), root, expect))

    # The game's FSR DLL in a plugin folder: the direct install is there, the hosted one goes next to the exe.
    root, win64, ffx_dir = unreal(t, "direct-plugin-to-hosted", model=GOOD, ffx_in_plugin=True)
    before = snapshot(root)
    rc = t.run("/DIR=" + root, "/FORCE", "/NOSCAN", "/EXPERIMENTAL", exe=direct_setup)
    t.check("an earlier direct install in a plugin folder", rc == 0 and "\nrename\t" in record(ffx_dir) and len(t.keys_for(ffx_dir)) == 1 and not record(win64), tail(rc, t.logtext()))
    plugin_ini = os.path.join(ffx_dir, "wronggpu.ini")
    write(plugin_ini, read(plugin_ini).decode("utf-8").rstrip("\r\n") + "\r\nview = depth\r\n")
    edited_plugin_ini = read(plugin_ini)
    state = snapshot(root)
    rc = t.run("/DIR=" + root, "/FORCE", "/NOSCAN", "/EXPERIMENTAL")
    t.check("...without /OPTISCALER=: refused (exit 2), left as it is", rc == 2 and snapshot(root) == state, tail(rc, t.logtext(), root, state))
    rc = t.run(*hosted(root, "/EXPERIMENTAL"))
    log = t.logtext()
    t.check("...with /OPTISCALER=: the plugin folder is back to the game's own (AMD's DLL byte for byte, nothing of WrongGPU, no state folder, no entry)",
            rc == 0 and os.listdir(ffx_dir) == [SLOTS[0]] and same(os.path.join(ffx_dir, SLOTS[0]), AMD_V1) and not t.keys_for(ffx_dir), tail(rc, log) + str(os.listdir(ffx_dir)))
    win_ini = os.path.join(win64, "wronggpu.ini")
    t.check("...the hosted install is next to the exe, with the player's settings", rc == 0 and hosted_ok(win64) and os.path.isfile(win_ini) and "view = depth" in read(win_ini).decode("utf-8"), record(win64)[:900])
    rc = t.uninstall(win64)
    # (0.6.1, the ownership rule: the player's edited settings, moved next to the exe, were backed up by the hosted install
    # and come back as they were; never deleted by their name)
    want = dict(before)
    want[key(os.path.relpath(win_ini, root))] = sha(edited_plugin_ini)
    t.check("...and uninstall leaves the game as it was before WrongGPU, the player's edited settings back as they were",
            rc == 0 and snapshot(root) == want, tail(rc, t.logtext(), root, want))

    # A failure while the hosted install is written, and the old install's removal by this installer.
    root, b, before = direct_install("direct-to-hosted-fails")
    write(os.path.join(b, "wronggpu_runtime.dll.wronggpu-tmp", "keep.txt"), "in the way")  # the new runtime cannot be written
    expect = dict(before)
    expect[key("bin/x64/wronggpu_runtime.dll.wronggpu-tmp") + "/"] = "dir"
    expect[key("bin/x64/wronggpu_runtime.dll.wronggpu-tmp/keep.txt")] = sha(b"in the way")
    rc = t.run(*hosted(root))
    log = t.logtext()
    t.check("setup over a direct install that fails while the hosted files are written (exit 1): the game's original files, nothing of WrongGPU or OptiScaler, no entry",
            rc == 1 and snapshot(root) == expect and not t.keys_for(b), tail(rc, log, root, expect))
    root, b, before = direct_install("direct-uninstall")
    rc = t.run("/UNINSTALL", "/DIR=" + root)
    t.check("/UNINSTALL of a direct install with this installer (no package): the game is restored byte for byte", rc == 0 and snapshot(root) == before and not t.keys_for(b),
            tail(rc, t.logtext(), root, before))
    t.clear_model()


def test_preset_k(t):
    """Preset K (an installer built with it): matching caches, update M <-> K, support report, dual source, K first.
    These 22 checks were the release drivers' (build/release/*/test_candidate_payload.py), which ran them only on the
    direct route; here they run on the route under test (t.route: the mock OptiScaler package for an installer without
    the direct route). On the OptiScaler route also: the plugin starts on the preset's entry (wronggpu-asi.ini)."""
    print("\n[Preset K] matching-cache, update and restore checks")
    opti = bool(t.route)
    _, game = cyberpunk(t, "preset-switch", t.model.file)
    ini = os.path.join(game, "wronggpu.ini")
    asi = os.path.join(game, "wronggpu-asi.ini")

    def selection():
        return read(asi).decode("utf-8", "replace").replace("\r", "").split("selection=")[-1].strip() if os.path.isfile(asi) else None

    # Preset K must not treat an existing M cache as a matching model.
    rc = t.run("/DIR=" + game, "/PRESET=K", *t.route)
    t.check("K rejects M-only source/cache before installing", rc == 5, t.logtext()[-900:])
    rc = t.run("/DIR=" + game, "/PRESET=M", *t.route)
    t.check("M can still install after rejected K", rc == 0, t.logtext()[-900:])
    if opti:
        t.check("OptiScaler route, Preset M: the plugin starts on WrongGPU (Model M) (wronggpu-asi.ini selection=wronggpu)",
                selection() == "wronggpu", str(selection()))
    mpath = t.model_file()
    mbytes = read(mpath)
    k = cases.k_mini_case(file_version=(310, 98, 2, 0))
    kdll = os.path.join(t.work, "k-synthetic-source.dll")
    write(kdll, k.file)
    rc = t.run("/DIR=" + game, "/PRESET=K", "/MODEL=" + kdll, *t.route)
    t.check("Update M to K converts a matching original synthetic K fixture", rc == 0, t.logtext()[-900:])
    kp = os.path.join(os.path.dirname(mpath), "K", "model.wgcm")
    t.check("M cache remains byte-identical after K preparation", os.path.isfile(mpath) and read(mpath) == mbytes)
    t.check("K cache includes its independently verified int32 net group",
            os.path.isfile(kp) and any(s[0] == "net" for s in read_container(kp)[2]))
    t.check("K is actually selected in installed options", "model=K" in read(ini).decode("utf-8", "replace"))
    if opti:
        t.check("OptiScaler route, Preset K: the plugin starts on WrongGPU (Model K) (wronggpu-asi.ini selection=wronggpu-k)",
                selection() == "wronggpu-k", str(selection()))
    rec = read(os.path.join(game, "wronggpu-setup", "install.record")).decode("utf-8", "replace")
    t.check("Install record identifies K model", kp.replace("/", "\\").lower() in rec.replace("/", "\\").lower())
    rc = t.run("/DIR=" + game, "/PRESET=K", *t.route)
    t.check("K cached reuse verifies integer types and all stage digests", rc == 0 and "model: reusing" in t.logtext(), t.logtext()[-900:])
    bundle = os.path.join(t.work, "k-support.zip")
    rc = t.run("/COLLECT", "/DIR=" + game, "/OUT=" + bundle)
    with zipfile.ZipFile(bundle) as z:
        info = z.read("system-info.txt").decode("utf-8")
        t.check("K support collection reports installed K despite default M wizard selection",
                rc == 0 and "Installed startup preset: K" in info and "present and matching" in info and "\\K\\model.wgcm" in info, info[:900])
        t.check("Support ZIP excludes both converted model caches", not any(n.endswith(".wgcm") for n in z.namelist()))
    kbytes = read(kp)
    rc = t.run("/DIR=" + game, "/PRESET=M", *t.route)
    t.check("Update K back to M succeeds", rc == 0, t.logtext()[-900:])
    t.check("Switch back preserves K cache", os.path.isfile(kp) and read(kp) == kbytes)
    t.check("M wins after previous K options", read(ini).decode("utf-8", "replace").split("model=")[-1].startswith("M"))
    if opti:
        t.check("OptiScaler route, back to M: the plugin starts on WrongGPU (Model M) again", selection() == "wronggpu", str(selection()))
    rc = t.uninstall(game)
    t.check("Preset-switched full uninstall succeeds", rc == 0, t.logtext()[-900:])
    t.check("Game model source remains untouched", read(os.path.join(game, "nvngx_dlss.dll")) == t.model.file)
    if opti:
        t.check("OptiScaler route: the uninstall removes the plugin's wronggpu-asi.ini", not os.path.exists(asi))
    t.clear_model()
    _, dualgame = cyberpunk(t, "dual-model-source", t.model.file + k.file)
    rc = t.run("/DIR=" + dualgame, "/PRESET=M", *t.route)
    t.check("A matching combined synthetic source prepares both models in one install",
            rc == 0 and "Both Model M and Preset K model caches are prepared" in t.logtext(), t.logtext()[-1500:])
    t.check("Dual install stores separate M and K containers",
            os.path.isfile(t.model_file()) and os.path.isfile(os.path.join(os.path.dirname(t.model_file()), "K", "model.wgcm")))
    t.check("Dual install preserves the exact local source file", read(os.path.join(dualgame, "nvngx_dlss.dll")) == t.model.file + k.file)
    rc = t.uninstall(dualgame)
    t.check("Dual-model uninstall restores the game", rc == 0, t.logtext()[-900:])
    # Preset K starts first: model M is then the other model, and an earlier M file (no output-filter group) is
    # converted again all the same; K's own file is reused as before.
    earlier = build_container(cases.mini_case(), hashlib.sha256(b"an earlier map").digest())
    t.clear_model()
    _, kgame = cyberpunk(t, "k-first-earlier-m", t.model.file + k.file)
    write(t.model_file(), earlier)
    rc = t.run("/DIR=" + kgame, "/PRESET=K", *t.route)
    rec = read(os.path.join(kgame, "wronggpu-setup", "install.record")).decode("utf-8", "replace")
    t.check("K first: an earlier M file is converted again as the other model and gains the filter group",
            rc == 0 and t.model_ok() and "\nmodel-m\treplaced: the file has no output-filter weights; converted again from " in rec,
            rec[:900] + t.logtext()[-1500:])
    kbytes, mbytes = read(kp), read(t.model_file())
    rc = t.run("/DIR=" + kgame, "/PRESET=K", *t.route)
    rec = read(os.path.join(kgame, "wronggpu-setup", "install.record")).decode("utf-8", "replace")
    t.check("K first, again: both files reused, the record says so for M",
            rc == 0 and read(kp) == kbytes and read(t.model_file()) == mbytes and "\nmodel-m\treused; output filter: from the DLSS file" in rec and
            t.logtext().count("model: reusing") == 2, rec[:900] + t.logtext()[-1500:])
    rc = t.uninstall(kgame)
    t.check("K-first uninstall restores the game", rc == 0, t.logtext()[-900:])
    t.clear_model()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default=os.path.join(INSTALLER, "build-tests"))
    ap.add_argument("--setup", help="installer built from --prepare-only output (default: build it with build-mingw.sh)")
    ap.add_argument("--prepare-only", action="store_true", help="write the test payload and map, then stop")
    ap.add_argument("--preset-k", choices=("auto", "on", "off"), default="auto",
                    help="the installer under test was built with Preset K (on), without it (off: "
                         "-DWRONGGPU_PRESET_K=OFF, adds the checks of that build), or as the installer itself shows")
    ap.add_argument("--direct-route", choices=("auto", "on", "off"), default="auto",
                    help="the installer under test was built with the direct FSR route (on), without it (off: "
                         "-DWRONGGPU_DIRECT_ROUTE=OFF, OptiScaler is required), or as the installer itself shows")
    ap.add_argument("--direct-setup", help="for an installer without the direct FSR route: an installer with it (the "
                                           "same sources and test map, default build), which makes the earlier direct "
                                           "installs that the installer under test is then run over")
    a = ap.parse_args()
    work = os.path.abspath(a.work)
    # The synthetic source and its map: version 4 with the optional source F (spec 08 draft 2), variant 1 in the file.
    model = cases.f_mini_case(present=0)
    prep = os.path.join(work, "prepared")
    shutil.rmtree(prep, ignore_errors=True)
    payload(os.path.join(prep, "payload"))
    write(os.path.join(prep, "test-map.json"), json.dumps(model.map, indent=1))
    if a.prepare_only:
        print("payload: %s\nmap: %s" % (os.path.join(prep, "payload"), os.path.join(prep, "test-map.json")))
        return 0
    setup = a.setup
    if not setup:
        out = os.path.join(work, "build")
        subprocess.run([os.path.join(INSTALLER, "build-mingw.sh"), os.path.join(prep, "payload"), os.path.join(prep, "test-map.json"), "0.3.0-test", out], check=True)
        setup = os.path.join(out, "WrongGPU.exe")  # the launcher; its /switches are the installer's
    env = Env(work)
    for d in ("games", "steam", "epic", "gog", "elsewhere"):
        shutil.rmtree(os.path.join(work, d), ignore_errors=True)
    shutil.rmtree(os.path.join(env.localappdata(), "WrongGPU"), ignore_errors=True)
    # Registry entries left by an earlier run: only those pointing into the work folder.
    for k, v in env.reg_keys().items():
        if k.startswith(("WrongGPU-", "DLSS45AMD")) and v.get("InstallLocation", "").lower().startswith(env.win(work).lower()):
            env.reg("delete", env.uninstall() + "\\" + k, "/f")
    shutil.rmtree(os.path.join(env.programdata(), "Epic"), ignore_errors=True)
    if not env.native:  # the Wine prefix is ours alone
        env.reg("delete", r"HKCU\Software\Valve\Steam", "/f")
        env.reg("delete", r"HKLM\SOFTWARE\GOG.com", "/f", "/reg:32")
    t = Tests(env, setup, work, model)
    t0 = time.time()
    preset_k = has_preset_k(setup)
    print("installer under test: %s Preset K" % ("with" if preset_k else "without"))
    if a.preset_k != "auto":
        t.check("the installer is built %s Preset K, as --preset-k=%s expects" % ("with" if a.preset_k == "on" else "without", a.preset_k),
                preset_k == (a.preset_k == "on"))
    direct = has_direct_route(setup)
    print("installer under test: %s the direct FSR route" % ("with" if direct else "without"))
    if a.direct_route != "auto":
        t.check("the installer is built %s the direct FSR route, as --direct-route=%s expects" % ("with" if a.direct_route == "on" else "without", a.direct_route),
                direct == (a.direct_route == "on"))
    if not direct:  # the cases that are not about the route install through a mock OptiScaler package
        t.route = ("/OPTISCALER=" + env.win(optiscaler_package(t, "optiscaler-route")),)
    try:
        if not preset_k:
            test_without_preset_k(t, os.path.join(prep, "payload"))
        if direct:
            run_all(t, os.path.join(prep, "payload"))
        else:
            print("\n(the cases of the direct FSR route are skipped: [1] to [11] and [14])")
            test_without_direct_route(t, os.path.join(prep, "payload"), a.direct_setup)
        if preset_k:
            test_preset_k(t)
        test_hip_probe(t)
        test_game_exe(t)
        test_anticheat_scope(t)
        test_amd_provider(t)
        test_asi_plugin(t, os.path.join(prep, "payload"))
        test_filter_group(t)
        if direct:
            test_upgrade_paths(t, os.path.join(prep, "payload"))
    finally:
        env.close()
    print("\n%d checks passed, %d failed (%.0f s)" % (t.passed, len(t.failures), time.time() - t0))
    for f in t.failures:
        print("  FAILED:", f)
    return 1 if t.failures else 0


if __name__ == "__main__":
    sys.exit(main())
