# SPDX-License-Identifier: MIT
"""Build and test the RR proxy (CPU only: no game; the probe test uses the software D3D12 adapter, WARP).

    python clean/rr_f/proxy/tools/build.py [--out DIR] [--binding FILE [--probe] [--hook]] [--no-tests]
                                           [--proxy-name NAME|interposer] [--probe-name NAME|interposer]

Default: builds, per Streamline version, the file-replacement proxy (release + probe), and ONE hook-mode ASI plugin
(release + probe, HOOK-MODE.md; release 0.6.0: built from the 2.14.1 table, it picks its route from the game
interface's version at run time) tested against the fake game of every version (tests/test_hook_dll.cpp). With
--binding FILE [--hook] it builds only that one (the hook plugin with --hook).

Default: builds, per Streamline version, from the real tables (binding/rr-f-2.7.json, binding/rr-f-2.8.json, binding/rr-f-2.14.json,
annexes A9 / A10, the names of binding/rr-functions.json and binding/interposer.json):
  - the release proxy (proxy_file_name, default wg_rr_proxy.dll; never contains the probe), and
  - the probe proxy (build flag WG_RR_PROBE, probe_file_name, default wg_rr_proxy_probe.dll),
and runs every test:
  1. tests/test_gen_forwarding.py, tests/test_optimal.py, tests/test_capture_replay.py (Python unittest; CPU)
  2. test_proxy_core.exe (release logic against a fake original and a fake runtime), test_probe_core.exe (probe
     logic and log rules against a fake original and a fake GPU) and test_replay_profile.exe once per game profile
     tests/data/*-traffic.json (an in-game probe run replayed into the release logic; prints the runtime gaps)
  3. per version: a fake original DLL with the real export names; the release proxy with test_proxy_dll.exe (with
     and without the fake runtime) and the probe proxy with test_probe_dll.exe (fake game host on WARP), each
     staged under the game's interposer file name next to the renamed fake original.
--proxy-name / --probe-name set the built file names ("interposer" = the game's interposer file name), so that
the install tool can take the file as is. Every proxy carries a VERSIONINFO resource (gen_forwarding.emit_rc: file
version = the game's interposer version, WrongGPU's own strings), read back after the link. Prints every built proxy
with its SHA-256 and FileVersion and writes install.json per
version (what the install tool does). With --binding <table> it only generates and builds that table.
"""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PROXY = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import gen_forwarding as gf  # noqa: E402

VSDEVCMD = os.environ.get("VSDEVCMD", r"C:\Program Files\Microsoft Visual Studio\18\Community\Common7\Tools\VsDevCmd.bat")
DEFAULT_OUT = os.environ.get("WG_RR_PROXY_BUILD", os.path.join(os.environ.get("WG_BUILD_ROOT", os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))), "build")), "rr-f-proxy", "out"))
CORE = ["proxy_core.cpp", "settings.cpp", "formats.cpp", "optimal.cpp", "answers.cpp"]
PROBE = ["probe.cpp", "probe_log.cpp", "probe_gpu_d3d12.cpp"]  # probe build only
SHADERS = (("WG_PROBE_UPSCALE", "kProbeUpscaleCs", "probe_upscale_cs.h"),
           ("WG_PROBE_PATCH", "kProbePatchCs", "probe_patch_cs.h"))
CLFLAGS = "/nologo /EHsc /std:c++17 /O2 /MD /W3 /wd4996 /DNOMINMAX"
VERSIONS = (7, 8, 14)
PROBE_LOG = "wronggpu-rr-probe.log"


def sh(cmd, cwd):
    r = subprocess.run(f'"{VSDEVCMD}" -arch=x64 -host_arch=x64 >nul && {cmd}', shell=True, cwd=cwd,
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(f"command failed in {cwd}:\n{cmd}\n{r.stdout}\n{r.stderr}")
    return r.stdout


def src(*p):
    return '"' + os.path.join(PROXY, *p) + '"'


def output_name(binding, choice):
    """None (the table's name), the game's interposer file name ("interposer"), or a literal file name."""
    if choice != "interposer":
        return choice
    b, _ = gf.load(binding)
    return b["interposer_file_name"]


def build_proxy(binding, out, probe=False, name=None):
    gen = os.path.join(out, "gen")
    n = gf.generate(binding, gen, probe, name)
    obj = os.path.join(out, "obj")
    os.makedirs(obj, exist_ok=True)
    sh(f'ml64 /nologo /c /Fo"{os.path.join(obj, "stubs.obj")}" "{os.path.join(gen, "wg_forward_stubs.asm")}"', out)
    files = CORE + (PROBE if probe else [])
    if probe:
        for define, var, header in SHADERS:
            sh(f'fxc /nologo /T cs_5_1 /E main /D {define}=1 /Vn {var} /Fh "{os.path.join(gen, header)}" '
               f'{src("src", "probe_upscale.hlsl")}', out)
    res = os.path.join(obj, "wg_version.res")  # VERSIONINFO (gen_forwarding.emit_rc)
    sh(f'rc /nologo /fo "{res}" "{os.path.join(gen, "wg_version.rc")}"', out)
    sources = " ".join(src("src", c) for c in files)
    flags = CLFLAGS + (" /DWG_RR_PROBE=1" if probe else "")
    built = os.path.join(out, n["proxy_file"])
    sh(f'cl {flags} /LD /I"{gen}" {src("src", "proxy_dll.cpp")} {sources} "{os.path.join(obj, "stubs.obj")}" '
       f'/Fo"{obj}\\\\" /Fe"{built}" /link /DEF:"{os.path.join(gen, "wg_proxy.def")}" "{res}"',
       out)
    want = ".".join(str(x) for x in n["file_version"])
    got = file_version(built)
    if got != want:
        raise SystemExit(f"{built}: FileVersion reads back {got!r}, the binding says {want}")
    return n, gen


def file_version(path):
    """The FileVersion of a built DLL as Windows reports it (GetFileVersionInfoW / VerQueryValueW of the fixed part),
    "a.b.c.d"; None without a version resource."""
    import ctypes
    from ctypes import wintypes
    ver = ctypes.WinDLL("version")
    ver.GetFileVersionInfoSizeW.argtypes = [wintypes.LPCWSTR, ctypes.POINTER(wintypes.DWORD)]
    ver.GetFileVersionInfoW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, ctypes.c_void_p]
    ver.VerQueryValueW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR, ctypes.POINTER(ctypes.c_void_p),
                                   ctypes.POINTER(wintypes.UINT)]
    size = ver.GetFileVersionInfoSizeW(path, None)
    if not size:
        return None
    buf = ctypes.create_string_buffer(size)
    if not ver.GetFileVersionInfoW(path, 0, size, buf):
        return None
    p, n = ctypes.c_void_p(), wintypes.UINT()
    if not ver.VerQueryValueW(buf, "\\", ctypes.byref(p), ctypes.byref(n)) or n.value < 52:
        return None
    f = (wintypes.DWORD * 13).from_address(p.value)  # VS_FIXEDFILEINFO
    if f[0] != 0xFEEF04BD:
        return None
    ms, ls = f[2], f[3]
    return f"{ms >> 16}.{ms & 0xFFFF}.{ls >> 16}.{ls & 0xFFFF}"


def write_imports(minor, out):
    """Imports file for the DLL tests: the games of binding/game-imports.json whose interposer is 2.<minor>."""
    with open(os.path.join(PROXY, "binding", "game-imports.json"), "rb") as f:
        gi = json.loads(f.read().decode("utf-8"))
    lines = []
    for g in gi["games"]:
        if int(g["interposer_version"].split(".")[1]) == minor:
            lines += [f"game {g['game']}", f"version {g['sdk_version_value']}"] + g["imports"]
    if not lines:
        raise SystemExit(f"game-imports.json lists no game with a 2.{minor} interposer")
    path = os.path.join(out, f"imports-2.{minor}.txt")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    return path


def stage_names(n):
    """The game-like stage: the proxy under the game's interposer file name, the original renamed next to it."""
    proxy = n["interposer_file"]
    of = n["original_file"]
    original = os.path.splitext(proxy)[0] + of[len(gf.STEM):] if of.startswith(gf.STEM) else of
    return proxy, original


def fake_original(binding, out, n):
    fake = os.path.join(out, "fake")
    _, orig_name = stage_names(n)
    r = subprocess.run([sys.executable, os.path.join(PROXY, "tests", "gen_fake_original.py"), binding, fake,
                        orig_name], capture_output=True, text=True)
    if r.returncode:
        raise SystemExit(r.stdout + r.stderr)
    sh(f'cl {CLFLAGS} /LD "{os.path.join(fake, "fake_original.cpp")}" /Fo"{fake}\\\\" '
       f'/Fe"{os.path.join(fake, orig_name)}" /link /DEF:"{os.path.join(fake, "fake_original.def")}"', out)
    return fake, orig_name


def run_stage(exe, built, fake, orig_name, n, stage, extra, logs, label):
    shutil.rmtree(stage, ignore_errors=True)
    os.makedirs(stage)
    proxy_name, _ = stage_names(n)
    shutil.copy(built, os.path.join(stage, proxy_name))  # the install tool's copy under the interposer file name
    shutil.copy(os.path.join(fake, orig_name), stage)
    for f in extra:
        shutil.copy(f, stage)
    return stage, proxy_name


def build_release(binding, out, logs, imports, name):
    n, gen = build_proxy(binding, out, False, name)
    built = os.path.join(out, n["proxy_file"])
    with open(built, "rb") as f:
        blob = f.read()
    if PROBE_LOG.encode("utf-16-le") in blob or PROBE_LOG.encode() in blob:
        raise SystemExit(f"{built}: the release build contains the probe")
    fake, orig_name = fake_original(binding, out, n)
    sh(f'cl {CLFLAGS} /LD {src("tests", "fake_runtime_dll.cpp")} /Fo"{fake}\\\\" '
       f'/Fe"{os.path.join(fake, "wronggpu_runtime.dll")}"', out)
    exe = os.path.join(out, "test_proxy_dll.exe")
    sh(f'cl {CLFLAGS} /I"{gen}" {src("tests", "test_proxy_dll.cpp")} /Fo"{out}\\\\" /Fe"{exe}" /link psapi.lib', out)
    for runtime in (True, False):
        extra = [os.path.join(fake, "wronggpu_runtime.dll")] if runtime else []
        stage, pname = run_stage(exe, built, fake, orig_name, n,
                                 os.path.join(out, "stage" if runtime else "stage-no-runtime"), extra, logs, "")
        args = [exe, stage, pname] + ([] if runtime else ["--no-runtime"]) + ["--imports", imports]
        r = subprocess.run(args, capture_output=True, text=True, timeout=120)
        logs.append(f"[2.{n['minor']} release{'' if runtime else ', no runtime'}] {r.stdout.strip()}"
                    + (f"\n{r.stderr.strip()}" if r.stderr.strip() else ""))
        if r.returncode:
            raise SystemExit("\n".join(logs))
        if os.path.exists(os.path.join(stage, PROBE_LOG)):
            raise SystemExit("the release build wrote a probe log")
    return n, built


def build_probe(binding, out, logs, imports, name):
    n, gen = build_proxy(binding, out, True, name)
    built = os.path.join(out, n["proxy_file"])
    fake, orig_name = fake_original(binding, out, n)
    exe = os.path.join(out, "test_probe_dll.exe")
    sh(f'cl {CLFLAGS} /I"{gen}" {src("tests", "test_probe_dll.cpp")} /Fo"{out}\\\\" /Fe"{exe}" '
       f'/link d3d12.lib dxgi.lib psapi.lib', out)
    stage, pname = run_stage(exe, built, fake, orig_name, n, os.path.join(out, "stage"), [], logs, "")
    r = subprocess.run([exe, stage, pname, "--imports", imports], capture_output=True, text=True, timeout=600)
    logs.append(f"[2.{n['minor']} probe] {r.stdout.strip()}" + (f"\n{r.stderr.strip()}" if r.stderr.strip() else ""))
    if r.returncode:
        raise SystemExit("\n".join(logs))
    return n, built


def build_hook(binding, out, probe, name):
    """Build the hook-mode ASI plugin (src/hook_dll.cpp; HOOK-MODE.md): the proxy core, settings and (probe) the
    probe, with the transport that patches the game interface's import tables. No forwarding stubs."""
    gen = os.path.join(out, "gen")
    n = gf.generate(binding, gen, probe, name, hook=True)
    obj = os.path.join(out, "obj")
    os.makedirs(obj, exist_ok=True)
    if probe:
        for define, var, header in SHADERS:
            sh(f'fxc /nologo /T cs_5_1 /E main /D {define}=1 /Vn {var} /Fh "{os.path.join(gen, header)}" '
               f'{src("src", "probe_upscale.hlsl")}', out)
    res = os.path.join(obj, "wg_version.res")
    sh(f'rc /nologo /fo "{res}" "{os.path.join(gen, "wg_version.rc")}"', out)
    files = CORE + (PROBE if probe else ["rr_panel.cpp", "ngx_rr.cpp"])  # the menu tab, the NGX route: release only
    files = files + ["setupapi_identity.cpp"]  # clean/rr-sm2-shim: the SetupAPI identity shim (Spider-Man2.exe only)
    sources = " ".join(src("src", c) for c in files)
    flags = CLFLAGS + (" /DWG_RR_PROBE=1" if probe else "")
    built = os.path.join(out, n["proxy_file"])
    sh(f'cl {flags} /LD /I"{gen}" {src("src", "hook_dll.cpp")} {sources} /Fo"{obj}\\\\" /Fe"{built}" '
       f'/link /DEF:"{os.path.join(gen, "wg_hook.def")}" "{res}" version.lib psapi.lib'
       + (" d3d12.lib dxgi.lib" if probe else ""), out)
    want = ".".join(str(x) for x in list(n["release"]) + [0])  # the hook plugin carries WrongGPU's release version
    got = file_version(built)
    if got != want:
        raise SystemExit(f"{built}: FileVersion reads back {got!r}, WrongGPU's release version is {want}")
    blob = open(built, "rb").read()
    if not probe and (PROBE_LOG.encode("utf-16-le") in blob or PROBE_LOG.encode() in blob):
        raise SystemExit(f"{built}: the release hook plugin contains the probe")
    return n, gen, built


def fake_interface(binding, out, n, file_version_parts):
    """A stand-in game interface DLL under the interposer file name with the real export names (A9.3), an import
    library so the fake game can import it statically, and a version resource with `file_version_parts` (the hook
    plugin's version gate reads it). Our own stand-in; no real library is read."""
    os.makedirs(out, exist_ok=True)
    proxy_name, orig_unused = stage_names(n)  # proxy_name = the interposer file name
    r = subprocess.run([sys.executable, os.path.join(PROXY, "tests", "gen_fake_original.py"), binding, out,
                        proxy_name], capture_output=True, text=True)
    if r.returncode:
        raise SystemExit(r.stdout + r.stderr)
    rc = os.path.join(out, "fake_iface.rc")
    comma = ",".join(str(x) for x in file_version_parts)
    with open(rc, "w") as f:
        f.write('#include <winver.h>\nVS_VERSION_INFO VERSIONINFO\n'
                f' FILEVERSION {comma}\n PRODUCTVERSION {comma}\n'
                ' FILEFLAGSMASK VS_FFI_FILEFLAGSMASK\n FILEFLAGS 0\n FILEOS VOS_NT_WINDOWS32\n FILETYPE VFT_DLL\n'
                'BEGIN\n BLOCK "StringFileInfo"\n BEGIN\n  BLOCK "040904b0"\n  BEGIN\n'
                f'   VALUE "FileVersion", "{".".join(str(x) for x in file_version_parts)}"\n'
                '  END\n END\n BLOCK "VarFileInfo"\n BEGIN\n  VALUE "Translation", 0x409, 1200\n END\nEND\n')
    res = os.path.join(out, "fake_iface.res")
    sh(f'rc /nologo /fo "{res}" "{rc}"', out)
    dll = os.path.join(out, proxy_name)
    sh(f'cl {CLFLAGS} /LD "{os.path.join(out, "fake_original.cpp")}" /Fo"{out}\\\\" /Fe"{dll}" '
       f'/link /DEF:"{os.path.join(out, "fake_original.def")}" "{res}" '
       f'/IMPLIB:"{os.path.join(out, "fake_ord.lib")}"', out)
    # A by-NAME import library (the supported games import by name, A9.2): build it from a names-only module
    # definition with lib.exe, so the fake game's import table carries names, not ordinals, for the ASI to match.
    names = [l.strip().split("=")[0].split(" ")[0]
             for l in open(os.path.join(out, "fake_original.def")).read().splitlines() if l.startswith("  ")]
    names_def = os.path.join(out, "fake_byname.def")
    with open(names_def, "w") as f:
        f.write(f'LIBRARY "{proxy_name}"\nEXPORTS\n' + "\n".join("  " + x for x in names) + "\n")
    implib = os.path.join(out, "fake_interface.lib")
    sh(f'lib /nologo /def:"{names_def}" /machine:x64 /out:"{implib}"', out)
    return dll, implib, proxy_name


def build_hook_and_test(binding, out, logs, probe):
    """Build the hook ASI from `binding` and test it with that table's fake game (the --binding path)."""
    n, gen, built = build_hook(binding, out, probe, None)
    test_hook(built, binding, out, logs, probe, n, gen)
    return n, built


# release 0.6.0 (owner 2026-11-02: one RR plugin): the hook plugin is built ONCE, from the 2.14.1 table (the complete role
# list), and picks its route from the game interface's version at run time (src/hook_dll.cpp); it is tested against
# the fake game of every version below (each test program built with its own version's table)
HOOK_TABLE_MINOR = 14


def test_hook(built, binding, out, logs, probe, n=None, gen=None):
    """Drive the built hook ASI through a fake game of `binding`'s version whose import table it patches. Release: a
    fake runtime. Probe: a short WARP sequence. Plus a wrong-version refusal run (release only)."""
    if n is None:
        gen = os.path.join(out, "gen")
        n = gf.generate(binding, gen, probe, None, hook=True)
    os.makedirs(out, exist_ok=True)
    minor = n["minor"]
    # the stand-in interface and its import library
    fi = os.path.join(out, "iface")
    good_fv = list(n["file_version"])  # the interface version the plugin requires (2.minor.1.0)
    iface_dll, implib, iface_name = fake_interface(binding, fi, n, good_fv)
    # the fake game (imports the interface statically; links against the stand-in import library), and a second one
    # that links the interface by delay-load import instead (the interface is then not loaded when the ASI runs)
    libs = "version.lib psapi.lib" + (" d3d12.lib dxgi.lib" if probe else "")
    for delay in (False, True):
        sub = os.path.join(out, "delay") if delay else out
        os.makedirs(sub, exist_ok=True)
        dexe = os.path.join(sub, "test_hook_dll.exe")
        sh(f'cl {CLFLAGS} {"/DWG_RR_PROBE=1 " if probe else ""}{"/DWG_TEST_DELAYLOAD=1 " if delay else ""}'
           f'/I"{gen}" {src("tests", "test_hook_dll.cpp")} "{implib}" /Fo"{sub}\\\\" /Fe"{dexe}" /link {libs}'
           + (f' delayimp.lib /DELAYLOAD:{iface_name}' if delay else ''), out)
    exe = os.path.join(out, "test_hook_dll.exe")
    delay_exe = os.path.join(out, "delay", "test_hook_dll.exe")
    # stage: the fake game, the stand-in under the interface name, the ASI, and (release) the fake runtime
    stage = os.path.join(out, "stage")
    shutil.rmtree(stage, ignore_errors=True)
    os.makedirs(stage)
    shutil.copy(exe, stage)
    shutil.copy(iface_dll, stage)
    shutil.copy(built, os.path.join(stage, n["proxy_file"]))
    if not probe:
        rt = os.path.join(out, "wronggpu_runtime.dll")
        sh(f'cl {CLFLAGS} /LD {src("tests", "fake_runtime_dll.cpp")} /Fo"{out}\\\\" /Fe"{rt}"', out)
        shutil.copy(rt, stage)
    staged_exe = os.path.join(stage, "test_hook_dll.exe")
    r = subprocess.run([staged_exe, stage], capture_output=True, text=True, timeout=600)
    kind = "probe" if probe else "release"
    logs.append(f"[2.{minor} hook {kind}] {r.stdout.strip()}" + (f"\n{r.stderr.strip()}" if r.stderr.strip() else ""))
    if r.returncode:
        raise SystemExit("\n".join(logs))
    if not probe:
        # the menu tab's exact mode: allowed by the launcher (three modes), and asked for without the permission
        # rr_enabled = 0 (the launcher's "Use WrongGPU" switch, or the plugin's own ini): RR reported unsupported
        # clean/060-rr-plugin-hunt: rr_mode from WrongGPU\wronggpu.ini (with and without the shared settings block) and
        # from the legacy line of the plugin's wronggpu-rr.ini; wronggpu-rr.log at its size cap (one rotation to .log.old)
        for flag in ("--exact", "--exact-denied", "--rr-off", "--rr-off-own", "--fast", "--fast-noblock", "--legacy-mode",
                     "--log-cap"):
            r = subprocess.run([staged_exe, stage, flag], capture_output=True, text=True, timeout=600)
            logs.append(f"[2.{minor} hook release {flag}] {r.stdout.strip()}" + (f"\n{r.stderr.strip()}" if r.stderr.strip() else ""))
            if r.returncode:
                raise SystemExit("\n".join(logs))
        # wrong interface version: the plugin refuses, nothing is patched
        bad = os.path.join(out, "stage-badver")
        shutil.rmtree(bad, ignore_errors=True)
        os.makedirs(bad)
        bad_iface = fake_interface(binding, os.path.join(out, "iface-bad"), n, [9, 9, 9, 9])[0]
        shutil.copy(exe, bad)
        shutil.copy(bad_iface, os.path.join(bad, iface_name))
        shutil.copy(built, os.path.join(bad, n["proxy_file"]))
        r = subprocess.run([os.path.join(bad, "test_hook_dll.exe"), bad, "--badver"], capture_output=True, text=True,
                           timeout=300)
        logs.append(f"[2.{minor} hook release, wrong version] {r.stdout.strip()}"
                    + (f"\n{r.stderr.strip()}" if r.stderr.strip() else ""))
        if r.returncode:
            raise SystemExit("\n".join(logs))
    # the same runs with the delay-load fake game
    for bad_run in ((False, True) if not probe else (False,)):
        dstage = os.path.join(out, "stage-delay-badver" if bad_run else "stage-delay")
        shutil.rmtree(dstage, ignore_errors=True)
        os.makedirs(dstage)
        shutil.copy(delay_exe, dstage)
        if bad_run:
            shutil.copy(os.path.join(out, "iface-bad", os.path.basename(iface_dll)), os.path.join(dstage, iface_name))
        else:
            shutil.copy(iface_dll, dstage)
        shutil.copy(built, os.path.join(dstage, n["proxy_file"]))
        if not probe:
            shutil.copy(os.path.join(out, "wronggpu_runtime.dll"), dstage)
        r = subprocess.run([os.path.join(dstage, "test_hook_dll.exe"), dstage] + (["--badver"] if bad_run else []),
                           capture_output=True, text=True, timeout=600)
        logs.append(f"[2.{minor} hook {kind}, delay-load{', wrong version' if bad_run else ''}] {r.stdout.strip()}"
                    + (f"\n{r.stderr.strip()}" if r.stderr.strip() else ""))
        if r.returncode:
            raise SystemExit("\n".join(logs))
        if not probe and not bad_run:  # rr_enabled = 0 with the delay-load game too
            r = subprocess.run([os.path.join(dstage, "test_hook_dll.exe"), dstage, "--rr-off"],
                               capture_output=True, text=True, timeout=600)
            logs.append(f"[2.{minor} hook {kind}, delay-load --rr-off] {r.stdout.strip()}"
                        + (f"\n{r.stderr.strip()}" if r.stderr.strip() else ""))
            if r.returncode:
                raise SystemExit("\n".join(logs))
    return n, built


def sha256(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--binding")
    ap.add_argument("--probe", action="store_true", help="with --binding: build the probe")
    ap.add_argument("--hook", action="store_true", help="with --binding: build the hook-mode ASI plugin")
    ap.add_argument("--no-tests", action="store_true")
    ap.add_argument("--proxy-name", help="file name of the built release proxy, or 'interposer'")
    ap.add_argument("--probe-name", help="file name of the built probe proxy, or 'interposer'")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    if a.binding:
        try:
            if a.hook:
                sub = "hook-probe" if a.probe else "hook"
                n, _, _ = build_hook(a.binding, os.path.join(a.out, sub), a.probe, None)
            else:
                name = output_name(a.binding, a.probe_name if a.probe else a.proxy_name)
                n, _ = build_proxy(a.binding, os.path.join(a.out, "probe" if a.probe else "release"), a.probe, name)
        except gf.BindingError as e:
            raise SystemExit(f"refused: {e}")
        print(f"built {n['proxy_file']} for API {n['api_version']}"
              f"{' (probe)' if a.probe else ''}{' (hook)' if a.hook else ''}")
        return
    logs = []
    core_dir = os.path.join(a.out, "core")
    os.makedirs(core_dir, exist_ok=True)
    core = " ".join(src("src", c) for c in CORE)
    probe_src = " ".join(src("src", c) for c in PROBE)
    exe = os.path.join(core_dir, "test_proxy_core.exe")
    sh(f'cl {CLFLAGS} {src("tests", "test_proxy_core.cpp")} {core} /Fo"{core_dir}\\\\" /Fe"{exe}" /link psapi.lib',
       core_dir)
    rexe = os.path.join(core_dir, "test_replay_profile.exe")
    robj = os.path.join(core_dir, "replay-obj")
    os.makedirs(robj, exist_ok=True)
    sh(f'cl {CLFLAGS} {src("tests", "test_replay_profile.cpp")} {core} /Fo"{robj}\\\\" /Fe"{rexe}"', core_dir)
    # clean/rr-ngx: the NGX RR provider (spec page 66) against a fake runtime and a mock parameter block, and the same
    # four frames through both routes (tests/data/cp2077-2.14-rr-frames.json: synthetic since the public snapshot)
    nexe = os.path.join(core_dir, "test_ngx_rr.exe")
    nobj = os.path.join(core_dir, "ngx_obj")
    os.makedirs(nobj, exist_ok=True)
    sh(f'cl {CLFLAGS} {src("tests", "test_ngx_rr.cpp")} {src("src", "ngx_rr.cpp")} {core} /Fo"{nobj}\\\\" /Fe"{nexe}"',
       core_dir)
    # clean/rr-sm2-shim: the SetupAPI identity shim (synthetic IDs; the import-slot part reads the PC's Display-class
    # devices through SetupAPI only, no GPU device)
    sexe = os.path.join(core_dir, "test_setupapi_identity.exe")
    sobj = os.path.join(core_dir, "setupapi-obj")
    os.makedirs(sobj, exist_ok=True)
    sh(f'cl {CLFLAGS} {src("tests", "test_setupapi_identity.cpp")} {src("src", "setupapi_identity.cpp")} /Fo"{sobj}\\\\" '
       f'/Fe"{sexe}"', core_dir)
    pexe = os.path.join(core_dir, "test_probe_core.exe")
    pobj = os.path.join(core_dir, "probe-obj")
    os.makedirs(pobj, exist_ok=True)
    sh(f'cl {CLFLAGS} {src("tests", "test_probe_core.cpp")} {core} {probe_src} /Fo"{pobj}\\\\" /Fe"{pexe}"',
       core_dir)
    if not a.no_tests:
        r = subprocess.run([sys.executable, "-m", "unittest", "-v", "test_gen_forwarding", "test_optimal", "test_capture_replay"],
                           cwd=os.path.join(PROXY, "tests"), capture_output=True, text=True,
                           env=dict(os.environ, WG_RR_PROXY_BUILD=a.out))
        tail = [l for l in r.stderr.strip().splitlines() if l.startswith(("Ran ", "OK", "FAILED"))]
        logs.append("[python] " + " ".join(tail))
        if r.returncode:
            raise SystemExit(r.stderr)
        profiles = sorted(os.path.join(PROXY, "tests", "data", f) for f in os.listdir(os.path.join(PROXY, "tests", "data"))
                          if f.endswith("-traffic.json"))
        runs = [("core", [exe]), ("probe core", [pexe]), ("setupapi identity", [sexe]),
                ("NGX provider", [nexe, os.path.join(PROXY, "tests", "data", "cp2077-2.14-rr-frames.json")])] + [(f"replay {os.path.basename(f)}", [rexe, f]) for f in profiles]
        for label, e in runs:
            r = subprocess.run(e, capture_output=True, text=True, timeout=300)
            logs.append(f"[{label}] " + r.stdout.strip() + (("\n" + r.stderr.strip()) if r.stderr.strip() else ""))
            if r.returncode:
                raise SystemExit("\n".join(logs))
    built = []
    # the one hook plugin (release + probe), from the 2.14.1 table
    hook_table = os.path.join(PROXY, "binding", f"rr-f-2.{HOOK_TABLE_MINOR}.json")
    nhk, _, hook = build_hook(hook_table, os.path.join(a.out, "hook"), False, None)
    nhp, _, hook_probe = build_hook(hook_table, os.path.join(a.out, "hook-probe"), True, None)
    built += [("2.7 / 2.8 / 2.14 (one build)", "hook", hook), ("2.7 / 2.8 / 2.14 (one build)", "hook-probe", hook_probe)]
    for minor in VERSIONS:
        vout = os.path.join(a.out, f"2.{minor}")
        os.makedirs(vout, exist_ok=True)
        imports = write_imports(minor, vout)
        table = os.path.join(PROXY, "binding", f"rr-f-2.{minor}.json")
        nr, release = build_release(table, os.path.join(vout, "release"), logs, imports,
                                    output_name(table, a.proxy_name))
        npb, probe = build_probe(table, os.path.join(vout, "probe"), logs, imports, output_name(table, a.probe_name))
        test_hook(hook, table, os.path.join(vout, "hook-test"), logs, False)
        test_hook(hook_probe, table, os.path.join(vout, "hook-probe-test"), logs, True)
        proxy_as, original_as = stage_names(nr)
        install = {
            "streamline_version": nr["api_version"],
            "interposer_file": proxy_as,
            "rename_game_file_to": original_as,
            "release": {"file": os.path.relpath(release, vout), "sha256": sha256(release), "copy_as": proxy_as,
                        "file_version": file_version(release)},
            "probe": {"file": os.path.relpath(probe, vout), "sha256": sha256(probe), "copy_as": proxy_as,
                      "file_version": file_version(probe), "log": PROBE_LOG},
            "hook": {"file": os.path.relpath(hook, vout), "sha256": sha256(hook), "copy_as": nhk["proxy_file"],
                     "file_version": file_version(hook), "log": "wronggpu-rr.log"},
            "hook_probe": {"file": os.path.relpath(hook_probe, vout), "sha256": sha256(hook_probe),
                           "copy_as": nhp["proxy_file"], "file_version": file_version(hook_probe), "log": PROBE_LOG},
            "steps": [f"file-replacement proxy: in the directory of the game's {proxy_as}: rename it to {original_as} "
                      f"(keep it), copy the release (or probe) proxy there as {proxy_as}",
                      "release proxy only: wronggpu_runtime.dll (with the RR runtime) and the model next to it (until then RR reports 7)",
                      f"file-replacement uninstall: delete {proxy_as}, rename {original_as} back",
                      f"hook-mode plugin (does not replace {proxy_as}): copy {nhk['proxy_file']} (or "
                      f"{nhp['proxy_file']}) into OptiScaler's plugins folder; keep wronggpu_runtime.dll and "
                      f"wronggpu-rr.ini next to the game exe (HOOK-MODE.md)"],
        }
        with open(os.path.join(vout, "install.json"), "w") as f:
            json.dump(install, f, indent=1)
        built += [(nr["api_version"], "release", release), (npb["api_version"], "probe", probe)]
    print("\n".join(logs))
    for api, kind, path in built:
        print(f"{kind} {api}: {path} sha256 {sha256(path)} FileVersion {file_version(path)}")


if __name__ == "__main__":
    main()
