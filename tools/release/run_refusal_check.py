# SPDX-License-Identifier: MIT
"""Release check B8: build and run the real-pixel integration test of the refused-frame paths
(clean/fsr/tests/test_refusal_integration.cpp, #228) against a build's runtime DLL.

The test drives the front end with the REAL network backend (the build's wronggpu_runtime.dll, executor=dx12-test), for
model K and the model M control: AMD FSR present, AMD absent with the stand-in working, and AMD absent with the stand-in
unavailable (the backend's bilinear last resort, checked on every output pixel). It needs a D3D12 device; WARP is
enough, so no GPU slot is needed. It fails (exit 2) without the DLL, so this check cannot pass by skipping.

    python tools/release/run_refusal_check.py --runtime-dll <build>\\wronggpu_runtime.dll --work <scratch folder>
        [--cxx <clang++>] [--keep]

Steps, from the repository root (the test file's header lists the same commands):
  1. python clean/wgshared/tools/gen_registry.py --out-dir <work>\\gen        (the registry header the front end needs)
  2. clang++ (C++17, -O1) of the test and the front end's sources, linked with d3d12 and dxgi, inside the
     Visual Studio developer environment (vswhere, VsDevCmd -arch=x64) for the Windows SDK libraries
  3. test_refusal_integration.exe <runtime DLL>
For the test run (step 3), APPDATA, LOCALAPPDATA, ProgramData (and ALLUSERSPROFILE), TMP and TEMP point at
<work>\\scratch, so the test writes nothing outside the work folder. The compile (step 2) moves only TMP and TEMP there:
VsDevCmd needs the real ProgramData to find Visual Studio. The result (commands, exit codes, the test's summary lines) goes to
<work>\\refusal-check.json. Exit code: the test's (0 = pass); 3 when a step before the test fails; 4 for bad arguments.

Written for this project; Python standard library only."""
import argparse
import glob
import json
import os
import shutil
import subprocess
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
TEST = os.path.join("clean", "fsr", "tests", "test_refusal_integration.cpp")
FE_SOURCES = ["frontend", "param_map", "ffx_describe", "config", "fsr30_hook", "fsr30_map", "amd_compare", "async_create"]
VSWHERE = os.path.join(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)"), "Microsoft Visual Studio",
                       "Installer", "vswhere.exe")


def header_compile(test_path):
    """The compile command the test file's header gives ('//   clang++ ...' and its '//     ' continuation lines), as an
    argument list with {a,b}.cpp groups expanded and the '-o' part dropped; None when the header has none. The test's
    owner keeps that header current, so this check builds what the test itself documents."""
    lines = open(test_path, encoding="utf-8", errors="replace").read().splitlines()
    out, on = [], False
    for l in lines:
        if not l.startswith("//"):
            if on:
                break
            continue
        body = l[2:]
        if body.strip().startswith("clang++"):
            on = True
        elif on and not body.startswith("     "):
            break
        if on:
            out.append(body.strip())
    if not out:
        return None
    args = []
    for tok in " ".join(out).split():
        if "{" in tok and "}" in tok:
            pre, rest = tok.split("{", 1)
            mid, post = rest.split("}", 1)
            args += [pre + m + post for m in mid.split(",")]
        else:
            args.append(tok)
    if "-o" in args:
        i = args.index("-o")
        del args[i:i + 2]
    return args[1:]  # without "clang++"


def find_cxx(given):
    if given:
        return given
    if os.environ.get("CXX"):
        return os.environ["CXX"]
    rocm = sorted(glob.glob(r"C:\Program Files\AMD\ROCm\*\bin\clang++.exe"))
    if rocm:
        return rocm[-1]
    return shutil.which("clang++") or "clang++"


def find_vsdevcmd():
    if not os.path.isfile(VSWHERE):
        return None
    r = subprocess.run([VSWHERE, "-latest", "-products", "*", "-property", "installationPath"], capture_output=True,
                       text=True)
    path = os.path.join(r.stdout.strip(), "Common7", "Tools", "VsDevCmd.bat") if r.stdout.strip() else ""
    return path if os.path.isfile(path) else None


def scratch_env(work):
    s = os.path.join(work, "scratch")
    dirs = {"LOCALAPPDATA": os.path.join(s, "LocalAppData"), "APPDATA": os.path.join(s, "Roaming"),
            "ProgramData": os.path.join(s, "ProgramData"), "TMP": os.path.join(s, "tmp")}
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)
    env = {k: v for k, v in os.environ.items()
           if k.upper() not in ("LOCALAPPDATA", "APPDATA", "PROGRAMDATA", "ALLUSERSPROFILE", "TMP", "TEMP")}
    env["LOCALAPPDATA"], env["APPDATA"] = dirs["LOCALAPPDATA"], dirs["APPDATA"]
    env["ProgramData"] = env["ALLUSERSPROFILE"] = dirs["ProgramData"]
    env["TMP"] = env["TEMP"] = dirs["TMP"]
    return env


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runtime-dll", required=True, help="the build's wronggpu_runtime.dll")
    ap.add_argument("--work", required=True, help="a scratch folder for the build and the redirected folders")
    ap.add_argument("--cxx", help="clang++ (default: CXX, else the newest ROCm clang++, else clang++ on PATH)")
    ap.add_argument("--keep", action="store_true", help="keep an existing work folder's contents (default: must be new or empty)")
    a = ap.parse_args(argv)
    work = os.path.abspath(a.work)
    if os.path.isdir(work) and os.listdir(work) and not a.keep:
        print("work folder is not empty: %s (use a new one, or --keep)" % work)
        return 4
    os.makedirs(work, exist_ok=True)
    res = {"root": ROOT, "runtime_dll": os.path.abspath(a.runtime_dll), "steps": []}
    out_json = os.path.join(work, "refusal-check.json")

    def done(code):
        res["exit"] = code
        json.dump(res, open(out_json, "w"), indent=1)
        print("refusal check: exit %d (%s)" % (code, out_json))
        return code

    if not os.path.isfile(os.path.join(ROOT, TEST)):
        res["error"] = "%s is not in this tree (it comes with #228)" % TEST
        print(res["error"])
        return done(3)
    env = scratch_env(work)
    gen = os.path.join(work, "gen")
    step = [sys.executable, os.path.join(ROOT, "clean", "wgshared", "tools", "gen_registry.py"), "--out-dir", gen]
    r = subprocess.run(step, cwd=ROOT, env=env, capture_output=True, text=True)
    res["steps"].append({"step": "gen_registry", "cmd": step, "exit": r.returncode, "tail": (r.stdout + r.stderr)[-2000:]})
    if r.returncode:
        return done(3)
    exe = os.path.join(work, "test_refusal_integration.exe")
    hdr = header_compile(os.path.join(ROOT, TEST))
    if hdr:  # the test's own documented command (<gen> is this run's generated-header folder)
        compile_cmd = [find_cxx(a.cxx)] + [x.replace("<gen>", gen) for x in hdr] + ["-o", exe]
        res["compile_from"] = "the test file's header"
    else:
        srcs = [TEST, os.path.join("clean", "fsr", "src", "passthrough_d3d12.cpp")] + \
            [os.path.join("clean", "fsr", "src", s + ".cpp") for s in FE_SOURCES]
        compile_cmd = [find_cxx(a.cxx), "-std=c++17", "-O1", "-D_CRT_SECURE_NO_WARNINGS", "-I", os.path.join(gen, "include"),
                       "-I", os.path.join("clean", "wgshared", "include"), "-I", os.path.join("clean", "fsr", "include"),
                       "-I", os.path.join("clean", "third_party", "fidelityfx-sdk-1.1.4")] + srcs + ["-ld3d12", "-ldxgi", "-o", exe]
        res["compile_from"] = "the built-in list (the test header has no clang++ command)"
    vsdev = find_vsdevcmd()
    if vsdev:
        bat = os.path.join(work, "compile.bat")
        with open(bat, "w") as f:
            f.write('@call "%s" -arch=x64 -host_arch=x64 >nul\r\n' % vsdev)
            f.write(subprocess.list2cmdline(compile_cmd) + "\r\n")
        run = ["cmd.exe", "/d", "/c", bat]
    else:
        run = compile_cmd
    # the compile keeps the real ProgramData (Visual Studio's instance registry, which VsDevCmd reads); only TMP / TEMP
    # move to the scratch folder. The test run below gets the full redirect.
    cenv = dict(os.environ, TMP=env["TMP"], TEMP=env["TMP"])
    r = subprocess.run(run, cwd=ROOT, env=cenv, capture_output=True, text=True)
    res["steps"].append({"step": "compile", "cmd": compile_cmd, "vsdevcmd": vsdev, "exit": r.returncode,
                         "tail": (r.stdout + r.stderr)[-4000:]})
    if r.returncode or not os.path.isfile(exe):
        return done(3)
    test_cmd = [exe, os.path.abspath(a.runtime_dll)]
    r = subprocess.run(test_cmd, cwd=work, env=env, capture_output=True, text=True)
    lines = (r.stdout + r.stderr).splitlines()
    res["steps"].append({"step": "test", "cmd": test_cmd, "exit": r.returncode,
                         "summary": [l for l in lines if "FAIL" in l or "checks" in l.lower() or "pass" in l.lower()][-40:],
                         "tail": "\n".join(lines[-60:])})
    with open(os.path.join(work, "test-output.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines[-15:]))
    return done(r.returncode)


if __name__ == "__main__":
    sys.exit(main())
