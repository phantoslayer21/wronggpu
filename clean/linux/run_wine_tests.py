# SPDX-License-Identifier: MIT
"""WrongGPU's D3D12 tests on Linux, under Wine with the D3D12 stack of a Proton install (vkd3d-proton, DXVK's dxgi,
vkd3d-shader), the stack a game runs on in Steam.

    python3 clean/linux/run_wine_tests.py --proton <Proton dir> [--vkd3d <dir>] [--runtime <wronggpu_runtime.dll>]
                                          [--tests interop,rr-leak,ags-layout,launcher] [--work <dir>]

Tests (all run by default; each prints PASS, FAIL or SKIP with the reason; the exit status is 1 when one fails):
  interop     clean/runtime/interop/tests/run-tests.sh: MinGW build of the runtime with the d3d12-test executor, run
              in the prefix (copy / convert passes, split protocol, failure paths; the interop shaders compiled by
              Wine's d3dcompiler_47).
  rr-leak     the RR runtime test host's `tagloop` job against --runtime (a wronggpu_runtime.dll with the RR runtime,
              e.g. the release build) and a synthetic RR model (pack_model --synthetic): RR input copies retained and
              released every frame with no instance must not pile up in video memory.
  ags-layout  WrongGPU's driver-channel helpers (wg_ags.hlsli: FP16 and FP8 WMMA, matrix conversions, the F16
              conversion, a wave64 conversion) on fixed register values, against a CPU reference of the gfx12 register
              semantics. Under Proton the channel is translated by vkd3d-proton's dxil-spirv.
  launcher    the launcher's unit tests (clean/installer: wg-core-tests, launcher-core-tests, coexist-core-tests,
              wizard-policy-tests), a WRONGGPU_TESTING MinGW build with the installer suite's generated payload, run in
              the prefix: the launcher as Linux players run it, under Wine.
rr-leak and ags-layout need the WMMA channel, i.e. a vkd3d-proton with dxil-spirv's shader-packed fragment support
(https://github.com/HansKristian-Work/dxil-spirv/pull/362): pass a self-built one with --vkd3d until Proton ships it.

Tools (environment): MINGW_CXX (default x86_64-w64-mingw32-g++-posix, then x86_64-w64-mingw32-g++), WG_DXC (a Linux
dxc binary of the DXC release, default `dxc` on PATH). Python with NumPy. Needs an AMD RDNA 4 GPU with RADV. The
launcher test also needs CMake, Ninja and the MinGW-w64 windres beside MINGW_CXX.
"""
import argparse
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
import wine_env  # noqa: E402


def log(msg):
    print(msg, flush=True)


def mingw_cxx():
    for c in (os.environ.get("MINGW_CXX"), "x86_64-w64-mingw32-g++-posix", "x86_64-w64-mingw32-g++"):
        if c and shutil.which(c):
            return shutil.which(c)
    return None


def build_exe(cxx, sources, out, includes=(), libs=()):
    cmd = [cxx, "-std=c++17", "-O2", "-static", *sources, "-o", str(out), *[f"-I{i}" for i in includes], *libs]
    if "clang" in subprocess.run([cxx, "--version"], capture_output=True, text=True).stdout.lower():
        cmd.insert(1, "-fms-extensions")
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"building {out.name} failed:\n{r.stderr[-3000:]}")


def run_wine(env, args, cwd, timeout):
    r = subprocess.run([env["WG_WINE"], *args], cwd=cwd, env=env, capture_output=True, text=True, errors="replace",
                       timeout=timeout)
    return r.returncode, r.stdout.replace("\r", "") + r.stderr.replace("\r", "")


# ---- interop --------------------------------------------------------------------------------------------------------
def test_interop(a, env, work):
    cxx = mingw_cxx()
    if not cxx:
        return "SKIP", "no MinGW-w64 C++ compiler (MINGW_CXX)"
    out = work / "interop"
    e = dict(env, MINGW_CXX=cxx, WG_TEST_DIR=str(out))
    # run-tests.sh makes its own prefix under WG_TEST_DIR unless WINEPREFIX is set: use the prepared one
    r = subprocess.run(["bash", str(REPO / "clean/runtime/interop/tests/run-tests.sh")], env=e, capture_output=True,
                       text=True, errors="replace", timeout=3600)
    (out / "run.log").write_text(r.stdout + r.stderr)
    text = (out / "result.txt").read_text(errors="replace") if (out / "result.txt").exists() else ""
    passed = len(re.findall(r": passed$", text, re.M))
    failed = re.findall(r"^(\S+): FAILED$", text, re.M)
    if "ALL TESTS PASSED" in text and r.returncode == 0:
        return "PASS", f"{passed} tests passed"
    if text.startswith("SKIP") or "SKIP:" in text:
        return "FAIL", "the test skipped itself: " + text.strip().splitlines()[-1] + " (no D3D12 device in the prefix?)"
    return "FAIL", f"{passed} passed, {len(failed)} failed ({', '.join(failed[:6])}); see {out / 'result.txt'}"


# ---- rr-leak --------------------------------------------------------------------------------------------------------
def test_rr_leak(a, env, work):
    if not a.runtime:
        return "SKIP", "needs --runtime <wronggpu_runtime.dll> with the RR runtime linked in"
    cxx = mingw_cxx()
    if not cxx:
        return "SKIP", "no MinGW-w64 C++ compiler (MINGW_CXX)"
    out = work / "rr-leak"
    stage, run = out / "stage", out / "run"
    for d in (stage, run):
        d.mkdir(parents=True, exist_ok=True)
    host = out / "rr_test_host.exe"
    rr = REPO / "clean/rr_f"
    build_exe(cxx, [str(rr / "runtime/tests/rr_test_host.cpp")], host,
              includes=[rr / "runtime/src", rr / "proxy/include", rr / "proxy/src"], libs=["-ld3d12", "-ldxgi"])
    model = stage / "wronggpu_rr_model.bin"   # original synthetic model (no learned weights)
    if not model.exists():
        subprocess.run([sys.executable, "-m", "clean.rr_f.runtime.tools.pack_model", "--synthetic", "5100", "--out",
                        str(model)], cwd=REPO, check=True, capture_output=True)
    shutil.copyfile(a.runtime, stage / "wronggpu_runtime.dll")
    job = out / "job.txt"
    # the limit of the Windows end-to-end test (clean/rr_f/runtime/tests/e2e.py, TAGLOOP_LIMIT_MIB)
    job.write_text(f"outdir {wine_env.windows_path(out)}\nruntime {wine_env.windows_path(stage / 'wronggpu_runtime.dll')}\n"
                   f"tagloop 200 1706 960 256\nend\n")
    rc, text = run_wine(env, [str(host), str(job)], run, 1800)
    (out / "host.log").write_text(text)
    m = re.search(r"^tagloop: .*$", text, re.M)
    if not m:
        sd = re.search(r"set_device (\d+)", text)
        why = f"runtime set_device returned {sd.group(1)}" if sd and sd.group(1) != "0" else "no tagloop line"
        return "FAIL", (f"{why}; RR could not start (stock vkd3d-proton without dxil-spirv #362? use --vkd3d); "
                        f"see {out / 'host.log'} and {stage / 'wronggpu-rr.log'}")
    return ("PASS" if rc == 0 and m.group(0).endswith(": ok") else "FAIL"), m.group(0)


# ---- ags-layout -----------------------------------------------------------------------------------------------------
E4M3 = np.array([(-1.0 if b & 0x80 else 1.0) * ((b & 7) / 8.0 * 2.0 ** -6 if (b >> 3) & 15 == 0
                 else (1 + (b & 7) / 8.0) * 2.0 ** (((b >> 3) & 15) - 7)) for b in range(256)])
E4M3[[0x7f, 0xff]] = np.nan


def e4m3_rne(x):
    """FP32 -> E4M3FN byte, round to nearest even, no saturation (inputs stay within +-448)."""
    mag = E4M3[:0x7f]
    i = int(np.argmin(np.abs(mag - abs(x))))
    if i + 1 < 0x7f and abs(mag[i + 1] - abs(x)) == abs(mag[i] - abs(x)) and (i & 1):
        i += 1
    return i | (0x80 if np.signbit(x) else 0)


def f16_bits(x):
    return int(np.float32(x).astype(np.float16).view(np.uint16))


def ags_inputs():
    rng = np.random.default_rng(20261112)
    lanes = 32

    def f16_regs():
        h = rng.uniform(-4.0, 4.0, size=(lanes, 8)).astype(np.float16)
        return np.concatenate([h.view(np.uint32)[:, :4], np.zeros((lanes, 4), np.uint32)], axis=1)

    def fp8_regs():
        b = rng.integers(0, 256, size=(lanes, 8), dtype=np.uint8)
        b[(b & 0x7f) == 0x7f] ^= 0x01   # no NaN codes in the WMMA operands
        return np.concatenate([b.view(np.uint32), np.zeros((lanes, 6), np.uint32)], axis=1)

    c = rng.uniform(-400.0, 400.0, size=(lanes, 8)).astype(np.float32)
    ties = np.array([1.0625, -1.0625, 1.1875, 2.125, 0.0, -0.0, 448.0, 1.0 + 2.0 ** -11, 1.0 + 3 * 2.0 ** -11,
                     6.103515625e-05, 3.0517578125e-05, -2.384185791015625e-07], np.float32)
    c.reshape(-1)[:ties.size] = ties
    every = np.zeros((lanes, 8), np.uint32)
    every[:, :2] = np.arange(256, dtype=np.uint8).reshape(lanes, 8).view(np.uint32)
    return np.stack([f16_regs(), f16_regs(), c.view(np.uint32), fp8_regs(), fp8_regs(), every])


def ags_expected(inp):
    """Expected output slots (7 x 32 x 8 dwords; slot 6: 64 lanes x 2 dwords) and the per-slot comparison kind."""
    c = inp[2].view(np.float32)
    exp = np.zeros((7, 32, 8), np.uint32)

    def mma(a, b):   # gfx12 wave32 16x16x16: A lane l = row l%16, k 8(l//16)+e; B col l%16; C/D col l%16, row 8(l//16)+e
        A = np.zeros((16, 16)); B = np.zeros((16, 16)); C = np.zeros((16, 16))
        for l in range(32):
            for e in range(8):
                A[l % 16, 8 * (l // 16) + e] = a[l, e]
                B[8 * (l // 16) + e, l % 16] = b[l, e]
                C[8 * (l // 16) + e, l % 16] = c[l, e]
        D = A @ B + C
        return np.array([[D[8 * (l // 16) + e, l % 16] for e in range(8)] for l in range(32)], np.float32)

    exp[0] = mma(inp[0][:, :4].copy().view(np.float16).astype(np.float64),
                 inp[1][:, :4].copy().view(np.float16).astype(np.float64)).view(np.uint32)
    exp[1] = mma(E4M3[inp[3][:, :2].copy().view(np.uint8)], E4M3[inp[4][:, :2].copy().view(np.uint8)]).view(np.uint32)
    for l in range(32):
        for i in range(4):
            exp[2, l, i] = f16_bits(c[l, 2 * i]) | f16_bits(c[l, 2 * i + 1]) << 16
        q = [e4m3_rne(float(v)) for v in c[l]]
        exp[3, l, 0] = q[0] | q[1] << 8 | q[2] << 16 | q[3] << 24
        exp[3, l, 1] = q[4] | q[5] << 8 | q[6] << 16 | q[7] << 24
        exp[4, l] = [f16_bits(v) for v in c[l]]
        exp[5, l] = np.float32(E4M3[inp[5][l, :2].copy().view(np.uint8)]).view(np.uint32)
    flat = c.reshape(-1)
    w64 = exp[6].reshape(-1)
    for l in range(64):
        h = [f16_bits(flat[(3 * l + i) % 256]) for i in range(3)]
        w64[2 * l], w64[2 * l + 1] = h[0] | h[1] << 16, h[2]
    return exp


AGS_CASES = ("mma f16->f32", "mma fp8->f32", "convert f32 acc->f16", "convert f32 acc->fp8 B",
             "FloatConversion f32->f16", "convert fp8 B->f32 (all 256 codes)", "wave64 cvt3_f16")


def ags_compare(got, exp):
    lines, ok = [], True
    for s, name in enumerate(AGS_CASES):
        g, e = got[s].reshape(-1), exp[s].reshape(-1)
        if s == 6:
            g, e = g[:128], e[:128]
        if s < 2:   # the hardware accumulates in f32 in its own order: compare against the f64 product with a tolerance
            gf, ef = g.view(np.float32).astype(np.float64), e.view(np.float32).astype(np.float64)
            err = float(np.max(np.abs(gf - ef) / np.maximum(1.0, np.abs(ef))))
            good = err < 1e-5
            lines.append(f"  {name:36s} {'ok ' if good else 'BAD'} max rel err {err:.2e}")
        else:
            if s == 5:   # NaN codes: any NaN
                gn, en = np.isnan(g.view(np.float32)), np.isnan(e.view(np.float32))
                bad = int(np.sum((g != e) & ~(gn & en)))
            else:
                bad = int(np.sum(g != e))
            good = bad == 0
            lines.append(f"  {name:36s} {'ok ' if good else 'BAD'} {bad} of {g.size} dwords differ")
        ok &= good
    return ok, lines


def test_ags_layout(a, env, work):
    cxx = mingw_cxx()
    if not cxx:
        return "SKIP", "no MinGW-w64 C++ compiler (MINGW_CXX)"
    dxc = shutil.which(os.environ.get("WG_DXC", "dxc"))
    if not dxc:
        return "SKIP", "no dxc (WG_DXC: the DXC release's Linux dxc)"
    out = work / "ags-layout"
    out.mkdir(parents=True, exist_ok=True)
    for entry in ("main32", "main64"):
        r = subprocess.run([dxc, "-T", "cs_6_6", "-E", entry, "-O3", "-HV", "2021", "-enable-16bit-types",
                            "-I", str(REPO / "clean/dx12/hlsl"), str(HERE / "ags_layout.hlsl"),
                            "-Fo", str(out / f"{entry}.dxil")], capture_output=True, text=True)
        if r.returncode != 0:
            return "FAIL", f"dxc {entry}: {r.stderr.strip()[-500:]}"
    host = out / "ags_layout_host.exe"
    build_exe(cxx, [str(HERE / "ags_layout_host.cpp")], host, libs=["-ld3d12"])
    inp = ags_inputs()
    inp.astype("<u4").tofile(out / "inputs.bin")
    (out / "outputs.bin").unlink(missing_ok=True)
    rc, text = run_wine(env, [str(host), "main32.dxil", "main64.dxil", "inputs.bin", "outputs.bin"], out, 300)
    (out / "host.log").write_text(text)
    if rc != 0 or not (out / "outputs.bin").exists():
        first = next((l for l in text.splitlines() if l.startswith("FAIL")), text.strip()[-300:])
        return "FAIL", (f"{first} (CreateComputePipelineState 0x80070057: vkd3d-proton without dxil-spirv #362? "
                        "use --vkd3d)" if "CreateComputePipelineState" in first else first)
    got = np.fromfile(out / "outputs.bin", "<u4").reshape(7, 32, 8)
    ok, lines = ags_compare(got, ags_expected(inp))
    for l in lines:
        log(l)
    return ("PASS" if ok else "FAIL"), ("all 7 cases match the gfx12 register semantics" if ok else
                                        "results differ from the gfx12 register semantics (see above)")


# ---- launcher -------------------------------------------------------------------------------------------------------
LAUNCHER_TESTS = ("wg-core-tests", "launcher-core-tests", "coexist-core-tests", "wizard-policy-tests")


def test_launcher(a, env, work):
    cxx = mingw_cxx()
    if not cxx:
        return "SKIP", "no MinGW-w64 C++ compiler (MINGW_CXX)"
    windres = Path(cxx).with_name(Path(cxx).name.split("-g++")[0].split("-clang++")[0] + "-windres")
    missing = [t for t in ("cmake", "ninja") if not shutil.which(t)] + ([] if windres.is_file() else [str(windres)])
    if missing:
        return "SKIP", "missing " + ", ".join(missing)
    installer = REPO / "clean" / "installer"
    prep, build = work / "launcher-prep", work / "launcher-build"
    # the installer suite's payload and test map (generated: names and synthetic bytes only)
    r = subprocess.run([sys.executable, str(installer / "tests" / "run_installer_tests.py"), "--work", str(prep), "--prepare-only"],
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError("run_installer_tests.py --prepare-only failed:\n" + (r.stdout + r.stderr)[-3000:])
    r = subprocess.run(["cmake", "-S", str(installer), "-B", str(build), "-G", "Ninja", "-DCMAKE_SYSTEM_NAME=Windows",
                        f"-DCMAKE_CXX_COMPILER={cxx}", f"-DCMAKE_RC_COMPILER={windres}", "-DCMAKE_EXE_LINKER_FLAGS=-static",
                        "-DWRONGGPU_TESTING=ON", f"-DWRONGGPU_PAYLOAD_DIR={prep / 'prepared' / 'payload'}",
                        f"-DWRONGGPU_MAP={prep / 'prepared' / 'test-map.json'}"], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError("configuring clean/installer failed:\n" + (r.stdout + r.stderr)[-3000:])
    r = subprocess.run(["cmake", "--build", str(build), "--target", *LAUNCHER_TESTS], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError("building the launcher's unit tests failed:\n" + (r.stdout + r.stderr)[-3000:])
    passed, failed = [], []
    for t in LAUNCHER_TESTS:
        rc, text = run_wine(env, [str(build / (t + ".exe"))], build, 600)
        # the test's own summary line (Wine's messages are mixed into the output)
        last = [line.strip() for line in text.splitlines() if "checks passed" in line or "FAILED" in line][-1:] or ["(no summary line)"]
        log(f"  {t}: exit {rc}, {last[0]}")
        (passed if rc == 0 and "checks passed" in last[0] else failed).append(f"{t} ({last[0]})")
    if failed:
        return "FAIL", "; ".join(failed)
    return "PASS", "; ".join(passed)


TESTS = {"interop": test_interop, "rr-leak": test_rr_leak, "ags-layout": test_ags_layout, "launcher": test_launcher}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--proton", default=os.environ.get("WG_PROTON"), help="Proton install (env WG_PROTON)")
    ap.add_argument("--vkd3d", help="folder with a self-built vkd3d-proton d3d12.dll + d3d12core.dll")
    ap.add_argument("--runtime", help="wronggpu_runtime.dll with the RR runtime (test rr-leak)")
    ap.add_argument("--tests", default=",".join(TESTS), help="comma list of: " + ", ".join(TESTS))
    ap.add_argument("--work", default=str(REPO / "build" / "linux-tests"), help="work folder (default build/linux-tests)")
    a = ap.parse_args(argv)
    if not a.proton:
        ap.error("--proton (or WG_PROTON) is required")
    names = [t for t in a.tests.split(",") if t]
    unknown = [t for t in names if t not in TESTS]
    if unknown:
        ap.error(f"unknown test(s): {', '.join(unknown)}")
    work = Path(a.work).resolve()
    work.mkdir(parents=True, exist_ok=True)
    env = wine_env.prepare(a.proton, work / "prefix", a.vkd3d, log=log)
    results = {}
    for name in names:
        log(f"== {name}")
        try:
            results[name] = TESTS[name](a, env, work)
        except Exception as e:  # a build or tool failure is this test's failure, not the run's
            results[name] = ("FAIL", f"{type(e).__name__}: {e}")
        log(f"{results[name][0]} {name}: {results[name][1]}")
    log("== summary")
    for name, (status, why) in results.items():
        log(f"{status:4s} {name}: {why}")
    return 1 if any(s == "FAIL" for s, _ in results.values()) else 0


if __name__ == "__main__":
    sys.exit(main())
