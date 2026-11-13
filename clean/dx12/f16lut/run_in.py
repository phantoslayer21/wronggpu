# SPDX-License-Identifier: MIT
"""Exhaustive proof of the direct encoder's colour table (clean/dx12/hlsl/wg_f16lut_in.hlsli).

    python clean/dx12/f16lut/run_in.py --work build/lev-f16lut/proof-in

1. The index (host_in rt): for all 65536 FP16 patterns h, f16tof32(h) is the exact value and, for non-NaN h,
   f32tof16(f16tof32(h)) == h; every one of the 2^24 - 2 FP32 NaN patterns converts to an FP16 NaN. So a texel value c of
   an FP16 colour texture (an FP16 value, or a NaN) indexes an entry generated from c itself (or from a NaN, whose
   entry is (0, 0) as the chain's).
2. The entries: for every parameter set of a grid (e0 = expo[0]: 0, FP16 exposures incl. extremes and random ones, and
   FP32 values a supplied exposure can have: negative, tiny, denormal, huge, +-inf, NaN; pi: 1, 1.375 and 15 other
   values incl. 0, negative, denormal, FLT_MIN / FLT_MAX, inf, NaN) and every one of the 65536 FP16 patterns c:
     gen  = f16lut_enc_entry (what f16lut_build_direct writes)   vs  orig = the encoder's chain (verify_in.hlsl, copied)
     orig vs the HIP oracle (oracle_in.hip: color_divide + encoder.hip's U.e and pixel_values channel loop)
Pass: 1 holds; gen == orig everywhere; orig == HIP except where an FP32 value of HIP is denormal (D3D12 flushes FP32
denormals: the existing D3D12 kernels' documented exception, unchanged by the table); the negative control (the
generator's v of c = 0x3c00 changed by one bit) gives exactly one difference per set. Writes <work>/proof.json.

"""
import argparse
import json
import os
import re
import struct
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DX12 = os.path.dirname(HERE)
CLEAN = os.path.dirname(DX12)
sys.path.insert(0, DX12)
import build_dx12  # noqa: E402

ROCM = os.environ.get("ROCM_PATH", r"C:\Program Files\AMD\ROCm\7.2")
# Optional GPU lock (WG_GPU_LOCK: a script run as `python <lock> --timeout <s> -- <command>`, e.g. to keep
# GPU jobs of several shells apart); unset: the command runs directly.
GPU_LOCK = os.environ.get("WG_GPU_LOCK", "")


def _locked(cmd, timeout="3600"):
    return ([sys.executable, GPU_LOCK, "--timeout", str(timeout), "--"] if GPU_LOCK else []) + list(cmd)


def hip_env():
    env = os.environ.copy()
    for k in ("HIP_PATH", "ROCM_PATH", "LLVM_PATH", "HIP_DEVICE_LIB_PATH", "HIP_CLANG_PATH"):
        env.pop(k, None)
    env["PATH"] = os.path.join(ROCM, "bin") + os.pathsep + env["PATH"]
    env["AMD_COMGR_CACHE"] = "0"
    return env


def run(cmd, **kw):
    r = subprocess.run(cmd, capture_output=True, text=True, **kw)
    if r.returncode not in (0, 5):
        print(r.stdout[-2000:], r.stderr[-2000:])
    return r


def gpu(args, **kw):
    return run(_locked(args), timeout=1800, **kw)


def f32(x):
    return struct.unpack("<I", struct.pack("<f", x))[0]


def param_sets():
    rng = np.random.default_rng(20261016)
    es = [0.0, 1.0, 0.5, 2.0, 0.18 / 0.82, 0.25, 4.0, 65504.0, 6.103515625e-05, 5.960464477539063e-08, 1e-3, 1e3, 0.1]
    es += [float(np.float16(abs(x))) for x in rng.standard_normal(40) * 4.0]
    es += [float(np.float16(x)) for x in rng.uniform(1e-3, 60000.0, 8)]
    e0 = [f32(np.float32(e)) for e in es]
    # supplied exposures (value / texture times the scale): any FP32
    e0 += [f32(np.float32(x)) for x in (0.37, 1.7e-5, 1e-30, 3e38, 123456.7, 0.7071067811865476, -1.0, -0.37)]
    # raw FP32 patterns: -0, the smallest / largest denormal, FLT_MIN, FLT_MAX, +inf, -inf, a quiet NaN
    e0 += [f32(np.float32(-0.0)), 1, (1 << 23) - 1, 1 << 23, f32(np.finfo(np.float32).max)]
    e0 += [f32(np.float32(np.inf)), f32(np.float32(-np.inf)), 0x7fc << 20]
    pis = [1.0, 1.375, 0.5, 2.0, 0.1, 10.0, 1e-3, 1e3, 0.7071067690849304, 3.0000001192092896]
    # FLT_MIN, FLT_MAX, the smallest denormal, +0, -1, +inf, a quiet NaN
    pis = [f32(p) for p in pis] + [1 << 23, f32(np.finfo(np.float32).max), 1, 0, f32(np.float32(-1.0))]
    pis += [f32(np.float32(np.inf)), 0x7fc << 20]
    return [(e, p) for e in e0 for p in pis]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    ap.add_argument("--batch", type=int, default=150)
    a = ap.parse_args()
    w = os.path.abspath(a.work)
    os.makedirs(w, exist_ok=True)
    co = os.path.join(w, "oracle_in.co")
    r = run([os.path.join(ROCM, "bin", "clang.exe"), "-x", "hip", "--offload-arch=gfx1201", "--offload-device-only",
             "-nogpuinc", "-O3", "--no-gpu-bundle-output", f"--rocm-path={ROCM}",
             f"--rocm-device-lib-path={os.path.join(ROCM, 'amdgcn', 'bitcode')}", "-I", os.path.join(CLEAN, "kernels"),
             os.path.join(HERE, "oracle_in.hip"), "-o", co], env=hip_env())
    assert r.returncode == 0, "oracle build"
    exe = os.path.join(w, "host_in.exe")
    r = run(f'"{build_dx12.VSDEVCMD}" -arch=x64 -host_arch=x64 >nul 2>nul && cl /nologo /O2 /EHsc /std:c++17 '
            f'/D__HIP_PLATFORM_AMD__ /I"{os.path.join(ROCM, "include")}" "{os.path.join(HERE, "host_in.cpp")}" '
            f'/Fe"{exe}" /Fo"{os.path.join(w, "host_in.obj")}" /link d3d12.lib dxgi.lib '
            f'/LIBPATH:"{os.path.join(ROCM, "lib")}" amdhip64.lib', shell=True)
    assert r.returncode == 0 and os.path.exists(exe), "host build"
    asm = build_dx12.build_asm_tool(w)
    dx = {}
    for tag, extra in (("verify", {}), ("negative", {"F16LUT_NEGATIVE": 1})):
        dx[tag] = os.path.join(w, f"{tag}.dxil")
        build_dx12.compile_hlsl(os.path.join(HERE, "verify_in.hlsl"), "f16lut_in_verify", dx[tag],
                                {"WG_BLOCK": 256, "WG_WAVE": 32, **extra}, asm_tool=asm)
    rr = gpu([exe, "rt", dx["verify"]])
    mr = re.search(r"RT values .*", rr.stdout)
    assert mr, rr.stdout[-3000:] + rr.stderr[-2000:]
    print(mr.group(0), flush=True)
    sets = param_sets()
    proof = {"sets": len(sets), "values_per_set": 65536, "batches": [], "index": mr.group(0), "index_pass": rr.returncode == 0}
    tot = {"gen_ne_orig": 0, "orig_ne_hip": 0, "orig_ne_hip_denormal": 0, "values": 0}
    for b0 in range(0, len(sets), a.batch):
        part = sets[b0:b0 + a.batch]
        pp = os.path.join(w, "params.bin")
        with open(pp, "wb") as f:
            f.write(struct.pack("<I", len(part)))
            for s in part:
                f.write(struct.pack("<II", *s))
        hb = os.path.join(w, "hip.bin")
        r = gpu([exe, "hip", co, pp, hb], env=hip_env())
        assert r.returncode == 0, r.stdout + r.stderr
        r = gpu([exe, "dx12", dx["verify"], pp, hb])
        m = re.search(r"TOTAL sets \d+ values (\d+): gen!=orig (\d+); orig!=hip (\d+), of them FP32-denormal intermediates (\d+)",
                      r.stdout)
        assert m, r.stdout[-3000:] + r.stderr[-2000:]
        for k, v in zip(("values", "gen_ne_orig", "orig_ne_hip", "orig_ne_hip_denormal"), m.groups()):
            tot[k] += int(v)
        proof["batches"].append({"first": b0, "sets": len(part), "summary": m.group(0),
                                 "detail": [l for l in r.stdout.splitlines() if l.startswith("set") or l.startswith("  set")][:40]})
        if b0 == 0:   # negative control on the first batch
            rn = gpu([exe, "dx12", dx["negative"], pp, hb])
            mn = re.search(r"TOTAL sets \d+ values \d+: gen!=orig (\d+)", rn.stdout)
            proof["negative_control_gen_ne_orig"] = int(mn.group(1)) if mn else rn.stdout[-500:]
        os.remove(hb)
        print(m.group(0), flush=True)
    proof["total"] = tot
    proof["pass"] = (proof["index_pass"] and tot["gen_ne_orig"] == 0 and tot["orig_ne_hip"] == tot["orig_ne_hip_denormal"]
                     and isinstance(proof.get("negative_control_gen_ne_orig"), int)
                     and proof["negative_control_gen_ne_orig"] == min(a.batch, len(sets)))
    with open(os.path.join(w, "proof.json"), "w") as f:
        json.dump(proof, f, indent=1)
    print(json.dumps({k: proof[k] for k in ("index", "sets", "total", "negative_control_gen_ne_orig", "pass")}, indent=1))
    return 0 if proof["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
