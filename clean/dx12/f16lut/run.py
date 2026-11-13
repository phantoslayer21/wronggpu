# SPDX-License-Identifier: MIT
"""Exhaustive proof of the direct post kernel's output table (clean/dx12/hlsl/wg_f16lut.hlsli).

    python clean/dx12/f16lut/run.py --work build/lev-f16lut/proof

For every parameter set of a grid (dec_scale = (float)(0.01 / (double)e) for 72 exposures e incl. FP16 extremes and
negative values, 12 raw extremes; pi: 1, 1.375 and 12 other finite values incl. 0, negative, denormal, FLT_MIN / FLT_MAX;
rne16 0 and 1) and every one of the 65536 FP16 bit patterns r:
  gen  = f16lut_out (the generator the encoder runs)    vs  orig = the post kernel's chain (verify.hlsl, copied)
  orig vs the HIP oracle (oracle.hip: post.hip's vertical Dec, the f32 row, scale_rgb, framebuffer_half)
Pass: gen == orig everywhere; orig == HIP except where an FP32 intermediate of HIP is denormal (D3D12 flushes FP32
denormals, wg_math.hlsli: the existing D3D12 kernels' documented exception, unchanged by the table). Also a negative
control (the generator's value of one input, r = 0x3c00, changed by one bit) must give exactly one difference per set. Writes <work>/proof.json.
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
    es = [1.0, 0.5, 2.0, 0.18 / 0.82, 0.25, 4.0, 65504.0, 6.103515625e-05, 5.960464477539063e-08, 1e-3, 1e3, 0.1]
    es += [float(np.float16(abs(x))) for x in rng.standard_normal(48) * 4.0]
    es += [float(np.float16(x)) for x in rng.uniform(1e-3, 60000.0, 8)]
    es += [-1.0, -0.37, -65504.0, -6.103515625e-05]
    decs = sorted({f32(np.float32(0.01 / e)) for e in es if e != 0.0})
    # raw extremes: the smallest / largest denormal, FLT_MIN, small / large normals, FLT_MAX, +-0, -denormal, -1
    decs += [1, 0x00800000, 0x00ffffff]
    decs += [0x0d000000, 0x1e3ce508, 0x7f7fffff]
    decs += [0x7f000000, 0x5d5e0b6b, 0x80000000]
    decs += [0x0, 0x807fffff, 0xbf800000]
    pis = [1.0, 1.375, 0.5, 2.0, 0.1, 10.0, 1e-3, 1e3, 0.7071067690849304, 3.0000001192092896]
    pis = [f32(p) for p in pis] + [0x00800000, 0x7f7fffff, 0x00000001, 0x0, 0xbf800000]   # incl. a denormal pi
    return [(d, p, r) for d in decs for p in pis for r in (0, 1)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    ap.add_argument("--batch", type=int, default=150)
    a = ap.parse_args()
    w = os.path.abspath(a.work)
    os.makedirs(w, exist_ok=True)
    co = os.path.join(w, "oracle.co")
    r = run([os.path.join(ROCM, "bin", "clang.exe"), "-x", "hip", "--offload-arch=gfx1201", "--offload-device-only",
             "-nogpuinc", "-O3", "--no-gpu-bundle-output", f"--rocm-path={ROCM}",
             f"--rocm-device-lib-path={os.path.join(ROCM, 'amdgcn', 'bitcode')}", "-I", os.path.join(CLEAN, "kernels"),
             os.path.join(HERE, "oracle.hip"), "-o", co], env=hip_env())
    assert r.returncode == 0, "oracle build"
    exe = os.path.join(w, "host.exe")
    r = run(f'"{build_dx12.VSDEVCMD}" -arch=x64 -host_arch=x64 >nul 2>nul && cl /nologo /O2 /EHsc /std:c++17 '
            f'/D__HIP_PLATFORM_AMD__ /I"{os.path.join(ROCM, "include")}" "{os.path.join(HERE, "host.cpp")}" '
            f'/Fe"{exe}" /Fo"{os.path.join(w, "host.obj")}" /link d3d12.lib dxgi.lib '
            f'/LIBPATH:"{os.path.join(ROCM, "lib")}" amdhip64.lib', shell=True)
    assert r.returncode == 0 and os.path.exists(exe), "host build"
    asm = build_dx12.build_asm_tool(w)
    dx = {}
    for tag, extra in (("verify", {}), ("negative", {"F16LUT_NEGATIVE": 1})):
        dx[tag] = os.path.join(w, f"{tag}.dxil")
        build_dx12.compile_hlsl(os.path.join(HERE, "verify.hlsl"), "f16lut_verify", dx[tag],
                                {"WG_BLOCK": 256, "WG_WAVE": 64, **extra}, asm_tool=asm)
    sets = param_sets()
    proof = {"sets": len(sets), "values_per_set": 65536, "batches": []}
    tot = {"gen_ne_orig": 0, "orig_ne_hip": 0, "orig_ne_hip_denormal": 0, "values": 0}
    for b0 in range(0, len(sets), a.batch):
        part = sets[b0:b0 + a.batch]
        pp = os.path.join(w, "params.bin")
        with open(pp, "wb") as f:
            f.write(struct.pack("<I", len(part)))
            for s in part:
                f.write(struct.pack("<III", *s))
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
    proof["pass"] = (tot["gen_ne_orig"] == 0 and tot["orig_ne_hip"] == tot["orig_ne_hip_denormal"]
                     and isinstance(proof.get("negative_control_gen_ne_orig"), int)
                     and proof["negative_control_gen_ne_orig"] == min(a.batch, len(sets)))
    with open(os.path.join(w, "proof.json"), "w") as f:
        json.dump(proof, f, indent=1)
    print(json.dumps({k: proof[k] for k in ("sets", "total", "negative_control_gen_ne_orig", "pass")}, indent=1))
    return 0 if proof["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
