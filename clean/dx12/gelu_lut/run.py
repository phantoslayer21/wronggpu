# SPDX-License-Identifier: MIT
"""Exact q8(GELU) lookup: oracle, generator and exhaustive proof in one run.

    python clean/dx12/gelu_lut/run.py --work build/gelu-lut [--runtime-co <dir with swin.co ...>]
        [--write-headers] [--skip-oracle]

1. Oracle (HIP): oracle.hip with the runtime's device flags (clean/runner/build.py device_command); its GELU chain is
   compared with the runtime code objects' (isa_chain.py), then every FP32 bit pattern 0 .. 2^32 - 1 is converted on the
   GPU and stored as the run-length form rle.bin (u where the byte changes); "check" recomputes all 2^32 bytes and
   compares them with rle.bin (must be 0).
2. Generator: gen.py -> gelu_lut_table.bin (and, with --write-headers, hlsl/wg_gelu_lut.hlsli, runtime/gelu_lut_table.h).
   Compact design (current headers): gen_compact.py + mk_headers.py write the headers and the table; pass that table
   with --table (the generator step is then skipped) and shader defines with --define NAME=VALUE.
3. Proof (D3D12): verify.hlsl compiled by build_dx12's exact pipeline with the header, every FP32 bit pattern against
   rle.bin on the GPU in 2^26-input chunks (counts and the first 30 examples only). Also the base ports' transcendental
   path (harness check) and two one-input mutations of rle.bin (sensitivity: each must give exactly one difference).
Writes <work>/proof.json. Every GPU step runs under the GPU lock (WG_GPU_LOCK, BUILDING.md).
"""
import argparse
import json
import os
import re
import shutil
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
VSDEVCMD = build_dx12.VSDEVCMD


def hip_env():
    env = os.environ.copy()
    for k in ("HIP_PATH", "ROCM_PATH", "LLVM_PATH", "HIP_DEVICE_LIB_PATH", "HIP_CLANG_PATH"):
        env.pop(k, None)
    env["PATH"] = os.path.join(ROCM, "bin") + os.pathsep + env["PATH"]
    return env


def run(cmd, **kw):
    print(">", " ".join(cmd) if isinstance(cmd, list) else cmd, flush=True)
    r = subprocess.run(cmd, capture_output=True, text=True, **kw)
    print(r.stdout.strip())
    if r.stderr.strip():
        print(r.stderr.strip())
    return r


def gpu(args, **kw):
    return run(_locked(args), **kw)


def build_tools(w):
    co = os.path.join(w, "oracle.co")
    r = run([os.path.join(ROCM, "bin", "clang.exe"), "-x", "hip", "--offload-arch=gfx1201", "--offload-device-only", "-nogpuinc",
             "-O3", "--no-gpu-bundle-output", f"--rocm-path={ROCM}",
             f"--rocm-device-lib-path={os.path.join(ROCM, 'amdgcn', 'bitcode')}", "-I", os.path.join(CLEAN, "kernels"),
             os.path.join(HERE, "oracle.hip"), "-o", co], env=hip_env())
    assert r.returncode == 0, "oracle build"
    exe = os.path.join(w, "host.exe")
    r = run(f'"{VSDEVCMD}" -arch=x64 -host_arch=x64 >nul 2>nul && cl /nologo /O2 /EHsc /std:c++17 /D__HIP_PLATFORM_AMD__ '
            f'/I"{os.path.join(ROCM, "include")}" "{os.path.join(HERE, "host.cpp")}" /Fe"{exe}" /Fo"{os.path.join(w, "host.obj")}" '
            f'/link d3d12.lib dxgi.lib /LIBPATH:"{os.path.join(ROCM, "lib")}" amdhip64.lib', shell=True)
    assert r.returncode == 0 and os.path.exists(exe), "host build"
    return co, exe


def isa_chains(path, kernels):
    od = os.path.join(ROCM, "bin", "llvm-objdump.exe")
    text = subprocess.run([od, "-d", path], capture_output=True, text=True).stdout
    sys.path.insert(0, HERE)
    import isa_chain
    return {k: dict(isa_chain.chains(isa_chain.parse(isa_chain.kernel_lines(text, [k])))) for k in kernels}


def parse_res(text):
    m = re.search(r"differences (\d+) \(denormal inputs (\d+), NaN inputs (\d+)\)", text)
    ex = re.findall(r"u=([0-9a-f]+) got=([0-9a-f]+) expected=([0-9a-f]+)", text)
    return {"differences": int(m.group(1)), "denormal_inputs": int(m.group(2)), "nan_inputs": int(m.group(3)),
            "examples": [{"u": "0x" + u, "got": "0x" + g, "hip": "0x" + e} for u, g, e in ex]} if m else {"error": text[-400:]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    ap.add_argument("--runtime-co", default=os.environ.get("WG_RUNTIME_CO", ""))
    ap.add_argument("--write-headers", action="store_true")
    ap.add_argument("--skip-oracle", action="store_true", help="reuse <work>/rle.bin")
    ap.add_argument("--table", help="compact design: this table (mk_headers.py output) instead of running gen.py")
    ap.add_argument("--define", action="append", default=[], help="NAME=VALUE for the verify shader (WG_GELU_LUT_ALT=1)")
    a = ap.parse_args()
    w = os.path.abspath(a.work)
    os.makedirs(w, exist_ok=True)
    proof = {}
    co, exe = build_tools(w)
    # oracle ISA against the runtime's code objects
    ref = {}
    for f, ks in (("swin.co", ["swin_block_c64_nc", "swin_block_c96_nc", "swin_block_c96_add_ex_nc"]),
                  ("dec1_dec0__pack12.co", ["swin_block_c64_add_dec0_nc"]), ("swin__bmegahg.co", ["swin_bneck6_nc"])):
        ref.update(isa_chains(os.path.join(a.runtime_co, f), ks))
    mine = isa_chains(co, ["gelu_rle", "gelu_check"])
    kinds_rt = {s for v in ref.values() for s in v}
    kinds_or = {s for v in mine.values() for s in v}
    proof["isa_chains"] = {"runtime": ref, "oracle": mine, "same_kinds": kinds_rt == kinds_or and len(kinds_rt) == 1}
    assert proof["isa_chains"]["same_kinds"], proof["isa_chains"]
    rle = os.path.join(w, "rle.bin")
    if not a.skip_oracle:
        r = gpu([exe, "oracle", co, rle], env=hip_env())
        assert r.returncode == 0
    r = gpu([exe, "check", co, rle], env=hip_env())
    proof["hip_oracle_vs_rle"] = parse_res(r.stdout)
    assert r.returncode == 0 and proof["hip_oracle_vs_rle"]["differences"] == 0
    d = np.fromfile(rle, np.uint32)
    proof["runs"] = int(d[0])
    # generator
    if a.table:
        table = a.table
        proof["generator"] = {"table": table, "bytes": os.path.getsize(table), "defines": a.define}
    else:
        gen_out = os.path.join(w, "gen")
        cmd = [sys.executable, os.path.join(HERE, "gen.py"), "--rle", rle, "--out", gen_out]
        if a.write_headers:
            cmd += ["--hlsl", os.path.join(DX12, "hlsl", "wg_gelu_lut.hlsli"), "--cpp", os.path.join(CLEAN, "runtime", "gelu_lut_table.h")]
        r = run(cmd)
        assert r.returncode == 0
        proof["generator"] = json.load(open(os.path.join(gen_out, "gelu_lut.json")))
        table = os.path.join(gen_out, "gelu_lut_table.bin")
    # D3D12 proof
    asm = build_dx12.build_asm_tool(w)
    src = os.path.join(HERE, "verify.hlsl")
    defs = dict(d.split("=", 1) for d in a.define)
    for tag, extra in (("lut", defs), ("orig", {"GELU_VERIFY_ORIG": 1})):
        build_dx12.compile_hlsl(src, "gelu_lut_verify", os.path.join(w, f"verify_{tag}.dxil"),
                                {"WG_BLOCK": 256, "WG_WAVE": 32, **extra}, asm_tool=asm)
    r = gpu([exe, "dx12", os.path.join(w, "verify_lut.dxil"), rle, table])
    proof["d3d12_lut_vs_hip"] = parse_res(r.stdout)
    r = gpu([exe, "dx12", os.path.join(w, "verify_orig.dxil"), rle])
    proof["d3d12_transcendental_vs_hip"] = parse_res(r.stdout)
    # sensitivity: one change moved by one input; the J1 flip erased
    n = int(d[0])
    m1 = d.copy(); m1[1 + 150] += 1
    m2 = d.copy(); i = int(np.nonzero(d[1:1 + n] == 0xbf6c7530)[0][0]); m2[1 + n + i] = d[1 + n + i - 1]
    sens = {}
    for tag, arr in (("moved_change", m1), ("erased_flip", m2)):
        p = os.path.join(w, f"rle_{tag}.bin")
        arr.tofile(p)
        r = gpu([exe, "dx12", os.path.join(w, "verify_lut.dxil"), p, table])
        sens[tag] = parse_res(r.stdout)
        os.remove(p)
    proof["sensitivity"] = sens
    ok = (proof["d3d12_lut_vs_hip"].get("differences") == proof["d3d12_lut_vs_hip"].get("denormal_inputs")
          and all(v.get("differences") == 1 for v in sens.values()))
    proof["pass"] = bool(ok)
    with open(os.path.join(w, "proof.json"), "w") as f:
        json.dump(proof, f, indent=1)
    print(json.dumps({k: proof[k] for k in ("runs", "hip_oracle_vs_rle", "d3d12_lut_vs_hip", "d3d12_transcendental_vs_hip", "pass")}, indent=1))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
