# SPDX-License-Identifier: MIT
"""Exact q8(GELU) lookup v2 (2 KB groupshared table): generator and exhaustive D3D12 proof in one run.

    python clean/dx12/gelu_lut/run2.py --work <dir> --rle <rle.bin> [--host <host.exe>] [--write-header] [--hip-check <oracle.co>]

1. Generator: gen2.py on HIP's run-length form (rle.bin from run.py: gelu_lut/oracle.hip, every FP32 bit pattern) ->
   gelu_lds.json, gelu_lds_table.bin (and with --write-header hlsl/wg_gelu_lds.hlsli).
2. Proof (D3D12): verify2.hlsl compiled by build_dx12's exact pipeline with the header, every FP32 bit pattern against
   rle.bin on the GPU (host.cpp "dx12" mode, 2^26-input chunks): the table is filled into groupshared memory as in the
   kernels. Also the transcendental path of the base ports (harness check: 0 differences outside denormals) and two
   one-input mutations of rle.bin (sensitivity: each must give exactly one difference).
   With --hip-check, HIP recomputes all 2^32 bytes against rle.bin first (host.cpp "check").
Writes <work>/proof2.json. Every GPU step runs under the GPU lock (WG_GPU_LOCK, BUILDING.md).
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DX12 = os.path.dirname(HERE)
sys.path.insert(0, DX12)
sys.path.insert(0, HERE)
import build_dx12  # noqa: E402
from run import run, gpu, parse_res, hip_env  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    ap.add_argument("--rle", required=True)
    ap.add_argument("--host", help="host.exe built from host.cpp (run.py builds it)")
    ap.add_argument("--write-header", action="store_true")
    ap.add_argument("--hip-check", help="oracle.co: recompute HIP's bytes against rle.bin first")
    a = ap.parse_args()
    w = os.path.abspath(a.work)
    os.makedirs(w, exist_ok=True)
    proof = {}
    exe = a.host or os.path.join(w, "host.exe")
    if a.hip_check:
        r = gpu([exe, "check", a.hip_check, a.rle], env=hip_env())
        proof["hip_oracle_vs_rle"] = parse_res(r.stdout)
        assert r.returncode == 0 and proof["hip_oracle_vs_rle"]["differences"] == 0
    d = np.fromfile(a.rle, np.uint32)
    proof["runs"] = int(d[0])
    gen_out = os.path.join(w, "gen2")
    cmd = [sys.executable, "-W", "ignore", os.path.join(HERE, "gen2.py"), "--rle", a.rle, "--out", gen_out]
    if a.write_header:
        cmd += ["--hlsl", os.path.join(DX12, "hlsl", "wg_gelu_lds.hlsli")]
    r = run(cmd)
    assert r.returncode == 0
    proof["generator"] = json.load(open(os.path.join(gen_out, "gelu_lds.json")))
    asm = build_dx12.build_asm_tool(w)
    src = os.path.join(HERE, "verify2.hlsl")
    for tag, extra in (("lds", {}), ("pair", {"GELU_VERIFY_PAIR": 1}), ("orig", {"GELU_VERIFY_ORIG": 1})):
        build_dx12.compile_hlsl(src, "gelu_lds_verify", os.path.join(w, f"verify2_{tag}.dxil"),
                                {"WG_BLOCK": 256, "WG_WAVE": 32, **extra}, asm_tool=asm)
    r = gpu([exe, "dx12", os.path.join(w, "verify2_lds.dxil"), a.rle])
    proof["d3d12_lds_vs_hip"] = parse_res(r.stdout)
    r = gpu([exe, "dx12", os.path.join(w, "verify2_pair.dxil"), a.rle])
    proof["d3d12_lds2_vs_hip"] = parse_res(r.stdout)
    r = gpu([exe, "dx12", os.path.join(w, "verify2_orig.dxil"), a.rle])
    proof["d3d12_transcendental_vs_hip"] = parse_res(r.stdout)
    n = int(d[0])
    m1 = d.copy(); m1[1 + 150] += 1
    m3 = d.copy(); m3[1 + 20] -= 1   # a change on the positive side, one input earlier
    m2 = d.copy(); i = int(np.nonzero(d[1:1 + n] == 0xbf6c7530)[0][0]); m2[1 + n + i] = d[1 + n + i - 1]
    sens = {}
    for tag, arr in (("moved_change", m1), ("erased_flip", m2), ("moved_positive", m3)):
        p = os.path.join(w, f"rle_{tag}.bin")
        arr.tofile(p)
        r = gpu([exe, "dx12", os.path.join(w, "verify2_lds.dxil"), p])
        sens[tag] = parse_res(r.stdout)
        os.remove(p)
    proof["sensitivity"] = sens
    ok = all(proof[k].get("differences") == proof[k].get("denormal_inputs") for k in ("d3d12_lds_vs_hip", "d3d12_lds2_vs_hip"))
    ok = ok and all(v.get("differences") == 1 for v in sens.values())
    proof["pass"] = bool(ok)
    with open(os.path.join(w, "proof2.json"), "w") as f:
        json.dump(proof, f, indent=1)
    print(json.dumps({k: proof[k] for k in ("runs", "d3d12_lds_vs_hip", "d3d12_lds2_vs_hip", "d3d12_transcendental_vs_hip", "sensitivity",
                                            "pass")}, indent=1))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
