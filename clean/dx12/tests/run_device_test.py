# SPDX-License-Identifier: MIT
"""Builds and runs the hybrid D3D12 device self-test (device_test.cpp with the kernels of device_test.hlsl).

    python clean/dx12/tests/run_device_test.py --out <work dir>
"""
import argparse
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DX12 = os.path.dirname(HERE)
CLEAN = os.path.dirname(DX12)
sys.path.insert(0, DX12)
import build_dx12  # noqa: E402

ROCM = os.environ.get("ROCM_PATH", r"C:\Program Files\AMD\ROCm\7.2")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--rocm", default=ROCM)
    ap.add_argument("--build-only", action="store_true", help="build without reserving or using the GPU")
    a = ap.parse_args()
    dst = os.path.join(a.out, "dx12")
    os.makedirs(dst, exist_ok=True)
    asm = build_dx12.build_asm_tool(a.out)
    src = os.path.join(HERE, "device_test.hlsl")
    kernels = []
    for d in build_dx12.kernel_decls(src):
        out = os.path.join(dst, build_dx12.output_name(d["name"]))
        build_dx12.compile_hlsl(src, d["entry"], out, {"WG_BLOCK": d["block"], "WG_WAVE": d["wave"]}, asm_tool=asm)
        kernels.append({"name": d["name"], "file": os.path.basename(out), "co": d["co"], "entry": d["entry"], "block": d["block"],
                        "wave": d["wave"], "args": d["args"], "arg_dwords": d["arg_dwords"], "source": d["source"]})
    with open(os.path.join(dst, build_dx12.MANIFEST), "w") as f:
        json.dump({"format": build_dx12.FORMAT, "kernels": kernels}, f, indent=1)
    rt = os.path.join(CLEAN, "runtime")
    exe = os.path.join(a.out, "device_test.exe")
    cmd = [os.path.join(a.rocm, "bin", "clang++.exe"), "-std=c++17", "-O2", "-D_CRT_SECURE_NO_WARNINGS",
           "-fms-runtime-lib=static", os.path.join(HERE, "device_test.cpp"), os.path.join(rt, "device.cpp"),
           os.path.join(rt, "device_dx12.cpp"), os.path.join(rt, "embedded.cpp"),
           os.path.join(CLEAN, "converter", "src", "json.cpp"), "-o", exe]
    subprocess.check_call(cmd)
    if a.build_only:
        return 0
    env = os.environ.copy()
    for k in ("HIP_PATH", "ROCM_PATH", "LLVM_PATH", "HIP_DEVICE_LIB_PATH", "HIP_CLANG_PATH", "WG_DX12_KERNELS"):
        env.pop(k, None)
    return subprocess.call([exe, a.rocm, a.out], env=env, timeout=300)


if __name__ == "__main__":
    sys.exit(main())
