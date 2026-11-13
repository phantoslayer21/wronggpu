# SPDX-License-Identifier: MIT
"""Build/run native Device contract tests in a private directory; no HIP code objects or HIP runtime are needed.

Run GPU work under the GPU lock (WG_GPU_LOCK, BUILDING.md). --build-only compiles without using the GPU.
The public AMD clang++ compiler is only a host compiler here, not a HIP runtime dependency.
"""
import argparse
import hashlib
import json
import os
import re
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
DX12 = HERE.parent
CLEAN = DX12.parent
sys.path.insert(0, str(DX12))
import build_dx12 as B  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    ap.add_argument("--rocm", default=r"C:\Program Files\AMD\ROCm\7.2", help="host clang++ compiler only")
    ap.add_argument("--build-only", action="store_true")
    ap.add_argument("--run-only", action="store_true")
    a = ap.parse_args(argv)
    if a.build_only and a.run_only:
        ap.error("choose at most one of --build-only and --run-only")
    out = Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    exe = out / "native_device_test.exe"
    # build_dx12 uses tempfile for shader listings; keep all project artifacts inside the private build.
    tmp = out / "tmp"
    tmp.mkdir(exist_ok=True)
    os.environ["TMP"] = os.environ["TEMP"] = str(tmp)
    import tempfile
    tempfile.tempdir = str(tmp)
    if not a.run_only:
        dst = out / "dx12"
        dst.mkdir(exist_ok=True)
        asm = B.build_asm_tool(str(out))
        entries = []
        for src in (HERE / "device_test.hlsl", DX12 / "kernels" / "native_helpers.hlsl"):
            for entry in B.kernel_decls(str(src)):
                if src.name == "native_helpers.hlsl" and entry["name"] != "native_fill":
                    continue
                file = dst / B.output_name(entry["name"])
                B.compile_hlsl(str(src), entry["entry"], str(file),
                               {"WG_BLOCK": entry["block"], "WG_WAVE": entry["wave"], **entry["defines"]}, asm_tool=asm)
                entries.append({**entry, "file": file.name, "sha256": hashlib.sha256(file.read_bytes()).hexdigest()})
        # A safe synthetic alias exercises exact code-object identity. The runtime test covers real 42/37-DWORD post.
        bound = next(dict(e) for e in entries if e["name"] == "wg_test_add")
        bound.update(name="bound/wg_test_add", co="post_w64__orig.co")
        entries.append(bound)
        (dst / "manifest.json").write_text(json.dumps({"format": B.FORMAT, "kernels": entries}, indent=2))
        rt = CLEAN / "runtime"
        cmd = [str(Path(a.rocm) / "bin" / "clang++.exe"), "-std=c++17", "-O2", "-D_CRT_SECURE_NO_WARNINGS",
               "-fms-runtime-lib=static", str(HERE / "native_device_test.cpp"), str(rt / "device.cpp"),
               str(rt / "device_d3d12.cpp"), str(rt / "embedded.cpp"),
               str(CLEAN / "converter" / "src" / "json.cpp"), str(CLEAN / "converter" / "src" / "sha256.cpp"),
               "-o", str(exe)]
        subprocess.check_call(cmd, cwd=out)
        (out / "build.json").write_text(json.dumps({"command": cmd, "exe_sha256": hashlib.sha256(exe.read_bytes()).hexdigest(),
                                                    "manifest": entries}, indent=2))
    if a.build_only:
        print(str(exe))
        return 0
    env = os.environ.copy()
    for key in ("HIP_PATH", "ROCM_PATH", "LLVM_PATH", "HIP_DEVICE_LIB_PATH", "HIP_CLANG_PATH", "WG_DX12_KERNELS"):
        env.pop(key, None)
    env["WG_D3D12_ARENA_MB"] = "128"
    result = subprocess.run([str(exe), str(out)], env=env, cwd=out, timeout=300,
                            capture_output=True, text=True)
    print(result.stdout, end="")
    print(result.stderr, end="", file=sys.stderr)
    debug_error = re.search(r"native D3D12 debug (?:ERROR|CORRUPTION)", result.stdout + result.stderr)
    return result.returncode or (1 if debug_error else 0)


if __name__ == "__main__":
    sys.exit(main())
