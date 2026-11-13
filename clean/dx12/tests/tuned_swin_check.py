# SPDX-License-Identifier: MIT
"""Reproducible synthetic six-Swin-kernel differential and arithmetic mutation checks.

Build privately without GPU access:
  python clean/dx12/tests/tuned_swin_check.py --build HIP_BUILD --native-build NATIVE_BUILD --out NEW_OUT --build-only
Run the same command with --run-only under the GPU lock (WG_GPU_LOCK, BUILDING.md). The default compares 16 seeds in both original
bounded and full-range FP8 regimes, then requires two deliberately changed rounding/order shaders to fail.
This compares FP8 stage outputs, not all FP32 intermediates; mutations show sensitivity but not universal coverage.
It never modifies the candidate manifest, shaders or reference code objects. These runs make no timing claim.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

HERE = Path(__file__).resolve().parent
DX12 = HERE.parent
CLEAN = DX12.parent
ROOT = CLEAN.parent
sys.path.insert(0, str(DX12))
import build_dx12 as B  # noqa: E402

MUTATIONS = [
    ("norm_fma_operand_swapped", "swin_block_c64_nc", "swin_common.hlsli",
     "mad(a.v[0], a.v[0], a.v[1] * a.v[1])", "mad(a.v[1], a.v[1], a.v[0] * a.v[0])"),
    ("softmax_exp_unfused", "swin_block_c64_nc", "swin_block.hlsl",
     "e.v[v4] = mad(tc.v[v4], asfloat(0x3f028297u), -mx);",
     "e.v[v4] = tc.v[v4] * asfloat(0x3f028297u) - mx;"),
]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build(a, out):
    tmp = out / "tmp"
    tmp.mkdir()
    os.environ["TMP"] = os.environ["TEMP"] = str(tmp)
    import tempfile
    tempfile.tempdir = str(tmp)
    sources = [HERE / "tuned_swin_diff.cpp", CLEAN / "runtime" / "device.cpp",
               CLEAN / "runtime" / "device_d3d12.cpp", CLEAN / "runtime" / "embedded.cpp",
               CLEAN / "converter" / "src" / "json.cpp", CLEAN / "converter" / "src" / "sha256.cpp"]
    exe = out / "tuned_swin_diff.exe"
    cmd = [str(Path(a.rocm) / "bin" / "clang++.exe"), "-std=c++17", "-O2", "-D_CRT_SECURE_NO_WARNINGS",
           "-fms-runtime-lib=static", *map(str, sources), "-o", str(exe), "-ld3d12", "-ldxgi", "-ldxguid"]
    subprocess.run(cmd, cwd=out, check=True)
    record = {"command": cmd, "source_sha256": {str(p): sha(p) for p in sources}, "exe_sha256": sha(exe),
              "source_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
              "mutations": []}
    (out / "source.diff").write_bytes(subprocess.check_output(["git", "diff", "--binary"], cwd=ROOT))
    asm = B.build_asm_tool(str(out))
    kernel_dir = DX12 / "kernels"
    fill_source = kernel_dir / "native_helpers.hlsl"
    fill_decl = next(k for k in B.kernel_decls(str(fill_source)) if k["name"] == "native_fill")
    for name, kernel, file, old, new in MUTATIONS:
        target = out / name
        shader_dir = target / "dx12"
        shader_dir.mkdir(parents=True)
        for source in ("swin_block.hlsl", "swin_gemm.hlsl", "swin_common.hlsli"):
            value = (kernel_dir / source).read_text()
            if source == file:
                if value.count(old) != 1:
                    raise RuntimeError(f"mutation {name} no longer matches exactly once")
                value = value.replace(old, new)
            (target / source).write_text(value)
        decl = next(k for k in B.kernel_decls(str(target / "swin_block.hlsl")) if k["name"] == kernel)
        result = shader_dir / B.output_name(kernel)
        B.compile_hlsl(str(target / "swin_block.hlsl"), decl["entry"], str(result),
                       {"WG_BLOCK": decl["block"], "WG_WAVE": decl["wave"], **decl["defines"]},
                       includes=[str(target)], asm_tool=asm, keep_listing=str(result.with_suffix(".ll")))
        entry = {**decl, "file": result.name, "sha256": sha(result)}
        fill_result = shader_dir / B.output_name(fill_decl["name"])
        B.compile_hlsl(str(fill_source), fill_decl["entry"], str(fill_result),
                       {"WG_BLOCK": fill_decl["block"], "WG_WAVE": fill_decl["wave"], **fill_decl["defines"]},
                       asm_tool=asm)
        fill = {**fill_decl, "file": fill_result.name, "sha256": sha(fill_result)}
        manifest = {"format": B.FORMAT, "kernels": [entry, fill]}
        (shader_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
        record["mutations"].append({"name": name, "kernel": kernel, "source": file, "before": old, "after": new,
                                     "manifest": manifest})
    (out / "build.json").write_text(json.dumps(record, indent=2))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--build", required=True, help="HIP reference code objects")
    ap.add_argument("--native-build", required=True)
    ap.add_argument("--out", required=True, help="fresh private work directory, reused only with --run-only")
    ap.add_argument("--rocm", default=r"C:\Program Files\AMD\ROCm\7.2")
    ap.add_argument("--seeds", type=int, default=16)
    group = ap.add_mutually_exclusive_group()
    group.add_argument("--build-only", action="store_true")
    group.add_argument("--run-only", action="store_true")
    a = ap.parse_args(argv)
    if a.seeds < 1:
        ap.error("--seeds must be positive")
    out = Path(a.out).resolve()
    if not a.run_only:
        out.mkdir(parents=True, exist_ok=False)
        build(a, out)
    if a.build_only:
        return 0
    if (out / "results.json").exists():
        raise RuntimeError("refusing to overwrite previous results")
    record = json.loads((out / "build.json").read_text())
    exe = out / "tuned_swin_diff.exe"
    if sha(exe) != record["exe_sha256"]:
        raise RuntimeError("test executable differs from build record")
    env = os.environ.copy()
    for key in ("HIP_PATH", "ROCM_PATH", "LLVM_PATH", "HIP_DEVICE_LIB_PATH", "HIP_CLANG_PATH", "WG_DX12_KERNELS"):
        env.pop(key, None)
    env["WG_D3D12_ARENA_MB"] = "512"
    runs = [("bounded", a.native_build, None, False, False), ("fullrange", a.native_build, None, True, False)]
    runs += [(name, str(out / name), kernel, False, True) for name, kernel, *_ in MUTATIONS]
    report = {"utc": datetime.now(timezone.utc).isoformat(), "seeds": a.seeds, "runs": [], "source_build": record,
              "hip_code_object_sha256": sha(Path(a.build) / "swin.co"), "timing_scope": "No timing qualification",
              "exceptions": "None: all output bytes are checked, including NaN FP8 codes."}
    report["candidate_payload_sha256"] = {
        str(p): sha(p) for p in (Path(a.native_build) / "dx12").glob("*") if p.is_file()}
    for name, candidate, kernel, fullrange, mutant in runs:
        if (out / (name + ".trace.txt")).exists() or (out / (name + ".log")).exists():
            raise RuntimeError("refusing to overwrite partial evidence; use a fresh private build")
        cmd = [str(exe), a.rocm, candidate, str(Path(a.build) / "swin.co"), str(a.seeds)]
        if kernel:
            cmd.append(kernel)
        env["SW_DIFF_FULLRANGE"] = "1" if fullrange else "0"
        env["WG_LAUNCH_TRACE"] = str(out / (name + ".trace.txt"))
        result = subprocess.run(cmd, cwd=out, env=env, capture_output=True, text=True, timeout=900)
        text = result.stdout + result.stderr
        (out / (name + ".log")).write_text(text)
        expected = result.returncode == (3 if mutant else 0)
        summary = re.findall(r"^(\S+)\s+(\d+) seeds: (\d+) differing of (\d+) bytes$", text, re.M)
        expected = expected and len(summary) == (1 if mutant else 6)
        if mutant:
            expected = expected and any(int(row[2]) > 0 for row in summary)
        trace = (out / (name + ".trace.txt")).read_text()
        native_counts = {}
        for line in trace.splitlines():
            fields = line.split()
            if len(fields) == 4 and fields[-1] == "d3d12":
                native_counts[fields[0]] = native_counts.get(fields[0], 0) + 1
        expected = expected and all(native_counts.get(row[0], 0) == a.seeds for row in summary)
        debug_errors = re.findall(r"^.*native D3D12 debug (?:ERROR|CORRUPTION).*$", text, re.M)
        expected = expected and not debug_errors
        report["runs"].append({"name": name, "command": cmd, "exit": result.returncode, "passed": expected,
                               "deliberately_broken": mutant, "summary": summary, "native_launches": native_counts,
                               "debug_errors": debug_errors})
        print(f"{name}: {'PASS' if expected else 'FAIL'} exit={result.returncode}", flush=True)
    report["passed"] = all(row["passed"] for row in report["runs"])
    (out / "results.json").write_text(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
