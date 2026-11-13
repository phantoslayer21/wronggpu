# SPDX-License-Identifier: MIT
"""Build/run the original synthetic fused-decoder differential. Use --build-only outside the GPU lock;
then --run-only under the GPU lock (WG_GPU_LOCK, BUILDING.md). Normal, odd and 4K Performance render geometry, both resets, four seeds.
This checks the actual stored pack12 A/C/latent bytes; B is intentionally unwritten. No performance claim.
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
CLEAN = HERE.parent.parent
ROOT = CLEAN.parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--build", required=True)
    ap.add_argument("--native-build", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--rocm", default=r"C:\Program Files\AMD\ROCm\7.2")
    ap.add_argument("--seeds", type=int, default=4)
    group = ap.add_mutually_exclusive_group()
    group.add_argument("--build-only", action="store_true")
    group.add_argument("--run-only", action="store_true")
    a = ap.parse_args(argv)
    if a.seeds < 1:
        ap.error("--seeds must be positive")
    out = Path(a.out).resolve()
    exe = out / "tuned_decoder_diff.exe"
    if not a.run_only:
        out.mkdir(parents=True, exist_ok=False)
        tmp = out / "tmp"
        tmp.mkdir()
        os.environ["TMP"] = os.environ["TEMP"] = str(tmp)
        sources = [HERE / "tuned_decoder_diff.cpp", CLEAN / "runtime" / "device.cpp",
                   CLEAN / "runtime" / "device_d3d12.cpp", CLEAN / "runtime" / "embedded.cpp",
                   CLEAN / "converter" / "src" / "json.cpp", CLEAN / "converter" / "src" / "sha256.cpp"]
        cmd = [str(Path(a.rocm) / "bin" / "clang++.exe"), "-std=c++17", "-O2", "-D_CRT_SECURE_NO_WARNINGS",
               "-fms-runtime-lib=static", *map(str, sources), "-o", str(exe), "-ld3d12", "-ldxgi", "-ldxguid"]
        subprocess.run(cmd, cwd=out, check=True)
        (out / "build.json").write_text(json.dumps({"command": cmd, "exe_sha256": sha(exe),
            "source_sha256": {str(p): sha(p) for p in sources},
            "source_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()}, indent=2))
    if a.build_only:
        return 0
    if (out / "results.json").exists() or (out / "launches.txt").exists():
        raise RuntimeError("refusing to overwrite previous evidence")
    built = json.loads((out / "build.json").read_text())
    if sha(exe) != built["exe_sha256"]:
        raise RuntimeError("test executable differs from build record")
    env = os.environ.copy()
    for key in ("HIP_PATH", "ROCM_PATH", "LLVM_PATH", "HIP_DEVICE_LIB_PATH", "HIP_CLANG_PATH", "WG_DX12_KERNELS"):
        env.pop(key, None)
    env["WG_LAUNCH_TRACE"] = str(out / "launches.txt")
    env["WG_D3D12_ARENA_MB"] = "512"
    cmd = [str(exe), a.rocm, a.native_build, a.build, str(a.seeds)]
    result = subprocess.run(cmd, cwd=out, env=env, capture_output=True, text=True, timeout=900)
    text = result.stdout + result.stderr
    (out / "run.log").write_text(text)
    print(text, end="")
    trace = (out / "launches.txt").read_text() if (out / "launches.txt").exists() else ""
    launches = sum(line.startswith("d1p12/swin_block_c64_add_dec0_nc ") and line.endswith(" d3d12")
                   for line in trace.splitlines())
    rows = re.findall(r"^decoder (\d+)x(\d+) seed (\d+) reset (\d+): (\d+) differing of (\d+) bytes", text, re.M)
    passed = result.returncode == 0 and len(rows) == 6 * a.seeds and launches == 6 * a.seeds
    passed = passed and not any(int(r[4]) for r in rows)
    errors = re.findall(r"^.*native D3D12 debug (?:ERROR|CORRUPTION).*$", text, re.M)
    passed = passed and not errors
    manifest = Path(a.native_build) / "dx12" / "manifest.json"
    payloads = [manifest] + [manifest.parent / k["file"] for k in json.loads(manifest.read_text())["kernels"]]
    report = {"utc": datetime.now(timezone.utc).isoformat(), "command": cmd, "exit": result.returncode,
              "passed": passed, "build": built, "cases": rows, "native_dispatches": launches, "debug_errors": errors,
              "hip_code_object_sha256": sha(Path(a.build) / "dec1_dec0__pack12.co"),
              "native_payload_sha256": {str(p): sha(p) for p in payloads},
              "exceptions": "No differing stored bytes excluded. B is not written on pack12 and is not compared.",
              "timing_scope": "Correctness only; no timing qualification."}
    (out / "results.json").write_text(json.dumps(report, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
