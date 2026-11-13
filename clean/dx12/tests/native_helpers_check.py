# SPDX-License-Identifier: MIT
"""Original synthetic FP32 helper differential: separate HIP/native processes, byte-fill guards, HIP module audit.

Build only (no GPU): python clean/dx12/tests/native_helpers_check.py --build <HIP build> --out <private out> --build-only
Then use --run-only under the GPU lock (WG_GPU_LOCK, BUILDING.md). Optional --verify-mutation builds/runs a one-bit-corrupting color shader
and requires failure. Compile outside the GPU lock; use the same --verify-mutation option for both invocations.
All shaders go through build_dx12.py's exact compiler/rewriter/validator. No learned model or frame data is used.
The independent HIP reference uses a HIP-only host executable; native and mutation runs retain the full native host.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone

import numpy as np

HERE = Path(__file__).resolve().parent
DX12 = HERE.parent
CLEAN = DX12.parent
ROOT = CLEAN.parent
sys.path.insert(0, str(DX12))
import build_dx12  # noqa: E402

ROCM = os.environ.get("ROCM_PATH", r"C:\Program Files\AMD\ROCm\7.2")
GUARD = 16
POISON = np.uint32(0xa5c31f7b)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build_hosts(out, rocm):
    """Build the two helper hosts without modifying any shader artifact or manifest."""
    out.mkdir(parents=True, exist_ok=True)
    tmp = out / "tmp"
    tmp.mkdir(exist_ok=True)
    env = os.environ.copy()
    env["TMP"] = env["TEMP"] = str(tmp)
    shared = [HERE / "native_helpers_test.cpp", CLEAN / "runtime" / "device.cpp",
              CLEAN / "runtime" / "embedded.cpp"]
    native = [shared[0], shared[1], CLEAN / "runtime" / "device_d3d12.cpp", shared[2],
              CLEAN / "converter" / "src" / "json.cpp", CLEAN / "converter" / "src" / "sha256.cpp"]
    compiler = [str(Path(rocm) / "bin" / "clang++.exe"), "-std=c++17", "-O2", "-D_CRT_SECURE_NO_WARNINGS",
                "-fms-runtime-lib=static"]
    records = []
    for name, sources, flags, libraries in (
            ("native_helpers_test.exe", native, [], ["-ld3d12", "-ldxgi", "-ldxguid"]),
            ("native_helpers_hip_test.exe", shared, ["-DWG_HELPER_HIP_ONLY=1"], [])):
        exe = out / name
        source_hashes = {str(p): sha(p) for p in sources}
        cmd = [*compiler, *flags, *map(str, sources), "-o", str(exe), *libraries]
        subprocess.run(cmd, env=env, check=True)
        records.append({"exe": str(exe), "command": cmd, "source_sha256": source_hashes, "exe_sha256": sha(exe)})
    (out / "helper-host-build.json").write_text(json.dumps(
        {"timestamp_utc": datetime.now(timezone.utc).isoformat(), "hosts": records,
         "builder_sha256": sha(Path(__file__)), "gpu_executed": False}, indent=2))
    return out / "native_helpers_test.exe"


def build(out, rocm, mutation):
    out.mkdir(parents=True, exist_ok=True)
    tmp = out / "tmp"
    tmp.mkdir(exist_ok=True)
    os.environ["TMP"] = str(tmp)
    os.environ["TEMP"] = str(tmp)
    src = DX12 / "kernels" / "native_helpers.hlsl"
    if mutation:
        original = src.read_text()
        before = "wg_st128(out_p, i * 16u, r);"
        after = "uint4 broken = r; broken.x ^= 1u; wg_st128(out_p, i * 16u, broken);"
        if original.count(before) != 1:
            raise RuntimeError("mutation site no longer matches exactly once")
        src = out / "native_helpers_mutant.hlsl"
        src.write_text(original.replace(before, after))
    dst = out / "dx12"
    dst.mkdir(exist_ok=True)
    asm = build_dx12.build_asm_tool(str(out))
    kernels = []
    for decl in build_dx12.kernel_decls(str(src)):
        result = dst / build_dx12.output_name(decl["name"])
        build_dx12.compile_hlsl(str(src), decl["entry"], str(result),
                               {"WG_BLOCK": decl["block"], "WG_WAVE": decl["wave"], **decl["defines"]},
                               asm_tool=asm, keep_listing=str(result.with_suffix(".ll")))
        kernels.append({k: decl[k] for k in ("name", "co", "entry", "block", "wave", "args", "arg_dwords", "source")}
                       | {"file": result.name, "sha256": sha(result)})
    (dst / build_dx12.MANIFEST).write_text(json.dumps({"format": build_dx12.FORMAT, "kernels": kernels}, indent=2))
    if mutation:
        return
    return build_hosts(out, rocm)


FRAC_MAX = (1 << 23) - 1           # all fraction bits of a binary32
NAN_Q = 1 << 22                    # the quiet bit of a NaN
NAN_PAYLOAD_A, NAN_PAYLOAD_B = 74565, 344865   # arbitrary NaN payloads (both sides of the quiet bit tested)


def _f32(sign, exponent, fraction=0):
    """IEEE binary32 bit pattern from its fields."""
    return (sign << 31) | (exponent << 23) | fraction


def synthetic(n, seed):
    rng = np.random.default_rng(seed)
    color = rng.integers(0, 1 << 32, n * 4, dtype=np.uint32)
    rgb = rng.integers(0, 1 << 32, n * 3, dtype=np.uint32)
    special = np.array([_f32(0, 0), _f32(1, 0), _f32(0, 0, 1), _f32(1, 0, 1),           # +-0, +-smallest subnormal
                        _f32(0, 0, FRAC_MAX), _f32(0, 1), _f32(0, 1, 1),                 # largest subnormal, smallest normals
                        _f32(0, 127), _f32(1, 127), _f32(0, 127, 1), _f32(0, 126, FRAC_MAX),  # 1, -1, 1 + ulp, 1 - ulp / 2
                        _f32(0, 126), _f32(0, 127, 1 << 22), _f32(0, 128, 1 << 21),      # 0.5, 1.5, 2.5
                        _f32(0, 150), _f32(0, 150, 1),                                   # 2^23, 2^23 + 1
                        _f32(0, 254, FRAC_MAX), _f32(1, 254, FRAC_MAX),                  # +-largest finite
                        _f32(0, 255), _f32(1, 255),                                      # +-infinity
                        _f32(0, 255, NAN_Q | NAN_PAYLOAD_A), _f32(1, 255, NAN_Q | NAN_PAYLOAD_B),  # quiet NaNs, payloads
                        _f32(0, 255, NAN_PAYLOAD_A), _f32(1, 255, NAN_PAYLOAD_B)], np.uint32)      # signalling NaNs
    for values in (color, rgb):
        values[:min(values.size, special.size * 8)] = np.resize(special, min(values.size, special.size * 8))
        # Half-way FP16 boundaries and their FP32 neighbours remain raw FP32 outputs in this test.
        lo, hi = values.size // 4, values.size // 2
        values[lo:hi] = (rng.integers(0x30000000, 0x48000000, hi - lo, dtype=np.uint32) & np.uint32(0xffffe000))
        values[lo:hi] |= np.uint32(0x1000)
        values[lo:hi] += rng.integers(-1, 2, hi - lo, dtype=np.int32).astype(np.uint32)
    factors = [_f32(0, 127), _f32(0, 126), _f32(0, 128), _f32(0, 128, 1 << 22),        # 1, 0.5, 2, 3
               _f32(0, 123, 5033165), _f32(0, 127, 1), _f32(0, 126, FRAC_MAX),           # 0.1f, 1 + ulp, 1 - ulp / 2
               _f32(1, 127), _f32(0, 1), _f32(0, 254, FRAC_MAX), _f32(0, 0), _f32(1, 0),  # -1, smallest normal, max, +-0
               _f32(0, 255), _f32(1, 255), _f32(0, 255, NAN_Q | NAN_PAYLOAD_A),          # +-infinity, quiet NaN
               _f32(1, 255, NAN_Q | NAN_PAYLOAD_B), _f32(0, 255, NAN_PAYLOAD_A), _f32(0, 0, 1)]  # NaNs, smallest subnormal
    boundary_inputs = []
    # Underflow before rounding: sample both sides of the smallest-normal midpoint and the exact normal boundary.
    # Five exponent splits x six factor significands x both signs. Integer quotient chooses neighbouring input
    # significands without computing test inputs from the implementation's floating-point results.
    for exponent in (1, 2, 31, 63, 126):
        for significand in (1 << 23, (1 << 23) + 1, 11184811, 3 << 22, (1 << 24) - 2, (1 << 24) - 1):  # 11184811: 2^24 * 2 / 3 rounded up
            for sign in (0, 0x80000000):
                factors.append(sign | (exponent << 23) | (significand & 0x7fffff))
            for bound in ((1 << 47) - (1 << 23), 1 << 47):
                center = bound // significand
                for delta in range(-2, 4):
                    other = center + delta
                    if 0x800000 <= other <= 0xffffff:
                        bits = ((127 - exponent) << 23) | (other & 0x7fffff)
                        boundary_inputs.extend((bits, bits | 0x80000000))
    # Division midpoint: cover the exact significand tie across low/middle/high exponent splits, both signs,
    # neighbouring numerators and denominator powers of two. Repeat each input four times so alpha placement cannot
    # omit either sign from the RGB division checks.
    for numerator_exponent in (1, 2, 31, 63, 126, 127):
        denominator = (numerator_exponent + 127) << 23
        for delta in (-1, 0, 1):
            factors.extend((denominator + delta, (denominator + delta) | 0x80000000))
        numerator = (numerator_exponent << 23) | 0x7fffff
        for delta in (-2, -1, 0, 1, 2):
            for sign in (0, 0x80000000):
                boundary_inputs.extend([(numerator + delta) | sign] * 4)
    pi = np.array(list(dict.fromkeys(factors)), np.uint32)
    boundary = np.array(boundary_inputs, np.uint32)
    for values in (color, rgb):
        start = min(special.size * 8, values.size)
        count = min(boundary.size, values.size - start)
        values[start:start + count] = boundary[:count]
    return color, rgb, pi


def denormal(v):
    return ((v & 0x7f800000) == 0) & ((v & 0x7fffff) != 0)


def nan(v):
    return ((v & 0x7f800000) == 0x7f800000) & ((v & 0x7fffff) != 0)


def records(path, n, cases):
    data = np.fromfile(path, dtype=np.uint32)
    width = n * 7 + GUARD * 2
    if data.size != 2 + cases * width or data[0] != n or data[1] != cases:
        raise RuntimeError(f"bad output length/header: {path}")
    return data[2:].reshape(cases, width)


def compare(reference, candidate, color, rgb, pi):
    n = color.size // 4
    rows = []
    unexpected = 0
    for case, factor in enumerate(pi):
        for kernel, start, count, inputs in (("color_divide", 0, n * 4, color),
                                             ("scale_rgb", n * 4 + GUARD, n * 3, rgb)):
            a = reference[case, start:start + count]
            b = candidate[case, start:start + count]
            active = np.ones(count, dtype=bool)
            if kernel == "color_divide":
                active[3::4] = False
            excluded = active & (denormal(inputs) | denormal(factor) | denormal(a))
            two_nan = active & nan(inputs) & nan(factor) & nan(a) & nan(b)
            different = a != b
            bad = different & ~excluded & ~two_nan
            # Alpha is a pure bit-copy. Check it against the input independently of the differential.
            alpha_bad = int(np.count_nonzero(b[3::4] != color[3::4])) if kernel == "color_divide" else 0
            guard_a = reference[case, start + count:start + count + GUARD]
            guard_b = candidate[case, start + count:start + count + GUARD]
            guard_bad = int(np.count_nonzero(guard_a != POISON) + np.count_nonzero(guard_b != POISON))
            boundary_count = boundary_hip_bad = boundary_native_bad = 0
            if kernel == "scale_rgb":
                exponent = (inputs >> 23) & 255
                factor_exponent = (int(factor) >> 23) & 255
                product = ((inputs & 0x7fffff) | 0x800000).astype(np.uint64) * ((int(factor) & 0x7fffff) | 0x800000)
                # Independent RNE of the exact product onto the subnormal lattice (one quantum is 2^-149).
                quotient, remainder = product >> 24, product & 0xffffff
                rounded = quotient + ((remainder > 0x800000) | ((remainder == 0x800000) & ((quotient & 1) != 0)))
                boundary = ((exponent != 0) & (factor_exponent != 0) & (exponent + factor_exponent == 127) &
                            (product < (1 << 47)) & (rounded == 0x800000))
                expected = ((inputs ^ factor) & 0x80000000) | 0x00800000
                boundary_count = int(boundary.sum())
                boundary_hip_bad = int(np.count_nonzero(boundary & (a != expected)))
                boundary_native_bad = int(np.count_nonzero(boundary & (b != expected)))
            else:
                exponent = ((inputs >> 23) & 255).astype(np.int32)
                factor_exponent = (int(factor) >> 23) & 255
                delta = exponent - factor_exponent
                numerator = ((inputs & 0x7fffff) | 0x800000).astype(np.uint64)
                numerator <<= np.where(delta == -126, 23, 22).astype(np.uint64)
                denominator = (int(factor) & 0x7fffff) | 0x800000
                quotient, remainder = numerator // denominator, numerator % denominator
                rounded = quotient + ((2 * remainder > denominator) |
                                       ((2 * remainder == denominator) & ((quotient & 1) != 0)))
                boundary = (active & (exponent > 0) & (exponent < 255) & (0 < factor_exponent < 255) &
                            ((delta == -126) | (delta == -127)) & (quotient < 0x800000) & (rounded == 0x800000))
                expected = ((inputs ^ factor) & 0x80000000) | 0x00800000
                boundary_count = int(boundary.sum())
                boundary_hip_bad = int(np.count_nonzero(boundary & (a != expected)))
                boundary_native_bad = int(np.count_nonzero(boundary & (b != expected)))
            first = np.flatnonzero(bad)[:8]
            row = {"kernel": kernel, "pi_bits": f"{int(factor):08x}", "values": count,
                   "raw_differences": int(np.count_nonzero(different)), "denormal_domain_values": int(excluded.sum()),
                   "denormal_differences": int(np.count_nonzero(different & excluded)),
                   "two_nan_payload_differences": int(np.count_nonzero(different & two_nan & ~excluded)),
                   "unexpected": int(bad.sum()), "alpha_errors": alpha_bad, "guard_errors": guard_bad,
                   "underflow_rounds_to_normal": boundary_count, "boundary_reference_errors": boundary_hip_bad,
                   "boundary_candidate_errors": boundary_native_bad,
                   "first": [{"index": int(i), "input": f"{int(inputs[i]):08x}",
                              "hip": f"{int(a[i]):08x}", "native": f"{int(b[i]):08x}"} for i in first]}
            unexpected += row["unexpected"] + alpha_bad + guard_bad + boundary_hip_bad + boundary_native_bad
            rows.append(row)
    return {"unexpected": unexpected, "passed": unexpected == 0, "cases": rows}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--build", required=True, type=Path, help="verified HIP reference build containing frame.co")
    ap.add_argument("--out", required=True, type=Path, help="private owned output directory")
    ap.add_argument("--rocm", default=ROCM)
    ap.add_argument("--n", default=65539, type=int)
    ap.add_argument("--seed", default=281, type=int)
    phase = ap.add_mutually_exclusive_group()
    phase.add_argument("--build-only", action="store_true", help="compile only, no GPU execution")
    phase.add_argument("--run-only", action="store_true", help="run existing compiled artifacts; no builds inside GPU lock")
    ap.add_argument("--verify-mutation", action="store_true", help="also run an intentionally corrupted HLSL candidate")
    args = ap.parse_args()
    if not 1 <= args.n < (1 << 24):
        ap.error("--n must be in [1, 2^24)")
    args.out = args.out.resolve()
    exe = args.out / "native_helpers_test.exe"
    hip_exe = args.out / "native_helpers_hip_test.exe"
    if args.run_only:
        required = [exe, hip_exe, args.out / "dx12" / build_dx12.MANIFEST]
        if args.verify_mutation:
            required.append(args.out / "mutation" / "dx12" / build_dx12.MANIFEST)
        for path in required:
            if not path.is_file():
                ap.error(f"missing prebuilt artifact {path}; run --build-only first")
    else:
        exe = build(args.out, args.rocm, False)
        if args.verify_mutation:
            build(args.out / "mutation", args.rocm, True)
    if args.build_only:
        print(f"compiled {exe} and {hip_exe}; GPU execution not requested")
        return 0
    color, rgb, pi = synthetic(args.n, args.seed)
    input_path = args.out / "inputs.bin"
    with input_path.open("wb") as stream:
        np.array([args.n, pi.size], np.uint32).tofile(stream)
        color.tofile(stream)
        rgb.tofile(stream)
        pi.tofile(stream)
    env = os.environ.copy()
    for key in ("HIP_PATH", "ROCM_PATH", "LLVM_PATH", "HIP_DEVICE_LIB_PATH", "HIP_CLANG_PATH", "WG_DX12_KERNELS"):
        env.pop(key, None)
    runs = [("hip", args.out), ("native", args.out)]
    if args.verify_mutation:
        runs.append(("mutation", args.out / "mutation"))
    commands = []
    for tag, dxbuild in runs:
        cmd = [str(hip_exe if tag == "hip" else exe), "hip" if tag == "hip" else "native", args.rocm,
               str(args.build.resolve()), str(dxbuild),
               str(input_path), str(args.out / f"{tag}.bin")]
        commands.append(cmd)
        env["WG_LAUNCH_TRACE"] = str(args.out / f"{tag}.trace.txt")
        with (args.out / f"{tag}.log").open("w") as log:
            subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=300, check=True)
    reference = records(args.out / "hip.bin", args.n, pi.size)
    result = compare(reference, records(args.out / "native.bin", args.n, pi.size), color, rgb, pi)
    result.update({"timestamp_utc": datetime.now(timezone.utc).isoformat(), "seed": args.seed, "pixels": args.n,
                   "factor_cases": int(pi.size), "commands": commands,
                   "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                   "domain": "Synthetic raw FP32, random bits, rounding boundaries, signed zero, infinities and NaNs; "
                             "targeted underflow-to-normal rounding boundaries across exponent splits and signs; "
                             "alpha always checked bitwise. FP32 denormal operands/results and two-NaN propagation "
                             "choice excluded only from arithmetic equality, with counts retained."})
    if args.verify_mutation:
        mutation = compare(reference, records(args.out / "mutation.bin", args.n, pi.size), color, rgb, pi)
        result["mutation"] = mutation
        result["mutation_detected"] = mutation["unexpected"] > 0
        result["passed"] = result["passed"] and result["mutation_detected"]
    files = [input_path, exe, hip_exe, args.build / "frame.co", DX12 / "kernels" / "native_helpers.hlsl",
             HERE / "native_helpers_test.cpp", Path(__file__), *(args.out / "dx12").glob("*.dxil")]
    result["sha256"] = {str(p.resolve()): sha(p) for p in files}
    (args.out / "report.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({"passed": result["passed"], "unexpected": result["unexpected"],
                      "mutation_detected": result.get("mutation_detected"), "report": str(args.out / "report.json")}))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
