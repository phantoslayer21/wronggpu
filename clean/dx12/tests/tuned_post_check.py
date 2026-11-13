# SPDX-License-Identifier: MIT
"""Original synthetic post FP32 intermediate differential, using private instrumented shader copies.

Build with --build-only, then --run-only under the GPU lock (WG_GPU_LOCK, BUILDING.md). Both builds remain private. Fifteen scalar
intermediates are written one at a time into the otherwise-unused output buffer; history and reduced history
continue normally. The final output-filter stores are disabled only in these diagnostic derivatives. pi=1 prevents
the runtime's later output scale from changing the dump. This does not replace whole-frame production-shader gates.
The instrumentation follows the original clean work/hipdbg experiment, applied to current clean source.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
DX12, CLEAN, ROOT = HERE.parent, HERE.parent.parent, HERE.parent.parent.parent
sys.path[:0] = [str(ROOT), str(DX12)]
import build_dx12 as B  # noqa: E402
import compare as C  # noqa: E402
from clean.runner import build as HB, output_filter_original_check as OC, synthetic_model as SM  # noqa: E402
from clean.runner.pipeline import Pipeline  # noqa: E402

KERNEL = "w64/post_tf_f32_p12_diet_vsit"
FIELDS = ["hh.r", "hh.g", "hh.b", "H.b", "weight1", "weight2", "q1.x", "q1.y", "q2.x", "q2.y",
          "sample1.b", "sample2.b", "gather.W", "motion_center.x", "motion_center.y"]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def once(text, old, new):
    if text.count(old) != 1:
        raise RuntimeError("instrumentation anchor no longer unique: " + old[:120])
    return text.replace(old, new)


def instrument_hip(text):
    start = text.index("WG_FN void post_core_s(")
    end = text.index('#include "post_motion_outres.h"', start)
    block = text[start:end]
    block = once(block, "float oh[3]) {", "float oh[3], float* dbg = nullptr) {")
    block = once(block, "  float H[3];", "  if (dbg) dbg[10] = h[2];\n  float H[3];")
    block = once(block, "  post_blend<kTrim>(g, sl, H, p, s, hh, oh);", """  if (dbg) dbg[11] = h[2];
  post_blend<kTrim>(g, sl, H, p, s, hh, oh);
  if (dbg) {
    dbg[0] = hh[0]; dbg[1] = hh[1]; dbg[2] = hh[2]; dbg[3] = H[2]; dbg[4] = sl.w1; dbg[5] = sl.w2;
    dbg[6] = q1x; dbg[7] = q1y; dbg[8] = q2x; dbg[9] = q2y; dbg[12] = g.W; dbg[13] = mx[1]; dbg[14] = my[1];
  }""")
    text = text[:start] + block + text[end:]
    text = once(text, "        float hh[3], oh3[3];\n        if (MoT::on)",
                "        float hh[3], oh3[3], dbgv[15] = {};\n        if (MoT::on)")
    text = once(text, "                                              oh3);\n        if (own_x && y >= own_r0 && y < own_r1) {",
                "                                              oh3, dbgv);\n        if (own_x && y >= own_r0 && y < own_r1) {\n"
                "          ((float*)out)[y * p.dw + x] = dbgv[WG_DBG_K];")
    text = once(text, "        const int y = y0 + d;\n        if (WG_TF_AB_V == 1)",
                "        const int y = y0 + d;\n        if (WG_DBG_K >= 0) continue;\n        if (WG_TF_AB_V == 1)")
    return text


def instrument_hlsl(text, mutant=False):
    start = text.index("void post_core_s(")
    end = text.index("// ---- output filter", start)
    block = text[start:end]
    block = once(block, "out float oh[3])", "out float oh[3], out float dbg[15])")
    block = once(block, "    [unroll] for (int i = 0; i < 3; ++i) H[i] = w1 * h[i];",
                "    dbg[10] = h[2];\n    [unroll] for (int i = 0; i < 3; ++i) H[i] = w1 * h[i];")
    block = once(block, "    // post_blend<kTrim = true>", "    dbg[11] = h[2];\n    // post_blend<kTrim = true>")
    tail = block.rfind("}")
    block = block[:tail] + """    dbg[0] = hh[0]; dbg[1] = hh[1]; dbg[2] = hh[2]; dbg[3] = H[2]; dbg[4] = w1; dbg[5] = w2;
    dbg[6] = q1x; dbg[7] = q1y; dbg[8] = q2x; dbg[9] = q2y; dbg[12] = g.W; dbg[13] = mx[1]; dbg[14] = my[1];
""" + block[tail:]
    text = text[:start] + block + text[end:]
    text = once(text, "                float hh[3], oh3[3];", "                float hh[3], oh3[3], dbgv[15];")
    text = once(text, "post_core_s(gk, x, y, fx, fy, mx, my, s, hh, oh3);",
                "post_core_s(gk, x, y, fx, fy, mx, my, s, hh, oh3, dbgv);\n"
                "                if (own_x && y >= own_r0 && y < own_r1)\n"
                "                    wg_st32(out_p, ((uint)y * (uint)P_dw + (uint)x) * 4u, "
                + ("asuint(dbgv[WG_DBG_K]) ^ 1u);" if mutant else "asuint(dbgv[WG_DBG_K]));"))
    text = once(text, "            wg_mem.Store3(out_p + o, uint3(asuint(vo[0] * dec_scale), asuint(vo[1] * dec_scale), asuint(vo[2] * dec_scale)));",
                "            // Diagnostic scalar dump owns out_p; no final filter write.")
    return text


def copy_runtime(source, dest, hip):
    dest.mkdir()
    for name in ("wronggpu_runtime.dll", "variants.json", "preset-M.v1.map.json"):
        shutil.copyfile(source / name, dest / name)
    if hip:
        for p in source.glob("*.co"):
            shutil.copyfile(p, dest / p.name)
    else:
        (dest / "dx12").mkdir()
        for p in (source / "dx12").glob("*"):
            if p.is_file():
                shutil.copyfile(p, dest / "dx12" / p.name)


def build(a, out, fields):
    temp = out / "tmp"
    temp.mkdir()
    os.environ["TMP"] = os.environ["TEMP"] = str(temp)
    import tempfile
    tempfile.tempdir = str(temp)
    asm = B.build_asm_tool(str(out))
    hip_source = CLEAN / "kernels" / "post.hip"
    hlsl_source = DX12 / "kernels" / "post.hlsl"
    hip_text = instrument_hip(hip_source.read_text())
    hlsl_text = hlsl_source.read_text()
    decl = next(k for k in B.kernel_decls(str(hlsl_source)) if k["name"] == KERNEL)
    report = {"utc": datetime.now(timezone.utc).isoformat(), "fields": fields, "commands": [],
              "basis_sha256": {str(p): sha(p) for p in (hip_source, hlsl_source)}, "artifacts": {}}
    for field in fields:
        target = out / f"field-{field}"
        target.mkdir()
        hip, native = target / "hip", target / "native"
        copy_runtime(Path(a.build), hip, True)
        copy_runtime(Path(a.native_build), native, False)
        src = target / "post_instrumented.hip"
        src.write_text(hip_text)
        cmd = HB.device_command(str(Path(a.rocm) / "bin" / "clang.exe"), a.rocm, str(src),
                                str(hip / "post_w64__orig.co"), defines={"WG_LZ_PROFILE": 2, "WG_DBG_K": field},
                                extra=["-mwavefrontsize64", "-I", str(CLEAN / "kernels")])
        subprocess.run(cmd, cwd=out, check=True, stdout=subprocess.DEVNULL)
        report["commands"].append(cmd)
        # Some builds retain a second platform-named code object; keep the private pair coherent.
        alias = hip / "gfx1201.post_w64__orig.co"
        if alias.exists():
            shutil.copyfile(hip / "post_w64__orig.co", alias)
        for broken in (False, True) if field == fields[0] else (False,):
            candidate = native
            if broken:
                candidate = target / "mutant"
                copy_runtime(native, candidate, False)
            src = target / ("post_mutant.hlsl" if broken else "post_instrumented.hlsl")
            src.write_text(instrument_hlsl(hlsl_text, broken))
            dxil = candidate / "dx12" / B.output_name(KERNEL)
            B.compile_hlsl(str(src), decl["entry"], str(dxil),
                           {"WG_BLOCK": decl["block"], "WG_WAVE": decl["wave"], "WG_DBG_K": field, **decl["defines"]},
                           includes=[str(DX12 / "kernels")], asm_tool=asm, keep_listing=str(dxil.with_suffix(".ll")))
            manifest_path = candidate / "dx12" / "manifest.json"
            manifest = json.loads(manifest_path.read_text())
            for k in manifest["kernels"]:
                if k["name"] == KERNEL:
                    k.update(sha256=sha(dxil), diagnostic=f"post FP32 field {field}, mutant={broken}")
            manifest_path.write_text(json.dumps(manifest, indent=2))
        print(f"built post intermediate {field}: {FIELDS[field]}", flush=True)
    report["artifacts"] = {str(p.relative_to(out)): sha(p) for p in out.rglob("*")
                           if p.is_file() and p.suffix in (".dll", ".co", ".dxil", ".hlsl", ".hip", ".json")}
    (out / "build.json").write_text(json.dumps(report, indent=2))


def worker(a):
    destination = Path(a.worker_output)
    p = Pipeline(a.build, a.rocm if a.backend == "hip" else "", a.models, (853, 481), (1279, 721),
                 {"depth_inverted": False, "auto_exposure": True}, backend=a.backend, model_format="raw", check_packing=False)
    try:
        records = {}
        for i in range(a.frames):
            p.frame(OC.frame_input(i, (853, 481), pi=1.0, reset=i in (0, 3)))
            raw = C.raw_buffers(p)
            count = p.dw * p.dh
            if raw["filter.out"].size < count * 4:
                raise RuntimeError("diagnostic output arena cannot hold one scalar per reconstructed pixel")
            records[f"fp32_{i}"] = raw["filter.out"].view(np.uint32)[:count].copy()
            records[f"history_{i}"] = raw["post.history"]
            records[f"reduced_{i}"] = raw["post.reduced"]
        np.savez(destination, **records)
        print(json.dumps({"dw": p.dw, "dh": p.dh, "frames": a.frames, "kernels": p.kernels()}), flush=True)
    finally:
        OC.close(p)
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--build", required=True)
    ap.add_argument("--native-build")
    ap.add_argument("--out", required=True)
    ap.add_argument("--models")
    ap.add_argument("--rocm", default=r"C:\Program Files\AMD\ROCm\7.2")
    ap.add_argument("--fields", default=",".join(map(str, range(15))))
    ap.add_argument("--frames", type=int, default=4)
    group = ap.add_mutually_exclusive_group()
    group.add_argument("--build-only", action="store_true")
    group.add_argument("--run-only", action="store_true")
    ap.add_argument("--worker-output", help=argparse.SUPPRESS)
    ap.add_argument("--backend", choices=("hip", "d3d12"), default="d3d12", help=argparse.SUPPRESS)
    a = ap.parse_args(argv)
    for key in ("HIP_PATH", "ROCM_PATH", "LLVM_PATH", "HIP_DEVICE_LIB_PATH", "HIP_CLANG_PATH"):
        os.environ.pop(key, None)
    if a.worker_output:
        return worker(a)
    fields = list(dict.fromkeys(int(x) for x in a.fields.split(",")))
    if not fields or any(k < 0 or k >= len(FIELDS) for k in fields) or a.frames < 4:
        ap.error("select valid field indices and at least four frames")
    out = Path(a.out).resolve()
    if not a.run_only:
        if not a.native_build:
            ap.error("--native-build is required for building")
        out.mkdir(parents=True, exist_ok=False)
        build(a, out, fields)
    if a.build_only:
        return 0
    build_record = json.loads((out / "build.json").read_text())
    if fields != build_record["fields"]:
        ap.error("--fields must match the private build")
    if (out / "results.json").exists():
        raise RuntimeError("refusing to overwrite results")
    model = out / "synthetic-model"
    if model.exists():
        raise RuntimeError("synthetic model directory already exists")
    SM.write_raw(str(model), SM.synthetic_tensors(OC.SEED, taps=SM.output_taps("random", OC.SEED)))
    report = {"utc": datetime.now(timezone.utc).isoformat(), "runs": [], "build": build_record,
              "synthetic_model_sha256": {str(p.relative_to(model)): sha(p) for p in model.rglob("*.bin")},
              "exceptions": "No differences excluded; compares raw FP32 and history bytes",
              "scope": "Instrumented post FP32 intermediates, odd shape, resets frames 0 and 3; no timing claim"}
    for field in fields:
        target = out / f"field-{field}"
        for label in ("hip", "native", "mutant") if field == fields[0] else ("hip", "native"):
            trace = target / (label + ".trace.txt")
            if trace.exists() or (target / (label + ".npz")).exists():
                raise RuntimeError("refusing to overwrite partial evidence; use a fresh private build")
            env = os.environ.copy()
            env["WG_LAUNCH_TRACE"] = str(trace)
            cmd = [sys.executable, str(Path(__file__).resolve()), "--build", str(target / label), "--out", str(out),
                   "--models", str(model), "--rocm", a.rocm, "--frames", str(a.frames),
                   "--backend", "hip" if label == "hip" else "d3d12", "--worker-output", str(target / (label + ".npz"))]
            result = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True, timeout=900)
            (target / (label + ".log")).write_text(result.stdout + result.stderr)
            if result.returncode:
                raise RuntimeError(f"field {field} {label} process failed; see its log")
            if label == "hip":
                continue
            reference, candidate = np.load(target / "hip.npz"), np.load(target / (label + ".npz"))
            differences = {k: int(np.count_nonzero(reference[k] != candidate[k])) for k in reference.files}
            first = {}
            for k, count in differences.items():
                if count:
                    indices = np.flatnonzero(reference[k] != candidate[k])[:8]
                    first[k] = [{"index": int(i), "hip": int(reference[k][i]), "native": int(candidate[k][i])} for i in indices]
            stage_counts = {}
            wrong = []
            for line in trace.read_text().splitlines():
                parts = line.split()
                if len(parts) != 4 or parts[-1] != "d3d12":
                    wrong.append(line)
                else:
                    stage_counts[parts[0]] = stage_counts.get(parts[0], 0) + 1
            gates = all(stage_counts.get(k) == a.frames for k in C.FRAME_KERNELS) and not wrong
            gates = gates and not re.search(r"native D3D12 debug (?:ERROR|CORRUPTION)", result.stdout + result.stderr)
            if label == "mutant":
                gates = gates and any(n > 0 for k, n in differences.items() if k.startswith("fp32_"))
                gates = gates and not any(n for k, n in differences.items() if not k.startswith("fp32_"))
            else:
                gates = gates and not any(differences.values())
            report["runs"].append({"field": field, "name": FIELDS[field], "candidate": label, "passed": bool(gates),
                                   "differences": differences, "first_differences": first, "command": cmd,
                                   "native_launches": stage_counts, "wrong_launches": wrong})
            print(f"field {field} {FIELDS[field]} {label}: {'PASS' if gates else 'FAIL'}", flush=True)
    report["passed"] = all(r["passed"] for r in report["runs"])
    (out / "results.json").write_text(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
