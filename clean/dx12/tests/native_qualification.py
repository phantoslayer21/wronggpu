# SPDX-License-Identifier: MIT
"""Native Model M correctness and fresh-process module gate. Run under the machine GPU lock.

    python clean/dx12/tests/native_qualification.py --build BUILD --models PRIVATE_SYNTHETIC_ROOT --out NEW_REPORT_DIR

First runs compare.py --backend d3d12 (HIP byte oracle). A separate fresh process then executes only native D3D12,
including debug decoding, checks every launch marker and enumerates loaded modules before creation, after every
frame and after destruction. No HIP reference is created in that process. No GPU timing claim is made here.
All frames and weights are produced by the project's original synthetic tools. The unchanged comparison exclusions
are listed by compare.py; FP32 denormal/NaN-payload exceptions are not broadened by this tool.
"""
import argparse
import ctypes
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import traceback

HERE = Path(__file__).resolve().parent
DX12 = HERE.parent
ROOT = DX12.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(DX12))
import numpy as np  # noqa: E402
import compare as C  # noqa: E402
from clean.runner import synthetic_model as SM  # noqa: E402
from clean.runner import output_filter_original_check as OC  # noqa: E402
from clean.runner.pipeline import Pipeline  # noqa: E402

REQUIRED = C.FRAME_KERNELS | C.DEBUG_KERNELS
HELPERS = C.DEBUG_KERNELS
HIP_MODULE = re.compile(r"^(?:amdhip|hiprtc|hipblas|hipfft|hiprand|hipsolver|hipsparse).*\.dll$", re.I)


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for data in iter(lambda: f.read(1024 * 1024), b""):
            h.update(data)
    return h.hexdigest()


def modules():
    """Windows public K32 process-status API; enumerate names only, never inspect DLL contents."""
    if sys.platform != "win32":
        raise RuntimeError("native module qualification requires Windows")
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.GetCurrentProcess.restype = ctypes.c_void_p
    k32.K32EnumProcessModules.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p), ctypes.c_ulong,
                                        ctypes.POINTER(ctypes.c_ulong)]
    k32.K32EnumProcessModules.restype = ctypes.c_int
    k32.K32GetModuleFileNameExW.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_wchar_p, ctypes.c_ulong]
    k32.K32GetModuleFileNameExW.restype = ctypes.c_ulong
    proc = k32.GetCurrentProcess()
    capacity = 256
    while True:
        items = (ctypes.c_void_p * capacity)()
        needed = ctypes.c_ulong()
        if not k32.K32EnumProcessModules(proc, items, ctypes.sizeof(items), ctypes.byref(needed)):
            raise ctypes.WinError(ctypes.get_last_error())
        if needed.value <= ctypes.sizeof(items):
            break
        capacity = needed.value // ctypes.sizeof(ctypes.c_void_p) + 16
    paths = []
    for handle in items[:needed.value // ctypes.sizeof(ctypes.c_void_p)]:
        buf = ctypes.create_unicode_buffer(32768)
        if not k32.K32GetModuleFileNameExW(proc, handle, buf, len(buf)):
            raise ctypes.WinError(ctypes.get_last_error())
        paths.append(buf.value)
    return sorted(paths, key=str.lower)


def trace_counts(trace):
    counts, wrong = {}, []
    for line in trace.read_text().splitlines() if trace.exists() else []:
        fields = line.split()
        if len(fields) != 4 or fields[-1] != "d3d12":
            wrong.append(line)
        else:
            counts[fields[0]] = counts.get(fields[0], 0) + 1
    return counts, wrong


def comparator_sensitivity():
    """CPU-only deliberate one-bit and one-ULP mutations through the actual debug-tree comparator."""
    original = np.array([1.0, -0.0, 2.0], dtype=np.float32)
    checks = {}
    checks["identity"] = sum(C.tree_diff({"x": original}, {"x": original.copy()}).values()) == 0
    for name, index in (("one_ulp", 0), ("signed_zero", 1)):
        changed = original.copy()
        changed.view(np.uint32)[index] ^= np.uint32(1 if name == "one_ulp" else 0x80000000)
        checks[name] = C.tree_diff({"stage": {"fp32": original}}, {"stage": {"fp32": changed}}) == {"stage.fp32": 1}
        checks[name + "_raw"] = int(np.count_nonzero(original.view(np.uint8) != changed.view(np.uint8))) == 1
    checks["shape"] = C.tree_diff({"x": original}, {"x": original[:2]}) == {"x": 3}
    checks["scalar"] = C.tree_diff({"reset": False}, {"reset": True}) == {"reset": 1}
    checks.update({"capacity_" + k: v for k, v in C.capacity_probe_sensitivity().items()})
    class RangeFixture:
        def __init__(self, last):
            self.last = last

        def kernels(self):
            return {"dec0": "decoder__pack12.co", "post_last": self.last}

        def info(self):
            return {"render_size": (2, 1)}

    raw = {"dec0.C": np.zeros(16, np.uint8), "dec0.B": np.zeros(8, np.uint8)}
    packed = C.buffer_ranges(RangeFixture("post_tf_f32_p12_diet_vsit"), raw)
    fallback = C.buffer_ranges(RangeFixture("post_recon"), raw)
    checks["pack12_C4_range"] = packed["dec0.C"]["checked_range"] == [0, 8]
    checks["pack12_C8_tail"] = packed["dec0.C"]["not_written_ranges"] == [[8, 16]]
    checks["pack12_B_unused"] = packed["dec0.B"]["checked_range"] == [0, 0]
    checks["fallback_full_C8"] = fallback["dec0.C"]["checked_range"] == [0, 16]
    checks["fallback_B_checked"] = fallback["dec0.B"]["checked_range"] == [0, 8]
    if not all(checks.values()):
        raise RuntimeError("comparator did not detect deliberate mutation: " + repr(checks))
    return checks


def native_worker(a):
    out, models = Path(a.out), Path(a.models) / "native_qualification"
    trace = out / "native-launches.txt"
    if trace.exists():
        raise RuntimeError("refusing to append native trace")
    os.environ["WG_LAUNCH_TRACE"] = str(trace)
    report = {"pid": os.getpid(), "backend": "d3d12", "hip_reference_created": False, "module_snapshots": [], "cases": []}

    def audit(label):
        loaded = modules()
        forbidden = [p for p in loaded if HIP_MODULE.match(Path(p).name)]
        report["module_snapshots"].append({"label": label, "utc": datetime.now(timezone.utc).isoformat(),
                                           "modules": loaded, "hip_modules": forbidden})
        if forbidden:
            raise RuntimeError("native-only process loaded HIP modules: " + repr(forbidden))

    try:
        audit("before-create")
        cases = C.selected_cases(a.include_4k)
        for title, render, output, flags, resets in cases:
            before_case, _ = trace_counts(trace)
            p = None
            row = {"case": title, "frames": [], "render": render, "output": output}
            report["cases"].append(row)
            try:
                # Empty ROCm path ensures the wrapper does not add an SDK DLL search directory.
                p = Pipeline(a.build, "", str(models / "taps"), render, output, flags, backend="d3d12",
                             model_format="raw", check_packing=False)
                audit(title + ": created")
                for i in range(a.frames):
                    tree = p.frame(OC.frame_input(i, render, reset=i in resets), debug=True)
                    raw = C.raw_buffers(p)
                    ranges = C.buffer_ranges(p, raw)
                    skip = {f"{s}.out" for s in OC.STALE}
                    if tree["state"]["info"]["reset"]:
                        skip.update("state." + n for n in ("latent_prev", "motion_history", "history_prev"))
                    if not OC.display_written(p):
                        skip.add("post.display")
                    row["frames"].append({"index": i, "reset": bool(tree["state"]["info"]["reset"]),
                                           "skipped": sorted(skip), "buffer_ranges": ranges, "buffers": {
                                               k: {"bytes": ranges[k]["checked_range"][1],
                                                   "sha256": hashlib.sha256(v[:ranges[k]["checked_range"][1]]).hexdigest()}
                                               for k, v in raw.items() if k not in skip and ranges[k]["checked_range"][1]}})
                    audit(title + f": frame {i}")
                    print(f"native-only {title}, frame {i}: completed, HIP DLLs absent", flush=True)
                row["kernels"] = p.kernels()
                row["notes"] = p.notes()
            finally:
                OC.close(p)
            audit(title + ": destroyed")
            after_case, _ = trace_counts(trace)
            row["launches"] = {k: n - before_case.get(k, 0) for k, n in after_case.items()
                               if n != before_case.get(k, 0)}
            case_minimum = {k: (1 if k in HELPERS else a.frames) for k in REQUIRED}
            row["missing"] = {k: {"required": n, "actual": row["launches"].get(k, 0)}
                              for k, n in case_minimum.items() if row["launches"].get(k, 0) < n}
            if row["missing"]:
                raise RuntimeError("native per-case dispatch gate failed: " + repr(row["missing"]))

        # A no-taps model selects the 37-DWORD analytic post variant; the native 42-DWORD port must refuse it.
        before, _ = trace_counts(trace)
        p = None
        rejection = ""
        try:
            title, render, output, flags, resets = C.CASES[1]
            p = Pipeline(a.build, "", str(models / "plain"), render, output, flags, backend="d3d12",
                         model_format="raw", check_packing=False)
            p.frame(OC.frame_input(0, render, reset=True), debug=True)
        except RuntimeError as exc:
            rejection = str(exc)
        finally:
            OC.close(p)
        if not rejection or not re.search(r"post_w64|w64/post", rejection, re.I):
            raise RuntimeError("unsupported analytic variant was not explicitly rejected: " + rejection)
        after, _ = trace_counts(trace)
        if after.get("w64/post_tf_f32_p12_diet_vsit", 0) != before.get("w64/post_tf_f32_p12_diet_vsit", 0):
            raise RuntimeError("analytic variant incorrectly dispatched the original-filter post port")
        report["unsupported_variant_rejection"] = rejection
        audit("after-unsupported-variant")
        counts, wrong = trace_counts(trace)
        required_minimum = {k: len(cases) * (1 if k in HELPERS else a.frames) for k in REQUIRED}
        missing = {k: {"required": n, "actual": counts.get(k, 0)} for k, n in required_minimum.items() if counts.get(k, 0) < n}
        report.update(launches=counts, non_native_launches=wrong, missing=missing, trace=str(trace))
        if wrong or missing:
            raise RuntimeError("native dispatch gate failed: " + repr({"wrong": wrong, "missing": missing}))
        report["passed"] = True
    except Exception:
        report.update(passed=False, error=traceback.format_exc())
        print(report["error"], flush=True)
    finally:
        (out / "native-process.json").write_text(json.dumps(report, indent=2))
    return 0 if report.get("passed") else 1


def prepare_models(base):
    root = base / "native_qualification"
    if root.exists():
        raise RuntimeError("use a fresh private synthetic model directory; refusing to reuse " + str(root))
    root.mkdir(parents=True)
    SM.write_raw(str(root / "taps"), SM.synthetic_tensors(OC.SEED, taps=SM.output_taps("random", OC.SEED)))
    SM.write_raw(str(root / "plain"), SM.synthetic_tensors(OC.SEED))
    (root / "synthetic-origin.json").write_text(json.dumps({"generator": "clean/runner/synthetic_model.py",
        "seed": OC.SEED, "taps": "random", "files": {str(p.relative_to(root)): sha(p) for p in root.rglob("*.bin")}}, indent=2))
    return root


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--build", required=True)
    ap.add_argument("--native-build", help="standalone native bundle; --build remains the HIP reference")
    ap.add_argument("--models", required=True)
    ap.add_argument("--out", required=True, help="new evidence directory (never overwritten)")
    ap.add_argument("--rocm", default=r"C:\Program Files\AMD\ROCm\7.2")
    ap.add_argument("--frames", type=int, default=8)
    ap.add_argument("--include-4k", action="store_true", help="also qualify 1920x1080 -> 3840x2160 Performance geometry")
    ap.add_argument("--probe-unused-capacity", action="store_true", help="also poison C capacity during the differential check")
    ap.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--skip-compare", action="store_true", help="diagnostic only: native process gate without differential run")
    a = ap.parse_args(argv)
    native_build = a.native_build or a.build
    if a.frames < 4:
        ap.error("at least four frames are needed to exercise the second reset")
    for key in ("HIP_PATH", "ROCM_PATH", "LLVM_PATH", "HIP_DEVICE_LIB_PATH", "HIP_CLANG_PATH"):
        os.environ.pop(key, None)
    if a.worker:
        return native_worker(a)
    out = Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    models = prepare_models(Path(a.models).resolve())
    report = {"utc": datetime.now(timezone.utc).isoformat(), "build": str(Path(a.build).resolve()), "frames": a.frames,
              "include_4k": a.include_4k, "cases": [c[0] for c in C.selected_cases(a.include_4k)],
              "comparator_mutations": comparator_sensitivity(), "commands": [], "qualification_complete": False}
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    report["source_head"] = head
    (out / "source.diff").write_bytes(subprocess.check_output(["git", "diff", "--binary"], cwd=ROOT))
    report["native_build"] = str(Path(native_build).resolve())
    report["native_code_objects_present"] = [str(p) for p in Path(native_build).rglob("*.co")]
    if a.native_build and report["native_code_objects_present"]:
        raise RuntimeError("standalone native bundle must contain no HIP code objects")
    manifest = Path(native_build) / "dx12" / "manifest.json"
    files = [Path(a.build) / "wronggpu_runtime.dll", Path(native_build) / "wronggpu_runtime.dll", manifest]
    files += [manifest.parent / k["file"] for k in json.loads(manifest.read_text())["kernels"]]
    report["build_sha256"] = {str(p): sha(p) for p in files}
    ok = True
    if not a.skip_compare:
        cmd = [sys.executable, str(DX12 / "compare.py"), "--build", a.build, "--models", str(models),
               "--candidate-build", native_build,
               "--rocm", a.rocm, "--backend", "d3d12", "--kernels", "all", "--frames", str(a.frames),
               "--json", str(out / "compare.json"), "--trace", str(out / "compare-launches.txt")]
        if a.include_4k:
            cmd.append("--include-4k")
        if a.probe_unused_capacity:
            cmd.append("--probe-unused-capacity")
        report["commands"].append(cmd)
        with (out / "compare.log").open("w") as log:
            result = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=ROOT, timeout=1800)
        report["compare_exit"] = result.returncode
        ok = result.returncode == 0
    cmd = [sys.executable, str(Path(__file__).resolve()), "--worker", "--build", native_build, "--models", a.models,
           "--out", str(out), "--frames", str(a.frames)]
    if a.include_4k:
        cmd.append("--include-4k")
    report["commands"].append(cmd)
    with (out / "native-process.log").open("w") as log:
        result = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=ROOT, timeout=1800)
    report["native_process_exit"] = result.returncode
    native_log = (out / "native-process.log").read_text()
    if (out / "compare.log").exists():
        native_log += "\n" + (out / "compare.log").read_text()
    report["debug_errors"] = [line for line in native_log.splitlines()
                              if re.search(r"native D3D12 debug (?:ERROR|CORRUPTION)", line)]
    ok = ok and result.returncode == 0 and not report["debug_errors"]
    report["qualification_complete"] = ok and not a.skip_compare and a.frames >= 8
    report["passed_requested_checks"] = ok
    report["timing_scope"] = "No performance qualification; frame transfers, debug downloads and module scans included."
    (out / "qualification.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({"report": str(out / "qualification.json"), "passed": ok,
                      "full_frame_gate": report["qualification_complete"]}, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
