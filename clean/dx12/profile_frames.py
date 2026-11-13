# SPDX-License-Identifier: MIT
"""Diagnostic native operation attribution on original synthetic Model M frames.

Run it with the GPU otherwise idle (see WG_GPU_LOCK in BUILDING.md). Profiled timestamps perturb scheduling; use time_frames.py
with profiling disabled for comparative performance. CPU scopes overlap.
"""
import argparse
import ctypes
import json
import os
from pathlib import Path
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from clean.runner.pipeline import Pipeline
from clean.runner import output_filter_original_check as OC
from clean.runner import synthetic_model as SM


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--build", required=True)
    ap.add_argument("--models", required=True)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--shape", choices=("1440p", "4k-performance"), default="1440p")
    ap.add_argument("--frames", type=int, default=100)
    ap.add_argument("--warm", type=float, default=3)
    a = ap.parse_args()
    if a.frames < 1 or a.warm < 3:
        ap.error("positive frames and warm-up >=3 seconds required")
    a.out.mkdir(parents=True, exist_ok=False)
    capture = a.out / "operations.jsonl"
    os.environ["WG_D3D12_PROFILE"] = str(capture.resolve())
    os.environ.pop("WG_DX12_DEBUG", None)
    model = Path(a.models) / "taps"
    if not (model / "raw").exists():
        SM.write_raw(str(model), SM.synthetic_tensors(OC.SEED, taps=SM.output_taps("random", OC.SEED)))
    render, output = ((1920, 1080), (3840, 2160)) if a.shape == "4k-performance" else ((1706, 960), (2560, 1440))
    p = Pipeline(a.build, "", str(model), render, output, {"depth_inverted": True, "auto_exposure": True},
                 backend="d3d12", model_format="raw", check_packing=False)
    phases = []
    def sequence():
        return max([r["sequence"] for r in map(json.loads, capture.read_text().splitlines()) if r["type"] == "gpu"] or [0])
    def frame():
        p._check(p.lib.wg_rt_frame(p.rt, ctypes.byref(fr)), "frame")
    def sync():
        p._check(p.lib.wg_rt_sync(p.rt), "sync")
    try:
        x = OC.frame_input(1, render, reset=False)
        p.frame(x)
        fr = p._upload(x)
        start = time.perf_counter()
        while time.perf_counter() - start < a.warm:
            for _ in range(20): frame()
            sync()
        for name, count in (("preuploaded_frames", a.frames), ("upload_frame_readback", 10)):
            begin_sequence, begin_ns = sequence(), time.perf_counter_ns()
            for _ in range(count):
                if name == "preuploaded_frames": frame()
                else: p.frame(x)
            sync()
            end_ns, end_sequence = time.perf_counter_ns(), sequence()
            phases.append({"name": name, "frames": count, "begin_sequence": begin_sequence,
                           "end_sequence": end_sequence, "begin_ns": begin_ns, "end_ns": end_ns,
                           "wall_ms_per_frame": (end_ns-begin_ns)/1e6/count})
        times = []
        for _ in range(30):
            ms = ctypes.c_float()
            p._check(p.lib.wg_rt_frame_timed(p.rt, ctypes.byref(fr), ctypes.byref(ms)), "timed frame")
            times.append(ms.value)
        kernels = p.kernels()
    finally:
        OC.close(p)
    records = [json.loads(s) for s in capture.read_text().splitlines()]
    for phase in phases:
        gpu = [r for r in records if r["type"] == "gpu" and phase["begin_sequence"] < r["sequence"] <= phase["end_sequence"]]
        totals = {}
        for row in gpu:
            key = row["kind"] + "/" + row["name"]
            v = totals.setdefault(key, {"count": 0, "ms_per_frame": 0, "bytes_per_frame": 0})
            v["count"] += 1
            v["ms_per_frame"] += row["ms"] / phase["frames"]
            v["bytes_per_frame"] += row["bytes"] / phase["frames"]
        phase["gpu_by_operation"] = totals
        cpu = {}
        for row in records:
            if row["type"] == "cpu" and phase["begin_ns"] <= row["start_ns"] and row["stop_ns"] <= phase["end_ns"]:
                cpu[row["name"]] = cpu.get(row["name"], 0) + row["ms"] / phase["frames"]
        phase["overlapping_cpu_ms_per_frame"] = cpu
    report = {"diagnostic_only": True, "shape": a.shape, "render": render, "output": output,
              "warm_seconds": a.warm, "phases": phases, "instrumented_gpu_span_samples_ms": times,
              "instrumented_gpu_span_median_ms": float(np.median(times)), "kernels": kernels,
              "capture": str(capture), "limits": "Instrumentation perturbs scheduling. BOP barrier intervals are not isolated barrier costs. CPU scopes overlap; do not add them."}
    (a.out / "summary.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({"summary": str(a.out / "summary.json"), "gpu_span_median_ms": report["instrumented_gpu_span_median_ms"]}))


if __name__ == "__main__":
    main()
