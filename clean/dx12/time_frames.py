# SPDX-License-Identifier: MIT
"""Whole-frame GPU time of the synthetic Model M frame, pure HIP against the hybrid D3D12 device (all ports, or one port
at a time), at 1706x960 -> 2560x1440 (depth inverted, auto-exposure; the compare.py default case, its synthetic model).

    python clean/dx12/time_frames.py --build <build dir>
        [--configs hip,all,<kernel>,...] [--rounds 2] [--repeats 100] [--warm 3] [--json out.json]

Every measurement runs in a fresh process, the configs interleaved round by round. Clocks: the GPU needs about 2-3 s of
sustained load to reach its full shader clock (tools/clk probe: 1.6-2.2 GHz cold, 2.6-2.9 GHz warm; frame times differ
by up to 40%), so every process first runs --warm seconds of whole frames back to back. Then:
  stream  wall time of --repeats frames submitted back to back, then one sync, divided by --repeats (3 blocks, median);
  event   the runtime's wg_rt_frame_timed (HIP events around one frame; on the hybrid device the stop event waits for the
          D3D12 queue; backend d3d12 uses native timestamp queries): median and minimum over --repeats frames,
          each frame on its own (submit, wait). This is a whole-frame GPU span, not dispatch-only timing.
  complete wall time of Pipeline.frame including input uploads, frame execution and output readback.
Raw samples, host submission (including backpressure), and final waits are retained in JSON.
Configs: "hip" (backend 0), "all" (backend 2, every port), "none" (backend 2, no port: the device's own overhead), or a
kernel name as launched (backend 2, only that port on D3D12); "<config>@<n>" also sets WG_DX12_BATCH=n (the device
executes its command list every n D3D12 dispatches instead of only when it hands back to HIP); "<config>#<mb>" sets
WG_DX12_ARENA_MB=mb ("none#0": the hybrid device without an arena, i.e. HIP memory and HIP kernels only). 
"d3d12" selects native backend 3. Synthetic parameters only, never real model data.
"""
import argparse
import datetime
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RENDER, OUT = (1706, 960), (2560, 1440)
FLAGS = {"depth_inverted": True, "auto_exposure": True}
DEFAULT_MODELS = os.path.join(os.environ.get("WG_BUILD_ROOT", os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "build")), "dx12-m", "work", "compare_models")


def event_statistics(values):
    bad = [{"index": i, "ms": v if math.isfinite(v) else repr(v)}
           for i, v in enumerate(values) if not math.isfinite(v) or v <= 0]
    return {"event_valid": bool(values) and not bad, "invalid_event_samples": bad,
            "event_median": float(np.median(values)) if values and not bad else None,
            "event_min": float(min(values)) if values and not bad else None}


def number(value):
    return "INVALID" if value is None else f"{value:.3f}"


def child(a):
    sys.path.insert(0, ROOT)
    from clean.runner import output_filter_original_check as OC
    from clean.runner import synthetic_model as SM
    from clean.runner.pipeline import Pipeline
    model = os.path.join(a.models, "taps")
    if not os.path.exists(os.path.join(model, "raw")):
        SM.write_raw(model, SM.synthetic_tensors(OC.SEED, taps=SM.output_taps("random", OC.SEED)))
    render, output = ((1920, 1080), (3840, 2160)) if a.shape == "4k-performance" else (RENDER, OUT)
    def marker(kind, phase):
        directory = os.environ.get("WG_GPU_EVIDENCE_MARKER_DIR")
        if directory:
            Path(directory).mkdir(parents=True, exist_ok=True)
            with (Path(directory) / f"{os.getpid()}.jsonl").open("a") as f:
                f.write(json.dumps({"kind": kind, "phase": phase, "pid": os.getpid(),
                    "monotonic_ns": time.perf_counter_ns(), "utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    "config": a.child, "round": os.environ.get("WG_TIMING_ROUND"), "warm_seconds": a.warm}) + "\n")
    rest, _, arena = a.child.partition("#")
    if arena:
        os.environ["WG_DX12_ARENA_MB"] = arena
    sel, _, batch = rest.partition("@")
    if batch:
        os.environ["WG_DX12_BATCH"] = batch
    if sel not in ("hip", "d3d12"):
        os.environ["WG_DX12_KERNELS"] = sel
    selected_build = (a.native_build or a.build) if sel == "d3d12" else a.build
    p = Pipeline(selected_build, a.rocm, model, render, output, FLAGS, backend=sel if sel in ("hip", "d3d12") else "dx12",
                 model_format="raw", check_packing=False)
    x = OC.frame_input(1, render, reset=False)
    p.frame(x)
    fr = p._upload(x)
    fr.reset = 0
    import ctypes
    ms = ctypes.c_float()
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < a.warm:  # clocks up (see the module doc)
        for _ in range(20):
            p._check(p.lib.wg_rt_frame(p.rt, ctypes.byref(fr)), "frame")
        p._check(p.lib.wg_rt_sync(p.rt), "sync")
    blocks = []
    submissions, waits = [], []
    marker("steady_begin", "stream")
    for _ in range(3):
        t0 = time.perf_counter()
        for _ in range(a.repeats):
            p._check(p.lib.wg_rt_frame(p.rt, ctypes.byref(fr)), "frame")
        submitted = time.perf_counter()
        p._check(p.lib.wg_rt_sync(p.rt), "sync")
        finished = time.perf_counter()
        blocks.append((finished - t0) * 1e3 / a.repeats)
        submissions.append((submitted - t0) * 1e3 / a.repeats)
        waits.append((finished - submitted) * 1e3 / a.repeats)
    marker("steady_end", "stream")
    stream = float(np.median(blocks))
    ev = []
    marker("steady_begin", "event")
    for _ in range(a.repeats):
        p._check(p.lib.wg_rt_frame_timed(p.rt, ctypes.byref(fr), ctypes.byref(ms)), "frame")
        ev.append(ms.value)
    marker("steady_end", "event")
    complete = []
    if not a.frame_only:
        marker("steady_begin", "complete")
    for _ in range(0 if a.frame_only else a.repeats):
        t0 = time.perf_counter()
        p.frame(x)  # uploads the input, executes the frame and downloads the output
        complete.append((time.perf_counter() - t0) * 1e3)
    if not a.frame_only:
        marker("steady_end", "complete")
    uploads, frames, downloads = [], [], []
    if not a.frame_only:
        marker("steady_begin", "transfer-separated")
    for _ in range(0 if a.frame_only else a.repeats):
        t0 = time.perf_counter()
        fr = p._upload(x)
        p._check(p.lib.wg_rt_sync(p.rt), "upload sync")
        t1 = time.perf_counter()
        p._check(p.lib.wg_rt_frame(p.rt, ctypes.byref(fr)), "frame")
        p._check(p.lib.wg_rt_sync(p.rt), "frame sync")
        t2 = time.perf_counter()
        p._output()
        t3 = time.perf_counter()
        uploads.append((t1-t0)*1e3); frames.append((t2-t1)*1e3); downloads.append((t3-t2)*1e3)
    if not a.frame_only:
        marker("steady_end", "transfer-separated")
    result = {"config": a.child, **event_statistics(ev),
              "stream": stream, "complete_median": float(np.median(complete)) if complete else None,
              "event_samples_ms": [v if math.isfinite(v) else repr(v) for v in ev],
              "stream_blocks_ms": blocks, "complete_samples_ms": complete,
              "submission_blocks_ms": submissions, "final_wait_blocks_ms": waits,
              "event_clock": "D3D12 timestamps" if sel == "d3d12" else "HIP events",
              "warm_seconds": a.warm, "utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "render": render, "output": output, "flags": FLAGS, "kernels": p.kernels(),
              "pid": os.getpid(), "build": selected_build,
              "upload_sync_samples_ms": uploads, "frame_sync_samples_ms": frames, "readback_samples_ms": downloads,
              "upload_sync_median": float(np.median(uploads)) if uploads else None,
              "frame_sync_median": float(np.median(frames)) if frames else None,
              "readback_median": float(np.median(downloads)) if downloads else None,
              "host_transfer_measurements": not a.frame_only}
    # Keep the context alive for a fixed tail outside all measured windows so PDH can sample the final interval
    # before counter retirement. Raw retirement intervals remain in the evidence wrapper's capture.
    if os.environ.get("WG_GPU_EVIDENCE_MARKER_DIR"):
        time.sleep(1.0)
    OC.close(p)
    print("RESULT " + json.dumps(result), flush=True)
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--build", required=True)
    ap.add_argument("--native-build", help="optional separate native bundle")
    ap.add_argument("--shape", choices=("1440p", "4k-performance"), default="1440p")
    ap.add_argument("--frame-only", action="store_true", help="kernel experiment: preuploaded whole frames only; omit host transfer phases")
    ap.add_argument("--rocm", default=r"C:\Program Files\AMD\ROCm\7.2")
    ap.add_argument("--models", default=DEFAULT_MODELS)
    ap.add_argument("--configs", default="hip,all")
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--repeats", type=int, default=100)
    ap.add_argument("--warm", type=float, default=3.0, help="seconds of back-to-back frames before measuring")
    ap.add_argument("--json")
    ap.add_argument("--child", help=argparse.SUPPRESS)
    a = ap.parse_args(argv)
    if a.warm < 3 or a.repeats < 1 or a.rounds < 1:
        ap.error("at least 3 seconds of warm-up and positive repeats/rounds are required")
    for k in ("HIP_PATH", "ROCM_PATH", "LLVM_PATH", "HIP_DEVICE_LIB_PATH", "HIP_CLANG_PATH"):
        os.environ.pop(k, None)
    if a.child:
        return child(a)
    configs = a.configs.split(",")
    res = {c: [] for c in configs}
    failures = []
    for r in range(a.rounds):
        for c in configs:
            cmd = [sys.executable, os.path.abspath(__file__), "--build", a.build, "--rocm", a.rocm, "--models", a.models,
                   "--repeats", str(a.repeats), "--warm", str(a.warm), "--child", c]
            cmd += ["--shape", a.shape]
            if a.native_build:
                cmd += ["--native-build", a.native_build]
            if a.frame_only:
                cmd += ["--frame-only"]
            env = os.environ.copy()
            env["WG_TIMING_ROUND"] = str(r)
            out = subprocess.run(cmd, capture_output=True, text=True, env=env)
            line = [l for l in out.stdout.splitlines() if l.startswith("RESULT ")]
            if out.returncode != 0 or not line:
                print(f"round {r} {c}: FAILED\n{out.stdout[-2000:]}{out.stderr[-2000:]}", flush=True)
                failures.append({"round": r, "config": c, "returncode": out.returncode,
                                 "stdout": out.stdout, "stderr": out.stderr})
                continue
            v = json.loads(line[0][7:])
            res[c].append(v)
            if not v["event_valid"]:
                failures.append({"round": r, "config": c, "reason": "invalid event clock samples; wall samples retained",
                                 "invalid_event_samples": v["invalid_event_samples"]})
            print(f"round {r} {c:36s} event median {number(v['event_median'])} ms  min {number(v['event_min'])} ms  "
                  f"stream {v['stream']:.3f} ms/frame", flush=True)
    base = res.get("hip") and {k: (float(np.median([v[k] for v in res["hip"]]))
                                    if all(v[k] is not None for v in res["hip"]) else None)
                              for k in ("stream", "event_median")}
    print(f"\n{'config (ms per frame)':36s} {'stream':>8s} {'vs hip':>8s} {'event med':>10s} {'vs hip':>8s} {'event min':>10s}")
    summary = {}
    for c in configs:
        if not res[c]:
            continue
        s = {k: (float(np.median([v[k] for v in res[c]])) if all(v[k] is not None for v in res[c]) else None)
             for k in ("event_median", "event_min", "stream", "complete_median", "upload_sync_median", "frame_sync_median", "readback_median")}
        s["event_min"] = float(min(v["event_min"] for v in res[c])) if all(v["event_valid"] for v in res[c]) else None
        summary[c] = s
        ds = f"{s['stream'] - base['stream']:+.3f}" if base else ""
        de = f"{s['event_median'] - base['event_median']:+.3f}" if base and s['event_median'] is not None and base['event_median'] is not None else ""
        print(f"{c:36s} {s['stream']:8.3f} {ds:>8s} {number(s['event_median']):>10s} {de:>8s} {number(s['event_min']):>10s}")
    if a.json:
        with open(a.json, "w") as f:
            json.dump({"rounds": res, "summary": summary, "failures": failures,
                       "definitions": {"stream": "pre-uploaded frames then final sync, wall ms/frame",
                                       "event": "frame bracket, GPU clock; includes frame copies/barriers",
                                       "complete": "Pipeline.frame: input upload, frame, output download, wall ms",
                                       "submission": "host frame calls including queue backpressure, ms/frame",
                                       "final_wait": "final stream synchronization, amortized ms/frame"}}, f, indent=1)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
