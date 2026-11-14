# SPDX-License-Identifier: MIT
"""Model L (issue #89): per-kernel GPU time of the native D3D12 frame (runtime backend 3), synthetic models only, by
the runtime's D3D12 operation profile (WG_D3D12_PROFILE, as clean/dx12/profile_frames.py), grouped by step: Model L's
encoder (step 2), the ten Swin stages, Model L's decoder (step 4), the reconstruction / filter, the rest.

    python clean/l/runner/l_d3d12_profile.py --build <dir> --model L|M --out <record.json> [--shape 1080p-quality]
        [--frames 100] [--warm 3]

<dir>: the build of clean/l/runner/l_frame_check.py --prepare (the synthetic models in <dir>/lframe_model) with the
D3D12 ports (clean/dx12/build_dx12.py --out <dir>). One model per process: the runtime library reads WG_D3D12_PROFILE
when it is loaded. Run it in a GPU slot with the GPU otherwise idle. Profiled timestamps perturb scheduling: the step
sums are for attribution; the whole-frame span without profiling is measured too (wg_rt_frame_timed, a separate
Pipeline in a second process would be cleaner; here it is the same process after the profiled phase, reported apart).
Output: the record JSON (per kernel: launches and GPU ms per frame; per step; the frame span samples).
"""
import argparse
import ctypes
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, ROOT)

SHAPES = {"1080p-quality": ((1280, 720), (1920, 1080)), "1440p-quality": ((1706, 960), (2560, 1440))}
ENC = ("lev/input_encoder", "l_linear_32_32_relu", "l_linear_32_32_f16", "l_swin_block_c32_enc", "l_project_gather",
       "input_encoder")
DEC = ("l_dec_project", "l_swin_block_c32_dec", "l_coef", "coef_decoder", "_add_dec0")
POST = ("post_", "scale_rgb", "f16lut")
SWIN = ("swin_", "patch_merge", "expand_", "bneck/")


def step(name):
    """The frame step of a launched kernel (by name)."""
    n = name.split("/", 1)[-1] if not name.startswith(("lev/", "bneck/")) else name
    if any(name.startswith(k) or n.startswith(k) for k in ENC):
        return "encoder"
    if any(k in name for k in DEC):
        return "decoder"
    if any(n.startswith(k) or name.startswith(k) for k in SWIN):
        return "swin"
    if any(n.startswith(k) for k in POST):
        return "post"
    return "other"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--build", required=True)
    ap.add_argument("--model", choices=("L", "M"), required=True)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--shape", choices=tuple(SHAPES), default="1080p-quality")
    ap.add_argument("--frames", type=int, default=100)
    ap.add_argument("--warm", type=float, default=3)
    a = ap.parse_args()
    capture = a.out.with_suffix(".operations.jsonl")
    if capture.exists():
        capture.unlink()
    os.environ["WG_D3D12_PROFILE"] = str(capture.resolve())  # before the runtime library is loaded
    os.environ.pop("WG_DX12_DEBUG", None)
    from clean.l.runner import l_frame_check as F
    from clean.runner.pipeline import Pipeline

    render, output = SHAPES[a.shape]
    d = F.model_dir(a.build)
    kw = dict(map_path=F.l_map(a.build), check_packing=False, kernels={"options": "model=L"}) if a.model == "L" else \
        dict(check_packing=False, kernels={})
    flags = {"depth_inverted": False, "auto_exposure": True, "hdr": a.model == "L"}
    p = Pipeline(a.build, "", d, render, output, flags, backend="d3d12", **kw)
    rng = np.random.default_rng(F.SEED)
    x = F.frame_inputs(rng, render[0], render[1], False, special=False)
    p.frame(x)
    fr = p._upload(x)

    def frame():
        p._check(p.lib.wg_rt_frame(p.rt, ctypes.byref(fr)), "frame")

    def sync():
        p._check(p.lib.wg_rt_sync(p.rt), "sync")

    def sequence():
        if not capture.exists():
            return 0
        return max([r["sequence"] for r in map(json.loads, capture.read_text().splitlines()) if r["type"] == "gpu"]
                   or [0])

    start = time.perf_counter()
    while time.perf_counter() - start < a.warm:
        for _ in range(20):
            frame()
        sync()
    begin = sequence()
    for _ in range(a.frames):
        frame()
    sync()
    end = sequence()
    spans = []
    for _ in range(30):
        ms = ctypes.c_float()
        p._check(p.lib.wg_rt_frame_timed(p.rt, ctypes.byref(fr), ctypes.byref(ms)), "timed frame")
        spans.append(ms.value)
    kernels = p.kernels()
    del p
    rows = [json.loads(s) for s in capture.read_text().splitlines()]
    gpu = [r for r in rows if r["type"] == "gpu" and begin < r["sequence"] <= end]
    per_kernel = {}
    for r in gpu:
        if r["kind"] != "dispatch":
            continue
        v = per_kernel.setdefault(r["name"], {"launches_per_frame": 0.0, "ms_per_frame": 0.0, "step": step(r["name"])})
        v["launches_per_frame"] += 1 / a.frames
        v["ms_per_frame"] += r["ms"] / a.frames
    other_ops = {}
    for r in gpu:
        if r["kind"] != "dispatch":
            key = r["kind"] + "/" + r["name"]
            other_ops[key] = other_ops.get(key, 0.0) + r["ms"] / a.frames
    steps = {}
    for k, v in per_kernel.items():
        steps[v["step"]] = steps.get(v["step"], 0.0) + v["ms_per_frame"]
    rec = {"diagnostic_only": True, "model": a.model, "shape": a.shape, "render": render, "output": output,
           "frames_profiled": a.frames, "warm_seconds": a.warm,
           "per_kernel": dict(sorted(per_kernel.items(), key=lambda kv: -kv[1]["ms_per_frame"])),
           "per_step_ms": steps, "dispatch_sum_ms": sum(steps.values()), "non_dispatch_ops_ms": other_ops,
           "frame_span_ms_samples": spans, "frame_span_ms_median": float(np.median(spans)),
           "kernels": kernels, "capture": str(capture),
           "limits": "Instrumented timestamps perturb scheduling; per-kernel times are attribution, not isolated "
                     "costs. Synthetic model and frames."}
    a.out.write_text(json.dumps(rec, indent=1))
    print(json.dumps({"model": a.model, "per_step_ms": {k: round(v, 4) for k, v in steps.items()},
                      "frame_span_ms_median": round(rec["frame_span_ms_median"], 4)}))
    for k, v in list(rec["per_kernel"].items())[:25]:
        print("%-45s %-8s %5.2f launches %8.4f ms" % (k, v["step"], v["launches_per_frame"], v["ms_per_frame"]))


if __name__ == "__main__":
    main()
