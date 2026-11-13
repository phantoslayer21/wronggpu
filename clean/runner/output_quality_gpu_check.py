# SPDX-License-Identifier: MIT
"""RX 9070 output-quality9 numerical, write-boundary, phase and resident-filter checks.

Only original synthetic display inputs. This tests the filter, not a learned network or native NVIDIA output.
Run under the GPU lock (WG_GPU_LOCK, BUILDING.md) from a console-backed process. CPU reference: clean/reference/output_quality.py.
"""
import argparse
import ctypes
import hashlib
import json
from pathlib import Path

import numpy as np

from clean.reference import post as ref
from clean.reference import output_quality as quality
from clean.runner import post
from clean.runner.stage_runner import Runner


def run(runner, image, exposure, kernel, co):
    dh, dw = image.shape[:2]
    ow, oh = post.output_size_for(dw, dh)
    x = {"display": image, "exposure": exposure, "output_size": (ow, oh)}
    return post.run_plan(runner, post.prepare_filter(runner.dev, b"", x, kernel), co)["out"]


def check_pitched_half(runner, image, exposure, pi, co):
    dh, dw = image.shape[:2]
    ow, oh = post.output_size_for(dw, dh)
    x = {"display": image, "exposure": exposure, "output_size": (ow, oh)}
    plan = post.prepare_filter(runner.dev, b"", x, "post_filter_quality9_o16")
    dev = runner.dev
    pitch = ((8 * ow + 255) // 256) * 256 + 256
    total = 32 + pitch * oh
    dev.free(plan["bufs"]["out"])
    allocation = dev.malloc(total)
    plan["bufs"]["out"] = allocation
    try:
        dev.memset(allocation, 0x55, total)
        P, I, F = ctypes.c_void_p, ctypes.c_int, ctypes.c_float
        args = [P(plan["bufs"]["display"]), P(allocation + 16), I(dw), I(dh), I(ow), I(oh),
                F(0.01 / ref.exposure_or_one(exposure)), P(0), F(pi), I(pitch)]
        grid = post.filter_grid(ow, oh, "post_filter_quality9")
        dev.launch(post._function(runner, "post_filter_quality9_h", co), grid, 64, args)
        dev.sync()
        raw = np.frombuffer(dev.download(allocation, total), dtype=np.uint8)
        rows = raw[16:-16].reshape(oh, pitch)
        if not np.all(raw[:16] == 0x55) or not np.all(raw[-16:] == 0x55) or not np.all(rows[:, 8 * ow:] == 0x55):
            raise AssertionError("quality9 wrote into guard bytes or row padding")
        got = np.ascontiguousarray(rows[:, :8 * ow]).view(np.float16).reshape(oh, ow, 4)
        if not np.isfinite(got).all() or not np.all(got[..., 3] == np.float16(1)):
            raise AssertionError("quality9 half output is nonfinite or alpha was not written")
        expected = np.clip(quality.output_filter(image, exposure, (ow, oh)) * pi, -65504, 65504).astype(np.float16)
        delta = np.abs(got[..., :3].astype(np.float64) - expected.astype(np.float64))
        rel = delta / np.maximum(abs(expected.astype(np.float64)), 1e-6)
        if rel.max(initial=0) > 0.002:
            raise AssertionError("quality9 pitched-half output differs materially from its reference")
        return {"output": [ow, oh], "exposure": exposure, "pre_exposure": pi, "pitch_bytes": pitch,
                "padding_and_guards_intact": True, "alpha_one": True, "nonfinite": 0,
                "max_relative_half_error": float(rel.max(initial=0))}
    finally:
        for ptr in plan["bufs"].values():
            dev.free(ptr)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--build", required=True)
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--rocm", required=True)
    ap.add_argument("--json", required=True)
    ap.add_argument("--timing", action="store_true")
    args = ap.parse_args()
    candidate_root, baseline_root = Path(args.build).resolve(), Path(args.baseline).resolve()
    if candidate_root == baseline_root:
        raise ValueError("Candidate and baseline build directories must be distinct")
    runner = Runner(str(candidate_root), args.rocm)
    control = Runner(str(baseline_root), args.rocm)
    rng = np.random.default_rng(61011)
    cases = []
    for ow, oh in ((1, 1), (2, 3), (3, 2), (17, 11), (101, 57), (128, 64), (127, 61)):
        image = rng.uniform(0.001, 2.45, (ref.recon_size_for(oh), ref.recon_size_for(ow), 3)).astype(np.float16).astype(np.float32)
        cases.append((image, 1.0))
    for value, exposure in ((0.0, 1.0), (1e-4, 1.0), (0.5, 1.0), (2.45, 0.001), (2.45, 2.0)):
        cases.append((np.full((17, 26, 3), np.float16(value), np.float32), exposure))
    for phase in np.linspace(0, 1, 8, endpoint=False):
        xs = np.arange(192)[None, :, None] + 0.5
        signal = 0.8 + 0.04 * np.sin(2 * np.pi * 0.25 * (xs - phase))
        cases.append((np.broadcast_to(signal, (48, 192, 3)).astype(np.float16).astype(np.float32), 1.0))
    records, half_records = [], []
    default_differences = 0
    for co in ("post.co", "post_w64.co"):
        for index, (image, exposure) in enumerate(cases):
            old = run(control, image, exposure, "post_filter_c1_s4", co)
            unchanged = run(runner, image, exposure, "post_filter_c1_s4", co)
            count = int(np.count_nonzero(old.view(np.uint32) != unchanged.view(np.uint32)))
            default_differences += count
            got = run(runner, image, exposure, "post_filter_quality9", co)
            output_size = post.output_size_for(image.shape[1], image.shape[0])
            expected = quality.output_filter(image, exposure, output_size)
            relative = abs(got.astype(np.float64) - expected) / np.maximum(abs(expected), 1e-8)
            if count or not np.isfinite(got).all() or relative.max() > 0.01 or np.mean(relative > 0.001) > 0.001:
                raise AssertionError((co, index, count, float(relative.max())))
            records.append({"code_object": co, "case": index, "values": int(got.size),
                            "default_changed_bits": count, "max_relative_error": float(relative.max()),
                            "fraction_over_0_001": float(np.mean(relative > 0.001)), "nonfinite": 0})
        for index in (4, 8, 10, 11):
            image, exposure = cases[index]
            for pi in (1.0, 2.0):
                half_records.append(check_pitched_half(runner, image, exposure, pi, co))
    ow, oh = 128, 32
    dw, dh = ref.recon_size_for(ow), ref.recon_size_for(oh)
    xs = np.arange(dw) + 0.5
    encoded_ramp = 0.5 + xs / 1024
    image = np.broadcast_to(encoded_ramp[None, :, None], (dh, dw, 3)).astype(np.float16).astype(np.float32)
    phase = {}
    target = (np.arange(ow) + 0.5) * 1.5
    for label, kernel in (("baseline", "post_filter_c1_s4"), ("quality9", "post_filter_quality9")):
        out = run(runner, image, 1.0, kernel, "post_w64.co")
        measured = (ref.enc(out[oh // 2, :, 0]) - 0.5) * 1024
        phase[label] = float(abs(measured[4:-4] - target[4:-4]).max())
    if phase["quality9"] > 0.001 or phase["baseline"] < 0.08:
        raise AssertionError(("phase gate", phase))
    timing = []
    if args.timing:
        for ow, oh in ((2560, 1440), (3840, 2160)):
            shape = (ref.recon_size_for(oh), ref.recon_size_for(ow), 3)
            image = rng.uniform(0.5, 1.0, shape).astype(np.float16).astype(np.float32)
            x = {"display": image, "exposure": 1.0, "output_size": (ow, oh)}
            for cycle in range(3):
                for label, kernel in (("baseline", "post_filter_c1_s4"), ("quality9", "post_filter_quality9")):
                    plan = post.prepare_filter(runner.dev, b"", x, kernel)
                    ms = post.time_plans(runner, [plan], "post_w64.co", repeats=30)
                    timing.append({"output": [ow, oh], "cycle": cycle, "filter": label, "resident_filter_ms": ms})
    report = {"passed": True, "hardware": runner.dev.device_name, "data": "original synthetic display patterns",
              "scope": "isolated GPU output filter; no neural inference, vendor comparison or real-game test",
              "default_differing_values": default_differences, "cases": records, "half_output_cases": half_records,
              "phase_error_D_pixels": phase, "resident_stage_timing": timing,
              "artifacts": {co: hashlib.sha256((candidate_root / co).read_bytes()).hexdigest()
                            for co in ("post.co", "post_w64.co")}}
    Path(args.json).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k not in ("cases", "half_output_cases")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
