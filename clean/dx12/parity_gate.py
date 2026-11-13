# SPDX-License-Identifier: MIT
"""Fail on the measured complete-frame native/HIP gap; never infer idle from unknown telemetry.

Consumes one warmed matched time_frames JSON and its activity capture. It gates synchronized streamed wall
time only. GPU event validity is reported independently; invalid event values are never silently discarded.
Exit 0 parity, 1 measured slower, 2 invalid/unqualified evidence. No GPU workload is run by this evaluator.
"""
import argparse
import json
import math
from pathlib import Path
import statistics


def evaluate(timing, telemetry):
    pairs = []
    problems = []
    rounds = timing.get("rounds", {})
    hip, native = rounds.get("hip", []), rounds.get("d3d12", [])
    if len(hip) != len(native) or not hip:
        problems.append("missing matched rounds")
    for i, (h, n) in enumerate(zip(hip, native)):
        for v in (h, n):
            samples = v.get("stream_blocks_ms", [])
            if v.get("warm_seconds", 0) < 3 or not samples or not all(isinstance(x, (int, float)) and math.isfinite(x) and x > 0 for x in samples):
                problems.append(f"round {i}: invalid wall samples/warm-up")
            windows = [w for w in telemetry.get("qualification", {}).get("windows", [])
                       if w["begin"].get("pid") == v.get("pid") and w["begin"].get("phase") == "stream"]
            if len(windows) != 1 or not windows[0].get("qualified"):
                problems.append(f"round {i} pid {v.get('pid')}: unqualified stream activity window")
        if (h.get("render"), h.get("output"), h.get("flags")) != (n.get("render"), n.get("output"), n.get("flags")):
            problems.append(f"round {i}: shape/settings mismatch")
        pairs.append({"round": i, "hip_ms": h["stream"], "native_ms": n["stream"],
                      "delta_ms": n["stream"]-h["stream"], "ratio": n["stream"]/h["stream"]})
    for phase in ("pre", "post"):
        if not telemetry.get("qualification", {}).get("pre_post", {}).get(phase, {}).get("qualified"):
            problems.append(phase + " activity gate failed")
    event_errors = []
    for c in ("hip", "d3d12"):
        for i, v in enumerate(rounds.get(c, [])):
            bad = [{"index": j, "value": x} for j, x in enumerate(v.get("event_samples_ms", []))
                   if not isinstance(x, (int, float)) or not math.isfinite(x) or x <= 0]
            if bad:
                event_errors.append({"config": c, "round": i, "invalid_samples": bad})
    median_delta = statistics.median(p["delta_ms"] for p in pairs) if pairs else None
    parity = not problems and median_delta <= 0 and all(p["delta_ms"] <= 0 for p in pairs)
    return {"status": "UNQUALIFIED" if problems else "PARITY" if parity else "NATIVE_SLOWER", "parity": parity,
            "activity_bounds": telemetry.get("criteria"), "strict_zero_background": False,
            "pairs": pairs, "median_paired_delta_ms": median_delta, "problems": problems,
            "invalid_event_windows": event_errors,
            "scope": "Synchronized pre-uploaded whole-frame wall time under recorded activity bounds; not idle-GPU proof or game FPS"}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--timing", required=True, type=Path)
    p.add_argument("--telemetry", required=True, type=Path)
    p.add_argument("--json", required=True, type=Path)
    a = p.parse_args()
    report = evaluate(json.loads(a.timing.read_text()), json.loads(a.telemetry.read_text()))
    a.json.write_text(json.dumps(report, indent=2))
    print(json.dumps({k: report[k] for k in ("status", "median_paired_delta_ms", "problems", "invalid_event_windows")}))
    return 2 if report["problems"] else 0 if report["parity"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
