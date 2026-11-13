# SPDX-License-Identifier: MIT
"""Alternating synthetic Model K full-interop benchmark; run under the shared GPU lock.

Requires a locally built test_interop_d3d12.exe with hipbench and two runtime/kernel
build directories. Generates its own random model; never reads an installed model.
TOTAL is the measured D3D12-input-through-output span, including HIP handover and
all exposure/input/network/output work. Summary is a median of 300-dispatch window
means after discarding the first window, not a median of individual frame samples.
"""
import argparse
import json
import os
from pathlib import Path
import re
import statistics
import subprocess

from clean.k.reference.network import random_network_params
from clean.k.tests.test_k_runtime_gpu import write_k_model


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--exe", required=True)
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-stage-events", action="store_true", help="D3D12 TOTAL only; require both builds to honor timing_stages=0")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--frames", type=int, default=1500)
    a = ap.parse_args()
    if a.rounds < 1 or a.frames < 900:
        ap.error("rounds >= 1 and frames >= 900 required")
    out = Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    model = out / "synthetic"
    wgcm, mapfile = write_k_model(str(model), random_network_params(11))
    exe = str(Path(a.exe).resolve())
    env = os.environ.copy()
    env.pop("WG_RUNTIME_OPTIONS", None)
    rows = []
    fingerprint = None
    for rnd in range(a.rounds):
        order = [("baseline", a.baseline), ("candidate", a.candidate)]
        if rnd % 2:
            order.reverse()
        for label, path in order:
            build = Path(path).resolve()
            opts = ";".join(("model=K", "network=fused", "timing_log=1", "done_mode=fence",
                             f"kernels_dir={build}", f"model_dir={model}",
                             f"model_k_dir={Path(wgcm).parent}", f"model_k_map={mapfile}",
                             "timing_stages=0" if a.no_stage_events else "timing_stages=1"))
            proc = subprocess.run([exe, str(build / "wronggpu_runtime.dll"), "hipbench", opts, str(a.frames)],
                                  capture_output=True, text=True, env=env, check=False, timeout=120)
            log = proc.stdout + proc.stderr
            (out / f"{rnd}-{label}.log").write_text(log, encoding="utf-8")
            if proc.returncode or not log.rstrip().endswith("PASS"):
                raise RuntimeError(f"{label} failed; see {out / f'{rnd}-{label}.log'}")
            for required in ("network kernels: model=K network=fused ", "base=640x384", "hdr=1",
                             "output: CopyTextureRegion"):
                if required not in log:
                    raise RuntimeError(f"missing required path evidence: {required}")
            if a.no_stage_events and "HIP kernel events disabled by timing_stages=0" not in log:
                raise RuntimeError("build did not honor timing_stages=0; unequal instrumentation would invalidate A/B")
            windows = [float(x) for x in re.findall(r"TOTAL\(t3-t0\) ([0-9.]+)", log)]
            if len(windows) < 3:
                raise RuntimeError("not enough measured full-interop windows")
            match = re.search(r"hipbench final output: (\d+) values, nonfinite=(\d+), fnv1a64=([0-9a-f]+)", log)
            if not match or int(match[2]):
                raise RuntimeError("missing or non-finite final output")
            if fingerprint is None:
                fingerprint = match[3]
            elif fingerprint != match[3]:
                raise RuntimeError("final output checksum differs; do not accept the speed result")
            stage_pattern = (r"network GPU ([0-9.]+) .*?\[exposure ([0-9.]+), input ([0-9.]+), "
                             r"network ([0-9.]+), output ([0-9.]+)\] \((\d+) frames\)")
            stage_windows = [dict(hip_ms=float(v[0]), exposure_ms=float(v[1]), input_ms=float(v[2]),
                                  network_ms=float(v[3]), output_ms=float(v[4]), frames=int(v[5]))
                             for v in re.findall(stage_pattern, log)]
            row = dict(round=rnd, build=label, windows_ms=windows, warmup_windows=1,
                       stage_windows=stage_windows,
                       median_window_mean_ms=statistics.median(windows[1:]),
                       output_values=int(match[1]), nonfinite=0, output_checksum=match[3])
            rows.append(row)
            print(json.dumps(row), flush=True)
    summary = {label: statistics.median([r["median_window_mean_ms"] for r in rows if r["build"] == label])
               for label in ("baseline", "candidate")}
    result = dict(scope="D3D12 input through final output, auto exposure, static synthetic colour and spatial motion field, cyclic jitter",
                  output=[2560, 1440], render=[1707, 960], base=[640, 384], network="fused FP32",
                  stage_events=not a.no_stage_events, frames_per_process=a.frames, rounds=rows, median_window_mean_ms=summary,
                  reduction_percent=100*(1-summary["candidate"]/summary["baseline"]))
    (out / "timings.json").write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
