# SPDX-License-Identifier: MIT
"""Original synthetic detail/phase/alias checks; no learned parameters or vendor outputs.

python -m clean.runner.output_quality_check --profile baseline|quality9 --json <result>
The baseline is expected to fail the geometric ramp gate; this measures filter design, not NVIDIA equivalence.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from clean.reference import post
from clean.reference import output_quality


def measure(fn):
    ow = 128
    dw = post.recon_size_for(ow)
    input_centers = np.arange(dw) + 0.5
    output_centers = (np.arange(ow) + 0.5) * 1.5
    ramp = fn(input_centers, ow)
    ramp_error = float(np.abs(ramp[4:-4] - output_centers[4:-4]).max())
    flat_error = float(np.abs(fn(np.ones(dw), ow) - 1.0).max())
    detail_rows = []
    temporal_rows = []
    for frequency in (0.05, 0.10, 0.15, 0.20, 0.25, 0.30):
        residuals = []
        gains = []
        for phase in np.linspace(0, 1.0, 32, endpoint=False):
            source = (0.8 + 0.04 * np.sin(2 * np.pi * frequency * (input_centers - phase))).astype(np.float16)
            truth = 0.8 + 0.04 * np.sin(2 * np.pi * frequency * (output_centers - phase))
            got = fn(source, ow)
            residuals.append((got[4:-4] - truth[4:-4]) / 0.04)
            basis = np.sin(2 * np.pi * frequency * (output_centers[4:-4] - phase))
            gains.append(float(np.dot(got[4:-4] - 0.8, basis) / np.dot(0.04 * basis, basis)))
        residuals = np.asarray(residuals)
        detail_rows.append({"cycles_per_output_pixel": frequency * 1.5,
                            "relative_signal_rmse": float(np.sqrt(np.mean(residuals ** 2))),
                            "contrast_retention": float(np.mean(gains))})
        temporal_rows.append(float(np.sqrt(np.mean(np.var(residuals, axis=0)))))
    alias_rows = []
    for frequency in np.linspace(1 / 3 + 0.005, 0.5, 40):
        for phase in np.linspace(0, 1.0, 16, endpoint=False):
            source = (0.8 + 0.04 * np.sin(2 * np.pi * frequency * (input_centers - phase))).astype(np.float16)
            got = fn(source, ow)
            alias_rows.append(float(np.sqrt(np.mean(((got[4:-4] - 0.8) / 0.04) ** 2))))
    overshoot = 0.0
    for cut in np.linspace(dw / 2 - 1, dw / 2 + 1, 17):
        source = np.where(input_centers < cut, 0.5, 1.0)
        got = fn(source, ow)
        overshoot = max(overshoot, float(np.maximum(got - 1, 0).max()), float(np.maximum(0.5 - got, 0).max()))
    return {"max_interior_ramp_phase_error_D_pixels": ramp_error, "flat_error": flat_error,
            "inband_detail_cases": detail_rows,
            "inband_aggregate_rmse": float(np.sqrt(np.mean([r["relative_signal_rmse"] ** 2 for r in detail_rows]))),
            "moving_pattern_residual_std": float(np.sqrt(np.mean(np.square(temporal_rows)))),
            "out_of_band_alias_rms": float(np.sqrt(np.mean(np.square(alias_rows)))),
            "out_of_band_alias_worst": float(max(alias_rows)), "step_overshoot": overshoot}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--profile", choices=("baseline", "quality9"), required=True)
    ap.add_argument("--json", required=True)
    args = ap.parse_args()
    control = measure(post.filter_pass)
    got = control if args.profile == "baseline" else measure(output_quality.filter_pass)
    gates = {"phase_aligned": got["max_interior_ramp_phase_error_D_pixels"] < 1e-5,
             "flat_preserved": got["flat_error"] < 1e-12,
             "detail_error_not_worse": got["inband_aggregate_rmse"] <= control["inband_aggregate_rmse"],
             "motion_residual_not_worse": got["moving_pattern_residual_std"] <= control["moving_pattern_residual_std"],
             "alias_rms_not_worse": got["out_of_band_alias_rms"] <= control["out_of_band_alias_rms"],
             "alias_worst_not_worse": got["out_of_band_alias_worst"] <= control["out_of_band_alias_worst"],
             "no_step_overshoot": got["step_overshoot"] < 1e-12}
    report = {"profile": args.profile, "data": "original synthetic encoded display patterns; no model/NVIDIA/game data",
              "reference_implementation": "CPU float64, half-quantized pattern inputs", "baseline": control,
              "candidate": got, "gates": gates, "passed": all(gates.values()),
              "weights_sha256": hashlib.sha256(output_quality.weights().astype('<f4').tobytes()).hexdigest()}
    Path(args.json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.json).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k not in ("baseline", "candidate")}, indent=2))
    print("baseline metrics", json.dumps(control))
    print("candidate metrics", json.dumps(got))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
