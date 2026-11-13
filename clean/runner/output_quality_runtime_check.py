# SPDX-License-Identifier: MIT
"""Full original synthetic Model M frame/state comparison: default output vs an output-filter profile
(--profile quality9 or quality5).

quality5 runs the default kernels (true fusion included) from post_w64__q5.co; its frames are also run a second time
without the stage outputs (the path a game frame takes) and must give the same output bits. The fused kernels never
write the display image, so for quality5 the reference is evaluated on the display image of a quality9 run of the same
frames (the unfused reconstruction; its state is bit-identical to the default's, checked by --profile quality9).

No learned model data or NVIDIA oracle is used. Run serialized under the shared GPU lock with console-backed stdio.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from clean.reference import encoder
from clean.reference import output_quality
from clean.runner.pipeline import Pipeline
from clean.runner.synthetic_model import synthetic_tensors, write_raw


def compare_tree(a, b, prefix=""):
    values = 0
    for name in a:
        path = prefix + name
        if path in ("out", "filter", "post.display"):
            continue
        x, y = a[name], b[name]
        if isinstance(x, dict):
            values += compare_tree(x, y, path + ".")
        elif isinstance(x, np.ndarray):
            if x.shape != y.shape or x.dtype != y.dtype or x.tobytes() != y.tobytes():
                raise AssertionError("Neural/reconstruction/history state changed: " + path)
            values += x.size
        elif x != y:
            raise AssertionError("Frame metadata changed: " + path)
    return values


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--build", required=True)
    ap.add_argument("--rocm", required=True)
    ap.add_argument("--model", required=True, help="directory exclusively owned by this synthetic test")
    ap.add_argument("--json", required=True)
    ap.add_argument("--profile", choices=("quality9", "quality5"), default="quality9")
    args = ap.parse_args()
    model = Path(args.model)
    if (model / "model.wgcm").exists():
        raise ValueError("Refusing to write synthetic parameters into a converted-model directory")
    write_raw(str(model), synthetic_tensors(61011))
    records = []
    for automatic in (False, True):
        default = Pipeline(args.build, args.rocm, str(model), (127, 73), (127, 73),
                           {"depth_inverted": False, "auto_exposure": automatic}, model_format="raw",
                           check_packing=True, kernels={"options": "output_filter=default"})
        candidate = Pipeline(args.build, args.rocm, str(model), (127, 73), (127, 73),
                             {"depth_inverted": False, "auto_exposure": automatic}, model_format="raw",
                             check_packing=True, kernels={"options": "output_filter=" + args.profile})
        plain = Pipeline(args.build, args.rocm, str(model), (127, 73), (127, 73),
                         {"depth_inverted": False, "auto_exposure": automatic}, model_format="raw",
                         check_packing=False, kernels={"options": "output_filter=" + args.profile})
        aux = None
        if args.profile == "quality5":
            aux = Pipeline(args.build, args.rocm, str(model), (127, 73), (127, 73),
                           {"depth_inverted": False, "auto_exposure": automatic}, model_format="raw",
                           check_packing=False, kernels={"options": "output_filter=quality9"})
        try:
            for frame in range(24):
                rw, rh = ((80, 49), (80, 49), (64, 40), (127, 73), (63, 37))[frame % 5]
                rng = np.random.default_rng([61011, frame])
                source = encoder.random_inputs(rng, rw=rw, rh=rh, fw=((rw + 31) // 32) * 32,
                                               fh=((rh + 31) // 32) * 32)
                pi = 1.0 if frame % 3 else 1.375
                x = {"color": source["color"], "depth": source["depth"], "motion": source["motion"],
                     "mv_scale": source["mv_scale"], "jitter": source["jitter"], "exposure": 0.75 + frame / 24,
                     "pre_exposure": pi, "reset": frame in (0, 5, 12)}
                old, got = default.frame(x, debug=True), candidate.frame(x, debug=True)
                state_values = compare_tree(old, got)
                selected = candidate.kernels()
                if args.profile == "quality9" and selected.get("filter_last") != "post_filter_quality9":
                    raise AssertionError("Runtime did not select the actual quality9 filter: " + str(selected))
                if selected.get("output_filter") != args.profile:
                    raise AssertionError("Runtime did not select the profile: " + str(selected))
                game = plain.frame(x)["out"]
                if game.tobytes() != got["out"].tobytes():
                    raise AssertionError(("Output differs between the frame with and without stage outputs", frame))
                display = aux.frame(x, debug=True)["post"]["display"] if aux else got["post"]["display"]
                expected = output_quality.output_filter(display, got["exposure"]["e"], (127, 73), args.profile)
                expected = (expected * np.float32(pi)).astype(np.float32)
                relative = abs(got["out"].astype(np.float64) - expected) / np.maximum(abs(expected), 1e-8)
                if not np.isfinite(got["out"]).all() or relative.max() > 0.01 or np.mean(relative > 0.001) > 0.001:
                    raise AssertionError(("Final output reference mismatch", frame, float(relative.max())))
                records.append({"automatic_exposure": automatic, "frame": frame, "render": [rw, rh],
                                "reset": bool(got["state"]["info"]["reset"]),
                                "state_values_bit_identical": int(state_values),
                                "default_vs_quality_output_differences": int(np.count_nonzero(old["out"].view(np.uint32) != got["out"].view(np.uint32))),
                                "quality_reference_max_relative_error": float(relative.max()), "nonfinite": 0,
                                "post_actual": selected["post_last"], "filter_actual": selected["filter_last"],
                                "post_plain": plain.kernels()["post_last"],
                                "filter_plain": plain.kernels()["filter_last"]})
        finally:
            for pipeline in (default, candidate, plain, aux):
                if pipeline and pipeline.rt:
                    pipeline.lib.wg_rt_destroy(pipeline.rt)
                    pipeline.rt = None
    report = {"passed": True, "profile": args.profile, "scope": "full Model M GPU runtime using original synthetic parameters and frames",
              "learned_model_or_game_qualification": False, "frames": records,
              "state_values_bit_identical": sum(r["state_values_bit_identical"] for r in records),
              "runtime_sha256": hashlib.sha256((Path(args.build) / "wronggpu_runtime.dll").read_bytes()).hexdigest()}
    Path(args.json).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "frames"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
