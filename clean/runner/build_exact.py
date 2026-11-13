# SPDX-License-Identifier: MIT
"""Whole frames through two builds of the runtime library with the SAME kernel choices and options: every stage
output, every state image and the output must be equal byte for byte. For a change that must not alter anything
unless a new option is given (e.g. motion vectors at output resolution: runtime option motion_outres).

    python the GPU lock (WG_GPU_LOCK, BUILDING.md) -- python -m clean.runner.build_exact --a-build <old build> --b-build <new build>
        --rocm <ROCm> --models <empty work dir> [--frames 24] [--json report.json]

Synthetic Model M parameters without and with made-up output-filter weights (clean/runner/synthetic_model.py), the
frame stream of clean/runner/output_filter_original_check.py (resets, render-size changes, changing exposure and
pre-exposure), each with a given exposure and with auto-exposure, over the kernel paths of that check: the default
(true fusion), post_tf=0 (banded), the unbanded two-kernel path, and the output filters analytic / quality5 /
quality9. Exit status 0 only if everything is identical.
"""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

from clean.runner import output_filter_original_check as OC
from clean.runner import synthetic_model as SM

CASES = [("plain", {}), ("taps", {}), ("taps", {"options": "post_tf=0"}),
         ("taps", {"post": "post_recon2_r", "filter": "post_filter_c1_s4"}),
         ("taps", {"options": "output_filter=analytic"}), ("taps", {"options": "output_filter=quality5"}),
         ("taps", {"options": "output_filter=quality9"}), ("plain", {"options": "post_diet=0"})]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--a-build", required=True)
    ap.add_argument("--b-build", required=True)
    ap.add_argument("--rocm", required=True)
    ap.add_argument("--models", required=True, help="work directory for the synthetic models")
    ap.add_argument("--frames", type=int, default=24)
    ap.add_argument("--json")
    args = ap.parse_args(argv)
    root = Path(args.models)
    if (root / "model.wgcm").exists():
        raise ValueError("refusing to write synthetic parameters into a converted-model directory")
    models = {"plain": root / "plain", "taps": root / "taps"}
    SM.write_raw(str(models["plain"]), SM.synthetic_tensors(OC.SEED))
    SM.write_raw(str(models["taps"]), SM.synthetic_tensors(OC.SEED, taps=SM.output_taps("random", OC.SEED)))
    sha = {k: hashlib.sha256((Path(b) / "wronggpu_runtime.dll").read_bytes()).hexdigest() if os.name == "nt" else None
           for k, b in (("a", args.a_build), ("b", args.b_build))}
    xs = OC.stream(args.frames)
    rep, bad = {"runtime_sha256": sha, "frames": args.frames, "cases": []}, 0
    for auto in (False, True):
        for model, kernels in CASES:
            a = OC.make(argparse.Namespace(build=args.a_build, rocm=args.rocm), models[model], auto, **kernels)
            b = OC.make(argparse.Namespace(build=args.b_build, rocm=args.rocm), models[model], auto, **kernels)
            row = {"model": model, "kernels": kernels, "auto_exposure": auto, "values": 0, "identical": True,
                   "post": set()}
            try:
                for x in xs:
                    da, db = a.frame(x, debug=True), b.frame(x, debug=True)
                    row["values"] += OC.compare_tree(da, db, skip=() if OC.display_written(b) else ("post.display",))
                    row["post"].add(b.kernels()["post_last"] + " / " + b.kernels()["filter_last"])
            except AssertionError as e:
                row["identical"], row["error"] = False, str(e)
                bad += 1
            finally:
                OC.close(a, b)
            row["post"] = sorted(row["post"])
            rep["cases"].append(row)
            print(json.dumps(row), flush=True)
    print(f"build_exact: {len(rep['cases'])} cases, {sum(r['values'] for r in rep['cases'])} values, "
          f"{bad} case(s) differ")
    if args.json:
        with open(args.json, "w") as f:
            json.dump(rep, f, indent=1)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
