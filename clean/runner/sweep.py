# SPDX-License-Identifier: MIT
"""Time every build variant of the Swin stages at the in-game sizes, with synthetic data (timing does not depend on
the values), and check that all variants produce bit-identical outputs.

    python clean/runner/build.py --rocm <ROCm> --out build --variants all
    python clean/runner/sweep.py --build build --rocm <ROCm> [--json sweep.json] [--repeats 50]

Prints, per stage, one line per variant: total median ms, the per-kernel split, and whether the output matches the
base variant bit for bit. Numbers only; safe to paste into results/.
"""
import argparse
import json
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np  # noqa: E402

from clean.reference.swin import STAGES, random_inputs, random_params, window_offset  # noqa: E402
from clean.runner import variants as V  # noqa: E402
from clean.runner.stage_runner import Runner  # noqa: E402

# stage -> (H, W, target ms) at the in-game size (1440p output, Quality): results/2026-10-05-swin-realdata-b0-b5.md
GAME_SIZES = {
    "enc1": (480, 864, 0.35), "enc2": (240, 432, 0.19),
    "b0": (120, 216, 0.07), "b1": (120, 216, 0.07),
    "dec2": (240, 432, 0.19), "dec1": (480, 864, 0.34),
    "bneck": (120, 216, 0.35),  # b0..b5 together: six launches, or the megakernel in the bmega* builds
    "mid": (240, 432, 0.75),    # enc2 (+ merge), b0..b5, dec2 (+ expand): ten launches, or swin_mid (bmega*)
}
MID_STAGES = ("enc2",) + tuple(f"b{k}" for k in range(6)) + ("dec2",)


def stage_params(stage, rng):
    """Synthetic parameters of a stage; "bneck": those of b0..b5."""
    if stage == "bneck":
        return {f"b{k}": random_params(f"b{k}", rng) for k in range(6)}
    if stage == "mid":
        return {s: random_params(s, rng) for s in MID_STAGES}
    return random_params(stage, rng)


def stage_inputs(stage, H, W, rng):
    return random_inputs({"bneck": "b0", "mid": "enc2"}.get(stage, stage), H, W, rng)


def stage_offset(stage, parity):
    return window_offset({"bneck": "b0", "mid": "enc2"}.get(stage, stage), parity)


def available_variants(build_dir, requested):
    manifest = os.path.join(build_dir, V.MANIFEST)
    if os.path.exists(manifest):
        with open(manifest) as f:
            built = list(json.load(f))
    else:
        built = ["base"]
    names = built if requested in (None, "all") else requested.split(",")  # 'all' includes ablation builds
    if requested in (None, "all"):
        names = [n for n in names if not V.is_profile(n)]    # instrumented builds are read with profile_stages.py
    missing = [n for n in names if n not in built]
    if missing:
        raise SystemExit(f"variants not built: {missing} (built: {built})")
    return ["base"] + [n for n in names if n != "base"]


def sweep(build_dir, rocm_dir="", backend="hip", stages=None, variants=None, repeats=50, sizes=None, seed=0):
    names = available_variants(build_dir, variants)
    runners = {n: Runner(build_dir, rocm_dir, backend=backend, variant=n) for n in names}
    report = {"device": runners["base"].dev.device_name, "repeats": repeats, "stages": {}}
    for stage in stages or list(GAME_SIZES):
        H, W, target = (sizes or GAME_SIZES)[stage]
        rng = np.random.default_rng(seed)
        params = stage_params(stage, rng)
        x, skip = stage_inputs(stage, H, W, rng)
        offset = stage_offset(stage, 0)
        base = runners["base"]
        weights = base.pack_weights(stage, params)
        ref_out = base.run(stage, weights, x, offset, skip)
        rows = {}
        for n in names:
            r = runners[n]
            ablation = V.is_ablation(n)
            exact = V.is_exact(n)
            if ablation:      # timing only: the output is wrong on purpose
                mism = None
            else:
                out = ref_out if n == "base" else r.run(stage, weights, x, offset, skip)
                mism = sum(int(np.sum(~((out[k] == ref_out[k]) | (np.isnan(out[k]) & np.isnan(ref_out[k])))))
                           for k in ref_out)
            kernels = r.time_kernels(stage, weights, x, offset, skip, repeats)
            rows[n] = {"ms": float(sum(kernels.values())), "kernels": kernels, "mismatches_vs_base": mism, "ablation": ablation, "exact": exact,
                       "whole_stage_ms": r.time(stage, weights, x, offset, skip, repeats)}
        report["stages"][stage] = {"size": [H, W], "target_ms": target, "offset": list(offset), "variants": rows}
    return report


def scaling(build_dir, rocm_dir="", backend="hip", stages=None, variant="base", repeats=30, factors=(0.5, 1, 2, 4), seed=0):
    """Time the base build of each stage at several map sizes (area factors relative to the in-game size). If the time
    per window drops a lot at larger sizes, the small grids pay a fixed or tail cost (launch ramp, partly filled last
    round of workgroups) that more work per launch would hide. Synthetic data."""
    runner = Runner(build_dir, rocm_dir, backend=backend, variant=variant)
    rows = {}
    for stage in stages or list(GAME_SIZES):
        H0, W0, _ = GAME_SIZES[stage]
        rng = np.random.default_rng(seed)
        params = stage_params(stage, rng)
        weights = runner.pack_weights(stage, params)
        offset = stage_offset(stage, 0)
        out = []
        for f in factors:
            s = np.sqrt(f)
            H, W = max(8, int(round(H0 * s / 4)) * 4), max(8, int(round(W0 * s / 4)) * 4)
            x, skip = stage_inputs(stage, H, W, rng)
            ms = runner.time(stage, weights, x, offset, skip, repeats)
            windows = ((W + offset[0] + 3) // 4) * ((H + offset[1] + 3) // 4) * {"bneck": 6, "mid": 1}.get(stage, 1)
            out.append({"factor": f, "size": [H, W], "ms": ms, "windows": windows, "ns_per_window": ms * 1e6 / windows})
        rows[stage] = out
    return rows


def format_scaling(rows):
    L = ["time per window at several map sizes (area factor vs the in-game size); last column: relative to the best (smallest) time per window"]
    for stage, out in rows.items():
        best = min(r["ns_per_window"] for r in out)
        L.append(f"\n{stage}")
        for r in out:
            L.append(f"  x{r['factor']:<4} {r['size'][0]}x{r['size'][1]:<5} {r['ms']:.3f} ms  {r['windows']:7d} windows  "
                     f"{r['ns_per_window']:.2f} ns/window   {r['ns_per_window'] / best:.2f}x of best")
    return "\n".join(L)


def format_report(report):
    lines = [f"device: {report['device']}   repeats: {report['repeats']}"]
    for stage, e in report["stages"].items():
        H, W = e["size"]
        lines.append(f"\n{stage}  {H}x{W}  offset {tuple(e['offset'])}  target {e['target_ms']:.2f} ms")
        real = {n: r for n, r in e["variants"].items() if not r["ablation"]}
        best = min(real, key=lambda n: real[n]["whole_stage_ms"])
        base_ms = e["variants"]["base"]["whole_stage_ms"]
        for n, r in e["variants"].items():
            split = "  ".join(f"{k}={v:.3f}" for k, v in r["kernels"].items())
            if r["ablation"]:
                lines.append(f"  {n:10s} {r['whole_stage_ms']:.3f} ms  (ablation, output wrong on purpose: removes "
                             f"{base_ms - r['whole_stage_ms']:+.3f} ms = {100 * (base_ms - r['whole_stage_ms']) / base_ms:.0f} % of base)")
                continue
            lines.append(f"  {n:10s} {r['whole_stage_ms']:.3f} ms ({r['whole_stage_ms'] / e['target_ms']:.2f}x target)"
                         f"  [{split}]  mismatches vs base: {r['mismatches_vs_base']}"
                         + ("" if r["exact"] else " (reorders FP32 sums, may differ)") + ("   <- best" if n == best else ""))
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--build", required=True)
    ap.add_argument("--rocm", default=os.environ.get("ROCM_PATH", ""))
    ap.add_argument("--backend", default="hip", choices=["hip", "emu"])
    ap.add_argument("--stages", help="comma-separated (default: all)")
    ap.add_argument("--variants", default="all", help="comma-separated names, or 'all' (everything in variants.json, "
                                                      "including ablation builds if they were built)")
    ap.add_argument("--repeats", type=int, default=50)
    ap.add_argument("--json", help="also write the full report as JSON")
    ap.add_argument("--scaling", action="store_true", help="instead: time the base build at several map sizes (see scaling())")
    args = ap.parse_args(argv)
    if args.scaling:
        rows = scaling(args.build, args.rocm, args.backend, args.stages.split(",") if args.stages else None,
                       repeats=args.repeats)
        print(format_scaling(rows))
        if args.json:
            with open(args.json, "w") as f:
                json.dump(rows, f, indent=1)
        return 0
    report = sweep(args.build, args.rocm, args.backend, args.stages.split(",") if args.stages else None,
                   args.variants, args.repeats)
    print(format_report(report))
    if args.json:
        with open(args.json, "w") as f:
            json.dump(report, f, indent=1)
    return 0 if all(r["mismatches_vs_base"] == 0 for e in report["stages"].values()
                    for r in e["variants"].values() if not r["ablation"] and r["exact"]) else 1


if __name__ == "__main__":
    sys.exit(main())
