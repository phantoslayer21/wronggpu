# SPDX-License-Identifier: MIT
"""Whole-stage timing of the persistent kernels against the launches they replace, with a bit-identity check of every
persistent-kernel build against its swin.hip reference build (variants.MEGA_REFERENCE: base, or vec16 for bmegahg16).
Synthetic data (timing does not depend on the values).
  --stage bneck: b0..b5 (C = 128, F/8): six block launches vs one swin_bneck6 launch
  --stage mid:   enc2 (block, merge), b0..b5, dec2 (expand, block) at F/4 and F/8: ten launches vs one swin_mid launch

    python clean/runner/build.py --rocm <ROCm> --out build --variants bmegahg,bmega,bmegahg16,vec16
    python clean/runner/bench_bneck.py --build build --rocm <ROCm> --stage mid [--groups 64,96,128] [--json mid.json]

Prints one line per configuration: median ms of the whole stage, and the number of output values that differ from
the reference build's launches (0 = bit-identical). "persistent" lines time one launch with the given number of
workgroups (default: the variant's resident capacity, variants.py). Numbers only; safe to paste into results/.
"""
import argparse
import json
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np  # noqa: E402

from clean.runner import variants as V  # noqa: E402
from clean.runner.stage_runner import Runner  # noqa: E402
from clean.runner.sweep import GAME_SIZES, stage_inputs, stage_params  # noqa: E402

# stage -> (geometry key of the persistent kernel, run method, time method, description)
STAGES = {"bneck": ("mega", "run_bneck", "time_bneck", "b0..b5 at {H}x{W}x128"),
          "mid": ("mid", "run_mid", "time_mid", "enc2 + merge, b0..b5, expand + dec2 at {H}x{W}x96 / F/8")}


def mismatches(a, b):
    return int(np.sum(~((a == b) | (np.isnan(a) & np.isnan(b)))))


def built_variants(build_dir):
    manifest = os.path.join(build_dir, V.MANIFEST)
    if not os.path.exists(manifest):
        return ["base"]
    with open(manifest) as f:
        return list(json.load(f))


def bench(build_dir, rocm_dir="", backend="hip", variants=None, groups=None, size=None, repeats=50, seed=0,
          parity=0, stage="bneck"):
    key, run_name, time_name, _ = STAGES[stage]
    H, W = size or GAME_SIZES[stage][:2]
    built = built_variants(build_dir)
    names = [n for n in built if n in V.V2_MEGA] if variants in (None, "all") else variants.split(",")
    missing = [n for n in names if n not in built]
    if missing:
        raise SystemExit(f"variants not built: {missing} (built: {built})")
    rng = np.random.default_rng(seed)
    params = stage_params(stage, rng)
    x, _ = stage_inputs(stage, H, W, rng)
    runners = {}

    def runner(n):
        if n not in runners:
            runners[n] = Runner(build_dir, rocm_dir, backend=backend, variant=n)
        return runners[n]

    def run(n, mega, g=None):
        return getattr(runner(n), run_name)(weights, x, parity, mega=mega, groups=g)

    def time(n, mega, g=None):
        return getattr(runner(n), time_name)(weights, x, parity, mega=mega, groups=g, repeats=repeats)

    weights = runner("base").pack_weights(stage, params)
    refs, rows = {}, []
    # the swin.hip builds the persistent builds must match, timed as separate launches (the current path)
    for ref in ["base"] + sorted({V.MEGA_REFERENCE.get(n, "base") for n in names} - {"base"}):
        if ref not in built:
            print(f"note: reference build {ref!r} not built; its persistent builds are compared with nothing")
            continue
        refs[ref] = run(ref, False)
        rows.append({"variant": ref, "mode": "launches", "groups": None, "reference": ref, "ms": time(ref, False),
                     "mismatches": 0})
    for n in names:
        ref = V.MEGA_REFERENCE.get(n, "base")
        geo = runner(n).geo.get(key)
        if geo is None:
            continue
        for g in (groups or [geo["groups"]]):
            out = run(n, True, g)
            rows.append({"variant": n, "mode": "persistent", "groups": int(g), "threads": geo["threads"],
                         "reference": ref, "ms": time(n, True, g),
                         "mismatches": mismatches(out, refs[ref]) if ref in refs else None})
    return {"device": runner("base").dev.device_name, "stage": stage, "size": [H, W], "parity": parity,
            "repeats": repeats, "rows": rows}


def format_report(rep):
    H, W = rep["size"]
    what = STAGES[rep["stage"]][3].format(H=H, W=W)
    L = [f"stage {rep['stage']} ({what}), parity {rep['parity']}, {rep['device']}, median of {rep['repeats']}"]
    for r in rep["rows"]:
        g = f"{r['groups']:4d} groups x {r['threads']} lanes" if r["mode"] == "persistent" else ""
        L.append(f"  {r['variant']:10s} {r['mode']:10s} {g:24s} {r['ms']:.3f} ms   mismatches vs {r['reference']}: "
                 f"{r['mismatches']}")
    return "\n".join(L)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--build", required=True, help="output directory of build.py")
    ap.add_argument("--rocm", default=os.environ.get("ROCM_PATH", ""))
    ap.add_argument("--backend", default="hip", choices=["hip", "emu"])
    ap.add_argument("--variants", help="comma-separated bmega* builds (default: every one built)")
    ap.add_argument("--groups", help="comma-separated workgroup counts for the megakernel (default: the variant's)")
    ap.add_argument("--stage", default="bneck", help="bneck, mid, or both (comma-separated)")
    ap.add_argument("--size", help="HxW of the stage's input map (default: the in-game size of sweep.GAME_SIZES)")
    ap.add_argument("--parity", type=int, default=0, choices=[0, 1])
    ap.add_argument("--repeats", type=int, default=50)
    ap.add_argument("--json", help="also write the report as JSON")
    args = ap.parse_args(argv)
    size = tuple(int(v) for v in args.size.lower().split("x")) if args.size else None
    groups = [int(g) for g in args.groups.split(",")] if args.groups else None
    reps = []
    for stage in args.stage.split(","):
        reps.append(bench(args.build, args.rocm, args.backend, args.variants, groups, size, args.repeats,
                          parity=args.parity, stage=stage))
        print(format_report(reps[-1]))
    if args.json:
        with open(args.json, "w") as f:
            json.dump(reps if len(reps) > 1 else reps[0], f, indent=1)


if __name__ == "__main__":
    main()
