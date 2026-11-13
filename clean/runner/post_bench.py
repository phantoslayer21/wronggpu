# SPDX-License-Identifier: MIT
"""Stages post and filter (spec 06): GPU timing at the in-game size and a small correctness check against the NumPy
reference, with synthetic data only (no model data needed).

    python clean/runner/post_bench.py --build build --rocm "C:\\Program Files\\AMD\\ROCm\\7.2" [--json out.json]

Prints numbers only: per stage the median kernel time (ms) at R 1706x960 -> D 3840x2160 -> O 2560x1440 (1440p
Quality) for a normal and a reset frame, and for a few small synthetic cases the spec 09 measures against
clean/reference/post.py.
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))

import numpy as np  # noqa: E402

from clean.reference import post as P  # noqa: E402
from clean.runner.stage_runner import Runner  # noqa: E402
from clean.runner import post as PS  # noqa: E402

IN_GAME = {"R": (1706, 960), "O": (2560, 1440)}
TARGET_MS = {"post": 0.98, "filter": 0.23}


def f16_steps(a, b):
    ia = np.asarray(a, np.float32).astype(np.float16).view(np.int16).astype(np.int64)
    ib = np.asarray(b, np.float32).astype(np.float16).view(np.int16).astype(np.int64)
    ia = np.where(ia < 0, -(ia & 0x7FFF), ia)
    ib = np.where(ib < 0, -(ib & 0x7FFF), ib)
    return np.abs(ia - ib)


def big_inputs(rng, reset=False):
    """In-game-size synthetic inputs, built without the (slow, memory-hungry) reference generators."""
    (rw, rh), (ow, oh) = IN_GAME["R"], IN_GAME["O"]
    dw, dh = P.recon_size_for(ow), P.recon_size_for(oh)
    f16 = lambda a: a.astype(np.float16).astype(np.float32)  # noqa: E731
    sx = rng.uniform(0.3, 1.5, (rh, rw)).astype(np.float32)
    sy = rng.uniform(0.3, 1.5, (rh, rw)).astype(np.float32)
    rho = rng.uniform(-0.8, 0.8, (rh, rw)).astype(np.float32)
    mot = lambda: f16(rng.normal(0.0, 2.0, (rh, rw, 2)).astype(np.float32))  # noqa: E731
    x = {"color": rng.uniform(0, 4, (rh, rw, 3)).astype(np.float32), "exposure": 1.2, "exposure_prev": 1.1,
         "A": (rng.integers(0, 256, (rh, rw, 3)) / 255.0).astype(np.float32),
         "B": (rng.integers(0, 256, (rh, rw, 3)) / 255.0).astype(np.float32),
         "C": f16(np.stack([sx * sx, sy * sy, sx * sy * rho], -1)),
         "history": f16(rng.uniform(0, 1.5, (dh, dw, 3)).astype(np.float32)),
         "m_near": mot(), "m_center": mot(), "m_far": mot(), "jitter": np.array([0.31, -0.17]), "reset": reset,
         "recon_size": (dw, dh)}
    disp = {"display": f16(rng.uniform(0, 2.0, (dh, dw, 3)).astype(np.float32)), "exposure": 1.2,
            "output_size": (ow, oh)}
    return x, disp


def check_small(r, report):
    cases = [((40, 24), None, False), ((40, 24), None, True), ((26, 15), (59, 33), False),
             ((36, 20), (36, 20), False), ((70, 40), None, False)]
    for i, (R, D, reset) in enumerate(cases):
        rng = np.random.default_rng(900 + i)
        D = D or (None, None)
        x = P.random_inputs(rng, R[0], R[1], dw=D[0], dh=D[1], reset=reset)
        s = float(rng.uniform(0.2, 2.0))
        ref = P.reconstruct(x, s)
        got = r.run("post", r.pack_weights("post", {"s": np.array(s)}), x)
        row = {"stage": "post", "R": R, "D": x["recon_size"], "reset": reset}
        for k in ("history", "display"):
            d = f16_steps(got[k], ref[k])
            row[k] = {"nonfinite": int(np.sum(~np.isfinite(got[k]))), "frac_over_1_step": float(np.mean(d > 1)),
                      "frac_diff": float(np.mean(d > 0))}
        dc = np.abs(np.rint(got["reduced"] * 1023) - np.rint(ref["reduced"] * 1023))
        row["reduced"] = {"max_code_diff": int(dc.max(initial=0)), "frac_diff": float(np.mean(dc > 0))}
        report["checks"].append(row)
        print(json.dumps(row))
    for O in ((8, 6), (7, 5), (130, 37)):
        rng = np.random.default_rng(950 + O[0])
        disp = P.f16(rng.uniform(0, 3.0, (P.recon_size_for(O[1]), P.recon_size_for(O[0]), 3)))
        ref = P.output_filter(disp, 1.3, O)
        got = r.run("filter", b"", {"display": disp, "exposure": 1.3, "output_size": O})["out"]
        rel = np.abs(got.astype(np.float64) / ref - 1.0)
        row = {"stage": "filter", "O": O, "nonfinite": int(np.sum(~np.isfinite(got))),
               "max_rel_err": float(rel.max()), "mean_rel_err": float(rel.mean())}
        report["checks"].append(row)
        print(json.dumps(row))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--build", required=True)
    ap.add_argument("--rocm", default=os.environ.get("ROCM_PATH", ""))
    ap.add_argument("--repeats", type=int, default=50)
    ap.add_argument("--json", help="also write the results to this file")
    ap.add_argument("--no-check", action="store_true", help="timing only")
    ap.add_argument("--no-timing", action="store_true", help="checks only")
    ap.add_argument("--backend", default="hip", help="hip (GPU) or emu (host emulator, checks only)")
    args = ap.parse_args(argv)
    r = Runner(args.build, args.rocm, backend=args.backend)
    report = {"device": getattr(r.dev, "device_name", "?"), "timing_ms": {}, "checks": []}
    if not args.no_check:
        check_small(r, report)
    if args.no_timing:
        return
    rng = np.random.default_rng(1)
    w = r.pack_weights("post", {"s": np.array(0.9)})
    for reset in (False, True):
        x, disp = big_inputs(rng, reset)
        key = "post_reset" if reset else "post"
        report["timing_ms"][key] = r.time("post", w, x, repeats=args.repeats)
        print(f"{key}: {report['timing_ms'][key]:.3f} ms (target {TARGET_MS['post']} ms)")
    report["timing_ms"]["filter"] = r.time("filter", b"", disp, repeats=args.repeats)
    print(f"filter: {report['timing_ms']['filter']:.3f} ms (target {TARGET_MS['filter']} ms)")
    # Filter variants (kernel name -> output row pairs per lane); outputs must be identical to the default kernel.
    base = r.run("filter", b"", disp)["out"]
    for k, steps in PS.FILTER_VARIANTS.items():
        stage = f"filter:{k}"
        ms = r.time(stage, b"", disp, repeats=args.repeats)
        same = bool(np.array_equal(r.run(stage, b"", disp)["out"], base))
        report["timing_ms"][stage] = ms
        report.setdefault("filter_variants_identical", {})[k] = same
        print(f"{stage} (S = {steps}): {ms:.3f} ms, identical to default: {same}")
    if args.json:
        with open(args.json, "w") as f:
            json.dump(report, f, indent=1)


if __name__ == "__main__":
    main()
