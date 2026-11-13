# SPDX-License-Identifier: MIT
"""Whole-frame A/B timing of kernel choices (spec 10 section 9 measure: Pipeline.time_frame), interleaved.

Each configuration is a set of kernel choices as for Pipeline(kernels=...): "base" (the defaults) or comma-separated
key=value pairs with keys swin, enc0, dec0, exposure, post, filter. Every configuration gets its own Pipeline (same
model, same frame); the rounds visit them in turn, so drifts of clock or temperature hit all of them alike.

    python clean\\runner\\build.py --rocm "C:\\Program Files\\AMD\\ROCm\\7.2" --out build --variants all --perf-variants
    python clean\\runner\\frame_ab.py --build build --rocm "C:\\Program Files\\AMD\\ROCm\\7.2" --model <model_dir> ^
        --configs base swin=geluexact swin=gelu16 swin=gelu725 --json gelu_ab.json

Input: --frame <file.npz> (keys color [rh, rw, 3], depth [rh, rw], motion [rh, rw, 2], mv_scale [2], jitter [2]; optional
exposure, pre_exposure), else a synthetic frame of --render size. Prints, per configuration, the median of the round
medians (ms) and the difference to the first configuration (numbers only).
"""
import argparse
import json
import os
import sys

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from clean.runner.pipeline import KERNEL_KEYS, Pipeline  # noqa: E402


def parse_config(text):
    if text == "base":
        return {}
    out = {}
    for kv in text.split(","):
        k, _, v = kv.partition("=")
        if k not in KERNEL_KEYS and k != "swin" and not k.startswith("swin."):
            raise SystemExit(f"unknown key {k!r} in {text!r}; known: swin, swin.<stage>, {', '.join(KERNEL_KEYS)}")
        out[k] = v
    return out


def synthetic_frame(rw, rh, seed=1):
    rng = np.random.default_rng(seed)
    return {"color": rng.uniform(0.0, 2.0, (rh, rw, 3)).astype(np.float32),
            "depth": rng.uniform(0.0, 1.0, (rh, rw)).astype(np.float32),
            "motion": rng.normal(0.0, 2.0, (rh, rw, 2)).astype(np.float32),
            "mv_scale": np.array([1.0, 1.0], np.float32), "jitter": np.array([0.25, -0.125], np.float32),
            "exposure": 1.0, "pre_exposure": 1.0}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--build", required=True)
    ap.add_argument("--rocm", default="")
    ap.add_argument("--model", required=True, help="model folder (model.wgcm or raw/)")
    ap.add_argument("--configs", nargs="+", default=["base"])
    ap.add_argument("--frame", help="npz with one frame's inputs (else synthetic)")
    ap.add_argument("--render", default="1706x960")
    ap.add_argument("--output", default="2560x1440")
    ap.add_argument("--rounds", type=int, default=8)
    ap.add_argument("--repeats", type=int, default=30)
    ap.add_argument("--no-auto-exposure", action="store_true")
    ap.add_argument("--depth-inverted", action="store_true")
    ap.add_argument("--backend", default="hip", help="hip, or emu (host emulator, for checks only)")
    ap.add_argument("--json")
    a = ap.parse_args(argv)

    rw, rh = (int(v) for v in a.render.lower().split("x"))
    ow, oh = (int(v) for v in a.output.lower().split("x"))
    x = dict(np.load(a.frame)) if a.frame else synthetic_frame(rw, rh)
    rh, rw = x["color"].shape[:2]
    flags = {"depth_inverted": a.depth_inverted, "auto_exposure": not a.no_auto_exposure}
    pipes = []
    for c in a.configs:
        p = Pipeline(a.build, a.rocm, a.model, (rw, rh), (ow, oh), flags, kernels=parse_config(c),
                     check_packing=False, backend=a.backend)
        p.frame(x)  # first frame (reset)
        pipes.append((c, p))
        print(f"{c}: {p.kernels()}", flush=True)
    times = {c: [] for c, _ in pipes}
    for r in range(a.rounds):
        for c, p in pipes:
            times[c].append(p.time_frame(x, repeats=a.repeats))
    res = {c: float(np.median(t)) for c, t in times.items()}
    first = a.configs[0]
    print(f"\nwhole frame, R {rw}x{rh}, O {ow}x{oh}, {a.rounds} rounds x {a.repeats} (median of round medians)")
    for c in a.configs:
        print(f"  {c:40s} {res[c]:.3f} ms   {res[c] - res[first]:+.3f} ms vs {first}")
    if a.json:
        with open(a.json, "w") as f:
            json.dump({"render": [rw, rh], "output": [ow, oh], "median_ms": res, "rounds_ms": times,
                       "kernels": {c: p.kernels() for c, p in pipes}}, f, indent=1)


if __name__ == "__main__":
    main()
