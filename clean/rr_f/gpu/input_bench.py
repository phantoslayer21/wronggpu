# SPDX-License-Identifier: MIT
"""GPU time of the spec 52 input-stage kernels at a render size (default 1440p Quality: render 1706 x 960 of a
2560 x 1440 output), synthetic textures, one host call under the GPU lock.

    python -m clean.rr_f.gpu.input_bench [--render 1706x960] [--output 2560x1440] [--smv Hd|SV|none] [--reps 20]"""
import argparse
import os
import re

import numpy as np

from . import build, input_stage as GI
from ..tests import synth_input as SI


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--render", default="1706x960")
    ap.add_argument("--output", default="2560x1440")
    ap.add_argument("--smv", default="Hd", choices=("Hd", "SV", "none"))
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--steps", default="E,X")
    ap.add_argument("--fast", default="", help="comma list of RR_IN_FAST timing variants (1 hw log2/exp2, 4 hw sqrt, "
                    "8 no fma zero check, 16 no fences)")
    ap.add_argument("--skip", default="", help="comma list of RR_IN_SKIP timing variants (1 recon, 2 history samples, "
                    "4 hidden/md, 8 guides, 16 quantisation)")
    ap.add_argument("--defs", default="", help="comma list of features-kernel define sets (K=V+K=V), timing only")
    a = ap.parse_args(argv)
    if "WG_RR_BUILD" not in os.environ:
        ap.error("WG_RR_BUILD must name a private build directory")
    Wr, Hr = (int(v) for v in a.render.split("x"))
    Wo, Ho = (int(v) for v in a.output.split("x"))
    rng = np.random.default_rng(52)
    frame, params, S, IN0, consts, wc = SI.make_case(rng, Wr=Wr, Hr=Hr, Wo=Wo, Ho=Ho,
                                                    smv=None if a.smv == "none" else a.smv,
                                                    guides=["B", "G1", "G2", "Q"])
    ar = build.Arena()
    kern, disp = {}, []
    for step in a.steps.split(","):
        call = GI.InputCall(ar, frame, params, S, IN0, wc, debug=False)
        k = GI.kernels(step)
        kern["input-features"] = k["input-features"]
        kern["input-replay"] = k["input-replay"]
        kern["input-decode"] = k["input-decode"]
        kern["input-clear"] = k["input-clear"]
        kern[f"input-mlp-{step}"] = k["input-mlp"]
        disp += call.dispatches(km=f"input-mlp-{step}")
    for fv in [int(v) for v in a.fast.split(",") if v]:
        call = GI.InputCall(ar, frame, params, S, IN0, wc, debug=False)
        kern[f"input-features-fast{fv}"] = (build.compile_kernel("rr_input.hlsl", "rr_input_features",
                                                                 {"WG_BLOCK": GI.FEAT_BLOCK, "RR_IN_FAST": fv, "RR_IN_EXACT": 0, "RR_IN_LDST": 1},
                                                                 tag=f"rr_input_features_fast{fv}"), 1)
        disp += call.dispatches(kf=f"input-features-fast{fv}", km="input-mlp-E")[:-2]
    for sv in [int(v) for v in a.skip.split(",") if v]:
        call = GI.InputCall(ar, frame, params, S, IN0, wc, debug=False)
        kern[f"input-features-skip{sv}"] = (build.compile_kernel("rr_input.hlsl", "rr_input_features",
                                                                 {"WG_BLOCK": GI.FEAT_BLOCK, "RR_IN_SKIP": sv, "RR_IN_EXACT": 0, "RR_IN_LDST": 1},
                                                                 tag=f"rr_input_features_skip{sv}"), 1)
        disp += call.dispatches(kf=f"input-features-skip{sv}", km="input-mlp-E")[:-2]
    for i, dv in enumerate(v for v in a.defs.split(",") if v):   # e.g. RR_IN_FAST=29+RR_IN_HCERT=0
        call = GI.InputCall(ar, frame, params, S, IN0, wc, debug=False)
        d = {"WG_BLOCK": GI.FEAT_BLOCK, "RR_IN_EXACT": 0, "RR_IN_LDST": 1}
        d.update({kv.split("=")[0]: int(kv.split("=")[1]) for kv in dv.split("+")})
        kern[f"input-features-d{i}"] = (build.compile_kernel("rr_input.hlsl", "rr_input_features", d,
                                                             tag=f"rr_input_features_d{i}"), 1)
        disp += call.dispatches(kf=f"input-features-d{i}", km="input-mlp-E")[:-2]
    img, log = build.run(ar, kern, disp, reps=a.reps, workdir=os.path.join(build.BUILD, "input-bench"),
                         require_double=True)
    for line in log.splitlines():
        if line.startswith("time"):
            print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


def micro(argv=None):
    """Time per evaluation of the scalar helpers (fast pass build): log2, exp2, hw log2, sqrt, rcp."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--lanes", type=int, default=1 << 20)
    ap.add_argument("--iters", type=int, default=64)
    a = ap.parse_args(argv)
    ar = build.Arena()
    kern = {"mb": (build.compile_kernel("rr_input.hlsl", "rr_input_mbench", {"WG_BLOCK": 256, "RR_IN_EXACT": 0, "RR_IN_LDST": 1},
                                        tag="rr_input_mbench"), 1)}
    plog, pexp = GI.ff_offsets(ar)
    disp = []
    for op in range(5):
        out = ar.reserve(a.lanes * 4)
        hdr = ar.add(np.array([op, a.iters, out, plog, pexp], np.uint32))
        disp.append(("mb", a.lanes // 256, [hdr]))
    img, log = build.run(ar, kern, disp, reps=10, workdir=os.path.join(build.BUILD, "input-micro"), require_double=True)
    names = ["log2_cr", "exp2_cr", "hw log2", "sqrt", "rcp"]
    for line in log.splitlines():
        if line.startswith("time"):
            parts = line.split()
            i, best = int(parts[1]), float(parts[4])
            print(f"{names[i]:8s} {best:8.3f} ms  {best * 1e6 / (a.lanes * a.iters):.4f} ns/eval")
    return 0
