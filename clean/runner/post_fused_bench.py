# SPDX-License-Identifier: MIT
"""Fused reconstruction + output filter (post_fused*) against the two-kernel path (post_recon + post_filter):
GPU timing at the in-game size and correctness checks, synthetic data only (no model data needed).

    python clean/runner/post_fused_bench.py --build build --rocm "C:\\Program Files\\AMD\\ROCm\\7.2" [--json out.json]

Builds build/post_w64.co (the same kernels in wave64 mode) if it is missing, then prints numbers only:

- timing (median ms) at R 1706x960 -> D 3840x2160 -> O 2560x1440 (1440p Quality), non-reset and reset frame: the
  two-kernel path and every fused variant (post_fused: 6-byte history, FP32 output; _o16: FP16 x 4 output; _h8:
  8-byte RGBA16F history as post_recon; post_fused3*: FUSED3_VARIANTS), each from post.co (wave32) and post_w64.co
  (wave64). Budget: 0.40 ms. Warm cache; runner/post_traffic.py also times them cold (the in-game number).
- checks on small synthetic cases: every fused variant and code object against the two-kernel path of the same code
  object (history, reduced history and output must be bit-identical; for _o16 the two-kernel output rounded to FP16;
  the post_fused3 approximations are marked "approximation" and not expected to be identical),
  and the fused output against clean/reference/post.py (spec 09 FP16 and float32 measures).
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))

import numpy as np  # noqa: E402

from clean.reference import post as P  # noqa: E402
from clean.runner import post as PS  # noqa: E402
from clean.runner.post_bench import big_inputs, f16_steps  # noqa: E402
from clean.runner.stage_runner import Runner  # noqa: E402

BUDGET_MS = 0.40


def two_kernel(r, w, x, code_object):
    """post_recon + post_filter of the given code object: {"history", "reduced", "out"}."""
    a = PS.run_plan(r, PS.prepare_post(r.dev, w, x), code_object)
    O = PS.output_size_for(*x["recon_size"])
    f = PS.run_plan(r, PS.prepare_filter(r.dev, b"", {"display": a["display"], "exposure": x["exposure"],
                                                      "output_size": O}), code_object)
    return {"history": a["history"], "reduced": a["reduced"], "out": f["out"]}


def float32_rule(got, ref):
    """Spec 09 float32 measures: fraction of values with relative error > 1e-3, maximum relative error."""
    rel = np.abs(got.astype(np.float64) - ref) / np.maximum(np.abs(ref), 1e-30)
    return {"frac_rel_over_1e-3": float(np.mean(rel > 1e-3)), "max_rel": float(rel.max()),
            "nonfinite": int(np.sum(~np.isfinite(got)))}


def fp16_rule(got, ref):
    d = f16_steps(got, ref)
    big = (d > 1) & (np.abs(got.astype(np.float64) - ref) > 1e-3)
    return {"frac_over_1_step_and_1e-3": float(np.mean(big)), "frac_diff": float(np.mean(d > 0)),
            "nonfinite": int(np.sum(~np.isfinite(got)))}


# kernel -> FP16 output (post_fused* and post_fused3*)
ALL_FUSED = {**{k: v[1] for k, v in PS.FUSED_VARIANTS.items()}, **{k: v[1] for k, v in PS.FUSED3_VARIANTS.items()}}


def check_small(r, code_objects, report):
    cases = [((40, 24), None, False), ((40, 24), None, True), ((26, 15), (59, 33), False), ((90, 40), None, False),
             ((70, 40), (158, 90), False)]
    for i, (R, D, reset) in enumerate(cases):
        rng = np.random.default_rng(700 + i)
        D = D or (None, None)
        x = P.random_inputs(rng, R[0], R[1], dw=D[0], dh=D[1], reset=reset)
        s = float(rng.uniform(0.2, 2.0))
        w = r.pack_weights("post", {"s": np.array(s)})
        ref = P.reconstruct(x, s)
        ref_out = P.output_filter(ref["display"], x["exposure"], PS.output_size_for(*x["recon_size"]))
        for co in code_objects:
            base = two_kernel(r, w, x, co)
            for k, o16 in ALL_FUSED.items():
                g = PS.run_fused(r, w, x, k, co)
                want = base["out"].astype(np.float16).astype(np.float32) if o16 else base["out"]
                row = {"code_object": co, "kernel": k, "R": R, "D": x["recon_size"], "reset": reset,
                       "identical_to_two_kernel": {"history": bool(np.array_equal(g["history"], base["history"])),
                                                   "reduced": bool(np.array_equal(g["reduced"], base["reduced"])),
                                                   "out": bool(np.array_equal(g["out"], want))},
                       "history_vs_reference": fp16_rule(g["history"], ref["history"]),
                       "out_vs_reference": float32_rule(g["out"], ref_out)}
                if k in PS.FUSED3_VARIANTS and PS.FUSED3_VARIANTS[k][2] is not True:
                    row["approximation"] = PS.FUSED3_VARIANTS[k][2]   # identity not expected
                report["checks"].append(row)
                print(json.dumps(row))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--build", required=True)
    ap.add_argument("--rocm", default=os.environ.get("ROCM_PATH", ""))
    ap.add_argument("--clang", help="clang for building post_w64.co (default: from --rocm)")
    ap.add_argument("--repeats", type=int, default=50)
    ap.add_argument("--json", help="also write the results to this file")
    ap.add_argument("--no-check", action="store_true", help="timing only")
    ap.add_argument("--no-timing", action="store_true", help="checks only")
    ap.add_argument("--no-w64", action="store_true", help="wave32 code object only")
    ap.add_argument("--backend", default="hip", help="hip (GPU) or emu (host emulator, checks only)")
    args = ap.parse_args(argv)
    code_objects = [PS.CODE_OBJECT]
    if not args.no_w64 and args.backend == "hip":
        if not os.path.exists(os.path.join(args.build, PS.CODE_OBJECT_W64)):
            PS.build_wave64(args.build, args.rocm, args.clang)
        code_objects.append(PS.CODE_OBJECT_W64)
    r = Runner(args.build, args.rocm, backend=args.backend)
    report = {"device": getattr(r.dev, "device_name", "?"), "budget_ms": BUDGET_MS, "timing_ms": {}, "checks": []}
    if not args.no_check:
        check_small(r, code_objects, report)
    if not args.no_timing:
        rng = np.random.default_rng(1)
        w = r.pack_weights("post", {"s": np.array(0.9)})
        for reset in (False, True):
            x, disp = big_inputs(rng, reset)
            x = dict(x, output_size=disp["output_size"])
            tag = "reset" if reset else "normal"
            for co in code_objects:
                key = f"{co}:two_kernel:{tag}"
                report["timing_ms"][key] = PS.time_two_kernel(r, w, x, disp, args.repeats, co)
                print(f"{key}: {report['timing_ms'][key]:.3f} ms")
                for k in ALL_FUSED:
                    if reset and k not in (PS.KERNEL_FUSED, "post_fused3_h6_o16"):
                        continue
                    key = f"{co}:{k}:{tag}"
                    report["timing_ms"][key] = PS.time_fused(r, w, x, k, args.repeats, co)
                    print(f"{key}: {report['timing_ms'][key]:.3f} ms (budget {BUDGET_MS} ms)")
    if args.json:
        with open(args.json, "w") as f:
            json.dump(report, f, indent=1)


if __name__ == "__main__":
    main()
