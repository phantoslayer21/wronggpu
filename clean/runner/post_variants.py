# SPDX-License-Identifier: MIT
"""Variants of the spec 06 kernels: GPU bit-identity checks and timing at the in-game size (synthetic data).

    python clean/runner/post_variants.py --build build --rocm "C:\\Program Files\\AMD\\ROCm\\7.2" --rounds 3 [--json f]

- checks: every RECON2_VARIANTS kernel, post_recon3, post_recon5 and post_recon6 against post_recon (history,
  reduced history, display) and every FILTER_VARIANTS kernel against post_filter, on small synthetic cases: all must
  be bit-identical; the approximate variant post_recon3_h against the reference with the spec 09 FP16 measures; the
  SPEED APPROXIMATIONS (APPROXIMATIONS below: post_recon4_*, post_recon5_bil, post_recon6_bil, post_recon6_eb,
  post_recon6_eb_bil; for the whole-pipeline rule of spec 09 draft 2): single-frame PSNR of the display image and of
  the filtered output against the exact path (informative, synthetic data);
- timing (median ms over interleaved rounds) at R 1706x960 -> D 3840x2160 -> O 2560x1440: post_recon and every
  post_recon2 variant (normal and reset frame), every filter variant, and the fastest reconstruction + filter pair.
  Budget for reconstruction + filter together: 0.40 ms.
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
RECON_KERNELS = ([PS.KERNEL_RECON] + list(PS.RECON2_VARIANTS) + list(PS.RECON3_VARIANTS) + list(PS.RECON4_VARIANTS)
                 + list(PS.PREP_VARIANTS))   # post_recon5*, post_recon6*: pre-pass + reconstruction, timed together
# SPEED APPROXIMATIONS checked with PSNR numbers (whole-pipeline rule, spec 09 draft 2): name -> what is approximated.
APPROXIMATIONS = {
    **PS.RECON4_VARIANTS,
    "post_recon5_bil": "pre-pass + reconstruction, bilinear history",
    "post_recon6_bil": "post_recon6 with bilinear history (as post_recon4_bil)",
    "post_recon6_eb": "post_recon6 with the blend in the E domain (no Dec / E per pixel; reduced history exact formula)",
    "post_recon6_eb_bil": "post_recon6_eb + bilinear history",
}


def psnr(a, b, peak):
    m = float(np.mean((np.asarray(a, np.float64) - b) ** 2))
    return 99.0 if m == 0 else float(10 * np.log10(peak * peak / m))


def check(r, report):
    cases = [((40, 24), None, False), ((40, 24), None, True), ((26, 15), (59, 33), False), ((90, 40), None, False)]
    for i, (R, D, reset) in enumerate(cases):
        rng = np.random.default_rng(1100 + i)
        D = D or (None, None)
        x = P.random_inputs(rng, R[0], R[1], dw=D[0], dh=D[1], reset=reset)
        w = r.pack_weights("post", {"s": np.array(float(rng.uniform(0.2, 2.0)))})
        base = PS.run_plan(r, PS.prepare_post(r.dev, w, x))
        exact = {**{k: True for k in PS.RECON2_VARIANTS}, **PS.RECON3_VARIANTS,
                 **{k: v for k, v in PS.PREP_VARIANTS.items() if v}}
        ref = None
        for k, must_match in exact.items():
            got = PS.run_plan(r, PS.prepare_post(r.dev, w, x, k))
            row = {"kernel": k, "D": x["recon_size"], "reset": reset}
            if must_match:
                row["identical"] = {n: bool(np.array_equal(got[n], base[n])) for n in base}
            else:   # approximate variant: spec 09 FP16 measures against the reference, and difference to post_recon
                if ref is None:
                    ref = P.reconstruct(x, float(np.frombuffer(w, np.float32)[0]))
                row["identical"] = {"reduced_within_1_code": bool(np.max(np.abs(np.rint(got["reduced"] * 1023)
                                                                          - np.rint(ref["reduced"] * 1023))) <= 1)}
                for n in ("history", "display"):
                    d = f16_steps(got[n], ref[n])
                    big = (d > 1) & (np.abs(got[n].astype(np.float64) - ref[n]) > 1e-3)
                    row[n] = {"frac_over_1_step_and_1e-3_vs_reference": float(np.mean(big)),
                              "frac_diff_vs_post_recon": float(np.mean(f16_steps(got[n], base[n]) > 0)),
                              "nonfinite": int(np.sum(~np.isfinite(got[n])))}
                row["identical"]["spec09"] = all(row[n]["frac_over_1_step_and_1e-3_vs_reference"] <= 1e-4
                                                 and row[n]["nonfinite"] == 0 for n in ("history", "display"))
            report["checks"].append(row)
            print(json.dumps(row))
        O = PS.output_size_for(*x["recon_size"])
        fbase = PS.run_plan(r, PS.prepare_filter(r.dev, b"", {"display": base["display"], "exposure": x["exposure"],
                                                              "output_size": O}))["out"]
        for k, what in APPROXIMATIONS.items():   # speed approximations: informative numbers, no pass/fail here
            got = PS.run_plan(r, PS.prepare_post(r.dev, w, x, k))
            out = PS.run_plan(r, PS.prepare_filter(r.dev, b"", {"display": got["display"], "exposure": x["exposure"],
                                                               "output_size": O}))["out"]
            row = {"kernel": k, "approximation": what, "D": x["recon_size"], "reset": reset,
                   "psnr_display_vs_post_recon_db": psnr(got["display"], base["display"], 2.45),
                   "psnr_output_vs_two_kernel_db": psnr(out, fbase, float(fbase.max())),
                   "nonfinite": int(sum(np.sum(~np.isfinite(got[n])) for n in got)),
                   "identical": {"finite": all(bool(np.all(np.isfinite(got[n]))) for n in got)}}
            report["checks"].append(row)
            print(json.dumps(row))
        y = {"display": base["display"], "exposure": x["exposure"], "output_size": O}
        fbase = PS.run_plan(r, PS.prepare_filter(r.dev, b"", y))["out"]
        for k in PS.FILTER_VARIANTS:
            got = PS.run_plan(r, PS.prepare_filter(r.dev, b"", y, k))["out"]
            row = {"kernel": k, "O": O, "identical": {"out": bool(np.array_equal(got, fbase))}}
            report["checks"].append(row)
            print(json.dumps(row))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--build", required=True)
    ap.add_argument("--rocm", default=os.environ.get("ROCM_PATH", ""))
    ap.add_argument("--repeats", type=int, default=50)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--json", help="also write the results to this file")
    ap.add_argument("--no-check", action="store_true")
    ap.add_argument("--no-timing", action="store_true")
    ap.add_argument("--backend", default="hip", help="hip (GPU) or emu (host emulator, checks only)")
    args = ap.parse_args(argv)
    r = Runner(args.build, args.rocm, backend=args.backend)
    report = {"device": getattr(r.dev, "device_name", "?"), "budget_ms": BUDGET_MS, "checks": [], "timing_ms": {}}
    if not args.no_check:
        check(r, report)
    if not args.no_timing:
        rng = np.random.default_rng(1)
        x_norm, disp = big_inputs(rng, False)
        x_reset, _ = big_inputs(rng, True)
        w = r.pack_weights("post", {"s": np.array(0.9)})
        jobs = {}
        for k in RECON_KERNELS:
            jobs[f"{k}:normal"] = lambda k=k: PS.time_plans(r, [PS.prepare_post(r.dev, w, x_norm, k)], None,
                                                            args.repeats)
            jobs[f"{k}:reset"] = lambda k=k: PS.time_plans(r, [PS.prepare_post(r.dev, w, x_reset, k)], None,
                                                           args.repeats)
        jobs["post_prep (alone)"] = lambda: PS.time_prep(r, w, x_norm, args.repeats)
        for k in PS.FILTER_VARIANTS:
            jobs[k] = lambda k=k: PS.time_plans(r, [PS.prepare_filter(r.dev, b"", disp, k)], None, args.repeats)
        runs = {name: [] for name in jobs}
        for _ in range(args.rounds):
            for name, job in jobs.items():
                runs[name].append(job())
        med = {name: float(np.median(v)) for name, v in runs.items()}
        report["timing_ms"] = med
        for name, ms in med.items():
            print(f"{name:28s} {ms:.3f} ms")
        best_r = min(RECON_KERNELS, key=lambda k: med[f"{k}:normal"])
        best_f = min(PS.FILTER_VARIANTS, key=lambda k: med[k])
        total = med[f"{best_r}:normal"] + med[best_f]
        report["best"] = {"recon": best_r, "filter": best_f, "total_ms": total}
        print(f"fastest pair: {best_r} + {best_f} = {total:.3f} ms (budget {BUDGET_MS} ms)")
    if args.json:
        with open(args.json, "w") as f:
            json.dump(report, f, indent=1)


if __name__ == "__main__":
    main()
