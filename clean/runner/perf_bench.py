# SPDX-License-Identifier: MIT
"""Stages enc0, dec0 and exposure (performance milestone, clean/PERF.md): GPU timing of every build variant at the
in-game size, output comparison of the variants with the default build, and spec 09 measures against the NumPy
references on smaller synthetic cases. Synthetic data only (no model data needed).

    python clean/runner/build.py --rocm "C:\\Program Files\\AMD\\ROCm\\7.2" --out build --perf-variants
    python clean/runner/perf_bench.py --build build --rocm "C:\\Program Files\\AMD\\ROCm\\7.2" --json perf.json

Prints numbers only:
  timing   median kernel time (ms) of each variant at R 1706x960, F 1728x960, D 3840x2160 (1440p Quality),
           non-reset frame (enc0 also a reset frame), with the budget of results/2026-10-05-perf-targets.md;
  same     for each non-ablation variant, how many output values differ from the default build at that size
           (ablations are wrong on purpose and are not compared);
  checks   default build and v1 against the reference on small cases (spec 09 measures; values identical to the
           reference);
  fusion   dec1 + dec0 separate (expand, block, coef_decoder) vs fused (expand, swin_block_c64_add_dec0): times,
           per-kernel times, and values that differ (must be 0).
Options: --stages enc0,dec0,exposure,dec1dec0  --variants base,v1,...  --repeats N  --no-check  --no-timing  --backend emu
(emu: checks of the default build only, in the host emulator).
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))

import numpy as np  # noqa: E402

from clean.reference import decoder as Dc  # noqa: E402
from clean.reference import encoder as E  # noqa: E402
from clean.reference import exposure as X  # noqa: E402
from clean.reference import swin as SW  # noqa: E402
from clean.reference.fp8 import fp8_encode  # noqa: E402
from clean.runner import perf_variants as PV  # noqa: E402
from clean.runner.stage_runner import Runner  # noqa: E402

IN_GAME = {"R": (1706, 960), "F": (1728, 960), "D": (3840, 2160)}
BUDGET_MS = {"enc0": 0.20, "dec0": 0.12, "exposure": 0.03}       # results/2026-10-05-perf-targets.md
BEFORE_MS = {"enc0": 0.331, "dec0": 0.152, "exposure": 0.035}    # spec team, before this milestone


def stage_name(stage, variant):
    return stage if variant == "base" else f"{stage}@{variant}"


def in_game_inputs(stage, rng, reset=False):
    (rw, rh), (fw, fh), (dw, dh) = IN_GAME["R"], IN_GAME["F"], IN_GAME["D"]
    if stage == "enc0":
        return E.random_inputs(rng, rw=rw, rh=rh, fw=fw, fh=fh, dw=dw, dh=dh, reset=reset, frame=3), E.random_params(rng)
    if stage == "dec0":
        return Dc.random_inputs(rng, rw=rw, rh=rh, fw=fw, fh=fh, dw=dw, dh=dh, reset=reset), Dc.random_params(rng)
    color = rng.lognormal(-1.0, 1.5, size=(rh, rw, 3)).astype(np.float32)
    return {"color": color, "pre_exposure": 1.0, "alpha": 0.18}, None


def weights(r, stage, variant, params):
    return b"" if params is None else r.pack_weights(stage_name(stage, variant), params)


def n_diff(a, b):
    """Values that differ (NaN equals NaN) per output."""
    out = {}
    for k in a:
        x, y = np.asarray(a[k]), np.asarray(b[k])
        out[k] = int(np.sum(~((x == y) | (np.isnan(x) & np.isnan(y)))))
    return out


def codes_diff(got, ref, scale):
    return np.abs(np.rint(np.asarray(got, np.float64) * scale) - np.rint(np.asarray(ref, np.float64) * scale))


def fp8_steps(a, b):
    ca, cb = fp8_encode(np.asarray(a, np.float32)).astype(int), fp8_encode(np.asarray(b, np.float32)).astype(int)
    oa, ob = np.where(ca & 0x80, -(ca & 0x7F), ca), np.where(cb & 0x80, -(cb & 0x7F), cb)
    return np.abs(oa - ob)


def f16_steps(a, b):
    ia = np.asarray(a, np.float32).astype(np.float16).view(np.int16).astype(np.int64)
    ib = np.asarray(b, np.float32).astype(np.float16).view(np.int16).astype(np.int64)
    return np.abs(np.where(ia < 0, -(ia & 0x7FFF), ia) - np.where(ib < 0, -(ib & 0x7FFF), ib))


def check_cases(stage):
    """Small synthetic cases: (x, params) pairs."""
    cases = []
    for i in range(6):
        rng = np.random.default_rng(4200 + i)
        R = [(128, 72), (96, 64), (130, 70), (64, 40), (128, 72), (111, 57)][i]
        reset = i == 3
        if stage == "enc0":
            x = E.random_inputs(rng, rw=R[0], rh=R[1], reset=reset, frame=i)
            if i == 4:
                x["exposure_prev"] = -1.0
            cases.append((x, E.random_params(rng, w_scale=[0.08, 0.3][i % 2])))
        elif stage == "dec0":
            cases.append((Dc.random_inputs(rng, rw=R[0], rh=R[1], reset=reset, motion_scale=[3.0, 20.0][i % 2]),
                          Dc.random_params(rng, w_scale=[0.25, 1.0][i % 2])))
        else:
            color = rng.lognormal(-1.0, 1.5, size=(R[1] * 4, R[0] * 4, 3)).astype(np.float32)
            if i == 5:
                color.reshape(-1, 3)[::97] = np.nan
            cases.append(({"color": color, "pre_exposure": 1.0, "alpha": 0.18}, None))
    return cases


def check(r, stage, variant, report):
    rows = []
    for x, p in check_cases(stage):
        got = r.run(stage_name(stage, variant), weights(r, stage, variant, p), x, None)
        if stage == "enc0":
            ref = E.encode(x, p)
            g, f = got["features"], ref["features"]
            d = fp8_steps(g, f)
            row = {"features": {"n": int(g.size), "differ": int(np.sum(d > 0)), "over_1_step": int(np.sum(d > 1)),
                                "nonfinite": int(np.sum(~np.isfinite(g))),
                                "mean_abs_err_rel_rms": float(np.mean(np.abs(g - f)) / np.sqrt(np.mean(f * f)))}}
            for k in ("m_near", "m_far", "m_center"):
                fin = np.isfinite(ref[k])
                row[k] = {"differ": int(np.sum(f16_steps(got[k][fin], ref[k][fin]) > 0))}
        elif stage == "dec0":
            ref = Dc.decode(x, p)
            row = {k: {"n": int(ref[k].size), "differ": int(np.sum(codes_diff(got[k], ref[k], s) > 0)),
                       "over_1_code": int(np.sum(codes_diff(got[k], ref[k], s) > 1))}
                   for k, s in (("A", 255), ("B", 255), ("latent", 127))}
            u = f16_steps(got["C"], ref["C"])
            row["C"] = {"n": int(u.size), "differ": int(np.sum(u > 0)), "over_1_step": int(np.sum(u > 1))}
        else:
            ref = X.exposure(x["color"], x["pre_exposure"], x["alpha"])
            row = {"e": float(got["e"]), "ref": float(ref), "f16_steps": int(f16_steps(got["e"], ref))}
        rows.append(row)
    report["checks"][stage_name(stage, variant)] = rows
    total = {}
    for row in rows:
        for k, v in row.items():
            if isinstance(v, dict):
                for m, n in v.items():
                    if isinstance(n, int):
                        total.setdefault(k, {}).setdefault(m, 0)
                        total[k][m] += n
    print(f"check {stage_name(stage, variant)} vs reference ({len(rows)} cases): "
          + (json.dumps(total) if total else json.dumps([row["f16_steps"] for row in rows])))


def bench_fusion(r, report, repeats, check=True, built=("fused",)):
    """dec1 + dec0 at the in-game size: expand + swin_block_c64_add[_nc] + coef_decoder vs expand + the fused kernel
    (kernels/dec1_dec0.hip). Times both (and every kernel), and counts output values that differ (must be 0)."""
    (rw, rh), (fw, fh), (dw, dh) = IN_GAME["R"], IN_GAME["F"], IN_GAME["D"]
    H, W = fh // 2, fw // 2
    rng = np.random.default_rng(91)
    y, skip = SW.random_inputs("dec1", H, W, rng)
    xd = Dc.random_inputs(rng, rw=rw, rh=rh, fw=fw, fh=fh, dw=dw, dh=dh)
    x = {k: v for k, v in xd.items() if k != "features"}
    x.update(y=y, skip=skip, offset=(2, 0))
    w = r.pack_weights("dec1dec0", {"dec1": SW.random_params("dec1", rng), "dec0": Dc.random_params(rng)})
    for name in ["dec1dec0"] + [f"dec1dec0@{v}" for v in PV.FUSION if v in built]:
        ms = r.time(name, w, x, repeats=repeats)
        k = r.time_kernels(name, w, x, repeats=repeats)
        report["timing_ms"][name] = ms
        report["timing_ms"][name + ":kernels"] = k
        print(f"{name}: {ms:.4f} ms (" + ", ".join(f"{n.split(':')[-1]} {t:.4f}" for n, t in k.items()) + ")",
              flush=True)
    for v in [v for v in PV.FUSION if v in built]:
        print(f"{v} saves {report['timing_ms']['dec1dec0'] - report['timing_ms']['dec1dec0@' + v]:+.4f} ms")
    if check:
        base = r.run("dec1dec0", w, x, None)
        for v in [v for v in PV.FUSION if v in built]:
            d = n_diff(r.run(f"dec1dec0@{v}", w, x, None), base)
            report["differ_from_base"][f"dec1dec0@{v}"] = d
            print(f"dec1dec0@{v} values differing from the separate kernels: {d}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--build", required=True)
    ap.add_argument("--rocm", default=os.environ.get("ROCM_PATH", ""))
    ap.add_argument("--repeats", type=int, default=100)
    ap.add_argument("--stages", default="enc0,dec0,exposure,dec1dec0",
                    help="comma-separated: enc0, dec0, exposure, dec1dec0 (dec1 + dec0 separate vs fused)")
    ap.add_argument("--variants", default="all", help="comma-separated variant names, or 'all'")
    ap.add_argument("--json", help="also write the results to this file")
    ap.add_argument("--no-check", action="store_true", help="timing and variant comparison only")
    ap.add_argument("--no-timing", action="store_true", help="checks only")
    ap.add_argument("--backend", default="hip", help="hip (GPU) or emu (host emulator: checks of 'base' only)")
    args = ap.parse_args(argv)
    r = Runner(args.build, args.rocm, backend=args.backend)
    report = {"device": getattr(r.dev, "device_name", "?"), "timing_ms": {}, "differ_from_base": {}, "checks": {}}
    built = {}
    manifest = os.path.join(args.build, PV.MANIFEST)
    if os.path.exists(manifest):
        with open(manifest) as f:
            built = json.load(f)
    for stage in args.stages.split(","):
        if stage == "dec1dec0":
            if args.backend == "hip" and not args.no_timing:
                bench_fusion(r, report, args.repeats, check=not args.no_check,
                             built=tuple(built.get("dec1dec0", {"fused": None})))
            continue
        names = list(PV.VARIANTS[stage]) if args.variants == "all" else \
            [v for v in args.variants.split(",") if v in PV.VARIANTS[stage]]
        if args.backend != "hip":
            names = [v for v in names if v == "base"]
        else:
            names = [v for v in names if v == "base" or v in built.get(stage, {})]
        if not args.no_check:
            for v in names:
                if v in ("base", "v1"):
                    check(r, stage, v, report)
        if args.no_timing:
            continue
        rng = np.random.default_rng(77)
        x, p = in_game_inputs(stage, rng)
        base_out = r.run(stage, weights(r, stage, "base", p), x, None)
        times = {}
        for v in names:
            name = stage_name(stage, v)
            w = weights(r, stage, v, p)
            times[v] = r.time(name, w, x, repeats=args.repeats)
            report["timing_ms"][name] = times[v]
            same = ""
            if v != "base" and not PV.is_ablation(stage, v):
                if any(PV.input_f16(stage, v)):  # compared with the default build on FP16-rounded inputs
                    xr = PV.round_inputs(x, *PV.input_f16(stage, v))
                    d = n_diff(r.run(name, w, x, None), r.run(stage, weights(r, stage, "base", p), xr, None))
                else:
                    d = n_diff(r.run(name, w, x, None), base_out)
                report["differ_from_base"][name] = d
                same = f", values differing from base: {d}"
            ab = " (ablation, output wrong on purpose)" if PV.is_ablation(stage, v) else ""
            rel = f", {times[v] - times['base']:+.4f} ms vs base" if v != "base" else \
                f" (budget {BUDGET_MS[stage]} ms; before this milestone {BEFORE_MS[stage]} ms)"
            print(f"{name}: {times[v]:.4f} ms{rel}{ab}{same}", flush=True)
        if stage == "enc0":
            xr, pr = in_game_inputs(stage, np.random.default_rng(78), reset=True)
            report["timing_ms"]["enc0_reset"] = r.time("enc0", weights(r, stage, "base", pr), xr, repeats=args.repeats)
            print(f"enc0 reset frame: {report['timing_ms']['enc0_reset']:.4f} ms")
        if stage == "exposure":
            for v in [v for v in names if v in ("base", "v1")]:
                k = r.time_kernels(stage_name(stage, v), b"", x, repeats=args.repeats)
                report["timing_ms"][f"{stage_name(stage, v)}:kernels"] = k
                print(f"{stage_name(stage, v)} per kernel: " + ", ".join(f"{n} {t:.4f} ms" for n, t in k.items()))
    if args.json:
        with open(args.json, "w") as f:
            json.dump(report, f, indent=1)


if __name__ == "__main__":
    main()
