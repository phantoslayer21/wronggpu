# SPDX-License-Identifier: MIT
"""Model K timing harness (spec 40 task K4): the whole eleven-stage network with random parameters at the base sizes
of spec 44 for 1080p, 1440p and 4K output, per-stage and per-kernel GPU times. Numbers only, no real data.

    python clean/k/runner/k_bench.py --build build --rocm "C:\\Program Files\\AMD\\ROCm\\7.2" --json k_bench.json
    python clean/k/runner/k_bench.py --build build --rocm <ROCm> --variants fused g2 --kernels    # + per-kernel times
    python clean/k/runner/k_bench.py --build build --rocm <ROCm> --check 512x288                   # GPU vs NumPy
    python clean/k/runner/k_bench.py --build build --rocm <ROCm> --frames --all-flags              # whole frames
    python clean/k/runner/k_bench.py --build <emu dir> --backend emu --check 256x256 --sizes       # emulator

--sizes: output sizes (1080p, 1440p, 4k or WxH); the base map is spec 44's rule with the factor --s (default 1).
--base: the sizes are base map sizes (e.g. small odd ones for the border rule) instead of output sizes.
--check WxH: runs the network once on a random input at that OUTPUT size and compares every stage's block output
(and the head output) of the chained network with the chained NumPy reference (clean/k/reference/network.py):
errors relative to the reference RMS, pass/fail by spec 49's whole-network rule (mean <= 8e-3, max <= 7e-2). Spec 49
draft 3: windows whose block input has a token at the FP16 overflow threshold of the norm are excluded from the
measures and counted; a run with any of them is reported as a "threshold run" and does not count.
Build first: clean/runner/build.py (builds model K too) or clean/k/runner/k_build.py.
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402

from clean.k.reference import network as N  # noqa: E402
from clean.k.reference.layout import (DEFAULT_S, FUSED_IO, STAGES, base_size, level_size, stage_offsets,  # noqa: E402
                                      threshold_windows, window_flags_to_map, window_grid)
from clean.k.runner import k_launch as KL  # noqa: E402
from clean.k.runner.k_net import FUSED_VARIANTS, KNet, stage_info  # noqa: E402

# Spec 49 draft 4, whole network: every output of the chained network vs the chained reference (same network input),
# relative to the reference RMS. d0 head: mean 8e-3, max 7e-2. Intermediate outputs (block outputs and merged maps):
# mean 1.2e-2; max 0.12 at levels 0-2 (e0, e1, e2, d2, d1 and the merged maps of e0-e2), 0.25 at levels 3-5 (e3, e4, m,
# d4, d3 and the merged maps of e3, e4).
WHOLE_MEAN, WHOLE_MAX = 8e-3, 7e-2  # the d0 head
MID_MEAN, MID_MAX_LOW, MID_MAX_HIGH = 1.2e-2, 0.12, 0.25


def whole_limits(stage, key):
    """(mean, max) limits of spec 49 draft 4 for output `key` ("block", "merged", "head") of `stage`."""
    if key == "head":
        return WHOLE_MEAN, WHOLE_MAX
    level = next(s["level"] for s in STAGES if s["name"] == stage)
    return MID_MEAN, (MID_MAX_LOW if level <= 2 else MID_MAX_HIGH)
NAMED = {"1080p": (1920, 1080), "1440p": (2560, 1440), "4k": (3840, 2160)}


class KFuncs:
    """Kernel lookup: bare names through KERNEL_CODE_OBJECTS, "<code object>:<kernel>" for variants."""

    def __init__(self, dev, build_dir, backend):
        self.dev, self.build_dir, self.backend = dev, build_dir, backend
        self.modules, self.funcs = {}, {}

    def __call__(self, name):
        if name not in self.funcs:
            co, kernel = name.split(":", 1) if ":" in name else (KL.KERNEL_CODE_OBJECTS[name], name)
            if self.backend == "emu":
                mod = None
            else:
                if co not in self.modules:
                    with open(os.path.join(self.build_dir, co), "rb") as f:
                        self.modules[co] = self.dev.load_module(f.read())
                mod = self.modules[co]
            self.funcs[name] = self.dev.get_function(mod, kernel)
        return self.funcs[name]


def parse_size(s):
    if s.lower() in NAMED:
        return NAMED[s.lower()]
    w, h = s.lower().split("x")
    return int(w), int(h)


def open_device(backend, build_dir, rocm):
    if backend == "emu":
        from clean.runner.emu import Emu
        lib = os.path.join(build_dir, "wg_emu_k.so")
        return Emu(lib if os.path.exists(lib) else os.path.join(build_dir, "wg_emu.so"))
    from clean.runner.hip import Hip
    return Hip(rocm)


def check(dev, funcs, out_size, s, variant, f, seed=0, base=None, params=None):
    wb, hb = base or base_size(*out_size, s)
    params = params or N.random_network_params(seed)
    r = check_input(dev, funcs, N.random_input(wb, hb, seed + 1), params, f, variant)
    r["output_size"] = list(out_size) if out_size else None
    return r


def check_input(dev, funcs, x0, params, f, variant):
    """The chained network on the GPU vs the chained reference for the network input x0 [Hb, Wb, 16] at frame f."""
    hb, wb = x0.shape[:2]
    net = KNet(dev, funcs, wb, hb, params, variant)
    try:
        net.upload_input(x0)
        ref = N.run_network(x0, params, f, debug=True)
        off = stage_offsets(f, net.offset_table)
        res = {}
        # stage by stage: the harness overwrites A / B, so run the stages one at a time and read their results
        plan = net.stage_plan(f)
        for st in STAGES:
            name = st["name"]
            net._launch(plan[name])
            dev.sync()
            lw, lh = level_size(wb, hb, st["level"])
            dst, shape = net.block_output(name)
            got = {"block": net.download(dst, shape)} if dst else {}
            if st["kind"] == "enc":  # the merged map (spec 49 draft 4 measures it too)
                mname = FUSED_IO[name]["merged"] if variant in FUSED_VARIANTS else "A"
                mw, mh = level_size(wb, hb, st["level"] + 1)
                got["merged"] = net.download(mname, (mh, mw, ref[name]["merged"].shape[-1]))
            if st.get("head"):
                got["head"] = net.output()
            # spec 49 draft 3: threshold windows (in the reference's block inputs) are reported separately
            thr = threshold_windows(ref[name]["windows"])
            nwx, nwy = window_grid(lw, lh, *off[name])
            excl = window_flags_to_map(thr.reshape(nwy, nwx), lw, lh, *off[name])
            for k, g in got.items():
                r = ref[name][k].astype(np.float64)
                rms = float(np.sqrt(np.mean(r ** 2)))
                d = np.abs(g.astype(np.float64) - r)
                keep = ~excl if k == "block" or k == "head" else np.ones(d.shape[:2], bool)
                dk = d[keep] if np.any(keep) else d.reshape(-1, d.shape[-1])[:0]
                res[f"{name}.{k}"] = {"nonfinite": int(np.count_nonzero(~np.isfinite(g))),
                                      "mean_rel": float(np.nanmean(dk) / rms) if dk.size else 0.0,
                                      "max_rel": float(np.nanmax(dk) / rms) if dk.size else 0.0,
                                      "rms": rms, "threshold_windows": int(np.count_nonzero(thr)),
                                      "mean_rel_all": float(np.nanmean(d) / rms),
                                      "max_rel_all": float(np.nanmax(d) / rms)}
        threshold_run = any(v["threshold_windows"] for v in res.values())
        for key, v in res.items():
            v["limit_mean"], v["limit_max"] = whole_limits(*key.split("."))
        ok = all(v["nonfinite"] == 0 and v["mean_rel"] <= v["limit_mean"] and v["max_rel"] <= v["limit_max"]
                 for v in res.values())
        # spec 49 draft 3: a run with any threshold window is reported separately and does not count
        return {"output_size": None, "base": [wb, hb], "f": f, "variant": variant,
                "stages": res, "threshold_run": threshold_run, "pass": None if threshold_run else ok}
    finally:
        net.release()


def bench(dev, funcs, out_size, s, variant, f, repeats, kernels, seed=0, base=None, params=None):
    wb, hb = base or base_size(*out_size, s)
    net = KNet(dev, funcs, wb, hb, params or N.random_network_params(seed), variant)
    try:
        net.upload_input(N.random_input(wb, hb, seed + 1))
        times = net.time_stages(f, repeats)
        info = stage_info(wb, hb, net.offset_table, f)
        rows = {}
        for st in STAGES:
            n = st["name"]
            rows[n] = dict(info[n], ms=times[n], tflops=info[n]["gflop"] / times[n] if times[n] > 0 else None)
        out = {"output_size": list(out_size), "base": [wb, hb], "s": s, "f": f, "variant": variant,
               "total_ms": times["total"], "gflop": sum(r["gflop"] for r in rows.values()),
               "memory_mib": net.memory_bytes() / 2 ** 20, "stages": rows}
        if kernels:
            out["kernels"] = {n: [[k, ms] for k, ms in v] for n, v in net.time_kernels(f, repeats).items()}
        return out
    finally:
        net.release()


def synthetic_frame(rng, out_size, render_size, flags):
    """Random frame inputs for timing (no real data)."""
    rw, rh = render_size
    W, H = out_size
    mshape = (rh, rw, 2) if flags["mv_lowres"] else (H, W, 2)
    return dict(color=(rng.random((rh, rw, 3)) * 2).astype(np.float32),
                motion=(rng.standard_normal(mshape) * 2).astype(np.float32),
                depth=rng.random((rh, rw)).astype(np.float32), render_size=(rw, rh), jitter=(0.25, -0.25),
                mv_scale=(1.0, 1.0), pre_exposure=1.0, reset=False, E_cur=1.0, E_aux=1.0, e1=1.0, e2=1.0)


def radical_inverse(n, b):
    r, f = 0.0, 1.0 / b
    while n:
        n, d = divmod(n, b)
        r += d * f
        f /= b
    return r


SEQ_FLAGS = dict(hdr=True, mv_lowres=True, depth_inverted=True)
SEQ_OUTPUT, SEQ_RENDER = (512, 512), (256, 256)


def realistic_frame(name, k):
    """Frame k (0..3) of spec 49 draft 4's realistic test sequence "pan" or "texture" (formulas only, no game data):
    output 512 x 512, render 256 x 256, HDR, MV low-res, depth inverted."""
    rw, rh = SEQ_RENDER
    y, x = np.mgrid[0:rh, 0:rw].astype(np.float64)
    color = np.ones((rh, rw, 4), np.float64)
    motion = np.zeros((rh, rw, 2), np.float64)
    if name == "pan":
        xs, ys = (x - 0.3 * k) / 256.0, y / 256.0
        color[..., 0] = 1.5 * (0.5 + 0.4 * np.sin(6 * xs + 2 * ys))
        color[..., 1] = 1.5 * (0.4 + 0.3 * np.cos(5 * ys))
        color[..., 2] = 1.5 * (0.3 + 0.2 * np.sin(9 * xs * ys))
        color[..., 3] = 1.5
        motion[..., 0] = 0.3
        depth = 0.3 + 0.2 * np.sin(3 * xs + ys)
    elif name == "texture":
        G = np.random.default_rng(77).gamma(0.8, 0.5, (65, 65, 3))
        xi, yi = x.astype(int), y.astype(int)
        rgb = 0.7 * G[yi // 4, xi // 4] + 0.6 * (((xi // 7) + (yi // 11)) % 2)[..., None]
        hot = (yi >= 100) & (yi < 120) & (xi >= 140) & (xi < 170)
        rgb[hot] = 12.0
        color[..., :3] = rgb
        depth = np.where(xi > yi, 0.4, 0.7)
    else:
        raise ValueError(name)
    return dict(color=color.astype(np.float32), motion=motion.astype(np.float32), depth=depth.astype(np.float32),
                render_size=SEQ_RENDER, jitter=(radical_inverse(k + 1, 2) - 0.5, radical_inverse(k + 1, 3) - 0.5),
                mv_scale=(1.0, 1.0), mv_offset=(0.0, 0.0), pre_exposure=1.0, reset=False, E_cur=1.1, E_prev=0.8,
                E_aux=1.3, e1=1.0, e2=1.0)


def check_sequences(dev, funcs, variant, params=None, seed=0):
    """Spec 49 draft 4's whole-network runs: both realistic sequences through the GPU frame (input stage -> network
    -> output stage, histories carried over, reset on frame 0 only); each frame's network is checked against the
    reference fed the same network input (the one the GPU input stage made)."""
    from clean.k.runner.k_frame import KFrame
    params = params or N.random_network_params(seed)
    out = []
    for name in ("pan", "texture"):
        kf = KFrame(dev, funcs, SEQ_OUTPUT, SEQ_RENDER, params, SEQ_FLAGS, variant, s=1.0)  # the recipe: s = 1
        try:
            for k in range(4):
                o = kf.frame(realistic_frame(name, k), debug=True)
                r = check_input(dev, funcs, o["x0"], params, o["f"], variant)
                r.update(sequence=name, frame=k, output_size=list(SEQ_OUTPUT))
                out.append(r)
        finally:
            kf.release()
    return out


def print_check(label, r):
    worst = max(r["stages"].items(), key=lambda kv: kv[1]["max_rel"] / kv[1]["limit_max"])
    verdict = "threshold run (not counted)" if r["pass"] is None else "pass" if r["pass"] else "FAIL"
    print(f"check {label} (base {r['base'][0]}x{r['base'][1]}): {verdict}  worst max/rms {worst[1]['max_rel']:.2e} "
          f"(limit {worst[1]['limit_max']:.2g}, {worst[0]}), d0 head mean/max "
          f"{r['stages']['d0.head']['mean_rel']:.2e}/{r['stages']['d0.head']['max_rel']:.2e}, mean/rms max "
          f"{max(x['mean_rel'] for x in r['stages'].values()):.2e}, non-finite "
          f"{sum(x['nonfinite'] for x in r['stages'].values())}", flush=True)


def bench_frame(dev, funcs, out_size, ratio, flags, variant, repeats, params=None, seed=0, s=None, sustained=0):
    """Median GPU ms of one whole frame (input stage, network, output stage) after two warm-up frames; with sustained = N
    the sustained-load measure instead (KFrame.time_sustained: N frames back to back, per frame over the second half)."""
    from clean.k.runner.k_frame import KFrame
    W, H = out_size
    rw, rh = int(round(W / ratio)), int(round(H / ratio))
    kf = KFrame(dev, funcs, (W, H), (rw, rh), params or N.random_network_params(seed), flags, variant, s=s)
    try:
        rng = np.random.default_rng(seed)
        x = synthetic_frame(rng, (W, H), (rw, rh), flags)
        kf.frame(x)
        kf.frame(x)  # history valid from here on
        ms = kf.time_sustained(x, sustained) if sustained else kf.time_frame(x, repeats)
        return {"output_size": [W, H], "render_size": [rw, rh], "base": [kf.Bw, kf.Bh], "variant": variant,
                "flags": flags, "ms": ms}
    finally:
        kf.release()


def print_bench(r):
    print(f"\noutput {r['output_size'][0]}x{r['output_size'][1]}  base {r['base'][0]}x{r['base'][1]}  "
          f"variant {r['variant']}  f {r['f']}  total {r['total_ms']:.3f} ms  {r['gflop']:.1f} GFLOP  "
          f"memory {r['memory_mib']:.0f} MiB")
    print(f"{'stage':6s} {'lvl':>3s} {'size':>11s} {'C':>4s} {'mix':>8s} {'windows':>8s} {'ms':>8s} {'GFLOP':>7s} "
          f"{'TFLOP/s':>8s}")
    for n, v in r["stages"].items():
        tf = f"{v['tflops']:.1f}" if v["tflops"] else "-"
        print(f"{n:6s} {v['level']:3d} {v['size'][0]:5d}x{v['size'][1]:<5d} {v['C']:4d} {v['mix']:>8s} "
              f"{v['windows']:8d} {v['ms']:8.3f} {v['gflop']:7.2f} {tf:>8s}")
    for n, ks in r.get("kernels", {}).items():
        print(f"  {n}: " + ", ".join(f"{k} {ms:.3f}" for k, ms in ks))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--build", required=True)
    ap.add_argument("--rocm", default=os.environ.get("ROCM_PATH", ""))
    ap.add_argument("--backend", default="hip", choices=("hip", "emu"))
    ap.add_argument("--sizes", nargs="*", default=["1080p", "1440p", "4k"])
    ap.add_argument("--s", type=float, default=DEFAULT_S,
                    help=f"base-size factor s of spec 44 (default {DEFAULT_S}, spec 44 draft 3)")
    ap.add_argument("--variants", nargs="+", default=["fused"], choices=list(FUSED_VARIANTS) + sorted(KL.LIN_CO),
                    help="fused: one kernel per stage, wide kernels at levels 3..5, two windows per workgroup in e0 "
                         "and d0 (default; fused2 = fused); fused_w1: one window in e0 and d0; fused1: k_stage_* everywhere; auto: e3, e4, d4, d3 "
                         "split; wide2 / wide4: NG = 2 / 4 at levels 3..5 (clean/k/runner/k_net.py); g2 / g1: separate "
                         "kernels, linear kernels with 32 / 16 tokens per wave. All of these accumulate in FP32. "
                         "Opt-in FP16 accumulation (spec 49 speed approximation, clean/k/README.md): <variant>@f16, "
                         "and fused16 (the fastest of them; fused16_w1: one window in e0 and d0)")
    ap.add_argument("--frame", type=int, default=0, help="frame index f (spec 44 offsets)")
    ap.add_argument("--repeats", type=int, default=30)
    ap.add_argument("--kernels", action="store_true", help="also time every kernel launch on its own")
    ap.add_argument("--check", nargs="*", default=[], metavar="WxH", help="GPU vs NumPy at these output sizes")
    ap.add_argument("--sequences", action="store_true",
                    help="spec 49 draft 4's counted whole-network check: the realistic pan and texture sequences "
                         "(4 frames each, 512x512 output) through the GPU frame, each frame's network vs the reference")
    ap.add_argument("--base", action="store_true",
                    help="the sizes given to --check / --sizes are base map sizes (any W x H), not output sizes")
    ap.add_argument("--model", help="the converter's output folder (spec 48 --raw): real parameters instead of random")
    ap.add_argument("--frames", action="store_true",
                    help="also time whole frames (input stage + network + output stage, clean/k/runner/k_frame.py) "
                         "at the --sizes output sizes, render size = output / --ratio, all eight flag combinations "
                         "with --all-flags (else HDR, render-resolution motion, inverted depth)")
    ap.add_argument("--sustained", type=int, default=0, metavar="N",
                    help="with --frames: N whole frames enqueued back to back (sustained load, as in a game; use for "
                         "absolute numbers): time per frame over the second half and per-part medians of sampled frames")
    ap.add_argument("--ratio", type=float, default=1.5, help="output / render size for --frames (1.5 = Quality)")
    ap.add_argument("--all-flags", action="store_true")
    ap.add_argument("--json", help="write all results here")
    a = ap.parse_args(argv)
    dev = open_device(a.backend, a.build, a.rocm)
    funcs = KFuncs(dev, a.build, a.backend)
    params = None
    if a.model:
        from clean.k.runner.k_model import load_raw
        params = load_raw(a.model)
    results = {"device": getattr(dev, "device_name", ""), "parameters": "model" if a.model else "random",
               "checks": [], "bench": []}
    for size in a.check:
        for v in a.variants:
            sz = parse_size(size)
            r = check(dev, funcs, sz, a.s, v, a.frame, base=sz if a.base else None, params=params)
            results["checks"].append(r)
            print_check(f"{size} {v}", r)
    if a.sequences:
        results["sequences"] = []
        for v in a.variants:
            if v not in FUSED_VARIANTS:
                continue
            for r in check_sequences(dev, funcs, v, params):
                results["sequences"].append(r)
                print_check(f"{r['sequence']} frame {r['frame']} {v}", r)
    for size in a.sizes:
        for v in a.variants:
            sz = parse_size(size)
            r = bench(dev, funcs, sz, a.s, v, a.frame, a.repeats, a.kernels, base=sz if a.base else None,
                      params=params)
            results["bench"].append(r)
            print_bench(r)
    if a.frames:
        results["frames"] = []
        flag_sets = [dict(hdr=h, mv_lowres=m, depth_inverted=i) for h in (True, False) for m in (True, False)
                     for i in (True, False)] if a.all_flags else [dict(hdr=True, mv_lowres=True, depth_inverted=True)]
        for size in a.sizes:
            for v in a.variants:
                if v not in FUSED_VARIANTS:
                    continue
                for flags in flag_sets:
                    r = bench_frame(dev, funcs, parse_size(size), a.ratio, flags, v, a.repeats, params, s=a.s,
                                    sustained=a.sustained)
                    results["frames"].append(r)
                    print(f"frame {r['output_size'][0]}x{r['output_size'][1]} render {r['render_size'][0]}x"
                          f"{r['render_size'][1]} {v} {flags}: " +
                          ", ".join(f"{k} {ms:.3f} ms" for k, ms in r["ms"].items()), flush=True)
    if a.json:
        with open(a.json, "w") as f:
            json.dump(results, f, indent=1)


if __name__ == "__main__":
    main()
