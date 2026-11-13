# SPDX-License-Identifier: MIT
"""Can the matrix unit (WMMA) and the vector ALU work at the same time? Synthetic data only.

    python clean/runner/build.py --rocm <ROCm> --out build            # builds overlap.co with the other code objects
    python clean/runner/overlap_probe.py --build build --rocm <ROCm> [--json overlap.json]

Runs the kernels of kernels/overlap.hip (see the table there) on the whole GPU and prints, for every mix, how much of
the possible overlap was achieved:

    overlap = (t_matrix + t_vector - t_mix) / (t_matrix + t_vector - max(t_matrix, t_vector))

    1.0  the two kinds of work run completely side by side (the mix takes as long as the slower one alone)
    0.0  they serialize (the mix takes the sum of both)

mix8 / mix32 / mix_trans: one wave issues both kinds of work, interleaved (1 WMMA : 8 or 32 FMA, or 1 WMMA : 8 v_exp).
split8 / split32: half of the waves do only WMMA work, the other half only vector work (cross-wave overlap).
Also prints the sustained WMMA rate, the dependent-chain WMMA rate and the FMA / v_exp rates.
"""
import argparse
import json
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import ctypes  # noqa: E402

import numpy as np  # noqa: E402

BLOCK = 256
WAVES_PER_BLOCK = BLOCK // 32
WMMA_FLOP = 2 * 16 * 16 * 16          # one 16x16x16 WMMA
KERNELS = ["ov_mma", "ov_mma_dep", "ov_valu8", "ov_valu32", "ov_trans", "ov_mix8", "ov_mix32", "ov_mix_trans",
           "ov_split8", "ov_split32"]


def _time(dev, func, grid, args, repeats):
    start, stop = dev.event(), dev.event()
    try:
        dev.launch(func, grid, BLOCK, args)  # warm-up
        dev.sync()
        times = []
        for _ in range(repeats):
            dev.record(start)
            dev.launch(func, grid, BLOCK, args)
            dev.record(stop)
            times.append(dev.elapsed_ms(start, stop))
        return float(np.median(times)), float(np.min(times))
    finally:
        dev.destroy_event(start)
        dev.destroy_event(stop)


def overlap_fraction(t_matrix, t_vector, t_mix):
    total, longest = t_matrix + t_vector, max(t_matrix, t_vector)
    return float(np.clip((total - t_mix) / (total - longest), -1.0, 1.5)) if total > longest else float("nan")


def run(build_dir, rocm_dir, workgroups=2048, repeats=9, target_ms=2.0):
    from clean.runner.hip import Hip
    dev = Hip(rocm_dir)
    with open(os.path.join(build_dir, "overlap.co"), "rb") as f:
        module = dev.load_module(f.read())
    funcs = {k: dev.get_function(module, k) for k in KERNELS}
    out = dev.malloc(workgroups * BLOCK * 4)
    try:
        def args(iters):
            return [ctypes.c_void_p(out), ctypes.c_int(iters), ctypes.c_float(0.9999), ctypes.c_float(0.0001)]

        # calibrate the iteration count so that the matrix-only kernel takes about target_ms
        iters = 100
        for _ in range(3):
            med, _ = _time(dev, funcs["ov_mma"], workgroups, args(iters), 3)
            iters = max(20, int(iters * target_ms / max(med, 1e-3)))
        waves = workgroups * WAVES_PER_BLOCK
        t = {}
        for k in KERNELS:
            med, mn = _time(dev, funcs[k], workgroups, args(iters), repeats)
            t[k] = {"median_ms": med, "min_ms": mn}
        report = {"device": dev.device_name, "workgroups": workgroups, "waves": waves, "iters": iters, "times_ms": t}

        def ms(k, which="median_ms"):
            return t[k][which]

        rates = {
            "wmma_fp8_tflops": waves * iters * 8 * WMMA_FLOP / (ms("ov_mma") * 1e-3) / 1e12,
            "wmma_fp8_dependent_chain_tflops": waves * iters * 8 * WMMA_FLOP / (ms("ov_mma_dep") * 1e-3) / 1e12,
            "fma_f32_tops_lane_ops": waves * iters * 64 * 32 / (ms("ov_valu8") * 1e-3) / 1e12,
            "v_exp_f32_tops_lane_ops": waves * iters * 64 * 32 / (ms("ov_trans") * 1e-3) / 1e12,
        }
        report["rates"] = rates
        res = {}
        for name, mm, vv in (("mix8", "ov_mix8", "ov_valu8"), ("mix32", "ov_mix32", "ov_valu32"),
                             ("mix_trans", "ov_mix_trans", "ov_trans")):
            for which in ("median_ms", "min_ms"):
                res.setdefault(name, {})[which] = {
                    "t_matrix": ms("ov_mma", which), "t_vector": ms(vv, which), "t_mix": ms(mm, which),
                    "overlap": overlap_fraction(ms("ov_mma", which), ms(vv, which), ms(mm, which))}
        for name, ss, vv in (("split8", "ov_split8", "ov_valu8"), ("split32", "ov_split32", "ov_valu32")):
            for which in ("median_ms", "min_ms"):
                # half of the waves each: serialized = (t_m + t_v) / 2, overlapped = max(t_m, t_v) / 2
                tm, tv, ts = ms("ov_mma", which), ms(vv, which), ms(ss, which)
                res.setdefault(name, {})[which] = {
                    "t_matrix": tm, "t_vector": tv, "t_split": ts,
                    "overlap": overlap_fraction(tm / 2, tv / 2, ts)}
        report["overlap"] = res
        return report
    finally:
        dev.free(out)


def format_report(r):
    L = [f"device: {r['device']}   waves: {r['waves']}   iterations per wave: {r['iters']}"]
    L.append("times (median / min, ms):  " + "  ".join(f"{k}={v['median_ms']:.3f}/{v['min_ms']:.3f}"
                                                        for k, v in r["times_ms"].items()))
    for k, v in r["rates"].items():
        L.append(f"  {k}: {v:.1f}")
    L.append("overlap fraction (1 = perfect overlap, 0 = serialized):")
    for name, d in r["overlap"].items():
        m, mn = d["median_ms"], d["min_ms"]
        L.append(f"  {name:10s} median {m['overlap']:+.2f}   min {mn['overlap']:+.2f}")
    return "\n".join(L)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--build", required=True)
    ap.add_argument("--rocm", default=os.environ.get("ROCM_PATH", ""))
    ap.add_argument("--json")
    ap.add_argument("--repeats", type=int, default=9)
    args = ap.parse_args(argv)
    r = run(args.build, args.rocm, repeats=args.repeats)
    print(format_report(r))
    if args.json:
        with open(args.json, "w") as f:
            json.dump(r, f, indent=1)


if __name__ == "__main__":
    main()
