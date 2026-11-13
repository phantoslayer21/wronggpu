# SPDX-License-Identifier: MIT
"""Throughput of the vector instructions the Swin kernels use (synthetic data only).

    python clean/runner/build.py --rocm <ROCm> --out build            # builds rates.co with the other code objects
    python clean/runner/rates.py --build build --rocm <ROCm> [--json rates.json]

For every instruction of kernels/rates.hip (16 independent chains per wave, all CUs busy) prints the sustained rate in
lane-operations per second and the cost in "FMA slots" (1.00 = as fast as v_fma_f32; a packed instruction that costs 1.00
slots does two values' work per slot). This is the table the issue-slot model of the vector work needs: transcendentals
are expected at about 4, packed FP16 at 1, but the conversions (v_cvt_pk_fp8_f32, v_cvt_pkrtz_f16_f32, v_cvt_f32_fp8,
v_cvt_f32_f16) and v_dot2_f32_f16 were never measured.
"""
import argparse
import ctypes
import json
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np  # noqa: E402

from clean.runner.gen_rates import MIX_K, OPS  # noqa: E402

BLOCK = 256
NAMES = list(OPS) + ["fma_c"] + [f"mix_exp_fma{k}" for k in MIX_K]


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


def run(build_dir, rocm_dir, workgroups=4096, repeats=9, target_ms=2.0):
    from clean.runner.hip import Hip
    dev = Hip(rocm_dir)
    with open(os.path.join(build_dir, "rates.co"), "rb") as f:
        module = dev.load_module(f.read())
    funcs = {n: dev.get_function(module, "rt_" + n) for n in NAMES}
    out = dev.malloc(workgroups * BLOCK * 4)
    try:
        def args(iters):
            return [ctypes.c_void_p(out), ctypes.c_int(iters)]

        iters = 200
        for _ in range(3):  # calibrate on the FMA chains
            med, _ = _time(dev, funcs["fma_f32"], workgroups, args(iters), 3)
            iters = max(50, int(iters * target_ms / max(med, 1e-3)))
        lanes = workgroups * BLOCK
        res = {}
        for n in NAMES:
            med, mn = _time(dev, funcs[n], workgroups, args(iters), repeats)
            ops_per_lane = iters * (16 if not n.startswith("mix_") else 0)  # mixes are reported by time
            res[n] = {"median_ms": med, "min_ms": mn, "Tlane_ops_per_s": lanes * ops_per_lane / (mn * 1e-3) / 1e12}
        base = res["fma_f32"]["Tlane_ops_per_s"]
        for n in NAMES:
            if not n.startswith("mix_"):
                res[n]["fma_slots"] = base / res[n]["Tlane_ops_per_s"]
        # transcendental unit beside the vector ALU? One wave issues 8 v_exp and 8k v_fma per iteration; compare with the
        # times of 8 v_exp alone and 8k v_fma alone (the 16-chain kernels do 16 per iteration).
        for k in MIX_K:
            t_exp = res["exp_f32"]["min_ms"] * 8 / 16
            t_fma = res["fma_f32"]["min_ms"] * 8 * k / 16
            t_mix = res[f"mix_exp_fma{k}"]["min_ms"]
            total, longest = t_exp + t_fma, max(t_exp, t_fma)
            res[f"mix_exp_fma{k}"].update({"t_exp_alone_ms": t_exp, "t_fma_alone_ms": t_fma,
                                           "overlap": (total - t_mix) / (total - longest)})
        return {"iters": iters, "workgroups": workgroups, "ops": res}
    finally:
        dev.free(out)


def format_table(r):
    L = [f"{'instruction':20s} {'T lane-ops/s':>13s} {'FMA slots':>10s}"]
    for n, v in r["ops"].items():
        if not n.startswith("mix_"):
            L.append(f"{n:20s} {v['Tlane_ops_per_s']:13.2f} {v['fma_slots']:10.2f}")
    L.append("")
    L.append("v_exp_f32 interleaved with v_fma_f32 in one wave (overlap 1.0 = the transcendental unit runs beside the ALU,"
             " 0.0 = they add up):")
    for n, v in r["ops"].items():
        if n.startswith("mix_"):
            L.append(f"  1 exp : {n[len('mix_exp_fma'):]:>2s} fma   mix {v['median_ms']:.3f} ms   exp alone {v['t_exp_alone_ms']:.3f}"
                     f"   fma alone {v['t_fma_alone_ms']:.3f}   overlap {v['overlap']:+.2f}")
    return "\n".join(L)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--build", required=True)
    ap.add_argument("--rocm", default=os.environ.get("ROCM_PATH", ""))
    ap.add_argument("--json")
    args = ap.parse_args(argv)
    r = run(args.build, args.rocm)
    print(format_table(r))
    if args.json:
        with open(args.json, "w") as f:
            json.dump(r, f, indent=1)


if __name__ == "__main__":
    main()
