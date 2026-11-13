# SPDX-License-Identifier: MIT
"""How far is the MLP tile loop from the sum of its parts? (synthetic data only)

    python clean/runner/build.py --rocm <ROCm> --out build            # builds mlpbench.co with the other code objects
    python clean/runner/mlp_probe.py --build build --rocm <ROCm> [--json mlp_probe.json]

kernels/mlpbench.hip runs the tile loop of the Swin block's MLP (W1 chain of KC WMMAs -> GELU + FP8 pack -> W2: KC WMMAs
into KC accumulators) with everything else removed: the weights are a fixed LDS unit, there are no global loads, no
staging, no barriers. KC = 4 is the C=64 block, KC = 8 the C=128 block. Every part can be removed (W1, GELU, W2), so the
loop can be decomposed the way the ablation builds decompose the whole block, but without the rest of the kernel.

Printed: SIMD time per tile (ns; the kernel time divided by the tiles each SIMD runs) for every combination of parts at
2, 4, 8 and 12 waves per SIMD (launches of 32 / 64 / 128 / 192 workgroups of 8 waves, spread over the
128 SIMDs). The ideal WMMA time is 8 cycles per WMMA (about 3 ns at 2.65 GHz). The questions:
  - does the full loop cost the sum of its parts, or more (WMMA <-> vector switching, dependent chains)?
  - does it get cheaper with more waves per SIMD (latency hiding), and where does it stop improving?
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

BLOCK = 256
SIMDS = 128
PARTS = ["111", "111p", "011", "101", "110", "100", "001", "010", "000"]  # W1 GELU W2 present (1) / removed (0)
WAVES_PER_SIMD = [2, 4, 8, 12]


def _time(dev, func, grid, args, repeats):
    start, stop = dev.event(), dev.event()
    try:
        dev.launch(func, grid, BLOCK, args)
        dev.sync()
        times = []
        for _ in range(repeats):
            dev.record(start)
            dev.launch(func, grid, BLOCK, args)
            dev.record(stop)
            times.append(dev.elapsed_ms(start, stop))
        return float(np.min(times))
    finally:
        dev.destroy_event(start)
        dev.destroy_event(stop)


def run(build_dir, rocm_dir, repeats=7, target_ms=2.0):
    from clean.runner.hip import Hip
    dev = Hip(rocm_dir)
    with open(os.path.join(build_dir, "mlpbench.co"), "rb") as f:
        module = dev.load_module(f.read())
    res = {}
    out = dev.malloc(192 * BLOCK * 4)
    try:
        for kc in (4, 8):
            funcs = {p: dev.get_function(module, f"mb{kc}_{p}") for p in PARTS}
            tiles = 200
            for _ in range(3):  # calibrate on the full loop at 4 waves per SIMD
                t = _time(dev, funcs["111"], 64, [ctypes.c_void_p(out), ctypes.c_int(tiles)], 3)
                tiles = max(50, int(tiles * target_ms / max(t, 1e-3)))
            res[kc] = {"tiles": tiles, "ns_per_tile": {}}
            for p in PARTS:
                res[kc]["ns_per_tile"][p] = {}
                for k in WAVES_PER_SIMD:
                    grid = SIMDS * k // 8
                    t = _time(dev, funcs[p], grid, [ctypes.c_void_p(out), ctypes.c_int(tiles)], repeats)
                    res[kc]["ns_per_tile"][p][k] = t * 1e6 / (k * tiles)
        return res
    finally:
        dev.free(out)


def format_table(res):
    L = []
    for kc, r in res.items():
        L.append(f"\nKC = {kc} ({'C=64' if kc == 4 else 'C=128'}), {r['tiles']} tiles per wave. SIMD time per tile, ns "
                 f"(ideal WMMA only: {2 * kc * 8 / 2.65:.0f} ns at 8 cycles per WMMA, 2.65 GHz)")
        L.append(f"  {'parts W1 GELU W2':18s}" + "".join(f"{k:>9d}w" for k in WAVES_PER_SIMD))
        for p in PARTS:
            L.append(f"  {p:18s}" + "".join(f"{r['ns_per_tile'][p][k]:10.1f}" for k in WAVES_PER_SIMD))
        L.append("  111p = the full loop with the LDS fragments of the W2 WMMAs and of the next tile's W1 loaded ahead of the GELU")
        L.append("  full loop vs sum of its parts (111 against 100 + 001 + 010 - 2 x 000):")
        for k in WAVES_PER_SIMD:
            n = r["ns_per_tile"]
            parts = n["100"][k] + n["001"][k] + n["010"][k] - 2 * n["000"][k]
            L.append(f"    {k:2d} waves/SIMD: full {n['111'][k]:7.1f} ns, sum of parts {parts:7.1f} ns, ratio {n['111'][k] / parts:5.2f}")
    return "\n".join(L)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--build", required=True)
    ap.add_argument("--rocm", default=os.environ.get("ROCM_PATH", ""))
    ap.add_argument("--json")
    args = ap.parse_args(argv)
    res = run(args.build, args.rocm)
    print(format_table(res))
    if args.json:
        with open(args.json, "w") as f:
            json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
