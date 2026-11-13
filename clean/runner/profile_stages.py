# SPDX-License-Identifier: MIT
"""Where does each Swin stage wait? Per-phase timing of the block kernels, per wave, at the in-game size.

    python clean/runner/build.py --rocm <ROCm> --out build --profile
    python clean/runner/profile_stages.py --build build --rocm <ROCm> [--variant prof] [--json profile.json]

The instrumented builds make a sample of the waves (every `stride`-th wave of the launch) read a clock at the boundaries
of the phases of the block and write the time spent in each phase. Reading a clock is not free, and the first version
that instrumented every wave through the realtime message slowed the kernel about 7x, so now:
  - `prof` (default) reads the shader cycle counter (one scalar s_getreg per read); phases are drained at every boundary
    so each phase pays its own latency (the phases serialize; the sum is a few percent above the normal kernel);
  - `prof2` the same without draining (a stall lands in the phase that first uses the data);
  - `profrt` reads the 100 MHz realtime clock through a message (the old way; use it to check the cycle counter);
  - only every `stride`-th wave is instrumented (default: about 512 waves per stage, `--stride N` to override). The
    others run the normal code, so the launch is almost undisturbed. Every wave also records its start on the realtime
    clock and its hardware id, which gives the wave timeline.
For every stage the tool prints the unperturbed kernel time (base build), the instrumented kernel time at the chosen
stride (the perturbation; it must be close to the base time or the phase numbers do not describe the normal kernel) and
at stride 1 (every wave, for comparison) unless --no-full.

Printed per stage:
  - the launch (windows, workgroups, waves), the instrumented waves, waves in flight (average and peak, scaled by the
    stride), the timeline of wave start times (when does each round of workgroups start?), the spread of hardware ids;
  - per phase: average and maximum over the instrumented waves, in microseconds and as a share of the average wave, after
    subtracting the measured cost of the clock reads (the share is relative to the wave time without the reads too, so the
    phases add up to 100 % except for the small remainder reported as "other").
"""
import argparse
import json
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np  # noqa: E402

from clean.reference.swin import STAGES, random_inputs, random_params, window_offset  # noqa: E402
from clean.runner import variants as V  # noqa: E402
from clean.runner.packing import block_layout  # noqa: E402
from clean.runner.stage_runner import Runner  # noqa: E402
from clean.runner.sweep import GAME_SIZES  # noqa: E402

TICK_US = 0.01  # the realtime clock runs at 100 MHz
PHASES = ["load", "norm1", "qkv", "attention", "wp", "stage(heads)", "norm2", "w1", "gelu+pack", "w2", "stage(mlp)",
          "store"]
SIMDS_9070XT = 128  # 64 CUs x 2 SIMDs


def reads_per_phase(stage):
    """Number of clock reads charged to each phase per wave (the read at a boundary is paid by the phase after it)."""
    cfg = STAGES[stage]
    lay = block_layout(cfg["C"], cfg["heads"], cfg["mlp"])
    heads, tiles, units = cfg["heads"], lay["NMLP"] * lay["HPU"], lay["NMLP"]
    return [1, 1, heads, heads, heads, heads, 1, tiles, tiles, tiles, units, 1]


def decode_hwid(v):
    """Fields of HW_ID1 (RDNA layout: wave slot [4:0], SIMD [9:8], WGP [13:10], SA [16], SE [19:18]); the unit key is
    everything from bit 8 to 19 so unknown fields cannot split one SIMD into several ids."""
    v = np.asarray(v, np.int64)
    return {"key": (v >> 8) & 0xFFF, "simd": (v >> 8) & 3, "wgp": (v >> 10) & 15, "sa": (v >> 16) & 1,
            "se": (v >> 18) & 3}


def analyse(words, info, stage, simds=SIMDS_9070XT):
    words = np.asarray(words).astype(np.int64)
    rows = words[words[:, 12] > 0]
    n = len(rows)
    if n == 0:
        raise ValueError("no instrumented wave wrote a record (stride too large or not an instrumented build)")
    stride = int(info.get("stride", 1))
    unit_us = TICK_US * float(rows[:, 16].sum()) / float(rows[:, 12].sum())  # clock units -> microseconds
    overhead = float(np.median(rows[:, 15]))
    reads = np.array(reads_per_phase(stage), dtype=np.float64)
    nreads = reads.sum() + 2  # reads inside the wave time: one per phase boundary, the first one and the last one
    phases = rows[:, :12].astype(np.float64)
    corrected = np.maximum(phases - reads[None, :] * overhead, 0.0)
    total = rows[:, 12].astype(np.float64)
    total_c = np.maximum(total - nreads * overhead, 1.0)
    start = (rows[:, 13] - rows[:, 13].min()) & 0xFFFFFFFF
    start = np.where(start > 1 << 31, start - (1 << 32), start)
    start = start - start.min()
    dur = rows[:, 16]
    end = start + dur
    span = float(end.max())
    # waves in flight (sampled waves scaled by the stride): sweep the start / end events
    ev = np.concatenate([np.stack([start, np.ones(n)], 1), np.stack([end, -np.ones(n)], 1)])
    ev = ev[np.lexsort((ev[:, 1], ev[:, 0]))]
    level = np.cumsum(ev[:, 1])
    dt = np.diff(ev[:, 0], append=ev[-1, 0])
    mean_flight = float((level * dt).sum() / max(span, 1.0)) * stride
    peak_flight = float(level.max()) * stride
    ids = decode_hwid(rows[:, 14])
    uniq, counts = np.unique(ids["key"], return_counts=True)
    q = lambda a, p: float(np.percentile(a, p))  # noqa: E731
    mean_total = total_c.mean()
    res = {
        "stage": stage, "waves": int(info["waves"]), "instrumented_waves": n, "stride": stride,
        "workgroups": info["grid"], "threads": info["threads"], "windows": info["windows"],
        "unit_ns": unit_us * 1e3, "read_cost_ns": overhead * unit_us * 1e3,
        "wave_us_mean": float(mean_total * unit_us), "wave_us_max": float(total_c.max() * unit_us),
        "wave_us_raw_mean": float(total.mean() * unit_us), "span_us": span * TICK_US,
        "waves_in_flight_mean": mean_flight, "waves_in_flight_peak": peak_flight,
        "in_flight_per_simd": mean_flight / simds,
        "hwid_distinct": int(len(uniq)), "hwid_waves_min": int(counts.min()), "hwid_waves_median": float(np.median(counts)),
        "hwid_waves_max": int(counts.max()),
        "hwid_fields_distinct": {k: int(len(np.unique(v))) for k, v in ids.items() if k != "key"},
        "start_us_quantiles": {str(p): q(start, p) * TICK_US for p in (0, 10, 25, 50, 75, 90, 100)},
        "phases": {nme: {"mean_us": float(corrected[:, i].mean() * unit_us), "max_us": float(corrected[:, i].max() * unit_us),
                         "raw_mean_us": float(phases[:, i].mean() * unit_us),
                         "share": float(corrected[:, i].mean() / mean_total)} for i, nme in enumerate(PHASES)},
        "other_share": float(1.0 - corrected.sum(axis=1).mean() / mean_total),
    }
    return res


def format_stage(r, times=None):
    L = [f"\n{r['stage']}: {r['windows']} windows, {r['workgroups']} workgroups of {r['threads']} lanes = {r['waves']} waves;"
         f" {r['instrumented_waves']} instrumented (stride {r['stride']})"]
    if times:
        L.append("  kernel time: " + ", ".join(f"{k} {v:.3f} ms" for k, v in times.items()))
    L += [f"  wave time without clock reads: mean {r['wave_us_mean']:.1f} us, max {r['wave_us_max']:.1f} us "
          f"(with the reads {r['wave_us_raw_mean']:.1f} us); one clock read ~ {r['read_cost_ns']:.0f} ns; "
          f"1 clock unit = {r['unit_ns']:.2f} ns",
          f"  sampled timeline span {r['span_us']:.1f} us; waves in flight: mean {r['waves_in_flight_mean']:.0f}, peak "
          f"{r['waves_in_flight_peak']:.0f} ({r['in_flight_per_simd']:.2f} per SIMD of {SIMDS_9070XT})",
          "  wave start times (us after the first): " + "  ".join(f"p{k}={v:.1f}" for k, v in r["start_us_quantiles"].items()),
          f"  hardware ids: {r['hwid_distinct']} distinct (fields: "
          + ", ".join(f"{k} {v}" for k, v in r["hwid_fields_distinct"].items())
          + f"); sampled waves per id min/median/max {r['hwid_waves_min']}/{r['hwid_waves_median']:.0f}/{r['hwid_waves_max']}"
          + ("  <-- more ids than SIMDs: the id decode is suspect" if r["hwid_distinct"] > SIMDS_9070XT else ""),
          f"  {'phase':14s} {'mean us':>8s} {'max us':>8s} {'share':>6s}"]
    for nme, p in r["phases"].items():
        L.append(f"  {nme:14s} {p['mean_us']:8.2f} {p['max_us']:8.2f} {100 * p['share']:5.1f}%")
    L.append(f"  {'other':14s} {'':8s} {'':8s} {100 * r['other_share']:5.1f}%")
    return "\n".join(L)


def block_ms(times):
    return next(v for k, v in times.items() if k.startswith("swin_block"))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--build", required=True)
    ap.add_argument("--rocm", default=os.environ.get("ROCM_PATH", ""))
    ap.add_argument("--backend", default="hip", choices=["hip", "emu"])
    ap.add_argument("--variant", default="prof")
    ap.add_argument("--stages", help="comma-separated (default: enc1, enc2, b0, dec2, dec1)")
    ap.add_argument("--stride", type=int, default=0, help="instrument every N-th wave (default: about 512 per stage)")
    ap.add_argument("--target-waves", type=int, default=512)
    ap.add_argument("--runs", type=int, default=3, help="repeat and keep the run with the shortest block kernel time")
    ap.add_argument("--no-full", action="store_true", help="skip the stride-1 comparison run")
    ap.add_argument("--json")
    args = ap.parse_args(argv)
    runner = Runner(args.build, args.rocm, backend=args.backend, variant=args.variant)
    base = None
    try:
        base = Runner(args.build, args.rocm, backend=args.backend, variant="base")
    except Exception:  # noqa: BLE001
        pass
    out = {}
    for stage in (args.stages.split(",") if args.stages else ["enc1", "enc2", "b0", "dec2", "dec1"]):
        H, W, _ = GAME_SIZES[stage]
        rng = np.random.default_rng(0)
        params = random_params(stage, rng)
        x, skip = random_inputs(stage, H, W, rng)
        weights = runner.pack_weights(stage, params)
        offset = window_offset(stage, 0)
        times = {}
        reps = 30 if args.backend == "hip" else 1
        if base is not None:
            times["base"] = block_ms(base.time_kernels(stage, base.pack_weights(stage, params), x, offset, skip, reps))
        times[f"{args.variant} stride 0"] = block_ms(runner.time_kernels(stage, weights, x, offset, skip, reps, stride=0))
        best = None
        for _ in range(args.runs):
            stride = args.stride or max(1, runner.profile(stage, weights, x, offset, skip, stride=0)[1]["waves"]
                                         // args.target_waves)
            words, info = runner.profile(stage, weights, x, offset, skip, stride=stride)
            r = analyse(words, info, stage)
            if best is None or r["span_us"] < best["span_us"]:
                best = r
        times[f"{args.variant} stride {best['stride']}"] = block_ms(
            runner.time_kernels(stage, weights, x, offset, skip, reps, stride=best["stride"]))
        if not args.no_full:
            times[f"{args.variant} stride 1"] = block_ms(runner.time_kernels(stage, weights, x, offset, skip, reps, stride=1))
        best["kernel_ms"] = times
        out[stage] = best
        print(format_stage(best, times), flush=True)
    if args.json:
        with open(args.json, "w") as f:
            json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
