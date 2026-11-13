# SPDX-License-Identifier: MIT
"""Per-phase cycle profile of the second Swin design (kernels/swin_v2.hip built as variant "v2prof").

    python clean/runner/build.py --rocm <ROCm> --out build --variants v2 --ablations      # builds swin__v2prof.co
    python clean/runner/profile_v2.py --build build --rocm <ROCm> [--stages enc1,b0,...] [--json prof.json]
    python clean/runner/build.py --rocm <ROCm> --out build --variants bmega --ablations   # builds swin__bmegaprof.co
    python clean/runner/profile_v2.py --build build --rocm <ROCm> --variant bmegaprof --stages bneck

Each wave reads the shader cycle counter (SHADER_CYCLES; gfx12 has no s_memtime) at every phase boundary and adds the
cycles of the phase, and one visit, to a device global (wg_v2_prof). Barrier waits are separate phases, so the table
shows where waves compute and where they wait. Synthetic data at the in-game sizes (timing does not depend on values).
Prints per stage and phase: visits per wave and window, average cycles per visit, and the share of all wave cycles.
Stages "bneck" and "mid" profile the persistent kernels (swin_bneck6, swin_mid) in bmega* builds.
Numbers only; safe to paste into results/.
"""
import argparse
import json
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np  # noqa: E402

from clean.reference.swin import random_inputs, random_params, window_offset  # noqa: E402
from clean.runner.stage_runner import Runner  # noqa: E402
from clean.runner.sweep import GAME_SIZES, stage_inputs, stage_offset, stage_params  # noqa: E402

NPROF = 32
# Phase ids of swin_v2.hip (WG_V2_PROF / WG_V2_BARRIER arguments).
PHASES = {
    0: "start: gather input, stage first weights (+GELU table)",
    1: "wait: barrier after start",
    # persistent kernels (stages "bneck" and "mid", variant bmegaprof)
    2: "mega: claim, wait for the previous block's rows, barrier",
    3: "mega: item end (release, barrier, row counters)",
    4: "mid: fused expand + gather (dec2 items)",
    5: "mid: fused patch merge (enc2 items)",
    # resident C=64 kernel / unsplit streamed kernel
    20: "resident: copy block section + table to LDS",
    21: "wait: barrier after copy",
    22: "window: norm1",
    23: "window: attention (all heads)",
    24: "window: norm2",
    25: "window: MLP (all hidden tiles)",
    26: "window: store",
    27: "streamed: head (incl. staging)",
    28: "wait: barrier after a streamed unit",
    29: "streamed: MLP unit",
}


def read_prof(runner):
    ptr, size = runner.dev.get_global(runner.module, "wg_v2_prof")
    raw = runner.dev.download(ptr, 16 * NPROF)
    return np.frombuffer(raw, dtype=np.uint64).reshape(NPROF, 2).astype(np.float64)


def zero_prof(runner):
    ptr, _ = runner.dev.get_global(runner.module, "wg_v2_prof")
    runner.dev.upload(ptr, bytes(16 * NPROF))


def profile(build_dir, rocm_dir="", backend="hip", variant="v2prof", stages=None, sizes=None, seed=0):
    r = Runner(build_dir, rocm_dir, backend=backend, variant=variant)
    report = {"device": r.dev.device_name, "variant": variant, "stages": {}}
    for stage in stages or list(GAME_SIZES):
        if stage in ("bneck", "mid"):
            report["stages"][stage] = _profile_persistent(r, stage, (sizes or GAME_SIZES)[stage], seed)
            continue
        H, W, _ = (sizes or GAME_SIZES)[stage]
        rng = np.random.default_rng(seed)
        p = random_params(stage, rng)
        x, skip = random_inputs(stage, H, W, rng)
        off = window_offset(stage, 0)
        wts = r.pack_weights(stage, p)
        r.run(stage, wts, x, off, skip)  # warm-up
        zero_prof(r)
        r.run(stage, wts, x, off, skip)
        prof = read_prof(r)
        nwin = ((W + off[0] + 3) // 4) * ((H + off[1] + 3) // 4)
        report["stages"][stage] = _rows(prof, H, W, nwin)
    return report


def _rows(prof, H, W, nwin):
    total = prof[:, 0].sum()
    rows = {}
    for k in range(NPROF):
        cyc, n = prof[k]
        if n == 0:
            continue
        rows[k] = {"phase": PHASES.get(k, f"phase {k}"), "visits": int(n), "cycles_per_visit": cyc / n,
                   "share": cyc / total if total else 0.0}
    return {"size": [H, W], "windows": nwin, "wave_cycles": total, "phases": rows}


def _profile_persistent(r, stage, size, seed):
    """Stage "bneck" (b0..b5) or "mid" (enc2 .. dec2): the persistent kernel in bmega* builds, else the launches."""
    H, W = size[:2]
    rng = np.random.default_rng(seed)
    wts = r.pack_weights(stage, stage_params(stage, rng))
    x, _ = stage_inputs(stage, H, W, rng)
    off = stage_offset(stage, 0)
    r.run(stage, wts, x, off)  # warm-up
    zero_prof(r)
    r.run(stage, wts, x, off)
    if stage == "bneck":
        grids = [(H, W, window_offset(f"b{k}", 0)) for k in range(6)]
    else:
        grids = [(H, W, window_offset("enc2", 0))] + [(H // 2, W // 2, window_offset(f"b{k}", 0)) for k in range(6)] \
            + [(H, W, window_offset("dec2", 0))]
    nwin = sum(((w + o[0] + 3) // 4) * ((h + o[1] + 3) // 4) for h, w, o in grids)
    return _rows(read_prof(r), H, W, nwin)


def format_report(rep):
    lines = [f"device: {rep['device']}   variant: {rep['variant']}"]
    for stage, e in rep["stages"].items():
        lines.append(f"\n{stage}  {e['size'][0]}x{e['size'][1]}  windows {e['windows']}  "
                     f"wave cycles total {e['wave_cycles']:.3e}")
        for k, row in e["phases"].items():
            lines.append(f"  {k:2d} {row['phase'][:52]:52s} visits {row['visits']:8d}  "
                         f"cycles/visit {row['cycles_per_visit']:9.0f}  share {100 * row['share']:5.1f} %")
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--build", required=True)
    ap.add_argument("--rocm", default=os.environ.get("ROCM_PATH", ""))
    ap.add_argument("--backend", default="hip", choices=["hip", "emu"])
    ap.add_argument("--variant", default="v2prof")
    ap.add_argument("--stages", help="comma-separated (default: all of sweep.GAME_SIZES)")
    ap.add_argument("--json")
    args = ap.parse_args(argv)
    rep = profile(args.build, args.rocm, args.backend, args.variant,
                  args.stages.split(",") if args.stages else None)
    print(format_report(rep))
    if args.json:
        with open(args.json, "w") as f:
            json.dump(rep, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
