# SPDX-License-Identifier: MIT
"""GPU time of the spec 51 stage kernels for several step / Z variants in ONE host call (one GPU lock): for every
stage the variants' dispatches follow each other on the same arguments, so every variant is timed under the same
GPU state (clock, power, other load). Synthetic weights and inputs as in bench.py.

    python -m clean.rr_f.gpu.bench_steps [--render 1706x960] [--variants E:f32,X:exact,Y:exact] [--reps 10]
                                         [--stages IN0,E1,...] [--json out.json]"""
import argparse
import json
import os
import sys
import time

import numpy as np

from clean.rr_f.gpu import bench, build, stage
from clean.rr_f.reference.fmt import f16
from clean.rr_f.reference.stages import stage_info
from clean.rr_f.tests import synth
from clean.rr_f.tests.vectors_io import load_manifest, window_consts


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--render", default="1706x960")
    ap.add_argument("--stages", default=",".join(bench.ALL))
    ap.add_argument("--variants", default="E:f32,X:exact")
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--tag", default="bench_steps")
    ap.add_argument("--json")
    a = ap.parse_args(argv)
    rw, rh = (int(v) for v in a.render.split("x"))
    Wt, Ht = -(-rw // 32) * 32, -(-rh // 32) * 32
    variants = [tuple(v.split(":")) for v in a.variants.split(",")]
    stages = a.stages.split(",")
    rng = np.random.default_rng(7)
    wc = window_consts(load_manifest(), "A")
    ar = build.Arena()
    t0 = time.time()
    stage.precompile([(n, z, s, 1) for n in stages for s, z in variants])
    kernels, disp, rows = {}, [], []
    for name in stages:
        lv = stage_info(name)["level"]
        Nx, Ny = Wt >> lv, Ht >> lv
        W = synth.stage_weights(rng, name)
        consts = {"ra": f16(0.75), "m": f16(0.5), "f": f16(1.0), "g": f16(1.25)}
        inp = bench.inputs_for(rng, name, Nx, Ny)
        scs = {}                                   # E1: one call per input layout (stage.e1_direct of the step)
        for s, z in variants:
            dr = stage.e1_direct(step=s)
            if dr not in scs:
                scs[dr] = stage.StageCall(name, inp, W, consts, wc, (4, 4), arena=ar, step=s)
            sc = scs[dr]
            for kid, kern, g, args in stage.dispatches_for(sc, z, s, 1, f"{name.replace('-', '')}_{s}_{z}"):
                kernels[kid] = kern
                disp.append((kid, g, args))
                rows.append((name, s, z))
    print(f"render {rw}x{rh}; arena {ar.size / 1e6:.0f} MB, prepared in {time.time() - t0:.0f} s", flush=True)
    _, log = build.run(ar, kernels, disp, reps=a.reps, workdir=os.path.join(build.BUILD, "runs", a.tag))
    times = {}
    for line in log.splitlines():
        if line.startswith("time "):
            p = line.split()
            times[int(p[1])] = (float(p[4]), float(p[7]))
    per_dispatch = [dict(stage=name, variant=f"{s}:{z}", kernel=disp[i][0], best_ms=times.get(i, (float("nan"),) * 2)[0])
                    for i, (name, s, z) in enumerate(rows)]
    res = {}
    for i, (name, s, z) in enumerate(rows):                 # a stage's dispatches (Y + fixup) are summed
        b, m = times.get(i, (float("nan"),) * 2)
        ob, om = res.setdefault(name, {}).get(f"{s}:{z}", (0.0, 0.0))
        res[name][f"{s}:{z}"] = (ob + b, om + m)
    keys = [f"{s}:{z}" for s, z in variants]
    print(f"{'stage':6s} " + " ".join(f"{k:>12s}" for k in keys) + "   (best ms)")
    tot = {k: 0.0 for k in keys}
    for name in stages:
        print(f"{name:6s} " + " ".join(f"{res[name][k][0]:12.3f}" for k in keys))
        for k in keys:
            tot[k] += res[name][k][0]
    print(f"{'total':6s} " + " ".join(f"{tot[k]:12.3f}" for k in keys))
    if a.json:
        with open(a.json, "w") as f:
            json.dump(dict(render=[rw, rh], reps=a.reps, per_stage=res, total_best=tot, per_dispatch=per_dispatch), f,
                      indent=1)
    fix = [d for d in per_dispatch if d["kernel"].endswith("_fix")]
    if fix:
        print("fixup dispatches (best ms): " + " ".join(f"{d['stage']}:{d['best_ms']:.3f}" for d in fix))
    return 0


if __name__ == "__main__":
    sys.exit(main())
