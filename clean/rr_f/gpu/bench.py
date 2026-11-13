# SPDX-License-Identifier: MIT
"""GPU time of every spec 51 stage kernel at a render size (default 1440p Quality: render 1706 x 960, level-0 tokens
1728 x 960 = the render size rounded up to multiples of 32), synthetic weights and inputs, shift (4, 4) (recurrent
halves (0, 0)). One host call (one GPU lock); best and mean of --reps runs per stage.

    python -m clean.rr_f.gpu.bench [--render 1706x960] [--stages E2,...] [--z f32] [--step E] [--reps 20]"""
import argparse
import os
import sys
import time

import numpy as np

from clean.rr_f.gpu import build, stage
from clean.rr_f.reference.fmt import f16
from clean.rr_f.reference.stages import STAGES, stage_info
from clean.rr_f.tests import synth
from clean.rr_f.tests.vectors_io import load_manifest, window_consts

ALL = ["IN0", "E1", "E2", "E3", "E4", "E5", "D4-F", "D4-R", "D3-F", "D3-R", "D2-F", "D2-R", "D1-F", "D1-R", "D0-F"]


def fast_e4m3(rng, shape):
    """Network-like e4m3 codes (|value| about 2^-6 .. 4, random sign), cheap to draw for large tensors."""
    e = rng.integers(1, 10, size=shape, dtype=np.uint8)
    m = rng.integers(0, 8, size=shape, dtype=np.uint8)
    s = rng.integers(0, 2, size=shape, dtype=np.uint8)
    return (s * 128 + e * 8 + m).astype(np.uint8)


def fast_hidden(rng, shape):
    return rng.standard_normal(shape + (4,), dtype=np.float32).astype(np.float16).astype(np.float64)


def inputs_for(rng, name, Nx, Ny):
    i = stage_info(name)
    C = i["C"]
    if i["kind"] == "fuse":
        return {"cin": fast_e4m3(rng, (Ny // 2, Nx // 2, i["Kc"])), "skip": fast_e4m3(rng, (Ny, Nx, C))}
    if i["adapter"]:
        return {"in": fast_e4m3(rng, (Ny, Nx, 128)), "hold": fast_hidden(rng, (Ny, Nx))}
    d = {"x": fast_e4m3(rng, (Ny, Nx, C)), "hold": fast_hidden(rng, (Ny, Nx))}
    if i["kind"] == "rec":
        d["Hc"] = fast_hidden(rng, (Ny // 2, Nx // 2))
    return d


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--render", default="1706x960")
    ap.add_argument("--stages", default=",".join(ALL))
    ap.add_argument("--z", default="f32")
    ap.add_argument("--step", default="E")
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--ldsw", type=int, default=1, help="1: weight units staged through LDS (RR_LDSW)")
    ap.add_argument("--tag", default="bench")
    ap.add_argument("--nolock", action="store_true")
    a = ap.parse_args(argv)
    rw, rh = (int(v) for v in a.render.split("x"))
    Wt, Ht = -(-rw // 32) * 32, -(-rh // 32) * 32
    rng = np.random.default_rng(7)
    wc = window_consts(load_manifest(), "A")
    ar = build.Arena()
    kernels, disp, names = {}, [], []
    t0 = time.time()
    stage.precompile([(n, a.z, a.step, a.ldsw) for n in a.stages.split(",")])
    for name in a.stages.split(","):
        lv = stage_info(name)["level"]
        Nx, Ny = Wt >> lv, Ht >> lv
        W = synth.stage_weights(rng, name)
        consts = {"ra": f16(0.75), "m": f16(0.5), "f": f16(1.0), "g": f16(1.25)}
        sc = stage.StageCall(name, inputs_for(rng, name, Nx, Ny), W, consts, wc, (4, 4), arena=ar, step=a.step)
        kid = name.replace("-", "")
        kernels[kid] = (stage.kernel_for(name, a.z, step=a.step, ldsw=a.ldsw), len(sc.args))
        disp.append((kid, sc.groups(), sc.args))
        names.append((name, Nx, Ny, sc.groups()))
    print(f"render {rw}x{rh} -> level-0 tokens {Wt}x{Ht}; arena {ar.size / 1e6:.0f} MB, prepared in "
          f"{time.time() - t0:.0f} s", flush=True)
    img, log = build.run(ar, kernels, disp, lock=not a.nolock, reps=a.reps,
                         workdir=os.path.join(build.BUILD, "runs", a.tag))
    times = {}
    for line in log.splitlines():
        if line.startswith("time "):
            p = line.split()
            times[int(p[1])] = (float(p[4]), float(p[7]))
    tot_b = tot_m = 0.0
    print(f"{'stage':6s} {'tokens':>11s} {'windows':>8s} {'best ms':>8s} {'mean ms':>8s}")
    for i, (name, Nx, Ny, g) in enumerate(names):
        b, m = times.get(i, (float("nan"), float("nan")))
        tot_b += b
        tot_m += m
        print(f"{name:6s} {Nx:5d}x{Ny:<5d} {g:8d} {b:8.3f} {m:8.3f}")
    print(f"{'total':6s} {'':>11s} {'':>8s} {tot_b:8.3f} {tot_m:8.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
