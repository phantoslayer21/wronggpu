# SPDX-License-Identifier: MIT
"""Step I against step X stage by stage (independent check of 2026-10-22, results/2026-10-21-body-levers.md section 5):
every listed stage runs with step X and with step I (+ its step-X fixup) on the SAME own synthetic weights and inputs in
one arena; the raw output bytes are compared (zero signs included) and the fixup windows counted.

    python -m clean.rr_f.gpu.step_cmp [--render 1706x960] [--profiles net,sub,ovf] [--stages IN0,...] [--seed 5]
                                      [--nofix] [--istep I+RR_ICOV=1] [--neg] [--json out.json]

--nofix: step I without its fixup dispatch; the bytes are then compared only on the tokens of the windows the I
dispatch did NOT mark (window wid = wj nwx + wi covers X = 8 wi - sx + 0..7, Y = 8 wj - sy + 0..7), i.e. I's own
arithmetic. --istep I+RR_ICOV=1: the coverage build (kernels/rr_step_i.hlsli RR_ICOV; bits above 0xff never mark a
window): per stage the number of unmarked windows that took the TwoSum rounding path (0x100), three activation digits
(0x200), a +-inf / NaN start value (0x400) and the 3 x 3 digit form of two activation blocks (0x800). --neg: one input
code of the I call changed (the comparison must then fail).
Profiles (e4m3 exponent fields e, E = max(e, 1)):
  net   synth.stage_weights + bench.inputs_for (network-like; the bench data)
  mid   weights e 2..9, inputs e 3..11          hi    weights e 4..9, inputs e 5..10
  ovf   weights e 9..13, inputs e 10..14 (binary16 overflow in the chains: +-inf start values)
  sub   weights e 0..5 (subnormals), inputs e 0..6
  big   weights e 1 / 8, inputs e 1 / 12 (two-level: three activation digits, |N| >= 2^24)
"""
import argparse
import json
import os
import shutil
import sys
import time

import numpy as np

from clean.rr_f.gpu import bench, build, stage
from clean.rr_f.reference.fmt import f16
from clean.rr_f.reference.stages import stage_info, weight_shapes
from clean.rr_f.tests import synth
from clean.rr_f.tests.vectors_io import load_manifest, window_consts

ALL = bench.ALL


def codes(rng, shape, elo, ehi, pzero=0.02, plo=None):
    e = rng.integers(elo, ehi + 1, size=shape)
    if plo is not None:                                  # two-level: mostly ehi, a share plo at elo
        e = np.where(rng.random(shape) < plo, elo, ehi)
    m = rng.integers(0, 8, size=shape)
    c = (e << 3) | m
    c = np.where(c >= 0x7F, 0x7E, c)
    c = np.where(rng.random(shape) < pzero, 0, c)
    s = rng.integers(0, 2, size=shape) << 7
    return (c | s).astype(np.uint8)


PROF = {"mid": ((2, 9, None), (3, 11, None)), "hi": ((4, 9, None), (5, 10, None)),
        "ovf": ((9, 13, None), (10, 14, None)), "sub": ((0, 5, None), (0, 6, None)),
        "big": ((1, 8, 0.03), (1, 12, 0.03))}


def weights(rng, name, prof):
    if prof == "net":
        return synth.stage_weights(rng, name)
    W = synth.stage_weights(rng, name)                   # biases etc. from synth, matrices replaced
    (wl, wh, wp), _ = PROF[prof]
    for t, (shape, typ) in weight_shapes(name).items():
        if typ == "e4m3":
            W[t] = codes(rng, shape, wl, wh, 0.02, wp)
    return W


def inputs(rng, name, Nx, Ny, prof):
    if prof == "net":
        return bench.inputs_for(rng, name, Nx, Ny)
    _, (xl, xh, xp) = PROF[prof]
    i = stage_info(name)
    C = i["C"]
    hid = lambda shp: synth.hidden(rng, shp)            # noqa: E731
    if i["kind"] == "fuse":
        return {"cin": codes(rng, (Ny // 2, Nx // 2, i["Kc"]), xl, xh, 0.02, xp),
                "skip": codes(rng, (Ny, Nx, C), xl, xh, 0.02, xp)}
    if i["adapter"]:
        return {"in": codes(rng, (Ny, Nx, 128), xl, xh, 0.02, xp), "hold": hid((Ny, Nx))}
    d = {"x": codes(rng, (Ny, Nx, C), xl, xh, 0.02, xp), "hold": hid((Ny, Nx))}
    if i["kind"] == "rec":
        d["Hc"] = hid((Ny // 2, Nx // 2))
    return d


def window_mask(sc, flags, shape, coarse):
    """Token mask [Ny][Nx] (or the coarse [Ny/2][Nx/2]) of the windows whose flag is clear (rr_block: window wid =
    wj nwx + wi covers X = 8 wi - sx + 0..7, Y = 8 wj - sy + 0..7; only real tokens are stored)."""
    Ny, Nx = sc.Ny, sc.Nx
    X = np.arange(Nx)
    Y = np.arange(Ny)
    wi = (X + sc.sx) // 8
    wj = (Y + sc.sy) // 8
    wid = wj[:, None] * sc.nwx + wi[None, :]
    ok = (flags.reshape(-1)[wid] & 0xFF) == 0
    if coarse:
        ok = ok[0::2, 0::2] & ok[1::2, 0::2] & ok[0::2, 1::2] & ok[1::2, 1::2]
    return ok


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--render", default="512x512")
    ap.add_argument("--profiles", default="net")
    ap.add_argument("--stages", default=",".join(ALL))
    ap.add_argument("--seed", type=int, default=5)
    ap.add_argument("--nofix", action="store_true")
    ap.add_argument("--istep", default="I", help="e.g. I+RR_ICOV=1 (coverage bits above 0xff, kernels/rr_step_i.hlsli)")
    ap.add_argument("--neg", action="store_true", help="negative control: one input code of the I call changed")
    ap.add_argument("--json")
    a = ap.parse_args(argv)
    rw, rh = (int(v) for v in a.render.split("x"))
    Wt, Ht = -(-rw // 32) * 32, -(-rh // 32) * 32
    wc = window_consts(load_manifest(), "A")
    rows, bad_total = [], 0
    for prof in a.profiles.split(","):
        for name in a.stages.split(","):
            rng = np.random.default_rng([a.seed, ALL.index(name), list(PROF).index(prof) + 1 if prof in PROF else 0])
            lv = stage_info(name)["level"]
            Nx, Ny = Wt >> lv, Ht >> lv
            W = weights(rng, name, prof)
            inp = inputs(rng, name, Nx, Ny, prof)
            consts = {"ra": f16(0.75), "m": f16(0.5), "f": f16(1.0), "g": f16(1.25)}
            shift = (int(rng.integers(0, 4)) * 2, int(rng.integers(0, 4)) * 2)
            ar = build.Arena()
            scx = stage.StageCall(name, inp, W, consts, wc, shift, arena=ar, step="X")
            if a.neg:                                    # flip the sign bit of one code in the middle of the I input
                inp = {k: v.copy() for k, v in inp.items()}
                key = "skip" if "skip" in inp else "in" if "in" in inp else "x"
                inp[key][Ny // 2, Nx // 2, 0] ^= 0x80
                if (inp[key][Ny // 2, Nx // 2, 0] & 0x7F) == 0:
                    inp[key][Ny // 2, Nx // 2, 0] = 0x38
            sci = stage.StageCall(name, inp, W, consts, wc, shift, arena=ar, step=a.istep)
            kernels, disp = {}, []
            for sc, st in ((scx, "X"), (sci, a.istep)):
                for kid, kern, g, args in stage.dispatches_for(sc, "exact", st, 1, f"{name.replace('-', '')}_{st}"):
                    if a.nofix and kid.endswith("_fix"):
                        continue
                    kernels[kid] = kern
                    disp.append((kid, g, args))
            wd = os.path.join(build.BUILD, "runs", "step_cmp")
            t0 = time.time()
            img, _ = build.run(ar, kernels, disp, workdir=wd)
            shutil.rmtree(wd, ignore_errors=True)
            ox, oi = scx.outputs(img), sci.outputs(img)
            flags = img[sci.pYflag:sci.pYflag + sci.groups() * 4].view(np.uint32)
            nflag = int(((flags & 0xFF) != 0).sum())
            cov = {hex(b): int(((flags & 0xFF) == 0).__and__((flags & b) != 0).sum()) for b in (0x100, 0x200, 0x400, 0x800)}
            diff, compared = {}, {}
            for k in ox:
                d = ox[k].view(np.uint8) != oi[k].view(np.uint8)
                d = d.reshape(ox[k].shape[0], ox[k].shape[1], -1).any(axis=2)
                if a.nofix:
                    m = window_mask(sci, flags, d.shape, coarse=(d.shape[0] != sci.Ny))
                    diff[k] = int((d & m).sum())
                    compared[k] = int(m.sum())
                else:
                    diff[k] = int(d.sum())
                    compared[k] = int(d.size)
            bad = any(diff.values())
            bad_total += bad
            r = dict(profile=prof, stage=name, render=[rw, rh], tokens=[Nx, Ny], shift=shift, i_ok=bool(sci.i_ok),
                     windows=sci.groups(), flagged=nflag, fixups=sci.fixups(img) if not a.nofix else None,
                     covered_unmarked_windows=cov, diff_tokens=diff, compared_tokens=compared, seconds=round(time.time() - t0, 1))
            rows.append(r)
            print(f"{prof:4s} {name:5s} i_ok={r['i_ok']} flagged {nflag}/{r['windows']} fixups={r['fixups']} "
                  f"diff={diff} compared={compared} cov={cov} ({r['seconds']} s)", flush=True)
    print(f"stage runs {len(rows)}, with byte differences {bad_total}")
    if a.json:
        with open(a.json, "w") as f:
            json.dump(dict(rows=rows, bad=bad_total), f, indent=1)
    return 1 if bad_total else 0


if __name__ == "__main__":
    sys.exit(main())
