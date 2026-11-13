# SPDX-License-Identifier: MIT
"""Fast-path rates of variant Z on one steady bench frame (diagnostics; the frame's results are not checked or used).

Builds the frame of `sequence --bench` with step Z and the kernels' RR_ZSTAT build, runs it once without the fixup
dispatches and prints, per body stage, the share of windows in which some wave left the fast path at each site
(kernels/rr_block.hlsl RR_ZSTAT), the share of the waves' decisions at each site that took the slow path ("wave ...")
and the mean count of slow score tiles per window.

    python -m clean.rr_f.gpu.zstat 1706x960:2560x1440 [--synthetic] [--json out.json]"""
import argparse
import json
import os
import sys

import numpy as np

from clean.rr_f.gpu import build, sequence as SQ, stage

SITES = ((1, "front"), (2, "x (q/v)"), (4, "scores"), (8, "O"), (16, "Dp"), (32, "W1"), (64, "W2"), (128, "merge"),
         (256, "Wh"), (0x8000, "Y flag"))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("size", help="render:output")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--model", default=SQ.DEFAULT_MODEL)
    ap.add_argument("--map", default=SQ.DEFAULT_MAP)
    ap.add_argument("--json")
    ap.add_argument("--step", default="Z", choices=("Z", "Z2"))
    a = ap.parse_args(argv)
    a.b2 = False
    if "WG_RR_BUILD" not in os.environ:
        ap.error("WG_RR_BUILD must name a private build directory")
    orig = stage.stage_defines

    def defines(name, zmode, step="E", ldsw=1):
        d = orig(name, zmode, step, ldsw)
        if step == a.step:
            d["RR_ZSTAT"] = 1
        return d

    stage.stage_defines = defines
    m = SQ.load_model(a)
    render_s, out_s = a.size.split(":")
    render = tuple(int(v) for v in render_s.split("x"))
    O = tuple(int(v) for v in out_s.split("x"))
    b = SQ.Builder(m, O, render, step=a.step, zmode="exact", debug=False, steady=True)
    b.add_frame(render)
    disp = [d for d in b.disp if not d[0].endswith("_fix")]
    img, _ = build.run(b.ar, b.kernels, disp, require_double=True, timeout=3000,
                       workdir=os.path.join(build.BUILD, "runs", "zstat"))
    rep = {}
    for name, sc in b.frames[-1]["calls"].items():
        if not hasattr(sc, "pYflag"):
            continue
        g = sc.groups()
        w = img[sc.pYflag:sc.pYflag + 4 * g].view(np.uint32)
        row = {nm: round(float(((w & bit) != 0).mean()), 4) for bit, nm in SITES}
        cnt = img[sc.pYflag + 4 * (g + 1):sc.pYflag + 4 * (g + 1) + 80].view(np.uint32)
        for bit, nm in SITES[:-1]:                           # per wave decision: share on the slow path
            i = bit.bit_length() - 1
            if cnt[2 * i]:
                row["wave " + nm] = round(float(cnt[2 * i + 1]) / float(cnt[2 * i]), 4)
        row["slow score tiles / window"] = round(float((w >> 16).mean()), 2)
        row["windows"] = int(g)
        rep[name] = row
        print(f"{name:6s} " + " ".join(f"{k}={v}" for k, v in row.items() if v), flush=True)
    if a.json:
        with open(a.json, "w") as f:
            json.dump(rep, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
