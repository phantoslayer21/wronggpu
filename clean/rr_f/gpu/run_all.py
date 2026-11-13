# SPDX-License-Identifier: MIT
"""One GPU session for the spec 51 kernels: probes P1 (WMMA accumulation), P2-P4 (Q8, attention weight,
activation, R(1/Z) for every f16 input) and every stage case of the vectors in the chosen Z modes, all in one arena and
one host call (one GPU lock).

    python -m clean.rr_f.gpu.run_all [--z supplied,exact,f32] [--stages E2,...] [--no-probes] [--json out.json]"""
import argparse
import json
import os
import sys
import time

from clean.rr_f.gpu import build, probe_p1, probe_p4, stage
from clean.rr_f.tests.vectors_io import load_manifest


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--z", default="supplied,exact,f32")
    ap.add_argument("--step", default="E", help="E (two WMMAs per step) and / or X (exact step), comma list")
    ap.add_argument("--ldsw", default="1", help="0 and / or 1 (weight units staged through LDS), comma list")
    ap.add_argument("--stages")
    ap.add_argument("--cases")
    ap.add_argument("--no-probes", action="store_true")
    ap.add_argument("--nolock", action="store_true")
    ap.add_argument("--json")
    ap.add_argument("--tag", default="all")
    a = ap.parse_args(argv)
    man = load_manifest()
    if a.cases:
        sel = set(a.cases.split(","))
    elif a.stages:
        st = set(a.stages.split(","))
        sel = {c["id"] for c in man["cases"] if c["stage"] in st}
    else:
        sel = {c["id"] for c in man["cases"]}
    ar = build.Arena()
    kernels, disp, an = {}, [], []
    if not a.no_probes:
        for mod in (probe_p1, probe_p4):
            k, d, f = mod.prepare(ar)
            kernels.update(k)
            disp += d
            an.append(f)
    k, d, fs = stage.prepare(ar, sel, a.z.split(","), steps=a.step.split(","),
                             ldsws=[int(v) for v in a.ldsw.split(",")])
    kernels.update(k)
    disp += d
    t0 = time.time()
    img, log = build.run(ar, kernels, disp, lock=not a.nolock, workdir=os.path.join(build.BUILD, "runs", a.tag))
    print(f"GPU run done ({time.time() - t0:.0f} s incl. lock wait): {log.strip()}", flush=True)
    for f in an:
        f(img)
    res = fs(img)
    if a.json:
        with open(a.json, "w") as f:
            json.dump(res, f, indent=1)
    bad = [r for r in res if not r["exact"]]
    print(f"{len(res) - len(bad)} / {len(res)} stage runs identical")
    return 0


if __name__ == "__main__":
    sys.exit(main())
