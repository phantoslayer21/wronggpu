# SPDX-License-Identifier: MIT
"""All vector cases (D0-R excluded as in stage.main) with several steps, the first one the baseline (e.g. X,I): every run
against the reference (stage.compare: e4m3 codes, zero codes equal; f16 bits, zero signs equal) and every run's raw
output bytes against the baseline step's (zero signs included), plus the fixup windows of steps with a fixup dispatch.

    python -m clean.rr_f.gpu.vec_steps X,I [--json out.json] [--stages IN0,E1] [--nolock]"""
import argparse
import json
import os
import sys

import numpy as np

from clean.rr_f.gpu import build, stage
from clean.rr_f.tests.vectors_io import DEFAULT_DIR, load_case, load_manifest


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("steps")
    ap.add_argument("--json")
    ap.add_argument("--stages")
    ap.add_argument("--nolock", action="store_true")
    a = ap.parse_args(argv)
    steps = a.steps.split(",")
    man = load_manifest()
    sel = [c for c in man["cases"] if not c["id"].startswith("D0-R")]
    if a.stages:
        sel = [c for c in sel if c["stage"] in a.stages.split(",")]
    stage.precompile([(c["stage"], "exact", st, 1) for st in steps for c in sel])
    ar = build.Arena()
    kernels, disp, calls = {}, [], []
    for st in steps:
        for case in sel:
            name, inputs, W, consts, wc, shift_row, expected, inter = load_case(man, case, DEFAULT_DIR)
            sc = stage.StageCall(name, inputs, W, consts, wc, shift_row, Z=None, arena=ar, step=st)
            kid = name.replace("-", "") + "_exact_" + st
            for k, kern, g, args in stage.dispatches_for(sc, "exact", st, 1, kid):
                kernels.setdefault(k, kern)
                disp.append((k, g, args))
            calls.append((st, case, sc, expected))
    img, _ = build.run(ar, kernels, disp, lock=not a.nolock, workdir=os.path.join(build.BUILD, "runs", "vec_steps"))
    base, rows, bad_ref, bad_bytes = {}, [], 0, 0
    for st, case, sc, expected in calls:
        got = sc.outputs(img)
        ref = {k: stage.compare(k, got[k], expected[k])["diff"] for k in got}
        if st == steps[0]:
            base[case["id"]] = got
        byt = {k: int((got[k].view(np.uint8) != base[case["id"]][k].view(np.uint8)).sum()) for k in got}
        r = dict(case=case["id"], step=st, ref_diff=ref, bytes_vs_base=byt, y_ok=bool(sc.y_ok), i_ok=bool(sc.i_ok),
                 fixups=sc.fixups(img), windows=sc.groups())
        bad_ref += any(ref.values())
        bad_bytes += any(byt.values())
        rows.append(r)
        print(f"{case['id']:22s} {st} ref={'OK' if not any(ref.values()) else ref} "
              f"bytes={'same' if not any(byt.values()) else byt} i_ok={r['i_ok']} fixups={r['fixups']}/{r['windows']}",
              flush=True)
    print(f"runs {len(rows)}: not exact vs reference {bad_ref}, bytes differing from {steps[0]} {bad_bytes}")
    if a.json:
        with open(a.json, "w") as f:
            json.dump(dict(rows=rows, bad_ref=bad_ref, bad_bytes=bad_bytes), f, indent=1)
    return 1 if bad_ref or bad_bytes else 0


if __name__ == "__main__":
    sys.exit(main())
