# SPDX-License-Identifier: MIT
"""Shader-clock profile of rr_output_fast's sections (OF_PROF=1; lane 0 of each wave adds the clock deltas of
sections 1-7 to trace words 1-7; diagnostics off). Profiling build only (the trace words are not outputs).

    python -m clean.rr_f.gpu.output_prof [--size 1706x960:2560x1440] [--defines K=V,...]
"""
import argparse
from pathlib import Path

import numpy as np

from . import build
from . import output as GO
from . import output_diff as OD

NAMES = ("", "LDS fill", "16-tap exponents", "sums+current+cand", "cand weights", "reprojections", "blend+store", "PH1")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--size", default="1706x960:2560x1440")
    ap.add_argument("--defines", default="")
    a = ap.parse_args()
    from . import sequence as SQ
    c = SQ.model_from_file(SQ.DEFAULT_MODEL, SQ.DEFAULT_MAP).osym
    r, o = a.size.split(":")
    render = tuple(int(v) for v in r.split("x"))
    output = tuple(int(v) for v in o.split("x"))
    inp = OD.make(np.random.default_rng(5454), render, output, c, "smooth")
    ar = build.Arena()
    args, _, _ = GO.prepare(ar, inp, packed=True, half_inputs=True)
    args[37] |= 8
    d = {"OF_PROF": 1, **{k: int(v) for k, v in (kv.split("=") for kv in a.defines.split(",") if kv)}}
    k = {f"p-{n}": v for n, v in GO.kernel_set(d).items()}
    img, log = build.run(ar, k, GO.dispatch_list(args, prefix="p-"), require_double=True, reps=0, timeout=3000,
                         workdir=str(Path(build.BUILD) / "output-prof"))
    t = img[args[13]:args[13] + 32].view("<u4").astype(np.float64)
    tot = t[1:8].sum()
    for i in range(1, 8):
        print(f"{NAMES[i]:22s} {t[i] / tot * 100:5.1f} %")


if __name__ == "__main__":
    main()
