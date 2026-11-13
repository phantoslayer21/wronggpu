# SPDX-License-Identifier: MIT
"""GPU time of the output stage alone (fast path and / or rr_output_main) at game sizes on synthetic inputs.

    python -m clean.rr_f.gpu.output_bench --size 1706x960:2560x1440 [--path fast,main] [--consts model|synthetic]

Inputs: output_diff.make (smooth data), the model file's or the sequence's synthetic constants. Timing: build.run
under the GPU lock, best / mean of --reps runs per dispatch. The sequence bench (sequence.py --bench) gives the
numbers of the connected frame.
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np

from . import build
from . import output as GO
from . import output_diff as OD


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--size", default="1706x960:2560x1440")
    ap.add_argument("--path", default="fast")
    ap.add_argument("--consts", default="model", choices=("model", "synthetic"))
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--defines", default="", help="extra kernel defines K=V,K=V")
    ap.add_argument("--variants", default="", help="fast-path variants timed in the same job: name:K=V,K=V;name2:...")
    ap.add_argument("--f32-inputs", action="store_true", help="decoded f32 motion / history (default binary16)")
    ap.add_argument("--no-diag", action="store_true", help="packed_head bit 3: no linear / trace outputs")
    ap.add_argument("--json")
    a = ap.parse_args()
    if "WG_RR_BUILD" not in os.environ:
        ap.error("set WG_RR_BUILD")
    from . import sequence as SQ
    c = (SQ.model_from_file(SQ.DEFAULT_MODEL, SQ.DEFAULT_MAP) if a.consts == "model" else SQ.model_synthetic()).osym
    r, o = a.size.split(":")
    render = tuple(int(v) for v in r.split("x"))
    output = tuple(int(v) for v in o.split("x"))
    rng = np.random.default_rng(5454)
    inp = OD.make(rng, render, output, c, "smooth")
    defines = dict(kv.split("=") for kv in a.defines.split(",") if kv)
    paths = a.path.split(",")
    k, disp = {}, []
    ar = build.Arena()
    args, _, _ = GO.prepare(ar, inp, packed=True, half_inputs=not a.f32_inputs)
    if a.no_diag:
        args[37] |= 8
    variants = [(p, defines, p == "fast", "") for p in paths]
    for v in filter(None, a.variants.split(";")):
        name, _, d = v.partition(":")
        # name:K=V,...|f32 : decoded f32 motion / history
        d, _, opt = d.partition("|")
        variants.append((name, {**defines, **dict(kv.split("=") for kv in d.split(",") if kv)}, True, opt))
    paths = [v[0] for v in variants]
    f32buf = None
    for p, d, fast, f32 in variants:
        k.update({f"{p}-{n}": v for n, v in GO.kernel_set(d, fast=fast).items()})
        aa = list(args)
        if f32 == "f32":
            f32buf = f32buf or [ar.add(np.asarray(inp[n], "<f4")) for n in ("PM", "Pe", "Pn", "Hprev")]
            aa[4:8] = f32buf
            aa[37] &= ~36
        if not fast:
            aa[38] = 0
        disp += GO.dispatch_list(aa, fast=fast, prefix=p + "-")
    img, log = build.run(ar, k, disp, require_double=True, reps=a.reps, timeout=3000,
                         workdir=str(Path(build.BUILD) / "output-bench"))
    per = []
    for line in log.splitlines():
        f = line.split()
        if line.startswith("time "):
            per.append(dict(kernel=f[2], best_ms=float(f[4]), mean_ms=float(f[7])))
    for p in per:
        print(f"{p['kernel']:20s} {p['best_ms']:8.3f} ms best {p['mean_ms']:8.3f} ms mean")
    scr = args[38]
    print("fallback pixels", int(img[scr + 8:scr + 12].view("<u4")[0]), "prep flags", int(img[scr + 12:scr + 16].view("<u4")[0]))
    for p in paths:
        print(p, "total", round(sum(x["best_ms"] for x in per if x["kernel"].startswith(p + "-")), 3), "ms")
    if a.json:
        Path(a.json).write_text(json.dumps(dict(size=a.size, consts=a.consts, per_dispatch=per), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
