# SPDX-License-Identifier: MIT
"""Diagnostics of rr_resolve_fast on a real model: which check sends pixels to rr_resolve_fixup.

Generates a variant of rr_resolve_fast_main.hlsli (kernels/_why_*.hlsl*, removed afterwards) in which every
`slow = ...` site k records bit k of a per-pixel mask; each slow pixel adds 1 to counter k for every bit set (the mask
is cumulative, so the first site where the count jumps is the cause). Runs one steady bench frame (no timing).

    python -m clean.rr_f.gpu.resolve_why --bench 1706x960:2560x1440 --model <wgcm> --map ""
"""
import argparse
import re
from pathlib import Path

import numpy as np

from . import build
from . import resolve_fast as GRF
from . import sequence as S

KD = Path(build.KDIR)


def gen():
    src = (KD / "rr_resolve_fast_main.hlsli").read_text(encoding="utf-8").splitlines()
    out, sites, k, i = [], [], 0, 0
    while i < len(src):
        ln = src[i]
        out.append(ln)
        if re.search(r"\bslow = (or\(slow,|!tok;)", ln) and "RF_MUT" not in src[i - 1]:
            j = i
            while not src[j].split("//")[0].rstrip().endswith(";"):
                j += 1
                out.append(src[j])
            ind = re.match(r"\s*", ln).group(0)
            out.append(f"{ind}g_why |= slow ? {1 << k}u : 0u;")
            sites.append((i + 1, ln.strip()[:110]))
            k += 1
            i = j
        if "wg_mem.InterlockedOr(bm + (pix >> 5)" in ln:
            out.append("                [unroll] for (uint wk = 0; wk < 32u; ++wk) if ((g_why >> wk) & 1u) "
                       "wg_mem.InterlockedAdd(aux + 4u * wk, 1u, ob);")
        i += 1
    txt = "\n".join(out).replace("bool slow = !tok;", "bool slow = !tok; g_why = 0u;", 1)
    (KD / "_why_main.hlsli").write_text("static uint g_why;\n" + txt + "\n", encoding="utf-8")
    top = (KD / "rr_resolve_fast.hlsl").read_text(encoding="utf-8")
    (KD / "_why_fast.hlsl").write_text(top.replace('#include "rr_resolve_fast_main.hlsli"',
                                                   '#include "_why_main.hlsli"'), encoding="utf-8")
    return sites


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", default="1706x960:2560x1440")
    ap.add_argument("--model", default=S.DEFAULT_MODEL)
    ap.add_argument("--map", default=S.DEFAULT_MAP)
    ap.add_argument("--step", default="E")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--b2", action="store_true")
    a = ap.parse_args()
    sites = gen()
    try:
        kx = build.compile_kernel("_why_fast.hlsl", "rr_resolve_fast",
                                  {"WG_BLOCK": 256, "RF_COUNT": 0, "RF_PROF": 0, "RF_SPEC": 1, "WG_WAVE": GRF.WAVE},
                                  tag="rr_resolve_fast_why")
    finally:
        for n in ("_why_main.hlsli", "_why_fast.hlsl"):
            (KD / n).unlink(missing_ok=True)
    m = S.load_model(a)
    r, o = a.bench.split(":")
    render = tuple(int(v) for v in r.split("x"))
    O = tuple(int(v) for v in o.split("x"))
    b = S.Builder(m, O, render, step=a.step, zmode="exact", debug=False, input_exact=True, resolve_exact=True,
                  steady=True, output_fast=True, resolve_kernel="fast", pool_kernel="tables", head_mode="default")
    b.add_frame(render)
    aux = b.ar.reserve(32 * 4)
    b.kernels["resolve-why"] = (kx, GRF.NARGS)
    disp = []
    for d in b.disp:
        if d[0] == "resolve-fast":
            d = ("resolve-why", d[1], [d[2][0], aux, 0])
        disp.append(d)
    wd = str(Path(build.BUILD) / "why")
    img, _ = build.run(b.ar, b.kernels, disp, require_double=True, workdir=wd, reps=0, timeout=3000)
    c = img[aux: aux + 128].view("<u4")
    We, He = O
    print(f"output {We}x{He} = {We * He} px; cumulative slow count after each site:")
    prev = 0
    for (ln, t), v in zip(sites, c):
        print(f"  L{ln:4d} {int(v):9d} (+{int(v) - prev:8d})  {t}")
        prev = int(v)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
