# SPDX-License-Identifier: MIT
"""Connected synthetic GPU body/head -> page-54 output stage -> page-54b resolve and quarter pooled history.

One arena, one submission. The resolve consumes the output stage's Q (binary16 RGB, Wh x Hh) and PH1 (RGB10A2
codes) and the head's packed PREC (half4) / GATES (byte4) directly by arena offset; the motion MV is the
output stage's PM input buffer. Two resolve frames run in sequence (TH / AH created on the first, read on the
second) with a layer, alpha upscaling and a separate output alpha; then the quarter pooling of PH1 (the start of
the following frame). Every resolve result is compared bit-exactly with reference/resolve.py evaluated on the
GPU's own upstream buffers (the upstream stages are qualified by gpu/pipeline.py within their own tolerances).
--mutation disconnects Q (the resolve reads a different, poisoned buffer): the comparison must then fail.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from . import build, network, pipeline
from . import resolve as gres
from ..reference import resolve as rr
from ..tests.test_resolve import symbols


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", required=True)
    ap.add_argument("--mutation", action="store_true")
    args = ap.parse_args()
    if "WG_RR_BUILD" not in os.environ:
        ap.error("set WG_RR_BUILD to a private build directory")
    ar, kernels, dispatches, calls, reference, base, ha, body_bindings, regroup = network.prepare()
    a = pipeline.output_inputs(reference["head"])
    rng = np.random.default_rng(5454)
    hr, wr = a["PM"].shape[:2]
    a["PM"] = rng.uniform(-2, 2, a["PM"].shape).astype(np.float16)    # non-zero motion for MV (54b.3)
    out_args, offs, sizes, expected_bindings, _ = pipeline.append_output(ar, kernels, dispatches, a, base, ha)
    q_off, ph1_off = offs[2], offs[3]
    pm_off, p_off, g_off = out_args[4], out_args[2], out_args[1]
    out_size = a["output_size"]
    sz = rr.sizes(out_size)
    sym = symbols()
    defines = {"WG_BLOCK": gres.BLOCK, "RR_ELEM_EXACT": 1, "RR_MUT": 0}
    for name in ("main", "pool"):
        kernels["resolve-" + name] = (build.compile_kernel("rr_resolve.hlsl", "rr_resolve_" + name, defines,
                                                           tag=f"rr_resolve_{name}_e1m0"), gres.NARGS)
    poison = ar.add(np.full(sz.Wh * sz.Hh * 3, 0x7E00, np.uint16))

    def bound(off, shape, fmt, ch, origin=(0, 0)):
        u = gres.Upload.__new__(gres.Upload)
        h, w = shape
        u.shape, u.fmt, u.origin, u.packed = (h, w, ch), fmt, origin, False
        u.off, u.nbytes = off, h * w * ch * {"f32": 4, "f16": 2, "unorm8": 1, "rgb10a2": 2}[fmt]
        u.desc = [off, w, h, w, {"f32": 0, "f16": 1, "unorm8": 2, "rgb10a2": 4}[fmt], ch, origin[0], origin[1]]
        return u

    Q_up = bound(poison if args.mutation else q_off, (sz.Hh, sz.Wh), "f16", 3)
    MV_up = bound(pm_off, (hr, wr), "f32", 4)
    P_up = bound(p_off, (hr, wr), "f16", 4)
    G_up = bound(g_off, (hr, wr), "unorm8", 4)
    TH = [gres.Upload(ar, np.zeros((sz.He, sz.We, 4), np.float16), "f16") for _ in range(2)]
    AH = [gres.Upload(ar, np.zeros((sz.He, sz.We, 1), np.float16), "f16") for _ in range(2)]
    layer = rr.Surface(rng.uniform(0, 1, (hr + 1, wr + 2, 4)).astype(np.float16), "f16", (1, 1))
    codes = rng.integers(0, 1024, (hr, wr, 4)).astype(np.uint16)
    codes[..., 3] &= 3
    colour_in = rr.Surface(codes, "rgb10a2")
    frames = []
    for n, (jit, E, Ep) in enumerate((((0.125, -0.25), 1.25, 0.75), ((-0.375, 0.0625), 1.0, 1.25))):
        dest = rr.Surface(np.zeros((sz.He, sz.We, 4), np.float16), "f16")
        aout = rr.Surface(np.zeros((sz.He, sz.We, 1), np.float32), "f32")
        ups = dict(Q=Q_up, OUT=gres.surface_upload(ar, dest), AOUT=gres.surface_upload(ar, aout),
                   LAYER=gres.surface_upload(ar, layer), MV=MV_up, P=P_up, G=G_up,
                   ASRC=gres.surface_upload(ar, colour_in), THr=TH[n % 2], THw=TH[1 - n % 2],
                   AHr=AH[n % 2], AHw=AH[1 - n % 2])
        flags = gres.F_EXT | gres.F_LAYER | gres.F_ALPHA | gres.F_AOUT | (gres.F_TRESET | gres.F_ARESET if n == 0 else 0)
        p = gres.params(sym, sizes=sz, render=(wr, hr), flags=flags, E=np.float32(E), Ep=np.float32(Ep),
                        jitter=tuple(np.float32(jit)), layer_origin=layer.origin, alpha_channel=3, res=ups)
        dispatches.append(("resolve-main", (sz.We * sz.He + gres.BLOCK - 1) // gres.BLOCK, [ar.add(p), 0, 0]))
        frames.append((n, jit, E, Ep, dest, aout, ups))
    (h1, w1), (h2, w2) = rr.pooled_sizes(out_size)
    PQ = gres.Upload(ar, np.zeros((h2, w2, 4), np.uint16), "rgb10a2", packed=False)
    PH1_up = bound(ph1_off, (h1, w1), "rgb10a2", 4)
    pp = gres.params(sym, sizes=sz, render=(1, 1), flags=0, res=dict(PH1=PH1_up, PQ=PQ))
    dispatches.append(("resolve-pool", (w2 * h2 + gres.BLOCK - 1) // gres.BLOCK, [ar.add(pp), 0, 0]))
    print(f"Running {len(dispatches)} connected GPU dispatches", flush=True)
    img, log = build.run(ar, kernels, dispatches, require_double=True,
                         workdir=str(Path(build.BUILD) / ("pipe-resolve-mut" if args.mutation else "pipe-resolve")))
    body = network.analyze(img, calls, reference, base, ha, regroup)
    # upstream buffers as the GPU produced them
    Qg = img[q_off:q_off + sz.Wh * sz.Hh * 6].copy().view("<f2").reshape(sz.Hh, sz.Wh, 3).astype(np.float64)
    PH1g = img[ph1_off:ph1_off + h1 * w1 * 8].copy().view("<u2").reshape(h1, w1, 4)
    headg = pipeline.head_from_gpu(img, base, ha)
    Pg = headg["PREC"][..., :3].astype(np.float64)
    Gg = headg["GATES_code"][..., :3]
    hist = rr.ResolveHistories(out_size, alpha_upscaling=True)
    checks = []
    for n, jit, E, Ep, dest, aout, ups in frames:
        res = rr.resolve(Qg, output_size=out_size, render_size=(wr, hr), E=np.float32(E), Ep=np.float32(Ep),
                         jitter=tuple(np.float32(jit)), reset=n == 0, parity=n % 2, symbols=sym, colour_out=dest,
                         histories=hist, alpha_out=aout, layer=layer, colour_in=colour_in,
                         motion=np.asarray(a["PM"], np.float64), P=Pg, G=Gg)
        hist = res.histories
        for label, up, want in (("colour", ups["OUT"], res.colour.data), ("alpha_out", ups["AOUT"], res.alpha.data)):
            ok, bad = gres.same_bits(up.read(img), want)
            checks.append(dict(frame=n, what=label, values=int(want.size), mismatches=bad, exact=ok))
    for i in range(2):
        for label, up, want in ((f"TH[{i}]", TH[i], hist.TH[i]), (f"AH[{i}]", AH[i], hist.AH[i])):
            ok, bad = gres.same_bits(up.read(img), want)
            checks.append(dict(frame="final", what=label, values=int(want.size), mismatches=bad, exact=ok))
    want_pq = rr.quarter_pool(PH1g, np.zeros((h2, w2, 4), np.uint16), output_size=out_size, reset=False, symbols=sym)
    ok, bad = gres.same_bits(PQ.read(img), want_pq)
    checks.append(dict(frame="next", what="PQ", values=int(want_pq.size), mismatches=bad, exact=ok))
    resolve_ok = all(c["exact"] for c in checks)
    nontrivial = bool(np.isfinite(Qg).mean() > 0.5 and np.any(Qg != 0))
    passed = bool(body["passed"] and resolve_ok and nontrivial and not args.mutation)
    report = dict(passed=passed, mutation=args.mutation, mutation_detected=bool(args.mutation and not resolve_ok),
                  body_passed=body["passed"], resolve_checks=checks, q_nontrivial=nontrivial,
                  bindings=dict(Q=q_off, PH1=ph1_off, PM=pm_off, PREC=p_off, GATES=g_off,
                                expected_head=expected_bindings),
                  dispatches=[d[0] for d in dispatches], host_log=log,
                  sources={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in
                           (Path(__file__), Path(gres.__file__), Path(build.KDIR) / "rr_resolve.hlsl")},
                  scope="one prepared synthetic frame (two resolve calls on its Q); live GPU body/head -> output -> "
                        "resolve/PQ; invented symbols; no input stage, player model or game")
    Path(args.json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.json).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(dict(passed=passed, body=body["passed"], checks=checks), indent=1), flush=True)
    if args.mutation:
        return 0 if report["mutation_detected"] else 2
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
