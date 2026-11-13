# SPDX-License-Identifier: MIT
"""Page 54b.5 quarter pool (PH2) through exact tables (kernels/rr_pool.hlsl): host side, qualification, exhaustive check.

The tables (per model: they depend on the symbols only) are built on the GPU by rr_pool_tables with the exact
elementary functions of rr_resolve_math.hlsli; rr_pool_main then needs no exp2 / log2 per texel. Usage (repository
root, WG_RR_BUILD set to a private build folder; every GPU call through the GPU lock):

    python -m clean.rr_f.gpu.pool --json pool.json [--model <model.wgcm> ...] [--synthetic] [--no-exhaustive]

Per model: fixtures of several output sizes against reference/resolve.quarter_pool (bit-exact, also against the direct
kernel rr_resolve_pool), and the code table against the direct chain for all 2^32 f32 patterns of the mean."""
import argparse
import dataclasses
import json
import os
from pathlib import Path

import numpy as np

from . import build
from . import resolve as GR
from ..reference import resolve as rr

BLOCK = 256
TPL = 4                       # PH2 texels per lane (RR_POOL_TPL)
NARGS = 4
TABLE_BYTES = 2050 * 4


def kernels(exact=True):
    e = int(exact)
    d = {"WG_BLOCK": BLOCK, "RR_ELEM_EXACT": e}
    return {"pool-tables": (build.compile_kernel("rr_pool.hlsl", "rr_pool_tables", d, tag=f"rr_pool_tables_e{e}"), NARGS),
            "pool-main": (build.compile_kernel("rr_pool.hlsl", "rr_pool_main", {**d, "RR_POOL_TPL": TPL},
                                               tag=f"rr_pool_main_e{e}"), NARGS)}


def table_dispatch(arena, sym):
    """(table offset, the build dispatch): the build needs only the symbols of the parameter block."""
    tab = arena.reserve(TABLE_BYTES, fill=0xEE)
    p = GR.params(sym, sizes=rr.sizes((2, 2)), render=(1, 1), flags=0)
    return tab, ("pool-tables", 1024 // BLOCK, [arena.add(p), tab, 0, 0])


def pool_dispatch(prm, tab, w2, h2):
    return ("pool-main", (w2 * h2 + BLOCK * TPL - 1) // (BLOCK * TPL), [prm, tab, 0, 0])


def fixtures(arena, disp, sym, rng, tab):
    recs = []
    for out_size in ((2, 2), (5, 3), (9, 7), (37, 21), (64, 36), (301, 157)):
        (h1, w1), (h2, w2) = rr.pooled_sizes(out_size)
        PH1 = rng.integers(0, 1024, (h1, w1, 4)).astype(np.uint16)
        PH1[..., 3] = 3
        edge = np.array([[0, 1023, 1], [1022, 2, 511], [512, 513, 1000], [0, 0, 0],
                         [1023, 1023, 1023], [3, 5, 7], [700, 701, 702]], np.uint16)
        k = min(len(edge), h1 * w1)
        PH1.reshape(-1, 4)[:k, :3] = edge[:k]
        PQ = rng.integers(0, 1024, (h2 + 1, w2 + 2, 4)).astype(np.uint16)
        PQ[..., 3] = 3
        want = rr.quarter_pool(PH1, PQ, output_size=out_size, reset=False, symbols=sym)
        sz = rr.sizes(out_size)
        u1 = GR.Upload(arena, PH1, "rgb10a2", packed=False)
        new = GR.Upload(arena, PQ, "rgb10a2", packed=False)
        old = GR.Upload(arena, PQ, "rgb10a2", packed=False)
        disp.append(pool_dispatch(arena.add(GR.params(sym, sizes=sz, render=(1, 1), flags=0, res=dict(PH1=u1, PQ=new))),
                                  tab, w2, h2))
        disp.append(("direct", (w2 * h2 + GR.BLOCK - 1) // GR.BLOCK,
                     [arena.add(GR.params(sym, sizes=sz, render=(1, 1), flags=0, res=dict(PH1=u1, PQ=old))), 0, 0]))
        recs.append((out_size, new, old, want))
    return recs


def check_model(name, sym, exact, exhaustive, rows, control=None):
    rng = np.random.default_rng(5450)
    ar = build.Arena()
    kern = kernels(exact)
    kern["direct"] = (build.compile_kernel("rr_resolve.hlsl", "rr_resolve_pool",
                                           {"WG_BLOCK": GR.BLOCK, "RR_ELEM_EXACT": int(exact), "RR_MUT": 0},
                                           tag=f"rr_resolve_pool_e{int(exact)}m0"), GR.NARGS)
    tab, td = table_dispatch(ar, sym)
    disp = [td]
    recs = fixtures(ar, disp, sym, rng, tab)
    out = None
    if exhaustive:
        kern["pool-check"] = (build.compile_kernel("rr_pool.hlsl", "rr_pool_check",
                                                   {"WG_BLOCK": BLOCK, "RR_ELEM_EXACT": int(exact)},
                                                   tag=f"rr_pool_check_e{int(exact)}"), NARGS)
        out = ar.reserve(16)
        chunk = 1 << 24
        for b in range(0, 1 << 32, chunk):
            disp.append(("pool-check", chunk // BLOCK, [td[2][0], tab, b, out]))
        if control is not None:
            # negative control: this model's table against changed symbols (2^28 patterns from 1.0 up)
            nc = ar.reserve(16)
            disp.append(("pool-check", (1 << 28) // BLOCK,
                         [ar.add(GR.params(control, sizes=rr.sizes((2, 2)), render=(1, 1), flags=0)), tab, 0x3f800000, nc]))
    img, _ = build.run(ar, kern, disp, require_double=True, timeout=36000,
                       workdir=str(Path(build.BUILD) / f"pool-{name}"))
    ok = True
    for out_size, new, old, want in recs:
        a, na = GR.same_bits(new.read(img), want)
        b, nb = GR.same_bits(old.read(img), want)
        row = dict(model=name, output=out_size, table_vs_reference=na, direct_vs_reference=nb)
        ok = ok and a and b
        rows.append(row)
        print(row, flush=True)
    t = img[tab:tab + TABLE_BYTES].view(np.uint32)
    info = dict(model=name, increasing=int(t[2048]), code_nan=int(t[2049]),
                thresholds_nondecreasing=bool(np.all(np.diff(t[1024:2048].astype(np.int64)) >= 0)))
    ok = ok and info["thresholds_nondecreasing"]
    if out is not None:
        w = img[out:out + 12].view(np.uint32)
        info.update(exhaustive_patterns=int(w[1]) * 32, exhaustive_mismatches=int(w[0]),
                    largest_mismatch=hex(int(w[2]) - 1) if w[2] else None)
        ok = ok and w[0] == 0 and int(w[1]) * 32 == 1 << 32
        if control is not None:
            c = img[nc:nc + 12].view(np.uint32)
            info.update(control_patterns=int(c[1]) * 32, control_mismatches=int(c[0]))
            ok = ok and c[0] > 0
    rows.append(info)
    print(info, flush=True)
    return ok


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", required=True)
    ap.add_argument("--model", action="append", default=[])
    ap.add_argument("--map", default=None)
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--no-exhaustive", action="store_true")
    ap.add_argument("--fast", action="store_true", help="tables of the hardware elementary functions (RR_ELEM_EXACT 0)")
    a = ap.parse_args(argv)
    if "WG_RR_BUILD" not in os.environ:
        ap.error("WG_RR_BUILD must name a private build directory")
    from . import sequence as SQ
    models = []
    for path in a.model:
        models.append((Path(path).parent.name, SQ.model_from_file(path, a.map or SQ.DEFAULT_MAP).rsym))
    if a.synthetic:
        models.append(("synthetic", SQ.model_synthetic().rsym))
    rows, ok = [], True
    for name, sym in models:
        # negative control: the same symbols with K7 one f32 step larger (the table must then disagree)
        other = dataclasses.replace(sym, K7=np.nextafter(np.float32(sym.K7), np.float32(np.inf)))
        ok = check_model(name, sym, not a.fast, not a.no_exhaustive, rows, control=other) and ok
    Path(a.json).write_text(json.dumps(dict(passed=bool(ok), rows=rows), indent=1, default=str), encoding="utf-8")
    print("PASS" if ok else "FAIL", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
