# SPDX-License-Identifier: MIT
"""Differential check of the fast output path (prep / fast / fallback / pool) against rr_output_main + pool.

Both run on the same packed inputs in one arena; every output byte is compared (linear f32, CH, Q, PH1, trace),
including signed zeros and NaN payloads. Cases: several render / output sizes (Quality, Performance, ratio 2/3, odd
sizes), realistic and wild data (random binary16 bit patterns with NaN / infinity, out-of-history motion), the
sequence's synthetic constants, the converted model file's constants, and constant sets that force the fallback
(tiny products at the smallest normal).

    python -m clean.rr_f.gpu.output_diff --json <report> [--quick]
"""
import argparse
import json
import os
from dataclasses import replace
from pathlib import Path

import numpy as np

from . import build
from . import output as GO
from ..reference import output as RO
from ..tests.test_output import symbols


def f16(rng, shape, lo, hi):
    return rng.uniform(lo, hi, shape).astype(np.float16).astype(np.float64)


def wild16(rng, shape, p_special=0.02):
    bits = rng.integers(0, 1 << 16, shape, dtype=np.uint64).astype(np.uint16)
    v = bits.view(np.float16).astype(np.float64)
    keep = rng.random(shape) < p_special
    finite = np.where(np.isfinite(v), v, 0)
    return np.where(keep | np.isfinite(v), np.where(keep, v, finite), 0)


def make(rng, render, output, consts, data="smooth", reset=False):
    wr, hr = render
    wo, ho = output
    wh, hh = 3 * (wo + wo % 2) // 2, 3 * (ho + ho % 2) // 2
    wt, ht = (wr + 31) // 32 * 32, (hr + 31) // 32 * 32
    a = dict(output_size=output, reset=reset, constants=consts)
    if data in ("smooth", "dupmv", "covband"):
        a["Cnet"] = f16(rng, (ht, wt, 4), -0.5, 6)
        a["P"] = f16(rng, (hr, wr, 3), 0.05, 3)
        mv = lambda c: f16(rng, (hr, wr, c), -4, 4)  # noqa: E731
        a["Hprev"] = f16(rng, (hh, wh, 4), 0, 2)
    else:
        a["Cnet"] = wild16(rng, (ht, wt, 4))
        a["P"] = wild16(rng, (hr, wr, 3))
        mv = lambda c: np.where(rng.random((hr, wr, c)) < .9, f16(rng, (hr, wr, c), -8, 8), wild16(rng, (hr, wr, c), .2))  # noqa: E731
        a["Hprev"] = np.where(rng.random((hh, wh, 4)) < .97, f16(rng, (hh, wh, 4), 0, 4), wild16(rng, (hh, wh, 4), .3))
    a["PM"], a["Pe"], a["Pn"] = mv(4), mv(2), mv(2)
    if data == "dupmv":   # adversarial for the motion dedupe (OF_DEDUP): candidates equal to each other, +0 / -0 mixes,
        # NaN / infinity, values from a small set so equal motions are common
        pick = np.array([0.0, -0.0, 0.5, -0.5, 1.0, 3.25, -7.0, np.inf, -np.inf, np.nan], np.float16)
        base = pick[rng.integers(0, len(pick), (hr, wr, 2))]
        flip = lambda v: np.where((v == 0) & (rng.random(v.shape) < .5), -v, v).astype(np.float16)  # noqa: E731
        cand = [flip(base) for _ in range(3)]
        sel = rng.integers(0, 3, (hr, wr, 3))   # per candidate: 0 = copy of PM.xy (signed zero flipped), 1 = own pick, 2 = keep
        own = lambda: pick[rng.integers(0, len(pick), (hr, wr, 2))]  # noqa: E731
        outs = []
        for j, cur in enumerate((a["PM"][..., 2:4], a["Pe"], a["Pn"])):
            v = np.where(sel[..., j:j + 1] == 0, cand[j], np.where(sel[..., j:j + 1] == 1, own(), cur))
            outs.append(v.astype(np.float16))
        a["PM"] = np.concatenate([base, outs[0]], axis=-1).astype(np.float16)
        a["Pe"], a["Pn"] = outs[1], outs[2]
    a["G"] = RO.decode_unorm8(rng.integers(0, 256, (hr, wr, 3), dtype=np.uint8))
    if data == "covband":   # see constant_sets cov-band: P = 0 (every tap weight exp2(0) = 1), gate 0 = byte 8 everywhere
        a["P"][:] = 0
        a["G"][..., 0] = RO.decode_unorm8(np.full((hr, wr), 8, np.uint8))
    a["W4"] = RO.decode_unorm8(rng.integers(0, 256, (hr, wr, 4), dtype=np.uint8))
    a["jitter"] = tuple(np.float32(np.float16(v)) for v in rng.uniform(-.49, .49, 2))
    a["E"], a["Ep"], a["invE"] = np.float32(1.25), np.float32(.75), np.float32(.8)
    return a


def table(rng, kind="random"):
    t = np.zeros((513, 4), np.float32)
    if kind == "sequence":
        t[:] = [.125, .75, .125, .25]
        t[:, 3] = np.linspace(0, .5, 513, dtype=np.float32)
    else:
        t[:, :3] = rng.uniform(-.25, 1, (513, 3)).astype(np.float32)
        t[:, 3] = rng.uniform(-.5, .75, 513).astype(np.float32)
    if kind == "tiny":
        t[::7, 0] = np.float32(2.0 ** -70)
        t[::5, 2] = np.float32(-2.0 ** -63)
    return t


def constant_sets(rng):
    from . import sequence as SQ
    sets = {"sequence-synthetic": SQ.model_synthetic().osym}
    try:
        sets["model-file"] = SQ.model_from_file(SQ.DEFAULT_MODEL, SQ.DEFAULT_MAP).osym
    except Exception as e:  # noqa: BLE001 - the model file is optional here
        print("model file not used:", e)
    base = sets["sequence-synthetic"]
    sets["random-table"] = replace(base, TR=table(rng))
    sets["tiny-products"] = replace(base, TR=table(rng, "tiny"), K12=np.float32(2.0 ** -100), K1=np.float32(2.0 ** -110))
    sets["tiny-prep"] = replace(base, K2n=np.float32(-2.0 ** -115))
    sets["plain"] = symbols(TR=table(rng, "sequence"), K3=np.float32(.75))
    # exact results in the band [2^-126 - 2^-150, 2^-126) (M / F give 2^-126, the hardware 0): the fallback must run.
    # band-table: a reprojection weight 0.5 x (2^-125 - 2^-149) in rows i % 17 == 0 of TR (per-pixel fallback);
    # band-prep: P K2n = 0.5 x -(2^-125 - 2^-149) for P = 0.5 (prep flag: the whole frame falls back).
    band = np.array(0x00ffffff, np.uint32).view(np.float32)
    t = table(rng, "sequence")
    t[:, 1] = .5
    t[::17, 0] = band
    sets["band-table"] = replace(base, TR=t)
    sets["band-prep"] = replace(base, K2n=-band)
    # cov-band (clean/rr-mp1-output, mpeak probe P8 section 3): an exact result in the band at the coverage product
    # M(norm[3], M(total, K1)), the one band check real frames enter. Data "covband": P = 0, so every tap weight is
    # exp2(0) = 1, total = 16, Rn = 1/16; gate 0 = byte 8 (binary16 record 0.0313720703125), so accum[3] = A =
    # 0.5009765625 (16 binary16 additions), norm[3] = A / 16 and M(total, K1) = 16 K1, both exact. With K1 below the
    # exact product A K1 lies 0.877 x 2^-150 under 2^-126, inside the band: M rounds it to 2^-126, the hardware flushes
    # it to 0, so the check must fire and every quad goes to the fallback. K12 stays normal (of_insane false).
    sets["cov-band"] = replace(base, K1=np.float32(2.3464057550170007e-38))
    return sets


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", required=True)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--defines", default="", help="kernel defines K=V,... (OF_NO_FALLBACK=1: negative control)")
    ap.add_argument("--only", default="", help="cases whose label contains this text")
    ap.add_argument("--seed", type=int, default=20261020)
    a = ap.parse_args()
    if "WG_RR_BUILD" not in os.environ:
        ap.error("set WG_RR_BUILD")
    rng = np.random.default_rng(a.seed)
    sets = constant_sets(rng)
    sizes = [((61, 37), (97, 53)), ((128, 72), (192, 108)), ((427, 240), (640, 360)), ((320, 180), (640, 360)),
             ((426, 240), (639, 360))]
    if a.quick:
        sizes = sizes[:2]
    cases = []
    for name, c in sets.items():
        for si, (render, output) in enumerate(sizes):
            if name in ("tiny-products", "tiny-prep", "band-table", "band-prep") and si > 1:
                continue
            if name.startswith("cov-band"):   # appended after the original cases (their data stays the same)
                continue
            for data in ("smooth", "wild", "dupmv"):
                for half in ((True, False) if si <= 1 else (True,)):
                    cases.append((f"{name}/{render[0]}x{render[1]}->{output[0]}x{output[1]}/{data}/{'f16' if half else 'f32'}",
                                  render, output, c, data, bool(si == 1 and data == "wild"), half))
    # ratio exactly 2/3 (wr * 3 == wh * 2)
    cases.append(("sequence-synthetic/ratio2-3/f16", (64, 36), (64, 36), sets["sequence-synthetic"], "smooth", False, True))
    for name in [n for n in sets if n.startswith("cov-band")]:
        for render, output in sizes[1:3]:
            cases.append((f"{name}/{render[0]}x{render[1]}->{output[0]}x{output[1]}/covband/f16", render, output, sets[name],
                          "covband", False, True))
    if a.only:
        cases = [c for c in cases if a.only in c[0]]
    defines = dict(kv.split("=") for kv in a.defines.split(",") if kv)
    k = GO.kernel_set(defines, names=("main", "pool", "prep", "fast", "fallback"))
    ar = build.Arena()
    disp, recs = [], []
    for label, render, output, c, data, reset, half in cases:
        inp = make(rng, render, output, c, data, reset)
        args, offs, sizes_ = GO.prepare(ar, inp, packed=True, half_inputs=half)
        old = list(args)
        if half:   # rr_output_main reads f32 motion / history
            old[4:8] = [ar.add(np.asarray(inp[n], "<f4")) for n in ("PM", "Pe", "Pn", "Hprev")]
            old[37] &= ~36
        old_offs = [ar.reserve(n + 64, fill=0xA5) for n in sizes_]
        old[9:14] = old_offs
        old[38] = 0
        disp += GO.dispatch_list(old, fast=False) + GO.dispatch_list(args, fast=True)
        recs.append((label, inp, offs, old_offs, sizes_, args[38]))
    img, log = build.run(ar, k, disp, require_double=True, timeout=6000, workdir=str(Path(build.BUILD) / "output-diff"))
    rows, ok = [], True
    for label, inp, offs, old_offs, sizes_, scr in recs:
        new = GO.read_outputs(img, inp, offs, sizes_)
        ref = GO.read_outputs(img, inp, old_offs, sizes_)
        diff = {}
        for nm, x, y in zip(("linear", "CH", "Q", "PH1", "trace"), new, ref):
            xb = np.ascontiguousarray(x).view(np.uint8).reshape(-1)
            yb = np.ascontiguousarray(y).view(np.uint8).reshape(-1)
            diff[nm] = int(np.count_nonzero(xb != yb))
        guards = all(np.all(img[p + n:p + n + 64] == 0xA5) for p, n in zip(offs, sizes_))
        flagged, prep_flags = (int(v) for v in img[scr + 8:scr + 16].view("<u4"))
        q = new[2].astype(np.float64)
        row = dict(case=label, values=int(sum(v.size for v in new)), differing_bytes=diff, guards=guards,
                   fallback_pixels=flagged, prep_flagged_records=prep_flags,
                   history_pixels=int(new[0].shape[0] * new[0].shape[1]),
                   finite_Q=float(np.isfinite(q).mean()), distinct_Q=int(len(np.unique(new[2].reshape(-1, 3), axis=0))))
        row["identical"] = guards and not any(diff.values())
        ok &= row["identical"]
        rows.append(row)
        print(f"{label:60s} {'identical' if row['identical'] else 'DIFF ' + str(diff)} fallback {flagged}"
              f" prep {prep_flags} finiteQ {row['finite_Q']:.3f} distinctQ {row['distinct_Q']}", flush=True)
    rep = dict(passed=bool(ok), cases=rows, host_log=log)
    Path(a.json).parent.mkdir(parents=True, exist_ok=True)
    Path(a.json).write_text(json.dumps(rep, indent=1))
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
