# SPDX-License-Identifier: MIT
"""Fast page-54b resolve (kernels/rr_resolve_fast.hlsl): hardware probes, exhaustive checks of its exact
elementary functions, differential qualification against the verified kernel (kernels/rr_resolve.hlsl) and timing.

    python -m clean.rr_f.gpu.resolve_fast probes --json p.json
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np

from . import build
from . import resolve as GR
from ..reference import resolve as RR

NARGS = 3
TILE = 16
NWG = 512           # persistent workgroups of the fast kernel (at most; fewer when the image has fewer tiles)


def kern(entry, defines=None, block=256, tag=None):
    d = {"WG_BLOCK": block}
    d.update(defines or {})
    return build.compile_kernel("rr_resolve_fast.hlsl", entry, d, tag=tag or entry)


def nan32(u):
    return (u & 0x7FFFFFFF) > 0x7F800000


def probes(json_path):
    rng = np.random.default_rng(5402)
    report = {}
    # ---- f32 min / max
    # IEEE special classes: +-0, +-inf, quiet / signalling NaN, +-FMINN, +-1, +-FMAX, denormals, 1 + ulp
    fin = np.finfo(np.float32)
    sv = np.array([0.0, -0.0, np.inf, -np.inf, np.nan, -np.nan, fin.tiny, -fin.tiny, 1.0, -1.0, fin.max, -fin.max],
                  np.float32).view(np.uint32)
    sv[5] |= np.uint32(1 << 31)
    sp = np.concatenate([sv, np.array([(255 << 23) | 1, 1, (1 << 31) | ((1 << 23) - 1), (127 << 23) | 1], np.uint32)])
    pa, pb = np.meshgrid(sp, sp)
    r = rng.integers(0, 2 ** 32, (200000, 2), dtype=np.uint64).astype(np.uint32)
    r[:100000, 1] = r[:100000, 0] ^ rng.integers(0, 4, 100000).astype(np.uint32)
    pairs = np.concatenate([np.stack([pa.ravel(), pb.ravel()], 1), r])
    ar = build.Arena()
    n = len(pairs)
    off = ar.add(np.concatenate([pairs.ravel(), np.zeros(4 * n, np.uint32)]))
    n16 = 2048                      # a codes per dispatch chunk
    disp = [("mm", (n + 255) // 256, [off, 0, n])]
    h16 = []
    for a0 in range(0, 65536, n16):
        o = ar.add(np.zeros(8 + 64, np.uint32))
        disp.append(("h16", n16 * 256 // 256, [o, a0, n16]))
        h16.append(o)
    m = 400000
    t = rng.integers(0, 65536, (m, 3)).astype(np.uint32)
    # binary16 special classes: +-0, +-inf, NaN, denormals, the smallest normal, +-HMAX, +-1, 1/2
    spec16 = np.array([0.0, -0.0, np.inf, -np.inf, np.nan, 2.0 ** -24, -2.0 ** -24, 2.0 ** -14 - 2.0 ** -24, 2.0 ** -14,
                       65504.0, -65504.0, 1.0, -1.0, 0.5], np.float16).view(np.uint16).astype(np.uint32)
    t[:20000] = spec16[rng.integers(0, len(spec16), (20000, 3))]
    t[20000:60000, 2] = t[20000:60000, 0] ^ 0x8000          # cancellation-prone
    offf = ar.add(np.concatenate([t.ravel(), np.zeros(3 * m, np.uint32)]))
    disp.append(("fma", (m + 255) // 256, [offf, 0, m]))
    offc = ar.add(np.zeros(2 * 3072, np.uint32))
    disp.append(("cv", 3072 // 256, [offc, 0, 0]))
    kernels = {"mm": (kern("rr_rf_probe_minmax"), NARGS), "h16": (kern("rr_rf_probe_h16"), NARGS),
               "fma": (kern("rr_rf_probe_fma16"), NARGS), "cv": (kern("rr_rf_probe_cv"), NARGS)}
    toff = exp2_table(ar, disp, kernels)
    img, log = build.run(ar, kernels, disp, workdir=str(Path(build.BUILD) / "rf-probes"), timeout=3600)
    res = img[off + 8 * n: off + 24 * n].copy().view(np.uint32).reshape(n, 4)

    def eq(x, y):
        return (x == y) | (nan32(x) & nan32(y))
    report["f32_min_eq_pmin_mismatches"] = int(np.count_nonzero(~eq(res[:, 0], res[:, 2])))
    report["f32_max_eq_pmax_mismatches"] = int(np.count_nonzero(~eq(res[:, 1], res[:, 3])))
    den = (((pairs[:, 0] & 0x7F800000) == 0) & ((pairs[:, 0] & 0x7FFFFF) != 0)) |         (((pairs[:, 1] & 0x7F800000) == 0) & ((pairs[:, 1] & 0x7FFFFF) != 0))
    report["f32_pairs"] = int(n)
    report["f32_pairs_without_denormal_operand"] = int(np.count_nonzero(~den))
    report["f32_min_mismatches_without_denormal_operand"] = int(np.count_nonzero(~eq(res[:, 0], res[:, 2]) & ~den))
    report["f32_max_mismatches_without_denormal_operand"] = int(np.count_nonzero(~eq(res[:, 1], res[:, 3]) & ~den))
    bad = np.flatnonzero((~eq(res[:, 0], res[:, 2]) | ~eq(res[:, 1], res[:, 3])) & ~den)[:12]
    report["f32_examples"] = [[hex(int(pairs[i, 0])), hex(int(pairs[i, 1])), hex(int(res[i, 0])), hex(int(res[i, 2])),
                               hex(int(res[i, 1])), hex(int(res[i, 3]))] for i in bad]
    tot = np.zeros(4, np.int64)
    ex = []
    for o in h16:
        c = img[o:o + 4 * 72].copy().view(np.uint32)
        tot += c[:4]
        ex += [hex(int(v)) for v in c[8:8 + min(int(c[7]), 64)]][:8]
    report["f16_exhaustive_mismatches(min,max,pkmin,pkmax)"] = tot.tolist()
    report["f16_examples"] = ex[:16]
    fr = img[offf + 12 * m: offf + 24 * m].copy().view(np.uint32).reshape(m, 3)

    def eq16(x, y):
        return (x == y) | (((x & 0x7FFF) > 0x7C00) & ((y & 0x7FFF) > 0x7C00))
    report["fma16_mad_vs_wg_fma_f16_mismatches"] = int(np.count_nonzero(~eq16(fr[:, 0], fr[:, 1])))
    report["fma16_pk_vs_wg_fma_f16_mismatches"] = int(np.count_nonzero(~eq16(fr[:, 2], fr[:, 1])))
    cvr = img[offc:offc + 8 * 3072].copy().view(np.uint32).reshape(3072, 2)
    report["cv_codes_checked"] = int(sum(m + 1 for m in (255, 1023, 3)))
    report["cv_fast_vs_division_mismatches"] = int(np.count_nonzero(cvr[:, 0] != cvr[:, 1]))
    want = np.zeros(3072, np.float32)
    for k, m in enumerate((255, 1023, 3)):
        c = np.arange(m + 1)
        want[k * 1024:k * 1024 + m + 1] = (c / m).astype(np.float32)      # numpy: IEEE double quotient -> f32
    report["cv_vs_numpy_mismatches"] = int(np.count_nonzero(cvr[:, 1] != want.view(np.uint32)))
    from . import gen_rf_tables
    gt = img[toff:toff + 8192].copy().view(np.uint32)
    report["exp2_table_words"] = int(gt.size)
    report["exp2_table_gpu_vs_exact_mismatches"] = int(np.count_nonzero(gt != gen_rf_tables.table()))
    Path(json_path).parent.mkdir(parents=True, exist_ok=True)
    Path(json_path).write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(json.dumps(report, indent=1))
    return 0


EXH = ("exp2", "rcp", "f2h", "unorm255", "unorm1023", "unorm3", "x_hook", "", "hwlog2")


def accx_check(json_path, window=1 << 14):
    """rf_accx (the tap accumulation for weights below 2^-100) against F: every v of the alpha / gate classes (every
    positive binary16 value, the RGB10A2 alpha values 1/3, 2/3, 1) with every normal w within `window` f32 steps of
    FMINN / v (where w v crosses the flush band) and acc = 0, FMINN, FMINN + 1 step."""
    h = np.arange(1, 0x7C00, dtype=np.uint16).view(np.float16).astype(np.float32)
    a3 = np.array([1, 2, 3], np.float32) / np.float32(3)                       # rf_cv(c, 3): RN(c / 3)
    vs = np.unique(np.concatenate([h, a3, np.ones(1, np.float32)]))
    centre = (np.float64(2.0 ** -126) / vs.astype(np.float64)).astype(np.float32)
    words = np.stack([vs.view(np.uint32), centre.view(np.uint32)], 1).ravel()
    ar = build.Arena()
    off = ar.add(np.concatenate([np.zeros(16, np.uint32), words]))
    kernels = {"accx": (kern("rr_rf_probe_accx"), NARGS)}
    lanes = len(vs) * 2 * window
    img, _ = build.run(ar, kernels, [("accx", (lanes + 255) // 256, [off, window, len(vs)])], require_double=True,
                       workdir=str(Path(build.BUILD) / "rf-accx"), timeout=7200)
    c = img[off:off + 64].copy().view(np.uint32)
    rep = dict(v_values=int(len(vs)), window_steps=int(window), checked=int(c[2]), accx_vs_F_mismatches=int(c[0]),
               hardware_fma_vs_F_differences=int(c[1]),
               examples=[[hex(int(x)) for x in c[8 + 4 * k: 12 + 4 * k]] for k in range(min(int(c[3]), 2))])
    rep["passed"] = rep["accx_vs_F_mismatches"] == 0 and rep["checked"] > 0
    Path(json_path).parent.mkdir(parents=True, exist_ok=True)
    Path(json_path).write_text(json.dumps(rep, indent=1), encoding="utf-8")
    print(json.dumps(rep, indent=1))
    return 0 if rep["passed"] else 1


def exhaustive(json_path, funcs=EXH):
    """Every f32 bit pattern: the fast function equals the verified kernel's exact helper, or flags the input as
    undecided (the fast resolve then takes the verified per-pixel path)."""
    report = {}
    for fi, func in enumerate(EXH):
        if not func or func not in funcs:
            continue
        extra = {"RR_ELEM_EXACT": 2, "RF_TABLE_ARENA": 1} if func == "x_hook" else {}
        k = kern("rr_rf_exhaustive", {"RF_EXH_FUNC": fi, **extra}, tag=f"rr_rf_exh_{func}")
        tot = dict(mismatches=0, undecided=0, checked=0, max_steps=0)
        examples = []
        for start in range(0, 1 << 32, 1 << 30):
            ar = build.Arena()
            kernels, disp = {"x": (k, NARGS)}, []
            table = exp2_table(ar, disp, kernels)
            ctr = np.zeros(8 + 3 * 256, np.uint32)
            ctr[6] = table
            off = ar.add(ctr)
            n = 1 << 30
            img, _ = build.run(ar, kernels, disp + [("x", n // (256 * 64), [off, start, n])],
                               workdir=str(Path(build.BUILD) / f"rf-exh-{func}"), timeout=7200)
            c = img[off:off + ctr.nbytes].copy().view(np.uint32)
            tot["mismatches"] += int(c[0]); tot["undecided"] += int(c[1]); tot["checked"] += int(c[2])
            tot["max_steps"] = max(tot["max_steps"], int(c[5]))
            m = min(int(c[3]), 256)
            examples += [[hex(int(v)) for v in c[8 + 3 * i: 11 + 3 * i]] for i in range(m)][:16]
            print(func, hex(start), c[:4], flush=True)
        tot["examples(input,fast,ref)"] = examples[:16]
        tot["passed"] = tot["mismatches"] == 0 and tot["checked"] == 1 << 32
        if func == "x_hook":
            tot["passed"] = tot["passed"] and tot["undecided"] == 0
        report[func] = tot
    Path(json_path).parent.mkdir(parents=True, exist_ok=True)
    Path(json_path).write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(json.dumps(report, indent=1))
    return 0 if all(v["passed"] for v in report.values()) else 1


# ------------------------------------------------------------------------------------------------ differential

def verified_kernel(dxil=None, hw=False):
    """The verified kernel as the sequence builds it (rr_resolve_main, 64 lanes, exact elementary functions; hw: the
    opt-in hardware elementary functions)."""
    if dxil:
        return dxil
    e = 0 if hw else 1
    return build.compile_kernel("rr_resolve.hlsl", "rr_resolve_main",
                                {"WG_BLOCK": GR.BLOCK, "RR_ELEM_EXACT": e, "RR_MUT": 0}, tag=f"rr_resolve_main_e{e}m0")


WAVE = 64   # clean/rr-f-a14-tail D2: the fast resolve as wave64 (same bits in every mode; measured -0.01 / -0.03 ms)


EXTRA = {}          # experiment defines of the fast kernel
# clean/rr-f-a14-tail: A/B builds only, every resolve.fast* kernel (also the .x ones of accurate / exact)
EXTRA.update(json.loads(os.environ.get("WG_RR_RFX_DEFINES", "{}")))


def fast_kernel(count=False, prof=0, spec=1, wave=None, extra=None):
    wave = wave or WAVE
    ex = dict(EXTRA, **(extra or {}))
    return kern("rr_resolve_fast", {"RF_COUNT": int(count), "RF_PROF": prof, "RF_SPEC": spec, "WG_WAVE": wave, **ex},
                tag=f"rr_resolve_fast_s{spec}w{wave}" + ("_count" if count else "") + (f"_prof{prof}" if prof else "")
                + "".join(f"_{k}{v}" for k, v in sorted(ex.items())))


def init_kernel():
    return kern("rr_rf_init_table", tag="rr_rf_init_table")


def exp2_table(ar, disp, kernels):
    """The arena exp2 table (8 KB) and the one-time dispatch that fills it (rr_rf_init_table, 1024 lanes)."""
    off = ar.reserve(8192)
    kernels.setdefault("init", (init_kernel(), NARGS))
    disp.append(("init", 1024 // 256, [off, 0, 0]))
    return off


def fixup_bitmap(ar, sz):
    """The zero-filled slow-pixel bitmap of rr_resolve_fast / rr_resolve_fixup (parameter word 37)."""
    return ar.reserve(((sz.We * sz.He + 31) // 32) * 4)


def fixup_groups(sz, block=256):
    return (((sz.We * sz.He + 31) // 32) + block - 1) // block


def fixup_kernel(hw=False):
    """The verified pixel code for the pixels rr_resolve_fast leaves (exact: exp2 / rcp through the exhaustively
    proven hooks; hw: the opt-in hardware elementary functions, as rr_resolve_main with RR_ELEM_EXACT=0)."""
    if hw:
        return kern("rr_resolve_fixup", {"RR_ELEM_EXACT": 0, "RR_MUT": 0}, tag="rr_resolve_fixup_hw")
    return kern("rr_resolve_fixup", {"RR_ELEM_EXACT": 2, "RF_TABLE_ARENA": 1, "RR_MUT": 0}, tag="rr_resolve_fixup")


def pool_fast_kernel():
    """rr_resolve_pool's code with the fast exact exp2 / rcp (exp2 table at parameter word 38)."""
    return kern("rr_resolve_pool_fast", {"RR_ELEM_EXACT": 2, "RF_TABLE_ARENA": 0, "RR_MUT": 0},
                tag="rr_resolve_pool_fast")


def pool_verified_kernel():
    return build.compile_kernel("rr_resolve.hlsl", "rr_resolve_pool",
                                {"WG_BLOCK": GR.BLOCK, "RR_ELEM_EXACT": 1, "RR_MUT": 0}, tag="rr_resolve_pool_e1m0")


def pool_check(json_path, reps=0):
    """The fast pool kernel against the CPU reference (small fixtures, packed and code layouts, edge codes) and
    against the verified pool kernel at game sizes (bit-identical PQ); with reps: their GPU times."""
    from ..tests.test_resolve import symbols
    sym = symbols()
    rng = np.random.default_rng(5405)
    kernels = {"ref": (pool_verified_kernel(), GR.NARGS), "fast": (pool_fast_kernel(), NARGS)}
    ar, disp, recs = build.Arena(), [], []
    table = exp2_table(ar, disp, kernels)
    sizes = [(2, 2), (5, 3), (9, 7), (37, 21), (64, 36), (1920, 1080), (2560, 1440), (3840, 2160)]
    for out_size in sizes:
        (h1, w1), (h2, w2) = RR.pooled_sizes(out_size)
        for packed in ((True, False) if out_size[0] < 100 else (True,)):
            PH1 = rng.integers(0, 1024, (h1, w1, 4)).astype(np.uint16)
            PH1[..., 3] = 3
            flat = PH1.reshape(-1, 4)
            k = min(8, len(flat))
            flat[:k, :3] = np.array([[0, 1023, 1], [1022, 2, 511], [512, 513, 1000], [0, 0, 0], [1023, 1023, 1023],
                                     [3, 5, 7], [700, 701, 702], [1, 1, 1]], np.uint16)[:k]
            PQ0 = rng.integers(0, 1024, (h2 + 1, w2 + 2, 4)).astype(np.uint16)
            PQ0[..., 3] = 3
            u1 = GR.Upload(ar, PH1, "rgb10a2", packed=packed)
            outs = [GR.Upload(ar, PQ0, "rgb10a2", packed=packed) for _ in range(2)]
            sz = RR.sizes(out_size)
            for i, o in enumerate(outs):
                pp = GR.params(sym, sizes=sz, render=(1, 1), flags=0, res=dict(PH1=u1, PQ=o))
                pp[38] = table
                disp.append(("ref" if i == 0 else "fast", (w2 * h2 + 255) // 256 if i else (w2 * h2 + GR.BLOCK - 1) // GR.BLOCK,
                             [ar.add(pp), 0, 0]))
            want = RR.quarter_pool(PH1, PQ0, output_size=out_size, reset=False, symbols=sym) if out_size[0] < 100 else None
            recs.append((f"{out_size}{'-packed' if packed else '-codes'}", outs, want))
    img, log = build.run(ar, kernels, disp, require_double=True, reps=reps, time_first=1,
                         workdir=str(Path(build.BUILD) / "rf-pool"), timeout=7200)
    rows, ok_all = [], True
    for name, outs, want in recs:
        a, b = outs[0].read(img), outs[1].read(img)
        ok, bad = GR.same_bits(a, b)
        row = dict(case=name, values=int(a.size), fast_vs_verified_mismatches=bad)
        if want is not None:
            okr, badr = GR.same_bits(b, want)
            row["fast_vs_reference_mismatches"] = badr
            ok = ok and okr
        ok_all &= ok
        rows.append(row)
        print(row, flush=True)
    times = {}
    for line in log.splitlines():
        if line.startswith("time "):
            f = line.split()
            times[int(f[1]) - 1] = dict(kernel=f[2], best_ms=float(f[4]), mean_ms=float(f[7]))
    rep = dict(passed=ok_all, cases=rows, times=times)
    Path(json_path).parent.mkdir(parents=True, exist_ok=True)
    Path(json_path).write_text(json.dumps(rep, indent=1), encoding="utf-8")
    if reps:
        for i, (name, _, _) in enumerate(recs):
            print(name, "verified", times.get(2 * i), "fast", times.get(2 * i + 1))
    print("PASS" if ok_all else "FAIL")
    return 0 if ok_all else 1


def fast_groups(sz):
    tiles = ((sz.We + TILE - 1) // TILE) * ((sz.He + TILE - 1) // TILE)
    return min(tiles, NWG)


def f16_specials(rng, a, frac):
    """Poison a fraction of an f16 array with NaN / +-inf / +-0 / subnormals / extremes."""
    a = a.copy()
    flat = a.reshape(-1)
    k = int(frac * flat.size)
    if k:
        idx = rng.choice(flat.size, k, replace=False)
        vals = np.array([np.nan, np.inf, -np.inf, 0.0, -0.0, 2.0 ** -24, -2.0 ** -24, 2.0 ** -14, 65504.0, -65504.0,
                         6e-5, 1.0, -1.0], np.float32).astype(np.float16)
        flat[idx] = vals[rng.integers(0, len(vals), k)]
    return a


def make_case(ar, rng, *, out, render, flags, layer_fmt="f16", asrc="rgb10a2", out_fmt="f16", aout_fmt="f32",
              mv="f16", special=0.0, E=1.25, Ep=1.0, jitter=(0.125, -0.25), sym=None, out_origin=(0, 0),
              tbw=None, f32_tiny=False, alpha_in_ch=None, p_range=(0.05, 3.0), table=0):
    """One resolve dispatch's resources: inputs shared by both kernels, outputs (OUT, AOUT, THw, AHw) per kernel."""
    from ..tests.test_resolve import symbols
    sym = sym or symbols()
    if tbw is not None:
        sym = RR.ResolveSymbols(**{**{f: getattr(sym, f) for f in sym.__dataclass_fields__}, "TBW": np.float32(tbw)})
    sz = RR.sizes(out)
    wr, hr = render
    res = {}
    q = rng.uniform(-0.5, 3, (sz.Hh, sz.Wh, 3)).astype(np.float16)
    res["Q"] = GR.Upload(ar, f16_specials(rng, q, special), "f16")
    lw, lh = wr + 2, hr + 1
    if layer_fmt == "f16":
        lay = f16_specials(rng, rng.uniform(-0.2, 1.2, (lh, lw, 4)).astype(np.float16), special)
    elif layer_fmt == "f32":
        lay = rng.uniform(-0.2, 1.2, (lh, lw, 4)).astype(np.float32)
        if f32_tiny:
            lay.reshape(-1)[::97] = np.float32(1e-30)
            lay.reshape(-1)[::89] = np.float32(1e-39)       # denormal
    elif layer_fmt == "unorm8":
        lay = rng.integers(0, 256, (lh, lw, 4)).astype(np.uint8)
    else:
        lay = rng.integers(0, 1024, (lh, lw, 4)).astype(np.uint16)
        lay[..., 3] &= 3
    res["LAYER"] = GR.Upload(ar, lay, layer_fmt, origin=(1, 1))
    if asrc in ("rgb10a2", "codes"):
        c = rng.integers(0, 1024, (hr, wr, 4)).astype(np.uint16)
        c[..., 3] &= 3
        res["ASRC"] = GR.Upload(ar, c, "rgb10a2", packed=asrc == "rgb10a2")
        ach = 3
    elif asrc == "f16":
        a = f16_specials(rng, rng.uniform(0, 1, (hr + 1, wr, alpha_in_ch or 1)).astype(np.float16), special)
        res["ASRC"] = GR.Upload(ar, a, "f16", origin=(0, 1))
        ach = 0
    elif asrc == "f32":
        a = rng.uniform(0, 1, (hr, wr, 1)).astype(np.float32)
        if f32_tiny:
            a.reshape(-1)[::53] = np.float32(1e-30)
        res["ASRC"] = GR.Upload(ar, a, "f32")
        ach = 0
    else:
        res["ASRC"] = GR.Upload(ar, rng.integers(0, 256, (hr, wr, 4)).astype(np.uint8), "unorm8")
        ach = 3
    if mv == "f16":
        mvv = f16_specials(rng, rng.uniform(-3, 3, (hr, wr, 4)).astype(np.float16), special / 4)
        res["MV"] = GR.Upload(ar, mvv, "f16")
    elif mv == "f32":
        m = rng.uniform(-3, 3, (hr + 2, wr + 1, 2)).astype(np.float32)
        if f32_tiny:
            m.reshape(-1)[::61] = np.float32(1e-31)
        res["MV"] = GR.Upload(ar, m, "f32", origin=(1, 2))
    p = rng.uniform(p_range[0], p_range[1], (hr, wr, 4)).astype(np.float16)
    res["P"] = GR.Upload(ar, f16_specials(rng, p, special), "f16")
    res["G"] = GR.Upload(ar, rng.integers(0, 256, (hr, wr, 4)).astype(np.uint8), "unorm8")
    th = f16_specials(rng, rng.uniform(-0.1, 1.5, (sz.He, sz.We, 4)).astype(np.float16), special)
    res["THr"] = GR.Upload(ar, th, "f16")
    ah = f16_specials(rng, rng.uniform(0, 1, (sz.He, sz.We, 1)).astype(np.float16), special)
    res["AHr"] = GR.Upload(ar, ah, "f16")
    odt = {"f16": np.float16, "f32": np.float32, "unorm8": np.uint8, "rgb10a2": np.uint16}
    outs = []
    for _ in range(2):
        o = {}
        ob = np.zeros((sz.He + out_origin[1] + 1, sz.We + out_origin[0] + 2, 4), odt[out_fmt])
        o["OUT"] = GR.Upload(ar, ob, out_fmt, origin=out_origin)
        if aout_fmt:
            o["AOUT"] = GR.Upload(ar, np.zeros((sz.He, sz.We, 1), odt[aout_fmt]), aout_fmt)
        o["THw"] = GR.Upload(ar, np.zeros((sz.He, sz.We, 4), np.float16), "f16", fill=0xEE)
        o["AHw"] = GR.Upload(ar, np.zeros((sz.He, sz.We, 1), np.float16), "f16", fill=0xEE)
        outs.append(o)
    blocks = []
    for i, o in enumerate(outs):
        ups = {**res, **o}
        if not (flags & GR.F_LAYER):
            ups.pop("LAYER")
        pr = GR.params(sym, sizes=sz, render=render, flags=flags, E=np.float32(E), Ep=np.float32(Ep),
                       jitter=tuple(np.float32(j) for j in jitter), layer_origin=(1, 1), alpha_channel=ach, res=ups)
        if i == 1:
            pr[37] = fixup_bitmap(ar, sz)
            pr[38] = table
        blocks.append(ar.add(pr))
    return sz, blocks, outs


def compare_outs(img, outs):
    rows = {}
    for key in ("OUT", "AOUT", "THw", "AHw"):
        if key not in outs[0]:
            continue
        a, b = outs[0][key].read(img), outs[1][key].read(img)
        ok, bad = GR.same_bits(a, b)
        rows[key] = dict(values=int(a.size), mismatches=bad)
    return rows


CASES = [
    # name, out, render, flags, options
    ("seq-like", (256, 144), (171, 96), "ext|layer|alpha|aout", {}),
    ("seq-odd", (61, 37), (41, 25), "ext|layer|alpha|aout", {}),
    ("reset", (100, 60), (67, 40), "ext|layer|alpha|aout|treset|areset", {}),
    ("treset-only", (100, 60), (67, 40), "ext|layer|alpha|aout|treset", {}),
    ("plain", (98, 54), (66, 36), "", {}),
    ("layer-only", (128, 72), (86, 48), "ext|layer", {}),
    # the game's configuration (extended + layer, no alpha upscaling: RF_SPEC 2)
    ("game-like", (256, 144), (171, 96), "ext|layer", {}),
    ("game-reset", (100, 60), (67, 40), "ext|layer|treset", {}),
    ("game-bounded", (127, 71), (85, 48), "ext|layer|bounded", {}),
    ("game-specials", (160, 96), (107, 64), "ext|layer", dict(special=0.05)),
    ("game-odd", (61, 37), (41, 25), "ext|layer", dict(jitter=(0.5, -0.5))),
    ("game-f32-layer", (96, 64), (64, 43), "ext|layer", dict(layer_fmt="f32", mv="f32")),
    ("alpha-only", (128, 72), (86, 48), "ext|alpha", {}),
    ("alpha-in-colour", (128, 72), (86, 48), "ext|layer|alpha", {}),
    ("bounded", (127, 71), (85, 48), "ext|layer|alpha|aout|bounded", {}),
    ("specials", (160, 96), (107, 64), "ext|layer|alpha|aout", dict(special=0.05)),
    ("specials-reset", (160, 96), (107, 64), "ext|layer|alpha|aout|treset|areset", dict(special=0.05)),
    ("specials-heavy", (64, 48), (43, 32), "ext|layer|alpha|aout", dict(special=0.4, asrc="f16")),
    ("f32-layer-tiny", (96, 64), (64, 43), "ext|layer|alpha|aout", dict(layer_fmt="f32", f32_tiny=True, mv="f32", asrc="f32")),
    ("f32-layer", (96, 64), (64, 43), "ext|layer|alpha|aout", dict(layer_fmt="f32", mv="f32", asrc="f32")),
    ("unorm8-layer", (96, 64), (64, 43), "ext|layer|alpha|aout", dict(layer_fmt="unorm8", out_fmt="unorm8", aout_fmt="unorm8", asrc="unorm8")),
    ("rgb10a2-layer", (96, 64), (64, 43), "ext|layer|alpha", dict(layer_fmt="rgb10a2", out_fmt="rgb10a2", asrc="codes")),
    ("f32-out-origin", (90, 50), (60, 34), "ext|layer|alpha|aout", dict(out_fmt="f32", out_origin=(3, 2), aout_fmt="f16", asrc="f16")),
    ("no-mv", (96, 64), (64, 43), "ext|layer|alpha|aout", dict(mv=None)),
    ("dlaa", (96, 64), (96, 64), "ext|layer|alpha|aout", {}),
    ("perf", (128, 72), (64, 36), "ext|layer|alpha|aout", {}),
    ("ultra-perf", (150, 90), (50, 30), "ext|layer|alpha|aout", {}),
    ("tiny-image", (3, 2), (2, 1), "ext|layer|alpha|aout", {}),
    ("E-small", (96, 64), (64, 43), "ext|layer|alpha|aout", dict(E=2.0 ** -30, Ep=2.0 ** -30)),
    ("E-large", (96, 64), (64, 43), "ext|layer|alpha|aout", dict(E=2.0 ** 60, Ep=1.0)),
    ("jitter-zero", (96, 64), (64, 43), "ext|layer|alpha|aout", dict(jitter=(0.0, 0.0))),
    ("jitter-half", (96, 64), (64, 43), "ext|layer|alpha|aout", dict(jitter=(0.5, -0.5))),
    ("jitter-tiny", (96, 64), (64, 43), "ext|layer|alpha|aout", dict(jitter=(1e-9, -3e-8))),
    ("tbw-one", (96, 64), (64, 43), "ext|layer|alpha|aout", dict(tbw=1.0)),
    ("p-wide", (96, 64), (64, 43), "ext|layer|alpha|aout", dict(p_range=(-40.0, 200.0))),
    ("p-small", (96, 64), (64, 43), "ext|layer|alpha|aout", dict(p_range=(0.0, 0.01))),
    # sharp footprints: tap weights below 2^-100 (the rf_accx batches)
    ("p-sharp-30", (96, 64), (64, 43), "ext|layer|alpha|aout", dict(p_range=(0.0, 30.0))),
    ("p-sharp-1000", (96, 64), (64, 43), "ext|layer|alpha|aout", dict(p_range=(0.0, 1000.0))),
]
FLAGBITS = dict(ext=GR.F_EXT, layer=GR.F_LAYER, alpha=GR.F_ALPHA, aout=GR.F_AOUT, treset=GR.F_TRESET,
                areset=GR.F_ARESET, bounded=GR.F_BOUNDED)


def parse_flags(f):
    v = 0
    for t in filter(None, f.split("|")):
        v |= FLAGBITS[t]
    if v & (GR.F_LAYER | GR.F_ALPHA):
        v |= GR.F_EXT
    return v


def differential(json_path, verified_dxil=None, seeds=3, count=False, spec=1, hw=False, mutation=0):
    """The fast kernel against the verified kernel on the same inputs: every output value bit-identical (hw: both
    with the opt-in hardware elementary functions; mutation: a negative control, which must fail)."""
    extra = {"RF_HW": 1} if hw else {}
    if mutation:
        extra["RF_MUT"] = mutation
    kernels = {"ref": (verified_kernel(verified_dxil, hw), GR.NARGS),
               "fast": (fast_kernel(count, spec=spec, extra=extra), NARGS), "fixup": (fixup_kernel(hw), NARGS)}
    report, all_ok = dict(cases=[], spec=spec, hw=hw, mutation=mutation), True
    for seed in range(seeds):
        ar = build.Arena()
        disp, recs = [], []
        table = exp2_table(ar, disp, kernels)
        rng = np.random.default_rng(54200 + seed)
        for name, out, render, fl, opt in CASES:
            flags = parse_flags(fl)
            sz, blocks, outs = make_case(ar, rng, out=out, render=render, flags=flags, table=table, **opt)
            disp.append(("ref", (sz.We * sz.He + GR.BLOCK - 1) // GR.BLOCK, [blocks[0], 0, 0]))
            cnt = ar.add(np.zeros(4, np.uint32))
            disp.append(("fast", fast_groups(sz), [blocks[1], cnt, 0]))
            disp.append(("fixup", fixup_groups(sz), [blocks[1], 0, 0]))
            recs.append((name, flags, outs, cnt, sz))
        img, log = build.run(ar, kernels, disp, require_double=True, workdir=str(Path(build.BUILD) / "rf-diff"),
                             timeout=7200)
        for name, flags, outs, cnt, sz in recs:
            rows = compare_outs(img, outs)
            if count:
                rows["slow_pixels"] = dict(values=int(sz.We * sz.He), mismatches=0,
                                           count=int(img[cnt:cnt + 4].view(np.uint32)[0]))
            ok = all(v["mismatches"] == 0 for v in rows.values())
            all_ok &= ok
            report["cases"].append(dict(seed=seed, case=name, flags=flags, passed=ok, outputs=rows))
            print(f"seed {seed} {name:18s} {'ok' if ok else 'MISMATCH'} " +
                  " ".join(f"{k}:{v.get('count', v['mismatches'])}/{v['values']}" for k, v in rows.items()), flush=True)
    report["passed"] = all_ok
    Path(json_path).parent.mkdir(parents=True, exist_ok=True)
    Path(json_path).write_text(json.dumps(report, indent=1), encoding="utf-8")
    print("PASS" if all_ok else "FAIL", flush=True)
    return 0 if all_ok else 1


def bench(json_path, sizes=("1280x720:1920x1080", "1706x960:2560x1440", "1920x1080:3840x2160"), reps=10,
          verified_dxil=None, profs=(), variants=None, flags="ext|layer|alpha|aout"):
    """GPU time of the verified and the fast kernel on the sequence's resolve configuration (layer + alpha
    upscaling + separate alpha, f16 layer, rgb10a2 colour input, f16 out, f32 alpha out), steady (no reset)."""
    kernels = {"ref": (verified_kernel(verified_dxil), GR.NARGS), "fast": (fast_kernel(), NARGS),
               "fixup": (fixup_kernel(), NARGS)}
    for pf in profs:
        kernels[f"prof{pf}"] = (fast_kernel(prof=pf, spec=1), NARGS)
    variants = dict(variants or {})          # name -> dxil: other builds timed interleaved with the same inputs
    for name, dx in variants.items():
        kernels["var-" + name] = (dx, NARGS)
    flags = parse_flags(flags)
    report = {}
    for s in sizes:
        r_s, o_s = s.split(":")
        render = tuple(int(v) for v in r_s.split("x"))
        out = tuple(int(v) for v in o_s.split("x"))
        ar = build.Arena()
        rng = np.random.default_rng(7)
        disp = []
        table = exp2_table(ar, disp, kernels)
        sz, blocks, outs = make_case(ar, rng, out=out, render=render, flags=flags, table=table)
        disp += [("ref", (sz.We * sz.He + GR.BLOCK - 1) // GR.BLOCK, [blocks[0], 0, 0]),
                ("fast", fast_groups(sz), [blocks[1], 0, 0]), ("fixup", fixup_groups(sz), [blocks[1], 0, 0])]
        disp += [(f"prof{pf}", fast_groups(sz), [blocks[1], 0, 0]) for pf in profs]
        disp += [("var-" + name, fast_groups(sz), [blocks[1], 0, 0]) for name in variants]
        img, log = build.run(ar, kernels, disp, require_double=True, reps=reps, time_first=1,
                             workdir=str(Path(build.BUILD) / "rf-bench"), timeout=7200)
        times = {}
        for line in log.splitlines():
            if line.startswith("time "):
                f = line.split()
                times[int(f[1]) - 1] = dict(best_ms=float(f[4]), mean_ms=float(f[7]))   # dispatch 0: the table fill
        rows = compare_outs(img, outs)
        report[s] = dict(verified=times.get(0), fast=times.get(1), fixup=times.get(2), outputs=rows,
                         profile={pf: times.get(3 + i) for i, pf in enumerate(profs)},
                         variants={name: times.get(3 + len(profs) + i) for i, name in enumerate(variants)})
        print(s, "verified", times.get(0), "fast", times.get(1), "fixup", times.get(2), rows, flush=True)
        for i, pf in enumerate(profs):
            print(f"  prof {pf}: {times.get(3 + i)}", flush=True)
        for i, name in enumerate(variants):
            print(f"  variant {name}: {times.get(3 + len(profs) + i)}", flush=True)
    Path(json_path).parent.mkdir(parents=True, exist_ok=True)
    Path(json_path).write_text(json.dumps(report, indent=1), encoding="utf-8")
    return 0


def phase_timing(json_path, size="1706x960:2560x1440"):
    """Shader cycles per kernel phase (RF_TIMING build, summed over waves): where the fast kernel spends its time."""
    k = kern("rr_resolve_fast", {"RF_SPEC": 1, "RF_TIMING": 1}, tag="rr_resolve_fast_s1_timing")
    r_s, o_s = size.split(":")
    render = tuple(int(v) for v in r_s.split("x"))
    out = tuple(int(v) for v in o_s.split("x"))
    ar = build.Arena()
    rng = np.random.default_rng(7)
    flags = parse_flags("ext|layer|alpha|aout")
    kernels, disp = {"fast": (k, NARGS)}, []
    table = exp2_table(ar, disp, kernels)
    sz, blocks, outs = make_case(ar, rng, out=out, render=render, flags=flags, table=table)
    ctr = ar.add(np.zeros(16, np.uint32))
    img, log = build.run(ar, kernels, disp + [("fast", fast_groups(sz), [blocks[1], ctr, 0])],
                         workdir=str(Path(build.BUILD) / "rf-timing"), timeout=3600)
    c = img[ctr:ctr + 64].view(np.uint32).astype(np.float64)
    names = ["Q tile + barrier", "horizontal + barrier", "render tile + prefetch + barrier", "pixel: coords, loads",
             "plain colour + decode", "alpha upscaling", "layer + final", "stores + barrier"]
    waves = c[8]
    tot = c[:8].sum()
    rep = {n: dict(cycles_per_wave=c[i] / waves, share=c[i] / tot) for i, n in enumerate(names)}
    rep["waves"] = int(waves)
    Path(json_path).write_text(json.dumps(rep, indent=1), encoding="utf-8")
    for n in names:
        print(f"{n:34s} {rep[n]['cycles_per_wave']:12.0f} cycles/wave {100 * rep[n]['share']:5.1f} %")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", choices=["probes", "exhaustive", "diff", "bench", "timing", "pool", "accx"])
    ap.add_argument("--verified-dxil", help="a prebuilt verified kernel (e.g. the base commit's build)")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--reps", type=int, default=100, help="bench: timed repetitions (also the clock warm-up)")
    ap.add_argument("--prof", default="", help="bench: also time profiling variants (RF_PROF values, comma list)")
    ap.add_argument("--spec", type=int, default=1, help="RF_SPEC of the fast kernel (0 generic, 1 standard)")
    ap.add_argument("--hw", action="store_true", help="diff: the opt-in hardware elementary functions (RF_HW)")
    ap.add_argument("--mutation", type=int, default=0, choices=[0, 1, 2, 3, 4, 5],
                    help="diff negative control: 1 first exp2 tier alone, 2 no verified-path pixels, 3 bilinear "
                         "without its negative-coordinate rule, 4 tiny tap weights with the plain fma (fails only where "
                         "a product meets the flush band: gpu/resolve_fast.py accx), 5 tiny-weight batches without "
                         "their products (must FAIL)")
    ap.add_argument("--count", action="store_true", help="diff: count the pixels taking the verified path")
    ap.add_argument("--sizes", default="1280x720:1920x1080,1706x960:2560x1440,1920x1080:3840x2160")
    ap.add_argument("--funcs", default=",".join(f for f in EXH if f))
    ap.add_argument("--flags", default="ext|layer|alpha|aout", help="bench: resolve configuration (game: ext|layer)")
    ap.add_argument("--variants", default="", help="bench: fast-kernel builds timed with the same inputs, "
                    "name:spec=S,K=V;name2:...")
    ap.add_argument("--json", required=True)
    a = ap.parse_args(argv)
    if "WG_RR_BUILD" not in os.environ:
        ap.error("WG_RR_BUILD must name a private build directory")
    if a.what == "probes":
        return probes(a.json)
    if a.what == "exhaustive":
        return exhaustive(a.json, tuple(a.funcs.split(",")))
    if a.what == "pool":
        return pool_check(a.json, a.reps)
    if a.what == "accx":
        return accx_check(a.json)
    if a.what == "timing":
        return phase_timing(a.json)
    if a.what == "diff":
        return differential(a.json, a.verified_dxil, a.seeds, a.count, a.spec, a.hw, a.mutation)
    if a.what == "bench":
        var = {}
        for item in filter(None, a.variants.split(";")):
            name, _, kv = item.partition(":")
            d = {k: int(v) for k, v in (p.split("=") for p in kv.split(",") if p)}
            var[name] = fast_kernel(spec=d.pop("spec", 1), wave=d.pop("wave", None), extra=d or None)
        return bench(a.json, tuple(a.sizes.split(",")), verified_dxil=a.verified_dxil, reps=a.reps,
                     profs=tuple(int(v) for v in a.prof.split(",")) if a.prof else (), variants=var, flags=a.flags)


if __name__ == "__main__":
    raise SystemExit(main())
