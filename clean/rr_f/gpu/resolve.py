# SPDX-License-Identifier: MIT
"""D3D12 qualification of the page-54b kernels (kernels/rr_resolve.hlsl) against reference/resolve.py.

Every comparison is bit-exact (NaN payloads and NaN signs excepted: any NaN equals any NaN). Synthetic data and
invented symbols only; no model constant. Usage (repository root, WG_RR_BUILD set to a private build folder):

    python -m clean.rr_f.gpu.resolve --json out.json                     # fixtures + probes (one GPU call)
    python -m clean.rr_f.gpu.resolve --exhaustive exp2 --json ex.json    # exp2 fast path vs double-double
    python -m clean.rr_f.gpu.resolve --exhaustive log2 --json lx.json
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from . import build
from ..reference import f32x as fx
from ..reference import resolve as rr

NARGS = 3
BLOCK = 64
NPARAM = 176
FMT = {"f32": 0, "f16": 1, "unorm8": 2, "rgb10a2": 3, "codes": 4}
RES = dict(Q=40, OUT=48, AOUT=56, LAYER=64, OPAC=72, ASRC=80, MV=88, P=96, G=104, THr=112, THw=120, OHr=128,
           OHw=136, AHr=144, AHw=152, PH1=160, PQ=168)
F_EXT, F_LAYER, F_OPAC, F_ALPHA, F_AOUT, F_TRESET, F_ARESET, F_BOUNDED = 1, 2, 4, 8, 16, 32, 64, 128


def fbits(v):
    return int(np.asarray(v, np.float32).view(np.uint32))


def hbits(v):
    return int(np.asarray(v, np.float16).view(np.uint16))


def pack_rgb10a2(codes):
    c = codes.astype(np.uint32)
    return c[..., 0] | (c[..., 1] << 10) | (c[..., 2] << 20) | (c[..., 3] << 30)


def unpack_rgb10a2(words):
    w = words.astype(np.uint32)
    return np.stack([w & 1023, (w >> 10) & 1023, (w >> 20) & 1023, w >> 30], -1).astype(np.uint16)


class Upload:
    """A surface in the arena: descriptor words plus how to read it back."""

    def __init__(self, arena, data, fmt, origin=(0, 0), packed=True, fill=None):
        data = np.ascontiguousarray(data)
        self.shape, self.fmt, self.origin = data.shape, fmt, tuple(origin)
        if fmt == "rgb10a2" and packed:
            raw, code = pack_rgb10a2(data), FMT["rgb10a2"]
        elif fmt == "rgb10a2":
            raw, code = data.astype("<u2"), FMT["codes"]
        else:
            raw, code = data, FMT[fmt]
        self.packed = fmt == "rgb10a2" and packed
        self.nbytes = raw.nbytes
        self.off = arena.add(raw) if fill is None else arena.reserve(raw.nbytes, fill=fill)
        h, w, c = data.shape
        self.desc = [self.off, w, h, w, code, c, origin[0], origin[1]]

    def read(self, img):
        b = img[self.off:self.off + self.nbytes].copy()
        h, w, c = self.shape
        if self.packed:
            return unpack_rgb10a2(b.view("<u4").reshape(h, w))
        dt = {"f32": "<f4", "f16": "<f2", "unorm8": np.uint8, "rgb10a2": "<u2"}[self.fmt]
        return b.view(dt).reshape(h, w, c)


def surface_upload(arena, s, packed=True):
    return Upload(arena, s.data, s.format, s.origin, packed)


def params(sym, *, sizes, render, flags, E=1, Ep=1, jitter=(0, 0), layer_origin=(0, 0), alpha_channel=0, res=None):
    p = np.zeros(NPARAM, np.uint32)
    sz = sizes
    p[:8] = [sz.We, sz.He, sz.Wh, sz.Hh, render[0], render[1], sz.Wo, sz.Ho]
    p[8] = flags
    p[9:13] = [fbits(E), fbits(Ep), fbits(jitter[0]), fbits(jitter[1])]
    p[13:15] = layer_origin
    p[17:22] = [hbits(v) for v in sym.even_weights()]
    for i, name in enumerate(("K1", "K2n", "K2x2n", "K4", "K5", "K7", "K8", "K10", "K11", "K12", "K13", "C_HDEC",
                              "T_EFL", "TBW")):
        p[22 + i] = fbits(getattr(sym, name))
    p[36] = alpha_channel
    for name, up in (res or {}).items():
        if up is not None:
            p[RES[name]:RES[name] + 8] = np.array(up.desc, np.int64).astype(np.uint32)
    return p


def same_bits(a, b):
    """Bit-exact equality of two arrays of one dtype; float NaNs form one class."""
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape or a.dtype != b.dtype:
        return False, -1
    if a.dtype.kind == "f":
        na, nb = np.isnan(a), np.isnan(b)
        ua = a.view(np.uint32 if a.itemsize == 4 else np.uint16)
        ub = b.view(np.uint32 if b.itemsize == 4 else np.uint16)
        ok = (na & nb) | (~na & ~nb & (ua == ub))
    else:
        ok = a == b
    return bool(np.all(ok)), int(np.count_nonzero(~ok))


# ------------------------------------------------------------------------------------------------ fixtures

def sequences():
    """Frame sequences (name, out_size, render, alpha_upscaling, [frame configs]) covering 54b.8's classes."""
    from ..tests.test_resolve import CONFIGS
    rng = np.random.default_rng(54008)
    rows = []
    sizes = [((2, 2), (2, 2)), ((5, 3), (3, 2)), ((7, 4), (5, 3)), ((16, 9), (11, 6)), ((33, 17), (22, 12)),
             ((64, 36), (43, 24))]
    for k, (out_size, render) in enumerate(sizes):
        for alpha in (False, True):
            frames = []
            for n in range(6):
                cfg = dict(CONFIGS[int(rng.integers(0, len(CONFIGS)))])
                cfg["alpha_in"] = alpha and bool(rng.integers(0, 2))
                cfg["alpha_out"] = alpha and bool(rng.integers(0, 2))
                frames.append(dict(cfg=cfg, reset=n == 0 or (n == 3 and k % 2 == 0)))
            rows.append((f"seq{k}-{'alpha' if alpha else 'noalpha'}", out_size, render, alpha, frames))
    # half-way jitter decisions, both signs, x and y
    rows.append(("halfway", (4, 4), (2, 2), False,
                 [dict(cfg=dict(layer=True), reset=False, jitter=(np.float32(jx), np.float32(jy)))
                  for jx, jy in ((0.25, 0.0), (-0.25, 0.0), (0.0, 0.25), (0.0, -0.25), (0.125, -0.125))]))
    # history copies across frames without the layer, then the opacity layer appearing
    rows.append(("copies", (6, 4), (4, 3), False,
                 [dict(cfg=dict(layer=True), reset=True), dict(cfg=dict(layer=True), reset=False),
                  dict(cfg=dict(), reset=False), dict(cfg=dict(layer=True), reset=False),
                  dict(cfg=dict(layer=True, opacity=True), reset=False),
                  dict(cfg=dict(opacity=True), reset=False), dict(cfg=dict(layer=True, opacity=True), reset=False)]))
    return rows


def specials(rng, f):
    """Poison some inputs with infinities / NaN / beyond-HMAX codes (54b.8 plain-resolve class)."""
    q = f["Q"].copy()
    m = rng.uniform(size=q.shape)
    q = np.where(m < 0.02, np.inf, q)
    q = np.where((m > 0.02) & (m < 0.04), -np.inf, q)
    q = np.where((m > 0.04) & (m < 0.05), np.nan, q)
    f["Q"] = q
    if f["layer"] is not None and f["layer"].format in ("f16", "f32"):
        d = f["layer"].data.copy()
        d.reshape(-1)[:: 7] = np.inf
        f["layer"] = rr.Surface(d, f["layer"].format, f["layer"].origin)
    return f


def build_sequence(arena, dispatches, kernels, name, out_size, render, alpha, frames, sym, rng):
    """CPU reference over the sequence plus the chained GPU dispatches; returns comparison records."""
    from ..tests.test_resolve import make_frame, run_frame
    sz = rr.sizes(out_size)
    hist = rr.ResolveHistories(out_size, alpha_upscaling=alpha)
    zero4 = np.zeros((sz.He, sz.We, 4), np.float16)
    TH = [Upload(arena, zero4, "f16") for _ in range(2)]
    OH = [Upload(arena, zero4, "f16") for _ in range(2)]
    AH = [Upload(arena, np.zeros((sz.He, sz.We, 1), np.float16), "f16") for _ in range(2)]
    th_live = oh_live = False
    records = []
    parity = 0
    for n, fr in enumerate(frames):
        f = make_frame(rng, out_size, render, jitter=fr.get("jitter"), **fr["cfg"])
        if n % 3 == 2:
            f = specials(rng, f)
        reset = fr["reset"]
        before = hist
        res = run_frame(f, out_size, render, before, reset=reset, parity=parity, sym=sym)
        layer_src = f["cbp"] if f["cbp"] is not None else f["layer"]
        created = False
        if layer_src is not None:
            created = not th_live or (f["opacity"] is not None and not oh_live)
            th_live = True
            oh_live = oh_live or f["opacity"] is not None
        flags = 0
        if res.variant == "extended":
            flags |= F_EXT
        if layer_src is not None:
            flags |= F_LAYER
            if f["opacity"] is not None:
                flags |= F_OPAC
        if alpha:
            flags |= F_ALPHA
            if f["alpha_out"] is not None:
                flags |= F_AOUT
        if reset or created or before.ah_created:
            flags |= F_TRESET
        if reset or before.ah_created:
            flags |= F_ARESET
        dest = rr.Surface(np.zeros((sz.He + 2, sz.We + 1, 4), np.float32), "f32", (1, 2))
        ups = dict(Q=Upload(arena, f["Q"].astype(np.float16), "f16"), OUT=surface_upload(arena, dest),
                   AOUT=surface_upload(arena, f["alpha_out"]) if f["alpha_out"] is not None else None,
                   LAYER=surface_upload(arena, layer_src) if layer_src is not None else None,
                   OPAC=surface_upload(arena, f["opacity"]) if (f["opacity"] is not None and layer_src is not None) else None,
                   MV=Upload(arena, f["motion"].astype(np.float16), "f16", f["mv_origin"]),
                   P=Upload(arena, f["P"].astype(np.float16), "f16"), G=Upload(arena, f["G"], "unorm8"),
                   THr=TH[parity], THw=TH[1 - parity], OHr=OH[parity], OHw=OH[1 - parity],
                   AHr=AH[parity], AHw=AH[1 - parity])
        if alpha:
            src = f["alpha_in"] if f["alpha_in"] is not None else f["colour_in"]
            ups["ASRC"] = surface_upload(arena, src)
        p = params(sym, sizes=sz, render=render, flags=flags, E=f["E"], Ep=f["Ep"], jitter=f["jitter"],
                   layer_origin=f["layer"].origin if f["layer"] is not None else (0, 0),
                   alpha_channel=0 if f["alpha_in"] is not None else 3, res=ups)
        po = arena.add(p)
        dispatches.append(("main", (sz.We * sz.He + BLOCK - 1) // BLOCK, [po, 0, 0]))
        records.append(dict(name=f"{name}/f{n}", variant=res.variant, flags=flags,
                            checks=[("colour", ups["OUT"], res.colour.data)] +
                                   ([("alpha_out", ups["AOUT"], res.alpha.data)] if ups["AOUT"] is not None else [])))
        hist = res.histories
        parity = 1 - parity
    final = []
    for i in range(2):
        if hist.TH is not None:
            final.append((f"TH[{i}]", TH[i], hist.TH[i]))
        if hist.OH is not None:
            final.append((f"OH[{i}]", OH[i], hist.OH[i]))
        if hist.AH is not None:
            final.append((f"AH[{i}]", AH[i], hist.AH[i]))
    records.append(dict(name=f"{name}/histories", variant="-", flags=0, checks=final))
    return records


def pool_fixtures(arena, dispatches, sym, rng):
    records = []
    for out_size in ((2, 2), (5, 3), (9, 7), (37, 21), (64, 36)):
        (h1, w1), (h2, w2) = rr.pooled_sizes(out_size)
        for packed in (True, False):
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
            u1 = Upload(arena, PH1, "rgb10a2", packed=packed)
            uq = Upload(arena, PQ, "rgb10a2", packed=packed)
            p = params(sym, sizes=sz, render=(1, 1), flags=0, res=dict(PH1=u1, PQ=uq))
            dispatches.append(("pool", (w2 * h2 + BLOCK - 1) // BLOCK, [arena.add(p), 0, 0]))
            records.append(dict(name=f"pool{out_size}{'-packed' if packed else '-codes'}", variant="pool", flags=0,
                                checks=[("PQ", uq, want)]))
    return records


def elem_probe(arena, dispatches, rng):
    """Random and edge inputs of exp2 / log2 / rcp; expected values from the exact reference."""
    out = []
    xe = np.concatenate([rng.uniform(-130, 130, 300000), rng.uniform(-2, 2, 100000),
                         np.arange(-126, 128, dtype=np.float64)]).astype(np.float32)
    xe = np.concatenate([xe, np.float32([np.nan, np.inf, -np.inf, 0, -0.0, 2.0 ** -149, -126, 127.99999])])
    xl = np.concatenate([np.exp2(rng.uniform(-126, 128, 300000)), 1 + rng.uniform(-2 ** -8, 2 ** -8, 100000)])
    xl = np.concatenate([xl.astype(np.float32), np.float32([0, -0.0, -1, np.inf, np.nan, 2.0 ** -149, 1, 2, 0.5])])
    bits = rng.integers(0, 2 ** 32, 300000, dtype=np.uint64).astype(np.uint32)
    xr = np.concatenate([bits.view(np.float32), np.float32([0, -0.0, np.inf, -np.inf, np.nan, 2.0 ** 126,
                                                            2.0 ** 127, 2.0 ** -126, 2.0 ** -149])])
    for mode, (name, x, fn) in enumerate((("exp2", xe, fx.exp2), ("log2", xl, fx.log2), ("rcp", xr, fx.rcp))):
        n = len(x)
        off = arena.add(np.concatenate([x.view(np.uint32), np.zeros(n, np.uint32)]))
        dispatches.append(("elem", (n + BLOCK - 1) // BLOCK, [off, mode, n]))
        with np.errstate(all="ignore"):
            want = fn(x).astype(np.float32)
        out.append((name, off, n, x, want))
    return out


def bilin_probe(arena, dispatches, rng, sym):
    """Bilinear samples of 52.1.4 on random textures, with many coordinates in the clamped edge cells."""
    out = []
    for k, (h, w, fmt) in enumerate(((1, 1, "f16"), (3, 5, "f32"), (7, 4, "unorm8"), (9, 13, "f16"), (2, 33, "rgb10a2"))):
        if fmt == "f16":
            d = rng.uniform(-2, 2, (h, w, 4)).astype(np.float16)
            d.reshape(-1)[::11] = [np.inf, -np.inf, -0.0, 0.0][k % 4]
        elif fmt == "f32":
            d = rng.uniform(-2, 2, (h, w, 4)).astype(np.float32)
        elif fmt == "unorm8":
            d = rng.integers(0, 256, (h, w, 4)).astype(np.uint8)
        else:
            d = rng.integers(0, 1024, (h, w, 4)).astype(np.uint16)
            d[..., 3] &= 3
        surf = rr.Surface(d, fmt)
        n = 4000
        uv = np.concatenate([rng.uniform(-0.1, 1.1, (n // 2, 2)), rng.uniform(0, 1.0 / max(w, h), (n // 2, 2))])
        uv = uv.astype(np.float32)
        uv[:16] = [[np.nan, 0.5], [0.5, np.nan], [np.inf, -np.inf], [0, 0], [1, 1], [-0.0, 0.0]] + [[0.5 / w, 0.5 / h]] * 10
        up = surface_upload(arena, surf)
        p = params(sym, sizes=rr.sizes((2, 2)), render=(1, 1), flags=0, res=dict(LAYER=up))
        buf = arena.add(np.concatenate([uv.reshape(-1).view(np.uint32), np.zeros(4 * n, np.uint32)]))
        dispatches.append(("bilin", (n + BLOCK - 1) // BLOCK, [arena.add(p), buf, n]))
        want = rr.bilinear(rr.read(surf), uv[:, 0].astype(np.float64), uv[:, 1].astype(np.float64)).astype(np.float32)
        out.append((f"bilinear-{fmt}-{w}x{h}", buf + 8 * n, want))
    return out


def arith_probe(arena, dispatches, rng):
    """f32 helpers of rr_resolve_math.hlsli vs reference/f32x.py: random over many binades plus the flush
    boundary, signed zeros, infinities, NaN, halves (rnd) and binary16 fma."""
    n = 200000
    e = rng.uniform(-70, 70, (n, 3))
    v = (np.sign(rng.uniform(-1, 1, (n, 3))) * np.exp2(e)).astype(np.float32)
    edge = []
    for u in range(0x00FFFFF0, 0x01000010):
        edge.append([np.array(u, np.uint32).view(np.float32), 0.5, 0.0])
    for bb in (0x3F7FFFFC, 0x3F7FFFFD, 0x3F7FFFFE, 0x3F7FFFFF):
        edge.append([np.array(0x00800001, np.uint32).view(np.float32), np.array(bb, np.uint32).view(np.float32), 0.0])
    for k in range(64):
        edge.append([np.float32(2.0 ** -63 * (1 + k / 64)), np.float32(2.0 ** -63 * (1 - k / 128)), 0.0])
    sp = [0.0, -0.0, np.inf, -np.inf, np.nan, 0.5, -0.5, 1.5, -2.5, 0.49999997, 2.0 ** -149, 65520.0, 65519.0]
    for x in sp:
        for y in sp:
            edge.append([x, y, sp[(sp.index(y) + 3) % len(sp)]])
    v = np.concatenate([v, np.array(edge, np.float32), rng.uniform(-4, 4, (20000, 3)).astype(np.float32)])
    off = arena.add(np.concatenate([v.reshape(-1).view(np.uint32), np.zeros(8 * len(v), np.uint32)]))
    dispatches.append(("arith", (len(v) + BLOCK - 1) // BLOCK, [off, 0, len(v)]))
    a, b, c = v[:, 0], v[:, 1], v[:, 2]
    with np.errstate(all="ignore"):
        want = np.zeros((len(v), 8), np.uint32)
        want[:, 0] = fx.mul(a, b).view(np.uint32)
        want[:, 1] = fx.add(a, b).view(np.uint32)
        want[:, 2] = fx.fma(a, b, c).view(np.uint32)
        want[:, 3] = fx.div(a, b).view(np.uint32)
        want[:, 4] = fx.ftz(fx.rnd_half_away(fx.ftz(a))).view(np.uint32)
        want[:, 5] = fx.fmin(fx.ftz(a), fx.ftz(b)).astype(np.float32).view(np.uint32)
        want[:, 6] = fx.fmax(fx.ftz(a), fx.ftz(b)).astype(np.float32).view(np.uint32)
        h = fx.f16(v.astype(np.float64))
        want[:, 7] = fx.f16(fx.fma16(h[:, 0], h[:, 1], h[:, 2])).astype(np.float16).view(np.uint16)
    return off + 12 * len(v), want


def run_fixtures(json_path, exact=True, mutation=0):
    defines = {"WG_BLOCK": BLOCK, "RR_ELEM_EXACT": int(exact), "RR_MUT": mutation}
    kernels = {name: (build.compile_kernel("rr_resolve.hlsl", "rr_resolve_" + name, defines,
                                           tag=f"rr_resolve_{name}_e{int(exact)}m{mutation}"), NARGS)
               for name in ("main", "pool", "elem", "bilin", "arith")}
    arena = build.Arena()
    dispatches, records = [], []
    rng = np.random.default_rng(54109)
    from ..tests.test_resolve import symbols
    sym = symbols()
    for name, out_size, render, alpha, frames in sequences():
        print("reference", name, flush=True)
        records += build_sequence(arena, dispatches, kernels, name, out_size, render, alpha, frames, sym, rng)
    records += pool_fixtures(arena, dispatches, sym, rng)
    probes = elem_probe(arena, dispatches, rng)
    bil = bilin_probe(arena, dispatches, rng, sym)
    ar_off, ar_want = arith_probe(arena, dispatches, rng)
    print(f"GPU: {len(dispatches)} dispatches, arena {arena.size >> 10} KiB", flush=True)
    img, log = build.run(arena, kernels, dispatches, require_double=True,
                         workdir=str(Path(build.BUILD) / f"resolve-run-e{int(exact)}m{mutation}"))
    rows, total_values, total_bad = [], 0, 0
    for r in records:
        checks = []
        for label, up, want in r["checks"]:
            got = up.read(img)
            ok, bad = same_bits(got, np.asarray(want).astype(got.dtype) if got.dtype != np.asarray(want).dtype else want)
            checks.append(dict(what=label, values=int(np.asarray(want).size), mismatches=bad, exact=ok))
            total_values += int(np.asarray(want).size)
            total_bad += max(bad, 0) if bad >= 0 else int(np.asarray(want).size)
        rows.append(dict(case=r["name"], variant=r["variant"], flags=r["flags"], checks=checks,
                         passed=all(c["exact"] for c in checks)))
    elem = []
    for name, off, n, x, want in probes:
        got = img[off + 4 * n: off + 8 * n].copy().view(np.float32)
        ok, bad = same_bits(got, want)
        idx = np.flatnonzero(~((np.isnan(got) & np.isnan(want)) | (got.view(np.uint32) == want.view(np.uint32))))[:10]
        elem.append(dict(function=name, values=n, mismatches=bad, exact=ok,
                         first=[dict(x=float(x[i]), gpu=float(got[i]), ref=float(want[i])) for i in idx]))
    for name, off, want in bil:
        got = img[off:off + want.nbytes].copy().view(np.float32).reshape(want.shape)
        ok, bad = same_bits(got, want)
        elem.append(dict(function=name, values=int(want.size), mismatches=bad, exact=ok, first=[]))
    got = img[ar_off:ar_off + ar_want.nbytes].copy().view(np.uint32).reshape(ar_want.shape)
    names = ["mul", "add", "fma", "div", "rnd", "min", "max", "fma16"]
    for k, nm in enumerate(names):
        g, w = got[:, k], ar_want[:, k]
        if nm == "fma16":
            nan = ((g & 0x7FFF) > 0x7C00) & ((w & 0x7FFF) > 0x7C00)
        else:
            nan = ((g & 0x7FFFFFFF) > 0x7F800000) & ((w & 0x7FFFFFFF) > 0x7F800000)
        bad = int(np.count_nonzero(~(nan | (g == w))))
        elem.append(dict(function="arith-" + nm, values=int(len(g)), mismatches=bad, exact=bad == 0, first=[]))
    passed = all(r["passed"] for r in rows) and all(e["exact"] for e in elem)
    report = dict(passed=passed, elem_exact=exact, mutation=mutation, values=total_values, mismatching_values=total_bad,
                  cases=rows, elementary=elem, host_log=log,
                  sources={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in
                           (Path(__file__), Path(build.KDIR) / "rr_resolve.hlsl", Path(build.KDIR) / "rr_resolve_math.hlsli",
                            Path(rr.__file__))},
                  scope="synthetic page-54b fixtures; chained histories; invented symbols; no learned data, no game")
    Path(json_path).parent.mkdir(parents=True, exist_ok=True)
    Path(json_path).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("PASS" if passed else "FAIL", f"{total_values} values, {total_bad} mismatching", flush=True)
    for r in rows:
        if not r["passed"]:
            print(r, flush=True)
    for e in elem:
        print(e["function"], e["values"], "mismatches", e["mismatches"], e["first"][:3], flush=True)
    return 0 if passed else 1


def run_exhaustive(func, json_path, chunk_log2=29, cap=1 << 18):
    """Fast path vs double-double decision over every relevant f32 input (chunks of 2^chunk_log2 patterns)."""
    defines = {"WG_BLOCK": 256, "RR_EXH_LOG2": int(func == "log2")}
    kern = build.compile_kernel("rr_resolve.hlsl", "rr_resolve_exhaustive", defines, tag=f"rr_resolve_exh_{func}")
    if func == "exp2":
        ranges = [(0x00000000, 0x43000000), (0x80000000, 0xC2FC0001)]
    else:
        ranges = [(0x00800000, 0x7F800000)]
    totals = dict(mismatch=0, window=0, hard=0)
    hard_pairs = []
    step = 1 << chunk_log2
    for lo, hi in ranges:
        for start in range(lo, hi, step):
            count = min(step, hi - start)
            arena = build.Arena()
            ctr = np.zeros(8 + 2 * cap, np.uint32)
            ctr[4] = cap
            off = arena.add(ctr)
            groups = (count + 255) // 256
            img, _ = build.run(arena, {"x": (kern, NARGS)}, [("x", groups, [off, start, count])], require_double=True,
                               workdir=str(Path(build.BUILD) / f"resolve-exh-{func}"), timeout=3600)
            c = img[off:off + ctr.nbytes].copy().view(np.uint32)
            totals["mismatch"] += int(c[0]); totals["window"] += int(c[1]); totals["hard"] += int(c[2])
            listed = min(int(c[3]), cap)
            hard_pairs.append(c[8:8 + 2 * listed].reshape(-1, 2).copy())
            print(f"{func} {start:08x}+{count:#x}: mismatch {c[0]} window {c[1]} hard {c[2]}", flush=True)
            if int(c[3]) > cap:
                raise SystemExit("hard-case list overflow: raise cap")
    pairs = np.concatenate(hard_pairs) if hard_pairs else np.zeros((0, 2), np.uint32)
    x = pairs[:, 0].view(np.float32)
    got = pairs[:, 1].view(np.float32)
    fn = fx.exp2 if func == "exp2" else fx.log2
    want = fn(x).astype(np.float32)
    ok = (got.view(np.uint32) == want.view(np.uint32)) | (np.isnan(got) & np.isnan(want))
    report = dict(function=func, **totals, hard_checked_against_exact=int(len(x)),
                  hard_mismatching_exact=int(np.count_nonzero(~ok)),
                  passed=totals["mismatch"] == 0 and bool(np.all(ok)))
    Path(json_path).parent.mkdir(parents=True, exist_ok=True)
    Path(json_path).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report), flush=True)
    return 0 if report["passed"] else 1


def run_bench(json_path, out_size=(2560, 1440), render=(1706, 960), reps=20):
    """GPU time of the resolve passes at a game size (synthetic data, no reference): plain, extended with a layer
    + opacity, extended with layer + alpha upscaling + separate alpha, quarter pool; exact and hardware
    elementary modes. One host call per mode."""
    from ..tests.test_resolve import symbols
    sym = symbols()
    rng = np.random.default_rng(1)
    sz = rr.sizes(out_size)
    wr, hr = render
    (h1, w1), (h2, w2) = rr.pooled_sizes(out_size)
    report = {}
    for exact in (1, 0):
        defines = {"WG_BLOCK": BLOCK, "RR_ELEM_EXACT": exact, "RR_MUT": 0}
        kernels = {n: (build.compile_kernel("rr_resolve.hlsl", "rr_resolve_" + n, defines,
                                            tag=f"rr_resolve_{n}_e{exact}m0"), NARGS) for n in ("main", "pool")}
        arena = build.Arena()
        Q = Upload(arena, rng.uniform(0, 2, (sz.Hh, sz.Wh, 3)).astype(np.float16), "f16")
        OUT = Upload(arena, np.zeros((sz.He, sz.We, 4), np.float16), "f16")
        AOUT = Upload(arena, np.zeros((sz.He, sz.We, 1), np.float16), "f16")
        LAY = Upload(arena, rng.uniform(0, 1, (hr, wr, 4)).astype(np.float16), "f16")
        OPA = Upload(arena, rng.integers(0, 256, (hr, wr, 4)).astype(np.uint8), "unorm8")
        col = rng.integers(0, 1024, (hr, wr, 4)).astype(np.uint16)
        col[..., 3] &= 3
        COL = Upload(arena, col, "rgb10a2")
        MV = Upload(arena, rng.uniform(-3, 3, (hr, wr, 4)).astype(np.float16), "f16")
        P = Upload(arena, rng.uniform(0.2, 2, (hr, wr, 4)).astype(np.float16), "f16")
        G = Upload(arena, rng.integers(0, 256, (hr, wr, 4)).astype(np.uint8), "unorm8")
        TH = [Upload(arena, np.zeros((sz.He, sz.We, 4), np.float16), "f16") for _ in range(2)]
        OH = [Upload(arena, np.zeros((sz.He, sz.We, 4), np.float16), "f16") for _ in range(2)]
        AH = [Upload(arena, np.zeros((sz.He, sz.We, 1), np.float16), "f16") for _ in range(2)]
        PH1 = Upload(arena, rng.integers(0, 1024, (h1, w1, 4)).astype(np.uint16) & np.uint16(1023), "rgb10a2",
                     packed=False)
        PQ = Upload(arena, np.zeros((h2, w2, 4), np.uint16), "rgb10a2", packed=False)
        base = dict(Q=Q, OUT=OUT, MV=MV, P=P, G=G, THr=TH[0], THw=TH[1], OHr=OH[0], OHw=OH[1], AHr=AH[0], AHw=AH[1])
        configs = [("plain", 0, {}),
                   ("layer+opacity", F_EXT | F_LAYER | F_OPAC, dict(LAYER=LAY, OPAC=OPA)),
                   ("layer+alpha+sep", F_EXT | F_LAYER | F_ALPHA | F_AOUT, dict(LAYER=LAY, ASRC=COL, AOUT=AOUT))]
        disp = []
        for name, flags, extra in configs:
            p = params(sym, sizes=sz, render=render, flags=flags, E=np.float32(1.25), Ep=np.float32(1.0),
                       jitter=(np.float32(0.125), np.float32(-0.25)), alpha_channel=3, res={**base, **extra})
            disp.append(("main", (sz.We * sz.He + BLOCK - 1) // BLOCK, [arena.add(p), 0, 0]))
        pp = params(sym, sizes=sz, render=(1, 1), flags=0, res=dict(PH1=PH1, PQ=PQ))
        disp.append(("pool", (w2 * h2 + BLOCK - 1) // BLOCK, [arena.add(pp), 0, 0]))
        _, log = build.run(arena, kernels, disp, reps=reps, require_double=True,
                           workdir=str(Path(build.BUILD) / f"resolve-bench-e{exact}"))
        times = {}
        for line in log.splitlines():
            if line.startswith("time "):
                f = line.split()
                times[int(f[1])] = dict(best_ms=float(f[4]), mean_ms=float(f[7]))
        names = [c[0] for c in configs] + ["quarter-pool"]
        report["exact" if exact else "hardware"] = {names[i]: times.get(i) for i in range(len(names))}
        print("exact" if exact else "hardware", report["exact" if exact else "hardware"], flush=True)
    report.update(output=out_size, render=render, reps=reps)
    Path(json_path).parent.mkdir(parents=True, exist_ok=True)
    Path(json_path).write_text(json.dumps(report, indent=2), encoding="utf-8")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", required=True)
    ap.add_argument("--exhaustive", choices=["exp2", "log2"])
    ap.add_argument("--bench", action="store_true", help="GPU time at 2560x1440 output, 1706x960 render")
    ap.add_argument("--hardware-elem", action="store_true", help="RR_ELEM_EXACT=0 (expect elementary differences)")
    ap.add_argument("--mutation", type=int, default=0, choices=[0, 1, 2, 3],
                    help="negative control: 1 history copy, 2 half-way test, 3 bilinear fraction (must FAIL)")
    a = ap.parse_args()
    if "WG_RR_BUILD" not in os.environ:
        ap.error("set WG_RR_BUILD to a private build directory")
    if a.bench:
        return run_bench(a.json)
    if a.exhaustive:
        return run_exhaustive(a.exhaustive, a.json)
    return run_fixtures(a.json, exact=not a.hardware_elem, mutation=a.mutation)


if __name__ == "__main__":
    raise SystemExit(main())
