# SPDX-License-Identifier: MIT
"""Rates of the exact-body levers L1-L4 (results/2026-10-21-body-levers.md), measured BEFORE building, in NumPy on a
sequence body dump (`sequence.py --dump-body`; stage inputs rebuilt as in gpu/ratesim.py, IN0 skipped). Nothing here
feeds a kernel.

Per matrix site (q, v, O, S, Wo, W1, W2, Wm, Wh, Wa / Wup) and kernel unit (wave = 16 tokens, step = 32 inputs,
K-block = 16 inputs, tile = 16 outputs):

L4 (exact INTEGER WMMA, v_wmma_i32_16x16x16_iu8 / _16x16x32_iu4). Every e4m3 value is M 2^(E - 10), M = 0..15,
E = max(e, 1). Scales may differ per ROW of A and per COLUMN of B (the WMMA sums over K only): activations get one base
exponent per token (the token's smallest E over the step), weights one base per (tile, step) (tile-uniform: one scale
per lane at the end) or per output row. A value is then the integer M 2^(E - base), split into signed digits of 7 bits
(iu8: |digit| <= 127) or 3 bits (iu4 signed: |digit| <= 7); digit count d(span) = the digits needed for 4 + span bits.
One WMMA per digit pair: WMMAs per K-block = dA x dB (wave-uniform: max over the wave's tokens / the tile's rows).
Reported: the distribution of dB (activation side), dA (weight side) and of dA x dB, the mean WMMA count per K-block
(fp8 step E: 1), and the share of outputs whose exact sum in grid units N reaches 2^24 (f32 conversion inexact:
fallback) or 2^31 (int32 overflow).

L1 / L2 / L3 (certify-and-fix on a hardware chain m = WMMA(WMMA(0)) with R(c + m) by one v_fma_mix): an output is
flagged when the interval [v - B, v + B] around its exact step value v = c + sum is not mapped to ONE value by the
rounding that matters: R16 (L1, L2), or Q8 o R16 for the last step of a chain consumed only through Q8 (L3: q, v, O,
Wa, Wup, Wm). B = eps T with T = sum |products| (L1: per output, needs an extra |a| |b| product) or T2 = L1row x
max|b| of the token (L2: row sums precomputed, one max per token); eps = 2^-18 / 2^-20 / 2^-23 (hardware error model
of the zero-start chain, see the results note). Reported: flagged outputs and 16 x 16 tiles with a flagged output.

    python -m clean.rr_f.gpu.lever_rates body.npz [--model synth|b2] [--frames 1,2,4,7]"""
import argparse
import collections
import sys

import numpy as np

from clean.rr_f.gpu import ratesim as RT
from clean.rr_f.gpu import sequence as SQ
from clean.rr_f.reference import block as RB, stages as RS
from clean.rr_f.reference.fmt import D8_TABLE, q8, round_f16

EXP = np.zeros(256, np.int64)        # E = max(e, 1) of a nonzero, non-NaN code; 0 marks zero / NaN
for _c in range(256):
    if (_c & 0x7F) not in (0, 0x7F):
        EXP[_c] = max((_c >> 3) & 0xF, 1)
MAG = np.abs(np.nan_to_num(D8_TABLE)).astype(np.float64)
EPS = (-18, -20, -23)
FINAL_Q8 = {"q", "v", "O", "Wa", "Wup", "Wm"}       # last step consumed only through Q8


def digits8(span):
    return np.where(span < 0, 0, np.ceil((4 + np.maximum(span, 0)) / 7.0)).astype(np.int64)


def digits4(span):
    return np.where(span < 0, 0, np.ceil((4 + np.maximum(span, 0)) / 3.0)).astype(np.int64)


def span(codes, axis):
    """Exponent span (max E - min E over nonzero codes) along axis; -1 when all zero."""
    e = EXP[codes]
    hi = e.max(axis=axis)
    lo = np.where(e == 0, 99, e).min(axis=axis)
    return np.where(hi == 0, -1, hi - lo), np.where(hi == 0, 0, lo)


L4 = collections.defaultdict(lambda: collections.defaultdict(float))
CF = collections.defaultdict(lambda: np.zeros(1 + 4 * len(EPS)))
CG = collections.defaultdict(lambda: np.zeros(9))   # outputs, !ideal, !gate2, (flag16, flag8) at 2^-18, 2^-20, tiles


def lowbit(v):
    """Exponent of the lowest set bit of each value (exact float64 e4m3 values); inf for zeros."""
    a = np.abs(np.asarray(v, np.float64))
    m, e = np.frexp(a)
    sig = np.round(np.ldexp(m, 8)).astype(np.int64)                # e4m3 significands: at most 4 bits, exact
    low = np.where(sig == 0, 1, sig & -sig)
    return np.where(a == 0, np.inf, e - 8 + np.log2(low))


def l4_stats(site, X, W, final):
    """X: activation codes [nw, 16, K] (16 tokens of a wave); W: codes [K, N] or [nw, K, N] (bounded operand)."""
    nw, _, K = X.shape
    pw = W.ndim == 3
    N = W.shape[-1]
    a = L4[site]
    for s0 in range(0, K, 32):
        xs = X[..., s0:s0 + 32]
        ws = W[..., s0:s0 + 32, :] if pw else W[s0:s0 + 32, :]
        sp_tok, lo_tok = span(xs, axis=2)                           # [nw, 16]
        dB8 = digits8(sp_tok).max(axis=1)                           # [nw]
        dBw = digits8(span(xs.reshape(nw, -1), axis=1)[0])          # [nw] one base per wave (the kernel's choice)
        dB4 = digits4(sp_tok).max(axis=1)
        # per K-block (16) bases instead of per step
        dB8k = np.stack([digits8(span(xs[..., k:k + 16], 2)[0]).max(axis=1) for k in (0, 16)], 1)   # [nw, 2]
        for t0 in range(0, N, 16):
            wt = ws[..., t0:t0 + 16]                                # [(nw,) 32, 16]
            if pw:
                sp_t, lo_t = span(wt.reshape(nw, -1), axis=1)       # [nw]
                sp_r, lo_r = span(wt, axis=1)                       # [nw, 16] per output row
                dA8, dA4 = digits8(sp_t), digits4(sp_t)
                dA8r = digits8(sp_r).max(axis=1)
            else:
                sp_t, lo_t = span(wt.reshape(-1), axis=0)
                sp_r, lo_r = span(wt, axis=0)                       # [16]
                dA8 = np.full(nw, digits8(np.array(sp_t)))
                dA4 = np.full(nw, digits4(np.array(sp_t)))
                dA8r = np.full(nw, digits8(sp_r).max())
                lo_t = np.full(nw, lo_t)
            # exact partial sums in grid units (grid = 2^(base_tile + base_token - 20)), from integers
            xv = np.where(EXP[xs] > 0, D8_TABLE[xs], 0.0)
            wv = np.where(EXP[wt] > 0, D8_TABLE[wt], 0.0)
            if pw:
                S = np.einsum("wtk,wkn->wtn", xv, wv)
                gexp = lo_t[:, None, None] + lo_tok[:, :, None] - 20
            else:
                S = xv @ wv
                gexp = lo_t[:, None, None] + lo_tok[:, :, None] - 20
            Ng = np.abs(np.ldexp(S, -gexp.astype(np.int64)))
            a["units"] += nw
            a["kblocks"] += 2 * nw
            for d in range(5):
                a[f"dB8={d}"] += (dB8 == d).sum()
                a[f"dA8={d}"] += (dA8 == d).sum()
                a[f"dA8row={d}"] += (dA8r == d).sum()
            a["wmma8"] += 2 * (np.maximum(dA8, 1) * np.maximum(dB8, 1)).sum()
            a["wmma8w"] += 2 * (np.maximum(dA8, 1) * np.maximum(dBw, 1)).sum()
            a["wmmaK"] += 2 * (2 * np.where(dBw > 2, 3, 2)).sum()           # as built: 2 weight planes x (2 or 3)
            for d in range(5):
                a[f"dBw={d}"] += (dBw == d).sum()
            a["wmma8row"] += 2 * (np.maximum(dA8r, 1) * np.maximum(dB8, 1)).sum()
            a["wmma8kb"] += (np.maximum(dA8, 1)[:, None] * np.maximum(dB8k, 1)).sum()
            a["wmma4"] += (np.maximum(dA4, 1) * np.maximum(dB4, 1)).sum()     # 16x16x32: one per 32 inputs, 2x rate
            a["outs"] += Ng.size
            a["N>=2^24"] += (Ng >= 2.0 ** 24).sum()
            a["N>=2^31"] += (Ng >= 2.0 ** 31).sum()
            a["units_N24"] += (Ng >= 2.0 ** 24).any(axis=(1, 2)).sum()


def cf_stats(site, x, W, c0, final):
    """Certify rates (see the module text); x, W decoded values, the chain over K in steps of 32."""
    x = np.asarray(x, np.float64)
    W = np.asarray(W, np.float64)
    K = x.shape[-1]
    acc = None
    nst = K // 32
    for si, k0 in enumerate(range(0, K, 32)):
        xs, Ws = np.nan_to_num(x[..., k0:k0 + 32]), np.nan_to_num(W[..., k0:k0 + 32, :])
        part = np.matmul(xs, Ws)
        T = np.matmul(np.abs(xs), np.abs(Ws))
        L1row = np.abs(Ws).sum(axis=-2)                              # [..., N]
        bmax = np.abs(xs).max(axis=-1)                               # [..., T] per token
        T2 = bmax[..., :, None] * L1row[..., None, :]
        if acc is None:
            acc = np.zeros(part.shape) if c0 is None else np.broadcast_to(np.asarray(c0, np.float64), part.shape)
        v = acc + part
        r = round_f16(v)
        last = si == nst - 1 and final
        row = [v.size]
        for e in EPS:
            for TT in (T, T2):
                B = np.ldexp(TT, e)
                lo, hi = round_f16(v - B), round_f16(v + B)
                f16flag = lo != hi
                if last:
                    q8flag = q8(lo) != q8(hi)
                else:
                    q8flag = f16flag
                row += [f16flag.sum(), q8flag.sum()]
        # tiles: only for eps 2^-20 with T2 (L2), recorded separately
        B = np.ldexp(T2, -20)
        fl = (round_f16(v - B) != round_f16(v + B))
        fl = fl.reshape(-1, fl.shape[-2], fl.shape[-1])
        R_, N_ = fl.shape[-2], fl.shape[-1]
        fl = np.pad(fl, ((0, 0), (0, (-R_) % 16), (0, (-N_) % 16)))
        tiles = fl.reshape(fl.shape[0], fl.shape[1] // 16, 16, fl.shape[2] // 16, 16).any(axis=(2, 4))
        CF[site + ("/last" if last else "")] += np.array(row[:1 + 4 * len(EPS)], np.float64)
        CF["tiles"] += np.r_[tiles.size, tiles.sum(), np.zeros(4 * len(EPS) - 1)]
        # certify by the H1 gate first: an output whose zero-start chain is exact by H1 needs no bound.
        #   ideal: sum |products| < 2^(lowest product bit + 24) (exact q, exact T)
        #   gate2: L1row * max|b| < 2^(qa + qb + 24), qa / qb = lowest set bit over the row's / token's 32 values
        qx = lowbit(xs)                                              # [..., T, 32] (inf for zeros)
        qw = lowbit(np.swapaxes(Ws, -1, -2))                         # [..., N, 32]
        qa = qw.min(axis=-1)                                         # [..., N]
        qb = qx.min(axis=-1)                                         # [..., T]
        with np.errstate(invalid="ignore", over="ignore"):
            gate2 = (T2 < np.exp2(qb[..., :, None] + qa[..., None, :] + 24)) | (T2 == 0)
        qprod = np.full(T.shape, np.inf)                             # [..., T, N]: min over k of qx[t, k] + qw[n, k]
        for k in range(32):
            qprod = np.minimum(qprod, qx[..., :, k, None] + qw[..., None, :, k])
        with np.errstate(invalid="ignore", over="ignore"):
            ideal = (T < np.exp2(qprod + 24)) | (T == 0)
        g = [v.size, (~ideal).sum(), (~gate2).sum()]
        for e in (-18, -20):
            B = np.ldexp(T2, e)
            lo, hi = round_f16(v - B), round_f16(v + B)
            fl16 = lo != hi
            fl8 = (q8(lo) != q8(hi)) if last else fl16
            g += [(fl16 & ~gate2).sum(), (fl8 & ~gate2).sum()]
            if e == -20:
                f = (fl8 & ~gate2).reshape(-1, fl8.shape[-2], fl8.shape[-1])
                f = np.pad(f, ((0, 0), (0, (-f.shape[1]) % 16), (0, (-f.shape[2]) % 16)))
                tl = f.reshape(f.shape[0], f.shape[1] // 16, 16, f.shape[2] // 16, 16).any(axis=(2, 4))
                g += [tl.size, tl.sum()]
        CG[site + ("/last" if last else "")] += np.array(g, np.float64)
        acc = r


CUR = {}


def make_mm(orig):
    def mm(x, W, c0=None):
        f = sys._getframe(1)
        fn = f.f_code.co_name
        name = None
        if fn == "window_block":
            i, H = CUR["blk_i"], CUR["H"]
            CUR["blk_i"] = i + 1
            name = ("q", "v", "O")[i % 3] if i < 3 * H else ("Wo", "W1", "W2")[i - 3 * H]
        elif fn == "merge_windows":
            name = "Wm"
        elif fn == "hidden_projection":
            name = "Wh"
        elif fn == "run_stage":
            name = "Wup" if CUR["info"]["kind"] == "fuse" else "Wa"
        if name is not None:
            final = name in FINAL_Q8
            cf_stats(name, x, W, c0, final)
            xc = np.asarray(q8(x), np.uint8)
            Wc = np.asarray(q8(W), np.uint8)
            if name == "O":                      # A = v^T (rows = features), B = P (columns = queries)
                Pw = xc.reshape(-1, 16, xc.shape[-1])               # [nw*4, 16 queries, 64 keys]
                Vw = np.repeat(Wc, 4, axis=0)                       # [nw*4, 64 keys, 32 features]
                l4_stats(CUR["stage"] + ":O", Pw, Vw, final)
            else:
                Ww = Wc.reshape(-1, Wc.shape[-1]) if Wc.ndim == 2 else Wc
                l4_stats(CUR["stage"] + ":" + name, RT.waves(xc), Ww, final)
        return orig(x, W, c0)
    return mm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dump")
    ap.add_argument("--model", default="synth")
    ap.add_argument("--frames", default="1,2,4,7")
    a = ap.parse_args()
    if a.model == "synth":
        m = SQ.model_synthetic()
    else:
        ns = argparse.Namespace(synthetic=False, b2=False, model=SQ.DEFAULT_MODEL, map=SQ.DEFAULT_MAP)
        m = SQ.load_model(ns)
    d = np.load(a.dump)
    RB.mm = make_mm(RB.mm)
    RS.mm = make_mm(RS.mm)
    origwb = RB.window_block

    def wb(x, xr, W, ra, mm_, wc, Z_override=None, trace=False):
        CUR["blk_i"], CUR["H"] = 0, np.asarray(W["Wq"]).shape[0]
        y2, info = origwb(x, xr, W, ra, mm_, wc, Z_override=Z_override, trace=True)
        q = np.asarray(info["q"], np.uint8)                  # [Nw, H, 64, 32]
        for h in range(q.shape[1]):
            qq = q[:, h]
            Xw = qq.reshape(-1, 16, 32)                         # queries (B side, per column)
            Kw = np.repeat(np.transpose(qq, (0, 2, 1)), 4, axis=0)   # keys as [K = 32 features][N = 64 keys]
            l4_stats(CUR["stage"] + ":S", Xw, Kw, False)
            qv = D8_TABLE[qq]
            cf_stats("S", qv, np.transpose(qv, (0, 2, 1)), None, False)
        return y2, info
    RS.window_block = wb

    def f16(k):
        return d[k].view(np.float16).astype(np.float64)
    for fr in [int(v) for v in a.frames.split(",")]:
        p = f"f{fr}_"
        for name in SQ.RN.ORDER[:-1]:
            if name == "IN0":
                continue
            info = RS.stage_info(name)
            lvl = info["level"]
            if name.startswith("E"):
                if lvl == 1:
                    y = d[p + "IN0_y2"]
                    kids = [y[dy::2, dx::2] for dy in (0, 1) for dx in (0, 1)]
                    inputs = {"in": np.concatenate(kids, axis=-1), "hold": f16(p + "IN0_hmerged")}
                else:
                    inputs = {"x": d[p + f"E{lvl - 1}_merged"], "hold": f16(p + f"E{lvl - 1}_hmerged")}
            elif name.endswith("-F"):
                par = "E5" if lvl == 4 else f"D{lvl + 1}-R"
                enc = "IN0" if lvl == 0 else f"E{lvl}"
                inputs = {"cin": d[p + par + "_y2"], "skip": d[p + enc + "_y2"]}
            else:
                par = "E5" if lvl == 4 else f"D{lvl + 1}-R"
                enc = "IN0" if lvl == 0 else f"E{lvl}"
                inputs = {"x": d[p + f"D{lvl}-F_y2"], "hold": f16(p + enc + "_hnew"), "Hc": f16(p + par + "_hnew")}
            CUR.update(stage=name, W=m.weights[name], info=info)
            RS.run_stage(name, inputs, m.weights[name], m.constants[name], m.wc, (0, 0))
    # ---- report
    print("L4: per site kind; WMMAs per 16-input K-block (step E: 1). int8 tile base / row base / per-K-block token "
          "base; int4 (16x16x32 at 2x: cost = WMMAs / 2 per K-block); dB / dA distributions (digits 1 / 2 / 3)")
    kinds = collections.defaultdict(lambda: collections.defaultdict(float))
    for k, v in L4.items():
        for kk, vv in v.items():
            kinds[k.split(":")[1]][kk] += vv
            kinds["all"][kk] += vv
    print("as built (wave base, 2 weight planes x 2 or 3 activation planes; S / O: 2x2 or 3x3): WMMAs per K-block")
    for k in sorted(kinds, key=lambda s: (s == "all", s)):
        v = kinds[k]
        print(f"  {k:6s} wave-base i8 {v['wmma8w'] / v['kblocks']:5.2f}  as built {v['wmmaK'] / v['kblocks']:5.2f}  "
              + " ".join(f"dBw{d}={v[f'dBw={d}'] / v['units']:.3f}" for d in (1, 2, 3)))
    hdr = (f"{'site':6s} {'kblk':>7s} {'i8':>5s} {'i8row':>5s} {'i8kb':>5s} {'i4/2':>5s}   "
           f"{'dB1':>5s} {'dB2':>5s} {'dB3':>5s}   {'dA1':>5s} {'dA2':>5s} {'dA3':>5s}  {'N>=2^24':>8s} {'unitN24':>8s}")
    print(hdr)
    for k in sorted(kinds, key=lambda s: (s == "all", s)):
        v = kinds[k]
        u = v["units"]
        print(f"{k:6s} {int(v['kblocks']):7d} {v['wmma8'] / v['kblocks']:5.2f} {v['wmma8row'] / v['kblocks']:5.2f} "
              f"{v['wmma8kb'] / v['kblocks']:5.2f} {v['wmma4'] / v['kblocks']:5.2f}   " +
              " ".join(f"{v[f'dB8={d}'] / u:5.3f}" for d in (1, 2, 3)) + "   " +
              " ".join(f"{v[f'dA8={d}'] / u:5.3f}" for d in (1, 2, 3)) +
              f"  {v['N>=2^24'] / v['outs']:8.5f} {v['units_N24'] / u:8.4f}")
    print("\nL1 / L2 / L3: flagged outputs (share). Columns per eps: L1 = bound eps*T (exact sum of |products|), "
          "L2 = eps*L1row*max|b|; f16 = R16 boundary, q8 = Q8 o R16 boundary (last steps feeding Q8 only)")
    hdr = f"{'site':10s} {'outputs':>9s} " + " ".join(f"{'L1f16':>7s} {'L1q8':>7s} {'L2f16':>7s} {'L2q8':>7s}" for _ in EPS)
    print(" " * 21 + "   ".join(f"eps 2^{e:<4d}{'':22s}" for e in EPS))
    print(hdr)
    tot = None
    for k in sorted(CF):
        if k == "tiles":
            continue
        v = CF[k]
        tot = v.copy() if tot is None else tot + v
        print(f"{k:10s} {int(v[0]):9d} " + " ".join(f"{x / v[0]:7.4f}" for x in v[1:]))
    print(f"{'all':10s} {int(tot[0]):9d} " + " ".join(f"{x / tot[0]:7.4f}" for x in tot[1:]))
    t = CF["tiles"]
    print(f"16x16 tiles with an L2 (eps 2^-20) f16 flag: {t[1] / t[0]:.4f} of {int(t[0])}")
    print("\nGate first, then bound: share of outputs failing the per-output H1 test (ideal: exact q and T; gate2: "
          "L1row max|b| < 2^(qa + qb + 24)), and failing gate2 AND near a boundary (eps 2^-18 / 2^-20 x L1row max|b|; "
          "q8 = at Q8 o R16 for last steps feeding Q8); tiles = 16 x 16 tiles with such a flag (2^-20, q8)")
    print(f"{'site':10s} {'outputs':>9s} {'!ideal':>7s} {'!gate2':>7s} {'f16/18':>7s} {'q8/18':>7s} {'f16/20':>7s} "
          f"{'q8/20':>7s} {'tiles':>7s}")
    tot = np.zeros(9)
    for k in sorted(CG):
        v = CG[k]
        tot += v
        print(f"{k:10s} {int(v[0]):9d} " + " ".join(f"{x / v[0]:7.4f}" for x in v[1:7]) + f" {v[8] / max(v[7], 1):7.4f}")
    print(f"{'all':10s} {int(tot[0]):9d} " + " ".join(f"{x / tot[0]:7.4f}" for x in tot[1:7]) + f" {tot[8] / tot[7]:7.4f}")


if __name__ == "__main__":
    main()
