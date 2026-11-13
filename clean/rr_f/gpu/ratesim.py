# SPDX-License-Identifier: MIT
"""Diagnostics (results/2026-10-20-speed-body.md): how often could a 51.1 step of the body be taken on ONE exact fp8
WMMA chain, and how often would a certify-and-fix scheme have to fix? NumPy, on a sequence body dump
(`sequence.py --dump-body`): the stage inputs are rebuilt from the dumped stage outputs (shift (0, 0); IN0 is skipped,
its input x0 is not dumped) and the reference stages run with every matrix step observed. Nothing here is used by a
kernel; the kernels' own decisions are counted on the GPU by gpu/zstat.py.

Per matrix site and (wave = 16 rows, step = 32 inputs, column tile = 16 outputs) unit, the share of units for which
one chain is exact by H1 under a check:
  zcur    variant Z's classes, decided per wave over all inputs (small: codes < T; int: codes 0 or >= 8)
  zstep   Z's classes per (wave, step)
  grid    per (wave, step): L1 (class, step) * Bmax < 2^(gW + gb + 24), gb the wave's exact lowest set bit
  tile    as grid with L1 and gW per (step, column tile)
  ideal   every output: sum |products| < 2^(lowest product bit + 24) (no cheap check can reach more)
  gridE   as grid with gb from the exponent fields only (the kernel gate z2_ok with a per-step host L1)
  gridEG  as gridE with the class-global L1 and gW (z2_ok as built for the variant Z2 experiment)
Sites: q, v (block input x), O (P v), S (scores q k: both operands activations, 32 Amax Bmax bound), Wo, W1, W2,
Wm, Wh, Wa / Wup. Also: the O gate with a bound of 1.1 on the row sums of P, two-chain scores with an adaptive split
grid (S3), and certify-and-fix: share of outputs whose exact step value lies within 2^-20 / 2^-23 (|c| + sum
|products|) of a binary16 rounding boundary, and share of 16 x 16 tiles with such an output.

    python -m clean.rr_f.gpu.ratesim body.npz [--model synth | b2 | b1] [--frames 0,1,...]"""
import argparse
import collections
import os
import sys

import numpy as np

from clean.rr_f.gpu import sequence as SQ, stage
from clean.rr_f.reference import block as RB, stages as RS
from clean.rr_f.reference.fmt import D8_TABLE, q8, round_f16

STEP = 32
LB = np.full(256, np.inf)
for _c in range(256):
    _v = abs(D8_TABLE[_c])
    if np.isfinite(_v) and _v != 0:
        _m, _e = np.frexp(_v)
        _sig = int(round(_m * 16))
        LB[_c] = _e - 4 + (_sig & -_sig).bit_length() - 1
LBE = np.full(256, np.inf)                      # conservative lowest bit from the exponent field only
for _c in range(256):
    if (_c & 0x7F) not in (0, 0x7F):
        LBE[_c] = max((_c >> 3) & 0xF, 1) - 10
MAG = np.abs(np.nan_to_num(D8_TABLE)).astype(np.float64)

acc = collections.defaultdict(lambda: np.zeros(8))   # site -> [units, zcur, zstep, grid, tile, ideal, gridE, gridEG]
CUR = {}


def is_int(a, axis):
    return ((a == 0) | ((a >= 0x50) & (a != 0x7F))).all(axis=axis)


def unit_stats(site, X, W, T, iok, both=False):
    """X: codes [nwave, 16, K]; W: codes [K, N] or [nwave, K, N]."""
    nwv, _, K = X.shape
    pw = W.ndim == 3
    N = W.shape[-1]
    ax, lx = MAG[X], LB[X]
    a = X & 0x7F
    zcur = (a.max(axis=(1, 2)) < T) | (is_int(a, (1, 2)) & iok)
    aw, lw = MAG[W], LB[W]
    res = np.zeros(8)
    L1glob = (aw.reshape(-1, K // STEP, STEP, N) if not pw else aw.reshape(nwv, K // STEP, STEP, N)).sum(axis=-2).max()
    gWglob = lw.min()
    for s0 in range(0, K, STEP):
        sl = slice(s0, s0 + STEP)
        xs, axs, lxs = X[..., sl], ax[..., sl], lx[..., sl]
        as_ = xs & 0x7F
        zstep = (as_.max(axis=(1, 2)) < T) | (is_int(as_, (1, 2)) & iok)
        Bmax = axs.max(axis=(1, 2))
        gb = lxs.min(axis=(1, 2))
        gbe = LBE[xs].min(axis=(1, 2))
        if pw:
            aws, lws = aw[:, sl, :], lw[:, sl, :]
            L1 = aws.sum(axis=1)                         # [nwave, N]
            L1g = L1.max(axis=1)
            gW = lws.min(axis=(1, 2))
        else:
            aws, lws = aw[sl, :], lw[sl, :]
            L1 = aws.sum(axis=0)                         # [N]
            L1g = np.full(nwv, L1.max())
            gW = np.full(nwv, lws.min())
        with np.errstate(invalid="ignore", over="ignore"):
            grid = (L1g * Bmax < np.exp2(gW + gb + 24)) | (Bmax == 0)
            if both:                                     # both operands activations: 32 * Amax * Bmax, actual grids
                Wmax = aws.max(axis=(1, 2))
                gWe = LBE[W[:, sl, :]].min(axis=(1, 2))
                gridE = (STEP * Wmax * Bmax < np.exp2(gWe + gbe + 24)) | (Bmax == 0) | (Wmax == 0)
                gridEG = gridE
            else:
                gridE = (L1g * Bmax < np.exp2(gW + gbe + 24)) | (Bmax == 0)
                gridEG = (L1glob * Bmax < np.exp2(gWglob + gbe + 24)) | (Bmax == 0)
        for t0 in range(0, N, 16):
            ts = slice(t0, t0 + 16)
            if pw:
                L1t = L1[:, ts].max(axis=1)
                gWt = lws[:, :, ts].min(axis=(1, 2))
                sums = np.einsum("wrk,wkn->wrn", axs, aws[:, :, ts])
                q = np.full(sums.shape, np.inf)
                for k in range(xs.shape[-1]):
                    q = np.minimum(q, lxs[:, :, k, None] + lws[:, None, k, ts])
            else:
                L1t = np.full(nwv, L1[ts].max())
                gWt = np.full(nwv, lws[:, ts].min())
                sums = axs @ aws[:, ts]
                q = np.full(sums.shape, np.inf)
                for k in range(xs.shape[-1]):
                    q = np.minimum(q, lxs[:, :, k, None] + lws[None, None, k, ts])
            with np.errstate(invalid="ignore", over="ignore"):
                tile = (L1t * Bmax < np.exp2(gWt + gb + 24)) | (Bmax == 0)
                ideal = ((sums < np.exp2(q + 24)) | (sums == 0)).all(axis=(1, 2))
            res += [nwv, zcur.sum(), zstep.sum(), grid.sum(), tile.sum(), ideal.sum(), gridE.sum(), gridEG.sum()]
    acc[site] += res


S2 = np.zeros(3)
OG = np.zeros(2)
S3 = np.zeros(2)   # tiles, k-unsplit gate, either side


def s2_stats(qq):
    """Scores with ONE operand split (integer + fraction) and the other unsplit: two chains instead of four.
    chain I = sum u * vh (grid 2^gu), chain F = sum u * vl (grid 2^(gu - 9), |vl| <= 1/2)."""
    Nw = qq.shape[0]
    blk = qq.reshape(Nw, 4, 16, 32)
    mag, emn = MAG[blk].max(axis=(2, 3)), LBE[blk].min(axis=(2, 3))   # per (window, wave)
    for J in range(4):
        for wv in range(4):
            U, V = (mag[:, J], emn[:, J]), (mag[:, wv], emn[:, wv])
            def ok(u, v):
                (um, ue), (vm, _) = u, v
                with np.errstate(invalid="ignore", over="ignore"):
                    return ((32 * um * vm < np.exp2(ue + 24)) & (32 * um * 0.5 < np.exp2(ue - 9 + 24)) & (ue <= 0)) | (um == 0)
            a, b = ok(U, V), ok(V, U)
            S2[:] += [Nw, a.sum(), (a | b).sum()]
            # adaptive split grid s: chain H = k qh (grid 2^(gk+s)), chain L = k ql (grid 2^(gk+gq), |ql| <= 2^(s-1))
            def oka(u, v):
                (km, ge), (qm, gq) = u, v
                res = np.zeros(km.shape, bool)
                with np.errstate(invalid="ignore", over="ignore"):
                    for sgrid in range(-9, 10):
                        h = 32 * km * (qm + 2.0 ** (sgrid - 1)) < np.exp2(ge + sgrid + 24)
                        l = 32 * km * 2.0 ** (sgrid - 1) < np.exp2(ge + gq + 24)
                        res |= h & l & (gq <= sgrid)
                return res | (km == 0) | (qm == 0)
            S3[:] += [Nw, (oka(U, V) | oka(V, U)).sum()]


def classT(W, keys):
    mats = []
    for k in keys:
        if k in W:
            arr = np.asarray(W[k], np.uint8)
            mats += list(arr) if arr.ndim == 3 else [arr]
    return stage.z_class(mats)


def waves(x):
    x = np.asarray(x, np.uint8)
    K = x.shape[-1]
    x = x.reshape(-1, K)
    pad = (-x.shape[0]) % 16
    if pad:
        x = np.concatenate([x, np.zeros((pad, K), np.uint8)])
    return x.reshape(-1, 16, K)


KEYS = {"q": ("Wq", "Wv"), "v": ("Wq", "Wv"), "Wo": ("Wo",), "W1": ("W1",), "W2": ("W2",), "Wm": ("Wm",),
        "Wh": ("Wh",), "Wup": ("Wa", "Wup"), "Wa": ("Wa", "Wup")}


CF = collections.defaultdict(lambda: np.zeros(5))   # site kind -> [outputs, flagged b20, flagged b23, tiles, tiles b20]


def cf_stats(kind, x, W, c0):
    """Certify-and-fix rates: the chain's exact step values v = c + sum (c the previous step's f16 result), an error
    bound beta = 2^-k (|c| + sum |products|) of a hardware chain, and the share of outputs whose interval
    [v - beta, v + beta] reaches a binary16 rounding boundary (k = 20: a few f32 roundings; k = 23: one ulp)."""
    x = np.asarray(x, np.float64)
    W = np.asarray(W, np.float64)
    K = x.shape[-1]
    acc = None
    for k0 in range(0, K, 32):
        xs, Ws = x[..., k0:k0 + 32], W[..., k0:k0 + 32, :]
        part = np.matmul(xs, Ws)
        T = np.matmul(np.abs(xs), np.abs(Ws))
        if acc is None:
            acc = np.zeros(part.shape) if c0 is None else np.broadcast_to(np.asarray(c0, np.float64), part.shape)
        v = acc + part
        T = T + np.abs(acc)
        r = round_f16(v)
        r16 = r.astype(np.float16)
        up = np.nextafter(r16, np.float16(np.inf)).astype(np.float64)
        dn = np.nextafter(r16, np.float16(-np.inf)).astype(np.float64)
        with np.errstate(invalid="ignore"):
            dist = np.minimum(v - (r + dn) / 2, (r + up) / 2 - v)
        fin = np.isfinite(dist)
        f20 = fin & (dist <= np.ldexp(T, -20))
        f23 = fin & (dist <= np.ldexp(T, -23))
        n = v.size
        # tiles: 16 rows (tokens) x 16 outputs
        fl = f20.reshape(-1, f20.shape[-2], f20.shape[-1])
        R_, N_ = fl.shape[-2], fl.shape[-1]
        pr, pn = (-R_) % 16, (-N_) % 16
        fl = np.pad(fl, ((0, 0), (0, pr), (0, pn)))
        tiles = fl.reshape(fl.shape[0], (R_ + pr) // 16, 16, (N_ + pn) // 16, 16).any(axis=(2, 4))
        CF[kind] += [n, f20.sum(), f23.sum(), tiles.size, tiles.sum()]
        acc = r


def make_mm(orig):
    def mm(x, W, c0=None):
        f = sys._getframe(1)
        fn = f.f_code.co_name
        name = None
        if fn == "window_block":                  # call order: per head q, v, O; then Wo, W1, W2
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
            cf_stats(name if name not in ("q", "v") else "qv", x, W, c0)
            xc = np.asarray(q8(x), np.uint8)
            Wc = np.asarray(q8(W), np.uint8)
            if name == "O":                       # P (bounded) x v (split operand; int class only in Z)
                Xw = xc.reshape(-1, 16, xc.shape[-1])
                Vw = np.repeat(Wc, 4, axis=0)
                unit_stats(CUR["stage"] + ":O", Xw, Vw, 0, True, both=True)
            elif name in KEYS:
                T, iok = classT(CUR["W"], KEYS[name])
                if name == "Wm":
                    T = 0
                unit_stats(CUR["stage"] + ":" + name, waves(xc), Wc.reshape(-1, Wc.shape[-1]) if Wc.ndim == 2 else Wc,
                           T, iok)
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
        ns = argparse.Namespace(synthetic=False, b2=False, model=SQ.DEFAULT_MODEL if a.model == "b2" else
                                os.environ.get("WG_RR_MODEL_V0", ""), map=SQ.DEFAULT_MAP)
        m = SQ.load_model(ns)
    d = np.load(a.dump)
    RB.mm = make_mm(RB.mm)
    RS.mm = make_mm(RS.mm)
    origwb = RB.window_block

    def wb(x, xr, W, ra, mm_, wc, Z_override=None, trace=False):
        CUR["blk_i"], CUR["H"] = 0, np.asarray(W["Wq"]).shape[0]
        y2, info = origwb(x, xr, W, ra, mm_, wc, Z_override=Z_override, trace=True)
        q = info["q"]                                    # [Nw, H, 64, 32] codes
        vv = np.asarray(info["v"], np.uint8)             # [Nw, H, 64, 32]
        for half in range(2):
            blk = vv[:, :, 32 * half:32 * half + 32, :]
            vm = MAG[blk].max(axis=(2, 3))
            em = LBE[blk].min(axis=(2, 3)) + 10
            with np.errstate(invalid="ignore", over="ignore"):
                ok = (vm * np.exp2(10 - em) < 1024 * 32 / 1.1) | (vm == 0)
            OG[:] += [ok.size, ok.sum()]
        for h in range(q.shape[1]):
            qq = np.asarray(q[:, h], np.uint8)
            Xw = qq.reshape(-1, 16, 32)
            Kw = np.repeat(np.transpose(qq, (0, 2, 1)), 4, axis=0)   # [nwave, 32, 64]: the window's keys
            unit_stats(CUR["stage"] + ":S", Xw, Kw, 0, True, both=True)
            s2_stats(qq)
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
    tot = np.zeros(8)
    by_kind = collections.defaultdict(lambda: np.zeros(8))
    hdr = f"{'site':14s} {'units':>7s} {'zcur':>6s} {'zstep':>6s} {'grid':>6s} {'tile':>6s} {'ideal':>6s} {'gridE':>6s} {'gridEG':>6s}"
    print(hdr)
    for k, v in acc.items():
        print(f"{k:14s} {int(v[0]):7d} " + " ".join(f"{x / v[0]:6.3f}" for x in v[1:]))
        tot += v
        by_kind[k.split(":")[1]] += v
    print("--- by site kind")
    for k, v in by_kind.items():
        print(f"{k:14s} {int(v[0]):7d} " + " ".join(f"{x / v[0]:6.3f}" for x in v[1:]))
    print("certify-and-fix (kind: outputs, flagged 2^-20, flagged 2^-23, tiles 16x16, tiles flagged 2^-20)")
    for k, v in CF.items():
        print(f"  {k:6s} {int(v[0]):9d} {v[1] / v[0]:.4f} {v[2] / v[0]:.4f} {int(v[3]):7d} {v[4] / v[3]:.3f}")
    ct = sum(CF.values())
    print(f"  {'all':6s} {int(ct[0]):9d} {ct[1] / ct[0]:.4f} {ct[2] / ct[0]:.4f} {int(ct[3]):7d} {ct[4] / ct[3]:.3f}")
    print("O gate (v range, sum P <= 1.1): blocks", int(OG[0]), "pass", OG[1] / OG[0])
    print("S3 (two-chain scores, adaptive split grid): tiles", int(S3[0]), "pass", S3[1] / S3[0])
    print("S2 (two-chain scores): tiles", int(S2[0]), "k unsplit", S2[1] / S2[0], "either", S2[2] / S2[0])
    print(f"{'all':14s} {int(tot[0]):7d} " + " ".join(f"{x / tot[0]:6.3f}" for x in tot[1:]))


if __name__ == "__main__":
    main()
