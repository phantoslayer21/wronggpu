# SPDX-License-Identifier: MIT
"""Model L NumPy oracle (issue #89): float64 arithmetic with only the prescribed roundings (q8, f16), as Model M's
oracle in clean/reference. Model L's structure comes from clean/l/reference/table.py; nothing here hard-codes it.

    enc = encode(x, params, parity)                    # x: Model M's enc0 input dict (clean/reference/encoder.py)
    out = network(x_enc, x_dec, params, parity)        # encoder -> enc1 ... dec1 (Model M's stages) -> decoder
    dec = decode(x, params, parity)                    # x: Model M's dec0 input dict (features = dec1's output)

params: {stage: tensors} with the converter map's stages (enc0, enc0_block, enc1 ... dec1, dec0_block, dec0, post).

The Swin stages shared with Model M call clean/reference/swin.py, so for them this oracle is Model M's oracle.
"""
import math

import numpy as np

from clean.reference import decoder as _dec
from clean.reference import encoder as _enc
from clean.reference import swin as _swin
from clean.reference.fp8 import f16, q8

from . import table as _table

_erf = np.frompyfunc(math.erf, 1, 1)


def _round(x, mode):
    if mode == "q8":
        return q8(x)
    if mode == "f16":
        return f16(x)
    if mode in (None, "none"):
        return np.asarray(x, dtype=np.float64)
    raise ValueError(f"unknown rounding {mode!r}")


# ---- Swin block with explicit constants -----------------------------------------------------------------------------

def gelu(x, kind):
    if kind == "exact":
        return 0.5 * x * (1.0 + _erf(x / math.sqrt(2.0)).astype(np.float64))
    return _swin.GELU_FUNCTIONS[kind](x)


def rmsnorm(v, gamma, C, eps_num):
    eps = eps_num / C
    return gamma * v / np.sqrt(np.mean(v * v, axis=-1, keepdims=True) + eps)


def soft_cap(S, cap):
    Sc = np.clip(S, -cap, cap)
    return Sc - Sc ** 3 / (3.0 * cap * cap)


def _block_tokens(xw, p, heads, k):
    """One block on gathered windows xw [..., T, C] (T = window^2), constants k. k["form"] "m" (default): Model M's
    step order in float64 with Model M's roundings; "q7": the FP16 form of _block_tokens_q7."""
    if k.get("form", "m") == "q7":
        return _block_tokens_q7(xw, p, heads, k)
    C = xw.shape[-1]
    T = xw.shape[-2]
    lead = xw.shape[:-2]
    a = q8(rmsnorm(xw, p["gamma1"], C, k["eps_num"]))
    u = q8(np.einsum("hdc,...tc->...htd", p["Wqk"], a))
    v = f16(np.einsum("hdc,...tc->...htd", p["Wv"], a))
    S = np.einsum("...htd,...hsd->...hts", u, u) + p["B"]
    z = k["tau"] * soft_cap(S, k["cap"])
    z = z - z.max(axis=-1, keepdims=True)
    A = np.exp(z)
    A = f16(A / A.sum(axis=-1, keepdims=True))
    o = np.einsum("...hts,...hsd->...htd", A, v)
    o = np.moveaxis(o, -3, -2).reshape(lead + (T, heads * k["head_dim"]))
    x1 = xw + q8(o) @ p["Wp"].T + p["bp"]
    m = q8(rmsnorm(x1, p["gamma2"], C, k["eps_num"]))
    h = q8(gelu(m @ p["W1"].T + p["b1"], k["gelu"]))
    return x1 + h @ p["W2"].T + p["b2"]


# ---- the block in its FP16 form ("q7") ------------------------------------------------------------------------------
#
# Every FP16 step rounds after each single operation (f16), FMA16 rounds once. An FP8 matrix op takes up to 32 FP8 x FP8
# products and an FP16 start value and gives an FP16 result; a longer sum is a chain of such ops over 32-wide chunks in
# ascending order, each op's result the next one's start. What happens inside one op is not settled; two models
# (MM8_MODELS, chosen by the table's "mm8" entry, named in every record):
# - "exact" (the default): the exact sum of the products and the start value, rounded once to FP16 (exact in float64:
#   the terms are multiples of 2^-24 below 2^23);
# - "trunc13": two blocks of 16 products, the start value in block 1; in each block every term is truncated toward
#   zero to 13 fractional bits below the largest term's exponent, the truncated terms summed exactly and rounded to
#   FP16; block 1's result is block 2's start value.
# The attention product P V (FP16 operands, 16 terms) is summed in float64 and rounded once (its accumulation precision
# does not move an FP16 result).

MM8_MODELS = ("exact", "trunc13")


def _trunc13_block(start, prods):
    """start [...], prods [..., k] (exact values): every term truncated toward zero to a multiple of 2^(E - 13), E the
    largest term's exponent, then the exact sum rounded to FP16. Non-finite terms: the plain sum, rounded."""
    terms = np.concatenate([np.asarray(start, np.float64)[..., None], prods], axis=-1)
    a = np.abs(terms)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        _, e = np.frexp(np.where(np.isfinite(a), a, 0.0))
        e = np.where(a > 0, e - 1, np.iinfo(np.int32).min // 2)
        emax = e.max(axis=-1, keepdims=True)
        quantum = np.ldexp(1.0, np.maximum(emax - 13, -1074))
        tt = np.trunc(terms / quantum) * quantum
        out = f16(tt.sum(axis=-1))
        bad = ~np.all(np.isfinite(terms), axis=-1)
        if bad.any():
            out = np.where(bad, f16(terms.sum(axis=-1)), out)
    return out


def _op8(start, a8, W8, model):
    """One FP8 matrix op: start [..., n] (FP16), a8 [..., k] (k <= 32), W8 [n, k] -> FP16 [..., n]."""
    start = np.broadcast_to(np.asarray(start, np.float64), a8.shape[:-1] + (W8.shape[0],))
    with np.errstate(over="ignore", invalid="ignore"):
        if model == "exact":
            return f16(start + a8 @ W8.T)
        if model != "trunc13":
            raise ValueError(f"unknown FP8 op model {model!r}")
        out = np.empty(start.shape)
        for r0 in range(0, W8.shape[0], 8):          # rows in groups (the products of 8 rows at a time)
            w = W8[r0:r0 + 8]
            prods = a8[..., None, :] * w                   # [..., rows, k]
            acc = start[..., r0:r0 + 8]
            for b0 in (0, 16):
                acc = _trunc13_block(acc, prods[..., b0:b0 + 16])
            out[..., r0:r0 + 8] = acc
        return out


def mm8(start, a8, W8, model="exact"):
    """The chain of FP8 matrix ops over the 32-wide chunks of K, ascending: a8 [..., K], W8 [n, K], start [..., n] or
    [n] (FP16) -> FP16 [..., n]."""
    a8 = np.asarray(a8, np.float64)
    W8 = np.asarray(W8, np.float64)
    acc = start
    for c0 in range(0, W8.shape[1], 32):
        acc = _op8(acc, a8[..., c0:c0 + 32], W8[:, c0:c0 + 32], model)
    return acc

def _h(bits):
    """The FP16 constant with this bit pattern, as float64."""
    return float(np.array([bits], np.uint16).view(np.float16)[0])


Q7_SCORE_SCALE = _h(0x2466)
Q7_SCORE_CLAMP = _h(0x3873)
Q7_POLY_T = _h(0x3B6B)
Q7_POLY_P = _h(0x3D80)
Q7_NORM_EPS = 2.0 ** -13           # 32 x the per-channel epsilon (the norm sums squares, it does not average)
Q7_GELU_A = _h(0x3698)
Q7_GELU_B = _h(0x2D30)


def _rsqrt32(v):
    """The single-precision reciprocal square root of float32(v) (its last bit does not move the FP16 result)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        s = np.asarray(v, np.float64).astype(np.float32)
        return (np.float32(1.0) / np.sqrt(s)).astype(np.float64)


def _rcp32(v):
    with np.errstate(divide="ignore", invalid="ignore"):
        return (np.float32(1.0) / np.asarray(v, np.float64).astype(np.float32)).astype(np.float64)


def gamma_q7(gamma, C=None):
    """g' = RN16(float32(gamma) x sqrtf(C)), the norm's gain with the sqrt(C) of its sum of squares folded in (C = the
    block width, default len(gamma))."""
    gamma = np.asarray(gamma, np.float64)
    C = gamma.shape[-1] if C is None else C
    g = gamma.astype(np.float32) * np.sqrt(np.float32(C))
    return f16(g.astype(np.float32))


NORM_WIDTHS = (32, 64, 96, 128)


def _group(ps):
    """The fixed sum tree of 1 to 4 FP16 partial sums: p0; p0 + p1; (p0 + p1) + p2; (p0 + p1) + (p2 + p3)."""
    if len(ps) == 1:
        return ps[0]
    if len(ps) == 2:
        return f16(ps[0] + ps[1])
    if len(ps) == 3:
        return f16(f16(ps[0] + ps[1]) + ps[2])
    return f16(f16(ps[0] + ps[1]) + f16(ps[2] + ps[3]))


def norm_q7(x, gp):
    """RMS norm over the last axis (C = 32, 64, 96 or 128 channels) of FP16 values x with gain gp (gamma_q7):
    FP16 squares q; channel tiles of 8, lane t = 0 ... 3 holding channels 8 j + 2 t (even half) and 8 j + 2 t + 1 (odd
    half) of every tile j; per lane and half the pairs (tile j, tile j + 2) of the even tiles (j = 0, 4, 8, 12) form the
    left group and those of the odd tiles (j = 1, 5, 9, 13) the right one, each summed by _group, left + right; across
    lanes ((0 + 1) + (2 + 3)) per half, even + odd, + 2^-13; r = RN16 of the single-precision reciprocal square root;
    then x RN16(r gp). Every + is one FP16 rounding."""
    x = np.asarray(x, np.float64)
    C = x.shape[-1]
    if C not in NORM_WIDTHS:
        raise ValueError(f"norm_q7: {C} channels (32, 64, 96 or 128)")
    tiles = C // 8
    with np.errstate(over="ignore", invalid="ignore"):
        q = f16(x * x)

        def lane(t, odd):
            Q = [q[..., 8 * j + 2 * t + odd] for j in range(tiles)]
            left = _group([f16(Q[j] + Q[j + 2]) for j in range(0, tiles, 4)])
            right = _group([f16(Q[j] + Q[j + 2]) for j in range(1, tiles, 4)])
            return f16(left + right)

        def half(odd):
            e = [lane(t, odd) for t in range(4)]
            return f16(f16(e[0] + e[1]) + f16(e[2] + e[3]))

        s = f16(f16(half(0) + half(1)) + Q7_NORM_EPS)
        r = f16(_rsqrt32(s))
        return f16(x * f16(r[..., None] * gp))


def softmax_q7(s):
    """Attention weights over the last axis (16 keys) from FP16 scores s: u = RN16(s c) clamped to +-c', the FP16
    polynomial p = FMA16(u, FMA16(u, -u, t0), p0) in [1, 2), the weight e = the FP16 number with bits (mantissa bits
    of p) << 5, the FP16 row sum in a fixed tree, w = RN16(rcp(sum)), P = RN16(e w). A NaN score takes the lower clamp
    (min / max return the number)."""
    s = np.asarray(s, np.float64)
    if s.shape[-1] != 16:
        raise ValueError("softmax_q7: 16 keys")
    with np.errstate(over="ignore", invalid="ignore"):
        u = np.fmin(np.fmax(f16(s * Q7_SCORE_SCALE), -Q7_SCORE_CLAMP), Q7_SCORE_CLAMP)
        t = f16(u * -u + Q7_POLY_T)
        p = f16(u * t + Q7_POLY_P)
        pb = p.astype(np.float16).view(np.uint16)
        e = (((pb & 0x03FF) << 5).astype(np.uint16)).view(np.float16).astype(np.float64)

        def pair(lo):   # a_t = RN16(e_{lo+2t} + e_{lo+8+2t}), tree over t
            a = [f16(e[..., lo + 2 * t] + e[..., lo + 8 + 2 * t]) for t in range(4)]
            return f16(f16(a[0] + a[1]) + f16(a[2] + a[3]))

        total = f16(pair(0) + pair(1))
        w = f16(_rcp32(total))
        return f16(e * w[..., None])


def gelu_q7(h):
    """y = clamp(h, -2, 2); g = RN16(h RN16(1/2 + RN16(y RN16(a - RN16(b |y|))))), every step its own rounding."""
    h = np.asarray(h, np.float64)
    with np.errstate(over="ignore", invalid="ignore"):
        y = np.fmin(np.fmax(h, -2.0), 2.0)
        return f16(h * f16(0.5 + f16(y * f16(Q7_GELU_A - f16(Q7_GELU_B * np.abs(y))))))


def _score_op(B, u, model):
    """Scores s[..., h, i, j] = one FP8 op of u_i and u_j (32 terms) started from B[h, i, j]; u [..., h, T, 32]."""
    B = np.asarray(B, np.float64)
    with np.errstate(over="ignore", invalid="ignore"):
        if model == "exact":
            return f16(B + np.einsum("...htd,...hsd->...hts", u, u))
        if model != "trunc13":
            raise ValueError(f"unknown FP8 op model {model!r}")
        prods = u[..., :, None, :] * u[..., None, :, :]           # [..., h, T, T, 32]
        acc = np.broadcast_to(B, prods.shape[:-1])
        for b0 in (0, 16):
            acc = _trunc13_block(acc, prods[..., b0:b0 + 16])
        return acc


def _block_tokens_q7(xw, p, heads, k):
    """The block in its FP16 form on gathered windows xw [..., T, C] (FP16 values; T = 16; C = 32, 64, 96 or 128;
    heads of 32), FP8 ops by the model k["mm8"] (MM8_MODELS, default "exact"):
    norm (norm_q7, gamma1) -> FP8 -> per head u = FP8(Wqk chain from 0), v = Wv chain from 0 (FP16), scores = one FP8 op
    u u started from B, weights softmax_q7, o = RN16(P v) -> FP8; z = the Wp chain (one 32-wide chunk per head, head
    order) started from RN16(bp + x); norm (gamma2) -> FP8; the MLP in 32-wide hidden slices, ascending: h = the W1
    slice's chain started from b1's slice, gelu_q7, FP8, added by W2's matching chunk onto the chain started from
    RN16(b2 + z). Returns the FP16 block output. The matrices are rounded to FP8 here (no change for an FP8 file)."""
    C = xw.shape[-1]
    T = xw.shape[-2]
    if C not in NORM_WIDTHS or T != 16 or k["head_dim"] != 32:
        raise ValueError("_block_tokens_q7: C 32, 64, 96 or 128, 4 x 4 windows, heads of 32")
    model = k.get("mm8", "exact")
    lead = xw.shape[:-2]
    x = np.asarray(xw, np.float64)
    W = {n: q8(p[n]) for n in ("Wqk", "Wv", "Wp", "W1", "W2")}
    with np.errstate(over="ignore", invalid="ignore"):
        a8 = q8(norm_q7(x, gamma_q7(p["gamma1"])))
        u = np.stack([q8(mm8(0.0, a8, W["Wqk"][h], model)) for h in range(heads)], axis=-3)   # [..., h, T, 32]
        v = np.stack([mm8(0.0, a8, W["Wv"][h], model) for h in range(heads)], axis=-3)
        s = _score_op(p["B"], u, model)
        o = f16(np.einsum("...hts,...hsd->...htd", softmax_q7(s), v))
        o8 = q8(np.moveaxis(o, -3, -2).reshape(lead + (T, heads * 32)))
        z = mm8(f16(p["bp"] + x), o8, W["Wp"], model)
        m8 = q8(norm_q7(z, gamma_q7(p["gamma2"])))
        acc = f16(p["b2"] + z)
        for sl in range(W["W1"].shape[0] // 32):
            r = slice(32 * sl, 32 * sl + 32)
            hs = mm8(p["b1"][r], m8, W["W1"][r], model)
            acc = mm8(acc, q8(gelu_q7(hs)), W["W2"][:, r], model)
    return acc


def window_grid(H, W, offset, win):
    ox, oy = offset
    nwx = (W + ox + win - 1) // win
    nwy = (H + oy + win - 1) // win
    cols = np.arange(nwx)[:, None] * win - ox + np.arange(win)[None, :]
    rows = np.arange(nwy)[:, None] * win - oy + np.arange(win)[None, :]
    return rows, cols


def border_index(border, n, halo=0):
    """Grid position -> index into the block's input along one axis of n grid positions.
    "mirror": about the edge pixel, edge not repeated (clean/reference/swin.py); "input": the input holds halo extra
    positions on each side (position -halo is index 0); "coarse": the mirror of the 2 x 2 token grid (n / 2 tokens),
    the sub-pixel kept."""
    if border == "mirror":
        return lambda q: _swin.mirror(q, n)
    if border == "input":
        def f(q):
            i = np.asarray(q) + halo
            if (i < 0).any() or (i >= n + 2 * halo).any():
                raise AssertionError("window reaches beyond the input halo")
            return i
        return f
    if border == "coarse":
        return lambda q: 2 * _swin.mirror(np.asarray(q) // 2, n // 2) + np.asarray(q) % 2
    raise ValueError(f"unknown border {border!r}")


def swin_block(x, p, offset, heads, consts, border="mirror", halo=0, max_tokens=1 << 16, skip=None):
    """One block on a whole map x [H, W, C] (border "input": [H + 2 halo, W + 2 halo, C], the grid in the middle);
    window members outside the grid per border_index. skip [H, W, C] (FP8 values): each member's input is
    RN16(x member + skip member), the skip member taken with its own border, the mirror about the grid size (a
    member outside the grid reads x by the block's border and skip by the mirror). Returns the block output on the
    grid [H, W, C] in float64, not yet rounded (the FP16 form: FP16)."""
    x = np.asarray(x, dtype=np.float64)
    h = halo if border == "input" else 0
    H, W, C = x.shape[0] - 2 * h, x.shape[1] - 2 * h, x.shape[2]
    win = int(consts["window"])
    p = {k: np.asarray(v, dtype=np.float64) for k, v in p.items()}
    rows, cols = window_grid(H, W, offset, win)
    out = np.full((H, W, C), np.nan)
    fr, fc = border_index(border, H, h), border_index(border, W, h)
    if consts.get("mm8") == "trunc13":
        max_tokens = min(max_tokens, 4096)
    rows_per_chunk = max(1, max_tokens // (win * win * cols.shape[0]))
    for l0 in range(0, rows.shape[0], rows_per_chunk):
        r = rows[l0:l0 + rows_per_chunk]
        rr = fr(r)[:, None, :, None]
        cc = fc(cols)[None, :, None, :]
        xw = x[rr, cc]
        if skip is not None:
            xw = f16(xw + skip[_swin.mirror(r, H)[:, None, :, None], _swin.mirror(cols, W)[None, :, None, :]])
        nl, nk = xw.shape[:2]
        x2 = _block_tokens(xw.reshape(nl, nk, win * win, C), p, heads, consts).reshape(nl, nk, win, win, C)
        rv = (r >= 0) & (r < H)
        cv = (cols >= 0) & (cols < W)
        sel = rv[:, None, :, None] & cv[None, :, None, :]
        R = np.broadcast_to(r[:, None, :, None], sel.shape)
        K = np.broadcast_to(cols[None, :, None, :], sel.shape)
        out[R[sel], K[sel]] = x2[sel]
    if np.isnan(out).any():
        raise AssertionError("window grid does not cover the map")
    return out


# ---- encoder --------------------------------------------------------------------------------------------------------

def space_to_depth(v):
    """[fh, fw, n] -> [fh/2, fw/2, 4n], sub-pixel p = dx + 2 dy, channel n * p + i."""
    fh, fw, n = v.shape
    a = v.reshape(fh // 2, 2, fw // 2, 2, n)            # [r, dy, c, dx, i]
    return np.concatenate([a[:, p // 2, :, p % 2, :] for p in range(4)], axis=-1)


def depth_to_space(v):
    """Inverse of space_to_depth: [h, w, 4n] -> [2h, 2w, n]."""
    h, w, c = v.shape
    n = c // 4
    out = np.empty((2 * h, 2 * w, n), dtype=np.asarray(v).dtype)
    for p in range(4):
        out[p // 2::2, p % 2::2, :] = v[..., n * p:n * (p + 1)]
    return out


def project_groups(z, W, b, groups, clamp=_enc.FEATURE_CLAMP, model="exact"):
    """Running sum s = b; for each channel group g: s = FP16(s + W[:, g] z[g]) (one FP8 op per group of at most
    32, by the FP8 op model); then clamp (None: no clamp) and round to FP8. With groups (20, 20, 20, 20) this is
    clean/reference/encoder.py project."""
    W = np.asarray(W, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    s = np.broadcast_to(np.asarray(b, dtype=np.float64), z.shape[:-1] + (W.shape[0],))
    lo = 0
    for g in groups:
        s = _op8(s, z[..., lo:lo + g], W[:, lo:lo + g], model)
        lo += g
    if clamp is not None:
        s = np.clip(s, -clamp, clamp)
    return q8(s)


def _out_round(y, mode):
    if mode == "relu_q8":
        return q8(np.maximum(y, 0.0))
    return _round(y, mode)


def linear(x, W, b, acc=None, out_round="q8", model="exact"):
    """out_round(acc(W x + b)); acc "f16": the FP8 op chain started from b (by the FP8 op model); acc None: exact
    sums."""
    if acc == "f16":
        y = mm8(np.asarray(b, dtype=np.float64), x, W, model)
    elif acc in (None, "none"):
        y = np.asarray(x, dtype=np.float64) @ np.asarray(W, dtype=np.float64).T + np.asarray(b, dtype=np.float64)
    else:
        raise ValueError(f"unknown accumulation {acc!r}")
    return _out_round(y, out_round)


def _run_ops(ops, maps, params, parity, t, default_stage, frame=None):
    """Interprets an op list over named maps; returns the last op's output. FP8 ops by the table's "mm8" model."""
    last = None
    model = t.get("mm8", "exact")
    if model not in MM8_MODELS:
        raise ValueError(f"unknown FP8 op model {model!r}")
    for op in ops:
        k = op["op"]
        st = params[op.get("params", default_stage)]
        if k == "quant":
            v = maps[op["src"]]
            if op["mode"] == "dither":
                h = maps.get("_halo", 0)
                d = dither_table(st["T"], frame["frame"], v.shape[1], v.shape[0], h)
                out = _enc.dither_fp8(v, d[..., None])
            elif op["mode"] == "q8":
                out = q8(v)
            else:
                raise ValueError(f"unknown quant mode {op['mode']!r}")
        elif k == "space_to_depth":
            out = space_to_depth(maps[op["src"]])
        elif k == "depth_to_space":
            out = depth_to_space(maps[op["src"]])
        elif k == "select":
            out = maps[op["src"]][..., op["lo"]:op["hi"]]
        elif k == "linear":
            out = linear(maps[op["src"]], st[op["w"]], st[op["b"]], op.get("acc"), op.get("out_round", "q8"), model)
        elif k == "swin":
            consts = dict(_table.block_consts(t, op["params"]), mm8=model)
            off = _table.block_offset(t, op["offsets"], parity)
            border = op.get("border", "mirror")
            skip = None
            if op.get("skip"):
                skip = maps.get(op["skip"])
                if skip is None:
                    raise ValueError(f"{default_stage}: the skip map {op['skip']!r} is needed (the encoder's block "
                                     "output)")
                skip = np.asarray(skip, dtype=np.float64)
                if skip.shape != maps[op["src"]].shape:
                    raise ValueError(f"swin skip: shapes {skip.shape} and {maps[op['src']].shape}")
            y = swin_block(maps[op["src"]], st, off, op["heads"], consts, border, maps.get("_halo", 0), skip=skip)
            if border == "input":
                maps["_halo"] = 0
            out = _round(y, op.get("out_round", "q8"))
        elif k == "concat":
            parts = [maps[s] for s in op["srcs"]]
            c = sum(p.shape[-1] for p in parts)
            pad = op.get("pad_to", c)
            if pad > c:
                parts.append(np.zeros(parts[0].shape[:-1] + (pad - c,)))
            out = np.concatenate(parts, axis=-1)
        elif k == "project":
            out = project_groups(maps[op["src"]], st[op["w"]], st[op["b"]], op["groups"], op.get("clamp"), model)
        elif k == "dec_project":
            out = dec_project(maps[op["src"]], st[op["w"]], st[op["b"]], op["groups"], op.get("out_round", "q8"),
                              model)
        elif k == "pick":
            out = np.asarray(maps[op["src"]])[..., list(op["values"])]
        else:
            raise ValueError(f"unknown op {k!r}")
        maps[op.get("dst", "out")] = out
        last = out
    return last


def dither_table(T, frame, w, h, halo=0):
    """t = T[n][|y| mod 16][|x| mod 16] at grid positions x = -halo ... w - halo - 1 (y likewise), [h, w]."""
    T = np.asarray(T)
    Y, X = np.mgrid[0:h, 0:w] - halo
    return T[int(frame)][np.abs(Y) % 16, np.abs(X) % 16]


def encoder_halo(t):
    """Extra positions per side the encoder's per-pixel ops run on (a block with border "input"), else 0."""
    hs = [_table.block_halo(t, op["offsets"]) for op in t["enc_ops"]
          if op["op"] == "swin" and op.get("border") == "input"]
    return max(hs, default=0)


def encode(x, params, parity=None, t=_table.TABLE, keep=False):
    """Model L's input encoder. x: Model M's enc0 input dict; params: {stage: tensors} with the stages of
    table.param_shapes(t) ("enc0", "enc0_block"). Returns Model M's enc0 outputs ("features" [fh/2, fw/2, 64]
    FP8-exact, motion surfaces) and, with keep=True, every named map ("maps")."""
    _table.encoder_plan(t)
    if parity is None:
        parity = int(x["frame"]) & 1
    rw, rh = _enc.check_inputs(x)[:2]
    enc = params["enc0"]
    pv_params = {"sigma1": enc.get("sigma1", 0.0), "sigma2": enc.get("sigma2", 0.0)}
    v = _enc.pixel_values(x, pv_params)
    h = encoder_halo(t)
    if h:
        # The 20 values depend only on the mirrored render position (clean/reference/encoder.py reflect), which is
        # a grid position, so the halo copies grid values; the dither is applied later at the unmirrored position.
        # Only for render sizes near the padding (fw + h > 2 rw - 2) would the mirrored position be negative; reflect
        # clamps it to 0 there (table.UNCONFIRMED "U1").
        fh, fw = v.shape[:2]
        ry = _enc.reflect(np.arange(-h, fh + h), rh)
        rx = _enc.reflect(np.arange(-h, fw + h), rw)
        v = v[ry][:, rx]
    maps = {"values": v, "_halo": h}
    feats = _run_ops(t["enc_ops"], maps, params, parity, t, "enc0", frame=x)
    if maps.pop("_halo"):
        raise ValueError("encoder: the per-pixel halo was not consumed by a block")
    m_near, m_far, m_center = _enc.motion_surfaces(x)
    skip = maps.get(t.get("skip_map"))
    res = {"skip": None if skip is None else np.asarray(skip, np.float32),
           "features": feats.astype(np.float32),
           "m_near": f16(m_near).astype(np.float32),
           "m_far": f16(m_far).astype(np.float32),
           "m_center": f16(m_center).astype(np.float32)}
    if keep:
        res["maps"] = maps
    return res


# ---- decoder --------------------------------------------------------------------------------------------------------

def dec_project(Z, W, b, groups=(32, 32), out_round="q8", model="exact"):
    """Z [h, w, 64], W [4, 32, 64], b [4, 32] -> [h, w, 128] (channel 32 s + c): Model M's dec0 projection arithmetic
    (clean/reference/decoder.py project) over all 4 x 32 rows, one FP8 op per group (by the FP8 op model)."""
    Z = np.asarray(Z, dtype=np.float64)
    W = np.asarray(W, dtype=np.float64).reshape(128, -1)
    s = np.broadcast_to(np.asarray(b, dtype=np.float64).reshape(128), Z.shape[:-1] + (128,))
    lo = 0
    for g in groups:
        s = _op8(s, Z[..., lo:lo + g], W[:, lo:lo + g], model)
        lo += g
    return _round(s, out_round)


def coefficients(x, a, S, T):
    """Model M's coefficient and latent functions (clean/reference/decoder.py decode after its projection) on the
    per-pixel values a [rh, rw, 17]."""
    rw, rh, fw, fh, dw, dh = _dec.check_inputs(x)
    a = np.asarray(a, dtype=np.float32)
    A = _dec.sigmoid16(a[..., 0:3])
    B = _dec.softmax3(a[..., 3:6])
    C = _dec.coefficient_c(a[..., 6], a[..., 7], a[..., 8], float(np.asarray(S)), float(np.asarray(T)))
    p = _dec.previous_latent(x, rw, rh, dw, dh)
    lat = _dec.latent_update(a, p)
    cA, cB, cL = _dec.unorm8_code(A), _dec.unorm8_code(B), _dec.snorm8_code(lat)
    return {"A": (cA / 255.0).astype(np.float32), "B": (cB / 255.0).astype(np.float32),
            "C": C.astype(np.float32), "latent": (cL / 127.0).astype(np.float32),
            "codes_A": cA, "codes_B": cB, "codes_L": cL, "p": p}


def tanh32(v):
    """tanh in float64, then one rounding to FP32."""
    return np.tanh(np.asarray(v, dtype=np.float64)).astype(np.float32)


def coefficients_l(a, S, T):
    """Model L's per-pixel functions on a [rh, rw, 13] (table.L_DEC_VALUES order; FP16 block outputs held in FP32):
    A, B and C with Model M's forms (C's gains from inputs clamped to [-20, 20]), latent = tanh(candidate). No gate,
    no previous latent. Same outputs and codes as coefficients()."""
    a = np.asarray(a, dtype=np.float32)
    A = _dec.sigmoid16(a[..., 0:3])
    B = _dec.softmax3(a[..., 3:6])
    g = _dec.sigmoid16(np.clip(a[..., 6:8], np.float32(-20), np.float32(20)))
    sx, sy = np.float32(S) * g[..., 0], np.float32(S) * g[..., 1]
    rho = _dec.rho32(a[..., 8], float(np.asarray(T)))
    C = np.stack([_dec.rn16(sx * sx), _dec.rn16(sy * sy), _dec.rn16((sx * sy) * rho)], axis=-1)
    lat = tanh32(a[..., 9:13])
    cA, cB, cL = _dec.unorm8_code(A), _dec.unorm8_code(B), _dec.snorm8_code(lat)
    return {"A": (cA / 255.0).astype(np.float32), "B": (cB / 255.0).astype(np.float32),
            "C": C.astype(np.float32), "latent": (cL / 127.0).astype(np.float32),
            "codes_A": cA, "codes_B": cB, "codes_L": cL}


def decode(x, params, parity, t=_table.TABLE, keep=False):
    """Model L's coefficient decoder. x: Model M's dec0 input dict (features = dec1's output [fh/2, fw/2, 64]; with
    dec_pixel "l" the motion / latent_prev / jitter / reset entries are not read); params: {stage: tensors} ("dec0",
    "dec0_block")."""
    kind = t["dec_pixel"]
    if kind not in _table.DEC_PIXEL_VALUES:
        raise NotImplementedError(f"decoder per-pixel variant {kind!r}: not known")
    _table.decoder_plan(t)
    if len(t["dec_ops"][-1]["values"]) != _table.DEC_PIXEL_VALUES[kind]:
        raise ValueError(f"dec_pixel {kind!r} needs {_table.DEC_PIXEL_VALUES[kind]} picked values")
    rw, rh = _dec.check_inputs(x)[:2]
    maps = {"features": np.asarray(x["features"], dtype=np.float64),
            "skip": None if x.get("skip") is None else np.asarray(x["skip"], dtype=np.float64)}
    a = _run_ops(t["dec_ops"], maps, params, parity, t, "dec0")
    dec = params["dec0"]
    if kind == "l":
        out = coefficients_l(a[:rh, :rw], dec["S"], dec["T"])
    else:
        out = coefficients(x, a[:rh, :rw], dec["S"], dec["T"])
    out["a"] = a[:rh, :rw].astype(np.float32)
    if keep:
        out["maps"] = maps
    return out


def reconstruct(x, params, t=_table.TABLE, ldr=False):
    """Model L's output reconstruction (clean/l/reference/recon.py) with the post stage's learned scale.
    x: Model M's reconstruction input dict (clean/reference/post.py reconstruct)."""
    from . import recon as _recon
    return _recon.reconstruct(x, params["post"]["s"], t, ldr)


# ---- whole network --------------------------------------------------------------------------------------------------

def network(x_enc, x_dec, params, parity, t=_table.TABLE, keep=False):
    """Encoder -> the ten shared Swin stages (Model M's oracle, L's weights) -> decoder.

    x_enc: Model M's enc0 input dict. x_dec: Model M's dec0 input dict without "features" (motion, latent_prev,
    jitter, jitter_prev, reset, feature_size, recon_size). params: {stage: tensors} for every stage.
    Returns {"enc0": ..., stage outputs ..., "dec0": ...}."""
    res = {"enc0": encode(x_enc, params, parity, t, keep)}
    x = res["enc0"]["features"]
    skips = {}
    for s in _table.SHARED_STAGES:
        cfg = _swin.STAGES[s]
        off = _swin.window_offset(s, parity)
        if cfg["kind"] == "enc":
            o = _swin.run_stage(s, params[s], x, off)
            skips[s] = o["skip"]
            x = o["merged"]
        elif cfg["kind"] == "dec":
            skip = skips["enc2" if s == "dec2" else "enc1"]
            o = _swin.run_stage(s, params[s], x, off, skip)
            x = o["out"]
        else:
            o = _swin.run_stage(s, params[s], x, off)
            x = o["out"]
        res[s] = o
    xd = dict(x_dec)
    xd["features"] = x
    xd["skip"] = res["enc0"]["skip"]
    res["dec0"] = decode(xd, params, parity, t, keep)
    return res


# ---- synthetic parameters (tests only; never model data) ------------------------------------------------------------

def random_block_params(rng, C, heads, mlp, prefix="", weight_scale=1.0):
    s = weight_scale
    p = {
        "gamma1": f16(rng.uniform(0.5, 1.5, C)),
        "gamma2": f16(rng.uniform(0.5, 1.5, C)),
        "Wqk": q8(rng.normal(0, s * 0.6 / math.sqrt(C), (heads, 32, C))),
        "Wv": q8(rng.normal(0, s * 1.0 / math.sqrt(C), (heads, 32, C))),
        "B": f16(rng.normal(0, 0.5, (heads, 16, 16))),
        "Wp": q8(rng.normal(0, s * 0.5 / math.sqrt(32 * heads), (C, 32 * heads))),
        "bp": f16(rng.normal(0, 0.1, C)),
        "W1": q8(rng.normal(0, s * 1.0 / math.sqrt(C), (mlp, C))),
        "b1": f16(rng.normal(0, 0.1, mlp)),
        "W2": q8(rng.normal(0, s * 0.5 / math.sqrt(mlp), (C, mlp))),
        "b2": f16(rng.normal(0, 0.1, C)),
    }
    return {prefix + k: v.astype(np.float32) for k, v in p.items()}


def random_params(rng, t=_table.TABLE):
    """Synthetic parameters for every stage of Model L ({stage: tensors}; shapes from the table and Model M's
    stages). Never model data."""
    p = {}
    for stage, sh in _table.param_shapes(t).items():
        st = {}
        if "Wqk" in sh:
            st.update(random_block_params(rng, sh["gamma1"][0], sh["Wqk"][0], sh["W1"][0]))
        for name, shape in sh.items():
            if name in st:
                continue
            if stage == "enc0" and name == "T":
                st[name] = rng.integers(0, 128, size=shape).astype(np.float32)
            elif name in ("sigma1", "sigma2"):
                st[name] = np.array(rng.uniform(0.5, 1.5), dtype=np.float32)
            elif stage == "dec0" and name == "S":
                st[name] = np.array(rng.uniform(1.0, 4.0), dtype=np.float32)
            elif stage == "dec0" and name == "T":
                st[name] = np.array(rng.uniform(0.2, 1.5), dtype=np.float32)
            elif name == "s":
                st[name] = np.array(rng.uniform(0.5, 2.0), dtype=np.float32)
            elif len(shape) >= 2:
                fan_in = shape[-1]
                scale = {"enc0": 0.6 if name == "W" else 1.0, "dec0": 0.25 * np.sqrt(fan_in)}.get(stage, 1.0)
                st[name] = q8(rng.normal(0, scale / np.sqrt(fan_in), shape)).astype(np.float32)
            else:
                st[name] = f16(rng.normal(0, 0.3 if stage == "dec0" else 0.1, shape)).astype(np.float32)
        p[stage] = st
    for s in _table.SHARED_STAGES:
        p[s] = {k: np.asarray(v, dtype=np.float32) for k, v in _swin.random_params(s, rng).items()}
    return p
