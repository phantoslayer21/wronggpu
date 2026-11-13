# SPDX-License-Identifier: MIT
"""Model L NumPy oracle (issue #89): float64 arithmetic with only the prescribed roundings (q8, f16), as Model M's
oracle in clean/reference. Model L's structure comes from clean/l/reference/table.py; nothing here hard-codes it.

    enc = encode(x, params["enc0"], parity)            # x: Model M's enc0 input dict (clean/reference/encoder.py)
    out = network(x_enc, x_dec, params, parity)        # encoder -> enc1 ... dec1 (Model M's stages) -> decoder
    dec = decode(x, params["dec0"], parity)            # x: Model M's dec0 input dict (features = dec1's output)

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
    """One block on gathered windows xw [..., T, C] (T = window^2), constants k. Same step order as Model M's."""
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


def window_grid(H, W, offset, win):
    ox, oy = offset
    nwx = (W + ox + win - 1) // win
    nwy = (H + oy + win - 1) // win
    cols = np.arange(nwx)[:, None] * win - ox + np.arange(win)[None, :]
    rows = np.arange(nwy)[:, None] * win - oy + np.arange(win)[None, :]
    return rows, cols


def swin_block(x, p, offset, heads, consts, max_tokens=1 << 16):
    """One block on a whole map x [H, W, C] with mirrored borders (as clean/reference/swin.py swin_block); returns the
    block output in float64, not yet rounded."""
    x = np.asarray(x, dtype=np.float64)
    H, W, C = x.shape
    win = int(consts["window"])
    p = {k: np.asarray(v, dtype=np.float64) for k, v in p.items()}
    rows, cols = window_grid(H, W, offset, win)
    out = np.full_like(x, np.nan)
    rows_per_chunk = max(1, max_tokens // (win * win * cols.shape[0]))
    for l0 in range(0, rows.shape[0], rows_per_chunk):
        r = rows[l0:l0 + rows_per_chunk]
        rr = _swin.mirror(r, H)[:, None, :, None]
        cc = _swin.mirror(cols, W)[None, :, None, :]
        xw = x[rr, cc]
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


def _block_params(params, prefix):
    n = len(prefix)
    return {k[n:]: v for k, v in params.items() if k.startswith(prefix)}


# ---- encoder --------------------------------------------------------------------------------------------------------

def space_to_depth(v):
    """[fh, fw, n] -> [fh/2, fw/2, 4n], sub-pixel p = dx + 2 dy, channel n * p + i."""
    fh, fw, n = v.shape
    a = v.reshape(fh // 2, 2, fw // 2, 2, n)            # [r, dy, c, dx, i]
    return np.concatenate([a[:, p // 2, :, p % 2, :] for p in range(4)], axis=-1)


def project_groups(z, W, b, groups, clamp=_enc.FEATURE_CLAMP):
    """Running sum s = b; for each channel group g: s = FP16(s + W[:, g] z[g]) (exact group sums); then clamp and
    round to FP8. With groups (20, 20, 20, 20) this is clean/reference/encoder.py project."""
    W = np.asarray(W, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    s = np.broadcast_to(np.asarray(b, dtype=np.float64), z.shape[:-1] + (W.shape[0],))
    lo = 0
    for g in groups:
        s = f16(s + z[..., lo:lo + g] @ W[:, lo:lo + g].T)
        lo += g
    return q8(np.clip(s, -clamp, clamp))


def encode(x, params, parity=None, t=_table.TABLE, keep=False):
    """Model L's input encoder. x: Model M's enc0 input dict; params: logical names of table.encoder_shapes(t).
    Returns Model M's enc0 outputs ("features" [fh/2, fw/2, 64] FP8-exact, motion surfaces) and, with keep=True, every
    named map ("maps")."""
    _table.encoder_plan(t)
    if parity is None:
        parity = int(x["frame"]) & 1
    rw, rh, fw, fh, dw, dh = _enc.check_inputs(x)
    pv_params = {"sigma1": params.get("sigma1", 0.0), "sigma2": params.get("sigma2", 0.0)}
    maps = {"values": _enc.pixel_values(x, pv_params)}
    feats = None
    for op in t["enc_ops"]:
        k = op["op"]
        if k == "quant":
            v = maps[op["src"]]
            if op["mode"] == "dither":
                d = _enc.dither_table(params["T"], x["frame"], fw, fh)
                out = _enc.dither_fp8(v, d[..., None])
            elif op["mode"] == "q8":
                out = q8(v)
            else:
                raise ValueError(f"unknown quant mode {op['mode']!r}")
        elif k == "space_to_depth":
            out = space_to_depth(maps[op["src"]])
        elif k == "select":
            out = maps[op["src"]][..., op["lo"]:op["hi"]]
        elif k == "linear":
            W = np.asarray(params[op["w"]], dtype=np.float64)
            b = np.asarray(params[op["b"]], dtype=np.float64)
            out = _round(maps[op["src"]] @ W.T + b, op.get("out_round", "q8"))
        elif k == "swin":
            consts = _table.block_consts(t, op.get("name", "enc0." + op["prefix"]))
            off = _table.block_offset(t, op["offsets"], parity)
            y = swin_block(maps[op["src"]], _block_params(params, op["prefix"]), off, op["heads"], consts)
            out = _round(y, op.get("out_round", "q8"))
        elif k == "concat":
            parts = [maps[s] for s in op["srcs"]]
            c = sum(p.shape[-1] for p in parts)
            pad = op.get("pad_to", c)
            if pad > c:
                parts.append(np.zeros(parts[0].shape[:-1] + (pad - c,)))
            out = np.concatenate(parts, axis=-1)
        elif k == "project":
            out = project_groups(maps[op["src"]], params[op["w"]], params[op["b"]], op["groups"])
            feats = out
        maps[op.get("dst", "features")] = out
    m_near, m_far, m_center = _enc.motion_surfaces(x)
    res = {"features": feats.astype(np.float32),
           "m_near": f16(m_near).astype(np.float32),
           "m_far": f16(m_far).astype(np.float32),
           "m_center": f16(m_center).astype(np.float32)}
    if keep:
        res["maps"] = maps
    return res


# ---- decoder --------------------------------------------------------------------------------------------------------

def decoder_front(features, params, parity, t=_table.TABLE):
    """dec1's FP8 map [fh/2, fw/2, 64] -> the 64-channel map Model M's decoder projects (table entry dec_connect)."""
    Z = np.asarray(features, dtype=np.float64)
    segs = t["dec_connect"]
    blk = [s for s in segs if s[0] == "block"]
    if not blk:
        return Z
    if t["dec_block_level"] != "token":
        raise NotImplementedError("decoder block level other than token: pending")
    b = t["dec_block"]
    lo = min(s[1] for s in blk)
    hi = max(s[2] for s in blk)
    if sum(s[2] - s[1] for s in blk) != hi - lo or hi - lo != b["C"]:
        raise ValueError("block segments must be one contiguous range of the block's width")
    consts = _table.block_consts(t, b.get("name", "dec0." + b["prefix"]))
    off = _table.block_offset(t, b["offsets"], parity)
    y = _round(swin_block(Z[..., lo:hi], _block_params(params, b["prefix"]), off, b["heads"], consts),
               b.get("out_round", "q8"))
    out = Z.copy()
    for kind, s0, s1 in segs:
        if kind == "block":
            out[..., s0:s1] = y[..., s0 - lo:s1 - lo]
        elif kind != "pass":
            raise ValueError(f"unknown decoder segment {kind!r}")
    return out


def decode(x, params, parity, t=_table.TABLE):
    """Model L's coefficient decoder: decoder_front, then Model M's decoder (table entry dec_pixel = "m")."""
    if t["dec_pixel"] != "m":
        raise NotImplementedError("decoder per-pixel variant other than Model M's: pending")
    xx = dict(x)
    xx["features"] = decoder_front(x["features"], params, parity, t).astype(np.float32)
    out = _dec.decode(xx, {k: params[k] for k in ("W", "b", "S", "T")})
    out["front"] = xx["features"]
    return out


def reconstruct(*args, **kwargs):
    raise NotImplementedError("Model L output reconstruction (table entry post_variant): pending")


# ---- whole network --------------------------------------------------------------------------------------------------

def network(x_enc, x_dec, params, parity, t=_table.TABLE, keep=False):
    """Encoder -> the ten shared Swin stages (Model M's oracle, L's weights) -> decoder.

    x_enc: Model M's enc0 input dict. x_dec: Model M's dec0 input dict without "features" (motion, latent_prev,
    jitter, jitter_prev, reset, feature_size, recon_size). params: {"enc0": ..., "enc1": ..., ..., "dec0": ...}.
    Returns {"enc0": ..., stage outputs ..., "dec0": ...}."""
    res = {"enc0": encode(x_enc, params["enc0"], parity, t, keep)}
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
    res["dec0"] = decode(xd, params["dec0"], parity, t)
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
    """Synthetic parameters for every stage of Model L (shapes from the table and Model M's stages)."""
    enc = {}
    for name, shape in _table.encoder_shapes(t).items():
        if name == "T":
            enc[name] = rng.integers(0, 128, size=shape).astype(np.float32)
        elif name in ("sigma1", "sigma2"):
            enc[name] = np.array(rng.uniform(0.5, 1.5), dtype=np.float32)
    for op in t["enc_ops"]:
        if op["op"] == "linear":
            enc[op["w"]] = q8(rng.normal(0, 1.0 / math.sqrt(op["n_in"]), (op["n_out"], op["n_in"]))).astype(np.float32)
            enc[op["b"]] = f16(rng.normal(0, 0.1, op["n_out"])).astype(np.float32)
        elif op["op"] == "swin":
            enc.update(random_block_params(rng, op["C"], op["heads"], op["mlp"], op["prefix"]))
        elif op["op"] == "project":
            n = sum(op["groups"])
            enc[op["w"]] = q8(rng.normal(0, 0.6 / math.sqrt(n), (op["n_out"], n))).astype(np.float32)
            enc[op["b"]] = f16(rng.normal(0, 0.3, op["n_out"])).astype(np.float32)
    dec = _dec.random_params(rng)
    if any(s[0] == "block" for s in t["dec_connect"]):
        b = t["dec_block"]
        dec.update(random_block_params(rng, b["C"], b["heads"], b["mlp"], b["prefix"]))
    p = {"enc0": enc, "dec0": dec}
    for s in _table.SHARED_STAGES:
        p[s] = {k: np.asarray(v, dtype=np.float32) for k, v in _swin.random_params(s, rng).items()}
    return p
