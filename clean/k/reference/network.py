# SPDX-License-Identifier: MIT
"""NumPy reference of model K (spec 41 draft 3, wiring and offsets of spec 44 draft 1): task K4's oracle.

All stored values are FP16 (float16 arrays). Matrix products (and their bias) are summed exactly (float64) and rounded
to FP16 once; the GPU kernels accumulate in FP32 (spec 41 allows any order; spec 49 sets the tolerance relative to
the output RMS). Elementwise steps are FP16 operations in the order written in spec 41.

Spec 41 draft 2 settles the intermediate roundings (ANSWERS.md, 2026-10-08):
- normalization: s = f16(sum_c f16(x_c^2)) (exact sum here), r = f16(1 / sqrt(s)); s = +inf gives r = 0 and n = 0
  (as the original); s = 0 gives n = 0 (a deliberate deviation: the original produces NaN there);
- softmax weights: g = f16(u f16(1 - f16(kappa f16(u u)))); w = 2^g and the sum over the 64 keys are not rounded
  (float64 here, FP32 on the GPU; spec 41 allows it), A = f16(w (1 / sum));
- every bias is the start of its product's sum (one rounding).
Spec 41 draft 3: the block input of a decoder stage at a window position outside the map is built from the coarser map
mirrored at the coarser level and the skip map mirrored at this level (decoder_windows).

Parameter tensors: clean/k/reference/params.py (canonical names and shapes, and the converter's of spec 48).
"""
import numpy as np

from clean.k.reference.act import act
from clean.k.reference.f16 import f16
from clean.k.reference.layout import (EMBED_C, HEAD_W, INPUT_C, OUTPUT_C, STAGE, STAGES, SYNTHETIC_OFFSETS,
                                      border_index, level_size, stage_offsets, window_grid, window_partition,
                                      window_reverse)
from clean.k.reference.linear import linear, matmul_f16, merge
from clean.k.reference.params import normalize

SOFTMAX_C = np.float16(21.640625)
SOFTMAX_KAPPA = np.float16(0.0007119178771972656)
assert float(SOFTMAX_C) == 21.640625 and float(SOFTMAX_KAPPA) == 0.0007119178771972656
ONE = np.float16(1.0)


def sum_squares(x):
    """s = f16(sum_c f16(x_c^2)) of tokens x [..., C] (exact sum of the rounded squares)."""
    x = np.asarray(x, dtype=np.float16)
    with np.errstate(over="ignore"):
        sq = (x * x).astype(np.float16).astype(np.float64)
    return f16(np.sum(sq, axis=-1, keepdims=True))


def norm(x, g1):
    """Spec 41 step 1 on tokens x [..., C] -> n [..., C] (float16)."""
    x = np.asarray(x, dtype=np.float16)
    s = sum_squares(x).astype(np.float64)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        r = np.where(s == 0, 0.0, f16(1.0 / np.sqrt(s)).astype(np.float64)).astype(np.float16)  # s = inf: r = 0
        rg = (r * np.asarray(g1, dtype=np.float16)).astype(np.float16)
        return (x * rg).astype(np.float16)


def softmax_weights(S):
    """Spec 41 softmax on FP16 scores S [..., 64] (last axis = keys) -> A (float16)."""
    S = np.asarray(S, dtype=np.float16)
    with np.errstate(over="ignore", invalid="ignore"):
        u = np.minimum(np.maximum(S, -SOFTMAX_C), SOFTMAX_C)
        u2 = (u * u).astype(np.float16)
        k = (SOFTMAX_KAPPA * u2).astype(np.float16)
        o = (ONE - k).astype(np.float16)
        g = (u * o).astype(np.float16)
        w = np.exp2(g.astype(np.float64))
        return f16(w * (1.0 / np.sum(w, axis=-1, keepdims=True)))


def mixing(n, p, mix, heads):
    """Token mixing on windows n [nw, 64, C] -> z [nw, 64, 32 heads] (head-major)."""
    zs = []
    for h in range(heads):
        v = matmul_f16(n, p["Wv"][h])  # [nw, 64, 32]
        if mix == "fixed":
            A = np.asarray(p["M"][h], dtype=np.float16)[None]
        else:
            q = matmul_f16(n, p["Wq"][h]).astype(np.float64)
            k = matmul_f16(n, p["Wk"][h]).astype(np.float64)
            S = f16(np.einsum("wtc,wsc->wts", q, k) + np.asarray(p["B"][h], dtype=np.float16).astype(np.float64))
            A = softmax_weights(S)
        zs.append(f16(np.einsum("wts,wsc->wtc", np.broadcast_to(A, (n.shape[0], 64, 64)).astype(np.float64),
                                v.astype(np.float64))))
    return np.concatenate(zs, axis=-1)


def block_windows(x, p, mix, heads):
    """One block on windows x [nw, 64, C] (all members, mirrored ones included) -> x2 [nw, 64, C]."""
    x = np.asarray(x, dtype=np.float16)
    n = norm(x, p["g1"])
    z = mixing(n, p, mix, heads)
    x1 = linear(z, p["Wo"], p["bo"], res=x)
    a = (x1 * np.asarray(p["g2"], dtype=np.float16)).astype(np.float16)
    h = linear(a, p["W1"], p["b1"], use_act=True)
    return linear(h, p["W2"], p["b2"], res=x1)


def block(x, p, mix, heads, offset):
    """One block on a map x [H, W, C] with window offset (ox, oy) and the mirror border rule of spec 41."""
    x = np.asarray(x, dtype=np.float16)
    H, W, _ = x.shape
    ox, oy = offset
    win = window_partition(x, ox, oy)
    return window_reverse(block_windows(win, p, mix, heads), W, H, ox, oy)


def decoder_windows(y, skip, Wx, bx, offset):
    """Spec 41 draft 3 "2x2 expand and skip": the block input windows [nw, 64, C] of a decoder stage. y: the coarser
    map [hc, wc, C_prev]; skip: S at this level [H, W, C]. Window member (r, c) (inside or outside the map) is
    f16(e(m'(r // 2), m'(c // 2))[k C .. k C + C) + S(m(r), m(c))), k = 2 (r mod 2) + (c mod 2), floor semantics,
    m / m' the mirror at this level's / the coarser level's size."""
    y = np.asarray(y, dtype=np.float16)
    skip = np.asarray(skip, dtype=np.float16)
    hc, wc, _ = y.shape
    H, W, C = skip.shape
    ox, oy = offset
    nwx, nwy = window_grid(W, H, ox, oy)
    e = matmul_f16(y, Wx, bx).reshape(hc, wc, 4, C)
    rows = (np.arange(nwy)[:, None] * 8 - oy + np.arange(8)[None, :]).reshape(-1)  # [nwy * 8]
    cols = (np.arange(nwx)[:, None] * 8 - ox + np.arange(8)[None, :]).reshape(-1)
    rc, cc = border_index(rows // 2, hc), border_index(cols // 2, wc)
    k = 2 * (rows % 2)[:, None] + (cols % 2)[None, :]
    ex = e[rc[:, None], cc[None, :], k]  # [nwy 8, nwx 8, C]
    sk = skip[border_index(rows, H)[:, None], border_index(cols, W)[None, :]]
    x = (ex + sk).astype(np.float16)
    return x.reshape(nwy, 8, nwx, 8, C).transpose(0, 2, 1, 3, 4).reshape(nwy * nwx, 64, C)


def decoder_input(y, skip, Wx, bx):
    """The decoder block input at the real positions (inside the map: the formula without mirrors)."""
    skip = np.asarray(skip, dtype=np.float16)
    H, W, _ = skip.shape
    win = decoder_windows(y, skip, Wx, bx, (0, 0))
    return window_reverse(win, W, H, 0, 0)


def run_stage(st, p, x, offset, skip=None, out_size=None, debug=False):
    """One stage of spec 41 / 44. x: the stage input map (e0: the 16-channel network input; dec: the coarser map).
    skip: S<level> (decoder stages, required; it gives the block level's size; out_size is checked against it).
    p: parameter tensors (canonical or spec 48 form). Returns {"block": block output, "merged": (enc), "head": (d0)};
    debug=True adds "windows": the block input windows [nw, 64, C] (mirrored members included)."""
    p = normalize(st["name"], p)
    x = np.asarray(x, dtype=np.float16)
    if st.get("embed"):
        x = linear(x, p["We"], p["be"], relu=True)
    if st["kind"] == "dec":
        if skip is None:
            raise ValueError(f"{st['name']}: skip map required")
        H, W = np.asarray(skip).shape[:2]
        if out_size is not None and tuple(out_size) != (W, H):
            raise ValueError(f"{st['name']}: out_size {out_size} does not match the skip map {(W, H)}")
        win = decoder_windows(x, skip, p["Wx"], p["bx"], offset)
        y = window_reverse(block_windows(win, p, st["mix"], st["heads"]), W, H, *offset)
    else:
        H, W = x.shape[:2]
        win = window_partition(x, *offset)
        y = window_reverse(block_windows(win, p, st["mix"], st["heads"]), W, H, *offset)
    out = {"block": y}
    if debug:
        out["windows"] = win
    if st["kind"] == "enc":
        out["merged"] = merge(y, p["Wm"], p["bm"])
    if st.get("head"):
        out["head"] = linear(y, p["Wh"], p["bh"])
    return out


def offset_table(params):
    """The window-offset table of a parameter set: params["net"]["window_offsets"] (converted models, spec 48)."""
    net = params.get("net") or {}
    if "window_offsets" not in net:
        raise ValueError("the parameters have no window-offset table (net/window_offsets, spec 48)")
    return net["window_offsets"]


def run_network(x0, params, f=0, debug=False):
    """x0 [Hb, Wb, 16] -> {stage: run_stage outputs}; the head output is outs["d0"]["head"] [Hb, Wb, 40]. The window
    offsets come from params["net"]["window_offsets"] (spec 44 draft 2)."""
    x = np.asarray(x0, dtype=np.float16)
    hb, wb = x.shape[:2]
    off = stage_offsets(f, offset_table(params))
    skips, outs = {}, {}
    for st in STAGES:
        name = st["name"]
        size = level_size(wb, hb, st["level"])
        o = run_stage(st, params[name], x, off[name], skips.get(st["skip"]) if st["kind"] == "dec" else None, size,
                      debug)
        outs[name] = o
        if st["kind"] == "enc":
            skips[st["skip"]] = o["block"]
            x = o["merged"]
        else:
            x = o["block"]
    return outs


# ---- random parameters and inputs (tests and the timing harness; never real data) ---------------------------------

def random_stage_params(st, rng=None):
    """Random FP16 parameters of one stage, scaled so that values stay O(1) through the network."""
    rng = rng if rng is not None else np.random.default_rng(0)
    C, D, H, HID = st["C"], st["D"], st["heads"], st["HID"]

    def mat(*shape, fan=None):
        fan = fan or shape[-1]
        return f16(rng.standard_normal(shape) / np.sqrt(fan))

    def vec(n, s=0.1):
        return f16(rng.standard_normal(n) * s)

    p = {"g1": f16(np.sqrt(C) * (1.0 + 0.1 * rng.standard_normal(C))), "g2": f16(1.0 + 0.1 *
                                                                                      rng.standard_normal(C)),
         "Wv": mat(H, HEAD_W, C), "Wo": mat(C, D), "bo": vec(C), "W1": mat(HID, C), "b1": vec(HID),
         "W2": mat(C, HID), "b2": vec(C)}
    if st["mix"] == "fixed":
        p["M"] = f16(rng.standard_normal((H, 64, 64)) / 8.0)
    else:
        p["Wq"] = f16(mat(H, HEAD_W, C) * 1.5)
        p["Wk"] = f16(mat(H, HEAD_W, C) * 1.5)
        p["B"] = f16(rng.standard_normal((H, 64, 64)) * 0.5)
    if st.get("embed"):
        p["We"], p["be"] = mat(EMBED_C, INPUT_C), vec(EMBED_C, 0.3)
    if st["kind"] == "enc":
        p["Wm"], p["bm"] = mat(st["C_next"], 4 * C), vec(st["C_next"])
    if st["kind"] == "dec":
        p["Wx"], p["bx"] = mat(4 * C, st["C_prev"]), vec(4 * C)
    if st.get("head"):
        p["Wh"], p["bh"] = mat(OUTPUT_C, C), vec(OUTPUT_C)
    return p


def random_network_params(seed=0):
    """Random parameters of all eleven stages, plus the synthetic window-offset table (layout.SYNTHETIC_OFFSETS)."""
    rng = np.random.default_rng(seed)
    p = {st["name"]: random_stage_params(st, rng) for st in STAGES}
    p["net"] = {"window_offsets": np.array(SYNTHETIC_OFFSETS, dtype=np.int32)}
    return p


def random_input(wb, hb, seed=1):
    return f16(np.random.default_rng(seed).standard_normal((hb, wb, INPUT_C)))
