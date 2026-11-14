# SPDX-License-Identifier: MIT
"""Frame generation (issue #96): net1 (the 17-blob U-Net) end to end, as a CPU reference with the delivered arithmetic.

Written for this project from the spec side's part 3 (net1 complete). Everything here is fixed; nothing is an argument:
- the 2 x 2 fold of the 10 per-pixel values into cells: cell channel c = 4 v + q with q = 2 dy + dx (value v of the
  pixel (2X + dx, 2Y + dy)); channels 40..47 are zero; pixels outside the input area are zero;
- 32 steps: 17 convolutions, five 2 x 2 max-pools, five bilinear upscales, five float16 adds (NET1_WIRING, as data);
  level sizes halve with floor; skips are the encoder outputs after the activation, added to the upscaled tensor;
- convolution: stride 1, zero border, cross-correlation over the natural weights; float16 accumulator started at the
  bias; chunks of 16 consecutive k (k = (3 ky + kx) * in + c, or c for 1 x 1) each summed exactly and added with one
  float16 rounding (fp16acc.acc_dot16), in the chunk order of the layer (CHUNK_ORDERS);
- activation: leaky ReLU y = max(x, h(s * x)) with the delivered slope s = 0.0099945068359375, after every
  convolution except the last (blob 17);
- max-pool: window 2 x 2, stride 2, m <- (x > m) * x + (x <= m) * m in float16 from the first tap;
- upscale: bilinear, align corners, from (wi, hi) to (wo, ho) with single-precision positions and float16 weights and
  products (upscale_to);
- add: h(a + b);
- output: the 16 channels per cell unfold to 4 values per pixel, channel c = 4 t + q.

Not part of net1 itself and still open: what the 10 input values are made of (images A-D, the warps) and the size of
the input grid relative to the output. The ratio (wi - 1) / (wo - 1) of the upscale is an approximate division in the
original; here it is the correctly rounded single-precision quotient (the two can differ in the last bit).
"""
import numpy as np

from clean.fg import layout as L
from clean.fg.reference.fp16acc import acc_dot16, h

# The slope as delivered: 0.0099945068359375 (float16 bits 0x211E). Note: nearest-even rounding of 0.01 gives the
# next value up (0x211F, 0.01000213623046875); the delivered constant is the one below it, so it is used as given.
LEAKY_SLOPE = 0.0099945068359375
VALUES = 10                               # per-pixel input values v0..v9
CELL = 48                                 # cell channels: 40 used, 8 zero

# chunk orders: "ascending" (1 x 1: channel blocks in order); "block_kx_ky": channel block outer, then the taps with
# kx outer and ky inner; "kx_block_ky": kx outer, then channel block, then ky
CHUNK_ORDERS = ("ascending", "block_kx_ky", "kx_block_ky")
LAYER_CHUNKS = {"net1_01": "block_kx_ky", "net1_03": "block_kx_ky", "net1_04": "block_kx_ky",
                "net1_05": "kx_block_ky"}


def _conv(layer, src, out, act="leaky"):
    return {"op": "conv", "layer": layer, "in": src, "out": out, "act": act,
            "chunks": LAYER_CHUNKS.get(layer, "ascending")}


def _wiring():
    w = [_conv("net1_01", "cells", "a0"),
         {"op": "pool", "in": "a0", "out": "p1"}, _conv("net1_02", "p1", "a1"),
         {"op": "pool", "in": "a1", "out": "p2"}, _conv("net1_03", "p2", "a2"),
         {"op": "pool", "in": "a2", "out": "p3"}, _conv("net1_04", "p3", "a3"),
         {"op": "pool", "in": "a3", "out": "p4"}, _conv("net1_05", "p4", "a4"),
         {"op": "pool", "in": "a4", "out": "p5"}, _conv("net1_06", "p5", "b5"), _conv("net1_07", "b5", "d5")]
    prev = "d5"
    for lvl, (l1, l2) in zip((4, 3, 2, 1, 0), (("net1_08", "net1_09"), ("net1_10", "net1_11"), ("net1_12", "net1_13"),
                                              ("net1_14", "net1_15"), ("net1_16", "net1_17"))):
        w += [{"op": "upscale", "in": prev, "like": "a%d" % lvl, "out": "u%d" % lvl},
              {"op": "add", "in": ["u%d" % lvl, "a%d" % lvl], "out": "s%d" % lvl},
              _conv(l1, "s%d" % lvl, "e%d" % lvl),
              _conv(l2, "e%d" % lvl, "d%d" % lvl if lvl else "out", act="leaky" if lvl else "none")]
        prev = "d%d" % lvl
    return w


NET1_WIRING = _wiring()     # 32 steps; the network output is buffer "out" (16 channels per cell)


# ---- the operations -----------------------------------------------------------------------------------------------

def leaky(x):
    """y = max(x, h(s * x)) with s = LEAKY_SLOPE: one float16 multiply, then the max."""
    x = np.asarray(x, dtype=np.float64)
    return np.maximum(x, h(LEAKY_SLOPE * x))


def _chunks(order, k, n_blocks):
    if order == "ascending":
        if k != 1:
            raise ValueError("ascending is the 1 x 1 order")
        return [(0, 0, cb) for cb in range(n_blocks)]
    if order == "block_kx_ky":
        return [(ky, kx, cb) for cb in range(n_blocks) for kx in range(k) for ky in range(k)]
    if order == "kx_block_ky":
        return [(ky, kx, cb) for kx in range(k) for cb in range(n_blocks) for ky in range(k)]
    raise ValueError("unknown chunk order %r" % (order,))


def conv16(x, w, b, *, chunks, act):
    """One net1 convolution: x [cin, H, W], natural canonical weights w [out, k, k, cin], bias b [out] (all float16
    values). Zero border, stride 1; float16 accumulator from the bias, 16-term chunks in `chunks` order; act "leaky" or
    "none"."""
    x = h(x)
    w = h(w)
    cout, k, _, cin = w.shape
    if cin % 16 or x.shape[0] != cin:
        raise ValueError("net1 convolutions take a multiple of 16 input channels matching the weights")
    H, W = x.shape[1:]
    p = k // 2
    xp = np.pad(x, ((0, 0), (p, p), (p, p))) if p else x
    acc = np.broadcast_to(h(b)[:, None, None], (cout, H, W)).copy()
    for ky, kx, cb in _chunks(chunks, k, cin // 16):
        a = w[:, ky, kx, 16 * cb:16 * cb + 16][:, :, None, None]           # out, 16, 1, 1
        xs = xp[16 * cb:16 * cb + 16, ky:ky + H, kx:kx + W]                # 16, H, W
        acc = acc_dot16(acc, a, xs[None], axis=1)
    if act == "leaky":
        return leaky(acc)
    if act == "none":
        return acc
    raise ValueError("act must be 'leaky' or 'none'")


def maxpool2(x):
    """2 x 2 max-pool, stride 2, floor sizes: m <- (x > m) * x + (x <= m) * m in float16, from the first tap (the
    result does not depend on the tap order, also for inf and NaN)."""
    x = h(x)
    c, H, W = x.shape
    Ho, Wo = H // 2, W // 2
    t = [x[:, dy:2 * Ho:2, dx:2 * Wo:2] for dy in (0, 1) for dx in (0, 1)]
    m = t[0]
    with np.errstate(invalid="ignore"):
        for v in t[1:]:
            m = h((v > m) * v + (v <= m) * m)
    return m


def _axis(n_in, n_out):
    """Per output index: (i0, i1, w0, w1) of the align-corners rule along one axis."""
    f32 = np.float32
    r = f32(n_in - 1) / f32(n_out - 1) if n_out > 1 else f32(0)
    i0s, i1s, w0s, w1s = [], [], [], []
    for o in range(n_out):
        s = f32(r * f32(o))
        i0 = int(np.trunc(s))
        i1 = int(np.trunc(f32(s + f32(1))))
        if i1 >= n_in:
            i0 = i1 = n_in - 1
        w1 = float(h(f32(s - f32(i0))))
        w0 = float(h(f32(f32(f32(i0) - s) + f32(1))))
        i0s.append(i0); i1s.append(i1); w0s.append(w0); w1s.append(w1)
    return np.array(i0s), np.array(i1s), np.array(w0s), np.array(w1s)


def upscale_to(p, wo, ho):
    """Bilinear upscale, align corners, from p [c, hi, wi] to [c, ho, wo]: single-precision positions, float16 weights,
    every product and sum rounded to float16: top = h(h(wx0 P00) + h(wx1 P01)), out = h(h(wy0 top) + h(wy1 bottom))."""
    p = h(p)
    _, hi, wi = p.shape
    x0, x1, wx0, wx1 = _axis(wi, wo)
    y0, y1, wy0, wy1 = _axis(hi, ho)

    def row(r):
        return h(h(wx0 * r[:, :, x0]) + h(wx1 * r[:, :, x1]))
    top = row(p[:, y0, :])
    bot = row(p[:, y1, :])
    return h(h(wy0[None, :, None] * top) + h(wy1[None, :, None] * bot))


def add16(a, b):
    return h(h(a) + h(b))


# ---- fold, wiring, network ----------------------------------------------------------------------------------------

def fold(values):
    """values [10, G1h, G1w] -> cells [48, ceil(G1h / 2), ceil(G1w / 2)]: channel 4 v + 2 dy + dx; pixels outside the
    input area and channels 40..47 are zero."""
    v = h(values)
    if v.shape[0] != VALUES:
        raise ValueError("net1 takes %d values per pixel" % VALUES)
    _, gh, gw = v.shape
    H0, W0 = -(-gh // 2), -(-gw // 2)
    padded = np.zeros((VALUES, 2 * H0, 2 * W0))
    padded[:, :gh, :gw] = v
    cells = np.zeros((CELL, H0, W0))
    for dy in (0, 1):
        for dx in (0, 1):
            cells[2 * dy + dx:4 * VALUES:4] = padded[:, dy::2, dx::2]
    return cells


def unfold(out, gh, gw):
    """out [16, H0, W0] -> [4, gh, gw]: value t of pixel (2X + dx, 2Y + dy) is channel 4 t + 2 dy + dx."""
    _, H0, W0 = out.shape
    px = np.zeros((4, 2 * H0, 2 * W0))
    for dy in (0, 1):
        for dx in (0, 1):
            px[:, dy::2, dx::2] = out[2 * dy + dx::4][:4]
    return px[:, :gh, :gw]


def run(cells, params, wiring=NET1_WIRING):
    """Runs net1's steps on cells [48, H0, W0]; params: {layer: (canonical weights [out, k, k, in], bias)} (forward.py's
    parameters). Returns every buffer."""
    buf = {"cells": h(cells)}
    for s in wiring:
        kind = s["op"]
        if kind == "conv":
            w, b = params[s["layer"]]
            y = conv16(buf[s["in"]], w, b, chunks=s["chunks"], act=s["act"])
        elif kind == "pool":
            y = maxpool2(buf[s["in"]])
        elif kind == "upscale":
            ref = buf[s["like"]]
            y = upscale_to(buf[s["in"]], ref.shape[2], ref.shape[1])
        elif kind == "add":
            y = add16(buf[s["in"][0]], buf[s["in"][1]])
        else:
            raise ValueError("unknown step %r" % kind)
        buf[s["out"]] = y
    return buf


def min_input_size():
    """The smallest input grid side (pixels) for which every level has at least one cell: 2 * 2^5."""
    return 64


def forward(values, params):
    """net1 on its per-pixel values [10, G1h, G1w] -> [4, G1h, G1w]."""
    _, gh, gw = np.shape(values)
    if min(gh, gw) < min_input_size():
        n = min_input_size()
        raise ValueError("net1 needs an input of at least %d x %d pixels (five pooling levels)" % (n, n))
    buf = run(fold(values), params)
    return unfold(buf["out"], gh, gw)


def check_wiring_layers():
    """Every net1 layer is used exactly once, in upload order, with the activation rule."""
    convs = [s for s in NET1_WIRING if s["op"] == "conv"]
    return [s["layer"] for s in convs] == [l.name for l in L.NET1]
