# SPDX-License-Identifier: MIT
"""Frame generation (issue #96): NumPy CPU reference of the building blocks of its two convolution networks.

Written for this project. float64 arithmetic with only the prescribed float16 roundings, like the other oracles here.
What is settled: float16 tensors in and out, convolutions with a bias, ReLU, a 2x pooling, a bilinear 2x upscale in
the align-corners form, element-wise combines, and net1's per-pixel input features (two lumas with fixed weights,
two warp errors clamped to [0, 1], a mask and its complement) folded 2 x 2 into cells padded to 48 values.

What is not settled is OPEN (clean/fg/FORMATS.md). The code never picks a default for it: such a choice is a required
argument (the caller states it, and tests run every allowed value), or the function raises OpenItem.

Tensors are channel-first: x[c, y, x]. Weights here are in this module's own canonical order w[out, ky, kx, in];
the order inside the library's blobs is OPEN, so converting a blob into this order is not part of this module.
"""
import numpy as np

from clean.reference.fp8 import f16


class OpenItem(NotImplementedError):
    """A behaviour the spec side has not delivered yet (clean/fg/FORMATS.md, "OPEN")."""


# ---- settled pieces -------------------------------------------------------------------------------------------------

LUMA_WEIGHTS = (0.299, 0.587, 0.114)   # net1's luma inputs: fixed weights on R, G, B
NET1_FEATURES = 10                     # float16 values per pixel of net1's sampling grid
NET1_CELL = 48                         # one 2 x 2 cell: 4 * 10 = 40 values, padded to 48
PAD_MODES = ("zero", "replicate", "mirror")   # the 3 x 3 border rule is OPEN: the caller picks one of these
POOL_KINDS = ("max", "avg")                   # the 2x pooling kind is OPEN: the caller picks one of these


def relu(x):
    return np.maximum(np.asarray(x, dtype=np.float64), 0.0)


def _pad(x, p, mode):
    if p == 0:
        return x
    if mode == "zero":
        return np.pad(x, ((0, 0), (p, p), (p, p)), mode="constant")
    if mode == "replicate":
        return np.pad(x, ((0, 0), (p, p), (p, p)), mode="edge")
    if mode == "mirror":
        return np.pad(x, ((0, 0), (p, p), (p, p)), mode="reflect")
    raise ValueError("pad must be one of %s (the border rule is OPEN; no default)" % (PAD_MODES,))


def conv2d(x, w, b, *, pad, relu_after, acc="exact"):
    """One convolution layer: stride 1, 'same' size, odd square kernel (3 x 3 or 1 x 1).

    x [cin, H, W], w [out, k, k, cin], b [out]: inputs, weights and bias are rounded to float16 first (the layers
    take float16 operands). acc="exact": products summed in float64, plus the bias, then one float16 rounding, then
    ReLU if relu_after. The original sums in float16 with the matrix unit's order, which is OPEN, so acc="f16"
    raises OpenItem. pad: the border rule for k > 1 (PAD_MODES; OPEN, so required). relu_after: required, because
    which net2 layers end in ReLU is OPEN (every net1 layer does).
    """
    if acc == "f16":
        raise OpenItem("float16 accumulation order of the matrix unit")
    if acc != "exact":
        raise ValueError("acc must be 'exact' (or 'f16', OPEN)")
    x = f16(x)
    w = f16(w)
    b = f16(b)
    cout, kh, kw, cin = w.shape
    if kh != kw or kh % 2 != 1:
        raise ValueError("kernel must be square and odd")
    if x.shape[0] != cin or b.shape != (cout,):
        raise ValueError("shape mismatch: x %s, w %s, b %s" % (x.shape, w.shape, b.shape))
    H, W = x.shape[1:]
    p = kh // 2
    xp = _pad(x, p, pad) if p else x
    y = np.zeros((cout, H, W), dtype=np.float64)
    for ky in range(kh):
        for kx in range(kw):
            patch = xp[:, ky:ky + H, kx:kx + W]                    # [cin, H, W]
            y += np.einsum("oc,chw->ohw", w[:, ky, kx, :], patch)   # float64 sums
    y = f16(y + b[:, None, None])
    return relu(y) if relu_after else y


def pool2x(x, *, kind):
    """2x pooling over 2 x 2 blocks (H and W even). kind: "max" or "avg" (OPEN, so required). Output float16."""
    x = f16(x)
    c, H, W = x.shape
    if H % 2 or W % 2:
        raise ValueError("pool2x needs even H and W")
    blocks = x.reshape(c, H // 2, 2, W // 2, 2)
    if kind == "max":
        return blocks.max(axis=(2, 4))
    if kind == "avg":
        return f16(blocks.mean(axis=(2, 4)))
    raise ValueError("kind must be one of %s (OPEN; no default)" % (POOL_KINDS,))


def upscale2x_bilinear(x):
    """Bilinear 2x upscale, align-corners form: output n' = 2n per axis, output i samples input position
    i * (n - 1) / (n' - 1), so the corner samples are copied exactly. float64 interpolation, one float16 rounding."""
    x = f16(x)

    def axis_weights(n):
        m = 2 * n
        if n == 1:
            return np.zeros(m, dtype=np.int64), np.zeros(m, dtype=np.int64), np.zeros(m)
        pos = np.arange(m, dtype=np.float64) * (n - 1) / (m - 1)
        i0 = np.minimum(np.floor(pos).astype(np.int64), n - 2)
        return i0, i0 + 1, pos - i0

    _, H, W = x.shape
    y0, y1, fy = axis_weights(H)
    x0, x1, fx = axis_weights(W)
    rows = x[:, y0, :] * (1.0 - fy)[None, :, None] + x[:, y1, :] * fy[None, :, None]
    out = rows[:, :, x0] * (1.0 - fx)[None, None, :] + rows[:, :, x1] * fx[None, None, :]
    return f16(out)


def add(a, b):
    """Element-wise sum of two float16 tensors, one float16 rounding (which combines the network uses is OPEN)."""
    return f16(f16(a) + f16(b))


def concat(*xs):
    """Channel concatenation (an input merge); values unchanged."""
    return np.concatenate([f16(x) for x in xs], axis=0)


# ---- net1's per-pixel input features --------------------------------------------------------------------------------

def luma(rgb):
    """rgb [3, H, W] -> [H, W] with the fixed weights, rounded to float16."""
    rgb = np.asarray(rgb, dtype=np.float64)
    return f16(LUMA_WEIGHTS[0] * rgb[0] + LUMA_WEIGHTS[1] * rgb[1] + LUMA_WEIGHTS[2] * rgb[2])


def warp_error(frame, warped_other):
    """Mean absolute RGB difference between a frame and the other frame already warped onto it, clamped to [0, 1],
    rounded to float16. How the warp samples (motion field, filter, borders) is OPEN, so the warped frame is an input."""
    d = np.abs(np.asarray(frame, dtype=np.float64) - np.asarray(warped_other, dtype=np.float64)).mean(axis=0)
    return f16(np.clip(d, 0.0, 1.0))


def mask_pair(mask):
    """A mask value and its complement (x, 1 - x), float16."""
    m = f16(mask)
    return m, f16(1.0 - m)


def net1_features(features, *, order):
    """Stacks net1's 10 per-pixel features into [10, H, W]. features: {name: [H, W]} with the names
    "luma0", "luma1", "err0", "err1", "in0".."in3", "mask", "mask_c". order: the channel order, a permutation of those
    ten names. Which order the original uses is OPEN, so it is required."""
    names = {"luma0", "luma1", "err0", "err1", "in0", "in1", "in2", "in3", "mask", "mask_c"}
    if sorted(order) != sorted(names) or len(order) != NET1_FEATURES:
        raise ValueError("order must be a permutation of %s" % sorted(names))
    return np.stack([f16(features[n]) for n in order])


def fold2x2(x, *, order, pad_to=NET1_CELL):
    """Folds each 2 x 2 block of pixels of x [c, H, W] into one cell of 4c values, zero-padded to pad_to:
    -> [pad_to, H/2, W/2]. order: the place of each value in the cell, a permutation of (dy, dx, channel) given as
    a string of the three axis names from slowest to fastest, e.g. "dy dx c" or "c dy dx". The original's order is
    OPEN (so is what the pad holds: zeros here), so order is required."""
    axes = order.split()
    if sorted(axes) != ["c", "dx", "dy"]:
        raise ValueError("order must name dy, dx and c once each")
    x = f16(x)
    c, H, W = x.shape
    if H % 2 or W % 2:
        raise ValueError("fold2x2 needs even H and W")
    if 4 * c > pad_to:
        raise ValueError("cell too small")
    v = x.reshape(c, H // 2, 2, W // 2, 2)                       # c, Y, dy, X, dx
    pos = {"c": 0, "dy": 2, "dx": 4}
    v = v.transpose([pos[a] for a in axes] + [1, 3]).reshape(4 * c, H // 2, W // 2)
    out = np.zeros((pad_to, H // 2, W // 2), dtype=np.float64)
    out[:4 * c] = v
    return out


def unfold2x2(cells, c, *, order):
    """The inverse of fold2x2 for the first 4c values of each cell: [>= 4c, h, w] -> [c, 2h, 2w]."""
    axes = order.split()
    if sorted(axes) != ["c", "dx", "dy"]:
        raise ValueError("order must name dy, dx and c once each")
    cells = f16(cells)
    _, h, w = cells.shape
    size = {"c": c, "dy": 2, "dx": 2}
    v = cells[:4 * c].reshape([size[a] for a in axes] + [h, w])
    # current axes: axes..., Y, X  ->  c, Y, dy, X, dx
    cur = axes + ["Y", "X"]
    v = v.transpose([cur.index(a) for a in ("c", "Y", "dy", "X", "dx")])
    return v.reshape(c, 2 * h, 2 * w)


# ---- the layers of clean/fg/layout.py in this module's canonical order -----------------------------------------------

def weight_shape(layer):
    """Canonical weight shape [out, k, k, in] of a layout layer (layout.Net1Layer or layout.Net2Layer). net1's blob 2
    (1 x 1 or 2 x 2) raises OpenItem."""
    taps = getattr(layer, "taps", 9)
    if taps is None:
        raise OpenItem("net1 blob 2: 1 x 1 or 2 x 2")
    k = 3 if taps == 9 else 1
    return (layer.out, k, k, layer.cin)


# ---- the networks as a whole ----------------------------------------------------------------------------------------

def net1_forward(*args, **kwargs):
    """net1 end to end: OPEN (decoder levels and skip connections, blob 2's tap count, the upsampling, the grid size
    and the launch order are not delivered)."""
    raise OpenItem("net1 wiring: launch order, decoder levels, skip connections, blob 2 (1 x 1 or 2 x 2)")


def net2_forward(*args, **kwargs):
    """net2 end to end: OPEN (which layer reads which buffer, where the pooling, upscales and combines sit, which
    layers end in ReLU, the grid size)."""
    raise OpenItem("net2 wiring: launch order, pooling / upscale / combine placement, ReLU per layer")
