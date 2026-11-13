# SPDX-License-Identifier: MIT
"""Model K FP16 linear layers (spec 41, task K1): per-token linear, 2x2 merge, 2x2 expand.

Numerics (spec 40/41): inputs, parameters and outputs are FP16. A matrix product plus its bias is rounded to FP16
once: y = f16(W x + b) (the sum is exact here, float64; the GPU accumulates in FP32 and starts from the bias). Then,
each step one FP16 operation rounded to nearest even, in this order: ReLU (embedding), activation (spec 40), and the
residual / skip add y = f16(r + y).

Orientation: W is [out, in] ("output index first", as spec 90). Merge input and expand output order (spec 41):
sub-token s = 2 dy + dx (top-left, top-right, bottom-left, bottom-right); merge input index s C + c, expand output
index s C + c.
"""
import numpy as np

from clean.k.reference.act import act
from clean.k.reference.f16 import f16
from clean.k.reference.layout import cdiv


def matmul_f16(x, W, b=None):
    """f16(x W^T + b) with an exact (float64) sum: x [..., K], W [N, K], b [N] -> [..., N] float16."""
    x = np.asarray(x, dtype=np.float16).astype(np.float64)
    W = np.asarray(W, dtype=np.float16).astype(np.float64)
    y = x @ W.T
    if b is not None:
        y = y + np.asarray(b, dtype=np.float16).astype(np.float64)
    return f16(y)


def epilogue(y, relu=False, use_act=False, res=None):
    y = np.asarray(y, dtype=np.float16)
    if relu:
        y = np.maximum(y, np.float16(0))
    if use_act:
        y = act(y)
    if res is not None:
        y = (np.asarray(res, dtype=np.float16) + y).astype(np.float16)
    return y


def linear(x, W, b=None, relu=False, use_act=False, res=None):
    """Per-token linear on a map x [..., C_in] -> [..., C_out]."""
    return epilogue(matmul_f16(x, W, b), relu, use_act, res)


def merge_gather(x):
    """x [H, W, C] -> z [ceil(H/2), ceil(W/2), 4C]: the four tokens (2i + dy, 2j + dx) in order s = 2 dy + dx.
    Members outside the map (odd H or W; never with the sizes of spec 44) are zero (QUESTIONS.md K-Q3)."""
    x = np.asarray(x, dtype=np.float16)
    h, w, c = x.shape
    h2, w2 = cdiv(h, 2), cdiv(w, 2)
    pad = np.zeros((2 * h2, 2 * w2, c), dtype=np.float16)
    pad[:h, :w] = x
    sub = [pad[dy::2, dx::2] for dy in (0, 1) for dx in (0, 1)]
    return np.concatenate(sub, axis=-1)


def merge(x, W, b=None):
    """2x2 merge: x [H, W, C] -> [ceil(H/2), ceil(W/2), C_next] with W [C_next, 4 C]."""
    return linear(merge_gather(x), W, b)


def expand(y, W, b=None, out_size=None, skip=None):
    """2x2 expand: y [h, w, C_prev] -> [H, W, C] with W [4 C, C_prev]; out_size (W, H) defaults to (2w, 2h) and must
    satisfy ceil(W / 2) = w, ceil(H / 2) = h. Output token (2i + dy, 2j + dx) is e[s C .. s C + C) of token (i, j),
    s = 2 dy + dx; then x = f16(e + S) with the skip map S [H, W, C] (optional)."""
    y = np.asarray(y, dtype=np.float16)
    h, w, _ = y.shape
    ow, oh = out_size if out_size is not None else (2 * w, 2 * h)
    if cdiv(ow, 2) != w or cdiv(oh, 2) != h:
        raise ValueError(f"expand: output size {(ow, oh)} does not match the source {(w, h)}")
    e = matmul_f16(y, W, b)  # [h, w, 4 C]
    c = e.shape[-1] // 4
    full = np.zeros((2 * h, 2 * w, c), dtype=np.float16)
    for dy in (0, 1):
        for dx in (0, 1):
            s = 2 * dy + dx
            full[dy::2, dx::2] = e[..., s * c:(s + 1) * c]
    out = full[:oh, :ow]
    if skip is not None:
        out = (np.asarray(skip, dtype=np.float16) + out).astype(np.float16)
    return out
