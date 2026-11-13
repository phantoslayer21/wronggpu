# SPDX-License-Identifier: MIT
"""NumPy reference ("oracle") for spec/02-swin-block.md.

Everything is computed in float64; the only roundings are the ones the spec writes down (q8, f16). Feature maps
are arrays shaped [H, W, C]. Parameter dictionaries use the logical tensor names of spec/90-test-interface.md.
"""
import math

import numpy as np

from .fp8 import q8, f16

TAU = 0.35337
CAP = 32.369
WIN = 4  # window side (4x4 = 16 tokens)
HEAD_DIM = 32

STAGES = {
    "enc1": dict(kind="enc", C=64, heads=2, mlp=256, C_out=96),
    "enc2": dict(kind="enc", C=96, heads=4, mlp=384, C_out=128),
    **{f"b{i}": dict(kind="b", C=128, heads=4, mlp=512) for i in range(6)},
    "dec2": dict(kind="dec", C=96, heads=4, mlp=384, C_low=128),
    "dec1": dict(kind="dec", C=64, heads=2, mlp=256, C_low=96),
}


def window_offset(stage, parity):
    """(ox, oy) from the table in spec 02 "Window partition"."""
    if stage in ("b1", "b3", "b5", "dec2", "dec1"):
        return (0, 0) if parity == 0 else (2, 0)
    return (2, 2) if parity == 0 else (0, 2)


def mirror(p, n):
    """Mirror about the edge pixel, edge not repeated: -1 -> 1, -2 -> 2, n -> n-2, n+1 -> n-3."""
    p = np.asarray(p)
    p = np.where(p < 0, -p, p)
    return np.where(p >= n, 2 * (n - 1) - p, p)


def window_grid(H, W, offset):
    """Return (rows, cols): rows[l] / cols[k] are the 4 unmirrored token coordinates of window row l / column k."""
    ox, oy = offset
    nwx = (W + ox + WIN - 1) // WIN
    nwy = (H + oy + WIN - 1) // WIN
    cols = np.arange(nwx)[:, None] * WIN - ox + np.arange(WIN)[None, :]
    rows = np.arange(nwy)[:, None] * WIN - oy + np.arange(WIN)[None, :]
    return rows, cols


def rmsnorm(v, gamma, C):
    eps = 2.0 ** -13 / C
    return gamma * v / np.sqrt(np.mean(v * v, axis=-1, keepdims=True) + eps)


_erf = np.frompyfunc(math.erf, 1, 1)


def gelu(x):
    return 0.5 * x * (1.0 + _erf(x / math.sqrt(2.0)).astype(np.float64))


def gelu_sigmoid(x):
    """The cheap GELU used by the kernels by default (WG_GELU_MODE 2): x / (1 + exp(-z)), z = x (a + b x^2) with (a, b)
    fitted by the clean team to minimize FP8 flips against the exact GELU. Accepted by the spec team on whole-frame
    quality (spec 09 draft 2); the per-stage limits of the exact GELU do not hold for it."""
    x = np.asarray(x, dtype=np.float64)
    z = x * (1.59989588569617 + 0.06951594437094213 * x * x)
    with np.errstate(over="ignore"):
        return x / (1.0 + np.exp(-z))


def gelu_as725(x):
    """GELU through the 3-term erfc approximation 7.1.25 of Abramowitz & Stegun (WG_GELU_MODE 1)."""
    x = np.asarray(x, dtype=np.float64)
    ax = np.abs(x)
    t = 1.0 / (1.0 + 0.47047 * ax / math.sqrt(2.0))
    q = 0.5 * (0.3480242 * t - 0.0958798 * t * t + 0.7478556 * t ** 3) * np.exp(-ax * ax / 2.0)   # Phi(-|x|)
    return 0.5 * x + ax * (0.5 - q)


def _h(a):
    return np.asarray(a).astype(np.float16)


def _fma16(a, b, c):
    """v_pk_fma_f16: one rounding of a*b+c to FP16 (computed in FP64, exact for FP16 inputs)."""
    return (a.astype(np.float64) * b.astype(np.float64) + c.astype(np.float64)).astype(np.float16)


def _mul16(a, b):
    return (a.astype(np.float64) * b.astype(np.float64)).astype(np.float16)


def gelu_poly16(x):
    """The packed FP16 polynomial GELU of the kernels (WG_GELU_MODE 3), FP16 arithmetic simulated operation by
    operation. u = clamp(x/3, -1, 1), Phi/2 = 0.25 + u p'(u^2) with p' of degree 4 (the halved polynomial), and
    y = x Phi from one dot2 (exact in FP32). FP32 -> FP16 rounds toward zero (v_cvt_pkrtz_f16_f32); x is clamped to 448
    before the product, so y <= 448 (Phi <= 1.0007)."""
    x32 = np.asarray(x, dtype=np.float32)
    xh = _h(x32)
    back = xh.astype(np.float32)
    toward = ((x32 > 0) & (back > x32)) | ((x32 < 0) & (back < x32))
    xh = np.where(toward, np.nextafter(xh, np.float16(0)), xh).astype(np.float16)
    c = [np.float16(v) for v in (0.5985, -0.8731, 1.0131, -0.69135, 0.20285)]
    full = lambda v: np.full_like(xh, v)  # noqa: E731
    u = _mul16(xh, full(np.float16(0.33333334)))
    u = np.minimum(np.maximum(u, full(np.float16(-1))), full(np.float16(1)))
    t = _mul16(u, u)
    p = _fma16(t, full(c[4]), full(c[3]))
    for k in (c[2], c[1], c[0]):
        p = _fma16(t, p, full(k))
    r = _fma16(u, p, full(np.float16(0.25)))                       # Phi / 2
    xc = np.minimum(xh, full(np.float16(448)))
    return 2.0 * xc.astype(np.float64) * r.astype(np.float64)      # the dot2: exact in FP32


GELU_FUNCTIONS = {"exact": gelu, "sigmoid": gelu_sigmoid, "as725": gelu_as725, "poly16": gelu_poly16}


def soft_cap(S):
    Sc = np.clip(S, -CAP, CAP)  # g is continuous at |S| = c: c - c^3/(3c^2) = 2c/3
    return Sc - Sc ** 3 / (3.0 * CAP * CAP)


def _block_tokens(xw, p, heads, round_attn_f16, gelu_fn=gelu):
    """One Swin block on gathered windows. xw: [..., 16, C] float64. Returns x2 [..., 16, C]."""
    C = xw.shape[-1]
    lead = xw.shape[:-2]
    a = q8(rmsnorm(xw, p["gamma1"], C))                                      # steps 1-2
    u = q8(np.einsum("hdc,...tc->...htd", p["Wqk"], a))                      # step 3
    v = f16(np.einsum("hdc,...tc->...htd", p["Wv"], a))
    S = np.einsum("...htd,...hsd->...hts", u, u) + p["B"]                    # step 4
    z = TAU * soft_cap(S)                                                    # step 5
    z = z - z.max(axis=-1, keepdims=True)
    A = np.exp(z)
    A = A / A.sum(axis=-1, keepdims=True)
    if round_attn_f16:
        A = f16(A)
    o = np.einsum("...hts,...hsd->...htd", A, v)                             # step 6
    o = np.moveaxis(o, -3, -2).reshape(lead + (16, heads * HEAD_DIM))        # head-major concat
    x1 = xw + q8(o) @ p["Wp"].T + p["bp"]                                    # step 7
    m = q8(rmsnorm(x1, p["gamma2"], C))                                      # step 8
    h = q8(gelu_fn(m @ p["W1"].T + p["b1"]))                                # step 9
    return x1 + h @ p["W2"].T + p["b2"]                                      # step 10


def swin_block(x, p, offset, heads, round_attn_f16=True, max_tokens=1 << 16, gelu_fn=gelu):
    """One Swin block on a whole feature map x [H, W, C]. Returns x2 (float64, not yet quantized).

    Window members outside the map are the tokens at the mirrored positions (spec 02 "Borders"); for the decoder
    stages x is the exact sum e + S from expand(), so such a member is e + S evaluated at the mirrored position.
    round_attn_f16: spec 02 step 5 rounds the attention weights to FP16 (False only for experiments).
    """
    x = np.asarray(x, dtype=np.float64)
    H, W, C = x.shape
    p = {k: np.asarray(v, dtype=np.float64) for k, v in p.items()}
    rows, cols = window_grid(H, W, offset)
    out = np.full_like(x, np.nan)
    rows_per_chunk = max(1, max_tokens // (16 * cols.shape[0]))
    for l0 in range(0, rows.shape[0], rows_per_chunk):
        r = rows[l0:l0 + rows_per_chunk]                       # [nl, 4]
        rr = mirror(r, H)[:, None, :, None]                    # [nl, 1, 4, 1]
        cc = mirror(cols, W)[None, :, None, :]                 # [1, nk, 1, 4]
        xw = x[rr, cc]                                         # [nl, nk, 4, 4, C]
        nl, nk = xw.shape[:2]
        x2 = _block_tokens(xw.reshape(nl, nk, 16, C), p, heads, round_attn_f16, gelu_fn).reshape(nl, nk, 4, 4, C)
        # keep only real (unmirrored, in-range) window members; each real token is in exactly one window
        rv = (r >= 0) & (r < H)
        cv = (cols >= 0) & (cols < W)
        sel = rv[:, None, :, None] & cv[None, :, None, :]
        R = np.broadcast_to(r[:, None, :, None], sel.shape)
        K = np.broadcast_to(cols[None, :, None, :], sel.shape)
        out[R[sel], K[sel]] = x2[sel]
    assert not np.isnan(out).any(), "window grid does not cover the feature map"
    return out


def patch_merge(s, Wm, bm):
    """s: q8 block output [H, W, C] -> merged q8 tokens [H/2, W/2, C']."""
    s = np.asarray(s, dtype=np.float64)
    H, W, C = s.shape
    z = np.concatenate([s[0::2, 0::2], s[0::2, 1::2], s[1::2, 0::2], s[1::2, 1::2]], axis=-1)
    return q8(z @ np.asarray(Wm, np.float64).T + np.asarray(bm, np.float64))


def expand(y, skip, We, be):
    """y: [H/2, W/2, C_low], skip: [H, W, C] -> block input x [H, W, C] (exact sum e + S, not re-quantized)."""
    y = np.asarray(y, dtype=np.float64)
    skip = np.asarray(skip, dtype=np.float64)
    H, W, C = skip.shape
    e = q8(y @ np.asarray(We, np.float64).T + np.asarray(be, np.float64))     # [H/2, W/2, 4C]
    r = np.arange(H)
    c = np.arange(W)
    sr = mirror(r // 2, y.shape[0])
    sc = mirror(c // 2, y.shape[1])
    k = 2 * (r % 2)[:, None] + (c % 2)[None, :]                                 # [H, W]
    e_full = e[sr[:, None], sc[None, :]].reshape(H, W, 4, C)
    x = np.take_along_axis(e_full, k[:, :, None, None], axis=2)[:, :, 0, :]
    return x + skip


def run_stage(stage, p, x, offset, skip=None, round_attn_f16=True, gelu="exact"):
    """Reference for one stage with the inputs/outputs of spec/90-test-interface.md (float32 FP8-exact arrays)."""
    cfg = STAGES[stage]
    if cfg["kind"] == "dec":
        xin = expand(x, skip, p["We"], p["be"])
    else:
        xin = np.asarray(x, dtype=np.float64)
    x2 = swin_block(xin, p, offset, cfg["heads"], round_attn_f16=round_attn_f16, gelu_fn=GELU_FUNCTIONS[gelu])
    out = q8(x2)
    if cfg["kind"] == "enc":
        return {"skip": out.astype(np.float32), "merged": patch_merge(out, p["Wm"], p["bm"]).astype(np.float32)}
    return {"out": out.astype(np.float32)}


def random_params(stage, rng, weight_scale=1.0):
    """Synthetic parameters with the shapes of spec/90 (matrices FP8-exact). Never model data."""
    cfg = STAGES[stage]
    C, heads, mlp = cfg["C"], cfg["heads"], cfg["mlp"]
    D = heads * HEAD_DIM

    def mat(rows, cols):
        return q8(rng.standard_normal((rows, cols)) * weight_scale / math.sqrt(cols)).astype(np.float32)

    def vec(n, scale=0.1):
        return (rng.standard_normal(n) * scale).astype(np.float32)

    p = {
        "gamma1": (0.5 + rng.random(C)).astype(np.float32),
        "gamma2": (0.5 + rng.random(C)).astype(np.float32),
        "Wqk": q8(rng.standard_normal((heads, HEAD_DIM, C)) * 2.0 / math.sqrt(C)).astype(np.float32),
        "Wv": q8(rng.standard_normal((heads, HEAD_DIM, C)) / math.sqrt(C)).astype(np.float32),
        "B": (rng.standard_normal((heads, 16, 16)) * 2.0).astype(np.float32),
        "Wp": mat(C, D), "bp": vec(C),
        "W1": mat(mlp, C), "b1": vec(mlp),
        "W2": mat(C, mlp), "b2": vec(C),
    }
    if cfg["kind"] == "enc":
        p["Wm"] = mat(cfg["C_out"], 4 * C)
        p["bm"] = vec(cfg["C_out"])
    if cfg["kind"] == "dec":
        p["We"] = mat(4 * C, cfg["C_low"])
        p["be"] = vec(4 * C)
    return p


def random_inputs(stage, H, W, rng, scale=1.0):
    """Synthetic FP8-exact inputs for a stage at stage resolution H x W: returns (x, skip)."""
    cfg = STAGES[stage]
    C = cfg["C"]
    if cfg["kind"] == "dec":
        x = q8(rng.standard_normal((H // 2, W // 2, cfg["C_low"])) * scale).astype(np.float32)
        skip = q8(rng.standard_normal((H, W, C)) * scale).astype(np.float32)
        return x, skip
    return q8(rng.standard_normal((H, W, C)) * scale).astype(np.float32), None
