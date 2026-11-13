# SPDX-License-Identifier: MIT
"""Page 51 window block (51.3) and window geometry / borders (51.6), vectorised over windows, as used by D0-R.

Written independently from the spec text for the spec-55 head reference; generic in C and H (D0-R: C = 32, H = 2).

Data conventions (ours):
* tensors are [Y][X][channel]; e4m3 tensors are uint8 codes, f16 tensors float64 arrays holding binary16 values;
* window enumeration: windows (i, j) in row-major order (j outer, i inner), token t = 8*ly + lx inside a window;
* weights (page 51.2 logical tensors): Wq, Wv [H][C][32] e4m3 codes; Tb [H][12][16][16] f16 (block k = the k-th pair
  of TB_PAIRS); Wo [H][32][C] e4m3; bo [C] f16; W1 [C][4C] e4m3, b1 [4C] f16; W2 [4C][C] e4m3, b2 [C] f16;
* window-block constants (f16): A_SCALE, A_CLAMP, A_CLAMPn, A_P1, A_P0, ACT_CL, ACT_CLn, ACT_C1, ACT_C2, ACT_C3;
  stage constants ra, m (f16).
"""
import numpy as np

from . import fp

f64 = np.float64

# The 12 stored 16 x 16 blocks of the score bias, in the order listed on page 51.3 step 2 (storage order: ours,
# see QUESTIONS-55.md item 1).
TB_PAIRS = [(0, 0), (1, 0), (0, 1), (1, 1), (0, 2), (1, 2), (0, 3), (1, 3), (2, 2), (3, 2), (2, 3), (3, 3)]
CONST_NAMES = ["A_SCALE", "A_CLAMP", "A_CLAMPn", "A_P1", "A_P0", "ACT_CL", "ACT_CLn", "ACT_C1", "ACT_C2", "ACT_C3"]


class OutOfRange(ValueError):
    """Attention polynomial value p outside [1 + 1/32, 1 + 31/32): not specified (the converter rejects such
    constants, page 51.3 step 3)."""


# ------------------------------------------------------------------------------------------------ geometry
def reflect(v, n):
    """Page 51.6 border rule: v' = 0 if N = 1, else with P = 2(N-1), u = v mod P: v' = min(u, P - u)."""
    v = np.asarray(v, np.int64)
    if n == 1:
        return np.zeros_like(v)
    p = 2 * (n - 1)
    u = np.mod(v, p)
    return np.minimum(u, p - u)


def recurrent_shift(row):
    """A decoder recurrent half uses ((s + 4) mod 8) for each component s of its level's SH row (page 51.6)."""
    sx, sy = row
    for s in (sx, sy):
        if s % 2 or not 0 <= s < 8:
            raise ValueError("shift components must be even and below 8 (page 51.6)")
    return ((sx + 4) % 8, (sy + 4) % 8)


def window_coords(nx, ny, sx, sy):
    """Absolute token coordinates (X, Y) of every window token: arrays [nW][64]; window (i, j) covers
    X = 8i - sx + lx, Y = 8j - sy + ly, i < ceil((Nx + sx)/8), j < ceil((Ny + sy)/8)."""
    nwx = -(-(nx + sx) // 8)
    nwy = -(-(ny + sy) // 8)
    j, i, ly, lx = np.meshgrid(np.arange(nwy), np.arange(nwx), np.arange(8), np.arange(8), indexing="ij")
    X = (8 * i - sx + lx).reshape(nwy * nwx, 64)
    Y = (8 * j - sy + ly).reshape(nwy * nwx, 64)
    return X, Y


# ------------------------------------------------------------------------------------------------ matrix step
def mm_chain(xv, wv, c0):
    """MM(x; W; c0) of page 51.1: chain of 32-input steps in ascending k, one f16 rounding per step.
    xv [..., M, K] and wv [..., K, N] (batched like np.matmul) are e4m3 values (float64), c0 an f16 start value
    broadcastable to the result [..., M, N]."""
    K = xv.shape[-1]
    assert K % 32 == 0 and wv.shape[-2] == K
    acc = np.asarray(c0, f64)
    with np.errstate(all="ignore"):
        for k0 in range(0, K, 32):
            part = np.matmul(xv[..., k0:k0 + 32], wv[..., k0:k0 + 32, :])    # exact (see fp.py)
            acc = fp.r16(acc + part)
    return acc


def bias_matrix(tb_h):
    """B_h(a, b) of page 51.3 step 2 from the 12 stored blocks [12][16][16]."""
    B = np.empty((64, 64), f64)
    idx = {p: k for k, p in enumerate(TB_PAIRS)}
    for I in range(4):
        for J in range(4):
            if (I, J) in idx:
                B[16 * I:16 * I + 16, 16 * J:16 * J + 16] = tb_h[idx[(I, J)]]
            else:
                B[16 * I:16 * I + 16, 16 * J:16 * J + 16] = tb_h[idx[(J, I)]].T
    return B


# ------------------------------------------------------------------------------------------------ pieces
def attention_weight(s, k):
    """Page 51.3 step 3 for f16 scores s (float64 array); raises OutOfRange outside the specified range."""
    u = fp.mul16(s, k["A_SCALE"])
    c = fp.fmin(fp.fmax(u, k["A_CLAMPn"]), k["A_CLAMP"])
    inner = fp.fma16(c, -c, k["A_P1"])
    p = fp.fma16(c, inner, k["A_P0"])
    y = 32.0 * (p - 1.0)
    n = np.floor(y)
    if np.any((n < 1) | (n > 30)) or np.any(~np.isfinite(p)):
        raise OutOfRange("attention polynomial outside [1 + 1/32, 1 + 31/32)")
    return np.ldexp(1.0 + y - n, (n - 15).astype(np.int32))     # exact f16 values


def act(x, k):
    """Page 51.3 step 8 activation of f16 values."""
    kk = fp.fmin(fp.fmax(x, k["ACT_CLn"]), k["ACT_CL"])
    u = fp.fma16(-np.abs(kk), k["ACT_C1"], k["ACT_C2"])
    v = fp.fma16(kk, u, k["ACT_C3"])
    return fp.mul16(x, v)


def check_attention_constants(k):
    """The converter's check of page 51.3 step 3: p of every f16 c with A_CLAMPn <= c <= A_CLAMP in range. Also
    returns w_max and whether 64 * w_max reaches the f16 overflow threshold (65,520)."""
    c = fp.all_f16()
    c = c[np.isfinite(c) & (c >= k["A_CLAMPn"]) & (c <= k["A_CLAMP"])]
    inner = fp.fma16(c, -c, k["A_P1"])
    p = fp.fma16(c, inner, k["A_P0"])
    y = 32.0 * (p - 1.0)
    n = np.floor(y)
    ok = bool(np.all((n >= 1) & (n <= 30)))
    wmax = float(np.max(np.ldexp(1.0 + y - n, (n - 15).astype(np.int32)))) if ok else float("nan")
    return ok, wmax, (64 * wmax >= 65520.0) if ok else None


# ------------------------------------------------------------------------------------------------ the block
def window_block(xw, W, k, ra, m, diag=None):
    """Page 51.3 common core on windows xw [nW][64][C] (uint8 e4m3 codes; residual base d8(x)).
    Returns y2h [nW][64][C] (f16 values as float64). diag, if a dict, receives Z [nW][H][64] and y2 codes."""
    H = W["Wq"].shape[0]
    C = xw.shape[-1]
    x = fp.d8(xw)
    O = []
    Zs = []
    for h in range(H):
        q = fp.d8(fp.q8(mm_chain(x, fp.d8(W["Wq"][h]), 0.0)))
        v = fp.d8(fp.q8(mm_chain(x, fp.d8(W["Wv"][h]), 0.0)))
        B = bias_matrix(np.asarray(W["Tb"][h], f64))
        with np.errstate(all="ignore"):
            s = fp.r16(np.matmul(q, np.swapaxes(q, -1, -2)) + B)    # one matrix step, start value B
        w = attention_weight(s, k)
        Z = fp.r16(np.sum(w, axis=-1))                                # exact sum of 64 values, rounded once
        with np.errstate(all="ignore"):
            r = fp.r16(1.0 / Z)                                       # 1/Z is never near an f16 midpoint
        P = fp.d8(fp.q8(fp.mul16(w, r[..., None])))
        O.append(fp.d8(fp.q8(mm_chain(P, v, 0.0))))                   # keys 0-31, then 32-63
        Zs.append(Z)
    Ocat = np.concatenate(O, axis=-1)                                 # head h -> inputs 32h .. 32h+31
    Wo = fp.d8(np.asarray(W["Wo"]).reshape(H * 32, C))
    Dp = mm_chain(Ocat, Wo, W["bo"])
    y1h = fp.fma16(Dp, ra, x)
    y1q = fp.d8(fp.q8(y1h))
    Du = mm_chain(y1q, fp.d8(W["W1"]), W["b1"])
    Au = fp.d8(fp.q8(act(Du, k)))
    Dd = mm_chain(Au, fp.d8(W["W2"]), W["b2"])
    y2h = fp.fma16(Dd, m, y1h)
    if diag is not None:
        diag["Z"] = np.stack(Zs, axis=1)
        diag["y2"] = fp.q8(y2h)
    return y2h


def run_block_stage(x, W, k, ra, m, shift, diag=None):
    """Window block over a whole level tensor x [Ny][Nx][C] (uint8 codes) with window shift (sx, sy).
    Reads are reflected per 51.6; results only for tokens inside the tensor. Returns y2h [Ny][Nx][C]."""
    ny, nx, C = x.shape
    sx, sy = shift
    Xw, Yw = window_coords(nx, ny, sx, sy)
    xw = x[reflect(Yw, ny), reflect(Xw, nx)]
    y2w = window_block(xw, W, k, ra, m, diag)
    out = np.full((ny, nx, C), np.nan)
    inside = (Xw >= 0) & (Xw < nx) & (Yw >= 0) & (Yw < ny)
    out[Yw[inside], Xw[inside]] = y2w[inside]
    if diag is not None:
        diag["windows"] = (Xw, Yw)
    return out
