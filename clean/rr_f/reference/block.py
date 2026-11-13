# SPDX-License-Identifier: MIT
"""The window block of spec 51.3 (common core) and the window grid / border rule of 51.6.

Tensors are [Y][X][C]. e4m3 tensors are uint8 codes; f16 tensors are float64 arrays of f16 values."""
import numpy as np

from .fmt import d8, fma16, mm, mul16, q8, round_f16
from .scalar import activation, attention_weight, recip16

# 51.3 step 2: the 12 stored 16 x 16 score-bias blocks (I, J); Tb_h[k] is the block of STORED_PAIRS[k]
# (PROVISIONAL: storage index = position in the list of 51.3 step 2; page 56 not available).
STORED_PAIRS = [(0, 0), (1, 0), (0, 1), (1, 1), (0, 2), (1, 2), (0, 3), (1, 3), (2, 2), (3, 2), (2, 3), (3, 3)]


def score_bias(Tb_h):
    """B_h(a, b) [64][64] from the 12 stored blocks Tb_h [12][16][16] (f16 values)."""
    B = np.empty((64, 64), dtype=np.float64)
    index = {p: k for k, p in enumerate(STORED_PAIRS)}
    for I in range(4):
        for J in range(4):
            if (I, J) in index:
                blk = Tb_h[index[(I, J)]]
            else:
                blk = Tb_h[index[(J, I)]].T
            B[16 * I:16 * I + 16, 16 * J:16 * J + 16] = blk
    return B


def window_block(x, xr, W, ra, m, wc, Z_override=None, trace=False):
    """51.3 steps 1-9 for a batch of windows.

    x: e4m3 codes [Nw, 64, C] (block input), xr: f16 values [Nw, 64, C] (residual base),
    W: dict with Wq, Wv [H][C][32] (codes), Tb [H][12][16][16] (f16), Wo [H][32][C] (codes), bo [C], W1 [C][4C]
    (codes), b1 [4C], W2 [4C][C] (codes), b2 [C]; ra, m: f16 scalars; wc: WindowConsts.
    Z_override: optional f16 row sums [Nw, H, 64] used instead of the reference's own (51.8 item 1).
    Returns (y2 codes [Nw, 64, C], info dict with Z [Nw, H, 64] and, if trace, every intermediate)."""
    Nw, T, C = x.shape
    assert T == 64
    H = W["Wq"].shape[0]
    xv = d8(x)
    info = {}
    Zs, Os = [], []
    tr = {k: [] for k in ("q", "v", "s", "w", "P", "O")} if trace else None
    for h in range(H):
        # step 1
        qc = q8(mm(xv, d8(W["Wq"][h])))                       # [Nw, 64, 32]
        vc = q8(mm(xv, d8(W["Wv"][h])))
        qv, vv = d8(qc), d8(vc)
        # step 2: one matrix step with start value B_h(a, b)
        B = score_bias(W["Tb"][h])
        with np.errstate(invalid="ignore"):
            part = np.einsum("wai,wbi->wab", qv, qv) + 0.0
            s = round_f16(part + B[None])
        # step 3
        w = attention_weight(s, wc)
        # step 4: Z = exact sum rounded once to f16 (the reference's order, WrongGPU's choice)
        Z = round_f16(w.sum(axis=-1))
        if Z_override is not None:
            Z = np.asarray(Z_override, dtype=np.float64)[:, h, :]
        r = recip16(Z)
        P = q8(mul16(w, r[..., None]))                       # [Nw, a, b]
        # step 5: keys 0-31 then 32-63
        O = q8(mm(d8(P), vv))                                # [Nw, 64, 32]
        Zs.append(Z)
        Os.append(O)
        if trace:
            for k, v in (("q", qc), ("v", vc), ("s", s), ("w", w), ("P", P), ("O", O)):
                tr[k].append(v)
    info["Z"] = np.stack(Zs, axis=1)                          # [Nw, H, 64]
    # step 6: chain over heads, start value bo
    Ocat = np.concatenate([d8(o) for o in Os], axis=-1)       # [Nw, 64, 32H], head-major
    Dp = mm(Ocat, d8(W["Wo"]).reshape(H * 32, C), W["bo"])
    # step 7
    y1h = fma16(Dp, ra, xr)
    y1q = q8(y1h)
    # step 8
    Du = mm(d8(y1q), d8(W["W1"]), W["b1"])
    Au = q8(activation(Du, wc))
    Dd = mm(d8(Au), d8(W["W2"]), W["b2"])
    # step 9
    y2h = fma16(Dd, m, y1h)
    y2 = q8(y2h)
    if trace:
        for k in tr:
            info[k] = np.stack(tr[k], axis=1)                 # [Nw, H, ...]
        info.update(Dp=Dp, y1h=y1h, y1q=y1q, Du=Du, Au=Au, Dd=Dd, y2h=y2h)
    return y2, info


# ---------------------------------------------------------------------------------------------------------------
# 51.6 window grid and border rule

def reflect(v, N):
    """Border rule of 51.6 for coordinates v (int array) of an axis with extent N."""
    v = np.asarray(v, dtype=np.int64)
    if N == 1:
        return np.zeros_like(v)
    P = 2 * (N - 1)
    u = np.mod(v, P)
    return np.minimum(u, P - u)


class Grid:
    """Window grid of a stage call on a tensor of Nx x Ny tokens with shift (sx, sy)."""

    def __init__(self, Nx, Ny, sx, sy):
        if sx % 2 or sy % 2 or not (0 <= sx < 8 and 0 <= sy < 8):
            raise ValueError("shift components must be even and below 8 (51.6)")
        self.Nx, self.Ny, self.sx, self.sy = Nx, Ny, sx, sy
        self.nwx = -(-(Nx + sx) // 8)
        self.nwy = -(-(Ny + sy) // 8)
        i = np.arange(self.nwx)
        j = np.arange(self.nwy)
        l = np.arange(8)
        # absolute coordinates of window token t = 8*ly + lx of window (i, j); window index w = j*nwx + i
        Xw = (8 * i[:, None] - sx + l[None, :])                     # [nwx, 8]
        Yw = (8 * j[:, None] - sy + l[None, :])                     # [nwy, 8]
        self.X = np.broadcast_to(Xw[None, :, None, :], (self.nwy, self.nwx, 8, 8)).reshape(-1, 64)
        self.Y = np.broadcast_to(Yw[:, None, :, None], (self.nwy, self.nwx, 8, 8)).reshape(-1, 64)
        self.real = (self.X >= 0) & (self.X < Nx) & (self.Y >= 0) & (self.Y < Ny)
        self.nw = self.nwx * self.nwy

    def gather(self, tensor):
        """Per-window read of a tensor at this grid's level (reflected per axis)."""
        Ny, Nx = tensor.shape[:2]
        assert (Nx, Ny) == (self.Nx, self.Ny)
        return tensor[reflect(self.Y, Ny), reflect(self.X, Nx)]

    def scatter(self, vals, fill):
        """Write per-window results of the real tokens into a full tensor."""
        out = np.full((self.Ny, self.Nx) + vals.shape[2:], fill, dtype=vals.dtype)
        cnt = np.zeros((self.Ny, self.Nx), dtype=np.int64)
        m = self.real
        out[self.Y[m], self.X[m]] = vals[m]
        np.add.at(cnt, (self.Y[m], self.X[m]), 1)
        assert np.all(cnt == 1), "every real token must belong to exactly one window"
        return out
