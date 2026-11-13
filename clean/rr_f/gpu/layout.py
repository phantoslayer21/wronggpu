# SPDX-License-Identifier: MIT
"""Runtime buffer layouts of the spec 51 D3D12 kernels (DESIGN-51.md section 3; spec 51.2: "how a runtime arranges
them in its own buffers is its own choice"). Built by our loader from the logical tensors of 51.2.

* WMMA fragment (256 bytes): lane l = 16 hf + r holds row r and K positions 16 ks + 8 hf .. + 7 (byte j = K 8 hf + j);
  the same layout serves as A operand (row = output row) and B operand (row = output column), clean/FORMATS.md.
  A matrix M [rows][K] becomes fragments [rows / 16][K / 16] (tile-major).
* Accumulator (v8f): lane l = 16 hf + c holds column c and rows 8 hf + v (v = 0..7).
* Activations: token-major [Y][X][C] e4m3 (the logical layout); hidden states [Y][X][4] f16.
* Score bias: the 12 stored blocks expanded to all 16 (I, J) blocks, as the C operand of the key x query score tile:
  [h][I = query tile][J = key tile][lane][8] f32 with element v of lane 16 hf + c = B_h(16 I + c, 16 J + 8 hf + v).
* Biases: f32 vectors (exact widening of the f16 values), followed by their split for the exact step Y
  (kernels/rr_step_y.hlsli): [values | cb | cs] with cb = the nearest integer (ties to even) and cs = value - cb.
"""
import numpy as np

from clean.rr_f.reference.block import score_bias
from clean.rr_f.reference.fmt import from_f16_bits, to_f16_bits


def frags(M):
    """M [R][K] (uint8 codes), R and K multiples of 16 -> fragments [R/16][K/16][32 lanes][8] uint8."""
    M = np.asarray(M, np.uint8)
    R, K = M.shape
    assert R % 16 == 0 and K % 16 == 0, M.shape
    t = M.reshape(R // 16, 16, K // 16, 2, 8)          # [mt, r, ks, hf, j]
    return np.ascontiguousarray(t.transpose(0, 2, 3, 1, 4)).reshape(R // 16, K // 16, 32, 8)


def pad_rows(M, rows):
    out = np.zeros((rows, M.shape[1]), np.uint8)
    out[:M.shape[0]] = M
    return out


def f32(v):
    return np.asarray(v, np.float64).astype(np.float32)


def bias_split(v):
    """f16 bias values -> f32 [values | cb | cs]: cb = nearest integer (ties to even), cs = v - cb (exact)."""
    v = np.asarray(v, np.float64)
    cb = np.round(v)                                     # halves to even, as the kernel's round (v_rndne)
    return np.concatenate([f32(v), f32(cb), f32(v - cb)])


def bias_tiles(Tb):
    """Tb [H][12][16][16] f16 values -> [H][4][4][32][8] f32 (C operand of the score tiles)."""
    H = Tb.shape[0]
    out = np.zeros((H, 4, 4, 32, 8), np.float32)
    for h in range(H):
        B = score_bias(Tb[h])                            # [a][b]
        for I in range(4):
            for J in range(4):
                # lane 16 hf + c, element v: query a = 16 I + c, key b = 16 J + 8 hf + v
                blk = B[16 * I:16 * I + 16, 16 * J:16 * J + 16]   # [c][kk]
                out[h, I, J] = blk.reshape(16, 2, 8).transpose(1, 0, 2).reshape(32, 8)
    return out


def h2pair(a, b):
    """Two f16 values -> one dword (a low, b high)."""
    ba = int(to_f16_bits(np.array([a]))[0])
    bb = int(to_f16_bits(np.array([b]))[0])
    return ba | (bb << 16)


def pack_weights(W, C, H):
    """Logical weights of one stage (dict of 51.2 tensors) -> {name: array} in the kernels' layouts."""
    P = {}
    # q / v projections: per head Wq^T [32][C] and Wv^T [32][C] (Wv used as the B operand: row = feature column)
    qv = []
    for h in range(H):
        qv.append(frags(W["Wq"][h].T))                   # [2][C/16]
        qv.append(frags(W["Wv"][h].T))
    P["Wqv"] = np.stack(qv).reshape(H, 2, 2, C // 16, 32, 8)   # [h][q|v][ft][ks]
    tb = bias_tiles(W["Tb"]).astype(np.float64)
    ca = np.round(tb)                                    # the score bias split for the exact step Y (ties to even)
    P["Tb"] = np.concatenate([f32(tb).ravel(), f32(ca).ravel(), f32(tb - ca).ravel()])   # [values | ca | cb]
    P["Wo"] = np.stack([frags(W["Wo"][h].T) for h in range(H)])  # [h][C/16][2]
    P["bo"] = bias_split(W["bo"])
    # MLP chunk units j (32 hidden channels each): W1^T rows 32 j .. 32 j + 31 [ft][ks], then W2^T [ct][K-slice 2 j + e]
    w1 = frags(W["W1"].T)                                # [4C/16][C/16]
    w2 = frags(W["W2"].T)                                # [C/16][4C/16]
    nj = 4 * C // 32
    mlp = []
    for j in range(nj):
        mlp.append(w1[2 * j:2 * j + 2].reshape(-1, 32, 8))
        mlp.append(w2[:, 2 * j:2 * j + 2].reshape(-1, 32, 8))
    P["MLP"] = np.concatenate(mlp).reshape(nj, 4 * (C // 16), 32, 8)
    P["b1"] = bias_split(W["b1"])
    P["b2"] = bias_split(W["b2"])
    if "Wm" in W:
        P["Wm"] = frags(W["Wm"].T)                       # [Cn/16][4C/16]
        P["bm"] = bias_split(W["bm"])
    if "Wa" in W:
        P["Wa"] = frags(W["Wa"].T)                       # [64/16][128/16]
        P["ba"] = bias_split(W["ba"])
    if "Wup" in W:
        P["Wup"] = frags(W["Wup"].T)                     # [4C/16][Kc/16]
        P["bup"] = bias_split(W["bup"])
    if "Wh" in W:
        P["Wh"] = frags(pad_rows(W["Wh"].T, 16))         # [1][C/16]
        bh = np.zeros(16, np.float64)
        bh[:W["bh"].shape[0]] = np.asarray(W["bh"], np.float64)
        P["bh"] = bias_split(bh)
    return P


def int_planes(M):
    """Variant I (kernels/rr_step_i.hlsli): a weight matrix M [R][K] (e4m3 codes, as passed to frags) -> (planes
    [2][R][K] uint8 (int8 digits), table [R/16][K/32][2] uint32, ok).

    Per block (16 rows x 32 inputs: one output tile and one 51.1 step): beta = the smallest E = max(exponent, 1) over
    the nonzero codes (1 for an all-zero block), v' = M 2^(E - beta) (an integer, |v'| < 2^(4 + span)), digits
    d0 = v' - 2^7 d1, d1 = trunc(v' / 2^7) (both with the sign of the value, |d| <= 127). ok is False when a block
    needs a third digit (span > 10) or holds a NaN code. table = (beta, f32 bits of L1' = the largest sum of |v'|
    over the block's 32 inputs of one row)."""
    from clean.rr_f.reference.fmt import D8_TABLE
    M = np.asarray(M, np.uint8)
    R, K = M.shape
    assert R % 16 == 0 and K % 32 == 0, M.shape
    mag = M & 0x7F
    nan = mag == 0x7F
    e = (mag >> 3).astype(np.int64)
    E = np.maximum(e, 1)
    nz = (mag != 0) & ~nan
    blk = lambda a: a.reshape(R // 16, 16, K // 32, 32)
    Eb = blk(np.where(nz, E, 99))
    beta = Eb.min(axis=(1, 3))
    beta = np.where(beta == 99, 1, beta)                                   # all-zero block
    Emax = blk(np.where(nz, E, 0)).max(axis=(1, 3))
    span = np.where(blk(nz).any(axis=(1, 3)), Emax - beta, 0)
    ok = bool(not nan.any() and (span <= 10).all())
    bfull = np.repeat(np.repeat(beta, 16, axis=0), 32, axis=1)             # [R][K]
    vabs = np.where(nz, np.abs(D8_TABLE[np.where(nan, 0, M)]), 0.0)
    vp = np.ldexp(vabs, (10 - bfull).astype(np.int64))                     # |v'|, exact integers
    assert np.all(vp == np.round(vp)) and (not ok or vp.max(initial=0) < 2 ** 14)
    vpi = vp.astype(np.int64)
    sign = np.where((M & 0x80) != 0, -1, 1)
    d1 = sign * (vpi >> 7)
    d0 = sign * (vpi & 127)
    planes = np.stack([d0, d1]).astype(np.int8).view(np.uint8)
    l1 = blk(vp).sum(axis=3).max(axis=1)                                   # [R/16][K/32]
    table = np.stack([beta.astype(np.uint32), l1.astype(np.float32).view(np.uint32)], axis=-1)
    return planes, table, ok


def pack_weights_int(W, C, H):
    """Variant I layouts (kernels/rr_block.hlsl RR_STEP 4): every weight tensor as two int8 digit planes followed by its
    table of (beta, L1') per (tile, step). Units staged through LDS hold [plane][unit] (QV_h, Wo_h, MLP chunk j);
    directly read tensors (Wm, Wa, Wup, Wh) hold [plane][fragments]. Returns ({name: array}, ok)."""
    P = pack_weights(W, C, H)                            # biases, score bias (unchanged)
    ok = True
    KS, NS, NJ = C // 16, C // 32, 4 * C // 32

    def planes_of(M):
        nonlocal ok
        pl, tb, o = int_planes(M)
        ok = ok and o
        return [frags(pl[0]), frags(pl[1])], tb

    units, tabs = [], []
    for h in range(H):
        pq, tq = planes_of(W["Wq"][h].T)                 # [32][C]: 2 tiles x NS steps
        pv, tv = planes_of(W["Wv"][h].T)
        units.append(np.stack([np.concatenate([pq[p].reshape(-1, 32, 8), pv[p].reshape(-1, 32, 8)]) for p in (0, 1)]))
        tabs.append(np.stack([tq, tv]))                  # [qv][ft][s][2]
    P["Wqv"] = np.concatenate([np.stack(units).ravel().view(np.uint8), np.stack(tabs).astype(np.uint32).ravel().view(np.uint8)])
    units, tabs = [], []
    for h in range(H):
        po, to = planes_of(W["Wo"][h].T)                 # [C][32]: KS tiles x 1 step
        units.append(np.stack([po[p].reshape(-1, 32, 8) for p in (0, 1)]))
        tabs.append(to[:, 0])                            # [ct][2]
    P["Wo"] = np.concatenate([np.stack(units).ravel().view(np.uint8), np.stack(tabs).astype(np.uint32).ravel().view(np.uint8)])
    p1, t1 = planes_of(W["W1"].T)                        # [4C][C]: tiles 2j + ft, NS steps
    p2, t2 = planes_of(W["W2"].T)                        # [C][4C]: KS tiles, NJ steps
    units = []
    for j in range(NJ):
        units.append(np.stack([np.concatenate([p1[p][2 * j:2 * j + 2].reshape(-1, 32, 8),
                                               p2[p][:, 2 * j:2 * j + 2].reshape(-1, 32, 8)]) for p in (0, 1)]))
    tab = np.concatenate([t1.reshape(NJ, 2, NS, 2).ravel(), np.transpose(t2, (1, 0, 2)).ravel()]).astype(np.uint32)
    P["MLP"] = np.concatenate([np.stack(units).ravel().view(np.uint8), tab.view(np.uint8)])

    def direct(M):
        pl, tb = planes_of(M)
        return np.concatenate([pl[0].ravel(), pl[1].ravel(), tb.astype(np.uint32).ravel().view(np.uint8)])
    if "Wm" in W:
        P["Wm"] = direct(W["Wm"].T)                      # [Cn][4C]
    if "Wa" in W:
        P["Wa"] = direct(W["Wa"].T)                      # [64][128]
    if "Wup" in W:
        P["Wup"] = direct(W["Wup"].T)                    # [4C][Kc]
    if "Wh" in W:
        P["Wh"] = direct(pad_rows(W["Wh"].T, 16))        # [16][C]
    assert KS * 16 == C
    return P, ok


def f16_bytes(v):
    return to_f16_bits(np.asarray(v, np.float64)).astype("<u2")


def f16_from_bytes(b, shape):
    return from_f16_bits(np.frombuffer(np.ascontiguousarray(b).tobytes(), "<u2").reshape(shape))
