# SPDX-License-Identifier: MIT
"""Weight packing for the Swin kernels (layouts documented in clean/FORMATS.md).

The kernels read every matrix as ready-made WMMA operand fragments: fragment = 32 lanes x 8 bytes, lane l holds row
(or B column) l % 16 and the 8 K values of slot set h = l // 16. Which logical row/column goes to which fragment row
and K slot is chosen here so that the FP32 result of one WMMA is directly the FP8 operand of the next one.

D_ROW describes the WMMA accumulator layout: element v of lane l (h = l // 16) holds row D_ROW[h][v]. The packing is
the only place that depends on it; the default is the gfx12 layout assumed by the kernels' design (row = v + 8h).
"""
import numpy as np

from clean.reference.fp8 import fp8_encode
from clean.reference.swin import STAGES, HEAD_DIM

D_ROW_LAYOUTS = {
    "gfx12": np.array([[v + 8 * h for v in range(8)] for h in range(2)]),   # row = v + 8h
    "gfx11": np.array([[2 * v + h for v in range(8)] for h in range(2)]),   # row = 2v + h (alternative, for probes)
}

FRAG_BYTES = 256
ALIGN = 256


# ---- row / K orders ---------------------------------------------------------------------------------------------
# R[16*mt + r] = logical row placed in fragment row r of tile mt.  K[16*kc + 8*h + j] = logical column in slot (h, j)
# of K group kc.

def rows_identity(n, drow):
    return np.arange(n)


def rows_layout_l(C, drow):
    """Output rows such that lane half h, element v of tile mt holds channel h*C/2 + 8*mt + v (layout L)."""
    R = np.empty(C, dtype=np.int64)
    for mt in range(C // 16):
        for h in range(2):
            for v in range(8):
                R[16 * mt + drow[h][v]] = h * (C // 2) + 8 * mt + v
    return R


def rows_expand(C, drow):
    """Expand output rows: tile mt = k*(C/16) + g holds sub-pixel k, layout L group g."""
    G = C // 16
    R = np.empty(4 * C, dtype=np.int64)
    for k in range(4):
        for g in range(G):
            for h in range(2):
                for v in range(8):
                    R[16 * (k * G + g) + drow[h][v]] = k * C + h * (C // 2) + 8 * g + v
    return R


def kord_layout_l(C):
    """K slot (h, j) of group g holds channel h*C/2 + 8g + j (activations in layout L)."""
    K = np.empty(C, dtype=np.int64)
    for g in range(C // 16):
        for h in range(2):
            for j in range(8):
                K[16 * g + 8 * h + j] = h * (C // 2) + 8 * g + j
    return K


def kord_drow(n, drow):
    """K slot (h, j) of group g holds index 16g + drow[h][j] (activations that come straight out of a WMMA)."""
    K = np.empty(n, dtype=np.int64)
    for g in range(n // 16):
        for h in range(2):
            for j in range(8):
                K[16 * g + 8 * h + j] = 16 * g + drow[h][j]
    return K


def kord_merge(C):
    """Patch-merge input: lane half h holds z[h*2C, (h+1)*2C) (two neighbouring tokens of row 2i + h)."""
    K = np.empty(4 * C, dtype=np.int64)
    for g in range(C // 4):
        for h in range(2):
            for j in range(8):
                K[16 * g + 8 * h + j] = h * 2 * C + 8 * g + j
    return K


# ---- packers ----------------------------------------------------------------------------------------------------

def frag_array(M, R, K):
    """Matrix M [rows, cols] (FP8-exact) -> uint8 array [MT, KC, 256]: fragment (mt, kc), lane 16h + r, byte j
    holds M[R[16 mt + r], K[16 kc + 8h + j]]."""
    M = np.asarray(M, dtype=np.float64)
    assert M.shape == (len(R), len(K)) and M.shape[0] % 16 == 0 and M.shape[1] % 16 == 0
    P = fp8_encode(M[np.ix_(R, K)])                       # [16*MT, 16*KC]
    MT, KC = P.shape[0] // 16, P.shape[1] // 16
    P = P.reshape(MT, 16, KC, 2, 8)                        # [mt, r, kc, h, j]
    return np.ascontiguousarray(P.transpose(0, 2, 3, 1, 4)).reshape(MT, KC, 256)  # [mt, kc, (h, r, j)]


def pack_frags(M, R, K):
    """Fragments of M in [mt][kc] order, as bytes."""
    return frag_array(M, R, K).tobytes()


def pack_tile_vector(vec, R, drow):
    """Bias vector in accumulator order: packed[16*mt + 8*h + v] = vec[R[16*mt + drow[h][v]]] (float32)."""
    vec = np.asarray(vec, dtype=np.float32)
    out = np.empty(len(R), dtype=np.float32)
    for mt in range(len(R) // 16):
        for h in range(2):
            for v in range(8):
                out[16 * mt + 8 * h + v] = vec[R[16 * mt + drow[h][v]]]
    return out.tobytes()


def _pad(b):
    return b + bytes((-len(b)) % ALIGN)


def block_layout(C, heads, mlp):
    """Byte offsets of the packed block weights; must match BlockLayout in kernels/swin.hip."""
    KC, MT = C // 16, mlp // 16
    o = {"FLAGS": 0, "G1": 16}
    o["G2"] = o["G1"] + 4 * C
    o["BP"] = o["G2"] + 4 * C
    o["B2"] = o["BP"] + 4 * C
    o["B1"] = o["B2"] + 4 * C
    o["BIAS"] = o["B1"] + 4 * mlp
    o["STREAM"] = o["BIAS"] + 4 * heads * 256
    o["HEAD_BYTES"] = (4 * KC + 2 * KC) * FRAG_BYTES                       # QKV + WP fragments of one head
    limit = 16384 // (2 * KC * FRAG_BYTES)
    o["HPU"] = next(d for d in range(min(limit, MT), 0, -1) if MT % d == 0)  # MLP hidden tiles per unit (<= 16 KB)
    o["MLP_BYTES"] = 2 * KC * o["HPU"] * FRAG_BYTES
    o["NMLP"] = MT // o["HPU"]
    o["SIZE"] = o["STREAM"] + heads * o["HEAD_BYTES"] + o["NMLP"] * o["MLP_BYTES"]
    return o


SAFE_LIMIT = 460.0   # an FP8 conversion without clamp gives the saturated result for finite values up to 464


def block_bounds(p, C):
    """Worst-case magnitudes of the values that are converted to FP8 inside a Swin block, from the weights alone.

    RMSNorm output n = gamma * x / sqrt(mean(x^2) + eps) has L2 norm <= max|gamma| * sqrt(C) for ANY input x, and q8
    changes each element by at most 1/16 relative (plus a negligible absolute part), so ||a||_2 <= 1.07 max|gamma1|
    sqrt(C). By Cauchy-Schwarz every row of a projection is bounded by its L2 norm times that: q/k elements (Wqk),
    values and so attention outputs o (convex combinations of values, Wv), the second norm's output m (gamma2) and the
    MLP hidden values (W1, b1; GELU never increases the magnitude). Returns the bounds and whether all are <= 460."""
    p = {k: np.asarray(v, dtype=np.float64) for k, v in p.items()}
    root_c = float(np.sqrt(C))
    na = 1.07 * float(np.max(np.abs(p["gamma1"]))) * root_c + 0.01 * root_c
    nm = 1.07 * float(np.max(np.abs(p["gamma2"]))) * root_c + 0.01 * root_c
    row_l2 = lambda w: np.sqrt(np.sum(w * w, axis=-1))  # noqa: E731
    bounds = {
        "a": na,
        "u": float(np.max(row_l2(p["Wqk"]))) * na,
        "o": float(np.max(row_l2(p["Wv"]))) * na,
        "m": nm,
        "hidden": float(np.max(row_l2(p["W1"]) * nm + np.abs(p["b1"]))),
    }
    return bounds, all(b <= SAFE_LIMIT for b in bounds.values())


def pack_block(p, C, heads, mlp, drow):
    """Header (FP32 vectors, bias table) + fragment stream in kernel consumption order (see BlockLayout)."""
    D = heads * HEAD_DIM
    KC, MT = C // 16, mlp // 16
    lay = block_layout(C, heads, mlp)
    f32 = lambda a: np.asarray(a, dtype=np.float32).tobytes()  # noqa: E731
    RL = rows_layout_l(C, drow)
    KL = kord_layout_l(C)
    B = np.asarray(p["B"], dtype=np.float32)
    bias = np.empty((heads, 16, 2, 8), dtype=np.float32)
    for h in range(2):
        for v in range(8):
            bias[:, :, h, v] = B[:, :, drow[h][v]]
    _, safe = block_bounds(p, C)
    header = [
        np.array([1 if safe else 0, 0, 0, 0], dtype=np.uint32).tobytes(),   # FLAGS
        f32(p["gamma1"]), f32(p["gamma2"]),
        pack_tile_vector(p["bp"], RL, drow),
        pack_tile_vector(p["b2"], RL, drow),
        pack_tile_vector(p["b1"], rows_identity(mlp, drow), drow),
        bias.tobytes(),
    ]
    wp = frag_array(p["Wp"], RL, kord_drow(D, drow))                       # [KC, 2*heads]
    w1 = frag_array(p["W1"], rows_identity(mlp, drow), KL)                 # [MT, KC]
    w2 = frag_array(p["W2"], RL, kord_drow(mlp, drow))                     # [KC, MT]
    stream = []
    for hd in range(heads):
        wq = frag_array(p["Wqk"][hd], rows_identity(HEAD_DIM, drow), KL)   # [2, KC]
        wv = frag_array(p["Wv"][hd], rows_identity(HEAD_DIM, drow), KL)    # [2, KC]
        stream += [np.stack([wq, wv], axis=2)]                             # unit QKV(hd): [mt][kc][q|v]
        stream += [wp[:, 2 * hd:2 * hd + 2]]                               # unit WP(hd): [mt][dt]
    for ht in range(MT):                                                   # units MLP(u): HPU slices each
        stream += [w1[ht], w2[:, ht]]
    blob = b"".join(header) + b"".join(np.ascontiguousarray(a).tobytes() for a in stream)
    assert len(blob) == lay["SIZE"]
    return blob


def pack_merge(Wm, bm, C, C_out, drow):
    RL = rows_layout_l(C_out, drow)
    return pack_tile_vector(bm, RL, drow) + pack_frags(Wm, RL, kord_merge(C))


def pack_expand(We, be, C, C_low, drow):
    RE = rows_expand(C, drow)
    return pack_tile_vector(be, RE, drow) + pack_frags(We, RE, kord_layout_l(C_low))


def stage_sections(stage):
    """(block size, offset of the merge/expand section) inside a packed stage blob."""
    cfg = STAGES[stage]
    size = block_layout(cfg["C"], cfg["heads"], cfg["mlp"])["SIZE"]
    return size, size + (-size) % ALIGN


def check_tensors(stage, t):
    cfg = STAGES[stage]
    C, heads, mlp, D = cfg["C"], cfg["heads"], cfg["mlp"], cfg["heads"] * HEAD_DIM
    shapes = {
        "gamma1": (C,), "gamma2": (C,), "Wqk": (heads, HEAD_DIM, C), "Wv": (heads, HEAD_DIM, C),
        "B": (heads, 16, 16), "Wp": (C, D), "bp": (C,), "W1": (mlp, C), "b1": (mlp,), "W2": (C, mlp), "b2": (C,),
    }
    if cfg["kind"] == "enc":
        shapes.update(Wm=(cfg["C_out"], 4 * C), bm=(cfg["C_out"],))
    if cfg["kind"] == "dec":
        shapes.update(We=(4 * C, cfg["C_low"]), be=(4 * C,))
    for k, s in shapes.items():
        if k not in t:
            raise KeyError(f"{stage}: missing tensor {k}")
        if tuple(np.shape(t[k])) != s:
            raise ValueError(f"{stage}: tensor {k} has shape {np.shape(t[k])}, expected {s}")


def resolve_d_row(d_row):
    """A layout name from D_ROW_LAYOUTS, or a 2 x 8 table (e.g. the "d_row" reported by selftest.py)."""
    if isinstance(d_row, str):
        if d_row.strip().startswith("["):
            import json
            d_row = json.loads(d_row)
        else:
            return D_ROW_LAYOUTS[d_row]
    t = np.asarray(d_row, dtype=np.int64)
    if t.shape != (2, 8) or sorted(t.ravel().tolist()) != list(range(16)):
        raise ValueError(f"d_row must be a 2 x 8 permutation of 0..15, got {d_row!r}")
    return t


def pack_stage(stage, tensors, d_row="gfx12"):
    """Logical tensors of spec/90 -> one byte blob: [block weights][pad to 256][merge or expand weights]."""
    check_tensors(stage, tensors)
    drow = resolve_d_row(d_row)
    cfg = STAGES[stage]
    blob = _pad(pack_block(tensors, cfg["C"], cfg["heads"], cfg["mlp"], drow))
    if cfg["kind"] == "enc":
        blob += pack_merge(tensors["Wm"], tensors["bm"], cfg["C"], cfg["C_out"], drow)
    elif cfg["kind"] == "dec":
        blob += pack_expand(tensors["We"], tensors["be"], cfg["C"], cfg["C_low"], drow)
    return blob


# Pseudo-stage "bneck": the six bottleneck blocks b0..b5 as one unit (megakernel swin_bneck6, FORMATS.md). Its blob is
# the six b-stage blobs back to back (equal sizes), so block k's section starts at k * len(blob) / 6.
BNECK_STAGES = tuple(f"b{k}" for k in range(6))


def pack_bneck(tensors, d_row="gfx12"):
    """tensors: {"b0": tensors of b0, ..., "b5": ...} or a sequence of the six -> the six packed b-stage blobs, joined."""
    seq = [tensors[s] for s in BNECK_STAGES] if isinstance(tensors, dict) else list(tensors)
    if len(seq) != 6:
        raise ValueError("bneck needs the tensors of the six blocks b0..b5")
    return b"".join(pack_stage(s, t, d_row) for s, t in zip(BNECK_STAGES, seq))


def split_bneck(blob):
    """The six b-stage blobs of a bneck blob (or a sequence of six blobs, returned as a list)."""
    if not isinstance(blob, (bytes, bytearray, memoryview)):
        seq = [bytes(b) for b in blob]
    else:
        blob = bytes(blob)
        n = len(blob) // 6
        if n * 6 != len(blob) or n == 0:
            raise ValueError("a bneck blob holds six equal b-stage blobs")
        seq = [blob[k * n:(k + 1) * n] for k in range(6)]
    if len(seq) != 6:
        raise ValueError("bneck needs six blobs")
    return seq


# Pseudo-stage "mid": enc2 (with its merge), b0..b5, dec2 (with its expand) as one unit (persistent kernel swin_mid,
# FORMATS.md). Its blob is the eight stage blobs back to back, in this order.
MID_STAGES = ("enc2",) + BNECK_STAGES + ("dec2",)


def stage_blob_size(stage):
    """Length of pack_stage(stage, ...): the padded block section plus the merge or expand section."""
    cfg = STAGES[stage]
    C = cfg["C"]
    tail = 0
    if cfg["kind"] == "enc":
        tail = 4 * cfg["C_out"] + cfg["C_out"] * 4 * C        # bias, then (C_out / 16) x (4 C / 16) fragments
    elif cfg["kind"] == "dec":
        tail = 4 * (4 * C) + (4 * C) * cfg["C_low"]           # bias, then (4 C / 16) x (C_low / 16) fragments
    return stage_sections(stage)[1] + tail


def pack_mid(tensors, d_row="gfx12"):
    """tensors: {"enc2": ..., "b0": ..., ..., "b5": ..., "dec2": ...} or the eight in that order -> blobs joined."""
    seq = [tensors[s] for s in MID_STAGES] if isinstance(tensors, dict) else list(tensors)
    if len(seq) != len(MID_STAGES):
        raise ValueError("mid needs the tensors of enc2, b0..b5 and dec2")
    return b"".join(pack_stage(s, t, d_row) for s, t in zip(MID_STAGES, seq))


def split_mid(blob):
    """The eight stage blobs of a mid blob (or a sequence of eight blobs, returned as a list)."""
    if not isinstance(blob, (bytes, bytearray, memoryview)):
        seq = [bytes(b) for b in blob]
    else:
        blob, seq, o = bytes(blob), [], 0
        for s in MID_STAGES:
            n = stage_blob_size(s)
            seq.append(blob[o:o + n])
            o += n
        if o != len(blob):
            raise ValueError(f"a mid blob holds {o} bytes, got {len(blob)}")
    if len(seq) != len(MID_STAGES):
        raise ValueError("mid needs eight blobs")
    return seq
