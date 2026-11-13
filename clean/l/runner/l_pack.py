# SPDX-License-Identifier: MIT
"""Model L weight packing (issue #89): the device blobs of Model L's own launches, from the structure table
(clean/l/reference/table.py) and the parameters of the oracle ({stage: tensors}, clean/l/reference/oracle.py).

Model L's encoder and decoder run as a chain of small FP8-map kernels, each with a blob in a format Model M already
uses (clean/FORMATS.md). Maps are [tokens][C] FP8, channel fastest; pixel maps are on the padded grid F, token maps
on F/2.

| Blob | Format | Kernel input -> output |
|---|---|---|
| `enc0.values` | Model M's enc0 blob (clean/runner/enc0.py pack) with W = 0 and b = 0 | frame -> the quantized values as a pixel map [F][32] (values 0..19, 12 zero bytes) |
| `enc0.<w>` per linear | merge / expand section format: bias tile vector (rows R_L(C_out)), then fragments [mt < C_out/16][kc < C_in/16], rows R_L(C_out), K order K_L(C_in); inputs narrower than C_in (20 values) get zero weights | FP8 map [C_in] -> FP8 map [C_out] |
| `enc0_block`, `dec0_block` | Model M's Swin block section (packing.pack_block) at the block's width, padded to 256 | FP8 map -> FP8 map |
| `enc0.project` | bias tile vector (rows R_L(64)), then fragments [mt][kb], rows R_L(64), K order: block 2i + s of group i holds channel 32 i + 16 h + 8 s + j (dec0's order, clean/runner/dec0.py kord_decoder, over every group) | the 2 x 2 pixels of each token (channel 32 p + i, p = dx + 2 dy) -> the 64 FP8 features enc1 reads |
| `dec0.project` | expand section format: bias tile vector (rows R_E(32)), then fragments [mt < 8][kb < 4], rows R_E(32) (tile 2 s + g: sub-pixel s, channels 16 g ..), K order as `enc0.project` (two groups of 32) | dec1's map [F/2][64] -> pixel map [F][32] |
| `dec0.coef` | Model M's dec0 blob (clean/runner/dec0.py pack) with W = 0 and b = 0: the scalars S, T and the code tables | the decoder block's pixel map -> A, B, C, latent |

The ten shared Swin stages use clean/runner/packing.py pack_stage unchanged.
"""
import numpy as np

from clean.l.reference import table as _table
from clean.reference.fp8 import fp8_decode
from clean.runner import dec0 as _dec0
from clean.runner import enc0 as _enc0
from clean.runner import packing as P

ALIGN = P.ALIGN


# ---- orders ---------------------------------------------------------------------------------------------------------

def kord_groups32(n):
    """K slot (h, j) of block kb holds channel 32 (kb // 2) + 16 h + 8 (kb % 2) + j: each group of 32 channels is two
    whole blocks, so a kernel can round its running sum after every second block."""
    if n % 32:
        raise ValueError(f"input of {n} channels: groups of 32 only")
    K = np.empty(n, dtype=np.int64)
    for kb in range(n // 16):
        for h in range(2):
            for j in range(8):
                K[16 * kb + 8 * h + j] = 32 * (kb // 2) + 16 * h + 8 * (kb % 2) + j
    return K


# ---- blob packers ---------------------------------------------------------------------------------------------------

def pack_linear(W, b, C_in, C_out, lo=0, d_row="gfx12"):
    """W [C_out, n] reading input channels lo .. lo + n of a C_in-wide map (zero weights elsewhere), bias [C_out]."""
    drow = P.resolve_d_row(d_row)
    W = np.asarray(W, dtype=np.float64)
    n = W.shape[1]
    if W.shape[0] != C_out or lo < 0 or lo + n > C_in or C_in % 16 or C_out % 16:
        raise ValueError(f"linear {W.shape} at {lo} of {C_in} -> {C_out}: bad shape (multiples of 16 needed)")
    Wx = np.zeros((C_out, C_in))
    Wx[:, lo:lo + n] = W
    RL = P.rows_layout_l(C_out, drow)
    return P.pack_tile_vector(b, RL, drow) + P.pack_frags(Wx, RL, P.kord_layout_l(C_in))


def pack_project(W, b, groups, d_row="gfx12"):
    drow = P.resolve_d_row(d_row)
    W = np.asarray(W, dtype=np.float64)
    n_out, n = W.shape
    if any(g != 32 for g in groups) or sum(groups) != n:
        raise ValueError(f"project groups {groups}: groups of 32 covering {n} channels only")
    RL = P.rows_layout_l(n_out, drow)
    return P.pack_tile_vector(b, RL, drow) + P.pack_frags(W, RL, kord_groups32(n))


def pack_dec_project(W, b, groups=(32, 32), d_row="gfx12"):
    """W [4, 32, n], b [4, 32] -> bias (rows R_E(32)) + fragments (rows R_E(32), K order kord_groups32(n))."""
    drow = P.resolve_d_row(d_row)
    W = np.asarray(W, dtype=np.float64).reshape(128, -1)
    if any(g != 32 for g in groups) or sum(groups) != W.shape[1]:
        raise ValueError(f"dec_project groups {groups}: groups of 32 covering {W.shape[1]} channels only")
    RE = P.rows_expand(32, drow)
    return P.pack_tile_vector(np.asarray(b, np.float32).reshape(128), RE, drow) + \
        P.pack_frags(W, RE, kord_groups32(W.shape[1]))


def pack_block(tensors, C, heads, mlp, d_row="gfx12"):
    drow = P.resolve_d_row(d_row)
    blob = P.pack_block(tensors, C, heads, mlp, drow)
    return blob + bytes((-len(blob)) % ALIGN)


def pack_values(tensors, d_row="gfx12"):
    """Model M's enc0 blob with zero projection: only the dither table and the two scalars are used."""
    t = {"W": np.zeros((64, 80), np.float32), "b": np.zeros(64, np.float32),
         "T": tensors.get("T", np.zeros((8, 16, 16), np.float32)),
         "sigma1": tensors.get("sigma1", np.float32(0)), "sigma2": tensors.get("sigma2", np.float32(0))}
    return _enc0.pack(t, d_row)


def pack_coef(tensors, d_row="gfx12"):
    """Model M's dec0 blob with zero projection: the scalars S, T and the code tables."""
    return _dec0.pack({"W": np.zeros((68, 64), np.float32), "b": np.zeros(68, np.float32),
                       "S": tensors["S"], "T": tensors["T"]}, d_row)


# ---- launch lists ---------------------------------------------------------------------------------------------------

def encoder_launches(t=_table.TABLE):
    """The encoder's kernels in order: [{"kind", "blob", ...}]. kinds: "values" (quant into a pixel map of 32
    channels), "linear", "swin", "project" (with the space_to_depth gather folded in)."""
    ops = list(t["enc_ops"])
    _table.encoder_plan(t)
    width = {"values": 20}
    out = []
    for i, op in enumerate(ops):
        k = op["op"]
        if k == "quant":
            if op["mode"] != "dither" or i != 0:
                raise ValueError("the encoder's value kernel quantizes the per-pixel values with the dither table")
            out.append({"kind": "values", "blob": "enc0.values", "dst": op["dst"], "C": 32})
            width[op["dst"]] = 32
        elif k == "linear":
            c_in = width[op["src"]]
            out.append({"kind": "linear", "blob": "enc0." + op["w"], "src": op["src"], "dst": op["dst"],
                        "C_in": c_in, "C_out": op["n_out"], "w": op["w"], "b": op["b"], "acc": op.get("acc"),
                        "out_round": op.get("out_round", "q8")})
            width[op["dst"]] = op["n_out"]
        elif k == "swin":
            out.append({"kind": "swin", "blob": op["params"], "src": op["src"], "dst": op["dst"], "C": op["C"],
                        "heads": op["heads"], "mlp": op["mlp"], "offsets": op["offsets"], "params": op["params"]})
        elif k == "space_to_depth":
            nxt = ops[i + 1] if i + 1 < len(ops) else None
            if not nxt or nxt["op"] != "project" or nxt["src"] != op["dst"]:
                raise ValueError("space_to_depth must feed the project directly (one gather kernel)")
            width[op["dst"]] = 4 * width[op["src"]]
            out.append({"kind": "project", "blob": "enc0.project", "src": op["src"], "groups": nxt["groups"],
                        "w": nxt["w"], "b": nxt["b"], "clamp": nxt.get("clamp")})
        elif k == "project":
            continue                                   # folded into the gather above
        else:
            raise ValueError(f"no kernel for encoder op {k!r}")
    return out


def decoder_launches(t=_table.TABLE):
    """The decoder's kernels in order: "dec_project" (with depth_to_space folded in), "swin", "coef" (pick + the
    coefficient functions)."""
    ops = list(t["dec_ops"])
    _table.decoder_plan(t)
    out = []
    for i, op in enumerate(ops):
        k = op["op"]
        if k == "dec_project":
            nxt = ops[i + 1] if i + 1 < len(ops) else None
            if not nxt or nxt["op"] != "depth_to_space" or nxt["src"] != op["dst"]:
                raise ValueError("dec_project must feed depth_to_space directly (one scatter kernel)")
            out.append({"kind": "dec_project", "blob": "dec0.project", "w": op["w"], "b": op["b"],
                        "groups": op["groups"], "dst": nxt["dst"]})
        elif k == "depth_to_space":
            continue
        elif k == "swin":
            out.append({"kind": "swin", "blob": op["params"], "src": op["src"], "dst": op["dst"], "C": op["C"],
                        "heads": op["heads"], "mlp": op["mlp"], "offsets": op["offsets"], "params": op["params"]})
        elif k == "pick":
            out.append({"kind": "coef", "blob": "dec0.coef", "src": op["src"], "values": tuple(op["values"])})
        else:
            raise ValueError(f"no kernel for decoder op {k!r}")
    return out


def pack_model_l(params, t=_table.TABLE, d_row="gfx12"):
    """{blob name: bytes} of Model L's own launches (the shared Swin stages: packing.pack_stage)."""
    out = {}
    enc = params["enc0"]
    for ln in encoder_launches(t):
        if ln["kind"] == "values":
            out[ln["blob"]] = pack_values(enc, d_row)
        elif ln["kind"] == "linear":
            out[ln["blob"]] = pack_linear(enc[ln["w"]], enc[ln["b"]], ln["C_in"], ln["C_out"], 0, d_row)
        elif ln["kind"] == "swin":
            out[ln["blob"]] = pack_block(params[ln["params"]], ln["C"], ln["heads"], ln["mlp"], d_row)
        elif ln["kind"] == "project":
            out[ln["blob"]] = pack_project(enc[ln["w"]], enc[ln["b"]], ln["groups"], d_row)
    dec = params["dec0"]
    for ln in decoder_launches(t):
        if ln["kind"] == "dec_project":
            out[ln["blob"]] = pack_dec_project(dec[ln["w"]], dec[ln["b"]], ln["groups"], d_row)
        elif ln["kind"] == "swin":
            out[ln["blob"]] = pack_block(params[ln["params"]], ln["C"], ln["heads"], ln["mlp"], d_row)
        elif ln["kind"] == "coef":
            out[ln["blob"]] = pack_coef(dec, d_row)
    return out


# ---- decoding (tests) -----------------------------------------------------------------------------------------------

def unpack_frags(raw, R, K):
    """Inverse of packing.frag_array: fragment bytes [MT][KC][256] -> the matrix M [len(R), len(K)] (FP8 values)."""
    MT, KC = len(R) // 16, len(K) // 16
    a = np.frombuffer(bytes(raw), dtype=np.uint8).reshape(MT, KC, 2, 16, 8)       # [mt, kc, h, r, j]
    P8 = a.transpose(0, 3, 1, 2, 4).reshape(16 * MT, 16 * KC)                     # [16 mt + r, 16 kc + 8 h + j]
    M = np.zeros((len(R), len(K)), dtype=np.float64)
    M[np.ix_(R, K)] = fp8_decode(P8)
    return M


def unpack_tile_vector(raw, R, drow):
    v = np.frombuffer(bytes(raw), dtype=np.float32)
    out = np.empty(len(R), dtype=np.float32)
    for mt in range(len(R) // 16):
        for h in range(2):
            for k in range(8):
                out[R[16 * mt + drow[h][k]]] = v[16 * mt + 8 * h + k]
    return out


def unpack_linear(blob, C_in, C_out, d_row="gfx12"):
    """(W [C_out, C_in], b [C_out]) from a linear blob."""
    drow = P.resolve_d_row(d_row)
    RL = P.rows_layout_l(C_out, drow)
    b = unpack_tile_vector(blob[:4 * C_out], RL, drow)
    return unpack_frags(blob[4 * C_out:], RL, P.kord_layout_l(C_in)), b


def unpack_project(blob, n_in, n_out, d_row="gfx12"):
    drow = P.resolve_d_row(d_row)
    RL = P.rows_layout_l(n_out, drow)
    b = unpack_tile_vector(blob[:4 * n_out], RL, drow)
    return unpack_frags(blob[4 * n_out:], RL, kord_groups32(n_in)), b


def unpack_dec_project(blob, n_in=64, d_row="gfx12"):
    """(W [4, 32, n_in], b [4, 32]) from a dec0.project blob."""
    drow = P.resolve_d_row(d_row)
    RE = P.rows_expand(32, drow)
    b = unpack_tile_vector(blob[:4 * 128], RE, drow)
    W = unpack_frags(blob[4 * 128:], RE, kord_groups32(n_in))
    return W.reshape(4, 32, n_in), b.reshape(4, 32)
