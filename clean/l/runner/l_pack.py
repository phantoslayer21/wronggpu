# SPDX-License-Identifier: MIT
"""Model L weight packing (issue #89): the device blobs of Model L's own launches, from the structure table
(clean/l/reference/table.py) and the logical tensors of the oracle (clean/l/reference/oracle.py).

Model L's encoder and decoder are run as a chain of small FP8-map kernels, each with a blob in a format Model M
already uses (clean/FORMATS.md):

| Blob | Format | Kernel input -> output |
|---|---|---|
| `enc0.values` | Model M's enc0 blob (clean/runner/enc0.py pack) with W = 0 and b = 0 | frame -> the quantised values as a token map [F/2][80] (space_to_depth order 20 p + i) |
| `enc0.<w>` per linear | merge / expand section format: bias tile vector (rows R_L(C_out)), then fragments [mt < C_out/16][kc < C_in/16], rows R_L(C_out), K order K_L(C_in) | FP8 map [C_in] -> FP8 map [C_out]; a `select` feeding the linear is folded in as zero weights outside its channels |
| `enc0.<prefix>` per block | Model M's Swin block section (packing.pack_block) at the block's width, padded to 256 | FP8 map -> FP8 map |
| `enc0.project` | bias tile vector (rows R_L(n_out)), then fragments [mt][kb], rows R_L(n_out), K order: groups of 32 channels, block 2i + s holds channel 32 i + 16 h + 8 s + j (dec0's order, clean/runner/dec0.py kord_decoder, over every group) | the concatenated maps -> the 64 FP8 features enc1 reads |
| `dec0.<prefix>` | Model M's Swin block section at the decoder block's width | dec1's channels of the block segment -> same width |
| `dec0` | Model M's dec0 blob (clean/runner/dec0.py pack) | as Model M |

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
        raise ValueError(f"project input of {n} channels: groups of 32 only")
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


def pack_block(tensors, prefix, C, heads, mlp, d_row="gfx12"):
    drow = P.resolve_d_row(d_row)
    p = {k[len(prefix):]: v for k, v in tensors.items() if k.startswith(prefix)}
    blob = P.pack_block(p, C, heads, mlp, drow)
    return blob + bytes((-len(blob)) % ALIGN)


def pack_values(tensors, d_row="gfx12"):
    """Model M's enc0 blob with zero projection: only the dither table and the two scalars are used."""
    t = {"W": np.zeros((64, 80), np.float32), "b": np.zeros(64, np.float32),
         "T": tensors.get("T", np.zeros((8, 16, 16), np.float32)),
         "sigma1": tensors.get("sigma1", np.float32(0)), "sigma2": tensors.get("sigma2", np.float32(0))}
    return _enc0.pack(t, d_row)


# ---- the encoder's launch list --------------------------------------------------------------------------------------

def encoder_launches(t=_table.TABLE):
    """The kernels of the table's encoder, in order: [{"kind", "blob", "src", "dst", ...}]. kinds: "values" (quant
    + space_to_depth into the raw map), "linear", "swin", "project". A select is folded into the linear it feeds."""
    ops = t["enc_ops"]
    plan = _table.encoder_plan(t)
    chans = {"values": 20}
    launches = []
    selects = {}
    i = 0
    while i < len(ops):
        op = ops[i]
        k = op["op"]
        if k == "quant":
            nxt = ops[i + 1] if i + 1 < len(ops) else None
            if not nxt or nxt["op"] != "space_to_depth" or nxt["src"] != op["dst"]:
                raise ValueError("the encoder's kernels need quant directly followed by its space_to_depth")
            if op["mode"] != "dither":
                raise ValueError("the encoder's value kernel quantises with the dither table")
            launches.append({"kind": "values", "blob": "enc0.values", "dst": nxt["dst"]})
            chans[nxt["dst"]] = 80
            i += 2
            continue
        if k == "select":
            selects[op["dst"]] = (op["src"], op["lo"], op["hi"])
            chans[op["dst"]] = op["hi"] - op["lo"]
        elif k == "linear":
            src, lo = op["src"], 0
            if src in selects:
                src, lo, _ = selects[src]
            launches.append({"kind": "linear", "blob": "enc0." + op["w"], "src": src, "dst": op["dst"],
                             "C_in": chans[src], "C_out": op["n_out"], "lo": lo, "w": op["w"], "b": op["b"]})
            chans[op["dst"]] = op["n_out"]
        elif k == "swin":
            launches.append({"kind": "swin", "blob": "enc0." + op["prefix"].rstrip("."), "src": op["src"],
                             "dst": op["dst"], "prefix": op["prefix"], "C": op["C"], "heads": op["heads"],
                             "mlp": op["mlp"], "offsets": op["offsets"]})
        elif k == "concat":
            chans[op["dst"]] = plan[i][2]
            launches.append({"kind": "concat", "srcs": op["srcs"], "dst": op["dst"], "pad_to": plan[i][2]})
        elif k == "project":
            launches.append({"kind": "project", "blob": "enc0.project", "src": op["src"], "groups": op["groups"]})
        else:
            raise ValueError(f"no kernel for encoder op {k!r}")
        if k != "select":
            selects.pop(op.get("dst"), None)          # the name now holds this op's output
        i += 1
    return launches


def pack_model_l(params, t=_table.TABLE, d_row="gfx12"):
    """{blob name: bytes} of Model L's own launches (the shared Swin stages: packing.pack_stage)."""
    enc = params["enc0"]
    out = {}
    for ln in encoder_launches(t):
        if ln["kind"] == "values":
            out[ln["blob"]] = pack_values(enc, d_row)
        elif ln["kind"] == "linear":
            out[ln["blob"]] = pack_linear(enc[ln["w"]], enc[ln["b"]], ln["C_in"], ln["C_out"], ln["lo"], d_row)
        elif ln["kind"] == "swin":
            out[ln["blob"]] = pack_block(enc, ln["prefix"], ln["C"], ln["heads"], ln["mlp"], d_row)
        elif ln["kind"] == "project":
            op = t["enc_ops"][-1]
            out[ln["blob"]] = pack_project(enc[op["w"]], enc[op["b"]], ln["groups"], d_row)
    dec = params["dec0"]
    if any(s[0] == "block" for s in t["dec_connect"]):
        b = t["dec_block"]
        out["dec0." + b["prefix"].rstrip(".")] = pack_block(dec, b["prefix"], b["C"], b["heads"], b["mlp"], d_row)
    out["dec0"] = _dec0.pack({k: dec[k] for k in ("W", "b", "S", "T")}, d_row)
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
