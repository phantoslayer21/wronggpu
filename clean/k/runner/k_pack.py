# SPDX-License-Identifier: MIT
"""Model K weight packing (layouts: clean/k/FORMATS.md). The converter (spec 48) will produce the same bytes.

All parameters are FP16. A weight matrix W [N, K] ("output index first", as spec 90) becomes WMMA A-operand tiles of
16 x 16: tile (nt, kt) holds W[16 nt .. 16 nt + 16, 16 kt .. 16 kt + 16] as 32 lanes x 8 halfs, lane l = 16 h + r
holding A row r, inputs 16 kt + 8 h + 0..7. A row r is the output channel that the accumulator layout puts into
element v of lane half h' with r = d_row[h'][v]; that element must hold channel 16 nt + 8 h' + v, so the kernels can
store 8 contiguous channels per lane. gfx12 (measured on the RX 9070 XT, results/2026-10-05-swin-selftest.md):
r = v + 8 h'. N is padded with zero rows to a multiple of 16 (the head's 40 outputs -> 48).

Blob kinds (every section starts at a multiple of 256 bytes):
- linear: tiles [nt][kt] | bias [N padded] (zeros when there is none)
- block (spec 41): g1 [C] | g2 [C] | qkv tiles | mix [heads][64][64] | Wo tiles [C/16][D/16] | bo [C] |
  W1 tiles [4C/16][C/16] | b1 [4C] | W2 tiles [4C/16][C/16] (input-tile major) | b2 [C]
  qkv tiles per head h: softmax Wq_h, Wk_h, Wv_h; fixed Wv_h; each [2][C/16] tiles. mix: B_h (softmax) or M_h
  (fixed) as [t][s]. The block kernel uses the Wv tiles as B operands, which needs the gfx12 layout (checked).
- stage: e0: embed | block | merge;  e1..e4: block | merge;  m: block;  d4..d1: expand | block;
  d0: expand | block | head.
"""
import json
import os

import numpy as np

from clean.k.reference.layout import HEAD_W, STAGE
from clean.k.reference.params import normalize

ALIGN = 256
GFX12 = [list(range(8)), list(range(8, 16))]


def resolve_d_row(d_row="gfx12"):
    """d_row as [[8 ints], [8 ints]]: "gfx12", a JSON string, or a nested list. None: WG_D_ROW (env) or gfx12."""
    if d_row is None:
        d_row = os.environ.get("WG_D_ROW", "gfx12")
    if isinstance(d_row, str):
        if d_row == "gfx12":
            return [r[:] for r in GFX12]
        d_row = json.loads(d_row)
    d = [list(map(int, r)) for r in d_row]
    if len(d) != 2 or any(len(r) != 8 for r in d) or sorted(d[0] + d[1]) != list(range(16)):
        raise ValueError(f"bad d_row {d_row!r}")
    return d


def f16_exact(a, what="parameters"):
    a = np.asarray(a, dtype=np.float32)
    h = a.astype(np.float16)
    if not np.array_equal(h.astype(np.float32), a, equal_nan=True):
        raise ValueError(f"{what} must be exactly representable in FP16")
    return h


def pad16(n):
    return -(-n // 16) * 16


def _rowsrc(d_row):
    d = resolve_d_row(d_row)
    rowsrc = np.empty(16, dtype=np.int64)  # physical A row r <- logical row 8 h' + v
    for hp in range(2):
        for v in range(8):
            rowsrc[d[hp][v]] = 8 * hp + v
    return rowsrc


def pack_tiles(W, d_row="gfx12", order="nk"):
    """W [N, K] (FP16-exact; N padded to 16 with zero rows) -> FP16 tiles [NT, KT, 32, 8] ("nk") or [KT, NT, 32, 8]."""
    W = f16_exact(W)
    n, k = W.shape
    if k % 16:
        raise ValueError(f"matrix {W.shape}: the input side must be a multiple of 16")
    if n % 16:
        W = np.concatenate([W, np.zeros((pad16(n) - n, k), np.float16)])
        n = W.shape[0]
    t = W.reshape(n // 16, 16, k // 16, 16).transpose(0, 2, 1, 3)[:, :, _rowsrc(d_row), :]  # [NT, KT, r, k]
    t = t.reshape(n // 16, k // 16, 16, 2, 8).transpose(0, 1, 3, 2, 4).reshape(n // 16, k // 16, 32, 8)
    if order == "kn":
        t = t.transpose(1, 0, 2, 3)
    return np.ascontiguousarray(t)


def unpack_tiles(t, d_row="gfx12", order="nk"):
    """Inverse of pack_tiles (tests; padded rows included)."""
    t = np.asarray(t, dtype=np.float16)
    if order == "kn":
        t = t.transpose(1, 0, 2, 3)
    nt, kt = t.shape[:2]
    a = t.reshape(nt, kt, 2, 16, 8).transpose(0, 1, 3, 2, 4).reshape(nt, kt, 16, 16)  # [NT, KT, r, k]
    W = np.empty_like(a)
    W[:, :, _rowsrc(d_row), :] = a
    return W.transpose(0, 2, 1, 3).reshape(nt * 16, kt * 16)


class Blob:
    """Concatenation of 256-byte aligned sections; offsets by name."""

    def __init__(self):
        self.parts, self.offsets, self.size = [], {}, 0

    def add(self, name, data):
        b = data if isinstance(data, (bytes, bytearray)) else np.ascontiguousarray(data).tobytes()
        self.offsets[name] = self.size
        pad = (-len(b)) % ALIGN
        self.parts.append(bytes(b) + b"\0" * pad)
        self.size += len(b) + pad

    def bytes(self):
        return b"".join(self.parts)


def _al(n):
    return -(-n // ALIGN) * ALIGN


# ---- linear ---------------------------------------------------------------------------------------------------------

def linear_size(n, k):
    return _al(pad16(n) * k * 2) + _al(pad16(n) * 2)


def linear_sections(n, k):
    """{"w": offset, "b": offset} of a linear blob for W [n, k]."""
    return {"w": 0, "b": _al(pad16(n) * k * 2)}


def pack_linear(W, b=None, d_row="gfx12"):
    W = np.asarray(W)
    n = W.shape[0]
    bias = np.zeros(pad16(n), np.float16)
    if b is not None:
        bias[:n] = f16_exact(b)
    blob = Blob()
    blob.add("w", pack_tiles(W, d_row, "nk"))
    blob.add("b", bias)
    return blob.bytes()


# ---- block -------------------------------------------------------------------------------------------------------

def block_sections(C, heads, mix):
    """{name: offset} and "_size" of a block blob."""
    D, HID = HEAD_W * heads, 4 * C
    qkv = (3 if mix == "softmax" else 1) * heads * 32 * C * 2
    s, off = {}, 0
    for name, nbytes in (("g1", C * 2), ("g2", C * 2), ("qkv", qkv), ("mix", heads * 64 * 64 * 2),
                         ("wo", C * D * 2), ("bo", C * 2), ("w1", HID * C * 2), ("b1", HID * 2),
                         ("w2", C * HID * 2), ("b2", C * 2)):
        s[name] = off
        off += _al(nbytes)
    s["_size"] = off
    return s


def pack_block(p, C, heads, mix, d_row="gfx12"):
    """Spec 41 block parameters (canonical names, clean/k/reference/params.py; also g1 as gamma1 and B / M as
    [H, 8, 8, 8, 8]) -> blob bytes."""
    p = dict(p)
    if "g1" not in p and "gamma1" in p:
        p["g1"] = p["gamma1"]
    for nm in ("B", "M"):
        if nm in p and np.asarray(p[nm]).shape == (heads, 8, 8, 8, 8):
            p[nm] = np.asarray(p[nm]).reshape(heads, 64, 64)
    if resolve_d_row(d_row) != GFX12:
        raise ValueError("the model K block kernels need the gfx12 WMMA accumulator layout (d_row rows v + 8 h)")
    blob = Blob()
    blob.add("g1", f16_exact(p["g1"]))
    blob.add("g2", f16_exact(p["g2"]))
    names = ("Wq", "Wk", "Wv") if mix == "softmax" else ("Wv",)
    for nm in names:
        if np.asarray(p[nm]).shape != (heads, 32, C):
            raise ValueError(f"{nm} must be [{heads}, 32, {C}], got {np.asarray(p[nm]).shape}")
    qkv = [pack_tiles(np.asarray(p[nm])[h], d_row) for h in range(heads) for nm in names]
    blob.add("qkv", np.concatenate([t.reshape(-1) for t in qkv]))
    table = p["B"] if mix == "softmax" else p["M"]
    if np.asarray(table).shape != (heads, 64, 64):
        raise ValueError(f"{'B' if mix == 'softmax' else 'M'} must be [{heads}, 64, 64]")
    blob.add("mix", f16_exact(table))
    blob.add("wo", pack_tiles(p["Wo"], d_row))
    blob.add("bo", f16_exact(p["bo"]))
    blob.add("w1", pack_tiles(p["W1"], d_row))
    blob.add("b1", f16_exact(p["b1"]))
    blob.add("w2", pack_tiles(p["W2"], d_row, "kn"))
    blob.add("b2", f16_exact(p["b2"]))
    expect = block_sections(C, heads, mix)
    assert blob.offsets == {k: v for k, v in expect.items() if k != "_size"} and blob.size == expect["_size"]
    return blob.bytes()


# ---- stage -------------------------------------------------------------------------------------------------------

def stage_parts(st):
    """[(part, n, k)] in blob order; part in embed / expand / block / merge / head (n, k: linear shape)."""
    parts = []
    if st.get("embed"):
        parts.append(("embed", 32, 16))
    if st["kind"] == "dec":
        parts.append(("expand", 4 * st["C"], st["C_prev"]))
    parts.append(("block", None, None))
    if st["kind"] == "enc":
        parts.append(("merge", st["C_next"], 4 * st["C"]))
    if st.get("head"):
        parts.append(("head", 40, st["C"]))
    return parts


def stage_sections(st):
    """{part: offset, "_size": total} of a stage blob."""
    s, off = {}, 0
    for part, n, k in stage_parts(st):
        s[part] = off
        off += block_sections(st["C"], st["heads"], st["mix"])["_size"] if part == "block" else linear_size(n, k)
    s["_size"] = off
    return s


LINEAR_TENSORS = {"embed": ("We", "be"), "expand": ("Wx", "bx"), "merge": ("Wm", "bm"), "head": ("Wh", "bh")}


def pack_stage(stage, p, d_row="gfx12"):
    """A model K stage (canonical or spec 48 tensors, clean/k/reference/params.py) -> blob bytes."""
    st = STAGE[stage]
    p = normalize(stage, p)
    out = b""
    for part, n, k in stage_parts(st):
        if part == "block":
            out += pack_block(p, st["C"], st["heads"], st["mix"], d_row)
        else:
            wn, bn = LINEAR_TENSORS[part]
            W = np.asarray(p[wn])
            if W.shape != (n, k):
                raise ValueError(f"{stage}: {wn} must be [{n}, {k}], got {W.shape}")
            out += pack_linear(W, p.get(bn), d_row)
    assert len(out) == stage_sections(st)["_size"]
    return out
