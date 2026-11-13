# SPDX-License-Identifier: MIT
"""Model L structure table (issue #89): the single source of Model L's shape for the oracle, the packers, the
runtime and the tests.

Every consumer interprets this table instead of hard-coding L's structure, so a provisional entry can be changed
without touching code. Entries listed in PROVISIONAL are pending confirmation; each carries a short key (I1 ...).

SHARED_STAGES run with Model M's code and stage table (clean/reference/swin.py STAGES) and L's own weights.
"""
import copy

from clean.reference import swin as _swin

# ---- stages shared with Model M -------------------------------------------------------------------------------------

SHARED_STAGES = ("enc1", "enc2", "b0", "b1", "b2", "b3", "b4", "b5", "dec2", "dec1")

# ---- block constants ------------------------------------------------------------------------------------------------

# Model M's block constants (clean/reference/swin.py), the default for every L block.
M_BLOCK_CONSTS = {"tau": _swin.TAU, "cap": _swin.CAP, "eps_num": 2.0 ** -13, "gelu": "exact",
                  "window": _swin.WIN, "head_dim": _swin.HEAD_DIM}

# ---- the table ------------------------------------------------------------------------------------------------------

C32_BLOCK = {"C": 32, "heads": 2, "mlp": 128}

TABLE = {
    # Input encoder: ops applied in order. Each op reads and writes named maps; "values" is the per-pixel input
    # (clean/reference/encoder.py pixel_values: [fh, fw, 20] on the padded grid F). A map's level follows from its
    # size: the pixel grid F, or the token grid F/2 after space_to_depth (2 x 2 pixels, sub-pixel p = dx + 2 dy,
    # channel = n * p + i). The last op must be "project" (the 64 FP8 features enc1 reads).
    "enc_ops": (
        {"op": "quant", "mode": "dither", "src": "values", "dst": "q"},                       # I5
        {"op": "space_to_depth", "src": "q", "dst": "raw"},                                   # I2
        {"op": "select", "src": "raw", "dst": "x", "lo": 0, "hi": 20},                        # I2
        {"op": "linear", "src": "x", "dst": "x", "w": "Wa", "b": "ba", "n_in": 20, "n_out": 32,
         "out_round": "q8"},                                                                   # I1
        {"op": "linear", "src": "x", "dst": "x", "w": "Wb", "b": "bb", "n_in": 32, "n_out": 32,
         "out_round": "q8"},                                                                   # I1
        {"op": "swin", "src": "x", "dst": "x", "prefix": "blk.", **C32_BLOCK, "offsets": "first",
         "out_round": "q8"},                                                                   # I1, I6, I10
        {"op": "concat", "srcs": ("raw", "x"), "dst": "z", "pad_to": 128},                    # I2
        {"op": "project", "src": "z", "w": "W", "b": "b", "groups": (32, 32, 32, 32), "n_out": 64},  # I2
    ),
    # Coefficient decoder: dec1's 64 channels split into segments; "block" segments go through one block, "pass"
    # segments are kept; the result (same channel order) feeds Model M's decoder (clean/reference/decoder.py).
    "dec_connect": (("block", 0, 32), ("pass", 32, 64)),                                     # I4
    "dec_block": {"prefix": "blk.", **C32_BLOCK, "offsets": "second", "out_round": "q8"},  # I6, I10
    "dec_block_level": "token",                                                              # I3
    "dec_pixel": "m",                                                                        # I8
    "post_variant": "unfolded",                                                              # I8
    "recon_ratio": (3, 2),                                                                   # I7
    # Window offsets (ox, oy) of an L block for frame parity 0 and 1.
    "window_offsets": {"first": ((2, 2), (0, 2)), "second": ((0, 0), (2, 0))},              # I10
    # Per-block constants; a block's own entry wins over "default".
    "consts": {"default": dict(M_BLOCK_CONSTS)},                                             # I6
    # Logical tensor name -> converter map name, per stage ("enc0", "dec0"); identity when absent.
    "map_names": {},                                                                         # I9
}

PROVISIONAL = {
    "I1": "enc_ops: order and kind of the ops",
    "I2": "enc_ops: levels, select / concat segments, project groups",
    "I3": "dec_block_level",
    "I4": "dec_connect",
    "I5": "enc_ops: quant mode",
    "I6": "consts",
    "I7": "recon_ratio",
    "I8": "dec_pixel, post_variant",
    "I9": "map_names",
    "I10": "window_offsets",
}


def table(**overrides):
    """A deep copy of TABLE with top-level entries replaced (tests and experiments)."""
    t = copy.deepcopy(TABLE)
    for k, v in overrides.items():
        if k not in t:
            raise KeyError(f"unknown table entry {k!r}")
        t[k] = v
    return t


def block_consts(t, name):
    c = dict(t["consts"]["default"])
    c.update(t["consts"].get(name, {}))
    return c


def block_offset(t, kind, parity):
    return tuple(t["window_offsets"][kind][int(parity) & 1])


def map_name(t, stage, logical):
    return t["map_names"].get(stage, {}).get(logical, logical)


# ---- parameter shapes -----------------------------------------------------------------------------------------------

def block_shapes(C, heads, mlp, head_dim=32, window=4):
    t = window * window
    return {"gamma1": (C,), "gamma2": (C,), "Wqk": (heads, head_dim, C), "Wv": (heads, head_dim, C),
            "B": (heads, t, t), "Wp": (C, heads * head_dim), "bp": (C,), "W1": (mlp, C), "b1": (mlp,),
            "W2": (C, mlp), "b2": (C,)}


def encoder_shapes(t=TABLE):
    """Tensor shapes of the L encoder stage, logical names."""
    shapes = {}
    for op in t["enc_ops"]:
        if op["op"] == "quant":
            shapes.update({"sigma1": (), "sigma2": ()})
            if op["mode"] == "dither":
                shapes["T"] = (8, 16, 16)
        elif op["op"] == "linear":
            shapes.update({op["w"]: (op["n_out"], op["n_in"]), op["b"]: (op["n_out"],)})
        elif op["op"] == "swin":
            for k, s in block_shapes(op["C"], op["heads"], op["mlp"]).items():
                shapes[op["prefix"] + k] = s
        elif op["op"] == "project":
            shapes.update({op["w"]: (op["n_out"], sum(op["groups"])), op["b"]: (op["n_out"],)})
    return shapes


def encoder_plan(t=TABLE):
    """(level, channels) of every named map after each op, from the op list alone: [(op, level, channels), ...]."""
    maps = {"values": ("pixel", 20)}
    plan = []
    for op in t["enc_ops"]:
        k = op["op"]
        if k == "quant":
            out = maps[op["src"]]
        elif k == "space_to_depth":
            lvl, c = maps[op["src"]]
            if lvl != "pixel":
                raise ValueError("space_to_depth needs a pixel-level map")
            out = ("token", 4 * c)
        elif k == "select":
            lvl, c = maps[op["src"]]
            if not 0 <= op["lo"] < op["hi"] <= c:
                raise ValueError(f"select {op['lo']}..{op['hi']} outside {c} channels")
            out = (lvl, op["hi"] - op["lo"])
        elif k == "linear":
            lvl, c = maps[op["src"]]
            if c != op["n_in"]:
                raise ValueError(f"linear {op['w']}: input has {c} channels, op wants {op['n_in']}")
            out = (lvl, op["n_out"])
        elif k == "swin":
            lvl, c = maps[op["src"]]
            if c != op["C"]:
                raise ValueError(f"swin: input has {c} channels, block width {op['C']}")
            out = (lvl, c)
        elif k == "concat":
            lv = {maps[s][0] for s in op["srcs"]}
            if len(lv) != 1:
                raise ValueError("concat of maps at different levels")
            c = sum(maps[s][1] for s in op["srcs"])
            pad = op.get("pad_to", c)
            if pad < c:
                raise ValueError("concat: pad_to below the channel count")
            out = (lv.pop(), pad)
        elif k == "project":
            lvl, c = maps[op["src"]]
            if lvl != "token" or c != sum(op["groups"]):
                raise ValueError(f"project: needs a token-level map of {sum(op['groups'])} channels, got {lvl} {c}")
            out = ("token", op["n_out"])
        else:
            raise ValueError(f"unknown encoder op {k!r}")
        maps[op.get("dst", "features")] = out
        plan.append((k, out[0], out[1]))
    if not plan or plan[-1][0] != "project":
        raise ValueError("the encoder's last op must be project")
    return plan


def decoder_shapes(t=TABLE):
    shapes = {"W": (68, 64), "b": (68,), "S": (), "T": ()}
    if any(seg[0] == "block" for seg in t["dec_connect"]):
        b = t["dec_block"]
        for k, s in block_shapes(b["C"], b["heads"], b["mlp"]).items():
            shapes[b["prefix"] + k] = s
    return shapes


# ---- work per frame (computed, not measured) ------------------------------------------------------------------------

def block_macs_per_token(C, heads, mlp, head_dim=32, window=4):
    """Multiply-adds of the matrix parts of one block per token: QKV, scores + weighted sum, output projection, MLP."""
    t = window * window
    return 2 * head_dim * heads * C + 2 * t * head_dim * heads + head_dim * heads * C + 2 * mlp * C


def frame_macs(t, render_w, render_h):
    """Matrix multiply-adds of one frame for Model L and for Model M at the same render size (no window padding).
    Returns {"M": ..., "L": ..., "parts": {...}}."""
    fw, fh = -(-render_w // 32) * 32, -(-render_h // 32) * 32
    n = {"pixel": fw * fh, "token": (fw // 2) * (fh // 2)}
    n2, n3 = n["token"] // 4, n["token"] // 16
    S = _swin.STAGES
    parts = {}
    for s in SHARED_STAGES:
        cfg = S[s]
        tokens = {64: n["token"], 96: n2, 128: n3}[cfg["C"]]
        m = block_macs_per_token(cfg["C"], cfg["heads"], cfg["mlp"]) * tokens
        if cfg["kind"] == "enc":
            m += 4 * cfg["C"] * cfg["C_out"] * (tokens // 4)
        if cfg["kind"] == "dec":
            m += cfg["C_low"] * 4 * cfg["C"] * (tokens // 4)
        parts[s] = m
    m_enc0 = 80 * 64 * n["token"]
    m_dec0 = 64 * 68 * n["token"]
    M = sum(parts.values()) + m_enc0 + m_dec0
    enc = 0
    for op, (_, lvl, _c) in zip(t["enc_ops"], encoder_plan(t)):
        if op["op"] == "linear":
            enc += op["n_in"] * op["n_out"] * n[lvl]
        elif op["op"] == "swin":
            enc += block_macs_per_token(op["C"], op["heads"], op["mlp"]) * n[lvl]
        elif op["op"] == "project":
            enc += sum(op["groups"]) * op["n_out"] * n["token"]
    dec = m_dec0
    if any(seg[0] == "block" for seg in t["dec_connect"]):
        b = t["dec_block"]
        lvl = n["token"] if t["dec_block_level"] == "token" else render_w * render_h
        dec += block_macs_per_token(b["C"], b["heads"], b["mlp"]) * lvl
    L = sum(parts.values()) + enc + dec
    parts.update({"enc0.M": m_enc0, "dec0.M": m_dec0, "enc0.L": enc, "dec0.L": dec})
    return {"M": M, "L": L, "parts": parts}
