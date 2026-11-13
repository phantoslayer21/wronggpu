# SPDX-License-Identifier: MIT
"""Model L structure table (issue #89): the single source of Model L's shape for the oracle, the packers, the
runtime and the tests.

Every consumer interprets this table instead of hard-coding L's structure, so a provisional entry can be changed
without touching code. Entries listed in PROVISIONAL are pending confirmation; each carries a short key (I5 ...).

Parameters are grouped by the converter map's stages (data/preset-L.v1.map.json): "enc0", "enc0_block", the ten
shared stages, "dec0_block", "dec0", "post". SHARED_STAGES run with Model M's code and stage table
(clean/reference/swin.py STAGES) and L's own weights.
"""
import copy

from clean.reference import swin as _swin

# ---- stages shared with Model M -------------------------------------------------------------------------------------

SHARED_STAGES = ("enc1", "enc2", "b0", "b1", "b2", "b3", "b4", "b5", "dec2", "dec1")

# ---- block constants ------------------------------------------------------------------------------------------------

# Model M's block constants (clean/reference/swin.py), the default for every L block.
M_BLOCK_CONSTS = {"tau": _swin.TAU, "cap": _swin.CAP, "eps_num": 2.0 ** -13, "gelu": "exact",
                  "window": _swin.WIN, "head_dim": _swin.HEAD_DIM}

# The value positions of the coefficient decoder (the map's perm "dec0_values"): value k of a render pixel is
# channel DEC0_VALUES[k] of its 32.
DEC0_VALUES = (0, 1, 10, 11, 12, 13, 14, 15, 16, 2, 3, 4, 5, 6, 7, 8, 9)

# ---- the table ------------------------------------------------------------------------------------------------------

C32_BLOCK = {"C": 32, "heads": 2, "mlp": 128}

TABLE = {
    # Input encoder: ops applied in order. Each op reads and writes named maps; "values" is the per-pixel input
    # (clean/reference/encoder.py pixel_values: [fh, fw, 20] on the padded grid F). A map's level follows from its
    # size: the pixel grid F, or the token grid F/2 after space_to_depth (2 x 2 pixels, sub-pixel p = dx + 2 dy,
    # channel = n p + i). Tensors are read from the stage named by "params" (default "enc0"). A linear's "acc" is the
    # rounding of bias + products before "out_round" ("relu_q8" = q8(max(0, .))). The last op is "project" (the 64 FP8
    # features enc1 reads): running sum s = b, s = RN16(s + group) per channel group, clamp, FP8. out_round "none"
    # after acc "f16" keeps the FP16 value (the block then starts from FP16, not FP8). A swin op's "border" names how
    # window members outside the grid get their values: "mirror" (clean/reference/swin.py, about the map size),
    # "input" (the per-pixel ops before the block are evaluated at those positions: the 20 values at the mirrored
    # render position, the dither at the unmirrored one) or "coarse" (the token-grid mirror, sub-pixel kept).
    "enc_ops": (
        {"op": "quant", "mode": "dither", "src": "values", "dst": "q"},
        {"op": "linear", "src": "q", "dst": "x", "w": "Win1", "b": "bin1", "n_in": 20, "n_out": 32,
         "acc": "f16", "out_round": "relu_q8"},
        {"op": "linear", "src": "x", "dst": "x", "w": "Win2", "b": "bin2", "n_in": 32, "n_out": 32,
         "acc": "f16", "out_round": "none"},
        {"op": "swin", "src": "x", "dst": "x", "params": "enc0_block", **C32_BLOCK, "offsets": "first",
         "border": "input", "out_round": "q8"},                                                           # I6
        {"op": "space_to_depth", "src": "x", "dst": "z"},
        {"op": "project", "src": "z", "w": "W", "b": "b", "groups": (32, 32, 32, 32), "n_out": 64,
         "clamp": 6.28125},
    ),
    # Coefficient decoder: ops on dec1's map ("features", [fh/2, fw/2, 64]). "dec_project": rows W [4, 32, 64] with
    # Model M's dec0 rounding (RN16 after each group of 32 input channels, then FP8), output [fh/2, fw/2, 4 x 32]
    # (sub-pixel s = dx + 2 dy, channel 32 s + c); "depth_to_space" makes it a pixel map [fh, fw, 32]; "pick" takes
    # the value positions of the coefficient functions, which then run as Model M's (clean/reference/decoder.py).
    "dec_ops": (
        {"op": "dec_project", "src": "features", "dst": "y", "w": "W", "b": "b", "groups": (32, 32),
         "out_round": "q8"},                                                                              # I13
        {"op": "depth_to_space", "src": "y", "dst": "p"},
        {"op": "swin", "src": "p", "dst": "p", "params": "dec0_block", **C32_BLOCK, "offsets": "second",
         "border": "coarse", "out_round": "q8"},                                                          # I6, I14
        {"op": "pick", "src": "p", "values": DEC0_VALUES},                                                # I14
    ),
    "dec_pixel": "m",                                                                                     # I8
    "post_variant": "unfolded",                                                                           # I8
    # Reconstruction variants and the rule that picks one per frame from the render and output sizes ("rule" names
    # the comparison, "ratio" its factor). Only the 3/2 form (D = floor((3 O + 1) / 2)) exists in WrongGPU so far.
    "recon": {"rule": "both_axes_at_least", "ratio": 3, "variants": ("3/2", "3/1")},                    # I7
    # Window offsets (ox, oy) of an L block for frame parity 0 and 1, in pixels; the blocks run on the padded grid F
    # and keep every grid position's output.
    "window_offsets": {"first": ((2, 2), (0, 2)), "second": ((0, 0), (2, 0))},
    # Per-block constants; a block's own entry (key = its params stage) wins over "default".
    "consts": {"default": dict(M_BLOCK_CONSTS)},                                                          # I6
}

PROVISIONAL = {
    "I6": "consts of the width-32 blocks",
    "I7": "recon: the 3/1 form and the exact rule",
    "I8": "dec_pixel (coefficient functions as Model M's), post_variant",
    "I13": "dec_project rounding",
    "I14": "pick value positions and the rounding of the decoder block's output",
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


BORDERS = ("mirror", "input", "coarse")


def block_halo(t, kind):
    """Pixels beyond the grid that a block's windows reach (grid sides multiples of the window): the largest offset."""
    return max(max(o) for o in t["window_offsets"][kind])


# ---- parameter shapes -----------------------------------------------------------------------------------------------

def block_shapes(C, heads, mlp, head_dim=32, window=4):
    t = window * window
    return {"gamma1": (C,), "gamma2": (C,), "Wqk": (heads, head_dim, C), "Wv": (heads, head_dim, C),
            "B": (heads, t, t), "Wp": (C, heads * head_dim), "bp": (C,), "W1": (mlp, C), "b1": (mlp,),
            "W2": (C, mlp), "b2": (C,)}


def param_shapes(t=TABLE):
    """{stage: {tensor: shape}} of Model L's own stages (enc0, its block, dec0, its block, post)."""
    out = {"enc0": {}, "dec0": {"S": (), "T": ()}, "post": {"s": ()}}
    for op in t["enc_ops"]:
        st = out.setdefault(op.get("params", "enc0"), {})
        if op["op"] == "quant":
            st.update({"sigma1": (), "sigma2": ()})
            if op["mode"] == "dither":
                st["T"] = (8, 16, 16)
        elif op["op"] == "linear":
            st.update({op["w"]: (op["n_out"], op["n_in"]), op["b"]: (op["n_out"],)})
        elif op["op"] == "swin":
            st.update(block_shapes(op["C"], op["heads"], op["mlp"]))
        elif op["op"] == "project":
            st.update({op["w"]: (op["n_out"], sum(op["groups"])), op["b"]: (op["n_out"],)})
    for op in t["dec_ops"]:
        st = out.setdefault(op.get("params", "dec0"), {})
        if op["op"] == "dec_project":
            st.update({op["w"]: (4, 32, sum(op["groups"])), op["b"]: (4, 32)})
        elif op["op"] == "swin":
            st.update(block_shapes(op["C"], op["heads"], op["mlp"]))
    return out


def encoder_plan(t=TABLE):
    """(op, level, channels) after each encoder op, from the op list alone."""
    return _plan(t["enc_ops"], {"values": ("pixel", 20)}, "encoder", last="project")


def decoder_plan(t=TABLE):
    """(op, level, channels) after each decoder op, from the op list alone."""
    return _plan(t["dec_ops"], {"features": ("token", 64)}, "decoder", last="pick")


def _plan(ops, maps, what, last):
    plan = []
    for op in ops:
        k = op["op"]
        src = maps.get(op.get("src"))
        if k != "concat" and src is None:
            raise ValueError(f"{what}: {k} reads unknown map {op.get('src')!r}")
        if k == "quant":
            out = src
        elif k == "space_to_depth":
            if src[0] != "pixel":
                raise ValueError("space_to_depth needs a pixel-level map")
            out = ("token", 4 * src[1])
        elif k == "depth_to_space":
            if src[0] != "token" or src[1] % 4:
                raise ValueError("depth_to_space needs a token-level map of 4 n channels")
            out = ("pixel", src[1] // 4)
        elif k == "select":
            if not 0 <= op["lo"] < op["hi"] <= src[1]:
                raise ValueError(f"select {op['lo']}..{op['hi']} outside {src[1]} channels")
            out = (src[0], op["hi"] - op["lo"])
        elif k == "linear":
            if src[1] != op["n_in"]:
                raise ValueError(f"linear {op['w']}: input has {src[1]} channels, op wants {op['n_in']}")
            out = (src[0], op["n_out"])
        elif k == "swin":
            if src[1] != op["C"]:
                raise ValueError(f"swin: input has {src[1]} channels, block width {op['C']}")
            border = op.get("border", "mirror")
            if border not in BORDERS:
                raise ValueError(f"swin: unknown border {border!r}")
            if border != "mirror" and src[0] != "pixel":
                raise ValueError(f"swin: border {border!r} needs a pixel-level map")
            if border == "input" and any(p[0] not in ("quant", "linear", "select", "concat") for p in plan):
                raise ValueError("swin: border 'input' needs only per-pixel ops before the block")
            out = src
        elif k == "concat":
            parts = [maps[s] for s in op["srcs"]]
            if len({p[0] for p in parts}) != 1:
                raise ValueError("concat of maps at different levels")
            c = sum(p[1] for p in parts)
            pad = op.get("pad_to", c)
            if pad < c:
                raise ValueError("concat: pad_to below the channel count")
            out = (parts[0][0], pad)
        elif k == "project":
            if src[0] != "token" or src[1] != sum(op["groups"]):
                raise ValueError(f"project: needs a token-level map of {sum(op['groups'])} channels, got {src}")
            out = ("token", op["n_out"])
        elif k == "dec_project":
            if src[0] != "token" or src[1] != sum(op["groups"]):
                raise ValueError(f"dec_project: needs a token-level map of {sum(op['groups'])} channels, got {src}")
            out = ("token", 128)
        elif k == "pick":
            if src[0] != "pixel" or max(op["values"]) >= src[1]:
                raise ValueError("pick: needs a pixel-level map holding every value position")
            out = ("pixel", len(op["values"]))
        else:
            raise ValueError(f"unknown {what} op {k!r}")
        maps[op.get("dst", "out")] = out
        plan.append((k, out[0], out[1]))
    if not plan or plan[-1][0] != last:
        raise ValueError(f"the {what}'s last op must be {last}")
    return plan


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

    def ops_macs(ops, plan):
        total = 0
        for op, (_, lvl, _c) in zip(ops, plan):
            if op["op"] == "linear":
                total += op["n_in"] * op["n_out"] * n[lvl]
            elif op["op"] == "swin":
                total += block_macs_per_token(op["C"], op["heads"], op["mlp"]) * n[lvl]
            elif op["op"] == "project":
                total += sum(op["groups"]) * op["n_out"] * n["token"]
            elif op["op"] == "dec_project":
                total += sum(op["groups"]) * 128 * n["token"]
        return total

    enc = ops_macs(t["enc_ops"], encoder_plan(t))
    dec = ops_macs(t["dec_ops"], decoder_plan(t))
    L = sum(parts.values()) + enc + dec
    parts.update({"enc0.M": m_enc0, "dec0.M": m_dec0, "enc0.L": enc, "dec0.L": dec})
    return {"M": M, "L": L, "parts": parts}
