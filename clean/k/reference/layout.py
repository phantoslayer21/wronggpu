# SPDX-License-Identifier: MIT
"""Model K layout: stage table (spec 40 draft 4), window offsets and base size (spec 44), level pyramid, buffer plan
and the 8x8 window partition (spec 41 "Windows", task K3).

Plain integer bookkeeping shared by the NumPy reference (network.py), the runner (clean/k/runner) and, mirrored in
C++, the runtime layer table (clean/k/runtime/model_k.cpp; tests/test_k_runtime_table.py checks they agree).
The base-size factor s is a parameter (spec 44 draft 3: s = 1/2, the default; tests also use other values).
"""
import numpy as np

WIN = 8  # window side (spec 41: 8x8 windows, 64 tokens)
LEVELS = 6  # levels 0 .. 5
HEAD_W = 32  # every head has width 32 (spec 40)
INPUT_C = 16  # network input channels (spec 42)
EMBED_C = 32  # embedding 16 -> 32 (spec 41)
OUTPUT_C = 40  # output head 32 -> 40 (spec 41)
MLP_MULT = 4  # MLP hidden width 4 C (spec 41)

# The eleven stages of spec 40 draft 4 in run order. Every stage runs one block at `level` with width C. Encoder
# stages ("enc") keep their block output as skip map S<level> and merge 2x2 to C_next channels at level + 1. Decoder
# stages ("dec") first expand 2x2 from C_prev channels at level + 1 and add S<level>. e0 embeds the 16-channel input
# first, d0 applies the output head after its block. mix: "fixed" or "softmax" token mixing (spec 41).
STAGES = [
    dict(name="e0", kind="enc", level=0, C=32, heads=2, mix="fixed", C_next=64, embed=True),
    dict(name="e1", kind="enc", level=1, C=64, heads=2, mix="fixed", C_next=64),
    dict(name="e2", kind="enc", level=2, C=64, heads=2, mix="softmax", C_next=96),
    dict(name="e3", kind="enc", level=3, C=96, heads=4, mix="softmax", C_next=128),
    dict(name="e4", kind="enc", level=4, C=128, heads=4, mix="softmax", C_next=160),
    dict(name="m", kind="mid", level=5, C=160, heads=8, mix="softmax"),
    dict(name="d4", kind="dec", level=4, C=128, heads=4, mix="softmax", C_prev=160),
    dict(name="d3", kind="dec", level=3, C=96, heads=4, mix="softmax", C_prev=128),
    dict(name="d2", kind="dec", level=2, C=64, heads=2, mix="softmax", C_prev=96),
    dict(name="d1", kind="dec", level=1, C=64, heads=2, mix="softmax", C_prev=64),
    dict(name="d0", kind="dec", level=0, C=32, heads=2, mix="fixed", C_prev=64, head=True),
]
for _st in STAGES:
    _st["D"] = HEAD_W * _st["heads"]
    _st["HID"] = MLP_MULT * _st["C"]
    _st["skip"] = f"S{_st['level']}" if _st["kind"] in ("enc", "dec") else None
STAGE = {s["name"]: s for s in STAGES}
STAGE_NAMES = [s["name"] for s in STAGES]
SKIPS = ["S0", "S1", "S2", "S3", "S4"]

# Spec 44 draft 2: the window-offset table O of stage m (8 rows of (ox, oy), row f mod 8) is part of the converted
# model (spec 48 output group "net", tensor window_offsets); it must not be written into code or data we ship. Tests
# with synthetic models use this made-up table instead (any values 0..7 are allowed there).
SYNTHETIC_OFFSETS = [(0, 7), (1, 6), (2, 5), (3, 4), (4, 3), (5, 2), (6, 1), (7, 0)]


def cdiv(a, b):
    return -(-a // b)


# Spec 44's base-size factor: s = 1/2 (spec 44 draft 3). It is the default everywhere; every function takes s explicitly
# for tests (spec 49 draft 4's test sequences use s = 1).
DEFAULT_S = 0.5


def base_size(ow, oh, s=None):
    """Spec 44 draft 3: Wb = max(64 ceil(ceil(s W_out) / 64), 512) / 2 (same for Hb), ceil(s W_out) in FP32; s defaults
    to DEFAULT_S (the spec's s = 1/2)."""
    s = DEFAULT_S if s is None else s

    def one(n):
        return max(64 * cdiv(int(np.ceil(np.float32(s) * np.float32(n))), 64), 512) // 2
    return one(ow), one(oh)


def level_size(wb, hb, level):
    """Level L of the pyramid: ceil(Wb / 2^L) x ceil(Hb / 2^L) tokens (any Wb, Hb >= 1)."""
    return cdiv(int(wb), 1 << level), cdiv(int(hb), 1 << level)


def pyramid(wb, hb):
    return [level_size(wb, hb, L) for L in range(LEVELS)]


def check_offset_table(table):
    """The window-offset table as 8 (ox, oy) int pairs in 0..7, or ValueError."""
    t = [tuple(int(v) for v in row) for row in (table.tolist() if hasattr(table, "tolist") else table)]
    if len(t) != 8 or any(len(r) != 2 or not all(0 <= v < 8 for v in r) for r in t):
        raise ValueError(f"window-offset table must be 8 rows of (ox, oy) in 0..7, got {table!r}")
    return t


def stage_offsets(f, table):
    """Spec 44: {stage: (ox, oy)} for frame index f (f = 0 on a reset frame; only f mod 8 matters). table: the
    window-offset table of the converted model (spec 48 net/window_offsets), or SYNTHETIC_OFFSETS in tests."""
    oxm, oym = check_offset_table(table)[f % 8]
    o45 = ((2 * oxm - 4) % 8, (2 * oym - 4) % 8)
    off = {name: (4, 4) for name in STAGE_NAMES}
    off["m"] = (oxm, oym)
    off["e4"] = off["d4"] = o45
    return off


def linear_shapes():
    """Every (kind, K, N) of model K's linear layers (spec 41), kind in plain / merge / expand. N = 40 (head) is
    stored from 48 padded rows."""
    shapes = {("plain", INPUT_C, EMBED_C), ("plain", EMBED_C, OUTPUT_C)}
    for st in STAGES:
        C, D, HID = st["C"], st["D"], st["HID"]
        qkv = D if st["mix"] == "fixed" else 3 * D
        shapes |= {("plain", C, qkv), ("plain", D, C), ("plain", C, HID), ("plain", HID, C)}
        if st["kind"] == "enc":
            shapes.add(("merge", 4 * C, st["C_next"]))
        if st["kind"] == "dec":
            shapes.add(("expand", st["C_prev"], 4 * C))
    return sorted(shapes)


# ---- buffer plan --------------------------------------------------------------------------------------------------
#
# FP16 [H][W][C] maps of one frame. X0: the network input (level 0, 16 ch, spec 42). S0..S4: skip maps (encoder block
# outputs, kept until the decoder stage of the same level). A, B: scratch maps the other results alternate between
# (sized for the largest map they hold). OUT: the 40-channel head output (level 0, to spec 43). 256-byte aligned.
# A decoder stage's block input (written by its expand) covers its whole window grid (spec 41 draft 3 border rule):
# 8 nwx x 8 nwy tokens, at most 8 ceil((W + 7) / 8) x 8 ceil((H + 7) / 8) over all offsets (padded_tokens).
#
# Stage operations (op, src, dst, extra): embed, block, merge, expand (+ skip add), head.

def stage_io(name):
    order = {
        "e0": [("embed", "X0", "A"), ("block", "A", "S0"), ("merge", "S0", "A")],
        "e1": [("block", "A", "S1"), ("merge", "S1", "A")],
        "e2": [("block", "A", "S2"), ("merge", "S2", "A")],
        "e3": [("block", "A", "S3"), ("merge", "S3", "A")],
        "e4": [("block", "A", "S4"), ("merge", "S4", "A")],
        "m": [("block", "A", "B")],
        "d4": [("expand", "B", "A"), ("block", "A", "B")],
        "d3": [("expand", "B", "A"), ("block", "A", "B")],
        "d2": [("expand", "B", "A"), ("block", "A", "B")],
        "d1": [("expand", "B", "A"), ("block", "A", "B")],
        "d0": [("expand", "B", "A"), ("block", "A", "B"), ("head", "B", "OUT")],
    }
    return order[name]


def _align(n, a=256):
    return cdiv(n, a) * a


def op_output(st, op):
    """(level, channels) of the map an operation of stage st writes."""
    if op == "embed":
        return 0, EMBED_C
    if op == "block" or op == "expand":
        return st["level"], st["C"]
    if op == "merge":
        return st["level"] + 1, st["C_next"]
    if op == "head":
        return 0, OUTPUT_C
    raise ValueError(op)


def padded_tokens(w, h):
    """Tokens of the largest window-grid map of a W x H level over all offsets 0..7."""
    return 8 * cdiv(w + 7, 8) * 8 * cdiv(h + 7, 8)


# The fused network (one kernel per stage, clean/k/kernels/k_block.hip k_stage_*): each stage reads one map and
# writes its block output and its merged map / head output; input and output must differ (windows read neighbours),
# so the merged maps and the decoder outputs alternate between A and B. d0's block output is not written.
FUSED_IO = {
    "e0": {"in": "X0", "block": "S0", "merged": "A"},
    "e1": {"in": "A", "block": "S1", "merged": "B"},
    "e2": {"in": "B", "block": "S2", "merged": "A"},
    "e3": {"in": "A", "block": "S3", "merged": "B"},
    "e4": {"in": "B", "block": "S4", "merged": "A"},
    "m": {"in": "A", "block": "B"},
    "d4": {"in": "B", "skip": "S4", "block": "A"},
    "d3": {"in": "A", "skip": "S3", "block": "B"},
    "d2": {"in": "B", "skip": "S2", "block": "A"},
    "d1": {"in": "A", "skip": "S1", "block": "B"},
    "d0": {"in": "B", "skip": "S0", "head": "OUT"},
}


def buffer_plan(wb, hb, mode="fused", split=()):
    """{name: bytes} of the FP16 maps of one frame: mode "fused" (FUSED_IO) or "split" (stage_io, separate kernels).
    split (mode "fused"): stages that run as separate kernels with the FUSED_IO wiring; they add buffer T for their
    intermediate map (the embedding output of e0, a decoder's padded block input)."""
    if mode == "fused":
        lv_sizes = pyramid(wb, hb)
        tok = [w * h for w, h in lv_sizes]
        plan = {"X0": tok[0] * INPUT_C * 2}
        for st in STAGES:
            io = FUSED_IO[st["name"]]
            outs = {"block": (st["level"], st["C"]), "merged": (st["level"] + 1, st.get("C_next")),
                    "head": (0, OUTPUT_C)}
            for key, (lv, c) in outs.items():
                if key in io:
                    plan[io[key]] = max(plan.get(io[key], 0), tok[lv] * c * 2)
            if st["name"] in split and (st.get("embed") or st["kind"] == "dec"):
                n = padded_tokens(*lv_sizes[st["level"]]) if st["kind"] == "dec" else tok[st["level"]]
                plan["T"] = max(plan.get("T", 0), n * st["C"] * 2)
        return {k: _align(v) for k, v in plan.items()}
    lv_sizes = pyramid(wb, hb)
    tok = [w * h for w, h in lv_sizes]
    plan = {"X0": tok[0] * INPUT_C * 2}
    for st in STAGES:
        for op, _, dst in stage_io(st["name"]):
            lv, c = op_output(st, op)
            n = padded_tokens(*lv_sizes[lv]) if op == "expand" else tok[lv]
            plan[dst] = max(plan.get(dst, 0), n * c * 2)
    return {k: _align(v) for k, v in plan.items()}


# ---- window partition (K3) ------------------------------------------------------------------------------------------
#
# Spec 41: window (a, b) covers columns [8a - ox, 8a - ox + 8) and rows [8b - oy, 8b - oy + 8), 0 <= ox, oy < 8, with
# ceil((W + ox) / 8) x ceil((H + oy) / 8) windows (spec 44). Members outside the map read the mirrored position (edge not
# repeated: -1 -> 1, W -> W - 2). Results are kept for real positions only. A margin of k tokens per side (members
# (8 + 2k)^2) and the other border rules are kept as options for experiments; model K uses margin 0 and "mirror".
#   "zero"   invalid member (gathered as zeros; index -1)
#   "clamp"  nearest map token
#   "mirror" spec 41 (repeated reflection, so any map size works; spec 44 sizes need at most one)

BORDERS = {"zero": 0, "clamp": 1, "mirror": 2}


def window_grid(w, h, ox=0, oy=0, win=WIN):
    if not (0 <= ox < win and 0 <= oy < win):
        raise ValueError(f"window origin ({ox}, {oy}) must be in [0, {win})")
    return cdiv(w + ox, win), cdiv(h + oy, win)


def border_index(p, n, border="mirror"):
    """Map coordinate p (any integer, numpy array) to [0, n) by the border rule; -1 = invalid ("zero")."""
    p = np.asarray(p, dtype=np.int64)
    inside = (p >= 0) & (p < n)
    if border == "zero":
        return np.where(inside, p, -1)
    if border == "clamp":
        return np.clip(p, 0, n - 1)
    if border == "mirror":
        if n == 1:
            return np.zeros_like(p)
        period = 2 * (n - 1)
        r = np.mod(p, period)
        return np.where(r < n, r, period - r)
    raise ValueError(f"unknown border rule {border!r}")


def window_members(w, h, ox=0, oy=0, margin=0, border="mirror", win=WIN):
    """Flat source token index (y W + x) of every window member: int64 [nwy, nwx, S, S], S = win + 2 margin;
    -1 for an invalid member (border "zero" only)."""
    nwx, nwy = window_grid(w, h, ox, oy, win)
    s = win + 2 * margin
    cols = np.arange(nwx)[:, None] * win - ox - margin + np.arange(s)[None, :]  # [nwx, S]
    rows = np.arange(nwy)[:, None] * win - oy - margin + np.arange(s)[None, :]  # [nwy, S]
    cx = border_index(cols, w, border)
    cy = border_index(rows, h, border)
    idx = cy[:, None, :, None] * w + cx[None, :, None, :]
    valid = (cy[:, None, :, None] >= 0) & (cx[None, :, None, :] >= 0)
    return np.where(valid, idx, -1)


def window_core_inside(w, h, ox=0, oy=0, win=WIN):
    """bool [nwy, nwx, win, win]: whether each core member is a real map position (the members a block writes)."""
    nwx, nwy = window_grid(w, h, ox, oy, win)
    cols = np.arange(nwx)[:, None] * win - ox + np.arange(win)[None, :]
    rows = np.arange(nwy)[:, None] * win - oy + np.arange(win)[None, :]
    inx = (cols >= 0) & (cols < w)
    iny = (rows >= 0) & (rows < h)
    return iny[:, None, :, None] & inx[None, :, None, :]


def window_partition(x, ox=0, oy=0, margin=0, border="mirror", win=WIN):
    """x [H, W, C] -> [nwy * nwx, S * S, C] (window-major, members in raster order; invalid members are zero)."""
    x = np.asarray(x)
    h, w, c = x.shape
    idx = window_members(w, h, ox, oy, margin, border, win)
    flat = x.reshape(h * w, c)
    g = np.where((idx >= 0)[..., None], flat[np.maximum(idx, 0)], np.zeros((), x.dtype))
    nwy, nwx, s, _ = idx.shape
    return g.reshape(nwy * nwx, s * s, c)


def window_reverse(win_x, w, h, ox=0, oy=0, margin=0, win=WIN):
    """Inverse for the cores: win_x [nw, S * S, C] -> [H, W, C] from each window's core members that are real map
    positions."""
    win_x = np.asarray(win_x)
    nwx, nwy = window_grid(w, h, ox, oy, win)
    s = win + 2 * margin
    c = win_x.shape[-1]
    core = win_x.reshape(nwy, nwx, s, s, c)[:, :, margin:margin + win, margin:margin + win]
    full = core.transpose(0, 2, 1, 3, 4).reshape(nwy * win, nwx * win, c)
    return full[oy:oy + h, ox:ox + w]


def window_flags_to_map(flags, w, h, ox=0, oy=0, win=WIN):
    """bool [nwy, nwx] per window -> bool [H, W]: the real positions of the flagged windows' cores."""
    flags = np.asarray(flags, dtype=bool)
    full = np.repeat(np.repeat(flags, win, axis=0), win, axis=1)
    return full[oy:oy + h, ox:ox + w]


# Spec 49 draft 3: a window is a threshold window if a block input token (mirrored members included) has an exact sum
# of squares in [0.995 x 65504, 1.005 x 65520] (the FP16 overflow of spec 41's s can go either way there).
THRESHOLD_LO, THRESHOLD_HI = 0.995 * 65504, 1.005 * 65520


def threshold_windows(windows):
    """windows [nw, 64, C] (FP16 block inputs) -> bool [nw]."""
    x = np.asarray(windows, dtype=np.float16).astype(np.float64)
    s = np.sum(x * x, axis=-1)
    return np.any((s >= THRESHOLD_LO) & (s <= THRESHOLD_HI), axis=-1)
