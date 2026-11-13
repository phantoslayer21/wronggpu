# SPDX-License-Identifier: MIT
"""Runs the spec 51 D3D12 window-block kernel (clean/rr_f/kernels/rr_block.hlsl) on the reference vectors
(clean/rr_f/tests/vectors) and compares every output with the stated tolerance (manifest 'tolerance_classes').

    python -m clean.rr_f.gpu.stage [--cases E2_r32x32,...|--stages E2,D2-F] [--z supplied|exact|f32] [--nolock]

Z modes (51.3 step 4, 51.8 item 1): supplied = the reference's row sums Z are given to the kernel (expected:
identical); exact = the kernel's own exact sum rounded once (the reference's rule; expected: identical); f32 = the
kernel's own f32 sum (spec-accepted order; P may differ by one code in rows whose Z differs).
Outputs: e4m3 tensors compared code for code (0x00 == 0x80 in the network), f16 hidden states bit for bit (+0 == -0)."""
import argparse
import json
import os
import re
import sys
import time

import numpy as np

from clean.rr_f.gpu import build, layout
from clean.rr_f.reference.fmt import to_f16_bits
from clean.rr_f.reference.scalar import sig_tables
from clean.rr_f.reference.stages import effective_shift, stage_info
from clean.rr_f.tests.vectors_io import DEFAULT_DIR, load_case, load_manifest

ZMODE = {"f32": 0, "supplied": 1, "exact": 2}
STEPS = {"E": 0, "X": 1, "Y": 3, "Z": 3, "Z2": 3, "XF": 1, "I": 4}   # XF: variant X restricted to the windows a Y dispatch marked (fixup)
# variant Z = variant Y with its fast single-chain steps (RR_ZFAST, kernels/rr_step_y.hlsli; host classes: z_params)
# variant I = the exact step on integer WMMA (kernels/rr_step_i.hlsli; weights as int8 digit planes, layout.pack_weights_int)
# variant Y (kernels/rr_step_y.hlsli): the bounded operand of every product (weights; P of the attention: e4m3 values
# in [0, 1], lsb 2^-9 at most)
Y_BOUNDED = ("Wq", "Wv", "Wo", "W1", "W2", "Wm", "Wa", "Wup", "Wh")


def _l1_g(codes):
    """(L1, g) of one bounded tensor (e4m3 codes [..., K, N], K inputs): L1 = the largest sum of |value| over the 32
    inputs of one 51.1 step of one output, g = the exponent of the lowest set bit over the nonzero values (None: all
    zero). NaN codes: (None, None) with L1 = inf."""
    from clean.rr_f.reference.fmt import D8_TABLE
    codes = np.asarray(codes, np.uint8)
    if np.any((codes & 0x7F) == 0x7F):
        return float("inf"), None
    v = np.abs(D8_TABLE[codes]).astype(np.float64)
    K, N = v.shape[-2], v.shape[-1]
    assert K % 32 == 0, v.shape
    L1 = float(v.reshape(-1, K // 32, 32, N).sum(axis=2).max(initial=0.0))
    nz = v[v != 0]
    if not len(nz):
        return L1, None
    m, e = np.frexp(nz)
    sig = np.round(np.ldexp(m, 8)).astype(np.int64)            # e4m3 significands: at most 4 bits, exact here
    return L1, int((e - 8 + np.log2(sig & -sig).astype(np.int64)).min())


def y_bounds(W, l1=False):
    """Host checks of variant Y for one stage's logical weights. Returns (ok, thr): ok when every bounded tensor T
    (weights, and P with L1 = 32 (32 values <= 1), g = -9) has finite codes and
      L1_T / 2 < 2^(g_T + 15)   (fraction chain: products multiples of 2^(g_T - 9), magnitudes <= L1_T / 2; H1),
      g_T <= 0                  (integer start values lie on the integer-part chain's grid 2^g_T),
      448 L1_T + L1_T / 2 < 2^20 - 65520  (+-inf start values clamped to +-2^20 stay infinite, rr_step_y.hlsli),
    and every bias is finite and below thr - 1; thr = the bound of the start values |c| of the integer-part chain,
    min over T of 2^(g_T + 24) - 448 L1_T - 64 (its products: multiples of 2^g_T, magnitudes <= 448 L1_T; larger
    start values mark the window for the fixup). L1_T = the largest sum of |value| over the 32 inputs of one step of
    one output (_l1_g), g_T = exponent of the lowest set bit over the nonzero values. (Lever L5: the bounds used
    32 A_T = 32 max |value| in place of L1_T before.)

    l1=False (default; steps Y / Z as verified in e11ec7a): the committed bounds, L1_T replaced by 32 A_T and the
    +-inf condition A_T <= 64. l1=True (lever L5): the L1_T bounds above; used by variant Z2 only (opt-in, not yet
    verified on the vector cases)."""
    if not l1:
        return _y_bounds_amax(W)
    tensors = [("P", 32.0, -9)]
    for k in Y_BOUNDED:
        if k not in W:
            continue
        a = np.asarray(W[k], np.uint8)
        L1, g = _l1_g(a)
        if not np.isfinite(L1):
            return False, 0.0
        if g is None:
            continue
        tensors.append((k, L1, g))
    thr = float("inf")
    for _, L1, g in tensors:
        if not (L1 / 2 < 2.0 ** (g + 15) and g <= 0 and 448 * L1 + L1 / 2 < 2.0 ** 20 - 65520):
            return False, 0.0
        thr = min(thr, 2.0 ** (g + 24) - 448 * L1 - 64)
    return _y_bias_check(W, thr)


def _y_bounds_amax(W):
    """y_bounds as committed in e11ec7a (A_T = max |value| of the bounded tensor T, 32 A_T in place of L1_T)."""
    from clean.rr_f.reference.fmt import D8_TABLE
    tensors = [("P", 1.0, -9)]
    for k in Y_BOUNDED:
        if k not in W:
            continue
        codes = np.asarray(W[k], np.uint8)
        if np.any((codes & 0x7F) == 0x7F):
            return False, 0.0
        v = np.abs(D8_TABLE[codes]).ravel()
        v = v[v != 0]
        if not len(v):
            continue
        m, e = np.frexp(v)
        sig = np.round(np.ldexp(m, 8)).astype(np.int64)        # e4m3 significands: at most 4 bits, exact here
        low = sig & -sig
        g = int((e - 8 + np.log2(low).astype(np.int64)).min())
        tensors.append((k, float(v.max()), g))
    thr = float("inf")
    for _, A, g in tensors:
        if not (16.0 * A < 2.0 ** (g + 15) and g <= 0 and A <= 64):     # A <= 64: +-inf starts (rr_step_y.hlsli)
            return False, 0.0
        thr = min(thr, 2.0 ** (g + 24) - 32 * 448 * A - 64)
    return _y_bias_check(W, thr)


def _y_bias_check(W, thr):
    # biases start the integer-part chain with their host split (no runtime check): finite and below the bound
    for k in ("bo", "b1", "b2", "bm", "ba", "bup", "bh", "Tb"):
        if k in W:
            b = np.asarray(W[k], np.float64)
            if not np.all(np.isfinite(b)) or (k != "Tb" and np.abs(b).max(initial=0.0) + 1 >= thr):
                return False, 0.0
    return thr > 64, thr


# fast-step classes of variant Z: (byte of c_zt0 / c_zt1, int bit, weight tensors); the matrices are [K][N] (K inputs)
Z_CLASSES = (("x", (0, 0), 1, ("Wq", "Wv")), ("o", (0, 1), 2, ("Wo",)), ("w1", (0, 2), 4, ("W1",)),
             ("w2", (0, 3), 8, ("W2",)), ("front", (1, 0), 16, ("Wa", "Wup")), ("h", (1, 1), 32, ("Wh",)),
             ("m", None, 64, ("Wm",)))


def z_class(mats):
    """(T, int_ok) of one weight class: T = the largest e4m3 code with L1max * maxval(codes < T) < 2^(g + 15) (0: no
    small class), int_ok = L1max * 448 < 2^(g + 24). L1max = the largest sum of |w| over the 32 inputs of one 51.1 step
    of one output, g = the exponent of the lowest set bit over the nonzero values (rr_step_y.hlsli, RR_ZFAST)."""
    from clean.rr_f.reference.fmt import D8_TABLE
    g, L1 = None, 0.0
    for M in mats:
        codes = np.asarray(M, np.uint8)
        if np.any((codes & 0x7F) == 0x7F):
            return 0, False
        v = np.abs(D8_TABLE[codes]).astype(np.float64)
        K, N = v.shape[-2], v.shape[-1]
        assert K % 32 == 0, v.shape
        L1 = max(L1, float(v.reshape(-1, K // 32, 32, N).sum(axis=2).max(initial=0.0)))
        nz = v[v != 0]
        if len(nz):
            m, e = np.frexp(nz)
            sig = np.round(np.ldexp(m, 8)).astype(np.int64)
            gg = int((e - 8 + np.log2(sig & -sig).astype(np.int64)).min())
            g = gg if g is None else min(g, gg)
    if g is None:                                         # all zero: every product is zero
        return 0x7F, True
    T = 0
    for t in range(1, 0x80):                              # codes 0 .. t-1 ascend in magnitude
        if L1 * float(D8_TABLE[t - 1]) < 2.0 ** (g + 15):
            T = t
    return T, L1 * 448.0 < 2.0 ** (g + 24)


def z_params(W):
    """c_zt0, c_zt1 of variant Z for one stage's logical weights (kernels/rr_block.hlsl RR_ZFAST)."""
    zt = [0, 0]
    bits = 0
    for _, pos, bit, keys in Z_CLASSES:
        mats = []
        for k in keys:
            if k in W:
                a = np.asarray(W[k], np.uint8)
                mats += list(a) if a.ndim == 3 else [a]    # per head [H][K][N]
        if not mats:
            continue
        T, iok = z_class(mats)
        if pos is not None:
            zt[pos[0]] |= T << (8 * pos[1])
        bits |= bit if iok else 0
    zt[1] |= bits << 16
    return zt[0], zt[1]


# range gates of variant Z2 (RR_ZFAST 2, kernels/rr_step_y.hlsli): R = 2^(g + 24) / L1 per class as an e5m3 byte
# (rounded down); c_zt0 bytes: x (Wq / Wv), o (Wo), W1, W2; c_zt1 bytes: front (Wa / Wup), Wh, Wm
Z2_CLASSES = (((0, 0), ("Wq", "Wv")), ((0, 1), ("Wo",)), ((0, 2), ("W1",)), ((0, 3), ("W2",)), ((1, 0), ("Wa", "Wup")),
              ((1, 1), ("Wh",)), ((1, 2), ("Wm",)))


def e5m3_down(R):
    """The largest (8 + m) 2^(e - 3), 0 <= e <= 31, 0 <= m <= 7, not above R (byte e * 8 + m); 0 (= 1.0) for R < 1:
    a gate bm < 1 passes only all-zero blocks."""
    if not R >= 1.0:
        return 0
    if not np.isfinite(R) or R >= 15.0 * 2.0 ** 28:
        return 31 * 8 + 7
    e = int(np.floor(np.log2(R)))
    m = int(np.floor((R / 2.0 ** e - 1.0) * 8.0))
    while (8 + m) * 2.0 ** (e - 3) > R:                    # guard against log2 rounding
        m -= 1
        if m < 0:
            e, m = e - 1, 7
    return e * 8 + m


def z2_params(W):
    """c_zt0, c_zt1, c_ythr of variant Z2 for one stage's logical weights: the R bytes, and the class grids as nibbles
    g + 15 (classes x, o, W1, W2, front, Wh, Wm; the bound 2^(g + 15) of the Y2 steps, rr_step_y.hlsli)."""
    zt = [0, 0]
    gn = 0
    for ci, ((w, b), keys) in enumerate(Z2_CLASSES):
        L1, g = 0.0, None
        for k in keys:
            if k in W:
                a = np.asarray(W[k], np.uint8)
                for m in (list(a) if a.ndim == 3 else [a]):
                    l, gg = _l1_g(m)
                    L1 = max(L1, l)
                    g = gg if g is None else (g if gg is None else min(g, gg))
        if g is None and L1 == 0:
            R = float("inf")                                 # all-zero weights (or none): every product is 0
        elif g is None or not np.isfinite(L1):
            R = 0.0                                          # NaN codes: never (the stage fails y_bounds anyway)
        else:
            R = 2.0 ** (g + 24) / L1
        zt[w] |= e5m3_down(R) << (8 * b)
        gg = 0 if g is None else g
        # e4m3: g >= -9. g > 0 (e.g. only even integer weights) fails y_bounds(l1=True) (g <= 0), so Z2 never runs
        # on such weights (the stage falls back to X); the nibble is clamped instead of asserting, because
        # StageCall evaluates these parameters for every step and must not fail on the default path.
        gn |= (min(max(gg + 15, 0), 15)) << (4 * ci)
    return zt[0], zt[1], gn


# exponent-span bounds D of the RR_ZSTAT candidate gates (kernels/rr_block.hlsl ZD(k)), k: 0 x, 1 Wo, 2 W1, 3 W2,
# 4 front, 5 Wh, 6 Wm, 7 P of the attention (L1 <= 9/8 for the exact / f32 row sums, 32 with a supplied Z)
ZSPAN_CLASSES = (("Wq", "Wv"), ("Wo",), ("W1",), ("W2",), ("Wa", "Wup"), ("Wh",), ("Wm",))


def z_spans(W, zmode="exact"):
    """D = floor(g + 20 - log2 L1) per class: a single chain on an operand block whose exponent fields (max(e, 1),
    nonzero codes) span at most D is exact (L1 Bmax < 2^(g + gb + 24))."""
    out = []
    for keys in ZSPAN_CLASSES:
        L1, g = 0.0, None
        for k in keys:
            if k in W:
                a = np.asarray(W[k], np.uint8)
                for m in (list(a) if a.ndim == 3 else [a]):
                    l, gg = _l1_g(m)
                    L1 = max(L1, l)
                    g = gg if g is None else (g if gg is None else min(g, gg))
        out.append(99 if g is None or L1 == 0 else (-99 if not np.isfinite(L1) else int(np.floor(g + 20 - np.log2(L1)))))
    L1p = 32.0 if zmode == "supplied" else 9.0 / 8.0
    out.append(int(np.floor(-9 + 20 - np.log2(L1p))))
    return out


def stage_defines(name, zmode, step="E", ldsw=1):
    """Kernel defines of a stage call. step: E / X / Y / Z / XF, optionally followed by "+NAME=VALUE" items (extra
    defines, for timing / diagnostic builds)."""
    if "+" in step:
        base, *extra = step.split("+")
        d = stage_defines(name, zmode, base, ldsw)
        d.update(dict(kv.split("=") for kv in extra))
        return d
    i = stage_info(name)
    kind = {"in0": 0, "enc": 0, "fuse": 2, "rec": 3}[i["kind"]]
    d = {"RR_C": i["C"], "RR_H": i["H"], "RR_KIND": kind, "RR_Z": ZMODE[zmode], "RR_STEP": STEPS[step],
         "RR_LDSW": ldsw}
    if step == "XF":
        d["RR_FIXUP"] = 1
    if step == "Z":
        d["RR_ZFAST"] = 1
    if step == "Z2":
        d["RR_ZFAST"] = 2
    if step == "I" and i["C"] >= 128:
        d["RR_LDSW"] = 0      # variant I: two weight planes per unit do not fit the 32 KB of LDS at C >= 128
    if i["Cn"]:
        d["RR_CN"] = i["Cn"]
    if i["merged_hidden"]:
        d["RR_HMERGE"] = 1
    if i["adapter"]:
        d["RR_ADAPT"] = 1
    if i["Kc"]:
        d["RR_KC"] = i["Kc"]
    if step == "E" and os.environ.get("WG_RR_EDEFS"):   # opt-in step-E variants (tier B, not the default), e.g.
        d.update(dict(kv.split("=") for kv in os.environ["WG_RR_EDEFS"].split(",") if kv))   # RR_VARL=1
    if os.environ.get("WG_RR_WSRC"):          # A/B of the specialised kernels' weight source (RR_WSRC)
        d["RR_WSRC"] = int(os.environ["WG_RR_WSRC"])
    return d


# Kernel of each stage (same arguments, same output bits as rr_block.hlsl for every input): route name -> (source,
# extra defines). Stages not listed run rr_block.hlsl. WG_RR_GENERIC=1 selects rr_block.hlsl for every stage (A/B and
# regression runs); WG_RR_ROUTE="E3=fast0,E4=gen,..." overrides single stages (A/B of the routes).
ROUTES = {"gen": ("rr_block.hlsl", {}), "gen0": ("rr_block.hlsl", {"RR_LDSW": 0}), "c32": ("rr_block_c32.hlsl", {}), "c32s0": ("rr_block_c32.hlsl", {"RR_STICKY": 0}),
          "c32cnt": ("rr_block_c32.hlsl", {"RR_DBG_CNT": 1}),
          "fasts0": ("rr_block_fast.hlsl", {"RR_STICKY": 0}),
          "fastcmp": ("rr_block_fast.hlsl", {"RR_NANSUM": 0}),
          "fastcnt": ("rr_block_fast.hlsl", {"RR_DBG_CNT": 1}),
          "w2cnt": ("rr_block_wide.hlsl", {"RR_G": 2, "WG_BLOCK": 256, "RR_DBG_CNT": 1}),
          "fast": ("rr_block_fast.hlsl", {}), "fast0": ("rr_block_fast.hlsl", {"RR_WSRC": 0}),
          "fast1": ("rr_block_fast.hlsl", {"RR_WSRC": 1}),
          "wide2": ("rr_block_wide.hlsl", {"RR_G": 2, "WG_BLOCK": 256}),
          "wide4": ("rr_block_wide.hlsl", {"RR_G": 4, "WG_BLOCK": 512})}


def route_def(r):
    """(source, extra defines) of a route; 'w<G>p<PRE>' = rr_block_wide.hlsl with RR_G = G, RR_PRE = PRE."""
    if r in ROUTES:
        return ROUTES[r]
    m = re.fullmatch(r"w(\d)p(\d+)(b?)", r)   # suffix b: RR_B32G4 1 (RR_B32 kept at G = 4)
    if not m:
        raise KeyError(r)
    b = m.group(3) == "b"
    G = int(m.group(1))
    return "rr_block_wide.hlsl", {"RR_G": G, "WG_BLOCK": 128 * G, "RR_PRE": int(m.group(2)), **({"RR_B32G4": 1} if b else {})}
SPECIAL = {"IN0": "c32", "D0-F": "c32", "E1": "fast", "D1-F": "fast", "D1-R": "fast",
           # levels 2-5 (few windows): the window spread over 2 or 4 groups of 4 waves (rr_block_wide.hlsl)
           "E2": "wide2", "D2-F": "wide2", "D2-R": "wide2", "E3": "wide2", "D3-F": "wide2", "D3-R": "wide2",
           "E4": "wide4", "E5": "wide4", "D4-F": "w2p4", "D4-R": "w2p4"}
# clean/rr-f-s10-body: with the relaxed body (step E, RR_B32 1 in WG_RR_EDEFS: the accurate and fast builds) level 4's
# decoder pair runs on four groups per window with RR_B32 kept (route w4p4b, RR_B32G4 1: 167 / 151 VGPRs, no spill;
# w2p4 had 209 / 200). Structure only: the same output bits as w2p4 (cap8 + cap48 byte-identical). Its 112 windows are
# latency-bound (D4-F / D4-R took 1.7-2x E4's time); -0.04 to -0.05 ms per frame (1080p / 1440p Q). The exact body
# (RR_B32 0) keeps w2p4. Levels 2-3 on G = 4 or with fewer operand prefetches, and D1 on wide2, measured slower.
# clean/rr-f-a14-frame E1: with every operand prefetch (w4p15b, RR_B32G4 1; the exact body's wide4 plus RR_B32): cap8 +
# cap48 byte-identical to ref MXR (accurate, 5x) and FAST; -0.035 / -0.010 ms at 1080p / 1440p Q against w4p4b
# (WG_RR_EROUTE=D4-F=w4p4b,D4-R=w4p4b at build time: the previous route).
SPECIAL_B32 = {"D4-F": "w4p15b", "D4-R": "w4p15b"}
# clean/rr-f-s13x-x-flow: the exact body (step E, RR_B32 off) runs level 4's decoder pair on four groups as well, with
# every operand prefetch (wide4 = w4p15: 192 / 190 VGPRs, 26 KB LDS; w2p4 had 186 / 166). Structure only: cap8 + cap48
# byte-identical to ref ACC. Only with the dataflow body (stagedx.*, exact drop set 0x7fc): -0.074 / -0.019 ms against
# the round-12 exact frame (1080p / 1440p Q); w4p4 / w4p0 -0.053 / -0.008, w4p7 = w4p15, w4p13 / w4p14 slightly slower;
# w4p4 without dflow +0.054 / +0.050. WG_RR_EXD4=0 at build time keeps w2p4; WG_RR_EXROUTE overrides it.
SPECIAL_EXACT = {"D4-F": "wide4", "D4-R": "wide4"}
# Step X (exact step): rr_block_wide.hlsl measured slower than rr_block.hlsl on levels 2-3 (1440p Q, body_ab, best of
# 10: E2 2.07 -> 3.12, E3 1.08 -> 1.30, D3-F 1.13 -> 1.40, D3-R 0.99 -> 1.22, D2-F 2.11 -> 2.19, D2-R 1.86 -> 2.05 ms),
# faster on levels 4-5 (E4 1.25 -> 0.60, E5 1.20 -> 0.28, D4-F 1.30 -> 0.82, D4-R 1.19 -> 0.76 ms): levels 2-3 stay on
# the generic kernel for step X.
SPECIAL_X = {"E2": "gen", "D2-F": "gen", "D2-R": "gen", "E3": "gen", "D3-F": "gen", "D3-R": "gen"}
# Step Z (r2 lever 4): levels 0-1 read the weights straight from memory (rr_block.hlsl RR_LDSW 0: one q / v exchange
# buffer, no weight staging, no MLP barriers); the other levels stay on rr_block.hlsl with staged weights.
SPECIAL_Z = {"IN0": "gen0", "D0-F": "gen0", "E1": "gen0", "D1-F": "gen0", "D1-R": "gen0"}


def route(name, ldsw=1, step="E"):
    if not ldsw or os.environ.get("WG_RR_GENERIC", "0") == "1":
        return "gen"
    over = dict(kv.split("=") for kv in os.environ.get("WG_RR_ROUTE", "").split(",") if kv)
    if step.split("+")[0] == "E":   # WG_RR_EROUTE: overrides of the step-E kernels only (A/B without touching X / Z)
        over.update(kv.split("=") for kv in os.environ.get("WG_RR_EROUTE", "").split(",") if kv)
        # clean/rr-f-s13x-x-flow: WG_RR_EXROUTE: overrides of the exact body's step-E kernels only (RR_B32 off in
        # WG_RR_EDEFS: stageex.* / stagedx.*), A/B of exact-only routes without touching the relaxed body
        if "RR_B32=1" not in os.environ.get("WG_RR_EDEFS", "").split(","):
            over.update(kv.split("=") for kv in os.environ.get("WG_RR_EXROUTE", "").split(",") if kv)
    if name in over:
        return over[name]
    base = step.split("+")[0]
    if base == "Z" and os.environ.get("WG_RR_ZLDS", "0") != "1":     # WG_RR_ZLDS=1: staged weights (A/B)
        return SPECIAL_Z.get(name, "gen")
    if base not in ("E", "X"):    # variants Y / Z / Z2 / XF / I exist in rr_block.hlsl only (merge clean/rr-f-speed1)
        return "gen"
    if base == "X" and name in SPECIAL_X:
        return SPECIAL_X[name]
    if base == "E" and name in SPECIAL_B32 and "RR_B32=1" in os.environ.get("WG_RR_EDEFS", "").split(","):
        return SPECIAL_B32[name]
    if base == "E" and name in SPECIAL_EXACT and "RR_B32=1" not in os.environ.get("WG_RR_EDEFS", "").split(",")             and os.environ.get("WG_RR_EXD4", "1") == "1":   # WG_RR_EXD4=0 at build time: exact keeps w2p4
        return SPECIAL_EXACT[name]
    if name in SPECIAL:
        return SPECIAL[name]
    if os.environ.get("WG_RR_FAST_ALL", "0") == "1":   # opt-in: every other stage through rr_block_fast.hlsl
        return "fast"
    return "gen"


def kernel_src(name, ldsw=1, step="E"):
    return route_def(route(name, ldsw, step))[0]


def e1_direct(ldsw=1, step="E"):
    """True when E1's kernel for the body step reads its four children straight from IN0's y2 (RR_ADAPT 2): no
    regroup dispatch, and E1's x argument is IN0's y2 ([2 Ny][2 Nx][32]). Steps other than E / X: rr_block.hlsl, the
    regrouped input (RR_ADAPT 1)."""
    return kernel_src("E1", ldsw, step) == "rr_block_fast.hlsl"


def ungroup_children(a):
    """[Ny][Nx][128] (children in stored order) -> the level-0 tensor [2 Ny][2 Nx][32] they came from."""
    Ny, Nx = a.shape[:2]
    return np.ascontiguousarray(a.reshape(Ny, Nx, 2, 2, 32).transpose(0, 2, 1, 3, 4).reshape(2 * Ny, 2 * Nx, 32))


def infold(step="E", ldsw=1):
    """Front-end MLP folded into IN0 (rr_block_c32.hlsl RR_INFOLD 1, clean/rr-f-r4-infold; the same bits as the separate
    rr_input_mlp dispatch, which is then skipped): opt-in WG_RR_INFOLD=1 for step E on the c32 route (measured about
    -0.04 ms on F-B1, results/2026-10-23-r4-infold.md, so not the default). clean/rr-f-s9-kernels: the runtime build
    (runtime/tools/build.py) sets WG_RR_INFOLD=1 with the relaxed body (RR_B32), the default fold
    (variant E, the same bits as input.mlp.e; RR_FOLD_B32 1: the relaxed MLP = rr_input_mlp.hlsl RR_MLP_B32);
    the runtime then skips input.mlp.e."""
    if os.environ.get("WG_RR_INFOLD", "0") != "1" or "+" in step:
        return 0
    return 1 if step == "E" and route("IN0", ldsw, step) == "c32" else 0


DFLOW_SRCS = ("rr_block_wide.hlsl", "rr_block_c32.hlsl", "rr_block_fast.hlsl")


# clean/rr-mp2-level0 L0a: level-0 stages of the relaxed body (step E with RR_B32: the stagee.* kernels of accurate and
# fast) compiled in wave64, design D of rr_block_c32.hlsl (one wave64 = two token tiles, every value computed by the
# same operation: cap8 + cap48 byte-identical in accurate and fast). Not the dataflow forms (staged.*) and not the exact
# body (stageex.*), which keep wave32. WG_RR_W64 at build time: the stage list (default "D0-F,IN0"; "" = wave32 everywhere).
W64_DEFAULT = "D0-F,IN0"


def w64_stages(step="E"):
    if step != "E" or "RR_B32=1" not in os.environ.get("WG_RR_EDEFS", "").split(","):
        return ()
    return tuple(s for s in os.environ.get("WG_RR_W64", W64_DEFAULT).split(",") if s)


def kernel_for(name, zmode, isa=False, step="E", ldsw=1, direct=None, fold=None, dflow=0):
    """direct (E1 only): the input layout the stage call was built with (StageCall.e1_direct); False keeps E1 on
    rr_block.hlsl with the regrouped input whatever the step (e.g. variant X instead of Y / Z / I on failed checks)."""
    r = route(name, ldsw, step)
    if name == "E1" and direct is False:
        r = "gen"
    src = route_def(r)[0]
    if name == "E1" and direct is True and src != "rr_block_fast.hlsl":
        raise ValueError(f"E1 call built for the direct input layout cannot run step {step}: build the StageCall "
                         f"with step={step!r}")
    tag = ("rr_" if r == "gen" else "rs" + r + "_") + name.replace("-", "") + "_z" + zmode + "_" + step +         ("_l" if ldsw else "")
    d = stage_defines(name, zmode, step, ldsw)
    d.update(route_def(r)[1])
    if name == "E1" and src == "rr_block_fast.hlsl" and direct is not False and e1_direct(ldsw, step):
        d["RR_ADAPT"] = 2
    fm = (infold(step, ldsw) if fold is None else fold) if name == "IN0" and r == "c32" else 0
    if fm:
        d["RR_INFOLD"] = fm
        tag += f"_f{fm}"
    if r == "c32" and not dflow and name in w64_stages(step):   # clean/rr-mp2-level0 L0a: the wave64 form (design D)
        d["WG_WAVE"] = 64
        tag += "_w64"
    if dflow:   # clean/rr-f-s12-dflow: the dataflow form (rr_dflow.hlsli; rr_block_wide.hlsl, and since clean/rr-f-a14-l01
        # A1b rr_block_c32.hlsl / rr_block_fast.hlsl)
        assert src in DFLOW_SRCS, (name, src)
        d["RR_DFLOW"] = 1
        tag += "_df"
        # clean/rr-f-a14-frame E4: kind-1 (merged / children) reads wait for both finer rows 2R and 2R + 1 (E1 reads
        # IN0's y2 children directly; E2 reads E1's merged): the level-0/1 forms and E2 only (the other wide stages keep
        # their DXIL)
        if os.environ.get("WG_RR_DFPOLL", "0") == "1":   # a14 E5 b: plain coherent polls (A/B builds)
            d["RR_DF_POLL"] = 1
            tag += "p"
        if src != "rr_block_wide.hlsl" or (name == "E2" and zmode != "exact"):
            d["RR_DF_K1E"] = 1
            tag += "k"
    return build.compile_kernel(src, "rr_block", d, isa=isa, tag=tag)


def kernel_dxil(name, zmode, step="E", ldsw=1):
    """The dxil the body uses for a stage."""
    return kernel_for(name, zmode, step=step, ldsw=ldsw)


def kernel_call(sc, zmode, step="E", ldsw=1):
    """(dxil, args, groups) of one stage call: the kernel the body uses for the stage."""
    return kernel_for(sc.name, zmode, step=step, ldsw=ldsw), sc.args, sc.groups()


def dispatches_for(sc, zmode, step, ldsw=1, kid=None):
    """Kernels and dispatches of one stage call: [(kid, (dxil, nargs), groups, args)]. Variant Y is the Y dispatch
    plus the fixup dispatch (variant X on the marked windows only); when the host checks fail (sc.y_ok False) the
    stage runs variant X."""
    kid = kid or sc.name.replace("-", "") + "_" + zmode + "_" + step
    y_ok = sc.y_ok_l1 if step.split("+")[0] == "Z2" else sc.y_ok     # Z2: lever L5 bounds (y_bounds l1=True)
    if step[0] in "YZ" and not y_ok:
        print(f"variant Y: host bounds fail for {sc.name}, variant X instead", flush=True)
        step, kid = "X", kid + "_X"
    if step[0] == "I" and not sc.i_ok:
        print(f"variant I: weight digits / NaN check fails for {sc.name}, variant X instead", flush=True)
        step, kid = "X", kid + "_X"
    args = sc.args_z2 if step.split("+")[0] == "Z2" else sc.args_i if step[0] == "I" else sc.args
    dr = getattr(sc, "e1_direct", None)
    out = [(kid, (kernel_for(sc.name, zmode, step=step, ldsw=ldsw, direct=dr), len(args)), sc.groups(), args)]
    if step[0] in "YZI":
        fargs = sc.args if step[0] == "I" else args              # the step-X fixup reads the e4m3 weights
        out.append((kid + "_fix", (kernel_for(sc.name, zmode, step="XF", ldsw=ldsw, direct=dr), len(fargs)),
                    sc.groups(), fargs))
    return out


def precompile(jobs, threads=int(os.environ.get("WG_RR_JOBS", "8"))):
    """jobs: [(name, zmode, step, ldsw)] -> compiled in parallel (cached)."""
    import concurrent.futures as cf
    # A fresh build has no assembler helper yet. Build it before the workers:
    # concurrent cl/link invocations otherwise race on the same object/executable.
    build.build_dx12.build_asm_tool(build.BUILD)
    jobs = list(jobs) + [(j[0], j[1], st, j[3]) for j in jobs if j[2][0] in "YZI" for st in ("X", "XF")]
    with cf.ThreadPoolExecutor(threads) as ex:
        list(ex.map(lambda j: kernel_for(j[0], j[1], step=j[2], ldsw=j[3]), sorted(set(jobs))))


def f16b(a):
    return to_f16_bits(np.asarray(a, np.float64)).astype("<u2")


def wc_dwords(wc):
    p = layout.h2pair
    return [p(wc.A_SCALE, wc.A_CLAMP), p(wc.A_CLAMPn, wc.A_P1), p(wc.A_P0, wc.ACT_CL), p(wc.ACT_CLn, wc.ACT_C1),
            p(wc.ACT_C2, wc.ACT_C3)]


class StageCall:
    """Arena and arguments of one stage call."""

    def __init__(self, name, inputs, W, consts, wc, shift_row, Z=None, arena=None, step="E", ldsw=1):
        self.name = name
        i = stage_info(name)
        self.info = i
        C, H = i["C"], i["H"]
        sx, sy = effective_shift(name, shift_row)
        if i["kind"] == "fuse":
            Ny, Nx = inputs["skip"].shape[:2]
        elif i["adapter"]:
            Ny, Nx = inputs["in"].shape[:2]
        else:
            Ny, Nx = inputs["x"].shape[:2]
        self.Nx, self.Ny, self.sx, self.sy = Nx, Ny, sx, sy
        self.nwx, self.nwy = -(-(Nx + sx) // 8), -(-(Ny + sy) // 8)
        ar = arena if arena is not None else build.Arena()
        P = layout.pack_weights(W, C, H)
        p = {k: 0 for k in ("pX", "pCin", "pHold", "pHc", "pWm", "pBm", "pWa", "pBa", "pWup", "pBup", "pWh", "pBh",
                            "pZ", "pMerged", "pHnew", "pHmerged")}
        if i["kind"] == "fuse":
            p["pX"] = ar.add(inputs["skip"])
            p["pCin"] = ar.add(inputs["cin"])
        elif i["adapter"]:
            # direct E1: the level-0 layout (a shape-only placeholder arena keeps its dummy, the pointer is rebound)
            # (step: the body step the call is built for; E1's input layout follows its kernel, e1_direct)
            self.e1_direct = e1_direct(ldsw, step)
            direct = self.e1_direct and not getattr(ar, "shapes_only", False)
            p["pX"] = ar.add(ungroup_children(inputs["in"]) if direct else inputs["in"])
        else:
            p["pX"] = ar.add(inputs["x"])
        if "hold" in inputs:
            p["pHold"] = ar.add(f16b(inputs["hold"]))
        if "Hc" in inputs:
            p["pHc"] = ar.add(f16b(inputs["Hc"]))
        for k, a in (("pWqv", "Wqv"), ("pTb", "Tb"), ("pWo", "Wo"), ("pBo", "bo"), ("pMlp", "MLP"), ("pB1", "b1"),
                     ("pB2", "b2"), ("pWm", "Wm"), ("pBm", "bm"), ("pWa", "Wa"), ("pBa", "ba"),
                     ("pWup", "Wup"), ("pBup", "bup"), ("pWh", "Wh"), ("pBh", "bh")):
            if a in P:
                p[k] = ar.add(P[a])
        if not hasattr(ar, "sig_off"):
            ar.sig_off = ar.add(f16b(sig_tables()[0]))
        p["pSig"] = ar.sig_off
        if Z is not None:
            p["pZ"] = ar.add(f16b(Z))
        self.out = {}
        p["pY2"] = ar.reserve(Ny * Nx * C, fill=0xEE)
        self.out["y2"] = (p["pY2"], (Ny, Nx, C), "e4m3")
        if i["Cn"]:
            p["pMerged"] = ar.reserve((Ny // 2) * (Nx // 2) * i["Cn"], fill=0xEE)
            self.out["merged"] = (p["pMerged"], (Ny // 2, Nx // 2, i["Cn"]), "e4m3")
        if i["hidden"]:
            p["pHnew"] = ar.reserve(Ny * Nx * 8, fill=0xEE)
            self.out["hnew"] = (p["pHnew"], (Ny, Nx, 4), "f16")
        if i["merged_hidden"]:
            p["pHmerged"] = ar.reserve((Ny // 2) * (Nx // 2) * 8, fill=0xEE)
            self.out["hmerged"] = (p["pHmerged"], (Ny // 2, Nx // 2, 4), "f16")
        self.arena = ar
        ra_m = layout.h2pair(consts["ra"], consts["m"])
        f_g = layout.h2pair(consts.get("f", 0.0), consts.get("g", 0.0))
        order = ["pX", "pCin", "pHold", "pHc", "pWqv", "pTb", "pWo", "pBo", "pMlp", "pB1", "pB2", "pWm", "pBm",
                 "pWa", "pBa", "pWup", "pBup", "pWh", "pBh", "pSig", "pZ", "pY2", "pMerged", "pHnew", "pHmerged"]
        # r2: the L1_T bounds (lever L5) for steps Y / Z: the committed A_T form (32 A_T in place of L1_T) fails on
        # the real model (subnormal weights, g = -9: 11 of 15 stages, e.g. IN0 Wv A = 7.5) while L1_T passes on all 15;
        # same proof with the exact sum-of-magnitudes bound (rr_step_y.hlsli, y_bounds)
        self.y_ok, self.y_thr = y_bounds(W, l1=True)
        self._W = W                                                   # variant Z2's host values: computed on use
        g = self.groups()
        marks = np.zeros(g + 1 + 64 + 192, np.uint32)                 # per-window marks, fixup count, RR_ZSTAT counts / bounds
        marks[g + 37:g + 45] = np.asarray(z_spans(W, zmode="supplied" if Z is not None else "exact"), np.int32).view(np.uint32)
        self.pYflag = ar.add(marks)
        thr = np.float32(min(self.y_thr, 3.0e38) if self.y_ok else 0.0)
        self.zt = z_params(W)
        self.args = ([p[k] for k in order] + [Nx, Ny, sx, sy, self.nwx, ra_m, f_g] + wc_dwords(wc) +
                     [self.pYflag, int(thr.view(np.uint32)), self.zt[0], self.zt[1]])
        # variant I: the weights as int8 digit planes + tables (layout.pack_weights_int), same arguments otherwise
        PI, self.i_ok = layout.pack_weights_int(W, C, H)
        self.i_slots = {}                                            # argument index -> offset of the int tensor
        for k, a in (("pWqv", "Wqv"), ("pWo", "Wo"), ("pMlp", "MLP"), ("pWm", "Wm"), ("pWa", "Wa"), ("pWup", "Wup"),
                     ("pWh", "Wh")):
            if a in PI:
                self.i_slots[order.index(k)] = ar.add(PI[a])

    @property
    def args_i(self):
        """The arguments with variant I's weight tensors (args may be relinked after construction)."""
        a = list(self.args)
        for i, off in self.i_slots.items():
            a[i] = off
        return a

    @property
    def y_ok_l1(self):
        """Variant Z2's host check (lever L5 bounds, y_bounds l1=True); evaluated only when Z2 is dispatched."""
        return y_bounds(self._W, l1=True)[0]

    @property
    def zt2(self):
        """Variant Z2's class bytes (z2_params); evaluated only when Z2 is dispatched."""
        return z2_params(self._W)

    @property
    def args_z2(self):
        """The arguments with variant Z2's class bytes (args may be relinked after construction)."""
        return self.args[:-3] + [self.zt2[2], self.zt2[0], self.zt2[1]]

    def groups(self):
        return self.nwx * self.nwy

    def fixups(self, img):
        """Windows the variant-X fixup redid (accumulated over every run of the dispatch)."""
        return int(img[self.pYflag + self.groups() * 4:self.pYflag + self.groups() * 4 + 4].view(np.uint32)[0])

    def outputs(self, img):
        res = {}
        for k, (off, shape, typ) in self.out.items():
            nb = int(np.prod(shape)) * (1 if typ == "e4m3" else 2)
            raw = img[off:off + nb]
            res[k] = raw.reshape(shape) if typ == "e4m3" else raw.copy().view("<u2").reshape(shape)
        return res


def compare(k, got, exp, mask=None):
    """Returns dict with counts. e4m3: codes, zero codes equal; f16: bits, zero signs equal."""
    if got.dtype == np.uint8:
        e = np.asarray(exp, np.uint8)
        g = got.copy()
        ez = (e & 0x7F) == 0
        gz = (g & 0x7F) == 0
        diff = (g != e) & ~(ez & gz)
        step1 = diff & ((g & 0x80) == (e & 0x80)) & (np.abs(g.astype(int) - e.astype(int)) == 1)
        return {"n": int(e.size), "diff": int(diff.sum()), "diff_1code": int(step1.sum())}
    e = f16b(exp)
    g = got
    ez = (e & 0x7FFF) == 0
    gz = (g & 0x7FFF) == 0
    diff = (g != e) & ~(ez & gz)
    step1 = diff & ((g & 0x8000) == (e & 0x8000)) & (np.abs(g.astype(int) - e.astype(int)) == 1)
    r = {"n": int(e.size), "diff": int(diff.sum()), "diff_1ulp": int(step1.sum())}
    if mask is not None:
        mm = np.broadcast_to(np.asarray(mask, bool)[..., None], diff.shape)
        r["diff_unflagged"] = int((diff & ~mm).sum())
    return r


def prepare(ar, cases, zmodes, root=DEFAULT_DIR, verbose=True, steps=("E",), ldsws=(1,)):
    """Adds stage runs to an arena: returns (kernels, dispatches, analyze(img) -> result rows)."""
    man = load_manifest(root)
    calls, kernels, disp, models = [], {}, [], {}
    t0 = time.time()
    precompile([(c["stage"], z, st, l) for l in ldsws for st in steps for z in zmodes for c in man["cases"]
                if c["id"] in cases])
    for ldsw, step, zmode in [(l, st, z) for l in ldsws for st in steps for z in zmodes]:
        for case in man["cases"]:
            if case["id"] not in cases:
                continue
            name, inputs, W, consts, wc, shift_row, expected, inter = load_case(man, case, root)
            Z = inter["Z"] if zmode == "supplied" else None
            sc = StageCall(name, inputs, W, consts, wc, shift_row, Z=Z, arena=ar, step=step, ldsw=ldsw)
            kid = name.replace("-", "") + "_" + zmode + "_" + step + ("l" if ldsw else "")
            for k, kern, g, args in dispatches_for(sc, zmode, step, ldsw, kid):
                kernels.setdefault(k, kern)
                disp.append((k, g, args))
            model = None
            if step == "E" and zmode in ("supplied", "exact"):   # variant E's step model (emu.py, probe P1)
                key = (case["id"], zmode)
                if key not in models:
                    from clean.rr_f.gpu import emu
                    from clean.rr_f.reference.stages import run_stage
                    with emu.variant_e():
                        models[key], _ = run_stage(name, inputs, W, consts, wc, shift_row, Z_override=Z)
                model = models[key]
            calls.append((case, zmode + "/" + step + ("/lds" if ldsw else ""), name, sc, expected, model))
    if verbose:
        print(f"prepared {len(calls)} runs, {ar.size / 1e6:.1f} MB arena in {time.time() - t0:.0f} s", flush=True)
    return kernels, disp, lambda img: _analyze(img, calls, root, verbose)


def _analyze(img, calls, root, verbose):
    results = []
    for case, zmode, name, sc, expected, model in calls:
        got = sc.outputs(img)
        row = {"case": case["id"], "stage": name, "z": zmode, "windows": sc.groups(), "outputs": {},
               "y_ok": bool(sc.y_ok), "fixup_windows": sc.fixups(img)}
        if model is not None:
            row["vs_model"] = {k: compare(k, got[k], model[k])["diff"] for k in got}
        for k in got:
            mask = None
            ent = case["outputs"][k]
            if "sig_near_mask" in ent:
                mask = np.fromfile(os.path.join(root, ent["sig_near_mask"]), np.uint8).reshape(ent["shape"][:2])
            row["outputs"][k] = compare(k, got[k], expected[k], mask)
        row["exact"] = all(v["diff"] == 0 for v in row["outputs"].values())
        results.append(row)
        if verbose:
            print(f"{case['id']:22s} {zmode:14s} {'EXACT' if row['exact'] else 'DIFF '} " +
                  " ".join(f"{k}:{v['diff']}/{v['n']}" for k, v in row["outputs"].items()) +
                  ("" if model is None else "   vs E-model: " +
                   " ".join(f"{k}:{v}" for k, v in row["vs_model"].items())), flush=True)
    return results


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases")
    ap.add_argument("--stages")
    ap.add_argument("--z", default="supplied")
    ap.add_argument("--nolock", action="store_true")
    ap.add_argument("--json")
    a = ap.parse_args(argv)
    man = load_manifest()
    ids = [c["id"] for c in man["cases"]]
    if a.cases:
        sel = set(a.cases.split(","))
    elif a.stages:
        st = set(a.stages.split(","))
        sel = {c["id"] for c in man["cases"] if c["stage"] in st}
    else:
        sel = set(i for i in ids if not i.startswith("D0-R"))
    ar = build.Arena()
    k, d, an = prepare(ar, sel, a.z.split(","))
    img, _ = build.run(ar, k, d, lock=not a.nolock, workdir=os.path.join(build.BUILD, "runs", "stage"))
    res = an(img)
    if a.json:
        with open(a.json, "w") as f:
            json.dump(res, f, indent=1)
    bad = [r for r in res if not r["exact"]]
    print(f"{len(res) - len(bad)} / {len(res)} case runs identical")
    return 0


if __name__ == "__main__":
    sys.exit(main())
