# SPDX-License-Identifier: MIT
"""Probes of the two hardware facts behind the exact matrix step Y (kernels/rr_step_y.hlsli, results/2026-10-20-speed-body.md):

H1  v_wmma_f32_16x16x16_fp8_fp8 returns the exact value when every one of its 16 products and its start value C is a
    multiple of 2^g and the sum of their magnitudes is below 2^(g + 24) ("integer regime": nothing to round, nothing
    to align away). Checked per output value on chains of two WMMAs (D1 = A1 B1 + C, D2 = A2 B2 + D1) for data of the
    step's three chains (fraction / integer-part splits of e4m3 operands) and adversarial data at the bound; every
    value whose premise holds must be exact.
H2  x + y feeding the driver's f16 matrix conversion compiles to v_fma_mixlo_f16 / v_fma_mixhi_f16 x, 1.0, y and gives
    R(x + y), the exact sum rounded once to f16 (nearest even, overflow at 65520, f16 subnormals kept). The harness
    checks the ISA and compares every value with the exact rounding (reference/fmt.round_f16).

    python -m clean.rr_f.gpu.probe_y [--waves N] [--seed S] [--json out.json]"""
import argparse
import json
import re
import sys

import numpy as np

from clean.rr_f.gpu import build
from clean.rr_f.gpu.probe_p1 import acc_to_mat, frag_mats, mat_to_acc
from clean.rr_f.reference.fmt import D8_TABLE, round_f16, two_sum

FINITE = np.array([c for c in range(256) if (c & 0x7F) != 0x7F], np.uint8)
VAL = D8_TABLE.astype(np.float64)
# integer part / fraction of every finite e4m3 code, as codes (both exactly representable)
_ENC = {}
for _c in FINITE:
    _ENC.setdefault(float(VAL[_c]), int(_c))
_ENC[0.0] = 0
INT_CODES = np.array(sorted({_ENC[float(np.trunc(VAL[c]))] for c in FINITE}), np.uint8)
FRAC_CODES = np.array(sorted({_ENC[float(VAL[c] - np.trunc(VAL[c]))] for c in FINITE}), np.uint8)


def lsb_exp(v):
    """Exponent of the lowest set bit of exact float64 values (huge for 0)."""
    v = np.abs(np.asarray(v, np.float64))
    m, e = np.frexp(v)
    sig = np.round(np.ldexp(m, 53)).astype(np.int64)
    low = sig & -sig
    tz = np.log2(np.where(low > 0, low, 1).astype(np.float64)).astype(np.int64)
    return np.where(v == 0, 10 ** 6, e - 53 + tz)


def codes_upto(rng, shape, amax, pool=FINITE):
    pool = pool[np.abs(VAL[pool]) <= amax]
    return rng.choice(pool, size=shape)


def gen_h1(rng, nw):
    """Per wave A1, B1, A2, B2 codes [nw, 32, 8] and C [nw, 32, 8] f32, and a mode name per wave."""
    A = np.zeros((4, nw, 32, 8), np.uint8)
    C = np.zeros((nw, 32, 8), np.float64)
    names = []
    kinds = ("M", "M2", "L", "I", "Mbound", "Ibound", "Lbound", "cancel", "mixed", "full")
    for w in range(nw):
        k = kinds[w % len(kinds)]
        names.append(k)
        if k in ("M", "M2", "Mbound"):          # weights |a| <= amax x integer parts, C on the 2^-9 grid
            amax = 1.0 if k != "M2" else 2.0
            for j in (0, 2):
                A[j, w] = codes_upto(rng, (32, 8), amax)
                A[j + 1, w] = rng.choice(INT_CODES, (32, 8))
            room = 2.0 ** 15 - 2 * 16 * amax * 448
            cmax = room * (0.999 if k == "Mbound" else 0.5)
            C[w] = np.round(rng.uniform(-cmax, cmax, (32, 8)) * 512) / 512
            if k == "Mbound":                    # all products at the maximum, same sign, C at the edge
                A[0, w] = A[2, w] = _ENC[1.0] if amax == 1.0 else _ENC[2.0]
                A[1, w] = A[3, w] = _ENC[448.0]
                C[w] = np.floor(room * 512 - 1) / 512 * np.sign(rng.uniform(-1, 1, (32, 8)))
        elif k in ("L", "Lbound"):               # weights x fractions, C on the 2^-18 grid
            for j in (0, 2):
                A[j, w] = codes_upto(rng, (32, 8), 1.0)
                A[j + 1, w] = rng.choice(FRAC_CODES, (32, 8))
            C[w] = np.round(rng.uniform(-2, 2, (32, 8)) * 2 ** 18) / 2 ** 18
            if k == "Lbound":
                A[0, w] = A[2, w] = _ENC[1.0]
                A[1, w] = A[3, w] = _ENC[0.9375]
                C[w] = (64 - 30.0 - 2.0 ** -18) * np.sign(rng.uniform(-1, 1, (32, 8)))
        elif k in ("I", "Ibound"):               # integer parts x integer parts, integer C
            for j in range(4):
                A[j, w] = rng.choice(INT_CODES, (32, 8))
            C[w] = np.round(rng.uniform(-65504, 65504, (32, 8)))
            if k == "Ibound":
                A[:, w] = _ENC[448.0]
                C[w] = 2.0 ** 24 - 2 * 16 * 448.0 ** 2 - 1
        elif k == "cancel":                      # second WMMA cancels the first (exact zero or tiny remainder)
            A[0, w] = codes_upto(rng, (32, 8), 1.0)
            A[1, w] = rng.choice(INT_CODES, (32, 8))
            A[2, w] = A[0, w]
            A[3, w] = A[1, w] ^ 0x80
            C[w] = np.round(rng.uniform(-8, 8, (32, 8)) * 512) / 512
        elif k == "mixed":                       # odd significands everywhere, lsb 2^-9 operands, small sums
            for j in range(4):
                A[j, w] = codes_upto(rng, (32, 8), 0.25)
            C[w] = np.round(rng.uniform(-1, 1, (32, 8)) * 2 ** 18) / 2 ** 18
        else:                                    # full range: the premise holds only for some values
            for j in range(4):
                A[j, w] = rng.choice(FINITE, (32, 8))
            C[w] = rng.standard_normal((32, 8)).astype(np.float16).astype(np.float64)
    return A, C.astype(np.float32), names


def premise(A1, B1, C):
    """Per output value: products and C multiples of 2^g, sum of magnitudes < 2^(g + 24). A1 [nw,16,16], B1 [nw,16,16],
    C [nw,32,8] (accumulator layout) -> bool [nw,32,8]."""
    out = []
    for i in range(0, len(A1), 512):
        P = A1[i:i + 512, :, :, None] * B1[i:i + 512, None, :, :]   # [w, r, k, c]
        Cm = acc_to_mat(C[i:i + 512])
        g = np.minimum(lsb_exp(P).min(axis=2), lsb_exp(Cm))
        tot = np.abs(P).sum(axis=2) + np.abs(Cm)
        out.append(mat_to_acc((g > 10 ** 5) | (tot < np.ldexp(1.0, np.minimum(g, 1000) + 24))))
    return np.concatenate(out)


def run_h1(nw, seed, lock=True):
    rng = np.random.default_rng(seed)
    A, C, names = gen_h1(rng, nw)
    inp = np.zeros((nw, 32, 64), np.uint8)
    for k in range(4):
        inp[:, :, 8 * k:8 * k + 8] = A[k]
    inp[:, :, 32:64] = C.view(np.uint8).reshape(nw, 32, 32)
    ar = build.Arena()
    pin = ar.add(inp)
    pout = ar.reserve(nw * 32 * 64)
    dx = build.compile_kernel("rr_probe_y.hlsl", "rr_py_wmma", {})
    img, _ = build.run(ar, {"h1": (dx, 3)}, [("h1", nw // 4, [pin, pout, nw * 32])], lock=lock)
    out = img[pout:pout + nw * 32 * 64].reshape(nw, 32, 64)
    D1 = out[:, :, 0:32].copy().view(np.float32).reshape(nw, 32, 8).astype(np.float64)
    D2 = out[:, :, 32:64].copy().view(np.float32).reshape(nw, 32, 8).astype(np.float64)
    A1, B1 = frag_mats(A[0], A[1])
    A2, B2 = frag_mats(A[2], A[3])
    Cd = C.astype(np.float64)
    S1 = mat_to_acc(np.matmul(A1, B1)) + Cd                       # exact in float64 (< 2^53 lattice span)
    S2 = S1 + mat_to_acc(np.matmul(A2, B2))
    p1 = premise(A1, B1, Cd)
    p2 = p1 & premise(A2, B2, S1)                                 # second WMMA starts from the exact D1
    rep = {}
    for k in sorted(set(names)):
        sel = np.array([n == k for n in names])
        rep[k] = dict(values=int(sel.sum()) * 256,
                      premise_d1=int(p1[sel].sum()), bad_d1=int(((D1 != S1) & p1)[sel].sum()),
                      premise_d2=int(p2[sel].sum()), bad_d2=int(((D2 != S2) & p2)[sel].sum()),
                      inexact_outside_premise=int(((D2 != S2) & ~p2)[sel].sum()))
    ok = all(v["bad_d1"] == 0 and v["bad_d2"] == 0 for v in rep.values())
    return ok, rep


def gen_h2(rng, n):
    """Pairs (x, y) of f32 whose exact sum is known in float64; adversarial for the f16 rounding."""
    x = np.zeros(n, np.float64)
    y = np.zeros(n, np.float64)
    q = n // 8
    # f16 midpoints (all binades incl. subnormal), y = 0 / +-tiny / +-larger
    hb = rng.integers(0, 0x7bff, 3 * q).astype(np.uint16)
    h = hb.view(np.float16).astype(np.float64)
    nxt = (hb + 1).view(np.float16).astype(np.float64)
    mid = (h + nxt) / 2
    sgn = np.where(rng.random(3 * q) < 0.5, -1.0, 1.0)
    x[:3 * q] = (mid * sgn).astype(np.float32)
    tiny = np.ldexp(1.0, rng.integers(-40, -12, 3 * q)) * rng.choice([-1.0, 0.0, 1.0], 3 * q)
    y[:3 * q] = tiny.astype(np.float32)
    # random step-like pairs: x on the 2^-9 grid up to 2^15, y on the 2^-18 grid below 2^6
    m = slice(3 * q, 6 * q)
    x[m] = np.round(rng.uniform(-2 ** 15, 2 ** 15, 3 * q) * 512) / 512 * np.ldexp(1.0, -rng.integers(0, 16, 3 * q))
    x[m] = np.round(x[m] * 512) / 512
    y[m] = np.round(rng.uniform(-64, 64, 3 * q) * 2 ** 18) / 2 ** 18 * np.ldexp(1.0, -rng.integers(0, 20, 3 * q))
    y[m] = np.round(y[m] * 2 ** 18) / 2 ** 18
    # overflow edge, zeros, cancellation to f16 subnormals
    m = slice(6 * q, 7 * q)
    x[m] = rng.choice([65504.0, 65520.0, -65520.0, 65536.0, 65519.0, 0.0, -0.0, 2.0 ** -24, 3 * 2.0 ** -25], q)
    y[m] = rng.choice([0.0, -0.0, 2.0 ** -30, -(2.0 ** -30), 2.0 ** -18, 1.0, -1.0], q)
    m = slice(7 * q, n)
    a = np.round(rng.uniform(-64, 64, n - 7 * q) * 2 ** 18) / 2 ** 18
    x[m] = a
    y[m] = -a + np.round(rng.uniform(-2.0 ** -12, 2.0 ** -12, n - 7 * q) * 2 ** 30) / 2 ** 30
    x = x.astype(np.float32).astype(np.float64)
    y = y.astype(np.float32).astype(np.float64)
    return x, y


def run_h2(n, seed, lock=True):
    rng = np.random.default_rng(seed + 7)
    n -= n % 256
    x, y = gen_h2(rng, n)
    inp = np.zeros((n // 8, 64), np.uint8)
    inp[:, :32] = x.astype(np.float32).view(np.uint8).reshape(n // 8, 32)
    inp[:, 32:] = y.astype(np.float32).view(np.uint8).reshape(n // 8, 32)
    ar = build.Arena()
    pin = ar.add(inp)
    pout = ar.reserve(n * 2)
    dx = build.compile_kernel("rr_probe_y.hlsl", "rr_py_mix", {}, isa=True)
    isa = build.isa_text(dx)
    nmix = len(re.findall(r"v_fma_mix(lo|hi)_f16", isa))
    ncvt = len(re.findall(r"v_cvt_f16_f32|v_cvt_pk_rtz|v_add_f32|v_dual_add_f32", isa))
    img, _ = build.run(ar, {"h2": (dx, 3)}, [("h2", n // 8 // 128, [pin, pout, n // 8])], lock=lock)
    got = img[pout:pout + n * 2].view(np.uint16)
    s, e = two_sum(x, y)
    want = round_f16(s, e).astype(np.float16).view(np.uint16)
    bad = got != want
    zero = (want & 0x7FFF) == 0
    rep = dict(values=int(n), isa_fma_mix=nmix, isa_add_or_cvt=ncvt, bad=int((bad & ~zero).sum()),
               zero_sign_differs=int((bad & zero).sum()))
    if rep["bad"]:
        i = np.flatnonzero(bad & ~zero)[:5]
        rep["examples"] = [dict(x=float(x[j]), y=float(y[j]), got=int(got[j]), want=int(want[j])) for j in i]
    return rep["bad"] == 0 and nmix == 8 and ncvt == 0, rep


STEP_MODES = {0: "step0 (b split, C = 0)", 1: "step (b split)", 2: "step0t (a split, C = 0)", 3: "stept (a split)",
              4: "score (both split)", 5: "z_fast (one chain, R(c + M))", 6: "z_fast0 (one chain, C = 0)"}
Z_KINDS = ("zsmall", "zsmallb", "zint", "zintb", "zzero")
INT8_CODES = np.array([c for c in FINITE if VAL[c] == 0 or abs(VAL[c]) >= 8], np.uint8)   # the 'int' class


def gen_zfast(rng, nw, mode, kind):
    """Operands of the fast single-chain step in its two classes (kernels/rr_step_y.hlsli, RR_ZFAST), at and inside
    the bounds: 'small' with |a| <= 1/4 (lowest bit 2^-9, so g = -9, L1 <= 8) and |b| < 8 (L1 * 7.5 = 60 < 2^6);
    'int' with integer a (g = 0, |a| <= 448) and b in {0} or [8, 448] (L1 * 448 <= 14336 * 448 < 2^24)."""
    sh = (2, nw, 32, 8)
    if kind in ("zsmall", "zsmallb"):
        A = codes_upto(rng, sh, 0.25)
        B = codes_upto(rng, sh, 7.5)
        if kind == "zsmallb":            # half the products at the maximum (one sign per value), half at 2^-18
            big = rng.random(sh) < 0.5
            A = np.where(big, _ENC[0.25], _ENC[2.0 ** -9]).astype(np.uint8)
            B = np.where(big, _ENC[7.5], _ENC[2.0 ** -9]).astype(np.uint8)
            neg = rng.random((nw, 1, 8)) < 0.5
            B = np.where(neg[None], B | 0x80, B).astype(np.uint8)
    elif kind in ("zint", "zintb"):
        A = rng.choice(INT_CODES[(VAL[INT_CODES] != 0)], sh)
        B = rng.choice(INT8_CODES, sh)
        if kind == "zintb":
            A[:] = _ENC[448.0]
            B[:] = _ENC[448.0]
            B = np.where(rng.random(sh) < 0.03, _ENC[8.0], B).astype(np.uint8)
            A = np.where(rng.random(sh) < 0.03, _ENC[1.0], A).astype(np.uint8)
    else:                                # zero products of both signs, exact cancellation, -0 / +0 start values
        A = rng.choice(np.array([_ENC[1.0], _ENC[1.0] | 0x80, _ENC[0.5], 0x80, 0x00], np.uint8), sh)
        B = rng.choice(np.array([0x00, 0x80, _ENC[2.0], _ENC[2.0] | 0x80], np.uint8), sh)
        B[1] = B[0] ^ 0x80
        A[1] = A[0]
    C = gen_steps(rng, nw, 1, "full")[4]
    if kind == "zzero":
        C = rng.choice(np.array([0.0, -0.0, 2.0 ** -24, -(2.0 ** -24), 1.0, -1.0], np.float32), (nw, 32, 8))
    if kind == "zintb":
        C = np.where(rng.random((nw, 32, 8)) < 0.2, np.where(rng.random((nw, 32, 8)) < 0.5, -np.inf, np.inf),
                     C).astype(np.float32)
    if mode == 6:
        C[:] = 0
    return A[0], A[1], B[0], B[1], C


def gen_steps(rng, nw, mode, kind):
    """A0, A1, B0, B1 codes [nw, 32, 8], C [nw, 32, 8] f32 for a step test; the bounded operand |x| <= 1.875."""
    bounded = codes_upto(rng, (2, nw, 32, 8), 1.875)
    free = rng.choice(FINITE, (2, nw, 32, 8))
    if kind == "net":                                     # network-like magnitudes
        free = codes_upto(rng, (2, nw, 32, 8), 8.0)
    if kind == "adv":                                     # second half cancels the first, extremes
        free[1] = free[0] ^ 0x80
        bounded[1] = bounded[0]
    if kind == "inf":                                     # integer weights 1..15 (the stand-in models), +-inf starts
        ints = np.array([c for c in FINITE if VAL[c] == np.round(VAL[c]) and 1 <= abs(VAL[c]) <= 15], np.uint8)
        bounded = rng.choice(ints, (2, nw, 32, 8))
    if kind in ("l1", "l1b"):                            # L5 bounds: large single weights, small row sums L1
        # every output's 32 bounded values (lanes r, r + 16 of both fragments): tiny ones (+-2^-9, 0) and a few
        # large ones; l1: one 32 (A = 32 is outside the old 16 A < 2^6 bound), l1b: 64 + 7 + 0.5 (L1 at the M-chain
        # bound 448 L1 < 2^15 - 64 with g = -9). The free operand: every finite code, 448 often.
        bounded = rng.choice(np.array([0x01, 0x81, 0x00], np.uint8), (2, nw, 32, 8), p=[0.45, 0.45, 0.1])
        bigs = [32.0] if kind == "l1" else [64.0, 7.0, 0.5]
        for w in range(nw):
            for r in range(16):
                slots = rng.choice(32, len(bigs), replace=False)      # 2 fragments x 2 lanes x 8 values
                for v, sl in zip(bigs, slots):
                    f, m = sl // 16, sl % 16
                    bounded[f, w, r + 16 * (m // 8), m % 8] = _ENC[v] | (0x80 if rng.random() < 0.5 else 0)
        free = np.where(rng.random((2, nw, 32, 8)) < 0.3, _ENC[448.0], rng.choice(FINITE, (2, nw, 32, 8))).astype(np.uint8)
        free = np.where(rng.random((2, nw, 32, 8)) < 0.5, free | 0x80, free & 0x7F).astype(np.uint8)
    if mode in (0, 1):
        A, B = bounded, free
    elif mode in (2, 3):
        A, B = free, bounded
    else:
        A, B = rng.choice(FINITE, (2, nw, 32, 8)), rng.choice(FINITE, (2, nw, 32, 8))
        if kind == "adv":
            B[1] = B[0] ^ 0x80
            A[1] = A[0]
    # C: f16 values over the whole range, tiny ones with fine bits, values on the 2^-9 / 64 grids, zeros, signs
    k = rng.integers(0, 6, (nw, 32, 8))
    k = np.where((k == 4) & (rng.random((nw, 32, 8)) < 0.97), 5, k)        # rare edge / out-of-range values
    mag = np.ldexp(rng.uniform(1, 2, (nw, 32, 8)), rng.integers(-24, 12, (nw, 32, 8)))   # below the 5824 bound
    C = np.where(rng.random((nw, 32, 8)) < 0.5, -mag, mag)
    C = np.where(k == 0, 0.0, C)
    C = np.where(k == 1, np.round(C * 512) / 512, C)
    C = np.where(k == 2, np.ldexp(rng.integers(-2047, 2048, (nw, 32, 8)).astype(np.float64), -24), C)
    C = np.where(k == 3, np.round(rng.uniform(-90, 90, (nw, 32, 8))) * 64 + rng.choice([-32.0, 32.0, 0.0], (nw, 32, 8)), C)
    C = np.where(k == 4, rng.choice([5760.0, -5760.0, 5792.0, -5792.0, 5823.0, 65504.0, -65504.0], (nw, 32, 8)), C)
    if kind == "inf":
        C = np.where(rng.random((nw, 32, 8)) < 0.3, np.where(rng.random((nw, 32, 8)) < 0.5, -np.inf, np.inf),
                     rng.choice([65504.0, -65504.0, 30000.0, 1.0], (nw, 32, 8)))
    if kind in ("l1", "l1b"):
        t = l1_thr(kind)
        C = np.where(k == 4, rng.choice([t - 1, -(t - 1), t - 64, -(t - 64)], (nw, 32, 8)),
                     np.clip(C, -(t - 1), t - 1))
    C = C.astype(np.float16).astype(np.float32)
    if mode in (0, 2):
        C[:] = 0
    return A[0], A[1], B[0], B[1], C


def l1_thr(kind):
    """Start-value bound of the l1 kinds: 2^(g + 24) - 448 L1 - 64 with g = -9 and the largest L1 the generator can
    make (all 32 values at their largest)."""
    L1 = (32.0 if kind == "l1" else 71.5) + 32 * 2.0 ** -9
    return float(np.floor(2.0 ** 15 - 448 * L1 - 64))


# variant Z2 (RR_ZFAST 2, kernels/rr_step_y.hlsli): Y2 steps (modes 1 / 3, bound 2^(g + 15) as the argument), the
# score gates (7 two chains, 8 one chain; A = k, B = q, C = the bias) and the range gate (9, R of the A operand)
Z2_MODES = {1: "Y2 step (b split)", 3: "Y2 stept (a split)", 7: "Z2 score, two chains", 8: "Z2 score, one chain",
            9: "Z2 range gate, one chain"}
Z2_KINDS = {1: ("full", "net", "adv", "inf", "l1", "l1b", "edge"), 3: ("full", "net", "adv", "inf", "l1", "l1b", "edge"),
            7: ("full", "net", "adv", "nar", "narb", "nars"), 8: ("full", "net", "adv", "nar", "narb", "nars"),
            9: ("full", "net", "adv", "l1b", "rgb")}


def field_window(rng, sh, nw):
    """Codes whose exponent fields lie in a random window per wave (low field 1..8, width 4..10), random signs."""
    lo = rng.integers(1, 9, (nw, 1, 1))
    wd = rng.integers(4, 11, (nw, 1, 1))
    fl = np.minimum(lo + (rng.random(sh) * (wd + 1)).astype(np.int64), 14)
    c = ((fl << 3) | rng.integers(0, 8, sh)).astype(np.uint8)
    return np.where(rng.random(sh) < 0.5, c | 0x80, c).astype(np.uint8)


def gen_z2(rng, nw, mode, kind):
    """Operands of the variant-Z2 tests. Scores (7, 8): 'nar' codes with exponent fields 4..12 (spans that pass the
    two-chain gate, not the one-chain gate), 'narb' fields 6..10 with values at the top field and cancelling halves;
    range gate (9): 'rgb' bounded operand 1.0 or 2^-9 per value, free operand over the exponent fields 1..14."""
    sh = (2, nw, 32, 8)
    if mode in (1, 3) and kind == "edge":                # sums near the bound 2^6 with products down to 2^-18
        bounded = np.where(rng.random(sh) < 0.05, _ENC[1.875], _ENC[2.0 ** -9]).astype(np.uint8)
        bounded = np.where(rng.random(sh) < 0.5, bounded | 0x80, bounded).astype(np.uint8)
        free = np.where(rng.random(sh) < 0.05, _ENC[8.0], rng.choice(FINITE[np.abs(VAL[FINITE]) <= 1.0], sh)).astype(np.uint8)
        C = gen_steps(rng, nw, 1, "full")[4]
        A, B = (bounded, free) if mode == 1 else (free, bounded)
        return A[0], A[1], B[0], B[1], C
    if mode in (1, 3) or (mode == 9 and kind != "rgb"):
        return gen_steps(rng, nw, 1 if mode != 3 else 3, kind)
    C = gen_steps(rng, nw, 1, "full")[4]
    if mode == 9:
        A = np.where(rng.random(sh) < 0.1, _ENC[1.0], _ENC[2.0 ** -9]).astype(np.uint8)
        A = np.where(rng.random(sh) < 0.5, A | 0x80, A).astype(np.uint8)
        B = field_window(rng, sh, nw)
        return A[0], A[1], B[0], B[1], C
    if kind == "nars":
        A, B = field_window(rng, sh, nw), field_window(rng, sh, nw)
        return A[0], A[1], B[0], B[1], C
    if kind in ("nar", "narb"):
        lo, hi = (4, 12) if kind == "nar" else (6, 10)
        A = ((rng.integers(lo, hi + 1, sh) << 3) | rng.integers(0, 8, sh)).astype(np.uint8)
        B = ((rng.integers(lo, hi + 1, sh) << 3) | rng.integers(0, 8, sh)).astype(np.uint8)
        if kind == "narb":
            A = np.where(rng.random(sh) < 0.2, (hi << 3) | 7, A).astype(np.uint8)
            B = np.where(rng.random(sh) < 0.2, (hi << 3) | 7, B).astype(np.uint8)
            A[1] = A[0]
            B[1] = B[0] ^ 0x80
        A = np.where(rng.random(sh) < 0.5, A | 0x80, A).astype(np.uint8)
        B = np.where(rng.random(sh) < 0.5, B | 0x80, B).astype(np.uint8)
        return A[0], A[1], B[0], B[1], C
    A, B = rng.choice(FINITE, sh), rng.choice(FINITE, sh)
    if kind == "net":
        A, B = codes_upto(rng, sh, 8.0), codes_upto(rng, sh, 8.0)
    if kind == "adv":
        B[1] = B[0] ^ 0x80
        A[1] = A[0]
    return A[0], A[1], B[0], B[1], C


def r_byte(A0, A1):
    """R = 2^(g + 24) / L1 of the A operand (rows = outputs, the 32 K values of both fragments), as stage.e5m3_down."""
    from clean.rr_f.gpu import stage
    Am0, _ = frag_mats(A0, A0)
    Am1, _ = frag_mats(A1, A1)
    Am = np.abs(np.concatenate([Am0, Am1], axis=2))           # [nw, 16 rows, 32 k]
    L1 = float(Am.sum(axis=2).max())
    g = int(lsb_exp(Am[Am != 0]).min())
    return stage.e5m3_down(2.0 ** (g + 24) / L1)


def run_steps(nw, seed, thr=5824.0, lock=True, kinds=("full", "net", "adv", "inf"), z2=False):
    rng = np.random.default_rng(seed + 11)
    rep, ok = {}, True
    ar = build.Arena()
    kernels, disp, cases = {}, [], []
    jobs = [(m, k, False) for m in STEP_MODES for k in (Z_KINDS if m >= 5 else kinds)]
    if z2:
        jobs += [(m, k, True) for m in Z2_MODES for k in Z2_KINDS[m]]
    for mode, kind, isz2 in jobs:
        if True:
            if not isz2 and (kind == "inf" and mode not in (1, 3) or kind in ("l1", "l1b") and mode == 4):
                continue
            if isz2:
                A0, A1, B0, B1, C = gen_z2(rng, nw, mode, kind)
            else:
                A0, A1, B0, B1, C = gen_zfast(rng, nw, mode, kind) if mode >= 5 else gen_steps(rng, nw, mode, kind)
            inp = np.zeros((nw, 32, 64), np.uint8)
            for j, x in enumerate((A0, A1, B0, B1)):
                inp[:, :, 8 * j:8 * j + 8] = x
            inp[:, :, 32:64] = C.view(np.uint8).reshape(nw, 32, 32)
            pin = ar.add(inp)
            pout = ar.reserve(nw * 32 * 20)
            kid = f"m{mode}" + ("z2" if isz2 else "")
            if kid not in kernels:
                defs = {"RR_PY_MODE": mode, **({"RR_PY_Z2": 1} if isz2 else {})}
                kernels[kid] = (build.compile_kernel("rr_probe_y.hlsl", "rr_py_step", defs,
                                                     tag=f"rr_py_step{mode}" + ("_z2" if isz2 else "")), 3)
            if isz2 and mode in (1, 3):                    # the Y2 bound 2^(g + 15) of the bounded operand
                arg = int(np.float32(2.0 ** 15 if kind == "inf" else 2.0 ** 6).view(np.uint32))
            elif isz2 and mode == 9:
                arg = r_byte(A0, A1)
            else:
                t = thr if kind not in ("inf", "l1", "l1b") else (2.0 ** 24 - 32 * 448 * 15 - 64 if kind == "inf" else l1_thr(kind))
                arg = int(np.float32(t).view(np.uint32))
            disp.append((kid, nw // 4, [pin, pout, arg]))
            cases.append(((mode, isz2), kind, A0, A1, B0, B1, C, pout))
    img, _ = build.run(ar, kernels, disp, lock=lock)
    for (mode, isz2), kind, A0, A1, B0, B1, C, pout in cases:
        out = img[pout:pout + nw * 32 * 20].reshape(nw, 32, 20)
        h = out[:, :, :16].copy().view(np.uint16).reshape(nw, 32, 8)
        flag = out[:, :, 16:20].copy().view(np.uint32).reshape(nw, 32)
        Am0, Bm0 = frag_mats(A0, B0)
        Am1, Bm1 = frag_mats(A1, B1)
        S = mat_to_acc(np.matmul(Am0, Bm0) + np.matmul(Am1, Bm1)) + 0.0     # exact (2^-18 lattice, < 2^53)
        Cd = C.astype(np.float64)
        with np.errstate(invalid="ignore"):
            s, e = two_sum(S, np.where(np.isfinite(Cd), Cd, 0.0))
            want = round_f16(s, e).astype(np.float16).view(np.uint16)
        want = np.where(np.isinf(Cd), np.where(Cd > 0, 0x7C00, 0xFC00), want).astype(np.uint16)
        # values of waves without a fixup mark must be exact (the mark is per wave in the kernel: any lane)
        wave_flag = np.repeat((flag != 0).reshape(nw, 32).any(axis=1)[:, None], 32, axis=1)
        unflagged = ~wave_flag[:, :, None].repeat(8, axis=2)
        zero = (want & 0x7FFF) == 0
        bad = (h != want) & unflagged & ~(zero & ((h & 0x7FFF) == 0))
        zs = (h != want) & unflagged & zero & ((h & 0x7FFF) == 0)
        key = f"{Z2_MODES[mode] if isz2 else STEP_MODES[mode]} / {kind}"
        rep[key] = dict(values=int(h.size), unflagged=int(unflagged.sum()), bad=int(bad.sum()),
                        zero_sign_differs=int(zs.sum()))
        if bad.any() or ((mode >= 5 or isz2) and zs.any()):   # the fast steps must also give the zero signs
            ok = False
            w, l, v = np.argwhere(bad)[0]
            rep[key]["example"] = dict(wave=int(w), lane=int(l), v=int(v), got=int(h[w, l, v]), want=int(want[w, l, v]),
                                       C=float(Cd[w, l, v]), S=float(S[w, l, v]))
    return ok, rep


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--waves", type=int, default=20000)
    ap.add_argument("--values", type=int, default=1 << 22)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--json")
    ap.add_argument("--nolock", action="store_true")
    ap.add_argument("--steps", type=int, default=0, help="waves per step test (0: skip)")
    ap.add_argument("--only-steps", action="store_true")
    ap.add_argument("--l5", action="store_true", help="step tests: also the kinds l1 / l1b (lever L5 bounds; not yet run)")
    ap.add_argument("--z2", action="store_true", help="step tests: also the variant-Z2 modes (opt-in; not yet run)")
    a = ap.parse_args(argv)
    if a.steps:
        kinds = ("full", "net", "adv", "inf") + (("l1", "l1b") if a.l5 else ())
        ok3, r3 = run_steps(a.steps - a.steps % 4, a.seed, lock=not a.nolock, kinds=kinds, z2=a.z2)
        for k, v in r3.items():
            print(f"step {k:32s} {v}")
        print("steps", "exact" if ok3 else "NOT EXACT")
        if a.only_steps:
            return 0 if ok3 else 1
    ok1, r1 = run_h1(a.waves - a.waves % 4, a.seed, lock=not a.nolock)
    for k, v in r1.items():
        print(f"H1 {k:8s} {v}")
    ok2, r2 = run_h2(a.values, a.seed, lock=not a.nolock)
    print("H2", r2)
    print("H1", "holds" if ok1 else "FAILS", "| H2", "holds" if ok2 else "FAILS")
    if a.json:
        with open(a.json, "w") as f:
            json.dump(dict(h1=r1, h2=r2, ok=ok1 and ok2, seed=a.seed), f, indent=1)
    return 0 if ok1 and ok2 else 1


if __name__ == "__main__":
    sys.exit(main())
