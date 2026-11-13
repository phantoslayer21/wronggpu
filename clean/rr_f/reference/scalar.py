# SPDX-License-Identifier: MIT
"""Scalar rules of spec 51: attention weight (51.3 step 3), activation (step 8), reciprocal of the row sum (step 4),
sigmoid (51.4), and the converter-side checks of the window-block constants that spec 51 requires (51.3 step 3/4).

All functions take and return float64 arrays holding f16 values. Constants are f16 values (Python floats)."""
from dataclasses import dataclass, asdict
from fractions import Fraction
import math

import numpy as np

from . import exact
from .fmt import (ALL_F16, F16_MAX, add16, fma16, from_f16_bits, mul16, round_f16, two_sum, to_f16_bits)


@dataclass
class WindowConsts:
    """Window-block constants of 51.2 (f16 values; the same in every stage)."""
    A_SCALE: float
    A_CLAMP: float
    A_CLAMPn: float
    A_P1: float
    A_P0: float
    ACT_CL: float
    ACT_CLn: float
    ACT_C1: float
    ACT_C2: float
    ACT_C3: float

    def as_dict(self):
        return asdict(self)


class AttentionRangeError(ValueError):
    pass


# ---------------------------------------------------------------------------------------------------------------
# 51.3 step 3

def clamp(x, lo, hi):
    """min(max(x, lo), hi) with min/max returning the other operand when exactly one is NaN (51.1)."""
    return np.fmin(np.fmax(x, lo), hi)


def attention_p(s, wc):
    """p of 51.3 step 3 for f16 scores s."""
    u = mul16(s, wc.A_SCALE)
    c = clamp(u, wc.A_CLAMPn, wc.A_CLAMP)
    cc = c * c                                   # exact
    s1, e1 = two_sum(wc.A_P1, -cc)
    inner = round_f16(s1, e1)                    # R(A_P1 - c*c)
    return fma16(c, inner, wc.A_P0)              # R(c*inner + A_P0)


def attention_weight(s, wc, check=True):
    """w of 51.3 step 3: w = 2^(n-15) * (1 + y - n), y = 32*(p - 1), n = floor(y); exact f16 for 1 <= n <= 30.
    Outside that range the spec gives no value: raises AttentionRangeError (check=True) or returns NaN there."""
    p = attention_p(s, wc)
    y = 32.0 * (p - 1.0)                         # exact
    n = np.floor(y)
    ok = (n >= 1) & (n <= 30)
    if check and not np.all(ok):
        raise AttentionRangeError("attention p outside [1+1/32, 1+31/32)")
    nn = np.where(ok, n, 15.0)
    w = np.ldexp(1.0 + y - nn, (nn - 15).astype(np.int64))
    w = np.where(ok, w, np.nan)
    # exactly representable (no rounding in this step)
    if not np.array_equal(round_f16(w[ok]), w[ok]):
        raise AssertionError("attention weight not exactly representable")
    return w


def check_attention_consts(wc):
    """The converter checks of 51.3 steps 3-4 (to be carried by page 57): every f16 c with
    A_CLAMPn <= c <= A_CLAMP must give p in [1 + 1/32, 1 + 31/32). Returns dict with ok, w_max, and whether
    64 * w_max reaches the f16 overflow threshold (Z may overflow)."""
    c = ALL_F16[np.isfinite(ALL_F16)]
    c = np.unique(c[(c >= wc.A_CLAMPn) & (c <= wc.A_CLAMP)])
    cc = c * c
    s1, e1 = two_sum(wc.A_P1, -cc)
    inner = round_f16(s1, e1)
    p = fma16(c, inner, wc.A_P0)
    y = 32.0 * (p - 1.0)
    n = np.floor(y)
    ok = bool(np.all((n >= 1) & (n <= 30)))
    w_max = None
    if ok:
        w = np.ldexp(1.0 + y - n, (n - 15).astype(np.int64))
        w_max = float(w.max())
    return {
        "ok": ok,
        "p_min": float(p.min()) if len(p) else None,
        "p_max": float(p.max()) if len(p) else None,
        "w_max": w_max,
        "z_may_overflow": (w_max is not None and 64 * w_max >= 65520.0),
    }


# ---------------------------------------------------------------------------------------------------------------
# 51.3 step 4: r = R(1/Z), exact table over all f16 patterns

_RECIP = None


def _build_recip():
    out = np.empty(65536, dtype=np.float64)
    for b in range(65536):
        z = float(ALL_F16[b])
        if math.isnan(z):
            out[b] = np.nan
        elif math.isinf(z):
            out[b] = math.copysign(0.0, z)
        elif z == 0.0:
            out[b] = math.copysign(math.inf, z)
        else:
            out[b] = exact.round_frac(1 / Fraction(z), exact.F16)
    return out


def recip_table():
    global _RECIP
    if _RECIP is None:
        _RECIP = _build_recip()
    return _RECIP


def recip16(z):
    """R(1/z) for f16 values z (exact table)."""
    return recip_table()[to_f16_bits(z)]


# ---------------------------------------------------------------------------------------------------------------
# 51.3 step 8

def activation(x, wc):
    """act(x): k = clamp(x); u = R(ACT_C2 - |k|*ACT_C1); v = R(k*u + ACT_C3); act = R(x*v)."""
    x = np.asarray(x, dtype=np.float64)
    k = clamp(x, wc.ACT_CLn, wc.ACT_CL)
    p = np.abs(k) * wc.ACT_C1                    # exact
    s1, e1 = two_sum(wc.ACT_C2, -p)
    u = round_f16(s1, e1)
    v = fma16(k, u, wc.ACT_C3)
    return mul16(x, v)


# ---------------------------------------------------------------------------------------------------------------
# 51.4 sigmoid: exact table over all f16 patterns

_SIG = None
SIG_NEAR_ULP32 = 4      # "near a midpoint": reference f32 s within 4 f32 ulps (about 2^-21 relative) of an f16 midpoint


def sigmoid_eval(x):
    """The single-precision evaluation of 51.4 for one f16 value x (Python float), with correctly rounded exp2 and
    division. Returns (sig as float, s as f32 float)."""
    if math.isnan(x):
        return math.nan, math.nan
    L = exact.log2e_f32()
    if math.isinf(x):
        t = -math.inf if x > 0 else math.inf
    else:
        t = exact.round_frac(Fraction(x) * Fraction(-L), exact.F32)
    q = exact.exp2_correct_f32(t)
    if math.isinf(q):
        d = math.inf
    else:
        d = exact.round_frac(1 + Fraction(q), exact.F32)
    if math.isinf(d):
        s = 0.0
    else:
        s = exact.round_frac(1 / Fraction(d), exact.F32)
    return exact.round_frac(Fraction(s), exact.F16), s


def _build_sig():
    sig = np.empty(65536, dtype=np.float64)
    s32 = np.empty(65536, dtype=np.float64)
    near = np.zeros(65536, dtype=bool)
    cache = {}
    for b in range(65536):
        x = float(ALL_F16[b])
        if math.isnan(x):
            sig[b] = np.nan
            s32[b] = np.nan
            continue
        g, s = sigmoid_eval(x)
        sig[b] = g
        s32[b] = s
        if s > 0:
            ulp = 2.0 ** (max(math.frexp(s)[1] - 1, -126) - 23)
            # distance of s from the nearest f16 midpoint, in f32 ulps
            near[b] = exact.is_midpoint_close(Fraction(s), exact.F16, Fraction(SIG_NEAR_ULP32) * Fraction(ulp) / Fraction(s))
    return sig, s32, near


def sig_tables():
    """(sig[65536], s_f32[65536], near_midpoint[65536]) indexed by f16 bit pattern."""
    global _SIG
    if _SIG is None:
        _SIG = _build_sig()
    return _SIG


def sig16(x):
    return sig_tables()[0][to_f16_bits(x)]


def sig_near(x):
    """True where sig(x) may legitimately be one f16 step off in a conforming implementation (51.4 accuracy)."""
    return sig_tables()[2][to_f16_bits(x)]
