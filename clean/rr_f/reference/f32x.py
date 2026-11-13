# SPDX-License-Identifier: MIT
"""Vectorized exact f32 / binary16 arithmetic for the image-sized references of page 54b.

Rules (54b.N, the f32 rule pages 54, 55 and 58 cite): binary32, every operation rounded to nearest even once,
subnormal inputs and results flushed to zero (a result is first rounded on the IEEE grid, subnormals included,
and then flushed, which is the convention of reference/output.py's f32()). exp2, log2, rcp: the exact function,
correctly rounded to f32 (the reference rule of 54b.8), subnormal results flushed.

Implementation notes (all exact, no tolerance):
* add, sub, mul, div of two f32 values are computed in binary64 and rounded once more to f32. For these
  operations binary64 (53 >= 2*24 + 2 bits) makes the double rounding innocuous.
* fma uses round-to-odd in binary64 (exact product, TwoSum residual) followed by one f32 rounding, which equals
  the single rounding of the exact a*b + c.
* exp2 / log2 start from NumPy's binary64 functions; every value whose binary64 result lies within 2^-36
  (relative) of an f32 rounding midpoint, and every value near the subnormal range, is recomputed exactly
  (Decimal / Fraction, reference/output.py's cached exact functions). The binary64 functions are far more
  accurate than that margin, so the result is the correctly rounded value.
* binary16 operations reuse reference/fmt.py (exact, error-term aware).

Arrays only; scalar inputs are accepted and give 0-d arrays.
"""
import numpy as np

from . import output as _exact_scalar
from .fmt import fma16, round_f16  # noqa: F401  (re-exported for the resolve module)

F32 = np.float32
F64 = np.float64
FMINN = F32(2.0 ** -126)
FMAX = np.finfo(np.float32).max
HMAX = 65504.0
# Mathematical constants of 54.6 / 54b.7 (not read): log10(2) and log2(10), each rounded to f32.
K6 = F32(np.log10(2.0))
K9 = F32(np.log2(10.0))


def ftz(v):
    """Flush f32 subnormals to zero, keeping the sign of zero."""
    v = np.asarray(v, dtype=F32)
    return np.where(np.abs(v) < FMINN, np.copysign(F32(0), v), v).astype(F32)


def r32(x):
    """Round binary64 values to f32 (RNE on the IEEE grid), then flush subnormal results."""
    with np.errstate(over="ignore", invalid="ignore"):
        return ftz(np.asarray(x, dtype=F64).astype(F32))


def _in(*values):
    return [ftz(v).astype(F64) for v in values]


def add(a, b):
    a, b = _in(a, b)
    with np.errstate(invalid="ignore", over="ignore"):
        return r32(a + b)


def sub(a, b):
    a, b = _in(a, b)
    with np.errstate(invalid="ignore", over="ignore"):
        return r32(a - b)


def mul(a, b):
    a, b = _in(a, b)
    with np.errstate(invalid="ignore", over="ignore"):
        return r32(a * b)


def div(a, b):
    a, b = _in(a, b)
    with np.errstate(invalid="ignore", over="ignore", divide="ignore"):
        return r32(a / b)


def rcp(a):
    return div(F32(1), a)


def fma(a, b, c):
    """One rounding of the exact a*b + c (binary64 round-to-odd, then RNE to f32)."""
    a, b, c = np.broadcast_arrays(*_in(a, b, c))
    with np.errstate(invalid="ignore", over="ignore"):
        p = a * b                      # exact: 48 significant bits, exponent well inside binary64
        s = p + c
        bb = s - p
        e = (p - (s - bb)) + (c - bb)  # TwoSum residual, exact where s is finite
    finite = np.isfinite(s)
    e = np.where(finite, e, 0.0)
    bits = s.view(np.uint64)
    even = (bits & np.uint64(1)) == 0
    bump = finite & (e != 0) & even
    toward = np.where(e > 0, np.inf, -np.inf)
    with np.errstate(invalid="ignore"):
        s = np.where(bump, np.nextafter(s, toward), s)
    return r32(s)


def _near_f32_midpoint(y, margin_bits=16):
    """True where binary64 y may be within ~2^(margin_bits-53) relative of an f32 midpoint, or outside the
    normal f32 range where the low-bit test does not apply."""
    y = np.asarray(y, dtype=F64)
    a = np.abs(y)
    bits = a.view(np.uint64)
    low = (bits & np.uint64(0x1FFFFFFF)).astype(np.int64)
    near = np.abs(low - (1 << 28)) <= (1 << margin_bits)
    finite = np.isfinite(y)
    small = a < 2.0 ** -124
    return finite & (a > 0) & (near | small)


def _fix(result, x, hard, kind):
    if np.any(hard):
        xs = x[hard]
        result[hard] = np.array([_exact_scalar._elementary(kind, float(v)) for v in xs], dtype=F32)
    return result


def exp2(x):
    """Correctly rounded f32 2^x (FTZ inputs and results)."""
    x = ftz(x)
    xf = np.atleast_1d(x).astype(F64)
    with np.errstate(over="ignore", under="ignore"):
        y = np.exp2(xf)
        out = y.astype(F32)
    out = _fix(out, xf, _near_f32_midpoint(y) & ~np.isnan(xf), "exp2")
    return ftz(out.reshape(np.shape(x)))


def log2(x):
    """Correctly rounded f32 log2(x) (FTZ inputs; log2(+-0) = -inf, negative -> NaN)."""
    x = ftz(x)
    xf = np.atleast_1d(x).astype(F64)
    with np.errstate(divide="ignore", invalid="ignore"):
        y = np.log2(xf)
        out = y.astype(F32)
    hard = _near_f32_midpoint(y) & (xf > 0) & np.isfinite(xf)
    out = _fix(out, xf, hard, "log2")
    out = np.where(xf == 0, F32(-np.inf), out)
    return ftz(out.reshape(np.shape(x)))


def isnan(v):
    return np.isnan(np.asarray(v, dtype=F64))


def fmin(a, b):
    """Plain min of 54b.N: the non-NaN operand when exactly one is NaN; NaN when both are.
    Signed zeros (WrongGPU's choice, deterministic): -0 is below +0."""
    a, b = np.broadcast_arrays(np.asarray(a, F64), np.asarray(b, F64))
    lt = (b < a) | ((b == a) & np.signbit(b) & ~np.signbit(a))
    r = np.where(np.isnan(a) | lt, b, a)
    return r


def fmax(a, b):
    """Plain max (as fmin): -0 is below +0."""
    a, b = np.broadcast_arrays(np.asarray(a, F64), np.asarray(b, F64))
    gt = (b > a) | ((b == a) & ~np.signbit(b) & np.signbit(a))
    return np.where(np.isnan(a) | gt, b, a)


def min_nan(a, b):
    """NaN if either operand is NaN, else plain min."""
    a, b = np.broadcast_arrays(np.asarray(a, F64), np.asarray(b, F64))
    return np.where(np.isnan(a) | np.isnan(b), np.nan, fmin(a, b))


def sat(v):
    """Clamp to [0, 1], NaN -> 0 (a -0 input stays -0: clamp of a value inside the range is that value)."""
    v = np.asarray(v, F64)
    return np.where(np.isnan(v), 0.0, np.minimum(np.maximum(v, 0.0), 1.0)).astype(F32)


def f16(v):
    """Round to binary16 (RNE, subnormals kept, overflow to infinity, NaN kept); float64 holder."""
    return round_f16(np.asarray(v, F64))


def rnd_half_away(v):
    """54b.3 rnd(v) = trunc(v + copysign(0.5, v)) with the addition rounded toward zero (f32).

    For every f32 v this is the nearest integer with halves away from zero: the toward-zero addition never
    crosses the next integer. NaN and infinities are returned unchanged (trunc of them)."""
    v = np.asarray(v, F64)
    t = np.trunc(v)
    with np.errstate(invalid="ignore"):
        frac = v - t                   # exact for f32 inputs
        up = np.abs(frac) >= 0.5
    return np.where(np.isfinite(v), np.where(up, t + np.sign(v), t), v)    # -0 stays -0 (trunc(-0 - 0.5))
