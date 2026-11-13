# SPDX-License-Identifier: MIT
"""Vectorised binary32 arithmetic with the rules of spec 52.3 / 54b.N (RR preset F input stage).

Every operation is one IEEE binary32 operation, round to nearest even; subnormal operands are flushed to zero
(sign kept) before the operation and subnormal results after it (round first, then flush: a result that rounds up
to the smallest normal stays). `log2`, `exp2`, `rcp` and `sqrt` are correctly rounded (the reference choice
52.3 names acceptable). `fmin` / `fmax` return the other operand when exactly one operand is NaN, NaN when both
are (52.3, 51.1, 58.2).

Arrays are numpy float32; scalars are accepted wherever arrays are.

Exactness arguments:
- add / sub / mul: numpy float32 operations are IEEE (gradual underflow), then the flush.
- fma: the product of two binary32 values is exact in binary64; the sum with c is split by an error-free
  transformation (two-sum) and rounded once to binary32 with the error term deciding exact midpoints.
- div / rcp / sqrt: the binary64 result is rounded again to binary32. Double rounding is innocuous for these
  operations because 53 >= 2 * 24 + 2.
- log2 / exp2: binary64 library value, rounded to binary32; when that value lies within 2^-40 (relative) of a
  binary32 rounding midpoint the result is recomputed with 100-digit decimal arithmetic (never reached by the
  tests so far; counted in `SLOW_PATH_COUNT`).
"""
from decimal import Decimal, localcontext
from fractions import Fraction
import math

import numpy as np

from . import exact

TINY = np.float32(2.0 ** -126)
F16_MAX = np.float32(65504.0)
LOG10_2 = np.float32(math.log10(2.0))       # f32 roundings of the mathematical constants (52.3)
LOG2_10 = np.float32(math.log2(10.0))
LN_2 = np.float32(math.log(2.0))

SLOW_PATH_COUNT = {"log2": 0, "exp2": 0}


def _check_constants():
    # The float64 values of log10(2), log2(10), ln(2) are within 1 ulp(double) of the exact values; their f32
    # roundings are therefore correct unless the double value were within 2^-29 relative of an f32 midpoint.
    with localcontext() as ctx:
        ctx.prec = 60
        exactv = {"LOG10_2": Decimal(2).log10(), "LOG2_10": Decimal(10).ln() / Decimal(2).ln(),
                  "LN_2": Decimal(2).ln()}
    for name, value in exactv.items():
        assert np.float32(exact.round_frac(Fraction(value), exact.F32)) == globals()[name], name


_check_constants()


def f32(x):
    with np.errstate(over="ignore", invalid="ignore"):
        return np.asarray(x, dtype=np.float32)


def ftz(x):
    x = f32(x)
    return np.where(np.abs(x) < TINY, np.copysign(np.float32(0), x), x).astype(np.float32)


def neg(x):
    return (-f32(x)).astype(np.float32)


def add(a, b):
    with np.errstate(over="ignore", invalid="ignore"):
        return ftz(np.add(ftz(a), ftz(b), dtype=np.float32))


def sub(a, b):
    with np.errstate(over="ignore", invalid="ignore"):
        return ftz(np.subtract(ftz(a), ftz(b), dtype=np.float32))


def mul(a, b):
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        return ftz(np.multiply(ftz(a), ftz(b), dtype=np.float32))


def _round32(s, err):
    """Round the exact real s + err (s float64, err its exact float64 error term, |err| < ulp64(s)/2... or 0) to
    binary32 with gradual underflow, ties to even; err decides exact midpoints."""
    s = np.asarray(s, dtype=np.float64)
    err = np.broadcast_to(np.asarray(err, dtype=np.float64), s.shape)
    with np.errstate(over="ignore", invalid="ignore"):
        r = s.astype(np.float32)                          # RNE; wrong only when s is exactly a midpoint
    fin = np.isfinite(s) & np.isfinite(r.astype(np.float64))
    r64 = r.astype(np.float64)
    # neighbour on the other side of s
    toward = np.where(s > r64, np.float32(np.inf), np.float32(-np.inf)).astype(np.float32)
    nb = np.nextafter(r, toward).astype(np.float32)
    with np.errstate(invalid="ignore", over="ignore"):
        mid = (r64 + nb.astype(np.float64)) * 0.5
    at_mid = fin & (s == mid) & (err != 0) & (s != r64)
    # s exactly at a midpoint: the true value is s + err; go to the side of err
    with np.errstate(invalid="ignore"):
        go_nb = at_mid & (np.sign(err) == np.sign(nb.astype(np.float64) - r64))
    return np.where(go_nb, nb, r).astype(np.float32)


def fma(a, b, c):
    """One rounding of a*b + c (52.3)."""
    a, b, c = ftz(a), ftz(b), ftz(c)
    with np.errstate(over="ignore", invalid="ignore"):
        p = a.astype(np.float64) * b.astype(np.float64)    # exact
        cc = c.astype(np.float64)
        s = p + cc
        bb = s - p
        e = (p - (s - bb)) + (cc - bb)
    e = np.where(np.isfinite(s), e, 0.0)
    return ftz(_round32(s, e))


def div(a, b):
    a, b = ftz(a), ftz(b)
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        q = a.astype(np.float64) / b.astype(np.float64)
    return ftz(f32(q))


def rcp(x):
    return div(np.float32(1.0), x)


def sqrt(x):
    x = ftz(x)
    with np.errstate(invalid="ignore"):
        return ftz(f32(np.sqrt(x.astype(np.float64))))


def fmin(a, b):
    return np.fmin(ftz(a), ftz(b)).astype(np.float32)


def fmax(a, b):
    return np.fmax(ftz(a), ftz(b)).astype(np.float32)


def nan0(v):
    v = f32(v)
    return np.where(np.isnan(v), np.float32(0), v).astype(np.float32)


def f16rt(x):
    """Round binary32 values to binary16 (nearest even, subnormals kept, overflow to inf) and widen exactly."""
    from .fmt import round_f16
    return f32(round_f16(f32(x).astype(np.float64)))


def _near_mid(v, r):
    """True where the float64 value v lies within 2^-40 (relative) of a binary32 rounding midpoint next to r."""
    r64 = r.astype(np.float64)
    toward = np.where(v > r64, np.float32(np.inf), np.float32(-np.inf)).astype(np.float32)
    nb = np.nextafter(r, toward).astype(np.float64)
    with np.errstate(invalid="ignore", over="ignore"):
        mid = (r64 + nb) * 0.5
        d = np.abs(v - mid)
    return np.isfinite(v) & np.isfinite(mid) & (d <= np.abs(v) * 2.0 ** -40)


def _log2_slow(x):
    with localcontext() as ctx:
        ctx.prec = 100
        val = Decimal(float(x)).ln() / Decimal(2).ln()
    fv = Fraction(val)
    if exact.is_midpoint_close(abs(fv), exact.F32, Fraction(1, 10 ** 90)):
        raise ArithmeticError("log2: undecidable rounding")
    return exact.round_frac(fv, exact.F32)


def log2(x):
    """Correctly rounded log2 of binary32 values; log2(0) = -inf, log2(+inf) = +inf, negative / NaN -> NaN."""
    x = ftz(x)
    shape = x.shape
    x = np.atleast_1d(x).astype(np.float32)
    with np.errstate(divide="ignore", invalid="ignore"):
        v = np.log2(x.astype(np.float64))
    with np.errstate(over="ignore", invalid="ignore"):
        r = v.astype(np.float32)
    bits = x.view(np.uint32)
    pow2 = ((bits & 0x7FFFFF) == 0) & (x > 0) & np.isfinite(x)        # powers of two: exact integers
    r = np.where(pow2, (((bits >> 23) & 0xFF).astype(np.int64) - 127).astype(np.float32), r).astype(np.float32)
    near = _near_mid(v, r) & ~pow2 & (x > 0)
    for idx in np.argwhere(near):
        idx = tuple(idx)
        SLOW_PATH_COUNT["log2"] += 1
        r[idx] = np.float32(_log2_slow(x[idx]))
    return ftz(r.reshape(shape))


def exp2(t):
    """Correctly rounded 2^t of binary32 values, then the flush (2^t below the smallest normal gives +0)."""
    t = ftz(t)
    shape = t.shape
    t = np.atleast_1d(t).astype(np.float32)
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        v = np.exp2(t.astype(np.float64))
        r = v.astype(np.float32)
    integral = np.isfinite(t) & (t == np.floor(t))
    near = _near_mid(v, r) & ~integral
    for idx in np.argwhere(near):
        idx = tuple(idx)
        SLOW_PATH_COUNT["exp2"] += 1
        r[idx] = np.float32(exact.exp2_correct_f32(float(t[idx])))
    return ftz(r.reshape(shape))


def sel(cond, a, b):
    return np.where(cond, f32(a), f32(b)).astype(np.float32)
