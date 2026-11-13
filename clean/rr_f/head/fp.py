# SPDX-License-Identifier: MIT
"""Vectorised number formats and rounding points for the spec-55 reference (NumPy).

Representation: every f16 quantity is held as a float64 array whose values are exactly binary16 values (or +-inf /
NaN); every rounding to f16 goes through `r16`, which rounds an exact float64 value to binary16 directly (round to
nearest even, subnormals kept, IEEE overflow) without a detour through float32. Exact intermediate values:

* a product of two f16 values (22 significant bits) is exact in float64;
* a matrix step of page 51.1 (32 e4m3 products plus an f16 start value) is exact in float64 in any summation order:
  every partial sum is a multiple of 2^-24 below 2^48 in magnitude;
* `fma16(a, b, c)` = R(a*b + c): the product is exact, the sum is rounded to float64 *to odd* (TwoSum error term
  folded into the last bit), and the round-to-odd value rounds to f16 exactly as the exact sum would (53 >= 11 + 2).

f32 steps of page 55 use NumPy float32 arithmetic (IEEE single, round to nearest even, subnormals kept).
"""
import math

import numpy as np

from . import exactfp as X

f64 = np.float64
f32 = np.float32


# ------------------------------------------------------------------------------------------------ rounding
def r16(x):
    """Exact float64 (or float32) values -> nearest binary16 values (as float64), ties to even, IEEE overflow."""
    x = np.asarray(x, f64)
    with np.errstate(all="ignore"):
        a = np.abs(x)
        _, e = np.frexp(a)                   # a = m * 2^e, 0.5 <= m < 1
        E = np.maximum(e - 1, -14)           # binary16 exponent of a (subnormal range: -14)
        sh = (10 - E).astype(np.int32)
        n = np.rint(np.ldexp(a, sh))         # ulp -> 1; rint = round half to even, exact on float64
        v = np.ldexp(n, -sh)
        v = np.where(v >= 65536.0, np.inf, v)
        v = np.where(np.isinf(a), np.inf, v)
        v = np.copysign(v, x)
        v = np.where(np.isnan(x), x, v)
    return v


def r16_from_f32(x32):
    return r16(np.asarray(x32, f32).astype(f64))


def as_f16(v):
    """float64 array of f16 values -> np.float16 array (exact)."""
    return np.asarray(v, f64).astype(np.float16)


def bits16(v):
    """Bit patterns (uint16) of f16 values held in float64; NaN -> 0x7E00 | sign."""
    h = as_f16(v).view(np.uint16).copy()
    nan = np.isnan(np.asarray(v, f64))
    sign = np.signbit(np.asarray(v, f64))
    h[nan] = np.where(sign[nan], 0xFE00, 0x7E00)
    return h


def from_bits16(b):
    return np.asarray(b, np.uint16).view(np.float16).astype(f64)


def all_f16():
    """All 65,536 binary16 values in bit order, as float64."""
    return from_bits16(np.arange(65536, dtype=np.uint32).astype(np.uint16))


def _sum_round_to_odd(p, c):
    with np.errstate(all="ignore"):
        s = p + c
        bb = s - p
        aa = s - bb
        err = (p - aa) + (c - bb)
        fix = np.isfinite(s) & (err != 0) & ((s.view(np.int64) & 1) == 0)
        if np.any(fix):
            s = np.where(fix, np.nextafter(s, np.where(err > 0, np.inf, -np.inf)), s)
    return s


def fma16(a, b, c):
    """R(a*b + c), one rounding (a, b, c f16 values or small exact constants as float64)."""
    a, b, c = (np.asarray(t, f64) for t in (a, b, c))
    with np.errstate(all="ignore"):
        p = a * b
    return r16(_sum_round_to_odd(p, c))


def mul16(a, b):
    with np.errstate(all="ignore"):
        return r16(np.asarray(a, f64) * np.asarray(b, f64))


def add16(a, b):
    with np.errstate(all="ignore"):
        return r16(np.asarray(a, f64) + np.asarray(b, f64))     # exact for two f16 values


def fmin(a, b):
    return np.fmin(a, b)     # returns the other operand when exactly one is NaN (page 51.1)


def fmax(a, b):
    return np.fmax(a, b)


# ------------------------------------------------------------------------------------------------ e4m3
D8 = np.array([X.e4m3_value(c) for c in range(256)], f64)       # d8 of every code
_POS = D8[:0x7F].copy()                                          # codes 0..126: increasing magnitudes 0 .. 448


def d8(codes):
    return D8[np.asarray(codes, np.uint8)]


def q8(h):
    """Q8 of page 51.1 (vectorised): uint8 codes."""
    x = np.asarray(h, f64)
    nan = np.isnan(x)
    neg = np.signbit(x)
    a = np.where(nan, 0.0, np.minimum(np.abs(x), 448.0))
    hi = np.clip(np.searchsorted(_POS, a, side="left"), 1, 0x7E)  # _POS[hi-1] < a <= _POS[hi] for a > 0
    lo = hi - 1
    dlo = a - _POS[lo]
    dhi = _POS[hi] - a
    pick = np.where(dlo < dhi, lo, np.where(dhi < dlo, hi, np.where(lo % 2 == 0, lo, hi)))
    pick = np.where(a == 0, 0, pick)
    code = pick.astype(np.uint8) | np.where(neg, 0x80, 0).astype(np.uint8)
    code = np.where(nan, np.where(neg, 0xFF, 0x7F), code).astype(np.uint8)
    return code


# ------------------------------------------------------------------------------------------------ 8-bit images
def unorm8_store(v):
    """Page 55.3 store rule, step 2, for f16 values v: NaN -> 0, clamp to [0, 1], round(c * 255) (only tie: 127.5
    -> 128, round half to even)."""
    x = np.asarray(v, f64)
    c = np.where(np.isnan(x), 0.0, np.minimum(np.maximum(x, 0.0), 1.0))
    return np.rint(c * 255.0).astype(np.uint8)


def snorm8_store(v):
    """HID store (page 53.3 as quoted on page 55.3): NaN -> 0, clamp to [-1, 1], round(c * 127) (only ties: +-63.5
    -> +-64; round half to even gives the same there)."""
    x = np.asarray(v, f64)
    c = np.where(np.isnan(x), 0.0, np.minimum(np.maximum(x, -1.0), 1.0))
    return np.rint(c * 127.0).astype(np.int8)


def unorm8_read(k):
    """Read-back of page 55.3: the f16 value nearest to k / 255."""
    return r16(np.asarray(k, f64) / 255.0)


# ------------------------------------------------------------------------------------------------ tables
_LUT = {}


def _lut(name, fn):
    if name not in _LUT:
        _LUT[name] = np.array([fn(float(v)) for v in all_f16()], f64)
    return _LUT[name]


def sig_table():
    """sig of page 51.4 / 55.1 for every f16 input (index = f16 bits), from the exact scalar evaluation."""
    return _lut("sig", X.sig_spec)


def _exp2_16_exact(v):
    """R(2^v) for an f16 value v, correctly rounded (55-ERRATUM-2 CANDW step e)."""
    return X.exp2_cr(v, X.F16)


def _rcp16_exact(v):
    """R(1/v) for an f16 value v, correctly rounded, IEEE specials (55-ERRATUM-2 CANDW step q)."""
    if math.isnan(v):
        return float("nan")
    if v == 0.0:
        return math.copysign(float("inf"), v)
    if math.isinf(v):
        return math.copysign(0.0, v)
    n, d = v.as_integer_ratio()
    r = X.round_rational(d if n > 0 else -d, abs(n), X.F16)
    return r


def exp2_16_table():
    """R(2^v) for every f16 v (index = bits): CANDW of 55-ERRATUM-2."""
    return _lut("exp2_16", _exp2_16_exact)


def rcp16_table():
    """R(1/v) for every f16 v (index = bits): CANDW of 55-ERRATUM-2."""
    return _lut("rcp16", _rcp16_exact)


def tanh16_table():
    """R(tanh(v)) for every f16 v (index = bits), correctly rounded exact tanh (page 55.3, HID)."""
    return _lut("tanh16", lambda v: X.tanh_cr(v, X.F16))


def lookup(table, v16):
    """Apply a 65,536-entry table to f16 values held in float64."""
    return table[bits16(v16)]


def sig16(v16):
    return lookup(sig_table(), v16)


def tanh16(v16):
    return lookup(tanh16_table(), v16)


# ------------------------------------------------------------------------------------------------ f32 functions
def _guarded(v64, rnd, exact_fn, xs, rel=1e-12):
    """Round float64 approximations v64 (relative error far below rel) with rnd; where the rounding is not decided
    by the approximation (a rounding boundary within rel), fall back to the exact scalar function on xs."""
    with np.errstate(all="ignore"):
        r = rnd(v64)
        lo = rnd(v64 * (1 - rel))
        hi = rnd(v64 * (1 + rel))
        bad = np.isfinite(v64) & ((lo != hi) | (lo != r))
    if np.any(bad):
        r = np.array(r, copy=True)
        idx = np.nonzero(bad)
        for i in zip(*idx):
            r[i] = exact_fn(float(xs[i]))
    return r


def _rnd32(v64):
    return np.asarray(v64, f64).astype(f32).astype(f64)


def tanh32(a32):
    """Correctly rounded f32 tanh of f32 values (rho of page 55.3; spec tolerance 1 f32 ulp)."""
    a = np.asarray(a32, f32).astype(f64)
    with np.errstate(all="ignore"):
        t = np.tanh(a)
    return _guarded(t, _rnd32, lambda v: X.tanh_cr(v, X.F32), a).astype(f32)


def logistic_half16(d32):
    """R(sigx(d) / 2) for f32 d: the exact logistic function halved, rounded once to f16 (CANDW, page 55.3)."""
    d = np.asarray(d32, f32).astype(f64)
    with np.errstate(all="ignore"):
        v = 0.5 / (1.0 + np.exp(-d))
    return _guarded(v, r16, lambda x: X.logistic_cr(x, X.F16, scale_half=True), d)


def expf32(h):
    """Correctly rounded single-precision exponential of f32 values."""
    a = np.asarray(h, f32).astype(f64)
    with np.errstate(all="ignore"):
        v = np.exp(a)
    return _guarded(v, _rnd32, lambda x: X.exp_cr(x, X.F32), a).astype(f32)


# ------------------------------------------------------------------------------------------------ f16 steps
def ordinal16(v):
    """Position of f16 values on the ordered line of binary16 values (+0 and -0 both 0; +-inf at +-31744)."""
    b = bits16(v).astype(np.int64)
    mag = b & 0x7FFF
    return np.where(b & 0x8000, -mag, mag)


def step_diff16(a, b):
    """f16 steps between two arrays of f16 values; NaN against NaN is 0, NaN against a number is a large value."""
    a = np.asarray(a, f64)
    b = np.asarray(b, f64)
    na, nb = np.isnan(a), np.isnan(b)
    d = np.abs(ordinal16(np.where(na, 0, a)) - ordinal16(np.where(nb, 0, b)))
    return np.where(na & nb, 0, np.where(na | nb, 1 << 20, d))
