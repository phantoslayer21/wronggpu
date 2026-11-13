# SPDX-License-Identifier: MIT
"""FP16 helpers for the model K reference (spec 40: FP16 throughout, round to nearest even after every operation).

Two independent ways to round to IEEE half precision:
- `f16(x)`: NumPy's float64 -> float16 conversion (one correctly rounded step, round to nearest even).
- `round_f16_exact(q)`: exact rational arithmetic (fractions.Fraction), written from the IEEE 754 binary16 definition
  (1 sign bit, 5 exponent bits with bias 15, 10 fraction bits; subnormals below 2^-14; overflow to infinity once the
  value reaches 65520 = 65504 + half a step). Used to check the NumPy path, independent of how NumPy rounds.
"""
from fractions import Fraction

import numpy as np

F16_MAX = 65504
_OVERFLOW = Fraction(65520)  # values >= this round to infinity under round-to-nearest-even


def f16(x):
    """Round to FP16 (array or scalar), returned as float16 (overflow gives +-inf, as IEEE rounding does)."""
    with np.errstate(over="ignore"):
        return np.asarray(x, dtype=np.float64).astype(np.float16)


def bits(x):
    """FP16 values -> uint16 bit patterns."""
    return np.asarray(x, dtype=np.float16).view(np.uint16)


def from_bits(b):
    """uint16 bit patterns -> FP16 values."""
    return np.asarray(b, dtype=np.uint16).view(np.float16)


def all_f16():
    """All 65536 FP16 values, index = bit pattern."""
    return from_bits(np.arange(65536, dtype=np.uint32).astype(np.uint16))


def decode_exact(b):
    """Bit pattern -> ("nan",) / ("inf", sign) / ("num", Fraction, sign). sign is -1 or 1 (keeps -0)."""
    b = int(b)
    s = -1 if b & 0x8000 else 1
    e = (b >> 10) & 0x1F
    m = b & 0x3FF
    if e == 0x1F:
        return ("nan",) if m else ("inf", s)
    v = Fraction(m, 1 << 24) if e == 0 else Fraction(1024 + m, 1 << 10) * Fraction(2) ** (e - 15)
    return ("num", v, s)


def round_f16_exact(q, sign_if_zero=1):
    """Exact rational q -> FP16 bit pattern, round to nearest even. A zero result keeps the sign of q (or
    sign_if_zero when q itself is 0)."""
    neg = q < 0 or (q == 0 and sign_if_zero < 0)
    a = -q if q < 0 else q
    sbit = 0x8000 if neg else 0
    if a >= _OVERFLOW:
        return sbit | 0x7C00
    # exponent: largest e with 2^e <= a (normal range e >= -14); subnormal step 2^-24
    if a < Fraction(1, 1 << 14):
        step = Fraction(1, 1 << 24)
        e_field = 0
        base = 0
    else:
        e = a.numerator.bit_length() - a.denominator.bit_length()
        if Fraction(2) ** e > a:
            e -= 1
        elif Fraction(2) ** (e + 1) <= a:
            e += 1
        step = Fraction(2) ** (e - 10)
        e_field = e + 15
        base = 1024
    n = a / step  # number of steps, >= base
    k = n.numerator // n.denominator
    r = n - k
    if r > Fraction(1, 2) or (r == Fraction(1, 2) and k % 2 == 1):
        k += 1
    if e_field == 0:
        return sbit | k  # k == 1024 is exactly the smallest normal (0x0400): the encoding carries over
    m = k - base  # 0 .. 1024; 1024 carries into the exponent
    return sbit | ((e_field << 10) + m)


def steps_apart(a, b):
    """Distance in FP16 steps between two FP16 arrays (ordinal difference of the bit patterns; +0 and -0 are 0 apart).
    NaN anywhere gives a large value (65536)."""
    def ordinal(x):
        u = bits(x).astype(np.int32)
        return np.where(u & 0x8000, -(u & 0x7FFF), u)
    a = np.asarray(a, dtype=np.float16)
    b = np.asarray(b, dtype=np.float16)
    d = np.abs(ordinal(a) - ordinal(b))
    return np.where(np.isnan(a) | np.isnan(b), np.where(np.isnan(a) & np.isnan(b), 0, 65536), d)
