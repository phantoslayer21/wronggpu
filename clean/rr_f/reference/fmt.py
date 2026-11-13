# SPDX-License-Identifier: MIT
"""Number formats and primitive operations of spec 51.1 (RR preset F window block).

Representation used throughout the reference:
- e4m3 tensors are uint8 code arrays (byte = 128*s + 8*e + m);
- f16 values are carried as float64 arrays holding exactly representable f16 values (or +-inf / NaN);
  `to_f16_bits` / `from_f16_bits` convert to and from the stored 16-bit patterns.

Every rounding R(x) of an exact real x is done by `round_f16(s, err)`, where the exact value is s + err with s a
float64 and err the exact error term of an error-free transform (0 when s is exact). The function rounds to nearest
even, keeps subnormals, sends values from 65520 up to inf (IEEE), and uses err only where s lies exactly on an f16
rounding midpoint (the only place where it can change the result).
"""
import numpy as np

F16_MAX = 65504.0
E4M3_MAX = 448.0

# ---------------------------------------------------------------------------------------------------------------
# e4m3

def _decode_table():
    t = np.empty(256, dtype=np.float64)
    for code in range(256):
        s = code >> 7
        e = (code >> 3) & 0xF
        m = code & 7
        if e == 15 and m == 7:
            v = np.nan
        elif e == 0:
            v = (m / 8.0) * 2.0 ** -6
        else:
            v = (1.0 + m / 8.0) * 2.0 ** (e - 7)
        t[code] = -v if s else v
    t[0x80] = -0.0
    return t


D8_TABLE = _decode_table()                     # float64 value of every code (NaN for 0x7F, 0xFF; -0 for 0x80)
_POS_VALUES = D8_TABLE[:0x7F].copy()           # codes 0x00..0x7E, ascending values 0 .. 448


def d8(codes):
    """Decoded value of e4m3 codes, as float64 (exact; equal to the f16 value of spec 51.1)."""
    return D8_TABLE[np.asarray(codes, dtype=np.uint8)]


def q8(h):
    """Q8 of spec 51.1: f16 or f32 values (given as float64 or any float array) -> e4m3 codes (uint8).

    Finite: round to nearest on the e4m3 grid, ties to the even code; magnitudes above 448 give 448.
    +-inf: +-448. NaN: the NaN code with the sign of h. The result keeps the sign of h (also for zeros and for
    negative values that round to zero)."""
    h = np.asarray(h, dtype=np.float64)
    neg = np.signbit(h)
    a = np.abs(h)
    nan = np.isnan(h)
    a_ok = np.where(nan, 0.0, a)
    sat = a_ok > E4M3_MAX
    a_ok = np.where(sat, 0.0, a_ok)
    # binade exponent E with 2^E <= a < 2^(E+1); frexp gives a = f * 2^ex with 0.5 <= f < 1
    _, ex = np.frexp(a_ok)
    E = ex.astype(np.int64) - 1
    qe = np.maximum(E, -6) - 3                  # quantum 2^qe (subnormal quantum 2^-9)
    qe = np.where(a_ok == 0.0, -9, qe)
    q = np.ldexp(a_ok, -qe)                     # exact
    n = np.floor(q)
    frac = q - n
    up = (frac > 0.5) | ((frac == 0.5) & (np.mod(n, 2.0) == 1.0))
    n = n + up
    val = np.ldexp(n, qe)
    idx = np.searchsorted(_POS_VALUES, val)     # exact match: val is on the grid
    idx = np.where(sat, 0x7E, idx)
    codes = idx.astype(np.int64)
    codes = np.where(nan, 0x7F, codes)
    codes = codes | (neg.astype(np.int64) << 7)
    return codes.astype(np.uint8)


def q8v(h):
    """Value of Q8(h) (float64), i.e. d8(Q8(h))."""
    return D8_TABLE[q8(h)]


# ---------------------------------------------------------------------------------------------------------------
# f16

def round_f16(s, err=None):
    """R(s + err): exact real given as float64 s plus exact float64 error err (|err| <= half an ulp of s).
    Returns float64 holding the f16 result (inf for overflow, NaN for NaN, signed zeros kept)."""
    s = np.asarray(s, dtype=np.float64)
    if err is None:
        err = np.zeros_like(s)
    else:
        err = np.broadcast_to(np.asarray(err, dtype=np.float64), s.shape)
    neg = np.signbit(s)
    a = np.abs(s)
    fin = np.isfinite(s)
    a_ok = np.where(fin, a, 0.0)
    _, ex = np.frexp(a_ok)
    E = ex.astype(np.int64) - 1
    E = np.minimum(E, 15)                       # anything from 2^16 up is handled with the top binade's quantum
    qe = np.maximum(E, -14) - 10
    qe = np.where(a_ok == 0.0, -24, qe)
    q = np.ldexp(a_ok, -qe)
    n = np.floor(q)
    frac = q - n
    # direction of the exact error relative to |s|
    es = np.sign(err) * np.where(neg, -1.0, 1.0)
    es = np.where(np.isfinite(es), es, 0.0)
    up = (frac > 0.5) | ((frac == 0.5) & ((es > 0) | ((es == 0) & (np.mod(n, 2.0) == 1.0))))
    n = n + up
    val = np.ldexp(n, qe)
    val = np.where(val > F16_MAX, np.inf, val)
    val = np.where(fin, val, a)                 # inf stays inf, NaN stays NaN
    return np.where(neg, -val, val)


def two_sum(a, b):
    """Error-free transform: a + b = s + e exactly (float64, no overflow). e = 0 where s is not finite."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    with np.errstate(invalid="ignore"):
        s = a + b
        bb = s - a
        e = (a - (s - bb)) + (b - bb)
    e = np.where(np.isfinite(s), e, 0.0)
    return s, e


def fma16(a, b, c):
    """R(a*b + c) for f16 values a, b, c (one rounding). a*b is exact in float64."""
    with np.errstate(invalid="ignore"):
        p = np.asarray(a, dtype=np.float64) * np.asarray(b, dtype=np.float64)
    s, e = two_sum(p, c)
    return round_f16(s, e)


def mul16(a, b):
    with np.errstate(invalid="ignore"):
        return round_f16(np.asarray(a, dtype=np.float64) * np.asarray(b, dtype=np.float64))


def add16(a, b):
    s, e = two_sum(a, b)
    return round_f16(s, e)


def to_f16_bits(v):
    """float64 holding f16 values -> uint16 bit patterns (exact; NaN -> quiet NaN with the sign of v)."""
    v = np.asarray(v, dtype=np.float64)
    nan = np.isnan(v)
    h = np.where(nan, 0.0, v).astype(np.float16)
    if not np.array_equal(h.astype(np.float64), np.where(nan, 0.0, v)):
        raise ValueError("to_f16_bits: value not representable in f16")
    bits = h.view(np.uint16).copy()
    nanbits = np.where(np.signbit(v), 0xFE00, 0x7E00).astype(np.uint16)
    return np.where(nan, nanbits, bits).astype(np.uint16)


def from_f16_bits(bits):
    return np.asarray(bits, dtype=np.uint16).view(np.float16).astype(np.float64)


ALL_F16 = from_f16_bits(np.arange(65536, dtype=np.uint32).astype(np.uint16))   # value of every f16 pattern


def f16_bits_of(v):
    """Bit patterns of f16-exact float64 values, for table lookups (NaN -> 0x7E00 / 0xFE00)."""
    return to_f16_bits(v)


def f16(x):
    """Round Python/NumPy numbers (exact as float64) to f16 values (float64)."""
    return round_f16(np.asarray(x, dtype=np.float64))


# ---------------------------------------------------------------------------------------------------------------
# Matrix step (51.1)

def mm(x, W, c0=None):
    """MM(x; W; c0): chain of 32-input matrix steps over the last axis of x.

    x: float64 [..., K] (decoded e4m3 values), W: float64 [..., K, N] (decoded e4m3; leading axes broadcast with
    those of x, for per-window right-hand sides such as the values of step 5), c0: f16 start values [N] or None.
    Each step: D = R(C + sum of 32 products), the sum exact. With e4m3 inputs every partial sum is a multiple of
    2^-18 below 2^23 and the f16 start value a multiple of 2^-24 below 2^16, so the float64 sums are exact whatever
    the summation order; one rounding per step, steps in ascending k. Returns float64 f16 values [..., N]."""
    x = np.asarray(x, dtype=np.float64)
    W = np.asarray(W, dtype=np.float64)
    K = x.shape[-1]
    if K % 32 or W.shape[-2] != K:
        raise ValueError("mm: K must be a multiple of 32 and match W")
    acc = None
    with np.errstate(invalid="ignore", over="ignore"):
        for k0 in range(0, K, 32):
            part = np.matmul(x[..., k0:k0 + 32], W[..., k0:k0 + 32, :])
            part = part + 0.0                   # zero sign of an all-zero sum: +0 (not significant, 51.1)
            if acc is None:
                acc = np.zeros(part.shape) if c0 is None else np.broadcast_to(
                    np.asarray(c0, dtype=np.float64), part.shape)
            acc = round_f16(acc + part)         # exact sum (see above), one rounding
    return acc
