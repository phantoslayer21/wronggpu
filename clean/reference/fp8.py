# SPDX-License-Identifier: MIT
"""FP8 E4M3FN and FP16 rounding helpers (spec 02, notation section; spec 01, numeric formats).

E4M3FN: 1 sign bit, 4 exponent bits (bias 7), 3 mantissa bits, no infinities, one NaN pattern per sign
(S.1111.111). Largest finite value 448, smallest normal 2^-6, smallest subnormal 2^-9.
"""
import numpy as np

FP8_MAX = 448.0
FP8_MIN_NORMAL_EXP = -6
FP8_MANT_BITS = 3


def q8(x):
    """Round to the nearest FP8 E4M3FN value, ties to even, saturating at +-448. Returns float64.

    NaN stays NaN; +-inf saturate to +-448 (spec 02: "saturating", to be confirmed by test 02-T1).
    """
    x = np.asarray(x, dtype=np.float64)
    a = np.abs(x)
    with np.errstate(invalid="ignore", over="ignore"):
        _, e = np.frexp(a)  # a = m * 2^e with m in [0.5, 1) -> floor(log2 a) = e - 1
        exp = np.maximum(e - 1, FP8_MIN_NORMAL_EXP)
        spacing = np.ldexp(1.0, exp - FP8_MANT_BITS)
        r = np.round(a / spacing) * spacing  # np.round rounds half to even; a / spacing is exact
        r = np.minimum(r, FP8_MAX)
    return np.copysign(r, x)


def f16(x):
    """Round to IEEE half precision (round to nearest even) and return float64."""
    return np.asarray(x, dtype=np.float64).astype(np.float16).astype(np.float64)


def _decode_table():
    t = np.empty(256, dtype=np.float64)
    for c in range(256):
        s = -1.0 if c & 0x80 else 1.0
        e = (c >> 3) & 0xF
        m = c & 0x7
        if e == 0xF and m == 0x7:
            t[c] = np.nan
        elif e == 0:
            t[c] = s * m * 2.0 ** -9
        else:
            t[c] = s * (1.0 + m / 8.0) * 2.0 ** (e - 7)
    return t


DECODE = _decode_table()
# Positive finite codes 0x00..0x7E are increasing in value; used for encoding.
_POS_VALUES = DECODE[:0x7F]


def fp8_decode(b):
    """uint8 array of E4M3FN codes -> float32 values."""
    return DECODE[np.asarray(b, dtype=np.uint8)].astype(np.float32)


def fp8_encode(x):
    """float array -> uint8 E4M3FN codes. Values are first rounded with q8 (so exact values map exactly)."""
    v = q8(x)
    a = np.abs(v)
    code = np.searchsorted(_POS_VALUES, a).astype(np.int64)
    code = np.where(np.isnan(a), 0x7F, np.minimum(code, 0x7E))
    sign = np.where(np.signbit(v), 0x80, 0)
    return (code | sign).astype(np.uint8)


def is_fp8_exact(x):
    x = np.asarray(x, dtype=np.float64)
    return bool(np.all((q8(x) == x) | np.isnan(x)))
