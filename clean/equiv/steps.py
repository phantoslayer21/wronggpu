# SPDX-License-Identifier: MIT
"""Steps: one representable value of a stage's storage format (clean/EQUIVALENCE-PROTOCOL.md, "Wording").

Every supported format is mapped to an ordinal: an integer per stored value such that two neighbouring representable
values differ by exactly 1 and positive and negative zero are the same ordinal. The step difference of two values is
then the difference of their ordinals. Non-finite values (NaN, +-inf; FP8 E4M3 has only NaN) get no ordinal; they are
classified separately ("nan", "+inf", "-inf").

Formats
  fp8      FP8 E4M3 (no infinities, NaN = S.1111.111): uint8 codes, or float values that are exactly representable
  fp16     IEEE binary16: float16 arrays, uint16 bit patterns, or float values that are exactly representable
  fp32     IEEE binary32: float32 arrays, uint32 bit patterns, or float64 values that are exactly representable
  int      any integer grid given as integer codes (8-, 10-bit or other, signed or unsigned)
  unormN   N-bit unsigned normalised grid (N = 8, 10, 16): integer codes, or floats in [0, 1] on the grid
  snormN   N-bit signed normalised grid: integer codes, or floats in [-1, 1] on the grid
"""
import re

import numpy as np

from clean.reference.fp8 import DECODE as _FP8_DECODE

FORMATS = ("fp8", "fp16", "fp32", "int", "unorm8", "unorm10", "unorm16", "snorm8", "snorm10", "snorm16")

# Non-finite classes (ordinal undefined)
NF_NONE, NF_NAN, NF_PINF, NF_NINF = 0, 1, 2, 3


class FormatError(ValueError):
    pass


def _check_format(fmt):
    if fmt not in FORMATS:
        raise FormatError(f"unknown format {fmt!r} (known: {', '.join(FORMATS)})")


def _float_bits(x, ftype, utype, name):
    a = np.asarray(x)
    if a.dtype == ftype:
        return a.view(utype)
    if a.dtype == utype:
        return a
    if a.dtype.kind == "f":
        c = a.astype(ftype)
        same = (c.astype(np.float64) == a.astype(np.float64)) | (np.isnan(a) & np.isnan(c))
        if not np.all(same):
            raise FormatError(f"{name}: {int(np.size(same) - np.count_nonzero(same))} value(s) are not representable")
        return c.view(utype)
    raise FormatError(f"{name}: dtype {a.dtype} is neither the float type nor its bit pattern")


def _sign_mag(bits, sign_bit, mag_mask):
    b = bits.astype(np.int64)
    mag = b & mag_mask
    neg = (b >> sign_bit) & 1
    return mag, neg


def ordinals(x, fmt):
    """-> (ord int64 array, nonfinite class int8 array). ord is 0 where the value is non-finite."""
    _check_format(fmt)
    if fmt == "fp8":
        a = np.asarray(x)
        if a.dtype == np.uint8:
            codes = a
        elif a.dtype.kind == "f":
            codes = _fp8_codes_exact(a)
        else:
            raise FormatError(f"fp8: dtype {a.dtype}: give uint8 codes or exactly representable floats")
        mag, neg = _sign_mag(codes, 7, 0x7F)
        nf = np.where(mag == 0x7F, NF_NAN, NF_NONE).astype(np.int8)
    elif fmt in ("fp16", "fp32"):
        if fmt == "fp16":
            bits = _float_bits(x, np.float16, np.uint16, fmt)
            mag, neg = _sign_mag(bits, 15, 0x7FFF)
            inf = 0x7C00
        else:
            bits = _float_bits(x, np.float32, np.uint32, fmt)
            mag, neg = _sign_mag(bits, 31, 0x7FFFFFFF)
            inf = 0x7F800000
        nf = np.where(mag > inf, NF_NAN, np.where(mag == inf, np.where(neg == 1, NF_NINF, NF_PINF), NF_NONE))
        nf = nf.astype(np.int8)
    else:
        mag, neg, nf = _grid(x, fmt)
        return np.where(neg == 1, -mag, mag), nf
    o = np.where(neg == 1, -mag, mag)
    return np.where(nf == NF_NONE, o, 0), nf


def _fp8_codes_exact(a):
    v = a.astype(np.float64)
    fin = np.isfinite(v)
    if np.any(np.isinf(v)):
        raise FormatError("fp8: E4M3 has no infinities")
    mag = np.abs(np.where(fin, v, 0.0))
    pos = _FP8_DECODE[:0x7F]
    idx = np.minimum(np.searchsorted(pos, mag), 0x7E)
    if not np.all(pos[idx] == mag):
        raise FormatError(f"fp8: {int(np.count_nonzero(pos[idx] != mag))} value(s) are not representable")
    codes = np.where(fin, idx, 0x7F) | np.where(np.signbit(v), 0x80, 0)
    return codes.astype(np.uint8)


def _grid(x, fmt):
    a = np.asarray(x)
    nf = np.zeros(a.shape, np.int8)
    if a.dtype.kind in "iub":
        o = a.astype(np.int64)
    elif fmt == "int":
        raise FormatError("int: give integer codes")
    else:
        n = int(re.sub(r"\D", "", fmt))
        scale = (1 << n) - 1 if fmt.startswith("unorm") else (1 << (n - 1)) - 1
        lo = 0.0 if fmt.startswith("unorm") else -1.0
        v = a.astype(np.float64)
        if not np.all(np.isfinite(v)) or np.any(v < lo) or np.any(v > 1.0):
            raise FormatError(f"{fmt}: values outside [{lo:g}, 1] or non-finite")
        s = v * scale
        o = np.rint(s).astype(np.int64)
        if np.any(np.abs(s - o) > 1e-3):
            raise FormatError(f"{fmt}: values are not on the {n}-bit grid")
    return np.abs(o), (o < 0).astype(np.int64), nf


def decoded(x, fmt):
    """Values as float64 (for value-range classes and change images; never printed)."""
    _check_format(fmt)
    a = np.asarray(x)
    if fmt == "fp8":
        if a.dtype == np.uint8:
            return _FP8_DECODE[a]
        return a.astype(np.float64)
    if fmt == "fp16":
        return (a.view(np.float16) if a.dtype == np.uint16 else a).astype(np.float64)
    if fmt == "fp32":
        return (a.view(np.float32) if a.dtype == np.uint32 else a).astype(np.float64)
    if a.dtype.kind in "iub" and fmt != "int":
        n = int(re.sub(r"\D", "", fmt))
        scale = (1 << n) - 1 if fmt.startswith("unorm") else (1 << (n - 1)) - 1
        return a.astype(np.float64) / scale
    return a.astype(np.float64)


def min_normal(fmt):
    return {"fp8": 2.0 ** -6, "fp16": 2.0 ** -14, "fp32": 2.0 ** -126}.get(fmt)


def range_classes(x, fmt):
    """Value-range class per element (the format's binades): int array of class keys, and a key -> label function.

    Float formats: 0 = zero, 1 = subnormal, 2 = non-finite, 100 + e = |v| in [2^e, 2^(e+1)).
    Integer grids (codes): 0 = zero, 100 + k = |code| in [2^(k-1), 2^k) (k = bit length of |code|).
    """
    if fmt in ("fp8", "fp16", "fp32"):
        v = decoded(x, fmt)
        a = np.abs(v)
        with np.errstate(invalid="ignore"):
            _, e = np.frexp(np.where(np.isfinite(a), a, 1.0))
        key = 100 + (e.astype(np.int64) - 1)
        key = np.where(a < min_normal(fmt), 1, key)
        key = np.where(a == 0, 0, key)
        key = np.where(~np.isfinite(a), 2, key)
        return key, _float_label
    o, nf = ordinals(x, fmt)
    m = np.abs(o)
    k = np.zeros(m.shape, np.int64)
    t = m.copy()
    while np.any(t):
        k += (t > 0)
        t >>= 1
    return np.where(m == 0, 0, 100 + k), _int_label


def _float_label(key):
    if key == 0:
        return "zero"
    if key == 1:
        return "subnormal"
    if key == 2:
        return "non-finite"
    e = key - 100
    return f"|value| in [2^{e}, 2^{e + 1})"


def _int_label(key):
    if key == 0:
        return "code 0"
    k = key - 100
    return f"|code| in [2^{k - 1}, 2^{k})"
