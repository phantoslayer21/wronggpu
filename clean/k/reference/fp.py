# SPDX-License-Identifier: MIT
"""Exact FP32 / FP16 arithmetic helpers for the model K input and output stages (specs 42, 43).

Specs 42 and 43 define their results by FP32 (or FP16) operations in a given order, each rounded to nearest even,
with "fma" meaning one rounding. NumPy float32 / float16 arithmetic rounds every +, -, *, / once (NumPy computes in a
wider type and rounds; for one operation of two p-bit numbers an intermediate of at least 2p + 2 bits makes that
exact, and float64 has 53 >= 2 * 24 + 2). NumPy has no fused multiply-add, so `fma32` / `fma16` compute it exactly:
the product of two binary32 (or binary16) numbers is exact in float64, the sum with the third operand is formed with
an error-free two-sum, and the float64 result is rounded to odd (its last bit set when the sum was inexact) before
the final rounding to the target format. Round-to-odd at 53 bits followed by round-to-nearest at <= 51 bits gives the
correctly rounded result (the classical double-rounding argument; see e.g. Boldo & Melquiond, "Emulation of FMA and
correctly rounded sums: proved algorithms using rounding to odd").

Functions the specs call "the mathematical function rounded to FP32" (log2, ln, 2^x, e^x, x^y) are evaluated in
float64 and rounded once; that is the correctly rounded value except in cases far rarer than the tolerance.
"""
import numpy as np

F32 = np.float32
F16 = np.float16


def f32(x):
    return np.asarray(x, dtype=np.float32)


def h(x):
    """Round to FP16 (nearest even), overflow to +-inf."""
    with np.errstate(over="ignore"):
        return np.asarray(x).astype(np.float16)


def _round_to_odd(hi, lo):
    """hi + lo (exact two-sum pair, float64) rounded to odd in float64."""
    inexact = lo != 0
    bits = hi.view(np.int64)
    even = (bits & 1) == 0
    nudge = inexact & even & np.isfinite(hi)
    toward = np.where(lo > 0, np.inf, -np.inf)
    return np.where(nudge, np.nextafter(hi, toward), hi)


def _fma64_odd(a, b, c):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    c = np.asarray(c, dtype=np.float64)
    with np.errstate(over="ignore", invalid="ignore"):
        p = a * b  # exact for binary32 / binary16 inputs
        s = p + c
        bp = s - c
        lo = (p - bp) + (c - (s - bp))  # two-sum error (exact)
        lo = np.where(np.isfinite(s), lo, 0.0)
    return _round_to_odd(s, lo)


def fma32(a, b, c):
    """RN32(a * b + c) of binary32 inputs."""
    with np.errstate(over="ignore", invalid="ignore"):
        return _fma64_odd(f32(a), f32(b), f32(c)).astype(np.float32)


def fma16(a, b, c):
    """RN16(a * b + c) of binary16 inputs."""
    with np.errstate(over="ignore", invalid="ignore"):
        return _fma64_odd(np.asarray(a, np.float16), np.asarray(b, np.float16),
                          np.asarray(c, np.float16)).astype(np.float16)


def R(t):
    """Round half away from zero (specs 42/43 "R"): trunc(t + sign(t) / 2) with the addition rounded toward zero.
    For binary32 t the exact sum is representable in float64, and truncating it equals truncating its toward-zero
    rounding, so the exact form is used. Returns float32 integers."""
    t = np.asarray(t, dtype=np.float64)
    return (np.sign(t) * np.trunc(np.abs(t) + 0.5)).astype(np.float32)


def rni(x):
    """Round to nearest integer, ties to even (spec 43 "rni"; QUESTIONS.md K-Q8), same dtype."""
    return np.rint(x)


def sat(x):
    """Clamp to [0, 1], NaN -> 0 (same dtype)."""
    x = np.asarray(x)
    return np.where(np.isnan(x), x.dtype.type(0), np.minimum(np.maximum(x, x.dtype.type(0)), x.dtype.type(1)))


def fn32(f, *args):
    """A mathematical function evaluated in float64 on binary32 arguments and rounded to binary32."""
    with np.errstate(all="ignore"):
        return f(*[np.asarray(a, dtype=np.float32).astype(np.float64) for a in args]).astype(np.float32)


def trunc16(x):
    """binary32 -> binary16 by truncation toward zero (spec 43 draft 2 section 7.8): a magnitude above 65504 becomes
    +-infinity (65519 -> inf); at or below 65504 the value is truncated toward zero; NaN stays NaN."""
    x = np.asarray(x, dtype=np.float32)
    a = np.abs(x).astype(np.float64)
    with np.errstate(over="ignore"):
        r = a.astype(np.float16).astype(np.float64)  # nearest
        down = np.where(r > a, np.nextafter(a.astype(np.float16), np.float16(0)).astype(np.float64), r)
    out = np.where(a > 65504.0, np.inf, down)
    out = np.where(np.isnan(a), np.nan, out)
    return np.copysign(out, x.astype(np.float64)).astype(np.float16)


def f32_bits(x):
    return np.asarray(x, dtype=np.float32).view(np.uint32)


def from_f32_bits(b):
    return np.asarray(b, dtype=np.uint32).view(np.float32)
