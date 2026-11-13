# SPDX-License-Identifier: MIT
"""Exact (rational) rounding to binary floating-point formats, for tables built once over a small domain and for the
slow test oracles. Pure Python (fractions.Fraction)."""
from fractions import Fraction
import functools
import math

INF = float("inf")


class Fmt:
    def __init__(self, name, p, emin, max_finite, saturate):
        self.name = name
        self.p = p                      # significand bits including the hidden bit
        self.emin = emin                # exponent of the smallest normal
        self.max_finite = Fraction(max_finite)
        self.saturate = saturate        # True: magnitudes above max give max (e4m3 Q8); False: IEEE overflow


F16 = Fmt("f16", 11, -14, 65504, False)
F32 = Fmt("f32", 24, -126, (2 - Fraction(1, 2 ** 23)) * Fraction(2) ** 127, False)
E4M3 = Fmt("e4m3", 4, -6, 448, True)


def floor_log2(a):
    """floor(log2(a)) for a positive Fraction."""
    a = Fraction(a)
    e = a.numerator.bit_length() - a.denominator.bit_length()
    if Fraction(2) ** e > a:
        e -= 1
    elif Fraction(2) ** (e + 1) <= a:
        e += 1
    return e


def round_mag(a, fmt):
    """Round a non-negative Fraction to fmt (nearest, ties to even). Returns Fraction, or INF on IEEE overflow."""
    if a == 0:
        return Fraction(0)
    if fmt.saturate and a > fmt.max_finite:
        return fmt.max_finite
    e = floor_log2(a)
    qe = max(e, fmt.emin) - (fmt.p - 1)
    quantum = Fraction(2) ** qe
    n = a / quantum
    fl = n.numerator // n.denominator
    rem = n - fl
    if rem > Fraction(1, 2) or (rem == Fraction(1, 2) and fl % 2 == 1):
        fl += 1
    v = fl * quantum
    if v > fmt.max_finite:
        return fmt.max_finite if fmt.saturate else INF
    return v


def round_frac(x, fmt):
    """Round a signed Fraction; returns a Python float (exact for f16/f32/e4m3 values), with the sign of x kept for
    results that round to zero (x < 0 -> -0.0)."""
    x = Fraction(x)
    m = round_mag(abs(x), fmt)
    if m == INF:
        return -INF if x < 0 else INF
    v = float(m)
    return -v if x < 0 else v


def to_frac(v):
    """Exact Fraction of a finite float."""
    return Fraction(v)


def is_midpoint_close(x, fmt, rel):
    """True if the positive Fraction x lies within rel (relative) of a rounding midpoint of fmt."""
    x = Fraction(x)
    if x <= 0:
        return False
    e = floor_log2(x)
    qe = max(e, fmt.emin) - (fmt.p - 1)
    quantum = Fraction(2) ** qe
    n = x / quantum
    fl = n.numerator // n.denominator
    mid = (fl + Fraction(1, 2)) * quantum
    # nearest midpoint is fl+1/2 or fl-1/2
    d = min(abs(x - mid), abs(x - (mid - quantum)))
    return d <= rel * x


@functools.lru_cache(maxsize=None)
def log2e_f32():
    """fl32(log2(e)), from a high-precision value (60 decimal digits)."""
    import decimal
    with decimal.localcontext() as ctx:
        ctx.prec = 80
        l2e = 1 / decimal.Decimal(2).ln()
        return round_frac(Fraction(l2e), F32)


def exp2_correct_f32(t):
    """Correctly rounded f32 value of 2^t for an f32 value t (Python float). Returns float (0.0 / inf on
    underflow / overflow)."""
    import decimal
    if math.isnan(t):
        return math.nan
    if t == INF:
        return INF
    if t == -INF:
        return 0.0
    if t >= 129:
        return INF
    if t <= -151:
        return 0.0
    if t == int(t):
        return round_frac(Fraction(2) ** int(t), F32)
    with decimal.localcontext() as ctx:
        ctx.prec = 90
        ctx.Emax = 999999
        ctx.Emin = -999999
        v = (decimal.Decimal(t) * decimal.Decimal(2).ln()).exp()
        fv = Fraction(v)
    # 2^t for non-integer rational t is irrational, so it is never on a midpoint; the 90-digit value decides the
    # rounding unless it lies within 1e-80 (relative) of a midpoint, which we check.
    if is_midpoint_close(fv, F32, Fraction(1, 10 ** 80)):
        raise ArithmeticError("exp2_correct_f32: too close to a midpoint at t=%r" % t)
    return round_frac(fv, F32)
