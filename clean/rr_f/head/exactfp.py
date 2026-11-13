# SPDX-License-Identifier: MIT
"""Exact scalar arithmetic for the spec-55 reference (D0-R head), written from the spec text only.

* `rnd16(num, den)`, `rnd32(num, den)`: the exact rational num/den rounded to binary16 / binary32 (round to nearest
  even, subnormals kept, IEEE overflow to +-inf), done with Python integers. Results are Python floats (exact).
* Correctly rounded elementary functions (exp, 2^x, tanh, logistic) evaluated with the `decimal` module at 60+
  significant digits and then rounded once with the integer rounding above. None of these functions has an exact
  rounding midpoint as its value for a finite non-trivial argument (the values are transcendental), so 60 digits
  decide every rounding.
* The spec-literal sigmoid of page 51.4 / 55.1 (`sig_spec`), and the alternative forms page 55.1 compares it with.

Nothing here uses NumPy: these are the slow oracles the vectorised code in `fp.py` is checked against.
"""
import math
from decimal import Decimal, localcontext
from fractions import Fraction

INF = float("inf")
NAN = float("nan")

F16 = (11, -14, 15)   # significand bits incl. the hidden bit, min normal exponent, max exponent
F32 = (24, -126, 127)


def round_rational(num, den, fmt):
    """num/den (integers, den > 0) rounded to the binary format fmt; returns a float (exact). A zero result keeps the
    sign of num (num == 0 gives +0; the caller decides the sign of an exact zero)."""
    p, emin, emax = fmt
    if num == 0:
        return 0.0
    neg = num < 0
    a = -num if neg else num
    e = a.bit_length() - den.bit_length()
    lhs = a << (-e) if e < 0 else a
    rhs = den << e if e >= 0 else den
    if lhs < rhs:
        e -= 1
    e = max(e, emin)
    if e > emax:
        return -INF if neg else INF
    k = p - 1 - e                      # q = a/den * 2^k has p significant bits (fewer when subnormal)
    if k >= 0:
        n, r = divmod(a << k, den)
        d2 = den
    else:
        d2 = den << (-k)
        n, r = divmod(a, d2)
    if 2 * r > d2 or (2 * r == d2 and (n & 1)):
        n += 1
    v = math.ldexp(n, -k) if n else 0.0
    if v >= 2.0 ** (emax + 1):
        v = INF
    return -v if neg else v


def ratio(x):
    """Exact (num, den) of a finite float, Fraction, int or Decimal."""
    if isinstance(x, Fraction):
        return x.numerator, x.denominator
    if isinstance(x, int):
        return x, 1
    return x.as_integer_ratio()


def rnd16(num, den=1):
    return round_rational(num, den, F16)


def rnd32(num, den=1):
    return round_rational(num, den, F32)


def r16(x):
    """Round an exact value (float / Fraction / Decimal, finite or not) to binary16. Keeps the sign of zero and of
    infinities; NaN stays NaN."""
    if isinstance(x, float):
        if math.isnan(x) or math.isinf(x):
            return x
        if x == 0.0:
            return x
    n, d = ratio(x)
    return rnd16(n, d)


def r32(x):
    if isinstance(x, float):
        if math.isnan(x) or math.isinf(x):
            return x
        if x == 0.0:
            return x
    n, d = ratio(x)
    return rnd32(n, d)


def is_neg(x):
    return math.copysign(1.0, x) < 0


# ---------------------------------------------------------------------------------------------------- IEEE ops
def exact_sum_round(terms, rnd, zero_sign_neg=False):
    """Round the exact sum of finite float terms with rnd; an exact zero gets -0 only if zero_sign_neg."""
    s = Fraction(0)
    for t in terms:
        s += Fraction(t)
    if s == 0:
        return -0.0 if zero_sign_neg else 0.0
    return rnd(s.numerator, s.denominator)


def fma(a, b, c, rnd=rnd16):
    """R(a*b + c) with one rounding, IEEE specials (float inputs)."""
    if not (math.isfinite(a) and math.isfinite(b) and math.isfinite(c)):
        v = a * b + c          # IEEE rules for the special cases (inf*0 = NaN, inf - inf = NaN, ...)
        return v
    p = Fraction(a) * Fraction(b)
    s = p + Fraction(c)
    if s == 0:
        pneg = (is_neg(a) != is_neg(b))
        neg = (p == 0 and c == 0 and pneg and is_neg(c))
        return -0.0 if neg else 0.0
    return rnd(s.numerator, s.denominator)


def mul(a, b, rnd=rnd16):
    if not (math.isfinite(a) and math.isfinite(b)):
        return a * b
    p = Fraction(a) * Fraction(b)
    if p == 0:
        return -0.0 if is_neg(a) != is_neg(b) else 0.0
    return rnd(p.numerator, p.denominator)


def add(a, b, rnd=rnd16):
    return fma(1.0, a, b, rnd)


def div(a, b, rnd=rnd32):
    if not (math.isfinite(a) and math.isfinite(b)) or b == 0:
        try:
            return a / b
        except ZeroDivisionError:
            if a == 0 or math.isnan(a):
                return NAN
            return math.copysign(INF, a) * math.copysign(1.0, b)
    q = Fraction(a) / Fraction(b)
    if q == 0:
        return -0.0 if is_neg(a) != is_neg(b) else 0.0
    return rnd(q.numerator, q.denominator)


def fmin(a, b):
    """min returning the other operand when exactly one is NaN (page 51.1)."""
    if math.isnan(a):
        return b
    if math.isnan(b):
        return a
    return a if a <= b else b


def fmax(a, b):
    if math.isnan(a):
        return b
    if math.isnan(b):
        return a
    return a if a >= b else b


# ------------------------------------------------------------------------------------- elementary functions
PREC = 60


def _dec(x):
    return Decimal(x) if not isinstance(x, Decimal) else x


def _ln2():
    with localcontext() as ctx:
        ctx.prec = PREC + 10
        return Decimal(2).ln()


LN2 = _ln2()


def exp_cr(x, fmt=F32):
    """Correctly rounded e^x of a float x in the format fmt."""
    if math.isnan(x):
        return NAN
    if x == INF:
        return INF
    if x == -INF:
        return 0.0
    if x > 100:          # e^100 > 2^144: above every format used here
        return INF
    if x < -120:         # e^-120 < 2^-173: rounds to +0 in binary16 and binary32
        return 0.0
    with localcontext() as ctx:
        ctx.prec = PREC
        D = _dec(x).exp()
    n, d = D.as_integer_ratio()
    return round_rational(n, d, fmt)


def exp2_cr(t, fmt=F32):
    """Correctly rounded 2^t of a float t."""
    if math.isnan(t):
        return NAN
    if t == INF:
        return INF
    if t == -INF:
        return 0.0
    if t > 200:
        return INF
    if t < -200:         # 2^-200 rounds to +0 in binary16 and binary32
        return 0.0
    with localcontext() as ctx:
        ctx.prec = PREC
        D = (_dec(t) * LN2).exp()
    n, d = D.as_integer_ratio()
    return round_rational(n, d, fmt)


def tanh_cr(v, fmt=F16):
    """Correctly rounded tanh of a float v; tanh(+-0) = +-0, tanh(+-inf) = +-1."""
    if math.isnan(v):
        return NAN
    if v == 0:
        return v
    if math.isinf(v):
        return math.copysign(1.0, v)
    a = abs(v)
    if a > 40:                          # 1 - tanh(40) < 1e-34: rounds to 1 in binary16 and binary32
        return math.copysign(1.0, v)
    with localcontext() as ctx:
        ctx.prec = PREC
        E = (2 * _dec(a)).exp()
        T = (E - 1) / (E + 1)
    n, d = T.as_integer_ratio()
    r = round_rational(n, d, fmt)
    return -r if v < 0 else r


def logistic_cr(v, fmt=F16, scale_half=False):
    """Correctly rounded 1/(1 + e^-v) (or half of it with scale_half) of a float v."""
    if math.isnan(v):
        return NAN
    h = 2 if scale_half else 1
    if v == INF:
        return 1.0 / h
    if v == -INF:
        return 0.0
    if v > 200:     # 1 - logistic < 1e-86: rounds to 1 (or 1/2)
        return 1.0 / h
    if v < -2000:   # < 1e-868: rounds to +0 in both formats
        return 0.0
    with localcontext() as ctx:
        ctx.prec = PREC
        L = 1 / (h * (1 + (-_dec(v)).exp()))
    n, d = L.as_integer_ratio()
    return round_rational(n, d, fmt)


def _l32():
    with localcontext() as ctx:
        ctx.prec = PREC + 10
        L = 1 / LN2                      # log2(e)
    n, d = L.as_integer_ratio()
    return rnd32(n, d)


L32 = _l32()   # fl32(log2(e)), page 51.4


def sig_spec(x):
    """The sigmoid of page 51.4 for an f16 value x (a float), rounded to f16: t = fl32(x * (-L)), q = fl32(2^t)
    correctly rounded, d = fl32(1 + q), s = fl32(1/d) correctly rounded, R(s)."""
    if math.isnan(x):
        return NAN
    t = mul(x, -L32, rnd32)
    q = exp2_cr(t, F32)
    d = add(1.0, q, rnd32)
    s = div(1.0, d, rnd32)
    return r16(s)


def sig_expf_form(x):
    """Page 55.1 comparison form: R(fl32(1 / fl32(1 + fl32(e^-x)))) with a correctly rounded exponential and
    division."""
    if math.isnan(x):
        return NAN
    q = exp_cr(-x, F32)
    d = add(1.0, q, rnd32)
    s = div(1.0, d, rnd32)
    return r16(s)


# ---------------------------------------------------------------------------------------- 8-bit formats
def e4m3_value(code):
    """d8 of page 51.1 (a float; NaN for the two NaN codes)."""
    s = -1.0 if code & 0x80 else 1.0
    e = (code >> 3) & 15
    m = code & 7
    if e == 15 and m == 7:
        return math.copysign(NAN, s)
    if e == 0:
        return s * math.ldexp(m, -9)
    return s * math.ldexp(8 + m, e - 10)


def q8(h):
    """Q8 of page 51.1 for a float h (an f16 or f32 value): round to nearest on the e4m3 grid, ties to the even code,
    saturating at 448, +-inf -> +-448, NaN -> NaN code with the sign of h, the sign of zero kept."""
    sign = 0x80 if is_neg(h) else 0
    if math.isnan(h):
        return sign | 0x7F
    a = abs(h)
    if a >= 448:
        return sign | 0x7E
    best = None
    for code in range(0, 0x7F):          # the 127 non-negative finite magnitudes, increasing with the code
        v = e4m3_value(code)
        dist = abs(Fraction(v) - Fraction(a))
        if best is None or dist < best[0] or (dist == best[0] and code % 2 == 0):
            best = (dist, code)
    return sign | best[1]


def unorm8_code(v):
    """Store rule of page 55.3 step 2 for an f16 value v."""
    if math.isnan(v):
        return 0
    c = min(max(v, 0.0), 1.0)
    x = Fraction(c) * 255
    n = math.floor(x)
    r = x - n
    if r > Fraction(1, 2) or (r == Fraction(1, 2) and n % 2 == 1):
        n += 1
    return int(n)


def snorm8_code(v):
    """HID store rule (page 53.3 as quoted on page 55.3): clamp to [-1, 1], scale by 127, nearest integer; NaN -> 0.
    Half-way cases (+-63.5 only) round away from zero, which equals round half to even there."""
    if math.isnan(v):
        return 0
    c = min(max(v, -1.0), 1.0)
    x = Fraction(c) * 127
    n = math.floor(abs(x))
    r = abs(x) - n
    if r >= Fraction(1, 2):
        n += 1
    return int(-n if x < 0 else n)


def unorm8_read(k):
    """Read-back of page 55.3: the f16 value nearest to k/255."""
    return rnd16(k, 255)


def f16_bits(v):
    """binary16 bit pattern of a float that is exactly an f16 value (NaN -> 0x7E00 with its sign)."""
    import struct
    if math.isnan(v):
        return (0xFE00 if is_neg(v) else 0x7E00)
    return struct.unpack("<H", struct.pack("<e", v))[0]


def f16_from_bits(b):
    import struct
    return struct.unpack("<e", struct.pack("<H", b))[0]


def f32_bits(v):
    import struct
    return struct.unpack("<I", struct.pack("<f", v))[0]


def f32_from_bits(b):
    import struct
    return struct.unpack("<f", struct.pack("<I", b))[0]
