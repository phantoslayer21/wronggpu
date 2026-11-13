# SPDX-License-Identifier: MIT
"""Slow, element-by-element oracles of spec 51, written independently of the vectorised reference (no NumPy in the
arithmetic): a small soft-float on Fractions with IEEE signs and specials, integer-exact matrix steps, and per-token /
per-window loops for the stage geometry. Used only by the tests."""
from fractions import Fraction
import math

from clean.rr_f.reference import exact

F16, F32, E4M3 = exact.F16, exact.F32, exact.E4M3


class SV:
    """Soft value: kind 'f' (finite, q Fraction), 'i' (infinity), 'n' (NaN); neg = sign bit."""
    __slots__ = ("k", "q", "neg")

    def __init__(self, k, q=Fraction(0), neg=False):
        self.k, self.q, self.neg = k, q, neg

    @staticmethod
    def of(x):
        x = float(x)
        if math.isnan(x):
            return SV("n", neg=math.copysign(1, x) < 0)
        if math.isinf(x):
            return SV("i", neg=x < 0)
        return SV("f", Fraction(x), math.copysign(1, x) < 0)

    def float(self):
        if self.k == "n":
            return math.nan
        if self.k == "i":
            return -math.inf if self.neg else math.inf
        if self.q == 0:
            return -0.0 if self.neg else 0.0
        return float(self.q)   # exact for values of the formats used here


def s_mul(a, b):
    if a.k == "n" or b.k == "n":
        return SV("n")
    neg = a.neg != b.neg
    if a.k == "i" or b.k == "i":
        other = b if a.k == "i" else a
        if other.k == "f" and other.q == 0:
            return SV("n")
        return SV("i", neg=neg)
    return SV("f", a.q * b.q, neg)


def s_add(a, b):
    if a.k == "n" or b.k == "n":
        return SV("n")
    if a.k == "i" and b.k == "i":
        return SV("n") if a.neg != b.neg else SV("i", neg=a.neg)
    if a.k == "i":
        return a
    if b.k == "i":
        return b
    q = a.q + b.q
    if q == 0:
        both_neg_zero = a.q == 0 and b.q == 0 and a.neg and b.neg
        return SV("f", Fraction(0), both_neg_zero)
    return SV("f", q, q < 0)


def s_negate(a):
    return SV(a.k, -a.q, not a.neg)


def s_abs(a):
    return SV(a.k, abs(a.q), False)


def s_round(a, fmt=F16):
    if a.k != "f":
        return a
    if a.q == 0:
        return SV("f", Fraction(0), a.neg)
    return SV.of(exact.round_frac(a.q, fmt))


def s_fmax(a, b):
    if a.k == "n":
        return b
    if b.k == "n":
        return a
    return a if a.float() >= b.float() else b


def s_fmin(a, b):
    if a.k == "n":
        return b
    if b.k == "n":
        return a
    return a if a.float() <= b.float() else b


def fma(a, b, c):
    return s_round(s_add(s_mul(a, b), c))


# ---------------------------------------------------------------------------------------------------------------
# scalar rules

def attention_weight(s, wc):
    """Returns float w, or None when p is outside the specified range."""
    s = SV.of(s)
    u = s_round(s_mul(s, SV.of(wc.A_SCALE)))
    c = s_fmin(s_fmax(u, SV.of(wc.A_CLAMPn)), SV.of(wc.A_CLAMP))
    inner = s_round(s_add(SV.of(wc.A_P1), s_negate(s_mul(c, c))))
    p = fma(c, inner, SV.of(wc.A_P0))
    assert p.k == "f"
    y = 32 * (p.q - 1)
    n = math.floor(y)
    if not (1 <= n <= 30):
        return None
    w = Fraction(2) ** (n - 15) * (1 + y - n)
    assert exact.round_mag(w, F16) == w
    return float(w)


def activation(x, wc):
    xv = SV.of(x)
    k = s_fmin(s_fmax(xv, SV.of(wc.ACT_CLn)), SV.of(wc.ACT_CL))
    u = s_round(s_add(SV.of(wc.ACT_C2), s_negate(s_mul(s_abs(k), SV.of(wc.ACT_C1)))))
    v = fma(k, u, SV.of(wc.ACT_C3))
    return s_round(s_mul(xv, v)).float()


def q8_value(h):
    """Q8 value (float) by its own rule: NaN -> NaN, +-inf -> +-448, saturating, ties to even, sign kept."""
    if math.isnan(h):
        return math.nan
    if math.isinf(h):
        return math.copysign(448.0, h)
    v = exact.round_frac(Fraction(h), E4M3)
    return math.copysign(abs(v), h)


def r16(x):
    """R(x) of an exact Fraction (finite)."""
    return exact.round_frac(x, F16)


# ---------------------------------------------------------------------------------------------------------------
# window block, integer-exact matrix steps

def _i9(v):
    """e4m3 value as an integer multiple of 2^-9 (None for NaN)."""
    if math.isnan(v):
        return None
    n = Fraction(v) * 512
    assert n.denominator == 1
    return int(n)


def _chain(ri, col, c0):
    """One output of MM: ri, col integer lists (None = NaN), c0 float or None. Exact per-step sums, IEEE specials
    through the soft-float. Returns float (f16)."""
    acc = SV.of(0.0 if c0 is None else c0)
    for k0 in range(0, len(ri), 32):
        s = 0
        nan = False
        for k in range(k0, k0 + 32):
            a, b = ri[k], col[k]
            if a is None or b is None:
                nan = True
            else:
                s += a * b
        part = SV("n") if nan else SV("f", Fraction(s, 1 << 18), False)
        acc = s_round(s_add(acc, part))
    return acc.float()


def mm_step_chain(xs, wcol, c0):
    """MM over one output: xs, wcol lists of e4m3 floats (length K), c0 float or None. Returns float (f16)."""
    return _chain([_i9(v) for v in xs], [_i9(v) for v in wcol], c0)


def mm_rows(X, W, c0):
    """X: list of rows (lists of floats), W: K x N nested list (floats), c0: list or None."""
    K = len(W)
    N = len(W[0])
    Wi = [[_i9(W[k][n]) for k in range(K)] for n in range(N)]
    out = []
    for row in X:
        ri = [_i9(v) for v in row]
        out.append([_chain(ri, Wi[n], None if c0 is None else c0[n]) for n in range(N)])
    return out


STORED = [(0, 0), (1, 0), (0, 1), (1, 1), (0, 2), (1, 2), (0, 3), (1, 3), (2, 2), (3, 2), (2, 3), (3, 3)]


def bias(Tb_h, a, b):
    I, J = a // 16, b // 16
    if (I, J) in STORED:
        return float(Tb_h[STORED.index((I, J))][a % 16][b % 16])
    return float(Tb_h[STORED.index((J, I))][b % 16][a % 16])


def window_block(xv, xr, W, ra, m, wc):
    """xv: 64 x C e4m3 values (floats), xr: 64 x C f16 floats. W: dict of nested lists of floats (decoded).
    Returns (y2 values 64 x C, Z [H][64])."""
    H = len(W["Wq"])
    C = len(xv[0])
    Os = []
    Zs = []
    for h in range(H):
        q = [[q8_value(v) for v in row] for row in mm_rows(xv, W["Wq"][h], None)]
        v = [[q8_value(vv) for vv in row] for row in mm_rows(xv, W["Wv"][h], None)]
        qi = [[_i9(e) for e in row] for row in q]
        Zh = []
        P = []
        for a in range(64):
            ws = []
            for b in range(64):
                s = _chain(qi[a], qi[b], bias(W["Tb"][h], a, b))
                w = attention_weight(s, wc)
                assert w is not None
                ws.append(w)
            Z = r16(sum(Fraction(w) for w in ws))           # w finite, positive
            r = exact.round_frac(1 / Fraction(Z), F16) if not math.isinf(Z) else 0.0
            P.append([q8_value(r16(Fraction(w) * Fraction(r))) for w in ws])
            Zh.append(Z)
        O = [[q8_value(e) for e in row] for row in mm_rows(P, v, None)]
        Os.append(O)
        Zs.append(Zh)
    Ocat = [sum((Os[h][a] for h in range(H)), []) for a in range(64)]
    Wo = [W["Wo"][h][i] for h in range(H) for i in range(32)]
    Dp = mm_rows(Ocat, Wo, W["bo"])
    y1h = [[fma(SV.of(Dp[a][c]), SV.of(ra), SV.of(xr[a][c])).float() for c in range(C)] for a in range(64)]
    y1q = [[q8_value(e) for e in row] for row in y1h]
    Du = mm_rows(y1q, W["W1"], W["b1"])
    Au = [[q8_value(activation(e, wc)) for e in row] for row in Du]
    Dd = mm_rows(Au, W["W2"], W["b2"])
    y2 = [[q8_value(fma(SV.of(Dd[a][c]), SV.of(m), SV.of(y1h[a][c])).float()) for c in range(C)] for a in range(64)]
    return y2, Zs


# ---------------------------------------------------------------------------------------------------------------
# geometry

def reflect(v, N):
    """Repeated reflection without repeating the edge, applied step by step."""
    if N == 1:
        return 0
    while v < 0 or v >= N:
        if v < 0:
            v = -v
        else:
            v = 2 * (N - 1) - v
    return v


def sigmoid_f64(x):
    """Independent evaluation of 51.4 with float64 helpers (math.exp2 / division in double, then f32 roundings).
    Agrees with the exact table except possibly near f16 midpoints."""
    import struct
    def f32(v):
        return struct.unpack("<f", struct.pack("<f", v))[0] if math.isfinite(v) and abs(v) < 3.4e38 else (
            math.copysign(math.inf, v) if not math.isnan(v) else v)
    if math.isnan(x):
        return math.nan
    L = f32(1.0 / math.log(2.0))
    t = f32(x * -L)
    try:
        q = f32(2.0 ** t)
    except OverflowError:
        q = math.inf
    d = f32(1.0 + q)
    s = f32(1.0 / d) if not math.isinf(d) else 0.0
    return float(exact.round_frac(Fraction(s), F16))
