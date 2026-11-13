# SPDX-License-Identifier: MIT
"""Float-float (pairs of binary32) log2 / exp2 used by the input-stage kernel's fast path (kernels/rr_in_math.hlsli),
as a NumPy model of the shader's operation sequence plus the tables the host uploads.

Correct rounding: the pair (H, L) approximates the exact value with a relative error below EPS_REL; H is the f32
rounding of H + L. The result H is the correctly rounded value unless H + L lies within EPS_REL of a rounding
midpoint; those evaluations are flagged and recomputed by the binary64 path (exact except its own counted cases).

Tables (WrongGPU's own, mathematical values only):
  log2: 256 entries (h, i): c = 1/m_centre rounded to 8 significant bits (c = 1 for the two buckets that contain
        1), T = -log2(c) as a float-float pair; m in [1, sqrt 2) for h = 0, m/2 in [sqrt 2 / 2, 1) for h = 1.
  exp2: 65 entries j: 2^(j/64 - 1/2) as a float-float pair (index j + 32 of j = -32 .. 32).
"""
from decimal import Decimal, localcontext
from fractions import Fraction

import numpy as np

from ..reference import exact

EPS_REL = 2.0 ** -42                         # flag threshold (relative), above the error bound of both functions


def _f32(q):
    return np.float32(exact.round_frac(Fraction(q), exact.F32))


def _pair(q):
    h = _f32(q)
    return h, _f32(Fraction(q) - Fraction(float(h)))


def round8(q):
    """Round a positive Fraction to 8 significant bits (nearest, ties to even)."""
    e = exact.floor_log2(q)
    quantum = Fraction(2) ** (e - 7)
    n = q / quantum
    fl = n.numerator // n.denominator
    rem = n - fl
    if rem > Fraction(1, 2) or (rem == Fraction(1, 2) and fl % 2):
        fl += 1
    return fl * quantum


def log_table():
    """[256][3] f32: c, Th, Tl."""
    t = np.zeros((256, 3), np.float32)
    with localcontext() as ctx:
        ctx.prec = 60
        ln2 = Decimal(2).ln()
        for h in (0, 1):
            for i in range(128):
                centre = 1 + Fraction(2 * i + 1, 256)
                if h:
                    centre /= 2
                c = Fraction(1) if (h == 0 and i == 0) or (h == 1 and i == 127) else round8(1 / centre)
                T = -(Decimal(c.numerator).ln() - Decimal(c.denominator).ln()) / ln2
                t[h * 128 + i, 0] = np.float32(float(c))
                t[h * 128 + i, 1:] = _pair(Fraction(T))
                assert Fraction(float(t[h * 128 + i, 0])) == c
    return t


def exp_table():
    """[65][2] f32: 2^(j/64) for j = -32 .. 32 as float-float pairs."""
    t = np.zeros((65, 2), np.float32)
    with localcontext() as ctx:
        ctx.prec = 60
        for j in range(-32, 33):
            v = (Decimal(j) / 64 * Decimal(2).ln()).exp()
            t[j + 32] = _pair(Fraction(v))
    return t


def _coef(k, kind):
    with localcontext() as ctx:
        ctx.prec = 60
        ln2 = Decimal(2).ln()
        if kind == "log":                     # (-1)^(k+1) / (k ln 2)
            v = Decimal((-1) ** (k + 1)) / (k * ln2)
        else:                                 # ln2^k / k!
            import math
            v = ln2 ** k / math.factorial(k)
        return _pair(Fraction(v))


LOGK = {k: _coef(k, "log") for k in range(1, 8)}
EXPK = {k: _coef(k, "exp") for k in range(1, 6)}


# ---------------------------------------------------------------------------------------------------------------
# f32 primitives of the model (numpy float32 operations are IEEE; fma exact through float64 two-sum)

def fma(a, b, c):
    from ..reference.f32 import _round32
    a, b, c = (np.asarray(v, np.float32) for v in (a, b, c))
    p = a.astype(np.float64) * b.astype(np.float64)
    s = p + c.astype(np.float64)
    bb = s - p
    e = (p - (s - bb)) + (c.astype(np.float64) - bb)
    return _round32(s, np.where(np.isfinite(s), e, 0.0))


def two_sum(a, b):
    s = (a + b).astype(np.float32)
    bb = (s - a).astype(np.float32)
    e = ((a - (s - bb)) + (b - bb)).astype(np.float32)
    return s, e


def near_mid(H, L):
    """True when H + L may lie within EPS_REL of a rounding midpoint next to H (H = RN(H + L))."""
    up = np.nextafter(H, np.float32(np.inf)).astype(np.float32)
    dn = np.nextafter(H, np.float32(-np.inf)).astype(np.float32)
    gap = np.where((L >= 0) == (H >= 0), np.abs(up.astype(np.float64) - H), np.abs(H - dn.astype(np.float64)))
    gap = np.where(L >= 0, up.astype(np.float64) - H, H.astype(np.float64) - dn)
    return np.abs(gap * 0.5 - np.abs(L.astype(np.float64))) <= np.abs(H.astype(np.float64)) * EPS_REL


LOGT = log_table()
EXPT = exp_table()


def log2_ff(x):
    """x: f32 array of positive normal non-power-of-two values. Returns (H, flagged)."""
    x = np.asarray(x, np.float32)
    u = x.view(np.uint32)
    e = ((u >> 23) & 255).astype(np.int64) - 127
    mb = u & 0x7FFFFF
    hi = mb > 0x3504F3
    m = np.where(hi, (mb | 0x3F000000), (mb | 0x3F800000)).astype(np.uint32).view(np.float32)
    e = e + hi
    idx = hi * 128 + (mb >> 16)
    c, Th, Tl = LOGT[idx, 0], LOGT[idx, 1], LOGT[idx, 2]
    r = fma(m, c, np.float32(-1))
    f = np.float32
    K1h, K1l = LOGK[1]
    K2h, K2l = LOGK[2]
    K3h, K3l = LOGK[3]
    Ah = (r * K1h).astype(f)
    Al = (fma(r, K1h, -Ah) + (r * K1l).astype(f)).astype(f)
    R2h = (r * r).astype(f)
    R2l = fma(r, r, -R2h)
    Bh = (R2h * K2h).astype(f)
    Bl = (fma(R2h, K2h, -Bh) + ((R2l * K2h).astype(f) + (R2h * K2l).astype(f)).astype(f)).astype(f)
    R3h = (R2h * r).astype(f)
    R3l = (fma(R2h, r, -R3h) + (R2l * r).astype(f)).astype(f)
    Ch = (R3h * K3h).astype(f)
    Cl = (fma(R3h, K3h, -Ch) + ((R3l * K3h).astype(f) + (R3h * K3l).astype(f)).astype(f)).astype(f)
    P = LOGK[7][0]
    for k in (6, 5, 4):
        P = fma(P, r, LOGK[k][0])
    R4 = (R3h * r).astype(f)
    Dv = (R4 * P).astype(f)
    s, e1 = two_sum(e.astype(f), Th)
    s, e2 = two_sum(s, Ah)
    s, e3 = two_sum(s, Bh)
    s, e4 = two_sum(s, Ch)
    lo = ((e1 + e2).astype(f) + (e3 + e4).astype(f)).astype(f)
    lo = (lo + ((Tl + Al).astype(f) + ((Bl + Cl).astype(f) + Dv).astype(f)).astype(f)).astype(f)
    H = (s + lo).astype(f)
    L = (lo - (H - s).astype(f)).astype(f)
    return H, near_mid(H, L)


def exp2_ff(t):
    """t: f32 array, -126 < t < 128, not integral. Returns (H, flagged)."""
    f = np.float32
    t = np.asarray(t, f)
    n = np.floor(t).astype(f)
    fh, fl = two_sum(t, -n)
    big = fh > 0.5
    fh = np.where(big, (fh - f(1)).astype(f), fh)
    n = np.where(big, n + f(1), n).astype(f)
    j = np.rint((fh * f(64)).astype(f)).astype(np.int64)
    gh = (fh - (j.astype(f) / f(64)).astype(f)).astype(f)      # exact
    gh, gl = two_sum(gh, fl)                                     # renormalised: |gl| <= ulp(gh) / 2
    M1h, M1l = EXPK[1]
    M2h, M2l = EXPK[2]
    G1h = (gh * M1h).astype(f)
    G1l = (fma(gh, M1h, -G1h) + ((gh * M1l).astype(f) + (gl * M1h).astype(f)).astype(f)).astype(f)
    R2h = (gh * gh).astype(f)
    R2l = (fma(gh, gh, -R2h) + (f(2) * (gh * gl).astype(f)).astype(f)).astype(f)
    G2h = (R2h * M2h).astype(f)
    G2l = (fma(R2h, M2h, -G2h) + ((R2l * M2h).astype(f) + (R2h * M2l).astype(f)).astype(f)).astype(f)
    Q = fma(fma(EXPK[5][0], gh, EXPK[4][0]), gh, EXPK[3][0])
    G3 = ((R2h * gh).astype(f) * Q).astype(f)
    s, e1 = two_sum(f(1), G1h)
    s, e2 = two_sum(s, G2h)
    lo = ((e1 + e2).astype(f) + ((G1l + G2l).astype(f) + G3).astype(f)).astype(f)
    Ph = (s + lo).astype(f)
    Pl = (lo - (Ph - s).astype(f)).astype(f)
    Eh, El = EXPT[j + 32, 0], EXPT[j + 32, 1]
    Rh = (Eh * Ph).astype(f)
    Rl = (fma(Eh, Ph, -Rh) + ((Eh * Pl).astype(f) + (El * Ph).astype(f)).astype(f)).astype(f)
    H = (Rh + Rl).astype(f)
    L = (Rl - (H - Rh).astype(f)).astype(f)
    flag = near_mid(H, L)
    scale = np.ldexp(np.float64(1), n.astype(np.int64))
    return (H.astype(np.float64) * scale).astype(f), flag
