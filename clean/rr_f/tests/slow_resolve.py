# SPDX-License-Identifier: MIT
"""Independent slow oracle of page 54b for tiny images: one pixel at a time, exact rationals (Fraction) for
every rounding, Decimal for exp2 / log2. It shares no arithmetic, read, store or sampling helper with
clean/rr_f/reference (only the plain dataclasses of the symbols and surfaces are read through attributes).
"""
from decimal import Decimal, localcontext
from fractions import Fraction as Fr
import math

INF = math.inf
TINY = 2.0 ** -126


# ------------------------------------------------------------------------------------------------ rounding

def _round(x, p, emin, emax):
    """Nearest-even rounding of a nonzero Fraction to a binary format; returns float (inf on overflow)."""
    s = -1 if x < 0 else 1
    a = abs(x)
    e = a.numerator.bit_length() - a.denominator.bit_length()
    if Fr(2) ** e > a:
        e -= 1
    q = Fr(2) ** (max(e, emin) - p + 1)
    n, r = divmod(a / q, 1)
    n = int(n)
    if r > Fr(1, 2) or (r == Fr(1, 2) and n % 2):
        n += 1
    v = n * q
    if v >= Fr(2) ** (emax + 1):
        return s * INF
    return s * float(v)


def R32(x):
    """Exact value (Fraction, or a non-finite float) -> f32 with FTZ of the result."""
    if isinstance(x, float) and not math.isfinite(x):
        return x
    x = Fr(x)
    if x == 0:
        return 0.0
    v = _round(x, 24, -126, 127)
    return math.copysign(0.0, v) if abs(v) < TINY else v


def R16(x):
    if isinstance(x, float) and math.isnan(x):
        return x
    if isinstance(x, float) and math.isinf(x):
        return x
    x = Fr(x)
    if x == 0:
        return 0.0
    return _round(x, 11, -14, 15)


def z(a):
    """FTZ of an input."""
    a = float(a)
    return math.copysign(0.0, a) if abs(a) < TINY else a


def _special(*v):
    return any(not math.isfinite(t) for t in v)


def add(a, b):
    a, b = z(a), z(b)
    if _special(a, b):
        return a + b
    s = Fr(a) + Fr(b)
    if s == 0:
        return -0.0 if (math.copysign(1, a) < 0 and math.copysign(1, b) < 0) else 0.0
    return R32(s)


def sub(a, b):
    return add(a, -z(b))


def mul(a, b):
    a, b = z(a), z(b)
    if _special(a, b):
        return a * b
    p = Fr(a) * Fr(b)
    if p == 0:
        return math.copysign(0.0, a) * math.copysign(1.0, b)
    return R32(p)


def div(a, b):
    a, b = z(a), z(b)
    if _special(a, b) or b == 0:
        if b == 0 and a == 0 or math.isnan(a) or math.isnan(b):
            return math.nan
        if b == 0:
            return math.copysign(INF, a) * math.copysign(1.0, b)
        return a / b
    q = Fr(a) / Fr(b)
    if q == 0:
        return math.copysign(0.0, a) * math.copysign(1.0, b)
    return R32(q)


def fma(a, b, c):
    a, b, c = z(a), z(b), z(c)
    if _special(a, b, c):
        return a * b + c
    p = Fr(a) * Fr(b)
    s = p + Fr(c)
    if s == 0:
        psign = math.copysign(1, a) * math.copysign(1, b)
        return -0.0 if (p == 0 and psign < 0 and math.copysign(1, c) < 0) else 0.0
    return R32(s)


def fma16(a, b, c):
    if _special(a, b, c):
        v = a * b + c
        return v
    s = Fr(a) * Fr(b) + Fr(c)
    if s == 0:
        psign = math.copysign(1, a) * math.copysign(1, b)
        return -0.0 if (Fr(a) * Fr(b) == 0 and psign < 0 and math.copysign(1, c) < 0) else 0.0
    return R16(s)


def exp2(x):
    x = z(x)
    if math.isnan(x):
        return x
    if x == INF:
        return INF
    if x == -INF or x < -160:
        return 0.0
    if x >= 129:
        return INF
    if x == int(x):
        return R32(Fr(2) ** int(x))
    with localcontext() as ctx:
        ctx.prec = 70
        v = (Decimal(x) * Decimal(2).ln()).exp()
    return R32(Fr(v))


def log2(x):
    x = z(x)
    if math.isnan(x) or x < 0:
        return math.nan
    if x == 0:
        return -INF
    if x == INF:
        return INF
    m, e = math.frexp(x)
    if m == 0.5:
        return float(e - 1)
    with localcontext() as ctx:
        ctx.prec = 70
        v = Decimal(x).ln() / Decimal(2).ln()
    return R32(Fr(v))


def rcp(x):
    return div(1.0, x)


def pmin(a, b):
    """Plain min; -0 below +0."""
    if math.isnan(a):
        return b
    if math.isnan(b):
        return a
    if a == b:
        return a if math.copysign(1, a) < 0 else b
    return a if a < b else b


def pmax(a, b):
    if math.isnan(a):
        return b
    if math.isnan(b):
        return a
    if a == b:
        return a if math.copysign(1, a) > 0 else b
    return a if a > b else b


def min_nan(a, b):
    return math.nan if math.isnan(a) or math.isnan(b) else pmin(a, b)


def sat(v):
    return 0.0 if math.isnan(v) else min(max(v, 0.0), 1.0)


def rnd(v):
    """trunc(v + copysign(0.5, v)) with the addition rounded toward zero, by its literal definition."""
    if not math.isfinite(v):
        return v
    neg = math.copysign(1.0, v) < 0
    s = Fr(v) + (-Fr(1, 2) if neg else Fr(1, 2))
    # toward-zero rounding of s to f32
    a = abs(s)
    if a == 0:
        return 0.0
    e = a.numerator.bit_length() - a.denominator.bit_length()
    if Fr(2) ** e > a:
        e -= 1
    q = Fr(2) ** (max(e, -126) - 23)
    t = int(a / q) * q
    return math.copysign(float(math.trunc(t)), -1.0 if neg else 1.0)


# ------------------------------------------------------------------------------------------------ resources

def texel(surface, x, y):
    """Integer fetch with the read conversion; outside the texture all channels 0."""
    d = surface.data
    h, w, c = d.shape
    if not (0 <= x < w and 0 <= y < h):
        return [0.0] * c
    v = d[y, x]
    if surface.format == "f32":
        return [z(t) for t in v]
    if surface.format == "f16":
        return [float(t) for t in v]
    if surface.format == "unorm8":
        return [R32(Fr(int(t), 255)) for t in v]
    return [R32(Fr(int(t), 1023 if k < 3 else 3)) for k, t in enumerate(v)]


def bilinear(read, w, h, u, v):
    """Page 52.1.4 for a texel function read(i, j) -> list."""
    u = 0.0 if math.isnan(u) else min(max(u, 0.0), 1.0)
    v = 0.0 if math.isnan(v) else min(max(v, 0.0), 1.0)
    s = sub(mul(u, float(w)), 0.5)
    t = sub(mul(v, float(h)), 0.5)
    i0, j0 = math.floor(s), math.floor(t)
    fx, fy = s - i0, t - j0
    cl = lambda n, m: min(max(n, 0), m - 1)  # noqa: E731
    a, b = read(cl(i0, w), cl(j0, h)), read(cl(i0 + 1, w), cl(j0, h))
    c, d = read(cl(i0, w), cl(j0 + 1, h)), read(cl(i0 + 1, w), cl(j0 + 1, h))
    out = []
    for k in range(len(a)):
        r0 = add(mul(sub(1.0, fx), a[k]), mul(fx, b[k]))
        r1 = add(mul(sub(1.0, fx), c[k]), mul(fx, d[k]))
        out.append(add(mul(sub(1.0, fy), r0), mul(fy, r1)))
    return out


def unorm(v, n):
    if math.isnan(v):
        return 0
    c = Fr(min(max(v, 0.0), 1.0)) * (2 ** n - 1)
    k, r = divmod(c, 1)
    k = int(k)
    if r > Fr(1, 2) or (r == Fr(1, 2) and k % 2):
        k += 1
    return k


# ------------------------------------------------------------------------------------------------ 54b

LOG2_10 = R32(Fr(Decimal(10).ln() / Decimal(2).ln()))
LOG10_2 = R32(Fr(Decimal(2).ln() / Decimal(10).ln()))


def decode(c, s):
    cp = min_nan(c, float(s.K10)) if (c > s.K5 or math.isnan(c)) else float(s.K5)
    return mul(sub(exp2(mul(mul(cp, float(s.K8)), LOG2_10)), 1.0), float(s.K12))


def plain_pixel(Q, x, y, s):
    """54b.1 steps 1-5 before the division by E: three f32 values."""
    Hh, Wh = len(Q), len(Q[0])
    cx = math.floor(mul(add(float(x), 0.5), 1.5))
    cy = math.floor(mul(add(float(y), 0.5), 1.5))
    we = [float(s.RW_B), float(s.RW_D), float(s.RW_0), float(s.RW_C), float(s.RW_A)]
    wx = we if x % 2 == 0 else we[::-1]
    wy = we if y % 2 == 0 else we[::-1]

    def five(ws, taps):
        acc, mn, mx = 0.0, 65504.0, -65504.0
        for w_, t in zip(ws, taps):
            acc = fma16(w_, t, acc)
            mn, mx = pmin(mn, t), pmax(mx, t)
        return pmin(pmax(acc, mn), mx)

    out = []
    for ch in range(3):
        hs = []
        for j in range(-2, 3):
            r = min(max(cy + j, 0), Hh - 1)
            hs.append(five(wx, [float(Q[r][min(max(cx + i, 0), Wh - 1)][ch]) for i in range(-2, 3)]))
        out.append(decode(five(wy, hs), s))
    return out


def clamp_hist(H, mn, mx):
    if H > mn or math.isnan(H) or math.isnan(mn):
        return mx if H >= mx else H
    return mn


def quarter_pixel(PH1, x, y, s):
    """54b.5 for one PQ texel: three 10-bit codes."""
    h1, w1 = PH1.shape[:2]
    out = []
    for ch in range(3):
        acc, n = 0.0, 0
        for dx, dy in ((0, 0), (1, 0), (0, 1), (1, 1)):
            tx, ty = 2 * x + dx, 2 * y + dy
            if tx < w1 and ty < h1:
                c = mul(R32(Fr(int(PH1[ty, tx, ch]), 1023)), float(s.C_HDEC))
                t = min_nan(c, float(s.K10)) if (c > s.K5 or math.isnan(c)) else float(s.K5)
                acc = fma(sub(exp2(mul(mul(t, float(s.K8)), LOG2_10)), 1.0), float(s.K12), acc)
                n += 1
        mean = mul(rcp(float(n)), acc)
        o = mul(mul(mul(log2(pmax(fma(mean, float(s.K4), 1.0), float(s.K5))), LOG10_2), float(s.K7)), float(s.K11))
        out.append(unorm(o, 10))
    return out


def extended_pixel(x, y, ctx):
    """54b.3 / 54b.4 for one pixel. ctx: dict with sizes, symbols, surfaces and history arrays (see tests).
    Returns dict with 'col' (3 f32 before /E, after composite), 'T', 'Tp', 'ah', 'oa', 's'."""
    s = ctx["sym"]
    We, He, Wh, Hh, Wr, Hr = ctx["We"], ctx["He"], ctx["Wh"], ctx["Hh"], ctx["Wr"], ctx["Hr"]
    E, Ep, jx, jy = ctx["E"], ctx["Ep"], ctx["jx"], ctx["jy"]
    sxr, syr = div(float(Wr), float(We)), div(float(Hr), float(He))
    sxh, syh = div(float(We), float(Wh)), div(float(He), float(Hh))
    jxp, jyp = mul(div(float(We), float(Wr)), jx), mul(div(float(He), float(Hr)), jy)
    xf, yf = float(x), float(y)
    kx = int(rnd(add(fma(add(xf, 0.5), sxr, -0.5), jx)))
    ky = int(rnd(add(fma(add(yf, 0.5), syr, -0.5), jy)))
    cl = lambda n, m: min(max(n, 0), m - 1)  # noqa: E731
    res = {}
    if ctx.get("MV") is not None:
        mvs = ctx["MV"]
        vbx, vby = ctx["mv_origin"]
        X, Y = cl(kx, Wr) + vbx, cl(ky, Hr) + vby
        if 0 <= X < mvs.shape[1] and 0 <= Y < mvs.shape[0]:
            mv = (float(mvs[Y, X, 0]), float(mvs[Y, X, 1]))
        else:
            mv = (0.0, 0.0)
        m = (mul(mv[0], sxh), mul(mv[1], syh))
    else:
        m = (0.0, 0.0)
    hu = mul(add(add(xf, m[0]), 0.5), rcp(float(We)))
    hv = mul(add(add(yf, m[1]), 0.5), rcp(float(He)))
    col = list(ctx["col"][y][x])
    reset = ctx["reset"]
    if ctx.get("layer") is not None:
        L, (lbx, lby) = ctx["layer"], ctx["layer_origin"]
        O = ctx.get("O")
        a, b = add(xf, jxp), add(yf, jyp)
        sflag = abs(sub(rnd(a), a)) < 0.5 and abs(sub(rnd(b), b)) < 0.5
        w = float(s.TBW) if sflag else mul(float(s.TBW), float(s.TBW))
        w1 = sub(1.0, w)
        u = mul(add(add(xf, 0.5), jxp), rcp(float(We)))
        v = mul(add(add(yf, 0.5), jyp), rcp(float(He)))
        cur = bilinear(lambda i, j: texel(L, i, j), L.width, L.height, u, v)
        cur = [mul(E, cur[0]), mul(E, cur[1]), mul(E, cur[2]), cur[3]]
        if O is not None:
            curO = bilinear(lambda i, j: texel(O, i, j), O.width, O.height, u, v)[:3]
        if reset or ctx["created"]:
            hist = list(cur)
            if O is not None:
                histO = list(curO)
        else:
            mn, mx = [3.4028234663852886e38] * 4, [TINY] * 4
            for j in (-1, 0, 1):
                for i in (-1, 0, 1):
                    t = texel(L, cl(kx + i, Wr) + lbx, cl(ky + j, Hr) + lby)
                    t = [mul(E, t[0]), mul(E, t[1]), mul(E, t[2]), t[3]]
                    mn = [pmin(p, q) for p, q in zip(mn, t)]
                    mx = [pmax(p, q) for p, q in zip(mx, t)]
            TH = ctx["TH"]
            H = bilinear(lambda i, j: [float(c) for c in TH[j, i]], We, He, hu, hv)
            f = mul(E, rcp(pmax(Ep, float(s.T_EFL))))
            H = [mul(H[0], f), mul(H[1], f), mul(H[2], f), H[3]]
            hist = [clamp_hist(H[c], mn[c], mx[c]) for c in range(4)]
            if O is not None:
                ox, oy = O.origin
                mn, mx = [3.4028234663852886e38] * 3, [TINY] * 3
                for j in (-1, 0, 1):
                    for i in (-1, 0, 1):
                        t = texel(O, cl(kx + i, Wr) + ox, cl(ky + j, Hr) + oy)[:3]
                        mn = [pmin(p, q) for p, q in zip(mn, t)]
                        mx = [pmax(p, q) for p, q in zip(mx, t)]
                OH = ctx["OH"]
                HO = bilinear(lambda i, j: [float(c) for c in OH[j, i]], We, He, hu, hv)[:3]
                histO = [clamp_hist(HO[c], mn[c], mx[c]) for c in range(3)]
        T = [fma(w, cur[c], mul(w1, hist[c])) for c in range(4)]
        Ts = [R16(Fr(t)) if math.isfinite(t) else t for t in T]
        res["T"], res["s"] = T, sflag
        if O is not None:
            Tp = [fma(w, curO[c], mul(w1, histO[c])) for c in range(3)]
            Tps = [R16(Fr(t)) if math.isfinite(t) else t for t in Tp]
            col = [fma(col[c], sub(1.0, Tps[c]), Ts[c]) for c in range(3)]
            res["Tp"] = Tp
        else:
            col = [fma(col[c], sub(1.0, Ts[3]), Ts[c]) for c in range(3)]
    res["col"] = col
    if ctx.get("alpha"):
        A, (ax, ay) = ctx["A"], ctx["A_origin"]
        ach = ctx["A_channel"]

        def aval(i, j):
            t = texel(A, i + ax, j + ay)
            return t[ach] if ach < len(t) else 1.0

        P, G = ctx["P"], ctx["G"]
        px = fma(add(xf, 0.5), sxr, jx)
        py = fma(add(yf, 0.5), syr, jy)
        fx_, fy_ = math.floor(sub(px, 0.5)), math.floor(sub(py, 0.5))
        dx0, dy0 = sub(add(float(fx_), 0.5), px), sub(add(float(fy_), 0.5), py)
        W = Aw = 0.0
        Gs = [0.0, 0.0, 0.0]
        for i in (-1, 0, 1, 2):
            dx = add(dx0, float(i))
            for j in (-1, 0, 1, 2):
                dy = add(dy0, float(j))
                tx, ty = cl(fx_ + i, Wr), cl(fy_ + j, Hr)
                p = [float(t) for t in P[ty, tx, :3]]
                qx, qy, qxy = mul(p[0], float(s.K2n)), mul(p[1], float(s.K2n)), mul(p[2], float(s.K2x2n))
                e = fma(qxy, mul(dx, dy), fma(qx, mul(dx, dx), mul(qy, mul(dy, dy))))
                wt = exp2(e)
                W = add(W, wt)
                Aw = fma(wt, aval(tx, ty), Aw)
                for q in range(3):
                    Gs[q] = fma(wt, R16(Fr(int(G[ty, tx, q]), 255)), Gs[q])
        rW = rcp(pmax(W, float(s.K13)))
        n = [mul(g, rW) for g in Gs]
        g = sat(mul(n[0], mul(W, float(s.K1))))
        beta = fma(g, sub(1.0, n[1]), n[1])
        gamma = fma(g, sub(1.0, n[2]), n[2])
        a_up = mul(Aw, rW)
        if reset or ctx["ah_created"]:
            hc = 0.0
        else:
            AH = ctx["AH"]
            h = bilinear(lambda i, j: [float(AH[j, i, 0])], We, He, hu, hv)[0]
            mn, mx = 3.4028234663852886e38, -3.4028234663852886e38
            for j in (-1, 0, 1):
                for i in (-1, 0, 1):
                    t = aval(cl(kx + i, Wr), cl(ky + j, Hr))
                    mn, mx = pmin(mn, t), pmax(mx, t)
            hc = clamp_hist(h, mn, mx)
        ah = fma(a_up, beta, mul(sub(1.0, beta), hc))
        oa = fma(a_up, gamma, mul(sub(1.0, gamma), hc))
        res["ah"] = a_up if math.isnan(ah) else ah
        res["oa"] = a_up if math.isnan(oa) else oa
    return res
