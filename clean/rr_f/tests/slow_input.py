# SPDX-License-Identifier: MIT
"""Independent slow oracle of spec 52 (input stage): one token at a time, scalar binary32 arithmetic by exact
rational rounding (fractions.Fraction) and high-precision decimal elementary functions.

Written separately from reference/input_stage.py: it shares only the data containers (InputFrame, Texture,
FrameParams, InputSymbols) and the e4m3 decode table. Per-frame constants and camera maps are recomputed here."""
from decimal import Decimal, localcontext
from fractions import Fraction
import math

import numpy as np

from clean.rr_f.reference import exact
from clean.rr_f.reference.exact import F16, F32, round_frac

INF = math.inf
TINY = 2.0 ** -126


# ---------------------------------------------------------------------------------------------------------------
# scalar binary32 (round to nearest even, flush subnormal operands and results)

def ftz(x):
    x = float(x)
    if x != x or x in (INF, -INF):
        return x
    return math.copysign(0.0, x) if abs(x) < TINY else x


def r32(q):
    """Round an exact Fraction (or a Python float, taken exactly) to f32 and flush."""
    if isinstance(q, float):
        if q != q or q in (INF, -INF):
            return q
        q = Fraction(q)
    v = round_frac(q, F32)
    if q == 0:
        v = 0.0
    return ftz(v)


def _special(*xs):
    return any(x != x or x in (INF, -INF) for x in xs)


def add(a, b):
    a, b = ftz(a), ftz(b)
    if _special(a, b):
        return a + b
    s = Fraction(a) + Fraction(b)
    if s == 0:
        return -0.0 if (math.copysign(1, a) < 0 and math.copysign(1, b) < 0) else 0.0
    return r32(s)


def sub(a, b):
    return add(a, -ftz(b))


def mul(a, b):
    a, b = ftz(a), ftz(b)
    if _special(a, b):
        with np.errstate(all="ignore"):
            return float(np.float64(a) * np.float64(b))
    p = Fraction(a) * Fraction(b)
    if p == 0:
        return math.copysign(0.0, a) * math.copysign(1.0, b)
    v = r32(p)
    return math.copysign(0.0, a * b) if v == 0 else v


def fma(a, b, c):
    a, b, c = ftz(a), ftz(b), ftz(c)
    if _special(a, b, c):
        with np.errstate(all="ignore"):
            return float(np.float64(a) * np.float64(b) + np.float64(c))
    s = Fraction(a) * Fraction(b) + Fraction(c)
    if s == 0:
        pz = math.copysign(1.0, a) * math.copysign(1.0, b) < 0 and (a == 0 or b == 0)
        return -0.0 if (pz and math.copysign(1, c) < 0 and c == 0) else 0.0
    v = r32(s)
    return math.copysign(0.0, float(s)) if v == 0 else v


def div(a, b):
    a, b = ftz(a), ftz(b)
    if _special(a, b) or b == 0:
        with np.errstate(all="ignore"):
            return float(np.float64(a) / np.float64(b))
    q = Fraction(a) / Fraction(b)
    if q == 0:
        return math.copysign(0.0, a) * math.copysign(1.0, b)
    v = r32(q)
    return math.copysign(0.0, float(q)) if v == 0 else v


def rcp(x):
    return div(1.0, x)


def fsqrt(x):
    x = ftz(x)
    if x != x or x < 0:
        return math.nan
    if x == 0 or x == INF:
        return x
    q = Fraction(x)
    n, d = q.numerator, q.denominator
    rn, rd = math.isqrt(n), math.isqrt(d)
    if rn * rn == n and rd * rd == d:
        return r32(Fraction(rn, rd))
    with localcontext() as ctx:
        ctx.prec = 80
        v = Fraction(Decimal(n).sqrt() / Decimal(d).sqrt())
    return r32(v)


def flog2(x):
    x = ftz(x)
    if x != x or x < 0:
        return math.nan
    if x == 0:
        return -INF
    if x == INF:
        return INF
    m, e = math.frexp(x)
    if m == 0.5:
        return float(e - 1)
    with localcontext() as ctx:
        ctx.prec = 100
        v = Fraction(Decimal(x).ln() / Decimal(2).ln())
    return r32(v)


def fexp2(t):
    t = ftz(t)
    return ftz(exact.exp2_correct_f32(t))


def fmin(a, b):
    a, b = ftz(a), ftz(b)
    if a != a:
        return b
    if b != b:
        return a
    return a if a < b else b if b < a else a


def fmax(a, b):
    a, b = ftz(a), ftz(b)
    if a != a:
        return b
    if b != b:
        return a
    return a if a > b else b if b > a else a


def nan0(v):
    return 0.0 if v != v else v


def f32c(x):
    """Exact constant to f32 (decimal / rational input)."""
    return r32(Fraction(x))


with localcontext() as _c:
    _c.prec = 60
    LOG10_2 = r32(Fraction(Decimal(2).log10()))
    LOG2_10 = r32(Fraction(Decimal(10).ln() / Decimal(2).ln()))
    LN_2 = r32(Fraction(Decimal(2).ln()))
F16MAX = 65504.0


def r16(x):
    """f32 value -> binary16 value (RNE, subnormals kept, overflow to inf), as a float."""
    x = float(x)
    if x != x or x in (INF, -INF):
        return x
    v = round_frac(Fraction(x), F16)
    if v == 0:
        return math.copysign(0.0, x)
    return v


# ---------------------------------------------------------------------------------------------------------------
# per-frame constants (52.1.1, 52.8), independent

def q32(a, b):
    return r32(Fraction(a, b))


def slow_constants(p):
    wr, hr = p.render_size
    wo, ho = p.output_size
    wh, hh = 3 * (wo + wo % 2) // 2, 3 * (ho + ho % 2) // 2
    c = dict(Wr=wr, Hr=hr, Wt=-(-wr // 32) * 32, Ht=-(-hr // 32) * 32, Wh=wh, Hh=hh)
    c["sx"], c["sy"] = int(p.SH[p.par][0][0]), int(p.SH[p.par][0][1])
    c["fc"], c["reset"], c["E"], c["Ep"], c["X3"] = p.fc, p.reset, float(p.E), float(p.Ep), float(p.X3)
    c["jx"], c["jy"] = float(p.jitter[0]), float(p.jitter[1])
    c["pjx"], c["pjy"] = float(p.prev_jitter[0]), float(p.prev_jitter[1])
    ms = (mul(float(p.mv_scale_raw[0]), float(wr)), mul(float(p.mv_scale_raw[1]), float(hr)))
    mo = []
    for a in range(2):
        off = 0.0
        if p.mv_jittered:
            off = add(off, r16(sub(float(p.jitter[a]), float(p.prev_jitter[a]))))
        mo.append(r16(div(off, ms[a])))
    c["ms"], c["mo"] = ms, mo
    c["rWhWr"], c["rHhHr"], c["rWrWh"], c["rHrHh"] = q32(wh, wr), q32(hh, hr), q32(wr, wh), q32(hr, hh)
    c["msh"] = (div(ms[0], c["rWrWh"]), div(ms[1], c["rHrHh"]))
    c["invWr"], c["invHr"], c["invWh"], c["invHh"] = q32(1, wr), q32(1, hr), q32(1, wh), q32(1, hh)
    c["hcs"] = (q32(1, 2 * -(-wh // 2)), q32(1, 2 * -(-hh // 2)))
    c["hoff"] = (mul(c["pjx"], c["invWr"]), mul(c["pjy"], c["invHr"]))
    c["rev"] = bool(p.hw_depth and p.depth_inverted)
    c["zfar"] = 0.0 if p.depth_inverted else 1.0
    c["hw_depth"], c["nrm_dec"], c["rgh_packed"] = p.hw_depth, p.nrm_dec, p.rgh_packed
    c["V2C"] = None if p.V2C is None else [[float(v) for v in row] for row in p.V2C]
    c["maps"] = None
    if p.V2C is not None and p.W2V is not None:
        cur = (p.V2C, p.W2V)
        prev = cur if (p.reset or p.V2Cp is None) else (p.V2Cp, p.W2Vp)
        c["maps"] = slow_camera(cur[0], cur[1], prev[0], prev[1], wr, hr)
    c["sfv"] = p.specular_from_v
    return c


def mat(A):
    return [[float(np.float32(v)) for v in row] for row in A]


def s_matmul(A, B):
    return [[add(add(add(mul(A[i][0], B[0][j]), mul(A[i][1], B[1][j])), mul(A[i][2], B[2][j])),
                 mul(A[i][3], B[3][j])) for j in range(4)] for i in range(4)]


def s_inv(A):
    cof = [[0.0] * 4 for _ in range(4)]
    for i in range(4):
        rs = [r for r in range(4) if r != i]
        for j in range(4):
            cs = [q for q in range(4) if q != j]
            a, b, cc = ([A[r][q] for q in cs] for r in rs)
            def E(p_, q_):
                return sub(mul(b[p_], cc[q_]), mul(b[q_], cc[p_]))
            d = add(sub(mul(a[0], E(1, 2)), mul(a[1], E(0, 2))), mul(a[2], E(0, 1)))
            cof[i][j] = -d if (i + j) & 1 else d
    dets = []
    for i in range(4):
        s = 0.0
        for j in range(4):
            s = add(s, mul(A[i][j], cof[i][j]))
        dets.append(s)
    if any(d == 0 for d in dets):
        return None
    r = rcp(dets[3])
    return [[mul(cof[i][j], r) for i in range(4)] for j in range(4)]


def slow_camera(V2C, W2V, V2Cp, W2Vp, wr, hr):
    V2C, W2V, V2Cp, W2Vp = mat(V2C), mat(W2V), mat(V2Cp), mat(W2Vp)
    Pc, Pp = s_inv(W2V), s_inv(W2Vp)
    if Pc is None or Pp is None:
        return None
    Rc = [row[:] for row in Pc]
    Rc[3][0] = Rc[3][1] = Rc[3][2] = 0.0
    Rp = [row[:] for row in Pp]
    for q in range(3):
        Rp[3][q] = sub(Pp[3][q], Pc[3][q])
    a, b = s_inv(Rc), s_inv(Rp)
    if a is None or b is None:
        return None
    Am, Bm = s_matmul(a, V2C), s_matmul(b, V2Cp)
    iA = s_inv(Am)
    if iA is None:
        return None
    Tcc = s_matmul(iA, Bm)
    K = [[1.0 if i == j else 0.0 for j in range(4)] for i in range(4)]
    K[0][0] = K[3][0] = float(wr) / 2
    K[1][1] = -float(hr) / 2
    K[3][1] = float(hr) / 2
    Mpos = s_inv(s_matmul(Am, K))
    if Mpos is None:
        return None
    return Mpos, s_matmul(s_matmul(Am, Tcc), K)


# ---------------------------------------------------------------------------------------------------------------
# texel reads

def texel(frame, role, x, y, oob_zero=False):
    src = {"SV": "V", "B": "C", "G1": "C", "G2": "C"}.get(role, role)
    ox, oy = frame.textures[src].origin
    d = frame.textures[role].data
    X, Y = x + ox, y + oy
    if not (0 <= X < d.shape[1] and 0 <= Y < d.shape[0]):
        if oob_zero:
            return [0.0] * d.shape[2]
        raise IndexError(role)
    return [ftz(float(v)) for v in d[Y, X]]


def bil(get, Wi, Hi, u, v):
    """52.1.4 reference sampler, scalar; get(i, j) -> list of f32 channel values."""
    u = fmin(fmax(nan0(u), 0.0), 1.0)
    v = fmin(fmax(nan0(v), 0.0), 1.0)
    s = sub(mul(u, float(Wi)), 0.5)
    t = sub(mul(v, float(Hi)), 0.5)
    i0, j0 = math.floor(s), math.floor(t)
    fx, fy = sub(s, float(i0)), sub(t, float(j0))
    cl = lambda a, n: min(max(a, 0), n - 1)
    T00, T10 = get(cl(i0, Wi), cl(j0, Hi)), get(cl(i0 + 1, Wi), cl(j0, Hi))
    T01, T11 = get(cl(i0, Wi), cl(j0 + 1, Hi)), get(cl(i0 + 1, Wi), cl(j0 + 1, Hi))
    out = []
    for c in range(len(T00)):
        r0 = add(mul(sub(1.0, fx), T00[c]), mul(fx, T10[c]))
        r1 = add(mul(sub(1.0, fx), T01[c]), mul(fx, T11[c]))
        out.append(add(mul(sub(1.0, fy), r0), mul(fy, r1)))
    return out


def unorm10(code):
    return r32(Fraction(int(code), 1023))


def snorm8(code):
    return max(r32(Fraction(int(code), 127)), -1.0)


# ---------------------------------------------------------------------------------------------------------------
# features of one token

def s_cc(v, E, S, clamp=True):
    u = mul(E, v)
    if clamp:
        u = fmin(nan0(u), F16MAX)
    y = fmax(fma(u, S["K4"], 1.0), S["K5"])
    return mul(mul(flog2(y), LOG10_2), S["K7"])


def s_slc(v, S):
    t = mul(flog2(fma(abs(v), S["K4"], 1.0)), LOG10_2)
    r = mul(S["K7"] if v >= 0 else S["K7n"], t)
    r = nan0(r)
    return fmin(fmax(r, S["C_GCLn"]), S["C_GCL"])


def s_clip(v, S):
    return fmin(nan0(v), S["C_FCLIP"])


def s_pos(X, Y, mh, c):
    return (fma(sub(add(float(X), 0.5), c["jx"]), c["rWhWr"], mh[0]),
            fma(sub(add(float(Y), 0.5), c["jy"]), c["rHhHr"], mh[1]))


def s_out(pu, pv, c):
    return pu < 0 or pu > c["Wh"] or pv < 0 or pv > c["Hh"]


def s_hist(frame, X, Y, mh, c, S, Cown, both):
    pu, pv = s_pos(X, Y, mh, c)
    invalid = c["reset"] or s_out(pu, pv, c)
    res = []
    for img in ([frame.PH1, frame.PH2] if both else [frame.PH1]):
        if invalid:
            hf = [mul(S["Gh"], s_cc(Cown[ch], c["E"], S, clamp=False)) for ch in range(3)]
        else:
            u, v = mul(pu, c["hcs"][0]), mul(pv, c["hcs"][1])
            Hi, Wi = img.shape[:2]
            h = bil(lambda i, j: [unorm10(img[j, i, ch]) for ch in range(3)], Wi, Hi, u, v)
            hf = []
            for ch in range(3):
                h1 = mul(h[ch], S["C_HDEC"])
                hp = fmin(h1, S["K10"]) if h1 > S["K5"] else S["K5"]
                e = fexp2(mul(mul(hp, S["K8"]), LOG2_10))
                lin = mul(mul(sub(e, 1.0), S["K12"]), rcp(c["Ep"]))
                hf.append(mul(s_cc(lin, c["E"], S, clamp=False), S["Gh"]))
        res.append([s_clip(v, S) for v in hf])
    return res, (pu, pv)


def s_hid(frame, pu, pv, c, S):
    if c["reset"]:
        return [0.0] * 4
    if s_out(pu, pv, c):
        samp = [0.0] * 4
    else:
        uh = fma(pu, c["invWh"], c["hoff"][0])
        vh = fma(pv, c["invHh"], c["hoff"][1])
        H = frame.HID
        samp = bil(lambda i, j: [snorm8(H[j, i, ch]) for ch in range(4)], c["Wr"], c["Hr"], uh, vh)
    return [mul(S["Gs"], s) for s in samp]


def s_md(frame, X, Y, mh, chans, dq, c, S):
    if c["reset"]:
        return 0.0
    mr = (mul(mh[0], c["rWrWh"]), mul(mh[1], c["rHrHh"]))
    cx, cy = sub(add(float(X), 0.5), c["jx"]), sub(add(float(Y), 0.5), c["jy"])
    u = mul(add(add(cx, c["pjx"]), mr[0]), c["invWr"])
    v = mul(add(add(cy, c["pjy"]), mr[1]), c["invHr"])
    P = frame.PM_read
    p = bil(lambda i, j: [float(P[j, i, ch]) for ch in range(4)], c["Wr"], c["Hr"], u, v)
    dx = fma(-p[chans[0]], c["rWrWh"], mr[0])
    dy = fma(-p[chans[1]], c["rHrHh"], mr[1])
    cl = lambda t: fmin(t, S["C_MDL"]) if t > S["C_MDLn"] else S["C_MDLn"]
    dxs, dys = mul(cl(dx), S["Gm"]), mul(cl(dy), S["Gm"])
    n2 = fma(dxs, dxs, mul(dys, dys))
    L = fsqrt(n2) if n2 > 0 else n2
    return fmin(mul(mul(flog2(add(add(1.0, L), dq)), LN_2), S["Gd"]), S["C_MDL"])


def s_mv(frame, role, x, y, c):
    t = texel(frame, role, x, y)
    return (mul(add(t[0], c["mo"][0]), c["msh"][0]), mul(add(t[1], c["mo"][1]), c["msh"][1]))


# 52.7.2 scalar
def d3(a, b):
    return fma(a[2], b[2], fma(a[0], b[0], mul(a[1], b[1])))


def l3(a):
    return fsqrt(d3(a, a))


def un(a, S):
    r = rcp(add(l3(a), S["C_SV3"]))
    return [mul(x, r) for x in a]


def hm(a, M):
    return [add(d3(a, [M[0][q], M[1][q], M[2][q]]), M[3][q]) for q in range(4)]


def s_recon(Z, nv, Hd, px, py, c, S, b2):
    """Returns (m2x, m2y) or None for a B1 exit."""
    V = c["V2C"]
    Mpos, Mrep = c["maps"]
    fin = lambda x: x == x and x not in (INF, -INF)
    if c["hw_depth"]:
        Dz = Z
    else:
        if not b2 and not fin(Z):
            return None
        num = fma(Z, V[2][2], V[3][2])
        Dz = mul(num, rcp(mul(Z, V[2][3]))) if V[3][3] < 1 else num
    if not b2 and Dz == c["zfar"]:
        return None
    h = hm([px, py, Dz], Mpos)
    if not b2 and not (h[3] > 0 and fin(h[3])):
        return None
    rw = rcp(h[3])
    p = [mul(h[q], rw) for q in range(3)]
    sl = fmin(Hd, S["C_SV2"])
    if not sl > 0:
        sl = S["C_SV2"]
    n = un(nv, S)
    v = un(p, S)
    t5 = add(d3(n, v), d3(n, v))
    s = [mul(fma(-n[q], t5, v[q]), sl) for q in range(3)]
    cc = d3(n, s)
    if cc < 0:
        n = [-x for x in n]
        cc = -cc
    mc = fmax(cc, S["C_SV1"])
    J = [n[2], -n[0], n[1]]
    dnj = d3(n, J)
    t = un([fma(-n[q], dnj, J[q]) for q in range(3)], S)
    b = [fma(n[1], t[2], -mul(n[2], t[1])), fma(n[2], t[0], -mul(n[0], t[2])), fma(n[0], t[1], -mul(n[1], t[0]))]
    gm = rcp(sub(0.0, rcp(mc)))
    km = mul(rcp(mc), gm)
    Lv = l3([mul(d3(s, t), km), mul(d3(s, b), km), gm])
    q = [fma(v[i], Lv, p[i]) for i in range(3)]
    r = hm(q, Mrep)
    if not b2:
        if not (r[3] > 0 and fin(r[3])):
            return None
        ox = mul(fma(r[0], rcp(r[3]), -px), c["rWhWr"])
        if not fin(ox):
            return None
        oy = mul(fma(r[1], rcp(r[3]), -py), c["rHhHr"])
        if not fin(oy):
            return None
        return ox, oy
    if abs(r[3]) > S["C_SV1"]:
        u = (mul(r[0], rcp(r[3])), mul(r[1], rcp(r[3])))
    else:
        u = (0.0, 0.0)
    e = (sub(u[0], px), sub(u[1], py))
    def cl2(a):
        if a != a:
            return a
        return fmin(a, S["C_SV4"]) if a > S["C_SV4n"] else S["C_SV4n"]
    return mul(cl2(e[0]), c["rWhWr"]), mul(cl2(e[1]), c["rHhHr"])


# ---------------------------------------------------------------------------------------------------------------
# quantisation (52.4), by exact values

E4M3_VALUES = sorted({float(v) for v in [(m / 8) * 2.0 ** -6 for m in range(8)] +
                      [(1 + m / 8) * 2.0 ** (e - 7) for e in range(1, 16) for m in range(8) if not (e == 15 and m == 7)]})


def e4m3_code(mag, neg):
    idx = E4M3_VALUES.index(mag)
    return (0x80 if neg else 0) | idx


def s_quant(v, d):
    x = r16(v)
    neg = math.copysign(1.0, x) < 0
    if x != x:
        return 0x7F | (0x80 if neg else 0)
    if x in (INF, -INF):
        return e4m3_code(448.0, neg)
    ax = Fraction(abs(x))
    if ax >= Fraction(1, 2 ** 14):
        e = exact.floor_log2(ax)
        fb = Fraction(2) ** e
        fa = fb * (1 + Fraction(d, 1024))
    else:
        fb = Fraction(0)
        fa = Fraction(d, 2 ** 24)
    sg = -1 if neg else 1
    s1 = round_frac(sg * ax + sg * fa, F16)
    s2 = round_frac(Fraction(s1) - sg * fb, F16) if s1 not in (INF, -INF) else s1
    if s2 in (INF, -INF):
        return e4m3_code(448.0, neg)
    a2 = abs(Fraction(s2))
    if a2 == 0:
        return e4m3_code(0.0, neg)
    e2 = exact.floor_log2(a2)
    q = Fraction(2) ** (e2 - 3)
    tv = (a2 // q) * q                              # truncate to 3 mantissa bits at s2's exponent
    if tv > 448:
        return e4m3_code(448.0, neg)
    if tv < Fraction(1, 64):
        n = tv / Fraction(1, 512)
        fl = n.numerator // n.denominator
        rem = n - fl
        if rem > Fraction(1, 2) or (rem == Fraction(1, 2) and fl % 2 == 1):
            fl += 1
        tv = fl * Fraction(1, 512)
    return e4m3_code(float(tv), neg)


# ---------------------------------------------------------------------------------------------------------------
# one token

def token(frame, p, c, S, D, X, Y, b2=False):
    """Features (64 floats), codes, hold, mO, mN, mF, m2 for the tensor token (X, Y)."""
    wr, hr = c["Wr"], c["Hr"]
    xc, yc = min(X, wr - 1), min(Y, hr - 1)
    E = c["E"]
    f = [0.0] * 64
    C = texel(frame, "C", xc, yc)[:3]
    f[0], f[1], f[4] = (s_cc(C[i], E, S) for i in range(3))
    Dd, Ds = texel(frame, "Dd", xc, yc), texel(frame, "Ds", xc, yc)
    A = S["C_AMAX"]
    f[2], f[3], f[29] = fmin(Dd[1], A), fmin(Dd[2], A), fmin(Dd[0], A)
    f[6], f[7], f[10] = fmin(Ds[0], A), fmin(Ds[1], A), fmin(Ds[2], A)
    N = texel(frame, "N", xc, yc)
    nrm = [mul(2.0, sub(N[i], 0.5)) if c["nrm_dec"] else N[i] for i in range(3)]
    f[11], f[14], f[15] = fmin(nrm[0], A), fmin(nrm[1], A), fmin(nrm[2], A)
    if c["rgh_packed"]:
        rough = N[3]
    else:
        pad = X >= wr or Y >= hr
        ly = (Y + c["sy"]) % 8
        if pad and ly < 4:
            rough = texel(frame, "R", X, yc, oob_zero=True)[0]
        else:
            rough = texel(frame, "R", xc, yc)[0]
    f[18] = fmin(rough, A)
    # depth neighbourhood
    taps = [(0, 0), (-1, -1), (0, -1), (1, -1), (-1, 0), (1, 0), (-1, 1), (0, 1), (1, 1)]
    pts = [(min(max(xc + dx, 0), wr - 1), min(max(yc + dy, 0), hr - 1)) for dx, dy in taps]
    zs = [texel(frame, "Z", *q)[0] for q in pts]
    ni = fi = 0
    for i in range(1, 9):
        if (zs[i] > zs[ni]) if c["rev"] else (zs[i] < zs[ni]):
            ni = i
        if (zs[i] < zs[fi]) if c["rev"] else (zs[i] > zs[fi]):
            fi = i
    mO = s_mv(frame, "V", xc, yc, c)
    mN = s_mv(frame, "V", *pts[ni], c)
    mF = s_mv(frame, "V", *pts[fi], c)
    m2 = mO
    if "SV" in frame.textures:
        m2 = s_mv(frame, "SV", xc, yc, c)
    elif "Hd" in frame.textures and not c["sfv"] and c["maps"] is not None:
        oz = frame.textures["Z"].origin
        px, py = add(float(xc + oz[0]), 0.5), add(float(yc + oz[1]), 0.5)
        r = s_recon(texel(frame, "Z", xc, yc)[0], nrm, texel(frame, "Hd", xc, yc)[0], px, py, c, S, b2)
        if r is not None:
            m2 = r
    (o1, o2), _ = s_hist(frame, X, Y, mO, c, S, C, True)
    (hn,), pn = s_hist(frame, X, Y, mN, c, S, C, False)
    (hf,), _ = s_hist(frame, X, Y, mF, c, S, C, False)
    (h2,), p2 = s_hist(frame, X, Y, m2, c, S, C, False)
    f[12], f[13], f[16] = o1
    f[17], f[20], f[21] = o2
    f[5], f[8], f[9] = hn
    f[24], f[25], f[28] = hf
    f[37], f[40], f[41] = h2
    a = s_hid(frame, pn[0], pn[1], c, S)
    f[22], f[23], f[26], f[27] = a
    b = s_hid(frame, p2[0], p2[1], c, S)
    f[45], f[48], f[49], f[52] = b
    dq = texel(frame, "Q", xc, yc)[0] if "Q" in frame.textures else 0.0
    f[19] = s_md(frame, X, Y, mN, (0, 1), dq, c, S)
    f[44] = s_md(frame, X, Y, m2, (2, 3), dq, c, S)
    f[30] = f[31] = f[32] = s_clip(0.0, S)
    if "B" in frame.textures:
        B = texel(frame, "B", xc, yc)
        d = {ch: fma(-E, B[ch], fmin(nan0(mul(E, C[ch])), F16MAX)) for ch in (2, 0, 1)}
        lum = fma(d[2], S["C_LWrb"], fma(d[0], S["C_LWrb"], mul(d[1], S["C_LWg"])))
        f[30] = s_clip(s_slc(lum, S), S)
    if "G1" in frame.textures:
        f[31] = s_clip(s_slc(mul(E, texel(frame, "G1", xc, yc)[0]), S), S)
    if "G2" in frame.textures:
        f[32] = s_clip(s_slc(mul(E, texel(frame, "G2", xc, yc)[0]), S), S)
    f[33] = s_clip(0.0, S)
    f[36] = c["X3"]
    i = (c["fc"] * 16 + Y % 16) * 16 + X % 16
    dd = int(D[i]) >> 9
    codes = [s_quant(v, dd) for v in f]
    hold = [r16(fmin(nan0(mul(E, C[ch])), F16MAX)) for ch in range(3)] + [0.0]
    return dict(f=f, codes=codes, hold=hold, mO=mO, mN=mN, mF=mF, m2=m2)
