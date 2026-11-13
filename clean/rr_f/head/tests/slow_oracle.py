# SPDX-License-Identifier: MIT
"""Slow element-by-element oracle of D0-R (pages 51.3, 51.4, 51.6, 55), written separately from the vectorised
reference: Python integers for the matrix steps, the integer rounding of exactfp for every f16 / f32 rounding, the
decimal-module elementary functions, explicit loops for windows, borders and the bilinear read. Only NumPy's integer
matmul is used (for exact integer dot products)."""
import math

import numpy as np

from .. import exactfp as X

NAN = float("nan")
INF = float("inf")


# ----------------------------------------------------------------------------------------- scalar helpers
def q8_fast(h):
    """Q8 by direct grid rounding (a third, independent form; checked against exactfp.q8 on all f16)."""
    sign = 0x80 if math.copysign(1.0, h) < 0 else 0
    if math.isnan(h):
        return sign | 0x7F
    a = abs(h)
    if a >= 448:
        return sign | 0x7E
    if a < 2.0 ** -6:
        n = _rne(a * 512)
        return sign | n
    e = math.frexp(a)[1] - 1
    n = _rne(math.ldexp(a, 3 - e))          # 8 <= n <= 16
    if n == 16:
        e, n = e + 1, 8
    return sign | (8 * (e + 7) + n - 8)


def _rne(x):
    f = math.floor(x)
    r = x - f
    if r > 0.5 or (r == 0.5 and f % 2 == 1):
        f += 1
    return int(f)


def reflect_slow(v, n):
    if n == 1:
        return 0
    while v < 0 or v >= n:
        v = -v if v < 0 else 2 * (n - 1) - v
    return v


def mstep_int(ints_a, ints_b, start):
    """One or more matrix steps: ints_a [K], ints_b [K] are e4m3 values scaled by 2^9 (exact integers) or None for
    NaN; start an f16 float. Chain of 32-input steps, one f16 rounding each."""
    acc = start
    K = len(ints_a)
    for k0 in range(0, K, 32):
        a = ints_a[k0:k0 + 32]
        b = ints_b[k0:k0 + 32]
        if any(t is None for t in a) or any(t is None for t in b) or math.isnan(acc):
            acc = NAN
            continue
        S = sum(int(p) * int(q) for p, q in zip(a, b))        # value S / 2^18
        if math.isinf(acc):
            continue                                          # inf + finite
        n = (S << 6) + int(acc * (1 << 24))                   # f16 start value * 2^24 is an integer
        acc = 0.0 if n == 0 else X.rnd16(n, 1 << 24)
    return acc


def e4_int(code):
    v = X.e4m3_value(code)
    return None if math.isnan(v) else int(v * 512)


# ----------------------------------------------------------------------------------------- block
class Block:
    def __init__(self, W, k, ra, m):
        self.W, self.k, self.ra, self.m = W, k, float(ra), float(m)
        self.H = W["Wq"].shape[0]
        self.C = W["Wq"].shape[1]
        cols = lambda Wm: [[e4_int(int(c)) for c in col] for col in np.asarray(Wm).T]
        self.Wq = [cols(W["Wq"][h]) for h in range(self.H)]
        self.Wv = [cols(W["Wv"][h]) for h in range(self.H)]
        self.Wo = cols(np.asarray(W["Wo"]).reshape(self.H * 32, self.C))
        self.W1 = cols(W["W1"])
        self.W2 = cols(W["W2"])
        self.pairs = {p: i for i, p in enumerate([(0, 0), (1, 0), (0, 1), (1, 1), (0, 2), (1, 2), (0, 3), (1, 3),
                                                  (2, 2), (3, 2), (2, 3), (3, 3)])}

    def bias(self, h, a, b):
        I, J = a // 16, b // 16
        tb = self.W["Tb"][h]
        if (I, J) in self.pairs:
            return float(tb[self.pairs[(I, J)]][a % 16][b % 16])
        return float(tb[self.pairs[(J, I)]][b % 16][a % 16])

    def attn(self, s):
        k = self.k
        u = X.mul(s, float(k["A_SCALE"]))
        c = X.fmin(X.fmax(u, float(k["A_CLAMPn"])), float(k["A_CLAMP"]))
        inner = X.fma(c, -c, float(k["A_P1"]))
        p = X.fma(c, inner, float(k["A_P0"]))
        y = 32 * (p - 1)
        n = math.floor(y)
        assert 1 <= n <= 30
        return math.ldexp(1 + y - n, n - 15)

    def act(self, x):
        k = self.k
        kk = X.fmin(X.fmax(x, float(k["ACT_CLn"])), float(k["ACT_CL"]))
        u = X.fma(-abs(kk), float(k["ACT_C1"]), float(k["ACT_C2"]))
        v = X.fma(kk, u, float(k["ACT_C3"]))
        return X.mul(x, v)

    def run(self, xcodes):
        """xcodes: 64 lists of C e4m3 codes. Returns (y2h [64][C] floats, Z [H][64])."""
        C, H = self.C, self.H
        xi = [[e4_int(int(c)) for c in row] for row in xcodes]
        xr = [[X.e4m3_value(int(c)) for c in row] for row in xcodes]
        O = [[] for _ in range(64)]
        Zall = []
        for h in range(H):
            q = [[q8_fast(mstep_int(xi[t], self.Wq[h][i], 0.0)) for i in range(32)] for t in range(64)]
            v = [[q8_fast(mstep_int(xi[t], self.Wv[h][i], 0.0)) for i in range(32)] for t in range(64)]
            qi = [[e4_int(c) for c in row] for row in q]
            vT = [[e4_int(v[t][i]) for t in range(64)] for i in range(32)]
            Zh = []
            for a in range(64):
                w = []
                for b in range(64):
                    s = mstep_int(qi[a], qi[b], self.bias(h, a, b))
                    w.append(self.attn(s))
                Z = X.exact_sum_round(w, X.rnd16)
                r = X.div(1.0, Z, X.rnd16)
                P = [e4_int(q8_fast(X.mul(wb, r))) for wb in w]
                Zh.append(Z)
                O[a].extend(e4_int(q8_fast(mstep_int(P, vT[i], 0.0))) for i in range(32))
            Zall.append(Zh)
        y2h = []
        for t in range(64):
            Dp = [mstep_int(O[t], self.Wo[c], float(self.W["bo"][c])) for c in range(C)]
            y1h = [X.fma(Dp[c], self.ra, xr[t][c]) for c in range(C)]
            y1q = [e4_int(q8_fast(v_)) for v_ in y1h]
            Au = []
            for j in range(4 * C):
                Du = mstep_int(y1q, self.W1[j], float(self.W["b1"][j]))
                Au.append(e4_int(q8_fast(self.act(Du))))
            row = []
            for c in range(C):
                Dd = mstep_int(Au, self.W2[c], float(self.W["b2"][c]))
                row.append(X.fma(Dd, self.m, y1h[c]))
            y2h.append(row)
        return y2h, Zall


def stage_y2h(x, W, k, ra, m, shift, windows=None):
    """y2h for the tokens of the selected windows (indices in row-major window order; None = all).
    Returns dict (X, Y) -> list of C floats, and dict window -> Z."""
    ny, nx, C = x.shape
    sx, sy = shift
    nwx = -(-(nx + sx) // 8)
    nwy = -(-(ny + sy) // 8)
    blk = Block(W, k, ra, m)
    out, zs = {}, {}
    for j in range(nwy):
        for i in range(nwx):
            widx = j * nwx + i
            if windows is not None and widx not in windows:
                continue
            coords, codes = [], []
            for ly in range(8):
                for lx in range(8):
                    Xa, Ya = 8 * i - sx + lx, 8 * j - sy + ly
                    coords.append((Xa, Ya))
                    codes.append(x[reflect_slow(Ya, ny), reflect_slow(Xa, nx)])
            y2h, Z = blk.run(codes)
            zs[widx] = Z
            for t, (Xa, Ya) in enumerate(coords):
                if 0 <= Xa < nx and 0 <= Ya < ny:
                    out[(Xa, Ya)] = y2h[t]
    return out, zs


# ----------------------------------------------------------------------------------------- head
_SIG = {}


def sig(v):
    key = X.f16_bits(v)
    if key not in _SIG:
        _SIG[key] = X.sig_spec(v)
    return _SIG[key]


# Block-output channels read by the head, written out here independently of head.py: the page 55 erratum's natural
# numbers (default) and, for comparison only, the numbers as printed on page 55.
CH_ERRATUM = dict(gate=(0, 1, 2, 3), hid=(14, 15, 16, 17), gates=(4, 5, 6), prec=(7, 8, 9), candw=(10, 11, 12, 13))
CH_ERRATUM1 = dict(gate=(0, 1, 2, 3), hid=(14, 16), gates=(4, 5, 6), prec=(7, 8, 9), candw=(10, 12))   # before 55-ERRATUM-2
CH_PAGE = dict(gate=(0, 1, 4, 5), hid=(28, 2), gates=(8, 9, 12), prec=(13, 16, 17), candw=(20, 24))


def head_token(y, hold, hc, g, Xa, Ya, s1, s2, ch=CH_ERRATUM):
    """Cnet and the image values of one token: y = 32 channel floats, hold = 4 floats, hc = coarse tensor
    (indexable [Y][X][j]). Returns dict."""
    g = float(g)
    my, mx = len(hc), len(hc[0])
    sg = [sig(y[c]) for c in ch["gate"]]
    z = sg[3]
    omz = X.add(1.0, -z)
    xm = math.floor((Xa - 0.5) / 2)
    ym = math.floor((Ya - 0.5) / 2)
    fx = (Xa - 0.5) / 2 - xm          # weight of the floor + 1 neighbour (3/4 or 1/4)
    fy = (Ya - 0.5) / 2 - ym
    wxp, wxm, wyp, wym = fx, 1 - fx, fy, 1 - fy
    rd = lambda yy, xx, j: float(hc[reflect_slow(yy, my)][reflect_slow(xx, mx)][j])
    cn = []
    for j in range(3):
        A, Bv, Cc, Dc = rd(ym, xm + 1, j), rd(ym, xm, j), rd(ym + 1, xm, j), rd(ym + 1, xm + 1, j)
        t = X.mul(X.mul(A, wxp), wym)
        t = X.fma(X.mul(Bv, wxm), wym, t)
        t = X.fma(X.mul(Cc, wxm), wyp, t)
        u = X.fma(X.mul(Dc, wxp), wyp, t)
        gate = X.mul(sg[j], g)
        hh = X.mul(X.mul(u, gate), z)
        cn.append(X.fma(float(hold[j]), omz, hh))
    cn.append(X.mul(float(hold[3]), omz))
    th = lambda v: X.tanh_cr(v, X.F16)
    if len(ch["hid"]) == 4:      # 55-ERRATUM-2
        hid = [th(y[c]) for c in ch["hid"]]
    else:
        ha, hb = ch["hid"]
        hid = [th(y[ha]), th(y[ha]), th(y[hb]), th(y[hb])]
    gates = [sig(y[c]) for c in ch["gates"]] + [0.0]
    s1f, s2f = float(s1), float(s2)
    cx, cy, cr = ch["prec"]
    Px = X.mul(sig(y[cx]), s1f, X.rnd32)
    Py = X.mul(sig(y[cy]), s1f, X.rnd32)
    rho = X.tanh_cr(X.mul(y[cr], s2f, X.rnd32), X.F32)
    prec = [X.r16(X.mul(Px, Px, X.rnd32)), X.r16(X.mul(Py, Py, X.rnd32)),
            X.r16(X.mul(X.mul(Px, Py, X.rnd32), rho, X.rnd32)), 0.0]
    if len(ch["candw"]) == 4:    # 55-ERRATUM-2
        candw_ = candw4([y[c] for c in ch["candw"]])
    else:
        w, w2 = candw(y[ch["candw"][0]], y[ch["candw"][1]])
        candw_ = [w, w, w2, w2]
    return {"Cnet": cn, "HID": hid, "HID_code": [X.snorm8_code(v) for v in hid],
            "GATES": gates, "GATES_code": [X.unorm8_code(v) for v in gates], "PREC": prec,
            "CANDW": candw_, "CANDW_code": [X.unorm8_code(v) for v in candw_]}


def candw4(z):
    """55-ERRATUM-2, one token, scalar exact arithmetic: m = max (NaN ignored unless all NaN); d = R(z - m);
    t = R(d R(log2 e)); e = R(2^t); S = R(R(e0 + e2) + R(e1 + e3)); q = R(1/S); W = R(e q)."""
    z = [float(v) for v in z]
    fin = [v for v in z if not math.isnan(v)]
    m = max(fin) if fin else NAN
    l2e = X.r16(math.log2(math.e))
    e = []
    for v in z:
        d = NAN if (math.isnan(v) or math.isnan(m) or (math.isinf(v) and math.isinf(m) and v == m)) else X.r16(v - m)
        t = NAN if math.isnan(d) else (d * l2e if math.isinf(d) else X.r16(d * l2e))
        e.append(X.exp2_cr(t, X.F16))
    S = X.add(X.add(e[0], e[2]), X.add(e[1], e[3])) if not any(math.isnan(v) for v in e) else NAN
    if math.isnan(S):
        return [NAN] * 4
    n, d_ = S.as_integer_ratio()
    q = X.round_rational(d_, n, X.F16) if S != 0 else INF
    return [X.mul(v, q) for v in e]


def candw(y20, y24):
    """(w, w') of 55.3: d = f32(y20 - y24) (IEEE for non-finite logits), exact logistic halved, rounded once."""
    y20, y24 = float(y20), float(y24)
    if math.isnan(y20) or math.isnan(y24):
        d = NAN
    elif math.isinf(y20) or math.isinf(y24):
        d = y20 - y24
    else:
        d = X.add(y20, -y24, X.rnd32)
    return X.logistic_cr(d, X.F16, scale_half=True), X.logistic_cr(-d, X.F16, scale_half=True)
