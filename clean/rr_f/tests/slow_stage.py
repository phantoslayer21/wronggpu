# SPDX-License-Identifier: MIT
"""Slow per-token / per-window implementation of a whole stage call of spec 51 (51.4-51.6), built on the slow
window block of slow.py. Independent of the vectorised reference's geometry code."""
from fractions import Fraction
import math

from clean.rr_f.reference.scalar import sigmoid_eval
from clean.rr_f.reference.stages import stage_info, effective_shift
from clean.rr_f.tests import slow
from clean.rr_f.tests.slow import SV, fma, mm_step_chain, q8_value, r16, s_mul, s_round, s_add, s_negate


def d8_slow(code):
    code = int(code)
    s, e, m = code >> 7, (code >> 3) & 15, code & 7
    if e == 15 and m == 7:
        return math.nan
    v = Fraction(m, 8) * Fraction(1, 64) if e == 0 else (1 + Fraction(m, 8)) * Fraction(2) ** (e - 7)
    v = float(v)
    return -v if s else v


def _decode_w(W):
    """numpy logical weights -> nested lists of floats (codes decoded)."""
    out = {}
    for k, v in W.items():
        if v.dtype == "uint8":
            out[k] = _tolist_map(v, d8_slow)
        else:
            out[k] = _tolist_map(v, float)
    return out


def _tolist_map(a, f):
    if a.ndim == 1:
        return [f(x) for x in a]
    return [_tolist_map(x, f) for x in a]


def sig(x):
    return sigmoid_eval(x)[0]


def run_stage(name, inputs, W, consts, wc, shift_row):
    info = stage_info(name)
    C = info["C"]
    sx, sy = effective_shift(name, shift_row)
    Wd = _decode_w(W)
    kind = info["kind"]
    base = inputs["skip"] if kind == "fuse" else inputs["in"] if info["adapter"] else inputs["x"]
    Ny, Nx = base.shape[:2]
    nwx = (Nx + sx + 7) // 8
    nwy = (Ny + sy + 7) // 8
    y2 = {}
    merged = {}
    Zall = {}
    upcache = {}

    def up_token(Yc, Xc):
        if (Yc, Xc) not in upcache:
            row = [d8_slow(v) for v in inputs["cin"][Yc, Xc]]
            upcache[(Yc, Xc)] = [q8_value(mm_step_chain(row, [Wd["Wup"][k][o] for k in range(len(row))],
                                                        Wd["bup"][o])) for o in range(4 * C)]
        return upcache[(Yc, Xc)]

    for j in range(nwy):
        for i in range(nwx):
            xv, xr, coords = [], [], []
            for ly in range(8):
                for lx in range(8):
                    X = 8 * i - sx + lx
                    Y = 8 * j - sy + ly
                    coords.append((X, Y))
                    if kind == "fuse":
                        My, Mx = Ny // 2, Nx // 2
                        sk = inputs["skip"][slow.reflect(Y, Ny), slow.reflect(X, Nx)]
                        Xc, Yc = math.floor(X / 2), math.floor(Y / 2)
                        n = 2 * (Y - 2 * Yc) + (X - 2 * Xc)
                        up = up_token(slow.reflect(Yc, My), slow.reflect(Xc, Mx))
                        xf = [fma(SV.of(d8_slow(sk[c])), SV.of(consts["f"]), SV.of(up[n * C + c])).float()
                              for c in range(C)]
                        xv.append([q8_value(v) for v in xf])
                        xr.append(xf)
                    elif info["adapter"]:
                        row = [d8_slow(v) for v in inputs["in"][slow.reflect(Y, Ny), slow.reflect(X, Nx)]]
                        xa = [q8_value(mm_step_chain(row, [Wd["Wa"][k][o] for k in range(128)], Wd["ba"][o]))
                              for o in range(C)]
                        xv.append(xa)
                        xr.append(xa)
                    else:
                        row = [d8_slow(v) for v in inputs["x"][slow.reflect(Y, Ny), slow.reflect(X, Nx)]]
                        xv.append(row)
                        xr.append(row)
            out, Z = slow.window_block(xv, xr, Wd, consts["ra"], consts["m"], wc)
            Zall[(j, i)] = Z
            for t, (X, Y) in enumerate(coords):
                if 0 <= X < Nx and 0 <= Y < Ny:
                    assert (Y, X) not in y2
                    y2[(Y, X)] = out[t]
            if info["Cn"]:
                for ry in range(4):
                    for rx in range(4):
                        Xc = (8 * i - sx) // 2 + rx
                        Yc = (8 * j - sy) // 2 + ry
                        if 0 <= Xc < Nx // 2 and 0 <= Yc < Ny // 2:
                            z = []
                            for n in range(4):
                                dy, dx = n // 2, n % 2
                                t = 8 * (2 * ry + dy) + (2 * rx + dx)
                                z += out[t]
                            merged[(Yc, Xc)] = [q8_value(mm_step_chain(z, [Wd["Wm"][k][o] for k in range(4 * C)],
                                                                       Wd["bm"][o])) for o in range(info["Cn"])]
    res = {"y2": y2, "Z": Zall}
    if info["Cn"]:
        res["merged"] = merged
    if info["hidden"]:
        nproj = info["n_hidden_proj"]
        hnew = {}
        for (Y, X), row in y2.items():
            Dh = [mm_step_chain(row, [Wd["Wh"][k][o] for k in range(C)], Wd["bh"][o]) for o in range(nproj)]
            hold = [float(v) for v in inputs["hold"][Y, X]]
            g = SV.of(consts["g"])
            if info["hidden"] == "enc":
                h = []
                for jj in range(3):
                    G = s_round(s_mul(SV.of(sig(Dh[jj])), g))
                    h.append(s_round(s_mul(SV.of(hold[jj]), G)).float())
                h.append(0.0)
            else:
                z = SV.of(sig(Dh[3]))
                omz = s_round(s_add(SV.of(1.0), s_negate(z)))
                Hc = inputs["Hc"]
                My, Mx = Hc.shape[:2]
                fx = Fraction(2 * X - 1, 4)
                fy = Fraction(2 * Y - 1, 4)
                xm, ym = math.floor(fx), math.floor(fy)
                wxp = Fraction(3, 4) if X % 2 == 0 else Fraction(1, 4)
                wyp = Fraction(3, 4) if Y % 2 == 0 else Fraction(1, 4)
                wxm, wym = 1 - wxp, 1 - wyp
                h = []
                for jj in range(3):
                    def at(yy, xx):
                        return SV.of(float(Hc[slow.reflect(yy, My), slow.reflect(xx, Mx), jj]))
                    A, B = at(ym, xm + 1), at(ym, xm)
                    Cc, Dc = at(ym + 1, xm), at(ym + 1, xm + 1)
                    WXP, WXM, WYP, WYM = (SV.of(float(v)) for v in (wxp, wxm, wyp, wym))
                    t = s_round(s_mul(s_round(s_mul(A, WXP)), WYM))
                    t = fma(s_round(s_mul(B, WXM)), WYM, t)
                    t = fma(s_round(s_mul(Cc, WXM)), WYP, t)
                    u = fma(s_round(s_mul(Dc, WXP)), WYP, t)
                    gate = s_round(s_mul(SV.of(sig(Dh[jj])), g))
                    hh = s_round(s_mul(s_round(s_mul(u, gate)), z))
                    h.append(fma(SV.of(hold[jj]), omz, hh).float())
                h.append(s_round(s_mul(SV.of(hold[3]), omz)).float())
            hnew[(Y, X)] = h
        res["hnew"] = hnew
        if info["merged_hidden"]:
            hm = {}
            for Yc in range(Ny // 2):
                for Xc in range(Nx // 2):
                    vals = []
                    for jj in range(4):
                        q = [s_round(s_mul(SV.of(hnew[(2 * Yc + n // 2, 2 * Xc + n % 2)][jj]), SV.of(0.25)))
                             for n in range(4)]
                        vals.append(s_round(s_add(s_round(s_add(q[0], q[1])), s_round(s_add(q[2], q[3])))).float())
                    hm[(Yc, Xc)] = vals
            res["hmerged"] = hm
    return res
