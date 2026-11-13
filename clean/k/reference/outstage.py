# SPDX-License-Identifier: MIT
"""NumPy reference of the model K output stage (spec 43 draft 2) and its filter-shape table (spec 43 section 7).

Every step follows the spec's order and rounding: FP32 (or FP16) operations rounded to nearest even, fma with one
rounding (clean/k/reference/fp.py), R = round half away, rni = round to nearest even. Arrays are over the pixel grid
ceil16(W) x ceil16(H) ([y, x] indexing); the four sub-samples (j, i) of a pixel are a trailing axis of 4 in the order
(0, 0), (1, 0), (0, 1), (1, 1) = index j + 2 i.

Draft 2 fixes the operation order where draft 1 left it open: bilinear samples (spec 42 section 1), 4.9 (rho_e (l D)),
4.10 (fma(H00, H10, H01 H11)), 5.3 (tap luma of max(RGB, 0)), 6.1 (gamma00 + gamma01, gamma10 + gamma11), 6.5
(reciprocal q), 6.7 (kappa' (Y_o d), G with fma(Co, 0, Y)), 6.8 (fma sum of squares, one rounded 1/sqrt) and 7.8
(truncation, magnitude above 65504 -> infinity).
"""
import numpy as np

from clean.k.reference.fp import F16, F32, R, f32, fma16, fma32, fn32, h, rni, sat, trunc16

# ---- constants (spec 43 section 2) -----------------------------------------------------------------------------------
LAMBDA = F32(1.37)
A_MIN = F32(0.01)
THETA = F16(0.02)
MU = F16(0.03)
NU = F16(1 / 0.97)
QSEL = F16(8.0008)
EPS5 = np.uint32(0x3727C5AC).view(np.float32)  # 1e-5
C0P99 = np.uint32(0x3F7D70A4).view(np.float32)  # 0.99
DEC_T = np.uint32(0x3F7FBE88).view(np.float32)  # 0.999 (spec 42 decode branch)
DEC_C = np.uint32(0x3C4FAE29).view(np.float32)  # 0.012676
LOG2E = np.uint32(0x3FB8AA3B).view(np.float32)


def ceil16(n):
    return -(-int(n) // 16) * 16


# ---- section 7: filter-shape table -------------------------------------------------------------------------------------

def filter_level(rw, rh, W, H):
    """l = clamp(R(7 (sqrt(1 / ((1/2 (r_h RN32(1/H))) (1/2 (r_w RN32(1/W))))) - 1) / 3), 0, 7) in FP32 steps."""
    rh_ = F32(rh) * (F32(1) / F32(H))
    rw_ = F32(rw) * (F32(1) / F32(W))
    prod = (F32(0.5) * rh_) * (F32(0.5) * rw_)
    v = (F32(7) * (np.sqrt(F32(1) / prod) - F32(1))) / F32(3)
    return int(min(max(float(R(np.float32(v))), 0.0), 7.0))


def filter_table(level, theta=1):
    """The 2048 entries of set 2 level + (1 - theta) as float16 [2048, 8]: (A, Cxx, Bxy, shear, O (bit pattern 0 or 1),
    K, kappa6, 0). Entry 256 (F0 + 2 F1 + 4 F2) + 16 i + j."""
    idx = np.arange(2048)
    flags = idx // 256
    F0, F1, F2 = (flags & 1) == 1, (flags & 2) == 2, (flags & 4) == 4
    i = f32((idx // 16) % 16)
    j = f32(idx % 16)
    with np.errstate(all="ignore"):
        Q = F32(8) * (F32(1) + np.uint32(0x38D1B717).view(np.float32))

        def value(k):
            z = (k - F32(8)) / Q
            return z / (F32(1) - np.abs(z))
        a, b = value(i), value(j)
        phi = (F32(3) * F32(level)) / F32(8) + F32(1)
        narrow = ~F2 & F1
        hi = np.where(narrow, F32(0.5), F32(2.1)).astype(np.float32)
        lo = np.where(narrow, F32(0.1), F32(0.17)).astype(np.float32)
        s = np.abs(b) + np.abs(a)
        omega = np.where(s < F32(0.25), F32(1) - F32(4) * np.abs(b), np.abs(a) / s).astype(np.float32)
        sigma_b = np.where(b > 0, F32(1), F32(-1)).astype(np.float32)
        tau = (F32(np.sqrt(np.float32(0.5))) * sigma_b) * (F32(1) - omega)
        tau = np.where(a >= 0, tau, -tau).astype(np.float32)
        gamma = np.sqrt(F32(1) - tau * tau)
        r2 = a * a + b * b
        M = (hi - lo) * (r2 / (r2 + F32(1))) + lo
        lo = np.where(narrow & (M > F32(0.4)), F32(0.08), lo).astype(np.float32)
        if theta == 0:
            lo = np.where(lo > F32(0.2), lo, F32(0.2))  # max(0.2, lo) as "x if x > y else y": max(0.2, lo)
            lo = np.where(F32(0.2) > lo, F32(0.2), lo).astype(np.float32)
            M = np.where(lo > M, lo, M).astype(np.float32)
        lo_new = np.where(F32(0.5) > lo, F32(0.5), lo).astype(np.float32)  # max(0.5, lo)
        M = np.where(F2, np.where(lo_new > M, lo_new, M), M).astype(np.float32)
        lo = np.where(F2, lo_new, lo).astype(np.float32)
        pos = a >= 0
        P = np.where(pos, gamma / lo, gamma / M).astype(np.float32)
        Qp = np.where(pos, tau / lo, tau / M).astype(np.float32)
        Rr = np.where(pos, tau / M, tau / lo).astype(np.float32)
        Mp = np.where(pos, M, lo).astype(np.float32)
        Sp = (-gamma) / Mp
        kappa = -(F32(1) / F32(np.log(2.0)))
        Bxy = (((Sp * Rr) + (Qp * P)) * F32(2)) * kappa
        u = np.where(pos, gamma, -tau).astype(np.float32)
        v = (-np.where(pos, tau, gamma)).astype(np.float32)
        O = np.abs(u) > np.abs(v)
        shear = np.where(O, v / u, u / v).astype(np.float32)
        A = np.where(O, kappa * (Sp * Sp + Qp * Qp), kappa * (Rr * Rr + P * P)).astype(np.float32)
        Cxx = np.where(O, (Rr * Rr + P * P) * kappa, (Sp * Sp + Qp * Qp) * kappa).astype(np.float32)
        w = r2 / (r2 + F32(1))
        w = np.where(~F2 & F0, F32(0), w).astype(np.float32)
        k6 = w * phi
        K = (((phi - phi * phi) * w) + phi * phi) - k6
    out = np.zeros((2048, 8), np.float16)
    for c, val in enumerate((A, Cxx, Bxy, shear)):
        out[:, c] = trunc16(val)
    out[:, 4] = np.where(O, np.uint16(1), np.uint16(0)).view(np.float16)
    out[:, 5] = trunc16(K)
    out[:, 6] = trunc16(k6)
    return out


# ---- shared pieces of specs 42 / 43 -----------------------------------------------------------------------------------

def tap_index(i, d, base, n):
    """Spec 42 section 3: index = b + min(i + d, n - 1) if i + d > 0, else b."""
    t = i + d
    return np.where(t > 0, base + np.fmin(t, n - 1), base).astype(np.int64)


def motion_index_lo(i, d, base, n):
    """Spec 42 section 3, [MVlo] motion fetch: i' = b + i + d; index = min(i' + b, b + n - 1) if i' > 0, else b."""
    ip = base + i + d
    return np.where(ip > 0, np.fmin(ip + base, base + n - 1), base).astype(np.int64)


def decode(t):
    """Spec 42 HDR decode factor f (t f is the linear value): 1 / (1 - t) if t < 0.999, else
    (2^((t + 0.012676) 9 log2 e) - 1) / (9 t)."""
    t = f32(t)
    with np.errstate(all="ignore"):
        e = ((t + DEC_C) * F32(9)) * LOG2E
        big = (fn32(np.exp2, e) - F32(1)) / (F32(9) * t)
        return np.where(t < DEC_T, F32(1) / (F32(1) - t), big).astype(np.float32)


def encode(Y):
    """Spec 43 e(Y): 1 / (Y + 1) if Y < 1000, else (ln(fma(Y, 9, 1)) / 9 - 0.012676) / Y; stored = Y e(Y)."""
    Y = f32(Y)
    with np.errstate(all="ignore"):
        big = (fn32(np.log, fma32(Y, F32(9), F32(1))) / F32(9) - DEC_C) / Y
        return np.where(Y < F32(1000), F32(1) / (Y + F32(1)), big).astype(np.float32)


def nearest_depth_pixel(depth, ix, iy, zbase, rsize, inverted):
    """Spec 42 step 5.2 [MVlo]: (dx, dy) offset of the motion pixel from render pixel (ix, iy)."""
    def Z(dx, dy):
        return depth[tap_index(iy, dy, zbase[1], rsize[1]), tap_index(ix, dx, zbase[0], rsize[0])].astype(np.float32)
    c, a, b, d, e = Z(0, 0), Z(-1, -1), Z(1, -1), Z(-1, 1), Z(1, 1)
    k = np.uint32(0x3A83126F).view(np.float32)
    with np.errstate(all="ignore"):
        if inverted:
            edge = (c < F32(0.5) * (a + e) - k * c) | (c < F32(0.5) * (b + d) - k * c)
        else:
            edge = (c > fma32(c, k, F32(0.5) * (a + e))) | (c > fma32(c, k, F32(0.5) * (b + d)))
    best, bdx, bdy = c, np.zeros_like(ix), np.zeros_like(iy)
    for val, dx, dy in ((a, -1, -1), (b, 1, -1), (d, -1, 1), (e, 1, 1)):
        nearer = val > best if inverted else val < best
        bdx = np.where(nearer, dx, bdx)
        bdy = np.where(nearer, dy, bdy)
        best = np.where(nearer, val, best)
    return np.where(edge, bdx, 0), np.where(edge, bdy, 0)


def bilinear(img, cx, cy):
    """Bilinear sample of img [h, w, ch] at normalized (cx, cy), spec 42 draft 2 section 1, per channel in FP32 (one
    rounding per operation, no fma): x = cx w - 1/2, i0 = floor(x), fx = x - i0 (likewise y); clamp-to-edge texels;
    t0 = v00 + (v10 - v00) fx, t1 = v01 + (v11 - v01) fx; result (t0 (1 - fy)) + (t1 fy). Returns [..., ch] float32."""
    hh, ww = img.shape[:2]
    px = f32(cx) * F32(ww) - F32(0.5)
    py = f32(cy) * F32(hh) - F32(0.5)
    with np.errstate(invalid="ignore"):
        x0 = np.floor(px)
        y0 = np.floor(py)
    fx = (px - x0).astype(np.float32)
    fy = (py - y0).astype(np.float32)
    x0 = np.nan_to_num(x0, nan=0, posinf=ww, neginf=-1).astype(np.int64)
    y0 = np.nan_to_num(y0, nan=0, posinf=hh, neginf=-1).astype(np.int64)

    def at(yy, xx):
        return img[np.clip(yy, 0, hh - 1), np.clip(xx, 0, ww - 1)].astype(np.float32)
    fx, fy = fx[..., None], fy[..., None]
    v00, v10 = at(y0, x0), at(y0, x0 + 1)
    v01, v11 = at(y0 + 1, x0), at(y0 + 1, x0 + 1)
    t0 = v00 + (v10 - v00) * fx
    t1 = v01 + (v11 - v01) * fx
    return ((t0 * (F32(1) - fy)) + (t1 * fy)).astype(np.float32)


def ycocg(rgb):
    """max(RGB, 0) -> (Y, Co, Cg) of spec 42/43: Y = (0.25R + 0.5G) + 0.25B, Co = fma(R, 0.5, 0 G) - 0.5B,
    Cg = (0.5G - 0.25R) - 0.25B."""
    c = np.fmax(f32(rgb)[..., :3], F32(0))
    Rr, G, B = c[..., 0], c[..., 1], c[..., 2]
    Y = (F32(0.25) * Rr + F32(0.5) * G) + F32(0.25) * B
    Co = fma32(Rr, F32(0.5), F32(0) * G) - F32(0.5) * B
    Cg = (F32(0.5) * G - F32(0.25) * Rr) - F32(0.25) * B
    return Y, Co, Cg


def luma_fma(rgb):
    """Y = fma(B, 0.25, fma(R, 0.25, 0.5 G)) of max(RGB, 0)."""
    c = np.fmax(f32(rgb)[..., :3], F32(0))
    return fma32(c[..., 2], F32(0.25), fma32(c[..., 0], F32(0.25), F32(0.5) * c[..., 1]))


# ---- the output stage ----------------------------------------------------------------------------------------------

def output_stage(x):
    """Spec 43 on one frame. x: dict with
      N [Bh, Bw, 40] (FP16 values), color [.., .., >=3 (alpha: 4th channel, else 1)], motion [.., .., 2], depth [.., ..],
      color_base / motion_base / depth_base (x, y) (default 0), render_size (r_w, r_h), output_size (W, H),
      hdr, mv_lowres, depth_inverted (bool), jitter (2), mv_scale (2), mv_offset (2), reset (bool),
      E_cur, E_prev (FP16 values), e1, pre_exposure, L [2 H16, 2 W16] (FP16), Hc [H16, W16, 4] (FP16),
      table [2048, 8] (FP16, the frame's set; default: filter_table(filter_level(...))).
    Returns {"out": [H, W, 4] float32 RGBA, "L": [2 H16, 2 W16] f16, "Hc": [H16, W16, 4] f16, "Hh": [Bh, Bw, 4] f16}."""
    W, H = (int(v) for v in x["output_size"])
    rw, rh = (int(v) for v in x["render_size"])
    W16, H16 = ceil16(W), ceil16(H)
    N = np.asarray(x["N"], dtype=np.float16)
    Bh, Bw = N.shape[:2]
    hdr, mvlo, inv = bool(x["hdr"]), bool(x["mv_lowres"]), bool(x["depth_inverted"])
    color = f32(x["color"])
    alpha_tex = color[..., 3] if color.shape[-1] > 3 else np.ones(color.shape[:2], np.float32)
    motion, depth = f32(x["motion"]), (f32(x["depth"]) if mvlo else None)
    cb = tuple(x.get("color_base", (0, 0)))
    mb = tuple(x.get("motion_base", (0, 0)))
    zb = tuple(x.get("depth_base", (0, 0)))
    jx, jy = (F32(v) for v in x["jitter"])
    sx, sy = (F32(v) for v in x["mv_scale"])
    ox_, oy_ = (F32(v) for v in x.get("mv_offset", (0, 0)))
    reset = bool(x["reset"])
    table = x.get("table")
    if table is None:
        table = filter_table(filter_level(rw, rh, W, H))
    table = np.asarray(table, dtype=np.float16)
    L = np.asarray(x["L"], dtype=np.float16)
    Hc = np.asarray(x["Hc"], dtype=np.float16)

    with np.errstate(all="ignore"):
        inv_W, inv_H = F32(1) / F32(W), F32(1) / F32(H)
        rho_x, rho_y = F32(rw) * inv_W, F32(rh) * inv_H
        rho2_x, rho2_y = F32(0.5) * rho_x, F32(0.5) * rho_y
        yy, xx = np.meshgrid(np.arange(H16), np.arange(W16), indexing="ij")
        px, py = f32(xx) + F32(0.5), f32(yy) + F32(0.5)

        # 3.1 network channels: display position (u, v) = (trunc(s x), trunc(s y)) (s of spec 44; 1 for now)
        from clean.k.reference.layout import DEFAULT_S
        s_ = F32(x.get("s", DEFAULT_S))
        uu, vv = np.trunc(s_ * f32(xx)).astype(np.int64), np.trunc(s_ * f32(yy)).astype(np.int64)
        k = (uu % 2) + 2 * (vv % 2)
        bx, by = np.fmin(uu // 2, Bw - 1), np.fmin(vv // 2, Bh - 1)
        n = N[by, bx]  # [H16, W16, 40]
        nc = np.take_along_axis(n, (10 * k)[..., None] + np.arange(10)[None, None, :], axis=2)  # [.., 10]

        # 3.2 blend logit
        t = F32(0.5) * nc[..., 9].astype(np.float32)
        t2 = t * t
        g = (t * (t2 + F32(27))) / fma32(t2, F32(9), F32(27))
        alpha = np.where(t < F32(-3), F32(0), np.where(t > F32(3), F32(1), fma32(g, F32(0.5), F32(0.5))))
        alpha = alpha.astype(np.float32)

        # 3.3 motion
        jmx, jmy = jx - F32(0.5), jy - F32(0.5)
        if mvlo:
            ix, iy = R(fma32(px, rho_x, jmx)).astype(np.int64), R(fma32(py, rho_y, jmy)).astype(np.int64)
            dx, dy = nearest_depth_pixel(depth, ix, iy, zb, (rw, rh), inv)
            m = motion[motion_index_lo(iy, dy, mb[1], rh), motion_index_lo(ix, dx, mb[0], rw)]
            spx, spy = sx / rho_x, sy / rho_y
        else:
            mx_, my_ = R(fma32(px, F32(1), F32(-0.5))).astype(np.int64), R(fma32(py, F32(1), F32(-0.5))).astype(np.int64)
            m = motion[tap_index(my_, 0, mb[1], H), tap_index(mx_, 0, mb[0], W)]
            spx, spy = sx, sy
        wx = fma32(spx, m[..., 0] + ox_, px)
        wy = fma32(spy, m[..., 1] + oy_, py)

        # 3.4 validity, 3.5 colour history
        valid = (~np.full(px.shape, reset)) & (wx * inv_W >= 0) & (wx * inv_W <= 1) & (wy * inv_H >= 0) & \
            (wy * inv_H <= 1)
        Hh_ = bilinear(Hc, wx * (F32(1) / F32(W16)), wy * (F32(1) / F32(H16)))  # Y, Co, Cg, age

        # 3.6 luma history (sub-samples)
        def axis_coeffs(w):
            q = fma32(w - F32(0.5), F32(2), F32(0.5))
            tt = np.floor(q - F32(0.5)) + F32(0.5)
            f = sat(q - tt).astype(np.float32)
            f2 = f * f
            f3 = f2 * f
            c0 = f2 - F32(0.5) * (f + f3)
            c1 = (F32(1.5) * f3 - F32(2.5) * f2) + F32(1)
            c3 = F32(0.5) * (f3 - f2)
            c2 = ((F32(1) - c0) - c1) - c3
            beta = c1 / (c1 + c2)
            betap = F32(1) - beta
            nn = np.nan_to_num(tt - F32(0.5), nan=0.0, posinf=1e9, neginf=-1e9).astype(np.int64)
            return nn, (c0, c1, c2, c3, beta, betap)

        nx_, cx_ = axis_coeffs(wx)
        ny_, cy_ = axis_coeffs(wy)
        LH, LW = L.shape

        def ell(v):
            v = f32(v)
            return np.where(v < C0P99, v, fma32(v - C0P99, F32(0.001), C0P99)).astype(np.float32)

        def Ltex(yv, xv):
            return ell(L[np.clip(yv, 0, LH - 1), np.clip(xv, 0, LW - 1)])

        def cubic(Am1, A0, A1, A2, cf):
            c0, c1, c2, c3, beta, betap = cf
            inner = fma32(beta, A0, betap * A1)
            inner = fma32(LAMBDA * c0, sat(Am1), inner)
            inner = fma32(LAMBDA * (c1 - beta), sat(A0), inner)
            inner = fma32(LAMBDA * (c2 - betap), sat(A1), inner)
            return fma32(LAMBDA * c3, sat(A2), inner)

        ellv = np.zeros(px.shape + (4,), np.float32)
        for i in (0, 1):
            for j in (0, 1):
                rows = []
                for mrow in (-1, 0, 1, 2):
                    yrow = ny_ + i + mrow
                    A = [Ltex(yrow, nx_ + j + mm) for mm in (-1, 0, 1, 2)]
                    rows.append(cubic(*A, cx_))
                val = cubic(*rows, cy_)
                near = [Ltex(ny_ + i + a2, nx_ + j + b2) for a2 in (0, 1) for b2 in (0, 1)]
                lo_, hi_ = np.fmin.reduce(near), np.fmax.reduce(near)
                ellv[..., j + 2 * i] = np.fmin(np.fmax(val, lo_), hi_)

        # 3.7 exposure, 3.8 / 3.9 history luma
        def fp16val(e):
            e = F32(np.float16(e))
            return F32(1) if e == 0 else e
        Ecur, Eprev = F32(x["e1"]) * fp16val(x["E_cur"]), F32(x["e1"]) * fp16val(x["E_prev"])
        if hdr:
            kappa = (F32(1) / F32(x["pre_exposure"])) * Ecur
            kappa = F32(1) if kappa == 0 else kappa
            rho_e = Ecur / Eprev
        else:
            kappa, rho_e = F32(1), F32(1)

        def stored(Y):
            return (Y * encode(Y)).astype(np.float32) if hdr else f32(Y)

        if hdr:
            Hl = stored(rho_e * (ellv * decode(ellv)))  # 4.9 (d2): the product l D(l) first
        else:
            Hl = (rho_e * ellv).astype(np.float32)

        # 3.10 confidence
        # sub-sample (j, i) is index j + 2 i: H00 = 0, H10 = 1, H01 = 2, H11 = 3
        # 4.10 (d2): fma(alpha, Cg, fma(Y, Co, fma(H00, H10, H01 H11)))
        prodH = Hl[..., 2] * Hl[..., 3]
        chk = fma32(alpha, Hh_[..., 2], fma32(Hh_[..., 0], Hh_[..., 1], fma32(Hl[..., 0], Hl[..., 1], prodH)))
        valid = valid & np.isfinite(chk)
        alpha_p = np.where(valid, alpha, F32(1)).astype(np.float32)
        b_p = np.where(valid, Hh_[..., 3], F32(1)).astype(np.float32)
        Hl = np.where(valid[..., None], Hl, F32(0)).astype(np.float32)
        Hh_ = np.where(valid[..., None], Hh_, F32(0)).astype(np.float32)
        m0 = fma32(alpha_p, F32(1) - b_p, b_p)
        m1 = fma32(b_p, F32(1) - alpha_p * alpha_p, alpha_p * alpha_p)

        # 4.1 local origin
        gx16, gy16 = f32(16 * (xx // 16)) + F32(0.5), f32(16 * (yy // 16)) + F32(0.5)
        o_x = R(fma32(gx16, rho_x, jmx)) - F32(4)
        o_y = R(fma32(gy16, rho_y, jmy)) - F32(4)
        Ux = h(fma32(px, rho_x, jmx) - o_x)
        Uy = h(fma32(py, rho_y, jmy) - o_y)

        # 4.2 stored current colour per render tap
        def color_at(ixr, iyr):
            yi, xi = tap_index(iyr, 0, cb[1], rh), tap_index(ixr, 0, cb[0], rw)
            return color[yi, xi], alpha_tex[yi, xi]

        def stored_tap(ixr, iyr):
            rgb, a_ = color_at(ixr, iyr)
            Y, Co, Cg = ycocg(rgb)
            if hdr:
                kY = kappa * Y
                e = encode(kY)
                return h(kY * e), h((kappa * Co) * e), h((kappa * Cg) * e), h(a_)
            return h(kappa * Y), h(kappa * Co), h(kappa * Cg), h(a_)

        # 4.3 bilinear current colour (FP16)
        fUx, fUy = np.floor(Ux), np.floor(Uy)
        fx = sat(h(Ux - fUx))
        fy = sat(h(Uy - fUy))
        one = F16(1)
        wts = [h(h(one - fx) * h(one - fy)), h(fx * h(one - fy)), h(h(one - fx) * fy), h(fx * fy)]
        bx0 = (o_x + fUx.astype(np.float32)).astype(np.int64)
        by0 = (o_y + fUy.astype(np.float32)).astype(np.int64)
        cbar = None
        for (ddx, ddy), wgt in zip(((0, 0), (1, 0), (0, 1), (1, 1)), wts):
            v = stored_tap(bx0 + ddx, by0 + ddy)
            if cbar is None:
                cbar = [h(vv * wgt) for vv in v]
            else:
                cbar = [fma16(vv, wgt, cc) for vv, cc in zip(v, cbar)]
        Ybar, Cobar, Cgbar, Abar = cbar

        # 5.1 table index
        def zq(nv):
            nv = np.asarray(nv, np.float16)
            z = h(nv.astype(np.float64) / h(one + np.abs(nv)).astype(np.float64))
            return np.clip(rni(h(z * QSEL)).astype(np.int64) + 8, 0, 15)
        q0, q1 = zq(nc[..., 0]), zq(nc[..., 1])
        F0 = THETA > m0
        F1 = (m0 < F32(0.25)) & (alpha < F32(0.25))
        big = Ybar > F16(0.8)
        g2 = big.reshape(H16 // 2, 2, W16 // 2, 2).any(axis=(1, 3))
        F2 = np.repeat(np.repeat(g2, 2, axis=0), 2, axis=1)
        tix = (16 * q0 + q1) + 256 * (F0.astype(np.int64) + 2 * F1.astype(np.int64) + 4 * F2.astype(np.int64))
        ent = table[tix]  # [H16, W16, 8]
        tA, tC, tB, tS, tK, tk6 = (ent[..., c] for c in (0, 1, 2, 3, 5, 6))
        O = ent[..., 4].view(np.uint16) != 0

        # 5.2 line coordinates (sub-samples)
        Xs = [h(fma32(f32(2 * xx + jj) + F32(0.5), rho2_x, jmx) - o_x) for jj in (0, 1)]
        Ys = [h(fma32(f32(2 * yy + ii) + F32(0.5), rho2_y, jmy) - o_y) for ii in (0, 1)]
        uP = np.where(O, Uy, Ux).astype(np.float16)
        vP = np.where(O, Ux, Uy).astype(np.float16)

        # 5.3 taps (shared by the four sub-samples)
        v0 = h(np.floor(vP) - one)
        cl = h(uP + h(h(v0 - vP) * tS))
        lines = []
        for ell_i in range(4):
            if ell_i > 0:
                cl = h(cl + tS)
            lv = h(v0 + F16(ell_i))
            col0 = h(rni(cl) - one)
            taps = []
            for tcol in range(3):
                colv = h(col0 + F16(tcol))
                # render pixel o + (column, line) [O = 0] or o + (line, column)
                rx = np.where(O, o_x + lv.astype(np.float32), o_x + colv.astype(np.float32)).astype(np.int64)
                ry = np.where(O, o_y + colv.astype(np.float32), o_y + lv.astype(np.float32)).astype(np.int64)
                rgb, _ = color_at(rx, ry)
                Yt = luma_fma(rgb)
                kY = kappa * Yt
                T = h(stored(kY))
                taps.append((colv, T))
            lines.append((lv, taps))

        # 5.4 - 5.6 weights and filtered values; 6.1 blend weights
        if True:
            ku_x, kv_y = h(F32(-8) / rho2_x), h(F32(-8) / rho2_y)
            k_u = np.where(O, kv_y, ku_x).astype(np.float16)
            k_v = np.where(O, ku_x, kv_y).astype(np.float16)
        Fv = np.zeros(px.shape + (4,), np.float16)
        gam = np.zeros(px.shape + (4,), np.float16)
        for i in (0, 1):
            for j in (0, 1):
                us = np.where(O, Ys[i], Xs[j]).astype(np.float16)
                vs = np.where(O, Xs[j], Ys[i]).astype(np.float16)
                sgT = sg = None
                bestE = bestT = None
                for lv, taps in lines:
                    dv = h(lv - vs)
                    t2 = h(h(dv * dv) * tA)
                    t3 = h(dv * tB)
                    du = mm = None
                    for ti, (colv, T) in enumerate(taps):
                        if ti == 0:
                            du = h(colv - us)
                            mm = fma16(du, t3, t2)
                        else:
                            du = h(du + one)
                            mm = h(mm + t3)
                        E = fma16(h(du * du), tC, mm)
                        gg = h(np.exp2(E.astype(np.float64)))
                        if sgT is None:
                            sgT, sg = h(gg * T), gg
                            bestE, bestT = E, T
                        else:
                            sgT, sg = fma16(gg, T, sgT), h(gg + sg)
                            repl = ~(bestE > E)
                            bestE = np.where(repl, E, bestE).astype(np.float16)
                            bestT = np.where(repl, T, bestT).astype(np.float16)
                Fq = h(sgT.astype(np.float64) / sg.astype(np.float64))
                Fq = np.where(bestE < F16(-12), bestT, Fq).astype(np.float16)
                Fv[..., j + 2 * i] = Fq
                # 6.1 per sub-sample blend weight
                dvf = h(rni(vs) - vs)
                up = fma16(dvf, tS, us)
                duf = h(rni(up) - up)
                wu = sat(fma16(np.abs(duf), k_u, F16(4.5)))
                wv = sat(fma16(np.abs(dvf), k_v, F16(4.5)))
                gam[..., j + 2 * i] = h(fma16(wv, tK, tk6) * wu)

        # 6.1 (d2): h(0.25 h(h(gamma00 + gamma01) + h(gamma10 + gamma11)))
        gbar = h(F16(0.25) * h(h(gam[..., 0] + gam[..., 2]) + h(gam[..., 1] + gam[..., 3])))
        mh = h(m0)
        rr = h(h(F16(0.1)) + h(h(F16(0.9)) * mh))
        d0 = h(h(mh - MU) * NU)
        d1 = h(h(rr - MU) * NU)
        beta_c = sat(fma16(gbar, h(rr - d1), d1))
        beta = np.stack([sat(fma16(gam[..., s], h(mh - d0), d0)) for s in range(4)], axis=-1)

        # 6.2 guard
        guard = (((Ybar.astype(np.float32) * Fv[..., 0]) * Fv[..., 1]) * Fv[..., 2]) * Fv[..., 3]
        bad = ~np.isfinite(guard)
        Fv = np.where(bad[..., None], F16(0), Fv).astype(np.float16)
        Ybar, Cobar, Cgbar = (np.where(bad, F16(0), v).astype(np.float16) for v in (Ybar, Cobar, Cgbar))

        # 6.3 / 6.4 new luma and luma history
        Nn = fma32(beta.astype(np.float32), Fv.astype(np.float32) - Hl, Hl)
        Lout = h(np.where(Nn < C0P99, Nn, fma32(Nn - C0P99, F32(1000), C0P99)))
        Lnew = np.zeros((2 * H16, 2 * W16), np.float16)
        for i in (0, 1):
            for j in (0, 1):
                Lnew[i::2, j::2] = Lout[..., j + 2 * i]

        # 6.5 colour
        bc = beta_c.astype(np.float32)
        Yc = fma32(Hh_[..., 0], F32(1) - bc, bc * Ybar.astype(np.float32))
        Coc = fma32(Hh_[..., 1], F32(1) - bc, bc * Cobar.astype(np.float32))
        Cgc = fma32(Hh_[..., 2], F32(1) - bc, bc * Cgbar.astype(np.float32))
        q = F32(1) / np.fmax(Yc, EPS5)  # one correctly rounded reciprocal (d2)
        rCo, rCg = Coc * q, Cgc * q
        Yo = F32(0.25) * (Nn[..., 3] + (Nn[..., 2] + (Nn[..., 0] + Nn[..., 1])))
        Coo, Cgo = rCo * Yo, rCg * Yo

        # 6.6 colour history out
        Yp = np.fmax(Yo, F32(0))
        eps = EPS5
        s1 = np.where(Cgo - eps < -Yp, eps - Cgo, Yp)
        s2 = np.where((Coo - Cgo) - eps < -s1, (Cgo - Coo) + eps, s1)
        s3 = np.where((-Coo - Cgo) - eps < -s2, (Coo + Cgo) + eps, s2)
        kk = np.where(s3 <= Yp, F32(1), Yp / s3).astype(np.float32)
        age = sat(np.fmax(m1 / (m1 + F32(1)), A_MIN))
        Hcn = np.stack([h(Yp), h(Coo * kk), h(Cgo * kk), h(age)], axis=-1)

        # 6.7 game output
        if hdr:
            d = decode(Yo)
            kinv = F32(1) / kappa
            Yg, Cog, Cgg = kinv * (Yo * d), kinv * (Coo * d), kinv * (Cgo * d)
        else:
            Yg, Cog, Cgg = Yo, Coo, Cgo
        Rg = (Yg + Cog) - Cgg
        Gg = Cgg + fma32(Cog, F32(0), Yg)
        Bg = (Yg - Cog) - Cgg
        out = np.stack([Rg, Gg, Bg, Abar.astype(np.float32)], axis=-1)[:H, :W].astype(np.float32)

        # 6.8 hidden history (base grid)
        Nb = N.astype(np.float32)
        hk = []
        for kk_ in range(4):
            c = Nb[..., 10 * kk_ + 3:10 * kk_ + 8]
            hk.append(F32(0.2) * ((((c[..., 0] + c[..., 1]) + c[..., 2]) + c[..., 3]) + c[..., 4]))
        ssum = hk[1] * hk[1]
        for c in (0, 2, 3):
            ssum = fma32(hk[c], hk[c], ssum)
        nu_h = fn32(lambda s: 1.0 / np.sqrt(s), ssum + F32(1))  # 1/sqrt evaluated once, rounded to FP32 (d2)
        Hhn = np.stack([h(hk[c] * nu_h) for c in range(4)], axis=-1)

    return {"out": out, "L": Lnew, "Hc": Hcn, "Hh": Hhn}
