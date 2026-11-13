# SPDX-License-Identifier: MIT
"""NumPy reference of the model K input stage (spec 42 draft 2): one frame's game inputs and the previous frame's
colour history Hc and hidden history Hh -> the network input map [Bh, Bw, 16] (FP16).

FP32 operations in the order written, fma with one rounding, R = round half away (clean/k/reference/fp.py).
Functions (reciprocal, division, log2, 2^x, ln) are the mathematical function rounded to FP32. Draft 2 fixes the
order of the bilinear samples (section 1, clean/k/reference/outstage.py `bilinear`), of the slot sum of channel
4k + 2 ((s0 + s3) + (s1 + s2)) and of the power in S (2^(log2(z') c), log2 and 2^x each rounded once).
"""
import numpy as np

from clean.k.reference.fp import F32, R, f32, fma32, fn32, h
from clean.k.reference.outstage import (bilinear, ceil16, decode, luma_fma, motion_index_lo, nearest_depth_pixel,
                                        tap_index, ycocg)

C1E4 = np.uint32(0x38D1B717).view(np.float32)  # 1e-4
C1E6 = np.uint32(0x358637BD).view(np.float32)  # 1e-6
LN2 = np.uint32(0x3F317218).view(np.float32)
NINTH = np.uint32(0x3DE38E39).view(np.float32)
C0P2 = np.uint32(0x3E4CCCCD).view(np.float32)
S_T = np.uint32(0x3B4D2E1C).view(np.float32)
S_P = np.uint32(0x3ED55555).view(np.float32)
S_A = np.uint32(0x3F870A3D).view(np.float32)
S_B = np.uint32(0xBD6147AE).view(np.float32)
S_C = np.uint32(0x414EB852).view(np.float32)


def host_factors(W, H, rw, rh, Bw, Bh, sigma, mv_lowres, s=1.0):
    """Spec 42 section 8 (per axis x, y): xi, a, h, g, sigma', and the [MVhi] motion-pixel factor."""
    out = {}
    for ax, n, r, B, sg in (("x", W, rw, Bw, sigma[0]), ("y", H, rh, Bh, sigma[1])):
        xi = (F32(1) / F32(F32(n) * F32(s))) * F32(0.5)
        out["xi_" + ax] = xi
        out["a_" + ax] = F32(r) * xi
        out["h_" + ax] = (F32(n) / F32(ceil16(n))) * xi
        out["g_" + ax] = F32(0.25) / F32(B)
        inv = F32(1) / F32(n)
        out["sp_" + ax] = F32(sg) / (inv * F32(r)) if mv_lowres else F32(sg) / (inv * F32(n))
        out["mp_" + ax] = xi * F32(n)
    return out


def ln32(z):
    """ln as log2 * ln2 (spec 42 6.2): RN32(RN32(log2 z) ln2)."""
    return fn32(np.log2, z) * LN2


def phi(Y, hdr):
    if hdr:
        Yp = np.fmax(Y, F32(0))
        return ln32(fma32(Yp, F32(12), F32(1))) / (Yp + C1E6)
    return fma32(Y, F32(3), F32(1))


def input_stage(x):
    """Spec 42 on one frame. x: dict with color [.., .., >=3], motion [.., .., 2], depth [.., ..] ([MVlo]),
    color_base / motion_base / depth_base (default 0), render_size (r_w, r_h), output_size (W, H), hdr, mv_lowres,
    depth_inverted, jitter, mv_scale, mv_offset, pre_exposure, reset, E_cur, E_prev, E_aux (FP16 values), e1, e2,
    Hc [H16, W16, 4], Hh [Bh, Bw, 4] (FP16). Returns [Bh, Bw, 16] float16."""
    from clean.k.reference.layout import base_size
    W, H = (int(v) for v in x["output_size"])
    rw, rh = (int(v) for v in x["render_size"])
    from clean.k.reference.layout import DEFAULT_S
    s_ = float(x.get("s", DEFAULT_S))  # spec 44's base-size factor
    Bw, Bh = tuple(x["base"]) if x.get("base") is not None else base_size(W, H, s_)  # "base": test-only override
    hdr, mvlo, inv = bool(x["hdr"]), bool(x["mv_lowres"]), bool(x["depth_inverted"])
    color, motion = f32(x["color"]), f32(x["motion"])
    depth = f32(x["depth"]) if mvlo else None
    cb = tuple(x.get("color_base", (0, 0)))
    mb = tuple(x.get("motion_base", (0, 0)))
    zb = tuple(x.get("depth_base", (0, 0)))
    jx, jy = (F32(v) for v in x["jitter"])
    ox_, oy_ = (F32(v) for v in x.get("mv_offset", (0, 0)))
    reset = bool(x["reset"])
    Hc = np.asarray(x["Hc"], np.float16)
    Hh = np.asarray(x["Hh"], np.float16)
    hf = host_factors(W, H, rw, rh, Bw, Bh, x["mv_scale"], mvlo, s_)

    def ev(e):
        e = F32(np.float16(e))
        return F32(1) if e == 0 else e
    with np.errstate(all="ignore"):
        ce = F32(x["e1"]) * ev(x["E_cur"])
        if hdr:
            kappa = (F32(1) / F32(x["pre_exposure"])) * ce
            kappa = F32(1) if kappa == 0 else kappa
            rho = ce / (F32(x["e1"]) * ev(x["E_prev"]))
        else:
            kappa, rho = F32(1), F32(1)
        lam = (F32(x["e2"]) * ev(x["E_aux"])) / ce

        # ---- section 5, per fine position (4 Bh x 4 Bw) ----
        fy, fx = np.meshgrid(np.arange(4 * Bh), np.arange(4 * Bw), indexing="ij")
        px, py = f32(fx) + F32(0.5), f32(fy) + F32(0.5)
        ix = R(jx + fma32(px, hf["a_x"], F32(-0.5))).astype(np.int64)
        iy = R(jy + fma32(py, hf["a_y"], F32(-0.5))).astype(np.int64)
        if mvlo:
            dx, dy = nearest_depth_pixel(depth, ix, iy, zb, (rw, rh), inv)
            M = motion[motion_index_lo(iy, dy, mb[1], rh), motion_index_lo(ix, dx, mb[0], rw)]
        else:
            mx = R(fma32(px, hf["mp_x"], F32(-0.5))).astype(np.int64)
            my = R(fma32(py, hf["mp_y"], F32(-0.5))).astype(np.int64)
            M = motion[tap_index(my, 0, mb[1], H), tap_index(mx, 0, mb[0], W)]
        mxv = hf["sp_x"] * (M[..., 0] + ox_)
        myv = hf["sp_y"] * (M[..., 1] + oy_)
        wx, wy = px + mxv, py + myv
        fvalid = (wx * hf["xi_x"] >= 0) & (wx * hf["xi_x"] <= 1) & (wy * hf["xi_y"] >= 0) & (wy * hf["xi_y"] <= 1)
        smp = bilinear(Hc, wx * hf["h_x"], wy * hf["h_y"])[..., :3]
        if hdr:
            f = decode(smp[..., 0])
            hist = rho * (smp * f[..., None])
        else:
            hist = smp
        nx = (f32(4 * (fx // 4) + 2) + mxv) * hf["g_x"]
        ny = (f32(4 * (fy // 4) + 2) + myv) * hf["g_y"]

        # ---- section 6, per display position (2 Bh x 2 Bw) ----
        dv, du = np.meshgrid(np.arange(2 * Bh), np.arange(2 * Bw), indexing="ij")
        fine = [(2 * du + 0, 2 * dv + 0), (2 * du + 1, 2 * dv + 0), (2 * du + 0, 2 * dv + 1), (2 * du + 1, 2 * dv + 1)]
        valid = ~np.full(du.shape, reset)
        for (fxx, fyy) in fine:
            valid &= fvalid[fyy, fxx]

        def kappa_rgb(rgb):
            return kappa * np.fmax(f32(rgb)[..., :3], F32(0))

        def color_at(ixr, iyr):
            return color[tap_index(iyr, 0, cb[1], rh), tap_index(ixr, 0, cb[0], rw)]

        cx = R(jx + fma32(f32(2 * du + 1), hf["a_x"], F32(-0.5))).astype(np.int64)
        cy = R(jy + fma32(f32(2 * dv + 1), hf["a_y"], F32(-0.5))).astype(np.int64)
        A = [np.zeros(du.shape, np.float32) for _ in range(3)]
        lo = hi = None
        for ddy in (-1, 0, 1):
            for ddx in (-1, 0, 1):
                rgb = color[tap_index(cy, ddy, cb[1], rh), tap_index(cx, ddx, cb[0], rw)]
                Y, Co, Cg = ycocg(kappa_rgb(rgb))
                if hdr:
                    om = np.fmax(F32(1) / (Y + F32(1)), C1E4)
                    A = [fma32(om, v, a) for v, a in zip((Y, Co, Cg), A)]
                else:
                    A = [a + v for v, a in zip((Y, Co, Cg), A)]
                ph = phi(Y, hdr)
                vals = [Y * ph, Co * ph, Cg * ph]
                lo = vals if lo is None else [np.fmin(a, b) for a, b in zip(lo, vals)]
                hi = vals if hi is None else [np.fmax(a, b) for a, b in zip(hi, vals)]
        mu = [a * NINTH for a in A]
        if hdr:
            den = np.fmax(F32(1) - mu[0], C1E4)
            mu = [m / den for m in mu]
        pm = phi(mu[0], hdr)
        muh = [m * pm for m in mu]
        eta = [F32(0.5) * (b - a) for a, b in zip(lo, hi)]
        nu = [fma32(b - a, F32(8), F32(1)) for a, b in zip(lo, hi)]
        Dsum = np.zeros(du.shape, np.float32)
        for (fxx, fyy) in fine:
            hv = hist[fyy, fxx]
            ph = phi(hv[..., 0], hdr)
            delta = [np.fmax(np.abs(muh[c] - hv[..., c] * ph) - eta[c], F32(0)) / nu[c] for c in range(3)]
            score = np.fmin(fma32(delta[2], F32(0.5), fma32(delta[0], F32(2), F32(0.5) * delta[1])), F32(1))
            Dsum = Dsum + score
        D = Dsum * F32(0.25)

        # own sample
        ofx, ofy = 2 * du + 1 - (dv % 2), 2 * dv + (du % 2)
        opx, opy = f32(ofx) + F32(0.5), f32(ofy) + F32(0.5)
        rx = R(jx + fma32(opx, hf["a_x"], F32(-0.5))).astype(np.int64)
        ry = R(jy + fma32(opy, hf["a_y"], F32(-0.5))).astype(np.int64)
        Yc = luma_fma(kappa_rgb(color_at(rx, ry)))
        Yh = np.where(valid, hist[ofy, ofx, 0], Yc).astype(np.float32)
        hid = bilinear(Hh, nx[ofy, ofx], ny[ofy, ofx])
        hid = np.where(valid[..., None], hid, F32(0)).astype(np.float32)

        def Lf(Y):
            return Y if hdr else Y / np.fmax(F32(1) - Y, C0P2)

        def Tf(z):
            return z * np.fmax(F32(1) / (z + F32(1)), C1E4)

        def Sf(z):
            zp = np.where(np.isnan(z), F32(0), np.fmin(np.fmax(z, F32(0)), F32(1))).astype(np.float32)
            pw = fn32(np.exp2, fn32(np.log2, zp) * S_P)  # 2^(log2(z') c): two functions, each rounded once
            return np.where(zp > S_T, fma32(pw, S_A, S_B), S_C * zp).astype(np.float32)

        ch0 = Sf(Tf(lam * Lf(Yc)))
        ch1 = Sf(Tf(lam * Lf(Yh)))
        ch3 = np.where(valid, D, F32(0))
        ch3 = np.where(np.isfinite(ch3), ch3, F32(0))

        out = np.zeros((Bh, Bw, 16), np.float16)
        sm = [hid[m // 2::2, m % 2::2] for m in range(4)]  # slot m's hidden sample per base position
        hidsum = F32(0.25) * ((sm[0] + sm[3]) + (sm[1] + sm[2]))
        hidsum = np.where(np.isfinite(hidsum), hidsum, F32(0))
        for k in range(4):
            sl = (slice(k // 2, None, 2), slice(k % 2, None, 2))
            out[..., 4 * k + 0] = h(ch0[sl])
            out[..., 4 * k + 1] = h(ch1[sl])
            out[..., 4 * k + 2] = h(hidsum[..., k])
            out[..., 4 * k + 3] = h(ch3[sl])
    return out
