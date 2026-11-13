# SPDX-License-Identifier: MIT
"""NumPy reference of spec/05-coefficient-decoder.md (stage "dec0" of spec/90-test-interface.md): the clean team's
oracle.

Where spec 05 says "computed in single" or RN32, the arithmetic below is float32 NumPy, one operation at a time in the
order the spec writes it (NumPy does not fuse multiply-adds). The projection sums are exact (float64).

    out = decode(x, params)
    x:      the dict of spec 90 "Coefficient decoder (spec 05): stage dec0"
    params: {"W": [68, 64], "b": [68], "S": scalar, "T": scalar}
    out:    {"A", "B", "C": [rh, rw, 3], "latent": [rh, rw, 4]} in the spec 90 value grids, plus "codes_A", "codes_B"
            (uint8), "codes_L" (int8), and the intermediates "h" ([fh/2, fw/2, 68]) and "p" (previous latent, FP16)

Choices where spec 05 (draft 1) leaves room are marked "ASSUMPTION" and asked in QUESTIONS.md (spec 05 section).
"""
import numpy as np

from .fp8 import q8, f16

N_OUT = 68         # projection outputs
N_PER_PIXEL = 17   # values per render pixel
N_IN = 64          # feature channels

f32 = np.float32


def rn16(x):
    """Round to FP16 (nearest even); returned as float32."""
    return np.asarray(x, dtype=np.float32).astype(np.float16).astype(np.float32)


# ---- element functions (also used to build the kernel's lookup tables) ---------------------------------------------

def sigmoid16(a):
    """RN16(sigma(a)), sigma computed in single: 1 / (1 + exp(-a)) in float32."""
    a = np.asarray(a, dtype=np.float32)
    with np.errstate(over="ignore"):
        return rn16(f32(1) / (f32(1) + np.exp(-a)))


def tanh16(v):
    """th(v) = RN16(tanh(v)) (tanh exact, then one rounding)."""
    return rn16(np.tanh(np.asarray(v, dtype=np.float64)))


def rho32(a8, T):
    """rho = RN32(tanh(RN32(T * a8)))."""
    t = f32(T) * np.asarray(a8, dtype=np.float32)
    return np.tanh(t.astype(np.float64)).astype(np.float32)


def snorm8(k):
    """Stored 8-bit signed code -> value RN32(k / 127), -128 -> -1."""
    k = np.asarray(k, dtype=np.int64)
    return np.maximum(k / 127.0, -1.0).astype(np.float32)


# ---- stage ---------------------------------------------------------------------------------------------------------

def check_inputs(x):
    m = np.asarray(x["motion"])
    if m.ndim != 3 or m.shape[2] != 2:
        raise ValueError(f"motion must be [rh, rw, 2], got {m.shape}")
    rh, rw = m.shape[:2]
    fw, fh = (int(v) for v in x["feature_size"])
    dw, dh = (int(v) for v in x["recon_size"])
    if np.shape(x["latent_prev"]) != (rh, rw, 4):
        raise ValueError(f"latent_prev has shape {np.shape(x['latent_prev'])}, expected {(rh, rw, 4)}")
    if fw < rw or fh < rh or fw % 2 or fh % 2:
        raise ValueError(f"feature_size {(fw, fh)} must be even and at least the render size {(rw, rh)}")
    if np.shape(x["features"]) != (fh // 2, fw // 2, N_IN):
        raise ValueError(f"features has shape {np.shape(x['features'])}, expected {(fh // 2, fw // 2, N_IN)}")
    if dw <= 0 or dh <= 0:
        raise ValueError(f"recon_size {(dw, dh)} must be positive")
    return rw, rh, fw, fh, dw, dh


def project(Z, W, b):
    """Spec 05 section 1: h = q8(RN16(RN16(b + W[:, :32] Z[:32]) + W[:, 32:] Z[32:])), exact partial sums."""
    Z = np.asarray(Z, dtype=np.float64)
    W = np.asarray(W, dtype=np.float64)
    u = f16(np.asarray(b, dtype=np.float64) + Z[..., :32] @ W[:, :32].T)
    v = f16(u + Z[..., 32:] @ W[:, 32:].T)
    return q8(v)


def pixel_values(h, rw, rh):
    """a[y, x, k] = h[floor(y/2), floor(x/2), 17 s + k], s = (x mod 2) + 2 (y mod 2)."""
    Y, X = np.mgrid[0:rh, 0:rw]
    s = (X % 2) + 2 * (Y % 2)
    idx = N_PER_PIXEL * s[..., None] + np.arange(N_PER_PIXEL)
    return np.take_along_axis(h[Y // 2, X // 2], idx, axis=-1)            # [rh, rw, 17]


def softmax3(a):
    """Spec 05 section 4 in single: e_i = exp(a_i - max), B_i = e_i / ((e_0 + e_1) + e_2)."""
    a = np.asarray(a, dtype=np.float32)
    m = np.maximum(np.maximum(a[..., 0], a[..., 1]), a[..., 2])
    e = np.exp(a - m[..., None])
    s = (e[..., 0] + e[..., 1]) + e[..., 2]
    return e / s[..., None]


def coefficient_c(a6, a7, a8, S, T):
    gx, gy = sigmoid16(a6), sigmoid16(a7)
    sx, sy = f32(S) * gx, f32(S) * gy
    rho = rho32(a8, T)
    return np.stack([rn16(sx * sx), rn16(sy * sy), rn16((sx * sy) * rho)], axis=-1)


def bilinear_single(tex, X, Y):
    """Bilinear sample at render-pixel coordinates (texel centres at i + 0.5), clamp-to-edge, in single.

    ASSUMPTION (order of the single-precision operations): t = X - 0.5, f = t - floor(t),
    w00 = (1-fx)(1-fy), w10 = fx(1-fy), w01 = (1-fx)fy, w11 = fx fy, P = ((w00 t00 + w10 t10) + w01 t01) + w11 t11."""
    tex = np.asarray(tex, dtype=np.float32)
    h, w = tex.shape[:2]
    tx, ty = X - f32(0.5), Y - f32(0.5)
    x0, y0 = np.floor(tx), np.floor(ty)
    fx, fy = (tx - x0)[..., None], (ty - y0)[..., None]
    x0, y0 = x0.astype(np.int64), y0.astype(np.int64)
    xa, xb = np.clip(x0, 0, w - 1), np.clip(x0 + 1, 0, w - 1)
    ya, yb = np.clip(y0, 0, h - 1), np.clip(y0 + 1, 0, h - 1)
    one = f32(1)
    w00, w10 = (one - fx) * (one - fy), fx * (one - fy)
    w01, w11 = (one - fx) * fy, fx * fy
    return ((w00 * tex[ya, xa] + w10 * tex[ya, xb]) + w01 * tex[yb, xa]) + w11 * tex[yb, xb]


def previous_latent(x, rw, rh, dw, dh):
    """Spec 05 section 6: p [rh, rw, 4] (FP16 values; 0 where invalid)."""
    M = np.asarray(x["motion"], dtype=np.float32)
    j = np.asarray(x["jitter"], dtype=np.float32)
    pj = np.asarray(x["jitter_prev"], dtype=np.float32)
    # ASSUMPTION: k = RN32(D / R), and hx = RN32(RN32(RN32((x + 0.5) - jx) * kx) + Mx), X = RN32(RN32(hx / kx) + pjx).
    kx, ky = f32(dw / rw), f32(dh / rh)
    Y, X = np.mgrid[0:rh, 0:rw]
    hx = ((X.astype(np.float32) + f32(0.5)) - j[0]) * kx + M[..., 0]
    hy = ((Y.astype(np.float32) + f32(0.5)) - j[1]) * ky + M[..., 1]
    valid = (hx >= 0) & (hx <= f32(dw)) & (hy >= 0) & (hy <= f32(dh))
    if bool(x["reset"]):
        valid[:] = False
    Xp, Yp = hx / kx + pj[0], hy / ky + pj[1]
    codes = np.rint(np.asarray(x["latent_prev"], dtype=np.float64) * 127.0).astype(np.int64)
    P = bilinear_single(snorm8(codes), np.where(valid, Xp, 0), np.where(valid, Yp, 0))
    return np.where(valid[..., None], rn16(P), f32(0))


def latent_update(a, p):
    """Spec 05 section 7 for channels c = 0..3: a [.., 17], p [.., 4] -> L [.., 4] (FP16 values)."""
    z = rn16(f32(0.5) + f32(0.5) * tanh16(a[..., 13:17] / 2))
    g = tanh16(a[..., 9:13])
    w = rn16(f32(1) - z)
    return rn16(rn16(w * p) + rn16(z * g))


def unorm8_code(v):
    return np.rint(np.clip(np.asarray(v, dtype=np.float32), f32(0), f32(1)) * f32(255)).astype(np.uint8)


def snorm8_code(v):
    return np.rint(np.clip(np.asarray(v, dtype=np.float32), f32(-1), f32(1)) * f32(127)).astype(np.int8)


def decode(x, params):
    rw, rh, fw, fh, dw, dh = check_inputs(x)
    h = project(x["features"], params["W"], params["b"])
    a = pixel_values(h, rw, rh).astype(np.float32)
    A = sigmoid16(a[..., 0:3])
    B = softmax3(a[..., 3:6])
    C = coefficient_c(a[..., 6], a[..., 7], a[..., 8], float(np.asarray(params["S"])), float(np.asarray(params["T"])))
    p = previous_latent(x, rw, rh, dw, dh)
    L = latent_update(a, p)
    cA, cB, cL = unorm8_code(A), unorm8_code(B), snorm8_code(L)
    return {"A": (cA / 255.0).astype(np.float32), "B": (cB / 255.0).astype(np.float32),
            "C": C.astype(np.float32), "latent": (cL / 127.0).astype(np.float32),
            "codes_A": cA, "codes_B": cB, "codes_L": cL, "h": h, "p": p}


# ---- synthetic data (tests only; never the real model) -------------------------------------------------------------

def random_params(rng, w_scale=0.25):
    return {"W": q8(rng.normal(0.0, w_scale, size=(N_OUT, N_IN))).astype(np.float32),
            "b": f16(rng.normal(0.0, 0.5, size=N_OUT)).astype(np.float32),
            "S": np.array(rng.uniform(1.0, 4.0), dtype=np.float32),
            "T": np.array(rng.uniform(0.2, 1.5), dtype=np.float32)}


def random_inputs(rng, rw=40, rh=24, fw=None, fh=None, dw=None, dh=None, reset=False, motion_scale=3.0):
    fw = fw or -(-rw // 32) * 32
    fh = fh or -(-rh // 32) * 32
    dw = dw or (3 * rw + 1) // 2 + 3
    dh = dh or (3 * rh + 1) // 2 + 1
    feats = q8(rng.normal(0.0, 1.5, size=(fh // 2, fw // 2, N_IN)))
    motion = f16(rng.normal(0.0, motion_scale, size=(rh, rw, 2)))
    return {
        "features": feats.astype(np.float32),
        "motion": motion.astype(np.float32),
        "latent_prev": (rng.integers(-127, 128, size=(rh, rw, 4)) / 127.0).astype(np.float32),
        "jitter": np.array(rng.uniform(-0.5, 0.5, size=2), dtype=np.float32),
        "jitter_prev": np.array(rng.uniform(-0.5, 0.5, size=2), dtype=np.float32),
        "reset": bool(reset),
        "feature_size": (fw, fh),
        "recon_size": (dw, dh),
    }
