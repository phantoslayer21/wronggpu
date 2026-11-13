# SPDX-License-Identifier: MIT
"""NumPy reference of "Motion at output resolution" (spec/03-input-encoder.md draft 4, spec/06-reconstruction-and-
output.md section 3): the game's motion texture has the output size M and is read as points, one texel per render
pixel in the input encoder and one per reconstruction (D) pixel in the reconstruction.

Everything here is the FP32 arithmetic the two pages fix (one rounding per operation, in the order written), so the
texel indices, the motion surface and the previous positions are exact, not approximations:

    s   = RN32(mv_scale / RN32(M / D))                                              (both pages)
    enc T = clamp(floor(RN32(RN32((x + 0.5) - jx) * RN32(M / R))), 0, M - 1)        (spec 03)
        m = RN32(s * RN32(b_mv + MV(origin + T)));  the stored surface is RN16(m)
    rec T = clamp(trunc(RN32(RN32((x + 0.5) / dw) * Mw)), 0, Mw - 1)                (spec 06)
        q = fma(s, RN32(b_mv + MV(origin + T)), x + 0.5)

A NaN or out-of-range texel coordinate cannot occur in the reconstruction rule (x, dw, Mw are sizes); in the encoder
rule a NaN coordinate (NaN jitter) gives texel 0 and +-inf clamp like any other value (our choice: the page does not
say; the kernel does the same).
"""
import numpy as np

F32 = np.float32


def rn32(v):
    return np.asarray(v, dtype=np.float64).astype(np.float32)


def fma32(a, b, c):
    """fma(a, b, c) for float32 arrays: the exact a b + c rounded ONCE to float32.

    a b is exact in float64 (two 24-bit significands). The float64 sum is rounded to odd (the TwoSum error says whether
    it was exact and on which side the exact value lies), so that the final rounding to float32 is the single rounding
    of the exact value (53 bits are more than 24 + 2). Non-finite operands follow the plain float64 result."""
    a, b, c = (np.asarray(v, dtype=np.float32).astype(np.float64) for v in (a, b, c))
    with np.errstate(over="ignore", invalid="ignore"):
        p = a * b
        s = p + c
        bb = s - p
        err = (p - (s - bb)) + (c - bb)                       # exact: s + err = p + c (finite operands)
        odd = (np.asarray(s).view(np.uint64) & np.uint64(1)) != 0
        fix = np.isfinite(s) & (err != 0.0) & ~odd
        s = np.where(fix, np.nextafter(s, np.where(err > 0.0, np.inf, -np.inf)), s)
        return s.astype(np.float32)


def scale(mv_scale, M, D):
    """s = RN32(mv_scale / RN32(M / D)) per axis; mv_scale converts the game's motion to pixels of the motion texture."""
    k = (np.asarray(M, np.float32) / np.asarray(D, np.float32)).astype(np.float32)
    with np.errstate(divide="ignore", invalid="ignore"):
        return (np.asarray(mv_scale, np.float32) / k).astype(np.float32)


def encoder_texels(n_render, n_motion, jitter):
    """Motion texel index of every render pixel 0 ... n_render - 1 along one axis (spec 03)."""
    k = F32(n_motion) / F32(n_render)
    c = (np.arange(n_render, dtype=np.float32) + F32(0.5)) - F32(jitter)
    with np.errstate(invalid="ignore", over="ignore"):
        t = np.floor(c * k)
    t = np.where(np.isnan(t), 0.0, t)
    return np.clip(t, 0, n_motion - 1).astype(np.int64)


def recon_texels(n_recon, n_motion):
    """Motion texel index of every D pixel 0 ... n_recon - 1 along one axis (spec 06 section 3)."""
    c = (np.arange(n_recon, dtype=np.float32) + F32(0.5)) / F32(n_recon)
    t = np.trunc((c * F32(n_motion)).astype(np.float32))
    return np.clip(t, 0, n_motion - 1).astype(np.int64)


def recon_boundary(n_recon, n_motion):
    """True where the D pixel's centre lies exactly between two motion texels: (x + 1/2) M / D is a whole number
    (spec 06: either neighbouring texel is an equally faithful result there). Exact integer test."""
    x = np.arange(n_recon, dtype=np.int64)
    return ((2 * x + 1) * int(n_motion)) % (2 * int(n_recon)) == 0


def recon_texel_alternatives(n_recon, n_motion):
    """(lo, hi) per D pixel: the texels a faithful implementation may read (equal except at recon_boundary, where
    they are the two neighbours k - 1 and k of the whole number k = (x + 1/2) M / D, clamped)."""
    x = np.arange(n_recon, dtype=np.int64)
    k = ((2 * x + 1) * int(n_motion)) // (2 * int(n_recon))    # floor of the exact position
    hi = np.clip(k, 0, n_motion - 1)
    lo = np.where(recon_boundary(n_recon, n_motion), np.clip(k - 1, 0, n_motion - 1), hi)
    return lo, hi


def _read(texture, origin, ty, tx):
    """Point read of the game's motion texture [rows][columns][2] at origin + (tx, ty) (float32)."""
    tex = np.asarray(texture, dtype=np.float32)
    ox, oy = (int(v) for v in origin)
    return tex[np.clip(oy + ty, 0, tex.shape[0] - 1)[:, None], np.clip(ox + tx, 0, tex.shape[1] - 1)[None, :]]


def encoder_motion(texture, R, M, D, jitter, mv_scale, b_mv=(0.0, 0.0), origin=(0, 0)):
    """Spec 03 "Motion at output resolution": m of every render pixel, [rh][rw][2] float32 (D pixels; m_near =
    m_center = m_far = m; the stored surfaces are its FP16 rounding)."""
    (rw, rh), (mw, mh) = R, M
    tx = encoder_texels(rw, mw, jitter[0])
    ty = encoder_texels(rh, mh, jitter[1])
    mv = _read(texture, origin, ty, tx)
    s = scale(mv_scale, M, D)
    with np.errstate(invalid="ignore", over="ignore"):
        v = (np.asarray(b_mv, np.float32) + mv).astype(np.float32)
        return (s * v).astype(np.float32)


def recon_positions(texture, M, D, mv_scale, b_mv=(0.0, 0.0), origin=(0, 0), texels=None):
    """Spec 06 section 3, motion at output resolution: the previous position q of every D pixel, (qx, qy) [dh][dw]
    float32. texels = (tx [dw], ty [dh]) overrides the texel rule (tests of the either-texel pixels)."""
    (mw, mh), (dw, dh) = M, D
    tx, ty = texels if texels is not None else (recon_texels(dw, mw), recon_texels(dh, mh))
    mv = _read(texture, origin, np.asarray(ty), np.asarray(tx))
    s = scale(mv_scale, M, D)
    with np.errstate(invalid="ignore", over="ignore"):
        v = (np.asarray(b_mv, np.float32) + mv).astype(np.float32)
    X = np.broadcast_to((np.arange(dw, dtype=np.float32) + F32(0.5))[None, :], (dh, dw))
    Y = np.broadcast_to((np.arange(dh, dtype=np.float32) + F32(0.5))[:, None], (dh, dw))
    return fma32(s[0], v[..., 0], X), fma32(s[1], v[..., 1], Y)
