# SPDX-License-Identifier: MIT
"""NumPy reference of spec/06-reconstruction-and-output.md: output reconstruction (stage "post") and output filter
(stage "filter"). The clean team's oracle for those stages.

Arithmetic is float64 throughout (spec 06: "FP32 or wider"); only the stored outputs are rounded (FP16, 10-bit).

Min, max and clamp follow the IEEE maxNum/minNum rule: a NaN operand is ignored, so clamp(NaN, lo, hi) = lo. That
is the reading under which the spec's stated special values hold (E(NaN) = 0.36 log10(1e-4) from
max(NaN, 1e-4) = 1e-4; "NaN gives 0" for alpha; NaN -> code 0 for the reduced history; and the NaN guard of step 5
keeping the history finite for non-finite coefficients). See QUESTIONS.md, spec 06.

Sizes follow spec 01: R = (rw, rh), D = (dw, dh), O = (ow, oh). Arrays are [rows, columns, channels].
"""
import numpy as np

from . import motion_hi

E_SCALE = 0.36
E_FLOOR = 1e-4
CAP_IN = 32752.0
DEC_LO, DEC_HI = 1e-4, 3.0
REDUCED_SCALE = 2.45
DISPLAY_MAX = 2.45   # spec 06 draft 2: display image = clamp(E(O^), 0, 2.45)
SCORE_SLOPE, SCORE_BIAS = 1.1, 0.05
OMEGA_EPS = 1e-10
OMEGA_MIN = 1.0 / 255.0


def clamp(v, lo, hi):
    """IEEE minNum(maxNum(v, lo), hi): NaN -> lo."""
    return np.fmin(np.fmax(v, lo), hi)


def enc(v):
    """E(v) = 0.36 log10(max(1 + 100 v, 1e-4)); E(NaN) = 0.36 log10(1e-4) (maxNum ignores the NaN)."""
    with np.errstate(invalid="ignore", over="ignore"):
        return E_SCALE * np.log10(np.fmax(1.0 + 100.0 * np.asarray(v, np.float64), E_FLOOR))


def enc_in(v):
    """E_in(v) = E(min(v, 32752)), E_in(NaN) = 0 (spec 03 / 06)."""
    v = np.asarray(v, np.float64)
    return np.where(np.isnan(v), 0.0, enc(np.fmin(v, CAP_IN)))


def dec(q):
    """Dec(q) = (10^(q'/0.36) - 1) / 100, q' = clamp(q, 1e-4, 3)."""
    return (10.0 ** (clamp(np.asarray(q, np.float64), DEC_LO, DEC_HI) / E_SCALE) - 1.0) / 100.0


def exposure_or_one(e):
    e = float(np.asarray(e, np.float64).reshape(-1)[0])
    return 1.0 if e == 0.0 else e


def f16(x):
    return np.asarray(x, np.float64).astype(np.float16).astype(np.float32)


# ---- reconstruction ------------------------------------------------------------------------------------------------

def target_positions(rw, rh, dw, dh, jx, jy):
    """Spec 06 step 1: render-space position of every D column (px) and row (py), float64."""
    px = (np.arange(dw) + 0.5) * (rw / dw) + jx
    py = (np.arange(dh) + 0.5) * (rh / dh) + jy
    return px, py


def kernel_window(px, py):
    """Step 2: a_x per column, b per row (from the even row of each row pair)."""
    ax = np.floor(px - 0.5).astype(np.int64)
    even = np.arange(py.size) & ~1
    b = np.floor(py[even] - 0.5).astype(np.int64)
    return ax, b


def render_features(color, A, B, e):
    """Per render pixel: f = (E_in(e color) x3, sigma_near, sigma_center, sigma_far, g, beta_h, beta_o)."""
    with np.errstate(invalid="ignore", over="ignore"):
        ec = enc_in(e * np.asarray(color, np.float64))
    return np.concatenate([ec, np.asarray(B, np.float64), np.asarray(A, np.float64)], axis=2)


def gather(feat, C, px, py, rw, rh):
    """Step 2: W [dh, dw] and normalized features [dh, dw, 9]."""
    ax, b = kernel_window(px, py)
    C = np.asarray(C, np.float64)
    W = np.zeros((py.size, px.size))
    acc = np.zeros((py.size, px.size, feat.shape[2]))
    with np.errstate(invalid="ignore", over="ignore"):
        for j in (-1, 0, 1):
            sy = np.clip(b + j, 0, rh - 1)
            v = (b + j + 0.5 - py)[:, None]
            for i in (-1, 0, 1):
                sx = np.clip(ax + i, 0, rw - 1)
                u = (ax + i + 0.5 - px)[None, :]
                c = C[sy][:, sx]
                w = np.exp(-0.5 * (c[..., 0] * u * u + c[..., 1] * v * v + 2.0 * c[..., 2] * u * v))
                W += w
                acc += w[..., None] * feat[sy][:, sx]
        fbar = acc / W[..., None]
    return W, fbar


def point_index(px, py, rw, rh):
    """Render pixel (clamp(floor px), clamp(floor py)) of every D pixel, as (row index [dh, 1], col index [1, dw])."""
    fx = np.clip(np.floor(px), 0, rw - 1).astype(np.int64)
    fy = np.clip(np.floor(py), 0, rh - 1).astype(np.int64)
    return fy[:, None], fx[None, :]


# ---- motion candidate positions in float32 (spec 06 step 3 / preset L item 3: float32 unless stated) ----------------
# The validity test 0 <= q <= D is a discontinuity: a position computed in float64 can fall on the other side of an edge
# than the kernel's float32 one (a real-weight frame had qy = 108.0000089 in float64 and exactly 108 = Dh in float32,
# invalid against valid). So the render position, the point reads' texel, m_center's bilinear and q = X + m are
# computed in float32 in the kernel's order (clean/kernels/post.hip pos_x / pos_y, motion_bilinear, q = X + m).

# Each multiply and add is rounded to float32 on its own: measured against the kernel's outputs, a fused multiply-add
# in the bilinear keeps the real-weight edge case apart, and a fused position matches fewer reduced codes.
_F = np.float32


def target_positions_f32(rw, rh, dw, dh, jx, jy):
    """The render position of every D column (px) and row (py) in float32: (x + 1/2) * (R / D) + j, the scale rounded
    once from double as the runtime passes it (WgPostParams rx, ry)."""
    rx, ry = _F(float(rw) / dw), _F(float(rh) / dh)
    xs, ys = np.arange(dw, dtype=_F) + _F(0.5), np.arange(dh, dtype=_F) + _F(0.5)
    return xs * rx + _F(jx), ys * ry + _F(jy)


def bilinear_f32(img, px, py):
    """m_center's bilinear read as the kernel's motion_bilinear: float32 throughout, texel centres at i + 1/2,
    clamp-to-edge, top = a (1 - wx) + b wx, bottom likewise, then top (1 - wy) + bottom wy."""
    h, w = img.shape[:2]
    tx, ty = (np.asarray(px, _F) - _F(0.5)), (np.asarray(py, _F) - _F(0.5))
    x0, y0 = np.floor(tx), np.floor(ty)
    wx, wy = (tx - x0)[None, :, None], (ty - y0)[:, None, None]
    xi, yi = x0.astype(np.int64), y0.astype(np.int64)
    c0, c1 = np.clip(xi, 0, w - 1), np.clip(xi + 1, 0, w - 1)
    r0, r1 = np.clip(yi, 0, h - 1), np.clip(yi + 1, 0, h - 1)
    img = np.asarray(img, _F)
    owx, owy = _F(1) - wx, _F(1) - wy
    with np.errstate(invalid="ignore", over="ignore"):
        top = img[r0][:, c0] * owx + img[r0][:, c1] * wx
        bot = img[r1][:, c0] * owx + img[r1][:, c1] * wx
        return top * owy + bot * wy


def bilinear(img, px, py):
    """Bilinear sample of img [h, w, c] at render-space positions (texel centres at i + 1/2, clamp-to-edge)."""
    h, w = img.shape[:2]
    tx, ty = px - 0.5, py - 0.5
    x0, y0 = np.floor(tx), np.floor(ty)
    fx, fy = (tx - x0)[None, :, None], (ty - y0)[:, None, None]
    xi, yi = x0.astype(np.int64), y0.astype(np.int64)
    c0, c1 = np.clip(xi, 0, w - 1), np.clip(xi + 1, 0, w - 1)
    r0, r1 = np.clip(yi, 0, h - 1), np.clip(yi + 1, 0, h - 1)
    img = np.asarray(img, np.float64)
    top = img[r0][:, c0] * (1 - fx) + img[r0][:, c1] * fx
    bot = img[r1][:, c0] * (1 - fx) + img[r1][:, c1] * fx
    return top * (1 - fy) + bot * fy


def select_two(s1, s2, s3):
    """Step 3: indices (0, 1, 2) of the first and second candidate. Ties prefer the lower index; a NaN score never
    wins a comparison (so all-NaN scores give first = 0, second = 1)."""
    s = [s1, s2, s3]
    first = np.zeros(s1.shape, np.int64)
    best = s1.copy()
    for k in (1, 2):
        take = s[k] > best
        first = np.where(take, k, first)
        best = np.where(take, s[k], best)
    # the other two, in index order
    o1 = np.where(first == 0, 1, 0)
    o2 = np.where(first == 2, 1, 2)
    so = np.stack(s)
    v1 = np.take_along_axis(so, o1[None], 0)[0]
    v2 = np.take_along_axis(so, o2[None], 0)[0]
    second = np.where(v2 > v1, o2, o1)
    return first, second


def candidate_weights(q_first, q_second):
    """Step 3: omega1, omega2 from the clamped scores."""
    den = np.fmax(q_first + q_second, OMEGA_EPS)
    w2 = q_second / den
    small = w2 < OMEGA_MIN
    w2 = np.where(small, 0.0, w2)
    w1 = np.where(small, 1.0, q_first / den)
    return w1, w2


def catmull_rom_weights(f):
    """Cubic convolution weights (a = -1/2) for taps -1, 0, 1, 2 at fraction f; shape [..., 4]."""
    f2, f3 = f * f, f * f * f
    return np.stack([-0.5 * f3 + f2 - 0.5 * f, 1.5 * f3 - 2.5 * f2 + 1.0, -1.5 * f3 + 2.0 * f2 + 0.5 * f,
                     0.5 * f3 - 0.5 * f2], axis=-1)


def history_sample(hist, qx, qy):
    """Step 4, valid case: 12-tap Catmull-Rom of hist [dh, dw, 3] at D positions qx, qy (arrays of one shape),
    without the four corner taps, normalized by the sum of the twelve weights; texels clamp to the edge."""
    dh, dw = hist.shape[:2]
    tx, ty = qx - 0.5, qy - 0.5
    x0, y0 = np.floor(tx), np.floor(ty)
    wx, wy = catmull_rom_weights(tx - x0), catmull_rom_weights(ty - y0)
    xi, yi = x0.astype(np.int64), y0.astype(np.int64)
    hist = np.asarray(hist, np.float64)
    acc = np.zeros(qx.shape + (3,))
    wsum = np.zeros(qx.shape)
    with np.errstate(invalid="ignore", over="ignore"):
        for j in range(4):
            r = np.clip(yi - 1 + j, 0, dh - 1)
            for i in range(4):
                if i in (0, 3) and j in (0, 3):
                    continue
                c = np.clip(xi - 1 + i, 0, dw - 1)
                w = wx[..., i] * wy[..., j]
                acc += w[..., None] * hist[r, c]
                wsum += w
        return acc / wsum[..., None]


def reconstruct(x, s):
    """Stage "post" (spec 06 Reconstruction). x: the spec 90 input dict; s: the model scalar.

    Motion at output resolution (spec 06 section 3): x["motion_output_res"] true, x["motion"] the game's motion
    texture [rows][columns][2], x["motion_size"] = M, x["motion_origin"] (default (0, 0)), x["mv_scale"] (to pixels
    of M), x["mv_bias"] = b_mv; m_near / m_center / m_far are then not read. x["motion_texels"] = (tx [dw], ty [dh])
    overrides the texel rule (tests of the pixels where either neighbouring texel is acceptable).

    Returns {"history": [dh, dw, 3], "reduced": [dh//2, dw//2, 3], "display": [dh, dw, 3]} as float32 (history and
    display FP16-exact, display clamped to [0, 2.45] (spec 06 draft 2), reduced on the k/1023 grid), plus the unrounded intermediates under "_lin" for tests."""
    color = np.asarray(x["color"], np.float64)
    rh, rw = color.shape[:2]
    dw, dh = (int(v) for v in x["recon_size"])
    jx, jy = (float(v) for v in np.asarray(x["jitter"], np.float64).reshape(2))
    e, e_prev = exposure_or_one(x["exposure"]), exposure_or_one(x["exposure_prev"])
    reset = bool(x["reset"])
    hist = np.asarray(x["history"], np.float64)
    s = float(np.asarray(s, np.float64).reshape(-1)[0])

    px, py = target_positions(rw, rh, dw, dh, jx, jy)
    feat = render_features(color, x["A"], x["B"], e)
    W, fbar = gather(feat, x["C"], px, py, rw, rh)
    cbar, sig, gbar, bh, bo = fbar[..., 0:3], fbar[..., 3:6], fbar[..., 6], fbar[..., 7], fbar[..., 8]

    # 3. motion candidates (positions in float32: target_positions_f32)
    px32, py32 = target_positions_f32(rw, rh, dw, dh, jx, jy)
    fy, fx = point_index(px32, py32, rw, rh)
    hi = bool(x.get("motion_output_res", False))
    # x["nan_position_valid"]: spec 06 draft 4 section 4 for a NaN previous position: it counts as valid, its sample is
    # NaN, and step 5's NaN guard replaces both results by C_lin. Default: on for motion at output resolution (that
    # path was written from draft 4; kernels/post_motion_outres.h does the same), off otherwise (the reading the
    # render-resolution kernels of this tree implement: a NaN position is invalid and takes the fallback).
    nan_valid = bool(x.get("nan_position_valid", hi))
    first, second = select_two(sig[..., 0], sig[..., 1], sig[..., 2])
    sel = lambda arr, k: np.take_along_axis(arr, k[..., None], -1)[..., 0]  # noqa: E731
    with np.errstate(invalid="ignore"):
        q = clamp(SCORE_SLOPE * sig - SCORE_BIAS, 0.0, 1.0)
    w1, w2 = candidate_weights(sel(q, first), sel(q, second))

    # 4. history along each candidate
    X = np.arange(dw)[None, :] + 0.5
    Y = np.arange(dh)[:, None] + 0.5
    with np.errstate(invalid="ignore", over="ignore"):
        fallback = enc(e_prev * color[fy, fx])

    def along(qx, qy):
        """(history term [dh, dw, 3], mask of the pixels whose position is NaN and counts as valid)."""
        with np.errstate(invalid="ignore"):
            valid = (~np.bool_(reset)) & (qx >= 0) & (qx <= dw) & (qy >= 0) & (qy <= dh)
        nanq = (np.isnan(qx) | np.isnan(qy)) & np.bool_(nan_valid and not reset)
        hs = history_sample(hist, np.where(valid, qx, 0.5), np.where(valid, qy, 0.5)) if not reset else 0.0
        return np.where(valid[..., None], hs, fallback), nanq

    if hi:
        # Motion at output resolution (spec 06 section 3): the three motion surfaces are not read. One motion per D
        # pixel, read as a point from the game's motion texture; q in the FP32 arithmetic of motion_hi. Both kept
        # candidates have this position, so there is one sample S and H = w1 S + w2 S (w2 S counts as 0 when w2 = 0).
        qx, qy = motion_hi.recon_positions(x["motion"], x["motion_size"], (dw, dh), x["mv_scale"],
                                           x.get("mv_bias", (0.0, 0.0)), x.get("motion_origin", (0, 0)),
                                           x.get("motion_texels"))
        h1, nan_h = along(qx.astype(np.float64), qy.astype(np.float64))
        h2 = h1
    else:
        m = [np.asarray(x["m_near"], _F)[fy, fx], bilinear_f32(x["m_center"], px32, py32),
             np.asarray(x["m_far"], _F)[fy, fx]]
        X32, Y32 = X.astype(_F), Y.astype(_F)
        with np.errstate(invalid="ignore", over="ignore"):   # q = X + m in float32, then compared and sampled
            al = [along((X32 + mk[..., 0]).astype(np.float64), (Y32 + mk[..., 1]).astype(np.float64)) for mk in m]
        Hk = np.stack([a[0] for a in al], axis=-1)  # [dh, dw, 3, 3 candidates]
        nk = np.stack([a[1] for a in al], axis=-1)
        h1 = np.take_along_axis(Hk, first[..., None, None], -1)[..., 0]
        h2 = np.take_along_axis(Hk, second[..., None, None], -1)[..., 0]
        nan_h = sel(nk, first) | ((w2 != 0) & sel(nk, second))  # H is NaN
    with np.errstate(invalid="ignore", over="ignore"):
        H = np.where((w2 == 0)[..., None], w1[..., None] * h1, w1[..., None] * h1 + w2[..., None] * h2)

        # 5. blend
        C_lin = dec(cbar)
        H_lin = dec(H) / e_prev * e
        alpha = clamp(s * W * gbar, 0.0, 1.0)
        bh2 = (alpha * (1.0 - bh) + bh)[..., None]
        bo2 = (alpha * (1.0 - bo) + bo)[..., None]
        Hh = H_lin + (C_lin - H_lin) * bh2
        Oh = H_lin + (C_lin - H_lin) * bo2
        Hh = np.where((np.isnan(Hh.sum(-1)) | nan_h)[..., None], C_lin, Hh)
        Oh = np.where((np.isnan(Oh.sum(-1)) | nan_h)[..., None], C_lin, Oh)

        # 6. outputs
        hw, hh = dw // 2, dh // 2
        M = Hh[:2 * hh, :2 * hw].reshape(hh, 2, hw, 2, 3).sum(axis=(1, 3)) / 4.0
        code = np.rint(clamp(enc(M) / REDUCED_SCALE, 0.0, 1.0) * 1023.0)
    return {"history": f16(enc(Hh)), "reduced": (code / 1023.0).astype(np.float32),
            "display": f16(clamp(enc(Oh), 0.0, DISPLAY_MAX)),
            "_lin": {"H_hat": Hh, "O_hat": Oh, "W": W, "first": first, "second": second, "w1": w1, "w2": w2}}


# ---- output filter -------------------------------------------------------------------------------------------------

FILTER_STRETCH = 1.5   # spec 06 draft 3: Lanczos-3 stretched by the resampling ratio


def lanczos3(t):
    t = np.asarray(t, np.float64)
    return np.where(np.abs(t) < 3.0, np.sinc(t) * np.sinc(t / 3.0), 0.0)


def filter_weights(x):
    """Centre column c and the 5 normalized weights (taps c-2 ... c+2) of every output column x (spec 06 draft 3:
    Lanczos-3 stretched by 1.5, truncated to these taps, normalized)."""
    pos = (np.asarray(x, np.float64) + 0.5) * 1.5
    c = np.floor(pos).astype(np.int64)
    delta = pos - (c + 0.5)
    L = lanczos3((np.arange(-2, 3)[None, :] - delta[:, None]) / FILTER_STRETCH)
    return c, L / L.sum(axis=1, keepdims=True)


def recon_size_for(o):
    return (3 * int(o) + 1) // 2


def filter_pass(img, n_out):
    """One pass along axis 0 of img [n_in, ...] -> [n_out, ...]: 5-tap filter (filter_weights) with the local min/max
    clamp."""
    n_in = img.shape[0]
    c, w = filter_weights(np.arange(n_out))
    idx = np.clip(c[:, None] + np.arange(-2, 3)[None, :], 0, n_in - 1)   # [n_out, 5]
    taps = img[idx]                                                    # [n_out, 5, ...]
    wb = w.reshape(w.shape + (1,) * (img.ndim - 1))
    val = (taps * wb).sum(axis=1)
    return clamp(val, taps.min(axis=1), taps.max(axis=1))


def output_filter(display, exposure, output_size):
    """Stage "filter": display [dh, dw, 3] (E-encoded) -> linear colour [oh, ow, 3] (float32)."""
    ow, oh = (int(v) for v in output_size)
    img = np.asarray(display, np.float64)
    dh, dw = img.shape[:2]
    if (dw, dh) != (recon_size_for(ow), recon_size_for(oh)):
        raise ValueError(f"display is {dw}x{dh}, expected D = {recon_size_for(ow)}x{recon_size_for(oh)} for O = "
                         f"{ow}x{oh}")
    e = exposure_or_one(exposure)
    h = filter_pass(img.transpose(1, 0, 2), ow).transpose(1, 0, 2)    # horizontal: [dh, ow, 3]
    v = filter_pass(h, oh)                                             # vertical:   [oh, ow, 3]
    return (dec(v) / e).astype(np.float32)


# ---- output filter, "original" arithmetic (spec 06 draft 4) --------------------------------------------------------
#
# The weight set "original" of spec 06 draft 4: five weights w(-2) ... w(+2) for an even output index that come from
# the model file (spec 08, group filter / output_taps), mirrored for an odd index, and FP16 arithmetic:
#   "One pass turns five FP16 values t(-2) ... t(+2) into one: with w^_k = h(w_k), a <- h(fma(w^_k, t_k, a)) for
#    k = -2, -1, 0, 1, 2 starting from a = +0 (FP16, one rounding per step); result = min(max(a, min_k t_k), max_k t_k)."
# The functions above (filter_weights, filter_pass, output_filter) stay the set "analytic" in float64.
#
# The pass input here is the display image as FP16 values. Draft 4 also stores the display image as 11/11/10-bit codes
# and decodes them to FP16 first (its step 1); that is a separate change, and from step 2 on nothing depends on where
# the FP16 values came from.

def filter_center(x):
    """Centre input index of output index x: floor((x + 1/2) * 3/2) = 3 floor(x / 2) + 2 (x mod 2) (spec 06 step 3)."""
    x = np.asarray(x, np.int64)
    return 3 * (x // 2) + 2 * (x & 1)


def h_fma(w, t, a):
    """h(fma(w, t, a)) for FP16 values w, t, a: the exact w t + a rounded ONCE to FP16 (nearest, ties to even), as
    float64 arrays holding FP16 values.

    w t is exact in float64 (two 11-bit significands). The sum is rounded to odd in float64 (the TwoSum error tells
    whether it was exact and on which side the exact value lies), so that the final rounding to FP16 is the single
    rounding of the exact value: 53 bits are more than FP16's 11 + 2. A plain float32 fma followed by a rounding to
    FP16 is NOT this operation: it rounds twice (clean/tests/test_output_filter_original.py shows a case).
    The sign of a zero result is the fma's (+0 unless product and addend are both -0, or the value underflows from
    below)."""
    w, t, a = (np.asarray(v, np.float64) for v in (w, t, a))
    with np.errstate(over="ignore", invalid="ignore"):
        p = w * t
        s = p + a
        bb = s - p
        err = (p - (s - bb)) + (a - bb)                       # exact: s + err = p + a (finite operands)
        odd = (np.asarray(s).view(np.uint64) & np.uint64(1)) != 0   # last significand bit of the float64 sum
        fix = np.isfinite(s) & (err != 0.0) & ~odd
        s = np.where(fix, np.nextafter(s, np.where(err > 0.0, np.inf, -np.inf)), s)
        return s.astype(np.float16).astype(np.float64)


def filter_taps_f16(taps):
    """w^_k = h(w_k): the five weights (even output index, w(-2) first) rounded to FP16 once."""
    w = np.asarray(taps, np.float64).reshape(-1)
    if w.shape != (5,) or not np.isfinite(w).all():
        raise ValueError("the output filter has five finite weights")
    return f16(w).astype(np.float64)


def filter_pass_original(img, n_out, taps):
    """One pass (spec 06 draft 4 step 2) along axis 0 of img [n_in, ...] -> [n_out, ...], FP16 values in and out.
    Output index x: taps t_k = img[clamp(c + k, 0, n_in - 1)] with c = filter_center(x), weights w^_k for an even x
    and w^_(-k) for an odd x; a <- h(fma(w, t_k, a)) for k = -2 ... 2 from a = +0; result min(max(a, min t), max t)."""
    v = f16(img).astype(np.float64)
    w16 = filter_taps_f16(taps)
    n_in = v.shape[0]
    x = np.arange(int(n_out))
    idx = np.clip(filter_center(x)[:, None] + np.arange(-2, 3)[None, :], 0, n_in - 1)   # [n_out, 5]
    t = v[idx]                                                                           # [n_out, 5, ...]
    w = np.where((x & 1)[:, None] == 0, w16[None, :], w16[None, ::-1])                   # [n_out, 5]
    w = w.reshape(w.shape + (1,) * (v.ndim - 1))
    a = np.zeros(t.shape[:1] + t.shape[2:])                                              # +0
    for k in range(5):
        a = h_fma(w[:, k], t[:, k], a)
    return np.minimum(np.maximum(a, t.min(axis=1)), t.max(axis=1))


def encoded_filter_original(display, output_size, taps):
    """Spec 06 draft 4 steps 3 and 4: horizontal pass on every row, then the vertical pass on its results (whose clamp
    uses the five horizontally filtered values). display [dh, dw, 3] (FP16 values) -> [oh, ow, 3] float16."""
    ow, oh = (int(v) for v in output_size)
    img = np.asarray(display)
    dh, dw = img.shape[:2]
    if (dw, dh) != (recon_size_for(ow), recon_size_for(oh)):
        raise ValueError(f"display is {dw}x{dh}, expected D = {recon_size_for(ow)}x{recon_size_for(oh)} for O = "
                         f"{ow}x{oh}")
    h = filter_pass_original(img.transpose(1, 0, 2), ow, taps).transpose(1, 0, 2)    # [dh, ow, 3]
    return filter_pass_original(h, oh, taps).astype(np.float16)                      # [oh, ow, 3]


def output_filter_original(display, exposure, output_size, taps):
    """Stage "filter" with the weight set "original" (spec 06 draft 4 steps 2 to 5): display [dh, dw, 3] (E-encoded
    FP16 values), the five weights of the model file -> linear colour [oh, ow, 3] (float32) = Dec(value) / e."""
    q = encoded_filter_original(display, output_size, taps).astype(np.float64)
    return (dec(q) / exposure_or_one(exposure)).astype(np.float32)


# ---- synthetic data (tests only) -----------------------------------------------------------------------------------

def random_inputs(rng, rw, rh, dw=None, dh=None, reset=False, motion_scale=3.0):
    """Synthetic spec 90 "post" inputs: values on the grids the real producers use (A, B on k/255, C, motions and
    history FP16). C is a positive-definite (a, b, c) built like spec 05 (sx^2, sy^2, sx sy rho)."""
    dw = dw or recon_size_for(int(rw * 1.5))   # Quality-like ratio: O = 1.5 R, D = (3 O + 1) / 2
    dh = dh or recon_size_for(int(rh * 1.5))
    color = np.exp(rng.normal(-1.5, 1.5, (rh, rw, 3))).astype(np.float32)
    A = rng.integers(0, 256, (rh, rw, 3)) / 255.0
    Bl = rng.normal(0, 1.5, (rh, rw, 3))
    B = np.rint(np.exp(Bl) / np.exp(Bl).sum(-1, keepdims=True) * 255.0) / 255.0
    sx, sy = rng.uniform(0.2, 2.0, (rh, rw)), rng.uniform(0.2, 2.0, (rh, rw))
    rho = rng.uniform(-0.9, 0.9, (rh, rw))
    C = f16(np.stack([sx * sx, sy * sy, sx * sy * rho], -1))
    hist = f16(enc(np.exp(rng.normal(-1.5, 1.5, (dh, dw, 3)))))
    base = rng.normal(0, motion_scale, 2)
    mot = lambda: f16(base + rng.normal(0, motion_scale * 0.3, (rh, rw, 2)))  # noqa: E731
    return {"color": color, "exposure": float(rng.uniform(0.5, 2.0)), "exposure_prev": float(rng.uniform(0.5, 2.0)),
            "A": A.astype(np.float32), "B": B.astype(np.float32), "C": C, "history": hist,
            "m_near": mot(), "m_center": mot(), "m_far": mot(), "jitter": rng.uniform(-0.5, 0.5, 2),
            "reset": bool(reset), "recon_size": (dw, dh)}
