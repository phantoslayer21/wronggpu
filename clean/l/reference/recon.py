# SPDX-License-Identifier: MIT
"""Model L's output reconstruction (stage "post"), NumPy float64 oracle (issue #89).

Model L's reconstruction is Model M's (clean/reference/post.py reconstruct, spec 06) with three changes, selected by
the structure table's "recon" entry (clean/l/reference/table.py):
1. the colour gather uses a window x window taps around the target (Model L: 4 x 4, offsets -1 ... 2; Model M: 3 x 3);
2. each history candidate is decoded and has the previous exposure removed before the two are combined
   ("history_combine": "linear"): L_i = D(H_i) / e_prev, H_lin = (w1 L1 + w2 L2) e (L2 only when w2 > 0);
3. a sample that is invalid (outside D, or reset) falls back to the render pixel's linear colour as it is
   ("fallback": "linear_colour"), with no exposure factor and no encode / decode.
The learned scale s is the post stage's tensor "s". Everything else (positions, the score and candidate rules, the
12-tap Catmull-Rom, the blend, the NaN guard, the stored outputs) is Model M's, by import.

The colour gather ("gather": "half", Model L) runs in the model's own precision as the spec side confirmed it (target
positions, FP16 offsets and exponent with the fused cross term, float32 weights flushed to zero below 2^-126, FP16
weights and half accumulators, float32 weight sum: gather_half). Everything after it is float64 with only the stored
outputs rounded, as Model M's oracle; the remaining kernel-order details (the hardware exp2, reciprocal and
divisions, the Catmull-Rom weight forms) are a matter of the kernel-order model.
ldr=True gives the LDR form: identity transfer, no exposure (e = e_prev = 1), codes from the values directly; the
display is the 11 / 11 / 10-bit codes of the plain colour (ldr_display_codes), which the output stage unpacks with
code RN32(1/N) and filters without a decode or an exposure (output_filter). Both confirmed by the spec side.

Zero total gather weight (every tap underflows) has no special path, as confirmed by the spec side for Model L and
Model M: the normalized features are 0 / 0 = NaN, both scores NaN (candidate 1 kept, w1 = 1, w2 = 0), C = D(NaN) =
NaN (the decode passes NaN through: dec), alpha = 0, and the NaN guard substitutes C, which is NaN. The stored history
is then h(E(NaN)) = -1.4404296875 (HDR) or NaN (LDR), the display and the reduced code 0.

The gather's tap order is i (columns) outer, j (rows) inner; the other sums of this oracle are float64 (the tolerance
reference).

The reconstruction variant (3/2 or 3/1) only decides how a kernel tiles the work; both forms compute this function
(table.recon_variant).
"""
import numpy as np

from clean.reference import motion_hi as _motion_hi
from clean.reference import post as _post

from . import table as _table


def dec(q):
    """The decode D(.) with a NaN passed through. The clamp to [1e-4, 3] applies to numbers only (confirmed by the spec
    side for Model L and Model M: a zero gather weight gives C = D(NaN) = NaN). clean/reference/post.py dec maps NaN to
    D(1e-4) instead."""
    q = np.asarray(q, np.float64)
    with np.errstate(invalid="ignore", over="ignore"):
        return np.where(np.isnan(q), np.nan, _post.dec(q))


def _transfer(ldr):
    """(encode, decode, input encode) for HDR or LDR."""
    if ldr:
        ident = lambda v: np.asarray(v, np.float64)  # noqa: E731
        return ident, ident, ident
    return _post.enc, dec, _post.enc_in


# LDR display codes (11 / 11 / 10 bits) and the output stage's unpack constants RN32(1/2047), RN32(1/1023).
LDR_CODE_MAX = np.array([2047.0, 2047.0, 1023.0])
LDR_UNPACK = np.array([np.float32(1.0 / 2047.0), np.float32(1.0 / 2047.0), np.float32(1.0 / 1023.0)], np.float64)


def ldr_display_codes(O):
    """LDR packed display codes of the plain display colour O^ [.., 3]: floor(clamp(O^ N + 1/2, 0, N)), N = 2047, 2047,
    1023; NaN gives 0."""
    with np.errstate(invalid="ignore", over="ignore"):
        return np.floor(_post.clamp(np.asarray(O, np.float64) * LDR_CODE_MAX + 0.5, 0.0, LDR_CODE_MAX))


def ldr_unpack(codes):
    """The output stage's LDR unpack: code RN32(1/N), one single-precision product, rounded to FP16 (no decode)."""
    v = (np.asarray(codes, np.float64) * LDR_UNPACK).astype(np.float32)
    return v.astype(np.float16).astype(np.float32)


def output_filter(display, exposure, output_size, ldr=False):
    """Stage "filter" for Model L: Model M's output filter (clean/reference/post.py, the project's own taps) on the
    reconstruction's "display". HDR: E-encoded display, decode and divide by the current exposure, as Model M. LDR:
    display holds the unpacked codes (ldr_unpack), filtered as they are: no decode, no exposure read."""
    if not ldr:
        return _post.output_filter(display, exposure, output_size)
    ow, oh = (int(v) for v in output_size)
    img = np.asarray(display, np.float64)
    dh, dw = img.shape[:2]
    if (dw, dh) != _table.recon_size(ow, oh):
        raise ValueError(f"display is {dw}x{dh}, expected D = {_table.recon_size(ow, oh)} for O = {ow}x{oh}")
    h = _post.filter_pass(img.transpose(1, 0, 2), ow).transpose(1, 0, 2)
    return _post.filter_pass(h, oh).astype(np.float32)


def gather(feat, C, px, py, rw, rh, window):
    """Step 2 with a window x window tap window: columns ax - 1 ... ax + window - 2, rows b - 1 ... b + window - 2
    (b from the even row of each row pair, shared by both rows). Tap order: i outer, j inner. Returns W [dh, dw]
    and the normalized features [dh, dw, 9]."""
    ax, b = _post.kernel_window(px, py)
    C = np.asarray(C, np.float64)
    W = np.zeros((py.size, px.size))
    acc = np.zeros((py.size, px.size, feat.shape[2]))
    offs = range(-1, window - 1)
    with np.errstate(invalid="ignore", over="ignore"):
        for i in offs:
            sx = np.clip(ax + i, 0, rw - 1)
            u = (ax + i + 0.5 - px)[None, :]
            for j in offs:
                sy = np.clip(b + j, 0, rh - 1)
                v = (b + j + 0.5 - py)[:, None]
                c = C[sy][:, sx]
                w = np.exp(-0.5 * (c[..., 0] * u * u + c[..., 1] * v * v + 2.0 * c[..., 2] * u * v))
                W += w
                acc += w[..., None] * feat[sy][:, sx]
        fbar = acc / W[..., None]
    return W, fbar


def _h(v):
    """Round to FP16 (from float64, one rounding)."""
    with np.errstate(over="ignore", invalid="ignore"):
        return np.asarray(v, np.float64).astype(np.float16).astype(np.float64)


def _f32(v):
    with np.errstate(over="ignore", invalid="ignore"):
        return np.asarray(v, np.float64).astype(np.float32).astype(np.float64)


def target_positions_f32(rw, rh, dw, dh, jx, jy):
    """Render-space positions in the model's precision: rx = float32(rw) / float32(dw), px = fused(x + 1/2, rx, jx)
    (one float32 rounding), py = fused(ry, y + 1/2, jy)."""
    rx, ry = _f32(np.float32(rw) / np.float32(dw)), _f32(np.float32(rh) / np.float32(dh))
    px = _f32((np.arange(dw) + 0.5) * rx + _f32(jx))
    py = _f32((np.arange(dh) + 0.5) * ry + _f32(jy))
    return px, py


def gather_half(feat, C, px, py, rw, rh, window):
    """Step 2 in the model's precision (spec side, confirmed for Model L and Model M):
    - per render pixel: A = h(f32(a -0.721325)), B = h(f32(b -0.721325)), C = h(f32(c -1.44265)); the nine features
      in FP16 (feat must already be FP16 values);
    - offsets: dx = h(f32((ax + 1/2) - px)); d_y = h(f32((b_y + 1/2) - py)), minus 1 (half) when the row's own
      b_y = floor(py - 1/2) is not the pair's b; u = h(dx + i), v = h(d_y + j);
    - exponent: eps = fused_half(h(u v), C, h(h(h(u u) A) + h(h(v v) B))) (C1);
    - w = 2^eps in float32, flushed to +0 below 2^-126; W = float32 sum of the unrounded w in tap order;
    - F_k = fused_half(h(w), f_k, F_k), from +0; r = 1 / W, f-bar_k = float32(r F_k).
    Tap order: i (columns) outer, j (rows) inner. Fused half multiply-adds are evaluated in float64 and rounded once
    to FP16 (exact unless the addends' exponents differ by more than about 30, far outside these values)."""
    ax, b = _post.kernel_window(px, py)
    Cc = np.asarray(C, np.float64)
    kA = _h(_f32(Cc[..., 0] * _f32(-0.721325)))
    kB = _h(_f32(Cc[..., 1] * _f32(-0.721325)))
    kC = _h(_f32(Cc[..., 2] * _f32(-1.44265)))
    dh, dw = py.size, px.size
    by = np.floor(py - 0.5).astype(np.int64)
    dx = _h(_f32((ax + 0.5) - px))[None, :]
    dyv = _h(_f32((by + 0.5) - py))
    dyv = np.where(by != b, _h(dyv - 1.0), dyv)[:, None]
    W = np.zeros((dh, dw))
    F = np.zeros((dh, dw, feat.shape[2]))
    offs = range(-1, window - 1)
    with np.errstate(invalid="ignore", over="ignore", under="ignore"):
        for i in offs:
            sx = np.clip(ax + i, 0, rw - 1)
            u = _h(dx + i)
            for j in offs:
                sy = np.clip(b + j, 0, rh - 1)
                v = _h(dyv + j)
                a_, b_, c_ = kA[sy][:, sx], kB[sy][:, sx], kC[sy][:, sx]
                s_ = _h(_h(_h(u * u) * a_) + _h(_h(v * v) * b_))
                eps = _h(_h(u * v) * c_ + s_)
                w = _f32(np.exp2(eps))
                w = np.where(w < 2.0 ** -126, 0.0, w)
                W = _f32(W + w)
                hw = _h(w)[..., None]
                F = _h(hw * feat[sy][:, sx] + F)
        r = _f32(1.0 / W)
        fbar = _f32(r[..., None] * F)
    return W, fbar


def reconstruct(x, s, t=_table.TABLE, ldr=False):
    """Stage "post" for Model L. x: Model M's reconstruction input dict (clean/reference/post.py reconstruct); s: the
    post stage's learned scale. Returns Model M's output dict ("history", "reduced", "display", "_lin")."""
    rc = t["recon"]
    window = int(rc["window"])
    linear = rc["history_combine"] == "linear"
    lin_fallback = rc["fallback"] == "linear_colour"
    f32 = rc.get("blend", "float64") == "float32"
    r32 = _f32 if f32 else (lambda v: v)
    fma = lambda a, b, c: np.asarray(_motion_hi.fma32(a, b, c), np.float64)  # noqa: E731 (float32, one rounding)
    enc, dec, enc_in = _transfer(ldr)

    color = np.asarray(x["color"], np.float64)
    rh, rw = color.shape[:2]
    dw, dh = (int(v) for v in x["recon_size"])
    jx, jy = (float(v) for v in np.asarray(x["jitter"], np.float64).reshape(2))
    e, e_prev = (1.0, 1.0) if ldr else (_post.exposure_or_one(x["exposure"]), _post.exposure_or_one(x["exposure_prev"]))
    reset = bool(x["reset"])
    hist = np.asarray(x["history"], np.float64)
    if not reset and hist.shape[:2] != (dh, dw):
        raise ValueError(f"history is {hist.shape[1]}x{hist.shape[0]}, expected D = {dw}x{dh} (a size change needs "
                         "reset)")
    s = float(np.asarray(s, np.float64).reshape(-1)[0])

    with np.errstate(invalid="ignore", over="ignore"):
        ec = enc_in(e * color)
    feat = np.concatenate([ec, np.asarray(x["B"], np.float64), np.asarray(x["A"], np.float64)], axis=2)
    if rc.get("gather", "float64") == "half":
        px, py = target_positions_f32(rw, rh, dw, dh, jx, jy)
        W, fbar = gather_half(_h(feat), x["C"], px, py, rw, rh, window)
    else:
        px, py = _post.target_positions(rw, rh, dw, dh, jx, jy)
        W, fbar = gather(feat, x["C"], px, py, rw, rh, window)
    cbar, sig, gbar, bh, bo = fbar[..., 0:3], fbar[..., 3:6], fbar[..., 6], fbar[..., 7], fbar[..., 8]

    # 3. motion candidates (Model M's rules)
    fy, fx = _post.point_index(px, py, rw, rh)
    hi = bool(x.get("motion_output_res", False))
    nan_valid = bool(x.get("nan_position_valid", hi))
    first, second = _post.select_two(sig[..., 0], sig[..., 1], sig[..., 2])
    sel = lambda arr, k: np.take_along_axis(arr, k[..., None], -1)[..., 0]  # noqa: E731
    with np.errstate(invalid="ignore"):
        q = _post.clamp(_post.SCORE_SLOPE * sig - _post.SCORE_BIAS, 0.0, 1.0)
    w1, w2 = _post.candidate_weights(sel(q, first), sel(q, second))

    # 4. history along each candidate
    X = np.arange(dw)[None, :] + 0.5
    Y = np.arange(dh)[:, None] + 0.5
    with np.errstate(invalid="ignore", over="ignore"):
        if lin_fallback:
            fallback = color[fy, fx]                                   # linear colour, no exposure
        else:
            fallback = enc(e_prev * color[fy, fx])                     # Model M: encoded domain

    def along(qx, qy):
        """(candidate [dh, dw, 3] in the combination's domain, mask of NaN positions that count as valid)."""
        with np.errstate(invalid="ignore"):
            valid = (~np.bool_(reset)) & (qx >= 0) & (qx <= dw) & (qy >= 0) & (qy <= dh)
        nanq = (np.isnan(qx) | np.isnan(qy)) & np.bool_(nan_valid and not reset)
        hs = _post.history_sample(hist, np.where(valid, qx, 0.5), np.where(valid, qy, 0.5)) if not reset else 0.0
        if linear:
            with np.errstate(invalid="ignore", over="ignore"):
                hs = r32(r32(dec(hs)) / e_prev)
        return np.where(valid[..., None], hs, fallback), nanq

    if hi:
        qx, qy = _motion_hi.recon_positions(x["motion"], x["motion_size"], (dw, dh), x["mv_scale"],
                                            x.get("mv_bias", (0.0, 0.0)), x.get("motion_origin", (0, 0)),
                                            x.get("motion_texels"))
        h1, nan_h = along(qx.astype(np.float64), qy.astype(np.float64))
        h2 = h1
    else:
        m = [np.asarray(x["m_near"], np.float64)[fy, fx], _post.bilinear(x["m_center"], px, py),
             np.asarray(x["m_far"], np.float64)[fy, fx]]
        with np.errstate(invalid="ignore"):
            al = [along(X + mk[..., 0], Y + mk[..., 1]) for mk in m]
        Hk = np.stack([a[0] for a in al], axis=-1)
        nk = np.stack([a[1] for a in al], axis=-1)
        h1 = np.take_along_axis(Hk, first[..., None, None], -1)[..., 0]
        h2 = np.take_along_axis(Hk, second[..., None, None], -1)[..., 0]
        nan_h = sel(nk, first) | ((w2 != 0) & sel(nk, second))
    with np.errstate(invalid="ignore", over="ignore"):
        if f32:
            # float32, one rounding per operation: Hl = f32(fused(w1 L1 + f32(w2 L2)) e) (w1 L1 alone when w2 = 0).
            w1, w2 = r32(w1), r32(w2)
            H = np.where((w2 == 0)[..., None], r32(w1[..., None] * h1),
                         fma(w1[..., None], h1, r32(w2[..., None] * h2)))
        else:
            H = np.where((w2 == 0)[..., None], w1[..., None] * h1, w1[..., None] * h1 + w2[..., None] * h2)

        # 5. blend
        if f32:
            # alpha = sat(f32(f32(s W) g-bar)), beta' = fused(alpha f32(1 - beta-bar) + beta-bar),
            # H^ = fused(f32(C - Hl) beta' + Hl), O^ likewise.
            C_lin = r32(dec(cbar))
            H_lin = r32(H * e)
            alpha = _post.clamp(r32(r32(np.float32(s) * W) * gbar), 0.0, 1.0)
            bh2 = fma(alpha, r32(1.0 - bh), bh)[..., None]
            bo2 = fma(alpha, r32(1.0 - bo), bo)[..., None]
            Hh = fma(r32(C_lin - H_lin), bh2, H_lin)
            Oh = fma(r32(C_lin - H_lin), bo2, H_lin)
        else:
            C_lin = dec(cbar)
            H_lin = H * e if linear else dec(H) / e_prev * e
            alpha = _post.clamp(s * W * gbar, 0.0, 1.0)
            bh2 = (alpha * (1.0 - bh) + bh)[..., None]
            bo2 = (alpha * (1.0 - bo) + bo)[..., None]
            Hh = H_lin + (C_lin - H_lin) * bh2
            Oh = H_lin + (C_lin - H_lin) * bo2
        Hh = np.where((np.isnan(Hh.sum(-1)) | nan_h)[..., None], C_lin, Hh)
        Oh = np.where((np.isnan(Oh.sum(-1)) | nan_h)[..., None], C_lin, Oh)

        # 6. outputs (Model M's)
        hw, hh = dw // 2, dh // 2
        M = Hh[:2 * hh, :2 * hw].reshape(hh, 2, hw, 2, 3).sum(axis=(1, 3)) / 4.0
        # Reduced codes: HDR k = E(.) / 2.45, LDR k = the value. "display": HDR the E value clamped to [0, 2.45] (Model
        # M's convention); LDR the unpacked 11/11/10-bit codes the output stage reads ("display_codes": the codes).
        if ldr:
            code = np.rint(_post.clamp(M, 0.0, 1.0) * 1023.0)
        else:
            code = np.rint(_post.clamp(enc(M) / _post.REDUCED_SCALE, 0.0, 1.0) * 1023.0)
        if ldr:
            codes = ldr_display_codes(Oh)
            disp = ldr_unpack(codes)
        else:
            codes = None
            disp = _post.clamp(enc(Oh), 0.0, _post.DISPLAY_MAX)
    return {"history": _post.f16(enc(Hh)), "reduced": (code / 1023.0).astype(np.float32),
            "display": _post.f16(disp),
            "_lin": {"H_hat": Hh, "O_hat": Oh, "W": W, "first": first, "second": second, "w1": w1, "w2": w2,
                     "C_lin": C_lin, "H_lin": H_lin, "beta_h": bh2[..., 0], "beta_o": bo2[..., 0]},
            **({"display_codes": codes.astype(np.uint16)} if codes is not None else {})}
