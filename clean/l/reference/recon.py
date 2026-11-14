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

Arithmetic is float64 with only the stored outputs rounded, as Model M's oracle: the reconstruction's half-precision
and fused-rounding forms (e.g. the gather exponent) are a matter of the kernel-order model, not of this oracle.
ldr=True gives the LDR form: identity transfer, no exposure (e = e_prev = 1), codes from the values directly.

The reconstruction variant (3/2 or 3/1) only decides how a kernel tiles the work; both forms compute this function
(table.recon_variant).
"""
import numpy as np

from clean.reference import motion_hi as _motion_hi
from clean.reference import post as _post

from . import table as _table


def _transfer(ldr):
    """(encode, decode, input encode) for HDR or LDR."""
    if ldr:
        ident = lambda v: np.asarray(v, np.float64)  # noqa: E731
        return ident, ident, ident
    return _post.enc, _post.dec, _post.enc_in


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


def reconstruct(x, s, t=_table.TABLE, ldr=False):
    """Stage "post" for Model L. x: Model M's reconstruction input dict (clean/reference/post.py reconstruct); s: the
    post stage's learned scale. Returns Model M's output dict ("history", "reduced", "display", "_lin")."""
    rc = t["recon"]
    window = int(rc["window"])
    linear = rc["history_combine"] == "linear"
    lin_fallback = rc["fallback"] == "linear_colour"
    enc, dec, enc_in = _transfer(ldr)

    color = np.asarray(x["color"], np.float64)
    rh, rw = color.shape[:2]
    dw, dh = (int(v) for v in x["recon_size"])
    jx, jy = (float(v) for v in np.asarray(x["jitter"], np.float64).reshape(2))
    e, e_prev = (1.0, 1.0) if ldr else (_post.exposure_or_one(x["exposure"]), _post.exposure_or_one(x["exposure_prev"]))
    reset = bool(x["reset"])
    hist = np.asarray(x["history"], np.float64)
    s = float(np.asarray(s, np.float64).reshape(-1)[0])

    px, py = _post.target_positions(rw, rh, dw, dh, jx, jy)
    with np.errstate(invalid="ignore", over="ignore"):
        ec = enc_in(e * color)
    feat = np.concatenate([ec, np.asarray(x["B"], np.float64), np.asarray(x["A"], np.float64)], axis=2)
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
                hs = dec(hs) / e_prev
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
        H = np.where((w2 == 0)[..., None], w1[..., None] * h1, w1[..., None] * h1 + w2[..., None] * h2)

        # 5. blend
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
        # Codes: HDR k = E(.) / 2.45; LDR k = the value itself. "display" keeps Model M's convention (code = display /
        # 2.45), so LDR stores 2.45 clamp(O^, 0, 1).
        if ldr:
            code = np.rint(_post.clamp(M, 0.0, 1.0) * 1023.0)
        else:
            code = np.rint(_post.clamp(enc(M) / _post.REDUCED_SCALE, 0.0, 1.0) * 1023.0)
        if ldr:
            disp = _post.clamp(Oh, 0.0, 1.0) * _post.DISPLAY_MAX
        else:
            disp = _post.clamp(enc(Oh), 0.0, _post.DISPLAY_MAX)
    return {"history": _post.f16(enc(Hh)), "reduced": (code / 1023.0).astype(np.float32),
            "display": _post.f16(disp),
            "_lin": {"H_hat": Hh, "O_hat": Oh, "W": W, "first": first, "second": second, "w1": w1, "w2": w2}}
