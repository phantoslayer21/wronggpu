# SPDX-License-Identifier: MIT
"""NumPy reference of spec/03-input-encoder.md (stage "enc0" of spec/90-test-interface.md): the clean team's oracle.

All arithmetic is float64 except where the spec prescribes a rounding (FP16 values, FP16 running sums, FP8 results).

    out = encode(x, params)
    x:      the dict of spec 90 "Input encoder (spec 03): stage enc0" (color, depth, motion, ... motion_history)
    params: {"W": [64, 80], "b": [64], "T": [8, 16, 16], "sigma1": scalar, "sigma2": scalar}
    out:    {"features": [fh/2, fw/2, 64] (FP8-exact), "m_near" / "m_far" / "m_center": [rh, rw, 2] (FP16-exact),
             "values": [fh, fw, 20] (the 20 values before quantization), "qvalues": [fh, fw, 20] (FP8 after dither)}

Follows spec 03 draft 2. Choices the clean team made where draft 1 left room were confirmed in ANSWERS.md (spec 03).
"""
import numpy as np

from .fp8 import q8, f16
from . import motion_hi

N_VALUES = 20          # values per pixel
N_FEATURES = 64        # output channels
FEATURE_CLAMP = 6.28125
VALUE_CAP = 1024.0
LOG_SCALE = 0.36
LOG_FLOOR = 1e-4
COLOR_CAP = 32752.0
HIST_SCALE = 2.45
HIST_MIN, HIST_MAX = 1e-4, 3.0
REF_HEIGHT = 540.0

# 3 x 3 depth neighbourhood in tie-break order (spec 03, "Depth neighbourhood"): (dx, dy).
NEIGHBOURS = ((0, 0), (-1, -1), (0, -1), (-1, 0), (1, -1), (1, 0), (-1, 1), (0, 1), (1, 1))

# Index ranges of the 20 values (spec 03, "Order of the 20 values").
V_COLOR, V_NEAR, V_CENTER, V_FAR, V_UNJIT, V_DISAGREE, V_LATENT = 0, 3, 6, 9, 12, 15, 16


# ---- helpers -------------------------------------------------------------------------------------------------------

def reflect(p, n):
    """Padded-grid coordinate -> real render coordinate: p < 0 -> |p|, p >= n -> 2n - p - 2 (spec 03).
    check_inputs() rejects padding wider than n - 1, so the final clamp never changes a value."""
    p = np.abs(np.asarray(p))
    p = np.where(p >= n, 2 * n - p - 2, p)
    return np.clip(p, 0, n - 1)


def exposure_or_one(e):
    e = float(e)
    return 1.0 if e == 0.0 else e


def bilinear(tex, u, v):
    """Bilinear sample of tex [h, w, c] at normalized (u, v): texel centers at (i + 0.5) / w, clamp-to-edge.

    Non-finite coordinates give NaN. Exact bilinear weights (ANSWERS.md, spec 03 question 3)."""
    tex = np.asarray(tex, dtype=np.float64)
    h, w = tex.shape[:2]
    u = np.asarray(u, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    bad = ~(np.isfinite(u) & np.isfinite(v))
    with np.errstate(invalid="ignore"):
        tx = np.clip(np.where(bad, 0.0, u) * w - 0.5, -1.0, w)   # clipping here does not change the result
        ty = np.clip(np.where(bad, 0.0, v) * h - 0.5, -1.0, h)
    x0 = np.floor(tx)
    y0 = np.floor(ty)
    fx = (tx - x0)[..., None]
    fy = (ty - y0)[..., None]
    x0 = x0.astype(np.int64)
    y0 = y0.astype(np.int64)
    xa, xb = np.clip(x0, 0, w - 1), np.clip(x0 + 1, 0, w - 1)
    ya, yb = np.clip(y0, 0, h - 1), np.clip(y0 + 1, 0, h - 1)
    r = ((1 - fx) * (1 - fy) * tex[ya, xa] + fx * (1 - fy) * tex[ya, xb]
         + (1 - fx) * fy * tex[yb, xa] + fx * fy * tex[yb, xb])
    return np.where(bad[..., None], np.nan, r)


def encode_color(c, e):
    """E(c) = 0.36 log10(max(1 + 100 min(e c, 32752), 1e-4)); NaN product -> 0."""
    with np.errstate(invalid="ignore", over="ignore"):
        ec = e * np.asarray(c, dtype=np.float64)
        r = LOG_SCALE * np.log10(np.maximum(1.0 + 100.0 * np.minimum(ec, COLOR_CAP), LOG_FLOOR))
    return np.where(np.isnan(ec), 0.0, r)


def encode_history(h, e):
    """G(h) = 0.36 log10(max(1 + 100 e h, 1e-4)) (no upper cap; NaN stays NaN and becomes 0 later)."""
    with np.errstate(invalid="ignore", over="ignore", divide="ignore"):
        return LOG_SCALE * np.log10(np.maximum(1.0 + 100.0 * e * np.asarray(h, dtype=np.float64), LOG_FLOOR))


def history_color(sample, e_prev):
    """Reduced-history sample -> linear colour h (spec 03, "History", valid case)."""
    s = np.asarray(sample, dtype=np.float64)
    if np.isnan(e_prev) or e_prev < 0:
        return HIST_SCALE * s
    q = np.clip(HIST_SCALE * s, HIST_MIN, HIST_MAX)
    with np.errstate(over="ignore"):
        return (10.0 ** (q / LOG_SCALE) - 1.0) / 100.0 / e_prev


def dither_fp8(v, t):
    """Dithered FP8 quantization, spec 03 draft 2, steps 1-5:
    1. h = RN16(v); normal h: 2^E <= |h| < 2^(E+1), u = sign(h) 2^E; zero or FP16 subnormal: E = -14, u = 0.
    2. delta = sign(h) 2^E t / 1024.
    3. s = RN16(h + u + delta), one rounding of the exact sum (for normal h on the grid of 2^(E+1)).
    4. d = s - u (exact).
    5. Truncate d toward zero to 3 mantissa bits; convert to E4M3 (nearest-even, saturating at +-448; ANSWERS.md).
    Non-finite h (only v <= -65520, below the spec's range of interest) passes through and saturates at -448."""
    h = np.asarray(v, dtype=np.float64).astype(np.float16)
    hv = h.astype(np.float64)
    a = np.abs(hv)
    fin = np.isfinite(a)
    normal = fin & (a >= 2.0 ** -14)
    _, ex = np.frexp(np.where(normal, a, 1.0))                # a = m 2^ex, m in [0.5, 1) -> E = ex - 1
    p2 = np.where(normal, np.ldexp(1.0, ex - 1), 2.0 ** -14)
    sgn = np.where(fin, np.sign(hv), 0.0)
    u = np.where(normal, sgn * p2, 0.0)
    delta = sgn * p2 * np.asarray(t, dtype=np.float64) / 1024.0
    s = (np.where(fin, hv + u + delta, hv)).astype(np.float16).astype(np.float64)  # exact sum, one rounding
    d = np.where(fin, s - u, hv).astype(np.float16)                                 # exact
    d = (d.view(np.uint16) & np.uint16(0xFF80)).view(np.float16)
    return q8(d.astype(np.float64))


# ---- the encoder ---------------------------------------------------------------------------------------------------

def check_inputs(x):
    color = np.asarray(x["color"])
    if color.ndim != 3 or color.shape[2] != 3:
        raise ValueError(f"color must be [rh, rw, 3], got {color.shape}")
    rh, rw = color.shape[:2]
    fw, fh = (int(v) for v in x["feature_size"])
    dw, dh = (int(v) for v in x["recon_size"])
    shapes = {"depth": (rh, rw), "motion": (rh, rw, 2), "latent": (rh, rw, 4), "motion_history": (rh, rw, 2)}
    if motion_output_res(x):  # the game's motion texture at output size M (plus its origin), not [rh, rw, 2]
        del shapes["motion"]
        mw, mh = (int(v) for v in x["motion_size"])
        ox, oy = (int(v) for v in x.get("motion_origin", (0, 0)))
        ms = np.shape(x["motion"])
        if len(ms) != 3 or ms[2] != 2 or mw <= 0 or mh <= 0 or min(ox, oy) < 0 or ms[0] < oy + mh or ms[1] < ox + mw:
            raise ValueError(f"motion texture {ms} does not hold motion_size {(mw, mh)} at origin {(ox, oy)}")
    for k, s in shapes.items():
        if np.shape(x[k]) != s:
            raise ValueError(f"{k} has shape {np.shape(x[k])}, expected {s}")
    hist = np.shape(x["reduced_history"])
    if len(hist) != 3 or hist[2] != 3:
        raise ValueError(f"reduced_history must be [dh/2, dw/2, 3], got {hist}")
    if fw < rw or fh < rh or fw % 2 or fh % 2:
        raise ValueError(f"feature_size {(fw, fh)} must be even and at least the render size {(rw, rh)}")
    if fw > 2 * rw - 1 or fh > 2 * rh - 1:
        raise ValueError(f"feature_size {(fw, fh)}: padding larger than the render size {(rw, rh)} - 1")
    if dw <= 0 or dh <= 0:
        raise ValueError(f"recon_size {(dw, dh)} must be positive")
    if not 0 <= int(x["frame"]) < 8:
        raise ValueError(f"frame {x['frame']} out of range 0..7")
    return rw, rh, fw, fh, dw, dh


def motion_output_res(x):
    """True for a frame whose motion is at output resolution (spec 03 draft 4, "Motion at output resolution"):
    x["motion"] is then the game's motion texture [rows][columns][2], x["motion_size"] = M = (Mw, Mh),
    x["motion_origin"] its subrect origin (default (0, 0)), and x["mv_scale"] converts to pixels of M."""
    return bool(x.get("motion_output_res", False))


def motion_surfaces(x):
    """Per render pixel: m_near, m_far, m_center (reconstruction pixels, float64, before FP16 storage)."""
    rw, rh, _, _, dw, dh, = check_inputs(x)
    if motion_output_res(x):
        # one point read per render pixel, no depth neighbourhood: the three motions are the same FP32 value
        m = motion_hi.encoder_motion(x["motion"], (rw, rh), x["motion_size"], (dw, dh), x["jitter"], x["mv_scale"],
                                     x["mv_bias"], x.get("motion_origin", (0, 0))).astype(np.float64)
        return m, m, m
    depth = np.asarray(x["depth"], dtype=np.float64)
    mv = np.asarray(x["motion"], dtype=np.float64)
    s = np.asarray(x["mv_scale"], dtype=np.float64) * np.array([dw / rw, dh / rh])
    m = s * (np.asarray(x["mv_bias"], dtype=np.float64) + mv)          # [rh, rw, 2], (x, y) order

    Y, X = np.mgrid[0:rh, 0:rw]
    lo_v, hi_v = depth.copy(), depth.copy()
    lo_x, lo_y, hi_x, hi_y = X.copy(), Y.copy(), X.copy(), Y.copy()
    for dx, dy in NEIGHBOURS[1:]:
        cx, cy = np.clip(X + dx, 0, rw - 1), np.clip(Y + dy, 0, rh - 1)
        dv = depth[cy, cx]
        lo = dv < lo_v                                                  # strict: ties keep the earlier candidate
        hi = dv > hi_v
        lo_v, lo_x, lo_y = np.where(lo, dv, lo_v), np.where(lo, cx, lo_x), np.where(lo, cy, lo_y)
        hi_v, hi_x, hi_y = np.where(hi, dv, hi_v), np.where(hi, cx, hi_x), np.where(hi, cy, hi_y)
    if bool(x["depth_inverted"]):
        near, far = (hi_y, hi_x), (lo_y, lo_x)
    else:
        near, far = (lo_y, lo_x), (hi_y, hi_x)
    return m[near], m[far], m


def pixel_values(x, params):
    """The 20 values of every padded-grid pixel, [fh, fw, 20] float64 (after NaN -> 0 and the cap at 1024)."""
    rw, rh, fw, fh, dw, dh = check_inputs(x)
    m_near, m_far, m_center = motion_surfaces(x)
    e = exposure_or_one(x["exposure"])
    e_prev = exposure_or_one(x["exposure_prev"])
    reset = bool(x["reset"])
    j = np.asarray(x["jitter"], dtype=np.float64)
    jp = np.asarray(x["jitter_prev"], dtype=np.float64)
    R = np.array([rw, rh], dtype=np.float64)
    D = np.array([dw, dh], dtype=np.float64)
    k = D / R
    sigma1 = float(np.asarray(params["sigma1"]))
    sigma2 = float(np.asarray(params["sigma2"]))

    Y, X = np.mgrid[0:fh, 0:fw]
    xr, yr = reflect(X, rw), reflect(Y, rh)
    color = np.asarray(x["color"], dtype=np.float64)[yr, xr]           # [fh, fw, 3]
    mn, mf, mc = m_near[yr, xr], m_far[yr, xr], m_center[yr, xr]        # [fh, fw, 2]
    pix = np.stack([xr, yr], axis=-1).astype(np.float64)
    base = pix + 0.5 - j
    p_near, p_center, p_far = base * k + mn, base * k + mc, base * k + mf
    p_unjit = (pix + 0.5) * k + mc

    def valid(p):
        if reset:
            return np.zeros(p.shape[:-1], dtype=bool)
        with np.errstate(invalid="ignore"):
            return np.all((p >= 0) & (p <= D), axis=-1)       # NaN positions are invalid

    vals = np.zeros((fh, fw, N_VALUES), dtype=np.float64)
    vals[..., V_COLOR:V_COLOR + 3] = encode_color(color, e)
    hist_tex = np.asarray(x["reduced_history"], dtype=np.float64)
    for off, p in ((V_NEAR, p_near), (V_CENTER, p_center), (V_FAR, p_far), (V_UNJIT, p_unjit)):
        ok = valid(p)
        smp = bilinear(hist_tex, p[..., 0] / dw, p[..., 1] / dh)
        h = np.where(ok[..., None], history_color(smp, e_prev), color)
        vals[..., off:off + 3] = encode_history(h, e)

    if not reset:
        with np.errstate(invalid="ignore", over="ignore"):
            u = np.clip((base + jp + mn / k) / R, 0.0, 1.0)
            m_old = bilinear(x["motion_history"], u[..., 0], u[..., 1])
            v = sigma1 * (REF_HEIGHT / rh) * (mn - m_old) / k
            vals[..., V_DISAGREE] = sigma2 * np.log(1.0 + np.hypot(v[..., 0], v[..., 1]))

    ok = valid(p_near)
    with np.errstate(invalid="ignore", over="ignore"):
        lp = (p_near / k + jp) / R
    lat = bilinear(x["latent"], lp[..., 0], lp[..., 1])
    vals[..., V_LATENT:V_LATENT + 4] = np.where(ok[..., None], lat, 0.0)

    vals = np.where(np.isnan(vals), 0.0, vals)
    return np.minimum(vals, VALUE_CAP)


def dither_table(T, frame, fw, fh):
    """t(x, y) = T[n][|y| mod 16][|x| mod 16] on the raw padded grid, [fh, fw]."""
    T = np.asarray(T)
    Y, X = np.mgrid[0:fh, 0:fw]
    return T[int(frame)][np.abs(Y) % 16, np.abs(X) % 16]


def project(qvalues, W, b):
    """Features from the quantized values [fh, fw, 20] (spec 03, "Projection to features").

    Running sum s = b; for sub-pixel p = 0..3: s = FP16(s + W[:, 20p:20p+20] a_p) (group sum exact);
    then clamp to +-6.28125 and round to FP8.
    The bias starts the running sum (ANSWERS.md, spec 03 question 2)."""
    fh, fw = qvalues.shape[:2]
    W = np.asarray(W, dtype=np.float64)
    a = qvalues.reshape(fh // 2, 2, fw // 2, 2, N_VALUES)              # [r, dy, c, dx, i]
    s = np.broadcast_to(np.asarray(b, dtype=np.float64), (fh // 2, fw // 2, N_FEATURES))
    for p in range(4):
        dx, dy = p % 2, p // 2
        s = f16(s + a[:, dy, :, dx, :] @ W[:, N_VALUES * p:N_VALUES * (p + 1)].T)
    return q8(np.clip(s, -FEATURE_CLAMP, FEATURE_CLAMP))


def encode(x, params):
    rw, rh, fw, fh, dw, dh = check_inputs(x)
    vals = pixel_values(x, params)
    t = dither_table(params["T"], x["frame"], fw, fh)
    qv = dither_fp8(vals, t[..., None])
    feats = project(qv, params["W"], params["b"])
    m_near, m_far, m_center = motion_surfaces(x)
    return {"features": feats.astype(np.float32),
            "m_near": f16(m_near).astype(np.float32),
            "m_far": f16(m_far).astype(np.float32),
            "m_center": f16(m_center).astype(np.float32),
            "values": vals, "qvalues": qv}


# ---- synthetic data (tests only; never the real model) -------------------------------------------------------------

def random_params(rng, w_scale=0.08):
    W = q8(rng.normal(0.0, w_scale, size=(N_FEATURES, 4 * N_VALUES)))
    b = f16(rng.normal(0.0, 0.5, size=N_FEATURES))
    return {"W": W.astype(np.float32), "b": b.astype(np.float32),
            "T": rng.integers(0, 128, size=(8, 16, 16)).astype(np.float32),
            "sigma1": np.array(rng.uniform(0.5, 2.0), dtype=np.float32),
            "sigma2": np.array(rng.uniform(0.2, 1.5), dtype=np.float32)}


def random_inputs(rng, rw=40, rh=24, fw=None, fh=None, dw=None, dh=None, reset=False, frame=None,
                  depth_inverted=None, special=True):
    """Synthetic frame. special=True adds NaN / huge / negative colours, depth ties and NaN depths, out-of-range
    motion, so that every branch of spec 03 is exercised."""
    fw = fw or -(-rw // 32) * 32
    fh = fh or -(-rh // 32) * 32
    dw = dw or (3 * rw + 1) // 2 + 3
    dh = dh or (3 * rh + 1) // 2 + 1
    color = rng.lognormal(-1.0, 1.5, size=(rh, rw, 3))
    depth = np.round(rng.uniform(0, 1, size=(rh, rw)) * 12) / 12          # many ties
    yy, xx = np.mgrid[0:rh, 0:rw]
    motion = np.stack([2.0 * np.sin(xx / 5.0) + rng.normal(0, 0.7, (rh, rw)),
                       1.5 * np.cos(yy / 4.0) + rng.normal(0, 0.7, (rh, rw))], axis=-1)
    if special:
        n = rw * rh
        flat = color.reshape(-1, 3)
        idx = rng.choice(n, size=max(4, n // 50), replace=False)
        flat[idx[0::4]] = np.nan
        flat[idx[1::4]] = rng.uniform(1e3, 1e6, size=(len(idx[1::4]), 3))
        flat[idx[2::4]] = -rng.uniform(0.0, 0.05, size=(len(idx[2::4]), 3))
        flat[idx[3::4], 0] = np.inf
        depth.reshape(-1)[rng.choice(n, size=max(1, n // 100), replace=False)] = np.nan
        mflat = motion.reshape(-1, 2)
        mflat[rng.choice(n, size=max(2, n // 40), replace=False)] *= 40.0  # some positions leave the frame
    hh_, hw_ = dh // 2, dw // 2
    return {
        "color": color.astype(np.float32),
        "depth": depth.astype(np.float32),
        "motion": motion.astype(np.float32),
        "mv_scale": np.array(rng.uniform(0.8, 1.2, size=2) * rng.choice([-1.0, 1.0], size=2), dtype=np.float32),
        "mv_bias": np.array(rng.normal(0, 0.05, size=2), dtype=np.float32),
        "jitter": np.array(rng.uniform(-0.5, 0.5, size=2), dtype=np.float32),
        "jitter_prev": np.array(rng.uniform(-0.5, 0.5, size=2), dtype=np.float32),
        "exposure": float(np.float32(rng.uniform(0.3, 3.0))),
        "exposure_prev": float(np.float32(rng.uniform(0.3, 3.0))),
        "reset": bool(reset),
        "frame": int(rng.integers(0, 8) if frame is None else frame),
        "depth_inverted": bool(rng.integers(0, 2) if depth_inverted is None else depth_inverted),
        "feature_size": (fw, fh),
        "recon_size": (dw, dh),
        "reduced_history": (rng.integers(0, 1024, size=(hh_, hw_, 3)) / 1023.0).astype(np.float32),
        "latent": (rng.integers(-127, 128, size=(rh, rw, 4)) / 127.0).astype(np.float32),
        "motion_history": f16(rng.normal(0, 3.0, size=(rh, rw, 2))).astype(np.float32),
    }
