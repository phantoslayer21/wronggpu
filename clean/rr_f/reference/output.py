# SPDX-License-Identifier: MIT
"""Original CPU oracle for published spec 54, through CH, Q and PH1 only.

Q is an internal output-gated image, NOT the game output: resolve/PH2 (missing
page 54b) and input-stage motion construction (missing page 52) are excluded.
All textures are [y, x, channel]. Callers supply decoded binary16 texture
values, including G/W4's binary16 UNORM read-back, and every symbolic constant.
No learned data or converter constants are supplied by this module.

Page 54.N explicitly restates the numeric rules it attributes to 54b; those
restated rules suffice here. Bilinear uses a fixed, permitted f32 evaluation.
This deliberately slow scalar/NumPy reference favours inspectable rounding.
"""
from dataclasses import dataclass, fields
from decimal import Decimal, localcontext
from fractions import Fraction
from functools import lru_cache
import math

import numpy as np

from . import exact
from .fmt import add16, fma16, mul16, round_f16


def f32(value):
    """Round to binary32, flush subnormal results while preserving zero sign."""
    with np.errstate(over="ignore", invalid="ignore"):
        value = np.asarray(value, dtype=np.float32)
    return np.where(np.abs(value) < np.finfo(np.float32).tiny,
                    np.copysign(np.float32(0), value), value).astype(np.float32)


def add(a, b):
    with np.errstate(invalid="ignore", over="ignore"):
        return f32(f32(a) + f32(b))


def mul(a, b):
    with np.errstate(invalid="ignore", over="ignore"):
        return f32(f32(a) * f32(b))


def _fma_scalar(a, b, c):
    if not all(math.isfinite(t) for t in (a, b, c)):
        with np.errstate(invalid="ignore", over="ignore"):
            return float(f32(a * b + c))
    v = Fraction(a) * Fraction(b) + Fraction(c)
    if v == 0:
        # IEEE exact cancellation is +0; only (-0 product) + (-0) is -0.
        return -0.0 if a * b == 0 and c == 0 and math.copysign(1, a * b) < 0 and math.copysign(1, c) < 0 else 0.0
    return float(f32(exact.round_frac(v, exact.F32)))


def fma(a, b, c):
    """One exact rounding, not a double-precision multiply/add approximation."""
    a, b, c = np.broadcast_arrays(f32(a), f32(b), f32(c))
    out = np.empty(a.shape, dtype=np.float32)
    for idx in np.ndindex(a.shape):
        out[idx] = _fma_scalar(float(a[idx]), float(b[idx]), float(c[idx]))
    return out


@lru_cache(maxsize=8192)
def _elementary(kind, value):
    if kind == "rcp":
        if math.isnan(value):
            return math.nan
        if value == 0:
            return math.copysign(math.inf, value)
        if math.isinf(value):
            return math.copysign(0.0, value)
        return exact.round_frac(1 / Fraction(value), exact.F32)
    if kind == "exp2":
        return exact.exp2_correct_f32(value)
    if math.isnan(value) or value < 0:
        return math.nan
    if value == 0:
        return -math.inf
    if value == math.inf:
        return math.inf
    with localcontext() as ctx:
        ctx.prec = 100
        answer = Decimal(value).ln() / Decimal(2).ln()
    return exact.round_frac(Fraction(answer), exact.F32)


def _map_elementary(kind, value):
    value = f32(value)
    out = np.empty(value.shape, dtype=np.float32)
    for idx in np.ndindex(value.shape):
        scalar = float(value[idx])
        # Python hashes +0.0 and -0.0 identically. Reciprocal must preserve
        # their distinct signs, including zeros produced by the input flush.
        out[idx] = (math.copysign(math.inf, scalar) if kind == "rcp" and scalar == 0
                    else _elementary(kind, scalar))
    return f32(out)


def rcp(value):
    return _map_elementary("rcp", value)


def exp2(value):
    return _map_elementary("exp2", value)


def log2(value):
    return _map_elementary("log2", value)


def sat(value):
    return np.fmin(np.fmax(f32(value), np.float32(0)), np.float32(1))


# Only the mathematical constants explicitly given in 54.11.
K6 = np.float32(math.log10(2))
K9 = np.float32(math.log2(10))


@dataclass(frozen=True)
class OutputConstants:
    K1: float
    K2n: float
    K2x2n: float
    K3: float
    K4: float
    K5: float
    K7: float
    K8: float
    K10: float
    K11: float
    K12: float
    K13: float
    TR: np.ndarray

    def __post_init__(self):
        # Validate representation and the only explicit sign constraints on 54.
        # Additional converter ranges await page 57; do not invent them here.
        for field in fields(self):
            value = np.asarray(getattr(self, field.name), dtype=np.float64)
            shape = (513, 4) if field.name == "TR" else ()
            if value.shape != shape:
                raise ValueError(f"{field.name} must have shape {shape}")
            with np.errstate(over="ignore", invalid="ignore"):
                rounded = value.astype(np.float32).astype(np.float64)
            if not np.array_equal(value, rounded, equal_nan=True):
                raise ValueError(f"{field.name} must contain exact f32 values")
            stored = value.astype(np.float32)
            if field.name == "TR":
                stored.setflags(write=False)
            object.__setattr__(self, field.name, stored)
        if not (self.K2n < 0 and self.K2x2n < 0):
            raise ValueError("54.3 requires K2n and K2x2n delivered negative")


def compress(value, constants):
    """C of 54.6, including max's NaN handling."""
    c = constants
    return mul(mul(log2(np.fmax(fma(value, c.K4, 1), c.K5)), K6), c.K7)


def decompress(value, constants):
    """D of 54.6; result is K4 times linear colour."""
    c = constants
    value = f32(value)
    clamped = np.where((value > c.K5) | np.isnan(value),
                       np.minimum(value, c.K10), c.K5)
    return add(exp2(mul(mul(clamped, c.K8), K9)), -1)


def decode_unorm8(codes):
    """54.N/55.3: decoded binary16 values, not f32 code/255."""
    codes = np.asarray(codes)
    if codes.dtype != np.uint8:
        raise ValueError("codes must be uint8")
    return round_f16(codes.astype(np.float64) / 255)


def integer_fetch(texture, x, y):
    """54.N: out-of-texture integer loads return exact zeros."""
    h, w = texture.shape[:2]
    if not (0 <= x < w and 0 <= y < h):
        return np.zeros(texture.shape[2:], dtype=np.float32)
    return f32(texture[y, x])


def bilinear(texture, s, t):
    """54.N formula: clamp-to-edge, NaN coordinates become zero.

    Infinite normalized coordinates have no floor/fraction in the published
    formula; they are rejected rather than given an invented sampling policy.
    Finite motion infinities take the specified reprojection fallback first.
    """
    h, w = texture.shape[:2]
    s, t = float(f32(s)), float(f32(t))
    s, t = (0.0 if math.isnan(s) else s), (0.0 if math.isnan(t) else t)
    if not math.isfinite(s) or not math.isfinite(t):
        raise ValueError("54.N does not define bilinear fractional weights at infinity")
    a, b = add(mul(s, w), -0.5), add(mul(t, h), -0.5)
    if not np.isfinite(a) or not np.isfinite(b):
        raise ValueError("54.N bilinear coordinate product overflow")
    i, j = math.floor(float(a)), math.floor(float(b))
    fa, fb = add(a, -i), add(b, -j)
    ia, ib = min(max(i, 0), w - 1), min(max(i + 1, 0), w - 1)
    ja, jb = min(max(j, 0), h - 1), min(max(j + 1, 0), h - 1)
    weights = [mul(add(1, -fa), add(1, -fb)), mul(fa, add(1, -fb)),
               mul(add(1, -fa), fb), mul(fa, fb)]
    taps = [texture[ja, ia], texture[ja, ib], texture[jb, ia], texture[jb, ib]]
    result = mul(weights[0], taps[0])
    for weight, tap in zip(weights[1:], taps[1:]):
        result = add(result, mul(weight, tap))
    return result


def candidate_order(weights):
    """Exact five-comparison network of 54.4, returning original indices."""
    weights = np.asarray(weights, dtype=np.float32)
    if weights.shape != (4,):
        raise ValueError("four candidate weights required")

    def cmp(a, b):
        wa, wb = weights[a], weights[b]
        return (a, b) if wa >= wb or np.isnan(wa) or np.isnan(wb) else (b, a)

    hi12, lo12 = cmp(0, 1)
    hi34, lo34 = cmp(2, 3)
    p0, mid1 = cmp(hi12, hi34)
    mid2, p3 = cmp(lo12, lo34)
    p1, p2 = cmp(mid2, mid1)
    return np.array((p0, p1, p2, p3), dtype=np.int64)


def candidate_weights(weights, constants):
    order = candidate_order(weights)
    w = sat(np.asarray(weights)[order])
    total = add(add(add(w[1], w[0]), w[2]), w[3])
    return order, mul(w, rcp(np.fmax(total, constants.K13)))


def reconstruct_history(history, x, y, motion, fallback, reset, constants):
    """One candidate h(i), 54.5. NaN anchors remain float until sampling."""
    h, w = history.shape[:2]
    px, py = add(x + 0.5, motion[0]), add(y + 0.5, motion[1])
    if reset or px < 0 or px > w or py < 0 or py > h:
        return f32(fallback)
    # np.floor retains NaN; sat(NaN) is 0, so TR[0] is well-defined.
    kx, ky = add(np.floor(add(px, -0.5)), 0.5), add(np.floor(add(py, -0.5)), 0.5)
    fx, fy = sat(add(px, -kx)), sat(add(py, -ky))
    ix = min(math.trunc(float(fma(fx, 512, 0.5))), 512)
    iy = min(math.trunc(float(fma(fy, 512, 0.5))), 512)
    tx, ty = constants.TR[ix], constants.TR[iy]
    weights = [mul(tx[1], ty[0]), mul(tx[0], ty[1]), mul(tx[1], ty[1]),
               mul(tx[1], ty[2]), mul(tx[2], ty[1])]
    positions = [(add(kx, tx[3]), add(ky, -1)), (add(kx, -1), add(ky, ty[3])),
                 (add(kx, tx[3]), add(ky, ty[3])), (add(kx, tx[3]), add(ky, 2)),
                 (add(kx, 2), add(ky, ty[3]))]
    taps = [bilinear(history[..., :3], mul(xx, rcp(w)), mul(yy, rcp(h))) for xx, yy in positions]
    total = add(weights[4], add(weights[3], add(weights[2], add(weights[1], weights[0]))))
    accum = mul(taps[0], weights[0])
    for tap, weight in zip(taps[1:], weights[1:]):
        accum = fma(tap, weight, accum)
    return mul(accum, rcp(total))


def history_estimate(history, x, y, weights, motions, current, E, Ep, invE,
                     reset, constants):
    """54.4-54.5 normalized candidate loop; p3 never sampled."""
    c = constants
    order, norm = candidate_weights(weights, c)
    fallback = compress(mul(Ep, mul(invE, mul(current, c.K12))), c)

    def sample(index):
        return reconstruct_history(history, x, y, motions[order[index]], fallback, reset, c)

    accum, total = mul(norm[0], sample(0)), norm[0]
    for index in (1, 2):
        if not total < c.K3:
            break
        accum = fma(norm[index], sample(index), accum)
        total = add(total, norm[index])
    linear = decompress(mul(accum, rcp(total)), c)
    return mul(E, mul(mul(linear, c.K12), rcp(Ep)))


def blend_guard(current, history, alpha, constants):
    """54.7 gate-local blend and three-channel NaN guard."""
    value = fma(alpha, fma(current, constants.K12, -np.asarray(history)), history)
    if np.isnan(add(value[2], add(value[0], value[1]))):
        return mul(current, constants.K12)
    return value


def unorm_codes(value, bits):
    """54.N exact scaled product, RNE integer store; no f16 intermediate."""
    if bits not in (2, 10):
        raise ValueError("RGB10A2 channels use 10 or 2 bits")
    return np.rint(sat(value).astype(np.float64) * ((1 << bits) - 1)).astype(np.uint16)


def pool_history(linear, constants):
    """54.7 PH1 as uint16 channel codes [R10,G10,B10,A2], not packed words."""
    linear = np.asarray(linear)
    if linear.ndim != 3 or linear.shape[-1] != 3 or min(linear.shape[:2]) < 1:
        raise ValueError("linear must be nonempty [Hh, Wh, 3]")
    h, w = linear.shape[:2]
    result = np.empty(((h + 1) // 2, (w + 1) // 2, 4), dtype=np.uint16)
    for y in range(0, h, 2):
        for x in range(0, w, 2):
            tl = integer_fetch(linear, x, y)
            bl = integer_fetch(linear, x, y + 1)
            tr = integer_fetch(linear, x + 1, y)
            br = integer_fetch(linear, x + 1, y + 1)
            count = min(2, w - x) * min(2, h - y)
            mean = mul(add(add(tl, bl), add(tr, br)), rcp(count))
            result[y // 2, x // 2, :3] = unorm_codes(mul(compress(mean, constants), constants.K11), 10)
            result[y // 2, x // 2, 3] = 3
    return result


@dataclass(frozen=True)
class OutputResult:
    """CH float16 RGBA, Q float16 RGB, PH1 uint16 RGBA channel codes."""
    CH: np.ndarray
    Q: np.ndarray
    PH1: np.ndarray


def _decoded_texture(value, shape, name):
    value = np.asarray(value, dtype=np.float64)
    if value.shape != shape:
        raise ValueError(f"{name} must have shape {shape}")
    if not np.array_equal(value, round_f16(value), equal_nan=True):
        raise ValueError(f"{name} must contain decoded binary16 values")
    return value


def _unorm_texture(value, shape, name):
    value = _decoded_texture(value, shape, name)
    decoded = decode_unorm8(np.arange(256, dtype=np.uint8))
    if not np.all(np.isin(value, decoded)):
        raise ValueError(f"{name} must contain decoded UNORM8 read-back values; use decode_unorm8 on stored codes")
    return value


def _exposure_scalar(value, name):
    value = np.asarray(value, dtype=np.float64)
    if value.shape != ():
        raise ValueError(f"{name} must be a scalar f32 value")
    with np.errstate(over="ignore", invalid="ignore"):
        rounded = value.astype(np.float32)
    if not np.array_equal(value, rounded.astype(np.float64), equal_nan=True):
        raise ValueError(f"{name} must be an exact f32 value")
    return rounded


def output_stage(Cnet, G, P, W4, PM, Pe, Pn, Hprev, *, output_size,
                 jitter, E, Ep, invE, reset, constants):
    """Run 54 from functional inputs already prepared by pages 52/55/58.

    output_size is (Wo, Ho); G/P have exactly 3 used channels, PM has 4,
    Pe/Pn have 2. Caller chooses parity and supplies the correct PM/Hprev.
    G/W4 must be decoded stored UNORM8 codes, not pre-store head values.
    E/Ep/invE must be scalar f32 values; reset must be boolean.
    This function does not change lifecycle state or apply absent page 52.
    """
    wo, ho = output_size
    if any(isinstance(v, bool) or not isinstance(v, (int, np.integer)) or v <= 0 for v in (wo, ho)):
        raise ValueError("output size must contain positive integers")
    wh, hh = 3 * (wo + wo % 2) // 2, 3 * (ho + ho % 2) // 2
    if np.ndim(G) != 3:
        raise ValueError("G must have shape [Hr, Wr, 3]")
    hr, wr = np.shape(G)[:2]
    if min(hr, wr) < 1 or wr * 3 > wh * 2 or hr * 3 > hh * 2:
        raise ValueError("54.8 requires positive render size and sx, sy <= 2/3")
    wt, ht = ((wr + 31) // 32) * 32, ((hr + 31) // 32) * 32
    Cnet = _decoded_texture(Cnet, (ht, wt, 4), "Cnet")
    G = _unorm_texture(G, (hr, wr, 3), "G")
    P = _decoded_texture(P, (hr, wr, 3), "P")
    W4 = _unorm_texture(W4, (hr, wr, 4), "W4")
    PM = _decoded_texture(PM, (hr, wr, 4), "PM")
    Pe = _decoded_texture(Pe, (hr, wr, 2), "Pe")
    Pn = _decoded_texture(Pn, (hr, wr, 2), "Pn")
    Hprev = _decoded_texture(Hprev, (hh, wh, 4), "Hprev")
    if np.shape(jitter) != (2,) or not np.all(np.isfinite(jitter)):
        raise ValueError("finite two-component jitter required for 54.2 floor")
    jx, jy = f32(jitter)
    if (wr * 3 == wh * 2 and jx == 0.5) or (hr * 3 == hh * 2 and jy == 0.5):
        raise ValueError("54.8 excludes ratio 2/3 with jitter exactly +0.5")
    E, Ep, invE = (_exposure_scalar(E, "E"), _exposure_scalar(Ep, "Ep"),
                   _exposure_scalar(invE, "invE"))
    if not isinstance(reset, (bool, np.bool_)):
        raise ValueError("reset must be boolean")
    c = constants
    # Division is rounded once, as 54.1 requires (not multiplication by rcp).
    sx, sy = f32(wr / wh), f32(hr / hh)
    prepared = np.concatenate((round_f16(compress(Cnet[:hr, :wr, :3], c)), G), axis=2)
    precision = round_f16(mul(P, [c.K2n, c.K2n, c.K2x2n]))
    linear_h, linear_o = np.empty((hh, wh, 3), np.float32), np.empty((hh, wh, 3), np.float32)
    for y in range(hh):
        va = fma((y // 2) * 2 + 0.5, sy, jy)
        cya = math.floor(float(add(va, -0.5)))
        v = fma(y + 0.5, sy, jy)
        cy = math.floor(float(add(v, -0.5)))
        hy = round_f16(add(cy + 0.5, -v))
        if cy != cya:
            hy = add16(hy, -1)
        for x in range(wh):
            u = fma(x + 0.5, sx, jx)
            cx = math.floor(float(add(u, -0.5)))
            hx = round_f16(add(cx + 0.5, -u))
            accum, total = np.zeros(6), np.float32(0)
            for k in range(4):
                for m in range(4):
                    xx, yy = min(max(cx - 1 + k, 0), wr - 1), min(max(cya - 1 + m, 0), hr - 1)
                    ox, oy = add16(hx, k - 1), add16(hy, m - 1)
                    ax, ay, axy = precision[yy, xx]
                    exponent = fma16(mul16(ox, oy), axy,
                                     fma16(mul16(oy, oy), ay, mul16(mul16(ox, ox), ax)))
                    weight32 = exp2(exponent)
                    total = add(total, weight32)
                    accum = fma16(round_f16(weight32), prepared[yy, xx], accum)
            norm = mul(accum, rcp(np.fmax(total, c.K13)))
            coverage = sat(mul(norm[3], mul(total, c.K1)))
            alpha_h = fma(coverage, add(1, -norm[4]), norm[4])
            alpha_o = fma(coverage, add(1, -norm[5]), norm[5])
            current = decompress(norm[:3], c)
            s, t = mul(u, rcp(wr)), mul(v, rcp(hr))
            weights = bilinear(W4, s, t)
            pair = integer_fetch(PM, math.floor(float(u)), math.floor(float(v)))
            mv3 = integer_fetch(Pe, math.floor(float(u)), math.floor(float(v)))
            motions = np.array([pair[:2], bilinear(Pn, s, t), mv3, pair[2:]])
            hist = history_estimate(Hprev, x, y, weights, motions, current, E, Ep, invE, reset, c)
            linear_h[y, x] = blend_guard(current, hist, alpha_h, c)
            linear_o[y, x] = blend_guard(current, hist, alpha_o, c)
    ch = np.zeros((hh, wh, 4), dtype=np.float16)
    ch[..., :3] = round_f16(compress(linear_h, c)).astype(np.float16)
    q = round_f16(compress(linear_o, c)).astype(np.float16)
    return OutputResult(ch, q, pool_history(linear_h, c))
