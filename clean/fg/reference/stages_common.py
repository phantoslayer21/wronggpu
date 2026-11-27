# SPDX-License-Identifier: GPL-3.0-only
"""Frame generation (issue #96): CPU reference of the shared non-network stages (both the light path, preset C, and
the front half of the full path).

Written for this project from the behaviour description of the clean side's frame-generation notes. Every stage is a
plain function on NumPy arrays; the D3D12 kernels (clean/dx12/kernels/fg_*.hlsl) compute the same bits.

Images are arrays [height, width, channels] in their storage type (the clean side's own choice, FORMATS.md):
float16 for colour, mask and logit images, float32 for the processed motion, uint32 for the packed depth and flag
images, int64 for the 64-bit scatter buffers. A stage reads them, computes in single precision (one rounding per
operation, in the order written; fma = one rounding) and writes its outputs into arrays the caller passes, only where
the stage writes, so that an unwritten element keeps whatever the caller put there (a sentinel in the GPU check).

Arithmetic conventions (the clean side's choices where the description names a GPU approximation):
- rcp(x) is the correctly rounded single-precision 1 / x, and "approximate division" a / b is a * rcp(b) (two roundings);
- ex2(y) is this module's own deterministic 2^y (floor, a degree-6 polynomial in fma form, an exact power-of-two scale);
- sat, min and max follow the GPU's minNum / maxNum (a NaN operand gives the other operand);
- float-to-integer conversions truncate, saturate and give 0 for NaN;
- single-precision denormals are flushed to zero on the GPU; the test data keeps every value away from them.

Textures are emulated on buffers: a nearest read takes texel clamp(floor(X * W)); a bilinear read takes the four
texels around X * W - 0.5 (clamped to the edge) and blends with fma. Application inputs are read through their region
(base, extent, reciprocal texture size) with the clamp to the region's texel centres.
"""
from fractions import Fraction

import numpy as np

from clean.reference.motion_hi import fma32 as fma

F32 = np.float32
F16 = np.float16
INF = F32(np.inf)


def fbits(u):
    """The float32 with bit pattern u."""
    return np.array(u, dtype=np.uint32).view(np.float32)[()]


# constants (bit patterns as the description gives them)
LUMA_R, LUMA_G, LUMA_B = fbits(0x3E991687), fbits(0x3F1645A2), fbits(0x3DE978D5)   # 0.299, 0.587, 0.114
UI_ALPHA_THRESHOLD = fbits(0x3ECCCCCD)        # 0.4
UI_DIFF_THRESHOLD = fbits(0x3D8F5C29)         # 0.07
PUSH = fbits(0xC47A0000)                      # -1000
NEG_LOG2E = fbits(0xBFB8AA3B)                 # -1.44269502
BIAS = F32(2.0 ** -23)
KEY_FLOOR = F32(1e-7)
WEIGHT_FLOOR = F32(1e-12)
SENTINEL_CAM = F32(10000.0)
EMPTY64 = np.int64(0x7FFFFFFF7FFFFFFF)
CLEAR32 = 0x7FFFFFFF
KMAX_DEFAULT, TAU_DEFAULT = F32(600.0), F32(40.0)
# 2^f on [0, 1): coefficients from the highest power down (float32 bit patterns), evaluated in fma (Horner) form.
# Made for this project: a degree-6 least-squares fit of 2^f at 400 Chebyshev nodes of [0, 1], each coefficient rounded
# to float32 and the constant term set to 1 (largest relative error about 6e-8 on [0, 1]; test_fg_common_stages checks
# it against exp2).
EX2_COEFFS = [fbits(b) for b in (0x396566FB, 0x3AA25EA2, 0x3C1EAC13, 0x3D633F72, 0x3E75FEFC, 0x3F317214, 0x3F800000)]


# ---- scalar helpers (arrays) -----------------------------------------------------------------------------------------

def f32(x):
    return np.asarray(x, dtype=np.float32)


def rcp(x):
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        return (F32(1.0) / f32(x)).astype(np.float32)


def adiv(a, b):
    """The approximate division: a * rcp(b)."""
    with np.errstate(invalid="ignore", over="ignore"):
        return (f32(a) * rcp(b)).astype(np.float32)


def fmax(a, b):
    return np.fmax(f32(a), f32(b)).astype(np.float32)


def fmin(a, b):
    return np.fmin(f32(a), f32(b)).astype(np.float32)


def sat(x):
    return fmin(fmax(x, F32(0.0)), F32(1.0))


def med3(x, lo, hi):
    return fmin(fmax(x, lo), hi)


def finite(x):
    with np.errstate(invalid="ignore"):
        return np.abs(f32(x)) < INF


def f2i(x):
    """Truncating, saturating float -> int32 (NaN -> 0), as int64 values."""
    x = np.asarray(x, dtype=np.float64)
    with np.errstate(invalid="ignore"):
        t = np.trunc(np.clip(np.nan_to_num(x, nan=0.0), -2.0 ** 31, 2.0 ** 31 - 1))
    return t.astype(np.int64)


def f2u(x):
    """Truncating, saturating float -> uint32 (NaN and negatives -> 0), as int64 values."""
    x = np.asarray(x, dtype=np.float64)
    with np.errstate(invalid="ignore"):
        t = np.trunc(np.clip(np.nan_to_num(x, nan=0.0), 0.0, 2.0 ** 32 - 1))
    return t.astype(np.int64)


def h(x):
    """Round to float16 (nearest even)."""
    with np.errstate(over="ignore", invalid="ignore"):
        return f32(x).astype(np.float16)


def bits32(x):
    return np.ascontiguousarray(f32(x)).view(np.uint32)


def from_bits32(u):
    return np.ascontiguousarray(np.asarray(u, dtype=np.uint32)).view(np.float32)


def h16bits(x):
    return np.ascontiguousarray(h(x)).view(np.uint16).astype(np.uint32)


def from_h16bits(u):
    return np.ascontiguousarray(np.asarray(u, dtype=np.uint16)).view(np.float16).astype(np.float32)


def ex2(y):
    """2^y, deterministic: n = floor(y), f = y - n (exact), p = the polynomial at f (fma), then p * 2^n with an exact
    power of two; 0 for n < -126 (flush to zero), +inf for n > 127, NaN for NaN."""
    y = f32(y)
    with np.errstate(invalid="ignore", over="ignore"):
        n = np.floor(y).astype(np.float32)
        f = (y - n).astype(np.float32)
        p = np.full(np.shape(y), EX2_COEFFS[0], dtype=np.float32)
        for c in EX2_COEFFS[1:]:
            p = fma(p, f, c)
        ni = f2i(n)
        scale = ((np.clip(ni, -126, 127) + 127) << 23).astype(np.uint32).view(np.float32)
        r = (p * scale).astype(np.float32)
    r = np.where(ni < -126, F32(0.0), r)
    r = np.where(ni > 127, INF, r)
    return np.where(np.isnan(y), y, r).astype(np.float32)


def sigmoid_pair(a, b):
    """sigma(a - b) as the description computes it: e = 2^(|a - b| * -log2(e)), (a > b ? 1 : e) / (e + 1)."""
    a, b = f32(a), f32(b)
    with np.errstate(invalid="ignore", over="ignore"):
        z = (a - b).astype(np.float32)
        e = ex2((np.abs(z) * NEG_LOG2E).astype(np.float32))
        num = np.where(a > b, F32(1.0), e).astype(np.float32)
        return adiv(num, (e + F32(1.0)).astype(np.float32))


def centres(n):
    """u = sat((x + 0.5) / n), approximate division, for x = 0 .. n - 1."""
    return sat((np.arange(n, dtype=np.float32) + F32(0.5)) * rcp(F32(n)))


def luma(c):
    """sat(fma(b, 0.114, fma(r, 0.299, g * 0.587))) of c [..., >= 3]."""
    c = f32(c)
    with np.errstate(invalid="ignore", over="ignore"):
        return sat(fma(c[..., 2], LUMA_B, fma(c[..., 0], LUMA_R, (c[..., 1] * LUMA_G).astype(np.float32))))


# ---- regions and texture reads ------------------------------------------------------------------------------------

def _round_dir(p, up):
    """The float32 next to the rational p in the given direction (p itself when representable)."""
    f = np.float32(float(p))
    while Fraction(float(f)) < p if up else Fraction(float(f)) > p:
        f = np.nextafter(f, np.float32(np.inf if up else -np.inf))
    while True:
        g = np.nextafter(f, np.float32(-np.inf if up else np.inf))
        if (Fraction(float(g)) >= p) if up else (Fraction(float(g)) <= p):
            f = g
        else:
            break
    return np.float32(f)


class Region:
    """An application texture's region: texture size, base offset and extent. rx = 1 / texture width (one IEEE
    single division), and the clamp bounds [(b + 0.5) rx rounded up, (b + e - 0.5) rx rounded down]."""

    def __init__(self, tex_w, tex_h, bx=0, by=0, ew=None, eh=None):
        self.tex_w, self.tex_h = int(tex_w), int(tex_h)
        self.bx, self.by = int(bx), int(by)
        self.ew = int(tex_w - bx if ew is None else ew)
        self.eh = int(tex_h - by if eh is None else eh)
        if self.bx + self.ew > self.tex_w or self.by + self.eh > self.tex_h or self.ew < 1 or self.eh < 1:
            raise ValueError("region outside its texture")
        self.rx = F32(1.0) / F32(self.tex_w)
        self.ry = F32(1.0) / F32(self.tex_h)
        self.lox = _round_dir(Fraction(float(F32(self.bx + 0.5))) * Fraction(float(self.rx)), True)
        self.hix = _round_dir(Fraction(float(F32(self.bx + self.ew - 0.5))) * Fraction(float(self.rx)), False)
        self.loy = _round_dir(Fraction(float(F32(self.by + 0.5))) * Fraction(float(self.ry)), True)
        self.hiy = _round_dir(Fraction(float(F32(self.by + self.eh - 0.5))) * Fraction(float(self.ry)), False)

    def floats(self):
        """The per-call values a kernel takes (as float32): ew, eh, bx, by, rx, ry, lox, hix, loy, hiy."""
        return [F32(self.ew), F32(self.eh), F32(self.bx), F32(self.by), self.rx, self.ry, self.lox, self.hix,
                self.loy, self.hiy]

    def coords(self, u, v, bias=False):
        """Normalised texture coordinates of the region point (u, v)."""
        with np.errstate(invalid="ignore", over="ignore"):
            tx = fma(u, F32(self.ew), F32(self.bx))
            ty = fma(v, F32(self.eh), F32(self.by))
            if bias:
                X, Y = fma(self.rx, tx, -BIAS), fma(self.ry, ty, -BIAS)
            else:
                X, Y = (self.rx * tx).astype(np.float32), (self.ry * ty).astype(np.float32)
        return med3(X, self.lox, self.hix), med3(Y, self.loy, self.hiy)


def _texel(img, iy, ix):
    v = img[iy, ix]
    return v.astype(np.float32) if img.dtype in (np.float16, np.float32) else v


def read_int(img, ix, iy):
    """Integer read at (ix, iy), clamped to the image."""
    hh, ww = img.shape[:2]
    return _texel(img, np.clip(iy, 0, hh - 1), np.clip(ix, 0, ww - 1))


def nearest_index(X, n):
    with np.errstate(invalid="ignore", over="ignore"):
        return np.clip(f2i(np.floor((f32(X) * F32(n)).astype(np.float32))), 0, n - 1)


def read_nearest(img, X, Y):
    """Nearest read at normalised (X, Y): texel clamp(floor(X * W)), clamp(floor(Y * H))."""
    hh, ww = img.shape[:2]
    return _texel(img, nearest_index(Y, hh), nearest_index(X, ww))


def read_linear(img, X, Y):
    """Bilinear read at normalised (X, Y): p = fma(X, W, -0.5), i0 = clamp(floor(p)), weight p - floor(p), clamped
    neighbours; rows blended first (fma(a, t1 - t0, t0)), then the two rows."""
    hh, ww = img.shape[:2]
    X, Y = f32(X), f32(Y)
    with np.errstate(invalid="ignore", over="ignore"):
        px, py = fma(X, F32(ww), F32(-0.5)), fma(Y, F32(hh), F32(-0.5))
        fx, fy = np.floor(px).astype(np.float32), np.floor(py).astype(np.float32)
        ax, ay = (px - fx).astype(np.float32), (py - fy).astype(np.float32)
        rx, ry = np.clip(f2i(fx), -1, ww - 1), np.clip(f2i(fy), -1, hh - 1)
        x0, x1 = np.maximum(rx, 0), np.minimum(rx + 1, ww - 1)
        y0, y1 = np.maximum(ry, 0), np.minimum(ry + 1, hh - 1)
        t00, t10 = img[y0, x0].astype(np.float32), img[y0, x1].astype(np.float32)
        t01, t11 = img[y1, x0].astype(np.float32), img[y1, x1].astype(np.float32)
        if img.ndim == 3:
            ax, ay = ax[..., None], ay[..., None]
        r0 = fma(ax, (t10 - t00).astype(np.float32), t00)
        r1 = fma(ax, (t11 - t01).astype(np.float32), t01)
        return fma(ay, (r1 - r0).astype(np.float32), r0)


def grid_uv(w, hgt):
    """u [1, w], v [hgt, 1] of a stage's own grid."""
    return centres(w)[None, :], centres(hgt)[:, None]


# ---- menu detection (once per game frame) -----------------------------------------------------------------------

def detect_points(x0, y0, wr, hr):
    """The 64 x 32 point grid: x = x0 + floor((wr - 1) j / 63), y = y0 + floor((hr - 1) k / 31) (unsigned integers)."""
    j, k = np.arange(64, dtype=np.int64), np.arange(32, dtype=np.int64)
    xs = x0 + ((wr - 1) * j) // 63
    ys = y0 + ((hr - 1) * k) // 31
    return np.meshgrid(xs, ys)


def detect_count_alpha(alpha_img, region, channel, threshold=F32(0.7)):
    """Number of grid points with UI alpha > threshold (strict); alpha_img [h, w, c], channel 0 (1-channel UI alpha) or
    3 (the 4-channel UI input's alpha)."""
    X, Y = detect_points(region.bx, region.by, region.ew, region.eh)
    a = alpha_img[Y, X, channel].astype(np.float32)
    return int(np.sum(a > F32(threshold)))


def hudless_opacity(c, hc):
    """d = max over r, g, b of max((H - C) / (H + 1e-10), (H - C) / ((H - 1) - 1e-10)), approximate divisions."""
    c, hc = f32(c)[..., :3], f32(hc)[..., :3]
    with np.errstate(invalid="ignore", over="ignore", divide="ignore"):
        dlt = (hc - c).astype(np.float32)
        a = adiv(dlt, (hc + F32(1e-10)).astype(np.float32))
        b = adiv(dlt, ((hc - F32(1.0)).astype(np.float32) - F32(1e-10)).astype(np.float32))
        m = fmax(a, b)
        return fmax(fmax(m[..., 0], m[..., 1]), m[..., 2])


def detect_count_hudless(colour, creg, hudless, hreg):
    """Number of grid points whose hudless opacity is > 0.7. The hudless is read at the same integer texel when both
    regions share their base offset, else with a bilinear region read at u = sat((x - x0 + 0.5) / wr)."""
    X, Y = detect_points(creg.bx, creg.by, creg.ew, creg.eh)
    c = colour[Y, X].astype(np.float32)
    if (hreg.bx, hreg.by) == (creg.bx, creg.by):
        hc = read_int(hudless, X, Y)
    else:
        u = sat(adiv((X - creg.bx).astype(np.float32) + F32(0.5), F32(creg.ew)))
        v = sat(adiv((Y - creg.by).astype(np.float32) + F32(0.5), F32(creg.eh)))
        hx, hy = hreg.coords(u, v)
        hc = read_linear(hudless, hx, hy)
    return int(np.sum(hudless_opacity(c, hc) > F32(0.7)))


def detect_decide(kind, count, fraction=F32(0.7)):
    """Detected: alpha detector count > fraction * 2048, hudless detector count > 0.3 * 2048 (single precision)."""
    f = F32(fraction) if kind == "alpha" else F32(0.3)
    return bool(F32(count) > (f * F32(2048.0)).astype(np.float32))


def detect_update(detres, kind, detected, hysteresis_off):
    """New DetRes (4 x u32) from the previous one: [0] latch, [1] detected, [2] (alpha detector: detected, 0 with
    hysteresis off; hudless: 0), [3] 0."""
    prev1 = int(detres[1])
    latch = 0xFFFFFFFF if detected else (0 if hysteresis_off else (0xFFFFFFFF if prev1 != 0 else 0))
    d1 = 1 if detected else 0
    d2 = (0 if hysteresis_off else d1) if kind == "alpha" else 0
    return np.array([latch, d1, d2, 0], dtype=np.uint32)


# ---- R3: downsample, UI estimate, colour alpha -------------------------------------------------------------------

def r3_case(flags):
    """The one R3 case that applies: "U" (F1), "A" (UI alpha active), "H" (hudless active) or "0"."""
    if flags["F1"]:
        return "U"
    if flags["A"]:
        return "A"
    return "H" if flags["Hd"] else "0"


def downsample_ui(colour, creg, hudless, hreg, alpha_img, areg, alpha_channel, cp, flags, d, wd, hd, out):
    """R3. colour / hudless / alpha_img: application textures (hudless, alpha_img None when not supplied) with their
    Regions; cp: previous colour image [Ho, Wo, 4] float16. flags: dict F1, A, Hd, L1, L2. out: dict of arrays to
    write: Cc [Ho, Wo, 4], Uc (case U), Lc (L2 in cases A / H), Dc [hd, wd, 4], Ld [hd, wd, 1], Um [hd, wd, 1] (cases
    A / H). Returns the case letter."""
    ho, wo = cp.shape[:2]
    u, v = grid_uv(wo, ho)
    u, v = np.broadcast_to(u, (ho, wo)), np.broadcast_to(v, (ho, wo))
    c = colour[creg.by:creg.by + ho, creg.bx:creg.bx + wo].astype(np.float32)
    F1, A, Hd, L1, L2 = (bool(flags[k]) for k in ("F1", "A", "Hd", "L1", "L2"))
    case = r3_case(flags)
    hc = None
    if case in ("U", "H") or (case == "A" and L2):
        X, Y = hreg.coords(u, v)
        hc = read_linear(hudless, X, Y) if L2 else read_nearest(hudless, X, Y)
    al = None
    if case in ("U", "A"):
        X, Y = areg.coords(u, v)
        al = read_nearest(alpha_img, X, Y)[..., alpha_channel]
    flag = None
    with np.errstate(invalid="ignore", over="ignore"):
        if case == "U":
            if L1:
                ap = np.where(finite(al), sat(al), F32(1.0)).astype(np.float32)
            else:
                ap = al
            U = fma(hc[..., :3], (ap - F32(1.0)).astype(np.float32)[..., None], c[..., :3])
            nf = ~np.all(finite(U), axis=-1)
            mn = fmin(fmin(U[..., 0], U[..., 1]), U[..., 2])
            low = (U < (c[..., :3] - (hc[..., :3] + hc[..., :3])).astype(np.float32)).any(axis=-1)
            bad = (ap == F32(1.0)) | nf | (~nf & (mn <= F32(-1.0))) | low
            zero = L1 & (ap == F32(0.0))
            uc = np.concatenate([U, ap[..., None]], axis=-1)
            uc = np.where(bad[..., None], np.concatenate([c[..., :3], np.ones((ho, wo, 1), np.float32)], -1), uc)
            uc = np.where(zero[..., None], F32(0.0), uc)
            out["Uc"][...] = h(uc)
            cc = np.concatenate([hc[..., :3], ap[..., None]], axis=-1)
            s = hc[..., :3]
        elif case == "A":
            flag = al > UI_ALPHA_THRESHOLD
            a = np.where(flag, F32(1.0), F32(0.0)) if L2 else al
            cc = np.concatenate([c[..., :3], f32(a)[..., None]], axis=-1)
            if L2:
                out["Lc"][...] = h(np.concatenate([hc[..., :3], np.zeros((ho, wo, 1), np.float32)], -1))
            s = c[..., :3]
        elif case == "H":
            dd = np.abs((c[..., :3] - hc[..., :3]).astype(np.float32))
            flag = fmax(fmax(dd[..., 0], dd[..., 1]), dd[..., 2]) > UI_DIFF_THRESHOLD
            a = np.where(flag, F32(1.0), F32(0.0)) if L2 else np.zeros((ho, wo), np.float32)
            cc = np.concatenate([c[..., :3], f32(a)[..., None]], axis=-1)
            if L2:
                out["Lc"][...] = h(np.concatenate([hc[..., :3], np.zeros((ho, wo, 1), np.float32)], -1))
            s = c[..., :3]
        else:
            cc = np.concatenate([c[..., :3], np.zeros((ho, wo, 1), np.float32)], -1)
            s = c[..., :3]
        cc16 = h(cc)
        out["Cc"][...] = cc16
        diff = np.abs((luma(cc16.astype(np.float32)) - luma(cp.astype(np.float32))).astype(np.float32))
    # block reduction to the net1 grid (pixels outside the frame contribute zeros)
    hb, wb = -(-ho // d) * d, -(-wo // d) * d
    sp = np.zeros((hb, wb, 3), np.float32)
    sp[:ho, :wo] = s
    dp = np.zeros((hb, wb), np.float32)
    dp[:ho, :wo] = diff
    fp = np.zeros((hb, wb), bool)
    if flag is not None:
        fp[:ho, :wo] = flag

    def S(a, i, j):
        return a[j::d, i::d]
    with np.errstate(invalid="ignore", over="ignore"):
        if d == 2:
            dc = (((S(sp, 0, 0) + S(sp, 1, 0)) + (S(sp, 0, 1) + S(sp, 1, 1))) * F32(0.25)).astype(np.float32)
            ld = fmin(fmin(S(dp, 0, 0), S(dp, 1, 0)), fmin(S(dp, 0, 1), S(dp, 1, 1)))
        elif d == 4:
            a = {(i, j): (S(sp, i, j) + S(sp, i + 1, j)).astype(np.float32) for i in (0, 2) for j in range(4)}
            b = {(i, j): (a[i, j] + a[i, j + 1]).astype(np.float32) for i in (0, 2) for j in (0, 2)}
            cs = {j: (b[0, j] + b[2, j]).astype(np.float32) for j in (0, 2)}
            dc = ((cs[0] + cs[2]).astype(np.float32) * F32(1.0 / 16.0)).astype(np.float32)
            m1 = {(i, j): fmin(S(dp, i, j), S(dp, i + 1, j)) for i in (0, 2) for j in range(4)}
            m2 = {(i, j): fmin(m1[i, j], m1[i, j + 1]) for i in (0, 2) for j in (0, 2)}
            s1 = {j: (m2[0, j] + m2[2, j]).astype(np.float32) for j in (0, 2)}
            ld = ((s1[0] + s1[2]).astype(np.float32) * F32(0.25)).astype(np.float32)
        else:
            raise ValueError("d must be 2 or 4")
    um = np.zeros(dc.shape[:2], bool)
    for j in range(d):
        for i in range(d):
            um |= S(fp, i, j)
    out["Dc"][...] = h(np.concatenate([dc[:hd, :wd], np.zeros((hd, wd, 1), np.float32)], -1))
    out["Ld"][..., 0] = h(ld[:hd, :wd])
    if case in ("A", "H"):
        out["Um"][..., 0] = h(np.where(um[:hd, :wd], F32(1.0), F32(0.0)))
    return case


# ---- R4: motion and depth ----------------------------------------------------------------------------------------

def depth_key(z, s_k, inverted):
    """s_k * rcp(max(D, 1e-7)), D = z if the depth is inverted, else 1 - z."""
    z = f32(z)
    dd = z if inverted else (F32(1.0) - z).astype(np.float32)
    with np.errstate(invalid="ignore", over="ignore"):
        return (F32(s_k) * rcp(fmax(dd, KEY_FLOOR))).astype(np.float32)


def pack_depth(z, ox_pos, oy_pos, centre):
    """The 32-bit packed depth of z = sat(chosen depth) and the chosen sample's offset signs."""
    B = bits32(z).astype(np.uint64)
    oy = oy_pos.astype(np.uint64) << np.uint64(31)
    ox30 = ox_pos.astype(np.uint64) << np.uint64(30)
    b29 = (B >> np.uint64(29)) & np.uint64(1)
    c = (b29 << np.uint64(31)) | (B & np.uint64(0x1FFFFFFF))
    n1 = oy | ox30 | (B & np.uint64(0x3FFFFFFF))
    n0 = oy | np.uint64(1 << 30) | (B & np.uint64(0x1FFFFFFE)) | ox_pos.astype(np.uint64)
    return np.where(centre, c, np.where(b29 == 1, n1, n0)).astype(np.uint32)


def unpack_depth(Z, step_x, step_y):
    """(z, offset x, offset y) of a packed depth word; step_x / step_y: the reader's one-depth-texel steps."""
    Z = np.asarray(Z, dtype=np.uint64)
    b31, b30, b29, b0 = ((Z >> np.uint64(k)) & np.uint64(1) for k in (31, 30, 29, 0))
    centre = (b30 == 0) & (b29 == 0)
    low = Z & np.uint64(0x1FFFFFFF)
    zb = np.where(centre, low | (b31 << np.uint64(29)), low | (b29 << np.uint64(29))).astype(np.uint32)
    xpos = np.where(b29 == 1, b30, b0) == 1
    ox = np.where(centre, F32(0.0), np.where(xpos, F32(step_x), F32(-step_x))).astype(np.float32)
    oy = np.where(centre, F32(0.0), np.where(b31 == 1, F32(step_y), F32(-step_y))).astype(np.float32)
    return from_bits32(zb), ox, oy


def depth_steps(dreg):
    """One depth texel in normalised units, host IEEE division: (1 / extent width, 1 / extent height)."""
    return F32(1.0) / F32(dreg.ew), F32(1.0) / F32(dreg.eh)


def mvec_process(depth, dreg, motion, mreg, w1, h1, inverted, s_k, scale_x, scale_y, out):
    """R4. depth [h, w, 1] float32, motion [h, w, 2] float32 (application textures). Writes out["Zc"] [h1, w1] uint32
    and out["Mv"] [h1, w1, 2] float32. scale_x / scale_y: the motion scales (1.0 when not supplied)."""
    u, v = grid_uv(w1, h1)
    u, v = np.broadcast_to(u, (h1, w1)), np.broadcast_to(v, (h1, w1))
    dx, dy = rcp(F32(dreg.ew)), rcp(F32(dreg.eh))
    offs = [(F32(0), F32(0)), (-dx, -dy), (dx, -dy), (-dx, dy), (dx, dy)]
    with np.errstate(invalid="ignore", over="ignore"):
        samples = []
        for ox, oy in offs:
            X, Y = dreg.coords((u + ox).astype(np.float32), (v + oy).astype(np.float32), bias=True)
            samples.append(read_nearest(depth, X, Y)[..., 0])
        b = samples[0]
        kb = depth_key(b, s_k, inverted)
        sel = np.zeros((h1, w1), np.int64)
        for i in range(1, 5):
            ki = depth_key(samples[i], s_k, inverted)
            take = ki < kb                        # NaN keeps the current choice
            b = np.where(take, samples[i], b)
            kb = np.where(take, ki, kb)
            sel = np.where(take, i, sel)
        z = sat(b)
        ox = np.choose(sel, [o[0] for o in offs]).astype(np.float32)
        oy = np.choose(sel, [o[1] for o in offs]).astype(np.float32)
        out["Zc"][...] = pack_depth(z, ox > 0, oy > 0, sel == 0)
        X, Y = mreg.coords((u + ox).astype(np.float32), (v + oy).astype(np.float32), bias=True)
        m = read_nearest(motion, X, Y)
        sx = F32(scale_x) / F32(mreg.ew)
        sy = F32(scale_y) / F32(mreg.eh)
        out["Mv"][..., 0] = (m[..., 0] * sx).astype(np.float32)
        out["Mv"][..., 1] = (m[..., 1] * sy).astype(np.float32)


# ---- R6: previous-to-current chain -------------------------------------------------------------------------------

def thresholds(ho):
    """T1 = Ho^2 / 900, T2 = Ho^2 / 10000 (host IEEE single divisions)."""
    q = F32(int(ho) * int(ho))
    return q / F32(900.0), q / F32(10000.0)


def camera(u, v, z, M, w1, h1):
    """NDC of (u, v) and the camera motion in pixels with the 4 x 4 matrix M (16 floats as supplied); returns
    (cx, cy, w)."""
    M = [F32(m) for m in np.asarray(M, np.float32).reshape(-1)]
    with np.errstate(invalid="ignore", over="ignore"):
        nx = fma(u, F32(2.0), F32(-1.0))
        ny = (F32(1.0) - (v + v).astype(np.float32)).astype(np.float32)
        xp = (M[12] + fma(M[8], z, fma(ny, M[4], (M[0] * nx).astype(np.float32)))).astype(np.float32)
        yp = (M[13] + fma(M[9], z, fma(ny, M[5], (M[1] * nx).astype(np.float32)))).astype(np.float32)
        w = (M[15] + fma(M[11], z, fma(ny, M[7], (M[3] * nx).astype(np.float32)))).astype(np.float32)
        aw = np.abs(w)
        cx = ((nx - adiv(xp, aw)).astype(np.float32) * (F32(w1) * F32(-0.5))).astype(np.float32)
        cy = ((ny - adiv(yp, aw)).astype(np.float32) * (F32(h1) * F32(0.5))).astype(np.float32)
    bad = ~(w > F32(0.0))
    return np.where(bad, SENTINEL_CAM, cx), np.where(bad, SENTINEL_CAM, cy), w


def sq(x, y):
    """fma(x, x, y * y)."""
    with np.errstate(invalid="ignore", over="ignore"):
        return fma(x, x, (f32(y) * f32(y)).astype(np.float32))


def sgn(x):
    return np.where(f32(x) > F32(0.0), 1, -1).astype(np.int64)


def p2c_scatter(Zc, Mv, M, ho, s_k, inverted, steps, S, Kc):
    """Previous-to-current scatter (signed 64-bit minimum into S [h1 * w1] int64, cleared beforehand) and the camera
    mask Kc [h1, w1, 1] float16 (0 = camera-consistent). steps: the depth reader's one-texel steps."""
    h1, w1 = Zc.shape[:2]
    u, v = grid_uv(w1, h1)
    u, v = np.broadcast_to(u, (h1, w1)), np.broadcast_to(v, (h1, w1))
    T1, T2 = thresholds(ho)
    z, ox, oy = unpack_depth(Zc, *steps)
    with np.errstate(invalid="ignore", over="ignore"):
        up, vp = (u + ox).astype(np.float32), (v + oy).astype(np.float32)
        kap = depth_key(z, s_k, inverted)
        cx, cy, w = camera(up, vp, z, M, w1, h1)
        mx = (Mv[..., 0] * F32(w1)).astype(np.float32)
        my = (Mv[..., 1] * F32(h1)).astype(np.float32)
        m2 = sq(mx, my)
        dd = sq((mx - cx).astype(np.float32), (my - cy).astype(np.float32))
        cons = ~(w <= F32(0.0)) & (dd <= fmax(adiv(m2, T1), F32(1.0)))
        Kc[..., 0] = h(np.where(cons, F32(0.0), F32(1.0)))
        key = np.where(cons, (kap + ((kap * w).astype(np.float32) - kap).astype(np.float32)).astype(np.float32), kap)
        hi = (bits32(key).astype(np.uint64) & np.uint64(0xFFFFFFFE)) | cons.astype(np.uint64)
        lo = (h16bits(mx).astype(np.uint64) << np.uint64(16)) | h16bits(my).astype(np.uint64)
        pay = ((hi << np.uint64(32)) | lo).view(np.int64)
        thr2 = fmax(adiv(m2, T2), F32(1.0))
        yy, xx = np.mgrid[0:h1, 0:w1]
        sx, sy = sgn(mx), sgn(my)
        bx, by = xx + f2i(mx), yy + f2i(my)
        flat = S.reshape(-1)
        for ex, ey in ((0, 0), (0, 1), (1, 0), (1, 1)):
            tx, ty = bx + ex * sx, by + ey * sy
            ok = (tx >= 0) & (tx < w1) & (ty >= 0) & (ty < h1)
            if (ex, ey) != (0, 0):
                nb = read_int(Mv, xx + ex * sx, yy + ey * sy)
                ndx = ((nb[..., 0] * F32(w1)).astype(np.float32) - mx).astype(np.float32)
                ndy = ((nb[..., 1] * F32(h1)).astype(np.float32) - my).astype(np.float32)
                ok &= sq(ndx, ndy) < thr2
            np.minimum.at(flat, (ty * w1 + tx)[ok], pay[ok])


def p2c_unpack(S, Zp, Kp, Mv, M2, s_k, inverted, steps, kmax, tau, out):
    """Unpack of S on the previous grid: out["F"] [h1, w1, 2] float16 (pixels, previous -> current), out["E"]
    [h1, w1, 1] float16 (1 = empty), out["Fl"] [h1, w1] uint32 (bit 0 dynamic, bit 1 dynamic and mismatch, bit 2
    empty) and out["Vf"] [h1, w1, 2] float32 (the vector before rounding, the pull's input)."""
    h1, w1 = Zp.shape[:2]
    u, v = grid_uv(w1, h1)
    u, v = np.broadcast_to(u, (h1, w1)), np.broadcast_to(v, (h1, w1))
    e64 = S.reshape(h1, w1)
    empty = e64 == EMPTY64
    ue = e64.view(np.uint64)
    hi = (ue >> np.uint64(32)) & np.uint64(0xFFFFFFFF)
    lo = ue & np.uint64(0xFFFFFFFF)
    sx = np.where(empty, F32(0.0), from_h16bits((lo >> np.uint64(16)).astype(np.uint16))).astype(np.float32)
    sy = np.where(empty, F32(0.0), from_h16bits((lo & np.uint64(0xFFFF)).astype(np.uint16))).astype(np.float32)
    keyE = from_bits32((hi & np.uint64(0xFFFFFFFE)).astype(np.uint32))
    cbit = (hi & np.uint64(1)) == 1
    z, ox, oy = unpack_depth(Zp, *steps)
    kp = Kp[..., 0].astype(np.float32)
    with np.errstate(invalid="ignore", over="ignore"):
        cx, cy, w = camera((u + ox).astype(np.float32), (v + oy).astype(np.float32), z, M2, w1, h1)
        kap = depth_key(z, s_k, inverted)
        kap = np.where(kp == F32(0.0), (kap * w).astype(np.float32), kap)
        kmax, tau = F32(kmax), F32(tau)
        kE = np.where(keyE <= F32(0.0), F32(0.0), np.where(kmax <= keyE, kmax, keyE)).astype(np.float32)
        kP = np.where(kap <= F32(0.0), F32(0.0), np.where(kap >= kmax, kmax, kap)).astype(np.float32)
        mismatch = empty | (np.abs((kE - kP).astype(np.float32)) > tau)
        gu = (u + adiv(cx, F32(w1))).astype(np.float32)
        gv = (v + adiv(cy, F32(h1))).astype(np.float32)
        g = read_nearest(Mv, gu, gv)
        rx = fma(g[..., 0], F32(w1), cx)
        ry = fma(g[..., 1], F32(h1), cy)
        dynamic = ~(kp == F32(0.0))
        use_cam = cbit | mismatch | (sq(rx, ry) < F32(0.25))
        vx = np.where(dynamic, -sx, np.where(use_cam, cx, -sx)).astype(np.float32)
        vy = np.where(dynamic, -sy, np.where(use_cam, cy, -sy)).astype(np.float32)
    out["F"][..., 0], out["F"][..., 1] = h(vx), h(vy)
    out["E"][..., 0] = h(np.where(empty, F32(1.0), F32(0.0)))
    out["Fl"][...] = (dynamic.astype(np.uint32) | ((dynamic & mismatch).astype(np.uint32) << 1) |
                      (empty.astype(np.uint32) << 2))
    out["Vf"][..., 0], out["Vf"][..., 1] = vx, vy


def p2c_inpaint(Fl, G):
    """The hole rule: inpaint(p) = each of the nine 3 x 3 windows containing p holds a pixel with flag bit 1 (outside
    the grid = 0); w(p) = dynamic and not inpaint; G [h1, w1] uint32 = (w ? 2 : 0) | (inpaint ? 1 : 0)."""
    h1, w1 = Fl.shape
    b1 = ((Fl >> 1) & 1).astype(bool)
    pad = np.zeros((h1 + 4, w1 + 4), bool)
    pad[2:-2, 2:-2] = b1
    win = np.zeros((h1 + 2, w1 + 2), bool)            # window centred at (x - 1 .. x + w1) -> index + 1
    for j in range(3):
        for i in range(3):
            win |= pad[j:j + h1 + 2, i:i + w1 + 2]
    inp = np.ones((h1, w1), bool)
    for j in range(3):
        for i in range(3):
            inp &= win[j:j + h1, i:i + w1]
    dyn = (Fl & 1).astype(bool)
    wgt = dyn & ~inp
    G[...] = (wgt.astype(np.uint32) << 1) | inp.astype(np.uint32)


def pyramid_levels(w, hgt):
    """Halving chain of a grid: each level ceil(previous / 2), until at least 9 levels exist and both extents are
    <= 1, at most 12. Returns [(w, h), ...] with level 0 first."""
    lv = [(int(w), int(hgt))]
    while len(lv) < 12 and not (len(lv) >= 9 and lv[-1][0] <= 1 and lv[-1][1] <= 1):
        a, b = lv[-1]
        lv.append((-(-a // 2), -(-b // 2)))
    return lv


def pull_top(levels):
    """The coarsest pull level: 8 when the pyramid has more than 9 levels, else 4 (capped by the level count)."""
    return min(8 if len(levels) > 9 else 4, len(levels) - 1)


def _tree(a):
    """2 x 2 pairwise sums (x first, then y) of a [h, w, c] (out-of-grid = 0) -> [ceil(h / 2), ceil(w / 2), c]."""
    hh, ww = a.shape[:2]
    p = np.zeros((hh + (hh & 1), ww + (ww & 1)) + a.shape[2:], np.float32)
    p[:hh, :ww] = a
    with np.errstate(invalid="ignore", over="ignore"):
        return ((p[0::2, 0::2] + p[0::2, 1::2]).astype(np.float32) +
                (p[1::2, 0::2] + p[1::2, 1::2]).astype(np.float32)).astype(np.float32)


def _element(sums, n):
    with np.errstate(invalid="ignore", over="ignore"):
        den = fmax(sums[..., 2], WEIGHT_FLOOR)
        return h(np.stack([adiv(sums[..., 0], den), adiv(sums[..., 1], den), adiv(sums[..., 2], F32(n))], -1))


def p2c_pull(G, Vf, top):
    """Pull levels 1 .. top: [h_l, w_l, 3] float16 each (mean vector, mean vector, mean weight). Levels 1 - 4 from the
    base pixels (w = G bit 1, products w * v), levels 5 - 8 from level 4's stored values."""
    w = ((G >> 1) & 1).astype(np.float32)
    with np.errstate(invalid="ignore", over="ignore"):
        cur = np.stack([(w * Vf[..., 0]).astype(np.float32), (w * Vf[..., 1]).astype(np.float32), w], -1)
    levels = {}
    for l in range(1, min(top, 4) + 1):
        cur = _tree(cur)
        levels[l] = _element(cur, 4 ** l)
    if top > 4:
        e = levels[4].astype(np.float32)
        with np.errstate(invalid="ignore", over="ignore"):
            cur = np.stack([(e[..., 0] * e[..., 2]).astype(np.float32), (e[..., 1] * e[..., 2]).astype(np.float32),
                            e[..., 2]], -1)
        for l in range(5, top + 1):
            cur = _tree(cur)
            levels[l] = _element(cur, 4 ** (l - 4))
    return levels


def push_upsample(parent, hq, wq):
    """(v_up [hq, wq, 2], w_up [hq, wq]) from the filled coarser level parent [hp, wp, 3] (float16)."""
    hp, wp = parent.shape[:2]
    yy, xx = np.mgrid[0:hq, 0:wq]
    px, py = xx // 2, yy // 2
    nx = np.clip(px + np.where(xx & 1, 1, -1), 0, wp - 1)
    ny = np.clip(py + np.where(yy & 1, 1, -1), 0, hp - 1)
    P = parent.astype(np.float32)
    taps = [(P[py, px], F32(0.5625)), (P[py, nx], F32(0.1875)), (P[ny, px], F32(0.1875)), (P[ny, nx], F32(0.0625))]
    with np.errstate(invalid="ignore", over="ignore"):
        ws = np.zeros((hq, wq), np.float32)
        vs = [np.zeros((hq, wq), np.float32), np.zeros((hq, wq), np.float32)]
        for t, k in taps:
            ws = fma(t[..., 2], k, ws)
            for c in (0, 1):
                vs[c] = fma((t[..., c] * t[..., 2]).astype(np.float32), k, vs[c])
        den = med3(ws, WEIGHT_FLOOR, F32(1.0))
        return np.stack([adiv(vs[0], den), adiv(vs[1], den)], -1), ws


def p2c_push(levels, top, G, F):
    """Fills the levels from top - 1 down to 1 (own value where its weight > 0, unordered, else the upsampled value;
    the coarsest level keeps its own values), then F := v_up at every pixel with G bit 0 (inpaint)."""
    for l in range(top - 1, 0, -1):
        own = levels[l]
        hq, wq = own.shape[:2]
        vu, wu = push_upsample(levels[l + 1], hq, wq)
        keep = ~(own[..., 2].astype(np.float32) <= F32(0.0))
        up = h(np.concatenate([vu, wu[..., None]], -1))
        levels[l] = np.where(keep[..., None], own, up)
    hh, ww = F.shape[:2]
    vu, _ = push_upsample(levels[1], hh, ww)
    inp = (G & 1).astype(bool)
    F[...] = np.where(inp[..., None], h(vu), F)


# ---- R7, R8: hole mask and quality image --------------------------------------------------------------------------

def hole_mask(Fimg, wd, hd, Hc):
    """R7 hole mask (no distortion field): Hc [hd, wd, 1] float16 := 1, then 0 at the four targets of each net1-grid pixel
    displaced by F (pixels on the scatter grid, read nearest at the pixel centre, divided by the scatter grid size,
    times the net1 grid size)."""
    h1, w1 = Fimg.shape[:2]
    Hc[...] = F16(1.0)
    u, v = grid_uv(wd, hd)
    u, v = np.broadcast_to(u, (hd, wd)), np.broadcast_to(v, (hd, wd))
    m = read_nearest(Fimg, u, v)
    with np.errstate(invalid="ignore", over="ignore"):
        mx = (adiv(m[..., 0], F32(w1)) * F32(wd)).astype(np.float32)
        my = (adiv(m[..., 1], F32(h1)) * F32(hd)).astype(np.float32)
    yy, xx = np.mgrid[0:hd, 0:wd]
    bx, by = xx + f2i(mx), yy + f2i(my)
    sx, sy = sgn(mx), sgn(my)
    for ex, ey in ((0, 0), (0, 1), (1, 0), (1, 1)):
        tx, ty = bx + ex * sx, by + ey * sy
        ok = (tx >= 0) & (tx < wd) & (ty >= 0) & (ty < hd)
        Hc[ty[ok], tx[ok], 0] = F16(0.0)


def quality_init(Dc, Qc):
    """Qc := (sat(((r + g) + b) / 3), 0), approximate division."""
    c = Dc.astype(np.float32)
    with np.errstate(invalid="ignore", over="ignore"):
        q = sat(adiv(((c[..., 0] + c[..., 1]).astype(np.float32) + c[..., 2]).astype(np.float32), F32(3.0)))
    Qc[..., 0], Qc[..., 1] = h(q), F16(0.0)


# ---- R9: net1 input packer and output unpacker --------------------------------------------------------------------

def warp_error(a, b):
    """sat((|db| + (|dr| + |dg|)) / 3) of d = a - b, approximate division."""
    with np.errstate(invalid="ignore", over="ignore"):
        d = np.abs((f32(a)[..., :3] - f32(b)[..., :3]).astype(np.float32))
        return sat(adiv((d[..., 2] + (d[..., 0] + d[..., 1]).astype(np.float32)).astype(np.float32), F32(3.0)))


def net1_values(Ld, Dc, Dp, Fimg, Mv, E, Hc, Hp, Qp, m):
    """The ten values of every net1 image pixel [10, hd, wd] (float32) and the side outputs Qc (v6, v2) and Om."""
    hd, wd = Dc.shape[:2]
    h1, w1 = Fimg.shape[:2]
    u, v = grid_uv(wd, hd)
    u, v = np.broadcast_to(u, (hd, wd)), np.broadcast_to(v, (hd, wd))
    dc, dp = Dc.astype(np.float32), Dp.astype(np.float32)
    with np.errstate(invalid="ignore", over="ignore"):
        f = read_nearest(Fimg, u, v)
        su = (u + adiv(f[..., 0], F32(w1))).astype(np.float32)
        sv = (v + adiv(f[..., 1], F32(h1))).astype(np.float32)
        v2 = warp_error(read_linear(Dc, su, sv), dp)
        mv = read_nearest(Mv, u, v)
        v6 = warp_error(read_linear(Dp, (u + mv[..., 0]).astype(np.float32), (v + mv[..., 1]).astype(np.float32)), dc)
        e = read_nearest(E, u, v)[..., 0]
        hc = read_nearest(Hc, u, v)[..., 0]
        hp = read_nearest(Hp, u, v)[..., 0]
        mm = m[..., 0].astype(np.float32)
        vals = np.stack([Ld[..., 0].astype(np.float32), luma(dp), v2, Qp[..., 0].astype(np.float32), e, luma(dc), v6,
                         hc, mm, (F32(1.0) - mm).astype(np.float32)])
        om = sat(((e * hp).astype(np.float32) * hc).astype(np.float32))
    return vals, np.stack([v6, v2], -1), om


def net1_pack(vals, W0, H0, pad_l, pad_t):
    """Cells [H0, W0, 48] float16 (HWC): value v of pixel (dx, dy) of cell (X, Y) at channel 4 v + 2 dy + dx, image
    pixel (x, y) = (G1 x - 2 pad_l, G1 y - 2 pad_t); outside the image and channels 40 - 47 zero."""
    _, hd, wd = vals.shape
    g = np.zeros((10, 2 * H0, 2 * W0), np.float32)
    x0, y0 = 2 * pad_l, 2 * pad_t
    hh, ww = min(hd, 2 * H0 - y0), min(wd, 2 * W0 - x0)
    g[:, y0:y0 + hh, x0:x0 + ww] = vals[:, :hh, :ww]
    cells = np.zeros((H0, W0, 48), np.float16)
    for dy in (0, 1):
        for dx in (0, 1):
            cells[:, :, 2 * dy + dx:40:4] = h(g[:, dy::2, dx::2]).transpose(1, 2, 0)
    return cells


def net1_input(Ld, Dc, Dp, Fimg, Mv, E, Hc, Hp, Qp, m, W0, H0, pad_l, pad_t, out):
    """net1 input packer: out["cells"] [H0, W0, 48] float16, out["Qc"] [hd, wd, 2] float16, out["Om"] [hd, wd, 1]. The side
    outputs are written for the image pixels inside the 2 x 2-fold grid (x < 2 (W0 - pad_l), y < 2 (H0 - pad_t))."""
    vals, qc, om = net1_values(Ld, Dc, Dp, Fimg, Mv, E, Hc, Hp, Qp, m)
    out["cells"][...] = net1_pack(vals, W0, H0, pad_l, pad_t)
    hh, ww = min(Dc.shape[0], 2 * (H0 - pad_t)), min(Dc.shape[1], 2 * (W0 - pad_l))
    out["Qc"][:hh, :ww] = h(qc[:hh, :ww])
    out["Om"][:hh, :ww, 0] = h(om[:hh, :ww])
    return vals


def net1_output(o16, Om, Um, Ump, wd, hd, pad_l, pad_t, ui_forcing, out):
    """net1 output unpacker: o16 [H0, W0, 16] float16 (HWC); writes out["NA"], out["NB"] [hd, wd, 2] and out["m"] [hd, wd, 1]
    (float16). ui_forcing: (hudless or UI alpha active) and neither the UI-recomposition path nor Lite flag 1."""
    yy, xx = np.mgrid[0:hd, 0:wd]
    ncx, ncy = max(wd // 2, 1), max(hd // 2, 1)
    cx = pad_l + np.minimum(xx >> 1, ncx - 1)
    cy = pad_t + np.minimum(yy >> 1, ncy - 1)
    q = 2 * (yy & 1) + (xx & 1)
    o = [o16[cy, cx, 4 * i + q].astype(np.float32) for i in range(4)]
    om = Om[..., 0].astype(np.float32)
    with np.errstate(invalid="ignore", over="ignore"):
        o[0] = fma((PUSH - o[0]).astype(np.float32), om, o[0])
        o[2] = fma((PUSH - o[2]).astype(np.float32), om, o[2])
        mm = sigmoid_pair(o[2], o[3])
        if ui_forcing:
            fp = Ump[..., 0].astype(np.float32) > F32(0.99)
            o[0] = np.where(fp, (o[0] - F32(1000.0)).astype(np.float32), o[0])
            o[1] = np.where(fp, (o[1] + F32(1000.0)).astype(np.float32), o[1])
            fc = Um[..., 0].astype(np.float32) > F32(0.99)
            o[2] = np.where(fc, (o[2] - F32(1000.0)).astype(np.float32), o[2])
            o[3] = np.where(fc, (o[3] + F32(1000.0)).astype(np.float32), o[3])
    out["NA"][...] = h(np.stack([o[0], o[1]], -1))
    out["NB"][...] = h(np.stack([o[2], o[3]], -1))
    out["m"][..., 0] = h(mm)


# ---- G2: intermediate-time scatter --------------------------------------------------------------------------------

def interm_scatter(dxy, keys, t2, tau, depth_test, target):
    """One pass of the intermediate-time scatter. dxy [h1, w1, 2]: each pixel's displacement in pixels (already times
    1 - t or t); keys [h1, w1]: depth keys; depth_test: pass A (the depth-and-displacement skip rule) or pass B
    (displacement only). Signed 64-bit minimum into target [h1 * w1] int64."""
    h1, w1 = keys.shape
    yy, xx = np.mgrid[0:h1, 0:w1]
    dx, dy = dxy[..., 0], dxy[..., 1]
    tau = F32(tau)
    with np.errstate(invalid="ignore", over="ignore"):
        kmax = np.full((h1, w1), -INF, np.float32)
        kmin = np.full((h1, w1), INF, np.float32)
        qxmax = np.full((h1, w1), F32(-10000.0), np.float32)
        qymax = qxmax.copy()
        qxmin = np.full((h1, w1), F32(10000.0), np.float32)
        qymin = qxmin.copy()
        for j in (-1, 0, 1):
            for i in (-1, 0, 1):
                kn = read_int(keys, xx + i, yy + j)
                dn = read_int(dxy, xx + i, yy + j)
                kmax, kmin = fmax(kmax, kn), fmin(kmin, kn)
                qx = ((xx + i).astype(np.float32) + F32(0.5) + dn[..., 0]).astype(np.float32)
                qy = ((yy + j).astype(np.float32) + F32(0.5) + dn[..., 1]).astype(np.float32)
                qxmax, qxmin = fmax(qxmax, qx), fmin(qxmin, qx)
                qymax, qymin = fmax(qymax, qy), fmin(qymin, qy)
        qpx = ((xx.astype(np.float32) + F32(0.5)) + dx).astype(np.float32)
        qpy = ((yy.astype(np.float32) + F32(0.5)) + dy).astype(np.float32)
        edge = ((kmax - keys).astype(np.float32) > tau) | ((keys - kmin).astype(np.float32) > tau)

        def radius(qmax, qmin, qp):
            a = fmax(np.abs((qmax - qp).astype(np.float32)), np.abs((qmin - qp).astype(np.float32)))
            return np.clip(f2i(med3(np.ceil((a + F32(0.5)).astype(np.float32)), F32(2.0), F32(5.0))), 2, 5)
        rx = np.where(edge, 2, radius(qxmax, qxmin, qpx))
        ry = np.where(edge, 2, radius(qymax, qymin, qpy))
        hi = bits32(keys).astype(np.uint64)
        lo = (h16bits(adiv(-dx, F32(w1))).astype(np.uint64) << np.uint64(16)) | \
            h16bits(adiv(-dy, F32(h1))).astype(np.uint64)
        pay = ((hi << np.uint64(32)) | lo).view(np.int64)
        thr = fmax(adiv(sq(dx, dy), t2), F32(1.0))
        sx, sy = sgn(dx), sgn(dy)
        bx, by = xx + f2i(dx), yy + f2i(dy)
        flat = target.reshape(-1)
        for j in range(5):
            for i in range(5):
                act = (i < rx) & (j < ry)
                tx, ty = bx + i * sx, by + j * sy
                ok = act & (tx >= 0) & (tx < w1) & (ty >= 0) & (ty < h1)
                nk = read_int(keys, xx + i * sx, yy + j * sy)
                nd = read_int(dxy, xx + i * sx, yy + j * sy)
                dd = sq((nd[..., 0] - dx).astype(np.float32), (nd[..., 1] - dy).astype(np.float32))
                if depth_test:
                    kt = f2u(nk).astype(np.float32)
                    skip = ~(np.abs((kt - keys).astype(np.float32)) < F32(3.0)) & ~(dd < thr)
                    ok &= ~skip
                else:
                    ok &= dd < thr
                np.minimum.at(flat, (ty * w1 + tx)[ok], pay[ok])


def interm_inputs(Mv, Zc, Fimg, Zp, t, s_k, inverted):
    """Displacements and keys of both passes: A from the current frame (Mv, Zc; factor 1 - t), B from the previous
    (F, Zp; factor t). Keys from the centre decode of the packed depth."""
    h1, w1 = Zc.shape[:2]
    t = F32(t)
    omt = (F32(1.0) - t).astype(np.float32)
    with np.errstate(invalid="ignore", over="ignore"):
        da = np.stack([(omt * (Mv[..., 0] * F32(w1)).astype(np.float32)).astype(np.float32),
                       (omt * (Mv[..., 1] * F32(h1)).astype(np.float32)).astype(np.float32)], -1)
        f = Fimg.astype(np.float32)
        db = np.stack([(t * f[..., 0]).astype(np.float32), (t * f[..., 1]).astype(np.float32)], -1)
    za, _, _ = unpack_depth(Zc, F32(0), F32(0))
    zb, _, _ = unpack_depth(Zp, F32(0), F32(0))
    return da, depth_key(za, s_k, inverted), db, depth_key(zb, s_k, inverted)


# ---- G3: blend weights ------------------------------------------------------------------------------------------

def vec_lookup(buf, wm, hm, u, v):
    """(vector [.., 2] float32 in normalised units, empty) of a 64-bit buffer [hm * wm] at index
    rint(fma(u, wm, -0.5) - 2^-23) (clamped)."""
    with np.errstate(invalid="ignore", over="ignore"):
        ix = np.clip(f2i(np.rint((fma(u, F32(wm), F32(-0.5)) + (-BIAS)).astype(np.float32))), 0, wm - 1)
        iy = np.clip(f2i(np.rint((fma(v, F32(hm), F32(-0.5)) + (-BIAS)).astype(np.float32))), 0, hm - 1)
    e = buf.reshape(-1)[iy * wm + ix]
    empty = e == EMPTY64
    lo = e.view(np.uint64) & np.uint64(0xFFFFFFFF)
    x = np.where(empty, F32(0.0), from_h16bits((lo >> np.uint64(16)).astype(np.uint16)))
    y = np.where(empty, F32(0.0), from_h16bits((lo & np.uint64(0xFFFF)).astype(np.uint16)))
    return np.stack([x, y], -1).astype(np.float32), empty


def blend_weights(MA, MB, NA, NB, w1, h1, Wimg):
    """Blend weights: Wimg [h1, w1, 2] float16 = (sigma(A1 - A0), sigma(B1 - B0))."""
    hd, wd = NA.shape[:2]
    u, v = grid_uv(w1, h1)
    u, v = np.broadcast_to(u, (h1, w1)), np.broadcast_to(v, (h1, w1))
    a, _ = vec_lookup(MA, w1, h1, u, v)
    b, _ = vec_lookup(MB, w1, h1, u, v)
    with np.errstate(invalid="ignore", over="ignore"):
        jx = np.clip(f2i(np.rint((fma(u, F32(wd), F32(-0.5)) + (-BIAS)).astype(np.float32))), 0, wd - 1)
        jy = np.clip(f2i(np.rint((fma(v, F32(hd), F32(-0.5)) + (-BIAS)).astype(np.float32))), 0, hd - 1)
        A1 = read_linear(NA, (u + b[..., 0]).astype(np.float32), (v + b[..., 1]).astype(np.float32))[..., 0]
        A0 = NA[jy, jx, 1].astype(np.float32)
        B1 = read_linear(NB, (u + a[..., 0]).astype(np.float32), (v + a[..., 1]).astype(np.float32))[..., 0]
        B0 = NB[jy, jx, 1].astype(np.float32)
    Wimg[..., 0] = h(sigmoid_pair(A1, A0))
    Wimg[..., 1] = h(sigmoid_pair(B1, B0))


# ---- copies, fills, menu output -------------------------------------------------------------------------------------

def copy4(dst, dx0, dy0, src, sx0, sy0, w, hgt):
    """dst[dy0 + y, dx0 + x] := src[sy0 + y, sx0 + x] (point copy, storage formats equal)."""
    dst[dy0:dy0 + hgt, dx0:dx0 + w] = src[sy0:sy0 + hgt, sx0:sx0 + w]


def menu_output(detres, colour, creg, hudless, hreg, alpha_img, areg, alpha_channel, out, ox, oy, wo, ho):
    """Menu output stage over the out extent: DetRes[0] != 0 -> current colour (point copy); else DetRes[2] != 0 ->
    out.rgb = C.rgb + (G.rgb - H.rgb) (1 - a), alpha kept; else nothing."""
    if int(detres[0]) != 0:
        copy4(out, ox, oy, colour, creg.bx, creg.by, wo, ho)
        return "copy"
    if int(detres[2]) == 0:
        return "none"
    u, v = grid_uv(wo, ho)
    u, v = np.broadcast_to(u, (ho, wo)), np.broadcast_to(v, (ho, wo))
    c = colour[creg.by:creg.by + ho, creg.bx:creg.bx + wo].astype(np.float32)
    g = out[oy:oy + ho, ox:ox + wo].astype(np.float32)
    X, Y = hreg.coords(u, v)
    hc = read_linear(hudless, X, Y)
    X, Y = areg.coords(u, v)
    a = read_nearest(alpha_img, X, Y)[..., alpha_channel]
    with np.errstate(invalid="ignore", over="ignore"):
        rgb = (c[..., :3] + ((g[..., :3] - hc[..., :3]).astype(np.float32) *
                             (F32(1.0) - a).astype(np.float32)[..., None]).astype(np.float32)).astype(np.float32)
    out[oy:oy + ho, ox:ox + wo, :3] = h(rgb)
    return "recompose"
