# SPDX-License-Identifier: MIT
"""NumPy reference of page 54b (final review copy, with errata 50-53): output resolve, transparency layer
(colour-before-particles rule of the erratum), alpha upscaling, quarter pooled history, clears.

Inputs are exactly what page 54 hands over: the output-gated image Q (54.7, binary16 RGB, Wh x Hh), the
pooled history PH1 (54.7, RGB10A2 codes), the previous-motion copy PM[1-par] written by this frame's input stage
(54b.3 MV), the head textures P / G (page 55.3), and the exposure block E / Ep (page 58).

Conventions
* Arrays are [y][x][channel]; sizes and origins are (x, y) / (width, height).
* Every f32 operation is one exact rounding with FTZ (reference/f32x.py); exp2 / log2 / rcp are correctly
  rounded (the rule the reference test vectors use, 54b.8). A GPU implementation of the same operation
  sequence that evaluates the three functions correctly rounded is bit-identical to this module; this module
  also fixes one f32 evaluation of the 52.1.4 bilinear sample, which 54b.3 names as its reference.
* RGB10A2 resources are represented as uint16 channel codes [R10, G10, B10, A2] (as reference/output.py).
* Surfaces: Surface(data, format, origin); formats 'f32', 'f16', 'unorm8', 'rgb10a2'.
* NaN payloads and NaN signs are not specified by any page; the comparison helpers treat NaNs as one class.

Nothing here embeds a model constant: every symbol of 54b.7 is supplied by the caller (converter, page 57).
WrongGPU's choices that this module makes beyond the page's text are listed in clean/rr_f/QUESTIONS-54b.md.
"""
from dataclasses import dataclass, field, fields
import operator

import numpy as np

from . import f32x as fx
from .f32x import F32, F64, FMAX, FMINN, HMAX, K6, K9

FORMATS = {"f32": np.dtype(np.float32), "f16": np.dtype(np.float16),
           "unorm8": np.dtype(np.uint8), "rgb10a2": np.dtype(np.uint16)}


# --------------------------------------------------------------------------------------------------------------
# Symbols

@dataclass(frozen=True)
class ResolveSymbols:
    """The symbols of 54b.7 (values: page 57.7.2; K1 derived in D0-R). RW_* binary16, the rest binary32.

    Values must already be exactly representable in their format; no rounding happens here."""
    RW_A: float
    RW_B: float
    RW_C: float
    RW_D: float
    RW_0: float
    K1: float
    K2n: float
    K2x2n: float
    K4: float
    K5: float
    K7: float
    K8: float
    K10: float
    K11: float
    K12: float
    K13: float
    C_HDEC: float
    T_EFL: float
    TBW: float

    def __post_init__(self):
        for f in fields(self):
            v = np.asarray(getattr(self, f.name))
            if v.shape != () or v.dtype.kind not in "fiu":
                raise ValueError(f"{f.name} must be a real scalar")
            dtype = np.float16 if f.name.startswith("RW_") else np.float32
            with np.errstate(all="ignore"):
                r = v.astype(dtype)
            if not np.array_equal(v.astype(F64), r.astype(F64), equal_nan=True):
                raise ValueError(f"{f.name} must be exactly representable as {np.dtype(dtype).name}")
            object.__setattr__(self, f.name, r[()])

    def even_weights(self):
        """w_even[k], k = -2..2 (54b.1 step 2, 54b.7 placement)."""
        return np.array([self.RW_B, self.RW_D, self.RW_0, self.RW_C, self.RW_A], dtype=F64)


# --------------------------------------------------------------------------------------------------------------
# Sizes

def _pair(value, name, positive=False):
    try:
        if len(value) != 2 or any(isinstance(v, (bool, np.bool_)) for v in value):
            raise ValueError
        out = tuple(operator.index(v) for v in value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must contain two integers") from None
    if positive and min(out) <= 0:
        raise ValueError(f"{name} must be positive")
    return out


@dataclass(frozen=True)
class Sizes:
    Wo: int
    Ho: int
    We: int
    He: int
    Wh: int
    Hh: int


def sizes(output_size):
    wo, ho = _pair(output_size, "output_size", positive=True)
    we, he = wo + wo % 2, ho + ho % 2
    return Sizes(wo, ho, we, he, 3 * we // 2, 3 * he // 2)


def pooled_sizes(output_size):
    """(PH1 shape, PQ shape) as (height, width): ceil(Hh/2) x ceil(Wh/2) and ceil(Hh/4) x ceil(Wh/4)."""
    s = sizes(output_size)
    return ((s.Hh + 1) // 2, (s.Wh + 1) // 2), ((s.Hh + 3) // 4, (s.Wh + 3) // 4)


# --------------------------------------------------------------------------------------------------------------
# Surfaces, reads and stores (54b.N conversions)

@dataclass
class Surface:
    """A game or internal resource: data [H][W][C] in its format's storage dtype, plus its sub-rectangle origin."""
    data: np.ndarray
    format: str
    origin: tuple = (0, 0)

    def __post_init__(self):
        if self.format not in FORMATS:
            raise ValueError(f"unsupported format {self.format!r}")
        a = np.asarray(self.data)
        if a.ndim != 3 or a.dtype != FORMATS[self.format]:
            raise ValueError(f"{self.format} surface needs a [H][W][C] {FORMATS[self.format].name} array")
        if self.format == "rgb10a2" and (a.shape[2] != 4 or np.any(a[..., :3] > 1023) or np.any(a[..., 3] > 3)):
            raise ValueError("rgb10a2 surfaces hold four channel codes (10, 10, 10, 2 bits)")
        self.data = a
        ox, oy = _pair(self.origin, "origin")
        if ox < 0 or oy < 0:
            raise ValueError("origins are nonnegative")
        self.origin = (ox, oy)

    @property
    def width(self):
        return self.data.shape[1]

    @property
    def height(self):
        return self.data.shape[0]

    def copy(self):
        return Surface(self.data.copy(), self.format, self.origin)


def read(surface):
    """54b.N read conversion of every texel: float64 holder of f32 values (f16 widened exactly; unorm
    code / (2^n - 1) rounded once to f32; f32 subnormals flushed, as every operation that uses them would)."""
    a = surface.data
    if surface.format == "f32":
        return fx.ftz(a).astype(F64)        # f32 subnormals: every use is an f32 operation, which flushes them
    if surface.format == "f16":
        return a.astype(F64)
    if surface.format == "unorm8":
        return fx.r32(a.astype(F64) / 255.0).astype(F64)
    return fx.r32(a.astype(F64) / np.array([1023.0, 1023.0, 1023.0, 3.0])).astype(F64)


def read_unorm8_as_half(codes):
    """The 8-bit read 'as binary16' of 54b.N / 55.3: code / 255 rounded once to binary16."""
    return fx.f16(np.asarray(codes, dtype=F64) / 255.0)


def encode(values, fmt):
    """54b.N store conversion of f32 values (float64 holders) into the storage dtype of fmt."""
    v = np.asarray(values, dtype=F64)
    if fmt == "f32":
        with np.errstate(all="ignore"):
            return v.astype(np.float32)
    if fmt == "f16":
        return fx.f16(v).astype(np.float16)
    scale = 255.0 if fmt == "unorm8" else np.array([1023.0, 1023.0, 1023.0, 3.0])[: v.shape[-1]]
    c = np.where(np.isnan(v), 0.0, np.clip(v, 0.0, 1.0))
    return np.rint(c * scale).astype(FORMATS[fmt])      # exact product, round half to even


def fetch(tex, xs, ys, fill=0.0):
    """Integer fetch of a float64 texel array at integer index arrays; outside the texture every channel reads
    `fill` (0: the graphics API's rule for out-of-range loads, as 54.N)."""
    h, w = tex.shape[:2]
    xs, ys = np.broadcast_arrays(np.asarray(xs, np.int64), np.asarray(ys, np.int64))
    inside = (xs >= 0) & (ys >= 0) & (xs < w) & (ys < h)
    v = tex[np.clip(ys, 0, h - 1), np.clip(xs, 0, w - 1)]
    return np.where(inside[..., None], v, fill)


def bilinear(tex, u, v):
    """The bilinear sample of page 52.1.4 (the reference named by 54b.3): NaN coordinate -> 0, clamp to
    [0, 1], s = u*W - 0.5 (each rounded), integer part / exact fraction, clamp-to-edge texels, then
    r0 = (1-fx)*T00 + fx*T10, r1 likewise, (1-fy)*r0 + fy*r1, every operation rounded to f32 in this order."""
    h, w = tex.shape[:2]
    u, v = np.broadcast_arrays(np.asarray(u, F64), np.asarray(v, F64))
    u = np.clip(np.where(np.isnan(u), 0.0, u), 0.0, 1.0)
    v = np.clip(np.where(np.isnan(v), 0.0, v), 0.0, 1.0)
    s = fx.sub(fx.mul(u, F32(w)), F32(0.5)).astype(F64)
    t = fx.sub(fx.mul(v, F32(h)), F32(0.5)).astype(F64)
    i0, j0 = np.floor(s), np.floor(t)
    # The fractions are exact (52.1.4): s - i0 is an f32 value for s >= 0; for s in [-0.5, 0) (i0 = -1) it is
    # 1 + s, a multiple of 2^-25 that binary64 holds exactly. Products with it are rounded once.
    fxr, fyr = (s - i0)[..., None], (t - j0)[..., None]
    i0, j0 = i0.astype(np.int64), j0.astype(np.int64)
    xa, xb = np.clip(i0, 0, w - 1), np.clip(i0 + 1, 0, w - 1)
    ya, yb = np.clip(j0, 0, h - 1), np.clip(j0 + 1, 0, h - 1)
    gx, gy = fx.r32(1.0 - fxr).astype(F64), fx.r32(1.0 - fyr).astype(F64)

    def term(weight, value):
        with np.errstate(invalid="ignore", over="ignore"):
            return fx.r32(weight * fx.ftz(value).astype(F64))
    r0 = fx.add(term(gx, tex[ya, xa]), term(fxr, tex[ya, xb]))
    r1 = fx.add(term(gx, tex[yb, xa]), term(fxr, tex[yb, xb]))
    return fx.add(term(gy, r0), term(fyr, r1)).astype(F64)


def store(dest, values, *, width, height):
    """Store values [He][We][C] at dest.origin + (x, y) for 0 <= x < width, 0 <= y < height (54b.0.1);
    positions outside dest are dropped; every other pixel keeps its content. Returns a new Surface."""
    out = dest.copy()
    c = min(out.data.shape[2], values.shape[2])
    bx, by = out.origin
    w = max(0, min(width, out.width - bx))
    h = max(0, min(height, out.height - by))
    if w and h:
        out.data[by:by + h, bx:bx + w, :c] = encode(values[:h, :w, :c], out.format)
    return out


# --------------------------------------------------------------------------------------------------------------
# 54b.1 plain resolve

def _decode(c, sym):
    """54b.1 step 5 up to and including K12 (before the division by E): (exp2((c'*K8)*K9) - 1) * K12."""
    c = np.asarray(c, F64)
    cp = np.where((c > sym.K5) | np.isnan(c), fx.min_nan(c, sym.K10), sym.K5)
    t = fx.mul(fx.mul(cp, sym.K8), K9)
    return fx.mul(fx.sub(fx.exp2(t), 1), sym.K12).astype(F64)


def _five(weights, taps):
    """One pass of 54b.1 steps 3/4: ordered fma16 chain, clamped to the range of its own five taps.
    weights: list of five arrays broadcastable to taps[k]; all binary16 arithmetic."""
    acc = np.zeros(np.shape(taps[0]))
    mn = np.full(np.shape(taps[0]), HMAX)
    mx = np.full(np.shape(taps[0]), -HMAX)
    for w, t in zip(weights, taps):
        acc = fx.fma16(w, t, acc)
        mn, mx = fx.fmin(mn, t), fx.fmax(mx, t)
    return fx.fmin(fx.fmax(acc, mn), mx)


def resample(Q, sz, sym):
    """54b.1 steps 1-4 over the We x He grid: binary16 RGB values v (float64 holder)."""
    Q = np.asarray(Q, F64)
    if Q.shape != (sz.Hh, sz.Wh, 3):
        raise ValueError(f"Q must have shape {(sz.Hh, sz.Wh, 3)}")
    if not np.array_equal(Q, fx.f16(Q), equal_nan=True):
        raise ValueError("Q must hold binary16 values")
    xs, ys = np.arange(sz.We), np.arange(sz.He)
    cx = np.floor(fx.mul(fx.add(xs.astype(F32), 0.5), 1.5)).astype(np.int64)
    cy = np.floor(fx.mul(fx.add(ys.astype(F32), 0.5), 1.5)).astype(np.int64)
    we = sym.even_weights()
    wx = [np.where(xs % 2 == 0, we[n], we[4 - n])[None, :, None] for n in range(5)]
    wy = [np.where(ys % 2 == 0, we[n], we[4 - n])[:, None, None] for n in range(5)]
    cols = [np.clip(cx + n - 2, 0, sz.Wh - 1) for n in range(5)]
    rows = []
    for j in range(5):
        r = np.clip(cy + j - 2, 0, sz.Hh - 1)
        rows.append(_five(wx, [Q[r][:, c] for c in cols]))
    return _five(wy, rows)


def plain_colour(Q, sz, sym):
    """col of 54b.1 step 5 before the division by E, f32 RGB [He][We][3]."""
    return _decode(resample(Q, sz, sym), sym)


# --------------------------------------------------------------------------------------------------------------
# 54b.5 quarter pooled history

def quarter_pool(PH1, PH2, *, output_size, reset, symbols):
    """54b.5. PH1 / PH2: uint16 RGB10A2 code arrays. Returns the new PH2 (a copy): unchanged on reset, otherwise
    the 54b.5 value at every texel of the ceil(Wh/4) x ceil(Hh/4) area; texels of a larger PH2 beyond that
    area keep their content. PH1 is not changed."""
    (h1, w1), (h2, w2) = pooled_sizes(output_size)
    PH1, PH2 = np.asarray(PH1), np.asarray(PH2)
    if PH1.shape != (h1, w1, 4) or PH1.dtype != np.uint16:
        raise ValueError(f"PH1 must be uint16 codes of shape {(h1, w1, 4)}")
    if PH2.ndim != 3 or PH2.shape[2] != 4 or PH2.shape[0] < h2 or PH2.shape[1] < w2 or PH2.dtype != np.uint16:
        raise ValueError(f"PH2 must be uint16 codes covering {(h2, w2, 4)}")
    if not isinstance(reset, (bool, np.bool_)):
        raise ValueError("reset must be boolean")
    out = PH2.copy()
    if reset:
        return out
    s = symbols
    ph = read(Surface(PH1, "rgb10a2"))[..., :3]
    acc = np.zeros((h2, w2, 3))
    n = np.zeros((h2, w2, 1))
    ys, xs = np.mgrid[0:h2, 0:w2]
    for dx, dy in ((0, 0), (1, 0), (0, 1), (1, 1)):
        tx, ty = 2 * xs + dx, 2 * ys + dy
        ok = ((tx < w1) & (ty < h1))[..., None]
        c = fx.mul(ph[np.minimum(ty, h1 - 1), np.minimum(tx, w1 - 1)], s.C_HDEC)
        t = np.where((c > s.K5) | np.isnan(c), fx.min_nan(c, s.K10), s.K5)
        term = fx.sub(fx.exp2(fx.mul(fx.mul(t, s.K8), K9)), 1)
        acc = np.where(ok, fx.fma(term, s.K12, acc), acc)
        n = n + ok
    mean = fx.mul(fx.rcp(n.astype(F32)), acc)
    value = fx.mul(fx.mul(fx.mul(fx.log2(fx.fmax(fx.fma(mean, s.K4, 1), s.K5)), K6), s.K7), s.K11)
    out[:h2, :w2, :3] = encode(value, "rgb10a2")[..., :3]
    out[:h2, :w2, 3] = 3
    return out


# --------------------------------------------------------------------------------------------------------------
# 54b.6 histories

@dataclass
class ResolveHistories:
    """TH / OH / AH pairs of 54b.6 (binary16, We x He; float16 arrays) and their creation conditions.

    AH exists from creation when alpha upscaling is enabled; the frame of its creation (the first resolve of
    the instance) is a reset frame for 54b.3 and 54b.4. TH / OH are created zero-filled (WrongGPU's choice) on
    the first frame with a layer (OH: with the opacity layer and a layer)."""
    output_size: tuple
    alpha_upscaling: bool = False
    TH: list = None
    OH: list = None
    AH: list = None
    ah_created: bool = False
    log: list = field(default_factory=list)

    def __post_init__(self):
        sz = sizes(self.output_size)
        self.alpha_upscaling = bool(self.alpha_upscaling)
        if self.alpha_upscaling and self.AH is None:
            self.AH = [np.zeros((sz.He, sz.We, 1), np.float16) for _ in range(2)]
            self.ah_created = True

    def copy(self):
        dup = lambda pair: None if pair is None else [a.copy() for a in pair]  # noqa: E731
        return ResolveHistories(self.output_size, self.alpha_upscaling, dup(self.TH), dup(self.OH),
                                dup(self.AH), self.ah_created, list(self.log))


# --------------------------------------------------------------------------------------------------------------
# 54b.2-54b.4 resolve

@dataclass
class ResolveResult:
    colour: Surface                 # the output colour resource after the stores
    alpha: Surface                  # the separate output alpha after the stores (None if not attached)
    histories: ResolveHistories     # TH / OH / AH after this frame's writes, conditions cleared
    variant: str                    # 'plain' or 'extended'
    values: dict                    # f32 results over the We x He grid before the store conversion


def _scalar32(v, name):
    a = np.asarray(v, F64)
    if a.shape != () or not np.array_equal(a, a.astype(F32).astype(F64), equal_nan=True):
        raise ValueError(f"{name} must be an f32 scalar")
    return F32(a)


def resolve(Q, *, output_size, render_size, E, Ep, jitter, reset, parity, symbols, colour_out,
            histories, alpha_out=None, layer=None, cbp=None, opacity=None, alpha_in=None, colour_in=None,
            motion=None, motion_origin=(0, 0), P=None, G=None, bounded_stores=False):
    """One frame's output resolve (54b.1-54b.4, 54b.6). Pure function: returns new surfaces and histories.

    Q: binary16 RGB [Hh][Wh][3] (54.7 output-gated image); E, Ep: f32 (page 58); jitter: (jx, jy) f32 render
    pixels; reset: the combined reset; parity: par of 53.7 (reads copy par, writes 1 - par).
    colour_out / alpha_out: destination Surfaces with their origins (alpha_out one channel).
    layer (L), cbp (Bp), opacity (O): RGBA layer surfaces; L's origin is the layer origin, Bp's origin unused.
    alpha_in (A, channel 0) / colour_in (C, channel 3): alpha source surfaces with their own origins.
    motion: PM[1-par] binary16 [Hr][Wr][>=2] (54b.3 MV) and the motion-vector origin.
    P: binary16 [Hr][Wr][>=3]; G: uint8 codes [Hr][Wr][>=3] (55.3), needed with alpha upscaling.
    """
    sz = sizes(output_size)
    wr, hr = _pair(render_size, "render_size", positive=True)
    sym = symbols
    E, Ep = _scalar32(E, "E"), _scalar32(Ep, "Ep")
    if not isinstance(reset, (bool, np.bool_)) or parity not in (0, 1):
        raise ValueError("reset must be boolean and parity 0 or 1")
    jx, jy = (_scalar32(j, "jitter") for j in jitter)
    if not (np.isfinite(jx) and np.isfinite(jy)):
        raise ValueError("finite jitter required (page 52.8 passes the game's jitter through)")
    if histories.output_size != tuple(output_size):
        raise ValueError("histories belong to another output size")
    hist = histories.copy()
    par, wpar = parity, 1 - parity
    layer_attached = layer is not None or cbp is not None
    extended = layer_attached or hist.alpha_upscaling
    values = {}

    col = plain_colour(Q, sz, sym)                         # [He][We][3], before the division by E
    alpha = np.ones((sz.He, sz.We, 1))
    rE = fx.rcp(E)

    if extended:
        xs = np.arange(sz.We).astype(F32)
        ys = np.arange(sz.He).astype(F32)
        fWe, fHe, fWr, fHr = F32(sz.We), F32(sz.He), F32(wr), F32(hr)
        sxr, syr = fx.div(fWr, fWe), fx.div(fHr, fHe)
        sxh, syh = fx.div(fWe, F32(sz.Wh)), fx.div(fHe, F32(sz.Hh))
        jxp, jyp = fx.mul(fx.div(fWe, fWr), jx), fx.mul(fx.div(fHe, fHr), jy)
        rWe, rHe = fx.rcp(fWe), fx.rcp(fHe)
        xh, yh = fx.add(xs, 0.5), fx.add(ys, 0.5)
        kx = fx.rnd_half_away(fx.add(fx.fma(xh, sxr, -0.5), jx)).astype(np.int64)     # [We]
        ky = fx.rnd_half_away(fx.add(fx.fma(yh, syr, -0.5), jy)).astype(np.int64)     # [He]
        kxc, kyc = np.clip(kx, 0, wr - 1), np.clip(ky, 0, hr - 1)
        # 54b.3 step 2: motion in output pixels
        if motion is not None:
            mvt = np.asarray(motion, F64)
            if mvt.ndim != 3 or mvt.shape[:2] != (hr, wr) or mvt.shape[2] < 2:
                raise ValueError("motion must be PM[1-par] of the render size")
            vbx, vby = _pair(motion_origin, "motion_origin")
            mv = fetch(mvt[..., :2], kxc[None, :] + vbx, kyc[:, None] + vby)
            mxo, myo = fx.mul(mv[..., 0], sxh).astype(F64), fx.mul(mv[..., 1], syh).astype(F64)
        else:
            mxo = myo = np.zeros((sz.He, sz.We))
        hu = fx.mul(fx.add(fx.add(xs[None, :], mxo), 0.5), rWe)
        hv = fx.mul(fx.add(fx.add(ys[:, None], myo), 0.5), rHe)
        values.update(k=(kx, ky), m=(mxo, myo))

        created = False
        if layer_attached:
            src = cbp if cbp is not None else layer
            lbx, lby = layer.origin if layer is not None else (0, 0)
            if src.data.shape[2] != 4:
                raise ValueError("the layer is an RGBA surface")
            if hist.TH is None:
                hist.TH = [np.zeros((sz.He, sz.We, 4), np.float16) for _ in range(2)]
                hist.log.append("TH created")
                created = True
            use_o = opacity is not None
            if use_o and hist.OH is None:
                hist.OH = [np.zeros((sz.He, sz.We, 4), np.float16) for _ in range(2)]
                hist.log.append("OH created")
                created = True
            Lt = read(src)
            a = fx.add(xs, jxp)
            b = fx.add(ys, jyp)
            sa = np.abs(fx.sub(fx.rnd_half_away(a), a)) < 0.5
            sb = np.abs(fx.sub(fx.rnd_half_away(b), b)) < 0.5
            s = sb[:, None] & sa[None, :]
            w = np.where(s, sym.TBW, fx.mul(sym.TBW, sym.TBW)).astype(F64)[..., None]
            w1 = fx.sub(1, w)
            u = fx.mul(fx.add(xh, jxp), rWe)[None, :]
            v = fx.mul(fx.add(yh, jyp), rHe)[:, None]
            cur = bilinear(Lt, u, v)
            cur[..., :3] = fx.mul(E, cur[..., :3])
            if use_o:
                Ot = read(opacity)
                curO = bilinear(Ot, u, v)[..., :3]
            if reset or created or hist.ah_created:
                hst = cur.copy()
                if use_o:
                    hstO = curO.copy()
            else:
                mn = np.full((sz.He, sz.We, 4), FMAX, F64)
                mx = np.full((sz.He, sz.We, 4), FMINN, F64)
                for j in (-1, 0, 1):
                    for i in (-1, 0, 1):
                        t = fetch(Lt, np.clip(kx + i, 0, wr - 1)[None, :] + lbx,
                                  np.clip(ky + j, 0, hr - 1)[:, None] + lby)
                        t[..., :3] = fx.mul(E, t[..., :3])
                        mn, mx = fx.fmin(mn, t), fx.fmax(mx, t)
                H = bilinear(hist.TH[par].astype(F64), hu, hv)
                H[..., :3] = fx.mul(H[..., :3], fx.mul(E, fx.rcp(fx.fmax(Ep, sym.T_EFL))))
                hst = _clamp(H, mn, mx)
                if use_o:
                    obx, oby = opacity.origin
                    mnO = np.full((sz.He, sz.We, 3), FMAX, F64)
                    mxO = np.full((sz.He, sz.We, 3), FMINN, F64)
                    for j in (-1, 0, 1):
                        for i in (-1, 0, 1):
                            t = fetch(Ot, np.clip(kx + i, 0, wr - 1)[None, :] + obx,
                                      np.clip(ky + j, 0, hr - 1)[:, None] + oby)[..., :3]
                            mnO, mxO = fx.fmin(mnO, t), fx.fmax(mxO, t)
                    HO = bilinear(hist.OH[par].astype(F64), hu, hv)[..., :3]
                    hstO = _clamp(HO, mnO, mxO)
            T = fx.fma(w, cur, fx.mul(w1, hst)).astype(F64)
            hist.TH[wpar] = fx.f16(T).astype(np.float16)
            Ts = fx.f16(T)
            if use_o:
                Tp = fx.fma(w, curO, fx.mul(w1, hstO)).astype(F64)
                oh = np.ones((sz.He, sz.We, 4))
                oh[..., :3] = Tp
                hist.OH[wpar] = fx.f16(oh).astype(np.float16)
                Tps = fx.f16(Tp)
                col = fx.fma(col, fx.sub(1, Tps), Ts[..., :3]).astype(F64)
                values.update(Tp=Tp)
            else:
                col = fx.fma(col, fx.sub(1, Ts[..., 3:4]), Ts[..., :3]).astype(F64)
            values.update(T=T, s=s, cur=cur, hist=hst)

        if hist.alpha_upscaling:
            alpha_vals = _alpha(sz, wr, hr, sym, jx, jy, sxr, syr, kx, ky, hu, hv, hist, par, wpar,
                                reset, alpha_in, colour_in, P, G, values)
            if alpha_out is not None:
                alpha_store = alpha_vals
            else:
                alpha = alpha_vals
        hist.ah_created = False
        hist.log.append(f"extended par={par}")

    lin = np.concatenate([fx.mul(rE, col).astype(F64), alpha], axis=2)
    values["colour"] = lin
    width, height = (sz.Wo, sz.Ho) if bounded_stores else (sz.We, sz.He)
    colour = store(colour_out, lin, width=width, height=height)
    out_alpha = alpha_out
    if extended and hist.alpha_upscaling and alpha_out is not None:
        out_alpha = store(alpha_out, alpha_store, width=width, height=height)
    return ResolveResult(colour, out_alpha, hist, "extended" if extended else "plain", values)


def _clamp(H, mn, mx):
    """hist = (H > mn or unordered) ? (H >= mx ? mx : H) : mn."""
    H = np.asarray(H, F64)
    inner = np.where(H >= mx, mx, H)
    return np.where((H > mn) | np.isnan(H) | np.isnan(mn), inner, mn)


def _alpha_source(alpha_in, colour_in):
    if alpha_in is not None:
        return read(alpha_in)[..., 0], alpha_in.origin
    if colour_in is None:
        raise ValueError("alpha upscaling needs the input alpha or the colour input")
    t = read(colour_in)
    a = t[..., 3] if t.shape[2] >= 4 else np.ones(t.shape[:2])   # a format without alpha reads alpha 1
    return a, colour_in.origin


def _alpha(sz, wr, hr, sym, jx, jy, sxr, syr, kx, ky, hu, hv, hist, par, wpar, reset,
           alpha_in, colour_in, P, G, values):
    """54b.4 steps 1-4; returns the output alpha oa [He][We][1] and writes AH[1 - par]."""
    A, (ax, ay) = _alpha_source(alpha_in, colour_in)
    P = np.asarray(P, F64)
    G = np.asarray(G)
    if P.shape[:2] != (hr, wr) or P.shape[2] < 3 or G.shape[:2] != (hr, wr) or G.shape[2] < 3:
        raise ValueError("P and G are render-size head textures with three used channels")
    if G.dtype != np.uint8:
        raise ValueError("G holds 8-bit unorm codes")
    if not np.array_equal(P, fx.f16(P), equal_nan=True):
        raise ValueError("P must hold binary16 values")
    Gh = read_unorm8_as_half(G[..., :3])
    A2 = A[..., None]
    xs = np.arange(sz.We).astype(F32)
    ys = np.arange(sz.He).astype(F32)
    px = fx.fma(fx.add(xs, 0.5), sxr, jx)
    py = fx.fma(fx.add(ys, 0.5), syr, jy)
    fxi = np.floor(fx.sub(px, 0.5).astype(F64))
    fyi = np.floor(fx.sub(py, 0.5).astype(F64))
    dx0 = fx.sub(fx.add(fxi, 0.5), px)
    dy0 = fx.sub(fx.add(fyi, 0.5), py)
    fxi, fyi = fxi.astype(np.int64), fyi.astype(np.int64)
    W = np.zeros((sz.He, sz.We))
    Aw = np.zeros((sz.He, sz.We))
    Gs = np.zeros((sz.He, sz.We, 3))
    for i in (-1, 0, 1, 2):
        dx = fx.add(dx0, F32(i)).astype(F64)[None, :]
        tx = np.clip(fxi + i, 0, wr - 1)[None, :]
        for j in (-1, 0, 1, 2):
            dy = fx.add(dy0, F32(j)).astype(F64)[:, None]
            ty = np.clip(fyi + j, 0, hr - 1)[:, None]
            p = P[ty, tx]
            qx, qy, qxy = fx.mul(p[..., 0], sym.K2n), fx.mul(p[..., 1], sym.K2n), fx.mul(p[..., 2], sym.K2x2n)
            e = fx.fma(qxy, fx.mul(dx, dy), fx.fma(qx, fx.mul(dx, dx), fx.mul(qy, fx.mul(dy, dy))))
            wt = fx.exp2(e).astype(F64)
            W = fx.add(W, wt).astype(F64)
            Aw = fx.fma(wt, fetch(A2, tx + ax, ty + ay)[..., 0], Aw).astype(F64)
            Gs = fx.fma(wt[..., None], Gh[ty, tx], Gs).astype(F64)
    rW = fx.rcp(fx.fmax(W, sym.K13)).astype(F64)
    n = fx.mul(Gs, rW[..., None]).astype(F64)
    g = fx.sat(fx.mul(n[..., 0], fx.mul(W, sym.K1))).astype(F64)
    beta = fx.fma(g, fx.sub(1, n[..., 1]), n[..., 1]).astype(F64)
    gamma = fx.fma(g, fx.sub(1, n[..., 2]), n[..., 2]).astype(F64)
    a_up = fx.mul(Aw, rW).astype(F64)
    if reset or hist.ah_created:
        hc = np.zeros((sz.He, sz.We))
    else:
        h = bilinear(hist.AH[par].astype(F64), hu, hv)[..., 0]
        mn = np.full((sz.He, sz.We), FMAX, F64)
        mx = np.full((sz.He, sz.We), -FMAX, F64)
        for j in (-1, 0, 1):
            for i in (-1, 0, 1):
                t = fetch(A2, np.clip(kx + i, 0, wr - 1)[None, :] + ax,
                          np.clip(ky + j, 0, hr - 1)[:, None] + ay)[..., 0]
                mn, mx = fx.fmin(mn, t), fx.fmax(mx, t)
        hc = _clamp(h, mn, mx)
    ah = fx.fma(a_up, beta, fx.mul(fx.sub(1, beta), hc)).astype(F64)
    ah = np.where(np.isnan(ah), a_up, ah)
    oa = fx.fma(a_up, gamma, fx.mul(fx.sub(1, gamma), hc)).astype(F64)
    oa = np.where(np.isnan(oa), a_up, oa)
    hist.AH[wpar] = fx.f16(ah)[..., None].astype(np.float16)
    values.update(a_up=a_up, beta=beta, gamma=gamma, hc=hc, ah=ah, oa=oa, W=W)
    return oa[..., None]
