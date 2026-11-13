# SPDX-License-Identifier: MIT
"""Independent NumPy reference of spec 52 (RR preset F input stage): per-frame host constants (52.1.1, 52.8),
camera maps (52.7.3), the 64 front-end inputs (52.3), the ordered-dither e4m3 quantisation (52.4), the front-end
MLP (52.5), the IN0 block with the state-stream seed (52.6, through reference/stages.py) and the three motion
surfaces with the second vector m2 and its specular-motion reconstruction (52.7, 52.7.1, 52.7.2).

Conventions
- Textures are float32 arrays [H][W][channels] holding the texel values AFTER the graphics API's read conversion
  (52.1.2; `decode_*` below give the conversions for unorm / snorm / sRGB codes), each with its own top/left origin.
- Temporal images: PH1 / PH2 as unpacked RGB10A2 codes uint16 [h][w][>=3]; HID as snorm8 codes int8 [Hr][Wr][4];
  PM pair, Pe, Pn as float16 arrays.
- Every f32 operation follows reference/f32.py (round to nearest even, subnormal operands and results flushed).
- Token grid: the tensor [Ht][Wt]; tokens outside the tensor are never computed here: their features and x0 equal
  those of the reflected token (52.2), which the IN0 block call (stages.run_stage) obtains by its reflected gather.

Provisional readings (QUESTIONS-52.md in the repository root of this track) are marked "Q52-n" below.
"""
from dataclasses import dataclass, field, replace
import math

import numpy as np

from . import f32 as F
from .fmt import add16, d8, from_f16_bits, mm, q8, round_f16, to_f16_bits
from .scalar import activation

FEATURES = 64
# k -> feature name (52.3.4); every other k is the constant 0
FEATURE_NAMES = {
    0: "cc.r", 1: "cc.g", 2: "Dd.g", 3: "Dd.b", 4: "cc.b", 5: "Hc1@near.r", 6: "Ds.r", 7: "Ds.g",
    8: "Hc1@near.g", 9: "Hc1@near.b", 10: "Ds.b", 11: "nrm.x", 12: "Hc1@own.r", 13: "Hc1@own.g", 14: "nrm.y",
    15: "nrm.z", 16: "Hc1@own.b", 17: "Hc2@own.r", 18: "rough", 19: "md(near)", 20: "Hc2@own.g",
    21: "Hc2@own.b", 22: "hid(near)0", 23: "hid(near)1", 24: "Hc1@far.r", 25: "Hc1@far.g", 26: "hid(near)2",
    27: "hid(near)3", 28: "Hc1@far.b", 29: "Dd.r", 30: "cbt", 31: "sss", 32: "dof", 33: "resp", 36: "X3",
    37: "Hc1@m2.r", 40: "Hc1@m2.g", 41: "Hc1@m2.b", 44: "md(m2)", 45: "hid(m2)0", 48: "hid(m2)1",
    49: "hid(m2)2", 52: "hid(m2)3", 53: "zero",
}
SAMPLED_FEATURES = (5, 8, 9, 12, 13, 16, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 37, 40, 41, 44, 45, 48, 49,
                    52)   # features computed from a bilinear sample (52.1.4 tolerance)
DEPTH_TAPS = ((0, 0), (-1, -1), (0, -1), (1, -1), (-1, 0), (1, 0), (-1, 1), (0, 1), (1, 1))
ROLES = ("C", "Z", "V", "SV", "Dd", "Ds", "N", "R", "Q", "B", "G1", "G2", "Hd")


def nu(k):
    """56.3.2: the stored feature column that serves front-end input k."""
    p, r = divmod(k, 32)
    r0, r1, r2, r3, r4 = (r >> 0) & 1, (r >> 1) & 1, (r >> 2) & 1, (r >> 3) & 1, (r >> 4) & 1
    return 32 * p + 16 * r1 + 8 * r4 + 4 * r3 + 2 * r2 + r0


# ---------------------------------------------------------------------------------------------------------------
# texel read conversion (52.1.2)

def decode_unorm(code, bits):
    """n-bit unorm: code / (2^n - 1), rounded once to f32."""
    return F.f32(np.asarray(code, np.float64) / float((1 << bits) - 1))


def decode_snorm(code, bits):
    """n-bit snorm (two's complement code): max(code / (2^(n-1) - 1), -1), rounded once to f32."""
    return F.f32(np.maximum(np.asarray(code, np.float64) / float((1 << (bits - 1)) - 1), -1.0))


def decode_srgb8(code):
    """Standard sRGB-to-linear decode of 8-bit codes, the exact value rounded once to f32 (WrongGPU's choice,
    52.1.2 / 58.1 open item 6)."""
    from decimal import Decimal, localcontext
    from fractions import Fraction
    from . import exact
    table = np.empty(256, np.float32)
    with localcontext() as ctx:
        ctx.prec = 60
        for c in range(256):
            s = Decimal(c) / Decimal(255)
            v = s / Decimal("12.92") if s <= Decimal("0.04045") else ((s + Decimal("0.055")) / Decimal("1.055")) ** Decimal("2.4")
            table[c] = exact.round_frac(Fraction(v), exact.F32)
    return table[np.asarray(code, np.int64)]


def ph_decode(codes):
    """RGB10A2 colour history: the RGB channels as code / 1023 (f32)."""
    return decode_unorm(np.asarray(codes)[..., :3].astype(np.int64), 10)


def hid_decode(codes):
    """HID snorm8: code / 127, code -128 reads -1 (53.3)."""
    return decode_snorm(np.asarray(codes).astype(np.int64), 8)


# ---------------------------------------------------------------------------------------------------------------
# symbols (57.7.2) and per-frame host constants (52.1.1, 52.8)

SYMBOLS_F32 = ("K4", "K5", "K7", "K7n", "K8", "K10", "K12", "C_HDEC", "C_FCLIP", "C_AMAX", "C_GCL", "C_GCLn",
               "C_LWrb", "C_LWg", "C_MDL", "C_MDLn", "Gm", "Gd", "C_SV1", "C_SV2", "C_SV3")
SYMBOLS_F16 = ("Gh", "Gs")
SYMBOLS_B2 = ("C_SV4", "C_SV4n")


@dataclass(frozen=True)
class InputSymbols:
    """The page-57 symbols the input stage uses, as f32 values (binary16 symbols widened exactly)."""
    values: dict
    b2: bool = False

    def __getattr__(self, name):
        try:
            return self.values[name]
        except KeyError:
            raise AttributeError(name) from None

    @classmethod
    def from_groups(cls, constants, constants_b2=None):
        """From the converter's groups (57.7.2): name -> array [1]."""
        v = {}
        for n in SYMBOLS_F32:
            a = np.asarray(constants[n])
            if a.dtype != np.float32:
                raise ValueError(f"{n} must be float32")
            v[n] = np.float32(a.reshape(-1)[0])
        for n in SYMBOLS_F16:
            a = np.asarray(constants[n])
            if a.dtype != np.float16:
                raise ValueError(f"{n} must be float16")
            v[n] = np.float32(a.reshape(-1)[0])
        b2 = constants_b2 is not None
        if b2:
            for n in SYMBOLS_B2:
                a = np.asarray(constants_b2[n])
                if a.dtype != np.float32:
                    raise ValueError(f"{n} must be float32")
                v[n] = np.float32(a.reshape(-1)[0])
        return cls(v, b2)


def history_size(output_size):
    """52.1.1: Wh = (3 * even(Wo)) / 2 (integer division), likewise Hh."""
    wo, ho = output_size
    ev = lambda n: n + (n % 2)
    return (3 * ev(wo)) // 2, (3 * ev(ho)) // 2


def fl32q(a, b):
    """fl32(a / b) for integers: the exact quotient rounded once to f32."""
    return F.f32(np.float64(a) / np.float64(b)) if abs(a) < 2 ** 53 and abs(b) < 2 ** 53 else None


@dataclass
class FrameParams:
    """Everything the host knows about one evaluate call (52.8) that the input stage uses."""
    render_size: tuple                  # (Wr, Hr)
    output_size: tuple                  # creation-time (Wo, Ho)
    jitter: tuple                       # (jx, jy) f32, render pixels, no sign conversion
    prev_jitter: tuple                  # pj (53.5)
    mv_scale_raw: tuple                 # normalised motion scale (x, y) f32
    mv_jittered: bool                   # creation decision
    E: np.float32
    Ep: np.float32
    X3: np.float32
    reset: bool
    par: int
    fc: int
    SH: np.ndarray                      # uint32 [2][11][2]
    hw_depth: bool = True
    depth_inverted: bool = False
    nrm_dec: bool = False
    rgh_packed: bool = False
    V2C: np.ndarray = None              # current view-to-clip, 4x4 f32, row vectors
    W2V: np.ndarray = None              # current world-to-view
    V2Cp: np.ndarray = None             # stored CP pair (53.1); replaced by the current pair on reset frames
    W2Vp: np.ndarray = None
    specular_from_v: bool = False       # opt-in SpecularMotionFromV (52.7.1)


@dataclass
class FrameConstants:
    Wr: int
    Hr: int
    Wt: int
    Ht: int
    Wh: int
    Hh: int
    sx: int
    sy: int
    fc: int
    par: int
    reset: bool
    E: np.float32
    Ep: np.float32
    X3: np.float32
    jx: np.float32
    jy: np.float32
    pjx: np.float32
    pjy: np.float32
    ms: tuple                           # (msx, msy) render pixels per stored unit
    mo: tuple                           # (mox, moy)
    msh: tuple                          # (mshx, mshy) history pixels per stored unit
    rWhWr: np.float32                   # fl32(Wh / Wr)
    rHhHr: np.float32
    rWrWh: np.float32                   # fl32(Wr / Wh)
    rHrHh: np.float32
    invWr: np.float32                   # fl32(1 / Wr)
    invHr: np.float32
    invWh: np.float32                   # fl32(1 / Wh)
    invHh: np.float32
    hc_scale: tuple                     # (fl32(1 / (2 ceil(Wh/2))), fl32(1 / (2 ceil(Hh/2))))
    hoff: tuple                         # (pjx * fl32(1/Wr), pjy * fl32(1/Hr)) hidden-history offset (52.3.6)
    rev: bool
    zfar: np.float32
    hw_depth: bool
    nrm_dec: bool
    rgh_packed: bool
    V2C: np.ndarray                     # the current view-to-clip (52.7.2 step 1 reads four entries)
    Mpos: np.ndarray = None             # 52.7.3, None when the camera is unusable
    Mrep: np.ndarray = None
    camera_usable: bool = False
    specular_from_v: bool = False


def mv_offset_scale(params):
    """52.8 'Motion and jitter': ms = (raw.x * Wr, raw.y * Hr); off' = mvJittered ? f16rt(j - pj) : 0;
    mo = f16rt(off' / ms) with the quotient rounded to f32 first (Q52-1)."""
    wr, hr = params.render_size
    ms = (F.mul(F.f32(params.mv_scale_raw[0]), F.f32(wr)), F.mul(F.f32(params.mv_scale_raw[1]), F.f32(hr)))
    mo = []
    for a in range(2):
        off = F.f32(0)
        if params.mv_jittered:
            off = F.add(off, F.f16rt(F.sub(F.f32(params.jitter[a]), F.f32(params.prev_jitter[a]))))
        mo.append(F.f16rt(F.div(off, ms[a])))
    return ms, tuple(mo)


def frame_constants(params):
    wr, hr = params.render_size
    if wr <= 0 or hr <= 0:
        raise ValueError("zero render size is rejected before any pass (52.8)")
    wh, hh = history_size(params.output_size)
    wt, ht = 32 * -(-wr // 32), 32 * -(-hr // 32)
    sx, sy = (int(v) for v in np.asarray(params.SH)[params.par][0])
    ms, mo = mv_offset_scale(params)
    rWrWh, rHrHh = fl32q(wr, wh), fl32q(hr, hh)
    invWr, invHr = fl32q(1, wr), fl32q(1, hr)
    pjx, pjy = F.f32(params.prev_jitter[0]), F.f32(params.prev_jitter[1])
    c = FrameConstants(
        Wr=wr, Hr=hr, Wt=wt, Ht=ht, Wh=wh, Hh=hh, sx=sx, sy=sy, fc=params.fc % 8, par=params.par & 1,
        reset=bool(params.reset), E=F.f32(params.E), Ep=F.f32(params.Ep), X3=F.f32(params.X3),
        jx=F.f32(params.jitter[0]), jy=F.f32(params.jitter[1]), pjx=pjx, pjy=pjy, ms=ms, mo=mo,
        msh=(F.div(ms[0], rWrWh), F.div(ms[1], rHrHh)),
        rWhWr=fl32q(wh, wr), rHhHr=fl32q(hh, hr), rWrWh=rWrWh, rHrHh=rHrHh, invWr=invWr, invHr=invHr,
        invWh=fl32q(1, wh), invHh=fl32q(1, hh),
        hc_scale=(fl32q(1, 2 * -(-wh // 2)), fl32q(1, 2 * -(-hh // 2))),
        hoff=(F.mul(pjx, invWr), F.mul(pjy, invHr)),
        rev=bool(params.hw_depth and params.depth_inverted),
        zfar=F.f32(0.0 if params.depth_inverted else 1.0), hw_depth=bool(params.hw_depth),
        nrm_dec=bool(params.nrm_dec), rgh_packed=bool(params.rgh_packed),
        V2C=None if params.V2C is None else F.f32(params.V2C), specular_from_v=bool(params.specular_from_v))
    if params.V2C is not None and params.W2V is not None:
        if params.reset or params.V2Cp is None:
            V2Cp, W2Vp = params.V2C, params.W2V
        else:
            V2Cp, W2Vp = params.V2Cp, params.W2Vp
        maps = camera_maps(params.V2C, params.W2V, V2Cp, W2Vp, wr, hr)
        if maps is not None:
            c.Mpos, c.Mrep = maps
            c.camera_usable = True
    return c


# ---------------------------------------------------------------------------------------------------------------
# 52.7.3 camera maps (host side)

def matmul4(A, B):
    """(A @ B)[i][j] = ((A[i][0]B[0][j] + A[i][1]B[1][j]) + A[i][2]B[2][j]) + A[i][3]B[3][j], all rounded."""
    A, B = F.f32(A), F.f32(B)
    out = np.empty((4, 4), np.float32)
    for i in range(4):
        for j in range(4):
            s = F.mul(A[i][0], B[0][j])
            for k in (1, 2, 3):
                s = F.add(s, F.mul(A[i][k], B[k][j]))
            out[i][j] = s
    return out


def inv4(A):
    """I32 of 52.7.3 (rounded cofactor inverse). Returns None when it fails (some det[i] == 0)."""
    A = F.f32(A)
    cof = np.empty((4, 4), np.float32)
    for i in range(4):
        rows = [r for r in range(4) if r != i]
        for j in range(4):
            cols = [c for c in range(4) if c != j]
            a = [A[rows[0]][c] for c in cols]
            b = [A[rows[1]][c] for c in cols]
            c_ = [A[rows[2]][c] for c in cols]
            E = lambda p, q: F.sub(F.mul(b[p], c_[q]), F.mul(b[q], c_[p]))
            det = F.add(F.sub(F.mul(a[0], E(1, 2)), F.mul(a[1], E(0, 2))), F.mul(a[2], E(0, 1)))
            cof[i][j] = F.neg(det) if (i + j) % 2 else det
    dets = []
    for i in range(4):
        s = F.f32(0.0)
        for j in range(4):
            s = F.add(s, F.mul(A[i][j], cof[i][j]))
        dets.append(s)
    if any(d == 0 for d in dets):
        return None
    r = F.rcp(dets[3])
    out = np.empty((4, 4), np.float32)
    for i in range(4):
        for j in range(4):
            out[j][i] = F.mul(cof[i][j], r)
    return out


def camera_maps(V2C, W2V, V2Cp, W2Vp, wr, hr):
    """52.7.3: (Mpos, Mrep) or None when the frame has no usable camera matrices."""
    Pc, Pp = inv4(W2V), inv4(W2Vp)
    if Pc is None or Pp is None:
        return None
    Rc = Pc.copy()
    Rc[3, :3] = 0.0
    Rp = Pp.copy()
    for c in range(3):
        Rp[3][c] = F.sub(Pp[3][c], Pc[3][c])
    iRc, iRp = inv4(Rc), inv4(Rp)
    if iRc is None or iRp is None:
        return None
    Am = matmul4(iRc, V2C)
    Bm = matmul4(iRp, V2Cp)
    iAm = inv4(Am)
    if iAm is None:
        return None
    Tcc = matmul4(iAm, Bm)
    K = np.eye(4, dtype=np.float32)
    hw, hh = F.f32(wr) * np.float32(0.5), F.f32(hr) * np.float32(0.5)
    K[0][0] = K[3][0] = hw
    K[1][1] = -hh
    K[3][1] = hh
    Mpos = inv4(matmul4(Am, K))
    if Mpos is None:
        return None
    Mrep = matmul4(matmul4(Am, Tcc), K)
    return Mpos, Mrep


# ---------------------------------------------------------------------------------------------------------------
# 52.1.4 bilinear sampling (reference sampler)

def bilinear(img, u, v):
    """img: f32 [Hi][Wi][C] (texel values after conversion); u, v: f32 arrays of normalised coordinates.
    Returns f32 [..., C]."""
    img = F.ftz(img)
    Hi, Wi = img.shape[:2]
    u = F.fmin(F.fmax(F.nan0(u), F.f32(0)), F.f32(1))
    v = F.fmin(F.fmax(F.nan0(v), F.f32(0)), F.f32(1))
    s = F.sub(F.mul(u, F.f32(Wi)), F.f32(0.5))
    t = F.sub(F.mul(v, F.f32(Hi)), F.f32(0.5))
    i0 = np.floor(s).astype(np.int64)
    j0 = np.floor(t).astype(np.int64)
    fx = F.sub(s, F.f32(i0))
    fy = F.sub(t, F.f32(j0))
    ia, ib = np.clip(i0, 0, Wi - 1), np.clip(i0 + 1, 0, Wi - 1)
    ja, jb = np.clip(j0, 0, Hi - 1), np.clip(j0 + 1, 0, Hi - 1)
    T00, T10 = img[ja, ia], img[ja, ib]
    T01, T11 = img[jb, ia], img[jb, ib]
    fx, fy = fx[..., None], fy[..., None]
    one = F.f32(1)
    r0 = F.add(F.mul(F.sub(one, fx), T00), F.mul(fx, T10))
    r1 = F.add(F.mul(F.sub(one, fx), T01), F.mul(fx, T11))
    return F.add(F.mul(F.sub(one, fy), r0), F.mul(fy, r1))


# ---------------------------------------------------------------------------------------------------------------
# inputs

@dataclass
class Texture:
    data: np.ndarray                    # f32 [H][W][C], values after the read conversion
    origin: tuple = (0, 0)              # (left, top)

    def __post_init__(self):
        self.data = np.asarray(self.data, np.float32)
        if self.data.ndim == 2:
            self.data = self.data[..., None]


@dataclass
class InputFrame:
    textures: dict                      # role -> Texture (attached roles only; 52.1.2 / 52.8)
    PH1: np.ndarray = None              # uint16 RGB10A2 codes [ceil(Wh/2)][ceil(Wh/2)][>=3]
    PH2: np.ndarray = None
    HID: np.ndarray = None              # int8 [Hr][Wr][4]
    PM_read: np.ndarray = None          # float16 [Hr][Wr][4]: PM[par]
    PM_write: np.ndarray = None         # prior content of PM[1-par]
    Pe: np.ndarray = None               # prior content
    Pn: np.ndarray = None

    def attached(self, role):
        return role in self.textures


def origin_of(frame, role):
    """52.1.2: SV uses V's origin; B, G1 and G2 use C's origin; every other role its own."""
    src = {"SV": "V", "B": "C", "G1": "C", "G2": "C"}.get(role, role)
    return frame.textures[src].origin


def fetch(frame, role, xs, ys, allow_oob=False):
    """Integer texel fetch at render-area coordinates (already clamped by the caller) plus the role's origin."""
    tex = frame.textures[role]
    ox, oy = origin_of(frame, role)
    X = np.asarray(xs, np.int64) + ox
    Y = np.asarray(ys, np.int64) + oy
    H, W = tex.data.shape[:2]
    inside = (X >= 0) & (X < W) & (Y >= 0) & (Y < H)
    if not allow_oob and not np.all(inside):
        raise ValueError(f"texture {role}: fetch outside the resource (52.0.1, 52.8 validation)")
    val = tex.data[np.clip(Y, 0, H - 1), np.clip(X, 0, W - 1)]
    val = np.where(inside[..., None], val, np.float32(0))
    return F.ftz(val)


# ---------------------------------------------------------------------------------------------------------------
# 52.3 features

def cc(v, E, S):
    u = F.fmin(F.nan0(F.mul(E, v)), F.F16_MAX)
    y = F.fmax(F.fma(u, S.K4, F.f32(1)), S.K5)
    return F.mul(F.mul(F.log2(y), F.LOG10_2), S.K7)


def ccf(v, E, S):
    u = F.mul(E, v)
    y = F.fmax(F.fma(u, S.K4, F.f32(1)), S.K5)
    return F.mul(F.mul(F.log2(y), F.LOG10_2), S.K7)


def slc(v, S):
    t = F.mul(F.log2(F.fma(np.abs(F.f32(v)), S.K4, F.f32(1))), F.LOG10_2)
    s = F.sel(F.f32(v) >= 0, S.K7, S.K7n)
    r = F.nan0(F.mul(s, t))
    return F.fmin(F.fmax(r, S.C_GCLn), S.C_GCL)


def clip(v, S):
    return F.fmin(F.nan0(v), S.C_FCLIP)


def hdecode(h, k, S):
    """52.3.2 decode and re-expose of a sampled history value h (f32)."""
    h1 = F.mul(h, S.C_HDEC)
    hp = F.sel(h1 > S.K5, F.fmin(h1, S.K10), S.K5)
    e = F.exp2(F.mul(F.mul(hp, S.K8), F.LOG2_10))
    lin = F.mul(F.mul(F.sub(e, F.f32(1)), S.K12), F.rcp(k.Ep))
    return lin


def history_position(X, Y, mh, k):
    """52.3.2: (pu, pv) for a vector mh (history pixels) at token (X, Y)."""
    cx = F.sub(F.add(F.f32(X), F.f32(0.5)), k.jx)
    cy = F.sub(F.add(F.f32(Y), F.f32(0.5)), k.jy)
    return F.fma(cx, k.rWhWr, mh[0]), F.fma(cy, k.rHhHr, mh[1])


def outside_history(pu, pv, k):
    return (pu < 0) | (pu > F.f32(k.Wh)) | (pv < 0) | (pv > F.f32(k.Hh))


def history_feature(frame, X, Y, mh, k, S, Cown, both=False):
    """HF of 52.3.2 for Hc1 (and Hc2 when both): returns clip(HF) per channel [..., 3] (and the Hc2 triple)."""
    pu, pv = history_position(X, Y, mh, k)
    invalid = k.reset | outside_history(pu, pv, k)
    fallback = F.mul(S.Gh, ccf(Cown, k.E, S))                        # [..., 3]
    out = []
    srcs = [frame.PH1] + ([frame.PH2] if both else [])
    for src in srcs:
        if k.reset or src is None:
            val = fallback
        else:
            u, v = F.mul(pu, k.hc_scale[0]), F.mul(pv, k.hc_scale[1])
            h = bilinear(ph_decode(src), u, v)
            hf = F.mul(ccf(hdecode(h, k, S), k.E, S), S.Gh)
            val = F.sel(invalid[..., None], fallback, hf)
        out.append(clip(val, S))
    return out, (pu, pv, invalid)


def hidden_features(frame, pu, pv, k, S):
    """52.3.6: hid(t)[0..3] for the history position (pu, pv) of the vector t."""
    shape = np.shape(pu)
    if k.reset:
        return np.zeros(shape + (4,), np.float32)
    uh = F.fma(pu, k.invWh, k.hoff[0])
    vh = F.fma(pv, k.invHh, k.hoff[1])
    samp = bilinear(hid_decode(frame.HID), uh, vh)
    samp = F.sel(outside_history(pu, pv, k)[..., None], F.f32(0), samp)
    return F.mul(S.Gs, samp)


def motion_difference(frame, X, Y, mh, chans, dq, k, S):
    """52.3.7: md for the vector mh compared with the Pv channels chans; 0 on reset frames."""
    shape = np.shape(X)
    if k.reset:
        return np.zeros(shape, np.float32)
    mrx, mry = F.mul(mh[0], k.rWrWh), F.mul(mh[1], k.rHrHh)
    cx = F.sub(F.add(F.f32(X), F.f32(0.5)), k.jx)
    cy = F.sub(F.add(F.f32(Y), F.f32(0.5)), k.jy)
    u = F.mul(F.add(F.add(cx, k.pjx), mrx), k.invWr)
    v = F.mul(F.add(F.add(cy, k.pjy), mry), k.invHr)
    p = bilinear(F.f32(frame.PM_read.astype(np.float32)), u, v)
    dx = F.fma(F.neg(p[..., chans[0]]), k.rWrWh, mrx)               # * MDK: identity (57.7.4)
    dy = F.fma(F.neg(p[..., chans[1]]), k.rHrHh, mry)
    cl = lambda t: F.sel(t > S.C_MDLn, F.fmin(t, S.C_MDL), S.C_MDLn)
    dxs, dys = F.mul(cl(dx), S.Gm), F.mul(cl(dy), S.Gm)
    n2 = F.fma(dxs, dxs, F.mul(dys, dys))
    L = F.sel(n2 > 0, F.sqrt(n2), n2)
    md = F.mul(F.mul(F.log2(F.add(F.add(F.f32(1), L), dq)), F.LN_2), S.Gd)
    return F.fmin(md, S.C_MDL)


def motion_at(frame, role_for_values, xs, ys, k):
    """mh = ((V.r + mox) * mshx, (V.g + moy) * mshy) of the texel (xs, ys) (V's origin for V and SV)."""
    t = fetch(frame, role_for_values, xs, ys)
    return (F.mul(F.add(t[..., 0], k.mo[0]), k.msh[0]), F.mul(F.add(t[..., 1], k.mo[1]), k.msh[1]))


def depth_taps(frame, xc, yc, k):
    """52.3.3: indices (0..8) of near and far among DEPTH_TAPS, and the tap texel coordinates."""
    zs, cs = [], []
    for dx, dy in DEPTH_TAPS:
        tx = np.clip(xc + dx, 0, k.Wr - 1)
        ty = np.clip(yc + dy, 0, k.Hr - 1)
        zs.append(fetch(frame, "Z", tx, ty)[..., 0])
        cs.append((tx, ty))
    near = np.zeros(np.shape(xc), np.int64)
    far = np.zeros(np.shape(xc), np.int64)
    bn, bf = zs[0], zs[0]
    for i in range(1, 9):
        z = zs[i]
        cn = (z > bn) if k.rev else (z < bn)
        cf = (z < bf) if k.rev else (z > bf)
        near = np.where(cn, i, near)
        bn = np.where(cn, z, bn)
        far = np.where(cf, i, far)
        bf = np.where(cf, z, bf)
    return near, far, cs


def decoded_normal(frame, xc, yc, k):
    n = fetch(frame, "N", xc, yc)
    xyz = n[..., :3]
    if k.nrm_dec:
        xyz = F.mul(F.f32(2), F.sub(xyz, F.f32(0.5)))
    return xyz, n


# ---------------------------------------------------------------------------------------------------------------
# 52.7.2 specular-motion reconstruction

def _dot3(a, b):
    return F.fma(a[2], b[2], F.fma(a[0], b[0], F.mul(a[1], b[1])))


def _len3(a):
    return F.sqrt(_dot3(a, a))


def _U(a, S):
    r = F.rcp(F.add(_len3(a), S.C_SV3))
    return [F.mul(c, r) for c in a]


def _Fv(a, s, b):
    return [F.fma(a[i], s, b[i]) for i in range(3)]


def _H(a, M):
    return [F.add(_dot3(a, [M[0][c], M[1][c], M[2][c]]), M[3][c]) for c in range(4)]


def _cross3(a, b):
    nxt, lst = (1, 2, 0), (2, 0, 1)
    return [F.fma(a[nxt[c]], b[lst[c]], F.neg(F.mul(a[lst[c]], b[nxt[c]]))) for c in range(3)]


def reconstruct_m2(Z, nv, Hd, px, py, k, S):
    """52.7.2 for arrays of tokens. Returns (m2x, m2y, exited) with exited True where B1 gives up (m2 = mO)."""
    shape = np.shape(Z)
    b2 = S.b2
    exited = np.zeros(shape, bool)
    V = k.V2C
    # step 1: device depth
    if k.hw_depth:
        Dz = Z
    else:
        if not b2:
            exited |= ~np.isfinite(Z)
        num = F.fma(Z, V[2][2], V[3][2])
        Dz = F.mul(num, F.rcp(F.mul(Z, V[2][3]))) if V[3][3] < 1 else num
    if not b2:
        exited |= (Dz == k.zfar)                                         # Q52-3: both depth kinds
    # step 2: position
    h = _H([px, py, Dz], k.Mpos)
    if not b2:
        exited |= ~((h[3] > 0) & np.isfinite(h[3]))
    rw = F.rcp(h[3])
    p = [F.mul(h[c], rw) for c in range(3)]
    # step 3: reflected-segment length
    sl = F.fmin(Hd, S.C_SV2)
    sl = F.sel(sl > 0, sl, S.C_SV2)
    # step 4
    n = _U([nv[..., 0], nv[..., 1], nv[..., 2]], S)
    v = _U(p, S)
    # step 5
    dnv = _dot3(n, v)
    t5 = F.add(dnv, dnv)
    s = [F.mul(c, sl) for c in _Fv([F.neg(c) for c in n], t5, v)]
    c = _dot3(n, s)
    flip = c < 0
    n = [F.sel(flip, F.neg(e), e) for e in n]
    c = F.sel(flip, F.neg(c), c)
    mc = F.fmax(c, S.C_SV1)
    # step 6
    J = [n[2], F.neg(n[0]), n[1]]
    t = _U(_Fv([F.neg(e) for e in n], _dot3(n, J), J), S)
    b = _cross3(n, t)
    gm = F.rcp(F.sub(F.f32(0), F.rcp(mc)))
    km = F.mul(F.rcp(mc), gm)
    Lv = _len3([F.mul(_dot3(s, t), km), F.mul(_dot3(s, b), km), gm])
    q = _Fv(v, Lv, p)
    # step 7
    r = _H(q, k.Mrep)
    rx, ry = k.rWhWr, k.rHhHr
    if not b2:
        exited |= ~((r[3] > 0) & np.isfinite(r[3]))
        rr = F.rcp(r[3])
        ox = F.mul(F.fma(r[0], rr, F.neg(px)), rx)
        exited |= ~np.isfinite(ox)
        oy = F.mul(F.fma(r[1], rr, F.neg(py)), ry)
        exited |= ~np.isfinite(oy)
        return ox, oy, exited
    big = np.abs(r[3]) > S.C_SV1
    rr = F.rcp(r[3])
    ux = F.sel(big, F.mul(r[0], rr), F.f32(0))
    uy = F.sel(big, F.mul(r[1], rr), F.f32(0))
    ex, ey = F.sub(ux, px), F.sub(uy, py)
    cl2 = lambda a: F.sel(np.isnan(a), a, F.sel(a > S.C_SV4n, F.fmin(a, S.C_SV4), S.C_SV4n))
    return F.mul(cl2(ex), rx), F.mul(cl2(ey), ry), exited


# ---------------------------------------------------------------------------------------------------------------
# 52.4 quantisation

def dither_values(D, fc, X, Y):
    """d = floor(D[i] / 512), i = (fc * 16 + (y mod 16)) * 16 + (x mod 16)."""
    D = np.asarray(D, np.uint16)
    i = (fc * 16 + np.mod(Y, 16)) * 16 + np.mod(X, 16)
    return (D[i] >> 9).astype(np.int64)


def quantise(v, d):
    """52.4: f32 feature values v and dither values d (0..127) -> e4m3 codes (uint8)."""
    v = F.f32(v)
    d = np.broadcast_to(np.asarray(d, np.int64), v.shape)
    x = round_f16(v.astype(np.float64))
    xb = to_f16_bits(x).astype(np.int64)
    sign = (xb >> 15) & 1
    ex = (xb >> 10) & 0x1F
    finite = np.isfinite(x)
    fb = from_f16_bits(((sign << 15) | (ex << 10)).astype(np.uint16))
    fa = from_f16_bits(((sign << 15) | (ex << 10) | d).astype(np.uint16))
    xs = np.where(finite, x, 0.0)
    fbs = np.where(finite, fb, 0.0)
    fas = np.where(finite, fa, 0.0)
    s1 = add16(xs, fas)
    s2 = add16(s1, -fbs)
    sb = to_f16_bits(s2).astype(np.int64)
    e2 = (sb >> 10) & 0x1F
    m3 = (sb & 0x3FF) >> 7
    e = e2 - 15
    code = np.zeros(v.shape, np.int64)
    sat = (e2 == 31) | (e > 8) | ((e == 8) & (m3 == 7))
    normal = ~sat & (e2 > 0) & (e >= -6)
    code = np.where(normal, ((e + 7) << 3) | m3, code)
    sub = ~sat & (e2 > 0) & (e < -6)                         # truncate, then RNE on the e4m3 subnormal grid
    sh = np.clip(-(e + 6), 1, 31)
    mant = 8 + m3
    q = mant >> sh
    rem = mant & ((1 << sh) - 1)
    half = 1 << (sh - 1)
    up = (rem > half) | ((rem == half) & ((q & 1) == 1))
    code = np.where(sub, q + up, code)
    code = np.where(sat, 0x7E, code)
    code = np.where(~finite & ~np.isnan(x), 0x7E, code)        # +-inf
    code = np.where(np.isnan(x), 0x7F, code)
    return (code | (sign << 7)).astype(np.uint8)


# ---------------------------------------------------------------------------------------------------------------
# the features of every tensor token

def compute_features(frame, k, S, D, trace=False):
    """52.3 / 52.7.1 on the tensor grid. Returns dict: feat f32 [Ht][Wt][64], codes u8 [Ht][Wt][64],
    hold (f16 values as float64) [Ht][Wt][4], m2 / mO / mN / mF (f32 pairs), surfaces dict, info."""
    for role in ("C", "Z", "V", "Dd", "Ds", "N"):
        if not frame.attached(role):
            raise ValueError(f"required input {role} missing (52.8)")
    if not k.rgh_packed and not frame.attached("R"):
        raise ValueError("roughness R required when rghPacked = 0")
    Ht, Wt, Wr, Hr = k.Ht, k.Wt, k.Wr, k.Hr
    Y, X = np.meshgrid(np.arange(Ht), np.arange(Wt), indexing="ij")
    xc, yc = np.minimum(X, Wr - 1), np.minimum(Y, Hr - 1)
    feat = np.zeros((Ht, Wt, FEATURES), np.float32)
    E = k.E

    C = fetch(frame, "C", xc, yc)[..., :3]
    feat[..., 0], feat[..., 1], feat[..., 4] = cc(C[..., 0], E, S), cc(C[..., 1], E, S), cc(C[..., 2], E, S)
    Dd = fetch(frame, "Dd", xc, yc)
    Ds = fetch(frame, "Ds", xc, yc)
    A = S.C_AMAX
    feat[..., 2], feat[..., 3], feat[..., 29] = F.fmin(Dd[..., 1], A), F.fmin(Dd[..., 2], A), F.fmin(Dd[..., 0], A)
    feat[..., 6], feat[..., 7], feat[..., 10] = F.fmin(Ds[..., 0], A), F.fmin(Ds[..., 1], A), F.fmin(Ds[..., 2], A)
    nrm, Nraw = decoded_normal(frame, xc, yc, k)
    feat[..., 11], feat[..., 14], feat[..., 15] = (F.fmin(nrm[..., i], A) for i in range(3))
    if k.rgh_packed:
        rough = Nraw[..., 3]
    else:
        rough = fetch(frame, "R", xc, yc)[..., 0]
        pad = (X >= Wr) | (Y >= Hr)
        if np.any(pad):
            ly = np.mod(Y + k.sy, 8)                                       # Q52-4: tensor tokens only
            top = pad & (ly < 4)
            r_top = fetch(frame, "R", X, yc, allow_oob=True)[..., 0]       # x unclamped, out-of-texture -> 0
            rough = np.where(top, r_top, rough)
    feat[..., 18] = F.fmin(rough, A)

    # motion: own, near, far
    near, far, taps = depth_taps(frame, xc, yc, k)
    tx = np.stack([t[0] for t in taps], -1)
    ty = np.stack([t[1] for t in taps], -1)
    pick = lambda idx, a: np.take_along_axis(a, idx[..., None], -1)[..., 0]
    mO = motion_at(frame, "V", xc, yc, k)
    mN = motion_at(frame, "V", pick(near, tx), pick(near, ty), k)
    mF = motion_at(frame, "V", pick(far, tx), pick(far, ty), k)

    # second vector m2 (52.7.1)
    smv = frame.attached("SV") or frame.attached("Hd")
    m2 = mO
    m2_src = "mO"
    exited = None
    if smv:
        if frame.attached("SV"):
            m2, m2_src = motion_at(frame, "SV", xc, yc, k), "SV"
        elif k.specular_from_v:
            m2_src = "mO(SpecularMotionFromV)"
        elif not k.camera_usable:
            m2_src = "mO(unusable camera)"
        else:
            zt = fetch(frame, "Z", xc, yc)[..., 0]
            hd = fetch(frame, "Hd", xc, yc)[..., 0]
            oz = origin_of(frame, "Z")
            px = F.add(F.f32(xc + oz[0]), F.f32(0.5))
            py = F.add(F.f32(yc + oz[1]), F.f32(0.5))
            rx_, ry_, exited = reconstruct_m2(zt, nrm, hd, px, py, k, S)
            m2 = (F.sel(exited, mO[0], rx_), F.sel(exited, mO[1], ry_))
            m2_src = "reconstruction-" + ("B2" if S.b2 else "B1")

    # history features
    (h_own1, h_own2), pos_own = history_feature(frame, X, Y, mO, k, S, C, both=True)
    (h_near,), pos_near = history_feature(frame, X, Y, mN, k, S, C)
    (h_far,), pos_far = history_feature(frame, X, Y, mF, k, S, C)
    (h_m2,), pos_m2 = history_feature(frame, X, Y, m2, k, S, C)
    feat[..., 12], feat[..., 13], feat[..., 16] = h_own1[..., 0], h_own1[..., 1], h_own1[..., 2]
    feat[..., 17], feat[..., 20], feat[..., 21] = h_own2[..., 0], h_own2[..., 1], h_own2[..., 2]
    feat[..., 5], feat[..., 8], feat[..., 9] = h_near[..., 0], h_near[..., 1], h_near[..., 2]
    feat[..., 24], feat[..., 25], feat[..., 28] = h_far[..., 0], h_far[..., 1], h_far[..., 2]
    feat[..., 37], feat[..., 40], feat[..., 41] = h_m2[..., 0], h_m2[..., 1], h_m2[..., 2]

    # hidden-history and motion-difference features
    hn = hidden_features(frame, pos_near[0], pos_near[1], k, S)
    hm = hidden_features(frame, pos_m2[0], pos_m2[1], k, S)
    feat[..., 22], feat[..., 23], feat[..., 26], feat[..., 27] = hn[..., 0], hn[..., 1], hn[..., 2], hn[..., 3]
    feat[..., 45], feat[..., 48], feat[..., 49], feat[..., 52] = hm[..., 0], hm[..., 1], hm[..., 2], hm[..., 3]
    dq = fetch(frame, "Q", xc, yc)[..., 0] if frame.attached("Q") else np.zeros(X.shape, np.float32)
    feat[..., 19] = motion_difference(frame, X, Y, mN, (0, 1), dq, k, S)
    feat[..., 44] = motion_difference(frame, X, Y, m2, (2, 3), dq, k, S)

    # optional guides (52.3.5); an unattached guide is the neutral value 0, still clipped (52.3.4 table)
    for kk in (30, 31, 32):
        feat[..., kk] = clip(F.f32(0), S)
    if frame.attached("B"):
        B = fetch(frame, "B", xc, yc)
        d = {}
        for ch in (2, 0, 1):
            d[ch] = F.fma(F.neg(E), B[..., ch], F.fmin(F.nan0(F.mul(E, C[..., ch])), F.F16_MAX))
        lum = F.fma(d[2], S.C_LWrb, F.fma(d[0], S.C_LWrb, F.mul(d[1], S.C_LWg)))
        feat[..., 30] = clip(slc(lum, S), S)
    if frame.attached("G1"):
        feat[..., 31] = clip(slc(F.mul(E, fetch(frame, "G1", xc, yc)[..., 0]), S), S)
    if frame.attached("G2"):
        feat[..., 32] = clip(slc(F.mul(E, fetch(frame, "G2", xc, yc)[..., 0]), S), S)
    feat[..., 33] = clip(F.f32(0), S)                                     # responsivity: constant 0 (57.7.4)
    feat[..., 36] = np.broadcast_to(k.X3, X.shape)

    codes = quantise(feat, dither_values(D, k.fc, X, Y)[..., None])
    # 52.6: the state-stream seed
    hold = np.zeros((Ht, Wt, 4), np.float64)
    hold[..., :3] = round_f16(F.fmin(F.nan0(F.mul(E, C)), F.F16_MAX).astype(np.float64))

    out = dict(feat=feat, codes=codes, hold=hold, mO=mO, mN=mN, mF=mF, m2=m2, m2_src=m2_src)
    out["surfaces"] = motion_surfaces(frame, k, mN, mF, mO, m2)
    if trace:
        out.update(near=near, far=far, exited=exited, pos=dict(own=pos_own, near=pos_near, far=pos_far, m2=pos_m2))
    return out


def motion_surfaces(frame, k, mN, mF, mO, m2):
    """52.7: write Pv / Pe / Pn for the render-area tokens at the motion-vector texel (origin included);
    writes outside the Wr x Hr image are discarded; unwritten texels keep their prior content."""
    Wr, Hr = k.Wr, k.Hr
    PM = np.array(frame.PM_write if frame.PM_write is not None else np.zeros((Hr, Wr, 4), np.float16), np.float16)
    Pe = np.array(frame.Pe if frame.Pe is not None else np.zeros((Hr, Wr, 2), np.float16), np.float16)
    Pn = np.array(frame.Pn if frame.Pn is not None else np.zeros((Hr, Wr, 2), np.float16), np.float16)
    for a in (PM, Pe, Pn):
        if a.shape[:2] != (Hr, Wr):
            raise ValueError("motion surfaces must have the render size")
    ox, oy = frame.textures["V"].origin
    h16 = lambda a: from_f16_bits(to_f16_bits(round_f16(np.asarray(a, np.float64)))).astype(np.float16)
    vals = dict(
        PM=np.stack([h16(mN[0][:Hr, :Wr]), h16(mN[1][:Hr, :Wr]), h16(m2[0][:Hr, :Wr]), h16(m2[1][:Hr, :Wr])], -1),
        Pe=np.stack([h16(mF[0][:Hr, :Wr]), h16(mF[1][:Hr, :Wr])], -1),
        Pn=np.stack([h16(mO[0][:Hr, :Wr]), h16(mO[1][:Hr, :Wr])], -1))
    x0, x1 = max(0, -ox), min(Wr, Wr - ox)
    y0, y1 = max(0, -oy), min(Hr, Hr - oy)
    if x1 > x0 and y1 > y0:
        for name, dst in (("PM", PM), ("Pe", Pe), ("Pn", Pn)):
            dst[y0 + oy:y1 + oy, x0 + ox:x1 + ox] = vals[name][y0:y1, x0:x1]
    return dict(PM=PM, Pe=Pe, Pn=Pn)


# ---------------------------------------------------------------------------------------------------------------
# 52.5 front-end MLP and 52.6 the IN0 block

def front_end(codes, FE, wc):
    """codes: e4m3 [..., 64]; FE: dict W1_0, W1_1 [64][32] codes, b1_0, b1_1 [32] f16, W2_0, W2_1 [32][32]
    codes, b2 [32] f16; wc: window constants (activation). Returns x0 codes [..., 32] (and the trace)."""
    f = d8(codes)
    A = []
    for br in (0, 1):
        H = mm(f, d8(FE[f"W1_{br}"]), FE[f"b1_{br}"])
        A.append(q8(activation(H, wc)))
    Z = mm(d8(A[0]), d8(FE["W2_0"]), FE["b2"])
    out0 = mm(d8(A[1]), d8(FE["W2_1"]), Z)
    return q8(out0), dict(A0=A[0], A1=A[1], Z=Z, out0=out0)


def fe_tensors(group):
    """Front-end tensors from the converter's IN0 group (56.4.3 names)."""
    return {"W1_0": group["FE.W1_0"], "W1_1": group["FE.W1_1"], "b1_0": group["FE.b1_0"],
            "b1_1": group["FE.b1_1"], "W2_0": group["FE.W2_0"], "W2_1": group["FE.W2_1"], "b2": group["FE.b2"]}


def input_stage(frame, params, S, IN0, consts, wc, trace=False):
    """The whole input stage of one frame.

    IN0: logical tensors of the IN0 group (block tensors of 51.2 by their names, FE.* of 52.5, D, Wh, bh);
    consts: dict ra, m, g (f16 values). Returns dict with x0, hold, feat, codes, y2, hnew, hmerged, surfaces,
    the frame constants and the block aux."""
    from .stages import run_stage
    k = frame_constants(params)
    feats = compute_features(frame, k, S, IN0["D"], trace=trace)
    x0, fe = front_end(feats["codes"], fe_tensors(IN0) if "FE.W1_0" in IN0 else IN0["FE"], wc)
    W = {n: IN0[n] for n in ("Wq", "Wv", "Tb", "Wo", "bo", "W1", "b1", "W2", "b2", "Wh", "bh")}
    sh = np.asarray(params.SH)[k.par][0]
    out, aux = run_stage("IN0", {"x": x0, "hold": feats["hold"]}, W, consts, wc, (int(sh[0]), int(sh[1])),
                         trace=trace)
    res = dict(feats, x0=x0, fe=fe, k=k, aux=aux)
    res.update(out)
    return res


# ---------------------------------------------------------------------------------------------------------------
# lifecycle connection (pages 52.8, 53.2 step 5)

def params_from_plan(plan, *, exposure, SH, mv_scale_raw, mv_jittered=False, hw_depth=True, depth_inverted=False,
                     nrm_dec=False, rgh_packed=False, specular_from_v=False):
    """FrameParams of one evaluate from the lifecycle's FramePlan (reference/lifecycle.py): render / output size,
    parity, counter, reset, pj, the current and the effective previous camera (CP, already the current pair on reset
    frames), and the exposure pass's E, Ep, X3."""
    m = lambda t: np.asarray(t, np.float32).reshape(4, 4)
    cur, eff = plan.current_camera, plan.effective_camera
    return FrameParams(
        render_size=plan.render_size, output_size=plan.output_size,
        jitter=tuple(F.f32(v) for v in plan.current_jitter), prev_jitter=tuple(F.f32(v) for v in plan.previous_jitter),
        mv_scale_raw=mv_scale_raw, mv_jittered=mv_jittered, E=exposure.E, Ep=exposure.Ep, X3=exposure.X3,
        reset=plan.reset, par=plan.parity, fc=plan.counter, SH=SH, hw_depth=hw_depth, depth_inverted=depth_inverted,
        nrm_dec=nrm_dec, rgh_packed=rgh_packed, V2C=m(cur.view_to_clip), W2V=m(cur.world_to_view),
        V2Cp=m(eff.view_to_clip), W2Vp=m(eff.world_to_view), specular_from_v=specular_from_v)


# ---------------------------------------------------------------------------------------------------------------
# 52.8 semantic stage-call preparation: interface buffer types (page 62.2 ids) -> page-52 roles and flags

NRM_DEC_FORMATS = ("R8_UNORM", "R8G8_UNORM", "R8G8B8A8_UNORM", "R10G10B10A2_UNORM", "R16G16B16A16_UNORM")


def nrm_dec_for(format_name):
    """52.8: nrmDec = 1 precisely for 8-bit unorm normals with 1, 2 or 4 channels, 10-10-10-2 unorm and 16-bit
    four-channel unorm; BGRA8 unorm, float, snorm (and sRGB, Q52-13) are not decoded."""
    return format_name in NRM_DEC_FORMATS


class MissingInput(ValueError):
    """A required input is missing (the evaluate fails before any pass, 61.4 step 2)."""


def prepare_roles(tags, *, packing, normals_format):
    """tags: {buffer type id: Texture with the tag's own origin}. packing: 0 separate normals / roughness, 1 normals
    with roughness in the 4th channel (creation decision). Returns (textures by page-52 role, flags dict)."""
    def need(t):
        if t not in tags:
            raise MissingInput(t)
        return tags[t]
    roles = {"C": need(3), "V": need(1), "Dd": need(7), "Ds": need(8)}
    if 49 in tags:
        roles["Z"], hw = tags[49], False                  # linear depth takes precedence
    elif 0 in tags:
        roles["Z"], hw = tags[0], True
    else:
        raise MissingInput("depth")
    if packing == 1:
        roles["N"] = need(14)
    else:
        roles["N"], roles["R"] = need(5), need(6)
    for t, r in ((10, "SV"), (42, "Hd"), (11, "Q"), (40, "B"), (64, "G2")):
        if t in tags:
            roles[r] = tags[t]
    if 41 in tags:                                         # colour before fog replaces the SSS guide
        roles["G1"] = tags[41]
    elif 58 in tags:
        roles["G1"] = tags[58]
    flags = dict(hw_depth=hw, rgh_packed=packing == 1, nrm_dec=nrm_dec_for(normals_format),
                 smv=("SV" in roles or "Hd" in roles), dis="Q" in roles)
    return roles, flags
