# SPDX-License-Identifier: GPL-3.0-only
"""Frame generation (issue #96): the Full path's per-call stages (presets A and B) outside the two networks, as a CPU
reference: the Full blend with net2's three input tensors, the output pull / pull-middle / push / push-fine pyramid
that makes the generated frame, and the optional distortion stages.

Written for this project from the spec side's part 3 (Full path). The D3D12 kernels clean/dx12/kernels/fg_full_blend,
fg_out_pull, fg_out_push and fg_distortion compute the same bits (clean/fg/runner/fg_full_stages_d3d12_check.py).

Arithmetic: single precision with subnormals flushed to zero (clean/rr_f/reference/f32.py: one IEEE operation per step,
round to nearest even, flush before and after). fma rounds once. The original's approximate division, reciprocal and
2^x are taken correctly rounded here (the GPU kernels use the exact division wg_div and, for the logistic of net2's
weight channel, a table of this module's values over all 65,536 float16 inputs). sat(x) = min(max(x, 0), 1) with IEEE
minNum / maxNum (NaN gives 0). h(x) = float16, round to nearest even. Comparisons are ordered unless a step says
"unordered" (NaN passes the test).

Storage: colour, UI, X0 / X1, the distortion pair and the output pyramid levels hold 4 x float16; the blend weights W,
the quality image Qc, the previous -> current field F and the two distorted fields 2 x float16; Xw and the fine flag
image one float16 (0 or 1); the processed motion Mv 2 x float32; packed depths 32-bit words; MA / MB 64-bit words (key
in the high word, h(vx) in bits 31..16 and h(vy) in bits 15..0, empty = 0x7FFFFFFF7FFFFFFF). Images are numpy arrays
[H, W, C] (float32 holding the stored values) or [H, W] for one channel.

Reads ("mode L" and "mode N"): all reads are computed by hand, so that the GPU kernels reproduce them bit for bit.
- Bilinear at texel position (tx, ty): i = floor(tx), f = tx - i (per axis); taps i and i + 1 clamped to the image;
  weights w00 = (1 - fx)(1 - fy), w10 = fx (1 - fy), w01 = fy (1 - fx), w11 = fx fy; value = fma(w11, c11, fma(w01,
  c01, fma(w00, c00, w10 c10))). A read at normalized (u, v) of a W x H image (mode L) uses tx = fma(u, W, -0.5),
  ty = fma(v, H, -0.5). A NaN position gives taps (0, 1) and a NaN value; an infinite one the edge taps and NaN.
- Nearest at normalized (u, v) (mode N): i = floor(u W) clamped to [0, W - 1] (NaN -> 0).
- Float to integer conversions truncate and saturate, NaN gives 0 (as the GPU's conversion).

Grids: frame Wo x Ho (u, v = sat((x + 0.5) / Wo) per pixel); scatter grid W1 x H1 (MA, MB, W, Mv, F, packed depth,
distortion images); net1 image grid Wd x Hd (Qc); net2's grid W2 x H2 = Wp / k x Hp / k on the padded frame
Wp = 16k ceil(Wo / 16k), Hp likewise, k in {1, 2, 4, 8, 16}.

Sums over blocks (net2's inputs, the pyramid levels): a pairwise tree, x before y at every scale: a 2^n x 2^n block sum
is (S(0, 0) + S(h, 0)) + (S(0, h) + S(h, h)) of its four 2^(n-1) x 2^(n-1) quarters (h = 2^(n-1)), which is the order
of a lane tree over Morton-ordered pixels with strides 1, 2, 4, ... (the GPU kernels do exactly that).

Output pyramid: level l is ceil(Wo / 2^l) x ceil(Ho / 2^l). The level count n_levels is the halving chain of the frame
(level 0 included), continued until it has at least 9 levels and both extents are <= 1, at most 12. Pull writes levels
1..4 from the frame's pixels; the middle pull writes levels 5..8 from level 4 when n_levels > 9. Push fills from the
coarsest written level (8 or 4) down to level 1; each filled level is stored (4 x float16) in its own buffer. The
coarsest level is taken as written (the original's tile-local fallback for an empty coarsest element of a 16 x 16 tile
is not reproduced). Push-fine writes the upsample of filled level 1 to every hole pixel of the frame.
"""
import numpy as np

from clean.rr_f.reference import f32 as F

EMPTY = 0x7FFFFFFF7FFFFFFF
NEG_TIE = np.float32(-2.0 ** -23)                 # the index tie-break, -2^-23
UI_EPS = np.float32(2.0 ** -23)                   # 0x34000000
NEG_LOG2E = np.float32(-1.44269502)               # 0xBFB8AA3B
FLOOR = np.float32(1e-12)
TAPS = (np.float32(0.5625), np.float32(0.1875), np.float32(0.1875), np.float32(0.0625))
K_VALUES = (1, 2, 4, 8, 16)
ONE = np.float32(1.0)
ZERO = np.float32(0.0)


# ---- scalar helpers -----------------------------------------------------------------------------------------------

def f32(x):
    return F.f32(x)


def h(x):
    """Round single precision values to float16 (nearest even, subnormals kept) and widen exactly."""
    with np.errstate(over="ignore", invalid="ignore"):
        return np.asarray(x, np.float32).astype(np.float16).astype(np.float32)


def h_bits(x):
    with np.errstate(over="ignore", invalid="ignore"):
        return np.asarray(x, np.float32).astype(np.float16).view(np.uint16)


def from_bits16(b):
    return np.asarray(b, np.uint16).view(np.float16).astype(np.float32)


def sat(x):
    return F.fmin(F.fmax(x, ZERO), ONE)


def f2i(x):
    """Float to int as the GPU converts: truncate, saturate to int32, NaN -> 0."""
    x = np.asarray(x, np.float64)
    with np.errstate(invalid="ignore"):
        return np.where(np.isnan(x), 0, np.clip(np.trunc(np.nan_to_num(x, nan=0.0)), -2.0 ** 31, 2.0 ** 31 - 1)
                        ).astype(np.int64)


def floor_i(x):
    return f2i(np.floor(f32(x)))


def mul_up(a, b):
    """a b (single precision operands) rounded toward +infinity."""
    p = np.float64(np.float32(a)) * np.float64(np.float32(b))
    r = np.float32(p)
    return np.nextafter(r, np.float32(np.inf)) if np.float64(r) < p else r


def mul_down(a, b):
    p = np.float64(np.float32(a)) * np.float64(np.float32(b))
    r = np.float32(p)
    return np.nextafter(r, np.float32(-np.inf)) if np.float64(r) > p else r


def centres(n):
    """sat((i + 0.5) / n) for i < n."""
    return sat(F.div(np.arange(n, dtype=np.float32) + np.float32(0.5), np.float32(n)))


def uv_grid(W, H):
    u, v = centres(W), centres(H)
    return np.broadcast_to(u[None, :], (H, W)), np.broadcast_to(v[:, None], (H, W))


def padded(Wo, Ho, k):
    if k not in K_VALUES:
        raise ValueError("k must be one of %r" % (K_VALUES,))
    return 16 * k * -(-Wo // (16 * k)), 16 * k * -(-Ho // (16 * k))


def n_levels(Wo, Ho):
    """Levels of the frame's halving chain (level 0 = the frame): until at least 9 levels exist and both extents are
    <= 1, at most 12."""
    w, h_, n = Wo, Ho, 1
    while n < 12 and (n < 9 or w > 1 or h_ > 1):
        w, h_, n = -(-w // 2), -(-h_ // 2), n + 1
    return n


def level_size(Wo, Ho, l):
    return -(-Wo // (1 << l)), -(-Ho // (1 << l))


def coarsest_level(Wo, Ho):
    return 8 if n_levels(Wo, Ho) > 9 else 4


# ---- reads --------------------------------------------------------------------------------------------------------

def bilinear_px(img, tx, ty):
    """Bilinear read at texel positions (tx, ty) (arrays of one shape) of img [H, W, C] -> [..., C]."""
    H, W = img.shape[:2]
    tx, ty = f32(tx), f32(ty)
    x0f, y0f = np.floor(tx), np.floor(ty)
    fx, fy = F.sub(tx, x0f), F.sub(ty, y0f)
    xi, yi = f2i(x0f), f2i(y0f)
    x0, x1 = np.clip(xi, 0, W - 1), np.clip(xi, -1, W - 2) + 1
    y0, y1 = np.clip(yi, 0, H - 1), np.clip(yi, -1, H - 2) + 1
    gx, gy = F.sub(ONE, fx), F.sub(ONE, fy)
    w00, w10, w01, w11 = F.mul(gx, gy), F.mul(fx, gy), F.mul(fy, gx), F.mul(fx, fy)
    c00, c10, c01, c11 = img[y0, x0], img[y0, x1], img[y1, x0], img[y1, x1]
    e = (Ellipsis, None)
    return F.fma(w11[e], c11, F.fma(w01[e], c01, F.fma(w00[e], c00, F.mul(w10[e], c10))))


def tex_l(img, u, v):
    """Mode L: bilinear read at normalized (u, v)."""
    H, W = img.shape[:2]
    return bilinear_px(img, F.fma(u, np.float32(W), np.float32(-0.5)), F.fma(v, np.float32(H), np.float32(-0.5)))


def nearest_index(u, n):
    return np.clip(floor_i(F.mul(u, np.float32(n))), 0, n - 1)


def tex_n(img, u, v):
    """Mode N: nearest read at normalized (u, v)."""
    H, W = img.shape[:2]
    return img[nearest_index(v, H), nearest_index(u, W)]


def unpack_mv(words):
    """MA / MB words -> (vector [..., 2] float32, empty mask)."""
    w = np.asarray(words, np.uint64)
    empty = w == np.uint64(EMPTY)
    vx = from_bits16(((w >> np.uint64(16)) & np.uint64(0xFFFF)).astype(np.uint16))
    vy = from_bits16((w & np.uint64(0xFFFF)).astype(np.uint16))
    vec = np.stack([vx, vy], -1)
    return np.where(empty[..., None], ZERO, vec).astype(np.float32), empty


def pack_mv(key, vx, vy):
    """A landed MA / MB word from a float32 key and the vector (rounded to float16)."""
    kb = np.asarray(key, np.float32).view(np.uint32).astype(np.uint64)
    return (kb << np.uint64(32)) | (h_bits(vx).astype(np.uint64) << np.uint64(16)) | h_bits(vy).astype(np.uint64)


def block_sums(a, k):
    """Pairwise-tree sums of k x k blocks of a [H, W, ...] (H, W multiples of k), x before y at every scale."""
    a = f32(a)
    while k > 1:
        a = F.add(a[:, 0::2], a[:, 1::2])
        a = F.add(a[0::2], a[1::2])
        k //= 2
    return a


# ---- G4-Full: the Full blend ----------------------------------------------------------------------------------------

BAND = 64            # rows per band of the per-pixel stages (memory bound only; the results do not depend on it)


def _blend_band(Cp, Cc, Wimg, Qc, MA, MB, ui, y0, y1):
    """The blend's per-pixel values for frame rows y0 .. y1 - 1: x0, x1 [rows, Wo, 4] and hole [rows, Wo]."""
    Ho, Wo = Cp.shape[:2]
    H1, W1 = MA.shape
    u = np.broadcast_to(centres(Wo)[None, :], (y1 - y0, Wo))
    v = np.broadcast_to(centres(Ho)[y0:y1, None], (y1 - y0, Wo))
    ix = np.clip(f2i(np.rint(F.add(F.fma(u, np.float32(W1), np.float32(-0.5)), NEG_TIE))), 0, W1 - 1)
    iy = np.clip(f2i(np.rint(F.add(F.fma(v, np.float32(H1), np.float32(-0.5)), NEG_TIE))), 0, H1 - 1)
    a, emA = unpack_mv(MA[iy, ix])
    b, emB = unpack_mv(MB[iy, ix])
    eA, eB = emA.astype(np.float32)[..., None], emB.astype(np.float32)[..., None]
    ub, vb = F.add(u, b[..., 0]), F.add(v, b[..., 1])
    ua, va = F.add(u, a[..., 0]), F.add(v, a[..., 1])
    P0, Q0 = Cp[y0:y1, :, :3], Cc[y0:y1, :, :3]
    cP = tex_l(Cp, ub, vb)[..., :3]
    cQ = tex_l(Cc, ua, va)[..., :3]
    w = tex_l(Wimg, u, v)
    wx, wy = w[..., 0:1], w[..., 1:2]
    qu, qb, qa = tex_l(Qc, u, v), tex_l(Qc, ub, vb), tex_l(Qc, ua, va)
    x0rgb = F.fma(wx, F.sub(F.fma(eB, F.sub(cQ, cP), cP), P0), P0)
    x0a = F.fma(w[..., 0], F.sub(qb[..., 0], qu[..., 0]), qu[..., 0])
    x1rgb = F.fma(wy, F.sub(F.fma(eA, F.sub(cP, cQ), cQ), Q0), Q0)
    x1a = F.fma(w[..., 1], F.sub(qa[..., 1], qu[..., 1]), qu[..., 1])
    with np.errstate(invalid="ignore"):
        hole = emA & emB & (~(w[..., 0] < 0.5) | ~(w[..., 1] < 0.5))
    if ui is not None:
        Up, Uc = f32(ui[0][y0:y1]), f32(ui[1][y0:y1])
        x0rgb = F.fma(x0rgb, F.sub(ONE, Up[..., 3:4]), Up[..., :3])
        x1rgb = F.fma(x1rgb, F.sub(ONE, Uc[..., 3:4]), Uc[..., :3])
        ad = [np.abs(F.sub(Up[..., c], Uc[..., c])).astype(np.float32) for c in range(3)]
        e = sat(F.div(F.add(ad[2], F.add(ad[0], ad[1])), np.float32(3.0)))
        with np.errstate(invalid="ignore"):
            x0a = np.where(Up[..., 3] > UI_EPS, e, x0a).astype(np.float32)
            x1a = np.where(Uc[..., 3] > UI_EPS, e, x1a).astype(np.float32)
            hole = hole & (Up[..., 3] < UI_EPS) & (Uc[..., 3] < UI_EPS)
    return (np.concatenate([x0rgb, x0a[..., None]], -1).astype(np.float32),
            np.concatenate([x1rgb, x1a[..., None]], -1).astype(np.float32), hole)


def g4_full_blend(Cp, Cc, Wimg, Qc, MA, MB, *, k, ui=None):
    """The Full blend. Cp, Cc [Ho, Wo, 4]; Wimg [H1, W1, 2]; Qc [Hd, Wd, 2]; MA, MB [H1, W1] (64-bit words); k: net2's
    downsample factor; ui: None (preset A) or (Up, Uc) [Ho, Wo, 4] (preset B, UI recomposition).
    Returns X0, X1 [Ho, Wo, 4], Xw [Ho, Wo] (stored values) and net2's inputs Ib, Ic [H2, W2, 4], Ia [H2, W2] (float16
    values), plus the single-precision per-pixel values ("x0", "x1", "hole") the tensors are averaged from."""
    Cp, Cc, Wimg, Qc = f32(Cp), f32(Cc), f32(Wimg), f32(Qc)
    Ho, Wo = Cp.shape[:2]
    x0 = np.empty((Ho, Wo, 4), np.float32)
    x1 = np.empty((Ho, Wo, 4), np.float32)
    hole = np.empty((Ho, Wo), bool)
    for y0 in range(0, Ho, BAND):
        y1 = min(Ho, y0 + BAND)
        x0[y0:y1], x1[y0:y1], hole[y0:y1] = _blend_band(Cp, Cc, Wimg, Qc, MA, MB, ui, y0, y1)
    Wp, Hp = padded(Wo, Ho, k)
    vals = np.zeros((Hp, Wp, 9), np.float32)
    vals[:Ho, :Wo, 0:4], vals[:Ho, :Wo, 4:8], vals[:Ho, :Wo, 8] = x0, x1, hole
    s = block_sums(vals, k)
    kk = np.float32(k * k)
    return {"X0": h(x0), "X1": h(x1), "Xw": hole.astype(np.float32),
            "Ib": h(F.div(s[..., 0:4], kk)), "Ic": h(F.div(s[..., 4:8], kk)),
            "Ia": (s[..., 8] > 0).astype(np.float32),
            "x0": x0, "x1": x1, "hole": hole}


# ---- G6-Full: output pull, pull-middle, push, push-fine ----------------------------------------------------------

def sig(z):
    """Logistic of net2's weight channel as the pull computes it per tap: 1 / (2^(z * -1.44269502) + 1)."""
    return F.rcp(F.add(F.exp2(F.mul(f32(z), NEG_LOG2E)), ONE))


def sig_table():
    """sig of every float16 bit pattern (index = the bits), float32: the table the GPU pull reads."""
    with np.errstate(over="ignore", invalid="ignore"):
        return sig(from_bits16(np.arange(65536, dtype=np.uint32).astype(np.uint16)))


def hole_rule(need):
    """need [H, W] bool -> hole [H, W]: every 3 x 3 window containing the pixel holds a pixel with need; outside the
    image counts as need."""
    H, W = need.shape
    p = np.ones((H + 4, W + 4), bool)
    p[2:-2, 2:-2] = need
    win = np.zeros((H + 2, W + 2), bool)                # any need in the 3 x 3 window centred at (y - 1, x - 1)
    for dy in range(3):
        for dx in range(3):
            win |= p[dy:dy + H + 2, dx:dx + W + 2]
    out = np.ones((H, W), bool)
    for dy in range(3):
        for dx in range(3):
            out &= win[dy:dy + H, dx:dx + W]
    return out


def n2_sample(N2, Wo, Ho, k, table=None, rows=None):
    """net2's output at each frame pixel: sigma [Ho, Wo], vA, vB [Ho, Wo, 2] (rows = (y0, y1): those rows only)."""
    N2 = f32(N2)
    H2, W2 = N2.shape[:2]
    y0, y1 = (0, Ho) if rows is None else rows
    rk = F.rcp(np.float32(k))
    s = F.fmin(F.mul(np.arange(Wo, dtype=np.float32), rk), np.float32(W2 - 1))
    t = F.fmin(F.mul(np.arange(y0, y1, dtype=np.float32), rk), np.float32(H2 - 1))
    i0 = floor_i(s)
    j0 = floor_i(t)
    i1, j1 = np.minimum(i0 + 1, W2 - 1), np.minimum(j0 + 1, H2 - 1)
    fx = F.sub(s, np.floor(s))[None, :, None]
    fy = F.sub(t, np.floor(t))[:, None, None]
    gx, gy = F.sub(ONE, fx), F.sub(ONE, fy)
    q = N2[:, :, :5].copy()
    tab = sig_table() if table is None else table
    q[..., 4] = tab[h_bits(q[..., 4])]
    q00, q10 = q[j0][:, i0], q[j0][:, i1]
    q01, q11 = q[j1][:, i0], q[j1][:, i1]
    r = F.fma(gy, F.fma(gx, q00, F.mul(fx, q10)), F.mul(fy, F.fma(gx, q01, F.mul(fx, q11))))
    return r[..., 4], r[..., 0:2], r[..., 2:4]


def g6_output_pull(X0, X1, Xw, N2, *, k, table=None):
    """Output pull. X0, X1 [Ho, Wo, 4], Xw [Ho, Wo] (stored values); N2 [H2, W2, 8].
    Returns "out" [Ho, Wo, 4] (the frame before push-fine), "fine" [Ho, Wo] (1 = hole), levels {1..4: [h, w, 4]}
    (float16 values) and the per-pixel colour "c" [Ho, Wo, 3] and weight "w" (single precision)."""
    X0, X1, Xw = f32(X0), f32(X1), f32(Xw)
    Ho, Wo = Xw.shape
    with np.errstate(invalid="ignore"):
        hole = hole_rule(Xw > 0)
    tab = sig_table() if table is None else table
    kf = np.float32(k)
    c = np.empty((Ho, Wo, 3), np.float32)
    for y0 in range(0, Ho, BAND):
        y1 = min(Ho, y0 + BAND)
        sigma, vA, vB = n2_sample(N2, Wo, Ho, k, tab, (y0, y1))
        xs = np.broadcast_to(np.arange(Wo, dtype=np.float32)[None, :], (y1 - y0, Wo))
        ys = np.broadcast_to(np.arange(y0, y1, dtype=np.float32)[:, None], (y1 - y0, Wo))
        A = bilinear_px(X0[..., :3], F.add(F.mul(vA[..., 0], kf), xs), F.add(F.mul(vA[..., 1], kf), ys))
        B = bilinear_px(X1[..., :3], F.add(F.mul(vB[..., 0], kf), xs), F.add(F.mul(vB[..., 1], kf), ys))
        sg = sigma[..., None]
        cb = sat(F.fma(sg, A, F.mul(F.sub(ONE, sg), B)))
        with np.errstate(invalid="ignore"):
            bad = np.isnan(np.abs(F.mul(cb[..., 2], F.mul(cb[..., 0], cb[..., 1]))))
        cb = np.where(bad[..., None], np.array([0, 1, 1], np.float32), cb)
        c[y0:y1] = np.where(hole[y0:y1, :, None], ZERO, cb)
    wt = (~hole).astype(np.float32)
    out = np.concatenate([h(c), np.ones((Ho, Wo, 1), np.float32)], -1)
    # levels 1..4 from 16 x 16 tiles
    T = 16
    Wt, Ht = T * -(-Wo // T), T * -(-Ho // T)
    base = np.zeros((Ht, Wt, 4), np.float32)
    base[:Ho, :Wo, :3], base[:Ho, :Wo, 3] = c, wt
    levels = {}
    s = base
    for l in range(1, 5):
        s = F.add(s[:, 0::2], s[:, 1::2])
        s = F.add(s[0::2], s[1::2])
        lw, lh = level_size(Wo, Ho, l)
        levels[l] = _level_value(s[:lh, :lw], 4 ** l)
    return {"out": out, "fine": hole.astype(np.float32), "levels": levels, "c": c, "w": wt}


def _level_value(s, n):
    """(sum r, sum g, sum b, sum w) -> (r / max(sum w, 1e-12), ..., sum w / n), float16 values."""
    den = F.fmax(s[..., 3:4], FLOOR)
    return h(np.concatenate([F.div(s[..., :3], den), F.div(s[..., 3:4], np.float32(n))], -1))


def g6_output_pull_mid(L4, Wo, Ho):
    """Middle pull: levels 5..8 from level 4's stored values (only when n_levels(Wo, Ho) > 9)."""
    if n_levels(Wo, Ho) <= 9:
        return {}
    L4 = f32(L4)
    h4, w4 = L4.shape[:2]
    T = 16
    base = np.zeros((T * -(-h4 // T), T * -(-w4 // T), 4), np.float32)
    base[:h4, :w4, :3] = F.mul(L4[..., :3], L4[..., 3:4])
    base[:h4, :w4, 3] = L4[..., 3]
    out, s = {}, base
    for l in range(5, 9):
        s = F.add(s[:, 0::2], s[:, 1::2])
        s = F.add(s[0::2], s[1::2])
        lw, lh = level_size(Wo, Ho, l)
        out[l] = _level_value(s[:lh, :lw], 4 ** (l - 4))
    return out


def upsample(coarse, wf, hf, rows=None):
    """The 9/3/3/1 upsample of a filled level [hc, wc, 4] to wf x hf: (v_up rgb, w_up) [hf, wf, 4], single
    precision (v_up normalized by clamp(Wsum, 1e-12, 1), w_up = Wsum unclamped); rows = (y0, y1): those rows only."""
    coarse = f32(coarse)
    hc, wc = coarse.shape[:2]
    qx, qy = np.arange(wf), np.arange(*((0, hf) if rows is None else rows))
    px, py = qx >> 1, qy >> 1
    nx = np.clip(px + np.where(qx & 1, 1, -1), 0, wc - 1)
    ny = np.clip(py + np.where(qy & 1, 1, -1), 0, hc - 1)
    eP = coarse[py][:, px]
    eX = coarse[py][:, nx]
    eY = coarse[ny][:, px]
    eD = coarse[ny][:, nx]

    def chain(p, x, y, d):
        return F.fma(d, TAPS[3], F.fma(y, TAPS[2], F.fma(x, TAPS[1], F.fma(p, TAPS[0], ZERO))))
    ws = chain(eP[..., 3], eX[..., 3], eY[..., 3], eD[..., 3])
    vs = chain(*(F.mul(e[..., :3], e[..., 3:4]) for e in (eP, eX, eY, eD)))
    den = F.fmin(F.fmax(ws, FLOOR), ONE)[..., None]
    return np.concatenate([F.div(vs, den), ws[..., None]], -1).astype(np.float32)


def g6_output_push(levels, Wo, Ho):
    """Push: filled levels {l: [h, w, 4]} from the coarsest written level down to level 1 (float16 values)."""
    Lc = coarsest_level(Wo, Ho)
    filled = {Lc: f32(levels[Lc])}
    for l in range(Lc - 1, 0, -1):
        own = f32(levels[l])
        lw, lh = level_size(Wo, Ho, l)
        up = h(upsample(filled[l + 1], lw, lh))
        with np.errstate(invalid="ignore"):
            keep = ~(own[..., 3:4] <= 0)                # own weight > 0, unordered
        filled[l] = np.where(keep, own, up).astype(np.float32)
    return filled


def g6_output_push_fine(out, fine, filled1):
    """Push-fine: every pixel with fine > 0 gets (h(v_up rgb), 1) from filled level 1; the others keep `out`."""
    out = f32(out).copy()
    Ho, Wo = fine.shape
    for y0 in range(0, Ho, BAND):
        y1 = min(Ho, y0 + BAND)
        up = upsample(filled1, Wo, Ho, (y0, y1))
        m = fine[y0:y1] > 0
        band = out[y0:y1]
        band[m, :3] = h(up[m, :3])
        band[m, 3] = 1.0
    return out


def g6_output(X0, X1, Xw, N2, *, k, table=None):
    """The whole output pyramid: the Full generated frame [Ho, Wo, 4] and every intermediate buffer."""
    Ho, Wo = f32(Xw).shape
    p = g6_output_pull(X0, X1, Xw, N2, k=k, table=table)
    levels = dict(p["levels"])
    levels.update(g6_output_pull_mid(levels[4], Wo, Ho))
    filled = g6_output_push(levels, Wo, Ho)
    final = g6_output_push_fine(p["out"], p["fine"], filled[1])
    return {"pull": p, "levels": levels, "filled": filled, "frame": final}


# ---- R5: distortion ------------------------------------------------------------------------------------------------

class Region:
    """A region-mapped read of an application image: texture tw x th, sub-rectangle (bx, by, ew, eh)."""

    def __init__(self, tw, th, bx=0, by=0, ew=None, eh=None):
        self.tw, self.th, self.bx, self.by = tw, th, bx, by
        self.ew, self.eh = (tw if ew is None else ew), (th if eh is None else eh)
        self.rx, self.ry = F.div(ONE, np.float32(tw)), F.div(ONE, np.float32(th))
        self.lox = mul_up(np.float32(bx + 0.5), self.rx)
        self.hix = mul_down(np.float32(bx + self.ew - 0.5), self.rx)
        self.loy = mul_up(np.float32(by + 0.5), self.ry)
        self.hiy = mul_down(np.float32(by + self.eh - 0.5), self.ry)

    def coords(self, u, v):
        X = F.mul(self.rx, F.fma(u, np.float32(self.ew), np.float32(self.bx)))
        Y = F.mul(self.ry, F.fma(v, np.float32(self.eh), np.float32(self.by)))
        return F.fmin(F.fmax(X, self.lox), self.hix), F.fmin(F.fmax(Y, self.loy), self.hiy)

    def args(self):
        """Kernel arguments: tw, th, bx, by, ew, eh as integers; rx, ry, lox, hix, loy, hiy as float bits."""
        return [self.tw, self.th, self.bx, self.by, self.ew, self.eh] + \
            [int(np.float32(x).view(np.uint32)) for x in (self.rx, self.ry, self.lox, self.hix, self.loy, self.hiy)]


def r5_downsample(field, region, W1, H1, lowp=None):
    """Downsample distortion: field [th, tw, 4] (application values), region: Region; lowp: None or (bias, scale).
    Returns [H1, W1, 4] float16 values: fma(scale, field, bias) per channel ((0, 1) without lowp)."""
    field = f32(field)
    u, v = uv_grid(W1, H1)
    X, Y = region.coords(u, v)
    f = field[nearest_index(Y, region.th), nearest_index(X, region.tw)]
    b, s = (ZERO, ONE) if lowp is None else (np.float32(lowp[0]), np.float32(lowp[1]))
    return h(F.fma(s, f, b))


def r5_pair(down, previous, was_active):
    """The swap pair: (current, previous) downsampled fields; the previous one is (0, 0, 0, 0) when distortion was not
    active in the previous frame."""
    prev = np.zeros_like(down) if (previous is None or not was_active) else previous
    return down, prev


def r5_fields(Dc, Dp, Zc, Mv, Fpc):
    """Distortion: Dc, Dp [H1, W1, 4] (downsampled current / previous fields); Zc [H1, W1] packed depth (uint32);
    Mv [H1, W1, 2] float32 (normalized); Fpc [H1, W1, 2] (previous -> current, pixels).
    Returns the distorted packed depth (uint32), the distorted current -> previous field (normalized) and the distorted
    previous -> current field (pixels), both float16 values."""
    Dc, Dp, Mv, Fpc = f32(Dc), f32(Dp), f32(Mv), f32(Fpc)
    H1, W1 = Zc.shape
    u, v = uv_grid(W1, H1)
    W1f, H1f = np.float32(W1), np.float32(H1)
    dcu = tex_l(Dc, u, v)
    p1x, p1y = F.add(u, dcu[..., 0]), F.add(v, dcu[..., 1])
    iy, ix = nearest_index(p1y, H1), nearest_index(p1x, W1)
    zd = np.asarray(Zc, np.uint32)[iy, ix]
    m = Mv[iy, ix]
    p2x, p2y = F.add(p1x, m[..., 0]), F.add(p1y, m[..., 1])
    q = tex_l(Dp, p2x, p2y)
    mcp = h(np.stack([F.sub(F.add(p2x, q[..., 2]), u), F.sub(F.add(p2y, q[..., 3]), v)], -1))
    dpu = tex_l(Dp, u, v)
    rx, ry = F.add(u, dpu[..., 0]), F.add(v, dpu[..., 1])
    f = tex_n(Fpc, rx, ry)
    sx, sy = F.add(rx, F.div(f[..., 0], W1f)), F.add(ry, F.div(f[..., 1], H1f))
    dcs = tex_l(Dc, sx, sy)
    fpc = h(np.stack([F.mul(F.sub(F.add(sx, dcs[..., 2]), u), W1f),
                      F.mul(F.sub(F.add(sy, dcs[..., 3]), v), H1f)], -1))
    return {"Zd": zd, "Mcp": mcp, "Fpc": fpc}


def net2_k(Ho):
    """net2's downsample factor for a frame height: the smallest power of two with floor(Ho / k) <= 480."""
    k = 1
    while Ho // k > 480:
        k *= 2
    return k


__all__ = ["g4_full_blend", "g6_output_pull", "g6_output_pull_mid", "g6_output_push", "g6_output_push_fine",
           "g6_output", "r5_downsample", "r5_pair", "r5_fields", "Region", "sig", "sig_table", "hole_rule",
           "n2_sample", "upsample", "block_sums", "tex_l", "tex_n", "bilinear_px", "pack_mv", "unpack_mv",
           "n_levels", "level_size", "coarsest_level", "padded", "net2_k", "h", "h_bits", "from_bits16"]
