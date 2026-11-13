# SPDX-License-Identifier: MIT
"""Generator of the exact q8(GELU) lookup (clean/dx12/hlsl/wg_gelu_lut.hlsli and its 64 KB table).

    python clean/dx12/gelu_lut/gen.py --rle <oracle rle.bin> --out <dir>

Input: the run-length form of HIP's function h -> byte (gelu_lut/oracle.hip via run.py: wg_gelu mode 2 + wg_pack_q8_nc,
every FP32 bit pattern). The function is piecewise constant with 204 changes over all 2^32 inputs; between two changes
of the positive or negative range it is monotone, except at two places where the hardware exp / rcp make it flip for one
input: u = 0xbf6c7530 (a3 between a2's) and 0xbff0c85e (97 between 96's).

Lookup (per value h, 3 VALU slots instead of 3 + 2 transcendentals):
    w = fma(h, S, B)                      index transform (one dual-issued fmaak)
    i = f16_rtz(w) & ~7                    v_cvt_pk_rtz_f16_f32 of two values, then a mask: byte offset of an 8-byte entry
    (a, b) = table[i]                      one 8-byte load from the arena at WG_GELU_LUT_BASE + i
    y = fma(h, a, b);  byte = fp8(y)       v_fmac + v_cvt_pk_fp8_f32 (no clamp)
The f16 truncation gives 128 buckets per binade over the whole FP32 range (saturating, infinities and NaN kept); within a
bucket y is affine in h, so the byte changes at most once per bucket, at a place set exactly by (a, b). S and B are
chosen so that both one-input flips sit on bucket boundaries, which makes every bucket monotone with <= 1 change.
Constant buckets use a = 0 (h * 0 keeps the sign of h, + -0: +-0 bytes come out right, -inf and NaN give NaN = 0xff);
buckets that hold +inf, NaN or finite values beyond the f16 range use a = 1, b = 0 (y = h) on the positive side.

Outputs in --out: gelu_lut_table.bin (65536 bytes: 8192 x (a, b) float32), gelu_lut.json (S, B, statistics), and with
--hlsl / --cpp the header files (clean/dx12/hlsl/wg_gelu_lut.hlsli, clean/runtime/gelu_lut_table.h).
"""
import argparse
import json
import os
from fractions import Fraction

import numpy as np

K = 7                      # f16 mantissa bits kept: 128 buckets per binade
SHIFT = 10 - K             # entry = 2^SHIFT bytes = 8 (a, b)
NB = 1 << (16 - SHIFT)     # 8192 entries
J1 = (0xbf6c752f, 0xbf6c7530)   # one-input flips: a bucket boundary must fall between these two inputs
J2 = (0xbff0c85d, 0xbff0c85e)


def f32(u):
    return np.asarray(u, np.uint32).view(np.float32)


def u32(x):
    return np.asarray(x, np.float32).view(np.uint32)


def read_rle(path):
    d = np.fromfile(path, np.uint32)
    n = int(d[0])
    return d[1:1 + n].astype(np.int64), d[1 + n:1 + 2 * n].astype(np.int64)


def byte_of(pos, byt, u):
    return byt[np.searchsorted(pos, np.asarray(u, np.int64), side="right") - 1]


def fp8_hw(y):
    """v_cvt_pk_fp8_f32 of float32 values (wg_math.hlsli wg_e4m3_hw: RNE, |y| > 464 -> 0x7f | sign, NaN -> 0xff)."""
    b = u32(y).astype(np.int64)
    s = (b >> 24) & 0x80
    a = b & 0x7fffffff
    sub = np.round(np.minimum(f32(a.astype(np.uint32)).astype(np.float64), 0.015625) * 512.0).astype(np.int64)
    nrm = ((a + 0x7ffff + ((a >> 20) & 1)) >> 20) - (120 << 3)
    code = np.where(a < 0x3c800000, sub, nrm)
    code = np.where(a > 0x43e80000, 0x7f, code)
    return np.where(a > 0x7f800000, 0xff, s | code)


def fp8_value(c):
    em = c & 0x7f
    v = (1.0 + (em & 7) / 8.0) * 2.0 ** ((em >> 3) - 7) if em >= 8 else em / 512.0
    return -v if c & 0x80 else v


def f16_rtz_bits(w):
    """v_cvt_pk_rtz_f16_f32 of finite float32 values in the f16 normal range or beyond (the cases the generator relies
    on; subnormal / zero / non-finite w get robust entries and are proven by the exhaustive GPU check)."""
    b = u32(w).astype(np.int64)
    s = (b >> 31) & 1
    e = (b >> 23) & 0xff
    m = b & 0x7fffff
    out = np.where(e >= 143, 0x7bff, ((e - 112) << 10) | (m >> 13))
    out = np.where(e < 113, 0, out)   # f16 subnormal or zero: not used for decisions (see near_zero_window)
    return out | (s << 15)


def fma32(x, y, z):
    """float32 fma(x, y, z) exactly as v_fma_f32 (one RNE rounding; FP32 denormal results are not expected here):
    x * y is exact in float64; t = RN64(x y + z) with error err (two-sum); RN32(t + err) = RN32(t) unless t is a float32
    midpoint and err != 0, which the sign of err then decides."""
    x, y, z = (np.asarray(v, np.float32).astype(np.float64) for v in (x, y, z))
    p = x * y
    t = p + z
    bb = t - p
    with np.errstate(invalid="ignore"):
        err = (p - (t - bb)) + (z - bb)
    err = np.where(np.isfinite(err), err, 0.0)
    r = np.asarray(t.astype(np.float32))
    rd = r.astype(np.float64)
    up = np.nextafter(r, np.float32(np.inf))
    dn = np.nextafter(r, np.float32(-np.inf))
    with np.errstate(invalid="ignore"):
        r = np.where((err > 0) & (t - rd == (up.astype(np.float64) - rd) / 2), up, r)
        r = np.where((err < 0) & (rd - t == (rd - dn.astype(np.float64)) / 2), dn, r)
    return np.asarray(r, np.float32)


def w_of(h, S, B):
    return fma32(h, S, B)


def rtz32_of_fma(x, y, z):
    """float32 truncation (toward zero) of the exact x * y + z: the index of a driver that folds the fma into the
    f16 conversion (v_fma_mixlo_f16 under round-toward-zero, seen on this driver) instead of rounding it to float32."""
    x, y, z = (np.asarray(v, np.float32).astype(np.float64) for v in (x, y, z))
    p = x * y
    t = p + z
    bb = t - p
    with np.errstate(invalid="ignore"):
        err = (p - (t - bb)) + (z - bb)
    err = np.where(np.isfinite(err), err, 0.0)
    r = np.asarray(t.astype(np.float32))
    with np.errstate(invalid="ignore", over="ignore"):
        over = np.abs(r.astype(np.float64)) > np.abs(t)
        r = np.where(over, np.nextafter(r, np.float32(0)), r)
        exact = r.astype(np.float64) == t
        below = exact & (err != 0) & (np.sign(err) != np.sign(t))   # |x y + z| < |t| = |r|
        r = np.where(below, np.nextafter(r, np.float32(0)), r)
    return np.asarray(r, np.float32)


SEMANTICS = ("a", "b")   # a: f16_rtz(RN32(fma(h, S, B)))  b: f16_rtz(fma(h, S, B) exact): the table must hold for both


def bucket(h, S, B, sem="a"):
    w = w_of(h, S, B) if sem == "a" else rtz32_of_fma(h, S, B)
    return f16_rtz_bits(w) >> SHIFT


def candidates(K_=K):
    """(S, B) with B > 0 that put a bucket boundary inside both flips: |w(J?a)| < V <= |w(J?b)| for f16 grid values V."""
    S = np.arange(0x3f800000, 0x40000000, dtype=np.uint32).view(np.float32).astype(np.float64)
    ha, hb = f32(J1).astype(np.float64)
    pa, pb = -ha * S, -hb * S
    step = 2.0 ** (np.floor(np.log2(pa)) - K_)
    V = np.floor(pa / step) * step - step
    lo, hi = pa - V, pb - V                      # B in (lo, hi]
    Bm = hi.astype(np.float32).astype(np.float64)
    Bm = np.where(Bm > hi, np.nextafter(hi.astype(np.float32), np.float32(0)).astype(np.float64), Bm)
    ok = (Bm > lo) & (Bm <= hi)
    Sk, Bk = S[ok].astype(np.float32), Bm[ok].astype(np.float32)
    keep = np.ones(len(Sk), bool)
    for x in (J1, J2):   # exact check of both flips under both index semantics, vectorized over the candidates
        h = f32(x)
        for sem in SEMANTICS:
            ha_ = np.full(len(Sk), h[0], np.float32)
            hb_ = np.full(len(Sk), h[1], np.float32)
            keep &= bucket(ha_, Sk, Bk, sem) != bucket(hb_, Sk, Bk, sem)
    return [(float(a), float(b)) for a, b in zip(Sk[keep], Bk[keep])]


def w_vec(h, S, B):
    """w_of for one h over arrays of (S, B)."""
    return fma32(np.full(len(S), h, np.float32), S, B)


def changes(pos):
    t = pos[1:]
    t = t[(t != 0x80000000)]                       # +NaN -> -0: different signs, never one bucket here
    finite = (t & 0x7fffffff) < 0x7f800000
    return t[finite & (((t - 1) & 0x7fffffff) < 0x7f800000)]


def inside_changes(pos, S, B):
    """{bucket: set of changes u inside it} over both index semantics (u inside when u - 1 and u share the bucket)."""
    t = changes(pos)
    out = {}
    for sem in SEMANTICS:
        b0 = bucket(f32(t), S, B, sem)
        b1 = bucket(f32(t - 1), S, B, sem)
        for k, u in zip(b0[b0 == b1], t[b0 == b1]):
            out.setdefault(int(k), set()).add(int(u))
    return out


def internal_counts(pos, S, B):
    return {k: len(v) for k, v in inside_changes(pos, S, B).items()}


def y_of(h, a, b):
    return fma32(h, a, b)


def fit(hl, hr, cl, cr, hmin, hmax):
    """(a, b) float32 with fp8(fma(h, a, b)) = cl for h in [hmin, hl] and cr for h in [hr, hmax] (hl < hr adjacent
    floats, cl / cr adjacent codes). y and fp8 are monotone, so the four ends decide. Slopes a = k m / h put a h near the
    crossing value m, so that b stays small and can place the crossing to the ulp."""
    vl, vr = fp8_value(cl), fp8_value(cr)
    sgn = 1.0 if vr > vl else -1.0
    m = 0.5 * (vl + vr)
    hs = np.array([hmin, hl, hr, hmax], np.float32)
    want = np.array([cl, cl, cr, cr])
    for k in (2.0, 3.0, 4.0, 1.5, 6.0, 8.0, 1.25, 12.0, 16.0, 1.0, 24.0, 32.0, 0.75, 0.5):
        a = np.float32(sgn * k * abs(m) / abs(float(hr)))
        b0 = np.float32(m - float(a) * float(hr))
        bs = (u32(b0).astype(np.int64) + np.arange(-256, 257)) & 0xffffffff
        bs = f32(bs.astype(np.uint32))
        for b in bs:
            if not np.isfinite(b):
                continue
            if np.array_equal(fp8_hw(y_of(hs, a, b)), want):
                return float(a), float(b)
    return None


def f16_float(c):
    return float(np.array([c], np.uint16).view(np.float16)[0])


def build(pos, byt, S, B, zc):
    """The table: one (a, b) per bucket (f16 code >> 3 of w)."""
    table = np.zeros((NB, 2), np.float32)
    table[:, 1] = np.float32(-0.0)
    inner = internal_counts(pos, S, B)
    assert max(inner.values(), default=0) <= 1, "a bucket holds more than one change"
    change = {k: next(iter(v)) for k, v in inside_changes(pos, S, B).items()}
    zero = lambda c: (0.0, -0.0) if c in (0x00, 0x80) else ((1.0, 0.0) if c == 0x7f else (0.0, fp8_value(c)))
    kinds = {"pass": 0, "const": 0, "affine": 0, "zero_w": 0}
    for idx in range(NB):
        c16 = idx << SHIFT
        neg, mag = bool(c16 & 0x8000), c16 & 0x7fff
        if mag >= 0x7bf8:   # largest finite f16 bucket (RTZ saturation), infinities, NaN
            table[idx] = (0.0, -0.0) if neg else (1.0, 0.0)
            kinds["pass"] += 1
            continue
        if mag < 0x0400:    # w zero or f16 subnormal
            table[idx] = zero(zc)
            kinds["zero_w"] += 1
            continue
        v0, v1 = f16_float(mag), f16_float(mag + (1 << SHIFT))
        w_lo, w_hi = (-v1, -v0) if neg else (v0, v1)
        h_lo, h_hi = (w_lo - B) / S, (w_hi - B) / S
        pad = max(abs(h_lo), abs(h_hi)) * 2.0 ** -20
        hmin, hmax = np.float32(h_lo - pad), np.float32(h_hi + pad)
        hmid = np.float32(0.5 * (h_lo + h_hi))
        assert all(int(bucket(np.array([hmid]), S, B, sem)[0]) == idx for sem in SEMANTICS), hex(c16)
        if idx in change:
            tu = change[idx]
            h_t, h_p = f32(tu), f32(tu - 1)
            c_t, c_p = (int(x) for x in byte_of(pos, byt, [tu, tu - 1]))
            if h_t > h_p:
                ent = fit(h_p, h_t, c_p, c_t, hmin, hmax)
            else:
                ent = fit(h_t, h_p, c_t, c_p, hmin, hmax)
            assert ent is not None, ("no fit", hex(c16), hex(tu))
            table[idx] = ent
            kinds["affine"] += 1
        else:
            if h_lo < 0.0 < h_hi:   # the bucket of h = +-0 (w = B): -0 / +0 bytes
                c = int(byte_of(pos, byt, [u32(np.float32(h_hi * 0.5))])[0])
                assert c == 0x00 and int(byte_of(pos, byt, [u32(np.float32(h_lo * 0.5))])[0]) == 0x80
            else:
                c = int(byte_of(pos, byt, [u32(hmid)])[0])
            table[idx] = zero(c)
            kinds["const"] += 1
    return table, inner, kinds


def check_points(pos, byt, S, B, table):
    """Model check of the table at every change (u - 2 .. u + 2) and its bucket ends: the lookup as the GPU computes it
    (exact fma, f16 RTZ, power-of-two a) against HIP's byte. The exhaustive GPU run is the proof; this catches generator
    slips early."""
    us = np.unique(np.concatenate([pos + d for d in (-2, -1, 0, 1, 2)]) & 0xffffffff)
    us = us[(us & 0x7fffffff) < 0x7f800000]
    us = us[((us & 0x7f800000) != 0) | ((us & 0x7fffff) == 0)]   # FP32 denormals: flushed on D3D12, skipped here
    h = f32(us)
    want = byte_of(pos, byt, us)
    bad = []
    for sem in SEMANTICS:
        idx = bucket(h, S, B, sem)
        got = fp8_hw(y_of(h, table[idx, 0], table[idx, 1]))
        bad += [(sem, hex(int(us[i])), hex(int(got[i])), hex(int(want[i]))) for i in np.nonzero(got != want)[0]]
    return len(us), bad[:10], len(bad)


def near_zero_window(pos, byt, S, B):
    """f16-subnormal / zero w (|w| < 2^-14) collect h around -B / S: that h-range must hold one byte, which every bucket
    with exponent 0 then gets (correct whether the conversion keeps or flushes f16 subnormals)."""
    lo = np.float32((-B - 2.0 ** -14) / S)
    hi = np.float32((-B + 2.0 ** -14) / S)
    us = np.array([u32(hi), u32(lo)], np.int64)   # negative h: u order reversed
    j0, j1 = np.searchsorted(pos, us, side="right") - 1
    assert j0 == j1, "a change near w = 0"
    return int(byt[j0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rle", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--hlsl", help="write the HLSL header here")
    ap.add_argument("--cpp", help="write the runtime's C++ table header here")
    a = ap.parse_args()
    pos, byt = read_rle(a.rle)
    assert pos[0] == 0 and len(pos) == 205, len(pos)
    cands = candidates()
    chosen = None
    for S, B in sorted(cands, key=lambda sb: sb[1]):
        inner = internal_counts(pos, S, B)
        if max(inner.values(), default=0) <= 1:
            try:
                near_zero_window(pos, byt, S, B)
            except AssertionError:
                continue
            chosen = (S, B)
            break
    assert chosen, "no (S, B)"
    S, B = chosen
    zc = near_zero_window(pos, byt, S, B)
    table, inner, kinds = build(pos, byt, S, B, zc)
    npts, bad, nbad = check_points(pos, byt, S, B, table)
    assert nbad == 0, bad
    os.makedirs(a.out, exist_ok=True)
    table.astype("<f4").tofile(os.path.join(a.out, "gelu_lut_table.bin"))
    stats = {"S": S, "B": B, "S_bits": f"0x{int(u32(S)):08x}", "B_bits": f"0x{int(u32(B)):08x}",
             "entries": NB, "entry_bytes": 8, "k": K, "runs": int(len(pos)),
             "buckets_with_change": int(sum(inner.values())),
             "kinds": kinds, "model_check_points": npts,
             "near_zero_byte": zc, "candidates": len(cands)}
    with open(os.path.join(a.out, "gelu_lut.json"), "w") as f:
        json.dump(stats, f, indent=1)
    print(json.dumps(stats))
    if a.hlsl:
        write_hlsl(a.hlsl, S, B)
    if a.cpp:
        write_cpp(a.cpp, table)


HLSL = r"""// Exact q8(GELU(h)) of the MLP hidden path by table lookup. GENERATED by clean/dx12/gelu_lut/gen.py: do not edit.
//
// HIP (swin.co *_nc blocks, dec1_dec0__pack12.co, swin__bmegahg.co bneck6_nc) computes per hidden value h
//   g = h * rcp(1 + exp2(h * fma(h * h, -0.10029031, -2.30816186)))   (wg_platform.h wg_gelu, WG_GELU_MODE 2)
//   byte = v_cvt_pk_fp8_f32(g)                                         (wg_pack_q8_nc: no clamp)
// That byte is a fixed function of h with 204 changes over all 2^32 FP32 inputs. Here, without transcendentals:
//   w = fma(h, S, B);  i = f16_rtz(w) & ~7;  (a, b) = table[i];  byte = fp8(fma(h, a, b))
// i is the byte offset of an 8-byte entry: 128 buckets per binade over the whole FP32 range; inside a bucket the byte
// changes at most once, at the place (a, b) sets. The table is valid whether the driver rounds fma(h, S, B) to FP32
// before the f16 conversion or folds both into one v_fma_mixlo_f16 (round toward zero). The 64 KB table (8192 x (a, b)
// float32) sits at arena byte offset WG_GELU_LUT_BASE; the runtime reserves and fills it (runtime/gelu_lut_table.h).
// Proven on the GPU against HIP for every FP32 bit pattern (gelu_lut/run.py): identical bytes for all 2^32 inputs
// (FP32 denormal inputs included).
//
// Use: hf = wg_gelu_q8_lut(h) for wg_pack_q8_nc(GELU(h)) of a v8f; or wg_gelu_lut_fetch8 early (the loads) and
// wg_gelu_lut_finish8 later (fma + conversion; wave-uniform control flow) to hide the load latency.
#pragma once

#ifndef WG_GELU_LUT_BASE
#define WG_GELU_LUT_BASE 256u
#endif
#ifndef WG_GELU_LUT_PAIRS
#define WG_GELU_LUT_PAIRS 1   // 1: one v_cvt_pk_rtz_f16_f32 per two values; 0: one conversion per value
#endif
#define WG_GELU_LUT_S asfloat(@S@u)   // @Sf@
#define WG_GELU_LUT_B asfloat(@B@u)   // @Bf@

// Table entry of one value (one 8-byte load).
uint2 wg_gelu_lut_fetch(float h)
{
    return wg_mem.Load2(WG_GELU_LUT_BASE + (f32tof16(mad(h, WG_GELU_LUT_S, WG_GELU_LUT_B)) & 0xfff8u));
}
// Entries of two values from one conversion. The low half's mask has an opaque bit 16 (0 at run time: wg_bid_ < 2^31),
// otherwise the compiler converts the first value a second time on its own.
void wg_gelu_lut_fetch2(float h0, float h1, out uint2 e0, out uint2 e1)
{
    const uint d = f32tof16(mad(h0, WG_GELU_LUT_S, WG_GELU_LUT_B)) | (f32tof16(mad(h1, WG_GELU_LUT_S, WG_GELU_LUT_B)) << 16);
    e0 = wg_mem.Load2(WG_GELU_LUT_BASE + (d & (0xfff8u | ((wg_bid_ >> 31) << 16))));
    e1 = wg_mem.Load2(WG_GELU_LUT_BASE + ((d >> 16) & 0xfff8u));
}
// The value to convert: fp8(wg_gelu_lut_apply(h, e)) is HIP's byte (one fused v_fma_f32).
float wg_gelu_lut_apply(float h, uint2 e) { return mad(h, asfloat(e.x), asfloat(e.y)); }

struct wg_gelu_lut_e { uint2 e[8]; };
wg_gelu_lut_e wg_gelu_lut_fetch8(v8f h)
{
    wg_gelu_lut_e r;
#if WG_GELU_LUT_PAIRS
    [unroll] for (uint i = 0; i < 4; ++i) wg_gelu_lut_fetch2(h.v[2 * i], h.v[2 * i + 1], r.e[2 * i], r.e[2 * i + 1]);
#else
    [unroll] for (uint i = 0; i < 8; ++i) r.e[i] = wg_gelu_lut_fetch(h.v[i]);
#endif
    return r;
}
// FP8 bytes of 8 values from their entries (driver conversion: wave-uniform control flow).
uint2 wg_gelu_lut_finish8(v8f h, wg_gelu_lut_e t)
{
    v8f y;
    [unroll] for (uint i = 0; i < 8; ++i) y.v[i] = wg_gelu_lut_apply(h.v[i], t.e[i]);
    return wg_pack_q8_nc_ags(y);
}
// = HIP wg_pack_q8_nc(wg_gelu(h[0..7])) byte for byte.
uint2 wg_gelu_q8_lut(v8f h) { return wg_gelu_lut_finish8(h, wg_gelu_lut_fetch8(h)); }
"""


def write_hlsl(path, S, B):
    text = (HLSL.replace("@S@", f"0x{int(u32(S)):08x}").replace("@B@", f"0x{int(u32(B)):08x}")
            .replace("@Sf@", repr(float(np.float32(S)))).replace("@Bf@", repr(float(np.float32(B)))))
    with open(path, "w", newline="\n") as f:
        f.write(text)


def write_cpp(path, table):
    """The runtime's copy of the table as runs of equal entries (expanded once at device creation)."""
    w = table.view(np.uint32).reshape(NB, 2)
    runs, i = [], 0
    while i < NB:
        j = i
        while j + 1 < NB and w[j + 1, 0] == w[i, 0] and w[j + 1, 1] == w[i, 1]:
            j += 1
        runs.append((i, j - i + 1, int(w[i, 0]), int(w[i, 1])))
        i = j + 1
    lines = [f"    {{{a}, {n}, 0x{x:08x}u, 0x{y:08x}u}}," for a, n, x, y in runs]
    text = f"""// Exact q8(GELU) lookup table of the D3D12 MLP kernels (clean/dx12/hlsl/wg_gelu_lut.hlsli). GENERATED by
// clean/dx12/gelu_lut/gen.py from HIP's exhaustive run-length form: do not edit. The D3D12 devices reserve arena bytes
// [kGeluLutBase, kGeluLutBase + kGeluLutBytes) and fill them with gelu_lut_expand() at creation.
#pragma once
#include <cstddef>
#include <cstdint>

namespace wg {{
constexpr uint64_t kGeluLutBase = 256;      // = WG_GELU_LUT_BASE
constexpr size_t kGeluLutBytes = {NB * 8};   // {NB} entries (a, b) of two float32
struct GeluLutRun {{ uint32_t first, count, a, b; }};
constexpr GeluLutRun kGeluLutRuns[] = {{
{chr(10).join(lines)}
}};
// Writes the {NB * 2} little-endian dwords of the table to out.
inline void gelu_lut_expand(uint32_t* out) {{
  for (const GeluLutRun& r : kGeluLutRuns)
    for (uint32_t i = 0; i < r.count; ++i) {{ out[2 * (r.first + i)] = r.a; out[2 * (r.first + i) + 1] = r.b; }}
}}
}}  // namespace wg
"""
    with open(path, "w", newline="\n") as f:
        f.write(text)


if __name__ == "__main__":
    main()
