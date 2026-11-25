# SPDX-License-Identifier: MIT
"""Tables and exhaustive facts behind the fast hybrid kernels' element-wise steps (WG_L_FORM=hybrid, SW_HYB): every
step whose input is one FP16 value is checked here for ALL 65,536 inputs against the oracle forms of
clean/l/reference/oracle.py (norm_q7, softmax_q7, gelu_q7).

- gelu16_q8: 65,536 bytes, index = the FP16 bits of h: q8(gelu_q7(h)) (the MLP's GELU16 then the FP8 point); a NaN
  result is written as 0xFF, the hardware FP8 conversion's NaN on HIP and D3D12 (it does not keep a NaN's sign), so
  the D3D12 table and the HIP computation give the same bytes.
- rsq16: the norm's r = RN16(rsqrt32(S')) for every positive finite FP16 S', as 2 x 1024 entries (exponent parity x
  mantissa) plus an exponent shift: exact for the norm's range S' >= 2^-13 (checked for every FP16 value there).
- rcp16: the softmax normaliser w = RN16(rcp32(sum)) for every positive normal FP16 sum, as 1024 entries (mantissa) plus
  an exponent shift; exact wherever the result is normal (checked).
- softmax_e: e(u') of softmax_q7 for every clamped u' = min(max(RN16(s c), -c'), c'): computed in float32 with one FP16
  rounding per step (t = RN16(fma32(u', -u', t0)), p = RN16(fma32(u', t, p0))) equals the FMA16 form for every u'
  (softmax_e_f32_equals_fma16), so no table is needed; the products u' u' and u' t are exact in float32, so every
  lowering (fused or not) gives the same float32 value.
"""
import numpy as np

from clean.l.reference import oracle as O
from clean.reference.fp8 import fp8_encode, q8

ALL16 = np.arange(65536, dtype=np.uint32).astype(np.uint16)
H = ALL16.view(np.float16).astype(np.float64)


def f16(x):
    return np.asarray(x, np.float64).astype(np.float16).astype(np.float64)


def gelu16_q8():
    """q8(gelu_q7(h)) for every FP16 bit pattern h (NaN in, NaN code out)."""
    with np.errstate(invalid="ignore", over="ignore"):
        g = O.gelu_q7(H)
        out = np.asarray(fp8_encode(q8(g)), np.uint8)
    out[np.isnan(g)] = 0xFF
    return out


def _fp8_point(r):
    """The FP8 point of FP16 values r (float64 of FP16 numbers): saturating E4M3 with every NaN -> 0xFF."""
    out = np.asarray(fp8_encode(q8(r)), np.uint8)
    out[np.isnan(r)] = 0xFF
    return out


def gelu_poly_q8(fused=False):
    """q8(gelu_q7(h)) for every FP16 bit pattern h by the arithmetic of hlsl/hyb_block.hlsli hyb_gelu_poly8: packed FP16 steps,
    each its own rounding (numpy float16 operations; min / max return the number for a NaN: fmin / fmax):
        y = min(max(h, -2), 2); p1 = RN16(|y| b); p2 = RN16(a - p1); p3 = RN16(y p2); p4 = RN16(1/2 + p3); g = RN16(h p4)
    fused=True: the contracted form (p4 = RN16(1/2 + y p2) with the product exact), the one the kernels' contraction barriers
    (HYB_OPAQUE0) keep the driver from forming: NOT equal to the table (37 of the 65,536 bytes differ)."""
    f = np.float16
    a, b = f(O.Q7_GELU_A), f(O.Q7_GELU_B)
    h = ALL16.view(np.float16)
    with np.errstate(all="ignore"):
        y = np.fmin(np.fmax(h, f(-2.0)), f(2.0))
        p1 = (np.abs(y) * b).astype(f)
        p2 = (a - p1).astype(f)
        if fused:
            p4 = (np.float64(f(0.5)) + y.astype(np.float64) * p2.astype(np.float64)).astype(f)
        else:
            p3 = (y * p2).astype(f)
            p4 = (f(0.5) + p3).astype(f)
        r = (h * p4).astype(f)
        return _fp8_point(r.astype(np.float64))


def _rsq_ref(s):
    return f16(O._rsqrt32(s))


def rsq16_table():
    """[2][1024] FP16 bits: r for S' = 1.m x 2^(e) with e even (row 0: e = 0) and odd (row 1: e = 1)."""
    m = np.arange(1024)
    base = 1.0 + m / 1024.0
    t0 = _rsq_ref(base).astype(np.float16).view(np.uint16)
    t1 = _rsq_ref(base * 2.0).astype(np.float16).view(np.uint16)
    return np.stack([t0, t1])


def rsq16_lookup(sbits, tab):
    """r bits for positive normal FP16 S' bits via the table: S' = 1.m 2^E, E = 2k + p -> r = T[p][m] x 2^-k."""
    sbits = np.asarray(sbits, np.int64)
    E = ((sbits >> 10) & 31) - 15
    m = sbits & 1023
    p = E & 1
    k = (E - p) // 2
    rb = tab[p, m].astype(np.int64)
    return (rb - (k << 10)).astype(np.uint16)       # r normal in the norm's range: exponent field minus k


def rcp16_table():
    m = np.arange(1024)
    return f16(O._rcp32(1.0 + m / 1024.0)).astype(np.float16).view(np.uint16)


def rcp16_lookup(sbits, tab):
    sbits = np.asarray(sbits, np.int64)
    E = ((sbits >> 10) & 31) - 15
    m = sbits & 1023
    rb = tab[m].astype(np.int64)
    return (rb - (E << 10)).astype(np.uint16)


def softmax_e_ref(u):
    """e of softmax_q7 for clamped u' (float64 values), by the oracle's own steps."""
    t = f16(u * -u + O.Q7_POLY_T)
    p = f16(u * t + O.Q7_POLY_P)
    pb = p.astype(np.float16).view(np.uint16)
    return ((pb & 0x03FF) << 5).astype(np.uint16)


def softmax_e_f32(u):
    """the kernels' form: float32 fused multiply-adds (exact products), one FP16 rounding per step."""
    u32 = u.astype(np.float32)
    t32 = (u.astype(np.float64) * -u.astype(np.float64) + O.Q7_POLY_T).astype(np.float32)    # fma32: exact, then RN32
    t = f16(t32)
    p32 = (u.astype(np.float64) * t + O.Q7_POLY_P).astype(np.float32)
    p = f16(p32)
    pb = p.astype(np.float16).view(np.uint16)
    del u32
    return ((pb & 0x03FF) << 5).astype(np.uint16)


def clamped_u():
    """every FP16 value u' can take: |u'| <= c' (FP16 0x3873)."""
    c = O.Q7_SCORE_CLAMP
    return H[np.isfinite(H) & (np.abs(H) <= c)]


def facts():
    """The exhaustive facts the kernels rely on (all must be True)."""
    out = {}
    with np.errstate(invalid="ignore", over="ignore", divide="ignore"):
        # rsqrt: every FP16 S' >= 2^-13 (the norm adds 2^-13 to a non-negative sum)
        sb = ALL16[(H >= 2.0 ** -13) & np.isfinite(H)]
        ref = _rsq_ref(sb.view(np.float16).astype(np.float64)).astype(np.float16).view(np.uint16)
        out["rsq16_table_exact"] = bool(np.array_equal(rsq16_lookup(sb, rsq16_table()), ref))
        # rcp: every positive normal FP16 sum whose reciprocal is normal
        sb = ALL16[(H >= 2.0 ** -14) & np.isfinite(H)]
        ref = f16(O._rcp32(sb.view(np.float16).astype(np.float64)))
        ok = ref >= 2.0 ** -14
        out["rcp16_table_exact"] = bool(np.array_equal(rcp16_lookup(sb[ok], rcp16_table()),
                                                       ref[ok].astype(np.float16).view(np.uint16)))
        u = clamped_u()
        out["softmax_e_f32_equals_fma16"] = bool(np.array_equal(softmax_e_f32(u), softmax_e_ref(u)))
        # the softmax weight range: every e from a clamped u' is a positive normal FP16 (never 0, never Inf / NaN)
        e = softmax_e_ref(u).view(np.float16).astype(np.float64)
        out["softmax_e_positive_normal"] = bool(np.all((e >= 2.0 ** -14) & np.isfinite(e)))
        g = gelu16_q8()
        out["gelu16_q8_nan_maps_to_nan"] = bool(np.all((g[np.isnan(H)] & 0x7F) == 0x7F))
    return out


if __name__ == "__main__":
    for k, v in facts().items():
        print(k, v)
