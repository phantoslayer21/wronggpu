// SPDX-License-Identifier: MIT
// Exact arithmetic of the D3D12 ports: HLSL that gives the bits of the HIP kernels' gfx1201 instructions.
//
// ARITHMETIC MODE (clean/dx12/build_dx12.py, README.md "Arithmetic mode"). Kernels are compiled by DXC with IEEE
// strictness (-Gis) and the DXIL is then rewritten so that only the intrinsics keep the driver's default lowering:
//   * a * b, a + b, a - b, comparisons, conversions: each is ONE IEEE operation, rounded to nearest even, exactly as
//     written. Nothing is contracted into an FMA, reassociated, constant-folded across operations or simplified with
//     NaN / infinity / signed-zero assumptions (neither by DXC nor by the driver). So a * b + c is HIP's v_mul_f32 then
//     v_add_f32 (two roundings), and (x + 0.5) + 0.25 stays two additions.
//   * mad(a, b, c) is ALWAYS one fused v_fma_f32 (single rounding): use it wherever HIP's ISA has v_fma_f32 / v_fmac_f32 /
//     v_fmaak_f32 / v_fmamk_f32 / v_fma_mix_f32 (f16 operands widened exactly by wg_h2f / f16tof32). mad on half / half2
//     is v_fma_f16 / v_pk_fma_f16. dot2add(h2, h2, f) is one v_dot2_f32_f16.
//   * rcp(x) is the hardware v_rcp_f32 (HIP __builtin_amdgcn_rcpf), exp2 = v_exp_f32, log2 = v_log_f32, sqrt =
//     v_sqrt_f32, rsqrt = v_rsq_f32, round = v_rndne (rint), floor, min / max = v_min_num / v_max_num (IEEE minNum:
//     a NaN operand gives the other), (int) / (uint) = v_cvt_i32 / u32_f32 (truncate, saturate, NaN -> 0).
//   * The '/' operator on floats is refused by the build (it would be an IEEE division with another sequence than
//     HIP's): HIP's a / b is wg_div. Float '%' is refused.
//   * Never use 'precise' (it is redundant in this mode) and never compile a kernel outside build_dx12.py: in DXC's
//     default mode mad() may be split, a * b + c fused, and sums reassociated.
// FP32 DENORMALS: the D3D12 shaders run with FP32 denormals flushed to zero (DXC's default mode; '-denorm preserve'
// would turn rcp into a division). Every result here equals HIP's except where an FP32 denormal is an input, an
// intermediate or the result (HIP keeps them); the network's data does not produce them. FP16 values keep denormals.
// NaN PAYLOADS: when two operands of one operation are NaN, which payload propagates may differ from HIP (the operand
// order of commutative operations is not kept). One NaN operand propagates as in HIP.
// HALF CONVERSIONS: (half) casts and f32tof16 round toward zero on this driver (= v_cvt_pkrtz_f16_f32, HIP wg_h2);
// HIP's (_Float16) x (v_cvt_f16_f32, round to nearest even) is wg_f2h. f16tof32 / (float) half are exact.
// Probe: clean/dx12/probe/arith_check.py and arith2_check.py (0 differences outside the denormal cases).
//
// DRIVER EXCEPTIONS found while porting the Model M kernels (the probes above did not have these shapes; see
// clean/dx12/README.md "Driver exceptions" and STATUS.md). The rules above hold for single operations; across
// operations the gfx1201 D3D12 shader compiler still:
//   * regroups chains of multiplications by constant / uniform / shared factors: (v * 0.01) * ratio can become
//     v * (0.01 * ratio). Break the chain where it matters with an opaque zero: asfloat(asuint(x) ^ z) with z a value the
//     compiler cannot prove zero (post.hlsl opq(), z = wg_grid_ >> 31);
//   * turns mad(a, b, 0) into a * b and may then fuse that product with the next addition: same remedy;
//   * fuses half (FP16) multiply + add into one v_fma_mix even under -Gis: do FP16-rounded arithmetic in FP32 and
//     round explicitly (wg_f2h, wg_mul_f16, the driver conversion);
//   * folds an FP32 operation that feeds the driver's matrix conversion into it (wg_to_f16_ags(1 - z) became one
//     v_fma_mixlo_f16): harmless when that operation is exact, otherwise one rounding instead of two: check the ISA.
#pragma once

// a * b + c with two roundings (HIP: v_mul_f32 then v_add_f32). In this mode the plain expression is exactly that.
float wg_mul_add(float a, float b, float c) { return a * b + c; }
// a * b + c with one rounding (HIP: v_fma_f32 and its v_fmac / v_fmaak / v_fmamk forms).
float wg_fma(float a, float b, float c) { return mad(a, b, c); }

// IEEE division, round to nearest even, as HIP's a / b (v_div_scale / v_rcp / v_div_fmas / v_div_fixup). The operands
// are scaled by powers of two into [1, 2), the quotient is refined with fused multiply-adds (correctly rounded there),
// and the exponent is put back (exact for normal results). Zero, infinite and NaN operands give HIP's results:
// 0 / 0 and inf / inf -> 0xffc00000, a NaN operand -> that NaN quieted (a first), x / 0 -> inf, x / inf -> 0.
float wg_div(float a, float b)
{
    uint ua = asuint(a), ub = asuint(b);
    uint ea = (ua >> 23) & 255u, eb = (ub >> 23) & 255u;
    bool special = ea == 0u || eb == 0u || ea == 255u || eb == 255u;
    float as = asfloat((ua & 0x807fffffu) | 0x3f800000u), bs = asfloat((ub & 0x807fffffu) | 0x3f800000u);
    float y = rcp(bs);
    float e = mad(-bs, y, 1.0f);
    y = mad(e, y, y);
    float q = as * y;
    float r = mad(-bs, q, as);
    q = mad(r, y, q);
    r = mad(-bs, q, as);
    q = mad(r, y, q);                                   // correctly rounded a' / b', |q| in (0.5, 2)
    int en = (int)((asuint(q) >> 23) & 255u) + (int)ea - (int)eb;
    uint res = (asuint(q) & 0x807fffffu) | ((uint)clamp(en, 1, 254) << 23);
    res = en > 254 ? (asuint(q) & 0x80000000u) | 0x7f800000u : res;   // overflow -> infinity
    res = en < 1 ? asuint(q) & 0x80000000u : res;                     // denormal result -> zero (FTZ)
    // special operands (FTZ: denormals count as zero)
    uint sg = (ua ^ ub) & 0x80000000u;
    bool za = ea == 0u, zb = eb == 0u, ia = ea == 255u && (ua & 0x7fffffu) == 0u, ib = eb == 255u && (ub & 0x7fffffu) == 0u;
    bool na = ea == 255u && !ia, nb = eb == 255u && !ib;
    uint sp = (za || ib) ? sg : sg | 0x7f800000u;
    sp = (za && zb) || (ia && ib) ? 0xffc00000u : sp;
    sp = nb ? ub | 0x400000u : sp;
    sp = na ? ua | 0x400000u : sp;
    return asfloat(special ? sp : res);
}

// x * 2^n (HIP __builtin_ldexpf, v_ldexp_f32) for normal results; two exact scalings so any n works.
float wg_ldexp(float x, int n)
{
    int n1 = clamp(n, -126, 127);
    int n2 = clamp(n - n1, -126, 127);
    return x * asfloat((uint)(n1 + 127) << 23) * asfloat((uint)(n2 + 127) << 23);
}

// f32 -> f16 bits, round to nearest even, NaN payload kept as v_cvt_f16_f32 does (HIP (_Float16) x).
uint wg_f2h(float f)
{
    uint u = asuint(f), s = (u >> 16) & 0x8000u, a = u & 0x7fffffffu;
    uint nrm = (a + 0xC8000FFFu + ((a >> 13) & 1u)) >> 13;                  // rebias the exponent, round to nearest even
    uint den = asuint(asfloat(a) + asfloat(0x3F000000u)) - 0x3F000000u;      // f16 subnormals: the f32 addition rounds
    uint r = a < 0x38800000u ? den : nrm;
    r = a >= 0x47800000u ? (a > 0x7f800000u ? (0x7e00u | ((a >> 13) & 0x3ffu)) : 0x7c00u) : r;
    return s | r;
}
// f16 bits (low 16 bits of h) -> f32, exact.
float wg_h2f(uint h) { return f16tof32(h); }

// Round to odd at FP32 precision: r is the rounded (nearest even) value of an exact x, and t has the sign of x - r (0 when
// r is exact). Returns the FP32 neighbour of x with an odd last bit (r itself when r is exact or odd). FP32 has 13 bits more
// than FP16, so rounding this value to FP16 gives the FP16 rounding of x itself (no double rounding).
uint wg_round_odd(float r, float t)
{
    uint ur = asuint(r);
    bool fix = t != 0.0f && (ur & 1u) == 0u && (ur & 0x7fffffffu) - 1u < 0x7f7fffffu;   // inexact, even, finite, not 0
    uint toward = ((asuint(t) ^ ur) & 0x80000000u) == 0u ? ur + 1u : ur - 1u;           // one ulp toward x
    return fix ? toward : ur;
}

// (f16)(a * b + c) with one rounding, as v_fma_mixlo_f16 a, b, c with FP32 operands. The exact residual of the fused
// result (Boldo-Muller, ErrFma) gives the round-to-odd FP32 value, which is then rounded to FP16 once.
uint wg_fma_f16(float a, float b, float c)
{
    float r = mad(a, b, c);
    float u1 = a * b, u2 = mad(a, b, -u1);              // a b = u1 + u2
    float a1 = c + u2, av = a1 - u2, z = (c - av) + (u2 - (a1 - av));      // c + u2 = a1 + z
    float b1 = u1 + a1, bv = b1 - a1, b2 = (u1 - bv) + (a1 - (b1 - bv));  // u1 + a1 = b1 + b2
    float g = (b1 - r) + b2;                            // a b + c - r = g + z (+ the rounding of g, below ulp(g))
    float s = g + z, sv = s - z, se = (g - sv) + (z - (s - sv));           // g + z = s + se
    float t = s != 0.0f ? s : se;                       // sign of (a b + c - r)
    uint ur = asuint(r);
    return wg_f2h(asfloat((ur & 0x7fffffffu) >= 0x7f800000u ? ur : wg_round_odd(r, t)));
}

// (f16)(a * b) with one rounding, as HIP compiles it: v_fma_mixlo_f16 a, b, 0. The addend +0 makes an exact zero
// product +0; a nonzero product that rounds to zero keeps its sign.
uint wg_mul_f16(float a, float b)
{
    float p = a * b;
    float e = mad(a, b, -p);                            // exact: a b = p + e
    uint h = wg_f2h(asfloat(wg_round_odd(p, e)));
    bool zero = ((asuint(a) & 0x7fffffffu) == 0u || (asuint(b) & 0x7fffffffu) == 0u) && !isnan(p);   // 0 x inf: NaN
    return zero ? 0u : h;
}

// IEEE-754-2019 minimum / maximum (a NaN propagates, quieted; -0 < +0), as __builtin_elementwise_minimum / maximum
// (v_minimum_f32 / v_maximum_f32 / v_maximumminimum_f32).
float wg_minimum(float a, float b)
{
    float m = (asuint(a) == 0x80000000u || asuint(b) == 0x80000000u) && min(a, b) == 0.0f ? asfloat(0x80000000u) : min(a, b);
    m = isnan(b) ? asfloat(asuint(b) | 0x400000u) : m;
    return isnan(a) ? asfloat(asuint(a) | 0x400000u) : m;
}
float wg_maximum(float a, float b)
{
    float m = (asuint(a) == 0u || asuint(b) == 0u) && max(a, b) == 0.0f ? 0.0f : max(a, b);
    m = isnan(b) ? asfloat(asuint(b) | 0x400000u) : m;
    return isnan(a) ? asfloat(asuint(a) | 0x400000u) : m;
}

// FP8 E4M3FN byte of HIP's wg_pack_q8 for one value (clamp to +-448 with NaN kept, then v_cvt_pk_fp8_f32: round to nearest
// even; NaN -> 0xff / 0x7f by sign as the hardware).
uint wg_e4m3_sat(float f)
{
    uint b = asuint(f), s = (b >> 24) & 0x80u, a = b & 0x7fffffffu;
    uint sub = (uint)round(min(asfloat(a), 0.015625f) * 512.0f);
    uint nrm = ((a + 0x7ffffu + ((a >> 20) & 1u)) >> 20) - (120u << 3);
    uint code = a < 0x3c800000u ? sub : nrm;
    code = a >= 0x43e00000u ? 0x7eu : code;
    return a > 0x7f800000u ? 0xffu : s | code;
}

// FP8 E4M3FN byte of v_cvt_pk_fp8_f32 itself (no clamp; HIP wg_pack_q8_nc): round to nearest even, |x| > 464 (and
// +-infinity) -> 0x7f | sign, NaN -> 0xff.
uint wg_e4m3_hw(float f)
{
    uint b = asuint(f), s = (b >> 24) & 0x80u, a = b & 0x7fffffffu;
    uint sub = (uint)round(min(asfloat(a), 0.015625f) * 512.0f);
    uint nrm = ((a + 0x7ffffu + ((a >> 20) & 1u)) >> 20) - (120u << 3);
    uint code = a < 0x3c800000u ? sub : nrm;
    code = a > 0x43e80000u ? 0x7fu : code;
    return a > 0x7f800000u ? 0xffu : s | code;
}
