// SPDX-License-Identifier: MIT
// Model L's per-pixel coefficient functions (issue #89; HIP l_coef_body of clean/l/kernels/l_ops.hip), shared by
// kernels/l_ops_coef.hlsl (l_coef, l_coef_pack12) and the decoder block that ends in them (kernels/l_swin.hlsl,
// l_swin_block_c32_dec_coef12_nc): one pixel's channels 0..15 of the decoder block's FP16 output (y2) -> A, B, C and
// the latent, stored at pixel index i of the render rectangle. Every FP32 operation follows HIP's ISA of l_coef (see
// l_ops_coef.hlsl). Needs COEF_PACK12 (0: Model M's base layout, 1: pack12) and wg_platform.hlsli; uses g_z, an opaque
// zero the kernel sets from its grid size (g_z = wg_grid_ >> 31).

static uint g_z;   // 0, opaque to the compiler (set from the grid size)
float opq(float x) { return asfloat(asuint(x) ^ g_z); }
float bits(uint b) { return asfloat(b); }

// sigma(x) = 1 / (1 + 2^(x * -log2 e)): v_mul, v_exp, v_add 1.0, v_rcp
float l_sigma(float x) { return wg_rcp(1.0f + wg_exp2(x * bits(0xBFB8AA3Bu))); }

// tanh(t): for |t| >= 0.6 (and not NaN), 1 - 2 / (2^(2 |t| log2 e) + 1), 1 from about 9.01 on, sign of t; below, the odd
// polynomial. Both sides are computed; the ISA's branch selects one, select() here (both are pure).
float l_tanh(float t)
{
    const float a = abs(t);
    float big = mad(wg_rcp(wg_exp2(a * bits(0x4038AA3Bu)) + 1.0f), -2.0f, 1.0f);
    big = select(bits(0x41102CB4u) > a || isnan(a), big, 1.0f);   // v_cmp_nle 9.01, |t|; v_cndmask
    big = asfloat((asuint(big) & 0x7fffffffu) | (asuint(t) & 0x80000000u));   // v_bfi copysign
    const float t2 = t * t;
    const float p = mad(mad(mad(bits(0x3C80F082u), t2, bits(0xBD563CAEu)), t2, bits(0x3E085941u)), t2, bits(0xBEAAA9EDu));
    const float p2 = mad(p, t2, asfloat(g_z));   // fma(p, t2, +0), the zero opaque
    const float small = mad(p2, t, t);
    return select(a >= bits(0x3F19999Au), big, small);   // v_cmp_le 0.6, |t| (false for a NaN: the polynomial)
}

uint unorm8_code(float v) { return wg_f2u(wg_rint(v * 255.0f)); }   // v in [0, 1]: v_mul 255, v_rndne, v_cvt_u32
uint snorm8_code(float v) { return (uint)wg_f2i(wg_rint(wg_med3f(v, -1.0f, 1.0f) * 127.0f)) & 0xffu; }

// the clamp of a spread input (an FP16 value) to +-20, NaN kept: HIP's v >= 20 ? 20 : (v <= -20 ? -20 : v)
float clamp20(float v) { return select(v >= 20.0f, 20.0f, select(v <= -20.0f, -20.0f, v)); }

// One pixel: lo / hi = its channels 0..7 / 8..15 (four dwords each, two FP16 per dword, channel 2k in the low half);
// Model L's 13 values are channels a0..a12 = 0, 1, 6, 7, 8, 9, 10, 11, 12, 2, 3, 4, 5 (clean/l/reference/table.py
// L_DEC_VALUES). S, T: the coefficient blob's scalars (dec0.coef, bytes 320 and 324). pA, pB, pC, pL: the output
// buffers (pB unused with COEF_PACK12); i: the pixel's index in the render rectangle (y rw + x).
void l_coef_px(uint4 lo, uint4 hi, float S, float T, uint pA, uint pB, uint pC, uint pL, uint i)
{
    const float a0 = wg_h2f_lo(lo.x), a1 = wg_h2f_hi(lo.x), a2 = wg_h2f_lo(lo.w), a3 = wg_h2f_hi(lo.w);
    const float a4 = wg_h2f_lo(hi.x), a5 = wg_h2f_hi(hi.x), a6 = wg_h2f_lo(hi.y), a7 = wg_h2f_hi(hi.y);
    const float a8 = wg_h2f_lo(hi.z);
    const float a9 = wg_h2f_lo(lo.y), a10 = wg_h2f_hi(lo.y), a11 = wg_h2f_lo(lo.z), a12 = wg_h2f_hi(lo.z);
    // A: unorm codes of RN16(sigma(a0..a2))
    const uint cA = unorm8_code(wg_rn16(l_sigma(a0))) | (unorm8_code(wg_rn16(l_sigma(a1))) << 8) |
                    (unorm8_code(wg_rn16(l_sigma(a2))) << 16);
    // B: softmax of a3..a5
    const float m = max(max(a3, a4), a5);
    const float e0 = wg_exp2((a3 - m) * bits(0x3FB8AA3Bu)), e1 = wg_exp2((a4 - m) * bits(0x3FB8AA3Bu)),
                e2 = wg_exp2((a5 - m) * bits(0x3FB8AA3Bu));
    const float rs = wg_rcp(e2 + (e0 + e1));
    const uint cB = unorm8_code(opq(e0 * rs)) | (unorm8_code(opq(e1 * rs)) << 8) | (unorm8_code(opq(e2 * rs)) << 16);
    // C: spreads and correlation
    const float g0 = wg_rn16(l_sigma(clamp20(a6))), g1 = wg_rn16(l_sigma(clamp20(a7)));
    const float sx = S * g0, sy = S * g1;
    const float sxy = sx * sy;
    const float rho = l_tanh(a8 * T);
    const uint c0 = wg_mul_f16(sx, sx) | (wg_mul_f16(sy, sy) << 16);
    const uint c1 = wg_mul_f16(sxy, rho);
    // latent: snorm codes of tanh(a9..a12)
    const uint cL = snorm8_code(l_tanh(a9)) | (snorm8_code(l_tanh(a10)) << 8) | (snorm8_code(l_tanh(a11)) << 16) |
                    (snorm8_code(l_tanh(a12)) << 24);
#if COEF_PACK12
    wg_st64(pA, i * 8u, uint2(cA | (cB << 24), (cB >> 8) | (c1 << 16)));
    wg_st32(pC, i * 4u, c0);
#else
    wg_st32(pA, i * 4u, cA);
    wg_st32(pB, i * 4u, cB);
    wg_st64(pC, i * 8u, uint2(c0, c1));
#endif
    wg_st32(pL, i * 4u, cL);
}

#if COEF_CVT_HW
// The folded decoder block's form (l_swin_block_c32_dec_coef12_nc; l_coef / l_coef_pack12 keep l_coef_px above): the
// same values, with the FP16 roundings by the driver's conversion (v_cvt_f16_f32; wg_platform.hlsli: the same bits as
// the software wg_f2h): the five RN16 of sigma in one conversion, the three FP16 products as their round-to-odd FP32
// values (wg_mul_f16's) in another. The driver conversions need wave-uniform control flow: every lane computes
// l_coef_vals_hw, only the stores (l_coef_store12) are per lane. Needs COEF_PACK12.
struct LCoefVals { uint cA, cB, c0, c1, cL; };

// a * b as the round-to-odd FP32 value whose FP16 rounding is (f16)(a b) (wg_mul_f16 before its conversion)
float coef_mul_odd(float a, float b)
{
    const float p = a * b;
    return asfloat(wg_round_odd(p, mad(a, b, -p)));
}
// wg_mul_f16's +0 for an exact zero product (a NaN product keeps its conversion)
bool coef_mul_zero(float a, float b)
{
    return ((asuint(a) & 0x7fffffffu) == 0u || (asuint(b) & 0x7fffffffu) == 0u) && !isnan(a * b);
}

LCoefVals l_coef_vals_hw(uint4 lo, uint4 hi, float S, float T)
{
    const float a0 = wg_h2f_lo(lo.x), a1 = wg_h2f_hi(lo.x), a2 = wg_h2f_lo(lo.w), a3 = wg_h2f_hi(lo.w);
    const float a4 = wg_h2f_lo(hi.x), a5 = wg_h2f_hi(hi.x), a6 = wg_h2f_lo(hi.y), a7 = wg_h2f_hi(hi.y);
    const float a8 = wg_h2f_lo(hi.z);
    const float a9 = wg_h2f_lo(lo.y), a10 = wg_h2f_hi(lo.y), a11 = wg_h2f_lo(lo.z), a12 = wg_h2f_hi(lo.z);
    LCoefVals o;
    // RN16 of sigma(a0), sigma(a1), sigma(a2), sigma(clamp20(a6)), sigma(clamp20(a7)): one conversion
    v8f sg;
    sg.v[0] = l_sigma(a0); sg.v[1] = l_sigma(a1); sg.v[2] = l_sigma(a2);
    sg.v[3] = l_sigma(clamp20(a6)); sg.v[4] = l_sigma(clamp20(a7));
    sg.v[5] = 0.0f; sg.v[6] = 0.0f; sg.v[7] = 0.0f;
    const uint4 hs = wg_to_f16_ags(sg);
    // A: unorm codes of RN16(sigma(a0..a2))
    o.cA = unorm8_code(wg_h2f_lo(hs.x)) | (unorm8_code(wg_h2f_hi(hs.x)) << 8) | (unorm8_code(wg_h2f_lo(hs.y)) << 16);
    // B: softmax of a3..a5
    const float m = max(max(a3, a4), a5);
    const float e0 = wg_exp2((a3 - m) * bits(0x3FB8AA3Bu)), e1 = wg_exp2((a4 - m) * bits(0x3FB8AA3Bu)),
                e2 = wg_exp2((a5 - m) * bits(0x3FB8AA3Bu));
    const float rs = wg_rcp(e2 + (e0 + e1));
    o.cB = unorm8_code(opq(e0 * rs)) | (unorm8_code(opq(e1 * rs)) << 8) | (unorm8_code(opq(e2 * rs)) << 16);
    // C: spreads and correlation; the three FP16 products in one conversion
    const float g0 = wg_h2f_hi(hs.y), g1 = wg_h2f_lo(hs.z);
    const float sx = S * g0, sy = S * g1;
    const float sxy = sx * sy;
    const float rho = l_tanh(a8 * T);
    v8f po;
    po.v[0] = coef_mul_odd(sx, sx); po.v[1] = coef_mul_odd(sy, sy); po.v[2] = coef_mul_odd(sxy, rho);
    po.v[3] = 0.0f; po.v[4] = 0.0f; po.v[5] = 0.0f; po.v[6] = 0.0f; po.v[7] = 0.0f;
    const uint4 hp = wg_to_f16_ags(po);
    const uint h0 = coef_mul_zero(sx, sx) ? 0u : (hp.x & 0xffffu), h1 = coef_mul_zero(sy, sy) ? 0u : (hp.x >> 16);
    o.c0 = h0 | (h1 << 16);
    o.c1 = coef_mul_zero(sxy, rho) ? 0u : (hp.y & 0xffffu);
    // latent: snorm codes of tanh(a9..a12)
    o.cL = snorm8_code(l_tanh(a9)) | (snorm8_code(l_tanh(a10)) << 8) | (snorm8_code(l_tanh(a11)) << 16) |
           (snorm8_code(l_tanh(a12)) << 24);
    return o;
}

void l_coef_store12(LCoefVals o, uint pA, uint pC, uint pL, uint i)
{
    wg_st64(pA, i * 8u, uint2(o.cA | (o.cB << 24), (o.cB >> 8) | (o.c1 << 16)));
    wg_st32(pC, i * 4u, o.c0);
    wg_st32(pL, i * 4u, o.cL);
}
#endif
