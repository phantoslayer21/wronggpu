// SPDX-License-Identifier: MIT
// D3D12 port of Model L's coefficient step (issue #89): l_coef of l_ops.co (clean/l/kernels/l_ops.hip), byte for byte.
// y2 [fh][fw][32] FP16 (the decoder block's output) -> A, B (3 unorm8 codes each), C (two dwords: C[2 i] = FP16 sx^2 |
// sy^2 << 16, C[2 i + 1] = FP16 sxy rho in the low half) and the new latent (4 snorm8 codes), [rh][rw] each, pixels
// inside R only. Lane (t, h) of wave i handles token 16 i + t =
// (r, c) of the (fh / 2) x (fw / 2) grid, pixels (2 c, 2 r + h) and (2 c + 1, 2 r + h). Reads no motion, previous latent,
// jitter or reset. w = blob dec0.coef (S, T at byte 320).
//
// Every FP32 operation follows HIP's ISA of l_coef (llvm-objdump of l_ops.co), not its source, where they differ:
// - C0 = (f16)(sx sx) | (f16)(sy sy) << 16 and C1 = (f16)(sxy rho), each ONE rounding (v_fma_mixlo_f16 a, b, 0 =
//   wg_mul_f16), not an FP32 product rounded again;
// - B: s = e2 + (e0 + e1) (commutes with the source's (e0 + e1) + e2), code = rint((e r) 255): opq() keeps the driver
//   from regrouping the product with the constant 255;
// - tanh below 0.6: fma(p, t^2, 0) with an opaque zero (the driver turns mad(a, b, 0) into a b, which changes the sign
//   of a zero), then fma(p2, t, t);
// - the clamp of the two spread inputs to +-20 keeps a NaN (compares, as HIP's v_cmp / v_cndmask on the FP16 value).
// The three 32-bit stores at the same per-lane offset into different buffers (A, B, latent) are the shape the driver was
// seen to merge (README "drop a store"): the pointers are read through wg_uniform().
//
// COEF_PACK12 (kernel l_coef_pack12, HIP l_coef_body<true>): the same values in Model M's dec0 "pack12" layout, read by
// the output stage's _p12 kernels: image A, 8 bytes per pixel = A0 A1 A2 B0 | B1 B2 C2 (FP16); image C, 4 bytes per
// pixel = C0 C1 (FP16); B unused (may be null); the latent as l_coef.
//
// @wg-kernel l_coef co=l_ops.co block=256 wave=32 args=p,p,p,p,p,p,b56 defines=COEF_PACK12=0
// @wg-kernel l_coef_pack12 co=l_ops.co block=256 wave=32 args=p,p,p,p,p,p,b56 defines=COEF_PACK12=1
#include "wg_platform.hlsli"

WG_ARGS_BEGIN
    uint y2, A, B, C, latent, w;   // pointers
    int P_rw, P_rh, P_fw, P_fh, P_reset, P_pad0;   // LDecParams (Model M's WgDecParams; only rw, rh, fw, fh read)
    float P_dw, P_dh, P_kx, P_ky, P_jx, P_jy, P_pjx, P_pjy;
WG_ARGS_END

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

#if COEF_PACK12
WG_KERNEL(l_coef_pack12)
#else
WG_KERNEL(l_coef)
#endif
{
    WG_SETUP();
    g_z = wg_grid_ >> 31;
    const uint tid = wg_lane_id(), hh = (tid >> 4) & 1u;
    const int tw = P_fw >> 1, ntok = (P_fh >> 1) * tw;
    const int tok = (int)(wg_block_id() * (WG_BLOCK / WG_WAVE) + (tid >> 5)) * 16 + (int)(tid & 15u);
    if (tok >= ntok) return;
    const int r = tok / tw, c = tok - r * tw;
    const int y = 2 * r + (int)hh;
    if (y >= P_rh) return;
    const float S = wg_ldf(w, 320u), T = wg_ldf(w, 324u);
    const uint pA = (uint)wg_uniform((int)A), pB = (uint)wg_uniform((int)B), pC = (uint)wg_uniform((int)C),
               pL = (uint)wg_uniform((int)latent);
    [unroll] for (int dx = 0; dx < 2; ++dx) {
        const int x = 2 * c + dx;
        if (x >= P_rw) continue;
        // channels 0..15 of the pixel (8 dwords, two FP16 each); Model L's 13 values are channels
        // a0..a12 = 0, 1, 6, 7, 8, 9, 10, 11, 12, 2, 3, 4, 5 (clean/l/reference/table.py L_DEC_VALUES)
        const uint off = (uint)(y * P_fw + x) * 64u;
        const uint4 lo = wg_ld128(y2, off), hi = wg_ld128(y2, off + 16u);
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
        const uint i = (uint)(y * P_rw + x);
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
}
