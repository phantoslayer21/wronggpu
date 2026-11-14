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

// the per-pixel functions (sigma, tanh, the codes, the clamp, the stores): hlsl/l_coef_px.hlsli, shared with the decoder
// block that ends in this step (l_swin.hlsl, l_swin_block_c32_dec_coef12_nc)
#include "l_coef_px.hlsli"

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
        // channels 0..15 of the pixel (8 dwords, two FP16 each)
        const uint off = (uint)(y * P_fw + x) * 64u;
        const uint4 lo = wg_ld128(y2, off), hi = wg_ld128(y2, off + 16u);
        l_coef_px(lo, hi, S, T, pA, pB, pC, pL, (uint)(y * P_rw + x));
    }
}
