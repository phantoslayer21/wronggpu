// SPDX-License-Identifier: MIT
// Model L wave 3 (WG_L_W3 bit 4): l_project_gather and l_dec_project (l_ops_project.hlsl, byte for byte the same maps) in
// wave64, "design D" (hlsl/wg_w64.hlsli): the same 256-lane workgroup of 128 tokens, now 4 wave64 waves of 32 tokens, a wave's
// two 16-token tiles = the tokens of two of the wave32 kernel's waves. Wave64 lane L = l + 32 h64, l = 16 hh + n the wave32
// kernel's lane (token n, lane half hh); it holds, for each tile, half h64 of that lane's fragment (4 of the 8 FP8 bytes of a
// weight / token fragment, 4 of the 8 accumulator elements):
//   * the weight fragments (and the bias) are loaded ONCE per wave, 4 bytes per lane, and feed both tiles' v_wmma (uint2(w, w)):
//     half the weight loads and registers per token of the wave32 kernel;
//   * a token fragment is the lane's dword h64 (and 2 + h64) of the wave32 lane's 16 bytes, so that the K order of every
//     v_wmma, hence its rounding, is the wave32 kernel's;
//   * every conversion is per value (v_cvt_f16_f32, v_cvt_pk_fp8_f32), the clamps are the wave32 kernel's per-value ones, and
//     each output element is stored by the lane that holds it (4 bytes: the two output tiles' dwords are 8 bytes apart).
// l_project_gather: wg_pack_q8_ags of the clamped values is the plain conversion (the clamp to +-6.28125 leaves no NaN and nothing
// above 448, where wg_sat448 is the identity), so the FP8 pack here has no extra clamp.
//
// @wg-kernel l_project_gather_w64 co=- block=256 wave=64 args=p,p,p,u,u defines=PJ_DEC=0,WG_LD_A16=2
// @wg-kernel l_dec_project_w64 co=- block=256 wave=64 args=p,p,p,u,u defines=PJ_DEC=1,WG_LD_A16=2
#include "wg_platform.hlsli"
#include "wg_w64.hlsli"

WG_ARGS_BEGIN
    uint src, dst, w;   // as l_ops_project.hlsl
    int tw, th;
WG_ARGS_END

#if PJ_DEC
#define MT 8
#define KB 4
#define FRAG0 512u
#else
#define MT 4
#define KB 8
#define FRAG0 256u
#endif

// RN16 of the lane's 8 values (two tiles' elements) by the driver's conversion, then the exact widening
v8f pj_rn16(v8f a)
{
    const uint4 h = wg_to_f16_ags(a);
    v8f r;
    [unroll] for (uint j = 0; j < 4u; ++j) { r.v[2 * j] = f16tof32(h[j]); r.v[2 * j + 1] = wg_h2f_hi(h[j]); }
    return r;
}

#if PJ_DEC
WG_KERNEL(l_dec_project_w64)
#else
WG_KERNEL(l_project_gather_w64)
#endif
{
    WG_SETUP();
    const uint tid = wg_lane_id(), L = tid & 63u, lp = L & 31u, h64 = L >> 5, hh = (L >> 4) & 1u, n = L & 15u;
    const int ntok = tw * th;
    // tile t: token 32 (4 block + wave) + 16 t + n, clamped to the last token as the wave32 kernel (no branch)
    const int tokb = (int)(wg_block_id() * 4u + (tid >> 6)) * 32 + (int)n;
    const int tk0 = select(tokb < ntok, tokb, ntok - 1), tk1 = select(tokb + 16 < ntok, tokb + 16, ntok - 1);
    const int r0 = tk0 / tw, c0 = tk0 - r0 * tw, r1 = tk1 / tw, c1 = tk1 - r1 * tw;
    uint zf0[KB], zf1[KB];   // fragment f of tile 0 / 1: dword h64 of the wave32 lane's 8 bytes
#if PJ_DEC
    [unroll] for (uint f = 0; f < KB; ++f) {
        const uint o = 32u * (f >> 1) + 16u * hh + 8u * (f & 1u) + 4u * h64;
        zf0[f] = wg_ld32(src, (uint)tk0 * 64u + o);
        zf1[f] = wg_ld32(src, (uint)tk1 * 64u + o);
    }
#else
    [unroll] for (uint p = 0; p < 4u; ++p) {
        const uint pix0 = (uint)((2 * r0 + (int)(p >> 1)) * (2 * tw) + 2 * c0 + (int)(p & 1u));
        const uint pix1 = (uint)((2 * r1 + (int)(p >> 1)) * (2 * tw) + 2 * c1 + (int)(p & 1u));
        [unroll] for (uint q = 0; q < 2u; ++q) {
            const uint o = 16u * hh + 8u * q + 4u * h64;
            zf0[2u * p + q] = wg_ld32(src, pix0 * 32u + o);
            zf1[2u * p + q] = wg_ld32(src, pix1 * 32u + o);
        }
    }
#endif
    [unroll] for (uint mt = 0; mt < MT; mt += 2u) {
        // two adjacent output tiles at a time (as the wave32 kernel): the lane's 4 bias floats (rows 8 hh + 4 h64 + 0 .. 3) once
        const float4 bi0 = wg_ld4f(w, (hh * 8u + 4u * h64 + mt * 16u) * 4u);
        const float4 bi1 = wg_ld4f(w, (hh * 8u + 4u * h64 + (mt + 1u) * 16u) * 4u);
        v8f acc0, acc1;
        acc0.v[0] = bi0.x; acc0.v[1] = bi0.y; acc0.v[2] = bi0.z; acc0.v[3] = bi0.w;
        acc0.v[4] = bi0.x; acc0.v[5] = bi0.y; acc0.v[6] = bi0.z; acc0.v[7] = bi0.w;
        acc1.v[0] = bi1.x; acc1.v[1] = bi1.y; acc1.v[2] = bi1.z; acc1.v[3] = bi1.w;
        acc1.v[4] = bi1.x; acc1.v[5] = bi1.y; acc1.v[6] = bi1.z; acc1.v[7] = bi1.w;
        uint fr0[KB], fr1[KB];   // the weight fragments: dword h64 of the wave32 lane l's 8 bytes, shared by both tiles
        [unroll] for (uint k = 0; k < KB; ++k) {
            fr0[k] = wg_ld32(w, FRAG0 + (mt * KB + k) * 256u + lp * 8u + h64 * 4u);
            fr1[k] = wg_ld32(w, FRAG0 + ((mt + 1u) * KB + k) * 256u + lp * 8u + h64 * 4u);
        }
        [unroll] for (uint g = 0; g < KB / 2; ++g) {
            acc0 = wg_wmma_fp8(uint2(fr0[2u * g], fr0[2u * g]), uint2(zf0[2u * g], zf1[2u * g]), acc0);
            acc1 = wg_wmma_fp8(uint2(fr1[2u * g], fr1[2u * g]), uint2(zf0[2u * g], zf1[2u * g]), acc1);
            acc0 = wg_wmma_fp8(uint2(fr0[2u * g + 1u], fr0[2u * g + 1u]), uint2(zf0[2u * g + 1u], zf1[2u * g + 1u]), acc0);
            acc1 = wg_wmma_fp8(uint2(fr1[2u * g + 1u], fr1[2u * g + 1u]), uint2(zf0[2u * g + 1u], zf1[2u * g + 1u]), acc1);
            acc0 = pj_rn16(acc0);
            acc1 = pj_rn16(acc1);
        }
#if PJ_DEC
        const int s = (int)(mt >> 1);
        const uint pix0 = (uint)((2 * r0 + (s >> 1)) * (2 * tw) + 2 * c0 + (s & 1));
        const uint pix1 = (uint)((2 * r1 + (s >> 1)) * (2 * tw) + 2 * c1 + (s & 1));
        const uint2 q0 = wg_pack_q8_ags(acc0), q1 = wg_pack_q8_ags(acc1);   // (tile 0, tile 1) dwords of output tile mt / mt + 1
        wg_st32(dst, pix0 * 32u + 16u * hh + 4u * h64, q0.x);
        wg_st32(dst, pix0 * 32u + 16u * hh + 8u + 4u * h64, q1.x);
        wg_st32(dst, pix1 * 32u + 16u * hh + 4u * h64, q0.y);
        wg_st32(dst, pix1 * 32u + 16u * hh + 8u + 4u * h64, q1.y);
#else
        [unroll] for (uint j = 0; j < 8u; ++j) {
            acc0.v[j] = wg_fmin(wg_fmax(acc0.v[j], -6.28125f), 6.28125f);
            acc1.v[j] = wg_fmin(wg_fmax(acc1.v[j], -6.28125f), 6.28125f);
        }
        const uint2 q0 = wg_pack_q8_nc_ags(acc0), q1 = wg_pack_q8_nc_ags(acc1);
        wg_st32(dst, (uint)tk0 * 64u + hh * 32u + 8u * mt + 4u * h64, q0.x);
        wg_st32(dst, (uint)tk0 * 64u + hh * 32u + 8u * mt + 8u + 4u * h64, q1.x);
        wg_st32(dst, (uint)tk1 * 64u + hh * 32u + 8u * mt + 4u * h64, q0.y);
        wg_st32(dst, (uint)tk1 * 64u + hh * 32u + 8u * mt + 8u + 4u * h64, q1.y);
#endif
    }
}
