// SPDX-License-Identifier: MIT
// D3D12 ports of Model L's two projections (issue #89): l_project_gather and l_dec_project of l_ops.co
// (clean/l/kernels/l_ops.hip), byte for byte. Maps and blobs: clean/l/FORMATS.md.
//
// 256 lanes; wave i handles tokens 16 i .. 16 i + 15 of its workgroup's 128 (raster order), lane (t, h) token 16 i + t;
// accumulator element v of output tile mt is output channel 16 h + 8 mt' + v of the tile's layout (HIP: layout L / E).
// K blocks in the order of clean/l/runner/l_pack.py kord_groups32: block kb, lane half h = channel
// 32 (kb / 2) + 16 h + 8 (kb % 2) + j. Per group of 32 inputs: two FP8 WMMAs, then RN16 (running FP16 sum, bias first).
// No branch: a lane past the last token computes the last token (its inputs are clamped there, as HIP's) and stores
// that token's bytes to that token's place, the same bytes its own lane stores (a branch around the store let the
// compiler sink the matrix instructions into it, see l_ops_linear.hlsl).
// Conversions by the driver's instructions (wg_platform.hlsli; the same bits as the software wg_rn16 / wg_pack_q8, and
// this kernel has no branch): RN16 = v_cvt_f16_f32 then the exact widening, the FP8 pack v_cvt_pk_fp8_f32 after the
// clamp. The software forms were most of the kernels' instructions.
//
// @wg-kernel l_project_gather co=l_ops.co block=256 wave=32 args=p,p,p,u,u defines=PJ_DEC=0
// @wg-kernel l_dec_project co=l_ops.co block=256 wave=32 args=p,p,p,u,u defines=PJ_DEC=1
#include "wg_platform.hlsli"

WG_ARGS_BEGIN
    uint src, dst, w;   // gather: x (pixel map [2 th][2 tw][32] FP8), features [th][tw][64]; dec: z [th][tw][64], y pixel map
    int tw, th;
WG_ARGS_END

#if PJ_DEC
#define MT 8
#define KB 4
#define FRAG0 512u   // 4 * 128: after the bias tile vector, rows R_E(32) x 4 sub-pixels
#else
#define MT 4
#define KB 8
#define FRAG0 256u   // 4 * 64: after the bias tile vector, rows R_L(64)
#endif

// RN16 of 8 values by the driver's conversion
v8f pj_rn16(v8f a)
{
    const uint4 h = wg_to_f16_ags(a);
    v8f r;
    [unroll] for (uint j = 0; j < 4u; ++j) { r.v[2 * j] = wg_h2f_lo(h[j]); r.v[2 * j + 1] = wg_h2f_hi(h[j]); }
    return r;
}

#if PJ_DEC
WG_KERNEL(l_dec_project)
#else
WG_KERNEL(l_project_gather)
#endif
{
    WG_SETUP();
    const uint tid = wg_lane_id(), lane = tid & 31u, hh = (tid >> 4) & 1u;
    const int ntok = tw * th;
    const int tok = (int)(wg_block_id() * (WG_BLOCK / WG_WAVE) + (tid >> 5)) * 16 + (int)(tid & 15u);
    const int tk = select(tok < ntok, tok, ntok - 1);
    const int r = tk / tw, c = tk - r * tw;
    uint2 zf[KB];
#if PJ_DEC
    [unroll] for (uint kb = 0; kb < KB; ++kb)
        zf[kb] = wg_ld64(src, (uint)tk * 64u + 32u * (kb >> 1) + 16u * hh + 8u * (kb & 1u));
#else
    [unroll] for (uint kb = 0; kb < KB; ++kb) {
        const int p = (int)(kb >> 1);
        const int pix = (2 * r + (p >> 1)) * (2 * tw) + 2 * c + (p & 1);
        zf[kb] = wg_ld64(src, (uint)pix * 32u + 16u * hh + 8u * (kb & 1u));
    }
#endif
    [unroll] for (uint mt = 0; mt < MT; ++mt) {
        // the loads of this tile's multiplies before its first matrix instruction
        v8f acc = wg_load8f(w, (hh * 8u + mt * 16u) * 4u);
        uint2 fr[KB];
        [unroll] for (uint k = 0; k < KB; ++k) fr[k] = wg_ld64(w, FRAG0 + (mt * KB + k) * 256u + lane * 8u);
        [unroll] for (uint g = 0; g < KB / 2; ++g) {
            acc = wg_wmma_fp8(fr[2u * g], zf[2u * g], acc);
            acc = wg_wmma_fp8(fr[2u * g + 1u], zf[2u * g + 1u], acc);
            acc = pj_rn16(acc);
        }
#if PJ_DEC
        const int s = (int)(mt >> 1), gg = (int)(mt & 1u);
        const int pix = (2 * r + (s >> 1)) * (2 * tw) + 2 * c + (s & 1);
        wg_st64(dst, (uint)pix * 32u + 16u * hh + 8u * (uint)gg, wg_pack_q8_ags(acc));
#else
        [unroll] for (uint j = 0; j < 8u; ++j) acc.v[j] = wg_fmin(wg_fmax(acc.v[j], -6.28125f), 6.28125f);
        wg_st64(dst, (uint)tk * 64u + hh * 32u + 8u * mt, wg_pack_q8_ags(acc));
#endif
    }
}
