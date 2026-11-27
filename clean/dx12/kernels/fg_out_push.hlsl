// SPDX-License-Identifier: GPL-3.0-only
// Frame generation (issue #96): the Full path's output push and push-fine, as clean/fg/reference/stages_full.py
// g6_output_push and g6_output_push_fine, bit for bit. Built only when --only names them.
//
// fg_out_push: one lane per element of one level (wf x hf), launched once per level from the coarsest written level
// minus one down to level 1. filled := own (pull's level, bits copied) where own weight > 0 (unordered: a NaN weight
// keeps own), else the 9/3/3/1 upsample of the filled coarser level (wc x hc), 4 x float16. In-place
// pushes only read the old weight, then write the filled pixel where it was missing.
// fg_out_push_fine: one lane per frame pixel; where the fine flag is > 0 (a hole), out := (upsample of filled level 1,
// 1.0) (row pitch out_pitch, offset (out_x, out_y)); other pixels are not written.
//
// fg_out_push_fine_pack (FGU_PACK): writes the presenter slot's packed row instead (fg_pack.hlsli): outp = the packed
// rows' offset, out_pitch = their pitch in bytes, out_x = the format code (1 R8G8B8A8, 2 B8G8R8A8, 3 R10G10B10A2),
// out_y = 0; the pixel is stored as the code fg_exec_pack makes of the float16 pixel fg_out_push_fine writes.
//
// fg_out_push_fine_l1 / fg_out_push_fine_pack_l1 (FGU_L1): level 1 is not pushed beforehand: a pixel of a hole takes the four
// level-1 taps of its upsample as push would have left them, computed on the spot: the own texel (bits as they are) where
// its weight is not <= 0 (a NaN weight keeps it), else the upsample of the filled level 2 (arguments filled2, w2, h2 at the
// end) at that texel, rounded to float16 as the push stores it. Only the taps of hole pixels are ever computed (the launch
// of the level-1 push, a read and a write of the whole level, is dropped).
// fg_out_tail (FGU_TAIL): the pushes of the coarsest levels in one workgroup: the levels (nlev of them, coarsest first:
// arena offset o<j>, size w<j> x h<j>) are loaded into LDS (at most 3584 texels in all), each level below the first is pushed
// in turn (in place, as fg_out_push does, one barrier between levels), and the last (finest) level goes back to memory:
// the same texels as nlev - 1 launches of fg_out_push, the coarser levels not being read again.
//
// @wg-kernel fg_out_push co=fg_full.co block=256 wave=32 args=p,p,p,u,u,u,u
// @wg-kernel fg_out_push_fine co=fg_full.co block=256 wave=32 args=p,p,p,u,u,u,u,u,u,u defines=FGU_FINE=1
// @wg-kernel fg_out_push_fine_pack co=- entry=fg_out_push_fine block=256 wave=32 args=p,p,p,u,u,u,u,u,u,u defines=FGU_FINE=1,FGU_PACK=1
// @wg-kernel fg_out_push_fine_l1 co=- entry=fg_out_push_fine block=256 wave=32 args=p,p,p,u,u,u,u,u,u,u,p,u,u defines=FGU_FINE=1,FGU_L1=1
// @wg-kernel fg_out_push_fine_pack_l1 co=- entry=fg_out_push_fine block=256 wave=32 args=p,p,p,u,u,u,u,u,u,u,p,u,u defines=FGU_FINE=1,FGU_PACK=1,FGU_L1=1
// @wg-kernel fg_out_tail co=- block=256 wave=32 args=u,p,p,p,p,p,p,p,p,u,u,u,u,u,u,u,u,u,u,u,u,u,u,u,u defines=FGU_TAIL=1
#include "wg_platform.hlsli"
#include "fg_full.hlsli"
#include "fg_pack.hlsli"

#ifndef FGU_FINE
#define FGU_FINE 0
#endif
#ifndef FGU_PACK
#define FGU_PACK 0
#endif
#ifndef FGU_L1
#define FGU_L1 0
#endif
#ifndef FGU_TAIL
#define FGU_TAIL 0
#endif
#if FGU_L1 || FGU_TAIL
#include "fg_full_up.hlsli"
#endif

WG_ARGS_BEGIN
#if FGU_TAIL
    uint nlev, o0, o1, o2, o3, o4, o5, o6, o7;
    int lw0, lw1, lw2, lw3, lw4, lw5, lw6, lw7, lh0, lh1, lh2, lh3, lh4, lh5, lh6, lh7;
#elif FGU_FINE
    uint fine, filled1, outp;
    int Wo, Ho, w1, h1, out_pitch, out_x, out_y;
#if FGU_L1
    uint filled2;
    int w2, h2;
#endif
#else
    uint own, coarse, filled;
    int wf, hf, wc, hc;
#endif
WG_ARGS_END

#if FGU_TAIL
static const int FGU_TAIL_CELLS = 3584;     // 28 KB of 8-byte texels
groupshared uint2 fgu_lds[FGU_TAIL_CELLS];

float4 fgu_ld(int base, int i)
{
    const uint2 d = fgu_lds[base + i];
    return float4(wg_h2f_lo(d.x), wg_h2f_hi(d.x), wg_h2f_lo(d.y), wg_h2f_hi(d.y));
}

// fgf_up on a level held in LDS
float4 fgu_up(int base, int wc, int hc, int qx, int qy)
{
    int iP, iX, iY, iD;
    fgf_up_idx(wc, hc, qx, qy, iP, iX, iY, iD);
    return fgf_up_taps(fgu_ld(base, iP), fgu_ld(base, iX), fgu_ld(base, iY), fgu_ld(base, iD));
}

WG_KERNEL(fg_out_tail)
{
    FGF_SETUP();
    const uint lane = wg_lane_id();
    const uint offs[8] = {o0, o1, o2, o3, o4, o5, o6, o7};
    const int ws[8] = {lw0, lw1, lw2, lw3, lw4, lw5, lw6, lw7};
    const int hs[8] = {lh0, lh1, lh2, lh3, lh4, lh5, lh6, lh7};
    int eoff[9];
    eoff[0] = 0;
    [unroll] for (int j = 0; j < 8; ++j) eoff[j + 1] = eoff[j] + (j < (int)nlev ? ws[j] * hs[j] : 0);
    // load every level
    [unroll] for (int j = 0; j < 8; ++j) {
        if (j < (int)nlev) {
            const int n = ws[j] * hs[j];
            for (int i = (int)lane; i < n; i += 256) fgu_lds[eoff[j] + i] = wg_ld64(offs[j], (uint)i * 8u);
        }
    }
    wg_barrier_lds();
    // push each level below the coarsest from the one above it (in place: own texel kept where its weight is not <= 0)
    [unroll] for (int j = 1; j < 8; ++j) {
        if (j < (int)nlev) {
            const int n = ws[j] * hs[j], wf = ws[j];
            for (int i = (int)lane; i < n; i += 256) {
                const uint2 o = fgu_lds[eoff[j] + i];
                if (!(wg_h2f_hi(o.y) <= 0.0f)) continue;
                const int qy = i / wf, qx = i - qy * wf;
                const float4 v = fgu_up(eoff[j - 1], ws[j - 1], hs[j - 1], qx, qy);
                fgu_lds[eoff[j] + i] = uint2(fg_pack_h2(v.x, v.y), fg_pack_h2(v.z, v.w));
            }
            wg_barrier_lds();
        }
    }
    // the finest level back to memory
    [unroll] for (int j = 1; j < 8; ++j) {
        if (j == (int)nlev - 1) {
            const int n = ws[j] * hs[j];
            for (int i = (int)lane; i < n; i += 256) wg_st64(offs[j], (uint)i * 8u, fgu_lds[eoff[j] + i]);
        }
    }
}
#elif FGU_FINE
#if FGU_L1
// level-1 texel idx as the in-place push leaves it: the own bits where the weight is not <= 0, else fgf_up of the filled
// level 2, rounded to float16 (fgf_st4h) and read back
float4 fgu_l1_texel(int idx)
{
    const uint2 o = wg_ld64(filled1, (uint)idx * 8u);
    uint2 b = o;
    if (wg_h2f_hi(o.y) <= 0.0f) {
        const int qy = idx / w1, qx = idx - qy * w1;
        const float4 v = fgf_up(filled2, w2, h2, qx, qy);
        b = uint2(fg_pack_h2(v.x, v.y), fg_pack_h2(v.z, v.w));
    }
    return float4(wg_h2f_lo(b.x), wg_h2f_hi(b.x), wg_h2f_lo(b.y), wg_h2f_hi(b.y));
}
#endif

WG_KERNEL(fg_out_push_fine)
{
    FGF_SETUP();
    const int idx = (int)(wg_block_id() * 256u + wg_lane_id());
    if (idx >= Wo * Ho) return;
    if (!(fgf_ld1h(fine, idx) > 0.0f)) return;
    const int qy = idx / Wo, qx = idx - qy * Wo;
#if FGU_L1
    int iP, iX, iY, iD;
    fgf_up_idx(w1, h1, qx, qy, iP, iX, iY, iD);
    const float4 up = fgf_up_taps(fgu_l1_texel(iP), fgu_l1_texel(iX), fgu_l1_texel(iY), fgu_l1_texel(iD));
#else
    const float4 up = fgf_up(filled1, w1, h1, qx, qy);
#endif
#if FGU_PACK
    const uint2 h = uint2(fg_pack_h2(up.x, up.y), fg_pack_h2(up.z, 1.0f));
    wg_st32(outp, (uint)(qy + out_y) * (uint)out_pitch + (uint)qx * 4u, fgx_pack_pixel(h, (uint)out_x));
#else
    fgf_st4h(outp, (qy + out_y) * out_pitch + qx + out_x, float4(up.xyz, 1.0f));
#endif
}
#else
WG_KERNEL(fg_out_push)
{
    FGF_SETUP();
    const int idx = (int)(wg_block_id() * 256u + wg_lane_id());
    if (idx >= wf * hf) return;
    const uint offset = (uint)idx * 8u;
    if (own == filled) {
        if (!(wg_h2f(wg_ld16(own, offset + 6u)) <= 0.0f)) return;
    } else {
        const uint2 o = wg_ld64(own, offset);
        if (!(wg_h2f_hi(o.y) <= 0.0f)) {
            wg_st64(filled, offset, o);
            return;
        }
    }
    const int qy = idx / wf, qx = idx - qy * wf;
    fgf_st4h(filled, idx, fgf_up(coarse, wc, hc, qx, qy));
}
#endif
