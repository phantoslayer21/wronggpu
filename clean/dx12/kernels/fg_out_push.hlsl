// SPDX-License-Identifier: MIT
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
// @wg-kernel fg_out_push co=fg_full.co block=256 wave=32 args=p,p,p,u,u,u,u
// @wg-kernel fg_out_push_fine co=fg_full.co block=256 wave=32 args=p,p,p,u,u,u,u,u,u,u defines=FGU_FINE=1
#include "wg_platform.hlsli"
#include "fg_full.hlsli"

#ifndef FGU_FINE
#define FGU_FINE 0
#endif

WG_ARGS_BEGIN
#if FGU_FINE
    uint fine, filled1, outp;
    int Wo, Ho, w1, h1, out_pitch, out_x, out_y;
#else
    uint own, coarse, filled;
    int wf, hf, wc, hc;
#endif
WG_ARGS_END

#if FGU_FINE
WG_KERNEL(fg_out_push_fine)
{
    FGF_SETUP();
    const int idx = (int)(wg_block_id() * 256u + wg_lane_id());
    if (idx >= Wo * Ho) return;
    if (!(fgf_ld1h(fine, idx) > 0.0f)) return;
    const int qy = idx / Wo, qx = idx - qy * Wo;
    const float4 up = fgf_up(filled1, w1, h1, qx, qy);
    fgf_st4h(outp, (qy + out_y) * out_pitch + qx + out_x, float4(up.xyz, 1.0f));
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
