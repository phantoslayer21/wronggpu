// SPDX-License-Identifier: MIT
// Frame generation (issue #96), light path: R4, motion and depth processing (clean/fg/reference/stages_common.py
// mvec_process). One lane per scatter-grid pixel: the nearest-depth sample of five (centre and the four diagonal depth
// texels, region reads with the -2^-23 bias), its 32-bit packed depth Zc, and the motion of that sample times the
// motion scales (float32 x 2). D3D12 only.
//
// @wg-kernel fg_lite_mvec co=- block=256 wave=32 args=p,u,u,u,u,u,u,u,u,u,u,u,u,p,u,u,u,u,u,u,u,u,u,u,u,u,p,p,u,u,u,u,u,u
#include "wg_platform.hlsli"
#include "fg_lite_common.hlsli"

WG_ARGS_BEGIN
    uint depth, dw, dh;
    FG_REGION_ARGS(d)
    uint motion, mw, mh;
    FG_REGION_ARGS(m)
    uint zc, mv, w1, h1, inverted, s_k, scx, scy;
WG_ARGS_END

WG_KERNEL(fg_lite_mvec)
{
    FG_SETUP();
    const uint idx = wg_block_id() * 256u + wg_lane_id();
    if (idx >= w1 * h1) return;
    const uint x = idx % w1, y = idx / w1;
    const float u = fg_centre(x, w1), v = fg_centre(y, h1);
    const FgRegion dr = FG_REGION(d);
    const float dx = fg_rcp(dr.ew), dy = fg_rcp(dr.eh);
    const float sk = asfloat(s_k);
    float ox[5] = { 0.0f, -dx, dx, -dx, dx };
    float oy[5] = { 0.0f, -dy, -dy, dy, dy };
    float b = 0.0f, kb = 0.0f;
    uint sel = 0u;
    [unroll] for (uint i = 0u; i < 5u; ++i)
    {
        const float2 XY = fg_region_xy(dr, u + ox[i], v + oy[i], true);
        const float di = fg_read_nearest(depth, 6u, (int)dw, (int)dh, XY.x, XY.y).x;
        const float ki = fg_depth_key(di, sk, inverted);
        if (i == 0u) { b = di; kb = ki; }
        else if (ki < kb) { b = di; kb = ki; sel = i; }
    }
    const float z = fg_sat(b);
    const float sx = ox[sel], sy = oy[sel];
    wg_st32(zc, idx * 4u, fg_pack_depth(z, sx > 0.0f, sy > 0.0f, sel == 0u));
    const float2 MXY = fg_region_xy(FG_REGION(m), u + sx, v + sy, true);
    const float4 m = fg_read_nearest(motion, 5u, (int)mw, (int)mh, MXY.x, MXY.y);
    wg_st64(mv, idx * 8u, uint2(asuint(fg_mul(m.x, asfloat(scx))), asuint(fg_mul(m.y, asfloat(scy)))));
}
