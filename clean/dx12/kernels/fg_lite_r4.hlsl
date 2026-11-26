// SPDX-License-Identifier: MIT
// Frame generation (issue #96), light path: R4, motion and depth processing (clean/fg/reference/stages_common.py
// mvec_process). One lane per scatter-grid pixel: the nearest-depth sample of five (centre and the four diagonal depth
// texels, region reads with the -2^-23 bias), its 32-bit packed depth Zc, and the motion of that sample times the
// motion scales (float32 x 2). D3D12 only.
//
// @wg-kernel fg_lite_mvec co=- block=256 wave=32 args=p,u,u,u,u,u,u,u,u,u,u,u,u,p,u,u,u,u,u,u,u,u,u,u,u,u,p,p,u,u,u,u,u,u
// @wg-kernel fg_lite_mvec_srv co=- entry=fg_lite_mvec block=256 wave=32 args=p,u,u,u,u,u,u,u,u,u,u,u,u,p,u,u,u,u,u,u,u,u,u,u,u,u,p,p,u,u,u,u,u,u rs=direct defines=FG_SRV=1
#include "wg_platform.hlsli"
#include "fg_lite_common.hlsli"

#if FG_SRV
Texture2D<float4> fg_depth_srv : register(t0);
Texture2D<float4> fg_motion_srv : register(t1);
#endif

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
#if FG_SRV
    const uint2 xy = fg_lite_tile_pixel(idx, w1, h1);
    const uint x = xy.x, y = xy.y;
    const uint dst = y * w1 + x;
#else
    const uint x = idx % w1, y = idx / w1;
    const uint dst = idx;
#endif
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
#if FG_SRV
        const float di = fg_depth_srv.Load(int3(fg_nearest_index(XY.x, (int)dw), fg_nearest_index(XY.y, (int)dh), 0)).x;
#else
        const float di = fg_read_nearest(depth, 6u, (int)dw, (int)dh, XY.x, XY.y).x;
#endif
        const float ki = fg_depth_key(di, sk, inverted);
        if (i == 0u) { b = di; kb = ki; }
        else if (ki < kb) { b = di; kb = ki; sel = i; }
    }
    const float z = fg_sat(b);
    const float sx = ox[sel], sy = oy[sel];
    wg_st32(zc, dst * 4u, fg_pack_depth(z, sx > 0.0f, sy > 0.0f, sel == 0u));
    const float2 MXY = fg_region_xy(FG_REGION(m), u + sx, v + sy, true);
#if FG_SRV
    const float2 m = fg_motion_srv.Load(int3(fg_nearest_index(MXY.x, (int)mw), fg_nearest_index(MXY.y, (int)mh), 0)).xy;
#else
    const float2 m = fg_read_nearest(motion, 5u, (int)mw, (int)mh, MXY.x, MXY.y).xy;
#endif
    wg_st64(mv, dst * 8u, uint2(asuint(fg_mul(m.x, asfloat(scx))), asuint(fg_mul(m.y, asfloat(scy)))));
}
