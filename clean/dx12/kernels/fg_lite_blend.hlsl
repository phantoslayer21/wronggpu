// SPDX-License-Identifier: MIT
// Frame generation (issue #96), light path: the light candidate blend of preset C (clean/fg/reference/lite.py
// lite_blend). One lane per frame pixel; writes (r, g, b, 1) as float16 x 4 at the output's base offset. flag2: the
// refined candidates (detail transfer from the reference images Lp / Lc where a candidate's source sits on UI and both
// co-located pixels are UI-free); flag1: the time-interpolated UI estimate laid over the result. D3D12 only.
// fg_lite_blend_mut is a negative control for the GPU check (the empty-MB fallback dropped), never used otherwise.
// fg_lite_blend_pack (FG_PACK) writes the presenter slot's packed row instead (fg_pack.hlsli): outp = the packed rows'
// offset, ostride = their pitch in bytes, ox = the format code (1 R8G8B8A8, 2 B8G8R8A8, 3 R10G10B10A2), oy = 0; each
// pixel is stored as the code fg_exec_pack makes of the float16 pixel fg_lite_blend writes.
//
// @wg-kernel fg_lite_blend co=- block=256 wave=32 args=p,p,p,p,p,p,p,p,p,p,u,u,u,u,u,u,u,u,u,u,u,u,u,u,u,u,u,u,u,u
// @wg-kernel fg_lite_blend_pack co=- entry=fg_lite_blend block=256 wave=32 args=p,p,p,p,p,p,p,p,p,p,u,u,u,u,u,u,u,u,u,u,u,u,u,u,u,u,u,u,u,u defines=FG_PACK=1
// @wg-kernel fg_lite_blend_mut co=- entry=fg_lite_blend block=256 wave=32 args=p,p,p,p,p,p,p,p,p,p,u,u,u,u,u,u,u,u,u,u,u,u,u,u,u,u,u,u,u,u defines=FG_MUT=1
#include "wg_platform.hlsli"
#include "fg_lite_common.hlsli"
#include "fg_pack.hlsli"

WG_ARGS_BEGIN
    uint cp, cc, ma, mb, wimg, up, uc, lp, lc, outp;
    uint ostride, ox, oy, wo, ho, wm, hm, tbits, flag1, flag2;
    FG_REGION_ARGS(l)
WG_ARGS_END

#ifndef FG_MUT
#define FG_MUT 0
#endif
#ifndef FG_PACK
#define FG_PACK 0
#endif

bool fg_in_box(float px, float py)
{
    return px >= 0.0f && px < (float)(wo - 1u) && py >= 0.0f && py < (float)(ho - 1u);
}

float3 fg_correct(float3 c, float alpha, float2 uv, float u, float v, bool empty, float4 P0, float4 Q0, uint L,
                  float3 own)
{
    bool ok = fg_finite3(c) && !empty && fg_finite(uv.x) && fg_finite(uv.y) && wo >= 2u && ho >= 2u;
    ok = ok && fg_in_box(mad(uv.x, (float)wo, -0.5f), mad(uv.y, (float)ho, -0.5f));
    ok = ok && fg_finite(u) && fg_finite(v) && fg_in_box(mad(u, (float)wo, -0.5f), mad(v, (float)ho, -0.5f));
    float s = fg_adiv(alpha - 0.5f, asfloat(0x3EE66666u));
    s = s <= 0.0f ? 0.0f : (s >= 1.0f ? 1.0f : s);
    const float g = fg_mul(fg_mul(s, s), 3.0f - (s + s));
    ok = ok && g > 0.0f;
    const float dx = fg_mul(uv.x - u, (float)wo), dy = fg_mul(uv.y - v, (float)ho);
    ok = ok && fg_sq(dx, dy) >= 4.0f;
    ok = ok && fg_finite3(P0.xyz) && fg_finite3(Q0.xyz) && P0.w == 0.0f && Q0.w == 0.0f;
    [branch] if (!ok) return c;

    const FgRegion lr = FG_REGION(l);
    const float2 X1 = fg_region_xy(lr, uv.x, uv.y, false);
    const float3 d1 = fg_read_linear(L, 4u, (int)wo, (int)ho, X1.x, X1.y).xyz;
    const float2 X0 = fg_region_xy(lr, u, v, false);
    const float3 d0 = fg_read_linear(L, 4u, (int)wo, (int)ho, X0.x, X0.y).xyz;
    const float3 dl = own - d0;
    ok = fg_finite3(d1) && fg_finite3(d0) && fg_finite3(dl);
    const float3 ad = abs(dl);
    ok = ok && max(max(ad.x, ad.y), ad.z) <= asfloat(0x3E19999Au);
    const float3 tgt = d1 + dl;
    const float3 corr = mad(g.xxx, float3(fg_sat(tgt.x), fg_sat(tgt.y), fg_sat(tgt.z)) - c, c);
    return ok ? corr : c;
}

WG_KERNEL(fg_lite_blend)
{
    FG_SETUP();
    const uint idx = wg_block_id() * 256u + wg_lane_id();
    if (idx >= wo * ho) return;
    const uint x = idx % wo, y = idx / wo;
    const float u = fg_centre(x, wo), v = fg_centre(y, ho);
    const uint mi = fg_index_rint(v, hm) * wm + fg_index_rint(u, wm);
    const float3 A = fg_vec64(ma, mi), B = fg_vec64(mb, mi);
    const bool eA = A.z != 0.0f, eB = B.z != 0.0f;
    const float4 P0 = fg_texel(cp, 4u, idx), Q0 = fg_texel(cc, 4u, idx);
    const float t = asfloat(tbits);
    const float2 uvP = float2(u + B.x, v + B.y), uvQ = float2(u + A.x, v + A.y);
    const float4 sP = fg_read_linear(cp, 4u, (int)wo, (int)ho, uvP.x, uvP.y);
    const float4 sQ = fg_read_linear(cc, 4u, (int)wo, (int)ho, uvQ.x, uvQ.y);
    float3 cP = sP.xyz, cQ = sQ.xyz;
    const float4 w = fg_read_linear(wimg, 2u, (int)wm, (int)hm, u, v);
    if (flag2 != 0u)
    {
        const float3 ncP = fg_correct(cP, sP.w, uvP, u, v, eB, P0, Q0, lp, P0.xyz);
        const float3 ncQ = fg_correct(cQ, sQ.w, uvQ, u, v, eA, P0, Q0, lc, Q0.xyz);
        cP = ncP;
        cQ = ncQ;
    }
    const float fB = eB ? 1.0f : 0.0f, fA = eA ? 1.0f : 0.0f;
#if FG_MUT
    const float3 a = cP;
#else
    const float3 a = mad(fB.xxx, cQ - cP, cP);
#endif
    const float3 b = mad(fA.xxx, cP - cQ, cQ);
    const float3 rP = mad(a - P0.xyz, w.xxx, P0.xyz);
    const float3 rQ = mad(b - Q0.xyz, w.yyy, Q0.xyz);
    float3 o = mad(rQ - rP, t.xxx, rP);
    if (flag1 != 0u)
    {
        const float4 upv = fg_texel(up, 4u, idx), ucv = fg_texel(uc, 4u, idx);
        const float4 U = mad(ucv - upv, t.xxxx, upv);
        o = mad(o, (1.0f - U.w).xxx, U.xyz);
    }
#if FG_PACK
    const uint2 h = uint2(fg_pack_h2(o.x, o.y), fg_pack_h2(o.z, 1.0f));
    wg_st32(outp, (oy + y) * ostride + x * 4u, fgx_pack_pixel(h, ox));
#else
    fg_st_h4(outp, (oy + y) * ostride + ox + x, float4(o, 1.0f));
#endif
}
