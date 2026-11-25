// SPDX-License-Identifier: MIT
// Frame generation (issue #96), light path: menu detection and the menu output stage (clean/fg/reference/
// stages_common.py detect_* and menu_output). D3D12 only. Built only when --only names them.
//
// fg_lite_detect: one workgroup of 256 lanes, 8 points each = the 64 x 32 grid; kind 0 = UI alpha > threshold, kind 1 =
//   hudless opacity > 0.7; lane 0 decides (count > fraction * 2048, or > 0.3 * 2048), writes DetRes (latch, detected,
//   alpha detector's second value, 0) from the previous DetRes, and disable = base | (latch != 0).
// fg_lite_menu_output: one lane per pixel of the out extent: DetRes[0] != 0 -> the current colour texel; else
//   DetRes[2] != 0 -> rgb = C + (G - H) (1 - a), alpha kept; else nothing.
//
// @wg-kernel fg_lite_detect co=- block=256 wave=32 args=u,p,u,u,u,u,u,u,u,u,u,p,u,u,u,u,u,u,u,u,u,u,u,u,u,p,p,u,u defines=FG_K=1
// @wg-kernel fg_lite_menu_output co=- block=256 wave=32 args=p,p,u,u,u,p,u,u,u,u,u,p,u,u,u,u,u,u,u,u,u,u,u,u,p,u,u,u,u,u,u,u,u,u,u,u,u,u,u defines=FG_K=2
#include "wg_platform.hlsli"
#include "fg_lite_common.hlsli"

#if FG_K == 1
WG_ARGS_BEGIN
    uint kind, img, img_w, img_kind, channel, x0, y0, wr, hr, thr, frac;
    uint hud, hud_w, hud_h, hud_same;
    FG_REGION_ARGS(h)
    uint detres, disable, base, hyst_off;
WG_ARGS_END

groupshared uint fg_count;

WG_KERNEL(fg_lite_detect)
{
    FG_SETUP();
    const uint lane = wg_lane_id();
    if (lane == 0u) fg_count = 0u;
    GroupMemoryBarrierWithGroupSync();
    uint n = 0u;
    FgRegion hr_ = FG_REGION(h);
    [loop] for (uint k = 0u; k < 8u; ++k)
    {
        const uint s = lane * 8u + k, j = s % 64u, kk = s / 64u;
        const uint x = x0 + ((wr - 1u) * j) / 63u, y = y0 + ((hr - 1u) * kk) / 31u;
        if (kind == 0u)
        {
            const float a = fg_texel(img, img_kind, y * img_w + x)[channel];
            n += a > asfloat(thr) ? 1u : 0u;
        }
        else
        {
            const float3 c = fg_texel(img, 4u, y * img_w + x).rgb;
            float3 hc;
            if (hud_same != 0u)
                hc = fg_read_int(hud, 4u, (int)hud_w, (int)hud_h, (int)x, (int)y).rgb;
            else
            {
                const float u = fg_sat(fg_adiv((float)(x - x0) + 0.5f, (float)wr));
                const float v = fg_sat(fg_adiv((float)(y - y0) + 0.5f, (float)hr));
                const float2 XY = fg_region_xy(hr_, u, v, false);
                hc = fg_read_linear(hud, 4u, (int)hud_w, (int)hud_h, XY.x, XY.y).rgb;
            }
            float m[3];
            [unroll] for (uint q = 0u; q < 3u; ++q)
            {
                const float dl = hc[q] - c[q];
                const float a = fg_adiv(dl, hc[q] + 1e-10f);
                const float b = fg_adiv(dl, fg_opq(hc[q] - 1.0f) - 1e-10f);
                m[q] = max(a, b);
            }
            n += max(max(m[0], m[1]), m[2]) > 0.7f ? 1u : 0u;
        }
    }
    InterlockedAdd(fg_count, n);
    GroupMemoryBarrierWithGroupSync();
    if (lane != 0u) return;
    const float f = kind == 0u ? asfloat(frac) : 0.3f;
    const bool detected = (float)fg_count > fg_mul(f, 2048.0f);
    const uint prev1 = wg_ld32(detres, 4u);
    const uint latch = detected ? 0xFFFFFFFFu : (hyst_off != 0u ? 0u : (prev1 != 0u ? 0xFFFFFFFFu : 0u));
    const uint d1 = detected ? 1u : 0u;
    const uint d2 = kind == 0u ? (hyst_off != 0u ? 0u : d1) : 0u;
    wg_st32(detres, 0u, latch);
    wg_st32(detres, 4u, d1);
    wg_st32(detres, 8u, d2);
    wg_st32(detres, 12u, 0u);
    wg_st32(disable, 0u, base | (latch != 0u ? 1u : 0u));
}
#endif

#if FG_K == 2
WG_ARGS_BEGIN
    uint detres, colour, cstride, cbx, cby, outp, ostride, ox, oy, wo, ho;
    uint hud, hud_w, hud_h;
    FG_REGION_ARGS(h)
    uint alpha, a_w, a_h, a_kind, a_ch;
    FG_REGION_ARGS(a)
WG_ARGS_END

WG_KERNEL(fg_lite_menu_output)
{
    FG_SETUP();
    const uint idx = wg_block_id() * 256u + wg_lane_id();
    if (idx >= wo * ho) return;
    const uint x = idx % wo, y = idx / wo;
    const uint oi = (oy + y) * ostride + ox + x;
    if (wg_ld32(detres, 0u) != 0u)
    {
        wg_st64(outp, oi * 8u, wg_ld64(colour, ((cby + y) * cstride + cbx + x) * 8u));
        return;
    }
    if (wg_ld32(detres, 8u) == 0u) return;
    const float u = fg_centre(x, wo), v = fg_centre(y, ho);
    const float4 c = fg_texel(colour, 4u, (cby + y) * cstride + cbx + x);
    const float4 g = fg_texel(outp, 4u, oi);
    const float2 HX = fg_region_xy(FG_REGION(h), u, v, false);
    const float4 hc = fg_read_linear(hud, 4u, (int)hud_w, (int)hud_h, HX.x, HX.y);
    const float2 AX = fg_region_xy(FG_REGION(a), u, v, false);
    const float a = fg_read_nearest(alpha, a_kind, (int)a_w, (int)a_h, AX.x, AX.y)[a_ch];
    const float oma = 1.0f - a;
    const float3 rgb = float3(c.x + fg_mul(g.x - hc.x, oma), c.y + fg_mul(g.y - hc.y, oma), c.z + fg_mul(g.z - hc.z, oma));
    wg_st64(outp, oi * 8u, uint2(wg_pack_h2(rgb.x, rgb.y), wg_pack_h2(rgb.z, g.w)));
}
#endif
