// SPDX-License-Identifier: GPL-3.0-only
// Frame generation (issue #96): the optional distortion stages, as clean/fg/reference/stages_full.py r5_downsample and
// r5_fields, bit for bit. Built only when --only names them.
//
// fg_dist_down: one lane per scatter-grid pixel (W1 x H1): the application's distortion field (tw x th, 4 x float32)
// read nearest at the region-mapped position of the pixel centre (region (bx, by, ew, eh); rx, ry = 1 / tw, 1 / th and
// the clamp bounds lox, hix, loy, hiy come from the host as float bits), out = fma(scale, field, bias) as 4 x float16
// ((bias, scale) = (0, 1) unless the low-precision setting gives them).
// fg_dist_fields: one lane per scatter-grid pixel: from the downsampled current / previous fields Dc, Dp (4 x float16),
// the packed depth Zc (32-bit words), the processed motion Mv (2 x float32, normalized) and the previous -> current
// field F (2 x float16, pixels): the distorted packed depth zd, the distorted current -> previous field mcp (normalized)
// and the distorted previous -> current field fout (pixels), both 2 x float16.
//
// @wg-kernel fg_dist_down co=fg_full.co block=256 wave=32 args=p,p,u,u,u,u,u,u,u,u,u,u,u,u,u,u,u,u
// @wg-kernel fg_dist_fields co=fg_full.co block=256 wave=32 args=p,p,p,p,p,p,p,p,u,u defines=FGD_FIELDS=1
#include "wg_platform.hlsli"
#include "fg_full.hlsli"

#ifndef FGD_FIELDS
#define FGD_FIELDS 0
#endif

WG_ARGS_BEGIN
#if FGD_FIELDS
    uint dc, dp, zc, mv, fpc, zd, mcp, fout;
    int W1, H1;
#else
    uint field, outp;
    int W1, H1, tw, th, bx, by, ew, eh;
    uint rx, ry, lox, hix, loy, hiy, bias, scale;   // float bits
#endif
WG_ARGS_END

#if FGD_FIELDS
WG_KERNEL(fg_dist_fields)
{
    FGF_SETUP();
    const int idx = (int)(wg_block_id() * 256u + wg_lane_id());
    if (idx >= W1 * H1) return;
    const int y = idx / W1, x = idx - y * W1;
    const float u = fgf_centre(x, W1), v = fgf_centre(y, H1);
    const float W1f = (float)W1, H1f = (float)H1;
    const float4 dcu = fgf_bil4h(dc, fgf_tap_uv(u, v, W1, H1));
    const float p1x = u + dcu.x, p1y = v + dcu.y;
    const int i1 = fgf_nearest(p1y, H1) * W1 + fgf_nearest(p1x, W1);
    wg_st32(zd, (uint)idx * 4u, wg_ld32(zc, (uint)i1 * 4u));
    const float2 m = asfloat(wg_ld64(mv, (uint)i1 * 8u));
    const float p2x = p1x + m.x, p2y = p1y + m.y;
    const float4 q = fgf_bil4h(dp, fgf_tap_uv(p2x, p2y, W1, H1));
    fgf_st2h(mcp, idx, float2((p2x + q.z) - u, (p2y + q.w) - v));
    const float4 dpu = fgf_bil4h(dp, fgf_tap_uv(u, v, W1, H1));
    const float rx = u + dpu.x, ry = v + dpu.y;
    const float2 f = fgf_ld2h(fpc, fgf_nearest(ry, H1) * W1 + fgf_nearest(rx, W1));
    const float sx = rx + wg_div(f.x, W1f), sy = ry + wg_div(f.y, H1f);
    const float4 dcs = fgf_bil4h(dc, fgf_tap_uv(sx, sy, W1, H1));
    fgf_st2h(fout, idx, float2(((sx + dcs.z) - u) * W1f, ((sy + dcs.w) - v) * H1f));
}
#else
WG_KERNEL(fg_dist_down)
{
    FGF_SETUP();
    const int idx = (int)(wg_block_id() * 256u + wg_lane_id());
    if (idx >= W1 * H1) return;
    const int y = idx / W1, x = idx - y * W1;
    const float u = fgf_centre(x, W1), v = fgf_centre(y, H1);
    const float X = min(max(asfloat(rx) * mad(u, (float)ew, (float)bx), asfloat(lox)), asfloat(hix));
    const float Y = min(max(asfloat(ry) * mad(v, (float)eh, (float)by), asfloat(loy)), asfloat(hiy));
    const float4 f = wg_ld4f(field, (uint)(fgf_nearest(Y, th) * tw + fgf_nearest(X, tw)) * 16u);
    fgf_st4h(outp, idx, mad(asfloat(scale), f, asfloat(bias)));
}
#endif
