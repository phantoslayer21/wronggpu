// SPDX-License-Identifier: MIT
// Frame generation (issue #96), light path: the light candidate blend of preset C (clean/fg/reference/lite.py
// lite_blend). One lane per frame pixel; writes (r, g, b, 1) as float16 x 4 at the output's base offset. flag2: the
// refined candidates (detail transfer from the reference images Lp / Lc where a candidate's source sits on UI and both
// co-located pixels are UI-free); flag1: the time-interpolated UI estimate laid over the result. D3D12 only.
// fg_lite_blend_mut is a negative control for the GPU check (the empty-MB fallback dropped), never used otherwise.
// fg_lite_blend_pack (FG_PACK) writes the presenter slot's packed row instead (fg_pack.hlsli): outp = the packed rows'
// offset, ostride = their pitch in bytes, ox = the format code (1 R8G8B8A8, 2 B8G8R8A8, 3 R10G10B10A2), oy = 0; each
// pixel is stored as the code fg_exec_pack makes of the float16 pixel fg_lite_blend writes.
// FG_WEIGHTS (fg_lite_blend_w, fg_lite_blend_pack_w): the blend weights W are not read from an image (the fg_lite_weights
// launch is dropped): the workgroup, one 32 x 8 tile of frame pixels (tiles_x = ceil(wo / 32), one wave per tile row;
// the launch is tiles_x * ceil(ho / 8) workgroups), computes the W texels its pixels' bilinear taps touch (at most 20 x 8
// of them: the scatter grid is at most half the frame's size) into LDS, each by fg_weight_texel (fg_weights.hlsli: the
// body of fg_lite_weights, the packed float16 x 2 word it stores), and the pixels read them from there with
// fg_read_linear's arithmetic. The tile's bounds come from the taps themselves (the wave minima / maxima of every
// in-frame pixel's four texels). Extra arguments at the end: na, nb (net1's logit pairs, wd x hd), wd, hd; wimg is not
// read. Same bits as fg_lite_weights followed by fg_lite_blend (any pixel order gives the same bits: every pixel is
// independent).
//
// @wg-kernel fg_lite_blend co=- block=256 wave=32 args=p,p,p,p,p,p,p,p,p,p,u,u,u,u,u,u,u,u,u,u,u,u,u,u,u,u,u,u,u,u
// @wg-kernel fg_lite_blend_pack co=- entry=fg_lite_blend block=256 wave=32 args=p,p,p,p,p,p,p,p,p,p,u,u,u,u,u,u,u,u,u,u,u,u,u,u,u,u,u,u,u,u defines=FG_PACK=1
// @wg-kernel fg_lite_blend_b co=- entry=fg_lite_blend block=256 wave=32 args=p,p,p,p,p,p,p,p,p,p,u,u,u,u,u,u,u,u,u,u,u,u,u,u,u,u,u,u,u,u defines=FG_BIL128=1
// @wg-kernel fg_lite_blend_pack_b co=- entry=fg_lite_blend block=256 wave=32 args=p,p,p,p,p,p,p,p,p,p,u,u,u,u,u,u,u,u,u,u,u,u,u,u,u,u,u,u,u,u defines=FG_PACK=1,FG_BIL128=1
// @wg-kernel fg_lite_blend_t co=- entry=fg_lite_blend block=256 wave=32 args=p,p,p,p,p,p,p,p,p,p,u,u,u,u,u,u,u,u,u,u,u,u,u,u,u,u,u,u,u,u defines=FG_TILE=1
// @wg-kernel fg_lite_blend_pack_t co=- entry=fg_lite_blend block=256 wave=32 args=p,p,p,p,p,p,p,p,p,p,u,u,u,u,u,u,u,u,u,u,u,u,u,u,u,u,u,u,u,u defines=FG_PACK=1,FG_TILE=1
// @wg-kernel fg_lite_blend_bt co=- entry=fg_lite_blend block=256 wave=32 args=p,p,p,p,p,p,p,p,p,p,u,u,u,u,u,u,u,u,u,u,u,u,u,u,u,u,u,u,u,u defines=FG_TILE=1,FG_BIL128=1
// @wg-kernel fg_lite_blend_pack_bt co=- entry=fg_lite_blend block=256 wave=32 args=p,p,p,p,p,p,p,p,p,p,u,u,u,u,u,u,u,u,u,u,u,u,u,u,u,u,u,u,u,u defines=FG_PACK=1,FG_TILE=1,FG_BIL128=1
// @wg-kernel fg_lite_blend_mut co=- entry=fg_lite_blend block=256 wave=32 args=p,p,p,p,p,p,p,p,p,p,u,u,u,u,u,u,u,u,u,u,u,u,u,u,u,u,u,u,u,u defines=FG_MUT=1
// @wg-kernel fg_lite_blend_w co=- entry=fg_lite_blend block=256 wave=32 args=p,p,p,p,p,p,p,p,p,p,u,u,u,u,u,u,u,u,u,u,u,u,u,u,u,u,u,u,u,u,p,p,u,u defines=FG_WEIGHTS=1
// @wg-kernel fg_lite_blend_pack_w co=- entry=fg_lite_blend block=256 wave=32 args=p,p,p,p,p,p,p,p,p,p,u,u,u,u,u,u,u,u,u,u,u,u,u,u,u,u,u,u,u,u,p,p,u,u defines=FG_PACK=1,FG_WEIGHTS=1
#include "wg_platform.hlsli"
#include "fg_lite_common.hlsli"
#include "fg_pack.hlsli"

#ifndef FG_MUT
#define FG_MUT 0
#endif
#ifndef FG_PACK
#define FG_PACK 0
#endif
#ifndef FG_WEIGHTS
#define FG_WEIGHTS 0
#endif
#ifndef FG_TILE
#define FG_TILE 0
#endif
#if FG_WEIGHTS
#include "fg_weights.hlsli"
#endif

WG_ARGS_BEGIN
    uint cp, cc, ma, mb, wimg, up, uc, lp, lc, outp;
    uint ostride, ox, oy, wo, ho, wm, hm, tbits, flag1, flag2;
    FG_REGION_ARGS(l)
#if FG_WEIGHTS
    uint na, nb, wd, hd;
#endif
WG_ARGS_END

#if FG_WEIGHTS
static const int FG_WTX = 20, FG_WTY = 8;   // the W tile (the taps of 32 x 8 pixels span at most 18 x 5 texels)
groupshared uint fg_wtile[FG_WTX * FG_WTY];
groupshared int4 fg_wbox[8];
static int fg_wx0, fg_wy0;                  // the tile's first texel

// fg_read_linear's texel coordinates and fractions, bit for bit
struct FgTapL { int x0, x1, y0, y1; float ax, ay; };
FgTapL fg_tap_l(float X, float Y, int w, int h)
{
    FgTapL t;
    const float px = mad(X, (float)w, -0.5f), py = mad(Y, (float)h, -0.5f);
    const float fx = floor(px), fy = floor(py);
    t.ax = px - fx;
    t.ay = py - fy;
    const int rx = clamp((int)fx, -1, w - 1), ry = clamp((int)fy, -1, h - 1);
    t.x0 = max(rx, 0); t.x1 = min(rx + 1, w - 1);
    t.y0 = max(ry, 0); t.y1 = min(ry + 1, h - 1);
    return t;
}

float2 fg_wld(int x, int y)
{
    const uint d = fg_wtile[min((uint)((y - fg_wy0) * FG_WTX + (x - fg_wx0)), (uint)(FG_WTX * FG_WTY - 1))];
    return float2(f16tof32(d), f16tof32(d >> 16));
}

// fg_read_linear(wimg, 2u, wm, hm, u, v).xy on the LDS tile
float2 fg_read_linear_w(FgTapL t)
{
    const float2 t00 = fg_wld(t.x0, t.y0), t10 = fg_wld(t.x1, t.y0), t01 = fg_wld(t.x0, t.y1), t11 = fg_wld(t.x1, t.y1);
    const float2 r0 = mad(t.ax.xx, t10 - t00, t00);
    const float2 r1 = mad(t.ax.xx, t11 - t01, t01);
    return mad(t.ay.xx, r1 - r0, r0);
}
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
#if FG_WEIGHTS
    const uint lane = wg_lane_id(), tile = wg_block_id(), tiles_x = (wo + 31u) >> 5u;
    const uint tile_y = tile / tiles_x;
    const uint x = ((tile - tile_y * tiles_x) << 5u) + (lane & 31u), y = (tile_y << 3u) + (lane >> 5u);
    const bool live = x < wo && y < ho;
    const uint idx = y * wo + x;
    const float u = fg_centre(x, wo), v = fg_centre(y, ho);
    FgTapL wtap;
    wtap.x0 = wtap.x1 = wtap.y0 = wtap.y1 = 0;
    wtap.ax = wtap.ay = 0.0f;
    if (live) wtap = fg_tap_l(u, v, (int)wm, (int)hm);
    {
        const int bx0 = WaveActiveMin(live ? wtap.x0 : 0x7fffffff), bx1 = WaveActiveMax(live ? wtap.x1 : -1);
        const int by0 = WaveActiveMin(live ? wtap.y0 : 0x7fffffff), by1 = WaveActiveMax(live ? wtap.y1 : -1);
        if (wg_wave_lane() == 0u) fg_wbox[lane >> 5] = int4(bx0, bx1, by0, by1);
    }
    wg_barrier_lds();
    int wx1, wy1;
    {
        int4 box = fg_wbox[0];
        [unroll] for (uint wi = 1u; wi < 8u; ++wi) {
            const int4 o = fg_wbox[wi];
            box = int4(min(box.x, o.x), max(box.y, o.y), min(box.z, o.z), max(box.w, o.w));
        }
        fg_wx0 = box.x;
        wx1 = box.y;
        fg_wy0 = box.z;
        wy1 = box.w;
    }
    if (lane < (uint)(FG_WTX * FG_WTY)) {
        const int gx = fg_wx0 + (int)(lane % (uint)FG_WTX), gy = fg_wy0 + (int)(lane / (uint)FG_WTX);
        uint wword = 0u;
        if (fg_wx0 <= wx1 && fg_wy0 <= wy1 && gx <= wx1 && gy <= wy1)
            wword = fg_weight_texel(ma, mb, na, nb, (uint)gx, (uint)gy, wm, hm, wd, hd);
        fg_wtile[lane] = wword;
    }
    wg_barrier_lds();
    if (!live) return;
#elif FG_TILE
    // A17: 32 x 8 pixel tiles (tiles_x = ceil(wo / 32), the launch is tiles_x * ceil(ho / 8) workgroups, one wave per tile
    // row) instead of 256-pixel scanlines: no per-pixel division, and the vertical bilinear taps of a tile share cache
    // lines. A lane past the frame computes the nearest valid pixel (the wave's quad exchange needs every lane) and does
    // not store. Every pixel is independent, so the bits are the scanline kernel's.
    const uint lane = wg_lane_id(), tile = wg_block_id(), tiles_x = (wo + 31u) >> 5u;
    const uint tile_y = tile / tiles_x;
    const uint xr = ((tile - tile_y * tiles_x) << 5u) + (lane & 31u), yr = (tile_y << 3u) + (lane >> 5u);
    const bool live = xr < wo && yr < ho;
    const uint x = min(xr, wo - 1u), y = min(yr, ho - 1u);
    const uint idx = y * wo + x;
    const float u = fg_centre(x, wo), v = fg_centre(y, ho);
#else
    const uint idx = wg_block_id() * 256u + wg_lane_id();
    if (idx >= wo * ho) return;
    const uint x = idx % wo, y = idx / wo;
    const float u = fg_centre(x, wo), v = fg_centre(y, ho);
#endif
    const uint mi = fg_index_rint(v, hm) * wm + fg_index_rint(u, wm);
    const float3 A = fg_vec64(ma, mi), B = fg_vec64(mb, mi);
    const bool eA = A.z != 0.0f, eB = B.z != 0.0f;
    const float4 P0 = fg_texel(cp, 4u, idx), Q0 = fg_texel(cc, 4u, idx);
    const float t = asfloat(tbits);
    const float2 uvP = float2(u + B.x, v + B.y), uvQ = float2(u + A.x, v + A.y);
    const float4 sP = fg_read_linear(cp, 4u, (int)wo, (int)ho, uvP.x, uvP.y);
    const float4 sQ = fg_read_linear(cc, 4u, (int)wo, (int)ho, uvQ.x, uvQ.y);
    float3 cP = sP.xyz, cQ = sQ.xyz;
#if FG_WEIGHTS
    const float4 w = float4(fg_read_linear_w(wtap), 0.0f, 0.0f);
#else
    const float4 w = fg_read_linear(wimg, 2u, (int)wm, (int)hm, u, v);
#endif
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
#if FG_TILE
    // adjacent lanes are a horizontal pair (the tile origin is a multiple of 32): one 8-byte store per pair
    const uint word = fgx_pack_pixel(h, ox);
    const uint pair = QuadReadAcrossX(word);
    if (live && (xr & 1u) == 0u)
    {
        const uint at = (oy + y) * ostride + x * 4u;
        if (x + 1u < wo) wg_st64(outp, at, uint2(word, pair));
        else wg_st32(outp, at, word);
    }
#else
    wg_st32(outp, (oy + y) * ostride + x * 4u, fgx_pack_pixel(h, ox));
#endif
#else
#if FG_TILE
    if (live)
#endif
    fg_st_h4(outp, (oy + y) * ostride + ox + x, float4(o, 1.0f));
#endif
}
