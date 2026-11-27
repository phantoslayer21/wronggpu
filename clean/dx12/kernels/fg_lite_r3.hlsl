// SPDX-License-Identifier: GPL-3.0-only
// Frame generation (issue #96), light path: R3, downsample to the net1 grid with the UI estimate and the colour alpha
// (clean/fg/reference/stages_common.py downsample_ui). One lane per d x d block (d = FG_D, 2 or 4) of the frame:
// every frame pixel of the block writes Cc (and Uc in case U, Lc with Lite flag 2 in cases A / H); blocks inside the
// net1 grid write Dc (block mean of the source colour), Ld (luma-change minima: min, or the mean of the four 2 x 2
// minima for d = 4) and Um (any UI flag, cases A / H). Pixels outside the frame contribute zeros. D3D12 only.
// flags: bit 0 F1 (UI-recomposition path or Lite flag 1), 1 UI alpha active, 2 hudless active, 3 Lite flag 1,
// 4 Lite flag 2.
//
// @wg-kernel fg_lite_downsample2 co=- entry=fg_lite_downsample block=256 wave=32 args=p,u,u,u,p,u,u,u,u,u,u,u,u,u,u,u,u,p,u,u,u,u,u,u,u,u,u,u,u,u,u,u,p,p,p,p,p,p,p,u,u,u,u,u defines=FG_D=2
// @wg-kernel fg_lite_downsample4 co=- entry=fg_lite_downsample block=256 wave=32 args=p,u,u,u,p,u,u,u,u,u,u,u,u,u,u,u,u,p,u,u,u,u,u,u,u,u,u,u,u,u,u,u,p,p,p,p,p,p,p,u,u,u,u,u defines=FG_D=4
// @wg-kernel fg_lite_downsample2_srv co=- entry=fg_lite_downsample block=256 wave=32 args=p,u,u,u,p,u,u,u,u,u,u,u,u,u,u,u,u,p,u,u,u,u,u,u,u,u,u,u,u,u,u,u,p,p,p,p,p,p,p,u,u,u,u,u rs=direct defines=FG_D=2,FG_SRV=1
// @wg-kernel fg_lite_downsample2_srv_hud_tile co=- entry=fg_lite_downsample block=256 wave=32 args=p,u,u,u,p,u,u,u,u,u,u,u,u,u,u,u,u,p,u,u,u,u,u,u,u,u,u,u,u,u,u,u,p,p,p,p,p,p,p,u,u,u,u,u rs=direct defines=FG_D=2,FG_SRV=1,FG_HUD_TILE=1
// @wg-kernel fg_lite_downsample4_srv co=- entry=fg_lite_downsample block=256 wave=32 args=p,u,u,u,p,u,u,u,u,u,u,u,u,u,u,u,u,p,u,u,u,u,u,u,u,u,u,u,u,u,u,u,p,p,p,p,p,p,p,u,u,u,u,u rs=direct defines=FG_D=4,FG_SRV=1
#include "wg_platform.hlsli"
#include "fg_lite_common.hlsli"

WG_ARGS_BEGIN
    uint colour, cw, cbx, cby;
    uint hud, hw, hh;
    FG_REGION_ARGS(h)
    uint alpha, aw, ah, akind, ach;
    FG_REGION_ARGS(a)
    uint cp, cc, uc, lc, dc, ld, um;
    uint wo, ho, wd, hd, flags;
WG_ARGS_END

#if FG_SRV
Texture2D<float4> fg_colour_srv : register(t0);
Texture2D<float4> fg_hud_srv : register(t1);
Texture2D<float4> fg_alpha_srv : register(t2);

float3 fg_r3_hud_texel(int x, int y)
{
    const float3 c = fg_hud_srv.Load(int3(x, y, 0)).xyz;
    return float3(wg_h2f(fg_f2h(c.x)), wg_h2f(fg_f2h(c.y)), wg_h2f(fg_f2h(c.z)));
}
#endif

uint2 fg_r3_colour_bits(uint x, uint y)
{
#if FG_SRV
    const float3 c = fg_colour_srv.Load(int3(cbx + x, cby + y, 0)).xyz;
    return uint2(fg_pack_h2(c.x, c.y), fg_f2h(c.z));
#else
    return wg_ld64(colour, ((cby + y) * cw + cbx + x) * 8u);
#endif
}

float3 fg_r3_unpack_colour(uint2 c)
{
    return float3(wg_h2f_lo(c.x), wg_h2f_hi(c.x), wg_h2f_lo(c.y));
}

bool fg_r3_special_colour(uint2 c)
{
    const uint3 mag = uint3(c.x, c.x >> 16, c.y) & 0x7fffu;
    // Zero round-trips unchanged. Retain explicit conversion for NaNs and subnormals.
    return any(mag > 0x7c00u) || any((mag - 1u) < 0x03ffu);
}

float3 fg_opq3(float3 v) { return float3(fg_opq(v.x), fg_opq(v.y), fg_opq(v.z)); }

#if FG_HUD_TILE
static const int fg_hud_tile_w = 34, fg_hud_tile_h = 10;
groupshared float fg_hud_r[fg_hud_tile_w * fg_hud_tile_h];
groupshared float fg_hud_g[fg_hud_tile_w * fg_hud_tile_h];
groupshared float fg_hud_b[fg_hud_tile_w * fg_hud_tile_h];
static int2 fg_hud_origin;

float3 fg_r3_hud_tap(int x, int y)
{
    const int2 p = int2(x, y) - fg_hud_origin;
    const int i = p.y * fg_hud_tile_w + p.x;
    return float3(fg_hud_r[i], fg_hud_g[i], fg_hud_b[i]);
}
#elif FG_SRV
#define fg_r3_hud_tap fg_r3_hud_texel
#endif

float3 fg_r3_hud(uint x, uint y, bool linear_filter)
{
    const float2 xy = fg_region_xy(FG_REGION(h), fg_centre(x, wo), fg_centre(y, ho), false);
#if FG_SRV
    if (!linear_filter)
        return fg_r3_hud_texel(fg_nearest_index(xy.x, (int)hw), fg_nearest_index(xy.y, (int)hh));
    const float2 p = mad(xy, float2(hw, hh), -0.5f);
    const float2 f = floor(p), a = p - f;
    const int2 hi = int2(hw, hh) - 1;
    const int2 r = clamp(int2(f), -1, hi);
    const int2 lo = max(r, 0), up = min(r + 1, hi);
    const float3 t00 = fg_r3_hud_tap(lo.x, lo.y), t10 = fg_r3_hud_tap(up.x, lo.y);
    const float3 t01 = fg_r3_hud_tap(lo.x, up.y), t11 = fg_r3_hud_tap(up.x, up.y);
    const float3 r0 = mad(a.xxx, t10 - t00, t00);
    const float3 r1 = mad(a.xxx, t11 - t01, t01);
    return mad(a.yyy, r1 - r0, r0);
#else
    return linear_filter ? fg_read_linear(hud, 4u, (int)hw, (int)hh, xy.x, xy.y).xyz
                         : fg_read_nearest(hud, 4u, (int)hw, (int)hh, xy.x, xy.y).xyz;
#endif
}

// one frame pixel: writes Cc / Uc / Lc, returns the block source colour, the luma change and the UI flag
void fg_r3_pixel(uint x, uint y, uint mode, out float3 s, out float diff, out bool flag)
{
    const uint pi = y * wo + x;
    if (mode == 0u) {
        const uint2 raw = fg_r3_colour_bits(x, y);
        s = fg_r3_unpack_colour(raw);
        uint2 packed = uint2(raw.x, raw.y & 0xffffu);
        float3 c16 = s;
        [branch] if (fg_r3_special_colour(raw)) {
            packed = uint2(fg_pack_h2(s.x, s.y), fg_f2h(s.z));
            c16 = float3(wg_h2f_lo(packed.x), wg_h2f_hi(packed.x), wg_h2f_lo(packed.y));
        }
        wg_st64(cc, pi * 8u, packed);
        diff = abs(fg_luma(c16) - fg_luma(fg_texel(cp, 4u, pi).xyz));
        flag = false;
        return;
    }

    const bool F1 = (mode & 1u) != 0u, A = (mode & 2u) != 0u, Hd = (mode & 4u) != 0u;
    const bool L1 = (mode & 8u) != 0u, L2 = (mode & 16u) != 0u;
    const uint caseU = F1 ? 1u : 0u, caseA = (!F1 && A) ? 1u : 0u, caseH = (!F1 && !A && Hd) ? 1u : 0u;
    const uint2 raw = fg_r3_colour_bits(x, y);
    const float3 c = fg_r3_unpack_colour(raw);
    float3 hc = 0.0f;
    if (caseU != 0u || caseH != 0u || (caseA != 0u && L2))
        hc = fg_r3_hud(x, y, L2);
    float al = 0.0f;
    if (caseU != 0u || caseA != 0u)
    {
        const float2 XY = fg_region_xy(FG_REGION(a), fg_centre(x, wo), fg_centre(y, ho), false);
#if FG_SRV
        const float4 texel = fg_alpha_srv.Load(int3(fg_nearest_index(XY.x, (int)aw),
                                                  fg_nearest_index(XY.y, (int)ah), 0));
        al = wg_h2f(fg_f2h(texel[ach]));
#else
        al = fg_read_nearest(alpha, akind, (int)aw, (int)ah, XY.x, XY.y)[ach];
#endif
    }
    float4 ccv;
    flag = false;
    if (caseU != 0u)
    {
        const float ap = L1 ? (fg_finite(al) ? fg_sat(al) : 1.0f) : al;
        const float apm1 = ap - 1.0f;
        const float3 U = float3(mad(hc.x, apm1, c.x), mad(hc.y, apm1, c.y), mad(hc.z, apm1, c.z));
        const bool nf = !fg_finite3(U);
        const float mn = min(min(U.x, U.y), U.z);
        const bool low = U.x < c.x - (hc.x + hc.x) || U.y < c.y - (hc.y + hc.y) || U.z < c.z - (hc.z + hc.z);
        const bool bad = ap == 1.0f || nf || (!nf && mn <= -1.0f) || low;
        const bool zero = L1 && ap == 0.0f;
        float4 ucv = bad ? float4(c.xyz, 1.0f) : float4(U, ap);
        ucv = zero ? float4(0.0f, 0.0f, 0.0f, 0.0f) : ucv;
        fg_st_h4(uc, pi, ucv);
        ccv = float4(hc.xyz, ap);
        s = hc.xyz;
    }
    else if (caseA != 0u)
    {
        flag = al > asfloat(0x3ECCCCCDu);
        ccv = float4(c.xyz, L2 ? (flag ? 1.0f : 0.0f) : al);
        if (L2) fg_st_h4(lc, pi, float4(hc.xyz, 0.0f));
        s = c.xyz;
    }
    else if (caseH != 0u)
    {
        const float3 dd = abs(c.xyz - hc.xyz);
        flag = max(max(dd.x, dd.y), dd.z) > asfloat(0x3D8F5C29u);
        ccv = float4(c.xyz, L2 ? (flag ? 1.0f : 0.0f) : 0.0f);
        if (L2) fg_st_h4(lc, pi, float4(hc.xyz, 0.0f));
        s = c.xyz;
    }
    else
    {
        ccv = float4(c.xyz, 0.0f);
        s = c.xyz;
    }
    uint2 packed;
    float3 c16;
#if FG_SRV
    [branch] if (caseU == 0u && !fg_r3_special_colour(raw)) {
        packed = uint2(raw.x, (raw.y & 0xffffu) | (fg_f2h(ccv.w) << 16));
        c16 = c;
    } else
#endif
    {
        packed = uint2(fg_pack_h2(ccv.x, ccv.y), fg_pack_h2(ccv.z, ccv.w));
        c16 = fg_r3_unpack_colour(packed);
    }
    wg_st64(cc, pi * 8u, packed);
    const float3 p = fg_texel(cp, 4u, pi).xyz;
    diff = abs(fg_luma(c16) - fg_luma(p));
}

float3 fg_r3_source(uint x, uint y, uint mode)
{
    if (x >= wo || y >= ho) return 0.0f;
    if ((mode & 1u) != 0u) return fg_r3_hud(x, y, (mode & 16u) != 0u);
    return fg_r3_unpack_colour(fg_r3_colour_bits(x, y));
}

float3 fg_r3_ordered_sum(float3 a, float3 b)
{
    return select(isnan(a), a, fg_opq3(a + b));
}

float3 fg_r3_nan_sum(uint X, uint Y, uint mode)
{
    float3 sum = 0.0f;
    [loop] for (uint j = 0u; j < FG_D; j += 2u) {
        float3 strip = 0.0f;
        [loop] for (uint i = 0u; i < FG_D; i += 2u) {
            const uint x = FG_D * X + i, y = FG_D * Y + j;
            const float3 top = fg_r3_ordered_sum(fg_r3_source(x, y, mode), fg_r3_source(x + 1u, y, mode));
            const float3 bottom = fg_r3_ordered_sum(fg_r3_source(x, y + 1u, mode), fg_r3_source(x + 1u, y + 1u, mode));
            const float3 square = fg_r3_ordered_sum(top, bottom);
            strip = i == 0u ? square : fg_r3_ordered_sum(strip, square);
        }
        sum = j == 0u ? strip : fg_r3_ordered_sum(sum, strip);
    }
    return sum;
}

void fg_r3_write(uint x, uint y, float3 c, float diff, bool any_flag, uint mode)
{
    if (x >= wd || y >= hd) return;
    const uint ni = y * wd + x;
    // A commuted instruction can choose either NaN when both operands are NaNs.
    // Reevaluate affected blocks in the baseline's left-first order.
    [branch] if (any(isnan(c))) c = fg_r3_nan_sum(x, y, mode);
    const float scale = 1.0f / (FG_D * FG_D);
    fg_st_h4(dc, ni, float4(fg_mul(c.x, scale), fg_mul(c.y, scale), fg_mul(c.z, scale), 0.0f));
    fg_st_h1(ld, ni, diff);
    if ((mode & 1u) == 0u && (mode & 6u) != 0u) fg_st_h1(um, ni, any_flag ? 1.0f : 0.0f);
}

void fg_r3_wave_m0(uint X, uint Y, uint lane, out float3 result, out float diff)
{
    const uint blocks = 32u / FG_D;
    const uint source = (lane & (blocks - 1u)) * FG_D;
    result = 0.0f;
    diff = 0.0f;
    [loop] for (uint p = 0u; p < FG_D; ++p)
    {
        const uint x = FG_D * X + p * 32u + lane;
        float3 c = 0.0f;
        float d = 0.0f;
        [unroll] for (uint j2 = 0u; j2 < FG_D; j2 += 2u)
        {
            const uint y0 = FG_D * Y + j2, y1 = y0 + 1u;
            const uint pi0 = y0 * wo + x, pi1 = y1 * wo + x;
#if FG_SRV
            const float3 r0 = fg_colour_srv.Load(int3(cbx + x, cby + y0, 0)).xyz;
            const float3 r1 = fg_colour_srv.Load(int3(cbx + x, cby + y1, 0)).xyz;
            const float3 p0 = fg_texel(cp, 4u, pi0).xyz;
            const float3 p1 = fg_texel(cp, 4u, pi1).xyz;
            v8f pv;
            pv.v[0] = fg_opq(r0.x); pv.v[1] = fg_opq(r0.y); pv.v[2] = fg_opq(r0.z);
            pv.v[3] = fg_opq(r1.x); pv.v[4] = fg_opq(r1.y); pv.v[5] = fg_opq(r1.z);
            pv.v[6] = 0.0f; pv.v[7] = 0.0f;
            const uint4 h = wg_to_f16_ags(pv);
            const uint2 pk0 = uint2(h.x, h.y & 0xffffu);
            const uint2 pk1 = uint2((h.y >> 16) | (h.z << 16), h.z >> 16);
            wg_st64(cc, pi0 * 8u, pk0);
            wg_st64(cc, pi1 * 8u, pk1);
            const float3 s0 = fg_r3_unpack_colour(pk0);
            const float3 s1 = fg_r3_unpack_colour(pk1);
            const float dv0 = abs(fg_luma(s0) - fg_luma(p0));
            const float dv1 = abs(fg_luma(s1) - fg_luma(p1));
#else
            const uint2 raw0 = wg_ld64(colour, ((cby + y0) * cw + cbx + x) * 8u);
            const uint2 raw1 = wg_ld64(colour, ((cby + y1) * cw + cbx + x) * 8u);
            const float3 p0 = fg_texel(cp, 4u, pi0).xyz;
            const float3 p1 = fg_texel(cp, 4u, pi1).xyz;
            const float3 s0 = fg_r3_unpack_colour(raw0), s1 = fg_r3_unpack_colour(raw1);
            uint2 pk0 = uint2(raw0.x, raw0.y & 0xffffu), pk1 = uint2(raw1.x, raw1.y & 0xffffu);
            float3 c16_0 = s0, c16_1 = s1;
            [branch] if (fg_r3_special_colour(raw0)) {
                pk0 = uint2(fg_pack_h2(s0.x, s0.y), fg_f2h(s0.z));
                c16_0 = fg_r3_unpack_colour(pk0);
            }
            [branch] if (fg_r3_special_colour(raw1)) {
                pk1 = uint2(fg_pack_h2(s1.x, s1.y), fg_f2h(s1.z));
                c16_1 = fg_r3_unpack_colour(pk1);
            }
            wg_st64(cc, pi0 * 8u, pk0);
            wg_st64(cc, pi1 * 8u, pk1);
            const float dv0 = abs(fg_luma(c16_0) - fg_luma(p0));
            const float dv1 = abs(fg_luma(c16_1) - fg_luma(p1));
#endif
            const float3 row0 = fg_opq3(s0 + QuadReadAcrossX(s0));
            const float pd0 = min(dv0, QuadReadAcrossX(dv0));
            const float3 row1 = fg_opq3(s1 + QuadReadAcrossX(s1));
            const float pd1 = min(dv1, QuadReadAcrossX(dv1));
            const float3 square = fg_opq3(row0 + row1);
            const float m = min(pd0, pd1);
#if FG_D == 4
            const float3 strip = fg_opq3(square + QuadReadAcrossY(square));
            const float md = fg_opq(m + QuadReadAcrossY(m));
            c = j2 == 0u ? strip : fg_opq3(c + strip);
            d = j2 == 0u ? md : fg_opq(d + md);
#else
            c = square;
            d = m;
#endif
        }
        const float3 gathered = WaveReadLaneAt(c, source);
        const float gathered_diff = WaveReadLaneAt(d, source);
        if (lane / blocks == p) {
            result = gathered;
            diff = gathered_diff;
        }
    }
}

void fg_r3_wave(uint X, uint Y, uint lane, uint mode, out float3 result, out float diff, out bool any_flag)
{
    // Each p visits a contiguous 32-pixel strip. Only the first lane of each
    // block supplies the reduction, preserving the left-to-right operand order.
    const uint blocks = 32u / FG_D;
    const uint source = (lane & (blocks - 1u)) * FG_D;
    result = 0.0f;
    diff = 0.0f;
    uint any = 0u;
    [loop] for (uint p = 0u; p < FG_D; ++p)
    {
        const uint x = FG_D * X + p * 32u + lane;
        float3 c = 0.0f, row = 0.0f;
        float d = 0.0f, row_diff = 0.0f;
        uint mask = 0u;
        [loop] for (uint j = 0u; j < FG_D; ++j)
        {
            float3 s;
            float dv;
            bool flag;
            fg_r3_pixel(x, FG_D * Y + j, mode, s, dv, flag);
            const float3 pair = fg_opq3(s + QuadReadAcrossX(s));
            const float pd = min(dv, QuadReadAcrossX(dv));
            mask |= flag ? 1u : 0u;
            if ((j & 1u) == 0u) {
                row = pair;
                row_diff = pd;
            } else {
                const float3 square = fg_opq3(row + pair);
                const float m = min(row_diff, pd);
#if FG_D == 4
                const float3 strip = fg_opq3(square + QuadReadAcrossY(square));
                const float md = fg_opq(m + QuadReadAcrossY(m));
                c = j == 1u ? strip : fg_opq3(c + strip);
                d = j == 1u ? md : fg_opq(d + md);
#else
                c = square;
                d = m;
#endif
            }
        }
        if ((mode & 1u) == 0u && (mode & 6u) != 0u) {
            mask |= QuadReadAcrossX(mask);
#if FG_D == 4
            mask |= QuadReadAcrossY(mask);
#endif
        }
        // Compact the blocks from all strips into one consecutive run of net1 elements.
        const float3 gathered = WaveReadLaneAt(c, source);
        const float gathered_diff = WaveReadLaneAt(d, source);
        const uint gathered_mask = WaveReadLaneAt(mask, source);
        if (lane / blocks == p) {
            result = gathered;
            diff = gathered_diff;
            any = gathered_mask;
        }
    }
    any_flag = any != 0u;
}

void fg_r3_block(uint X, uint Y, uint mode, out float3 sum, out float diff, out bool any_flag)
{
    // A wrapped wave or partial block keeps each lane on its own pixels.
    sum = 0.0f;
    diff = 0.0f;
    any_flag = false;
    [loop] for (uint j = 0u; j < FG_D; j += 2u) {
        float3 strip = 0.0f;
        float strip_diff = 0.0f;
        [loop] for (uint i = 0u; i < FG_D; i += 2u)
        {
            float3 left = 0.0f, top = 0.0f, square = 0.0f;
            float left_diff = 0.0f, top_diff = 0.0f, square_diff = 0.0f;
            [loop] for (uint q = 0u; q < 4u; ++q) {
                const uint x = FG_D * X + i + (q & 1u), y = FG_D * Y + j + (q >> 1u);
                float3 s = 0.0f;
                float d = 0.0f;
                bool flag = false;
                if (x < wo && y < ho) fg_r3_pixel(x, y, mode, s, d, flag);
                any_flag = any_flag || flag;
                if ((q & 1u) == 0u) {
                    left = s;
                    left_diff = d;
                } else {
                    const float3 row = fg_opq3(left + s);
                    const float row_diff = min(left_diff, d);
                    if (q == 1u) {
                        top = row;
                        top_diff = row_diff;
                    } else {
                        square = fg_opq3(top + row);
                        square_diff = min(top_diff, row_diff);
                    }
                }
            }
            strip = i == 0u ? square : fg_opq3(strip + square);
            strip_diff = i == 0u ? square_diff : fg_opq(strip_diff + square_diff);
        }
        sum = j == 0u ? strip : fg_opq3(sum + strip);
        diff = j == 0u ? strip_diff : fg_opq(diff + strip_diff);
    }
}

#if FG_HUD_TILE
WG_KERNEL(fg_lite_downsample)
{
    FG_SETUP();
    const uint tile = wg_block_id(), lane = wg_lane_id(), tiles_x = (wo + 31u) / 32u;
    const uint2 base = uint2((tile % tiles_x) * 32u, (tile / tiles_x) * 8u);
    fg_hud_origin = int2(base) - 1;
    for (uint i = lane; i < fg_hud_tile_w * fg_hud_tile_h; i += 256u) {
        const int2 p = clamp(fg_hud_origin + int2(i % fg_hud_tile_w, i / fg_hud_tile_w), 0, int2(hw, hh) - 1);
        const float3 c = fg_r3_hud_texel(p.x, p.y);
        fg_hud_r[i] = c.x;
        fg_hud_g[i] = c.y;
        fg_hud_b[i] = c.z;
    }
    wg_barrier_lds();
    const uint q = lane & 3u;
    const uint2 p = base + uint2(((lane >> 2u) & 15u) * 2u + (q & 1u), (lane >> 6u) * 2u + (q >> 1u));
    float3 s = 0.0f;
    float dv = 0.0f;
    bool flag = false;
    // The executor selects this entry only for the Lite flag 2 paths.
    const uint mode = (flags & 6u) | 16u;
    if (p.x < wo && p.y < ho) fg_r3_pixel(p.x, p.y, mode, s, dv, flag);
    float3 pair = fg_opq3(s + QuadReadAcrossX(s));
    float d = min(dv, QuadReadAcrossX(dv));
    uint f = flag ? 1u : 0u;
    f |= QuadReadAcrossX(f);
    pair = fg_opq3(pair + QuadReadAcrossY(pair));
    d = min(d, QuadReadAcrossY(d));
    f |= QuadReadAcrossY(f);
    if (q == 0u) fg_r3_write(p.x >> 1u, p.y >> 1u, pair, d, f != 0u, mode);
}
#else
WG_KERNEL(fg_lite_downsample)
{
    FG_SETUP();
    const uint nbx = (wo + FG_D - 1u) / FG_D, nby = (ho + FG_D - 1u) / FG_D;
    const uint idx = wg_block_id() * 256u + wg_lane_id();
    if (idx >= nbx * nby) return;
    const uint2 xy = fg_lite_tile_pixel(idx, nbx, nby);
    const uint X = xy.x, Y = xy.y;
    const uint lane = wg_wave_lane(), Xbase = WaveReadLaneFirst(X);
    const uint mode = (uint)wg_uniform((int)flags);
    float3 sum;
    float diff;
    bool any_flag;
    if (WaveActiveAllEqual(Y) && FG_D * (Xbase + 32u) <= wo && FG_D * (Y + 1u) <= ho)
    {
        [branch] if ((mode & 7u) == 0u) {
            fg_r3_wave_m0(Xbase, Y, lane, sum, diff);
            any_flag = false;
        } else {
            fg_r3_wave(Xbase, Y, lane, mode, sum, diff, any_flag);
        }
    }
    else
    {
        fg_r3_block(X, Y, mode, sum, diff, any_flag);
    }
#if FG_D == 4
    diff = fg_mul(diff, 0.25f);
#endif
    fg_r3_write(X, Y, sum, diff, any_flag, mode);
}
#endif
