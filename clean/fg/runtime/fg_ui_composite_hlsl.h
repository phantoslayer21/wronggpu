// SPDX-License-Identifier: GPL-3.0-only
#pragma once
// Clean-room HLSL compute shaders (cs_5_1) for HUD-less scene separation and post-composite UI restoration.
// Bound to FgD3d12Executor's shared root signature:
//   u0 (space0): arena RWByteAddressBuffer
//   b0 (space0): root 32-bit constants (wg_group_base_, wg_grid_, args...)
//   t0..t1 (space0): colour and hudless SRVs (for the direct-SRV mask variant)

namespace wgfg {

inline constexpr const char* kFgUiCompositeHlsl = R"HLSL(
RWByteAddressBuffer wg_mem : register(u0);

// IEEE-754 FP32 -> FP16 bits, round to nearest even (matches fg_exec_capture / fg_lite_r3).
uint ui_f2h(float f)
{
    uint u = asuint(f);
    uint s = (u >> 16) & 0x8000u;
    uint a = u & 0x7fffffffu;
    uint nrm = (a + 0xC8000FFFu + ((a >> 13) & 1u)) >> 13;
    uint den = asuint(asfloat(a) + asfloat(0x3F000000u)) - 0x3F000000u;
    uint r = a < 0x38800000u ? den : nrm;
    r = a >= 0x47800000u ? (a > 0x7f800000u ? (0x7e00u | ((a >> 13) & 0x3ffu)) : 0x7c00u) : r;
    return s | r;
}

bool ui_finite3(uint2 raw)
{
    return ((raw.x & 0x7c00u) != 0x7c00u) && (((raw.x >> 16) & 0x7c00u) != 0x7c00u) && ((raw.y & 0x7c00u) != 0x7c00u);
}

float3 ui_unpack_h3(uint2 raw)
{
    return float3(f16tof32(raw.x), f16tof32(raw.x >> 16), f16tof32(raw.y));
}

#if UI_MASK

#if UI_SRV
cbuffer wg_args : register(b0) {
    uint wg_group_base_;
    uint wg_grid_;
    uint cbx, cby, hbx, hby, uc_off, flags_off, wo, ho;
};
Texture2D<float4> fg_col_srv : register(t0);
Texture2D<float4> fg_hud_srv : register(t1);

uint2 ui_load_col(uint px, uint py)
{
    float3 c = fg_col_srv.Load(int3(cbx + px, cby + py, 0)).xyz;
    return uint2(ui_f2h(c.x) | (ui_f2h(c.y) << 16), ui_f2h(c.z));
}

uint2 ui_load_hud(uint px, uint py)
{
    float3 c = fg_hud_srv.Load(int3(hbx + px, hby + py, 0)).xyz;
    return uint2(ui_f2h(c.x) | (ui_f2h(c.y) << 16), ui_f2h(c.z));
}
#else
cbuffer wg_args : register(b0) {
    uint wg_group_base_;
    uint wg_grid_;
    uint col_off, cw, cbx, cby, hud_off, hw, hbx, hby, uc_off, flags_off, wo, ho;
};

uint2 ui_load_col(uint px, uint py)
{
    return wg_mem.Load2(col_off + ((cby + py) * cw + (cbx + px)) * 8u);
}

uint2 ui_load_hud(uint px, uint py)
{
    return wg_mem.Load2(hud_off + ((hby + py) * hw + (hbx + px)) * 8u);
}
#endif

float ui_pixel_diff(uint2 raw_b, uint2 raw_h)
{
    if (!ui_finite3(raw_b) || !ui_finite3(raw_h)) return 0.0f;
    float3 b = ui_unpack_h3(raw_b);
    float3 h = ui_unpack_h3(raw_h);
    float3 ad = abs(b - h);
    float mx = max(ad.x, max(ad.y, ad.z));
    float3 ab = abs(b);
    float3 ah = abs(h);
    float sb = max(ab.x, max(ab.y, ab.z));
    float sh = max(ah.x, max(ah.y, ah.z));
    float scale = max(1.0f, max(sb, sh));
    return mx / scale;
}

groupshared float gs_d[324];
groupshared uint gs_any;

[numthreads(256, 1, 1)]
void main(uint tid : SV_GroupThreadID, uint gid : SV_GroupID)
{
    const uint tile = gid + wg_group_base_;
    const uint tiles_x = (wo + 15u) >> 4u;
    const uint tx = tile % tiles_x;
    const uint ty = tile / tiles_x;
    const uint lx = tid & 15u;
    const uint ly = tid >> 4u;
    const uint px = (tx << 4u) + lx;
    const uint py = (ty << 4u) + ly;
    const bool live = px < wo && py < ho;
    const uint cpx = min(px, wo - 1u);
    const uint cpy = min(py, ho - 1u);

    if (tid == 0u) gs_any = 0u;

    const uint2 raw_b = ui_load_col(cpx, cpy);
    const uint2 raw_h = ui_load_hud(cpx, cpy);
    gs_d[(ly + 1u) * 18u + (lx + 1u)] = ui_pixel_diff(raw_b, raw_h);

    // 68-pixel 1-texel halo around the 16x16 tile for opposite-neighbor closing.
    if (tid < 68u) {
        uint hx, hy;
        if (tid < 18u) {
            hx = tid;
            hy = 0u;
        } else if (tid < 36u) {
            hx = tid - 18u;
            hy = 17u;
        } else if (tid < 52u) {
            hx = 0u;
            hy = (tid - 36u) + 1u;
        } else {
            hx = 17u;
            hy = (tid - 52u) + 1u;
        }
        const int gx = clamp((int)(tx << 4u) + (int)hx - 1, 0, (int)wo - 1);
        const int gy = clamp((int)(ty << 4u) + (int)hy - 1, 0, (int)ho - 1);
        gs_d[hy * 18u + hx] = ui_pixel_diff(ui_load_col((uint)gx, (uint)gy), ui_load_hud((uint)gx, (uint)gy));
    }

    GroupMemoryBarrierWithGroupSync();

    // 1-px opposite-neighbor morphological closing fills 1-px interior glyph gaps without outer dilation.
    const uint c = (ly + 1u) * 18u + (lx + 1u);
    const float d0 = gs_d[c];
    const float hv = max(min(gs_d[c - 1u], gs_d[c + 1u]), min(gs_d[c - 18u], gs_d[c + 18u]));
    const float dg = max(min(gs_d[c - 19u], gs_d[c + 19u]), min(gs_d[c - 17u], gs_d[c + 17u]));
    const float cl = max(d0, max(hv, dg));

    // Smooth mask: rejects <= 5/255 8-bit dither, reaches full opacity at >= 24/255.
    const float t_lo = 6.0f / 255.0f;
    const float inv_span = 255.0f / 18.0f;
    const float t = saturate((cl - t_lo) * inv_span);
    const float m = t * t * (3.0f - 2.0f * t);
    const uint mh = ui_f2h(m);

    if (live && mh != 0u) InterlockedOr(gs_any, 1u);
    GroupMemoryBarrierWithGroupSync();

    if (tid == 0u) wg_mem.Store(flags_off + tile * 4u, gs_any);
    if (gs_any == 0u) return;

    if (live) {
        wg_mem.Store2(uc_off + (py * wo + px) * 8u, uint2(raw_b.x, (raw_b.y & 0xffffu) | (mh << 16)));
    }
}

#else

cbuffer wg_args : register(b0) {
    uint wg_group_base_;
    uint wg_grid_;
    uint uc_off, flags_off, out_off, out_stride, wo, ho, slot_fmt;
};

uint ui_unorm(float v, float mx)
{
    float s = min(max(v, 0.0f), 1.0f);
    return (uint)floor(mad(s, mx, 0.5f));
}

uint ui_pack_pixel(uint2 v, uint fmt)
{
    float r = f16tof32(v.x);
    float g = f16tof32(v.x >> 16);
    float b = f16tof32(v.y);
    float a = f16tof32(v.y >> 16);
    if (fmt == 1u)
        return ui_unorm(r, 255.0f) | (ui_unorm(g, 255.0f) << 8) | (ui_unorm(b, 255.0f) << 16) |
               (ui_unorm(a, 255.0f) << 24);
    if (fmt == 2u)
        return ui_unorm(b, 255.0f) | (ui_unorm(g, 255.0f) << 8) | (ui_unorm(r, 255.0f) << 16) |
               (ui_unorm(a, 255.0f) << 24);
    return ui_unorm(r, 1023.0f) | (ui_unorm(g, 1023.0f) << 10) | (ui_unorm(b, 1023.0f) << 20) |
           (ui_unorm(a, 3.0f) << 30);
}

float3 ui_unpack_pixel(uint w, uint fmt)
{
    if (fmt == 1u)
        return float3((float)(w & 255u), (float)((w >> 8) & 255u), (float)((w >> 16) & 255u)) * (1.0f / 255.0f);
    if (fmt == 2u)
        return float3((float)((w >> 16) & 255u), (float)((w >> 8) & 255u), (float)(w & 255u)) * (1.0f / 255.0f);
    return float3((float)(w & 1023u), (float)((w >> 10) & 1023u), (float)((w >> 20) & 1023u)) * (1.0f / 1023.0f);
}

[numthreads(256, 1, 1)]
void main(uint tid : SV_GroupThreadID, uint gid : SV_GroupID)
{
    const uint tile = gid + wg_group_base_;
    if (wg_mem.Load(flags_off + tile * 4u) == 0u) return;

    const uint tiles_x = (wo + 15u) >> 4u;
    const uint px = ((tile % tiles_x) << 4u) + (tid & 15u);
    const uint py = ((tile / tiles_x) << 4u) + (tid >> 4u);
    if (px >= wo || py >= ho) return;

    const uint2 uc = wg_mem.Load2(uc_off + (py * wo + px) * 8u);
    const float m = f16tof32(uc.y >> 16);
    if (m <= 0.0f) return;

    const float3 ui = ui_unpack_h3(uc);
#if UI_PACK
    const uint addr = py * out_stride + px * 4u;
    const uint word = wg_mem.Load(out_off + addr);
    const float3 gen = ui_unpack_pixel(word, slot_fmt);
    const float3 out_c = (m >= 1.0f) ? ui : (gen * (1.0f - m) + ui * m);
    const uint2 h = uint2(ui_f2h(out_c.x) | (ui_f2h(out_c.y) << 16), ui_f2h(out_c.z) | 0x3C000000u);
    const uint amask = (slot_fmt == 3u) ? 0xC0000000u : 0xFF000000u;
    const uint packed = (ui_pack_pixel(h, slot_fmt) & ~amask) | (word & amask);
    wg_mem.Store(out_off + addr, packed);
#else
    const uint addr = (py * out_stride + px) * 8u;
    const uint2 raw_g = wg_mem.Load2(out_off + addr);
    float3 gen;
    if (slot_fmt != 0u)
        gen = ui_unpack_pixel(ui_pack_pixel(raw_g, slot_fmt), slot_fmt);
    else
        gen = ui_finite3(raw_g) ? ui_unpack_h3(raw_g) : ui;
    const float3 out_c = (m >= 1.0f) ? ui : (gen * (1.0f - m) + ui * m);
    const uint2 h = uint2(ui_f2h(out_c.x) | (ui_f2h(out_c.y) << 16), ui_f2h(out_c.z) | (raw_g.y & 0xFFFF0000u));
    wg_mem.Store2(out_off + addr, h);
#endif
}

#endif
)HLSL";

}  // namespace wgfg
