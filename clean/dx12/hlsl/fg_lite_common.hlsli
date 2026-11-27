// SPDX-License-Identifier: GPL-3.0-only
// Frame generation (issue #96): shared helpers of the light path's D3D12 kernels (clean/dx12/kernels/fg_lite_*.hlsl).
// They compute the bits of the CPU reference clean/fg/reference/stages_common.py (its module text lists the
// conventions): rcp = the correctly rounded 1 / x (wg_div), the approximate division a * rcp(b), the own deterministic
// 2^y (floor, a degree-6 polynomial in fma form, an exact power-of-two scale), minNum / maxNum clamps, textures emulated
// on arena buffers (nearest: clamp(floor(X * W)); bilinear: fma blends of the four clamped texels around X * W - 0.5),
// application inputs read through their region with the clamp to its texel centres.
//
// Image kinds (element layout in the arena): 1 = float16, 2 = float16 x 2, 3 = float16 x 3, 4 = float16 x 4,
// 5 = float32 x 2, 6 = float32. Row-major, texel (x, y) at element y * W + x.
//
// The driver's compiler regroups chains of operations with constant or uniform terms (clean/dx12/README.md "Driver
// exceptions"); fg_opq() passes a value through an integer xor with a zero it cannot see, which cuts such a chain.
#pragma once
#include "fg_fp16.hlsli"

#ifndef FG_BIL128
#define FG_BIL128 0
#endif
static uint fg_z_;   // 0 at run time (the grid size is below 2^31), unknown to the compiler
#define FG_SETUP() (FG_FP16_SETUP(), fg_z_ = wg_grid_ >> 31)

float fg_opq(float x) { return asfloat(asuint(x) ^ fg_z_); }
// a * b exactly as one rounded product, as an fma with an opaque -0: neither regrouped nor fused
float fg_mul(float a, float b) { return mad(a, b, asfloat(fg_z_ | 0x80000000u)); }

#define FG_INF asfloat(0x7f800000u)
#define FG_BIAS asfloat(0x34000000u)              // 2^-23
#define FG_NEG_LOG2E asfloat(0xBFB8AA3Bu)
#define FG_PUSH asfloat(0xC47A0000u)              // -1000
#define FG_KEY_FLOOR 1e-7f
#define FG_WEIGHT_FLOOR 1e-12f

float fg_rcp(float x) { return wg_div(1.0f, x); }
float fg_adiv(float a, float b) { return fg_mul(a, fg_rcp(b)); }
float fg_sat(float x) { return min(max(x, 0.0f), 1.0f); }
float fg_med3(float x, float lo, float hi) { return min(max(x, lo), hi); }
bool fg_finite(float x) { return abs(x) < FG_INF; }
bool fg_finite3(float3 c) { return fg_finite(c.x) && fg_finite(c.y) && fg_finite(c.z); }
float fg_f(uint bits) { return asfloat(bits); }
int fg_sgn(float x) { return x > 0.0f ? 1 : -1; }
float fg_sq(float x, float y) { return mad(x, x, fg_mul(y, y)); }
// u = sat((i + 0.5) / n), approximate division
float fg_centre(uint i, uint n) { return fg_sat(fg_mul((float)i + 0.5f, fg_rcp((float)n))); }

// Remap the existing 256-thread linear dispatch when the frame fits 32 x 8 tiles.
uint2 fg_lite_tile_pixel(uint idx, uint w, uint h)
{
    if ((w & 31u) == 0u && (h & 7u) == 0u) {
        const uint tile = wg_block_id(), tiles_x = w >> 5u, lane = wg_lane_id();
        const uint ty = tile / tiles_x;
        return uint2(((tile - ty * tiles_x) << 5u) + (lane & 31u), (ty << 3u) + (lane >> 5u));
    }
    const uint y = idx / w;
    return uint2(idx - y * w, y);
}

// 2^y: the coefficients are stages_common.EX2_COEFFS (this project's least-squares fit of 2^f on [0, 1])
float fg_ex2(float y)
{
    float n = floor(y);
    float f = y - n;
    float p = asfloat(0x396566FBu);
    p = mad(p, f, asfloat(0x3AA25EA2u));
    p = mad(p, f, asfloat(0x3C1EAC13u));
    p = mad(p, f, asfloat(0x3D633F72u));
    p = mad(p, f, asfloat(0x3E75FEFCu));
    p = mad(p, f, asfloat(0x3F317214u));
    p = mad(p, f, 1.0f);
    int ni = (int)n;
    float sc = asfloat((uint)(clamp(ni, -126, 127) + 127) << 23);
    float r = fg_mul(p, sc);
    r = ni < -126 ? 0.0f : r;
    r = ni > 127 ? FG_INF : r;
    return isnan(y) ? y : r;
}

// sigma(a - b) as the reference computes it
float fg_sigmoid(float a, float b)
{
    float z = a - b;
    float e = fg_ex2(fg_mul(abs(z), FG_NEG_LOG2E));
    float num = a > b ? 1.0f : e;
    return fg_adiv(num, e + 1.0f);
}

float fg_luma(float3 c)
{
    return fg_sat(mad(c.z, asfloat(0x3DE978D5u), mad(c.x, asfloat(0x3E991687u), fg_mul(c.y, asfloat(0x3F1645A2u)))));
}

// ---- image elements --------------------------------------------------------------------------------------------------
float4 fg_texel(uint base, uint kind, uint idx)
{
    if (kind == 1u) return float4(wg_h2f(wg_ld16(base, idx * 2u)), 0.0f, 0.0f, 0.0f);
    if (kind == 2u) { uint w = wg_ld32(base, idx * 4u); return float4(f16tof32(w), f16tof32(w >> 16), 0.0f, 0.0f); }
    if (kind == 3u)
        return float4(wg_h2f(wg_ld16(base, idx * 6u)), wg_h2f(wg_ld16(base, idx * 6u + 2u)),
                      wg_h2f(wg_ld16(base, idx * 6u + 4u)), 0.0f);
    if (kind == 4u)
    {
        uint2 w = wg_ld64(base, idx * 8u);
        return float4(f16tof32(w.x), f16tof32(w.x >> 16), f16tof32(w.y), f16tof32(w.y >> 16));
    }
    if (kind == 5u) { uint2 w = wg_ld64(base, idx * 8u); return float4(asfloat(w.x), asfloat(w.y), 0.0f, 0.0f); }
    return float4(asfloat(wg_ld32(base, idx * 4u)), 0.0f, 0.0f, 0.0f);
}

void fg_st_h1(uint base, uint idx, float a) { wg_st16(base, idx * 2u, fg_f2h(a)); }
void fg_st_h2(uint base, uint idx, float a, float b) { wg_st32(base, idx * 4u, fg_pack_h2(a, b)); }
void fg_st_h3(uint base, uint idx, float3 a)
{
    wg_st16(base, idx * 6u, fg_f2h(a.x));
    wg_st16(base, idx * 6u + 2u, fg_f2h(a.y));
    wg_st16(base, idx * 6u + 4u, fg_f2h(a.z));
}
void fg_st_h4(uint base, uint idx, float4 a)
{
    wg_st64(base, idx * 8u, uint2(fg_pack_h2(a.x, a.y), fg_pack_h2(a.z, a.w)));
}

// integer read at (x, y), clamped
float4 fg_read_int(uint base, uint kind, int w, int h, int x, int y)
{
    return fg_texel(base, kind, (uint)(clamp(y, 0, h - 1) * w + clamp(x, 0, w - 1)));
}

int fg_nearest_index(float X, int n) { return clamp((int)floor(X * (float)n), 0, n - 1); }

float4 fg_read_nearest(uint base, uint kind, int w, int h, float X, float Y)
{
    return fg_texel(base, kind, (uint)(fg_nearest_index(Y, h) * w + fg_nearest_index(X, w)));
}

float4 fg_read_linear(uint base, uint kind, int w, int h, float X, float Y)
{
    float px = mad(X, (float)w, -0.5f), py = mad(Y, (float)h, -0.5f);
    float fx = floor(px), fy = floor(py);
    float ax = px - fx, ay = py - fy;
    int rx = clamp((int)fx, -1, w - 1), ry = clamp((int)fy, -1, h - 1);
    int x0 = max(rx, 0), x1 = min(rx + 1, w - 1), y0 = max(ry, 0), y1 = min(ry + 1, h - 1);
#if FG_BIL128
    // A16 (kinds 4 and 2): x1 is x0 + 1 or x0 itself (the border clamp), so one load per row covers both texels
    // (16 bytes for float16 x 4, 8 for float16 x 2) and the same four texel values reach the blend below.
    float4 t00, t10, t01, t11;
    if (kind == 4u)
    {
        const uint4 a = wg_ld128(base, (uint)(y0 * w + x0) * 8u), b = wg_ld128(base, (uint)(y1 * w + x0) * 8u);
        const uint2 a1 = select(x1 == x0, a.xy, a.zw), b1 = select(x1 == x0, b.xy, b.zw);
        t00 = float4(f16tof32(a.x), f16tof32(a.x >> 16), f16tof32(a.y), f16tof32(a.y >> 16));
        t10 = float4(f16tof32(a1.x), f16tof32(a1.x >> 16), f16tof32(a1.y), f16tof32(a1.y >> 16));
        t01 = float4(f16tof32(b.x), f16tof32(b.x >> 16), f16tof32(b.y), f16tof32(b.y >> 16));
        t11 = float4(f16tof32(b1.x), f16tof32(b1.x >> 16), f16tof32(b1.y), f16tof32(b1.y >> 16));
    }
    else if (kind == 2u)
    {
        const uint2 a = wg_ld64(base, (uint)(y0 * w + x0) * 4u), b = wg_ld64(base, (uint)(y1 * w + x0) * 4u);
        const uint a1 = select(x1 == x0, a.x, a.y), b1 = select(x1 == x0, b.x, b.y);
        t00 = float4(f16tof32(a.x), f16tof32(a.x >> 16), 0.0f, 0.0f);
        t10 = float4(f16tof32(a1), f16tof32(a1 >> 16), 0.0f, 0.0f);
        t01 = float4(f16tof32(b.x), f16tof32(b.x >> 16), 0.0f, 0.0f);
        t11 = float4(f16tof32(b1), f16tof32(b1 >> 16), 0.0f, 0.0f);
    }
    else
    {
        t00 = fg_texel(base, kind, (uint)(y0 * w + x0)); t10 = fg_texel(base, kind, (uint)(y0 * w + x1));
        t01 = fg_texel(base, kind, (uint)(y1 * w + x0)); t11 = fg_texel(base, kind, (uint)(y1 * w + x1));
    }
#else
    float4 t00 = fg_texel(base, kind, (uint)(y0 * w + x0)), t10 = fg_texel(base, kind, (uint)(y0 * w + x1));
    float4 t01 = fg_texel(base, kind, (uint)(y1 * w + x0)), t11 = fg_texel(base, kind, (uint)(y1 * w + x1));
#endif
    float4 r0 = mad(ax.xxxx, t10 - t00, t00);
    float4 r1 = mad(ax.xxxx, t11 - t01, t01);
    return mad(ay.xxxx, r1 - r0, r0);
}

// ---- regions of application inputs -------------------------------------------------------------------------------
struct FgRegion { float ew, eh, bx, by, rx, ry, lox, hix, loy, hiy; };
#define FG_REGION_ARGS(p) uint p##_ew, p##_eh, p##_bx, p##_by, p##_rx, p##_ry, p##_lox, p##_hix, p##_loy, p##_hiy;
FgRegion fg_region_make(uint ew, uint eh, uint bx, uint by, uint rx, uint ry, uint lox, uint hix, uint loy, uint hiy)
{
    FgRegion r;
    r.ew = asfloat(ew); r.eh = asfloat(eh); r.bx = asfloat(bx); r.by = asfloat(by); r.rx = asfloat(rx);
    r.ry = asfloat(ry); r.lox = asfloat(lox); r.hix = asfloat(hix); r.loy = asfloat(loy); r.hiy = asfloat(hiy);
    return r;
}
#define FG_REGION(p) fg_region_make(p##_ew, p##_eh, p##_bx, p##_by, p##_rx, p##_ry, p##_lox, p##_hix, p##_loy, p##_hiy)

float2 fg_region_xy(FgRegion r, float u, float v, bool bias)
{
    float tx = mad(u, r.ew, r.bx), ty = mad(v, r.eh, r.by);
    float X = bias ? mad(r.rx, tx, -FG_BIAS) : fg_mul(r.rx, tx);
    float Y = bias ? mad(r.ry, ty, -FG_BIAS) : fg_mul(r.ry, ty);
    return float2(fg_med3(X, r.lox, r.hix), fg_med3(Y, r.loy, r.hiy));
}

// ---- depth keys and the packed depth -----------------------------------------------------------------------------
float fg_depth_key(float z, float s_k, uint inverted)
{
    float d = inverted != 0u ? z : 1.0f - z;
    return fg_mul(s_k, fg_rcp(max(d, FG_KEY_FLOOR)));
}

uint fg_pack_depth(float z, bool oxp, bool oyp, bool centre)
{
    uint B = asuint(z);
    uint b29 = (B >> 29) & 1u;
    uint oy = oyp ? 0x80000000u : 0u;
    if (centre) return (b29 << 31) | (B & 0x1FFFFFFFu);
    if (b29 != 0u) return oy | (oxp ? 0x40000000u : 0u) | (B & 0x3FFFFFFFu);
    return oy | 0x40000000u | (B & 0x1FFFFFFEu) | (oxp ? 1u : 0u);
}

// (z, offset x, offset y)
float3 fg_unpack_depth(uint Z, float sx, float sy)
{
    uint b31 = Z >> 31, b30 = (Z >> 30) & 1u, b29 = (Z >> 29) & 1u, b0 = Z & 1u;
    bool centre = b30 == 0u && b29 == 0u;
    uint low = Z & 0x1FFFFFFFu;
    uint zb = centre ? (low | (b31 << 29)) : (low | (b29 << 29));
    bool xpos = (b29 != 0u ? b30 : b0) != 0u;
    float ox = centre ? 0.0f : (xpos ? sx : -sx);
    float oy = centre ? 0.0f : (b31 != 0u ? sy : -sy);
    return float3(asfloat(zb), ox, oy);
}

// camera motion (pixels) of normalised (u, v) at raw depth z with the 4 x 4 matrix m0..m15 (as supplied); (cx, cy, w)
float3 fg_camera(float u, float v, float z, float m0, float m1, float m3, float m4, float m5, float m7, float m8,
                 float m9, float m11, float m12, float m13, float m15, uint w1, uint h1)
{
    float nx = mad(u, 2.0f, -1.0f);
    float ny = 1.0f - (v + v);
    float xp = m12 + mad(m8, z, mad(ny, m4, fg_mul(m0, nx)));
    float yp = m13 + mad(m9, z, mad(ny, m5, fg_mul(m1, nx)));
    float w = m15 + mad(m11, z, mad(ny, m7, fg_mul(m3, nx)));
    float aw = abs(w);
    float cx = fg_mul(nx - fg_adiv(xp, aw), fg_mul((float)w1, -0.5f));
    float cy = fg_mul(ny - fg_adiv(yp, aw), fg_mul((float)h1, 0.5f));
    bool bad = !(w > 0.0f);
    return float3(bad ? 10000.0f : cx, bad ? 10000.0f : cy, w);
}

// a 64-bit scatter element: (vector in float, empty)
float3 fg_vec64(uint base, uint idx)
{
    uint2 e = wg_ld64(base, idx * 8u);
    bool empty = e.x == 0x7FFFFFFFu && e.y == 0x7FFFFFFFu;
    return float3(empty ? 0.0f : f16tof32(e.x >> 16), empty ? 0.0f : f16tof32(e.x & 0xffffu), empty ? 1.0f : 0.0f);
}

uint fg_index_rint(float u, uint n)
{
    float p = fg_opq(mad(u, (float)n, -0.5f));
    return (uint)clamp((int)round(p + (-FG_BIAS)), 0, (int)n - 1);
}

void fg_atomic_min64(uint base, uint idx, uint hi, uint lo)
{
    int64_t v = (int64_t)(((uint64_t)hi << 32) | (uint64_t)lo);
    wg_mem_gc.InterlockedMin64(base + idx * 8u, v);
}
