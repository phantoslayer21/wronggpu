// SPDX-License-Identifier: MIT
// Frame generation (issue #96): shared helpers of the Full path's stage kernels (clean/dx12/kernels/fg_full_blend.hlsl,
// fg_out_pull.hlsl, fg_out_push.hlsl, fg_distortion.hlsl), with the bits of clean/fg/reference/stages_full.py.
//
// - Images live in the arena, row-major: 4 x float16 = 8 bytes per texel (r | g << 16, b | a << 16), 2 x float16 = 4
//   bytes, 1 x float16 = 2 bytes, 2 x float32 = 8 bytes; float16 is written with fg_f2h (nearest even).
// - The reference's approximate division is wg_div (IEEE); float -> int is (int) (truncate, saturate, NaN -> 0).
// - Bilinear reads by hand (fgf_tap / fgf_mix): i = floor(t), f = t - i, taps i and i + 1 clamped to the image, weights
//   (1 - fx)(1 - fy), fx (1 - fy), fy (1 - fx), fx fy, value fma(w11, c11, fma(w01, c01, fma(w00, c00, w10 c10))).
//   The four weights pass through fgf_opq (an xor with a zero the compiler cannot see) so that the driver cannot regroup
//   the products that share a factor with the texel values.
// - Morton order inside a 16 x 16 tile: lane bits 0, 2, 4, 6 are x, bits 1, 3, 5, 7 are y, so a lane tree with strides
//   1, 2, 4, ... sums pairs in x, then y, at every scale (the reference's block_sums order).
// The including file includes wg_platform.hlsli first and starts its kernel with FGF_SETUP().
#pragma once
#include "fg_fp16.hlsli"

static uint fgf_z_;   // an opaque zero: wg_grid_ >> 31 (a grid never has 2^31 workgroups)
#define FGF_SETUP() (FG_FP16_SETUP(), fgf_z_ = wg_grid_ >> 31)

float fgf_opq(float x) { return asfloat(asuint(x) ^ fgf_z_); }
float3 fgf_opq3(float3 x) { return asfloat(asuint(x) ^ fgf_z_); }
float fgf_zero() { return asfloat(fgf_z_); }
float fgf_sat(float x) { return min(max(x, 0.0f), 1.0f); }
float3 fgf_sat3(float3 x) { return min(max(x, 0.0f), 1.0f); }

// The refinement in wg_div without exponent normalization. Use it for a positive
// normal divisor and finite operands in the range needed by the Full path.
float fgf_div_reciprocal(float b)
{
    const float y = rcp(b);
    return mad(mad(-b, y, 1.0f), y, y);
}

float3 fgf_div_normal3(float3 a, float b, float y)
{
    float3 q = a * y;
    q = mad(mad(-b, q, a), y, q);
    return mad(mad(-b, q, a), y, q);
}

float fgf_pixel_centre(int i, int n)
{
    const float a = (float)i + 0.5f, b = (float)n;
    const float y = fgf_div_reciprocal(b);
    float q = a * y;
    q = mad(mad(-b, q, a), y, q);
    return fgf_sat(mad(mad(-b, q, a), y, q));
}

// sat((i + 0.5) / n): the normalized centre of pixel i of an n-pixel axis
float fgf_centre(int i, int n) { return fgf_sat(wg_div((float)i + 0.5f, (float)n)); }

// nearest texel of normalized u on an n-texel axis
int fgf_nearest(float u, int n) { return clamp((int)floor(u * (float)n), 0, n - 1); }

uint fgf_mx(uint l) { return (l & 1u) | ((l >> 1u) & 2u) | ((l >> 2u) & 4u) | ((l >> 3u) & 8u); }
uint fgf_my(uint l) { return fgf_mx(l >> 1u); }

// ---- texels -----------------------------------------------------------------------------------------------------------
float4 fgf_ld4h(uint p, int i)
{
    const uint2 d = wg_ld64(p, (uint)i * 8u);
    return float4(wg_h2f_lo(d.x), wg_h2f_hi(d.x), wg_h2f_lo(d.y), wg_h2f_hi(d.y));
}
float2 fgf_ld2h(uint p, int i) { const uint d = wg_ld32(p, (uint)i * 4u); return float2(wg_h2f_lo(d), wg_h2f_hi(d)); }
float fgf_ld1h(uint p, int i) { return wg_h2f(wg_ld16(p, (uint)i * 2u)); }
void fgf_st4h(uint p, int i, float4 v) { wg_st64(p, (uint)i * 8u, uint2(fg_pack_h2(v.x, v.y), fg_pack_h2(v.z, v.w))); }
void fgf_st2h(uint p, int i, float2 v) { wg_st32(p, (uint)i * 4u, fg_pack_h2(v.x, v.y)); }
void fgf_st1h(uint p, int i, float v) { wg_st16(p, (uint)i * 2u, fg_f2h(v)); }

// Adjacent Morton lanes cover an aligned horizontal pair when the row pitch is even.
// Call with the full wave active, including lanes outside the frame.
void fgf_store_morton_flag(uint p, int x, int y, int w, int h, bool flag)
{
    const uint bits = flag ? 0x3c00u : 0u;
    if ((w & 1) == 0 && (p & 3u) == 0u) {
        const uint other = QuadReadAcrossX(bits);
        if ((x & 1) == 0 && x < w && y < h)
            wg_st32(p, (uint)(y * w + x) * 2u, bits | (other << 16));
    } else if (x < w && y < h) {
        wg_st16(p, (uint)(y * w + x) * 2u, bits);
    }
}

// ---- bilinear reads ---------------------------------------------------------------------------------------------------
struct FgfTap { int i00, i10, i01, i11; float w00, w10, w01, w11; };

// taps of texel position (tx, ty) in a W x H image (element indices y * W + x)
FgfTap fgf_tap(float tx, float ty, int W, int H)
{
    FgfTap t;
    const float x0f = floor(tx), y0f = floor(ty);
    const float fx = tx - x0f, fy = ty - y0f;
    const int xi = clamp((int)x0f, -1, W - 1), yi = clamp((int)y0f, -1, H - 1);
    const int x0 = max(xi, 0), x1 = min(xi + 1, W - 1);
    const int y0 = max(yi, 0), y1 = min(yi + 1, H - 1);
    t.i00 = y0 * W + x0; t.i10 = y0 * W + x1; t.i01 = y1 * W + x0; t.i11 = y1 * W + x1;
    const float gx = 1.0f - fx, gy = 1.0f - fy;
    t.w00 = fgf_opq(gx * gy);
    t.w10 = fgf_opq(fx * gy);
    t.w01 = fgf_opq(fy * gx);
    t.w11 = fgf_opq(fx * fy);
    return t;
}

// the same at normalized (u, v): texel position (fma(u, W, -0.5), fma(v, H, -0.5))
FgfTap fgf_tap_uv(float u, float v, int W, int H) { return fgf_tap(mad(u, (float)W, -0.5f), mad(v, (float)H, -0.5f), W, H); }

float4 fgf_mix4(FgfTap t, float4 c00, float4 c10, float4 c01, float4 c11)
{
    return mad(t.w11, c11, mad(t.w01, c01, mad(t.w00, c00, t.w10 * c10)));
}
float2 fgf_mix2(FgfTap t, float2 c00, float2 c10, float2 c01, float2 c11)
{
    return mad(t.w11, c11, mad(t.w01, c01, mad(t.w00, c00, t.w10 * c10)));
}

#ifndef FG_BIL128
#define FG_BIL128 0
#endif
#if FG_BIL128
// A16: the two texels of a tap row are adjacent (i10 == i00 + 1) or the same texel (i10 == i00: the frame's left / right
// border clamps x1 to x0), so one load at i00 covers both: 16 bytes for 4 x float16 texels (8 for 2 x float16), the second
// texel taken from it unless the border made it the first. The same four texel values reach fgf_mix4 / fgf_mix2, so the
// result is bit for bit the 4-load version's. (Reads one texel past i00 at the frame's last texel: robust buffer access.)
float4 fgf_unpack4h(uint2 d) { return float4(wg_h2f_lo(d.x), wg_h2f_hi(d.x), wg_h2f_lo(d.y), wg_h2f_hi(d.y)); }
float4 fgf_bil4h(uint p, FgfTap t)
{
    const uint4 a = wg_ld128(p, (uint)t.i00 * 8u), b = wg_ld128(p, (uint)t.i01 * 8u);
    const uint2 c10 = select(t.i10 == t.i00, a.xy, a.zw), c11 = select(t.i11 == t.i01, b.xy, b.zw);
    return fgf_mix4(t, fgf_unpack4h(a.xy), fgf_unpack4h(c10), fgf_unpack4h(b.xy), fgf_unpack4h(c11));
}
float2 fgf_unpack2h(uint d) { return float2(wg_h2f_lo(d), wg_h2f_hi(d)); }
float2 fgf_bil2h(uint p, FgfTap t)
{
    const uint2 a = wg_ld64(p, (uint)t.i00 * 4u), b = wg_ld64(p, (uint)t.i01 * 4u);
    const uint c10 = select(t.i10 == t.i00, a.x, a.y), c11 = select(t.i11 == t.i01, b.x, b.y);
    return fgf_mix2(t, fgf_unpack2h(a.x), fgf_unpack2h(c10), fgf_unpack2h(b.x), fgf_unpack2h(c11));
}
#else
float4 fgf_bil4h(uint p, FgfTap t) { return fgf_mix4(t, fgf_ld4h(p, t.i00), fgf_ld4h(p, t.i10), fgf_ld4h(p, t.i01), fgf_ld4h(p, t.i11)); }
float2 fgf_bil2h(uint p, FgfTap t) { return fgf_mix2(t, fgf_ld2h(p, t.i00), fgf_ld2h(p, t.i10), fgf_ld2h(p, t.i01), fgf_ld2h(p, t.i11)); }
#endif

// ---- output pyramid -----------------------------------------------------------------------------------------------------
// (sum r, sum g, sum b, sum w) -> (r / max(sum w, 1e-12), ..., sum w / n)
float4 fgf_level(float4 s, float n)
{
    const float den = max(s.w, 1e-12f);
    return float4(wg_div(s.x, den), wg_div(s.y, den), wg_div(s.z, den), wg_div(s.w, n));
}

// The 9/3/3/1 upsample of the filled coarse level lv (wc x hc, 4 x float16) at fine element (qx, qy): (v_up rgb, w_up),
// v_up normalized by clamp(Wsum, 1e-12, 1), w_up = Wsum unclamped. The chains start from fma(., 0.5625, +0) with an
// opaque +0 (the driver would otherwise turn fma(a, b, 0) into a b, which keeps -0).
float4 fgf_up(uint lv, int wc, int hc, int qx, int qy)
{
    const int px = qx >> 1, py = qy >> 1;
    const int nx = clamp(px + ((qx & 1) != 0 ? 1 : -1), 0, wc - 1);
    const int ny = clamp(py + ((qy & 1) != 0 ? 1 : -1), 0, hc - 1);
    const float4 eP = fgf_ld4h(lv, py * wc + px), eX = fgf_ld4h(lv, py * wc + nx);
    const float4 eY = fgf_ld4h(lv, ny * wc + px), eD = fgf_ld4h(lv, ny * wc + nx);
    const float z = fgf_zero();
    const float ws = mad(eD.w, 0.0625f, mad(eY.w, 0.1875f, mad(eX.w, 0.1875f, mad(eP.w, 0.5625f, z))));
    const float3 vs = mad(fgf_opq3(eD.xyz * eD.w), 0.0625f, mad(fgf_opq3(eY.xyz * eY.w), 0.1875f,
                      mad(fgf_opq3(eX.xyz * eX.w), 0.1875f, mad(fgf_opq3(eP.xyz * eP.w), 0.5625f, z))));
    const float den = min(max(ws, 1e-12f), 1.0f);
    return float4(wg_div(vs.x, den), wg_div(vs.y, den), wg_div(vs.z, den), ws);
}
