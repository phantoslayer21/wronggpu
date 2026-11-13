// SPDX-License-Identifier: MIT
// Original direct-texture Model M I/O, 2026-10-15. Include after wg_platform.hlsli, only for WG_DIRECT_TEXTURES.
// Basis: clean/runtime/interop/convert.cpp's input pass and its own compiled inputs DXBC (motion is mad),
// clean/dx12/kernels/native_helpers.hlsl's qualified normal-result boundary repairs, and the existing dense layout.
// This header does not change the shared arithmetic helpers or the established buffer-port builds.
#pragma once

cbuffer NativeDirect : register(b1) {
    uint nd_rx, nd_ry, nd_rw, nd_rh;
    float nd_msx, nd_msy, nd_mox, nd_moy;
    float nd_color_pi;
    uint nd_exposure_mode;
    uint nd_pad0, nd_pad1, nd_pad2, nd_pad3, nd_pad4, nd_pad5; // six scalars: total cbuffer size is exactly 64 bytes
};
// Word 10 (NativeDirectParams.output_rne16): 1 = the RGBA16F output gets framebuffer_half's bytes (HIP output-copy rows:
// +-65504 clamp, round to nearest even) instead of the typed store's own conversion (round toward zero on this driver).
#define nd_output_rne16 nd_pad0
// Words 11..13 (NativeDirectParams.pi_rcp / pi_lo / pi_hi): the pre-exposure division window set by the device from
// color_pi (clean/runtime/native_direct_pi.h): y = RN(1 / pi), and the |a| range [lo, hi) of the fast form.
#define nd_pi_rcp asfloat(nd_pad1)
#define nd_pi_lo asfloat(nd_pad2)
#define nd_pi_hi asfloat(nd_pad3)

Texture2D<float4> native_direct_color_tex : register(t0);
Texture2D<float> native_direct_depth_tex : register(t1);
Texture2D<float4> native_direct_motion_tex : register(t2);
Texture2D<float> native_direct_exposure_tex : register(t3);
RWTexture2D<float4> native_direct_output : register(u3);

// Same operations and unique underflow-tie repair as native_color_div in native_helpers.hlsl.
float native_direct_color_div_ref(float a, float b)
{
    const float r = wg_div(a, b);
    const uint ua = asuint(a), ub = asuint(b);
    const uint ea = (ua >> 23u) & 255u, eb = (ub >> 23u) & 255u;
    // For normal operands the only tiny quotient that rounds to minnormal has Ma=2^24-1, Mb=2^23 and Ea-Eb=-127.
    // The test covers all 127 exponent splits and both signs; no new denormal or NaN exception is introduced.
    if ((asuint(r) & 0x7fffffffu) == 0u && ea != 0u && eb != 255u && eb == ea + 127u &&
        (ua & 0x7fffffu) == 0x7fffffu && (ub & 0x7fffffu) == 0u)
        return asfloat(((ua ^ ub) & 0x80000000u) | 0x00800000u);
    return r;
}

// native_direct_color_div_ref(c, pi) per channel with three operations on the window lanes: q0 = a y, rn = pi q0 - a
// (fused), q = q0 - rn y (fused), y = RN(1 / pi) (clean/runtime/native_direct_pi.h has the exactness argument; all
// 2^23 x 2^23 significand pairs checked, build/big-async/r3/mproof). Lanes with |a| in [lo, hi) or |a| below FLT_MIN (zero
// / FP32 denormal: the signed zero, the negated residual keeps the sign of a -0) take it; any other lane (NaN / inf, a
// quotient near the FP32 range ends, a pi outside the window domain: hi = 0) runs native_direct_color_div_ref itself.
// Checked for all 2^32 a against native_direct_color_div_ref on the GPU for a set of pi (build/big-async/r3/pidiv3).
float3 native_direct_color_div3(float3 c)
{
    const float b = nd_color_pi, y = nd_pi_rcp;
    const float3 q0 = c * y;
    const float3 rn = mad(b, q0, -c);
    float3 q = mad(-rn, y, q0);
    const float3 m = abs(c);
    const bool3 fast = and(m < nd_pi_hi, or(m >= nd_pi_lo, m < asfloat(0x00800000u)));
    [branch] if (!all(fast)) {
        q.x = fast.x ? q.x : native_direct_color_div_ref(c.x, b);
        q.y = fast.y ? q.y : native_direct_color_div_ref(c.y, b);
        q.z = fast.z ? q.z : native_direct_color_div_ref(c.z, b);
    }
    return q;
}

// Same operations and exact normal-result underflow repair as native_scale in native_helpers.hlsl.
float native_direct_scale_ref(float a, float b)
{
    const float r = a * b;
    const uint ua = asuint(a), ub = asuint(b);
    const uint ea = (ua >> 23u) & 255u, eb = (ub >> 23u) & 255u;
    if ((asuint(r) & 0x7fffffffu) == 0u && ea != 0u && eb != 0u && ea + eb == 127u) {
        const uint64_t product = (uint64_t)((ua & 0x7fffffu) | 0x800000u) *
                                 (uint64_t)((ub & 0x7fffffu) | 0x800000u);
        const uint64_t normal = (uint64_t)1u << 47u;
        if (product >= normal - ((uint64_t)1u << 23u) && product < normal)
            return asfloat(((ua ^ ub) & 0x80000000u) | 0x00800000u);
    }
    return r;
}

// native_direct_scale_ref's bits: its repair needs a zero product (FTZ: never a denormal) of a nonzero normal a (ea != 0),
// so only such lanes (none in practice) run it; every other lane's result is the product itself.
float native_direct_scale(float a, float b)
{
    float r = a * b;
    [branch] if (and(abs(r) == 0.0f, abs(a) >= asfloat(0x00800000u))) r = native_direct_scale_ref(a, b);
    return r;
}

// All indices are relative to the dense render rectangle. Convert EACH index to (x,y) before adding the origin:
// an original wide buffer load at the right edge crosses to the next dense row, not outside the texture subrect.
int3 native_direct_coord(uint index)
{
    const uint y = index / nd_rw;
    return int3(nd_rx + index - y * nd_rw, nd_ry + y, 0);
}

float4 native_direct_raw_color(uint index)
{
    if (index >= nd_rw * nd_rh) return 0.0f;
    // The original Converter discards source alpha and writes +0 to its dense RGBA buffer.
    return float4(native_direct_color_tex.Load(native_direct_coord(index)).rgb, 0.0f);
}

float4 native_direct_normalized_color(uint index)
{
    const float4 c = native_direct_raw_color(index);
    if (nd_color_pi == 1.0f) return c; // original runtime skips color_divide when pi == 1
    const float3 rgb = native_direct_color_div3(c.rgb);
    // Preserve the removed color_divide store's arithmetic boundary. Valid texture widths make this XOR zero,
    // while the compiler cannot fold operations across it (same original opaque-zero technique as the frame ports).
    return float4(asfloat(asuint(rgb) ^ (nd_rw >> 31u)), c.w);
}

float native_direct_depth(uint index)
{
    if (index >= nd_rw * nd_rh) return 0.0f;
    return native_direct_depth_tex.Load(native_direct_coord(index));
}

float4 native_direct_depth4(uint index)
{
    return float4(native_direct_depth(index), native_direct_depth(index + 1u),
                  native_direct_depth(index + 2u), native_direct_depth(index + 3u));
}

float2 native_direct_motion(uint index)
{
    if (index >= nd_rw * nd_rh) return 0.0f;
    const float2 raw = native_direct_motion_tex.Load(native_direct_coord(index)).xy;
    // The Converter's CS5 inputs DXBC contains mad. Keep this FMA separate from the encoder/post's later motion math.
    const float2 converted = float2(mad(nd_msx, raw.x, nd_mox), mad(nd_msy, raw.y, nd_moy));
    return asfloat(asuint(converted) ^ (nd_rw >> 31u)); // preserve the removed Converter UAV store boundary
}

float4 native_direct_motion2pairs(uint index)
{
    return float4(native_direct_motion(index), native_direct_motion(index + 1u));
}

// Coordinate forms of the loads above (x, y inside the dense render rectangle, 0 <= x < nd_rw, 0 <= y < nd_rh): the
// same texels and the same arithmetic without the index division and the bounds test (a caller passes coordinates it has
// clamped or reflected into the rectangle; the kernels' dense width P_rw is nd_rw: both are the frame's render width).
float4 nd_color_xy(int x, int y)
{
    const float4 c = float4(native_direct_color_tex.Load(int3((int)nd_rx + x, (int)nd_ry + y, 0)).rgb, 0.0f);
    if (nd_color_pi == 1.0f) return c;
    const float3 rgb = native_direct_color_div3(c.rgb);
    return float4(asfloat(asuint(rgb) ^ (nd_rw >> 31u)), c.w);
}

float4 nd_raw_color_xy(int x, int y)
{
    return float4(native_direct_color_tex.Load(int3((int)nd_rx + x, (int)nd_ry + y, 0)).rgb, 0.0f);
}

float2 nd_motion_xy(int x, int y)
{
    const float2 raw = native_direct_motion_tex.Load(int3((int)nd_rx + x, (int)nd_ry + y, 0)).xy;
    const float2 converted = float2(mad(nd_msx, raw.x, nd_mox), mad(nd_msy, raw.y, nd_moy));
    return asfloat(asuint(converted) ^ (nd_rw >> 31u));
}

// Four depth texels x .. x + 3 of row y (x + 3 < nd_rw).
float4 nd_depth4_xy(int x, int y)
{
    const int3 b = int3((int)nd_rx + x, (int)nd_ry + y, 0);
    return float4(native_direct_depth_tex.Load(b), native_direct_depth_tex.Load(b + int3(1, 0, 0)),
                  native_direct_depth_tex.Load(b + int3(2, 0, 0)), native_direct_depth_tex.Load(b + int3(3, 0, 0)));
}

float native_direct_exposure()
{
    return native_direct_exposure_tex.Load(int3(0, 0, 0));
}
