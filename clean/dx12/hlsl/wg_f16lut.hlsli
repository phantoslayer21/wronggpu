// SPDX-License-Identifier: MIT
// Exact per-frame table of the direct post kernel's output chain (kernels/post.hlsl, POST_OUT 3, vertical pass): every
// output value is a function of ONE FP16 value, the lz5 filter result r, and three per-frame scalars (dec_scale, the
// pre-exposure pi, NativeDirectParams.output_rne16):
//     q = clamp(r, 1e-4, 3);  v = q < kDecPolyMax ? poly(q) : 2^(kDecK q) - 1;  x = fma(v, dec_scale, -0);
//     pi != 1: x = native_direct_scale(x, pi);  rne16: x = (float)(FP16 RNE)(clamp(x, +-65504))
// so the output is T[r] over all 65536 bit patterns of r. f16lut_build_direct (kernels/exposure.hlsl, with
// wg_f16lut_in.hlsli; round 1: the direct encoder), which runs after the exposure pair and before post in every frame,
// writes T for its frame's (dec_scale, pi, rne16) together with a header naming them; the direct post kernel reads
// T[r] in place of the chain when the header names its own three values (one wave-uniform test per launch), else it runs
// the chain itself. T is a pure function of the header, so a table left from an earlier frame
// with the same header is the same table.
// Buffer (runtime.cpp f16lut, passed to the direct kernels as their colour pointer, which the direct kernels do not
// otherwise use; 0 = no table): [0, 16) header (magic, dec_scale bits, pi bits, rne16), [256, 256 + 256 KB) T (FP32);
// wg_f16lut_in.hlsli puts the encoder's colour table behind it.
// Proof: clean/dx12/f16lut (all 65536 r for a grid of finite (dec_scale, pi) x rne16 against the post's own chain and a
// HIP oracle of post.hip + scale_rgb + framebuffer_half).
// Needs: wg_platform.hlsli and native_direct.hlsli (native_direct_scale) included before it.
#pragma once

#define WG_F16LUT_MAGIC 0x4c363146u
#define WG_F16LUT_DEC 256u
#define WG_F16LUT_BYTES (256u + 65536u * 4u)

// post.hlsl's constants (FP32 bit patterns of post.hip's constexpr values)
#define kF16lutDecK asfloat(0x4113a428u)
#define kF16lutDecLn asfloat(0x40ccac9au)
#define kF16lutPolyMax asfloat(0x3d60233du)
#define kF16lut1em4 asfloat(0x38d1b717u)

// post.hlsl dec_m1_poly
float f16lut_poly(float qc)
{
    float y = qc * kF16lutDecLn;
    float t = mad(y, asfloat(0x39500d01u), asfloat(0x3ab60b61u));
    t = mad(t, y, asfloat(0x3c088889u));
    t = mad(t, y, asfloat(0x3d2aaaabu));
    t = mad(t, y, asfloat(0x3e2aaaabu));
    t = mad(t, y, 0.5f);
    t = mad(t, y, 1.0f);
    return (t * y);
}

// The output value of FP16 bits r (post.hlsl's vertical pass after lz5): z is an opaque zero (mul_nz's addend -0 the
// driver cannot drop). For finite dec_scale and pi no value is NaN (v is finite and positive), so the software RNE of
// the clamped value (wg_f2h) has the bits of the post's driver conversion (v_cvt_f16_f32).
float f16lut_out(uint r, float dec_scale, float pi, bool rne16, uint z)
{
    const float q = min(max(f16tof32(r), kF16lut1em4), 3.0f);
    float v = exp2(q * kF16lutDecK) - 1.0f;
    v = q < kF16lutPolyMax ? f16lut_poly(q) : v;
    float x = mad(v, dec_scale, asfloat(z | 0x80000000u));
    if (pi != 1.0f) x = native_direct_scale(x, pi);
    if (rne16) {
        const float c = select(x > 65504.0f, 65504.0f, select(x < -65504.0f, -65504.0f, x));
        x = f16tof32(wg_f2h(c));
    }
    return x;
}

// Header test: the table may be used for (dec_scale, pi, rne16).
bool f16lut_valid(uint tab, float dec_scale, float pi, uint rne16)
{
    if (tab == 0u) return false;
    const uint4 h = wg_mem.Load4(tab);
    return and(and(h.x == WG_F16LUT_MAGIC, h.y == asuint(dec_scale)), and(h.z == asuint(pi), h.w == rne16));
}

// Generator: entries i = lane, lane + stride, ... of T (lane = a global lane index, stride = the number of generating
// lanes); the lane that is 0 writes the header when every entry is covered and the values are finite (no NaN output).
void f16lut_generate(uint tab, uint lane, uint stride, float dec_scale, float pi, uint rne16, uint z)
{
    for (uint i = lane; i < 65536u; i += stride)
        wg_mem.Store(tab + WG_F16LUT_DEC + 4u * i, asuint(f16lut_out(i, dec_scale, pi, rne16 != 0u, z)));
    const bool finite = and((asuint(dec_scale) & 0x7f800000u) != 0x7f800000u, (asuint(pi) & 0x7f800000u) != 0x7f800000u);
    if (lane == 0u)
        wg_mem.Store4(tab, finite ? uint4(WG_F16LUT_MAGIC, asuint(dec_scale), asuint(pi), rne16) : uint4(0u, 0u, 0u, 0u));
}
