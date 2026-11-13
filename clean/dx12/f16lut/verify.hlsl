// SPDX-License-Identifier: MIT
// D3D12 side of the f16lut proof (run.py): for every FP16 bit pattern r (lane r of a 65536-lane grid, wave64 as the post
// kernel) and one parameter set (root constants: dec_scale bits, pi bits, rne16):
//   orig[r] = the direct post kernel's vertical-pass chain (kernels/post.hlsl POST_OUT 3, copied: clampf, exp2, the
//             wave-uniform polynomial branch, mul_nz, native_direct_scale, fb_clamp + cvt3_f16)
//   gen[r]  = f16lut_out (hlsl/wg_f16lut.hlsli), the encoder's table generator
// written at obase + 4 r and obase + 262144 + 4 r.
#include "wg_platform.hlsli"
#include "wg_ags.hlsli"
#include "native_direct.hlsli"
#include "wg_f16lut.hlsli"

WG_ARGS_BEGIN
    uint obase, dec_bits, pi_bits, rne16;
WG_ARGS_END

// ---- copied from kernels/post.hlsl ----
static const float kDecK = asfloat(0x4113a428u);
static const float kDecLn = asfloat(0x40ccac9au);
static const float kDecPolyMax = asfloat(0x3d60233du);
static const float kP5040 = asfloat(0x39500d01u), kP720 = asfloat(0x3ab60b61u), kP120 = asfloat(0x3c088889u);
static const float kP24 = asfloat(0x3d2aaaabu), kP6 = asfloat(0x3e2aaaabu);
static const float k1em4 = asfloat(0x38d1b717u);
static uint g_z;
float mul_nz(float a, float b) { return mad(a, b, asfloat(g_z | 0x80000000u)); }
uint2 cvt3_f16(float v0, float v1, float v2)
{
    wg_frag f;
    f.v[0] = asuint(v0); f.v[1] = asuint(v1); f.v[2] = asuint(v2);
    [unroll] for (uint i = 3; i < 8; ++i) f.v[i] = 0u;
    wg_frag r = wg_ags_convert(f, WG_AGS_T_F32_ACC, WG_AGS_T_F16_ACC);
    return uint2(r.v[0], r.v[1] & 0xffffu);
}
float clampf(float v, float lo, float hi) { return min(max(v, lo), hi); }
float dec_m1_poly(float qc)
{
    float y = qc * kDecLn;
    float t = mad(y, kP5040, kP720);
    t = mad(t, y, kP120);
    t = mad(t, y, kP24);
    t = mad(t, y, kP6);
    t = mad(t, y, 0.5f);
    t = mad(t, y, 1.0f);
    return (t * y);
}
float fb_clamp(float v) { return select(v > 65504.0f, 65504.0f, select(v < -65504.0f, -65504.0f, v)); }
// ---------------------------------------

WG_KERNEL(f16lut_verify)
{
    WG_SETUP();
    g_z = wg_grid_ >> 31;
    const uint r = wg_block_id() * WG_BLOCK + wg_lane_id();
    const float dec_scale = asfloat(dec_bits), pi = asfloat(pi_bits);
    // post.hlsl's vertical pass for the three channels (here all three = r)
    float vo[3], qv[3];
    bool vsmall = false;
    [unroll] for (int c = 0; c < 3; ++c) {
        qv[c] = clampf(f16tof32(r), k1em4, 3.0f);
        vo[c] = exp2(qv[c] * kDecK) - 1.0f;
        vsmall = or(vsmall, qv[c] < kDecPolyMax);
    }
    if (WaveActiveAnyTrue(vsmall)) {
        [unroll] for (int c = 0; c < 3; ++c) vo[c] = qv[c] < kDecPolyMax ? dec_m1_poly(qv[c]) : vo[c];
    }
    float3 result = float3(mul_nz(vo[0], dec_scale), mul_nz(vo[1], dec_scale), mul_nz(vo[2], dec_scale));
    if (pi != 1.0f) {
        result.x = native_direct_scale(result.x, pi);
        result.y = native_direct_scale(result.y, pi);
        result.z = native_direct_scale(result.z, pi);
    }
    if (rne16 != 0u) {
        const uint2 hb = cvt3_f16(fb_clamp(result.x), fb_clamp(result.y), fb_clamp(result.z));
        result = float3(f16tof32(hb.x & 0xffffu), f16tof32(hb.x >> 16), f16tof32(hb.y));
    }
    float gen = f16lut_out(r, dec_scale, pi, rne16 != 0u, g_z);
#if defined(F16LUT_NEGATIVE)
    gen = asfloat(asuint(gen) ^ (r == 0x3c00u ? 1u : 0u));   // sensitivity: exactly one input per set changed
#endif
    // the three channels must agree (one chain); a disagreement is reported as a difference of orig
    const uint o = asuint(result.x) == asuint(result.y) && asuint(result.x) == asuint(result.z) ? asuint(result.x) : 0xdeadbeefu;
    wg_mem.Store(obase + 4u * r, o);
    wg_mem.Store(obase + 262144u + 4u * r, asuint(gen));
}
