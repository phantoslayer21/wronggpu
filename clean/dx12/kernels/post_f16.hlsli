// SPDX-License-Identifier: MIT
// Fast exact FP32 -> FP16 conversions of the post port (kernels/post.hlsl): the same bits as wg_f2h / wg_mul_f16
// (wg_math.hlsli; HIP's v_cvt_f16_f32 and v_fma_mixlo_f16 a, b, 0) for the value ranges of that kernel, in fewer
// instructions. Checked on the GPU against those helpers (2026-10-15): 6.5 million inputs each (the display range, the
// log2 range, every binade from 2^-30 to 2^15 incl. FP16 subnormal results, every FP16 tie point and its FP32
// neighbours, products with kEncK and random factors): 0 differences; post_mul_f16 also equal to the correctly rounded
// product (numpy, float64 -> float16).
// Needs: opq() (an opaque identity: the driver must not fold (x + c) - c) defined before inclusion.
#pragma once

// RNE of x to FP16 for |x| < 32768 and x not NaN (FP32 denormal x as zero, the accepted FTZ case): |x| + c - c with
// c = 2^(e + 13) (at least 2^-1, the scale of the FP16 subnormal spacing 2^-24) rounds |x| to nearest even at the FP16
// spacing in one FP32 addition (the sum stays in [c, 2c)); the exact result then converts with the RTZ conversion.
uint post_f2h(float x)
{
    uint ux = asuint(x), ax = ux & 0x7fffffffu;
    float c = asfloat(max((ax & 0x7f800000u) + 0x06800000u, 0x3f000000u));
    float r = opq(asfloat(ax) + c) - c;
    return f32tof16(r) | ((ux >> 16) & 0x8000u);
}

// a b rounded to odd at FP32 precision (exact error by fma; 13 spare bits over FP16, so a round-to-nearest-even
// conversion of it to FP16 is the single rounding of the exact product: no double rounding), a and b finite; an exact
// zero product is +0 (as v_fma_mixlo_f16 a, b, 0 gives).
float post_rto_mul(float a, float b)
{
    float p = a * b;
    float e = mad(a, b, -p);
    uint up = asuint(p);
    bool fix = e != 0.0f && (up & 1u) == 0u;
    up = fix ? up + (((asuint(e) ^ up) & 0x80000000u) != 0u ? 0xffffffffu : 1u) : up;   // one ulp toward the exact value
    return asfloat(p == 0.0f && e == 0.0f ? 0u : up);
}

// (FP16)(a b) with one rounding as v_fma_mixlo_f16 a, b, 0 for |a b| < 32768, a and b finite (any control flow; the post
// kernel converts post_rto_mul with the driver's conversion instead, in wave-uniform control flow).
uint post_mul_f16(float a, float b) { return post_f2h(post_rto_mul(a, b)); }
