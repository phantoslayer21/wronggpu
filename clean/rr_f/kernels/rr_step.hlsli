// SPDX-License-Identifier: MIT
// The 51.1 matrix step of rr_block.hlsl as a header for the input-stage MLP (rr_input_mlp.hlsl): one 32-input step
// on WMMA fragments with an f16 rounding. RR_STEP 0: variant E (two WMMAs on the f32 accumulator), 1: variant X
// (exact under probe P1's model), 2: exact binary64 scalar step, 3: variant X with a fast path for exact sums (the
// same results as 1). Steps 0-2 copied from rr_block.hlsl (kept byte-identical there).
#pragma once
#ifndef RR_STEP
#define RR_STEP 0
#endif
#if RR_STEP == 2
#include "rr_exact_math.hlsli"
v8f step2(uint2 a0, uint2 b0, uint2 a1, uint2 b1, v8f c)
{
    return rr_exact_step2(a0, b0, a1, b1, c);
}
#elif RR_STEP == 1 || RR_STEP == 3
// Variant X (DESIGN-51.md 7.2): the 51.1 step exact by construction, assuming each WMMA returns the exact sum of its
// 16 products and its start value rounded once to f32 (probe P1). p = WMMA(a, b, 0) and e = WMMA(a, b, -p) give the
// 16-product sum S = p + e exactly (the residual fits in f32); V = c + S0 + S1 is summed with error-free transforms
// (every term a multiple of 2^-24) and rounded once: round to odd at f32, then f32 -> f16 (RNE).
void rr_two_sum(float a, float b, out float s, out float e)
{
    s = a + b;
    const float bb = s - a;
    e = (a - (s - bb)) + (b - bb);
}
float rr_exact5(float c, float p1, float p2, float e1, float e2)
{
    float s, t1, s2, t2, r, t3;
    rr_two_sum(p1, p2, s, t1);
    rr_two_sum(s, c, s2, t2);
    const float rest = (e1 + e2) + (t1 + t2);
    rr_two_sum(s2, rest, r, t3);
    const float ro = asfloat(wg_round_odd(r, t3));
    const float plain = (c + p1) + p2;                        // non-finite start value or NaN products
    return select(isfinite(c) && isfinite(p1) && isfinite(p2), ro, plain);
}
#if RR_STEP == 1
v8f step2(uint2 a0, uint2 b0, uint2 a1, uint2 b1, v8f c)
{
    const v8f z = wg_splat(0.0f);
    const v8f p1 = wg_wmma_fp8(a0, b0, z), p2 = wg_wmma_fp8(a1, b1, z);
    v8f n1, n2;
    [unroll] for (uint i = 0; i < 8; ++i) { n1.v[i] = -p1.v[i]; n2.v[i] = -p2.v[i]; }
    const v8f e1 = wg_wmma_fp8(a0, b0, n1), e2 = wg_wmma_fp8(a1, b1, n2);
    v8f r;
    [unroll] for (uint k = 0; k < 8; ++k) r.v[k] = rr_exact5(c.v[k], p1.v[k], p2.v[k], e1.v[k], e2.v[k]);
    return rr_r16(r);
}
#else
// RR_STEP 3 (input-stage MLP): variant X's results with a fast path for exact sums. With e1 = e2 = 0 and s = p1 + p2,
// s2 = s + c exact (s - p1 == p2 and s - p2 == p1 make rr_two_sum's bb = p2 and its error (p1 - p1) + (p2 - p2) = +-0;
// the same for s2), rr_exact5's rest is +-0 and its result R16(RO(two_sum(s2, +-0))) = R16(s2) for s2 != 0. A wave
// with any other element runs rr_exact5 for those elements (wave-uniform branch, no WMMA inside).
static uint rr_x_fallbacks = 0u;
#ifndef RR_STEP3_MUT
#define RR_STEP3_MUT 0
#endif
v8f step2(uint2 a0, uint2 b0, uint2 a1, uint2 b1, v8f c)
{
    const v8f z = wg_splat(0.0f);
    const v8f p1 = wg_wmma_fp8(a0, b0, z), p2 = wg_wmma_fp8(a1, b1, z);
    v8f n1, n2;
    [unroll] for (uint i = 0; i < 8; ++i) { n1.v[i] = -p1.v[i]; n2.v[i] = -p2.v[i]; }
    const v8f e1 = wg_wmma_fp8(a0, b0, n1), e2 = wg_wmma_fp8(a1, b1, n2);
    v8f r;
    uint ok = 0u;
    [unroll] for (uint k = 0; k < 8; ++k) {
        const float s = p1.v[k] + p2.v[k], s2 = s + c.v[k];
        const bool ex = and(and(and(e1.v[k] == 0.0f, e2.v[k] == 0.0f), and(s - p1.v[k] == p2.v[k], s - p2.v[k] == p1.v[k])),
                            and(and(s2 - s == c.v[k], s2 - c.v[k] == s), s2 != 0.0f));
        r.v[k] = s2;
        ok |= (ex || RR_STEP3_MUT) ? (1u << k) : 0u;     // RR_STEP3_MUT: negative control (every sum taken as exact)
    }
    [branch] if (WaveActiveAnyTrue(ok != 0xffu)) {
        rr_x_fallbacks += 1u;
        [unroll] for (uint k = 0; k < 8; ++k)
            if (((ok >> k) & 1u) == 0u) r.v[k] = rr_exact5(c.v[k], p1.v[k], p2.v[k], e1.v[k], e2.v[k]);
    }
    return rr_r16(r);
}
#endif
#else
// One 51.1 step (variant E): two 16-deep WMMAs on the f32 accumulator, then the f16 rounding.
// RR_VARL 1 (opt-in, NOT exact to variant E or 51.1; tier-B speed variant): no f16 rounding of the step's f32 result
// (chains carry f32 between steps, as variant L of DESIGN-51; consumers that round or quantise do so from f32).
#ifndef RR_VARL
#define RR_VARL 0
#endif
// RR_FADD 1 (opt-in, NOT exact to variant E; clean/rr-f-r7-spec): the two WMMAs start from C = 0 and the previous
// (f16) value is added afterwards with one f32 RNE add before the f16 rounding (the WMMAs no longer wait for it).
// RR_Q8D 1 (opt-in, NOT exact to variant E): at the steps whose result only feeds the Q8 quantisation (step2l(true, ..)
// call sites) the f16 rounding is skipped: Q8 straight from the f32 sum (one rounding instead of f32 -> f16 -> e4m3).
#ifndef RR_FADD
#define RR_FADD 0
#endif
#ifndef RR_Q8D
#define RR_Q8D 0
#endif
v8f step2_raw(uint2 a0, uint2 b0, uint2 a1, uint2 b1, v8f c)
{
#if RR_FADD
    v8f p = wg_wmma_fp8(a0, b0, wg_splat(0.0f));
    p = wg_wmma_fp8(a1, b1, p);
    [unroll] for (uint i = 0; i < 8; ++i) p.v[i] = p.v[i] + c.v[i];
    return p;
#else
    c = wg_wmma_fp8(a0, b0, c);
    return wg_wmma_fp8(a1, b1, c);
#endif
}
v8f step2(uint2 a0, uint2 b0, uint2 a1, uint2 b1, v8f c)
{
    c = step2_raw(a0, b0, a1, b1, c);
#if RR_VARL
    return c;
#else
    return rr_r16(c);
#endif
}
#define RR_STEP2RAW 1
#endif
// clean/rr-f-a14-l01 A5 wmma-chain-interleave (L01_WIL, relaxed body only): independent steps written WMMA-interleaved
// (the first WMMA of every chain, then the second ones). Each chain keeps its two fp8 WMMAs on the same operands in the
// same order (step2 = step2_raw = bv_step there: RR_VARL, no RR_FADD, step E, RR_B32), so the bits are unchanged.
#ifndef L01_WIL
#define L01_WIL 1         // default since a14 (-0.032 / -0.043 ms at 1080p / 1440p Q on top of A2)
#endif
#if L01_WIL && defined(RR_STEP2RAW) && RR_VARL && !RR_FADD && RR_STEP == 0 && defined(RR_B32) && RR_B32
#define RR_WIL 1
#else
#define RR_WIL 0
#endif
void wil2(inout v8f c0, inout v8f c1, uint2 a00, uint2 b00, uint2 a01, uint2 b01, uint2 a10, uint2 b10, uint2 a11, uint2 b11)
{
    c0 = wg_wmma_fp8(a00, b00, c0); c1 = wg_wmma_fp8(a10, b10, c1);
    c0 = wg_wmma_fp8(a01, b01, c0); c1 = wg_wmma_fp8(a11, b11, c1);
}
void wil4(inout v8f c0, inout v8f c1, inout v8f c2, inout v8f c3, uint2 a00, uint2 b00, uint2 a01, uint2 b01,
          uint2 a10, uint2 b10, uint2 a11, uint2 b11, uint2 a20, uint2 b20, uint2 a21, uint2 b21,
          uint2 a30, uint2 b30, uint2 a31, uint2 b31)
{
    c0 = wg_wmma_fp8(a00, b00, c0); c1 = wg_wmma_fp8(a10, b10, c1); c2 = wg_wmma_fp8(a20, b20, c2); c3 = wg_wmma_fp8(a30, b30, c3);
    c0 = wg_wmma_fp8(a01, b01, c0); c1 = wg_wmma_fp8(a11, b11, c1); c2 = wg_wmma_fp8(a21, b21, c2); c3 = wg_wmma_fp8(a31, b31, c3);
}
// step2l(last, ..): step2, except with RR_Q8D on a last step (its result only feeds Q8): no f16 rounding.
#if defined(RR_STEP2RAW) && RR_Q8D
v8f step2l(bool last, uint2 a0, uint2 b0, uint2 a1, uint2 b1, v8f c)
{
    if (last) return step2_raw(a0, b0, a1, b1, c);
    return step2(a0, b0, a1, b1, c);
}
#else
v8f step2l(bool last, uint2 a0, uint2 b0, uint2 a1, uint2 b1, v8f c) { return step2(a0, b0, a1, b1, c); }
#endif
// The step with the result as packed f16 (for chains kept packed between steps, and the scores).
uint4 step2h(uint2 a0, uint2 b0, uint2 a1, uint2 b1, v8f c)
{
#if RR_STEP == 1 || RR_STEP == 2 || RR_STEP == 3
    return rr_h16(step2(a0, b0, a1, b1, c));
#else
    return rr_h16(step2_raw(a0, b0, a1, b1, c));
#endif
}
