// SPDX-License-Identifier: MIT
// The front-end MLP's 51.1 matrix steps for kernels that fold the MLP into another stage (rr_block_c32.hlsl
// RR_INFOLD), independent of that kernel's own RR_STEP / RR_VARL: copies of rr_step.hlsli's variant E (RR_STEP 0,
// RR_VARL 0) and of variant X with the fast path for exact sums (RR_STEP 3), the same operations in the same order
// (without the RR_STEP 3 diagnostic counter).
#pragma once
v8f mlp_step2e(uint2 a0, uint2 b0, uint2 a1, uint2 b1, v8f c)
{
    c = wg_wmma_fp8(a0, b0, c);
    c = wg_wmma_fp8(a1, b1, c);
    return rr_r16(c);
}
uint4 mlp_step2he(uint2 a0, uint2 b0, uint2 a1, uint2 b1, v8f c)
{
    c = wg_wmma_fp8(a0, b0, c);
    c = wg_wmma_fp8(a1, b1, c);
    return rr_h16(c);
}
void mlp_two_sum(float a, float b, out float s, out float e)
{
    s = a + b;
    const float bb = s - a;
    e = (a - (s - bb)) + (b - bb);
}
float mlp_exact5(float c, float p1, float p2, float e1, float e2)
{
    float s, t1, s2, t2, r, t3;
    mlp_two_sum(p1, p2, s, t1);
    mlp_two_sum(s, c, s2, t2);
    const float rest = (e1 + e2) + (t1 + t2);
    mlp_two_sum(s2, rest, r, t3);
    const float ro = asfloat(wg_round_odd(r, t3));
    const float plain = (c + p1) + p2;
    return select(isfinite(c) && isfinite(p1) && isfinite(p2), ro, plain);
}
v8f mlp_step2x(uint2 a0, uint2 b0, uint2 a1, uint2 b1, v8f c)
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
        ok |= ex ? (1u << k) : 0u;
    }
    [branch] if (WaveActiveAnyTrue(ok != 0xffu)) {
        [unroll] for (uint k = 0; k < 8; ++k)
            if (((ok >> k) & 1u) == 0u) r.v[k] = mlp_exact5(c.v[k], p1.v[k], p2.v[k], e1.v[k], e2.v[k]);
    }
    return rr_r16(r);
}
uint4 mlp_step2hx(uint2 a0, uint2 b0, uint2 a1, uint2 b1, v8f c) { return rr_h16(mlp_step2x(a0, b0, a1, b1, c)); }
#if RR_INFOLD == 3
#define MLP_STEP2 mlp_step2x
#define MLP_STEP2H mlp_step2hx
#else
#define MLP_STEP2 mlp_step2e
#define MLP_STEP2H mlp_step2he
#endif
