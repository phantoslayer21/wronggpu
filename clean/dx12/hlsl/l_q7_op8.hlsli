// SPDX-License-Identifier: GPL-3.0-only
// One FP8 matrix op of the q7 D3D12 ports, exact (l_q7_op8.h), and the 32-value FP8 row loader it is fed with:
// apart from the FP16 helpers (l_q7_common.hlsli) so that the FP8 op (S2a) and the FP16 steps (S2b)
// change in separate files. Include after l_q7_common.hlsli.
#pragma once

float op8(float start, float a[32], float wv[32])
{
    float acc = 0.0f, aacc = 0.0f;
    [unroll] for (uint k = 0; k < 32; ++k) {
        float p = a[k] * wv[k];
        acc = acc + p;
        aacc = aacc + abs(p);
    }
    if (!fin(acc) || !fin(start)) return h16(acc + start);
    float s = acc, c = 0.0f;
    if (!(aacc < 63.0f)) {
        s = 0.0f;
        [unroll] for (uint k2 = 0; k2 < 32; ++k2) {
            float e;
            two_sum(s, a[k2] * wv[k2], s, e);
            c = c + e;
        }
    }
    float u, e1, v, e2, z, e3, t;
    two_sum(s, start, u, e1);
    two_sum(e1, c, v, e2);
    two_sum(u, v, z, e3);
    t = e3 + e2;
    float z2, t2;
    two_sum(z, t, z2, t2);
    return rn16_pair(z2, t2);
}

// ---- S2a: the FP8 matrix instruction under a precomputed bound (l_q7_op8.h, same names) ------
// Wave32 only (wg_wmma_fp8 of wg_platform.hlsli; no wave64 matrix code until an on-device layout probe passes for it).
// The calls must stay in wave-uniform control flow: q7_tile_sum is called unconditionally, the per-lane fallback
// branch comes after the sums.
#ifndef Q7_S2A_MUTANT
#define Q7_S2A_MUTANT 0
#endif
#if Q7_S2A_MUTANT == 1
#define Q7_S2A_LIMIT 128.0f
#else
#define Q7_S2A_LIMIT 63.0f
#endif

v8f q7_tile_sum(uint2 a0, uint2 b0, uint2 a1, uint2 b1)
{
    v8f zero;
    [unroll] for (uint i = 0; i < 8; ++i) zero.v[i] = 0.0f;
    v8f lo = wg_wmma_fp8(a0, b0, zero);
#if Q7_S2A_MUTANT == 2
    return wg_wmma_fp8(a1, b1, lo);
#else
    v8f hi = wg_wmma_fp8(a1, b1, zero);
    v8f s;
    [unroll] for (uint j = 0; j < 8; ++j) s.v[j] = lo.v[j] + hi.v[j];
    return s;
#endif
}
uint q7_umax(uint a, uint b) { return a > b ? a : b; }
uint q7_maxcode(uint m, uint code) { return q7_umax(m, code & 0x7fu); }
uint q7_maxcode4(uint m, uint word)
{
    m = q7_maxcode(m, word);
    m = q7_maxcode(m, word >> 8);
    m = q7_maxcode(m, word >> 16);
    return q7_maxcode(m, word >> 24);
}
float q7_amax(uint m0, uint m1) { return fp8v(q7_umax(m0, m1)); }
bool q7_bounded(float wsum, float amax) { return wsum * amax < Q7_S2A_LIMIT; }
float q7_join(float S, float start) { return add16(S == 0.0f ? 0.0f : S, start); }
// (q7_combine: HLSL has no lambdas; the ports write  y = q7_bounded(..) ? q7_join(S, start) : <exact op8>  inline)

// 32 FP8 values from global memory (byte offset a multiple of 4)
void q7_load8(uint base, uint off, out float v[32])
{
    [unroll] for (uint i = 0; i < 8; ++i) {
        uint u = wg_ld32(base, off + 4u * i);
        [unroll] for (uint j = 0; j < 4; ++j) v[4u * i + j] = fp8v(u >> (8u * j));
    }
}

// 32 FP16 weights from global memory (byte offset a multiple of 16): exact values, one native conversion each (Q7F-1)
void q7_load16(uint base, uint off, out float v[32])
{
    [unroll] for (uint i = 0; i < 4; ++i) {
        uint4 u = wg_ld128(base, off + 16u * i);
        uint q[4] = { u.x, u.y, u.z, u.w };
        [unroll] for (uint j = 0; j < 4; ++j) {
            v[8u * i + 2u * j] = f16tof32(q[j] & 0xffffu);
            v[8u * i + 2u * j + 1u] = f16tof32(q[j] >> 16);
        }
    }
}
