// SPDX-License-Identifier: MIT
// Original exact spec 51.1 scalar dot baseline, 2026-10-18.
// Each finite e4m3 is an integer times 2^-9; products are multiples of 2^-18.
// (32*448^2 + 65504)*2^24 < 2^47, hence every finite partial sum is exact in binary64.
// Deliberately independent of matrix hardware and the existing block/head kernels.
#include "wg_platform.hlsli"
WG_ARGS_BEGIN
    uint pIn, pOut, n;
WG_ARGS_END

#include "rr_exact_math.hlsli"

// One record: 32 A bytes, 32 B bytes, one low-half f16 C in a uint (68 bytes).
// Result: low-half f16 in one uint. In-network zero signs are equivalent by 51.1;
// we match own fmt.mm's +0 convention for a zero product sum and exact cancellation.
WG_KERNEL(rr_exact_dot32)
{
    WG_SETUP();
    const uint i = wg_block_id() * WG_BLOCK + wg_lane_id();
    if (i >= n) return;
    const uint base = pIn + i * 68u;
    const uint cb = wg_ld32(base, 64u) & 0xffffu;
    bool hasNan = (cb & 0x7fffu) > 0x7c00u;
    precise double sum = double(0.0f);
    [loop] for (uint k = 0; k < 32u; ++k) {
        const uint a = wg_ld8(base, k), b = wg_ld8(base, 32u + k);
        hasNan = hasNan || (a & 0x7fu) == 0x7fu || (b & 0x7fu) == 0x7fu;
        precise double product = rr_exact_d8(a) * rr_exact_d8(b);
        sum = sum + product;
    }
    // No binary64 infinity arithmetic needed. Finite products cannot cancel infinite C.
    uint result;
    if (hasNan) result = 0x7e00u;
    else if ((cb & 0x7fffu) == 0x7c00u) result = cb;
    else {
        sum = sum + double(0.0f);
        precise double total = double(f16tof32(cb)) + sum;
        result = total == double(0.0f) ? 0u : rr_exact_r16(total);
    }
    wg_st32(pOut, i * 4u, result);
}

// Independent conversion probe: input raw binary64, output low-half f16 in uint.
WG_KERNEL(rr_exact_round64)
{
    WG_SETUP();
    const uint i = wg_block_id() * WG_BLOCK + wg_lane_id();
    if (i >= n) return;
    const uint2 bits = wg_ld64(pIn, i * 8u);
    wg_st32(pOut, i * 4u, rr_exact_r16(asdouble(bits.x, bits.y)));
}

// Fragment mapping probe; n is a multiple of 32 so no wave is partially active.
// Per lane input: a0,b0,a1,b1 (32 bytes), then eight f32 C (32 bytes).
WG_KERNEL(rr_exact_frag)
{
    WG_SETUP();
    const uint i = wg_block_id() * WG_BLOCK + wg_lane_id();
    if (i >= n) return;
    const uint base = pIn + i * 64u;
    const uint4 ab0 = wg_ld128(base, 0u), ab1 = wg_ld128(base, 16u);
    const v8f r = rr_exact_step2(ab0.xy, ab0.zw, ab1.xy, ab1.zw, wg_load8f(base, 32u));
    [unroll] for (uint v = 0; v < 8u; ++v) wg_st32(pOut, i * 32u + 4u * v, asuint(r.v[v]));
}
