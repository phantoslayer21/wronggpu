// SPDX-License-Identifier: MIT
// Exhaustive check of the D3D12 hidden-path conversion h -> q8(GELU(h)) against HIP's run-length form (gelu_lut/host.cpp
// "dx12" mode, driven by run.py). Every lane takes 8 consecutive FP32 bit patterns u0 .. u0 + 7, converts them as the MLP
// kernels do (8 values -> 2 dwords of FP8 bytes) and compares each byte with HIP's (binary search in pos[]).
//   default:           wg_gelu_q8_lut (hlsl/wg_gelu_lut.hlsli; the table at arena offset WG_GELU_LUT_BASE)
//   GELU_VERIFY_ORIG:  the transcendental path of the base ports (bneck.hlsl gelu_q8), as a check of the harness
// res: [0] differences, [1] of them on FP32-denormal inputs, [2] on NaN inputs, [4 + 2k], [5 + 2k] examples (k < 30).
#include "wg_platform.hlsli"
#include "wg_gelu_lut.hlsli"

WG_ARGS_BEGIN
    uint a_base, a_pos, a_byt, a_n, a_res, a_tab;
WG_ARGS_END

#if GELU_VERIFY_ORIG
uint2 convert8(v8f h)
{
    v8f a, g;
    [unroll] for (uint v = 0; v < 8; ++v) a.v[v] = h.v[v] * h.v[v];
    [unroll] for (uint v1 = 0; v1 < 8; ++v1) a.v[v1] = mad(asfloat(0xbdcd6502u), a.v[v1], asfloat(0xc013b8edu));
    [unroll] for (uint v2 = 0; v2 < 8; ++v2) a.v[v2] = h.v[v2] * a.v[v2];
    [unroll] for (uint v3 = 0; v3 < 8; ++v3) a.v[v3] = exp2(a.v[v3]);
    [unroll] for (uint v4 = 0; v4 < 8; ++v4) a.v[v4] = 1.0f + a.v[v4];
    [unroll] for (uint v5 = 0; v5 < 8; ++v5) a.v[v5] = rcp(a.v[v5]);
    [unroll] for (uint v6 = 0; v6 < 8; ++v6) g.v[v6] = h.v[v6] * a.v[v6];
    return wg_pack_q8_nc_ags(g);
}
#else
uint2 convert8(v8f h) { return wg_gelu_q8_lut(h); }
#endif

// run containing u: the largest j with pos[j] <= u (pos[0] = 0)
uint run_of(uint u)
{
    uint lo = 0u, hi = a_n;
    [loop] while (hi - lo > 1u) {
        uint mid = (lo + hi) >> 1;
        bool le = wg_ld32(a_pos, 4u * mid) <= u;
        lo = le ? mid : lo;
        hi = le ? hi : mid;
    }
    return lo;
}

WG_KERNEL(gelu_lut_verify)
{
    WG_SETUP();
    const uint u0 = a_base + 8u * (wg_block_id() * 256u + wg_lane_id());
    v8f h;
    [unroll] for (uint i = 0; i < 8; ++i) h.v[i] = asfloat(u0 + i);
    const uint2 p = convert8(h);
    uint j = run_of(u0);
    [unroll] for (uint k = 0; k < 8; ++k) {
        const uint u = u0 + k;
        [loop] while (j + 1u < a_n && wg_ld32(a_pos, 4u * (j + 1u)) <= u) ++j;
        const uint b = ((k < 4 ? p.x : p.y) >> (8u * (k & 3u))) & 0xffu, e = wg_ld32(a_byt, 4u * j);
        if (b != e) {
            const uint n = wg_atomic_add(a_res, 0u, 1u);
            if ((u & 0x7f800000u) == 0u && (u & 0x7fffffu) != 0u) wg_atomic_add(a_res, 4u, 1u);
            if ((u & 0x7fffffffu) > 0x7f800000u) wg_atomic_add(a_res, 8u, 1u);
            if (n < 30u) {
                wg_st32_gc(a_res, 16u + 8u * n, u);
                wg_st32_gc(a_res, 20u + 8u * n, b | (e << 8));
            }
        }
    }
}
