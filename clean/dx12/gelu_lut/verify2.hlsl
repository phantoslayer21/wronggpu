// SPDX-License-Identifier: MIT
// Exhaustive check of the v2 lookup (hlsl/wg_gelu_lds.hlsli: the 2 KB groupshared table) against HIP's run-length form
// (gelu_lut/host.cpp "dx12" mode, driven by run2.py). Every lane takes 8 consecutive FP32 bit patterns u0 .. u0 + 7,
// converts them as the MLP kernels do (wg_gelu_q8_lds: 8 values -> 2 dwords of FP8 bytes) and compares each byte with
// HIP's (binary search in pos[]).
//   default:           wg_gelu_q8_lds, the table filled into groupshared memory from wg_gl_table as the kernels do
//   GELU_VERIFY_PAIR:  wg_gelu_q8_lds2 (the 16-value form of the Swin / dec1 kernels) on the 8 values and on the same 8
//                      values in reverse order; both results are checked
//   GELU_VERIFY_ORIG:  the transcendental path of the base ports (bneck.hlsl gelu_q8), as a check of the harness
// res: [0] differences, [1] of them on FP32-denormal inputs, [2] on NaN inputs, [4 + 2k], [5 + 2k] examples (k < 30).
#include "wg_platform.hlsli"

groupshared uint64_t gl_tab[256];
#define WG_GL_GET(off) uint2((uint)gl_tab[(off) >> 3], (uint)(gl_tab[(off) >> 3] >> 32))
#include "wg_gelu_lds.hlsli"

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
#elif GELU_VERIFY_PAIR
static uint2 g_rev;   // the reversed tile's bytes
uint2 convert8(v8f h)
{
    v8f r;
    [unroll] for (uint i = 0; i < 8; ++i) r.v[i] = h.v[7 - i];
    uint2 q0, q1;
    wg_gelu_q8_lds2(h, r, q0, q1);
    g_rev = q1;
    return q0;
}
#else
uint2 convert8(v8f h) { return wg_gelu_q8_lds(h); }
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

WG_KERNEL(gelu_lds_verify)
{
    WG_SETUP();
    const uint tid = wg_lane_id();
    const uint2 te = wg_gl_entry(tid);
    gl_tab[tid] = (uint64_t)te.x | ((uint64_t)te.y << 32);
    GroupMemoryBarrierWithGroupSync();
    const uint u0 = a_base + 8u * (wg_block_id() * 256u + tid);
    v8f h;
    [unroll] for (uint i = 0; i < 8; ++i) h.v[i] = asfloat(u0 + i);
    const uint2 p = convert8(h);
    uint j = run_of(u0);
    [unroll] for (uint k = 0; k < 8; ++k) {
        const uint u = u0 + k;
        [loop] while (j + 1u < a_n && wg_ld32(a_pos, 4u * (j + 1u)) <= u) ++j;
        uint b = ((k < 4 ? p.x : p.y) >> (8u * (k & 3u))) & 0xffu;
        const uint e = wg_ld32(a_byt, 4u * j);
#if GELU_VERIFY_PAIR
        const uint k2 = 7u - k, b2 = ((k2 < 4 ? g_rev.x : g_rev.y) >> (8u * (k2 & 3u))) & 0xffu;
        b = b2 != e ? b2 : b;   // a difference of either tile is reported
#endif
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
