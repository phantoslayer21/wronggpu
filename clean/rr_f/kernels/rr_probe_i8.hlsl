// SPDX-License-Identifier: MIT
// Probes of the integer step "I" (rr_step_i.hlsli; gpu/probe_i8.py):
//   rr_pi_wmma : H1', v_wmma_i32_16x16x16_iu8 (channel format 0xC) returns A B + C exactly in int32 (A, B signed int8).
//   rr_pi_fin  : the one rounding R(c + N 2^G) of ri_fin (both regimes: |N| < 2^24 and the TwoSum path).
//   rr_pi_step : whole steps against the exact 51.1 step: RR_PI_MODE 0 ri_step_aa (both operands activations, digits
//                and bases computed in the kernel), 1 ri_step_w (A = host digit planes + table entry, B activations),
//                2 ri_step_t (A activations, B = host planes + table).
#include "wg_platform.hlsli"
WG_ARGS_BEGIN
    uint pIn, pOut, n;
WG_ARGS_END
#include "rr_common.hlsli"
static uint rr_hf;
#include "rr_step_i.hlsli"

// H1'. Per lane in: A (8 B), B (8 B), C (8 int32). Out: D (8 int32).
WG_KERNEL(rr_pi_wmma)
{
    WG_SETUP();
    const uint i = wg_block_id() * WG_BLOCK + wg_lane_id();
    const uint ib = pIn + i * 48u;
    const uint4 ab = wg_ld128(ib, 0u);
    RrIA c;
    const uint4 c0 = wg_ld128(ib, 16u), c1 = wg_ld128(ib, 32u);
    c.v[0] = c0.x; c.v[1] = c0.y; c.v[2] = c0.z; c.v[3] = c0.w; c.v[4] = c1.x; c.v[5] = c1.y; c.v[6] = c1.z; c.v[7] = c1.w;
    const RrIA d = ri_mma(ab.xy, ab.zw, c);
    wg_st128(pOut, i * 32u, uint4(d.v[0], d.v[1], d.v[2], d.v[3]));
    wg_st128(pOut, i * 32u + 16u, uint4(d.v[4], d.v[5], d.v[6], d.v[7]));
}

// ri_fin. Per lane in: N (8 int32), c (8 f32), G (int, wave-uniform by the host). Out: f16 (4 dwords), flag.
WG_KERNEL(rr_pi_fin)
{
    WG_SETUP();
    rr_yflag = 0u;
    const uint i = wg_block_id() * WG_BLOCK + wg_lane_id();
    const uint ib = pIn + i * 80u;
    RrIA nn;
    const uint4 n0 = wg_ld128(ib, 0u), n1 = wg_ld128(ib, 16u);
    nn.v[0] = n0.x; nn.v[1] = n0.y; nn.v[2] = n0.z; nn.v[3] = n0.w; nn.v[4] = n1.x; nn.v[5] = n1.y; nn.v[6] = n1.z; nn.v[7] = n1.w;
    const v8f c = wg_load8f(ib, 32u);
    const int G = (int)WaveReadLaneFirst(wg_ld32(ib, 64u));
    const uint4 h = ri_fin(nn, c, G, true);
    wg_st128(pOut, i * 20u, h);
    wg_st32(pOut, i * 20u + 16u, rr_yflag);
}

#ifndef RR_PI_MODE
#define RR_PI_MODE 0
#endif
// Steps. Per lane in (96 B): A K-block 0 / 1 (16 B), B K-block 0 / 1 (16 B), C (32 B), host planes for A or B
// (K-block 0 plane 0 / 1, K-block 1 plane 0 / 1: 32 B; table entry in the first 8 B of the next 16). Out: f16, flag.
WG_KERNEL(rr_pi_step)
{
    WG_SETUP();
    rr_yflag = 0u;
    rr_hf = WaveGetLaneIndex() >> 4;
    const uint i = wg_block_id() * WG_BLOCK + wg_lane_id();
    const uint ib = pIn + i * 112u;
    const uint4 aa = wg_ld128(ib, 0u), bb = wg_ld128(ib, 16u);
    const v8f c = wg_load8f(ib, 32u);
    const uint4 h0 = wg_ld128(ib, 64u), h1 = wg_ld128(ib, 80u);
    const uint2 t = uint2(WaveReadLaneFirst(wg_ld32(ib, 96u)), WaveReadLaneFirst(wg_ld32(ib, 100u)));
    uint4 h;
#if RR_PI_MODE == 3
    {   // debug: the integer sums N and B's digit planes (K-block 0) instead of the step
        const RrAB a = ri_act(aa.xy, aa.zw);
        const RrAB b = ri_act(bb.xy, bb.zw);
        const RrIA n = ri_mm_aa(a.d0, b.d0, a.d1, b.d1, a.d3 || b.d3);
        wg_st128(pOut, i * 64u, uint4(n.v[0], n.v[1], n.v[2], n.v[3]));
        wg_st128(pOut, i * 64u + 16u, uint4(n.v[4], n.v[5], n.v[6], n.v[7]));
        wg_st128(pOut, i * 64u + 32u, uint4(b.d0.p0, b.d0.p1));
        wg_st128(pOut, i * 64u + 48u, uint4(b.d0.p2, (uint)b.beta, a.beta));
        return;
    }
#elif RR_PI_MODE == 0
    const RrAB a = ri_act(aa.xy, aa.zw);
    const RrAB b = ri_act(bb.xy, bb.zw);
    h = ri_step_aa(a, b, c);
#elif RR_PI_MODE == 1
    const RrAB b = ri_act(bb.xy, bb.zw);
    h = ri_step_w(h0.xy, h0.zw, h1.xy, h1.zw, b, c, t);
#else
    const RrAB a = ri_act(aa.xy, aa.zw);
    h = ri_step_t(a, h0.xy, h0.zw, h1.xy, h1.zw, c, t);
#endif
    wg_st128(pOut, i * 20u, h);
    wg_st32(pOut, i * 20u + 16u, rr_yflag);
}
