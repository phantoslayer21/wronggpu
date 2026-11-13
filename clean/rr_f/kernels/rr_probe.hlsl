// SPDX-License-Identifier: MIT
// Spec 51 kernel probes (DESIGN-51.md section 9), D3D12, driver channel. One wave per 32 lanes, data per lane.
//   rr_p1_wmma : P1, accumulation of v_wmma_f32_16x16x16_fp8_fp8: D1 = A1 B1 + C, D2 = A2 B2 + D1 (f32 results),
//                and the 51.1 step as the kernels do it: f16 of D2 (v_cvt_f16_f32 through the driver conversion).
//   rr_p4_q8   : P4, the kernels' Q8 of f16 values (rr_q8 of rr_common.hlsli) and the f16 helpers, every f16 input.
#include "wg_platform.hlsli"
WG_ARGS_BEGIN
    uint pIn, pOut, n;
WG_ARGS_END
#include "rr_common.hlsli"

// P1. Input per lane: A1 (8 B), B1 (8 B), A2 (8 B), B2 (8 B), C (8 f32) = 64 B. Output per lane: D1 (8 f32), D2 (8 f32),
// f16 of D2 (4 dwords) = 80 B.
WG_KERNEL(rr_p1_wmma)
{
    WG_SETUP();
    const uint i = wg_block_id() * WG_BLOCK + wg_lane_id();
    const uint ib = pIn + i * 64u;
    const uint4 ab = wg_ld128(ib, 0u), ab2 = wg_ld128(ib, 16u);
    v8f c = wg_load8f(ib, 32u);
    v8f d1 = wg_wmma_fp8(ab.xy, ab.zw, c);
    v8f d2 = wg_wmma_fp8(ab2.xy, ab2.zw, d1);
    const uint4 h = wg_to_f16_ags(d2);
    const uint ob = pOut + i * 80u;
    [unroll] for (uint k = 0; k < 8; ++k) {
        wg_st32(ob, 4u * k, asuint(d1.v[k]));
        wg_st32(ob, 32u + 4u * k, asuint(d2.v[k]));
    }
    wg_st128(ob, 64u, h);
}

// P2-P4. Input: one f16 pattern per lane (low half of a dword, n lanes), then the five window-constant dwords
// (rr_wc). Output per lane, 4 dwords: the kernels' Q8 code (rr_q8), the attention weight (rr_attn_w2), the
// activation (rr_act2) and r = R(1 / Z) as the kernels compute it (f16 bits).
WG_KERNEL(rr_p4_q8)
{
    WG_SETUP();
    const uint i = wg_block_id() * WG_BLOCK + wg_lane_id();
    const uint hb = wg_ld32(pIn, i * 4u) & 0xffffu;
    const RrWc wc = rr_wc(wg_ld32(pIn, n * 4u), wg_ld32(pIn, n * 4u + 4u), wg_ld32(pIn, n * 4u + 8u),
                          wg_ld32(pIn, n * 4u + 12u), wg_ld32(pIn, n * 4u + 16u));
    v8f x;
    [unroll] for (uint k = 0; k < 8; ++k) x.v[k] = f16tof32(hb);
    const uint2 q = rr_q8(x);
    const uint w = rr_attn_w2(hb | (hb << 16), wc);
    const uint a = rr_act2(hb | (hb << 16), wc);
    const uint r = wg_f2h(rcp(f16tof32(hb)));
    wg_st128(pOut, i * 16u, uint4(q.x & 0xffu, w, a, r));
}

// P1b. Accumulation model of the fp8 and the f16 WMMA: input per lane A (8 B fp8), B (8 B fp8), Ah (16 B, the same
// values as f16), Bh (16 B), C (8 f32) = 80 B; output per lane D_fp8 (8 f32), D_f16 (8 f32) = 64 B.
WG_KERNEL(rr_p1b_wmma)
{
    WG_SETUP();
    const uint i = wg_block_id() * WG_BLOCK + wg_lane_id();
    const uint ib = pIn + i * 80u;
    const uint4 ab = wg_ld128(ib, 0u), ah = wg_ld128(ib, 16u), bh = wg_ld128(ib, 32u);
    const v8f c = wg_load8f(ib, 48u);
    const v8f d8 = wg_wmma_fp8(ab.xy, ab.zw, c);
    const v8f d16 = wg_wmma_f16(ah, bh, c);
    const uint ob = pOut + i * 64u;
    [unroll] for (uint k = 0; k < 8; ++k) {
        wg_st32(ob, 4u * k, asuint(d8.v[k]));
        wg_st32(ob, 32u + 4u * k, asuint(d16.v[k]));
    }
}
