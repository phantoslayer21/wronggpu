// SPDX-License-Identifier: MIT
// Spec 52.5 front-end MLP on RDNA 4 / D3D12: x0 = Q8(MM(A_1; W2_1; MM(A_0; W2_0; b2))), A_br = Q8(act(MM(f; W1_br; b1_br))),
// for every tensor token. One workgroup = 64 consecutive tokens of the row-major [Ht][Wt] tensor, 4 waves of 16 tokens;
// lane l = 16 hf + n: token n of the wave's tile, half hf of every 16-wide fragment (layouts: gpu/layout.py).
// The 51.1 matrix step is the one of rr_block.hlsl (RR_STEP 0: variant E, 1: variant X, 2: exact binary64; 3: variant X
// with a fast path for exact sums, the same results as 1).
// Weight fragments (gpu/input_stage.py): W1T [br][2 out tiles][4 K slices], W2T [br][2 out tiles][2 K slices] (each
// 256 bytes), biases f32 b1 [br][32], b2 [32].
#include "wg_platform.hlsli"
WG_ARGS_BEGIN
    uint pCodes, pW1, pW2, pB1, pB2, pX0, ntok;
    uint wc0, wc1, wc2, wc3, wc4;
    uint pDiag;                 // RR_STEP 3: nonzero = count the wave steps that took the rr_exact5 branch (one u32)
WG_ARGS_END
#include "rr_common.hlsli"
#include "rr_step.hlsli"

static uint rr_lane;
static uint rr_hf;
// RR_MLP_B32 1 (clean/rr-f-s9-kernels; NOT exact to variant E: network arithmetic relaxed as the body's RR_B32): the
// two matrix steps of each layer carry f32 (no f16 rounding), the activation runs in f32 (same formula), and Q8 takes
// the f32 values through the fast conversion with the wave-uniform slow path of rr_fast.hlsli (rr_q8's bits there).
// Needs RR_STEP 0.
#ifndef RR_MLP_B32
#define RR_MLP_B32 0
#endif
#if RR_MLP_B32
#if RR_STEP != 0
#error RR_MLP_B32 needs RR_STEP 0
#endif
#include "rr_fast.hlsli"
#endif
uint2 wfr(uint base, uint f) { return wg_ld64(base, f * 256u + rr_lane * 8u); }
v8f bias_tile(uint p, uint tile) { return wg_load8f(p, (16u * tile + 8u * rr_hf) * 4u); }

WG_KERNEL(rr_input_mlp)
{
    WG_SETUP();
    rr_lane = WaveGetLaneIndex();
    rr_hf = rr_lane >> 4;
    const uint wv = wg_lane_id() >> 5;
    const uint n = rr_lane & 15u;
    const uint tok = (wg_block_id() * 4u + wv) * 16u + n;
    const RrWc wc = rr_wc(wc0, wc1, wc2, wc3, wc4);
    const uint tl = min(tok, ntok - 1u);                       // wave-uniform control flow for the WMMAs
    uint2 f[4];
    [unroll] for (uint ks = 0; ks < 4u; ++ks) f[ks] = wg_ld64(pCodes, tl * 64u + 16u * ks + 8u * rr_hf);
#if RR_MLP_B32
    rr_sat = false;
    rr_nan = false;
    const float actcl = (float)wc.actcl.x, actcln = (float)wc.actcln.x, c1 = (float)wc.c1.x, c2 = (float)wc.c2.x,
                c3 = (float)wc.c3.x;
    v8f ah[2][2];
    [unroll] for (uint br = 0; br < 2u; ++br) {
        [unroll] for (uint ft = 0; ft < 2u; ++ft) {
            const uint w = (br * 2u + ft) * 4u;
            v8f acc = bias_tile(pB1 + br * 128u, ft);
            acc = step2_raw(wfr(pW1, w + 0u), f[0], wfr(pW1, w + 1u), f[1], acc);
            acc = step2_raw(wfr(pW1, w + 2u), f[2], wfr(pW1, w + 3u), f[3], acc);
            [unroll] for (uint i = 0; i < 8; ++i) {
                const float x = acc.v[i];
                const float k = min(max(x, actcln), actcl);
                const float u = mad(-abs(k), c1, c2);
                const float v = mad(k, u, c3);
                ah[br][ft].v[i] = x * v;
            }
        }
    }
    uint2 a[2][2];
    {
        uint bad = 0u;
        Q8_TRY a[0][0] = q8f(ah[0][0], bad); a[0][1] = q8f(ah[0][1], bad); a[1][0] = q8f(ah[1][0], bad);
               a[1][1] = q8f(ah[1][1], bad);
        Q8_ELSE a[0][0] = q8s(ah[0][0]); a[0][1] = q8s(ah[0][1]); a[1][0] = q8s(ah[1][0]); a[1][1] = q8s(ah[1][1]);
        Q8_END
    }
    v8f o[2];
    [unroll] for (uint ot = 0; ot < 2u; ++ot) {
        v8f z = bias_tile(pB2, ot);
        z = step2_raw(wfr(pW2, (0u * 2u + ot) * 2u), a[0][0], wfr(pW2, (0u * 2u + ot) * 2u + 1u), a[0][1], z);
        o[ot] = step2_raw(wfr(pW2, (1u * 2u + ot) * 2u), a[1][0], wfr(pW2, (1u * 2u + ot) * 2u + 1u), a[1][1], z);
    }
    uint2 x0[2];
    {
        uint bad = 0u;
        Q8_TRY x0[0] = q8f(o[0], bad); x0[1] = q8f(o[1], bad);
        Q8_ELSE x0[0] = q8s(o[0]); x0[1] = q8s(o[1]);
        Q8_END
    }
    if (tok < ntok) {
        wg_st64(pX0, tok * 32u + 8u * rr_hf, x0[0]);
        wg_st64(pX0, tok * 32u + 16u + 8u * rr_hf, x0[1]);
    }
#else
    uint2 a[2][2];                                               // A_br as B fragments (K slice = 16 hidden units)
    [unroll] for (uint br = 0; br < 2u; ++br) {
        [unroll] for (uint ft = 0; ft < 2u; ++ft) {
            const uint w = (br * 2u + ft) * 4u;
            v8f acc = bias_tile(pB1 + br * 128u, ft);
            acc = step2(wfr(pW1, w + 0u), f[0], wfr(pW1, w + 1u), f[1], acc);
            const uint4 h = step2h(wfr(pW1, w + 2u), f[2], wfr(pW1, w + 3u), f[3], acc);
            a[br][ft] = rr_q8(rr_unh(uint4(rr_act2(h.x, wc), rr_act2(h.y, wc), rr_act2(h.z, wc), rr_act2(h.w, wc))));
        }
    }
    [unroll] for (uint ot = 0; ot < 2u; ++ot) {
        v8f z = bias_tile(pB2, ot);
        z = step2(wfr(pW2, (0u * 2u + ot) * 2u), a[0][0], wfr(pW2, (0u * 2u + ot) * 2u + 1u), a[0][1], z);
        const v8f o = step2(wfr(pW2, (1u * 2u + ot) * 2u), a[1][0], wfr(pW2, (1u * 2u + ot) * 2u + 1u), a[1][1], z);
        const uint2 x0 = rr_q8(o);
        if (tok < ntok) wg_st64(pX0, tok * 32u + 16u * ot + 8u * rr_hf, x0);
    }
#endif
#if RR_STEP == 3
    if (pDiag != 0u && WaveIsFirstLane() && rr_x_fallbacks != 0u) wg_atomic_add(pDiag, 0u, rr_x_fallbacks);
#endif
}
