// SPDX-License-Identifier: MIT
// Spec 51 window block (51.3) with its stage fronts (51.5) and tails (51.4) for the small, latency-bound levels: the
// window's work spread over G x 4 waves (one workgroup of 128 G lanes per window) instead of 4. Same arguments, buffer
// layouts, rounding points, step (rr_step.hlsli, RR_STEP) and Q8 / attention / activation forms as rr_block_fast.hlsl,
// and the same output bits as rr_block.hlsl for every input: every value is computed by the same operations in the
// same order; only which wave computes it changes (structure only).
//
// Wave w = 4 g + t: token tile t (tokens 16 t .. 16 t + 15, as wave t of rr_block.hlsl), group g (0 .. G - 1).
//   * heads in rounds of G: group g computes q / v, scores, P and O of head r G + g (its own q / v exchange buffer);
//     O goes to LDS, then every wave advances the Dp chain over the round's G heads in head order for the channel
//     tiles it owns (ct = g + G i; a group past the last tile repeats the last tile and stores the same bits).
//   * first residual of the owned tiles; Q8(y1) to LDS, read back by every wave (all channels of its token tile).
//   * MLP in rounds of G chunks: group g computes Du / Au of chunk r G + g (Au to a double-buffered LDS slot), then
//     every wave advances the Dd chain of its owned tiles over the round's chunks in chunk order.
//   * y2 of the owned tiles to memory and LDS; merge tiles over all waves; the hidden rule in every group (identical
//     values, identical stores).
// The Dp / Dd chains start from the packed f16 bias (the base form: the bias values are f16 values, so the start value
// is the same exact number as the fast kernel's f32 bias).
//
// Defines as rr_block_fast.hlsl (RR_C, RR_H, RR_KIND 0 / 2 / 3, RR_CN, RR_HMERGE, RR_KC, RR_STEP, RR_Z; no adapter)
// plus RR_G (groups; RR_H and RR_C / 8 divisible by RR_G; WG_BLOCK = 128 RR_G). Weight fragments from memory.
#include "wg_platform.hlsli"

WG_ARGS_BEGIN
    uint pX, pCin, pHold, pHc;
    uint pWqv, pTb, pWo, pBo, pMlp, pB1, pB2, pWm, pBm, pWa, pBa, pWup, pBup, pWh, pBh;
    uint pSig, pZ, pY2, pMerged, pHnew, pHmerged;
    int Nx, Ny, sx, sy, nwx;
    uint c_ra_m, c_f_g;
    uint wc0, wc1, wc2, wc3, wc4;
#if RR_DFLOW
    uint dfCnt, dfSelf, dfQ0, dfQ1, dfQ2;   // clean/rr-f-s12-dflow (rr_dflow.hlsli)
#endif
WG_ARGS_END

#include "rr_common.hlsli"
#include "rr_step.hlsli"

#ifndef RR_KIND
#define RR_KIND 0
#endif
#ifndef RR_CN
#define RR_CN 0
#endif
#ifndef RR_HMERGE
#define RR_HMERGE 0
#endif
#ifndef RR_KC
#define RR_KC 0
#endif
#ifndef RR_Z
#define RR_Z 0
#endif
#ifndef RR_G
#define RR_G 4
#endif
#ifndef RR_MLK_CMAX
#define RR_MLK_CMAX 96
#endif
#ifndef RR_PRE                 // operand loads issued ahead of the matrix steps (bits): 1 heads, 2 MLP, 4 merge, 8 hidden
#if defined(RR_B32) && RR_G == 2 && RR_C == 128
#if RR_B32
// round 14 (B2): level 3 of the relaxed body (E3, D3-F, D3-R on wide2) without the head-operand prefetch: 221-229 ->
// 165-192 VGPRs (6 -> 8-9 waves); structure only (the same bits), measured -0.029 / -0.021 ms per frame (1080p / 1440p Q)
#define RR_PRE 14
#endif
#endif
#ifndef RR_PRE
#define RR_PRE 15
#endif
#endif
#ifdef RR_ADAPT
#if RR_ADAPT
#error "rr_block_wide.hlsl: no E1 adapter"
#endif
#endif

#define RC RR_C
#define RH RR_H
#define G RR_G
#define KS (RC / 16)
#define NS (RC / 32)
#define HID (4 * RC)
#define NJ (HID / 32)
#define HIDDEN (RR_KIND == 0 || RR_KIND == 3)
#define UNIT (64 * RC)
#define OWN ((KS + G - 1) / G)     // channel tiles per wave in the chains
#define HR (RH / G)                // head rounds
#ifndef RR_MLK                     // round 13: MLP chunks per group per round (2: half the rounds, one barrier per round)
#if (RR_PRE & 2) && RR_G == 2 && ((4 * RR_C / 32) % 4) == 0 && RR_C <= RR_MLK_CMAX
#define RR_MLK 2
#else
#define RR_MLK 1
#endif
#endif
#define MR (NJ / (G * RR_MLK))     // MLP rounds
#if (RH % G) != 0 || (NJ % G) != 0 || WG_BLOCK != 128 * G
#error "rr_block_wide.hlsl: RR_H and RR_C / 8 must be multiples of RR_G, WG_BLOCK = 128 RR_G"
#endif

uint2 rr_q2(uint64_t q) { return uint2((uint)q, (uint)(q >> 32)); }
uint64_t rr_2q(uint2 v) { return (uint64_t)v.x | ((uint64_t)v.y << 32); }
#ifndef RR_MOFF                // round 13: merge LDS indices without the zi / RC, zi % RC divisions (same integers)
#define RR_MOFF 1
#endif
#if (RC % 16) != 0
#error "rr_block_wide.hlsl: RR_C must be a multiple of 16"
#endif
uint mrg_k(uint z0) { const uint cn = z0 / RC; return (8u * (cn >> 1) + (cn & 1u)) * (RC / 8u) + (z0 % RC) / 8u; }

// ---- LDS (uint64 entries) ---------------------------------------------------------------------------------------------
// heads:  QV [g][512] (q [tile][ft][lane], v^T at +256) | OB [g][tile][ft][lane]
// MLP:    Y [token][C] bytes (y1q; later y2) | AU [buf][g][tile][f][lane]
// front:  UP [coarse r][4C] bytes (fusion halves)
#define QV_OFF 0u
#define OB_OFF (512u * G)
#define Y_OFF 0u
#define Y_N (8u * RC)
#define AU_OFF Y_N
#define LDS_HEAD (768u * G)
#define LDS_MLP (Y_N + 512u * G * RR_MLK)
#define LDS_N (LDS_HEAD > LDS_MLP ? LDS_HEAD : LDS_MLP)
groupshared uint64_t rr_l[LDS_N];

static uint rr_lane;
static uint rr_hf;

uint2 wfrag(uint base, uint f) { return wg_ld64(base, f * 256u + rr_lane * 8u); }
v8f zero8() { return wg_splat(0.0f); }
v8f bias_tile(uint p, uint tile) { return wg_load8f(p, (16u * tile + 8u * rr_hf) * 4u); }
uint2 qv_frag(uint h, uint f) { return wfrag(pWqv + h * UNIT, f); }
uint2 wo_frag(uint h, uint f) { return wfrag(pWo + h * (UNIT / 2), f); }
uint2 mlp_frag(uint j, uint f) { return wfrag(pMlp + j * UNIT, f); }

#ifndef RR_B32G4                    // clean/rr-f-s10-body: RR_B32G4 1 keeps RR_B32 at G = 4 (D4-F / D4-R, see SPECIAL)
#define RR_B32G4 0
#endif
#if defined(RR_B32) && RR_G == 4 && !RR_B32G4   // RR_B32 at G = 4 (E4, E5) spills (192 VGPRs, 512-lane groups): f16 forms there
#undef RR_B32
#define RR_B32 0
#endif
#include "rr_fast.hlsli"
#ifndef RR_DFLOW
#define RR_DFLOW 0
#endif
#if RR_DFLOW
#include "rr_dflow.hlsli"
#else
#define RR_PLD64(b, o) wg_ld64(b, o)
#define RR_PST64(b, o, v) wg_st64(b, o, v)
#endif

WG_KERNEL(rr_block)
{
    WG_SETUP();
    rr_lane = WaveGetLaneIndex();
    rr_hf = rr_lane >> 4;
    rr_sat = false;
    rr_nan = false;
    DBG_INIT
    const uint w = WaveReadLaneFirst(wg_lane_id() >> 5);
    const uint wv = w & 3u;                                 // token tile
    const uint g = w >> 2;                                  // group
    const uint n = rr_lane & 15u;
    const uint wid = wg_block_id();
    const int wi = (int)(wid % (uint)nwx), wj = (int)(wid / (uint)nwx);
    const int lx = (int)(n & 7u), ly = (int)(2u * wv + (n >> 3));
    const int X = 8 * wi - sx + lx, Y = 8 * wj - sy + ly;
    const bool r1 = min(Nx, Ny) >= 10;
    const int Xr = rr_refl(X, Nx, r1), Yr = rr_refl(Y, Ny, r1);
    const bool real = X >= 0 && X < Nx && Y >= 0 && Y < Ny;
    const uint tok = (uint)(Yr * Nx + Xr);
    const uint t = 16u * wv + n;                            // token of the window (LDS rows)
    const RrWc wc = rr_wc(wc0, wc1, wc2, wc3, wc4);
    const half2 ra2 = rr_splat_lo(c_ra_m), m2 = rr_splat_hi(c_ra_m);
    const half2 f2 = rr_splat_lo(c_f_g), g2 = rr_splat_hi(c_f_g);
#if RR_B32
    const RrWx wx = rr_wx(wc);
    const float ra2x = (float)ra2.x, m2x = (float)m2.x, f2x = (float)f2.x;
#define BVK(h) h##x
#else
#define BVK(h) h
#endif
    const int Mx = Nx >> 1, My = Ny >> 1;
    const int cx0 = (8 * wi - sx) >> 1, cy0 = (8 * wj - sy) >> 1;
    uint cto[OWN];                                          // owned channel tiles
    [unroll] for (uint i0 = 0; i0 < OWN; ++i0) cto[i0] = min(g + G * i0, KS - 1u);
#ifndef RR_WPF                  // round 14 (B21 wait-prefetch): the first head round's q / v weight fragments (and with
#if defined(RR_VARL) && RR_VARL // RR_WPF 2 its score biases) loaded before the dataflow wait; weights are never produced
#define RR_WPF 1                // by a body stage, so the same bytes arrive earlier (structure only). Default in the
#else                           // relaxed body (accurate / fast): -0.006 / -0.014 ms alone, -0.022 / -0.022 ms with D4-F
#define RR_WPF 0                // on w4p6b (1080p / 1440p Q); the exact body keeps 0 (not timed there)
#endif
#endif
#if RR_DFLOW && RR_WPF && (RR_PRE & 1) && RR_KIND != 2
#define RR_WPF_ON 1
    uint2 wq_pf[4 * KS];
    [unroll] for (uint f9 = 0; f9 < 4 * KS; ++f9) wq_pf[f9] = qv_frag(g, f9);
#if RR_WPF >= 2
    v8f sb_pf[4];
    [unroll] for (uint J9 = 0; J9 < 4; ++J9) sb_pf[J9] = wg_load8f(pTb + (g * 16u + wv * 4u) * 1024u + rr_lane * 32u, J9 * 1024u);
#endif
#else
#define RR_WPF_ON 0
#endif
#if RR_DFLOW
    rr_df_wait(w, wj, sy, Ny, My, r1);                      // every producer row this window reads is final
#endif

    // ---- block input x (all channels, every wave) and the residual base of the owned tiles ---------------------------
    uint2 xf[KS];
    rrbv rb[OWN];
#if RR_KIND == 2
    {   // fusion half (51.5): up-projection tiles over all waves (a wave past the last tile repeats it: same bytes)
        const int crx = cx0 + (int)(n & 3u), cry = cy0 + (int)(n >> 2);
        const uint ctok = (uint)(rr_refl(cry, My, r1) * Mx + rr_refl(crx, Mx, r1));
        uint2 cf[RR_KC / 16];
        [unroll] for (uint k = 0; k < RR_KC / 16; ++k) cf[k] = RR_PLD64(pCin, ctok * RR_KC + 16u * k + 8u * rr_hf);
        RR_LOOP for (uint t2 = 0; t2 < (HID / 16 + 4u * G - 1u) / (4u * G) + RR_ZL; ++t2) {
            const uint ot = min(w + 4u * G * t2, HID / 16 - 1u);
            uint2 a[RR_KC / 16];
            [unroll] for (uint k2 = 0; k2 < RR_KC / 16; ++k2) a[k2] = wfrag(pWup, ot * (RR_KC / 16) + k2);
            v8f acc = bias_tile(pBup, ot);
            [unroll] for (uint s = 0; s < RR_KC / 32; ++s)
                acc = step2l(s + 1 == RR_KC / 32, a[2 * s], cf[2 * s], a[2 * s + 1], cf[2 * s + 1], acc);
            uint bad = 0u;
            uint2 uq;
            Q8_TRY uq = q8f(acc, bad);
            Q8_ELSE uq = q8s(acc); Q8_END
            rr_l[(n * HID + 16u * ot + 8u * rr_hf) >> 3] = rr_2q(uq);
        }
        GroupMemoryBarrierWithGroupSync();
        const uint par = wv * 4u + (uint)(lx >> 1);
        const uint ch = 2u * ((uint)Y & 1u) + ((uint)X & 1u);
        rrbv xr[KS];
        uint bad = 0u;
        [unroll] for (uint ct = 0; ct < KS; ++ct) {
            const uint2 sk = RR_PLD64(pX, tok * RC + 16u * ct + 8u * rr_hf);
            const uint2 up = rr_q2(rr_l[(par * HID + ch * RC + 16u * ct + 8u * rr_hf) >> 3]);
            xr[ct] = bv_mad(bv_d8(sk), BVK(f2), bv_d8(up));
        }
        Q8_TRY [unroll] for (uint c0 = 0; c0 < KS; ++c0) xf[c0] = bv_q8f(xr[c0], bad);
        Q8_ELSE [unroll] for (uint c = 0; c < KS; ++c) xf[c] = bv_q8s(xr[c]); Q8_END
        [unroll] for (uint i1 = 0; i1 < OWN; ++i1) {
            const uint ct = cto[i1];
            const uint2 sk = RR_PLD64(pX, tok * RC + 16u * ct + 8u * rr_hf);
            const uint2 up = rr_q2(rr_l[(par * HID + ch * RC + 16u * ct + 8u * rr_hf) >> 3]);
            rb[i1] = bv_mad(bv_d8(sk), BVK(f2), bv_d8(up));
        }
        GroupMemoryBarrierWithGroupSync();                  // UP overlays the head buffers
    }
#else
    [unroll] for (uint ks = 0; ks < KS; ++ks) xf[ks] = RR_PLD64(pX, tok * RC + 16u * ks + 8u * rr_hf);
    [unroll] for (uint i1 = 0; i1 < OWN; ++i1) rb[i1] = bv_d8(RR_PLD64(pX, tok * RC + 16u * cto[i1] + 8u * rr_hf));
#endif

    // ---- heads: steps 1-6, rounds of G heads -------------------------------------------------------------------------
    rrbv dp[OWN];
    [unroll] for (uint i2 = 0; i2 < OWN; ++i2) dp[i2] = bv_of(bias_tile(pBo, cto[i2]));
    const uint qb = QV_OFF + g * 512u;
    [unroll] for (uint r = 0; r < HR; ++r) {
        const uint h = r * G + g;
        const uint tb = pTb + (h * 16u + wv * 4u) * 1024u + rr_lane * 32u;
        v8f aq0 = zero8(), aq1 = zero8(), av0 = zero8(), av1 = zero8();
#if RR_PRE & 1
        // every operand load of the round ahead (the driver keeps loads behind the matrix calls, which are UAV atomics)
        uint2 wq[4 * KS];
#if RR_WPF_ON
        [unroll] for (uint f0 = 0; f0 < 4 * KS; ++f0) wq[f0] = r == 0 ? wq_pf[f0] : qv_frag(h, f0);
#else
        [unroll] for (uint f0 = 0; f0 < 4 * KS; ++f0) wq[f0] = qv_frag(h, f0);
#endif
        [unroll] for (uint s = 0; s < NS; ++s) {
            aq0 = step2l(s + 1 == NS, wq[2 * s], xf[2 * s], wq[2 * s + 1], xf[2 * s + 1], aq0);
            aq1 = step2l(s + 1 == NS, wq[KS + 2 * s], xf[2 * s], wq[KS + 2 * s + 1], xf[2 * s + 1], aq1);
            av0 = step2l(s + 1 == NS, xf[2 * s], wq[2 * KS + 2 * s], xf[2 * s + 1], wq[2 * KS + 2 * s + 1], av0);
            av1 = step2l(s + 1 == NS, xf[2 * s], wq[3 * KS + 2 * s], xf[2 * s + 1], wq[3 * KS + 2 * s + 1], av1);
        }
        // the score biases and the round's Wo fragments (used after the exchanges)
        v8f sb[4];
#if RR_WPF_ON && RR_WPF >= 2
        [unroll] for (uint J5 = 0; J5 < 4; ++J5) { if (r == 0) sb[J5] = sb_pf[J5]; else sb[J5] = wg_load8f(tb, J5 * 1024u); }
#else
        [unroll] for (uint J5 = 0; J5 < 4; ++J5) sb[J5] = wg_load8f(tb, J5 * 1024u);
#endif
        uint2 wo[G][2 * OWN];
        [unroll] for (uint g3 = 0; g3 < G; ++g3) {
            [unroll] for (uint i3 = 0; i3 < OWN; ++i3) {
                wo[g3][2 * i3] = wo_frag(r * G + g3, 2u * cto[i3]);
                wo[g3][2 * i3 + 1] = wo_frag(r * G + g3, 2u * cto[i3] + 1u);
            }
        }
#else
        [unroll] for (uint s = 0; s < NS; ++s) {
            const uint2 q00 = qv_frag(h, 2u * s), q01 = qv_frag(h, 2u * s + 1u);
            const uint2 q10 = qv_frag(h, KS + 2u * s), q11 = qv_frag(h, KS + 2u * s + 1u);
            const uint2 v00 = qv_frag(h, 2u * KS + 2u * s), v01 = qv_frag(h, 2u * KS + 2u * s + 1u);
            const uint2 v10 = qv_frag(h, 3u * KS + 2u * s), v11 = qv_frag(h, 3u * KS + 2u * s + 1u);
            aq0 = step2l(s + 1 == NS, q00, xf[2 * s], q01, xf[2 * s + 1], aq0);
            aq1 = step2l(s + 1 == NS, q10, xf[2 * s], q11, xf[2 * s + 1], aq1);
            av0 = step2l(s + 1 == NS, xf[2 * s], v00, xf[2 * s + 1], v01, av0);
            av1 = step2l(s + 1 == NS, xf[2 * s], v10, xf[2 * s + 1], v11, av1);
        }
#endif
        uint bad = 0u;
        uint2 qf0, qf1, vf0, vf1;
        Q8_TRY qf0 = q8f(aq0, bad); qf1 = q8f(aq1, bad); vf0 = q8f(av0, bad); vf1 = q8f(av1, bad);
        Q8_ELSE qf0 = q8s(aq0); qf1 = q8s(aq1); vf0 = q8s(av0); vf1 = q8s(av1); Q8_END
        rr_l[qb + (wv * 2u) * 32u + rr_lane] = rr_2q(qf0);
        rr_l[qb + (wv * 2u + 1u) * 32u + rr_lane] = rr_2q(qf1);
        rr_l[qb + 256u + (wv * 2u) * 32u + rr_lane] = rr_2q(vf0);
        rr_l[qb + 256u + (wv * 2u + 1u) * 32u + rr_lane] = rr_2q(vf1);
        GroupMemoryBarrierWithGroupSync();

        uint2 kk[8];
        [unroll] for (uint J0 = 0; J0 < 8; ++J0) kk[J0] = rr_q2(rr_l[qb + J0 * 32u + rr_lane]);
        rrbv wt[4];
        [unroll] for (uint J = 0; J < 4; ++J) {
#if RR_PRE & 1
            const rrbv s16 = bv_step(kk[2 * J], qf0, kk[2 * J + 1], qf1, sb[J]);
#else
            const rrbv s16 = bv_step(kk[2 * J], qf0, kk[2 * J + 1], qf1, wg_load8f(tb, J * 1024u));
#endif
            wt[J] = bv_attn(s16, BVW);
        }
#if RR_B32
        // f32 softmax normalisation (as rr_block_fast.hlsl RR_B32)
        v8f zt;
        [unroll] for (uint i3 = 0; i3 < 8; ++i3) zt.v[i3] = (wt[0].v[i3] + wt[1].v[i3]) + (wt[2].v[i3] + wt[3].v[i3]);
        float zs = ((zt.v[0] + zt.v[1]) + (zt.v[2] + zt.v[3])) + ((zt.v[4] + zt.v[5]) + (zt.v[6] + zt.v[7]));
        const float rz = rcp(zs + wg_xor16(zs));
        uint2 pf[4];
        [unroll] for (uint J4 = 0; J4 < 4; ++J4) {
            v8f p;
            [unroll] for (uint i4 = 0; i4 < 8; ++i4) p.v[i4] = wt[J4].v[i4] * rz;
            pf[J4] = rr_q8_small(p);
        }
#else
#if RR_Z == 1
        const uint zb = wg_ld16(pZ, ((wid * RH + h) * 64u + 16u * wv + n) * 2u);
#elif RR_Z == 2
        float hs = 0.0f, ls = 0.0f;
        [unroll] for (uint J2 = 0; J2 < 4; ++J2) {
            const uint ww[4] = {wt[J2].x, wt[J2].y, wt[J2].z, wt[J2].w};
            [unroll] for (uint k = 0; k < 4; ++k) {
                [unroll] for (uint e = 0; e < 2; ++e) {
                    const float wf = f16tof32(ww[k] >> (16u * e));
                    const float hi = floor(wf * 8.0f);
                    hs = hs + hi;
                    ls = ls + mad(hi, -0.125f, wf);
                }
            }
        }
        const uint hi_t = (uint)hs + (uint)wg_xor16(hs);
        const float lo_t = ls + wg_xor16(ls);
        const float za = (float)(hi_t & ~1u) * 0.125f, zbb = lo_t + (float)(hi_t & 1u) * 0.125f;
        const float zr = za + zbb, zv = zr - za, ze = (za - (zr - zv)) + (zbb - zv);
        const uint zur = asuint(zr);
        const uint zb = f2h_hw(asfloat((zur & 0x7fffffffu) >= 0x7f800000u ? zur : wg_round_odd(zr, ze)));
#else
        float zs = 0.0f;
        [unroll] for (uint J3 = 0; J3 < 4; ++J3) {
            zs = dot2add(rr_h2(wt[J3].x), half2(1.0, 1.0), zs);
            zs = dot2add(rr_h2(wt[J3].y), half2(1.0, 1.0), zs);
            zs = dot2add(rr_h2(wt[J3].z), half2(1.0, 1.0), zs);
            zs = dot2add(rr_h2(wt[J3].w), half2(1.0, 1.0), zs);
        }
        const uint zb = f2h_hw(zs + wg_xor16(zs));
#endif
        const uint rbz = f2h_hw(rcp(f16tof32(zb)));
        const half2 r2 = rr_h2(rbz * 0x10001u);
        uint2 pf[4];
        [unroll] for (uint J4 = 0; J4 < 4; ++J4) {
            uint4 p;
            p.x = rr_b2(rr_h2(wt[J4].x) * r2);
            p.y = rr_b2(rr_h2(wt[J4].y) * r2);
            p.z = rr_b2(rr_h2(wt[J4].z) * r2);
            p.w = rr_b2(rr_h2(wt[J4].w) * r2);
            pf[J4] = q8h_small(p);
        }
#endif
        uint2 va[8];
        [unroll] for (uint J1 = 0; J1 < 8; ++J1) va[J1] = rr_q2(rr_l[qb + 256u + J1 * 32u + rr_lane]);
        v8f ov[2];
        [unroll] for (uint ft = 0; ft < 2; ++ft)
            ov[ft] = step2l(true, va[4 + ft], pf[2], va[6 + ft], pf[3], step2(va[ft], pf[0], va[2 + ft], pf[1], zero8()));
        bad = 0u;
        uint2 of[2];
        Q8_TRY of[0] = q8f(ov[0], bad); of[1] = q8f(ov[1], bad);
        Q8_ELSE of[0] = q8s(ov[0]); of[1] = q8s(ov[1]); Q8_END
        rr_l[OB_OFF + (g * 4u + wv) * 64u + rr_lane] = rr_2q(of[0]);
        rr_l[OB_OFF + (g * 4u + wv) * 64u + 32u + rr_lane] = rr_2q(of[1]);
        GroupMemoryBarrierWithGroupSync();

        // Dp steps of the round's heads, in head order, for the owned channel tiles
#if RR_PRE & 1
        uint2 ob[G][2];
        [unroll] for (uint g4 = 0; g4 < G; ++g4) {
            ob[g4][0] = rr_q2(rr_l[OB_OFF + (g4 * 4u + wv) * 64u + rr_lane]);
            ob[g4][1] = rr_q2(rr_l[OB_OFF + (g4 * 4u + wv) * 64u + 32u + rr_lane]);
        }
        [unroll] for (uint gg = 0; gg < G; ++gg) {
            [unroll] for (uint i4 = 0; i4 < OWN; ++i4)
                dp[i4] = bv_step(wo[gg][2 * i4], ob[gg][0], wo[gg][2 * i4 + 1], ob[gg][1], bv_un(dp[i4]));
        }
#else
        [unroll] for (uint gg = 0; gg < G; ++gg) {
            const uint hh = r * G + gg;
            const uint2 o0 = rr_q2(rr_l[OB_OFF + (gg * 4u + wv) * 64u + rr_lane]);
            const uint2 o1 = rr_q2(rr_l[OB_OFF + (gg * 4u + wv) * 64u + 32u + rr_lane]);
            uint2 wa[OWN], wb[OWN];
            [unroll] for (uint i3 = 0; i3 < OWN; ++i3) {
                wa[i3] = wo_frag(hh, 2u * cto[i3]);
                wb[i3] = wo_frag(hh, 2u * cto[i3] + 1u);
            }
            [unroll] for (uint i4 = 0; i4 < OWN; ++i4) dp[i4] = bv_step(wa[i4], o0, wb[i4], o1, bv_un(dp[i4]));
        }
#endif
    }
    GroupMemoryBarrierWithGroupSync();                      // Y overlays the head buffers

    // ---- step 7: first residual of the owned tiles; Q8(y1) of every channel to LDS ----------------------------------
    rrbv y1h[OWN];
    {
        uint2 yq[OWN];
        uint bad = 0u;
        [unroll] for (uint i5 = 0; i5 < OWN; ++i5) {
            y1h[i5] = bv_mad(dp[i5], BVK(ra2), rb[i5]);
        }
        Q8_TRY [unroll] for (uint c0 = 0; c0 < OWN; ++c0) yq[c0] = bv_q8f(y1h[c0], bad);
        Q8_ELSE [unroll] for (uint c = 0; c < OWN; ++c) yq[c] = bv_q8s(y1h[c]); Q8_END
        [unroll] for (uint i6 = 0; i6 < OWN; ++i6) rr_l[Y_OFF + ((t * RC + 16u * cto[i6] + 8u * rr_hf) >> 3)] = rr_2q(yq[i6]);
    }
    GroupMemoryBarrierWithGroupSync();
    uint2 y1q[KS];
    [unroll] for (uint c7 = 0; c7 < KS; ++c7) y1q[c7] = rr_q2(rr_l[Y_OFF + ((t * RC + 16u * c7 + 8u * rr_hf) >> 3)]);

    // ---- step 8: MLP in rounds of G chunks ---------------------------------------------------------------------------
    rrbv dd[OWN];
    [unroll] for (uint i7 = 0; i7 < OWN; ++i7) dd[i7] = bv_of(bias_tile(pB2, cto[i7]));
#if RR_MLK == 2
    // round 13: two chunks per group per round, (2 r2 + k) G + g; the round's 2 G Au tiles in AU [r2 & 1][k][g], one
    // barrier, then the Dd chain over the round's chunks in chunk order (the same steps and operands as RR_MLK 1)
    RR_LOOP for (uint r2 = 0; r2 < MR + RR_ZL; ++r2) {
        const uint abase = AU_OFF + ((r2 & 1u) * 2u * G) * 256u + wv * 64u + rr_lane;
        [unroll] for (uint k = 0; k < 2; ++k) {
            const uint j = (2u * r2 + k) * G + g;
            uint2 w1[2 * KS];
            [unroll] for (uint f1 = 0; f1 < 2 * KS; ++f1) w1[f1] = mlp_frag(j, f1);
            v8f u0 = bias_tile(pB1, 2u * j), u1 = bias_tile(pB1, 2u * j + 1u);
            rrbv h0, h1;
            [unroll] for (uint s = 0; s < NS; ++s) {
                if (s + 1 < NS) {
                    u0 = step2(w1[2 * s], y1q[2 * s], w1[2 * s + 1], y1q[2 * s + 1], u0);
                    u1 = step2(w1[KS + 2 * s], y1q[2 * s], w1[KS + 2 * s + 1], y1q[2 * s + 1], u1);
                } else {
                    h0 = bv_step(w1[2 * s], y1q[2 * s], w1[2 * s + 1], y1q[2 * s + 1], u0);
                    h1 = bv_step(w1[KS + 2 * s], y1q[2 * s], w1[KS + 2 * s + 1], y1q[2 * s + 1], u1);
                }
            }
            const rrbv a0h = bv_act(h0, BVW);
            const rrbv a1h = bv_act(h1, BVW);
            uint bad = 0u;
            uint2 au0, au1;
            Q8_TRY au0 = bv_q8f(a0h, bad); au1 = bv_q8f(a1h, bad);
            Q8_ELSE au0 = bv_q8s(a0h); au1 = bv_q8s(a1h); Q8_END
            rr_l[abase + (k * G + g) * 256u] = rr_2q(au0);
            rr_l[abase + (k * G + g) * 256u + 32u] = rr_2q(au1);
        }
        uint2 w2[G][2 * OWN];
        [unroll] for (uint g5 = 0; g5 < G; ++g5) {
            [unroll] for (uint i8 = 0; i8 < OWN; ++i8) {
                w2[g5][2 * i8] = mlp_frag(2u * r2 * G + g5, 2u * KS + 2u * cto[i8]);
                w2[g5][2 * i8 + 1] = mlp_frag(2u * r2 * G + g5, 2u * KS + 2u * cto[i8] + 1u);
            }
        }
        GroupMemoryBarrierWithGroupSync();
        [unroll] for (uint hh = 0; hh < 2; ++hh) {
            uint2 ab2[G][2];
            [unroll] for (uint g6 = 0; g6 < G; ++g6) {
                ab2[g6][0] = rr_q2(rr_l[abase + (hh * G + g6) * 256u]);
                ab2[g6][1] = rr_q2(rr_l[abase + (hh * G + g6) * 256u + 32u]);
            }
            uint2 w3[G][2 * OWN];
            [unroll] for (uint g7 = 0; g7 < G; ++g7) {
                [unroll] for (uint i8 = 0; i8 < OWN; ++i8) {
                    w3[g7][2 * i8] = hh ? mlp_frag((2u * r2 + 1u) * G + g7, 2u * KS + 2u * cto[i8]) : w2[g7][2 * i8];
                    w3[g7][2 * i8 + 1] = hh ? mlp_frag((2u * r2 + 1u) * G + g7, 2u * KS + 2u * cto[i8] + 1u) : w2[g7][2 * i8 + 1];
                }
            }
            [unroll] for (uint gg2 = 0; gg2 < G; ++gg2) {
                [unroll] for (uint i9 = 0; i9 < OWN; ++i9)
                    dd[i9] = bv_step(w3[gg2][2 * i9], ab2[gg2][0], w3[gg2][2 * i9 + 1], ab2[gg2][1], bv_un(dd[i9]));
            }
        }
    }
#else
    RR_LOOP for (uint r2 = 0; r2 < MR + RR_ZL; ++r2) {
        const uint j = r2 * G + g;
#if RR_PRE & 2
        uint2 w1[2 * KS];
        [unroll] for (uint f1 = 0; f1 < 2 * KS; ++f1) w1[f1] = mlp_frag(j, f1);
#define W1A(s) w1[2 * (s)]
#define W1B(s) w1[2 * (s) + 1]
#define W1C(s) w1[KS + 2 * (s)]
#define W1D(s) w1[KS + 2 * (s) + 1]
#else
#define W1A(s) mlp_frag(j, 2u * (s))
#define W1B(s) mlp_frag(j, 2u * (s) + 1u)
#define W1C(s) mlp_frag(j, KS + 2u * (s))
#define W1D(s) mlp_frag(j, KS + 2u * (s) + 1u)
#endif
        v8f u0 = bias_tile(pB1, 2u * j), u1 = bias_tile(pB1, 2u * j + 1u);
        rrbv h0, h1;
        [unroll] for (uint s = 0; s < NS; ++s) {
            if (s + 1 < NS) {
                u0 = step2(W1A(s), y1q[2 * s], W1B(s), y1q[2 * s + 1], u0);
                u1 = step2(W1C(s), y1q[2 * s], W1D(s), y1q[2 * s + 1], u1);
            } else {
                h0 = bv_step(W1A(s), y1q[2 * s], W1B(s), y1q[2 * s + 1], u0);
                h1 = bv_step(W1C(s), y1q[2 * s], W1D(s), y1q[2 * s + 1], u1);
            }
        }
#undef W1A
#undef W1B
#undef W1C
#undef W1D
#if RR_PRE & 2
        uint2 w2[G][2 * OWN];
        [unroll] for (uint g5 = 0; g5 < G; ++g5) {
            [unroll] for (uint i8 = 0; i8 < OWN; ++i8) {
                w2[g5][2 * i8] = mlp_frag(r2 * G + g5, 2u * KS + 2u * cto[i8]);
                w2[g5][2 * i8 + 1] = mlp_frag(r2 * G + g5, 2u * KS + 2u * cto[i8] + 1u);
            }
        }
#endif
        const rrbv a0h = bv_act(h0, BVW);
        const rrbv a1h = bv_act(h1, BVW);
        uint bad = 0u;
        uint2 au0, au1;
        Q8_TRY au0 = bv_q8f(a0h, bad); au1 = bv_q8f(a1h, bad);
        Q8_ELSE au0 = bv_q8s(a0h); au1 = bv_q8s(a1h); Q8_END
        const uint ab = AU_OFF + ((r2 & 1u) * G) * 256u + wv * 64u + rr_lane;
        rr_l[ab + g * 256u] = rr_2q(au0);
        rr_l[ab + g * 256u + 32u] = rr_2q(au1);
        GroupMemoryBarrierWithGroupSync();
#if RR_PRE & 2
        uint2 ab2[G][2];
        [unroll] for (uint g6 = 0; g6 < G; ++g6) {
            ab2[g6][0] = rr_q2(rr_l[ab + g6 * 256u]);
            ab2[g6][1] = rr_q2(rr_l[ab + g6 * 256u + 32u]);
        }
        [unroll] for (uint gg2 = 0; gg2 < G; ++gg2) {
            [unroll] for (uint i9 = 0; i9 < OWN; ++i9)
                dd[i9] = bv_step(w2[gg2][2 * i9], ab2[gg2][0], w2[gg2][2 * i9 + 1], ab2[gg2][1], bv_un(dd[i9]));
        }
#else
        [unroll] for (uint gg2 = 0; gg2 < G; ++gg2) {
            const uint jj = r2 * G + gg2;
            const uint2 b0 = rr_q2(rr_l[ab + gg2 * 256u]), b1 = rr_q2(rr_l[ab + gg2 * 256u + 32u]);
            uint2 wa[OWN], wb[OWN];
            [unroll] for (uint i8 = 0; i8 < OWN; ++i8) {
                wa[i8] = mlp_frag(jj, 2u * KS + 2u * cto[i8]);
                wb[i8] = mlp_frag(jj, 2u * KS + 2u * cto[i8] + 1u);
            }
            [unroll] for (uint i9 = 0; i9 < OWN; ++i9) dd[i9] = bv_step(wa[i9], b0, wb[i9], b1, bv_un(dd[i9]));
        }
#endif
    }
#endif

    // ---- step 9: second residual, y2 of the owned tiles (memory, LDS) ------------------------------------------------
    {
        rrbv yh[OWN];
        uint2 y2o[OWN];
        uint bad = 0u;
        [unroll] for (uint c3 = 0; c3 < OWN; ++c3) {
            yh[c3] = bv_mad(dd[c3], BVK(m2), y1h[c3]);
        }
        Q8_TRY [unroll] for (uint c0 = 0; c0 < OWN; ++c0) y2o[c0] = bv_q8f(yh[c0], bad);
        Q8_ELSE [unroll] for (uint c = 0; c < OWN; ++c) y2o[c] = bv_q8s(yh[c]); Q8_END
        const uint otok = (uint)(Y * Nx + X);
        [unroll] for (uint c4 = 0; c4 < OWN; ++c4) {
            if (real) RR_PST64(pY2, otok * RC + 16u * cto[c4] + 8u * rr_hf, y2o[c4]);
            rr_l[Y_OFF + ((t * RC + 16u * cto[c4] + 8u * rr_hf) >> 3)] = rr_2q(y2o[c4]);
        }
    }
    GroupMemoryBarrierWithGroupSync();

    // ---- merge (51.4, E2-E4): output tiles over all waves --------------------------------------------------------
#if RR_CN > 0
    {
        const uint rx = n & 3u, ry = n >> 2;
        const int Xc = cx0 + (int)rx, Yc = cy0 + (int)ry;
        const bool cin = Xc >= 0 && Xc < Mx && Yc >= 0 && Yc < My;
#if RR_MOFF
        // round 13: the merge operand row of zi = z0 + 8 rr_hf (z0 a multiple of 16; RC a multiple of 16, so zi / RC =
        // z0 / RC and zi % RC = z0 % RC + 8 rr_hf), tt = 16 ry + 2 rx + 8 (cn >> 1) + (cn & 1): every term of tt RC + cc
        // is a multiple of 8, so the LDS index is a lane base plus a per-z0 constant (no division, the same integers)
        const uint mlb = Y_OFF + (16u * ry + 2u * rx) * (RC / 8u) + rr_hf;
#endif
#if RR_COUNT   // counting build: one wave's iterations unrolled (the first wave's count)
        [unroll] for (uint ot = 0; ot < RR_CN / 16; ot += 4u * G) {
#else
        RR_LOOP for (uint ot = w; ot < RR_CN / 16 + RR_ZL; ot += 4u * G) {
#endif
#if RR_PRE & 4
            // operands in batches of 8 fragments (4 steps), the next batch loaded before the current one's steps
            uint2 am[2][8], bm[2][8];
            [unroll] for (uint f2 = 0; f2 < 8; ++f2) am[0][f2] = wfrag(pWm, ot * (HID / 16) + f2);
            v8f acc = bias_tile(pBm, ot);
            [unroll] for (uint bt = 0; bt < HID / 128; ++bt) {
                const uint cb = bt & 1u;
                [unroll] for (uint f3 = 0; f3 < 8; ++f3) {
#if RR_MOFF
                    bm[cb][f3] = rr_q2(rr_l[mlb + mrg_k(16u * (8u * bt + f3))]);
#else
                    const uint zi = 16u * (8u * bt + f3) + 8u * rr_hf;
                    const uint cn = zi / RC, cc = zi % RC;
                    const uint tt = 8u * (2u * ry + (cn >> 1)) + 2u * rx + (cn & 1u);
                    bm[cb][f3] = rr_q2(rr_l[Y_OFF + ((tt * RC + cc) >> 3)]);
#endif
                }
                if (bt + 1 < HID / 128) {
                    [unroll] for (uint f4 = 0; f4 < 8; ++f4) am[cb ^ 1u][f4] = wfrag(pWm, ot * (HID / 16) + 8u * (bt + 1u) + f4);
                }
                [unroll] for (uint s = 0; s < 4; ++s)
                    acc = step2l(and(bt + 1 == HID / 128, s == 3u), am[cb][2 * s], bm[cb][2 * s], am[cb][2 * s + 1], bm[cb][2 * s + 1], acc);
            }
#else
            v8f acc = bias_tile(pBm, ot);
            RR_LOOP for (uint s = 0; s < HID / 32 + RR_ZL; ++s) {
                uint2 b[2], a[2];
                [unroll] for (uint e = 0; e < 2; ++e) {
#if RR_MOFF
                    b[e] = rr_q2(rr_l[mlb + mrg_k(32u * s + 16u * e)]);
#else
                    const uint zi = 32u * s + 16u * e + 8u * rr_hf;
                    const uint cn = zi / RC, cc = zi % RC;
                    const uint tt = 8u * (2u * ry + (cn >> 1)) + 2u * rx + (cn & 1u);
                    b[e] = rr_q2(rr_l[Y_OFF + ((tt * RC + cc) >> 3)]);
#endif
                    a[e] = wfrag(pWm, ot * (HID / 16) + 2u * s + e);
                }
                acc = step2l(s + 1 == HID / 32 + RR_ZL, a[0], b[0], a[1], b[1], acc);
            }
#endif
            uint bad = 0u;
            uint2 mq;
            Q8_TRY mq = q8f(acc, bad);
            Q8_ELSE mq = q8s(acc); Q8_END
            if (cin && rr_zero() == 0u) RR_PST64(pMerged, (uint)(Yc * Mx + Xc) * RR_CN + 16u * ot + 8u * rr_hf, mq);
        }
    }
#endif

    // ---- hidden state (51.4): one group per token tile (round 13: g == wv % G; every group computed the same values
    // and made the same stores, so the other groups' copies were redundant; the guard is wave-uniform and holds no barrier)
#if HIDDEN
#ifndef RR_HID1
#define RR_HID1 1
#endif
    if (!RR_HID1 || g == wv % G) {
        uint2 y2[KS];
        [unroll] for (uint c8 = 0; c8 < KS; ++c8) y2[c8] = rr_q2(rr_l[Y_OFF + ((t * RC + 16u * c8 + 8u * rr_hf) >> 3)]);
        uint2 wh[KS];
#if RR_PRE & 8
        [unroll] for (uint f4 = 0; f4 < KS; ++f4) wh[f4] = wfrag(pWh, f4);
#endif
        v8f dh = wg_load8f(pBh, 32u * rr_hf);
        const uint2 hold = RR_PLD64(pHold, tok * 8u);
        uint4 dhb;
        [unroll] for (uint s = 0; s < NS; ++s) {
#if !(RR_PRE & 8)
            wh[2 * s] = wfrag(pWh, 2u * s);
            wh[2 * s + 1] = wfrag(pWh, 2u * s + 1u);
#endif
            if (s + 1 < NS) dh = step2(wh[2 * s], y2[2 * s], wh[2 * s + 1], y2[2 * s + 1], dh);
            else dhb = step2h(wh[2 * s], y2[2 * s], wh[2 * s + 1], y2[2 * s + 1], dh);
        }
        const uint s01 = wg_ld16(pSig, (dhb.x & 0xffffu) * 2u) | (wg_ld16(pSig, (dhb.x >> 16) * 2u) << 16);
        const uint s23 = wg_ld16(pSig, (dhb.y & 0xffffu) * 2u) | (wg_ld16(pSig, (dhb.y >> 16) * 2u) << 16);
        uint2 hn;
#if RR_KIND == 0
        const uint G01 = rr_fence_u(rr_b2(rr_h2(s01) * g2)), G23 = rr_fence_u(rr_b2(rr_h2(s23) * g2));
        hn.x = rr_fence_u(rr_b2(rr_h2(hold.x) * rr_h2(G01)));
        hn.y = rr_fence_u(rr_b2(rr_h2(hold.y) * rr_h2(G23))) & 0xffffu;
#else
        const half2 z2 = rr_splat_hi(s23);
        const uint omzb = rr_fence_u(rr_b2(half2(1.0, 1.0) - z2));
        const half2 omz = rr_h2(omzb);
        const int xm = (X - 1) >> 1, ym = (Y - 1) >> 1;
        const int rxm = rr_refl(xm, Mx, r1), rxp = rr_refl(xm + 1, Mx, r1);
        const int rym = rr_refl(ym, My, r1), ryp = rr_refl(ym + 1, My, r1);
        const half2 wxp = (X & 1) ? half2(0.25, 0.25) : half2(0.75, 0.75);
        const half2 wyp = (Y & 1) ? half2(0.25, 0.25) : half2(0.75, 0.75);
        const half2 wxm = (X & 1) ? half2(0.75, 0.75) : half2(0.25, 0.25);
        const half2 wym = (Y & 1) ? half2(0.75, 0.75) : half2(0.25, 0.25);
        const uint2 cA = RR_PLD64(pHc, (uint)(rym * Mx + rxp) * 8u), cB = RR_PLD64(pHc, (uint)(rym * Mx + rxm) * 8u);
        const uint2 cC = RR_PLD64(pHc, (uint)(ryp * Mx + rxm) * 8u), cD = RR_PLD64(pHc, (uint)(ryp * Mx + rxp) * 8u);
        uint2 u;
        [unroll] for (uint e = 0; e < 2; ++e) {
            const uint a = e ? cA.y : cA.x, b = e ? cB.y : cB.x, c = e ? cC.y : cC.x, d = e ? cD.y : cD.x;
            half2 tt = rr_h2(rr_fence_u(rr_b2(rr_h2(rr_fence_u(rr_b2(rr_h2(a) * wxp))) * wym)));
            tt = rr_h2(rr_fence_u(rr_b2(mad(rr_h2(rr_fence_u(rr_b2(rr_h2(b) * wxm))), wym, tt))));
            tt = rr_h2(rr_fence_u(rr_b2(mad(rr_h2(rr_fence_u(rr_b2(rr_h2(c) * wxm))), wyp, tt))));
            tt = rr_h2(rr_fence_u(rr_b2(mad(rr_h2(rr_fence_u(rr_b2(rr_h2(d) * wxp))), wyp, tt))));
            if (e == 0) u.x = rr_b2(tt); else u.y = rr_b2(tt);
        }
        const uint gt01 = rr_fence_u(rr_b2(rr_h2(s01) * g2)), gt23 = rr_fence_u(rr_b2(rr_h2(s23) * g2));
        const uint ug01 = rr_fence_u(rr_b2(rr_h2(u.x) * rr_h2(gt01))), ug23 = rr_fence_u(rr_b2(rr_h2(u.y) * rr_h2(gt23)));
        const uint hh01 = rr_fence_u(rr_b2(rr_h2(ug01) * z2));
        const uint hh23 = rr_fence_u(rr_b2(rr_h2(ug23) * z2)) & 0xffffu;
        hn.x = rr_fence_u(rr_b2(mad(rr_h2(hold.x), omz, rr_h2(hh01))));
        hn.y = rr_fence_u(rr_b2(mad(rr_h2(hold.y), omz, rr_h2(hh23))));
#endif
        const uint otok2 = (uint)(Y * Nx + X);
        if (real && rr_hf == 0u) RR_PST64(pHnew, otok2 * 8u, hn);
#if RR_HMERGE
        const half2 quarter = half2(0.25, 0.25);
        const uint qx = rr_fence_u(rr_b2(rr_h2(hn.x) * quarter)), qy = rr_fence_u(rr_b2(rr_h2(hn.y) * quarter));
        const uint l0 = 2u * (n & 3u);
        const uint q0x = WaveReadLaneAt(qx, l0), q1x = WaveReadLaneAt(qx, l0 + 1u);
        const uint q2x = WaveReadLaneAt(qx, l0 + 8u), q3x = WaveReadLaneAt(qx, l0 + 9u);
        const uint q0y = WaveReadLaneAt(qy, l0), q1y = WaveReadLaneAt(qy, l0 + 1u);
        const uint q2y = WaveReadLaneAt(qy, l0 + 8u), q3y = WaveReadLaneAt(qy, l0 + 9u);
        const uint ax = rr_fence_u(rr_b2(rr_h2(q0x) + rr_h2(q1x))), bx = rr_fence_u(rr_b2(rr_h2(q2x) + rr_h2(q3x)));
        const uint ay = rr_fence_u(rr_b2(rr_h2(q0y) + rr_h2(q1y))), by = rr_fence_u(rr_b2(rr_h2(q2y) + rr_h2(q3y)));
        const uint2 hm = uint2(rr_b2(rr_h2(ax) + rr_h2(bx)), rr_b2(rr_h2(ay) + rr_h2(by)));
        const int hXc = cx0 + (int)(n & 3u), hYc = cy0 + (int)wv;
        if (n < 4u && rr_hf == 0u && hXc >= 0 && hXc < Mx && hYc >= 0 && hYc < My)
            RR_PST64(pHmerged, (uint)(hYc * Mx + hXc) * 8u, hm);
#endif
    }
#endif
#if RR_DFLOW
    rr_df_signal(wj);
#endif
    DBG_FLUSH
}
