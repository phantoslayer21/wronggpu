// SPDX-License-Identifier: MIT
// Spec 51 window block (51.3) with its stage fronts (51.5) and tails (51.4) on RDNA 4 / D3D12 (DESIGN-51.md).
// One workgroup = one 8 x 8 window, 4 waves of 32 lanes; wave wv owns tokens 16 wv .. 16 wv + 15 (window rows 2 wv,
// 2 wv + 1); lane l = 16 hf + n: token n of the wave's tile, half hf of every 16-channel fragment.
// Every product is a chain of 51.1 steps: two v_wmma_f32_16x16x16_fp8_fp8 on an f32 accumulator whose start value is
// the f16 start value, then one f16 rounding (v_cvt_f16_f32, RNE). Layouts: clean/rr_f/gpu/layout.py.
//
// Defines (one kernel per stage kind; clean/rr_f/gpu/stage.py):
//   RR_C, RR_H      channels, heads
//   RR_KIND         0 IN0 / encoder (hidden rule of the encoder), 2 fusion half Dk-F, 3 recurrent half Dk-R
//   RR_CN           merge output channels (E1-E4), 0 = no merge
//   RR_HMERGE       1: merged hidden (IN0 and the encoder stages with a merge)
//   RR_ADAPT        1: E1 input adapter 128 -> 64
//   RR_KC           fusion halves: channels of the coarse input
//   RR_STEP         0: variant E, 1: variant X (4 WMMAs), 2: optional exact binary64 scalar fragment step,
//                   3: variant Y, the exact step on split operands (rr_step_y.hlsli)
//   RR_ZFAST        1 (with RR_STEP 3, "variant Z"): a wave takes a step on one chain of the unsplit operands when its
//                   operand block is in a proven class (rr_step_y.hlsli); c_zt0 = codes T of the classes x, o, W1, W2,
//                   c_zt1 = T of the front (Wa / Wup), Wh, then the int-class bits (1 x, 2 o, 4 W1, 8 W2, 16 front,
//                   32 Wh, 64 Wm)
//                   2 ("variant Z2", levers L2 / L5 of results/2026-10-20-speed-body.md): range gates per operand block
//                   (bm = value(largest code) 2^(10 - Elo) < R, R = 2^(g + 24) / L1 per class; c_zt0 = R bytes of x, o,
//                   W1, W2, c_zt1 = R bytes of the front, Wh, Wm; stage.z2_params), also for O (P: L1 <= 9/8) and the
//                   merge; scores on one chain (bm(q) bm(k) < 2^19) or two chains (zs2_ok) before the four-chain step;
//                   every other step "Y2": both chains from 0, one v_fma_mix when |M + L| < 2^(g + 15) (c_ythr: the
//                   class grids g + 15 as nibbles), else the window's fixup mark (rr_step_y.hlsli)
//   RR_FIXUP        1 (with RR_STEP 1): only the windows a variant-Y dispatch marked in pYflag, then clears the mark
//   RR_LDSW         1 (default): weight units staged through LDS (QV_h, Wo_h, MLP chunks; DESIGN-51.md 7.3),
//                   0: weight fragments read straight from memory by every wave
//   RR_Z            0: Z = f32 sum (spec-accepted order), 1: Z supplied (pZ, [window][head][query] f16),
//                   2: Z = the exact sum rounded once (the reference's choice)
#include "wg_platform.hlsli"

WG_ARGS_BEGIN
    uint pX, pCin, pHold, pHc;
    uint pWqv, pTb, pWo, pBo, pMlp, pB1, pB2, pWm, pBm, pWa, pBa, pWup, pBup, pWh, pBh;
    uint pSig, pZ, pY2, pMerged, pHnew, pHmerged;
    int Nx, Ny, sx, sy, nwx;
    uint c_ra_m, c_f_g;
    uint wc0, wc1, wc2, wc3, wc4;
    uint pYflag, c_ythr;   // variant Y: per-window fixup marks, bound of the start values (f32 bits)
    uint c_zt0, c_zt1;     // variant Z (Y + RR_ZFAST): fast-step classes (stage.z_params)
WG_ARGS_END

#include "rr_common.hlsli"

#ifndef RR_KIND
#define RR_KIND 0
#endif
#ifndef RR_CN
#define RR_CN 0
#endif
#ifndef RR_HMERGE
#define RR_HMERGE 0
#endif
#ifndef RR_ADAPT
#define RR_ADAPT 0
#endif
#ifndef RR_KC
#define RR_KC 0
#endif
#ifndef RR_Z
#define RR_Z 0
#endif

#define RC RR_C
#define RH RR_H
#define KS (RC / 16)          // 16-channel fragments of a token
#define NS (RC / 32)          // 51.1 steps over the channels
#define HID (4 * RC)          // MLP width
#define NJ (HID / 32)         // MLP chunks
#define HIDDEN (RR_KIND == 0 || RR_KIND == 3)
#define NHP (RR_KIND == 3 ? 4 : 3)   // hidden projection outputs used

// ---- LDS --------------------------------------------------------------------------------------------------------------
#ifndef RR_LDSW
#define RR_LDSW 1
#endif
#define UNIT (64 * RC)        // bytes of a weight unit: QV_h (Wq_h | Wv_h) or an MLP chunk (W1 | W2); Wo_h is half
#if RR_LDSW
#if RR_STEP == 4
// variant I (rr_step_i.hlsli): every weight unit holds two int8 digit planes [plane][unit] (layout.int_planes)
#define WUNIT (2 * UNIT)
groupshared uint64_t rr_wb[2 * WUNIT / 8];
groupshared uint64_t rr_xq[512];                  // q codes, v^T codes (the digits are made by each consumer)
#define XQ_BUF(h) 0u
#define MY(i) rr_wb[i]
#define UP(i) rr_wb[WUNIT / 8 + (i)]
#else
// two unit buffers (QV_h / MLP chunk j even in buffer 0, Wo_h / MLP chunk j odd in buffer 1) and one q / v exchange
// buffer (two barriers per head); the merge input and the fusion up-projections overlay the unit buffers.
groupshared uint64_t rr_wb[2 * UNIT / 8];
#if RR_STEP == 3
groupshared uint64_t rr_xq[1024];                 // q integer parts, v^T integer parts, fractions, q fractions
#else
groupshared uint64_t rr_xq[512];
#endif
#define XQ_BUF(h) 0u
#define MY(i) rr_wb[i]                            // buffer 0: free after the MLP (last read by chunk NJ - 2)
#define UP(i) rr_wb[UNIT / 8 + (i)]               // buffer 1: free until the first Wo unit
#endif
#else
#if RR_STEP == 4
#define WUNIT (2 * UNIT)                          // variant I with RR_LDSW 0 (C >= 128: the 32 KB of LDS): planes read
#endif                                            // from memory
// q / v exchange, two buffers of 512 elements (8 bytes): q [wave][ft][lane] then v^T [wave][ft][lane].
// Variant Y / Z (RR_STEP 3, r2 lever 4): ONE exchange buffer of 1024 elements (q integer parts | v^T integer parts |
// v^T fractions or q unsplit | q fractions) and a second barrier at the end of every head (RR_XQ1), so the 16 KB
// double buffer is not needed; the merge input (planes h | l) overlays the exchange buffer at C <= 64 (written after
// the last head's closing barrier, when every wave has read the exchange).
#if RR_STEP == 3 && defined(RR_ZFAST) && RR_ZFAST >= 2
#error "variant Z2 needs RR_LDSW 1"
#endif
#define RR_XQ1 (RR_STEP == 3)
#define MY_OV (RR_STEP == 3 && RC <= 64)
groupshared uint64_t rr_xq[1024 + (RR_CN > 0 && !MY_OV ? (RR_STEP == 3 ? 2 : 1) * 64 * RC / 8 : 0)];
#define XQ_BUF(h) (RR_XQ1 ? 0u : ((h) & 1u) * 512u)
#if RR_CN > 0
#define MY(i) rr_xq[(MY_OV ? 0u : 1024u) + (i)]      // the window's y2, [token][C] bytes (merge input; Y / Z: + fractions)
#endif
#if RR_KIND == 2
groupshared uint64_t rr_up[16 * HID / 8];     // the window's 4 x 4 quantised up-projections, [coarse r][4C] bytes
#define UP(i) rr_up[i]
#endif
#endif
#if RR_STEP == 3 && defined(RR_ZFAST) && RR_ZFAST >= 2
groupshared uint rr_zsq[8], rr_zsv[8], rr_zsy[4];   // RR_ZFAST 2: per wave range statistics of q, v (head h: slots
                                                     // 4 (h & 1) + wave) and of y2
#endif
#if RR_STEP == 3 && defined(RR_ZFAST) && RR_ZFAST
// per wave: bit 0 q int, bit 1 v int (head h: ZFL_H slots 4 (h & 1) + wave), bit 2 y2 int (merge: ZFL_M slots wave)
#if RR_LDSW && RR_C == 64 && RR_ZFAST == 1
// Level 1 (lever 1 of the r2 research): the 48 B of flags put E1 / D1-F / D1-R at 16 432 B of LDS, one workgroup per
// CU less (3 instead of 4). The flags live in space that is free in their barrier intervals instead: the head flags in
// the upper half of unit buffer 1 (during the head loop it holds only Wo_h, UNIT / 2 bytes; the flags are written
// before the head's q / v exchange barrier and read after it, as Wo_h; MLP chunk 1 overwrites it only after the
// loop's chunk-0 barrier), the merge flags in rr_xq (last read by the O step of the last head, before that barrier).
// Same values, same barrier intervals: storage only.
#define ZFL_H(i) rr_wb[UNIT / 8 + UNIT / 16 + (i)]
#define ZFL_M(i) rr_xq[i]
#else
groupshared uint rr_zfl[12];  // (head slots double-buffered as rr_xq for RR_LDSW 0)
#define ZFL_H(i) rr_zfl[i]
#define ZFL_M(i) rr_zfl[8u + (i)]
#endif
#endif
uint2 rr_q2(uint64_t q) { return uint2((uint)q, (uint)(q >> 32)); }
uint64_t rr_2q(uint2 v) { return (uint64_t)v.x | ((uint64_t)v.y << 32); }

// ---- small helpers ---------------------------------------------------------------------------------------------------
static uint rr_lane;   // lane in the wave
static uint rr_hf;     // lane >> 4

uint2 wfrag(uint base, uint f) { return wg_ld64(base, f * 256u + rr_lane * 8u); }
v8f zero8() { return wg_splat(0.0f); }
// 8 f32 bias values of accumulator tile `tile` for this lane: rows 16 tile + 8 hf .. + 7.
v8f bias_tile(uint p, uint tile) { return wg_load8f(p, (16u * tile + 8u * rr_hf) * 4u); }

// Weight units (layout.py): QV_h = [q|v][ft][ks] (4 KS fragments), Wo_h = [ct][2] (2 KS), MLP chunk j =
// W1 [ft][ks] (2 KS) then W2 [ct][2] (2 KS).
#if RR_LDSW && RR_STEP == 4
// digit plane p of a fragment of the staged units (QV_h in buffer 0, Wo_h in buffer 1, MLP chunk j in buffer j & 1)
uint2 qv_fp(uint f, uint p) { return rr_q2(rr_wb[p * (UNIT / 8) + f * 32u + rr_lane]); }
uint2 wo_fp(uint f, uint p) { return rr_q2(rr_wb[WUNIT / 8 + p * (UNIT / 16) + f * 32u + rr_lane]); }
uint2 mlp_fp(uint j, uint f, uint p) { return rr_q2(rr_wb[(j & 1u) * (WUNIT / 8) + p * (UNIT / 8) + f * 32u + rr_lane]); }
#define PF (WUNIT / 16 / WG_BLOCK)
struct RrPf { uint4 v[PF]; };
RrPf st_fetch(uint src, uint bytes)
{
    RrPf p;
    [unroll] for (uint i = 0; i < PF; ++i) {
        const uint off = (i * WG_BLOCK + wg_lane_id()) * 16u;
        p.v[i] = wg_ld128(src, min(off, bytes - 16u));
    }
    return p;
}
void st_commit(RrPf p, uint buf, uint bytes)
{
    [unroll] for (uint i = 0; i < PF; ++i) {
        const uint off = (i * WG_BLOCK + wg_lane_id()) * 16u;
        if (off < bytes) {
            rr_wb[buf * (WUNIT / 8) + (off >> 3)] = rr_2q(p.v[i].xy);
            rr_wb[buf * (WUNIT / 8) + (off >> 3) + 1u] = rr_2q(p.v[i].zw);
        }
    }
}
#elif RR_STEP == 4
// RR_LDSW 0: the planes straight from memory (h / the chunk index in scope at the call sites)
#define qv_fp(f, p) wfrag(pWqv + h * WUNIT + (p) * UNIT, (f))
#define wo_fp(f, p) wfrag(pWo + h * (WUNIT / 2) + (p) * (UNIT / 2), (f))
#define mlp_fp(j, f, p) wfrag(pMlp + (j) * WUNIT + (p) * UNIT, (f))
#elif RR_LDSW
uint2 qv_frag(uint h, uint f) { return rr_q2(rr_wb[f * 32u + rr_lane]); }
uint2 wo_frag(uint h, uint f) { return rr_q2(rr_wb[UNIT / 8 + f * 32u + rr_lane]); }
uint2 mlp_frag(uint j, uint f) { return rr_q2(rr_wb[(j & 1u) * (UNIT / 8) + f * 32u + rr_lane]); }
// Stager: every lane fetches 16-byte chunks of the next unit into registers during the current phase and writes them
// to LDS after it (one barrier makes them visible).
#define PF (UNIT / 16 / WG_BLOCK)
struct RrPf { uint4 v[PF]; };
RrPf st_fetch(uint src, uint bytes)
{
    RrPf p;
    [unroll] for (uint i = 0; i < PF; ++i) {
        const uint off = (i * WG_BLOCK + wg_lane_id()) * 16u;
        p.v[i] = wg_ld128(src, min(off, bytes - 16u));
    }
    return p;
}
void st_commit(RrPf p, uint buf, uint bytes)
{
    [unroll] for (uint i = 0; i < PF; ++i) {
        const uint off = (i * WG_BLOCK + wg_lane_id()) * 16u;
        if (off < bytes) {
            rr_wb[buf * (UNIT / 8) + (off >> 3)] = rr_2q(p.v[i].xy);
            rr_wb[buf * (UNIT / 8) + (off >> 3) + 1u] = rr_2q(p.v[i].zw);
        }
    }
}
#else
uint2 qv_frag(uint h, uint f) { return wfrag(pWqv + h * UNIT, f); }
uint2 wo_frag(uint h, uint f) { return wfrag(pWo + h * (UNIT / 2), f); }
uint2 mlp_frag(uint j, uint f) { return wfrag(pMlp + j * UNIT, f); }
#endif
#ifndef RR_STEP
#define RR_STEP 0
#endif
#if RR_STEP == 2
#include "rr_exact_math.hlsli"
v8f step2(uint2 a0, uint2 b0, uint2 a1, uint2 b1, v8f c)
{
    return rr_exact_step2(a0, b0, a1, b1, c);
}
#endif
#if RR_STEP == 1 || RR_STEP == 3
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
v8f rr_xstep2(uint2 a0, uint2 b0, uint2 a1, uint2 b1, v8f c)
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
#endif
#if RR_STEP == 4
#include "rr_step_i.hlsli"
// tables of the weight tensors (after the planes; layout.pack_weights_int): per (tile, step) (beta, L1' bits)
#define TQV(h, qv, ft, s) wg_ld64(pWqv + RH * WUNIT, ((((h) * 2u + (qv)) * 2u + (ft)) * NS + (s)) * 8u)
#define TWO(h, ct) wg_ld64(pWo + RH * (WUNIT / 2), ((h) * KS + (ct)) * 8u)
#define TW1(j, ft, s) wg_ld64(pMlp + NJ * WUNIT, (((j) * 2u + (ft)) * NS + (s)) * 8u)
#define TW2(j, ct) wg_ld64(pMlp + NJ * WUNIT, (NJ * 2u * NS + (j) * KS + (ct)) * 8u)
// directly read weight tensors: planes of PB bytes, then the table [tile][step] of uint2
#define PB_A (4u * 8u * 256u)                                  // Wa^T [64][128]
#define PB_U ((HID / 16u) * (RR_KC / 16u) * 256u)              // Wup^T [4C][Kc]
#define PB_M ((RR_CN / 16u) * (HID / 16u) * 256u)              // Wm^T [Cn][4C]
#define PB_H (KS * 256u)                                       // Wh^T [16][C]
#endif
#if RR_STEP == 1 || RR_STEP == 3
v8f step2(uint2 a0, uint2 b0, uint2 a1, uint2 b1, v8f c) { return rr_xstep2(a0, b0, a1, b1, c); }
#elif RR_STEP == 0
// One 51.1 step (variant E): two 16-deep WMMAs on the f32 accumulator, then the f16 rounding.
v8f step2(uint2 a0, uint2 b0, uint2 a1, uint2 b1, v8f c)
{
#if defined(RR_YABL) && (RR_YABL & 64)
    v8f dd0 = wg_wmma_fp8(a0, b0 ^ 1u, wg_splat(0.0f));
    dd0 = wg_wmma_fp8(a1, b1 ^ 1u, dd0);
    c.v[0] = asfloat(asuint(c.v[0]) ^ (asuint(dd0.v[0]) & rr_zero_bits()));
#endif
    c = wg_wmma_fp8(a0, b0, c);
    c = wg_wmma_fp8(a1, b1, c);
    return rr_r16(c);
}
#endif
// The step with the result as packed f16 (for chains kept packed between steps, and the scores).
uint4 step2h(uint2 a0, uint2 b0, uint2 a1, uint2 b1, v8f c)
{
#if RR_STEP == 1 || RR_STEP == 2 || RR_STEP == 3
    return rr_h16(step2(a0, b0, a1, b1, c));
#else
#if defined(RR_YABL) && (RR_YABL & 64)
    v8f dd0 = wg_wmma_fp8(a0, b0 ^ 1u, wg_splat(0.0f));
    dd0 = wg_wmma_fp8(a1, b1 ^ 1u, dd0);
    c.v[0] = asfloat(asuint(c.v[0]) ^ (asuint(dd0.v[0]) & rr_zero_bits()));
#endif
    c = wg_wmma_fp8(a0, b0, c);
    c = wg_wmma_fp8(a1, b1, c);
    return rr_h16(c);
#endif
}
uint4 pk_mad(uint4 a, half2 b, uint4 c)   // R(a b + c) per half, fenced
{
    uint4 r;
    r.x = rr_fence_u(rr_b2(mad(rr_h2(a.x), b, rr_h2(c.x))));
    r.y = rr_fence_u(rr_b2(mad(rr_h2(a.y), b, rr_h2(c.y))));
    r.z = rr_fence_u(rr_b2(mad(rr_h2(a.z), b, rr_h2(c.z))));
    r.w = rr_fence_u(rr_b2(mad(rr_h2(a.w), b, rr_h2(c.w))));
    return r;
}
uint4 d8h(uint2 q) { return rr_h16(rr_d8(q)); }   // d8 as packed f16 (exact)
#if RR_STEP == 4
#define step2h(a0, b0, a1, b1, c) step2h_is_not_exact_in_variant_I
#endif
#if RR_STEP == 3
#include "rr_step_y.hlsli"
#endif
#if RR_STEP != 3 || !defined(RR_ZFAST)
#undef RR_ZFAST
#define RR_ZFAST 0
#endif
// class threshold byte k of a dword, as the per-byte addend of z_sbad; int-class bit
#ifndef RR_ZOFF
#define RR_ZOFF 0                                 // debug: fast paths switched off (1 front, 2 x, 4 scores, 8 O, 16 Dp,
#endif                                            // 32 W1, 64 W2, 128 merge, 256 Wh)
// RR_ZFORCE (timing only, NOT exact, never in a shipped build): the fast path taken at the sites of the set bits
// regardless of the class checks (same bits as RR_ZOFF); ZG(bit, check) = the site's decision
#ifndef RR_ZFORCE
#define RR_ZFORCE 0
#endif
#if RR_ZFORCE
#define ZG(bit, check) ((((RR_ZFORCE) & (bit)) != 0 || (check)) && !((RR_ZOFF) & (bit)))
#else
// (same tokens as the RR_ZOFF tests of e11ec7a, so that the step X / Y / Z kernels preprocess unchanged)
#define ZG(bit, check) check && !(RR_ZOFF & bit)
#endif
#define ZT(w, k) (((w) >> (8u * (k))) & 0xffu)
#define ZTK(w, k) z_tk(ZT(w, k))
#if RR_ZFAST >= 2
#define ZTT(w, k) zr_T(ZT(w, k))                  // RR_ZFAST 2: the bytes are R; the split threshold T from R
// Y2 steps: the bound 2^(g + 15) of class k (c_ythr nibble k = g + 15: 0 x, 1 Wo, 2 W1, 3 W2, 4 front, 5 Wh, 6 Wm)
#define YB(k) ldexp(1.0f, (float)((c_ythr >> (4u * (k))) & 15u))
#define YSET(k) rr_yb = YB(k)
#define RR_ZO_R (RR_Z == 1 ? 1024.0f : 29127.0f)  // O: 2^15 / (bound of the row sums of P over 32 keys: 9/8, or 32)
#else
#define ZTT(w, k) ZT(w, k)
#define YSET(k)
#endif
#define ZIM(bit) (((c_zt1 >> 16) & (bit)) != 0u)
// RR_ZSTAT (diagnostics only, the results are not used): every wave ORs the bit of each site where it left the fast
// path into its window's pYflag word (1 front, 2 x, 4 scores, 8 O, 16 Dp, 32 W1, 64 W2, 128 merge, 256 Wh; the
// slow score tiles are counted in bits 16-31); run without the fixup dispatches (gpu/zstat.py)
#ifdef RR_ZSTAT
// and counts per wave decision: after the fixup count, per site (index = log2 bit) two words: decisions, slow ones
#define ZST(bit, fast) if (WaveIsFirstLane()) { uint zo_;     const uint zc_ = pYflag + (wg_grid_size() + 1u + 2u * firstbitlow((uint)(bit))) * 4u;     wg_mem.InterlockedAdd(zc_, 1u, zo_);     if (!(fast)) { wg_mem.InterlockedAdd(zc_ + 4u, 1u, zo_); wg_mem.InterlockedOr(pYflag + wid * 4u, (bit), zo_); } }
#define ZSTN(v)
// Candidate gates (lever measurements, RR_ZSTAT only; the decisions above are unchanged): per wave, the exponent-field
// span of an operand block (largest max(e, 1) minus smallest max(e, 1) over the nonzero codes) against the host bound
// D of the weight class, D = floor(g + 20 - log2 L1) (exact single chain: L1 Bmax < 2^(g + gb + 24) with Bmax <
// 2^(Ehi - 6), gb >= Elo - 10). Counts per candidate after the site counts (decisions, failures), window bits 16-23;
// the bounds D (int) are read from the mark buffer at word grid + 37 + k (k: 0 x, 1 Wo, 2 W1, 3 W2, 4 front, 5 Wh,
// 6 Wm, 7 P of the attention (L1 <= 9/8 for the 32 keys of a step)).
uint zs_acc(uint acc, uint x)
{
    const uint a = x & 0x7f7f7f7fu, w = (0x80808080u - a) & 0x7f7f7f7fu;
    const uint m = max(acc & 0xffu, max(max(a & 0xffu, (a >> 8) & 0xffu), max((a >> 16) & 0xffu, a >> 24)));
    const uint n = max((acc >> 8) & 0xffu, max(max(w & 0xffu, (w >> 8) & 0xffu), max((w >> 16) & 0xffu, w >> 24)));
    return m | (n << 8);
}
uint zs_acc2(uint acc, uint2 f) { return zs_acc(zs_acc(acc, f.x), f.y); }
uint zs_wave(uint acc) { return WaveActiveMax(acc & 0xffu) | (WaveActiveMax((acc >> 8) & 0xffu) << 8); }
uint zs_merge(uint a, uint b) { return max(a & 0xffu, b & 0xffu) | max(a & 0xff00u, b & 0xff00u); }
int zs_span(uint st)
{
    const uint am = st & 0xffu, wm = (st >> 8) & 0xffu;
    if (am == 0x7fu) return 99;
    if (wm == 0u) return 0;
    return (int)max(am >> 3, 1u) - (int)max((0x80u - wm) >> 3, 1u);
}
#define ZD(k) ((int)wg_ld32(pYflag, (wg_grid_size() + 37u + (k)) * 4u))
#define ZC(idx, pass) if (WaveIsFirstLane()) { uint zo_; const uint zc_ = pYflag + (wg_grid_size() + 21u + 2u * (idx)) * 4u;     wg_mem.InterlockedAdd(zc_, 1u, zo_);     if (!(pass)) { wg_mem.InterlockedAdd(zc_ + 4u, 1u, zo_); wg_mem.InterlockedOr(pYflag + wid * 4u, 0x10000u << (idx), zo_); } }
groupshared uint rr_zst[16];   // per wave: q stats, v stats (slots 0-3, 4-7), y2 stats (8-11), q max token span (12-15)
// histograms (32 bins from word grid + 45): 0 front span - D + 16, 1 S span_q + span_k, 2 S max token spans summed,
// 3 W2 span - D + 16, 4 Dp span - D + 16
#define ZH(hh, v) if (WaveIsFirstLane()) { uint zo_; wg_mem.InterlockedAdd(pYflag + (wg_grid_size() + 45u + 32u * (hh) + (uint)clamp((int)(v), 0, 31)) * 4u, 1u, zo_); }
// per lane: span of its token (the 16 bytes of this lane and of lane ^ 16), as zs_span
uint zs_tok(uint acc) { const uint o = WaveReadLaneAt(acc, WaveGetLaneIndex() ^ 16u); return zs_merge(acc, o); }
#else
#define ZST(bit, fast)
#define ZSTN(v)
#define ZC(idx, pass)
#define ZH(hh, v)
#endif

WG_KERNEL(rr_block)
{
    WG_SETUP();
    rr_lane = WaveGetLaneIndex();
    rr_hf = rr_lane >> 4;
    const uint wv = wg_lane_id() >> 5;
    const uint n = rr_lane & 15u;
    const uint wid = wg_block_id();
#ifndef RR_FIXUP
#define RR_FIXUP 0
#endif
#if RR_FIXUP
    if (wg_ld32(pYflag, wid * 4u) == 0u) return;   // window computed exactly by the variant-Y dispatch
#endif
#if RR_STEP == 3
    rr_yflag = 0u;
    rr_ymax = 0.0f;
#elif RR_STEP == 4
    rr_yflag = 0u;
#endif
#ifdef RR_ZSTAT
    if (wg_lane_id() == 0u) { uint zp_; wg_mem.InterlockedOr(pYflag + wid * 4u, 0x4000u, zp_); }   // window present
#endif
    const int wi = (int)(wid % (uint)nwx), wj = (int)(wid / (uint)nwx);
    const int lx = (int)(n & 7u), ly = (int)(2u * wv + (n >> 3));
    const int X = 8 * wi - sx + lx, Y = 8 * wj - sy + ly;
    const int Xr = rr_reflect(X, Nx), Yr = rr_reflect(Y, Ny);
    const bool real = X >= 0 && X < Nx && Y >= 0 && Y < Ny;
    const uint tok = (uint)(Yr * Nx + Xr);
    const RrWc wc = rr_wc(wc0, wc1, wc2, wc3, wc4);
    const half2 ra2 = rr_splat_lo(c_ra_m), m2 = rr_splat_hi(c_ra_m);
    const half2 f2 = rr_splat_lo(c_f_g), g2 = rr_splat_hi(c_f_g);
    const int Mx = Nx >> 1, My = Ny >> 1;                    // coarse extents (fusion / recurrent / merge)
    const int cx0 = (8 * wi - sx) >> 1, cy0 = (8 * wj - sy) >> 1;   // coarse coordinates of the window's corner

    // ---- block input x (B fragments: K-slice ks = channels 16 ks .. + 15 of the lane's token) and residual base -----
    uint2 xf[KS];
#if RR_KIND == 2
    uint4 xr[KS];                                            // xf of 51.5 (f16), the first residual base
#endif
#if RR_ADAPT
    {   // E1: x = Q8(MM(in; Wa; ba)), in = the level-1 token's 128 channels as stored (51.5, 51.6)
        uint2 inf[8];
        [unroll] for (uint k = 0; k < 8; ++k) inf[k] = wg_ld64(pX, tok * 128u + 16u * k + 8u * rr_hf);
#if RR_STEP == 3
        RrSp ins[8];
#if RR_ZFAST
        uint zsb = 0u, zib = 0u;
#if RR_ZFAST >= 2
        [unroll] for (uint k0 = 0; k0 < 8; ++k0) zsb = zr_acc(zsb, inf[k0]);
        const bool zfa = ZG(1, zr_pass(zr_wave(zsb), ZT(c_zt1, 0)));
        YSET(4);
#else
        [unroll] for (uint k0 = 0; k0 < 8; ++k0) { zsb |= z_sbad(inf[k0], ZTK(c_zt1, 0)); zib |= z_ibad(inf[k0]); }
        const bool zfa = ZG(1, z_fast_ok(zsb, zib, ZIM(16u)));
#endif
#ifdef RR_ZSTAT
        { uint st = 0u; [unroll] for (uint q0 = 0; q0 < 8; ++q0) st = zs_acc2(st, inf[q0]); ZC(0, zs_span(zs_wave(st)) <= ZD(4)); }
#endif
        ZST(1u, zfa);
        [unroll] for (uint k1 = 0; k1 < 8; ++k1) { ins[k1].h = inf[k1]; ins[k1].l = inf[k1]; }
        [branch] if (!zfa) {
            [unroll] for (uint k1 = 0; k1 < 8; ++k1) ins[k1] = z_split(inf[k1], ZTT(c_zt1, 0));
        }
#else
        [unroll] for (uint k1 = 0; k1 < 8; ++k1) ins[k1] = rr_split(inf[k1]);
#endif
#endif
#if RR_STEP == 4
        RrAB ina[4];
        [unroll] for (uint s4 = 0; s4 < 4; ++s4) ina[s4] = ri_act(inf[2 * s4], inf[2 * s4 + 1]);
        [unroll] for (uint ct = 0; ct < KS; ++ct) {
            v8f acc = bias_tile(pBa, ct);
            [unroll] for (uint s = 0; s < 4; ++s) {
                const uint f0 = ct * 8u + 2u * s;
                const uint2 a00 = wfrag(pWa, f0), a01 = wfrag(pWa + PB_A, f0);
                const uint2 a10 = wfrag(pWa, f0 + 1u), a11 = wfrag(pWa + PB_A, f0 + 1u);
                const uint2 t = wg_ld64(pWa + 2u * PB_A, (ct * 4u + s) * 8u);
                acc = rr_unh(ri_step_w(a00, a01, a10, a11, ina[s], acc, t));
            }
            xf[ct] = rr_q8(acc);
        }
#else
        [unroll] for (uint ct = 0; ct < KS; ++ct) {
            uint2 a[8];
            [unroll] for (uint k2 = 0; k2 < 8; ++k2) a[k2] = wfrag(pWa, ct * 8u + k2);
            v8f acc = bias_tile(pBa, ct);
#if RR_STEP == 3
#if RR_ZFAST
            [branch] if (zfa) {
                [unroll] for (uint s = 0; s < 4; ++s)
                    acc = rr_unh(z_fast(a[2 * s], inf[2 * s], a[2 * s + 1], inf[2 * s + 1], acc));
            } else
#endif
            {
                acc = rr_unh(y_step_b(a[0], ins[0], a[1], ins[1], pBa, 64u, ct));
                [unroll] for (uint s = 1; s < 4; ++s)
                    acc = rr_unh(y_step(a[2 * s], ins[2 * s], a[2 * s + 1], ins[2 * s + 1], acc));
            }
#else
            [unroll] for (uint s = 0; s < 4; ++s) acc = step2(a[2 * s], inf[2 * s], a[2 * s + 1], inf[2 * s + 1], acc);
#endif
            xf[ct] = rr_q8(acc);
        }
#endif
    }
#elif RR_KIND == 2
    {   // fusion half (51.5): up-projection of the window's 4 x 4 coarse tokens (coarse reads reflected with M),
        // quantised, to LDS; then xf = R(d8(skip) f + d8(upq)) per fine token, x = Q8(xf), xr = xf.
        const int crx = cx0 + (int)(n & 3u), cry = cy0 + (int)(n >> 2);   // lane n = coarse token r = 4 ry + rx
        const uint ctok = (uint)(rr_reflect(cry, My) * Mx + rr_reflect(crx, Mx));
        uint2 cf[RR_KC / 16];
        [unroll] for (uint k = 0; k < RR_KC / 16; ++k) cf[k] = wg_ld64(pCin, ctok * RR_KC + 16u * k + 8u * rr_hf);
#if RR_STEP == 3
        RrSp cfs[RR_KC / 16];
#if RR_ZFAST
        uint zsb = 0u, zib = 0u;
#if RR_ZFAST >= 2
        [unroll] for (uint k0 = 0; k0 < RR_KC / 16; ++k0) zsb = zr_acc(zsb, cf[k0]);
        const bool zfu = ZG(1, zr_pass(zr_wave(zsb), ZT(c_zt1, 0)));
        YSET(4);
#else
        [unroll] for (uint k0 = 0; k0 < RR_KC / 16; ++k0) { zsb |= z_sbad(cf[k0], ZTK(c_zt1, 0)); zib |= z_ibad(cf[k0]); }
        const bool zfu = ZG(1, z_fast_ok(zsb, zib, ZIM(16u)));
#endif
#ifdef RR_ZSTAT
        { uint st = 0u; [unroll] for (uint q0 = 0; q0 < RR_KC / 16; ++q0) st = zs_acc2(st, cf[q0]); ZC(0, zs_span(zs_wave(st)) <= ZD(4)); ZH(0, zs_span(zs_wave(st)) - ZD(4) + 16); }
#endif
        ZST(1u, zfu);
        [unroll] for (uint k1 = 0; k1 < RR_KC / 16; ++k1) { cfs[k1].h = cf[k1]; cfs[k1].l = cf[k1]; }
        [branch] if (!zfu) {
            [unroll] for (uint k1 = 0; k1 < RR_KC / 16; ++k1) cfs[k1] = z_split(cf[k1], ZTT(c_zt1, 0));
        }
#else
        [unroll] for (uint k1 = 0; k1 < RR_KC / 16; ++k1) cfs[k1] = rr_split(cf[k1]);
#endif
#endif
#if RR_STEP == 4
        RrAB cfa[RR_KC / 32];
        [unroll] for (uint s4 = 0; s4 < RR_KC / 32; ++s4) cfa[s4] = ri_act(cf[2 * s4], cf[2 * s4 + 1]);
        [loop] for (uint ot = wv; ot < HID / 16 + rr_zero(); ot += 4u) {
            v8f acc = bias_tile(pBup, ot);
            [unroll] for (uint s = 0; s < RR_KC / 32; ++s) {
                const uint f0 = ot * (RR_KC / 16) + 2u * s;
                const uint2 a00 = wfrag(pWup, f0), a01 = wfrag(pWup + PB_U, f0);
                const uint2 a10 = wfrag(pWup, f0 + 1u), a11 = wfrag(pWup + PB_U, f0 + 1u);
                const uint2 t = wg_ld64(pWup + 2u * PB_U, (ot * (RR_KC / 32) + s) * 8u);
                acc = rr_unh(ri_step_w(a00, a01, a10, a11, cfa[s], acc, t));
            }
            UP((n * HID + 16u * ot + 8u * rr_hf) >> 3) = rr_2q(rr_q8(acc));
        }
#else
        [loop] for (uint ot = wv; ot < HID / 16 + rr_zero(); ot += 4u) {
            uint2 a[RR_KC / 16];
            [unroll] for (uint k2 = 0; k2 < RR_KC / 16; ++k2) a[k2] = wfrag(pWup, ot * (RR_KC / 16) + k2);
            v8f acc = bias_tile(pBup, ot);
#if RR_STEP == 3
#if RR_ZFAST
            [branch] if (zfu) {
                [unroll] for (uint s = 0; s < RR_KC / 32; ++s)
                    acc = rr_unh(z_fast(a[2 * s], cf[2 * s], a[2 * s + 1], cf[2 * s + 1], acc));
            } else
#endif
            {
                acc = rr_unh(y_step_b(a[0], cfs[0], a[1], cfs[1], pBup, HID, ot));
                [unroll] for (uint s = 1; s < RR_KC / 32; ++s)
                    acc = rr_unh(y_step(a[2 * s], cfs[2 * s], a[2 * s + 1], cfs[2 * s + 1], acc));
            }
#else
            [unroll] for (uint s = 0; s < RR_KC / 32; ++s)
                acc = step2(a[2 * s], cf[2 * s], a[2 * s + 1], cf[2 * s + 1], acc);
#endif
            UP((n * HID + 16u * ot + 8u * rr_hf) >> 3) = rr_2q(rr_q8(acc));
        }
#endif
        GroupMemoryBarrierWithGroupSync();
        const uint par = wv * 4u + (uint)(lx >> 1);                     // parent r: coarse row wv, column lx / 2
        const uint ch = 2u * ((uint)Y & 1u) + ((uint)X & 1u);           // child index from the fine coordinates
        [unroll] for (uint ct = 0; ct < KS; ++ct) {
            const uint2 sk = wg_ld64(pX, tok * RC + 16u * ct + 8u * rr_hf);
            const uint2 up = rr_q2(UP((par * HID + ch * RC + 16u * ct + 8u * rr_hf) >> 3));
            xr[ct] = pk_mad(d8h(sk), f2, d8h(up));
            xf[ct] = rr_q8(rr_unh(xr[ct]));
        }
    }
#else
    [unroll] for (uint ks = 0; ks < KS; ++ks) xf[ks] = wg_ld64(pX, tok * RC + 16u * ks + 8u * rr_hf);
#endif

#if RR_STEP == 3
    RrSp xfs[KS];
#if RR_ZFAST
    // one decision for all steps of the block input: on the fast path xfs = (x, 0), so that xf need not stay live
    uint zxs = 0u, zxi = 0u;
#if RR_ZFAST >= 2
    [unroll] for (uint s0 = 0; s0 < KS; ++s0) zxs = zr_acc(zxs, xf[s0]);
    const bool zx = ZG(2, zr_pass(zr_wave(zxs), ZT(c_zt0, 0)));
    YSET(0);
#else
    [unroll] for (uint s0 = 0; s0 < KS; ++s0) { zxs |= z_sbad(xf[s0], ZTK(c_zt0, 0)); zxi |= z_ibad(xf[s0]); }
    const bool zx = ZG(2, z_fast_ok(zxs, zxi, ZIM(1u)));
#endif
#ifdef RR_ZSTAT
    { uint st = 0u; [unroll] for (uint q0 = 0; q0 < KS; ++q0) st = zs_acc2(st, xf[q0]); ZC(1, zs_span(zs_wave(st)) <= ZD(0)); }
#endif
    ZST(2u, zx);
    [unroll] for (uint ks1 = 0; ks1 < KS; ++ks1) { xfs[ks1].h = xf[ks1]; xfs[ks1].l = uint2(0u, 0u); }
    [branch] if (!zx) {
        [unroll] for (uint ks0 = 0; ks0 < KS; ++ks0) xfs[ks0] = z_split(xf[ks0], ZTT(c_zt0, 0));
    }
#else
    [unroll] for (uint ks0 = 0; ks0 < KS; ++ks0) xfs[ks0] = rr_split(xf[ks0]);
#endif
#elif RR_STEP == 4
    RrAB xa[NS];                                             // digit planes of the block input, per step
    [unroll] for (uint s0 = 0; s0 < NS; ++s0) xa[s0] = ri_act(xf[2 * s0], xf[2 * s0 + 1]);
#endif
    // ---- heads: steps 1-6 ------------------------------------------------------------------------------------------
    uint4 dp[KS];                                            // Dp (f16), chain over the heads, start bo
    [unroll] for (uint c0 = 0; c0 < KS; ++c0) dp[c0] = rr_h16(bias_tile(pBo, c0));
#if RR_STEP == 4 && RR_LDSW
    st_commit(st_fetch(pWqv, WUNIT), 0u, WUNIT);
    GroupMemoryBarrierWithGroupSync();
#elif RR_LDSW && RR_STEP != 4
    st_commit(st_fetch(pWqv, UNIT), 0u, UNIT);
    GroupMemoryBarrierWithGroupSync();
#endif
    [loop] for (uint h = 0; h < RH + rr_zero(); ++h) {
#if RR_STEP == 4 && RR_LDSW
        const RrPf pfw = st_fetch(pWo + h * (WUNIT / 2), WUNIT / 2);
#elif RR_LDSW && RR_STEP != 4
        const RrPf pfw = st_fetch(pWo + h * (UNIT / 2), UNIT / 2);
#endif
        // [A] q_h (features x tokens) and v_h^T (tokens x features: the x fragments as A operand)
        v8f aq0 = zero8(), aq1 = zero8(), av0 = zero8(), av1 = zero8();
#if RR_STEP == 3
        // two passes (q, then v^T): two chain pairs live instead of four
        [unroll] for (uint s = 0; s < NS; ++s) {
            const uint2 q00 = qv_frag(h, 2u * s), q01 = qv_frag(h, 2u * s + 1u);
            const uint2 q10 = qv_frag(h, KS + 2u * s), q11 = qv_frag(h, KS + 2u * s + 1u);
#if RR_ZFAST
            [branch] if (zx) {
                if (s == 0) {
                    aq0 = rr_unh(z_fast0(q00, xfs[0].h, q01, xfs[1].h));
                    aq1 = rr_unh(z_fast0(q10, xfs[0].h, q11, xfs[1].h));
                } else {
                    aq0 = rr_unh(z_fast(q00, xfs[2 * s].h, q01, xfs[2 * s + 1].h, aq0));
                    aq1 = rr_unh(z_fast(q10, xfs[2 * s].h, q11, xfs[2 * s + 1].h, aq1));
                }
            } else
#endif
            if (s == 0) {
                aq0 = rr_unh(y_step0(q00, xfs[0], q01, xfs[1]));
                aq1 = rr_unh(y_step0(q10, xfs[0], q11, xfs[1]));
            } else {
                aq0 = rr_unh(y_step(q00, xfs[2 * s], q01, xfs[2 * s + 1], aq0));
                aq1 = rr_unh(y_step(q10, xfs[2 * s], q11, xfs[2 * s + 1], aq1));
            }
        }
        [unroll] for (uint s2 = 0; s2 < NS; ++s2) {
            const uint2 v00 = qv_frag(h, 2u * KS + 2u * s2), v01 = qv_frag(h, 2u * KS + 2u * s2 + 1u);
            const uint2 v10 = qv_frag(h, 3u * KS + 2u * s2), v11 = qv_frag(h, 3u * KS + 2u * s2 + 1u);
#if RR_ZFAST
            [branch] if (zx) {
                if (s2 == 0) {
                    av0 = rr_unh(z_fast0(xfs[0].h, v00, xfs[1].h, v01));
                    av1 = rr_unh(z_fast0(xfs[0].h, v10, xfs[1].h, v11));
                } else {
                    av0 = rr_unh(z_fast(xfs[2 * s2].h, v00, xfs[2 * s2 + 1].h, v01, av0));
                    av1 = rr_unh(z_fast(xfs[2 * s2].h, v10, xfs[2 * s2 + 1].h, v11, av1));
                }
            } else
#endif
            if (s2 == 0) {
                av0 = rr_unh(y_step0t(xfs[0], v00, xfs[1], v01));
                av1 = rr_unh(y_step0t(xfs[0], v10, xfs[1], v11));
            } else {
                av0 = rr_unh(y_stept(xfs[2 * s2], v00, xfs[2 * s2 + 1], v01, av0));
                av1 = rr_unh(y_stept(xfs[2 * s2], v10, xfs[2 * s2 + 1], v11, av1));
            }
        }
#elif RR_STEP == 4
        // q_h = Wq^T x (A = weights, B = x), v_h^T: A = x, B = Wv (two digit planes each, tables per (ft, step))
        [unroll] for (uint s = 0; s < NS; ++s) {
            const uint q00 = 2u * s, q10 = KS + 2u * s;
            const uint2 a00 = qv_fp(q00, 0u), a01 = qv_fp(q00, 1u), a10 = qv_fp(q00 + 1u, 0u), a11 = qv_fp(q00 + 1u, 1u);
            const uint2 b00 = qv_fp(q10, 0u), b01 = qv_fp(q10, 1u), b10 = qv_fp(q10 + 1u, 0u), b11 = qv_fp(q10 + 1u, 1u);
            const uint2 t0 = TQV(h, 0u, 0u, s), t1 = TQV(h, 0u, 1u, s);
            aq0 = rr_unh(ri_step_w(a00, a01, a10, a11, xa[s], aq0, t0));
            aq1 = rr_unh(ri_step_w(b00, b01, b10, b11, xa[s], aq1, t1));
        }
        [unroll] for (uint s2 = 0; s2 < NS; ++s2) {
            const uint v00 = 2u * KS + 2u * s2, v10 = 3u * KS + 2u * s2;
            const uint2 a00 = qv_fp(v00, 0u), a01 = qv_fp(v00, 1u), a10 = qv_fp(v00 + 1u, 0u), a11 = qv_fp(v00 + 1u, 1u);
            const uint2 b00 = qv_fp(v10, 0u), b01 = qv_fp(v10, 1u), b10 = qv_fp(v10 + 1u, 0u), b11 = qv_fp(v10 + 1u, 1u);
            const uint2 t0 = TQV(h, 1u, 0u, s2), t1 = TQV(h, 1u, 1u, s2);
            av0 = rr_unh(ri_step_t(xa[s2], a00, a01, a10, a11, av0, t0));
            av1 = rr_unh(ri_step_t(xa[s2], b00, b01, b10, b11, av1, t1));
        }
#else
        [unroll] for (uint s = 0; s < NS; ++s) {
            const uint2 q00 = qv_frag(h, 2u * s), q01 = qv_frag(h, 2u * s + 1u);
            const uint2 q10 = qv_frag(h, KS + 2u * s), q11 = qv_frag(h, KS + 2u * s + 1u);
            const uint2 v00 = qv_frag(h, 2u * KS + 2u * s), v01 = qv_frag(h, 2u * KS + 2u * s + 1u);
            const uint2 v10 = qv_frag(h, 3u * KS + 2u * s), v11 = qv_frag(h, 3u * KS + 2u * s + 1u);
            aq0 = step2(q00, xf[2 * s], q01, xf[2 * s + 1], aq0);
            aq1 = step2(q10, xf[2 * s], q11, xf[2 * s + 1], aq1);
            av0 = step2(xf[2 * s], v00, xf[2 * s + 1], v01, av0);
            av1 = step2(xf[2 * s], v10, xf[2 * s + 1], v11, av1);
        }
#endif
        const uint2 qf0 = rr_q8(aq0), qf1 = rr_q8(aq1);
        const uint xb = XQ_BUF(h);
#if RR_STEP == 4
        const RrAB qa = ri_act(qf0, qf1);                    // this wave's q block: B of its scores
        rr_xq[xb + (wv * 2u) * 32u + rr_lane] = rr_2q(qf0);
        rr_xq[xb + (wv * 2u + 1u) * 32u + rr_lane] = rr_2q(qf1);
        rr_xq[xb + 256u + (wv * 2u) * 32u + rr_lane] = rr_2q(rr_q8(av0));
        rr_xq[xb + 256u + (wv * 2u + 1u) * 32u + rr_lane] = rr_2q(rr_q8(av1));
#elif RR_STEP == 3
#if RR_ZFAST
        // integer q / v (the int class: 0 or |x| >= 8): split h = x, l = 0 without the conversion work
        const bool zqi = WaveActiveAllTrue(((z_ibad(qf0) | z_ibad(qf1)) & 0x80808080u) == 0u);
        const uint2 vq0 = rr_q8(av0), vq1 = rr_q8(av1);
        const bool zvi = WaveActiveAllTrue(((z_ibad(vq0) | z_ibad(vq1)) & 0x80808080u) == 0u);
        if (WaveIsFirstLane()) ZFL_H((h & 1u) * 4u + wv) = (zqi ? 1u : 0u) | (zvi ? 2u : 0u);
#if RR_ZFAST >= 2
        {
            const uint sq = zr_wave(zr_acc(zr_acc(0u, qf0), qf1)), sv = zr_wave(zr_acc(zr_acc(0u, vq0), vq1));
            if (WaveIsFirstLane()) { rr_zsq[(h & 1u) * 4u + wv] = sq; rr_zsv[(h & 1u) * 4u + wv] = sv; }
        }
#endif
#ifdef RR_ZSTAT
        {
            const uint sq = zs_wave(zs_acc2(zs_acc2(0u, qf0), qf1)), sv = zs_wave(zs_acc2(zs_acc2(0u, vq0), vq1));
            const uint tq = WaveActiveMax((uint)zs_span(zs_tok(zs_acc2(zs_acc2(0u, qf0), qf1))));
            if (WaveIsFirstLane()) { rr_zst[wv] = sq; rr_zst[4u + wv] = sv; rr_zst[12u + wv] = tq; }
        }
#endif
        RrSp qs0, qs1;
        qs0.h = qf0; qs0.l = uint2(0u, 0u); qs1.h = qf1; qs1.l = uint2(0u, 0u);
        [branch] if (!zqi) { qs0 = rr_split(qf0); qs1 = rr_split(qf1); }
#else
        const RrSp qs0 = rr_split(qf0), qs1 = rr_split(qf1);
#endif
        rr_xq[xb + (wv * 2u) * 32u + rr_lane] = rr_2q(qs0.h);
        rr_xq[xb + (wv * 2u + 1u) * 32u + rr_lane] = rr_2q(qs1.h);
        rr_xq[xb + 768u + (wv * 2u) * 32u + rr_lane] = rr_2q(qs0.l);
        rr_xq[xb + 768u + (wv * 2u + 1u) * 32u + rr_lane] = rr_2q(qs1.l);
#else
        rr_xq[xb + (wv * 2u) * 32u + rr_lane] = rr_2q(qf0);
        rr_xq[xb + (wv * 2u + 1u) * 32u + rr_lane] = rr_2q(qf1);
#endif
#if RR_STEP == 3
        {
#if RR_ZFAST
            RrSp v0s, v1s;
            v0s.h = vq0; v0s.l = uint2(0u, 0u); v1s.h = vq1; v1s.l = uint2(0u, 0u);
#if RR_ZFAST < 2                                     // RR_ZFAST 2: v unsplit, the O fallback splits on its load
            [branch] if (!zvi) { v0s = rr_split(vq0); v1s = rr_split(vq1); }
#endif
#else
            const RrSp v0s = rr_split(rr_q8(av0)), v1s = rr_split(rr_q8(av1));
#endif
            rr_xq[xb + 256u + (wv * 2u) * 32u + rr_lane] = rr_2q(v0s.h);
            rr_xq[xb + 256u + (wv * 2u + 1u) * 32u + rr_lane] = rr_2q(v1s.h);
#if RR_ZFAST >= 2
            rr_xq[xb + 512u + (wv * 2u) * 32u + rr_lane] = rr_2q(qf0);         // q unsplit (the keys of the scores)
            rr_xq[xb + 512u + (wv * 2u + 1u) * 32u + rr_lane] = rr_2q(qf1);
#else
            rr_xq[xb + 512u + (wv * 2u) * 32u + rr_lane] = rr_2q(v0s.l);
            rr_xq[xb + 512u + (wv * 2u + 1u) * 32u + rr_lane] = rr_2q(v1s.l);
#endif
        }
#elif RR_STEP != 4
        rr_xq[xb + 256u + (wv * 2u) * 32u + rr_lane] = rr_2q(rr_q8(av0));
        rr_xq[xb + 256u + (wv * 2u + 1u) * 32u + rr_lane] = rr_2q(rr_q8(av1));
#endif
#if RR_STEP == 4
#if RR_LDSW
        st_commit(pfw, 1u, WUNIT / 2);
#endif
        GroupMemoryBarrierWithGroupSync();
#if RR_LDSW
        const RrPf pfn = st_fetch(h + 1u < RH ? pWqv + (h + 1u) * WUNIT : pMlp, WUNIT);
#endif
#else
#if RR_LDSW
        st_commit(pfw, 1u, UNIT / 2);
#endif
        GroupMemoryBarrierWithGroupSync();
#if RR_LDSW
        const RrPf pfn = st_fetch(h + 1u < RH ? pWqv + (h + 1u) * UNIT : pMlp, UNIT);
#endif
#endif

#ifdef RR_ZSTAT
        {   // S: q of this wave x q of wave J (32 Qmax Kmax < 2^(lbq + lbk + 24): span_q + span_k <= 11); O: v of the
            // key half (waves 0-1, 2-3) against D of P
            const int sp = zs_span(rr_zst[wv]);
            [unroll] for (uint J5 = 0; J5 < 4; ++J5) { ZC(2, sp + zs_span(rr_zst[J5]) <= 11); ZH(1, sp + zs_span(rr_zst[J5])); ZH(2, rr_zst[12u + wv] + rr_zst[12u + J5]); }
            ZC(3, zs_span(zs_merge(rr_zst[4], rr_zst[5])) <= ZD(7));
            ZC(3, zs_span(zs_merge(rr_zst[6], rr_zst[7])) <= ZD(7));
        }
#endif
        // [B] scores (key tile J x the wave's queries), start value = the score bias (f32 fragment)
        const uint tb = pTb + (h * 16u + wv * 4u) * 1024u + rr_lane * 32u;
        uint4 wt[4];                                         // attention weights, packed f16, per key tile
#if RR_ZFAST
        const uint zfb = (h & 1u) * 4u;
        const uint zf0 = (uint)ZFL_H(zfb), zf1 = (uint)ZFL_H(zfb + 1u), zf2 = (uint)ZFL_H(zfb + 2u), zf3 = (uint)ZFL_H(zfb + 3u);
        const uint zfo = wv == 0u ? zf0 : wv == 1u ? zf1 : wv == 2u ? zf2 : zf3;
#endif
        [unroll] for (uint J = 0; J < 4; ++J) {
#if RR_STEP == 4
            // keys: wave J's q block as A (its digits and statistics made here: the same block, the same values)
            const RrAB ka = ri_act(rr_q2(rr_xq[xb + (J * 2u) * 32u + rr_lane]), rr_q2(rr_xq[xb + (J * 2u + 1u) * 32u + rr_lane]));
            const uint4 s16 = ri_step_aa(ka, qa, wg_load8f(tb, J * 1024u));
#elif RR_STEP == 3
            RrSp k0, k1;
            k0.h = rr_q2(rr_xq[xb + (J * 2u) * 32u + rr_lane]);
            k1.h = rr_q2(rr_xq[xb + (J * 2u + 1u) * 32u + rr_lane]);
            k0.l = rr_q2(rr_xq[xb + 768u + (J * 2u) * 32u + rr_lane]);
            k1.l = rr_q2(rr_xq[xb + 768u + (J * 2u + 1u) * 32u + rr_lane]);
            uint4 s16;
#if RR_ZFAST
            // int class on both sides: integer products below 448^2, |sum| < 2^23 (H1 on the 2^0 grid)
            const uint zfJ = J == 0 ? zf0 : J == 1 ? zf1 : J == 2 ? zf2 : zf3;
#if RR_ZFAST < 2
            ZST(4u, (zfo & zfJ & 1u) != 0u);
            if ((zfo & zfJ & 1u) == 0u) ZSTN(0x10000u);
#endif
#if RR_ZFAST >= 2
            {
                // one chain on the unsplit operands when 32 Qmax Kmax < 2^(lbq + lbk + 24): bm(q) bm(k) < 2^19; else
                // two chains (k unsplit, q split; zs2_ok, s2_score); else the four-chain step
                const uint2 ku0 = rr_q2(rr_xq[xb + 512u + (J * 2u) * 32u + rr_lane]);
                const uint2 ku1 = rr_q2(rr_xq[xb + 512u + (J * 2u + 1u) * 32u + rr_lane]);
                const uint sqw = rr_zsq[zfb + wv], sqj = rr_zsq[zfb + J];
                const float bq = zr_bm(sqw), bk = zr_bm(sqj);
                bool sok = false;
                ZST(4u, ZG(4, bq * bk < 524288.0f));
                [branch] if (ZG(4, bq * bk < 524288.0f)) {
                    s16 = z_fast(ku0, qf0, ku1, qf1, wg_load8f(tb, J * 1024u));
                    sok = true;
                } else if (ZG(512, zs2_ok(sqw, bk))) {
                    const uint ek = max((0x80u - ((sqj >> 8) & 0xffu)) >> 3, 1u), eq = max((0x80u - ((sqw >> 8) & 0xffu)) >> 3, 1u);
                    s16 = s2_score(ku0, qs0, ku1, qs1, wg_load8f(tb, J * 1024u), ldexp(1.0f, (float)(ek + eq + 4u)), sok);
                }
                ZST(512u, sok);
                [branch] if (!sok)
                    s16 = y_score(k0, qs0, k1, qs1, wg_load8f(tb, RH * 16384u + J * 1024u),
                                  wg_load8f(tb, 2u * RH * 16384u + J * 1024u));
            }
#else
            [branch] if (ZG(4, (zfo & zfJ & 1u) != 0u))
                s16 = z_fast(k0.h, qs0.h, k1.h, qs1.h, wg_load8f(tb, J * 1024u));
            else
            s16 = y_score(k0, qs0, k1, qs1, wg_load8f(tb, RH * 16384u + J * 1024u),
                          wg_load8f(tb, 2u * RH * 16384u + J * 1024u));
#endif
#else
            s16 = y_score(k0, qs0, k1, qs1, wg_load8f(tb, RH * 16384u + J * 1024u),
                          wg_load8f(tb, 2u * RH * 16384u + J * 1024u));
#endif
#else
            const uint2 k0 = rr_q2(rr_xq[xb + (J * 2u) * 32u + rr_lane]);
            const uint2 k1 = rr_q2(rr_xq[xb + (J * 2u + 1u) * 32u + rr_lane]);
            const uint4 s16 = step2h(k0, qf0, k1, qf1, wg_load8f(tb, J * 1024u));
#endif
            wt[J] = uint4(rr_attn_w2(s16.x, wc), rr_attn_w2(s16.y, wc), rr_attn_w2(s16.z, wc), rr_attn_w2(s16.w, wc));
        }
        // step 4: Z, r = R(1 / Z), P = Q8(R(w r))
#if RR_Z == 1
        const uint zb = wg_ld16(pZ, ((wid * RH + h) * 64u + 16u * wv + n) * 2u);
#elif RR_Z == 2
        float hs = 0.0f, ls = 0.0f;                          // exact: w = hi / 8 + lo, both sums exact in f32
        [unroll] for (uint J2 = 0; J2 < 4; ++J2) {
            const uint ww[4] = {wt[J2].x, wt[J2].y, wt[J2].z, wt[J2].w};
            [unroll] for (uint k = 0; k < 4; ++k) {
                [unroll] for (uint e = 0; e < 2; ++e) {
                    const float w = f16tof32(ww[k] >> (16u * e));
                    const float hi = floor(w * 8.0f);
                    hs = hs + hi;
                    ls = ls + mad(hi, -0.125f, w);
                }
            }
        }
        const uint hi_t = (uint)hs + (uint)wg_xor16(hs);
        const float lo_t = ls + wg_xor16(ls);
        const float za = (float)(hi_t & ~1u) * 0.125f, zbb = lo_t + (float)(hi_t & 1u) * 0.125f;
        const float zr = za + zbb, zv = zr - za, ze = (za - (zr - zv)) + (zbb - zv);
        const uint zur = asuint(zr);
        const uint zb = wg_f2h(asfloat((zur & 0x7fffffffu) >= 0x7f800000u ? zur : wg_round_odd(zr, ze)));
#else
        float zs = 0.0f;
        [unroll] for (uint J3 = 0; J3 < 4; ++J3) {
            zs = dot2add(rr_h2(wt[J3].x), half2(1.0, 1.0), zs);
            zs = dot2add(rr_h2(wt[J3].y), half2(1.0, 1.0), zs);
            zs = dot2add(rr_h2(wt[J3].z), half2(1.0, 1.0), zs);
            zs = dot2add(rr_h2(wt[J3].w), half2(1.0, 1.0), zs);
        }
        const uint zb = wg_f2h(zs + wg_xor16(zs));
#endif
        const uint rb = wg_f2h(rcp(f16tof32(zb)));
        const half2 r2 = rr_h2(rb * 0x10001u);
        uint2 pf[4];
        [unroll] for (uint J4 = 0; J4 < 4; ++J4) {
            uint4 p;
            p.x = rr_fence_u(rr_b2(rr_h2(wt[J4].x) * r2));
            p.y = rr_fence_u(rr_b2(rr_h2(wt[J4].y) * r2));
            p.z = rr_fence_u(rr_b2(rr_h2(wt[J4].z) * r2));
            p.w = rr_fence_u(rr_b2(rr_h2(wt[J4].w) * r2));
            pf[J4] = rr_q8_small(rr_unh(p));
#if RR_STEP == 3
            y_check_p(pf[J4]);
#endif
        }
        // step 5: O_h = Q8(MM(P; v_h)), keys 0-31 then 32-63 (A = v^T fragments of the key tiles)
        uint2 of[2];
#if RR_STEP == 4
        const RrAB pa0 = ri_act(pf[0], pf[1]), pa1 = ri_act(pf[2], pf[3]);   // P of keys 0-31, 32-63 (B)
#endif
        [unroll] for (uint ft = 0; ft < 2; ++ft) {
#if RR_STEP == 4
            {
                const uint2 a0 = rr_q2(rr_xq[xb + 256u + (0u + ft) * 32u + rr_lane]);
                const uint2 a1 = rr_q2(rr_xq[xb + 256u + (2u + ft) * 32u + rr_lane]);
                const uint2 a2 = rr_q2(rr_xq[xb + 256u + (4u + ft) * 32u + rr_lane]);
                const uint2 a3 = rr_q2(rr_xq[xb + 256u + (6u + ft) * 32u + rr_lane]);
                const RrAB va0 = ri_act(a0, a1);
                const v8f o1 = rr_unh(ri_step_aa(va0, pa0, zero8()));
                const RrAB va1 = ri_act(a2, a3);
                of[ft] = rr_q8(rr_unh(ri_step_aa(va1, pa1, o1)));
            }
#elif RR_STEP == 3
            RrSp a[4];
            [unroll] for (uint kt = 0; kt < 4; ++kt) {
                a[kt].h = rr_q2(rr_xq[xb + 256u + (2u * kt + ft) * 32u + rr_lane]);
#if RR_ZFAST >= 2
                a[kt].l = uint2(0u, 0u);
#else
                a[kt].l = rr_q2(rr_xq[xb + 512u + (2u * kt + ft) * 32u + rr_lane]);
#endif
            }
#if RR_ZFAST
            // int class v (keys of waves 0-1, then 2-3) with P <= 1 (checked): multiples of 2^-9, sum <= 32 * 448
            v8f o1;
            ZST(8u, (zf0 & zf1 & zf2 & zf3 & 2u) != 0u);
#if RR_ZFAST >= 2
            rr_yb = 64.0f;                                   // Y2 steps of O: P, g = -9
            // v of keys 0-31 (waves 0-1), 32-63 (waves 2-3) against R of P: L1 <= 9/8 (exact / f32 row sums of the
            // attention weights; results file), lsb 2^-9
            [branch] if (ZG(8, zr_bm(zr_merge(rr_zsv[zfb], rr_zsv[zfb + 1u])) < RR_ZO_R))
                o1 = rr_unh(z_fast0(a[0].h, pf[0], a[1].h, pf[1]));
            else o1 = rr_unh(y_step0t(rr_split(a[0].h), pf[0], rr_split(a[1].h), pf[1]));
            [branch] if (ZG(8, zr_bm(zr_merge(rr_zsv[zfb + 2u], rr_zsv[zfb + 3u])) < RR_ZO_R))
                of[ft] = rr_q8(rr_unh(z_fast(a[2].h, pf[2], a[3].h, pf[3], o1)));
            else of[ft] = rr_q8(rr_unh(y_stept(rr_split(a[2].h), pf[2], rr_split(a[3].h), pf[3], o1)));
#else
            [branch] if (ZG(8, (zf0 & zf1 & 2u) != 0u)) o1 = rr_unh(z_fast0(a[0].h, pf[0], a[1].h, pf[1]));
            else o1 = rr_unh(y_step0t(a[0], pf[0], a[1], pf[1]));
            [branch] if (ZG(8, (zf2 & zf3 & 2u) != 0u)) of[ft] = rr_q8(rr_unh(z_fast(a[2].h, pf[2], a[3].h, pf[3], o1)));
            else of[ft] = rr_q8(rr_unh(y_stept(a[2], pf[2], a[3], pf[3], o1)));
#endif
#else
            const v8f o1 = rr_unh(y_step0t(a[0], pf[0], a[1], pf[1]));
            of[ft] = rr_q8(rr_unh(y_stept(a[2], pf[2], a[3], pf[3], o1)));
#endif
#else
            const uint2 a0 = rr_q2(rr_xq[xb + 256u + (0u + ft) * 32u + rr_lane]);
            const uint2 a1 = rr_q2(rr_xq[xb + 256u + (2u + ft) * 32u + rr_lane]);
            const uint2 a2 = rr_q2(rr_xq[xb + 256u + (4u + ft) * 32u + rr_lane]);
            const uint2 a3 = rr_q2(rr_xq[xb + 256u + (6u + ft) * 32u + rr_lane]);
            v8f o = zero8();
            o = step2(a0, pf[0], a1, pf[1], o);
            o = step2(a2, pf[2], a3, pf[3], o);
            of[ft] = rr_q8(o);
#endif
        }
        // step 6: Dp step h (one 32-wide chunk per head)
#ifdef RR_ZSTAT
        ZC(4, zs_span(zs_wave(zs_acc2(zs_acc2(0u, of[0]), of[1]))) <= ZD(1));
        ZH(4, zs_span(zs_wave(zs_acc2(zs_acc2(0u, of[0]), of[1]))) - ZD(1) + 16);
#endif
#if RR_STEP == 4
        {
            const RrAB oa = ri_act(of[0], of[1]);
            [unroll] for (uint ct = 0; ct < KS; ++ct) {
                const uint2 a00 = wo_fp(2u * ct, 0u), a01 = wo_fp(2u * ct, 1u);
                const uint2 a10 = wo_fp(2u * ct + 1u, 0u), a11 = wo_fp(2u * ct + 1u, 1u);
                dp[ct] = ri_step_w(a00, a01, a10, a11, oa, rr_unh(dp[ct]), TWO(h, ct));
            }
        }
#if RR_LDSW
        st_commit(pfn, 0u, WUNIT);
        GroupMemoryBarrierWithGroupSync();
#endif
#else
#if RR_ZFAST
#if RR_ZFAST >= 2
        const bool zfo2 = ZG(16, zr_pass(zr_wave(zr_acc(zr_acc(0u, of[0]), of[1])), ZT(c_zt0, 1)));
        YSET(1);
#else
        const bool zfo2 = ZG(16, z_fast_ok(z_sbad(of[0], ZTK(c_zt0, 1)) | z_sbad(of[1], ZTK(c_zt0, 1)),
                                           z_ibad(of[0]) | z_ibad(of[1]), ZIM(2u)));
#endif
        ZST(16u, zfo2);
        [branch] if (zfo2) {
            [unroll] for (uint ct = 0; ct < KS; ++ct) {
                const uint2 a0 = wo_frag(h, 2u * ct), a1 = wo_frag(h, 2u * ct + 1u);
                dp[ct] = z_fast(a0, of[0], a1, of[1], rr_unh(dp[ct]));
            }
        } else
#endif
        {
#if RR_STEP == 3 && RR_ZFAST
        const RrSp ofs0 = z_split(of[0], ZTT(c_zt0, 1)), ofs1 = z_split(of[1], ZTT(c_zt0, 1));
#elif RR_STEP == 3
        const RrSp ofs0 = rr_split(of[0]), ofs1 = rr_split(of[1]);
#endif
        [unroll] for (uint ct = 0; ct < KS; ++ct) {
            const uint2 a0 = wo_frag(h, 2u * ct), a1 = wo_frag(h, 2u * ct + 1u);
#if RR_STEP == 3
            dp[ct] = y_step(a0, ofs0, a1, ofs1, rr_unh(dp[ct]));
#else
            dp[ct] = step2h(a0, of[0], a1, of[1], rr_unh(dp[ct]));
#endif
        }
        }
#if RR_LDSW
        st_commit(pfn, 0u, UNIT);
        GroupMemoryBarrierWithGroupSync();
#endif
#endif
#if !RR_LDSW && RR_STEP == 3
        GroupMemoryBarrierWithGroupSync();      // RR_XQ1: every wave has read this head's exchange buffer
#endif
    }

    // ---- step 7: first residual -------------------------------------------------------------------------------------
    uint4 y1h[KS];
    uint2 y1q[KS];
    [unroll] for (uint c1 = 0; c1 < KS; ++c1) {
#if RR_KIND == 2
        y1h[c1] = pk_mad(dp[c1], ra2, xr[c1]);
#elif RR_STEP == 3
        {   // the block input from its planes (h + l is exact in f16; fenced against folding into the fma)
            const uint4 bh = d8h(xfs[c1].h), bl = d8h(xfs[c1].l);
            const uint4 base = uint4(rr_fence_u(rr_b2(rr_h2(bh.x) + rr_h2(bl.x))), rr_fence_u(rr_b2(rr_h2(bh.y) + rr_h2(bl.y))),
                                     rr_fence_u(rr_b2(rr_h2(bh.z) + rr_h2(bl.z))), rr_fence_u(rr_b2(rr_h2(bh.w) + rr_h2(bl.w))));
            y1h[c1] = pk_mad(dp[c1], ra2, base);
        }
#else
        y1h[c1] = pk_mad(dp[c1], ra2, d8h(xf[c1]));
#endif
        y1q[c1] = rr_q8(rr_unh(y1h[c1]));
    }

    // ---- step 8: MLP --------------------------------------------------------------------------------------------------
    uint4 dd[KS];
    [unroll] for (uint c2 = 0; c2 < KS; ++c2) dd[c2] = rr_h16(bias_tile(pB2, c2));
#if RR_STEP == 3
    RrSp y1s[KS];
#if RR_ZFAST
    uint z1s = 0u, z1i = 0u;                                 // one decision for all W1 steps (as x)
#if RR_ZFAST >= 2
    [unroll] for (uint s1 = 0; s1 < KS; ++s1) z1s = zr_acc(z1s, y1q[s1]);
    const bool z1 = ZG(32, zr_pass(zr_wave(z1s), ZT(c_zt0, 2)));
    YSET(2);
#else
    [unroll] for (uint s1 = 0; s1 < KS; ++s1) { z1s |= z_sbad(y1q[s1], ZTK(c_zt0, 2)); z1i |= z_ibad(y1q[s1]); }
    const bool z1 = ZG(32, z_fast_ok(z1s, z1i, ZIM(4u)));
#endif
#ifdef RR_ZSTAT
    { uint st = 0u; [unroll] for (uint q0 = 0; q0 < KS; ++q0) st = zs_acc2(st, y1q[q0]); ZC(5, zs_span(zs_wave(st)) <= ZD(2)); }
#endif
    ZST(32u, z1);
    [unroll] for (uint c7 = 0; c7 < KS; ++c7) { y1s[c7].h = y1q[c7]; y1s[c7].l = uint2(0u, 0u); }
    [branch] if (!z1) {
        [unroll] for (uint c6 = 0; c6 < KS; ++c6) y1s[c6] = z_split(y1q[c6], ZTT(c_zt0, 2));
    }
#else
    [unroll] for (uint c6 = 0; c6 < KS; ++c6) y1s[c6] = rr_split(y1q[c6]);
#endif
#endif
#if RR_STEP == 4
    RrAB y1a[NS];
    [unroll] for (uint s1 = 0; s1 < NS; ++s1) y1a[s1] = ri_act(y1q[2 * s1], y1q[2 * s1 + 1]);
    [loop] for (uint j = 0; j < NJ + rr_zero(); ++j) {
#if RR_LDSW
        const RrPf pfm = st_fetch(pMlp + min(j + 1u, NJ - 1u) * WUNIT, WUNIT);
#endif
        v8f u0 = bias_tile(pB1, 2u * j), u1 = bias_tile(pB1, 2u * j + 1u);
        uint4 h0, h1;
        [unroll] for (uint s = 0; s < NS; ++s) {
            const uint f0 = 2u * s, f1 = KS + 2u * s;
            const uint2 a00 = mlp_fp(j, f0, 0u), a01 = mlp_fp(j, f0, 1u), a10 = mlp_fp(j, f0 + 1u, 0u), a11 = mlp_fp(j, f0 + 1u, 1u);
            const uint2 b00 = mlp_fp(j, f1, 0u), b01 = mlp_fp(j, f1, 1u), b10 = mlp_fp(j, f1 + 1u, 0u), b11 = mlp_fp(j, f1 + 1u, 1u);
            const uint2 t0 = TW1(j, 0u, s), t1 = TW1(j, 1u, s);
            h0 = ri_step_w(a00, a01, a10, a11, y1a[s], u0, t0);
            h1 = ri_step_w(b00, b01, b10, b11, y1a[s], u1, t1);
            if (s + 1 < NS) {
                u0 = rr_unh(h0);
                u1 = rr_unh(h1);
            }
        }
        const uint2 au0 = rr_q8(rr_unh(uint4(rr_act2(h0.x, wc), rr_act2(h0.y, wc), rr_act2(h0.z, wc), rr_act2(h0.w, wc))));
        const uint2 au1 = rr_q8(rr_unh(uint4(rr_act2(h1.x, wc), rr_act2(h1.y, wc), rr_act2(h1.z, wc), rr_act2(h1.w, wc))));
        {
            const RrAB aua = ri_act(au0, au1);
            [unroll] for (uint ct = 0; ct < KS; ++ct) {
                const uint f2 = 2u * KS + 2u * ct;
                const uint2 a00 = mlp_fp(j, f2, 0u), a01 = mlp_fp(j, f2, 1u), a10 = mlp_fp(j, f2 + 1u, 0u), a11 = mlp_fp(j, f2 + 1u, 1u);
                dd[ct] = ri_step_w(a00, a01, a10, a11, aua, rr_unh(dd[ct]), TW2(j, ct));
            }
        }
#if RR_LDSW
        if (j + 1u < NJ) st_commit(pfm, (j + 1u) & 1u, WUNIT);
        GroupMemoryBarrierWithGroupSync();
#endif
    }
#else
    [loop] for (uint j = 0; j < NJ + rr_zero(); ++j) {
#if RR_LDSW
        const RrPf pfm = st_fetch(pMlp + min(j + 1u, NJ - 1u) * UNIT, UNIT);
#endif
        v8f u0 = bias_tile(pB1, 2u * j), u1 = bias_tile(pB1, 2u * j + 1u);
        uint4 h0, h1;
        [unroll] for (uint s = 0; s < NS; ++s) {
            const uint2 a00 = mlp_frag(j, 2u * s), a01 = mlp_frag(j, 2u * s + 1u);
            const uint2 a10 = mlp_frag(j, KS + 2u * s), a11 = mlp_frag(j, KS + 2u * s + 1u);
#if RR_STEP == 3
#if RR_ZFAST
            [branch] if (z1) {
                h0 = z_fast(a00, y1s[2 * s].h, a01, y1s[2 * s + 1].h, u0);   // u0 / u1: the b1 values at s = 0
                h1 = z_fast(a10, y1s[2 * s].h, a11, y1s[2 * s + 1].h, u1);
            } else
#endif
            if (s == 0) {
                h0 = y_step_b(a00, y1s[0], a01, y1s[1], pB1, HID, 2u * j);
                h1 = y_step_b(a10, y1s[0], a11, y1s[1], pB1, HID, 2u * j + 1u);
            } else {
                h0 = y_step(a00, y1s[2 * s], a01, y1s[2 * s + 1], u0);
                h1 = y_step(a10, y1s[2 * s], a11, y1s[2 * s + 1], u1);
            }
            if (s + 1 < NS) {
                u0 = rr_unh(h0);
                u1 = rr_unh(h1);
            }
#else
            if (s + 1 < NS) {
                u0 = step2(a00, y1q[2 * s], a01, y1q[2 * s + 1], u0);
                u1 = step2(a10, y1q[2 * s], a11, y1q[2 * s + 1], u1);
            } else {
                h0 = step2h(a00, y1q[2 * s], a01, y1q[2 * s + 1], u0);
                h1 = step2h(a10, y1q[2 * s], a11, y1q[2 * s + 1], u1);
            }
#endif
        }
        const uint2 au0 = rr_q8(rr_unh(uint4(rr_act2(h0.x, wc), rr_act2(h0.y, wc), rr_act2(h0.z, wc), rr_act2(h0.w, wc))));
        const uint2 au1 = rr_q8(rr_unh(uint4(rr_act2(h1.x, wc), rr_act2(h1.y, wc), rr_act2(h1.z, wc), rr_act2(h1.w, wc))));
#ifdef RR_ZSTAT
        ZC(6, zs_span(zs_wave(zs_acc2(zs_acc2(0u, au0), au1))) <= ZD(3));
        ZH(3, zs_span(zs_wave(zs_acc2(zs_acc2(0u, au0), au1))) - ZD(3) + 16);
#endif
#if RR_ZFAST
#if RR_ZFAST >= 2
        const bool zf2 = ZG(64, zr_pass(zr_wave(zr_acc(zr_acc(0u, au0), au1)), ZT(c_zt0, 3)));
        YSET(3);
#else
        const bool zf2 = ZG(64, z_fast_ok(z_sbad(au0, ZTK(c_zt0, 3)) | z_sbad(au1, ZTK(c_zt0, 3)), z_ibad(au0) | z_ibad(au1),
                                          ZIM(8u)));
#endif
        ZST(64u, zf2);
        [branch] if (zf2) {
            [unroll] for (uint ct = 0; ct < KS; ++ct) {
                const uint2 a0 = mlp_frag(j, 2u * KS + 2u * ct), a1 = mlp_frag(j, 2u * KS + 2u * ct + 1u);
                dd[ct] = z_fast(a0, au0, a1, au1, rr_unh(dd[ct]));
            }
        } else
#endif
        {
#if RR_STEP == 3 && RR_ZFAST
        const RrSp aus0 = z_split(au0, ZTT(c_zt0, 3)), aus1 = z_split(au1, ZTT(c_zt0, 3));
#elif RR_STEP == 3
        const RrSp aus0 = rr_split(au0), aus1 = rr_split(au1);
#endif
        [unroll] for (uint ct = 0; ct < KS; ++ct) {
            const uint2 a0 = mlp_frag(j, 2u * KS + 2u * ct), a1 = mlp_frag(j, 2u * KS + 2u * ct + 1u);
#if RR_STEP == 3
            dd[ct] = y_step(a0, aus0, a1, aus1, rr_unh(dd[ct]));
#else
            dd[ct] = step2h(a0, au0, a1, au1, rr_unh(dd[ct]));
#endif
        }
        }
#if RR_LDSW
        if (j + 1u < NJ) st_commit(pfm, (j + 1u) & 1u, UNIT);
        GroupMemoryBarrierWithGroupSync();
#endif
    }
#endif

    // ---- step 9: second residual, block output y2 ---------------------------------------------------------------------
    uint2 y2[KS];
    [unroll] for (uint c3 = 0; c3 < KS; ++c3) y2[c3] = rr_q8(rr_unh(pk_mad(dd[c3], m2, y1h[c3])));
    const uint otok = (uint)(Y * Nx + X);
    if (real) {
        [unroll] for (uint c4 = 0; c4 < KS; ++c4) wg_st64(pY2, otok * RC + 16u * c4 + 8u * rr_hf, y2[c4]);
    }

    // ---- merge (51.4, E1-E4) ----------------------------------------------------------------------------------------
#if RR_CN > 0
    {
        const uint t = 16u * wv + n;
#if RR_STEP == 3
#if RR_ZFAST
        uint zyb = 0u;
        [unroll] for (uint c8 = 0; c8 < KS; ++c8) zyb |= z_ibad(y2[c8]);
        const bool zyi = WaveActiveAllTrue((zyb & 0x80808080u) == 0u);
        if (WaveIsFirstLane()) ZFL_M(wv) = zyi ? 4u : 0u;
#ifdef RR_ZSTAT
        { uint st = 0u; [unroll] for (uint q0 = 0; q0 < KS; ++q0) st = zs_acc2(st, y2[q0]); st = zs_wave(st); if (WaveIsFirstLane()) rr_zst[8u + wv] = st; }
#endif
#endif
#if RR_ZFAST >= 2
        {   // y2 unsplit (the fallback splits on its load), range statistics of the wave
            uint zy = 0u;
            [unroll] for (uint c5 = 0; c5 < KS; ++c5) {
                MY((t * RC + 16u * c5 + 8u * rr_hf) >> 3) = rr_2q(y2[c5]);
                zy = zr_acc(zy, y2[c5]);
            }
            zy = zr_wave(zy);
            if (WaveIsFirstLane()) rr_zsy[wv] = zy;
        }
#else
        [unroll] for (uint c5 = 0; c5 < KS; ++c5) {
            const RrSp ys = rr_split(y2[c5]);
            MY((t * RC + 16u * c5 + 8u * rr_hf) >> 3) = rr_2q(ys.h);
            MY(UNIT / 8 + ((t * RC + 16u * c5 + 8u * rr_hf) >> 3)) = rr_2q(ys.l);
        }
#endif
#else
        [unroll] for (uint c5 = 0; c5 < KS; ++c5) MY((t * RC + 16u * c5 + 8u * rr_hf) >> 3) = rr_2q(y2[c5]);
#endif
        GroupMemoryBarrierWithGroupSync();
        const uint rx = n & 3u, ry = n >> 2;                 // lane n = coarse token 4 ry + rx of the window
        const int Xc = cx0 + (int)rx, Yc = cy0 + (int)ry;
        const bool cin = Xc >= 0 && Xc < Mx && Yc >= 0 && Yc < My;
#if RR_ZFAST
#if RR_ZFAST >= 2
        const bool zfm = ZG(128, zr_pass(zr_merge(zr_merge(rr_zsy[0], rr_zsy[1]), zr_merge(rr_zsy[2], rr_zsy[3])), ZT(c_zt1, 2)));
        YSET(6);
#else
        const bool zfm = ZG(128, (((uint)ZFL_M(0) & (uint)ZFL_M(1) & (uint)ZFL_M(2) & (uint)ZFL_M(3) & 4u) != 0u) && ZIM(64u));
#endif
#ifdef RR_ZSTAT
        ZC(7, zs_span(zs_merge(zs_merge(rr_zst[8], rr_zst[9]), zs_merge(rr_zst[10], rr_zst[11]))) <= ZD(6));
#endif
        ZST(128u, zfm);
#endif
        [loop] for (uint ot = wv; ot < RR_CN / 16 + rr_zero(); ot += 4u) {
            v8f acc = bias_tile(pBm, ot);
            [loop] for (uint s = 0; s < HID / 32 + rr_zero(); ++s) {
#if RR_STEP == 3
                RrSp b[2];
#else
                uint2 b[2];
#endif
                uint2 a[2];
                [unroll] for (uint e = 0; e < 2; ++e) {
                    const uint zi = 32u * s + 16u * e + 8u * rr_hf;   // z index n * C + c
                    const uint cn = zi / RC, cc = zi % RC;
                    const uint tt = 8u * (2u * ry + (cn >> 1)) + 2u * rx + (cn & 1u);
#if RR_STEP == 3
                    b[e].h = rr_q2(MY((tt * RC + cc) >> 3));
#if RR_ZFAST >= 2
                    b[e].l = uint2(0u, 0u);
#else
                    b[e].l = rr_q2(MY(UNIT / 8 + ((tt * RC + cc) >> 3)));
#endif
#else
                    b[e] = rr_q2(MY((tt * RC + cc) >> 3));
#endif
                    a[e] = wfrag(pWm, ot * (HID / 16) + 2u * s + e);
                }
#if RR_STEP == 3
#if RR_ZFAST
                [branch] if (zfm) acc = rr_unh(z_fast(a[0], b[0].h, a[1], b[1].h, acc));   // integer y2: h = y2
                else
#endif
#if RR_ZFAST >= 2
                acc = rr_unh(y_step(a[0], rr_split(b[0].h), a[1], rr_split(b[1].h), acc));
#else
                acc = rr_unh(y_step(a[0], b[0], a[1], b[1], acc));
#endif
#elif RR_STEP == 4
                {
                    const uint f = ot * (HID / 16) + 2u * s;
                    const uint2 a01 = wfrag(pWm + PB_M, f), a11 = wfrag(pWm + PB_M, f + 1u);
                    const uint2 t = wg_ld64(pWm + 2u * PB_M, (ot * (HID / 32) + s) * 8u);
                    acc = rr_unh(ri_step_w(a[0], a01, a[1], a11, ri_act(b[0], b[1]), acc, t));
                }
#else
                acc = step2(a[0], b[0], a[1], b[1], acc);
#endif
            }
            const uint2 mq = rr_q8(acc);
            if (cin && rr_zero() == 0u) wg_st64(pMerged, (uint)(Yc * Mx + Xc) * RR_CN + 16u * ot + 8u * rr_hf, mq);
        }
    }
#endif

    // ---- hidden state (51.4) -----------------------------------------------------------------------------------------
#if HIDDEN
    {
        v8f dh = wg_load8f(pBh, 32u * rr_hf);                 // bh padded to 16 rows
#if RR_STEP == 3
#if RR_ZFAST
        uint zhs = 0u, zhi = 0u;
#if RR_ZFAST >= 2
        [unroll] for (uint c9 = 0; c9 < KS; ++c9) zhs = zr_acc(zhs, y2[c9]);
        const bool zfh = ZG(256, zr_pass(zr_wave(zhs), ZT(c_zt1, 1)));
        YSET(5);
#else
        [unroll] for (uint c9 = 0; c9 < KS; ++c9) { zhs |= z_sbad(y2[c9], ZTK(c_zt1, 1)); zhi |= z_ibad(y2[c9]); }
        const bool zfh = ZG(256, z_fast_ok(zhs, zhi, ZIM(32u)));
#endif
        ZST(256u, zfh);
        [branch] if (zfh) {
            [unroll] for (uint s = 0; s < NS; ++s)
                dh = rr_unh(z_fast(wfrag(pWh, 2u * s), y2[2 * s], wfrag(pWh, 2u * s + 1u), y2[2 * s + 1], dh));
        } else {
            [unroll] for (uint s = 0; s < NS; ++s) {
                const RrSp h0 = z_split(y2[2 * s], ZTT(c_zt1, 1)), h1 = z_split(y2[2 * s + 1], ZTT(c_zt1, 1));
                if (s == 0)
                    dh = rr_unh(y_step_b(wfrag(pWh, 0u), h0, wfrag(pWh, 1u), h1, pBh, 16u, 0u));
                else
                    dh = rr_unh(y_step(wfrag(pWh, 2u * s), h0, wfrag(pWh, 2u * s + 1u), h1, dh));
            }
        }
#else
        [unroll] for (uint s = 0; s < NS; ++s) {
            if (s == 0)
                dh = rr_unh(y_step_b(wfrag(pWh, 0u), rr_split(y2[0]), wfrag(pWh, 1u), rr_split(y2[1]), pBh, 16u, 0u));
            else
                dh = rr_unh(y_step(wfrag(pWh, 2u * s), rr_split(y2[2 * s]), wfrag(pWh, 2u * s + 1u),
                                   rr_split(y2[2 * s + 1]), dh));
        }
#endif
#elif RR_STEP == 4
        [unroll] for (uint s = 0; s < NS; ++s) {
            const uint2 a00 = wfrag(pWh, 2u * s), a01 = wfrag(pWh + PB_H, 2u * s);
            const uint2 a10 = wfrag(pWh, 2u * s + 1u), a11 = wfrag(pWh + PB_H, 2u * s + 1u);
            const uint2 t = wg_ld64(pWh + 2u * PB_H, s * 8u);
            dh = rr_unh(ri_step_w(a00, a01, a10, a11, ri_act(y2[2 * s], y2[2 * s + 1]), dh, t));
        }
#else
        [unroll] for (uint s = 0; s < NS; ++s)
            dh = step2(wfrag(pWh, 2u * s), y2[2 * s], wfrag(pWh, 2u * s + 1u), y2[2 * s + 1], dh);
#endif
        const uint4 dhb = rr_h16(dh);                        // lanes hf = 0: Dh_0..3 in dhb.x, dhb.y
        const uint s01 = wg_ld16(pSig, (dhb.x & 0xffffu) * 2u) | (wg_ld16(pSig, (dhb.x >> 16) * 2u) << 16);
        const uint s23 = wg_ld16(pSig, (dhb.y & 0xffffu) * 2u) | (wg_ld16(pSig, (dhb.y >> 16) * 2u) << 16);
        const uint2 hold = wg_ld64(pHold, tok * 8u);
        uint2 hn;
#if RR_KIND == 0
        // encoder rule: G_j = R(sig(Dh_j) g), hnew_j = R(hold_j G_j), hnew_3 = 0
        const uint G01 = rr_fence_u(rr_b2(rr_h2(s01) * g2)), G23 = rr_fence_u(rr_b2(rr_h2(s23) * g2));
        hn.x = rr_fence_u(rr_b2(rr_h2(hold.x) * rr_h2(G01)));
        hn.y = rr_fence_u(rr_b2(rr_h2(hold.y) * rr_h2(G23))) & 0xffffu;
#else
        // recurrent rule (D4-R .. D1-R)
        const half2 z2 = rr_splat_hi(s23);                   // z = sig(Dh_3)
        const uint omzb = rr_fence_u(rr_b2(half2(1.0, 1.0) - z2));
        const half2 omz = rr_h2(omzb);
        // bilinear coarse hidden u_j (51.4): sample ((X - 1/2) / 2, (Y - 1/2) / 2), corners A, B, Cc, Dc
        const int xm = (X - 1) >> 1, ym = (Y - 1) >> 1;
        const int rxm = rr_reflect(xm, Mx), rxp = rr_reflect(xm + 1, Mx);
        const int rym = rr_reflect(ym, My), ryp = rr_reflect(ym + 1, My);
        const half2 wxp = (X & 1) ? half2(0.25, 0.25) : half2(0.75, 0.75);
        const half2 wyp = (Y & 1) ? half2(0.25, 0.25) : half2(0.75, 0.75);
        const half2 wxm = (X & 1) ? half2(0.75, 0.75) : half2(0.25, 0.25);
        const half2 wym = (Y & 1) ? half2(0.75, 0.75) : half2(0.25, 0.25);
        const uint2 cA = wg_ld64(pHc, (uint)(rym * Mx + rxp) * 8u), cB = wg_ld64(pHc, (uint)(rym * Mx + rxm) * 8u);
        const uint2 cC = wg_ld64(pHc, (uint)(ryp * Mx + rxm) * 8u), cD = wg_ld64(pHc, (uint)(ryp * Mx + rxp) * 8u);
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
        const uint hh23 = rr_fence_u(rr_b2(rr_h2(ug23) * z2)) & 0xffffu;   // hh_3 = 0: hnew_3 = R(hold_3 omz)
        hn.x = rr_fence_u(rr_b2(mad(rr_h2(hold.x), omz, rr_h2(hh01))));
        hn.y = rr_fence_u(rr_b2(mad(rr_h2(hold.y), omz, rr_h2(hh23))));
#endif
        if (real && rr_hf == 0u) wg_st64(pHnew, otok * 8u, hn);
#if RR_HMERGE
        // merged hidden: q_n = R(h_n / 4), add16(add16(q0, q1), add16(q2, q3)); children of coarse token rx = n (n < 4)
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
            wg_st64(pHmerged, (uint)(hYc * Mx + hXc) * 8u, hm);
#endif
    }
#endif
#if RR_STEP == 3
#if RR_ZFAST < 2
    rr_yflag |= rr_ymax < asfloat(c_ythr) ? 0u : 1u;
#endif
#ifdef RR_ZSTAT
    if (WaveActiveAnyTrue(rr_yflag != 0u) && WaveIsFirstLane()) { uint zo_; wg_mem.InterlockedOr(pYflag + wid * 4u, 0x8000u, zo_); }
#else
    if (WaveActiveAnyTrue(rr_yflag != 0u) && WaveIsFirstLane()) wg_st32(pYflag, wid * 4u, 1u);
#endif
#elif RR_STEP == 4
#if defined(RR_IDBG) || defined(RR_ICOV)
    { const uint yb_ = WaveActiveBitOr(rr_yflag); if (yb_ != 0u && WaveIsFirstLane()) { uint o_; wg_mem.InterlockedOr(pYflag + wid * 4u, yb_, o_); } }
#else
    if (WaveActiveAnyTrue(rr_yflag != 0u) && WaveIsFirstLane()) wg_st32(pYflag, wid * 4u, 1u);
#endif
#endif
#if RR_FIXUP
    if (wg_lane_id() == 0u) {
        wg_st32(pYflag, wid * 4u, 0u);
        uint old_count;
        wg_mem.InterlockedAdd(pYflag + wg_grid_size() * 4u, 1u, old_count);   // diagnostics: windows redone
    }
#endif
}
