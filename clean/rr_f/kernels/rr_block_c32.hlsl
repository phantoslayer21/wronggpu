// SPDX-License-Identifier: MIT
// Spec 51 window block for the full-resolution stages IN0 and D0-F (C = 32, H = 2; DESIGN-51.md), specialised from
// rr_block.hlsl with the same arguments, buffer layouts, rounding points and step (rr_step.hlsli, RR_STEP), and the
// same output bits for every input. One workgroup = one 8 x 8 window, 4 waves of 32 lanes; wave wv owns tokens
// 16 wv .. 16 wv + 15 (window rows 2 wv, 2 wv + 1); lane l = 16 hf + n: token n of the wave's tile, half hf of every
// 16-channel fragment.
//
// What differs from rr_block.hlsl (structure only, every value is computed by the same operations):
//   * Q8 with a wave-uniform fast path: the unclamped driver conversion equals Q8 for every finite |x| <= 464, and any
//     other input (|x| > 464, +-inf, NaN) gives a code 0x7f / 0xff, which no Q8 result has; when a wave sees such a
//     code in any of the values of a Q8 group, the whole group is recomputed with the clamping Q8 (rr_q8).
//
// Defines: RR_KIND 0 (IN0: encoder hidden rule + merged hidden) or 2 (D0-F: fusion half, RR_KC = 64), RR_Z, RR_STEP
// as rr_block.hlsl.
#include "wg_platform.hlsli"
#if WG_WAVE == 64
// clean/rr-mp2-level0 L0a: the window block in wave64 (design D of mpeak W64-REPORT section 2). One wave64 = the work
// of two wave32 waves: tile t (0, 1) = the tokens of wave32 wave wv = 2 w + t. Wave64 lane L = l + 32 hh holds, for
// each tile, half hh of wave32 lane l's fragment: fp8 A / B bytes 4 hh .. 4 hh + 3, accumulator elements 4 hh .. 4 hh + 3
// (the v_wmma wave64 layout, HWPEAK section 4). A v8f holds elements 0-3 of tile 0 and 0-3 of tile 1; a uint2 fp8
// fragment dword x = tile 0, y = tile 1; a uint4 of packed f16 x, y = tile 0, z, w = tile 1. Every value is computed
// by the same operation as in wave32 (wave64 v_wmma = wave32 v_wmma bit for bit, mpeak w64 probe; conversions per
// value), so the outputs keep their bits; memory and LDS byte layouts are unchanged.
// wg_platform.hlsli includes the matrix helpers for wave32 only; the wave64 forms (two products / conversions per
// call, one per tile) are defined here.
#include "wg_ags.hlsli"
wg_frag wg_frag_v8f(v8f c) { wg_frag f; [unroll] for (uint i = 0; i < 8; ++i) f.v[i] = asuint(c.v[i]); return f; }
v8f wg_v8f_frag(wg_frag f) { v8f c; [unroll] for (uint i = 0; i < 8; ++i) c.v[i] = asfloat(f.v[i]); return c; }
wg_frag wg_frag2(uint2 a) { wg_frag f; f.v[0] = a.x; f.v[1] = a.y; [unroll] for (uint i = 2; i < 8; ++i) f.v[i] = 0u; return f; }
wg_frag rr_w64f(uint a, uint b, uint c, uint d)   // a wave64 fragment: the first 4 / 2 / 1 dwords are used
{
    wg_frag f;
    f.v[0] = a; f.v[1] = b; f.v[2] = c; f.v[3] = d;
    [unroll] for (uint i = 4; i < 8; ++i) f.v[i] = 0u;
    return f;
}
v8f wg_wmma_fp8(uint2 a, uint2 b, v8f c)   // tile 0: (a.x, b.x, c 0-3), tile 1: (a.y, b.y, c 4-7)
{
    const wg_frag r0 = wg_ags_mma(WG_AGS_FP8_F32, rr_w64f(a.x, 0u, 0u, 0u), rr_w64f(b.x, 0u, 0u, 0u),
                                  rr_w64f(asuint(c.v[0]), asuint(c.v[1]), asuint(c.v[2]), asuint(c.v[3])));
    const wg_frag r1 = wg_ags_mma(WG_AGS_FP8_F32, rr_w64f(a.y, 0u, 0u, 0u), rr_w64f(b.y, 0u, 0u, 0u),
                                  rr_w64f(asuint(c.v[4]), asuint(c.v[5]), asuint(c.v[6]), asuint(c.v[7])));
    v8f d;
    [unroll] for (uint i = 0; i < 4; ++i) { d.v[i] = asfloat(r0.v[i]); d.v[4 + i] = asfloat(r1.v[i]); }
    return d;
}
uint2 wg_pack_q8_nc_ags(v8f d)   // v_cvt_pk_fp8_f32 per value pair, element i -> byte i of its tile's dword
{
    const wg_frag r0 = wg_ags_convert(rr_w64f(asuint(d.v[0]), asuint(d.v[1]), asuint(d.v[2]), asuint(d.v[3])),
                                      WG_AGS_T_F32_ACC, WG_AGS_T_FP8_B);
    const wg_frag r1 = wg_ags_convert(rr_w64f(asuint(d.v[4]), asuint(d.v[5]), asuint(d.v[6]), asuint(d.v[7])),
                                      WG_AGS_T_F32_ACC, WG_AGS_T_FP8_B);
    return uint2(r0.v[0], r1.v[0]);
}
uint4 wg_to_f16_ags(v8f d)       // v_cvt_f16_f32 per value (round to nearest even)
{
    const wg_frag r0 = wg_ags_convert(rr_w64f(asuint(d.v[0]), asuint(d.v[1]), asuint(d.v[2]), asuint(d.v[3])),
                                      WG_AGS_T_F32_ACC, WG_AGS_T_F16_ACC);
    const wg_frag r1 = wg_ags_convert(rr_w64f(asuint(d.v[4]), asuint(d.v[5]), asuint(d.v[6]), asuint(d.v[7])),
                                      WG_AGS_T_F32_ACC, WG_AGS_T_F16_ACC);
    return uint4(r0.v[0], r0.v[1], r1.v[0], r1.v[1]);
}
#define rr_d8 rr_d8_w32   // rr_common.hlsli's wave32 text, renamed; the wave64 rr_d8 follows the includes
#endif

WG_ARGS_BEGIN
    uint pX, pCin, pHold, pHc;
    uint pWqv, pTb, pWo, pBo, pMlp, pB1, pB2, pWm, pBm, pWa, pBa, pWup, pBup, pWh, pBh;
    uint pSig, pZ, pY2, pMerged, pHnew, pHmerged;
    int Nx, Ny, sx, sy, nwx;
    uint c_ra_m, c_f_g;
    uint wc0, wc1, wc2, wc3, wc4;
#if RR_DFLOW
    uint dfCnt, dfSelf, dfQ0, dfQ1, dfQ2;   // clean/rr-f-a14-l01 A1b: dataflow arguments (rr_dflow.hlsli), as rr_block_wide.hlsl
#endif
WG_ARGS_END

// clean/rr-f-a14-l01 A40 b16-rtz-elementwise (L01_B16R, gated opt-in; bit as L01_DEFERMASK): in the relaxed body the
// stage runs the packed-f16 elementwise forms of the exact body (act2, attn_w2, pk_mad, Q8 from f16) on WMMA results
// packed with round-toward-zero (v_cvt_pkrtz) instead of the f32 forms of RR_B32. NOT exact; quality gate.
#define RR_L01BIT (RR_KIND == 0 ? 1 : 2)   // clean/rr-f-a14-l01: this stage's bit in the L01_* masks
#ifndef L01_PFB
#define L01_PFB 0             // clean/rr-f-a14-l01 A7 post-barrier-prefetch: stages (L01 bits) whose score-bias tiles J = 0, 1 are loaded before the q / v barrier
#endif
#define RR_PFB ((L01_PFB & RR_L01BIT) != 0 && RR_WSRC == 1 && !RR_QV1)
v8f rr_sbt3(uint J, v8f a, v8f b, uint tb) { v8f r; if (J == 0u) r = a; else if (J == 1u) r = b; else r = wg_load8f(tb, J * 1024u); return r; }
#ifndef L01_B16R
#define L01_B16R 0
#endif
#if defined(RR_B32) && RR_B32 && ((RR_KIND == 0 ? 1 : 2) & L01_B16R) != 0
#undef RR_B32
#define RR_B32 0
#define RR_B16R 1
#endif
#include "rr_common.hlsli"
#include "rr_step.hlsli"
#if defined(RR_INFOLD) && RR_INFOLD
#include "rr_mlp_step.hlsli"
#endif
#if WG_WAVE == 64
#undef rr_d8
v8f rr_d8(uint2 q)   // e4m3 bytes -> f32 (exact), one conversion per tile
{
    const wg_frag r0 = wg_ags_convert(rr_w64f(q.x, 0u, 0u, 0u), WG_AGS_T_FP8_B, WG_AGS_T_F32_ACC);
    const wg_frag r1 = wg_ags_convert(rr_w64f(q.y, 0u, 0u, 0u), WG_AGS_T_FP8_B, WG_AGS_T_F32_ACC);
    v8f d;
    [unroll] for (uint i = 0; i < 4; ++i) { d.v[i] = asfloat(r0.v[i]); d.v[4 + i] = asfloat(r1.v[i]); }
    return d;
}
#endif

#ifndef RR_KIND
#define RR_KIND 0
#endif
#ifndef RR_KC
#define RR_KC 64
#endif
#ifndef RR_Z
#define RR_Z 0
#endif
#ifndef RR_INFOLD
#define RR_INFOLD 0           // IN0 only: 1 = front-end MLP folded in (variant E), 3 = folded in (variant X, fast path)
#endif
#if RR_INFOLD && RR_KIND != 0
#error "rr_block_c32.hlsl: RR_INFOLD is for IN0 (RR_KIND 0) only"
#endif
#if RR_C != 32 || RR_H != 2
#error "rr_block_c32.hlsl: C = 32, H = 2 only"
#endif

#define RC 32
#define RH 2
#define KS 2                  // 16-channel fragments of a token
#define HID 128               // MLP width
#define NJ 4                  // MLP chunks
#define UNIT (64 * RC)        // bytes of a weight unit: QV_h (Wq_h | Wv_h) or an MLP chunk (W1 | W2); Wo_h is half

#ifndef RR_WSRC
#define RR_WSRC 1             // 1: weight fragments read from memory by every wave (14 KB, cache-resident), 0: staged
#endif
// clean/rr-f-s13x-x-l01 RR_QV1 1: one q / v barrier per window (both heads' projections, Q8 and exchange writes before
// one barrier, then both attentions; each wave re-reads its own q fragments from the exchange buffer). The same
// operations on the same operands; only the order of the Q8 groups changes (both forms give rr_q8's bits).
#ifndef L01_ACC
#define L01_ACC 0             // clean/rr-f-a14-l01: 1 = RR_DEFER / RR_QV1 masks apply to the relaxed body (accurate / fast) too
#endif
#ifndef L01_QV1MASK
#define L01_QV1MASK 0         // exact body kernels with RR_QV1: 1 IN0, 2 D0-F, 4 E1, 8 D1-F, 16 D1-R
#endif
#ifndef RR_QV1
#define RR_QV1 ((RR_Z == 2 || L01_ACC) && RR_WSRC == 1 && ((RR_KIND == 0 ? 1 : 2) & L01_QV1MASK) != 0)
#endif
#if RR_QV1 && RR_WSRC == 0
#error "RR_QV1 needs RR_WSRC 1"
#endif
#if WG_WAVE == 64
// the same LDS bytes as the wave32 form (rr_xq[1024] / rr_up[256] of uint64), addressed per dword: wave32 slot i
// (8 bytes) = dwords 2 i, 2 i + 1; a wave64 lane holds dword 2 i + hh of each slot of its tiles
groupshared uint rr_xq32[2048];
#define XB(h) ((h) * 512u)
#if RR_KIND == 2
groupshared uint rr_up32[16 * HID / 4];
#endif
#elif RR_WSRC == 0
// two unit buffers (QV_h / MLP chunk j even in buffer 0, Wo_h / MLP chunk j odd in buffer 1) and one q / v exchange
// buffer; the fusion up-projections overlay buffer 1 (free until the first Wo unit).
groupshared uint64_t rr_wb[2 * UNIT / 8];
groupshared uint64_t rr_xq[512];
#define XB(h) 0u
#define UP(i) rr_wb[UNIT / 8 + (i)]
#else
// q / v exchange, one buffer per head (no buffer is written twice: one barrier per head, between its writes and reads);
// the window's quantised up-projections (fusion halves) in their own buffer.
groupshared uint64_t rr_xq[1024];
#define XB(h) ((h) * 512u)
#if RR_KIND == 2 && RR_Z == 2 && !RR_QV1
// clean/rr-f-s13x-x-l01 (exact body only; round-11 item 3): the up-projections (16 x HID / 8 = 256 slots) live in head
// 1's exchange buffer. They are written and read before the head loop, and head 1 first writes its buffer after head
// 0's barrier, which every wave reaches only after its last UP read. Same operations and bits; LDS 10240 -> 8192 B.
#define UP(i) rr_xq[512u + (i)]
#else
groupshared uint64_t rr_up[16 * HID / 8];
#define UP(i) rr_up[i]
#endif
#endif
uint2 rr_q2(uint64_t q) { return uint2((uint)q, (uint)(q >> 32)); }
uint64_t rr_2q(uint2 v) { return (uint64_t)v.x | ((uint64_t)v.y << 32); }

static uint rr_lane;   // lane in the wave
static uint rr_hf;     // lane >> 4

#if WG_WAVE == 64
static uint rr_lp;     // wave32 lane whose fragment halves this lane holds (lane & 31)
static uint rr_hh;     // lane >> 5: the half (fp8 bytes 4 hh .., accumulator elements 4 hh ..)
// per-lane offsets, fenced (rr_fence_u: the same value with no known bits) so that the compiler adds them to the base
// once and keeps the constant parts in the instructions' offset fields (with known bits it ORs every constant into
// the lane offset first, one or two VALU per load)
static uint rr_lo;     // lp * 8 + 4 hh: byte of a fragment-ordered 8-byte slot
static uint rr_xo;     // 2 lp + hh: dword of an LDS slot row
static uint rr_t4;     // 8 hf + 4 hh: byte within a 16-channel group of a token / an output tile row
static uint rr_uo, rr_f2;   // D0-F: dword offsets in the up-projection buffer (rr_block_c32 wave64 kernel)
uint wfrag1(uint base, uint f) { return wg_ld32(base + rr_lo, f * 256u); }
uint2 wfrag(uint base, uint f) { const uint w = wfrag1(base, f); return uint2(w, w); }   // shared by both tiles
v8f zero8() { return wg_splat(0.0f); }
v8f rr_dup4(float4 a)
{
    v8f r;
    r.v[0] = a.x; r.v[1] = a.y; r.v[2] = a.z; r.v[3] = a.w; r.v[4] = a.x; r.v[5] = a.y; r.v[6] = a.z; r.v[7] = a.w;
    return r;
}
v8f rr_cat4(float4 a, float4 b)
{
    v8f r;
    r.v[0] = a.x; r.v[1] = a.y; r.v[2] = a.z; r.v[3] = a.w; r.v[4] = b.x; r.v[5] = b.y; r.v[6] = b.z; r.v[7] = b.w;
    return r;
}
v8f bias_tile(uint p, uint tile) { return rr_dup4(wg_ld4f(p + 4u * rr_t4, 64u * tile)); }
v8f bias_tile2(uint p, uint t0, uint t1)   // per tile: output tile t0 for tile 0, t1 for tile 1
{
    return rr_cat4(wg_ld4f(p + 4u * rr_t4, 64u * t0), wg_ld4f(p + 4u * rr_t4, 64u * t1));
}
// per tile: this lane's dword of the 8-byte group at byte o0 (tile 0) / o1 (tile 1) (o = token row + 16 k)
uint2 rr_tld(uint base, uint o0, uint o1) { return uint2(wg_ld32(base + rr_t4, o0), wg_ld32(base + rr_t4, o1)); }
uint rr_xqr(uint r) { return rr_xo + 2u * r; }   // this lane's dword of wave32 LDS slot r + lp (r = the slot row base)
uint2 rr_xqs(uint r) { const uint w = rr_xq32[rr_xqr(r)]; return uint2(w, w); }   // a shared operand from LDS
#else
uint2 wfrag(uint base, uint f) { return wg_ld64(base, f * 256u + rr_lane * 8u); }
v8f zero8() { return wg_splat(0.0f); }
v8f bias_tile(uint p, uint tile) { return wg_load8f(p, (16u * tile + 8u * rr_hf) * 4u); }
#endif

#if RR_WSRC == 0
uint2 qv_frag(uint h, uint f) { return rr_q2(rr_wb[f * 32u + rr_lane]); }
uint2 wo_frag(uint h, uint f) { return rr_q2(rr_wb[UNIT / 8 + f * 32u + rr_lane]); }
uint2 mlp_frag(uint j, uint f) { return rr_q2(rr_wb[(j & 1u) * (UNIT / 8) + f * 32u + rr_lane]); }
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
void st_sync() { GroupMemoryBarrierWithGroupSync(); }
#else
uint2 qv_frag(uint h, uint f) { return wfrag(pWqv + h * UNIT, f); }
uint2 wo_frag(uint h, uint f) { return wfrag(pWo + h * (UNIT / 2), f); }
uint2 mlp_frag(uint j, uint f) { return wfrag(pMlp + j * UNIT, f); }
struct RrPf { uint v; };
RrPf st_fetch(uint src, uint bytes) { RrPf p; p.v = 0u; return p; }
void st_commit(RrPf p, uint buf, uint bytes) {}
void st_sync() {}
#endif

#include "rr_fast.hlsli"
// clean/rr-f-a14-l01 A1b: dataflow hooks (rr_dflow.hlsli, RR_DFLOW 1: kernels staged.*; track E builds and drops the
// barriers). RR_DFLOW 0 (every build today): the macros expand to the plain loads / stores, nothing else changes.
// Wait: wave 0 polls the producer row counters before the window's first load (one group barrier); signal: after the
// window's last store (after the rerun of a deferred window). Producer tensors (pX, pCin, pHold, pHc) are read and the
// window's outputs (pY2, pHnew, pHmerged, pMerged) written through the coherent view.
#ifndef L01_WREV
#if defined(RR_B32) && RR_B32   // a14 default (relaxed body): E1 and D1-R windows in reverse order
#define L01_WREV 20
#else
#define L01_WREV 0
#endif
#endif
#ifndef L01_WREV_DOC
#define L01_WREV_DOC 0            // clean/rr-f-a14-l01 A18 RR_WORDER: stages whose window order is reversed (1 IN0, 2 D0-F, 4 E1, 8 D1-F, 16 D1-R)
#endif
#if ((RR_KIND == 0 ? 1 : 2) & L01_WREV) != 0
#define RR_WBID() (wg_grid_size() - 1u - wg_block_id())   // the same windows, the last one first (each window's work unchanged)
#else
#define RR_WBID() wg_block_id()
#endif
#ifndef RR_DFLOW
#define RR_DFLOW 0
#endif
#if RR_DFLOW
#include "rr_dflow.hlsli"
#define RR_DF_WJ ((int)(RR_WBID() / (uint)nwx))
#define RR_DF_WAIT() rr_df_wait(WaveReadLaneFirst(wg_lane_id() >> 5), RR_DF_WJ, sy, Ny, Ny >> 1, min(Nx, Ny) >= 10)
#define RR_DF_SIGNAL() rr_df_signal(RR_DF_WJ)
#else
#define RR_PLD64(b, o) wg_ld64(b, o)
#define RR_PST64(b, o, v) wg_st64(b, o, v)
#define RR_DF_WAIT()
#define RR_DF_SIGNAL()
#endif
// clean/rr-f-s13x-x-l01 RR_DEFER 1: deferred Q8 decision (fable B1 step A). The window first runs straight through
// with the fast Q8 forms only (no per-group vote or branch); every group still collects its bad codes, into rr_badw.
// After the window, the waves' flags meet in LDS (two barriers; four exchange slots that the rerun does not write
// before its first barrier: head 1's buffer, or head 0's for the fusion kinds whose up-projections come first), and
// a flagged window (any wave saw a code
// 0x7f / 0xff in any group, so any value of the window may be off) reruns the whole window in the original form (the
// per-group checks and slow forms), rewriting every output at the same addresses. The fast forms give rr_q8's bits
// whenever no group sees such a code, so the outputs have the base's bits for every input.
#ifndef L01_DEFERMASK
#define L01_DEFERMASK 31      // exact body kernels with RR_DEFER: 1 IN0, 2 D0-F, 4 E1, 8 D1-F, 16 D1-R
#endif
#ifndef RR_DEFER
#define RR_DEFER ((RR_Z == 2 || L01_ACC) && ((RR_KIND == 0 ? 1 : 2) & L01_DEFERMASK) != 0)
#endif
#if RR_DEFER
#if !RR_STICKY
#error "RR_DEFER needs RR_STICKY 1"
#endif
static uint rr_badw;
#ifndef L01_DEFERNEG
#define L01_DEFERNEG 0        // negative controls (body_ab only): 1 never rerun, 2 the y2 group's codes not checked
#endif
#undef Q8_TRY
#undef Q8_ELSE
#undef Q8_END
#ifndef L01_DEFERFENCE
#define L01_DEFERFENCE 1      // 1: the fast pass keeps one scalar (never taken) branch per group as a scheduling fence
#endif
#if L01_DEFERFENCE
#define Q8_TRY [branch] if (rr_fp ? rr_zero() == 0u : !rr_sat) {
#else
#define Q8_TRY [branch] if (rr_fp || !rr_sat) {
#endif
#define Q8_ELSE if (rr_fp) { if (!(L01_DEFERNEG == 2 && rr_negoff)) rr_badw |= bad; } else { rr_sat = q8_slow(bad); if (!rr_sat) { DBG_C(0) } } }         [branch] if (!rr_fp && rr_sat) { DBG_C(1)
#define Q8_END }
static bool rr_negoff;
#define RR_FPP , const bool rr_fp
#define RR_OPQ | (rr_fp ? 0u : rr_zero())   // the rerun's indices opaque: nothing is shared with the fast pass
#define RR_FPA , rr_fp
#else
#define RR_FPP
#define RR_FPA
#define RR_OPQ
#endif
#ifndef RR_FOLD_B32
#define RR_FOLD_B32 0   // clean/rr-f-s9-kernels: 1 = the relaxed folded MLP (RR_INFOLD with RR_B32; opt-in)
#endif

#if WG_WAVE != 64
#if RR_DEFER
void rr_window(const bool rr_fp)
{
#else
WG_KERNEL(rr_block)
{
    WG_SETUP();
    RR_DF_WAIT();
#endif
    rr_lane = WaveGetLaneIndex() RR_OPQ;
    rr_hf = rr_lane >> 4;
    rr_sat = false;
    rr_nan = false;
    DBG_INIT
    const uint wv = (wg_lane_id() RR_OPQ) >> 5;
    const uint n = rr_lane & 15u;
    const uint wid = RR_WBID() RR_OPQ;
    const int wi = (int)(wid % (uint)nwx), wj = (int)(wid / (uint)nwx);
    const int lx = (int)(n & 7u), ly = (int)(2u * wv + (n >> 3));
    const int X = 8 * wi - sx + lx, Y = 8 * wj - sy + ly;
    const bool r1 = min(Nx, Ny) >= 10;                     // one reflection suffices (rr_fast.hlsli rr_refl)
    const int Xr = rr_refl(X, Nx, r1), Yr = rr_refl(Y, Ny, r1);
    const bool real = X >= 0 && X < Nx && Y >= 0 && Y < Ny;
    const uint tok = (uint)(Yr * Nx + Xr);
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

    // ---- block input x and residual base ----------------------------------------------------------------------------
    uint2 xf[KS];
    rrbv xr[KS];                                            // residual base of step 7 (f16)
#if RR_KIND == 2
    {   // fusion half (51.5): up-projection of the window's 4 x 4 coarse tokens, quantised, to LDS; then
        // xf = R(d8(skip) f + d8(upq)) per fine token, x = Q8(xf), xr = xf.
        const int crx = cx0 + (int)(n & 3u), cry = cy0 + (int)(n >> 2);
        const uint ctok = (uint)(rr_refl(cry, My, r1) * Mx + rr_refl(crx, Mx, r1));
        uint2 cf[RR_KC / 16];
        [unroll] for (uint k = 0; k < RR_KC / 16; ++k) cf[k] = RR_PLD64(pCin, ctok * RR_KC + 16u * k + 8u * rr_hf);
        [unroll] for (uint t2 = 0; t2 < HID / 64; ++t2) {
            const uint ot = wv + 4u * t2;
            uint2 a[RR_KC / 16];
            [unroll] for (uint k2 = 0; k2 < RR_KC / 16; ++k2) a[k2] = wfrag(pWup, ot * (RR_KC / 16) + k2);
            v8f acc = bias_tile(pBup, ot);
            [unroll] for (uint s = 0; s < RR_KC / 32; ++s)
                acc = step2l(s + 1 == RR_KC / 32, a[2 * s], cf[2 * s], a[2 * s + 1], cf[2 * s + 1], acc);
            uint bad = 0u;
            uint2 uq;
            Q8_TRY uq = q8f(acc, bad);
            Q8_ELSE uq = q8s(acc); Q8_END
            UP((n * HID + 16u * ot + 8u * rr_hf) >> 3) = rr_2q(uq);
        }
        GroupMemoryBarrierWithGroupSync();
        const uint par = wv * 4u + (uint)(lx >> 1);
        const uint ch = 2u * ((uint)Y & 1u) + ((uint)X & 1u);
        uint bad = 0u;
        [unroll] for (uint ct = 0; ct < KS; ++ct) {
            const uint2 sk = RR_PLD64(pX, tok * RC + 16u * ct + 8u * rr_hf);
            const uint2 up = rr_q2(UP((par * HID + ch * RC + 16u * ct + 8u * rr_hf) >> 3));
            xr[ct] = bv_mad(bv_d8(sk), BVK(f2), bv_d8(up));
        }
        Q8_TRY [unroll] for (uint c0 = 0; c0 < KS; ++c0) xf[c0] = bv_q8f(xr[c0], bad);
        Q8_ELSE [unroll] for (uint c = 0; c < KS; ++c) xf[c] = bv_q8s(xr[c]); Q8_END
    }
#elif RR_INFOLD
    {   // front-end MLP folded into IN0 (spec 52.5; rr_input_mlp.hlsl, the same operations per token): x = Q8(MM(A_1;
        // W2_1; MM(A_0; W2_0; b2))), A_br = Q8(act(MM(f; W1_br; b1_br))) from the token's 64 input codes. Arguments
        // re-bound by the host: pX = codes [tok][64], pCin = W1T, pWup = W2T, pBup = b1 [2][32] f32, pHc = b2 [32] f32.
        uint2 f[4];
        [unroll] for (uint ks = 0; ks < 4u; ++ks) f[ks] = RR_PLD64(pX, tok * 64u + 16u * ks + 8u * rr_hf);
#if RR_B32 && RR_FOLD_B32
        // clean/rr-f-s9-kernels opt-in RR_FOLD_B32 (relaxed body only; NOT variant E): the folded MLP is
        // rr_input_mlp.hlsl RR_MLP_B32 (the same operations: f32 steps, f32 activation, fast Q8 with the slow path)
        v8f ah[2][2];
        [unroll] for (uint br = 0; br < 2u; ++br) {
            [unroll] for (uint ft = 0; ft < 2u; ++ft) {
                const uint w = (br * 2u + ft) * 4u;
                v8f acc = bias_tile(pBup + br * 128u, ft);
                acc = step2_raw(wfrag(pCin, w + 0u), f[0], wfrag(pCin, w + 1u), f[1], acc);
                acc = step2_raw(wfrag(pCin, w + 2u), f[2], wfrag(pCin, w + 3u), f[3], acc);
                ah[br][ft] = bv_act(acc, wx);
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
            v8f z = bias_tile(pHc, ot);
            z = step2_raw(wfrag(pWup, (0u * 2u + ot) * 2u), a[0][0], wfrag(pWup, (0u * 2u + ot) * 2u + 1u), a[0][1], z);
            o[ot] = step2_raw(wfrag(pWup, (1u * 2u + ot) * 2u), a[1][0], wfrag(pWup, (1u * 2u + ot) * 2u + 1u),
                              a[1][1], z);
        }
        {
            uint bad = 0u;
            Q8_TRY xf[0] = q8f(o[0], bad); xf[1] = q8f(o[1], bad);
            Q8_ELSE xf[0] = q8s(o[0]); xf[1] = q8s(o[1]);
            Q8_END
        }
#elif RR_B32 || RR_Z == 2 || defined(RR_B16R)
        // clean/rr-f-s13x-x-l01: the exact body (RR_Z 2) folds through this form too (it is variant E exactly; before,
        // it ran rr_q8 per value on the fenced activation, the same bits for every input).
        // clean/rr-f-s9-kernels (default of the relaxed-body build): the folded MLP on variant E exactly (rr_input_mlp.hlsl
        // RR_STEP 0: the steps of rr_mlp_step.hlsli, f16 activation), its Q8 through the fast forms of rr_fast.hlsli (act2 +
        // q8hf / q8f with the wave-uniform slow path: rr_q8's bits for every input), so x0 has the bits of input.mlp.e
        uint4 ah[2][2];
#ifndef L01_WILF
#define L01_WILF 0            // clean/rr-f-a14-l01 A5 on the folded front-end MLP (relaxed body): the four first-layer chains and
#endif                        // the two second-layer chains WMMA-interleaved (each chain's WMMAs and f16 roundings unchanged)
#if L01_WILF && RR_B32 && RR_INFOLD == 1
        {
            v8f c[4];   // chain i = (br, ft) = (i >> 1, i & 1)
            [unroll] for (uint i0 = 0; i0 < 4u; ++i0) c[i0] = bias_tile(pBup + (i0 >> 1) * 128u, i0 & 1u);
            [unroll] for (uint i1 = 0; i1 < 4u; ++i1) c[i1] = wg_wmma_fp8(wfrag(pCin, i1 * 4u + 0u), f[0], c[i1]);
            [unroll] for (uint i2 = 0; i2 < 4u; ++i2) c[i2] = wg_wmma_fp8(wfrag(pCin, i2 * 4u + 1u), f[1], c[i2]);
            [unroll] for (uint i3 = 0; i3 < 4u; ++i3) c[i3] = rr_r16(c[i3]);
            [unroll] for (uint i4 = 0; i4 < 4u; ++i4) c[i4] = wg_wmma_fp8(wfrag(pCin, i4 * 4u + 2u), f[2], c[i4]);
            [unroll] for (uint i5 = 0; i5 < 4u; ++i5) c[i5] = wg_wmma_fp8(wfrag(pCin, i5 * 4u + 3u), f[3], c[i5]);
            [unroll] for (uint i6 = 0; i6 < 4u; ++i6) {
                const uint4 h = rr_h16(c[i6]);
                ah[i6 >> 1][i6 & 1u] = uint4(act2(h.x, wc), act2(h.y, wc), act2(h.z, wc), act2(h.w, wc));
            }
        }
#else
        [unroll] for (uint br = 0; br < 2u; ++br) {
            [unroll] for (uint ft = 0; ft < 2u; ++ft) {
                const uint w = (br * 2u + ft) * 4u;
                v8f acc = bias_tile(pBup + br * 128u, ft);
                acc = MLP_STEP2(wfrag(pCin, w + 0u), f[0], wfrag(pCin, w + 1u), f[1], acc);
                const uint4 h = MLP_STEP2H(wfrag(pCin, w + 2u), f[2], wfrag(pCin, w + 3u), f[3], acc);
#if RR_SMAJOR >= 2
                ah[br][ft] = act2_4(h, wc);
#else
                ah[br][ft] = uint4(act2(h.x, wc), act2(h.y, wc), act2(h.z, wc), act2(h.w, wc));
#endif
            }
        }
#endif
        uint2 a[2][2];
        {
            uint bad = 0u;
            Q8_TRY a[0][0] = q8hf(ah[0][0], bad); a[0][1] = q8hf(ah[0][1], bad); a[1][0] = q8hf(ah[1][0], bad);
                   a[1][1] = q8hf(ah[1][1], bad);
            Q8_ELSE a[0][0] = q8h_slow(ah[0][0]); a[0][1] = q8h_slow(ah[0][1]); a[1][0] = q8h_slow(ah[1][0]);
                    a[1][1] = q8h_slow(ah[1][1]);
            Q8_END
        }
        v8f o[2];
#if L01_WILF && RR_B32 && RR_INFOLD == 1
        {
            v8f z[2];
            [unroll] for (uint o0 = 0; o0 < 2u; ++o0) z[o0] = bias_tile(pHc, o0);
            [unroll] for (uint o1 = 0; o1 < 2u; ++o1) z[o1] = wg_wmma_fp8(wfrag(pWup, 2u * o1), a[0][0], z[o1]);
            [unroll] for (uint o2 = 0; o2 < 2u; ++o2) z[o2] = wg_wmma_fp8(wfrag(pWup, 2u * o2 + 1u), a[0][1], z[o2]);
            [unroll] for (uint o3 = 0; o3 < 2u; ++o3) z[o3] = rr_r16(z[o3]);
            [unroll] for (uint o4 = 0; o4 < 2u; ++o4) z[o4] = wg_wmma_fp8(wfrag(pWup, 4u + 2u * o4), a[1][0], z[o4]);
            [unroll] for (uint o5 = 0; o5 < 2u; ++o5) z[o5] = wg_wmma_fp8(wfrag(pWup, 4u + 2u * o5 + 1u), a[1][1], z[o5]);
            [unroll] for (uint o6 = 0; o6 < 2u; ++o6) o[o6] = rr_r16(z[o6]);
        }
#else
        [unroll] for (uint ot = 0; ot < 2u; ++ot) {
            v8f z = bias_tile(pHc, ot);
            z = MLP_STEP2(wfrag(pWup, (0u * 2u + ot) * 2u), a[0][0], wfrag(pWup, (0u * 2u + ot) * 2u + 1u), a[0][1], z);
            o[ot] = MLP_STEP2(wfrag(pWup, (1u * 2u + ot) * 2u), a[1][0], wfrag(pWup, (1u * 2u + ot) * 2u + 1u),
                              a[1][1], z);
        }
#endif
        {
            uint bad = 0u;
            Q8_TRY xf[0] = q8f(o[0], bad); xf[1] = q8f(o[1], bad);
            Q8_ELSE xf[0] = q8s(o[0]); xf[1] = q8s(o[1]);
            Q8_END
        }
#else
        uint2 a[2][2];
        [unroll] for (uint br = 0; br < 2u; ++br) {
            [unroll] for (uint ft = 0; ft < 2u; ++ft) {
                const uint w = (br * 2u + ft) * 4u;
                v8f acc = bias_tile(pBup + br * 128u, ft);
                acc = MLP_STEP2(wfrag(pCin, w + 0u), f[0], wfrag(pCin, w + 1u), f[1], acc);
                const uint4 h = MLP_STEP2H(wfrag(pCin, w + 2u), f[2], wfrag(pCin, w + 3u), f[3], acc);
                a[br][ft] = rr_q8(rr_unh(uint4(rr_act2(h.x, wc), rr_act2(h.y, wc), rr_act2(h.z, wc), rr_act2(h.w, wc))));
            }
        }
        [unroll] for (uint ot = 0; ot < 2u; ++ot) {
            v8f z = bias_tile(pHc, ot);
            z = MLP_STEP2(wfrag(pWup, (0u * 2u + ot) * 2u), a[0][0], wfrag(pWup, (0u * 2u + ot) * 2u + 1u), a[0][1], z);
            const v8f o = MLP_STEP2(wfrag(pWup, (1u * 2u + ot) * 2u), a[1][0], wfrag(pWup, (1u * 2u + ot) * 2u + 1u),
                                    a[1][1], z);
            xf[ot] = rr_q8(o);
        }
#endif
    }
#else
    [unroll] for (uint ks = 0; ks < KS; ++ks) xf[ks] = RR_PLD64(pX, tok * RC + 16u * ks + 8u * rr_hf);
#endif

    // ---- heads: steps 1-6 (unrolled; Dp chain kept as the exact f32 widening of its f16 steps) -----------------------
    v8f dpv[KS];
    rrbv dp[KS];
    [unroll] for (uint c0 = 0; c0 < KS; ++c0) dpv[c0] = bias_tile(pBo, c0);
    st_commit(st_fetch(pWqv, UNIT), 0u, UNIT);
    st_sync();
#if RR_QV1
    [unroll] for (uint hq = 0; hq < RH; ++hq) {   // [A] of both heads, one barrier
        const uint h = hq;
        const v8f aq0 = step2l(true, qv_frag(h, 0u), xf[0], qv_frag(h, 1u), xf[1], zero8());
        const v8f aq1 = step2l(true, qv_frag(h, KS), xf[0], qv_frag(h, KS + 1u), xf[1], zero8());
        const v8f av0 = step2l(true, xf[0], qv_frag(h, 2u * KS), xf[1], qv_frag(h, 2u * KS + 1u), zero8());
        const v8f av1 = step2l(true, xf[0], qv_frag(h, 3u * KS), xf[1], qv_frag(h, 3u * KS + 1u), zero8());
        uint bad = 0u;
        uint2 qf0, qf1, vf0, vf1;
        Q8_TRY qf0 = q8f(aq0, bad); qf1 = q8f(aq1, bad); vf0 = q8f(av0, bad); vf1 = q8f(av1, bad);
        Q8_ELSE qf0 = q8s(aq0); qf1 = q8s(aq1); vf0 = q8s(av0); vf1 = q8s(av1); Q8_END
        rr_xq[XB(h) + (wv * 2u) * 32u + rr_lane] = rr_2q(qf0);
        rr_xq[XB(h) + (wv * 2u + 1u) * 32u + rr_lane] = rr_2q(qf1);
        rr_xq[XB(h) + 256u + (wv * 2u) * 32u + rr_lane] = rr_2q(vf0);
        rr_xq[XB(h) + 256u + (wv * 2u + 1u) * 32u + rr_lane] = rr_2q(vf1);
    }
    GroupMemoryBarrierWithGroupSync();
#endif
    [unroll] for (uint h = 0; h < RH; ++h) {
#if RR_QV1
        uint bad = 0u;
        const uint2 qf0 = rr_q2(rr_xq[XB(h) + (wv * 2u) * 32u + rr_lane]);
        const uint2 qf1 = rr_q2(rr_xq[XB(h) + (wv * 2u + 1u) * 32u + rr_lane]);
#else
        const RrPf pfw = st_fetch(pWo + h * (UNIT / 2), UNIT / 2);
        // [A] q_h (features x tokens) and v_h^T (tokens x features)
#if RR_WIL
        v8f aq0 = zero8(), aq1 = zero8(), av0 = zero8(), av1 = zero8();
        wil4(aq0, aq1, av0, av1, qv_frag(h, 0u), xf[0], qv_frag(h, 1u), xf[1], qv_frag(h, KS), xf[0], qv_frag(h, KS + 1u), xf[1],
             xf[0], qv_frag(h, 2u * KS), xf[1], qv_frag(h, 2u * KS + 1u), xf[0], qv_frag(h, 3u * KS), xf[1], qv_frag(h, 3u * KS + 1u));
#else
        const v8f aq0 = step2l(true, qv_frag(h, 0u), xf[0], qv_frag(h, 1u), xf[1], zero8());
        const v8f aq1 = step2l(true, qv_frag(h, KS), xf[0], qv_frag(h, KS + 1u), xf[1], zero8());
        const v8f av0 = step2l(true, xf[0], qv_frag(h, 2u * KS), xf[1], qv_frag(h, 2u * KS + 1u), zero8());
        const v8f av1 = step2l(true, xf[0], qv_frag(h, 3u * KS), xf[1], qv_frag(h, 3u * KS + 1u), zero8());
#endif
        uint bad = 0u;
        uint2 qf0, qf1, vf0, vf1;
        Q8_TRY qf0 = q8f(aq0, bad); qf1 = q8f(aq1, bad); vf0 = q8f(av0, bad); vf1 = q8f(av1, bad);
        Q8_ELSE qf0 = q8s(aq0); qf1 = q8s(aq1); vf0 = q8s(av0); vf1 = q8s(av1); Q8_END
        rr_xq[XB(h) + (wv * 2u) * 32u + rr_lane] = rr_2q(qf0);
        rr_xq[XB(h) + (wv * 2u + 1u) * 32u + rr_lane] = rr_2q(qf1);
        rr_xq[XB(h) + 256u + (wv * 2u) * 32u + rr_lane] = rr_2q(vf0);
        rr_xq[XB(h) + 256u + (wv * 2u + 1u) * 32u + rr_lane] = rr_2q(vf1);
        st_commit(pfw, 1u, UNIT / 2);
#if RR_PFB
        // the first two score-bias tiles in flight across the barrier (the same loads, issued earlier)
        const uint tbp = pTb + (h * 16u + wv * 4u) * 1024u + rr_lane * 32u;
        const v8f sbp0 = wg_load8f(tbp, 0u), sbp1 = wg_load8f(tbp, 1024u);
#define RR_SBT(J_) rr_sbt3((J_), sbp0, sbp1, tb)
#else
#define RR_SBT(J_) wg_load8f(tb, (J_) * 1024u)
#endif
        GroupMemoryBarrierWithGroupSync();
        const RrPf pfn = st_fetch(h + 1u < RH ? pWqv + (h + 1u) * UNIT : pMlp, UNIT);
#endif

        // [B] scores (key tile J x the wave's queries), start value = the score bias (f32 fragment)
#ifndef RR_SBT
#define RR_SBT(J_) wg_load8f(tb, (J_) * 1024u)
#endif
        const uint tb = pTb + (h * 16u + wv * 4u) * 1024u + rr_lane * 32u;
        rrbv wt[4];
#if RR_WIL
        {
            uint2 k[8];
            [unroll] for (uint i8 = 0; i8 < 8; ++i8) k[i8] = rr_q2(rr_xq[XB(h) + i8 * 32u + rr_lane]);
            v8f s0 = RR_SBT(0u), s1 = RR_SBT(1u), s2 = RR_SBT(2u), s3 = RR_SBT(3u);
            wil4(s0, s1, s2, s3, k[0], qf0, k[1], qf1, k[2], qf0, k[3], qf1, k[4], qf0, k[5], qf1, k[6], qf0, k[7], qf1);
            wt[0] = bv_attn(s0, BVW); wt[1] = bv_attn(s1, BVW); wt[2] = bv_attn(s2, BVW); wt[3] = bv_attn(s3, BVW);
        }
#else
        [unroll] for (uint J = 0; J < 4; ++J) {
            const uint2 k0 = rr_q2(rr_xq[XB(h) + (J * 2u) * 32u + rr_lane]);
            const uint2 k1 = rr_q2(rr_xq[XB(h) + (J * 2u + 1u) * 32u + rr_lane]);
            const rrbv s16 = bv_step(k0, qf0, k1, qf1, RR_SBT(J));
            wt[J] = bv_attn(s16, BVW);
        }
#endif
        // step 4: Z, r = R(1 / Z), P = Q8(R(w r))
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
        const uint rb = f2h_hw(rcp(f16tof32(zb)));
        const half2 r2 = rr_h2(rb * 0x10001u);
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
        // step 5: O_h = Q8(MM(P; v_h)), keys 0-31 then 32-63
        v8f ov[2];
#if RR_WIL
        {
            uint2 vt[8];
            [unroll] for (uint i8 = 0; i8 < 8; ++i8) vt[i8] = rr_q2(rr_xq[XB(h) + 256u + i8 * 32u + rr_lane]);
            ov[0] = zero8(); ov[1] = zero8();
            wil2(ov[0], ov[1], vt[0], pf[0], vt[2], pf[1], vt[1], pf[0], vt[3], pf[1]);
            wil2(ov[0], ov[1], vt[4], pf[2], vt[6], pf[3], vt[5], pf[2], vt[7], pf[3]);
        }
#else
        [unroll] for (uint ft = 0; ft < 2; ++ft) {
            const uint2 a0 = rr_q2(rr_xq[XB(h) + 256u + (0u + ft) * 32u + rr_lane]);
            const uint2 a1 = rr_q2(rr_xq[XB(h) + 256u + (2u + ft) * 32u + rr_lane]);
            const uint2 a2 = rr_q2(rr_xq[XB(h) + 256u + (4u + ft) * 32u + rr_lane]);
            const uint2 a3 = rr_q2(rr_xq[XB(h) + 256u + (6u + ft) * 32u + rr_lane]);
            ov[ft] = step2l(true, a2, pf[2], a3, pf[3], step2(a0, pf[0], a1, pf[1], zero8()));
        }
#endif
        bad = 0u;
        uint2 of[2];
        Q8_TRY of[0] = q8f(ov[0], bad); of[1] = q8f(ov[1], bad);
        Q8_ELSE of[0] = q8s(ov[0]); of[1] = q8s(ov[1]); Q8_END
        // step 6: Dp step h (the last one as packed f16)
#if RR_WIL
        wil2(dpv[0], dpv[1], wo_frag(h, 0u), of[0], wo_frag(h, 1u), of[1], wo_frag(h, 2u), of[0], wo_frag(h, 3u), of[1]);
        if (h + 1u == RH) { dp[0] = dpv[0]; dp[1] = dpv[1]; }
#else
        [unroll] for (uint ct = 0; ct < KS; ++ct) {
            if (h + 1u < RH) dpv[ct] = step2(wo_frag(h, 2u * ct), of[0], wo_frag(h, 2u * ct + 1u), of[1], dpv[ct]);
            else dp[ct] = bv_step(wo_frag(h, 2u * ct), of[0], wo_frag(h, 2u * ct + 1u), of[1], dpv[ct]);
        }
#endif
#if !RR_QV1
        st_commit(pfn, 0u, UNIT);
        st_sync();
#endif
    }

    // ---- step 7: first residual -------------------------------------------------------------------------------------
    rrbv y1h[KS];
    uint2 y1q[KS];
    {
        uint bad = 0u;
        [unroll] for (uint c1 = 0; c1 < KS; ++c1) {
#if RR_KIND == 2
            y1h[c1] = bv_mad(dp[c1], BVK(ra2), xr[c1]);
#else
            y1h[c1] = bv_mad(dp[c1], BVK(ra2), bv_d8(xf[c1]));
#endif
        }
        Q8_TRY [unroll] for (uint c0 = 0; c0 < KS; ++c0) y1q[c0] = bv_q8f(y1h[c0], bad);
        Q8_ELSE [unroll] for (uint c = 0; c < KS; ++c) y1q[c] = bv_q8s(y1h[c]); Q8_END
    }

    // ---- step 8: MLP (unrolled; Dd chain as f32 widening, the last step packed) ---------------------------------------
    v8f ddv[KS];
    rrbv dd[KS];
    [unroll] for (uint c2 = 0; c2 < KS; ++c2) ddv[c2] = bias_tile(pB2, c2);
#ifndef L01_MLA
#define L01_MLA 0             // clean/rr-f-a14-l01 A7 mlp-lookahead: stages (L01 bits) that issue chunk j+1's hidden WMMAs before chunk j's activation / Q8 / Dd
#endif
#define RR_MLA ((L01_MLA & RR_L01BIT) != 0 && RR_WIL && RR_WSRC == 1)
#if RR_MLA
    rrbv hl0, hl1;
#endif
    [unroll] for (uint j = 0; j < NJ; ++j) {
        const RrPf pfm = st_fetch(pMlp + min(j + 1u, NJ - 1u) * UNIT, UNIT);
#if RR_MLA
        // chunk j's hidden pair was issued during chunk j - 1; chunk j + 1's goes out before chunk j's VALU (same steps)
        if (j == 0u) {
            hl0 = bias_tile(pB1, 0u); hl1 = bias_tile(pB1, 1u);
            wil2(hl0, hl1, mlp_frag(0u, 0u), y1q[0], mlp_frag(0u, 1u), y1q[1], mlp_frag(0u, KS), y1q[0], mlp_frag(0u, KS + 1u), y1q[1]);
        }
        const rrbv h0 = hl0, h1 = hl1;
        if (j + 1u < NJ) {
            hl0 = bias_tile(pB1, 2u * (j + 1u)); hl1 = bias_tile(pB1, 2u * (j + 1u) + 1u);
            wil2(hl0, hl1, mlp_frag(j + 1u, 0u), y1q[0], mlp_frag(j + 1u, 1u), y1q[1], mlp_frag(j + 1u, KS), y1q[0],
                 mlp_frag(j + 1u, KS + 1u), y1q[1]);
        }
#elif RR_WIL
        rrbv h0 = bias_tile(pB1, 2u * j), h1 = bias_tile(pB1, 2u * j + 1u);
        wil2(h0, h1, mlp_frag(j, 0u), y1q[0], mlp_frag(j, 1u), y1q[1], mlp_frag(j, KS), y1q[0], mlp_frag(j, KS + 1u), y1q[1]);
#else
        const rrbv h0 = bv_step(mlp_frag(j, 0u), y1q[0], mlp_frag(j, 1u), y1q[1], bias_tile(pB1, 2u * j));
        const rrbv h1 = bv_step(mlp_frag(j, KS), y1q[0], mlp_frag(j, KS + 1u), y1q[1], bias_tile(pB1, 2u * j + 1u));
#endif
        const rrbv a0h = bv_act(h0, BVW);
        const rrbv a1h = bv_act(h1, BVW);
        uint bad = 0u;
        uint2 au0, au1;
        Q8_TRY au0 = bv_q8f(a0h, bad); au1 = bv_q8f(a1h, bad);
        Q8_ELSE au0 = bv_q8s(a0h); au1 = bv_q8s(a1h); Q8_END
#if RR_WIL
        wil2(ddv[0], ddv[1], mlp_frag(j, 2u * KS), au0, mlp_frag(j, 2u * KS + 1u), au1, mlp_frag(j, 2u * KS + 2u), au0,
             mlp_frag(j, 2u * KS + 3u), au1);
        if (j + 1u == NJ) { dd[0] = ddv[0]; dd[1] = ddv[1]; }
#else
        [unroll] for (uint ct = 0; ct < KS; ++ct) {
            const uint2 w0 = mlp_frag(j, 2u * KS + 2u * ct), w1 = mlp_frag(j, 2u * KS + 2u * ct + 1u);
            if (j + 1u < NJ) ddv[ct] = step2(w0, au0, w1, au1, ddv[ct]);
            else dd[ct] = bv_step(w0, au0, w1, au1, ddv[ct]);
        }
#endif
        if (j + 1u < NJ) st_commit(pfm, (j + 1u) & 1u, UNIT);
        st_sync();
    }

    // ---- step 9: second residual, block output y2 ---------------------------------------------------------------------
    uint2 y2[KS];
#if RR_DEFER
    rr_negoff = true;
#endif
    {
        rrbv yh[KS];
        uint bad = 0u;
        [unroll] for (uint c3 = 0; c3 < KS; ++c3) {
            yh[c3] = bv_mad(dd[c3], BVK(m2), y1h[c3]);
        }
        Q8_TRY [unroll] for (uint c0 = 0; c0 < KS; ++c0) y2[c0] = bv_q8f(yh[c0], bad);
        Q8_ELSE [unroll] for (uint c = 0; c < KS; ++c) y2[c] = bv_q8s(yh[c]); Q8_END
    }
    const uint otok = (uint)(Y * Nx + X);
    if (real) {
        [unroll] for (uint c4 = 0; c4 < KS; ++c4) RR_PST64(pY2, otok * RC + 16u * c4 + 8u * rr_hf, y2[c4]);
    }

#if RR_KIND == 0
    // ---- hidden state (51.4), encoder rule, and the merged hidden -----------------------------------------------------
    {
        const uint4 dhb = step2h(wfrag(pWh, 0u), y2[0], wfrag(pWh, 1u), y2[1], wg_load8f(pBh, 32u * rr_hf));
        const uint s01 = wg_ld16(pSig, (dhb.x & 0xffffu) * 2u) | (wg_ld16(pSig, (dhb.x >> 16) * 2u) << 16);
        const uint s23 = wg_ld16(pSig, (dhb.y & 0xffffu) * 2u) | (wg_ld16(pSig, (dhb.y >> 16) * 2u) << 16);
        const uint2 hold = RR_PLD64(pHold, tok * 8u);
        uint2 hn;
        const uint G01 = rr_fence_u(rr_b2(rr_h2(s01) * g2)), G23 = rr_fence_u(rr_b2(rr_h2(s23) * g2));
        hn.x = rr_fence_u(rr_b2(rr_h2(hold.x) * rr_h2(G01)));
        hn.y = rr_fence_u(rr_b2(rr_h2(hold.y) * rr_h2(G23))) & 0xffffu;
        if (real && rr_hf == 0u) RR_PST64(pHnew, otok * 8u, hn);
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
    }
#endif
#if !RR_DEFER
    RR_DF_SIGNAL();
#endif
    DBG_FLUSH
}
#if RR_DEFER
WG_KERNEL(rr_block)
{
    WG_SETUP();
    RR_DF_WAIT();
    rr_badw = 0u;
    rr_negoff = false;
    rr_window(true);
    const uint fb = (RR_KIND == 2 ? 0u : 512u);
    const bool wbad = WaveActiveAnyTrue((rr_badw & 0x80808080u) != 0u);
    GroupMemoryBarrierWithGroupSync();                 // every wave's last exchange read is done
    if (WaveIsFirstLane()) rr_xq[fb + (wg_lane_id() >> 5)] = wbad ? 1u : 0u;
    GroupMemoryBarrierWithGroupSync();
    const bool any = (rr_xq[fb] | rr_xq[fb + 1u] | rr_xq[fb + 2u] | rr_xq[fb + 3u]) != 0u;
    [branch] if (any && L01_DEFERNEG != 1) rr_window(false);
    RR_DF_SIGNAL();
}
#endif
#else   // WG_WAVE == 64
// clean/rr-mp2-level0 L0a: the wave64 window block (design D, see the head of this file). Only the configuration of the
// relaxed body's stagee kernels: RR_B32 (accurate / fast), weights from memory (RR_WSRC 1), WMMA-interleaved chains
// (RR_WIL), no RR_QV1 / RR_DEFER / RR_DFLOW / A7 forms; IN0 with the folded front-end MLP of variant E (RR_INFOLD 1) or
// D0-F (RR_KIND 2). The text follows the wave32 path of that configuration operation by operation; the differences:
//   * per-token quantities (coordinates, token loads / stores, score bias, the hidden-state tail) once per tile;
//   * shared operands (weights, biases, the k / v fragments of the exchange, the coarse fragments of D0-F) loaded once,
//     4 bytes per lane, and used by both tiles' v_wmma (wg_wmma_fp8 above);
//   * the softmax sum Z: wave32 lane l forms zs = A + B with A = ((z0 + z1) + (z2 + z3)) and B = ((z4 + z5) + (z6 + z7))
//     of its 8 elements; here lane l + 32 hh holds elements 4 hh .. 4 hh + 3, forms the bracket of its half and adds
//     the other half's bracket (lane ^ 32): A + B in the hh = 0 lane and B + A in the hh = 1 lane. IEEE addition is
//     commutative, so the sums are equal unless the result is a NaN (payload / sign may then differ); rz = rcp(..) only
//     feeds w * rz -> rr_q8_small (v_cvt_pk_fp8_f32), which gives 0xff for every NaN whatever its sign and payload,
//     so the P codes, and everything after them, have the wave32 bits for every input;
//   * the Q8 votes (WaveActiveAnyTrue) cover both tiles: both forms of a Q8 group give rr_q8's bits for every input,
//     so which one runs is a speed choice only (as in wave32).
#if !RR_B32 || RR_WSRC != 1 || !RR_WIL || RR_QV1 || RR_DEFER || RR_DFLOW || RR_PFB || RR_MLA || RR_FOLD_B32 || !RR_STICKY
#error "rr_block_c32.hlsl wave64: the relaxed stagee configuration only"
#endif
#if RR_KIND == 0 && RR_INFOLD != 1
#error "rr_block_c32.hlsl wave64: IN0 with the folded variant-E front-end MLP only"
#endif
#if RR_KIND == 0 && (L01_WILF || RR_SMAJOR >= 2)
#error "rr_block_c32.hlsl wave64: L01_WILF / RR_SMAJOR 2 forms not ported"
#endif
WG_KERNEL(rr_block)
{
    WG_SETUP();
    rr_lane = WaveGetLaneIndex();
    rr_lp = rr_lane & 31u;
    rr_hh = rr_lane >> 5;
    rr_hf = (rr_lane >> 4) & 1u;
    rr_lo = rr_fence_u(rr_lp * 8u + 4u * rr_hh);
    rr_xo = rr_fence_u(rr_lp * 2u + rr_hh);
    rr_t4 = rr_fence_u(8u * rr_hf + 4u * rr_hh);
    rr_sat = false;
    rr_nan = false;
    DBG_INIT
    const uint wv0 = 2u * WaveReadLaneFirst(wg_lane_id() >> 6);   // wave32 wave of tile 0 (tile 1: wv0 + 1)
    const uint n = rr_lane & 15u;
#if RR_KIND == 2
    rr_uo = rr_fence_u(32u * n + 2u * rr_hf + rr_hh);   // dword of UP slot ((n HID + 16 ot + 8 hf) >> 3) - 4 ot
    rr_f2 = rr_fence_u(2u * rr_hf + rr_hh);
#endif
    const uint wid = RR_WBID();
    const int wi = (int)(wid % (uint)nwx), wj = (int)(wid / (uint)nwx);
    const int lx = (int)(n & 7u);
    const int X = 8 * wi - sx + lx;
    const bool r1 = min(Nx, Ny) >= 10;                     // one reflection suffices (rr_fast.hlsli rr_refl)
    const int Xr = rr_refl(X, Nx, r1);
    int Yt[2];
    uint tok[2];
    bool real[2];
    [unroll] for (uint t0 = 0; t0 < 2u; ++t0) {
        const int ly = (int)(2u * (wv0 + t0) + (n >> 3));
        Yt[t0] = 8 * wj - sy + ly;
        const int Yr = rr_refl(Yt[t0], Ny, r1);
        real[t0] = X >= 0 && X < Nx && Yt[t0] >= 0 && Yt[t0] < Ny;
        tok[t0] = (uint)(Yr * Nx + Xr);
    }
    const RrWc wc = rr_wc(wc0, wc1, wc2, wc3, wc4);
    const half2 ra2 = rr_splat_lo(c_ra_m), m2 = rr_splat_hi(c_ra_m);
    const half2 f2 = rr_splat_lo(c_f_g), g2 = rr_splat_hi(c_f_g);
    const RrWx wx = rr_wx(wc);
    const float ra2x = (float)ra2.x, m2x = (float)m2.x, f2x = (float)f2.x;
    const int Mx = Nx >> 1, My = Ny >> 1;
    const int cx0 = (8 * wi - sx) >> 1, cy0 = (8 * wj - sy) >> 1;

    // ---- block input x and residual base ----------------------------------------------------------------------------
    uint2 xf[KS];
    rrbv xr[KS];
#if RR_KIND == 2
    {   // fusion half (51.5): the window's up-projection (8 output tiles of 16 x 16 coarse tokens; wave32 wave wv
        // computes tiles wv and wv + 4) to LDS; then xf = R(d8(skip) f + d8(upq)) per fine token, x = Q8(xf), xr = xf.
        const int crx = cx0 + (int)(n & 3u), cry = cy0 + (int)(n >> 2);
        const uint ctok = (uint)(rr_refl(cry, My, r1) * Mx + rr_refl(crx, Mx, r1));
        uint2 cf[RR_KC / 16];
        [unroll] for (uint k = 0; k < RR_KC / 16; ++k) {
            const uint c = wg_ld32(pCin + rr_t4, ctok * RR_KC + 16u * k);
            cf[k] = uint2(c, c);
        }
        [unroll] for (uint t2 = 0; t2 < HID / 64; ++t2) {
            const uint ot0 = wv0 + 4u * t2, ot1 = ot0 + 1u;
            uint2 a[RR_KC / 16];
            [unroll] for (uint k2 = 0; k2 < RR_KC / 16; ++k2)
                a[k2] = uint2(wfrag1(pWup, ot0 * (RR_KC / 16) + k2), wfrag1(pWup, ot1 * (RR_KC / 16) + k2));
            v8f acc = bias_tile2(pBup, ot0, ot1);
            [unroll] for (uint s = 0; s < RR_KC / 32; ++s)
                acc = step2l(s + 1 == RR_KC / 32, a[2 * s], cf[2 * s], a[2 * s + 1], cf[2 * s + 1], acc);
            uint bad = 0u;
            uint2 uq;
            Q8_TRY uq = q8f(acc, bad);
            Q8_ELSE uq = q8s(acc); Q8_END
            rr_up32[rr_uo + 4u * ot0] = uq.x;
            rr_up32[rr_uo + 4u * ot1] = uq.y;
        }
        GroupMemoryBarrierWithGroupSync();
        const uint par0 = wv0 * 4u + (uint)(lx >> 1), par1 = par0 + 4u;
        const uint ch0 = 2u * ((uint)Yt[0] & 1u) + ((uint)X & 1u), ch1 = 2u * ((uint)Yt[1] & 1u) + ((uint)X & 1u);
        // dword of UP slot ((par HID + ch RC + 16 ct + 8 hf) >> 3) held by this lane: 32 par + 8 ch + 4 ct + 2 hf + hh
        const uint ub0 = rr_f2 + 32u * par0 + 8u * ch0, ub1 = rr_f2 + 32u * par1 + 8u * ch1;
        uint bad = 0u;
        [unroll] for (uint ct = 0; ct < KS; ++ct) {
            const uint2 sk = rr_tld(pX, tok[0] * RC + 16u * ct, tok[1] * RC + 16u * ct);
            const uint2 up = uint2(rr_up32[ub0 + 4u * ct], rr_up32[ub1 + 4u * ct]);
            xr[ct] = bv_mad(bv_d8(sk), f2x, bv_d8(up));
        }
        Q8_TRY [unroll] for (uint c0 = 0; c0 < KS; ++c0) xf[c0] = bv_q8f(xr[c0], bad);
        Q8_ELSE [unroll] for (uint c = 0; c < KS; ++c) xf[c] = bv_q8s(xr[c]); Q8_END
    }
#else
    {   // front-end MLP folded into IN0 (variant E, as the wave32 RR_B32 form): x = Q8(MM(A_1; W2_1; MM(A_0; W2_0; b2))),
        // A_br = Q8(act(MM(f; W1_br; b1_br))) from the token's 64 input codes (pX = codes, pCin = W1T, pWup = W2T,
        // pBup = b1, pHc = b2)
        uint2 f[4];
        [unroll] for (uint ks = 0; ks < 4u; ++ks)
            f[ks] = rr_tld(pX, tok[0] * 64u + 16u * ks, tok[1] * 64u + 16u * ks);
        uint4 ah[2][2];
        [unroll] for (uint br = 0; br < 2u; ++br) {
            [unroll] for (uint ft = 0; ft < 2u; ++ft) {
                const uint w = (br * 2u + ft) * 4u;
                v8f acc = bias_tile(pBup + br * 128u, ft);
                acc = MLP_STEP2(wfrag(pCin, w + 0u), f[0], wfrag(pCin, w + 1u), f[1], acc);
                const uint4 h = MLP_STEP2H(wfrag(pCin, w + 2u), f[2], wfrag(pCin, w + 3u), f[3], acc);
                ah[br][ft] = uint4(act2(h.x, wc), act2(h.y, wc), act2(h.z, wc), act2(h.w, wc));
            }
        }
        uint2 a[2][2];
        {
            uint bad = 0u;
            Q8_TRY a[0][0] = q8hf(ah[0][0], bad); a[0][1] = q8hf(ah[0][1], bad); a[1][0] = q8hf(ah[1][0], bad);
                   a[1][1] = q8hf(ah[1][1], bad);
            Q8_ELSE a[0][0] = q8h_slow(ah[0][0]); a[0][1] = q8h_slow(ah[0][1]); a[1][0] = q8h_slow(ah[1][0]);
                    a[1][1] = q8h_slow(ah[1][1]);
            Q8_END
        }
        v8f o[2];
        [unroll] for (uint ot = 0; ot < 2u; ++ot) {
            v8f z = bias_tile(pHc, ot);
            z = MLP_STEP2(wfrag(pWup, (0u * 2u + ot) * 2u), a[0][0], wfrag(pWup, (0u * 2u + ot) * 2u + 1u), a[0][1], z);
            o[ot] = MLP_STEP2(wfrag(pWup, (1u * 2u + ot) * 2u), a[1][0], wfrag(pWup, (1u * 2u + ot) * 2u + 1u),
                              a[1][1], z);
        }
        {
            uint bad = 0u;
            Q8_TRY xf[0] = q8f(o[0], bad); xf[1] = q8f(o[1], bad);
            Q8_ELSE xf[0] = q8s(o[0]); xf[1] = q8s(o[1]);
            Q8_END
        }
    }
#endif

    // ---- heads: steps 1-6 ------------------------------------------------------------------------------------------
    v8f dpv[KS];
    rrbv dp[KS];
    [unroll] for (uint c0 = 0; c0 < KS; ++c0) dpv[c0] = bias_tile(pBo, c0);
    [unroll] for (uint h = 0; h < RH; ++h) {
        // [A] q_h (features x tokens) and v_h^T (tokens x features)
        v8f aq0 = zero8(), aq1 = zero8(), av0 = zero8(), av1 = zero8();
        wil4(aq0, aq1, av0, av1, qv_frag(h, 0u), xf[0], qv_frag(h, 1u), xf[1], qv_frag(h, KS), xf[0], qv_frag(h, KS + 1u), xf[1],
             xf[0], qv_frag(h, 2u * KS), xf[1], qv_frag(h, 2u * KS + 1u), xf[0], qv_frag(h, 3u * KS), xf[1], qv_frag(h, 3u * KS + 1u));
        uint bad = 0u;
        uint2 qf0, qf1, vf0, vf1;
        Q8_TRY qf0 = q8f(aq0, bad); qf1 = q8f(aq1, bad); vf0 = q8f(av0, bad); vf1 = q8f(av1, bad);
        Q8_ELSE qf0 = q8s(aq0); qf1 = q8s(aq1); vf0 = q8s(av0); vf1 = q8s(av1); Q8_END
        rr_xq32[rr_xqr(XB(h) + (wv0 * 2u) * 32u)] = qf0.x;
        rr_xq32[rr_xqr(XB(h) + (wv0 * 2u + 1u) * 32u)] = qf1.x;
        rr_xq32[rr_xqr(XB(h) + 256u + (wv0 * 2u) * 32u)] = vf0.x;
        rr_xq32[rr_xqr(XB(h) + 256u + (wv0 * 2u + 1u) * 32u)] = vf1.x;
        rr_xq32[rr_xqr(XB(h) + (wv0 * 2u + 2u) * 32u)] = qf0.y;
        rr_xq32[rr_xqr(XB(h) + (wv0 * 2u + 3u) * 32u)] = qf1.y;
        rr_xq32[rr_xqr(XB(h) + 256u + (wv0 * 2u + 2u) * 32u)] = vf0.y;
        rr_xq32[rr_xqr(XB(h) + 256u + (wv0 * 2u + 3u) * 32u)] = vf1.y;
        GroupMemoryBarrierWithGroupSync();

        // [B] scores (key tile J x the tile's queries), start value = the score bias (f32 fragment, per tile)
        const uint tb0 = pTb + 4u * rr_lo + (h * 16u + wv0 * 4u) * 1024u, tb1 = tb0 + 4u * 1024u;
        rrbv wt[4];
        {
            uint2 k[8];
            [unroll] for (uint i8 = 0; i8 < 8; ++i8) k[i8] = rr_xqs(XB(h) + i8 * 32u);
            v8f s0 = rr_cat4(wg_ld4f(tb0, 0u), wg_ld4f(tb1, 0u)), s1 = rr_cat4(wg_ld4f(tb0, 1024u), wg_ld4f(tb1, 1024u));
            v8f s2 = rr_cat4(wg_ld4f(tb0, 2048u), wg_ld4f(tb1, 2048u)), s3 = rr_cat4(wg_ld4f(tb0, 3072u), wg_ld4f(tb1, 3072u));
            wil4(s0, s1, s2, s3, k[0], qf0, k[1], qf1, k[2], qf0, k[3], qf1, k[4], qf0, k[5], qf1, k[6], qf0, k[7], qf1);
            wt[0] = bv_attn(s0, BVW); wt[1] = bv_attn(s1, BVW); wt[2] = bv_attn(s2, BVW); wt[3] = bv_attn(s3, BVW);
        }
        // step 4: Z, r = 1 / Z, P = Q8(w r) (f32 softmax normalisation; Z as the wave32 tree, see the head comment)
        v8f zt;
        [unroll] for (uint i3 = 0; i3 < 8; ++i3) zt.v[i3] = (wt[0].v[i3] + wt[1].v[i3]) + (wt[2].v[i3] + wt[3].v[i3]);
        const float zh0 = (zt.v[0] + zt.v[1]) + (zt.v[2] + zt.v[3]), zh1 = (zt.v[4] + zt.v[5]) + (zt.v[6] + zt.v[7]);
        const float zs0 = zh0 + wg_xor(zh0, 32u), zs1 = zh1 + wg_xor(zh1, 32u);
        const float rz0 = rcp(zs0 + wg_xor16(zs0)), rz1 = rcp(zs1 + wg_xor16(zs1));
        uint2 pf[4];
        [unroll] for (uint J4 = 0; J4 < 4; ++J4) {
            v8f p;
            [unroll] for (uint i4 = 0; i4 < 4; ++i4) { p.v[i4] = wt[J4].v[i4] * rz0; p.v[4 + i4] = wt[J4].v[4 + i4] * rz1; }
            pf[J4] = rr_q8_small(p);
        }
        // step 5: O_h = Q8(MM(P; v_h)), keys 0-31 then 32-63
        v8f ov[2];
        {
            uint2 vt[8];
            [unroll] for (uint i8 = 0; i8 < 8; ++i8) vt[i8] = rr_xqs(XB(h) + 256u + i8 * 32u);
            ov[0] = zero8(); ov[1] = zero8();
            wil2(ov[0], ov[1], vt[0], pf[0], vt[2], pf[1], vt[1], pf[0], vt[3], pf[1]);
            wil2(ov[0], ov[1], vt[4], pf[2], vt[6], pf[3], vt[5], pf[2], vt[7], pf[3]);
        }
        bad = 0u;
        uint2 of[2];
        Q8_TRY of[0] = q8f(ov[0], bad); of[1] = q8f(ov[1], bad);
        Q8_ELSE of[0] = q8s(ov[0]); of[1] = q8s(ov[1]); Q8_END
        // step 6: Dp step h
        wil2(dpv[0], dpv[1], wo_frag(h, 0u), of[0], wo_frag(h, 1u), of[1], wo_frag(h, 2u), of[0], wo_frag(h, 3u), of[1]);
        if (h + 1u == RH) { dp[0] = dpv[0]; dp[1] = dpv[1]; }
    }

    // ---- step 7: first residual -------------------------------------------------------------------------------------
    rrbv y1h[KS];
    uint2 y1q[KS];
    {
        uint bad = 0u;
        [unroll] for (uint c1 = 0; c1 < KS; ++c1) {
#if RR_KIND == 2
            y1h[c1] = bv_mad(dp[c1], ra2x, xr[c1]);
#else
            y1h[c1] = bv_mad(dp[c1], ra2x, bv_d8(xf[c1]));
#endif
        }
        Q8_TRY [unroll] for (uint c0 = 0; c0 < KS; ++c0) y1q[c0] = bv_q8f(y1h[c0], bad);
        Q8_ELSE [unroll] for (uint c = 0; c < KS; ++c) y1q[c] = bv_q8s(y1h[c]); Q8_END
    }

    // ---- step 8: MLP ------------------------------------------------------------------------------------------------
    v8f ddv[KS];
    rrbv dd[KS];
    [unroll] for (uint c2 = 0; c2 < KS; ++c2) ddv[c2] = bias_tile(pB2, c2);
    [unroll] for (uint j = 0; j < NJ; ++j) {
        rrbv h0 = bias_tile(pB1, 2u * j), h1 = bias_tile(pB1, 2u * j + 1u);
        wil2(h0, h1, mlp_frag(j, 0u), y1q[0], mlp_frag(j, 1u), y1q[1], mlp_frag(j, KS), y1q[0], mlp_frag(j, KS + 1u), y1q[1]);
        const rrbv a0h = bv_act(h0, BVW);
        const rrbv a1h = bv_act(h1, BVW);
        uint bad = 0u;
        uint2 au0, au1;
        Q8_TRY au0 = bv_q8f(a0h, bad); au1 = bv_q8f(a1h, bad);
        Q8_ELSE au0 = bv_q8s(a0h); au1 = bv_q8s(a1h); Q8_END
        wil2(ddv[0], ddv[1], mlp_frag(j, 2u * KS), au0, mlp_frag(j, 2u * KS + 1u), au1, mlp_frag(j, 2u * KS + 2u), au0,
             mlp_frag(j, 2u * KS + 3u), au1);
        if (j + 1u == NJ) { dd[0] = ddv[0]; dd[1] = ddv[1]; }
    }

    // ---- step 9: second residual, block output y2 ---------------------------------------------------------------------
    uint2 y2[KS];
    {
        rrbv yh[KS];
        uint bad = 0u;
        [unroll] for (uint c3 = 0; c3 < KS; ++c3) {
            yh[c3] = bv_mad(dd[c3], m2x, y1h[c3]);
        }
        Q8_TRY [unroll] for (uint c0 = 0; c0 < KS; ++c0) y2[c0] = bv_q8f(yh[c0], bad);
        Q8_ELSE [unroll] for (uint c = 0; c < KS; ++c) y2[c] = bv_q8s(yh[c]); Q8_END
    }
    [unroll] for (uint t4 = 0; t4 < 2u; ++t4) {
        const uint otok = (uint)(Yt[t4] * Nx + X);
        if (real[t4]) {
            [unroll] for (uint c4 = 0; c4 < KS; ++c4)
                wg_st32(pY2 + rr_t4, otok * RC + 16u * c4, t4 == 0u ? y2[c4].x : y2[c4].y);
        }
    }

#if RR_KIND == 0
    // ---- hidden state (51.4), encoder rule, and the merged hidden -------------------------------------------------
    // Rows 0-3 of the Wh tile (the 4 hidden channels) are in the hf = 0, hh = 0 lanes: dwords x, y for tile 0, z, w for
    // tile 1. Tile 1's are moved to the hh = 1 lanes (lane ^ 32), so lanes 0-15 run tile 0's token n and lanes 32-47
    // tile 1's, with the operations of the wave32 lanes 0-15 (the other lanes compute unused values, as the wave32
    // hf = 1 lanes do).
    {
        const uint4 dhb = step2h(wfrag(pWh, 0u), y2[0], wfrag(pWh, 1u), y2[1], rr_dup4(wg_ld4f(pBh + 4u * rr_t4, 0u)));
        const uint z1 = wg_xor_u(dhb.z, 32u), w1 = wg_xor_u(dhb.w, 32u);
        const bool t1 = rr_hh != 0u;
        const uint dx = t1 ? z1 : dhb.x, dy = t1 ? w1 : dhb.y;
        const uint tk = t1 ? tok[1] : tok[0];
        const int Yh = t1 ? Yt[1] : Yt[0];
        const bool rl = t1 ? real[1] : real[0];
        const uint s01 = wg_ld16(pSig, (dx & 0xffffu) * 2u) | (wg_ld16(pSig, (dx >> 16) * 2u) << 16);
        const uint s23 = wg_ld16(pSig, (dy & 0xffffu) * 2u) | (wg_ld16(pSig, (dy >> 16) * 2u) << 16);
        const uint2 hold = wg_ld64(pHold, tk * 8u);
        uint2 hn;
        const uint G01 = rr_fence_u(rr_b2(rr_h2(s01) * g2)), G23 = rr_fence_u(rr_b2(rr_h2(s23) * g2));
        hn.x = rr_fence_u(rr_b2(rr_h2(hold.x) * rr_h2(G01)));
        hn.y = rr_fence_u(rr_b2(rr_h2(hold.y) * rr_h2(G23))) & 0xffffu;
        const uint otok = (uint)(Yh * Nx + X);
        if (rl && rr_hf == 0u) wg_st64(pHnew, otok * 8u, hn);
        const half2 quarter = half2(0.25, 0.25);
        const uint qx = rr_fence_u(rr_b2(rr_h2(hn.x) * quarter)), qy = rr_fence_u(rr_b2(rr_h2(hn.y) * quarter));
        const uint l0 = 2u * (n & 3u) + 32u * rr_hh;
        const uint q0x = WaveReadLaneAt(qx, l0), q1x = WaveReadLaneAt(qx, l0 + 1u);
        const uint q2x = WaveReadLaneAt(qx, l0 + 8u), q3x = WaveReadLaneAt(qx, l0 + 9u);
        const uint q0y = WaveReadLaneAt(qy, l0), q1y = WaveReadLaneAt(qy, l0 + 1u);
        const uint q2y = WaveReadLaneAt(qy, l0 + 8u), q3y = WaveReadLaneAt(qy, l0 + 9u);
        const uint ax = rr_fence_u(rr_b2(rr_h2(q0x) + rr_h2(q1x))), bx = rr_fence_u(rr_b2(rr_h2(q2x) + rr_h2(q3x)));
        const uint ay = rr_fence_u(rr_b2(rr_h2(q0y) + rr_h2(q1y))), by = rr_fence_u(rr_b2(rr_h2(q2y) + rr_h2(q3y)));
        const uint2 hm = uint2(rr_b2(rr_h2(ax) + rr_h2(bx)), rr_b2(rr_h2(ay) + rr_h2(by)));
        const int hXc = cx0 + (int)(n & 3u), hYc = cy0 + (int)(wv0 + rr_hh);
        if (n < 4u && rr_hf == 0u && hXc >= 0 && hXc < Mx && hYc >= 0 && hYc < My)
            wg_st64(pHmerged, (uint)(hYc * Mx + hXc) * 8u, hm);
    }
#endif
    DBG_FLUSH
}
#endif   // WG_WAVE
