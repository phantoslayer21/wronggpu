// SPDX-License-Identifier: MIT
// Spec 51 window block (51.3) with its stage fronts (51.5) and tails (51.4): the kernel of rr_block.hlsl with the
// instruction diet of rr_block_c32.hlsl (rr_fast.hlsli), for any stage kind; same arguments, buffer layouts, rounding
// points and step (rr_step.hlsli, RR_STEP), and the same output bits as rr_block.hlsl for every input. Used for the
// stages listed in gpu/stage.py SPECIAL (E1, D1-F, D1-R).
//
// Differences from rr_block.hlsl (structure only):
//   * Q8 everywhere through the wave-uniform fast path (rr_fast.hlsli q8f / q8hf; the clamping rr_q8 on a wave that
//     sees a NaN code); Q8 of f16 values straight from f16 (no fence for f16 products, fma results fenced).
//   * attention weight with the packed bit step (attn_w2), activation without its fence (act2).
//   * heads unrolled (H = 2; rolled for H > 2); the first Dp step and the first MLP chunk's Dd steps take the f32 bias as loaded (the base
//     rounds it to f16 and widens it back: the same exact value); the last step of a chain packed directly.
//   * RR_WSRC 1 (default): weight fragments read from memory by every wave; 0: staged through LDS (the base Stager).
//     The q / v exchange is double-buffered (one barrier per head).
//
// Defines as rr_block.hlsl: RR_C, RR_H, RR_KIND, RR_CN, RR_HMERGE, RR_ADAPT, RR_KC, RR_STEP, RR_Z (RR_LDSW: see
// RR_WSRC); RR_ADAPT 2: the E1 adapter reads the four children straight from IN0's y2 (the regroup dispatch is gone).
#include "wg_platform.hlsli"

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
#define RR_L01BIT (RR_ADAPT ? 4 : RR_KIND == 2 ? 8 : 16)   // clean/rr-f-a14-l01: this stage's bit in the L01_* masks
#ifndef L01_PFB
#define L01_PFB 0             // clean/rr-f-a14-l01 A7 post-barrier-prefetch: stages (L01 bits) whose score-bias tiles J = 0, 1 are loaded before the q / v barrier
#endif
#define RR_PFB ((L01_PFB & RR_L01BIT) != 0 && RR_WSRC == 1 && !RR_QV1)
// clean/rr-mp2-level1 (round 2, loop shape / load issue order only: the same operations in the same order, SB-C).
// Defaults with the relaxed body (RR_B32: modes accurate and fast; the exact body's kernels keep their DXIL):
// L01_MRGU 4, L01_UNRALL 28, L01_MDPF 28, L01_FPF 8 (cap8 + cap48 byte-identical in accurate and fast; game_bench
// 1440p Q accurate -0.059 / fast -0.056 ms, 6 interleaved rounds); L01_HPF and L01_MLPDEF stay opt-in (no gain).
#if defined(RR_B32) && RR_B32
#define RR_L1R2 1
#else
#define RR_L1R2 0
#endif
//   L01_MRGU   L1a: stages (L01 bits) whose merge inner loop is unrolled with the MY indices as a per-lane base plus
//              compile-time offsets (C = 64 only)
//   L01_UNRALL L1b: relaxed-body stages (L01 bits) whose MLP chunk loop is fully unrolled (the exact body's
//              L01_UNROLLMASK form, RR_WSRC 1)
#ifndef L01_MRGU
#if RR_L1R2
#define L01_MRGU 4
#else
#define L01_MRGU 0
#endif
#endif
#define RR_MRGU ((L01_MRGU & RR_L01BIT) != 0 && RC == 64 && HID == 256 && !RR_COUNT)
#ifndef L01_UNRALL
#if RR_L1R2
#define L01_UNRALL 28
#else
#define L01_UNRALL 0
#endif
#endif
//   L01_MDPF   L1d: stages (L01 bits) whose MLP Dd weight fragments are loaded right after the chunk's hidden WMMAs,
//              before the activation and the Q8 branch (the same loads, issued earlier; the driver does not hoist
//              loads above the Q8 branch). Registers: the hidden fragments die at those WMMAs.
#ifndef L01_MDPF
#if RR_L1R2
#define L01_MDPF 28
#else
#define L01_MDPF 0
#endif
#endif
#define RR_MDPF ((L01_MDPF & RR_L01BIT) != 0 && RR_WIL && !L01_WIL4)
//   L01_FPF    L1e: front prefetch: E1's adapter tiles (bit 4) and D1-F's up-projection tiles (bit 8) load the next
//              tile's weight fragments and bias before the current tile's WMMAs and Q8 group (the same loads in a
//              different issue order; the up-projection loop unrolled). Same operations in the same order.
#ifndef L01_FPF
#if RR_L1R2
#define L01_FPF 8
#else
#define L01_FPF 0
#endif
#endif
#define RR_FPF ((L01_FPF & RR_L01BIT) != 0)
//   L01_HPF    L1f: hidden-state tail: the previous hidden state (pHold) and, in the decoder (kind 3), the four coarse
//              hidden-state texels (pHc) are loaded before the hidden-projection WMMAs instead of after them (the same
//              loads and operations; only the issue point moves).
#ifndef L01_HPF
#define L01_HPF 0
#endif
#define RR_HPF ((L01_HPF & RR_L01BIT) != 0)
v8f rr_sbt3(uint J, v8f a, v8f b, uint tb) { v8f r; if (J == 0u) r = a; else if (J == 1u) r = b; else r = wg_load8f(tb, J * 1024u); return r; }
#ifndef L01_B16R
#define L01_B16R 0
#endif
#if defined(RR_B32) && RR_B32 && ((RR_ADAPT ? 4 : RR_KIND == 2 ? 8 : 16) & L01_B16R) != 0
#undef RR_B32
#define RR_B32 0
#define RR_B16R 1
#endif
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
#ifndef RR_ADAPT
#define RR_ADAPT 0
#endif
#ifndef RR_KC
#define RR_KC 0
#endif
#ifndef RR_Z
#define RR_Z 0
#endif
#ifndef RR_WSRC
#define RR_WSRC 1
#endif

#define RC RR_C
#define RH RR_H
#define KS (RC / 16)
#define NS (RC / 32)
#define HID (4 * RC)
#define NJ (HID / 32)
#define HIDDEN (RR_KIND == 0 || RR_KIND == 3)
#define UNIT (64 * RC)

uint2 rr_q2(uint64_t q) { return uint2((uint)q, (uint)(q >> 32)); }
uint64_t rr_2q(uint2 v) { return (uint64_t)v.x | ((uint64_t)v.y << 32); }

// clean/rr-f-s13x-x-l01 RR_QV1 1 (H = 2, RR_WSRC 1): one q / v barrier per window, as rr_block_c32.hlsl RR_QV1.
#ifndef L01_ACC
#define L01_ACC 0             // clean/rr-f-a14-l01: 1 = RR_DEFER / RR_QV1 masks apply to the relaxed body (accurate / fast) too
#endif
#ifndef L01_QV1MASK
#define L01_QV1MASK 0         // exact body kernels with RR_QV1: 1 IN0, 2 D0-F, 4 E1, 8 D1-F, 16 D1-R
#endif
#ifndef RR_QV1
#define RR_QV1 ((RR_Z == 2 || L01_ACC) && RR_H == 2 && RR_WSRC == 1 && ((RR_ADAPT ? 4 : RR_KIND == 2 ? 8 : 16) & L01_QV1MASK) != 0)
#endif
#if RR_QV1 && (RR_WSRC == 0 || RR_H != 2)
#error "RR_QV1 needs RR_WSRC 1 and H = 2"
#endif
// ---- LDS --------------------------------------------------------------------------------------------------------------
groupshared uint64_t rr_xq[1024];                    // q / v exchange, two head buffers: q [wave][ft][lane], v^T after
#define XB(h) (((h) & 1u) * 512u)
#if RR_WSRC == 0
groupshared uint64_t rr_wb[2 * UNIT / 8];
#define MY(i) rr_wb[i]
#define UP(i) rr_wb[UNIT / 8 + (i)]
#else
#if RR_CN > 0 && RR_Z == 2 && RH == 2 && 64 * RC / 8 <= 512 && !RR_QV1
// clean/rr-f-s13x-x-l01 (exact body only; round-11 item 5): the merge buffer lives in head 0's exchange buffer (head 1
// uses buffer 1). Its last reader is head 0's PV, and every wave passes head 1's barrier only after those reads. Same
// operations and bits; LDS 12288 -> 8192 B (E1).
#define MY(i) rr_xq[(i)]
#elif RR_CN > 0
groupshared uint64_t rr_my[64 * RC / 8];
#define MY(i) rr_my[i]
#endif
#if RR_KIND == 2 && RR_Z == 2 && RH == 2 && 16 * HID / 8 <= 512 && !RR_QV1
// clean/rr-f-s13x-x-l01 (exact body only): the fusion up-projections live in head 1's exchange buffer. They are
// written and read before the head loop, and head 1 first writes its buffer after head 0's barrier, which every wave
// reaches only after its last UP read. Same operations and bits; LDS 12288 -> 8192 B (D1-F).
#define UP(i) rr_xq[512u + (i)]
#elif RR_KIND == 2
groupshared uint64_t rr_up[16 * HID / 8];
#define UP(i) rr_up[i]
#endif
#endif

static uint rr_lane;
static uint rr_hf;

uint2 wfrag(uint base, uint f) { return wg_ld64(base, f * 256u + rr_lane * 8u); }
v8f zero8() { return wg_splat(0.0f); }
// clean/rr-mp2-level1 L1c (L01_MLPDEF, relaxed body, stages by L01 bit; SB-P: mpeak ALGEBRA 2.A1 + probe P3): the MLP's
// Q8 code checks (y1q and every Au_j) deferred to the y2 group's vote: a NaN code from an unchecked Q8 reaches the y2
// codes through the WMMA / act / fma chain, so the y2 vote sees it; a wave whose y2 vote fails reruns steps 7-9 in the
// checked slow forms (pass 1, wave-local; the loads of pass 1 use an opaque zero offset so they are not shared with
// pass 0). The shipped probe P3 form restricted to level 1 (E1, D1-F, D1-R).
#if defined(RR_B32) && RR_B32 && defined(L01_MLPDEF) && (L01_MLPDEF & RR_L01BIT) != 0
#define RR_MLPDEF 1
#else
#define RR_MLPDEF 0
#endif
#if RR_MLPDEF
static uint pa_x;   // 0; opaque per pass so that the rerun's loads are not hoisted
v8f bias_tile(uint p, uint tile) { return wg_load8f(p + pa_x, (16u * tile + 8u * rr_hf) * 4u); }
#else
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
#if RR_MLPDEF
uint2 mlp_frag(uint j, uint f) { return wfrag(pMlp + j * UNIT + pa_x, f); }
#else
uint2 mlp_frag(uint j, uint f) { return wfrag(pMlp + j * UNIT, f); }
#endif
struct RrPf { uint v; };
RrPf st_fetch(uint src, uint bytes) { RrPf p; p.v = 0u; return p; }
void st_commit(RrPf p, uint buf, uint bytes) {}
void st_sync() {}
#endif

#include "rr_fast.hlsli"
#if RR_MLPDEF
#define bv_q8n(X_) rr_q8_small(X_)
static bool pa_slow;
#endif
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
#if ((RR_ADAPT ? 4 : RR_KIND == 2 ? 8 : 16) & L01_WREV) != 0
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
#define RR_DEFER ((RR_Z == 2 || L01_ACC) && ((RR_ADAPT ? 4 : RR_KIND == 2 ? 8 : 16) & L01_DEFERMASK) != 0)
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
#if RH <= 2
#define HEAD_LOOP [unroll]
#define HEAD_ZL 0u
#else
#define HEAD_LOOP RR_LOOP
#define HEAD_ZL RR_ZL
#endif

// One MLP chunk j (step 8): Du_j (NS steps, start b1), Au_j = Q8(act(Du_j)), then one Dd step per channel tile.
// first: the Dd start values are the f32 bias tiles (the chunk-0 form); otherwise the widened packed dd.
void mlp_chunk(uint j, bool first, uint2 y1q[KS], inout rrbv dd[KS], RrWx wc RR_FPP)
{
    v8f u0 = bias_tile(pB1, 2u * j), u1 = bias_tile(pB1, 2u * j + 1u);
    rrbv h0, h1;
    [unroll] for (uint s = 0; s < NS; ++s) {
        const uint2 a00 = mlp_frag(j, 2u * s), a01 = mlp_frag(j, 2u * s + 1u);
        const uint2 a10 = mlp_frag(j, KS + 2u * s), a11 = mlp_frag(j, KS + 2u * s + 1u);
#if RR_WIL
        wil2(u0, u1, a00, y1q[2 * s], a01, y1q[2 * s + 1], a10, y1q[2 * s], a11, y1q[2 * s + 1]);
        if (s + 1 == NS) { h0 = u0; h1 = u1; }
#else
        if (s + 1 < NS) {
            u0 = step2(a00, y1q[2 * s], a01, y1q[2 * s + 1], u0);
            u1 = step2(a10, y1q[2 * s], a11, y1q[2 * s + 1], u1);
        } else {
            h0 = bv_step(a00, y1q[2 * s], a01, y1q[2 * s + 1], u0);
            h1 = bv_step(a10, y1q[2 * s], a11, y1q[2 * s + 1], u1);
        }
#endif
    }
#if RR_MDPF
    uint2 wd[2 * KS];
    [unroll] for (uint q = 0; q < 2 * KS; ++q) wd[q] = mlp_frag(j, 2u * KS + q);
#define RR_WD(I_) wd[(I_)]
#else
#define RR_WD(I_) mlp_frag(j, 2u * KS + (I_))
#endif
    const rrbv a0h = bv_act(h0, wc);
    const rrbv a1h = bv_act(h1, wc);
    uint bad = 0u;
    uint2 au0, au1;
#if RR_MLPDEF
    [branch] if (!pa_slow) { au0 = bv_q8n(a0h); au1 = bv_q8n(a1h); } else { au0 = bv_q8s(a0h); au1 = bv_q8s(a1h); }
#else
    Q8_TRY au0 = bv_q8f(a0h, bad); au1 = bv_q8f(a1h, bad);
    Q8_ELSE au0 = bv_q8s(a0h); au1 = bv_q8s(a1h); Q8_END
#endif
#if RR_WIL
#if L01_WIL4 && KS == 4
    {   // A5 wil4: the four Dd chains interleaved
        v8f e[4];
        [unroll] for (uint c3 = 0; c3 < 4u; ++c3) { if (first) e[c3] = bias_tile(pB2, c3); else e[c3] = dd[c3]; }
        wil4(e[0], e[1], e[2], e[3], mlp_frag(j, 2u * KS + 0u), au0, mlp_frag(j, 2u * KS + 1u), au1,
             mlp_frag(j, 2u * KS + 2u), au0, mlp_frag(j, 2u * KS + 3u), au1, mlp_frag(j, 2u * KS + 4u), au0,
             mlp_frag(j, 2u * KS + 5u), au1, mlp_frag(j, 2u * KS + 6u), au0, mlp_frag(j, 2u * KS + 7u), au1);
        [unroll] for (uint c4 = 0; c4 < 4u; ++c4) dd[c4] = e[c4];
    }
#else
    [unroll] for (uint c2 = 0; c2 < KS; c2 += 2u) {
        v8f e0, e1;
        if (first) { e0 = bias_tile(pB2, c2); e1 = bias_tile(pB2, c2 + 1u); } else { e0 = dd[c2]; e1 = dd[c2 + 1u]; }
        wil2(e0, e1, RR_WD(2u * c2), au0, RR_WD(2u * c2 + 1u), au1,
             RR_WD(2u * c2 + 2u), au0, RR_WD(2u * c2 + 3u), au1);
        dd[c2] = e0; dd[c2 + 1u] = e1;
    }
#endif
#undef RR_WD
#else
    [unroll] for (uint ct = 0; ct < KS; ++ct) {
        const uint2 w0 = mlp_frag(j, 2u * KS + 2u * ct), w1 = mlp_frag(j, 2u * KS + 2u * ct + 1u);
        v8f c;
        if (first) c = bias_tile(pB2, ct); else c = bv_un(dd[ct]);
        dd[ct] = bv_step(w0, au0, w1, au1, c);
    }
#endif
}

#ifndef L01_WIL4
#define L01_WIL4 0            // clean/rr-f-a14-l01 A5: fast kernels' Dp / Dd as four interleaved chains (C = 64)
#endif
#ifndef L01_MLA
#define L01_MLA 0             // clean/rr-f-a14-l01 A7 mlp-lookahead: stages (L01 bits) that issue chunk j+1's hidden WMMAs before chunk j's activation / Q8 / Dd
#endif
#define RR_MLA ((L01_MLA & RR_L01BIT) != 0 && RR_WIL && RR_WSRC == 1)
#if RR_MLA
// mlp_chunk split in two (the same operations): the hidden pair of chunk j, then its activation, Q8 and Dd steps
void mlp_hid(uint j, uint2 y1q[KS], out v8f u0, out v8f u1)
{
    u0 = bias_tile(pB1, 2u * j); u1 = bias_tile(pB1, 2u * j + 1u);
    [unroll] for (uint s = 0; s < NS; ++s)
        wil2(u0, u1, mlp_frag(j, 2u * s), y1q[2 * s], mlp_frag(j, 2u * s + 1u), y1q[2 * s + 1],
             mlp_frag(j, KS + 2u * s), y1q[2 * s], mlp_frag(j, KS + 2u * s + 1u), y1q[2 * s + 1]);
}
void mlp_out(uint j, bool first, v8f h0, v8f h1, inout rrbv dd[KS], RrWx wc RR_FPP)
{
    const rrbv a0h = bv_act(h0, wc);
    const rrbv a1h = bv_act(h1, wc);
    uint bad = 0u;
    uint2 au0, au1;
#if RR_MLPDEF
    [branch] if (!pa_slow) { au0 = bv_q8n(a0h); au1 = bv_q8n(a1h); } else { au0 = bv_q8s(a0h); au1 = bv_q8s(a1h); }
#else
    Q8_TRY au0 = bv_q8f(a0h, bad); au1 = bv_q8f(a1h, bad);
    Q8_ELSE au0 = bv_q8s(a0h); au1 = bv_q8s(a1h); Q8_END
#endif
#if L01_WIL4 && KS == 4
    {   // A5 wil4: the four Dd chains interleaved
        v8f e[4];
        [unroll] for (uint c3 = 0; c3 < 4u; ++c3) { if (first) e[c3] = bias_tile(pB2, c3); else e[c3] = dd[c3]; }
        wil4(e[0], e[1], e[2], e[3], mlp_frag(j, 2u * KS + 0u), au0, mlp_frag(j, 2u * KS + 1u), au1,
             mlp_frag(j, 2u * KS + 2u), au0, mlp_frag(j, 2u * KS + 3u), au1, mlp_frag(j, 2u * KS + 4u), au0,
             mlp_frag(j, 2u * KS + 5u), au1, mlp_frag(j, 2u * KS + 6u), au0, mlp_frag(j, 2u * KS + 7u), au1);
        [unroll] for (uint c4 = 0; c4 < 4u; ++c4) dd[c4] = e[c4];
    }
#else
    [unroll] for (uint c2 = 0; c2 < KS; c2 += 2u) {
        v8f e0, e1;
        if (first) { e0 = bias_tile(pB2, c2); e1 = bias_tile(pB2, c2 + 1u); } else { e0 = dd[c2]; e1 = dd[c2 + 1u]; }
        wil2(e0, e1, mlp_frag(j, 2u * KS + 2u * c2), au0, mlp_frag(j, 2u * KS + 2u * c2 + 1u), au1,
             mlp_frag(j, 2u * KS + 2u * c2 + 2u), au0, mlp_frag(j, 2u * KS + 2u * c2 + 3u), au1);
        dd[c2] = e0; dd[c2 + 1u] = e1;
    }
#endif
}
#endif

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
#if RR_KIND == 2
    rrbv xr[KS];                                             // xf of 51.5 (f16; f32 with RR_B32), the first residual base
#endif
#if RR_ADAPT
    {   // E1: x = Q8(MM(in; Wa; ba)), in = the level-1 token's 128 channels as stored (51.5, 51.6)
        uint2 inf[8];
#if RR_ADAPT == 2
        // pX = IN0's y2 itself ([2 Ny][2 Nx][32], no regroup dispatch): child c = 2 dy + dx of the level-1 token is
        // the level-0 token (2 Yr + dy, 2 Xr + dx); its 32 bytes are channels 32 c .. 32 c + 31 of the 128 (51.5).
        [unroll] for (uint k = 0; k < 8; ++k) {
            const uint t0 = (2u * (uint)Yr + (k >> 2)) * (2u * (uint)Nx) + 2u * (uint)Xr + ((k >> 1) & 1u);
            inf[k] = RR_PLD64(pX, t0 * 32u + 16u * (k & 1u) + 8u * rr_hf);
        }
#else
        [unroll] for (uint k = 0; k < 8; ++k) inf[k] = RR_PLD64(pX, tok * 128u + 16u * k + 8u * rr_hf);
#endif
#if RR_FPF
        uint2 an[8];
        v8f accn = bias_tile(pBa, 0u);
        [unroll] for (uint k0 = 0; k0 < 8; ++k0) an[k0] = wfrag(pWa, k0);
#endif
        [unroll] for (uint ct = 0; ct < KS; ++ct) {
            uint2 a[8];
#if RR_FPF
            [unroll] for (uint k1 = 0; k1 < 8; ++k1) a[k1] = an[k1];
            v8f acc = accn;
            if (ct + 1u < KS) {
                accn = bias_tile(pBa, ct + 1u);
                [unroll] for (uint k3 = 0; k3 < 8; ++k3) an[k3] = wfrag(pWa, (ct + 1u) * 8u + k3);
            }
#else
            [unroll] for (uint k2 = 0; k2 < 8; ++k2) a[k2] = wfrag(pWa, ct * 8u + k2);
            v8f acc = bias_tile(pBa, ct);
#endif
            [unroll] for (uint s = 0; s < 4; ++s) acc = step2l(s == 3u, a[2 * s], inf[2 * s], a[2 * s + 1], inf[2 * s + 1], acc);
            uint bad = 0u;
            Q8_TRY xf[ct] = q8f(acc, bad);
            Q8_ELSE xf[ct] = q8s(acc); Q8_END
        }
    }
#elif RR_KIND == 2
    {   // fusion half (51.5)
        const int crx = cx0 + (int)(n & 3u), cry = cy0 + (int)(n >> 2);
        const uint ctok = (uint)(rr_refl(cry, My, r1) * Mx + rr_refl(crx, Mx, r1));
        uint2 cf[RR_KC / 16];
        [unroll] for (uint k = 0; k < RR_KC / 16; ++k) cf[k] = RR_PLD64(pCin, ctok * RR_KC + 16u * k + 8u * rr_hf);
#if RR_FPF
        uint2 an[RR_KC / 16];
        v8f accn = bias_tile(pBup, wv);
        [unroll] for (uint k0 = 0; k0 < RR_KC / 16; ++k0) an[k0] = wfrag(pWup, wv * (RR_KC / 16) + k0);
        [unroll] for (uint t2 = 0; t2 < HID / 64; ++t2) {
            const uint ot = wv + 4u * t2;
            uint2 a[RR_KC / 16];
            [unroll] for (uint k1 = 0; k1 < RR_KC / 16; ++k1) a[k1] = an[k1];
            v8f acc = accn;
            if (t2 + 1u < HID / 64) {
                accn = bias_tile(pBup, ot + 4u);
                [unroll] for (uint k3 = 0; k3 < RR_KC / 16; ++k3) an[k3] = wfrag(pWup, (ot + 4u) * (RR_KC / 16) + k3);
            }
#else
        RR_LOOP for (uint t2 = 0; t2 < HID / 64 + RR_ZL; ++t2) {
            const uint ot = wv + 4u * t2;
            uint2 a[RR_KC / 16];
            [unroll] for (uint k2 = 0; k2 < RR_KC / 16; ++k2) a[k2] = wfrag(pWup, ot * (RR_KC / 16) + k2);
            v8f acc = bias_tile(pBup, ot);
#endif
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
#else
    [unroll] for (uint ks = 0; ks < KS; ++ks) xf[ks] = RR_PLD64(pX, tok * RC + 16u * ks + 8u * rr_hf);
#endif

    // ---- heads: steps 1-6 ------------------------------------------------------------------------------------------
    rrbv dp[KS];
#if RH > 2                                                   // heads rolled: the base's packed bias start
    [unroll] for (uint c0 = 0; c0 < KS; ++c0) dp[c0] = bv_of(bias_tile(pBo, c0));
#endif
    st_commit(st_fetch(pWqv, UNIT), 0u, UNIT);
    st_sync();
#if RR_QV1
    [unroll] for (uint hq = 0; hq < RH; ++hq) {   // [A] of both heads, one barrier
        const uint h = hq;
        v8f aq0 = zero8(), aq1 = zero8(), av0 = zero8(), av1 = zero8();
        [unroll] for (uint s = 0; s < NS; ++s) {
            const uint2 q00 = qv_frag(h, 2u * s), q01 = qv_frag(h, 2u * s + 1u);
            const uint2 q10 = qv_frag(h, KS + 2u * s), q11 = qv_frag(h, KS + 2u * s + 1u);
            const uint2 v00 = qv_frag(h, 2u * KS + 2u * s), v01 = qv_frag(h, 2u * KS + 2u * s + 1u);
            const uint2 v10 = qv_frag(h, 3u * KS + 2u * s), v11 = qv_frag(h, 3u * KS + 2u * s + 1u);
#if RR_WIL
            wil4(aq0, aq1, av0, av1, q00, xf[2 * s], q01, xf[2 * s + 1], q10, xf[2 * s], q11, xf[2 * s + 1],
                 xf[2 * s], v00, xf[2 * s + 1], v01, xf[2 * s], v10, xf[2 * s + 1], v11);
#else
            aq0 = step2l(s + 1 == NS, q00, xf[2 * s], q01, xf[2 * s + 1], aq0);
            aq1 = step2l(s + 1 == NS, q10, xf[2 * s], q11, xf[2 * s + 1], aq1);
            av0 = step2l(s + 1 == NS, xf[2 * s], v00, xf[2 * s + 1], v01, av0);
            av1 = step2l(s + 1 == NS, xf[2 * s], v10, xf[2 * s + 1], v11, av1);
#endif
        }
        uint bad = 0u;
        uint2 qf0, qf1, vf0, vf1;
        Q8_TRY qf0 = q8f(aq0, bad); qf1 = q8f(aq1, bad); vf0 = q8f(av0, bad); vf1 = q8f(av1, bad);
        Q8_ELSE qf0 = q8s(aq0); qf1 = q8s(aq1); vf0 = q8s(av0); vf1 = q8s(av1); Q8_END
        const uint xb = XB(h);
        rr_xq[xb + (wv * 2u) * 32u + rr_lane] = rr_2q(qf0);
        rr_xq[xb + (wv * 2u + 1u) * 32u + rr_lane] = rr_2q(qf1);
        rr_xq[xb + 256u + (wv * 2u) * 32u + rr_lane] = rr_2q(vf0);
        rr_xq[xb + 256u + (wv * 2u + 1u) * 32u + rr_lane] = rr_2q(vf1);
    }
    GroupMemoryBarrierWithGroupSync();
#endif
    HEAD_LOOP for (uint h = 0; h < RH + HEAD_ZL; ++h) {
#if RR_QV1
        uint bad = 0u;
        const uint xb = XB(h);
        const uint2 qf0 = rr_q2(rr_xq[xb + (wv * 2u) * 32u + rr_lane]);
        const uint2 qf1 = rr_q2(rr_xq[xb + (wv * 2u + 1u) * 32u + rr_lane]);
#else
        const RrPf pfw = st_fetch(pWo + h * (UNIT / 2), UNIT / 2);
        v8f aq0 = zero8(), aq1 = zero8(), av0 = zero8(), av1 = zero8();
        [unroll] for (uint s = 0; s < NS; ++s) {
            const uint2 q00 = qv_frag(h, 2u * s), q01 = qv_frag(h, 2u * s + 1u);
            const uint2 q10 = qv_frag(h, KS + 2u * s), q11 = qv_frag(h, KS + 2u * s + 1u);
            const uint2 v00 = qv_frag(h, 2u * KS + 2u * s), v01 = qv_frag(h, 2u * KS + 2u * s + 1u);
            const uint2 v10 = qv_frag(h, 3u * KS + 2u * s), v11 = qv_frag(h, 3u * KS + 2u * s + 1u);
#if RR_WIL
            wil4(aq0, aq1, av0, av1, q00, xf[2 * s], q01, xf[2 * s + 1], q10, xf[2 * s], q11, xf[2 * s + 1],
                 xf[2 * s], v00, xf[2 * s + 1], v01, xf[2 * s], v10, xf[2 * s + 1], v11);
#else
            aq0 = step2l(s + 1 == NS, q00, xf[2 * s], q01, xf[2 * s + 1], aq0);
            aq1 = step2l(s + 1 == NS, q10, xf[2 * s], q11, xf[2 * s + 1], aq1);
            av0 = step2l(s + 1 == NS, xf[2 * s], v00, xf[2 * s + 1], v01, av0);
            av1 = step2l(s + 1 == NS, xf[2 * s], v10, xf[2 * s + 1], v11, av1);
#endif
        }
        uint bad = 0u;
        uint2 qf0, qf1, vf0, vf1;
        Q8_TRY qf0 = q8f(aq0, bad); qf1 = q8f(aq1, bad); vf0 = q8f(av0, bad); vf1 = q8f(av1, bad);
        Q8_ELSE qf0 = q8s(aq0); qf1 = q8s(aq1); vf0 = q8s(av0); vf1 = q8s(av1); Q8_END
        const uint xb = XB(h);
        rr_xq[xb + (wv * 2u) * 32u + rr_lane] = rr_2q(qf0);
        rr_xq[xb + (wv * 2u + 1u) * 32u + rr_lane] = rr_2q(qf1);
        rr_xq[xb + 256u + (wv * 2u) * 32u + rr_lane] = rr_2q(vf0);
        rr_xq[xb + 256u + (wv * 2u + 1u) * 32u + rr_lane] = rr_2q(vf1);
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

#ifndef RR_SBT
#define RR_SBT(J_) wg_load8f(tb, (J_) * 1024u)
#endif
        const uint tb = pTb + (h * 16u + wv * 4u) * 1024u + rr_lane * 32u;
        rrbv wt[4];
#if RR_WIL
        {
            uint2 k[8];
            [unroll] for (uint i8 = 0; i8 < 8; ++i8) k[i8] = rr_q2(rr_xq[xb + i8 * 32u + rr_lane]);
            v8f s0 = RR_SBT(0u), s1 = RR_SBT(1u), s2 = RR_SBT(2u), s3 = RR_SBT(3u);
            wil4(s0, s1, s2, s3, k[0], qf0, k[1], qf1, k[2], qf0, k[3], qf1, k[4], qf0, k[5], qf1, k[6], qf0, k[7], qf1);
            wt[0] = bv_attn(s0, BVW); wt[1] = bv_attn(s1, BVW); wt[2] = bv_attn(s2, BVW); wt[3] = bv_attn(s3, BVW);
        }
#else
        [unroll] for (uint J = 0; J < 4; ++J) {
            const uint2 k0 = rr_q2(rr_xq[xb + (J * 2u) * 32u + rr_lane]);
            const uint2 k1 = rr_q2(rr_xq[xb + (J * 2u + 1u) * 32u + rr_lane]);
            const rrbv s16 = bv_step(k0, qf0, k1, qf1, RR_SBT(J));
            wt[J] = bv_attn(s16, BVW);
        }
#endif
#if RR_B32
        // f32 softmax normalisation: row sum of the f32 weights (this lane's 32, then the other half-wave's), one
        // reciprocal, p = w / sum in f32, Q8 straight from f32 (p in [0, 1])
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
        v8f ov[2];
#if RR_WIL
        {
            uint2 vt[8];
            [unroll] for (uint i8 = 0; i8 < 8; ++i8) vt[i8] = rr_q2(rr_xq[xb + 256u + i8 * 32u + rr_lane]);
            ov[0] = zero8(); ov[1] = zero8();
            wil2(ov[0], ov[1], vt[0], pf[0], vt[2], pf[1], vt[1], pf[0], vt[3], pf[1]);
            wil2(ov[0], ov[1], vt[4], pf[2], vt[6], pf[3], vt[5], pf[2], vt[7], pf[3]);
        }
#else
        [unroll] for (uint ft = 0; ft < 2; ++ft) {
            const uint2 a0 = rr_q2(rr_xq[xb + 256u + (0u + ft) * 32u + rr_lane]);
            const uint2 a1 = rr_q2(rr_xq[xb + 256u + (2u + ft) * 32u + rr_lane]);
            const uint2 a2 = rr_q2(rr_xq[xb + 256u + (4u + ft) * 32u + rr_lane]);
            const uint2 a3 = rr_q2(rr_xq[xb + 256u + (6u + ft) * 32u + rr_lane]);
            ov[ft] = step2l(true, a2, pf[2], a3, pf[3], step2(a0, pf[0], a1, pf[1], zero8()));
        }
#endif
        bad = 0u;
        uint2 of[2];
        Q8_TRY of[0] = q8f(ov[0], bad); of[1] = q8f(ov[1], bad);
        Q8_ELSE of[0] = q8s(ov[0]); of[1] = q8s(ov[1]); Q8_END
        [unroll] for (uint ct = 0; ct < KS; ++ct) {
            v8f c;
#if RH > 2
            c = bv_un(dp[ct]);
#else
            if (h == 0u) c = bias_tile(pBo, ct); else c = bv_un(dp[ct]);
#endif
#if RR_WIL
            dp[ct] = c;
        }
#if L01_WIL4 && KS == 4
        wil4(dp[0], dp[1], dp[2], dp[3], wo_frag(h, 0u), of[0], wo_frag(h, 1u), of[1], wo_frag(h, 2u), of[0],
             wo_frag(h, 3u), of[1], wo_frag(h, 4u), of[0], wo_frag(h, 5u), of[1], wo_frag(h, 6u), of[0], wo_frag(h, 7u), of[1]);
#else
        [unroll] for (uint c2 = 0; c2 < KS; c2 += 2u)
            wil2(dp[c2], dp[c2 + 1u], wo_frag(h, 2u * c2), of[0], wo_frag(h, 2u * c2 + 1u), of[1],
                 wo_frag(h, 2u * c2 + 2u), of[0], wo_frag(h, 2u * c2 + 3u), of[1]);
#endif
#else
            dp[ct] = bv_step(wo_frag(h, 2u * ct), of[0], wo_frag(h, 2u * ct + 1u), of[1], c);
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
#if !RR_MLPDEF
        Q8_TRY [unroll] for (uint c0 = 0; c0 < KS; ++c0) y1q[c0] = bv_q8f(y1h[c0], bad);
        Q8_ELSE [unroll] for (uint c = 0; c < KS; ++c) y1q[c] = bv_q8s(y1h[c]); Q8_END
#endif
    }
#if RR_MLPDEF
    uint2 y2[KS];
    pa_slow = rr_sat;
    [loop] for (uint pa_pass = 0u; pa_pass < 2u + RR_ZL; ++pa_pass) {
    pa_x = pa_pass & rr_zero_bits();
    [branch] if (!pa_slow) { [unroll] for (uint c0 = 0; c0 < KS; ++c0) y1q[c0] = bv_q8n(y1h[c0]); }
    else { [unroll] for (uint c = 0; c < KS; ++c) y1q[c] = bv_q8s(y1h[c]); }
#endif

    // ---- step 8: MLP (chunk 0 peeled: its Dd steps start from the f32 bias) -----------------------------------------
    rrbv dd[KS];
#if RR_MLA
    {   // A7 lookahead: chunk j + 1's hidden pair issued before chunk j's activation / Q8 / Dd (same chunks, same order)
        v8f hc0, hc1, hn0, hn1;
        mlp_hid(0u, y1q, hc0, hc1);
        mlp_hid(1u, y1q, hn0, hn1);
        mlp_out(0u, true, hc0, hc1, dd, BVW RR_FPA);
        hc0 = hn0; hc1 = hn1;
        RR_LOOP for (uint j = 1; j < NJ - 1u + RR_ZL; ++j) {
            mlp_hid(j + 1u, y1q, hn0, hn1);
            mlp_out(j, false, hc0, hc1, dd, BVW RR_FPA);
            hc0 = hn0; hc1 = hn1;
        }
        mlp_out(NJ - 1u, false, hc0, hc1, dd, BVW RR_FPA);
    }
    if (false) { {   // the plain chunk loop below, dead under RR_MLA
#else
    {
#endif
        const RrPf pfm = st_fetch(pMlp + min(1u, NJ - 1u) * UNIT, UNIT);
        mlp_chunk(0u, true, y1q, dd, BVW RR_FPA);
        if (NJ > 1u) st_commit(pfm, 1u, UNIT);
        st_sync();
    }
#ifndef L01_UNR2MASK
#if defined(RR_B32) && RR_B32   // a14 default (relaxed body): 2x MLP chunk loop on E1, D1-F, D1-R
#define L01_UNR2MASK 28
#else
#define L01_UNR2MASK 0
#endif
#endif
#ifndef L01_UNROLLMASK
#define L01_UNROLLMASK 0      // clean/rr-f-s13x-x-l01: exact kernels whose MLP chunk loop is unrolled: 4 E1, 8 D1-F, 16 D1-R
#endif
#if (RR_Z == 2 && ((RR_ADAPT ? 4 : RR_KIND == 2 ? 8 : 16) & L01_UNROLLMASK) != 0) || (RR_WSRC == 1 && (RR_L01BIT & L01_UNRALL) != 0)
    [unroll] for (uint j = 1; j < NJ; ++j) {
        const RrPf pfm = st_fetch(pMlp + min(j + 1u, NJ - 1u) * UNIT, UNIT);
        mlp_chunk(j, false, y1q, dd, BVW RR_FPA);
        if (j + 1u < NJ) st_commit(pfm, (j + 1u) & 1u, UNIT);
        st_sync();
#elif RR_WSRC == 1 && (NJ & 1u) == 0 && ((RR_ADAPT ? 4 : RR_KIND == 2 ? 8 : 16) & L01_UNR2MASK) != 0
    // clean/rr-f-a14-l01 A7 mlp-unroll2 (L01_UNR2MASK: 4 E1, 8 D1-F, 16 D1-R): the rolled chunk loop two chunks per
    // iteration (the same chunks in the same order; NJ - 1 odd: the last chunk after the loop)
    RR_LOOP for (uint j2 = 1; j2 < NJ - 1u + RR_ZL; j2 += 2u) {
        mlp_chunk(j2, false, y1q, dd, BVW RR_FPA);
        mlp_chunk(j2 + 1u, false, y1q, dd, BVW RR_FPA);
    }
    {
        mlp_chunk(NJ - 1u, false, y1q, dd, BVW RR_FPA);
#else
    RR_LOOP for (uint j = 1; j < NJ + RR_ZL; ++j) {
        const RrPf pfm = st_fetch(pMlp + min(j + 1u, NJ - 1u) * UNIT, UNIT);
        mlp_chunk(j, false, y1q, dd, BVW RR_FPA);
        if (j + 1u < NJ) st_commit(pfm, (j + 1u) & 1u, UNIT);
        st_sync();
#endif
    }
#if RR_MLA
    }
#endif

    // ---- step 9: second residual, block output y2 ---------------------------------------------------------------------
#if !RR_MLPDEF
    uint2 y2[KS];
#endif
#if RR_DEFER
    rr_negoff = true;
#endif
    {
        rrbv yh[KS];
        uint bad = 0u;
        [unroll] for (uint c3 = 0; c3 < KS; ++c3) {
            yh[c3] = bv_mad(dd[c3], BVK(m2), y1h[c3]);
        }
#if RR_MLPDEF
        [branch] if (!pa_slow) { [unroll] for (uint c0 = 0; c0 < KS; ++c0) y2[c0] = bv_q8f(yh[c0], bad); }
        else { [unroll] for (uint c = 0; c < KS; ++c) y2[c] = bv_q8s(yh[c]); }
        if (pa_slow) break;
        if (!q8_slow(bad)) break;
        pa_slow = true;
        rr_sat = true;
#else
        Q8_TRY [unroll] for (uint c0 = 0; c0 < KS; ++c0) y2[c0] = bv_q8f(yh[c0], bad);
        Q8_ELSE [unroll] for (uint c = 0; c < KS; ++c) y2[c] = bv_q8s(yh[c]); Q8_END
#endif
    }
#if RR_MLPDEF
    }
#endif
    const uint otok = (uint)(Y * Nx + X);
    if (real) {
        [unroll] for (uint c4 = 0; c4 < KS; ++c4) RR_PST64(pY2, otok * RC + 16u * c4 + 8u * rr_hf, y2[c4]);
    }

    // ---- merge (51.4, E1-E4) ----------------------------------------------------------------------------------------
#if RR_CN > 0
    {
        const uint t = 16u * wv + n;
        [unroll] for (uint c5 = 0; c5 < KS; ++c5) MY((t * RC + 16u * c5 + 8u * rr_hf) >> 3) = rr_2q(y2[c5]);
        GroupMemoryBarrierWithGroupSync();
        const uint rx = n & 3u, ry = n >> 2;
        const int Xc = cx0 + (int)rx, Yc = cy0 + (int)ry;
        const bool cin = Xc >= 0 && Xc < Mx && Yc >= 0 && Yc < My;
#if RR_COUNT   // counting build: one wave's iterations unrolled (the first wave's count)
        [unroll] for (uint ot = 0; ot < RR_CN / 16; ot += 4u) {
#else
        RR_LOOP for (uint ot = wv; ot < RR_CN / 16 + RR_ZL; ot += 4u) {
#endif
            v8f acc = bias_tile(pBm, ot);
#if RR_MRGU
            // round-2 L1a: the inner loop unrolled; MY index of (s, e) = per-lane base + compile-time offset. With
            // C = 64: zi = 32 s + 16 e + 8 rr_hf, cn = zi / 64 = s >> 1 (32 (s & 1) + 16 e + 8 rr_hf <= 56),
            // cc = 32 (s & 1) + 16 e + 8 rr_hf, so (tt * 64 + cc) >> 3 = 128 ry + 16 rx + rr_hf + 64 (cn >> 1)
            // + 8 (cn & 1) + 4 (s & 1) + 2 e: the same words, the same products in the same order.
            const uint mb = 128u * ry + 16u * rx + rr_hf;
            [unroll] for (uint s = 0; s < HID / 32; ++s) {
                uint2 b[2], a[2];
                [unroll] for (uint e = 0; e < 2; ++e) {
                    b[e] = rr_q2(MY(mb + 64u * (s >> 2) + 8u * ((s >> 1) & 1u) + 4u * (s & 1u) + 2u * e));
                    a[e] = wfrag(pWm, ot * (HID / 16) + 2u * s + e);
                }
                acc = step2l(s + 1 == HID / 32, a[0], b[0], a[1], b[1], acc);
            }
#else
            RR_LOOP for (uint s = 0; s < HID / 32 + RR_ZL; ++s) {
                uint2 b[2], a[2];
                [unroll] for (uint e = 0; e < 2; ++e) {
                    const uint zi = 32u * s + 16u * e + 8u * rr_hf;
                    const uint cn = zi / RC, cc = zi % RC;
                    const uint tt = 8u * (2u * ry + (cn >> 1)) + 2u * rx + (cn & 1u);
                    b[e] = rr_q2(MY((tt * RC + cc) >> 3));
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

    // ---- hidden state (51.4) -----------------------------------------------------------------------------------------
#if HIDDEN
    {
#if RR_HPF
        const uint2 hold = RR_PLD64(pHold, tok * 8u);
#if RR_KIND != 0
        const int xm = (X - 1) >> 1, ym = (Y - 1) >> 1;
        const int rxm = rr_refl(xm, Mx, r1), rxp = rr_refl(xm + 1, Mx, r1);
        const int rym = rr_refl(ym, My, r1), ryp = rr_refl(ym + 1, My, r1);
        const uint2 cA = RR_PLD64(pHc, (uint)(rym * Mx + rxp) * 8u), cB = RR_PLD64(pHc, (uint)(rym * Mx + rxm) * 8u);
        const uint2 cC = RR_PLD64(pHc, (uint)(ryp * Mx + rxm) * 8u), cD = RR_PLD64(pHc, (uint)(ryp * Mx + rxp) * 8u);
#endif
#endif
        v8f dh = wg_load8f(pBh, 32u * rr_hf);
        uint4 dhb;
        [unroll] for (uint s = 0; s < NS; ++s) {
            if (s + 1 < NS) dh = step2(wfrag(pWh, 2u * s), y2[2 * s], wfrag(pWh, 2u * s + 1u), y2[2 * s + 1], dh);
            else dhb = step2h(wfrag(pWh, 2u * s), y2[2 * s], wfrag(pWh, 2u * s + 1u), y2[2 * s + 1], dh);
        }
        const uint s01 = wg_ld16(pSig, (dhb.x & 0xffffu) * 2u) | (wg_ld16(pSig, (dhb.x >> 16) * 2u) << 16);
        const uint s23 = wg_ld16(pSig, (dhb.y & 0xffffu) * 2u) | (wg_ld16(pSig, (dhb.y >> 16) * 2u) << 16);
#if !RR_HPF
        const uint2 hold = RR_PLD64(pHold, tok * 8u);
#endif
        uint2 hn;
#if RR_KIND == 0
        const uint G01 = rr_fence_u(rr_b2(rr_h2(s01) * g2)), G23 = rr_fence_u(rr_b2(rr_h2(s23) * g2));
        hn.x = rr_fence_u(rr_b2(rr_h2(hold.x) * rr_h2(G01)));
        hn.y = rr_fence_u(rr_b2(rr_h2(hold.y) * rr_h2(G23))) & 0xffffu;
#else
        const half2 z2 = rr_splat_hi(s23);
        const uint omzb = rr_fence_u(rr_b2(half2(1.0, 1.0) - z2));
        const half2 omz = rr_h2(omzb);
#if !RR_HPF
        const int xm = (X - 1) >> 1, ym = (Y - 1) >> 1;
        const int rxm = rr_refl(xm, Mx, r1), rxp = rr_refl(xm + 1, Mx, r1);
        const int rym = rr_refl(ym, My, r1), ryp = rr_refl(ym + 1, My, r1);
#endif
        const half2 wxp = (X & 1) ? half2(0.25, 0.25) : half2(0.75, 0.75);
        const half2 wyp = (Y & 1) ? half2(0.25, 0.25) : half2(0.75, 0.75);
        const half2 wxm = (X & 1) ? half2(0.75, 0.75) : half2(0.25, 0.25);
        const half2 wym = (Y & 1) ? half2(0.75, 0.75) : half2(0.25, 0.25);
#if !RR_HPF
        const uint2 cA = RR_PLD64(pHc, (uint)(rym * Mx + rxp) * 8u), cB = RR_PLD64(pHc, (uint)(rym * Mx + rxm) * 8u);
        const uint2 cC = RR_PLD64(pHc, (uint)(ryp * Mx + rxm) * 8u), cD = RR_PLD64(pHc, (uint)(ryp * Mx + rxp) * 8u);
#endif
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
        if (real && rr_hf == 0u) RR_PST64(pHnew, otok * 8u, hn);
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
