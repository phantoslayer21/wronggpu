// SPDX-License-Identifier: MIT
// D0-R of spec 55 on RDNA 4 / D3D12: the window block of spec 51.3 at level 0 (C = 32, H = 2, MLP width 128) fused
// with the network's output head (55.2 Cnet, 55.3 HID / GATES / PREC / CANDW; channel numbers per the page 55 erratum,
// RR_P55_ERRATUM below). Design and measurements: ../DESIGN-55.md.
//
// One workgroup = one 8 x 8 window (window index = workgroup index, j-major, i-minor), 4 waves of 32 lanes; wave w owns
// window tokens t = 16 w + r (r = lane & 15), i.e. window rows 2w, 2w + 1. Fragment layouts of
// v_wmma_f32_16x16x16_fp8_fp8 (wave32): lane l, r = l & 15, hf = l >> 4; A = row r, K 8hf .. 8hf+7; B = column r,
// K 8hf .. 8hf+7; C / D = column r, rows 8hf .. 8hf+7. Activations are D tiles "channels x tokens": lane (r, hf) of
// channel tile ct holds channels 16ct + 8hf .. +7 of token r, which is also the B fragment of K-slice ct.
//
// Built by run_dx12.py through clean/dx12/build_dx12.py's compile_hlsl (exact arithmetic mode, README there).
// Defines: RR_HEAD_ONLY 1 = no block (y2h read from p_x as an f16 tensor [Ht][Wt][32]); RR_ZSUP 1 = the row sums Z
// read from p_zs ([window][head][64] f16, test hook of 51.8 item 1); RR_DEBUG 1 = also write y2h, Z and the f16
// values of HID / GATES / CANDW (test builds).
// RR_STRICT_STEP=1 opts into the exact binary64 spec51 matrix step, 2 = the same results on WMMA (variant X);
// default (0, variant E) remains unchanged.
// RR_ZEXACT=1 (opt-in; the RR runtime, clean/rr_f/runtime): Z = the exact sum of the 64 f16 weights rounded once to
// f16 (the reference's rule, the exact Z mode RR_Z=2 of kernels/rr_block.hlsl, same code), instead of the f32 sum.
// RR_P55_ERRATUM: the D0-R block-output channels read by the head (55.2, 55.3). 1 (default) = the page 55 erratum's
// natural (converter) numbers: gate logits y0, y1, y2, y3; HID tanh(y14), tanh(y16); GATES y4, y5, y6; PREC y7, y8, y9;
// CANDW y10, y12 (head.py head_channels). 0 = the numbers as printed on page 55 (y0, y1, y4, y5; y28, y2; y8, y9, y12;
// y13, y16, y17; y20, y24), for comparison only.
#ifndef RR_P55_ERRATUM
#define RR_P55_ERRATUM 1
#endif
#ifndef RR_ZEXACT
#define RR_ZEXACT 0
#endif
#ifndef RR_STRICT_STEP
#define RR_STRICT_STEP 0
#endif
#ifndef RR_HEAD_ONLY
#define RR_HEAD_ONLY 0
#endif
#ifndef RR_ZSUP
#define RR_ZSUP 0
#endif
#ifndef RR_FIXUP
#define RR_FIXUP 0
#endif
#ifndef RR_DEBUG
#define RR_DEBUG 0
#endif
// RR_OUT_PREP 1 (clean/rr-f-r6-tail; the runtime's head.fast): the head also writes the output stage's render records
// (kernels/rr_output_fast.hlsli rr_output_prep: A, B, W planes in the output scratch p_oscr; 0 = skipped) from the
// values it stores, with prep's operations (same bits); prep then runs as OF_PREPU (the uniform words only).
#ifndef RR_OUT_PREP
#define RR_OUT_PREP 0
#endif
// RR_HVARL 7 (opt-in, NOT exact to variant E; clean/rr-f-r8-head, tier B): the body's relaxed arithmetic (RR_VARL,
// EZMODE f32, f32 elementwise) on the default step's D0-R core and MLP: no f16 rounding inside or at the end of a
// matrix chain (Q8 straight from the f32 sums: one rounding instead of f32 -> f16 -> e4m3), the attention row sums Z
// in f32 (dot2add) with r = R(1 / Z) from the f32 Z, Dp, y1 = Dp ra + x, the MLP's down-projection chain and
// y2 = Dd m + y1 in f32, y2h = R(y2) once (the head's table lookups take f16 logits). The head itself (55.2 Cnet with
// its recurrent blend, 55.3) is unchanged.
// RR_HVARL is a bit mask: 1 = the chain roundings (q / v, P V, Dp, the down projection), 2 = Z in f32,
// 4 = Dp, y1, y2 in f32 (the elementwise steps 7 and 9); 7 = all.
#ifndef RR_HVARL
#define RR_HVARL 0
#endif
#if RR_HVARL && (RR_STRICT_STEP || RR_HEAD_ONLY)
#error RR_HVARL is a variant of the default step with the block (RR_STRICT_STEP 0, RR_HEAD_ONLY 0)
#endif
// RR_HACT (clean/rr-mp2-head H4; default 1 with RR_HVARL, 0 elsewhere: the other heads keep their text): rr_act's
// |k| written as max(k, -k) instead of the mask k & 0x7fff7fff, which the driver splits into two v_and_b16 per pair.
// The driver reads it as |k| and packs the two halves once with source modifiers (v_pack_b32_f16 d, -|k.l|, -|k.h|,
// feeding u's v_pk_fma_f16): one VALU per pair instead of two. Same bits: k = min(max(x, CLn), CL) is never NaN (the
// max returns the non-NaN operand) or infinite, so the value is -|k| as before for every k != 0; for k = +-0 it can be
// -0 where the mask gave +0, and the only reader is u = R(C2 - |k| C1) = fma(-|k|, C1, C2), which is C2 exactly for
// either zero when C2 != 0 (model F-B1: ACT_C2 = 0.412). An f16 denormal |k| (also if a modifier flushed it) gives
// |k| C1 < 2^-17 < half an ulp of C2, so u = C2 either way; k itself (v = R(k u + C3)) is not touched.
#ifndef RR_HACT
#define RR_HACT (RR_HVARL != 0)
#endif
// RR_HTRIM (clean/rr-mp2-head H3; bit mask, default 7 with RR_HVARL, 0 elsewhere): the head's fences (HF / rr_fence4:
// an opaque xor) stay only where the driver could otherwise merge two roundings: a product feeding an add (v_fma_mix /
// v_pk_fma), a multiplication chain it could regroup, a value feeding the f32 -> f16 conversion or a widening. RR_HTRIM
// drops the fences whose value can take part in no such merge (each use below says why: add -> mul, fma -> fma / mul,
// a value that only feeds integer work): 1 = the y2h fences, 2 = the 18-channel CANDW chain's T / SP / SS, 4 = Cnet's
// fma results, hh and R(1 - z). The operations and their order are unchanged, so every rounding stays where it is
// (checked in the RGA ISA: per-half f16 and f32 arithmetic counts unchanged, 16 fewer v_xor_b32, no new fused op).
#ifndef RR_HTRIM
#define RR_HTRIM (RR_HVARL ? 7 : 0)
#endif
// RR_HGRP (clean/rr-mp2-head H5; default 1 with RR_HVARL, 0 elsewhere): the scores' 8 WMMAs of a head are issued as one
// run (the four tiles' first K-halves, then their second halves) before the four conversions, instead of 2 WMMAs then
// that tile's conversion: one WMMA -> VALU switch per head instead of four, and no conversion reads a fresh accumulator.
// Each tile gets the same two WMMAs on the same operands (a WMMA output element depends only on its own A row, B column
// and C element), so the bits are the same.
#ifndef RR_HGRP
#define RR_HGRP (RR_HVARL != 0)
#endif
#if RR_HGRP && RR_STRICT_STEP
#error RR_HGRP is a variant of the default step
#endif
// RR_HCVB (clean/rr-mp2-head H6; default 1 with RR_HVARL, 0 elsewhere): RR_S9 bit 64's wave-uniform branch for the paired
// CANDW reading only (cv = sigx(d) / 2, rr_lhalf: an exp, a reciprocal and about 20 VALU per lane), not for the 18-channel
// chain: the frame's reading is a launch constant (hmode), and cv is read only when !hmode (cwa / cwb select by hmode),
// so the stored values keep their bits. (Bit 64 as a whole was measured slower in head.fast.rx: clean/rr-f-lw-fold.)
#ifndef RR_HCVB
#define RR_HCVB (RR_HVARL != 0)
#endif
#if RR_HTRIM & 1
#define RR_YF4(v) (v)
#else
#define RR_YF4(v) rr_fence4(v)
#endif
#if RR_HTRIM & 2
#define HFT2(h) (h)
#else
#define HFT2(h) HF(h)
#endif
#if RR_HTRIM & 4
#define HFT(h) (h)
#else
#define HFT(h) HF(h)
#endif
// RR_HCOMPACT 1 (default where it applies: step E and step I with the page 55 erratum numbering and the block):
// the head reads only y2h channels 0 .. 10, 12, 14 and 16, so the MLP's down projection (51.3 step 8) computes ONE
// channel tile: tile 0 with its row 11 (channel 11, unused) replaced by channel 16 (row 0 of tile 1). Every channel the
// head reads gets the same step chain (its own W2 column, b2, m and y1h value) as with two tiles, so the outputs are
// the same bits (track r3-head); y2h channels 11, 13, 15 and 17 .. 31 are not computed (RR_DEBUG's y2h output then
// holds y16 in channel 11 and zeros in 16 .. 31; channels 0 .. 10, 12, 14 are the block's values).
// RR_HEAD18 1 (clean/rr-f-flow, default here): the head can also take the 18-channel reading of D0-R's block output
// of functional erratum 55-ERRATUM-2 (Dh 0..3, GATES 4..6, PREC 7..9, CANDW from y10, y11, y12, y13 by its binary16
// chain, HID = tanh(y14, y15, y16, y17)) instead of
// page 55's paired reading (CANDW from (y10, y10, y12, y12), HID (y14, y14, y16, y16)). Evidence (results doc of the
// branch): in the converted model, D0-R's down projection columns / biases 0..17 are trained, 18..31 are not (zero
// bias, init-sized norms), so channels 11, 13, 15, 17 must have a reader. The reading is chosen per frame (bit 8 of
// the sx argument: 1 = 18-channel reading); it needs every channel, so RR_HCOMPACT is off.
#ifndef RR_HEAD18
#define RR_HEAD18 1
#endif
#if RR_HEAD18 && !RR_P55_ERRATUM
#error RR_HEAD18 needs the page 55 erratum numbering
#endif
#if !defined(RR_HCOMPACT)
#if !RR_HEAD_ONLY && RR_P55_ERRATUM && !RR_HEAD18 && (RR_STRICT_STEP == 0 || RR_STRICT_STEP == 4)
#define RR_HCOMPACT 1
#else
#define RR_HCOMPACT 0
#endif
#endif

// RR_S9 (clean/rr-f-s9-head18; bit mask, default 61 for head.fast = step E with the block, 0 elsewhere): a faster
// schedule of the SAME arithmetic (outputs bit-identical to RR_S9 0, the 7992448 head):
//   1 = the f32 -> f16 roundings of Z and r = R(1 / Z) through the hardware RNE conversion (op 0x1C imm 1,
//       v_cvt_f16_f32; INTRINSICS.md: bit-equal to wg_f2h, the operands here are positive normal f32) instead of wg_f2h;
//   2 = rr_act's |k| as one opaque 32-bit mask (opt-in: same VALU count on this driver, it adds half swaps);
//   4 = the exact MLP down-projection chain (not RR_HVARL 1/4) rounds only the channel tile 1 values the head reads
//       (elements 0, 1 = channels 16, 17 in lanes hf 0; 24, 25 in hf 1, read but unused): the other six rows of tile 1
//       carry their f32 accumulator unrounded; a WMMA output element depends only on its own A row, B column and C
//       element, so channels 0 .. 17 keep their bits (channels 18 .. 31 of y2h differ; no reader);
//   8 = the exact row sum Z (RR_ZEXACT, not RR_HVARL 2) certified: an f32 pairwise tree sum zf of the 64 nonnegative
//       weights (depth 6 including the lane exchange: |zf - Z| <= 6u Z / (1 - 6u), u = 2^-24) and the hardware RNE of
//       zf (1 -+ 2^-20); when both brackets give the same f16 (RNE is monotonic), that is R(Z); a wave with any lane
//       whose bracket straddles an f16 rounding boundary computes the exact sum as before (wave-uniform branch).
//  16 = the MLP activation's rare RR_Q8 path recomputes act(h) from the kept f16 pre-activations (same operations)
//       instead of keeping the products: the driver otherwise computed every product twice on the common path;
//  32 = the 8-bit codes of CANDW / GATES computed once for the stores and the render record (RR_OUT_PREP);
//  64 = (clean/rr-f-lw-fold) the head's two CANDW / HID readings computed only for the reading the frame uses (hmode,
//       bit 8 of the launch constant sx): the 18-channel chain (T3, the binary16 softmax of y10 .. y13) when hmode, the
//       paired sigx value cv when not. The other reading's values are never read (hv2 / cwa / cwb select by hmode), so
//       every stored value keeps its bits; the branch is wave-uniform (all lanes take it: the lane exchange inside is safe).
#ifndef RR_S9_NOFB
#define RR_S9_NOFB 0   // ISA counting only: never take the exact Z fallback (NOT exact)
#endif
#ifndef RR_S9
#if RR_STRICT_STEP == 0 && !RR_HEAD_ONLY && !RR_DEBUG && !RR_ZSUP
#define RR_S9 125
#else
#define RR_S9 0
#endif
#endif

#include "wg_platform.hlsli"

WG_ARGS_BEGIN
    uint p_x, p_wf, p_bias, p_tb, p_hold, p_hc, p_tab, p_rho;   // arena byte offsets (0 = null)
    uint p_cnet, p_hid, p_gates, p_prec, p_candw;
    uint p_dy, p_dimg, p_dz, p_zs;
    uint Wt, Ht, Wr, Hr, sx, sy, nwx;
    uint k_as, k_acl, k_acln, k_p1, k_p0, k_actcl, k_actcln, k_c1, k_c2, k_c3, k_ra, k_m, k_g;  // f16 bits
    uint k_s1;                                                                                   // f32 bits
    uint p_flag;   // RR_STRICT_STEP 4: [window][wave] u32 marks (written by every wave); RR_FIXUP reads them
    uint p_wpl;    // RR_STRICT_STEP 4: the weights' digit planes (rr_d0r_wplanes, 28 pairs x 32 lanes x 48 B + 28 x 16 B)
#if RR_OUT_PREP
    uint p_oscr, o_wh, o_hh, k_o2n, k_o2x2n, k_o4, k_o5, k_o7;   // output scratch, history size, out.K2n / K2x2n / K4 / K5 / K7 (f32)
#endif
WG_ARGS_END

#ifndef SW_LDS_BYTES
#define SW_LDS_BYTES 8192   // q / v^T exchange: [head 2][wave 4][frag 4: q ft0, q ft1, vT ft0, vT ft1][lane 32] x 8 B
#endif
#include "swin_common.hlsli"
#include "speed_swin.hlsli"

// Weight fragments (256 B each = 32 lanes x 8 B; lane l, byte i = W[16 ks + 8 (l >> 4) + i][16 nt + (l & 15)]).
#define F_Q 0u    // Wq [h][ft][ks]
#define F_V 8u    // Wv [h][ft][ks]
#define F_O 16u   // Wo [h][ct][ks]
#define F_1 24u   // W1 [nt 8][ks 2]
#define F_2 40u   // W2 [ct 2][ks 8]
// p_bias: f32 copies of bo [32] (byte 0), b1 [128] (byte 128), b2 [32] (byte 640)
// p_tab: sig of 51.4 (f16 [65536], byte 0), R(tanh) (f16 [65536], byte 131072); p_rho: rho (f32 [65536]) for this s2
#define TAB_TANH 131072u
#define TAB_EXP2 262144u    // 55-ERRATUM-2: R(2^t) for every f16 t (clean/rr-f-flow)
#define TAB_RCP16 393216u   // 55-ERRATUM-2: R(1/v) for every f16 v

uint rr_zero() { return 0u - (wg_grid_ >> 31); }           // 0 with no known bits (README "Driver exceptions")
uint rr_fence(uint v) { return v ^ rr_zero(); }
uint4 rr_fence4(uint4 v) { return v ^ rr_zero(); }
half2 H2(uint v) { return wg_as_h2(v); }
uint U2(half2 h) { return wg_h2_bits(h); }
half2 HF(half2 h) { return H2(rr_fence(U2(h))); }            // fenced f16 pair: a rounding point stays a rounding point
float FF(float f) { return asfloat(rr_fence(asuint(f))); }
uint rr_u(uint p) { return WaveReadLaneFirst(p); }       // a pointer argument read as wave-uniform (store-merge hazard)
#if RR_S9 & 1
uint rr_f2h(float x) { return wg_ags_call(0x1cu, 0u, 1u, asuint(x), 0u); }   // hardware RNE f32 -> f16 (op 0x1C imm 1)
#else
uint rr_f2h(float x) { return wg_f2h(x); }
#endif
#if RR_OUT_PREP
// rr_output.hlsl / rr_output_fast.hlsli's fence, Cf, Mq and of_d255, operation for operation
float rr_ofence(float x) { return asfloat(asuint(x) ^ (0u - (wg_grid_ >> 31))); }
float rr_ocf(float x)
{
    return rr_ofence(rr_ofence(log2(max(mad(x, asfloat(k_o4), 1.0f), asfloat(k_o5))) * asfloat(0x3e9a209bu)) * asfloat(k_o7));
}
#ifndef RR_OMQB
#define RR_OMQB 0   // clean/rr-f-a14-tail D36: 1 = the scaled check without the branch (same flag, same bits; measured slower)
#endif
float rr_omq(float a, float b, inout bool rare)
{
    const float r = rr_ofence(a * b);
#if RR_OMQB
    const float s2 = rr_ofence(a * asfloat(0x5f800000u)) * b;
    rare = or(rare, and(r == 0, and(abs(s2) >= asfloat(0x207ffff0u), abs(s2) <= asfloat(0x20800000u))));
#else
    [branch] if (r == 0) {
        const float s2 = rr_ofence(a * asfloat(0x5f800000u)) * b;
        rare = or(rare, and(abs(s2) >= asfloat(0x207ffff0u), abs(s2) <= asfloat(0x20800000u)));
    }
#endif
    return r;
}
float rr_od255(uint b)
{
    const float a = (float)b, rc = asfloat(0x3b808081u);
    const float q = a * rc;
    return mad(mad(-255.0f, q, a), rc, q);
}
#endif
half2 rr_c2(uint b) { return H2((b & 0xffffu) * 0x10001u); }

uint2 rr_frag(uint f, uint lane) { return wg_ld64(p_wf, f * 256u + lane * 8u); }
// RR_HCOMPACT: fragment f of W2's channel tile 0 with row 11 taken from tile 1's row 0 (channel 16; tile 1 = f + 8)
uint2 rr_frag_m(uint f, uint lane)
{
    const bool s = (lane & 15u) == 11u;
    return rr_frag(f + (s ? 8u : 0u), s ? (lane & 16u) : lane);
}

v8f rr_widen(uint4 p)
{
    v8f r;
    r.v[0] = f16tof32(p.x); r.v[1] = f16tof32(p.x >> 16); r.v[2] = f16tof32(p.y); r.v[3] = f16tof32(p.y >> 16);
    r.v[4] = f16tof32(p.z); r.v[5] = f16tof32(p.z >> 16); r.v[6] = f16tof32(p.w); r.v[7] = f16tof32(p.w >> 16);
    return r;
}
// the 51.1 matrix-step rounding of an f32 accumulator, back in f32 (the start value of the next step, or Q8's input)
v8f rr_r16(v8f a) { return rr_widen(wg_to_f16_ags(a)); }
// rr_r16 at the D0-R core's chain roundings, none with RR_HVARL (a macro: the default kernels' text is unchanged)
#if RR_HVARL & 1
#define RR_HR16(a) (a)
#else
#define RR_HR16(a) rr_r16(a)
#endif

#if RR_STRICT_STEP == 1
#include "../../kernels/rr_exact_math.hlsli"
v8f rr_head_step(uint2 a0, uint2 b0, uint2 a1, uint2 b1, v8f c)
{
    return rr_exact_step2(a0, b0, a1, b1, c);
}
#elif RR_STRICT_STEP == 2
// The same exact step (old 51.1 rule: exact sum, one rounding) on WMMA: variant X of the body with its fast path for
// exact sums (rr_step.hlsli RR_STEP 3, the input-stage MLP's step; proof DESIGN-51 7.2 / probe P1). Bit-identical to
// RR_STRICT_STEP 1 (checked by exact_head.py against the binary64 step on random and adversarial windows).
uint4 rr_h16(v8f d) { return wg_to_f16_ags(d); }
#define RR_STEP 3
#include "../../kernels/rr_step.hlsli"
// The strict step's special-value rules on top: any NaN -> +NaN (0x7e00), an exact zero -> +0 (branchless selects).
v8f rr_head_step(uint2 a0, uint2 b0, uint2 a1, uint2 b1, v8f c)
{
    const v8f r = step2(a0, b0, a1, b1, c);
    v8f o;
    [unroll] for (uint k = 0; k < 8; ++k) {
        const uint u = asuint(r.v[k]), m = u & 0x7fffffffu;
        o.v[k] = asfloat(m > 0x7f800000u ? 0x7fc00000u : (m == 0u ? 0u : u));
    }
    return o;
}
#elif RR_STRICT_STEP == 4
// The same exact step on INTEGER WMMA (variant I of the body, kernels/rr_step_i.hlsli, proof in its header and in
// results/2026-10-21-body-levers.md section 3): both operand blocks decomposed per wave (ri_act, wave-uniform scale),
// N = the exact integer sum (v_wmma_i32_16x16x16_iu8), ONE rounding R(c + N 2^G). A step it cannot take (a NaN code,
// the int32 bound, G > -8) marks the wave (rr_yflag); the window is then recomputed by the strict binary64 kernel
// (RR_STRICT_STEP 1, RR_FIXUP 1) from p_flag. On top, the strict step's NaN rule (a NaN start value -> +NaN 0x7e00).
uint4 rr_h16(v8f d) { return wg_to_f16_ags(d); }
v8f rr_d8(uint2 q) { return wg_v8f_frag(wg_ags_convert(wg_frag2(q), WG_AGS_T_FP8_B, WG_AGS_T_F32_ACC)); }
float rr_fence(float x) { return asfloat(rr_fence(asuint(x))); }   // rr_step_i fences f32 values (rr_common's form)
#include "../../kernels/rr_step_i.hlsli"
v8f rr_step_ab(RrAB A, RrAB Bb, v8f c)
{
    uint4 h = ri_step_aa(A, Bb, c);
    // f16 NaN (either half) -> 0x7e00 per half, branchless
    const uint4 lo = h & 0x7fffu, hi = (h >> 16) & 0x7fffu;
    const uint4 ml = select(lo > 0x7c00u, 0xffffu, 0u), mh = select(hi > 0x7c00u, 0xffff0000u, 0u);
    h = (h & ~(ml | mh)) | (0x7e007e00u & (ml | mh));
    return rr_widen(h);
}
// Weight fragment pair i (fragments 2i, 2i + 1 of p_wf) decomposed once per frame by rr_d0r_wplanes: the same RrAB
// as ri_act on the fragments (ri_act is wave-local and the weights are the same for every window).
#define RR_WPL_STATS (32u * 1536u)   // 28 pairs + 4 merged W2 pairs (RR_HCOMPACT: 28 + j)
RrAB rr_wpl(uint i, uint lane)
{
    const uint pb = rr_u(p_wpl), o = i * 1536u + lane * 48u;
    const uint4 q0 = wg_ld128(pb, o), q1 = wg_ld128(pb, o + 16u), q2 = wg_ld128(pb, o + 32u);
    const uint4 st = wg_ld128(pb, RR_WPL_STATS + i * 16u);
    RrAB r;
    r.d0.p0 = q0.xy; r.d0.p1 = q0.zw; r.d0.p2 = q1.xy; r.d1.p0 = q1.zw; r.d1.p1 = q2.xy; r.d1.p2 = q2.zw;
    r.beta = (int)WaveReadLaneFirst(st.x);
    const uint fl = WaveReadLaneFirst(st.y);
    r.d3 = (fl & 1u) != 0u;
    r.vmax = asfloat(WaveReadLaneFirst(st.z));
    r.l1 = asfloat(WaveReadLaneFirst(st.w));
    rr_yflag |= (fl >> 1) & 1u;                    // a NaN code among the weights: every window is marked
    return r;
}
// RR_ACT2 1 (default): ri_act with the third digit plane computed only when the block needs it (d3, wave-uniform).
// For d3 false every |v'| < 2^14, so ri_digits_d's t2 = trunc(v / 2^14) is 0 (or -0), r = v and the plane-2 bytes are
// 0: rr_digits2 returns the same planes without that arithmetic.
#ifndef RR_ACT2
#define RR_ACT2 1
#endif
RrDg rr_digits2(v8f d, float s)
{
    int l[8], m[8];
    [unroll] for (uint i = 0; i < 8; ++i) {
        const float v = d.v[i] * s;
        const float t1 = trunc(v * (1.0f / 128.0f));
        l[i] = (int)mad(t1, -128.0f, v);
        m[i] = (int)t1;
    }
    RrDg g;
    g.p0 = uint2(ri_pack(l[0], l[1], l[2], l[3]), ri_pack(l[4], l[5], l[6], l[7]));
    g.p1 = uint2(ri_pack(m[0], m[1], m[2], m[3]), ri_pack(m[4], m[5], m[6], m[7]));
    g.p2 = uint2(0u, 0u);
    return g;
}
RrAB rr_act2(uint2 f0, uint2 f1)
{
    const v8f d0 = rr_d8(f0), d1 = rr_d8(f1);
    uint mx = 0u, mn = 0xffffffffu;
    float sa = 0.0f;
    [unroll] for (uint i = 0; i < 8; ++i) {
        const uint a = asuint(d0.v[i]) & 0x7fffffffu, b = asuint(d1.v[i]) & 0x7fffffffu;
        mx = max(mx, max(a, b));
        mn = min(mn, min(a - 1u, b - 1u));
        sa = sa + abs(d0.v[i]) + abs(d1.v[i]);
    }
    const float rsum = WaveActiveMax(sa + WaveReadLaneAt(sa, WaveGetLaneIndex() ^ 16u));
    const uint nanb = rr_nan_bytes_i(f0.x) | rr_nan_bytes_i(f0.y) | rr_nan_bytes_i(f1.x) | rr_nan_bytes_i(f1.y);
    rr_yflag |= WaveActiveAnyTrue(nanb != 0u) ? 1u : 0u;
    mx = WaveActiveMax(mx);
    mn = WaveActiveMin(mn);
    RrAB r;
    const bool any = mn != 0xffffffffu;
    const int emax = max((int)(mx >> 23) - 127 + 7, 1);
    r.beta = any ? max((int)((mn + 1u) >> 23) - 127 + 7, 1) : 1;
    r.d3 = any && emax - r.beta > 10;
    r.vmax = any ? asfloat(mx) * ldexp(1.0f, (float)(10 - r.beta)) : 0.0f;
    r.l1 = rsum * ldexp(1.0f, (float)(10 - r.beta));
    const float sc = ldexp(1.0f, (float)(10 - r.beta));
    [branch] if (r.d3) {
        r.d0 = ri_digits_d(d0, sc);
        r.d1 = ri_digits_d(d1, sc);
    } else {
        r.d0 = rr_digits2(d0, sc);
        r.d1 = rr_digits2(d1, sc);
    }
    return r;
}
// RR_SHARE 1 (default): the attention's activation blocks are decomposed once per workgroup instead of once per wave:
// wave w decomposes its own q block (every wave's key tile w) and V^T block w (2 f1 + half) and puts the planes and
// statistics into groupshared memory; the waves read them back for the scores (AK) and P V (AV0 / AV1). ri_act is a
// function of the block's codes alone, and the codes are the same in every wave (the q / v^T exchange), so every wave
// sees the RrAB it would have computed. A NaN code marks the window through the decomposing wave's flag word.
#ifndef RR_SHARE
#define RR_SHARE 1
#endif
#if RR_SHARE
groupshared uint4 rr_shp[8u * 32u * 3u];   // [slot: q of wave 0..3, V^T block 0..3][lane][3]
groupshared uint4 rr_shs[8];
void rr_put_ab(uint slot, uint lane, RrAB a)
{
    const uint o = (slot * 32u + lane) * 3u;
    rr_shp[o] = uint4(a.d0.p0, a.d0.p1);
    rr_shp[o + 1u] = uint4(a.d0.p2, a.d1.p0);
    rr_shp[o + 2u] = uint4(a.d1.p1, a.d1.p2);
    rr_shs[slot] = uint4((uint)a.beta, a.d3 ? 1u : 0u, asuint(a.vmax), asuint(a.l1));   // wave-uniform values
}
RrAB rr_get_ab(uint slot, uint lane)
{
    const uint o = (slot * 32u + lane) * 3u;
    const uint4 q0 = rr_shp[o], q1 = rr_shp[o + 1u], q2 = rr_shp[o + 2u], st = rr_shs[slot];
    RrAB r;
    r.d0.p0 = q0.xy; r.d0.p1 = q0.zw; r.d0.p2 = q1.xy; r.d1.p0 = q1.zw; r.d1.p1 = q2.xy; r.d1.p2 = q2.zw;
    r.beta = (int)WaveReadLaneFirst(st.x);
    r.d3 = WaveReadLaneFirst(st.y) != 0u;
    r.vmax = asfloat(WaveReadLaneFirst(st.z));
    r.l1 = asfloat(WaveReadLaneFirst(st.w));
    return r;
}
#endif
#define RR_HSTEP(AA, BB, a0, b0, a1, b1, c) rr_step_ab(AA, BB, c)
#if RR_ACT2
#define RR_AB(name, f0, f1) const RrAB name = rr_act2(f0, f1);
#else
#define RR_AB(name, f0, f1) const RrAB name = ri_act(f0, f1);
#endif
#define RR_WAB(i) rr_wpl(i, lane)
#endif
#if RR_STRICT_STEP != 4
#define RR_HSTEP(AA, BB, a0, b0, a1, b1, c) rr_head_step(a0, b0, a1, b1, c)
#define RR_AB(name, f0, f1)
#endif

// Border rule of 51.6 for extents >= 8 (level 0 and level 1 here: Wt, Ht >= 32): the single reflection.
int rr_refl(int v, int n) { const int a = abs(v); return a >= n ? 2 * (n - 1) - a : a; }

// Q8 of 51.1 for groups of tiles of f16 values held in f32: saturation at +-448, RNE; one wave-uniform test skips the
// clamp when no value of the wave needs it (speed_swin.hlsli; NaN -> 0xff, see DESIGN-55 section 4).
#define RR_Q8(N, src, dst)                                                                                  \
    {                                                                                                       \
        float am_ = 0.0f;                                                                                   \
        [unroll] for (uint qi_ = 0; qi_ < N; ++qi_) am_ = sw_absmax8(am_, src[qi_]);                         \
        if (sw_in448(am_)) { [unroll] for (uint qj_ = 0; qj_ < N; ++qj_) dst[qj_] = wg_pack_q8_nc_ags(src[qj_]); } \
        else { [unroll] for (uint qk_ = 0; qk_ < N; ++qk_) dst[qk_] = sw_pack_q8_v2(src[qk_]); }            \
    }

// 51.3 step 3 on a pair of f16 scores: u = R(s A_SCALE), clamp (a NaN gives A_CLAMPn), p = R(c R(A_P1 - c c) + A_P0),
// w = 2^(n-15) (1 + y - n), y = 32 (p - 1): for p in [1 + 1/32, 1 + 31/32) the f16 bits of w are (bits(p) << 5) & 0x7fe0.
uint rr_attw(uint sh, half2 as2, half2 acln2, half2 acl2, half2 p12, half2 p02)
{
    const half2 u = H2(sh) * as2;
    const half2 c = min(max(u, acln2), acl2);
    const half2 inner = mad(-c, c, p12);
    const half2 p = mad(c, inner, p02);
    // per half (bits(p) << 5) & 0x7fe0 as a packed 16-bit shift and one opaque 32-bit mask (the 32-bit shift moves no
    // bit across the halves that survives the mask, so the bits are the same for every p; the driver otherwise splits
    // the mask into 16-bit pieces: 10 -> 7 VALU per pair)
    const uint16_t2 pb = asuint16(p) << (uint16_t)5;
    return ((uint)pb.x | ((uint)pb.y << 16)) & (0x7fe07fe0u & ~rr_zero());
}
// Q8 of 8 packed f16 values known finite and within +-448 (P) through the driver's f16 -> fp8 conversion: the f16
// product feeding it needs no fence there (the driver does not fold it into this conversion; checked in the ISA:
// no v_fma_mix).
uint2 rr_q8h_small(uint4 h)
{
    wg_frag f;
    f.v[0] = h.x; f.v[1] = h.y; f.v[2] = h.z; f.v[3] = h.w;
    [unroll] for (uint i = 4; i < 8; ++i) f.v[i] = 0u;
    const wg_frag q = wg_ags_convert(f, WG_AGS_T_F16_ACC, WG_AGS_T_FP8_B);
    return uint2(q.v[0], q.v[1]);
}
// The same conversion for any f16 values, with RR_Q8's result: the unclamped conversion equals RR_Q8 for every finite
// |x| <= 464 (448 < |x| <= 464 rounds to +-448 = the clamp), and every other value gives a code 0x7f / 0xff, which
// marks the wave (bad: bit 7 of a byte set when its low 7 bits are 0x7f) for the RR_Q8 path on fenced widened values.
uint2 rr_q8h(uint4 h, inout uint bad)
{
    const uint2 q = rr_q8h_small(h);
    bad |= ((q.x & 0x7f7f7f7fu) + 0x01010101u) | ((q.y & 0x7f7f7f7fu) + 0x01010101u);
    return q;
}

// 51.3 step 8 activation on a pair: k = clamp, u = R(C2 - |k| C1), v = R(k u + C3), act = R(x v)
uint rr_act(uint xh, half2 cln2, half2 cl2, half2 c12, half2 c22, half2 c32)
{
    const half2 x = H2(xh);
    const half2 k = min(max(x, cln2), cl2);
#if RR_HACT
    const half2 ak = max(k, -k);
#elif RR_S9 & 2
    const half2 ak = H2(U2(k) & (0x7fff7fffu & ~rr_zero()));
#else
    const half2 ak = H2(U2(k) & 0x7fff7fffu);
#endif
    const half2 u = mad(-ak, c12, c22);
    const half2 v = mad(k, u, c32);
    return U2(x * v);   // consumed by rr_q8h (no fold into that conversion) or fenced before a widening
}

// sigx(dd) / 2 = 0.5 / (1 + e^-dd) in f32 (CANDW, 55.3: within about 2 f32 ulp before the rounding to f16).
float rr_lhalf(float dd)
{
    const float t = min(max(-dd, -100.0f), 87.0f);   // e^t stays finite; beyond, the f16 result is 1/2 or 0 anyway
    const float L = asfloat(0x3fb8aa3bu), Llo = asfloat(0x32a57060u), LN2 = asfloat(0x3f317218u);
    const float th = t * L;
    const float tl = mad(t, L, -th) + t * Llo;        // t log2(e) = th + tl
    const float e0 = exp2(th);
    const float e = mad(e0, tl * LN2, e0);
    const float den = 1.0f + e;
    float y = rcp(den);
    y = mad(mad(-den, y, 1.0f), y, y);
    const float v = 0.5f * y;
    return isnan(dd) ? dd : v;
}

// 8-bit stores of 55.3: unsigned normalised (NaN -> 0 by the max), HID (page 53.3 as quoted on 55.3)
uint rr_u8(uint h) { const float c = min(max(f16tof32(h), 0.0f), 1.0f); return (uint)round(c * 255.0f); }
uint rr_s8(uint h)
{
    const float f = f16tof32(h);
    const float c = isnan(f) ? 0.0f : min(max(f, -1.0f), 1.0f);
    return ((uint)(int)round(c * 127.0f)) & 0xffu;
}

WG_KERNEL(rr_d0r_head)
{
    WG_SETUP();
    const uint tid = wg_lane_id(), lane = tid & 31u, r = lane & 15u, hf = lane >> 4;
    const uint wv = (uint)wg_uniform((int)(tid >> 5));
    const uint win = wg_block_id();
    const uint wj = win / nwx, wi = win - wj * nwx;
    const int X = (int)(8u * wi + (r & 7u)) - (int)(sx & 0xffu);
    const bool hmode = RR_HEAD18 && ((sx >> 8) & 1u) != 0u;   // 18-channel reading (RR_HEAD18)
    const int Y = (int)(8u * wj + 2u * wv + (r >> 3)) - (int)sy;
    const int rx = rr_refl(X, (int)Wt), ry = rr_refl(Y, (int)Ht);
    const uint tok = (uint)ry * Wt + (uint)rx;
#if RR_FIXUP
    {   // only the windows marked by the integer pass (wave-uniform: one window per workgroup)
        const uint4 fl = wg_ld128(rr_u(p_flag), win * 16u);
        if ((fl.x | fl.y | fl.z | fl.w) == 0u) return;
    }
#endif

    uint4 y2h[2];
#if RR_HEAD_ONLY
    y2h[0] = wg_ld128(p_x, tok * 64u + 16u * hf);
    y2h[1] = wg_ld128(p_x, tok * 64u + 32u + 16u * hf);
#else
    // ---------------------------------------------------------------- 51.3 step 1: q_h, v_h (both heads)
    const uint2 xb0 = wg_ld64(p_x, tok * 32u + 8u * hf), xb1 = wg_ld64(p_x, tok * 32u + 16u + 8u * hf);
    uint2 fq[8], fv[8];
    [unroll] for (uint i0 = 0; i0 < 8; ++i0) { fq[i0] = rr_frag(F_Q + i0, lane); fv[i0] = rr_frag(F_V + i0, lane); }
    RR_AB(AX, xb0, xb1)
    v8f qvt[8];   // per head h: [4h] q ft0, [4h+1] q ft1, [4h+2] vT ft0, [4h+3] vT ft1
    [unroll] for (uint h0 = 0; h0 < 2; ++h0)
        [unroll] for (uint f0 = 0; f0 < 2; ++f0) {
            const uint b = (h0 * 2u + f0) * 2u;
#if RR_STRICT_STEP
            v8f a = RR_HSTEP(RR_WAB(b >> 1), AX, fq[b], xb0, fq[b + 1u], xb1, sw_zero8());
#else
            v8f a = wg_wmma_fp8(fq[b], xb0, sw_zero8());
            a = wg_wmma_fp8(fq[b + 1u], xb1, a);
#endif
#if RR_STRICT_STEP
            v8f c = RR_HSTEP(AX, RR_WAB(4u + (b >> 1)), xb0, fv[b], xb1, fv[b + 1u], sw_zero8());
#else
            v8f c = wg_wmma_fp8(xb0, fv[b], sw_zero8());
            c = wg_wmma_fp8(xb1, fv[b + 1u], c);
#endif
            qvt[4u * h0 + f0] = RR_HR16(a);
            qvt[4u * h0 + 2u + f0] = RR_HR16(c);
        }
    uint2 qv[8];
    RR_Q8(8u, qvt, qv);
    [unroll] for (uint s0 = 0; s0 < 8; ++s0) SW_LDS_PUT((((s0 >> 2) * 4u + wv) * 4u + (s0 & 3u)) * 32u + lane, qv[s0]);
    wg_barrier_lds();

    // ---------------------------------------------------------------- steps 2-5 per head
    const half2 as2 = rr_c2(k_as), acl2 = rr_c2(k_acl), acln2 = rr_c2(k_acln), p12 = rr_c2(k_p1), p02 = rr_c2(k_p0);
    const half2 one2 = H2(0x3c003c00u);
    v8f ot[4];    // O_h tiles before Q8: [2h + ft]
    [unroll] for (uint h = 0; h < 2; ++h) {
        uint2 kq[4][2], vt[4][2];
        v8f bc[4];
        [unroll] for (uint J = 0; J < 4; ++J) {
            const uint e = ((h * 4u + J) * 4u) * 32u + lane;
            kq[J][0] = SW_LDS_GET(e); kq[J][1] = SW_LDS_GET(e + 32u);
            vt[J][0] = SW_LDS_GET(e + 64u); vt[J][1] = SW_LDS_GET(e + 96u);
            bc[J] = wg_load8f(p_tb, (((h * 4u + wv) * 4u + J) * 32u + lane) * 32u);
        }
        // scores (keys of tile J as rows, the wave's queries as columns), start value = the score bias, 51.3 step 2
        RR_AB(AQ, qv[4u * h], qv[4u * h + 1u])
#if RR_STRICT_STEP == 4 && RR_SHARE
        {
            if (h == 1u) wg_barrier_lds();                // head 0's planes are read by every wave
            const bool fh = (wv & 2u) != 0u, hh = (wv & 1u) != 0u;     // V^T block wv: f1 = wv >> 1, half = wv & 1
            const uint2 va = hh ? (fh ? vt[2][1] : vt[2][0]) : (fh ? vt[0][1] : vt[0][0]);
            const uint2 vb = hh ? (fh ? vt[3][1] : vt[3][0]) : (fh ? vt[1][1] : vt[1][0]);
            RR_AB(AVW, va, vb)
            rr_put_ab(wv, lane, AQ);
            rr_put_ab(4u + wv, lane, AVW);
            wg_barrier_lds();
        }
#endif
        uint4 w4[4];
#if RR_HGRP
        v8f s4[4];
        [unroll] for (uint J7 = 0; J7 < 4; ++J7) s4[J7] = wg_wmma_fp8(kq[J7][0], qv[4u * h], bc[J7]);
        [unroll] for (uint J8 = 0; J8 < 4; ++J8) s4[J8] = wg_wmma_fp8(kq[J8][1], qv[4u * h + 1u], s4[J8]);
        [unroll] for (uint J9 = 0; J9 < 4; ++J9) {
            const uint4 sh = wg_to_f16_ags(s4[J9]);
            w4[J9] = uint4(rr_attw(sh.x, as2, acln2, acl2, p12, p02), rr_attw(sh.y, as2, acln2, acl2, p12, p02),
                           rr_attw(sh.z, as2, acln2, acl2, p12, p02), rr_attw(sh.w, as2, acln2, acl2, p12, p02));
        }
#else
        [unroll] for (uint J1 = 0; J1 < 4; ++J1) {
#if RR_STRICT_STEP == 4 && RR_SHARE
            const RrAB AK = rr_get_ab(J1, lane);
#else
            RR_AB(AK, kq[J1][0], kq[J1][1])
#endif
#if RR_STRICT_STEP
            v8f s = RR_HSTEP(AK, AQ, kq[J1][0], qv[4u * h], kq[J1][1], qv[4u * h + 1u], bc[J1]);
#else
            v8f s = wg_wmma_fp8(kq[J1][0], qv[4u * h], bc[J1]);
            s = wg_wmma_fp8(kq[J1][1], qv[4u * h + 1u], s);
#endif
            const uint4 sh = wg_to_f16_ags(s);
            w4[J1] = uint4(rr_attw(sh.x, as2, acln2, acl2, p12, p02), rr_attw(sh.y, as2, acln2, acl2, p12, p02),
                           rr_attw(sh.z, as2, acln2, acl2, p12, p02), rr_attw(sh.w, as2, acln2, acl2, p12, p02));
        }
#endif
        // step 4: Z in f32 (the spec's accepted form), r = R(1 / Z), P = Q8(R(w r))
        float z = 0.0f;
        [unroll] for (uint J2 = 0; J2 < 4; ++J2) {
            z = dot2add(H2(w4[J2].x), one2, z); z = dot2add(H2(w4[J2].y), one2, z);
            z = dot2add(H2(w4[J2].z), one2, z); z = dot2add(H2(w4[J2].w), one2, z);
        }
        z = z + wg_xor16(z);
#if RR_ZSUP
        const uint zh = wg_ld16(p_zs, ((win * 2u + h) * 64u + 16u * wv + r) * 2u);
#elif RR_ZEXACT && !(RR_HVARL & 2)
#if RR_S9 & 8
        uint zh;
        {   // RR_S9 8: pairwise tree (16 pair sums, 8, 4, 2, 1, the lane exchange), then the bracket check
            float t8[8];
            [unroll] for (uint J6 = 0; J6 < 4; ++J6) {
                t8[2u * J6] = (f16tof32(w4[J6].x) + f16tof32(w4[J6].x >> 16)) + (f16tof32(w4[J6].y) + f16tof32(w4[J6].y >> 16));
                t8[2u * J6 + 1u] = (f16tof32(w4[J6].z) + f16tof32(w4[J6].z >> 16)) + (f16tof32(w4[J6].w) + f16tof32(w4[J6].w >> 16));
            }
            const float t4a = t8[0] + t8[1], t4b = t8[2] + t8[3], t4c = t8[4] + t8[5], t4d = t8[6] + t8[7];
            const float tl = (t4a + t4b) + (t4c + t4d);
            const float zf = tl + wg_xor16(tl);
            const uint hlo = rr_f2h(zf * asfloat(0x3f7ffff0u)), hhi = rr_f2h(zf * asfloat(0x3f800008u));   // 1 -+ 2^-20
            zh = hlo;
            [branch] if (WaveActiveAnyTrue(hlo != hhi) && !RR_S9_NOFB) {
#endif
        float hs = 0.0f, ls = 0.0f;                          // exact: w = hi / 8 + lo, both sums exact in f32
        [unroll] for (uint J4 = 0; J4 < 4; ++J4) {
            const uint ww[4] = {w4[J4].x, w4[J4].y, w4[J4].z, w4[J4].w};
            [unroll] for (uint k4 = 0; k4 < 4; ++k4) {
                [unroll] for (uint e4 = 0; e4 < 2; ++e4) {
                    const float w = f16tof32(ww[k4] >> (16u * e4));
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
#if RR_S9 & 8
                zh = rr_f2h(asfloat((zur & 0x7fffffffu) >= 0x7f800000u ? zur : wg_round_odd(zr, ze)));
            }
        }
#else
        const uint zh = rr_f2h(asfloat((zur & 0x7fffffffu) >= 0x7f800000u ? zur : wg_round_odd(zr, ze)));
#endif
#else
        const uint zh = wg_f2h(z);
#endif
#if RR_DEBUG
        if (hf == 0u) wg_st16(rr_u(p_dz), ((win * 2u + h) * 64u + 16u * wv + r) * 2u, zh);
#endif
#if RR_HVARL & 2
        const uint rh = rr_f2h(rcp(z));
#else
        const uint rh = rr_f2h(rcp(f16tof32(zh)));
#endif
        const half2 r2 = H2(rh * 0x10001u);
        uint2 pf[4];
        [unroll] for (uint J3 = 0; J3 < 4; ++J3) {
            const uint4 pw = uint4(U2(H2(w4[J3].x) * r2), U2(H2(w4[J3].y) * r2), U2(H2(w4[J3].z) * r2),
                                   U2(H2(w4[J3].w) * r2));
            pf[J3] = rr_q8h_small(pw);   // P <= 1 + 2^-11: no saturation
        }
        // step 5: O_h = Q8(MM(P; v_h; 0)), keys 0-31 then 32-63
        RR_AB(P01, pf[0], pf[1])
        RR_AB(P23, pf[2], pf[3])
        [unroll] for (uint f1 = 0; f1 < 2; ++f1) {
#if RR_STRICT_STEP == 4 && RR_SHARE
            const RrAB AV0 = rr_get_ab(4u + 2u * f1, lane);
#else
            RR_AB(AV0, vt[0][f1], vt[1][f1])
#endif
#if RR_STRICT_STEP
            v8f a = RR_HSTEP(AV0, P01, vt[0][f1], pf[0], vt[1][f1], pf[1], sw_zero8());
#else
            v8f a = wg_wmma_fp8(vt[0][f1], pf[0], sw_zero8());
            a = wg_wmma_fp8(vt[1][f1], pf[1], a);
#endif
            a = RR_HR16(a);
#if RR_STRICT_STEP == 4 && RR_SHARE
            const RrAB AV1 = rr_get_ab(5u + 2u * f1, lane);
#else
            RR_AB(AV1, vt[2][f1], vt[3][f1])
#endif
#if RR_STRICT_STEP
            a = RR_HSTEP(AV1, P23, vt[2][f1], pf[2], vt[3][f1], pf[3], a);
#else
            a = wg_wmma_fp8(vt[2][f1], pf[2], a);
            a = wg_wmma_fp8(vt[3][f1], pf[3], a);
#endif
            ot[2u * h + f1] = RR_HR16(a);
        }
    }
    uint2 of[4];
    RR_Q8(4u, ot, of);

    // ---------------------------------------------------------------- step 6: Dp, chain over the heads, start bo
    uint2 fo[8];
    RR_AB(O01, of[0], of[1])
    RR_AB(O23, of[2], of[3])
    [unroll] for (uint i1 = 0; i1 < 8; ++i1) fo[i1] = rr_frag(F_O + i1, lane);
    uint4 dp[2];
#if RR_HVARL & 4
    v8f dpf[2];
#endif
    [unroll] for (uint c0 = 0; c0 < 2; ++c0) {
        v8f a = wg_load8f(p_bias, (16u * c0 + 8u * hf) * 4u);
#if RR_STRICT_STEP
        a = RR_HSTEP(RR_WAB(8u + c0), O01, fo[c0 * 2u], of[0], fo[c0 * 2u + 1u], of[1], a);
#else
        a = wg_wmma_fp8(fo[c0 * 2u], of[0], a);
        a = wg_wmma_fp8(fo[c0 * 2u + 1u], of[1], a);
#endif
        a = RR_HR16(a);
#if RR_STRICT_STEP
        a = RR_HSTEP(RR_WAB(10u + c0), O23, fo[4u + c0 * 2u], of[2], fo[4u + c0 * 2u + 1u], of[3], a);
#else
        a = wg_wmma_fp8(fo[4u + c0 * 2u], of[2], a);
        a = wg_wmma_fp8(fo[4u + c0 * 2u + 1u], of[3], a);
#endif
#if RR_HVARL & 4
        dpf[c0] = a;
#else
        dp[c0] = wg_to_f16_ags(a);
#endif
    }
    // ---------------------------------------------------------------- step 7: y1h = R(Dp ra + d8(x)), y1q
    const half2 ra2 = rr_c2(k_ra), m2 = rr_c2(k_m);
    uint4 y1h[2];
#if RR_HVARL & 4
    uint2 y1q[2];
    v8f y1f[2];
    {
        const float ra = f16tof32(k_ra & 0xffffu);
        const v8f x0 = sw_unpack_fp8(xb0), x1 = sw_unpack_fp8(xb1);   // exact
        [unroll] for (uint i5 = 0; i5 < 8; ++i5) {
            y1f[0].v[i5] = mad(dpf[0].v[i5], ra, x0.v[i5]);
            y1f[1].v[i5] = mad(dpf[1].v[i5], ra, x1.v[i5]);
        }
        RR_Q8(2u, y1f, y1q);
    }
#else
    {
        const uint4 xr0 = wg_to_f16_ags(sw_unpack_fp8(xb0)), xr1 = wg_to_f16_ags(sw_unpack_fp8(xb1));   // exact
        y1h[0] = uint4(U2(mad(H2(dp[0].x), ra2, H2(xr0.x))), U2(mad(H2(dp[0].y), ra2, H2(xr0.y))),
                       U2(mad(H2(dp[0].z), ra2, H2(xr0.z))), U2(mad(H2(dp[0].w), ra2, H2(xr0.w))));
        y1h[1] = uint4(U2(mad(H2(dp[1].x), ra2, H2(xr1.x))), U2(mad(H2(dp[1].y), ra2, H2(xr1.y))),
                       U2(mad(H2(dp[1].z), ra2, H2(xr1.z))), U2(mad(H2(dp[1].w), ra2, H2(xr1.w))));
    }
    uint2 y1q[2];
    {
        v8f t[2];
        t[0] = rr_widen(rr_fence4(y1h[0]));   // fenced: the widening must see the rounded f16 values
        t[1] = rr_widen(rr_fence4(y1h[1]));
        RR_Q8(2u, t, y1q);
    }
#endif
    // ---------------------------------------------------------------- step 8: MLP in 4 chunks of 32
    const half2 cl2 = rr_c2(k_actcl), cln2 = rr_c2(k_actcln), c12 = rr_c2(k_c1), c22 = rr_c2(k_c2), c32 = rr_c2(k_c3);
    uint4 dd[2];
#if RR_HVARL & 5
    v8f ddf[2];
#endif
#define RR_T1P ((RR_S9 & 4) && !(RR_HVARL & 5) && !RR_HCOMPACT)
#if RR_T1P
    v8f d1f = sw_zero8();   // RR_S9 4: tile 1's accumulator; only elements 0, 1 go through the f16 rounding
#endif
    uint2 w1[4], w2[4];
    RR_AB(AY, y1q[0], y1q[1])
#if RR_HCOMPACT
    [unroll] for (uint i2 = 0; i2 < 4; ++i2) { w1[i2] = rr_frag(F_1 + i2, lane); w2[i2] = rr_frag_m(F_2 + (i2 & 1u), lane); }
#else
    [unroll] for (uint i2 = 0; i2 < 4; ++i2) { w1[i2] = rr_frag(F_1 + i2, lane); w2[i2] = rr_frag(F_2 + (i2 >> 1) * 8u + (i2 & 1u), lane); }
#endif
    [unroll] for (uint j = 0; j < 4; ++j) {
        uint2 n1[4] = { uint2(0, 0), uint2(0, 0), uint2(0, 0), uint2(0, 0) }, n2[4] = n1;
        if (j < 3u) {
            [unroll] for (uint i3 = 0; i3 < 4; ++i3) {
                n1[i3] = rr_frag(F_1 + 4u * (j + 1u) + i3, lane);
#if RR_HCOMPACT
                n2[i3] = rr_frag_m(F_2 + 2u * (j + 1u) + (i3 & 1u), lane);
#else
                n2[i3] = rr_frag(F_2 + (i3 >> 1) * 8u + 2u * (j + 1u) + (i3 & 1u), lane);
#endif
            }
        }
        v8f du[2];
        uint4 ah[2];
#if RR_S9 & 16
        uint4 hh[2];
#endif
        uint2 au[2];
        uint abad = 0u;
        [unroll] for (uint k0 = 0; k0 < 2; ++k0) {
            const uint nt = 2u * j + k0;
            v8f a = wg_load8f(p_bias, 128u + (16u * nt + 8u * hf) * 4u);
#if RR_STRICT_STEP
            a = RR_HSTEP(RR_WAB(12u + 2u * j + k0), AY, w1[2u * k0], y1q[0], w1[2u * k0 + 1u], y1q[1], a);
#else
            a = wg_wmma_fp8(w1[2u * k0], y1q[0], a);
            a = wg_wmma_fp8(w1[2u * k0 + 1u], y1q[1], a);
#endif
            const uint4 h = wg_to_f16_ags(a);
#if RR_S9 & 16
            hh[k0] = h;
#endif
            ah[k0] = uint4(rr_act(h.x, cln2, cl2, c12, c22, c32), rr_act(h.y, cln2, cl2, c12, c22, c32),
                           rr_act(h.z, cln2, cl2, c12, c22, c32), rr_act(h.w, cln2, cl2, c12, c22, c32));
            au[k0] = rr_q8h(ah[k0], abad);
        }
        if (WaveActiveAnyTrue((abad & 0x80808080u) != 0u) && !RR_S9_NOFB) {
#if RR_S9 & 16   // the same activation, recomputed from the kept f16 pre-activations (rare path)
            [unroll] for (uint k1 = 0; k1 < 2; ++k1) {
                const uint4 h1 = rr_fence4(hh[k1]);
                ah[k1] = uint4(rr_act(h1.x, cln2, cl2, c12, c22, c32), rr_act(h1.y, cln2, cl2, c12, c22, c32),
                               rr_act(h1.z, cln2, cl2, c12, c22, c32), rr_act(h1.w, cln2, cl2, c12, c22, c32));
            }
#endif
            du[0] = rr_widen(rr_fence4(ah[0]));
            du[1] = rr_widen(rr_fence4(ah[1]));
            RR_Q8(2u, du, au);
        }
        RR_AB(AU, au[0], au[1])
        [unroll] for (uint c1 = 0; c1 < 2 - RR_HCOMPACT; ++c1) {
            v8f a;
            if (j == 0u) {
                a = wg_load8f(p_bias, 640u + (16u * c1 + 8u * hf) * 4u);
#if RR_HCOMPACT
                a.v[3] = hf != 0u ? asfloat(wg_ld32(p_bias, 640u + 16u * 4u)) : a.v[3];   // slot 11 = b2[16]
#endif
            }
#if RR_HVARL & 5
            else a = ddf[c1];
#elif RR_T1P
            else if (c1 == 1u) { a = d1f; a.v[0] = f16tof32(dd[1].x); a.v[1] = f16tof32(dd[1].x >> 16); }
            else a = rr_widen(dd[c1]);
#else
            else a = rr_widen(dd[c1]);
#endif
#if RR_STRICT_STEP
            a = RR_HSTEP(RR_WAB(RR_HCOMPACT ? 28u + j : 20u + 4u * c1 + j), AU, w2[2u * c1], au[0], w2[2u * c1 + 1u], au[1], a);
#else
            a = wg_wmma_fp8(w2[2u * c1], au[0], a);
            a = wg_wmma_fp8(w2[2u * c1 + 1u], au[1], a);
#endif
#if RR_HVARL & 5
            ddf[c1] = RR_HR16(a);
#elif RR_T1P
            if (c1 == 1u) {
                d1f = a;
                v8f t1 = sw_zero8();
                t1.v[0] = a.v[0]; t1.v[1] = a.v[1];
                dd[1] = wg_to_f16_ags(t1);
            } else dd[c1] = wg_to_f16_ags(a);
#else
            dd[c1] = wg_to_f16_ags(a);
#endif
        }
        if (j < 3u) { [unroll] for (uint i4 = 0; i4 < 4; ++i4) { w1[i4] = n1[i4]; w2[i4] = n2[i4]; } }
    }
    // ---------------------------------------------------------------- step 9: y2h = R(Dd m + y1h)
#if RR_HVARL & 4
    {
        const float m = f16tof32(k_m & 0xffffu);
#if RR_HCOMPACT
        // slot 11 = y1 channel 16 (as below); the lane exchange outside the select (HLSL 2021's ?: short-circuits:
        // inside it the source lanes would be inactive)
        const float y16 = wg_xor16(y1f[1].v[0]);
        y1f[0].v[3] = hf != 0u ? y16 : y1f[0].v[3];
        y2h[1] = uint4(0u, 0u, 0u, 0u);
#endif
        [unroll] for (uint c3 = 0; c3 < 2 - RR_HCOMPACT; ++c3) {
            v8f t;
            [unroll] for (uint i6 = 0; i6 < 8; ++i6) t.v[i6] = mad(ddf[c3].v[i6], m, y1f[c3].v[i6]);
            y2h[c3] = wg_to_f16_ags(t);   // RNE, one rounding
        }
    }
#else
#if RR_HVARL & 1   // the f32 chain's result, rounded once for the f16 step 9
    [unroll] for (uint c4 = 0; c4 < 2 - RR_HCOMPACT; ++c4) dd[c4] = wg_to_f16_ags(ddf[c4]);
#endif
#if RR_HCOMPACT
    {   // slot 11 (lane hf 1, element 3) takes y1h channel 16 (lane hf 0 of the token, tile 1, element 0)
        const uint y16 = wg_xor16_u(y1h[1].x) & 0xffffu;
        y1h[0].y = hf != 0u ? (y1h[0].y & 0xffffu) | (y16 << 16) : y1h[0].y;
        dd[1] = uint4(0u, 0u, 0u, 0u);
        y2h[1] = uint4(0u, 0u, 0u, 0u);
    }
#endif
    [unroll] for (uint c2 = 0; c2 < 2 - RR_HCOMPACT; ++c2)
        y2h[c2] = uint4(U2(mad(H2(dd[c2].x), m2, H2(y1h[c2].x))), U2(mad(H2(dd[c2].y), m2, H2(y1h[c2].y))),
                        U2(mad(H2(dd[c2].z), m2, H2(y1h[c2].z))), U2(mad(H2(dd[c2].w), m2, H2(y1h[c2].w))));
#endif
#endif

    // ==================================================================================== the head (spec 55)
    // lane (r, hf) holds y2h channels 8hf .. 8hf+7 (y2h[0]) and 16+8hf .. 16+8hf+7 (y2h[1]) of token r. Each lane
    // writes its half of every texel. Roles (names below: g0 .. g3 = gate logits; G0..G2 = GATES logits; px, py, rho =
    // PREC logits; la, lb = CANDW logits d = la - lb; ha, hb = HID logits of values 0, 1 and 2, 3), five 32-bit words:
    //   hf0: WA = (g0, g1), YS = (g2, g3 = z), W1 = (py, rho), W2.lo = la, WT.lo = hb
    //   hf1: WA = (G0, G1), YS = (G2, px),     W1.lo = lb,     W2.lo = ha
#if RR_P55_ERRATUM
    // natural numbering: hf0 holds y0 .. y7, y16 .. y23; hf1 y8 .. y15, y24 .. y31. Two lane swaps:
    // hf0 WA = (y0, y1), YS = (y2, y3), W1 = (y8, y9) and W2 = (y10, -) from hf1, WT = y16;
    // hf1 WA = (y4, y5) and YS = (y6, y7) from hf0, W1 = y12, W2 = y14.
    // RR_HTRIM: y2h is RR_HVARL's conversion result; its readers are integer work (table addresses, masks, lane
    // swaps) and, behind a select, the f16 max / subtract of the 18-channel reading: no merge with the conversion.
    const uint4 Y0 = RR_YF4(y2h[0]), Y1 = RR_YF4(y2h[1]);
    const bool h0 = hf == 0u;
    const uint X1 = wg_xor16_u(h0 ? Y0.z : Y0.x), X2 = wg_xor16_u(h0 ? Y0.w : Y0.y);
#if RR_HCOMPACT
    const uint WA = h0 ? Y0.x : X1, YS = h0 ? Y0.y : X2, W1 = h0 ? X1 : Y0.z, W2 = h0 ? X2 : Y0.w, WT = X2 >> 16;
#else
    const uint WA = h0 ? Y0.x : X1, YS = h0 ? Y0.y : X2, W1 = h0 ? X1 : Y0.z, W2 = h0 ? X2 : Y0.w, WT = Y1.x;
#endif
#else
    // page 55 numbering: hf0 WA = (y0, y1), YS = (y4, y5), W1 = (y16, y17), W2 = y20, WT = y2;
    // hf1 WA = (y8, y9), YS = (y12, y13), W1 = y24, W2 = y28 (all local)
    const uint4 Y0 = rr_fence4(y2h[0]), Y1 = rr_fence4(y2h[1]);
    const bool h0 = hf == 0u;
    const uint WA = Y0.x, YS = Y0.z, W1 = Y1.x, W2 = Y1.z, WT = Y0.y;
#endif
    const uint ptab = rr_u(p_tab);
    // table lookups: hf0 sig(g0, g1, g2, g3), sig(py), tanh(hb), rho(rho); hf1 sig(G0, G1, G2, px), tanh(ha)
    const uint S0 = wg_ld16(ptab, (WA & 0xffffu) * 2u), S1 = wg_ld16(ptab, (WA >> 16) * 2u);
    const uint S2 = wg_ld16(ptab, (YS & 0xffffu) * 2u), S3 = wg_ld16(ptab, (YS >> 16) * 2u);
    const uint S4 = wg_ld16(ptab, h0 ? (W1 & 0xffffu) * 2u : TAB_TANH + (W2 & 0xffffu) * 2u);
    const uint T2 = wg_ld16(ptab, TAB_TANH + (WT & 0xffffu) * 2u);
    const float RHO = asfloat(wg_ld32(p_rho, (W1 >> 16) * 4u));
    // the two lanes of a token swap: hf0 -> (sig g2, sig g3), la; hf1 -> (sig px, lb)
    const uint Q0 = wg_xor16_u(h0 ? (S2 | (S3 << 16)) : (S3 | ((W1 & 0xffffu) << 16)));
    const uint Q1 = wg_xor16_u(W2 & 0xffffu);
#if RR_HEAD18
    // 18-channel reading: hf0 holds (y10, y11) in W2 and (y16, y17) in WT, hf1 (y12, y13) in W1 and (y14, y15) in W2.
    // HID: hf0 values 2, 3 = tanh(y16), tanh(y17); hf1 values 0, 1 = tanh(y14), tanh(y15) (T3 = the high halves).
    // CANDW (55-ERRATUM-2): W4.r..a from (y10, y11, y12, y13), every step binary16 (each lane gets the other lane's
    // pair and computes all four): m = max; d = R(z - m); t = R(d R(log2 e)); e = R(2^t) (table, correctly rounded);
    // S = R(R(e0 + e2) + R(e1 + e3)); q = R(1/S) (table, correctly rounded); W = R(e q). hf0 stores W0, W1, hf1 W2, W3.
#if RR_S9 & 64   // the 18-channel CANDW / HID chain only when the frame uses that reading (hmode: a launch constant)
    uint T3 = 0u, cw18a = 0u, cw18b = 0u;
    [branch] if (hmode) {
        T3 = wg_ld16(ptab, TAB_TANH + ((h0 ? WT : W2) >> 16) * 2u);
        const uint P18 = wg_xor16_u(h0 ? W2 : W1);
        const half2 Z01 = H2(h0 ? W2 : P18), Z23 = H2(h0 ? P18 : W1);
        const half2 Zm = max(Z01, Z23);
        const half Zmax = max(Zm.x, Zm.y);
        const half2 Zmm = half2(Zmax, Zmax);
        const half2 L2E = H2(0x3dc53dc5u);                            // R(log2 e)
        const uint T01 = U2(HF(HF(Z01 - Zmm) * L2E)), T23 = U2(HF(HF(Z23 - Zmm) * L2E));
        const uint E01 = wg_ld16(ptab, TAB_EXP2 + (T01 & 0xffffu) * 2u) | (wg_ld16(ptab, TAB_EXP2 + (T01 >> 16) * 2u) << 16);
        const uint E23 = wg_ld16(ptab, TAB_EXP2 + (T23 & 0xffffu) * 2u) | (wg_ld16(ptab, TAB_EXP2 + (T23 >> 16) * 2u) << 16);
        const uint SP = U2(HF(H2(E01) + H2(E23)));                    // (R(e0 + e2), R(e1 + e3))
        const uint SS = U2(HF(H2(SP) + H2((SP >> 16) | (SP << 16))));  // R(R(e0 + e2) + R(e1 + e3)) in both halves
        const half2 Q18 = H2(wg_ld16(ptab, TAB_RCP16 + (SS & 0xffffu) * 2u) * 0x10001u);
        const uint WL = U2(HF(H2(h0 ? E01 : E23) * Q18));
        cw18a = WL & 0xffffu; cw18b = WL >> 16;
    }
#else
    const uint T3 = wg_ld16(ptab, TAB_TANH + ((h0 ? WT : W2) >> 16) * 2u);
    const uint P18 = wg_xor16_u(h0 ? W2 : W1);
    const half2 Z01 = H2(h0 ? W2 : P18), Z23 = H2(h0 ? P18 : W1);
    const half2 Zm = max(Z01, Z23);
    const half Zmax = max(Zm.x, Zm.y);
    const half2 Zmm = half2(Zmax, Zmax);
    const half2 L2E = H2(0x3dc53dc5u);                            // R(log2 e)
    // RR_HTRIM: subtract -> multiply (no fma form), and the product only feeds the table address (integer work)
    const uint T01 = U2(HFT2(HFT2(Z01 - Zmm) * L2E)), T23 = U2(HFT2(HFT2(Z23 - Zmm) * L2E));
    const uint E01 = wg_ld16(ptab, TAB_EXP2 + (T01 & 0xffffu) * 2u) | (wg_ld16(ptab, TAB_EXP2 + (T01 >> 16) * 2u) << 16);
    const uint E23 = wg_ld16(ptab, TAB_EXP2 + (T23 & 0xffffu) * 2u) | (wg_ld16(ptab, TAB_EXP2 + (T23 >> 16) * 2u) << 16);
    // RR_HTRIM: an add feeding an add (no fma form), and SS only feeds the table address
    const uint SP = U2(HFT2(H2(E01) + H2(E23)));                    // (R(e0 + e2), R(e1 + e3))
    const uint SS = U2(HFT2(H2(SP) + H2((SP >> 16) | (SP << 16))));  // R(R(e0 + e2) + R(e1 + e3)) in both halves
    const half2 Q18 = H2(wg_ld16(ptab, TAB_RCP16 + (SS & 0xffffu) * 2u) * 0x10001u);
    const uint WL = U2(HF(H2(h0 ? E01 : E23) * Q18));
    const uint cw18a = WL & 0xffffu, cw18b = WL >> 16;
#endif
#endif

    // ---------------------------------------------------------------- 55.2 Cnet (hf0: channels 0, 1; hf1: 2, 3)
    const half2 g2 = rr_c2(k_g);
    const uint zs = h0 ? S3 : (Q0 >> 16);                         // sig(g3) = z
    const half2 z2 = H2(zs * 0x10001u);
    const half2 omz = HFT(H2(0x3c003c00u) - z2);                  // R(1 - z) (RR_HTRIM: a multiplicand of the fma below)
    const half2 gate = HF(H2(h0 ? (S0 | (S1 << 16)) : Q0) * g2);   // R(sig(Dh_j) g): hf0 j = 0, 1; hf1 j = 2 (low)
    const int Mx = (int)(Wt >> 1), My = (int)(Ht >> 1);
    const int xm = (X - 1) >> 1, ym = (Y - 1) >> 1;               // floor((X - 1/2) / 2)
    const uint rxm = (uint)rr_refl(xm, Mx), rxp = (uint)rr_refl(xm + 1, Mx);
    const uint rym = (uint)rr_refl(ym, My), ryp = (uint)rr_refl(ym + 1, My);
    const uint phc = rr_u(p_hc), cb = 4u * hf;
    const uint cA = wg_ld32(phc, (rym * (uint)Mx + rxp) * 8u + cb), cB = wg_ld32(phc, (rym * (uint)Mx + rxm) * 8u + cb);
    const uint cC = wg_ld32(phc, (ryp * (uint)Mx + rxm) * 8u + cb), cD = wg_ld32(phc, (ryp * (uint)Mx + rxp) * 8u + cb);
    const uint hold = wg_ld32(p_hold, tok * 8u + cb);
    const half2 wxp = H2((X & 1) != 0 ? 0x34003400u : 0x3a003a00u), wxm = H2((X & 1) != 0 ? 0x3a003a00u : 0x34003400u);
    const half2 wyp = H2((Y & 1) != 0 ? 0x34003400u : 0x3a003a00u), wym = H2((Y & 1) != 0 ? 0x3a003a00u : 0x34003400u);
    half2 bt = HF(HF(H2(cA) * wxp) * wym);
    // RR_HTRIM: an fma result feeds the next fma's addend or a multiply (no merge); hh only feeds integer work (the
    // select below) before it is the fma's addend
    bt = HFT(mad(HF(H2(cB) * wxm), wym, bt));
    bt = HFT(mad(HF(H2(cC) * wxm), wyp, bt));
    const half2 uu = HFT(mad(HF(H2(cD) * wxp), wyp, bt));
    uint hh = U2(HFT(HF(uu * gate) * z2));
    hh = h0 ? hh : ((hh & 0xffffu) | 0x80000000u);                 // hnew_3 = R(hold_3 omz): addend -0
    const uint hn = U2(mad(H2(hold), omz, H2(hh)));

    // ---------------------------------------------------------------- 55.3 PREC, CANDW values (f32, then R)
    const float s1 = asfloat(k_s1);
#if (RR_S9 & 64) || RR_HCVB   // CANDW from page 55's paired reading (sigx) only when the frame uses that reading
    float cv = 0.0f;
    [branch] if (!hmode) {
        const uint la = h0 ? (W2 & 0xffffu) : Q1, lb = h0 ? (Q0 >> 16) : (W1 & 0xffffu);
        const float d = f16tof32(la) - f16tof32(lb);                   // f32(la - lb), one rounding
        cv = rr_lhalf(h0 ? d : -d);                        // hf0: w = sigx(d)/2, hf1: w' = sigx(-d)/2
    }
#else
    const uint la = h0 ? (W2 & 0xffffu) : Q1, lb = h0 ? (Q0 >> 16) : (W1 & 0xffffu);
    const float d = f16tof32(la) - f16tof32(lb);                   // f32(la - lb), one rounding
    const float cv = rr_lhalf(h0 ? d : -d);                        // hf0: w = sigx(d)/2, hf1: w' = sigx(-d)/2
#endif
    const float po = FF(f16tof32(h0 ? S4 : S3) * s1);              // hf0: Py, hf1: Px
    const float sq = po * po;                                      // hf0: PREC 1, hf1: PREC 0
    const float px = FF(f16tof32(Q0 & 0xffffu) * s1);              // hf0: Px
    const float p2 = FF(px * po) * RHO;                            // hf0: PREC 2 = (Px Py) rho
    v8f cf = sw_zero8();
    cf.v[0] = cv; cf.v[1] = sq; cf.v[2] = p2;
    const uint4 c16 = wg_to_f16_ags(cf);
    const uint cv16 = c16.x & 0xffffu, sq16 = c16.x >> 16, p216 = c16.y & 0xffffu;

    // ---------------------------------------------------------------- stores
    const bool inT = X >= 0 && Y >= 0 && X < (int)Wt && Y < (int)Ht;
    const bool inI = X >= 0 && Y >= 0 && X < (int)Wr && Y < (int)Hr;
    const uint tt = (uint)Y * Wt + (uint)X, ti = (uint)Y * Wr + (uint)X;
    const uint hv = h0 ? T2 : S4;                                  // hf0: R(tanh hb) (values 2, 3), hf1: R(tanh ha)
#if RR_HEAD18
    const uint hv2 = hmode ? T3 : hv;                              // second value of the lane's HID pair
    const uint cwa = hmode ? cw18a : cv16, cwb = hmode ? cw18b : cv16;   // the lane's two CANDW values
#else
    const uint hv2 = hv, cwa = cv16, cwb = cv16;
#endif
#if RR_S9 & 32   // the 8-bit codes once, shared by the stores and the render record (same function of the same values)
    const uint u8a = rr_u8(cwa), u8b = rr_u8(cwb), u80 = rr_u8(S0), u81 = rr_u8(S1), u82 = rr_u8(S2);
#else
#define u8a rr_u8(cwa)
#define u8b rr_u8(cwb)
#define u80 rr_u8(S0)
#define u81 rr_u8(S1)
#define u82 rr_u8(S2)
#endif
#ifndef RR_HSTM
#define RR_HSTM 0   // clean/rr-f-a14-tail D42 (same bytes; compile-only: same store count, +3 VALU, not timed): the hf1 lane's record parts reach the hf0 lane (xor 16), which
#endif              // writes each record whole (Cnet b64, CANDW b32, HID b32, PREC b64, GATES b32, A b128, W b64)
#if RR_HSTM
    {
        const uint cw16 = (u8a | (u8b << 8)) & 0xffffu, hd16 = (rr_s8(hv) | (rr_s8(hv2) << 8)) & 0xffffu;
        const uint gt = u80 | (u81 << 8) | (u82 << 16);
        const uint hnx = wg_xor16_u(hn), cwx = wg_xor16_u(cw16), hdx = wg_xor16_u(hd16), sqx = wg_xor16_u(sq16), gtx = wg_xor16_u(gt);
        if (h0) {
            if (inT) wg_st64(rr_u(p_cnet), tt * 8u, uint2(hn, hnx));
            if (inI) {
                wg_st32(rr_u(p_candw), ti * 4u, cw16 | (cwx << 16));
                wg_st32(rr_u(p_hid), ti * 4u, hdx | (hd16 << 16));
                wg_st64(rr_u(p_prec), ti * 8u, uint2((sqx & 0xffffu) | (sq16 << 16), p216));
                wg_st32(rr_u(p_gates), ti * 4u, gtx);
            }
        }
    }
#else
    if (inT) wg_st32(rr_u(p_cnet), tt * 8u + cb, hn);
    if (inI) {
        wg_st16(rr_u(p_candw), ti * 4u + 2u * hf, u8a | (u8b << 8));
        wg_st16(rr_u(p_hid), ti * 4u + (h0 ? 2u : 0u), rr_s8(hv) | (rr_s8(hv2) << 8));
        if (h0) {
            wg_st16(rr_u(p_prec), ti * 8u + 2u, sq16);
            wg_st32(rr_u(p_prec), ti * 8u + 4u, p216);
        } else {
            wg_st16(rr_u(p_prec), ti * 8u, sq16);
            wg_st32(rr_u(p_gates), ti * 4u, u80 | (u81 << 8) | (u82 << 16));
        }
    }
#endif
#if RR_OUT_PREP
    // the render record of token ti (rr_output_prep's a / b vectors): hf0 holds Cnet 0, 1, PREC 1 (Py), PREC 2, CANDW
    // bytes 0, 1; hf1 Cnet 2, PREC 0 (Px), GATES, CANDW bytes 2, 3 (the stores above)
    [branch] if (p_oscr != 0u) {
        bool orare = false;
        v8f e = sw_zero8();
        e.v[0] = rr_ocf(f16tof32(hn & 0xffffu));
        e.v[1] = rr_ocf(f16tof32(hn >> 16));
        e.v[2] = rr_omq(f16tof32(sq16), asfloat(k_o2n), orare);
        e.v[3] = h0 ? rr_omq(f16tof32(p216), asfloat(k_o2x2n), orare) : rr_od255(u8b);   // hf1: W half 3
        e.v[4] = rr_od255(u8a);
#if RR_S9 & 32
        e.v[5] = rr_od255(h0 ? u8b : u80);                                                       // hf0: W half 1
#else
        e.v[5] = h0 ? rr_od255(u8b) : rr_od255(u80);                                             // hf0: W half 1
#endif
        e.v[6] = rr_od255(u81); e.v[7] = rr_od255(u82);
        const uint4 o16 = wg_to_f16_ags(e);
        if (inI) {
            const uint opw = (o_wh + 1u) / 2u, olb = 256u + opw * ((o_hh + 1u) / 2u) * 4u;
            const uint ra = p_oscr + ((olb + 255u) & ~255u), rb = ra + Wr * Hr * 16u, rw = rb + ((Wr * Hr * 4u + 15u) & ~15u);
            const uint w2 = (o16.z & 0xffffu) | ((h0 ? (o16.z >> 16) : (o16.y >> 16)) << 16);   // W halves (a, b)
#if !RR_HSTM
            if (h0) {
                wg_st32(rr_u(ra), ti * 16u, o16.x);                 // A halves 0, 1: Cf(Cnet 0), Cf(Cnet 1)
                wg_st16(rr_u(ra), ti * 16u + 14u, o16.y & 0xffffu);  // A half 7: Mq(Py, K2n)
                wg_st32(rr_u(rb), ti * 4u, o16.y >> 16);            // B: Mq(PREC 2, K2x2n)
                wg_st32(rr_u(rw), ti * 8u, w2);                     // W halves 0, 1
            } else {
                wg_st32(rr_u(ra), ti * 16u + 4u, (o16.x & 0xffffu) | (o16.z & 0xffff0000u));   // A halves 2, 3
                wg_st32(rr_u(ra), ti * 16u + 8u, o16.w);            // A halves 4, 5: the gates
                wg_st16(rr_u(ra), ti * 16u + 12u, o16.y & 0xffffu);  // A half 6: Mq(Px, K2n)
                wg_st32(rr_u(rw), ti * 8u + 4u, w2);                // W halves 2, 3
            }
#else
            {
                const uint a1 = (o16.x & 0xffffu) | (o16.z & 0xffff0000u), a2 = o16.w, a3 = o16.y & 0xffffu;   // hf1: A 2-3, 4-5, 6
                const uint a1x = wg_xor16_u(a1), a2x = wg_xor16_u(a2), a3x = wg_xor16_u(a3), w2x = wg_xor16_u(w2);
                if (h0) {
                    wg_st128(rr_u(ra), ti * 16u, uint4(o16.x, a1x, a2x, a3x | (o16.y << 16)));   // A halves 0-7
                    wg_st32(rr_u(rb), ti * 4u, o16.y >> 16);                                    // B
                    wg_st64(rr_u(rw), ti * 8u, uint2(w2, w2x));                                 // W halves 0-3
                }
            }
#endif
        }
        if (WaveActiveAnyTrue(orare) && WaveIsFirstLane()) { uint o; wg_mem_gc.InterlockedOr(rr_u(p_oscr) + 4u, 1u, o); }
    }
#endif
#if RR_STRICT_STEP == 4
    if (lane == 0u) wg_st32(rr_u(p_flag), win * 16u + 4u * wv, WaveActiveAnyTrue((rr_yflag & 0xffu) != 0u) ? 1u : 0u);
#endif
#if RR_DEBUG
    if (inT) {
        wg_st128(rr_u(p_dy), tt * 64u + 16u * hf, Y0);
        wg_st128(rr_u(p_dy), tt * 64u + 32u + 16u * hf, Y1);
    }
    if (inI) {
        const uint img = Wr * Hr * 8u, pd = rr_u(p_dimg);
        wg_st32(pd, ti * 8u + (h0 ? 4u : 0u), hv | (hv2 << 16));             // HID values
        if (!h0) wg_st64(pd + img, ti * 8u, uint2(S0 | (S1 << 16), S2));      // GATES values
        wg_st32(pd + 2u * img, ti * 8u + 4u * hf, cwa | (cwb << 16));         // CANDW values
    }
#endif
}

#if RR_STRICT_STEP == 4
// The weights' digit planes for RR_STRICT_STEP 4 (once per frame, before rr_d0r_head): wave w of workgroup g takes
// weight fragment pair i = 4 g + w (fragments 2i, 2i + 1; 28 pairs, 7 workgroups) and stores ri_act's planes per lane
// and its wave-uniform statistics (beta, d3 | NaN mark << 1, vmax, l1).
WG_KERNEL(rr_d0r_wplanes)
{
    WG_SETUP();
    const uint tid = wg_lane_id(), lane = tid & 31u;
    const uint i = wg_block_id() * 4u + (uint)wg_uniform((int)(tid >> 5));
    if (i >= 32u) return;
    // pairs 28 + j: W2's merged channel tile of K-step j (RR_HCOMPACT; fragments of pair 20 + j with row 11 from 24 + j)
    const bool mg = i >= 28u, sw = mg && (lane & 15u) == 11u;
    const uint fb = (mg ? 2u * (i - 8u) : 2u * i) + (sw ? 8u : 0u), fl = sw ? (lane & 16u) : lane;
    const uint2 f0 = rr_frag(fb, fl), f1 = rr_frag(fb + 1u, fl);
    const RrAB r = ri_act(f0, f1);
    const uint pb = rr_u(p_wpl), o = i * 1536u + lane * 48u;
    wg_st128(pb, o, uint4(r.d0.p0, r.d0.p1));
    wg_st128(pb, o + 16u, uint4(r.d0.p2, r.d1.p0));
    wg_st128(pb, o + 32u, uint4(r.d1.p1, r.d1.p2));
    if (lane == 0u)
        wg_st128(pb, RR_WPL_STATS + i * 16u,
                 uint4((uint)r.beta, (r.d3 ? 1u : 0u) | ((rr_yflag & 1u) << 1), asuint(r.vmax), asuint(r.l1)));
}
#endif
