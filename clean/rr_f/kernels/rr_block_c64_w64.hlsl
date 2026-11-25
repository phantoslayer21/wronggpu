// SPDX-License-Identifier: MIT
// clean/rr-f-wave4b R4b.2: the spec 51 window block of the level-1 stages E1, D1-F and D1-R (C = 64, H = 2, MLP width
// 256) in wave64, design D of rr_block_c32.hlsl (the wave64 form of the C = 32 stages): the relaxed body's stagee
// kernels only (RR_B32, RR_VARL 1, step E, weights from memory, WMMA-interleaved chains), WG_BLOCK 128, WG_WAVE 64,
// the window text of rr_block_fast.hlsl operation by operation. Same arguments and buffer layouts; same output bits.
//
// One workgroup = one 8 x 8 window and two wave64 waves; wave w owns the window tokens of the wave32 waves wv = 2 w
// (tile 0) and 2 w + 1 (tile 1) of rr_block_fast.hlsl (16 tokens each, token = 16 wv + n, n = lane & 15). Wave64 lane
// L = l + 32 hh holds, for each tile, half hh of wave32 lane l's fragment (l = 16 hf + n): fp8 A / B bytes 4 hh ..
// 4 hh + 3, accumulator elements 4 hh .. 4 hh + 3 (the v_wmma wave64 layout, see rr_block_c32.hlsl). A v8f holds
// elements 0-3 of tile 0 and 0-3 of tile 1; a uint2 fp8 fragment: x = tile 0, y = tile 1; a uint4 of packed f16:
// x, y = tile 0, z, w = tile 1. Every value is computed by the operation the wave32 kernel computes it with (wave64
// v_wmma = wave32 v_wmma bit for bit, the conversions are per value), so the outputs keep their bits; memory and LDS
// byte layouts are unchanged. What moves is where a value sits:
//   * per-token quantities (coordinates, token loads / stores, the score bias, the hidden-state tail) once per tile;
//   * shared operands (weights, biases, the k / v fragments of the exchange, the coarse fragments of D1-F, the merge's
//     coarse tokens) loaded once, 4 bytes per lane, and used by both tiles' v_wmma (wg_wmma_fp8);
//   * the fusion up-projection (D1-F) and the merge (E1) give the two tiles different output tiles (ot0, ot1 = ot0 + 1)
//     of the same input: per-tile weights and biases, one shared B fragment;
//   * the softmax sum Z, as rr_block_c32.hlsl: the wave32 lane's tree sum zs = A + B of its 8 elements becomes the
//     bracket of the lane's 4 elements plus the other half's (lane ^ 32), A + B or B + A: IEEE addition is
//     commutative, so the sums are equal unless the result is a NaN (payload / sign may differ), and rz = rcp(..) only
//     feeds w * rz -> rr_q8_small (v_cvt_pk_fp8_f32), which gives 0xff for every NaN whatever its sign and payload;
//   * the Q8 votes (WaveActiveAnyTrue) cover both tiles: both forms of a Q8 group give rr_q8's bits for every input.
//
// Defines: RR_KIND 0 (E1: adapter RR_ADAPT 2, merge RR_CN, merged hidden RR_HMERGE; the encoder hidden rule), 2 (D1-F:
// fusion half, RR_KC = 96) or 3 (D1-R: recurrent hidden state), RR_Z 0, RR_STEP 0, RR_VARL 1, RR_B32 1.
#include "wg_platform.hlsli"
#if WG_WAVE != 64 || WG_BLOCK != 128
#error "rr_block_c64_w64.hlsl: WG_BLOCK 128, WG_WAVE 64 only"
#endif
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

WG_ARGS_BEGIN
    uint pX, pCin, pHold, pHc;
    uint pWqv, pTb, pWo, pBo, pMlp, pB1, pB2, pWm, pBm, pWa, pBa, pWup, pBup, pWh, pBh;
    uint pSig, pZ, pY2, pMerged, pHnew, pHmerged;
    int Nx, Ny, sx, sy, nwx;
    uint c_ra_m, c_f_g;
    uint wc0, wc1, wc2, wc3, wc4;
WG_ARGS_END

#define RR_L01BIT (RR_ADAPT ? 4 : RR_KIND == 2 ? 8 : 16)   // this stage's bit in the L01_* masks (as rr_block_fast.hlsl)
// clean/rr-f-wave5 (fast mode only: the stagef.* kernels): L01_B16R (the bit of the stage in the mask) takes the relaxed body's packed-f16
// elementwise forms (act2 / attn_w2 / pk_mad / Q8 from f16 on v_cvt_pkrtz-packed WMMA results, rr_fast.hlsli bv_*) instead of the f32
// forms of RR_B32, as rr_block_c32.hlsl does (the matrix chains stay f32 between their steps, packed once at their ends; the
// hidden-state pre-activation is rounded to nearest even). RR_Q8_FAST 1 (rr_fast.hlsli): branchless Q8. Not exact; quality gate.
#ifndef L01_B16R
#define L01_B16R 0
#endif
#if defined(RR_B32) && RR_B32 && (RR_L01BIT & L01_B16R) != 0
#undef RR_B32
#define RR_B32 0
#define RR_B16R 1
#endif
#if defined(RR_B16R)
#define RR_STEP2HH(a0_, b0_, a1_, b1_, c_) wg_to_f16_ags(step2_raw(a0_, b0_, a1_, b1_, c_))   // history: RNE
#else
#define RR_STEP2HH step2h
#endif
#include "rr_common.hlsli"
#include "rr_step.hlsli"
#undef rr_d8
v8f rr_d8(uint2 q)   // e4m3 bytes -> f32 (exact), one conversion per tile
{
    const wg_frag r0 = wg_ags_convert(rr_w64f(q.x, 0u, 0u, 0u), WG_AGS_T_FP8_B, WG_AGS_T_F32_ACC);
    const wg_frag r1 = wg_ags_convert(rr_w64f(q.y, 0u, 0u, 0u), WG_AGS_T_FP8_B, WG_AGS_T_F32_ACC);
    v8f d;
    [unroll] for (uint i = 0; i < 4; ++i) { d.v[i] = asfloat(r0.v[i]); d.v[4 + i] = asfloat(r1.v[i]); }
    return d;
}

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
#if RR_C != 64 || RR_H != 2
#error "rr_block_c64_w64.hlsl: C = 64, H = 2 only"
#endif
#define RC RR_C
#define RH RR_H
#define KS (RC / 16)          // 4 channel fragments of a token
#define NS (RC / 32)          // 2
#define HID (4 * RC)          // 256: MLP width
#define NJ (HID / 32)         // 8 MLP chunks
#define HIDDEN (RR_KIND == 0 || RR_KIND == 3)
#define UNIT (64 * RC)        // bytes of a weight unit: QV_h (Wq_h | Wv_h) or an MLP chunk (W1 | W2); Wo_h is half

// ---- LDS: the same 8192 bytes as rr_block_fast.hlsl's rr_xq[1024] (two head buffers: q [wave][ft][lane], v^T after),
// addressed per dword: wave32 slot i (8 bytes) = dwords 2 i, 2 i + 1; a wave64 lane holds dword 2 i + hh of each slot of
// its tiles. RR_OVL (as rr_block_fast.hlsl): the merge buffer MY (E1: 64 tokens x 64 channels = 512 slots) lives in
// head 0's exchange buffer (its last reader is head 0's PV; head 1 uses buffer 1, and every wave passes head 1's
// barrier only after those reads) and the fusion up-projections UP (D1-F: 16 x HID / 8 = 512 slots) in head 1's (written and
// read before the head loop; head 1 first writes its buffer after head 0's barrier, which every wave reaches only
// after its last UP read).
groupshared uint rr_xq32[2048];
#define XB(h) (((h) & 1u) * 512u)
#define MY32(i) rr_xq32[(i)]
#define RR_UP32(i) rr_xq32[1024u + (i)]
#if RR_WSRC != 1
#error "rr_block_c64_w64.hlsl: RR_WSRC 1 only"
#endif

static uint rr_lane;   // lane in the wave
static uint rr_hf;     // (lane >> 4) & 1
static uint rr_lp;     // wave32 lane whose fragment halves this lane holds (lane & 31)
static uint rr_hh;     // lane >> 5: the half (fp8 bytes 4 hh .., accumulator elements 4 hh ..)
// per-lane offsets, fenced (rr_fence_u: the same value with no known bits) so that the compiler adds them to the base
// once and keeps the constant parts in the instructions' offset fields
static uint rr_lo;     // lp * 8 + 4 hh: byte of a fragment-ordered 8-byte slot
static uint rr_xo;     // 2 lp + hh: dword of an LDS slot row
static uint rr_t4;     // 8 hf + 4 hh: byte within a 16-channel group of a token / an output tile row
static uint rr_uo;     // D1-F: dword offset in the up-projection buffer
static uint rr_f2;     // 2 hf + hh
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

uint2 qv_frag(uint h, uint f) { return wfrag(pWqv + h * UNIT, f); }
uint2 wo_frag(uint h, uint f) { return wfrag(pWo + h * (UNIT / 2), f); }
uint2 mlp_frag(uint j, uint f) { return wfrag(pMlp + j * UNIT, f); }

#include "rr_fast.hlsli"
#ifndef L01_WREV
#define L01_WREV 20      // a14 default (relaxed body): E1 and D1-R windows in reverse order
#endif
#if ((RR_ADAPT ? 4 : RR_KIND == 2 ? 8 : 16) & L01_WREV) != 0
#define RR_WBID() (wg_grid_size() - 1u - wg_block_id())   // the same windows, the last one first (each window's work unchanged)
#else
#define RR_WBID() wg_block_id()
#endif
#define RR_FPP
#define RR_FPA
#if ((!RR_B32 && !defined(RR_B16R)) || !RR_VARL || (!RR_WIL && !defined(RR_B16R)) || RR_Z != 0 || RR_STEP != 0 || !RR_STICKY)
#error "rr_block_c64_w64.hlsl: the relaxed stagee configuration only"
#endif
#if RR_KIND == 0 && (RR_ADAPT != 2 || RR_CN != 96 || RR_HMERGE != 1)
#error "rr_block_c64_w64.hlsl: E1 with the direct adapter, merge 96 and the merged hidden only"
#endif
#if RR_KIND == 2 && RR_KC != 96
#error "rr_block_c64_w64.hlsl: D1-F with RR_KC 96 only"
#endif
#if RR_KIND == 1
#error "rr_block_c64_w64.hlsl: kinds 0, 2, 3"
#endif

// One MLP chunk j (step 8) as rr_block_fast.hlsl's mlp_chunk (RR_WIL, RR_MDPF forms): Du_j (NS steps, start b1), Au_j =
// Q8(act(Du_j)), then one Dd step per channel tile. first: the Dd start values are the f32 bias tiles (the chunk-0
// form); otherwise the widened packed dd. L01_MDPF: the Dd weight fragments are loaded right after the chunk's hidden
// WMMAs, before the activation and the Q8 branch.
void mlp_chunk(uint j, bool first, uint2 y1q[KS], inout RR_CHAIN dd[KS], RrWx wc)
{
    v8f u0 = bias_tile(pB1, 2u * j), u1 = bias_tile(pB1, 2u * j + 1u);
    rrbv h0, h1;
    [unroll] for (uint s = 0; s < NS; ++s) {
        const uint2 a00 = mlp_frag(j, 2u * s), a01 = mlp_frag(j, 2u * s + 1u);
        const uint2 a10 = mlp_frag(j, KS + 2u * s), a11 = mlp_frag(j, KS + 2u * s + 1u);
        wil2(u0, u1, a00, y1q[2 * s], a01, y1q[2 * s + 1], a10, y1q[2 * s], a11, y1q[2 * s + 1]);
        if (s + 1 == NS) { h0 = RR_BCV(u0); h1 = RR_BCV(u1); }
    }
    uint2 wd[2 * KS];
    [unroll] for (uint q = 0; q < 2 * KS; ++q) wd[q] = mlp_frag(j, 2u * KS + q);
    const rrbv a0h = bv_act(h0, wc);
    const rrbv a1h = bv_act(h1, wc);
    uint bad = 0u;
    uint2 au0, au1;
    Q8_TRY au0 = bv_q8f(a0h, bad); au1 = bv_q8f(a1h, bad);
    Q8_ELSE au0 = bv_q8s(a0h); au1 = bv_q8s(a1h); Q8_END
    [unroll] for (uint c2 = 0; c2 < KS; c2 += 2u) {
        v8f e0, e1;
        if (first) { e0 = bias_tile(pB2, c2); e1 = bias_tile(pB2, c2 + 1u); } else { e0 = dd[c2]; e1 = dd[c2 + 1u]; }
        wil2(e0, e1, wd[2u * c2], au0, wd[2u * c2 + 1u], au1, wd[2u * c2 + 2u], au0, wd[2u * c2 + 3u], au1);
        dd[c2] = e0; dd[c2 + 1u] = e1;
    }
}

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
    rr_f2 = rr_fence_u(2u * rr_hf + rr_hh);
    rr_sat = false;
    rr_nan = false;
    DBG_INIT
    const uint wv0 = 2u * WaveReadLaneFirst(wg_lane_id() >> 6);   // wave32 wave of tile 0 (tile 1: wv0 + 1)
    const uint n = rr_lane & 15u;
#if RR_KIND == 2
    rr_uo = rr_fence_u(64u * n + 2u * rr_hf + rr_hh);   // dword of UP slot ((n HID + 16 ot + 8 hf) >> 3) - 4 ot
#endif
    const uint wid = RR_WBID();
    const int wi = (int)(wid % (uint)nwx), wj = (int)(wid / (uint)nwx);
    const int lx = (int)(n & 7u);
    const int X = 8 * wi - sx + lx;
    const bool r1 = min(Nx, Ny) >= 10;                     // one reflection suffices (rr_fast.hlsli rr_refl)
    const int Xr = rr_refl(X, Nx, r1);
    int Yt[2], Yrt[2];
    uint tok[2];
    bool real[2];
    [unroll] for (uint t0 = 0; t0 < 2u; ++t0) {
        const int ly = (int)(2u * (wv0 + t0) + (n >> 3));
        Yt[t0] = 8 * wj - sy + ly;
        Yrt[t0] = rr_refl(Yt[t0], Ny, r1);
        real[t0] = X >= 0 && X < Nx && Yt[t0] >= 0 && Yt[t0] < Ny;
        tok[t0] = (uint)(Yrt[t0] * Nx + Xr);
    }
    const RrWc wc = rr_wc(wc0, wc1, wc2, wc3, wc4);
    const half2 ra2 = rr_splat_lo(c_ra_m), m2 = rr_splat_hi(c_ra_m);
    const half2 f2 = rr_splat_lo(c_f_g), g2 = rr_splat_hi(c_f_g);
#if RR_B32
    const RrWx wx = rr_wx(wc);
    const float ra2x = (float)ra2.x, m2x = (float)m2.x, f2x = (float)f2.x;
#endif
    const int Mx = Nx >> 1, My = Ny >> 1;
    const int cx0 = (8 * wi - sx) >> 1, cy0 = (8 * wj - sy) >> 1;

    // ---- block input x and residual base ----------------------------------------------------------------------------
    uint2 xf[KS];
#if RR_KIND == 2
    rrbv xr[KS];                                             // xf of 51.5 (f32), the first residual base
#endif
#if RR_ADAPT
    {   // E1: x = Q8(MM(in; Wa; ba)), in = the level-1 token's 128 channels as stored (51.5, 51.6); pX = IN0's y2 itself
        // ([2 Ny][2 Nx][32]): child c = 2 dy + dx of the level-1 token is the level-0 token (2 Yr + dy, 2 Xr + dx); its 32
        // bytes are channels 32 c .. 32 c + 31 of the 128
        uint2 inf[8];
        [unroll] for (uint k = 0; k < 8; ++k) {
            const uint tc0 = (2u * (uint)Yrt[0] + (k >> 2)) * (2u * (uint)Nx) + 2u * (uint)Xr + ((k >> 1) & 1u);
            const uint tc1 = (2u * (uint)Yrt[1] + (k >> 2)) * (2u * (uint)Nx) + 2u * (uint)Xr + ((k >> 1) & 1u);
            inf[k] = rr_tld(pX, tc0 * 32u + 16u * (k & 1u), tc1 * 32u + 16u * (k & 1u));
        }
        [unroll] for (uint ct = 0; ct < KS; ++ct) {
            uint2 a[8];
            [unroll] for (uint k2 = 0; k2 < 8; ++k2) a[k2] = wfrag(pWa, ct * 8u + k2);
            v8f acc = bias_tile(pBa, ct);
            [unroll] for (uint s = 0; s < 4; ++s) acc = step2l(s == 3u, a[2 * s], inf[2 * s], a[2 * s + 1], inf[2 * s + 1], acc);
            uint bad = 0u;
            Q8_TRY xf[ct] = q8f(acc, bad);
            Q8_ELSE xf[ct] = q8s(acc); Q8_END
        }
    }
#elif RR_KIND == 2
    {   // fusion half (51.5): the window's up-projection (16 output tiles of 16 x 16 coarse tokens; wave32 wave wv computes
        // tiles wv + 4 t2) to LDS; then xf = R(d8(skip) f + d8(upq)) per fine token, x = Q8(xf), xr = xf.
        const int crx = cx0 + (int)(n & 3u), cry = cy0 + (int)(n >> 2);
        const uint ctok = (uint)(rr_refl(cry, My, r1) * Mx + rr_refl(crx, Mx, r1));
        uint2 cf[RR_KC / 16];
        [unroll] for (uint k = 0; k < RR_KC / 16; ++k) {
            const uint c = wg_ld32(pCin + rr_t4, ctok * RR_KC + 16u * k);
            cf[k] = uint2(c, c);
        }
        // L01_FPF: the next tile's weight fragments and bias are loaded before the current tile's WMMAs and Q8 group
        uint2 an[RR_KC / 16];
        v8f accn = bias_tile2(pBup, wv0, wv0 + 1u);
        [unroll] for (uint k0 = 0; k0 < RR_KC / 16; ++k0)
            an[k0] = uint2(wfrag1(pWup, wv0 * (RR_KC / 16) + k0), wfrag1(pWup, (wv0 + 1u) * (RR_KC / 16) + k0));
        [unroll] for (uint t2 = 0; t2 < HID / 64; ++t2) {
            const uint ot0 = wv0 + 4u * t2, ot1 = ot0 + 1u;
            uint2 a[RR_KC / 16];
            [unroll] for (uint k1 = 0; k1 < RR_KC / 16; ++k1) a[k1] = an[k1];
            v8f acc = accn;
            if (t2 + 1u < HID / 64) {
                accn = bias_tile2(pBup, ot0 + 4u, ot1 + 4u);
                [unroll] for (uint k3 = 0; k3 < RR_KC / 16; ++k3)
                    an[k3] = uint2(wfrag1(pWup, (ot0 + 4u) * (RR_KC / 16) + k3), wfrag1(pWup, (ot1 + 4u) * (RR_KC / 16) + k3));
            }
            [unroll] for (uint s = 0; s < RR_KC / 32; ++s)
                acc = step2l(s + 1 == RR_KC / 32, a[2 * s], cf[2 * s], a[2 * s + 1], cf[2 * s + 1], acc);
            uint bad = 0u;
            uint2 uq;
            Q8_TRY uq = q8f(acc, bad);
            Q8_ELSE uq = q8s(acc); Q8_END
            RR_UP32(rr_uo + 4u * ot0) = uq.x;
            RR_UP32(rr_uo + 4u * ot1) = uq.y;
        }
        GroupMemoryBarrierWithGroupSync();
        const uint par0 = wv0 * 4u + (uint)(lx >> 1), par1 = par0 + 4u;
        const uint ch0 = 2u * ((uint)Yt[0] & 1u) + ((uint)X & 1u), ch1 = 2u * ((uint)Yt[1] & 1u) + ((uint)X & 1u);
        // dword of UP slot ((par HID + ch RC + 16 ct + 8 hf) >> 3) held by this lane: 64 par + 16 ch + 4 ct + 2 hf + hh
        const uint ub0 = rr_f2 + 64u * par0 + 16u * ch0, ub1 = rr_f2 + 64u * par1 + 16u * ch1;
        uint bad = 0u;
        [unroll] for (uint ct = 0; ct < KS; ++ct) {
            const uint2 sk = rr_tld(pX, tok[0] * RC + 16u * ct, tok[1] * RC + 16u * ct);
            const uint2 up = uint2(RR_UP32(ub0 + 4u * ct), RR_UP32(ub1 + 4u * ct));
            xr[ct] = bv_mad(bv_d8(sk), RR_BVK(f2), bv_d8(up));
        }
        Q8_TRY [unroll] for (uint c0 = 0; c0 < KS; ++c0) xf[c0] = bv_q8f(xr[c0], bad);
        Q8_ELSE [unroll] for (uint c = 0; c < KS; ++c) xf[c] = bv_q8s(xr[c]); Q8_END
    }
#else
    [unroll] for (uint ks = 0; ks < KS; ++ks) xf[ks] = rr_tld(pX, tok[0] * RC + 16u * ks, tok[1] * RC + 16u * ks);
#endif

    // ---- heads: steps 1-6 ------------------------------------------------------------------------------------------
    rrbv dp[KS];
#if !RR_B32
    v8f dpv[KS];   // RR_B16R: the Dp chain stays f32 between the heads, packed once after the loop
#endif
    [unroll] for (uint h = 0; h < RH; ++h) {
        // [A] q_h (features x tokens) and v_h^T (tokens x features)
        v8f aq0 = zero8(), aq1 = zero8(), av0 = zero8(), av1 = zero8();
        [unroll] for (uint s = 0; s < NS; ++s) {
            const uint2 q00 = qv_frag(h, 2u * s), q01 = qv_frag(h, 2u * s + 1u);
            const uint2 q10 = qv_frag(h, KS + 2u * s), q11 = qv_frag(h, KS + 2u * s + 1u);
            const uint2 v00 = qv_frag(h, 2u * KS + 2u * s), v01 = qv_frag(h, 2u * KS + 2u * s + 1u);
            const uint2 v10 = qv_frag(h, 3u * KS + 2u * s), v11 = qv_frag(h, 3u * KS + 2u * s + 1u);
            wil4(aq0, aq1, av0, av1, q00, xf[2 * s], q01, xf[2 * s + 1], q10, xf[2 * s], q11, xf[2 * s + 1],
                 xf[2 * s], v00, xf[2 * s + 1], v01, xf[2 * s], v10, xf[2 * s + 1], v11);
        }
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
            wt[0] = bv_attn(RR_BCV(s0), BVW); wt[1] = bv_attn(RR_BCV(s1), BVW); wt[2] = bv_attn(RR_BCV(s2), BVW); wt[3] = bv_attn(RR_BCV(s3), BVW);
        }
#if RR_B32
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
#else
        // step 4 (RR_B16R, the packed form of the exact body): Z = the f32 dot2add sum of the packed weights (per tile the lane's
        // 4 elements of every key tile, then the other half's (lane ^ 32) and the other row group's (lane ^ 16)), R(Z), r =
        // R(1 / Z) in f16, P = Q8(R(w r)) packed
        float za = 0.0f, zc = 0.0f;
        [unroll] for (uint J3 = 0; J3 < 4; ++J3) {
            za = dot2add(rr_h2(wt[J3].x), half2(1.0, 1.0), za);
            za = dot2add(rr_h2(wt[J3].y), half2(1.0, 1.0), za);
            zc = dot2add(rr_h2(wt[J3].z), half2(1.0, 1.0), zc);
            zc = dot2add(rr_h2(wt[J3].w), half2(1.0, 1.0), zc);
        }
        const float zs0 = za + wg_xor(za, 32u), zs1 = zc + wg_xor(zc, 32u);
        const uint zb0 = f2h_hw(zs0 + wg_xor16(zs0)), zb1 = f2h_hw(zs1 + wg_xor16(zs1));
        const half2 r2a = rr_h2(f2h_hw(rcp(f16tof32(zb0))) * 0x10001u), r2b = rr_h2(f2h_hw(rcp(f16tof32(zb1))) * 0x10001u);
        uint2 pf[4];
        [unroll] for (uint J4 = 0; J4 < 4; ++J4) {
            uint4 p;
            p.x = rr_b2(rr_h2(wt[J4].x) * r2a);
            p.y = rr_b2(rr_h2(wt[J4].y) * r2a);
            p.z = rr_b2(rr_h2(wt[J4].z) * r2b);
            p.w = rr_b2(rr_h2(wt[J4].w) * r2b);
            pf[J4] = q8h_small(p);
        }
#endif
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
#if RR_B32
        [unroll] for (uint ct = 0; ct < KS; ++ct) {
            v8f c;
            if (h == 0u) c = bias_tile(pBo, ct); else c = bv_un(dp[ct]);
            dp[ct] = c;
        }
        [unroll] for (uint c2 = 0; c2 < KS; c2 += 2u)
            wil2(dp[c2], dp[c2 + 1u], wo_frag(h, 2u * c2), of[0], wo_frag(h, 2u * c2 + 1u), of[1],
                 wo_frag(h, 2u * c2 + 2u), of[0], wo_frag(h, 2u * c2 + 3u), of[1]);
#else
        if (h == 0u) { [unroll] for (uint ct = 0; ct < KS; ++ct) dpv[ct] = bias_tile(pBo, ct); }
        [unroll] for (uint c2 = 0; c2 < KS; c2 += 2u)
            wil2(dpv[c2], dpv[c2 + 1u], wo_frag(h, 2u * c2), of[0], wo_frag(h, 2u * c2 + 1u), of[1],
                 wo_frag(h, 2u * c2 + 2u), of[0], wo_frag(h, 2u * c2 + 3u), of[1]);
#endif
    }
#if !RR_B32
    [unroll] for (uint c6 = 0; c6 < KS; ++c6) dp[c6] = RR_BCV(dpv[c6]);
#endif

    // ---- step 7: first residual -------------------------------------------------------------------------------------
    rrbv y1h[KS];
    uint2 y1q[KS];
    {
        uint bad = 0u;
        [unroll] for (uint c1 = 0; c1 < KS; ++c1) {
#if RR_KIND == 2
            y1h[c1] = bv_mad(dp[c1], RR_BVK(ra2), xr[c1]);
#else
            y1h[c1] = bv_mad(dp[c1], RR_BVK(ra2), bv_d8(xf[c1]));
#endif
        }
        Q8_TRY [unroll] for (uint c0 = 0; c0 < KS; ++c0) y1q[c0] = bv_q8f(y1h[c0], bad);
        Q8_ELSE [unroll] for (uint c = 0; c < KS; ++c) y1q[c] = bv_q8s(y1h[c]); Q8_END
    }

    // ---- step 8: MLP (chunk 0 peeled: its Dd steps start from the f32 bias; the chunk loop fully unrolled) ------------
#if RR_B32
    rrbv dd[KS];
    mlp_chunk(0u, true, y1q, dd, BVW);
    [unroll] for (uint j = 1; j < NJ; ++j) mlp_chunk(j, false, y1q, dd, BVW);
#else
    v8f ddc[KS];   // RR_B16R: the Dd chain stays f32 over the chunks, packed once after them
    mlp_chunk(0u, true, y1q, ddc, BVW);
    [unroll] for (uint j = 1; j < NJ; ++j) mlp_chunk(j, false, y1q, ddc, BVW);
    rrbv dd[KS];
    [unroll] for (uint c6 = 0; c6 < KS; ++c6) dd[c6] = RR_BCV(ddc[c6]);
#endif

    // ---- step 9: second residual, block output y2 ---------------------------------------------------------------------
    uint2 y2[KS];
    {
        rrbv yh[KS];
        uint bad = 0u;
        [unroll] for (uint c3 = 0; c3 < KS; ++c3) yh[c3] = bv_mad(dd[c3], RR_BVK(m2), y1h[c3]);
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

    // ---- merge (51.4, E1) ----------------------------------------------------------------------------------------------
#if RR_CN > 0
    {
        // y2 of token t = 16 (wv0 + tile) + n into the merge buffer: dword 16 t + 4 c + 2 hf + hh of MY (slot 8 t + 2 c + hf)
        const uint mt = 16u * wv0 + n;
        [unroll] for (uint c5 = 0; c5 < KS; ++c5) {
            MY32(16u * mt + rr_f2 + 4u * c5) = y2[c5].x;
            MY32(16u * mt + 256u + rr_f2 + 4u * c5) = y2[c5].y;
        }
        GroupMemoryBarrierWithGroupSync();
        const uint rx = n & 3u, ry = n >> 2;
        const int Xc = cx0 + (int)rx, Yc = cy0 + (int)ry;
        const bool cin = Xc >= 0 && Xc < Mx && Yc >= 0 && Yc < My;
        // wave32 wave wv computes output tiles ot = wv, wv + 4 (< RR_CN / 16): tile 0 = ot0, tile 1 = ot0 + 1; the second
        // pair exists for wave64 wave 0 only (ot0 = 4); wave-uniform
        [unroll] for (uint t2 = 0; t2 < 2u; ++t2) {
            const uint ot0 = wv0 + 4u * t2, ot1 = ot0 + 1u;
            [branch] if (ot0 < RR_CN / 16) {
                v8f acc = bias_tile2(pBm, ot0, ot1);
                // C = 64, HID = 256: the coarse token's four children: zi = 32 s + 16 e + 8 hf, cn = s >> 1, MY slot of (s, e)
                // = 128 ry + 16 rx + hf + 64 (s >> 2) + 8 ((s >> 1) & 1) + 4 (s & 1) + 2 e; this lane's dword = 2 slot + hh
                const uint mb = 256u * ry + 32u * rx + rr_f2;
                [unroll] for (uint s = 0; s < HID / 32; ++s) {
                    uint2 b[2], a[2];
                    [unroll] for (uint e = 0; e < 2; ++e) {
                        const uint w = MY32(mb + 128u * (s >> 2) + 16u * ((s >> 1) & 1u) + 8u * (s & 1u) + 4u * e);
                        b[e] = uint2(w, w);
                        a[e] = uint2(wfrag1(pWm, ot0 * (HID / 16) + 2u * s + e), wfrag1(pWm, ot1 * (HID / 16) + 2u * s + e));
                    }
                    acc = step2l(s + 1 == HID / 32, a[0], b[0], a[1], b[1], acc);
                }
                uint bad = 0u;
                uint2 mq;
                Q8_TRY mq = q8f(acc, bad);
                Q8_ELSE mq = q8s(acc); Q8_END
                if (cin && rr_zero() == 0u) {
                    wg_st32(pMerged + rr_t4, (uint)(Yc * Mx + Xc) * RR_CN + 16u * ot0, mq.x);
                    wg_st32(pMerged + rr_t4, (uint)(Yc * Mx + Xc) * RR_CN + 16u * ot1, mq.y);
                }
            }
        }
    }
#endif

    // ---- hidden state (51.4) -----------------------------------------------------------------------------------------
#if HIDDEN
    {
        // Rows 0-3 of the Wh tile (the 4 hidden channels) are in the hf = 0, hh = 0 lanes: dwords x, y for tile 0, z, w for
        // tile 1. Tile 1's are moved to the hh = 1 lanes (lane ^ 32), so lanes 0-15 run tile 0's token n and lanes 32-47
        // tile 1's, with the operations of the wave32 lanes 0-15 (the other lanes compute unused values, as the wave32
        // hf = 1 lanes do).
        v8f dh = rr_dup4(wg_ld4f(pBh + 4u * rr_t4, 0u));
        uint4 dhb;
        [unroll] for (uint s = 0; s < NS; ++s) {
            if (s + 1 < NS) dh = step2(wfrag(pWh, 2u * s), y2[2 * s], wfrag(pWh, 2u * s + 1u), y2[2 * s + 1], dh);
            else dhb = RR_STEP2HH(wfrag(pWh, 2u * s), y2[2 * s], wfrag(pWh, 2u * s + 1u), y2[2 * s + 1], dh);
        }
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
#if RR_KIND == 0
        const uint G01 = rr_fence_u(rr_b2(rr_h2(s01) * g2)), G23 = rr_fence_u(rr_b2(rr_h2(s23) * g2));
        hn.x = rr_fence_u(rr_b2(rr_h2(hold.x) * rr_h2(G01)));
        hn.y = rr_fence_u(rr_b2(rr_h2(hold.y) * rr_h2(G23))) & 0xffffu;
#else
        const half2 z2 = rr_splat_hi(s23);
        const uint omzb = rr_fence_u(rr_b2(half2(1.0, 1.0) - z2));
        const half2 omz = rr_h2(omzb);
        const int xm = (X - 1) >> 1, ym = (Yh - 1) >> 1;
        const int rxm = rr_refl(xm, Mx, r1), rxp = rr_refl(xm + 1, Mx, r1);
        const int rym = rr_refl(ym, My, r1), ryp = rr_refl(ym + 1, My, r1);
        const half2 wxp = (X & 1) ? half2(0.25, 0.25) : half2(0.75, 0.75);
        const half2 wyp = (Yh & 1) ? half2(0.25, 0.25) : half2(0.75, 0.75);
        const half2 wxm = (X & 1) ? half2(0.75, 0.75) : half2(0.25, 0.25);
        const half2 wym = (Yh & 1) ? half2(0.75, 0.75) : half2(0.25, 0.25);
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
        const uint hh23 = rr_fence_u(rr_b2(rr_h2(ug23) * z2)) & 0xffffu;
        hn.x = rr_fence_u(rr_b2(mad(rr_h2(hold.x), omz, rr_h2(hh01))));
        hn.y = rr_fence_u(rr_b2(mad(rr_h2(hold.y), omz, rr_h2(hh23))));
#endif
        const uint otok = (uint)(Yh * Nx + X);
        if (rl && rr_hf == 0u) wg_st64(pHnew, otok * 8u, hn);
#if RR_HMERGE
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
#endif
    }
#endif
    DBG_FLUSH
}
