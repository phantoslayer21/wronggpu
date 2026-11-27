// SPDX-License-Identifier: GPL-3.0-only
// clean/rr-f-wave4c W4C.1: the spec 51 window block of the wide stages E2, E3, D2-F, D2-R, D3-F, D3-R, D4-F and D4-R (RR_G
// groups of four token tiles, RR_B32 1) in wave64, design D of rr_block_c32.hlsl: the same operations as rr_block_wide.hlsl
// (and so, value for value, the same output bits), arranged as rr_block_c64_w64.hlsl arranges the level-1 stages.
//
// One workgroup = one 8 x 8 window and 2 RR_G wave64 waves (WG_BLOCK 128 RR_G): wave w = 2 g + tw owns, for group g, the
// token tiles wv = 2 tw (tile 0) and 2 tw + 1 (tile 1) of the wave32 kernel's waves (4 g + wv). Group g computes the
// same heads / MLP chunks as the wave32 group g, with the same LDS buffers (byte layout and size unchanged: QV, OB, Y, AU, UP).
// Wave64 lane L = l + 32 hh holds, for each tile, half hh of wave32 lane l's fragment (l = 16 hf + n): fp8 A / B bytes
// 4 hh .. 4 hh + 3, accumulator elements 4 hh .. 4 hh + 3 (the v_wmma wave64 layout). A v8f holds elements 0-3 of tile 0
// and 0-3 of tile 1; a uint2 fp8 fragment: x = tile 0, y = tile 1. Every value is computed by the operation the wave32
// kernel computes it with (wave64 v_wmma = wave32 v_wmma bit for bit, conversions per value). What moves:
//   * per-token quantities (coordinates, loads, stores, score biases, the hidden state) once per tile;
//   * shared operands (weights, biases, the k / v fragments of the exchange, the coarse fragments of the fusion, the merge's
//     coarse tokens) loaded once, 4 bytes per lane, used by both tiles (halving weight VMEM and VGPRs);
//   * the fusion up-projection and the merge compute two output tiles (ot0, ot1 = ot0 + 1) per wave from one shared B;
//   * the softmax sum Z as the wave32 tree (A + B or B + A over the two halves: IEEE addition commutes; only a NaN's
//     payload could differ, and rz only feeds w * rz -> v_cvt_pk_fp8_f32, which gives 0xff for every NaN);
//   * the hidden state: wave (g, tw) with g == tw covers all four tiles with the lane swap of rr_block_c64_w64.hlsl.
// The Q8 votes (WaveActiveAnyTrue) cover both tiles: both forms of a Q8 group give rr_q8's bits for every input.
//
// Defines as rr_block_wide.hlsl (RR_C, RR_H, RR_KIND 0 / 2 / 3, RR_CN, RR_HMERGE, RR_KC, RR_G, RR_PRE, RR_MLK, RR_B32G4),
// RR_B32 1, RR_VARL 1, RR_Z 0, RR_STEP 0, WG_BLOCK = 128 RR_G, WG_WAVE 64. No E1 adapter, no dataflow form.
#include "wg_platform.hlsli"
#if WG_WAVE != 64 || WG_BLOCK != 128 * RR_G
#error "rr_block_wide_w64.hlsl: WG_WAVE 64, WG_BLOCK = 128 RR_G only"
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

#define RR_L01BIT (RR_KIND == 2 ? 8 : 16)   // this stage's bit in the L01_* masks (as rr_block_fast.hlsl)
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
#ifndef RR_KC
#define RR_KC 0
#endif
#ifndef RR_Z
#define RR_Z 0
#endif
#ifndef RR_G
#define RR_G 2
#endif
#ifndef RR_MLK_CMAX
#define RR_MLK_CMAX 96
#endif
// operand loads issued ahead of the matrix steps (bits): 1 heads, 2 MLP, 4 merge, 8 hidden (defaults as rr_block_wide.hlsl)
#ifndef RR_PRE
#if RR_G == 2 && RR_C == 128
#define RR_PRE 14
#else
#define RR_PRE 15
#endif
#endif
#ifdef RR_ADAPT
#if RR_ADAPT
#error "rr_block_wide_w64.hlsl: no E1 adapter"
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
#ifndef RR_MLK                     // MLP chunks per group per round (2: half the rounds, one barrier per round)
#if (RR_PRE & 2) && RR_G == 2 && ((4 * RR_C / 32) % 4) == 0 && RR_C <= RR_MLK_CMAX
#define RR_MLK 2
#else
#define RR_MLK 1
#endif
#endif
#define MR (NJ / (G * RR_MLK))     // MLP rounds
#if (RH % G) != 0 || (NJ % G) != 0 || WG_BLOCK != 128 * G
#error "rr_block_wide_w64.hlsl: RR_H and RR_C / 8 must be multiples of RR_G, WG_BLOCK = 128 RR_G"
#endif
#if (RC % 16) != 0 || ((HID / 16) % 2) != 0 || (KS % 1) != 0
#error "rr_block_wide_w64.hlsl: RR_C must be a multiple of 16 and 4 RR_C / 16 even"
#endif
#if RR_CN > 0 && ((RR_CN / 16) % 2) != 0
#error "rr_block_wide_w64.hlsl: RR_CN / 16 must be even (the merge's two output tiles per wave)"
#endif
uint mrg_k(uint z0) { const uint cn = z0 / RC; return (8u * (cn >> 1) + (cn & 1u)) * (RC / 8u) + (z0 % RC) / 8u; }

// ---- LDS: the wave32 kernel's rr_l[LDS_N] (uint64 entries), addressed per dword: wave32 slot i = dwords 2 i, 2 i + 1; a
// wave64 lane holds dword 2 i + hh of each slot of its tiles. Layouts (slots) as rr_block_wide.hlsl:
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
groupshared uint rr_l32[2 * LDS_N];

static uint rr_lane;   // lane in the wave
static uint rr_hf;     // (lane >> 4) & 1
static uint rr_lp;     // wave32 lane whose fragment halves this lane holds (lane & 31)
static uint rr_hh;     // lane >> 5: the half (fp8 bytes 4 hh .., accumulator elements 4 hh ..)
// per-lane offsets, fenced (rr_fence_u: the same value with no known bits) so that the compiler adds them to the base
// once and keeps the constant parts in the instructions' offset fields
static uint rr_lo;     // lp * 8 + 4 hh: byte of a fragment-ordered 8-byte slot
static uint rr_xo;     // 2 lp + hh: dword of an LDS slot row
static uint rr_t4;     // 8 hf + 4 hh: byte within a 16-channel group of a token / an output tile row
static uint rr_uo;     // fusion: dword offset in the up-projection buffer
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
uint2 rr_xqs(uint r) { const uint w = rr_l32[rr_xqr(r)]; return uint2(w, w); }   // a shared operand from LDS
uint2 rr_xq2(uint r0, uint r1) { return uint2(rr_l32[rr_xqr(r0)], rr_l32[rr_xqr(r1)]); }   // per tile
// Y [token][C] bytes: the dword of this lane's part of channel tile ct of token t (wave32 slot (t C + 16 ct + 8 hf) >> 3)
uint rr_yd(uint t, uint ct) { return 2u * Y_OFF + t * (RC / 4u) + 4u * ct + rr_f2; }

uint2 qv_frag(uint h, uint f) { return wfrag(pWqv + h * UNIT, f); }
uint2 wo_frag(uint h, uint f) { return wfrag(pWo + h * (UNIT / 2), f); }
uint2 mlp_frag(uint j, uint f) { return wfrag(pMlp + j * UNIT, f); }

#ifndef RR_B32G4                    // RR_B32G4 1 keeps RR_B32 at G = 4 (D4-F / D4-R, see rr_block_wide.hlsl)
#define RR_B32G4 0
#endif
#if RR_G == 4 && !RR_B32G4
#error "rr_block_wide_w64.hlsl: RR_G 4 needs RR_B32G4 1"
#endif
#include "rr_fast.hlsli"
#define RR_FPP
#define RR_FPA
#if !RR_B32 || !RR_VARL || !RR_WIL || RR_Z != 0 || RR_STEP != 0 || !RR_STICKY || RR_COUNT
#error "rr_block_wide_w64.hlsl: the relaxed stagee configuration only"
#endif
#if RR_KIND == 2 && RR_KC == 0
#error "rr_block_wide_w64.hlsl: the fusion kind needs RR_KC"
#endif
#if RR_KIND == 1
#error "rr_block_wide_w64.hlsl: kinds 0, 2, 3"
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
    rr_f2 = rr_fence_u(2u * rr_hf + rr_hh);
    rr_sat = false;
    rr_nan = false;
    DBG_INIT
    const uint w = WaveReadLaneFirst(wg_lane_id() >> 6);   // 0 .. 2 G - 1
    const uint g = w >> 1;                                  // group
    const uint tw = w & 1u;
    const uint wv0 = 2u * tw;                               // wave32 token tile of tile 0 (tile 1: wv0 + 1)
    const uint n = rr_lane & 15u;
#if RR_KIND == 2
    rr_uo = rr_fence_u(RC * n + 2u * rr_hf + rr_hh);        // dword of UP slot ((n HID + 16 ot + 8 hf) >> 3) - 4 ot
#endif
    const uint wid = wg_block_id();
    const int wi = (int)(wid % (uint)nwx), wj = (int)(wid / (uint)nwx);
    const int lx = (int)(n & 7u);
    const int X = 8 * wi - sx + lx;
    const bool r1 = min(Nx, Ny) >= 10;
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
    const uint tt0 = 16u * wv0 + n, tt1 = 16u * (wv0 + 1u) + n;   // token of the window (LDS rows), per tile
    const RrWc wc = rr_wc(wc0, wc1, wc2, wc3, wc4);
    const half2 ra2 = rr_splat_lo(c_ra_m), m2 = rr_splat_hi(c_ra_m);
    const half2 f2 = rr_splat_lo(c_f_g), g2 = rr_splat_hi(c_f_g);
    const RrWx wx = rr_wx(wc);
    const float ra2x = (float)ra2.x, m2x = (float)m2.x, f2x = (float)f2.x;
    const int Mx = Nx >> 1, My = Ny >> 1;
    const int cx0 = (8 * wi - sx) >> 1, cy0 = (8 * wj - sy) >> 1;
    uint cto[OWN];                                          // owned channel tiles
    [unroll] for (uint i0 = 0; i0 < OWN; ++i0) cto[i0] = min(g + G * i0, KS - 1u);

    // ---- block input x (all channels, every wave) and the residual base of the owned tiles ---------------------------
    uint2 xf[KS];
    rrbv rb[OWN];
#if RR_KIND == 2
    {   // fusion half (51.5): up-projection tiles over all waves: wave w computes the tiles 2 w + 4 G t2 (tile 0) and the
        // next (tile 1); a wave past the last tile repeats it (the same bytes, same values: equal weights and biases)
        const int crx = cx0 + (int)(n & 3u), cry = cy0 + (int)(n >> 2);
        const uint ctok = (uint)(rr_refl(cry, My, r1) * Mx + rr_refl(crx, Mx, r1));
        uint2 cf[RR_KC / 16];
        [unroll] for (uint k = 0; k < RR_KC / 16; ++k) {
            const uint c = wg_ld32(pCin + rr_t4, ctok * RR_KC + 16u * k);
            cf[k] = uint2(c, c);
        }
        RR_LOOP for (uint t2 = 0; t2 < (HID / 16 + 4u * G - 1u) / (4u * G) + RR_ZL; ++t2) {
            const uint ot0 = min(2u * w + 4u * G * t2, HID / 16 - 1u), ot1 = min(ot0 + 1u, HID / 16 - 1u);
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
            rr_l32[rr_uo + 4u * ot0] = uq.x;
            rr_l32[rr_uo + 4u * ot1] = uq.y;
        }
        GroupMemoryBarrierWithGroupSync();
        const uint par0 = wv0 * 4u + (uint)(lx >> 1), par1 = par0 + 4u;
        const uint ch0 = 2u * ((uint)Yt[0] & 1u) + ((uint)X & 1u), ch1 = 2u * ((uint)Yt[1] & 1u) + ((uint)X & 1u);
        // dword of UP slot ((par HID + ch RC + 16 ct + 8 hf) >> 3) held by this lane: RC par + RC / 4 ch + 4 ct + 2 hf + hh
        const uint ub0 = rr_f2 + RC * par0 + (RC / 4u) * ch0, ub1 = rr_f2 + RC * par1 + (RC / 4u) * ch1;
        rrbv xr[KS];
        uint bad = 0u;
        [unroll] for (uint ct = 0; ct < KS; ++ct) {
            const uint2 sk = rr_tld(pX, tok[0] * RC + 16u * ct, tok[1] * RC + 16u * ct);
            const uint2 up = uint2(rr_l32[ub0 + 4u * ct], rr_l32[ub1 + 4u * ct]);
            xr[ct] = bv_mad(bv_d8(sk), f2x, bv_d8(up));
        }
        Q8_TRY [unroll] for (uint c0 = 0; c0 < KS; ++c0) xf[c0] = bv_q8f(xr[c0], bad);
        Q8_ELSE [unroll] for (uint c = 0; c < KS; ++c) xf[c] = bv_q8s(xr[c]); Q8_END
        [unroll] for (uint i1 = 0; i1 < OWN; ++i1) {
            const uint ct = cto[i1];
            const uint2 sk = rr_tld(pX, tok[0] * RC + 16u * ct, tok[1] * RC + 16u * ct);
            const uint2 up = uint2(rr_l32[ub0 + 4u * ct], rr_l32[ub1 + 4u * ct]);
            rb[i1] = bv_mad(bv_d8(sk), f2x, bv_d8(up));
        }
        GroupMemoryBarrierWithGroupSync();                  // UP overlays the head buffers
    }
#else
    [unroll] for (uint ks = 0; ks < KS; ++ks) xf[ks] = rr_tld(pX, tok[0] * RC + 16u * ks, tok[1] * RC + 16u * ks);
    [unroll] for (uint i1 = 0; i1 < OWN; ++i1) rb[i1] = bv_d8(rr_tld(pX, tok[0] * RC + 16u * cto[i1], tok[1] * RC + 16u * cto[i1]));
#endif

    // ---- heads: steps 1-6, rounds of G heads -------------------------------------------------------------------------
    rrbv dp[OWN];
    [unroll] for (uint i2 = 0; i2 < OWN; ++i2) dp[i2] = bv_of(bias_tile(pBo, cto[i2]));
    const uint qb = QV_OFF + g * 512u;
    [unroll] for (uint r = 0; r < HR; ++r) {
        const uint h = r * G + g;
        const uint tb0 = pTb + 4u * rr_lo + (h * 16u + wv0 * 4u) * 1024u, tb1 = tb0 + 4u * 1024u;
        v8f aq0 = zero8(), aq1 = zero8(), av0 = zero8(), av1 = zero8();
#if RR_PRE & 1
        // every operand load of the round ahead (the driver keeps loads behind the matrix calls, which are UAV atomics)
        uint2 wq[4 * KS];
        [unroll] for (uint f0 = 0; f0 < 4 * KS; ++f0) wq[f0] = qv_frag(h, f0);
        [unroll] for (uint s = 0; s < NS; ++s)
            wil4(aq0, aq1, av0, av1, wq[2 * s], xf[2 * s], wq[2 * s + 1], xf[2 * s + 1],
                 wq[KS + 2 * s], xf[2 * s], wq[KS + 2 * s + 1], xf[2 * s + 1],
                 xf[2 * s], wq[2 * KS + 2 * s], xf[2 * s + 1], wq[2 * KS + 2 * s + 1],
                 xf[2 * s], wq[3 * KS + 2 * s], xf[2 * s + 1], wq[3 * KS + 2 * s + 1]);
        // the score biases and the round's Wo fragments (used after the exchanges)
        v8f sb[4];
        [unroll] for (uint J5 = 0; J5 < 4; ++J5) sb[J5] = rr_cat4(wg_ld4f(tb0, J5 * 1024u), wg_ld4f(tb1, J5 * 1024u));
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
            wil4(aq0, aq1, av0, av1, q00, xf[2 * s], q01, xf[2 * s + 1], q10, xf[2 * s], q11, xf[2 * s + 1],
                 xf[2 * s], v00, xf[2 * s + 1], v01, xf[2 * s], v10, xf[2 * s + 1], v11);
        }
#endif
        uint bad = 0u;
        uint2 qf0, qf1, vf0, vf1;
        Q8_TRY qf0 = q8f(aq0, bad); qf1 = q8f(aq1, bad); vf0 = q8f(av0, bad); vf1 = q8f(av1, bad);
        Q8_ELSE qf0 = q8s(aq0); qf1 = q8s(aq1); vf0 = q8s(av0); vf1 = q8s(av1); Q8_END
        rr_l32[rr_xqr(qb + (wv0 * 2u) * 32u)] = qf0.x;
        rr_l32[rr_xqr(qb + (wv0 * 2u + 1u) * 32u)] = qf1.x;
        rr_l32[rr_xqr(qb + 256u + (wv0 * 2u) * 32u)] = vf0.x;
        rr_l32[rr_xqr(qb + 256u + (wv0 * 2u + 1u) * 32u)] = vf1.x;
        rr_l32[rr_xqr(qb + (wv0 * 2u + 2u) * 32u)] = qf0.y;
        rr_l32[rr_xqr(qb + (wv0 * 2u + 3u) * 32u)] = qf1.y;
        rr_l32[rr_xqr(qb + 256u + (wv0 * 2u + 2u) * 32u)] = vf0.y;
        rr_l32[rr_xqr(qb + 256u + (wv0 * 2u + 3u) * 32u)] = vf1.y;
        GroupMemoryBarrierWithGroupSync();

        // [B] scores (key tile J x the tile's queries), start value = the score bias (f32 fragment, per tile)
        uint2 kk[8];
        [unroll] for (uint J0 = 0; J0 < 8; ++J0) kk[J0] = rr_xqs(qb + J0 * 32u);
#if !(RR_PRE & 1)
        v8f sb[4];
        [unroll] for (uint J5 = 0; J5 < 4; ++J5) sb[J5] = rr_cat4(wg_ld4f(tb0, J5 * 1024u), wg_ld4f(tb1, J5 * 1024u));
#endif
        rrbv wt[4];
        wil4(sb[0], sb[1], sb[2], sb[3], kk[0], qf0, kk[1], qf1, kk[2], qf0, kk[3], qf1, kk[4], qf0, kk[5], qf1, kk[6], qf0, kk[7], qf1);
        [unroll] for (uint J = 0; J < 4; ++J) wt[J] = bv_attn(sb[J], BVW);
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
            [unroll] for (uint i8 = 0; i8 < 8; ++i8) vt[i8] = rr_xqs(qb + 256u + i8 * 32u);
            ov[0] = zero8(); ov[1] = zero8();
            wil2(ov[0], ov[1], vt[0], pf[0], vt[2], pf[1], vt[1], pf[0], vt[3], pf[1]);
            wil2(ov[0], ov[1], vt[4], pf[2], vt[6], pf[3], vt[5], pf[2], vt[7], pf[3]);
        }
        bad = 0u;
        uint2 of[2];
        Q8_TRY of[0] = q8f(ov[0], bad); of[1] = q8f(ov[1], bad);
        Q8_ELSE of[0] = q8s(ov[0]); of[1] = q8s(ov[1]); Q8_END
        rr_l32[rr_xqr(OB_OFF + (g * 4u + wv0) * 64u)] = of[0].x;
        rr_l32[rr_xqr(OB_OFF + (g * 4u + wv0 + 1u) * 64u)] = of[0].y;
        rr_l32[rr_xqr(OB_OFF + (g * 4u + wv0) * 64u + 32u)] = of[1].x;
        rr_l32[rr_xqr(OB_OFF + (g * 4u + wv0 + 1u) * 64u + 32u)] = of[1].y;
        GroupMemoryBarrierWithGroupSync();

        // Dp steps of the round's heads, in head order, for the owned channel tiles
#if RR_PRE & 1
        uint2 ob[G][2];
        [unroll] for (uint g4 = 0; g4 < G; ++g4) {
            ob[g4][0] = rr_xq2(OB_OFF + (g4 * 4u + wv0) * 64u, OB_OFF + (g4 * 4u + wv0 + 1u) * 64u);
            ob[g4][1] = rr_xq2(OB_OFF + (g4 * 4u + wv0) * 64u + 32u, OB_OFF + (g4 * 4u + wv0 + 1u) * 64u + 32u);
        }
        [unroll] for (uint gg = 0; gg < G; ++gg) {
            [unroll] for (uint i4 = 0; i4 < OWN; ++i4)
                dp[i4] = bv_step(wo[gg][2 * i4], ob[gg][0], wo[gg][2 * i4 + 1], ob[gg][1], bv_un(dp[i4]));
        }
#else
        [unroll] for (uint gg = 0; gg < G; ++gg) {
            const uint hh = r * G + gg;
            const uint2 o0 = rr_xq2(OB_OFF + (gg * 4u + wv0) * 64u, OB_OFF + (gg * 4u + wv0 + 1u) * 64u);
            const uint2 o1 = rr_xq2(OB_OFF + (gg * 4u + wv0) * 64u + 32u, OB_OFF + (gg * 4u + wv0 + 1u) * 64u + 32u);
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
            y1h[i5] = bv_mad(dp[i5], ra2x, rb[i5]);
        }
        Q8_TRY [unroll] for (uint c0 = 0; c0 < OWN; ++c0) yq[c0] = bv_q8f(y1h[c0], bad);
        Q8_ELSE [unroll] for (uint c = 0; c < OWN; ++c) yq[c] = bv_q8s(y1h[c]); Q8_END
        [unroll] for (uint i6 = 0; i6 < OWN; ++i6) {
            rr_l32[rr_yd(tt0, cto[i6])] = yq[i6].x;
            rr_l32[rr_yd(tt1, cto[i6])] = yq[i6].y;
        }
    }
    GroupMemoryBarrierWithGroupSync();
    uint2 y1q[KS];
    [unroll] for (uint c7 = 0; c7 < KS; ++c7) y1q[c7] = uint2(rr_l32[rr_yd(tt0, c7)], rr_l32[rr_yd(tt1, c7)]);

    // ---- step 8: MLP in rounds of G chunks ---------------------------------------------------------------------------
    rrbv dd[OWN];
    [unroll] for (uint i7 = 0; i7 < OWN; ++i7) dd[i7] = bv_of(bias_tile(pB2, cto[i7]));
#if RR_MLK == 2
    // two chunks per group per round, (2 r2 + k) G + g; the round's 2 G Au tiles in AU [r2 & 1][k][g], one barrier, then
    // the Dd chain over the round's chunks in chunk order (the same steps and operands as RR_MLK 1)
    RR_LOOP for (uint r2 = 0; r2 < MR + RR_ZL; ++r2) {
        const uint abase = AU_OFF + ((r2 & 1u) * 2u * G) * 256u;   // + wv * 64 + lane (per tile)
        [unroll] for (uint k = 0; k < 2; ++k) {
            const uint j = (2u * r2 + k) * G + g;
            uint2 w1[2 * KS];
            [unroll] for (uint f1 = 0; f1 < 2 * KS; ++f1) w1[f1] = mlp_frag(j, f1);
            v8f u0 = bias_tile(pB1, 2u * j), u1 = bias_tile(pB1, 2u * j + 1u);
            rrbv h0, h1;
            [unroll] for (uint s = 0; s < NS; ++s) {
                wil2(u0, u1, w1[2 * s], y1q[2 * s], w1[2 * s + 1], y1q[2 * s + 1], w1[KS + 2 * s], y1q[2 * s], w1[KS + 2 * s + 1], y1q[2 * s + 1]);
                if (s + 1 == NS) { h0 = u0; h1 = u1; }
            }
            const rrbv a0h = bv_act(h0, BVW);
            const rrbv a1h = bv_act(h1, BVW);
            uint bad = 0u;
            uint2 au0, au1;
            Q8_TRY au0 = bv_q8f(a0h, bad); au1 = bv_q8f(a1h, bad);
            Q8_ELSE au0 = bv_q8s(a0h); au1 = bv_q8s(a1h); Q8_END
            const uint ak = abase + (k * G + g) * 256u;
            rr_l32[rr_xqr(ak + wv0 * 64u)] = au0.x;
            rr_l32[rr_xqr(ak + (wv0 + 1u) * 64u)] = au0.y;
            rr_l32[rr_xqr(ak + wv0 * 64u + 32u)] = au1.x;
            rr_l32[rr_xqr(ak + (wv0 + 1u) * 64u + 32u)] = au1.y;
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
                const uint ak = abase + (hh * G + g6) * 256u;
                ab2[g6][0] = rr_xq2(ak + wv0 * 64u, ak + (wv0 + 1u) * 64u);
                ab2[g6][1] = rr_xq2(ak + wv0 * 64u + 32u, ak + (wv0 + 1u) * 64u + 32u);
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
            wil2(u0, u1, W1A(s), y1q[2 * s], W1B(s), y1q[2 * s + 1], W1C(s), y1q[2 * s], W1D(s), y1q[2 * s + 1]);
            if (s + 1 == NS) { h0 = u0; h1 = u1; }
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
        const uint ab = AU_OFF + ((r2 & 1u) * G) * 256u;   // + wv * 64 + lane (per tile)
        rr_l32[rr_xqr(ab + g * 256u + wv0 * 64u)] = au0.x;
        rr_l32[rr_xqr(ab + g * 256u + (wv0 + 1u) * 64u)] = au0.y;
        rr_l32[rr_xqr(ab + g * 256u + wv0 * 64u + 32u)] = au1.x;
        rr_l32[rr_xqr(ab + g * 256u + (wv0 + 1u) * 64u + 32u)] = au1.y;
        GroupMemoryBarrierWithGroupSync();
#if RR_PRE & 2
        uint2 ab2[G][2];
        [unroll] for (uint g6 = 0; g6 < G; ++g6) {
            ab2[g6][0] = rr_xq2(ab + g6 * 256u + wv0 * 64u, ab + g6 * 256u + (wv0 + 1u) * 64u);
            ab2[g6][1] = rr_xq2(ab + g6 * 256u + wv0 * 64u + 32u, ab + g6 * 256u + (wv0 + 1u) * 64u + 32u);
        }
        [unroll] for (uint gg2 = 0; gg2 < G; ++gg2) {
            [unroll] for (uint i9 = 0; i9 < OWN; ++i9)
                dd[i9] = bv_step(w2[gg2][2 * i9], ab2[gg2][0], w2[gg2][2 * i9 + 1], ab2[gg2][1], bv_un(dd[i9]));
        }
#else
        [unroll] for (uint gg2 = 0; gg2 < G; ++gg2) {
            const uint jj = r2 * G + gg2;
            const uint2 b0 = rr_xq2(ab + gg2 * 256u + wv0 * 64u, ab + gg2 * 256u + (wv0 + 1u) * 64u);
            const uint2 b1 = rr_xq2(ab + gg2 * 256u + wv0 * 64u + 32u, ab + gg2 * 256u + (wv0 + 1u) * 64u + 32u);
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
            yh[c3] = bv_mad(dd[c3], m2x, y1h[c3]);
        }
        Q8_TRY [unroll] for (uint c0 = 0; c0 < OWN; ++c0) y2o[c0] = bv_q8f(yh[c0], bad);
        Q8_ELSE [unroll] for (uint c = 0; c < OWN; ++c) y2o[c] = bv_q8s(yh[c]); Q8_END
        const uint otok0 = (uint)(Yt[0] * Nx + X), otok1 = (uint)(Yt[1] * Nx + X);
        [unroll] for (uint c4 = 0; c4 < OWN; ++c4) {
            if (real[0]) wg_st32(pY2 + rr_t4, otok0 * RC + 16u * cto[c4], y2o[c4].x);
            if (real[1]) wg_st32(pY2 + rr_t4, otok1 * RC + 16u * cto[c4], y2o[c4].y);
            rr_l32[rr_yd(tt0, cto[c4])] = y2o[c4].x;
            rr_l32[rr_yd(tt1, cto[c4])] = y2o[c4].y;
        }
    }
    GroupMemoryBarrierWithGroupSync();

    // ---- merge (51.4, E2 / E3): output tiles over all waves, two per wave ---------------------------------------------
#if RR_CN > 0
    {
        const uint rx = n & 3u, ry = n >> 2;
        const int Xc = cx0 + (int)rx, Yc = cy0 + (int)ry;
        const bool cin = Xc >= 0 && Xc < Mx && Yc >= 0 && Yc < My;
        // the merge operand row of zi = z0 + 8 hf + 4 hh (z0 a multiple of 16): LDS dword = lane base + 2 mrg_k(z0)
        const uint mlb = 2u * Y_OFF + (16u * ry + 2u * rx) * (RC / 4u) + rr_f2;
        RR_LOOP for (uint ot0 = 2u * w; ot0 < RR_CN / 16 + RR_ZL; ot0 += 4u * G) {
            const uint ot1 = ot0 + 1u;
#if RR_PRE & 4
            // operands in batches of 8 fragments (4 steps), the next batch loaded before the current one's steps
            uint2 am[2][8], bm[2][8];
            [unroll] for (uint f2i = 0; f2i < 8; ++f2i)
                am[0][f2i] = uint2(wfrag1(pWm, ot0 * (HID / 16) + f2i), wfrag1(pWm, ot1 * (HID / 16) + f2i));
            v8f acc = bias_tile2(pBm, ot0, ot1);
            [unroll] for (uint bt = 0; bt < HID / 128; ++bt) {
                const uint cb = bt & 1u;
                [unroll] for (uint f3 = 0; f3 < 8; ++f3) {
                    const uint wm = rr_l32[mlb + 2u * mrg_k(16u * (8u * bt + f3))];
                    bm[cb][f3] = uint2(wm, wm);
                }
                if (bt + 1 < HID / 128) {
                    [unroll] for (uint f4 = 0; f4 < 8; ++f4)
                        am[cb ^ 1u][f4] = uint2(wfrag1(pWm, ot0 * (HID / 16) + 8u * (bt + 1u) + f4),
                                                wfrag1(pWm, ot1 * (HID / 16) + 8u * (bt + 1u) + f4));
                }
                [unroll] for (uint s = 0; s < 4; ++s)
                    acc = step2l(and(bt + 1 == HID / 128, s == 3u), am[cb][2 * s], bm[cb][2 * s], am[cb][2 * s + 1], bm[cb][2 * s + 1], acc);
            }
#else
            v8f acc = bias_tile2(pBm, ot0, ot1);
            RR_LOOP for (uint s = 0; s < HID / 32 + RR_ZL; ++s) {
                uint2 b[2], a[2];
                [unroll] for (uint e = 0; e < 2; ++e) {
                    const uint wm = rr_l32[mlb + 2u * mrg_k(32u * s + 16u * e)];
                    b[e] = uint2(wm, wm);
                    a[e] = uint2(wfrag1(pWm, ot0 * (HID / 16) + 2u * s + e), wfrag1(pWm, ot1 * (HID / 16) + 2u * s + e));
                }
                acc = step2l(s + 1 == HID / 32 + RR_ZL, a[0], b[0], a[1], b[1], acc);
            }
#endif
            uint bad = 0u;
            uint2 mq;
            Q8_TRY mq = q8f(acc, bad);
            Q8_ELSE mq = q8s(acc); Q8_END
#if RR_Q8_FAST
            // Convergent wave barrier on mq before branching on cin: prevents LLVM from hoisting
            // `if (cin)` above the merge WMMAs (RDNA 4 v_wmma under narrow EXEC produces non-finite lanes).
            const bool mq_ok = !WaveActiveAnyTrue(((mq.x | mq.y) & rr_zero()) != 0u);
            if (and(mq_ok, cin)) {
#else
            if (cin && rr_zero() == 0u) {
#endif
                wg_st32(pMerged + rr_t4, (uint)(Yc * Mx + Xc) * RR_CN + 16u * ot0, mq.x);
                wg_st32(pMerged + rr_t4, (uint)(Yc * Mx + Xc) * RR_CN + 16u * ot1, mq.y);
            }
        }
    }
#endif

    // ---- hidden state (51.4): wave (g, tw) with g == tw covers all four token tiles (tile 0 / tile 1 of tw = 0 and 1); every
    // group computed the same values in the wave32 kernel, so the others' copies are redundant (the guard is wave-uniform,
    // holds no barrier). Rows 0-3 of the Wh tile (the 4 hidden channels) are in the hf = 0, hh = 0 lanes: dwords x, y for tile 0,
    // z, w for tile 1; tile 1's are moved to the hh = 1 lanes (lane ^ 32), so lanes 0-15 run tile 0's token n and lanes 32-47
    // tile 1's, with the operations of the wave32 lanes 0-15 (the other lanes compute unused values, as the wave32 hf = 1
    // lanes do).
#if HIDDEN
    if (g == tw) {
        uint2 y2[KS];
        [unroll] for (uint c8 = 0; c8 < KS; ++c8) y2[c8] = uint2(rr_l32[rr_yd(tt0, c8)], rr_l32[rr_yd(tt1, c8)]);
        uint2 wh[KS];
#if RR_PRE & 8
        [unroll] for (uint f4 = 0; f4 < KS; ++f4) wh[f4] = wfrag(pWh, f4);
#endif
        v8f dh = rr_dup4(wg_ld4f(pBh + 4u * rr_t4, 0u));
        uint4 dhb;
        [unroll] for (uint s = 0; s < NS; ++s) {
#if !(RR_PRE & 8)
            wh[2 * s] = wfrag(pWh, 2u * s);
            wh[2 * s + 1] = wfrag(pWh, 2u * s + 1u);
#endif
            if (s + 1 < NS) dh = step2(wh[2 * s], y2[2 * s], wh[2 * s + 1], y2[2 * s + 1], dh);
            else dhb = step2h(wh[2 * s], y2[2 * s], wh[2 * s + 1], y2[2 * s + 1], dh);
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
