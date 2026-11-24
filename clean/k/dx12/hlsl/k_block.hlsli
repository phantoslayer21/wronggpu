// SPDX-License-Identifier: MIT
// Model K block of spec 41 with its fused stage neighbours, for the D3D12 ports: the HLSL counterpart of the k_block
// template of clean/k/kernels/k_block.hip (WrongGPU's own code), FP32 accumulation only (template parameter F16 = 0:
// the network's default exact path; the FP16-accumulating k_stageh_* / k_stage2h_* forms and the FP8 forms are not
// ported). Steps, layouts and orders are those of k_block.hip (read its header comment): one 128-lane workgroup (4 waves)
// per G windows of 8x8 tokens; wave w owns window-local tokens 16 w .. 16 w + 15 as the WMMA columns; lane l holds
// token 16 w + l % 16 and channels 16 kt + 8 (l / 16) + 0..7 of it in every fragment.
//
// Every floating-point operation is the HIP kernel's, in its order and with its roundings (k_common.hlsli says how
// FP16 operations, conversions and the softmax weights reproduce the gfx1201 instructions); every FP32 accumulator
// starts from the same bias and runs over the same tiles in the same order, and is rounded once. Differences in form
// only (bit-identical):
//  * driver-channel rules of model M (clean/dx12/README.md): each group of matrix products loads all its weight /
//    operand fragments, in source order, before its first multiply (the channel calls are UAV atomics: the compiler
//    never moves loads across them); no branches around the products (the mirror, the window clamps and the
//    per-token conditions are selects; only stores are predicated);
//  * LDS (one groupshared array of KB_LDS_HALFS: 8192 halfs = 16 KB, what every two-window stage needs; the one-window
//    e0 / e1 forms declare their exact 6144 / 4096 halfs in k_block.hlsl, more workgroups per WGP) holds the
//    same fragment-major K / VT tiles, the same x staging and the same NB (1 or 2 head buffers) as HIP;
//  * the XG step 3 re-read of x for a member outside the map reads the mirrored member's slot of P.y instead and
//    discards it (HIP loads nothing there): the value is the same zero; the address stays inside the map;
//  * the hidden-history epilogue's exact rsqrt (k_fp.h kf_rsqrt: v_rsq_f32 estimate + binary64 rounding tests) uses
//    HLSL doubles (DoublePrecisionFloatShaderOps incl. fma, as the RR / exposure ports); its binary64 fallback
//    kd_rsqrt32 is reached on gfx1201 only for x = +inf (-> 0) and NaN (-> NaN).
#pragma once

#include "k_blkparams.hlsli"
#include "k_common.hlsli"
#include "k_norm.hlsli"

// KB_LEAN: the opaque zero only where a result feeds an FP16 add or multiply (k_common.hlsli wgk_*n); 0: everywhere
#ifndef KB_LEAN
#define KB_LEAN 2
#endif
#if KB_LEAN >= 2   // also the softmax polynomial's 1 - k u^2 (feeds a multiply) and u (1 - k u^2) (feeds the f32 widening)
#define KB_SUB8S wgk_sub8n
#define KB_MUL8S wgk_mul8n
#else
#define KB_SUB8S wgk_sub8
#define KB_MUL8S wgk_mul8
#endif
#if KB_LEAN
#define KB_MUL8W wgk_mul8n
#define KB_ADD8W wgk_add8n
#define KB_CLAMP8W wgk_clamp8cn
#define KB_MAXC8W wgk_maxc8n
#define KB_ACT8W wgk_act8n
#else
#define KB_MUL8W wgk_mul8
#define KB_ADD8W wgk_add8
#define KB_CLAMP8W wgk_clamp8c
#define KB_MAXC8W wgk_maxc8
#define KB_ACT8W wgk_act8
#endif

#define KMIX_FIXED 0
#define KMIX_SOFTMAX 1
#define KIN_MAP 0
#define KIN_PADDED 1
#define KIN_EMBED 2
#define KIN_EXPAND 3

#define WGK_SOFTMAX_C 0x4d69u      // 21.640625
#define WGK_SOFTMAX_KAPPA 0x11d5u  // 0.0007119178771972656

#define KB_FRAG 256        // halfs per 16 x 16 tile (32 lanes x 8 halfs, fragment order)
#ifndef KB_LDS_HALFS
#define KB_LDS_HALFS 8192  // LDS of the stage kernels of k_block.hlsl (NB / LH1 / LH2 as in HIP: 16 KB each)
#endif
groupshared uint4 kb_lds[KB_LDS_HALFS / 8];
uint4 kb_ldl(uint off_h) { return kb_lds[off_h >> 3]; }
void kb_stl(uint off_h, uint4 v) { kb_lds[off_h >> 3] = v; }

// KB_PREF (default 0): every weight section of the stage is touched once per workgroup at the start (one dword per
// 128-byte line, spread over the 128 lanes), so a workgroup that finds them cold (in the frame, after the previous
// stage streamed its activations) waits for one round of parallel misses instead of one miss per weight tile along its
// serial chain. The loaded words only feed a pin (0 at run time) added to one later load's base: no arithmetic changes.
// Measured in the game-like hipbench (2026-11-02): on all six stages +0.019 ms per frame, on e1 alone -0.001 ms: off.
#ifndef KB_PREF
#define KB_PREF 0
#endif
uint kb_touch(uint base, int halfs, uint tid)
{
    uint v = 0u;
    [unroll] for (int j = 0; j < (2 * halfs + 16383) / 16384; ++j)
        v ^= wg_ld32(base, min(tid * 128u + (uint)j * 16384u, (uint)(2 * halfs) - 4u));
    return v;
}

// ---- hidden-history epilogue (spec 43 section 6.8; HIP wgk_hidden_norm2 / wgk_store_hidden*) --------------------------
float wgk_hidden_norm2(float h[4])
{
    float ss = h[1] * h[1];
    ss = mad(h[0], h[0], ss);
    ss = mad(h[2], h[2], ss);
    ss = mad(h[3], h[3], ss);
    return ss + 1.0f;
}

// k_fp.h kf_rsqrt: 1 / sqrt(x) correctly rounded to FP32 for x >= 1 (estimate v_rsq_f32, then up to three exact
// binary64 midpoint tests moving it by one ulp); kd_rsqrt32 (binary64 sqrt + division) for the rest, which here is only
// x = +inf (-> +0) or NaN (the hidden norm x = sum of squares + 1 is never below 1).
float wgk_kf_rsqrt(float x)
{
    const bool fin = x >= 1.0f && x - x == 0.0f;
    float y = wg_rsq(x);
    bool done = !fin;
    const double xd = (double)x;
    [unroll] for (int it = 0; it < 3; ++it) {
        const uint b = asuint(y);
        const double yd = (double)y;
        const double ulp = (double)asfloat((b & 0x7F800000u) - (23u << 23));
        const double up = yd + 0.5 * ulp;
        const double lo = yd - ((b & 0x7FFFFFu) == 0u ? 0.25 : 0.5) * ulp;
        const bool inc = fma(up * up, xd, -1.0) < 0.0;
        const bool dec = !inc && fma(lo * lo, xd, -1.0) > 0.0;
        const uint nb = inc ? b + 1u : (dec ? b - 1u : b);
        y = done ? y : asfloat(nb);
        done = done || (!inc && !dec);
    }
    // not finite / below 1 (only +inf and NaN reach here): kd_rsqrt32
    const float other = isnan(x) ? asfloat(asuint(x) | 0x400000u) : (x == asfloat(0x7f800000u) ? 0.0f : y);
    return fin ? y : other;
}

bool wgk_hidden_half_safe(float p)
{
    const uint a = asuint(p) & 0x7fffffffu;
    const uint low = a & 0x1fffu;
    return a >= 0x38800000u && a <= 0x477fe000u && (low < 0x0ff8u || low > 0x1008u);
}

// HIP wgk_store_hidden_guarded: the hardware estimate when every product is provably on the same side of its FP16
// rounding midpoint (gfx1201 bound: rsq is within one float of kf_rsqrt for x in [1, 2^34]), else the exact form.
// The four FP16 values go out as one 8-byte store at dst (the same bytes as HIP's four 2-byte stores).
void wgk_store_hidden(uint base, uint off_h, float h[4])
{
    const float x = wgk_hidden_norm2(h);
    const float est = wg_rsq(x);
    bool safe = x >= 1.0f && x <= 17179869184.0f;
    float pe[4];
    [unroll] for (int k = 0; k < 4; ++k) {
        pe[k] = h[k] * est;
        safe = safe && wgk_hidden_half_safe(pe[k]);
    }
    float pr[4];
    if (!safe) {
        const float nu = wgk_kf_rsqrt(x);
        [unroll] for (int k2 = 0; k2 < 4; ++k2) pr[k2] = h[k2] * nu;
    } else {
        [unroll] for (int k3 = 0; k3 < 4; ++k3) pr[k3] = pe[k3];
    }
    wg_st64(base, off_h * 2u, uint2(wg_f2h(pr[0]) | (wg_f2h(pr[1]) << 16), wg_f2h(pr[2]) | (wg_f2h(pr[3]) << 16)));
}

#ifndef KB_BIAS32
#define KB_BIAS32 0
#endif

#if KB_BIAS32
static const int kb_head_row[40] = {
#include "k_head_rows.inc"
};

// Rows 8*hh through 8*hh+7 of the FP32 bias tile.
v8f kb_bias32(uint base, uint off_f)
{
    const uint4 lo = wgk_ld128_a16(base, off_f * 4u);
    const uint4 hi = wgk_ld128_a16(base, off_f * 4u + 16u);
    v8f r;
    r.v[0] = asfloat(lo.x);
    r.v[1] = asfloat(lo.y);
    r.v[2] = asfloat(lo.z);
    r.v[3] = asfloat(lo.w);
    r.v[4] = asfloat(hi.x);
    r.v[5] = asfloat(hi.y);
    r.v[6] = asfloat(hi.z);
    r.v[7] = asfloat(hi.w);
    return r;
}

#endif

#if KB_BASE_FORM
uint4 kb_weight128(uint base, uint off)
{
    return wgk_ld128_a16(base, off);
}
#endif

// ---- the block ----------------------------------------------------------------------------------------------------
// Template parameters as k_block.hip (F16 = 0): C channels, HEADS, MIX (KMIX_*), IN (KIN_MAP / KIN_EMBED / KIN_EXPAND;
// KIN_PADDED as well), CIN input channels of EMBED / EXPAND, CNEXT merge output channels (0: none), HEAD (32 -> 40 output
// head), G windows per workgroup, HIDDEN (hidden-history epilogue; G = 2 head stage; hidden = its pointer), PIX (compact
// pixel head), XG (decoder: x = f16(e + S) through P.y instead of LDS). blk: the workgroup's window group (windows
// G blk .. G blk + G - 1): wg_block_id().
template <int C, int HEADS, int MIX, int IN, int CIN, int CNEXT, bool HEAD, int G, bool HIDDEN, bool PIX, bool XG>
void k_block(uint hidden, uint blk, uint bias32 = 0u, uint compact_head = 0u)
{
#if KB_BIAS32
    // DXC lacks static_assert. A full head cannot use the compact row layout.
    const int compact_bias_requires_pixel_head[!HEAD || PIX ? 1 : -1] = {0};
#endif
    const int KT = C / 16;
    const int DT = HEADS * 2;
    const int HT = 4 * C / 16;
    const int QKV = MIX == KMIX_SOFTMAX ? 3 : 1;
    const bool XS = IN == KIN_EMBED || IN == KIN_EXPAND;
    const bool MS = CNEXT > 0;
    const int KH = QKV == 3 ? 8 * KB_FRAG : 0, VH = 8 * KB_FRAG, XH = (XS || MS) ? 64 * C : 0;
    const bool XA = (MS && !XS) || XG;
    const int LH1 = XA ? max(G * (KH + VH), G * XH) : G * (KH + VH) + G * XH;
    const int LH2 = XA ? max(G * 2 * (KH + VH), G * XH) : G * 2 * (KH + VH) + G * XH;
    // kb_wg_per_wgp(bytes) = 131072 / bytes; NB = 2 if wg(2 LH2) >= min(8, wg(2 LH1)). KB_NB_FORCE (a kernel's
    // defines): that NB instead (the number of head buffers changes only which LDS slots and barriers are used)
#ifndef KB_NB_FORCE
#define KB_NB_FORCE 0
#endif
    const int NB = KB_NB_FORCE ? KB_NB_FORCE : ((131072 / (2 * LH2)) >= min(8, 131072 / (2 * LH1)) ? 2 : 1);
    const uint lds_x = XA ? 0u : (uint)(G * NB * (KH + VH));
    const uint KVW = (uint)(NB * (KH + VH));   // LDS halfs between the windows' K / VT sets

    const uint tid = wg_lane_id();
    const uint lane = tid & 31u;
    const uint wave = (uint)wg_uniform((int)(tid >> 5));   // wave-uniform (scalar): the loops over tiles by wave
    const int n16 = (int)(lane & 15u), hh = (int)(lane >> 4);
    const int nwin = P_nwx * P_nwy;
    const int t = 16 * (int)wave + n16;
#if KB_BIAS32
    const int OFF_BA = 0, OFF_BO = IN == KIN_EMBED ? C : (IN == KIN_EXPAND ? 4 * C : 0);
    const int OFF_B1 = OFF_BO + C, OFF_B2 = OFF_B1 + 4 * C, OFF_BZ = OFF_B2 + C;
    const int OFF_SMX = OFF_BZ + CNEXT;
    const uint kb_b32base = (bias32 + (uint)hh * 32u) ^ wgk_zero();
#endif
#if KB_BASE_FORM
    const uint weight_lane = lane * 16u;
    const uint wb_wa = (P_wa + weight_lane) ^ wgk_zero();
    const uint wb_wqkv = (P_wqkv + weight_lane) ^ wgk_zero();
    const uint wb_wo = (P_wo + weight_lane) ^ wgk_zero();
    const uint wb_w1 = (P_w1 + weight_lane) ^ wgk_zero();
    const uint wb_w2 = (P_w2 + weight_lane) ^ wgk_zero();
    const uint wb_wz = ((KB_BIAS32 && HEAD ? compact_head : P_wz) + weight_lane) ^ wgk_zero();
    const uint wb_mix = (P_mix + (uint)(t * 64 + 8 * hh) * 2u) ^ wgk_zero();
#endif
    int wa[G], wb[G], gx[G], gy[G];
    bool real[G];
    uint mtok[G], xt[G], xl[G], ytok[G];
    [unroll] for (int g = 0; g < G; ++g) {
        const int wraw = (int)blk * G + g;
        const int win = select(wraw < nwin, wraw, nwin - 1);
        wa[g] = (int)((uint)win % (uint)P_nwx);
        wb[g] = (int)((uint)win / (uint)P_nwx);
        gx[g] = 8 * wa[g] - P_ox + (t & 7);
        gy[g] = 8 * wb[g] - P_oy + (t >> 3);
        real[g] = and(and(wraw < nwin, and(gx[g] >= 0, gx[g] < P_w)), and(gy[g] >= 0, gy[g] < P_h));
        mtok[g] = (uint)(wgk_mirror(gy[g], P_h) * P_w + wgk_mirror(gx[g], P_w));
        const uint tok = IN == KIN_PADDED ? (uint)((8 * wb[g] + (t >> 3)) * (8 * P_nwx) + 8 * wa[g] + (t & 7)) : mtok[g];
        xt[g] = tok * (uint)C + 8u * (uint)hh;                         // halfs from P_x (MAP / PADDED)
        xl[g] = lds_x + (uint)(g * XH + t * C + 8 * hh);                // halfs in LDS (XS / MS)
        ytok[g] = real[g] ? (uint)(gy[g] * P_w + gx[g]) : mtok[g];      // this member's map token (in range)
    }

    // g1 first: its loads overlap the block input's
    uint4 g1v[C / 16];
    [unroll] for (int k0 = 0; k0 < KT; ++k0) g1v[k0] = wgk_ld8(P_g1, (uint)(16 * k0 + 8 * hh));

    // 1. the block input
    uint4 n[G][C / 16];
    if (IN == KIN_EMBED) {  // e(t) = max(f16(We in(t) + be), 0) of the mirrored member
        uint4 inp[G], ea[C / 16];
#if KB_BIAS32
        v8f eb[C / 16];
#else
        uint4 eb[C / 16];
#endif
        [unroll] for (int g = 0; g < G; ++g) inp[g] = wgk_ld8(P_x, mtok[g] * 16u + 8u * (uint)hh);
        [unroll] for (int o0 = 0; o0 < KT; ++o0) {
#if KB_BASE_FORM
            ea[o0] = kb_weight128(wb_wa, (uint)(o0 * 512));
#else
            ea[o0] = wgk_ld8(P_wa, (uint)(o0 * 256) + lane * 8u);
#endif
#if KB_BIAS32
            eb[o0] = kb_bias32(kb_b32base, OFF_BA + 16 * o0);
#else
            eb[o0] = wgk_ld8(P_ba, (uint)(16 * o0 + 8 * hh));
#endif
        }
        [unroll] for (int ot = 0; ot < KT; ++ot) {
#if KB_BIAS32
            const v8f b0 = eb[ot];
#else
            const v8f b0 = wgk_widen8(eb[ot]);
#endif
            [unroll] for (int g = 0; g < G; ++g) {
                n[g][ot] = KB_MAXC8W(wgk_round8(wgk_mma(ea[ot], inp[g], b0)), 0u);
                kb_stl(xl[g] + (uint)(16 * ot), n[g][ot]);
            }
        }
    } else if (IN == KIN_EXPAND) {
        // the window's 4 x 4 coarse tokens: column j = lane % 16 is coarse (R0 + j / 4, C0 + j % 4), mirrored
        const int KTI = CIN / 16, TOT = 4 * C / 16;
        const int jr = n16 >> 2, jc = n16 & 3;
        uint4 yb[G][CIN / 16];
        [unroll] for (int g = 0; g < G; ++g) {
            const int cr = (8 * wb[g] - P_oy) / 2 + jr, cc = (8 * wa[g] - P_ox) / 2 + jc;  // even offsets: exact halves
            const uint yc = (uint)(wgk_mirror(cr, P_ch) * P_cw + wgk_mirror(cc, P_cw)) * (uint)CIN + 8u * (uint)hh;
            [unroll] for (int kt = 0; kt < KTI; ++kt) yb[g][kt] = wgk_ld8(P_x, yc + (uint)(16 * kt));
        }
        [loop] for (int tt = (int)wave; tt < TOT; tt += 4) {  // output tile tt of 4C: sub-position s, channels c0 ..
            uint4 wt[CIN / 16];
#if KB_BIAS32
            const v8f b0 = kb_bias32(kb_b32base, OFF_BA + 16 * tt);
#else
            const uint4 bb = wgk_ld8(P_ba, (uint)(16 * tt + 8 * hh));
#endif
#if KB_BASE_FORM
            [unroll] for (int k1 = 0; k1 < KTI; ++k1) wt[k1] = kb_weight128(wb_wa, (uint)((tt * KTI + k1) * 512));
#else
            [unroll] for (int k1 = 0; k1 < KTI; ++k1) wt[k1] = wgk_ld8(P_wa, (uint)((tt * KTI + k1) * 256) + lane * 8u);
#endif
            v8f e[G];
#if !KB_BIAS32
            const v8f b0 = wgk_widen8(bb);
#endif
            [unroll] for (int g = 0; g < G; ++g) e[g] = b0;
            [unroll] for (int kt = 0; kt < KTI; ++kt)
                [unroll] for (int g = 0; g < G; ++g) e[g] = wgk_mma(wt[kt], yb[g][kt], e[g]);
            const int s = (16 * tt) / C, c0 = 16 * tt - s * C;
            const int ft = (2 * jr + (s >> 1)) * 8 + 2 * jc + (s & 1);  // the window member this sub-position feeds
            [unroll] for (int g = 0; g < G; ++g)
                kb_stl(lds_x + (uint)(g * XH + ft * C + c0 + 8 * hh), wgk_round8(e[g]));
        }
        wg_barrier_lds();
        [unroll] for (int g = 0; g < G; ++g) {
            const uint sk = mtok[g] * (uint)C + 8u * (uint)hh;
            uint4 sv[C / 16];
            [unroll] for (int k2 = 0; k2 < KT; ++k2) sv[k2] = wgk_ld8(P_skip, sk + (uint)(16 * k2));
            [unroll] for (int kt = 0; kt < KT; ++kt) {
                n[g][kt] = wgk_add8(kb_ldl(xl[g] + (uint)(16 * kt)), sv[kt]);  // x = f16(e + S); own slots only
                if (!XG)
                    kb_stl(xl[g] + (uint)(16 * kt), n[g][kt]);
                else if (real[g])
                    wgk_st8(P_y, ytok[g] * (uint)C + 8u * (uint)hh + (uint)(16 * kt), n[g][kt]);
            }
        }
        if (XG) wg_barrier_lds();  // the staging is read; LDS goes to K / VT
    } else {
        [unroll] for (int g = 0; g < G; ++g)
            [unroll] for (int kt = 0; kt < KT; ++kt) n[g][kt] = wgk_ld8(P_x, xt[g] + (uint)(16 * kt));
    }

    // (after the block input's loads: the touches' wait coincides with theirs)
    uint pref = 0u;
    if (KB_PREF) {
        pref ^= kb_touch(P_wqkv, HEADS * QKV * 2 * KT * 256, tid);
        pref ^= kb_touch(P_mix, HEADS * 64 * 64, tid);
        pref ^= kb_touch(P_wo, KT * DT * 256, tid);
        pref ^= kb_touch(P_w1, HT * KT * 256, tid);
        pref ^= kb_touch(P_w2, HT * KT * 256, tid);
        if (MS) pref ^= kb_touch(P_wz, (CNEXT / 16) * (4 * C / 16) * 256, tid);
    }

    // normalization: s = f16(sum f16(x^2)) (FP32 sum, channel order), r = f16(1 / sqrt(s)); s = 0 gives r = 0
    float ss[G];
    [unroll] for (int g = 0; g < G; ++g) ss[g] = 0.0f;
    [unroll] for (int kt = 0; kt < KT; ++kt)
        [unroll] for (int g = 0; g < G; ++g) ss[g] = wgk_sumsq8(n[g][kt], ss[g]);
    [unroll] for (int g = 0; g < G; ++g) ss[g] = ss[g] + wgk_xor16(ss[g]);
    [unroll] for (int g = 0; g < G; ++g) {
        const uint4 rv = wgk_splat8h(wgk_norm_reciprocal(wg_f2h(ss[g])));
        [unroll] for (int kt = 0; kt < KT; ++kt) n[g][kt] = KB_MUL8W(n[g][kt], wgk_mul8(rv, g1v[kt]));
    }

    // 2. token mixing, head by head; y accumulates the output projection
    v8f yacc[G][C / 16];
    {
#if !KB_BIAS32
        uint4 bo[C / 16];
        [unroll] for (int o1 = 0; o1 < KT; ++o1) bo[o1] = wgk_ld8(P_bo, (uint)(16 * o1 + 8 * hh));
#endif
        [unroll] for (int ot = 0; ot < KT; ++ot) {
#if KB_BIAS32
            const v8f b0 = kb_bias32(kb_b32base, OFF_BO + 16 * ot);
#else
            const v8f b0 = wgk_widen8(bo[ot]);
#endif
            [unroll] for (int g = 0; g < G; ++g) yacc[g][ot] = b0;
        }
    }

#if !KB_BASE_FORM
    const uint wl = lane * 8u;
    const uint pmix = KB_PREF ? P_mix + wgk_pin(pref) : P_mix;   // KB_PREF: the touched words arrive before the heads
#endif
#ifndef KB_GRPLATE
#define KB_GRPLATE 0
#endif
    // (the head and MLP loops are left to the driver, which unrolls them: an opaque trip count that keeps them rolled
    // measured slower here, d1 +7 us / e1 +10 us with the MLP rolled one tile per iteration, no gain with two)
    [loop] for (int hd = 0; hd < HEADS; ++hd) {
#if KB_BASE_FORM
        const uint wh = (uint)(hd * QKV * 2 * KT * 512);
#else
        const uint wh = wl + (uint)(hd * QKV * 2 * KT * 256);
#endif
        const uint kv = (uint)((NB == 2 ? (hd & 1) : 0) * (KH + VH)) + lane * 8u;   // this lane's slot, window 0
        uint4 q[G][2];
        [unroll] for (int ct = 0; ct < 2; ++ct) {
            if (MIX == KMIX_SOFTMAX) {
                uint4 wq[C / 16], wk[C / 16];
                [unroll] for (int k3 = 0; k3 < KT; ++k3) {
#if KB_BASE_FORM
                    wq[k3] = kb_weight128(wb_wqkv, wh + (uint)((0 * 2 * KT + ct * KT + k3) * 512));
#else
                    wq[k3] = wgk_ld8(P_wqkv, wh + (uint)((0 * 2 * KT + ct * KT + k3) * 256));
#endif
#if KB_BASE_FORM
                    wk[k3] = kb_weight128(wb_wqkv, wh + (uint)((1 * 2 * KT + ct * KT + k3) * 512));
#else
                    wk[k3] = wgk_ld8(P_wqkv, wh + (uint)((1 * 2 * KT + ct * KT + k3) * 256));
#endif
                }
                v8f aq[G], ak[G];
                [unroll] for (int g = 0; g < G; ++g) { aq[g] = wgk_zero8f(); ak[g] = wgk_zero8f(); }
                [unroll] for (int kt = 0; kt < KT; ++kt)
                    [unroll] for (int g = 0; g < G; ++g) {
                        aq[g] = wgk_mma(wq[kt], n[g][kt], aq[g]);
                        ak[g] = wgk_mma(wk[kt], n[g][kt], ak[g]);
                    }
                if (NB == 1 && ct == 0 && hd > 0) wg_barrier_lds();  // every wave has read the previous head's K / VT
                [unroll] for (int g = 0; g < G; ++g) {
                    q[g][ct] = wgk_round8(aq[g]);
                    kb_stl(kv + (uint)g * KVW + (uint)((2 * (int)wave + ct) * KB_FRAG), wgk_round8(ak[g]));  // K tile
                }
            }
            // v^T: rows = channels 16 ct + lane % 16 (B columns), elements = tokens 16 wave + 8 hh + 0..7
            uint4 wv[C / 16];
            [unroll] for (int k4 = 0; k4 < KT; ++k4)
#if KB_BASE_FORM
                wv[k4] = kb_weight128(wb_wqkv, wh + (uint)(((QKV - 1) * 2 * KT + ct * KT + k4) * 512));
#else
                wv[k4] = wgk_ld8(P_wqkv, wh + (uint)(((QKV - 1) * 2 * KT + ct * KT + k4) * 256));
#endif
            v8f av[G];
            [unroll] for (int g = 0; g < G; ++g) av[g] = wgk_zero8f();
            [unroll] for (int kt = 0; kt < KT; ++kt)
                [unroll] for (int g = 0; g < G; ++g) av[g] = wgk_mma(n[g][kt], wv[kt], av[g]);
            if (NB == 1 && MIX == KMIX_FIXED && ct == 0 && hd > 0) wg_barrier_lds();
            [unroll] for (int g = 0; g < G; ++g)
                kb_stl(kv + (uint)g * KVW + (uint)(KH + (4 * ct + (int)wave) * KB_FRAG), wgk_round8(av[g]));  // VT
        }
        wg_barrier_lds();

#if !KB_BASE_FORM
        const uint mrow = (uint)((hd * 64 + t) * 64 + 8 * hh);
#endif
        uint4 mx[4];   // A^T = M[h][t][s] (fixed) or the bias B[h][t][s] (softmax): the same for every window
#if KB_BIAS32
        if (MIX == KMIX_FIXED) {
#endif
#if KB_BASE_FORM
        const uint mix_base = KB_PREF ? wb_mix + wgk_pin(pref) : wb_mix;
        [unroll] for (int s0 = 0; s0 < 4; ++s0)
            mx[s0] = kb_weight128(mix_base, (uint)((hd * 64 * 64 + 16 * s0) * 2));
#else
        [unroll] for (int s0 = 0; s0 < 4; ++s0) mx[s0] = wgk_ld8(pmix, mrow + (uint)(16 * s0));
#endif
#if KB_BIAS32
        }
        const uint smx_row = OFF_SMX + (uint)((hd * 64 + t) * 64 + 8 * hh);
#endif
        uint4 z[G][2];
        [unroll] for (int g = 0; g < G; ++g) {
            const uint lk = kv + (uint)g * KVW;
            const uint lv = lk + (uint)KH;
            uint4 pa[4];
            if (MIX == KMIX_FIXED) {
                [unroll] for (int s1 = 0; s1 < 4; ++s1) pa[s1] = mx[s1];
            } else {
                uint4 kf[8];
                [unroll] for (int f0 = 0; f0 < 8; ++f0) kf[f0] = kb_ldl(lk + (uint)(f0 * KB_FRAG));
                uint4 gg[4];
                [unroll] for (int st = 0; st < 4; ++st) {
#if KB_BIAS32
                    v8f sc = kb_bias32(bias32, smx_row + (uint)(16 * st));
#else
                    v8f sc = wgk_widen8(mx[st]);  // + B[h][t][s]
#endif
                    [unroll] for (int ct = 0; ct < 2; ++ct) sc = wgk_mma(kf[2 * st + ct], q[g][ct], sc);
                    const uint4 u = KB_CLAMP8W(wgk_round8(sc), WGK_SOFTMAX_C | 0x8000u, WGK_SOFTMAX_C);
                    const uint4 u2 = wgk_mul8(u, u);
                    const uint4 k2 = wgk_mul8(wgk_splat8h(WGK_SOFTMAX_KAPPA), u2);
                    const uint4 o = KB_SUB8S(wgk_splat8h(WGK_H_ONE), k2);
                    gg[st] = KB_MUL8S(u, o);
                }
                wgk_softmax_weights(gg, pa);
            }
            // z_h^T = VT A^T (KB_GRPLATE: each channel tile's 4 VT fragments just before its 4 products)
            uint4 vf[8];
            if (!KB_GRPLATE)
                [unroll] for (int f1 = 0; f1 < 8; ++f1) vf[f1] = kb_ldl(lv + (uint)(f1 * KB_FRAG));
            [unroll] for (int ct = 0; ct < 2; ++ct) {
                if (KB_GRPLATE)
                    [unroll] for (int f2 = 0; f2 < 4; ++f2) vf[4 * ct + f2] = kb_ldl(lv + (uint)((4 * ct + f2) * KB_FRAG));
                v8f az = wgk_zero8f();
                [unroll] for (int st = 0; st < 4; ++st) az = wgk_mma(vf[4 * ct + st], pa[st], az);
                z[g][ct] = wgk_round8(az);
            }
        }
        // the output projection: each Wo tile once for the G windows (KB_GRPLATE: one channel tile's Wo tiles at a time)
        uint4 wo[2][C / 16];
        if (!KB_GRPLATE)
            [unroll] for (int c1 = 0; c1 < 2; ++c1)
                [unroll] for (int o2 = 0; o2 < KT; ++o2)
#if KB_BASE_FORM
                    wo[c1][o2] = kb_weight128(wb_wo, (uint)((o2 * DT + 2 * hd + c1) * 512));
#else
                    wo[c1][o2] = wgk_ld8(P_wo, (uint)((o2 * DT + 2 * hd + c1) * 256) + lane * 8u);
#endif
        [unroll] for (int ct = 0; ct < 2; ++ct) {
            if (KB_GRPLATE)
                [unroll] for (int o2 = 0; o2 < KT; ++o2)
#if KB_BASE_FORM
                    wo[ct][o2] = kb_weight128(wb_wo, (uint)((o2 * DT + 2 * hd + ct) * 512));
#else
                    wo[ct][o2] = wgk_ld8(P_wo, (uint)((o2 * DT + 2 * hd + ct) * 256) + lane * 8u);
#endif
            [unroll] for (int ot = 0; ot < KT; ++ot)
                [unroll] for (int g = 0; g < G; ++g) yacc[g][ot] = wgk_mma(wo[ct][ot], z[g][ct], yacc[g][ot]);
        }
    }

    // 3. x1 = f16(x + y);  4. a = f16(x1 g2)
    uint4 x1[G][C / 16], a[G][C / 16];
    {
        uint4 g2v[C / 16], xin[G][C / 16];
        [unroll] for (int k5 = 0; k5 < KT; ++k5) g2v[k5] = wgk_ld8(P_g2, (uint)(16 * k5 + 8 * hh));
        [unroll] for (int kt = 0; kt < KT; ++kt)
            [unroll] for (int g = 0; g < G; ++g) {
                if (XG)
                    xin[g][kt] = wgk_ld8(P_y, ytok[g] * (uint)C + 8u * (uint)hh + (uint)(16 * kt));
                else if (XS)
                    xin[g][kt] = kb_ldl(xl[g] + (uint)(16 * kt));
                else
                    xin[g][kt] = wgk_ld8(P_x, xt[g] + (uint)(16 * kt));
            }
        [unroll] for (int kt = 0; kt < KT; ++kt)
            [unroll] for (int g = 0; g < G; ++g) {
                const uint4 xv = XG ? (real[g] ? xin[g][kt] : wgk_zero8()) : xin[g][kt];
                x1[g][kt] = wgk_add8(xv, wgk_round8(yacc[g][kt]));
                a[g][kt] = KB_MUL8W(x1[g][kt], g2v[kt]);
            }
    }

    // x1 waits in LDS during the MLP (X1L: the two-window C = 64 stages, whose MLP is the register peak): each lane
    // its own x staging slots (xl); where x shares the K / VT space (XA) every wave must have read the last head first
#ifndef KB_X1LDS
#define KB_X1LDS 0
#endif
    const bool X1L = KB_X1LDS && G == 2 && C >= 64 && XH > 0;
    if (X1L) {
        if (XA) wg_barrier_lds();
        [unroll] for (int g = 0; g < G; ++g)
            [unroll] for (int kt = 0; kt < KT; ++kt) kb_stl(xl[g] + (uint)(16 * kt), x1[g][kt]);
    }

    // 4. MLP: per hidden tile p = f16(W1 a + b1), h = act(p), acc += W2 h (FP32, started from b2)
    v8f acc[G][C / 16];
    {
#if !KB_BIAS32
        uint4 b2v[C / 16];
        [unroll] for (int o3 = 0; o3 < KT; ++o3) b2v[o3] = wgk_ld8(P_b2, (uint)(16 * o3 + 8 * hh));
#endif
        [unroll] for (int ot = 0; ot < KT; ++ot) {
#if KB_BIAS32
            const v8f b0 = kb_bias32(kb_b32base, OFF_B2 + 16 * ot);
#else
            const v8f b0 = wgk_widen8(b2v[ot]);
#endif
            [unroll] for (int g = 0; g < G; ++g) acc[g][ot] = b0;
        }
    }
    [loop] for (int ht = 0; ht < HT; ++ht) {
        uint4 w1t[C / 16], w2t[C / 16];
#if KB_BIAS32
        const v8f b0 = kb_bias32(kb_b32base, OFF_B1 + 16 * ht);
#else
        const uint4 b1v = wgk_ld8(P_b1, (uint)(16 * ht + 8 * hh));
#endif
#if KB_BASE_FORM
        [unroll] for (int k6 = 0; k6 < KT; ++k6) w1t[k6] = kb_weight128(wb_w1, (uint)((ht * KT + k6) * 512));
#else
        [unroll] for (int k6 = 0; k6 < KT; ++k6) w1t[k6] = wgk_ld8(P_w1, (uint)((ht * KT + k6) * 256) + lane * 8u);
#endif
#ifndef KB_W2LATE
#define KB_W2LATE 0
#endif
        if (!KB_W2LATE)
#if KB_BASE_FORM
            [unroll] for (int o4 = 0; o4 < KT; ++o4) w2t[o4] = kb_weight128(wb_w2, (uint)((ht * KT + o4) * 512));
#else
            [unroll] for (int o4 = 0; o4 < KT; ++o4) w2t[o4] = wgk_ld8(P_w2, (uint)((ht * KT + o4) * 256) + lane * 8u);
#endif
        v8f p[G];
#if !KB_BIAS32
        const v8f b0 = wgk_widen8(b1v);
#endif
        [unroll] for (int g = 0; g < G; ++g) p[g] = b0;
        [unroll] for (int kt = 0; kt < KT; ++kt)
            [unroll] for (int g = 0; g < G; ++g) p[g] = wgk_mma(w1t[kt], a[g][kt], p[g]);
        uint4 hv[G];
        [unroll] for (int g = 0; g < G; ++g) hv[g] = KB_ACT8W(wgk_round8(p[g]));
#ifndef KB_W2SEQ
#define KB_W2SEQ 0   // with KB_W2LATE: one W2 tile at a time, loaded just before its own products (not all KT before the first)
#endif
        if (KB_W2LATE && !KB_W2SEQ)   // W2's tiles just before their multiplies (fewer live registers across the first GEMM)
#if KB_BASE_FORM
            [unroll] for (int o4 = 0; o4 < KT; ++o4) w2t[o4] = kb_weight128(wb_w2, (uint)((ht * KT + o4) * 512));
#else
            [unroll] for (int o4 = 0; o4 < KT; ++o4) w2t[o4] = wgk_ld8(P_w2, (uint)((ht * KT + o4) * 256) + lane * 8u);
#endif
        [unroll] for (int ot = 0; ot < KT; ++ot) {
            if (KB_W2LATE && KB_W2SEQ)
#if KB_BASE_FORM
                w2t[ot] = kb_weight128(wb_w2, (uint)((ht * KT + ot) * 512));
#else
                w2t[ot] = wgk_ld8(P_w2, (uint)((ht * KT + ot) * 256) + lane * 8u);
#endif
            [unroll] for (int g = 0; g < G; ++g) acc[g][ot] = wgk_mma(w2t[ot], hv[g], acc[g][ot]);
        }
    }

    // 5. x2, stored for the real positions (the stores' base pinned to every x2 fragment: products before the branch)
    uint pin5 = 0u;
    if (X1L)
        [unroll] for (int g = 0; g < G; ++g)
            [unroll] for (int kt = 0; kt < KT; ++kt) x1[g][kt] = kb_ldl(xl[g] + (uint)(16 * kt));
    [unroll] for (int g = 0; g < G; ++g)
        [unroll] for (int ot = 0; ot < KT; ++ot) {
            x1[g][ot] = KB_ADD8W(x1[g][ot], wgk_round8(acc[g][ot]));  // x1 now holds x2
            pin5 ^= x1[g][ot].x;
        }
    const uint py = P_y + wgk_pin(pin5);
    [unroll] for (int g = 0; g < G; ++g) {
        if (real[g] && P_y != 0u) {
            [unroll] for (int o5 = 0; o5 < KT; ++o5)
                wgk_st8(py, ytok[g] * (uint)C + 8u * (uint)hh + (uint)(16 * o5), x1[g][o5]);
        }
    }

    if (MS) {  // 2x2 merge of each window's 16 groups: f16(Wm concat(TL, TR, BL, BR) + bm)
        if (XA && !X1L) wg_barrier_lds();  // x shares the K / VT space: every wave has read the last head's
        [unroll] for (int g = 0; g < G; ++g)
            [unroll] for (int ot = 0; ot < KT; ++ot) kb_stl(xl[g] + (uint)(16 * ot), real[g] ? x1[g][ot] : wgk_zero8());
        wg_barrier_lds();
        const int KTM = 4 * C / 16, NTM = CNEXT / 16;
        const int gr = n16 >> 2, gc = n16 & 3;  // group (column n16): window rows 2 gr, 2 gr + 1, columns 2 gc, 2 gc + 1
        const int mw = (P_w + 1) / 2, mh = (P_h + 1) / 2;
        [loop] for (int tt = (int)wave; tt < NTM; tt += 4) {
            v8f m[G];
#if KB_BIAS32
            const v8f b0 = kb_bias32(kb_b32base, OFF_BZ + 16 * tt);
#else
            const v8f b0 = wgk_widen8(wgk_ld8(P_bz, (uint)(16 * tt + 8 * hh)));
#endif
            [unroll] for (int g = 0; g < G; ++g) m[g] = b0;
            [unroll] for (int k7 = 0; k7 < KTM; k7 += 4) {   // in groups of 4 input tiles (HIP: unroll 4)
                uint4 wm[4], xm[G][4];
                [unroll] for (int i = 0; i < 4; ++i) {
                    const int kt = k7 + i;
                    const int s = (16 * kt) / C, c0 = 16 * kt - s * C;
                    const int ft = (2 * gr + (s >> 1)) * 8 + 2 * gc + (s & 1);
#if KB_BASE_FORM
                    wm[i] = kb_weight128(wb_wz, (uint)((tt * KTM + kt) * 512));
#else
                    wm[i] = wgk_ld8(P_wz, (uint)((tt * KTM + kt) * 256) + lane * 8u);
#endif
                    [unroll] for (int g = 0; g < G; ++g) xm[g][i] = kb_ldl(lds_x + (uint)(g * XH + ft * C + c0 + 8 * hh));
                }
                [unroll] for (int i2 = 0; i2 < 4; ++i2)
                    [unroll] for (int g = 0; g < G; ++g) m[g] = wgk_mma(wm[i2], xm[g][i2], m[g]);
            }
            uint4 mo[G];
            uint pinm = 0u;
            [unroll] for (int g = 0; g < G; ++g) {
                mo[g] = wgk_round8(m[g]);
                pinm ^= mo[g].x;
            }
            const uint pm = P_merged + wgk_pin(pinm);
            [unroll] for (int g = 0; g < G; ++g) {
                const int my = (8 * wb[g] - P_oy) / 2 + gr, mxx = (8 * wa[g] - P_ox) / 2 + gc;
                const bool here = (int)blk * G + g < nwin;
                if (here && my >= 0 && my < mh && mxx >= 0 && mxx < mw)
                    wgk_st8(pm, (uint)((my * mw + mxx) * CNEXT + 16 * tt + 8 * hh), mo[g]);
            }
        }
    }

    if (HEAD) {  // out = f16(Wh x2 + bh)
        const int NT = KB_BIAS32 ? 2 : 3;
        uint4 heads[G][NT], whd[NT][C / 16];
#if KB_BIAS32
        v8f bhd[NT];
#else
        uint4 bhd[NT];
#endif
        [unroll] for (int o6 = 0; o6 < NT; ++o6) {
#if KB_BIAS32
            bhd[o6] = kb_bias32(kb_b32base, OFF_BZ + 16 * o6);
#else
            bhd[o6] = wgk_ld8(P_bz, (uint)(16 * o6 + 8 * hh));
#endif
#if KB_BASE_FORM
            [unroll] for (int k8 = 0; k8 < KT; ++k8)
                whd[o6][k8] = kb_weight128(wb_wz, (uint)((o6 * KT + k8) * 512));
#else
            [unroll] for (int k8 = 0; k8 < KT; ++k8)
                whd[o6][k8] = wgk_ld8(KB_BIAS32 ? compact_head : P_wz, (uint)((o6 * KT + k8) * 256) + lane * 8u);
#endif
        }
        [unroll] for (int ot = 0; ot < NT; ++ot) {
            v8f o[G];
#if KB_BIAS32
            const v8f b0 = bhd[ot];
#else
            const v8f b0 = wgk_widen8(bhd[ot]);
#endif
            [unroll] for (int g = 0; g < G; ++g) o[g] = b0;
            [unroll] for (int kt = 0; kt < KT; ++kt)
                [unroll] for (int g = 0; g < G; ++g) o[g] = wgk_mma(whd[ot][kt], x1[g][kt], o[g]);
            [unroll] for (int g = 0; g < G; ++g) heads[g][ot] = wgk_round8(o[g]);
        }
        uint pinh = 0u;
        [unroll] for (int g = 0; g < G; ++g)
            [unroll] for (int o8 = 0; o8 < NT; ++o8) pinh ^= heads[g][o8].x;
        const uint phead = P_head + wgk_pin(pinh);   // the head products before the predicated stores
        if (!PIX) {
            [unroll] for (int g = 0; g < G; ++g)
                [unroll] for (int o7 = 0; o7 < NT; ++o7)
                    if (real[g] && 16 * o7 + 8 * hh < 40)
                        wgk_st8(phead, ytok[g] * 40u + (uint)(16 * o7 + 8 * hh), heads[g][o7]);
        } else {
            // Compact pixel head (k_block.hip): P.head is [2 h][2 w][4] FP16; slot k of token (gx, gy) at row
            // 2 gy + (k >> 1), column 2 gx + (k & 1) holds (channel 10 k, 10 k + 1, 10 k + 9, 0). A lane of half hh has
            // channels 10 k, 10 k + 1 of slots k = hh and 2 + hh and gets the third value from its partner lane.
            [unroll] for (int g = 0; g < G; ++g) {
#if KB_BIAS32
                // Each half owns the three channels of both compact slots in the first packed tile.
                const uint4 h0 = heads[g][0];
                const uint a0 = wgk_el(h0, kb_head_row[0] % 8), a1 = wgk_el(h0, kb_head_row[1] % 8);
                const uint c0 = wgk_el(h0, kb_head_row[20] % 8), c1 = wgk_el(h0, kb_head_row[21] % 8);
                const uint rcv = wgk_el(h0, kb_head_row[9] % 8) | (wgk_el(h0, kb_head_row[29] % 8) << 16);
#else
                const uint4 h0 = heads[g][0], h1 = heads[g][1], h2 = heads[g][2];
                const uint a0 = hh ? wgk_el(h0, 2u) : wgk_el(h0, 0u), a1 = hh ? wgk_el(h0, 3u) : wgk_el(h0, 1u);
                const uint c0 = hh ? wgk_el(h1, 6u) : wgk_el(h1, 4u), c1 = hh ? wgk_el(h1, 7u) : wgk_el(h1, 5u);
                // to the partner: channel 9 + 10 k of its slots (hh = 0 holds 19, 39; hh = 1 holds 9, 29)
                const uint snd = (hh ? wgk_el(h0, 1u) : wgk_el(h1, 3u)) | ((hh ? wgk_el(h1, 5u) : wgk_el(h2, 7u)) << 16);
                const uint rcv = wgk_xor16_u(snd);
#endif
                if (real[g]) {
                    const uint d = ((uint)(2 * gy[g]) * (uint)(2 * P_w) + (uint)(2 * gx[g] + hh)) * 4u;
                    wg_st64(phead, d * 2u, uint2(a0 | (a1 << 16), rcv & 0xffffu));
                    wg_st64(phead, (d + (uint)(2 * P_w) * 4u) * 2u, uint2(c0 | (c1 << 16), rcv >> 16));
                }
            }
        }
        if (HIDDEN) {
            // Each lane half owns one of the G = 2 windows for this epilogue; the partner lane sends the opposite
            // window's fragment, so xor16 supplies the missing half. All lanes exchange before the real-position guard.
            float hs[4];
            [unroll] for (int k = 0; k < 4; ++k) {
                float v[5];
                [unroll] for (int j = 0; j < 5; ++j) {
                    const int c = 10 * k + 3 + j;
#if KB_BIAS32
                    const int r = kb_head_row[c];
#else
                    const int r = c;
#endif
                    const float own = hh == 0 ? wgk_elf(heads[0][r / 16], (uint)(r % 8)) : wgk_elf(heads[G - 1][r / 16], (uint)(r % 8));
                    const float snd = hh == 0 ? wgk_elf(heads[G - 1][r / 16], (uint)(r % 8)) : wgk_elf(heads[0][r / 16], (uint)(r % 8));
                    const float other = wgk_xor16(snd);
                    v[j] = ((r % 16) / 8) == hh ? own : other;
                }
                // opaque: (0.2 s) est must stay two products (the driver regroups multiplication chains)
                hs[k] = asfloat(wgk_opq(asuint(0.2f * ((((v[0] + v[1]) + v[2]) + v[3]) + v[4]))));
            }
            const bool rh = hh == 0 ? real[0] : real[G - 1];
            const uint tk = hh == 0 ? ytok[0] : ytok[G - 1];
            if (rh) wgk_store_hidden(hidden, tk * 4u, hs);
        }
    }
}
