// SPDX-License-Identifier: MIT
// Model K block of spec 41 with its fused stage neighbours, native RDNA 3 (gfx11) form: the k_block template of
// k_block.hlsli with every matrix product as ONE K = 16 instruction on operands laid out by the kernel itself
// (wg_gfx11.hlsli "native K = 16 operands"; needs the translator's K = 16 opcode). WrongGPU's own code. Developer-only:
// kernels/k_block_g11.hlsl declares the stages gfx=11; no RDNA 4 kernel includes this file, and the self-check digest
// does not cover it (its kernels are not the self-check's).
//
// Arithmetic: everything but the product operands is k_block.hlsli's, in its order: the same bias starts, the same tiles
// in the same order, the same roundings, FP16 steps and opaque zeros. Every product sums K in index order (K = 16 per
// instruction, the order of the shortcut's operands), so a stage should give the shortcut's bits.
//
// Layout (the point of this form): the weight blob is the RDNA 4 one (k_pack.py, d_row gfx12) and every accumulator
// comes out in the gfx12 layout, so all code between products is k_block.hlsli's:
//  * A operands read from a tile in memory or LDS (weights, the K tiles, the VT tiles) are loaded as the two 16-byte
//    slots of the row wg_a_row(lane % 16) (wg_ldA16 / kn_ldlA): the row map puts the accumulator rows in gfx12 order,
//    with no exchange;
//  * B operands read from memory or LDS (the embed and expand inputs, Wv, the fixed mix table, the merge's x staging)
//    are loaded as the 16 contiguous values of the lane's column (two slots), with no exchange;
//  * B operands made in registers (n, q, z, a, the MLP's hidden tile, x2 for the head, the softmax weights) are made once
//    with one half-wave exchange (wg_b16) and reused for every output tile;
//  * v^T = n^T Wv^T takes n's B form as its A operand (rows = tokens, no row map), so its result has the instruction's
//    row order (token 2 i + h); wg_rows_to_gfx12 puts each VT fragment into the gfx12 order before it goes to LDS (one
//    exchange of 2 dwords), where z reads it as an A operand like a K tile.
// Knobs (numbers only; the template's arithmetic does not depend on them):
//  KN_NB_HOIST 1 (default): n's B form is made once per block and kept (8 VGPRs per tile instead of 4); 0: remade for
//    each head (one exchange per tile and head, fewer live registers).
#pragma once

#include "k_block.hlsli"

#if WG_GFX != 11
#error "k_block_g11.hlsli is the RDNA 3 form (build_dx12.py --gfx 11)"
#endif

#ifndef KN_NB_HOIST
#define KN_NB_HOIST 1
#endif
// Register knobs of the two-window encoders (numbers only; none changes an operand's value or a sum's order):
//  KN_NB_HOIST 2: n's B form made at each use (one exchange per product), never held.
//  KN_VSTREAM 1: q, k and v^T load each weight tile at its product instead of all KT tiles first.
//  KN_ZSTREAM 1: per window and channel tile, z's product and then its Wo products, loading the VT tile, the mix row
//    (fixed mix) and the Wo tile at each product; z is never held for both windows.
//  KN_AB 1: a's B form made at each hidden tile from the held a (one exchange per product); 2: a staged in the x slots
//    in LDS (each lane its own token's slots) and read back as 16 contiguous channels per product (no exchange, nothing
//    held). Needs the x staging (a merge or an input staging) and not KB_X1LDS.
//  KN_MERGE_ROLL 1: the merge's K loop stays a loop (opaque trip count), KN_MERGE_STEP tiles per iteration.
#ifndef KN_VSTREAM
#define KN_VSTREAM 0
#endif
#ifndef KN_ZSTREAM
#define KN_ZSTREAM 0
#endif
#ifndef KN_AB
#define KN_AB 0
#endif
//  KN_XPIN 1: the residual's reload of x (KIN_MAP) has its base pinned to the last head's output, so the driver cannot
//    merge it with the block-input load and hold x through the heads.
#ifndef KN_XPIN
#define KN_XPIN 0
#endif
//  KN_HEAD_ROLL 1: the head loop stays a loop (opaque trip count): the driver cannot merge the heads' identical
//    exchanges and loads (n's B form, Wv / Wo tiles of one head are distinct, but per-head remade forms are not held).
#ifndef KN_HEAD_ROLL
#define KN_HEAD_ROLL 0
#endif
#ifndef KN_MERGE_ROLL
#define KN_MERGE_ROLL 0
#endif
#ifndef KN_MERGE_STEP
#define KN_MERGE_STEP 4
#endif

// A from a 16 x 16 tile in LDS (gfx12 fragment order, tile_h in halfs): the two slots of row wg_a_row(lane % 16).
wg_h16 kn_ldlA(uint tile_h)
{
    const uint r = wg_a_row_lane();
    wg_h16 a;
    a.lo = kb_ldl(tile_h + 8u * r);
    a.hi = kb_ldl(tile_h + 8u * (16u + r));
    return a;
}
// B: 16 contiguous FP16 values (one token's 16 channels, one query's 16 keys) in LDS / in the arena (offsets in halfs).
wg_h16 kn_ldlB(uint off_h)
{
    wg_h16 b;
    b.lo = kb_ldl(off_h);
    b.hi = kb_ldl(off_h + 8u);
    return b;
}
wg_h16 kn_ldB(uint base, uint off_h)
{
    wg_h16 b;
    b.lo = wgk_ld8(base, off_h);
    b.hi = wgk_ld8(base, off_h + 8u);
    return b;
}

// n's B form at a product: held (KN_NB_HOIST 0 / 1) or made here (2).
wg_h16 kn_nbsel(uint4 own, wg_h16 held)
{
    if (KN_NB_HOIST == 2) return wg_b16(own);
    return held;
}

// Template parameters as k_block (k_block.hlsli).
template <int C, int HEADS, int MIX, int IN, int CIN, int CNEXT, bool HEAD, int G, bool HIDDEN, bool PIX, bool XG>
void k_block_n11(uint hidden, uint blk)
{
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
    const int NB = KB_NB_FORCE ? KB_NB_FORCE : ((131072 / (2 * LH2)) >= min(8, 131072 / (2 * LH1)) ? 2 : 1);
    const uint lds_x = XA ? 0u : (uint)(G * NB * (KH + VH));
    const uint KVW = (uint)(NB * (KH + VH));

    const uint tid = wg_lane_id();
    const uint lane = tid & 31u;
    const uint wave = (uint)wg_uniform((int)(tid >> 5));
    const int n16 = (int)(lane & 15u), hh = (int)(lane >> 4);
    const int nwin = P_nwx * P_nwy;
    const int t = 16 * (int)wave + n16;
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
        xt[g] = tok * (uint)C + 8u * (uint)hh;
        xl[g] = lds_x + (uint)(g * XH + t * C + 8 * hh);
        ytok[g] = real[g] ? (uint)(gy[g] * P_w + gx[g]) : mtok[g];
    }

    uint4 g1v[C / 16];
    [unroll] for (int k0 = 0; k0 < KT; ++k0) g1v[k0] = wgk_ld8(P_g1, (uint)(16 * k0 + 8 * hh));

    // 1. the block input
    uint4 n[G][C / 16];
    if (IN == KIN_EMBED) {  // e(t) = max(f16(We in(t) + be), 0); B: the token's 16 input channels, A: We's row-mapped tile
        wg_h16 inp[G], ea[C / 16];
        uint4 eb[C / 16];
        [unroll] for (int g = 0; g < G; ++g) inp[g] = kn_ldB(P_x, mtok[g] * 16u);
        [unroll] for (int o0 = 0; o0 < KT; ++o0) {
            ea[o0] = wg_ldA16(P_wa, (uint)(o0 * 256));
            eb[o0] = wgk_ld8(P_ba, (uint)(16 * o0 + 8 * hh));
        }
        [unroll] for (int ot = 0; ot < KT; ++ot) {
            const v8f b0 = wgk_widen8(eb[ot]);
            [unroll] for (int g = 0; g < G; ++g) {
                n[g][ot] = KB_MAXC8W(wgk_round8(wg_mma16(ea[ot], inp[g], b0)), 0u);
                kb_stl(xl[g] + (uint)(16 * ot), n[g][ot]);
            }
        }
    } else if (IN == KIN_EXPAND) {
        const int KTI = CIN / 16, TOT = 4 * C / 16;
        const int jr = n16 >> 2, jc = n16 & 3;
        wg_h16 yb[G][CIN / 16];   // B: the coarse token's 16 channels of each input tile, straight from memory
        [unroll] for (int g = 0; g < G; ++g) {
            const int cr = (8 * wb[g] - P_oy) / 2 + jr, cc = (8 * wa[g] - P_ox) / 2 + jc;
            const uint yc = (uint)(wgk_mirror(cr, P_ch) * P_cw + wgk_mirror(cc, P_cw)) * (uint)CIN;
            [unroll] for (int kt = 0; kt < KTI; ++kt) yb[g][kt] = kn_ldB(P_x, yc + (uint)(16 * kt));
        }
        [loop] for (int tt = (int)wave; tt < TOT; tt += 4) {
            wg_h16 wt[CIN / 16];
            const uint4 bb = wgk_ld8(P_ba, (uint)(16 * tt + 8 * hh));
            [unroll] for (int k1 = 0; k1 < KTI; ++k1) wt[k1] = wg_ldA16(P_wa, (uint)((tt * KTI + k1) * 256));
            v8f e[G];
            const v8f b0 = wgk_widen8(bb);
            [unroll] for (int g = 0; g < G; ++g) e[g] = b0;
            [unroll] for (int kt = 0; kt < KTI; ++kt)
                [unroll] for (int g = 0; g < G; ++g) e[g] = wg_mma16(wt[kt], yb[g][kt], e[g]);
            const int s = (16 * tt) / C, c0 = 16 * tt - s * C;
            const int ft = (2 * jr + (s >> 1)) * 8 + 2 * jc + (s & 1);
            [unroll] for (int g = 0; g < G; ++g)
                kb_stl(lds_x + (uint)(g * XH + ft * C + c0 + 8 * hh), wgk_round8(e[g]));
        }
        wg_barrier_lds();
        [unroll] for (int g = 0; g < G; ++g) {
            const uint sk = mtok[g] * (uint)C + 8u * (uint)hh;
            uint4 sv[C / 16];
            [unroll] for (int k2 = 0; k2 < KT; ++k2) sv[k2] = wgk_ld8(P_skip, sk + (uint)(16 * k2));
            [unroll] for (int kt = 0; kt < KT; ++kt) {
                n[g][kt] = wgk_add8(kb_ldl(xl[g] + (uint)(16 * kt)), sv[kt]);
                if (!XG)
                    kb_stl(xl[g] + (uint)(16 * kt), n[g][kt]);
                else if (real[g])
                    wgk_st8(P_y, ytok[g] * (uint)C + 8u * (uint)hh + (uint)(16 * kt), n[g][kt]);
            }
        }
        if (XG) wg_barrier_lds();
    } else {
        [unroll] for (int g = 0; g < G; ++g)
            [unroll] for (int kt = 0; kt < KT; ++kt) n[g][kt] = wgk_ld8(P_x, xt[g] + (uint)(16 * kt));
    }

    uint pref = 0u;
    if (KB_PREF) {
        pref ^= kb_touch(P_wqkv, HEADS * QKV * 2 * KT * 256, tid);
        pref ^= kb_touch(P_mix, HEADS * 64 * 64, tid);
        pref ^= kb_touch(P_wo, KT * DT * 256, tid);
        pref ^= kb_touch(P_w1, HT * KT * 256, tid);
        pref ^= kb_touch(P_w2, HT * KT * 256, tid);
        if (MS) pref ^= kb_touch(P_wz, (CNEXT / 16) * (4 * C / 16) * 256, tid);
    }

    float ss[G];
    [unroll] for (int g = 0; g < G; ++g) ss[g] = 0.0f;
    [unroll] for (int kt = 0; kt < KT; ++kt)
        [unroll] for (int g = 0; g < G; ++g) ss[g] = wgk_sumsq8(n[g][kt], ss[g]);
    [unroll] for (int g = 0; g < G; ++g) ss[g] = ss[g] + wgk_xor16(ss[g]);
    [unroll] for (int g = 0; g < G; ++g) {
        const uint4 rv = wgk_splat8h(wgk_norm_reciprocal(wg_f2h(ss[g])));
        [unroll] for (int kt = 0; kt < KT; ++kt) n[g][kt] = KB_MUL8W(n[g][kt], wgk_mul8(rv, g1v[kt]));
    }
    // n's B form (the token's 16 channels of each tile): the B operand of q and k, and the A operand of v^T
    wg_h16 nb[G][C / 16];
    if (KN_NB_HOIST)
        [unroll] for (int g = 0; g < G; ++g)
            [unroll] for (int kt = 0; kt < KT; ++kt) nb[g][kt] = wg_b16(n[g][kt]);

    // 2. token mixing, head by head; y accumulates the output projection
    v8f yacc[G][C / 16];
    {
        uint4 bo[C / 16];
        [unroll] for (int o1 = 0; o1 < KT; ++o1) bo[o1] = wgk_ld8(P_bo, (uint)(16 * o1 + 8 * hh));
        [unroll] for (int ot = 0; ot < KT; ++ot) {
            const v8f b0 = wgk_widen8(bo[ot]);
            [unroll] for (int g = 0; g < G; ++g) yacc[g][ot] = b0;
        }
    }

    const uint wl = lane * 8u;
    const uint pmix = KB_PREF ? P_mix + wgk_pin(pref) : P_mix;
#if KN_HEAD_ROLL
    [loop] for (int hd = 0; hd < HEADS + (int)wgk_zero(); ++hd) {
#else
    [loop] for (int hd = 0; hd < HEADS; ++hd) {
#endif
        const uint whn = (uint)(hd * QKV * 2 * KT * 256);                         // this head's tiles (halfs)
        const uint kvb = (uint)((NB == 2 ? (hd & 1) : 0) * (KH + VH));            // its K / VT set, window 0
        const uint kv = kvb + wl;                                                  // this lane's slot
        if (!KN_NB_HOIST)
            [unroll] for (int g = 0; g < G; ++g)
                [unroll] for (int kt = 0; kt < KT; ++kt) nb[g][kt] = wg_b16(n[g][kt]);
        wg_h16 qb[G][2];
        [unroll] for (int ct = 0; ct < 2; ++ct) {
            if (MIX == KMIX_SOFTMAX) {
                wg_h16 wq[C / 16], wk[C / 16];
                if (!KN_VSTREAM)
                    [unroll] for (int k3 = 0; k3 < KT; ++k3) {
                        wq[k3] = wg_ldA16(P_wqkv, whn + (uint)((0 * 2 * KT + ct * KT + k3) * 256));
                        wk[k3] = wg_ldA16(P_wqkv, whn + (uint)((1 * 2 * KT + ct * KT + k3) * 256));
                    }
                v8f aq[G], ak[G];
                [unroll] for (int g = 0; g < G; ++g) { aq[g] = wgk_zero8f(); ak[g] = wgk_zero8f(); }
                [unroll] for (int kt = 0; kt < KT; ++kt) {
                    if (KN_VSTREAM) {
                        wq[kt] = wg_ldA16(P_wqkv, whn + (uint)((0 * 2 * KT + ct * KT + kt) * 256));
                        wk[kt] = wg_ldA16(P_wqkv, whn + (uint)((1 * 2 * KT + ct * KT + kt) * 256));
                    }
                    [unroll] for (int g = 0; g < G; ++g) {
                        const wg_h16 nbt = kn_nbsel(n[g][kt], nb[g][kt]);
                        aq[g] = wg_mma16(wq[kt], nbt, aq[g]);
                        ak[g] = wg_mma16(wk[kt], nbt, ak[g]);
                    }
                }
                if (NB == 1 && ct == 0 && hd > 0) wg_barrier_lds();
                [unroll] for (int g = 0; g < G; ++g) {
                    qb[g][ct] = wg_b16(wgk_round8(aq[g]));   // q: only the scores' B operand
                    kb_stl(kv + (uint)g * KVW + (uint)((2 * (int)wave + ct) * KB_FRAG), wgk_round8(ak[g]));  // K tile
                }
            }
            // v^T: A = n's B form (rows = tokens 16 wave + 0..15 in the instruction's row order), B = Wv's column
            wg_h16 wv[C / 16];
            if (!KN_VSTREAM)
                [unroll] for (int k4 = 0; k4 < KT; ++k4)
                    wv[k4] = wg_ldB16(P_wqkv, whn + (uint)(((QKV - 1) * 2 * KT + ct * KT + k4) * 256));
            v8f av[G];
            [unroll] for (int g = 0; g < G; ++g) av[g] = wgk_zero8f();
            [unroll] for (int kt = 0; kt < KT; ++kt) {
                if (KN_VSTREAM) wv[kt] = wg_ldB16(P_wqkv, whn + (uint)(((QKV - 1) * 2 * KT + ct * KT + kt) * 256));
                [unroll] for (int g = 0; g < G; ++g) av[g] = wg_mma16(kn_nbsel(n[g][kt], nb[g][kt]), wv[kt], av[g]);
            }
            if (NB == 1 && MIX == KMIX_FIXED && ct == 0 && hd > 0) wg_barrier_lds();
            [unroll] for (int g = 0; g < G; ++g)
                kb_stl(kv + (uint)g * KVW + (uint)(KH + (4 * ct + (int)wave) * KB_FRAG),
                       wg_rows_to_gfx12(wgk_round8(av[g])));  // VT, gfx12 token order
        }
        wg_barrier_lds();

        const uint mrow = (uint)((hd * 64 + t) * 64);   // this lane's query row of the head's table (halfs)
        uint4 mx[4];      // softmax: the bias B[h][t][s] as the scores' accumulator start (gfx12 shape)
        wg_h16 mb[4];     // fixed: M[h][t][s] as z's B operand (16 keys per lane, straight from memory)
        if (MIX == KMIX_FIXED) {
            if (!KN_ZSTREAM)
                [unroll] for (int s0 = 0; s0 < 4; ++s0) mb[s0] = kn_ldB(pmix, mrow + (uint)(16 * s0));
        } else
            [unroll] for (int s0 = 0; s0 < 4; ++s0) mx[s0] = wgk_ld8(pmix, mrow + (uint)(16 * s0 + 8 * hh));
        wg_h16 zb[G][2];
        [unroll] for (int g = 0; g < G; ++g) {
            const uint lkb = kvb + (uint)g * KVW;   // the window's K tiles (tile bases, halfs)
            const uint lvb = lkb + (uint)KH;        // its VT tiles
            wg_h16 pb[4];
            if (MIX == KMIX_FIXED) {
                if (!KN_ZSTREAM)
                    [unroll] for (int s1 = 0; s1 < 4; ++s1) pb[s1] = mb[s1];
            } else {
                wg_h16 kf[8];
                if (!KN_ZSTREAM)
                    [unroll] for (int f0 = 0; f0 < 8; ++f0) kf[f0] = kn_ldlA(lkb + (uint)(f0 * KB_FRAG));
                uint4 gg[4];
                [unroll] for (int st = 0; st < 4; ++st) {
                    v8f sc = wgk_widen8(mx[st]);
                    [unroll] for (int ct = 0; ct < 2; ++ct) {
                        if (KN_ZSTREAM) kf[2 * st + ct] = kn_ldlA(lkb + (uint)((2 * st + ct) * KB_FRAG));
                        sc = wg_mma16(kf[2 * st + ct], qb[g][ct], sc);
                    }
                    const uint4 u = KB_CLAMP8W(wgk_round8(sc), WGK_SOFTMAX_C | 0x8000u, WGK_SOFTMAX_C);
                    const uint4 u2 = wgk_mul8(u, u);
                    const uint4 k2 = wgk_mul8(wgk_splat8h(WGK_SOFTMAX_KAPPA), u2);
                    const uint4 o = KB_SUB8S(wgk_splat8h(WGK_H_ONE), k2);
                    gg[st] = KB_MUL8S(u, o);
                }
                uint4 pa[4];
                wgk_softmax_weights(gg, pa);
                [unroll] for (int s2 = 0; s2 < 4; ++s2) pb[s2] = wg_b16(pa[s2]);
            }
            wg_h16 vf[8];
            if (!KB_GRPLATE && !KN_ZSTREAM)
                [unroll] for (int f1 = 0; f1 < 8; ++f1) vf[f1] = kn_ldlA(lvb + (uint)(f1 * KB_FRAG));
            [unroll] for (int ct = 0; ct < 2; ++ct) {
                if (KB_GRPLATE && !KN_ZSTREAM)
                    [unroll] for (int f2 = 0; f2 < 4; ++f2) vf[4 * ct + f2] = kn_ldlA(lvb + (uint)((4 * ct + f2) * KB_FRAG));
                v8f az = wgk_zero8f();
                [unroll] for (int st = 0; st < 4; ++st) {
                    if (KN_ZSTREAM) {
                        vf[4 * ct + st] = kn_ldlA(lvb + (uint)((4 * ct + st) * KB_FRAG));
                        if (MIX == KMIX_FIXED) pb[st] = kn_ldB(pmix, mrow + (uint)(16 * st));
                    }
                    az = wg_mma16(vf[4 * ct + st], pb[st], az);
                }
                zb[g][ct] = wg_b16(wgk_round8(az));   // z: only Wo's B operand
                if (KN_ZSTREAM)   // Wo for this window and channel tile now: each y tile still gets ct 0 then ct 1 per head
                    [unroll] for (int ot = 0; ot < KT; ++ot)
                        yacc[g][ot] = wg_mma16(wg_ldA16(P_wo, (uint)((ot * DT + 2 * hd + ct) * 256)), zb[g][ct], yacc[g][ot]);
            }
        }
        if (!KN_ZSTREAM) {
            wg_h16 wo[2][C / 16];
            if (!KB_GRPLATE)
                [unroll] for (int c1 = 0; c1 < 2; ++c1)
                    [unroll] for (int o2 = 0; o2 < KT; ++o2)
                        wo[c1][o2] = wg_ldA16(P_wo, (uint)((o2 * DT + 2 * hd + c1) * 256));
            [unroll] for (int ct = 0; ct < 2; ++ct) {
                if (KB_GRPLATE)
                    [unroll] for (int o2 = 0; o2 < KT; ++o2) wo[ct][o2] = wg_ldA16(P_wo, (uint)((o2 * DT + 2 * hd + ct) * 256));
                [unroll] for (int ot = 0; ot < KT; ++ot)
                    [unroll] for (int g = 0; g < G; ++g) yacc[g][ot] = wg_mma16(wo[ct][ot], zb[g][ct], yacc[g][ot]);
            }
        }
    }

    // 3. x1 = f16(x + y);  4. a = f16(x1 g2)
    uint4 x1[G][C / 16], a[G][C / 16];
    {
        uint4 g2v[C / 16], xin[G][C / 16];
#if KN_XPIN
        const uint px3 = P_x + wgk_pin(asuint(yacc[G - 1][KT - 1].v[7]));
#else
#define px3 P_x
#endif
        [unroll] for (int k5 = 0; k5 < KT; ++k5) g2v[k5] = wgk_ld8(P_g2, (uint)(16 * k5 + 8 * hh));
        [unroll] for (int kt = 0; kt < KT; ++kt)
            [unroll] for (int g = 0; g < G; ++g) {
                if (XG)
                    xin[g][kt] = wgk_ld8(P_y, ytok[g] * (uint)C + 8u * (uint)hh + (uint)(16 * kt));
                else if (XS)
                    xin[g][kt] = kb_ldl(xl[g] + (uint)(16 * kt));
                else
                    xin[g][kt] = wgk_ld8(px3, xt[g] + (uint)(16 * kt));
            }
#undef px3
        [unroll] for (int kt = 0; kt < KT; ++kt)
            [unroll] for (int g = 0; g < G; ++g) {
                const uint4 xv = XG ? (real[g] ? xin[g][kt] : wgk_zero8()) : xin[g][kt];
                x1[g][kt] = wgk_add8(xv, wgk_round8(yacc[g][kt]));
                a[g][kt] = KB_MUL8W(x1[g][kt], g2v[kt]);
            }
    }

    const bool X1L = KB_X1LDS && G == 2 && C >= 64 && XH > 0;
    if (X1L) {
        if (XA) wg_barrier_lds();
        [unroll] for (int g = 0; g < G; ++g)
            [unroll] for (int kt = 0; kt < KT; ++kt) kb_stl(xl[g] + (uint)(16 * kt), x1[g][kt]);
    }
    // KN_AB 2: a waits in this lane's x slots; the products read the token's 16 channels of a tile from both half-waves'
    // slots (the barriers: XA's K / VT reads by every wave are over, and the other half-wave's slots are written)
    const bool ABL = KN_AB == 2 && XH > 0 && !X1L;
    if (ABL) {
        if (XA) wg_barrier_lds();
        [unroll] for (int g = 0; g < G; ++g)
            [unroll] for (int kt = 0; kt < KT; ++kt) kb_stl(xl[g] + (uint)(16 * kt), a[g][kt]);
        wg_barrier_lds();
    }

    // 4. MLP: per hidden tile p = f16(W1 a + b1), h = act(p), acc += W2 h. a's B form is made once for all hidden tiles.
    v8f acc[G][C / 16];
    {
        uint4 b2v[C / 16];
        [unroll] for (int o3 = 0; o3 < KT; ++o3) b2v[o3] = wgk_ld8(P_b2, (uint)(16 * o3 + 8 * hh));
        [unroll] for (int ot = 0; ot < KT; ++ot) {
            const v8f b0 = wgk_widen8(b2v[ot]);
            [unroll] for (int g = 0; g < G; ++g) acc[g][ot] = b0;
        }
    }
    wg_h16 ab[G][C / 16];
    if (KN_AB == 0 || (KN_AB == 2 && !ABL))
        [unroll] for (int g = 0; g < G; ++g)
            [unroll] for (int kt = 0; kt < KT; ++kt) ab[g][kt] = wg_b16(a[g][kt]);
    [loop] for (int ht = 0; ht < HT; ++ht) {
        wg_h16 w1t[C / 16], w2t[C / 16];
        const uint4 b1v = wgk_ld8(P_b1, (uint)(16 * ht + 8 * hh));
        [unroll] for (int k6 = 0; k6 < KT; ++k6) w1t[k6] = wg_ldA16(P_w1, (uint)((ht * KT + k6) * 256));
        if (!KB_W2LATE)
            [unroll] for (int o4 = 0; o4 < KT; ++o4) w2t[o4] = wg_ldA16(P_w2, (uint)((ht * KT + o4) * 256));
        v8f p[G];
        const v8f b0 = wgk_widen8(b1v);
        [unroll] for (int g = 0; g < G; ++g) p[g] = b0;
        [unroll] for (int kt = 0; kt < KT; ++kt)
            [unroll] for (int g = 0; g < G; ++g) {
                wg_h16 abt = ab[g][kt];
                if (ABL) abt = kn_ldlB(xl[g] - 8u * (uint)hh + (uint)(16 * kt));
                else if (KN_AB == 1) abt = wg_b16(a[g][kt]);
                p[g] = wg_mma16(w1t[kt], abt, p[g]);
            }
        wg_h16 hb[G];
        [unroll] for (int g = 0; g < G; ++g) hb[g] = wg_b16(KB_ACT8W(wgk_round8(p[g])));
        if (KB_W2LATE)
            [unroll] for (int o4 = 0; o4 < KT; ++o4) w2t[o4] = wg_ldA16(P_w2, (uint)((ht * KT + o4) * 256));
        [unroll] for (int ot = 0; ot < KT; ++ot)
            [unroll] for (int g = 0; g < G; ++g) acc[g][ot] = wg_mma16(w2t[ot], hb[g], acc[g][ot]);
    }

    // 5. x2, stored for the real positions (the stores' base pinned to every x2 fragment: products before the branch)
    uint pin5 = 0u;
    if (X1L)
        [unroll] for (int g = 0; g < G; ++g)
            [unroll] for (int kt = 0; kt < KT; ++kt) x1[g][kt] = kb_ldl(xl[g] + (uint)(16 * kt));
    [unroll] for (int g = 0; g < G; ++g)
        [unroll] for (int ot = 0; ot < KT; ++ot) {
            x1[g][ot] = KB_ADD8W(x1[g][ot], wgk_round8(acc[g][ot]));
            pin5 ^= x1[g][ot].x;
        }
    const uint py = P_y + wgk_pin(pin5);
    [unroll] for (int g = 0; g < G; ++g) {
        if (real[g] && P_y != 0u) {
            [unroll] for (int o5 = 0; o5 < KT; ++o5)
                wgk_st8(py, ytok[g] * (uint)C + 8u * (uint)hh + (uint)(16 * o5), x1[g][o5]);
        }
    }

    if (MS) {  // 2x2 merge: A = Wm's row-mapped tiles, B = the group member's 16 channels from the x staging in LDS
        if ((XA && !X1L) || ABL) wg_barrier_lds();
        [unroll] for (int g = 0; g < G; ++g)
            [unroll] for (int ot = 0; ot < KT; ++ot) kb_stl(xl[g] + (uint)(16 * ot), real[g] ? x1[g][ot] : wgk_zero8());
        wg_barrier_lds();
        const int KTM = 4 * C / 16, NTM = CNEXT / 16;
        const int gr = n16 >> 2, gc = n16 & 3;
        const int mw = (P_w + 1) / 2, mh = (P_h + 1) / 2;
        [loop] for (int tt = (int)wave; tt < NTM; tt += 4) {
            v8f m[G];
            const v8f b0 = wgk_widen8(wgk_ld8(P_bz, (uint)(16 * tt + 8 * hh)));
            [unroll] for (int g = 0; g < G; ++g) m[g] = b0;
            const int MST = KN_MERGE_ROLL ? KN_MERGE_STEP : 4;
#if KN_MERGE_ROLL
            [loop] for (int k7 = 0; k7 < KTM + (int)wgk_zero(); k7 += MST) {
#else
            [unroll] for (int k7 = 0; k7 < KTM; k7 += 4) {
#endif
                wg_h16 wm[4], xm[G][4];
                [unroll] for (int i = 0; i < MST; ++i) {
                    const int kt = k7 + i;
                    const int s = (16 * kt) / C, c0 = 16 * kt - s * C;
                    const int ft = (2 * gr + (s >> 1)) * 8 + 2 * gc + (s & 1);
                    wm[i] = wg_ldA16(P_wz, (uint)((tt * KTM + kt) * 256));
                    [unroll] for (int g = 0; g < G; ++g) xm[g][i] = kn_ldlB(lds_x + (uint)(g * XH + ft * C + c0));
                }
                [unroll] for (int i2 = 0; i2 < MST; ++i2)
                    [unroll] for (int g = 0; g < G; ++g) m[g] = wg_mma16(wm[i2], xm[g][i2], m[g]);
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

    if (HEAD) {  // out = f16(Wh x2 + bh), 40 of 48 padded rows; x2's B form once for the three output tiles
        uint4 heads[G][3];
        wg_h16 whd[3][C / 16], xb[G][C / 16];
        uint4 bhd[3];
        [unroll] for (int o6 = 0; o6 < 3; ++o6) {
            bhd[o6] = wgk_ld8(P_bz, (uint)(16 * o6 + 8 * hh));
            [unroll] for (int k8 = 0; k8 < KT; ++k8) whd[o6][k8] = wg_ldA16(P_wz, (uint)((o6 * KT + k8) * 256));
        }
        [unroll] for (int g = 0; g < G; ++g)
            [unroll] for (int kt = 0; kt < KT; ++kt) xb[g][kt] = wg_b16(x1[g][kt]);
        [unroll] for (int ot = 0; ot < 3; ++ot) {
            v8f o[G];
            const v8f b0 = wgk_widen8(bhd[ot]);
            [unroll] for (int g = 0; g < G; ++g) o[g] = b0;
            [unroll] for (int kt = 0; kt < KT; ++kt)
                [unroll] for (int g = 0; g < G; ++g) o[g] = wg_mma16(whd[ot][kt], xb[g][kt], o[g]);
            [unroll] for (int g = 0; g < G; ++g) heads[g][ot] = wgk_round8(o[g]);
        }
        uint pinh = 0u;
        [unroll] for (int g = 0; g < G; ++g)
            [unroll] for (int o8 = 0; o8 < 3; ++o8) pinh ^= heads[g][o8].x;
        const uint phead = P_head + wgk_pin(pinh);
        if (!PIX) {
            [unroll] for (int g = 0; g < G; ++g)
                [unroll] for (int o7 = 0; o7 < 3; ++o7)
                    if (real[g] && 16 * o7 + 8 * hh < 40)
                        wgk_st8(phead, ytok[g] * 40u + (uint)(16 * o7 + 8 * hh), heads[g][o7]);
        } else {
            // compact pixel head: as k_block.hlsli (the accumulators are in the gfx12 layout)
            [unroll] for (int g = 0; g < G; ++g) {
                const uint4 h0 = heads[g][0], h1 = heads[g][1], h2 = heads[g][2];
                const uint a0 = hh ? wgk_el(h0, 2u) : wgk_el(h0, 0u), a1 = hh ? wgk_el(h0, 3u) : wgk_el(h0, 1u);
                const uint c0 = hh ? wgk_el(h1, 6u) : wgk_el(h1, 4u), c1 = hh ? wgk_el(h1, 7u) : wgk_el(h1, 5u);
                const uint snd = (hh ? wgk_el(h0, 1u) : wgk_el(h1, 3u)) | ((hh ? wgk_el(h1, 5u) : wgk_el(h2, 7u)) << 16);
                const uint rcv = wgk_xor16_u(snd);
                if (real[g]) {
                    const uint d = ((uint)(2 * gy[g]) * (uint)(2 * P_w) + (uint)(2 * gx[g] + hh)) * 4u;
                    wg_st64(phead, d * 2u, uint2(a0 | (a1 << 16), rcv & 0xffffu));
                    wg_st64(phead, (d + (uint)(2 * P_w) * 4u) * 2u, uint2(c0 | (c1 << 16), rcv >> 16));
                }
            }
        }
        if (HIDDEN) {   // as k_block.hlsli
            float hs[4];
            [unroll] for (int k = 0; k < 4; ++k) {
                float v[5];
                [unroll] for (int j = 0; j < 5; ++j) {
                    const int c = 10 * k + 3 + j;
                    const float own = hh == 0 ? wgk_elf(heads[0][c / 16], (uint)(c % 8)) : wgk_elf(heads[G - 1][c / 16], (uint)(c % 8));
                    const float snd = hh == 0 ? wgk_elf(heads[G - 1][c / 16], (uint)(c % 8)) : wgk_elf(heads[0][c / 16], (uint)(c % 8));
                    const float other = wgk_xor16(snd);
                    v[j] = ((c % 16) / 8) == hh ? own : other;
                }
                hs[k] = asfloat(wgk_opq(asuint(0.2f * ((((v[0] + v[1]) + v[2]) + v[3]) + v[4]))));
            }
            const bool rh = hh == 0 ? real[0] : real[G - 1];
            const uint tk = hh == 0 ? ytok[0] : ytok[G - 1];
            if (rh) wgk_store_hidden(hidden, tk * 4u, hs);
        }
    }
}
