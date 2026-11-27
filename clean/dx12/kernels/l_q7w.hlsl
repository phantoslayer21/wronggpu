// SPDX-License-Identifier: GPL-3.0-only
// D3D12 ports of Model L's shared-stage kernels in the FP16 form (clean/l/kernels/l_q7w.hip, l_q7w.co; issue #89):
// the same arithmetic step for step (l_q7_common.hlsli; porting notes in l_q7.hlsl), so the same bytes
// (clean/l/runner/l_q7w_check.py --d3d12). One source, one build per kernel: Q7_KIND 0 = block (FP8 map in, mirror
// border), 1 = decoder block (E coarse mirror + S fine mirror), 2 = merge, 3 = expansion. Groupshared FP16 values are
// 16-bit words and FP8 values packed codes (4 per dword), so that the width-128 block fits in 32 KB.
//
// @wg-kernel l_q7_block_c64 co=l_q7w.co block=32 wave=32 args=p,p,p,p,p,u,u,u,u defines=Q7_KIND=0,Q7_C=64,Q7_HEADS=2,Q7_ENTRY=l_q7_block_c64
// @wg-kernel l_q7_block_c96 co=l_q7w.co block=32 wave=32 args=p,p,p,p,p,u,u,u,u defines=Q7_KIND=0,Q7_C=96,Q7_HEADS=4,Q7_ENTRY=l_q7_block_c96
// @wg-kernel l_q7_block_c128 co=l_q7w.co block=32 wave=32 args=p,p,p,p,p,u,u,u,u defines=Q7_KIND=0,Q7_C=128,Q7_HEADS=4,Q7_ENTRY=l_q7_block_c128
// @wg-kernel l_q7_block_c64_dec co=l_q7w.co block=32 wave=32 args=p,p,p,p,p,u,u,u,u defines=Q7_KIND=1,Q7_C=64,Q7_HEADS=2,Q7_ENTRY=l_q7_block_c64_dec
// @wg-kernel l_q7_block_c96_dec co=l_q7w.co block=32 wave=32 args=p,p,p,p,p,u,u,u,u defines=Q7_KIND=1,Q7_C=96,Q7_HEADS=4,Q7_ENTRY=l_q7_block_c96_dec
// @wg-kernel l_q7_merge_c64 co=l_q7w.co block=32 wave=32 args=p,p,p,u,u defines=Q7_KIND=2,Q7_C=64,Q7_COUT=96,Q7_ENTRY=l_q7_merge_c64
// @wg-kernel l_q7_merge_c96 co=l_q7w.co block=32 wave=32 args=p,p,p,u,u defines=Q7_KIND=2,Q7_C=96,Q7_COUT=128,Q7_ENTRY=l_q7_merge_c96
// @wg-kernel l_q7_expand_c128 co=l_q7w.co block=32 wave=32 args=p,p,p,u,u defines=Q7_KIND=3,Q7_CL=128,Q7_C=96,Q7_ENTRY=l_q7_expand_c128
// @wg-kernel l_q7_expand_c96 co=l_q7w.co block=32 wave=32 args=p,p,p,u,u defines=Q7_KIND=3,Q7_CL=96,Q7_C=64,Q7_ENTRY=l_q7_expand_c96
#include "wg_platform.hlsli"

#if Q7_KIND <= 1
WG_ARGS_BEGIN
    uint x, skip, out_, w, tab;
    int H, W, ox, oy;
WG_ARGS_END
#else
WG_ARGS_BEGIN
    uint in_, out_, w;
    int H, W;
WG_ARGS_END
#endif

#include "l_q7_common.hlsli"
#include "l_q7_op8.hlsli"

uint q7_zero() { return wg_grid_ >> 31; }
float ldh(uint off) { return from16(wg_ld16(w, off)); }

#if Q7_KIND <= 1
// ---- block --------------------------------------------------------------------------------------------------------
#define C Q7_C
#define HEADS Q7_HEADS
#define MLP (4 * C)
#define CH (C / 2)
#define HPL (HEADS / 2)
#define NC (C / 32)
// blob offsets (l_pack.py q7_block_offsets)
#define O_G1 0u
#define O_G2 (2u * C)
#define O_BP (4u * C)
#define O_B1 (6u * C)
#define O_B2 (6u * C + 2u * MLP)
#define O_B (8u * C + 2u * MLP)
#define O_WQK (O_B + 2u * HEADS * 256u)
#define O_WV (O_WQK + HEADS * 32u * C)
#define O_WP (O_WV + HEADS * 32u * C)
#define O_W1 (O_WP + C * 32u * HEADS)
#define O_W2 (O_W1 + MLP * C)
// the same matrices as FP16 (Q7F-1)
#define O_END8 (O_W2 + C * MLP)
#define O_WQK16 O_END8
#define O_WV16 (O_WQK16 + 2u * HEADS * 32u * C)
#define O_WP16 (O_WV16 + 2u * HEADS * 32u * C)
#define O_W1_16 (O_WP16 + 2u * C * 32u * HEADS)
#define O_W2_16 (O_W1_16 + 2u * MLP * C)

groupshared uint16_t g_x[16][C];
groupshared uint16_t g_z[16][C];
groupshared uint16_t g_v[HEADS][16][32];
groupshared uint16_t g_pw[HEADS][16][16];
// FP8 activations: packed codes, 4 per dword (byte i = element i), token-major with the channels contiguous, so 8
// consecutive codes of a token are one FP8 matrix-instruction operand fragment (S2a; l_q7.hlsl)
groupshared uint g_a8[16][C / 4];
groupshared uint g_u[HEADS][16][8];
groupshared uint g_o8[16][8 * HEADS];
groupshared uint g_g8[16][8];
groupshared float g_r[16];

float cd8(uint w4, uint j) { return fp8v(w4 >> (8u * j)); }
float gx(uint t, uint c) { return from16(g_x[t][c]); }
float gz(uint t, uint c) { return from16(g_z[t][c]); }

// the Q8 B3 tree, one half (odd) of lane tt's tile channels 8 j + 2 tt + odd
float lane_half(float q[C], uint tt, uint odd)
{
    float g[2];
    [unroll] for (uint grp = 0; grp < 2; ++grp) {
        float pr[4];
        [unroll] for (uint i = 0; i < C / 32; ++i)
            pr[i] = add16(q[8u * (4u * i + grp) + 2u * tt + odd], q[8u * (4u * i + 2u + grp) + 2u * tt + odd]);
#if C == 32
        g[grp] = pr[0];
#elif C == 64
        g[grp] = add16(pr[0], pr[1]);
#elif C == 96
        g[grp] = add16(add16(pr[0], pr[1]), pr[2]);
#else
        g[grp] = add16(add16(pr[0], pr[1]), add16(pr[2], pr[3]));
#endif
    }
    return add16(g[0], g[1]);
}

float norm_r(float xv[C])
{
    float q[C];
    [unroll] for (uint c = 0; c < C; ++c) q[c] = mul16(xv[c], xv[c]);
    float hv[2];
    [unroll] for (uint odd = 0; odd < 2; ++odd) {
        float e[4];
        [unroll] for (uint tt = 0; tt < 4; ++tt) e[tt] = lane_half(q, tt, odd);
        hv[odd] = add16(add16(e[0], e[1]), add16(e[2], e[3]));
    }
    float s = add16(add16(hv[0], hv[1]), asfloat(0x39000000u));
    return from16(wg_ld16(tab, 2u * bits16(s)));
}

// chain over n chunks of 32: a = FP8 codes in a groupshared row (passed as values), row = FP8 codes in the blob
float chain_a8(float start, uint t, uint row, uint n)
{
    float a[32], wv[32];
    [loop] for (uint ch = 0; ch < n + q7_zero(); ++ch) {
        [unroll] for (uint k = 0; k < 32; ++k) a[k] = cd8(g_a8[t][8u * ch + (k >> 2)], k & 3u);
        q7_load16(w, row + 64u * ch, wv);
        start = op8(start, a, wv);
    }
    return start;
}

void block(uint t, uint hf)
{
    const uint Z = q7_zero();
    float wv[32], a[32];
    if (hf == 0u) {
        float xv[C];
        [unroll] for (uint c = 0; c < C; ++c) xv[c] = gx(t, c);
        g_r[t] = norm_r(xv);
    }
    wg_barrier_lds();
    [loop] for (uint i = 0; i < CH / 4u + Z; ++i) {
        uint pk = 0u;
        [unroll] for (uint jj = 0; jj < 4; ++jj) {
            uint c = CH * hf + 4u * i + jj;
            pk |= wg_f32_to_e4m3(mul16(gx(t, c), mul16(g_r[t], ldh(O_G1 + 2u * c)))) << (8u * jj);
        }
        g_a8[t][CH / 4u * hf + i] = pk;
    }
    wg_barrier_lds();
    [loop] for (uint hh = 0; hh < HPL + Z; ++hh) {
        uint hd = HPL * hf + hh;
        [loop] for (uint d4 = 0; d4 < 8u + Z; ++d4) {
            uint pk = 0u;
            [unroll] for (uint jj = 0; jj < 4; ++jj) {
                uint d = 4u * d4 + jj;
                pk |= wg_f32_to_e4m3(chain_a8(0.0f, t, O_WQK16 + 2u * (hd * 32u + d) * C, NC)) << (8u * jj);
                g_v[hd][t][d] = (uint16_t)bits16(chain_a8(0.0f, t, O_WV16 + 2u * (hd * 32u + d) * C, NC));
            }
            g_u[hd][t][d4] = pk;
        }
    }
    wg_barrier_lds();
    [loop] for (uint hh2 = 0; hh2 < HPL + Z; ++hh2) {
        uint hd = HPL * hf + hh2;
        [unroll] for (uint k = 0; k < 32; ++k) a[k] = cd8(g_u[hd][t][k >> 2], k & 3u);
        [loop] for (uint j = 0; j < 16u + Z; ++j) {
            [unroll] for (uint k2 = 0; k2 < 32; ++k2) wv[k2] = cd8(g_u[hd][j][k2 >> 2], k2 & 3u);
            float s = op8(ldh(O_B + 2u * ((hd * 16u + t) * 16u + j)), a, wv);
            const float cs = 0.017181396484375f, lim = 0.55615234375f;
            float uu = min(max(mul16(s, cs), -lim), lim);
            float tt = fma16(uu, -uu, 0.92724609375f);
            float p = fma16(uu, tt, 1.375f);
            g_pw[hd][t][j] = (uint16_t)((bits16(p) & 0x3ffu) << 5);
        }
        float e[16];
        [unroll] for (uint j2 = 0; j2 < 16; ++j2) e[j2] = from16(g_pw[hd][t][j2]);
        float sm[2];
        [unroll] for (uint lo = 0; lo < 2; ++lo) {
            float pr[4];
            [unroll] for (uint k3 = 0; k3 < 4; ++k3) pr[k3] = add16(e[lo + 2u * k3], e[lo + 8u + 2u * k3]);
            sm[lo] = add16(add16(pr[0], pr[1]), add16(pr[2], pr[3]));
        }
        float wsum = from16(wg_ld16(tab, 2u * (65536u + bits16(add16(sm[0], sm[1])))));
        [unroll] for (uint j3 = 0; j3 < 16; ++j3) g_pw[hd][t][j3] = (uint16_t)bits16(mul16(e[j3], wsum));
    }
    wg_barrier_lds();
    [loop] for (uint hh3 = 0; hh3 < HPL + Z; ++hh3) {
        uint hd = HPL * hf + hh3;
        float pw[16];
        [unroll] for (uint j4 = 0; j4 < 16; ++j4) pw[j4] = from16(g_pw[hd][t][j4]);
        [loop] for (uint q4 = 0; q4 < 8u + Z; ++q4) {
            uint pk = 0u;
            [unroll] for (uint jj = 0; jj < 4; ++jj) {
                uint d2 = 4u * q4 + jj;
                float s = 0.0f, c = 0.0f;
                [unroll] for (uint j5 = 0; j5 < 16; ++j5) {
                    float e;
                    two_sum(s, pw[j5] * from16(g_v[hd][j5][d2]), s, e);
                    c = c + e;
                }
                float z, tl;
                two_sum(s, c, z, tl);
                pk |= wg_f32_to_e4m3(rn16_pair(z, tl)) << (8u * jj);
            }
            g_o8[t][8u * hd + q4] = pk;
        }
    }
    wg_barrier_lds();
    [loop] for (uint i2 = 0; i2 < CH + Z; ++i2) {
        uint c = CH * hf + i2;
        float acc = add16(ldh(O_BP + 2u * c), gx(t, c));
        [loop] for (uint hd2 = 0; hd2 < HEADS + Z; ++hd2) {
            [unroll] for (uint k4 = 0; k4 < 32; ++k4) a[k4] = cd8(g_o8[t][8u * hd2 + (k4 >> 2)], k4 & 3u);
            q7_load16(w, O_WP16 + 2u * (c * 32u * HEADS + 32u * hd2), wv);
            acc = op8(acc, a, wv);
        }
        g_z[t][c] = (uint16_t)bits16(acc);
    }
    wg_barrier_lds();
    if (hf == 0u) {
        float zv[C];
        [unroll] for (uint c2 = 0; c2 < C; ++c2) zv[c2] = gz(t, c2);
        g_r[t] = norm_r(zv);
    }
    wg_barrier_lds();
    [loop] for (uint i3 = 0; i3 < CH / 4u + Z; ++i3) {
        uint pk = 0u;
        [unroll] for (uint jj = 0; jj < 4; ++jj) {
            uint c = CH * hf + 4u * i3 + jj;
            float zc = gz(t, c);
            pk |= wg_f32_to_e4m3(mul16(zc, mul16(g_r[t], ldh(O_G2 + 2u * c)))) << (8u * jj);
            g_z[t][c] = (uint16_t)bits16(add16(ldh(O_B2 + 2u * c), zc));
        }
        g_a8[t][CH / 4u * hf + i3] = pk;
    }
    wg_barrier_lds();
    [loop] for (uint sl = 0; sl < MLP / 32u + Z; ++sl) {
        [loop] for (uint i4 = 0; i4 < 4u + Z; ++i4) {
            uint pk = 0u;
            [unroll] for (uint jj = 0; jj < 4; ++jj) {
                uint j6 = 32u * sl + 16u * hf + 4u * i4 + jj;
                pk |= wg_f32_to_e4m3(gelu16(chain_a8(ldh(O_B1 + 2u * j6), t, O_W1_16 + 2u * j6 * C, NC))) << (8u * jj);
            }
            g_g8[t][4u * hf + i4] = pk;
        }
        wg_barrier_lds();
        [unroll] for (uint k5 = 0; k5 < 32; ++k5) a[k5] = cd8(g_g8[t][k5 >> 2], k5 & 3u);
        [loop] for (uint i5 = 0; i5 < CH + Z; ++i5) {
            uint c = CH * hf + i5;
            q7_load16(w, O_W2_16 + 2u * (c * MLP + 32u * sl), wv);
            g_z[t][c] = (uint16_t)bits16(op8(gz(t, c), a, wv));
        }
        wg_barrier_lds();
    }
}

WG_KERNEL(Q7_ENTRY)
{
    WG_SETUP();
    uint lane = wg_lane_id(), t = lane & 15u, hf = lane >> 4;
    int nwx = (W + ox + 3) / 4;
    int win = (int)wg_block_id();
    int pr = 4 * (win / nwx) - oy + (int)(t >> 2), pc = 4 * (win % nwx) - ox + (int)(t & 3u);
#if Q7_KIND == 1
    int r = 2 * q7_mirror(pr >> 1, H >> 1) + (pr & 1), c = 2 * q7_mirror(pc >> 1, W >> 1) + (pc & 1);
    int sr = q7_mirror(pr, H), sc = q7_mirror(pc, W);
    uint eo = ((uint)r * (uint)W + (uint)c) * C + CH * hf, so = ((uint)sr * (uint)W + (uint)sc) * C + CH * hf;
    [loop] for (uint i = 0; i < CH / 4u + q7_zero(); ++i) {
        uint ev = wg_ld32(x, eo + 4u * i), sv = wg_ld32(skip, so + 4u * i);
        [unroll] for (uint j = 0; j < 4; ++j)
            g_x[t][CH * hf + 4u * i + j] = (uint16_t)bits16(fp8v(ev >> (8u * j)) + fp8v(sv >> (8u * j)));
    }
#else
    uint xo = ((uint)q7_mirror(pr, H) * (uint)W + (uint)q7_mirror(pc, W)) * C + CH * hf;
    [loop] for (uint i = 0; i < CH / 4u + q7_zero(); ++i) {
        uint xv = wg_ld32(x, xo + 4u * i);
        [unroll] for (uint j = 0; j < 4; ++j) g_x[t][CH * hf + 4u * i + j] = (uint16_t)bits16(fp8v(xv >> (8u * j)));
    }
#endif
    wg_barrier_lds();
    block(t, hf);
    if (pr >= 0 && pr < H && pc >= 0 && pc < W) {
        uint oo = ((uint)pr * (uint)W + (uint)pc) * C + CH * hf;
        [loop] for (uint q = 0; q < CH / 4u + q7_zero(); ++q) {
            uint cb = CH * hf + 4u * q;
            wg_st32(out_, oo + 4u * q, out8(gz(t, cb)) | (out8(gz(t, cb + 1u)) << 8) | (out8(gz(t, cb + 2u)) << 16) |
                                       (out8(gz(t, cb + 3u)) << 24));
        }
    }
}

#elif Q7_KIND == 2
// ---- merge: in [H][W][C] -> out [H/2][W/2][COUT] ---------------------------------------------------------------------
WG_KERNEL(Q7_ENTRY)
{
    WG_SETUP();
    const uint C = Q7_C, COUT = Q7_COUT;
    uint lane = wg_lane_id();
    uint w2 = (uint)W / 2u, px = wg_block_id(), r = px / w2, c = px % w2;
    if (4u * lane < COUT) {
        uint rows = (2u * COUT + 15u) & ~15u;
        float a[32], wv[32];
        uint ob = 0u;
        [loop] for (uint j = 0; j < 4u + q7_zero(); ++j) {
            uint n = 4u * lane + j;
            float acc = ldh(2u * n);
            [loop] for (uint ch = 0; ch < 4u * C / 32u + q7_zero(); ++ch) {
                uint s = (32u * ch) / C, i0 = (32u * ch) % C;      // a chunk never crosses a sub-pixel (C % 32 == 0)
                q7_load8(in_, ((2u * r + (s >> 1)) * (uint)W + (2u * c + (s & 1u))) * C + i0, a);
                q7_load16(w, rows + COUT * 4u * C + 2u * (n * 4u * C + 32u * ch), wv);
                acc = op8(acc, a, wv);
            }
            if (!isnan(acc)) acc = min(max(acc, -6.28125f), 6.28125f);
            ob |= out8(acc) << (8u * j);
        }
        wg_st32(out_, (r * w2 + c) * COUT + 4u * lane, ob);
    }
}

#else
// ---- expansion: in [h][w][CL] -> out [2h][2w][C] ----------------------------------------------------------------------
WG_KERNEL(Q7_ENTRY)
{
    WG_SETUP();
    const uint C = Q7_C, CL = Q7_CL;
    uint lane = wg_lane_id();
    uint wd = (uint)W, px = wg_block_id(), r = px / wd, c = px % wd;
    uint rows = (2u * 4u * C + 15u) & ~15u;
    float a[32], wv[32];
    [loop] for (uint g = 0; g < C / 32u + q7_zero(); ++g) {
        uint n0 = 128u * g + 4u * lane, s = n0 / C, chn = n0 % C;
        uint ob = 0u;
        [loop] for (uint j = 0; j < 4u + q7_zero(); ++j) {
            uint n = n0 + j;
            float acc = ldh(2u * n);
            [loop] for (uint ch = 0; ch < CL / 32u + q7_zero(); ++ch) {
                q7_load8(in_, (r * wd + c) * CL + 32u * ch, a);
                q7_load16(w, rows + 4u * C * CL + 2u * (n * CL + 32u * ch), wv);
                acc = op8(acc, a, wv);
            }
            ob |= out8(acc) << (8u * j);
        }
        wg_st32(out_, ((2u * r + (s >> 1)) * (2u * wd) + (2u * c + (s & 1u))) * C + chn, ob);
    }
}
#endif
