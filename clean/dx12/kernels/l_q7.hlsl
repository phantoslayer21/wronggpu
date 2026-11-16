// SPDX-License-Identifier: MIT
// D3D12 ports of Model L's FP16-form width-32 blocks (clean/l/kernels/l_q7.hip, l_q7.co; issue #89): the same
// arithmetic step for step, so the same bytes (clean/l/runner/l_q7_check.py compares both back ends and the oracle).
//
// Porting notes (clean/dx12/README.md "Arithmetic mode", "Driver exceptions"):
// - every FP16 rounding is wg_rn16 / wg_f2h (software, nearest even: HIP's (_Float16) x); no half arithmetic;
// - every product that is rounded or summed is exact in float32 (FP8 x FP8, FP16 x FP16), so a fusion of a product
//   into the following addition by the driver cannot change a result;
// - no FP32 denormal occurs: the smallest nonzero values are 2^-18 (FP8 products), 2^-48 (FP16 products) and the
//   errors of sums of multiples of those;
// - the window's rows live in groupshared memory and are read into local arrays with constant indices; the outer
//   loops' trip counts carry an opaque zero (wg_grid_ >> 31) so that the driver keeps them as loops;
// - no byte stores: the FP8 output is packed four to a dword and written as two 8-byte stores per lane;
// - Q7F-2 (S2a): the FP8 ops on the matrix instruction (l_q7_op8.hlsli, wave32), the bounded joins select-stored, a
//   failed output marked in a bitmask and recomputed by op8 in a loop after the phase's matrix calls.
//
// @wg-kernel l_q7_block_c32_enc co=l_q7.co block=32 wave=32 args=p,p,p,p,p,u,u,u,u
// @wg-kernel l_q7_block_c32_dec co=l_q7.co block=32 wave=32 args=p,p,p,p,p,u,u,u,u
#include "wg_platform.hlsli"

WG_ARGS_BEGIN
    uint x, skip, out_, w, tab;
    int H, W, ox, oy;
WG_ARGS_END

#define Q7_C 32
#define Q7_T 16
// q7 blob offsets (clean/l/runner/l_pack.py pack_q7_block for C 32, 2 heads, MLP 128), bytes
#define OFF_G1 0u
#define OFF_G2 64u
#define OFF_BP 128u
#define OFF_B1 192u
#define OFF_B2 448u
#define OFF_B 512u
#define OFF_WQK 1536u
#define OFF_WV 3584u
#define OFF_WP 5632u
#define OFF_W1 7680u
#define OFF_W2 11776u
// the same matrices as FP16 (Q7F-1: exact copies, read without an FP8 decode)
#define OFF_WQK16 15872u
#define OFF_WV16 19968u
#define OFF_WP16 24064u
#define OFF_W1_16 28160u
#define OFF_W2_16 36352u

// FP8 activations (a8, u, o8, m8, g8) are packed codes, 4 per dword (byte i = element i), token-major with the
// channels contiguous: 8 consecutive codes of a token (a uint2) are one operand fragment of the FP8 matrix instruction
// (wg_wmma_fp8), as the blob's row-major FP8 rows are for the weights (S2a: the hardware sums drop in
// without a format change).
groupshared float g_x[Q7_T][Q7_C];
groupshared uint g_a8[Q7_T][Q7_C / 4];
groupshared uint g_u[2][Q7_T][8];
groupshared float g_v[2][Q7_T][32];
groupshared float g_pw[2][Q7_T][Q7_T];
groupshared uint g_o8[Q7_T][16];
groupshared float g_z[Q7_T][Q7_C];     // the Wp chain, after norm 2 the MLP accumulators (each lane its own 16)
groupshared uint g_m8[Q7_T][Q7_C / 4];
groupshared uint g_g8[Q7_T][8];
groupshared float g_r[Q7_T];
// S2a (Q7F-2): max code of the values each lane wrote, [token][writer half] (a chunk's amax = the max of its writers'),
// and sum |u| per head / token / writer half (the scores' wsum)
groupshared uint g_ma[Q7_T][2];
groupshared uint g_mu[2][Q7_T][2];
groupshared uint g_mo[Q7_T][2];
groupshared uint g_mm[Q7_T][2];
groupshared uint g_mg[Q7_T][2];
groupshared float g_su[2][Q7_T][2];
// sum |w| tables (chunk-major [K / 32][rows] float32)
#define WS_WQK 44544u
#define WS_WV 44800u
#define WS_WP 45056u
#define WS_W1 45312u
#define WS_W2 45824u

uint q7_zero() { return wg_grid_ >> 31; }      // 0, not provable by the compiler

#include "l_q7_common.hlsli"
#include "l_q7_op8.hlsli"
float cd8(uint w4, uint j) { return fp8v(w4 >> (8u * j)); }   // element j of a packed dword
float ldh(uint off) { return from16(wg_ld16(w, off)); }   // an FP16 value of the blob
// 32 FP8 weights of a blob row (byte offset a multiple of 4)
void load_row8(uint off, out float wv[32])
{
    [unroll] for (uint i = 0; i < 8; ++i) {
        uint u = wg_ld32(w, off + 4u * i);
        [unroll] for (uint j = 0; j < 4; ++j) wv[4u * i + j] = fp8v(u >> (8u * j));
    }
}

// r = RN16(1 / sqrt(S')) of one token by the table: the even / odd lane tree (l_q7.hip norm_r).
float norm_r(float xv[32])
{
    float q[32];
    [unroll] for (uint c = 0; c < 32; ++c) q[c] = mul16(xv[c], xv[c]);
    float hv[2];
    [unroll] for (uint odd = 0; odd < 2; ++odd) {
        float e[4];
        [unroll] for (uint t = 0; t < 4; ++t) {
            uint c0 = 2u * t + odd;
            e[t] = add16(add16(q[c0], q[c0 + 16u]), add16(q[c0 + 8u], q[c0 + 24u]));
        }
        hv[odd] = add16(add16(e[0], e[1]), add16(e[2], e[3]));
    }
    float s = add16(add16(hv[0], hv[1]), asfloat(0x39000000u));   // + 2^-13
    return from16(wg_ld16(tab, 2u * bits16(s)));
}

// ---- S2a (Q7F-2) helpers: the matrix-instruction sums are joined branch-free (select stores); an output whose bound
// fails is marked in a bitmask and recomputed exactly by op8 in a separate loop after the phase's matrix calls (the
// driver wants the calls in wave-uniform, branch-free surroundings, wg_ags.hlsli)
float score_e(float s)                                   // the scores' exp polynomial (l_q7.hip)
{
    const float cs = 0.017181396484375f, lim = 0.55615234375f;   // 0x2466, 0x3873
    float uu = min(max(mul16(s, cs), -lim), lim);
    float tt = fma16(uu, -uu, 0.92724609375f);                    // 0x3B6B
    float p = fma16(uu, tt, 1.375f);                              // 0x3D80
    return from16((bits16(p) & 0x3ffu) << 5);
}
void ws8(uint off, out float ws[8])                      // 8 consecutive float32 of the blob (32-byte aligned)
{
    uint4 a = wg_ld128(w, off), b = wg_ld128(w, off + 16u);
    ws[0] = asfloat(a.x); ws[1] = asfloat(a.y); ws[2] = asfloat(a.z); ws[3] = asfloat(a.w);
    ws[4] = asfloat(b.x); ws[5] = asfloat(b.y); ws[6] = asfloat(b.z); ws[7] = asfloat(b.w);
}
uint pack4(uint c0, uint c1, uint c2, uint c3) { return c0 | (c1 << 8) | (c2 << 16) | (c3 << 24); }
uint setb(uint word, uint i, uint code) { return (word & ~(0xffu << (8u * i))) | ((code & 0xffu) << (8u * i)); }
uint max16(uint4 q) { return q7_maxcode4(q7_maxcode4(q7_maxcode4(q7_maxcode4(0u, q.x), q.y), q.z), q.w); }
float sum16(uint4 q)                                     // sum |code values| of 16 codes (exact)
{
    uint a[4] = { q.x, q.y, q.z, q.w };
    float s = 0.0f;
    [unroll] for (uint i = 0; i < 16; ++i) s = s + abs(cd8(a[i >> 2], i & 3u));
    return s;
}
// the lane's 16 outputs of a 32-row pair of tiles: output i is row 16 (i >> 3) + 8 hf + (i & 7)
uint orow(uint i, uint hf) { return 16u * (i >> 3) + 8u * hf + (i & 7u); }

// The block on one window (g_x already holds the 16 members). Out: the FP16 block output of member t, channel
// orow(i, hf) in g_z[t][orow(i, hf)].
void block(uint t, uint hf)
{
    const uint Z = q7_zero();
    float xa[32], wv[32];
    // norm 1 -> FP8 a8
    if (hf == 0u) {
        [unroll] for (uint c = 0; c < 32; ++c) xa[c] = g_x[t][c];
        g_r[t] = norm_r(xa);
    }
    wg_barrier_lds();
    {
        uint m = 0u;
        [unroll] for (uint i = 0; i < 4; ++i) {
            uint pk = 0u;
            [unroll] for (uint jj = 0; jj < 4; ++jj) {
                uint c = 16u * hf + 4u * i + jj;
                pk |= wg_f32_to_e4m3(mul16(g_x[t][c], mul16(g_r[t], ldh(OFF_G1 + 2u * c)))) << (8u * jj);
            }
            g_a8[t][4u * hf + i] = pk;
            m = q7_maxcode4(m, pk);
        }
        g_ma[t][hf] = m;
    }
    wg_barrier_lds();
    // u = FP8(Wqk a8), v = Wv a8: 64 rows each = 4 tiles (tiles 2h, 2h + 1 = head h), one chunk
    {
        float amx = q7_amax(g_ma[t][0], g_ma[t][1]);
        uint2 b0 = uint2(g_a8[t][2u * hf], g_a8[t][2u * hf + 1u]), b1 = uint2(g_a8[t][4u + 2u * hf], g_a8[t][5u + 2u * hf]);
        uint fq = 0u, fv = 0u;                           // fallback bits 8 tile + j
        [unroll] for (uint tile = 0; tile < 4; ++tile) {
            uint r0 = 16u * tile, ra = OFF_WQK + (r0 + t) * 32u + 8u * hf;
            v8f S = q7_tile_sum(wg_ld64(w, ra), b0, wg_ld64(w, ra + 16u), b1);
            float ws[8];
            ws8(WS_WQK + 4u * (r0 + 8u * hf), ws);
            uint cd[8];
            [unroll] for (uint j = 0; j < 8; ++j) {
                cd[j] = wg_f32_to_e4m3(q7_join(S.v[j], 0.0f));
                fq |= q7_bounded(ws[j], amx) ? 0u : (1u << (8u * tile + j));
            }
            uint dw = ((r0 & 31u) + 8u * hf) >> 2;
            g_u[tile >> 1][t][dw] = pack4(cd[0], cd[1], cd[2], cd[3]);
            g_u[tile >> 1][t][dw + 1u] = pack4(cd[4], cd[5], cd[6], cd[7]);
        }
        [unroll] for (uint tile2 = 0; tile2 < 4; ++tile2) {
            uint r0 = 16u * tile2, ra = OFF_WV + (r0 + t) * 32u + 8u * hf;
            v8f S = q7_tile_sum(wg_ld64(w, ra), b0, wg_ld64(w, ra + 16u), b1);
            float ws[8];
            ws8(WS_WV + 4u * (r0 + 8u * hf), ws);
            [unroll] for (uint j = 0; j < 8; ++j) {
                uint r = r0 + 8u * hf + j;
                g_v[r >> 5][t][r & 31u] = q7_join(S.v[j], 0.0f);
                fv |= q7_bounded(ws[j], amx) ? 0u : (1u << (8u * tile2 + j));
            }
        }
        if ((fq | fv) != 0u) {
            [unroll] for (uint c1 = 0; c1 < 32; ++c1) xa[c1] = cd8(g_a8[t][c1 >> 2], c1 & 3u);
            while (fq != 0u) {
                uint b = firstbitlow(fq);
                fq &= fq - 1u;
                uint r = 16u * (b >> 3) + 8u * hf + (b & 7u);
                q7_load16(w, OFF_WQK16 + 2u * r * 32u, wv);
                uint code = wg_f32_to_e4m3(op8(0.0f, xa, wv));
                g_u[r >> 5][t][(r & 31u) >> 2] = setb(g_u[r >> 5][t][(r & 31u) >> 2], r & 3u, code);
            }
            while (fv != 0u) {
                uint b = firstbitlow(fv);
                fv &= fv - 1u;
                uint r = 16u * (b >> 3) + 8u * hf + (b & 7u);
                q7_load16(w, OFF_WV16 + 2u * r * 32u, wv);
                g_v[r >> 5][t][r & 31u] = op8(0.0f, xa, wv);
            }
        }
        [unroll] for (uint h = 0; h < 2; ++h) {          // this lane's 16 final codes of head h
            uint4 q = uint4(g_u[h][t][2u * hf], g_u[h][t][2u * hf + 1u], g_u[h][t][4u + 2u * hf], g_u[h][t][5u + 2u * hf]);
            g_mu[h][t][hf] = max16(q);
            g_su[h][t][hf] = sum16(q);
        }
    }
    wg_barrier_lds();
    // scores s[h][t][j] = RN16(B + u_t . u_j): per head one tile, A = keys (row j), B = queries (token t), the same
    // fragments; the lane's outputs: query t, keys 8 hf + jj; wsum = sum |u_j|, amax = max |u_t|; e into g_pw
    {
        uint fs = 0u;
        [unroll] for (uint h = 0; h < 2; ++h) {
            uint2 u0 = uint2(g_u[h][t][2u * hf], g_u[h][t][2u * hf + 1u]);
            uint2 u1 = uint2(g_u[h][t][4u + 2u * hf], g_u[h][t][5u + 2u * hf]);
            v8f S = q7_tile_sum(u0, u0, u1, u1);
            float amx = q7_amax(g_mu[h][t][0], g_mu[h][t][1]);
            [unroll] for (uint jj = 0; jj < 8; ++jj) {
                uint j = 8u * hf + jj;
                float bij = ldh(OFF_B + 2u * ((h * 16u + t) * 16u + j));
                g_pw[h][t][j] = score_e(q7_join(S.v[jj], bij));
                fs |= q7_bounded(g_su[h][j][0] + g_su[h][j][1], amx) ? 0u : (1u << (8u * h + jj));
            }
        }
        while (fs != 0u) {
            uint b = firstbitlow(fs);
            fs &= fs - 1u;
            uint h = b >> 3, j = 8u * hf + (b & 7u);
            [unroll] for (uint k = 0; k < 32; ++k) {
                xa[k] = cd8(g_u[h][t][k >> 2], k & 3u);
                wv[k] = cd8(g_u[h][j][k >> 2], k & 3u);
            }
            g_pw[h][t][j] = score_e(op8(ldh(OFF_B + 2u * ((h * 16u + t) * 16u + j)), xa, wv));
        }
    }
    wg_barrier_lds();
    // the attention weights of head hf, query t (e in g_pw's own row)
    {
        float e[16];
        [unroll] for (uint j2 = 0; j2 < 16; ++j2) e[j2] = g_pw[hf][t][j2];
        float a[2];
        [unroll] for (uint lo = 0; lo < 2; ++lo) {
            float pr[4];
            [unroll] for (uint k3 = 0; k3 < 4; ++k3) pr[k3] = add16(e[lo + 2u * k3], e[lo + 8u + 2u * k3]);
            a[lo] = add16(add16(pr[0], pr[1]), add16(pr[2], pr[3]));
        }
        float wsum = from16(wg_ld16(tab, 2u * (65536u + bits16(add16(a[0], a[1])))));
        [unroll] for (uint j3 = 0; j3 < 16; ++j3) g_pw[hf][t][j3] = mul16(e[j3], wsum);
    }
    wg_barrier_lds();
    // o = RN16(P v) of head hf (16 FP16 x FP16 products, summed exactly), FP8
    float pw[16];
    [unroll] for (uint j4 = 0; j4 < 16; ++j4) pw[j4] = g_pw[hf][t][j4];
    uint mo = 0u;
    [loop] for (uint q4 = 0; q4 < 8u + Z; ++q4) {
        uint pk = 0u;
        [unroll] for (uint jj = 0; jj < 4; ++jj) {
            uint d2 = 4u * q4 + jj;
            float s = 0.0f, c = 0.0f;
            [unroll] for (uint j5 = 0; j5 < 16; ++j5) {
                float e;
                two_sum(s, pw[j5] * g_v[hf][j5][d2], s, e);
                c = c + e;
            }
            float z, tl;
            two_sum(s, c, z, tl);
            pk |= wg_f32_to_e4m3(rn16_pair(z, tl)) << (8u * jj);
        }
        g_o8[t][8u * hf + q4] = pk;
        mo = q7_maxcode4(mo, pk);
    }
    g_mo[t][hf] = mo;                                    // chunk hf of o8 is this lane's alone
    wg_barrier_lds();
    // z = Wp chain (head 0, then head 1) from RN16(bp + x): 32 rows = 2 tiles, K 64 = 2 chunks; acc in g_z (the
    // bounded joins are select-stored, a failed output keeps its start for the fallback)
    [unroll] for (uint i2 = 0; i2 < 16; ++i2) {
        uint c = orow(i2, hf);
        g_z[t][c] = add16(ldh(OFF_BP + 2u * c), g_x[t][c]);
    }
    [unroll] for (uint hd = 0; hd < 2; ++hd) {
        float amx = q7_amax(g_mo[t][hd], 0u);
        uint2 o0 = uint2(g_o8[t][8u * hd + 2u * hf], g_o8[t][8u * hd + 2u * hf + 1u]);
        uint2 o1 = uint2(g_o8[t][8u * hd + 4u + 2u * hf], g_o8[t][8u * hd + 5u + 2u * hf]);
        uint fp = 0u;
        [unroll] for (uint tile = 0; tile < 2; ++tile) {
            uint r0 = 16u * tile, ra = OFF_WP + (r0 + t) * 64u + 32u * hd + 8u * hf;
            v8f S = q7_tile_sum(wg_ld64(w, ra), o0, wg_ld64(w, ra + 16u), o1);
            float ws[8];
            ws8(WS_WP + 4u * (32u * hd + r0 + 8u * hf), ws);
            [unroll] for (uint j = 0; j < 8; ++j) {
                uint c = r0 + 8u * hf + j;
                float a0 = g_z[t][c];
                bool bd = q7_bounded(ws[j], amx);
                g_z[t][c] = bd ? q7_join(S.v[j], a0) : a0;
                fp |= bd ? 0u : (1u << (8u * tile + j));
            }
        }
        if (fp != 0u) {
            [unroll] for (uint k4 = 0; k4 < 32; ++k4) xa[k4] = cd8(g_o8[t][8u * hd + (k4 >> 2)], k4 & 3u);
            while (fp != 0u) {
                uint b = firstbitlow(fp);
                fp &= fp - 1u;
                uint c = 16u * (b >> 3) + 8u * hf + (b & 7u);
                q7_load16(w, OFF_WP16 + 2u * (c * 64u + 32u * hd), wv);
                g_z[t][c] = op8(g_z[t][c], xa, wv);
            }
        }
    }
    wg_barrier_lds();
    // norm 2 -> FP8 m8; then the MLP accumulators RN16(b2 + z) in place of z (each lane its own channels)
    if (hf == 0u) {
        [unroll] for (uint c3 = 0; c3 < 32; ++c3) xa[c3] = g_z[t][c3];
        g_r[t] = norm_r(xa);
    }
    wg_barrier_lds();
    {
        uint m = 0u;
        [unroll] for (uint i3 = 0; i3 < 4; ++i3) {
            uint pk = 0u;
            [unroll] for (uint jj = 0; jj < 4; ++jj) {
                uint c = 16u * hf + 4u * i3 + jj;
                pk |= wg_f32_to_e4m3(mul16(g_z[t][c], mul16(g_r[t], ldh(OFF_G2 + 2u * c)))) << (8u * jj);
            }
            g_m8[t][4u * hf + i3] = pk;
            m = q7_maxcode4(m, pk);
        }
        g_mm[t][hf] = m;
    }
    wg_barrier_lds();                                    // every lane has read z for its m8 before acc overwrites it
    [unroll] for (uint i6 = 0; i6 < 16; ++i6) {
        uint c = orow(i6, hf);
        g_z[t][c] = add16(ldh(OFF_B2 + 2u * c), g_z[t][c]);
    }
    // MLP: 4 hidden slices of 32, ascending: W1 slice (2 tiles, K 32) -> GELU -> FP8 g8, then W2's chunk (2 tiles)
    float amx_m = q7_amax(g_mm[t][0], g_mm[t][1]);
    uint2 m0 = uint2(g_m8[t][2u * hf], g_m8[t][2u * hf + 1u]), m1 = uint2(g_m8[t][4u + 2u * hf], g_m8[t][5u + 2u * hf]);
    [loop] for (uint sl = 0; sl < 4u + Z; ++sl) {
        uint f1 = 0u;
        [unroll] for (uint tile = 0; tile < 2; ++tile) {
            uint r0 = 32u * sl + 16u * tile, ra = OFF_W1 + (r0 + t) * 32u + 8u * hf;
            v8f S = q7_tile_sum(wg_ld64(w, ra), m0, wg_ld64(w, ra + 16u), m1);
            float ws[8];
            ws8(WS_W1 + 4u * (r0 + 8u * hf), ws);
            uint cd[8];
            [unroll] for (uint j = 0; j < 8; ++j) {
                uint jr = r0 + 8u * hf + j;
                cd[j] = wg_f32_to_e4m3(gelu16(q7_join(S.v[j], ldh(OFF_B1 + 2u * jr))));
                f1 |= q7_bounded(ws[j], amx_m) ? 0u : (1u << (8u * tile + j));
            }
            uint dw = (16u * tile + 8u * hf) >> 2;
            g_g8[t][dw] = pack4(cd[0], cd[1], cd[2], cd[3]);
            g_g8[t][dw + 1u] = pack4(cd[4], cd[5], cd[6], cd[7]);
        }
        if (f1 != 0u) {
            [unroll] for (uint c4 = 0; c4 < 32; ++c4) xa[c4] = cd8(g_m8[t][c4 >> 2], c4 & 3u);
            while (f1 != 0u) {
                uint b = firstbitlow(f1);
                f1 &= f1 - 1u;
                uint jl = 16u * (b >> 3) + 8u * hf + (b & 7u), jr = 32u * sl + jl;
                q7_load16(w, OFF_W1_16 + 2u * jr * 32u, wv);
                uint code = wg_f32_to_e4m3(gelu16(op8(ldh(OFF_B1 + 2u * jr), xa, wv)));
                g_g8[t][jl >> 2] = setb(g_g8[t][jl >> 2], jl & 3u, code);
            }
        }
        g_mg[t][hf] = max16(uint4(g_g8[t][2u * hf], g_g8[t][2u * hf + 1u], g_g8[t][4u + 2u * hf], g_g8[t][5u + 2u * hf]));
        wg_barrier_lds();
        float amx_g = q7_amax(g_mg[t][0], g_mg[t][1]);
        uint2 q0 = uint2(g_g8[t][2u * hf], g_g8[t][2u * hf + 1u]), q1 = uint2(g_g8[t][4u + 2u * hf], g_g8[t][5u + 2u * hf]);
        uint f2 = 0u;
        [unroll] for (uint tile2 = 0; tile2 < 2; ++tile2) {
            uint r0 = 16u * tile2, ra = OFF_W2 + (r0 + t) * 128u + 32u * sl + 8u * hf;
            v8f S = q7_tile_sum(wg_ld64(w, ra), q0, wg_ld64(w, ra + 16u), q1);
            float ws[8];
            ws8(WS_W2 + 4u * (32u * sl + r0 + 8u * hf), ws);
            [unroll] for (uint j = 0; j < 8; ++j) {
                uint c = r0 + 8u * hf + j;
                float a0 = g_z[t][c];
                bool bd = q7_bounded(ws[j], amx_g);
                g_z[t][c] = bd ? q7_join(S.v[j], a0) : a0;
                f2 |= bd ? 0u : (1u << (8u * tile2 + j));
            }
        }
        if (f2 != 0u) {
            float gv[32];
            [unroll] for (uint k5 = 0; k5 < 32; ++k5) gv[k5] = cd8(g_g8[t][k5 >> 2], k5 & 3u);
            while (f2 != 0u) {
                uint b = firstbitlow(f2);
                f2 &= f2 - 1u;
                uint c = 16u * (b >> 3) + 8u * hf + (b & 7u);
                q7_load16(w, OFF_W2_16 + 2u * (c * 128u + 32u * sl), wv);
                g_z[t][c] = op8(g_z[t][c], gv, wv);
            }
        }
        wg_barrier_lds();
    }
}

void window_member(uint t, out int pr, out int pc)
{
    int nwx = (W + ox + 3) / 4;
    int win = (int)wg_block_id();
    int wy = win / nwx, wx = win % nwx;
    pr = 4 * wy - oy + (int)(t >> 2);
    pc = 4 * wx - ox + (int)(t & 3u);
}

// Encoder block: x = the FP16 halo map [H + 4][W + 4][32], out = the FP8 map [H][W][32] (l_q7.hip).
WG_KERNEL(l_q7_block_c32_enc)
{
    WG_SETUP();
    uint lane = wg_lane_id(), t = lane & 15u, hf = lane >> 4;
    int pr, pc;
    window_member(t, pr, pc);
    uint src = ((uint)(pr + 2) * (uint)(W + 4) + (uint)(pc + 2)) * 64u + 32u * hf;
    [unroll] for (uint i = 0; i < 8; ++i) {
        uint v = wg_ld32(x, src + 4u * i);
        g_x[t][16u * hf + 2u * i] = wg_h2f_lo(v);
        g_x[t][16u * hf + 2u * i + 1u] = wg_h2f_hi(v);
    }
    wg_barrier_lds();
    block(t, hf);
    if (pr >= 0 && pr < H && pc >= 0 && pc < W) {
        uint ob[4];                                      // channels 8 hf ... + 7, then 16 + 8 hf ... + 7 (orow)
        [unroll] for (uint q = 0; q < 4; ++q) {
            uint c = 16u * (q >> 1) + 8u * hf + 4u * (q & 1u);
            ob[q] = out8(g_z[t][c]) | (out8(g_z[t][c + 1u]) << 8) | (out8(g_z[t][c + 2u]) << 16) |
                    (out8(g_z[t][c + 3u]) << 24);
        }
        uint ob0 = ((uint)pr * (uint)W + (uint)pc) * 32u + 8u * hf;
        wg_st64(out_, ob0, uint2(ob[0], ob[1]));
        wg_st64(out_, ob0 + 16u, uint2(ob[2], ob[3]));
    }
}

// Decoder block: x = the FP8 projection map [H][W][32] (coarse mirror, sub-pixel kept), skip = the encoder block's FP8
// map [H][W][32] (fine mirror), the member's input RN16(P + S); out = the FP16 map [H][W][32] (l_q7.hip).
WG_KERNEL(l_q7_block_c32_dec)
{
    WG_SETUP();
    uint lane = wg_lane_id(), t = lane & 15u, hf = lane >> 4;
    int pr, pc;
    window_member(t, pr, pc);
    int r = 2 * q7_mirror(pr >> 1, H >> 1) + (pr & 1), c = 2 * q7_mirror(pc >> 1, W >> 1) + (pc & 1);
    int sr = q7_mirror(pr, H), sc = q7_mirror(pc, W);
    uint4 pp = wg_ld128(x, ((uint)r * (uint)W + (uint)c) * 32u + 16u * hf);
    uint4 sp = wg_ld128(skip, ((uint)sr * (uint)W + (uint)sc) * 32u + 16u * hf);
    uint pa[4] = { pp.x, pp.y, pp.z, pp.w };
    uint sa[4] = { sp.x, sp.y, sp.z, sp.w };
    [unroll] for (uint i = 0; i < 16; ++i)
        g_x[t][16u * hf + i] = h16(fp8v(pa[i >> 2] >> (8u * (i & 3u))) + fp8v(sa[i >> 2] >> (8u * (i & 3u))));
    wg_barrier_lds();
    block(t, hf);
    if (pr >= 0 && pr < H && pc >= 0 && pc < W) {
        uint ob[8];                                      // channels 8 hf ... + 7, then 16 + 8 hf ... + 7 (orow)
        [unroll] for (uint q = 0; q < 8; ++q) {
            uint c = 16u * (q >> 2) + 8u * hf + 2u * (q & 3u);
            ob[q] = out16(g_z[t][c]) | (out16(g_z[t][c + 1u]) << 16);
        }
        uint o = (((uint)pr * (uint)W + (uint)pc) * 32u + 8u * hf) * 2u;
        wg_st128(out_, o, uint4(ob[0], ob[1], ob[2], ob[3]));
        wg_st128(out_, o + 32u, uint4(ob[4], ob[5], ob[6], ob[7]));
    }
}
