// SPDX-License-Identifier: GPL-3.0-only
// Model L (issue #89): the width-32 Swin block in its FP16 form (clean/l/reference/oracle.py _block_tokens_q7,
// table consts form "q7", FP8 op model "exact"), for Model L's encoder block and decoder block.
//
// Arithmetic (equal to the oracle bit for bit):
// - one FP8 matrix op (up to 32 FP8 x FP8 products and an FP16 start, one FP16 result) is computed exactly: the
//   products (exact in float32) are summed in float32 in ascending k with the float32 sum of |products| alongside;
//   while that stays below 63 the sum is exact, otherwise a TwoSum chain gives it exactly as a head and an error sum;
//   the start is added with error-free sums and the head rounded to FP16, the tail's sign picking the side where the
//   head is exactly an FP16 midpoint (clean/l/runner/l_mm8_f32_check.py op_kernel, tests test_l_mm8_kernel);
// - Q7F-2 (S2a): the FP8 ops run on the FP8 matrix instruction, 16 rows x the window's 16 tokens per 32-wide chunk
//   (l_q7_op8.h q7_tile_sum), while the precomputed bound sum |w| x max |a| < 63 holds; an output whose bound fails
//   takes the exact scalar op8 above (rare; l_q7_s2a_fallbacks.py counts them on a recorded run);
// - every FP16 addition is exact the same way (an FP16 sum is not always exact in float32); every FP16 product is
//   exact in float32 and rounded once; the FMA16 steps round the exact sum once;
// - the norm's r = RN16(1 / sqrt(S')) and the attention weights' w = RN16(1 / sum) are table lookups by the FP16 bit
//   pattern (clean/l/runner/l_pack.py q7_tables, computed as the oracle computes them);
// - FP8 rounding: nearest even, saturating at +-448, NaN kept (wg_f32_to_e4m3).
// Weights: the q7 blob (clean/l/runner/l_pack.py pack_q7_block, natural order): FP8 rows (matrix instruction), FP16
// rows (fallback), sum |w| tables (bound).
//
// Geometry: one 4 x 4 window per workgroup of 32 lanes (one wave); lane = token t (0 ... 15, raster 4 row + col) +
// 16 x half. Window (wy, wx) covers grid rows 4 wy - oy ... and columns 4 wx - ox ...; grid = nwy x nwx workgroups,
// nwx = ceil((W + ox) / 4), nwy = ceil((H + oy) / 4). Only grid positions inside H x W are written.
//
// Build: clang -x hip ... -I clean/kernels clean/l/kernels/l_q7.hip -o l_q7.co
#pragma clang fp contract(off)
#include "wg_platform.h"
#include "l_q7_op8.h"

namespace q7 {

using namespace q7c;

constexpr int C = 32, HEADS = 2, MLP = 128, T = 16;
// q7 blob offsets (pack_q7_block for C 32, 2 heads, MLP 128)
constexpr int OFF_G1 = 0, OFF_G2 = 64, OFF_BP = 128, OFF_B1 = 192, OFF_B2 = 448, OFF_B = 512;
constexpr int OFF_WQK = 1536, OFF_WV = 3584, OFF_WP = 5632, OFF_W1 = 7680, OFF_W2 = 11776;
// the same matrices as FP16 (Q7F-1: exact copies, read without an FP8 decode)
constexpr int OFF_WQK16 = 15872, OFF_WV16 = 19968, OFF_WP16 = 24064, OFF_W1_16 = 28160, OFF_W2_16 = 36352;

// RMS norm of one token (FP16 values x[32]) -> r = RN16(1 / sqrt(S')) by the table: the even / odd lane tree.
WG_FN float norm_r(const float* x, const u16* rsqrt_tab) {
  float q[C];
  for (int c = 0; c < C; ++c) q[c] = mul16(x[c], x[c]);
  float hv[2];
  for (int odd = 0; odd < 2; ++odd) {
    float e[4];
    for (int t = 0; t < 4; ++t) {
      const int c0 = 2 * t + odd;   // tiles j = 0 .. 3: channels c0, c0 + 8, c0 + 16, c0 + 24
      e[t] = add16(add16(q[c0], q[c0 + 16]), add16(q[c0 + 8], q[c0 + 24]));
    }
    hv[odd] = add16(add16(e[0], e[1]), add16(e[2], e[3]));
  }
  const float s = add16(add16(hv[0], hv[1]), 0x1p-13f);
  return from16(rsqrt_tab[bits16(s)]);
}

// sum |w| tables of the FP8 matrices (Q7F-2, chunk-major [K / 32][rows] float32)
constexpr int WS_WQK = 44544, WS_WV = 44800, WS_WP = 45056, WS_W1 = 45312, WS_W2 = 45824;

// FP8 activations are kept as packed codes, token-major with the channels contiguous (a8, u, o8, m8, g8): 8
// consecutive codes of one token are one operand fragment of the FP8 matrix instruction (S2a), as the blob's
// row-major FP8 rows are for the weights. Beside each, the max code of the values each lane wrote (ma, mu, mo, mm, mg:
// [token][writer half]), so a token chunk's amax is the max of its writers' (the same 32 values the sums use); su:
// sum |u| of each head's query / key per writer half (exact: at most 32 FP8 magnitudes), the scores' wsum.
struct Lds {
  float x[T][C], v[HEADS][T][32], pw[HEADS][T][T], z[T][C], r[T];
  u8 a8[T][C], u[HEADS][T][32], o8[T][32 * HEADS], m8[T][C], g8[T][32];
  u32 ma[T][2], mu[HEADS][T][2], mo[T][2], mm[T][2], mg[T][2];
  float su[HEADS][T][2];
};

WG_FN void dec8(const u8* p, float* a) {          // 32 FP8 codes -> values
  for (int k = 0; k < 32; ++k) a[k] = fp8v(p[k]);
}
// The exact fallback of one output (rare: the bound failed): op8 on 32 FP8 activation codes and an FP16 weight row,
// or on two code rows (the scores).
WG_FN_NOINLINE float fb_w(float start, const u8* act, const _Float16* row) {
  float av[32];
  dec8(act, av);
  return op8(start, av, [&](int k) { return (float)row[k]; }, 32);
}
WG_FN_NOINLINE float fb_uu(float start, const u8* q, const u8* key) {
  float av[32], kv[32];
  dec8(q, av);
  dec8(key, kv);
  return op8(start, av, [&](int k) { return kv[k]; }, 32);
}
// Q7_HYB (Q11 part C evaluation, code object l_q7__hyb.co, runtime WG_L_FORM=hybrid; NOT the original): every FP8 chain
// of the C32 block accumulates in float32 (the products, exact in float32, added one at a time from the start, k and
// chunks ascending) and is rounded to FP16 once at its end: the single-chunk chains, the scores (32 terms from B), Wp
// across both heads and W2 across the four MLP slices. Everything else as the FP16 form. Oracle:
// clean/l/reference/oracle.py FP8 op model "f32chain" (as l_q7w.hip Q7_HYB). Q7_HYB_MUTANT (check only,
// l_q7__hybmut.co): W2 rounded to FP16 after every slice (the per-chunk rounding) - the check must catch it.
#ifndef Q7_HYB
#define Q7_HYB 0
#endif
#ifndef Q7_HYB_MUTANT
#define Q7_HYB_MUTANT 0
#endif
#if Q7_HYB
WG_FN float hyb_w(float acc, const u8* act, const _Float16* row) {      // float32, not rounded
  for (int k = 0; k < 32; ++k) acc = acc + fp8v(act[k]) * (float)row[k];
  return acc;
}
WG_FN float hyb_uu(float acc, const u8* q, const u8* key) {
  for (int k = 0; k < 32; ++k) acc = acc + fp8v(q[k]) * fp8v(key[k]);
  return acc;
}
#endif
// The lane's 16 outputs of a 32-row pair of tiles: output i is row 16 (i >> 3) + 8 hf + (i & 7).
WG_FN int orow(int i, u32 hf) { return 16 * (i >> 3) + 8 * (int)hf + (i & 7); }

// The block on one window. In: x (FP16 values) for the 16 members already in L.x. Out: the FP16 block output of
// member t, channel orow(i, hf) in out16[i].
WG_FN void block(Lds& L, const u8* w, const u16* tab, u32 t, u32 hf, float* out16) {
  const _Float16* wh = (const _Float16*)w;
  const u8* wb = w;
  // norm 1 -> FP8 a8
  if (hf == 0) L.r[t] = norm_r(L.x[t], tab);
  wg_barrier_lds();
  {
    u32 m = 0;
    for (int i = 0; i < 16; ++i) {
      const int c = 16 * hf + i;
      const float g = (float)wh[OFF_G1 / 2 + c];
      const u8 code = e8(mul16(L.x[t][c], mul16(L.r[t], g)));
      L.a8[t][c] = code;
      m = q7_maxcode(m, code);
    }
    L.ma[t][hf] = m;
  }
  wg_barrier_lds();
  // u = FP8(Wqk a8), v = Wv a8: 64 rows each = 4 tiles of 16 rows (tiles 2h, 2h + 1 = head h), one chunk
  {
    const float amx = q7_amax(L.ma[t][0], L.ma[t][1]);
    const u8* bt = &L.a8[t][8 * hf];
    u32 mq[HEADS] = {0, 0};
    float sq[HEADS] = {0.0f, 0.0f};
#pragma unroll
    for (int tile = 0; tile < 4; ++tile) {
      const int r0 = 16 * tile;
      const v8f S = q7_tile_sum(wb + OFF_WQK + (r0 + (int)t) * C + 8 * hf, bt);
      const float* ws = (const float*)(wb + WS_WQK) + r0 + 8 * hf;
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        const int r = r0 + 8 * (int)hf + j;
#if Q7_HYB
        const float y = h16(hyb_w(0.0f, L.a8[t], (const _Float16*)(wb + OFF_WQK16) + r * C));
        (void)S; (void)ws; (void)amx;
#else
        const float y = q7_combine(S[j], 0.0f, ws[j], amx, [&](float s0) {
          return fb_w(s0, L.a8[t], (const _Float16*)(wb + OFF_WQK16) + r * C);
        });
#endif
        const u8 code = e8(y);
        L.u[r >> 5][t][r & 31] = code;
        mq[tile >> 1] = q7_maxcode(mq[tile >> 1], code);
        sq[tile >> 1] = sq[tile >> 1] + __builtin_fabsf(fp8v(code));
      }
    }
#pragma unroll
    for (int tile = 0; tile < 4; ++tile) {
      const int r0 = 16 * tile;
      const v8f S = q7_tile_sum(wb + OFF_WV + (r0 + (int)t) * C + 8 * hf, bt);
      const float* ws = (const float*)(wb + WS_WV) + r0 + 8 * hf;
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        const int r = r0 + 8 * (int)hf + j;
#if Q7_HYB
        L.v[r >> 5][t][r & 31] = h16(hyb_w(0.0f, L.a8[t], (const _Float16*)(wb + OFF_WV16) + r * C));
#else
        L.v[r >> 5][t][r & 31] = q7_combine(S[j], 0.0f, ws[j], amx, [&](float s0) {
          return fb_w(s0, L.a8[t], (const _Float16*)(wb + OFF_WV16) + r * C);
        });
#endif
      }
    }
    for (int h = 0; h < HEADS; ++h) {
      L.mu[h][t][hf] = mq[h];
      L.su[h][t][hf] = sq[h];
    }
  }
  wg_barrier_lds();
  // scores s[h][t][j] = RN16(B + u_t . u_j): per head one tile, A = keys (row j), B = queries (token t), the same
  // fragments; the lane's outputs are query t, keys 8 hf + jj; wsum = sum |u_j|, amax = max |u_t|
#pragma unroll
  for (int h = 0; h < HEADS; ++h) {
    const u8* up = &L.u[h][t][8 * hf];
    const v8f S = q7_tile_sum(up, up);
    const float amx = q7_amax(L.mu[h][t][0], L.mu[h][t][1]);
#pragma unroll
    for (int jj = 0; jj < 8; ++jj) {
      const int j = 8 * (int)hf + jj;
      const float bij = (float)wh[OFF_B / 2 + (h * T + (int)t) * T + j];
#if Q7_HYB
      const float s = h16(hyb_uu(bij, L.u[h][t], L.u[h][j]));
#else
      const float s = q7_combine(S[jj], bij, L.su[h][j][0] + L.su[h][j][1], amx, [&](float s0) {
        return fb_uu(s0, L.u[h][t], L.u[h][j]);
      });
#endif
      const float cs = 0.017181396484375f, lim = 0.55615234375f;           // 0x2466, 0x3873
      const float uu = __builtin_fminf(__builtin_fmaxf(mul16(s, cs), -lim), lim);
      const float tt = fma16(uu, -uu, 0.92724609375f);                      // 0x3B6B
      const float p = fma16(uu, tt, 1.375f);                                // 0x3D80
      L.pw[h][t][j] = from16((bits16(p) & 0x3ffu) << 5);
    }
  }
  wg_barrier_lds();
  // the attention weights of head hf, query t (e in pw's row)
  {
    float e[T];
    for (int j = 0; j < T; ++j) e[j] = L.pw[hf][t][j];
    float a[2];
    for (int lo = 0; lo < 2; ++lo) {
      float pr[4];
      for (int k = 0; k < 4; ++k) pr[k] = add16(e[lo + 2 * k], e[lo + 8 + 2 * k]);
      a[lo] = add16(add16(pr[0], pr[1]), add16(pr[2], pr[3]));
    }
    const float wsum = from16(tab[65536 + bits16(add16(a[0], a[1]))]);
    for (int j = 0; j < T; ++j) L.pw[hf][t][j] = mul16(e[j], wsum);
  }
  wg_barrier_lds();
  // o = RN16(P v) of head hf (16 FP16 x FP16 products, summed exactly), FP8
  {
    u32 m = 0;
    for (int d = 0; d < 32; ++d) {
      float s = 0.0f, c = 0.0f;
      for (int j = 0; j < T; ++j) {
        float e;
        two_sum(s, L.pw[hf][t][j] * L.v[hf][j][d], s, e);
        c = c + e;
      }
      float z, tl;
      two_sum(s, c, z, tl);
      const u8 code = e8(rn16_pair(z, tl));
      L.o8[t][32 * hf + d] = code;
      m = q7_maxcode(m, code);
    }
    L.mo[t][hf] = m;                                   // chunk hf of o8 is this lane's alone
  }
  wg_barrier_lds();
  // z = Wp chain (head 0, then head 1) from RN16(bp + x): 32 rows = 2 tiles, K 64 = 2 chunks
  float acc[16];
  for (int i = 0; i < 16; ++i) {
    const int c = orow(i, hf);
    acc[i] = add16((float)wh[OFF_BP / 2 + c], L.x[t][c]);
  }
#pragma unroll
  for (int hd = 0; hd < HEADS; ++hd) {
    const float amx = q7_amax(L.mo[t][hd], 0u);
#pragma unroll
    for (int tile = 0; tile < 2; ++tile) {
      const int r0 = 16 * tile;
      const v8f S = q7_tile_sum(wb + OFF_WP + (r0 + (int)t) * 32 * HEADS + 32 * hd + 8 * hf, &L.o8[t][32 * hd + 8 * hf]);
      const float* ws = (const float*)(wb + WS_WP) + 32 * hd + r0 + 8 * hf;
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        const int i = 8 * tile + j, c = r0 + 8 * (int)hf + j;
#if Q7_HYB
        acc[i] = hyb_w(acc[i], L.o8[t] + 32 * hd, (const _Float16*)(wb + OFF_WP16) + c * 32 * HEADS + 32 * hd);
#else
        acc[i] = q7_combine(S[j], acc[i], ws[j], amx, [&](float s0) {
          return fb_w(s0, L.o8[t] + 32 * hd, (const _Float16*)(wb + OFF_WP16) + c * 32 * HEADS + 32 * hd);
        });
#endif
      }
    }
  }
#if Q7_HYB
  for (int i = 0; i < 16; ++i) acc[i] = h16(acc[i]);   // Wp's chain over both heads: one FP16 rounding
#endif
  for (int i = 0; i < 16; ++i) L.z[t][orow(i, hf)] = acc[i];
  wg_barrier_lds();
  // norm 2 -> FP8 m8
  if (hf == 0) L.r[t] = norm_r(L.z[t], tab);
  wg_barrier_lds();
  {
    u32 m = 0;
    for (int i = 0; i < 16; ++i) {
      const int c = 16 * hf + i;
      const u8 code = e8(mul16(L.z[t][c], mul16(L.r[t], (float)wh[OFF_G2 / 2 + c])));
      L.m8[t][c] = code;
      m = q7_maxcode(m, code);
    }
    L.mm[t][hf] = m;
  }
  for (int i = 0; i < 16; ++i) {
    const int c = orow(i, hf);
    acc[i] = add16((float)wh[OFF_B2 / 2 + c], L.z[t][c]);
  }
  wg_barrier_lds();
  // MLP: 4 hidden slices of 32, ascending: W1 slice (2 tiles, K 32) -> GELU -> FP8 g8, then W2's chunk (2 tiles)
  const float amx_m = q7_amax(L.mm[t][0], L.mm[t][1]);
  for (int sl = 0; sl < MLP / 32; ++sl) {
    u32 m = 0;
#pragma unroll
    for (int tile = 0; tile < 2; ++tile) {
      const int r0 = 32 * sl + 16 * tile;
      const v8f S = q7_tile_sum(wb + OFF_W1 + (r0 + (int)t) * C + 8 * hf, &L.m8[t][8 * hf]);
      const float* ws = (const float*)(wb + WS_W1) + r0 + 8 * hf;
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        const int jr = r0 + 8 * (int)hf + j;
#if Q7_HYB
        const float hval = h16(hyb_w((float)wh[OFF_B1 / 2 + jr], L.m8[t], (const _Float16*)(wb + OFF_W1_16) + jr * C));
#else
        const float hval = q7_combine(S[j], (float)wh[OFF_B1 / 2 + jr], ws[j], amx_m, [&](float s0) {
          return fb_w(s0, L.m8[t], (const _Float16*)(wb + OFF_W1_16) + jr * C);
        });
#endif
        const u8 code = e8(gelu16(hval));
        L.g8[t][jr - 32 * sl] = code;
        m = q7_maxcode(m, code);
      }
    }
    L.mg[t][hf] = m;
    wg_barrier_lds();
    const float amx_g = q7_amax(L.mg[t][0], L.mg[t][1]);
#pragma unroll
    for (int tile = 0; tile < 2; ++tile) {
      const int r0 = 16 * tile;
      const v8f S = q7_tile_sum(wb + OFF_W2 + (r0 + (int)t) * MLP + 32 * sl + 8 * hf, &L.g8[t][8 * hf]);
      const float* ws = (const float*)(wb + WS_W2) + 32 * sl + r0 + 8 * hf;
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        const int i = 8 * tile + j, c = r0 + 8 * (int)hf + j;
#if Q7_HYB
        acc[i] = hyb_w(acc[i], L.g8[t], (const _Float16*)(wb + OFF_W2_16) + c * MLP + 32 * sl);
#if Q7_HYB_MUTANT
        acc[i] = h16(acc[i]);
#endif
#else
        acc[i] = q7_combine(S[j], acc[i], ws[j], amx_g, [&](float s0) {
          return fb_w(s0, L.g8[t], (const _Float16*)(wb + OFF_W2_16) + c * MLP + 32 * sl);
        });
#endif
      }
    }
    wg_barrier_lds();
  }
#if Q7_HYB
  for (int i = 0; i < 16; ++i) acc[i] = h16(acc[i]);   // W2's chain over the four slices: one FP16 rounding
#endif
  for (int i = 0; i < 16; ++i) out16[i] = acc[i];
}

WG_FN void window_member(int H, int W, int ox, int oy, u32 t, int& pr, int& pc) {
  const int nwx = (W + ox + 3) / 4;
  const int win = (int)wg_block_id();
  const int wy = win / nwx, wx = win % nwx;
  pr = 4 * wy - oy + (int)(t >> 2);
  pc = 4 * wx - ox + (int)(t & 3u);
}

}  // namespace q7

// Encoder block: x = the FP16 halo map [H + 4][W + 4][32] (grid position (pr, pc) at map pixel (pr + 2, pc + 2)),
// skip unused (may be null), out = the FP8 map [H][W][32] (also the decoder block's skip). w = q7 blob, tab = tables.
WG_KERNEL_NW(l_q7_block_c32_enc, 32, 1)(const u8* x, const u8* skip, u8* out, const u8* w, const u16* tab, int H,
                                        int W, int ox, int oy) {
  (void)skip;
  WG_LDS q7::Lds L;
  const u32 lane = wg_lane_id(), t = lane & 15u, hf = lane >> 4;
  int pr, pc;
  q7::window_member(H, W, ox, oy, t, pr, pc);
  const _Float16* src = (const _Float16*)x + ((size_t)(pr + 2) * (W + 4) + (pc + 2)) * q7::C;
  for (int i = 0; i < 16; ++i) L.x[t][16 * hf + i] = (float)src[16 * hf + i];
  wg_barrier_lds();
  float o[16];
  q7::block(L, w, tab, t, hf, o);
  if (pr >= 0 && pr < H && pc >= 0 && pc < W) {
    u8* dst = out + ((size_t)pr * W + pc) * q7::C;
    for (int i = 0; i < 16; ++i) dst[q7::orow(i, hf)] = q7::out8(o[i]);
  }
}

// Decoder block: x = the FP8 projection map [H][W][32] (a member outside the grid takes the coarse mirror of its 2 x 2
// group, sub-pixel kept), skip = the encoder block's FP8 map [H][W][32] (a member outside the grid takes the mirror
// of its own position), the member's input RN16(P + S); out = the FP16 map [H][W][32].
WG_KERNEL_NW(l_q7_block_c32_dec, 32, 1)(const u8* x, const u8* skip, u8* out, const u8* w, const u16* tab, int H,
                                        int W, int ox, int oy) {
  WG_LDS q7::Lds L;
  const u32 lane = wg_lane_id(), t = lane & 15u, hf = lane >> 4;
  int pr, pc;
  q7::window_member(H, W, ox, oy, t, pr, pc);
  const int r = 2 * q7::mirror(pr >> 1, H >> 1) + (pr & 1), c = 2 * q7::mirror(pc >> 1, W >> 1) + (pc & 1);
  const int sr = q7::mirror(pr, H), sc = q7::mirror(pc, W);
  const u8* pp = x + ((size_t)r * W + c) * q7::C + 16 * hf;
  const u8* sp = skip + ((size_t)sr * W + sc) * q7::C + 16 * hf;
  for (int i = 0; i < 16; ++i) L.x[t][16 * hf + i] = q7::h16(q7::fp8v(pp[i]) + q7::fp8v(sp[i]));   // exact sum, one rounding
  wg_barrier_lds();
  float o[16];
  q7::block(L, w, tab, t, hf, o);
  if (pr >= 0 && pr < H && pc >= 0 && pc < W) {
    u16* dst = (u16*)out + ((size_t)pr * W + pc) * q7::C;
    for (int i = 0; i < 16; ++i) dst[q7::orow(i, hf)] = q7::out16(o[i]);
  }
}
