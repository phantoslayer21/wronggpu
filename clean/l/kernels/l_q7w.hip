// SPDX-License-Identifier: MIT
// Model L (issue #89): the shared stages (enc1, enc2, b0-b5, dec2, dec1) in their FP16 form, as Model L's own kernel
// variants (Model M's kernels are not changed): clean/l/reference/oracle.py run_stage_q7 (table shared_form "q7").
// The arithmetic of every step is l_q7.hip's (exact FP8 ops, exact FP16 additions, table rsqrt / rcp), here for the
// widths C = 64, 96, 128 (heads C / 32 rounded to 2 or 4 as Model M's stages: 2, 4, 4) and MLP 4 C.
//
// Kernels (one 4 x 4 window, or one pixel, per workgroup of 32 lanes):
//   l_q7_block_c<C>       x = FP8 map [H][W][C] (the block's input is its FP8 values), border mirror; out = FP8 map
//   l_q7_block_c<C>_dec   x = the expansion's FP8 fine map E [H][W][C] (outside the grid: the coarse mirror, sub-pixel
//                         kept; table shared_dec_border "coarse"), skip = the encoder stage's FP8 map S [H][W][C] (the
//                         fine mirror); the member's input RN16(E + S); out = FP8 map
//   l_q7_merge_c<C>       in = FP8 map [H][W][C] (H, W even), out = FP8 [H/2][W/2][C_out]: the chain over the 2 x 2
//                         patch (sub-pixel s = dx + 2 dy, input channel C s + i) from the bias, clamped to +-6.28125
//                         (a NaN kept), FP8. Grid = (H/2) (W/2) workgroups.
//   l_q7_expand_c<Cl>     in = FP8 [h][w][Cl], out = the FP8 fine map [2h][2w][C] (row C s + c of the expansion goes to
//                         sub-pixel s, channel c): the chain over Cl from the bias, FP8. Grid = h w workgroups.
// Block weights: the q7 blob (l_pack.py pack_q7_block); merge / expansion weights: the q7 linear blob (l_pack.py
// pack_q7_linear: FP16 bias [n], then the FP8 rows [n][K] at q7_linear_offsets).
//
// Build: clang -x hip ... -I clean/kernels clean/l/kernels/l_q7w.hip -o l_q7w.co
#pragma clang fp contract(off)
#include "wg_platform.h"
#include "l_q7_op8.h"

namespace q7w {

using namespace q7c;

constexpr int T = 16;

WG_FN void row16(const u8* p, float* w) {          // 32 FP16 weights -> values (Q7F-1: no FP8 decode)
  const _Float16* h = (const _Float16*)p;
  for (int k = 0; k < 32; ++k) w[k] = (float)h[k];
}

WG_FN void row8(const u8* p, float* w) {           // 32 FP8 codes -> values
  for (int k = 0; k < 32; ++k) w[k] = fp8v(p[k]);
}

// Q7_HYB (Model M's HYBRID evaluation form, Q11 part C, runtime WG_M_FORM=hybrid; code object l_q7w__hyb.co; NOT
// the original): every FP8 chain accumulates in float32 over all its chunks (the products, exact in float32, added one
// at a time, chunks and k ascending, from the start widened exactly), ONE FP16 rounding at the end of the chain; the
// scores likewise (32 terms from B); W2's chain runs over all MLP slices in float32 (Lds::zf). Everything else as the
// FP16 form. Oracle: clean/l/reference/oracle.py FP8 op model "f32chain". Q7_HYB_MUTANT (check only, l_q7w__hybmut.co):
// the Wqk chains keep an FP16 rounding after every chunk (the original's per-chunk rounding) - the check must catch it.
#ifndef Q7_HYB
#define Q7_HYB 0
#endif
#ifndef Q7_HYB_MUTANT
#define Q7_HYB_MUTANT 0
#endif

// float32 chain over K = 32 n terms, not rounded (Q7_HYB); per_chunk_rn16: the mutant's FP16 rounding per chunk
WG_FN float chainf(float acc, const u8* a8, const u8* row, int n, bool per_chunk_rn16 = false) {
  float a[32], w[32];
  for (int ch = 0; ch < n; ++ch) {
    row8(a8 + 32 * ch, a);
    row16(row + 64 * ch, w);
    for (int k = 0; k < 32; ++k) acc = acc + a[k] * w[k];   // contraction off: the product (exact) then one rounding
    if (per_chunk_rn16) acc = h16(acc);
  }
  return acc;
}

// The chain over K = 32 n terms (ascending chunks): a8 = FP8 codes, row = FP8 codes, both contiguous
WG_FN float chain8(float start, const u8* a8, const u8* row, int n) {
#if Q7_HYB
  return h16(chainf(start, a8, row, n));
#endif
  float a[32], w[32];
  for (int ch = 0; ch < n; ++ch) {
    row8(a8 + 32 * ch, a);
    row16(row + 64 * ch, w);
    start = op8(start, a, w);
  }
  return start;
}

// Blob offsets (l_pack.py q7_block_offsets)
template <int C, int HEADS>
struct Off {
  static constexpr int MLP = 4 * C;
  static constexpr int G1 = 0, G2 = 2 * C, BP = 4 * C, B1 = 6 * C, B2 = 6 * C + 2 * MLP, B = 8 * C + 2 * MLP;
  static constexpr int WQK = B + 2 * HEADS * 256, WV = WQK + HEADS * 32 * C, WP = WV + HEADS * 32 * C,
                       W1 = WP + C * 32 * HEADS, W2 = W1 + MLP * C;
  static constexpr int END8 = W2 + C * MLP;            // FP16 copies (Q7F-1), the same order
  static constexpr int WQK16 = END8, WV16 = WQK16 + 2 * HEADS * 32 * C, WP16 = WV16 + 2 * HEADS * 32 * C,
                       W1_16 = WP16 + 2 * C * 32 * HEADS, W2_16 = W1_16 + 2 * MLP * C;
};

// norm tree of Q8 B3 for width C: one half (odd 0 / 1) of lane t's tile channels 8 j + 2 t + odd
template <int C>
WG_FN float lane_half(const float* q, int t, int odd) {
  constexpr int J = C / 8, NP = J / 4;              // pairs per group: 1 (C 32), 2, 3, 4 (C 128)
  float g[2];
  for (int grp = 0; grp < 2; ++grp) {               // grp 0: even tiles, grp 1: odd tiles
    float pr[4];
    for (int i = 0; i < NP; ++i) {
      const int j0 = 4 * i + grp, j1 = 4 * i + 2 + grp;
      pr[i] = add16(q[8 * j0 + 2 * t + odd], q[8 * j1 + 2 * t + odd]);
    }
    float s;
    if (NP == 1) s = pr[0];
    else if (NP == 2) s = add16(pr[0], pr[1]);
    else if (NP == 3) s = add16(add16(pr[0], pr[1]), pr[2]);
    else s = add16(add16(pr[0], pr[1]), add16(pr[2], pr[3]));
    g[grp] = s;
  }
  return add16(g[0], g[1]);
}

template <int C>
WG_FN float norm_r(const _Float16* x, const u16* tab) {
  float q[C];
  for (int c = 0; c < C; ++c) q[c] = mul16((float)x[c], (float)x[c]);
  float hv[2];
  for (int odd = 0; odd < 2; ++odd) {
    float e[4];
    for (int t = 0; t < 4; ++t) e[t] = lane_half<C>(q, t, odd);
    hv[odd] = add16(add16(e[0], e[1]), add16(e[2], e[3]));
  }
  const float s = add16(add16(hv[0], hv[1]), 0x1p-13f);
  return from16(tab[bits16(s)]);
}

template <int C, int HEADS>
struct Lds {
  _Float16 x[T][C], z[T][C], v[HEADS][T][32], pw[HEADS][T][T];
  u8 a8[T][C], u[HEADS][T][32], o8[T][32 * HEADS], g8[T][32];
#if Q7_HYB
  float zf[T][C];                                    // W2's float32 chain across the MLP slices
#endif
  float r[T];
};

// The block on one window (L.x holds the 16 members' FP16 inputs). Out: lane (t, hf)'s channels C/2 hf ... of member
// t, FP16, in L.z[t][...].
template <int C, int HEADS>
WG_FN void block(Lds<C, HEADS>& L, const u8* w, const u16* tab, int t, int hf) {
  using O = Off<C, HEADS>;
  constexpr int CH = C / 2, HPL = HEADS / 2, NC = C / 32;
  const _Float16* wh = (const _Float16*)w;
  if (hf == 0) L.r[t] = norm_r<C>(L.x[t], tab);
  wg_barrier_lds();
  for (int i = 0; i < CH; ++i) {
    const int c = CH * hf + i;
    L.a8[t][c] = (u8)e8(mul16((float)L.x[t][c], mul16(L.r[t], (float)wh[O::G1 / 2 + c])));
  }
  wg_barrier_lds();
  for (int hh = 0; hh < HPL; ++hh) {
    const int hd = HPL * hf + hh;
    for (int d = 0; d < 32; ++d) {
#if Q7_HYB && Q7_HYB_MUTANT
      L.u[hd][t][d] = (u8)e8(h16(chainf(0.0f, L.a8[t], w + O::WQK16 + 2 * (hd * 32 + d) * C, NC, true)));
#else
      L.u[hd][t][d] = (u8)e8(chain8(0.0f, L.a8[t], w + O::WQK16 + 2 * (hd * 32 + d) * C, NC));
#endif
      L.v[hd][t][d] = (_Float16)chain8(0.0f, L.a8[t], w + O::WV16 + 2 * (hd * 32 + d) * C, NC);
    }
  }
  wg_barrier_lds();
  for (int hh = 0; hh < HPL; ++hh) {
    const int hd = HPL * hf + hh;
    float e[T], a[32], b[32];
    row8(L.u[hd][t], a);
    for (int j = 0; j < T; ++j) {
      row8(L.u[hd][j], b);
#if Q7_HYB
      float s = (float)wh[O::B / 2 + (hd * T + t) * T + j];
      for (int k = 0; k < 32; ++k) s = s + a[k] * b[k];
      s = h16(s);
#else
      const float s = op8((float)wh[O::B / 2 + (hd * T + t) * T + j], a, b);
#endif
      const float cs = 0.017181396484375f, lim = 0.55615234375f;
      const float uu = __builtin_fminf(__builtin_fmaxf(mul16(s, cs), -lim), lim);
      const float tt = fma16(uu, -uu, 0.92724609375f);
      const float p = fma16(uu, tt, 1.375f);
      e[j] = from16((bits16(p) & 0x3ffu) << 5);
    }
    float sm[2];
    for (int lo = 0; lo < 2; ++lo) {
      float pr[4];
      for (int k = 0; k < 4; ++k) pr[k] = add16(e[lo + 2 * k], e[lo + 8 + 2 * k]);
      sm[lo] = add16(add16(pr[0], pr[1]), add16(pr[2], pr[3]));
    }
    const float wsum = from16(tab[65536 + bits16(add16(sm[0], sm[1]))]);
    for (int j = 0; j < T; ++j) L.pw[hd][t][j] = (_Float16)mul16(e[j], wsum);
  }
  wg_barrier_lds();
  for (int hh = 0; hh < HPL; ++hh) {
    const int hd = HPL * hf + hh;
    for (int d = 0; d < 32; ++d) {
      float s = 0.0f, c = 0.0f;
      for (int j = 0; j < T; ++j) {
        float e;
        two_sum(s, (float)L.pw[hd][t][j] * (float)L.v[hd][j][d], s, e);
        c = c + e;
      }
      float z, tl;
      two_sum(s, c, z, tl);
      L.o8[t][32 * hd + d] = (u8)e8(rn16_pair(z, tl));
    }
  }
  wg_barrier_lds();
  for (int i = 0; i < CH; ++i) {
    const int c = CH * hf + i;
    float acc = add16((float)wh[O::BP / 2 + c], (float)L.x[t][c]);
    acc = chain8(acc, L.o8[t], w + O::WP16 + 2 * c * 32 * HEADS, HEADS);
    L.z[t][c] = (_Float16)acc;
  }
  wg_barrier_lds();
  if (hf == 0) L.r[t] = norm_r<C>(L.z[t], tab);
  wg_barrier_lds();
  for (int i = 0; i < CH; ++i) {                     // m8 in a8's place; the MLP accumulators in z's
    const int c = CH * hf + i;
    const float zc = (float)L.z[t][c];
    L.a8[t][c] = (u8)e8(mul16(zc, mul16(L.r[t], (float)wh[O::G2 / 2 + c])));
    L.z[t][c] = (_Float16)add16((float)wh[O::B2 / 2 + c], zc);
#if Q7_HYB
    L.zf[t][c] = (float)L.z[t][c];
#endif
  }
  wg_barrier_lds();
  for (int sl = 0; sl < O::MLP / 32; ++sl) {
    for (int i = 0; i < 16; ++i) {
      const int j = 32 * sl + 16 * hf + i;
      const float hv = chain8((float)wh[O::B1 / 2 + j], L.a8[t], w + O::W1_16 + 2 * j * C, NC);
      L.g8[t][16 * hf + i] = (u8)e8(gelu16(hv));
    }
    wg_barrier_lds();
    for (int i = 0; i < CH; ++i) {
      const int c = CH * hf + i;
#if Q7_HYB
      L.zf[t][c] = chainf(L.zf[t][c], L.g8[t], w + O::W2_16 + 2 * (c * O::MLP + 32 * sl), 1);
      if (sl == O::MLP / 32 - 1) L.z[t][c] = (_Float16)L.zf[t][c];   // the chain's one FP16 rounding
#else
      L.z[t][c] = (_Float16)chain8((float)L.z[t][c], L.g8[t], w + O::W2_16 + 2 * (c * O::MLP + 32 * sl), 1);
#endif
    }
    wg_barrier_lds();
  }
}

WG_FN void window_member(int W, int ox, int oy, int t, int& pr, int& pc) {
  const int nwx = (W + ox + 3) / 4;
  const int win = (int)wg_block_id();
  const int wy = win / nwx, wx = win % nwx;
  pr = 4 * wy - oy + (t >> 2);
  pc = 4 * wx - ox + (t & 3);
}

template <int C, int HEADS, bool DEC>
WG_FN void block_kernel(const u8* x, const u8* skip, u8* out, const u8* w, const u16* tab, int H, int W, int ox,
                        int oy) {
  WG_LDS Lds<C, HEADS> L;
  constexpr int CH = C / 2;
  const int lane = (int)wg_lane_id(), t = lane & 15, hf = lane >> 4;
  int pr, pc;
  window_member(W, ox, oy, t, pr, pc);
  if (DEC) {
    const int r = 2 * mirror(pr >> 1, H >> 1) + (pr & 1), c = 2 * mirror(pc >> 1, W >> 1) + (pc & 1);
    const int sr = mirror(pr, H), sc = mirror(pc, W);
    const u8* ep = x + ((size_t)r * W + c) * C + CH * hf;
    const u8* sp = skip + ((size_t)sr * W + sc) * C + CH * hf;
    for (int i = 0; i < CH; ++i) L.x[t][CH * hf + i] = (_Float16)h16(fp8v(ep[i]) + fp8v(sp[i]));
  } else {
    const u8* xp = x + ((size_t)mirror(pr, H) * W + mirror(pc, W)) * C + CH * hf;
    for (int i = 0; i < CH; ++i) L.x[t][CH * hf + i] = (_Float16)fp8v(xp[i]);
  }
  wg_barrier_lds();
  block<C, HEADS>(L, w, tab, t, hf);
  if (pr >= 0 && pr < H && pc >= 0 && pc < W) {
    u8* dst = out + ((size_t)pr * W + pc) * C + CH * hf;
    for (int i = 0; i < CH; ++i) dst[i] = out8((float)L.z[t][CH * hf + i]);
  }
}

// Linear blob offsets (l_pack.py q7_linear_offsets): bias FP16 [n] at 0, rows FP8 [n][K] at align16(2 n)
WG_FN int lin_rows(int n) { return (2 * n + 15) & ~15; }

template <int C, int COUT>
WG_FN void merge_kernel(const u8* in, u8* out, const u8* w, int H, int W) {
  const int lane = (int)wg_lane_id();
  const int px = (int)wg_block_id(), w2 = W / 2, r = px / w2, c = px % w2;
  if (4 * lane >= COUT) return;
  const _Float16* bias = (const _Float16*)w;
  const u8* rows16 = w + lin_rows(COUT) + (size_t)COUT * 4 * C;   // the FP16 rows (Q7F-1)
  u8 z[4 * C];
  for (int s = 0; s < 4; ++s) {
    const u8* src = in + ((size_t)(2 * r + (s >> 1)) * W + (2 * c + (s & 1))) * C;
    for (int i = 0; i < C; ++i) z[C * s + i] = src[i];
  }
  u8 o[4];
  for (int j = 0; j < 4; ++j) {
    const int n = 4 * lane + j;
    float acc = chain8((float)bias[n], z, rows16 + (size_t)2 * n * 4 * C, 4 * C / 32);
    if (!__builtin_isnan(acc)) acc = __builtin_fminf(__builtin_fmaxf(acc, -6.28125f), 6.28125f);
    o[j] = out8(acc);
  }
  u8* dst = out + ((size_t)r * w2 + c) * COUT + 4 * lane;
  for (int j = 0; j < 4; ++j) dst[j] = o[j];
}

template <int CL, int C>
WG_FN void expand_kernel(const u8* in, u8* out, const u8* w, int h, int wd) {
  const int lane = (int)wg_lane_id();
  const int px = (int)wg_block_id(), r = px / wd, c = px % wd;
  const _Float16* bias = (const _Float16*)w;
  const u8* rows16 = w + lin_rows(4 * C) + (size_t)4 * C * CL;    // the FP16 rows (Q7F-1)
  const u8* src = in + ((size_t)r * wd + c) * CL;
  for (int g = 0; g < C / 32; ++g) {               // 4 C outputs in groups of 128: lane's 4 consecutive rows
    const int n0 = 128 * g + 4 * lane, s = n0 / C, ch = n0 % C;
    u8 o[4];
    for (int j = 0; j < 4; ++j) o[j] = out8(chain8((float)bias[n0 + j], src, rows16 + (size_t)2 * (n0 + j) * CL, CL / 32));
    u8* dst = out + ((size_t)(2 * r + (s >> 1)) * (2 * wd) + (2 * c + (s & 1))) * C + ch;
    for (int j = 0; j < 4; ++j) dst[j] = o[j];
  }
}

}  // namespace q7w

#define Q7W_BLOCK(name, C, HEADS, DEC)                                                                              \
  WG_KERNEL_NW(name, 32, 1)(const u8* x, const u8* skip, u8* out, const u8* w, const u16* tab, int H, int W, int ox, \
                            int oy) {                                                                               \
    q7w::block_kernel<C, HEADS, DEC>(x, skip, out, w, tab, H, W, ox, oy);                                          \
  }
Q7W_BLOCK(l_q7_block_c64, 64, 2, false)
Q7W_BLOCK(l_q7_block_c96, 96, 4, false)
Q7W_BLOCK(l_q7_block_c128, 128, 4, false)
Q7W_BLOCK(l_q7_block_c64_dec, 64, 2, true)
Q7W_BLOCK(l_q7_block_c96_dec, 96, 4, true)

WG_KERNEL_NW(l_q7_merge_c64, 32, 1)(const u8* in, u8* out, const u8* w, int H, int W) {
  q7w::merge_kernel<64, 96>(in, out, w, H, W);
}
WG_KERNEL_NW(l_q7_merge_c96, 32, 1)(const u8* in, u8* out, const u8* w, int H, int W) {
  q7w::merge_kernel<96, 128>(in, out, w, H, W);
}
WG_KERNEL_NW(l_q7_expand_c128, 32, 1)(const u8* in, u8* out, const u8* w, int h, int wd) {
  q7w::expand_kernel<128, 96>(in, out, w, h, wd);
}
WG_KERNEL_NW(l_q7_expand_c96, 32, 1)(const u8* in, u8* out, const u8* w, int h, int wd) {
  q7w::expand_kernel<96, 64>(in, out, w, h, wd);
}
