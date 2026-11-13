// SPDX-License-Identifier: MIT
// Model L (issue #89): the small FP8-map kernels of Model L's encoder and decoder chains (clean/l/runner/l_pack.py
// encoder_launches). Maps are [tokens][C] FP8 E4M3 bytes, channel fastest (clean/FORMATS.md "Feature maps"). Blobs:
// clean/l/runner/l_pack.py. Launch: 1-D grid of ceil(tokens / 128) workgroups of 256 lanes (8 waves); wave i handles
// tokens 16 i .. 16 i + 15; lane (t, h) works on token 16 i + t, lane half h on channels [h C/2, (h + 1) C/2) (layout L)
// or, for the projection, on the 16 channels of each group that its K slots name.
#include "wg_platform.h"
#include "wg_mem.h"

namespace {

constexpr int kThreads = 256;
constexpr int kTokensPerWave = 16;

WG_FN int token_of(u32 tid) {
  return (int)(wg_block_id() * (kThreads / WG_WAVE) + (tid >> 5)) * kTokensPerWave + (int)(tid & 15u);
}

// y = q8(W x + b), FP32 accumulation over the K groups in order. Blob: bias tile vector (rows R_L(CO)), then fragments
// [mt < CO/16][kc < CI/16] with rows R_L(CO) and K order K_L(CI) (merge / expand section format).
template <int CI, int CO>
WG_FN void linear_map(const u8* __restrict__ x, u8* __restrict__ y, const u8* __restrict__ w, int ntok) {
  constexpr int KC = CI / 16, MT = CO / 16;
  static_assert(CI % 16 == 0 && CO % 16 == 0, "widths in multiples of 16");
  const u32 tid = wg_lane_id(), lane = tid & 31u, hh = (tid >> 4) & 1u;
  const int tok = token_of(tid);
  const int tk = tok < ntok ? tok : ntok - 1;
  v2i xf[KC];
#pragma unroll
  for (int kc = 0; kc < KC; ++kc) xf[kc] = *(const v2i*)(x + (size_t)tk * CI + hh * (CI / 2) + 8 * kc);
  const float* bias = (const float*)w + hh * 8;
  const u8* frag = w + 4 * CO;
#pragma unroll
  for (int mt = 0; mt < MT; ++mt) {
    v8f acc = wg_load8f(bias + mt * 16);
#pragma unroll
    for (int kc = 0; kc < KC; ++kc) acc = wg_wmma_fp8(((const v2i*)(frag + (mt * KC + kc) * 256))[lane], xf[kc], acc);
    const v2i q = wg_pack_q8(acc);
    if (tok < ntok) *(v2i*)(y + (size_t)tok * CO + hh * (CO / 2) + 8 * mt) = q;
  }
}

// 8 bytes of the concatenated input z = [a (CA channels) | b (CB channels) | zeros] at channel c (c a multiple of 8;
// CA and CB are multiples of 8, so the 8 bytes never straddle two parts).
template <int CA, int CB>
WG_FN v2i concat8(const u8* a, const u8* b, int tk, int c) {
  if (c < CA) return *(const v2i*)(a + (size_t)tk * CA + c);
  if (c < CA + CB) return *(const v2i*)(b + (size_t)tk * CB + (c - CA));
  v2i z = {0, 0};
  return z;
}

// features = q8(clamp(s, +-6.28125)), s = b; for each group g of 32 channels of z: s = RN16(s + W[:, g] z[g]).
// Blob: bias tile vector (rows R_L(64)), then fragments [mt < 4][kb < NZ/16], rows R_L(64), K order: block kb slot
// (h, j) = channel 32 (kb / 2) + 16 h + 8 (kb % 2) + j (clean/l/runner/l_pack.py kord_groups32).
template <int CA, int CB, int NZ>
WG_FN void project_map(const u8* __restrict__ a, const u8* __restrict__ b, u8* __restrict__ feat,
                       const u8* __restrict__ w, int ntok) {
  constexpr int CO = 64, MT = CO / 16, KB = NZ / 16;
  static_assert(NZ % 32 == 0 && CA + CB <= NZ && CA % 8 == 0 && CB % 8 == 0, "groups of 32, parts of 8 bytes");
  const u32 tid = wg_lane_id(), lane = tid & 31u, hh = (tid >> 4) & 1u;
  const int tok = token_of(tid);
  const int tk = tok < ntok ? tok : ntok - 1;
  v2i zf[KB];
#pragma unroll
  for (int kb = 0; kb < KB; ++kb) zf[kb] = concat8<CA, CB>(a, b, tk, 32 * (kb >> 1) + 16 * (int)hh + 8 * (kb & 1));
  const float* bias = (const float*)w + hh * 8;
  const u8* frag = w + 4 * CO;
#pragma unroll
  for (int mt = 0; mt < MT; ++mt) {
    v8f acc = wg_load8f(bias + mt * 16);
#pragma unroll
    for (int g = 0; g < KB / 2; ++g) {
      acc = wg_wmma_fp8(((const v2i*)(frag + (mt * KB + 2 * g) * 256))[lane], zf[2 * g], acc);
      acc = wg_wmma_fp8(((const v2i*)(frag + (mt * KB + 2 * g + 1) * 256))[lane], zf[2 * g + 1], acc);
#pragma unroll
      for (int i = 0; i < 8; ++i) acc[i] = wg_rn16(acc[i]);
    }
#pragma unroll
    for (int i = 0; i < 8; ++i) acc[i] = wg_fmin(wg_fmax(acc[i], -6.28125f), 6.28125f);
    const v2i q = wg_pack_q8(acc);
    if (tok < ntok) *(v2i*)(feat + (size_t)tok * CO + hh * (CO / 2) + 8 * mt) = q;
  }
}

}  // namespace

// Pointwise linears of Model L's encoder: 80 -> 32 (the raw value map; the select is zero weights) and 32 -> 32.
WG_KERNEL_N(l_linear_80_32, kThreads)(const u8* x, u8* y, const u8* w, int ntok) { linear_map<80, 32>(x, y, w, ntok); }
WG_KERNEL_N(l_linear_32_32, kThreads)(const u8* x, u8* y, const u8* w, int ntok) { linear_map<32, 32>(x, y, w, ntok); }

// The encoder's output projection over [raw (80) | x (32) | 16 zeros] -> the 64 FP8 features enc1 reads.
WG_KERNEL_N(l_project_80_32, kThreads)(const u8* raw, const u8* x, u8* feat, const u8* w, int ntok) {
  project_map<80, 32, 128>(raw, x, feat, w, ntok);
}

// dst[t][doff .. doff + n) = src[t][soff .. soff + n) for every token t < ntok, n a multiple of 16 bytes; one lane per
// 16 bytes. Splits dec1's map for the decoder block and joins the block output back (clean/l/runner).
WG_KERNEL_N(l_copy_channels, kThreads)(const u8* src, int sstride, int soff, u8* dst, int dstride, int doff, int n,
                                       int ntok) {
  const int chunks = n >> 4;
  const long long i = (long long)wg_block_id() * kThreads + wg_lane_id();
  const long long t = i / chunks;
  if (t >= ntok) return;
  const int k = (int)(i - t * chunks);
  *(v4u*)(dst + (size_t)t * dstride + doff + 16 * k) = *(const v4u*)(src + (size_t)t * sstride + soff + 16 * k);
}
