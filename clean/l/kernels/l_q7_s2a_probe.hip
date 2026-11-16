// SPDX-License-Identifier: MIT
// Model L (issue #89): measurement probe of the S2a primitive (clean/l/kernels/l_q7_op8.h: q7_tile_sum, q7_maxcode4,
// q7_amax, q7_combine with the exact op8 as the fallback), not used by the runtime (clean/l/runner/l_q7_s2a_check.py).
// 16 single-chunk ops per workgroup (one wave of 32 lanes) on the tile's diagonal: A row m = op base + m's weight row,
// B token m = op base + m's activations; op i's result is D[row i][token i]. amax: each lane folds the 16 codes of its
// own B operands (both instructions), the token's two lanes l, l ^ 16 are combined (the same 32 values the sums use).
// a, w: [n][32] FP8 codes; w16: [n][32] FP16 bits (the weight row's exact copy, the fallback's operand); st, wsum: [n]
// float32; out: [n] float32; path: [n] (1 bounded = the instruction's sum, 0 = the exact scalar fallback). n % 16 == 0.
// Mutants (-DQ7_S2A_MUTANT=N, probe builds only): 1, 2 in l_q7_op8.h; 3 amax from the lane's own 16 codes only;
// 4 the start fed into the first instruction's accumulator (joined with 0 afterwards).
#pragma clang fp contract(off)
#include "wg_platform.h"
#include "l_q7_op8.h"

using namespace q7c;

WG_KERNEL_NW(l_q7_s2a_probe, 32, 1)(const u8* a, const u8* w, const u16* w16, const float* st, const float* wsum,
                                    float* out, u32* path) {
  WG_LDS u32 lm[32];
  const int base = 16 * (int)wg_block_id();
  const u32 lane = wg_lane_id(), t = lane & 15u, hf = lane >> 4;
  const u8* pw = w + (size_t)(base + (int)t) * 32 + 8 * hf;
  const u8* pa = a + (size_t)(base + (int)t) * 32 + 8 * hf;
  const v2i a0 = *(const v2i*)pw, a1 = *(const v2i*)(pw + 16);
  const v2i b0 = *(const v2i*)pa, b1 = *(const v2i*)(pa + 16);
  u32 m = q7_maxcode4(0u, (u32)b0[0]);
  m = q7_maxcode4(m, (u32)b0[1]);
  m = q7_maxcode4(m, (u32)b1[0]);
  m = q7_maxcode4(m, (u32)b1[1]);
  lm[lane] = m;
  wg_barrier_lds();
#if Q7_S2A_MUTANT == 3
  const float amax = q7_amax(lm[lane], lm[lane]);
#else
  const float amax = q7_amax(lm[t], lm[t + 16u]);
#endif
  const bool diag = (t >> 3) == hf;
  const int i = base + (int)t;
#if Q7_S2A_MUTANT == 4
  v8f c = {};
  for (int j = 0; j < 8; ++j) c[j] = (8 * (int)hf + j == (int)t) ? st[i] : 0.0f;
  const v8f zero = {};
  const v8f S = wg_wmma_fp8(a0, b0, c) + wg_wmma_fp8(a1, b1, zero);
  const float start = 0.0f;
#else
  const v8f S = q7_tile_sum(a0, b0, a1, b1);
  const float start = diag ? st[i] : 0.0f;
#endif
  if (diag) {
    const float s = S[t & 7u];
    const float ws = wsum[i];
    const u8* act = a + (size_t)i * 32;
    const u16* row = w16 + (size_t)i * 32;
    out[i] = q7_combine(s, start, ws, amax, [&](float) {     // (mutant 4: the fallback keeps the real start)
      float av[32];
      for (int k = 0; k < 32; ++k) av[k] = fp8v(act[k]);
      return op8(st[i], av, [&](int k) { return from16(row[k]); }, 32);
    });
    path[i] = q7_bounded(ws, amax) ? 1u : 0u;
  }
}
