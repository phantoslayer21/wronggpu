// SPDX-License-Identifier: GPL-3.0-only
// Model L (issue #89): one FP8 matrix op of the q7 kernels (up to 32 FP8 x FP8 products and an FP16 start, one FP16
// result), computed exactly (clean/l/runner/l_mm8_f32_check.py op_kernel). Kept apart from the FP16 helpers
// (l_q7_common.h) so that the FP8 op (S2a: the matrix instruction with a precomputed bound) and the FP16
// steps (S2b: native packed FP16) change in separate files. D3D12 counterpart: clean/dx12/hlsl/l_q7_op8.hlsli.
#pragma once

#include "l_q7_common.h"

namespace q7c {

// RN16(start + sum_k a[k] w(k)), exact. a: FP8 values (float), w(k): FP8 value of the weight. The products (exact in
// float32) are summed in float32 in ascending k with the float32 sum of |products| alongside; while that stays below
// 63 the sum is exact, otherwise a TwoSum chain gives it exactly as a head and an error sum; the start is added with
// error-free sums and the head rounded to FP16, the tail's sign picking the side at an FP16 midpoint.
template <typename WF>
WG_FN float op8(float start, const float* a, WF w, int K) {
  float acc = 0.0f, aacc = 0.0f;
  for (int k = 0; k < K; ++k) {
    const float p = a[k] * w(k);
    acc = acc + p;
    aacc = aacc + __builtin_fabsf(p);
  }
  if (!finite(acc) || !finite(start)) return h16(acc + start);
  float s = acc, c = 0.0f;
  if (!(aacc < 63.0f)) {
    s = 0.0f;
    for (int k = 0; k < K; ++k) {
      float e;
      two_sum(s, a[k] * w(k), s, e);
      c = c + e;
    }
  }
  float u, e1, v, e2, z, e3, t;
  two_sum(s, start, u, e1);
  two_sum(e1, c, v, e2);
  two_sum(u, v, z, e3);
  t = e3 + e2;
  two_sum(z, t, z, t);
  return rn16_pair(z, t);
}
WG_FN float op8(float start, const float* a, const float* w) {                // 32 terms, both as values
  return op8(start, a, [&](int k) { return w[k]; }, 32);
}

// ---- S2a: the FP8 matrix instruction under a precomputed bound --------------------------------
//
// One 32-wide chunk of 16 rows (A) x 16 tokens (B) = two v_wmma_f32_16x16x16_fp8_fp8 (wave32 layout: A lane l row
// l & 15, B lane l token l & 15, both K values k0 + 8 (l >> 4) ... + 7 and k0 + 16 + 8 (l >> 4) ... + 7; D lane l
// token l & 15, rows 8 (l >> 4) + j), each started from 0, their results added in float32. Exactness: every product
// of two FP8 values is a multiple of 2^-18; while sum |a w| < 64 every partial sum, in any order, is a multiple of
// 2^-18 below 2^6 (24 bits): exact in float32 (measured: 0 inexact with a zero accumulator, by the WMMA FP8 probe).
// The kernels bound sum |a w| <= wsum x amax with wsum = sum |w| of the row's chunk (blob table, l_pack.wsum_table)
// and amax = max |a| over the SAME 32 values of the token's chunk (from the codes: max (code & 0x7f), so a NaN code
// 0x7f wins and gives NaN). bounded: wsum x amax < 63 in float32 (a NaN fails it); otherwise the exact scalar op8.
// The start stays out of the accumulator (measured inexact there) and is added afterwards with one exact FP16
// addition, equal to op8's tail for an exact sum (op8's float32 sum of an all-zero chunk is +0, so a -0 sum is +0).
// Mutants for the probe only (l_q7_s2a_check.py; never set in a kernel build): 1 the bound limit 128 instead of 63,
// 2 the first instruction's result fed as the second's accumulator.
#ifndef Q7_S2A_MUTANT
#define Q7_S2A_MUTANT 0
#endif
#if Q7_S2A_MUTANT == 1
constexpr float Q7_S2A_LIMIT = 128.0f;
#else
constexpr float Q7_S2A_LIMIT = 63.0f;
#endif

// The chunk's exact product sums (when bounded) of the lane's 8 outputs; a0 / b0 the lane's 8 FP8 codes of the
// first instruction, a1 / b1 of the second (+16 K values).
WG_FN v8f q7_tile_sum(v2i a0, v2i b0, v2i a1, v2i b1) {
  const v8f zero = {};
  const v8f lo = wg_wmma_fp8(a0, b0, zero);
#if Q7_S2A_MUTANT == 2
  return wg_wmma_fp8(a1, b1, lo);
#else
  const v8f hi = wg_wmma_fp8(a1, b1, zero);
  return lo + hi;
#endif
}
// The same from the lane's operand addresses (8-byte aligned; the second instruction's codes at +16 bytes).
WG_FN v8f q7_tile_sum(const u8* a_lane, const u8* b_lane) {
  return q7_tile_sum(*(const v2i*)a_lane, *(const v2i*)b_lane, *(const v2i*)(a_lane + 16),
                     *(const v2i*)(b_lane + 16));
}
// amax from codes: fold each FP8 code into m (start 0); q7_amax of the two halves' m (or one m covering all 32).
WG_FN u32 q7_umax(u32 a, u32 b) { return a > b ? a : b; }
WG_FN u32 q7_maxcode(u32 m, u32 code) { return q7_umax(m, code & 0x7fu); }
WG_FN u32 q7_maxcode4(u32 m, u32 word) {                                      // four packed codes
  m = q7_maxcode(m, word);
  m = q7_maxcode(m, word >> 8);
  m = q7_maxcode(m, word >> 16);
  return q7_maxcode(m, word >> 24);
}
WG_FN float q7_amax(u32 m0, u32 m1) { return fp8v(q7_umax(m0, m1)); }
WG_FN bool q7_bounded(float wsum, float amax) { return wsum * amax < Q7_S2A_LIMIT; }
// RN16(start + S) for an exact chunk sum S (op8's result for the same chunk)
WG_FN float q7_join(float S, float start) { return add16(S == 0.0f ? 0.0f : S, start); }
// One output: the bounded join, else fallback(start) (the exact scalar op8 on the FP16 weights and decoded codes).
template <typename FB>
WG_FN float q7_combine(float S, float start, float wsum, float amax, FB fallback) {
  if (q7_bounded(wsum, amax)) return q7_join(S, start);
  return fallback(start);
}

}  // namespace q7c
