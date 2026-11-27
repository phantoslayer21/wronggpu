// SPDX-License-Identifier: GPL-3.0-only
// Model L (issue #89): measurement probe, not used by the runtime. One 32-term FP8 op per output computed with RDNA 4's
// FP8 matrix instruction (two 16 x 16 x 16 steps, float32 accumulator started at the op's start value), to record
// whether the instruction gives the exact sum for these shapes (clean/l/runner/l_mm8_f32_check.py --wmma).
//
// a, b: [n][32] FP8 codes (op i = sum_k a[i][k] b[i][k]); st: [n] float32 start values; out: [n] float32 results.
// Workgroup = one wave of 32 lanes computes the 16 x 16 tile of ops base ... base + 15 against each other and keeps
// the diagonal (op i's own row and column). Operand layout (gfx12, wave32): A lane l holds row l & 15, K values
// 8 (l >> 4) ... + 7; B lane l holds column l & 15, the same K values; D lane l holds column l & 15, rows
// 8 (l >> 4) ... + 7.
#include "wg_platform.h"

WG_KERNEL_NW(l_wmma_probe, 32, 1)(const u8* a, const u8* b, const float* st, float* out, int n) {
  const int base = 16 * (int)wg_block_id();
  const u32 lane = wg_lane_id();
  const int r = (int)(lane & 15u), kh = (int)(lane >> 4);
  const int op = base + r < n ? base + r : n - 1;
  v8f c;
  for (int j = 0; j < 8; ++j) {
    const int m = 8 * kh + j;
    c[j] = (m == r && base + m < n) ? st[base + m] : 0.0f;
  }
  for (int half = 0; half < 2; ++half) {
    const v2i qa = *(const v2i*)(a + (size_t)op * 32 + 16 * half + 8 * kh);
    const v2i qb = *(const v2i*)(b + (size_t)op * 32 + 16 * half + 8 * kh);
    c = wg_wmma_fp8(qa, qb, c);
  }
  for (int j = 0; j < 8; ++j) {
    const int m = 8 * kh + j;
    if (m == r && base + m < n) out[base + m] = c[j];
  }
}
