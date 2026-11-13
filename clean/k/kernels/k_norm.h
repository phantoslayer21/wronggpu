// SPDX-License-Identifier: MIT
#pragma once
#include "k_fp.h"
// Spec41 normalization consumes a half-rounded nonnegative sum of squares.
// Every half input was exhaustively checked on gfx1201: all finite/Inf/zero
// result bits and NaN classes match the original sqrt-then-divide expression.
// NaN payload/sign bits are not part of that contract.
// SELECT (gfx1201): s = 0 by a select after the rsq (rsq(0) = +inf is replaced) instead of a branch around it: the
// same bits, no exec-mask switch per token. k_block.hip uses it (1440p Quality about -1 us per frame); the wide
// kernels of k_block_wide.hip keep the branch (with the select e3 took more VGPRs, one wave less per SIMD, and the
// FP32 network was about 3.5 us slower).
template <bool SELECT = false>
WG_FN h16 wgk_norm_reciprocal(h16 s16) {
#if defined(__AMDGCN__) && defined(__gfx1201__)
  if (SELECT) {
    const h16 r = kh_round(__builtin_amdgcn_rsqf((float)s16));
    return s16 == (h16)0.0f ? (h16)0.0f : r;
  }
#endif
  if (s16 == (h16)0.0f) return (h16)0.0f;
#if defined(__AMDGCN__) && defined(__gfx1201__)
  return kh_round(__builtin_amdgcn_rsqf((float)s16));
#else
  return (h16)(1.0f / __builtin_sqrtf((float)s16));
#endif
}
