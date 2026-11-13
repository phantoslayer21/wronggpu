// SPDX-License-Identifier: MIT
// Original full-support Lanczos-3 output-filter profile, D/O=1.5, phase +1/4.
// Derived from spec 06's mathematical definition, normalized over k=-4..4, then minimally corrected so
// sum(w)=1 and sum(k*w)=1/4. See clean/reference/output_quality.py. No learned parameters or vendor code.
#pragma once
#ifndef WG_Q9_FN
#define WG_Q9_FN inline
#endif

namespace wg_quality9 {
constexpr float even_weights[9] = {
    0x1.5482dcp-9f, 0x1.149a98p-6f, -0x1.712cecp-4f, 0x1.cb91acp-4f, 0x1.454272p-1f,
    0x1.a0241ap-2f, -0x1.202ff8p-4f, -0x1.d666d2p-6f, 0x1.03d8c6p-6f};

template <bool Even>
WG_Q9_FN float filter(const float* t) {
  float value = even_weights[Even ? 0 : 8] * t[0];
  float lo = t[0], hi = t[0];
#pragma unroll
  for (int i = 1; i < 9; ++i) {
    value = __builtin_fmaf(even_weights[Even ? i : 8 - i], t[i], value);
    lo = __builtin_fminf(lo, t[i]);
    hi = __builtin_fmaxf(hi, t[i]);
  }
  return __builtin_fminf(__builtin_fmaxf(value, lo), hi);
}
}  // namespace wg_quality9
