// SPDX-License-Identifier: MIT
// GELU table T5 of the D3D12 MLP kernels (clean/dx12/hlsl/wg_gelu_t5.hlsli): the exact q8(GELU) table of
// gelu_lut_table.h (T16, 2048 entries indexed by f16_rtz(w) >> 5) re-indexed by the FP32 encoding of w = fma(h, S, B):
// entry j = asuint(w) >> 18 (sign, 8 exponent bits, the top 5 mantissa bits), 16384 entries (a, b) of two 32-bit words.
// T5[j] = T16[f16_rtz(w) >> 5] for every FP32 w of bucket j: RTZ keeps the top 5 mantissa bits of a normal FP16 code,
// and FP16 denormal, zero, finite-overflow (0x7bff) and infinity codes depend only on bits above bit 18, so the slot is
// that of the bucket's first member. Buckets of NaN encodings take the infinity entry (a NaN w comes only from a NaN h,
// and fma(NaN, a, b) is NaN with any entry). Checked for all 2^32 w by clean/dx12/gelu_lut/t5_check.py.
// The D3D12 devices keep it in a read-only buffer behind a structured SRV (t0, space 1).
#pragma once
#include <cstddef>
#include <cstdint>
#include <vector>

#include "gelu_lut_table.h"

namespace wg {
constexpr uint32_t kGeluT5Entries = 16384;
constexpr size_t kGeluT5Bytes = size_t(kGeluT5Entries) * 8;

// FP16 code of FP32 bits b, rounded toward zero (FP16 denormals kept; FP32 denormals and zeros give a signed zero;
// finite values beyond the FP16 range give the largest finite code; NaN encodings give the infinity code).
inline uint32_t gelu_t5_f16_rtz(uint32_t b) {
  const uint32_t s = (b >> 31) << 15, e = (b >> 23) & 0xffu, m = b & 0x7fffffu;
  if (e == 255u) return s | 0x7c00u;
  if (e == 0u) return s;
  const int e16 = int(e) - 127 + 15;
  if (e16 >= 31) return s | 0x7bffu;
  if (e16 >= 1) return s | (uint32_t(e16) << 10) | (m >> 13);
  const int sh = 14 - e16;  // FP16 denormal: floor(|x| 2^24)
  return s | (sh < 32 ? ((m | 0x800000u) >> sh) : 0u);
}

// out: 2 * kGeluT5Entries words (a, b per entry).
inline void gelu_t5_expand(uint32_t* out) {
  std::vector<uint32_t> t16(kGeluLutBytes / 4);
  gelu_lut_expand(t16.data());
  for (uint32_t j = 0; j < kGeluT5Entries; ++j) {
    const uint32_t slot = gelu_t5_f16_rtz(j << 18) >> 5;
    out[2 * j] = t16[2 * slot];
    out[2 * j + 1] = t16[2 * slot + 1];
  }
}
}  // namespace wg
