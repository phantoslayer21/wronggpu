// SPDX-License-Identifier: GPL-3.0-only
// The arena tables of the HYBRID block form (hlsl/wg_hyb.hlsli): devices that run SW_HYB kernels reserve arena bytes
// [kHybBase, kHybBase + kHybBytes) and fill them with hyb_fill(). The same values as clean/l/reference/hyb_tables.py
// (checked by clean/l/tests/test_hyb_tables_cpp.py through clean/tests/hyb_dump.cpp: identical bytes):
//   [0, 65536)        q8(GELU16(h)) by the FP16 bits of h (oracle gelu_q7 + the saturating FP8 rounding)
//   [65536, 69632)    r = RN16(1 / sqrtf(S')) for S' = 1.m x 2^p, p = 0 / 1 (u16, [p][m])
//   [69632, 71680)    w = RN16(1 / x) for x = 1.m (u16, [m])
//   [71680, 202752)   the GELU16 table again with 16-bit entries (u16 = the FP8 byte, high byte 0): the D3D12
//                     kernels load one u16 per value at h << 1 (no byte extraction; hlsl/wg_hyb.hlsli hyb_gelu_q8)
// FP16 steps are float operations rounded once to FP16 (exact: float has 24 >= 2 x 11 + 2 bits); sqrtf and the
// division are IEEE correctly rounded, as numpy's float32 in the oracle.
#pragma once
#include <cfenv>
#include <cmath>
#include <cstdint>
#include <cstring>

#include "gelu_lut_table.h"

namespace wg {

constexpr uint64_t kHybBase = kGeluLutBase + kGeluLutBytes;   // = WG_HYB_BASE
constexpr size_t kHybBytes = 65536 + 4096 + 2048 + 131072;

inline uint16_t hyb_f2h(float f) {   // round to nearest even, IEEE half (subnormals, overflow to inf, NaN quiet)
  uint32_t x;
  std::memcpy(&x, &f, 4);
  const uint32_t sign = (x >> 16) & 0x8000u;
  const uint32_t ax = x & 0x7fffffffu;
  if (ax > 0x7f800000u) return (uint16_t)(sign | 0x7e00u);
  if (ax >= 0x477ff000u) return (uint16_t)(sign | 0x7c00u);   // >= 65520: inf (65504 + half ulp rounds up)
  if (ax < 0x38800000u) {                                      // below 2^-14: subnormal half
    const float a = std::fabs(f) * 16777216.0f;                // x 2^24: exact scaling
    const float r = std::nearbyint(a);                         // ties to even (default rounding mode)
    return (uint16_t)(sign | (uint32_t)r);
  }
  const uint32_t m = ax & 0x7fffffu;
  uint32_t h = ((ax >> 23) - 112u) << 10 | (m >> 13);
  const uint32_t rest = m & 0x1fffu;
  if (rest > 0x1000u || (rest == 0x1000u && (h & 1u))) ++h;
  return (uint16_t)(sign | h);
}

inline float hyb_h2f(uint16_t h) {
  const uint32_t s = (uint32_t)(h & 0x8000u) << 16, e = (h >> 10) & 31u, m = h & 1023u;
  uint32_t x;
  if (e == 0) {
    const float v = (float)m * (1.0f / 16777216.0f);
    std::memcpy(&x, &v, 4);
    x |= s;
  } else if (e == 31) {
    x = s | 0x7f800000u | (m << 13);
  } else {
    x = s | ((e + 112u) << 23) | (m << 13);
  }
  float f;
  std::memcpy(&f, &x, 4);
  return f;
}

inline float hyb_rn16(float f) { return hyb_h2f(hyb_f2h(f)); }

// FP8 E4M3FN, round to nearest even, saturating at +-448 (clean/reference/fp8.py q8 + encode); NaN -> 0xff, the
// hardware conversion's NaN on HIP and D3D12.
inline uint8_t hyb_fp8(float v) {
  if (std::isnan(v)) return 0xffu;
  const uint8_t s = std::signbit(v) ? 0x80u : 0u;
  float a = std::fabs(v);
  if (a >= 448.0f) return (uint8_t)(s | 0x7eu);
  int e;
  std::frexp(a, &e);                                   // a = m 2^e, m in [0.5, 1)
  int ex = e - 1;
  if (ex < -6) ex = -6;
  const float spacing = std::ldexp(1.0f, ex - 3);
  const float r = std::nearbyint(a / spacing) * spacing;   // exact quotient; ties to even
  if (r >= 448.0f) return (uint8_t)(s | 0x7eu);
  if (r == 0.0f) return s;
  int er;
  const float mr = std::frexp(r, &er);
  int field = er - 1 + 7;
  if (field <= 0) return (uint8_t)(s | (uint8_t)(r / std::ldexp(1.0f, -9)));   // subnormal: r = k 2^-9
  const int mant = (int)((mr * 2.0f - 1.0f) * 8.0f);
  return (uint8_t)(s | (uint8_t)(field << 3) | (uint8_t)mant);
}

inline float hyb_gelu16(float h) {   // oracle gelu_q7: every step its own FP16 rounding, no fused operation
  const float a = hyb_h2f(0x3698u), b = hyb_h2f(0x2d30u);
  const float y = std::fmin(std::fmax(h, -2.0f), 2.0f);
  const float inner = hyb_rn16(a - hyb_rn16(b * std::fabs(y)));
  return hyb_rn16(h * hyb_rn16(0.5f + hyb_rn16(y * inner)));
}

// The values above assume round-to-nearest-even float arithmetic (nearbyint, the float steps, sqrtf and the division
// all follow the thread's rounding mode); hyb_fill() sets that mode for the generation and restores the caller's.
inline void hyb_fill_nearest(uint8_t* dst) {
  for (uint32_t i = 0; i < 65536; ++i) dst[i] = hyb_fp8(hyb_gelu16(hyb_h2f((uint16_t)i)));
  uint8_t* g16 = dst + 71680;   // little-endian u16 entries: byte, 0
  for (uint32_t i = 0; i < 65536; ++i) { g16[2 * i] = dst[i]; g16[2 * i + 1] = 0; }
  uint16_t* rsq = (uint16_t*)(dst + 65536);
  uint16_t* rcp = (uint16_t*)(dst + 69632);
  for (uint32_t m = 0; m < 1024; ++m) {
    const float base = 1.0f + (float)m / 1024.0f;
    rsq[m] = hyb_f2h(1.0f / std::sqrt(base));
    rsq[1024 + m] = hyb_f2h(1.0f / std::sqrt(base * 2.0f));
    rcp[m] = hyb_f2h(1.0f / base);
  }
}

// The tables with round-to-nearest-even arithmetic whatever the host thread's rounding mode; the caller's mode is
// restored (test_hyb_tables_cpp.py: identical bytes under FE_DOWNWARD, mode restored).
inline void hyb_fill(uint8_t* dst) {
  const int mode = std::fegetround();
  std::fesetround(FE_TONEAREST);
  hyb_fill_nearest(dst);
  std::fesetround(mode);
}
}  // namespace wg
