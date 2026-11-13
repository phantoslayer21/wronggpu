// SPDX-License-Identifier: MIT
// Number-format helpers of the runtime (host side): FP16 and FP8 E4M3FN conversions, written from the format
// definitions (IEEE 754 binary16; E4M3FN as in spec 08 section 2). Round to nearest, ties to even.
#pragma once

#include <cmath>
#include <cstdint>
#include <cstring>

namespace wgrt {

inline uint32_t f32_bits(float f) {
  uint32_t u;
  std::memcpy(&u, &f, 4);
  return u;
}
inline float bits_f32(uint32_t u) {
  float f;
  std::memcpy(&f, &u, 4);
  return f;
}

// binary32 -> binary16 bits, round to nearest even (overflow to infinity, NaN kept as a quiet NaN).
inline uint16_t f32_to_f16(float f) {
  const uint32_t x = f32_bits(f);
  const uint32_t sign = (x >> 16) & 0x8000u;
  const uint32_t a = x & 0x7fffffffu;
  if (a > 0x7f800000u) return (uint16_t)(sign | 0x7e00u);           // NaN
  if (a >= 0x477ff000u) return (uint16_t)(sign | 0x7c00u);          // >= 65520: rounds to infinity
  if (a < 0x38800000u) {                                             // below 2^-14: FP16 subnormal or zero
    // value = a_float; subnormal step 2^-24. Compute round(a * 2^24) with ties to even, exactly.
    const int e = (int)(a >> 23);                                    // biased exponent (0 for float subnormals)
    if (e < 102) return (uint16_t)sign;                              // < 2^-25: rounds to 0 (2^-25 ties to 0)
    const uint32_t m = (a & 0x7fffffu) | 0x800000u;                  // 24-bit significand, value m 2^(e-150)
    const int shift = 126 - e;                                       // value * 2^24 = m 2^(e-126) = m >> shift
    const uint32_t q = m >> shift, rem = m & ((1u << shift) - 1u), half = 1u << (shift - 1);
    const uint32_t r = q + ((rem > half || (rem == half && (q & 1u))) ? 1u : 0u);
    return (uint16_t)(sign | r);
  }
  // normal: rebias exponent (127 -> 15), round 23 -> 10 mantissa bits
  const uint32_t r = a + 0xfffu + ((a >> 13) & 1u);
  return (uint16_t)(sign | ((r - (112u << 23)) >> 13));
}

inline float f16_to_f32(uint16_t h) {
  const uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
  const uint32_t e = (h >> 10) & 0x1fu, m = h & 0x3ffu;
  if (e == 0) return bits_f32(sign) + (sign ? -1.0f : 1.0f) * (float)m * 5.9604644775390625e-08f;  // m 2^-24
  if (e == 31) return bits_f32(sign | 0x7f800000u | (m << 13));
  return bits_f32(sign | ((e + 112u) << 23) | (m << 13));
}

inline float rn16(float x) { return f16_to_f32(f32_to_f16(x)); }

// E4M3FN code -> value (NaN for the two NaN codes).
inline double e4m3_value(uint8_t c) {
  const int e = (c >> 3) & 15, m = c & 7;
  double v;
  if (e == 15 && m == 7) v = NAN;
  else if (e == 0) v = m * std::ldexp(1.0, -9);
  else v = (1.0 + m / 8.0) * std::ldexp(1.0, e - 7);
  return (c & 0x80) ? -v : v;
}

// q8: nearest E4M3FN value, ties to even, saturating at +-448 (NaN stays NaN).
inline double q8(double x) {
  if (std::isnan(x)) return x;
  const double a = std::fabs(x);
  if (a == 0.0) return x;
  int ex;
  std::frexp(a, &ex);                                    // a = m 2^ex, m in [0.5, 1)
  const int exp = ex - 1 < -6 ? -6 : ex - 1;
  const double spacing = std::ldexp(1.0, exp - 3);
  double r = std::nearbyint(a / spacing) * spacing;      // default rounding mode: nearest even
  if (r > 448.0) r = 448.0;
  return std::copysign(r, x);
}

// Value -> E4M3FN code (after q8; -0 keeps its sign bit, NaN -> 0x7F).
inline uint8_t fp8_encode(double x) {
  if (std::isnan(x)) return 0x7f;
  const double v = q8(x);
  const double a = std::fabs(v);
  uint8_t code;
  if (a < 0.015625) {
    code = (uint8_t)std::lrint(a * 512.0);               // subnormal: m 2^-9 (8 -> the smallest normal, code 8)
  } else {
    int ex;
    std::frexp(a, &ex);
    const int e = ex - 1;                                 // 2^e <= a < 2^(e+1)
    const int m = (int)std::lrint((a / std::ldexp(1.0, e) - 1.0) * 8.0);
    code = (uint8_t)(((e + 7) << 3) | m);
  }
  return (uint8_t)(code | (std::signbit(v) ? 0x80 : 0));
}

}  // namespace wgrt
