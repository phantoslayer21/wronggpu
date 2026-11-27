// SPDX-License-Identifier: GPL-3.0-only
#pragma once

#include "../../runtime/fpconv.h"

#include <algorithm>
#include <array>
#include <cmath>

namespace wgrt::ki_s_breakpoints {

inline constexpr uint32_t kFirst = 0x3b4d2e1du, kLast = 0x3f800000u;
inline constexpr uint32_t kBucket = 4096, kBase = kFirst & ~(kBucket - 1);
using Table = std::array<uint32_t, (kLast - kBase) / kBucket + 1>;

// FNV-1a of the little-endian table verified by prove_ki_s_breakpoints.py.
inline constexpr uint64_t kProvenDigest = 0xfe032345814078feULL;

inline uint64_t table_digest(const Table& table) {
  uint64_t hash = 0xcbf29ce484222325ULL;
  for (uint32_t entry : table)
    for (unsigned shift = 0; shift < 32; shift += 8)
      hash = (hash ^ ((entry >> shift) & 0xffu)) * 0x100000001b3ULL;
  return hash;
}

#if defined(_MSC_VER) && !defined(__clang__)
#pragma float_control(precise, on, push)
#pragma fp_contract(off)
#elif defined(__GNUC__) && !defined(__clang__)
#pragma GCC push_options
#pragma GCC optimize ("fp-contract=off")
#endif

inline float log32(float x) {
#if defined(__clang__)
#pragma STDC FP_CONTRACT OFF
#endif
  constexpr double kLn2 = 0.6931471805599453;
  int exponent;
  double m = std::frexp(static_cast<double>(x), &exponent);
  if (m < 0.7071067811865476) {
    m *= 2.0;
    --exponent;
  }
  const double t = (m - 1.0) / (m + 1.0);
  const double t2 = t * t;
  double p = 1.0 / 23.0;
  for (int n = 21; n >= 1; n -= 2) p = p * t2 + 1.0 / n;
  return static_cast<float>(exponent + (2.0 / kLn2) * t * p);
}

inline float exp32(float x) {
#if defined(__clang__)
#pragma STDC FP_CONTRACT OFF
#endif
  constexpr double kLn2 = 0.6931471805599453;
  // The reduced exponent is integral. Either side of a tie is valid here.
  const int n = static_cast<int>(std::floor(static_cast<double>(x) + 0.5));
  const double t = (static_cast<double>(x) - n) * kLn2;
  double p = 1.0;
  for (int k = 14; k >= 1; --k) p = 1.0 + (t / k) * p;
  return static_cast<float>(std::ldexp(p, n));
}

#if defined(_MSC_VER) && !defined(__clang__)
#pragma float_control(pop)
#elif defined(__GNUC__) && !defined(__clang__)
#pragma GCC pop_options
#endif

inline uint16_t half_code(uint32_t word) {
  const float y = log32(bits_f32(word)) * bits_f32(0x3ed55555u);
  return f32_to_f16(std::fma(exp32(y), bits_f32(0x3f870a3du), bits_f32(0xbd6147aeu)));
}

inline Table make_table() {
  Table table;
  for (size_t i = 0; i < table.size(); ++i) {
    const uint32_t base = kBase + static_cast<uint32_t>(i) * kBucket;
    uint32_t lo = std::max(base, kFirst), hi = std::min(base + kBucket - 1, kLast);
    const uint16_t first = half_code(lo);
    uint32_t cut = kBucket;
    if (first != half_code(hi)) {
      // The exhaustive CPU proof verifies that each bucket is flat or
      // increments by one once. Find the first code that increments.
      while (lo < hi) {
        const uint32_t mid = lo + (hi - lo) / 2;
        if (half_code(mid) == first)
          lo = mid + 1;
        else
          hi = mid;
      }
      cut = lo - base;
    }
    table[i] = first | (cut << 16);
  }
  return table;
}

}  // namespace wgrt::ki_s_breakpoints
