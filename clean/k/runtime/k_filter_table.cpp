// SPDX-License-Identifier: MIT
// Model K filter-shape table (spec 43 draft 2 section 7); see k_filter_table.h.
#include "k_filter_table.h"

#include <cmath>
#include <cstring>

// Section 7: every operation rounded on its own; a fused multiply-add would round once and change the bits.
#if defined(_MSC_VER) && !defined(__clang__)
#pragma fp_contract(off)
#else
#pragma STDC FP_CONTRACT OFF
#endif

namespace wgrt {

namespace {

float bits_f32(uint32_t b) {
  float f;
  std::memcpy(&f, &b, 4);
  return f;
}

// "max(x, y) means x if x > y, else y" (section 7).
float vmax(float x, float y) { return x > y ? x : y; }

}  // namespace

uint16_t k_f32_to_f16_trunc(float x) {
  uint32_t b;
  std::memcpy(&b, &x, 4);
  const uint16_t sign = (uint16_t)((b >> 16) & 0x8000u);
  const float a = std::fabs(x);
  if (std::isnan(x)) return (uint16_t)(sign | 0x7E00u);
  if (a > 65504.0f) return (uint16_t)(sign | 0x7C00u);  // 7.8: above 65504 -> infinity (65519 too)
  if (a < 0x1p-14f) {
    // FP16 subnormal (or zero): multiples of 2^-24; a * 2^24 is exact (power-of-two scale, < 2^10), floor truncates.
    return (uint16_t)(sign | (uint16_t)std::floor(a * 0x1p24f));
  }
  const uint32_t e = (b >> 23) & 0xFFu;  // 113 ... 142 here
  const uint32_t m = b & 0x7FFFFFu;
  return (uint16_t)(sign | ((e - 112u) << 10) | (m >> 13));  // dropping the low 13 mantissa bits truncates
}

int k_filter_level(int rw, int rh, int W, int H) {
  const float rh_ = (float)rh * (1.0f / (float)H);  // r_h RN32(1/H)
  const float rw_ = (float)rw * (1.0f / (float)W);
  const float prod = (0.5f * rh_) * (0.5f * rw_);
  const float v = (7.0f * (std::sqrt(1.0f / prod) - 1.0f)) / 3.0f;
  if (std::isnan(v)) return 0;
  // R: round half away from zero (spec 42 section 1); exact in double for a binary32 argument.
  const double t = (double)v;
  const double r = std::copysign(std::trunc(std::fabs(t) + 0.5), t);
  return r < 0.0 ? 0 : r > 7.0 ? 7 : (int)r;
}

void k_filter_table(int level, int theta, uint16_t* out) {
  const float Q = 8.0f * (1.0f + bits_f32(0x38D1B717u));  // 7.1: Q = 8 (1 + 1e-4)
  const float phi = (3.0f * (float)level) / 8.0f + 1.0f;   // level factor
  const float sqrt_half = std::sqrt(0.5f);                 // RN32(sqrt(1/2))
  const float kappa = -(1.0f / (float)std::log(2.0));      // 7.6: -(1 / RN32(ln 2))
  auto value = [Q](int k) {
    const float z = ((float)k - 8.0f) / Q;
    return z / (1.0f - std::fabs(z));
  };
  for (int idx = 0; idx < kFilterEntries; ++idx) {
    const int flags = idx / 256;
    const bool F0 = (flags & 1) != 0, F1 = (flags & 2) != 0, F2 = (flags & 4) != 0;
    const float a = value((idx / 16) % 16), b = value(idx % 16);
    // 7.2 widths
    const bool narrow = !F2 && F1;
    const float hi = narrow ? 0.5f : 2.1f;
    float lo = narrow ? 0.1f : 0.17f;
    // 7.3 direction
    const float s = std::fabs(b) + std::fabs(a);
    const float omega = s < 0.25f ? 1.0f - 4.0f * std::fabs(b) : std::fabs(a) / s;
    const float sigma_b = b > 0.0f ? 1.0f : -1.0f;
    float tau = (sqrt_half * sigma_b) * (1.0f - omega);
    if (!(a >= 0.0f)) tau = -tau;
    const float gamma = std::sqrt(1.0f - tau * tau);
    const float r2 = a * a + b * b;
    // 7.4 major width
    float M = (hi - lo) * (r2 / (r2 + 1.0f)) + lo;
    if (narrow && M > 0.4f) lo = 0.08f;
    if (theta == 0) {
      lo = vmax(0.2f, lo);
      M = vmax(lo, M);
    }
    if (F2) {
      lo = vmax(0.5f, lo);
      M = vmax(lo, M);
    }
    // 7.5 axes
    const bool pos = a >= 0.0f;
    const float P = pos ? gamma / lo : gamma / M;
    const float Qp = pos ? tau / lo : tau / M;
    const float Rr = pos ? tau / M : tau / lo;
    const float Mp = pos ? M : lo;
    const float Sp = (-gamma) / Mp;
    // 7.6 coefficients
    const float Bxy = (((Sp * Rr) + (Qp * P)) * 2.0f) * kappa;
    const float u = pos ? gamma : -tau;
    const float v = -(pos ? tau : gamma);
    const bool O = std::fabs(u) > std::fabs(v);
    float shear, A, Cxx;
    if (O) {
      shear = v / u;
      A = kappa * (Sp * Sp + Qp * Qp);
      Cxx = (Rr * Rr + P * P) * kappa;
    } else {
      shear = u / v;
      A = kappa * (Rr * Rr + P * P);
      Cxx = (Sp * Sp + Qp * Qp) * kappa;
    }
    // 7.7 blend constants
    float w = r2 / (r2 + 1.0f);
    if (!F2 && F0) w = 0.0f;
    const float k6 = w * phi;
    const float K = (((phi - phi * phi) * w) + phi * phi) - k6;
    // 7.8 entry
    uint16_t* e = out + (size_t)idx * kFilterValues;
    e[0] = k_f32_to_f16_trunc(A);
    e[1] = k_f32_to_f16_trunc(Cxx);
    e[2] = k_f32_to_f16_trunc(Bxy);
    e[3] = k_f32_to_f16_trunc(shear);
    e[4] = O ? 1 : 0;  // O as the integer bit pattern 0 or 1
    e[5] = k_f32_to_f16_trunc(K);
    e[6] = k_f32_to_f16_trunc(k6);
    e[7] = 0;
  }
}

std::vector<uint16_t> k_filter_table(int level, int theta) {
  std::vector<uint16_t> t((size_t)kFilterEntries * kFilterValues);
  k_filter_table(level, theta, t.data());
  return t;
}

}  // namespace wgrt
