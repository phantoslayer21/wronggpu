// SPDX-License-Identifier: MIT
// Frame-uniform input constants, identical FP32 operation order to ki_host in k_io.hip.
// Native tiled callers establish the geometry guard before using these helpers.
#pragma once
#include <cfenv>
#include <cmath>
#include <cstdint>
#include <cstring>

namespace wgrt {
struct KiHost { float xi_x, xi_y, a_x, a_y, h_x, h_y, g_x, g_y, sp_x, sp_y, mp_x, mp_y; };
template <class Frame>
inline KiHost input_uniforms(const Frame& f) {
  KiHost k = {};
  const int w16 = (f.W + 15) / 16 * 16, h16 = (f.H + 15) / 16 * 16;
  k.xi_x = (1.0f / ((float)f.W * f.s)) * 0.5f;
  k.xi_y = (1.0f / ((float)f.H * f.s)) * 0.5f;
  k.a_x = (float)f.rw * k.xi_x;
  k.a_y = (float)f.rh * k.xi_y;
  k.h_x = ((float)f.W / (float)w16) * k.xi_x;
  k.h_y = ((float)f.H / (float)h16) * k.xi_y;
  k.g_x = 0.25f / (float)f.Bw;
  k.g_y = 0.25f / (float)f.Bh;
  k.sp_x = f.sx / ((1.0f / (float)f.W) * (float)((f.flags & 2) ? f.rw : f.W));
  k.sp_y = f.sy / ((1.0f / (float)f.H) * (float)((f.flags & 2) ? f.rh : f.H));
  k.mp_x = k.xi_x * (float)f.W;
  k.mp_y = k.xi_y * (float)f.H;
  return k;
}
template <class Frame>
inline bool input_uniforms_safe(const Frame& f, KiHost* result) {
  // Reuse the original GPU computation under unusual CPU rounding or scale domains.
  if (std::fegetround() != FE_TONEAREST) return false;
  const float scales[2] = {f.sx, f.sy};
  for (float v : scales) {
    uint32_t bits;
    std::memcpy(&bits, &v, sizeof bits);
    const uint32_t mag = bits & 0x7fffffffu;
    if (mag != 0 && (mag < 0x01000000u || mag >= 0x7f800000u)) return false;
  }
  *result = input_uniforms(f);
  return std::isfinite(result->sp_x) && std::isfinite(result->sp_y);
}

static_assert(sizeof(KiHost) == 48, "input uniform layout");
}  // namespace wgrt
