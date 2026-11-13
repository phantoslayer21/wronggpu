// SPDX-License-Identifier: MIT
// Optimal render settings (spec 61.5, WrongGPU's rule) and the accepted render-scale window (61.4 step 3).
#pragma once
#include <stdint.h>

namespace wgrr {

// Per-mode ratio r as the single-precision value nearest to the exact ratio; returns false for modes without a
// ratio (0 off, 5 ultra quality, > 6).
bool mode_ratio(uint32_t mode, float* r);

// round(output x r): p = fl32(fl32(output) x r), then the integer nearest to p, exact halves rounded up.
uint32_t scaled_size(uint32_t output, float r);

// Minimum accepted / reported render size per axis: round(output x 1/3).
uint32_t min_render_size(uint32_t output);

struct OptimalAnswer {
  bool has_ratio = false;  // false: zero render size (modes 0 and 5) or unknown mode
  uint32_t width = 0, height = 0, min_width = 0, min_height = 0, max_width = 0, max_height = 0;
  float sharpness = 0.0f;
};
OptimalAnswer optimal_settings(uint32_t mode, uint32_t output_width, uint32_t output_height);

}  // namespace wgrr
