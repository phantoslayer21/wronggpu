// SPDX-License-Identifier: MIT
#include "optimal.h"

#include <math.h>

namespace wgrr {

namespace {
// The single-precision values nearest to the exact ratios 2/3, 0.58 (= 29/50), 1/2, 1/3 and 1. A double holds
// each exact ratio to 53 bits; none of these lies within 2^-29 relative of a float midpoint, so rounding the
// double to float gives the float nearest to the exact ratio (tests/test_optimal.py checks this with Fractions).
const float kTwoThirds = (float)(2.0 / 3.0);
const float kBalanced = (float)(29.0 / 50.0);
const float kHalf = 0.5f;
const float kThird = (float)(1.0 / 3.0);
}  // namespace

bool mode_ratio(uint32_t mode, float* r) {
  switch (mode) {
    case 1: *r = kHalf; return true;       // max performance
    case 2: *r = kBalanced; return true;   // balanced
    case 3: *r = kTwoThirds; return true;  // max quality
    case 4: *r = kThird; return true;      // ultra performance
    case 6: *r = 1.0f; return true;        // native-resolution AA
    default: return false;                 // 0 off, 5 ultra quality (no ratio), > 6 not a mode
  }
}

uint32_t scaled_size(uint32_t output, float r) {
  volatile float o = (float)output;  // fl32(output)
  volatile float p = o * r;          // one single-precision product, round to nearest even
  double q = floor((double)p + 0.5);  // nearest integer, halves up (exact in double for p < 2^52)
  if (q <= 0.0) return 0;
  if (q >= 4294967295.0) return 0xffffffffu;
  return (uint32_t)q;
}

uint32_t min_render_size(uint32_t output) { return scaled_size(output, kThird); }

OptimalAnswer optimal_settings(uint32_t mode, uint32_t w, uint32_t h) {
  OptimalAnswer a;
  float r;
  if (!mode_ratio(mode, &r)) return a;
  a.has_ratio = true;
  a.width = scaled_size(w, r);
  a.height = scaled_size(h, r);
  a.min_width = min_render_size(w);
  a.min_height = min_render_size(h);
  a.max_width = w;
  a.max_height = h;
  a.sharpness = 0.0f;
  return a;
}

}  // namespace wgrr
