// SPDX-License-Identifier: MIT
// Original pre-exposure division window of the direct-resource kernels, 2026-10-16.
// Basis: IEEE-754 binary32 arithmetic; clean/dx12/hlsl/native_direct.hlsli native_direct_color_div3 (the GPU side).
//
// The direct kernels divide the game's colour by the pre-exposure pi (HIP: color_divide, wg_div). For one pi per frame
// the kernels take q0 = a * y, rn = fma(pi, q0, -a), q = fma(-rn, y, q0) with y = RN(1 / pi) on the lanes whose |a| lies
// in [lo, hi) (or is zero / an FP32 denormal); every other lane runs the reference division. With a = as 2^A,
// pi = bs 2^B (as, bs in [1, 2)) and every intermediate normal, each operation is the [1, 2) significand operation
// scaled by a power of two, so q equals the reference exactly when it does for (as, bs); that was checked for all
// 2^23 x 2^23 significand pairs (build/big-async/r3/mproof). The window keeps every intermediate normal and finite:
//   q0, q in [2^(A-B-1), 2^(A-B+1)]          -> A - B in [-125, 126]
//   rn = pi q0 - a, a nonzero multiple of 2^(A-47) -> A >= -79
//   y in [2^(-B-1), 2^-B]                     -> B in [-126, 125]
// lo = 2^max(-79, B - 125), hi = 2^(B + 127) (+inf for B > 0: |a| < hi also excludes inf and NaN).
// A pi outside the domain (not a positive normal with B <= 125), or a y that is not RN(1 / pi) (the CPU rounding mode
// changed by the host), gets hi = 0: no lane takes the fast form.
#pragma once

#include <cmath>
#include <cstdint>
#include <cstring>

namespace wgrt {

struct NativeDirectPiWindow { uint32_t rcp = 0, lo = 0x7f800000u, hi = 0; };   // FP32 bit patterns

inline NativeDirectPiWindow native_direct_pi_window(float pi) {
  NativeDirectPiWindow w;
  uint32_t ub; std::memcpy(&ub, &pi, 4);
  const int eb = (int)((ub >> 23) & 255u);
  if ((ub >> 31) || eb < 1 || eb > 252) return w;
  volatile float one = 1.0f;   // the division happens at run time, in the current rounding mode
  const float y = one / pi;
  // y is RN(1 / pi) when no neighbour is closer: pi * y is exact in double (48-bit product), 1 - pi * y too.
  const double dp = (double)pi;
  const double e0 = std::fabs(1.0 - dp * (double)y);
  const double e1 = std::fabs(1.0 - dp * (double)std::nextafter(y, 0.0f));
  const double e2 = std::fabs(1.0 - dp * (double)std::nextafter(y, INFINITY));
  if (!(e0 < e1 && e0 < e2)) return w;
  const int B = eb - 127;
  const int L = B - 125 > -79 ? B - 125 : -79;
  std::memcpy(&w.rcp, &y, 4);
  w.lo = (uint32_t)(L + 127) << 23;
  w.hi = B > 0 ? 0x7f800000u : (uint32_t)(B + 254) << 23;
  return w;
}

}  // namespace wgrt
