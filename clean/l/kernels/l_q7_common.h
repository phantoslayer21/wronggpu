// SPDX-License-Identifier: MIT
// Model L (issue #89): the FP16 form's arithmetic shared by the q7 kernels (l_q7.hip, l_q7w.hip), in ONE place so
// that a native packed-FP16 form (S2b) can replace the helpers here without touching the kernels. The
// D3D12 counterpart is clean/dx12/hlsl/l_q7_common.hlsli (same functions, same bits).
// - h16: one rounding to FP16 (nearest even); add16 / mul16 / fma16: the FP16 steps, each the exact result rounded
//   once (an FP16 sum is not always exact in float32: error-free sums, the tail picking the side at a midpoint);
// - the FP8 matrix op is in l_q7_op8.h (S2a), apart from these FP16 steps (S2b);
// - out8 / out16: output stores (plain codes; a NaN's sign is masked by the checks).
// Include after wg_platform.h, with "#pragma clang fp contract(off)" in the including file.
#pragma once

typedef unsigned short u16;

namespace q7c {

WG_FN float h16(float x) { return (float)(_Float16)x; }
WG_FN u32 bits16(float x) { return (u32)__builtin_bit_cast(u16, (_Float16)x); }
WG_FN float from16(u32 b) { return (float)__builtin_bit_cast(_Float16, (u16)b); }
WG_FN float fp8v(u32 code) { return wg_e4m3_to_f32(code); }
WG_FN u8 e8(float x) { return (u8)wg_f32_to_e4m3(x); }
WG_FN bool finite(float x) { return __builtin_isfinite(x); }
WG_FN int mirror(int p, int n) {                                              // about the edge, edge not repeated
  p = p < 0 ? -p : p;
  return p >= n ? 2 * (n - 1) - p : p;
}
// Output stores: the plain codes (a NaN's sign is back-end dependent; the checks mask it, l_q7_common.hlsli)
WG_FN u8 out8(float x) { return (u8)wg_f32_to_e4m3(x); }
WG_FN u16 out16(float x) { return (u16)bits16(x); }

WG_FN void two_sum(float a, float b, float& s, float& e) {
  s = a + b;
  const float bb = s - a;
  e = (a - (s - bb)) + (b - bb);
}

// RN16 of the exact value z + t (z = fl(z + t), |t| at most half an ulp of z): where z is exactly an FP16 midpoint
// the tail's sign picks the side (no tail: ties to even).
WG_FN float rn16_pair(float z, float t) {
  if (!finite(z) || t == 0.0f) return h16(z);
  const u32 b = __builtin_bit_cast(u32, z);
  bool mid;
  if (__builtin_fabsf(z) >= 0x1p-14f) {
    mid = (b & 0x1fffu) == 0x1000u;
  } else {
    const float q = z * 0x1p25f;                                             // exact
    const int qi = (int)q;
    mid = (float)qi == q && (qi & 1);
  }
  if (!mid) return h16(z);
  const bool up = t > 0.0f;                                                   // towards +inf
  const u32 nb = (up == (z > 0.0f)) ? b + 1u : b - 1u;                        // next float32 on the tail's side
  return h16(__builtin_bit_cast(float, nb));
}

WG_FN float add16(float a, float b) {                                         // RN16(a + b), exact
  float s, e;
  two_sum(a, b, s, e);
  return rn16_pair(s, e);
}
WG_FN float mul16(float a, float b) { return h16(a * b); }                    // FP16 x FP16: exact in float32
WG_FN float fma16(float a, float b, float c) { return add16(a * b, c); }      // a * b exact: RN16 of the exact sum

WG_FN float gelu16(float h) {
  const float y = __builtin_fminf(__builtin_fmaxf(h, -2.0f), 2.0f);
  const float A = 0.412109375f, B = 0.0810546875f;                            // 0x3698, 0x2D30
  return mul16(h, add16(0.5f, mul16(y, add16(A, -mul16(B, __builtin_fabsf(y))))));
}

}  // namespace q7c
