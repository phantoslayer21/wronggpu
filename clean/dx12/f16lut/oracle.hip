// SPDX-License-Identifier: MIT
// HIP oracle of the output value the direct post kernel's table replaces (clean/dx12/hlsl/wg_f16lut.hlsli), as the HIP
// executor computes it from the lz5 result r (FP16): post.hip's vertical pass of post_tf_f32_p12_diet_vsit (qc =
// clampf(r, 1e-4, 3), dec_m1_poly below kDecPolyMax else wg_exp2(qc kDecK) - 1, the f32 row v * dec_scale), then
// scale_rgb (frame.hip: x * pi, launched only when pi != 1) and, for the RGBA16F output copy, framebuffer_half (the
// kH form: framebuffer_half((v * dec_scale) * pi)). The definitions below are post.hip's (lines 150-190), copied.
#include "wg_platform.h"
#pragma clang fp contract(off)

constexpr float kDecK = 3.32192809488736235f / 0.36f;
constexpr float kDecLn = 2.30258509299404568f / 0.36f;
WG_FN float clampf(float v, float lo, float hi) { return wg_fmin(wg_fmax(v, lo), hi); }
WG_FN _Float16 framebuffer_half(float v) {
  const float finite = v > 65504.0f ? 65504.0f : v < -65504.0f ? -65504.0f : v;
  return (_Float16)finite;
}
WG_FN float dec_m1_poly(float qc) {
  const float y = qc * kDecLn;
  float t = __builtin_fmaf(y, 1.0f / 5040.0f, 1.0f / 720.0f);
  t = __builtin_fmaf(t, y, 1.0f / 120.0f);
  t = __builtin_fmaf(t, y, 1.0f / 24.0f);
  t = __builtin_fmaf(t, y, 1.0f / 6.0f);
  t = __builtin_fmaf(t, y, 0.5f);
  t = __builtin_fmaf(t, y, 1.0f);
  return t * y;
}
constexpr float kDecPolyMax = 0.35f / kDecLn;

// prm: (dec_scale bits, pi bits, rne16) per set; grid (256, nsets): out[set * 65536 + r]; out2: the f32 row v * dec_scale
WG_KERNEL_N(f16lut_oracle, 256)(const u32* prm, u32* out, u32* row) {
  const u32 r = wg_block_id() * 256u + wg_lane_id();
  const u32 set = __builtin_amdgcn_workgroup_id_y();
  const float dec_scale = __builtin_bit_cast(float, prm[3 * set]), pi = __builtin_bit_cast(float, prm[3 * set + 1]);
  const u32 rne16 = prm[3 * set + 2];
  const float qc = clampf((float)__builtin_bit_cast(_Float16, (unsigned short)r), 1e-4f, 3.0f);
  const float v = qc < kDecPolyMax ? dec_m1_poly(qc) : wg_exp2(qc * kDecK) - 1.0f;
  const float x0 = v * dec_scale;
  float x = x0;
  if (rne16) x = (float)framebuffer_half(x0 * pi);
  else if (pi != 1.0f) x = x0 * pi;
  out[set * 65536u + r] = __builtin_bit_cast(u32, x);
  row[set * 65536u + r] = __builtin_bit_cast(u32, x0);
}
