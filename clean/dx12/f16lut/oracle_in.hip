// SPDX-License-Identifier: MIT
// HIP oracle of the encoder colour values the direct encoder's table replaces (clean/dx12/hlsl/wg_f16lut_in.hlsli), as the
// HIP executor computes them from an FP16 colour channel: input_convert / color_divide ((float)c / pi, IEEE division,
// only when pi != 1; frame.hip), then encoder.hip's U.e (expo[0] == 0 -> 1) and pixel_values' channel loop (lines
// 289-296), copied.
#include "wg_platform.h"
#include "wg_mem.h"
#pragma clang fp contract(off)

constexpr float kG = 0.108370798438993f;

// prm: (e0 bits, pi bits) per set; grid (256, nsets): out[set * 131072 + 2 r] = v, + 1 = gc; div[set * 65536 + r] = c / pi
WG_KERNEL_N(f16lut_in_oracle, 256)(const u32* prm, u32* out, u32* div) {
  const u32 r = wg_block_id() * 256u + wg_lane_id();
  const u32 set = __builtin_amdgcn_workgroup_id_y();
  const float e0 = __builtin_bit_cast(float, prm[2 * set]), pi = __builtin_bit_cast(float, prm[2 * set + 1]);
  float c = (float)__builtin_bit_cast(_Float16, (unsigned short)r);
  if (pi != 1.0f) c = c / pi;
  const float e = e0 == 0.0f ? 1.0f : e0;
  const float ec = e * c;
  const float rr = kG * wg_log2(wg_fmax(__builtin_fmaf(100.0f, wg_fmin(ec, 32752.0f), 1.0f), 1e-4f));
  float g = rr;
  if (ec > 32752.0f) g = kG * wg_log2(__builtin_fmaf(100.0f, ec, 1.0f));
  out[set * 131072u + 2u * r] = __builtin_bit_cast(u32, ec != ec ? 0.0f : rr);
  out[set * 131072u + 2u * r + 1u] = __builtin_bit_cast(u32, ec != ec ? 0.0f : g);
  div[set * 65536u + r] = __builtin_bit_cast(u32, c);
}
