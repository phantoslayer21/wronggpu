// SPDX-License-Identifier: MIT
// GPU-only comparison of the ACTUAL production helpers. Native and candidate
// results are written by separate kernels before an integer-only comparison.
#include "../kernels/k_io.hip"

WG_FN void k_probe_axis_args(float f, float* a, float* b) {
  // Spec43/ko_axis operations, including every separate FP32 rounding.
  const float f2 = f*f, f3 = f2*f;
  const float c0 = f2 - 0.5f*(f+f3);
  const float c1 = (1.5f*f3 - 2.5f*f2) + 1.0f;
  const float c3 = 0.5f*(f3-f2);
  const float c2 = ((1.0f-c0)-c1)-c3;
  *a = c1;
  *b = c1+c2;
}

WG_KERNEL_N(k_probe_native_rcp, 128)(u32* dst, u32 first, u32 count) {
  const u32 lane = wg_block_id()*128u + wg_lane_id();
  for (u32 i=lane; i<count; i+=128u*4096u) {
    const float x = __builtin_bit_cast(float, first+i);
    dst[i] = __builtin_bit_cast(u32, 1.0f/x);
  }
}
WG_KERNEL_N(k_probe_candidate_rcp, 128)(u32* dst, u32 first, u32 count) {
  const u32 lane = wg_block_id()*128u + wg_lane_id();
  for (u32 i=lane; i<count; i+=128u*4096u) {
    const float x = __builtin_bit_cast(float, first+i);
    dst[i] = __builtin_bit_cast(u32, ko_rcp_refined(x));
  }
}
WG_KERNEL_N(k_probe_candidate_bounded, 128)(u32* dst, u32 first, u32 count) {
  const u32 lane = wg_block_id()*128u + wg_lane_id();
  for (u32 i=lane; i<count; i+=128u*4096u) {
    const float x = __builtin_bit_cast(float, first+i);
    dst[i] = __builtin_bit_cast(u32, ko_rcp_output_bounded(x));
  }
}
WG_KERNEL_N(k_probe_native_axis, 128)(u32* dst, u32 first, u32 count) {
  const u32 lane = wg_block_id()*128u + wg_lane_id();
  for (u32 i=lane; i<count; i+=128u*4096u) {
    float a,b;
    k_probe_axis_args(__builtin_bit_cast(float, first+i), &a, &b);
    dst[i] = __builtin_bit_cast(u32, a/b);
  }
}
WG_KERNEL_N(k_probe_candidate_axis, 128)(u32* dst, u32 first, u32 count) {
  const u32 lane = wg_block_id()*128u + wg_lane_id();
  for (u32 i=lane; i<count; i+=128u*4096u) {
    float a,b;
    k_probe_axis_args(__builtin_bit_cast(float, first+i), &a, &b);
    dst[i] = __builtin_bit_cast(u32, ko_div_axis(a,b));
  }
}
WG_KERNEL_N(k_probe_compare, 128)(const u32* a, const u32* b, u64* errors, u32* lowest, u32 first, u32 count) {
  const u32 lane = wg_block_id()*128u + wg_lane_id();
  u32 bad=0, low=0xffffffffu;
  for (u32 i=lane; i<count; i+=128u*4096u) {
    if (a[i] != b[i]) {
      ++bad;
      low = low < first+i ? low : first+i;
    }
  }
  if (bad) {
    __atomic_fetch_add(errors, (u64)bad, __ATOMIC_RELAXED);
    __atomic_fetch_min(lowest, low, __ATOMIC_RELAXED);
  }
}
