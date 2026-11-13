// SPDX-License-Identifier: MIT
// Exhaustive audit against original sqrt-then-divide using actual helper.
#include "../kernels/k_norm.h"
WG_KERNEL_N(k_norm_r16_original, 128)(unsigned short* out) {
  const u32 i = wg_block_id() * 128u + wg_lane_id();
  const h16 s16 = __builtin_bit_cast(h16, (unsigned short)i);
  const h16 r16 = s16 == (h16)0.0f ? (h16)0.0f :
      (h16)(1.0f / __builtin_sqrtf((float)s16));
  out[i] = __builtin_bit_cast(unsigned short, r16);
}
template <bool SELECT>
WG_FN void k_norm_r16_cand(unsigned short* out, u32* marker) {
  const u32 i = wg_block_id() * 128u + wg_lane_id();
  out[i] = __builtin_bit_cast(unsigned short,
      wgk_norm_reciprocal<SELECT>(__builtin_bit_cast(h16, (unsigned short)i)));
  if (i == 0) {
#if defined(__AMDGCN__) && defined(__gfx1201__)
    marker[0] = 0x1201u;
#else
    marker[0] = 0u;
#endif
  }
}
// the branch form (k_block_wide.hip) and the select form (k_block.hip)
WG_KERNEL_N(k_norm_r16_candidate, 128)(unsigned short* out, u32* marker) { k_norm_r16_cand<false>(out, marker); }
WG_KERNEL_N(k_norm_r16_candidate_select, 128)(unsigned short* out, u32* marker) { k_norm_r16_cand<true>(out, marker); }
