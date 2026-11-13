// SPDX-License-Identifier: MIT
// Input encoder for motion vectors at output resolution (spec/03-input-encoder.md draft 4, "Motion at output
// resolution"): the kernel input_encoder_mo, in its own code object (encoder_mo.co) so that encoder.co stays the file
// it was. The whole of encoder.hip is included with its kernel made an internal function (the compiler drops it); the
// per-pixel code, the dithered quantization and the projection are the ones of that file, only the motions differ
// (encoder_motion_outres.h).
#include "wg_platform.h"

#pragma push_macro("WG_KERNEL_N")
#pragma push_macro("WG_KERNEL_NW")
#pragma push_macro("WG_EMU_REGISTER")
#undef WG_KERNEL_N
#undef WG_KERNEL_NW
#undef WG_EMU_REGISTER
#if defined(__AMDGCN__)
#define WG_KERNEL_N(name, threads) static __attribute__((device)) __attribute__((unused)) void name
#else
#define WG_KERNEL_N(name, threads) static __attribute__((unused)) void name
#endif
#define WG_KERNEL_NW(name, threads, waves) WG_KERNEL_N(name, threads)
#define WG_EMU_REGISTER(name)
#include "encoder.hip"
#pragma pop_macro("WG_EMU_REGISTER")
#pragma pop_macro("WG_KERNEL_NW")
#pragma pop_macro("WG_KERNEL_N")

// Stage enc0 with motion at output resolution. Arguments as input_encoder without depth; `motion` is the raw motion
// buffer of wg_motion_hi.h (M texels, rows of mo.pitch bytes), P.sx / P.sy = s = RN32(mv_scale / RN32(M / D)),
// P.bx / P.by = b_mv; P.depth_inverted is not used. Grid and workgroup size as input_encoder.
#if WG_ENC_WAVES
WG_KERNEL_NW(input_encoder_mo, 256, WG_ENC_WAVES)
#else
WG_KERNEL_N(input_encoder_mo, 256)
#endif
(const float* color, const void* motion, const u32* hist, const u32* latent, const u32* mhist, u8* features,
 u32* m_near, u32* m_far, u32* m_center, const float* expo, const u8* w, WgEncParams P, WgEncMoParams mo) {
  const Frame f = {color, nullptr, (const float*)motion, hist, latent, mhist};
  encoder_main<EncMoOn>(f, features, m_near, m_far, m_center, expo, w, P, EncMoOn{mo});
}

#if defined(WG_EMU)
WG_EMU_REGISTER(input_encoder_mo)
#endif
