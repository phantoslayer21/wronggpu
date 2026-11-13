// SPDX-License-Identifier: MIT
// Reconstruction (and true fusion with the output filter) for motion vectors at output resolution (spec/06-
// reconstruction-and-output.md section 3, "Motion at output resolution"): the kernels below, in their own code
// objects (post_mo_w64*.co) so that the post code objects stay the files they were. The whole of post.hip is included
// with its kernels made internal functions (the compiler drops them); gather, history sample, blend, outputs and the
// output filter are the ones of that file, only the motion differs (post_motion_outres.h: one point read of the
// motion buffer per D pixel, one history sample).
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
#include "post.hip"
#pragma pop_macro("WG_EMU_REGISTER")
#pragma pop_macro("WG_KERNEL_NW")
#pragma pop_macro("WG_KERNEL_N")

// Stage "post" + "filter" in one kernel: post_tf_p12_diet_vsit / post_tf_f32_p12_diet_vsit (the default true-fusion
// kernels of the runtime) with the motion of post_motion_outres.h. Arguments as those kernels, except: one motion
// surface pointer m_near (render size; staged like the three surfaces of the original kernel but not used), then
// `motion` (the raw motion buffer of wg_motion_hi.h), and mo after c2 (before the output-filter weights of the
// "original" build). Same grid, workgroup size and conditions (tf_supported).
#define WG_TF_MO_KERNEL(name, H)                                                                                    \
  WG_KERNEL_NW(name, 64 * 8, WG_TF_WAVES)(const v4f* color, const u32* A, const u32* B, const v4h* C,              \
                                          const v4h* hist_prev, const u32* m_near, const void* motion,            \
                                          const float* w, v4h* hist_out, u32* reduced, void* out, WgPostParams p, \
                                          const float* expo, int ow, int oh, float dec_scale, float pi,           \
                                          int pitch_bytes, int cpairs, int k1, int c2, WgMoParams mo WG_LZ_KARGS) { \
    WG_LDS TfShared<8, tf_cap<8>()> sh;                                                                             \
    (void)B;                                                                                                        \
    post_tf_body<8, H, tf_cap<8>(), true, true, true, true, true, MoOn>(                                            \
        color, A, C, hist_prev, m_near, m_near, m_near, w, hist_out, reduced, out, p, expo, ow, oh, dec_scale, pi, \
        pitch_bytes, cpairs, k1, c2 WG_LZ_KW, sh, MoOn{motion, mo});                                                \
  }
WG_TF_MO_KERNEL(post_tf_p12_mo, true)       // interop output copy: RGBA16F rows
WG_TF_MO_KERNEL(post_tf_f32_p12_mo, false)  // own output: 3 x f32

// Stage "post" alone (frames the true-fusion kernel does not take; the unbanded output filter follows):
// post_recon2_r_p12 with the motion of post_motion_outres.h. Arguments as post_recon2_r_p12 with the three motion
// surfaces replaced by `motion`, and mo after expo. Same grid; needs recon2_supported.
WG_KERNEL_N(post_recon2_r_p12_mo, kPostThreads)(const v4f* color, const u32* A, const u32* B, const v4h* C,
                                                const v4h* hist_prev, const void* motion, const float* w,
                                                v4h* hist_out, u32* reduced, _Float16* display, WgPostParams p,
                                                const float* expo, WgMoParams mo) {
  WG_LDS R2Shared sh;
  post_recon2_body<true, true, false, false, false, true, (WG_POST_DIET != 0), MoOn>(
      color, A, B, C, hist_prev, nullptr, nullptr, nullptr, w, hist_out, reduced, display, p, expo, sh, 0.0f, 0, 0,
      MoOn{motion, mo});
}

// Test kernel: the motion texel index pair and the previous position of every D pixel, as the kernels above compute
// them (mo_position): out[4 (y dw + x) ...] = tx, ty, bits of qx, bits of qy. Grid ceil(dw dh / 256) workgroups of 256.
WG_KERNEL_N(post_mo_probe, 256)(const void* motion, u32* out, int dw, int dh, WgMoParams mo) {
  const u32 i = wg_block_id() * 256u + wg_lane_id();
  if (i >= (u32)dw * (u32)dh) return;
  const int y = (int)(i / (u32)dw), x = (int)(i - (u32)y * (u32)dw);
  const MoOn on = {motion, mo};
  MoQ mq;
  mo_fetch(on, x, y, dw, dh, mq);
  float qx, qy;
  mo_position(on, mq, 0, x, y, qx, qy);
  out[4u * i] = (u32)mo_texel(x, dw, mo.idw, mo.mw);
  out[4u * i + 1u] = (u32)mo_texel(y, dh, mo.idh, mo.mh);
  out[4u * i + 2u] = __builtin_bit_cast(u32, qx);
  out[4u * i + 3u] = __builtin_bit_cast(u32, qy);
}

#if defined(WG_EMU)
WG_EMU_REGISTER(post_tf_p12_mo)
WG_EMU_REGISTER(post_tf_f32_p12_mo)
WG_EMU_REGISTER(post_recon2_r_p12_mo)
WG_EMU_REGISTER(post_mo_probe)
#endif
