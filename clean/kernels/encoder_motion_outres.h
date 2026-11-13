// SPDX-License-Identifier: MIT
// Input encoder with motion vectors at output resolution (spec/03-input-encoder.md draft 4, "Motion at output
// resolution"; create flag WG_CREATE_MOTION_OUTPUT_RES). Included by encoder.hip before encoder_main; the kernel that
// uses it is in encoder_mo.hip. Written for WrongGPU from spec 03.
//
// Per render pixel (x, y): no depth is read and there is no near / far pixel. One point read of the motion buffer at
//   T = (clamp(floor(RN32(RN32((x + 1/2) - jx) k_Mx)), 0, Mw - 1), likewise y with jy, k_My, Mh),  k_M = RN32(M / R),
// m = RN32(s RN32(b_mv + MV(T))) with s = RN32(mv_scale / RN32(M / D)) (WgEncParams sx, sy; the host evaluates s and
// k_M), and m_near = m_center = m_far = m. Everything else of the encoder runs unchanged with these three equal
// motions; all three motion surfaces receive RN16(m).
// A NaN texel coordinate (NaN jitter) reads texel 0 (our choice: the page does not say).
#ifndef WG_ENCODER_MOTION_OUTRES_H
#define WG_ENCODER_MOTION_OUTRES_H

#include "wg_motion_hi.h"

// Kernel argument (by value; 4-byte fields; must match EncMoParams in clean/runtime/runtime.cpp).
struct WgEncMoParams {
  int mw, mh;      // M: size of the motion data (the output size given at creation)
  int pitch;       // bytes per row of the motion buffer
  int wide;        // texel format: 0 = 2 x FP16 (4 bytes), 1 = 2 x f32 (8 bytes)
  float kmx, kmy;  // k_M = RN32(M / R) per axis
};
static_assert(sizeof(WgEncMoParams) == 24, "WgEncMoParams must match EncMoParams in clean/runtime/runtime.cpp");

// Template argument of encoder_main: EncMoOff = motion at render resolution with the depth neighbourhood (the kernel
// input_encoder), EncMoOn = motion at output resolution (Frame::motion is then the raw motion buffer).
struct EncMoOff {
  static constexpr bool on = false;
};
struct EncMoOn {
  static constexpr bool on = true;
  WgEncMoParams m;
};

WG_FN u32 enc_mo_index(int i, float j, float k, int n) {
  const float t = __builtin_floorf((((float)i + 0.5f) - j) * k);
  return (u32)(int)wg_fmin(wg_fmax(t, 0.0f), (float)(n - 1));  // NaN -> 0
}

// The motion of the render pixels (xs[0], y) and (xs[1], y): mn = mf = mc = m.
WG_FN void enc_mo_motions(const EncMoOn& mo, const void* motion, const WgEncParams& P, const u32 xs[2], int y,
                          float mn[2][2], float mf[2][2], float mc[2][2]) {
  const u32 row = enc_mo_index(y, P.jy, mo.m.kmy, mo.m.mh) * (u32)mo.m.pitch;
#pragma unroll
  for (int dx = 0; dx < 2; ++dx) {
    const u32 tx = enc_mo_index((int)xs[dx], P.jx, mo.m.kmx, mo.m.mw);
    float vx, vy;
    wg_mo_texel(motion, row + (mo.m.wide ? tx * 8u : tx * 4u), mo.m.wide, vx, vy);
    mc[dx][0] = P.sx * (P.bx + vx);
    mc[dx][1] = P.sy * (P.by + vy);
#pragma unroll
    for (int i = 0; i < 2; ++i) mn[dx][i] = mf[dx][i] = mc[dx][i];
  }
}

WG_FN void enc_mo_motions(const EncMoOff&, const void*, const WgEncParams&, const u32[2], int, float[2][2],
                          float[2][2], float[2][2]) {}

#endif  // WG_ENCODER_MOTION_OUTRES_H
