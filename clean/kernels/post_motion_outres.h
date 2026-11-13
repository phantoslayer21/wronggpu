// SPDX-License-Identifier: MIT
// Reconstruction with motion vectors at output resolution (spec/06-reconstruction-and-output.md section 3, "Motion at
// output resolution"; create flag WG_CREATE_MOTION_OUTPUT_RES). Included by post.hip after post_core_s; the kernels
// that use it are in post_mo.hip. Written for WrongGPU from spec 06 (and spec 03 draft 4 for s and b_mv).
//
// Per D pixel (x, y), instead of the three motion candidates read from the render-size motion surfaces:
//   T = (clamp(trunc(RN32(RN32((x + 1/2) / dw) Mw)), 0, Mw - 1), likewise y with dh, Mh)   one point read
//   v = b_mv + MV(T);  q = (fma(sx, vx, x + 1/2), fma(sy, vy, y + 1/2))                     previous position
// There is only this one motion: the scores still give w1 and w2 (post_select: the two highest of the three averaged
// scores), section 4 is evaluated once (S: the history sample at q, or the fallback if q is invalid) and
// H = fma(w1, S, w2 S), the second term counting as 0 when w2 = 0. Step 5 on is post_blend, unchanged.
// A NaN position follows spec 06 draft 4 section 4: it counts as valid, its sample is NaN, and the NaN guard of step 5
// replaces both results by C_lin. Here: the blend is run with 1 - beta' = 0, which gives C_lin exactly.
// (The render-resolution kernels of this file's includer still treat a NaN position as invalid.)
#ifndef WG_POST_MOTION_OUTRES_H
#define WG_POST_MOTION_OUTRES_H

#include "wg_motion_hi.h"

// Kernel argument (by value; 4-byte fields; must match MoParams in clean/runtime/runtime.cpp and clean/runner/post.py).
struct WgMoParams {
  int mw, mh;      // M: size of the motion data (the output size given at creation)
  int pitch;       // bytes per row of the motion buffer
  int wide;        // texel format: 0 = 2 x FP16 (4 bytes), 1 = 2 x f32 (8 bytes)
  float sx, sy;    // s = RN32(mv_scale / RN32(M / D)) per axis (the host evaluates it)
  float bx, by;    // b_mv
  float idw, idh;  // RN32(1 / dw), RN32(1 / dh): the correctly rounded reciprocals (the host divides), for mo_center
};
static_assert(sizeof(WgMoParams) == 40, "WgMoParams must match MoParams in clean/runtime/runtime.cpp");

namespace {

// Template argument of the reconstruction bodies: MoOff = the three motion surfaces (every kernel of post.hip),
// MoOn = motion at output resolution.
struct MoOff {
  static constexpr bool on = false;
};
struct MoOn {
  static constexpr bool on = true;
  const void* motion;
  WgMoParams m;
};

// RN32((i + 1/2) / n) for a pixel index i and a size n: the FP32 division of the page, exactly, from the correctly
// rounded reciprocal rn = RN32(1 / n) (a kernel argument: the host's division is correctly rounded, the GPU's plain
// FP32 division and reciprocal need not be). Markstein's correction, as div_rn of decoder.hip: with q0 = RN(a rn),
// RN(q0 + RN(a - n q0) rn) = RN(a / n). Checked against the division for every pair i < n <= 12000 (72 million
// quotients, clean/tests/test_motion_hi_reference.py checks a subset; the test kernel post_mo_probe on the GPU).
WG_FN float mo_center(int i, int n, float rn) {
  const float a = (float)i + 0.5f, q0 = a * rn;
  return __builtin_fmaf(__builtin_fmaf(-(float)n, q0, a), rn, q0);
}

// Motion texel index along one axis: clamp(trunc(RN32(RN32((i + 1/2) / n) m)), 0, m - 1).
WG_FN int mo_texel(int i, int n, float rn, int m) {
  const float t = mo_center(i, n, rn) * (float)m;
  const int k = (int)t;  // t >= 0: truncation
  return k > m - 1 ? m - 1 : (k < 0 ? 0 : k);
}

// The raw motion texels of the D pixels (x, y0) and (x, y0 + 1) (the two rows a lane reconstructs), as loaded. The
// kernels fetch them at the start of a step, well before the values are needed (mo_position): the loads then overlap
// the staging and the kernel gather instead of standing between the gather and the history taps.
struct MoQ {
  u32 w[2][2];
};
WG_FN void mo_fetch(const MoOn& mo, int x, int y0, int dw, int dh, MoQ& q) {
  const u32 tx = (u32)mo_texel(x, dw, mo.m.idw, mo.m.mw);
  const u32 col = mo.m.wide ? tx * 8u : tx * 4u;
#pragma unroll
  for (int k = 0; k < 2; ++k) {
    const u32 ty = (u32)mo_texel(y0 + k, dh, mo.m.idh, mo.m.mh);  // (a row past the image reads the last texel row)
    wg_mo_load(mo.motion, (ty & 0xffffffu) * ((u32)mo.m.pitch & 0xffffffu) + col, q.w[k][0], q.w[k][1]);
  }
}
WG_FN void mo_fetch(const MoOff&, int, int, int, int, MoQ&) {}

// The previous position q of D pixel (x, y) from its fetched texel (row k of q: y = y0 + k).
WG_FN void mo_position(const MoOn& mo, const MoQ& q, int k, int x, int y, float& qx, float& qy) {
  float vx, vy;
  wg_mo_value(q.w[k][0], q.w[k][1], mo.m.wide, vx, vy);
  qx = __builtin_fmaf(mo.m.sx, mo.m.bx + vx, (float)x + 0.5f);
  qy = __builtin_fmaf(mo.m.sy, mo.m.by + vy, (float)y + 0.5f);
}

// Spec 06 steps 3 to 5 for one D pixel with the single motion (the counterpart of post_core_s): the linear new
// history H^ in hh and the linear display value O^ in oh. (fx, fy): the clamped render pixel of the fallback.
// History: the 8-byte format read as spans (history_cr_span<false>, as the default kernels; needs dw >= 4).
template <bool kTrim>
WG_FN void mo_core(const MoOn& mo, const MoQ& mq, int k, const Gather& g, int x, int y, int fx, int fy,
                   const WgPostParams& p, float s, const v4f* color, const void* hist_prev, float hh[3], float oh[3]) {
  float qx, qy;
  mo_position(mo, mq, k, x, y, qx, qy);
  PostSel sl = post_select(g);
  const bool live = !p.reset;
  const bool nanq = live & ((qx != qx) | (qy != qy));
  const bool valid = live & (qx >= 0.0f) & (qx <= (float)p.dw) & (qy >= 0.0f) & (qy <= (float)p.dh);  // not NaN
  float h[3];
  if (valid) {
    history_cr_span<false>(hist_prev, p.dw, p.dh, qx, qy, h);
  } else {  // the fallback E(e_prev color) of the render pixel (floor px, floor py), as post_core_s
    const v4f c = *(const v4f*)((const char*)color + (mul24((u32)fy, (u32)p.rw * 16u) + (u32)fx * 16u));
    h[0] = enc_e(p.e_prev * c[0]);
    h[1] = enc_e(p.e_prev * c[1]);
    h[2] = enc_e(p.e_prev * c[2]);
  }
  float H[3];
#pragma unroll
  for (int i = 0; i < 3; ++i) H[i] = __builtin_fmaf(sl.w1, h[i], sl.w2 != 0.0f ? sl.w2 * h[i] : 0.0f);
  if (nanq) {  // 1 - beta' = 0: post_blend returns C_lin for both (H is the finite fallback term, times 0)
    sl.cbh = 0.0f;
    sl.cbo = 0.0f;
  }
  post_blend<kTrim>(g, sl, H, p, s, hh, oh);
}

// MoOff never calls it (the call sits in a branch on MoT::on); this overload only lets that branch compile.
template <bool kTrim, class... A>
WG_FN void mo_core(const MoOff&, A&&...) {}

}  // namespace

#endif  // WG_POST_MOTION_OUTRES_H
