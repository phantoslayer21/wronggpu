// SPDX-License-Identifier: MIT
// Model K input stage (spec 42) and output stage (spec 43) for gfx1201: one lane per position, FP32 / FP16 steps in
// the spec's order (contraction off, fma explicit, k_fp.h). NumPy references: clean/k/reference/instage.py and
// outstage.py; bilinear sampling and the other choices noted there (QUESTIONS.md K-Q8).
//
// k_out_color   one lane per render pixel: the stored colour tap (spec 43 4.2) and the filter-tap value (5.3), computed
//               once instead of for every pixel that reads them (~16 and ~12 per pixel).
// k_out_pixel   one lane per pixel of the ceil16(W) x ceil16(H) grid (workgroups of 64 x 2 pixels): network channels, motion, histories, current
//               colour, shaped filter over 12 taps x 4 sub-samples, blend; writes the luma history (2 x 2 per pixel),
//               the colour history, and the game output for x < W, y < H.
// k_out_hidden  one lane per base position: the hidden history (spec 43 6.8).
// k_in_color    one lane per render pixel: the colour-neighbourhood values of spec 42 section 6.2 (Y, Co, Cg, phi(Y),
//               the HDR weight) and the own-sample luma channel, computed once instead of for every display position
//               that reads them.
// k_in_base     one lane per base position: its 4 display positions (16 fine positions): the 16 network inputs.
// Cross-lane handoffs in this file use LDS only. Global inputs/weights are read-only
// during a launch; global outputs are consumed by later stream-ordered kernels.
// LDS-scoped barriers preserve those handoffs without invalidating global caches.
#include "k_fp.h"

#pragma clang fp contract(off)

enum { KIO_HDR = 1, KIO_MVLO = 2, KIO_INV = 4, KIO_RESET = 8, KIO_C16 = 16, KIO_M16 = 32 };
// KIO_C16: colour is RGBA16F texels (row pitch cpitch texels); KIO_M16: motion is FP16 texels of mch channels (row
// pitch mpitch texels) holding the game's raw values r, the motion is m = ms r + mo (FP32 multiply, then add: the
// interop's conversion, clean/kernels/frame.hip input_convert). Both read the interop's input copies (or its compact
// input pass) directly; the values equal those of the FP32 inputs the conversion would have written.
// Game-output formats of k_out_pixel (KOutParams.omode): RGBA float32 [H][opitch] (the spec's output, tests); RGBA16F
// rows of opitch texels with RGB rounded to nearest even and alpha 1 (the interop's output copy); dense RGB float32
// [H][opitch] (the interop's output pass, which writes alpha 1 itself).
enum { KIO_OUT_RGBA32F = 0, KIO_OUT_RGBA16F = 1, KIO_OUT_RGB32F = 2 };

// Frame inputs shared by both stages (must match KFrameIn in clean/k/runner/k_io.py).
struct KFrameIn {
  const float* color;   // RGBA float32 texels, row pitch cpitch
  const float* motion;  // 2 x float32, row pitch mpitch
  const float* depth;   // float32, row pitch zpitch ([MVlo] only)
  const float* xs;      // null, or the exposure scalars on the GPU (k_expo_scalars): kappa, rho, lam, RN(1/kappa); they
                        // replace kappa, rho, lam below and KOutUni.kinv (the runtime's GPU-resident exposure)
  int cpitch, mpitch, zpitch;
  int cbx, cby, mbx, mby, zbx, zby;  // subrect bases
  int W, H, rw, rh, Bw, Bh;          // output, render, base sizes
  int flags;                         // KIO_*
  float jx, jy, sx, sy, mox, moy;    // jitter, motion scale, motion offset
  float kappa;                       // colour scale (spec 42 4 / spec 43 3.7), computed on the host
  float rho;                         // history scale (spec 42 rho / spec 43 rho_e)
  float lam;                         // spec 42 luma scale lambda
  float s;                           // spec 44's base-size factor (draft 3: 1/2)
  int mch;                           // KIO_M16: FP16 channels per motion texel (2 or 4)
  float rmsx, rmsy, rmox, rmoy;      // KIO_M16: ms, mo of the raw motion
  int pad_;
};
static_assert(sizeof(KFrameIn) == 160, "KFrameIn must match clean/k/runner/k_io.py");

// Frame-uniform FP32 values of the output stage (spec 43 section 1: host FP32 values), computed on the host by
// clean/k/runner/k_io.py out_uniforms() with the same FP32 operations, each correctly rounded, so the per-pixel kernel
// does no division on them.
struct KOutUni {
  float invW, invH;      // RN(1/W), RN(1/H)
  float rhox, rhoy;      // rw invW, rh invH
  float rho2x, rho2y;    // 0.5 rho
  float jmx, jmy;        // jx - 0.5, jy - 0.5
  float spx, spy;        // motion scale: sx / rhox ([MVlo]) or sx
  float invW16, invH16;  // RN(1/W16), RN(1/H16) (colour-history bilinear)
  float kinv;            // RN(1/kappa) (6.7)
  unsigned short kux, kvy;  // FP16 bits of h(-8 / rho2x), h(-8 / rho2y) (6.1)
};
static_assert(sizeof(KOutUni) == 56, "KOutUni must match clean/k/runner/k_io.py");

struct KOutParams {
  KFrameIn f;
  const h16* N;        // [Bh][Bw][40]
  const h16* L;        // [2 H16][2 W16]
  const h16* Hc;       // [H16][W16][4]
  const h16* table;    // [2048][8], the frame's set (spec 43 section 7)
  const h16* exp2tab;  // [65536]: h(2^E) for every binary16 bit pattern E
  float* out;          // game output, row pitch opitch (texels), format omode (KIO_OUT_*)
  h16* Lout;
  h16* Hcout;
  h16* Hhout;          // [Bh][Bw][4]
  int opitch;
  int omode;           // KIO_OUT_RGBA32F (default), KIO_OUT_RGBA16F (alpha 1), KIO_OUT_RGB32F (runtime interop)
  h16* ctap;           // k_out_color -> k_out_pixel: stored tap (Y, Co, Cg, A) of spec 43 4.2 per render pixel [rh][rw][4]
  h16* ctapT;          // the same: filter-tap value T of spec 43 5.3 per render pixel [rh][rw]
  KOutUni u;           // frame-uniform values (host)
};
static_assert(sizeof(KOutParams) == 312, "KOutParams must match clean/k/runner/k_io.py");

struct KInParams {
  KFrameIn f;
  const h16* Hc;  // previous colour history [H16][W16][4]
  const h16* Hh;  // previous hidden history [Bh][Bw][4]
  h16* x0;        // network input [Bh][Bw][16]
  float* cpre;    // k_in_color -> k_in_base: (Y, Co, Cg, phi(Y)) per render pixel [rh][rw][4]
  float* cpre2;   // the same: (HDR weight max(1 / (Y + 1), 1e-4), own-sample channel S(T(lambda L(Yc)))) [rh][rw][2]
};
static_assert(sizeof(KInParams) == 200, "KInParams must match clean/k/runner/k_io.py");

WG_FN int kio_ceil16(int n) { return (n + 15) / 16 * 16; }

// The kernels' view of their parameters: with KFrameIn.xs the exposure scalars come from the GPU (k_expo_scalars,
// the same FP32 values as the host's k_io.scalars / out_uniforms give for the same exposures), else from the host.
WG_FN KFrameIn kio_frame(const KFrameIn& f) {
  KFrameIn F = f;
  if (f.xs) {
    F.kappa = f.xs[0];
    F.rho = f.xs[1];
    F.lam = f.xs[2];
  }
  return F;
}
WG_FN KInParams ki_params(const KInParams& p) {
  KInParams P = p;
  P.f = kio_frame(p.f);
  return P;
}
WG_FN KOutParams ko_params(const KOutParams& p) {
  KOutParams P = p;
  P.f = kio_frame(p.f);
  if (p.f.xs) P.u.kinv = p.f.xs[3];
  return P;
}

// &b[i] with the byte offset computed in 32 bits: the compiler can then address with the uniform base in scalar
// registers and a 32-bit vector offset (no 64-bit address arithmetic per access). Every buffer here is below 4 GB.
template <class T>
WG_FN T* kio_ptr(T* b, u32 i) {
  return (T*)((char*)b + i * (u32)sizeof(T));
}
template <class T>
WG_FN const T* kio_ptr(const T* b, u32 i) {
  return (const T*)((const char*)b + i * (u32)sizeof(T));
}

// spec 42 section 3: index = b + min(i + d, n - 1) if i + d > 0, else b
WG_FN int kio_tap(int i, int d, int b, int n) { return i + d > 0 ? b + ki_min(i + d, n - 1) : b; }
// [MVlo] motion fetch: i' = b + i + d; index = min(i' + b, b + n - 1) if i' > 0, else b
WG_FN int kio_mtap(int i, int d, int b, int n) {
  const int ip = b + i + d;
  return ip > 0 ? ki_min(ip + b, b + n - 1) : b;
}

// colour texel (x, y) of the texture (subrect base included by the caller), FP32 or RGBA16F (KIO_C16, exact widening)
WG_FN v4f kio_texel(const KFrameIn& F, int x, int y) {
  const u32 i = ((u32)y * F.cpitch + x) * 4u;
  if (F.flags & KIO_C16) {
    const v4h h = *(const v4h*)kio_ptr((const h16*)F.color, i);
    const v4f r = {(float)h[0], (float)h[1], (float)h[2], (float)h[3]};
    return r;
  }
  return *(const v4f*)kio_ptr(F.color, i);
}
WG_FN v4f kio_color(const KFrameIn& F, int ix, int iy) {
  return kio_texel(F, kio_tap(ix, 0, F.cbx, F.rw), kio_tap(iy, 0, F.cby, F.rh));
}
// motion texel (x, y) of the texture (subrect base included by the caller): FP32 pairs, or KIO_M16's ms r + mo
WG_FN void kio_motion(const KFrameIn& F, int x, int y, float* mvx, float* mvy) {
  if (F.flags & KIO_M16) {
    const h16* r = kio_ptr((const h16*)F.motion, ((u32)y * F.mpitch + x) * (u32)F.mch);
    *mvx = F.rmsx * (float)r[0] + F.rmox;
    *mvy = F.rmsy * (float)r[1] + F.rmoy;
    return;
  }
  const float* m = kio_ptr(F.motion, ((u32)y * F.mpitch + x) * 2u);
  *mvx = m[0];
  *mvy = m[1];
}

WG_FN float kio_depth(const KFrameIn& F, int ix, int iy, int dx, int dy) {
  return *kio_ptr(F.depth, (u32)kio_tap(iy, dy, F.zby, F.rh) * F.zpitch + kio_tap(ix, dx, F.zbx, F.rw));
}

// spec 42 5.2 [MVlo]: offset of the nearest-depth motion pixel
WG_FN void kio_nearest(const KFrameIn& F, int ix, int iy, int* odx, int* ody) {
  const float c = kio_depth(F, ix, iy, 0, 0), a = kio_depth(F, ix, iy, -1, -1), b = kio_depth(F, ix, iy, 1, -1);
  const float d = kio_depth(F, ix, iy, -1, 1), e = kio_depth(F, ix, iy, 1, 1);
  const float k = kf_bits(0x3A83126Fu);
  const bool inv = (F.flags & KIO_INV) != 0;
  bool edge;
  if (inv)
    edge = (c < 0.5f * (a + e) - k * c) || (c < 0.5f * (b + d) - k * c);
  else
    edge = (c > kf_fma(c, k, 0.5f * (a + e))) || (c > kf_fma(c, k, 0.5f * (b + d)));
  int bx = 0, by = 0;
  if (edge) {
    float best = c;
    const float v[4] = {a, b, d, e};
    const int ox[4] = {-1, 1, -1, 1}, oy[4] = {-1, -1, 1, 1};
#pragma unroll
    for (int q = 0; q < 4; ++q) {
      const bool nearer = inv ? v[q] > best : v[q] < best;
      if (nearer) {
        best = v[q];
        bx = ox[q];
        by = oy[q];
      }
    }
  }
  *odx = bx;
  *ody = by;
}

// spec 42 decode factor f (t f is linear)
#define kKioDecodeNear kf_bits(0x3F7FBE88u)  // below it: 1 / (1 - t)
#define kKioEncodeNear 1000.0f               // below it: 1 / (Y + 1)
WG_FN float kio_decode(float t) {
  if (t < kKioDecodeNear) return 1.0f / (1.0f - t);
  const float e = ((t + kf_bits(0x3C4FAE29u)) * 9.0f) * kf_bits(0x3FB8AA3Bu);
  return (kf_exp2(e) - 1.0f) / (9.0f * t);
}
// spec 43 encode e(Y); stored = Y e(Y)
WG_FN float kio_encode(float Y) {
  if (Y < kKioEncodeNear) return 1.0f / (Y + 1.0f);
  return (kf_ln(kf_fma(Y, 9.0f, 1.0f)) / 9.0f - kf_bits(0x3C4FAE29u)) / Y;
}

// bilinear sample of a 4-channel FP16 image at normalized (cx, cy), clamp to edge; the arithmetic of spec 42 section 1
// (draft 2): t0 = v00 + (v10 - v00) fx, t1 = v01 + (v11 - v01) fx, (t0 (1 - fy)) + (t1 fy)
WG_FN v4f kio_bilinear4(const h16* img, int w, int h, float cx, float cy) {
  const float px = cx * (float)w - 0.5f, py = cy * (float)h - 0.5f;
  const float fx0 = kf_floor(px), fy0 = kf_floor(py);
  const float fx = px - fx0, fy = py - fy0;
  int x0 = kf_finite(fx0) ? (int)kf_max(kf_min(fx0, (float)w), -1.0f) : (fx0 > 0.0f ? w : (fx0 < 0.0f ? -1 : 0));
  int y0 = kf_finite(fy0) ? (int)kf_max(kf_min(fy0, (float)h), -1.0f) : (fy0 > 0.0f ? h : (fy0 < 0.0f ? -1 : 0));
  const int xa = ki_max(ki_min(x0, w - 1), 0), xb = ki_max(ki_min(x0 + 1, w - 1), 0);
  const int ya = ki_max(ki_min(y0, h - 1), 0), yb = ki_max(ki_min(y0 + 1, h - 1), 0);
  const float omfy = 1.0f - fy;
  // one 8-byte load per corner (the four channels of a texel)
  const v4h q00 = *(const v4h*)kio_ptr(img, ((u32)ya * w + xa) * 4u), q10 = *(const v4h*)kio_ptr(img, ((u32)ya * w + xb) * 4u);
  const v4h q01 = *(const v4h*)kio_ptr(img, ((u32)yb * w + xa) * 4u), q11 = *(const v4h*)kio_ptr(img, ((u32)yb * w + xb) * 4u);
  v4f r;
#pragma unroll
  for (int c = 0; c < 4; ++c) {
    const float v00 = (float)q00[c], v10 = (float)q10[c], v01 = (float)q01[c], v11 = (float)q11[c];
    const float t0 = v00 + (v10 - v00) * fx;
    const float t1 = v01 + (v11 - v01) * fx;
    r[c] = t0 * omfy + t1 * fy;
  }
  return r;
}

// (Y, Co, Cg) of max(RGB, 0) scaled by kappa first (spec 42 6.2) or not (spec 43 4.2: kappa after)
WG_FN void kio_ycocg(float R, float G, float B, float* Y, float* Co, float* Cg) {
  *Y = (0.25f * R + 0.5f * G) + 0.25f * B;
  *Co = kf_fma(R, 0.5f, 0.0f * G) - 0.5f * B;
  *Cg = (0.5f * G - 0.25f * R) - 0.25f * B;
}
WG_FN float kio_luma_fma(float R, float G, float B) { return kf_fma(B, 0.25f, kf_fma(R, 0.25f, 0.5f * G)); }

// =====================================================================================================================
// output stage
// =====================================================================================================================

struct KoTap {  // spec 43 4.2 stored values of a render tap, FP16
  h16 Y, Co, Cg, A;
};

// FAST: kio_encode's near branch 1 / (Y + 1) as the reciprocal estimate (v_rcp_f32)
template <bool FAST = false>
WG_FN float kio_encode_f(float Y) {
#if defined(__AMDGCN__)
  if (FAST && Y < kKioEncodeNear) return __builtin_amdgcn_rcpf(Y + 1.0f);
#endif
  return kio_encode(Y);
}

template <bool FAST = false>
WG_FN KoTap ko_stored_tap_from_raw(const KFrameIn& F, v4f c) {
  float Y, Co, Cg;
  kio_ycocg(kf_max(c[0], 0.0f), kf_max(c[1], 0.0f), kf_max(c[2], 0.0f), &Y, &Co, &Cg);
  KoTap t;
  if (F.flags & KIO_HDR) {
    const float kY = F.kappa * Y;
    const float e = kio_encode_f<FAST>(kY);
    t.Y = kh_round(kY * e);
    t.Co = kh_round((F.kappa * Co) * e);
    t.Cg = kh_round((F.kappa * Cg) * e);
  } else {
    t.Y = kh_round(F.kappa * Y);
    t.Co = kh_round(F.kappa * Co);
    t.Cg = kh_round(F.kappa * Cg);
  }
  t.A = kh_round(c[3]);
  return t;
}

WG_FN KoTap ko_stored_tap(const KFrameIn& F, int ix, int iy) {
  return ko_stored_tap_from_raw(F, kio_color(F, ix, iy));
}

// Per render pixel (i, j): the stored tap (4.2) and the filter-tap value T (5.3) of clamped position (i, j), exactly
// as k_out_pixel computed them for each read.
WG_KERNEL_N(k_out_color, 128)(KOutParams P0) {
  const KOutParams P = ko_params(P0);
  const KFrameIn& F = P.f;
  const int id = (int)(wg_block_id() * 128u + wg_lane_id());
  if (id >= F.rw * F.rh) return;
  const int i = id % F.rw, j = id / F.rw;
  const KoTap t = ko_stored_tap(F, i, j);
  const v4h o = {t.Y, t.Co, t.Cg, t.A};
  *(v4h*)kio_ptr(P.ctap, (u32)id * 4u) = o;
  const v4f c = kio_color(F, i, j);
  const float kY = F.kappa * kio_luma_fma(kf_max(c[0], 0.0f), kf_max(c[1], 0.0f), kf_max(c[2], 0.0f));
  P.ctapT[id] = kh_round((F.flags & KIO_HDR) ? kY * kio_encode(kY) : kY);
}
// k_fast=1: the same with kio_encode_f<true>
WG_KERNEL_N(k_out_color_f, 128)(KOutParams P0) {
  const KOutParams P = ko_params(P0);
  const KFrameIn& F = P.f;
  const int id = (int)(wg_block_id() * 128u + wg_lane_id());
  if (id >= F.rw * F.rh) return;
  const int i = id % F.rw, j = id / F.rw;
  const v4f c = kio_color(F, i, j);
  const KoTap t = ko_stored_tap_from_raw<true>(F, c);
  const v4h o = {t.Y, t.Co, t.Cg, t.A};
  *(v4h*)kio_ptr(P.ctap, (u32)id * 4u) = o;
  const float kY = F.kappa * kio_luma_fma(kf_max(c[0], 0.0f), kf_max(c[1], 0.0f), kf_max(c[2], 0.0f));
  P.ctapT[id] = kh_round((F.flags & KIO_HDR) ? kY * kio_encode_f<true>(kY) : kY);
}

// a * b for a, b below 2^24 (row index times row length): the full-rate 24-bit multiply instead of the quarter-rate
// 32-bit one; the masks only tell the compiler the operand range (every caller's a and b are below 2^24, so the
// product is the same)
WG_FN u32 ko_mul24(u32 a, u32 b) { return (a & 0xFFFFFFu) * (b & 0xFFFFFFu); }

// kio_tap(i, 0, 0, n) for n >= 1 (every render size): i > 0 ? min(i, n - 1) : 0 = max(min(i, n - 1), 0), one
// v_minmax_i32 instead of a compare, a minimum and a select
WG_FN int ko_clampi(int i, int n) { return ki_max(ki_min(i, n - 1), 0); }

// index of render pixel (ix, iy) after spec-42 clamping, in the k_out_color buffers
WG_FN u32 ko_tap_index(const KFrameIn& F, int ix, int iy) {
  return ko_mul24((u32)ko_clampi(iy, F.rh), (u32)F.rw) + (u32)ko_clampi(ix, F.rw);
}

// Unclamped fast forms of the output stage's addressing (same loads and values; 0 gives the previous form, for A/B
// builds): luma window, filter taps, colour-history corners, current-colour taps, nearest-depth taps
#ifndef KO_WIN_FAST
#define KO_WIN_FAST 1
#endif
#ifndef KO_ADDR_FAST
#define KO_ADDR_FAST 1
#endif
#ifndef KO_SOFT_MAX
#define KO_SOFT_MAX 1
#endif
#ifndef KO_BIL_MIX
#define KO_BIL_MIX 1
#endif
#ifndef KO_TAP_FAST
#define KO_TAP_FAST 1
#endif

// The output stage's own forms of kio_depth / kio_nearest / kio_mtap / kio_motion (same values; the input stage keeps
// the shared ones): kio_tap(i, d, b, n) = b + clamp(i + d, 0, n - 1) and kio_mtap(i, d, b, n) = b + clamp(b + i + d,
// 0, n - 1) for n >= 1, 24-bit multiplies for row index times pitch, the depth comparison chosen once by the
// (uniform) flag instead of per candidate, the FP16 motion's channel count 2 or 4 as a shift.
WG_FN float ko_depth(const KFrameIn& F, int ix, int iy, int dx, int dy) {
  return *kio_ptr(F.depth, (u32)kio_tap(iy, dy, F.zby, F.rh) * F.zpitch + (u32)(F.zbx + ko_clampi(ix + dx, F.rw)));
}
WG_FN void ko_nearest(const KFrameIn& F, int ix, int iy, int* odx, int* ody) {
  float c, a, b, d, e;
#if KO_ADDR_FAST
  // centre at least one texel inside the render image: kio_tap clamps nothing (a row index iy + dy = 0 gives the base
  // row either way), so the five taps are one byte offset +- one pitch / one texel
  if (ix >= 1 && ix <= F.rw - 2 && iy >= 1 && iy <= F.rh - 2) {
    const u32 zp = (u32)F.zpitch * 4u;
    const u32 voff = ((u32)(F.zby + iy) * (u32)F.zpitch + (u32)(F.zbx + ix)) * 4u;
    const char* r1 = (const char*)F.depth;
    const char* r0 = r1 - zp;
    const char* r2 = r1 + zp;
    c = *(const float*)(r1 + voff);
    a = *(const float*)((r0 - 4) + voff);
    b = *(const float*)((r0 + 4) + voff);
    d = *(const float*)((r2 - 4) + voff);
    e = *(const float*)((r2 + 4) + voff);
#if defined(__AMDGCN__)
    __asm__ volatile("");  // keeps the two load sets apart
#endif
  } else
#endif
  {
    c = ko_depth(F, ix, iy, 0, 0), a = ko_depth(F, ix, iy, -1, -1), b = ko_depth(F, ix, iy, 1, -1);
    d = ko_depth(F, ix, iy, -1, 1), e = ko_depth(F, ix, iy, 1, 1);
  }
  const float k = kf_bits(0x3A83126Fu);
  const bool inv = (F.flags & KIO_INV) != 0;
  bool edge;
  if (inv)
    edge = (c < 0.5f * (a + e) - k * c) || (c < 0.5f * (b + d) - k * c);
  else
    edge = (c > kf_fma(c, k, 0.5f * (a + e))) || (c > kf_fma(c, k, 0.5f * (b + d)));
  int bx = 0, by = 0;
  if (edge) {
    float best = c;
    const float v[4] = {a, b, d, e};
    const int ox[4] = {-1, 1, -1, 1}, oy[4] = {-1, -1, 1, 1};
    if (inv) {
#pragma unroll
      for (int q = 0; q < 4; ++q)
        if (v[q] > best) {
          best = v[q];
          bx = ox[q];
          by = oy[q];
        }
    } else {
#pragma unroll
      for (int q = 0; q < 4; ++q)
        if (v[q] < best) {
          best = v[q];
          bx = ox[q];
          by = oy[q];
        }
    }
  }
  *odx = bx;
  *ody = by;
}
WG_FN int ko_mtap(int i, int d, int b, int n) { return b + ko_clampi(b + i + d, n); }
WG_FN void ko_motion(const KFrameIn& F, int x, int y, float* mvx, float* mvy) {
  const u32 t = ko_mul24((u32)y, (u32)F.mpitch) + (u32)x;
  if (F.flags & KIO_M16) {
    const u32 i = F.mch == 2 ? t << 1 : (F.mch == 4 ? t << 2 : t * (u32)F.mch);
    const h16* r = kio_ptr((const h16*)F.motion, i);
    *mvx = F.rmsx * (float)r[0] + F.rmox;
    *mvy = F.rmsy * (float)r[1] + F.rmoy;
    return;
  }
  const float* m = kio_ptr(F.motion, t * 2u);
  *mvx = m[0];
  *mvy = m[1];
}

// h(2^E) for an FP16 E (spec 43 5.4): the GPU's v_exp_f16 equals it for all 65536 inputs (TestExp2Half); the table
// elsewhere
WG_FN h16 ko_exp2h(const KOutParams& P, h16 E) {
#if defined(__AMDGCN__)
  return __builtin_elementwise_exp2(E);
#else
  return P.exp2tab[__builtin_bit_cast(unsigned short, E)];
#endif
}
// the same for both halves of a packed value (no packed exp: two v_exp_f16)
WG_FN v2h ko_exp2h2(const KOutParams& P, v2h E) {
  const h16 E0 = E[0], E1 = E[1];
  return kh2(ko_exp2h(P, E0), ko_exp2h(P, E1));
}

// Only for the beta quotient formed from ko_axis coefficients below. The final
// residual correction rounds the quotient; multiplying by a rounded reciprocal
// alone is not equivalent. Validated against native RN32 division for every
// FP32 f in [0,1] (and negative zero), with the exact coefficient operations.
WG_FN float ko_div_axis(float a, float b) {
#if defined(__AMDGCN__)
  float r = __builtin_amdgcn_rcpf(b);
  r = kf_fma(kf_fma(-b, r, 1.0f), r, r);
  const float q = a * r;
  return kf_fma(kf_fma(-b, q, a), r, q);
#else
  return a / b;
#endif
}

// Spec43 selector normalization only. For |nv|<=2^-11 the half denominator
// rounds to one; otherwise the quotient lies outside the half-underflow region.
// All 65536 input patterns yield the same selector index as native division
// (finite quotient bits also match; NaN payloads are not observable in the index).
WG_FN h16 ko_selector_normalize(h16 nv) {
  const h16 den = kh_add((h16)1.0f, __builtin_fabsf16(nv));
#if defined(__AMDGCN__)
  return kh_round((float)nv * __builtin_amdgcn_rcpf((float)den));
#else
  return kh_div(nv, den);
#endif
}

struct KoAxis {  // spec 43 3.6 per-axis cubic coefficients
  int n;
  float c0, c1, c2, c3, beta, betap;
};

// KIO_FAST (runtime option k_fast=1, the "_f" kernels; not bit-exact): the hardware reciprocal estimate (v_rcp_f32 /
// v_rcp_f16, about 1 ulp) instead of a correctly rounded division or reciprocal
WG_FN float ko_rcp_fast(float x) {
#if defined(__AMDGCN__)
  return __builtin_amdgcn_rcpf(x);
#else
  return 1.0f / x;
#endif
}
WG_FN h16 kh_rcp_fast(h16 x) {
#if defined(__AMDGCN__)
  return __builtin_amdgcn_rcph(x);
#else
  return kh_div((h16)1.0f, x);
#endif
}
// packed FP16 minNum / maxNum (v_pk_min_num_f16 / v_pk_max_num_f16) and the NaN-propagating maximum
#if defined(__AMDGCN__)
WG_FN v2h ko2_min(v2h a, v2h b) { return __builtin_elementwise_min(a, b); }
WG_FN v2h ko2_max(v2h a, v2h b) { return __builtin_elementwise_max(a, b); }
WG_FN v2h ko2_maximum(v2h a, v2h b) { return __builtin_elementwise_maximum(a, b); }
#else
WG_FN h16 ko_minnum(h16 a, h16 b) { return a != a ? b : (b != b ? a : (a < b ? a : b)); }
WG_FN h16 ko_maximum(h16 a, h16 b) { return a != a ? a : (b != b ? b : (a > b ? a : b)); }
WG_FN v2h ko2_min(v2h a, v2h b) { return kh2(ko_minnum(a[0], b[0]), ko_minnum(a[1], b[1])); }
WG_FN v2h ko2_max(v2h a, v2h b) { return kh2_maxnum(a, b); }
WG_FN v2h ko2_maximum(v2h a, v2h b) { return kh2(ko_maximum(a[0], b[0]), ko_maximum(a[1], b[1])); }
#endif

template <bool FAST = false>
WG_FN KoAxis ko_axis(float w) {
  KoAxis a;
  const float q = kf_fma(w - 0.5f, 2.0f, 0.5f);
  const float t = kf_floor(q - 0.5f) + 0.5f;
  const float f = kf_sat(q - t);
  const float f2 = f * f, f3 = f2 * f;
  a.c0 = f2 - 0.5f * (f + f3);
  a.c1 = (1.5f * f3 - 2.5f * f2) + 1.0f;
  a.c3 = 0.5f * (f3 - f2);
  a.c2 = ((1.0f - a.c0) - a.c1) - a.c3;
  // c1 + c2 = 1 + f (1 - f) / 2 lies in [1, 1.125]
  a.beta = FAST ? a.c1 * ko_rcp_fast(a.c1 + a.c2) : ko_div_axis(a.c1, a.c1 + a.c2);
  a.betap = 1.0f - a.beta;
  const float nn = t - 0.5f;
  // = kf_finite(nn) ? (int)clamp(nn) : (nn > 0 ? 1e9 : -1e9), without a branch: the clamp already gives +-1e9 for
  // +-inf (1e9 is exact in FP32); only NaN (clamp: 1e9) needs the select
  const int nc = (int)kf_max(kf_min(nn, 1.0e9f), -1.0e9f);
  a.n = nn != nn ? -1000000000 : nc;
  return a;
}

// kio_bilinear4 split in two for the output stage, so its loads can be issued with the luma-window loads: the corner
// texels, then the same arithmetic. x0 / y0 without a branch: the clamp gives w / -1 for +-inf as the original does;
// only NaN (original 0) needs the select.
struct KoBil {
  v4h q00, q10, q01, q11;
  float fx, fy;
};
WG_FN KoBil ko_bilinear4_load(const h16* img, int w, int h, float cx, float cy) {
  KoBil b;
  const float px = cx * (float)w - 0.5f, py = cy * (float)h - 0.5f;
  const float fx0 = kf_floor(px), fy0 = kf_floor(py);
  b.fx = px - fx0;
  b.fy = py - fy0;
  int x0 = (int)kf_max(kf_min(fx0, (float)w), -1.0f);
  int y0 = (int)kf_max(kf_min(fy0, (float)h), -1.0f);
  x0 = fx0 != fx0 ? 0 : x0;
  y0 = fy0 != fy0 ? 0 : y0;
#if KO_ADDR_FAST
  // all four corners inside (no clamp): one byte offset, the corners at +0 / +8 bytes on two rows
  if ((u32)x0 <= (u32)(w - 2) && (u32)y0 <= (u32)(h - 2)) {
    const u32 voff = (ko_mul24((u32)y0, (u32)w) + (u32)x0) * 8u;
    const char* r0 = (const char*)img;
    const char* r1 = r0 + (u32)w * 8u;
    b.q00 = *(const v4h*)(r0 + voff);
    b.q10 = *(const v4h*)((r0 + 8) + voff);
    b.q01 = *(const v4h*)(r1 + voff);
    b.q11 = *(const v4h*)((r1 + 8) + voff);
#if defined(__AMDGCN__)
    __asm__ volatile("");  // keeps the two load sets apart
#endif
    return b;
  }
#endif
  const int xa = ki_max(ki_min(x0, w - 1), 0), xb = ki_max(ki_min(x0 + 1, w - 1), 0);
  const int ya = ki_max(ki_min(y0, h - 1), 0), yb = ki_max(ki_min(y0 + 1, h - 1), 0);
  const u32 ra = ko_mul24((u32)ya, (u32)w), rb = ko_mul24((u32)yb, (u32)w);
  b.q00 = *(const v4h*)kio_ptr(img, (ra + xa) * 4u);
  b.q10 = *(const v4h*)kio_ptr(img, (ra + xb) * 4u);
  b.q01 = *(const v4h*)kio_ptr(img, (rb + xa) * 4u);
  b.q11 = *(const v4h*)kio_ptr(img, (rb + xb) * 4u);
  return b;
}
WG_FN v4f ko_bilinear4_math(const KoBil& b) {
  const float fx = b.fx, fy = b.fy;
  const float omfy = 1.0f - fy;
  v4f r;
#if KO_BIL_MIX && defined(__AMDGCN__)
  // the same FP32 steps with the FP16 corners read directly by v_fma_mix_f32 (fma(v, 1, w) is the exact sum v + w
  // rounded once, FP16 -> FP32 is exact): no conversions or half extractions. The opaque 1 keeps the fma form.
  float one = 1.0f;
  __asm__("" : "+v"(one));
#pragma unroll
  for (int c = 0; c < 4; ++c) {
    const float v00 = (float)b.q00[c], v10 = (float)b.q10[c], v01 = (float)b.q01[c], v11 = (float)b.q11[c];
    const float t0 = kf_fma(v00, one, kf_fma(v10, one, -v00) * fx);
    const float t1 = kf_fma(v01, one, kf_fma(v11, one, -v01) * fx);
    r[c] = t0 * omfy + t1 * fy;
  }
  return r;
#endif
#pragma unroll
  for (int c = 0; c < 4; ++c) {
    const float v00 = (float)b.q00[c], v10 = (float)b.q10[c], v01 = (float)b.q01[c], v11 = (float)b.q11[c];
    const float t0 = v00 + (v10 - v00) * fx;
    const float t1 = v01 + (v11 - v01) * fx;
    r[c] = t0 * omfy + t1 * fy;
  }
  return r;
}

// ko_cbar split in two: the four stored taps (loaded at the start of the kernel, they depend only on the pixel
// position), then the same FP16 arithmetic
struct KoCbar {
  v4h q[4];
  h16 w[4];
};
WG_FN KoCbar ko_cbar_load(const KOutParams& P, float Ux_, float Uy_, float o_x, float o_y) {
  const KFrameIn& F = P.f;
  KoCbar c;
  const h16 Ux = kh_round(Ux_), Uy = kh_round(Uy_);
  const h16 flx = kh_floor(Ux), fly = kh_floor(Uy);
  const h16 fx = kh_sat(kh_sub(Ux, flx)), fy = kh_sat(kh_sub(Uy, fly));
  const h16 one = (h16)1.0f;
  c.w[0] = kh_mul(kh_sub(one, fx), kh_sub(one, fy));
  c.w[1] = kh_mul(fx, kh_sub(one, fy));
  c.w[2] = kh_mul(kh_sub(one, fx), fy);
  c.w[3] = kh_mul(fx, fy);
  const int bx = (int)(o_x + (float)flx), by = (int)(o_y + (float)fly);
  const int dx[4] = {0, 1, 0, 1}, dy[4] = {0, 0, 1, 1};
#if KO_ADDR_FAST
  // the 2 x 2 taps inside the render image (no clamp): one byte offset, +0 / +8 bytes on two rows
  if (bx >= 0 && bx <= F.rw - 2 && by >= 0 && by <= F.rh - 2) {
    const u32 voff = (ko_mul24((u32)by, (u32)F.rw) + (u32)bx) * 8u;
    const char* r0 = (const char*)P.ctap;
    const char* r1 = r0 + (u32)F.rw * 8u;
    c.q[0] = *(const v4h*)(r0 + voff);
    c.q[1] = *(const v4h*)((r0 + 8) + voff);
    c.q[2] = *(const v4h*)(r1 + voff);
    c.q[3] = *(const v4h*)((r1 + 8) + voff);
#if defined(__AMDGCN__)
    __asm__ volatile("");  // keeps the two load sets apart
#endif
    return c;
  }
#endif
#pragma unroll
  for (int q = 0; q < 4; ++q) c.q[q] = *(const v4h*)kio_ptr(P.ctap, ko_tap_index(F, bx + dx[q], by + dy[q]) * 4u);
  return c;
}
// channels (Y, Co) and (Cg, A) as packed pairs: each half rounded exactly like the scalar kh_mul / kh_fma
WG_FN void ko_cbar_math(const KoCbar& c, h16 out[4]) {
  v2h lo, hi;
#pragma unroll
  for (int q = 0; q < 4; ++q) {
    const v2h vl = kh2(c.q[q][0], c.q[q][1]), vh = kh2(c.q[q][2], c.q[q][3]), w = kh2_splat(c.w[q]);
    lo = q == 0 ? kh2_mul(vl, w) : kh2_fma(vl, w, lo);
    hi = q == 0 ? kh2_mul(vh, w) : kh2_fma(vh, w, hi);
  }
  out[0] = lo[0];
  out[1] = lo[1];
  out[2] = hi[0];
  out[3] = hi[1];
}

// split workgroup barrier for the F2 exchange: signal after the LDS store, wait before the LDS reads (the work in
// between overlaps the other wave's arrival)
WG_FN void ko_barrier_signal() {
#if defined(__AMDGCN__) && WG_LDS_BARRIER
  __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup", "local");
  __builtin_amdgcn_s_barrier_signal(-1);
#endif
}
WG_FN void ko_barrier_wait() {
#if defined(__AMDGCN__) && WG_LDS_BARRIER
  __builtin_amdgcn_s_barrier_wait(-1);
  __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup", "local");
#else
  wg_barrier_lds();
#endif
}

WG_FN float ko_ell(h16 v) {
  const float x = (float)v, c = kf_bits(0x3F7D70A4u);
  return x < c ? x : kf_fma(x - c, 0.001f, c);
}

// kf_sat in three instructions instead of four: the clamp to [0, 1] (v_med3 / clamp: NaN -> +0, negative -> +0) gives
// +0 for -0 where kf_sat keeps -0, so a zero is passed through. Equal to kf_sat for all 2^32 inputs (checked on the
// GPU, exhaustively).
WG_FN float ko_sat(float x) {
#if defined(__AMDGCN__)
  return x == 0.0f ? x : __builtin_amdgcn_fmed3f(x, 0.0f, 1.0f);
#else
  return kf_sat(x);
#endif
}

WG_FN float ko_cubic(float Am1, float A0, float A1, float A2, const KoAxis& a) {
  const float lam = 1.37f;
  float r = kf_fma(a.beta, A0, a.betap * A1);
  r = kf_fma(lam * a.c0, ko_sat(Am1), r);
  r = kf_fma(lam * (a.c1 - a.beta), ko_sat(A0), r);
  r = kf_fma(lam * (a.c2 - a.betap), ko_sat(A1), r);
  return kf_fma(lam * a.c3, ko_sat(A2), r);
}
// ko_cubic for taps known to lie in [0, 1) (ko_sat is then the identity): the same operations without the clamps
WG_FN float ko_cubic_unit(float Am1, float A0, float A1, float A2, const KoAxis& a) {
  const float lam = 1.37f;
  float r = kf_fma(a.beta, A0, a.betap * A1);
  r = kf_fma(lam * a.c0, Am1, r);
  r = kf_fma(lam * (a.c1 - a.beta), A0, r);
  r = kf_fma(lam * (a.c2 - a.betap), A1, r);
  return kf_fma(lam * a.c3, A2, r);
}

// Correctly rounded reciprocal in the exhaustively checked positive normal
// domain. Retain native division for every other value, including signed zeros,
// negatives, non-finite values and denominators outside [2^-32,2^32].
// One Newton step on v_rcp_f32 already gives the native 1 / x for every FP32 x in [2^-32, 2^32] (exhaustive GPU
// check, all 2^32 patterns; KO_RCP2 = 1 restores the second step).
#ifndef KO_RCP2
#define KO_RCP2 0
#endif
WG_FN float ko_rcp_refined(float x) {
#if defined(__AMDGCN__)
  if (x >= 0x1p-32f && x <= 0x1p32f) {
    float r = __builtin_amdgcn_rcpf(x);
#if KO_RCP2
    r = kf_fma(kf_fma(-x, r, 1.0f), r, r);
#endif
    return kf_fma(kf_fma(-x, r, 1.0f), r, r);
  }
#endif
  return 1.0f / x;
}

// Output-only reciprocal with required precondition x in [2^-32, 2^32].
// The history batch predicates establish this for 1-luma and 1+linear-luma.
// For max(Yc,eps), confidence/current-colour guards and bounded FP16 inputs
// establish eps <= x < 2^21. Other denominators retain the guarded/native path.
WG_FN float ko_rcp_output_bounded(float x) {
#if defined(__AMDGCN__)
  float r = __builtin_amdgcn_rcpf(x);
#if KO_RCP2
  r = kf_fma(kf_fma(-x, r, 1.0f), r, r);
#endif
  return kf_fma(kf_fma(-x, r, 1.0f), r, r);
#else
  return 1.0f / x;
#endif
}

// COMPACT: P.N is the compact pixel head [2 Bh][2 Bw][4] FP16 (display-position raster; channels 0, 1, 9 of the
// position's slot, then 0) that k_stage2_d0_hp / k_stage2h_d0_hp write, instead of the 40-channel head map
// [Bh][Bw][40]: the same three values per pixel in a quarter of the bytes (clean/k/kernels/k_block.hip).
template <bool COMPACT, bool FAST = false>
WG_FN void ko_pixel(KOutParams P0) {
  const KOutParams P = ko_params(P0);
  const KFrameIn& F = P.f;
  const int W16 = kio_ceil16(F.W), H16 = kio_ceil16(F.H);
  // workgroup = 64 x 2 pixels (H16 is even), so the aligned 2 x 2 groups of F2 meet in LDS; lanes beyond W16 compute a
  // clamped pixel and write nothing
  const int lane = (int)wg_lane_id(), tiles_x = (W16 + 63) / 64;
  const int tile = (int)wg_block_id();
  const int x_raw = (tile % tiles_x) * 64 + (lane & 63);
  const bool active = x_raw < W16;
  // a wave (32 or 64 lanes) lies in one row: y is wave-uniform (readfirstlane), so the row's values (py, o_y, Uy, Ys,
  // v, ...) are computed once per wave in scalar registers (k_out_pixel -3 %)
  const int x = active ? x_raw : W16 - 1, y = (tile / tiles_x) * 2 + wg_uniform(lane >> 6);
  WG_LDS h16 lds_ybar[128];
  const bool hdr = (F.flags & KIO_HDR) != 0, mvlo = (F.flags & KIO_MVLO) != 0;
  const float px = (float)x + 0.5f, py = (float)y + 0.5f;
  const KOutUni& U = P.u;
  const float invW = U.invW, invH = U.invH;
  const float rhox = U.rhox, rhoy = U.rhoy;
  const float rho2x = U.rho2x, rho2y = U.rho2y;
  const float jmx = U.jmx, jmy = U.jmy;

  // 3.1 network channels: display position (u, v) = (trunc(s x), trunc(s y))
  const int u = (int)kf_trunc(F.s * (float)x), v = (int)kf_trunc(F.s * (float)y);
  const int k = (u & 1) + 2 * (v & 1);
  const int bxn = ki_min(u >> 1, F.Bw - 1), byn = ki_min(v >> 1, F.Bh - 1);
  const h16* nc = COMPACT ? kio_ptr(P.N, ((u32)(2 * byn + (k >> 1)) * (2u * (u32)F.Bw) + (u32)(2 * bxn + (k & 1))) * 4u)
                          : kio_ptr(P.N, ((u32)byn * F.Bw + bxn) * 40u + 10u * k);
  // the compact entry in one 8-byte load: (channel 0, channel 1, channel 9, 0) of the slot
  const v4h zero4 = {(h16)0.0f, (h16)0.0f, (h16)0.0f, (h16)0.0f};
  const v4h nv4 = COMPACT ? *(const v4h*)nc : zero4;

  // The kernel's memory accesses form a chain (depth -> motion -> colour / luma history -> table -> filter taps). The
  // loads that depend only on the pixel position (head entry above, the four current-colour taps of 4.3 here) are
  // issued first, and the colour-history corners together with the luma window; every value and operation is the
  // same as before, only the order of independent steps differs.
  // 4.1 local origin, 4.3 current-colour taps (the arithmetic follows below)
  const float o_x = kf_R(kf_fma((float)(16 * (x / 16)) + 0.5f, rhox, jmx)) - 4.0f;
  const float o_y = kf_R(kf_fma((float)(16 * (y / 16)) + 0.5f, rhoy, jmy)) - 4.0f;
  const float Pxf = kf_fma(px, rhox, jmx) - o_x, Pyf = kf_fma(py, rhoy, jmy) - o_y;
  const h16 Ux = kh_round(Pxf), Uy = kh_round(Pyf);
  const KoCbar cbq = ko_cbar_load(P, (float)Ux, (float)Uy, o_x, o_y);

  // 3.3 motion
  float mvx, mvy, spx, spy;
  if (mvlo) {
    const int ix = (int)kf_R(kf_fma(px, rhox, jmx)), iy = (int)kf_R(kf_fma(py, rhoy, jmy));
    int dx, dy;
    ko_nearest(F, ix, iy, &dx, &dy);
    ko_motion(F, ko_mtap(ix, dx, F.mbx, F.rw), ko_mtap(iy, dy, F.mby, F.rh), &mvx, &mvy);
  } else {
    const int mx = (int)kf_R(kf_fma(px, 1.0f, -0.5f)), my = (int)kf_R(kf_fma(py, 1.0f, -0.5f));
    ko_motion(F, F.mbx + ko_clampi(mx, F.W), F.mby + ko_clampi(my, F.H), &mvx, &mvy);
  }
  spx = U.spx;  // [MVlo]: sx / rho_x, else sx
  spy = U.spy;
  const float wx = kf_fma(spx, mvx + F.mox, px), wy = kf_fma(spy, mvy + F.moy, py);

  // 3.4 validity, 3.5 colour history
  bool valid = !(F.flags & KIO_RESET) && wx * invW >= 0.0f && wx * invW <= 1.0f && wy * invH >= 0.0f &&
               wy * invH <= 1.0f;

  // 4.3 current colour (taps loaded above); F2: Ybar of the aligned 2 x 2 group (same local origin, so each pixel's
  // own Ybar is the one the others would compute); the group's lanes are lane, lane ^ 1, lane ^ 64, lane ^ 65. Stored
  // here, before the history loads are issued, so the release before the barrier signal has no load to wait for.
  h16 cbar[4];
  ko_cbar_math(cbq, cbar);
  lds_ybar[lane] = cbar[0];
  ko_barrier_signal();

  const KoBil hcq = ko_bilinear4_load(P.Hc, W16, H16, wx * U.invW16, wy * U.invH16);

  // 3.6 luma history
  const KoAxis axx = ko_axis<FAST>(wx), axy = ko_axis<FAST>(wy);
  const int LW = 2 * W16, LH = 2 * H16;
  // the four sub-samples read rows n_y - 1 .. n_y + 3 and columns n_x - 1 .. n_x + 3 (clamped): read that 5 x 5
  // window once; sub-sample (i, j) uses rows i .. i + 3, columns j .. j + 3 of it
  h16 Lr[5][5];
#if KO_WIN_FAST
  // Window inside the image (no index is clamped; every lane but those at the borders): the same 25 texels addressed
  // as uniform row base + one per-lane byte offset + immediate column offset, instead of a clamp and an address per
  // texel. (u32)(n - 1) <= size - 5 is 0 <= n - 1 and n + 3 <= size - 1.
  if ((u32)(axx.n - 1) <= (u32)(LW - 5) && (u32)(axy.n - 1) <= (u32)(LH - 5)) {
    const u32 voff = (ko_mul24((u32)(axy.n - 1), (u32)LW) + (u32)(axx.n - 1)) * 2u;
#pragma unroll
    for (int r = 0; r < 5; ++r) {
      const char* row = (const char*)P.L + (u32)(r * LW) * 2u;
#pragma unroll
      for (int c = 0; c < 5; ++c) Lr[r][c] = *(const h16*)((row + 2 * c) + voff);
    }
#if defined(__AMDGCN__)
    __asm__ volatile("");  // keeps the two load sets apart (no merged loads through selected 64-bit addresses)
#endif
  } else
#endif
  {
#pragma unroll
    for (int r = 0; r < 5; ++r) {
      const int yr = ki_max(ki_min(axy.n - 1 + r, LH - 1), 0);
#pragma unroll
      for (int c = 0; c < 5; ++c)
        Lr[r][c] = *kio_ptr(P.L, ko_mul24((u32)yr, (u32)LW) + ki_max(ki_min(axx.n - 1 + c, LW - 1), 0));
    }
  }

  // 3.2 blend logit
  const float t = 0.5f * (float)(COMPACT ? nv4[2] : nc[9]);
  const float t2 = t * t;
  const float g = FAST ? (t * (t2 + 27.0f)) * ko_rcp_fast(kf_fma(t2, 9.0f, 27.0f))
                       : (t * (t2 + 27.0f)) / kf_fma(t2, 9.0f, 27.0f);
  const float alpha = t < -3.0f ? 0.0f : (t > 3.0f ? 1.0f : kf_fma(g, 0.5f, 0.5f));

  // 3.5 colour history
  v4f Hh = ko_bilinear4_math(hcq);

  float ellv[4];
  bool ell_done = false;
  if (FAST) {
    // FAST: the window as packed FP16 pairs P2[r][c] = (L[r][c], L[r][c + 1]), so the sub-samples (i, 0) and (i, 1)
    // run as the two halves of v_pk_* operations: the cubics with FP16 coefficients (each row cubic folded to four
    // terms, k0..k3, rounded to FP16 once per lane), the 2 x 2 min / max clamp per half. Windows with a tap >= 0.99
    // (soft limit) or NaN take the FP32 form below. FP16 0.990234375 is the first FP16 value above 0.99f, so h < it
    // is h < 0.99f for every FP16 h.
    v2h P2[5][4];
#pragma unroll
    for (int r = 0; r < 5; ++r)
#pragma unroll
      for (int c = 0; c < 4; ++c) P2[r][c] = kh2(Lr[r][c], Lr[r][c + 1]);
    // NaN-propagating maximum of the 25 taps: pairs 0, 2, 3 of each row hold columns 0..4
    v2h m2 = ko2_maximum(P2[0][0], ko2_maximum(P2[0][2], P2[0][3]));
#pragma unroll
    for (int r = 1; r < 5; ++r) m2 = ko2_maximum(m2, ko2_maximum(P2[r][0], ko2_maximum(P2[r][2], P2[r][3])));
    if (m2[0] < (h16)0.990234375f && m2[1] < (h16)0.990234375f) {
      const float lam = 1.37f;
      const v2h k0 = kh2_splat(kh_round(lam * axx.c0)), k1 = kh2_splat(kh_round(kf_fma(lam, axx.c1 - axx.beta, axx.beta)));
      const v2h k2 = kh2_splat(kh_round(kf_fma(lam, axx.c2 - axx.betap, axx.betap))), k3 = kh2_splat(kh_round(lam * axx.c3));
      const v2h yb = kh2_splat(kh_round(axy.beta)), ybp = kh2_splat(kh_round(axy.betap));
      const v2h y0 = kh2_splat(kh_round(lam * axy.c0)), y1 = kh2_splat(kh_round(lam * (axy.c1 - axy.beta)));
      const v2h y2 = kh2_splat(kh_round(lam * (axy.c2 - axy.betap))), y3 = kh2_splat(kh_round(lam * axy.c3));
#pragma unroll
      for (int i = 0; i < 2; ++i) {
        v2h R[4];
#pragma unroll
        for (int mr = 0; mr < 4; ++mr) {
          const v2h* p = P2[i + mr];
          R[mr] = kh2_fma(k3, p[3], kh2_fma(k2, p[2], kh2_fma(k1, p[1], kh2_mul(k0, p[0]))));
        }
        v2h val = kh2_fma(ybp, R[2], kh2_mul(yb, R[1]));
        val = kh2_fma(y0, kh2_sat_nz(R[0]), val);
        val = kh2_fma(y1, kh2_sat_nz(R[1]), val);
        val = kh2_fma(y2, kh2_sat_nz(R[2]), val);
        val = kh2_fma(y3, kh2_sat_nz(R[3]), val);
        const v2h lo = ko2_min(ko2_min(P2[i + 1][1], P2[i + 1][2]), ko2_min(P2[i + 2][1], P2[i + 2][2]));
        const v2h hi = ko2_max(ko2_max(P2[i + 1][1], P2[i + 1][2]), ko2_max(P2[i + 2][1], P2[i + 2][2]));
        const v2h e = ko2_min(ko2_max(val, lo), hi);
        ellv[2 * i] = (float)e[0];
        ellv[2 * i + 1] = (float)e[1];
      }
      ell_done = true;
    }
  }
  if (!ell_done) {  // (not indented: the exact form as before)
  float Lw[5][5];
  // Soft limiting is the identity below 0.99. Keep the exact fallback for
  // every high or NaN tap, but skip its arithmetic for an ordinary window.
#if KO_SOFT_MAX && defined(__AMDGCN__)
  // !(x < 0.99) for some x of the window <=> !(maximum < 0.99) with the NaN-propagating maximum (v_maximum3_f32): one
  // instruction per two values and one compare instead of a compare per value
  float wmax;
#pragma unroll
  for (int r = 0; r < 5; ++r)
#pragma unroll
    for (int c = 0; c < 5; ++c) {
      Lw[r][c] = (float)Lr[r][c];
      wmax = (r | c) ? __builtin_elementwise_maximum(wmax, Lw[r][c]) : Lw[r][c];
    }
  const bool needs_soft = !(wmax < kf_bits(0x3F7D70A4u));
#else
  bool needs_soft = false;
#pragma unroll
  for (int r = 0; r < 5; ++r)
#pragma unroll
    for (int c = 0; c < 5; ++c) {
      Lw[r][c] = (float)Lr[r][c];
      needs_soft = needs_soft || !(Lw[r][c] < kf_bits(0x3F7D70A4u));
    }
#endif
  // FAST: Ls = the row cubics' clamped taps ko_sat(Lw), clamped once per tap in the (rare) soft-limit branch only. The
  // stored luma history is never negative (k_out_pixel writes RN16 of a convex combination of non-negative values, or
  // of the soft-limit expansion above 0.99), so without soft limiting every tap lies in [0, 0.99) and ko_sat is the
  // identity.
  float Ls[5][5];
#pragma unroll
  for (int r = 0; r < 5; ++r)
#pragma unroll
    for (int c = 0; c < 5; ++c) Ls[r][c] = Lw[r][c];
  if (needs_soft) {
#pragma unroll
    for (int r = 0; r < 5; ++r)
#pragma unroll
      for (int c = 0; c < 5; ++c) {
        Lw[r][c] = ko_ell((h16)Lw[r][c]);
        Ls[r][c] = ko_sat(Lw[r][c]);
      }
  }
#pragma unroll
  for (int i = 0; i < 2; ++i)
#pragma unroll
    for (int j = 0; j < 2; ++j) {
      float rows[4];
#pragma unroll
      for (int mr = 0; mr < 4; ++mr)
        rows[mr] = FAST ? ko_cubic_unit(Ls[i + mr][j], Ls[i + mr][j + 1], Ls[i + mr][j + 2], Ls[i + mr][j + 3], axx)
                        : ko_cubic(Lw[i + mr][j], Lw[i + mr][j + 1], Lw[i + mr][j + 2], Lw[i + mr][j + 3], axx);
      const float val = ko_cubic(rows[0], rows[1], rows[2], rows[3], axy);
      float lo = 0.0f, hi = 0.0f;
#pragma unroll
      for (int a2 = 0; a2 < 2; ++a2)
#pragma unroll
        for (int b2 = 0; b2 < 2; ++b2) {
          const float v = Lw[i + a2 + 1][j + b2 + 1];
          lo = (a2 | b2) ? __builtin_fminf(lo, v) : v;
          hi = (a2 | b2) ? __builtin_fmaxf(hi, v) : v;
        }
      ellv[j + 2 * i] = __builtin_fminf(__builtin_fmaxf(val, lo), hi);
    }
  }

  // 3.8 / 3.9 history luma
  // Use the near decode/encode batch only after both denominator domains are
  // established. Negative/non-finite values and either far branch retain the
  // full functions for all four sub-samples, in the original operation order.
  float Hl[4];
  if (hdr) {
    bool near = true;
#pragma unroll
    for (int s = 0; s < 4; ++s)
      near = near && ellv[s] >= 0.0f && ellv[s] < kKioDecodeNear;
    if (FAST && near) {
      // FAST: decode then encode in one step, lin = rho e / (1 - e) and lin / (lin + 1) = rho e / (1 + (rho - 1) e)
      // (e exactly when rho = 1), one reciprocal estimate; near also needs lin < 1000, i.e. rho e < 1000 (1 - e)
      const float rm1 = F.rho - 1.0f;
#pragma unroll
      for (int s = 0; s < 4; ++s) {
        const float re = F.rho * ellv[s];
        near = near && re < 1000.0f * (1.0f - ellv[s]);
        Hl[s] = re * ko_rcp_fast(kf_fma(rm1, ellv[s], 1.0f));
      }
    } else if (near) {
      float lin[4];
#pragma unroll
      for (int s = 0; s < 4; ++s) {
        lin[s] = F.rho * (ellv[s] * ko_rcp_output_bounded(1.0f - ellv[s]));
        near = near && lin[s] >= 0.0f && lin[s] < kKioEncodeNear;
      }
      if (near)
#pragma unroll
        for (int s = 0; s < 4; ++s) Hl[s] = lin[s] * ko_rcp_output_bounded(lin[s] + 1.0f);
    }
    if (!near)
#pragma unroll
      for (int s = 0; s < 4; ++s) {
        const float lin = F.rho * (ellv[s] * kio_decode(ellv[s]));
        Hl[s] = lin * kio_encode(lin);
      }
  } else {
#pragma unroll
    for (int s = 0; s < 4; ++s) Hl[s] = F.rho * ellv[s];
  }
  // 3.10 confidence (H00 = 0, H10 = 1, H01 = 2, H11 = 3)
  const float chk = kf_fma(alpha, Hh[2], kf_fma(Hh[0], Hh[1], kf_fma(Hl[0], Hl[1], Hl[2] * Hl[3])));
  valid = valid && kf_finite(chk);
  const float alphap = valid ? alpha : 1.0f, bp = valid ? Hh[3] : 1.0f;
  if (!valid) {
#pragma unroll
    for (int s = 0; s < 4; ++s) Hl[s] = 0.0f;
    Hh = wg_splat(0.0f).lo;
  }
  const float m0 = kf_fma(alphap, 1.0f - bp, bp);
  const float m1 = kf_fma(bp, 1.0f - alphap * alphap, alphap * alphap);

  // F2 (Ybar of the group, stored above)
  ko_barrier_wait();
  const int g0 = lane & ~65;
  const bool F2 = lds_ybar[g0] > (h16)0.8f || lds_ybar[g0 + 1] > (h16)0.8f || lds_ybar[g0 + 64] > (h16)0.8f ||
                  lds_ybar[g0 + 65] > (h16)0.8f;

  // 5.1 table index
  int qs[2];
#pragma unroll
  for (int c = 0; c < 2; ++c) {
    const h16 nv = COMPACT ? nv4[c] : nc[c];
    const h16 z = ko_selector_normalize(nv);
    const int qv = (int)(float)kh_rni(kh_mul(z, (h16)8.0008f)) + 8;
    qs[c] = ki_max(ki_min(qv, 15), 0);
  }
  const bool F0 = (float)(h16)0.02f > m0;
  const bool F1 = m0 < 0.25f && alpha < 0.25f;
  const int tix = (16 * qs[0] + qs[1]) + 256 * ((int)F0 + 2 * (int)F1 + 4 * (int)F2);
  const h16* ent = kio_ptr(P.table, (u32)tix * 8u);
  const h16 tA = ent[0], tC = ent[1], tB = ent[2], tS = ent[3], tK = ent[5], tk6 = ent[6];
  const bool O = __builtin_bit_cast(unsigned short, ent[4]) != 0;

  // 5.2 line coordinates
  h16 Xs[2], Ys[2];
#pragma unroll
  for (int s = 0; s < 2; ++s) {
    Xs[s] = kh_round(kf_fma((float)(2 * x + s) + 0.5f, rho2x, jmx) - o_x);
    Ys[s] = kh_round(kf_fma((float)(2 * y + s) + 0.5f, rho2y, jmy) - o_y);
  }
  const h16 uP = O ? Uy : Ux, vP = O ? Ux : Uy;

  // 5.3 taps (shared by the four sub-samples)
  const h16 one = (h16)1.0f;
  const h16 v0 = kh_sub(kh_floor(vP), one);
  h16 cl = kh_add(uP, kh_mul(kh_sub(v0, vP), tS));
  h16 lines[4], cols[4];
  h16 T[12];
  // tap (l, tc) is render pixel rx = (int)(o_x + line), ry = (int)(o_y + col) for O, else rx = (int)(o_x + col),
  // ry = (int)(o_y + line): the line coordinate (one per l) and the column coordinate (one per tap) computed once each
  // with their own origin and clamp, then placed by O (the same conversions and clamps as ko_tap_index(F, rx, ry))
  const float o_ln = O ? o_x : o_y, o_cl = O ? o_y : o_x;
  const int n_ln = O ? F.rw : F.rh, n_cl = O ? F.rh : F.rw;
  int as[4];
#pragma unroll
  for (int l = 0; l < 4; ++l) {
    if (l > 0) cl = kh_add(cl, tS);
    lines[l] = kh_add(v0, (h16)(float)l);
    cols[l] = kh_sub(kh_rni(cl), one);
    as[l] = ko_clampi((int)(o_ln + (float)lines[l]), n_ln);
  }
#if KO_TAP_FAST
  // Fast form where no column index of the lane is clamped: with |cols| <= 2045 the FP16 sums cols + tc (tc = 0, 1,
  // 2) are exact integers, and with c0 = (int)(o_cl + cols) in [0, n_cl - 3] the FP32 sums o_cl + cols + tc are
  // exact too (|o_cl| < 2^15), so tap tc is column c0 + tc, unclamped: one address per line, the taps at +0, +1, +2
  // texels along the column direction (one row of the buffer apart for O). NaN / inf columns fail the first test, a
  // NaN origin (whose (int) conversion gives 0) the second.
  bool tfast = __builtin_fabsf(o_cl) < 32768.0f;
  int cb[4];
#pragma unroll
  for (int l = 0; l < 4; ++l) {
    cb[l] = (int)(o_cl + (float)cols[l]);
    tfast = tfast && __builtin_fabsf((float)cols[l]) <= 2045.0f && cb[l] >= 0 && cb[l] <= n_cl - 3;
  }
  if (tfast) {
    const u32 st = O ? (u32)F.rw * 2u : 2u;
#pragma unroll
    for (int l = 0; l < 4; ++l) {
      const u32 b = (ko_mul24((u32)(O ? cb[l] : as[l]), (u32)F.rw) + (u32)(O ? as[l] : cb[l])) * 2u;
#pragma unroll
      for (int tc = 0; tc < 3; ++tc) T[3 * l + tc] = *(const h16*)((const char*)P.ctapT + (b + (u32)tc * st));
    }
#if defined(__AMDGCN__)
    __asm__ volatile("");  // keeps the two load sets apart
#endif
  } else
#endif
  {
#pragma unroll
    for (int l = 0; l < 4; ++l)
#pragma unroll
      for (int tc = 0; tc < 3; ++tc) {
        const h16 colv = kh_add(cols[l], (h16)(float)tc);
        const int c = ko_clampi((int)(o_cl + (float)colv), n_cl);
        T[3 * l + tc] = *kio_ptr(P.ctapT, ko_mul24((u32)(O ? c : as[l]), (u32)F.rw) + (u32)(O ? as[l] : c));
      }
  }

  // 5.4 - 5.6 filtered values, 6.1 blend weights
  const h16 kux = __builtin_bit_cast(h16, U.kux), kvy = __builtin_bit_cast(h16, U.kvy);  // h(-8 / rho2)
  const h16 k_u = O ? kvy : kux, k_v = O ? kux : kvy;
  // The sub-samples (i, 0) and (i, 1) run together as the two halves of packed FP16 values (kh2_*: each half is
  // rounded exactly like the scalar operation); 2^E and the division stay per sub-sample.
  // Best-E tracking ("unless bestE > E: bestE = E, bestT = T") decides only where the final bestE < -12. Without a NaN
  // E that final bestE is a largest E (a zero of either sign at worst, which is not < -12), so the main loop keeps
  // only the largest E (maxNum) and the division stands where it is >= -12 and sg is not NaN (sg, a sum of 2^E >= 0,
  // is NaN exactly when an E is NaN). Other lanes (rare) run the tracking itself again on the same E values: with the
  // mask kh2_gtmask(bestE, E) and bit selects. It differs from the comparison only for bestE = +0, E = -0 (kept instead
  // of replaced): bestE then stays a zero of the other sign, which compares the same, and bestT can differ only while
  // bestE is a zero (the next replacement of both forms agrees again); a final zero bestE is not < -12, so bestT is
  // unused and Fq is the spec's.
  h16 Fv[4], gam[4];
  v2h gam2[2];
  const v2h one2 = kh2_splat(one), tA2 = kh2_splat(tA), tB2 = kh2_splat(tB), tC2 = kh2_splat(tC);
  // E of the 12 taps (index 3 l + tc) for the sub-sample pair at (us, vs), in the spec's operation order
  auto each_E = [&](v2h us, v2h vs, auto&& f) {
#pragma unroll
    for (int l = 0; l < 4; ++l) {
      const v2h dv = kh2_sub(kh2_splat(lines[l]), vs);
      const v2h t2v = kh2_mul(kh2_mul(dv, dv), tA2);
      const v2h t3v = kh2_mul(dv, tB2);
      v2h du = kh2_sub(kh2_splat(cols[l]), us), mm = kh2_fma(du, t3v, t2v);
#pragma unroll
      for (int tc = 0; tc < 3; ++tc) {
        if (tc > 0) {
          du = kh2_add(du, one2);
          mm = kh2_add(mm, t3v);
        }
        f(3 * l + tc, kh2_fma(kh2_mul(du, du), tC2, mm));
      }
    }
  };
  v2h us2[2], vs2[2];
  bool plain = true;  // every sub-sample's largest E >= -12 and sg not NaN: Fv is the quotient
#pragma unroll
  for (int i = 0; i < 2; ++i) {
    const v2h us = O ? kh2_splat(Ys[i]) : kh2(Xs[0], Xs[1]), vs = O ? kh2(Xs[0], Xs[1]) : kh2_splat(Ys[i]);
    us2[i] = us;
    vs2[i] = vs;
    v2h sgT = kh2_splat((h16)0.0f), sg = sgT, maxE = sgT;
    each_E(us, vs, [&](int n, v2h E) {
      const v2h gg = ko_exp2h2(P, E);
      const v2h Tv = kh2_splat(T[n]);
      if (n == 0) {
        sgT = kh2_mul(gg, Tv);
        sg = gg;
        if (!FAST) maxE = E;
      } else {
        sgT = kh2_fma(gg, Tv, sgT);
        sg = kh2_add(gg, sg);
        if (!FAST) maxE = kh2_maxnum(maxE, E);
      }
    });
    if (FAST) {  // FP16 reciprocal estimates (v_rcp_f16) and one packed multiply
      const v2h q = kh2_mul(sgT, kh2(kh_rcp_fast(sg[0]), kh_rcp_fast(sg[1])));
      Fv[2 * i] = q[0];
      Fv[2 * i + 1] = q[1];
      // FAST: the weight sum instead of the largest E (no maxE tracking): sg >= 2^-8 needs some 2^E >= 2^-12, i.e. E >= -12
      // (twelve weights below 2^-12 sum to less than 12 2^-12, with the FP16 rounding of the sum still below 2^-8), so
      // the quotient stands as in the exact form; any other sum (small, or NaN) takes the exact best-E decision below
      plain = plain && sg[0] >= (h16)0.00390625f && sg[1] >= (h16)0.00390625f;
    } else {
#pragma unroll
      for (int j = 0; j < 2; ++j) Fv[j + 2 * i] = kh_div(sgT[j], sg[j]);
      plain = plain && maxE[0] >= (h16)-12.0f && maxE[1] >= (h16)-12.0f && sg[0] == sg[0] && sg[1] == sg[1];
    }
    const v2h dvf = kh2_sub(kh2_rni(vs), vs);
    const v2h up = kh2_fma(dvf, kh2_splat(tS), us);
    const v2h duf = kh2_sub(kh2_rni(up), up);
    // the fma results below are never -0 (addend 4.5), as kh2_sat_nz needs
    const v2h wu = kh2_sat_nz(kh2_fma(kh2_abs(duf), kh2_splat(k_u), kh2_splat((h16)4.5f)));
    const v2h wv = kh2_sat_nz(kh2_fma(kh2_abs(dvf), kh2_splat(k_v), kh2_splat((h16)4.5f)));
    gam2[i] = kh2_mul(kh2_fma(wv, kh2_splat(tK), kh2_splat(tk6)), wu);
    gam[2 * i] = gam2[i][0];
    gam[2 * i + 1] = gam2[i][1];
  }
  if (!plain)
#pragma unroll
    for (int i = 0; i < 2; ++i) {
      v2h bestE = us2[i], bestT = us2[i];
      each_E(us2[i], vs2[i], [&](int n, v2h E) {
        const v2h Tv = kh2_splat(T[n]);
        if (n == 0) {
          bestE = E;
          bestT = Tv;
        } else {
          const u32 keep = kh2_gtmask(bestE, E);
          bestE = kh2_bsel(keep, bestE, E);
          bestT = kh2_bsel(keep, bestT, Tv);
        }
      });
#pragma unroll
      for (int j = 0; j < 2; ++j)
        if (bestE[j] < (h16)-12.0f) Fv[j + 2 * i] = bestT[j];
    }
  // gbar = h(0.25 h(h(gamma00 + gamma01) + h(gamma10 + gamma11))), gamma_ji = gam[j + 2i]
  const h16 gbar = kh_mul((h16)0.25f, kh_add(kh_add(gam[0], gam[2]), kh_add(gam[1], gam[3])));
  const h16 mh = kh_round(m0);
  const h16 MU = (h16)0.03f, NU = (h16)(1.0 / 0.97);
  const h16 rr = kh_add((h16)0.1f, kh_mul((h16)0.9f, mh));
  const h16 d0 = kh_mul(kh_sub(mh, MU), NU);
  const h16 d1 = kh_mul(kh_sub(rr, MU), NU);
  const h16 beta_c = kh_sat(kh_fma(gbar, kh_sub(rr, d1), d1));

  // 6.2 guard
  h16 Ybar = cbar[0], Cobar = cbar[1], Cgbar = cbar[2];
  const float guard = ((((float)Ybar * (float)Fv[0]) * (float)Fv[1]) * (float)Fv[2]) * (float)Fv[3];
  if (!kf_finite(guard)) {
#pragma unroll
    for (int s = 0; s < 4; ++s) Fv[s] = (h16)0.0f;
    Ybar = Cobar = Cgbar = (h16)0.0f;
  }

  // 6.3 / 6.4 new luma, luma history out
  // beta per sub-sample, two at a time; d0 = (mh - MU) NU is never -0 (an exact zero difference is +0, a nonzero one
  // stays nonzero), so neither is the fma result, as kh2_sat_nz needs
  const v2h md2 = kh2_splat(kh_sub(mh, d0)), d02 = kh2_splat(d0);
  const v2h beta2[2] = {kh2_sat_nz(kh2_fma(gam2[0], md2, d02)), kh2_sat_nz(kh2_fma(gam2[1], md2, d02))};
  float Nn[4];
  h16 Lo[4];
#pragma unroll
  for (int s = 0; s < 4; ++s) {
    const h16 beta = beta2[s >> 1][s & 1];
    Nn[s] = kf_fma((float)beta, (float)Fv[s] - Hl[s], Hl[s]);
    const float c99 = kf_bits(0x3F7D70A4u);
    const float lv = Nn[s] < c99 ? Nn[s] : kf_fma(Nn[s] - c99, 1000.0f, c99);
    Lo[s] = kh_round(lv);
  }
  if (active)  // the two sub-samples of a row as one dword (2 x is even)
#pragma unroll
    for (int r = 0; r < 2; ++r) *(v2h*)kio_ptr(P.Lout, (u32)(2 * y + r) * LW + 2 * x) = kh2(Lo[2 * r], Lo[2 * r + 1]);

  // 6.5 colour
  const float bc = (float)beta_c;
  const float Yc = kf_fma(Hh[0], 1.0f - bc, bc * (float)Ybar);
  const float Coc = kf_fma(Hh[1], 1.0f - bc, bc * (float)Cobar);
  const float Cgc = kf_fma(Hh[2], 1.0f - bc, bc * (float)Cgbar);
  const float eps = kf_bits(0x3727C5ACu);
  const float q = FAST ? ko_rcp_fast(__builtin_fmaxf(Yc, eps)) : ko_rcp_output_bounded(__builtin_fmaxf(Yc, eps));
  const float rCo = Coc * q, rCg = Cgc * q;
  const float Yo = 0.25f * (Nn[3] + (Nn[2] + (Nn[0] + Nn[1])));
  const float Coo = rCo * Yo, Cgo = rCg * Yo;

  // 6.6 colour history out
  const float Yp = __builtin_fmaxf(Yo, 0.0f);
  const float s1 = (Cgo - eps < -Yp) ? eps - Cgo : Yp;
  const float s2 = ((Coo - Cgo) - eps < -s1) ? (Cgo - Coo) + eps : s1;
  const float s3 = ((-Coo - Cgo) - eps < -s2) ? (Coo + Cgo) + eps : s2;
  const float kk = s3 <= Yp ? 1.0f : (FAST ? Yp * ko_rcp_fast(s3) : Yp / s3);
  const float age = kf_sat(__builtin_fmaxf(FAST ? m1 * ko_rcp_fast(m1 + 1.0f) : m1 / (m1 + 1.0f), 0.01f));
  if (!active) return;
  const v4h hc = {kh_round(Yp), kh_round(Coo * kk), kh_round(Cgo * kk), kh_round(age)};
  *(v4h*)kio_ptr(P.Hcout, ((u32)y * W16 + x) * 4u) = hc;

  // 6.7 game output
  if (x < F.W && y < F.H) {
    float Yg = Yo, Cog = Coo, Cgg = Cgo;
    if (hdr) {
      // kio_decode with its near branch's 1 / (1 - t) as ko_rcp_refined (the correctly rounded reciprocal)
      const float d = Yo < kKioDecodeNear ? (FAST ? ko_rcp_fast(1.0f - Yo) : ko_rcp_refined(1.0f - Yo)) : kio_decode(Yo),
                  kinv = U.kinv;
      Yg = kinv * (Yo * d);
      Cog = kinv * (Coo * d);
      Cgg = kinv * (Cgo * d);
    }
    const float R = (Yg + Cog) - Cgg, G = Cgg + kf_fma(Cog, 0.0f, Yg), B = (Yg - Cog) - Cgg;
    if (P.omode == KIO_OUT_RGBA16F) {  // RGB rounded once to FP16 (nearest even), alpha 1
      const v4h o = {kh_round(R), kh_round(G), kh_round(B), (h16)1.0f};
      *(v4h*)kio_ptr((h16*)P.out, ((u32)y * P.opitch + x) * 4u) = o;
    } else if (P.omode == KIO_OUT_RGB32F) {
      float* o = kio_ptr(P.out, ((u32)y * P.opitch + x) * 3u);
      o[0] = R;
      o[1] = G;
      o[2] = B;
    } else {
      float* o = kio_ptr(P.out, ((u32)y * P.opitch + x) * 4u);
      o[0] = R;
      o[1] = G;
      o[2] = B;
      o[3] = (float)cbar[3];
    }
  }
}

WG_KERNEL_N(k_out_pixel, 128)(KOutParams P0) { ko_pixel<false>(P0); }
WG_KERNEL_N(k_out_pixel_c, 128)(KOutParams P0) { ko_pixel<true>(P0); }
// runtime option k_fast=1 (not bit-exact): the KIO_FAST forms. (Not compiled with real-true16: -12 us measured, but it
// changes the output of the exact arithmetic too, with the same instruction mix, by up to 0.02 in tonemapped luma at
// edges after one frame; not understood, so not used.)
WG_KERNEL_N(k_out_pixel_f, 128)(KOutParams P0) { ko_pixel<false, true>(P0); }
WG_KERNEL_N(k_out_pixel_c_f, 128)(KOutParams P0) { ko_pixel<true, true>(P0); }

WG_KERNEL_N(k_out_hidden, 128)(KOutParams P) {
  const KFrameIn& F = P.f;
  const int id = (int)(wg_block_id() * 128u + wg_lane_id());
  if (id >= F.Bw * F.Bh) return;
  const h16* n = kio_ptr(P.N, (u32)id * 40u);
  float hk[4];
#pragma unroll
  for (int k = 0; k < 4; ++k) {
    const h16* c = n + 10 * k + 3;
    hk[k] = 0.2f * (((((float)c[0] + (float)c[1]) + (float)c[2]) + (float)c[3]) + (float)c[4]);
  }
  float ss = hk[1] * hk[1];
  ss = kf_fma(hk[0], hk[0], ss);
  ss = kf_fma(hk[2], hk[2], ss);
  ss = kf_fma(hk[3], hk[3], ss);
  const float nu = kf_rsqrt(ss + 1.0f);  // 1/sqrt, one rounding
#pragma unroll
  for (int k = 0; k < 4; ++k) *kio_ptr(P.Hhout, (u32)id * 4u + k) = kh_round(hk[k] * nu);
}

// =====================================================================================================================
// input stage
// =====================================================================================================================

// Arithmetic of the input stage. Spec 42 section 9 (42-T1) accepts >= 99.9 % of the values within one FP16 step of
// the reference (the rest only at decision points), so the default build (k_io.co) uses the GPU's log2 / 2^x
// (v_log_f32 / v_exp_f32, about 1 ulp) and reciprocal estimates (v_rcp_f32) where a division feeds the FP16 outputs.
// WGK_IN_EXACT (k_io_exact.co, and the host emulator) keeps every operation correctly rounded: bit for bit the NumPy
// reference (clean/k/reference/instage.py).
#if defined(__AMDGCN__) && !defined(WGK_IN_EXACT)
#define KI_FAST 1
#else
#define KI_FAST 0
#endif
WG_FN float ki_log2(float x) {
#if KI_FAST
  return __builtin_amdgcn_logf(x);
#else
  return kf_log2(x);
#endif
}
WG_FN float ki_exp2(float x) {
#if KI_FAST
  return __builtin_amdgcn_exp2f(x);
#else
  return kf_exp2(x);
#endif
}
WG_FN float ki_rcp(float x) {
#if KI_FAST
  return __builtin_amdgcn_rcpf(x);
#else
  return 1.0f / x;
#endif
}
WG_FN float ki_div(float a, float b) {
#if KI_FAST
  return a * __builtin_amdgcn_rcpf(b);
#else
  return a / b;
#endif
}
// spec 42 decode factor D(t) (as kio_decode, with the input stage's arithmetic)
WG_FN float ki_decode(float t) {
  if (t < kf_bits(0x3F7FBE88u)) return ki_rcp(1.0f - t);
  const float e = ((t + kf_bits(0x3C4FAE29u)) * 9.0f) * kf_bits(0x3FB8AA3Bu);
  return ki_div(ki_exp2(e) - 1.0f, 9.0f * t);
}

struct KiFine {  // spec 42 section 5 per fine position
  bool valid;
  float h[3];    // reprojected history (Y, Co, Cg)
  float nx, ny;  // hidden-history position (normalized)
};

WG_FN float ki_phi(float Y, bool hdr) {
  if (hdr) {
    const float Yp = kf_max(Y, 0.0f);
    return ki_div(ki_log2(kf_fma(Yp, 12.0f, 1.0f)) * kf_bits(0x3F317218u), Yp + kf_bits(0x358637BDu));
  }
  return kf_fma(Y, 3.0f, 1.0f);
}

struct KiHost {  // spec 42 section 8
  float xi_x, xi_y, a_x, a_y, h_x, h_y, g_x, g_y, sp_x, sp_y, mp_x, mp_y;
};

WG_FN KiHost ki_host(const KFrameIn& F) {
  KiHost k;
  const int W16 = kio_ceil16(F.W), H16 = kio_ceil16(F.H);
  k.xi_x = (1.0f / ((float)F.W * F.s)) * 0.5f;  // spec 42 section 8: RN32(RN32(1/RN32(W s)) 0.5)
  k.xi_y = (1.0f / ((float)F.H * F.s)) * 0.5f;
  k.a_x = (float)F.rw * k.xi_x;
  k.a_y = (float)F.rh * k.xi_y;
  k.h_x = ((float)F.W / (float)W16) * k.xi_x;
  k.h_y = ((float)F.H / (float)H16) * k.xi_y;
  k.g_x = 0.25f / (float)F.Bw;
  k.g_y = 0.25f / (float)F.Bh;
  const bool mvlo = (F.flags & KIO_MVLO) != 0;
  k.sp_x = F.sx / ((1.0f / (float)F.W) * (float)(mvlo ? F.rw : F.W));
  k.sp_y = F.sy / ((1.0f / (float)F.H) * (float)(mvlo ? F.rh : F.H));
  k.mp_x = k.xi_x * (float)F.W;
  k.mp_y = k.xi_y * (float)F.H;
  return k;
}

// FAST (the "_f" kernels): the output stage's address forms of the same loads (ko_nearest, ko_motion, ko_mtap,
// ko_bilinear4_load / _math: the same values), and the decode factor's near branch as the reciprocal estimate (as
// KI_FAST already has it)
template <bool FAST = false>
WG_FN KiFine ki_fine(const KInParams& P, const KiHost& k, int fx, int fy) {
  const KFrameIn& F = P.f;
  const int W16 = kio_ceil16(F.W), H16 = kio_ceil16(F.H);
  const float px = (float)fx + 0.5f, py = (float)fy + 0.5f;
  float mvx, mvy;
  if (F.flags & KIO_MVLO) {
    const int ix = (int)kf_R(F.jx + kf_fma(px, k.a_x, -0.5f)), iy = (int)kf_R(F.jy + kf_fma(py, k.a_y, -0.5f));
    int dx, dy;
    if (FAST) {
      ko_nearest(F, ix, iy, &dx, &dy);
      ko_motion(F, ko_mtap(ix, dx, F.mbx, F.rw), ko_mtap(iy, dy, F.mby, F.rh), &mvx, &mvy);
    } else {
      kio_nearest(F, ix, iy, &dx, &dy);
      kio_motion(F, kio_mtap(ix, dx, F.mbx, F.rw), kio_mtap(iy, dy, F.mby, F.rh), &mvx, &mvy);
    }
  } else {
    const int mx = (int)kf_R(kf_fma(px, k.mp_x, -0.5f)), my = (int)kf_R(kf_fma(py, k.mp_y, -0.5f));
    kio_motion(F, kio_tap(mx, 0, F.mbx, F.W), kio_tap(my, 0, F.mby, F.H), &mvx, &mvy);
  }
  const float mx = k.sp_x * (mvx + F.mox), my = k.sp_y * (mvy + F.moy);
  const float wx = px + mx, wy = py + my;
  KiFine r;
  r.valid = wx * k.xi_x >= 0.0f && wx * k.xi_x <= 1.0f && wy * k.xi_y >= 0.0f && wy * k.xi_y <= 1.0f;
  const v4f s = FAST ? ko_bilinear4_math(ko_bilinear4_load(P.Hc, W16, H16, wx * k.h_x, wy * k.h_y))
                     : kio_bilinear4(P.Hc, W16, H16, wx * k.h_x, wy * k.h_y);
  if (F.flags & KIO_HDR) {
    const float f = ki_decode(s[0]);
#pragma unroll
    for (int c = 0; c < 3; ++c) r.h[c] = F.rho * (s[c] * f);
  } else {
#pragma unroll
    for (int c = 0; c < 3; ++c) r.h[c] = s[c];
  }
  r.nx = ((float)(4 * (fx / 4) + 2) + mx) * k.g_x;
  r.ny = ((float)(4 * (fy / 4) + 2) + my) * k.g_y;
  return r;
}

WG_FN float ki_S(float z) {
  const float zp = z != z ? 0.0f : kf_min(kf_max(z, 0.0f), 1.0f);
  if (!(zp > kf_bits(0x3B4D2E1Cu))) return kf_bits(0x414EB852u) * zp;
  const float p = ki_exp2(ki_log2(zp) * kf_bits(0x3ED55555u));  // log2 and 2^x each rounded once (spec 42 draft 2)
  return kf_fma(p, kf_bits(0x3F870A3Du), kf_bits(0xBD6147AEu));
}
WG_FN float ki_T(float z) { return z * kf_max(ki_rcp(z + 1.0f), kf_bits(0x38D1B717u)); }
WG_FN float ki_L(float Y, bool hdr) { return hdr ? Y : ki_div(Y, kf_max(1.0f - Y, kf_bits(0x3E4CCCCDu))); }

// Per render pixel (i, j) of the colour texture: the operations k_in_base would do for a colour tap (spec 42 6.2) or an
// own sample (slot channel 0) at clamped position (i, j); k_in_base reads them, so its results are unchanged.
WG_KERNEL_N(k_in_color, 128)(KInParams P0) {
  const KInParams P = ki_params(P0);
  const KFrameIn& F = P.f;
  const int id = (int)(wg_block_id() * 128u + wg_lane_id());
  if (id >= F.rw * F.rh) return;
  const int i = id % F.rw, j = id / F.rw;
  const bool hdr = (F.flags & KIO_HDR) != 0;
  const v4f c = kio_texel(F, F.cbx + i, F.cby + j);
  const float R = F.kappa * kf_max(c[0], 0.0f), G = F.kappa * kf_max(c[1], 0.0f), B = F.kappa * kf_max(c[2], 0.0f);
  float Y, Co, Cg;
  kio_ycocg(R, G, B, &Y, &Co, &Cg);
  const v4f v = {Y, Co, Cg, ki_phi(Y, hdr)};
  *(v4f*)kio_ptr(P.cpre, (u32)id * 4u) = v;
  const float Yc = kio_luma_fma(R, G, B);
  *kio_ptr(P.cpre2, (u32)id * 2u) = hdr ? kf_max(ki_rcp(Y + 1.0f), kf_bits(0x38D1B717u)) : 0.0f;
  *kio_ptr(P.cpre2, (u32)id * 2u + 1u) = ki_S(ki_T(F.lam * ki_L(Yc, hdr)));
}

// On reset, spec 42 sections 6.1, 6.5 and 7 make channel 1 equal channel 0 and
// channels 2/3 zero. Only the own current-colour sample is needed; no histories,
// motion, depth, colour box, disagreement or LDS exchange contribute to x0.
// Separate entry point keeps the steady-frame kernel and its resource use intact.
WG_KERNEL_N(k_in_reset, 128)(KInParams P) {
  const KFrameIn& F = P.f;
  const int lane = (int)wg_lane_id();
  const int id = (int)(wg_block_id() * 32u) + (lane >> 2), k = lane & 3;
  if (id >= F.Bw * F.Bh) return;
  const int bx = id % F.Bw, by = id / F.Bw;
  const int u = 2 * bx + (k & 1), v = 2 * by + (k >> 1);
  const KiHost hk = ki_host(F);
  const int fx = 2 * u + 1 - (v & 1), fy = 2 * v + (u & 1);
  const int rx = (int)kf_R(F.jx + kf_fma((float)fx + 0.5f, hk.a_x, -0.5f));
  const int ry = (int)kf_R(F.jy + kf_fma((float)fy + 0.5f, hk.a_y, -0.5f));
  const float ch0 = *kio_ptr(P.cpre2, ((u32)kio_tap(ry, 0, 0, F.rh) * F.rw + kio_tap(rx, 0, 0, F.rw)) * 2u + 1u);
  const h16 h = kh_round(ch0);
  const v4h out = {h, h, (h16)0, (h16)0};
  *(v4h*)kio_ptr(P.x0, (u32)id * 16u + 4u * k) = out;
}

// One lane per display position: lanes 4b .. 4b + 3 are the display positions k = 0..3 of base position b (32 base
// positions per workgroup). The hidden-history channel averages the 4 display positions; they meet in LDS.
WG_KERNEL_N(k_in_base, 128)(KInParams P0) {
  const KInParams P = ki_params(P0);
  const KFrameIn& F = P.f;
  const int lane = (int)wg_lane_id();
  const int id = (int)(wg_block_id() * 32u) + (lane >> 2), k = lane & 3;
  const bool active = id < F.Bw * F.Bh;  // inactive lanes still reach the barrier
  const int bx = active ? id % F.Bw : 0, by = active ? id / F.Bw : 0;
  const bool hdr = (F.flags & KIO_HDR) != 0;
  const KiHost hk = ki_host(F);
  const float c1e4 = kf_bits(0x38D1B717u);
  float ch0 = 0.0f, ch1 = 0.0f, ch3 = 0.0f;
  float hid[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  WG_LDS float lds_hid[128][4];
  if (active) {
    const int u = 2 * bx + (k & 1), v = 2 * by + (k >> 1);
    // fine positions of the display position, in the order (2u, 2v), (2u+1, 2v), (2u, 2v+1), (2u+1, 2v+1)
    KiFine fp[4];
    bool valid = !(F.flags & KIO_RESET);
#pragma unroll
    for (int q = 0; q < 4; ++q) {
      fp[q] = ki_fine(P, hk, 2 * u + (q & 1), 2 * v + (q >> 1));
      valid = valid && fp[q].valid;
    }
    // colour neighbourhood
    const int cx = (int)kf_R(F.jx + kf_fma((float)(2 * u + 1), hk.a_x, -0.5f));
    const int cy = (int)kf_R(F.jy + kf_fma((float)(2 * v + 1), hk.a_y, -0.5f));
    float A[3] = {0.0f, 0.0f, 0.0f}, lo[3], hi[3];
#pragma unroll
    for (int ddy = -1; ddy <= 1; ++ddy)
#pragma unroll
      for (int ddx = -1; ddx <= 1; ++ddx) {
        // the tap's values from k_in_color (render pixel kio_tap(.., 0, n) of the subrect)
        const u32 t = (u32)kio_tap(cy, ddy, 0, F.rh) * F.rw + kio_tap(cx, ddx, 0, F.rw);
        const v4f cv = *(const v4f*)kio_ptr(P.cpre, t * 4u);
        const float vv[3] = {cv[0], cv[1], cv[2]};
        if (hdr) {
          const float om = *kio_ptr(P.cpre2, t * 2u);
#pragma unroll
          for (int c2 = 0; c2 < 3; ++c2) A[c2] = kf_fma(om, vv[c2], A[c2]);
        } else {
#pragma unroll
          for (int c2 = 0; c2 < 3; ++c2) A[c2] = A[c2] + vv[c2];
        }
        const float ph = cv[3];
        const bool first = ddx == -1 && ddy == -1;
#pragma unroll
        for (int c2 = 0; c2 < 3; ++c2) {
          const float val = vv[c2] * ph;
          lo[c2] = first ? val : __builtin_fminf(lo[c2], val);
          hi[c2] = first ? val : __builtin_fmaxf(hi[c2], val);
        }
      }
    float mu[3];
#pragma unroll
    for (int c2 = 0; c2 < 3; ++c2) mu[c2] = A[c2] * kf_bits(0x3DE38E39u);
    if (hdr) {
      const float den = kf_max(1.0f - mu[0], c1e4);
#pragma unroll
      for (int c2 = 0; c2 < 3; ++c2) mu[c2] = ki_div(mu[c2], den);
    }
    const float pm = ki_phi(mu[0], hdr);
    float D = 0.0f;
#pragma unroll
    for (int q = 0; q < 4; ++q) {
      const float ph = ki_phi(fp[q].h[0], hdr);
      float dl[3];
#pragma unroll
      for (int c2 = 0; c2 < 3; ++c2) {
        const float eta = 0.5f * (hi[c2] - lo[c2]);
        const float nu = kf_fma(hi[c2] - lo[c2], 8.0f, 1.0f);
        dl[c2] = ki_div(__builtin_fmaxf(__builtin_fabsf(mu[c2] * pm - fp[q].h[c2] * ph) - eta, 0.0f), nu);
      }
      const float score = __builtin_fminf(kf_fma(dl[2], 0.5f, kf_fma(dl[0], 2.0f, 0.5f * dl[1])), 1.0f);
      D = D + score;
    }
    D = D * 0.25f;
    // own sample
    const int oq = (1 - (v & 1)) + 2 * (u & 1);  // (2u + 1 - (v mod 2), 2v + (u mod 2)) as an index into fp
    const int ofx = 2 * u + 1 - (v & 1), ofy = 2 * v + (u & 1);
    const int rx = (int)kf_R(F.jx + kf_fma((float)ofx + 0.5f, hk.a_x, -0.5f));
    const int ry = (int)kf_R(F.jy + kf_fma((float)ofy + 0.5f, hk.a_y, -0.5f));
    // channel 0 of the own sample (render pixel (rx, ry)) from k_in_color
    ch0 = *kio_ptr(P.cpre2, ((u32)kio_tap(ry, 0, 0, F.rh) * F.rw + kio_tap(rx, 0, 0, F.rw)) * 2u + 1u);
    KiFine own = fp[0];  // fp[oq] by selects (a dynamic index would go to scratch memory)
#pragma unroll
    for (int q = 1; q < 4; ++q)
      if (q == oq) own = fp[q];
    const v4f hs = kio_bilinear4(P.Hh, F.Bw, F.Bh, own.nx, own.ny);
#pragma unroll
    for (int c2 = 0; c2 < 4; ++c2) hid[c2] = valid ? hs[c2] : 0.0f;
    // channel 1 is channel 0's function of Yh, and Yh = Yc without a valid history: then it equals channel 0
    ch1 = valid ? ki_S(ki_T(F.lam * ki_L(own.h[0], hdr))) : ch0;
    const float d3 = valid ? D : 0.0f;
    ch3 = kf_finite(d3) ? d3 : 0.0f;
  }
#pragma unroll
  for (int c2 = 0; c2 < 4; ++c2) lds_hid[lane][c2] = hid[c2];
  wg_barrier_lds();
  if (!active) return;
  // slot k, channel 2: 0.25 ((s0 + s3) + (s1 + s2)), s_m = channel k of slot m's hidden-history sample
  const int g = lane & ~3;
  float hsum = (lds_hid[g][k] + lds_hid[g + 3][k]) + (lds_hid[g + 1][k] + lds_hid[g + 2][k]);
  hsum = hsum * 0.25f;
  h16* o = kio_ptr(P.x0, (u32)id * 16u + 4u * k);
  o[0] = kh_round(ch0);
  o[1] = kh_round(ch1);
  o[2] = kh_round(kf_finite(hsum) ? hsum : 0.0f);
  o[3] = kh_round(ch3);
}



#if defined(__AMDGCN__)
// Complete-frame input fusion only: an 8x8 base-token group has 256 display
// positions. Cooperatively stage a 24x24 render-colour footprint as aligned SoA
// (Y,Co,Cg,phi plus omega), then preserve each slot's original input arithmetic.
// The planner bounds ratios/jitter/coordinate sizes; every cached lookup is also
// checked, with the exact raw-tap computation for an unexpected wider footprint.
// Other frames retain both original kernels. X0 remains a complete global map.
// Own-sample transfer is computed only for actual slots. Four wave shuffles
// transpose the hidden-history samples, preserving the specified addition order.
struct KiTiledTap {
  v4f cv;
  float om;
};
WG_FN KiTiledTap ki_tiled_tap_from_raw(const KFrameIn& F, v4f c) {
  const float R = F.kappa * kf_max(c[0], 0.0f), G = F.kappa * kf_max(c[1], 0.0f), B = F.kappa * kf_max(c[2], 0.0f);
  float Y, Co, Cg;
  kio_ycocg(R, G, B, &Y, &Co, &Cg);
  const bool hdr = (F.flags & KIO_HDR) != 0;
  KiTiledTap r;
  r.cv = {Y, Co, Cg, ki_phi(Y, hdr)};
  r.om = hdr ? kf_max(ki_rcp(Y + 1.0f), kf_bits(0x38D1B717u)) : 0.0f;
  return r;
}

WG_FN KiTiledTap ki_tiled_tap(const KFrameIn& F, int rx, int ry) {
  return ki_tiled_tap_from_raw(F, kio_color(F, rx, ry));
}
WG_FN float ki_tiled_ch0(const KFrameIn& F, int rx, int ry) {
  const v4f c = kio_color(F, rx, ry);
  const bool hdr = (F.flags & KIO_HDR) != 0;
  const float R = F.kappa * kf_max(c[0], 0.0f), G = F.kappa * kf_max(c[1], 0.0f), B = F.kappa * kf_max(c[2], 0.0f);
  const float Yc = kio_luma_fma(R, G, B);
  return ki_S(ki_T(F.lam * ki_L(Yc, hdr)));
}
WG_FN float ki_tiled_xor(float v, int mask) {
  return __builtin_bit_cast(
      float, __builtin_amdgcn_ds_bpermute(((wg_lane_id() ^ mask) & 63u) << 2, __builtin_bit_cast(int, v)));
}
WG_FN v4f ki_tiled_transpose(v4f v, int lane) {
  const bool b0 = (lane & 1) != 0, b1 = (lane & 2) != 0;
  const float x = ki_tiled_xor(b0 ? v[0] : v[1], 1), y = ki_tiled_xor(b0 ? v[2] : v[3], 1);
  const v4f a = {b0 ? x : v[0], b0 ? v[1] : x, b0 ? y : v[2], b0 ? v[3] : y};
  const float z = ki_tiled_xor(b1 ? a[0] : a[2], 2), w = ki_tiled_xor(b1 ? a[1] : a[3], 2);
  return {b1 ? z : a[0], b1 ? w : a[1], b1 ? a[2] : z, b1 ? a[3] : w};
}
// Explicit additional output destinations; KInParams stays unchanged.
struct KInColorParams { KInParams in; h16* ctap; h16* ctapT; };
static_assert(sizeof(KInColorParams) == 216, "tiled color arguments");

template <bool FAST = false>
WG_FN void ki_tiled_store_output_color(const KFrameIn& F, v4f color, int id, h16* ctap, h16* ctapT) {
  const KoTap t = ko_stored_tap_from_raw<FAST>(F, color);
  const v4h packed = {t.Y, t.Co, t.Cg, t.A};
  *(v4h*)kio_ptr(ctap, (u32)id * 4u) = packed;
  // T intentionally retains the distinct fused-luma ordering of k_out_color.
  const float kY = F.kappa * kio_luma_fma(kf_max(color[0], 0.0f), kf_max(color[1], 0.0f), kf_max(color[2], 0.0f));
  ctapT[id] = kh_round((F.flags & KIO_HDR) ? kY * kio_encode_f<FAST>(kY) : kY);
}

template <bool StoreColor, bool Precomputed = false, bool FAST = false>
WG_FN void ki_tiled_body(KInParams P0, h16* ctap, h16* ctapT, KiHost supplied = {}) {
  const KInParams P = ki_params(P0);
  const KFrameIn& F = P.f;
  const int lane = (int)wg_lane_id();
  const int tiles_x = (F.Bw + 7) / 8, tile = (int)wg_block_id();
  const int bx0 = (tile % tiles_x) * 8, by0 = (tile / tiles_x) * 8;
  const int k = lane & 3, local = lane >> 2;
  const int bx = bx0 + (local & 7), by = by0 + (local >> 3);
  const bool active = bx < F.Bw && by < F.Bh;
  const int id = by * F.Bw + bx;
  const bool hdr = (F.flags & KIO_HDR) != 0;
  const KiHost hk = Precomputed ? supplied : ki_host(F);
  const float c1e4 = kf_bits(0x38D1B717u);
  float ch0 = 0.0f, ch1 = 0.0f, ch3 = 0.0f;
  float hid[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  if (F.flags & KIO_RESET) {
    if (active) {
      const int u = 2 * bx + (k & 1), v = 2 * by + (k >> 1);
      const int ofx = 2 * u + 1 - (v & 1), ofy = 2 * v + (u & 1);
      const int rx = (int)kf_R(F.jx + kf_fma((float)ofx + 0.5f, hk.a_x, -0.5f));
      const int ry = (int)kf_R(F.jy + kf_fma((float)ofy + 0.5f, hk.a_y, -0.5f));
      const h16 h = kh_round(ki_tiled_ch0(F, rx, ry));
      const v4h out = {h, h, (h16)0, (h16)0};
      *(v4h*)kio_ptr(P.x0, (u32)id * 16u + 4u * k) = out;
    }
    return;
  }
  // SoA preserves 16-byte alignment for every v4f; no 5-float AoS cast.
  WG_LDS __attribute__((aligned(16))) v4f tile_cv[24 * 24];
  WG_LDS float tile_om[24 * 24];
  const int tx0 = ki_max(ki_min((int)kf_R(F.jx + kf_fma((float)(4 * bx0 + 1), hk.a_x, -0.5f)) - 1, F.rw - 1), 0);
  const int ty0 = ki_max(ki_min((int)kf_R(F.jy + kf_fma((float)(4 * by0 + 1), hk.a_y, -0.5f)) - 1, F.rh - 1), 0);
  int own_x1 = 0, own_y1 = 0;
  if constexpr (StoreColor) {
    own_x1 = bx0 + 8 < F.Bw ? ki_max(ki_min((int)kf_R(F.jx + kf_fma((float)(4 * (bx0 + 8) + 1), hk.a_x, -0.5f)) - 1, F.rw - 1), 0) : F.rw;
    own_y1 = by0 + 8 < F.Bh ? ki_max(ki_min((int)kf_R(F.jy + kf_fma((float)(4 * (by0 + 8) + 1), hk.a_y, -0.5f)) - 1, F.rh - 1), 0) : F.rh;
  }
  for (int q = lane; q < 24 * 24; q += 256) {
    const int raw_x = tx0 + q % 24, raw_y = ty0 + q / 24;
    const int rx = ki_min(raw_x, F.rw - 1), ry = ki_min(raw_y, F.rh - 1);
    const v4f color = kio_color(F, rx, ry);
    const KiTiledTap value = ki_tiled_tap_from_raw(F, color);
    tile_cv[q] = value.cv;
    tile_om[q] = value.om;
    if constexpr (StoreColor) {
      // Consecutive origins partition logical render coordinates. Test the raw
      // producer coordinates so replicated clamp loads never duplicate stores.
      if (raw_x < own_x1 && raw_y < own_y1)
        ki_tiled_store_output_color<FAST>(F, color, raw_y * F.rw + raw_x, ctap, ctapT);
    }
  }
  wg_barrier_lds();
  if (active) {
    const int u = 2 * bx + (k & 1), v = 2 * by + (k >> 1);
    // fine positions of the display position, in the order (2u, 2v), (2u+1, 2v), (2u, 2v+1), (2u+1, 2v+1)
    KiFine fp[4];
    bool valid = !(F.flags & KIO_RESET);
#pragma unroll
    for (int q = 0; q < 4; ++q) {
      fp[q] = ki_fine<FAST>(P, hk, 2 * u + (q & 1), 2 * v + (q >> 1));
      valid = valid && fp[q].valid;
    }
    // colour neighbourhood
    const int cx = (int)kf_R(F.jx + kf_fma((float)(2 * u + 1), hk.a_x, -0.5f));
    const int cy = (int)kf_R(F.jy + kf_fma((float)(2 * v + 1), hk.a_y, -0.5f));
    float A[3] = {0.0f, 0.0f, 0.0f}, lo[3], hi[3];
#pragma unroll
    for (int ddy = -1; ddy <= 1; ++ddy)
#pragma unroll
      for (int ddx = -1; ddx <= 1; ++ddx) {
        // the tap's values from k_in_color (render pixel kio_tap(.., 0, n) of the subrect)
        const int rx = kio_tap(cx, ddx, 0, F.rw), ry = kio_tap(cy, ddy, 0, F.rh);
        const int lx = rx - tx0, ly = ry - ty0;
        // The planner only selects supported geometry. This defensive path
        // also keeps direct synthetic calls correct if their footprint is wider.
        const bool cached = (unsigned)lx < 24u && (unsigned)ly < 24u;
        KiTiledTap fallback;
        if (!cached) fallback = ki_tiled_tap(F, rx, ry);
        const v4f cv = cached ? tile_cv[ly * 24 + lx] : fallback.cv;
        const float vv[3] = {cv[0], cv[1], cv[2]};
        if (hdr) {
          const float om = cached ? tile_om[ly * 24 + lx] : fallback.om;
#pragma unroll
          for (int c2 = 0; c2 < 3; ++c2) A[c2] = kf_fma(om, vv[c2], A[c2]);
        } else {
#pragma unroll
          for (int c2 = 0; c2 < 3; ++c2) A[c2] = A[c2] + vv[c2];
        }
        const float ph = cv[3];
        const bool first = ddx == -1 && ddy == -1;
#pragma unroll
        for (int c2 = 0; c2 < 3; ++c2) {
          const float val = vv[c2] * ph;
          lo[c2] = first ? val : __builtin_fminf(lo[c2], val);
          hi[c2] = first ? val : __builtin_fmaxf(hi[c2], val);
        }
      }
    float mu[3];
#pragma unroll
    for (int c2 = 0; c2 < 3; ++c2) mu[c2] = A[c2] * kf_bits(0x3DE38E39u);
    if (hdr) {
      const float den = kf_max(1.0f - mu[0], c1e4);
#pragma unroll
      for (int c2 = 0; c2 < 3; ++c2) mu[c2] = ki_div(mu[c2], den);
    }
    const float pm = ki_phi(mu[0], hdr);
    float D = 0.0f;
#pragma unroll
    for (int q = 0; q < 4; ++q) {
      const float ph = ki_phi(fp[q].h[0], hdr);
      float dl[3];
#pragma unroll
      for (int c2 = 0; c2 < 3; ++c2) {
        const float eta = 0.5f * (hi[c2] - lo[c2]);
        const float nu = kf_fma(hi[c2] - lo[c2], 8.0f, 1.0f);
        dl[c2] = ki_div(__builtin_fmaxf(__builtin_fabsf(mu[c2] * pm - fp[q].h[c2] * ph) - eta, 0.0f), nu);
      }
      const float score = __builtin_fminf(kf_fma(dl[2], 0.5f, kf_fma(dl[0], 2.0f, 0.5f * dl[1])), 1.0f);
      D = D + score;
    }
    D = D * 0.25f;
    // own sample
    const int oq = (1 - (v & 1)) + 2 * (u & 1);  // (2u + 1 - (v mod 2), 2v + (u mod 2)) as an index into fp
    const int ofx = 2 * u + 1 - (v & 1), ofy = 2 * v + (u & 1);
    const int rx = (int)kf_R(F.jx + kf_fma((float)ofx + 0.5f, hk.a_x, -0.5f));
    const int ry = (int)kf_R(F.jy + kf_fma((float)ofy + 0.5f, hk.a_y, -0.5f));
    // channel 0 of the own sample (render pixel (rx, ry)) from k_in_color
    ch0 = ki_tiled_ch0(F, rx, ry);
    KiFine own = fp[0];  // fp[oq] by selects (a dynamic index would go to scratch memory)
#pragma unroll
    for (int q = 1; q < 4; ++q)
      if (q == oq) own = fp[q];
    const v4f hs = FAST ? ko_bilinear4_math(ko_bilinear4_load(P.Hh, F.Bw, F.Bh, own.nx, own.ny))
                        : kio_bilinear4(P.Hh, F.Bw, F.Bh, own.nx, own.ny);
#pragma unroll
    for (int c2 = 0; c2 < 4; ++c2) hid[c2] = valid ? hs[c2] : 0.0f;
    // channel 1 is channel 0's function of Yh, and Yh = Yc without a valid history: then it equals channel 0
    ch1 = valid ? ki_S(ki_T(F.lam * ki_L(own.h[0], hdr))) : ch0;
    const float d3 = valid ? D : 0.0f;
    ch3 = kf_finite(d3) ? d3 : 0.0f;
  }
  const v4f hv = {hid[0], hid[1], hid[2], hid[3]};
  const v4f hs = ki_tiled_transpose(hv, lane);
  if (!active) return;
  float hsum = (hs[0] + hs[3]) + (hs[1] + hs[2]);
  hsum = hsum * 0.25f;
  h16* o = kio_ptr(P.x0, (u32)id * 16u + 4u * k);
  o[0] = kh_round(ch0);
  o[1] = kh_round(ch1);
  o[2] = kh_round(kf_finite(hsum) ? hsum : 0.0f);
  o[3] = kh_round(ch3);
}

WG_KERNEL_N(k_in_tiled, 256)(KInParams P0) {
  ki_tiled_body<false>(P0, nullptr, nullptr);
}

// Caller selects only nonreset frames satisfying the original tiled guard and
// 4*Bw >= W, 4*Bh >= H; all other frames retain separate k_out_color/input.
WG_KERNEL_N(k_in_tiled_color, 256)(KInColorParams P0) {
  ki_tiled_body<true>(P0.in, P0.ctap, P0.ctapT);
}

// Explicit frame-uniform extensions. Existing public argument structs/entries stay unchanged.
struct KInUniformParams { KInParams in; KiHost host; };
struct KInColorUniformParams { KInColorParams in; KiHost host; };
static_assert(sizeof(KiHost) == 48, "input uniforms");
static_assert(sizeof(KInUniformParams) == 248, "plain tiled uniform arguments");
static_assert(sizeof(KInColorUniformParams) == 264, "color tiled uniform arguments");
static_assert(__builtin_offsetof(KInUniformParams, host) == 200, "plain input uniform offset");
static_assert(__builtin_offsetof(KInColorUniformParams, host) == 216, "color input uniform offset");
WG_KERNEL_N(k_in_tiled_uniform, 256)(KInUniformParams P0) {
  ki_tiled_body<false, true>(P0.in, nullptr, nullptr, P0.host);
}
WG_KERNEL_N(k_in_tiled_color_uniform, 256)(KInColorUniformParams P0) {
  ki_tiled_body<true, true>(P0.in.in, P0.in.ctap, P0.in.ctapT, P0.host);
}
// runtime option k_fast=1 (not bit-exact): the FAST forms of the four tiled input kernels
WG_KERNEL_N(k_in_tiled_f, 256)(KInParams P0) { ki_tiled_body<false, false, true>(P0, nullptr, nullptr); }
WG_KERNEL_N(k_in_tiled_color_f, 256)(KInColorParams P0) { ki_tiled_body<true, false, true>(P0.in, P0.ctap, P0.ctapT); }
WG_KERNEL_N(k_in_tiled_uniform_f, 256)(KInUniformParams P0) {
  ki_tiled_body<false, true, true>(P0.in, nullptr, nullptr, P0.host);
}
WG_KERNEL_N(k_in_tiled_color_uniform_f, 256)(KInColorUniformParams P0) {
  ki_tiled_body<true, true, true>(P0.in.in, P0.in.ctap, P0.in.ctapT, P0.host);
}

#endif

// =====================================================================================================================
// exposure scalars on the GPU (the per-frame runtime, clean/k/runtime/runtime_k.cpp)
// =====================================================================================================================

// Host-side arguments of k_expo_scalars (must match KExpoArgs in clean/k/runtime/runtime_k.cpp and k_io.py).
struct KExpoArgs {
  float value;   // pair mode: this frame's exposure value (when use_src = 0)
  int pair;      // 1: update the exposure pair first (as kernels/exposure.hip exposure_pair: e = src[0] or value,
                 //    e_prev = e on the first frame, else the previous e); 0: the pair is already written
                 //    (auto-exposure: exposure_auto_diet)
  int use_src, first;
  int hdr, reset;
  float e1, e2;  // exposure factors of spec 42 section 4 (1 unless the game supplies an exposure scale)
  float inv_pi;  // RN32(1 / pi), pi the pre-exposure (host)
};
static_assert(sizeof(KExpoArgs) == 36, "KExpoArgs");

// Spec 42 section 4 / spec 43 section 3.7 from the exposure pair ex = [e, e_prev, last e] (the model M pair buffer of
// the runtime, spec 07 / spec 10 section 3): E_cur = e, E_prev = e_prev (E_cur on a reset), E_aux = E_cur (the
// runtime has one exposure value per frame: spec 07's result, the game's texel or value), each read as FP16 (nearest
// even) with 0 -> 1; then exactly the FP32 operations of clean/k/runner/k_io.py scalars() and out_uniforms()
// (kinv = RN(1/kappa); FP32 division is correctly rounded on the GPU). Writes xs = (kappa, rho, lambda, kinv), which
// the stage kernels read through KFrameIn.xs. One lane.
WG_FN float kx_ev(float e) {
  const float h = (float)kh_round(e);
  return h == 0.0f ? 1.0f : h;
}
WG_KERNEL_N(k_expo_scalars, 32)(float* ex, const float* src, float* xs, KExpoArgs A) {
  if (wg_lane_id() != 0) return;
  if (A.pair) {
    const float e = A.use_src ? src[0] : A.value;
    const float e_prev = A.first ? e : ex[2];
    ex[0] = e;
    ex[1] = e_prev;
    ex[2] = e;
  }
  const float Ec = kx_ev(ex[0]), Ep = kx_ev(A.reset ? ex[0] : ex[1]), Ea = Ec;
  const float ce = A.e1 * Ec;
  float kappa = 1.0f, rho = 1.0f;
  if (A.hdr) {
    kappa = A.inv_pi * ce;
    if (kappa == 0.0f) kappa = 1.0f;
    rho = ce / (A.e1 * Ep);
  }
  const float lam = (A.e2 * Ea) / ce;
  xs[0] = kappa;
  xs[1] = rho;
  xs[2] = lam;
  xs[3] = 1.0f / kappa;
}

WG_EMU_REGISTER(k_expo_scalars)
WG_EMU_REGISTER(k_out_color)
WG_EMU_REGISTER(k_out_pixel)
WG_EMU_REGISTER(k_out_pixel_c)
WG_EMU_REGISTER(k_out_hidden)
WG_EMU_REGISTER(k_in_color)
WG_EMU_REGISTER(k_in_base)
WG_EMU_REGISTER(k_in_reset)
