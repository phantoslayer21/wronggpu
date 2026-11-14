// SPDX-License-Identifier: MIT
// Model L (issue #89): the per-pixel and per-token matrix kernels of Model L's encoder and decoder
// (clean/l/FORMATS.md: maps, blobs, arguments, grids). Maps are [rows][cols][C], channel fastest; FP8 E4M3 bytes or
// FP16. Launch: 1-D grid of ceil(n / 128) workgroups of 256 lanes (8 waves); wave i handles items 16 i .. 16 i + 15
// (pixels or tokens in raster order); lane (t, h) works on item 16 i + t, and its accumulator element v of output tile
// mt is output channel h C/2 + 8 mt + v (layout L, clean/FORMATS.md).
#include "wg_platform.h"
#include "wg_mem.h"

// Single-precision steps are rounded one operation at a time (no contraction), as in clean/kernels/decoder.hip.
#pragma clang fp contract(off)

// Frame constants of the coefficient step: Model M's WgDecParams (clean/kernels/decoder.hip); l_coef reads rw, rh, fw.
struct LDecParams {
  int rw, rh, fw, fh, reset, pad0;
  float dw, dh, kx, ky, jx, jy, pjx, pjy;
};
static_assert(sizeof(LDecParams) == 56, "LDecParams must match WgDecParams");

namespace {

constexpr int kThreads = 256;
constexpr int kItemsPerWave = 16;

WG_FN int item_of(u32 tid) {
  return (int)(wg_block_id() * (kThreads / WG_WAVE) + (tid >> 5)) * kItemsPerWave + (int)(tid & 15u);
}

WG_FN v8f rn16_8(v8f a) {
#pragma unroll
  for (int i = 0; i < 8; ++i) a[i] = wg_rn16(a[i]);
  return a;
}

// Linear 32 -> 32 on a pixel map: acc = b + W x (bias as the accumulator start, one WMMA K block of 32 inputs), then
// RN16. RELU: y = q8(max(0, acc)) as FP8; otherwise y = acc as FP16. Blob: bias tile vector (rows R_L(32)), then
// fragments [mt < 2][kc < 2], rows R_L(32), K order K_L(32).
template <bool RELU>
WG_FN void linear_32_32(const u8* __restrict__ x, u8* __restrict__ y, const u8* __restrict__ w, int npix) {
  constexpr int C = 32, KC = 2, MT = 2;
  const u32 tid = wg_lane_id(), lane = tid & 31u, hh = (tid >> 4) & 1u;
  const int pix = item_of(tid);
  const int pk = pix < npix ? pix : npix - 1;
  v2i xf[KC];
#pragma unroll
  for (int kc = 0; kc < KC; ++kc) xf[kc] = *(const v2i*)(x + (size_t)pk * C + hh * (C / 2) + 8 * kc);
  const float* bias = (const float*)w + hh * 8;
  const u8* frag = w + 4 * C;
#pragma unroll
  for (int mt = 0; mt < MT; ++mt) {
    v8f acc = wg_load8f(bias + mt * 16);
#pragma unroll
    for (int kc = 0; kc < KC; ++kc) acc = wg_wmma_fp8(((const v2i*)(frag + (mt * KC + kc) * 256))[lane], xf[kc], acc);
    acc = rn16_8(acc);
    if (pix >= npix) continue;
    if constexpr (RELU) {
#pragma unroll
      for (int i = 0; i < 8; ++i) acc[i] = wg_fmax(acc[i], 0.0f);
      *(v2i*)(y + (size_t)pix * C + hh * (C / 2) + 8 * mt) = wg_pack_q8(acc);
    } else {
      *(v8h*)((_Float16*)y + (size_t)pix * C + hh * (C / 2) + 8 * mt) = wg_to_f16(acc);
    }
  }
}

WG_FN float bits(u32 b) { return __builtin_bit_cast(float, b); }

// sigma(x) = 1 / (1 + 2^(x * -log2 e)) with the hardware exponential and reciprocal.
WG_FN float l_sigma(float x) { return wg_rcp(1.0f + wg_exp2(x * bits(0xBFB8AA3Bu))); }

// tanh(t): for |t| >= 0.6, 1 - 2 / (2^(2 |t| log2 e) + 1) (1 from about 9.01 on), sign of t; below, an odd polynomial.
WG_FN float l_tanh(float t) {
  const float a = __builtin_fabsf(t);
  if (a >= bits(0x3F19999Au)) {
    float r = __builtin_fmaf(wg_rcp(wg_exp2(a * bits(0x4038AA3Bu)) + 1.0f), -2.0f, 1.0f);
    if (a >= bits(0x41102CB4u)) r = 1.0f;
    return __builtin_copysignf(r, t);
  }
  const float t2 = t * t;
  const float p = __builtin_fmaf(
      __builtin_fmaf(__builtin_fmaf(__builtin_fmaf(bits(0x3C80F082u), t2, bits(0xBD563CAEu)), t2, bits(0x3E085941u)),
                     t2, bits(0xBEAAA9EDu)),
      t2, 0.0f);
  return __builtin_fmaf(p, t, t);
}

WG_FN u32 unorm8_code(float v) { return (u32)wg_rint(v * 255.0f); }  // v in [0, 1]
WG_FN u32 snorm8_code(float v) { return (u32)(int)wg_rint(wg_fmin(wg_fmax(v, -1.0f), 1.0f) * 127.0f) & 0xffu; }

struct LCodes {
  u32 A, B, C0, C1, L;
};

// Model L's per-pixel functions on the 13 values a (clean/l/reference/table.py L_DEC_VALUES order).
WG_FN LCodes l_pixel(const float a[13], float S, float T) {
  LCodes o;
  o.A = unorm8_code(wg_rn16(l_sigma(a[0]))) | (unorm8_code(wg_rn16(l_sigma(a[1]))) << 8) |
        (unorm8_code(wg_rn16(l_sigma(a[2]))) << 16);
  const float m = wg_fmax(wg_fmax(a[3], a[4]), a[5]);
  const float e0 = wg_exp2((a[3] - m) * bits(0x3FB8AA3Bu)), e1 = wg_exp2((a[4] - m) * bits(0x3FB8AA3Bu)),
              e2 = wg_exp2((a[5] - m) * bits(0x3FB8AA3Bu));
  const float s = (e0 + e1) + e2;
  const float r = wg_rcp(s);
  o.B = unorm8_code(e0 * r) | (unorm8_code(e1 * r) << 8) | (unorm8_code(e2 * r) << 16);
  float g[2];
#pragma unroll
  for (int i = 0; i < 2; ++i) {
    const float v = a[6 + i];
    const float c = v >= 20.0f ? 20.0f : (v <= -20.0f ? -20.0f : v);
    g[i] = wg_rn16(l_sigma(wg_rn16(c)));
  }
  const float sx = S * g[0], sy = S * g[1];
  const float sxy = sx * sy;
  const float rho = l_tanh(a[8] * T);
  o.C0 = wg_pack_h2(sx * sx, sy * sy);
  o.C1 = wg_f2h_bits(sxy * rho);
  o.L = 0;
#pragma unroll
  for (int i = 0; i < 4; ++i) o.L |= snorm8_code(l_tanh(a[9 + i])) << (8 * i);
  return o;
}

// The 13 values of pixel (x, y) of the FP16 pixel map y2 [fh][fw][32].
constexpr int kLValues[13] = {0, 1, 6, 7, 8, 9, 10, 11, 12, 2, 3, 4, 5};

WG_FN void l_values(const _Float16* __restrict__ y2, int fw, int x, int y, float a[13]) {
  const _Float16* p = y2 + (size_t)(y * fw + x) * 32;
  const v8h lo = *(const v8h*)p, hi = *(const v8h*)(p + 8);
#pragma unroll
  for (int i = 0; i < 13; ++i) a[i] = (float)(kLValues[i] < 8 ? lo[kLValues[i]] : hi[kLValues[i] - 8]);
}

// 8 FP8 bytes of input block kb for lane half hh in the K order of clean/l/runner/l_pack.py kord_groups32: block kb,
// slot (hh, j) = channel 32 (kb / 2) + 16 hh + 8 (kb % 2) + j.
WG_FN int group_channel(int kb, u32 hh) { return 32 * (kb >> 1) + 16 * (int)hh + 8 * (kb & 1); }

}  // namespace

// x [npix][32] FP8 -> y [npix][32] FP8 = q8(max(0, RN16(b + W x))). w = blob enc0.Win1.
WG_KERNEL_N(l_linear_32_32_relu, kThreads)(const u8* x, u8* y, const u8* w, int npix) {
  linear_32_32<true>(x, y, w, npix);
}

// x [npix][32] FP8 -> y [npix][32] FP16 = RN16(b + W x). w = blob enc0.Win2.
WG_KERNEL_N(l_linear_32_32_f16, kThreads)(const u8* x, u8* y, const u8* w, int npix) {
  linear_32_32<false>(x, y, w, npix);
}

// Encoder output projection. x: pixel map [2 th][2 tw][32] FP8; token (r, c) reads pixel (2 r + p / 2, 2 c + p % 2) as
// channels 32 p + i. features [th][tw][64] = q8(clamp(s, +-6.28125)) with s = b, then s = RN16(s + W[:, g] z[g]) per
// group g = p of 32 channels. w = blob enc0.project (bias tile vector rows R_L(64), fragments [mt < 4][kb < 8]).
WG_KERNEL_N(l_project_gather, kThreads)(const u8* x, u8* feat, const u8* w, int tw, int th) {
  constexpr int CO = 64, MT = 4, KB = 8;
  const u32 tid = wg_lane_id(), lane = tid & 31u, hh = (tid >> 4) & 1u;
  const int ntok = tw * th;
  const int tok = item_of(tid);
  const int tk = tok < ntok ? tok : ntok - 1;
  const int r = tk / tw, c = tk - r * tw;
  v2i zf[KB];
#pragma unroll
  for (int kb = 0; kb < KB; ++kb) {
    const int p = kb >> 1;
    const int pix = (2 * r + (p >> 1)) * (2 * tw) + 2 * c + (p & 1);
    zf[kb] = *(const v2i*)(x + (size_t)pix * 32 + (group_channel(kb, hh) - 32 * p));
  }
  const float* bias = (const float*)w + hh * 8;
  const u8* frag = w + 4 * CO;
#pragma unroll
  for (int mt = 0; mt < MT; ++mt) {
    v8f acc = wg_load8f(bias + mt * 16);
#pragma unroll
    for (int g = 0; g < KB / 2; ++g) {
      acc = wg_wmma_fp8(((const v2i*)(frag + (mt * KB + 2 * g) * 256))[lane], zf[2 * g], acc);
      acc = wg_wmma_fp8(((const v2i*)(frag + (mt * KB + 2 * g + 1) * 256))[lane], zf[2 * g + 1], acc);
      acc = rn16_8(acc);
    }
#pragma unroll
    for (int i = 0; i < 8; ++i) acc[i] = wg_fmin(wg_fmax(acc[i], -6.28125f), 6.28125f);
    const v2i q = wg_pack_q8(acc);
    if (tok < ntok) *(v2i*)(feat + (size_t)tok * CO + hh * (CO / 2) + 8 * mt) = q;
  }
}

// Decoder projection. z [th][tw][64] FP8 (dec1's output) -> y: pixel map [2 th][2 tw][32] FP8. Per token: 128 outputs
// n = 32 s + k, s = RN16(b + W[:, 0:32] z[0:32]), then RN16(s + W[:, 32:64] z[32:64]), then FP8; output n goes to pixel
// (2 r + s / 2, 2 c + s % 2), channel k. w = blob dec0.project (bias tile vector rows R_E(32), fragments [mt < 8][kb <
// 4]; tile mt = 2 s + g holds sub-pixel s, channels 16 hh + 8 g + v).
WG_KERNEL_N(l_dec_project, kThreads)(const u8* z, u8* y, const u8* w, int tw, int th) {
  constexpr int MT = 8, KB = 4;
  const u32 tid = wg_lane_id(), lane = tid & 31u, hh = (tid >> 4) & 1u;
  const int ntok = tw * th;
  const int tok = item_of(tid);
  const int tk = tok < ntok ? tok : ntok - 1;
  const int r = tk / tw, c = tk - r * tw;
  v2i zf[KB];
#pragma unroll
  for (int kb = 0; kb < KB; ++kb) zf[kb] = *(const v2i*)(z + (size_t)tk * 64 + group_channel(kb, hh));
  const float* bias = (const float*)w + hh * 8;
  const u8* frag = w + 4 * 128;
#pragma unroll
  for (int mt = 0; mt < MT; ++mt) {
    v8f acc = wg_load8f(bias + mt * 16);
#pragma unroll
    for (int g = 0; g < KB / 2; ++g) {
      acc = wg_wmma_fp8(((const v2i*)(frag + (mt * KB + 2 * g) * 256))[lane], zf[2 * g], acc);
      acc = wg_wmma_fp8(((const v2i*)(frag + (mt * KB + 2 * g + 1) * 256))[lane], zf[2 * g + 1], acc);
      acc = rn16_8(acc);
    }
    const v2i q = wg_pack_q8(acc);
    const int s = mt >> 1, g = mt & 1;
    const int pix = (2 * r + (s >> 1)) * (2 * tw) + 2 * c + (s & 1);
    if (tok < ntok) *(v2i*)(y + (size_t)pix * 32 + 16 * hh + 8 * g) = q;
  }
}

// Coefficient step. y2 [fh][fw][32] FP16 (the decoder block's output) -> A, B (RGBA8 unorm codes), C (4 x FP16: C0, C1,
// C2, 0) and the new latent (4 x s8 snorm), [rh][rw] each, pixels inside R only; Model M's coef_decoder output formats.
// Grid as coef_decoder: ceil((fh / 2)(fw / 2) / 128); lane (t, h) of wave i handles token t' = 16 i + t = (r, c), pixels
// (2 c, 2 r + h) and (2 c + 1, 2 r + h). w = blob dec0.coef (Model M's dec0 blob: S, T at byte 320). No motion,
// previous latent, jitter or reset is read.
WG_KERNEL_N(l_coef, kThreads)(const u8* y2, u32* A, u32* B, u32* C, u32* latent, const u8* w, LDecParams P) {
  const u32 tid = wg_lane_id(), hh = (tid >> 4) & 1u;
  const int tw = P.fw >> 1, ntok = (P.fh >> 1) * tw;
  const int tok = item_of(tid);
  if (tok >= ntok) return;
  const int r = tok / tw, c = tok - r * tw;
  const int y = 2 * r + (int)hh;
  if (y >= P.rh) return;
  const float S = ((const float*)w)[80], T = ((const float*)w)[81];
#pragma unroll
  for (int dx = 0; dx < 2; ++dx) {
    const int x = 2 * c + dx;
    if (x >= P.rw) continue;
    float a[13];
    l_values((const _Float16*)y2, P.fw, x, y, a);
    const LCodes o = l_pixel(a, S, T);
    const u32 i = (u32)(y * P.rw + x);
    A[i] = o.A;
    B[i] = o.B;
    C[2 * i] = o.C0;
    C[2 * i + 1] = o.C1;
    latent[i] = o.L;
  }
}
