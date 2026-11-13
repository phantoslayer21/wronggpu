// SPDX-License-Identifier: MIT
// Output reconstruction (stage "post") and output filter (stage "filter") of spec/06-reconstruction-and-output.md
// for gfx1201 (RDNA4). Data layouts: clean/FORMATS.md, "Output reconstruction and output filter"; launch plans and
// input conversion: clean/runner/post.py. Reference: clean/reference/post.py.
//
// post_recon: one lane computes the two D pixels (x, 2k) and (x, 2k + 1) of a row pair, which share the 3 x 3 render
// window (spec 06 step 2 takes the window rows from the even row). A 256-lane workgroup covers 32 columns x 8 row
// pairs. Its render footprint (at most 18 x 11 render pixels at the Quality ratio) is staged once into LDS with the
// per-pixel features already computed (E_in of the exposed colour, the A and B codes, the kernel shape C), so the
// nine taps of both pixels read LDS only. A footprint larger than the LDS area (D smaller than about 1.5 R) falls
// back to reading the taps from global memory. The 2 x 2 means of the reduced history pair lane x with lane x + 1
// through LDS.
//
// post_filter: separable 5-tap filter (spec 06 draft 3 weights) with the local min/max clamp; see "Output filter"
// below.
//
// Min/max/clamp use maxNum/minNum (__builtin_fmaxf/fminf): a NaN operand is ignored, so clamp(NaN, lo, hi) = lo,
// as in the reference (see its header and QUESTIONS.md, spec 06).
#include "wg_platform.h"
#define WG_Q9_FN WG_FN
#include "output_quality9.h"
#undef WG_Q9_FN
#ifndef WG_LZ_PROFILE
#define WG_LZ_PROFILE 0  // 1: the 5-tap output filter with the quality5 weights (kLz0 ... kLz4 below)
                         // 2: spec 06 draft 4 "original": the five weights are kernel arguments, FP16 steps (LzW below)
#endif
#if WG_LZ_PROFILE == 1
#include "output_quality5.h"
#endif

// WG_POST_DIET (default 0): the default-named kernels keep their machine code; the _diet kernels (post_recon2_r_out_diet,
// post_tf_f32_p12_diet; runtime option post_diet=1 with exposure_pair_diet) take the cuts: ratio / dec_scale read
// from the exposure pair (expo[3], expo[4]: no FP64 divides per wave), the 1-D tile index decoded with a host
// reciprocal (no runtime integer divide), and the vertical pass's ring slots from one modulo.
#ifndef WG_POST_DIET
#define WG_POST_DIET 0
#endif
WG_FN u32 wg_umulhi(u32 a, u32 b) { return (u32)(((unsigned long long)a * b) >> 32); }

// Keep every multiply and add a separate IEEE operation (no implicit FMA): the window positions a_x and b must come
// out the same for a lane and for the footprint computation of its workgroup. Explicit __builtin_fmaf is used where
// a fused operation is wanted.
#pragma clang fp contract(off)

// Timing-only ablations (runner/post_ablate.py builds each as its own code object; the output is wrong on purpose).
// Each removes one part of the work while keeping the outputs data-dependent, so the difference to the base build is
// what that part costs.
#ifndef WG_AB_GATHER
#define WG_AB_GATHER 0   // 1: kernel gather of the centre render sample only (1 tap instead of 9)
#endif
#ifndef WG_AB_EXP
#define WG_AB_EXP 0      // 1: Gaussian tap weight 1 + 2^-10 q instead of exp(q) (the quadratic form stays)
#endif
#ifndef WG_AB_STAGE
#define WG_AB_STAGE 0    // 1: LDS staging writes a constant texel (no global loads, no E_in logs)
#endif
#ifndef WG_AB_MOTION
#define WG_AB_MOTION 0   // 1: no motion reads; the three candidates are fixed offsets (distinct positions)
#endif
#ifndef WG_AB_SCORE
#define WG_AB_SCORE 0    // 1: no candidate scoring: first = near, second = center, weights 1/2
#endif
#ifndef WG_AB_HIST
#define WG_AB_HIST 0     // 1: no history taps (the 12-tap Catmull-Rom returns its position)
#endif
#ifndef WG_AB_TRANS
#define WG_AB_TRANS 0    // 1: E() and Dec() linear (no log / exp outside the kernel gather)
#endif
#ifndef WG_AB_HISTADDR
#define WG_AB_HISTADDR 0  // 1: history taps of every pixel at (8 or 9, 8 or 9) (cache-resident; arithmetic kept)
#endif
#ifndef WG_AB_HISTROWS
#define WG_AB_HISTROWS 0  // 1: history tap rows 0 and 3 not loaded (timing only)
#endif
#ifndef WG_AB_HISTW
#define WG_AB_HISTW 0    // 1: no full-history write
#endif
#ifndef WG_AB_REDUCED
#define WG_AB_REDUCED 0  // 1: no reduced history (no 2 x 2 exchange, no write)
#endif
#ifndef WG_AB_DISPLAY
#define WG_AB_DISPLAY 0  // 1: no display write (post_recon: global; fused: LDS, the filter reads stale values)
#endif
#ifndef WG_AB_CODES
#define WG_AB_CODES 0    // 1: the gather accumulates W and the colour only (not the six per-sample codes)
#endif
#ifndef WG_AB_ENC
#define WG_AB_ENC 0      // 1: E() linear (no log), Dec() kept
#endif
#ifndef WG_AB_DEC
#define WG_AB_DEC 0      // 1: Dec() linear (no exp), E() kept
#endif
#ifndef WG_AB_BLEND
#define WG_AB_BLEND 0    // 1: no step 5 (H^ = H, O^ = c_bar: no Dec, no blend, no NaN guard)
#endif
#ifndef WG_AB_FCLAMP
#define WG_AB_FCLAMP 0   // 1: output filter without the min/max clamp
#endif
#ifndef WG_AB_FDEC
#define WG_AB_FDEC 0     // 1: output filter without Dec (linear decode)
#endif

typedef _Float16 v4h __attribute__((ext_vector_type(4)));

// Per-frame constants, passed by value (4-byte fields; must match PostParams in clean/runner/post.py).
struct WgPostParams {
  int rw, rh;       // render size R
  int dw, dh;       // reconstruction size D
  int reset;        // 1 on reset
  int pad0, pad1, pad2;
  float rx, ry;     // R / D per axis
  float jx, jy;     // jitter (render pixels)
  float e, e_prev;  // exposures, 0 already replaced by 1
  float ratio;      // e / e_prev
  float pad3;
};
static_assert(sizeof(WgPostParams) == 64, "WgPostParams must match PostParams in clean/runner/post.py");

namespace {

constexpr int kPostThreads = 256;
constexpr int kTileX = 32;       // D columns per workgroup (one per lane of a wave)
constexpr int kTilePairs = 8;    // D row pairs per workgroup (one per wave)
constexpr int kStageCap = 512;   // render pixels staged in LDS (32 B each)

constexpr float kLog2E = 1.44269504088896341f;
constexpr float kEncK = 0.36f * 0.30102999566398120f;   // E: 0.36 log10(v) = kEncK log2(v)
constexpr float kDecK = 3.32192809488736235f / 0.36f;   // Dec: 10^(q / 0.36) = 2^(kDecK q)
constexpr float kDecLn = 2.30258509299404568f / 0.36f;  // Dec: 10^(q / 0.36) = e^(kDecLn q)

#if defined(__AMDGCN__)
WG_FN float post_log2(float x) { return __builtin_amdgcn_logf(x); }  // v_log_f32; arguments here are >= 1e-4
#else
WG_FN float post_log2(float x) { return std::log2(x); }
#endif

// Ablation builds: true only for one impossible value, so that a skipped store's inputs stay live (never stores).
WG_FN bool ab_live(float v) { return v == -0x1.2345p+111f; }

WG_FN int clampi(int v, int lo, int hi) {  // min, then max: one v_min_i32 and one v_max_i32
  const int t = v > hi ? hi : v;
  return t < lo ? lo : t;
}
WG_FN int mini(int a, int b) { return a < b ? a : b; }
// 24-bit multiplies (v_mul_u32_u24 / v_mul_i32_i24, full rate; v_mul_lo_u32 is quarter rate). Exact when both
// factors fit in 24 bits (signed: -2^23 ... 2^23 - 1), true for every image size, pitch and LDS index here; the
// product may use all 32 bits. The masks cost nothing (the instructions read the low 24 bits only).
WG_FN u32 mul24(u32 a, u32 b) { return (a & 0xffffffu) * (b & 0xffffffu); }
WG_FN int mul_i24(int a, int b) { return ((int)((u32)a << 8) >> 8) * ((int)((u32)b << 8) >> 8); }
WG_FN int floor_i(float v) { return (int)__builtin_floorf(v); }
WG_FN float clampf(float v, float lo, float hi) { return wg_fmin(wg_fmax(v, lo), hi); }
// Final framebuffer storage only: exposure undo / pre-exposure can push a
// finite linear result beyond binary16's finite range. Preserve all in-range
// values (including signed zero) and NaNs; avoid generating infinity solely
// from the storage conversion. Reconstruction/history arithmetic is unchanged.
WG_FN _Float16 framebuffer_half(float v) {
  const float finite = v > 65504.0f ? 65504.0f : v < -65504.0f ? -65504.0f : v;
  return (_Float16)finite;
}
#if defined(__AMDGCN__)
WG_FN float med3(float v, float lo, float hi) { return __builtin_amdgcn_fmed3f(v, lo, hi); }
#else
WG_FN float med3(float v, float lo, float hi) { return wg_fmin(wg_fmax(v, lo), hi); }
#endif
constexpr float kDisplayMax = 2.45f;  // upper limit of the display image (spec 06 draft 2)

// E(v) = 0.36 log10(max(1 + 100 v, 1e-4)); NaN -> 0.36 log10(1e-4) (maxNum drops the NaN).
WG_FN float enc_e(float v) {
  if (WG_AB_TRANS || WG_AB_ENC) return 0.36f * wg_fmax(v, -4.0f);
  return kEncK * post_log2(wg_fmax(__builtin_fmaf(100.0f, v, 1.0f), 1e-4f));
}
// E_in(v) = E(min(v, 32752)), E_in(NaN) = 0.
WG_FN float enc_in(float v) { return v != v ? 0.0f : enc_e(wg_fmin(v, 32752.0f)); }
// 10^(q' / 0.36) - 1 with q' = clamp(q, 1e-4, 3), i.e. expm1(y), y = q' ln(10) / 0.36 in [6.4e-4, 19.2]. Below
// y = 0.35 the difference 2^(...) - 1 would lose up to 2e-4 of relative precision to cancellation (q' near 1e-4), so
// a Taylor polynomial is used there (truncation error < 2e-8 relative).
WG_FN float dec_m1_poly(float qc) {  // qc already clamped, qc * kDecLn < 0.35
  const float y = qc * kDecLn;
  float t = __builtin_fmaf(y, 1.0f / 5040.0f, 1.0f / 720.0f);
  t = __builtin_fmaf(t, y, 1.0f / 120.0f);
  t = __builtin_fmaf(t, y, 1.0f / 24.0f);
  t = __builtin_fmaf(t, y, 1.0f / 6.0f);
  t = __builtin_fmaf(t, y, 0.5f);
  t = __builtin_fmaf(t, y, 1.0f);
  return t * y;
}
constexpr float kDecPolyMax = 0.35f / kDecLn;  // dec_m1_poly below this clamped q (about 0.0547)
WG_FN float dec_m1(float q) {
  const float qc = clampf(q, 1e-4f, 3.0f);
  if (WG_AB_TRANS || WG_AB_DEC) return qc * 6.4f;
  return qc < kDecPolyMax ? dec_m1_poly(qc) : wg_exp2(qc * kDecK) - 1.0f;
}
// Dec(q) = (10^(q' / 0.36) - 1) / 100.
WG_FN float dec_e(float q) { return dec_m1(q) * 0.01f; }

// Row pitch (in FP16 elements) of the display image planes: dw rounded up to a multiple of 64 (128 bytes), so that
// every row starts 4-byte aligned for the filter's dword loads. Plane size = pitch * dh.
WG_FN int display_pitch(int dw) { return (dw + 63) & ~63; }

WG_FN float byte_f(u32 v, int i) { return (float)((v >> (8 * i)) & 0xffu); }

WG_FN void unpack_h2(u32 t, float& a, float& b) {
  a = (float)__builtin_bit_cast(_Float16, (unsigned short)(t & 0xffffu));
  b = (float)__builtin_bit_cast(_Float16, (unsigned short)(t >> 16));
}

// Render-space position of D column x / row y (spec 06 step 1).
WG_FN float pos_x(int x, const WgPostParams& p) { return ((float)x + 0.5f) * p.rx + p.jx; }
WG_FN float pos_y(int y, const WgPostParams& p) { return ((float)y + 0.5f) * p.ry + p.jy; }

// Features of one render pixel (spec 06 step 2), as staged in LDS.
struct PostTexel {
  float ec0, ec1, ec2;  // E_in(e * colour)
  u32 a;                // A codes: g, beta_h, beta_o in bytes 0..2
  float ca, cb, cc;     // kernel shape C = (a, b, c)
  u32 b;                // B codes: sigma_near, sigma_center, sigma_far in bytes 0..2
};

WG_FN PostTexel load_texel(const v4f* color, const u32* A, const u32* B, const v4h* C, int idx, float e) {
  if (WG_AB_STAGE) {
    PostTexel t;
    t.ec0 = t.ec1 = t.ec2 = 0.5f + (float)(idx & 7) * 0.01f;
    t.ca = t.cb = 1.0f;
    t.cc = 0.1f;
    t.a = t.b = 0x405080u + (u32)(idx & 15);
    return t;
  }
  const u32 ui = (u32)idx;  // idx >= 0: 32-bit offsets (see hist_load)
  const v4f c = *(const v4f*)((const char*)color + ui * 16u);
  const v4h k = *(const v4h*)((const char*)C + ui * 8u);
  PostTexel t;
  t.ec0 = enc_in(e * c[0]);
  t.ec1 = enc_in(e * c[1]);
  t.ec2 = enc_in(e * c[2]);
  t.a = *(const u32*)((const char*)A + ui * 4u);
  t.ca = (float)k[0];
  t.cb = (float)k[1];
  t.cc = (float)k[2];
  t.b = *(const u32*)((const char*)B + ui * 4u);
  return t;
}

// load_texel for the dec0 "pack12" layout (A, B, C without padding; FORMATS.md): image A8, 8 bytes per pixel = codes
// A0 A1 A2 B0 | B1 B2, C2 (FP16); image C4, 4 bytes per pixel = C0, C1 (FP16). The same values as load_texel (whose
// padding bytes are never used), so the features are bit-identical.
WG_FN PostTexel load_texel12(const v4f* color, const u32* A8, const u32* C4, int idx, float e) {
  if (WG_AB_STAGE) return load_texel(color, A8, A8, nullptr, idx, e);
  const u32 ui = (u32)idx;
  const v4f c = *(const v4f*)((const char*)color + ui * 16u);
  const u32 d0 = *(const u32*)((const char*)A8 + ui * 8u), d1 = *(const u32*)((const char*)A8 + ui * 8u + 4u);
  const u32 c01 = *(const u32*)((const char*)C4 + ui * 4u);
  PostTexel t;
  t.ec0 = enc_in(e * c[0]);
  t.ec1 = enc_in(e * c[1]);
  t.ec2 = enc_in(e * c[2]);
  t.a = d0 & 0xffffffu;
  t.b = (d0 >> 24) | ((d1 & 0xffffu) << 8);
  t.ca = (float)__builtin_bit_cast(_Float16, (unsigned short)(c01 & 0xffffu));
  t.cb = (float)__builtin_bit_cast(_Float16, (unsigned short)(c01 >> 16));
  t.cc = (float)__builtin_bit_cast(_Float16, (unsigned short)(d1 >> 16));
  return t;
}

// Catmull-Rom (cubic convolution, a = -1/2) weights for taps -1, 0, 1, 2 at fraction f.
WG_FN void cr_weights(float f, float w[4]) {
  w[0] = f * (-0.5f + f * (1.0f - 0.5f * f));
  w[1] = 1.0f + f * f * (-2.5f + 1.5f * f);
  w[2] = f * (0.5f + f * (2.0f - 1.5f * f));
  w[3] = f * f * (0.5f * f - 0.5f);
}

// Full history texel storage (spec 06: D, FP16 x 3). kHist6 = false: 8 bytes per pixel (R, G, B, 0; RGBA16F), the
// layout of post_recon. kHist6 = true: 6 bytes per pixel, tightly packed (R, G, B), used by the fused kernels; a
// texel is read with one unaligned 8-byte load (the buffer needs 2 bytes of slack after the last pixel).
typedef u64 u64_a2 __attribute__((aligned(2)));
typedef u32 u32_a2 __attribute__((aligned(2)));

// idx >= 0; the byte offset is formed in 32 bits so that the load uses the uniform base address plus a 32-bit
// per-lane offset (no 64-bit address arithmetic per tap).
template <bool kHist6>
WG_FN void hist_load(const void* hist, int idx, float& r, float& g, float& b) {
  if (kHist6) {
    const u64 v = *(const u64_a2*)((const char*)hist + (u32)idx * 6u);
    r = (float)__builtin_bit_cast(_Float16, (unsigned short)(v & 0xffffu));
    g = (float)__builtin_bit_cast(_Float16, (unsigned short)((v >> 16) & 0xffffu));
    b = (float)__builtin_bit_cast(_Float16, (unsigned short)((v >> 32) & 0xffffu));
  } else {
    const v4h t = *(const v4h*)((const char*)hist + (u32)idx * 8u);
    r = (float)t[0];
    g = (float)t[1];
    b = (float)t[2];
  }
}

template <bool kHist6>
WG_FN void hist_load_off(const void* hist, u32 byte_off, float& r, float& g, float& b) {
  const char* q = (const char*)hist + byte_off;
  if (kHist6) {
    const u64 v = *(const u64_a2*)q;
    r = (float)__builtin_bit_cast(_Float16, (unsigned short)(v & 0xffffu));
    g = (float)__builtin_bit_cast(_Float16, (unsigned short)((v >> 16) & 0xffffu));
    b = (float)__builtin_bit_cast(_Float16, (unsigned short)((v >> 32) & 0xffffu));
  } else {
    const v4h t = *(const v4h*)q;
    r = (float)t[0];
    g = (float)t[1];
    b = (float)t[2];
  }
}

template <bool kHist6>
WG_FN void hist_store(void* hist, int idx, float hr, float hg, float hb) {
  const _Float16 r = (_Float16)hr, g = (_Float16)hg, b = (_Float16)hb;
  if (kHist6) {
    unsigned short* q = (unsigned short*)hist + 3 * idx;
    *(u32_a2*)q = (u32)__builtin_bit_cast(unsigned short, r) | ((u32)__builtin_bit_cast(unsigned short, g) << 16);
    q[2] = __builtin_bit_cast(unsigned short, b);
  } else {
    v4h t;
    t[0] = r;
    t[1] = g;
    t[2] = b;
    t[3] = (_Float16)0.0f;
    ((v4h*)hist)[idx] = t;
  }
}

// Spec 06 step 4, valid case: 12-tap Catmull-Rom of the previous full history at D position (qx, qy). Every tap is
// the uniform base address plus a 32-bit byte offset (clamped row offset + clamped column offset), the form of the
// global loads with a scalar base, so a tap costs one address add.
template <bool kHist6>
WG_FN void history_cr(const void* hist, int dw, int dh, float qx, float qy, float out[3]) {
  if (WG_AB_HIST) {
    out[0] = qx * 0x1p-12f;
    out[1] = qy * 0x1p-12f;
    out[2] = 0.5f;
    return;
  }
  const float tx = qx - 0.5f, ty = qy - 0.5f;
  const float fx0 = __builtin_floorf(tx), fy0 = __builtin_floorf(ty);
  float wx[4], wy[4];
  cr_weights(tx - fx0, wx);
  cr_weights(ty - fy0, wy);
  const int ix = (int)fx0, iy = (int)fy0;
  constexpr u32 kBpp = kHist6 ? 6u : 8u;
  u32 coff[4], roff[4];
#pragma unroll
  for (int k = 0; k < 4; ++k) {
    coff[k] = (u32)clampi(ix - 1 + k, 0, dw - 1) * kBpp;
    roff[k] = (u32)clampi(iy - 1 + k, 0, dh - 1) * ((u32)dw * kBpp);
  }
  float t[4][4][3];  // [row j][column i][channel]; corners unused
#pragma unroll
  for (int j = 0; j < 4; ++j)
#pragma unroll
    for (int i = 0; i < 4; ++i)
      if (!((j == 0 || j == 3) && (i == 0 || i == 3)))
        hist_load_off<kHist6>(hist, roff[j] + coff[i], t[j][i][0], t[j][i][1], t[j][i][2]);
  float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f;
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    const int i0 = (j == 0 || j == 3) ? 1 : 0, i1 = (j == 0 || j == 3) ? 2 : 3;
    float r0 = 0.0f, r1 = 0.0f, r2 = 0.0f;
#pragma unroll
    for (int i = i0; i <= i1; ++i) {
      r0 = __builtin_fmaf(wx[i], t[j][i][0], r0);
      r1 = __builtin_fmaf(wx[i], t[j][i][1], r1);
      r2 = __builtin_fmaf(wx[i], t[j][i][2], r2);
    }
    acc0 = __builtin_fmaf(wy[j], r0, acc0);
    acc1 = __builtin_fmaf(wy[j], r1, acc1);
    acc2 = __builtin_fmaf(wy[j], r2, acc2);
  }
  // Sum of the twelve tap weights.
  const float norm = (wx[1] + wx[2]) * (wy[0] + wy[3]) + (wx[0] + wx[1] + wx[2] + wx[3]) * (wy[1] + wy[2]);
  const float inv = wg_rcp(norm);  // norm is in [0.984, 1]
  out[0] = acc0 * inv;
  out[1] = acc1 * inv;
  out[2] = acc2 * inv;
}

// Catmull-Rom with the taps of a row read as one span of contiguous texels: rows 1 and 2 (4 texels: 32 bytes as two
// 16-byte loads, or 24 bytes for the 6-byte format) and rows 0 and 3 (2 texels). The spans start at column
// cs = clamp(ix - 1, 0, dw - 4), which contains every clamped column a tap needs; away from the left and right edges
// (ix - 1 = cs) the taps are the span's texels in order, otherwise the lane reloads its taps one by one. Same values
// and the same arithmetic as history_cr (bit-identical result); half the load instructions. Needs dw >= 4.
typedef u32 v4u_a2 __attribute__((ext_vector_type(4))) __attribute__((aligned(2)));
typedef u32 v3u_a2 __attribute__((ext_vector_type(3))) __attribute__((aligned(2)));
typedef u32 v2u_a2 __attribute__((ext_vector_type(2))) __attribute__((aligned(2)));
typedef u32 v2u_a4 __attribute__((ext_vector_type(2))) __attribute__((aligned(4)));
typedef u32 v4u_a4 __attribute__((ext_vector_type(4))) __attribute__((aligned(4)));

WG_FN float half_of(const u32* d, int h) {  // FP16 number h of the dword array d
  const u32 v = d[h >> 1];
  return (float)__builtin_bit_cast(_Float16, (unsigned short)((h & 1) ? (v >> 16) : (v & 0xffffu)));
}

// n = 2 or 4 contiguous texels at byte offset off -> t[i][c]
template <bool kHist6, int n>
WG_FN void hist_span(const void* hist, u32 off, float t[][3]) {
  const char* q = (const char*)hist + off;
  u32 d[8];
  if (kHist6) {
    if (n == 4) {
      const v4u_a2 a = *(const v4u_a2*)q;
      const v2u_a2 b = *(const v2u_a2*)(q + 16);
      d[0] = a[0]; d[1] = a[1]; d[2] = a[2]; d[3] = a[3]; d[4] = b[0]; d[5] = b[1];
    } else {
      const v3u_a2 a = *(const v3u_a2*)q;
      d[0] = a[0]; d[1] = a[1]; d[2] = a[2];
    }
#pragma unroll
    for (int i = 0; i < n; ++i)
#pragma unroll
      for (int c = 0; c < 3; ++c) t[i][c] = half_of(d, 3 * i + c);
  } else {
    const v4u_a2 a = *(const v4u_a2*)q;
    d[0] = a[0]; d[1] = a[1]; d[2] = a[2]; d[3] = a[3];
    if (n == 4) {
      const v4u_a2 b = *(const v4u_a2*)(q + 16);
      d[4] = b[0]; d[5] = b[1]; d[6] = b[2]; d[7] = b[3];
    }
#pragma unroll
    for (int i = 0; i < n; ++i)
#pragma unroll
      for (int c = 0; c < 3; ++c) t[i][c] = half_of(d, 4 * i + c);
  }
}

template <bool kHist6>
WG_FN void history_cr_span(const void* hist, int dw, int dh, float qx, float qy, float out[3]) {
  if (WG_AB_HIST) {
    out[0] = qx * 0x1p-12f;
    out[1] = qy * 0x1p-12f;
    out[2] = 0.5f;
    return;
  }
  const float tx = qx - 0.5f, ty = qy - 0.5f;
  const float fx0 = __builtin_floorf(tx), fy0 = __builtin_floorf(ty);
  float wx[4], wy[4];
  cr_weights(tx - fx0, wx);
  cr_weights(ty - fy0, wy);
  const int ix = WG_AB_HISTADDR ? 8 + ((int)fx0 & 1) : (int)fx0, iy = WG_AB_HISTADDR ? 8 + ((int)fy0 & 1) : (int)fy0;
  constexpr u32 kBpp = kHist6 ? 6u : 8u;
  const int cs = clampi(ix - 1, 0, dw - 4);
  float t[4][4][3];  // [row j][column i][channel]; corners unused
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    const u32 roff = mul24((u32)clampi(iy - 1 + j, 0, dh - 1), (u32)dw * kBpp);
    if (j == 0 || j == 3) {
      if (!WG_AB_HISTROWS) hist_span<kHist6, 2>(hist, roff + (u32)(cs + 1) * kBpp, &t[j][1]);
    } else {
      hist_span<kHist6, 4>(hist, roff + (u32)cs * kBpp, t[j]);
    }
  }
  if (WG_AB_HISTROWS) {  // timing only: rows 0 and 3 not loaded (copies of rows 1 and 2)
#pragma unroll
    for (int i = 1; i <= 2; ++i)
#pragma unroll
      for (int c = 0; c < 3; ++c) {
        t[0][i][c] = t[1][i][c] * 0.5f;
        t[3][i][c] = t[2][i][c] * 0.5f;
      }
  }
  if (ix - 1 != cs) {  // left or right image edge: clamped columns
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const u32 roff = mul24((u32)clampi(iy - 1 + j, 0, dh - 1), (u32)dw * kBpp);
#pragma unroll
      for (int i = 0; i < 4; ++i)
        if (!((j == 0 || j == 3) && (i == 0 || i == 3)))
          hist_load_off<kHist6>(hist, roff + (u32)clampi(ix - 1 + i, 0, dw - 1) * kBpp, t[j][i][0], t[j][i][1],
                                t[j][i][2]);
    }
  }
  float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f;
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    const int i0 = (j == 0 || j == 3) ? 1 : 0, i1 = (j == 0 || j == 3) ? 2 : 3;
    float r0 = 0.0f, r1 = 0.0f, r2 = 0.0f;
#pragma unroll
    for (int i = i0; i <= i1; ++i) {
      r0 = __builtin_fmaf(wx[i], t[j][i][0], r0);
      r1 = __builtin_fmaf(wx[i], t[j][i][1], r1);
      r2 = __builtin_fmaf(wx[i], t[j][i][2], r2);
    }
    acc0 = __builtin_fmaf(wy[j], r0, acc0);
    acc1 = __builtin_fmaf(wy[j], r1, acc1);
    acc2 = __builtin_fmaf(wy[j], r2, acc2);
  }
  const float norm = (wx[1] + wx[2]) * (wy[0] + wy[3]) + (wx[0] + wx[1] + wx[2] + wx[3]) * (wy[1] + wy[2]);
  const float inv = wg_rcp(norm);
  out[0] = acc0 * inv;
  out[1] = acc1 * inv;
  out[2] = acc2 * inv;
}

// ---- planar 6-byte full history (post_recon2_r_hp*): exact, every access dword- or half-aligned ----
// Plane 0: R, G of pixel i = y dw + x as two FP16 at byte 4 i; plane 1 at byte 4 dw dh: B as FP16 at 4 dw dh + 2 i.
// Buffer 6 dw dh + 8 bytes (a B span is read from the dword at or below its start: up to 4 bytes past the plane).
WG_FN u32 hp_bbase(int dw, int dh) { return (u32)dw * (u32)dh * 4u; }
WG_FN void hist_store_planar(void* hist, u32 bbase, u32 i, float r, float g, float b) {
  const _Float16 hr = (_Float16)r, hg = (_Float16)g, hb = (_Float16)b;
  *(u32*)((char*)hist + i * 4u) =
      (u32)__builtin_bit_cast(unsigned short, hr) | ((u32)__builtin_bit_cast(unsigned short, hg) << 16);
  *(unsigned short*)((char*)hist + bbase + i * 2u) = __builtin_bit_cast(unsigned short, hb);
}
WG_FN void hist_load_planar(const void* hist, u32 bbase, u32 i, float t[3]) {
  const u32 rg = *(const u32*)((const char*)hist + i * 4u);
  t[0] = half_of(&rg, 0);
  t[1] = half_of(&rg, 1);
  const u32 b = *(const unsigned short*)((const char*)hist + bbase + i * 2u);
  t[2] = half_of(&b, 0);
}
WG_FN u32 shr64(u32 lo, u32 hi, u32 sh) { return (u32)((((u64)hi << 32) | lo) >> sh); }  // v_alignbit
// n consecutive texels from pixel index i: RG as one aligned n-dword load, B from the aligned dwords around it.
template <int n>
WG_FN void hp_span(const void* hist, u32 bbase, u32 i, float t[][3]) {
  const char* base = (const char*)hist;
  u32 rg[4], bw[3];
  if (n == 4) {
    const v4u a = *(const v4u*)(base + i * 4u);
    rg[0] = a[0]; rg[1] = a[1]; rg[2] = a[2]; rg[3] = a[3];
  } else {
    const v2u_a4 a = *(const v2u_a4*)(base + i * 4u);
    rg[0] = a[0]; rg[1] = a[1];
  }
  const u32 bo = bbase + i * 2u, ba = bo & ~3u, sh = (bo & 2u) * 8u;  // shift 0 or 16 bits
  if (n == 4) {
    const v4u_a4 b = *(const v4u_a4*)(base + ba);  // 3 dwords used; the 4th keeps the load a single b128
    bw[0] = shr64(b[0], b[1], sh);
    bw[1] = shr64(b[1], b[2], sh);
  } else {
    const v2u_a4 b = *(const v2u_a4*)(base + ba);
    bw[0] = shr64(b[0], b[1], sh);
  }
#pragma unroll
  for (int k = 0; k < n; ++k) {
    t[k][0] = half_of(rg, 2 * k);
    t[k][1] = half_of(rg, 2 * k + 1);
    t[k][2] = half_of(bw, k);
  }
}

// history_cr_span on the planar history (same taps, same arithmetic: bit-identical).
WG_FN void history_cr_planar(const void* hist, int dw, int dh, float qx, float qy, float out[3]) {
  if (WG_AB_HIST) {
    out[0] = qx * 0x1p-12f;
    out[1] = qy * 0x1p-12f;
    out[2] = 0.5f;
    return;
  }
  const float tx = qx - 0.5f, ty = qy - 0.5f;
  const float fx0 = __builtin_floorf(tx), fy0 = __builtin_floorf(ty);
  float wx[4], wy[4];
  cr_weights(tx - fx0, wx);
  cr_weights(ty - fy0, wy);
  const int ix = (int)fx0, iy = (int)fy0;
  const u32 bbase = hp_bbase(dw, dh);
  float t[4][4][3];  // [row j][column i][channel]; corners unused
  const bool spans = dw >= 4;
  const int cs = spans ? clampi(ix - 1, 0, dw - 4) : 0;
  if (spans) {
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const u32 row = mul24((u32)clampi(iy - 1 + j, 0, dh - 1), (u32)dw);
      if (j == 0 || j == 3)
        hp_span<2>(hist, bbase, row + (u32)(cs + 1), &t[j][1]);
      else
        hp_span<4>(hist, bbase, row + (u32)cs, t[j]);
    }
  }
  if (!spans || ix - 1 != cs) {  // image edge (or narrow image): clamped columns one texel at a time
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const u32 row = mul24((u32)clampi(iy - 1 + j, 0, dh - 1), (u32)dw);
#pragma unroll
      for (int i = 0; i < 4; ++i)
        if (!((j == 0 || j == 3) && (i == 0 || i == 3)))
          hist_load_planar(hist, bbase, row + (u32)clampi(ix - 1 + i, 0, dw - 1), t[j][i]);
    }
  }
  float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f;
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    const int i0 = (j == 0 || j == 3) ? 1 : 0, i1 = (j == 0 || j == 3) ? 2 : 3;
    float r0 = 0.0f, r1 = 0.0f, r2 = 0.0f;
#pragma unroll
    for (int i = i0; i <= i1; ++i) {
      r0 = __builtin_fmaf(wx[i], t[j][i][0], r0);
      r1 = __builtin_fmaf(wx[i], t[j][i][1], r1);
      r2 = __builtin_fmaf(wx[i], t[j][i][2], r2);
    }
    acc0 = __builtin_fmaf(wy[j], r0, acc0);
    acc1 = __builtin_fmaf(wy[j], r1, acc1);
    acc2 = __builtin_fmaf(wy[j], r2, acc2);
  }
  const float norm = (wx[1] + wx[2]) * (wy[0] + wy[3]) + (wx[0] + wx[1] + wx[2] + wx[3]) * (wy[1] + wy[2]);
  const float inv = wg_rcp(norm);
  out[0] = acc0 * inv;
  out[1] = acc1 * inv;
  out[2] = acc2 * inv;
}

// Gathered (unnormalized) sums of one D pixel (spec 06 step 2). For the floors beta_h and beta_o the complements
// 255 - code are summed, so that 1 - beta (needed by the blend, see finish_pixel) keeps its relative precision.
struct Gather {
  float W;
  float c[3];  // sum w * E_in colour
  float s[3];  // sum w * B code
  float a[3];  // sum w * g code, sum w * (255 - beta_h code), sum w * (255 - beta_o code)
};

WG_FN void gather_zero(Gather& g) {
  g.W = 0.0f;
#pragma unroll
  for (int i = 0; i < 3; ++i) g.c[i] = g.s[i] = g.a[i] = 0.0f;
}

// Tap offsets pre-scaled by -1/2 log2(e): uu = k u^2, vv = k v^2, uv = 2 k u v, so that
// w = exp(-1/2 (a u^2 + b v^2 + 2 c u v)) = 2^(a uu + b vv + c uv).
constexpr float kTapK = -0.5f * kLog2E;

WG_FN void gather_add(Gather& g, const PostTexel& t, float uu, float vv, float uv) {
  const float q = __builtin_fmaf(t.ca, uu, __builtin_fmaf(t.cb, vv, t.cc * uv));
  const float w = WG_AB_EXP ? __builtin_fmaf(q, 0x1p-10f, 1.0f) : wg_exp2(q);
  g.W += w;
  g.c[0] = __builtin_fmaf(w, t.ec0, g.c[0]);
  g.c[1] = __builtin_fmaf(w, t.ec1, g.c[1]);
  g.c[2] = __builtin_fmaf(w, t.ec2, g.c[2]);
  const u32 ac = t.a ^ 0xffff00u;  // bytes 1, 2 -> 255 - code
#pragma unroll
  for (int i = 0; i < 3; ++i) {
    g.s[i] = __builtin_fmaf(w, byte_f(t.b, i), g.s[i]);
    g.a[i] = __builtin_fmaf(w, byte_f(ac, i), g.a[i]);
  }
}

// Features of one render pixel with the codes already converted (fused kernels' LDS staging). gather_add_f performs
// exactly the operations of gather_add, so both give bit-identical sums.
struct alignas(16) PostTexelF {  // 48 bytes: three 16-byte LDS loads
  float ec0, ec1, ec2;  // E_in(e * colour)
  float ca, cb, cc;     // kernel shape C
  float s0, s1, s2;     // B codes (sigma_near, sigma_center, sigma_far)
  float g, cbh, cbo;    // A: g code, 255 - beta_h code, 255 - beta_o code
};

WG_FN PostTexelF texel_f(const PostTexel& t) {
  PostTexelF f;
  f.ec0 = t.ec0;
  f.ec1 = t.ec1;
  f.ec2 = t.ec2;
  f.ca = t.ca;
  f.cb = t.cb;
  f.cc = t.cc;
  f.s0 = byte_f(t.b, 0);
  f.s1 = byte_f(t.b, 1);
  f.s2 = byte_f(t.b, 2);
  const u32 ac = t.a ^ 0xffff00u;
  f.g = byte_f(ac, 0);
  f.cbh = byte_f(ac, 1);
  f.cbo = byte_f(ac, 2);
  return f;
}

WG_FN void gather_add_f(Gather& g, const PostTexelF& t, float uu, float vv, float uv) {
  const float q = __builtin_fmaf(t.ca, uu, __builtin_fmaf(t.cb, vv, t.cc * uv));
  const float w = WG_AB_EXP ? __builtin_fmaf(q, 0x1p-10f, 1.0f) : wg_exp2(q);
  g.W += w;
  g.c[0] = __builtin_fmaf(w, t.ec0, g.c[0]);
  g.c[1] = __builtin_fmaf(w, t.ec1, g.c[1]);
  g.c[2] = __builtin_fmaf(w, t.ec2, g.c[2]);
  if (WG_AB_CODES) {  // codes from the centre colour only (no per-tap accumulation)
    g.s[0] = g.s[1] = g.s[2] = g.c[0] * 40.0f;
    g.a[0] = g.a[1] = g.a[2] = g.c[1] * 40.0f;
    return;
  }
  g.s[0] = __builtin_fmaf(w, t.s0, g.s[0]);
  g.a[0] = __builtin_fmaf(w, t.g, g.a[0]);
  g.s[1] = __builtin_fmaf(w, t.s1, g.s[1]);
  g.a[1] = __builtin_fmaf(w, t.cbh, g.a[1]);
  g.s[2] = __builtin_fmaf(w, t.s2, g.s[2]);
  g.a[2] = __builtin_fmaf(w, t.cbo, g.a[2]);
}

// m_center bilinear at render position (px, py) from a fetch function m(c, r) of unclamped texel coordinates.
template <class F>
WG_FN void motion_bilinear(float px, float py, F m, float& mx, float& my) {
  const float tx = px - 0.5f, ty = py - 0.5f;
  const float x0f = __builtin_floorf(tx), y0f = __builtin_floorf(ty);
  const float wx = tx - x0f, wy = ty - y0f;
  const int c0 = (int)x0f, r0 = (int)y0f;
  float a0, a1, b0, b1, c0x, c0y, d0, d1;
  unpack_h2(m(c0, r0), a0, a1);
  unpack_h2(m(c0 + 1, r0), b0, b1);
  unpack_h2(m(c0, r0 + 1), c0x, c0y);
  unpack_h2(m(c0 + 1, r0 + 1), d0, d1);
  const float topx = a0 * (1.0f - wx) + b0 * wx, topy = a1 * (1.0f - wx) + b1 * wx;
  const float botx = c0x * (1.0f - wx) + d0 * wx, boty = c0y * (1.0f - wx) + d1 * wx;
  mx = topx * (1.0f - wy) + botx * wy;
  my = topy * (1.0f - wy) + boty * wy;
}

// motion_bilinear with the FP16 -> FP32 conversion folded into the products (v_fma_mix_f32 reads the FP16 operand
// directly): same sums in the same order. Exactness: fma(h, w, +0) is the exact product h w rounded once, i.e. the
// same value as h * w, except that an exact zero product -0 (h or w = -0, or opposite signs with a zero factor) comes
// out as -0 + +0 = +0. Only the sign of a zero can differ. Every such term is then added to another term (top, bot)
// or scaled and added (mx = top (1 - wy) + bot wy), and the motion is used only as qx = X + mx, qy = Y + my with
// X, Y >= 0.5: x + (+0) and x + (-0) are equal for any nonzero x, and so are their comparisons and the history
// positions. (NaN / inf factors give the same NaN / inf as the multiply.) The display and the history are therefore
// bit-identical (verified with runner/post_exact.py).
WG_FN float hw_lo(u32 t, float w) {
  return __builtin_fmaf((float)__builtin_bit_cast(_Float16, (unsigned short)(t & 0xffffu)), w, 0.0f);
}
WG_FN float hw_hi(u32 t, float w) {
  return __builtin_fmaf((float)__builtin_bit_cast(_Float16, (unsigned short)(t >> 16)), w, 0.0f);
}
template <class F>
WG_FN void motion_bilinear_mix(float px, float py, F m, float& mx, float& my) {
  const float tx = px - 0.5f, ty = py - 0.5f;
  const float x0f = __builtin_floorf(tx), y0f = __builtin_floorf(ty);
  const float wx = tx - x0f, wy = ty - y0f;
  const int c0 = (int)x0f, r0 = (int)y0f;
  const u32 a = m(c0, r0), b = m(c0 + 1, r0), c = m(c0, r0 + 1), d = m(c0 + 1, r0 + 1);
  const float owx = 1.0f - wx;
  const float topx = hw_lo(a, owx) + hw_lo(b, wx), topy = hw_hi(a, owx) + hw_hi(b, wx);
  const float botx = hw_lo(c, owx) + hw_lo(d, wx), boty = hw_hi(c, owx) + hw_hi(d, wx);
  mx = topx * (1.0f - wy) + botx * wy;
  my = topy * (1.0f - wy) + boty * wy;
}
// motion_bilinear_mix with the column passed relative to floor(px - 0.5) (m(0, r), m(1, r)): the caller maps it to a
// precomputed column offset (post_tf_body kIdx). Same operations in the same order.
template <class F>
WG_FN void motion_bilinear_mix_rel(float px, float py, F m, float& mx, float& my) {
  const float tx = px - 0.5f, ty = py - 0.5f;
  const float x0f = __builtin_floorf(tx), y0f = __builtin_floorf(ty);
  const float wx = tx - x0f, wy = ty - y0f;
  const int r0 = (int)y0f;
  const u32 a = m(0, r0), b = m(1, r0), c = m(0, r0 + 1), d = m(1, r0 + 1);
  const float owx = 1.0f - wx;
  const float topx = hw_lo(a, owx) + hw_lo(b, wx), topy = hw_hi(a, owx) + hw_hi(b, wx);
  const float botx = hw_lo(c, owx) + hw_lo(d, wx), boty = hw_hi(c, owx) + hw_hi(d, wx);
  mx = topx * (1.0f - wy) + botx * wy;
  my = topy * (1.0f - wy) + boty * wy;
}

// Spec 06 steps 3 (candidate choice) to 5 for one D pixel (x, y), from its gathered sums and its three motion
// candidates (mx, my: near, center, far); (fx, fy) = the clamped render pixel (floor px, floor py) for the fallback.
// Returns the linear new history H^ in hh and the linear display value O^ in oh (NaN guard applied).
// Normalized gathered values and the candidate choice of spec 06 step 3 for one pixel.
struct PostSel {
  float cbar[3];     // normalized E_in colour
  float cbh, cbo;    // 1 - beta_h, 1 - beta_o (normalized)
  int first, second;
  float w1, w2;      // candidate weights (w2 = 0: second not needed)
};

WG_FN PostSel post_select(const Gather& g) {
  PostSel r;
  const float invW = wg_rcp(g.W);  // rcp(0) = inf, rcp(inf) = 0 and rcp(NaN) = NaN, as needed below
  const float inv255 = invW * (1.0f / 255.0f);
  float sig[3];
#pragma unroll
  for (int i = 0; i < 3; ++i) {
    r.cbar[i] = g.c[i] * invW;
    sig[i] = g.s[i] * inv255;
  }
  r.cbh = g.a[1] * inv255;  // 1 - beta_h, 1 - beta_o (normalized)
  r.cbo = g.a[2] * inv255;

  // 3. Best two by score; ties prefer the lower index; a NaN score never wins a comparison.
  int first = 0, second = 1;
  float w1 = 0.5f, w2 = 0.5f;
  if (!WG_AB_SCORE) {
    float best = sig[0];
    if (sig[1] > best) { first = 1; best = sig[1]; }
    if (sig[2] > best) { first = 2; }
    const int o1 = first == 0 ? 1 : 0, o2 = first == 2 ? 1 : 2;
    second = sig[o2] > sig[o1] ? o2 : o1;
    const float q1 = clampf(1.1f * sig[first] - 0.05f, 0.0f, 1.0f);
    const float q2 = clampf(1.1f * sig[second] - 0.05f, 0.0f, 1.0f);
    const float rden = wg_rcp(wg_fmax(q1 + q2, 1e-10f));  // v_rcp (1 ulp) instead of two IEEE divisions
    w2 = q2 * rden;
    if (w2 < 1.0f / 255.0f) {
      w2 = 0.0f;
      w1 = 1.0f;
    } else {
      w1 = q1 * rden;
    }
  }

  r.first = first;
  r.second = second;
  r.w1 = w1;
  r.w2 = w2;
  return r;
}

// post_select with the two chosen candidates' motions (m1: first, m2: second) carried along the score comparisons by
// selects (no branch tree indexing mx / my by first / second): the same comparisons in the same order, so first,
// second, sig[first], sig[second], q1, q2, w1, w2 and the motions are the identical values (ties, NaN: the same >).
// b1, b2: the two "best" tests (first = 2 when b2, 1 when b1 && !b2, else 0); o1 = first == 0 ? 1 : 0,
// o2 = first == 2 ? 1 : 2, second = sig[o2] > sig[o1] ? o2 : o1.
WG_FN PostSel post_select_m(const Gather& g, const float mx[3], const float my[3], float& m1x, float& m1y,
                            float& m2x, float& m2y) {
  PostSel r;
  const float invW = wg_rcp(g.W);
  const float inv255 = invW * (1.0f / 255.0f);
  float sig[3];
#pragma unroll
  for (int i = 0; i < 3; ++i) {
    r.cbar[i] = g.c[i] * invW;
    sig[i] = g.s[i] * inv255;
  }
  r.cbh = g.a[1] * inv255;
  r.cbo = g.a[2] * inv255;

  int first = 0, second = 1;
  float w1 = 0.5f, w2 = 0.5f;
  m1x = mx[0];
  m1y = my[0];
  m2x = mx[1];
  m2y = my[1];
  if (!WG_AB_SCORE) {
    const bool b1 = sig[1] > sig[0];
    const float best = b1 ? sig[1] : sig[0];
    const bool b2 = sig[2] > best;
    const float s1 = b2 ? sig[2] : best;  // sig[first]
    m1x = b2 ? mx[2] : (b1 ? mx[1] : mx[0]);
    m1y = b2 ? my[2] : (b1 ? my[1] : my[0]);
    const bool f0 = !(b1 || b2);  // first == 0
    const float so1 = f0 ? sig[1] : sig[0], so2 = b2 ? sig[1] : sig[2];
    const float mo1x = f0 ? mx[1] : mx[0], mo1y = f0 ? my[1] : my[0];
    const float mo2x = b2 ? mx[1] : mx[2], mo2y = b2 ? my[1] : my[2];
    const bool c = so2 > so1;
    const float s2 = c ? so2 : so1;  // sig[second]
    m2x = c ? mo2x : mo1x;
    m2y = c ? mo2y : mo1y;
    first = b2 ? 2 : (b1 ? 1 : 0);
    second = c ? (b2 ? 1 : 2) : (f0 ? 1 : 0);
    const float q1 = clampf(1.1f * s1 - 0.05f, 0.0f, 1.0f);
    const float q2 = clampf(1.1f * s2 - 0.05f, 0.0f, 1.0f);
    const float rden = wg_rcp(wg_fmax(q1 + q2, 1e-10f));
    w2 = q2 * rden;
    if (w2 < 1.0f / 255.0f) {
      w2 = 0.0f;
      w1 = 1.0f;
    } else {
      w1 = q1 * rden;
    }
  }

  r.first = first;
  r.second = second;
  r.w1 = w1;
  r.w2 = w2;
  return r;
}

// Spec 06 step 5 for one pixel from its selection and its combined history H (E-encoded): the linear new history
// H^ (hh) and display value O^ (oh), NaN guard applied.
// kTrim (runtime option post_trim, kernel suffix t): the Dec clamp of the history half by med3 directly: H is a
// product / sum / fma result, never a signalling NaN, so the canonicalise before the clamp is the identity.
template <bool kTrim = false>
WG_FN void post_blend(const Gather& g, const PostSel& sl, const float H[3], const WgPostParams& p, float s,
                      float hh[3], float oh[3]) {
  if (WG_AB_BLEND) {
#pragma unroll
    for (int i = 0; i < 3; ++i) {
      hh[i] = H[i];
      oh[i] = sl.cbar[i] + sl.cbh;
    }
    return;
  }
  const float* cbar = sl.cbar;
  const float cbh = sl.cbh, cbo = sl.cbo;
  // 5. Blend in linear colour times the current exposure. With beta' = alpha (1 - beta) + beta, the spec's
  // H + (C - H) beta' equals H (1 - alpha)(1 - beta) + C beta'. That form is used because H_lin can be far larger
  // than the result (a bright history pixel rejected towards a dark current colour, beta' close to 1): the literal
  // form then cancels catastrophically in FP32, the product form does not. alpha = s W g_bar = s sum(w g) without
  // the division by W (equal unless W is 0, infinite or NaN; then g_bar is NaN and alpha is 0).
  float Cl[3], Hl[3];
  bool hl_inf = false;
  // The six Dec() share one branch: v = 2^(kDecK qc) - 1 for all six, and only when some lane has a value below
  // kDecPolyMax is the polynomial evaluated, then selected per value (the same per-value result as dec_e; one
  // divergent branch instead of six).
  {
    float qc[6], v[6];
    bool any_small = false;
#pragma unroll
    for (int i = 0; i < 6; ++i) {
      qc[i] = (kTrim && i >= 3) ? med3(H[i - 3], 1e-4f, 3.0f) : clampf(i < 3 ? cbar[i] : H[i - 3], 1e-4f, 3.0f);
      v[i] = (WG_AB_TRANS || WG_AB_DEC) ? qc[i] * 6.4f : wg_exp2(qc[i] * kDecK) - 1.0f;
      any_small |= !(WG_AB_TRANS || WG_AB_DEC) && qc[i] < kDecPolyMax;
    }
    if (any_small) {
#pragma unroll
      for (int i = 0; i < 6; ++i)
        if (qc[i] < kDecPolyMax) v[i] = dec_m1_poly(qc[i]);
    }
#pragma unroll
    for (int i = 0; i < 3; ++i) {
      Cl[i] = v[i] * 0.01f;
      Hl[i] = (v[3 + i] * 0.01f) * p.ratio;
      hl_inf |= __builtin_fabsf(Hl[i]) == __builtin_inff();
    }
  }
  const float alpha = g.W < __builtin_inff() ? clampf(s * (g.a[0] * (1.0f / 255.0f)), 0.0f, 1.0f) : 0.0f;
  const float ombh = (1.0f - alpha) * cbh, ombo = (1.0f - alpha) * cbo;  // 1 - beta'
  const float bh2 = 1.0f - ombh, bo2 = 1.0f - ombo;
#pragma unroll
  for (int i = 0; i < 3; ++i) {
    hh[i] = __builtin_fmaf(Hl[i], ombh, Cl[i] * bh2);
    oh[i] = __builtin_fmaf(Hl[i], ombo, Cl[i] * bo2);
  }
  const float sh = hh[0] + hh[1] + hh[2], so = oh[0] + oh[1] + oh[2];
  if (sh != sh || hl_inf) {
#pragma unroll
    for (int i = 0; i < 3; ++i) hh[i] = Cl[i];
  }
  if (so != so || hl_inf) {
#pragma unroll
    for (int i = 0; i < 3; ++i) oh[i] = Cl[i];
  }
}

template <bool kHist6, bool kSpan = false, bool kPlanar = false>
WG_FN void post_core(const Gather& g, int x, int y, int fx, int fy, const float mx[3], const float my[3],
                     const WgPostParams& p, float s, const v4f* color, const void* hist_prev, float hh[3],
                     float oh[3]) {
  const PostSel sl = post_select(g);
  const int first = sl.first, second = sl.second;
  const float w1 = sl.w1, w2 = sl.w2;

  // 4. History along the first (and, if w2 != 0, the second) candidate.
  const float X = (float)x + 0.5f, Y = (float)y + 0.5f;
  const int ncand = w2 != 0.0f ? 2 : 1;
  float H[3] = {0.0f, 0.0f, 0.0f};
  bool fb_ready = false;
  float fb[3];
  float h[3], qx_prev = 0.0f, qy_prev = 0.0f;
  bool valid_prev = false;
  for (int n = 0; n < ncand; ++n) {
    const int k = n == 0 ? first : second;
    const float wk = n == 0 ? w1 : w2;
    const float qx = X + mx[k], qy = Y + my[k];
    const bool valid = !p.reset && qx >= 0.0f && qx <= (float)p.dw && qy >= 0.0f && qy <= (float)p.dh;
    if (n == 1 && valid && valid_prev && qx == qx_prev && qy == qy_prev) {
      // same reprojected position as the first candidate (static scenes, uniform motion): same sample, h kept
    } else if (valid) {
      if (kPlanar)
        history_cr_planar(hist_prev, p.dw, p.dh, qx, qy, h);
      else if (kSpan)  // post_recon2* only: the host takes them for dw >= 4 only (recon2_supported)
        history_cr_span<kHist6>(hist_prev, p.dw, p.dh, qx, qy, h);
      else
        history_cr<kHist6>(hist_prev, p.dw, p.dh, qx, qy, h);
    } else {
      if (!fb_ready) {
        const v4f c = *(const v4f*)((const char*)color + (mul24((u32)fy, (u32)p.rw * 16u) + (u32)fx * 16u));
        fb[0] = enc_e(p.e_prev * c[0]);
        fb[1] = enc_e(p.e_prev * c[1]);
        fb[2] = enc_e(p.e_prev * c[2]);
        fb_ready = true;
      }
      h[0] = fb[0];
      h[1] = fb[1];
      h[2] = fb[2];
    }
    qx_prev = qx;
    qy_prev = qy;
    valid_prev = valid;
#pragma unroll
    for (int i = 0; i < 3; ++i) H[i] = n == 0 ? wk * h[i] : H[i] + wk * h[i];
  }

  post_blend(g, sl, H, p, s, hh, oh);
}

// post_core with the two candidates as straight-line code (no loop over ncand): same operations and values.
// kSel (post_tf_body kSel, runtime option post_sel): the candidate motions from post_select_m (selects, no branch
// tree); the same values.
template <bool kHist6, bool kSpan = false, bool kPlanar = false, bool kSel = false, bool kTrim = false>
WG_FN void post_core_s(const Gather& g, int x, int y, int fx, int fy, const float mx[3], const float my[3],
                       const WgPostParams& p, float s, const v4f* color, const void* hist_prev, float hh[3],
                       float oh[3]) {
  float m1x, m1y, m2x, m2y;
  PostSel sl;
  if (kSel) {
    sl = post_select_m(g, mx, my, m1x, m1y, m2x, m2y);
  } else {
    sl = post_select(g);
    m1x = sl.first == 0 ? mx[0] : (sl.first == 1 ? mx[1] : mx[2]);
    m1y = sl.first == 0 ? my[0] : (sl.first == 1 ? my[1] : my[2]);
    m2x = sl.second == 1 ? mx[1] : (sl.second == 2 ? mx[2] : mx[0]);
    m2y = sl.second == 1 ? my[1] : (sl.second == 2 ? my[2] : my[0]);
  }
  const float X = (float)x + 0.5f, Y = (float)y + 0.5f;
  const bool two = sl.w2 != 0.0f;
  const float q1x = X + m1x, q1y = Y + m1y, q2x = X + m2x, q2y = Y + m2y;
  // kTrim: q = X + m with X >= 0.5, so q is never -0.0 (an exact zero sum is +0) and for q >= +0 the unsigned bit
  // order is the float order; negative values, NaNs and +inf have bits above those of (float)dw, so they fail as the
  // float compares do. Bitwise & on the bools: no short-circuit branches.
  const u32 bw = __builtin_bit_cast(u32, (float)p.dw), bh = __builtin_bit_cast(u32, (float)p.dh);
  const auto in_range = [&](float qx, float qy) -> bool {
    if (kTrim)
      return (!p.reset) & (__builtin_bit_cast(u32, qx) <= bw) & (__builtin_bit_cast(u32, qy) <= bh);
    return !p.reset && qx >= 0.0f && qx <= (float)p.dw && qy >= 0.0f && qy <= (float)p.dh;
  };
  const bool v1 = in_range(q1x, q1y);
  const bool v2 = in_range(q2x, q2y);
  const bool same = v1 && v2 && q1x == q2x && q1y == q2y;
  float fb[3] = {0.0f, 0.0f, 0.0f};
  if (!v1 || (two && !v2)) {
    const v4f c = *(const v4f*)((const char*)color + (mul24((u32)fy, (u32)p.rw * 16u) + (u32)fx * 16u));
    fb[0] = enc_e(p.e_prev * c[0]);
    fb[1] = enc_e(p.e_prev * c[1]);
    fb[2] = enc_e(p.e_prev * c[2]);
  }
  const auto sample = [&](float qx, float qy, float h[3]) {
    if (kPlanar)
      history_cr_planar(hist_prev, p.dw, p.dh, qx, qy, h);
    else if (kSpan)  // post_recon2* only: the host takes them for dw >= 4 only (recon2_supported)
      history_cr_span<kHist6>(hist_prev, p.dw, p.dh, qx, qy, h);
    else
      history_cr<kHist6>(hist_prev, p.dw, p.dh, qx, qy, h);
  };
  float h[3] = {fb[0], fb[1], fb[2]};
  if (v1) sample(q1x, q1y, h);
  float H[3];
#pragma unroll
  for (int i = 0; i < 3; ++i) H[i] = sl.w1 * h[i];
  if (two) {
    if (!same) {
      h[0] = fb[0]; h[1] = fb[1]; h[2] = fb[2];
      if (v2) sample(q2x, q2y, h);
    }
#pragma unroll
    for (int i = 0; i < 3; ++i) H[i] = H[i] + sl.w2 * h[i];
  }
  post_blend<kTrim>(g, sl, H, p, s, hh, oh);
}

#include "post_motion_outres.h"  // MoOff / MoOn, mo_core: motion at output resolution (post_mo.hip)

// ---- batched history taps (post_recon3*) ----------------------------------------------------------------------------
// The Catmull-Rom sample is split into a fetch (positions, all loads) and a finish (weights, arithmetic), so a pixel
// issues the loads of both candidates before it waits for either: one memory round trip per pixel instead of two.
// 8-byte history only (post_recon's format); needs dw >= 4. Same loads and arithmetic as history_cr_span, so
// cr_finish<false> is bit-identical to it; cr_finish<true> computes the tap weights in packed FP16 (see below).
typedef _Float16 v2h __attribute__((ext_vector_type(2)));

struct CrFetch {
  u32 d[4][8];  // raw texels per row: rows 1, 2: 4 texels (columns cs ... cs + 3); rows 0, 3: 2 texels (cs + 1, cs + 2)
  float fx, fy; // fractions
  int ix, iy, cs;
};

// Position part of the fetch (no loads).
WG_FN void cr_pos(int dw, float qx, float qy, CrFetch& f) {
  const float tx = qx - 0.5f, ty = qy - 0.5f;
  const float fx0 = __builtin_floorf(tx), fy0 = __builtin_floorf(ty);
  f.fx = tx - fx0;
  f.fy = ty - fy0;
  f.ix = (int)fx0;
  f.iy = (int)fy0;
  f.cs = clampi(f.ix - 1, 0, dw - 4);
}

// The span loads (6 x 16 bytes). Every load = uniform base + 32-bit offset.
WG_FN void cr_spans(const void* hist, int dw, int dh, CrFetch& f) {
  const char* base = (const char*)hist;
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    const u32 roff = mul24((u32)clampi(f.iy - 1 + j, 0, dh - 1), (u32)dw * 8u);
    if (j == 0 || j == 3) {
      const v4u_a2 a = *(const v4u_a2*)(base + (roff + (u32)(f.cs + 1) * 8u));
      f.d[j][0] = a[0]; f.d[j][1] = a[1]; f.d[j][2] = a[2]; f.d[j][3] = a[3];
    } else {
      const u32 o = roff + (u32)f.cs * 8u;
      const v4u_a2 a = *(const v4u_a2*)(base + o);
      const v4u_a2 b = *(const v4u_a2*)(base + (o + 16u));
      f.d[j][0] = a[0]; f.d[j][1] = a[1]; f.d[j][2] = a[2]; f.d[j][3] = a[3];
      f.d[j][4] = b[0]; f.d[j][5] = b[1]; f.d[j][6] = b[2]; f.d[j][7] = b[3];
    }
  }
}

// All loads of f are complete after this point, on every control-flow path. The compiler's wait insertion is
// path-insensitive: loads issued under one condition and consumed under an equivalent one look possibly pending
// afterwards, and the next write to those registers then waits on the in-order load counter, i.e. on loads issued
// later (here: the other candidate's). An empty asm statement that reads all destination registers forces one
// wait here and leaves no pending state. (Values never loaded are just read; nothing uses them.)
WG_FN void cr_complete(CrFetch& f) {
#if defined(__AMDGCN__)
#pragma unroll
  for (int j = 0; j < 4; ++j)
    asm volatile("" ::"v"(f.d[j][0]), "v"(f.d[j][1]), "v"(f.d[j][2]), "v"(f.d[j][3]), "v"(f.d[j][4]),
                 "v"(f.d[j][5]), "v"(f.d[j][6]), "v"(f.d[j][7]));
#else
  (void)f;
#endif
}

// Left or right image edge (ix - 1 != cs): the clamped columns, one texel at a time (overwrites the span values).
WG_FN void cr_edge(const void* hist, int dw, int dh, CrFetch& f) {
  const char* base = (const char*)hist;
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    const u32 roff = mul24((u32)clampi(f.iy - 1 + j, 0, dh - 1), (u32)dw * 8u);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      if ((j == 0 || j == 3) && (i == 0 || i == 3)) continue;
      const int k = (j == 0 || j == 3) ? i - 1 : i;  // texel slot in the row
      const v2u_a2 v = *(const v2u_a2*)(base + (roff + (u32)clampi(f.ix - 1 + i, 0, dw - 1) * 8u));
      f.d[j][2 * k] = v[0];
      f.d[j][2 * k + 1] = v[1];
    }
  }
}

template <bool kF16>
WG_FN void cr_finish(const CrFetch& f, float out[3]) {
  float wx[4], wy[4];
  if (kF16) {
    // Both axes' weights at once in packed FP16 (2 x the FP32 rate); the normalization below uses the rounded
    // weights, so the sample stays an exact weighted mean of its taps (weight error < 2^-11 relative).
    v2h t;
    t[0] = (_Float16)f.fx;
    t[1] = (_Float16)f.fy;
    const v2h one = {(_Float16)1.0f, (_Float16)1.0f}, half = {(_Float16)0.5f, (_Float16)0.5f};
    const v2h tt = t * t;
    const v2h w0 = t * __builtin_elementwise_fma(t, __builtin_elementwise_fma(t, -half, one), -half);
    const v2h w1 = __builtin_elementwise_fma(tt, __builtin_elementwise_fma(t, v2h{(_Float16)1.5f, (_Float16)1.5f},
                                                                          v2h{(_Float16)-2.5f, (_Float16)-2.5f}), one);
    const v2h w2 = t * __builtin_elementwise_fma(t, __builtin_elementwise_fma(t, v2h{(_Float16)-1.5f, (_Float16)-1.5f},
                                                                              v2h{(_Float16)2.0f, (_Float16)2.0f}), half);
    const v2h w3 = tt * __builtin_elementwise_fma(t, half, -half);
    wx[0] = (float)w0[0]; wx[1] = (float)w1[0]; wx[2] = (float)w2[0]; wx[3] = (float)w3[0];
    wy[0] = (float)w0[1]; wy[1] = (float)w1[1]; wy[2] = (float)w2[1]; wy[3] = (float)w3[1];
  } else {
    cr_weights(f.fx, wx);
    cr_weights(f.fy, wy);
  }
  float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f;
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    const int i0 = (j == 0 || j == 3) ? 1 : 0, i1 = (j == 0 || j == 3) ? 2 : 3;
    float r0 = 0.0f, r1 = 0.0f, r2 = 0.0f;
#pragma unroll
    for (int i = i0; i <= i1; ++i) {
      const int k = (j == 0 || j == 3) ? i - 1 : i;
      r0 = __builtin_fmaf(wx[i], half_of(f.d[j], 4 * k + 0), r0);
      r1 = __builtin_fmaf(wx[i], half_of(f.d[j], 4 * k + 1), r1);
      r2 = __builtin_fmaf(wx[i], half_of(f.d[j], 4 * k + 2), r2);
    }
    acc0 = __builtin_fmaf(wy[j], r0, acc0);
    acc1 = __builtin_fmaf(wy[j], r1, acc1);
    acc2 = __builtin_fmaf(wy[j], r2, acc2);
  }
  const float norm = (wx[1] + wx[2]) * (wy[0] + wy[3]) + (wx[0] + wx[1] + wx[2] + wx[3]) * (wy[1] + wy[2]);
  float inv;
  if (kF16) {
    // norm is in [0.984, 1] (about that, with the rounded weights): 1 / norm = 1 + e + e^2 + e^3 + O(e^4),
    // e = 1 - norm <= 0.017 (error < 1e-7); three FMAs instead of a transcendental v_rcp (4 FMA slots).
    const float e = 1.0f - norm;
    inv = __builtin_fmaf(e, __builtin_fmaf(e, __builtin_fmaf(e, 1.0f, 1.0f), 1.0f), 1.0f);
  } else {
    inv = wg_rcp(norm);
  }
  out[0] = acc0 * inv;
  out[1] = acc1 * inv;
  out[2] = acc2 * inv;
}

// post_core with batched history taps: the fallback colour (raw, as staged in LDS) is passed in instead of read from
// global memory. kF16 = false: bit-identical to post_core<false, true>.
template <bool kF16>
WG_FN void post_core_b(const Gather& g, int x, int y, const float mx[3], const float my[3], const float col[3],
                       const WgPostParams& p, float s, const void* hist_prev, float hh[3], float oh[3]) {
  const PostSel sl = post_select(g);
  const float X = (float)x + 0.5f, Y = (float)y + 0.5f;
  const int ncand = sl.w2 != 0.0f ? 2 : 1;
  const float q1x = X + mx[sl.first], q1y = Y + my[sl.first];
  const float q2x = X + mx[sl.second], q2y = Y + my[sl.second];
  const auto in_range = [&](float qx, float qy) {
    return !p.reset && qx >= 0.0f && qx <= (float)p.dw && qy >= 0.0f && qy <= (float)p.dh;
  };
  const bool v1 = in_range(q1x, q1y);
  const bool v2 = ncand == 2 && in_range(q2x, q2y);
  const bool same = v1 && v2 && q1x == q2x && q1y == q2y;
  const bool need2 = v2 && !same;
  // Both candidates' loads are issued before either is used; the rare edge reloads come after both.
  CrFetch f1, f2;
  if (!WG_AB_HIST) {
    cr_pos(p.dw, q1x, q1y, f1);
    cr_pos(p.dw, q2x, q2y, f2);
    if (v1) cr_spans(hist_prev, p.dw, p.dh, f1);
    if (need2) cr_spans(hist_prev, p.dw, p.dh, f2);
    cr_complete(f1);
    cr_complete(f2);
    if (v1 && f1.ix - 1 != f1.cs) cr_edge(hist_prev, p.dw, p.dh, f1);
    if (need2 && f2.ix - 1 != f2.cs) cr_edge(hist_prev, p.dw, p.dh, f2);
    cr_complete(f1);
    cr_complete(f2);
  }
  float fb[3] = {0.0f, 0.0f, 0.0f};
  if (!v1 || (ncand == 2 && !v2)) {
    fb[0] = enc_e(p.e_prev * col[0]);
    fb[1] = enc_e(p.e_prev * col[1]);
    fb[2] = enc_e(p.e_prev * col[2]);
  }
  float h1[3], h2[3];
  if (v1) {
    if (WG_AB_HIST) { h1[0] = q1x * 0x1p-12f; h1[1] = q1y * 0x1p-12f; h1[2] = 0.5f; }
    else cr_finish<kF16>(f1, h1);
  } else {
    h1[0] = fb[0]; h1[1] = fb[1]; h1[2] = fb[2];
  }
  float H[3];
#pragma unroll
  for (int i = 0; i < 3; ++i) H[i] = sl.w1 * h1[i];
  if (ncand == 2) {
    if (same) {
      h2[0] = h1[0]; h2[1] = h1[1]; h2[2] = h1[2];
    } else if (v2) {
      if (WG_AB_HIST) { h2[0] = q2x * 0x1p-12f; h2[1] = q2y * 0x1p-12f; h2[2] = 0.5f; }
      else cr_finish<kF16>(f2, h2);
    } else {
      h2[0] = fb[0]; h2[1] = fb[1]; h2[2] = fb[2];
    }
#pragma unroll
    for (int i = 0; i < 3; ++i) H[i] = H[i] + sl.w2 * h2[i];
  }
  post_blend(g, sl, H, p, s, hh, oh);
}

// spec 06 draft 2: display = clamp(E(O^), 0, 2.45) (keeps the final linear output within FP16)
WG_FN _Float16 display_value(float o) { return (_Float16)clampf(enc_e(o), 0.0f, kDisplayMax); }

// post_recon: steps 3-6 for one D pixel (x, y) at render position (px, py), motions read from global memory. Writes
// the full history (RGBA16F) and display texels when `store` and returns the linear new history H^ in hh.
WG_FN void finish_pixel(const Gather& g, int x, int y, float px, float py, bool store, const WgPostParams& p, float s,
                        const v4f* color, const v4h* hist_prev, const u32* m_near, const u32* m_center,
                        const u32* m_far, v4h* hist_out, _Float16* display, float hh[3]) {
  // 3. Motion candidates: m_near and m_far point-read at (floor px, floor py), m_center bilinear at p.
  const int fx = clampi(floor_i(px), 0, p.rw - 1), fy = clampi(floor_i(py), 0, p.rh - 1);
  float mx[3], my[3];
  if (WG_AB_MOTION) {
    mx[0] = my[0] = 0.25f;
    mx[1] = my[1] = 0.5f;
    mx[2] = my[2] = -0.75f;
  } else {
    unpack_h2(m_near[fy * p.rw + fx], mx[0], my[0]);
    unpack_h2(m_far[fy * p.rw + fx], mx[2], my[2]);
    motion_bilinear(px, py,
                    [&](int c, int r) { return m_center[clampi(r, 0, p.rh - 1) * p.rw + clampi(c, 0, p.rw - 1)]; },
                    mx[1], my[1]);
  }
  float oh[3];
  post_core<false>(g, x, y, fx, fy, mx, my, p, s, color, hist_prev, hh, oh);

  // 6. Full history (RGBA16F, alpha 0) and display image (three FP16 planes).
  if (store && !WG_AB_HISTW) hist_store<false>(hist_out, y * p.dw + x, enc_e(hh[0]), enc_e(hh[1]), enc_e(hh[2]));
  if (store && !WG_AB_DISPLAY) {
    const int pitch = display_pitch(p.dw), plane = pitch * p.dh, di = y * pitch + x;
    display[di] = display_value(oh[0]);
    display[plane + di] = display_value(oh[1]);
    display[2 * plane + di] = display_value(oh[2]);
  }
}

}  // namespace

// Stage "post". color: R, 4 x f32 (alpha ignored); A, B: R, 4 x u8 (codes k, value k / 255; byte 3 ignored);
// C: R, 4 x f16 (a, b, c, -); hist_prev / hist_out: D, 4 x f16 (E-encoded RGB, alpha 0); m_*: R, 2 x f16;
// w: model data (w[0] = s); reduced: floor(D / 2), R10G10B10A2 (R in bits 0..9, alpha 0); display: three FP16
// planes of D (R, G, B), row pitch display_pitch(dw). Grid: ceil(dw / 32) * ceil(ceil(dh / 2) / 8) workgroups of 256 lanes.
WG_KERNEL_N(post_recon, kPostThreads)(const v4f* color, const u32* A, const u32* B, const v4h* C,
                                      const v4h* hist_prev, const u32* m_near, const u32* m_center,
                                      const u32* m_far, const float* w, v4h* hist_out, u32* reduced,
                                      _Float16* display, WgPostParams p, const float* expo) {
  // Per-frame runtime (spec 10): the exposure pair [e, e_prev] (raw values) may live in GPU memory instead of the
  // parameters. Same values as post.py computes on the host: 0 -> 1, ratio = RN32(e / e_prev) via binary64.
  if (expo) {
    p.e = expo[0] == 0.0f ? 1.0f : expo[0];
    p.e_prev = expo[1] == 0.0f ? 1.0f : expo[1];
    p.ratio = (float)((double)p.e / (double)p.e_prev);
  }
  WG_LDS PostTexel sh_tex[kStageCap];
  WG_LDS float sh_red[kPostThreads * 3];

  const int lane = (int)wg_lane_id();
  const int tiles_x = (p.dw + kTileX - 1) / kTileX;
  const int tile = (int)wg_block_id();
  const int x0 = (tile % tiles_x) * kTileX;
  const int k0 = (tile / tiles_x) * kTilePairs;
  const int npairs = (p.dh + 1) / 2;
  const float s = w[0];

  // Render footprint of the tile: window columns of its first and last column, window rows of its first and last
  // row pair (a_x and b do not decrease with x and y).
  const int xl = mini(x0 + kTileX - 1, p.dw - 1), kl = mini(k0 + kTilePairs - 1, npairs - 1);
  const int c0 = floor_i(pos_x(x0, p) - 0.5f) - 1, c1 = floor_i(pos_x(xl, p) - 0.5f) + 1;
  const int r0 = floor_i(pos_y(2 * k0, p) - 0.5f) - 1, r1 = floor_i(pos_y(2 * kl, p) - 0.5f) + 1;
  const int ncol = c1 - c0 + 1, nrow = r1 - r0 + 1;
  const bool staged = ncol > 0 && nrow > 0 && ncol <= kStageCap && nrow <= kStageCap && ncol * nrow <= kStageCap;
  if (staged) {
    for (int i = lane; i < ncol * nrow; i += kPostThreads) {
      const int r = i / ncol, c = i - r * ncol;
      const int idx = clampi(r0 + r, 0, p.rh - 1) * p.rw + clampi(c0 + c, 0, p.rw - 1);
      sh_tex[i] = load_texel(color, A, B, C, idx, p.e);
    }
  }
  wg_barrier();

  // This lane: column x, rows y0 = 2k and y0 + 1. Lanes past the image edge compute a clamped duplicate and store
  // nothing (they still take part in the barrier below).
  const bool active = x0 + (lane & 31) < p.dw && k0 + (lane >> 5) < npairs;
  const int x = mini(x0 + (lane & 31), p.dw - 1);
  const int y0 = 2 * mini(k0 + (lane >> 5), npairs - 1);
  const bool has1 = y0 + 1 < p.dh;
  const float px = pos_x(x, p), py0 = pos_y(y0, p), py1 = pos_y(y0 + 1, p);
  const int ax = floor_i(px - 0.5f), b = floor_i(py0 - 0.5f);

  Gather g0, g1;
  gather_zero(g0);
  gather_zero(g1);
#pragma unroll
  for (int j = WG_AB_GATHER ? 0 : -1; j <= (WG_AB_GATHER ? 0 : 1); ++j) {
    const float v0 = ((float)(b + j) + 0.5f) - py0;
    const float v1 = ((float)(b + j) + 0.5f) - py1;
    const float vv0 = kTapK * v0 * v0, vv1 = kTapK * v1 * v1;
#pragma unroll
    for (int i = WG_AB_GATHER ? 0 : -1; i <= (WG_AB_GATHER ? 0 : 1); ++i) {
      const float u = ((float)(ax + i) + 0.5f) - px;
      const float uu = kTapK * u * u, uv0 = 2.0f * kTapK * u * v0, uv1 = 2.0f * kTapK * u * v1;
      PostTexel t;
      if (staged) {
        t = sh_tex[(b + j - r0) * ncol + (ax + i - c0)];
      } else {
        const int idx = clampi(b + j, 0, p.rh - 1) * p.rw + clampi(ax + i, 0, p.rw - 1);
        t = load_texel(color, A, B, C, idx, p.e);
      }
      gather_add(g0, t, uu, vv0, uv0);
      gather_add(g1, t, uu, vv1, uv1);
    }
  }

  float h0[3], h1[3] = {0.0f, 0.0f, 0.0f};
  finish_pixel(g0, x, y0, px, py0, active, p, s, color, hist_prev, m_near, m_center, m_far, hist_out, display, h0);
  if (has1)
    finish_pixel(g1, x, y0 + 1, px, py1, active, p, s, color, hist_prev, m_near, m_center, m_far, hist_out, display,
                 h1);

  // Reduced history: mean of H^ over the 2 x 2 block (x, x + 1) x (y0, y0 + 1); lane x + 1 is lane ^ 1.
  if (WG_AB_REDUCED) {
    if (ab_live(h0[0] + h0[1] + h0[2] + h1[0] + h1[1] + h1[2])) reduced[0] = 0u;
    return;
  }
#pragma unroll
  for (int i = 0; i < 3; ++i) sh_red[lane * 3 + i] = h0[i] + h1[i];
  wg_barrier();
  if (active && has1 && (x & 1) == 0 && x + 1 < p.dw) {
    u32 code = 0;
#pragma unroll
    for (int i = 0; i < 3; ++i) {
      const float m = (sh_red[lane * 3 + i] + sh_red[(lane ^ 1) * 3 + i]) * 0.25f;
      const float v = clampf(enc_e(m) * (1.0f / 2.45f), 0.0f, 1.0f);
      code |= (u32)wg_rint(v * 1023.0f) << (10 * i);
    }
    reduced[(y0 / 2) * (p.dw / 2) + x / 2] = code;
  }
}

// ---------------------------------------------------------------------------------------------------------------
// post_recon2*: drop-in replacements for post_recon (same arguments, layouts and grid; bit-identical results).
// Differences: the render footprint is staged with the codes already converted to floats and with the three motion
// surfaces (no global motion reads per pixel); lanes l and l ^ 16 of a wave hold the D columns x and x + 1, so the
// reduced history's 2 x 2 means use one lane exchange (no LDS round trip, no second barrier); the pixels' steps 3-6
// run one pixel at a time (fewer registers). Variants: _r = the 3 x 3 gather's tap rows in a loop (fewer registers,
// more waves); _ns = history taps loaded one by one as post_recon (otherwise as spans, history_cr_span).

namespace {

constexpr int kR2Cap = 336;  // render pixels staged per workgroup (enough for D >= about 1.43 R; see recon2_supported)

// LDS texel of the post_recon2* kernels: the features of one render pixel (PostTexelF) and its three raw motions in
// one 48-byte record (three 16-byte LDS loads for the features; 336 x 48 = 16128 bytes per workgroup, so 8 workgroups
// of 256 lanes fit a WGP's 128 KB: 8 waves64 per SIMD instead of 6 with PostTexelF + three separate motion arrays,
// 20160 bytes). The six codes are stored as FP16, which is exact: s0, s1, s2, g, cbh, cbo are only ever produced by
// texel_f as byte_f(v, i) = (float)((v >> 8 i) & 0xff) of a u32 (t.b, or t.a ^ 0xffff00), for every staging source
// (load_texel = base A, B, C layout; load_texel12 = pack12 layout; the WG_AB_STAGE ablation constant), so each is an
// integer in 0..255, and FP16 (11-bit significand) represents every integer up to 2048 exactly: (float)(_Float16)v
// == v. The gather converts back with (float)h (v_fma_mix_f32 / v_cvt_f32_f16, exact), so the sums are bit-identical
// to the f32 texel. ca, cb, cc stay f32 (a 32-byte record with them in FP16 measured slower at equal occupancy).
struct alignas(16) R2Texel {
  float ec0, ec1, ec2;  // E_in(e * colour)
  float ca, cb, cc;     // kernel shape C
  _Float16 s0, s1, s2;  // B codes (sigma_near, sigma_center, sigma_far), integers 0..255
  _Float16 g, cbh, cbo; // A: g code, 255 - beta_h code, 255 - beta_o code, integers 0..255
  u32 m[3];             // m_near, m_center, m_far (FP16 x, y each), as read from global memory
};
static_assert(sizeof(R2Texel) == 48, "R2Texel: 48 bytes");

WG_FN void r2_store_features(R2Texel& d, const PostTexelF& f) {
  d.ec0 = f.ec0;
  d.ec1 = f.ec1;
  d.ec2 = f.ec2;
  d.ca = f.ca;
  d.cb = f.cb;
  d.cc = f.cc;
  d.s0 = (_Float16)f.s0;
  d.s1 = (_Float16)f.s1;
  d.s2 = (_Float16)f.s2;
  d.g = (_Float16)f.g;
  d.cbh = (_Float16)f.cbh;
  d.cbo = (_Float16)f.cbo;
}

WG_FN PostTexelF r2_features(const R2Texel& t) {
  PostTexelF f;
  f.ec0 = t.ec0;
  f.ec1 = t.ec1;
  f.ec2 = t.ec2;
  f.ca = t.ca;
  f.cb = t.cb;
  f.cc = t.cc;
  f.s0 = (float)t.s0;
  f.s1 = (float)t.s1;
  f.s2 = (float)t.s2;
  f.g = (float)t.g;
  f.cbh = (float)t.cbh;
  f.cbo = (float)t.cbo;
  return f;
}

// Plain struct (a later fused filter epilogue can alias its LDS with tex[] through a union).
struct R2Shared {
  R2Texel tex[kR2Cap];
};

// Output mode D = O (post_recon2_r_out): the output filter is the identity, so the output pixel is the decode of the
// FP16 display value exactly as the filter decodes a filtered value (clamp to [1e-4, 3], Dec, times 0.01 / e), stored
// as RGBA16F (alpha 1, round to nearest even).
WG_FN void store_out_o16(void* out, u32 i, const _Float16 d[3], float dec_scale) {
  v4h t;
#pragma unroll
  for (int c = 0; c < 3; ++c) t[c] = framebuffer_half(dec_m1((float)d[c]) * dec_scale);
  t[3] = (_Float16)1.0f;
  *(v4h*)((char*)out + i * 8u) = t;
}

// kOut: write the output (store_out_o16, into `display` taken as the RGBA16F output, D = O) instead of the display
// image.
template <bool kRolled, bool kSpan, bool kHist6 = false, bool kOut = false, bool kPlanar = false, bool kPack12 = false,
          bool kDiet = (WG_POST_DIET != 0), class MoT = MoOff>
WG_FN void post_recon2_body(const v4f* color, const u32* A, const u32* B, const v4h* C, const v4h* hist_prev,
                            const u32* m_near, const u32* m_center, const u32* m_far, const float* w,
                            v4h* hist_out, u32* reduced, _Float16* display, WgPostParams p, const float* expo,
                            R2Shared& sh, float dec_scale = 0.0f, int tile0 = 0, u32 div_magic = 0,
                            const MoT mo = MoT()) {
  // history_cr_span (kSpan) needs dw >= 4; the host takes post_recon below 4 columns (recon2_supported)
  if (kSpan && !kPlanar && p.dw < 4) return;
  if (expo) {  // as post_recon
    p.e = expo[0] == 0.0f ? 1.0f : expo[0];
    p.e_prev = expo[1] == 0.0f ? 1.0f : expo[1];
    if (kDiet) {  // precomputed by exposure_pair_diet (the same expressions, once)
      p.ratio = expo[3];
      dec_scale = expo[4];
    } else {
      p.ratio = (float)((double)p.e / (double)p.e_prev);
      dec_scale = (float)(0.01 / (double)p.e);  // as the filter with the exposure pair in GPU memory
    }
  }
  const int lane = (int)wg_lane_id();
  const int tiles_x = (p.dw + kTileX - 1) / kTileX;
  const int tile = (int)wg_block_id() + tile0;  // tile0: first tile of a band (banded launches), else 0
  int x0, k0;
  if (kDiet) {  // tile / tiles_x by the host's reciprocal (verified exact on the host for every tile of the launch)
    const int ty = (int)wg_umulhi((u32)tile, div_magic);
    x0 = (tile - ty * tiles_x) * kTileX;
    k0 = ty * kTilePairs;
  } else {
    x0 = (tile % tiles_x) * kTileX;
    k0 = (tile / tiles_x) * kTilePairs;
  }
  const int npairs = (p.dh + 1) / 2;
  const float s = w[0];

  // Render footprint: window columns of the tile's first and last column; window rows of its first and last row
  // pair, plus one row below for the motion reads of the odd rows.
  const int xl = mini(x0 + kTileX - 1, p.dw - 1), kl = mini(k0 + kTilePairs - 1, npairs - 1);
  const int c0 = floor_i(pos_x(x0, p) - 0.5f) - 1, c1 = floor_i(pos_x(xl, p) - 0.5f) + 1;
  const int r0 = floor_i(pos_y(2 * k0, p) - 0.5f) - 1, r1 = floor_i(pos_y(2 * kl, p) - 0.5f) + 2;
  const int ncol = c1 - c0 + 1, nrow = r1 - r0 + 1;
  const int nstage = ncol * nrow < kR2Cap ? ncol * nrow : kR2Cap;  // the host checks recon2_supported
  const float inv_ncol = 1.0f / (float)ncol;
  for (int i = lane; i < nstage; i += kPostThreads) {
    const int r = (int)(((float)i + 0.5f) * inv_ncol), c = i - mul_i24(r, ncol);
    const int idx = (int)mul24((u32)clampi(r0 + r, 0, p.rh - 1), (u32)p.rw) + clampi(c0 + c, 0, p.rw - 1);
    r2_store_features(sh.tex[i], texel_f(kPack12 ? load_texel12(color, A, (const u32*)C, idx, p.e)  // pack12: A8, C4
                                                 : load_texel(color, A, B, C, idx, p.e)));
    if (MoT::on) continue;  // the motion surfaces are not read
    const u32 mo = (u32)idx * 4u;
    sh.tex[i].m[0] = *(const u32*)((const char*)m_near + mo);
    sh.tex[i].m[1] = *(const u32*)((const char*)m_center + mo);
    sh.tex[i].m[2] = *(const u32*)((const char*)m_far + mo);
  }
  wg_barrier();
  const auto sidx = [&](int c, int r) { return clampi(mul_i24(r - r0, ncol) + (c - c0), 0, kR2Cap - 1); };

  // This lane: column x (lane ^ 16 has x + 1), rows y0 = 2k and y0 + 1.
  const int wl = lane & 31;
  const int col = 2 * (wl & 15) + (wl >> 4);
  const bool active = x0 + col < p.dw && k0 + (lane >> 5) < npairs;
  const int x = mini(x0 + col, p.dw - 1);
  const int y0 = 2 * mini(k0 + (lane >> 5), npairs - 1);
  const bool has1 = y0 + 1 < p.dh;
  const float px = pos_x(x, p), py0 = pos_y(y0, p), py1 = pos_y(y0 + 1, p);
  const int ax = floor_i(px - 0.5f), b = floor_i(py0 - 0.5f);
  MoQ mq;
  if (MoT::on) mo_fetch(mo, x, y0, p.dw, p.dh, mq);

  Gather g0, g1;
  gather_zero(g0);
  gather_zero(g1);
  const auto tap_row = [&](int j) {
    const float v0 = ((float)(b + j) + 0.5f) - py0;
    const float v1 = ((float)(b + j) + 0.5f) - py1;
    const float vv0 = kTapK * v0 * v0, vv1 = kTapK * v1 * v1;
#pragma unroll
    for (int i = WG_AB_GATHER ? 0 : -1; i <= (WG_AB_GATHER ? 0 : 1); ++i) {
      const float u = ((float)(ax + i) + 0.5f) - px;
      const float uu = kTapK * u * u, uv0 = 2.0f * kTapK * u * v0, uv1 = 2.0f * kTapK * u * v1;
      const PostTexelF t = r2_features(sh.tex[sidx(ax + i, b + j)]);
      gather_add_f(g0, t, uu, vv0, uv0);
      gather_add_f(g1, t, uu, vv1, uv1);
    }
  };
  if (kRolled) {
#pragma nounroll
    for (int j = WG_AB_GATHER ? 0 : -1; j <= (WG_AB_GATHER ? 0 : 1); ++j) tap_row(j);
  } else {
#pragma unroll
    for (int j = WG_AB_GATHER ? 0 : -1; j <= (WG_AB_GATHER ? 0 : 1); ++j) tap_row(j);
  }

  float hsum[3] = {0.0f, 0.0f, 0.0f};
  const int fx = clampi(floor_i(px), 0, p.rw - 1);
#pragma nounroll
  for (int k = 0; k < (has1 ? 2 : 1); ++k) {
    const int y = y0 + k;
    const float py = k == 0 ? py0 : py1;
    const int fy = clampi(floor_i(py), 0, p.rh - 1);
    float mx[3], my[3];
    if (WG_AB_MOTION) {
      mx[0] = my[0] = 0.25f;
      mx[1] = my[1] = 0.5f;
      mx[2] = my[2] = -0.75f;
    } else {
      const int pxi = floor_i(px), pyi = floor_i(py);
      unpack_h2(sh.tex[sidx(pxi, pyi)].m[0], mx[0], my[0]);
      unpack_h2(sh.tex[sidx(pxi, pyi)].m[2], mx[2], my[2]);
      motion_bilinear_mix(px, py, [&](int c, int r) { return sh.tex[sidx(c, r)].m[1]; }, mx[1], my[1]);
    }
    float hh[3], oh[3];
    if (MoT::on)
      mo_core<false>(mo, mq, k, k == 0 ? g0 : g1, x, y, fx, fy, p, s, color, hist_prev, hh, oh);
    else
      post_core_s<kHist6, kSpan, kPlanar>(k == 0 ? g0 : g1, x, y, fx, fy, mx, my, p, s, color, hist_prev, hh, oh);
    if (active) {
      if (!WG_AB_HISTW || ab_live(enc_e(hh[0]) + enc_e(hh[1]) + enc_e(hh[2]))) {
        if (kPlanar)
          hist_store_planar(hist_out, hp_bbase(p.dw, p.dh), mul24((u32)y, (u32)p.dw) + (u32)x, enc_e(hh[0]),
                            enc_e(hh[1]), enc_e(hh[2]));
        else if (kHist6)
          hist_store<true>(hist_out, y * p.dw + x, enc_e(hh[0]), enc_e(hh[1]), enc_e(hh[2]));
        else {  // 32-bit byte offset on the uniform base (mul24; recon2_supported keeps 8 dw dh < 2^32)
          v4h t;
          t[0] = (_Float16)enc_e(hh[0]);
          t[1] = (_Float16)enc_e(hh[1]);
          t[2] = (_Float16)enc_e(hh[2]);
          t[3] = (_Float16)0.0f;
          *(v4h*)((char*)hist_out + (mul24((u32)y, (u32)p.dw * 8u) + (u32)x * 8u)) = t;
        }
      }
      if (kOut) {
        const _Float16 d[3] = {display_value(oh[0]), display_value(oh[1]), display_value(oh[2])};
        store_out_o16(display, mul24((u32)y, (u32)p.dw) + (u32)x, d, dec_scale);
      } else if (!WG_AB_DISPLAY ||
                 ab_live((float)display_value(oh[0]) + (float)display_value(oh[1]) + (float)display_value(oh[2]))) {
        const u32 pitch2 = (u32)display_pitch(p.dw) * 2u, plane2 = pitch2 * (u32)p.dh;
        const u32 di = mul24((u32)y, pitch2) + (u32)x * 2u;
        char* dbase = (char*)display;
        *(_Float16*)(dbase + di) = display_value(oh[0]);
        *(_Float16*)(dbase + (di + plane2)) = display_value(oh[1]);
        *(_Float16*)(dbase + (di + 2u * plane2)) = display_value(oh[2]);
      }
    }
#pragma unroll
    for (int c = 0; c < 3; ++c) hsum[c] += hh[c];
  }

  // Reduced history: mean of H^ over the 2 x 2 block; the partner column x + 1 is in lane ^ 16.
  if (WG_AB_REDUCED) {
    if (ab_live(hsum[0] + hsum[1] + hsum[2])) reduced[0] = 0u;
    return;
  }
  float nb[3];
#pragma unroll
  for (int c = 0; c < 3; ++c) nb[c] = wg_xor16(hsum[c]);
  if (active && has1 && wl < 16 && x + 1 < p.dw) {
    u32 code = 0;
#pragma unroll
    for (int c = 0; c < 3; ++c) {
      const float m = (hsum[c] + nb[c]) * 0.25f;
      const float v = clampf(enc_e(m) * (1.0f / 2.45f), 0.0f, 1.0f);
      code |= (u32)wg_rint(v * 1023.0f) << (10 * c);
    }
    *(u32*)((char*)reduced + (mul24((u32)(y0 >> 1), (u32)(p.dw >> 1) * 4u) + (u32)(x >> 1) * 4u)) = code;
  }
}

}  // namespace

#define WG_RECON2_KERNEL_X(name, ROLLED, SPAN, HIST6)                                                              \
  WG_KERNEL_N(name, kPostThreads)(const v4f* color, const u32* A, const u32* B, const v4h* C,                     \
                                  const v4h* hist_prev, const u32* m_near, const u32* m_center, const u32* m_far, \
                                  const float* w, v4h* hist_out, u32* reduced, _Float16* display,                \
                                  WgPostParams p, const float* expo) {                                            \
    WG_LDS R2Shared sh;                                                                                           \
    post_recon2_body<ROLLED, SPAN, HIST6>(color, A, B, C, hist_prev, m_near, m_center, m_far, w, hist_out,       \
                                          reduced, display, p, expo, sh);                                         \
  }
#define WG_RECON2_KERNEL(name, ROLLED, SPAN) WG_RECON2_KERNEL_X(name, ROLLED, SPAN, false)
WG_RECON2_KERNEL(post_recon2, false, true)
WG_RECON2_KERNEL(post_recon2_r, true, true)
WG_RECON2_KERNEL(post_recon2_ns, false, false)
WG_RECON2_KERNEL(post_recon2_r_ns, true, false)
// post_recon2_r with the 6-byte full history (RGB16F tightly packed, the format of post_fused: hist_prev / hist_out
// hold 3 halves per pixel, 6 dw dh + 8 bytes); same values, so bit-identical results. (Wave64: post_w64.co, the
// whole file built with -mwavefrontsize64; a single wave64 kernel cannot inline the wave32 helpers.)
WG_RECON2_KERNEL_X(post_recon2_r_h6, true, true, true)
// post_recon2_r with the planar 6-byte history (history_cr_planar: RG16F plane + B16F plane, 6 dw dh + 8 bytes, all
// accesses aligned); bit-identical. _band: the banded launch (post_recon2_r_band, tile0 after expo).
WG_KERNEL_N(post_recon2_r_hp, kPostThreads)(const v4f* color, const u32* A, const u32* B, const v4h* C,
                                            const v4h* hist_prev, const u32* m_near, const u32* m_center,
                                            const u32* m_far, const float* w, v4h* hist_out, u32* reduced,
                                            _Float16* display, WgPostParams p, const float* expo) {
  WG_LDS R2Shared sh;
  post_recon2_body<true, true, false, false, true>(color, A, B, C, hist_prev, m_near, m_center, m_far, w, hist_out,
                                                   reduced, display, p, expo, sh);
}
WG_KERNEL_N(post_recon2_r_hp_band, kPostThreads)(const v4f* color, const u32* A, const u32* B, const v4h* C,
                                                 const v4h* hist_prev, const u32* m_near, const u32* m_center,
                                                 const u32* m_far, const float* w, v4h* hist_out, u32* reduced,
                                                 _Float16* display, WgPostParams p, const float* expo, int tile0) {
  WG_LDS R2Shared sh;
  post_recon2_body<true, true, false, false, true>(color, A, B, C, hist_prev, m_near, m_center, m_far, w, hist_out,
                                                   reduced, display, p, expo, sh, 0.0f, tile0);
}

// The same reconstructions reading dec0's "pack12" coefficients (A = image A8, 8 bytes per pixel; C = image C4,
// 4 bytes per pixel; B unused, may be null): bit-identical. _band kernels take tile0 after expo.
#define WG_RECON2_P12_KERNEL(name, PLANAR, BAND)                                                                    \
  WG_KERNEL_N(name, kPostThreads)(const v4f* color, const u32* A, const u32* B, const v4h* C,                     \
                                  const v4h* hist_prev, const u32* m_near, const u32* m_center, const u32* m_far, \
                                  const float* w, v4h* hist_out, u32* reduced, _Float16* display,                \
                                  WgPostParams p, const float* expo, int tile0) {                                 \
    WG_LDS R2Shared sh;                                                                                           \
    post_recon2_body<true, true, false, false, PLANAR, true>(color, A, B, C, hist_prev, m_near, m_center, m_far, \
                                                             w, hist_out, reduced, display, p, expo, sh, 0.0f,    \
                                                             BAND ? tile0 : 0);                                   \
  }
WG_RECON2_P12_KERNEL(post_recon2_r_p12, false, false)
WG_RECON2_P12_KERNEL(post_recon2_r_hp_p12, true, false)
WG_RECON2_P12_KERNEL(post_recon2_r_band_p12, false, true)
WG_RECON2_P12_KERNEL(post_recon2_r_hp_band_p12, true, true)

// Reconstruction at D = O with the identity output filter (runtime option recon = output): post_recon2_r writing the
// final output instead of the display image. Arguments as post_recon2_r with `display` replaced by out (RGBA16F,
// ow * oh * 8 bytes, row pitch 8 ow, ow = dw, oh = dh) and dec_scale (0.01 / e; from the exposure pair when expo is
// not null) after the params; grid as post_recon2_r (needs recon2_supported: D >= about 1.43 R per axis). Full and
// reduced history as post_recon2_r. Bit-identical to post_recon2_r + post_decode_o16 at D = O.
WG_KERNEL_N(post_recon2_r_out, kPostThreads)(const v4f* color, const u32* A, const u32* B, const v4h* C,
                                             const v4h* hist_prev, const u32* m_near, const u32* m_center,
                                             const u32* m_far, const float* w, v4h* hist_out, u32* reduced, void* out,
                                             WgPostParams p, float dec_scale, const float* expo) {
  WG_LDS R2Shared sh;
  post_recon2_body<true, true, false, true>(color, A, B, C, hist_prev, m_near, m_center, m_far, w, hist_out, reduced,
                                            (_Float16*)out, p, expo, sh, dec_scale);
}

// post_recon2_r_out with the WG_POST_DIET cuts (runtime option post_diet=1); div_magic: ceil(2^32 / tiles_x).
WG_KERNEL_N(post_recon2_r_out_diet, kPostThreads)(const v4f* color, const u32* A, const u32* B, const v4h* C,
                                                  const v4h* hist_prev, const u32* m_near, const u32* m_center,
                                                  const u32* m_far, const float* w, v4h* hist_out, u32* reduced,
                                                  void* out, WgPostParams p, float dec_scale, const float* expo,
                                                  u32 div_magic) {
  WG_LDS R2Shared sh;
  post_recon2_body<true, true, false, true, false, false, true>(color, A, B, C, hist_prev, m_near, m_center, m_far, w,
                                                                hist_out, reduced, (_Float16*)out, p, expo, sh,
                                                                dec_scale, 0, div_magic);
}

// The identity output filter as a kernel (reference for post_recon2_r_out, and the filter step of D = O when the
// display image is written): display planes (D = O, pitch display_pitch(dw)) -> out RGBA16F as store_out_o16. One
// lane per pixel, grid ceil(dw * dh / 256) x 256 lanes.
WG_KERNEL_N(post_decode_o16, kPostThreads)(const _Float16* display, void* out, int dw, int dh, float dec_scale,
                                           const float* expo) {
  if (expo) dec_scale = (float)(0.01 / (double)(expo[0] == 0.0f ? 1.0f : expo[0]));
  const u32 i = wg_block_id() * (u32)kPostThreads + wg_lane_id();
  if (i >= (u32)dw * (u32)dh) return;
  const u32 y = i / (u32)dw, x = i - y * (u32)dw;
  const u32 plane = (u32)display_pitch(dw) * (u32)dh, di = y * (u32)display_pitch(dw) + x;
  const _Float16 d[3] = {display[di], display[plane + di], display[2u * plane + di]};
  store_out_o16(out, i, d, dec_scale);
}

// Banded two-kernel path: post_recon2_r and post_filter_c1_s<S> over horizontal bands, launched alternately
// (reconstruction of band b, then the filter rows whose 7-row footprint is complete), so that each band of the display
// image is still in the caches when the filter reads it. Arguments as post_recon2_r / post_filter_c1_s4 plus tile0,
// the first workgroup tile of the band (tile = block + tile0; tiles row-major as in the unbanded grid); the grid of a
// launch is the band's tile count. Same arithmetic: bit-identical to the unbanded kernels. Band layout:
// runner/post.py: band_plan.
WG_KERNEL_N(post_recon2_r_band, kPostThreads)(const v4f* color, const u32* A, const u32* B, const v4h* C,
                                              const v4h* hist_prev, const u32* m_near, const u32* m_center,
                                              const u32* m_far, const float* w, v4h* hist_out, u32* reduced,
                                              _Float16* display, WgPostParams p, const float* expo, int tile0) {
  WG_LDS R2Shared sh;
  post_recon2_body<true, true>(color, A, B, C, hist_prev, m_near, m_center, m_far, w, hist_out, reduced, display, p,
                               expo, sh, 0.0f, tile0);
}

// ---------------------------------------------------------------------------------------------------------------
// post_recon3*: drop-in replacements for post_recon as post_recon2_r, plus: the raw colour of the footprint is staged
// too (the fallback of an invalid candidate needs no global read), and each pixel issues the history loads of both
// candidates before waiting (post_core_b). post_recon3: bit-identical to post_recon. post_recon3_h: the history tap
// weights in packed FP16 and 1 / norm without v_rcp (cr_finish<true>), and the Gaussian's a u^2 term shared by the
// two rows of a pair (a different rounding order); within spec 09, see post_variants.py and the tests.

namespace {

struct R3Shared {
  PostTexelF tex[kR2Cap];
  u32 mot[3][kR2Cap];
  float col[3][kR2Cap];  // raw colour (fallback of invalid candidates)
};

WG_FN void gather_add_pair(Gather& g0, Gather& g1, const PostTexelF& t, float uu, float vv0, float uv0, float vv1,
                           float uv1) {
  const float cauu = t.ca * uu;
  const float w0 = wg_exp2(__builtin_fmaf(t.cb, vv0, __builtin_fmaf(t.cc, uv0, cauu)));
  const float w1 = wg_exp2(__builtin_fmaf(t.cb, vv1, __builtin_fmaf(t.cc, uv1, cauu)));
  Gather* gs[2] = {&g0, &g1};
  const float ws[2] = {w0, w1};
#pragma unroll
  for (int r = 0; r < 2; ++r) {
    Gather& g = *gs[r];
    const float w = ws[r];
    g.W += w;
    g.c[0] = __builtin_fmaf(w, t.ec0, g.c[0]);
    g.c[1] = __builtin_fmaf(w, t.ec1, g.c[1]);
    g.c[2] = __builtin_fmaf(w, t.ec2, g.c[2]);
    g.s[0] = __builtin_fmaf(w, t.s0, g.s[0]);
    g.a[0] = __builtin_fmaf(w, t.g, g.a[0]);
    g.s[1] = __builtin_fmaf(w, t.s1, g.s[1]);
    g.a[1] = __builtin_fmaf(w, t.cbh, g.a[1]);
    g.s[2] = __builtin_fmaf(w, t.s2, g.s[2]);
    g.a[2] = __builtin_fmaf(w, t.cbo, g.a[2]);
  }
}

template <bool kF16>
WG_FN void post_recon3_body(const v4f* color, const u32* A, const u32* B, const v4h* C, const v4h* hist_prev,
                            const u32* m_near, const u32* m_center, const u32* m_far, const float* w,
                            v4h* hist_out, u32* reduced, _Float16* display, WgPostParams p, const float* expo,
                            R3Shared& sh) {
  if (expo) {  // as post_recon
    p.e = expo[0] == 0.0f ? 1.0f : expo[0];
    p.e_prev = expo[1] == 0.0f ? 1.0f : expo[1];
    p.ratio = (float)((double)p.e / (double)p.e_prev);
  }
  const int lane = (int)wg_lane_id();
  const int tiles_x = (p.dw + kTileX - 1) / kTileX;
  const int tile = (int)wg_block_id();
  const int x0 = (tile % tiles_x) * kTileX;
  const int k0 = (tile / tiles_x) * kTilePairs;
  const int npairs = (p.dh + 1) / 2;
  const float s = w[0];

  const int xl = mini(x0 + kTileX - 1, p.dw - 1), kl = mini(k0 + kTilePairs - 1, npairs - 1);
  const int c0 = floor_i(pos_x(x0, p) - 0.5f) - 1, c1 = floor_i(pos_x(xl, p) - 0.5f) + 1;
  const int r0 = floor_i(pos_y(2 * k0, p) - 0.5f) - 1, r1 = floor_i(pos_y(2 * kl, p) - 0.5f) + 2;
  const int ncol = c1 - c0 + 1, nrow = r1 - r0 + 1;
  const int nstage = ncol * nrow < kR2Cap ? ncol * nrow : kR2Cap;  // the host checks recon2_supported
  for (int i = lane; i < nstage; i += kPostThreads) {
    const int r = i / ncol, c = i - r * ncol;
    const int idx = clampi(r0 + r, 0, p.rh - 1) * p.rw + clampi(c0 + c, 0, p.rw - 1);
    sh.tex[i] = texel_f(load_texel(color, A, B, C, idx, p.e));
    const v4f cv = *(const v4f*)((const char*)color + (u32)idx * 16u);
    sh.col[0][i] = cv[0];
    sh.col[1][i] = cv[1];
    sh.col[2][i] = cv[2];
    const u32 mo = (u32)idx * 4u;
    sh.mot[0][i] = *(const u32*)((const char*)m_near + mo);
    sh.mot[1][i] = *(const u32*)((const char*)m_center + mo);
    sh.mot[2][i] = *(const u32*)((const char*)m_far + mo);
  }
  wg_barrier();
  const auto sidx = [&](int c, int r) { return clampi((r - r0) * ncol + (c - c0), 0, kR2Cap - 1); };

  const int wl = lane & 31;
  const int col = 2 * (wl & 15) + (wl >> 4);
  const bool active = x0 + col < p.dw && k0 + (lane >> 5) < npairs;
  const int x = mini(x0 + col, p.dw - 1);
  const int y0 = 2 * mini(k0 + (lane >> 5), npairs - 1);
  const bool has1 = y0 + 1 < p.dh;
  const float px = pos_x(x, p), py0 = pos_y(y0, p), py1 = pos_y(y0 + 1, p);
  const int ax = floor_i(px - 0.5f), b = floor_i(py0 - 0.5f);

  Gather g0, g1;
  gather_zero(g0);
  gather_zero(g1);
#pragma nounroll
  for (int j = WG_AB_GATHER ? 0 : -1; j <= (WG_AB_GATHER ? 0 : 1); ++j) {
    const float v0 = ((float)(b + j) + 0.5f) - py0;
    const float v1 = ((float)(b + j) + 0.5f) - py1;
    const float vv0 = kTapK * v0 * v0, vv1 = kTapK * v1 * v1;
#pragma unroll
    for (int i = WG_AB_GATHER ? 0 : -1; i <= (WG_AB_GATHER ? 0 : 1); ++i) {
      const float u = ((float)(ax + i) + 0.5f) - px;
      const float uu = kTapK * u * u, uv0 = 2.0f * kTapK * u * v0, uv1 = 2.0f * kTapK * u * v1;
      const PostTexelF t = sh.tex[sidx(ax + i, b + j)];
      if (kF16) {
        gather_add_pair(g0, g1, t, uu, vv0, uv0, vv1, uv1);
      } else {
        gather_add_f(g0, t, uu, vv0, uv0);
        gather_add_f(g1, t, uu, vv1, uv1);
      }
    }
  }

  float hsum[3] = {0.0f, 0.0f, 0.0f};
#pragma nounroll
  for (int k = 0; k < (has1 ? 2 : 1); ++k) {
    const int y = y0 + k;
    const float py = k == 0 ? py0 : py1;
    const int pxi = floor_i(px), pyi = floor_i(py);
    float mx[3], my[3];
    if (WG_AB_MOTION) {
      mx[0] = my[0] = 0.25f;
      mx[1] = my[1] = 0.5f;
      mx[2] = my[2] = -0.75f;
    } else {
      unpack_h2(sh.mot[0][sidx(pxi, pyi)], mx[0], my[0]);
      unpack_h2(sh.mot[2][sidx(pxi, pyi)], mx[2], my[2]);
      motion_bilinear(px, py, [&](int c, int r) { return sh.mot[1][sidx(c, r)]; }, mx[1], my[1]);
    }
    const int ci = sidx(pxi, pyi);  // the clamped render pixel (floor px, floor py): fallback colour
    const float colv[3] = {sh.col[0][ci], sh.col[1][ci], sh.col[2][ci]};
    float hh[3], oh[3];
    post_core_b<kF16>(k == 0 ? g0 : g1, x, y, mx, my, colv, p, s, hist_prev, hh, oh);
    if (active) {
      if (!WG_AB_HISTW || ab_live(enc_e(hh[0]) + enc_e(hh[1]) + enc_e(hh[2])))
        hist_store<false>(hist_out, y * p.dw + x, enc_e(hh[0]), enc_e(hh[1]), enc_e(hh[2]));
      if (!WG_AB_DISPLAY || ab_live((float)display_value(oh[0]) + (float)display_value(oh[1]) + (float)display_value(oh[2]))) {
        const int pitch = display_pitch(p.dw), plane = pitch * p.dh, di = y * pitch + x;
        display[di] = display_value(oh[0]);
        display[plane + di] = display_value(oh[1]);
        display[2 * plane + di] = display_value(oh[2]);
      }
    }
#pragma unroll
    for (int c = 0; c < 3; ++c) hsum[c] += hh[c];
  }

  if (WG_AB_REDUCED) {
    if (ab_live(hsum[0] + hsum[1] + hsum[2])) reduced[0] = 0u;
    return;
  }
  float nb[3];
#pragma unroll
  for (int c = 0; c < 3; ++c) nb[c] = wg_xor16(hsum[c]);
  if (active && has1 && wl < 16 && x + 1 < p.dw) {
    u32 code = 0;
#pragma unroll
    for (int c = 0; c < 3; ++c) {
      const float m = (hsum[c] + nb[c]) * 0.25f;
      const float v = clampf(enc_e(m) * (1.0f / 2.45f), 0.0f, 1.0f);
      code |= (u32)wg_rint(v * 1023.0f) << (10 * c);
    }
    reduced[(y0 / 2) * (p.dw / 2) + x / 2] = code;
  }
}

}  // namespace

#define WG_RECON3_KERNEL(name, F16)                                                                                \
  WG_KERNEL_N(name, kPostThreads)(const v4f* color, const u32* A, const u32* B, const v4h* C,                     \
                                  const v4h* hist_prev, const u32* m_near, const u32* m_center, const u32* m_far, \
                                  const float* w, v4h* hist_out, u32* reduced, _Float16* display,                \
                                  WgPostParams p, const float* expo) {                                            \
    WG_LDS R3Shared sh;                                                                                           \
    if (p.dw < 4) return; /* the host uses post_recon below 4 columns (recon2_supported) */                       \
    post_recon3_body<F16>(color, A, B, C, hist_prev, m_near, m_center, m_far, w, hist_out, reduced, display, p,   \
                          expo, sh);                                                                              \
  }
WG_RECON3_KERNEL(post_recon3, false)
WG_RECON3_KERNEL(post_recon3_h, true)

// ---------------------------------------------------------------------------------------------------------------
// post_recon4_*: SPEED APPROXIMATIONS of post_recon (same arguments, layouts and grid; NOT spec 06-exact). Judged by
// the whole-pipeline rule of spec 09 draft 2 (>= 48 dB against the exact runtime, temporal stability and edge
// sharpness within 2 %). Built on post_recon3. Three independent switches:
//   kG     gather: 0 exact (as post_recon); 1 Gaussian a u^2 term shared by the row pair, the six per-sample codes
//          (scores, g, 1 - beta_h, 1 - beta_o) accumulated in packed FP16 (both rows in one register), history tap
//          weights in packed FP16; 2 as 1, plus the Gaussian exponent, exp, the weight sum and the colour in packed FP16.
//   kHist  history sample: 0 12-tap Catmull-Rom (spec); 1 bilinear (4 taps) for both candidates; 2 Catmull-Rom for
//          the stronger candidate, bilinear for the other.
//   kSkip  0 spec rule (second candidate dropped below weight 1/255); N > 0: also dropped below 1/N, its weight given
//          to the first (one history sample for most pixels).

namespace {

// Bilinear history sample (texel centres at i + 1/2, clamp to edge): 2 rows x 2 texels, one 16-byte load per row.
struct BiFetch {
  u32 d[2][4];
  float fx, fy;
  int ix, iy, cs;
};

WG_FN void bi_pos(int dw, float qx, float qy, BiFetch& f) {
  const float tx = qx - 0.5f, ty = qy - 0.5f;
  const float fx0 = __builtin_floorf(tx), fy0 = __builtin_floorf(ty);
  f.fx = tx - fx0;
  f.fy = ty - fy0;
  f.ix = (int)fx0;
  f.iy = (int)fy0;
  f.cs = clampi(f.ix, 0, dw - 2);
}

WG_FN void bi_load(const void* hist, int dw, int dh, BiFetch& f) {
  const char* base = (const char*)hist;
#pragma unroll
  for (int j = 0; j < 2; ++j) {
    const u32 roff = mul24((u32)clampi(f.iy + j, 0, dh - 1), (u32)dw * 8u);
    const v4u_a2 a = *(const v4u_a2*)(base + (roff + (u32)f.cs * 8u));
    f.d[j][0] = a[0]; f.d[j][1] = a[1]; f.d[j][2] = a[2]; f.d[j][3] = a[3];
  }
}

WG_FN void bi_edge(const void* hist, int dw, int dh, BiFetch& f) {
  const char* base = (const char*)hist;
#pragma unroll
  for (int j = 0; j < 2; ++j) {
    const u32 roff = mul24((u32)clampi(f.iy + j, 0, dh - 1), (u32)dw * 8u);
#pragma unroll
    for (int i = 0; i < 2; ++i) {
      const v2u_a2 v = *(const v2u_a2*)(base + (roff + (u32)clampi(f.ix + i, 0, dw - 1) * 8u));
      f.d[j][2 * i] = v[0];
      f.d[j][2 * i + 1] = v[1];
    }
  }
}

WG_FN void bi_complete(BiFetch& f) {
#if defined(__AMDGCN__)
  asm volatile("" ::"v"(f.d[0][0]), "v"(f.d[0][1]), "v"(f.d[0][2]), "v"(f.d[0][3]), "v"(f.d[1][0]), "v"(f.d[1][1]),
               "v"(f.d[1][2]), "v"(f.d[1][3]));
#else
  (void)f;
#endif
}

WG_FN void bi_finish(const BiFetch& f, float out[3]) {
  const float wx0 = 1.0f - f.fx, wy0 = 1.0f - f.fy;
#pragma unroll
  for (int c = 0; c < 3; ++c) {
    const float top = __builtin_fmaf(f.fx, half_of(f.d[0], 4 + c), wx0 * half_of(f.d[0], c));
    const float bot = __builtin_fmaf(f.fx, half_of(f.d[1], 4 + c), wx0 * half_of(f.d[1], c));
    out[c] = __builtin_fmaf(f.fy, bot, wy0 * top);
  }
}

// One history sample at (qx, qy): Catmull-Rom (kCr) or bilinear; fetch, then finish.
template <bool kCr>
struct HistSample {
  CrFetch cr;
  BiFetch bi;
  WG_FN void pos(int dw, float qx, float qy) {
    if (kCr) cr_pos(dw, qx, qy, cr);
    else bi_pos(dw, qx, qy, bi);
  }
  WG_FN void load(const void* h, int dw, int dh) {
    if (kCr) cr_spans(h, dw, dh, cr);
    else bi_load(h, dw, dh, bi);
  }
  WG_FN bool edge() const { return kCr ? cr.ix - 1 != cr.cs : bi.ix != bi.cs; }
  WG_FN void reload_edge(const void* h, int dw, int dh) {
    if (kCr) cr_edge(h, dw, dh, cr);
    else bi_edge(h, dw, dh, bi);
  }
  WG_FN void complete() {
    if (kCr) cr_complete(cr);
    else bi_complete(bi);
  }
  template <bool kF16>
  WG_FN void finish(float out[3]) const {
    if (kCr) cr_finish<kF16>(cr, out);
    else bi_finish(bi, out);
  }
};

// The raw colour of one render pixel in global memory, read on use (fallback of post_core_x).
struct GlobalCol {
  const float* c;
  WG_FN float operator[](int i) const { return c[i]; }
};

// ---- compact 4-byte full history (post_fused3*_h4; an approximation of the FP16 history) ----
// The E-encoded history as fixed point over [0, 3]: R and G 11 bits (step 3/2047 = 0.0015), B 10 bits (step 0.0029);
// FP16 has steps of 0.001 to 0.002 for E in [1, 3]. Bits 0-10 R, 11-21 G, 22-31 B. Values are clamped to [0, 3] (Dec
// clamps there anyway; E of a non-negative colour is >= 0). The samplers below weight the integer codes and scale once.
constexpr float kH4CodeRG = 2047.0f / 3.0f, kH4CodeB = 1023.0f / 3.0f;
WG_FN u32 h4_pack(float r, float g, float b) {
  const u32 cr = (u32)wg_rint(clampf(r, 0.0f, 3.0f) * kH4CodeRG);
  const u32 cg = (u32)wg_rint(clampf(g, 0.0f, 3.0f) * kH4CodeRG);
  const u32 cb = (u32)wg_rint(clampf(b, 0.0f, 3.0f) * kH4CodeB);
  return cr | (cg << 11) | (cb << 22);
}
WG_FN float h4_code(u32 v, int c) {  // integer code of channel c
  return c == 0 ? (float)(v & 0x7ffu) : c == 1 ? (float)((v >> 11) & 0x7ffu) : (float)(v >> 22);
}
WG_FN float h4_scale(int c) { return c == 2 ? 3.0f / 1023.0f : 3.0f / 2047.0f; }


// Catmull-Rom over 4-byte texels: same footprint and loads as cr_spans (rows 1, 2: 4 texels, rows 0, 3: the middle 2).
struct Cr4Fetch {
  u32 d[4][4];
  float fx, fy;
  int ix, iy, cs;
};
WG_FN void cr4_pos(int dw, float qx, float qy, Cr4Fetch& f) {
  const float tx = qx - 0.5f, ty = qy - 0.5f;
  const float fx0 = __builtin_floorf(tx), fy0 = __builtin_floorf(ty);
  f.fx = tx - fx0;
  f.fy = ty - fy0;
  f.ix = (int)fx0;
  f.iy = (int)fy0;
  f.cs = clampi(f.ix - 1, 0, dw - 4);
}
WG_FN void cr4_spans(const void* hist, int dw, int dh, Cr4Fetch& f) {
  const char* base = (const char*)hist;
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    const u32 roff = mul24((u32)clampi(f.iy - 1 + j, 0, dh - 1), (u32)dw * 4u);
    if (j == 0 || j == 3) {
      const v2u_a4 a = *(const v2u_a4*)(base + (roff + (u32)(f.cs + 1) * 4u));
      f.d[j][0] = a[0]; f.d[j][1] = a[1];
    } else {
      const v4u_a4 a = *(const v4u_a4*)(base + (roff + (u32)f.cs * 4u));
      f.d[j][0] = a[0]; f.d[j][1] = a[1]; f.d[j][2] = a[2]; f.d[j][3] = a[3];
    }
  }
}
WG_FN void cr4_edge(const void* hist, int dw, int dh, Cr4Fetch& f) {
  const char* base = (const char*)hist;
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    const u32 roff = mul24((u32)clampi(f.iy - 1 + j, 0, dh - 1), (u32)dw * 4u);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      if ((j == 0 || j == 3) && (i == 0 || i == 3)) continue;
      const int k = (j == 0 || j == 3) ? i - 1 : i;
      f.d[j][k] = *(const u32*)(base + (roff + (u32)clampi(f.ix - 1 + i, 0, dw - 1) * 4u));
    }
  }
}
WG_FN void cr4_complete(Cr4Fetch& f) {
#if defined(__AMDGCN__)
#pragma unroll
  for (int j = 0; j < 4; ++j) asm volatile("" ::"v"(f.d[j][0]), "v"(f.d[j][1]), "v"(f.d[j][2]), "v"(f.d[j][3]));
#else
  (void)f;
#endif
}
WG_FN void cr4_finish(const Cr4Fetch& f, float out[3]) {
  float wx[4], wy[4];
  cr_weights(f.fx, wx);
  cr_weights(f.fy, wy);
  float acc[3] = {0.0f, 0.0f, 0.0f};
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    const int i0 = (j == 0 || j == 3) ? 1 : 0, i1 = (j == 0 || j == 3) ? 2 : 3;
    float r[3] = {0.0f, 0.0f, 0.0f};
#pragma unroll
    for (int i = i0; i <= i1; ++i) {
      const int k = (j == 0 || j == 3) ? i - 1 : i;
#pragma unroll
      for (int c = 0; c < 3; ++c) r[c] = __builtin_fmaf(wx[i], h4_code(f.d[j][k], c), r[c]);
    }
#pragma unroll
    for (int c = 0; c < 3; ++c) acc[c] = __builtin_fmaf(wy[j], r[c], acc[c]);
  }
  const float norm = (wx[1] + wx[2]) * (wy[0] + wy[3]) + (wx[0] + wx[1] + wx[2] + wx[3]) * (wy[1] + wy[2]);
  const float inv = wg_rcp(norm);
#pragma unroll
  for (int c = 0; c < 3; ++c) out[c] = acc[c] * (inv * h4_scale(c));
}

// Bilinear over 4-byte texels (as bi_*: 2 rows x 2 texels).
struct Bi4Fetch {
  u32 d[2][2];
  float fx, fy;
  int ix, iy, cs;
};
WG_FN void bi4_pos(int dw, float qx, float qy, Bi4Fetch& f) {
  const float tx = qx - 0.5f, ty = qy - 0.5f;
  const float fx0 = __builtin_floorf(tx), fy0 = __builtin_floorf(ty);
  f.fx = tx - fx0;
  f.fy = ty - fy0;
  f.ix = (int)fx0;
  f.iy = (int)fy0;
  f.cs = clampi(f.ix, 0, dw - 2);
}
WG_FN void bi4_load(const void* hist, int dw, int dh, Bi4Fetch& f) {
  const char* base = (const char*)hist;
#pragma unroll
  for (int j = 0; j < 2; ++j) {
    const u32 roff = mul24((u32)clampi(f.iy + j, 0, dh - 1), (u32)dw * 4u);
    const v2u_a4 a = *(const v2u_a4*)(base + (roff + (u32)f.cs * 4u));
    f.d[j][0] = a[0];
    f.d[j][1] = a[1];
  }
}
WG_FN void bi4_edge(const void* hist, int dw, int dh, Bi4Fetch& f) {
  const char* base = (const char*)hist;
#pragma unroll
  for (int j = 0; j < 2; ++j) {
    const u32 roff = mul24((u32)clampi(f.iy + j, 0, dh - 1), (u32)dw * 4u);
#pragma unroll
    for (int i = 0; i < 2; ++i) f.d[j][i] = *(const u32*)(base + (roff + (u32)clampi(f.ix + i, 0, dw - 1) * 4u));
  }
}
WG_FN void bi4_complete(Bi4Fetch& f) {
#if defined(__AMDGCN__)
  asm volatile("" ::"v"(f.d[0][0]), "v"(f.d[0][1]), "v"(f.d[1][0]), "v"(f.d[1][1]));
#else
  (void)f;
#endif
}
WG_FN void bi4_finish(const Bi4Fetch& f, float out[3]) {
  const float wx0 = 1.0f - f.fx, wy0 = 1.0f - f.fy;
#pragma unroll
  for (int c = 0; c < 3; ++c) {
    const float top = __builtin_fmaf(f.fx, h4_code(f.d[0][1], c), wx0 * h4_code(f.d[0][0], c));
    const float bot = __builtin_fmaf(f.fx, h4_code(f.d[1][1], c), wx0 * h4_code(f.d[1][0], c));
    out[c] = __builtin_fmaf(f.fy, bot, wy0 * top) * h4_scale(c);
  }
}

// History sample in the history format kFmt (8: RGBA16F as post_recon, HistSample; 6: RGB16F, 4: the compact format;
// specializations below).
template <bool kCr, int kFmt>
struct HistSampleF : HistSample<kCr> {};
template <bool kCr>
struct HistSampleF<kCr, 4> {
  Cr4Fetch cr;
  Bi4Fetch bi;
  WG_FN void pos(int dw, float qx, float qy) {
    if (kCr) cr4_pos(dw, qx, qy, cr);
    else bi4_pos(dw, qx, qy, bi);
  }
  WG_FN void load(const void* h, int dw, int dh) {
    if (kCr) cr4_spans(h, dw, dh, cr);
    else bi4_load(h, dw, dh, bi);
  }
  WG_FN bool edge() const { return kCr ? cr.ix - 1 != cr.cs : bi.ix != bi.cs; }
  WG_FN void reload_edge(const void* h, int dw, int dh) {
    if (kCr) cr4_edge(h, dw, dh, cr);
    else bi4_edge(h, dw, dh, bi);
  }
  WG_FN void complete() {
    if (kCr) cr4_complete(cr);
    else bi4_complete(bi);
  }
  template <bool kF16>
  WG_FN void finish(float out[3]) const {
    if (kCr) cr4_finish(cr, out);
    else bi4_finish(bi, out);
  }
};

// ---- 6-byte full history fetches (RGB FP16 tightly packed, the format of post_fused; exact) ----
// Same footprints and arithmetic as cr_* / bi_* (8-byte texels), so the samples are bit-identical; texel k of a row
// holds the FP16 numbers 3k, 3k + 1, 3k + 2 of the row's dword array. Loads at 2-byte alignment (unaligned access).
WG_FN void set_half(u32* d, int h, u32 v) {
  d[h >> 1] = (h & 1) ? ((d[h >> 1] & 0xffffu) | (v << 16)) : ((d[h >> 1] & 0xffff0000u) | (v & 0xffffu));
}
WG_FN void load_texel6(const char* base, u32 off, u32* d, int k) {  // one texel into slot k (edges)
  const u32 lo = *(const u32_a2*)(base + off);
  const u32 hi = *(const unsigned short*)(base + off + 4u);
  set_half(d, 3 * k, lo & 0xffffu);
  set_half(d, 3 * k + 1, lo >> 16);
  set_half(d, 3 * k + 2, hi);
}

struct Cr6Fetch {
  u32 d[4][6];
  float fx, fy;
  int ix, iy, cs;
};
WG_FN void cr6_pos(int dw, float qx, float qy, Cr6Fetch& f) {
  const float tx = qx - 0.5f, ty = qy - 0.5f;
  const float fx0 = __builtin_floorf(tx), fy0 = __builtin_floorf(ty);
  f.fx = tx - fx0;
  f.fy = ty - fy0;
  f.ix = (int)fx0;
  f.iy = (int)fy0;
  f.cs = clampi(f.ix - 1, 0, dw - 4);
}
WG_FN void cr6_spans(const void* hist, int dw, int dh, Cr6Fetch& f) {
  const char* base = (const char*)hist;
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    const u32 roff = mul24((u32)clampi(f.iy - 1 + j, 0, dh - 1), (u32)dw * 6u);
    if (j == 0 || j == 3) {  // texels cs + 1, cs + 2: 12 bytes
      const u32 o = roff + (u32)(f.cs + 1) * 6u;
      const v2u_a2 a = *(const v2u_a2*)(base + o);
      f.d[j][0] = a[0];
      f.d[j][1] = a[1];
      f.d[j][2] = *(const u32_a2*)(base + (o + 8u));
    } else {                 // texels cs ... cs + 3: 24 bytes
      const u32 o = roff + (u32)f.cs * 6u;
      const v4u_a2 a = *(const v4u_a2*)(base + o);
      const v2u_a2 b = *(const v2u_a2*)(base + (o + 16u));
      f.d[j][0] = a[0]; f.d[j][1] = a[1]; f.d[j][2] = a[2]; f.d[j][3] = a[3];
      f.d[j][4] = b[0]; f.d[j][5] = b[1];
    }
  }
}
WG_FN void cr6_edge(const void* hist, int dw, int dh, Cr6Fetch& f) {
  const char* base = (const char*)hist;
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    const u32 roff = mul24((u32)clampi(f.iy - 1 + j, 0, dh - 1), (u32)dw * 6u);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      if ((j == 0 || j == 3) && (i == 0 || i == 3)) continue;
      const int k = (j == 0 || j == 3) ? i - 1 : i;
      load_texel6(base, roff + (u32)clampi(f.ix - 1 + i, 0, dw - 1) * 6u, f.d[j], k);
    }
  }
}
WG_FN void cr6_complete(Cr6Fetch& f) {
#if defined(__AMDGCN__)
#pragma unroll
  for (int j = 0; j < 4; ++j)
    asm volatile("" ::"v"(f.d[j][0]), "v"(f.d[j][1]), "v"(f.d[j][2]), "v"(f.d[j][3]), "v"(f.d[j][4]),
                 "v"(f.d[j][5]));
#else
  (void)f;
#endif
}
WG_FN void cr6_finish(const Cr6Fetch& f, float out[3]) {  // = cr_finish<false> on the same values
  float wx[4], wy[4];
  cr_weights(f.fx, wx);
  cr_weights(f.fy, wy);
  float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f;
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    const int i0 = (j == 0 || j == 3) ? 1 : 0, i1 = (j == 0 || j == 3) ? 2 : 3;
    float r0 = 0.0f, r1 = 0.0f, r2 = 0.0f;
#pragma unroll
    for (int i = i0; i <= i1; ++i) {
      const int k = (j == 0 || j == 3) ? i - 1 : i;
      r0 = __builtin_fmaf(wx[i], half_of(f.d[j], 3 * k + 0), r0);
      r1 = __builtin_fmaf(wx[i], half_of(f.d[j], 3 * k + 1), r1);
      r2 = __builtin_fmaf(wx[i], half_of(f.d[j], 3 * k + 2), r2);
    }
    acc0 = __builtin_fmaf(wy[j], r0, acc0);
    acc1 = __builtin_fmaf(wy[j], r1, acc1);
    acc2 = __builtin_fmaf(wy[j], r2, acc2);
  }
  const float norm = (wx[1] + wx[2]) * (wy[0] + wy[3]) + (wx[0] + wx[1] + wx[2] + wx[3]) * (wy[1] + wy[2]);
  const float inv = wg_rcp(norm);
  out[0] = acc0 * inv;
  out[1] = acc1 * inv;
  out[2] = acc2 * inv;
}

struct Bi6Fetch {
  u32 d[2][3];
  float fx, fy;
  int ix, iy, cs;
};
WG_FN void bi6_pos(int dw, float qx, float qy, Bi6Fetch& f) {
  const float tx = qx - 0.5f, ty = qy - 0.5f;
  const float fx0 = __builtin_floorf(tx), fy0 = __builtin_floorf(ty);
  f.fx = tx - fx0;
  f.fy = ty - fy0;
  f.ix = (int)fx0;
  f.iy = (int)fy0;
  f.cs = clampi(f.ix, 0, dw - 2);
}
WG_FN void bi6_load(const void* hist, int dw, int dh, Bi6Fetch& f) {
  const char* base = (const char*)hist;
#pragma unroll
  for (int j = 0; j < 2; ++j) {
    const u32 o = mul24((u32)clampi(f.iy + j, 0, dh - 1), (u32)dw * 6u) + (u32)f.cs * 6u;
    const v2u_a2 a = *(const v2u_a2*)(base + o);
    f.d[j][0] = a[0];
    f.d[j][1] = a[1];
    f.d[j][2] = *(const u32_a2*)(base + (o + 8u));
  }
}
WG_FN void bi6_edge(const void* hist, int dw, int dh, Bi6Fetch& f) {
  const char* base = (const char*)hist;
#pragma unroll
  for (int j = 0; j < 2; ++j) {
    const u32 roff = mul24((u32)clampi(f.iy + j, 0, dh - 1), (u32)dw * 6u);
#pragma unroll
    for (int i = 0; i < 2; ++i) load_texel6(base, roff + (u32)clampi(f.ix + i, 0, dw - 1) * 6u, f.d[j], i);
  }
}
WG_FN void bi6_complete(Bi6Fetch& f) {
#if defined(__AMDGCN__)
  asm volatile("" ::"v"(f.d[0][0]), "v"(f.d[0][1]), "v"(f.d[0][2]), "v"(f.d[1][0]), "v"(f.d[1][1]), "v"(f.d[1][2]));
#else
  (void)f;
#endif
}
WG_FN void bi6_finish(const Bi6Fetch& f, float out[3]) {  // = bi_finish on the same values
  const float wx0 = 1.0f - f.fx, wy0 = 1.0f - f.fy;
#pragma unroll
  for (int c = 0; c < 3; ++c) {
    const float top = __builtin_fmaf(f.fx, half_of(f.d[0], 3 + c), wx0 * half_of(f.d[0], c));
    const float bot = __builtin_fmaf(f.fx, half_of(f.d[1], 3 + c), wx0 * half_of(f.d[1], c));
    out[c] = __builtin_fmaf(f.fy, bot, wy0 * top);
  }
}

template <bool kCr>
struct HistSampleF<kCr, 6> {
  Cr6Fetch cr;
  Bi6Fetch bi;
  WG_FN void pos(int dw, float qx, float qy) {
    if (kCr) cr6_pos(dw, qx, qy, cr);
    else bi6_pos(dw, qx, qy, bi);
  }
  WG_FN void load(const void* h, int dw, int dh) {
    if (kCr) cr6_spans(h, dw, dh, cr);
    else bi6_load(h, dw, dh, bi);
  }
  WG_FN bool edge() const { return kCr ? cr.ix - 1 != cr.cs : bi.ix != bi.cs; }
  WG_FN void reload_edge(const void* h, int dw, int dh) {
    if (kCr) cr6_edge(h, dw, dh, cr);
    else bi6_edge(h, dw, dh, bi);
  }
  WG_FN void complete() {
    if (kCr) cr6_complete(cr);
    else bi6_complete(bi);
  }
  template <bool kF16>
  WG_FN void finish(float out[3]) const {
    if (kCr) cr6_finish(cr, out);
    else bi6_finish(bi, out);
  }
};

// SPEED APPROXIMATION of step 5 (post_recon6_eb*): the blend done in the E domain instead of in linear colour,
//   C_E = clamp(c_bar, 1e-4, 3)                        (= E(Dec(c_bar)) exactly)
//   H_E = clamp(H, 1e-4, 3)                            if e = e_prev (ratio 1: = E(Dec(H)) exactly)
//         E(Dec(H) * e / e_prev)                       otherwise (a uniform branch)
//   new history E^_h = H_E + (C_E - H_E) beta'_h,  display E^_o = H_E + (C_E - H_E) beta'_o
// (beta' as in post_blend, as 1 - (1 - alpha)(1 - beta)). hh, oh are returned E-encoded (no Dec, no E per pixel: six
// exp and six log fewer); a non-finite result falls back to C_E, as post_blend falls back to C.
WG_FN void post_blend_e(const Gather& g, const PostSel& sl, const float H[3], const WgPostParams& p, float s,
                        float hh[3], float oh[3]) {
  float Ce[3], He[3];
  bool he_inf = false;
#pragma unroll
  for (int i = 0; i < 3; ++i) {
    Ce[i] = clampf(sl.cbar[i], 1e-4f, 3.0f);
    He[i] = p.ratio == 1.0f ? clampf(H[i], 1e-4f, 3.0f) : enc_e(dec_e(H[i]) * p.ratio);
    he_inf |= __builtin_fabsf(He[i]) == __builtin_inff();
  }
  const float alpha = g.W < __builtin_inff() ? clampf(s * (g.a[0] * (1.0f / 255.0f)), 0.0f, 1.0f) : 0.0f;
  const float bh2 = 1.0f - (1.0f - alpha) * sl.cbh, bo2 = 1.0f - (1.0f - alpha) * sl.cbo;  // beta'
#pragma unroll
  for (int i = 0; i < 3; ++i) {
    const float d = Ce[i] - He[i];
    hh[i] = __builtin_fmaf(d, bh2, He[i]);
    oh[i] = __builtin_fmaf(d, bo2, He[i]);
  }
  const float sh = hh[0] + hh[1] + hh[2], so = oh[0] + oh[1] + oh[2];
  if (sh != sh || he_inf) {
#pragma unroll
    for (int i = 0; i < 3; ++i) hh[i] = Ce[i];
  }
  if (so != so || he_inf) {
#pragma unroll
    for (int i = 0; i < 3; ++i) oh[i] = Ce[i];
  }
}

// kEb: encoded-domain blend (post_blend_e; hh, oh E-encoded) instead of post_blend (hh, oh linear). kFmt: history
// format (8 or 4 bytes per pixel, HistSampleF). col: the raw colour of the fallback pixel, indexable (a float array in
// LDS, or GlobalCol: read only when a candidate falls back).
template <bool kF16, int kHist, int kSkip, bool kEb = false, int kFmt = 8, class ColT = const float*>
WG_FN void post_core_x(const Gather& g, int x, int y, const float mx[3], const float my[3], ColT col,
                       const WgPostParams& p, float s, const void* hist_prev, float hh[3], float oh[3]) {
  PostSel sl = post_select(g);
  if (kSkip > 0 && sl.w2 != 0.0f && sl.w2 < 1.0f / (float)kSkip) {
    sl.w1 = 1.0f;
    sl.w2 = 0.0f;
  }
  const float X = (float)x + 0.5f, Y = (float)y + 0.5f;
  const int ncand = sl.w2 != 0.0f ? 2 : 1;
  const float q1x = X + mx[sl.first], q1y = Y + my[sl.first];
  const float q2x = X + mx[sl.second], q2y = Y + my[sl.second];
  const auto in_range = [&](float qx, float qy) {
    return !p.reset && qx >= 0.0f && qx <= (float)p.dw && qy >= 0.0f && qy <= (float)p.dh;
  };
  const bool v1 = in_range(q1x, q1y);
  const bool v2 = ncand == 2 && in_range(q2x, q2y);
  constexpr bool kCr1 = kHist != 1, kCr2 = kHist == 0;
  const bool same = kCr1 == kCr2 && v1 && v2 && q1x == q2x && q1y == q2y;
  const bool need2 = v2 && !same;
  HistSampleF<kCr1, kFmt> s1;
  HistSampleF<kCr2, kFmt> s2;
  s1.pos(p.dw, q1x, q1y);
  s2.pos(p.dw, q2x, q2y);
  if (v1) s1.load(hist_prev, p.dw, p.dh);
  if (need2) s2.load(hist_prev, p.dw, p.dh);
  s1.complete();
  s2.complete();
  if (v1 && s1.edge()) s1.reload_edge(hist_prev, p.dw, p.dh);
  if (need2 && s2.edge()) s2.reload_edge(hist_prev, p.dw, p.dh);
  s1.complete();
  s2.complete();
  float fb[3] = {0.0f, 0.0f, 0.0f};
  if (!v1 || (ncand == 2 && !v2)) {
    fb[0] = enc_e(p.e_prev * col[0]);
    fb[1] = enc_e(p.e_prev * col[1]);
    fb[2] = enc_e(p.e_prev * col[2]);
  }
  float h1[3], h2[3], H[3];
  if (v1) s1.template finish<kF16>(h1);
  else { h1[0] = fb[0]; h1[1] = fb[1]; h1[2] = fb[2]; }
#pragma unroll
  for (int i = 0; i < 3; ++i) H[i] = sl.w1 * h1[i];
  if (ncand == 2) {
    if (same) { h2[0] = h1[0]; h2[1] = h1[1]; h2[2] = h1[2]; }
    else if (v2) s2.template finish<kF16>(h2);
    else { h2[0] = fb[0]; h2[1] = fb[1]; h2[2] = fb[2]; }
#pragma unroll
    for (int i = 0; i < 3; ++i) H[i] = H[i] + sl.w2 * h2[i];
  }
  if (kEb) post_blend_e(g, sl, H, p, s, hh, oh);
  else post_blend(g, sl, H, p, s, hh, oh);
}

// Per-sample values in packed FP16 for the approximate gathers: (a, b), (c, ec0), (ec1, ec2), codes / 255 as
// (s0, s1), (s2, g), (1 - beta_h, 1 - beta_o).
struct alignas(8) Tex16 {
  v2h ab, cec0, ec12, s01, s2g, bhbo;
};

template <int kG>  // only the texel formats the gather mode reads
struct R4Shared {
  PostTexelF tex[kG < 2 ? kR2Cap : 1];
  Tex16 t16[kG > 0 ? kR2Cap : 1];
  u32 mot[3][kR2Cap];
  float col[3][kR2Cap];
};

WG_FN Tex16 tex16_of(const PostTexelF& t) {
  Tex16 h;
  const float k = 1.0f / 255.0f;
  h.ab = v2h{(_Float16)t.ca, (_Float16)t.cb};
  h.cec0 = v2h{(_Float16)t.cc, (_Float16)t.ec0};
  h.ec12 = v2h{(_Float16)t.ec1, (_Float16)t.ec2};
  h.s01 = v2h{(_Float16)(t.s0 * k), (_Float16)(t.s1 * k)};
  h.s2g = v2h{(_Float16)(t.s2 * k), (_Float16)(t.g * k)};
  h.bhbo = v2h{(_Float16)(t.cbh * k), (_Float16)(t.cbo * k)};
  return h;
}

WG_FN v2h bc(_Float16 v) { return v2h{v, v}; }  // broadcast (op_sel_hi, free)

// Packed accumulators of the approximate gathers: element 0 = row y0, element 1 = row y0 + 1.
struct GatherH {
  v2h W, c[3], s[3], a[3];
};

WG_FN void gatherh_zero(GatherH& h) {
  const v2h z = {(_Float16)0.0f, (_Float16)0.0f};
  h.W = z;
#pragma unroll
  for (int i = 0; i < 3; ++i) h.c[i] = h.s[i] = h.a[i] = z;
}

// kG = 1: weights in FP32 (as gather_add_pair), codes in packed FP16.
WG_FN void gather_g1(Gather& g0, Gather& g1, GatherH& h, const PostTexelF& t, const Tex16& t16, float uu, float vv0,
                     float uv0, float vv1, float uv1) {
  const float cauu = t.ca * uu;
  const float w0 = WG_AB_EXP ? 1.0f : wg_exp2(__builtin_fmaf(t.cb, vv0, __builtin_fmaf(t.cc, uv0, cauu)));
  const float w1 = WG_AB_EXP ? 1.0f : wg_exp2(__builtin_fmaf(t.cb, vv1, __builtin_fmaf(t.cc, uv1, cauu)));
  g0.W += w0;
  g1.W += w1;
  g0.c[0] = __builtin_fmaf(w0, t.ec0, g0.c[0]);
  g0.c[1] = __builtin_fmaf(w0, t.ec1, g0.c[1]);
  g0.c[2] = __builtin_fmaf(w0, t.ec2, g0.c[2]);
  g1.c[0] = __builtin_fmaf(w1, t.ec0, g1.c[0]);
  g1.c[1] = __builtin_fmaf(w1, t.ec1, g1.c[1]);
  g1.c[2] = __builtin_fmaf(w1, t.ec2, g1.c[2]);
  const v2h w = {(_Float16)w0, (_Float16)w1};
  h.s[0] = __builtin_elementwise_fma(w, bc(t16.s01[0]), h.s[0]);
  h.s[1] = __builtin_elementwise_fma(w, bc(t16.s01[1]), h.s[1]);
  h.s[2] = __builtin_elementwise_fma(w, bc(t16.s2g[0]), h.s[2]);
  h.a[0] = __builtin_elementwise_fma(w, bc(t16.s2g[1]), h.a[0]);
  h.a[1] = __builtin_elementwise_fma(w, bc(t16.bhbo[0]), h.a[1]);
  h.a[2] = __builtin_elementwise_fma(w, bc(t16.bhbo[1]), h.a[2]);
}

// kG = 2: everything in packed FP16 (offsets uu, (vv0, vv1), (uv0, uv1) given in FP16).
WG_FN void gather_g2(GatherH& h, const Tex16& t, v2h uu, v2h vv, v2h uv) {
  const v2h q = __builtin_elementwise_fma(bc(t.ab[0]), uu,
                                          __builtin_elementwise_fma(bc(t.ab[1]), vv, bc(t.cec0[0]) * uv));
  const v2h w = WG_AB_EXP ? v2h{(_Float16)1.0f, (_Float16)1.0f} : __builtin_elementwise_exp2(q);
  h.W = h.W + w;
  h.c[0] = __builtin_elementwise_fma(w, bc(t.cec0[1]), h.c[0]);
  h.c[1] = __builtin_elementwise_fma(w, bc(t.ec12[0]), h.c[1]);
  h.c[2] = __builtin_elementwise_fma(w, bc(t.ec12[1]), h.c[2]);
  h.s[0] = __builtin_elementwise_fma(w, bc(t.s01[0]), h.s[0]);
  h.s[1] = __builtin_elementwise_fma(w, bc(t.s01[1]), h.s[1]);
  h.s[2] = __builtin_elementwise_fma(w, bc(t.s2g[0]), h.s[2]);
  h.a[0] = __builtin_elementwise_fma(w, bc(t.s2g[1]), h.a[0]);
  h.a[1] = __builtin_elementwise_fma(w, bc(t.bhbo[0]), h.a[1]);
  h.a[2] = __builtin_elementwise_fma(w, bc(t.bhbo[1]), h.a[2]);
}

template <int kG, int kHist, int kSkip>
WG_FN void post_recon4_body(const v4f* color, const u32* A, const u32* B, const v4h* C, const v4h* hist_prev,
                            const u32* m_near, const u32* m_center, const u32* m_far, const float* w,
                            v4h* hist_out, u32* reduced, _Float16* display, WgPostParams p, const float* expo,
                            R4Shared<kG>& sh) {
  if (expo) {
    p.e = expo[0] == 0.0f ? 1.0f : expo[0];
    p.e_prev = expo[1] == 0.0f ? 1.0f : expo[1];
    p.ratio = (float)((double)p.e / (double)p.e_prev);
  }
  const int lane = (int)wg_lane_id();
  const int tiles_x = (p.dw + kTileX - 1) / kTileX;
  const int tile = (int)wg_block_id();
  const int x0 = (tile % tiles_x) * kTileX;
  const int k0 = (tile / tiles_x) * kTilePairs;
  const int npairs = (p.dh + 1) / 2;
  const float s = w[0];

  const int xl = mini(x0 + kTileX - 1, p.dw - 1), kl = mini(k0 + kTilePairs - 1, npairs - 1);
  const int c0 = floor_i(pos_x(x0, p) - 0.5f) - 1, c1 = floor_i(pos_x(xl, p) - 0.5f) + 1;
  const int r0 = floor_i(pos_y(2 * k0, p) - 0.5f) - 1, r1 = floor_i(pos_y(2 * kl, p) - 0.5f) + 2;
  const int ncol = c1 - c0 + 1, nrow = r1 - r0 + 1;
  const int nstage = ncol * nrow < kR2Cap ? ncol * nrow : kR2Cap;
  for (int i = lane; i < nstage; i += kPostThreads) {
    const int r = i / ncol, c = i - r * ncol;
    const int idx = clampi(r0 + r, 0, p.rh - 1) * p.rw + clampi(c0 + c, 0, p.rw - 1);
    const PostTexelF tf = texel_f(load_texel(color, A, B, C, idx, p.e));
    if (kG < 2) sh.tex[i] = tf;
    if (kG > 0) sh.t16[i] = tex16_of(tf);
    const v4f cv = *(const v4f*)((const char*)color + (u32)idx * 16u);
    sh.col[0][i] = cv[0];
    sh.col[1][i] = cv[1];
    sh.col[2][i] = cv[2];
    const u32 mo = (u32)idx * 4u;
    sh.mot[0][i] = *(const u32*)((const char*)m_near + mo);
    sh.mot[1][i] = *(const u32*)((const char*)m_center + mo);
    sh.mot[2][i] = *(const u32*)((const char*)m_far + mo);
  }
  wg_barrier();
  const auto sidx = [&](int c, int r) { return clampi((r - r0) * ncol + (c - c0), 0, kR2Cap - 1); };

  const int wl = lane & 31;
  const int colx = 2 * (wl & 15) + (wl >> 4);
  const bool active = x0 + colx < p.dw && k0 + (lane >> 5) < npairs;
  const int x = mini(x0 + colx, p.dw - 1);
  const int y0 = 2 * mini(k0 + (lane >> 5), npairs - 1);
  const bool has1 = y0 + 1 < p.dh;
  const float px = pos_x(x, p), py0 = pos_y(y0, p), py1 = pos_y(y0 + 1, p);
  const int ax = floor_i(px - 0.5f), b = floor_i(py0 - 0.5f);

  Gather g0, g1;
  gather_zero(g0);
  gather_zero(g1);
  GatherH gh;
  gatherh_zero(gh);
#pragma nounroll
  for (int j = -1; j <= 1; ++j) {
    const float v0 = ((float)(b + j) + 0.5f) - py0;
    const float v1 = ((float)(b + j) + 0.5f) - py1;
    const float vv0 = kTapK * v0 * v0, vv1 = kTapK * v1 * v1;
#pragma unroll
    for (int i = -1; i <= 1; ++i) {
      const float u = ((float)(ax + i) + 0.5f) - px;
      const float uu = kTapK * u * u, uv0 = 2.0f * kTapK * u * v0, uv1 = 2.0f * kTapK * u * v1;
      const int si = sidx(ax + i, b + j);
      if (kG == 0) {
        const PostTexelF t = sh.tex[si];
        gather_add_f(g0, t, uu, vv0, uv0);
        gather_add_f(g1, t, uu, vv1, uv1);
      } else if (kG == 1) {
        gather_g1(g0, g1, gh, sh.tex[si], sh.t16[si], uu, vv0, uv0, vv1, uv1);
      } else {
        gather_g2(gh, sh.t16[si], bc((_Float16)uu), v2h{(_Float16)vv0, (_Float16)vv1},
                  v2h{(_Float16)uv0, (_Float16)uv1});
      }
    }
  }
  if (kG > 0) {  // packed sums -> the FP32 Gather the selection and blend use (codes back to the 0..255 scale)
    Gather* gs[2] = {&g0, &g1};
#pragma unroll
    for (int r = 0; r < 2; ++r) {
      Gather& g = *gs[r];
      if (kG == 2) {
        g.W = (float)gh.W[r];
#pragma unroll
        for (int i = 0; i < 3; ++i) g.c[i] = (float)gh.c[i][r];
      }
#pragma unroll
      for (int i = 0; i < 3; ++i) {
        g.s[i] = (float)gh.s[i][r] * 255.0f;
        g.a[i] = (float)gh.a[i][r] * 255.0f;
      }
    }
  }

  float hsum[3] = {0.0f, 0.0f, 0.0f};
#pragma nounroll
  for (int k = 0; k < (has1 ? 2 : 1); ++k) {
    const int y = y0 + k;
    const float py = k == 0 ? py0 : py1;
    const int pxi = floor_i(px), pyi = floor_i(py);
    float mx[3], my[3];
    unpack_h2(sh.mot[0][sidx(pxi, pyi)], mx[0], my[0]);
    unpack_h2(sh.mot[2][sidx(pxi, pyi)], mx[2], my[2]);
    motion_bilinear(px, py, [&](int c, int r) { return sh.mot[1][sidx(c, r)]; }, mx[1], my[1]);
    const int ci = sidx(pxi, pyi);
    const float colv[3] = {sh.col[0][ci], sh.col[1][ci], sh.col[2][ci]};
    float hh[3], oh[3];
    post_core_x<(kG > 0), kHist, kSkip>(k == 0 ? g0 : g1, x, y, mx, my, colv, p, s, hist_prev, hh, oh);
    if (active) {
      hist_store<false>(hist_out, y * p.dw + x, enc_e(hh[0]), enc_e(hh[1]), enc_e(hh[2]));
      const int pitch = display_pitch(p.dw), plane = pitch * p.dh, di = y * pitch + x;
      display[di] = display_value(oh[0]);
      display[plane + di] = display_value(oh[1]);
      display[2 * plane + di] = display_value(oh[2]);
    }
#pragma unroll
    for (int c = 0; c < 3; ++c) hsum[c] += hh[c];
  }

  float nb[3];
#pragma unroll
  for (int c = 0; c < 3; ++c) nb[c] = wg_xor16(hsum[c]);
  if (active && has1 && wl < 16 && x + 1 < p.dw) {
    u32 code = 0;
#pragma unroll
    for (int c = 0; c < 3; ++c) {
      const float m = (hsum[c] + nb[c]) * 0.25f;
      const float v = clampf(enc_e(m) * (1.0f / 2.45f), 0.0f, 1.0f);
      code |= (u32)wg_rint(v * 1023.0f) << (10 * c);
    }
    reduced[(y0 / 2) * (p.dw / 2) + x / 2] = code;
  }
}

}  // namespace

#define WG_RECON4_KERNEL(name, G, HIST, SKIP)                                                                      \
  WG_KERNEL_N(name, kPostThreads)(const v4f* color, const u32* A, const u32* B, const v4h* C,                     \
                                  const v4h* hist_prev, const u32* m_near, const u32* m_center, const u32* m_far, \
                                  const float* w, v4h* hist_out, u32* reduced, _Float16* display,                \
                                  WgPostParams p, const float* expo) {                                            \
    WG_LDS R4Shared<G> sh;                                                                                        \
    if (p.dw < 4) return; /* the host checks (prepare_post) */                                                    \
    post_recon4_body<G, HIST, SKIP>(color, A, B, C, hist_prev, m_near, m_center, m_far, w, hist_out, reduced,     \
                                    display, p, expo, sh);                                                        \
  }
WG_RECON4_KERNEL(post_recon4_bil, 0, 1, 0)        // bilinear history, both candidates
WG_RECON4_KERNEL(post_recon4_crbil, 0, 2, 0)      // Catmull-Rom for the stronger candidate, bilinear for the other
WG_RECON4_KERNEL(post_recon4_skip16, 0, 0, 16)    // second candidate dropped below weight 1/16
WG_RECON4_KERNEL(post_recon4_skip4, 0, 0, 4)      // second candidate dropped below weight 1/4
WG_RECON4_KERNEL(post_recon4_f16, 1, 0, 0)        // codes and history tap weights in packed FP16
WG_RECON4_KERNEL(post_recon4_f16x, 2, 0, 0)       // whole gather and history tap weights in packed FP16
WG_RECON4_KERNEL(post_recon4_fast, 1, 2, 16)      // f16 + crbil + skip16
WG_RECON4_KERNEL(post_recon4_fastest, 2, 1, 4)    // f16x + bil + skip4

// ---------------------------------------------------------------------------------------------------------------
// Pre-pass + reconstruction (post_prep, post_recon5*). post_prep runs at render resolution, one lane per render sample
// (fully coalesced), and writes everything the reconstruction needs that depends only on the sample as a 32-byte record:
//   dwords 0-2  E_in(e * colour) R, G, B   (f32: exact)
//   dword  3    (a, b)                     (f16: C is an FP16 input, exact)
//   dword  4    (c, sigma_near code)       (f16; codes 0..255 are exact in FP16)
//   dword  5    (sigma_center, sigma_far)
//   dword  6    (g, 255 - beta_h)
//   dword  7    (255 - beta_o, 0)
// post_recon5 then stages these records (expanded to FP32 features, no logs, no decoding), the motions and the raw
// colour (fallback); everything else is post_recon3, so post_recon5 is bit-identical to post_recon. post_recon5_bil:
// bilinear history (approximation, as post_recon4_bil).

namespace {

struct alignas(16) PrepRec {
  float ec0, ec1, ec2;
  v2h ab, cs0, s12, gbh, bo;
};
static_assert(sizeof(PrepRec) == 32, "PrepRec is 32 bytes");

// The records are expanded to FP32 features (PostTexelF) when staged: the FP32 gather (gather_add_f) dual-issues
// its FMAs (v_dual_fmac), FP16 operands through v_fma_mix would not (measured in the ISA: 113 instead of 95
// instructions per tap row).
WG_FN PostTexelF texel_of_rec(const PrepRec& r) {
  PostTexelF t;
  t.ec0 = r.ec0;
  t.ec1 = r.ec1;
  t.ec2 = r.ec2;
  t.ca = (float)r.ab[0];
  t.cb = (float)r.ab[1];
  t.cc = (float)r.cs0[0];
  t.s0 = (float)r.cs0[1];
  t.s1 = (float)r.s12[0];
  t.s2 = (float)r.s12[1];
  t.g = (float)r.gbh[0];
  t.cbh = (float)r.gbh[1];
  t.cbo = (float)r.bo[0];
  return t;
}

struct R5Shared {
  PostTexelF tex[kR2Cap];
  u32 mot[3][kR2Cap];
  float col[3][kR2Cap];
};

template <int kHist>
WG_FN void post_recon5_body(const PrepRec* prep, const v4f* color, const v4h* hist_prev, const u32* m_near,
                            const u32* m_center, const u32* m_far, const float* w, v4h* hist_out, u32* reduced,
                            _Float16* display, WgPostParams p, const float* expo, R5Shared& sh) {
  if (expo) {
    p.e = expo[0] == 0.0f ? 1.0f : expo[0];
    p.e_prev = expo[1] == 0.0f ? 1.0f : expo[1];
    p.ratio = (float)((double)p.e / (double)p.e_prev);
  }
  const int lane = (int)wg_lane_id();
  const int tiles_x = (p.dw + kTileX - 1) / kTileX;
  const int tile = (int)wg_block_id();
  const int x0 = (tile % tiles_x) * kTileX;
  const int k0 = (tile / tiles_x) * kTilePairs;
  const int npairs = (p.dh + 1) / 2;
  const float s = w[0];

  const int xl = mini(x0 + kTileX - 1, p.dw - 1), kl = mini(k0 + kTilePairs - 1, npairs - 1);
  const int c0 = floor_i(pos_x(x0, p) - 0.5f) - 1, c1 = floor_i(pos_x(xl, p) - 0.5f) + 1;
  const int r0 = floor_i(pos_y(2 * k0, p) - 0.5f) - 1, r1 = floor_i(pos_y(2 * kl, p) - 0.5f) + 2;
  const int ncol = c1 - c0 + 1, nrow = r1 - r0 + 1;
  const int nstage = ncol * nrow < kR2Cap ? ncol * nrow : kR2Cap;
  for (int i = lane; i < nstage; i += kPostThreads) {  // plain copies
    const int r = i / ncol, c = i - r * ncol;
    const u32 idx = (u32)(clampi(r0 + r, 0, p.rh - 1) * p.rw + clampi(c0 + c, 0, p.rw - 1));
    sh.tex[i] = WG_AB_STAGE ? texel_f(load_texel(color, nullptr, nullptr, nullptr, (int)idx, p.e))
                            : texel_of_rec(*(const PrepRec*)((const char*)prep + idx * 32u));
    const v4f cv = *(const v4f*)((const char*)color + idx * 16u);
    sh.col[0][i] = cv[0];
    sh.col[1][i] = cv[1];
    sh.col[2][i] = cv[2];
    sh.mot[0][i] = *(const u32*)((const char*)m_near + idx * 4u);
    sh.mot[1][i] = *(const u32*)((const char*)m_center + idx * 4u);
    sh.mot[2][i] = *(const u32*)((const char*)m_far + idx * 4u);
  }
  wg_barrier();
  const auto sidx = [&](int c, int r) { return clampi((r - r0) * ncol + (c - c0), 0, kR2Cap - 1); };

  const int wl = lane & 31;
  const int colx = 2 * (wl & 15) + (wl >> 4);
  const bool active = x0 + colx < p.dw && k0 + (lane >> 5) < npairs;
  const int x = mini(x0 + colx, p.dw - 1);
  const int y0 = 2 * mini(k0 + (lane >> 5), npairs - 1);
  const bool has1 = y0 + 1 < p.dh;
  const float px = pos_x(x, p), py0 = pos_y(y0, p), py1 = pos_y(y0 + 1, p);
  const int ax = floor_i(px - 0.5f), b = floor_i(py0 - 0.5f);

  Gather g0, g1;
  gather_zero(g0);
  gather_zero(g1);
#pragma nounroll
  for (int j = WG_AB_GATHER ? 0 : -1; j <= (WG_AB_GATHER ? 0 : 1); ++j) {
    const float v0 = ((float)(b + j) + 0.5f) - py0;
    const float v1 = ((float)(b + j) + 0.5f) - py1;
    const float vv0 = kTapK * v0 * v0, vv1 = kTapK * v1 * v1;
#pragma unroll
    for (int i = WG_AB_GATHER ? 0 : -1; i <= (WG_AB_GATHER ? 0 : 1); ++i) {
      const float u = ((float)(ax + i) + 0.5f) - px;
      const float uu = kTapK * u * u, uv0 = 2.0f * kTapK * u * v0, uv1 = 2.0f * kTapK * u * v1;
      const PostTexelF t = sh.tex[sidx(ax + i, b + j)];
      gather_add_f(g0, t, uu, vv0, uv0);
      gather_add_f(g1, t, uu, vv1, uv1);
    }
  }

  float hsum[3] = {0.0f, 0.0f, 0.0f};
#pragma nounroll
  for (int k = 0; k < (has1 ? 2 : 1); ++k) {
    const int y = y0 + k;
    const float py = k == 0 ? py0 : py1;
    const int pxi = floor_i(px), pyi = floor_i(py);
    float mx[3], my[3];
    if (WG_AB_MOTION) {
      mx[0] = my[0] = 0.25f;
      mx[1] = my[1] = 0.5f;
      mx[2] = my[2] = -0.75f;
    } else {
      unpack_h2(sh.mot[0][sidx(pxi, pyi)], mx[0], my[0]);
      unpack_h2(sh.mot[2][sidx(pxi, pyi)], mx[2], my[2]);
      motion_bilinear(px, py, [&](int c, int r) { return sh.mot[1][sidx(c, r)]; }, mx[1], my[1]);
    }
    const int ci = sidx(pxi, pyi);
    const float colv[3] = {sh.col[0][ci], sh.col[1][ci], sh.col[2][ci]};
    float hh[3], oh[3];
    post_core_x<false, kHist, 0>(k == 0 ? g0 : g1, x, y, mx, my, colv, p, s, hist_prev, hh, oh);
    if (active) {
      if (!WG_AB_HISTW || ab_live(enc_e(hh[0]) + enc_e(hh[1]) + enc_e(hh[2])))
        hist_store<false>(hist_out, y * p.dw + x, enc_e(hh[0]), enc_e(hh[1]), enc_e(hh[2]));
      if (!WG_AB_DISPLAY ||
          ab_live((float)display_value(oh[0]) + (float)display_value(oh[1]) + (float)display_value(oh[2]))) {
        const int pitch = display_pitch(p.dw), plane = pitch * p.dh, di = y * pitch + x;
        display[di] = display_value(oh[0]);
        display[plane + di] = display_value(oh[1]);
        display[2 * plane + di] = display_value(oh[2]);
      }
    }
#pragma unroll
    for (int c = 0; c < 3; ++c) hsum[c] += hh[c];
  }

  if (WG_AB_REDUCED) {
    if (ab_live(hsum[0] + hsum[1] + hsum[2])) reduced[0] = 0u;
    return;
  }
  float nb[3];
#pragma unroll
  for (int c = 0; c < 3; ++c) nb[c] = wg_xor16(hsum[c]);
  if (active && has1 && wl < 16 && x + 1 < p.dw) {
    u32 code = 0;
#pragma unroll
    for (int c = 0; c < 3; ++c) {
      const float m = (hsum[c] + nb[c]) * 0.25f;
      const float v = clampf(enc_e(m) * (1.0f / 2.45f), 0.0f, 1.0f);
      code |= (u32)wg_rint(v * 1023.0f) << (10 * c);
    }
    reduced[(y0 / 2) * (p.dw / 2) + x / 2] = code;
  }
}

}  // namespace

// Pre-pass: one lane per render sample, grid ceil(rw * rh / 256) workgroups of 256 lanes; prep: rw * rh records.
WG_KERNEL_N(post_prep, kPostThreads)(const v4f* color, const u32* A, const u32* B, const v4h* C, PrepRec* prep,
                                     WgPostParams p, const float* expo) {
  if (expo) p.e = expo[0] == 0.0f ? 1.0f : expo[0];
  const int i = (int)(wg_block_id() * kPostThreads + wg_lane_id());
  if (i >= p.rw * p.rh) return;
  const PostTexel t = load_texel(color, A, B, C, i, p.e);
  const u32 ac = t.a ^ 0xffff00u;  // bytes 1, 2 -> 255 - code
  PrepRec r;
  r.ec0 = t.ec0;
  r.ec1 = t.ec1;
  r.ec2 = t.ec2;
  r.ab = v2h{(_Float16)t.ca, (_Float16)t.cb};
  r.cs0 = v2h{(_Float16)t.cc, (_Float16)byte_f(t.b, 0)};
  r.s12 = v2h{(_Float16)byte_f(t.b, 1), (_Float16)byte_f(t.b, 2)};
  r.gbh = v2h{(_Float16)byte_f(ac, 0), (_Float16)byte_f(ac, 1)};
  r.bo = v2h{(_Float16)byte_f(ac, 2), (_Float16)0.0f};
  prep[i] = r;
}

// Reconstruction from the pre-pass records. Arguments as post_recon, with the records in front and without A, B, C;
// grid and launch as post_recon (needs recon2_supported and dw >= 4).
#define WG_RECON5_KERNEL(name, HIST)                                                                               \
  WG_KERNEL_N(name, kPostThreads)(const PrepRec* prep, const v4f* color, const v4h* hist_prev, const u32* m_near, \
                                  const u32* m_center, const u32* m_far, const float* w, v4h* hist_out,          \
                                  u32* reduced, _Float16* display, WgPostParams p, const float* expo) {           \
    WG_LDS R5Shared sh;                                                                                           \
    if (p.dw < 4) return;                                                                                         \
    post_recon5_body<HIST>(prep, color, hist_prev, m_near, m_center, m_far, w, hist_out, reduced, display, p,     \
                           expo, sh);                                                                             \
  }
WG_RECON5_KERNEL(post_recon5, 0)
WG_RECON5_KERNEL(post_recon5_bil, 1)

// ---------------------------------------------------------------------------------------------------------------
// post_recon6*: post_recon5 (same pre-pass, arguments, grid) with a cheaper per-pixel head and addressing:
//  - motions staged in LDS as FP32 pairs (converted once per staged sample instead of at every read);
//  - the motion reads and the m_center bilinear of both pixels of a lane (same D column, rows y0 and y0 + 1) done
//    before the pixel loop, sharing the column position and the x-interpolated motion rows (3 rows for the two pixels
//    instead of 2 + 2);
//  - LDS indices, history row offsets and output indices with 24-bit multiplies (full rate);
//  - the outputs stored through 32-bit offsets from the base pointers;
//  - the candidate choice by selects on the motions (post_core_6), the validity tests without branches;
//  - no clamps on the LDS indices (needs R < D per axis: recon6_supported), so neighbouring taps are immediate offsets.
// post_recon6 is bit-identical to post_recon. Switches, SPEED APPROXIMATIONS for the whole-pipeline rule: _bil
// bilinear history (as post_recon4_bil), _eb blend in the E domain (post_blend_e; the reduced history stays E of the
// linear mean, from Dec of the new E-encoded history), _eb_bil both.

namespace {

struct alignas(8) Mot2 {
  float x, y;
};

struct R6Shared {
  PostTexelF tex[kR2Cap];
  Mot2 mot[3][kR2Cap];
  float col[3][kR2Cap];
};

// post_core_x for post_recon6: the candidate choice of post_select done by selects on the candidates' motions and
// scores instead of through candidate indices (the same comparisons in the same order: same choice, ties and NaN
// behaviour), the validity tests without short-circuit branches. Bit-identical to post_core_x<false, kHist, 0, kEb>.
template <int kHist, bool kEb>
WG_FN void post_core_6(const Gather& g, int x, int y, const float mx[3], const float my[3], const float col[3],
                       const WgPostParams& p, float s, const void* hist_prev, float hh[3], float oh[3]) {
  PostSel sl;
  const float invW = wg_rcp(g.W);
  const float inv255 = invW * (1.0f / 255.0f);
  float sig[3];
#pragma unroll
  for (int i = 0; i < 3; ++i) {
    sl.cbar[i] = g.c[i] * invW;
    sig[i] = g.s[i] * inv255;
  }
  sl.cbh = g.a[1] * inv255;
  sl.cbo = g.a[2] * inv255;
  sl.first = 0;
  sl.second = 1;
  float m1x = mx[0], m1y = my[0], m2x = mx[1], m2y = my[1];
  float w1 = 0.5f, w2 = 0.5f;
  if (!WG_AB_SCORE) {
    // first: best = sig0; sig1 > best -> 1; sig2 > best -> 2. second: the better of the other two (o1 < o2),
    // o2 only if strictly better.
    const bool g10 = sig[1] > sig[0];
    const float b01 = g10 ? sig[1] : sig[0];
    const bool f2 = sig[2] > b01;
    const bool f0 = !f2 && !g10;
    const float s1 = f2 ? sig[2] : b01;
    m1x = f2 ? mx[2] : (g10 ? mx[1] : mx[0]);
    m1y = f2 ? my[2] : (g10 ? my[1] : my[0]);
    const float so1 = f0 ? sig[1] : sig[0], so2 = f2 ? sig[1] : sig[2];
    const bool t = so2 > so1;
    const float s2 = t ? so2 : so1;
    m2x = t ? (f2 ? mx[1] : mx[2]) : (f0 ? mx[1] : mx[0]);
    m2y = t ? (f2 ? my[1] : my[2]) : (f0 ? my[1] : my[0]);
    const float q1 = clampf(1.1f * s1 - 0.05f, 0.0f, 1.0f);
    const float q2 = clampf(1.1f * s2 - 0.05f, 0.0f, 1.0f);
    const float rden = wg_rcp(wg_fmax(q1 + q2, 1e-10f));
    w2 = q2 * rden;
    if (w2 < 1.0f / 255.0f) {
      w2 = 0.0f;
      w1 = 1.0f;
    } else {
      w1 = q1 * rden;
    }
  }
  sl.w1 = w1;
  sl.w2 = w2;

  const float X = (float)x + 0.5f, Y = (float)y + 0.5f;
  const int ncand = w2 != 0.0f ? 2 : 1;
  const float q1x = X + m1x, q1y = Y + m1y;
  const float q2x = X + m2x, q2y = Y + m2y;
  const float dwf = (float)p.dw, dhf = (float)p.dh;
  const bool live = !p.reset;
  const bool v1 = live & (q1x >= 0.0f) & (q1x <= dwf) & (q1y >= 0.0f) & (q1y <= dhf);
  const bool v2 = (ncand == 2) & live & (q2x >= 0.0f) & (q2x <= dwf) & (q2y >= 0.0f) & (q2y <= dhf);
  constexpr bool kCr1 = kHist != 1, kCr2 = kHist == 0;
  const bool same = kCr1 == kCr2 && v1 && v2 && q1x == q2x && q1y == q2y;
  const bool need2 = v2 && !same;
  HistSample<kCr1> s1;
  HistSample<kCr2> s2;
  s1.pos(p.dw, q1x, q1y);
  s2.pos(p.dw, q2x, q2y);
  if (v1) s1.load(hist_prev, p.dw, p.dh);
  if (need2) s2.load(hist_prev, p.dw, p.dh);
  s1.complete();
  s2.complete();
  if (v1 && s1.edge()) s1.reload_edge(hist_prev, p.dw, p.dh);
  if (need2 && s2.edge()) s2.reload_edge(hist_prev, p.dw, p.dh);
  s1.complete();
  s2.complete();
  float fb[3] = {0.0f, 0.0f, 0.0f};
  if (!v1 || (ncand == 2 && !v2)) {
    fb[0] = enc_e(p.e_prev * col[0]);
    fb[1] = enc_e(p.e_prev * col[1]);
    fb[2] = enc_e(p.e_prev * col[2]);
  }
  float h1[3], h2[3], H[3];
  if (v1) s1.template finish<false>(h1);
  else { h1[0] = fb[0]; h1[1] = fb[1]; h1[2] = fb[2]; }
#pragma unroll
  for (int i = 0; i < 3; ++i) H[i] = sl.w1 * h1[i];
  if (ncand == 2) {
    if (same) { h2[0] = h1[0]; h2[1] = h1[1]; h2[2] = h1[2]; }
    else if (v2) s2.template finish<false>(h2);
    else { h2[0] = fb[0]; h2[1] = fb[1]; h2[2] = fb[2]; }
#pragma unroll
    for (int i = 0; i < 3; ++i) H[i] = H[i] + sl.w2 * h2[i];
  }
  if (kEb) post_blend_e(g, sl, H, p, s, hh, oh);
  else post_blend(g, sl, H, p, s, hh, oh);
}

template <int kHist, bool kEb>
WG_FN void post_recon6_body(const PrepRec* prep, const v4f* color, const v4h* hist_prev, const u32* m_near,
                            const u32* m_center, const u32* m_far, const float* w, v4h* hist_out, u32* reduced,
                            _Float16* display, WgPostParams p, const float* expo, R6Shared& sh) {
  if (expo) {
    p.e = expo[0] == 0.0f ? 1.0f : expo[0];
    p.e_prev = expo[1] == 0.0f ? 1.0f : expo[1];
    p.ratio = (float)((double)p.e / (double)p.e_prev);
  }
  const int lane = (int)wg_lane_id();
  const int tiles_x = (p.dw + kTileX - 1) / kTileX;
  const int tile = (int)wg_block_id();
  const int x0 = (tile % tiles_x) * kTileX;
  const int k0 = (tile / tiles_x) * kTilePairs;
  const int npairs = (p.dh + 1) / 2;
  const float s = w[0];

  const int xl = mini(x0 + kTileX - 1, p.dw - 1), kl = mini(k0 + kTilePairs - 1, npairs - 1);
  const int c0 = floor_i(pos_x(x0, p) - 0.5f) - 1, c1 = floor_i(pos_x(xl, p) - 0.5f) + 1;
  const int r0 = floor_i(pos_y(2 * k0, p) - 0.5f) - 1, r1 = floor_i(pos_y(2 * kl, p) - 0.5f) + 2;
  const int ncol = c1 - c0 + 1, nrow = r1 - r0 + 1;
  const int nstage = ncol * nrow < kR2Cap ? ncol * nrow : kR2Cap;
  for (int i = lane; i < nstage; i += kPostThreads) {  // plain copies
    const int r = i / ncol, c = i - r * ncol;
    const u32 idx = mul24((u32)clampi(r0 + r, 0, p.rh - 1), (u32)p.rw) + (u32)clampi(c0 + c, 0, p.rw - 1);
    sh.tex[i] = WG_AB_STAGE ? texel_f(load_texel(color, nullptr, nullptr, nullptr, (int)idx, p.e))
                            : texel_of_rec(*(const PrepRec*)((const char*)prep + idx * 32u));
    const v4f cv = *(const v4f*)((const char*)color + idx * 16u);
    sh.col[0][i] = cv[0];
    sh.col[1][i] = cv[1];
    sh.col[2][i] = cv[2];
    unpack_h2(*(const u32*)((const char*)m_near + idx * 4u), sh.mot[0][i].x, sh.mot[0][i].y);
    unpack_h2(*(const u32*)((const char*)m_center + idx * 4u), sh.mot[1][i].x, sh.mot[1][i].y);
    unpack_h2(*(const u32*)((const char*)m_far + idx * 4u), sh.mot[2][i].x, sh.mot[2][i].y);
  }
  wg_barrier();
  // No clamp: for R < D per axis (checked by the host) every index below lies inside the staged block (gather taps,
  // the motion reads of both rows, the bilinear rows ra ... rb + 1), so neighbours are immediate LDS offsets.
  const auto sidx = [&](int c, int r) { return mul_i24(r - r0, ncol) + (c - c0); };

  const int wl = lane & 31;
  const int colx = 2 * (wl & 15) + (wl >> 4);
  const bool active = x0 + colx < p.dw && k0 + (lane >> 5) < npairs;
  const int x = mini(x0 + colx, p.dw - 1);
  const int y0 = 2 * mini(k0 + (lane >> 5), npairs - 1);
  const bool has1 = y0 + 1 < p.dh;
  const float px = pos_x(x, p), py0 = pos_y(y0, p), py1 = pos_y(y0 + 1, p);
  const int ax = floor_i(px - 0.5f), b = floor_i(py0 - 0.5f);

  Gather g0, g1;
  gather_zero(g0);
  gather_zero(g1);
#pragma nounroll
  for (int j = WG_AB_GATHER ? 0 : -1; j <= (WG_AB_GATHER ? 0 : 1); ++j) {
    const float v0 = ((float)(b + j) + 0.5f) - py0;
    const float v1 = ((float)(b + j) + 0.5f) - py1;
    const float vv0 = kTapK * v0 * v0, vv1 = kTapK * v1 * v1;
#pragma unroll
    for (int i = WG_AB_GATHER ? 0 : -1; i <= (WG_AB_GATHER ? 0 : 1); ++i) {
      const float u = ((float)(ax + i) + 0.5f) - px;
      const float uu = kTapK * u * u, uv0 = 2.0f * kTapK * u * v0, uv1 = 2.0f * kTapK * u * v1;
      const PostTexelF t = sh.tex[sidx(ax + i, b + j)];
      gather_add_f(g0, t, uu, vv0, uv0);
      gather_add_f(g1, t, uu, vv1, uv1);
    }
  }

  // Motion candidates of both pixels (near, center, far): near and far at the render pixel (floor px, floor py),
  // center bilinear (as motion_bilinear: same column, same x-interpolated rows, same arithmetic).
  const int pxi = floor_i(px), pyi0 = floor_i(py0), pyi1 = floor_i(py1);
  const int ci0 = sidx(pxi, pyi0), ci1 = sidx(pxi, pyi1);
  float mx0[3], my0[3], mx1[3], my1[3];
  if (WG_AB_MOTION) {
    mx0[0] = my0[0] = mx1[0] = my1[0] = 0.25f;
    mx0[1] = my0[1] = mx1[1] = my1[1] = 0.5f;
    mx0[2] = my0[2] = mx1[2] = my1[2] = -0.75f;
  } else {
    mx0[0] = sh.mot[0][ci0].x;
    my0[0] = sh.mot[0][ci0].y;
    mx0[2] = sh.mot[2][ci0].x;
    my0[2] = sh.mot[2][ci0].y;
    mx1[0] = sh.mot[0][ci1].x;
    my1[0] = sh.mot[0][ci1].y;
    mx1[2] = sh.mot[2][ci1].x;
    my1[2] = sh.mot[2][ci1].y;
    const float tx = px - 0.5f, ty0 = py0 - 0.5f, ty1 = py1 - 0.5f;
    const float x0f = __builtin_floorf(tx), ya = __builtin_floorf(ty0), yb = __builtin_floorf(ty1);
    const float wx = tx - x0f, wy0 = ty0 - ya, wy1 = ty1 - yb;
    const int cc = (int)x0f, ra = (int)ya, rb = (int)yb;
    const auto mrow = [&](int r, float& rx, float& ry) {
      const Mot2 a = sh.mot[1][sidx(cc, r)], bq = sh.mot[1][sidx(cc + 1, r)];
      rx = a.x * (1.0f - wx) + bq.x * wx;
      ry = a.y * (1.0f - wx) + bq.y * wx;
    };
    float tax, tay, bax, bay, tbx, tby, bbx, bby;
    mrow(ra, tax, tay);
    mrow(ra + 1, bax, bay);
    mrow(rb + 1, bbx, bby);  // pixel 1's bottom row (unused if rb = ra)
    tbx = rb == ra ? tax : bax;  // rb is ra or ra + 1 (ry < 1)
    tby = rb == ra ? tay : bay;
    if (rb == ra) {
      bbx = bax;
      bby = bay;
    }
    mx0[1] = tax * (1.0f - wy0) + bax * wy0;
    my0[1] = tay * (1.0f - wy0) + bay * wy0;
    mx1[1] = tbx * (1.0f - wy1) + bbx * wy1;
    my1[1] = tby * (1.0f - wy1) + bby * wy1;
  }

  const int pitch = display_pitch(p.dw), plane = pitch * p.dh;
  float hsum[3] = {0.0f, 0.0f, 0.0f};
#pragma nounroll
  for (int k = 0; k < (has1 ? 2 : 1); ++k) {
    const int y = y0 + k;
    float mx[3], my[3];
#pragma unroll
    for (int c = 0; c < 3; ++c) {
      mx[c] = k == 0 ? mx0[c] : mx1[c];
      my[c] = k == 0 ? my0[c] : my1[c];
    }
    const int ci = k == 0 ? ci0 : ci1;
    const float colv[3] = {sh.col[0][ci], sh.col[1][ci], sh.col[2][ci]};
    float hh[3], oh[3];
    post_core_6<kHist, kEb>(k == 0 ? g0 : g1, x, y, mx, my, colv, p, s, hist_prev, hh, oh);
    float he[3];
    _Float16 dv[3];
#pragma unroll
    for (int c = 0; c < 3; ++c) {
      he[c] = kEb ? hh[c] : enc_e(hh[c]);
      dv[c] = kEb ? (_Float16)clampf(oh[c], 0.0f, kDisplayMax) : display_value(oh[c]);
    }
    if (active) {
      if (!WG_AB_HISTW || ab_live(he[0] + he[1] + he[2])) {
        v4h t;
        t[0] = (_Float16)he[0];
        t[1] = (_Float16)he[1];
        t[2] = (_Float16)he[2];
        t[3] = (_Float16)0.0f;
        *(v4h*)((char*)hist_out + (mul24((u32)y, (u32)p.dw) + (u32)x) * 8u) = t;
      }
      if (!WG_AB_DISPLAY || ab_live((float)dv[0] + (float)dv[1] + (float)dv[2])) {
        const u32 di = mul24((u32)y, (u32)pitch) + (u32)x;
        *(_Float16*)((char*)display + di * 2u) = dv[0];
        *(_Float16*)((char*)display + (di + (u32)plane) * 2u) = dv[1];
        *(_Float16*)((char*)display + (di + 2u * (u32)plane) * 2u) = dv[2];
      }
    }
#pragma unroll
    for (int c = 0; c < 3; ++c) hsum[c] += kEb ? dec_e(hh[c]) : hh[c];  // linear (the reduced history is exact E(mean))
  }

  if (WG_AB_REDUCED) {
    if (ab_live(hsum[0] + hsum[1] + hsum[2])) reduced[0] = 0u;
    return;
  }
  float nb[3];
#pragma unroll
  for (int c = 0; c < 3; ++c) nb[c] = wg_xor16(hsum[c]);
  if (active && has1 && wl < 16 && x + 1 < p.dw) {
    u32 code = 0;
#pragma unroll
    for (int c = 0; c < 3; ++c) {
      const float m = (hsum[c] + nb[c]) * 0.25f;
      const float v = clampf(enc_e(m) * (1.0f / 2.45f), 0.0f, 1.0f);
      code |= (u32)wg_rint(v * 1023.0f) << (10 * c);
    }
    *(u32*)((char*)reduced + (mul24((u32)(y0 / 2), (u32)(p.dw / 2)) + (u32)(x / 2)) * 4u) = code;
  }
}

}  // namespace

// Reconstruction from the pre-pass records (post_prep); arguments, grid and launch as post_recon5.
#define WG_RECON6_KERNEL(name, HIST, EB)                                                                           \
  WG_KERNEL_N(name, kPostThreads)(const PrepRec* prep, const v4f* color, const v4h* hist_prev, const u32* m_near, \
                                  const u32* m_center, const u32* m_far, const float* w, v4h* hist_out,          \
                                  u32* reduced, _Float16* display, WgPostParams p, const float* expo) {           \
    WG_LDS R6Shared sh;                                                                                           \
    if (p.dw < 4) return;                                                                                         \
    post_recon6_body<HIST, EB>(prep, color, hist_prev, m_near, m_center, m_far, w, hist_out, reduced, display, p, \
                               expo, sh);                                                                         \
  }
WG_RECON6_KERNEL(post_recon6, 0, false)
WG_RECON6_KERNEL(post_recon6_bil, 1, false)
WG_RECON6_KERNEL(post_recon6_eb, 0, true)
WG_RECON6_KERNEL(post_recon6_eb_bil, 1, true)

// ---------------------------------------------------------------------------------------------------------------
// Memory-traffic floor (timing only, runner/post_traffic.py). One lane per D pixel moves exactly the bytes of one D
// pixel of the reconstruction with the given image formats (and every lane below rw * rh the bytes of one render
// sample: colour, A, B, C, three motions), with no arithmetic beyond folding the loaded bits into the stored ones,
// fully coalesced. Formats: history in / out kHin / kHout bytes per pixel (8: RGBA16F, 6: RGB16F, 4: one 32-bit word,
// e.g. 10:11:11 codes), the display planes (3 x FP16) if kDisp, the reduced history (one word per 2 x 2 block), and
// kOut bytes per output pixel (12: RGB f32, 8: RGBA16F; 0: none) for the lanes below ow * oh. The filter-side kernel
// reads the display planes and writes the output. The display pitch is taken as dw (true for dw a multiple of 64).
// The outputs are garbage. Arguments: buffers, then rw, rh, dw, dh, ow, oh.

namespace {

WG_FN u32 ld32(const void* base, u32 off) { return *(const u32_a2*)((const char*)base + off); }
WG_FN u32 ld16(const void* base, u32 off) { return *(const unsigned short*)((const char*)base + off); }
WG_FN void st32(void* base, u32 off, u32 v) { *(u32_a2*)((char*)base + off) = v; }
WG_FN void st16(void* base, u32 off, u32 v) { *(unsigned short*)((char*)base + off) = (unsigned short)v; }

template <int kBytes>
WG_FN u32 traffic_load(const void* base, u32 i) {
  if (kBytes == 8) {
    const v2u_a2 v = *(const v2u_a2*)((const char*)base + i * 8u);
    return v[0] ^ v[1];
  }
  if (kBytes == 6) return ld32(base, i * 6u) ^ ld16(base, i * 6u + 4u);
  return ld32(base, i * 4u);
}

template <int kBytes>
WG_FN void traffic_store(void* base, u32 i, u32 v) {
  if (kBytes == 12) {
    st32(base, i * 12u, v);
    st32(base, i * 12u + 4u, v + 1u);
    st32(base, i * 12u + 8u, v + 2u);
  } else if (kBytes == 8) {
    v2u_a2 t;
    t[0] = v;
    t[1] = v + 1u;
    *(v2u_a2*)((char*)base + i * 8u) = t;
  } else if (kBytes == 6) {
    st32(base, i * 6u, v);
    st16(base, i * 6u + 4u, v >> 7);
  } else if (kBytes == 4) {
    st32(base, i * 4u, v);
  }
}

template <int kHin, int kHout, bool kDisp, int kOut>
WG_FN void traffic_recon(const v4f* color, const u32* A, const u32* B, const v4h* C, const u32* m_near,
                         const u32* m_center, const u32* m_far, const void* hist_prev, void* hist_out, u32* reduced,
                         _Float16* display, void* out, int rw, int rh, int dw, int dh, int ow, int oh) {
  const u32 i = wg_block_id() * (u32)kPostThreads + wg_lane_id();
  const u32 n = (u32)dw * (u32)dh;
  if (i >= n) return;
  u32 acc = i;
  if (i < (u32)rw * (u32)rh) {
    const v4u c = *(const v4u*)((const char*)color + i * 16u);
    const v2u_a2 cc = *(const v2u_a2*)((const char*)C + i * 8u);
    acc ^= c[0] ^ c[1] ^ c[2] ^ c[3] ^ cc[0] ^ cc[1] ^ ld32(A, i * 4u) ^ ld32(B, i * 4u) ^ ld32(m_near, i * 4u) ^
           ld32(m_center, i * 4u) ^ ld32(m_far, i * 4u);
  }
  acc ^= traffic_load<kHin>(hist_prev, i);
  traffic_store<kHout>(hist_out, i, acc);
  if (kDisp) {
    st16(display, i * 2u, acc);
    st16(display, (i + n) * 2u, acc >> 3);
    st16(display, (i + 2u * n) * 2u, acc >> 5);
  }
  if ((i & 3u) == 0u) reduced[i >> 2] = acc;
  if (kOut > 0 && i < (u32)ow * (u32)oh) traffic_store<kOut>(out, i, acc);
}

}  // namespace

#define WG_TRAFFIC_KERNEL(name, HIN, HOUT, DISP, OUT)                                                              \
  WG_KERNEL_N(name, kPostThreads)(const v4f* color, const u32* A, const u32* B, const v4h* C, const u32* m_near,  \
                                  const u32* m_center, const u32* m_far, const void* hist_prev, void* hist_out,   \
                                  u32* reduced, _Float16* display, void* out, int rw, int rh, int dw, int dh,     \
                                  int ow, int oh) {                                                               \
    traffic_recon<HIN, HOUT, DISP, OUT>(color, A, B, C, m_near, m_center, m_far, hist_prev, hist_out, reduced,    \
                                        display, out, rw, rh, dw, dh, ow, oh);                                    \
  }
WG_TRAFFIC_KERNEL(post_traffic_r_h8, 8, 8, true, 0)     // two-kernel path, reconstruction (current formats)
WG_TRAFFIC_KERNEL(post_traffic_r_h6, 6, 6, true, 0)
WG_TRAFFIC_KERNEL(post_traffic_r_h4, 4, 4, true, 0)
WG_TRAFFIC_KERNEL(post_traffic_x_h8_o12, 8, 8, false, 12)  // fused: no display image, output written directly
WG_TRAFFIC_KERNEL(post_traffic_x_h6_o12, 6, 6, false, 12)
WG_TRAFFIC_KERNEL(post_traffic_x_h4_o12, 4, 4, false, 12)
WG_TRAFFIC_KERNEL(post_traffic_x_h6_o8, 6, 6, false, 8)
WG_TRAFFIC_KERNEL(post_traffic_x_h4_o8, 4, 4, false, 8)

// Filter side of the two-kernel path: read the display planes of every D pixel, write kOut bytes per output pixel.
#define WG_TRAFFIC_FILTER(name, OUT)                                                                               \
  WG_KERNEL_N(name, kPostThreads)(const _Float16* display, void* out, int dw, int dh, int ow, int oh) {           \
    const u32 i = wg_block_id() * (u32)kPostThreads + wg_lane_id();                                               \
    const u32 n = (u32)dw * (u32)dh;                                                                              \
    if (i >= n) return;                                                                                           \
    const u32 acc = ld16(display, i * 2u) ^ ld16(display, (i + n) * 2u) ^ ld16(display, (i + 2u * n) * 2u);      \
    if (i < (u32)ow * (u32)oh) traffic_store<OUT>(out, i, acc);                                                   \
  }
WG_TRAFFIC_FILTER(post_traffic_f_o12, 12)
WG_TRAFFIC_FILTER(post_traffic_f_o8, 8)

// Cache flush for cold-cache timing: every lane reads and rewrites one word of a scratch buffer larger than the
// last-level cache (grid ceil(n / 256)).
WG_KERNEL_N(post_traffic_flush, kPostThreads)(u32* buf, u32 n) {
  const u32 i = wg_block_id() * (u32)kPostThreads + wg_lane_id();
  if (i < n) buf[i] = buf[i] + 1u;
}

// ---------------------------------------------------------------------------------------------------------------
// Output filter (D -> O).
//
// A lane computes 4 output columns 4m ... 4m + 3 (centres 6m, 6m + 2, 6m + 3, 6m + 5; phases +1/4, -1/4, +1/4, -1/4)
// for a strip of S output row pairs. Its 10 input columns 6m - 2 ... 6m + 7 start at an even index, so away from the
// left and right image edges they are read with 5 aligned dword loads per plane (the planes have an even pitch).
// Vertically, output row pair p needs input rows 3p - 2 ... 3p + 4; the 7 horizontally filtered rows are kept in
// registers and slide by 3 rows per pair. 64-lane workgroups (32 column groups x 2 strips): no LDS, no barriers.

namespace {

constexpr int kFiltThreads = 64;
constexpr int kFiltColGroups = 32;  // per workgroup: 32 x 4 = 128 output columns
constexpr int kFiltStrips = kFiltThreads / kFiltColGroups;

// Normalized weights of taps c - 2 ... c + 2 for an even output index (phase +1/4): spec 06 draft 3, Lanczos-3
// stretched by 1.5 and truncated to 5 taps, from reference/post.py filter_weights (FP64, rounded to FP32). Odd
// indices (phase -1/4) use them mirrored.
// WG_LZ_PROFILE = 1 (build of post_w64__q5.co, runtime option output_filter=quality5): WrongGPU's own quality5
// weights instead (output_quality5.h, reference/output_quality.py weights5): the same 5 taps and operations, so every
// kernel below runs unchanged with other constants.
// WG_LZ_PROFILE = 2 (build of post_w64__orig.co, runtime option output_filter=original): spec 06 draft 4's weight set
// "original". The five weights are not in this file: they come from the model file (spec 08, group filter /
// output_taps) and reach every filtering kernel as five extra trailing arguments (LzW below). The arithmetic is the
// spec's: weights rounded to FP16 once (by the host), a <- h(fma(w_k, t_k, a)) for k = -2 ... 2 from a = +0 with one
// FP16 rounding per step (v_fma_f16), then the clamp to the range of the five taps. The same kernels, launch
// geometry and LDS layouts as the other profiles.
#if WG_LZ_PROFILE == 1
constexpr float kLz0 = wg_quality5::w[0], kLz1 = wg_quality5::w[1], kLz2 = wg_quality5::w[2],
                kLz3 = wg_quality5::w[3], kLz4 = wg_quality5::w[4];
#elif WG_LZ_PROFILE == 0
constexpr float kLz0 = -0x1.747696b6fedacp-4f, kLz1 = 0x1.ce39514e828fap-4f, kLz2 = 0x1.476e1ef0b2af5p-1f,
                kLz3 = 0x1.a305698ddeb62p-2f, kLz4 = -0x1.2149585494080p-4f;
#elif WG_LZ_PROFILE != 2
#error "WG_LZ_PROFILE: 0 (analytic), 1 (quality5) or 2 (original)"
#endif

WG_FN float min3(float a, float b, float c) { return wg_fmin(wg_fmin(a, b), c); }
WG_FN float max3(float a, float b, float c) { return wg_fmax(wg_fmax(a, b), c); }

// The output filter's weights where they are kernel arguments (WG_LZ_PROFILE = 2). In profiles 0 and 1 the weights
// are the constants above and the macros below expand to nothing (or to the plain expression), so those builds
// compile exactly the source they had before profile 2 existed (the accepted code objects' machine code).
//   WG_LZ_KARGS  the extra trailing kernel parameters     WG_LZ_KW   the LzW made from them, as a further argument
//   WG_LZ_PARAM  the weights as a further function parameter    WG_LZ_ARG  ... passed on
// Filter values between the two passes (registers, the LDS rings) are FP32 values in profiles 0 and 1. In profile 2
// every value is an FP16 value and travels in the same 32-bit slots as a "box": its FP16 bit pattern in the low half,
// zero above (moved, stored and loaded as bits; only the functions here look inside).
//   WG_LZ_LO / WG_LZ_HI  the two taps packed in one dword of the FP16 display image
//   WG_LZ_TAP            one FP16 display value as a tap
//   WG_LZ_VAL            a filter result as an FP32 number (for Dec)
#if WG_LZ_PROFILE == 2
struct LzW {
  u32 b[5];  // FP16 bit patterns (low 16 bits) of the weights of taps -2 ... +2 for an even output index
};
#define WG_LZ_KARGS , u32 lz_w0, u32 lz_w1, u32 lz_w2, u32 lz_w3, u32 lz_w4
#define WG_LZ_KW , (LzW{{lz_w0, lz_w1, lz_w2, lz_w3, lz_w4}})
#define WG_LZ_PARAM , const LzW& lw
#define WG_LZ_ARG , lw
#define WG_LZ_LO(v) lz_lo(v)
#define WG_LZ_HI(v) lz_hi(v)
#define WG_LZ_TAP(v) lz_box(v)
#define WG_LZ_VAL(v) lz_val(v)
WG_FN float lz_box(_Float16 v) {
  return __builtin_bit_cast(float, (u32)__builtin_bit_cast(unsigned short, v));
}
WG_FN _Float16 lz_unbox(float c) { return __builtin_bit_cast(_Float16, (unsigned short)__builtin_bit_cast(u32, c)); }
WG_FN float lz_lo(u32 v) { return __builtin_bit_cast(float, v & 0xffffu); }
WG_FN float lz_hi(u32 v) { return __builtin_bit_cast(float, v >> 16); }
WG_FN float lz_val(float c) { return (float)lz_unbox(c); }
WG_FN _Float16 lz_weight(const LzW& lw, int k) { return __builtin_bit_cast(_Float16, (unsigned short)lw.b[k]); }
// The weight of tap k - 2 as a box, for a lane of the even (ev) or the odd phase (mirrored): a per-lane select.
WG_FN float lz_wsel(const LzW& lw, bool ev, int k) { return __builtin_bit_cast(float, ev ? lw.b[k] : lw.b[4 - k]); }
WG_FN _Float16 min3h(_Float16 a, _Float16 b, _Float16 c) { return __builtin_fminf16(__builtin_fminf16(a, b), c); }
WG_FN _Float16 max3h(_Float16 a, _Float16 b, _Float16 c) { return __builtin_fmaxf16(__builtin_fmaxf16(a, b), c); }
// Spec 06 draft 4 step 2 on five boxed taps with the FP16 weights w0 ... w4 of taps -2 ... +2 (already mirrored for
// an odd output index): a <- h(fma(w_k, t_k, a)) from a = +0, each step one FP16 fma (one rounding), then
// min(max(a, min t), max t) (med3: the taps are finite). Returns a box.
WG_FN float lz5_original(const float* t, _Float16 w0, _Float16 w1, _Float16 w2, _Float16 w3, _Float16 w4) {
  const _Float16 t0 = lz_unbox(t[0]), t1 = lz_unbox(t[1]), t2 = lz_unbox(t[2]), t3 = lz_unbox(t[3]),
                 t4 = lz_unbox(t[4]);
  _Float16 a = __builtin_fmaf16(w0, t0, (_Float16)0.0f);
  a = __builtin_fmaf16(w1, t1, a);
  a = __builtin_fmaf16(w2, t2, a);
  a = __builtin_fmaf16(w3, t3, a);
  a = __builtin_fmaf16(w4, t4, a);
  const _Float16 mid_lo = min3h(t1, t2, t3), mid_hi = max3h(t1, t2, t3);
  return lz_box(__builtin_amdgcn_fmed3h(a, min3h(mid_lo, t0, t4), max3h(mid_hi, t0, t4)));
}
#else
#define WG_LZ_KARGS
#define WG_LZ_KW
#define WG_LZ_PARAM
#define WG_LZ_ARG
#define WG_LZ_LO(v) half_lo(v)
#define WG_LZ_HI(v) half_hi(v)
#define WG_LZ_TAP(v) (float)v
#define WG_LZ_VAL(v) v
#endif

// One filtered value from the 5 taps t[0..4] (centre t[2]) with the clamp to their range (inputs are finite, so
// med3 equals clamp(v, min, max)). kEven: phase +1/4; else phase -1/4 (mirrored weights).
template <bool kEven>
WG_FN float lz5(const float* t WG_LZ_PARAM) {
#if WG_LZ_PROFILE == 2
  return lz5_original(t, lz_weight(lw, kEven ? 0 : 4), lz_weight(lw, kEven ? 1 : 3), lz_weight(lw, 2),
                      lz_weight(lw, kEven ? 3 : 1), lz_weight(lw, kEven ? 4 : 0));
#else
  float v = (kEven ? kLz0 : kLz4) * t[0];
  v = __builtin_fmaf(kEven ? kLz1 : kLz3, t[1], v);
  v = __builtin_fmaf(kLz2, t[2], v);
  v = __builtin_fmaf(kEven ? kLz3 : kLz1, t[3], v);
  v = __builtin_fmaf(kEven ? kLz4 : kLz0, t[4], v);
  if (WG_AB_FCLAMP) return v;
  const float mid_lo = min3(t[1], t[2], t[3]), mid_hi = max3(t[1], t[2], t[3]);
  return med3(v, min3(mid_lo, t[0], t[4]), max3(mid_hi, t[0], t[4]));
#endif
}
// lz5 with the phase-dependent weights in registers (w0 w1 w3 w4 = kLz0 kLz1 kLz3 kLz4, or mirrored): the same
// operations in the same order as lz5<kEven>, so bit-identical; one code path for lanes of either phase.
// (Profile 2: w0 ... w4 are boxes from lz_wsel; the centre weight comes from lw.)
WG_FN float lz5w(const float* t, float w0, float w1, float w3, float w4 WG_LZ_PARAM) {
#if WG_LZ_PROFILE == 2
  return lz5_original(t, lz_unbox(w0), lz_unbox(w1), lz_weight(lw, 2), lz_unbox(w3), lz_unbox(w4));
#else
  float v = w0 * t[0];
  v = __builtin_fmaf(w1, t[1], v);
  v = __builtin_fmaf(kLz2, t[2], v);
  v = __builtin_fmaf(w3, t[3], v);
  v = __builtin_fmaf(w4, t[4], v);
  if (WG_AB_FCLAMP) return v;
  const float mid_lo = min3(t[1], t[2], t[3]), mid_hi = max3(t[1], t[2], t[3]);
  return med3(v, min3(mid_lo, t[0], t[4]), max3(mid_hi, t[0], t[4]));
#endif
}

WG_FN float half_lo(u32 v) { return (float)__builtin_bit_cast(_Float16, (unsigned short)(v & 0xffffu)); }
WG_FN float half_hi(u32 v) { return (float)__builtin_bit_cast(_Float16, (unsigned short)(v >> 16)); }

// Horizontal pass of input row r for output columns 4m ... 4m + 3: h[3 k + c] = column 4m + k, channel c.
WG_FN void filter_row(const _Float16* display, int pitch, int plane, int dw, int r, int m, bool fast, float h[12]
                      WG_LZ_PARAM) {
  const _Float16* row = display + r * pitch;
  const int c0 = 6 * m - 2;
#pragma unroll
  for (int c = 0; c < 3; ++c) {
    // Both paths produce the 10 values as 5 packed half pairs, so they share registers and the unpacking.
    u32 q[5];
    if (fast) {
      const u32* src = (const u32*)(row + c * plane + c0);  // c0 even, pitch and plane even: 4-byte aligned
#pragma unroll
      for (int i = 0; i < 5; ++i) q[i] = src[i];
    } else {
      const unsigned short* src = (const unsigned short*)(row + c * plane);
#pragma unroll
      for (int i = 0; i < 5; ++i)
        q[i] = (u32)src[clampi(c0 + 2 * i, 0, dw - 1)] | ((u32)src[clampi(c0 + 2 * i + 1, 0, dw - 1)] << 16);
    }
    float t[10];
#pragma unroll
    for (int i = 0; i < 5; ++i) {
      t[2 * i] = WG_LZ_LO(q[i]);
      t[2 * i + 1] = WG_LZ_HI(q[i]);
    }
    h[c] = lz5<true>(t WG_LZ_ARG);
    h[3 + c] = lz5<false>(t + 2 WG_LZ_ARG);
    h[6 + c] = lz5<true>(t + 3 WG_LZ_ARG);
    h[9 + c] = lz5<false>(t + 5 WG_LZ_ARG);
  }
}

// Half = false: out is O, 3 x f32 (row-major RGB). Half = true (interop output copy): out is O as RGBA16F rows of
// hpitch halves, each value times pi (the runtime's separate scale_rgb step folded in, same rounding), alpha 1.
template <int S, bool Half = false>
WG_FN void filter_strip(const _Float16* display, void* outv, int dw, int dh, int ow, int oh, float dec_scale
                        WG_LZ_PARAM, float pi = 1.0f, int hpitch = 0) {
  const int lane = (int)wg_lane_id();
  const int tiles_x = (ow + 4 * kFiltColGroups - 1) / (4 * kFiltColGroups);
  const int tile = (int)wg_block_id();
  const int m = (tile % tiles_x) * kFiltColGroups + (lane % kFiltColGroups);
  const int p0 = ((tile / tiles_x) * kFiltStrips + lane / kFiltColGroups) * S;
  if (4 * m >= ow || 2 * p0 >= oh) return;
  const int pitch = display_pitch(dw), plane = pitch * dh;
  const bool fast = m >= 1 && 6 * m + 7 <= dw - 1;
  const int ncols = ow - 4 * m < 4 ? ow - 4 * m : 4;

  // hr[k]: horizontal results of input row 3p - 2 + k (k = 0..6) for the current output row pair p. Each step
  // shifts the window by 3 rows and filters the 3 new rows 3p + 2 ... 3p + 4; step -1 only fills the window
  // (the row before it, 3 p0 - 2, is filtered first into hr[6] and shifted into hr[3]).
  float hr[7][12];
  filter_row(display, pitch, plane, dw, clampi(3 * p0 - 2, 0, dh - 1), m, fast, hr[6] WG_LZ_ARG);
#pragma nounroll
  for (int st = -1; st < S; ++st) {
    const int p = p0 + st;
    if (2 * p >= oh) break;
#pragma unroll
    for (int k = 0; k < 4; ++k)
#pragma unroll
      for (int i = 0; i < 12; ++i) hr[k][i] = hr[k + 3][i];
#pragma unroll
    for (int k = 4; k < 7; ++k)
      filter_row(display, pitch, plane, dw, clampi(3 * p - 2 + k, 0, dh - 1), m, fast, hr[k] WG_LZ_ARG);
    if (st < 0) continue;
#pragma unroll
    for (int dy = 0; dy < 2; ++dy) {
      const int y = 2 * p + dy;
      if (y >= oh) break;
      // Dec as in dec_m1, with the polynomial (near-zero values) evaluated only by lanes that need it.
      float v[12], qc[12];
      bool any_small = false;
#pragma unroll
      for (int i = 0; i < 12; ++i) {
        float t[5];
#pragma unroll
        for (int k = 0; k < 5; ++k) t[k] = hr[2 * dy + k][i];
        qc[i] = clampf(WG_LZ_VAL(dy == 0 ? lz5<true>(t WG_LZ_ARG) : lz5<false>(t WG_LZ_ARG)), 1e-4f, 3.0f);
        v[i] = WG_AB_FDEC ? qc[i] * 6.4f : wg_exp2(qc[i] * kDecK) - 1.0f;
        any_small |= !WG_AB_FDEC && qc[i] < kDecPolyMax;
      }
      if (any_small) {
#pragma unroll
        for (int i = 0; i < 12; ++i)
          if (qc[i] < kDecPolyMax) v[i] = dec_m1_poly(qc[i]);
      }
#pragma unroll
      for (int i = 0; i < 12; ++i) v[i] *= dec_scale;
      if constexpr (!Half) {
        float* o = (float*)outv + ((size_t)y * ow + 4 * m) * 3;
        if (ncols == 4) {
#pragma unroll
          for (int i = 0; i < 12; ++i) o[i] = v[i];
        } else {
          for (int i = 0; i < 3 * ncols; ++i) o[i] = v[i];
        }
      } else {
        _Float16* o = (_Float16*)outv + (size_t)y * hpitch + 16 * m;
        for (int j = 0; j < ncols; ++j) {
          o[4 * j] = framebuffer_half(v[3 * j] * pi);
          o[4 * j + 1] = framebuffer_half(v[3 * j + 1] * pi);
          o[4 * j + 2] = framebuffer_half(v[3 * j + 2] * pi);
          o[4 * j + 3] = (_Float16)1.0f;
        }
      }
    }
  }
}

// One channel per lane (post_filter_c1*): lane 3 g + c of a wave (g < 10, the last 2 lanes idle) computes channel c of
// output columns 4m ... 4m + 3, m = 10 (block column) + g, for a strip of S row pairs. The sliding window is 7 rows x
// 4 values (instead of 7 x 12), so far fewer registers and more waves; a wave's stores still cover its 40 output
// columns' bytes contiguously. Same arithmetic per value as post_filter (bit-identical output).
constexpr int kFc1Groups = 10;  // column groups per wave

WG_FN void filter_row_c1(const _Float16* display, int pitch, int plane, int dw, int r, int m, bool fast, int c,
                         float h[4] WG_LZ_PARAM) {
  const _Float16* row = display + r * pitch + c * plane;
  const int c0 = 6 * m - 2;
  u32 q[5];
  if (fast) {
    const u32* src = (const u32*)(row + c0);
#pragma unroll
    for (int i = 0; i < 5; ++i) q[i] = src[i];
  } else {
    const unsigned short* src = (const unsigned short*)row;
#pragma unroll
    for (int i = 0; i < 5; ++i)
      q[i] = (u32)src[clampi(c0 + 2 * i, 0, dw - 1)] | ((u32)src[clampi(c0 + 2 * i + 1, 0, dw - 1)] << 16);
  }
  float t[10];
#pragma unroll
  for (int i = 0; i < 5; ++i) {
    t[2 * i] = WG_LZ_LO(q[i]);
    t[2 * i + 1] = WG_LZ_HI(q[i]);
  }
  h[0] = lz5<true>(t WG_LZ_ARG);
  h[1] = lz5<false>(t + 2 WG_LZ_ARG);
  h[2] = lz5<true>(t + 3 WG_LZ_ARG);
  h[3] = lz5<false>(t + 5 WG_LZ_ARG);
}

// kH (interop output copy): RGBA16F rows of hpitch bytes, each value (v * dec_scale) * pi rounded to FP16 (the
// operation order of post_filter_h, so bit-identical to it; with pi = 1 also to post_filter_c1_s4_o16), alpha 1.
template <int S, bool kO16 = false, bool kH = false>
WG_FN void filter_strip_c1(const _Float16* display, void* out, int dw, int dh, int ow, int oh, float dec_scale
                           WG_LZ_PARAM, int tile0 = 0, float pi = 1.0f, int hpitch = 0) {
  const int lane = (int)wg_lane_id();
  const int wl = lane & 31;
  if (wl >= 3 * kFc1Groups) return;
  const int g = wl / 3, c = wl - 3 * g;
  const int tiles_x = (ow + 4 * kFc1Groups - 1) / (4 * kFc1Groups);
  const int tile = (int)wg_block_id() + tile0;
  const int m = (tile % tiles_x) * kFc1Groups + g;
  const int p0 = ((tile / tiles_x) * kFiltStrips + (lane >> 5)) * S;
  if (4 * m >= ow || 2 * p0 >= oh) return;
  const int pitch = display_pitch(dw), plane = pitch * dh;
  const bool fast = m >= 1 && 6 * m + 7 <= dw - 1;
  const int ncols = ow - 4 * m < 4 ? ow - 4 * m : 4;

  float hr[7][4];
  filter_row_c1(display, pitch, plane, dw, clampi(3 * p0 - 2, 0, dh - 1), m, fast, c, hr[6] WG_LZ_ARG);
#pragma nounroll
  for (int st = -1; st < S; ++st) {
    const int p = p0 + st;
    if (2 * p >= oh) break;
#pragma unroll
    for (int k = 0; k < 4; ++k)
#pragma unroll
      for (int i = 0; i < 4; ++i) hr[k][i] = hr[k + 3][i];
#pragma unroll
    for (int k = 4; k < 7; ++k)
      filter_row_c1(display, pitch, plane, dw, clampi(3 * p - 2 + k, 0, dh - 1), m, fast, c, hr[k] WG_LZ_ARG);
    if (st < 0) continue;
#pragma unroll
    for (int dy = 0; dy < 2; ++dy) {
      const int y = 2 * p + dy;
      if (y >= oh) break;
      const u32 o0 = mul24((u32)y, (u32)ow) + (u32)(4 * m);  // first output pixel of this lane's 4
      float v[4], qc[4];
      bool any_small = false;
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        float t[5];
#pragma unroll
        for (int k = 0; k < 5; ++k) t[k] = hr[2 * dy + k][i];
        qc[i] = clampf(WG_LZ_VAL(dy == 0 ? lz5<true>(t WG_LZ_ARG) : lz5<false>(t WG_LZ_ARG)), 1e-4f, 3.0f);
        v[i] = WG_AB_FDEC ? qc[i] * 6.4f : wg_exp2(qc[i] * kDecK) - 1.0f;
        any_small |= !WG_AB_FDEC && qc[i] < kDecPolyMax;
      }
      if (any_small) {
#pragma unroll
        for (int i = 0; i < 4; ++i)
          if (qc[i] < kDecPolyMax) v[i] = dec_m1_poly(qc[i]);
      }
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        if (i >= ncols) continue;
        if (kH) {  // RGBA16F rows of hpitch bytes, pi folded in; the lane of channel 2 also writes alpha = 1
          _Float16* o = (_Float16*)((char*)out + (size_t)y * (u32)hpitch + (u32)(4 * m + i) * 8u);
          const float sv = v[i] * dec_scale;
          o[c] = framebuffer_half(sv * pi);
          if (c == 2) o[3] = (_Float16)1.0f;
        } else if (kO16) {  // RGBA16F: channel c; the lane of channel 2 also writes alpha = 1
          _Float16* o = (_Float16*)((char*)out + (o0 + (u32)i) * 8u);
          o[c] = framebuffer_half(v[i] * dec_scale);
          if (c == 2) o[3] = (_Float16)1.0f;
        } else {
          ((float*)((char*)out + (o0 + (u32)i) * 12u))[c] = v[i] * dec_scale;
        }
      }
    }
  }
}

// Private output-quality profile. Four columns share 14 input texels, and row pairs share an 11-row sliding
// window. Reconstruction/history are untouched. This standalone filter reuses the display image; the private
// runtime profile disables true reconstruction/filter fusion and therefore can add a launch versus that path.
WG_FN void quality9_row(const _Float16* display, int pitch, int plane, int dw, int row_index, int m, int c,
                         bool fast, float h[4]) {
  const _Float16* row = display + (size_t)row_index * pitch + (size_t)c * plane;
  const int x0 = 6 * m - 4;
  u32 q[7];
#pragma unroll
  for (int i = 0; i < 7; ++i) {
    if (fast) q[i] = ((const u32*)(row + x0))[i];
    else {
      const unsigned short* p = (const unsigned short*)row;
      q[i] = (u32)p[clampi(x0 + 2 * i, 0, dw - 1)] |
             ((u32)p[clampi(x0 + 2 * i + 1, 0, dw - 1)] << 16);
    }
  }
  float t[14];
#pragma unroll
  for (int i = 0; i < 7; ++i) t[2 * i] = half_lo(q[i]), t[2 * i + 1] = half_hi(q[i]);
  h[0] = wg_quality9::filter<true>(t);
  h[1] = wg_quality9::filter<false>(t + 2);
  h[2] = wg_quality9::filter<true>(t + 3);
  h[3] = wg_quality9::filter<false>(t + 5);
}

template <bool Half>
WG_FN void quality9_strip(const _Float16* display, void* out, int dw, int dh, int ow, int oh,
                           float dec_scale, float pi = 1.0f, int pitch_bytes = 0) {
  constexpr int S = 4;
  const int lane = (int)wg_lane_id(), wl = lane & 31;
  if (wl >= 30) return;
  const int group = wl / 3, c = wl % 3, tiles_x = (ow + 39) / 40;
  const int tile = (int)wg_block_id(), m = (tile % tiles_x) * 10 + group;
  const int p0 = ((tile / tiles_x) * 2 + (lane >> 5)) * S;
  if (4 * m >= ow || 2 * p0 >= oh) return;
  const int pitch = display_pitch(dw), plane = pitch * dh;
  const bool fast = m >= 1 && 6 * m + 9 < dw;
  const int ncols = ow - 4 * m < 4 ? ow - 4 * m : 4;
  float h[11][4];
#pragma unroll
  for (int j = 0; j < 11; ++j)
    quality9_row(display, pitch, plane, dw, clampi(3 * p0 - 4 + j, 0, dh - 1), m, c, fast, h[j]);
#pragma nounroll
  for (int step = 0; step < S; ++step) {
    const int p = p0 + step;
    if (2 * p >= oh) break;
    if (step > 0) {
#pragma unroll
      for (int j = 0; j < 8; ++j)
#pragma unroll
        for (int i = 0; i < 4; ++i) h[j][i] = h[j + 3][i];
#pragma unroll
      for (int j = 8; j < 11; ++j)
        quality9_row(display, pitch, plane, dw, clampi(3 * p - 4 + j, 0, dh - 1), m, c, fast, h[j]);
    }
#pragma unroll
    for (int dy = 0; dy < 2; ++dy) {
      const int y = 2 * p + dy;
      if (y >= oh) break;
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        if (i >= ncols) continue;
        float taps[9];
#pragma unroll
        for (int j = 0; j < 9; ++j) taps[j] = h[2 * dy + j][i];
        const float q = dy == 0 ? wg_quality9::filter<true>(taps) : wg_quality9::filter<false>(taps);
        const float value = dec_m1(q) * dec_scale;
        const int x = 4 * m + i;
        if constexpr (Half) {
          _Float16* dst = (_Float16*)((char*)out + (size_t)y * (unsigned)pitch_bytes + (size_t)x * 8);
          dst[c] = framebuffer_half(value * pi);
          if (c == 2) dst[3] = (_Float16)1.0f;
        } else ((float*)out)[((size_t)y * ow + x) * 3 + c] = value;
      }
    }
  }
}

}  // namespace

WG_KERNEL_N(post_filter_quality9, 64)(const _Float16* display, void* out, int dw, int dh, int ow, int oh,
                                     float dec_scale, const float* expo) {
  if (expo) dec_scale = (float)(0.01 / (double)(expo[0] == 0.0f ? 1.0f : expo[0]));
  quality9_strip<false>(display, out, dw, dh, ow, oh, dec_scale);
}
WG_KERNEL_N(post_filter_quality9_o16, 64)(const _Float16* display, void* out, int dw, int dh, int ow, int oh,
                                         float dec_scale, const float* expo) {
  if (expo) dec_scale = (float)(0.01 / (double)(expo[0] == 0.0f ? 1.0f : expo[0]));
  quality9_strip<true>(display, out, dw, dh, ow, oh, dec_scale, 1.0f, 8 * ow);
}
WG_KERNEL_N(post_filter_quality9_h, 64)(const _Float16* display, void* out, int dw, int dh, int ow, int oh,
                                       float dec_scale, const float* expo, float pi, int pitch_bytes) {
  if (expo) dec_scale = (float)(0.01 / (double)(expo[0] == 0.0f ? 1.0f : expo[0]));
  quality9_strip<true>(display, out, dw, dh, ow, oh, dec_scale, pi, pitch_bytes);
}

// Stage "filter". display: three FP16 planes of D (E-encoded), row pitch display_pitch(dw); out: O, 3 x f32
// (row-major RGB). dec_scale = 0.01 / e, or from the exposure pair in GPU memory when expo is not null (runtime).
// D = floor((3 O + 1) / 2) per axis.
// WG_LZ_PROFILE = 2 (post_w64__orig.co): every kernel that filters (these, post_tf*, post_fused*) takes five more
// arguments after the ones listed for it, u32 each: the FP16 bit patterns of the weights of taps -2 ... +2 for an even
// output index (WG_LZ_KARGS).
// post_filter_s<S>: S output row pairs per lane; grid ceil(ow / 128) * ceil(ceil(oh / 2) / (2 S)) workgroups of 64
// lanes. post_filter is the default (S = 8).
#define WG_FILTER_KERNEL(name, S)                                                                                  \
  WG_KERNEL_N(name, kFiltThreads)(const _Float16* display, float* out, int dw, int dh, int ow, int oh,            \
                                  float dec_scale, const float* expo WG_LZ_KARGS) {                               \
    if (expo) dec_scale = (float)(0.01 / (double)(expo[0] == 0.0f ? 1.0f : expo[0]));                            \
    filter_strip<S>(display, out, dw, dh, ow, oh, dec_scale WG_LZ_KW);                                           \
  }
WG_FILTER_KERNEL(post_filter, 8)
WG_FILTER_KERNEL(post_filter_s4, 4)
WG_FILTER_KERNEL(post_filter_s16, 16)
// Interop output copy (option output_copy=1): post_filter writing RGBA16F rows (pitch in bytes, multiple of 8) with
// pi folded in, for a plain CopyTextureRegion into the game's RGBA16F output texture.
WG_KERNEL_N(post_filter_h, kFiltThreads)(const _Float16* display, _Float16* out, int dw, int dh, int ow, int oh,
                                         float dec_scale, const float* expo, float pi, int pitch_bytes WG_LZ_KARGS) {
  if (expo) dec_scale = (float)(0.01 / (double)(expo[0] == 0.0f ? 1.0f : expo[0]));
  filter_strip<8, true>(display, out, dw, dh, ow, oh, dec_scale WG_LZ_KW, pi, pitch_bytes / 2);
}

// One channel per lane (filter_strip_c1); grid ceil(ow / 40) * ceil(ceil(oh / 2) / (2 S)) workgroups of 64 lanes.
#define WG_FILTER_C1_KERNEL_X(name, S, O16)                                                                        \
  WG_KERNEL_N(name, kFiltThreads)(const _Float16* display, void* out, int dw, int dh, int ow, int oh,             \
                                  float dec_scale, const float* expo WG_LZ_KARGS) {                               \
    if (expo) dec_scale = (float)(0.01 / (double)(expo[0] == 0.0f ? 1.0f : expo[0]));                            \
    filter_strip_c1<S, O16>(display, out, dw, dh, ow, oh, dec_scale WG_LZ_KW);                                   \
  }
#define WG_FILTER_C1_KERNEL(name, S) WG_FILTER_C1_KERNEL_X(name, S, false)
WG_FILTER_C1_KERNEL(post_filter_c1, 8)
WG_FILTER_C1_KERNEL(post_filter_c1_s4, 4)
WG_FILTER_C1_KERNEL(post_filter_c1_s16, 16)
// _o16: output RGBA16F (R, G, B, 1; the FP16 rounding, to nearest even, of the f32 output), 8 bytes per pixel, row
// pitch 8 ow.
WG_FILTER_C1_KERNEL_X(post_filter_c1_s4_o16, 4, true)

// Banded filter launches (see post_recon2_r_band): plus tile0. _s2: 2 row pairs per lane (twice the workgroups per
// band).
#define WG_FILTER_C1_BAND_KERNEL(name, S)                                                                          \
  WG_KERNEL_N(name, kFiltThreads)(const _Float16* display, void* out, int dw, int dh, int ow, int oh,             \
                                  float dec_scale, const float* expo, int tile0 WG_LZ_KARGS) {                    \
    if (expo) dec_scale = (float)(0.01 / (double)(expo[0] == 0.0f ? 1.0f : expo[0]));                            \
    filter_strip_c1<S>(display, out, dw, dh, ow, oh, dec_scale WG_LZ_KW, tile0);                                 \
  }
WG_FILTER_C1_BAND_KERNEL(post_filter_c1_s4_band, 4)
WG_FILTER_C1_BAND_KERNEL(post_filter_c1_s2_band, 2)
// Banded filter for the interop output copy: RGBA16F rows (pitch_bytes, a multiple of 8) with pi folded in, as
// post_filter_h (bit-identical to it, and with pi = 1 and pitch 8 ow to post_filter_c1_s4_o16), plus tile0. With it the
// in-game path (output copy) runs banded like the runtime's own output: the filter reads the display image while it is
// still in the caches.
WG_KERNEL_N(post_filter_c1_s4_h_band, kFiltThreads)(const _Float16* display, void* out, int dw, int dh, int ow,
                                                    int oh, float dec_scale, const float* expo, float pi,
                                                    int pitch_bytes, int tile0 WG_LZ_KARGS) {
  if (expo) dec_scale = (float)(0.01 / (double)(expo[0] == 0.0f ? 1.0f : expo[0]));
  filter_strip_c1<4, false, true>(display, out, dw, dh, ow, oh, dec_scale WG_LZ_KW, tile0, pi, pitch_bytes);
}

// ---------------------------------------------------------------------------------------------------------------
// True fusion (post_tf*): reconstruction + output filter in ONE kernel; the display image never goes to memory.
//
// A workgroup owns a strip of 40 output columns (filter cells m = 10 s ... 10 s + 9; D columns X0 = 60 s ... X0 + 59)
// and C output row pairs P0 = C t ... P0 + C - 1 (C even, a kernel argument). The filter of those outputs reads the
// display values of D columns X0 - 2 ... X0 + 61 (64 columns: one wave64 lane each; columns outside D are clamped
// duplicates, computed and never stored) and of D rows 3 P0 - 2 ... 3 (P0 + C) + 1 (clamped to D). The workgroup
// walks down those rows in steps of NW row pairs (one wave64 per pair, NW waves):
//   1. stage the step's render footprint in LDS (R2Texel, the conversion of post_recon2_r_p12); the global loads of a
//      step's footprint are issued during the previous step (after its reconstruction) and only converted and written
//      here, so their latency is hidden;
//   2. each wave reconstructs its row pair (2k, 2k + 1) with the functions and operation order of post_recon2_body,
//      so history, reduced history and display values are bit-identical to it. Owned pixels (columns X0 ... X0 + 59,
//      rows 3 P0 ... 3 (P0 + C) - 1) write the full and the reduced history (2 x 2 partner column in lane ^ 1). The
//      wave's 2 x 64 display values (FP16, spec 06) go to its two slots of an LDS ring of filter rows, the wave runs
//      the output filter's horizontal pass on them (filter_row_c1 arithmetic: lane = row, channel, cell of 4 output
//      columns; a wave-local exchange, no workgroup barrier) and overwrites the two slots with the 2 x 3 x 40 FP32
//      horizontal results;
//   3. after one barrier, every output row pair whose 7 input rows (3p - 2 ... 3p + 4, clamped) are complete runs
//      the vertical pass, Dec and the store: one lane per output pixel, the arithmetic of filter_strip_c1 (kH:
//      RGBA16F rows of pitch_bytes, (v dec_scale) pi rounded to FP16, alpha 1, as post_filter_c1_s4_h_band;
//      otherwise 3 x f32 as post_filter_c1_s4).
// The ring holds 2 NW + 6 rows: the step's rows and the 6 older rows that pending pairs still need (row r in slot
// (r - first row) mod (2 NW + 6); slots 0 ... 5 are also written to 6 extra rows past the end, so the 5 rows of a
// vertical tap never wrap). Two barriers per step (after the staging, after the horizontal pass). Halo work: 4 of 64
// columns, and 4 D rows per 3 C rows (C = 68: 2 %). Chunks 0 ... k1 - 1 have C pairs, later ones c2 (arguments; the
// last workgroups dispatched can be smaller to shorten the tail of the launch).
// LDS (NW = 8): 352 staged texels x 48 B + 28 ring rows x 480 B = 30336 bytes, 4 workgroups of 8 waves64 per WGP;
// 96 VGPRs (wave index and per-frame floats kept scalar, per-phase lane decompositions recomputed per step:
// tf_uniform, tf_opaque), so 8 waves64 per SIMD as the two-kernel path.
// Needs tf_supported (runner/post.py, runtime.cpp): the step footprint (floor(63 rx) + 4) x (floor((2 NW - 2) ry) + 5)
// fits kTfCap (true for D >= 2.25 R, i.e. the Quality, Balanced, Performance and Ultra Performance modes), dw >= 4
// (history spans) and the 32-bit offsets of post_recon2_body.

namespace {

constexpr int kTfCols = 64;   // D columns computed per strip (one wave64 row)
constexpr int kTfOwn = 60;    // D columns owned per strip
constexpr int kTfOutX = 40;   // output columns per strip
#ifndef WG_TF_CAP
#define WG_TF_CAP 352
#endif
// Render pixels staged per step: WG_TF_CAP, at most one per lane.
#ifndef WG_TF_CAP4
#define WG_TF_CAP4 224  // NW = 4: 32 x 7 texels (the Quality footprint of 4 row pairs)
#endif
#ifndef WG_TF_CAP16
#define WG_TF_CAP16 576  // NW = 16: 32 x 18 texels
#endif
template <int NW>
constexpr int tf_cap() { return NW == 4 ? WG_TF_CAP4 : NW == 16 ? WG_TF_CAP16 : WG_TF_CAP; }
#ifndef WG_TF_AB_V
#define WG_TF_AB_V 0  // timing only: 1 = vertical pass without the output stores, 2 = no vertical pass
#endif
#ifndef WG_TF_AB_B2
#define WG_TF_AB_B2 0  // timing only: no barrier between the horizontal and the vertical pass
#endif
#ifndef WG_TF_AB_PAD
#define WG_TF_AB_PAD 0  // timing only: extra FP32 FMAs per wave and step (is the kernel issue-bound?)
#endif
#ifndef WG_TF_LDSPAD
#define WG_TF_LDSPAD 0  // experiments: LDS padding per workgroup (fewer workgroups per WGP)
#endif
#ifndef WG_TF_AB_H
#define WG_TF_AB_H 0  // timing only: no horizontal pass (the ring keeps the display scratch)
#endif
#ifndef WG_TF_VMODE
#define WG_TF_VMODE 1  // vertical pass: 1 = lane per output pixel, 2 = lane per row pair and column, 3 = wave per
                       // output row (all exact; 1 measured best)
#endif
#ifndef WG_TF_LANEREC
#define WG_TF_LANEREC 0  // the lane's column values recomputed in every step (not hoisted out of the walk)
#endif
#ifndef WG_TF_DECANY
#define WG_TF_DECANY 0  // vertical pass Dec: polynomial under one uniform branch (else a branch per value)
#endif
#ifndef WG_TF_VDP
#define WG_TF_VDP 15  // ablation of the kVDiet parts: 1 lz5w, 2 ring base + offsets, 4 decomposition, 8 store offset
#endif
#ifndef WG_TF_MIRROR
#define WG_TF_MIRROR 1  // ring rows 0 ... 5 mirrored past the end (wrap-free vertical reads)
#endif
#ifndef WG_TF_PROF
#define WG_TF_PROF 0  // profiling build: phase cycle counts instead of the reduced history
#endif
#ifndef WG_TF_NT
#define WG_TF_NT 0  // non-temporal history and output stores
#endif
#ifndef WG_TF_PIPE
#define WG_TF_PIPE 2  // staging loads of step s + 1 issued before the vertical pass of step s (2: right after
                       // the reconstruction of step s, before the horizontal pass)
#endif
#ifndef WG_TF_WAVES
#define WG_TF_WAVES 8  // register target: 8 waves64 per SIMD (at most 96 VGPRs)
#endif

// Mirrored ring rows (WG_TF_MIRROR) only where the 6 extra rows keep 4 workgroups per WGP (not for the native-AA cap).
template <int CAP>
constexpr bool tf_mirror() { return WG_TF_MIRROR && CAP <= 512; }

template <int NW, int CAP>
struct TfShared {
  R2Texel tex[CAP];
  // filter rows: FP32 horizontal results (or a wave's FP16 display scratch); WG_TF_MIRROR: rows 0 ... 5 are also
  // kept in rows 2 NW + 6 ... 2 NW + 11, so 7 consecutive slots never wrap
  float ring[2 * NW + 6 + (tf_mirror<CAP>() ? 6 : 0)][3][kTfOutX];
#if WG_TF_LDSPAD > 0
  char pad[WG_TF_LDSPAD];  // experiments: fewer workgroups per WGP
#endif
};

// The same value, opaque to the optimizer: per-phase lane decompositions are recomputed in every step of the walk
// instead of being hoisted out of the loop (where they would hold registers through the reconstruction).
WG_FN int tf_opaque(int v) {
#if defined(__AMDGCN__)
  asm volatile("" : "+v"(v));
#endif
  return v;
}
#if defined(__AMDGCN__)
// Value of lane ^ 1 (DPP quad_perm [1, 0, 3, 2]: no LDS traffic).
WG_FN float tf_xor1(float v) {
  return __builtin_bit_cast(float, __builtin_amdgcn_mov_dpp(__builtin_bit_cast(int, v), 0xb1, 0xf, 0xf, true));
}
WG_FN bool tf_wave64() { return __builtin_amdgcn_wavefrontsize() == 64; }
// v is the same in every lane of the wave (tells the compiler so: values derived from it stay scalar).
WG_FN int tf_uniform(int v) { return tf_wave64() ? __builtin_amdgcn_readfirstlane(v) : v; }
WG_FN float tf_uniformf(float v) { return __builtin_bit_cast(float, tf_uniform(__builtin_bit_cast(int, v))); }
#else
WG_FN int tf_uniform(int v) { return v; }
WG_FN float tf_uniformf(float v) { return v; }
WG_FN float tf_xor1(float v) { return wg_emu_xor(v, 1u); }
WG_FN bool tf_wave64() { return false; }
#endif

// The 64 lanes of a strip row exchange values through LDS. In wave64 they are one wave: its LDS operations execute
// in order, and the fences keep the compiler from moving them across this point (no hardware wait at wavefront scope).
// In wave32 (post.co, the emulator) the row spans two waves: a workgroup barrier. Every lane of the workgroup calls it.
WG_FN void tf_row_sync() {
#if defined(__AMDGCN__)
  if (tf_wave64()) {
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "wavefront");
    __builtin_amdgcn_wave_barrier();
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "wavefront");
    return;
  }
#endif
  wg_barrier_lds();
}

// 8-byte store of the full history / the output: WG_TF_NT = 1 non-temporal (neither is read again in this frame).
WG_FN void tf_store8(char* q, v4h v) {
#if defined(__AMDGCN__)
  if (WG_TF_NT) {
    __builtin_nontemporal_store(__builtin_bit_cast(u64, v), (u64*)q);
    return;
  }
#endif
  *(v4h*)q = v;
}

// Raw inputs of one staged render pixel (pack12 layout), loaded ahead of their conversion.
struct TfRaw {
  v4f c;          // colour
  u32 d0, d1;     // A8: codes A0 A1 A2 B0 | B1 B2, C2
  u32 c01;        // C4: C0, C1
  u32 m0, m1, m2; // motions
};

WG_FN void tf_fetch(TfRaw& t, const v4f* color, const u32* A8, const u32* C4, const u32* m_near, const u32* m_center,
                    const u32* m_far, int idx) {
  const u32 ui = (u32)idx;
  t.c = *(const v4f*)((const char*)color + ui * 16u);
  t.d0 = *(const u32*)((const char*)A8 + ui * 8u);
  t.d1 = *(const u32*)((const char*)A8 + ui * 8u + 4u);
  t.c01 = *(const u32*)((const char*)C4 + ui * 4u);
  t.m0 = *(const u32*)((const char*)m_near + ui * 4u);
  t.m1 = *(const u32*)((const char*)m_center + ui * 4u);
  t.m2 = *(const u32*)((const char*)m_far + ui * 4u);
}

// The conversion of load_texel12 + texel_f + r2_store_features (same operations: bit-identical texel).
WG_FN void tf_put(R2Texel& d, const TfRaw& r, float e) {
  PostTexel t;
  t.ec0 = enc_in(e * r.c[0]);
  t.ec1 = enc_in(e * r.c[1]);
  t.ec2 = enc_in(e * r.c[2]);
  t.a = r.d0 & 0xffffffu;
  t.b = (r.d0 >> 24) | ((r.d1 & 0xffffu) << 8);
  t.ca = (float)__builtin_bit_cast(_Float16, (unsigned short)(r.c01 & 0xffffu));
  t.cb = (float)__builtin_bit_cast(_Float16, (unsigned short)(r.c01 >> 16));
  t.cc = (float)__builtin_bit_cast(_Float16, (unsigned short)(r.d1 >> 16));
  r2_store_features(d, texel_f(t));
  d.m[0] = r.m0;
  d.m[1] = r.m1;
  d.m[2] = r.m2;
}

// kVDiet (WG_TF_VMODE 1): vertical pass with the phase weights selected per lane (lz5w, no divergent lz5 pair), the
// interior taps read from one ring address plus constant offsets, and 24-bit lane decomposition; bit-identical.
// kSel: the candidate motions by selects (post_core_s kSel, post_select_m); kIdx: unclamped, pre-scaled LDS texel
// offsets for the gather and motion reads (stex). Each the same values; all three on in the default kernels.
template <int NW, bool kH, int CAP, bool kDiet = (WG_POST_DIET != 0), bool kVDiet = false, bool kSel = false,
          bool kIdx = false, bool kTrim = false, class MoT = MoOff>
WG_FN void post_tf_body(const v4f* color, const u32* A, const v4h* C, const v4h* hist_prev, const u32* m_near,
                        const u32* m_center, const u32* m_far, const float* w, v4h* hist_out, u32* reduced, void* out,
                        WgPostParams p, const float* expo, int ow, int oh, float dec_scale, float pi,
                        int pitch_bytes, int cpairs, int k1, int c2 WG_LZ_PARAM, TfShared<NW, CAP>& sh,
                        const MoT mo = MoT()) {
  constexpr int kRing = 2 * NW + 6;
  constexpr int kThreads = 64 * NW;
  constexpr int kTfCap = CAP;
  constexpr bool kMirror = tf_mirror<CAP>();
  constexpr int kPer = (CAP + kThreads - 1) / kThreads;  // staged texels per lane
  if (p.dw < 4) return;  // history spans (the host checks tf_supported)
  if (expo) {  // as post_recon2_body and the filters
    p.e = expo[0] == 0.0f ? 1.0f : expo[0];
    p.e_prev = expo[1] == 0.0f ? 1.0f : expo[1];
    if (kDiet) {  // precomputed by exposure_pair_diet (the same expressions, once)
      p.ratio = tf_uniformf(expo[3]);
      dec_scale = tf_uniformf(expo[4]);
    } else {
      p.ratio = tf_uniformf((float)((double)p.e / (double)p.e_prev));
      dec_scale = tf_uniformf((float)(0.01 / (double)p.e));
    }
  }
  const int lane = (int)wg_lane_id();
  const int wave = tf_uniform(lane >> 6), wl = lane & 63;  // wave index: uniform (a scalar register)
  const int strips = (ow + kTfOutX - 1) / kTfOutX;
  const int strip = (int)wg_block_id() % strips, chunk = (int)wg_block_id() / strips;
  const int X0 = strip * kTfOwn;
  // Chunks 0 ... k1 - 1 have cpairs output row pairs, the later ones (dispatched last) c2: smaller workgroups at the end
  // of the launch shorten its tail.
  const int csize = chunk < k1 ? cpairs : c2;
  const int P0 = chunk < k1 ? chunk * cpairs : k1 * cpairs + (chunk - k1) * c2;
  const int p_end = mini(P0 + csize, (oh + 1) / 2);          // output row pairs P0 ... p_end - 1
  const int own_r0 = 3 * P0, own_r1 = mini(3 * (P0 + csize), p.dh);
  const int r_first = 3 * P0 - 2 < 0 ? 0 : 3 * P0 - 2;       // even (P0 even)
  const int r_need = mini(3 * (p_end - 1) + 4, p.dh - 1);    // last D row the filter reads
  const float s = w[0];

  // Footprint columns of the strip (a_x does not decrease with x).
  const int c0 = floor_i(pos_x(clampi(X0 - 2, 0, p.dw - 1), p) - 0.5f) - 1;
  const int c1 = floor_i(pos_x(clampi(X0 + kTfCols - 3, 0, p.dw - 1), p) - 0.5f) + 1;
  const int ncol = c1 - c0 + 1;
  const float inv_ncol = tf_uniformf(1.0f / (float)ncol);

  // Footprint rows of the step starting at row ys: window rows of its row pairs ys ... yl, plus one row below for the
  // odd rows' motion reads (as post_recon2_body).
  const auto step_rows = [&](int ys, int& r0s, int& nst) {
    const int yl = mini(ys + 2 * (NW - 1), r_need) & ~1;
    r0s = floor_i(pos_y(ys, p) - 0.5f) - 1;
    const int nrow = floor_i(pos_y(yl, p) - 0.5f) + 2 - r0s + 1;
    nst = ncol * nrow < kTfCap ? ncol * nrow : kTfCap;
  };
  const auto fetch = [&](int i, int r0s, TfRaw& t) {  // raw inputs of staged texel i
    const int r = (int)(((float)i + 0.5f) * inv_ncol), c = i - mul_i24(r, ncol);
    const int idx = (int)mul24((u32)clampi(r0s + r, 0, p.rh - 1), (u32)p.rw) + clampi(c0 + c, 0, p.rw - 1);
    tf_fetch(t, color, A, (const u32*)C, m_near, m_center, m_far, idx);
  };
  // Staging is software-pipelined: the loads of step s + 1 are issued right after the reconstruction of step s
  // (WG_TF_PIPE = 2; 1: after the second barrier) and converted and written to LDS at the start of step s + 1, so their
  // latency hides behind the horizontal and vertical passes. kPer texels per lane (1, or 2 for the native-AA cap).
  TfRaw raw[kPer];
  int r0 = 0, nstage = 0;
  step_rows(r_first, r0, nstage);
  if (WG_TF_PIPE) {
#pragma unroll
    for (int q = 0; q < kPer; ++q) fetch(mini(tf_opaque(lane) + q * kThreads, nstage - 1), r0, raw[q]);
  }

#if WG_TF_PROF
  // Profiling build (timing analysis only; the reduced history is not written): per wave, the summed shader cycles of
  // each phase over the walk go to reduced[(block * NW + wave) * 12 + phase] (0 ... 5, then steps, 1, and the
  // reconstruction's gather / pixel 0 / pixel 1 in 8 ... 10); timing analysis only.
  u64 tf_t[7] = {0, 0, 0, 0, 0, 0, 0};
  u64 tf_acc[6] = {0, 0, 0, 0, 0, 0};
  u64 tf_sub[3] = {0, 0, 0}, tf_last = 0;
#define TF_PROF_MARK(i)                    \
  do {                    \
    tf_t[i] = __builtin_readcyclecounter();    \
    if (i > 0) tf_acc[(i) - 1] += tf_t[i] - tf_t[(i) - 1];    \
    if (i == 2) tf_last = tf_t[2];    \
  } while (0)
#define TF_PROF_SUB(j)                                                          \
  do {                                                                          \
    const u64 now = __builtin_readcyclecounter();                               \
    tf_sub[j] += now - tf_last;                                                 \
    tf_last = now;                                                              \
  } while (0)
#else
#define TF_PROF_MARK(i) ((void)0)
#define TF_PROF_SUB(j) ((void)0)
#endif
  int base_slot = 0;  // ring slot of row y_step: (y_step - r_first) mod kRing
  int p_next = P0;    // first output row pair not written yet
#pragma nounroll
  for (int y_step = r_first; y_step <= r_need; y_step += 2 * NW) {
    // 1. Stage the footprint (loaded ahead into raw).
    TF_PROF_MARK(0);
    MoQ mq;
    if (MoT::on) mo_fetch(mo, clampi(X0 - 2 + wl, 0, p.dw - 1), y_step + 2 * wave, p.dw, p.dh, mq);
    {
#pragma unroll
      for (int q = 0; q < kPer; ++q) {
        const int i = tf_opaque(lane) + q * kThreads;
        if (!WG_TF_PIPE && i < nstage) fetch(i, r0, raw[q]);
        if (i < nstage) tf_put(sh.tex[i], raw[q], p.e);
      }
    }
    if (WG_TF_AB_PAD > 0) {  // timing only: WG_TF_AB_PAD dependent FP32 FMAs (4 independent chains) per wave and step
      float a0 = (float)lane, a1 = a0 + 1.0f, a2 = a0 + 2.0f, a3 = a0 + 3.0f;
#pragma unroll
      for (int i = 0; i < WG_TF_AB_PAD / 4; ++i) {
        a0 = __builtin_fmaf(a0, 0.999f, 0.5f);
        a1 = __builtin_fmaf(a1, 0.999f, 0.5f);
        a2 = __builtin_fmaf(a2, 0.999f, 0.5f);
        a3 = __builtin_fmaf(a3, 0.999f, 0.5f);
      }
      if (ab_live(a0 + a1 + a2 + a3)) *(float*)out = 0.0f;
    }
    TF_PROF_MARK(1);
    wg_barrier_lds();
    TF_PROF_MARK(2);
    const int r0c = r0;  // this step's first footprint row (r0 moves on to the next step after the reconstruction)
    const auto sidx = [&](int c, int r) { return clampi(mul_i24(r - r0c, ncol) + (c - c0), 0, kTfCap - 1); };
    // kIdx (runtime option post_idx, kernel suffix i): the texel at byte offset row term (scalar, (r - r0c) ncol 48) +
    // column term (per lane, pre-scaled by 48), no clamp. Under tf_supported every index is already inside
    // [0, ncol nrow - 1] <= kTfCap - 1: columns ax - 1 ... ax + 1 and floor(px) lie in [c0, c1] (pos_x does not
    // decrease, c0 / c1 come from the clamped strip edges), rows b - 1 ... b + 2 in [r0c, r0c + nrow - 1] (ry < 1), so
    // the clamp never changes the index: the same LDS texels, bit-identical.
    const int rowb = tf_uniform(48 * ncol);
    // kTrim (post_trim): the row term (r - r0c) rowb is wave-uniform (r comes from y only): a scalar value.
    const auto stex = [&](int colb, int r) -> const R2Texel& {
      return *(const R2Texel*)((const char*)&sh.tex[0] +
                               (u32)((kTrim ? tf_uniform((r - r0c) * rowb) : (r - r0c) * rowb) + colb));
    };

    // 2. This wave: row pair (yv, yv + 1). This lane's D column (virtual column xv, clamped to D). WG_TF_LANEREC = 1
    // recomputes these from an opaque lane index in every step (fewer VGPRs, measured slower; default 0).
    const int xv = X0 - 2 + (WG_TF_LANEREC ? tf_opaque(wl) : wl);
    const int x = clampi(xv, 0, p.dw - 1);
    const bool own_x = xv >= X0 && xv < X0 + kTfOwn && xv < p.dw;
    const float px = pos_x(x, p);
    const int ax = floor_i(px - 0.5f);
    const int fx = clampi(floor_i(px), 0, p.rw - 1);
    const int cb0 = 48 * (ax - c0), cbf = 48 * (floor_i(px) - c0);  // kIdx: column byte offsets of ax and floor(px)
    const int yv = y_step + 2 * wave;
    const bool wact = yv <= r_need;  // uniform per wave
    int slot = base_slot + 2 * wave;
    if (slot >= kRing) slot -= kRing;
    char* const scr = (char*)&sh.ring[slot][0][0];  // rows slot, slot + 1 (480 bytes each)
    if (wact) {
      const int y0 = yv;
      const bool has1 = y0 + 1 < p.dh;
      const float py0 = pos_y(y0, p), py1 = pos_y(y0 + 1, p);
      const int b = floor_i(py0 - 0.5f);
      Gather g0, g1;
      gather_zero(g0);
      gather_zero(g1);
#pragma nounroll
      for (int j = -1; j <= 1; ++j) {
        const float v0 = ((float)(b + j) + 0.5f) - py0;
        const float v1 = ((float)(b + j) + 0.5f) - py1;
        const float vv0 = kTapK * v0 * v0, vv1 = kTapK * v1 * v1;
#pragma unroll
        for (int i = -1; i <= 1; ++i) {
          const float u = ((float)(ax + i) + 0.5f) - px;
          const float uu = kTapK * u * u, uv0 = 2.0f * kTapK * u * v0, uv1 = 2.0f * kTapK * u * v1;
          const PostTexelF t = r2_features(kIdx ? stex(cb0 + 48 * i, b + j) : sh.tex[sidx(ax + i, b + j)]);
          gather_add_f(g0, t, uu, vv0, uv0);
          gather_add_f(g1, t, uu, vv1, uv1);
        }
      }
      TF_PROF_SUB(0);
      float hsum[3] = {0.0f, 0.0f, 0.0f};
#pragma nounroll
      for (int k = 0; k < (has1 ? 2 : 1); ++k) {
        const int y = y0 + k;
        const float py = k == 0 ? py0 : py1;
        const int fy = clampi(floor_i(py), 0, p.rh - 1);
        float mx[3], my[3];
        const int pxi = floor_i(px), pyi = floor_i(py);
        if (kIdx) {
          unpack_h2(stex(cbf, pyi).m[0], mx[0], my[0]);
          unpack_h2(stex(cbf, pyi).m[2], mx[2], my[2]);
          motion_bilinear_mix_rel(px, py, [&](int c, int r) { return stex(cb0 + 48 * c, r).m[1]; }, mx[1], my[1]);
        } else {
          unpack_h2(sh.tex[sidx(pxi, pyi)].m[0], mx[0], my[0]);
          unpack_h2(sh.tex[sidx(pxi, pyi)].m[2], mx[2], my[2]);
          motion_bilinear_mix(px, py, [&](int c, int r) { return sh.tex[sidx(c, r)].m[1]; }, mx[1], my[1]);
        }
        float hh[3], oh3[3];
        if (MoT::on)
          mo_core<kTrim>(mo, mq, k, k == 0 ? g0 : g1, x, y, fx, fy, p, s, color, hist_prev, hh, oh3);
        else
        post_core_s<false, true, false, kSel, kTrim>(k == 0 ? g0 : g1, x, y, fx, fy, mx, my, p, s, color, hist_prev, hh,
                                              oh3);
        if (own_x && y >= own_r0 && y < own_r1) {
          v4h t;
          t[0] = (_Float16)enc_e(hh[0]);
          t[1] = (_Float16)enc_e(hh[1]);
          t[2] = (_Float16)enc_e(hh[2]);
          t[3] = (_Float16)0.0f;
          tf_store8((char*)hist_out + ((kTrim ? (u32)tf_uniform((int)mul24((u32)y, (u32)p.dw * 8u))
                                              : mul24((u32)y, (u32)p.dw * 8u)) + (u32)x * 8u), t);
        }
        char* const srow = scr + (kTrim ? tf_uniform(k * (3 * kTfOutX * 4)) : k * (3 * kTfOutX * 4));
#pragma unroll
        for (int c = 0; c < 3; ++c) *(_Float16*)(srow + c * 128 + wl * 2) = display_value(oh3[c]);
#pragma unroll
        for (int c = 0; c < 3; ++c) hsum[c] += hh[c];
        TF_PROF_SUB(1 + k);
      }
      // Reduced history: mean of H^ over the 2 x 2 block; the partner column x + 1 is in lane ^ 1.
      float nb[3];
#pragma unroll
      for (int c = 0; c < 3; ++c) nb[c] = tf_xor1(hsum[c]);
      if ((wl & 1) == 0 && own_x && x + 1 < p.dw && has1 && y0 >= own_r0 && y0 + 1 < own_r1) {
        u32 code = 0;
#pragma unroll
        for (int c = 0; c < 3; ++c) {
          const float m = (hsum[c] + nb[c]) * 0.25f;
          const float v = clampf(enc_e(m) * (1.0f / 2.45f), 0.0f, 1.0f);
          code |= (u32)wg_rint(v * 1023.0f) << (10 * c);
        }
        if (!WG_TF_PROF) *(u32*)((char*)reduced + ((kTrim ? (u32)tf_uniform((int)mul24((u32)(y0 >> 1), (u32)(p.dw >> 1) * 4u))
                                                  : mul24((u32)(y0 >> 1), (u32)(p.dw >> 1) * 4u)) + (u32)(x >> 1) * 4u)) = code;
      }
    }
    if (WG_TF_PIPE == 2) {
      // Issue the next step's staging loads. Every lane loads (lanes past the footprint, and the last step, load a
      // valid clamped texel that is never written to LDS): an unconditional definition, so the compiler does not
      // keep the registers of raw alive through the reconstruction.
      step_rows(y_step + 2 * NW, r0, nstage);
#pragma unroll
      for (int q = 0; q < kPer; ++q) fetch(mini(tf_opaque(lane) + q * kThreads, nstage - 1 < 0 ? 0 : nstage - 1), r0, raw[q]);
    }
    TF_PROF_MARK(3);
    // Horizontal pass of the wave's two rows: lane (row hk, channel hc, cell hg) reads display columns 6 hg ... 6 hg + 9
    // of the strip (5 dwords) and computes output columns 4 hg ... 4 hg + 3 (filter_row_c1).
    tf_row_sync();
    const int hl = tf_opaque(wl);
    const bool hact = wact && hl < 60;
    const int hk = hl >= 30 ? 1 : 0, hrem = hl - 30 * hk, hc = hrem >= 20 ? 2 : (hrem >= 10 ? 1 : 0),
              hg = hrem - 10 * hc;
    float hv[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    if (hact && !WG_TF_AB_H) {
      const u32* src = (const u32*)(scr + hk * (3 * kTfOutX * 4) + hc * 128 + hg * 12);
      float t[10];
#pragma unroll
      for (int i = 0; i < 5; ++i) {
        const u32 q = src[i];
        t[2 * i] = WG_LZ_LO(q);
        t[2 * i + 1] = WG_LZ_HI(q);
      }
      hv[0] = lz5<true>(t WG_LZ_ARG);
      hv[1] = lz5<false>(t + 2 WG_LZ_ARG);
      hv[2] = lz5<true>(t + 3 WG_LZ_ARG);
      hv[3] = lz5<false>(t + 5 WG_LZ_ARG);
    }
    tf_row_sync();
    if (hact && !WG_TF_AB_H) {
      v4f h;
      h[0] = hv[0];
      h[1] = hv[1];
      h[2] = hv[2];
      h[3] = hv[3];
      *(v4f*)(scr + hk * (3 * kTfOutX * 4) + hc * (kTfOutX * 4) + hg * 16) = h;
      if (kMirror && slot + hk < 6) *(v4f*)(scr + (kRing + hk) * (3 * kTfOutX * 4) + hc * (kTfOutX * 4) + hg * 16) = h;
    }
    TF_PROF_MARK(4);
    if (!WG_TF_AB_B2) wg_barrier_lds();
    TF_PROF_MARK(5);

    if (WG_TF_PIPE == 1) {
      // Issue the next step's staging loads. Every lane loads (lanes past the footprint, and the last step, load a
      // valid clamped texel that is never written to LDS): an unconditional definition, so the compiler does not
      // keep the registers of raw alive through the reconstruction.
      step_rows(y_step + 2 * NW, r0, nstage);
#pragma unroll
      for (int q = 0; q < kPer; ++q) fetch(mini(tf_opaque(lane) + q * kThreads, nstage - 1 < 0 ? 0 : nstage - 1), r0, raw[q]);
    }
    // 3. Vertical pass, Dec and output of the row pairs whose input rows are complete (rows r_first ... r_done - 1).
    const int r_done = mini(y_step + 2 * NW, r_need + 1);
    const int p_rdy = r_done > r_need ? p_end : mini(p_end, (r_done - 5) / 3 + 1);
    // Output pixels (xg, y0 + d), d < n, from the 7 (n = 2) or 5 (n = 1, phase dy0) filtered rows t: vertical lz5, then
    // Dec as filter_strip_c1 (exp2 - 1 for every value, the polynomial only when some lane has a small value), store.
    const auto emit = [&](int xg, int y0, int n, int dy0, const float (*t)[3]) {
      float v[2][3], qc[2][3];
      bool any_small = false;
#pragma unroll
      for (int d = 0; d < 2; ++d) {
        if (d >= n) break;
        // kVDiet: the phase weights selected once per lane and row (one lz5w for both phases, no divergent branch pair)
        // as integer adds of an opaque per-lane bit delta (b(kLz0) + (b(kLz4) - b(kLz0)) = b(kLz4) exactly): selects
        // of two constants would keep the constants in registers through the whole kernel (one VGPR over 96).
        const bool ev = d + dy0 == 0;
#if WG_LZ_PROFILE == 2
        // The weights are kernel arguments (scalar registers): the lane's phase picks them by selects.
        const bool wsel = kVDiet && (WG_TF_VDP & 1);
        const float w0 = lz_wsel(lw, ev, 0), w4 = lz_wsel(lw, ev, 4);
        const float w1 = lz_wsel(lw, ev, 1), w3 = lz_wsel(lw, ev, 3);
#else
        const u32 b0 = __builtin_bit_cast(u32, kLz0), b1 = __builtin_bit_cast(u32, kLz1);
        const u32 b3 = __builtin_bit_cast(u32, kLz3), b4 = __builtin_bit_cast(u32, kLz4);
        const bool wsel = kVDiet && (WG_TF_VDP & 1);
        const u32 e04 = wsel ? (u32)tf_opaque(ev ? 0 : (int)(b4 - b0)) : 0u;
        const u32 e13 = wsel ? (u32)tf_opaque(ev ? 0 : (int)(b3 - b1)) : 0u;
        const float w0 = __builtin_bit_cast(float, b0 + e04), w4 = __builtin_bit_cast(float, b4 - e04);
        const float w1 = __builtin_bit_cast(float, b1 + e13), w3 = __builtin_bit_cast(float, b3 - e13);
#endif
#pragma unroll
        for (int c = 0; c < 3; ++c) {
          float tv[5];
#pragma unroll
          for (int k = 0; k < 5; ++k) tv[k] = t[2 * d + k][c];
          qc[d][c] = clampf(WG_LZ_VAL(wsel ? lz5w(tv, w0, w1, w3, w4 WG_LZ_ARG) : ev ? lz5<true>(tv WG_LZ_ARG) : lz5<false>(tv WG_LZ_ARG)),
                            1e-4f, 3.0f);
          if (WG_TF_DECANY) {
            v[d][c] = wg_exp2(qc[d][c] * kDecK) - 1.0f;
            any_small |= qc[d][c] < kDecPolyMax;
          } else {
            v[d][c] = qc[d][c] < kDecPolyMax ? dec_m1_poly(qc[d][c]) : wg_exp2(qc[d][c] * kDecK) - 1.0f;
          }
        }
      }
      if (any_small) {
#pragma unroll
        for (int d = 0; d < 2; ++d) {
          if (d >= n) break;
#pragma unroll
          for (int c = 0; c < 3; ++c)
            if (qc[d][c] < kDecPolyMax) v[d][c] = dec_m1_poly(qc[d][c]);
        }
      }
#pragma unroll
      for (int d = 0; d < 2; ++d) {
        if (d >= n) break;
        const int y = y0 + d;
        if (WG_TF_AB_V == 1) {
          if (ab_live(v[d][0] + v[d][1] + v[d][2])) *(float*)out = 0.0f;
        } else if (kH) {
          v4h o;
          o[0] = framebuffer_half((v[d][0] * dec_scale) * pi);
          o[1] = framebuffer_half((v[d][1] * dec_scale) * pi);
          o[2] = framebuffer_half((v[d][2] * dec_scale) * pi);
          o[3] = (_Float16)1.0f;
          tf_store8((char*)out + (mul24((u32)y, (u32)pitch_bytes) + (u32)xg * 8u), o);
        } else {
          const u32 xb = kVDiet && (WG_TF_VDP & 8) ? mul24((u32)xg, 12u) : (u32)xg * 12u;
          float* o = (float*)((char*)out + (mul24((u32)y, (u32)ow * 12u) + xb));
          o[0] = v[d][0] * dec_scale;
          o[1] = v[d][1] * dec_scale;
          o[2] = v[d][2] * dec_scale;
        }
      }
    };
    if (WG_TF_VMODE == 3) {  // wave = output row (rows of the ready pairs, round robin), lane = output column
      const int nrows = WG_TF_AB_V == 2 ? 0 : mini(2 * p_rdy, oh) - 2 * p_next;
      const int xo = tf_opaque(wl), xg = strip * kTfOutX + xo;
      for (int r = wave; r < nrows; r += NW) {  // uniform per wave
        const int pr = r >> 1, dy = r & 1;
        const int pp = p_next + pr, y = 2 * pp + dy;
        int sl[5];
#pragma unroll
        for (int k = 0; k < 5; ++k) sl[k] = tf_uniform((clampi(3 * pp - 2 + 2 * dy + k, 0, p.dh - 1) - r_first) % kRing);
        if (xo < kTfOutX && xg < ow) {
          float t[5][3];
#pragma unroll
          for (int k = 0; k < 5; ++k)
#pragma unroll
            for (int c = 0; c < 3; ++c) t[k][c] = sh.ring[sl[k]][c][xo];
          emit(xg, y, 1, dy, t);
        }
      }
    } else if (WG_TF_VMODE == 1 && kVDiet) {  // as below, with the vertical pass diet (kVDiet)
      const int nitems = WG_TF_AB_V == 2 ? 0 : (p_rdy - p_next) * 2 * kTfOutX;
      // Rows 3 p_next - 2 ... 3 p_rdy + 1 inside D (no clamping): the ring slot of row 3 pp - 2 + 2 dy + k is
      // (s0 + 3 pr + 2 dy + k) mod kRing with the uniform s0 (WG_TF_MIRROR: 5 consecutive slots never wrap).
      const bool interior = 3 * p_next - 2 >= 0 && 3 * p_rdy + 1 <= p.dh - 1;
      const int s0 = interior ? (3 * p_next - 2 - r_first) % kRing : 0;
      // One item (output pixel). kVDiet: the interior steps run their own loop (vin = true), so the compiler cannot
      // merge the two load paths into one block of 15 per-tap addresses (it did: the constant offsets stayed adds).
      const auto vitem = [&](int it, bool vin) {
        // it < 2^16 (a step adds 2 NW input rows, so at most about 2 NW / 3 + 1 ready pairs of 80 items, whatever the
        // chunk size): it / 80 = (it * 52429) >> 22 exactly (checked for every it < 2^16), all products fit 24 bits.
        static_assert(2 * kTfOutX == 80, "kVDiet decomposition assumes 80 items per row pair");
        const int pr = kVDiet && (WG_TF_VDP & 4) ? (int)(mul24((u32)it, 52429u) >> 22) : it / (2 * kTfOutX);
        const int rem = kVDiet && (WG_TF_VDP & 4) ? it - (int)mul24((u32)pr, 2u * kTfOutX) : it - pr * (2 * kTfOutX);
        const int dy = rem >= kTfOutX ? 1 : 0, xo = rem - dy * kTfOutX;
        const int pp = p_next + pr, y = 2 * pp + dy, xg = strip * kTfOutX + xo;
        if (y >= oh || xg >= ow) return;
        float t[5][3];
        if (kVDiet && (WG_TF_VDP & 2) && kMirror && vin) {
          // One ring address; the 15 taps at constant offsets (ds_load offset field). 5 consecutive slots never wrap.
          int u = s0 + 3 * pr + 2 * dy;
          u = u >= kRing ? u - kRing : u;
          u = u >= kRing ? u - kRing : u;
          const float* rp = &sh.ring[0][0][0] + (mul24((u32)u, 3u * kTfOutX) + (u32)xo);
#pragma unroll
          for (int k = 0; k < 5; ++k)
#pragma unroll
            for (int c = 0; c < 3; ++c) t[k][c] = rp[k * 3 * kTfOutX + c * kTfOutX];
        } else if (interior) {
          int u = s0 + 3 * pr + 2 * dy;
          u = u >= kRing ? u - kRing : u;
          u = u >= kRing ? u - kRing : u;
#pragma unroll
          for (int k = 0; k < 5; ++k) {
            const int sl = kMirror ? u + k : (u + k >= kRing ? u + k - kRing : u + k);
#pragma unroll
            for (int c = 0; c < 3; ++c) t[k][c] = sh.ring[sl][c][xo];
          }
        } else if (kDiet) {
          // One modulo: row k is row 0 plus a step of 0 ... 4 (the clamps), so its slot is slot 0 plus that step,
          // wrapped once. clamp(x, 0, dh - 1) - r_first >= 0: rows >= 3 P0 - 2 = r_first, or 0 = r_first when P0 = 0.
          const int xr = 3 * pp - 2 + 2 * dy;
          const int row0 = clampi(xr, 0, p.dh - 1);
          const int sl0 = (int)((u32)(row0 - r_first) % (u32)kRing);
#pragma unroll
          for (int k = 0; k < 5; ++k) {
            const int st = clampi(xr + k, 0, p.dh - 1) - row0;
            const int sl = sl0 + st >= kRing ? sl0 + st - kRing : sl0 + st;
#pragma unroll
            for (int c = 0; c < 3; ++c) t[k][c] = sh.ring[sl][c][xo];
          }
        } else {
#pragma unroll
          for (int k = 0; k < 5; ++k) {
            const int sl = (clampi(3 * pp - 2 + 2 * dy + k, 0, p.dh - 1) - r_first) % kRing;
#pragma unroll
            for (int c = 0; c < 3; ++c) t[k][c] = sh.ring[sl][c][xo];
          }
        }
        emit(xg, y, 1, dy, t);
      };
      if (kVDiet && (WG_TF_VDP & 2) && kMirror && interior) {
        for (int it = tf_opaque(lane); it < nitems; it += kThreads) vitem(it, true);
      } else {
        for (int it = tf_opaque(lane); it < nitems; it += kThreads) vitem(it, false);
      }
    } else if (WG_TF_VMODE == 1) {  // lane = output pixel (row pair, row, column): 5 rows x 3 channels
      const int nitems = WG_TF_AB_V == 2 ? 0 : (p_rdy - p_next) * 2 * kTfOutX;
      // Rows 3 p_next - 2 ... 3 p_rdy + 1 inside D (no clamping): the ring slot of row 3 pp - 2 + 2 dy + k is
      // (s0 + 3 pr + 2 dy + k) mod kRing with the uniform s0 (WG_TF_MIRROR: 5 consecutive slots never wrap).
      const bool interior = 3 * p_next - 2 >= 0 && 3 * p_rdy + 1 <= p.dh - 1;
      const int s0 = interior ? (3 * p_next - 2 - r_first) % kRing : 0;
      for (int it = tf_opaque(lane); it < nitems; it += kThreads) {
        const int pr = it / (2 * kTfOutX), rem = it - pr * (2 * kTfOutX);
        const int dy = rem >= kTfOutX ? 1 : 0, xo = rem - dy * kTfOutX;
        const int pp = p_next + pr, y = 2 * pp + dy, xg = strip * kTfOutX + xo;
        if (y >= oh || xg >= ow) continue;
        float t[5][3];
        if (interior) {
          int u = s0 + 3 * pr + 2 * dy;
          u = u >= kRing ? u - kRing : u;
          u = u >= kRing ? u - kRing : u;
#pragma unroll
          for (int k = 0; k < 5; ++k) {
            const int sl = kMirror ? u + k : (u + k >= kRing ? u + k - kRing : u + k);
#pragma unroll
            for (int c = 0; c < 3; ++c) t[k][c] = sh.ring[sl][c][xo];
          }
        } else if (kDiet) {
          // One modulo: row k is row 0 plus a step of 0 ... 4 (the clamps), so its slot is slot 0 plus that step,
          // wrapped once. clamp(x, 0, dh - 1) - r_first >= 0: rows >= 3 P0 - 2 = r_first, or 0 = r_first when P0 = 0.
          const int xr = 3 * pp - 2 + 2 * dy;
          const int row0 = clampi(xr, 0, p.dh - 1);
          const int sl0 = (int)((u32)(row0 - r_first) % (u32)kRing);
#pragma unroll
          for (int k = 0; k < 5; ++k) {
            const int st = clampi(xr + k, 0, p.dh - 1) - row0;
            const int sl = sl0 + st >= kRing ? sl0 + st - kRing : sl0 + st;
#pragma unroll
            for (int c = 0; c < 3; ++c) t[k][c] = sh.ring[sl][c][xo];
          }
        } else {
#pragma unroll
          for (int k = 0; k < 5; ++k) {
            const int sl = (clampi(3 * pp - 2 + 2 * dy + k, 0, p.dh - 1) - r_first) % kRing;
#pragma unroll
            for (int c = 0; c < 3; ++c) t[k][c] = sh.ring[sl][c][xo];
          }
        }
        emit(xg, y, 1, dy, t);
      }
    } else {  // lane = (row pair, output column): the 7 input rows x 3 channels once, both output rows
      const int nitems = WG_TF_AB_V == 2 ? 0 : (p_rdy - p_next) * kTfOutX;
      // Rows 3 p_next - 2 ... 3 p_rdy + 1 inside D (no clamping): the ring slot of row 3 pp - 2 + k is
      // (s0 + 3 pr + k) mod kRing with the uniform s0.
      const bool interior = 3 * p_next - 2 >= 0 && 3 * p_rdy + 1 <= p.dh - 1;
      const int s0 = interior ? (3 * p_next - 2 - r_first) % kRing : 0;
      for (int it = tf_opaque(lane); it < nitems; it += kThreads) {
        const int pr = it / kTfOutX, xo = it - pr * kTfOutX;
        const int pp = p_next + pr, xg = strip * kTfOutX + xo;
        if (xg >= ow) continue;
        float t[7][3];
        if (interior) {
          int u = s0 + 3 * pr;
          u = u >= kRing ? u - kRing : u;
          u = u >= kRing ? u - kRing : u;
#pragma unroll
          for (int k = 0; k < 7; ++k) {
            const int sl = kMirror ? u + k : (u + k >= kRing ? u + k - kRing : u + k);
#pragma unroll
            for (int c = 0; c < 3; ++c) t[k][c] = sh.ring[sl][c][xo];
          }
        } else {
#pragma unroll
          for (int k = 0; k < 7; ++k) {
            const int sl = (clampi(3 * pp - 2 + k, 0, p.dh - 1) - r_first) % kRing;
#pragma unroll
            for (int c = 0; c < 3; ++c) t[k][c] = sh.ring[sl][c][xo];
          }
        }
        emit(xg, 2 * pp, 2 * pp + 1 < oh ? 2 : 1, 0, t);
      }
    }
    TF_PROF_MARK(6);
    p_next = p_rdy;
    base_slot += 2 * NW;
    if (base_slot >= kRing) base_slot -= kRing;
  }
#if WG_TF_PROF
  if (wl == 0) {
    u32* pr = reduced + ((int)wg_block_id() * NW + wave) * 12;
    for (int i = 0; i < 6; ++i) pr[i] = (u32)tf_acc[i];
    pr[6] = (u32)((r_need - r_first) / (2 * NW) + 1);
    pr[7] = 1u;
    for (int i = 0; i < 3; ++i) pr[8 + i] = (u32)tf_sub[i];
  }
#endif
#undef TF_PROF_MARK
#undef TF_PROF_SUB
}

}  // namespace

// Stage "post" + "filter" in one kernel (true fusion, see above). Arguments: as post_recon2_r_p12 without tile0 (A, C
// in the pack12 layout, B unused), then out (kH: RGBA16F rows of pitch_bytes with pi folded in; _f32: 3 x f32 rows of
// 12 ow bytes, pi and pitch_bytes unused), ow, oh, dec_scale (0.01 / e; from expo when not null), pi, pitch_bytes,
// cpairs (C: output row pairs per workgroup, even). Grid: ceil(ow / 40) * ceil(ceil(oh / 2) / C) workgroups of 64 NW
// lanes. Needs tf_supported. post_tf_p12 / post_tf_f32_p12: NW = 8; post_tf4_*: NW = 4 (256 lanes).
#define WG_TF_KERNEL_C(name, NW, H, CAP) WG_TF_KERNEL_CD(name, NW, H, CAP, (WG_POST_DIET != 0))
#define WG_TF_KERNEL_CD(name, NW, H, CAP, DIET) WG_TF_KERNEL_CDX(name, NW, H, CAP, DIET, false, false, false, false)
#define WG_TF_KERNEL_CDX(name, NW, H, CAP, DIET, VDIET, SEL, IDX, TRIM)                                                    \
  WG_KERNEL_NW(name, 64 * NW, WG_TF_WAVES)(const v4f* color, const u32* A, const u32* B, const v4h* C,             \
                                           const v4h* hist_prev, const u32* m_near, const u32* m_center,         \
                                           const u32* m_far, const float* w, v4h* hist_out, u32* reduced,        \
                                           void* out, WgPostParams p, const float* expo, int ow, int oh,         \
                                           float dec_scale, float pi, int pitch_bytes, int cpairs, int k1, int c2      \
                                           WG_LZ_KARGS) {                                                              \
    WG_LDS TfShared<NW, CAP> sh;                                                                                       \
    (void)B;                                                                                                      \
    post_tf_body<NW, H, CAP, DIET, VDIET, SEL, IDX, TRIM>(color, A, C, hist_prev, m_near, m_center, m_far, w, hist_out,    \
                                                    reduced, out, p, expo, ow, oh, dec_scale, pi, pitch_bytes, cpairs, \
                                                    k1, c2 WG_LZ_KW, sh);                                              \
  }
#define WG_TF_KERNEL(name, NW, H) WG_TF_KERNEL_C(name, NW, H, tf_cap<NW>())
WG_TF_KERNEL(post_tf_p12, 8, true)
WG_TF_KERNEL(post_tf_f32_p12, 8, false)
WG_TF_KERNEL_CD(post_tf_f32_p12_diet, 8, false, tf_cap<8>(), true)  // WG_POST_DIET cuts (runtime option post_diet=1)
WG_TF_KERNEL_CD(post_tf_p12_diet, 8, true, tf_cap<8>(), true)       // the same for the interop output copy (RGBA16F rows)
// The _diet kernels with the bit-identical improvements on top, one kernel per combination (runtime options tf_vdiet,
// post_sel, post_idx; all default 1): suffix letters v = kVDiet (vertical pass diet), s = kSel (candidate motions by
// selects), i = kIdx (unclamped, pre-scaled LDS texel offsets). The defaults are post_tf_f32_p12_diet_vsi (own output,
// 3 x f32) and post_tf_p12_diet_vsi (interop output copy, RGBA16F rows); all options 0 = the plain _diet kernels.
#define WG_TF_KERNEL_DIET_X(sfx, VDIET, SEL, IDX, TRIM)                                                                    \
  WG_TF_KERNEL_CDX(post_tf_f32_p12_diet_##sfx, 8, false, tf_cap<8>(), true, VDIET, SEL, IDX, TRIM)                          \
  WG_TF_KERNEL_CDX(post_tf_p12_diet_##sfx, 8, true, tf_cap<8>(), true, VDIET, SEL, IDX, TRIM)
WG_TF_KERNEL_DIET_X(vsi, true, true, true, false)
WG_TF_KERNEL_DIET_X(si, false, true, true, false)
WG_TF_KERNEL_DIET_X(vi, true, false, true, false)
WG_TF_KERNEL_DIET_X(vs, true, true, false, false)
WG_TF_KERNEL_DIET_X(v, true, false, false, false)
WG_TF_KERNEL_DIET_X(s, false, true, false, false)
WG_TF_KERNEL_DIET_X(i, false, false, true, false)
// post_trim (suffix t) on the default kernels only: in_range as unsigned bit compares, no canonicalise before the Dec
// clamp of the history, wave-uniform address products on the SALU (post_core_s / post_blend / post_tf_body kTrim).
WG_TF_KERNEL_DIET_X(vsit, true, true, true, true)
WG_TF_KERNEL(post_tf4_p12, 4, true)
WG_TF_KERNEL(post_tf16_p12, 16, true)
// Native AA (D = 1.5 R): footprint (floor(63 / 1.5) + 4) x (floor(14 / 1.5) + 5) = 46 x 14 = 644 texels, 2 per lane.
WG_TF_KERNEL_C(post_tf_dl_p12, 8, true, 644)
WG_TF_KERNEL_C(post_tf_dl_f32_p12, 8, false, 644)

// ---------------------------------------------------------------------------------------------------------------
// Fused reconstruction + output filter (post_fused*): the display image never goes to memory.
//
// A workgroup owns a strip of kFuOutX = 40 output columns (60 D columns, starting at X0 = 60 s) and kFuOutY = 96
// output rows (pairs P0 ... P0 + 47, D rows 3 P0 ... 3 P0 + 143). It computes the reconstruction for the 64 D columns
// X0 - 2 ... X0 + 61 (the filter needs 2 D columns of margin on each side) and walks down the strip in bands of 3 D
// row pairs. Band t (t >= 0) computes D rows 3 P0 + 2 + 6t ... 3 P0 + 7 + 6t and then outputs the pairs P0 + 2t and
// P0 + 2t + 1 (output pair p needs D rows 3p - 2 ... 3p + 4); a prologue band computes rows 3 P0 - 2 ... 3 P0 + 1.
// Only the 4 margin columns and 4 rows per strip are computed twice (about 10 %). Per band:
//   1. stage the render footprint of the band in LDS (features converted as for post_recon, plus the 3 motions);
//   2. one lane per (column, row pair): spec 06 steps 1-5 exactly as post_recon (same functions, same operation
//      order, so the results are bit-identical); owned pixels write the full and the reduced history; the display
//      values (FP16, spec 06) of all 64 columns go to an LDS band buffer;
//   3. horizontal filter pass of the band's 6 display rows into a 10-row ring of FP32 rows (40 output columns);
//   4. vertical pass, decode and output for the band's 2 output row pairs.
// Rows and columns outside D are the clamped edge pixels (computed, never stored), as the filter's clamp-to-edge
// needs. Lane layout: 6 waves = 2 column halves x 3 row pairs; within a wave, lanes l and l ^ 16 hold the D columns
// x and x + 1, so the 2 x 2 means of the reduced history use one lane exchange (wg_xor16) and no barrier.

namespace {

constexpr int kFuThreads = 192;
constexpr int kFuCols = 64;        // D columns computed per strip
constexpr int kFuOwnD = 60;        // D columns owned per strip
constexpr int kFuOutX = 40;        // output columns per strip
constexpr int kFuOutY = 96;        // output rows per workgroup
constexpr int kFuPairs = 3;        // row pairs per band
constexpr int kFuRing = 10;        // filtered rows kept for the vertical pass
constexpr int kFuStageCap = 336;   // render pixels staged per band (enough for D >= 1.5 R)

struct FuShared {
  PostTexelF tex[kFuStageCap];
  u32 mot[3][kFuStageCap];                       // m_near, m_center, m_far (2 x f16 each)
  _Float16 disp[2 * kFuPairs][3][kFuCols];       // display rows of the band
  float ring[kFuRing][3][kFuOutX];               // horizontally filtered rows
};

template <bool kHist6, bool kOutF16>
WG_FN void post_fused_body(const v4f* color, const u32* A, const u32* B, const v4h* C, const void* hist_prev,
                           const u32* m_near, const u32* m_center, const u32* m_far, const float* w, void* hist_out,
                           u32* reduced, void* out, WgPostParams p, int ow, int oh, float dec_scale,
                           const float* expo WG_LZ_PARAM, FuShared& sh) {
  if (expo) {  // exposure pair in GPU memory (runtime), as post_recon / post_filter
    p.e = expo[0] == 0.0f ? 1.0f : expo[0];
    p.e_prev = expo[1] == 0.0f ? 1.0f : expo[1];
    p.ratio = (float)((double)p.e / (double)p.e_prev);
    dec_scale = (float)(0.01 / (double)p.e);
  }
  const int lane = (int)wg_lane_id();
  const int strips = (ow + kFuOutX - 1) / kFuOutX;
  const int strip = (int)wg_block_id() % strips, chunk = (int)wg_block_id() / strips;
  const int X0 = strip * kFuOwnD, x0o = strip * kFuOutX;
  const int P0 = chunk * (kFuOutY / 2);
  const int own_r0 = 3 * P0, own_r1 = 3 * (P0 + kFuOutY / 2);
  const int ring0 = 3 * P0 - 2;  // D row of ring slot 0
  const float s = w[0];

  // This lane's column (lanes l, l ^ 16 = columns x, x + 1) and row pair within the band.
  const int wave = lane >> 5, wl = lane & 31;
  const int pr = wave >> 1;
  const int cl = 32 * (wave & 1) + 2 * (wl & 15) + (wl >> 4);
  const int xv = X0 - 2 + cl;
  const int x = clampi(xv, 0, p.dw - 1);
  const bool own_x = xv >= X0 && xv < X0 + kFuOwnD && xv < p.dw;
  const float px = pos_x(x, p);
  const int ax = floor_i(px - 0.5f);
  const int fx = clampi(floor_i(px), 0, p.rw - 1);
  // Footprint columns of the strip (a_x does not decrease with x).
  const int c0 = floor_i(pos_x(clampi(X0 - 2, 0, p.dw - 1), p) - 0.5f) - 1;
  const int c1 = floor_i(pos_x(clampi(X0 + kFuCols - 3, 0, p.dw - 1), p) - 0.5f) + 1;
  const int ncol = c1 - c0 + 1;

  for (int t = -1; t < kFuOutY / 4; ++t) {
    if (t >= 0 && 2 * (P0 + 2 * t) >= oh) break;          // no output rows left (uniform)
    const int rs = t < 0 ? 3 * P0 - 2 : 3 * P0 + 2 + 6 * t;  // first D row of the band (even)
    const int npr = t < 0 ? 2 : kFuPairs;

    // 1. Stage. Window rows of row pair k: b(k) - 1 ... b(k) + 1, plus b + 2 for the motion reads of the odd row.
    const int ye_first = clampi(rs, 0, p.dh - 1) & ~1, ye_last = clampi(rs + 2 * (npr - 1), 0, p.dh - 1) & ~1;
    const int r0 = floor_i(pos_y(ye_first, p) - 0.5f) - 1;
    const int r1 = floor_i(pos_y(ye_last, p) - 0.5f) + 2;
    const int nrow = r1 - r0 + 1;
    // The host launches the fused kernels only when every band footprint fits (fused_supported in
    // runner/post.py: D >= about 1.43 R per axis, true for every FSR mode since D = 1.5 O); otherwise the staging is
    // cut to the LDS size (wrong values, but no out-of-bounds access).
    const int nstage = ncol * nrow < kFuStageCap ? ncol * nrow : kFuStageCap;
    {
      for (int i = lane; i < nstage; i += kFuThreads) {
        const int r = i / ncol, c = i - r * ncol;
        const int idx = clampi(r0 + r, 0, p.rh - 1) * p.rw + clampi(c0 + c, 0, p.rw - 1);
        sh.tex[i] = texel_f(load_texel(color, A, B, C, idx, p.e));
        const u32 mo = (u32)idx * 4u;
        sh.mot[0][i] = *(const u32*)((const char*)m_near + mo);
        sh.mot[1][i] = *(const u32*)((const char*)m_center + mo);
        sh.mot[2][i] = *(const u32*)((const char*)m_far + mo);
      }
    }
    wg_barrier();

    // 2. Reconstruction of this lane's two pixels (virtual rows yv, yv + 1; clamped to D).
    if (pr < npr) {
      const auto sidx = [&](int c, int r) { return clampi((r - r0) * ncol + (c - c0), 0, kFuStageCap - 1); };
      const auto tex = [&](int c, int r) -> PostTexelF { return sh.tex[sidx(c, r)]; };
      const auto mot = [&](int k, int c, int r) -> u32 { return sh.mot[k][sidx(c, r)]; };
      const int yv = rs + 2 * pr;
      const int y0 = clampi(yv, 0, p.dh - 1), y1 = clampi(yv + 1, 0, p.dh - 1);
      const float py0 = pos_y(y0, p), py1 = pos_y(y1, p);
      const int b = floor_i(pos_y(y0 & ~1, p) - 0.5f);  // both rows belong to the same pair (see header)

      Gather g0, g1;
      gather_zero(g0);
      gather_zero(g1);
#pragma unroll
      for (int j = WG_AB_GATHER ? 0 : -1; j <= (WG_AB_GATHER ? 0 : 1); ++j) {
        const float v0 = ((float)(b + j) + 0.5f) - py0;
        const float v1 = ((float)(b + j) + 0.5f) - py1;
        const float vv0 = kTapK * v0 * v0, vv1 = kTapK * v1 * v1;
#pragma unroll
        for (int i = WG_AB_GATHER ? 0 : -1; i <= (WG_AB_GATHER ? 0 : 1); ++i) {
          const float u = ((float)(ax + i) + 0.5f) - px;
          const float uu = kTapK * u * u, uv0 = 2.0f * kTapK * u * v0, uv1 = 2.0f * kTapK * u * v1;
          const PostTexelF tt = tex(ax + i, b + j);
          gather_add_f(g0, tt, uu, vv0, uv0);
          gather_add_f(g1, tt, uu, vv1, uv1);
        }
      }

      float hsum[3];
#pragma nounroll  // one pixel at a time: unrolled, the two pixels' steps 3-5 interleave and double the registers
      for (int k = 0; k < 2; ++k) {
        const int y = k == 0 ? y0 : y1;
        const float py = k == 0 ? py0 : py1;
        const int fy = clampi(floor_i(py), 0, p.rh - 1);
        float mx[3], my[3];
        const int pxi = floor_i(px), pyi = floor_i(py);
        if (WG_AB_MOTION) {
          mx[0] = my[0] = 0.25f;
          mx[1] = my[1] = 0.5f;
          mx[2] = my[2] = -0.75f;
        } else {
          unpack_h2(mot(0, pxi, pyi), mx[0], my[0]);
          unpack_h2(mot(2, pxi, pyi), mx[2], my[2]);
          motion_bilinear(px, py, [&](int c, int r) { return mot(1, c, r); }, mx[1], my[1]);
        }
        float hh[3], ohv[3];
        post_core<kHist6>(k == 0 ? g0 : g1, x, y, fx, fy, mx, my, p, s, color, hist_prev, hh, ohv);
        const int yrow = yv + k;
        if (!WG_AB_HISTW && own_x && yrow >= own_r0 && yrow < own_r1 && yrow < p.dh)
          hist_store<kHist6>(hist_out, y * p.dw + x, enc_e(hh[0]), enc_e(hh[1]), enc_e(hh[2]));
#pragma unroll
        for (int c = 0; c < 3; ++c) {
          if (!WG_AB_DISPLAY) sh.disp[2 * pr + k][c][cl] = display_value(ohv[c]);
          hsum[c] = k == 0 ? hh[c] : hsum[c] + hh[c];
        }
      }
      // Reduced history: lane l < 16 holds the even column x, lane l ^ 16 the column x + 1 (same wave).
      float nb[3] = {0.0f, 0.0f, 0.0f};
      if (!WG_AB_REDUCED) {
#pragma unroll
        for (int c = 0; c < 3; ++c) nb[c] = wg_xor16(hsum[c]);
      }
      if (!WG_AB_REDUCED && wl < 16 && own_x && x + 1 < p.dw && yv >= own_r0 && yv + 1 < own_r1 && yv + 1 < p.dh) {
        u32 code = 0;
#pragma unroll
        for (int c = 0; c < 3; ++c) {
          const float m = (hsum[c] + nb[c]) * 0.25f;
          const float v = clampf(enc_e(m) * (1.0f / 2.45f), 0.0f, 1.0f);
          code |= (u32)wg_rint(v * 1023.0f) << (10 * c);
        }
        reduced[(yv / 2) * (p.dw / 2) + x / 2] = code;
      }
    }
    wg_barrier();

    // 3. Horizontal pass: item = (band row, output column pair m); input columns (strip-local) 3m ... 3m + 6.
    if (lane < 2 * npr * (kFuOutX / 2)) {
      const int row = lane / (kFuOutX / 2), m = lane - row * (kFuOutX / 2);
      const int slot = (rs + row - ring0) % kFuRing;
#pragma unroll
      for (int c = 0; c < 3; ++c) {
        float tv[7];
#pragma unroll
        for (int i = 0; i < 7; ++i) tv[i] = WG_LZ_TAP(sh.disp[row][c][3 * m + i]);
        sh.ring[slot][c][2 * m] = lz5<true>(tv WG_LZ_ARG);
        sh.ring[slot][c][2 * m + 1] = lz5<false>(tv + 2 WG_LZ_ARG);
      }
    }
    wg_barrier();

    // 4. Vertical pass and output: item = (output row of the band's 2 pairs, output column).
    if (t >= 0 && lane < 4 * kFuOutX) {
      const int q = lane / kFuOutX, xo = lane - q * kFuOutX;
      const int pp = P0 + 2 * t + (q >> 1), dy = q & 1;
      const int yo = 2 * pp + dy, xg = x0o + xo;
      if (yo < oh && xg < ow) {
        float v[3];
        bool any_small = false;
        float qc[3];
#pragma unroll
        for (int c = 0; c < 3; ++c) {
          float tv[5];
#pragma unroll
          for (int k = 0; k < 5; ++k) tv[k] = sh.ring[(3 * pp - 2 + 2 * dy + k - ring0) % kFuRing][c][xo];
          qc[c] = clampf(WG_LZ_VAL(dy == 0 ? lz5<true>(tv WG_LZ_ARG) : lz5<false>(tv WG_LZ_ARG)), 1e-4f, 3.0f);
          v[c] = WG_AB_FDEC ? qc[c] * 6.4f : wg_exp2(qc[c] * kDecK) - 1.0f;
          any_small |= !WG_AB_FDEC && qc[c] < kDecPolyMax;
        }
        if (any_small) {
#pragma unroll
          for (int c = 0; c < 3; ++c)
            if (qc[c] < kDecPolyMax) v[c] = dec_m1_poly(qc[c]);
        }
        const size_t o = (size_t)yo * ow + xg;
        if (kOutF16) {
          v4h t4;
          t4[0] = framebuffer_half(v[0] * dec_scale);
          t4[1] = framebuffer_half(v[1] * dec_scale);
          t4[2] = framebuffer_half(v[2] * dec_scale);
          t4[3] = (_Float16)1.0f;
          ((v4h*)out)[o] = t4;
        } else {
          float* of = (float*)out + 3 * o;
          of[0] = v[0] * dec_scale;
          of[1] = v[1] * dec_scale;
          of[2] = v[2] * dec_scale;
        }
      }
    }
  }
}

}  // namespace

// Fused stage post + filter. Inputs as post_recon (hist_prev in the kernel's history format); hist_out: new full
// history; reduced as post_recon; out: O, 3 x f32 (row-major RGB), or 4 x f16 (R, G, B, 1) for the _o16 kernels;
// ow, oh: output size; dec_scale = 0.01 / e; expo: exposure pair or null (as post_recon / post_filter).
// post_fused / post_fused_o16: 6-byte full history; post_fused_h8 / post_fused_h8_o16: 8-byte (RGBA16F, alpha 0) as
// post_recon. Grid: ceil(ow / 40) * ceil(oh / 96) workgroups of 192 lanes.
#define WG_FUSED_KERNEL(name, H6, O16)                                                                             \
  WG_KERNEL_N(name, kFuThreads)(const v4f* color, const u32* A, const u32* B, const v4h* C, const void* hist_prev, \
                                const u32* m_near, const u32* m_center, const u32* m_far, const float* w,         \
                                void* hist_out, u32* reduced, void* out, WgPostParams p, int ow, int oh,         \
                                float dec_scale, const float* expo WG_LZ_KARGS) {                                \
    WG_LDS FuShared sh;                                                                                           \
    post_fused_body<H6, O16>(color, A, B, C, hist_prev, m_near, m_center, m_far, w, hist_out, reduced, out, p,    \
                             ow, oh, dec_scale, expo WG_LZ_KW, sh);                                              \
  }
WG_FUSED_KERNEL(post_fused, true, false)
WG_FUSED_KERNEL(post_fused_o16, true, true)
WG_FUSED_KERNEL(post_fused_h8, false, false)
WG_FUSED_KERNEL(post_fused_h8_o16, false, true)

// ---------------------------------------------------------------------------------------------------------------
// post_fused3*: fused reconstruction + output filter built for latency. post_fused walks each strip in bands with a
// chain of dependent waits per band (staging loads, then two history round trips per pixel, then three barriers) and
// only a few workgroups per WGP, so it is latency-bound (measured 3x its memory-traffic floor). post_fused3 keeps its
// geometry (strips of 40 output columns, 64 D columns, bands of 3 row pairs, 192 lanes, a 10-row ring for the
// vertical pass) and changes the schedule:
//   - the next band's render footprint is loaded into registers at the start of the current band and written to LDS
//     after its reconstruction (the loads' latency hides behind the band's work); two barriers per band instead of
//     three;
//   - the history loads of a pixel's two candidates are issued together (post_core_x, as post_recon3): one memory
//     round trip per pixel instead of two;
//   - the fallback colour is read from global memory only by pixels that fall back (not staged);
//   - history 6 bytes per pixel (RGB16F, exact) or 4 (h4_pack, an approximation); output f32 or RGBA16F (alpha 1,
//     round to nearest even).
// Template switches: kG gather (0 FP32 exact, 2 packed FP16 as post_recon4_f16x), kHist (0 Catmull-Rom, 1 bilinear),
// kSkip (as post_recon4_skip*), kFmt (6 or 4), kO16. Arguments and grid as post_fused.
//   post_fused3_h6, post_fused3_h6_o16: exact (bit-identical to post_recon + post_filter; _o16: the FP16 cast of it);
//   post_fused3_fast_h6_o16: post_recon4_fastest arithmetic (packed FP16 gather, bilinear history, skip4);
//   post_fused3_h4_o16, post_fused3_fast_h4_o16: the same two with the compact 4-byte history.

namespace {

// Raw inputs of one render pixel, loaded ahead and converted when written to LDS (as load_texel + texel_f).
struct StageRaw {
  v4f c;
  v4h k;
  u32 a, b, m0, m1, m2;
};
WG_FN void stage_raw(const v4f* color, const u32* A, const u32* B, const v4h* C, const u32* m_near,
                     const u32* m_center, const u32* m_far, u32 idx, StageRaw& r) {
  r.c = *(const v4f*)((const char*)color + idx * 16u);
  r.k = *(const v4h*)((const char*)C + idx * 8u);
  r.a = *(const u32*)((const char*)A + idx * 4u);
  r.b = *(const u32*)((const char*)B + idx * 4u);
  r.m0 = *(const u32*)((const char*)m_near + idx * 4u);
  r.m1 = *(const u32*)((const char*)m_center + idx * 4u);
  r.m2 = *(const u32*)((const char*)m_far + idx * 4u);
}
WG_FN PostTexelF texel_of_raw(const StageRaw& r, float e) {
  PostTexel t;
  t.ec0 = enc_in(e * r.c[0]);
  t.ec1 = enc_in(e * r.c[1]);
  t.ec2 = enc_in(e * r.c[2]);
  t.a = r.a;
  t.ca = (float)r.k[0];
  t.cb = (float)r.k[1];
  t.cc = (float)r.k[2];
  t.b = r.b;
  return texel_f(t);
}

constexpr int kFu3Slots = (kFuStageCap + kFuThreads - 1) / kFuThreads;  // staged pixels per lane (2)

template <int kG>
struct Fu3Shared {
  PostTexelF tex[kG == 0 ? kFuStageCap : 1];
  Tex16 t16[kG == 2 ? kFuStageCap : 1];
  u32 mot[3][kFuStageCap];
  _Float16 disp[2 * kFuPairs][3][kFuCols];
  float ring[kFuRing][3][kFuOutX];
};

template <int kG, int kHist, int kSkip, int kFmt, bool kO16>
WG_FN void post_fused3_body(const v4f* color, const u32* A, const u32* B, const v4h* C, const void* hist_prev,
                            const u32* m_near, const u32* m_center, const u32* m_far, const float* w, void* hist_out,
                            u32* reduced, void* out, WgPostParams p, int ow, int oh, float dec_scale,
                            const float* expo WG_LZ_PARAM, Fu3Shared<kG>& sh) {
  if (expo) {
    p.e = expo[0] == 0.0f ? 1.0f : expo[0];
    p.e_prev = expo[1] == 0.0f ? 1.0f : expo[1];
    p.ratio = (float)((double)p.e / (double)p.e_prev);
    dec_scale = (float)(0.01 / (double)p.e);
  }
  const int lane = (int)wg_lane_id();
  const int strips = (ow + kFuOutX - 1) / kFuOutX;
  const int strip = (int)wg_block_id() % strips, chunk = (int)wg_block_id() / strips;
  const int X0 = strip * kFuOwnD, x0o = strip * kFuOutX;
  const int P0 = chunk * (kFuOutY / 2);
  const int own_r0 = 3 * P0, own_r1 = 3 * (P0 + kFuOutY / 2);
  const int ring0 = 3 * P0 - 2;
  const float s = w[0];

  const int wave = lane >> 5, wl = lane & 31;
  const int pr = wave >> 1;
  const int cl = 32 * (wave & 1) + 2 * (wl & 15) + (wl >> 4);
  const int xv = X0 - 2 + cl;
  const int x = clampi(xv, 0, p.dw - 1);
  const bool own_x = xv >= X0 && xv < X0 + kFuOwnD && xv < p.dw;
  const float px = pos_x(x, p);
  const int ax = floor_i(px - 0.5f);
  const int c0 = floor_i(pos_x(clampi(X0 - 2, 0, p.dw - 1), p) - 0.5f) - 1;
  const int c1 = floor_i(pos_x(clampi(X0 + kFuCols - 3, 0, p.dw - 1), p) - 0.5f) + 1;
  const int ncol = c1 - c0 + 1;

  // Band t (t = -1: prologue): first D row, row pairs, first staged render row, staged pixel count (as post_fused).
  const auto band_rs = [&](int t) { return t < 0 ? 3 * P0 - 2 : 3 * P0 + 2 + 6 * t; };
  const auto band_npr = [&](int t) { return t < 0 ? 2 : kFuPairs; };
  const auto band_r0 = [&](int t) { return floor_i(pos_y(clampi(band_rs(t), 0, p.dh - 1) & ~1, p) - 0.5f) - 1; };
  const auto band_nstage = [&](int t) {
    const int rs = band_rs(t), npr = band_npr(t);
    const int ye_last = clampi(rs + 2 * (npr - 1), 0, p.dh - 1) & ~1;
    const int nrow = floor_i(pos_y(ye_last, p) - 0.5f) + 2 - band_r0(t) + 1;
    return ncol * nrow < kFuStageCap ? ncol * nrow : kFuStageCap;
  };
  const auto band_live = [&](int t) { return t < kFuOutY / 4 && (t < 0 || 2 * (P0 + 2 * t) < oh); };

  // Slot 0 (staged pixel = lane) is prefetched into registers; slot 1 (pixels 192 ... nstage - 1, only when the band
  // footprint exceeds 192 render pixels, i.e. D < about 2.1 R) is loaded when it is written.
  StageRaw raw0;
  const auto stage_idx = [&](int t, int i) {
    const int r = i / ncol, c = i - r * ncol;
    return mul24((u32)clampi(band_r0(t) + r, 0, p.rh - 1), (u32)p.rw) + (u32)clampi(c0 + c, 0, p.rw - 1);
  };
  const auto stage_put = [&](int i, const StageRaw& r) {
    const PostTexelF tf = texel_of_raw(r, p.e);
    if (kG == 0) sh.tex[i] = tf;
    else sh.t16[i] = tex16_of(tf);
    sh.mot[0][i] = r.m0;
    sh.mot[1][i] = r.m1;
    sh.mot[2][i] = r.m2;
  };
  const auto stage_issue = [&](int t) {
    if (lane < band_nstage(t)) stage_raw(color, A, B, C, m_near, m_center, m_far, stage_idx(t, lane), raw0);
  };
  const auto stage_commit = [&](int t) {
    const int n = band_nstage(t);
    if (lane < n) stage_put(lane, raw0);
#pragma unroll
    for (int q = 1; q < kFu3Slots; ++q) {
      const int i = lane + q * kFuThreads;
      if (i < n) {
        StageRaw r;
        stage_raw(color, A, B, C, m_near, m_center, m_far, stage_idx(t, i), r);
        stage_put(i, r);
      }
    }
  };

  stage_issue(-1);
  stage_commit(-1);
  wg_barrier();
  for (int t = -1; band_live(t); ++t) {
    const int rs = band_rs(t), npr = band_npr(t), r0 = band_r0(t);
    const bool more = band_live(t + 1);
    if (more) stage_issue(t + 1);  // into registers; written to LDS after this band's reconstruction

    // Reconstruction of this lane's two pixels.
    if (pr < npr) {
      const auto sidx = [&](int c, int r) { return clampi(mul_i24(r - r0, ncol) + (c - c0), 0, kFuStageCap - 1); };
      const int yv = rs + 2 * pr;
      const int y0 = clampi(yv, 0, p.dh - 1), y1 = clampi(yv + 1, 0, p.dh - 1);
      const float py0 = pos_y(y0, p), py1 = pos_y(y1, p);
      const int b = floor_i(pos_y(y0 & ~1, p) - 0.5f);

      Gather g0, g1;
      gather_zero(g0);
      gather_zero(g1);
      GatherH gh;
      gatherh_zero(gh);
#pragma unroll
      for (int j = -1; j <= 1; ++j) {
        const float v0 = ((float)(b + j) + 0.5f) - py0;
        const float v1 = ((float)(b + j) + 0.5f) - py1;
        const float vv0 = kTapK * v0 * v0, vv1 = kTapK * v1 * v1;
#pragma unroll
        for (int i = -1; i <= 1; ++i) {
          const float u = ((float)(ax + i) + 0.5f) - px;
          const float uu = kTapK * u * u, uv0 = 2.0f * kTapK * u * v0, uv1 = 2.0f * kTapK * u * v1;
          const int si = sidx(ax + i, b + j);
          if (kG == 0) {
            const PostTexelF tt = sh.tex[si];
            gather_add_f(g0, tt, uu, vv0, uv0);
            gather_add_f(g1, tt, uu, vv1, uv1);
          } else {
            gather_g2(gh, sh.t16[si], bc((_Float16)uu), v2h{(_Float16)vv0, (_Float16)vv1},
                      v2h{(_Float16)uv0, (_Float16)uv1});
          }
        }
      }
      if (kG == 2) {
        Gather* gs[2] = {&g0, &g1};
#pragma unroll
        for (int r = 0; r < 2; ++r) {
          Gather& g = *gs[r];
          g.W = (float)gh.W[r];
#pragma unroll
          for (int i = 0; i < 3; ++i) {
            g.c[i] = (float)gh.c[i][r];
            g.s[i] = (float)gh.s[i][r] * 255.0f;
            g.a[i] = (float)gh.a[i][r] * 255.0f;
          }
        }
      }

      const int pxi = floor_i(px), fx = clampi(pxi, 0, p.rw - 1);
      float hsum[3];
#pragma nounroll  // one pixel at a time (unrolled, the two pixels' steps 3-5 interleave and double the registers)
      for (int k = 0; k < 2; ++k) {
        const int y = k == 0 ? y0 : y1;
        const float py = k == 0 ? py0 : py1;
        const int pyi = floor_i(py);
        float mx[3], my[3];
        unpack_h2(sh.mot[0][sidx(pxi, pyi)], mx[0], my[0]);
        unpack_h2(sh.mot[2][sidx(pxi, pyi)], mx[2], my[2]);
        motion_bilinear(px, py, [&](int c, int r) { return sh.mot[1][sidx(c, r)]; }, mx[1], my[1]);
        const u32 fi = mul24((u32)clampi(pyi, 0, p.rh - 1), (u32)p.rw) + (u32)fx;
        const GlobalCol colv{(const float*)((const char*)color + fi * 16u)};  // read only by a fallback
        float hh[3], ohv[3];
        post_core_x<(kG > 0), kHist, kSkip, false, kFmt>(k == 0 ? g0 : g1, x, y, mx, my, colv, p, s, hist_prev, hh,
                                                         ohv);
        const int yrow = yv + k;
        if (own_x && yrow >= own_r0 && yrow < own_r1 && yrow < p.dh) {
          const u32 hi = mul24((u32)y, (u32)p.dw) + (u32)x;
          if (kFmt == 4) *(u32*)((char*)hist_out + hi * 4u) = h4_pack(enc_e(hh[0]), enc_e(hh[1]), enc_e(hh[2]));
          else hist_store<true>(hist_out, (int)hi, enc_e(hh[0]), enc_e(hh[1]), enc_e(hh[2]));
        }
#pragma unroll
        for (int c = 0; c < 3; ++c) {
          sh.disp[2 * pr + k][c][cl] = display_value(ohv[c]);
          hsum[c] = k == 0 ? hh[c] : hsum[c] + hh[c];
        }
      }
      float nb[3];
#pragma unroll
      for (int c = 0; c < 3; ++c) nb[c] = wg_xor16(hsum[c]);
      if (wl < 16 && own_x && x + 1 < p.dw && yv >= own_r0 && yv + 1 < own_r1 && yv + 1 < p.dh) {
        u32 code = 0;
#pragma unroll
        for (int c = 0; c < 3; ++c) {
          const float m = (hsum[c] + nb[c]) * 0.25f;
          const float v = clampf(enc_e(m) * (1.0f / 2.45f), 0.0f, 1.0f);
          code |= (u32)wg_rint(v * 1023.0f) << (10 * c);
        }
        reduced[mul24((u32)(yv / 2), (u32)(p.dw / 2)) + (u32)(x / 2)] = code;
      }
    }
    wg_barrier();  // display rows complete; the staged footprint is free

    // Horizontal pass (as post_fused) and the next band's staging.
    if (lane < 2 * npr * (kFuOutX / 2)) {
      const int row = lane / (kFuOutX / 2), m = lane - row * (kFuOutX / 2);
      const int slot = (rs + row - ring0) % kFuRing;
#pragma unroll
      for (int c = 0; c < 3; ++c) {
        float tv[7];
#pragma unroll
        for (int i = 0; i < 7; ++i) tv[i] = WG_LZ_TAP(sh.disp[row][c][3 * m + i]);
        sh.ring[slot][c][2 * m] = lz5<true>(tv WG_LZ_ARG);
        sh.ring[slot][c][2 * m + 1] = lz5<false>(tv + 2 WG_LZ_ARG);
      }
    }
    if (more) stage_commit(t + 1);
    wg_barrier();  // ring rows and the next band's footprint complete

    // Vertical pass, decode and output (as post_fused).
    if (t >= 0 && lane < 4 * kFuOutX) {
      const int q = lane / kFuOutX, xo = lane - q * kFuOutX;
      const int pp = P0 + 2 * t + (q >> 1), dy = q & 1;
      const int yo = 2 * pp + dy, xg = x0o + xo;
      if (yo < oh && xg < ow) {
        float v[3];
        bool any_small = false;
        float qc[3];
#pragma unroll
        for (int c = 0; c < 3; ++c) {
          float tv[5];
#pragma unroll
          for (int k = 0; k < 5; ++k) tv[k] = sh.ring[(3 * pp - 2 + 2 * dy + k - ring0) % kFuRing][c][xo];
          qc[c] = clampf(WG_LZ_VAL(dy == 0 ? lz5<true>(tv WG_LZ_ARG) : lz5<false>(tv WG_LZ_ARG)), 1e-4f, 3.0f);
          v[c] = wg_exp2(qc[c] * kDecK) - 1.0f;
          any_small |= qc[c] < kDecPolyMax;
        }
        if (any_small) {
#pragma unroll
          for (int c = 0; c < 3; ++c)
            if (qc[c] < kDecPolyMax) v[c] = dec_m1_poly(qc[c]);
        }
        const u32 o = mul24((u32)yo, (u32)ow) + (u32)xg;
        if (kO16) {
          v4h t4;
          t4[0] = framebuffer_half(v[0] * dec_scale);
          t4[1] = framebuffer_half(v[1] * dec_scale);
          t4[2] = framebuffer_half(v[2] * dec_scale);
          t4[3] = (_Float16)1.0f;
          *(v4h*)((char*)out + o * 8u) = t4;
        } else {
          float* of = (float*)((char*)out + o * 12u);
          of[0] = v[0] * dec_scale;
          of[1] = v[1] * dec_scale;
          of[2] = v[2] * dec_scale;
        }
      }
    }
  }
}

}  // namespace

#define WG_FUSED3_KERNEL(name, G, HIST, SKIP, FMT, O16)                                                            \
  WG_KERNEL_N(name, kFuThreads)(const v4f* color, const u32* A, const u32* B, const v4h* C, const void* hist_prev, \
                                const u32* m_near, const u32* m_center, const u32* m_far, const float* w,         \
                                void* hist_out, u32* reduced, void* out, WgPostParams p, int ow, int oh,         \
                                float dec_scale, const float* expo WG_LZ_KARGS) {                                \
    WG_LDS Fu3Shared<G> sh;                                                                                       \
    if (p.dw < 4) return; /* the host checks (fused_supported) */                                                 \
    post_fused3_body<G, HIST, SKIP, FMT, O16>(color, A, B, C, hist_prev, m_near, m_center, m_far, w,            \
                                              hist_out, reduced, out, p, ow, oh, dec_scale, expo WG_LZ_KW, sh);  \
  }
WG_FUSED3_KERNEL(post_fused3_h6, 0, 0, 0, 6, false)
WG_FUSED3_KERNEL(post_fused3_h6_o16, 0, 0, 0, 6, true)
WG_FUSED3_KERNEL(post_fused3_fast_h6_o16, 2, 1, 4, 6, true)
WG_FUSED3_KERNEL(post_fused3_h4_o16, 0, 0, 0, 4, true)
WG_FUSED3_KERNEL(post_fused3_fast_h4_o16, 2, 1, 4, 4, true)


// Experimental (runtime option recon=output, D = O): the output filter replaced by the identity: the display image's
// decoded colour, Dec(q) * dec_scale per channel (the filter's decode and exposure handling without the 5-tap
// resampling and clamp). display: three FP16 planes of O, row pitch display_pitch(ow); out: O, 3 x f32.
// Grid ceil(ow * oh / 256) workgroups of 256 lanes.
WG_KERNEL_N(post_copy, 256)(const _Float16* display, float* out, int ow, int oh, float dec_scale, const float* expo) {
  if (expo) dec_scale = (float)(0.01 / (double)(expo[0] == 0.0f ? 1.0f : expo[0]));
  const int i = (int)(wg_block_id() * 256 + wg_lane_id());
  if (i >= ow * oh) return;
  const int y = i / ow, x = i - y * ow;
  const int pitch = display_pitch(ow), plane = pitch * oh, di = y * pitch + x;
#pragma unroll
  for (int c = 0; c < 3; ++c) out[3 * i + c] = dec_m1((float)display[c * plane + di]) * dec_scale;
}

// post_copy writing RGBA16F rows (option output_copy=1 with recon=output), pi folded in.
WG_KERNEL_N(post_copy_h, 256)(const _Float16* display, _Float16* out, int ow, int oh, float dec_scale,
                              const float* expo, float pi, int pitch_bytes) {
  if (expo) dec_scale = (float)(0.01 / (double)(expo[0] == 0.0f ? 1.0f : expo[0]));
  const int i = (int)(wg_block_id() * 256 + wg_lane_id());
  if (i >= ow * oh) return;
  const int y = i / ow, x = i - y * ow;
  const int pitch = display_pitch(ow), plane = pitch * oh, di = y * pitch + x;
  _Float16* o = out + (size_t)y * (pitch_bytes / 2) + 4 * x;
#pragma unroll
  for (int c = 0; c < 3; ++c) o[c] = framebuffer_half(dec_m1((float)display[c * plane + di]) * dec_scale * pi);
  o[3] = (_Float16)1.0f;
}

#if defined(WG_EMU)
WG_EMU_REGISTER(post_copy_h)
WG_EMU_REGISTER(post_filter_h)
WG_EMU_REGISTER(post_copy)
WG_EMU_REGISTER(post_recon)
WG_EMU_REGISTER(post_recon2)
WG_EMU_REGISTER(post_recon2_r)
WG_EMU_REGISTER(post_recon2_r_h6)
WG_EMU_REGISTER(post_recon2_r_hp)
WG_EMU_REGISTER(post_recon2_r_hp_band)
WG_EMU_REGISTER(post_recon2_r_p12)
WG_EMU_REGISTER(post_recon2_r_hp_p12)
WG_EMU_REGISTER(post_recon2_r_band_p12)
WG_EMU_REGISTER(post_recon2_r_hp_band_p12)
WG_EMU_REGISTER(post_recon2_r_out)
WG_EMU_REGISTER(post_recon2_r_out_diet)
WG_EMU_REGISTER(post_decode_o16)
WG_EMU_REGISTER(post_recon2_r_band)
WG_EMU_REGISTER(post_filter_c1_s4_band)
WG_EMU_REGISTER(post_filter_c1_s4_h_band)
WG_EMU_REGISTER(post_filter_c1_s2_band)
WG_EMU_REGISTER(post_tf_p12)
WG_EMU_REGISTER(post_tf_f32_p12)
WG_EMU_REGISTER(post_tf_f32_p12_diet)
WG_EMU_REGISTER(post_tf_f32_p12_diet_vsi)
WG_EMU_REGISTER(post_tf_f32_p12_diet_vsit)
WG_EMU_REGISTER(post_tf_f32_p12_diet_si)
WG_EMU_REGISTER(post_tf_f32_p12_diet_vi)
WG_EMU_REGISTER(post_tf_f32_p12_diet_vs)
WG_EMU_REGISTER(post_tf_f32_p12_diet_v)
WG_EMU_REGISTER(post_tf_f32_p12_diet_s)
WG_EMU_REGISTER(post_tf_f32_p12_diet_i)
WG_EMU_REGISTER(post_tf4_p12)
WG_EMU_REGISTER(post_tf16_p12)
WG_EMU_REGISTER(post_tf_dl_p12)
WG_EMU_REGISTER(post_tf_dl_f32_p12)
WG_EMU_REGISTER(post_recon2_ns)
WG_EMU_REGISTER(post_recon2_r_ns)
WG_EMU_REGISTER(post_recon3)
WG_EMU_REGISTER(post_recon3_h)
WG_EMU_REGISTER(post_recon4_bil)
WG_EMU_REGISTER(post_recon4_crbil)
WG_EMU_REGISTER(post_recon4_skip16)
WG_EMU_REGISTER(post_recon4_skip4)
WG_EMU_REGISTER(post_recon4_f16)
WG_EMU_REGISTER(post_recon4_f16x)
WG_EMU_REGISTER(post_recon4_fast)
WG_EMU_REGISTER(post_recon4_fastest)
WG_EMU_REGISTER(post_prep)
WG_EMU_REGISTER(post_recon5)
WG_EMU_REGISTER(post_recon5_bil)
WG_EMU_REGISTER(post_recon6)
WG_EMU_REGISTER(post_recon6_bil)
WG_EMU_REGISTER(post_recon6_eb)
WG_EMU_REGISTER(post_recon6_eb_bil)
WG_EMU_REGISTER(post_traffic_r_h8)
WG_EMU_REGISTER(post_traffic_r_h6)
WG_EMU_REGISTER(post_traffic_r_h4)
WG_EMU_REGISTER(post_traffic_x_h8_o12)
WG_EMU_REGISTER(post_traffic_x_h6_o12)
WG_EMU_REGISTER(post_traffic_x_h4_o12)
WG_EMU_REGISTER(post_traffic_x_h6_o8)
WG_EMU_REGISTER(post_traffic_x_h4_o8)
WG_EMU_REGISTER(post_traffic_f_o12)
WG_EMU_REGISTER(post_traffic_f_o8)
WG_EMU_REGISTER(post_traffic_flush)
WG_EMU_REGISTER(post_filter)
WG_EMU_REGISTER(post_filter_quality9)
WG_EMU_REGISTER(post_filter_quality9_o16)
WG_EMU_REGISTER(post_filter_quality9_h)
WG_EMU_REGISTER(post_filter_s4)
WG_EMU_REGISTER(post_filter_s16)
WG_EMU_REGISTER(post_filter_c1)
WG_EMU_REGISTER(post_filter_c1_s4_o16)
WG_EMU_REGISTER(post_filter_c1_s4)
WG_EMU_REGISTER(post_filter_c1_s16)
WG_EMU_REGISTER(post_fused)
WG_EMU_REGISTER(post_fused_o16)
WG_EMU_REGISTER(post_fused_h8)
WG_EMU_REGISTER(post_fused_h8_o16)
WG_EMU_REGISTER(post_fused3_h6)
WG_EMU_REGISTER(post_fused3_h6_o16)
WG_EMU_REGISTER(post_fused3_fast_h6_o16)
WG_EMU_REGISTER(post_fused3_h4_o16)
WG_EMU_REGISTER(post_fused3_fast_h4_o16)
#endif
