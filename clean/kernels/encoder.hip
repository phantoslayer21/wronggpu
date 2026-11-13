// SPDX-License-Identifier: MIT
// Input encoder of spec/03-input-encoder.md (stage enc0) for gfx1201 (RDNA4).
//
// One fused kernel: per-pixel values (spec 03 "The 20 values per pixel"), dithered FP8 quantization, and the 64 x 80
// projection to the enc1 input tokens on FP8 WMMA. It also writes the three motion surfaces (m_near = next
// frame's motion history, m_far, m_center). Data and weight layouts: clean/FORMATS.md, "Input encoder (enc0)";
// packing: clean/runner/enc0.py. Kernel interface, weight blob and launch geometry are those of the first version
// (clean/kernels/baseline/encoder_v1.hip); this version does the same work with about half the instructions
// (performance milestone, see clean/PERF.md).
//
// Work split: a wave handles 16 consecutive output tokens (token t of the wave = WMMA column t, lanes t and t + 16).
// Lane (t, h) computes the two pixels of row dy = h of its token's 2 x 2 block, i.e. sub-pixels p = 2h and 2h + 1.
// Projection: sub-pixel p's 20 FP8 values form K group p, which takes two 16-deep WMMA K blocks:
//   block 2p:     slots of the owner half o = p / 2 hold values 0..7, slots of the other half values 8..15;
//   block 2p + 1: owner slots 0..3 hold values 16..19; all other slots are zero weight (padding).
// So each lane keeps 12 bytes of each of its own pixels and swaps values 8..15 with its partner lane (lane ^ 16):
// four lane exchanges per lane. The FP32 accumulator is rounded to FP16 after each sub-pixel's two blocks, as in
// the spec's reference ("rounds the running sum to FP16 after each sub-pixel's 20-value group").
//
// Per-pixel work of a lane: its two padded-grid pixels map (by reflection) to two neighbouring render columns
// xa, xa + 1 of one row (in the right padding in reverse order). They share one 3 x 4 depth window (three 16-byte
// loads; near / far search for "pixel A" at xa and "pixel B" at xa + 1) and one 16-byte load of their centre
// motions; then the two sub-pixel slots are computed one after the other.
//
// Bilinear samples (reduced history, motion history, latent) read two 64-bit texel pairs (wg_mem.h, wg_footprint)
// and fold the per-texel scale factors into the weights. Samples at invalid positions are still read (at a clamped
// position) and then replaced by the spec's fallback with a select, so that no lane waits on a branch and all loads
// of a pixel can be in flight together; on reset frames no history, motion history or latent is read at all.
//
// Arithmetic differs from the first version only in FP32 rounding order (the reference is FP64); the dithered
// quantization is the exact integer form of spec 03 draft 2 (dither_q below). Compile-time switches for timing
// experiments: WG_ENC_SEQ, WG_ENC_PIN, WG_ENC_FRAG_GLOBAL and the WG_ENC_AB_* ablations
// (clean/runner/perf_variants.py).
#include "wg_platform.h"
#include "wg_mem.h"

// No implicit multiply-add fusion: the rounding of every step is fixed by the source (explicit fmaf where wanted),
// so every build variant and compiler schedule gives the same bits.
#pragma clang fp contract(off)

#ifndef WG_ENC_SEQ
#define WG_ENC_SEQ 0  // 1: scheduling barrier between the two pixels of a lane (fewer registers, less overlap)
#endif
#ifndef WG_ENC_PIN
#define WG_ENC_PIN 1  // 1: all per-pixel results are computed before the projection starts (see wg_pin below)
#endif
#ifndef WG_ENC_FRAG_GLOBAL
// 1: WMMA A fragments straight from global memory (cached; no LDS staging, no barrier). Default since the spec
// team's timing of 1668645 (0.236 vs 0.253 ms, identical outputs); 0 = variant "lds".
#define WG_ENC_FRAG_GLOBAL 1
#endif
#ifndef WG_ENC_WAVES
#define WG_ENC_WAVES 0  // > 0: ask the register allocator for at least this many waves per SIMD (amdgpu_waves_per_eu)
#endif
#ifndef WG_ENC_COLOR_F16
#define WG_ENC_COLOR_F16 0  // 1: colour buffer as 4 x FP16 per pixel (8 bytes) instead of 4 x f32 (runner: "colorh")
#endif
#ifndef WG_ENC_MOTION_F16
#define WG_ENC_MOTION_F16 0  // 1: motion-vector buffer as 2 x FP16 per pixel (4 bytes) instead of 2 x f32 ("motionh")
#endif
#ifndef WG_ENC_NT_STORE
#define WG_ENC_NT_STORE 0  // 1: non-temporal stores for the motion surfaces (read again only by later stages)
#endif
// Timing-only ablations (the output is wrong on purpose): 1 removes that part of the work.
#ifndef WG_ENC_AB_HIST
#define WG_ENC_AB_HIST 0   // no reduced-history samples (history values = current colour)
#endif
#ifndef WG_ENC_AB_LAT
#define WG_ENC_AB_LAT 0    // no latent sample
#endif
#ifndef WG_ENC_AB_DIS
#define WG_ENC_AB_DIS 0    // no motion-history sample / disagreement value
#endif
#ifndef WG_ENC_AB_DEPTH
#define WG_ENC_AB_DEPTH 0  // no depth neighbourhood (near / far = fixed neighbours)
#endif
#ifndef WG_ENC_AB_PROJ
#define WG_ENC_AB_PROJ 0   // no projection (features = quantized values)
#endif
#ifndef WG_ENC_AB_QUANT
#define WG_ENC_AB_QUANT 0  // no dither (plain FP8 conversion)
#endif
#ifndef WG_ENC_AB_STORE
#define WG_ENC_AB_STORE 0  // no motion-surface stores
#endif

// Per-frame constants, passed by value (all 4-byte fields; must match EncParams in clean/runner/enc0.py).
struct WgEncParams {
  int rw, rh;          // render size R
  int fw, fh;          // padded size F
  int hw, hh;          // reduced-history texture size (D / 2)
  int frame;           // frame counter n, 0..7
  int reset;           // 1 on reset
  int depth_inverted;  // 1: near = largest depth
  int pad0;
  float dw, dh;        // reconstruction size D
  float kx, ky;        // D / R
  float ikx, iky;      // R / D
  float sx, sy;        // mv_scale * D / R
  float bx, by;        // mv_bias
  float jx, jy;        // jitter (render pixels)
  float jpx, jpy;      // previous jitter
  float dis_scale;     // 540 / rh
  float idw, idh;      // 1 / D
  float irw, irh;      // 1 / R
};
static_assert(sizeof(WgEncParams) == 116, "WgEncParams must match EncParams in clean/runner/enc0.py");

namespace {

constexpr int kEncThreads = 256;  // 8 waves, 128 tokens per workgroup
constexpr int kTokensPerWave = 16;
constexpr int kFeat = 64, kMT = kFeat / 16, kKB = 8;

// Weight blob offsets (bytes); see FORMATS.md.
constexpr int kOffBias = 0;      // 64 x f32, tile ordered (R_L(64))
constexpr int kOffSigma = 256;   // sigma1, sigma2 (f32)
constexpr int kOffDither = 512;  // T: u8 [8][16][16]
constexpr int kOffFrags = 2560;  // W: fragments [mt < 4][kb < 8], 256 B each
constexpr int kFragBytes = kMT * kKB * 256;

constexpr float kLn2 = 0.69314718055994531f;
constexpr float kL2 = 9.22757804135378390f;   // log2(10) / 0.36: 10^(q / 0.36) = 2^(q kL2)
constexpr float kG = 0.108370798438993f;       // 0.36 log10(2): 0.36 log10(a) = kG log2(a)

WG_FN int reflect(int p, int n) {
  p = p < 0 ? -p : p;
  p = p >= n ? 2 * n - p - 2 : p;
  return wg_med3i(p, 0, n - 1);
}

// Spec 03 draft 2 "Dithered FP8 quantization", steps 1-4 and the truncation of step 5, on the FP32 bit pattern b of
// v. For a normal FP16 h = RN16(v) = 2^E (1 + m / 1024), steps 2-4 give d = 2^E (1 + k / 512) with
// k = RN((m + t) / 2) (ties to even) and the truncation keeps floor(k / 64) eighths, which equals
// floor((m + t + 1) / 128): the result is h with t + 1 added to its 10-bit mantissa (carry into the exponent) and
// the low 7 mantissa bits cleared. In FP32 bits: RN16 is "add 0xfff plus bit 13, clear bits 0..12"; the dither adds
// (t + 1) << 13 and the truncation clears bits 0..19, and the first clear can be skipped because the second covers
// it. With tt = 0xfff + ((t + 1) << 13):  result = (b + bit13(b) + tt) & 0xfff00000.
// FP16 zero and subnormal h (|v| < 2^-14) end up below 2^-13 either way and become 0 in E4M3, like the spec's d.
// v is clamped to +-447.75 first: every value at or beyond that saturates to +-448 in the spec (its h is at least
// 447.75 and the dither never lowers the truncated value below 448), and the clamped value gives exactly 448, so the
// FP8 conversion needs no saturation. Exhaustively checked against the step-by-step definition
// (clean/tests/test_encoder_dither.py). The caller guarantees v is not NaN.
WG_FN float dither_q(float v, u32 tt) {
  const u32 b = __builtin_bit_cast(u32, wg_med3f(v, -447.75f, 447.75f));
  return __builtin_bit_cast(float, (b + ((b >> 13) & 1u) + tt) & 0xfff00000u);
}

// Four FP8 E4M3 codes (round to nearest even; inputs already within +-448) packed little-endian.
WG_FN u32 pack4_fp8(float a, float b, float c, float d) {
#if defined(__AMDGCN__) && !defined(WG_SOFT_FP8)
  int r = __builtin_amdgcn_cvt_pk_fp8_f32(a, b, 0, false);
  return (u32)__builtin_amdgcn_cvt_pk_fp8_f32(c, d, r, true);
#else
  return wg_f32_to_e4m3(a) | (wg_f32_to_e4m3(b) << 8) | (wg_f32_to_e4m3(c) << 16) | (wg_f32_to_e4m3(d) << 24);
#endif
}

struct Frame {
  const float* color;   // [rh][rw][4] f32
  const float* depth;   // [rh][rw] f32
  const float* motion;  // [rh][rw][2] f32
  const u32* hist;      // [hh][hw] R10G10B10A2
  const u32* latent;    // [rh][rw] 4 x s8
  const u32* mhist;     // [rh][rw] 2 x f16
};

// Uniform per-frame values derived from the parameters and the exposure pair.
struct Uni {
  float e;      // exposure (0 -> 1)
  float rr;     // e / e_prev (history, e_prev > 0); 0 when e is NaN (then every history value is 0, as in the spec)
  float ce;     // 100 e / kL2 (history, e_prev negative or NaN: h = 2.45 s without the clamp); 0 when e is NaN
  bool prev_ok; // e_prev > 0
  int mode;     // history formula, kHist* below
  float shx, shy;  // hw / dw, hh / dh: D pixels -> reduced-history texels
  float cdx, cdy;  // sigma1 (540 / rh) / k: disagreement scale per axis
  float s2;        // sigma2 ln 2
};

// History formulas (uniform per frame; a template argument of the per-pixel code, so that it has no branches: a
// uniform branch per channel would split the code into blocks and serialise the samples' loads).
constexpr int kHistGeneral = 0;  // e_prev > 0, e != e_prev
constexpr int kHistRaw = 1;      // e_prev negative or NaN: h = 2.45 s without the clamp
constexpr int kHistSame = 2;     // e_prev > 0 and e == e_prev: 1 + 100 e h = 10^(q / 0.36), so G(h) = q exactly

// Reduced-history texel pair (two R10G10B10A2 texels) decoded and weighted into acc[3] (acc += wa t0 + wb t1).
WG_FN void hist_pair(v2u t, float wa, float wb, float acc[3]) {
#pragma unroll
  for (int c = 0; c < 3; ++c) {
    const float a = (float)((t[0] >> (10 * c)) & 1023u), b = (float)((t[1] >> (10 * c)) & 1023u);
    acc[c] = __builtin_fmaf(wa, a, __builtin_fmaf(wb, b, acc[c]));
  }
}

// Encoded history colour G(h) for one position: valid -> from the reduced history, else gc (= G(c)).
// Spec 03: q = clamp(2.45 s, 1e-4, 3); h = (10^(q / 0.36) - 1) / 100 / e_prev; G = 0.36 log10(max(1 + 100 e h, 1e-4)),
// i.e. 1 + 100 e h = 1 + (e / e_prev) (2^(q kL2) - 1). The bilinear weights carry the factor 2.45 kL2 / 1023, so
// the interpolated value x is already q kL2 before the clamp (kHistSame: q itself, and G = q needs neither exp nor
// log).
template <int kMode>
WG_FN void hist_values(const Frame& f, const WgEncParams& P, const Uni& U, float px, float py, bool valid,
                       const float gc[3], float* out) {
#if WG_ENC_AB_HIST
#pragma unroll
  for (int c = 0; c < 3; ++c) out[c] = gc[c];
  (void)f; (void)P; (void)U; (void)px; (void)py; (void)valid;
#else
  const WgFoot ft = wg_footprint(px * U.shx - 0.5f, py * U.shy - 0.5f, P.hw, P.hh, 4);
  const v2u t0 = wg_ld64(f.hist, ft.off0), t1 = wg_ld64(f.hist, ft.off1);
  constexpr float kS = kMode == kHistSame ? 2.45f / 1023.0f : 2.45f * kL2 / 1023.0f;
  const float wy1 = ft.fy * kS, wy0 = kS - wy1;
  const float w00 = (1.0f - ft.fx) * wy0, w10 = ft.fx * wy0, w01 = (1.0f - ft.fx) * wy1, w11 = ft.fx * wy1;
  float x[3] = {0.0f, 0.0f, 0.0f};
  hist_pair(t0, w00, w10, x);
  hist_pair(t1, w01, w11, x);
#pragma unroll
  for (int c = 0; c < 3; ++c) {
    float g;
    if (kMode == kHistSame) {
      g = wg_med3f(x[c], 1e-4f, 3.0f);
    } else {
      float a;
      if (kMode == kHistGeneral) {
        const float pm1 = wg_exp2(wg_med3f(x[c], 1e-4f * kL2, 3.0f * kL2)) - 1.0f;
        a = __builtin_fmaf(U.rr, pm1, 1.0f);
      } else {
        a = __builtin_fmaf(x[c], U.ce, 1.0f);
      }
      g = kG * wg_log2(wg_fmax(a, 1e-4f));
    }
    out[c] = valid ? g : gc[c];
  }
#endif
}

struct Pix {
  float v[20];           // the 20 values (finite)
  float mn[2], mf[2], mc[2];  // motions (reconstruction pixels)
};

// Near / far candidate search over the 3 x 3 neighbourhood of the pixel whose window columns are cols[o .. o + 2]
// (depth values D*[o .. o + 2], pixel indices row* + col[o ..]). Strict comparisons in the spec's tie-break order.
template <int o>
WG_FN void near_far(const v4f& Dm, const v4f& Dc, const v4f& Dp, u32 rm, u32 rc, u32 rp, const u32 col[4],
                    bool inverted, u32& inear, u32& ifar) {
#if WG_ENC_AB_DEPTH
  inear = rm + col[o];  // fixed distinct neighbours, so that the three history samples stay distinct
  ifar = rp + col[o + 2];
  (void)Dm; (void)Dc; (void)Dp; (void)rc; (void)inverted;
#else
  const float dv[9] = {Dc[o + 1], Dm[o], Dm[o + 1], Dc[o], Dm[o + 2], Dc[o + 2], Dp[o], Dp[o + 1], Dp[o + 2]};
  const u32 di[9] = {rc + col[o + 1], rm + col[o], rm + col[o + 1], rc + col[o], rm + col[o + 2],
                     rc + col[o + 2], rp + col[o], rp + col[o + 1], rp + col[o + 2]};
  float lo = dv[0], hi = dv[0];
  u32 ilo = di[0], ihi = di[0];
#pragma unroll
  for (int n = 1; n < 9; ++n) {
    if (dv[n] < lo) { lo = dv[n]; ilo = di[n]; }
    if (dv[n] > hi) { hi = dv[n]; ihi = di[n]; }
  }
  inear = inverted ? ihi : ilo;
  ifar = inverted ? ilo : ihi;
#endif
}

// Game motion vector of render pixel index i.
WG_FN v2f_t motion_vector(const Frame& f, u32 i) {
#if WG_ENC_MOTION_F16
  const u32 m = wg_ld32(f.motion, i * 4u);
  return (v2f_t){wg_h2f_lo(m), wg_h2f_hi(m)};
#else
  return __builtin_bit_cast(v2f_t, wg_ld64(f.motion, i * 8u));
#endif
}

WG_FN void motion_of(const WgEncParams& P, v2f_t m, float out[2]) {
  out[0] = P.sx * (P.bx + m[0]);
  out[1] = P.sy * (P.by + m[1]);
}

// Values 0..19 of the pixel at render column x, row y (motions already known).
template <int kMode>
WG_FN void pixel_values(const Frame& f, const WgEncParams& P, const Uni& U, int x, int y, v4f c, Pix& px) {
  // Current colour (values 0..2) and G(c), the history value of an invalid position. E(c) and G(c) are the same
  // value, 0.36 log10(max(1 + 100 e c, 1e-4)), unless e c > 32752 (E caps it there): one logarithm per channel, and a
  // second one only in the (rare) lanes above the cap. A NaN colour channel gives 0 for both (spec: NaN -> 0); +-inf
  // follow the formulas.
  float gc[3];
#pragma unroll
  for (int ch = 0; ch < 3; ++ch) {
    const float ec = U.e * c[ch];
    const float r = kG * wg_log2(wg_fmax(__builtin_fmaf(100.0f, wg_fmin(ec, 32752.0f), 1.0f), 1e-4f));
    float g = r;
    if (ec > 32752.0f) g = kG * wg_log2(__builtin_fmaf(100.0f, ec, 1.0f));
    px.v[ch] = ec != ec ? 0.0f : r;
    gc[ch] = ec != ec ? 0.0f : g;
  }

  const float basex = (float)x + 0.5f - P.jx, basey = (float)y + 0.5f - P.jy;
  const float bkx = basex * P.kx, bky = basey * P.ky;
  float qx[4], qy[4];
  qx[0] = bkx + px.mn[0]; qy[0] = bky + px.mn[1];  // near
  qx[1] = bkx + px.mc[0]; qy[1] = bky + px.mc[1];  // center
  qx[2] = bkx + px.mf[0]; qy[2] = bky + px.mf[1];  // far
  qx[3] = ((float)x + 0.5f) * P.kx + px.mc[0];
  qy[3] = ((float)y + 0.5f) * P.ky + px.mc[1];    // unjittered
  bool valid[4];
#pragma unroll
  for (int k = 0; k < 4; ++k) valid[k] = qx[k] >= 0.0f && qx[k] <= P.dw && qy[k] >= 0.0f && qy[k] <= P.dh;

  if (P.reset) {  // uniform: nothing from the previous frame is read
#pragma unroll
    for (int i = 3; i < 15; ++i) px.v[i] = gc[i % 3];
#pragma unroll
    for (int i = 15; i < 20; ++i) px.v[i] = 0.0f;
    return;
  }
#pragma unroll
  for (int k = 0; k < 4; ++k) hist_values<kMode>(f, P, U, qx[k], qy[k], valid[k], gc, px.v + 3 + 3 * k);

  // Motion disagreement (value 15): u = clamp((base + j_prev + m_near / k) / R, 0, 1), texel position u R - 0.5.
#if WG_ENC_AB_DIS
  px.v[15] = 0.0f;
#else
  {
    const float ux = wg_med3f(basex + P.jpx + px.mn[0] * P.ikx, 0.0f, (float)P.rw);
    const float uy = wg_med3f(basey + P.jpy + px.mn[1] * P.iky, 0.0f, (float)P.rh);
    const WgFoot ft = wg_footprint(ux - 0.5f, uy - 0.5f, P.rw, P.rh, 4);
    const v2u t0 = wg_ld64(f.mhist, ft.off0), t1 = wg_ld64(f.mhist, ft.off1);
    const float a0 = 1.0f - ft.fx, b0 = 1.0f - ft.fy;
    const float ox = b0 * (a0 * wg_h2f_lo(t0[0]) + ft.fx * wg_h2f_lo(t0[1])) +
                     ft.fy * (a0 * wg_h2f_lo(t1[0]) + ft.fx * wg_h2f_lo(t1[1]));
    const float oy = b0 * (a0 * wg_h2f_hi(t0[0]) + ft.fx * wg_h2f_hi(t0[1])) +
                     ft.fy * (a0 * wg_h2f_hi(t1[0]) + ft.fx * wg_h2f_hi(t1[1]));
    const float vx = (px.mn[0] - ox) * U.cdx, vy = (px.mn[1] - oy) * U.cdy;
    const float d = U.s2 * wg_log2(1.0f + wg_sqrt(__builtin_fmaf(vx, vx, vy * vy)));
    px.v[15] = d != d ? 0.0f : d;  // NaN motion -> 0
  }
#endif

  // Latent (values 16..19) at ((p_near / k) + j_prev) / R, texel position p_near / k + j_prev - 0.5; 0 if p_near is
  // invalid. Texels k / 127 (k = -127..127); the 1 / 127 is folded into the weights.
#if WG_ENC_AB_LAT
#pragma unroll
  for (int ch = 0; ch < 4; ++ch) px.v[16 + ch] = 0.0f;
#else
  {
    const WgFoot ft = wg_footprint(qx[0] * P.ikx + P.jpx - 0.5f, qy[0] * P.iky + P.jpy - 0.5f, P.rw, P.rh, 4);
    const v2u t0 = wg_ld64(f.latent, ft.off0), t1 = wg_ld64(f.latent, ft.off1);
    constexpr float kS = 1.0f / 127.0f;
    const float wy1 = ft.fy * kS, wy0 = kS - wy1;
    const float w00 = (1.0f - ft.fx) * wy0, w10 = ft.fx * wy0, w01 = (1.0f - ft.fx) * wy1, w11 = ft.fx * wy1;
#pragma unroll
    for (int ch = 0; ch < 4; ++ch) {
      const float l00 = (float)(int)(signed char)(t0[0] >> (8 * ch)), l10 = (float)(int)(signed char)(t0[1] >> (8 * ch));
      const float l01 = (float)(int)(signed char)(t1[0] >> (8 * ch)), l11 = (float)(int)(signed char)(t1[1] >> (8 * ch));
      const float s = __builtin_fmaf(w00, l00, __builtin_fmaf(w10, l10, __builtin_fmaf(w01, l01, w11 * l11)));
      px.v[16 + ch] = valid[0] ? s : 0.0f;
    }
  }
#endif
}

WG_FN v8f round_f16(v8f a) {
#pragma unroll
  for (int i = 0; i < 8; ++i) a[i] = wg_rn16(a[i]);
  return a;
}

// Quantize the 20 values with dither t -> 5 words of FP8 codes (value i = byte i).
WG_FN void quantize(const float v[20], u32 t, u32 q[5]) {
  const u32 tt = 0xfffu + ((t + 1u) << 13);
#pragma unroll
  for (int w = 0; w < 5; ++w) {
#if WG_ENC_AB_QUANT
    q[w] = pack4_fp8(wg_med3f(v[4 * w], -448.0f, 448.0f), wg_med3f(v[4 * w + 1], -448.0f, 448.0f),
                     wg_med3f(v[4 * w + 2], -448.0f, 448.0f), wg_med3f(v[4 * w + 3], -448.0f, 448.0f)) ^ t;
#else
    q[w] = pack4_fp8(dither_q(v[4 * w], tt), dither_q(v[4 * w + 1], tt), dither_q(v[4 * w + 2], tt),
                     dither_q(v[4 * w + 3], tt));
#endif
  }
}

#include "encoder_motion_outres.h"  // EncMoOff / EncMoOn, enc_mo_motions: motion at output resolution (encoder_mo.hip)

template <class MoT = EncMoOff>
WG_FN void encoder_main(const Frame& f, u8* __restrict__ features, u32* __restrict__ m_near,
                        u32* __restrict__ m_far, u32* __restrict__ m_center, const float* __restrict__ expo,
                        const u8* __restrict__ w, const WgEncParams& P, const MoT mo = MoT()) {
  const u32 tid = wg_lane_id();
  const u32 lane = tid & 31u, t = lane & 15u, hh = lane >> 4;
  const int Wt = P.fw >> 1, ntok = (P.fh >> 1) * Wt;
  const int tok = (int)(wg_block_id() * (kEncThreads / WG_WAVE) + (tid >> 5)) * kTokensPerWave + (int)t;
  const int tk = tok < ntok ? tok : ntok - 1;
  const int r = tk / Wt, cc = tk - r * Wt;

#if !WG_ENC_FRAG_GLOBAL
  // Stage the projection weights in LDS; the copy overlaps the per-pixel work, the barrier comes after it.
  WG_LDS __attribute__((aligned(16))) u8 lds[kFragBytes];
  for (int i = (int)tid; i < kFragBytes / 16; i += kEncThreads)
    ((v4u*)lds)[i] = ((const v4u*)(w + kOffFrags))[i];
#endif

  // Uniform values.
  Uni U;
  {
    const float e = expo[0] == 0.0f ? 1.0f : expo[0];
    const float ep = expo[1] == 0.0f ? 1.0f : expo[1];
    const bool e_ok = e == e;
    U.e = e;
    U.prev_ok = ep > 0.0f;  // false for negative and NaN
    U.mode = !U.prev_ok ? kHistRaw : (e == ep ? kHistSame : kHistGeneral);
    U.rr = e_ok ? e / ep : 0.0f;
    U.ce = e_ok ? 100.0f * e / kL2 : 0.0f;
    U.shx = (float)P.hw / P.dw;
    U.shy = (float)P.hh / P.dh;
    const float s1 = ((const float*)(w + kOffSigma))[0], s2 = ((const float*)(w + kOffSigma))[1];
    U.cdx = s1 * P.dis_scale * P.ikx;
    U.cdy = s1 * P.dis_scale * P.iky;
    U.s2 = s2 * kLn2;
  }

  const int rw = P.rw, rh = P.rh;
  const int X0 = 2 * cc, Y = 2 * r + (int)hh;
  const int y = reflect(Y, rh);
  const int x0 = reflect(X0, rw), x1 = reflect(X0 + 1, rw);
  const bool rev = x0 > x1;  // right padding: slot 0 is pixel B
  const int xa = rev ? x1 : x0;

  // Depth window: rows y - 1, y, y + 1 and columns xa - 1 .. xa + 2, clamped to the render area.
  const u32 rm = (u32)(wg_med3i(y - 1, 0, rh - 1) * rw), rc = (u32)(y * rw), rp = (u32)(wg_med3i(y + 1, 0, rh - 1) * rw);
  const u32 col[4] = {(u32)(xa > 0 ? xa - 1 : 0), (u32)xa, (u32)xa + 1u, (u32)(xa + 2 < rw ? xa + 2 : rw - 1)};
  u32 inA = 0, ifA = 0, inB = 0, ifB = 0;
  if (!MoT::on) {  // (motion at output resolution: no depth neighbourhood)
#if WG_ENC_AB_DEPTH
    const v4f Dm = {0, 0, 0, 0}, Dc = Dm, Dp = Dm;
#else
    const u32 s = (u32)wg_med3i(xa - 1, 0, rw - 4);
    v4f Dm = __builtin_bit_cast(v4f, wg_ld128(f.depth, (rm + s) * 4u));
    v4f Dc = __builtin_bit_cast(v4f, wg_ld128(f.depth, (rc + s) * 4u));
    v4f Dp = __builtin_bit_cast(v4f, wg_ld128(f.depth, (rp + s) * 4u));
    if (xa == 0) {  // window loaded from column 0: columns (0, 0, 1, 2)
      Dm = (v4f){Dm[0], Dm[0], Dm[1], Dm[2]};
      Dc = (v4f){Dc[0], Dc[0], Dc[1], Dc[2]};
      Dp = (v4f){Dp[0], Dp[0], Dp[1], Dp[2]};
    } else if (xa == rw - 2) {  // window loaded from column rw - 4: columns (rw - 3, rw - 2, rw - 1, rw - 1)
      Dm = (v4f){Dm[1], Dm[2], Dm[3], Dm[3]};
      Dc = (v4f){Dc[1], Dc[2], Dc[3], Dc[3]};
      Dp = (v4f){Dp[1], Dp[2], Dp[3], Dp[3]};
    }
#endif
    near_far<0>(Dm, Dc, Dp, rm, rc, rp, col, P.depth_inverted != 0, inA, ifA);
    near_far<1>(Dm, Dc, Dp, rm, rc, rp, col, P.depth_inverted != 0, inB, ifB);
  }

  // Per slot (sub-pixel dx of this lane): render column, near / far pixel indices, centre motion, colour, dither.
  const u32 xs[2] = {(u32)x0, (u32)x1};
  const u32 ins[2] = {rev ? inB : inA, rev ? inA : inB}, ifs[2] = {rev ? ifB : ifA, rev ? ifA : ifB};
#if WG_ENC_MOTION_F16
  const v2u mh = wg_ld64(f.motion, (rc + (u32)xa) * 4u);
  const v4f mcab = {wg_h2f_lo(mh[0]), wg_h2f_hi(mh[0]), wg_h2f_lo(mh[1]), wg_h2f_hi(mh[1])};
#else
  const v4f mcab = __builtin_bit_cast(v4f, wg_ld128(f.motion, (rc + (u32)xa) * 8u));
#endif
  const float mcs[2][2] = {{rev ? mcab[2] : mcab[0], rev ? mcab[3] : mcab[1]},
                           {rev ? mcab[0] : mcab[2], rev ? mcab[1] : mcab[3]}};
  const u32 t2 = wg_ld16(w, (u32)kOffDither + 256u * (u32)P.frame + (((u32)Y & 15u) << 4) + ((u32)X0 & 15u));
  const bool store = tok < ntok && Y < rh;
  const u32 so = (u32)(Y * rw + X0) * 4u;

  // Motions of both slots, and the motion surfaces of the real pixels: one 64-bit store per surface when both pixels
  // lie inside R (then slot 0 is pixel A), so that each store instruction writes whole cache lines.
  float mn[2][2], mf[2][2], mc[2][2];
  if (MoT::on) {
    enc_mo_motions(mo, f.motion, P, xs, y, mn, mf, mc);
  } else {
#pragma unroll
    for (int dx = 0; dx < 2; ++dx) {
      motion_of(P, (v2f_t){mcs[dx][0], mcs[dx][1]}, mc[dx]);
      motion_of(P, motion_vector(f, ins[dx]), mn[dx]);
      motion_of(P, motion_vector(f, ifs[dx]), mf[dx]);
    }
  }
#if !WG_ENC_AB_STORE
  if (store && X0 < rw) {
    const v2u sn = {wg_pack_h2(mn[0][0], mn[0][1]), wg_pack_h2(mn[1][0], mn[1][1])};
    const v2u sf = {wg_pack_h2(mf[0][0], mf[0][1]), wg_pack_h2(mf[1][0], mf[1][1])};
    const v2u sc = {wg_pack_h2(mc[0][0], mc[0][1]), wg_pack_h2(mc[1][0], mc[1][1])};
    if (X0 + 1 < rw) {
      wg_st64(m_near, so, sn);
      wg_st64_hint<WG_ENC_NT_STORE>(m_far, so, sf);
      wg_st64_hint<WG_ENC_NT_STORE>(m_center, so, sc);
    } else {
      wg_st32(m_near, so, sn[0]);
      wg_st32(m_far, so, sf[0]);
      wg_st32(m_center, so, sc[0]);
    }
  }
#endif

  // Slot 0, then slot 1: each pixel's 20 values are reduced to 5 words (and their lane exchange is issued) before
  // the next pixel starts, which keeps the register count, and so the number of resident waves, in check.
  u32 q[2][5], x8[2][2];
#pragma unroll
  for (int dx = 0; dx < 2; ++dx) {
    Pix px;
#pragma unroll
    for (int i = 0; i < 2; ++i) {
      px.mn[i] = mn[dx][i];
      px.mf[i] = mf[dx][i];
      px.mc[i] = mc[dx][i];
    }
#if WG_ENC_COLOR_F16
    const v2u ch = wg_ld64(f.color, (rc + xs[dx]) * 8u);
    const v4f c = {wg_h2f_lo(ch[0]), wg_h2f_hi(ch[0]), wg_h2f_lo(ch[1]), 0.0f};
#else
    const v4f c = __builtin_bit_cast(v4f, wg_ld128_a16(f.color, (rc + xs[dx]) * 16u));
#endif
    if (U.mode == kHistGeneral)  // uniform
      pixel_values<kHistGeneral>(f, P, U, (int)xs[dx], y, c, px);
    else if (U.mode == kHistSame)
      pixel_values<kHistSame>(f, P, U, (int)xs[dx], y, c, px);
    else
      pixel_values<kHistRaw>(f, P, U, (int)xs[dx], y, c, px);
    quantize(px.v, dx ? t2 >> 8 : t2 & 0xffu, q[dx]);
#if !WG_ENC_AB_PROJ
    // Values 8..15 go to the partner lane (the other lane half of the same token).
    x8[dx][0] = wg_xor16_u(q[dx][2]);
    x8[dx][1] = wg_xor16_u(q[dx][3]);
#endif
#if WG_ENC_SEQ
    wg_sched_barrier();
#endif
  }

#if WG_ENC_AB_PROJ
  if (tok < ntok) {
    v4u o0 = {q[0][0], q[0][1], q[0][2], q[0][3]}, o1 = {q[1][0], q[1][1], q[1][2], q[1][3] ^ q[0][4] ^ q[1][4]};
    wg_st128(features, (u32)tok * kFeat + hh * 32u, o0);
    wg_st128(features, (u32)tok * kFeat + hh * 32u + 16u, o1);
  }
  (void)w; (void)x8;
#else
#if WG_ENC_PIN
  // Without this, LLVM sinks slot 1's arithmetic below the barrier into the projection (it only needs it for half of
  // the K blocks), where both phases' registers are live at once (134 VGPRs with clang 18).
  wg_pin(q[0][0], q[0][1], q[0][4], q[1][0]);
  wg_pin(q[1][1], q[1][4], x8[0][0], x8[0][1]);
  wg_pin(x8[1][0], x8[1][1], x8[1][0], x8[1][1]);
#endif
#if !WG_ENC_FRAG_GLOBAL
  wg_barrier();
  const u8* frag = lds;
#else
  wg_fence();  // the fragment loads (read-only memory) would otherwise be hoisted into the per-pixel phase
  const u8* frag = w + kOffFrags;
#endif

  // B fragments (this lane's slots) of the 8 K blocks.
  v2i bf[kKB];
#pragma unroll
  for (int p = 0; p < 4; ++p) {
    const bool own = (u32)(p >> 1) == hh;
    const int dx = p & 1;
    bf[2 * p][0] = (int)(own ? q[dx][0] : x8[dx][0]);
    bf[2 * p][1] = (int)(own ? q[dx][1] : x8[dx][1]);
    bf[2 * p + 1][0] = (int)(own ? q[dx][4] : 0u);
    bf[2 * p + 1][1] = 0;
  }

  // Two output tiles at a time (two independent accumulator chains), then their 16 bytes of the token are stored.
  // The fence keeps the next pair's 16 A fragments from being loaded early (LLVM would otherwise load all 32 at the
  // start, 64 VGPRs).
  const float* bias = (const float*)(w + kOffBias) + hh * 8;
  const u32 fo = (u32)tok * kFeat + hh * (kFeat / 2);
#pragma unroll
  for (int pr = 0; pr < kMT / 2; ++pr) {
    v8f acc[2];
    u32 o[4];
#pragma unroll
    for (int j = 0; j < 2; ++j) acc[j] = wg_load8f(bias + (2 * pr + j) * 16);
#pragma unroll
    for (int p = 0; p < 4; ++p) {
#pragma unroll
      for (int j = 0; j < 2; ++j) {
        const int mt = 2 * pr + j;
        acc[j] = wg_wmma_fp8(((const v2i*)(frag + (mt * kKB + 2 * p) * 256))[lane], bf[2 * p], acc[j]);
        acc[j] = wg_wmma_fp8(((const v2i*)(frag + (mt * kKB + 2 * p + 1) * 256))[lane], bf[2 * p + 1], acc[j]);
        acc[j] = round_f16(acc[j]);
      }
    }
#pragma unroll
    for (int j = 0; j < 2; ++j) {
#pragma unroll
      for (int i = 0; i < 8; ++i) acc[j][i] = wg_fmin(wg_fmax(acc[j][i], -6.28125f), 6.28125f);
      const v2i qo = wg_pack_q8(acc[j]);
      o[2 * j] = (u32)qo[0];
      o[2 * j + 1] = (u32)qo[1];
    }
    if (tok < ntok) wg_st128(features, fo + 16u * pr, (v4u){o[0], o[1], o[2], o[3]});
    wg_fence();
  }
#endif
}

}  // namespace

#if WG_ENC_WAVES
WG_KERNEL_NW(input_encoder, 256, WG_ENC_WAVES)
#else
WG_KERNEL_N(input_encoder, 256)
#endif
(const float* color, const float* depth, const float* motion, const u32* hist, const u32* latent, const u32* mhist,
 u8* features, u32* m_near, u32* m_far, u32* m_center, const float* expo, const u8* w, WgEncParams P) {
  const Frame f = {color, depth, motion, hist, latent, mhist};
  encoder_main<>(f, features, m_near, m_far, m_center, expo, w, P);
}

#if defined(WG_EMU)
WG_EMU_REGISTER(input_encoder)
#endif
