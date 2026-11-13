// SPDX-License-Identifier: MIT
// BASELINE COPY for timing comparisons only (perf_variants.py variant "v1"; not used by the runtime): clean/kernels/encoder.hip
// as of commit 2ecfc32 (branch clean/runtime-core), i.e. the version in use before the performance milestone (the
// spec team's timings 0.331 / 0.152 / 0.035 ms were taken on earlier commits of the same kernel). Unchanged except
// for the include path.
// Input encoder of spec/03-input-encoder.md (stage enc0) for gfx1201 (RDNA4).
//
// One fused kernel: per-pixel values (spec 03 "The 20 values per pixel"), dithered FP8 quantization, and the 64 x 80
// projection to the enc1 input tokens on FP8 WMMA. It also writes the three motion surfaces (m_near = next
// frame's motion history, m_far, m_center). Data and weight layouts: clean/FORMATS.md, "Input encoder (enc0)";
// packing: clean/runner/enc0.py.
//
// Work split: a wave handles 16 consecutive output tokens (token t of the wave = WMMA column t, lanes t and t + 16).
// Lane (t, h) computes the two pixels of row dy = h of its token's 2 x 2 block, i.e. sub-pixels p = 2h and 2h + 1.
// Projection: sub-pixel p's 20 FP8 values form K group p, which takes two 16-deep WMMA K blocks:
//   block 2p:     slots of the owner half o = p / 2 hold values 0..7, slots of the other half values 8..15;
//   block 2p + 1: owner slots 0..3 hold values 16..19; all other slots are zero weight (padding).
// So each lane keeps 12 bytes of each of its own pixels and swaps values 8..15 with its partner lane (lane ^ 16):
// four lane exchanges per lane. The FP32 accumulator is rounded to FP16 after each sub-pixel's two blocks, as in
// the spec's reference ("rounds the running sum to FP16 after each sub-pixel's 20-value group").
#include "../wg_platform.h"

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
constexpr int kFeat = 64, kVals = 20, kMT = kFeat / 16, kKB = 8;

// Weight blob offsets (bytes); see FORMATS.md.
constexpr int kOffBias = 0;      // 64 x f32, tile ordered (R_L(64))
constexpr int kOffSigma = 256;   // sigma1, sigma2 (f32)
constexpr int kOffDither = 512;  // T: u8 [8][16][16]
constexpr int kOffFrags = 2560;  // W: fragments [mt < 4][kb < 8], 256 B each
constexpr int kFragBytes = kMT * kKB * 256;

constexpr float kLog10_2 = 0.30102999566398120f;
constexpr float kLn2 = 0.69314718055994531f;
constexpr float kLog2_10 = 3.32192809488736235f;

#if defined(__AMDGCN__)
WG_FN float enc_log2(float x) { return __builtin_amdgcn_logf(x); }
WG_FN u32 enc_xor16(u32 v) { return (u32)__builtin_amdgcn_ds_bpermute((int)((wg_lane_id() ^ 16u) << 2), (int)v); }
#else
WG_FN float enc_log2(float x) { return std::log2(x); }
WG_FN u32 enc_xor16(u32 v) { return __builtin_bit_cast(u32, wg_xor16(__builtin_bit_cast(float, v))); }
#endif

WG_FN int clampi(int v, int lo, int hi) { return v < lo ? lo : (v > hi ? hi : v); }
WG_FN int reflect(int p, int n) {
  p = p < 0 ? -p : p;
  p = p >= n ? 2 * n - p - 2 : p;
  return clampi(p, 0, n - 1);
}
WG_FN float lerp(float a, float b, float f) { return a + f * (b - a); }
WG_FN float log10f_(float x) { return enc_log2(x) * kLog10_2; }

// Bilinear footprint at normalized (u, v) for a w x h texture: texel centers at (i + 0.5) / w, clamp-to-edge.
// The coordinate is clamped to [-1, size] first; this does not change the result and keeps NaN/inf harmless.
struct Bilin {
  int i0, i1, k0, k1;
  float fx, fy;
};
WG_FN Bilin bilin(float u, float v, int w, int h) {
  const float tx = wg_fmin(wg_fmax(u * (float)w - 0.5f, -1.0f), (float)w);
  const float ty = wg_fmin(wg_fmax(v * (float)h - 0.5f, -1.0f), (float)h);
  const float x0 = __builtin_floorf(tx), y0 = __builtin_floorf(ty);
  Bilin b;
  b.fx = tx - x0;
  b.fy = ty - y0;
  b.i0 = clampi((int)x0, 0, w - 1);
  b.i1 = clampi((int)x0 + 1, 0, w - 1);
  b.k0 = clampi((int)y0, 0, h - 1) * w;
  b.k1 = clampi((int)y0 + 1, 0, h - 1) * w;
  return b;
}

// Reduced history: R10G10B10A2 unorm (R in bits 0..9).
WG_FN v4f hist_texel(u32 t) {
  const float s = 1.0f / 1023.0f;
  v4f r;
  r[0] = (float)(t & 1023u) * s;
  r[1] = (float)((t >> 10) & 1023u) * s;
  r[2] = (float)((t >> 20) & 1023u) * s;
  r[3] = 0.0f;
  return r;
}
// Latent: 4 x signed 8-bit normalized (channel c in byte c), value max(k / 127, -1).
WG_FN v4f latent_texel(u32 t) {
  v4f r;
#pragma unroll
  for (int c = 0; c < 4; ++c) r[c] = wg_fmax((float)(signed char)(t >> (8 * c)) * (1.0f / 127.0f), -1.0f);
  return r;
}
// Motion history: 2 x FP16 (x in the low half).
WG_FN v4f mhist_texel(u32 t) {
  v4f r;
  r[0] = (float)__builtin_bit_cast(_Float16, (unsigned short)(t & 0xffffu));
  r[1] = (float)__builtin_bit_cast(_Float16, (unsigned short)(t >> 16));
  r[2] = r[3] = 0.0f;
  return r;
}

template <v4f (*DECODE)(u32)>
WG_FN v4f sample(const u32* __restrict__ tex, int w, int h, float u, float v) {
  const Bilin b = bilin(u, v, w, h);
  const v4f a = DECODE(tex[b.k0 + b.i0]), c = DECODE(tex[b.k0 + b.i1]);
  const v4f d = DECODE(tex[b.k1 + b.i0]), e = DECODE(tex[b.k1 + b.i1]);
  v4f r;
#pragma unroll
  for (int i = 0; i < 4; ++i) r[i] = lerp(lerp(a[i], c[i], b.fx), lerp(d[i], e[i], b.fx), b.fy);
  return r;
}

// G(h) = 0.36 log10(max(1 + 100 e h, 1e-4)); NaN gives 0 (the final NaN -> 0 rule; fmax would hide the NaN).
WG_FN float enc_hist(float h, float e100) {
  const float a = 1.0f + e100 * h;
  return a != a ? 0.0f : 0.36f * log10f_(wg_fmax(a, 1e-4f));
}

// Spec 03 draft 2 "Dithered FP8 quantization", steps 1-5 up to the final E4M3 conversion:
//   h = RN16(v); u = sign(h) 2^E (0 for zero/subnormal h, where E = -14); delta = sign(h) 2^E t / 1024;
//   s = RN16(h + u + delta) (one rounding, on the grid of 2^(E+1)); d = s - u; drop the low 7 mantissa bits of d.
// h + u + delta (< 2^(E+2), a multiple of 2^(E-10)) and s - u are exact in FP32. v is first made finite and clamped
// to [-1024, 1024]: NaN -> 0 and the upper cap are the spec's; the lower clamp only replaces values that end up
// saturated at -448 anyway (ANSWERS.md, spec 03 question 1).
WG_FN float dither_trunc(float v, float t) {
  v = v != v ? 0.0f : wg_fmin(wg_fmax(v, -1024.0f), 1024.0f);
  const _Float16 h = (_Float16)v;
  const float hf = (float)h;
  const u32 hb = __builtin_bit_cast(unsigned short, h);
  const u32 ex = hb & 0x7c00u;
  // 2^E: from the exponent field for a normal h, 2^-14 for zero and subnormals.
  const float p2 = ex != 0u ? __builtin_bit_cast(float, (ex + (112u << 10)) << 13) : 0.00006103515625f;
  const float sg = hf > 0.0f ? 1.0f : (hf < 0.0f ? -1.0f : 0.0f);
  const float u = ex != 0u ? sg * p2 : 0.0f;
  const float delta = sg * p2 * t * (1.0f / 1024.0f);  // exact
  const float s = (float)(_Float16)(hf + u + delta);    // exact FP32 sum, rounded once to FP16
  const _Float16 d = (_Float16)(s - u);                 // exact
  return (float)__builtin_bit_cast(_Float16, (unsigned short)(__builtin_bit_cast(unsigned short, d) & 0xff80u));
}

WG_FN u32 pack_h2(float x, float y) {
  return (u32)__builtin_bit_cast(unsigned short, (_Float16)x) |
         ((u32)__builtin_bit_cast(unsigned short, (_Float16)y) << 16);
}

struct Frame {
  const float* color;   // [rh][rw][4] f32
  const float* depth;   // [rh][rw] f32
  const float* motion;  // [rh][rw][2] f32
  const u32* hist;      // [hh][hw] R10G10B10A2
  const u32* latent;    // [rh][rw] 4 x s8
  const u32* mhist;     // [rh][rw] 2 x f16
};

// Motion of render pixel index i in reconstruction pixels: s * (b_mv + MV).
WG_FN void motion_at(const Frame& f, const WgEncParams& P, int i, float& mx, float& my) {
  const float* m = f.motion + 2 * (size_t)i;
  mx = P.sx * (P.bx + m[0]);
  my = P.sy * (P.by + m[1]);
}

// The 20 values of padded-grid pixel (X, Y) -> FP8 codes packed little-endian into q[0..4] (value i = byte i).
// If the pixel is a real render pixel (X < rw, Y < rh) and `store`, writes its motion surfaces.
// Exposures of this frame and the previous one (spec 03: a value of 0 is replaced by 1).
struct Expo {
  float e, e_prev, ie_prev;  // ie_prev = 1 / e_prev, only used when e_prev > 0
};

WG_FN Expo load_expo(const float* expo) {
  Expo x;
  x.e = expo[0] == 0.0f ? 1.0f : expo[0];
  x.e_prev = expo[1] == 0.0f ? 1.0f : expo[1];
  x.ie_prev = 1.0f / x.e_prev;
  return x;
}

WG_FN void encode_pixel(const Frame& f, const WgEncParams& P, const Expo& ex, float sigma1, float sigma2, const u8* dither, int X,
                        int Y, bool store, u32* __restrict__ m_near, u32* __restrict__ m_far,
                        u32* __restrict__ m_center, u32 q[5]) {
  const int rw = P.rw, rh = P.rh;
  const int x = reflect(X, rw), y = reflect(Y, rh), ci = y * rw + x;

  // Depth neighbourhood: strict comparisons, candidates in the spec's tie-break order.
  const int ox[9] = {0, -1, 0, -1, 1, 1, -1, 0, 1};
  const int oy[9] = {0, -1, -1, 0, -1, 0, 1, 1, 1};
  float lo = f.depth[ci], hi = lo;
  int ilo = ci, ihi = ci;
#pragma unroll
  for (int n = 1; n < 9; ++n) {
    const int j = clampi(y + oy[n], 0, rh - 1) * rw + clampi(x + ox[n], 0, rw - 1);
    const float d = f.depth[j];
    if (d < lo) { lo = d; ilo = j; }
    if (d > hi) { hi = d; ihi = j; }
  }
  const int inear = P.depth_inverted ? ihi : ilo, ifar = P.depth_inverted ? ilo : ihi;
  float mnx, mny, mfx, mfy, mcx, mcy;
  motion_at(f, P, inear, mnx, mny);
  motion_at(f, P, ifar, mfx, mfy);
  motion_at(f, P, ci, mcx, mcy);
  if (store) {
    const int o = Y * rw + X;
    m_near[o] = pack_h2(mnx, mny);
    m_far[o] = pack_h2(mfx, mfy);
    m_center[o] = pack_h2(mcx, mcy);
  }

  const float basex = (float)x + 0.5f - P.jx, basey = (float)y + 0.5f - P.jy;
  const float bkx = basex * P.kx, bky = basey * P.ky;
  float px[4], py[4];
  px[0] = bkx + mnx; py[0] = bky + mny;  // near
  px[1] = bkx + mcx; py[1] = bky + mcy;  // center
  px[2] = bkx + mfx; py[2] = bky + mfy;  // far
  px[3] = ((float)x + 0.5f) * P.kx + mcx;
  py[3] = ((float)y + 0.5f) * P.ky + mcy;  // unjittered

  float v[24];
  const float e100 = 100.0f * ex.e;
  const v4f c = *(const v4f*)(f.color + 4 * (size_t)ci);
#pragma unroll
  for (int ch = 0; ch < 3; ++ch) {
    const float ec = ex.e * c[ch];
    const float r = 0.36f * log10f_(wg_fmax(1.0f + 100.0f * wg_fmin(ec, 32752.0f), 1e-4f));
    v[ch] = ec != ec ? 0.0f : r;
  }
  const bool prev_ok = ex.e_prev > 0.0f;  // e_prev is never 0 here; false for negative and NaN
  bool valid_near = false;
#pragma unroll
  for (int k = 0; k < 4; ++k) {
    const bool valid = !P.reset && px[k] >= 0.0f && px[k] <= P.dw && py[k] >= 0.0f && py[k] <= P.dh;
    if (k == 0) valid_near = valid;
    float h[3] = {c[0], c[1], c[2]};
    if (valid) {
      const v4f s = sample<hist_texel>(f.hist, P.hw, P.hh, px[k] * P.idw, py[k] * P.idh);
#pragma unroll
      for (int ch = 0; ch < 3; ++ch) {
        const float qv = wg_fmin(wg_fmax(2.45f * s[ch], 1e-4f), 3.0f);
        const float hv = (wg_exp2(qv * (kLog2_10 / 0.36f)) - 1.0f) * 0.01f * ex.ie_prev;
        h[ch] = prev_ok ? hv : 2.45f * s[ch];
      }
    }
#pragma unroll
    for (int ch = 0; ch < 3; ++ch) v[3 + 3 * k + ch] = enc_hist(h[ch], e100);
  }

  // Position of p_near in the previous render frame, normalized: (p_near / k + j_prev) / R.
  const float lu = (px[0] * P.ikx + P.jpx) * P.irw, lv = (py[0] * P.iky + P.jpy) * P.irh;
  v[15] = 0.0f;
  if (!P.reset) {
    const float ux = wg_fmin(wg_fmax((basex + P.jpx + mnx * P.ikx) * P.irw, 0.0f), 1.0f);
    const float uy = wg_fmin(wg_fmax((basey + P.jpy + mny * P.iky) * P.irh, 0.0f), 1.0f);
    const v4f mo = sample<mhist_texel>(f.mhist, rw, rh, ux, uy);
    const float sc = sigma1 * P.dis_scale;
    const float vx = sc * (mnx - mo[0]) * P.ikx, vy = sc * (mny - mo[1]) * P.iky;
    v[15] = sigma2 * enc_log2(1.0f + __builtin_sqrtf(vx * vx + vy * vy)) * kLn2;
  }
  v4f lat = {0.0f, 0.0f, 0.0f, 0.0f};
  if (valid_near) lat = sample<latent_texel>(f.latent, rw, rh, lu, lv);
#pragma unroll
  for (int ch = 0; ch < 4; ++ch) v[16 + ch] = lat[ch];

  const float t = (float)dither[((Y & 15) << 4) | (X & 15)];
#pragma unroll
  for (int i = 0; i < kVals; ++i) v[i] = dither_trunc(v[i], t);
#pragma unroll
  for (int i = kVals; i < 24; ++i) v[i] = 0.0f;
  v8f a, b, d;
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    a[i] = v[i];
    b[i] = v[8 + i];
    d[i] = v[16 + i];
  }
  const v2i qa = wg_pack_q8(a), qb = wg_pack_q8(b), qd = wg_pack_q8(d);
  q[0] = (u32)qa[0];
  q[1] = (u32)qa[1];
  q[2] = (u32)qb[0];
  q[3] = (u32)qb[1];
  q[4] = (u32)qd[0];
}

WG_FN v8f round_f16(v8f a) {
#pragma unroll
  for (int i = 0; i < 8; ++i) a[i] = (float)(_Float16)a[i];
  return a;
}

WG_FN void encoder_main(const Frame& f, u8* __restrict__ features, u32* __restrict__ m_near,
                         u32* __restrict__ m_far, u32* __restrict__ m_center, const float* __restrict__ expo,
                         const u8* __restrict__ w, const WgEncParams& P) {
  WG_LDS __attribute__((aligned(16))) u8 lds[kFragBytes];
  const u32 tid = wg_lane_id();
  const u32 lane = tid & 31u, t = lane & 15u, hh = lane >> 4;
  const int Wt = P.fw >> 1, ntok = (P.fh >> 1) * Wt;
  const int tok = (int)(wg_block_id() * (kEncThreads / WG_WAVE) + (tid >> 5)) * kTokensPerWave + (int)t;
  const int tk = tok < ntok ? tok : ntok - 1;
  const int r = tk / Wt, cc = tk - r * Wt;

  // Stage the projection weights in LDS; the copy overlaps the per-pixel work, the barrier comes after it.
  for (int i = (int)tid; i < kFragBytes / 16; i += kEncThreads)
    ((v4u*)lds)[i] = ((const v4u*)(w + kOffFrags))[i];

  const Expo ex = load_expo(expo);
  const float sigma1 = ((const float*)(w + kOffSigma))[0], sigma2 = ((const float*)(w + kOffSigma))[1];
  const u8* dither = w + kOffDither + 256 * P.frame;
  u32 q[2][5];
#pragma unroll
  for (int dx = 0; dx < 2; ++dx) {
    const int X = 2 * cc + dx, Y = 2 * r + (int)hh;
    const bool store = tok < ntok && X < P.rw && Y < P.rh;
    encode_pixel(f, P, ex, sigma1, sigma2, dither, X, Y, store, m_near, m_far, m_center, q[dx]);
  }
  // Values 8..15 of each pixel go to the partner lane (the other lane half of the same token).
  u32 x8[2][2];
#pragma unroll
  for (int dx = 0; dx < 2; ++dx) {
    x8[dx][0] = enc_xor16(q[dx][2]);
    x8[dx][1] = enc_xor16(q[dx][3]);
  }
  wg_barrier();

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

  const float* bias = (const float*)(w + kOffBias) + hh * 8;
#pragma unroll
  for (int mt = 0; mt < kMT; ++mt) {
    v8f acc = wg_load8f(bias + mt * 16);
#pragma unroll
    for (int p = 0; p < 4; ++p) {
      acc = wg_wmma_fp8(((const v2i*)(lds + (mt * kKB + 2 * p) * 256))[lane], bf[2 * p], acc);
      acc = wg_wmma_fp8(((const v2i*)(lds + (mt * kKB + 2 * p + 1) * 256))[lane], bf[2 * p + 1], acc);
      acc = round_f16(acc);
    }
#pragma unroll
    for (int i = 0; i < 8; ++i) acc[i] = wg_fmin(wg_fmax(acc[i], -6.28125f), 6.28125f);
    if (tok < ntok) *(v2i*)(features + (size_t)tok * kFeat + hh * (kFeat / 2) + 8 * mt) = wg_pack_q8(acc);
  }
}

}  // namespace

WG_KERNEL_N(input_encoder, 256)(const float* color, const float* depth, const float* motion, const u32* hist,
                                const u32* latent, const u32* mhist, u8* features, u32* m_near, u32* m_far,
                                u32* m_center, const float* expo, const u8* w, WgEncParams P) {
  const Frame f = {color, depth, motion, hist, latent, mhist};
  encoder_main(f, features, m_near, m_far, m_center, expo, w, P);
}

#if defined(WG_EMU)
WG_EMU_REGISTER(input_encoder)
#endif
