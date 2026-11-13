// SPDX-License-Identifier: MIT
// BASELINE COPY for timing comparisons only (perf_variants.py variant "v1"; not used by the runtime): clean/kernels/decoder.hip
// as of commit 2ecfc32 (branch clean/runtime-core), i.e. the version in use before the performance milestone (the
// spec team's timings 0.331 / 0.152 / 0.035 ms were taken on earlier commits of the same kernel). Unchanged except
// for the include path.
// Coefficient decoder of spec/05-coefficient-decoder.md (stage dec0) for gfx1201 (RDNA4).
//
// One fused kernel. Projection (spec 05 section 1) on FP8 WMMA, then the per-pixel coefficients A, B, C and the gated
// latent update (sections 2-8) for the four render pixels of each F/2 token. Layouts: clean/FORMATS.md,
// "Coefficient decoder (dec0)"; packing: clean/runner/dec0.py.
//
// Work split: a wave handles 16 consecutive tokens (token t = WMMA column t, lanes t and t + 16). K blocks 0-1 hold
// channels 0..31 and blocks 2-3 channels 32..63, so the accumulator is rounded to FP16 between them (spec: u, then
// v). The 68 outputs (padded to 80 = 5 tiles) are ordered so that lane half h receives outputs 34h .. 34h + 33 in
// its 5 x 8 accumulator slots: exactly the 2 x 17 values of the token's pixels in row dy = h. So every lane decodes
// its own two pixels with no data exchange.
//
// Exactness: every function whose argument is a projection output (an FP8 value) is a 256-entry table indexed by
// the FP8 code (built by the packer from the reference definitions): RN16(sigma), th(a), th(a / 2), rho(a8), and the
// latent texel values k / 127. The remaining single-precision steps run here one IEEE operation at a time in the
// spec's order; FP contraction is off in this file so that no multiply-add is fused.
#include "../wg_platform.h"

#pragma clang fp contract(off)

// Per-frame constants, passed by value (all 4-byte fields; must match DecParams in clean/runner/dec0.py).
struct WgDecParams {
  int rw, rh;          // render size R
  int fw, fh;          // padded size F
  int reset;           // 1 on reset
  int pad0;
  float dw, dh;        // reconstruction size D
  float kx, ky;        // RN32(D / R)
  float jx, jy;        // jitter
  float pjx, pjy;      // previous jitter
};
static_assert(sizeof(WgDecParams) == 56, "WgDecParams must match DecParams in clean/runner/dec0.py");

namespace {

constexpr int kDecThreads = 256;  // 8 waves, 128 tokens per workgroup
constexpr int kTokensPerWave = 16;
constexpr int kMT = 5, kKB = 4;

// Weight blob offsets (bytes); see FORMATS.md. Everything from kOffTables on is copied to LDS.
constexpr int kOffBias = 0;      // 80 x f32, tile ordered
constexpr int kOffScal = 320;    // S, T
constexpr int kOffTables = 512;  // 5 tables x 256 f32: sigma16, th16, th16(a/2), rho, snorm
constexpr int kOffFrags = 5632;  // W: fragments [mt < 5][kb < 4]
constexpr int kLdsBytes = 10240;
constexpr int kTabSig = 0, kTabTh = 256, kTabThHalf = 512, kTabRho = 768, kTabSnorm = 1024;  // float index in LDS
constexpr int kFragLds = kOffFrags - kOffTables;

WG_FN float rn16(float x) { return (float)(_Float16)x; }
WG_FN int clampi(int v, int lo, int hi) { return v < lo ? lo : (v > hi ? hi : v); }

// exp(x) in single for x <= 0 (softmax), about 2 ulp: x = n ln2 + r (two-part ln2, fused steps), 2^(r log2 e) 2^n.
WG_FN float exp_single(float x) {
  const float n = wg_rint(x * 1.44269504088896341f);
  float r = __builtin_fmaf(-n, 0.693145751953125f, x);
  r = __builtin_fmaf(-n, 1.42860682030941723e-06f, r);
  return __builtin_ldexpf(wg_exp2(r * 1.44269504088896341f), (int)n);
}

WG_FN u32 byte_of(const u32* c, int i) { return (c[i >> 2] >> (8 * (i & 3))) & 0xffu; }

WG_FN u32 unorm8(float v) { return (u32)wg_rint(wg_fmin(wg_fmax(v, 0.0f), 1.0f) * 255.0f); }
WG_FN u32 snorm8(float v) { return (u32)(int)wg_rint(wg_fmin(wg_fmax(v, -1.0f), 1.0f) * 127.0f) & 0xffu; }

struct Out {
  u32* A;       // [rh][rw] RGBA8 unorm
  u32* B;       // [rh][rw] RGBA8 unorm
  u32* C;       // [rh][rw] 4 x FP16 (2 u32)
  u32* latent;  // [rh][rw] 4 x s8
};

// Pixel (x, y) inside R; its 17 FP8 codes are bytes base .. base + 16 of c.
WG_FN void decode_pixel(const float* __restrict__ tab, const u32* c, int base, float S, int x, int y,
                        const WgDecParams& P, const u32* __restrict__ motion, const u32* __restrict__ lprev,
                        const Out& o) {
  const int idx = y * P.rw + x;
  u32 a[17];
#pragma unroll
  for (int k = 0; k < 17; ++k) a[k] = byte_of(c, base + k);

  // A (section 3)
  const u32 cA = unorm8(tab[kTabSig + a[0]]) | (unorm8(tab[kTabSig + a[1]]) << 8) | (unorm8(tab[kTabSig + a[2]]) << 16);
  // B (section 4): softmax in single
  const float b0 = wg_e4m3_to_f32(a[3]), b1 = wg_e4m3_to_f32(a[4]), b2 = wg_e4m3_to_f32(a[5]);
  const float m = wg_fmax(wg_fmax(b0, b1), b2);
  const float e0 = exp_single(b0 - m), e1 = exp_single(b1 - m), e2 = exp_single(b2 - m);
  const float s = (e0 + e1) + e2;
  const u32 cB = unorm8(e0 / s) | (unorm8(e1 / s) << 8) | (unorm8(e2 / s) << 16);
  // C (section 5)
  const float sx = S * tab[kTabSig + a[6]], sy = S * tab[kTabSig + a[7]];
  const float rho = tab[kTabRho + a[8]];
  const float sxy = sx * sy;
  const float c0 = rn16(sx * sx), c1 = rn16(sy * sy), c2 = rn16(sxy * rho);

  // Previous latent at the reprojected position (section 6).
  const u32 mv = motion[idx];
  const float mx = (float)__builtin_bit_cast(_Float16, (unsigned short)(mv & 0xffffu));
  const float my = (float)__builtin_bit_cast(_Float16, (unsigned short)(mv >> 16));
  const float hx = (((float)x + 0.5f) - P.jx) * P.kx + mx;
  const float hy = (((float)y + 0.5f) - P.jy) * P.ky + my;
  float p[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  if (!P.reset && hx >= 0.0f && hx <= P.dw && hy >= 0.0f && hy <= P.dh) {
    const float X = hx / P.kx + P.pjx, Y = hy / P.ky + P.pjy;
    const float tx = X - 0.5f, ty = Y - 0.5f;
    const float x0 = __builtin_floorf(tx), y0 = __builtin_floorf(ty);
    const float fx = tx - x0, fy = ty - y0;
    const int i0 = clampi((int)x0, 0, P.rw - 1), i1 = clampi((int)x0 + 1, 0, P.rw - 1);
    const int k0 = clampi((int)y0, 0, P.rh - 1) * P.rw, k1 = clampi((int)y0 + 1, 0, P.rh - 1) * P.rw;
    const float w00 = (1.0f - fx) * (1.0f - fy), w10 = fx * (1.0f - fy);
    const float w01 = (1.0f - fx) * fy, w11 = fx * fy;
    const u32 t00 = lprev[k0 + i0], t10 = lprev[k0 + i1], t01 = lprev[k1 + i0], t11 = lprev[k1 + i1];
#pragma unroll
    for (int ch = 0; ch < 4; ++ch) {
      const int sh = 8 * ch;
      const float P_ = ((w00 * tab[kTabSnorm + ((t00 >> sh) & 0xffu)] + w10 * tab[kTabSnorm + ((t10 >> sh) & 0xffu)]) +
                        w01 * tab[kTabSnorm + ((t01 >> sh) & 0xffu)]) +
                       w11 * tab[kTabSnorm + ((t11 >> sh) & 0xffu)];
      p[ch] = rn16(P_);
    }
  }
  // Gated latent update (section 7).
  u32 cL = 0;
#pragma unroll
  for (int ch = 0; ch < 4; ++ch) {
    const float z = rn16(0.5f + 0.5f * tab[kTabThHalf + a[13 + ch]]);
    const float g = tab[kTabTh + a[9 + ch]];
    const float w = rn16(1.0f - z);
    const float L = rn16(rn16(w * p[ch]) + rn16(z * g));
    cL |= snorm8(L) << (8 * ch);
  }

  o.A[idx] = cA;
  o.B[idx] = cB;
  o.C[2 * idx] = (u32)__builtin_bit_cast(unsigned short, (_Float16)c0) |
                 ((u32)__builtin_bit_cast(unsigned short, (_Float16)c1) << 16);
  o.C[2 * idx + 1] = (u32)__builtin_bit_cast(unsigned short, (_Float16)c2);
  o.latent[idx] = cL;
}

WG_FN v8f round_f16(v8f a) {
#pragma unroll
  for (int i = 0; i < 8; ++i) a[i] = rn16(a[i]);
  return a;
}

WG_FN void decoder_main(const u8* __restrict__ Z, const u32* __restrict__ motion, const u32* __restrict__ lprev,
                        const Out& o, const u8* __restrict__ w, const WgDecParams& P) {
  WG_LDS __attribute__((aligned(16))) u8 lds[kLdsBytes];
  const u32 tid = wg_lane_id();
  const u32 lane = tid & 31u, t = lane & 15u, hh = lane >> 4;
  const int Wt = P.fw >> 1, ntok = (P.fh >> 1) * Wt;
  const int tok = (int)(wg_block_id() * (kDecThreads / WG_WAVE) + (tid >> 5)) * kTokensPerWave + (int)t;
  const int tk = tok < ntok ? tok : ntok - 1;
  const int r = tk / Wt, cc = tk - r * Wt;

  // B fragments: K block kb holds channels 32 (kb / 2) + 16 h + 8 (kb % 2) + j in slot (h, j).
  const v4u zlo = *(const v4u*)(Z + (size_t)tk * 64 + 16 * hh);
  const v4u zhi = *(const v4u*)(Z + (size_t)tk * 64 + 32 + 16 * hh);
  v2i bf[kKB];
  bf[0][0] = (int)zlo[0]; bf[0][1] = (int)zlo[1];
  bf[1][0] = (int)zlo[2]; bf[1][1] = (int)zlo[3];
  bf[2][0] = (int)zhi[0]; bf[2][1] = (int)zhi[1];
  bf[3][0] = (int)zhi[2]; bf[3][1] = (int)zhi[3];

  for (int i = (int)tid; i < kLdsBytes / 16; i += kDecThreads) ((v4u*)lds)[i] = ((const v4u*)(w + kOffTables))[i];
  wg_barrier();

  const float* bias = (const float*)(w + kOffBias) + hh * 8;
  u32 code[2 * kMT];
#pragma unroll
  for (int mt = 0; mt < kMT; ++mt) {
    const u8* f = lds + kFragLds + mt * kKB * 256;
    v8f acc = wg_load8f(bias + mt * 16);
    acc = wg_wmma_fp8(((const v2i*)(f + 0 * 256))[lane], bf[0], acc);
    acc = wg_wmma_fp8(((const v2i*)(f + 1 * 256))[lane], bf[1], acc);
    acc = round_f16(acc);  // u
    acc = wg_wmma_fp8(((const v2i*)(f + 2 * 256))[lane], bf[2], acc);
    acc = wg_wmma_fp8(((const v2i*)(f + 3 * 256))[lane], bf[3], acc);
    acc = round_f16(acc);  // v
    const v2i q = wg_pack_q8(acc);  // h = q8(v)
    code[2 * mt] = (u32)q[0];
    code[2 * mt + 1] = (u32)q[1];
  }

  const float S = ((const float*)(w + kOffScal))[0];
  const float* tab = (const float*)lds;
#pragma unroll
  for (int dx = 0; dx < 2; ++dx) {
    const int x = 2 * cc + dx, y = 2 * r + (int)hh;
    if (tok < ntok && x < P.rw && y < P.rh) decode_pixel(tab, code, 17 * dx, S, x, y, P, motion, lprev, o);
  }
}

}  // namespace

WG_KERNEL_N(coef_decoder, 256)(const u8* features, const u32* motion, const u32* latent_prev, u32* A, u32* B, u32* C,
                               u32* latent, const u8* w, WgDecParams P) {
  const Out o = {A, B, C, latent};
  decoder_main(features, motion, latent_prev, o, w, P);
}

#if defined(WG_EMU)
WG_EMU_REGISTER(coef_decoder)
#endif
