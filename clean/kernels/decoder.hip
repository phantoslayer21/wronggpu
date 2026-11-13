// SPDX-License-Identifier: MIT
// Coefficient decoder of spec/05-coefficient-decoder.md (stage dec0) for gfx1201 (RDNA4).
//
// One fused kernel. Projection (spec 05 section 1) on FP8 WMMA, then the per-pixel coefficients A, B, C and the gated
// latent update (sections 2-8) for the four render pixels of each F/2 token. Layouts: clean/FORMATS.md,
// "Coefficient decoder (dec0)"; packing: clean/runner/dec0.py (and clean/runtime/model.cpp, pack_dec0). Kernel
// interface, blob size and launch geometry are those of the first version (clean/kernels/baseline/decoder_v1.hip);
// this version is reorganised for speed (performance milestone, clean/PERF.md). Results are bit-identical to it.
//
// Work split: a wave handles 16 consecutive tokens (token t = WMMA column t, lanes t and t + 16). K blocks 0-1 hold
// channels 0..31 and blocks 2-3 channels 32..63, so the accumulator is rounded to FP16 between them (spec: u, then
// v). The 68 outputs (padded to 80 = 5 tiles) are ordered so that lane half h receives outputs 34h .. 34h + 33 in
// its 5 x 8 accumulator slots: exactly the 2 x 17 values of the token's pixels in row dy = h. So every lane decodes
// its own two (horizontally neighbouring) pixels with no data exchange.
//
// Order of work, so that memory latency overlaps: (1) load the feature token and the two pixels' motion, start the
// LDS staging; (2) compute the two reprojected positions and issue the previous-latent texel loads; (3) barrier and
// the projection on WMMA, while those loads are in flight; (4) decode and store both pixels with 64/128-bit stores.
//
// Exactness: every function whose argument is a projection output (an FP8 value) is a 256-entry table indexed by
// the FP8 code, built by the packer from the reference definitions (FORMATS.md): the A code, C0/C1, sx = RN32(S gx),
// rho, the gate value z and the candidate g, and the latent texel values k / 127. The remaining single-precision
// steps run here one IEEE operation at a time in the spec's order; FP contraction is off in this file. The FP16
// steps of the latent update (products and sum of FP16 values, each rounded to FP16) are done with packed FP16
// arithmetic, which gives the same results: rounding the exact value to FP32 first and then to FP16 equals one
// rounding to FP16 (24 >= 2 x 11 + 2), and products of two FP16 values are exact in FP32 anyway.
#include "wg_platform.h"
#include "wg_mem.h"

#pragma clang fp contract(off)

#ifndef WG_DEC_EARLY
// 1: previous-latent loads issued before the projection; 0: after it, as in version 1 (default since the spec team's
// timing of 1668645: 0.1321 vs 0.1337 ms, identical outputs).
#define WG_DEC_EARLY 0
#endif
#ifndef WG_DEC_FRAG_GLOBAL
#define WG_DEC_FRAG_GLOBAL 0  // 1: WMMA A fragments from global memory; only the 5 KB of tables are staged in LDS
#endif
#ifndef WG_DEC_NT
// 1: non-temporal stores of A, B, C (read only by the output stage, much later). Default 1 since clean/integration:
// identical outputs and faster on the RX 9070 XT (spec team, perf-enc 469572c: 0.1071 vs 0.1282 ms).
#define WG_DEC_NT 1
#endif
#ifndef WG_DEC_RDIV
// 1: the IEEE divisions (softmax e / s, previous-latent position hx / kx, hy / ky) as one correctly rounded reciprocal
// and fused correction steps (Markstein): the same quotients (bit-identical, see recip_rn / div_rn), ~110 fewer
// instructions per lane. 0 = the divisions as written (runtime option dec_rdiv=0: the pack12 builds __divr / __divrb).
#define WG_DEC_RDIV 1
#endif
#ifndef WG_DEC_BDIV
// 1: dec0's B softmax e / s by one shared rcp(s) and an FMA correction per quotient (no Newton step; the stored bytes
// are bit-identical, see decode_pixel); takes the softmax over from WG_DEC_RDIV. 0 = the softmax as WG_DEC_RDIV says
// (runtime option dec_bdiv=0: the pack12 builds __divb / __divrb).
#define WG_DEC_BDIV 1
#endif
#ifndef WG_DEC_PACK12
// 1: A, B and C without padding bytes, 12 bytes per pixel in two images (variant "pack12"; needs the matching read
// side in the output stage): image A (8 bytes) = A0 A1 A2 B0 | B1 B2 C2(FP16); image C (4 bytes) = C0 C1 (FP16).
#define WG_DEC_PACK12 0
#endif
// Timing-only ablations (the output is wrong on purpose): 1 removes that part of the work.
#ifndef WG_DEC_AB_PREV
#define WG_DEC_AB_PREV 0   // no previous-latent sample (p = 0)
#endif
#ifndef WG_DEC_AB_SOFT
#define WG_DEC_AB_SOFT 0   // no softmax (B = codes)
#endif
#ifndef WG_DEC_AB_PROJ
#define WG_DEC_AB_PROJ 0   // no projection (codes = feature bytes)
#endif
#ifndef WG_DEC_AB_STORE
#define WG_DEC_AB_STORE 0  // only the latent is stored (A, B, C folded into it)
#endif

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
constexpr int kOffTables = 512;  // 5 tables x 256 x 4 bytes (below)
constexpr int kOffFrags = 5632;  // W: fragments [mt < 5][kb < 4]
constexpr int kLdsBytes = 10240;
constexpr int kLdsStaged = WG_DEC_FRAG_GLOBAL ? kOffFrags - kOffTables : kLdsBytes;  // bytes copied to LDS
// Tables (u32 / f32 index in LDS), indexed by an FP8 code a (the snorm table by an 8-bit latent code):
constexpr int kTabA = 0;      // u32: bits 0-7 code of A = unorm8(RN16(sigma(a))); bits 16-31 FP16 C0(a) = RN16(RN32(sx(a)^2))
constexpr int kTabSx = 256;   // f32: sx(a) = RN32(S RN16(sigma(a)))
constexpr int kTabRho = 512;  // f32: rho(a) = RN32(tanh(RN32(T a)))
constexpr int kTabZG = 768;   // u32: bits 0-15 FP16 z(a) = RN16(0.5 + 0.5 th(a / 2)); bits 16-31 FP16 g(a) = th(a)
constexpr int kTabSn = 1024;  // f32: latent texel value max(k / 127, -1) of the signed 8-bit code k
constexpr int kFragLds = kOffFrags - kOffTables;

typedef _Float16 v2h __attribute__((ext_vector_type(2)));

// exp(x) in single for x <= 0 (softmax), about 2 ulp: x = n ln2 + r (two-part ln2, fused steps), 2^(r log2 e) 2^n.
WG_FN float exp_single(float x) {
  const float n = wg_rint(x * 1.44269504088896341f);
  float r = __builtin_fmaf(-n, 0.693145751953125f, x);
  r = __builtin_fmaf(-n, 1.42860682030941723e-06f, r);
  return __builtin_ldexpf(wg_exp2(r * 1.44269504088896341f), (int)n);
}

// Byte i (compile-time) of the code words.
WG_FN u32 code_byte(const u32* c, int i) { return (c[i >> 2] >> (8 * (i & 3))) & 0xffu; }
// FP8 E4M3 value of byte i of the code words.
WG_FN float code_value(const u32* c, int i) {
#if defined(__AMDGCN__) && !defined(WG_SOFT_FP8)
  switch (i & 3) {  // the byte selector of v_cvt_f32_fp8 must be a constant
    case 0: return __builtin_amdgcn_cvt_f32_fp8((int)c[i >> 2], 0);
    case 1: return __builtin_amdgcn_cvt_f32_fp8((int)c[i >> 2], 1);
    case 2: return __builtin_amdgcn_cvt_f32_fp8((int)c[i >> 2], 2);
    default: return __builtin_amdgcn_cvt_f32_fp8((int)c[i >> 2], 3);
  }
#else
  return wg_e4m3_to_f32(code_byte(c, i));
#endif
}

// Two values -> FP8 codes in bytes 0 and 1 (nearest even, saturating at +-448).
WG_FN u32 wg_q8_pair(float a, float b) {
#if defined(__AMDGCN__) && !defined(WG_SOFT_FP8)
  return (u32)__builtin_amdgcn_cvt_pk_fp8_f32(wg_sat448_inner(a), wg_sat448_inner(b), 0, false) & 0xffffu;
#else
  return wg_f32_to_e4m3(a) | (wg_f32_to_e4m3(b) << 8);
#endif
}

// Correctly rounded 1 / b for 1 <= b < 4 (checked exhaustively on the RX 9070 XT: all 2^24 floats) from the hardware
// reciprocal and one Newton step, and the correctly rounded quotient a / b from it (Markstein: with r = RN(1 / b) and
// q0 = RN(a r), RN(q0 + RN(a - b q0) r) = RN(a / b) while nothing overflows or goes subnormal). Used where b is in
// [1, 4): the softmax sum s = (e0 + e1) + e2 in [1, 3] (the maximum's term is exp_single(0) = 1) and kx, ky = D / R in
// [1, 4); a quotient that only an invalid previous-latent position produces (q.valid false) is never used.
WG_FN float recip_rn(float b) {
#if defined(__AMDGCN__)
  const float r0 = wg_rcp(b);
  return __builtin_fmaf(r0, __builtin_fmaf(-b, r0, 1.0f), r0);
#else
  return 1.0f / b;  // host emulation: the correctly rounded reciprocal, which the Newton step reaches on the GPU
#endif
}
WG_FN float div_rn(float a, float b, float r) {
  const float q0 = a * r;
  return __builtin_fmaf(__builtin_fmaf(-b, q0, a), r, q0);
}

WG_FN u32 unorm8(float v) { return (u32)wg_rint(v * 255.0f); }  // v in [0, 1]

// Position of a pixel's previous latent (section 6) and its texels, fetched before the projection.
struct Prev {
  bool valid;
  float fx, fy;
  bool lo, hi;   // x0 < 0 (both left taps = left texel), x0 > rw - 2 (both = right texel)
  v2u r0, r1;    // texel pairs of rows k0, k1 (columns clamp(x0, 0, rw - 2), + 1)
};

WG_FN Prev prev_fetch(const WgDecParams& P, const u32* __restrict__ lprev, int x, int y, u32 mv) {
  Prev q;
  const float mx = wg_h2f_lo(mv), my = wg_h2f_hi(mv);
  const float hx = (((float)x + 0.5f) - P.jx) * P.kx + mx;
  const float hy = (((float)y + 0.5f) - P.jy) * P.ky + my;
  q.valid = !P.reset && hx >= 0.0f && hx <= P.dw && hy >= 0.0f && hy <= P.dh;
#if WG_DEC_RDIV
  const float X = div_rn(hx, P.kx, recip_rn(P.kx)) + P.pjx, Y = div_rn(hy, P.ky, recip_rn(P.ky)) + P.pjy;
#else
  const float X = hx / P.kx + P.pjx, Y = hy / P.ky + P.pjy;
#endif
  const float tx = X - 0.5f, ty = Y - 0.5f;
  const float x0 = __builtin_floorf(tx), y0 = __builtin_floorf(ty);
  q.fx = tx - x0;
  q.fy = ty - y0;
  // Clamp-to-edge taps i0 = clamp(x0), i1 = clamp(x0 + 1) from one texel pair at clamp(x0, 0, rw - 2). For a valid
  // position x0 is in [-1, rw]; invalid positions (even NaN) are clamped in the integer domain and discarded.
  const int xi = wg_f2i(x0), yi = wg_f2i(y0);
  q.lo = xi < 0;
  q.hi = xi > P.rw - 2;
  const u32 xs = (u32)wg_med3i(xi, 0, P.rw - 2);
  const u32 k0 = (u32)wg_med3i(yi, 0, P.rh - 1), k1 = (u32)wg_med3i(yi + 1, 0, P.rh - 1);
  // Read unconditionally (no branch, so the loads stay where they are issued); on reset or for an invalid position
  // the values are not used.
#if WG_DEC_AB_PREV
  q.valid = false;
  q.r0 = q.r1 = (v2u){k0, k1};
  (void)lprev;
#else
  q.r0 = wg_ld64(lprev, (k0 * (u32)P.rw + xs) * 4u);
  q.r1 = wg_ld64(lprev, (k1 * (u32)P.rw + xs) * 4u);
#endif
  return q;
}

// Value of channel ch of a latent texel from the snorm table.
WG_FN float sn(const float* tab, u32 t, int ch) { return tab[kTabSn + ((t >> (8 * ch)) & 0xffu)]; }

// p = RN16(P) for the four channels (0 where invalid), packed as two FP16 pairs.
WG_FN void prev_values(const float* tab, const Prev& q, v2h& p01, v2h& p23) {
  const u32 t00 = q.hi ? q.r0[1] : q.r0[0], t10 = q.lo ? q.r0[0] : q.r0[1];
  const u32 t01 = q.hi ? q.r1[1] : q.r1[0], t11 = q.lo ? q.r1[0] : q.r1[1];
  const float w00 = (1.0f - q.fx) * (1.0f - q.fy), w10 = q.fx * (1.0f - q.fy);
  const float w01 = (1.0f - q.fx) * q.fy, w11 = q.fx * q.fy;
  _Float16 p[4];
#pragma unroll
  for (int ch = 0; ch < 4; ++ch) {
    const float P_ = ((w00 * sn(tab, t00, ch) + w10 * sn(tab, t10, ch)) + w01 * sn(tab, t01, ch)) + w11 * sn(tab, t11, ch);
    p[ch] = q.valid ? (_Float16)P_ : (_Float16)0.0f;
  }
  p01 = (v2h){p[0], p[1]};
  p23 = (v2h){p[2], p[3]};
}

struct Codes {  // the four stored words of one pixel (C is two words)
  u32 A, B, C0, C1, L;
};

// Pixel with its 17 FP8 codes at bytes base .. base + 16 of c.
WG_FN Codes decode_pixel(const float* __restrict__ tab, const u32* __restrict__ tabu, const u32* c, int base,
                         const Prev& q) {
  Codes o;
  // A (section 3): codes from the table.
  o.A = (tabu[kTabA + code_byte(c, base + 0)] & 0xffu) | ((tabu[kTabA + code_byte(c, base + 1)] & 0xffu) << 8) |
        ((tabu[kTabA + code_byte(c, base + 2)] & 0xffu) << 16);
  // B (section 4): softmax in single.
  const float b0 = code_value(c, base + 3), b1 = code_value(c, base + 4), b2 = code_value(c, base + 5);
  const float m = wg_fmax(wg_fmax(b0, b1), b2);
#if WG_DEC_AB_SOFT
  o.B = code_byte(c, base + 3) | (code_byte(c, base + 4) << 8) | (code_byte(c, base + 5) << 16);
  (void)m;
#else
  const float e0 = exp_single(b0 - m), e1 = exp_single(b1 - m), e2 = exp_single(b2 - m);
  const float s = (e0 + e1) + e2;
#if WG_DEC_BDIV
  // e / s as q = e r, q += fma(-s, q, e) r with one shared r = rcp(s): q may differ from the IEEE quotient in the last
  // bit, the stored bytes unorm8(q) do not (exhaustive check over all 2^24 code triples b0, b1, b2).
  const float r = wg_rcp(s);
  float qb[3] = {e0 * r, e1 * r, e2 * r};
  const float ev[3] = {e0, e1, e2};
#pragma unroll
  for (int k = 0; k < 3; ++k) qb[k] = __builtin_fmaf(__builtin_fmaf(-s, qb[k], ev[k]), r, qb[k]);
  o.B = unorm8(qb[0]) | (unorm8(qb[1]) << 8) | (unorm8(qb[2]) << 16);
#elif WG_DEC_RDIV
  const float rs = recip_rn(s);
  o.B = unorm8(div_rn(e0, s, rs)) | (unorm8(div_rn(e1, s, rs)) << 8) | (unorm8(div_rn(e2, s, rs)) << 16);
#else
  o.B = unorm8(e0 / s) | (unorm8(e1 / s) << 8) | (unorm8(e2 / s) << 16);
#endif
#endif
  // C (section 5): C0, C1 from the table; C2 = RN16(RN32(RN32(sx sy) rho)).
  const u32 a6 = code_byte(c, base + 6), a7 = code_byte(c, base + 7);
  o.C0 = (tabu[kTabA + a6] >> 16) | (tabu[kTabA + a7] & 0xffff0000u);
  o.C1 = wg_f2h_bits((tab[kTabSx + a6] * tab[kTabSx + a7]) * tab[kTabRho + code_byte(c, base + 8)]);
  // Gated latent update (section 7) in packed FP16: w = 1 - z, L = RN16(w p) + RN16(z g), each step rounded.
  v2h p01, p23;
  prev_values(tab, q, p01, p23);
  u32 zg[8];
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    zg[i] = tabu[kTabZG + code_byte(c, base + 13 + i)];    // z in the low half
    zg[4 + i] = tabu[kTabZG + code_byte(c, base + 9 + i)];  // g in the high half
  }
  const v2h z01 = __builtin_bit_cast(v2h, (zg[0] & 0xffffu) | (zg[1] << 16));
  const v2h z23 = __builtin_bit_cast(v2h, (zg[2] & 0xffffu) | (zg[3] << 16));
  const v2h g01 = __builtin_bit_cast(v2h, (zg[4] >> 16) | (zg[5] & 0xffff0000u));
  const v2h g23 = __builtin_bit_cast(v2h, (zg[6] >> 16) | (zg[7] & 0xffff0000u));
  const v2h one = {(_Float16)1.0f, (_Float16)1.0f};
  // One FP16 operation per statement: every intermediate is rounded to FP16 (on the host emulator clang would
  // otherwise evaluate a whole _Float16 expression in float and round once).
  const v2h w01 = one - z01, w23 = one - z23;
  const v2h wp01 = w01 * p01, wp23 = w23 * p23;
  const v2h zg01 = z01 * g01, zg23 = z23 * g23;
  const v2h L01 = wp01 + zg01, L23 = wp23 + zg23;
  // Stored code round(RN32(clamp(L, -1, 1) 127)): |L| <= 1 + 2^-11 here, so the clamp never changes the code.
  o.L = ((u32)(int)wg_rint((float)L01[0] * 127.0f) & 0xffu) | (((u32)(int)wg_rint((float)L01[1] * 127.0f) & 0xffu) << 8) |
        (((u32)(int)wg_rint((float)L23[0] * 127.0f) & 0xffu) << 16) | ((u32)(int)wg_rint((float)L23[1] * 127.0f) << 24);
  return o;
}

WG_FN v8f round_f16(v8f a) {
#pragma unroll
  for (int i = 0; i < 8; ++i) a[i] = wg_rn16(a[i]);
  return a;
}

struct Out {
  u32* A;       // [rh][rw] RGBA8 unorm
  u32* B;       // [rh][rw] RGBA8 unorm
  u32* C;       // [rh][rw] 4 x FP16 (2 u32)
  u32* latent;  // [rh][rw] 4 x s8
};

// The steps of the kernel as functions, shared with the fused dec1 + dec0 kernel (dec1_dec0.hip).

// Copy the tables (and, unless WG_DEC_FRAG_GLOBAL, the fragments) of the blob to lds (kLdsStaged bytes). The caller
// synchronizes before the tables are read.
template <int THREADS>
WG_FN void dec_stage(u8* lds, const u8* __restrict__ w, u32 tid) {
  for (int i = (int)tid; i < kLdsStaged / 16; i += THREADS)
    ((v4u*)lds)[i] = wg_ld128_a16(w, (u32)kOffTables + 16u * (u32)i);
}

// Motion of the lane's two pixels (x0, y) and (x0 + 1, y): one 64-bit load; the second pixel may lie outside R,
// then its value is unused. The offset is clamped so that the load stays inside the image.
WG_FN void dec_motion(const WgDecParams& P, const u32* __restrict__ motion, int x0, int y, u32& mv0, u32& mv1) {
  const u32 npix = (u32)(P.rw * P.rh);
  const u32 pix = (u32)(wg_med3i(y, 0, P.rh - 1) * P.rw + wg_min_i(x0, P.rw - 1));
  const u32 mo = pix + 1u < npix ? pix : npix - 2u;
  const v2u mv = wg_ld64(motion, mo * 4u);
  mv0 = pix == mo ? mv[0] : mv[1];
  mv1 = mv[1];
}

// Projection (section 1) of the token in this lane's WMMA column: B fragments bf (K block kb holds channels
// 32 (kb / 2) + 16 h + 8 (kb % 2) + j in slot (h, j)) -> the FP8 codes of this lane half's 34 outputs.
WG_FN void dec_project(const v2i bf[kKB], const u8* lds, const u8* __restrict__ w, u32 lane, u32 hh,
                       u32 code[2 * kMT]) {
  const float* bias = (const float*)(w + kOffBias) + hh * 8;
#if WG_DEC_AB_PROJ
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    code[i] = (u32)bf[i >> 1][i & 1];
    code[4 + i] = (u32)bf[2 + (i >> 1)][i & 1];
  }
  code[8] = code[0] ^ code[4];
  code[9] = 0u;
  (void)bias; (void)lds; (void)lane;
#else
#pragma unroll
  for (int mt = 0; mt < kMT; ++mt) {
#if WG_DEC_FRAG_GLOBAL
    wg_fence();  // fragments (read-only memory) are loaded here, not hoisted to the kernel start
    const u8* f = w + kOffFrags + mt * kKB * 256;
    (void)lds;
#else
    const u8* f = lds + kFragLds + mt * kKB * 256;
#endif
    v8f acc = wg_load8f(bias + mt * 16);
    acc = wg_wmma_fp8(((const v2i*)(f + 0 * 256))[lane], bf[0], acc);
    acc = wg_wmma_fp8(((const v2i*)(f + 1 * 256))[lane], bf[1], acc);
    if (mt < kMT - 1) {
      acc = round_f16(acc);  // u
    } else {  // tile 4: only slots 0 and 1 hold outputs (32, 33); the rest is padding
      acc[0] = wg_rn16(acc[0]);
      acc[1] = wg_rn16(acc[1]);
    }
    acc = wg_wmma_fp8(((const v2i*)(f + 2 * 256))[lane], bf[2], acc);
    acc = wg_wmma_fp8(((const v2i*)(f + 3 * 256))[lane], bf[3], acc);
    if (mt < kMT - 1) {
      acc = round_f16(acc);  // v
      const v2i qq = wg_pack_q8_inner(acc);  // h = q8(v); finite (FP16-rounded, saturated)
      code[2 * mt] = (u32)qq[0];
      code[2 * mt + 1] = (u32)qq[1];
    } else {
      code[2 * mt] = wg_q8_pair(wg_rn16(acc[0]), wg_rn16(acc[1]));
      code[2 * mt + 1] = 0u;
    }
    wg_fence();  // the next tile's fragments are loaded after this one (LLVM would load all 20 up front)
  }
#endif
  // The projection is finished here (otherwise LLVM moves parts of it into the store branches that follow).
  wg_pin(code[0], code[1], code[2], code[3]);
  wg_pin(code[4], code[5], code[6], code[7]);
  wg_pin(code[8], code[8], code[8], code[8]);
}

// Decode and store the lane's pixels (x0, y) and (x0 + 1, y) when they lie inside R (row_ok: the token is real and
// y < rh). lds holds the tables.
WG_FN void dec_store(const u32 code[2 * kMT], const u8* lds, const WgDecParams& P, const Prev& q0, const Prev& q1,
                     const Out& o, int x0, int y, bool row_ok) {
  const float* tab = (const float*)lds;
  const u32* tabu = (const u32*)lds;
  const int rw = P.rw;
  if (row_ok && x0 < rw) {
    const Codes c0 = decode_pixel(tab, tabu, code, 0, q0);
    const u32 i0 = (u32)(y * rw + x0);
    if (x0 + 1 < rw) {
      const Codes c1 = decode_pixel(tab, tabu, code, 17, q1);
#if WG_DEC_AB_STORE
      wg_st64(o.latent, i0 * 4u, (v2u){c0.L ^ c0.A ^ c0.B ^ c0.C0 ^ c0.C1, c1.L ^ c1.A ^ c1.B ^ c1.C0 ^ c1.C1});
#elif WG_DEC_PACK12
      wg_st128_hint<WG_DEC_NT>(o.A, i0 * 8u, (v4u){c0.A | (c0.B << 24), (c0.B >> 8) | (c0.C1 << 16),
                                                   c1.A | (c1.B << 24), (c1.B >> 8) | (c1.C1 << 16)});
      wg_st64_hint<WG_DEC_NT>(o.C, i0 * 4u, (v2u){c0.C0, c1.C0});
      wg_st64(o.latent, i0 * 4u, (v2u){c0.L, c1.L});
#else
      wg_st64_hint<WG_DEC_NT>(o.A, i0 * 4u, (v2u){c0.A, c1.A});
      wg_st64_hint<WG_DEC_NT>(o.B, i0 * 4u, (v2u){c0.B, c1.B});
      wg_st128_hint<WG_DEC_NT>(o.C, i0 * 8u, (v4u){c0.C0, c0.C1, c1.C0, c1.C1});
      wg_st64(o.latent, i0 * 4u, (v2u){c0.L, c1.L});
#endif
    } else {
#if WG_DEC_PACK12
      wg_st64(o.A, i0 * 8u, (v2u){c0.A | (c0.B << 24), (c0.B >> 8) | (c0.C1 << 16)});
      wg_st32(o.C, i0 * 4u, c0.C0);
#else
      wg_st32(o.A, i0 * 4u, c0.A);
      wg_st32(o.B, i0 * 4u, c0.B);
      wg_st64(o.C, i0 * 8u, (v2u){c0.C0, c0.C1});
#endif
      wg_st32(o.latent, i0 * 4u, c0.L);
    }
  }
}

WG_FN void decoder_main(const u8* __restrict__ Z, const u32* __restrict__ motion, const u32* __restrict__ lprev,
                        const Out& o, const u8* __restrict__ w, const WgDecParams& P) {
  WG_LDS __attribute__((aligned(16))) u8 lds[kLdsStaged];
  const u32 tid = wg_lane_id();
  const u32 lane = tid & 31u, t = lane & 15u, hh = lane >> 4;
  const int Wt = P.fw >> 1, ntok = (P.fh >> 1) * Wt;
  const int tok = (int)(wg_block_id() * (kDecThreads / WG_WAVE) + (tid >> 5)) * kTokensPerWave + (int)t;
  const int tk = tok < ntok ? tok : ntok - 1;
  const int r = tk / Wt, cc = tk - r * Wt;
  const int x0 = 2 * cc, y = 2 * r + (int)hh;
  const bool row_ok = tok < ntok && y < P.rh;

  // (1) Feature token (B fragments), motion of the two pixels, LDS staging.
  const v4u zlo = wg_ld128_a16(Z, (u32)tk * 64u + 16u * hh);
  const v4u zhi = wg_ld128_a16(Z, (u32)tk * 64u + 32u + 16u * hh);
  u32 mv0, mv1;
  dec_motion(P, motion, x0, y, mv0, mv1);
  dec_stage<kDecThreads>(lds, w, tid);

#if WG_DEC_EARLY
  // (2) Previous-latent texels of both pixels, in flight during the projection.
  const Prev q0 = prev_fetch(P, lprev, x0, y, mv0), q1 = prev_fetch(P, lprev, x0 + 1, y, mv1);
#endif
  wg_barrier();

  // (3) Projection.
  v2i bf[kKB];
  bf[0][0] = (int)zlo[0]; bf[0][1] = (int)zlo[1];
  bf[1][0] = (int)zlo[2]; bf[1][1] = (int)zlo[3];
  bf[2][0] = (int)zhi[0]; bf[2][1] = (int)zhi[1];
  bf[3][0] = (int)zhi[2]; bf[3][1] = (int)zhi[3];
  u32 code[2 * kMT];
  dec_project(bf, lds, w, lane, hh, code);

  // (4) Decode and store.
#if !WG_DEC_EARLY
  const Prev q0 = prev_fetch(P, lprev, x0, y, mv0), q1 = prev_fetch(P, lprev, x0 + 1, y, mv1);
#endif
  dec_store(code, lds, P, q0, q1, o, x0, y, row_ok);
}

}  // namespace

// WG_DEC_NO_EXPORTS: another kernel file includes this one for the functions only (dec1_dec0.hip).
#if !defined(WG_DEC_NO_EXPORTS)
WG_KERNEL_N(coef_decoder, 256)(const u8* features, const u32* motion, const u32* latent_prev, u32* A, u32* B, u32* C,
                               u32* latent, const u8* w, WgDecParams P) {
  const Out o = {A, B, C, latent};
  decoder_main(features, motion, latent_prev, o, w, P);
}

#if defined(WG_EMU)
WG_EMU_REGISTER(coef_decoder)
#endif
#endif  // !WG_DEC_NO_EXPORTS
