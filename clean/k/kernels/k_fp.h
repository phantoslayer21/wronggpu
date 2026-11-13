// SPDX-License-Identifier: MIT
// Exact scalar FP32 / FP16 helpers for the model K input and output stages (specs 42, 43).
//
// Specs 42 and 43 define every result by FP32 (or FP16) operations in a given order, each rounded to nearest even;
// "fma" is one rounding. Contraction is therefore switched off in every function here and in the stage kernels, and
// fma is always written out. On the GPU: v_fma_f32 / v_fma_f16, correctly rounded division and square root (the HIP
// default). In the host emulator: std::fma, and an exact FP16 fma (round-to-odd in double, then one rounding).
//
// "The mathematical function rounded to FP32" (log2, ln, 2^x): kf_log2 / kf_ln / kf_exp2 (end of this file) evaluate
// in FP32 double-word arithmetic and accept the result by a rounding test; the rare undecided inputs use the binary64
// implementations (kd_*: argument reduction and series from the textbook identities, a few binary64 ulps, rounded
// once). Both give the correctly rounded binary32 result except in cases far rarer than any test can see, and the
// two paths agree bit for bit on every input (exhaustive GPU check). The same code runs on the GPU and in the emulator.
#pragma once

#include "k_common.h"

typedef unsigned long long u64k;

// ---- elementary steps -----------------------------------------------------------------------------------------------

WG_FN float kf_fma(float a, float b, float c) {
#if defined(__AMDGCN__)
  return __builtin_fmaf(a, b, c);
#else
  return std::fma(a, b, c);
#endif
}

WG_FN double kd_next_odd(double s, double lo) {  // s + lo (two-sum pair) rounded to odd
  if (lo == 0.0 || s != s || s - s != 0.0) return s;
  u64k b = __builtin_bit_cast(u64k, s);
  if (b & 1ull) return s;
  const bool away = (lo > 0.0) == (s > 0.0);
  b = away ? b + 1ull : b - 1ull;
  return __builtin_bit_cast(double, b);
}

WG_FN h16 kh_fma(h16 a, h16 b, h16 c) {
#if defined(__AMDGCN__)
  return __builtin_fmaf16(a, b, c);
#else
#pragma clang fp contract(off)
  const double p = (double)a * (double)b;  // exact
  const double cc = (double)c;
  const double s = p + cc;
  const double bp = s - cc;
  const double lo = (p - bp) + (cc - (s - bp));
  return (h16)kd_next_odd(s, lo);  // round to odd at 53 bits, then once to binary16
#endif
}

WG_FN h16 kh_add(h16 a, h16 b) {
#pragma clang fp contract(off)
  return (h16)(a + b);
}
WG_FN h16 kh_sub(h16 a, h16 b) {
#pragma clang fp contract(off)
  return (h16)(a - b);
}
WG_FN h16 kh_mul(h16 a, h16 b) {
#pragma clang fp contract(off)
  return (h16)(a * b);
}
// a / b of FP16 values rounded once to FP16: the FP32 quotient (correctly rounded) rounded to FP16 is RN16 of the exact
// quotient, because double rounding is innocuous for division when 24 >= 2 * 11 + 2 (S. Figueroa, "When is double
// rounding innocuous?", 1995). The barrier keeps the FP32 rounding (as kh_round below). No FP64 (slow on RDNA4).
WG_FN h16 kh_div(h16 a, h16 b) {
  float q = (float)a / (float)b;
#if defined(__AMDGCN__)
  __asm__("" : "+v"(q));
#endif
  return (h16)q;
}

// RN16 of a value that the spec has already rounded to FP32. Without the barrier the GPU compiler may fuse the FP32
// operation that produced x into the conversion (v_fma_mixlo_f16 / v_fma_mixhi_f16: a*b+c rounded once to FP16),
// which skips the FP32 rounding: seen for (h16)(Coo * kk) of spec 43 section 6.6, 1-ulp differences in ~1e-5 of
// the pixels. Use this for every FP32 -> FP16 rounding of a computed value in the stage kernels.
WG_FN h16 kh_round(float x) {
#if defined(__AMDGCN__)
  __asm__("" : "+v"(x));
#endif
  return (h16)x;
}

WG_FN float kf_floor(float x) { return __builtin_floorf(x); }
WG_FN float kf_trunc(float x) { return __builtin_truncf(x); }
WG_FN h16 kh_floor(h16 x) { return (h16)__builtin_floorf((float)x); }
WG_FN h16 kh_rni(h16 x) { return (h16)__builtin_rintf((float)x); }  // round to nearest, ties to even
WG_FN float kf_rni(float x) { return __builtin_rintf(x); }

// R(t): round half away from zero (specs 42 / 43), exact for every binary32 t
WG_FN float kf_R(float t) {
  const float a = __builtin_fabsf(t);
  const float fl = __builtin_floorf(a);
  const float r = (a - fl) >= 0.5f ? fl + 1.0f : fl;  // a - fl is exact
  return t < 0.0f ? -r : r;
}

WG_FN float kf_sat(float x) { return x != x ? 0.0f : (x < 0.0f ? 0.0f : (x > 1.0f ? 1.0f : x)); }
WG_FN h16 kh_sat(h16 x) {
  return x != x ? (h16)0.0f : (x < (h16)0.0f ? (h16)0.0f : (x > (h16)1.0f ? (h16)1.0f : x));
}
// max / min of specs 42 / 43: IEEE maxNum / minNum (a NaN operand gives the other operand), as GPU max / min do
// (QUESTIONS.md K-Q8). The NumPy references use np.fmax / np.fmin.
WG_FN float kf_max(float a, float b) { return __builtin_fmaxf(a, b); }
WG_FN float kf_min(float a, float b) { return __builtin_fminf(a, b); }
WG_FN int ki_min(int a, int b) { return a < b ? a : b; }
WG_FN int ki_max(int a, int b) { return a > b ? a : b; }
WG_FN bool kf_finite(float x) { return x - x == 0.0f; }

WG_FN float kf_bits(u32 b) { return __builtin_bit_cast(float, b); }

// ---- packed FP16: two independent values, each operation rounded once per element exactly like the kh_* helper ----
// On the GPU one v_pk_* instruction (v_pk_add_f16 / v_pk_mul_f16 / v_pk_fma_f16 / v_pk_max_f16 / v_pk_min_f16 round
// each half like the scalar instruction); in the host emulator the scalar helpers per element (vector _Float16
// arithmetic on x86 may keep excess precision).
WG_FN v2h kh2(h16 a, h16 b) {
  v2h r = {a, b};
  return r;
}
WG_FN v2h kh2_splat(h16 a) { return kh2(a, a); }
#if defined(__AMDGCN__)
WG_FN v2h kh2_add(v2h a, v2h b) {
#pragma clang fp contract(off)
  return a + b;
}
WG_FN v2h kh2_sub(v2h a, v2h b) {
#pragma clang fp contract(off)
  return a - b;
}
WG_FN v2h kh2_mul(v2h a, v2h b) {
#pragma clang fp contract(off)
  return a * b;
}
WG_FN v2h kh2_fma(v2h a, v2h b, v2h c) { return __builtin_elementwise_fma(a, b, c); }
WG_FN v2h kh2_abs(v2h a) { return __builtin_elementwise_abs(a); }
WG_FN v2h kh2_rni(v2h a) { return __builtin_elementwise_roundeven(a); }  // two v_rndne_f16 (no packed form)
WG_FN v2h kh2_maxnum(v2h a, v2h b) { return __builtin_elementwise_max(a, b); }  // IEEE maxNum per half (v_pk_max_num_f16)
// kh_sat per element as maxNum(x, 0) then minNum(., 1): equal to kh_sat for every x except x = -0 (kh_sat keeps -0),
// so callers pass only values that cannot be -0 (an fma / add whose addend is a nonzero or +0 value, see k_io.hip).
WG_FN v2h kh2_sat_nz(v2h a) {
  const v2h z = {(h16)0.0f, (h16)0.0f}, o = {(h16)1.0f, (h16)1.0f};
  return __builtin_elementwise_min(__builtin_elementwise_max(a, z), o);
}
#else
WG_FN v2h kh2_add(v2h a, v2h b) { return kh2(kh_add(a[0], b[0]), kh_add(a[1], b[1])); }
WG_FN v2h kh2_sub(v2h a, v2h b) { return kh2(kh_sub(a[0], b[0]), kh_sub(a[1], b[1])); }
WG_FN v2h kh2_mul(v2h a, v2h b) { return kh2(kh_mul(a[0], b[0]), kh_mul(a[1], b[1])); }
WG_FN v2h kh2_fma(v2h a, v2h b, v2h c) { return kh2(kh_fma(a[0], b[0], c[0]), kh_fma(a[1], b[1], c[1])); }
WG_FN v2h kh2_abs(v2h a) { return kh2(__builtin_fabsf16(a[0]), __builtin_fabsf16(a[1])); }
WG_FN v2h kh2_rni(v2h a) { return kh2(kh_rni(a[0]), kh_rni(a[1])); }
// maxNum per element (a NaN operand gives the other; the sign of a zero result is not specified, as on the GPU)
WG_FN h16 kh_maxnum(h16 a, h16 b) { return a != a ? b : (b != b ? a : (a > b ? a : b)); }
WG_FN v2h kh2_maxnum(v2h a, v2h b) { return kh2(kh_maxnum(a[0], b[0]), kh_maxnum(a[1], b[1])); }
WG_FN v2h kh2_sat_nz(v2h a) { return kh2(kh_sat(a[0]), kh_sat(a[1])); }
#endif
typedef short v2s16 __attribute__((ext_vector_type(2)));
// Per half, as a 32-bit mask: all ones where a > b, else 0, without compares (RDNA has no packed FP16 compare):
// d = h(b - a) is negative (sign set, not NaN) iff a > b, for every pair including infinities (inf - inf = NaN: not
// greater, as IEEE says) and NaN operands (d NaN of either sign: not greater), since FP16 subtraction with gradual
// underflow gets the sign right and gives 0 only for equal operands; the one difference from IEEE "a > b" is
// a = +0, b = -0 (d = -0 - +0 = -0: all ones). NaN test on the bits: |d| <= 0x7C00 iff |d| - 0x7C01 < 0.
// v_pk_add_f16, v_and_b32, v_pk_add_u16, v_and_b32, v_pk_ashrrev_i16. The barrier keeps the compiler from turning
// the mask back into per-half compares and selects.
WG_FN u32 kh2_gtmask(v2h a, v2h b) {
  const v2s16 d = __builtin_bit_cast(v2s16, kh2_sub(b, a));
  const v2s16 notnan = (v2s16)(d & (short)0x7FFF) - (short)0x7C01;
  u32 m = __builtin_bit_cast(u32, (v2s16)(d & notnan) >> (short)15);
#if defined(__AMDGCN__)
  __asm__("" : "+v"(m));
#endif
  return m;
}
// per half: a where m is all ones, else b (v_bfi_b32)
WG_FN v2h kh2_bsel(u32 m, v2h a, v2h b) {
  const u32 ua = __builtin_bit_cast(u32, a), ub = __builtin_bit_cast(u32, b);
  return __builtin_bit_cast(v2h, (ua & m) | (ub & ~m));
}

// ---- double-precision functions --------------------------------------------------------------------------------------

#define KD_LN2 0.6931471805599453094
#define KD_INV_LN2 1.4426950408889634074

WG_FN double kd_fma(double a, double b, double c) {
#if defined(__AMDGCN__)
  return __builtin_fma(a, b, c);
#else
  return std::fma(a, b, c);
#endif
}

// 1 / d for d in [1.7, 2.5]: an FP32 reciprocal estimate and two Newton steps (relative error ~2^-46, then below
// one binary64 ulp); no IEEE division sequence (slow in binary64 on the GPU).
WG_FN double kd_rcp_near2(double d) {
#if defined(__AMDGCN__)
  double q = (double)__builtin_amdgcn_rcpf((float)d);
#else
  double q = (double)(1.0f / (float)d);
#endif
  q = kd_fma(q, kd_fma(-d, q, 1.0), q);
  q = kd_fma(q, kd_fma(-d, q, 1.0), q);
  return q;
}

// For finite x > 0 (normal binary64 range; inputs come from binary32): x = 2^e m with m in (1/sqrt 2, sqrt 2] and
// ln m = 2 s (1 + s^2/3 + ... + s^22/23), s = (m - 1)/(m + 1), |s| <= 0.172 (the truncation is below 2^-64).
// Returns e and 2 s (1 + ...) (= ln m); NaN / 0 / inf handled by the callers.
WG_FN double kd_log_core(double x, int* e_out) {
#pragma clang fp contract(off)
  const u64k b = __builtin_bit_cast(u64k, x);
  int e = (int)((b >> 52) & 0x7ffull) - 1023;
  double m = __builtin_bit_cast(double, (b & 0x000fffffffffffffull) | 0x3ff0000000000000ull);  // [1, 2)
  if (m > 1.4142135623730951) {
    m = m * 0.5;
    e += 1;
  }
  const double s = (m - 1.0) * kd_rcp_near2(m + 1.0);  // m - 1 and m + 1 are exact
  const double s2 = s * s;
  double p = 1.0 / 23.0;
#pragma unroll
  for (int k = 10; k >= 0; --k) p = kd_fma(p, s2, 1.0 / (double)(2 * k + 1));  // constants folded
  *e_out = e;
  return 2.0 * s * p;
}

WG_FN double kd_log_special(double x, bool* special) {
  *special = true;
  if (x != x || x < 0.0) return __builtin_bit_cast(double, 0x7ff8000000000000ull);
  if (x == 0.0) return -__builtin_bit_cast(double, 0x7ff0000000000000ull);
  if (x - x != 0.0) return x;  // +inf
  *special = false;
  return 0.0;
}

// ln x
WG_FN double kd_log(double x) {
#pragma clang fp contract(off)
  bool sp;
  const double v = kd_log_special(x, &sp);
  if (sp) return v;
  int e;
  const double l = kd_log_core(x, &e);
  return kd_fma((double)e, KD_LN2, l);
}

// log2 x = e + ln(m) / ln 2 (x near 1 has e = 0: no cancellation)
WG_FN double kd_log2(double x) {
#pragma clang fp contract(off)
  bool sp;
  const double v = kd_log_special(x, &sp);
  if (sp) return v;
  int e;
  const double l = kd_log_core(x, &e);
  return kd_fma(l, KD_INV_LN2, (double)e);
}

WG_FN double kd_ldexp(double v, int n) {  // v * 2^n, n in [-2044, 2046]
  for (int i = 0; i < 2; ++i) {
    const int k = n > 1023 ? 1023 : (n < -1022 ? -1022 : n);
    v = v * __builtin_bit_cast(double, (u64k)(k + 1023) << 52);
    n -= k;
  }
  return v;
}

// 2^y
WG_FN double kd_exp2(double y) {
#pragma clang fp contract(off)
  if (y != y) return y;
  if (y > 1100.0) return __builtin_bit_cast(double, 0x7ff0000000000000ull);
  if (y < -1100.0) return 0.0;
  const double n = __builtin_rint(y);
  const double r = (y - n) * KD_LN2;  // |r| <= 0.347
  // e^r = 1 + r (1 + r/2 (1 + r/3 (...))) to r^16 / 16! (truncation below 2^-64); r/k as r * (1/k) with the
  // constant folded: no division
  double p = 1.0;
#pragma unroll
  for (int k = 16; k >= 1; --k) p = kd_fma(p, r * (1.0 / (double)k), 1.0);
  return kd_ldexp(p, (int)n);
}

// 1/sqrt(x) rounded once to FP32 (spec 43 6.8), binary64 form: the binary64 square root and division are correctly
// rounded, and the binary64 quotient is close enough to round to the correctly rounded binary32 result.
WG_FN float kd_rsqrt32(float x) {
#if defined(__AMDGCN__)
  return (float)(1.0 / __builtin_sqrt((double)x));
#else
  return (float)(1.0 / std::sqrt((double)x));
#endif
}
// The same for the stage's domain x >= 1 on the GPU without the binary64 square root and division (slow on RDNA4):
// the hardware estimate y (v_rsq_f32, about 1 ulp), then exact rounding tests. z = x^(-1/2) lies above the midpoint
// m = y + ulp/2 iff m^2 x < 1; m^2 is exact in binary64 (25-bit m) and fma(m^2, x, -1) has the sign of the exact value
// (never 0: m^2 x = 1 would need m to be a power of two). Moving y by one ulp per test; equal to kd_rsqrt32 for every
// input >= 1 (exhaustive GPU check, TestFp32Functions).
WG_FN float kf_rsqrt(float x) {
#if defined(__AMDGCN__)
  if (x >= 1.0f && x - x == 0.0f) {
    float y = __builtin_amdgcn_rsqf(x);
    const double xd = (double)x;
    for (int it = 0; it < 3; ++it) {
      const u32 b = __builtin_bit_cast(u32, y);
      const double yd = (double)y;
      const double ulp = (double)__builtin_bit_cast(float, (b & 0x7F800000u) - (23u << 23));
      const double up = yd + 0.5 * ulp;
      const double lo = yd - ((b & 0x7FFFFFu) == 0u ? 0.25 : 0.5) * ulp;  // below a power of two the spacing halves
      if (__builtin_fma(up * up, xd, -1.0) < 0.0) {
        y = __builtin_bit_cast(float, b + 1u);
      } else if (__builtin_fma(lo * lo, xd, -1.0) > 0.0) {
        y = __builtin_bit_cast(float, b - 1u);
      } else {
        return y;
      }
    }
  }
#endif
  return kd_rsqrt32(x);
}

// ---- correctly rounded binary32 log2 / ln / 2^x in FP32 double-word arithmetic --------------------------------
//
// The value as an unevaluated sum hi + lo of two binary32 numbers, relative error far below 2^-38; a rounding test
// (Ziv) accepts hi when every value within 2^-38 |hi| of hi + lo rounds to hi, which is then RN32 of the exact value.
// Otherwise (about 1e-4 of the inputs in the ranges the stages use) the binary64 path decides, so the results are the
// binary64 path's bit for bit: checked exhaustively on the RX 9070 XT for every normal positive binary32 input (log2,
// ln) and every input in [-125, 127) (2^x), 0 differences. FP64 is slow on RDNA4; this path uses FP32 only.
// Tables 2^(j/64) and -log2(inv_j) are generated at compile time by constexpr binary64 series, split into hi + lo.

struct kf2 {
  float h, l;
};

WG_FN kf2 kf2_two_sum(float a, float b) {
#pragma clang fp contract(off)
  const float s = a + b;
  const float bb = s - a;
  return {s, (a - (s - bb)) + (b - bb)};
}
WG_FN kf2 kf2_fast_two_sum(float a, float b) {  // |a| >= |b| or a == 0
#pragma clang fp contract(off)
  const float s = a + b;
  return {s, b - (s - a)};
}
WG_FN kf2 kf2_two_prod(float a, float b) {
#pragma clang fp contract(off)
  const float p = a * b;
  return {p, kf_fma(a, b, -p)};
}
WG_FN kf2 kf2_add(kf2 a, kf2 b) {
#pragma clang fp contract(off)
  kf2 s = kf2_two_sum(a.h, b.h);
  s.l = s.l + (a.l + b.l);
  return kf2_fast_two_sum(s.h, s.l);
}
WG_FN kf2 kf2_mul(kf2 a, kf2 b) {
#pragma clang fp contract(off)
  kf2 p = kf2_two_prod(a.h, b.h);
  p.l = kf_fma(a.h, b.l, kf_fma(a.l, b.h, p.l));
  return kf2_fast_two_sum(p.h, p.l);
}

// ---- compile-time binary64 helpers (series; used only to build the tables) ----
constexpr double kc_exp(double x) {  // |x| <= 0.4
  double t = 1.0, s = 1.0;
  for (int k = 1; k < 30; ++k) {
    t = t * x / k;
    s = s + t;
  }
  return s;
}
constexpr double kc_log(double x) {  // x in [0.5, 2]
  const double s = (x - 1.0) / (x + 1.0), s2 = s * s;
  double t = s, r = 0.0;
  for (int k = 0; k < 40; ++k) {
    r = r + t / (2 * k + 1);
    t = t * s2;
  }
  return 2.0 * r;
}
constexpr double kc_ln2 = 0.69314718055994530942;

// 2^(j/64), j = -32 .. 32
struct KfExpTab {
  float h[65], l[65];
  constexpr KfExpTab() : h(), l() {
    for (int i = 0; i < 65; ++i) {
      const double v = kc_exp((i - 32) / 64.0 * kc_ln2);
      h[i] = (float)v;
      l[i] = (float)(v - (double)h[i]);
    }
  }
};
// log2 bins: m in [0.75, 1.5), j = round((m - 1) * 128) in -32 .. 64, c_j = 1 + j/128, inv_j = RN32(1/c_j),
// -log2(inv_j) as hi + lo.
struct KfLogTab {
  float inv[97], h[97], l[97];
  constexpr KfLogTab() : inv(), h(), l() {
    for (int i = 0; i < 97; ++i) {
      const double c = 1.0 + (i - 32) / 128.0;
      inv[i] = (float)(1.0 / c);
      const double v = -kc_log((double)inv[i]) / kc_ln2;
      h[i] = (float)v;
      l[i] = (float)(v - (double)h[i]);
    }
  }
};

WG_FN const KfExpTab& kf_exp_tab() {
  static constexpr KfExpTab t{};
  return t;
}
WG_FN const KfLogTab& kf_log_tab() {
  static constexpr KfLogTab t{};
  return t;
}

// 1/ln 2 and ln 2 as hi + lo
#define KF2_INV_LN2_H ((float)1.4426950408889634074)
#define KF2_INV_LN2_L ((float)(1.4426950408889634074 - (double)(float)1.4426950408889634074))
#define KF2_LN2_H ((float)0.69314718055994530942)
#define KF2_LN2_L ((float)(0.69314718055994530942 - (double)(float)0.69314718055994530942))

// Ziv test: is RN32(v) = r.h for every v with |v - (r.h + r.l)| <= 2^-38 |r.h| ?
WG_FN bool kf2_round_ok(kf2 r) {
#pragma clang fp contract(off)
  const u32 b = __builtin_bit_cast(u32, r.h);
  const u32 ex = (b >> 23) & 0xFFu;
  if (ex < 40u || ex == 0xFFu) return false;  // tiny, subnormal, inf / NaN: exact path
  const float ulp = __builtin_bit_cast(float, (ex - 23u) << 23);
  // below a power of two (in magnitude) the spacing halves
  const bool pow2 = (b & 0x7FFFFFu) == 0u;
  const bool toward_zero = (r.l < 0.0f) != (r.h < 0.0f);
  const float half = (pow2 && toward_zero) ? 0.25f * ulp : 0.5f * ulp;
  const float err = __builtin_fabsf(r.h) * 0x1p-38f;
  return __builtin_fabsf(r.l) + err < half;
}

// log2 x as hi + lo for finite normal x > 0
WG_FN kf2 kf2_log2(float x) {
#pragma clang fp contract(off)
  const u32 b = __builtin_bit_cast(u32, x);
  int e = (int)((b >> 23) & 0xFFu) - 127;
  float m = __builtin_bit_cast(float, (b & 0x7FFFFFu) | 0x3F800000u);  // [1, 2)
  if (m >= 1.5f) {
    m = m * 0.5f;
    e += 1;
  }
  const int j = (int)__builtin_rintf((m - 1.0f) * 128.0f);  // -32 .. 64
  const KfLogTab& T = kf_log_tab();
  const float inv = T.inv[j + 32];
  const kf2 p = kf2_two_prod(m, inv);  // m inv = 1 + t exactly
  const kf2 t = kf2_two_sum(p.h - 1.0f, p.l);  // p.h - 1 is exact (p.h in [0.98, 1.02])
  // ln(1 + t) = t (1 - t/2 + t^2/3 - t^3 (1/4 - t/5 + t^2/6)), |t| < 0.0053
  const float th = t.h;
  const float q = -kf_fma(th, kf_fma(th, 1.0f / 6.0f, -0.2f), 0.25f);
  kf2 P = kf2_add({(float)(1.0 / 3.0), (float)(1.0 / 3.0 - (double)(float)(1.0 / 3.0))}, kf2_two_prod(th, q));
  P = kf2_add({-0.5f, 0.0f}, kf2_mul(t, P));
  P = kf2_add({1.0f, 0.0f}, kf2_mul(t, P));
  const kf2 ln1 = kf2_mul(t, P);
  kf2 r = kf2_mul(ln1, {KF2_INV_LN2_H, KF2_INV_LN2_L});
  r = kf2_add(r, {T.h[j + 32], T.l[j + 32]});
  return kf2_add({(float)e, 0.0f}, r);
}

// 2^y as hi + lo for y in [-125, 127)
WG_FN kf2 kf2_exp2(float y) {
#pragma clang fp contract(off)
  const float n = __builtin_rintf(y);
  const float f = y - n;  // exact, |f| <= 0.5
  const float jf = __builtin_rintf(f * 64.0f);
  const float r = f - jf * (1.0f / 64.0f);  // exact, |r| <= 1/128
  const kf2 x = kf2_mul({r, 0.0f}, {KF2_LN2_H, KF2_LN2_L});  // |x| < 0.0055
  // e^x = 1 + x + x^2/2 + x^3/6 + x^4/24 + x^5/120
  const float xh = x.h;
  const float tail = xh * xh * xh * kf_fma(xh, kf_fma(xh, 1.0f / 120.0f, 1.0f / 24.0f), 1.0f / 6.0f);
  kf2 q = kf2_two_prod(xh, xh);  // (xh + xl)^2 / 2 = xh^2/2 + xh xl + ...
  q.h *= 0.5f;
  q.l = kf_fma(xh, x.l, q.l * 0.5f);
  kf2 s = kf2_add({1.0f, 0.0f}, x);
  s = kf2_add(s, kf2_add(q, {tail, 0.0f}));
  const KfExpTab& T = kf_exp_tab();
  const int j = (int)jf + 32;
  kf2 v = kf2_mul({T.h[j], T.l[j]}, s);
  const float sc = __builtin_bit_cast(float, (u32)((int)n + 127) << 23);  // 2^n, exact (n in [-125, 127])
  v.h *= sc;
  v.l *= sc;
  return v;
}

// The binary32 functions of specs 42 / 43: the double-word path, else the binary64 path above.
WG_FN float kf_log2(float x) {
  if (x >= 0x1p-126f && x - x == 0.0f) {  // normal, finite
    const kf2 r = kf2_log2(x);
    if (kf2_round_ok(r)) return r.h;
  }
  return (float)kd_log2((double)x);
}
WG_FN float kf_ln(float x) {
  if (x >= 0x1p-126f && x - x == 0.0f) {
    const kf2 r = kf2_mul(kf2_log2(x), {KF2_LN2_H, KF2_LN2_L});
    if (kf2_round_ok(r)) return r.h;
  }
  return (float)kd_log((double)x);
}
WG_FN float kf_exp2(float y) {
  if (y >= -125.0f && y < 127.0f) {
    const kf2 r = kf2_exp2(y);
    if (kf2_round_ok(r)) return r.h;
  }
  return (float)kd_exp2((double)y);
}
