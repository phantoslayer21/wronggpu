// SPDX-License-Identifier: GPL-3.0-only
// D3D12 port of model K's exact scalar FP32 / FP16 / FP64 helpers (clean/k/kernels/k_fp.h; specs 42, 43). WrongGPU's own
// code. Same names and the same bits as the HIP helpers on gfx1201, in the exact arithmetic mode of build_dx12.py
// (clean/dx12/hlsl/wg_math.hlsli: one IEEE operation per written operation, mad() = one v_fma_f32, rcp() = v_rcp_f32,
// no float '/': wg_div is HIP's correctly rounded a / b).
//
// Include after wg_platform.hlsli (wg_math.hlsli) and, in a kernel, call KF_SETUP() right after WG_SETUP().
//
// Representation: an FP16 value (HIP h16) is a float holding that value exactly (f16tof32 / wg_h2f of the bits);
// kh_round(x) is HIP's (h16) rounding of an FP32 value (round to nearest even, v_cvt_f16_f32: wg_f2h; the driver's
// (half) / f32tof16 round toward zero and are never used), kh_bits(x) the bits of such a value for a store. FP16
// arithmetic is done in FP32 and rounded once: a sum, difference or product of two FP16 values rounded to FP32 and then
// to FP16 is the FP16 operation's result (products are exact in FP32; for + and - double rounding is innocuous since
// 24 >= 2 * 11 + 2, Figueroa 1995), kh_fma is wg_fma_f16 (one rounding), kh_div as k_fp.h's argument. This also avoids
// the driver's fusion of half multiply + add (wg_math.hlsli "driver exceptions").
//
// Binary64 (kd_*): add / sub / mul / compare / convert and fma (the D3D11.1 extended doubles, which D3D12 includes in
// D3D12_FEATURE_DATA_D3D12_OPTIONS.DoublePrecisionFloatShaderOps): build_dx12.py marks ports with binary64 fma or
// division "fp64": "ext" and the devices refuse them on an adapter without doubles. HLSL has no binary64 sqrt / rint /
// constexpr: the constants and tables of k_fp.h come as bits
// from k_fp_tables.hlsli (clean/k/dx12/gen_k_fp_tables.py: the same binary64 series and roundings), kd_exp2's rint is
// done in FP32 (its argument is always a binary32 value, see there), and kf_rsqrt's binary64 fallback is replaced by an
// exact power-of-four scaling (see there).
//
// Opaque barriers (kf_opq): the gfx1201 D3D12 shader compiler may contract a product into a following addition or
// subtraction (one fma: one rounding instead of HIP's two; seen in the first port check of the input stage, about one
// FP16 value in 10^5 differing by one ulp), regroup chains of multiplications by constant / uniform / shared factors,
// and fold a product into a later fused operation (wg_math.hlsli). Wherever HIP rounds a product that then meets an
// addition (and the product is not exact, e.g. a power-of-two scaling), the product goes through kf_opq (an XOR with a
// zero the compiler cannot prove: wg_grid_ >> 31, the launch's workgroup count is below 2^31; one VALU operation).
// kf_exp2 applies it to its argument itself (its callers pass products).
//
// Differences from HIP (as for every D3D12 port, wg_math.hlsli): FP32 denormals are flushed to zero here (HIP keeps
// them); none of the input stage's values is an FP32 denormal for the frames the port checks run (clean/k/dx12/
// k_port_check.py), and FP16 denormals are kept as in HIP.
#pragma once

#include "k_fp_tables.hlsli"

static uint kf_opq_z_;                         // 0, opaque to the compiler (KF_SETUP)
static float kf_nz_;                           // -0.0 ^ the opaque zero: -0.0 at run time, unknown to the compiler (KF_SETUP, one SGPR)
#define KF_SETUP() (kf_opq_z_ = wg_grid_ >> 31, kf_nz_ = asfloat(0x80000000u ^ (wg_grid_ >> 31)))
float kf_opq(float x) { return asfloat(asuint(x) ^ kf_opq_z_); }
float2 kf_opq2(float2 x) { return asfloat(asuint(x) ^ kf_opq_z_); }
float4 kf_opq4(float4 x) { return asfloat(asuint(x) ^ kf_opq_z_); }
// A product rounded as its own instruction: kf_pm(a, b) = RN(a b) = fma(a, b, -0.0 ^ opaque zero) bit for bit (the exact
// product plus -0.0 rounds once: a zero product keeps its sign, NaN / infinities / FTZ as the multiply), the addend
// unknown to the compiler: no fusion into a following addition, no regrouping. One v_fma_f32 with the SGPR addend instead
// of the multiply and the xor of kf_opq(a * b) (k_common.hlsli wgk_fm is the FP16 twin). KF_PM = 0: kf_opq(a * b).
#ifndef KF_PM
#define KF_PM 1
#endif
#if KF_PM
float kf_pm(float a, float b) { return mad(a, b, kf_nz_); }
float3 kf_pm3(float3 a, float3 b) { return mad(a, b, (float3)kf_nz_); }
float4 kf_pm4(float4 a, float4 b) { return mad(a, b, (float4)kf_nz_); }
#else
float kf_pm(float a, float b) { return kf_opq(a * b); }
float3 kf_pm3(float3 a, float3 b) { return asfloat(asuint(a * b) ^ kf_opq_z_); }
float4 kf_pm4(float4 a, float4 b) { return kf_opq4(a * b); }
#endif

// ---- elementary steps -----------------------------------------------------------------------------------------------

float kf_fma(float a, float b, float c) { return mad(a, b, c); }            // v_fma_f32
float kf_bits(uint b) { return asfloat(b); }
float kf_floor(float x) { return floor(x); }
float kf_trunc(float x) { return trunc(x); }
float kf_rni(float x) { return round(x); }                                  // v_rndne_f32 (ties to even)
// max / min of specs 42 / 43: IEEE maxNum / minNum (v_max_num_f32 / v_min_num_f32), as __builtin_fmaxf / fminf
float kf_max(float a, float b) { return max(a, b); }
float kf_min(float a, float b) { return min(a, b); }
int ki_min(int a, int b) { return a < b ? a : b; }
int ki_max(int a, int b) { return a > b ? a : b; }
// HIP's x - x == 0 test as bits: the D3D12 shader compiler folds x - x to 0 (seen for float and double: an infinite
// input passed as finite in the first probe, clean/k/dx12/probe/kfp_check.py)
bool kf_finite(float x) { return (asuint(x) & 0x7f800000u) != 0x7f800000u; }
bool kd_isnan(double x) { uint lo, hi; asuint(x, lo, hi); return (hi & 0x7ff00000u) == 0x7ff00000u && ((hi & 0xfffffu) | lo) != 0u; }
bool kd_finite(double x) { uint lo, hi; asuint(x, lo, hi); return (hi & 0x7ff00000u) != 0x7ff00000u; }

// wg_div (HIP's correctly rounded a / b) with its common case first: both operands normal and finite and the quotient's
// exponent in range, where wg_div's result is the scaled quotient q with the exponent difference added to its exponent
// field (the same q: the same operations as wg_div); every other operand pair takes wg_div itself in a cold block. Equal
// to wg_div for every input (probe: build\k-d3d12\L2work\probe\kdiv_check.py); about half of wg_div's instructions and none of
// its selects on the common path.
float kf_div(float a, float b)
{
    const uint ua = asuint(a), ub = asuint(b);
    const uint ea = (ua >> 23) & 255u, eb = (ub >> 23) & 255u;
    const float as = asfloat((ua & 0x807fffffu) | 0x3f800000u), bs = asfloat((ub & 0x807fffffu) | 0x3f800000u);
    float y = rcp(bs);
    const float e = mad(-bs, y, 1.0f);
    y = mad(e, y, y);
    float q = as * y;
    float r = mad(-bs, q, as);
    q = mad(r, y, q);
    r = mad(-bs, q, as);
    q = mad(r, y, q);
    const int d = (int)ea - (int)eb;
    const uint en = ((asuint(q) >> 23) & 255u) + (uint)d;
    float res;
    [branch] if (ea - 1u < 254u && eb - 1u < 254u && en - 1u < 254u)
        res = asfloat(asuint(q) + ((uint)d << 23));
    else
        res = wg_div(a, b);
    return res;
}

// R(t): round half away from zero (specs 42 / 43), exact for every binary32 t
float kf_R(float t)
{
    const float a = abs(t);
    const float fl = floor(a);
    const float r = (a - fl) >= 0.5f ? fl + 1.0f : fl;   // a - fl is exact
    return t < 0.0f ? -r : r;
}
float kf_sat(float x) { return isnan(x) ? 0.0f : (x < 0.0f ? 0.0f : (x > 1.0f ? 1.0f : x)); }

// ---- FP16 values (floats holding FP16 values exactly) ---------------------------------------------------------------

uint kh_bits(float h) { return wg_f2h(h); }                 // exact for an FP16 value (NaN payload as v_cvt_f16_f32)
float kh_round(float x) { return wg_h2f(wg_f2h(x)); }       // HIP kh_round: (h16) x, nearest even
uint kh_round_bits(float x) { return wg_f2h(x); }           // the bits of kh_round(x)
float kh_add(float a, float b) { return kh_round(a + b); }
float kh_sub(float a, float b) { return kh_round(a - b); }
float kh_mul(float a, float b) { return kh_round(a * b); }
float kh_fma(float a, float b, float c) { return wg_h2f(wg_fma_f16(a, b, c)); }   // v_fma_f16: one rounding
// a / b of FP16 values rounded once to FP16: the correctly rounded FP32 quotient rounded to FP16 (k_fp.h kh_div)
float kh_div(float a, float b) { return kh_round(wg_div(a, b)); }
float kh_floor(float x) { return floor(x); }                // exact on FP16 values
float kh_rni(float x) { return round(x); }
float kh_sat(float x) { return isnan(x) ? 0.0f : (x < 0.0f ? 0.0f : (x > 1.0f ? 1.0f : x)); }
float kh_maxnum(float a, float b) { return max(a, b); }

// packed FP16 (two independent values; each element exactly as the scalar helper, as HIP's v_pk_* do)
float2 kh2(float a, float b) { return float2(a, b); }
float2 kh2_splat(float a) { return float2(a, a); }
float2 kh2_add(float2 a, float2 b) { return float2(kh_add(a.x, b.x), kh_add(a.y, b.y)); }
float2 kh2_sub(float2 a, float2 b) { return float2(kh_sub(a.x, b.x), kh_sub(a.y, b.y)); }
float2 kh2_mul(float2 a, float2 b) { return float2(kh_mul(a.x, b.x), kh_mul(a.y, b.y)); }
float2 kh2_fma(float2 a, float2 b, float2 c) { return float2(kh_fma(a.x, b.x, c.x), kh_fma(a.y, b.y, c.y)); }
float2 kh2_abs(float2 a) { return abs(a); }
float2 kh2_rni(float2 a) { return round(a); }
float2 kh2_maxnum(float2 a, float2 b) { return max(a, b); }
// k_fp.h kh2_sat_nz: maxNum(x, 0) then minNum(., 1) per element (callers never pass -0)
float2 kh2_sat_nz(float2 a) { return min(max(a, 0.0f), 1.0f); }
// k_fp.h kh2_gtmask as a bool2: true where h(b - a) is negative and not NaN (IEEE a > b, except a = +0, b = -0: true)
bool2 kh2_gtmask(float2 a, float2 b)
{
    const float2 d = float2(kh_sub(b.x, a.x), kh_sub(b.y, a.y));
    return bool2((asuint(d.x) >> 31) != 0u && !isnan(d.x), (asuint(d.y) >> 31) != 0u && !isnan(d.y));
}
float2 kh2_bsel(bool2 m, float2 a, float2 b) { return float2(m.x ? a.x : b.x, m.y ? a.y : b.y); }

// ---- double-precision functions --------------------------------------------------------------------------------------

double kd_fma(double a, double b, double c) { return fma(a, b, c); }       // extended doubles
double kd_from_bits(uint lo, uint hi) { return asdouble(lo, hi); }

// 1 / d for d in [1.7, 2.5]: the FP32 reciprocal estimate (v_rcp_f32 of RN32(d), as HIP) and two Newton steps
double kd_rcp_near2(double d)
{
    double q = (double)rcp((float)d);
    q = kd_fma(q, kd_fma(-d, q, 1.0L), q);
    q = kd_fma(q, kd_fma(-d, q, 1.0L), q);
    return q;
}

// k_fp.h kd_log_core: x = 2^e m, m in (1/sqrt 2, sqrt 2], ln m = 2 s (1 + s^2/3 + ... + s^22/23), s = (m - 1)/(m + 1)
double kd_log_core(double x, out int e_out)
{
    uint lo, hi;
    asuint(x, lo, hi);
    int e = (int)((hi >> 20) & 0x7ffu) - 1023;
    double m = asdouble(lo, (hi & 0x000fffffu) | 0x3ff00000u);              // [1, 2)
    if (m > KD_SQRT2) {
        m = m * 0.5L;
        e += 1;
    }
    const double s = (m - 1.0L) * kd_rcp_near2(m + 1.0L);                  // m - 1 and m + 1 are exact
    const double s2 = s * s;
    double p = KD_INV23;
    [unroll] for (int k = 10; k >= 0; --k) p = kd_fma(p, s2, asdouble(KD_INV_ODD[k].x, KD_INV_ODD[k].y));
    e_out = e;
    return 2.0L * s * p;
}

double kd_log_special(double x, out bool special)
{
    special = true;
    if (kd_isnan(x) || x < 0.0L) return asdouble(0u, 0x7ff80000u);
    if (x == 0.0L) return -asdouble(0u, 0x7ff00000u);
    if (!kd_finite(x)) return x;                                            // +inf
    special = false;
    return 0.0L;
}

// ln x
double kd_log(double x)
{
    bool sp;
    const double v = kd_log_special(x, sp);
    if (sp) return v;
    int e;
    const double l = kd_log_core(x, e);
    return kd_fma((double)e, KD_LN2, l);
}

// log2 x = e + ln(m) / ln 2
double kd_log2(double x)
{
    bool sp;
    const double v = kd_log_special(x, sp);
    if (sp) return v;
    int e;
    const double l = kd_log_core(x, e);
    return kd_fma(l, KD_INV_LN2, (double)e);
}

double kd_ldexp(double v, int n)                                            // v * 2^n, n in [-2044, 2046]
{
    [unroll] for (int i = 0; i < 2; ++i) {
        const int k = n > 1023 ? 1023 : (n < -1022 ? -1022 : n);
        v = v * asdouble(0u, (uint)(k + 1023) << 20);
        n -= k;
    }
    return v;
}

// 2^y. Precondition: y is a binary32 value (every caller passes (double) of a float: kf_exp2), so rint(y) is the FP32
// v_rndne of (float) y, which is exact (HLSL has no binary64 rint).
double kd_exp2(double y)
{
    if (kd_isnan(y)) return y;
    if (y > 1100.0L) return asdouble(0u, 0x7ff00000u);
    if (y < -1100.0L) return 0.0L;
    const double n = (double)round((float)y);
    const double r = (y - n) * KD_LN2;                                      // |r| <= 0.347
    double p = 1.0L;
    [unroll] for (int k = 16; k >= 1; --k) p = kd_fma(p, r * asdouble(KD_INV_K[k].x, KD_INV_K[k].y), 1.0L);
    return kd_ldexp(p, (int)n);
}

// 1/sqrt(x) rounded once to FP32 (spec 43 6.8). HIP: for x >= 1 the hardware estimate v_rsq_f32 and exact rounding
// tests in binary64 (as here); otherwise kd_rsqrt32 = (float)(1.0 / sqrt((double)x)), the correctly rounded value.
// HLSL has no binary64 sqrt: a normal x < 1 is scaled by 4^k into [1, 4) (exact), the tested result scaled back by 2^k
// (exact: the result is normal), which is the correctly rounded value too. Zero (and FP32 denormals, flushed) -> +-inf
// as 1 / +-0, +inf -> 0, NaN -> quieted, negative -> the canonical NaN. Port check: the output stage's (G2).
float kf_rsqrt_ge1(float x)
{
    float y = rsqrt(x);                                                     // v_rsq_f32
    const double xd = (double)x;
    [unroll] for (int it = 0; it < 3; ++it) {
        const uint b = asuint(y);
        const double yd = (double)y;
        const double ulp = (double)asfloat((b & 0x7F800000u) - (23u << 23));
        const double up = yd + 0.5L * ulp;
        const double lo = yd - ((b & 0x7FFFFFu) == 0u ? 0.25L : 0.5L) * ulp;   // below a power of two the spacing halves
        if (kd_fma(up * up, xd, -1.0L) < 0.0L) {
            y = asfloat(b + 1u);
        } else if (kd_fma(lo * lo, xd, -1.0L) > 0.0L) {
            y = asfloat(b - 1u);
        } else {
            return y;
        }
    }
    return y;
}
float kf_rsqrt(float x)
{
    if (x >= 1.0f && kf_finite(x)) return kf_rsqrt_ge1(x);
    if (isnan(x)) return asfloat(asuint(x) | 0x400000u);
    if (x == 0.0f) return asfloat((asuint(x) & 0x80000000u) | 0x7f800000u);
    if (x < 0.0f) return asfloat(0x7fc00000u);
    if (!kf_finite(x)) return 0.0f;                                         // +inf
    const int e = (int)((asuint(x) >> 23) & 255u) - 127;                    // x normal in [2^-126, 1): e in [-126, -1]
    const int k = (1 - e) >> 1;                                             // 4^k x in [1, 4)
    const float xs = x * asfloat((uint)(2 * k + 127) << 23);
    return kf_rsqrt_ge1(xs) * asfloat((uint)(k + 127) << 23);
}

// ---- correctly rounded binary32 log2 / ln / 2^x in FP32 double-word arithmetic (k_fp.h: Ziv test, else binary64) ----

struct kf2 { float h, l; };
kf2 kf2_make(float h, float l) { kf2 r; r.h = h; r.l = l; return r; }

// Error-free transformations: the sums go through kf_opq, else the D3D12 compiler simplifies (a + b) - a to b and the
// like (the first probe found the double-word results off by one ulp in 16 % of 2^x and 0.1 % of log2 inputs)
kf2 kf2_two_sum(float a, float b)
{
    const float s = kf_opq(a + b);
    const float bb = kf_opq(s - a);
    return kf2_make(s, (a - (s - bb)) + (b - bb));
}
kf2 kf2_fast_two_sum(float a, float b)                                      // |a| >= |b| or a == 0
{
    const float s = kf_opq(a + b);
    return kf2_make(s, b - (s - a));
}
kf2 kf2_two_prod(float a, float b)
{
    const float p = kf_pm(a, b);
    return kf2_make(p, kf_fma(a, b, -p));
}
kf2 kf2_add(kf2 a, kf2 b)
{
    kf2 s = kf2_two_sum(a.h, b.h);
    s.l = s.l + (a.l + b.l);
    return kf2_fast_two_sum(s.h, s.l);
}
kf2 kf2_mul(kf2 a, kf2 b)
{
    kf2 p = kf2_two_prod(a.h, b.h);
    p.l = kf_fma(a.h, b.l, kf_fma(a.l, b.h, p.l));
    return kf2_fast_two_sum(p.h, p.l);
}

// Ziv test: is RN32(v) = r.h for every v with |v - (r.h + r.l)| <= 2^-38 |r.h| ?
bool kf2_round_ok(kf2 r)
{
    const uint b = asuint(r.h);
    const uint ex = (b >> 23) & 0xFFu;
    if (ex - 40u > 214u) return false;                                      // ex < 40 or 255: tiny, subnormal, inf / NaN: exact path
    // half = 0.5 ulp, or 0.25 ulp for a power of two whose low part points toward zero (below it the spacing halves).
    // Bit form: (sign(r.l) xor sign(r.h)) and (mantissa - 1 borrows), bit 31. It differs from (r.l < 0) != (r.h < 0) only
    // for r.l = -0, where |r.l| + err = err < 2^-37 |r.h| is below either half: the same answer.
    const uint s = ((asuint(r.l) ^ b) & ((b & 0x7FFFFFu) - 1u)) >> 31;
    const float half = asfloat((ex - 24u - s) << 23);
    const float err = abs(r.h) * asfloat(0x2C800000u);                      // 2^-38
    return abs(r.l) + err < half;
}

// log2 x as hi + lo for finite normal x > 0
kf2 kf2_log2(float x)
{
    const uint b = asuint(x);
    int e = (int)((b >> 23) & 0xFFu) - 127;
    float m = asfloat((b & 0x7FFFFFu) | 0x3F800000u);                       // [1, 2)
    if (m >= 1.5f) {
        m = m * 0.5f;
        e += 1;
    }
    const int j = (int)round((m - 1.0f) * 128.0f);                          // -32 .. 64
    const float inv = asfloat(KF_LOG_INV[j + 32]);
    const kf2 p = kf2_two_prod(m, inv);                                     // m inv = 1 + t exactly
    const kf2 t = kf2_two_sum(p.h - 1.0f, p.l);                             // p.h - 1 is exact
    // ln(1 + t) = t (1 - t/2 + t^2/3 - t^3 (1/4 - t/5 + t^2/6)), |t| < 0.0053
    const float th = t.h;
    const float q = -kf_fma(th, kf_fma(th, KF_SIXTH, -0.2f), 0.25f);
    kf2 P = kf2_add(kf2_make(KF_THIRD_H, KF_THIRD_L), kf2_two_prod(th, q));
    P = kf2_add(kf2_make(-0.5f, 0.0f), kf2_mul(t, P));
    P = kf2_add(kf2_make(1.0f, 0.0f), kf2_mul(t, P));
    const kf2 ln1 = kf2_mul(t, P);
    kf2 r = kf2_mul(ln1, kf2_make(KF2_INV_LN2_H, KF2_INV_LN2_L));
    r = kf2_add(r, kf2_make(asfloat(KF_LOG_H[j + 32]), asfloat(KF_LOG_L[j + 32])));
    return kf2_add(kf2_make((float)e, 0.0f), r);
}

// 2^y as hi + lo for y in [-125, 127)
kf2 kf2_exp2(float y)
{
    const float n = round(y);
    const float f = y - n;                                                  // exact, |f| <= 0.5
    const float jf = round(f * 64.0f);
    const float r = f - jf * KF_1_64;                                       // exact, |r| <= 1/128
    const kf2 x = kf2_mul(kf2_make(r, 0.0f), kf2_make(KF2_LN2_H, KF2_LN2_L));   // |x| < 0.0055
    // e^x = 1 + x + x^2/2 + x^3/6 + x^4/24 + x^5/120; HIP's order ((xh xh) xh) poly, each product rounded
    const float xh = x.h;
    const float x2 = kf_pm(xh, xh), x3 = kf_pm(x2, xh);
    const float tail = kf_pm(x3, kf_fma(xh, kf_fma(xh, KF_1_120, KF_1_24), KF_SIXTH));   // not fused into q.h + tail
    kf2 q = kf2_two_prod(xh, xh);
    q.h = q.h * 0.5f;
    q.l = kf_fma(xh, x.l, q.l * 0.5f);
    kf2 s = kf2_add(kf2_make(1.0f, 0.0f), x);
    s = kf2_add(s, kf2_add(q, kf2_make(tail, 0.0f)));
    const int j = (int)jf + 32;
    kf2 v = kf2_mul(kf2_make(asfloat(KF_EXP_H[j]), asfloat(KF_EXP_L[j])), s);
    const float sc = asfloat((uint)((int)n + 127) << 23);                   // 2^n, exact (n in [-125, 127])
    v.h = v.h * sc;
    v.l = v.l * sc;
    return v;
}

// The binary32 functions of specs 42 / 43: the double-word path, else the binary64 path above.
float kf_log2(float x)
{
    if (asuint(x) - 0x00800000u < 0x7f000000u) {                            // positive, normal, finite (bits 0x00800000 .. 0x7f7fffff)
        const kf2 r = kf2_log2(x);
        if (kf2_round_ok(r)) return r.h;
    }
    return (float)kd_log2((double)x);
}
float kf_ln(float x)
{
    if (asuint(x) - 0x00800000u < 0x7f000000u) {
        const kf2 r = kf2_mul(kf2_log2(x), kf2_make(KF2_LN2_H, KF2_LN2_L));
        if (kf2_round_ok(r)) return r.h;
    }
    return (float)kd_log((double)x);
}
float kf_exp2(float y)
{
    // callers pass products (log2(z) c, e 9 c) that HIP rounds before kf2_exp2's y - rint(y); the gfx1201 ISA keeps
    // them apart without a kf_opq here (G1's precaution link, dropped in speed round 1 after the ISA check)
    if (y >= -125.0f && y < 127.0f) {
        const kf2 r = kf2_exp2(y);
        if (kf2_round_ok(r)) return r.h;
    }
    return (float)kd_exp2((double)y);
}
