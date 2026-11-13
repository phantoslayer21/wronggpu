// SPDX-License-Identifier: MIT
// Exact arithmetic of the page-54b kernels (rr_resolve.hlsl): the f32 rule of 54b.N (one RNE rounding per
// operation, FTZ of inputs and results, results first rounded on the IEEE grid), binary16 fma chains, the
// plain min / max of 54b.N with -0 below +0 (WrongGPU's choice), and correctly rounded f32 exp2 / log2 / rcp.
// The f32 helpers follow kernels/rr_output.hlsl (same build mode, clean/dx12/README.md "Arithmetic mode").
//
// RR_ELEM_EXACT (default 1): exp2 / log2 / rcp correctly rounded (binary64 evaluation with a margin test and a
// double-double decision for the inputs near an f32 rounding midpoint). 0: the hardware approximations
// (v_exp_f32 / v_log_f32, within the 1-ulp allowance of 54b.8) with rcp still correctly rounded. 2: as 1, exp2 and
// rcp through the hooks below (the fast resolve's fixup kernel).
#pragma once
#include "rr_elem_consts.hlsli"
#include "rr_exact_math.hlsli"   // rr_exact_r16: binary64 -> binary16 RNE

#ifndef RR_ELEM_EXACT
#define RR_ELEM_EXACT 1
#endif

static const float RR_FMAX = asfloat(0x7f7fffffu);
static const float RR_FMINN = asfloat(0x00800000u);
static const float RR_HMAX = 65504.0f;
static const float RR_K6 = asfloat(0x3e9a209bu);   // f32(log10 2)
static const float RR_K9 = asfloat(0x40549a78u);   // f32(log2 10)
#define RR_TWO128 asdouble(0u, 0x47f00000u)
#define RR_TWO103 asdouble(0u, 0x46600000u)

// Opaque fence: XOR with a zero the compiler cannot prove (stops regrouping / contraction across helpers).
float fence(float x) { return asfloat(asuint(x) ^ (0u - (wg_grid_ >> 31))); }
double fenced(double x) { uint lo, hi; asuint(x, lo, hi); return asdouble(lo ^ (0u - (wg_grid_ >> 31)), hi); }
float ftz(float x) { uint u = asuint(x); return asfloat((u & 0x7f800000u) == 0u ? u & 0x80000000u : u); }
bool nanf(float x) { return (asuint(x) & 0x7fffffffu) > 0x7f800000u; }

// binary64 -> f32 rounded on the IEEE grid (subnormals included), then FTZ; |x| below FMINN rounds up to FMINN
// only from the midpoint FMINN - 2^-150 on.
float rounded(double x)
{
    double mn = (double)RR_FMINN, boundary = mn - mn * 0.000000059604644775390625;
    if (abs(x) < mn) { uint lo, hi; asuint(x, lo, hi); return asfloat((hi & 0x80000000u) | (abs(x) >= boundary ? 0x00800000u : 0u)); }
    return fence(ftz((float)x));
}
// f32 product: the hardware product (one RNE rounding) except near the flush boundary, where the driver flushes
// before rounding (clean/dx12/README.md): there the exact binary64 product decides.
float M(float a, float b)
{
    a = ftz(a); b = ftz(b);
    float r = fence(a * b);
    if ((asuint(r) & 0x7fffffffu) <= 0x00800000u && a != 0.0f && b != 0.0f && isfinite(a) && isfinite(b))
        r = rounded((double)a * (double)b);
    return r;
}
float A(float a, float b) { return fence(ftz(ftz(a) + ftz(b))); }
float F(float a, float b, float c)
{
    a = ftz(a); b = ftz(b); c = ftz(c);
    float r = fence(ftz(mad(a, b, c)));
    if ((asuint(r) & 0x7fffffffu) <= 0x00800000u && isfinite(a) && isfinite(b) && isfinite(c)) {
        double p = (double)a * (double)b, cc = (double)c, sum = p + cc;
        double bv = sum - p, residual = (p - (sum - bv)) + (cc - bv);
        double mn = (double)RR_FMINN, boundary = mn - mn * 0.000000059604644775390625;
        if (sum != 0 && abs(sum) <= mn) {
            double er = sum < 0 ? -residual : residual;
            bool normal = abs(sum) > boundary || (abs(sum) == boundary && er >= 0);
            r = asfloat((sum < 0 ? 0x80000000u : 0u) | (normal ? 0x00800000u : 0u));
        }
    }
    return r;
}
float S(float x) { return nanf(x) ? 0.0f : min(max(ftz(x), 0.0f), 1.0f); }
float H(float x) { return fence(wg_h2f(wg_f2h(x))); }                       // binary16 rounding (RNE)
float HF(float a, float b, float c) { return fence(wg_h2f(wg_fma_f16(a, b, c))); }   // fma16

// Plain min / max of 54b.N: the non-NaN operand when exactly one is NaN; -0 below +0.
float pmin(float a, float b)
{
    bool lt = b < a || (b == a && (asuint(b) & 0x80000000u) != 0u && (asuint(a) & 0x80000000u) == 0u);
    return (nanf(a) || lt) ? b : a;
}
float pmax(float a, float b)
{
    bool gt = b > a || (b == a && (asuint(b) & 0x80000000u) == 0u && (asuint(a) & 0x80000000u) != 0u);
    return (nanf(a) || gt) ? b : a;
}
float min_nan(float a, float b) { return (nanf(a) || nanf(b)) ? asfloat(0x7fc00000u) : pmin(a, b); }

// ---- correctly rounded reciprocal: IEEE binary64 division, then the f32 rounding above ----------------------------
#if RR_ELEM_EXACT == 2
// 2: the same correctly rounded values from the including file's faster equivalents (rr_resolve_fast.hlsl:
// rr_x_hook = EXP2_CR and rr_rcp_hook = RCP for every f32 input, proven exhaustively by gpu/resolve_fast.py).
float rr_x_hook(float x);
float rr_rcp_hook(float x);
#endif
float RCP(float x)
{
#if RR_ELEM_EXACT == 2
    return rr_rcp_hook(x);
#endif
    x = ftz(x);
    if (nanf(x)) return x;
    if (x == 0.0f) return asfloat((asuint(x) & 0x80000000u) | 0x7f800000u);
    return rounded(1.0 / (double)x);
}

// ---- double-double -------------------------------------------------------------------------------------------------
struct rr_dd { double h; double l; };
rr_dd dd_make(double h, double l) { rr_dd r; r.h = h; r.l = l; return r; }
rr_dd two_sum(double a, double b) { double s = fenced(a + b); double bb = fenced(s - a); return dd_make(s, (a - (s - bb)) + (b - bb)); }
rr_dd quick_sum(double a, double b) { double s = fenced(a + b); return dd_make(s, b - (s - a)); }
rr_dd two_prod(double a, double b) { double p = fenced(a * b); return dd_make(p, fma(a, b, -p)); }
rr_dd dd_add(rr_dd a, rr_dd b)
{
    rr_dd s = two_sum(a.h, b.h), t = two_sum(a.l, b.l);
    s.l = s.l + t.h; s = quick_sum(s.h, s.l);
    s.l = s.l + t.l; return quick_sum(s.h, s.l);
}
rr_dd dd_mul(rr_dd a, rr_dd b)
{
    rr_dd p = two_prod(a.h, b.h);
    p.l = p.l + (a.h * b.l + a.l * b.h);
    return quick_sum(p.h, p.l);
}

// 2^m for a binary64 m with |m| < 200 whose fraction r = m - round(m) is exact: double-double, relative error
// below 2^-100 (Taylor of exp(r ln 2) to degree 24 in double-double, |r ln 2| <= 0.347).
rr_dd exp2_dd(double m)
{
    int ni = (int)(m + (m >= 0.0 ? 0.5 : -0.5));
    double n = (double)ni;
    double r = m - n;                                    // exact (|r| <= 1/2 up to the rounding of m + 1/2)
    rr_dd t = two_prod(r, RR_LN2_HI);
    t.l = t.l + r * RR_LN2_LO;
    t = quick_sum(t.h, t.l);
    double ch, cl;
    rr_inv_fact_dd(24u, ch, cl);
    rr_dd s = dd_make(ch, cl);
    [loop] for (int k = 23; k >= 0; --k) {
        rr_inv_fact_dd((uint)k, ch, cl);
        s = dd_add(dd_mul(s, t), dd_make(ch, cl));
    }
    // scale by 2^n (exact, two steps keep each factor normal)
    int n1 = ni / 2, n2 = ni - n1;
    double f1 = asdouble(0u, (uint)(n1 + 1023) << 20), f2 = asdouble(0u, (uint)(n2 + 1023) << 20);
    return dd_make(s.h * f1 * f2, s.l * f1 * f2);
}

// f32 neighbours in value order (finite nonzero f, FTZ grid: the step below FMINN goes to 0, not used here)
double up_neighbour(float f)
{
    uint u = asuint(f);
    if (u == 0x7f7fffffu) return RR_TWO128;
    return (double)asfloat((u & 0x80000000u) != 0u ? u - 1u : u + 1u);
}
double down_neighbour(float f)
{
    uint u = asuint(f);
    if (u == 0xff7fffffu) return -RR_TWO128;
    return (double)asfloat((u & 0x80000000u) != 0u ? u + 1u : u - 1u);
}

// Low 29 bits of the binary64 significand within 2^12 of the f32 midpoint pattern (relative distance < 2^-40).
bool near_mid(double y)
{
    uint lo, hi; asuint(y, lo, hi);
    int low = (int)(lo & 0x1fffffffu) - 0x10000000;
    return abs(low) <= 4096;
}

// Round the double-double value V (positive, normal f32 range or above) correctly to f32.
float round_dd(rr_dd V)
{
    float f = (float)V.h;
    double fd = (double)f;
    if (isinf(f)) {                                       // V.h rounded up to 2^128: decide at FMAX + 2^103
        double mid = RR_TWO128 - RR_TWO103;
        return (V.h - mid) + V.l > 0 ? f : RR_FMAX;
    }
    double d = (V.h - fd) + V.l;
    if (d > 0) { double mid = 0.5 * (fd + up_neighbour(f)); if ((V.h - mid) + V.l > 0) f = (float)up_neighbour(f); }
    else if (d < 0) { double mid = 0.5 * (fd + down_neighbour(f)); if ((V.h - mid) + V.l < 0) f = (float)down_neighbour(f); }
    return f;
}

// ---- exp2 ----------------------------------------------------------------------------------------------------------
float exp2_slow_in_range(float x)                         // -126 <= x < 128: double-double decision
{
    return round_dd(exp2_dd((double)x));
}
double exp2_fast64(float x, out double scaledN)
{
    float n = round(x);
    float r = x - n;                                      // exact
    double t = (double)r, p = rr_exp2_c(13u);
    [unroll] for (int k = 12; k >= 0; --k) p = fma(p, t, rr_exp2_c((uint)k));
    scaledN = asdouble(0u, (uint)((int)n + 1023) << 20);
    return p * scaledN;
}
// mode 0: fast path with the margin test; 1: always the double-double decision (validation); out hard flag
float EXP2_CR(float xin, uint mode, out bool hard)
{
    float x = ftz(xin);
    hard = false;
    if (nanf(x)) return asfloat(asuint(x) | 0x00400000u);
    if (x >= 128.0f) return asfloat(0x7f800000u);
    if (x < -126.0f) return 0.0f;                          // 2^x below FMINN - 2^-150: flushed (x = -126 - 2^-17 already)
    double sN;
    double y = exp2_fast64(x, sN);
    hard = near_mid(y);
    if (mode == 0u && !hard) return fence(ftz((float)y));
    return fence(exp2_slow_in_range(x));
}

// ---- log2 ----------------------------------------------------------------------------------------------------------
double log2_fast64(float x)
{
    uint u = asuint(x);
    int e = (int)((u >> 23) & 255u) - 127;
    uint mb = (u & 0x007fffffu) | 0x3f800000u;            // m in [1, 2)
    if ((u & 0x007fffffu) > 0x3504f3u) { mb -= 0x00800000u; e += 1; }   // m in [sqrt(1/2), sqrt(2))
    double m = (double)asfloat(mb);
    double s = (m - 1.0) / (m + 1.0);                     // m - 1, m + 1 exact; IEEE division
    double s2 = s * s, p = rr_log_c(12u);
    [unroll] for (int k = 11; k >= 0; --k) p = fma(p, s2, rr_log_c((uint)k));
    double lnm = s * p;
    return (double)e + lnm * RR_INV_LN2_HI;
}
float log2_decide(float x, double y)                      // correct rounding from y (|error| << 1 f32 ulp)
{
    float f = (float)y;
    double fd = (double)f, xd = (double)x;
    if (y >= fd) {
        double mid = 0.5 * (fd + up_neighbour(f));
        rr_dd p = exp2_dd(mid);                            // log2 x > mid  <=>  x > 2^mid
        if ((xd - p.h) - p.l > 0) f = (float)up_neighbour(f);
    } else {
        double mid = 0.5 * (fd + down_neighbour(f));
        rr_dd p = exp2_dd(mid);
        if ((xd - p.h) - p.l < 0) f = (float)down_neighbour(f);
    }
    return f;
}
float LOG2_CR(float xin, uint mode, out bool hard)
{
    float x = ftz(xin);
    hard = false;
    if (nanf(x)) return asfloat(asuint(x) | 0x00400000u);
    if (x == 0.0f) return asfloat(0xff800000u);
    if ((asuint(x) & 0x80000000u) != 0u) return asfloat(0x7fc00000u);
    if (isinf(x)) return x;
    double y = log2_fast64(x);
    if (y == 0.0) return 0.0f;                             // x = 1
    hard = near_mid(y);
    if (mode == 0u && !hard) return fence((float)y);
    return fence(log2_decide(x, y));
}

float X(float x)
{
#if RR_ELEM_EXACT == 2
    return rr_x_hook(x);
#elif RR_ELEM_EXACT
    bool h; return EXP2_CR(x, 0u, h);
#else
    return fence(ftz(exp2(ftz(x))));
#endif
}
float L(float x)
{
#if RR_ELEM_EXACT
    bool h; return LOG2_CR(x, 0u, h);
#else
    return fence(ftz(log2(ftz(x))));
#endif
}
