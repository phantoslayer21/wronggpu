// SPDX-License-Identifier: MIT
// Binary32 arithmetic of spec 52.3 / 52.7.2 / 54b.N on the gfx1201 D3D12 path (compile only with gpu/build.py):
// round to nearest even, f32 subnormal operands and results flushed to zero with the sign kept, where a result that
// rounds up to the smallest normal stays (round first, then flush). Correctly rounded log2, exp2, rcp, sqrt
// (the reference choice of 52.3). Every result is fenced so that the driver cannot regroup or fuse across it.
//
// Two builds of the same operations (RR_IN_EXACT):
//   0, the fast pass: branch-free. Native f32 add / sub are exact under the rule above. Native mul / fma flush the
//      interval [mn (1 - 2^-24), mn) before rounding (clean/dx12/README.md): a mul / fma with a zero result that is
//      not an exact zero marks the token (in_flag). log2 / exp2: float-float paths (log2_ff / exp2_ff); a result
//      they cannot round with certainty marks the token. rcp, sqrt and the unorm / snorm decodes are exact in f32
//      (hardware value corrected once with an exact fma remainder; exhaustively checked, gpu/input_stage.py --xcheck).
//   1, the replay pass for the marked tokens: mul / fma zero results recomputed in binary64, log2 / exp2 through the
//      float-float path with the binary64 series as fallback; a binary64 value within 2^-49 (relative) of an f32
//      rounding midpoint is counted in in_ziv: only there the rounding is not proven correct.
#pragma once

static uint in_ziv = 0u;                  // log2 / exp2 results whose rounding is not proven (diagnostics)
static uint in_flag = 0u;                 // fast pass: this token needs the exact replay
#ifndef RR_IN_EXACT
#define RR_IN_EXACT 1
#endif
#ifndef RR_IN_FMBR
#define RR_IN_FMBR 1      // fast pass: the exact-zero test of FM only behind a (rare) zero result (0: branch-free)
#endif

#ifndef RR_IN_FZT
#define RR_IN_FZT 0
#endif
// clean/rr-mp1-input I2 (RR_IN_FZT 1): the fast pass's zero-result tests of M / FM as r == 0.0f instead of the bit test.
// Same set of r: +-0 pass both, NaN fails both, and the D3D12 path flushes f32 mul / fma results, so r is never
// subnormal (where a flushing compare could differ). Even then only extra tokens would be marked, and the exact replay
// writes the bits the fast pass gives an unmarked token: the outputs keep their bits either way.
#if RR_IN_FZT
#define IN_ZT(r) ((r) == 0.0f)
#else
#define IN_ZT(r) ((asuint(r) & 0x7fffffffu) == 0u)
#endif
#ifndef RR_IN_LITE
#define RR_IN_LITE 0      // opt-in near-exact features (tier B, gpu/input_stage.py variant "lite"): no zero-result tests,
#endif                    // no replay marks; with RR_IN_FAST 29, RR_IN_HCERT 0, RR_IN_L2CERT 0 (plain hardware f32 math)
#ifndef RR_IN_FAND
#define RR_IN_FAND 0
#endif
#ifndef RR_IN_FAST
#define RR_IN_FAST 0      // timing experiments only (gpu/input_bench.py --fast): 1 hw log2/exp2, 4 hw sqrt,
#endif                    // 8 no zero-result repair in M/FM, 16 no fences; 2 hw rcp (opt-in WG_RR_INPUT=hw)
uint in_zb() { return 0u - (wg_grid_ >> 31); }
#if RR_IN_FAST & 16
float FN(float x) { return x; }
#elif RR_IN_FAND
// the fence as an AND with an opaque all-ones word: the same identity, but v_and_b32 can pair (VOPD) where v_xor cannot
float FN(float x) { return asfloat(asuint(x) & ~in_zb()); }
#else
float FN(float x) { return asfloat(asuint(x) ^ in_zb()); }
#endif
float in_ftz(float x) { const uint u = asuint(x); return asfloat((u & 0x7f800000u) == 0u ? (u & 0x80000000u) : u); }
float in_nan0(float x) { return isnan(x) ? 0.0f : x; }

#define IN_MN asfloat(0x00800000u)
// mn (1 - 2^-24): the midpoint between the largest subnormal and mn; from it up IEEE rounding gives mn
#define IN_BOUNDARY_LO 0xe0000000u
#define IN_BOUNDARY_HI 0x380fffffu
double in_boundary() { return asdouble(IN_BOUNDARY_LO, IN_BOUNDARY_HI); }

// binary64 -> binary32: RNE, then the flush; the interval rounding to mn is explicit.
float in_round(double x)
{
    const double mn = (double)IN_MN;
    const bool neg = x < 0.0;
    const double ax = neg ? -x : x;
    if (ax < mn) {
        uint lo, hi;
        asuint(x, lo, hi);
        return asfloat((hi & 0x80000000u) | (ax >= in_boundary() ? 0x00800000u : 0u));
    }
    return FN(in_ftz((float)x));
}

float A(float a, float b) { return FN(a + b); }
float S(float a, float b) { return FN(a - b); }
#ifndef RR_IN_MFMA
#define RR_IN_MFMA 1      // round 13: the fenced product as fma(a, b, opaque -0) (one VALU instead of mul + xor)
#endif
float M(float a, float b)
{
#if RR_IN_MFMA && !(RR_IN_FAST & 16)
    // RN(a b + -0) = RN(a b) for every input (fused or not, the same flush), and the opaque addend keeps the driver
    // from fusing the product into a later addition (the role of the fence)
    float r = mad(a, b, asfloat(0x80000000u ^ in_zb()));
#else
    float r = FN(a * b);
#endif
#if RR_IN_EXACT
    if ((asuint(r) & 0x7fffffffu) == 0u) r = in_round((double)a * (double)b);   // exact product
#elif RR_IN_LITE
#else
    // a zero result with nonzero operands (both then finite: inf or NaN operands give inf / NaN results)
    in_flag |= select(and(IN_ZT(r), and(a != 0.0f, b != 0.0f)), 1u, 0u);
#endif
    return r;
}
float FM(float a, float b, float c)
{
    float r = FN(mad(a, b, c));
#if RR_IN_FAST & 8
    return r;
#endif
#if RR_IN_EXACT
    if ((asuint(r) & 0x7fffffffu) == 0u && isfinite(a) && isfinite(b) && isfinite(c)) {
        const double p = (double)a * (double)b, cc = (double)c;
        const double s = p + cc;
        const double bv = s - p;
        const double e = (p - (s - bv)) + (cc - bv);
        const double as = s < 0.0 ? -s : s;
        const double bd = in_boundary();
        uint lo, hi;
        asuint(s, lo, hi);
        const bool up = as > bd || (as == bd && (s < 0.0 ? e <= 0.0 : e >= 0.0));
        r = s == 0.0 ? r : asfloat((hi & 0x80000000u) | (up ? 0x00800000u : 0u));
        if (as >= (double)IN_MN) r = in_round(s);
    }
#else
    // a zero result is exact only when a b = -c exactly (or a b = 0 = c); otherwise the replay decides
    // (a zero result implies finite operands: an inf or NaN operand gives an inf or NaN result)
#if RR_IN_FMBR
    [branch] if (IN_ZT(r)) {
        const float q = FN(a * b);
        const float qe = FN(mad(a, b, -q));
        const bool exactzero = select(or(a == 0.0f, b == 0.0f), c == 0.0f,
                                      and(and(q != 0.0f, q == -c), qe == 0.0f));
        in_flag |= select(exactzero, 0u, 1u);
    }
#else
    const float q = FN(a * b);
    const float qe = FN(mad(a, b, -q));
    const bool exactzero = select(or(a == 0.0f, b == 0.0f), c == 0.0f,
                                  and(and(q != 0.0f, q == -c), qe == 0.0f));
    in_flag |= select(and(IN_ZT(r), !exactzero), 1u, 0u);
#endif
#endif
    return r;
}
float D(float a, float b) { return in_round((double)a / (double)b); }
// 1 / x for 2^-64 <= x <= 2^64: hardware reciprocal (faithful there) corrected once with its exact fma remainder
float in_rcp_mid(float ax)
{
    float r = FN(rcp(ax));
    const float e = FN(mad(-ax, r, 1.0f));                              // exact remainder 1 - x r
    // x times half the gap above / below r, by exponent arithmetic
    const uint rb = asuint(r);
    const int er = (int)((rb >> 23) & 0xffu) - 127;
    const float hup = FN(ldexp(ax, er - 24));
    const float hdn = FN(ldexp(ax, (rb & 0x7fffffu) == 0u ? er - 25 : er - 24));
    const bool up = and(e > 0.0f, FN(e - hup) > 0.0f), dn = and(e < 0.0f, FN(FN(-e) - hdn) > 0.0f);
    return asfloat(select(up, rb + 1u, select(dn, rb - 1u, rb)));
}
float RCP(float x)
{
#if RR_IN_FAST & 2
    return FN(in_ftz(rcp(in_ftz(x))));   // erratum 51-58 (opt-in): the hardware reciprocal (1 ulp), FTZ kept
#endif
    const uint u = asuint(x);
    const float ax = abs(x);
    if (isnan(x)) return x;
    if (ax == 0.0f) return asfloat((u & 0x80000000u) | 0x7f800000u);
    if (ax > asfloat(0x7e800000u)) return asfloat(u & 0x80000000u);      // |x| > 2^126 (and inf): below mn -> 0
    float r;
    if (ax > asfloat(0x5f800000u)) r = FN(in_rcp_mid(FN(ax * asfloat(0x1f800000u))) * asfloat(0x1f800000u));  // 2^64
    else if (ax < asfloat(0x1f800000u)) r = FN(in_rcp_mid(FN(ax * asfloat(0x5f800000u))) * asfloat(0x5f800000u));
    else r = in_rcp_mid(ax);
    return asfloat(asuint(r) | (u & 0x80000000u));
}
// code / d correctly rounded for small integer codes (d = 1023: unorm10, d = 127: snorm8), rd = RN(1/d)
float in_udiv(uint code, float d, float rd)
{
    const float cf = (float)code;
    float v = FN(cf * rd);                                               // faithful
    const float e = FN(mad(-v, d, cf));                                 // exact remainder c - d v
    const float up = asfloat(asuint(v) + 1u), dn = asfloat(asuint(v) - 1u);
    const bool gu = and(e > 0.0f, FN(e - FN(d * FN(FN(up - v) * 0.5f))) > 0.0f);
    const bool gd = and(e < 0.0f, FN(FN(-e) - FN(d * FN(FN(v - dn) * 0.5f))) > 0.0f);
    return select(gu, up, select(gd, dn, v));
}
float in_un10(uint code) { return in_udiv(code & 0x3ffu, 1023.0f, asfloat(0x3a802008u)); }
float in_sn8(uint b)
{
    const int k = ((int)(b << 24)) >> 24;
    if (k == -128) return -1.0f;
    const float v = in_udiv((uint)abs(k), 127.0f, asfloat(0x3c010204u));
    return k < 0 ? -v : v;
}
uint in_zl() { return wg_grid_ >> 31; }      // an opaque 0: keeps loops rolled

// sqrt of a positive normal x: s = hardware sqrt (faithful), r = x - s^2 exact (fma remainder of a faithful root),
// the rounding decided exactly in units of h^2 (h = half an ulp of s): sqrt(x) > s + h  <=>  r > 2 s h + h^2.
float in_sqrt_pos(float x)
{
    float s = FN(sqrt(x));
    const float r = FN(mad(-s, s, x));
    const uint sb = asuint(s);
    const int es = (int)((sb >> 23) & 0xffu) - 127;
    const int Si = (int)((sb & 0x7fffffu) | 0x800000u);
    const int R = (int)FN(ldexp(r, 48 - 2 * es));                       // r / h^2, an integer (|R| < 2^28)
    const bool gu = R - 4 * Si > 1;                                      // above the upper midpoint
    const bool gd = select((sb & 0x7fffffu) != 0u, R < 1 - 4 * Si, 4 * R < 1 - 8 * Si);
    return FN(asfloat(select(gu, sb + 1u, select(gd, sb - 1u, sb))));
}
float in_sqrt(float x)
{
#if RR_IN_FAST & 4
    return FN(sqrt(x));
#endif
    // below 2^-60: scale by 2^64 (exact) and the root back by 2^-32; non-positive, inf and NaN: the hardware root
    const bool small = x < asfloat(0x21800000u);
    const float xs = FN(x * select(small, asfloat(0x5f800000u), 1.0f));
    const float rt = FN(in_sqrt_pos(select(and(x > 0.0f, isfinite(x)), xs, 1.0f)) * select(small, asfloat(0x2f800000u), 1.0f));
    return select(and(x > 0.0f, isfinite(x)), rt, FN(sqrt(x)));
}
float in_sqrt_dbl(float x)
{
    float s = FN(sqrt(x));
    if (!(x > 0.0f) || !isfinite(x)) return s;
    const double xd = (double)x;
    [unroll] for (uint it = 0; it < 2u; ++it) {
        const float up = asfloat(asuint(s) + 1u), dn = asfloat(asuint(s) - 1u);
        const double mu = ((double)s + (double)up) * 0.5, md = ((double)s + (double)dn) * 0.5;
        s = xd >= mu * mu ? up : (xd < md * md ? dn : s);
    }
    return FN(s);
}

double in_d(uint lo, uint hi) { return asdouble(lo, hi); }

// Ziv check: true when the binary64 value r lies within 2^-49 |r| of a midpoint next to its f32 rounding f.
bool in_near_mid(double r, float f)
{
    const double fd = (double)f;
    const bool above = r > fd;
    const float nb = asfloat(asuint(f) + ((above == (f >= 0.0f)) ? 1u : 0xffffffffu));
    const double mid = (fd + (double)nb) * 0.5;
    const double d = r > mid ? r - mid : mid - r;
    const double ar = r < 0.0 ? -r : r;
    return d <= ar * in_d(0x00000000u, 0x3ce00000u);       // 2^-49
}

float log2_dbl(float x)
{
    const uint u = asuint(x);
    if (isnan(x) || ((u >> 31) != 0u && (u & 0x7fffffffu) != 0u)) return asfloat(0x7fc00000u);
    if ((u & 0x7fffffffu) == 0u) return asfloat(0xff800000u);
    if (u == 0x7f800000u) return x;
    int e = (int)((u >> 23) & 0xffu) - 127;
    const uint mb = u & 0x7fffffu;
    if (mb == 0u) return (float)e;
    double m = (double)asfloat(mb | 0x3f800000u);
    if (mb > 0x3504f3u) { m = m * 0.5; e += 1; }
    const double z = (m - 1.0) / (m + 1.0);
    const double z2 = z * z;
    double p = in_d(0x590b2164u, 0x3fa642c8u);                 // 1/23
    p = p * z2 + in_d(0x18618618u, 0x3fa86186u);               // 1/21
    p = p * z2 + in_d(0xbca1af28u, 0x3faaf286u);               // 1/19
    p = p * z2 + in_d(0x1e1e1e1eu, 0x3fae1e1eu);               // 1/17
    p = p * z2 + in_d(0x11111111u, 0x3fb11111u);               // 1/15
    p = p * z2 + in_d(0x13b13b14u, 0x3fb3b13bu);               // 1/13
    p = p * z2 + in_d(0x745d1746u, 0x3fb745d1u);               // 1/11
    p = p * z2 + in_d(0x1c71c71cu, 0x3fbc71c7u);               // 1/9
    p = p * z2 + in_d(0x92492492u, 0x3fc24924u);               // 1/7
    p = p * z2 + in_d(0x9999999au, 0x3fc99999u);               // 1/5
    p = p * z2 + in_d(0x55555555u, 0x3fd55555u);               // 1/3
    p = p * z2 + 1.0;
    const double lnm = (z + z) * p;
    const double l2 = lnm * in_d(0x652b82feu, 0x3ff71547u) + lnm * in_d(0xffda0d24u, 0x3c7777d0u);   // / ln 2
    const double r = (double)e + l2;
    const float f = (float)r;
    if (in_near_mid(r, f)) in_ziv += 1u;
    return FN(f);
}

float exp2_dbl(float t)
{
    if (isnan(t)) return t;
    if (t >= 128.0f) return asfloat(0x7f800000u);
    if (t < -126.0f) return 0.0f;
    float n = floor(t);
    double fd = (double)t - (double)n;                          // exact in binary64 (not in f32 for t < 0)
    if (fd > 0.5) { fd = fd - 1.0; n = n + 1.0f; }
    if (fd == 0.0) return in_round(asdouble(0u, (uint)((int)n + 1023) << 20));
    const double y = fd * in_d(0xfefa39efu, 0x3fe62e42u) + fd * in_d(0x3b39803fu, 0x3c7abc9eu);   // f ln 2
    double q = in_d(0x13a86d09u, 0x3de61246u);                  // 1/13!
    q = q * y + in_d(0xeff8d898u, 0x3e21eed8u);
    q = q * y + in_d(0x67f544e4u, 0x3e5ae645u);
    q = q * y + in_d(0xb7789f5cu, 0x3e927e4fu);
    q = q * y + in_d(0xa556c734u, 0x3ec71de3u);
    q = q * y + in_d(0x1a01a01au, 0x3efa01a0u);
    q = q * y + in_d(0x1a01a01au, 0x3f2a01a0u);
    q = q * y + in_d(0x16c16c17u, 0x3f56c16cu);
    q = q * y + in_d(0x11111111u, 0x3f811111u);
    q = q * y + in_d(0x55555555u, 0x3fa55555u);
    q = q * y + in_d(0x55555555u, 0x3fc55555u);
    q = q * y + 0.5;
    q = q * y + 1.0;
    q = q * y + 1.0;
    const int ni = (int)n;
    // 2^n in two exact factors (n can be 128 with f < 0 or -126)
    const int n1 = ni / 2, n2 = ni - n1;
    const double r = (q * asdouble(0u, (uint)(n1 + 1023) << 20)) * asdouble(0u, (uint)(n2 + 1023) << 20);
    const float o = in_round(r);
    if (in_near_mid(r, o)) in_ziv += 1u;
    return o;
}


// ---- fast paths ------------------------------------------------------------------------------------------------------
// Float-float log2 / exp2 (gpu/ffmath.py is the NumPy model of exactly this sequence, with the error measurements):
// the pair (H, L) has a relative error below 2^-45.2 (log2) / 2^-45.6 (exp2) in the measured sets and below EPS =
// 2^-42 by construction (term bounds in ffmath.py); H = RN(H + L) is the correctly rounded result unless H + L lies
// within EPS |H| of a rounding midpoint, in which case the binary64 path recomputes it (counted in in_slow).
// Tables (host-built, gpu/ffmath.py): log2 [256] (c, Th, Tl, 0), exp2 [65] (Eh, El).
static uint in_slow = 0u;
static uint in_plog = 0u, in_pexp = 0u;
#ifndef RR_IN_LDST
#define RR_IN_LDST 0         // 1: the two tables are staged in LDS per workgroup (in_stage_tables), else read from memory
#endif
#if RR_IN_LDST
groupshared float4 gs_logt[256];
groupshared float2 gs_expt[65];
float4 in_logt(uint i) { return gs_logt[i]; }
float2 in_expt(uint j) { return gs_expt[j]; }
void in_stage_tables()
{
    for (uint i = wg_lane_id(); i < 256u; i += WG_BLOCK) gs_logt[i] = asfloat(wg_ld128(in_plog, i * 16u));
    for (uint j = wg_lane_id(); j < 65u; j += WG_BLOCK) gs_expt[j] = asfloat(wg_ld64(in_pexp, j * 8u));
    GroupMemoryBarrierWithGroupSync();
}
#else
float4 in_logt(uint i) { return asfloat(wg_ld128(in_plog, i * 16u)); }
float2 in_expt(uint j) { return asfloat(wg_ld64(in_pexp, j * 8u)); }
void in_stage_tables() {}
#endif

float PR(float a, float b) { return a * b; }
float PRF(float a, float b) { return FN(a * b); }   // product that feeds another product
float PE(float a, float b, float p) { return mad(a, b, -p); }      // exact error of p = RN(a b)
void tsum(float a, float b, out float s, out float e)
{
    s = a + b;
    const float bb = s - a;
    e = (a - (s - bb)) + (b - bb);
}
bool ff_near_mid(float H, float L)
{
    const float nb = asfloat(asuint(H) + (((L >= 0.0f) == (H >= 0.0f)) ? 1u : 0xffffffffu));
    const float halfgap = FN(abs((nb - H)) * 0.5f);
    return abs((halfgap - abs(L))) <= FN(abs(H) * asfloat(0x2a800000u));   // 2^-42
}

float log2_ff(uint u, out bool slow)
{
    const int e0 = (int)((u >> 23) & 0xffu) - 127;
    const uint mb = u & 0x7fffffu;
    const uint hi = mb > 0x3504f3u ? 1u : 0u;
    const float m = asfloat(mb | (hi != 0u ? 0x3f000000u : 0x3f800000u));
    const float4 tb = in_logt(hi * 128u + (mb >> 16));
    const float r = (mad(m, tb.x, -1.0f));                                  // exact
    const float K1h = asfloat(0x3fb8aa3bu), K1l = asfloat(0x32a57060u);
    const float K2h = asfloat(0xbf38aa3bu), K2l = asfloat(0xb2257060u);
    const float K3h = asfloat(0x3ef6384fu), K3l = asfloat(0xb1717f01u);
    const float Ah = PR(r, K1h);
    const float Al = FN(PE(r, K1h, Ah) + PR(r, K1l));
    const float R2h = PRF(r, r);
    const float R2l = PE(r, r, R2h);
    const float Bh = PR(R2h, K2h);
    const float Bl = FN(PE(R2h, K2h, Bh) + FN(PR(R2l, K2h) + PR(R2h, K2l)));
    const float R3h = PRF(R2h, r);
    const float R3l = FN(PE(R2h, r, R3h) + PR(R2l, r));
    const float Ch = PR(R3h, K3h);
    const float Cl = FN(PE(R3h, K3h, Ch) + FN(PR(R3l, K3h) + PR(R3h, K3l)));
    float Pp = FN(mad(asfloat(0x3e530bb1u), r, asfloat(0xbe76384fu)));
    Pp = FN(mad(Pp, r, asfloat(0x3e93bb63u)));
    Pp = FN(mad(Pp, r, asfloat(0xbeb8aa3bu)));
    const float Dv = PR(PRF(R3h, r), Pp);
    float s1, s2, s3, s4, e1, e2, e3, e4;
    tsum((float)(e0 + (int)hi), tb.y, s1, e1);
    tsum(s1, Ah, s2, e2);
    tsum(s2, Bh, s3, e3);
    tsum(s3, Ch, s4, e4);
    float lo = FN((e1 + e2) + (e3 + e4));
    lo = FN(lo + FN((tb.z + Al) + ((Bl + Cl) + Dv)));
    const float H = (s4 + lo);
    const float L = (lo - (H - s4));
    slow = ff_near_mid(H, L);
    return H;
}

float PEF(float a, float b, float p) { return FN(mad(a, b, -p)); }
void tsumF(float a, float b, out float s, out float e)
{
    s = FN(a + b);
    const float bb = FN(s - a);
    e = FN(FN(a - FN(s - bb)) + FN(b - bb));
}
// exp2: every operation fenced (without the fences the driver's regrouping changed results: --xcheck)
// t in [-126, 128), not an integer
float exp2_ff(float t, out bool slow)
{
    float n = floor(t);
    float fh, fl;
    tsumF(t, -n, fh, fl);
    if (fh > 0.5f) { fh = FN(fh - 1.0f); n = FN(n + 1.0f); }
    const float jf = round(FN(fh * 64.0f));
    const float g0 = FN(fh - FN(jf * 0.015625f));                               // exact
    float gh, gl;
    tsumF(g0, fl, gh, gl);
    const float M1h = asfloat(0x3f317218u), M1l = asfloat(0xb102e308u);
    const float M2h = asfloat(0x3e75fdf0u), M2l = asfloat(0xaf7a74e3u);
    const float G1h = PRF(gh, M1h);
    const float G1l = FN(PEF(gh, M1h, G1h) + FN(PRF(gh, M1l) + PRF(gl, M1h)));
    const float R2h = PRF(gh, gh);
    const float R2l = FN(PEF(gh, gh, R2h) + PRF(2.0f, PRF(gh, gl)));
    const float G2h = PRF(R2h, M2h);
    const float G2l = FN(PEF(R2h, M2h, G2h) + FN(PRF(R2l, M2h) + PRF(R2h, M2l)));
    const float Q = FN(mad(FN(mad(asfloat(0x3aaec3ffu), gh, asfloat(0x3c1d955bu))), gh, asfloat(0x3d635847u)));
    const float G3 = PRF(PRF(R2h, gh), Q);
    float s1, s2, e1, e2;
    tsumF(1.0f, G1h, s1, e1);
    tsumF(s1, G2h, s2, e2);
    const float lo = FN(FN(e1 + e2) + FN(FN(G1l + G2l) + G3));
    const float Ph = FN(s2 + lo);
    const float Pl = FN(lo - FN(Ph - s2));
    const float2 et = in_expt((uint)((int)jf + 32));
    const float Rh = PRF(et.x, Ph);
    const float Rl = FN(PEF(et.x, Ph, Rh) + FN(PRF(et.x, Pl) + PRF(et.y, Ph)));
    const float H = FN(Rh + Rl);
    const float L = FN(Rl - FN(H - Rh));
    slow = ff_near_mid(H, L);
    const int ni = (int)n;
    const int n1 = ni / 2, n2 = ni - n1;
    return FN(FN(H * asfloat((uint)(n1 + 127) << 23)) * asfloat((uint)(n2 + 127) << 23));
}

float log2_cr(float x)
{
#if RR_IN_FAST & 1
    return FN(log2(x));      // timing experiment only: hardware log2 (not correctly rounded)
#endif
    const uint u = asuint(x);
    if (isnan(x) || ((u >> 31) != 0u && (u & 0x7fffffffu) != 0u)) return asfloat(0x7fc00000u);
    if ((u & 0x7fffffffu) == 0u) return asfloat(0xff800000u);
    if (u == 0x7f800000u) return x;
    if ((u & 0x7fffffu) == 0u) return (float)((int)((u >> 23) & 0xffu) - 127);
    bool slow;
    const float h = log2_ff(u, slow);
#if RR_IN_EXACT
    if (!slow) return h;
    in_slow += 1u;
    return log2_dbl(x);
#else
    in_flag |= slow ? 1u : 0u;
    return h;
#endif
}
float exp2_cr(float t)
{
#if RR_IN_FAST & 1
    return FN(in_ftz(exp2(t)));
#endif
    if (isnan(t)) return t;
    if (t >= 128.0f) return asfloat(0x7f800000u);
    if (t < -126.0f) return 0.0f;
    if (t == floor(t)) return asfloat((uint)((int)t + 127) << 23);
    bool slow;
    const float h = exp2_ff(t, slow);
#if RR_IN_EXACT
    if (!slow) return h;
    in_slow += 1u;
    return exp2_dbl(t);
#else
    in_flag |= slow ? 1u : 0u;
    return h;
#endif
}

// f16 helpers: wg_f2h = round to nearest even (v_cvt_f16_f32), f16tof32 exact
float in_r16(float x) { return f16tof32(wg_f2h(x)); }
