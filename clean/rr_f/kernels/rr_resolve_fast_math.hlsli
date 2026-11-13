// SPDX-License-Identifier: MIT
// Fast exact helpers of the fast page-54b resolve (rr_resolve_fast.hlsl). Every function here returns, for every
// f32 input, exactly the bits of its counterpart in rr_resolve_math.hlsli (the verified kernel's helpers), or sets a
// flag that sends the caller to that counterpart. Proof: exhaustive GPU comparison over all 2^32 inputs
// (gpu/resolve_fast.py exhaustive). Plain f32 operations follow the build's arithmetic mode (clean/dx12/README.md).
#pragma once
#include "rr_rf_tables.hlsli"

// The including kernel provides RF_TGET(j): the uint2 (hi, lo bits) of T[j] = 2^(j/1024) (0 <= j < 1024), usually
// from a groupshared copy of the arena table (rr_rf_init_table).

// ---- correctly rounded exp2 (= EXP2_CR of rr_resolve_math.hlsli, mode 0) -----------------------------------------
// x = n + j/1024 + r (|r| <= 2^-11, exact), 2^x = 2^n T[j] (1 + s + s^2/2 + s^3/6 + s^4/24), s = r ln 2 carried as
// sh + sl. T[j] (1 + ...) is assembled in [1/2, 2) with an error far below 2^-43 (sum of the f32 rounding errors of the
// small terms; the tail beyond s^4 is below 2^-58); f = RN(H + R) with its exact rounding error z. When |z| is closer
// than RF_EPS to half an ulp of f (a quarter-ulp step below a power of two) the decision is left to the caller ("hard",
// about one input in 2^20); otherwise f is the correctly rounded value. The exponent is added to the bits of f (f in
// [1/2, 2), result exponent >= -126; f = 1, n = 128 gives exactly +inf, as the correct rounding does).
#define RF_EPS asfloat(0x2a000000u)          // 2^-43

float rf_exp2_core(float x, out bool hard)  // x already flushed (FTZ)
{
    float k = round(x * 1024.0f);
    float r = mad(k, -0.0009765625f, x);     // exact
    int ki = (int)k;
    uint j = (uint)ki & 1023u;
    int n = ki >> 10;
    uint2 T = RF_TGET(j);
    float Th = asfloat(T.x), Tl = asfloat(T.y);
    float sh = r * RF_L1;
    float se = mad(r, RF_L1, -sh);
    float sl = mad(r, RF_L2, se);
    float p = mad(sh, RF_C4, RF_C3);
    p = mad(sh, p, 0.5f);
    float S2 = fence(sh * sh);
    float q = S2 * p;
    q = mad(sh, sl, q);
    float A1 = Th * sh;
    float a1 = mad(Th, sh, -A1);
    float H = Th + A1;
    float h = (Th - H) + A1;
    float R = mad(Th, sl + q, Tl);
    R = mad(Tl, sh, R);
    R = R + a1;
    R = R + h;
    float f = H + R;
    float z = (H - f) + R;
    uint fb = asuint(f);
    float thr = asfloat((fb & 0x7f800000u) - 0x0c000000u);
    thr = select(and((fb & 0x007fffffu) == 0u, z < 0.0f), thr * 0.5f, thr);
    hard = abs(z) >= thr - RF_EPS;
    return asfloat(fb + ((uint)n << 23));
}

// Second tier (rf_exp2_core) for an input in [-126, 128): slow |= undecided (1314 f32 inputs in all; the caller
// then takes the verified code, whose exp2 is EXP2_CR).
float rf_exp2_precise(float x, inout bool slow)
{
    bool hd;
    float v = rf_exp2_core(x, hd);
    slow = or(slow, hd);
    return v;
}

// First tier: 2^x = 2^n T[j] (1 + p), p = r (P1 + r (P2 + r P3)) in one f32 word, S = Th p + Tl (Th + S within
// 2^-33.2 of 2^x / 2^n). When Th + S - RF_EPS1 and Th + S + RF_EPS1 round to the same f32, so does 2^x / 2^n;
// otherwise (about one input in 2^9) the second tier decides. The input is clamped to [-127, 128] first: 128 gives
// exactly +inf (f = 1, n = 128), x < -126 is set to +0, NaN to the quieted NaN afterwards (EXP2_CR's specials).
#define RF_EPS1 asfloat(0x2f000000u)         // 2^-33 (the tier's own bound is 2^-33.2; checked exhaustively)
#ifndef RF_HW
#define RF_HW 0      // 1: the opt-in non-exact variant: exp2 = the hardware's (X of rr_resolve_math.hlsli, RR_ELEM_EXACT=0)
#endif
#ifndef RF_MUT
#define RF_MUT 0     // negative controls (gpu/resolve_fast.py diff --mutation): each must make the comparison fail
#endif
float rf_exp2_t1f(float x, out bool und)     // x flushed and not NaN; und: rf_exp2_core(x) decides (x in (-126, 128))
{
#if RF_HW
    und = false;
    return fence(ftz(exp2(ftz(x))));
#endif
    float xc = min(max(x, -126.0f), 128.0f);  // -126 and 128 are decided (exact powers of two; 2^128 gives +inf)
    float k = round(xc * 1024.0f);
    float r = mad(k, -0.0009765625f, xc);    // exact
    int ki = (int)k;
    uint2 T = RF_TGET((uint)ki & 1023u);
    float Th = asfloat(T.x), Tl = asfloat(T.y);
    float p = r * mad(r, mad(r, RF_P3, RF_P2), RF_P1);
    float S = mad(Th, p, Tl);
    float fa = Th + (S - RF_EPS1), fb = Th + (S + RF_EPS1);  // S +- 2^-33 exact (|S| < 2^-10: ulp(S) <= 2^-34)
    float v = asfloat(asuint(fa) + ((uint)(ki >> 10) << 23));
    und = fa != fb;                                       // the interval rounds two ways
#if RF_MUT == 1
    und = false;                                          // mutation: the first tier decides alone
#endif
    return select(x < -126.0f, 0.0f, v);
}
float rf_exp2_t1(float x, out bool und)      // x already flushed (an arithmetic result)
{
#if RF_HW
    und = false;
    return fence(ftz(exp2(ftz(x))));
#endif
    float v = rf_exp2_t1f(x, und);
    return select(nanf(x), asfloat(asuint(x) | 0x00400000u), v);   // a NaN: und is false (xc = -126)
}
float rf_exp2_f(float x, inout bool slow)
{
    bool und;
    float v = rf_exp2_t1(x, und);
#if (RF_PROF & 64) == 0
    if (und) v = rf_exp2_precise(x, slow);               // divergent: about one wave-call in 8
#endif
    return v;
}

// Third tier: EXP2_CR (rr_resolve_math.hlsli) for the inputs both fast tiers leave undecided (x in [-126, 128)), at
// one call site per group: a loop whose trip count the compiler cannot see, so it is not unrolled into copies.
void rf_exp2_fix3(float3 x, inout float3 v, bool3 hd)
{
    uint n = 3u + (wg_grid_ >> 31);
    [loop] for (uint k = 0; k < n; ++k) {
        float xk = select(k == 0u, x.x, select(k == 1u, x.y, x.z));
        bool hk = select(k == 0u, hd.x, select(k == 1u, hd.y, hd.z));
        if (hk) {
            bool h;
            float r = EXP2_CR(xk, 0u, h);
            v.x = select(k == 0u, r, v.x); v.y = select(k == 1u, r, v.y); v.z = select(k == 2u, r, v.z);
        }
    }
}
void rf_exp2_fix8(float x[8], inout float v[8], bool hd[8])
{
    uint n = 8u + (wg_grid_ >> 31);
    [loop] for (uint k = 0; k < n; ++k) {
        float xk = x[0];
        bool hk = hd[0];
        [unroll] for (uint i = 1; i < 8u; ++i) { xk = select(k == i, x[i], xk); hk = select(k == i, hd[i], hk); }
        if (hk) {
            bool h;
            float r = EXP2_CR(xk, 0u, h);
            [unroll] for (uint i2 = 0; i2 < 8u; ++i2) v[i2] = select(k == i2, r, v[i2]);
        }
    }
}

// Full function (any f32 input): EXP2_CR exactly, or slow |= an undecided input (exhaustive check).
float rf_exp2(float xin, inout bool slow) { return rf_exp2_f(ftz(xin), slow); }

// ---- correctly rounded reciprocal (= RCP of rr_resolve_math.hlsli) ------------------------------------------------
// One Newton step with fused multiply-adds from the hardware reciprocal; |x| >= 2^126 is decided directly (the result
// is below the normal range: FMINN for |x| = 2^126, 0 beyond, with the sign of x).
#ifndef RF_HWRCP
#define RF_HWRCP 0   // 1 (opt-in, erratum 51-58 / 54b "within 1 ulp"): RCP = the hardware reciprocal, FTZ kept
#endif
float rf_rcp(float xin)
{
#if RF_HWRCP
    return ftz(rcp(ftz(xin)));
#endif
    float x = ftz(xin);
    float y = rcp(x);
    float e = mad(-x, y, 1.0f);
    float r = mad(e, y, y);
    uint ax = asuint(x) & 0x7fffffffu, sg = asuint(x) & 0x80000000u;
    r = select(ax >= 0x7e800000u, asfloat(sg | select(ax == 0x7e800000u, 0x00800000u, 0u)), r);
    r = select(ax == 0u, asfloat(sg | 0x7f800000u), r);
    r = select(ax == 0x7f800000u, asfloat(sg), r);
    r = select(ax > 0x7f800000u, x, r);
    return r;
}

// ---- tap accumulation with tiny weights (= F of rr_resolve_math.hlsli) -------------------------------------------
// F(w, v, acc) for w, v, acc each +0 or a positive normal f32 with v = 0 or v >= 2^-24 (alpha and gate values of
// 54b.4). Every term is non-negative, so the exact result is below FMINN only when acc = 0 and w v < FMINN; there the
// hardware result is 0 (the flush before rounding) where F gives FMINN from FMINN - 2^-150 on. That one case is
// decided exactly from ws v = p + er (ws = w 2^100: w v < FMINN with v >= 2^-24 means w < 2^-102, so ws < 2^-2 and
// ws >= 2^-26; p >= 2^-50 and er a multiple of 2^-96: both normal, the fma residual exact).
#define RF_BS asfloat(0x327fffffu)          // (FMINN - 2^-150) 2^100 = 2^-26 - 2^-50
float rf_accx(float w, float v, float acc)
{
    float r = mad(w, v, acc);
#if RF_MUT == 5
    return acc;                             // mutation: the tiny-weight batches drop their products
#endif
    float ws = w * asfloat(0x71800000u);    // w 2^100 (exact; only used when w < 2^-102)
    float p = ws * v, er = mad(ws, v, -p);
    bool up = or(p > RF_BS, and(p == RF_BS, er >= 0.0f));
    return select(and(r == 0.0f, and(w != 0.0f, v != 0.0f)), select(up, asfloat(0x00800000u), 0.0f), r);
}

// ---- conversions ---------------------------------------------------------------------------------------------------
// f32 -> binary16 bits, round to nearest even (= wg_f2h for every f32): the driver channel's hardware conversion
// (op 0x1C, immediate 1: v_cvt_f16_f32 in the default rounding mode; any control flow).
uint rf_f2h(float x) { return wg_ags_call(0x1Cu, 0u, 1u, asuint(x), 0u) & 0xffffu; }
float rf_h(float x) { return f16tof32(rf_f2h(x)); }          // = H(x)

// Unorm read code / maxc rounded once to f32 (= cv / wg_div for every code 0 <= code <= maxc of the formats):
// the product with RN(1/maxc) corrected by one fused residual step.
float rf_cv(uint code, float m, float rinv)
{
    float c = (float)code;
    float q = c * rinv;
    float e = mad(-q, m, c);
    return mad(e, rinv, q);
}
#define RF_R255 asfloat(0x3b808081u)    // RN(1/255)
#define RF_R1023 asfloat(0x3a802008u)   // RN(1/1023)
#define RF_R3 asfloat(0x3eaaaaabu)      // RN(1/3)

// Unorm store code (= unorm_code of rr_resolve.hlsl for every f32 v): round half to even of the exact product
// clamp(v) * maxc, from its rounded product p, the exact residual err (c maxc = p + err) and n = rint(p).
uint rf_unorm_code(float v, uint maxc)
{
    float c = min(max(v, 0.0f), 1.0f);
    float m = (float)maxc;
    float p = c * m;
    float err = mad(c, m, -p);
    float n = round(p);
    float d = p - n;
    n = n + select(and(d == 0.5f, err > 0.0f), 1.0f, select(and(d == -0.5f, err < 0.0f), -1.0f, 0.0f));
    return select(nanf(v), 0u, min((uint)n, maxc));
}
