// SPDX-License-Identifier: MIT
// Model K output stage helpers (ko_*) for the D3D12 ports of k_out_color / k_out_pixel(_c) and their direct forms
// (clean/k/dx12/kernels/k_out.hlsl). WrongGPU's own code: HLSL twins of our HIP output stage, clean/k/kernels/k_io.hip
// (spec 43 sections 3 - 7 as that file implements them) and of the exact helpers it uses from clean/k/kernels/k_fp.h.
// Include after wg_platform.hlsli (and native_direct.hlsli for WG_DIRECT_TEXTURES).
//
// NOTE (group G2 io-out): this header holds its own copies of the few k_fp.h / k_io.hip pieces the output stage needs
// (the compact KOutParams argument block, kio_* fetches, kio_encode / kio_decode, the binary64 log / exp, kh_* FP16
// steps). Group G1 owns the shared IO headers clean/k/dx12/hlsl/k_fp.hlsli / k_frame.hlsli, which did not exist when
// this port was written; G2 does not edit them. The coordinator may move these pieces there when merging.
//
// ARGUMENTS: one KOutParams by value, manifest token s312:p4.b12.x24.b52.x12.b24.x4.p4.x8.p4.b8.p2.b56 (52 DWORDs; the
// full struct would be 63 > 57). Skipped fields and why they are dead on the runtime path (clean/k/runtime/runtime_k.cpp
// KRuntime::frame):
//   cbx, cby, mbx, mby, zbx, zby  (x24)  the subrect bases: the runtime zero-initialises KFrameIn (fin = {}) and never
//                                       sets them (its inputs are dense render-size images); the port uses 0.
//   kappa, rho, lam               (x12)  the runtime sets them to 1 and always passes KFrameIn.xs (k_expo_scalars' GPU
//                                       result), and ko_params / kio_frame replace kappa, rho, lam and KOutUni.kinv by
//                                       xs[0..3] whenever xs is set; the port reads xs always (lam is unused here).
//   pad_                          (x4)   padding.
//   exp2tab                       (x8)   h(2^E) table read only by the host emulator's ko_exp2h; the GPU kernel uses
//                                       v_exp_f16 (as this port: exp2 on half, checked with RGA, see k_out.hlsl).
#pragma once

WG_ARGS_BEGIN
    uint f_color, f_motion, f_depth, f_xs;            // KFrameIn pointers (p4)
    int f_cpitch, f_mpitch, f_zpitch;                 // (b12); cbx .. zby skipped (x24)
    int f_W, f_H, f_rw, f_rh, f_Bw, f_Bh, f_flags;    // (b52)
    float f_jx, f_jy, f_sx, f_sy, f_mox, f_moy;
    float f_s; int f_mch; float f_rmsx, f_rmsy, f_rmox, f_rmoy;   // kappa / rho / lam skipped (x12); (b24); pad_ (x4)
    uint p_N, p_L, p_Hc, p_table;                     // (p4); exp2tab skipped (x8)
    uint p_out, p_Lout, p_Hcout, p_Hhout;             // (p4)
    int p_opitch, p_omode;                            // (b8)
    uint p_ctap, p_ctapT;                             // (p2)
    float u_invW, u_invH, u_rhox, u_rhoy, u_rho2x, u_rho2y, u_jmx, u_jmy, u_spx, u_spy, u_invW16, u_invH16, u_kinv;
    uint u_kuxkvy;                                    // KOutUni (b56): kux (low 16 bits), kvy (high)
WG_ARGS_END

#define KIO_HDR 1
#define KIO_MVLO 2
#define KIO_INV 4
#define KIO_RESET 8
#define KIO_C16 16
#define KIO_M16 32
#define KIO_OUT_RGBA32F 0
#define KIO_OUT_RGBA16F 1
#define KIO_OUT_RGB32F 2

// opaque zero (the launch's workgroup count is below 2^31): x ^ ko_z is x, but the D3D12 shader compiler cannot see
// through it. Used where the driver was seen to regroup chains of products by a shared / uniform factor or to fuse
// FP16 multiply + add (wg_math.hlsli "DRIVER EXCEPTIONS"); HIP keeps every such product as written (contract off).
#ifndef KO_APPROX
#define KO_APPROX 0
#endif
// KO_APPROX 1: CLOSE-APPROXIMATION variant a2 (runtime option k_approx=a2; opt-in, never the default): ko_div is
// v_rcp_f32 times the dividend plus one correction of the quotient from its exact residual (q + r (a - b q), two fma)
// instead of the exact division, and ko_div_axis and ko_rcp_refined drop their Newton step. ko_rcp_output_bounded and
// every opaque barrier (kof, koh, kp_opq, the conversion input) stay: without them, measured, one-FP16-ulp differences
// of mid-range intermediates reach outputs near zero (a few at 4 ulp). Not byte-identical; measured on the port-check
// output family (6 seeds, plain and special-value inputs): every FP16 output within 1 ulp of exact, about 1e-5 of the
// values differ.
#define ko_z (wg_grid_ >> 31)
float kof(float x) { return asfloat(asuint(x) ^ ko_z); }
double kod(double x) { uint lo, hi; asuint(x, lo, hi); return asdouble(lo ^ ko_z, hi); }
half koh(half x) { return asfloat16((uint16_t)(asuint16(x) ^ (uint16_t)ko_z)); }
uint kh2_u(half2 h) { const uint16_t2 v = asuint16(h); return (uint)v.x | ((uint)v.y << 16); }
half2 kh2_h(uint u) { return asfloat16(uint16_t2((uint16_t)(u & 0xffffu), (uint16_t)(u >> 16))); }
// FP16 pairs held as packed 32-bit words (low half = element 0, kp_*): every operation unpacks, operates on the half2
// and repacks. Both halves of a pair then stay in one register (the wave64 kernels get v_*_f16 on its .l / .h halves;
// the driver issues v_pk_*_f16 only in wave32 kernels) and the opaque zero of a product is one 32-bit xor on the
// pair, xor'ed into both halves (ko_z * 0x10001: with a 0 / 1 value in the low half only, the driver proved the high
// half's xor zero, k_common.hlsli); the per-half 16-bit form split every pair into separate registers.
uint kp_splat(half h) { return (uint)asuint16(h) * 0x10001u; }
uint kp_pack(half lo, half hi) { return (uint)asuint16(lo) | ((uint)asuint16(hi) << 16); }
half kp_lo(uint u) { return asfloat16((uint16_t)(u & 0xffffu)); }
half kp_hi(uint u) { return asfloat16((uint16_t)(u >> 16)); }
uint kp_add(uint a, uint b) { return kh2_u(kh2_h(a) + kh2_h(b)); }
uint kp_sub(uint a, uint b) { return kh2_u(kh2_h(a) - kh2_h(b)); }
// product rounded to FP16 as its own instruction, as fma(a, b, -0.0 ^ opaque zero): the addend is -0.0 in both halves at
// run time (the same bits as the multiply: one rounding of the exact product, a zero product keeps its sign, NaN /
// infinities / denormals as the multiply), unknown to the compiler: no fusion into a following add and no regrouping,
// without the xor of the packed word (k_common.hlsli wgk_fm). Never directly into an FP16 -> FP32 conversion (kp_opq).
half2 kp_nz2() { return kh2_h(0x80008000u ^ (ko_z * 0x10001u)); }
uint kp_mul(uint a, uint b) { return kh2_u(mad(kh2_h(a), kh2_h(b), kp_nz2())); }
// product without the opaque, for products that feed only a multiply or a multiplicand (no add can absorb them; each use
// checked in the ISA: the same v_mul_f16 / v_fma_f16 counts, no v_fma_mix*)
uint kp_mul_n(uint a, uint b) { return kh2_u(kh2_h(a) * kh2_h(b)); }
uint kp_fma(uint a, uint b, uint c) { return kh2_u(mad(kh2_h(a), kh2_h(b), kh2_h(c))); }
uint kp_max(uint a, uint b) { return kh2_u(max(kh2_h(a), kh2_h(b))); }
uint kp_sat_nz(uint a) { return kh2_u(min(max(kh2_h(a), (half2)0), (half2)1)); }
uint kp_rni(uint a) { return kh2_u(round(kh2_h(a))); }
uint kp_abs(uint a) { return a & 0x7fff7fffu; }
uint kp_opq(uint a) { return a ^ (ko_z * 0x10001u); }
// the same with one operand a scalar FP16 value used in both halves (no splatted register)
uint kp_rsub_s(half s, uint a) { return kh2_u((half2)s - kh2_h(a)); }
uint kp_add_s(uint a, half s) { return kh2_u(kh2_h(a) + (half2)s); }
uint kp_mul_s(uint a, half s) { return kh2_u(mad(kh2_h(a), (half2)s, kp_nz2())); }   // (fma form: see kp_mul)
uint kp_mul_s_n(uint a, half s) { return kh2_u(kh2_h(a) * (half2)s); }   // (no opaque: see kp_mul_n)
uint kp_fma_s(uint a, half s, uint c) { return kh2_u(mad(kh2_h(a), (half2)s, kh2_h(c))); }
uint kp_fma_ss(uint a, half s, half c) { return kh2_u(mad(kh2_h(a), (half2)s, (half2)c)); }

// ---- FP32 helpers (k_fp.h) ------------------------------------------------------------------------------------------
float kf_bits(uint b) { return asfloat(b); }
bool kf_finite(float x) { return (asuint(x) & 0x7f800000u) != 0x7f800000u; }   // = x - x == 0
int ki_min(int a, int b) { return a < b ? a : b; }
int ki_max(int a, int b) { return a > b ? a : b; }
int ko_clampi(int i, int n) { return ki_max(ki_min(i, n - 1), 0); }
int kio_ceil16(int n) { return (n + 15) / 16 * 16; }
// R(t): round half away from zero
float kf_R(float t)
{
    const float a = abs(t);
    const float fl = floor(a);
    const float r = (a - fl) >= 0.5f ? fl + 1.0f : fl;
    return t < 0.0f ? -r : r;
}
// kf_sat (x != x ? 0 : clamp(x, 0, 1), -0 kept) as HIP ko_sat (x == 0 ? x : med3(x, 0, 1)): equal for every input,
// also with FTZ (a denormal compares equal to 0 and is passed through as kf_sat passes it)
float ko_sat(float x) { return x == 0.0f ? x : min(max(x, 0.0f), 1.0f); }

// ---- FP16 (k_fp.h kh_*; HLSL half arithmetic is v_*_f16 / v_pk_*_f16, round to nearest even, FP16 denormals kept) --
// RN16 of an FP32 value (HIP kh_round / (_Float16) x = v_cvt_f16_f32); the D3D12 casts round toward zero.
// KO_HW16 = 1: the hardware conversion (v_cvt_f16_f32, round to nearest even) through the driver channel's op 0x1C,
// immediate 1 (clean/dx12/hlsl/wg_ags.hlsli; GPU-checked bit-equal to RNE incl. subnormals, inf, NaN, divergent calls;
// never immediates 2 / 3, whose rounding mode is not restored); 0: the software RNE wg_f2h. Same bits either way.
#ifndef KO_HW16
#define KO_HW16 1
#endif
#if KO_HW16
#if WG_WAVE != 32
#include "wg_ags.hlsli"   // (wg_platform.hlsli includes it for wave32 kernels)
#endif
// the input through the opaque zero: the driver folded the FP32 operation feeding the conversion into it (a - b then
// RN16 became one v_fma_mixlo_f16, one rounding where HIP rounds twice)
uint ko_f2h(float x) { return wg_ags_call(0x1cu, 0u, 1u, asuint(x) ^ ko_z, 0u); }   // high half 0
#else
uint ko_f2h(float x) { return wg_f2h(x); }
#endif
half kh_round(float x) { return asfloat16((uint16_t)ko_f2h(x)); }
half kh_bits(uint b) { return asfloat16((uint16_t)b); }
uint kh_u(half h) { return (uint)asuint16(h); }
// exact FP16 -> FP32 (the RTZ conversion of an FP16-representable value is exact too)
half kh_exact(float x) { return (half)x; }
// kh_sat (x != x ? 0 : clamp(x, 0, 1), -0 kept) as the clamp with a zero passed through: minNum / maxNum give +0 for NaN
// and negative x, the zero test keeps -0 (as HIP's ko_sat for FP32)
half kh_sat(half x) { return x == (half)0 ? x : min(max(x, (half)0), (half)1); }
// a * b rounded to FP16 as its own instruction (no fusion into a following add: koh)
half kh_mul(half a, half b) { return koh(a * b); }
// kh_div: the correctly rounded FP32 quotient rounded to FP16 (innocuous double rounding)
// a / b rounded to nearest even (HIP '/'), = wg_div for every input. Operands with exponent fields in [92, 162]
// (|a|, |b| in [2^-35, 2^36): every nonzero finite FP16 value, the stage's usual FP32 operands) run wg_div's own
// sequence without its scaling: the reciprocal is formed on the scaled divisor bs in [1, 2) exactly as wg_div does and
// scaled back by the exact power of two 2^(127 - eb); every later step is then wg_div's step times a power of two with
// all values normal (quotient within 2^+-71, residuals >= 2^-82), so each rounds to the scaled bits of wg_div's result,
// and wg_div's exponent reassembly gives that same value. Other operands (zero, inf, NaN, far exponents): wg_div.
#if KO_APPROX
// a2: a rcp(b), branch free (|b| >= 2^126 scaled by 2^-64 first, see k_in_common.hlsli ki_div), plus one correction of
// the quotient from its exact residual: within about half an FP32 ulp
float ko_div(float a, float b)
{
    const float s = abs(b) < 8.5070592e+37f ? 1.0f : 5.4210109e-20f;
    const float bs = b * s, r = rcp(bs);
    const float q = a * r;
    return mad(mad(-bs, q, a), r, q) * s;
}
float ko_div_inrange(float a, float b) { return ko_div(a, b); }
#else
float ko_div_inrange(float a, float b)
{
    const uint eb = (asuint(b) >> 23) & 255u;
    const float bs = asfloat((asuint(b) & 0x807fffffu) | 0x3f800000u);
    float ys = rcp(bs);
    ys = mad(mad(-bs, ys, 1.0f), ys, ys);
    const float y = ys * asfloat((254u - eb) << 23);
    float q = a * y;
    q = mad(mad(-b, q, a), y, q);
    return mad(mad(-b, q, a), y, q);
}
float ko_div(float a, float b)
{
    const uint ea = (asuint(a) >> 23) & 255u, eb = (asuint(b) >> 23) & 255u;
    float q = ko_div_inrange(a, b);
    [branch] if (ea - 92u > 70u || eb - 92u > 70u) q = wg_div(a, b);
    return q;
}
#endif
// Exhaustively proven 0-ULP FP32 exact across all 2047 x 2047 nonzero finite FP16 significand pairs (even under +/-1 ULP
// v_rcp_f32 error); [branch] on the rare non-finite/zero guard also keeps the compiler from hoisting the second
// sub-sample pair's E[12] across the first (+10 VGPRs without it).
half kh_div(half a, half b)
{
    const float fa = (float)a, fb = (float)b;
    const float r = rcp(fb);
    const float q = fa * r;
    const uint ua = asuint(fa), ub = asuint(fb);
    float res = mad(mad(-fb, q, fa), r, q);
    [branch] if (((ua & 0x7f800000u) - 1u >= 0x7f7fffffu) || ((ub & 0x7f800000u) - 1u >= 0x7f7fffffu))
        res = q;
    return kh_round(res);
}

// ---- binary64 log / 2^x (k_fp.h kd_*) ---------------------------------------------------------------------------------
// HIP's kf_ln / kf_exp2 (FP32 double-word path, binary64 path for undecided inputs) equal (float)kd_log((double)x) /
// (float)kd_exp2((double)y) on every input (k_fp.h: exhaustive GPU check, every normal positive binary32 input for
// log2 / ln and every input in [-125, 127) for 2^x; outside those domains HIP takes the binary64 path itself). The port
// therefore runs the binary64 functions directly (rare far branches of kio_encode / kio_decode only; the double-word
// path is a speed form for a later round). Doubles: IEEE operations of the D3D12 doubles (division / fma need the
// extended-doubles feature, as model M's exposure port uses); the double -> float rounding is done in software
// (RNE, as v_cvt_f32_f64) so it does not depend on the driver's conversion mode.
float kd_to_f(double d)
{
    uint lo, hi;
    asuint(d, lo, hi);
    const uint s = hi & 0x80000000u, e = (hi >> 20) & 0x7ffu;
    const uint m = ((hi & 0xfffffu) << 3) | (lo >> 29), rest = lo & 0x1fffffffu;
    const int ef = (int)e - 1023 + 127;
    uint r = s | ((uint)max(ef, 0) << 23) | m;
    r += (rest > 0x10000000u || (rest == 0x10000000u && (m & 1u) != 0u)) ? 1u : 0u;   // carries into the exponent
    r = ef >= 255 ? s | 0x7f800000u : r;
    r = ef <= 0 ? s : r;   // FP32 denormal results flush (D3D12 FTZ; the stage's values never get there)
    r = e == 0x7ffu ? s | 0x7f800000u | (((hi & 0xfffffu) | lo) != 0u ? 0x400000u | m : 0u) : r;
    return asfloat(r);
}
double kd_rcp_near2(double d)
{
    double q = (double)rcp(kd_to_f(d));
    q = fma(q, fma(-d, q, (double)1.0), q);
    q = fma(q, fma(-d, q, (double)1.0), q);
    return q;
}
#define kd_one ((double)1.0)
// ln x for finite x > 0 (kd_log), NaN / 0 / negative / inf handled as k_fp.h kd_log_special
double kd_log(double x)
{
    uint lo, hi;
    asuint(x, lo, hi);
    if (x != x || x < 0.0) return asdouble(0u, 0x7ff80000u);
    if (x == 0.0) return asdouble(0u, 0xfff00000u);
    if (((hi >> 20) & 0x7ffu) == 0x7ffu) return x;
    int e = (int)((hi >> 20) & 0x7ffu) - 1023;
    double m = asdouble(lo, (hi & 0xfffffu) | 0x3ff00000u);
    if (m > asdouble(0x667F3BCDu, 0x3FF6A09Eu)) {
        m = m * (double)0.5;
        e += 1;
    }
    const double s = (m - kd_one) * kd_rcp_near2(m + kd_one);
    const double s2 = s * s;
    double p = asdouble(0x590B2164u, 0x3FA642C8u);   // 1/23
    p = fma(p, s2, asdouble(0x18618618u, 0x3FA86186u));   // 1/21
    p = fma(p, s2, asdouble(0xBCA1AF28u, 0x3FAAF286u));   // 1/19
    p = fma(p, s2, asdouble(0x1E1E1E1Eu, 0x3FAE1E1Eu));   // 1/17
    p = fma(p, s2, asdouble(0x11111111u, 0x3FB11111u));   // 1/15
    p = fma(p, s2, asdouble(0x13B13B14u, 0x3FB3B13Bu));   // 1/13
    p = fma(p, s2, asdouble(0x745D1746u, 0x3FB745D1u));   // 1/11
    p = fma(p, s2, asdouble(0x1C71C71Cu, 0x3FBC71C7u));   // 1/9
    p = fma(p, s2, asdouble(0x92492492u, 0x3FC24924u));   // 1/7
    p = fma(p, s2, asdouble(0x9999999Au, 0x3FC99999u));   // 1/5
    p = fma(p, s2, asdouble(0x55555555u, 0x3FD55555u));   // 1/3
    p = fma(p, s2, kd_one);
    const double l = kod((double)2.0 * s) * p;   // 2 s p as (2 s) p
    return fma((double)e, asdouble(0xFEFA39EFu, 0x3FE62E42u), l);
}
double kd_ldexp(double v, int n)
{
    [unroll] for (int i = 0; i < 2; ++i) {
        const int k = n > 1023 ? 1023 : (n < -1022 ? -1022 : n);
        v = v * asdouble(0u, (uint)(k + 1023) << 20);
        n -= k;
    }
    return v;
}
double kd_exp2(double y)
{
    if (y != y) return y;
    if (y > 1100.0) return asdouble(0u, 0x7ff00000u);
    if (y < -1100.0) return 0.0;
    const double mg = asdouble(0u, 0x43380000u);   // 1.5 2^52: (y + mg) - mg = rint(y) for |y| < 2^51
    const double n = kod(y + mg) - mg;
    const double r = kod((y - n) * asdouble(0xFEFA39EFu, 0x3FE62E42u));
    double p = kd_one;
    p = fma(p, r * asdouble(0x00000000u, 0x3FB00000u), kd_one);   // 1/16
    p = fma(p, r * asdouble(0x11111111u, 0x3FB11111u), kd_one);   // 1/15
    p = fma(p, r * asdouble(0x92492492u, 0x3FB24924u), kd_one);   // 1/14
    p = fma(p, r * asdouble(0x13B13B14u, 0x3FB3B13Bu), kd_one);   // 1/13
    p = fma(p, r * asdouble(0x55555555u, 0x3FB55555u), kd_one);   // 1/12
    p = fma(p, r * asdouble(0x745D1746u, 0x3FB745D1u), kd_one);   // 1/11
    p = fma(p, r * asdouble(0x9999999Au, 0x3FB99999u), kd_one);   // 1/10
    p = fma(p, r * asdouble(0x1C71C71Cu, 0x3FBC71C7u), kd_one);   // 1/9
    p = fma(p, r * asdouble(0x00000000u, 0x3FC00000u), kd_one);   // 1/8
    p = fma(p, r * asdouble(0x92492492u, 0x3FC24924u), kd_one);   // 1/7
    p = fma(p, r * asdouble(0x55555555u, 0x3FC55555u), kd_one);   // 1/6
    p = fma(p, r * asdouble(0x9999999Au, 0x3FC99999u), kd_one);   // 1/5
    p = fma(p, r * asdouble(0x00000000u, 0x3FD00000u), kd_one);   // 1/4
    p = fma(p, r * asdouble(0x55555555u, 0x3FD55555u), kd_one);   // 1/3
    p = fma(p, r * asdouble(0x00000000u, 0x3FE00000u), kd_one);   // 1/2
    p = fma(p, r, kd_one);                                        // r * 1
    return kd_ldexp(p, (int)n);
}
float kf_exp2(float y) { return kd_to_f(kd_exp2((double)y)); }
float kf_ln(float x) { return kd_to_f(kd_log((double)x)); }

// spec 42 decode factor / spec 43 encode (k_io.hip kio_decode / kio_encode)
#define kKioDecodeNear kf_bits(0x3F7FBE88u)
#define kKioEncodeNear 1000.0f
float kio_decode_far(float t)
{
    const float e = kof(kof((t + kf_bits(0x3C4FAE29u)) * 9.0f) * kf_bits(0x3FB8AA3Bu));
    return wg_div(kf_exp2(e) - 1.0f, 9.0f * t);
}
float kio_decode(float t)
{
    [branch] if (t < kKioDecodeNear) return ko_div(1.0f, 1.0f - t);
    return kio_decode_far(t);
}
float kio_encode(float Y)
{
    [branch] if (Y < kKioEncodeNear) return ko_div(1.0f, Y + 1.0f);
    return wg_div(wg_div(kf_ln(mad(Y, 9.0f, 1.0f)), 9.0f) - kf_bits(0x3C4FAE29u), Y);
}

// ---- frame view (ko_params): the exposure scalars from xs ------------------------------------------------------------
float ko_kappa() { return wg_ldf(f_xs, 0u); }
float ko_rho() { return wg_ldf(f_xs, 4u); }
float ko_kinv() { return wg_ldf(f_xs, 12u); }

// colour texel (x, y) of the dense render image (k_io.hip kio_texel; subrect base 0): RGBA f32, or RGBA16F (KIO_C16)
float4 kio_texel(int x, int y)
{
#if defined(WG_DIRECT_TEXTURES)
    // direct form: the game's colour texel as the interop's input pass writes it into the dense buffer for model K: raw
    // RGB (not divided by the pre-exposure: model K applies kappa) and the texel's own alpha, unmodified (issue #26:
    // the output alpha follows it, ko_color_tap's A). A colour format without an alpha channel reads 1, the value
    // the texture fetch defines for a missing component. (Not nd_raw_color_xy, model M's form with alpha +0, which K's
    // input stage keeps: it reads RGB only.)
    return native_direct_color_tex.Load(int3((int)nd_rx + x, (int)nd_ry + y, 0));
#else
    const uint i = ((uint)y * (uint)f_cpitch + (uint)x) * 4u;
    [branch] if ((f_flags & KIO_C16) != 0) {
        const uint2 h = wg_ld64(f_color, i * 2u);
        return float4(wg_h2f_lo(h.x), wg_h2f_hi(h.x), wg_h2f_lo(h.y), wg_h2f_hi(h.y));
    }
    return wg_ld4f(f_color, i * 4u);
#endif
}

float ko_depth_at(int x, int y)   // dense depth texel (x, y) inside the render image
{
#if defined(WG_DIRECT_TEXTURES)
    return native_direct_depth_tex.Load(int3((int)nd_rx + x, (int)nd_ry + y, 0));
#else
    return wg_ldf(f_depth, ((uint)y * (uint)f_zpitch + (uint)x) * 4u);
#endif
}

// k_io.hip ko_motion (subrect base 0). Direct form: the raw value of the game's motion texture t2 (not model M's
// nd_motion_xy, which applies b1's scale and offset per texel); ko_dsp / ko_dmo apply them (below).
// Direct forms (issue #26): the motion scale and offset of the game's motion texture t2 come in b1 (ms, mo; the
// executor passes 1 / 0 in WgRtFrame), and they are applied as the buffer forms and the reference (clean/k/reference
// instage.py / outstage.py, spec 42 / 43) apply WgRtFrame's mv_scale / mv_bias: the texel's raw value r plus the offset
// in raw units, times the motion factor with the scale folded in (sp = RN(s / RN(RN(1 / n) r)) of the host, s = the
// scale). Before, the direct forms scaled every texel (RN(ms r) + mo: input; one fused mad(ms, r, mo): output) and
// used the factor of scale 1: with a non-unit scale the products round differently, which moved history taps (0.02-0.04 %
// of the output and history values differed from the buffer form from the first frame after a reset, in-game packs).
// Offset: b1's mo is in render pixels after the scale (mo = ms b, one FP32 product); the raw-unit offset is RN(mo / ms)
// (exactly 0 for mo = 0; equal to the buffer drive's b whenever that quotient recovers it).
void ko_motion(int x, int y, out float mvx, out float mvy)
{
#if defined(WG_DIRECT_TEXTURES)
    const float2 r = native_direct_motion_tex.Load(int3((int)nd_rx + x, (int)nd_ry + y, 0)).xy;
    mvx = r.x;
    mvy = r.y;
#else
    const uint t = (uint)y * (uint)f_mpitch + (uint)x;
    [branch] if ((f_flags & KIO_M16) != 0) {
        const uint i = t * (uint)f_mch;
        mvx = wg_h2f(wg_ld16(f_motion, i * 2u));   // KIO_M16: the raw value (ko_dsp / ko_dmo apply ms / mo)
        mvy = wg_h2f(wg_ld16(f_motion, i * 2u + 2u));
        return;
    }
    const float2 m = asfloat(wg_ld64(f_motion, t * 8u));
    mvx = m.x;
    mvy = m.y;
#endif
}

// the motion factor with a texture's scale folded in (direct forms: b1's ms; KIO_M16: f_rmsx ..): k_io.py out_uniforms'
// spx = s / rhox [MVlo], else s; s = sx ms; and the raw-unit offset
// KIO_M16 (the interop's raw-buffer route: input copies or model K's compact input pass, ms / mo in KFrameIn rmsx ..,
// WgRtFrame's scale 1): the raw FP16 value r, the scale folded into the motion factor and the offset RN(mo / ms) in raw
// units, as for the direct forms (issue #26; before: ms r + mo per texel, then the factor of scale 1).
float ko_dsp(float s, float ms, float rho, bool mvlo) { const float sm = s * ms; return mvlo ? ko_div(sm, rho) : sm; }
float ko_dmo(float o, float mo, float ms) { return mo == 0.0f ? o : o + ko_div(mo, ms); }

// spec 42 5.2 [MVlo]: offset of the nearest-depth motion pixel (k_io.hip ko_nearest; kio_tap(i, d, 0, n) = clamp)
void ko_nearest(int ix, int iy, out int odx, out int ody)
{
    float c, a, b, d, e;
#if !defined(WG_DIRECT_TEXTURES)
    // HIP KO_ADDR_FAST: centre at least one texel inside the render image: nothing is clamped, the five taps are one
    // byte offset (of texel (ix - 1, iy - 1)) + 0 / 1 / 2 pitches and 0 / 1 / 2 texels
    [branch] if (ix >= 1 && ix <= f_rw - 2 && iy >= 1 && iy <= f_rh - 2) {
        const uint zp = (uint)f_zpitch * 4u, voff = ((uint)(iy - 1) * (uint)f_zpitch + (uint)(ix - 1)) * 4u;
        c = wg_ldf(f_depth + zp + 4u, voff);
        a = wg_ldf(f_depth, voff);
        b = wg_ldf(f_depth + 8u, voff);
        d = wg_ldf(f_depth + 2u * zp, voff);
        e = wg_ldf(f_depth + 2u * zp + 8u, voff);
    } else
#endif
    {
        const int x0 = ko_clampi(ix - 1, f_rw), x1 = ko_clampi(ix, f_rw), x2 = ko_clampi(ix + 1, f_rw);
        const int y0 = ko_clampi(iy - 1, f_rh), y1 = ko_clampi(iy, f_rh), y2 = ko_clampi(iy + 1, f_rh);
        c = ko_depth_at(x1, y1);
        a = ko_depth_at(x0, y0);
        b = ko_depth_at(x2, y0);
        d = ko_depth_at(x0, y2);
        e = ko_depth_at(x2, y2);
    }
    const float k = kf_bits(0x3A83126Fu);
    const bool inv = (f_flags & KIO_INV) != 0;
    bool edge;
    if (inv)
        edge = (c < 0.5f * (a + e) - k * c) || (c < 0.5f * (b + d) - k * c);
    else
        edge = (c > mad(c, k, 0.5f * (a + e))) || (c > mad(c, k, 0.5f * (b + d)));
    int bx = 0, by = 0;
    if (edge) {
        float best = c;
        const float v[4] = {a, b, d, e};
        const int ox[4] = {-1, 1, -1, 1}, oy[4] = {-1, -1, 1, 1};
        [unroll] for (int q = 0; q < 4; ++q) {
            const bool nearer = inv ? v[q] > best : v[q] < best;
            if (nearer) {
                best = v[q];
                bx = ox[q];
                by = oy[q];
            }
        }
    }
    odx = bx;
    ody = by;
}

// ---- output colour taps (spec 43 4.2 / 5.3; k_io.hip ko_stored_tap_from_raw, k_out_color) ---------------------------
void kio_ycocg(float R, float G, float B, out float Y, out float Co, out float Cg)
{
    Y = (0.25f * R + 0.5f * G) + 0.25f * B;
    Co = mad(R, 0.5f, kof(0.0f) * G) - 0.5f * B;   // the zero opaque: 0 G stays an operation (NaN for inf G)
    Cg = (0.5f * G - 0.25f * R) - 0.25f * B;
}
float kio_luma_fma(float R, float G, float B) { return mad(B, 0.25f, mad(R, 0.25f, 0.5f * G)); }

// stored tap (Y, Co, Cg, A) as 2 x 2 FP16 bits, and the filter-tap value T, of a raw colour c
void ko_color_tap(float4 c, out uint2 tap, out uint tT)
{
    const float kappa = ko_kappa();
    const bool hdr = (f_flags & KIO_HDR) != 0;
    float Y, Co, Cg;
    kio_ycocg(max(c.x, 0.0f), max(c.y, 0.0f), max(c.z, 0.0f), Y, Co, Cg);
    uint hY, hCo, hCg;
    // each product as written: kappa Y, (kappa Co) e, (kappa Cg) e (opaque, so the shared kappa / e are not regrouped)
    const float kY = kof(kappa * Y), kCo = kof(kappa * Co), kCg = kof(kappa * Cg);
    [branch] if (hdr) {
        const float e = kio_encode(kY);
        hY = ko_f2h(kY * e);
        hCo = ko_f2h(kCo * e);
        hCg = ko_f2h(kCg * e);
    } else {
        hY = ko_f2h(kY);
        hCo = ko_f2h(kCo);
        hCg = ko_f2h(kCg);
    }
    tap = uint2(hY | (hCo << 16), hCg | (ko_f2h(c.w) << 16));
    const float kL = kof(kappa * kio_luma_fma(max(c.x, 0.0f), max(c.y, 0.0f), max(c.z, 0.0f)));
    float vT = kL;
    [branch] if (hdr) vT = kL * kio_encode(kL);
    tT = ko_f2h(vT);
}

// ---- output stage helpers (k_io.hip) ----------------------------------------------------------------------------------
// beta quotient of the axis coefficients (ko_div_axis: v_rcp_f32, one Newton step, residual correction)
float ko_div_axis(float a, float b)
{
#if KO_APPROX  // a2: without the Newton step
    return a * rcp(b);
#endif
    float r = rcp(b);
    r = mad(mad(-b, r, 1.0f), r, r);
    const float q = a * r;
    return mad(mad(-b, q, a), r, q);
}
// selector normalisation: h(nv * rcp(h(1 + |nv|)))
half ko_selector_normalize(half nv)
{
    const half den = koh((half)1 + abs(nv));   // opaque: no FP16 add folded into the FP32 conversion
    return kh_round((float)nv * rcp((float)den));
}

struct KoAxis {   // spec 43 3.6 per-axis cubic coefficients
    int n;
    float c0, c1, c2, c3, beta, betap;
};
KoAxis ko_axis(float w)
{
    KoAxis a;
    const float q = mad(w - 0.5f, 2.0f, 0.5f);
    const float t = floor(q - 0.5f) + 0.5f;
    const float f = ko_sat(q - t);
    const float f2 = f * f, f3 = kof(f2) * f;
    a.c0 = f2 - 0.5f * (f + f3);
    a.c1 = (1.5f * f3 - 2.5f * f2) + 1.0f;
    a.c3 = 0.5f * (f3 - f2);
    a.c2 = ((1.0f - a.c0) - a.c1) - a.c3;
    a.beta = ko_div_axis(a.c1, a.c1 + a.c2);
    a.betap = 1.0f - a.beta;
    const float nn = t - 0.5f;
    const int nc = (int)max(min(nn, 1.0e9f), -1.0e9f);
    a.n = nn != nn ? -1000000000 : nc;
    return a;
}

float ko_ell(half v)
{
    const float x = (float)v, c = kf_bits(0x3F7D70A4u);
    return x < c ? x : mad(x - c, 0.001f, c);
}
float ko_cubic(float Am1, float A0, float A1, float A2, KoAxis a)
{
    const float lam = 1.37f;
    float r = mad(a.beta, A0, a.betap * A1);
    r = mad(lam * a.c0, ko_sat(Am1), r);
    r = mad(lam * (a.c1 - a.beta), ko_sat(A0), r);
    r = mad(lam * (a.c2 - a.betap), ko_sat(A1), r);
    return mad(lam * a.c3, ko_sat(A2), r);
}
// ko_cubic for taps known to lie in [0, 1) (ko_sat is then the identity): the same operations without the clamps
float ko_cubic_unit(float Am1, float A0, float A1, float A2, KoAxis a)
{
    const float lam = 1.37f;
    float r = mad(a.beta, A0, a.betap * A1);
    r = mad(lam * a.c0, Am1, r);
    r = mad(lam * (a.c1 - a.beta), A0, r);
    r = mad(lam * (a.c2 - a.betap), A1, r);
    return mad(lam * a.c3, A2, r);
}
// Correctly rounded reciprocal: one Newton step on v_rcp_f32 in [2^-32, 2^32] (HIP ko_rcp_refined, exhaustive GPU check),
// the IEEE division elsewhere
float ko_rcp_refined(float x)
{
#if KO_APPROX  // a2: without the Newton step
    return rcp(x);
#endif
    [branch] if (x >= asfloat(0x2f800000u) && x <= asfloat(0x4f800000u)) {
        const float r = rcp(x);
        return mad(mad(-x, r, 1.0f), r, r);
    }
    return wg_div(1.0f, x);
}
// the same without the guard, for denominators the callers bound to [2^-32, 2^32] (ko_rcp_output_bounded)
float ko_rcp_output_bounded(float x)
{
    const float r = rcp(x);
    return mad(mad(-x, r, 1.0f), r, r);
}
