// SPDX-License-Identifier: MIT
// Fast path of the spec-54 output stage (included at the end of rr_output.hlsl; same argument block and helpers).
// Packed head inputs only (packed_head bit 0). Three dispatches replace rr_output_main + rr_output_pool:
//
//   rr_output_prep      per render pixel: H(C(Cnet)), the decoded gates, H(P K2n) / H(P K2x2n) and the decoded candidate
//                       weights (the values rr_output_main recomputes for each of its 16 taps / bilinear taps), stored
//                       as binary16 records in the scratch area; checks of the TR table;
//   rr_output_fast      one lane per history pixel, 32 x 8 pixels per workgroup; the tile's render records in LDS; the
//                       16-tap filter in native binary16 arithmetic; the rest in f32 hardware operations; the 2 x 2
//                       pool (PH1) of the tile from LDS;
//   rr_output_fallback  rr_output_main's and rr_output_pool's own code (out_main_pixel, out_pool_quad) for the 2 x 2
//                       quads the fast kernel listed.
//
// EXACTNESS (gpu/output_probe.py on the GPU, against the helpers of rr_output.hlsl; gpu/output_diff.py end to end):
//  * exp2 / log2 without the ftz around them = X / L, min(max(x, 0), 1) = S and the driver conversion 0x1c = H for all
//    2^32 inputs; every nonzero |log2 x| is >= 2^-30;
//  * f16 multiply / add of two f16 values = HM / HA for all 2^32 pairs; f16 fma = HF except for the NaN payload when
//    two or more operands are NaN, which cannot reach an output (comment at the filter sums);
//  * hardware a * b, a + b, fma(a, b, c) (FP32 denormals flushed) = M, A, F on 2.2M sampled operand sets (random bit
//    patterns, denormal operands, results at the smallest normal), except one case: an exact product / fma result in
//    the band [2^-126 - 2^-150, 2^-126), which M / F round up to +-2^-126 and the hardware flushes to +-0.
//    A product / fma that can reach the band is checked (Mq / Fq: a zero result whose exact value, scaled by 2^64, lies
//    near the band); the quad of a checked pixel goes to the fallback. The others (Mn / Fn / An) cannot reach it, by
//    the magnitude or granularity of their operands (comments at each), given the uniform conditions of of_insane()
//    and the TR table checks of the prep pass; when those fail, every quad goes to the fallback.
//  * fences (an XOR with an opaque zero) where the driver could regroup products by uniform factors or fold an
//    operation into a conversion (clean/dx12/README.md "Driver exceptions").

// clean/rr-f-a14-tail (same bits): OFX_LD 1 / OFX_LD2 1 load binary16 texels as whole dwords (history taps one b64,
// W records one b64, motion_non one b32) and hand the halves to v_fma_mix through op_sel (static VMEM 124 -> 68);
// OFX_LD 2 widens them with f16tof32 instead (not measured). OFX_SPEC 1: the runtime's fixed flags (kOutFastFlags:
// motion and history binary16, no diagnostics) at compile time; gpu/output.py kernel_set(spec=True), the generic
// kernel stays for the harness.
#ifndef OFX_LD
#define OFX_LD 1
#endif
#ifndef OFX_LD2
#define OFX_LD2 1
#endif
#ifndef OFX_SPEC
#define OFX_SPEC 0
#endif
#ifndef OFX_LD3
#define OFX_LD3 0
#endif
#ifndef OFX_PH1W
#define OFX_PH1W 0  // D7: PH1 pooled inside each wave64 (no groupshared copy, no barrier)
#endif
#ifndef OFX_QST
#define OFX_QST 0   // D17: Q texels of two neighbour lanes as one 12-byte store (even history width)
#endif
#define OF_TX 32
#define OF_TY 8
#define OF_LW 30
#define OF_LH 12
groupshared uint4 of_ra[OF_LW * OF_LH];
groupshared uint of_rb[OF_LW * OF_LH];
groupshared float3 of_lin[OF_TX * OF_TY];
groupshared uint of_rowrare[OF_TY];

// scratch (bytes from the scratch argument): word 0 the fallback count (cleared by prep), word 1 the prep flags (bit 0:
// every quad falls back, bit 1: the TR table has nonzero entries below 2^-26; cleared by the fallback kernel), words
// 2-3 the last frame's words 0-1 (diagnostics), words 4-15 uniform values (prep); the list of quad indices from byte 256; then the render records:
// A (16 bytes), B (4 bytes), W (8 bytes) planes.
uint of_pw() { return (wh + 1u) / 2u; }
uint of_list_bytes() { return 256u + of_pw() * ((hh + 1u) / 2u) * 4u; }
uint of_rec_a() { return scratch + ((of_list_bytes() + 255u) & ~255u); }
uint of_rec_b() { return of_rec_a() + wr * hr * 16u; }
uint of_rec_w() { return of_rec_b() + ((wr * hr * 4u + 15u) & ~15u); }

uint of_zb() { return 0u - (wg_grid_ >> 31); }
#if WG_WAVE == 64
#include "wg_ags.hlsli"
#endif
uint of_cvt(float x) { return wg_ags_call(0x1cu, 0u, 1u, asuint(x), 0u); }      // v_cvt_f16_f32 (RNE) = H's bits
// OF_FAST_WAVE=64 (exact, same bits; host define, gpu/output.py): rr_output_fast as wave64. The VALU-bound kernel
// then issues its FP32 operations two halves per wave instruction; the driver's matrix conversion (wave32 only) is
// replaced by the scalar conversion 0x1c per value, the same instruction v_cvt_f16_f32 (RNE) = H for all 2^32 inputs.
half of_hh(uint b) { return asfloat16((uint16_t)(b & 0xffffu)); }
#if WG_WAVE == 64
uint4 of_to_f16(v8f d)
{
    uint h[8];
    [unroll] for (uint i = 0; i < 8; ++i) h[i] = of_cvt(d.v[i]);
    return uint4(h[0] | (h[1] << 16), h[2] | (h[3] << 16), h[4] | (h[5] << 16), h[6] | (h[7] << 16));
}
#else
uint4 of_to_f16(v8f d) { return wg_to_f16_ags(d); }
#endif
half of_fh(half x) { return asfloat16((uint16_t)(asuint16(x) ^ (uint16_t)of_zb())); }
half2 of_fh2(half2 x) { return wg_as_h2(wg_h2_bits(x) ^ of_zb()); }   // keeps the last fma f16 (else v_fma_mix_f32)
float of_f(half x) { return f16tof32((uint)asuint16(x)); }
// wg_div((float)b, 255) for every byte b (clean/rr-f-r6-tail): q = b * RN(1/255), one fma correction step; checked with
// exact rationals for all 256 inputs (results 0 or >= 1/255: no denormal, no special operand), and on the GPU.
float of_d255(uint b)
{
    const float a = (float)b, rc = asfloat(0x3b808081u);   // RN(1 / 255)
    const float q = a * rc;
    return mad(mad(-255.0f, q, a), rc, q);
}

#ifndef OF_NO_FALLBACK
#ifndef OF_DEDUP
#define OF_DEDUP 1         // exact: candidates with an equal motion share one reprojection
#endif
#ifndef OF_REPROJ_MAX
#define OF_REPROJ_MAX 3    // 3 = exact (default); 1 / 2: NOT exact, opt-in only (WG_RR_OUT_DEFINES)
#endif
#define OF_NO_FALLBACK 0   // negative control only: never list a quad
#endif
// A second opaque zero: the driver splits a fused mad(a, b, c) into a * b + c when it can share the product a * b with
// another use (seen in of_blend), so products of the same operands get distinct fences (fence / fence2).
float fence2(float x) { return asfloat(asuint(x) ^ (0u - (wg_group_base_ >> 31))); }
float Mn(float a, float b) { return a * b; }
#ifndef OF_MFMA
#define OF_MFMA 1          // round 13: fenced products as fma(a, b, opaque -0) (the of_mhf form; one VALU, the same bits)
#endif
#if OF_MFMA
float Mf(float a, float b) { return mad(a, b, asfloat(0x80000000u ^ of_zb())); }
#else
float Mf(float a, float b) { return fence(a * b); }
#endif
float An(float a, float b) { return a + b; }
float Fn(float a, float b, float c) { return mad(a, b, c); }
// A zero result is checked when the exact result, scaled by 2^64 (exact for every operand set whose exact result is
// below 2^-126), lies in [2^-62 (1 - 2^-20), 2^-62]: a superset of the band; exact cancellations and products far below
// it (which M / F also flush) are not listed.
// (and() / or() / select(): no short-circuit branches)
bool of_win(float s2) { return and(abs(s2) >= asfloat(0x207ffff0u), abs(s2) <= asfloat(0x20800000u)); }
// the scaled check only where the result is 0 (a branch, skipped by waves without such a lane)
#ifndef OFX_MQB
#define OFX_MQB 0          // clean/rr-f-a14-tail D36: 1 = the scaled check without the branch (same flags, same bits;
#endif                     // measured slower: +0.035 / +0.078 ms, the branch skips the check in most waves)
#ifndef OFX_NOCHK
#define OFX_NOCHK 0        // D31, FAST MODE ONLY (not exact): no band check (results in the band flush like the hardware)
#endif
#if OFX_NOCHK
float Mq(float a, float b, inout bool rare) { return Mf(a, b); }
float Fq(float a, float b, float c, inout bool rare) { return fence(mad(a, b, c)); }
#elif OFX_MQB
float Mq(float a, float b, inout bool rare)
{
    const float r = Mf(a, b);
    rare = or(rare, and(r == 0, of_win(fence(a * asfloat(0x5f800000u)) * b)));
    return r;
}
float Fq(float a, float b, float c, inout bool rare)
{
    const float r = fence(mad(a, b, c));
    rare = or(rare, and(r == 0, of_win(mad(fence(a * asfloat(0x5f800000u)), b, fence(c * asfloat(0x5f800000u))))));
    return r;
}
#else
float Mq(float a, float b, inout bool rare)
{
    const float r = Mf(a, b);
    [branch] if (r == 0) rare = or(rare, of_win(fence(a * asfloat(0x5f800000u)) * b));
    return r;
}
float Fq(float a, float b, float c, inout bool rare)
{
    const float r = fence(mad(a, b, c));
    [branch] if (r == 0) rare = or(rare, of_win(mad(fence(a * asfloat(0x5f800000u)), b, fence(c * asfloat(0x5f800000u)))));
    return r;
}
#endif
// clean/rr-mp1-output O1 (mpeak probe P8 "p8a"; same bits by construction, byte-checked; default on in output.fast and
// output.fast.f, measured -0.040 ms (accurate) / -0.056 ms (fast) output stage at 1440p Q; OFX_GZ=0 OFX_M24=0 = before):
//  OFX_GZ 1: grouped zero-result checks. Each group of up to three checked results r_i takes one branch on
//    z = min(|r0|, |r1|, |r2|) == 0 (v_min3_num_f32 with abs modifiers, one compare) instead of one compare and branch
//    per result; inside it each site's own predicate and(r_i == 0, of_win(scaled)) (the expression of Mq / Fq) is OR-ed
//    into rare. z == 0 iff some r_i is +-0: min_num returns the non-NaN operand and a NaN is never a zero result; FP32
//    mul / fma results are never denormal (FTZ), so min / compare cannot flush a nonzero r_i. Every predicate contains
//    its own r_i == 0, so the group branch is taken whenever any predicate can be true: rare is the per-site value.
//    The checked set is exactly the shipped kernel's 42 sites. (The source's nested calls Mq(norm3, Mq(total, K1, rare),
//    rare), Df(Mq(acc, Rt, rare), rare) and Fq(alpha, Fq(current, K12, -hist, rare), hist, rare) lose the inner call's
//    update of rare to HLSL's inout copy-out, so DXC deletes those 10 inner checks; the grouped form leaves them out
//    explicitly. The blend's inner six are provably unneeded (probe P8 (ii)); the other four are an open question for
//    the owner (P8 section 3, "p8b"), unchanged here.) Not with OFX_NOCHK (no checks at all).
//  OFX_M24: 24-bit index products (v_mul_u32_u24, full rate, instead of v_mul_lo_u32) where both operands are provably
//    below 2^24 at any output size: bit 1 the record-fill row rr * W (rr < 12, W <= 30 when the fill loop runs), bit 8
//    the PH1 pixel count (both operands <= 2). The low 32 bits of the product are the same.
#ifndef OFX_GZ
#define OFX_GZ 1
#endif
#ifndef OFX_M24
#define OFX_M24 9
#endif
#define OF_GZ (OFX_GZ && !OFX_NOCHK)
// the per-site predicates of Mq / Fq (the same expressions as inside their branches) and the group key
bool of_cm(float r, float a, float b) { return and(r == 0, of_win(fence(a * asfloat(0x5f800000u)) * b)); }
bool of_cf(float r, float a, float b, float c) { return and(r == 0, of_win(mad(fence(a * asfloat(0x5f800000u)), b, fence(c * asfloat(0x5f800000u))))); }
float of_z3(float a, float b, float c) { return min(min(abs(a), abs(b)), abs(c)); }
float of_fqr(float a, float b, float c) { return fence(mad(a, b, c)); }   // Fq's result
// R of rr_output.hlsl (1 / x by wg_div, FTZ) for all 2^32 inputs (gpu/output_probe.py): wg_div's correction steps
// without its exponent scaling for normal results, the reference's special values elsewhere.
#ifndef OF_HWRCP
#define OF_HWRCP 0   // 1 (opt-in, erratum 51-58 / 54 "within 1 ulp"): R = the hardware reciprocal (FP32 denormals flushed)
#endif
#ifndef OF_RF1
#define OF_RF1 1   // round 13: Rf by one Newton step (the same bits, exhaustive)
#endif
float Rf(float x)
{
#if OF_HWRCP
    return rcp(x);
#endif
    float y = rcp(x);
#if OF_RF1
    // round 13: one Newton step from the hardware reciprocal (rf_rcp's form) gives the same bits as the three-step form
    // for every normal-range input (gpu/output_probe.py R_one_step: 0 of 2^32 differ from R of rr_output.hlsl)
    const float q = mad(mad(-x, y, 1.0f), y, y);
#else
    y = mad(mad(-x, y, 1.0f), y, y);
    float q = mad(mad(-x, y, 1.0f), y, y);
    q = mad(mad(-x, q, 1.0f), y, q);
#endif
    const uint ax = asuint(x) & 0x7fffffffu, sg = asuint(x) & 0x80000000u;
    uint sp = ax < 0x00800000u ? (sg | 0x7f800000u) : (ax > 0x7f800000u ? (asuint(x) | 0x400000u) : sg);
    sp = ax == 0x7e800000u ? sg | 0x00800000u : sp;
    return asfloat(and(ax >= 0x00800000u, ax < 0x7e800000u) ? asuint(q) : sp);
}
bool of_small(float k, float lo) { return and(k != 0, abs(k) < lo); }   // NaN: false
// Uniform conditions behind the unchecked products (else every quad falls back): a value of D (exp2 - 1) is 0 or
// >= 2^-24 in magnitude, a value of log2 0 or >= 2^-30; with these bounds the chains below stay >= 2^-114.
bool of_insane(float E, float Ep, float invE)
{
    const float lo = asfloat(0x30800000u);   // 2^-30
    return of_small(K12, lo) || of_small(invE, lo) || of_small(Ep, lo) || abs(Ep) > asfloat(0x4e800000u) ||
           of_small(E, lo) || of_small(K7, lo) || of_small(K11, asfloat(0x21800000u)) || K13 > asfloat(0x4e800000u);
}
// C: F(x, K4, 1) cannot be a nonzero value below 2^-126 (x K4 + 1 is 0, or >= 2^-48, or near 1); |log2| >= 2^-30, so
// with |K7| >= 2^-30 both products stay >= 2^-62 (or 0).
float Cf(float x) { return Mf(Mf(log2(max(Fn(x, K4, 1), K5)), asfloat(0x3e9a209bu)), K7); }
float of_dt(float x) { return or(x > K5, isnan(x)) ? (or(isnan(x), isnan(K10)) ? asfloat(0x7fc00000u) : min(x, K10)) : K5; }
float of_dtail(float r) { return An(exp2(Mn(r, asfloat(0x40549a78u))), -1); }   // Df after its checked product
float Df(float x, inout bool rare)
{
    float t = of_dt(x);
    // M(M(t, K8), K9): K9 > 1 cannot make a normal (or zero) first product small
    return An(exp2(Mn(Mq(t, K8, rare), asfloat(0x40549a78u))), -1);
}

// sample4 of rr_output.hlsl, one axis. p (a position) is 0 or >= 2^-25 in magnitude (a half-integer plus a value) and
// R >= 2^-16, so s = M(p, R) and M(s, n) need no check; a is then a multiple of 2^-25 (or large), so f and 1 - f are
// 0 or multiples of 2^-25 and the bilinear weights multiples of 2^-50: their products with binary16 values (multiples
// of 2^-24) never reach the band. The same holds for the candidate taps (s = M(u, 1 / wr), u a multiple of 2^-41).
struct OfAxis { float f, g; int c0, c1; uint o0, o1; };   // o0 / o1: byte offsets c0 / c1 * stride (+ base)
// clean/rr-f-s10-tail: 24-bit products (v_mul_u32_u24 / v_mad_u32_u24, full rate, instead of the quarter-rate
// v_mul_lo_u32) for index arithmetic whose operands are below 2^24 (coordinates and pitches of images whose sides are
// below 2^21): the low 32 bits of the product are the same, so every address is the same.
uint of_m24(uint a, uint b) { return (a & 0xffffffu) * (b & 0xffffffu); }
OfAxis of_axis_s(float s, uint n, uint stride = 0u, uint base = 0u, bool m24 = false)
{
    OfAxis r;
    s = isnan(s) ? 0 : s;
    const float a = An(Mn(s, n), -.5f), i = floor(a);
    // fenced: the driver turns a - floor(a) into v_fract_f32, which differs when a + 1 rounds up to 1 (a = -2^-26)
    r.f = An(a, -fence(i)); r.g = An(1, -r.f);
    r.c0 = (int)clamp(i, 0.0f, (float)(n - 1)); r.c1 = (int)clamp(An(i, 1), 0.0f, (float)(n - 1));
    r.o0 = (m24 ? of_m24((uint)r.c0, stride) : (uint)r.c0 * stride) + base;
    r.o1 = (m24 ? of_m24((uint)r.c1, stride) : (uint)r.c1 * stride) + base;
    return r;
}
OfAxis of_axis(float p, float R, uint n, uint stride, uint base, bool m24 = false) { return of_axis_s(Mf(p, R), n, stride, base, m24); }
struct OfTaps { int x0, x1, y0, y1; uint ox0, ox1, oy0, oy1; float wa[4]; };
OfTaps of_taps(OfAxis ax, OfAxis ay)
{
    OfTaps r;
    r.x0 = ax.c0; r.x1 = ax.c1; r.y0 = ay.c0; r.y1 = ay.c1; r.ox0 = ax.o0; r.ox1 = ax.o1; r.oy0 = ay.o0; r.oy1 = ay.o1;
    r.wa[0] = Mn(ax.g, ay.g); r.wa[1] = Mn(ax.f, ay.g); r.wa[2] = Mn(ax.g, ay.f); r.wa[3] = Mn(ax.f, ay.f);
    return r;
}
float of_bil(OfTaps q, float v0, float v1, float v2, float v3)
{
    return An(An(An(Mn(q.wa[0], v0), Mn(q.wa[1], v1)), Mn(q.wa[2], v2)), Mn(q.wa[3], v3));
}

// packed_head bit 2: the motion inputs (PM half4, Pe half2, Pn half2) are binary16 images; bit 5: the history (Hprev
// half4) is (the producers' own buffers; exact widening here); else decoded f32 (Hprev rgba, 16 bytes).
#if OFX_SPEC
bool of_half() { return true; }
bool of_hhalf() { return true; }
#else
bool of_half() { return (packed_head & 4u) != 0u; }
bool of_hhalf() { return (packed_head & 32u) != 0u; }
#endif
// M(w, h) of a binary16 value h as fma(w, h, -0): the same rounding, and the driver reads h as f16 (v_fma_mix_f32).
// The addend is an opaque -0: with a literal -0 the driver dropped it and fused the product into the next addition.
float of_mhf(float w, half h) { return mad(w, (float)h, asfloat(0x80000000u ^ of_zb())); }
float of_bilf(OfTaps q, half h0, half h1, half h2, half h3)
{
    return An(An(An(of_mhf(q.wa[0], h0), of_mhf(q.wa[1], h1)), of_mhf(q.wa[2], h2)), of_mhf(q.wa[3], h3));
}
float3 of_hist(OfTaps q)
{
    if (of_hhalf()) {   // binary16 rgba texels (8 bytes)
        // OFX_LD 0: half4 loads (the driver issues one 16-bit load per used channel); OFX_LD 1: the dword form feeds
        // v_fma_mix through op_sel without 16-bit copies
#if OFX_LD == 1
        const uint2 a0 = wg_mem.Load2(q.oy0 + q.ox0), a1 = wg_mem.Load2(q.oy0 + q.ox1);
        const uint2 b0 = wg_mem.Load2(q.oy1 + q.ox0), b1 = wg_mem.Load2(q.oy1 + q.ox1);
        return float3(of_bilf(q, of_hh(a0.x), of_hh(a1.x), of_hh(b0.x), of_hh(b1.x)),
                      of_bilf(q, of_hh(a0.x >> 16), of_hh(a1.x >> 16), of_hh(b0.x >> 16), of_hh(b1.x >> 16)),
                      of_bilf(q, of_hh(a0.y), of_hh(a1.y), of_hh(b0.y), of_hh(b1.y)));
#elif OFX_LD == 2
        const uint2 a0 = wg_mem.Load2(q.oy0 + q.ox0), a1 = wg_mem.Load2(q.oy0 + q.ox1);
        const uint2 b0 = wg_mem.Load2(q.oy1 + q.ox0), b1 = wg_mem.Load2(q.oy1 + q.ox1);
        return float3(of_bil(q, f16tof32(a0.x), f16tof32(a1.x), f16tof32(b0.x), f16tof32(b1.x)),
                      of_bil(q, f16tof32(a0.x >> 16), f16tof32(a1.x >> 16), f16tof32(b0.x >> 16), f16tof32(b1.x >> 16)),
                      of_bil(q, f16tof32(a0.y), f16tof32(a1.y), f16tof32(b0.y), f16tof32(b1.y)));
#else
        const half4 a0 = wg_mem.Load<half4>(q.oy0 + q.ox0), a1 = wg_mem.Load<half4>(q.oy0 + q.ox1);
        const half4 b0 = wg_mem.Load<half4>(q.oy1 + q.ox0), b1 = wg_mem.Load<half4>(q.oy1 + q.ox1);
        return float3(of_bilf(q, a0.x, a1.x, b0.x, b1.x), of_bilf(q, a0.y, a1.y, b0.y, b1.y), of_bilf(q, a0.z, a1.z, b0.z, b1.z));
#endif
    }
    const uint r0 = of_m24((uint)q.y0, wh), r1 = of_m24((uint)q.y1, wh);
    const float3 v0 = asfloat(wg_mem.Load3(hprev + (r0 + (uint)q.x0) * 16u)), v1 = asfloat(wg_mem.Load3(hprev + (r0 + (uint)q.x1) * 16u));
    const float3 v2 = asfloat(wg_mem.Load3(hprev + (r1 + (uint)q.x0) * 16u)), v3 = asfloat(wg_mem.Load3(hprev + (r1 + (uint)q.x1) * 16u));
    return float3(of_bil(q, v0.x, v1.x, v2.x, v3.x), of_bil(q, v0.y, v1.y, v2.y, v3.y), of_bil(q, v0.z, v1.z, v2.z, v3.z));
}

// reproj of rr_output.hlsl. With TR entries 0 or >= 2^-26 in magnitude (prep flag bit 1 clear) the weights are 0 or
// multiples of 2^-75 and the taps multiples of 2^-74, so every exact value of the accumulation is a multiple of 2^-149:
// none lies in the band. Otherwise (fine table) its products / fmas are checked.
float3 of_fallback(float3 current, float Ep, float invE)
{
    float3 f;
    [unroll] for (uint c = 0; c < 3; c++) f[c] = Cf(Mf(Ep, Mf(invE, Mf(fence(current[c]), K12))));   // >= 2^-114 (of_insane)
    return f;
}
// (clean/rr-f-r6-tail) the fallback colour is computed only for a candidate that falls (waves without such a lane skip it)
float3 of_reproj(uint x, uint y, float2 mv, float3 current, float Ep, float invE, out uint fell, float Rwh, float Rhh, bool fine, inout bool rare)
{
    float px = An(x + .5f, mv.x), py = An(y + .5f, mv.y);
    fell = or(or(or(px < 0, px > wh), or(py < 0, py > hh)), reset != 0 && !BAD_RESET) ? 1 : 0;
    float3 acc = 0;
    [branch] if (fell) acc = of_fallback(current, Ep, invE);
    [branch] if (!fell) {
        float kx = An(floor(An(px, -.5f)), .5f), ky = An(floor(An(py, -.5f)), .5f);
        // F(S(.), 512, .5) >= .5: no small result
        uint ix = min((uint)Fn(min(max(An(px, -kx), 0.0f), 1.0f), 512, .5f), 512u), iy = min((uint)Fn(min(max(An(py, -ky), 0.0f), 1.0f), 512, .5f), 512u);
        float4 tx = wg_ld4f(table, ix * 16), ty = wg_ld4f(table, iy * 16);
        // TR entries 0 or >= 2^-60 (prep, else everything falls back): products 0 or >= 2^-120
        float weights[5] = {Mn(tx.y, ty.x), Mn(tx.x, ty.y), Mn(tx.y, ty.y), Mn(tx.y, ty.z), Mn(tx.z, ty.y)};
        // the five positions use three distinct x and three distinct y coordinates
        // byte offsets of the binary16 texels: column * 8, row * wh * 8 + hprev
        const uint ys = wh * 8u;
        const OfAxis X0 = of_axis(An(kx, tx.w), Rwh, wh, 8u, 0u), X1 = of_axis(An(kx, -1), Rwh, wh, 8u, 0u), X2 = of_axis(An(kx, 2), Rwh, wh, 8u, 0u);
        const OfAxis Y0 = of_axis(An(ky, -1), Rhh, hh, ys, hprev, true), Y1 = of_axis(An(ky, ty.w), Rhh, hh, ys, hprev, true),
                     Y2 = of_axis(An(ky, 2), Rhh, hh, ys, hprev, true);
        [unroll] for (uint k = 0; k < 5; k++) {
            OfAxis qx = X0, qy = Y1;
            if (k == 1) qx = X1;
            if (k == 4) qx = X2;
            if (k == 0) qy = Y0;
            if (k == 3) qy = Y2;
            const OfTaps q = of_taps(qx, qy);
            const float3 tap = of_hist(q);
            if (fine) {
                [unroll] for (uint c = 0; c < 3; c++) acc[c] = k == 0 ? Mq(tap[c], weights[k], rare) : Fq(tap[c], weights[k], acc[c], rare);
            } else {
                [unroll] for (uint c = 0; c < 3; c++) acc[c] = k == 0 ? Mn(tap[c], weights[k]) : Fn(tap[c], weights[k], acc[c]);
            }
        }
        float total = An(weights[4], An(weights[3], An(weights[2], An(weights[1], weights[0]))));
        float Rt = Rf(total);
#if OF_GZ
        float3 r3;
        [unroll] for (uint c = 0; c < 3; c++) r3[c] = Mf(acc[c], Rt);
        [branch] if (of_z3(r3.x, r3.y, r3.z) == 0)
            rare = or(rare, or(or(of_cm(r3.x, acc.x, Rt), of_cm(r3.y, acc.y, Rt)), of_cm(r3.z, acc.z, Rt)));
        acc = r3;
#else
        [unroll] for (uint c = 0; c < 3; c++) acc[c] = Mq(acc[c], Rt, rare);
#endif
    }
    return acc;
}

float3 of_blend(float3 current, float3 hist, float alpha, out uint guarded, inout bool rare)
{
    float3 b;
#if OF_GZ
    // the inner fma unchecked, as in the shipped kernel (its check is lost to the inout copy-out; P8 (ii) proves it
    // unneeded: under !insane the exact fma value is a multiple of 2^-149, never in the band)
    float3 in3;
    [unroll] for (uint c = 0; c < 3; c++) in3[c] = of_fqr(current[c], K12, -hist[c]);
    [unroll] for (uint c = 0; c < 3; c++) b[c] = of_fqr(alpha, in3[c], hist[c]);
    [branch] if (of_z3(b.x, b.y, b.z) == 0)
        rare = or(rare, or(or(of_cf(b.x, alpha, in3.x, hist.x), of_cf(b.y, alpha, in3.y, hist.y)), of_cf(b.z, alpha, in3.z, hist.z)));
#else
    [unroll] for (uint c = 0; c < 3; c++) b[c] = Fq(alpha, Fq(current[c], K12, -hist[c], rare), hist[c], rare);
#endif
    guarded = isnan(An(b.z, An(b.x, b.y))) ? 1 : 0;
    float3 g;
    // >= 2^-54 (of_insane); fence2 keeps the driver from sharing this product with the fma above (two roundings)
    [unroll] for (uint c = 0; c < 3; c++) g[c] = Mn(fence2(current[c]), K12);
    return guarded ? g : b;
}

// code10 of rr_output.hlsl without binary64: p = RNE(s 1023) and its exact error e decide the half-integer ties
// (s >= 2^-11 when p >= .5, so e is a multiple of 2^-34, never flushed).
uint of_code10(float x)
{
    const float s = min(max(x, 0.0f), 1.0f), p = s * 1023.0f, e = mad(fence(s), 1023.0f, -p), fl = floor(p);   // fence: see fence2
    float a = round(p);
    a = p - fl == .5f ? (e > 0 ? fl + 1 : (e < 0 ? fl : a)) : a;
    return min(1023u, (uint)a + BAD_CODE);
}

// ---- prep: one lane per render pixel ------------------------------------------------------------------------------
// OF_PREPU 1 (clean/rr-f-r6-tail): the uniform words, the fallback-count clear and the TR checks only; the render records
// come from the head (rr_d0r.hlsl RR_OUT_PREP, the runtime's fast head), dispatched with (513 * 3) / WG_BLOCK + 1 groups
#ifndef OF_PREPU
#define OF_PREPU 0
#endif
WG_KERNEL(rr_output_prep)
{
    WG_SETUP();
    const uint n = wr * hr, i0 = wg_block_id() * WG_BLOCK + wg_lane_id(), i = min(i0, n - 1);
#if !OF_PREPU
    const uint x = i % wr, y = i / wr;
    bool rare = false;
    v8f a, b;
    [unroll] for (uint c = 0; c < 3; c++) {
        a.v[c] = Cf(wg_h2f(wg_ld16(color, ((y * wt + x) * 4 + c) * 2)));
        a.v[3 + c] = of_d255(wg_ld8(gates, (y * wr + x) * 4 + c));
    }
    const float px = wg_h2f(wg_ld16(precision, ((y * wr + x) * 4 + 0) * 2)), py = wg_h2f(wg_ld16(precision, ((y * wr + x) * 4 + 1) * 2)),
                pz = wg_h2f(wg_ld16(precision, ((y * wr + x) * 4 + 2) * 2));
    a.v[6] = Mq(px, K2n, rare); a.v[7] = Mq(py, K2n, rare);
    b.v[0] = Mq(pz, K2x2n, rare);
    [unroll] for (uint c = 0; c < 4; c++) b.v[1 + c] = of_d255(wg_ld8(candidates, (y * wr + x) * 4 + c));
    b.v[5] = 0; b.v[6] = 0; b.v[7] = 0;
    const uint4 ha = of_to_f16(a), hb = of_to_f16(b);
#else
    const bool rare = false;
#endif
    if (i0 == 0u) {
        wg_st32(scratch, 0u, 0u);   // the fallback count of this frame
        // uniform values of the fast kernel (words 4-15)
        float E = E_arg, Ep = Ep_arg, invE = invE_arg;
        if ((packed_head & 2u) != 0u) { uint pe = asuint(E_arg); E = asfloat(wg_ld32(pe, 0u)); Ep = asfloat(wg_ld32(pe, 4u)); invE = asfloat(wg_ld32(pe, 8u)); }
        wg_st128(scratch, 16u, uint4(asuint(wg_div((float)wr, (float)wh)), asuint(wg_div((float)hr, (float)hh)), asuint(R(wr)), asuint(R(hr))));
        wg_st128(scratch, 32u, uint4(asuint(R(wh)), asuint(R(hh)), asuint(E), asuint(Ep)));
        wg_st128(scratch, 48u, uint4(asuint(invE), asuint(R(Ep)), of_insane(E, Ep, invE) ? 1u : 0u, 0u));
    }
    uint flags = rare ? 1u : 0u;
    if (i0 < 513u * 3u) {   // TR weight columns 0-2
        const float t = asfloat(wg_ld32(table, (i0 / 3u) * 16u + (i0 % 3u) * 4u));
        flags |= (of_small(t, asfloat(0x21800000u)) ? 1u : 0u) | (of_small(t, asfloat(0x32800000u)) ? 2u : 0u);
    }
#if !OF_PREPU
    if (i0 < n) {
        wg_st128(of_rec_a(), i * 16, ha);
        wg_st32(of_rec_b(), i * 4, hb.x & 0xffffu);
        wg_st64(of_rec_w(), i * 8, uint2((hb.x >> 16) | (hb.y << 16), (hb.y >> 16) | (hb.z << 16)));
    }
#endif
    if (flags != 0u) { uint o; wg_mem_gc.InterlockedOr(scratch + 4u, flags, o); }
}

// ---- main: one lane per history pixel -----------------------------------------------------------------------------
// OF_PROF: shader-cycle profile of the fast kernel's sections (lane 0 of each wave adds the deltas to trace words)
#ifndef OF_PROF
#define OF_PROF 0
#endif
#if OF_PROF
static uint of_t0, of_tl;
#define OF_CLK(k) { const uint t_ = wg_ags_call(0x1fu, 0u, 0u, 0u, 0u); if ((k) > 0 && WaveIsFirstLane()) { uint o_; wg_mem_gc.InterlockedAdd(trace + 4u * (k), t_ - of_tl, o_); } of_tl = t_; }
#else
#define OF_CLK(k)
#endif
void of_cs(inout float wa, inout uint ia, inout float2 ma, inout float wb, inout uint ib, inout float2 mb)
{
    const bool keep = !(wa < wb);
    const float tw = wa; const uint ti = ia; const float2 tm = ma;
    wa = keep ? wa : wb; ia = keep ? ia : ib; ma = keep ? ma : mb;
    wb = keep ? wb : tw; ib = keep ? ib : ti; mb = keep ? mb : tm;
}
// OF_UNIF 1 (clean/rr-f-lw-fold; Fast mode's output set, runtime/tools/build.py FAST_OUT_DEFINES): the prep pass's
// uniform words (scratch words 1, 4-15: the same address in every lane, written by the prep pass before this dispatch)
// read as wave-uniform (v_readfirstlane: the same values in scalar registers); every operation is unchanged. Measured
// (1080p / 1440p Q, clock-normalised, 3 rounds): output.fast.f -0.003 / -0.010 ms; the exact output.fast was slower
// with it (+0.020 / +0.034 ms), so it stays 0 there.
#ifndef OF_UNIF
#define OF_UNIF 0
#endif
#if OF_UNIF
uint of_u1(uint v) { return WaveReadLaneFirst(v); }
uint4 of_u4(uint4 v) { return uint4(WaveReadLaneFirst(v.x), WaveReadLaneFirst(v.y), WaveReadLaneFirst(v.z), WaveReadLaneFirst(v.w)); }
#else
uint of_u1(uint v) { return v; }
uint4 of_u4(uint4 v) { return v; }
#endif
int of_cx(uint x, float sx) { return (int)floor(An(Fn(x + .5f, sx, jx), -.5f)); }
int of_cya(uint y, float sy) { return (int)floor(An(Fn((y / 2) * 2 + .5f, sy, jy), -.5f)); }

WG_KERNEL(rr_output_fast)
{
    WG_SETUP();
    OF_CLK(0);
    const uint lane = wg_lane_id(), ntx = (wh + OF_TX - 1) / OF_TX, gid = wg_block_id();

    const uint tx0 = (gid % ntx) * OF_TX, ty0 = (gid / ntx) * OF_TY;
    const uint lx = lane & (OF_TX - 1), ly = lane / OF_TX;
    const uint xo = tx0 + lx, yo = ty0 + ly;
    const bool valid = and(xo < wh, yo < hh);
    const uint x = min(xo, wh - 1), y = min(yo, hh - 1), idx = of_m24(y, wh) + x;
    // uniform values computed once by the prep pass (scratch words 4-15)
    const uint4 u0 = of_u4(wg_ld128(scratch, 16u)), u1 = of_u4(wg_ld128(scratch, 32u)), u2 = of_u4(wg_ld128(scratch, 48u));
    const float sx = asfloat(u0.x), sy = asfloat(u0.y), Rwr = asfloat(u0.z), Rhr = asfloat(u0.w);
    const float Rwh = asfloat(u1.x), Rhh = asfloat(u1.y), E = asfloat(u1.z), Ep = asfloat(u1.w);
    const float invE = asfloat(u2.x), REp = asfloat(u2.y);
    const bool insane = u2.z != 0u;
    // the tile's render records (columns cx - 1 .. cx + 2, rows cya - 1 .. cya + 2 of every pixel; cx, cya monotonic)
    const uint xl = min(tx0 + OF_TX - 1, wh - 1), yl = min(ty0 + OF_TY - 1, hh - 1);
    const int rx0 = of_cx(tx0, sx) - 1, ry0 = of_cya(ty0, sy) - 1;
    const int W = of_cx(xl, sx) + 2 - rx0 + 1, Hn = of_cya(yl, sy) + 2 - ry0 + 1;
    const bool fits = W <= OF_LW && Hn <= OF_LH;
    const uint pflags = of_u1(wg_ld32(scratch, 4u));
    const bool fine = (pflags & 2u) != 0u;
    bool rare = !fits || (pflags & 1u) != 0u || insane;
    if (fits) {
        const uint ra = of_rec_a(), rb = of_rec_b();
        const uint cc = lane & 31u;   // W <= 30: one lane per record column, WG_BLOCK / 32 rows per step
        for (uint rr = lane >> 5; rr < (uint)Hn; rr += WG_BLOCK / 32u) {
            if (cc < (uint)W) {
                const int rx = clamp(rx0 + (int)cc, 0, (int)wr - 1), ry = clamp(ry0 + (int)rr, 0, (int)hr - 1);
#if OFX_M24 & 1
                const uint r = of_m24((uint)ry, wr) + (uint)rx, i = of_m24(rr, (uint)W) + cc;   // rr < 12, W <= 30
#else
                const uint r = of_m24((uint)ry, wr) + (uint)rx, i = rr * (uint)W + cc;
#endif
                of_ra[i] = wg_ld128(ra, r * 16u);
                of_rb[i] = wg_ld32(rb, r * 4u);
            }
        }
    }
    GroupMemoryBarrierWithGroupSync();
    OF_CLK(1);

    // u = F(x + .5, sx, jx): (x + .5) sx is a multiple of 2^-40 and jx of 2^-41 where they can cancel: never small
    const float u = Fn(x + .5f, sx, jx), v = Fn(y + .5f, sy, jy), va = Fn((y / 2) * 2 + .5f, sy, jy);
    const int cx = (int)floor(An(u, -.5f)), cy = (int)floor(An(v, -.5f)), cya = (int)floor(An(va, -.5f));
    const uint hxy = of_cvt(fence(An(cx + .5f, -u))) | (of_cvt(fence(An(cy + .5f, -v))) << 16);
    const half hx = of_hh(hxy), hy0 = of_hh(hxy >> 16);
    const half hy = cy != cya ? of_fh(hy0 + (half)-1.0) : hy0;
    half ox[4], oy[4], oxx[4], oyy[4];
    [unroll] for (int k = 0; k < 4; k++) {
        // (a folded hx + 0 could only change the sign of a zero offset, which no product below can show)
        ox[k] = hx + (half)(k - 1); oy[k] = hy + (half)(k - 1);
        oxx[k] = ox[k] * ox[k]; oyy[k] = oy[k] * oy[k];
    }
    const int Wl = fits ? W : 0, lbase = fits ? (cya - 1 - ry0) * W + (cx - 1 - rx0) : 0;
#ifndef OFX_FUSE
#define OFX_FUSE 0   // clean/rr-f-a14-tail D4: 1 = phases 1 and 2 in one k-outer m-inner loop (the same operations and order;
#endif               // each weight converted by 0x1c = H on its own, one b128 + one b32 LDS read per tap)
#if OFX_FUSE
    float total = 0;
    half2 s0 = half2(0, 0), s1 = half2(0, 0), s2 = half2(0, 0);
    [unroll] for (int k = 0; k < 4; k++) {
        uint4 ra4[4]; uint pb[4];
        [unroll] for (int m = 0; m < 4; m++) { const int li = lbase + m * Wl + k; ra4[m] = of_ra[li]; pb[m] = of_rb[li]; }
        [unroll] for (int m = 0; m < 4; m++) {
            const half ax = of_hh(ra4[m].w), ay = of_hh(ra4[m].w >> 16), axy = of_hh(pb[m]);
            const half t1 = oxx[k] * ax;
            const half t2 = mad(oyy[m], ay, t1);
            const half oxy = ox[k] * oy[m];
            const half e = of_fh(mad(oxy, axy, t2));
            const float w32 = exp2(of_f(e));
            total = An(total, w32);
            const half w = of_hh(of_cvt(w32));
            const half2 ww = half2(w, w);
            s0 = mad(ww, wg_as_h2(ra4[m].x), s0); s1 = mad(ww, wg_as_h2(ra4[m].y), s1); s2 = mad(ww, wg_as_h2(ra4[m].z), s2);
        }
    }
    OF_CLK(2);
#else
    // phase 1: exponents and f32 weights of the 16 taps (k outer, m inner as in rr_output_main)
    v8f w8[2];
    float total = 0;
    w8[0] = wg_splat(0); w8[1] = wg_splat(0);
    [unroll] for (int k = 0; k < 4; k++) {
        uint pa[4], pb[4];
        [unroll] for (int m = 0; m < 4; m++) { const int li = lbase + m * Wl + k; pa[m] = of_ra[li].w; pb[m] = of_rb[li]; }
        [unroll] for (int m = 0; m < 4; m++) {
            const half ax = of_hh(pa[m]), ay = of_hh(pa[m] >> 16), axy = of_hh(pb[m]);
            const half t1 = oxx[k] * ax;
            const half t2 = mad(oyy[m], ay, t1);
            const half oxy = ox[k] * oy[m];
            const half e = of_fh(mad(oxy, axy, t2));
            const float w32 = exp2(of_f(e));
            total = An(total, w32);
            w8[k >> 1].v[(k & 1) * 4 + m] = w32;
        }
    }
    OF_CLK(2);
    const uint4 wh0 = of_to_f16(w8[0]), wh1 = of_to_f16(w8[1]);
    const uint wbits[8] = {wh0.x, wh0.y, wh0.z, wh0.w, wh1.x, wh1.y, wh1.z, wh1.w};
    // phase 2: the six binary16 sums (three packed pairs)
    half2 s0 = half2(0, 0), s1 = half2(0, 0), s2 = half2(0, 0);
    [unroll] for (int k = 0; k < 4; k++) [unroll] for (int m = 0; m < 4; m++) {
        const int t = k * 4 + m;
        const half w = of_hh((t & 1) != 0 ? wbits[t >> 1] >> 16 : wbits[t >> 1]);
        const uint4 r = of_ra[lbase + m * Wl + k];
        const half2 ww = half2(w, w);
        s0 = mad(ww, wg_as_h2(r.x), s0); s1 = mad(ww, wg_as_h2(r.y), s1); s2 = mad(ww, wg_as_h2(r.z), s2);
    }
#endif
    // A NaN payload of these sums (the only possible difference to HF) never reaches an output: sums 0-2 enter D (a
    // NaN gives D's own NaN constant), sum 3 and the total enter S / max (a NaN gives 0 / K13), sums 4-5 make alpha
    // NaN and the blend's NaN guard then replaces the value by M(current, K12).
    s0 = of_fh2(s0); s1 = of_fh2(s1); s2 = of_fh2(s2);
    const float accum[6] = {of_f(s0.x), of_f(s0.y), of_f(s1.x), of_f(s1.y), of_f(s2.x), of_f(s2.y)};
    // norm = accum / max(total, K13): accum is 0 or >= 2^-24, so only a reciprocal below 2^-102 can reach the band
    const float Rn = Rf(max(total, K13));
    rare = or(rare, of_small(Rn, asfloat(0x0d800000u)));
    float norm[6];
    [unroll] for (uint c = 0; c < 6; c++) norm[c] = Mn(accum[c], Rn);
#if OF_GZ
    // shipped site set: Mq(total, K1) unchecked (lost to the inout copy-out in the per-site form)
    const float cm1 = Mf(total, K1), cm2 = Mf(norm[3], cm1);
    const float coverage = min(max(cm2, 0.0f), 1.0f);
    const float ch4 = An(1, -norm[4]), ch5 = An(1, -norm[5]);
    const float ah = of_fqr(coverage, ch4, norm[4]), ao = of_fqr(coverage, ch5, norm[5]);
    const float dt0 = of_dt(norm[0]), dt1 = of_dt(norm[1]), dt2 = of_dt(norm[2]);
    const float dr0 = Mf(dt0, K8), dr1 = Mf(dt1, K8), dr2 = Mf(dt2, K8);
    [branch] if (min(abs(cm2), min(abs(ah), abs(ao))) == 0)
        rare = or(rare, or(of_cm(cm2, norm[3], cm1), or(of_cf(ah, coverage, ch4, norm[4]), of_cf(ao, coverage, ch5, norm[5]))));
    [branch] if (of_z3(dr0, dr1, dr2) == 0)
        rare = or(rare, or(or(of_cm(dr0, dt0, K8), of_cm(dr1, dt1, K8)), of_cm(dr2, dt2, K8)));
    const float3 current = float3(of_dtail(dr0), of_dtail(dr1), of_dtail(dr2));
#else
    const float coverage = min(max(Mq(norm[3], Mq(total, K1, rare), rare), 0.0f), 1.0f);
    const float ah = Fq(coverage, An(1, -norm[4]), norm[4], rare), ao = Fq(coverage, An(1, -norm[5]), norm[5], rare);
    const float3 current = float3(Df(norm[0], rare), Df(norm[1], rare), Df(norm[2], rare));
#endif
    const float s = Mf(u, Rwr), t = Mf(v, Rhr);
    // candidate weights: bilinear of the decoded binary16 W4 records
    const OfTaps qc = of_taps(of_axis_s(s, wr), of_axis_s(t, hr));
    const uint rw = of_rec_w();
    // (byte pitches in the 24-bit products: the compiler distributes a byte scale over the index sum)
    const uint qr0 = of_m24((uint)qc.y0, wr * 8u), qr1 = of_m24((uint)qc.y1, wr * 8u);
#if OFX_LD2
    const uint2 c0 = wg_mem.Load2(rw + qr0 + qc.x0 * 8), c1 = wg_mem.Load2(rw + qr0 + qc.x1 * 8),
                c2 = wg_mem.Load2(rw + qr1 + qc.x0 * 8), c3 = wg_mem.Load2(rw + qr1 + qc.x1 * 8);
    float4 weights;
    [unroll] for (uint c = 0; c < 4; c++) { const uint sh = (c & 1u) * 16u;
        weights[c] = of_bilf(qc, of_hh((c < 2 ? c0.x : c0.y) >> sh), of_hh((c < 2 ? c1.x : c1.y) >> sh), of_hh((c < 2 ? c2.x : c2.y) >> sh), of_hh((c < 2 ? c3.x : c3.y) >> sh)); }
#else
    const half4 c0 = wg_mem.Load<half4>(rw + qr0 + qc.x0 * 8), c1 = wg_mem.Load<half4>(rw + qr0 + qc.x1 * 8),
                c2 = wg_mem.Load<half4>(rw + qr1 + qc.x0 * 8), c3 = wg_mem.Load<half4>(rw + qr1 + qc.x1 * 8);
    float4 weights;
    [unroll] for (uint c = 0; c < 4; c++) weights[c] = of_bilf(qc, c0[c], c1[c], c2[c], c3[c]);
#endif
    const int fu = (int)floor(u), fv = (int)floor(v);
    const bool inr = and(and(fu >= 0, fv >= 0), and(fu < (int)wr, fv < (int)hr));
    const uint mi = inr ? (uint)fv * wr + (uint)fu : 0u;
    const uint mi8 = inr ? of_m24((uint)fv, wr * 8u) + (uint)fu * 8u : 0u, mi4 = mi8 / 2u;
    float4 pair;
    float2 ero, non;
    if (of_half()) {
#if OFX_LD3   // D5 rest (same values): the motion pair / eroded texels as one b64 / b32 load, widened exactly
        const uint2 pm = wg_mem.Load2(motion_pair + mi8);
        const uint pe = wg_mem.Load(motion_eroded + mi4);
        pair = inr ? float4(f16tof32(pm.x), f16tof32(pm.x >> 16), f16tof32(pm.y), f16tof32(pm.y >> 16)) : float4(0, 0, 0, 0);
        ero = inr ? float2(f16tof32(pe), f16tof32(pe >> 16)) : float2(0, 0);
#else
        const half4 pm = wg_mem.Load<half4>(motion_pair + mi8);
        const half2 pe = wg_mem.Load<half2>(motion_eroded + mi4);
        pair = inr ? float4(pm) : float4(0, 0, 0, 0);
        ero = inr ? float2(pe) : float2(0, 0);
#endif
        const uint qh0 = qr0 / 2u, qh1 = qr1 / 2u;
#if OFX_LD2
        const uint n0 = wg_mem.Load(motion_non + qh0 + qc.x0 * 4), n1 = wg_mem.Load(motion_non + qh0 + qc.x1 * 4),
                   n2 = wg_mem.Load(motion_non + qh1 + qc.x0 * 4), n3 = wg_mem.Load(motion_non + qh1 + qc.x1 * 4);
        non = float2(of_bilf(qc, of_hh(n0), of_hh(n1), of_hh(n2), of_hh(n3)), of_bilf(qc, of_hh(n0 >> 16), of_hh(n1 >> 16), of_hh(n2 >> 16), of_hh(n3 >> 16)));
#else
        const half2 n0 = wg_mem.Load<half2>(motion_non + qh0 + qc.x0 * 4), n1 = wg_mem.Load<half2>(motion_non + qh0 + qc.x1 * 4),
                    n2 = wg_mem.Load<half2>(motion_non + qh1 + qc.x0 * 4), n3 = wg_mem.Load<half2>(motion_non + qh1 + qc.x1 * 4);
        non = float2(of_bilf(qc, n0.x, n1.x, n2.x, n3.x), of_bilf(qc, n0.y, n1.y, n2.y, n3.y));
#endif
    } else {
        pair = inr ? wg_ld4f(motion_pair, mi * 16) : float4(0, 0, 0, 0);
        ero = inr ? asfloat(wg_ld64(motion_eroded, mi * 8)) : float2(0, 0);
        const float2 n0 = asfloat(wg_ld64(motion_non, qr0 + qc.x0 * 8)), n1 = asfloat(wg_ld64(motion_non, qr0 + qc.x1 * 8)),
                     n2 = asfloat(wg_ld64(motion_non, qr1 + qc.x0 * 8)), n3 = asfloat(wg_ld64(motion_non, qr1 + qc.x1 * 8));
        non = float2(of_bil(qc, n0.x, n1.x, n2.x, n3.x), of_bil(qc, n0.y, n1.y, n2.y, n3.y));
    }
    // order() of rr_output.hlsl on (weight, index, motion) triples: cmp keeps (a, b) iff w[a] >= w[b] or either is NaN,
    // i.e. iff !(w[a] < w[b]); the five compare-exchanges carry the values instead of indexing them
    float w0 = weights.x, w1 = weights.y, w2 = weights.z, w3 = weights.w;
    uint i0 = 0, i1 = 1, i2 = 2, i3 = 3;
    float2 m0 = pair.xy, m1 = non, m2 = ero, m3 = pair.zw;
    of_cs(w0, i0, m0, w1, i1, m1);   // ab
    of_cs(w2, i2, m2, w3, i3, m3);   // cd
    of_cs(w0, i0, m0, w2, i2, m2);   // hi = cmp(ab.x, cd.x): (w0, w2)
    of_cs(w1, i1, m1, w3, i3, m3);   // lo = cmp(ab.y, cd.y): (w1, w3)
    of_cs(w1, i1, m1, w2, i2, m2);   // mid = cmp(lo.x, hi.y): (w1, w2); order = (hi.x, mid.x, mid.y, lo.y) = 0, 1, 2, 3
#if BAD_ORDER
    { const float tw = w1; w1 = w2; w2 = tw; const uint ti = i1; i1 = i2; i2 = ti; const float2 tm = m1; m1 = m2; m2 = tm; }
#endif
    const uint4 ord = uint4(i0, i1, i2, i3);
    OF_CLK(3);
    float4 nw = float4(w0, w1, w2, w3);
    [unroll] for (uint i = 0; i < 4; i++) nw[i] = min(max(nw[i], 0.0f), 1.0f);
    const float sum = An(An(An(nw.y, nw.x), nw.z), nw.w);
    // candidate weights are 0 or >= 2^-58, the reciprocal >= 2^-30 (K13 <= 2^30, of_insane)
    const float Rs = Rf(max(sum, K13));
    [unroll] for (uint i = 0; i < 4; i++) nw[i] = Mf(nw[i], Rs);
    uint fallmask = 0, usedmask = 0;
    float3 acc = 0;
    float tot = 0;
    OF_CLK(4);
    float3 tp0 = 0, tp1 = 0;
    uint fl0 = 0, fl1 = 0;
    [loop] for (uint i = 0; i < 3; i++) {
        if (i > 0 && !(tot < K3)) break;
#if OF_REPROJ_MAX < 3
        if (i >= OF_REPROJ_MAX) break;  // NEAR-EXACT opt-in (tier B): at most OF_REPROJ_MAX candidates reprojected
#endif
        uint fell;
        const float2 mi = i == 0 ? m0 : (i == 1 ? m1 : m2);
#if OF_DEDUP
        // of_reproj is a function of (x, y, mv, fallback) only: a candidate whose motion equals an earlier one's (==:
        // +0 / -0 give the same An(x + .5, mv) as x + .5 >= .5; NaN never matches) reuses that result and its fell
        // bit (rare is already set by the first call when it applies). Waves where every lane matches skip the taps.
        const bool d0 = and(i > 0, and(mi.x == m0.x, mi.y == m0.y)), d1 = and(i > 1, and(mi.x == m1.x, mi.y == m1.y));
        float3 tap;
        [branch] if (or(d0, d1)) {
            tap = d0 ? tp0 : tp1; fell = d0 ? fl0 : fl1;
        } else {
            tap = of_reproj(x, y, mi, current, Ep, invE, fell, Rwh, Rhh, fine, rare);
        }
        if (i == 0) { tp0 = tap; fl0 = fell; } else if (i == 1) { tp1 = tap; fl1 = fell; }
#else
        const float3 tap = of_reproj(x, y, mi, current, Ep, invE, fell, Rwh, Rhh, fine, rare);
#endif
        fallmask |= fell << i; usedmask |= 1u << i;
#if OF_GZ
        if (i == 0) {
            float3 r3;
            [unroll] for (uint c = 0; c < 3; c++) r3[c] = Mf(nw.x, tap[c]);
            [branch] if (of_z3(r3.x, r3.y, r3.z) == 0)
                rare = or(rare, or(or(of_cm(r3.x, nw.x, tap.x), of_cm(r3.y, nw.x, tap.y)), of_cm(r3.z, nw.x, tap.z)));
            acc = r3;
            tot = nw.x;
        } else {
            float3 r3;
            [unroll] for (uint c = 0; c < 3; c++) r3[c] = of_fqr(nw[i], tap[c], acc[c]);
            [branch] if (of_z3(r3.x, r3.y, r3.z) == 0)
                rare = or(rare, or(or(of_cf(r3.x, nw[i], tap.x, acc.x), of_cf(r3.y, nw[i], tap.y, acc.y)), of_cf(r3.z, nw[i], tap.z, acc.z)));
            acc = r3;
            tot = An(tot, nw[i]);
        }
#else
        if (i == 0) {
            [unroll] for (uint c = 0; c < 3; c++) acc[c] = Mq(nw.x, tap[c], rare);
            tot = nw.x;
        } else {
            [unroll] for (uint c = 0; c < 3; c++) acc[c] = Fq(nw[i], tap[c], acc[c], rare);
            tot = An(tot, nw[i]);
        }
#endif
    }
    OF_CLK(5);
    const float Rt = Rf(tot);
    float3 hist;
#if OF_GZ
    {   // shipped site set: Mq(acc, Rt) unchecked (lost to the inout copy-out in the per-site form), Df's own product checked
        float3 q3, t3, d3;
        [unroll] for (uint c = 0; c < 3; c++) { q3[c] = Mf(acc[c], Rt); t3[c] = of_dt(q3[c]); d3[c] = Mf(t3[c], K8); }
        [branch] if (of_z3(d3.x, d3.y, d3.z) == 0)
            rare = or(rare, or(or(of_cm(d3.x, t3.x, K8), of_cm(d3.y, t3.y, K8)), of_cm(d3.z, t3.z, K8)));
        [unroll] for (uint c = 0; c < 3; c++) hist[c] = Mf(E, Mf(Mf(of_dtail(d3[c]), K12), REp));   // >= 2^-114
    }
#else
    [unroll] for (uint c = 0; c < 3; c++) hist[c] = Mf(E, Mf(Mf(Df(Mq(acc[c], Rt, rare), rare), K12), REp));   // >= 2^-114
#endif
    uint gh, go;
    const float3 bh = of_blend(current, hist, ah, gh, rare), bo = of_blend(current, hist, ao, go, rare);
    v8f cq;
    [unroll] for (uint c = 0; c < 3; c++) { cq.v[c] = Cf(bh[c]); cq.v[3 + c] = Cf(bo[c]); }
    cq.v[6] = 0; cq.v[7] = 0;
    const uint4 hq = of_to_f16(cq);
    // packed_head bit 3: no diagnostic outputs (linear f32 and trace; the fallback still writes them for its quads)
    const bool diag = !OFX_SPEC && (packed_head & 8u) == 0u;
    if (valid) {
        if (diag) { wg_st32(linear_buf, idx * 12, asuint(bh.x)); wg_st32(linear_buf, idx * 12 + 4, asuint(bh.y)); wg_st32(linear_buf, idx * 12 + 8, asuint(bh.z)); }
        wg_st64(ch, idx * 8, uint2(hq.x, hq.y & 0xffffu));
#if OFX_QST
        if ((wh & 1u) != 0u) {
            wg_st16(q, idx * 6, hq.y >> 16); wg_st16(q, idx * 6 + 2, hq.z); wg_st16(q, idx * 6 + 4, hq.z >> 16);
        }
#else
        wg_st16(q, idx * 6, hq.y >> 16); wg_st16(q, idx * 6 + 2, hq.z); wg_st16(q, idx * 6 + 4, hq.z >> 16);
#endif
        if (diag) wg_st128(trace, idx * 16, uint4(ord.x | (ord.y << 2) | (ord.z << 4) | (ord.w << 6), usedmask, fallmask, gh | (go << 1)));
#if OF_DEBUG
        wg_st128(trace, idx * 16, uint4(asuint(OF_DEBUG==5?current.y:OF_DEBUG==6?current.z:OF_DEBUG==1?current.x:OF_DEBUG==2?norm[0]:OF_DEBUG==3?accum[0]:hist.x),asuint(OF_DEBUG==5?hist.y:OF_DEBUG==6?hist.z:OF_DEBUG==1?hist.x:OF_DEBUG==2?total:OF_DEBUG==3?accum[3]:acc.x),asuint(OF_DEBUG==5?acc.y:OF_DEBUG==6?acc.z:OF_DEBUG==1?ah:OF_DEBUG==2?coverage:OF_DEBUG==3?accum[5]:nw.x),asuint(OF_DEBUG==5?fallback.y:OF_DEBUG==6?bo.y:OF_DEBUG==1?bh.x:OF_DEBUG==2?norm[3]:OF_DEBUG==3?accum[1]:fallback.x)));
#endif
    }
#if OFX_QST
    // clean/rr-f-a14-tail D17 (same bytes): an even history width makes idx even exactly on the even lanes (x = tx0 + lx,
    // tx0 a multiple of 32), whose odd neighbour (lane + 1, the same row) is valid with it: the even lane writes both
    // 6-byte Q texels as one 12-byte store at the 4-byte aligned idx * 6
    {
        const uint qy_ = QuadReadAcrossX(hq.y), qz_ = QuadReadAcrossX(hq.z);
        if ((wh & 1u) == 0u && valid && (lx & 1u) == 0u)
            wg_mem.Store3(q + idx * 6, uint3((hq.y >> 16) | (hq.z << 16), (hq.z >> 16) | (qy_ & 0xffff0000u), qz_));
    }
#endif
    OF_CLK(6);
    // ---- PH1: the tile's 16 x 4 quads (pixels outside the image count as 0, as rr_output_pool's fetch)
#if OFX_PH1W && WG_WAVE == 64
    // clean/rr-f-a14-tail D7 (same bits): a quad's 4 pixels are lanes l, l + 1, l + 32, l + 33 of one wave64 (rows ly,
    // ly + 1): each wave pools its own 16 quads from lane reads (the same An(An(tl, bl), An(tr, br)) order), no
    // groupshared copy, no barrier
    {
        const float3 vt = valid ? bh : float3(0, 0, 0);
        const uint wl = lane & 63u, wx = wl ^ 32u;
        const uint rt = and(rare, valid) ? 1u : 0u;
        const float3 vb = float3(WaveReadLaneAt(vt.x, wx), WaveReadLaneAt(vt.y, wx), WaveReadLaneAt(vt.z, wx));
        const uint rb = WaveReadLaneAt(rt, wx);
        const float3 vtr = float3(QuadReadAcrossX(vt.x), QuadReadAcrossX(vt.y), QuadReadAcrossX(vt.z));
        const float3 vbr = float3(QuadReadAcrossX(vb.x), QuadReadAcrossX(vb.y), QuadReadAcrossX(vb.z));
        const uint rq = rt | rb | QuadReadAcrossX(rt) | QuadReadAcrossX(rb);
        if (wl < 32u && (lx & 1u) == 0u) {
            const uint gx = tx0 + lx, gy = ty0 + ly;
            bool qrare = rq != 0u;
            const uint count = min(2u, wh - min(gx, wh)) * min(2u, hh - min(gy, hh));
            const float Rc = Rf(count);
            uint code[3];
            [unroll] for (uint c = 0; c < 3; c++) code[c] = of_code10(Mf(Cf(Mq(An(An(vt[c], vb[c]), An(vtr[c], vbr[c])), Rc, qrare)), K11));
            if (gx < wh && gy < hh) {
                const uint pidx = of_m24(gy / 2u, of_pw()) + gx / 2u;
                wg_st64(ph, pidx * 8u, uint2(code[0] | (code[1] << 16), code[2] | (3u << 16)));
                if (qrare && !OF_NO_FALLBACK) { const uint slot = wg_atomic_add(scratch, 0u, 1u); wg_st32(scratch, 256u + slot * 4u, pidx); }
            }
        }
    }
#else
    of_lin[lane] = valid ? bh : float3(0, 0, 0);
    const uint4 rowballot = WaveActiveBallot(and(rare, valid));
#if WG_WAVE == 64
    if (WaveIsFirstLane()) { of_rowrare[ly] = rowballot.x; of_rowrare[ly + 1u] = rowballot.y; }   // rows ly, ly + 1
#else
    if (WaveIsFirstLane()) of_rowrare[ly] = rowballot.x;
#endif
    GroupMemoryBarrierWithGroupSync();
    if (lane < 64u) {
        const uint qx = lane & 15u, qy = lane >> 4, gx = tx0 + 2u * qx, gy = ty0 + 2u * qy;
        const uint l0 = (2u * qy) * OF_TX + 2u * qx;
        bool qrare = (((of_rowrare[2u * qy] | of_rowrare[2u * qy + 1u]) >> (2u * qx)) & 3u) != 0u;
#if OFX_M24 & 8
        const uint count = of_m24(min(2u, wh - min(gx, wh)), min(2u, hh - min(gy, hh)));   // both <= 2
#else
        const uint count = min(2u, wh - min(gx, wh)) * min(2u, hh - min(gy, hh));
#endif
        const float Rc = Rf(count);
        const float3 tl = of_lin[l0], tr = of_lin[l0 + 1u], bl = of_lin[l0 + OF_TX], br = of_lin[l0 + OF_TX + 1u];
        uint code[3];
#if OF_GZ
        float3 s3, r3;
        [unroll] for (uint c = 0; c < 3; c++) { s3[c] = An(An(tl[c], bl[c]), An(tr[c], br[c])); r3[c] = Mf(s3[c], Rc); }
        [branch] if (of_z3(r3.x, r3.y, r3.z) == 0)
            qrare = or(qrare, or(or(of_cm(r3.x, s3.x, Rc), of_cm(r3.y, s3.y, Rc)), of_cm(r3.z, s3.z, Rc)));
        [unroll] for (uint c = 0; c < 3; c++) code[c] = of_code10(Mf(Cf(r3[c]), K11));
#else
        [unroll] for (uint c = 0; c < 3; c++) code[c] = of_code10(Mf(Cf(Mq(An(An(tl[c], bl[c]), An(tr[c], br[c])), Rc, qrare)), K11));
#endif
        if (gx < wh && gy < hh) {
            const uint pidx = of_m24(gy / 2u, of_pw()) + gx / 2u;
            wg_st64(ph, pidx * 8u, uint2(code[0] | (code[1] << 16), code[2] | (3u << 16)));
            if (qrare && !OF_NO_FALLBACK) { const uint slot = wg_atomic_add(scratch, 0u, 1u); wg_st32(scratch, 256u + slot * 4u, pidx); }
        }
    }
#endif
    OF_CLK(7);
}

// ---- fallback: rr_output_main's and rr_output_pool's code for the listed quads (a fixed grid strides over the list) --
WG_KERNEL(rr_output_fallback)
{
    WG_SETUP();
    if (wg_block_id() == 0u && wg_lane_id() == 0u) {
        wg_st64(scratch, 8u, wg_ld64(scratch, 0u));   // diagnostics: this frame's count and prep flags
        wg_st32(scratch, 4u, 0u);                     // prep flags of the next frame
    }
    const uint n = wg_ld32(scratch, 0u), pw = of_pw();
    for (uint i = wg_block_id() * WG_BLOCK + wg_lane_id(); i < n; i += wg_grid_size() * WG_BLOCK) {
        const uint qi = wg_ld32(scratch, 256u + i * 4u), x0 = (qi % pw) * 2u, y0 = (qi / pw) * 2u;
        for (uint d = 0; d < 4u; d++) {
            const uint x = x0 + (d & 1u), y = y0 + (d >> 1);
            if (x < wh && y < hh) out_main_pixel(y * wh + x);
        }
        out_pool_quad(qi);
    }
}
