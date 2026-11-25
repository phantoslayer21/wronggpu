// SPDX-License-Identifier: MIT
// The per-pixel reconstruction of Model L (l_post.hlsl: the staged-texel loads, the FP16 4 x 4 gather, the history candidates,
// steps 3 to 5 of one D pixel) shared by l_post.hlsl (the kernel that writes the display image) and l_post_tf.hlsl (the
// true-fusion kernel). The including kernel declares WG_ARGS (the names color, A, B, C, hist_prev, m_near, m_center, m_far,
// w_p, hist_out, reduced, P_*, expo are used here) and the LP_* switches first; every function below is moved from
// l_post.hlsl unchanged (the DXIL of its kernels is unchanged by the move).
#pragma once
#ifndef LP_HSEQ
#define LP_HSEQ 0   // 1: history_cr issues its loads in two dependent groups (fewer live registers)
#endif
#ifndef LP_FBLAZY
#define LP_FBLAZY 0
#endif
#ifndef LP_PIDIV
#define LP_PIDIV 0   // 1: the staged route's pre-exposure division folded into every colour read (l_post_tf_p12_pf)
#endif
#if LP_PIDIV
#include "wg_pidiv.hlsli"
#endif
#if LP_W64
// The driver's matrix conversion fragment (wg_ags.hlsli) is a wave32 form (RGA: the same intrinsic compiled in a wave64
// kernel gives another conversion sequence: 112 instead of 182 v_cvt_f16_f32, no packs; not validated, not used), and the
// software RNE conversion (wg_to_f16: wg_f2h) is ~12 000 instructions per kernel in the 64-lane form. In 64-lane waves every
// "driver conversion" is the single-value conversion of the driver's intrinsic channel (op 0x1C, immediate 1: one hardware
// v_cvt_f16_f32, round to nearest even; the same call as rr_d0r.hlsl / rr_output_fast.hlsli), the same bits as wg_f2h (NaN
// payloads included). Wave-uniform control flow only, as every call of the channel.
#include "wg_ags.hlsli"
uint lp_f2h_hw(float x) { return wg_ags_call(0x1cu, 0u, 1u, asuint(x), 0u) & 0xffffu; }
uint4 wg_to_f16_ags(v8f d)
{
    return uint4(lp_f2h_hw(d.v[0]) | (lp_f2h_hw(d.v[1]) << 16), lp_f2h_hw(d.v[2]) | (lp_f2h_hw(d.v[3]) << 16),
                 lp_f2h_hw(d.v[4]) | (lp_f2h_hw(d.v[5]) << 16), lp_f2h_hw(d.v[6]) | (lp_f2h_hw(d.v[7]) << 16));
}
#endif
#ifndef kStageCap
#define kStageCap 512   // staged render texels (l_post_tf.hlsl: its own cap)
#endif
#define kLWin (LP_MORIG ? 3 : 4)
#define kTexWave WG_WAVE   // the wave's first lane of the staging rounds (LP_WTEX)
#define kTexDw 6   // staged texel: f0 f1 | f2 f3 | f4 f5 | f6 f7 | f8 kA | kB kC (FP16 bits)

static const float kEncK = asfloat(0x3dddf184u);        // 0.36f * 0.30102999566398120f
static const float kDecK = asfloat(0x4113a428u);        // 3.32192809488736235f / 0.36f
static const float kDecLn = asfloat(0x40ccac9au);       // 2.30258509299404568f / 0.36f
static const float kDecPolyMax = asfloat(0x3d60233du);  // 0.35f / kDecLn
static const float kInv255 = asfloat(0x3b808081u);      // 1.0f / 255.0f
static const float kInv245 = asfloat(0x3ed0fac6u);      // 1.0f / 2.45f
static const float kP5040 = asfloat(0x39500d01u), kP720 = asfloat(0x3ab60b61u), kP120 = asfloat(0x3c088889u);
static const float kP24 = asfloat(0x3d2aaaabu), kP6 = asfloat(0x3e2aaaabu);
static const float k1em4 = asfloat(0x38d1b717u);        // 1e-4f
static const float k1em10 = asfloat(0x2edbe6ffu);       // 1e-10f
static const float k1p1 = asfloat(0x3f8ccccdu);         // 1.1f
static const float k0p05 = asfloat(0x3d4ccccdu);        // 0.05f
static const float kDisplayMax = asfloat(0x401ccccdu);  // 2.45f
static const float k0p01 = asfloat(0x3c23d70au);        // 0.01f
static const float kShapeAB = asfloat(0xbf38a8c1u);     // -0.721325f
static const float kShapeC = asfloat(0xbfb8a8c1u);      // -1.44265f
static const float kMinNormal = asfloat(0x00800000u);   // 2^-126

groupshared uint sh_tex[kStageCap * kTexDw];

static uint g_z;          // 0, opaque to the compiler (from the grid size)
static float g_e, g_ep;   // exposures (0 replaced by 1)
static float g_rp;       // 1 / e_prev (IEEE division, round to nearest even)
#if LP_PIDIV
static PiDiv g_pd;       // the pre-exposure division (pidiv_make of the launch)
#endif
float opq(float x) { return asfloat(asuint(x) ^ g_z); }
half opqh(half x) { return asfloat16((uint16_t)(asuint16(x) ^ (uint16_t)g_z)); }
float mul_nz(float a, float b) { return mad(a, b, asfloat(g_z | 0x80000000u)); }
#ifndef LP_MUT
#define LP_MUT 0   // 1, 2, 4 ... 9: negative controls of the byte checks (never shipped)
#endif
#ifndef LP_DP
#define LP_DP 1
#endif
#ifndef LP_FMANZ
#define LP_FMANZ 1
#endif
#ifndef LP_HSCOUNT
#define LP_HSCOUNT 0   // 1: test builds only (clean/l/runner/l_post_edge_check.py): the history fast path's wave counter
#endif
// h(a b) as its own rounding that the driver cannot fuse into a following FP16 add: fma(a, b, -0) with an opaque -0
// (RN16(a b + -0) = RN16(a b) for every non-NaN pair, zeros included); LP_FMANZ 0: the product through the xor guard
half mulh(half a, half b)
{
#if LP_FMANZ
    return mad(a, b, asfloat16((uint16_t)(0x8000u | g_z)));
#else
    return opqh(a * b);
#endif
}
half h_bits(uint b) { return asfloat16((uint16_t)b); }
uint bits_h(half h) { return (uint)asuint16(h); }

int clampi(int v, int lo, int hi) { int t = v > hi ? hi : v; return t < lo ? lo : t; }
int mini(int a, int b) { return a < b ? a : b; }
int floor_i(float v) { return (int)floor(v); }
float clampf(float v, float lo, float hi) { return min(max(v, lo), hi); }

// E(v) = kEncK log2(max(fma(100, v, 1), 1e-4)); E_in(v) = E(min(v, 32752)), E_in(NaN) = 0
float enc_log(float v) { return log2(max(mad(100.0f, v, 1.0f), k1em4)); }
float enc_e(float v) { return kEncK * enc_log(v); }
float enc_in(float v) { return isnan(v) ? 0.0f : enc_e(min(v, 32752.0f)); }

float dec_m1_poly(float qc)
{
    float y = qc * kDecLn;
    float t = mad(y, kP5040, kP720);
    t = mad(t, y, kP120);
    t = mad(t, y, kP24);
    t = mad(t, y, kP6);
    t = mad(t, y, 0.5f);
    t = mad(t, y, 1.0f);
    return t * y;
}
// D(q) = dec_m1(clamp(q, 1e-4, 3)) * 0.01, a NaN passed through (l_dec)
float l_dec(float q)
{
    const float qc = clampf(q, k1em4, 3.0f);
#if LP_DP
    // Model M's wave-level skip (post.hlsl post_core_s): the exponential for every lane, the polynomial only when some
    // active lane of the wave is below kDecPolyMax; per lane the same select as below, so the same value
    float v = exp2(qc * kDecK) - 1.0f;
    const bool sm = qc < kDecPolyMax;
#if LP_MUT == 6
    [branch] if (WaveActiveAllTrue(sm)) v = sm ? dec_m1_poly(qc) : v;   // mutant 6: all lanes instead of any lane
#else
    [branch] if (WaveActiveAnyTrue(sm)) v = sm ? dec_m1_poly(qc) : v;
#endif
#else
    const float v = qc < kDecPolyMax ? dec_m1_poly(qc) : exp2(qc * kDecK) - 1.0f;
#endif
    return isnan(q) ? q : mul_nz(v, k0p01);
}

// (_Float16)((float)k / 255.0f) for the code k = 0 ... 255 (an IEEE FP32 division, then round to nearest even: HIP's
// v_div_* sequence and v_cvt_f16_f32): built per workgroup in LDS (code16_build), read by code16
groupshared uint sh_code[256];
uint code16(uint v, uint i) { return sh_code[(v >> (8u * i)) & 0xffu]; }

// lane k of the workgroup (256 lanes) writes code k's FP16 bits: wg_div, then the driver's conversion (wave-uniform)
void code16_build(int lane)
{
#if LP_W64
    // the 64-lane form: one hardware conversion (the other seven slots of wg_to_f16_ags(cv) are +0 and unused)
#if LP_CODE
    sh_code[lane] = lp_f2h_hw(opq((float)lane * kInv255));
#else
    sh_code[lane] = lp_f2h_hw(opq(wg_div((float)lane, 255.0f)));
#endif
#else
    v8f cv;
#if LP_CODE
    cv.v[0] = opq((float)lane * kInv255);   // f16(f32(k) f32(1 / 255)) = f16(RN32(k / 255)), all 256 codes (header)
#else
    cv.v[0] = opq(wg_div((float)lane, 255.0f));
#endif
    [unroll] for (int i = 1; i < 8; ++i) cv.v[i] = 0.0f;
    sh_code[lane] = wg_to_f16_ags(cv).x & 0xffffu;
#endif
}

// a * b as the round-to-odd FP32 value whose FP16 rounding is (f16)(a b) in one rounding (wg_mul_f16 before its
// conversion), and wg_mul_f16's +0 for an exact zero product (a NaN product keeps its conversion)
float mul_odd(float a, float b)
{
    const float p = a * b;
    return asfloat(wg_round_odd(p, mad(a, b, -p)));
}
bool mul_zero(float a, float b)
{
    return ((asuint(a) & 0x7fffffffu) == 0u || (asuint(b) & 0x7fffffffu) == 0u) && !isnan(a * b);
}
// mul_zero(a, c) for a nonzero finite constant c: a is +-0 (the product is then +-0, never NaN); a NaN or infinity a is not
bool mul_zero_k(float a)
{
    return (asuint(a) & 0x7fffffffu) == 0u;
}

// render pixel idx's raw inputs: colour, codes a (A0 A1 A2), b (B0 B1 B2), kernel shape k (C0 | C1 << 16, C2)
void texel_inputs(int idx, out float4 c, out uint a, out uint b, out uint2 k)
{
    const uint ui = (uint)idx;
    c = wg_ld4f(color, ui * 16u);
#if LP_P12
    const uint2 d = wg_ld64(A, ui * 8u);
    const uint c01 = wg_ld32(C, ui * 4u);
    a = d.x & 0xffffffu;
    b = (d.x >> 24) | ((d.y & 0xffffu) << 8);
    k = uint2(c01, d.y >> 16);   // C0 | C1 << 16, C2 (l_coef's C dword 0 and the low half of dword 1)
#else
    b = wg_ld32(B, ui * 4u);
    a = wg_ld32(A, ui * 4u);
    k = wg_ld64(C, ui * 8u);
#endif
}

void pack_texel(uint f0, uint f1, uint f2, uint a, uint b, uint kA, uint kB, uint kC, out uint t[kTexDw])
{
    t[0] = f0 | (f1 << 16);
    t[1] = f2 | (code16(b, 0u) << 16);
    t[2] = code16(b, 1u) | (code16(b, 2u) << 16);
    t[3] = code16(a, 0u) | (code16(a, 1u) << 16);
    t[4] = code16(a, 2u) | (kA << 16);
    t[5] = kB | (kC << 16);
}

// l_load_texel: the nine FP16 features and the FP16 kernel shape of render pixel idx, as six dwords of FP16 bits. The six
// roundings (E_in of the colour: wg_f2h; the kernel shape: wg_mul_f16) by the driver's conversion (v_cvt_f16_f32, the same
// bits): wave-uniform control flow only. The values going in pass opq or are round-to-odd values (the driver must not
// fold an FP32 operation into the conversion).
// the conversions of load_texel_hw on the raw inputs (colour c, codes a (A0 A1 A2), b (B0 B1 B2), kernel shape k); LP_PIDIV: the
// colour is divided by the pre-exposure first, as color_divide stored it
void texel_make(float4 c, uint a, uint b, uint2 k, out uint t[kTexDw])
{
#if LP_PIDIV
    c.xyz = asfloat(asuint(pidiv3(c.xyz, g_pd, g_z)) ^ g_z);
#endif
    const float k0 = f16tof32(k.x), k1 = f16tof32(k.x >> 16), k2 = f16tof32(k.y);
    v8f cv;
#if LP_LDR
    cv.v[0] = opq(c.x);
    cv.v[1] = opq(c.y);
    cv.v[2] = opq(c.z);
#else
    cv.v[0] = opq(enc_in(opq(g_e * c.x)));
    cv.v[1] = opq(enc_in(opq(g_e * c.y)));
    cv.v[2] = opq(enc_in(opq(g_e * c.z)));
#endif
    cv.v[3] = mul_odd(k0, kShapeAB);
    cv.v[4] = mul_odd(k1, kShapeAB);
    cv.v[5] = mul_odd(k2, kShapeC);
    cv.v[6] = 0.0f;
    cv.v[7] = 0.0f;
    const uint4 h = wg_to_f16_ags(cv);
    const uint kA = mul_zero_k(k0) ? 0u : (h.y >> 16), kB = mul_zero_k(k1) ? 0u : (h.z & 0xffffu);
    const uint kC = mul_zero_k(k2) ? 0u : (h.z >> 16);
    pack_texel(h.x & 0xffffu, h.x >> 16, h.y & 0xffffu, a, b, kA, kB, kC, t);
}

// (the original body, kept inline: the DXIL of the kernels that stage per workgroup is unchanged)
void load_texel_hw(int idx, out uint t[kTexDw])
{
    float4 c;
    uint a, b;
    uint2 k;
    texel_inputs(idx, c, a, b, k);
    const float k0 = f16tof32(k.x), k1 = f16tof32(k.x >> 16), k2 = f16tof32(k.y);
    v8f cv;
#if LP_LDR
    cv.v[0] = opq(c.x);
    cv.v[1] = opq(c.y);
    cv.v[2] = opq(c.z);
#else
    cv.v[0] = opq(enc_in(opq(g_e * c.x)));
    cv.v[1] = opq(enc_in(opq(g_e * c.y)));
    cv.v[2] = opq(enc_in(opq(g_e * c.z)));
#endif
    cv.v[3] = mul_odd(k0, kShapeAB);
    cv.v[4] = mul_odd(k1, kShapeAB);
    cv.v[5] = mul_odd(k2, kShapeC);
    cv.v[6] = 0.0f;
    cv.v[7] = 0.0f;
    const uint4 h = wg_to_f16_ags(cv);
    const uint kA = mul_zero_k(k0) ? 0u : (h.y >> 16), kB = mul_zero_k(k1) ? 0u : (h.z & 0xffffu);
    const uint kC = mul_zero_k(k2) ? 0u : (h.z >> 16);
    pack_texel(h.x & 0xffffu, h.x >> 16, h.y & 0xffffu, a, b, kA, kB, kC, t);
}

struct LGather { float W; half F[9]; };

LGather gather_zero()
{
    LGather g;
    g.W = 0.0f;
    const half z = h_bits(g_z);   // +0, opaque: fma(w, f, +0) keeps its addend
    [unroll] for (int k = 0; k < 9; ++k) g.F[k] = z;
    return g;
}

// One tap: eps = fma(h(u v), C, h(h(h(u u) A) + h(h(v v) B))), w = 2^eps (+0 below 2^-126), W += w,
// F_k = fma(h(w), f_k, F_k). uu, vv, uv come in already rounded (and opaque). tap_weight gives w; gather_acc adds the
// tap with h(w) = hw (the FP16 rounding of w: wg_f2h, or the driver's conversion of eight taps' w at once).
float tap_weight(uint t[kTexDw], half uu, half vv, half uv)
{
    const half kA = h_bits(t[4] >> 16), kB = h_bits(t[5]), kC = h_bits(t[5] >> 16);
#if LP_C12
    const half ta = mulh(uu, kA);                 // (#231's product: opqh(uu kA), or mad(uu, kA, -0) with LP_FMANZ)
    const half s = mad(vv, kB, ta);               // C1: the row term fused into the sum
    const half eps = mad(uv, kC, s);
#else
#if LP_MUT == 1
    const half ta = mulh(uu, kB), tb = mulh(vv, kB);   // mutant 1: wrong factor
#else
    const half ta = mulh(uu, kA), tb = mulh(vv, kB);
#endif
    const half s = tb + ta;
    const half eps = mad(uv, kC, s);
#endif
    const float w = exp2((float)opqh(eps));
    return w >= kMinNormal ? w : 0.0f;
}
void gather_acc(inout LGather g, uint t[kTexDw], float w, uint hwb)
{
    g.W = g.W + w;
    const half hw = h_bits(hwb);
    [unroll] for (int k = 0; k < 9; ++k) {
        const uint d = t[k >> 1];
        g.F[k] = mad(hw, h_bits((k & 1) ? (d >> 16) : d), g.F[k]);
    }
}

// ---- history: 12-tap Catmull-Rom of the previous full history (history_cr<false>, 8-byte RGBA16F texels) ------------
float4 cr_weights(float f)
{
    float4 w;
#if LP_C12
    const float f2 = opq(f * f), f3 = opq(f * f2);
    w.x = mad(-(f + f3), 0.5f, f2);
    w.y = mad(f3, 1.5f, -opq(2.5f * f2)) + 1.0f;
    w.w = opq(opq(f3 - f2) * 0.5f);
    w.z = ((1.0f - w.x) - w.y) - w.w;
    return w;
#endif
    w.x = f * (-0.5f + f * (1.0f - 0.5f * f));
    w.y = 1.0f + f * f * (-2.5f + 1.5f * f);
    w.z = f * (0.5f + f * (2.0f - 1.5f * f));
    w.w = f * f * (0.5f * f - 0.5f);
    return w;
}
float3 h3(uint2 v) { return float3(f16tof32(v.x), f16tof32(v.x >> 16), f16tof32(v.y)); }

float3 history_cr(float qx, float qy)
{
    const float tx = qx - 0.5f, ty = qy - 0.5f;
    const float fx0 = floor(tx), fy0 = floor(ty);
    const float4 wx = cr_weights(tx - fx0), wy = cr_weights(ty - fy0);
    const int ix = (int)fx0, iy = (int)fy0;
    const uint pitch = (uint)P_dw * 8u;
    // interior fast path (Model M's history_cr_span): every lane's 4 x 4 footprint inside the image, so the clamps are
    // the identity; six 16-byte span loads from one base instead of twelve 8-byte loads, the same texels in the same
    // sums (same mad order). Footprint inside: 1 <= ix <= P_dw - 3 and 1 <= iy <= P_dh - 3, the unsigned tests valid only
    // for an image of at least 4 x 4 (below, P_d* - 4 would wrap to a huge bound): smaller images take the clamped path.
#if LP_MUT == 8   // mutant 8: the predicate always false (the fast path never taken; same bytes, found by the counter)
    const bool interior = false;
#elif LP_MUT == 9   // mutant 9: without the size guard (the span reads past a history narrower or shorter than 4)
    const bool interior = and((uint)(ix - 1) <= (uint)(P_dw - 4), (uint)(iy - 1) <= (uint)(P_dh - 4));
#else
    const bool interior = and(and(P_dw >= 4, P_dh >= 4),
                              and((uint)(ix - 1) <= (uint)(P_dw - 4), (uint)(iy - 1) <= (uint)(P_dh - 4)));
#endif
    [branch] if (WaveActiveAllTrue(interior)) {
#if LP_HSCOUNT   // test builds only: count the waves taking the fast path, in the first dword after the full history
        if (WaveIsFirstLane()) wg_atomic_add(hist_out, (uint)(P_dw * P_dh) * 8u, 1u);
#endif
#if LP_HSEQ
        const uint b0 = hist_prev + (uint)(ix - 1) * 8u + (uint)(iy - 1) * pitch;
        const uint b1 = b0 + pitch, b2 = b0 + 2u * pitch, b3 = b0 + 3u * pitch;
        const uint4 a0 = wg_mem.Load4(b0 + 8u), a10 = wg_mem.Load4(b1), a11 = wg_mem.Load4(b1 + 16u);
        const float3 zi = float3(0.0f, 0.0f, 0.0f);
        const float3 q0 = mad(wx.z, h3(a0.zw), mad(wx.y, h3(a0.xy), zi));
        const float3 q1 = mad(wx.w, h3(a11.zw), mad(wx.z, h3(a11.xy), mad(wx.y, h3(a10.zw), mad(wx.x, h3(a10.xy), zi))));
        const uint dz = (asuint(q1.x) ^ asuint(q0.y)) & g_z;   // the second group of loads after the first group's sums (an opaque +0 in their address): 12 live load registers, not 24
        const uint4 a20 = wg_mem.Load4(b2 + dz), a21 = wg_mem.Load4(b2 + 16u + dz), a3 = wg_mem.Load4(b3 + (LP_MUT == 4 ? 16u : 8u) + dz);   // mutant 4: wrong row span
        const float3 q2 = mad(wx.w, h3(a21.zw), mad(wx.z, h3(a21.xy), mad(wx.y, h3(a20.zw), mad(wx.x, h3(a20.xy), zi))));
        const float3 q3 = mad(wx.z, h3(a3.zw), mad(wx.y, h3(a3.xy), zi));
        const float3 acci = mad(wy.w, q3, mad(wy.z, q2, mad(wy.y, q1, mad(wy.x, q0, zi))));
        const float sxi = ((wx.x + wx.y) + wx.z) + wx.w;
        const float normi = (wx.y + wx.z) * (wy.x + wy.w) + sxi * (wy.y + wy.z);
#else
        const uint b0 = hist_prev + (uint)(ix - 1) * 8u + (uint)(iy - 1) * pitch;
        const uint b1 = b0 + pitch, b2 = b0 + 2u * pitch, b3 = b0 + 3u * pitch;
        const uint4 a0 = wg_mem.Load4(b0 + 8u), a10 = wg_mem.Load4(b1), a11 = wg_mem.Load4(b1 + 16u);
        const uint4 a20 = wg_mem.Load4(b2), a21 = wg_mem.Load4(b2 + 16u), a3 = wg_mem.Load4(b3 + (LP_MUT == 4 ? 16u : 8u));   // mutant 4: wrong row span
        const float3 zi = float3(0.0f, 0.0f, 0.0f);
        const float3 q0 = mad(wx.z, h3(a0.zw), mad(wx.y, h3(a0.xy), zi));
        const float3 q1 = mad(wx.w, h3(a11.zw), mad(wx.z, h3(a11.xy), mad(wx.y, h3(a10.zw), mad(wx.x, h3(a10.xy), zi))));
        const float3 q2 = mad(wx.w, h3(a21.zw), mad(wx.z, h3(a21.xy), mad(wx.y, h3(a20.zw), mad(wx.x, h3(a20.xy), zi))));
        const float3 q3 = mad(wx.z, h3(a3.zw), mad(wx.y, h3(a3.xy), zi));
        const float3 acci = mad(wy.w, q3, mad(wy.z, q2, mad(wy.y, q1, mad(wy.x, q0, zi))));
        const float sxi = ((wx.x + wx.y) + wx.z) + wx.w;
        const float normi = (wx.y + wx.z) * (wy.x + wy.w) + sxi * (wy.y + wy.z);
#endif
        return acci * rcp(normi);
    }
    uint ro[4], co[4];
    [unroll] for (int k = 0; k < 4; ++k) {
        co[k] = hist_prev + (uint)clampi(ix - 1 + k, 0, P_dw - 1) * 8u;
        ro[k] = (uint)clampi(iy - 1 + k, 0, P_dh - 1) * pitch;
    }
#if LP_HSEQ
    const float3 z = float3(0.0f, 0.0f, 0.0f);
    const float3 r0 = mad(wx.z, h3(wg_mem.Load2(ro[0] + co[2])), mad(wx.y, h3(wg_mem.Load2(ro[0] + co[1])), z));
    const float3 r1 = mad(wx.w, h3(wg_mem.Load2(ro[1] + co[3])), mad(wx.z, h3(wg_mem.Load2(ro[1] + co[2])),
                      mad(wx.y, h3(wg_mem.Load2(ro[1] + co[1])), mad(wx.x, h3(wg_mem.Load2(ro[1] + co[0])), z))));
    const uint dzs = (asuint(r1.x) ^ asuint(r0.y)) & g_z;   // rows 2 and 3 after rows 0 and 1 (an opaque +0 in their address)
    const float3 r2 = mad(wx.w, h3(wg_mem.Load2(ro[2] + dzs + co[3])), mad(wx.z, h3(wg_mem.Load2(ro[2] + dzs + co[2])),
                      mad(wx.y, h3(wg_mem.Load2(ro[2] + dzs + co[1])), mad(wx.x, h3(wg_mem.Load2(ro[2] + dzs + co[0])), z))));
    const float3 r3 = mad(wx.z, h3(wg_mem.Load2(ro[3] + dzs + co[2])), mad(wx.y, h3(wg_mem.Load2(ro[3] + dzs + co[1])), z));
#else
    const float3 z = float3(0.0f, 0.0f, 0.0f);
    const float3 r0 = mad(wx.z, h3(wg_mem.Load2(ro[0] + co[2])), mad(wx.y, h3(wg_mem.Load2(ro[0] + co[1])), z));
    const float3 r1 = mad(wx.w, h3(wg_mem.Load2(ro[1] + co[3])), mad(wx.z, h3(wg_mem.Load2(ro[1] + co[2])),
                      mad(wx.y, h3(wg_mem.Load2(ro[1] + co[1])), mad(wx.x, h3(wg_mem.Load2(ro[1] + co[0])), z))));
    const float3 r2 = mad(wx.w, h3(wg_mem.Load2(ro[2] + co[3])), mad(wx.z, h3(wg_mem.Load2(ro[2] + co[2])),
                      mad(wx.y, h3(wg_mem.Load2(ro[2] + co[1])), mad(wx.x, h3(wg_mem.Load2(ro[2] + co[0])), z))));
    const float3 r3 = mad(wx.z, h3(wg_mem.Load2(ro[3] + co[2])), mad(wx.y, h3(wg_mem.Load2(ro[3] + co[1])), z));
#endif
    const float3 acc = mad(wy.w, r3, mad(wy.z, r2, mad(wy.y, r1, mad(wy.x, r0, z))));
    const float sx = ((wx.x + wx.y) + wx.z) + wx.w;
    const float norm = (wx.y + wx.z) * (wy.x + wy.w) + sx * (wy.y + wy.z);
    return acc * rcp(norm);
}

// ---- steps 3 to 5 of one D pixel (l_select + l_post_core) -----------------------------------------------------------
void l_post_core(LGather g, int x, int y, int fx, int fy, float mx[3], float my[3], float s, out float hh[3],
                 out float oh[3])
{
    const float inv = rcp(g.W);   // rcp(0) = inf: 0 * inf = NaN features (zero total weight)
#if LP_FBLAZY   // l_post_tf: each normalized feature from its FP16 sum where it is used (the nine floats are not live together)
#define fbar_k(k) (inv * (float)opqh(g.F[k]))
    const float sig0 = fbar_k(3), sig1 = fbar_k(4), sig2 = fbar_k(5);
#else
    float fbar[9];
    [unroll] for (int k = 0; k < 9; ++k) fbar[k] = inv * (float)opqh(g.F[k]);
#define fbar_k(k) fbar[k]
    const float sig0 = fbar[3], sig1 = fbar[4], sig2 = fbar[5];
#endif
    const bool sb1 = sig1 > sig0;
    const float best = sb1 ? sig1 : sig0;
    const bool sb2 = sig2 > best;
    const float s1 = sb2 ? sig2 : best;
    const float m1x = sb2 ? mx[2] : (sb1 ? mx[1] : mx[0]), m1y = sb2 ? my[2] : (sb1 ? my[1] : my[0]);
    const bool sf0 = !or(sb1, sb2);
    const float so1 = sf0 ? sig1 : sig0, so2 = sb2 ? sig1 : sig2;
    const float mo1x = sf0 ? mx[1] : mx[0], mo1y = sf0 ? my[1] : my[0];
#if LP_MUT == 7
    const float mo2x = sb2 ? mx[2] : mx[1], mo2y = sb2 ? my[1] : my[2];   // mutant 7: wrong candidate motion
#else
    const float mo2x = sb2 ? mx[1] : mx[2], mo2y = sb2 ? my[1] : my[2];
#endif
    const bool scc = so2 > so1;
    const float s2 = scc ? so2 : so1;
    const float m2x = scc ? mo2x : mo1x, m2y = scc ? mo2y : mo1y;
    const float q1 = clampf(k1p1 * s1 - k0p05, 0.0f, 1.0f);
    const float q2 = clampf(k1p1 * s2 - k0p05, 0.0f, 1.0f);
    const float rden = rcp(max(q1 + q2, k1em10));
    float w2 = q2 * rden, w1;
    if (w2 < kInv255) {
        w2 = 0.0f;
        w1 = 1.0f;
    } else {
        w1 = q1 * rden;
    }
    const float X = (float)x + 0.5f, Y = (float)y + 0.5f;
    const int ncand = w2 != 0.0f ? 2 : 1;
    float L[2][3];
    [unroll] for (int n = 0; n < 2; ++n) {
        L[n][0] = 0.0f; L[n][1] = 0.0f; L[n][2] = 0.0f;
        if (n < ncand) {
            const float qx = X + (n == 0 ? m1x : m2x);
            const float qy = Y + (n == 0 ? m1y : m2y);
#if LP_MUT == 5
            const bool valid = and(and(and(and(P_reset == 0, qx >= 0.0f), qx <= (float)P_dh), qy >= 0.0f), qy <= (float)P_dh);   // mutant 5
#else
            const bool valid = and(and(and(and(P_reset == 0, qx >= 0.0f), qx <= (float)P_dw), qy >= 0.0f), qy <= (float)P_dh);
#endif
            if (valid) {
                const float3 h = history_cr(qx, qy);
#if LP_LDR || LP_MORIG
                const float3 lq = h;
#else
                const float3 lq = float3(l_dec(h.x), l_dec(h.y), l_dec(h.z)) * g_rp;
#endif
                L[n][0] = lq.x;
                L[n][1] = lq.y;
                L[n][2] = lq.z;
            } else {
                float4 c = wg_ld4f(color, (uint)(fy * P_rw + fx) * 16u);
#if LP_PIDIV
                c.xyz = asfloat(asuint(pidiv3(c.xyz, g_pd, g_z)) ^ g_z);
#endif
#if LP_MORIG && !LP_LDR
                L[n][0] = enc_e(mul_nz(g_ep, c.x));
                L[n][1] = enc_e(mul_nz(g_ep, c.y));
                L[n][2] = enc_e(mul_nz(g_ep, c.z));
#else
                L[n][0] = c.x; L[n][1] = c.y; L[n][2] = c.z;
#endif
            }
        }
    }
    float Hl[3], Cl[3];
    bool hl_inf = false;
    [unroll] for (int i = 0; i < 3; ++i) {
        const float comb = ncand == 2 ? mad(w1, L[0][i], w2 * L[1][i]) : w1 * L[0][i];
#if LP_LDR
        Hl[i] = opq(comb);
#elif LP_MORIG
        Hl[i] = mul_nz(mul_nz(l_dec(comb), g_rp), g_e);
#else
        Hl[i] = mul_nz(comb, g_e);
#endif
        hl_inf = or(hl_inf, abs(Hl[i]) == asfloat(0x7f800000u));
#if LP_LDR
        Cl[i] = fbar_k(i);
#else
        Cl[i] = l_dec(fbar_k(i));
#endif
    }
    const float alpha = clampf(opq(s * g.W) * fbar_k(6), 0.0f, 1.0f);
    const float bh2 = mad(alpha, 1.0f - fbar_k(7), fbar_k(7));
    const float bo2 = mad(alpha, 1.0f - fbar_k(8), fbar_k(8));
    [unroll] for (int i = 0; i < 3; ++i) {
        hh[i] = mad(Cl[i] - Hl[i], bh2, Hl[i]);
        oh[i] = mad(Cl[i] - Hl[i], bo2, Hl[i]);
    }
    const float sh = hh[0] + hh[1] + hh[2], so = oh[0] + oh[1] + oh[2];
    if (or(isnan(sh), hl_inf)) {
        [unroll] for (int i = 0; i < 3; ++i) hh[i] = Cl[i];
    }
    if (or(isnan(so), hl_inf)) {
        [unroll] for (int i = 0; i < 3; ++i) oh[i] = Cl[i];
    }
}

// the render-resolution motion words of one pixel row: m_near / m_far at the nearest texel (fx, fy), m_center's four
// bilinear neighbours (clamped texel coordinates) of (px, py); loaded ahead of the gather (they do not depend on it)
struct MotRaw { uint mn, mf, ma, mb, mc, md; int fx, fy; float wx, wy; };

MotRaw motion_load(float px, float py)
{
    MotRaw m;
    m.fx = clampi(floor_i(px), 0, P_rw - 1);
    m.fy = clampi(floor_i(py), 0, P_rh - 1);
    m.mn = wg_ld32(m_near, (uint)(m.fy * P_rw + m.fx) * 4u);
    m.mf = wg_ld32(m_far, (uint)(m.fy * P_rw + m.fx) * 4u);
    // motion_bilinear of m_center at (px, py)
    const float tx = px - 0.5f, ty = py - 0.5f;
    const float x0f = floor(tx), y0f = floor(ty);
    m.wx = tx - x0f;
    m.wy = ty - y0f;
    const int c0 = (int)x0f, r0 = (int)y0f;
    const int ca = clampi(c0, 0, P_rw - 1), cb = clampi(c0 + 1, 0, P_rw - 1);
    const int ra = clampi(r0, 0, P_rh - 1), rb = clampi(r0 + 1, 0, P_rh - 1);
    if (ca + 1 == cb) {
        const uint2 top = wg_ld64(m_center, (uint)(ra * P_rw + ca) * 4u);
        const uint2 bot = wg_ld64(m_center, (uint)(rb * P_rw + ca) * 4u);
        m.ma = top.x; m.mb = top.y;
        m.mc = bot.x; m.md = bot.y;
    } else {
        m.ma = wg_ld32(m_center, (uint)(ra * P_rw + ca) * 4u);
        m.mb = wg_ld32(m_center, (uint)(ra * P_rw + cb) * 4u);
        m.mc = wg_ld32(m_center, (uint)(rb * P_rw + ca) * 4u);
        m.md = wg_ld32(m_center, (uint)(rb * P_rw + cb) * 4u);
    }
    return m;
}

// l_finish_pixel without its stores: motions, steps 3 to 5; H^ in hh, O^ in oh (the kernel converts and stores them)
void finish_pixel(LGather g, int x, int y, MotRaw m, float s, out float hh[3], out float oh[3])
{
    float mx[3], my[3];
    mx[0] = f16tof32(m.mn); my[0] = f16tof32(m.mn >> 16);
    mx[2] = f16tof32(m.mf); my[2] = f16tof32(m.mf >> 16);
    {   // motion_bilinear of m_center, clamped texel coordinates
        const float owx = 1.0f - m.wx, owy = 1.0f - m.wy;
        const float topx = f16tof32(m.ma) * owx + f16tof32(m.mb) * m.wx, topy = f16tof32(m.ma >> 16) * owx + f16tof32(m.mb >> 16) * m.wx;
        const float botx = f16tof32(m.mc) * owx + f16tof32(m.md) * m.wx, boty = f16tof32(m.mc >> 16) * owx + f16tof32(m.md >> 16) * m.wx;
        mx[1] = topx * owy + botx * m.wy;
        my[1] = topy * owy + boty * m.wy;
    }
    l_post_core(g, x, y, m.fx, m.fy, mx, my, s, hh, oh);
}

float lpx(int xx) { return mad(opq((float)xx + 0.5f), P_rx, P_jx); }   // fused(x + 1/2, rx, jx)
float lpy(int yy) { return mad(P_ry, opq((float)yy + 0.5f), P_jy); }   // fused(ry, y + 1/2, jy)

// one column i of the 4 x 4 gather (taps j = -1 ... 2 for both rows of the pair), from the staged texels (STAGED) or loaded
// per tap; the eight weights of the column are rounded to FP16 by one driver conversion
template<bool STAGED>
void gather_col(int i, half dx, int tb0, int rstride, int b, int ax, half v0[kLWin], half v1[kLWin], half vv0[kLWin],
                half vv1[kLWin], inout LGather g0, inout LGather g1)
{
    const half u = dx + (half)(float)i;
    const half uu = mulh(u, u);
    v8f wv;
#if LP_MORIG
    wv.v[6] = 0.0f;
    wv.v[7] = 0.0f;
#endif
    if (STAGED) {
        // Read only kA/kB/kC (dwords 4..5) from LDS during weight evaluation so 24 texel VGPRs aren't live across exp2/cvt.
        const int col_base = tb0 + (i + 1) * kTexDw;
        [unroll] for (int j = -1; j <= kLWin - 2; ++j) {
            const int base = col_base + (j + 1) * rstride
#if LP_MUT == 2
                             + (j == 2 && i == 2 ? kTexDw : 0)   // mutant 2: one tap reads the wrong texel
#endif
                             ;
            uint tk[kTexDw];
            tk[0] = 0u; tk[1] = 0u; tk[2] = 0u; tk[3] = 0u;
            tk[4] = sh_tex[base + 4];
            tk[5] = sh_tex[base + 5];
            wv.v[2 * (j + 1)] = tap_weight(tk, uu, vv0[j + 1], mulh(u, v0[j + 1]));
            wv.v[2 * (j + 1) + 1] = tap_weight(tk, uu, vv1[j + 1], mulh(u, v1[j + 1]));
        }
        const uint4 hw = wg_to_f16_ags(wv);
        [unroll] for (int jj = 0; jj < kLWin; ++jj) {
            const int base = col_base + jj * rstride
#if LP_MUT == 2
                             + (jj == 3 && i == 2 ? kTexDw : 0)
#endif
                             ;
            uint tf[kTexDw];
            [unroll] for (int d = 0; d < 5; ++d) tf[d] = sh_tex[base + d];
            tf[5] = 0u;
            const uint hp = jj == 0 ? hw.x : (jj == 1 ? hw.y : (jj == 2 ? hw.z : hw.w));
            gather_acc(g0, tf, wv.v[2 * jj], hp & 0xffffu);
            gather_acc(g1, tf, wv.v[2 * jj + 1], hp >> 16);
        }
    } else {
        // Rare unstaged fallback (footprint > kStageCap): avoid keeping uint t[4][6] (24 VGPRs) live across the kernel!
        [unroll] for (int j = -1; j <= kLWin - 2; ++j) {
            uint tk[kTexDw];
            load_texel_hw(clampi(b + j, 0, P_rh - 1) * P_rw + clampi(ax + i, 0, P_rw - 1), tk);
            wv.v[2 * (j + 1)] = tap_weight(tk, uu, vv0[j + 1], mulh(u, v0[j + 1]));
            wv.v[2 * (j + 1) + 1] = tap_weight(tk, uu, vv1[j + 1], mulh(u, v1[j + 1]));
        }
        const uint4 hw = wg_to_f16_ags(wv);
        [unroll] for (int jj = 0; jj < kLWin; ++jj) {
            uint tf[kTexDw];
            load_texel_hw(clampi(b + jj - 1, 0, P_rh - 1) * P_rw + clampi(ax + i, 0, P_rw - 1), tf);
            const uint hp = jj == 0 ? hw.x : (jj == 1 ? hw.y : (jj == 2 ? hw.z : hw.w));
            gather_acc(g0, tf, wv.v[2 * jj], hp & 0xffffu);
            gather_acc(g1, tf, wv.v[2 * jj + 1], hp >> 16);
        }
    }
}

