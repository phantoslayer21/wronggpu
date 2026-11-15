// SPDX-License-Identifier: MIT
// D3D12 port of Model L's reconstruction kernel (issue #89): l_post_recon of l_post.co (clean/l/kernels/l_post.hip, HDR),
// byte for byte. Arguments, buffers, layouts and grid are Model M's post_recon (clean/kernels/post.hip): grid
// ceil(dw / 32) * ceil(ceil(dh / 2) / 8) workgroups of 256 lanes; lane (l, k) of wave k handles D column x0 + l and the
// row pair k0 + k. w = the post blob (w[0] = s). Model L's changes against post_recon: the 4 x 4 colour gather in the
// model's precision (FP16 features, offsets, exponent and accumulators), linear-colour history candidates
// L = D(h) r_prev (r_prev = 1 / e_prev) combined as fma(w1, L1, w2 L2) e, and the render pixel's colour as the
// fallback.
//
// The operations follow HIP's ISA of l_post_recon (llvm-objdump of l_post.co), not only its source:
// - E_in(e c) and the B / A codes k / 255 (an IEEE division: wg_div) are rounded to FP16 by v_cvt_f16_f32 (wg_f2h);
//   the kernel shape A, B, C = (_Float16)(k * const) is one rounding (v_fma_mixlo_f16 k, const, 0: wg_mul_f16);
// - the gather's FP16 products and sums are v_mul_f16 / v_add_f16, the exponent and the accumulators v_fma_f16 (mad on
//   half). The driver fuses an FP16 multiply with a following add (wg_math.hlsli), so each FP16 product passes through
//   opqh (an integer xor with a zero it cannot see); the accumulators start from an opaque +0 (fma(w, f, +0) as HIP);
// - an FP16 result that is widened to FP32 (eps for the exponential, the accumulators for the normalization) passes
//   through opqh first: otherwise the driver computes its last fma in FP32 (v_fma_mix_f32) and drops the FP16 rounding;
// - the tap weight w = 2^eps (v_exp_f32) is flushed to +0 below 2^-126 and rounded to FP16 by v_cvt_f16_f32 (wg_f2h);
// - the full history's texel is (_Float16)(kEncK log2(..)) in one rounding (v_fma_mixlo_f16: wg_mul_f16); the display
//   value is clamp(E(o), 0, 2.45) rounded by v_cvt_f16_f32 (wg_f2h);
// - r_prev = 1 / e_prev is one IEEE division per launch (wg_div), then L = D(h) r_prev (HIP: the same product; the
//   spec's division as a product with its reciprocal is free within the post tolerance, spec answer Q5).
// Model M's DXILs are untouched: this file only adds the kernel (bound to l_post.co).
// l_post_recon_p12 (LP_P12 = 1): the same kernel on the pack12 coefficients of l_coef_pack12: A = image A8 (8 bytes per
// render pixel: codes A0 A1 A2 B0 | B1 B2, C2 FP16), C = image C4 (4 bytes per render pixel: C0, C1 FP16), B unused.
//
// LDR forms (LP_LDR = 1; l_post.hip's l_post_recon_ldr / l_post_recon_p12_ldr, spec preset L part 3 item 6, Q1, Q2):
// identity transfer (the colour feature is the driver's FP16 conversion of the colour, a NaN kept; no decode of the
// history candidates or of C), no exposure (not read: e = e_prev = 1), the full history the driver's FP16 conversion of
// H^, the display image h(code RN32(1 / N)) of the 11 / 11 / 10-bit code floor(clamp(fused(O^ N + 1/2), 0, N)) (a NaN
// gives 0), the reduced codes rint(clamp(mean, 0, 1) 1023).
//
// @wg-kernel l_post_recon co=l_post.co block=256 wave=32 args=p,p,p,p,p,p,p,p,p,p,p,p,b64,p
// @wg-kernel l_post_recon_p12 co=l_post.co entry=l_post_recon_p12 block=256 wave=32 args=p,p,p,p,p,p,p,p,p,p,p,p,b64,p defines=LP_P12=1,LP_ENCFOLD=1 needs=fold isa=v_fma_mixlo_f16@3dddf184+v_fma_mixhi_f16@3dddf184==6,s_round_mode==0
// @wg-kernel l_post_recon_p12 co=l_post.co entry=l_post_recon_p12 file=l_post_recon_p12__nofold block=256 wave=32 args=p,p,p,p,p,p,p,p,p,p,p,p,b64,p defines=LP_P12=1 needs=nofold
// @wg-kernel l_post_recon_ldr co=l_post.co entry=l_post_recon_ldr block=256 wave=32 args=p,p,p,p,p,p,p,p,p,p,p,p,b64,p defines=LP_LDR=1
// @wg-kernel l_post_recon_p12_ldr co=l_post.co entry=l_post_recon_p12_ldr block=256 wave=32 args=p,p,p,p,p,p,p,p,p,p,p,p,b64,p defines=LP_P12=1,LP_LDR=1
#include "wg_platform.hlsli"
#ifndef LP_P12
#define LP_P12 0
#endif
#ifndef LP_LDR
#define LP_LDR 0
#endif

WG_ARGS_BEGIN
    uint color, A, B, C, hist_prev, m_near, m_center, m_far, w_p, hist_out, reduced, display;   // pointers
    int P_rw, P_rh, P_dw, P_dh, P_reset, P_pad0, P_pad1, P_pad2;                                  // WgPostParams
    float P_rx, P_ry, P_jx, P_jy, P_e, P_e_prev, P_ratio, P_pad3;
    uint expo;
WG_ARGS_END

#include "l_post_encode.hlsli"   // kEncK and the history encode through the driver's conversion fold
#define kThreads 256
#define kTileX 32
#define kTilePairs 8
#define kStageCap 512
#define kLWin 4
#define kTexDw 6   // staged texel: f0 f1 | f2 f3 | f4 f5 | f6 f7 | f8 kA | kB kC (FP16 bits)

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
float opq(float x) { return asfloat(asuint(x) ^ g_z); }
half opqh(half x) { return asfloat16((uint16_t)(asuint16(x) ^ (uint16_t)g_z)); }
float mul_nz(float a, float b) { return mad(a, b, asfloat(g_z | 0x80000000u)); }
#ifndef LP_MUT
#define LP_MUT 0   // 1..9: negative controls of the byte checks (never shipped)
#endif
#ifndef LP_DP
#define LP_DP 1
#endif
#ifndef LP_ENCFOLD
#define LP_ENCFOLD 0   // 1 (l_post_recon_p12's fold port): the history encode through the driver's conversion fold
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
    v8f cv;
    cv.v[0] = opq(wg_div((float)lane, 255.0f));
    [unroll] for (int i = 1; i < 8; ++i) cv.v[i] = 0.0f;
    sh_code[lane] = wg_to_f16_ags(cv).x & 0xffffu;
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
    const uint kA = mul_zero(k0, kShapeAB) ? 0u : (h.y >> 16), kB = mul_zero(k1, kShapeAB) ? 0u : (h.z & 0xffffu);
    const uint kC = mul_zero(k2, kShapeC) ? 0u : (h.z >> 16);
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
#if LP_MUT == 1
    const half ta = mulh(uu, kB), tb = mulh(vv, kB);   // mutant 1: wrong factor
#else
    const half ta = mulh(uu, kA), tb = mulh(vv, kB);
#endif
    const half s = tb + ta;
    const half eps = mad(uv, kC, s);
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
        return acci * rcp(normi);
    }
    uint ro[4], co[4];
    [unroll] for (int k = 0; k < 4; ++k) {
        co[k] = hist_prev + (uint)clampi(ix - 1 + k, 0, P_dw - 1) * 8u;
        ro[k] = (uint)clampi(iy - 1 + k, 0, P_dh - 1) * pitch;
    }
    const float3 z = float3(0.0f, 0.0f, 0.0f);
    const float3 r0 = mad(wx.z, h3(wg_mem.Load2(ro[0] + co[2])), mad(wx.y, h3(wg_mem.Load2(ro[0] + co[1])), z));
    const float3 r1 = mad(wx.w, h3(wg_mem.Load2(ro[1] + co[3])), mad(wx.z, h3(wg_mem.Load2(ro[1] + co[2])),
                      mad(wx.y, h3(wg_mem.Load2(ro[1] + co[1])), mad(wx.x, h3(wg_mem.Load2(ro[1] + co[0])), z))));
    const float3 r2 = mad(wx.w, h3(wg_mem.Load2(ro[2] + co[3])), mad(wx.z, h3(wg_mem.Load2(ro[2] + co[2])),
                      mad(wx.y, h3(wg_mem.Load2(ro[2] + co[1])), mad(wx.x, h3(wg_mem.Load2(ro[2] + co[0])), z))));
    const float3 r3 = mad(wx.z, h3(wg_mem.Load2(ro[3] + co[2])), mad(wx.y, h3(wg_mem.Load2(ro[3] + co[1])), z));
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
    float fbar[9];
    [unroll] for (int k = 0; k < 9; ++k) fbar[k] = inv * (float)opqh(g.F[k]);
    const float sig0 = fbar[3], sig1 = fbar[4], sig2 = fbar[5];
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
#if LP_LDR
                const float3 lq = h;
#else
                const float3 lq = float3(l_dec(h.x), l_dec(h.y), l_dec(h.z)) * g_rp;
#endif
                L[n][0] = lq.x;
                L[n][1] = lq.y;
                L[n][2] = lq.z;
            } else {
                const float4 c = wg_ld4f(color, (uint)(fy * P_rw + fx) * 16u);
                L[n][0] = c.x; L[n][1] = c.y; L[n][2] = c.z;
            }
        }
    }
    float Hl[3], Cl[3];
    bool hl_inf = false;
    [unroll] for (int i = 0; i < 3; ++i) {
        const float comb = ncand == 2 ? mad(w1, L[0][i], w2 * L[1][i]) : w1 * L[0][i];
#if LP_LDR
        Hl[i] = opq(comb);
#else
        Hl[i] = mul_nz(comb, g_e);
#endif
        hl_inf = or(hl_inf, abs(Hl[i]) == asfloat(0x7f800000u));
#if LP_LDR
        Cl[i] = fbar[i];
#else
        Cl[i] = l_dec(fbar[i]);
#endif
    }
    const float alpha = clampf(opq(s * g.W) * fbar[6], 0.0f, 1.0f);
    const float bh2 = mad(alpha, 1.0f - fbar[7], fbar[7]);
    const float bo2 = mad(alpha, 1.0f - fbar[8], fbar[8]);
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

// l_finish_pixel without its stores: motions, steps 3 to 5; H^ in hh, O^ in oh (the kernel converts and stores them)
void finish_pixel(LGather g, int x, int y, float px, float py, float s, out float hh[3], out float oh[3])
{
    const int fx = clampi(floor_i(px), 0, P_rw - 1), fy = clampi(floor_i(py), 0, P_rh - 1);
    float mx[3], my[3];
    const uint mn = wg_ld32(m_near, (uint)(fy * P_rw + fx) * 4u), mf = wg_ld32(m_far, (uint)(fy * P_rw + fx) * 4u);
    mx[0] = f16tof32(mn); my[0] = f16tof32(mn >> 16);
    mx[2] = f16tof32(mf); my[2] = f16tof32(mf >> 16);
    {   // motion_bilinear of m_center at (px, py), clamped texel coordinates
        const float tx = px - 0.5f, ty = py - 0.5f;
        const float x0f = floor(tx), y0f = floor(ty);
        const float wx = tx - x0f, wy = ty - y0f;
        const int c0 = (int)x0f, r0 = (int)y0f;
        const int ca = clampi(c0, 0, P_rw - 1), cb = clampi(c0 + 1, 0, P_rw - 1);
        const int ra = clampi(r0, 0, P_rh - 1), rb = clampi(r0 + 1, 0, P_rh - 1);
        const uint ma = wg_ld32(m_center, (uint)(ra * P_rw + ca) * 4u), mb = wg_ld32(m_center, (uint)(ra * P_rw + cb) * 4u);
        const uint mc = wg_ld32(m_center, (uint)(rb * P_rw + ca) * 4u), md = wg_ld32(m_center, (uint)(rb * P_rw + cb) * 4u);
        const float owx = 1.0f - wx, owy = 1.0f - wy;
        const float topx = f16tof32(ma) * owx + f16tof32(mb) * wx, topy = f16tof32(ma >> 16) * owx + f16tof32(mb >> 16) * wx;
        const float botx = f16tof32(mc) * owx + f16tof32(md) * wx, boty = f16tof32(mc >> 16) * owx + f16tof32(md >> 16) * wx;
        mx[1] = topx * owy + botx * wy;
        my[1] = topy * owy + boty * wy;
    }
    l_post_core(g, x, y, fx, fy, mx, my, s, hh, oh);
}

float lpx(int xx) { return mad(opq((float)xx + 0.5f), P_rx, P_jx); }   // fused(x + 1/2, rx, jx)
float lpy(int yy) { return mad(P_ry, opq((float)yy + 0.5f), P_jy); }   // fused(ry, y + 1/2, jy)

#if LP_P12 && LP_LDR
WG_KERNEL(l_post_recon_p12_ldr)
#elif LP_LDR
WG_KERNEL(l_post_recon_ldr)
#elif LP_P12
WG_KERNEL(l_post_recon_p12)
#else
WG_KERNEL(l_post_recon)
#endif
{
    WG_SETUP();
    g_z = wg_grid_ >> 31;
#if LP_LDR
    g_e = 1.0f;   // LDR: no exposure (the exposure pair is not read)
    g_ep = 1.0f;
    g_rp = 1.0f;
#else
    g_e = P_e;
    g_ep = P_e_prev;
    if (expo != 0u) {
        const float e0 = wg_ldf(expo, 0u), e1 = wg_ldf(expo, 4u);
        g_e = e0 == 0.0f ? 1.0f : e0;
        g_ep = e1 == 0.0f ? 1.0f : e1;
    }
    g_rp = wg_div(1.0f, g_ep);
#endif
    const int lane = (int)wg_lane_id();
    const int tiles_x = (P_dw + kTileX - 1) / kTileX;
    const int tile = (int)wg_block_id();
    const int x0 = (tile % tiles_x) * kTileX;
    const int k0 = (tile / tiles_x) * kTilePairs;
    const int npairs = (P_dh + 1) / 2;
    const float s = wg_ldf(w_p, 0u);
    code16_build(lane);
    GroupMemoryBarrierWithGroupSync();

    // render footprint of the tile: window columns ax - 1 ... ax + 2 of its first and last column, window rows of its
    // first and last row pair
    const int xl = mini(x0 + kTileX - 1, P_dw - 1), kl = mini(k0 + kTilePairs - 1, npairs - 1);
    const int c0 = floor_i(lpx(x0) - 0.5f) - 1, c1 = floor_i(lpx(xl) - 0.5f) + (kLWin - 2);
    const int r0 = floor_i(lpy(2 * k0) - 0.5f) - 1, r1 = floor_i(lpy(2 * kl) - 0.5f) + (kLWin - 2);
    const int ncol = c1 - c0 + 1, nrow = r1 - r0 + 1;
    const bool staged = ncol > 0 && nrow > 0 && ncol <= kStageCap && nrow <= kStageCap && ncol * nrow <= kStageCap;
    if (staged) {   // wave-uniform rounds (the driver's conversion in load_texel_hw); lanes past the end repeat the last
        const int ntex = ncol * nrow;
        [loop] for (int base = 0; base < ntex; base += kThreads) {
            const int i = base + lane, ii = mini(i, ntex - 1);
            const int r = ii / ncol, c = ii - r * ncol;
            const int idx = clampi(r0 + r, 0, P_rh - 1) * P_rw + clampi(c0 + c, 0, P_rw - 1);
            uint t[kTexDw];
            load_texel_hw(idx, t);
            if (i < ntex) {
                [unroll] for (int d = 0; d < kTexDw; ++d) sh_tex[i * kTexDw + d] = t[d];
            }
        }
    }
    GroupMemoryBarrierWithGroupSync();

    const bool active = and(x0 + (lane & 31) < P_dw, k0 + (lane >> 5) < npairs);
    const int x = mini(x0 + (lane & 31), P_dw - 1);
    const int y0 = 2 * mini(k0 + (lane >> 5), npairs - 1);
    const bool has1 = y0 + 1 < P_dh;
    const float px = lpx(x), py0 = lpy(y0), py1 = lpy(y0 + 1);
    const int ax = floor_i(px - 0.5f), b = floor_i(py0 - 0.5f);

    // gather offsets: dx = h((ax + 1/2) - px), d_y = h((b_y + 1/2) - py(y)) minus 1 (FP16) when the row's own b_y differs
    // from the pair's b; u = h(dx + i), v = h(d_y + j); taps i = -1 ... 2 (outer), j = -1 ... 2 (inner)
    const int b1 = floor_i(py1 - 0.5f);
    v8f ov;   // the three offsets in one conversion
    ov.v[0] = opq(((float)ax + 0.5f) - px);
    ov.v[1] = opq(((float)b + 0.5f) - py0);
    ov.v[2] = opq(((float)b1 + 0.5f) - py1);
    [unroll] for (int oi = 3; oi < 8; ++oi) ov.v[oi] = 0.0f;
    const uint4 oh16 = wg_to_f16_ags(ov);
    const half dx = h_bits(oh16.x), dy0 = h_bits(oh16.x >> 16);
    half dy1 = h_bits(oh16.y);
    if (b1 != b) dy1 = dy1 - (half)1.0f;
    half v0[kLWin], v1[kLWin], vv0[kLWin], vv1[kLWin];
    [unroll] for (int jj = 0; jj < kLWin; ++jj) {
        v0[jj] = dy0 + (half)(float)(jj - 1);
        v1[jj] = dy1 + (half)(float)(jj - 1);
        vv0[jj] = mulh(v0[jj], v0[jj]);
        vv1[jj] = mulh(v1[jj], v1[jj]);
    }
    // taps column by column (i outer, j inner, each gather in that order as before); per column the eight weights (four
    // rows, two pixels) are rounded to FP16 by one driver conversion
    LGather g0 = gather_zero(), g1 = gather_zero();
    // the staged texel's LDS index without per-tap multiplies: one base per lane, the row stride and the column
    // step wave-uniform (integer identity: ((b + j - r0) ncol + (ax + i - c0)) kTexDw)
    const int tb0 = ((b - 1 - r0) * ncol + (ax - 1 - c0)) * kTexDw, rstride = ncol * kTexDw;
    [loop] for (int i = -1; i <= kLWin - 2; ++i) {
        const half u = dx + (half)(float)i;
        const half uu = mulh(u, u);
        uint t[kLWin][kTexDw];
        v8f wv;
        [unroll] for (int j = -1; j <= kLWin - 2; ++j) {
            if (staged) {
                const int base = tb0 + (j + 1) * rstride + (i + 1) * kTexDw
#if LP_MUT == 2
                                 + (j == 2 && i == 2 ? kTexDw : 0)   // mutant 2: one tap reads the wrong texel
#endif
                                 ;
                [unroll] for (int d = 0; d < kTexDw; ++d) t[j + 1][d] = sh_tex[base + d];
            } else {
                load_texel_hw(clampi(b + j, 0, P_rh - 1) * P_rw + clampi(ax + i, 0, P_rw - 1), t[j + 1]);
            }
            wv.v[2 * (j + 1)] = tap_weight(t[j + 1], uu, vv0[j + 1], mulh(u, v0[j + 1]));
            wv.v[2 * (j + 1) + 1] = tap_weight(t[j + 1], uu, vv1[j + 1], mulh(u, v1[j + 1]));
        }
        const uint4 hw = wg_to_f16_ags(wv);
        [unroll] for (int jj = 0; jj < kLWin; ++jj) {
            const uint hp = jj == 0 ? hw.x : (jj == 1 ? hw.y : (jj == 2 ? hw.z : hw.w));
            gather_acc(g0, t[jj], wv.v[2 * jj], hp & 0xffffu);
            gather_acc(g1, t[jj], wv.v[2 * jj + 1], hp >> 16);
        }
    }

    // the two rows of the pair (wave-uniform: every lane runs both; a missing second row is not stored and adds +0 as
    // before); the full history and display values of a row in one driver conversion
    float r[3] = {0.0f, 0.0f, 0.0f};
    const int pitch = (P_dw + 63) & ~63, plane = pitch * P_dh;
    [loop] for (int row = 0; row < 2; ++row) {
        const bool second = row == 1;
        LGather g = g0;
        if (second) g = g1;
        float hh[3], oo[3];
        finish_pixel(g, x, y0 + row, px, second ? py1 : py0, s, hh, oo);
#if !LP_LDR && LP_ENCFOLD
        float xe[3], de[3];
        [unroll] for (int ce = 0; ce < 3; ++ce) {
            xe[ce] = enc_log(hh[ce]);
            de[ce] = opq(clampf(enc_e(oo[ce]), 0.0f, kDisplayMax));
        }
        const uint4 hc = lp_encode_fold(xe[0], xe[1], xe[2], de[0], de[1], de[2], g_z);
#else
        v8f cv;
        [unroll] for (int ci = 0; ci < 3; ++ci) {
#if LP_LDR
            // h(H^); the display code floor(clamp(fused(O^ N + 1/2), 0, N)), then h(code RN32(1 / N)) (one rounding or
            // two give the same FP16 value for every code 0 ... N)
            const float n = ci == 2 ? 1023.0f : 2047.0f;
            const float k = asfloat(ci == 2 ? 0x3a802008u : 0x3a001002u);
            cv.v[ci] = opq(hh[ci]);
            cv.v[3 + ci] = opq(floor(clampf(mad(oo[ci], n, 0.5f), 0.0f, n)) * k);
#else
            cv.v[ci] = mul_odd(kEncK, enc_log(hh[ci]));
            cv.v[3 + ci] = opq(clampf(enc_e(oo[ci]), 0.0f, kDisplayMax));
#endif
        }
        cv.v[6] = 0.0f;
        cv.v[7] = 0.0f;
        const uint4 hc = wg_to_f16_ags(cv);
#endif
        const int y = y0 + row;
        if (and(active, or(!second, has1))) {
#if LP_LDR
            const uint q0 = hc.x & 0xffffu, q1 = hc.x >> 16, q2 = hc.y & 0xffffu;
#else
#if LP_ENCFOLD
            const uint q0 = hc.x & 0xffffu, q1 = hc.x >> 16, q2 = hc.y & 0xffffu;   // the fold's +0 addend gives +0
#else
            const uint q0 = mul_zero(kEncK, enc_log(hh[0])) ? 0u : (hc.x & 0xffffu);
            const uint q1 = mul_zero(kEncK, enc_log(hh[1])) ? 0u : (hc.x >> 16);
            const uint q2 = mul_zero(kEncK, enc_log(hh[2])) ? 0u : (hc.y & 0xffffu);
#endif
#endif
            wg_st64(hist_out, (uint)(y * P_dw + x) * 8u, uint2(q0 | (q1 << 16), q2));
            const int di = y * pitch + x;
            wg_st16(display, (uint)di * 2u, hc.y >> 16);
            wg_st16(display, (uint)(plane + di) * 2u, hc.z & 0xffffu);
            wg_st16(display, (uint)(2 * plane + di) * 2u, hc.z >> 16);
        }
        // reduced history: mean of H^ over the 2 x 2 block (x, x + 1) x (y0, y0 + 1): r = h0 + h1 (h1 = 0 without a
        // second row)
        [unroll] for (int ri = 0; ri < 3; ++ri) r[ri] = second ? r[ri] + (has1 ? hh[ri] : 0.0f) : hh[ri];
    }
    // lane x + 1 is lane ^ 1 (same wave)
    float o[3];
    [unroll] for (int i = 0; i < 3; ++i) o[i] = wg_xor(r[i], 1u);
    if (active && has1 && (x & 1) == 0 && x + 1 < P_dw) {
        uint code = 0u;
        [unroll] for (int i = 0; i < 3; ++i) {
            const float m = opq((r[i] + o[i]) * 0.25f);
#if LP_LDR
            const float v = clampf(m, 0.0f, 1.0f);
#else
            const float v = clampf(opq(enc_e(m)) * kInv245, 0.0f, 1.0f);
#endif
            code |= wg_f2u(round(v * 1023.0f)) << (10 * i);
        }
        wg_st32(reduced, (uint)((y0 / 2) * (P_dw / 2) + x / 2) * 4u, code);
    }
}
