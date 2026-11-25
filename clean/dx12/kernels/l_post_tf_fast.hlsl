// SPDX-License-Identifier: MIT
// Model L's true-fusion output kernel in its fast-mode form (runtime l_mode=fast, native D3D12 only, no HIP kernel):
// w64_lc12/l_post_tf_p12_fast and _pf_fast. The same workgroup / chunk plan, staging, ring and filter as
// l_post_tf.hlsl (read its header: one wave64 kernel of 512 lanes, the reconstruction of a D pixel from the staged FP16
// features, the history candidates, steps 3 to 5, the FP16 display values, then the analytic Lanczos passes on the ring in
// LDS, Dec, and the output times dec_scale), with the arithmetic of the hybrid route's bit-exactness removed:
//   - no driver intrinsic channel (wg_ags_call, 0x1C): FP32 -> FP16 is the native pack (v_cvt_pk_rtz_f16_f32, round toward
//     zero) with the round-to-nearest bias folded into the multiplier in front of it (kRn: x (1 + 2^-11.7) truncated has
//     the mean error of the nearest rounding) or, where no multiplier precedes, an integer add of half an FP16 step
//     (rn16); the opaque-zero guards (opq, opqh, mulh, mul_nz, mul_odd, mul_zero_k) are gone, the driver may fuse and
//     regroup (the values are the model's, not the reference's bits);
//   - the FP16 code table (sh_code, 1 KB of LDS, a barrier and six lookups per texel) is a multiply: (half)(byte / 255);
//   - the gather's tap weight is one native FP16 exponential (v_exp_f16) of the FP16 exponent, and the total weight W is
//     summed in FP16 from the same rounded weights as the nine features (the normalization is then self-consistent and the
//     rounding of the weights adds no bias), instead of FP32 exp2 + a round to FP16 + an FP32 sum;
//   - Dec is exp2(q kDecK) - 1 for every lane (no polynomial branch, no wave vote: the cancellation error of the plain form
//     is 1.2e-7 of dec_scale at the darkest clamp, 1e-4, below the display's FP16 resolution);
//   - the exposures, 0.01 and 1 / e_prev fold into two wave-uniform factors; the output scale is dec_scale * pi (one
//     uniform product; with _pf the colours are multiplied by 1 / pi);
//   - the horizontal Lanczos results sit in the ring as FP16 pairs (6.7 KB instead of 13.4 KB) and the vertical pass
//     filters two adjacent output columns per lane from whole dwords.
// LDS: 12 288 B staged texels + 10 752 B ring = 23 040 B (l_post_tf_p12: 26 752 B).
// The quality gate is measured on the GPU (tone-mapped PSNR / SSIM against l_mode=hybrid, temporal stability): the hybrid
// route stays the bit-exact one; this kernel is selected only by l_mode=fast.
// @wg-kernel w64_lc12/l_post_tf_p12_fast co=- entry=l_post_tf_p12_fast block=512 wave=64 args=p,p,p,p,p,p,p,p,p,p,p,p,b64,p,u,u,u,u,u,u,u,u defines=LP_W64=1,LP_FAST=1
// @wg-kernel w64_lc12/l_post_tf_p12_pf_fast co=- entry=l_post_tf_p12_fast block=512 wave=64 args=p,p,p,p,p,p,p,p,p,p,p,p,b64,p,u,u,u,u,u,u,u,u defines=LP_W64=1,LP_FAST=1,LP_PIDIV=1
#include "wg_platform.hlsli"
#ifndef LP_PIDIV
#define LP_PIDIV 0
#endif
#if WG_WAVE != 64 || WG_BLOCK != 512
#error "l_post_tf_fast.hlsl: wave64, 512 lanes"
#endif

WG_ARGS_BEGIN
    uint color, A, B, C, hist_prev, m_near, m_center, m_far, w_p, hist_out, reduced, out_p;   // pointers (A: A8, C: C4)
    int P_rw, P_rh, P_dw, P_dh, P_reset, P_pad0, P_pad1, P_pad2;                                // WgPostParams
    float P_rx, P_ry, P_jx, P_jy, P_e, P_e_prev, P_ratio, P_pad3;
    uint expo;
    int ow, oh;
    float dec_scale_arg, pi_arg;
    int pitch_bytes, cpairs, k1, c2;
WG_ARGS_END

#define kStageCap 512   // staged render texels per step (6 dwords each: 12288 bytes); one per lane
#define kTexDw 6        // staged texel: f0 f1 | f2 f3 | f4 f5 | f6 f7 | f8 kA | kB kC (FP16 bits)
#define kLWin 4
#define kTfCols 64
#define kTfOwn 60
#define kTfOutX 40
#define kNW 8                      // row pairs per step (= waves64 of the workgroup)
#define kRing (2 * kNW + 6)        // 22
#define kThreads (64 * kNW)        // 512
#define kRowDw 96                  // one ring row: the display pass (3 channels x 32 dwords) / the filtered row (3 x 20 dwords of two FP16)
#define kLdsDw ((kRing + 6) * kRowDw)   // 2688 dwords = 10752 bytes (+ 12288 staged texels = 23040)
#define kPairs (kTfOutX / 2)       // 20 dwords of two adjacent output columns per channel

groupshared uint sh_tex[kStageCap * kTexDw];
groupshared uint lds[kLdsDw];

static const float kEncK = asfloat(0x3dddf184u);        // 0.36f * 0.30102999566398120f
static const float kDecK = asfloat(0x4113a428u);        // 3.32192809488736235f / 0.36f
static const float kInv255 = asfloat(0x3b808081u);      // 1.0f / 255.0f
static const float kInv245 = asfloat(0x3ed0fac6u);      // 1.0f / 2.45f
static const float k1em4 = asfloat(0x38d1b717u);        // 1e-4f
static const float k1em10 = asfloat(0x2edbe6ffu);       // 1e-10f
static const float k1p1 = asfloat(0x3f8ccccdu);         // 1.1f
static const float k0p05 = asfloat(0x3d4ccccdu);        // 0.05f
static const float kDisplayMax = asfloat(0x401ccccdu);  // 2.45f
static const float k0p01 = asfloat(0x3c23d70au);        // 0.01f
static const float kShapeAB = asfloat(0xbf38a8c1u);     // -0.721325f
static const float kShapeC = asfloat(0xbfb8a8c1u);      // -1.44265f
static const float kRn = 1.00034f;                      // 1 + 2^-11.7: the pack truncates, this makes its mean error ~0
static const float kEncKr = kEncK * kRn, kInv255r = kInv255 * kRn, kShapeABr = kShapeAB * kRn, kShapeCr = kShapeC * kRn;
static const float kLz0 = asfloat(0xbdba3b4bu), kLz1 = asfloat(0x3de71ca9u), kLz2 = asfloat(0x3f23b70fu);
static const float kLz3 = asfloat(0x3ed182b5u), kLz4 = asfloat(0xbd90a4acu);

static float g_e;        // exposure (0 replaced by 1)
static float g_k_hist;   // 0.01 / e_prev * e: the history candidates' scale into the current exposure's linear units
static float g_ipi;      // 1 / pi (_pf)
// The option word is pitch_bytes (unused by the hybrid kernel; the runtime's l_mode=fast levers 16 and 64):
static bool g_secbl;     // bit 0: the weaker history candidate (n = 1, weight <= 1/2) is a 4-tap bilinear of the previous history
                         // instead of the 12-tap Catmull-Rom
static bool g_w2hi;      // bit 2 (off by default): a weaker candidate of weight below 0.05 is dropped (the hybrid route's threshold: 1 / 255)
static bool g_secsame;   // bit 1: a weaker candidate whose motion is within 0.01 render pixel of the first one's is the same sample
                         // (the weights add to 1: the first candidate alone); static and smooth regions skip the second history read

float unif(float x) { return asfloat(WaveReadLaneFirst(asuint(x))); }
uint mul24(uint a, uint b) { return (a & 0xffffffu) * (b & 0xffffffu); }
int mul_i24(int a, int b) { return ((int)((uint)a << 8) >> 8) * ((int)((uint)b << 8) >> 8); }
int clampi(int v, int lo, int hi) { int t = v > hi ? hi : v; return t < lo ? lo : t; }
int mini(int a, int b) { return a < b ? a : b; }
int floor_i(float v) { return (int)floor(v); }
float clampf(float v, float lo, float hi) { return min(max(v, lo), hi); }
half h_bits(uint b) { return asfloat16((uint16_t)b); }
uint pk2(float a, float b) { return f32tof16(a) | (f32tof16(b) << 16); }           // v_cvt_pk_rtz_f16_f32
float rn16(float x) { return asfloat(asuint(x) + 0x1000u); }                          // + half an FP16 step, then the truncating pack

// E(v) = kEncK log2(max(fma(100, v, 1), 1e-4)); the encoded colour of a staged texel: E_in(v) = E(min(v, 32752)), E_in(NaN) = 0
float enc_log(float v) { return log2(max(mad(100.0f, v, 1.0f), k1em4)); }
float enc_in_r(float v) { return isnan(v) ? 0.0f : kEncKr * enc_log(min(v, 32752.0f)); }

// ---- the staged texel -------------------------------------------------------------------------------------------------
struct LRaw { float4 c; uint2 d; uint c01; };
LRaw raw_fetch(int idx)
{
    LRaw r;
    const uint ui = (uint)idx;
    r.c = wg_ld4f(color, ui * 16u);
    r.d = wg_ld64(A, ui * 8u);
    r.c01 = wg_ld32(C, ui * 4u);
    return r;
}
float code_f(uint v, uint i) { return (float)((v >> (8u * i)) & 0xffu) * kInv255r; }
// six dwords of FP16 bits, each one native pack of two values: (f0 f1) (f2 b0) (b1 b2) (a0 a1) (a2 kA) (kB kC)
void raw_put(LRaw r, uint dst, bool store)
{
    const uint a = r.d.x & 0xffffffu;
    const uint b = (r.d.x >> 24) | ((r.d.y & 0xffffu) << 8);
    const uint2 k = uint2(r.c01, r.d.y >> 16);
    float3 c = r.c.xyz;
#if LP_PIDIV
    c *= g_ipi;
#endif
    const float k0 = f16tof32(k.x), k1 = f16tof32(k.x >> 16), k2 = f16tof32(k.y);
    const float e0 = enc_in_r(g_e * c.x), e1 = enc_in_r(g_e * c.y), e2 = enc_in_r(g_e * c.z);
    if (store) {
        sh_tex[dst + 0u] = pk2(e0, e1);
        sh_tex[dst + 1u] = pk2(e2, code_f(b, 0u));
        sh_tex[dst + 2u] = pk2(code_f(b, 1u), code_f(b, 2u));
        sh_tex[dst + 3u] = pk2(code_f(a, 0u), code_f(a, 1u));
        sh_tex[dst + 4u] = pk2(code_f(a, 2u), k0 * kShapeABr);
        sh_tex[dst + 5u] = pk2(k1 * kShapeABr, k2 * kShapeCr);
    }
}

// The footprint of one step: rows r0 ..., ncol columns from c0
float lpx(int xx) { return mad((float)xx + 0.5f, P_rx, P_jx); }
float lpy(int yy) { return mad(P_ry, (float)yy + 0.5f, P_jy); }
int step_rows(int ys, int r_need, out int r0)
{
    const int yl = mini(ys + 2 * (kNW - 1), r_need) & ~1;
    r0 = floor_i(lpy(ys) - 0.5f) - 1;
    return floor_i(lpy(yl) - 0.5f) + 2 - r0 + 1;
}

// ---- the gather: nine FP16 feature sums and the FP16 total weight, per pixel ----------------------------------------------
struct LGather { half F[9]; half W; };
LGather gather_zero()
{
    LGather g;
    [unroll] for (int k = 0; k < 9; ++k) g.F[k] = (half)0.0f;
    g.W = (half)0.0f;
    return g;
}
// one tap: eps = fma(u v, kC, fma(v v, kB, u u kA)) in FP16, w = 2^eps (v_exp_f16); F_k += w f_k, W += w
void tap_acc(inout LGather g, uint t[kTexDw], half w)
{
    g.W += w;
    [unroll] for (int k = 0; k < 9; ++k) {
        const uint d = t[k >> 1];
        g.F[k] = mad(w, h_bits((k & 1) ? (d >> 16) : d), g.F[k]);
    }
}
// one column i of the 4 x 4 gather (taps j = -1 ... 2 for both rows of the pair) from the staged texels
void gather_col(int i, half dx, int tb0, int rstride, half v0[kLWin], half v1[kLWin], half vv0[kLWin], half vv1[kLWin],
                inout LGather g0, inout LGather g1)
{
    const half u = dx + (half)(float)i;
    const half uu = u * u;
    const int col_base = tb0 + (i + 1) * kTexDw;
    [unroll] for (int jj = 0; jj < kLWin; ++jj) {
        const int base = col_base + jj * rstride;
        uint t[kTexDw];
        [unroll] for (int d = 0; d < kTexDw; ++d) t[d] = sh_tex[base + d];
        const half kA = h_bits(t[4] >> 16), kB = h_bits(t[5]), kC = h_bits(t[5] >> 16);
        const half ta = uu * kA;
        const half w0 = exp2(mad(u * v0[jj], kC, mad(vv0[jj], kB, ta)));
        const half w1 = exp2(mad(u * v1[jj], kC, mad(vv1[jj], kB, ta)));
        tap_acc(g0, t, w0);
        tap_acc(g1, t, w1);
    }
}

// ---- history: 12-tap Catmull-Rom of the previous full history (8-byte RGBA16F texels) --------------------------------
float4 cr_weights(float f)   // Catmull-Rom (a = -1/2) in Horner form: 12 operations (the weights sum to 1)
{
    const float f2 = f * f;
    float4 w;
    w.x = f * mad(f, mad(-0.5f, f, 1.0f), -0.5f);
    w.y = mad(f2, mad(1.5f, f, -2.5f), 1.0f);
    w.z = f * mad(f, mad(-1.5f, f, 2.0f), 0.5f);
    w.w = f2 * mad(0.5f, f, -0.5f);
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
    // interior fast path (every lane's 4 x 4 footprint inside the image): six 16-byte span loads from one base
    const bool interior = and(and(P_dw >= 4, P_dh >= 4),
                              and((uint)(ix - 1) <= (uint)(P_dw - 4), (uint)(iy - 1) <= (uint)(P_dh - 4)));
    const float3 zi = float3(0.0f, 0.0f, 0.0f);
    [branch] if (WaveActiveAllTrue(interior)) {
        const uint b0 = hist_prev + (uint)(ix - 1) * 8u + (uint)(iy - 1) * pitch;
        const uint b1 = b0 + pitch, b2 = b0 + 2u * pitch, b3 = b0 + 3u * pitch;
        const uint4 a0 = wg_mem.Load4(b0 + 8u), a10 = wg_mem.Load4(b1), a11 = wg_mem.Load4(b1 + 16u);
        const uint4 a20 = wg_mem.Load4(b2), a21 = wg_mem.Load4(b2 + 16u), a3 = wg_mem.Load4(b3 + 8u);
        const float3 q0 = mad(wx.z, h3(a0.zw), wx.y * h3(a0.xy));
        const float3 q1 = mad(wx.w, h3(a11.zw), mad(wx.z, h3(a11.xy), mad(wx.y, h3(a10.zw), wx.x * h3(a10.xy))));
        const float3 q2 = mad(wx.w, h3(a21.zw), mad(wx.z, h3(a21.xy), mad(wx.y, h3(a20.zw), wx.x * h3(a20.xy))));
        const float3 q3 = mad(wx.z, h3(a3.zw), wx.y * h3(a3.xy));
        const float3 acci = mad(wy.w, q3, mad(wy.z, q2, mad(wy.y, q1, wy.x * q0)));
        const float normi = mad(wx.y + wx.z, wy.x + wy.w, wy.y + wy.z);   // the weights of a row sum to 1
        return acci * rcp(normi);
    }
    uint ro[4], co[4];
    [unroll] for (int k = 0; k < 4; ++k) {
        co[k] = hist_prev + (uint)clampi(ix - 1 + k, 0, P_dw - 1) * 8u;
        ro[k] = (uint)clampi(iy - 1 + k, 0, P_dh - 1) * pitch;
    }
    const float3 r0 = mad(wx.z, h3(wg_mem.Load2(ro[0] + co[2])), wx.y * h3(wg_mem.Load2(ro[0] + co[1])));
    const float3 r1 = mad(wx.w, h3(wg_mem.Load2(ro[1] + co[3])), mad(wx.z, h3(wg_mem.Load2(ro[1] + co[2])),
                      mad(wx.y, h3(wg_mem.Load2(ro[1] + co[1])), wx.x * h3(wg_mem.Load2(ro[1] + co[0])))));
    const float3 r2 = mad(wx.w, h3(wg_mem.Load2(ro[2] + co[3])), mad(wx.z, h3(wg_mem.Load2(ro[2] + co[2])),
                      mad(wx.y, h3(wg_mem.Load2(ro[2] + co[1])), wx.x * h3(wg_mem.Load2(ro[2] + co[0])))));
    const float3 r3 = mad(wx.z, h3(wg_mem.Load2(ro[3] + co[2])), wx.y * h3(wg_mem.Load2(ro[3] + co[1])));
    const float3 acc = mad(wy.w, r3, mad(wy.z, r2, mad(wy.y, r1, wy.x * r0)));
    const float norm = mad(wx.y + wx.z, wy.x + wy.w, wy.y + wy.z);
    return acc * rcp(norm);
}

// 4-tap bilinear of the previous history (the weaker candidate under LP_SEC_BILINEAR): clamped texel coordinates
float3 history_bilinear(float qx, float qy)
{
    const float tx = qx - 0.5f, ty = qy - 0.5f;
    const float fx0 = floor(tx), fy0 = floor(ty);
    const float fx = tx - fx0, fy = ty - fy0;
    const int ix = (int)fx0, iy = (int)fy0;
    const uint pitch = (uint)P_dw * 8u;
    const uint ca = hist_prev + (uint)clampi(ix, 0, P_dw - 1) * 8u, cb = hist_prev + (uint)clampi(ix + 1, 0, P_dw - 1) * 8u;
    const uint ra = (uint)clampi(iy, 0, P_dh - 1) * pitch, rb = (uint)clampi(iy + 1, 0, P_dh - 1) * pitch;
    const float3 t0 = mad(fx, h3(wg_mem.Load2(ra + cb)) - h3(wg_mem.Load2(ra + ca)), h3(wg_mem.Load2(ra + ca)));
    const float3 t1 = mad(fx, h3(wg_mem.Load2(rb + cb)) - h3(wg_mem.Load2(rb + ca)), h3(wg_mem.Load2(rb + ca)));
    return mad(fy, t1 - t0, t0);
}

// D(h) in the current exposure's linear units: (exp2(clamp(h, 1e-4, 3) kDecK) - 1) 0.01 e / e_prev (the 12-bit polynomial
// branch of the hybrid route is not needed: no wave vote, one exponential per value)
float dec_hist(float h) { return (exp2(clampf(h, k1em4, 3.0f) * kDecK) - 1.0f) * g_k_hist; }
float dec_cur(float h) { return (exp2(clampf(h, k1em4, 3.0f) * kDecK) - 1.0f) * k0p01; }

// ---- steps 3 to 5 of one D pixel (l_select + l_post_core) -----------------------------------------------------------
void l_post_core(LGather g, int x, int y, int fx, int fy, float mx[3], float my[3], float s, out float hh[3], out float oh[3])
{
    const float inv = rcp(max((float)g.W, asfloat(0x33800000u)));   // >= 2^-24: a pixel whose weights all flushed has zero features, not NaN
    const float sig0 = inv * (float)g.F[3], sig1 = inv * (float)g.F[4], sig2 = inv * (float)g.F[5];
    const bool sb1 = sig1 > sig0;
    const float best = sb1 ? sig1 : sig0;
    const bool sb2 = sig2 > best;
    const float s1 = sb2 ? sig2 : best;
    const float m1x = sb2 ? mx[2] : (sb1 ? mx[1] : mx[0]), m1y = sb2 ? my[2] : (sb1 ? my[1] : my[0]);
    const bool sf0 = !or(sb1, sb2);
    const float so1 = sf0 ? sig1 : sig0, so2 = sb2 ? sig1 : sig2;
    const float mo1x = sf0 ? mx[1] : mx[0], mo1y = sf0 ? my[1] : my[0];
    const float mo2x = sb2 ? mx[1] : mx[2], mo2y = sb2 ? my[1] : my[2];
    const bool scc = so2 > so1;
    const float s2 = scc ? so2 : so1;
    const float m2x = scc ? mo2x : mo1x, m2y = scc ? mo2y : mo1y;
    const float q1 = clampf(mad(k1p1, s1, -k0p05), 0.0f, 1.0f);
    const float q2 = clampf(mad(k1p1, s2, -k0p05), 0.0f, 1.0f);
    const float rden = rcp(max(q1 + q2, k1em10));
    float w2 = q2 * rden, w1;
    if (w2 < (g_w2hi ? k0p05 : kInv255)) {
        w2 = 0.0f;
        w1 = 1.0f;
    } else {
        w1 = q1 * rden;
    }
    if (g_secsame && abs(m2x - m1x) + abs(m2y - m1y) < 0.01f) {   // the same sample twice: the first candidate alone
        w2 = 0.0f;
        w1 = 1.0f;
    }
    const float X = (float)x + 0.5f, Y = (float)y + 0.5f;
    const int ncand = w2 != 0.0f ? 2 : 1;
    float L[2][3];
    [unroll] for (int n = 0; n < 2; ++n) {
        L[n][0] = 0.0f; L[n][1] = 0.0f; L[n][2] = 0.0f;
        if (n < ncand) {
            const float qx = X + (n == 0 ? m1x : m2x);
            const float qy = Y + (n == 0 ? m1y : m2y);
            const bool valid = and(and(and(and(P_reset == 0, qx >= 0.0f), qx <= (float)P_dw), qy >= 0.0f), qy <= (float)P_dh);
            if (valid) {
                const float3 h = (n == 1 && g_secbl) ? history_bilinear(qx, qy) : history_cr(qx, qy);
                L[n][0] = dec_hist(h.x);
                L[n][1] = dec_hist(h.y);
                L[n][2] = dec_hist(h.z);
            } else {
                float4 c = wg_ld4f(color, (uint)(fy * P_rw + fx) * 16u);
#if LP_PIDIV
                c.xyz *= g_ipi;
#endif
                const float ge = g_e;   // L is in the current exposure's units already: the raw colour times e
                L[n][0] = c.x * ge; L[n][1] = c.y * ge; L[n][2] = c.z * ge;
            }
        }
    }
    float Hl[3], Cl[3];
    [unroll] for (int i = 0; i < 3; ++i) {
        Hl[i] = ncand == 2 ? mad(w1, L[0][i], w2 * L[1][i]) : w1 * L[0][i];
        Cl[i] = dec_cur(inv * (float)g.F[i]);
    }
    const float f7 = inv * (float)g.F[7], f8 = inv * (float)g.F[8];
    const float alpha = clampf(s * (float)g.F[6], 0.0f, 1.0f);   // s W (F6 / W) = s F6: the total weight cancels
    const float bh2 = mad(alpha, 1.0f - f7, f7);
    const float bo2 = mad(alpha, 1.0f - f8, f8);
    [unroll] for (int i = 0; i < 3; ++i) {
        hh[i] = mad(Cl[i] - Hl[i], bh2, Hl[i]);
        oh[i] = mad(Cl[i] - Hl[i], bo2, Hl[i]);
    }
    const float sh = hh[0] + hh[1] + hh[2], so = oh[0] + oh[1] + oh[2];
    if (isnan(sh)) {
        [unroll] for (int i = 0; i < 3; ++i) hh[i] = Cl[i];
    }
    if (isnan(so)) {
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

// ---- the output filter's FP32 tap sums (post_filter_c1_s4's lz5a: v = w0 t0, fused multiply-adds, clamp to the taps' range)
float lz5a(float t0, float t1, float t2, float t3, float t4, float w0, float w1, float w3, float w4)
{
    float v = w0 * t0;
    v = mad(w1, t1, v);
    v = mad(kLz2, t2, v);
    v = mad(w3, t3, v);
    v = mad(w4, t4, v);
    const float mid_lo = min(min(t1, t2), t3), mid_hi = max(max(t1, t2), t3);
    const float lo = min(min(mid_lo, t0), t4), hi = max(max(mid_hi, t0), t4);
    return min(max(v, lo), hi);   // med3 (the taps are finite, lo <= hi)
}
float lz5a_even(float t[10], int o) { return lz5a(t[o], t[o + 1], t[o + 2], t[o + 3], t[o + 4], kLz0, kLz1, kLz3, kLz4); }
float lz5a_odd(float t[10], int o) { return lz5a(t[o], t[o + 1], t[o + 2], t[o + 3], t[o + 4], kLz4, kLz3, kLz1, kLz0); }

WG_KERNEL(l_post_tf_p12_fast)
{
    WG_SETUP();
    if (P_dw < 4) return;
    g_e = P_e;
    float g_ep = P_e_prev;
    float dec_scale = dec_scale_arg;
    if (expo != 0u) {   // the diet pair: [e, e_prev, last e, ratio, dec_scale]
        const float e0 = unif(wg_ldf(expo, 0u)), e1 = unif(wg_ldf(expo, 4u));
        g_e = e0 == 0.0f ? 1.0f : e0;
        g_ep = e1 == 0.0f ? 1.0f : e1;
        dec_scale = unif(wg_ldf(expo, 16u));
    }
    g_k_hist = unif(k0p01 * g_e * rcp(g_ep));
    g_secbl = (pitch_bytes & 1) != 0;
    g_secsame = (pitch_bytes & 2) != 0;
    g_w2hi = (pitch_bytes & 4) != 0;
#if LP_PIDIV
    g_ipi = unif(rcp(asfloat((uint)P_pad0)));
#endif
    const float out_scale = unif(dec_scale * pi_arg);   // dec_scale; with pi != 1 the runtime's scale_rgb folded in
    const int lane = (int)wg_lane_id();
    const int wave = wg_uniform(lane >> 6), wl = lane & 63;   // wave index: uniform (a scalar register)
    const int strips = (ow + kTfOutX - 1) / kTfOutX;
    const int bid = (int)wg_block_id();
    // workgroup bid -> (strip, chunk) as post_tf: the strips of chunk c rotated by c (the same set of workgroups)
    const int chunk = wg_uniform(bid / strips);
    const int strip = wg_uniform((bid - chunk * strips + chunk) % strips);
    const int X0 = strip * kTfOwn;
    const int csize = chunk < k1 ? cpairs : c2;
    const int P0 = chunk < k1 ? chunk * cpairs : k1 * cpairs + (chunk - k1) * c2;
    const int p_end = wg_uniform(mini(P0 + csize, (oh + 1) / 2));
    const int own_r0 = wg_uniform(3 * P0), own_r1 = wg_uniform(mini(3 * (P0 + csize), P_dh));
    const int r_first = wg_uniform(3 * P0 - 2 < 0 ? 0 : 3 * P0 - 2);
    const int r_need = wg_uniform(mini(3 * (p_end - 1) + 4, P_dh - 1));
    const float s = wg_ldf(w_p, 0u);

    // the strip's render columns c0 ... c1 (taps ax - 1 ... ax + 2 of its first and last D column)
    const int c0 = floor_i(lpx(clampi(X0 - 2, 0, P_dw - 1)) - 0.5f) - 1;
    const int c1 = floor_i(lpx(clampi(X0 + kTfCols - 3, 0, P_dw - 1)) - 0.5f) + (kLWin - 2);
    const int ncol = c1 - c0 + 1;
    const float inv_ncol = unif(rcp((float)ncol));

    // the staging of step y_step: footprint rows r0 ..., texels loaded one step ahead
    int r0, nstage;
    {
        const int nrow = step_rows(r_first, r_need, r0);
        nstage = ncol * nrow < kStageCap ? ncol * nrow : kStageCap;
    }
    LRaw raw;
    {
        const int i = mini(lane, nstage - 1);
        const int r = (int)(((float)i + 0.5f) * inv_ncol), c = i - mul_i24(r, ncol);
        raw = raw_fetch(clampi(r0 + r, 0, P_rh - 1) * P_rw + clampi(c0 + c, 0, P_rw - 1));
    }

    // this lane's D column (fixed over the walk)
    const int xv = X0 - 2 + wl;
    const int x = clampi(xv, 0, P_dw - 1);
    const bool own_x = and(and(xv >= X0, xv < X0 + kTfOwn), xv < P_dw);
    const float px = lpx(x);
    const int ax = floor_i(px - 0.5f);
    const uint wlu = (uint)wl;

    int base_slot = 0;
    int p_next = P0;
    [loop] for (int y_step = r_first; y_step <= r_need; y_step += 2 * kNW) {
        // 1. stage the footprint
        [branch] if (wave * 64 < nstage) {   // wave-uniform: a wave wholly past the staged range has nothing to store
            raw_put(raw, (uint)lane * kTexDw, lane < nstage);
        }
        wg_barrier_lds();
        const int r0c = r0;

        // 2. reconstruction of the wave's row pair (yv, yv + 1)
        const int yv = y_step + 2 * wave;
        const bool wact = yv <= r_need;
        int slot = base_slot + 2 * wave;
        if (slot >= kRing) slot -= kRing;
        const uint scr = (uint)slot * kRowDw;   // dword of ring row slot (rows slot, slot + 1)
        if (wact) {
            const int y0 = yv;
            const bool has1 = y0 + 1 < P_dh;
            const float py0 = unif(lpy(y0)), py1 = unif(lpy(y0 + 1));   // wave-uniform (y0 is)
            const int b = wg_uniform(floor_i(py0 - 0.5f));
            const int b1 = wg_uniform(floor_i(py1 - 0.5f));
            // gather offsets: dx = h((ax + 1/2) - px), d_y = h((b_y + 1/2) - py(y)) minus 1 (FP16) when the row's own b_y
            // differs from the pair's b
            const half dx = (half)rn16(((float)ax + 0.5f) - px), dy0 = (half)rn16(((float)b + 0.5f) - py0);
            half dy1 = (half)rn16(((float)b1 + 0.5f) - py1);
            if (b1 != b) dy1 = dy1 - (half)1.0f;
            half v0[kLWin], v1[kLWin], vv0[kLWin], vv1[kLWin];
            [unroll] for (int jj = 0; jj < kLWin; ++jj) {
                v0[jj] = dy0 + (half)(float)(jj - 1);
                v1[jj] = dy1 + (half)(float)(jj - 1);
                vv0[jj] = v0[jj] * v0[jj];
                vv1[jj] = v1[jj] * v1[jj];
            }
            // taps column by column (i outer, j inner), both rows' gathers in one pass over the staged texels
            LGather g0 = gather_zero(), g1 = gather_zero();
            const int tb0 = ((b - 1 - r0c) * ncol + (ax - 1 - c0)) * kTexDw, rstride = ncol * kTexDw;
            [loop] for (int i = -1; i <= (kLWin - 2); ++i)
                gather_col(i, dx, tb0, rstride, v0, v1, vv0, vv1, g0, g1);

            float r[3] = {0.0f, 0.0f, 0.0f};
            [loop] for (int row = 0; row < 2; ++row) {
                const bool second = row == 1;
                if (second && !has1) break;   // wave-uniform: the missing second row is not stored and adds nothing
                const int y = y0 + row;
                LGather g = g0;
                if (second) g = g1;
                float hh[3], oo[3];
                const MotRaw mot = motion_load(px, second ? py1 : py0);
                finish_pixel(g, x, y, mot, s, hh, oo);
                float el[3], dv[3];
                [unroll] for (int ci = 0; ci < 3; ++ci) {
                    el[ci] = kEncKr * enc_log(hh[ci]);
                    dv[ci] = clampf(kEncKr * enc_log(oo[ci]), 0.0f, kDisplayMax);
                }
                const uint hx = pk2(el[0], el[1]), hy = pk2(el[2], dv[0]), hz = pk2(dv[1], dv[2]);
                if (and(and(own_x, y >= own_r0), y < own_r1))
                    wg_st64(hist_out, (uint)(y * P_dw + x) * 8u, uint2(hx, hy & 0xffffu));
                // display values (FP16) of the 64 columns -> scratch row slot + row: lanes 2m, 2m + 1 share a dword
                const uint srow = scr + (uint)row * kRowDw;
                [unroll] for (int c = 0; c < 3; ++c) {
                    const uint dvb = c == 0 ? hy >> 16 : (c == 1 ? hz & 0xffffu : hz >> 16);
                    const uint nbv = QuadReadAcrossX(dvb);
                    if ((wlu & 1u) == 0u) lds[srow + (uint)c * 32u + (wlu >> 1)] = dvb | (nbv << 16);
                }
                // reduced history: mean of H^ over the 2 x 2 block, r = h0 + h1
                [unroll] for (int ri = 0; ri < 3; ++ri) r[ri] = second ? r[ri] + hh[ri] : hh[ri];
            }
            float o[3];
            [unroll] for (int i = 0; i < 3; ++i) o[i] = QuadReadAcrossX(r[i]);   // lane ^ 1
            if (and(and(and(and(and((wl & 1) == 0, own_x), x + 1 < P_dw), has1), y0 >= own_r0), y0 + 1 < own_r1)) {
                uint code = 0u;
                [unroll] for (int i = 0; i < 3; ++i) {
                    const float m = (r[i] + o[i]) * 0.25f;
                    const float v = clampf(kEncK * enc_log(m) * kInv245, 0.0f, 1.0f);
                    code |= (uint)mad(v, 1023.0f, 0.5f) << (10 * i);
                }
                wg_st32(reduced, (uint)((y0 / 2) * (P_dw / 2) + x / 2) * 4u, code);
            }
        }
        // the next step's staging loads (every lane loads a valid clamped texel)
        {
            const int nrow = step_rows(y_step + 2 * kNW, r_need, r0);
            nstage = ncol * nrow < kStageCap ? ncol * nrow : kStageCap;
            const int i = mini(lane, nstage - 1 < 0 ? 0 : nstage - 1);
            const int r = (int)(((float)i + 0.5f) * inv_ncol), c = i - mul_i24(r, ncol);
            raw = raw_fetch(clampi(r0 + r, 0, P_rh - 1) * P_rw + clampi(c0 + c, 0, P_rw - 1));
        }

        // horizontal pass of the wave's two rows: lane (row hk, channel hc, cell hg), output columns 4 hg ... 4 hg + 3
        GroupMemoryBarrier();   // the 64 lanes are one wave: its LDS operations execute in order (the fence orders the compiler)
        const int hl = wl;
        const bool hact = and(wact, hl < 60);
        const int hk = hl >= 30 ? 1 : 0, hrem = hl - 30 * hk, hc = hrem >= 20 ? 2 : (hrem >= 10 ? 1 : 0),
                  hg = hrem - 10 * hc;
        uint hv0 = 0u, hv1 = 0u;   // the four filtered columns as two FP16 pairs
        if (hact) {
            const uint src = scr + (uint)hk * kRowDw + (uint)hc * 32u + (uint)hg * 3u;
            float t[10];   // the FP16 display taps as FP32 values (exact)
            [unroll] for (int i = 0; i < 5; ++i) {
                const uint q = lds[src + (uint)i];
                t[2 * i] = f16tof32(q);
                t[2 * i + 1] = f16tof32(q >> 16);
            }
            hv0 = pk2(rn16(lz5a_even(t, 0)), rn16(lz5a_odd(t, 2)));
            hv1 = pk2(rn16(lz5a_even(t, 3)), rn16(lz5a_odd(t, 5)));
        }
        GroupMemoryBarrier();
        if (hact) {
            const uint dst = scr + (uint)hk * kRowDw + (uint)hc * (uint)kPairs + (uint)hg * 2u;
            lds[dst] = hv0;
            lds[dst + 1u] = hv1;
            if (slot + hk < 6) {   // the first six ring rows are mirrored after the ring (wrap-free vertical taps)
                const uint dm = dst + (uint)kRing * kRowDw;
                lds[dm] = hv0;
                lds[dm + 1u] = hv1;
            }
        }
        wg_barrier_lds();

        // 3. vertical pass, Dec and output of the row pairs whose input rows are complete; one item = two adjacent columns
        const int r_done = mini(y_step + 2 * kNW, r_need + 1);
        const int p_rdy = r_done > r_need ? p_end : mini(p_end, (r_done - 5) / 3 + 1);
        const int nitems = (p_rdy - p_next) * 2 * kPairs;
        const bool interior = and(3 * p_next - 2 >= 0, 3 * p_rdy + 1 <= P_dh - 1);
        const int s0 = interior ? (3 * p_next - 2 - r_first) % kRing : 0;
        [loop] for (int it = lane; it < nitems; it += kThreads) {
            const int pr = (int)(mul24((uint)it, 52429u) >> 21);   // it / 40
            const int rem = it - (int)mul24((uint)pr, 2u * kPairs);
            const int dy = rem >= kPairs ? 1 : 0, m = rem - dy * kPairs;
            const int pp = p_next + pr, y = 2 * pp + dy, xg = strip * kTfOutX + 2 * m;
            if (or(y >= oh, xg >= ow)) continue;
            uint t[5][3];
            if (interior) {   // one ring address, the 15 taps at constant offsets (mirrored rows: no wrap)
                int u = s0 + 3 * pr + 2 * dy;
                u = u >= kRing ? u - kRing : u;
                u = u >= kRing ? u - kRing : u;
                const uint rp = mul24((uint)u, (uint)kRowDw) + (uint)m;
                [unroll] for (int k = 0; k < 5; ++k)
                    [unroll] for (int c = 0; c < 3; ++c) t[k][c] = lds[rp + (uint)(k * kRowDw + c * kPairs)];
            } else {
                const int xr = 3 * pp - 2 + 2 * dy;
                const int row0 = clampi(xr, 0, P_dh - 1);
                const int sl0 = (int)((uint)(row0 - r_first) % (uint)kRing);
                [unroll] for (int k = 0; k < 5; ++k) {
                    const int st = clampi(xr + k, 0, P_dh - 1) - row0;
                    const int sl = sl0 + st >= kRing ? sl0 + st - kRing : sl0 + st;
                    [unroll] for (int c = 0; c < 3; ++c) t[k][c] = lds[(uint)sl * kRowDw + (uint)(c * kPairs + m)];
                }
            }
            // emit: weights of the lane's phase by selects, lz5a, Dec, store 3 x f32 per column
            const bool ev = dy == 0;
            const float w0 = ev ? kLz0 : kLz4, w4 = ev ? kLz4 : kLz0, w1 = ev ? kLz1 : kLz3, w3 = ev ? kLz3 : kLz1;
            const uint o = mul24((uint)y, (uint)ow * 12u) + mul24((uint)xg, 12u);
            [unroll] for (int cs = 0; cs < 2; ++cs) {
                if (cs == 1 && xg + 1 >= ow) break;
                float vo[3];
                [unroll] for (int c = 0; c < 3; ++c) {
                    const float q = clampf(lz5a(f16tof32(t[0][c] >> (16 * cs)), f16tof32(t[1][c] >> (16 * cs)),
                                                f16tof32(t[2][c] >> (16 * cs)), f16tof32(t[3][c] >> (16 * cs)),
                                                f16tof32(t[4][c] >> (16 * cs)), w0, w1, w3, w4), k1em4, 3.0f);
                    vo[c] = mad(exp2(q * kDecK), out_scale, -out_scale);   // (2^(q kDecK) - 1) out_scale
                }
                wg_mem.Store3(out_p + o + (uint)cs * 12u, uint3(asuint(vo[0]), asuint(vo[1]), asuint(vo[2])));
            }
        }
        p_next = p_rdy;
        base_slot += 2 * kNW;
        if (base_slot >= kRing) base_slot -= kRing;
    }
}
