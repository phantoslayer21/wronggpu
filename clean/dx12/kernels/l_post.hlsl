// SPDX-License-Identifier: MIT
// D3D12 port of Model L's reconstruction kernel (issue #89): l_post_recon of l_post.co (clean/l/kernels/l_post.hip, HDR),
// byte for byte. Arguments, buffers, layouts and grid are Model M's post_recon (clean/kernels/post.hip): grid
// ceil(dw / 32) * ceil(ceil(dh / 2) / 8) workgroups of 256 lanes; lane (l, k) of wave k handles D column x0 + l and the
// row pair k0 + k. w = the post blob (w[0] = s). Model L's changes against post_recon: the 4 x 4 colour gather in the
// model's precision (FP16 features, offsets, exponent and accumulators), linear-colour history candidates
// L = D(h) / e_prev combined as fma(w1, L1, w2 L2) e, and the render pixel's colour as the fallback.
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
// - D(h) / e_prev is an IEEE division (wg_div).
// Model M's DXILs are untouched: this file only adds the kernel (bound to l_post.co).
// l_post_recon_p12 (LP_P12 = 1): the same kernel on the pack12 coefficients of l_coef_pack12: A = image A8 (8 bytes per
// render pixel: codes A0 A1 A2 B0 | B1 B2, C2 FP16), C = image C4 (4 bytes per render pixel: C0, C1 FP16), B unused.
//
// @wg-kernel l_post_recon co=l_post.co block=256 wave=32 args=p,p,p,p,p,p,p,p,p,p,p,p,b64,p
// @wg-kernel l_post_recon_p12 co=l_post.co entry=l_post_recon_p12 block=256 wave=32 args=p,p,p,p,p,p,p,p,p,p,p,p,b64,p defines=LP_P12=1
#include "wg_platform.hlsli"
#ifndef LP_P12
#define LP_P12 0
#endif

WG_ARGS_BEGIN
    uint color, A, B, C, hist_prev, m_near, m_center, m_far, w_p, hist_out, reduced, display;   // pointers
    int P_rw, P_rh, P_dw, P_dh, P_reset, P_pad0, P_pad1, P_pad2;                                  // WgPostParams
    float P_rx, P_ry, P_jx, P_jy, P_e, P_e_prev, P_ratio, P_pad3;
    uint expo;
WG_ARGS_END

#define kThreads 256
#define kTileX 32
#define kTilePairs 8
#define kStageCap 512
#define kLWin 4
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
float opq(float x) { return asfloat(asuint(x) ^ g_z); }
half opqh(half x) { return asfloat16((uint16_t)(asuint16(x) ^ (uint16_t)g_z)); }
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
    const float v = qc < kDecPolyMax ? dec_m1_poly(qc) : exp2(qc * kDecK) - 1.0f;
    return isnan(q) ? q : opq(v) * k0p01;
}

float byte_f(uint v, uint i) { return (float)((v >> (8u * i)) & 0xffu); }

// l_load_texel: the nine FP16 features and the FP16 kernel shape of render pixel idx, as six dwords of FP16 bits
void load_texel(int idx, out uint t[kTexDw])
{
    const uint ui = (uint)idx;
    const float4 c = wg_ld4f(color, ui * 16u);
#if LP_P12
    const uint2 d = wg_ld64(A, ui * 8u);
    const uint c01 = wg_ld32(C, ui * 4u);
    const uint a = d.x & 0xffffffu, b = (d.x >> 24) | ((d.y & 0xffffu) << 8);
    const uint2 k = uint2(c01, d.y >> 16);   // C0 | C1 << 16, C2 (l_coef's C dword 0 and the low half of dword 1)
#else
    const uint b = wg_ld32(B, ui * 4u), a = wg_ld32(A, ui * 4u);
    const uint2 k = wg_ld64(C, ui * 8u);
#endif
    uint f[9];
    f[0] = wg_f2h(enc_in(opq(g_e * c.x)));
    f[1] = wg_f2h(enc_in(opq(g_e * c.y)));
    f[2] = wg_f2h(enc_in(opq(g_e * c.z)));
    [unroll] for (uint i = 0; i < 3; ++i) {
        f[3 + i] = wg_f2h(wg_div(byte_f(b, i), 255.0f));
        f[6 + i] = wg_f2h(wg_div(byte_f(a, i), 255.0f));
    }
    const uint kA = wg_mul_f16(f16tof32(k.x), kShapeAB), kB = wg_mul_f16(f16tof32(k.x >> 16), kShapeAB);
    const uint kC = wg_mul_f16(f16tof32(k.y), kShapeC);
    t[0] = f[0] | (f[1] << 16);
    t[1] = f[2] | (f[3] << 16);
    t[2] = f[4] | (f[5] << 16);
    t[3] = f[6] | (f[7] << 16);
    t[4] = f[8] | (kA << 16);
    t[5] = kB | (kC << 16);
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
// F_k = fma(h(w), f_k, F_k). uu, vv, uv come in already rounded (and opaque).
void gather_add(inout LGather g, uint t[kTexDw], half uu, half vv, half uv)
{
    const half kA = h_bits(t[4] >> 16), kB = h_bits(t[5]), kC = h_bits(t[5] >> 16);
    const half ta = opqh(uu * kA), tb = opqh(vv * kB);
    const half s = tb + ta;
    const half eps = mad(uv, kC, s);
    float w = exp2((float)opqh(eps));
    w = w >= kMinNormal ? w : 0.0f;
    g.W = g.W + w;
    const half hw = h_bits(wg_f2h(w));
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
    int first = 0;
    float best = sig0;
    if (sig1 > best) { first = 1; best = sig1; }
    if (sig2 > best) first = 2;
    const int o1 = first == 0 ? 1 : 0, o2 = first == 2 ? 1 : 2;
    const float so1 = o1 == 1 ? sig1 : sig0, so2 = o2 == 1 ? sig1 : sig2;
    const int second = so2 > so1 ? o2 : o1;
    const float s1 = first == 0 ? sig0 : (first == 1 ? sig1 : sig2);
    const float s2 = second == 0 ? sig0 : (second == 1 ? sig1 : sig2);
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
            const int k = n == 0 ? first : second;
            const float qx = X + (k == 0 ? mx[0] : (k == 1 ? mx[1] : mx[2]));
            const float qy = Y + (k == 0 ? my[0] : (k == 1 ? my[1] : my[2]));
            const bool valid = P_reset == 0 && qx >= 0.0f && qx <= (float)P_dw && qy >= 0.0f && qy <= (float)P_dh;
            if (valid) {
                const float3 h = history_cr(qx, qy);
                L[n][0] = wg_div(l_dec(h.x), g_ep);
                L[n][1] = wg_div(l_dec(h.y), g_ep);
                L[n][2] = wg_div(l_dec(h.z), g_ep);
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
        Hl[i] = opq(comb) * g_e;
        hl_inf = hl_inf || abs(Hl[i]) == asfloat(0x7f800000u);
        Cl[i] = l_dec(fbar[i]);
    }
    const float alpha = clampf(opq(s * g.W) * fbar[6], 0.0f, 1.0f);
    const float bh2 = mad(alpha, 1.0f - fbar[7], fbar[7]);
    const float bo2 = mad(alpha, 1.0f - fbar[8], fbar[8]);
    [unroll] for (int i = 0; i < 3; ++i) {
        hh[i] = mad(Cl[i] - Hl[i], bh2, Hl[i]);
        oh[i] = mad(Cl[i] - Hl[i], bo2, Hl[i]);
    }
    const float sh = hh[0] + hh[1] + hh[2], so = oh[0] + oh[1] + oh[2];
    if (isnan(sh) || hl_inf) {
        [unroll] for (int i = 0; i < 3; ++i) hh[i] = Cl[i];
    }
    if (isnan(so) || hl_inf) {
        [unroll] for (int i = 0; i < 3; ++i) oh[i] = Cl[i];
    }
}

// l_finish_pixel: motions, steps 3 to 5, the full history and display stores (when store); returns H^ in hh
void finish_pixel(LGather g, int x, int y, float px, float py, bool store, float s, out float hh[3])
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
    float oh[3];
    l_post_core(g, x, y, fx, fy, mx, my, s, hh, oh);
    if (store) {
        const uint h0 = wg_mul_f16(kEncK, enc_log(hh[0])), h1 = wg_mul_f16(kEncK, enc_log(hh[1]));
        const uint h2 = wg_mul_f16(kEncK, enc_log(hh[2]));
        wg_st64(hist_out, (uint)(y * P_dw + x) * 8u, uint2(h0 | (h1 << 16), h2));
        const int pitch = (P_dw + 63) & ~63, plane = pitch * P_dh, di = y * pitch + x;
        wg_st16(display, (uint)di * 2u, wg_f2h(clampf(enc_e(oh[0]), 0.0f, kDisplayMax)));
        wg_st16(display, (uint)(plane + di) * 2u, wg_f2h(clampf(enc_e(oh[1]), 0.0f, kDisplayMax)));
        wg_st16(display, (uint)(2 * plane + di) * 2u, wg_f2h(clampf(enc_e(oh[2]), 0.0f, kDisplayMax)));
    }
}

float lpx(int xx) { return mad(opq((float)xx + 0.5f), P_rx, P_jx); }   // fused(x + 1/2, rx, jx)
float lpy(int yy) { return mad(P_ry, opq((float)yy + 0.5f), P_jy); }   // fused(ry, y + 1/2, jy)

#if LP_P12
WG_KERNEL(l_post_recon_p12)
#else
WG_KERNEL(l_post_recon)
#endif
{
    WG_SETUP();
    g_z = wg_grid_ >> 31;
    g_e = P_e;
    g_ep = P_e_prev;
    if (expo != 0u) {
        const float e0 = wg_ldf(expo, 0u), e1 = wg_ldf(expo, 4u);
        g_e = e0 == 0.0f ? 1.0f : e0;
        g_ep = e1 == 0.0f ? 1.0f : e1;
    }
    const int lane = (int)wg_lane_id();
    const int tiles_x = (P_dw + kTileX - 1) / kTileX;
    const int tile = (int)wg_block_id();
    const int x0 = (tile % tiles_x) * kTileX;
    const int k0 = (tile / tiles_x) * kTilePairs;
    const int npairs = (P_dh + 1) / 2;
    const float s = wg_ldf(w_p, 0u);

    // render footprint of the tile: window columns ax - 1 ... ax + 2 of its first and last column, window rows of its
    // first and last row pair
    const int xl = mini(x0 + kTileX - 1, P_dw - 1), kl = mini(k0 + kTilePairs - 1, npairs - 1);
    const int c0 = floor_i(lpx(x0) - 0.5f) - 1, c1 = floor_i(lpx(xl) - 0.5f) + (kLWin - 2);
    const int r0 = floor_i(lpy(2 * k0) - 0.5f) - 1, r1 = floor_i(lpy(2 * kl) - 0.5f) + (kLWin - 2);
    const int ncol = c1 - c0 + 1, nrow = r1 - r0 + 1;
    const bool staged = ncol > 0 && nrow > 0 && ncol <= kStageCap && nrow <= kStageCap && ncol * nrow <= kStageCap;
    if (staged) {
        for (int i = lane; i < ncol * nrow; i += kThreads) {
            const int r = i / ncol, c = i - r * ncol;
            const int idx = clampi(r0 + r, 0, P_rh - 1) * P_rw + clampi(c0 + c, 0, P_rw - 1);
            uint t[kTexDw];
            load_texel(idx, t);
            [unroll] for (int d = 0; d < kTexDw; ++d) sh_tex[i * kTexDw + d] = t[d];
        }
    }
    GroupMemoryBarrierWithGroupSync();

    const bool active = x0 + (lane & 31) < P_dw && k0 + (lane >> 5) < npairs;
    const int x = mini(x0 + (lane & 31), P_dw - 1);
    const int y0 = 2 * mini(k0 + (lane >> 5), npairs - 1);
    const bool has1 = y0 + 1 < P_dh;
    const float px = lpx(x), py0 = lpy(y0), py1 = lpy(y0 + 1);
    const int ax = floor_i(px - 0.5f), b = floor_i(py0 - 0.5f);

    // gather offsets: dx = h((ax + 1/2) - px), d_y = h((b_y + 1/2) - py(y)) minus 1 (FP16) when the row's own b_y differs
    // from the pair's b; u = h(dx + i), v = h(d_y + j); taps i = -1 ... 2 (outer), j = -1 ... 2 (inner)
    const half dx = h_bits(wg_f2h(((float)ax + 0.5f) - px));
    const int b1 = floor_i(py1 - 0.5f);
    const half dy0 = h_bits(wg_f2h(((float)b + 0.5f) - py0));
    half dy1 = h_bits(wg_f2h(((float)b1 + 0.5f) - py1));
    if (b1 != b) dy1 = dy1 - (half)1.0f;
    half v0[kLWin], v1[kLWin], vv0[kLWin], vv1[kLWin];
    [unroll] for (int jj = 0; jj < kLWin; ++jj) {
        v0[jj] = dy0 + (half)(float)(jj - 1);
        v1[jj] = dy1 + (half)(float)(jj - 1);
        vv0[jj] = opqh(v0[jj] * v0[jj]);
        vv1[jj] = opqh(v1[jj] * v1[jj]);
    }
    LGather g0 = gather_zero(), g1 = gather_zero();
    [unroll] for (int i = -1; i <= kLWin - 2; ++i) {
        const half u = dx + (half)(float)i;
        const half uu = opqh(u * u);
        [unroll] for (int j = -1; j <= kLWin - 2; ++j) {
            uint t[kTexDw];
            if (staged) {
                const int base = ((b + j - r0) * ncol + (ax + i - c0)) * kTexDw;
                [unroll] for (int d = 0; d < kTexDw; ++d) t[d] = sh_tex[base + d];
            } else {
                load_texel(clampi(b + j, 0, P_rh - 1) * P_rw + clampi(ax + i, 0, P_rw - 1), t);
            }
            gather_add(g0, t, uu, vv0[j + 1], opqh(u * v0[j + 1]));
            gather_add(g1, t, uu, vv1[j + 1], opqh(u * v1[j + 1]));
        }
    }

    float h0[3], h1[3] = {0.0f, 0.0f, 0.0f};
    finish_pixel(g0, x, y0, px, py0, active, s, h0);
    if (has1) finish_pixel(g1, x, y0 + 1, px, py1, active, s, h1);

    // reduced history: mean of H^ over the 2 x 2 block (x, x + 1) x (y0, y0 + 1); lane x + 1 is lane ^ 1 (same wave)
    float r[3];
    [unroll] for (int i = 0; i < 3; ++i) r[i] = h0[i] + h1[i];
    float o[3];
    [unroll] for (int i = 0; i < 3; ++i) o[i] = wg_xor(r[i], 1u);
    if (active && has1 && (x & 1) == 0 && x + 1 < P_dw) {
        uint code = 0u;
        [unroll] for (int i = 0; i < 3; ++i) {
            const float m = opq((r[i] + o[i]) * 0.25f);
            const float v = clampf(opq(enc_e(m)) * kInv245, 0.0f, 1.0f);
            code |= wg_f2u(round(v * 1023.0f)) << (10 * i);
        }
        wg_st32(reduced, (uint)((y0 / 2) * (P_dw / 2) + x / 2) * 4u, code);
    }
}
