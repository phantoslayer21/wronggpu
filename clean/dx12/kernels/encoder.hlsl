// SPDX-License-Identifier: MIT
// D3D12 port of input_encoder (clean/kernels/encoder.hip; HIP build encoder.co = the base variant: no -D flags, so
// WG_ENC_FRAG_GLOBAL=1 (A fragments straight from global memory), WG_ENC_PIN=1, every other switch 0, motion at render
// resolution (EncMoOff), hardware FP8 conversion). HIP ISA: <build>\dx12-m\isa\input_encoder.s (build_dx12.py --isa).
//
// encoder.hip is compiled with "#pragma clang fp contract(off)": every a * b + c of its source is two roundings (the
// plain expression here) and every __builtin_fmaf one v_fma_f32 (mad here); sums keep the source's grouping (checked
// against the ISA). The four IEEE divisions of the uniform values are wg_div. Constants are the FP32 values clang folds
// (the ISA's literals): kG 0x3dddf182, kL2 0x4113a429, 1e-4 kL2 0x3a71e53a, 3 kL2 0x41dd763e, history weight scale
// 2.45 kL2 / 1023 = 0x3cb5098e (2.45 / 1023 = 0x3b1cf40a for kHistSame), latent 1 / 127 = 0x3c010204.
//
// Work split as in HIP: a wave handles 16 tokens, lane (t, h) the two pixels of row h of token t's 2 x 2 block. The
// per-pixel code exists in three copies, one per (uniform) history formula. The FP8 conversions of the dithered values
// and the projection's FP16 roundings / FP8 packs use the driver's matrix conversion (the bits of v_cvt_pk_fp8_f32 /
// v_cvt_f16_f32), the projection the driver's WMMA; all of these are in wave-uniform control flow. HIP sinks the
// projection into "tok < ntok" (its only effect is that store); here every lane runs it (the WMMA needs the whole wave)
// and only the store is conditional: the same bytes.
// @wg-kernel input_encoder co=encoder.co block=256 wave=32 args=p,p,p,p,p,p,p,p,p,p,p,p,b116
// @wg-kernel input_encoder_direct co=encoder.co entry=input_encoder block=256 wave=32 args=p,p,p,p,p,p,p,p,p,p,p,p,b116 rs=direct defines=WG_DIRECT_TEXTURES=1
//
// Motion at output resolution (WG_ENC_MO; HIP: input_encoder_mo of kernels/encoder_mo.hip, code object encoder_mo.co
// under the tag "emo"; encoder_motion_outres.h): no depth neighbourhood; per pixel (x, y) one point read of the raw motion
// at T = (clamp(floor(RN32(RN32((x + 1/2) - jx) k_Mx)), 0, Mw - 1), likewise y), m = sx (bx + MV(T)) (WgEncParams sx, sy,
// bx, by are s and b_mv here) and m_near = m_center = m_far = m; everything else as input_encoder. Arguments as that
// kernel without depth, then WgEncMoParams (mw, mh, pitch, wide, kmx, kmy) after WgEncParams. Buffer form: the raw
// motion buffer (wg_motion_hi.hlsli); direct form: the game's motion texture (its `motion` pointer is not read).
// @wg-kernel emo/input_encoder_mo co=encoder_mo.co entry=input_encoder block=256 wave=32 args=p,p,p,p,p,p,p,p,p,p,p,b116,b24 defines=WG_ENC_MO=1
// @wg-kernel emo/input_encoder_mo_direct co=encoder_mo.co entry=input_encoder block=256 wave=32 args=p,p,p,p,p,p,p,p,p,p,p,b116,b24 rs=direct defines=WG_DIRECT_TEXTURES=1,WG_ENC_MO=1
//
// Model L's value kernel (WG_ENC_VALUES_OUT; HIP: encoder.hip built with WG_ENC_VALUES_OUT=1 as l_enc_values.co, issue
// #89, clean/l/FORMATS.md `lev/input_encoder`): no projection. The grid is the padded grid plus a 2-pixel halo per side,
// X0 = 2 c - 2, Y = 2 r + h - 2; each pixel's 20 dithered FP8 values (12 zero bytes after them) go to the halo map
// `features` [fh + 4][fw + 4][32]. The values come from the mirrored render position, the dither from the unmirrored
// one (two byte loads at |Y| mod 16, |x| mod 16). When both halo columns of a lane mirror onto one render column
// (x0 == x1), slot 1 takes slot 0's depth neighbourhood and centre motion and keeps its own dither. Motion surfaces only
// for pixels inside R (0 <= Y < rh, 0 <= X0 < rw). Buffer form, and the direct form (the game's textures, as
// input_encoder_direct: the same switch on the direct-texture loads; no HIP kernel of its own, the bytes of the buffer
// form). Not with motion at output resolution.
// @wg-kernel lev/input_encoder co=l_enc_values.co entry=input_encoder block=256 wave=32 args=p,p,p,p,p,p,p,p,p,p,p,p,b116 defines=WG_ENC_VALUES_OUT=1
// @wg-kernel lev/input_encoder_direct co=l_enc_values.co entry=input_encoder block=256 wave=32 args=p,p,p,p,p,p,p,p,p,p,p,p,b116 rs=direct defines=WG_DIRECT_TEXTURES=1,WG_ENC_VALUES_OUT=1
#include "wg_platform.hlsli"
#ifndef WG_ENC_VALUES_OUT
#define WG_ENC_VALUES_OUT 0
#endif
#if WG_ENC_VALUES_OUT && defined(WG_ENC_MO)
#error "WG_ENC_VALUES_OUT: motion at render resolution only"
#endif
#if defined(WG_DIRECT_TEXTURES)
#include "native_direct.hlsli"
#include "wg_f16lut.hlsli"
#include "wg_f16lut_in.hlsli"
#endif
#if defined(WG_ENC_MO)
#include "wg_motion_hi.hlsli"
#endif

WG_ARGS_BEGIN
#if defined(WG_ENC_MO)
    uint color, motion, hist, latent, mhist;          // Frame without depth (motion: the raw motion buffer)
#else
    uint color, depth, motion, hist, latent, mhist;   // Frame
#endif
    uint features, m_near, m_far, m_center, expo, w;
    int P_rw, P_rh, P_fw, P_fh, P_hw, P_hh, P_frame, P_reset, P_depth_inverted, P_pad0;   // WgEncParams (116 bytes)
    float P_dw, P_dh, P_kx, P_ky, P_ikx, P_iky, P_sx, P_sy, P_bx, P_by, P_jx, P_jy, P_jpx, P_jpy, P_dis_scale;
    float P_idw, P_idh, P_irw, P_irh;
#if defined(WG_ENC_MO)
    int mo_mw, mo_mh, mo_pitch, mo_wide;              // WgEncMoParams (24 bytes)
    float mo_kmx, mo_kmy;
#endif
WG_ARGS_END

// Weight blob (FORMATS.md, "Input encoder (enc0)")
#define kOffBias 0u
#define kOffSigma 256u
#define kOffDither 512u
#define kOffFrags 2560u

#define kLn2 asfloat(0x3f317218u)
#define kL2 asfloat(0x4113a429u)
#define kG asfloat(0x3dddf182u)
#define kL2_1em4 asfloat(0x3a71e53au)    // 1e-4f * kL2
#define kL2_3 asfloat(0x41dd763eu)       // 3.0f * kL2
#define kS_GEN asfloat(0x3cb5098eu)      // 2.45f * kL2 / 1023.0f
#define kS_SAME asfloat(0x3b1cf40au)     // 2.45f / 1023.0f
#define kS_LAT asfloat(0x3c010204u)      // 1.0f / 127.0f
#define k1em4 asfloat(0x38d1b717u)       // 1e-4f

#define kHistGeneral 0
#define kHistRaw 1
#define kHistSame 2

struct Uni {
    float e, rr, ce;
    int mode;
    float shx, shy, cdx, cdy, s2;
};

int enc_reflect(int p, int n)
{
    p = p < 0 ? -p : p;
    p = p >= n ? 2 * n - p - 2 : p;
    return wg_med3i(p, 0, n - 1);
}

// wg_mem.h wg_footprint
struct Foot { uint off0, off1; float fx, fy; };
Foot enc_footprint(float tx, float ty, int w_, int h_, uint tb)
{
    float fw = (float)w_, fh = (float)h_;
    float cx = wg_med3f(tx, 0.0f, fw - 1.0f), cy = wg_med3f(ty, 0.0f, fh - 1.0f);
    float x0 = wg_fmin(floor(cx), fw - 2.0f), y0 = wg_fmin(floor(cy), fh - 2.0f);
    Foot f;
    f.fx = cx - x0;
    f.fy = cy - y0;
    f.off0 = (uint)mad(y0, fw, x0) * tb;
    f.off1 = f.off0 + (uint)w_ * tb;
    return f;
}

// G(h) of one position (hist_values of encoder.hip); mode is a literal at every call (three inlined copies).
float3 hist_values(int mode, Uni U, float px, float py, bool valid, float3 gc)
{
    Foot ft = enc_footprint(px * U.shx - 0.5f, py * U.shy - 0.5f, P_hw, P_hh, 4u);
    uint2 t0 = wg_ld64(hist, ft.off0), t1 = wg_ld64(hist, ft.off1);
    float kS = mode == kHistSame ? kS_SAME : kS_GEN;
    float wy1 = ft.fy * kS, wy0 = kS - wy1;
    float ifx = 1.0f - ft.fx;
    float w00 = ifx * wy0, w10 = ft.fx * wy0, w01 = ifx * wy1, w11 = ft.fx * wy1;
    float3 o;
    [unroll] for (uint c = 0; c < 3; ++c) {
        float a0 = (float)((t0.x >> (10u * c)) & 1023u), b0 = (float)((t0.y >> (10u * c)) & 1023u);
        float a1 = (float)((t1.x >> (10u * c)) & 1023u), b1 = (float)((t1.y >> (10u * c)) & 1023u);
        float x = mad(w00, a0, mad(w10, b0, 0.0f));
        x = mad(w01, a1, mad(w11, b1, x));
        float g;
        if (mode == kHistSame) {
            g = wg_med3f(x, k1em4, 3.0f);
        } else {
            float a;
            if (mode == kHistGeneral) {
                float pm1 = wg_exp2(wg_med3f(x, kL2_1em4, kL2_3)) - 1.0f;
                a = mad(U.rr, pm1, 1.0f);
            } else {
                a = mad(x, U.ce, 1.0f);
            }
            g = kG * wg_log2(wg_fmax(a, k1em4));
        }
        o[c] = valid ? g : gc[c];
    }
    return o;
}

// The 20 values of the pixel at render column x, row y (pixel_values of encoder.hip).
void pixel_values(int mode, Uni U, int x, int y, float4 c, float2 mn, float2 mf, float2 mc, out float v[20]
#if defined(WG_DIRECT_TEXTURES)
                  , bool lut, uint2 lt[3]
#endif
                  )
{
    float3 gc;
#if defined(WG_DIRECT_TEXTURES)
    if (lut) {   // uniform: the colour values from this frame's table (wg_f16lut_in.hlsli)
        [unroll] for (uint ch = 0; ch < 3; ++ch) {
            v[ch] = asfloat(lt[ch].x);
            gc[ch] = asfloat(lt[ch].y);
        }
    } else
#endif
    [unroll] for (uint ch = 0; ch < 3; ++ch) {
        float ec = U.e * c[ch];
        float r = kG * wg_log2(wg_fmax(mad(100.0f, wg_fmin(ec, 32752.0f), 1.0f), k1em4));
        float g = ec > 32752.0f ? kG * wg_log2(mad(100.0f, ec, 1.0f)) : r;
        bool nan = ec != ec;
        v[ch] = nan ? 0.0f : r;
        gc[ch] = nan ? 0.0f : g;
    }

    float xh = (float)x + 0.5f, yh = (float)y + 0.5f;
    float basex = xh - P_jx, basey = yh - P_jy;
    float bkx = basex * P_kx, bky = basey * P_ky;
    float qx[4], qy[4];
    qx[0] = bkx + mn.x; qy[0] = bky + mn.y;   // near
    qx[1] = bkx + mc.x; qy[1] = bky + mc.y;   // center
    qx[2] = bkx + mf.x; qy[2] = bky + mf.y;   // far
    qx[3] = xh * P_kx + mc.x; qy[3] = yh * P_ky + mc.y;   // unjittered
    bool valid[4];
    [unroll] for (uint k = 0; k < 4; ++k) valid[k] = qx[k] >= 0.0f && qx[k] <= P_dw && qy[k] >= 0.0f && qy[k] <= P_dh;

    if (P_reset != 0) {   // uniform: nothing of the previous frame is read
        [unroll] for (uint i = 3; i < 15; ++i) v[i] = gc[i % 3u];
        [unroll] for (uint i2 = 15; i2 < 20; ++i2) v[i2] = 0.0f;
        return;
    }
    [unroll] for (uint k2 = 0; k2 < 4; ++k2) {
        float3 h = hist_values(mode, U, qx[k2], qy[k2], valid[k2], gc);
        v[3 + 3 * k2] = h.x; v[4 + 3 * k2] = h.y; v[5 + 3 * k2] = h.z;
    }

    // Motion disagreement (value 15)
    {
        float ux = wg_med3f(basex + P_jpx + mn.x * P_ikx, 0.0f, (float)P_rw);
        float uy = wg_med3f(basey + P_jpy + mn.y * P_iky, 0.0f, (float)P_rh);
        Foot ft = enc_footprint(ux - 0.5f, uy - 0.5f, P_rw, P_rh, 4u);
        uint2 t0 = wg_ld64(mhist, ft.off0), t1 = wg_ld64(mhist, ft.off1);
        float a0 = 1.0f - ft.fx, b0 = 1.0f - ft.fy;
        float ox = b0 * (a0 * wg_h2f_lo(t0.x) + ft.fx * wg_h2f_lo(t0.y)) +
                   ft.fy * (a0 * wg_h2f_lo(t1.x) + ft.fx * wg_h2f_lo(t1.y));
        float oy = b0 * (a0 * wg_h2f_hi(t0.x) + ft.fx * wg_h2f_hi(t0.y)) +
                   ft.fy * (a0 * wg_h2f_hi(t1.x) + ft.fx * wg_h2f_hi(t1.y));
        float vx = (mn.x - ox) * U.cdx, vy = (mn.y - oy) * U.cdy;
        float d = U.s2 * wg_log2(1.0f + wg_sqrt(mad(vx, vx, vy * vy)));
        v[15] = d != d ? 0.0f : d;
    }

    // Latent (values 16..19); texels k / 127, the 1 / 127 folded into the weights
    {
        Foot ft = enc_footprint(qx[0] * P_ikx + P_jpx - 0.5f, qy[0] * P_iky + P_jpy - 0.5f, P_rw, P_rh, 4u);
        uint2 t0 = wg_ld64(latent, ft.off0), t1 = wg_ld64(latent, ft.off1);
        float wy1 = ft.fy * kS_LAT, wy0 = kS_LAT - wy1;
        float ifx = 1.0f - ft.fx;
        float w00 = ifx * wy0, w10 = ft.fx * wy0, w01 = ifx * wy1, w11 = ft.fx * wy1;
        [unroll] for (uint ch2 = 0; ch2 < 4; ++ch2) {
            uint sh = 24u - 8u * ch2;
            float l00 = (float)((int)(t0.x << sh) >> 24), l10 = (float)((int)(t0.y << sh) >> 24);
            float l01 = (float)((int)(t1.x << sh) >> 24), l11 = (float)((int)(t1.y << sh) >> 24);
            float s = mad(w00, l00, mad(w10, l10, mad(w01, l01, w11 * l11)));
            v[16 + ch2] = valid[0] ? s : 0.0f;
        }
    }
}

// Spec 03 dithered quantization of one value (dither_q of encoder.hip): integer form on the FP32 bits.
float dither_q(float x, uint tt)
{
    uint b = asuint(wg_med3f(x, -447.75f, 447.75f));
    return asfloat((b + ((b >> 13) & 1u) + tt) & 0xfff00000u);
}

// 20 values with dither t -> 5 words of FP8 codes (value i = byte i). The dithered values are within +-448, so the
// conversion without clamp (the driver's = v_cvt_pk_fp8_f32) gives HIP's bytes.
void quantize(float v[20], uint t, out uint q[5])
{
    uint tt = 0xfffu + ((t + 1u) << 13);
    v8f a, b, c;
    [unroll] for (uint i = 0; i < 8; ++i) {
        a.v[i] = dither_q(v[i], tt);
        b.v[i] = dither_q(v[8 + i], tt);
        c.v[i] = i < 4 ? dither_q(v[16 + i], tt) : 0.0f;
    }
    uint2 qa = wg_pack_q8_nc_ags(a), qb = wg_pack_q8_nc_ags(b), qc = wg_pack_q8_nc_ags(c);
    q[0] = qa.x; q[1] = qa.y; q[2] = qb.x; q[3] = qb.y; q[4] = qc.x;
}

// (float)(_Float16) of the 8 accumulator values (the driver's conversion, RNE).
v8f round_f16(v8f a)
{
    // Both conversions through the driver's matrix conversion (f32 -> f16 RNE, then f16 -> f32, which is exact): the
    // bits of (float)(_Float16) without the v_mov_b16 the driver puts before each low-half v_cvt_f32_f16 of
    // f16tof32(h & 0xffff). Proven over every FP16 bit pattern in all 8 fragment positions (probe/rf16_check.py).
    wg_frag hf = wg_ags_convert(wg_frag_v8f(a), WG_AGS_T_F32_ACC, WG_AGS_T_F16_ACC);
    wg_frag bf = wg_ags_convert(hf, WG_AGS_T_F16_ACC, WG_AGS_T_F32_ACC);
    v8f r;
    [unroll] for (uint i = 0; i < 8; ++i) r.v[i] = asfloat(bf.v[i]);
    return r;
}

// Near / far search over the 3 x 3 neighbourhood of the pixel whose window columns are o .. o + 2 (strict comparisons
// in the spec's tie-break order).
void near_far(uint o, float4 Dm, float4 Dc, float4 Dp, uint rm, uint rc, uint rp, uint col[4], bool inverted,
              out uint inear, out uint ifar)
{
    float dv[9] = {Dc[o + 1], Dm[o], Dm[o + 1], Dc[o], Dm[o + 2], Dc[o + 2], Dp[o], Dp[o + 1], Dp[o + 2]};
    uint di[9] = {rc + col[o + 1], rm + col[o], rm + col[o + 1], rc + col[o], rm + col[o + 2],
                  rc + col[o + 2], rp + col[o], rp + col[o + 1], rp + col[o + 2]};
    float lo = dv[0], hi = dv[0];
    uint ilo = di[0], ihi = di[0];
    [unroll] for (uint n = 1; n < 9; ++n) {
        bool l = dv[n] < lo, g = dv[n] > hi;
        lo = l ? dv[n] : lo; ilo = l ? di[n] : ilo;
        hi = g ? dv[n] : hi; ihi = g ? di[n] : ihi;
    }
    inear = inverted ? ihi : ilo;
    ifar = inverted ? ilo : ihi;
}

float2 motion_of(float2 m) { return float2(P_sx * (P_bx + m.x), P_sy * (P_by + m.y)); }

#if defined(WG_ENC_MO)
// encoder_motion_outres.h enc_mo_index: clamp(floor(RN32(RN32((i + 1/2) - j) k)), 0, n - 1); a NaN coordinate gives 0
uint enc_mo_index(int i, float j, float k, int n)
{
    const float t = floor((((float)i + 0.5f) - j) * k);
    return (uint)(int)wg_fmin(wg_fmax(t, 0.0f), (float)(n - 1));
}

// The raw motion of render column x in the texel row ty
float2 enc_mo_raw(int x, uint ty)
{
    const uint tx = enc_mo_index(x, P_jx, mo_kmx, mo_mw);
#if defined(WG_DIRECT_TEXTURES)
    return wg_mo_texel_direct(tx, ty, wg_mo_dims());
#else
    return wg_mo_value(wg_mo_load(motion, ty * (uint)mo_pitch + (mo_wide != 0 ? tx * 8u : tx * 4u)), mo_wide);
#endif
}
#endif

void encoder_main()
{
    uint tid = wg_lane_id();
    uint lane = tid & 31u, t = lane & 15u, hh = lane >> 4;
#if WG_ENC_VALUES_OUT
    int Wt = (P_fw + 4) >> 1, ntok = ((P_fh + 4) >> 1) * Wt;   // the padded grid plus the 2-pixel halo
#else
    int Wt = P_fw >> 1, ntok = (P_fh >> 1) * Wt;
#endif
    // Workgroups in reverse token order (the last token rows first; the per-token work does not depend on the order):
    // the encoder starts on the rows whose colour the exposure pass read last, and its last writes are the rows enc1
    // reads first, so both find more of their data in the Infinity Cache (-3.3 us per frame at 1706x960).
    int tok = (int)((wg_grid_size() - 1u - wg_block_id()) * (WG_BLOCK / 32u) + (tid >> 5)) * 16 + (int)t;
    int tk = tok < ntok ? tok : ntok - 1;
    int r = tk / Wt, cc = tk - r * Wt;

    int rw = P_rw, rh = P_rh;
#if WG_ENC_VALUES_OUT
    int X0 = 2 * cc - 2, Y = 2 * r + (int)hh - 2;
#else
    int X0 = 2 * cc, Y = 2 * r + (int)hh;
#endif
    int y = enc_reflect(Y, rh);
    int x0 = enc_reflect(X0, rw), x1 = enc_reflect(X0 + 1, rw);
    bool rev = x0 > x1;   // right padding: slot 0 is pixel B
    int xa = rev ? x1 : x0;

    // Depth window: rows y - 1, y, y + 1 and columns xa - 1 .. xa + 2, clamped to the render area
    uint rm = (uint)(wg_med3i(y - 1, 0, rh - 1) * rw), rc = (uint)(y * rw), rp = (uint)(wg_med3i(y + 1, 0, rh - 1) * rw);
    uint col[4] = {(uint)(xa > 0 ? xa - 1 : 0), (uint)xa, (uint)xa + 1u, (uint)(xa + 2 < rw ? xa + 2 : rw - 1)};
    // All loads that do not depend on other loads first (the uniform values are computed while they are in flight)
    float e0 = wg_ldf(expo, 0u), e1 = wg_ldf(expo, 4u);
    float s1 = wg_ldf(w, kOffSigma), s2 = wg_ldf(w, kOffSigma + 4u);
#if defined(WG_ENC_MO)
    // motion at output resolution: no depth neighbourhood; one raw motion texel per pixel (texel row of y)
    const uint mo_ty = enc_mo_index(y, P_jy, mo_kmy, mo_mh);
    uint xs[2] = {(uint)x0, (uint)x1};
    const float2 mraw0 = enc_mo_raw(x0, mo_ty), mraw1 = enc_mo_raw(x1, mo_ty);
#if defined(WG_DIRECT_TEXTURES)
    float4 cs[2] = {nd_raw_color_xy((int)xs[0], y), nd_raw_color_xy((int)xs[1], y)};   // divided below (no table)
    const uint4 lhead = wg_mem.Load4(color + WG_F16LUT_IN);   // the input tables' header (color: the tables, 0 = none)
#else
    float4 cs[2] = {wg_ld4f(color, (rc + xs[0]) * 16u), wg_ld4f(color, (rc + xs[1]) * 16u)};
#endif
#else
    uint s = (uint)wg_med3i(xa - 1, 0, rw - 4);
#if defined(WG_DIRECT_TEXTURES)
    // Direct textures: the texels at their coordinates (no dense index divided by the width again). The rows are
    // clamped / reflected into the render area and s + 3 <= rw - 1, so every texel is one the dense index named.
    const int ymr = wg_med3i(y - 1, 0, rh - 1), ypr = wg_med3i(y + 1, 0, rh - 1);
    float4 Dm = nd_depth4_xy((int)s, ymr);
    float4 Dc = nd_depth4_xy((int)s, y);
    float4 Dp = nd_depth4_xy((int)s, ypr);
#else
    float4 Dm = wg_ld4f(depth, (rm + s) * 4u);
    float4 Dc = wg_ld4f(depth, (rc + s) * 4u);
    float4 Dp = wg_ld4f(depth, (rp + s) * 4u);
#endif
    uint xs[2] = {(uint)x0, (uint)x1};
#if defined(WG_DIRECT_TEXTURES)
    float4 cs[2] = {nd_raw_color_xy((int)xs[0], y), nd_raw_color_xy((int)xs[1], y)};   // divided below (no table)
    const uint4 lhead = wg_mem.Load4(color + WG_F16LUT_IN);   // the input tables' header (color: the tables, 0 = none)
    float4 mcab = float4(nd_motion_xy(xa, y), nd_motion_xy(xa + 1, y));   // xa <= rw - 2
#else
    float4 cs[2] = {wg_ld4f(color, (rc + xs[0]) * 16u), wg_ld4f(color, (rc + xs[1]) * 16u)};
    float4 mcab = wg_ld4f(motion, (rc + (uint)xa) * 8u);
#endif
#endif
#if WG_ENC_VALUES_OUT
    // dither word per pixel at the unmirrored position (X0, Y may be negative): |y| mod 16, |x| mod 16
    const uint trow = kOffDither + 256u * (uint)P_frame + (((uint)abs(Y) & 15u) << 4);
    uint t2 = wg_ld8(w, trow + ((uint)abs(X0) & 15u)) | (wg_ld8(w, trow + ((uint)abs(X0 + 1) & 15u)) << 8);
#else
    uint t2 = wg_ld16(w, kOffDither + 256u * (uint)P_frame + (((uint)Y & 15u) << 4) + ((uint)X0 & 15u));
#endif

    // Uniform values
    Uni U;
    {
        float e = e0 == 0.0f ? 1.0f : e0;
        float ep = e1 == 0.0f ? 1.0f : e1;
        bool e_ok = e == e;
        bool prev_ok = ep > 0.0f;
        U.e = e;
        U.mode = !prev_ok ? kHistRaw : (e == ep ? kHistSame : kHistGeneral);
        U.rr = e_ok ? wg_div(e, ep) : 0.0f;
        U.ce = e_ok ? wg_div(100.0f * e, kL2) : 0.0f;
        U.shx = wg_div((float)P_hw, P_dw);
        U.shy = wg_div((float)P_hh, P_dh);
        U.cdx = s1 * P_dis_scale * P_ikx;
        U.cdy = s1 * P_dis_scale * P_iky;
        U.s2 = s2 * kLn2;
    }

#if defined(WG_DIRECT_TEXTURES)
    // this frame's colour table (wg_f16lut_in.hlsli; wave-uniform test), else nd_color_xy's division
    const bool lut = WaveReadLaneFirst(f16lut_in_valid(color, lhead, U.e, nd_color_pi) ? 1u : 0u) != 0u;
    uint2 lt[2][3];
    if (lut) {
        [unroll] for (uint d = 0; d < 2; ++d)
            [unroll] for (uint ch = 0; ch < 3; ++ch) lt[d][ch] = wg_ld64(color, WG_F16LUT_ENC + (f32tof16(cs[d][ch]) << 3));
    } else {
        [unroll] for (uint d = 0; d < 2; ++d) {
            [unroll] for (uint ch = 0; ch < 3; ++ch) lt[d][ch] = uint2(0u, 0u);
            if (nd_color_pi != 1.0f) {
                const float3 rgb = native_direct_color_div3(cs[d].rgb);
                cs[d] = float4(asfloat(asuint(rgb) ^ (nd_rw >> 31u)), cs[d].w);
            }
        }
    }
#endif

#if !defined(WG_ENC_MO)
    uint inA, ifA, inB, ifB;
    {
        if (xa == 0) {   // window loaded from column 0: columns (0, 0, 1, 2)
            Dm = float4(Dm.x, Dm.x, Dm.y, Dm.z);
            Dc = float4(Dc.x, Dc.x, Dc.y, Dc.z);
            Dp = float4(Dp.x, Dp.x, Dp.y, Dp.z);
        } else if (xa == rw - 2) {   // window loaded from column rw - 4: columns (rw - 3, rw - 2, rw - 1, rw - 1)
            Dm = float4(Dm.y, Dm.z, Dm.w, Dm.w);
            Dc = float4(Dc.y, Dc.z, Dc.w, Dc.w);
            Dp = float4(Dp.y, Dp.z, Dp.w, Dp.w);
        }
#if defined(WG_DIRECT_TEXTURES)
        // the same search with packed coordinates (row << 16 | column) in place of the dense indices
        near_far(0u, Dm, Dc, Dp, (uint)ymr << 16, (uint)y << 16, (uint)ypr << 16, col, P_depth_inverted != 0, inA, ifA);
        near_far(1u, Dm, Dc, Dp, (uint)ymr << 16, (uint)y << 16, (uint)ypr << 16, col, P_depth_inverted != 0, inB, ifB);
#else
        near_far(0u, Dm, Dc, Dp, rm, rc, rp, col, P_depth_inverted != 0, inA, ifA);
        near_far(1u, Dm, Dc, Dp, rm, rc, rp, col, P_depth_inverted != 0, inB, ifB);
#endif
    }

    // Per slot: near / far pixel indices, centre motion
#if WG_ENC_VALUES_OUT
    // both halo columns mirrored onto one render column (x0 == x1): slot 1 is that pixel too
    const bool same = x0 == x1;
    uint ins[2] = {rev ? inB : inA, same ? inA : (rev ? inA : inB)}, ifs[2] = {rev ? ifB : ifA, same ? ifA : (rev ? ifA : ifB)};
    float2 mcs[2] = {rev ? mcab.zw : mcab.xy, same ? mcab.xy : (rev ? mcab.xy : mcab.zw)};
#else
    uint ins[2] = {rev ? inB : inA, rev ? inA : inB}, ifs[2] = {rev ? ifB : ifA, rev ? ifA : ifB};
    float2 mcs[2] = {rev ? mcab.zw : mcab.xy, rev ? mcab.xy : mcab.zw};
#endif
#endif
#if WG_ENC_VALUES_OUT
    bool store = tok < ntok && Y >= 0 && Y < rh && X0 >= 0;
#else
    bool store = tok < ntok && Y < rh;
#endif
    uint so = (uint)(Y * rw + X0) * 4u;

    float2 mn[2], mf[2], mc[2];
#if defined(WG_ENC_MO)
    // enc_mo_motions: the motion of both pixels (slot 0: column x0), m_near = m_far = m_center = m
    mc[0] = motion_of(mraw0);
    mc[1] = motion_of(mraw1);
    [unroll] for (uint dxm = 0; dxm < 2; ++dxm) {
        mn[dxm] = mc[dxm];
        mf[dxm] = mc[dxm];
    }
#else
    [unroll] for (uint dx = 0; dx < 2; ++dx) {
        mc[dx] = motion_of(mcs[dx]);
#if defined(WG_DIRECT_TEXTURES)
        mn[dx] = motion_of(nd_motion_xy((int)(ins[dx] & 0xffffu), (int)(ins[dx] >> 16)));
        mf[dx] = motion_of(nd_motion_xy((int)(ifs[dx] & 0xffffu), (int)(ifs[dx] >> 16)));
#else
        mn[dx] = motion_of(asfloat(wg_ld64(motion, ins[dx] * 8u)));
        mf[dx] = motion_of(asfloat(wg_ld64(motion, ifs[dx] * 8u)));
#endif
    }
#endif
    // FP16 (RNE) motion surfaces: the driver's conversion (v_cvt_f16_f32, as HIP's wg_pack_h2), in uniform control flow
    v8f m8a, m8b;
    m8a.v[0] = mn[0].x; m8a.v[1] = mn[0].y; m8a.v[2] = mn[1].x; m8a.v[3] = mn[1].y;
    m8a.v[4] = mf[0].x; m8a.v[5] = mf[0].y; m8a.v[6] = mf[1].x; m8a.v[7] = mf[1].y;
    m8b.v[0] = mc[0].x; m8b.v[1] = mc[0].y; m8b.v[2] = mc[1].x; m8b.v[3] = mc[1].y;
    m8b.v[4] = 0.0f; m8b.v[5] = 0.0f; m8b.v[6] = 0.0f; m8b.v[7] = 0.0f;
    uint4 hnf = wg_to_f16_ags(m8a), hc = wg_to_f16_ags(m8b);
    if (store && X0 < rw) {
        uint2 sn = hnf.xy, sf = hnf.zw, sc = hc.xy;
        // Driver bug (gfx1201 D3D12 compiler): pointers read from the same 16-byte row of the argument cbuffer
        // (m_near, m_far, m_center) are taken as one value when they address memory with the same offset, so the
        // first two stores were dropped as overwritten. wg_uniform (readfirstlane) makes them distinct values.
        uint pn = (uint)wg_uniform((int)m_near), pf = (uint)wg_uniform((int)m_far), pc = m_center;
        if (X0 + 1 < rw) {
            wg_st64(pn, so, sn);
            wg_st64(pf, so, sf);
            wg_st64(pc, so, sc);
        } else {
            wg_st32(pn, so, sn.x);
            wg_st32(pf, so, sf.x);
            wg_st32(pc, so, sc.x);
        }
    }

    // Slot 0, then slot 1: the 20 values -> 5 words of FP8, values 8..15 swapped with the partner lane (lane ^ 16)
    uint q[2][5], x8[2][2];
    [unroll] for (uint dx2 = 0; dx2 < 2; ++dx2) {
        float4 c = cs[dx2];
        float v[20];
        if (U.mode == kHistGeneral)   // uniform
            pixel_values(kHistGeneral, U, (int)xs[dx2], y, c, mn[dx2], mf[dx2], mc[dx2], v
#if defined(WG_DIRECT_TEXTURES)
                         , lut, lt[dx2]
#endif
                         );
        else if (U.mode == kHistSame)
            pixel_values(kHistSame, U, (int)xs[dx2], y, c, mn[dx2], mf[dx2], mc[dx2], v
#if defined(WG_DIRECT_TEXTURES)
                         , lut, lt[dx2]
#endif
                         );
        else
            pixel_values(kHistRaw, U, (int)xs[dx2], y, c, mn[dx2], mf[dx2], mc[dx2], v
#if defined(WG_DIRECT_TEXTURES)
                         , lut, lt[dx2]
#endif
                         );
        uint qq[5];
        quantize(v, dx2 != 0u ? t2 >> 8 : t2 & 0xffu, qq);
        [unroll] for (uint i = 0; i < 5; ++i) q[dx2][i] = qq[i];
#if !WG_ENC_VALUES_OUT
        x8[dx2][0] = wg_xor16_u(qq[2]);
        x8[dx2][1] = wg_xor16_u(qq[3]);
#endif
    }

#if WG_ENC_VALUES_OUT
    // pixel (X0 + dx, Y) of the halo map: 20 value bytes and 12 zero bytes
    if (tok < ntok) {
        const uint mw = (uint)(P_fw + 4);
        [unroll] for (uint dx3 = 0; dx3 < 2; ++dx3) {
            const uint ho = ((uint)(Y + 2) * mw + (uint)(X0 + (int)dx3 + 2)) * 32u;
            wg_st128(features, ho, uint4(q[dx3][0], q[dx3][1], q[dx3][2], q[dx3][3]));
            wg_st128(features, ho + 16u, uint4(q[dx3][4], 0u, 0u, 0u));
        }
    }
#else

    // B fragments (this lane's slots) of the 8 K blocks
    uint2 bf[8];
    [unroll] for (uint p = 0; p < 4; ++p) {
        bool own = (p >> 1) == hh;
        uint d = p & 1u;
        bf[2 * p] = own ? uint2(q[d][0], q[d][1]) : uint2(x8[d][0], x8[d][1]);
        bf[2 * p + 1] = uint2(own ? q[d][4] : 0u, 0u);
    }

    // Two output tiles at a time; every A fragment and bias of a pair is loaded before its first WMMA
    uint o[8];
    [unroll] for (uint pr = 0; pr < 2; ++pr) {
        v8f acc[2];
        uint2 A[2][8];
        [unroll] for (uint j = 0; j < 2; ++j) {
            acc[j] = wg_load8f(w, kOffBias + hh * 32u + (2u * pr + j) * 64u);
            [unroll] for (uint kb = 0; kb < 8; ++kb)
                A[j][kb] = wg_ld64(w, kOffFrags + ((2u * pr + j) * 8u + kb) * 256u + lane * 8u);
        }
        [unroll] for (uint p2 = 0; p2 < 4; ++p2) {
            [unroll] for (uint j2 = 0; j2 < 2; ++j2) {
                acc[j2] = wg_wmma_fp8(A[j2][2 * p2], bf[2 * p2], acc[j2]);
                acc[j2] = wg_wmma_fp8(A[j2][2 * p2 + 1], bf[2 * p2 + 1], acc[j2]);
                acc[j2] = round_f16(acc[j2]);
            }
        }
        [unroll] for (uint j3 = 0; j3 < 2; ++j3) {
            [unroll] for (uint i = 0; i < 8; ++i) acc[j3].v[i] = wg_fmin(wg_fmax(acc[j3].v[i], -6.28125f), 6.28125f);
            uint2 qo = wg_pack_q8_nc_ags(acc[j3]);   // within +-6.28125: the bytes of wg_pack_q8
            o[4 * pr + 2 * j3] = qo.x;
            o[4 * pr + 2 * j3 + 1] = qo.y;
        }
    }
    if (tok < ntok) {
        uint fo = (uint)tok * 64u + hh * 32u;
        wg_st128(features, fo, uint4(o[0], o[1], o[2], o[3]));
        wg_st128(features, fo + 16u, uint4(o[4], o[5], o[6], o[7]));
    }
#endif
}

WG_KERNEL(input_encoder)
{
    WG_SETUP();
    encoder_main();
}
