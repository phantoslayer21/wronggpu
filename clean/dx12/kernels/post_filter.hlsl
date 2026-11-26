// SPDX-License-Identifier: MIT
// D3D12 port of the standalone output filter w64/post_filter_c1_s4 of post_w64.co (clean/kernels/post.hip
// filter_strip_c1<4>, built with -mwavefrontsize64 and the analytic filter, WG_LZ_PROFILE = 0), byte for byte. The
// runtime's default unbanded filter (kDefaultFilter): it runs after a reconstruction kernel that writes the display image
// (for example Model L's l_post_recon_p12) when the model file has no output-filter weights.
// display: three FP16 planes of D (E-encoded), row pitch display_pitch(dw) = (dw + 63) & ~63 halves; out: O, 3 x f32
// (row-major RGB); dec_scale = 0.01 / e, or (float)(0.01 / (double)e) from the exposure pair when expo is not null.
// Workgroup of 64 lanes (one wave64): lane 3 g + c of each 32-lane half (g < 10; lanes 30, 31 idle) computes channel c of
// output columns 4m ... 4m + 3, m = 10 (tile column) + g, for 4 row pairs from p0 = (2 (tile row) + half) 4. Grid
// ceil(ow / 40) * ceil(ceil(oh / 2) / 8).
// Per value: the horizontal pass of five display taps per output column (v = w0 t0, one fused multiply-add per further
// tap, clamp to the taps' range; spec 06's Lanczos weights kLz0 ... kLz4 as FP32 constants, mirrored for an odd output
// index), a 7-row sliding window of the horizontal results, the vertical pass of the same form, clamp to [1e-4, 3],
// Dec (2^(kDecK q) - 1, the polynomial below kDecPolyMax), times dec_scale. The arithmetic of post.hlsl's analytic
// filter (WG_LZ_ANALYTIC), which is HIP's ISA.
//
// l_post_filter_direct (rs=direct; D3D12 only: Model L's own post on the direct-resource route, runtime option l_post=own):
// the same filter with the output in the game's output texture instead of out_p (not written), as the runtime's FP32
// output followed by l_direct_store stored it: times pi (native_direct_scale) when pi != 1, with output_rne16
// framebuffer_half's bytes (+-65504 clamp, round to nearest even), alpha 1. The lane of channel 0 of each output pixel
// gathers the other two channels from its lanes 3 g + 1, 3 g + 2 and stores the texel.
//
// l_post_filter_ldr (WG_FILTER_LDR = 1): l_post.hip's LDR output stage l_post_filter_ldr (spec preset L part 3 Q2), the
// same filter with the filtered value written as it is (no clamp to [1e-4, 3], no decode, no exposure: dec_scale and
// expo are not read); l_post_filter_direct_ldr: its direct form, stored as l_post_filter_direct stores.
//
// @wg-kernel w64/post_filter_c1_s4 co=post_w64.co entry=post_filter_c1_s4 block=64 wave=64 args=p,p,u,u,u,u,u,p
// @wg-kernel w64/post_filter_c1_s4 co=post_w64__orig.co file=w64__post_filter_c1_s4__orig entry=post_filter_c1_s4 block=64 wave=64 args=p,p,u,u,u,u,u,p,u,u,u,u,u defines=WG_LZ_ORIG=1
// @wg-kernel l_post_filter_direct co=- entry=post_filter_c1_s4 block=64 wave=64 args=p,p,u,u,u,u,u,p rs=direct defines=WG_DIRECT_TEXTURES=1
// @wg-kernel l_post_filter_direct_orig co=- entry=post_filter_c1_s4 block=64 wave=64 args=p,p,u,u,u,u,u,p,u,u,u,u,u rs=direct defines=WG_DIRECT_TEXTURES=1,WG_LZ_ORIG=1
// @wg-kernel l_post_filter_ldr co=l_post.co entry=post_filter_c1_s4 block=64 wave=64 args=p,p,u,u,u,u,u,p defines=WG_FILTER_LDR=1
// @wg-kernel l_post_filter_direct_ldr co=- entry=post_filter_c1_s4 block=64 wave=64 args=p,p,u,u,u,u,u,p rs=direct defines=WG_DIRECT_TEXTURES=1,WG_FILTER_LDR=1
#include "wg_platform.hlsli"
#if defined(WG_DIRECT_TEXTURES)
#include "native_direct.hlsli"
#endif

WG_ARGS_BEGIN
    uint display, out_p;   // pointers
    int dw, dh, ow, oh;
    float dec_scale_arg;
    uint expo;             // pointer (0 = null)
#if defined(WG_LZ_ORIG)
    uint lz_w0, lz_w1, lz_w2, lz_w3, lz_w4;   // FP16 bit patterns of the original 5-tap filter weights
#endif
WG_ARGS_END

static const float kLz0 = asfloat(0xbdba3b4bu), kLz1 = asfloat(0x3de71ca9u), kLz2 = asfloat(0x3f23b70fu);
static const float kLz3 = asfloat(0x3ed182b5u), kLz4 = asfloat(0xbd90a4acu);
static const float kDecK = asfloat(0x4113a428u);        // 3.32192809488736235f / 0.36f
static const float kDecLn = asfloat(0x40ccac9au);       // 2.30258509299404568f / 0.36f
static const float kDecPolyMax = asfloat(0x3d60233du);  // 0.35f / kDecLn
static const float kP5040 = asfloat(0x39500d01u), kP720 = asfloat(0x3ab60b61u), kP120 = asfloat(0x3c088889u);
static const float kP24 = asfloat(0x3d2aaaabu), kP6 = asfloat(0x3e2aaaabu);
static const float k1em4 = asfloat(0x38d1b717u);        // 1e-4f

static uint g_z;   // 0, opaque to the compiler (from ow > 0)
float opq(float x) { return asfloat(asuint(x) ^ g_z); }
int clampi(int v, int lo, int hi) { int t = v > hi ? hi : v; return t < lo ? lo : t; }
float clampf(float v, float lo, float hi) { return min(max(v, lo), hi); }

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

#if defined(WG_LZ_ORIG)
// Original 5-tap FP16 Lanczos filter (post.hip WG_LZ_PROFILE=2, post.hlsl lz5_original): boxed FP16 bit patterns.
half hbox(uint v) { return asfloat16((uint16_t)(v & 0xffffu)); }
uint hunbox(half v) { return (uint)asuint16(v); }
uint lz5_original(uint t0b, uint t1b, uint t2b, uint t3b, uint t4b, half w0, half w1, half w2, half w3, half w4)
{
    half t0 = hbox(t0b), t1 = hbox(t1b), t2 = hbox(t2b), t3 = hbox(t3b), t4 = hbox(t4b);
    half a = mad(w0, t0, (half)0.0);
    a = mad(w1, t1, a);
    a = mad(w2, t2, a);
    a = mad(w3, t3, a);
    a = mad(w4, t4, a);
    half mid_lo = min(min(t1, t2), t3), mid_hi = max(max(t1, t2), t3);
    half lo = min(min(mid_lo, t0), t4), hi = max(max(mid_hi, t0), t4);
    return hunbox(min(max(a, lo), hi));
}
uint lz5e_o(uint t0, uint t1, uint t2, uint t3, uint t4)
{
    return lz5_original(t0, t1, t2, t3, t4, hbox(lz_w0), hbox(lz_w1), hbox(lz_w2), hbox(lz_w3), hbox(lz_w4));
}
uint lz5o_o(uint t0, uint t1, uint t2, uint t3, uint t4)
{
    return lz5_original(t0, t1, t2, t3, t4, hbox(lz_w4), hbox(lz_w3), hbox(lz_w2), hbox(lz_w1), hbox(lz_w0));
}
uint4 row4_o(int pitch, int plane, int r, int m, bool fast, int c)
{
    const uint row = (uint)(r * pitch + c * plane) * 2u;
    const int c0 = 6 * m - 2;
    uint q[5];
    [unroll] for (int i = 0; i < 5; ++i) {
        if (fast) {
            q[i] = wg_ld32(display, row + (uint)(c0 + 2 * i) * 2u);
        } else {
            q[i] = wg_ld16(display, row + (uint)clampi(c0 + 2 * i, 0, dw - 1) * 2u) |
                   (wg_ld16(display, row + (uint)clampi(c0 + 2 * i + 1, 0, dw - 1) * 2u) << 16);
        }
    }
    uint t[10];
    [unroll] for (int j = 0; j < 5; ++j) {
        t[2 * j] = q[j] & 0xffffu;
        t[2 * j + 1] = q[j] >> 16;
    }
    return uint4(lz5e_o(t[0], t[1], t[2], t[3], t[4]), lz5o_o(t[2], t[3], t[4], t[5], t[6]),
                 lz5e_o(t[3], t[4], t[5], t[6], t[7]), lz5o_o(t[5], t[6], t[7], t[8], t[9]));
}
#else
// lz5 of post.hip, profile 0: v = w0 t0, then fma per tap; med3 with the taps' range (the taps are finite)
float lz5a(float t0, float t1, float t2, float t3, float t4, float w0, float w1, float w3, float w4)
{
    float v = w0 * t0;
    v = mad(w1, t1, v);
    v = mad(kLz2, t2, v);
    v = mad(w3, t3, v);
    v = mad(w4, t4, v);
    const float mid_lo = min(min(t1, t2), t3), mid_hi = max(max(t1, t2), t3);
    const float lo = min(min(mid_lo, t0), t4), hi = max(max(mid_hi, t0), t4);
    return min(max(v, lo), hi);
}
float lz5e(float t0, float t1, float t2, float t3, float t4) { return lz5a(t0, t1, t2, t3, t4, kLz0, kLz1, kLz3, kLz4); }
float lz5o(float t0, float t1, float t2, float t3, float t4) { return lz5a(t0, t1, t2, t3, t4, kLz4, kLz3, kLz1, kLz0); }

// filter_row_c1: horizontal results of display row r, channel c, for output columns 4m ... 4m + 3 (taps 6m - 2 ... 6m + 7)
void row_c1(int pitch, int plane, int r, int m, bool fast, int c, out float h[4])
{
    const uint row = (uint)(r * pitch + c * plane) * 2u;
    const int c0 = 6 * m - 2;
    uint q[5];
    [unroll] for (int i = 0; i < 5; ++i) {
        if (fast) {   // c0 even, pitch and plane even: 4-byte aligned
            q[i] = wg_ld32(display, row + (uint)(c0 + 2 * i) * 2u);
        } else {
            q[i] = wg_ld16(display, row + (uint)clampi(c0 + 2 * i, 0, dw - 1) * 2u) |
                   (wg_ld16(display, row + (uint)clampi(c0 + 2 * i + 1, 0, dw - 1) * 2u) << 16);
        }
    }
    float t[10];
    [unroll] for (int j = 0; j < 5; ++j) {
        t[2 * j] = f16tof32(q[j]);
        t[2 * j + 1] = f16tof32(q[j] >> 16);
    }
    h[0] = lz5e(t[0], t[1], t[2], t[3], t[4]);
    h[1] = lz5o(t[2], t[3], t[4], t[5], t[6]);
    h[2] = lz5e(t[3], t[4], t[5], t[6], t[7]);
    h[3] = lz5o(t[5], t[6], t[7], t[8], t[9]);
}

float4 row4(int pitch, int plane, int r, int m, bool fast, int c)
{
    float h[4];
    row_c1(pitch, plane, r, m, fast, c, h);
    return float4(h[0], h[1], h[2], h[3]);
}
#endif

#if defined(WG_DIRECT_TEXTURES)
// framebuffer_half's clamp (compares: a NaN passes), then wg_f2h (round to nearest even, NaN kept)
float fd_half(float v)
{
    const float c = select(v > 65504.0f, 65504.0f, select(v < -65504.0f, -65504.0f, v));
    return f16tof32(wg_f2h(c));
}
#endif

#ifndef PF_MUT
#define PF_MUT 0   // 1: negative control of the store check (never shipped)
#endif
#define kS 4         // row pairs per lane (post_filter_c1_s4)
#define kGroups 10   // column groups per 32-lane half

WG_KERNEL(post_filter_c1_s4)
{
    WG_SETUP();
    g_z = (uint)ow >> 31;
#if !defined(WG_FILTER_LDR)
    float dec_scale = dec_scale_arg;
    if (expo != 0u) {
        const float e = wg_ldf(expo, 0u);
        dec_scale = (float)(0.01 / (double)(e == 0.0f ? 1.0f : e));
    }
#endif
    const int lane = (int)wg_lane_id();
    const int wl = lane & 31;
    if (wl >= 3 * kGroups) return;
    const int g = wl / 3, c = wl - 3 * g;
    const int tiles_x = (ow + 4 * kGroups - 1) / (4 * kGroups);
    const int tile = (int)wg_block_id();
    const int m = (tile % tiles_x) * kGroups + g;
    const int p0 = ((tile / tiles_x) * 2 + (lane >> 5)) * kS;
    if (4 * m >= ow || 2 * p0 >= oh) return;
    const int pitch = (dw + 63) & ~63, plane = pitch * dh;
    const bool fast = m >= 1 && 6 * m + 7 <= dw - 1;
    const int ncols = ow - 4 * m < 4 ? ow - 4 * m : 4;

    // the sliding window: horizontal results of rows 3p - 2 + k in r0 ... r6 (named vectors, shifted one by one: a
    // shifted local array was compiled into a block copy that does not keep an overlapping shift's order)
#if defined(WG_LZ_ORIG)
    uint4 r0 = 0u, r1 = 0u, r2 = 0u, r3 = 0u, r4 = 0u, r5 = 0u, r6 = row4_o(pitch, plane,
                                                                            clampi(3 * p0 - 2, 0, dh - 1), m, fast, c);
#else
    float4 r0 = 0.0f, r1 = 0.0f, r2 = 0.0f, r3 = 0.0f, r4 = 0.0f, r5 = 0.0f, r6 = row4(pitch, plane,
                                                                                       clampi(3 * p0 - 2, 0, dh - 1), m,
                                                                                       fast, c);
#endif
    [loop] for (int st = -1; st < kS; ++st) {
        const int p = p0 + st;
        if (2 * p >= oh) break;
        r0 = r3;
        r1 = r4;
        r2 = r5;
        r3 = r6;
#if defined(WG_LZ_ORIG)
        r4 = row4_o(pitch, plane, clampi(3 * p + 2, 0, dh - 1), m, fast, c);
        r5 = row4_o(pitch, plane, clampi(3 * p + 3, 0, dh - 1), m, fast, c);
        r6 = row4_o(pitch, plane, clampi(3 * p + 4, 0, dh - 1), m, fast, c);
#else
        r4 = row4(pitch, plane, clampi(3 * p + 2, 0, dh - 1), m, fast, c);
        r5 = row4(pitch, plane, clampi(3 * p + 3, 0, dh - 1), m, fast, c);
        r6 = row4(pitch, plane, clampi(3 * p + 4, 0, dh - 1), m, fast, c);
#endif
        if (st < 0) continue;
        [unroll] for (int dy = 0; dy < 2; ++dy) {
            const int y = 2 * p + dy;
            if (y >= oh) break;
            const uint o0 = (uint)(y * ow + 4 * m);
            [unroll] for (int i = 0; i < 4; ++i) {
#if defined(WG_LZ_ORIG)
                const float r = f16tof32(dy == 0 ? lz5e_o(r0[i], r1[i], r2[i], r3[i], r4[i])
                                                 : lz5o_o(r2[i], r3[i], r4[i], r5[i], r6[i]));
#else
                const float r = dy == 0 ? lz5e(r0[i], r1[i], r2[i], r3[i], r4[i]) : lz5o(r2[i], r3[i], r4[i], r5[i], r6[i]);
#endif
#if defined(WG_FILTER_LDR)
                const float v = r;
#else
                const float qc = clampf(r, k1em4, 3.0f);
                const float v = qc < kDecPolyMax ? dec_m1_poly(qc) : exp2(qc * kDecK) - 1.0f;
#endif
#if defined(WG_DIRECT_TEXTURES)
                // channels 1, 2 of this pixel from the lanes after channel 0's (the same pixel, so the same loop exits)
#if defined(WG_FILTER_LDR)
                const float val = v;
#else
                const float val = opq(v) * dec_scale;
#endif
                // each lane finishes its OWN channel value (the pre-exposure scale, the FP16 rounding) before the lane
                // gather, instead of channel 0's lane finishing all three after it: the same function on the same
                // values, one finish per lane instead of three on channel 0's lane
                float vo = val;
                if (nd_color_pi != 1.0f) vo = native_direct_scale(vo, nd_color_pi);
#if PF_MUT == 1   // negative control of the store check (never shipped): only channel 0 rounded
                if (nd_output_rne16 != 0u && c == 0) vo = fd_half(vo);
#else
                if (nd_output_rne16 != 0u) vo = fd_half(vo);
#endif
                const uint l0 = (uint)(lane - c);
                const float v1 = WaveReadLaneAt(vo, l0 + 1u), v2 = WaveReadLaneAt(vo, l0 + 2u);
                if (c == 0 && i < ncols) {
                    const float3 o = float3(vo, v1, v2);
                    native_direct_output[uint2((uint)(4 * m + i), (uint)y)] = float4(o, 1.0f);
                }
#else
#if defined(WG_FILTER_LDR)
                if (i < ncols) wg_stf(out_p, (o0 + (uint)i) * 12u + (uint)c * 4u, v);
#else
                if (i < ncols) wg_stf(out_p, (o0 + (uint)i) * 12u + (uint)c * 4u, opq(v) * dec_scale);
#endif
#endif
            }
        }
    }
}
