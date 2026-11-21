// SPDX-License-Identifier: MIT
// Frame generation (issue #96): exact float16 operations of net2's fast-path kernels (fg_net2_mconv.hlsl,
// fg_net2_upwarp.hlsl) on 8 values at once, with the bits of fg_net2.hlsli's one-value operations (fg2_mul, fg2_fma,
// fg2_add, fg2_leaky). Values are float16 bits, 8 packed in a uint4 (element i in dword i / 2, low half first).
//
// - The product of two float16 values is exact in single precision (at most 22 significant bits, never below 2^-48,
//   never a single-precision subnormal), so h(w P) is one rounding of an exact value.
// - fma(w, P, t) = w P + t with one float16 rounding: s = a + t rounded to nearest in single precision with its exact
//   residual (two-sum), made round-to-odd (wg_round_odd), then rounded once to float16 (13 spare bits: no double
//   rounding). A non-finite s passes as it is, as in wg_fma_f16.
// - The float16 roundings of 8 values are one driver-channel conversion (wg_to_f16_ags: v_cvt_f16_f32 per element,
//   nearest even); the opaque XOR keeps the driver from folding the producing operation into the conversion
//   (fg_mpair.hlsl). The conversion is a driver-channel call: uniform control flow only.
// The including file includes wg_platform.hlsli and fg_net2.hlsli first and this header after its argument block
// (fg_opq0 reads the launch's grid size); wave32.
#pragma once

uint fg_opq0() { return wg_grid_ >> 31; }      // an opaque zero (the grid size is below 2^31)

uint4 fg_pack8(uint r[8]) { return uint4(r[0] | (r[1] << 16), r[2] | (r[3] << 16), r[4] | (r[5] << 16), r[6] | (r[7] << 16)); }
uint fg_half(uint4 u, uint j)
{
    const uint d = select(j < 2u, u.x, select(j < 4u, u.y, select(j < 6u, u.z, u.w)));
    return (d >> ((j & 1u) * 16u)) & 0xffffu;
}

// 8 single-precision values -> 8 packed float16, nearest even
uint4 fg_cvt8(float m[8])
{
    v8f t;
    const uint z = fg_opq0();
    [unroll] for (uint i = 0; i < 8; ++i) t.v[i] = asfloat(asuint(m[i]) ^ z);
    return wg_to_f16_ags(t);
}

// the single-precision rounding of a + c made round-to-odd (a, c exact single-precision values)
float fg_sum_odd(float a, float c)
{
    const float s = a + c, bv = s - a;
    const float e = (a - (s - bv)) + (c - bv);
    return (asuint(s) & 0x7fffffffu) >= 0x7f800000u ? s : asfloat(wg_round_odd(s, e));
}

// h(w P) for 8 values (fg2_mul)
uint4 fg_mul8(float w, uint4 p)
{
    float m[8];
    [unroll] for (uint i = 0; i < 8; ++i) m[i] = w * wg_h2f(fg_half(p, i));
    return fg_cvt8(m);
}
// h(w_i P_i): one float16 weight per value
uint4 fg_mul8v(float w[8], uint4 p)
{
    float m[8];
    [unroll] for (uint i = 0; i < 8; ++i) m[i] = w[i] * wg_h2f(fg_half(p, i));
    return fg_cvt8(m);
}
// fma(w, P, t) for 8 values (fg2_fma; w = 1 is fg2_add)
uint4 fg_fma8(float w, uint4 p, uint4 t)
{
    float m[8];
    [unroll] for (uint i = 0; i < 8; ++i) m[i] = fg_sum_odd(w * wg_h2f(fg_half(p, i)), wg_h2f(fg_half(t, i)));
    return fg_cvt8(m);
}
// fma(w_i, P_i, t_i): one float16 weight per value
uint4 fg_fma8v(float w[8], uint4 p, uint4 t)
{
    float m[8];
    [unroll] for (uint i = 0; i < 8; ++i) m[i] = fg_sum_odd(w[i] * wg_h2f(fg_half(p, i)), wg_h2f(fg_half(t, i)));
    return fg_cvt8(m);
}
// fg2_leaky on 8 values: max(h(s v), v) by the keys (NaN unchanged)
uint4 fg_leaky8(uint4 v)
{
    const uint4 t = fg_mul8(wg_h2f(FG2_SLOPE16), v);
    uint r[8];
    [unroll] for (uint i = 0; i < 8; ++i) {
        const uint hv = fg_half(v, i), ht = fg_half(t, i);
        r[i] = select(fg_is_nan16(hv) != 0u, hv, select(fg_key16(ht) > fg_key16(hv), ht, hv));
    }
    return fg_pack8(r);
}

// net2's 2x half-pixel upsample (fg_net2_up2's bits) of 8 values at output (ox, oy) from q00 = (y0, x0), q01 =
// (y0, x1), q10 = (y1, x0), q11 = (y1, x1) of the taps tX, tY (fg2_up_tap): top = fma(wx1, P01, h(wx0 P00)), bottom
// likewise, out = fma(wy1, bottom, h(wy0 top)); the weights (0.25, 0.75, 1.25, -0.25) are float16 values
uint4 fg_up8(Fg2Tap tX, Fg2Tap tY, uint4 q00, uint4 q01, uint4 q10, uint4 q11)
{
    const float wx0 = wg_h2f(tX.w0), wx1 = wg_h2f(tX.w1), wy0 = wg_h2f(tY.w0), wy1 = wg_h2f(tY.w1);
    const uint4 top = fg_fma8(wx1, q01, fg_mul8(wx0, q00));
    const uint4 bot = fg_fma8(wx1, q11, fg_mul8(wx0, q10));
    return fg_fma8(wy1, bot, fg_mul8(wy0, top));
}
