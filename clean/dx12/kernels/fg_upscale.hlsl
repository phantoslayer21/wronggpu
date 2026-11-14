// SPDX-License-Identifier: MIT
// D3D12 port of frame generation's net1 upscale (issue #96): fg_upscale_f16 of fg_net1.co (clean/fg/kernels/fg_net1.hip),
// byte for byte. Bilinear, align corners, x [C][hi][wi] -> y [C][ho][wo] float16. rx, ry: the ratios (wi - 1) / (wo - 1)
// and (hi - 1) / (ho - 1) as single-precision bits (0 for an extent of 1), from the caller, so no division runs here.
// Per axis: s = r * o, i0 = trunc(s), i1 = trunc(s + 1) (both n_in - 1 when i1 >= n_in), w1 = h(s - i0),
// w0 = h((i0 - s) + 1). top = h(h(wx0 P00) + h(wx1 P01)), bottom on row y1, out = h(h(wy0 top) + h(wy1 bottom)): each
// single-precision operation as written (one IEEE operation each, build_dx12.py's exact mode), each float16 rounding
// explicit (wg_f2h, nearest even). NaN results are written as 0x7E00. Built only when --only names it.
//
// @wg-kernel fg_upscale_f16 co=fg_net1.co block=256 wave=32 args=p,p,u,u,u,u,u,u,u
#include "wg_platform.hlsli"

#define FG_FN
#define FG_I64 int64_t
#define FG_FBH32(v) firstbithigh((uint)(v))
#include "../../fg/kernels/fg_exact16.h"

WG_ARGS_BEGIN
    uint x, y;           // pointers (arena byte offsets)
    int C, hi, wi, ho, wo;
    uint rx_bits, ry_bits;
WG_ARGS_END

struct FgTap { int i0; int i1; float w0; float w1; };

FgTap fg_axis(float r, int o, int n_in)
{
    FgTap t;
    const float s = r * (float)o;
    t.i0 = (int)s;
    t.i1 = (int)(s + 1.0f);
    if (t.i1 >= n_in) { t.i0 = n_in - 1; t.i1 = n_in - 1; }
    t.w1 = wg_h2f(wg_f2h(s - (float)t.i0));
    t.w0 = wg_h2f(wg_f2h(((float)t.i0 - s) + 1.0f));
    return t;
}

float fg_lerp16(float wa, float a, float wb, float b)
{
    return wg_h2f(wg_f2h(wg_h2f(wg_f2h(wa * a)) + wg_h2f(wg_f2h(wb * b))));
}

WG_KERNEL(fg_upscale_f16)
{
    WG_SETUP();
    const int idx = (int)(wg_block_id() * 256u + wg_lane_id());
    if (idx >= C * ho * wo) return;
    const int c = idx / (ho * wo);
    const int rem = idx - c * (ho * wo);
    const int oy = rem / wo;
    const int ox = rem - oy * wo;
    const FgTap tx = fg_axis(asfloat(rx_bits), ox, wi);
    const FgTap ty = fg_axis(asfloat(ry_bits), oy, hi);
    const uint base = (uint)(c * hi * wi);
    const float p00 = wg_h2f(wg_ld16(x, (base + (uint)(ty.i0 * wi + tx.i0)) * 2u));
    const float p01 = wg_h2f(wg_ld16(x, (base + (uint)(ty.i0 * wi + tx.i1)) * 2u));
    const float p10 = wg_h2f(wg_ld16(x, (base + (uint)(ty.i1 * wi + tx.i0)) * 2u));
    const float p11 = wg_h2f(wg_ld16(x, (base + (uint)(ty.i1 * wi + tx.i1)) * 2u));
    const float top = fg_lerp16(tx.w0, p00, tx.w1, p01);
    const float bot = fg_lerp16(tx.w0, p10, tx.w1, p11);
    uint outv = wg_f2h(fg_lerp16(ty.w0, top, ty.w1, bot));
    if (fg_is_nan16(outv) != 0u) outv = FG_NAN16;
    wg_st16(y, (uint)idx * 2u, outv);
}
