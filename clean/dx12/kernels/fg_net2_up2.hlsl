// SPDX-License-Identifier: GPL-3.0-only
// Frame generation (issue #96): net2's 2x bilinear upsample with half-pixel centres (clean/fg/reference/net2.py up2),
// exact, one output element per lane. x [hi][wi][C] float16 -> y [2 hi][2 wi][C]. Per axis (single precision, exact here):
// s = (o - 0.5) * 0.5, i0 = trunc(s), i1 = trunc(s + 1), both n - 1 when i1 >= n, both 0 when s < 0, w1 = h(s - i0),
// w0 = h((i0 - s) + 1). top = fma(wx1, P01, h(wx0 P00)), bottom likewise on row y1, out = fma(wy1, bottom, h(wy0 top)),
// each fma with one float16 rounding (fg_net2.hlsli). Built only when --only names it.
//
// @wg-kernel fg_net2_up2 co=fg_net2.co block=256 wave=32 args=p,p,u,u,u
#include "wg_platform.hlsli"
#include "fg_net2.hlsli"

WG_ARGS_BEGIN
    uint x, y;           // pointers (arena byte offsets)
    int C, hi, wi;
WG_ARGS_END

WG_KERNEL(fg_net2_up2)
{
    WG_SETUP();
    const int ho = 2 * hi;
    const int wo = 2 * wi;
    const int idx = (int)(wg_block_id() * 256u + wg_lane_id());
    if (idx >= ho * wo * C) return;
    const int pix = idx / C;
    const int c = idx - pix * C;
    const int oy = pix / wo;
    const int ox = pix - oy * wo;
    const Fg2Tap tx = fg2_up_tap(ox, wi);
    const Fg2Tap ty = fg2_up_tap(oy, hi);
    const uint p00 = wg_ld16(x, (uint)((ty.i0 * wi + tx.i0) * C + c) * 2u);
    const uint p01 = wg_ld16(x, (uint)((ty.i0 * wi + tx.i1) * C + c) * 2u);
    const uint p10 = wg_ld16(x, (uint)((ty.i1 * wi + tx.i0) * C + c) * 2u);
    const uint p11 = wg_ld16(x, (uint)((ty.i1 * wi + tx.i1) * C + c) * 2u);
    const uint top = fg2_fma(tx.w1, p01, fg2_mul(tx.w0, p00));
    const uint bot = fg2_fma(tx.w1, p11, fg2_mul(tx.w0, p10));
    wg_st16(y, (uint)idx * 2u, fg2_fma(ty.w1, bot, fg2_mul(ty.w0, top)));
}
