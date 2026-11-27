// SPDX-License-Identifier: GPL-3.0-only
// Frame generation (issue #96): net2's input merge (clean/fg/reference/net2.py merge_in), one pixel per lane:
// x [npx][10] float16 = Ib [npx][4], Ic [npx][4], Ia [npx][1], then the single-precision scalar f (bits fbits) rounded to
// float16 (nearest even, wg_f2h). Built only when --only names it.
//
// @wg-kernel fg_net2_merge_pixels co=fg_net2.co block=256 wave=32 args=p,p,p,p,u,u
#include "wg_platform.hlsli"

WG_ARGS_BEGIN
    uint ib, ic, ia, x;  // pointers (arena byte offsets)
    int npx;
    uint fbits;
WG_ARGS_END

WG_KERNEL(fg_net2_merge_pixels)
{
    WG_SETUP();
    const int p = (int)(wg_block_id() * WG_BLOCK + wg_lane_id());
    if (p >= npx) return;
    const uint rgba = (uint)p * 8u;
    const uint2 b = wg_ld64(ib, rgba), c = wg_ld64(ic, rgba);
    const uint a = wg_ld16(ia, (uint)p * 2u);
    const uint f = wg_f2h(asfloat(fbits));
    const uint o = (uint)p * 20u;
    wg_st128(x, o, uint4(b.x, b.y, c.x, c.y));
    wg_st32(x, o + 16u, a | (f << 16));
}
