// SPDX-License-Identifier: GPL-3.0-only
// Frame generation (issue #96): net2's input merge (clean/fg/reference/net2.py merge_in), one output element per lane:
// x [npx][10] float16 = Ib [npx][4], Ic [npx][4], Ia [npx][1], then the single-precision scalar f (bits fbits) rounded to
// float16 (nearest even, wg_f2h). Built only when --only names it.
//
// @wg-kernel fg_net2_merge co=fg_net2.co block=256 wave=32 args=p,p,p,p,u,u
#include "wg_platform.hlsli"
#include "fg_net2.hlsli"

WG_ARGS_BEGIN
    uint ib, ic, ia, x;  // pointers (arena byte offsets)
    int npx;
    uint fbits;
WG_ARGS_END

WG_KERNEL(fg_net2_merge)
{
    WG_SETUP();
    const int idx = (int)(wg_block_id() * 256u + wg_lane_id());
    if (idx >= npx * 10) return;
    const int p = idx / 10;
    const int c = idx - p * 10;
    uint v;
    if (c < 4) v = wg_ld16(ib, (uint)(p * 4 + c) * 2u);
    else if (c < 8) v = wg_ld16(ic, (uint)(p * 4 + c - 4) * 2u);
    else if (c == 8) v = wg_ld16(ia, (uint)p * 2u);
    else v = wg_f2h(asfloat(fbits));
    wg_st16(x, (uint)idx * 2u, v);
}
