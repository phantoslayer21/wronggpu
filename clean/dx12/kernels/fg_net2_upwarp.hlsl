// SPDX-License-Identifier: MIT
// Frame generation (issue #96): net2's head upsample (launch 15) and warp-merge (launch 16) in one launch for the fast
// path, exact: the bits of fg_net2_up2 on A4 / A1 / A3 and of fg_net2_warp, one pixel of the W x H grid per lane.
// - The three heads are 8 channels per half-resolution pixel (A4 0..3, A1, A3 0..2), upsampled together (fg_up8 of
//   fg_net2_fast.hlsli) and written to U4 [H][W][4], U1 [H][W][1], U3 [H][W][3].
// - The warp sample of fg_net2_warp (the same single-value operations for the sample point and the four weights),
//   then the bilinear of Ib (by U4 0, 1) and Ic (by U4 2, 3) as one 8-value chain with a weight per value:
//   value = fma(w11, P11, fma(w01, P10, fma(w10, P01, h(w00 P00)))) (fg_mul8v / fg_fma8v).
// - y [H][W][18] = the 8 warped values, U4, U1, Ia, U3, h(f) (fbits: f's single-precision bits).
// Every lane runs the conversions (a lane past the grid computes the last pixel and stores nothing). Ib, Ic
// [H][W][4], Ia [H][W][1]; A4 / A1 / A3 at H / 2 x W / 2. Built only when --only names it.
//
// @wg-kernel fg_net2_upwarp co=fg_net2_mma.co block=256 wave=32 args=p,p,p,p,p,p,p,p,p,p,u,u,u
#include "wg_platform.hlsli"
#include "fg_net2.hlsli"

WG_ARGS_BEGIN
    uint ib, ic, ia, a4, a1, a3, u4, u1, u3, y;    // pointers (arena byte offsets)
    int W, H;
    uint fbits;
WG_ARGS_END
#define FG_NATIVE_UP 1
#include "fg_net2_fast.hlsli"

#include "fg_net2_upwarp.hlsli"

WG_KERNEL(fg_net2_upwarp)
{
    WG_SETUP();
    const int idx = (int)(wg_block_id() * 256u + wg_lane_id());
    const bool live = idx < W * H;
    const int p = min(idx, W * H - 1);
    uint4 U, val;
    fg_upwarp_eval(p, U, val);

    const uint iav = wg_ld16(ia, (uint)p * 2u);
    const uint hf = wg_f2h(asfloat(fbits));
    if (live) {
        wg_st64(u4, (uint)p * 8u, U.xy);
        wg_st16(u1, (uint)p * 2u, fg_half(U, 4u));
        wg_st16(u3, (uint)(p * 3) * 2u, fg_half(U, 5u));
        wg_st16(u3, (uint)(p * 3 + 1) * 2u, fg_half(U, 6u));
        wg_st16(u3, (uint)(p * 3 + 2) * 2u, fg_half(U, 7u));
        wg_st128(y, (uint)p * 36u, val);
        wg_st128(y, (uint)p * 36u + 16u, uint4(U.x, U.y, fg_half(U, 4u) | (iav << 16),
                                               fg_half(U, 5u) | (fg_half(U, 6u) << 16)));
        wg_st32(y, (uint)p * 36u + 32u, fg_half(U, 7u) | (hf << 16));
    }
}
