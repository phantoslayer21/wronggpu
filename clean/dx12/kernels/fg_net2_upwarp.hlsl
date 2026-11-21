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
#include "fg_net2_fast.hlsli"

// the 8 head channels of half-resolution pixel q
uint4 fg_heads8(int q)
{
    const uint2 h4 = wg_ld64(a4, (uint)q * 8u);
    const uint h1 = wg_ld16(a1, (uint)q * 2u);
    const uint c0 = wg_ld16(a3, (uint)(q * 3) * 2u), c1 = wg_ld16(a3, (uint)(q * 3 + 1) * 2u);
    const uint c2 = wg_ld16(a3, (uint)(q * 3 + 2) * 2u);
    return uint4(h4.x, h4.y, h1 | (c0 << 16), c1 | (c2 << 16));
}

// fg_net2_warp's sample point and weights for the displacement (ux, uy) at pixel (px_, py_)
struct FgSample { int x0; int x1; int y0; int y1; float w00; float w10; float w01; float w11; };
FgSample fg_sample(uint ux, uint uy, int px_, int py_)
{
    FgSample s;
    const uint sx = fg2_fma(FG2_TWO16, ux, wg_f2h((float)px_));
    const uint sy = fg2_fma(FG2_TWO16, uy, wg_f2h((float)py_));
    const float fX0 = floor(wg_h2f(sx));
    const float fY0 = floor(wg_h2f(sy));
    const uint fx = wg_f2h(wg_h2f(sx) - fX0);
    const uint fy = wg_f2h(wg_h2f(sy) - fY0);
    const int X0 = (int)fX0;
    const int Y0 = (int)fY0;
    s.x0 = clamp(X0, 0, W - 1); s.x1 = clamp(X0 + 1, 0, W - 1);
    s.y0 = clamp(Y0, 0, H - 1); s.y1 = clamp(Y0 + 1, 0, H - 1);
    const uint gx = wg_f2h(1.0f - wg_h2f(fx));
    const uint gy = wg_f2h(1.0f - wg_h2f(fy));
    s.w00 = wg_h2f(fg2_mul(gx, gy)); s.w10 = wg_h2f(fg2_mul(fx, gy));
    s.w01 = wg_h2f(fg2_mul(gx, fy)); s.w11 = wg_h2f(fg2_mul(fx, fy));
    return s;
}

// 4 channels of Ib at (yy, xx) and 4 of Ic at (zy, zx) as 8 values
uint4 fg_pair8(int yy, int xx, int zy, int zx)
{
    const uint2 b = wg_ld64(ib, (uint)((yy * W + xx) * 4) * 2u);
    const uint2 c = wg_ld64(ic, (uint)((zy * W + zx) * 4) * 2u);
    return uint4(b.x, b.y, c.x, c.y);
}

WG_KERNEL(fg_net2_upwarp)
{
    WG_SETUP();
    const int idx = (int)(wg_block_id() * 256u + wg_lane_id());
    const bool live = idx < W * H;
    const int p = min(idx, W * H - 1);
    const int py_ = p / W, px_ = p - (p / W) * W;
    const int wi = W / 2, hi = H / 2;
    const Fg2Tap tX = fg2_up_tap(px_, wi), tY = fg2_up_tap(py_, hi);
    const uint4 U = fg_up8(tX, tY, fg_heads8(tY.i0 * wi + tX.i0), fg_heads8(tY.i0 * wi + tX.i1),
                           fg_heads8(tY.i1 * wi + tX.i0), fg_heads8(tY.i1 * wi + tX.i1));

    const FgSample sb = fg_sample(fg_half(U, 0u), fg_half(U, 1u), px_, py_);
    const FgSample sc = fg_sample(fg_half(U, 2u), fg_half(U, 3u), px_, py_);
    float w00[8], w10[8], w01[8], w11[8];
    [unroll] for (int i = 0; i < 4; ++i) {
        w00[i] = sb.w00; w10[i] = sb.w10; w01[i] = sb.w01; w11[i] = sb.w11;
        w00[i + 4] = sc.w00; w10[i + 4] = sc.w10; w01[i + 4] = sc.w01; w11[i + 4] = sc.w11;
    }
    const uint4 p00 = fg_pair8(sb.y0, sb.x0, sc.y0, sc.x0), p01 = fg_pair8(sb.y0, sb.x1, sc.y0, sc.x1);
    const uint4 p10 = fg_pair8(sb.y1, sb.x0, sc.y1, sc.x0), p11 = fg_pair8(sb.y1, sb.x1, sc.y1, sc.x1);
    uint4 acc = fg_fma8v(w10, p01, fg_mul8v(w00, p00));
    acc = fg_fma8v(w01, p10, acc);
    const uint4 val = fg_fma8v(w11, p11, acc);

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
