// SPDX-License-Identifier: MIT
// Frame generation (issue #96): net2's warp-merge (clean/fg/reference/net2.py warp_merge / warp_sample), exact, one output
// element per lane: y [H][W][18] float16 = Ib warped by (U4 0, U4 1), Ic warped by (U4 2, U4 3), U4 0..3, U1, Ia, U3 0..2,
// h(f). The warp sample of pixel (x, y): px = fma(2, u, h(x)); X0 = floor(px), fx = h(px - X0); x0, x1 = X0, X0 + 1
// clamped to [0, W - 1] (y likewise); gx = h(1 - fx); w00 = h(gx gy), w10 = h(fx gy), w01 = h(gx fy), w11 = h(fx fy);
// value = fma(w11, P11, fma(w01, P10, fma(w10, P01, h(w00 P00)))) with P01 = (y0, x1), P10 = (y1, x0). The single-
// precision steps (px - X0, 1 - fx, the weight products) are exact; each h is one wg_f2h. Ib, Ic [H][W][4], Ia, U1
// [H][W][1], U4 [H][W][4], U3 [H][W][3]. Built only when --only names it.
//
// @wg-kernel fg_net2_warp co=fg_net2.co block=256 wave=32 args=p,p,p,p,p,p,p,u,u,u
#include "wg_platform.hlsli"
#include "fg_net2.hlsli"

WG_ARGS_BEGIN
    uint ib, ic, ia, u4, u1, u3, y;   // pointers (arena byte offsets)
    int W, H;
    uint fbits;
WG_ARGS_END

WG_KERNEL(fg_net2_warp)
{
    WG_SETUP();
    const int idx = (int)(wg_block_id() * 256u + wg_lane_id());
    if (idx >= W * H * 18) return;
    const int p = idx / 18;
    const int c = idx - p * 18;
    const int py_ = p / W;
    const int px_ = p - py_ * W;
    uint v;
    if (c < 8) {
        const uint img = c < 4 ? ib : ic;
        const int ch = c & 3;
        const uint ux = wg_ld16(u4, (uint)(p * 4 + (c < 4 ? 0 : 2)) * 2u);
        const uint uy = wg_ld16(u4, (uint)(p * 4 + (c < 4 ? 1 : 3)) * 2u);
        const uint sx = fg2_fma(FG2_TWO16, ux, wg_f2h((float)px_));
        const uint sy = fg2_fma(FG2_TWO16, uy, wg_f2h((float)py_));
        const float fX0 = floor(wg_h2f(sx));
        const float fY0 = floor(wg_h2f(sy));
        const uint fx = wg_f2h(wg_h2f(sx) - fX0);
        const uint fy = wg_f2h(wg_h2f(sy) - fY0);
        const int X0 = (int)fX0;
        const int Y0 = (int)fY0;
        const int x0 = clamp(X0, 0, W - 1), x1 = clamp(X0 + 1, 0, W - 1);
        const int y0 = clamp(Y0, 0, H - 1), y1 = clamp(Y0 + 1, 0, H - 1);
        const uint gx = wg_f2h(1.0f - wg_h2f(fx));
        const uint gy = wg_f2h(1.0f - wg_h2f(fy));
        const uint w00 = fg2_mul(gx, gy), w10 = fg2_mul(fx, gy), w01 = fg2_mul(gx, fy), w11 = fg2_mul(fx, fy);
        const uint q00 = wg_ld16(img, (uint)((y0 * W + x0) * 4 + ch) * 2u);
        const uint q01 = wg_ld16(img, (uint)((y0 * W + x1) * 4 + ch) * 2u);
        const uint q10 = wg_ld16(img, (uint)((y1 * W + x0) * 4 + ch) * 2u);
        const uint q11 = wg_ld16(img, (uint)((y1 * W + x1) * 4 + ch) * 2u);
        uint acc = fg2_fma(w10, q01, fg2_mul(w00, q00));
        acc = fg2_fma(w01, q10, acc);
        v = fg2_fma(w11, q11, acc);
    } else if (c < 12) {
        v = wg_ld16(u4, (uint)(p * 4 + c - 8) * 2u);
    } else if (c == 12) {
        v = wg_ld16(u1, (uint)p * 2u);
    } else if (c == 13) {
        v = wg_ld16(ia, (uint)p * 2u);
    } else if (c < 17) {
        v = wg_ld16(u3, (uint)(p * 3 + c - 14) * 2u);
    } else {
        v = wg_f2h(asfloat(fbits));
    }
    wg_st16(y, (uint)idx * 2u, v);
}
