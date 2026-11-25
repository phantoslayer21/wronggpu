// SPDX-License-Identifier: MIT
// Frame generation (issue #96): net2's head upsample + warp-merge of ONE pixel of the W x H grid (fg_net2_upwarp's lane work),
// shared by fg_net2_upwarp (which stores y, U4, U1, U3) and fg_net2_mconv_m1b2_uw (which builds L14's input tile from it in
// LDS, fg_net2_mconv.hlsl FG_UPWARP_STAGE): one definition, the same bits. The including file defines the argument block
// (ib, ic, ia, a4, a1, a3, W, H: arena offsets and the grid size), FG_NATIVE_UP = 1, and includes wg_platform.hlsli,
// fg_net2.hlsli and fg_net2_fast.hlsli first.
#pragma once

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

// The taps have fixed FP16 weights on the 2x grid; preserve the extrapolation at pixel zero.
Fg2Tap fg_upwarp_tap(int o, int n)
{
    const bool odd = (o & 1) != 0;
    const int i = (o >> 1) - (int)!odd;
    Fg2Tap t;
    t.i0 = clamp(i, 0, n - 1);
    t.i1 = min(i + 1, n - 1);
    t.w0 = select(o == 0, 0x3d00u, select(odd, 0x3a00u, FG2_QUARTER16));
    t.w1 = select(o == 0, 0xb400u, select(odd, FG2_QUARTER16, 0x3a00u));
    return t;
}

// The upsampled heads U (8 values: A4 0..3, A1, A3 0..2) and the warped 8 values val of pixel p (0 <= p < W x H), as
// fg_net2_upwarp computes them; every lane of the wave takes part (the upsample and the conversions are wave operations).
void fg_upwarp_eval(int p, out uint4 U, out uint4 val)
{
    const int py_ = p / W, px_ = p - (p / W) * W;
    const int wi = W / 2, hi = H / 2;
    const Fg2Tap tX = fg_upwarp_tap(px_, wi), tY = fg_upwarp_tap(py_, hi);
    const uint4 top = fg_up_axis8(tX, fg_heads8(tY.i0 * wi + tX.i0), fg_heads8(tY.i0 * wi + tX.i1));
    const uint4 bottom = fg_up_axis8(tX, fg_heads8(tY.i1 * wi + tX.i0), fg_heads8(tY.i1 * wi + tX.i1));
    U = fg_up_axis8(tY, top, bottom);

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
    val = fg_fma8v(w11, p11, acc);
}
