// SPDX-License-Identifier: MIT
// Frame generation (issue #96): HIP kernels of net1's operations (clean/fg/reference/net1.py), bit for bit against the
// CPU reference. D3D12 ports: clean/dx12/kernels/fg_*.hlsl. Tensors are float16, channel-first [c][h][w]; weights are
// natural [out][in][k][k] float16; biases [out] float16. 1-D grid of 256-lane workgroups, one output element per lane.
#include "wg_platform.h"
#include "wg_mem.h"

typedef unsigned int uint;
typedef unsigned short u16;
#define FG_FN WG_FN
#define FG_I64 long long
#define FG_FBH32(v) (31u - (uint)__builtin_clz(v))
#include "fg_exact16.h"

// Single-precision steps are one IEEE operation each, as written (the upscale's positions): no contraction.
#pragma clang fp contract(off)

WG_FN uint fg_f2h(float f) { return (uint)__builtin_bit_cast(u16, (_Float16)f); }      // round to nearest even
WG_FN float fg_h2f(uint h) { return (float)__builtin_bit_cast(_Float16, (u16)h); }     // exact

// fg_conv_f16: one convolution of net1. Stride 1, zero border (k = 3) or none (k = 1), cross-correlation; the float16
// accumulator starts at the bias and takes 16-term chunks (16 input channels of one tap), each summed exactly and added
// with one float16 rounding, in the chunk order `order`: 0 ascending (1 x 1: channel blocks in order), 1 channel block
// outer, then kx, then ky; 2 kx outer, then channel block, then ky. act: 0 none, 1 leaky ReLU (slope 0x211E).
// cin must be a multiple of 16. Output element idx = (o * H + y) * W + x.
WG_KERNEL_N(fg_conv_f16, 256)(const u16* __restrict__ x, const u16* __restrict__ w, const u16* __restrict__ b,
                              u16* __restrict__ y, int cin, int cout, int H, int W, int k, int order, int act) {
    const int idx = (int)(wg_block_id() * 256u + wg_lane_id());
    if (idx >= cout * H * W) return;
    const int o = idx / (H * W);
    const int rem = idx - o * (H * W);
    const int yy = rem / W;
    const int xx = rem - yy * W;
    const int p = k / 2;
    const int nb = cin / 16;
    const int taps = k * k;
    uint acc = b[o];
    for (int n = 0; n < nb * taps; ++n) {
        int cb, kx, ky;
        if (order == 0) {
            cb = n; kx = 0; ky = 0;
        } else if (order == 1) {
            cb = n / taps;
            const int t = n - cb * taps;
            kx = t / k;
            ky = t - kx * k;
        } else {
            kx = n / (nb * k);
            const int r = n - kx * (nb * k);
            cb = r / k;
            ky = r - cb * k;
        }
        const int sy = yy + ky - p, sx = xx + kx - p;
        const bool inside = sy >= 0 && sy < H && sx >= 0 && sx < W;
        FgAcc s = fg_acc_begin(acc);
        for (int i = 0; i < 16; ++i) {
            const int c = cb * 16 + i;
            const uint wv = w[((o * cin + c) * k + ky) * k + kx];
            const uint xv = inside ? (uint)x[(c * H + sy) * W + sx] : 0u;   // outside the grid: 0 (0 * inf is NaN)
            s = fg_acc_add(s, wv, xv);
        }
        acc = fg_acc_end(s);
    }
    if (act == 1) acc = fg_leaky16(acc);
    y[idx] = (u16)acc;
}

// fg_maxpool2_f16: net1's 2 x 2 max-pool, stride 2, floor sizes (Ho = H / 2, Wo = W / 2): from the first tap (dy, dx) =
// (0, 0), then (0, 1), (1, 0), (1, 1), the delivered step m <- (x > m) * x + (x <= m) * m in float16 (fg_pool_step16).
// x [C][H][W], y [C][Ho][Wo]; output element idx = (c * Ho + oy) * Wo + ox.
WG_KERNEL_N(fg_maxpool2_f16, 256)(const u16* __restrict__ x, u16* __restrict__ y, int C, int H, int W) {
    const int Ho = H / 2, Wo = W / 2;
    const int idx = (int)(wg_block_id() * 256u + wg_lane_id());
    if (idx >= C * Ho * Wo) return;
    const int c = idx / (Ho * Wo);
    const int rem = idx - c * (Ho * Wo);
    const int oy = rem / Wo;
    const int ox = rem - oy * Wo;
    const u16* p = x + (c * H + 2 * oy) * W + 2 * ox;
    uint m = p[0];
    m = fg_pool_step16(m, p[1]);
    m = fg_pool_step16(m, p[W]);
    m = fg_pool_step16(m, p[W + 1]);
    y[idx] = (u16)m;
}

// One axis of the align-corners rule: s = r * o (single precision), i0 = trunc(s), i1 = trunc(s + 1), both n_in - 1 when
// i1 >= n_in; weights w1 = h(s - i0), w0 = h((i0 - s) + 1) as float16 values.
struct FgTap { int i0, i1; float w0, w1; };
WG_FN FgTap fg_axis(float r, int o, int n_in) {
    FgTap t;
    const float s = r * (float)o;
    t.i0 = (int)s;
    t.i1 = (int)(s + 1.0f);
    if (t.i1 >= n_in) { t.i0 = n_in - 1; t.i1 = n_in - 1; }
    t.w1 = fg_h2f(fg_f2h(s - (float)t.i0));
    t.w0 = fg_h2f(fg_f2h(((float)t.i0 - s) + 1.0f));
    return t;
}

// h(h(wa * a) + h(wb * b)): float16 products (exact in single precision, then rounded) and a float16 sum (single
// precision then float16: the same as one rounding of the exact sum, 24 >= 2 * 11 + 2)
WG_FN float fg_lerp16(float wa, float a, float wb, float b) {
    return fg_h2f(fg_f2h(fg_h2f(fg_f2h(wa * a)) + fg_h2f(fg_f2h(wb * b))));
}

// fg_upscale_f16: net1's bilinear upscale, align corners, x [C][hi][wi] -> y [C][ho][wo] float16. rx, ry: the ratios
// (wi - 1) / (wo - 1) and (hi - 1) / (ho - 1) as single-precision bits (0 for an extent of 1), computed by the caller:
// top = h(h(wx0 P[y0][x0]) + h(wx1 P[y0][x1])), bottom the same on row y1, out = h(h(wy0 top) + h(wy1 bottom)).
// NaN results are written as 0x7E00. Output element idx = (c * ho + oy) * wo + ox.
WG_KERNEL_N(fg_upscale_f16, 256)(const u16* __restrict__ x, u16* __restrict__ y, int C, int hi, int wi, int ho, int wo,
                                 uint rx_bits, uint ry_bits) {
    const int idx = (int)(wg_block_id() * 256u + wg_lane_id());
    if (idx >= C * ho * wo) return;
    const int c = idx / (ho * wo);
    const int rem = idx - c * (ho * wo);
    const int oy = rem / wo;
    const int ox = rem - oy * wo;
    const FgTap tx = fg_axis(__builtin_bit_cast(float, rx_bits), ox, wi);
    const FgTap ty = fg_axis(__builtin_bit_cast(float, ry_bits), oy, hi);
    const u16* p = x + c * hi * wi;
    const float top = fg_lerp16(tx.w0, fg_h2f(p[ty.i0 * wi + tx.i0]), tx.w1, fg_h2f(p[ty.i0 * wi + tx.i1]));
    const float bot = fg_lerp16(tx.w0, fg_h2f(p[ty.i1 * wi + tx.i0]), tx.w1, fg_h2f(p[ty.i1 * wi + tx.i1]));
    uint out = fg_f2h(fg_lerp16(ty.w0, top, ty.w1, bot));
    if (fg_is_nan16(out) != 0u) out = FG_NAN16;
    y[idx] = (u16)out;
}
