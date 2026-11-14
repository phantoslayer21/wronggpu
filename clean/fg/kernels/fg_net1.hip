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
