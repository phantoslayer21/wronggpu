// SPDX-License-Identifier: GPL-3.0-only
// D3D12 port of frame generation's net1 convolution (issue #96): fg_conv_f16 of fg_net1.co (clean/fg/kernels/fg_net1.hip),
// byte for byte. The arithmetic is the shared integer code of clean/fg/kernels/fg_exact16.h (exact float16 chunk
// accumulation, one nearest-even rounding per chunk, the leaky ReLU), so both back ends run the same operations; only
// memory access differs (byte offsets into the arena here).
//
// One output element per lane: idx = (o * H + y) * W + x of [cout][H][W] float16; x [cin][H][W] float16; w natural
// [cout][cin][k][k] float16; b [cout] float16. Stride 1, zero border; chunk order: 0 ascending (1 x 1), 1 channel block,
// kx, ky; 2 kx, channel block, ky. act: 0 none, 1 leaky ReLU. Not part of Model M's or Model L's kernel set (fg_ kernels
// are built only when --only names them).
//
// @wg-kernel fg_conv_f16 co=fg_net1.co block=256 wave=32 args=p,p,p,p,u,u,u,u,u,u,u
#include "wg_platform.hlsli"

#define FG_FN
#define FG_I64 int64_t
#define FG_FBH32(v) firstbithigh((uint)(v))
#include "../../fg/kernels/fg_exact16.h"

WG_ARGS_BEGIN
    uint x, w, b, y;     // pointers (arena byte offsets)
    int cin, cout, H, W, k, order, act;
WG_ARGS_END

WG_KERNEL(fg_conv_f16)
{
    WG_SETUP();
    const int idx = (int)(wg_block_id() * 256u + wg_lane_id());
    if (idx >= cout * H * W) return;
    const int o = idx / (H * W);
    const int rem = idx - o * (H * W);
    const int yy = rem / W;
    const int xx = rem - yy * W;
    const int p = k / 2;
    const int nb = cin / 16;
    const int taps = k * k;
    uint acc = wg_ld16(b, (uint)o * 2u);
    [loop] for (int n = 0; n < nb * taps; ++n) {
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
        const int sy = yy + ky - p;
        const int sx = xx + kx - p;
        const bool inside = sy >= 0 && sy < H && sx >= 0 && sx < W;
        FgAcc s = fg_acc_begin(acc);
        [loop] for (int i = 0; i < 16; ++i) {
            const int c = cb * 16 + i;
            const uint wv = wg_ld16(w, (uint)(((o * cin + c) * k + ky) * k + kx) * 2u);
            uint xv = 0u;                                   // outside the grid: 0 (0 * inf is NaN)
            if (inside) xv = wg_ld16(x, (uint)((c * H + sy) * W + sx) * 2u);
            s = fg_acc_add(s, wv, xv);
        }
        acc = fg_acc_end(s);
    }
    if (act == 1) acc = fg_leaky16(acc);
    wg_st16(y, (uint)idx * 2u, acc);
}
