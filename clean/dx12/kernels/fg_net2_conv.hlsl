// SPDX-License-Identifier: GPL-3.0-only
// Frame generation (issue #96): net2's 3 x 3 convolution layer (clean/fg/reference/net2.py conv3 and its post-ops), exact,
// one output element per lane. Tensors are float16 [H][W][C] (channel innermost) with a channel stride and offset, so a
// launch can read one group of a wider tensor and write one channel range of a wider output.
//
// x: input at the conv grid H x W, channels x[(y W + x) in_cs + in_co + c] for c < cin (channels cin .. cw - 1 read 0);
// w: weights [cout][3][3][cw] (out, ky, kx, in); b: bias [cout]. The lane computes weight output o = co_lo + j
// (j < co_n): float16 accumulator from b[o]; chunks per tap (ky, kx) in row-major order, inside a tap ascending blocks
// of 16 weight channels (the last may be shorter); each chunk exact with one rounding (fg_exact16.h); zero border.
// Post-ops in order: leaky ReLU (leaky = 1), clamp to [-6, 6] when o >= clamp_from, then the residual:
// res 0 none, 1 h(v + 0), 2 fma(res_mul, skip[(y W + x) skip_cs + skip_co + j], v) (res_mul float16 bits). pool = 1:
// the conv runs at H x W and the output is the 2 x 2 average h(h(h(h(a00 + a01) + a10) + a11) * 0.25) at H/2 x W/2.
// The result goes to y[(oy Wo + ox) out_cs + out_co + j]. Built only when --only names it.
//
// @wg-kernel fg_net2_conv co=fg_net2.co block=256 wave=32 args=p,p,p,p,p,u,u,u,u,u,u,u,u,u,u,u,u,u,u,u,u,u
#include "wg_platform.hlsli"
#include "fg_net2.hlsli"

WG_ARGS_BEGIN
    uint x, w, b, y, skip;     // pointers (arena byte offsets)
    int in_cs, in_co, cin, cw, co_lo, co_n, out_cs, out_co, H, W, pool, leaky, clamp_from, res;
    uint res_mul;
    int skip_cs, skip_co;
WG_ARGS_END

uint fg2_conv_at(int o, int cy, int cx)
{
    uint acc = wg_ld16(b, (uint)o * 2u);
    const int nb = (cw + 15) / 16;
    [loop] for (int t = 0; t < 9; ++t) {
        const int ky = t / 3;
        const int kx = t - ky * 3;
        const int sy = cy + ky - 1;
        const int sx = cx + kx - 1;
        const bool inside = sy >= 0 && sy < H && sx >= 0 && sx < W;
        const uint wrow = (uint)(((o * 3 + ky) * 3 + kx) * cw);
        const uint xrow = inside ? (uint)((sy * W + sx) * in_cs + in_co) : 0u;
        [loop] for (int cb = 0; cb < nb; ++cb) {
            FgAcc s = fg_acc_begin(acc);
            const int c1 = min(cw, cb * 16 + 16);
            [loop] for (int c = cb * 16; c < c1; ++c) {
                const uint wv = wg_ld16(w, (wrow + (uint)c) * 2u);
                uint xv = 0u;                                  // outside the grid or past cin: 0 (0 * inf is NaN)
                if (inside && c < cin) xv = wg_ld16(x, (xrow + (uint)c) * 2u);
                s = fg_acc_add(s, wv, xv);
            }
            acc = fg_acc_end(s);
        }
    }
    return acc;
}

WG_KERNEL(fg_net2_conv)
{
    WG_SETUP();
    const int Ho = pool != 0 ? H / 2 : H;
    const int Wo = pool != 0 ? W / 2 : W;
    const int idx = (int)(wg_block_id() * 256u + wg_lane_id());
    if (idx >= Ho * Wo * co_n) return;
    const int pix = idx / co_n;
    const int j = idx - pix * co_n;
    const int oy = pix / Wo;
    const int ox = pix - oy * Wo;
    const int o = co_lo + j;
    const int npos = pool != 0 ? 4 : 1;
    uint sum = 0u;
    [loop] for (int k = 0; k < npos; ++k) {
        const int cy = pool != 0 ? 2 * oy + (k >> 1) : oy;
        const int cx = pool != 0 ? 2 * ox + (k & 1) : ox;
        uint v = fg2_conv_at(o, cy, cx);
        if (leaky != 0) v = fg2_leaky(v);
        if (o >= clamp_from) v = fg2_clamp6(v);
        if (res == 1) v = fg2_plus0(v);
        if (res == 2) v = fg2_fma(res_mul, wg_ld16(skip, (uint)((cy * W + cx) * skip_cs + skip_co + j) * 2u), v);
        sum = k == 0 ? v : fg2_add(sum, v);
    }
    if (pool != 0) sum = fg2_mul(sum, FG2_QUARTER16);
    wg_st16(y, (uint)((oy * Wo + ox) * out_cs + out_co + j) * 2u, sum);
}
