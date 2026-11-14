// SPDX-License-Identifier: MIT
// D3D12 port of frame generation's matrix convolution (issue #96; tier B of the net1 fast path): fg_mconv<MT> of
// fg_net1_mma.co (clean/fg/kernels/fg_net1_mma.hip), byte-identical to HIP: the same fragments and chunk order,
// v_wmma_f32_16x16x16_f16 through the driver channel (wg_wmma_f16) with the float16 accumulator as C, its result rounded
// to float16 after every chunk (wg_f2h = v_cvt_f16_f32, nearest even), the exact integer leaky ReLU and pool
// (fg_exact16.h). Driver rules (clean/dx12/README.md): every load of a chunk before its first matrix call; branchless
// selects; the chunk loop bound through an opaque zero (no forced unrolling); store pointers through wg_uniform().
// Built only when --only names the kernels (fg_ kernels are in no model's set).
//
// @wg-kernel fg_mconv1 co=fg_net1_mma.co entry=fg_mconv block=256 wave=32 args=p,p,p,p,p,u,u,u,u,u,u,u,u,u,u defines=FG_MT=1
// @wg-kernel fg_mconv2 co=fg_net1_mma.co entry=fg_mconv block=256 wave=32 args=p,p,p,p,p,u,u,u,u,u,u,u,u,u,u defines=FG_MT=2
// @wg-kernel fg_mconv4 co=fg_net1_mma.co entry=fg_mconv block=256 wave=32 args=p,p,p,p,p,u,u,u,u,u,u,u,u,u,u defines=FG_MT=4
#include "wg_platform.hlsli"

#define FG_FN
#define FG_I64 int64_t
#define FG_FBH32(v) firstbithigh((uint)(v))
#include "../../fg/kernels/fg_exact16.h"

#ifndef FG_MT
#define FG_MT 4
#endif

WG_ARGS_BEGIN
    uint x, wf, b, y, p;  // pointers (arena byte offsets)
    int cin, cout, H, W, k, order, act, pool, tiles_x, tiles_y;
WG_ARGS_END

uint fg_opq0() { return wg_grid_ >> 31; }      // an opaque zero (the grid size is below 2^31)

uint4 fg_pack8(uint r[8]) { return uint4(r[0] | (r[1] << 16), r[2] | (r[3] << 16), r[4] | (r[5] << 16), r[6] | (r[7] << 16)); }

WG_KERNEL(fg_mconv)
{
    WG_SETUP();
    const uint lane = wg_lane_id() & 31u, wave = wg_lane_id() >> 5;
    const int blk = (int)wg_block_id();
    const int tx = blk % tiles_x, rest = blk / tiles_x;
    const int ty = rest % tiles_y, mg = rest / tiles_y;
    const int n = (int)(lane & 15u), hh = (int)(lane >> 4);
    const int px = tx * 16 + n;
    const int py0 = ty * 16 + 2 * (int)wave;
    const int pad = k / 2, nb = cin / 16, taps = k * k, nq = nb * taps;
    const uint py = (uint)wg_uniform((int)y), pp = (uint)wg_uniform((int)p);

    v8f a32[FG_MT][2];
    [unroll] for (int mt = 0; mt < FG_MT; ++mt) {
        const int o0 = (mg * FG_MT + mt) * 16 + 8 * hh;
        [unroll] for (uint v = 0; v < 8; ++v) {
            const float bv = wg_h2f(wg_ld16(b, (uint)(o0 + (int)v) * 2u));
            a32[mt][0].v[v] = bv;
            a32[mt][1].v[v] = bv;
        }
    }
    const int nq_loop = nq + (int)fg_opq0();
    for (int q = 0; q < nq_loop; ++q) {
        // the chunk's (channel block, ky, kx) in the layer's order, branchless
        const int t1 = q % taps, cb1 = q / taps;
        const int kx2 = q / (nb * k), r2 = q % (nb * k);
        const int cb = select(order == 0, q, select(order == 1, cb1, r2 / k));
        const int kx = select(order == 0, 0, select(order == 1, t1 / k, kx2));
        const int ky = select(order == 0, 0, select(order == 1, t1 % k, r2 % k));
        uint4 fb[2], fa[FG_MT];
        const int sx = px + kx - pad;
        [unroll] for (int nt = 0; nt < 2; ++nt) {
            const int sy = py0 + nt + ky - pad;
            const bool inb = and(and(sx >= 0, sx < W), and(sy >= 0, sy < H));
            const uint e = select(inb, (uint)((sy * W + sx) * cin + cb * 16 + 8 * hh), 0u);
            const uint4 u = wg_ld128(x, e * 2u);
            fb[nt] = uint4(select(inb, u.x, 0u), select(inb, u.y, 0u), select(inb, u.z, 0u), select(inb, u.w, 0u));
        }
        [unroll] for (int mt2 = 0; mt2 < FG_MT; ++mt2)
            fa[mt2] = wg_ld128(wf, (uint)((((mg * FG_MT + mt2) * nq + q) * 32 + (int)lane) * 16));
        [unroll] for (int mt3 = 0; mt3 < FG_MT; ++mt3)
            [unroll] for (int nt2 = 0; nt2 < 2; ++nt2) {
                const v8f d = wg_wmma_f16(fa[mt3], fb[nt2], a32[mt3][nt2]);
                [unroll] for (uint v3 = 0; v3 < 8; ++v3) a32[mt3][nt2].v[v3] = wg_h2f(wg_f2h(d.v[v3]));
            }
    }
    [unroll] for (int mt4 = 0; mt4 < FG_MT; ++mt4) {
        const int o0 = (mg * FG_MT + mt4) * 16 + 8 * hh;
        uint r0[8], r1[8];
        [unroll] for (uint v4 = 0; v4 < 8; ++v4) {
            const uint h0 = wg_f2h(a32[mt4][0].v[v4]), h1 = wg_f2h(a32[mt4][1].v[v4]);
            r0[v4] = select(act == 1, fg_leaky16(h0), h0);
            r1[v4] = select(act == 1, fg_leaky16(h1), h1);
        }
        if (px < W && py0 < H) wg_st128(py, (uint)((py0 * W + px) * cout + o0) * 2u, fg_pack8(r0));
        if (px < W && py0 + 1 < H) wg_st128(py, (uint)(((py0 + 1) * W + px) * cout + o0) * 2u, fg_pack8(r1));
        if (pool == 1) {
            uint m[8];
            [unroll] for (uint v5 = 0; v5 < 8; ++v5) {
                const uint x01 = wg_xor_u(r0[v5], 1u), x11 = wg_xor_u(r1[v5], 1u);
                uint t = r0[v5];
                t = fg_pool_step16(t, x01);
                t = fg_pool_step16(t, r1[v5]);
                t = fg_pool_step16(t, x11);
                m[v5] = t;
            }
            const int Ho = H / 2, Wo = W / 2, ox = px / 2, oy = py0 / 2;
            if ((n & 1) == 0 && ox < Wo && oy < Ho) wg_st128(pp, (uint)((oy * Wo + ox) * cout + o0) * 2u, fg_pack8(m));
        }
    }
}
