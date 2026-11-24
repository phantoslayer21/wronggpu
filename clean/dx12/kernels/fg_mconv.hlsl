// SPDX-License-Identifier: MIT
// D3D12 port of frame generation's matrix convolution (issue #96; tier B of the net1 fast path): fg_mconv<MT> of
// fg_net1_mma.co (clean/fg/kernels/fg_net1_mma.hip), byte-identical to HIP: the same fragments and chunk order,
// v_wmma_f32_16x16x16_f16 through the driver channel (wg_wmma_f16) with the float16 accumulator as C, its result rounded
// to float16 after every chunk (wg_to_f16_ags, nearest even), followed by the exact leaky ReLU and pool. The
// BlockKxKy variants stage one channel block's 3 x 3 halo in LDS. Workgroups of up to 128 lanes unroll the nine
// taps within each staged channel block. Other paths keep the chunk loop bound opaque. Issue all loads for a
// chunk before its first matrix call and keep store pointers uniform (clean/dx12/README.md).
// Built only when --only names the kernels (fg_ kernels are in no model's set).
//
// @wg-kernel fg_mconv1 co=fg_net1_mma.co entry=fg_mconv block=256 wave=32 args=p,p,p,p,p,u,u,u,u,u,u,u,u,u,u defines=FG_MT=1,WG_LD_A16=1
// @wg-kernel fg_mconv2 co=fg_net1_mma.co entry=fg_mconv block=256 wave=32 args=p,p,p,p,p,u,u,u,u,u,u,u,u,u,u defines=FG_MT=2,WG_LD_A16=1,FG_HALO=1
// @wg-kernel fg_mconv4 co=fg_net1_mma.co entry=fg_mconv block=256 wave=32 args=p,p,p,p,p,u,u,u,u,u,u,u,u,u,u defines=FG_MT=4,WG_LD_A16=1
// @wg-kernel fg_mconv2_narrow co=fg_net1_mma.co entry=fg_mconv block=64 wave=32 args=p,p,p,p,p,u,u,u,u,u,u,u,u,u,u defines=FG_MT=2,WG_LD_A16=1,FG_HALO=1
// @wg-kernel fg_mconv1_late co=fg_net1_mma.co entry=fg_mconv block=64 wave=32 args=p,p,p,p,p,u,u,u,u,u,u,u,u,u,u defines=FG_MT=1,WG_LD_A16=1,FG_HALO=1
// @wg-kernel fg_mconv2_mid co=fg_net1_mma.co entry=fg_mconv block=128 wave=32 args=p,p,p,p,p,u,u,u,u,u,u,u,u,u,u defines=FG_MT=2,WG_LD_A16=1,FG_HALO=1
#include "wg_platform.hlsli"

#define FG_FN
#define FG_I64 int64_t
#define FG_FBH32(v) firstbithigh((uint)(v))
#include "../../fg/kernels/fg_exact16.h"

#ifndef FG_MT
#define FG_MT 4
#endif
#define FG_TH (2 * (WG_BLOCK / 32))
#ifndef FG_HALO
#define FG_HALO 0
#endif
#if FG_HALO
#define FG_HALO_PX (18 * (FG_TH + 2))
groupshared uint4 fg_halo[2 * FG_HALO_PX];
#endif

WG_ARGS_BEGIN
    uint x, wf, b, y, p;  // pointers (arena byte offsets)
    int cin, cout, H, W, k, order, act, pool, tiles_x, tiles_y;
WG_ARGS_END

uint fg_opq0() { return wg_grid_ >> 31; }      // an opaque zero (the grid size is below 2^31)

uint4 fg_pack8(uint r[8]) { return uint4(r[0] | (r[1] << 16), r[2] | (r[3] << 16), r[4] | (r[5] << 16), r[6] | (r[7] << 16)); }
uint fg_half(uint4 u, uint j)
{
    const uint d = select(j < 2u, u.x, select(j < 4u, u.y, select(j < 6u, u.z, u.w)));
    return (d >> ((j & 1u) * 16u)) & 0xffffu;
}
// 8 packed float16 values -> the float32 accumulator fragment (exact; mask-free low halves via f16tof32)
v8f fg_widen(uint4 hp)
{
    v8f r;
    r.v[0] = f16tof32(hp.x); r.v[1] = wg_h2f_hi(hp.x);
    r.v[2] = f16tof32(hp.y); r.v[3] = wg_h2f_hi(hp.y);
    r.v[4] = f16tof32(hp.z); r.v[5] = wg_h2f_hi(hp.z);
    r.v[6] = f16tof32(hp.w); r.v[7] = wg_h2f_hi(hp.w);
    return r;
}
// Software-pipelined Tier B chunks across (mt = 0..FG_MT-1, nt = 0..1): overlaps fg_widen (8x v_cvt_f32_f16)
// and wg_to_f16_ags (8x v_cvt_f16_f32) directly with independent wg_wmma_f16 ops (only 2 v8f live; bit-identical).
void fg_chunk_b_pipe(uint4 fa[FG_MT], uint4 fb[2], inout uint4 h16[FG_MT][2])
{
    const int N = FG_MT * 2;
    v8f w_next = fg_widen(h16[0][0]);
    v8f d_prev = w_next;
    [unroll] for (int s = 0; s < N; ++s) {
        const v8f d_cur = wg_wmma_f16(fa[s >> 1], fb[s & 1], w_next);
        if (s > 0) h16[(s - 1) >> 1][(s - 1) & 1] = wg_to_f16_ags(d_prev);
        if (s + 1 < N) w_next = fg_widen(h16[(s + 1) >> 1][(s + 1) & 1]);
        d_prev = d_cur;
    }
    h16[(N - 1) >> 1][(N - 1) & 1] = wg_to_f16_ags(d_prev);
}

#if FG_HALO
void fg_stage_halo(int tx, int ty, int cb)
{
    for (int i = (int)wg_lane_id(); i < 2 * FG_HALO_PX; i += WG_BLOCK) {
        const int p = i / 2, frag = i % 2;
        const int sx = tx * 16 + p % 18 - 1, sy = ty * FG_TH + p / 18 - 1;
        const bool inside = sx >= 0 && sx < W && sy >= 0 && sy < H;
        const uint e = inside ? (uint)((sy * W + sx) * cin + cb * 16 + frag * 8) : 0u;
        fg_halo[frag * FG_HALO_PX + p] = wg_ld128(x, e * 2u) & select(inside, ~0u, 0u);
    }
    wg_barrier_lds();
}
#endif

#if FG_HALO && WG_BLOCK <= 128
void fg_accumulate_halo(int tx, int ty, int mg, int n, int hh, int wy, uint lane, inout uint4 h16[FG_MT][2])
{
    const int nb = cin / 16;
    const int nq = nb * 9;
    const int pixel = hh * FG_HALO_PX + wy * 18 + n;
    for (int cb = 0; cb < nb; ++cb) {
        // Finish reading this block's final tap before overwriting the halo.
        if (cb != 0) wg_barrier_lds();
        fg_stage_halo(tx, ty, cb);
        // Software-pipeline tap loads with WMMAs: prefetch tap+1's weight/halo fragments before tap's WMMAs.
        uint4 fb[2], fa[FG_MT];
        {
            fb[0] = fg_halo[pixel];
            fb[1] = fg_halo[pixel + 18];
            [unroll] for (int mt = 0; mt < FG_MT; ++mt)
                fa[mt] = wg_ld128(wf, (uint)((((mg * FG_MT + mt) * nq + cb * 9) * 32 + (int)lane) * 16));
        }
        [unroll] for (int tap = 0; tap < 9; ++tap) {
            uint4 fb_n[2], fa_n[FG_MT];
            if (tap + 1 < 9) {
                const int nt = tap + 1;
                const int qn = cb * 9 + nt;
                const int ppn = pixel + 18 * (nt % 3) + nt / 3;
                fb_n[0] = fg_halo[ppn];
                fb_n[1] = fg_halo[ppn + 18];
                [unroll] for (int mt = 0; mt < FG_MT; ++mt)
                    fa_n[mt] = wg_ld128(wf, (uint)((((mg * FG_MT + mt) * nq + qn) * 32 + (int)lane) * 16));
            }
            fg_chunk_b_pipe(fa, fb, h16);
            if (tap + 1 < 9) {
                fb[0] = fb_n[0]; fb[1] = fb_n[1];
                [unroll] for (int mt2 = 0; mt2 < FG_MT; ++mt2) fa[mt2] = fa_n[mt2];
            }
        }
    }
}
#endif

void fg_accumulate(int tx, int ty, int mg, int n, int hh, int wy, uint lane,
                   bool staged, inout uint4 h16[FG_MT][2])
{
    const int px = tx * 16 + n, py0 = ty * FG_TH + wy;
    const int pad = k / 2, nb = cin / 16, nq = nb * k * k;
    const int nq_loop = nq + (int)fg_opq0();
    const bool one_tap = order == 0, block_first = order == 1;
    int cb = 0, ky = 0, kx = 0;
    for (int q = 0; q < nq_loop; ++q) {
#if FG_HALO
        if (staged && kx == 0 && ky == 0) {
            // The old block's final tap must finish reading before the next block overwrites it.
            if (cb != 0) wg_barrier_lds();
            fg_stage_halo(tx, ty, cb);
        }
#endif
        uint4 fb[2], fa[FG_MT];
        const int sx = px + kx - pad;
        [unroll] for (int nt = 0; nt < 2; ++nt) {
#if FG_HALO
            if (staged) {
                fb[nt] = fg_halo[hh * FG_HALO_PX + (wy + nt + ky) * 18 + n + kx];
            } else
#endif
            {
                const int sy = py0 + nt + ky - pad;
                const bool inb = and(and(sx >= 0, sx < W), and(sy >= 0, sy < H));
                const uint e = select(inb, (uint)((sy * W + sx) * cin + cb * 16 + 8 * hh), 0u);
                const uint4 u = wg_ld128(x, e * 2u);
                fb[nt] = u & select(inb, ~0u, 0u);
            }
        }
        [unroll] for (int mt2 = 0; mt2 < FG_MT; ++mt2)
            fa[mt2] = wg_ld128(wf, (uint)((((mg * FG_MT + mt2) * nq + q) * 32 + (int)lane) * 16));
        fg_chunk_b_pipe(fa, fb, h16);

        // ky varies fastest in both spatial orders. The middle digit is kx for
        // block_kx_ky and cb for kx_block_ky. The 1 x 1 path just advances cb.
        const bool next_ky = ky + 1 == k;
        const bool next_kx = kx + 1 == k;
        const bool next_cb = cb + 1 == nb;
        const bool step_x = !one_tap && next_ky && (block_first || next_cb);
        const bool step_c = one_tap || (next_ky && (!block_first || next_kx));
        ky = select(one_tap || next_ky, 0, ky + 1);
        kx = select(step_x, select(next_kx, 0, kx + 1), kx);
        cb = select(step_c, select(!one_tap && !block_first && next_cb, 0, cb + 1), cb);
    }
}

uint fg_leaky2(uint packed)
{
    const half2 slope = asfloat16((uint16_t)FG_LEAKY_SLOPE16);
    const half2 negzero = asfloat16((uint16_t)(0x8000u | fg_opq0()));
    const uint scaled = wg_h2_bits(mad(slope, wg_as_h2(packed), negzero));
    const uint lo = packed & 0xffffu, hi = packed >> 16;
    // Positive values, zeros and NaNs keep their incoming bits. For negative numbers the scaled value wins.
    const uint mask = select(lo > 0x8000u && lo <= 0xfc00u, 0xffffu, 0u) |
                      select(hi > 0x8000u && hi <= 0xfc00u, 0xffff0000u, 0u);
    return (packed & ~mask) | (scaled & mask);
}

uint4 fg_leaky8(uint4 bits)
{
    return uint4(fg_leaky2(bits.x), fg_leaky2(bits.y), fg_leaky2(bits.z), fg_leaky2(bits.w));
}

// A finite maximum with +0 ranked above -0 gives the same zero sign as the spec's two products and sum.
// A NaN or any negative infinity is invalid, as is a second positive infinity (0 * infinity).
uint fg_ordered_half(uint bits)
{
    return bits ^ select(bits < 0x8000u, 0x8000u, 0xffffu);
}

uint fg_pool4(uint a, uint b, uint c, uint d)
{
    const uint ka = fg_ordered_half(a), kb = fg_ordered_half(b);
    const uint kc = fg_ordered_half(c), kd = fg_ordered_half(d);
    const uint top = max(max(ka, kb), max(kc, kd));
    const bool invalid = min(min(ka, kb), min(kc, kd)) <= 0x03ffu ||
                         top > 0xfc00u ||
                         ((uint)(a == 0x7c00u) + (uint)(b == 0x7c00u) + (uint)(c == 0x7c00u) + (uint)(d == 0x7c00u) > 1u);
    return select(invalid, FG_NAN16, top ^ select(top >= 0x8000u, 0x8000u, 0xffffu));
}

WG_KERNEL(fg_mconv)
{
    WG_SETUP();
    const uint lane = wg_lane_id() & 31u, wave = wg_lane_id() >> 5;
    const int blk = (int)wg_block_id();
    const int tx = blk % tiles_x, rest = blk / tiles_x;
    const int ty = rest % tiles_y, mg = rest / tiles_y;
    const int n = (int)(lane & 15u), hh = (int)(lane >> 4);
    const int px = tx * 16 + n;
    const int wy = 2 * (int)wave, py0 = ty * FG_TH + wy;
    const uint py = (uint)wg_uniform((int)y), pp = (uint)wg_uniform((int)p);

    uint4 h16[FG_MT][2];                         // one packed float16 accumulator per output fragment
    [unroll] for (int mt = 0; mt < FG_MT; ++mt) {
        const int o0 = (mg * FG_MT + mt) * 16 + 8 * hh;
        h16[mt][0] = wg_ld128(b, (uint)o0 * 2u);
        h16[mt][1] = h16[mt][0];
    }
#if FG_HALO
    if (k == 3 && order == 1) {
#if WG_BLOCK <= 128
        fg_accumulate_halo(tx, ty, mg, n, hh, wy, lane, h16);
#else
        fg_accumulate(tx, ty, mg, n, hh, wy, lane, true, h16);
#endif
    } else
#endif
    {
        fg_accumulate(tx, ty, mg, n, hh, wy, lane, false, h16);
    }
    [unroll] for (int mt4 = 0; mt4 < FG_MT; ++mt4) {
        const int o0 = (mg * FG_MT + mt4) * 16 + 8 * hh;
        const uint4 r0 = act == 1 ? fg_leaky8(h16[mt4][0]) : h16[mt4][0];
        const uint4 r1 = act == 1 ? fg_leaky8(h16[mt4][1]) : h16[mt4][1];
        if (px < W && py0 < H) wg_st128(py, (uint)((py0 * W + px) * cout + o0) * 2u, r0);
        if (px < W && py0 + 1 < H) wg_st128(py, (uint)(((py0 + 1) * W + px) * cout + o0) * 2u, r1);
        if (pool == 1) {
            uint m[8];
            [unroll] for (uint v5 = 0; v5 < 8; ++v5) {
                const uint h0 = fg_half(r0, v5), h1 = fg_half(r1, v5);
                m[v5] = fg_pool4(h0, wg_xor_u(h0, 1u), h1, wg_xor_u(h1, 1u));
            }
            const int Ho = H / 2, Wo = W / 2, ox = px / 2, oy = py0 / 2;
            if ((n & 1) == 0 && ox < Wo && oy < Ho) wg_st128(pp, (uint)((oy * Wo + ox) * cout + o0) * 2u, fg_pack8(m));
        }
    }
}
