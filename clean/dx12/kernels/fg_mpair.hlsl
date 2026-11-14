// SPDX-License-Identifier: MIT
// D3D12 port of frame generation's fused 1 x 1 pairs (issue #96; tier B): fg_mpair<MODE, C1T, C2T, NT, LAZY> of
// fg_net1_mma.co (clean/fg/kernels/fg_net1_mma.hip), byte-identical to HIP. Two chained 1 x 1 convolutions in registers
// (the first's result fragment is the second's input fragment), each chunk one v_wmma_f32_16x16x16_f16 through the driver
// channel with float16 rounding after it; MODE 0: input buffer (bottleneck); MODE 1: net1's upscale of the coarse input
// plus the skip, computed per input fragment with the exact upscale / add arithmetic. Driver rules (clean/dx12/README.md):
// loads before the matrix calls of a group, branchless selects, runtime loop bounds through an opaque zero, the store
// pointer through wg_uniform(). Built only when --only names the kernels.
//
// @wg-kernel fg_mpair_b co=fg_net1_mma.co entry=fg_mpair block=256 wave=32 args=p,p,p,p,p,p,p,u,u,u,u,u,u,u,u defines=FG_MODE=0,FG_C1T=32,FG_C2T=16,FG_NT=1,FG_LAZY=1
// @wg-kernel fg_mpair_d4 co=fg_net1_mma.co entry=fg_mpair block=256 wave=32 args=p,p,p,p,p,p,p,u,u,u,u,u,u,u,u defines=FG_MODE=1,FG_C1T=16,FG_C2T=8,FG_NT=1,FG_LAZY=1,FG_CINT=16
// @wg-kernel fg_mpair_d3 co=fg_net1_mma.co entry=fg_mpair block=256 wave=32 args=p,p,p,p,p,p,p,u,u,u,u,u,u,u,u defines=FG_MODE=1,FG_C1T=8,FG_C2T=8,FG_NT=1,FG_LAZY=0
// @wg-kernel fg_mpair_d2 co=fg_net1_mma.co entry=fg_mpair block=256 wave=32 args=p,p,p,p,p,p,p,u,u,u,u,u,u,u,u defines=FG_MODE=1,FG_C1T=8,FG_C2T=4,FG_NT=1,FG_LAZY=0
// @wg-kernel fg_mpair_d1 co=fg_net1_mma.co entry=fg_mpair block=256 wave=32 args=p,p,p,p,p,p,p,u,u,u,u,u,u,u,u defines=FG_MODE=1,FG_C1T=4,FG_C2T=2,FG_NT=2,FG_LAZY=0
// @wg-kernel fg_mpair_d0 co=fg_net1_mma.co entry=fg_mpair block=256 wave=32 args=p,p,p,p,p,p,p,u,u,u,u,u,u,u,u defines=FG_MODE=1,FG_C1T=2,FG_C2T=1,FG_NT=2,FG_LAZY=0
#include "wg_platform.hlsli"

#define FG_FN
#define FG_I64 int64_t
#define FG_FBH32(v) firstbithigh((uint)(v))
#include "../../fg/kernels/fg_exact16.h"
#define FG_F2H(f) wg_f2h(f)
#define FG_H2F(h) wg_h2f(h)
#define FG_SEL(c, a, b) select(c, a, b)
#include "../../fg/kernels/fg_fast16.h"

#ifndef FG_MODE
#define FG_MODE 1
#define FG_C1T 2
#define FG_C2T 1
#define FG_NT 2
#define FG_LAZY 0
#endif

WG_ARGS_BEGIN
    uint x, a, wf1, b1, wf2, b2, y;   // pointers (arena byte offsets)
    int C, H, W, hi, wi;
    uint rx, ry;
    int act2;
WG_ARGS_END

uint fg_opq0() { return wg_grid_ >> 31; }      // an opaque zero (the grid size is below 2^31)

struct FgTap { int i0; int i1; float w0; float w1; };

FgTap fg_axis(float r, int o, int n_in)
{
    FgTap t;
    const float s = r * (float)o;
    t.i0 = (int)s;
    t.i1 = (int)(s + 1.0f);
    if (t.i1 >= n_in) { t.i0 = n_in - 1; t.i1 = n_in - 1; }
    t.w1 = wg_h2f(wg_f2h(s - (float)t.i0));
    t.w0 = wg_h2f(wg_f2h(((float)t.i0 - s) + 1.0f));
    return t;
}

float fg_lerp16(float wa, float av, float wb, float bv)
{
    return wg_h2f(wg_f2h(wg_h2f(wg_f2h(wa * av)) + wg_h2f(wg_f2h(wb * bv))));
}

uint fg_half(uint4 u, uint j)
{
    const uint d = select(j < 2u, u.x, select(j < 4u, u.y, select(j < 6u, u.z, u.w)));
    return (d >> ((j & 1u) * 16u)) & 0xffffu;
}
uint4 fg_pack8(uint r[8]) { return uint4(r[0] | (r[1] << 16), r[2] | (r[3] << 16), r[4] | (r[5] << 16), r[6] | (r[7] << 16)); }

// 8 single-precision values -> 8 packed float16 (nearest even) by one driver-channel conversion (v_cvt_f16_f32 per
// element, HIP's (_Float16)). The opaque XOR keeps the driver from folding the producing operation into the
// conversion (README "Driver exceptions": a folded product becomes v_fma_mixlo_f16 x, y, 0, which turns -0 into +0).
uint4 fg_f2h8(float m[8])
{
    v8f t;
    const uint z = fg_opq0();
    [unroll] for (uint i = 0; i < 8; ++i) t.v[i] = asfloat(asuint(m[i]) ^ z);
    return wg_to_f16_ags(t);
}

// h(h(wa * a_j) + h(wb * b_j)) for the 8 channels of two packed fragments (fg_lerp16 per channel)
uint4 fg_lerp16x8(float wa, uint4 av, float wb, uint4 bv)
{
    float m[8];
    [unroll] for (uint i = 0; i < 8; ++i) m[i] = wa * wg_h2f(fg_half(av, i));
    const uint4 pa = fg_f2h8(m);
    [unroll] for (uint i2 = 0; i2 < 8; ++i2) m[i2] = wb * wg_h2f(fg_half(bv, i2));
    const uint4 pb = fg_f2h8(m);
    [unroll] for (uint i3 = 0; i3 < 8; ++i3) m[i3] = wg_h2f(fg_half(pa, i3)) + wg_h2f(fg_half(pb, i3));
    return fg_f2h8(m);
}

// the 8 channels with every NaN as 0x7E00
uint4 fg_canon8(uint4 v)
{
    uint r[8];
    [unroll] for (uint i = 0; i < 8; ++i) { const uint h = fg_half(v, i); r[i] = select(fg_is_nan16(h) != 0u, FG_NAN16, h); }
    return fg_pack8(r);
}

// one input fragment: channels c0 .. c0 + 7 of pixel pc (MODE 1: net1's upscale of x plus the skip a, the arithmetic of
// fg_upscale_f16 and fg_add_f16, with the float16 roundings of the 8 channels done together)
uint4 fg_pair_in(int pc, int c0, FgTap tx, FgTap ty)
{
#if FG_MODE == 0
    return wg_ld128(x, (uint)(pc * C + c0) * 2u);
#else
    const uint4 q00 = wg_ld128(x, (uint)((ty.i0 * wi + tx.i0) * C + c0) * 2u);
    const uint4 q01 = wg_ld128(x, (uint)((ty.i0 * wi + tx.i1) * C + c0) * 2u);
    const uint4 q10 = wg_ld128(x, (uint)((ty.i1 * wi + tx.i0) * C + c0) * 2u);
    const uint4 q11 = wg_ld128(x, (uint)((ty.i1 * wi + tx.i1) * C + c0) * 2u);
    const uint4 sk = wg_ld128(a, (uint)(pc * C + c0) * 2u);
    const uint4 top = fg_lerp16x8(tx.w0, q00, tx.w1, q01);
    const uint4 bot = fg_lerp16x8(tx.w0, q10, tx.w1, q11);
    const uint4 u = fg_canon8(fg_lerp16x8(ty.w0, top, ty.w1, bot));
    float m[8];
    [unroll] for (uint i = 0; i < 8; ++i) m[i] = wg_h2f(fg_half(u, i)) + wg_h2f(fg_half(sk, i));
    return fg_canon8(fg_f2h8(m));
#endif
}

// 8 packed float16 values -> the float32 accumulator fragment (exact)
v8f fg_widen(uint4 hp)
{
    v8f r;
    [unroll] for (uint v = 0; v < 8; ++v) r.v[v] = wg_h2f(fg_half(hp, v));
    return r;
}

// One chunk (tier B): the matrix instruction, then the whole fragment rounded to float16 by the driver channel's
// conversion (v_cvt_f16_f32, nearest even; see fg_mconv.hlsl). hp keeps the packed float16 accumulator.
void fg_chunk_b(uint4 av, uint4 bv, inout v8f c, inout uint4 hp)
{
    const v8f d = wg_wmma_f16(av, bv, c);
    hp = wg_to_f16_ags(d);
    c = fg_widen(hp);
}

// the packed float16 accumulator, leaky ReLU if act
uint4 fg_acc_out(uint4 hp, int act)
{
    uint r[8];
    [unroll] for (uint v = 0; v < 8; ++v) {
        const uint hb = fg_half(hp, v);
        r[v] = select(act == 1, fg_leaky_fast16(hb), hb);
    }
    return fg_pack8(r);
}

// the bias of output tile mt as packed float16
uint4 fg_bias8(uint bp, int mt, int hh)
{
    uint bb[8];
    [unroll] for (uint v = 0; v < 8; ++v) bb[v] = wg_ld16(bp, (uint)(16 * mt + 8 * hh + (int)v) * 2u);
    return fg_pack8(bb);
}

WG_KERNEL(fg_mpair)
{
    WG_SETUP();
    const uint lane = wg_lane_id() & 31u, wave = wg_lane_id() >> 5;
    const int n = (int)(lane & 15u), hh = (int)(lane >> 4);
    const int p0 = ((int)wg_block_id() * 8 + (int)wave) * 16 * FG_NT;
    const int nq = C / 16;
    const uint py = (uint)wg_uniform((int)y);
    // every lane computes a pixel inside the grid (the last one for lanes past the end); only the store tests the range
    int pc[FG_NT];
    FgTap tx[FG_NT], ty[FG_NT];
    [unroll] for (int nt = 0; nt < FG_NT; ++nt) {
        pc[nt] = min(p0 + 16 * nt + n, H * W - 1);
        const int yy = pc[nt] / W, xx = pc[nt] - (pc[nt] / W) * W;
        tx[nt] = fg_axis(asfloat(rx), xx, wi);
        ty[nt] = fg_axis(asfloat(ry), yy, hi);
    }
    v8f acc2[FG_C2T][FG_NT];
    uint4 h2[FG_C2T][FG_NT];
    [unroll] for (int mt = 0; mt < FG_C2T; ++mt) {
        const uint4 bh = fg_bias8(b2, mt, hh);
        [unroll] for (int nt = 0; nt < FG_NT; ++nt) { h2[mt][nt] = bh; acc2[mt][nt] = fg_widen(bh); }
    }
#if FG_LAZY == 0
    v8f acc1[FG_C1T][FG_NT];
    uint4 h1[FG_C1T][FG_NT];
    [unroll] for (int mt1 = 0; mt1 < FG_C1T; ++mt1) {
        const uint4 bh = fg_bias8(b1, mt1, hh);
        [unroll] for (int nt1 = 0; nt1 < FG_NT; ++nt1) { h1[mt1][nt1] = bh; acc1[mt1][nt1] = fg_widen(bh); }
    }
    const int nq_loop = nq + (int)fg_opq0();
    for (int cb = 0; cb < nq_loop; ++cb) {
        uint4 fb[FG_NT], fa[FG_C1T];
        [unroll] for (int nt2 = 0; nt2 < FG_NT; ++nt2) fb[nt2] = fg_pair_in(pc[nt2], cb * 16 + 8 * hh, tx[nt2], ty[nt2]);
        [unroll] for (int mt2 = 0; mt2 < FG_C1T; ++mt2)
            fa[mt2] = wg_ld128(wf1, (uint)(((mt2 * nq + cb) * 32 + (int)lane) * 16));
        [unroll] for (int mt3 = 0; mt3 < FG_C1T; ++mt3)
            [unroll] for (int nt3 = 0; nt3 < FG_NT; ++nt3) fg_chunk_b(fa[mt3], fb[nt3], acc1[mt3][nt3], h1[mt3][nt3]);
    }
    uint4 e[FG_C1T][FG_NT];
    [unroll] for (int mt4 = 0; mt4 < FG_C1T; ++mt4)
        [unroll] for (int nt4 = 0; nt4 < FG_NT; ++nt4) e[mt4][nt4] = fg_acc_out(h1[mt4][nt4], 1);
    [unroll] for (int cb2 = 0; cb2 < FG_C1T; ++cb2) {
        uint4 fa2[FG_C2T];
        [unroll] for (int mt5 = 0; mt5 < FG_C2T; ++mt5)
            fa2[mt5] = wg_ld128(wf2, (uint)(((mt5 * FG_C1T + cb2) * 32 + (int)lane) * 16));
        [unroll] for (int mt6 = 0; mt6 < FG_C2T; ++mt6)
            [unroll] for (int nt6 = 0; nt6 < FG_NT; ++nt6) fg_chunk_b(fa2[mt6], e[cb2][nt6], acc2[mt6][nt6], h2[mt6][nt6]);
    }
#else
#ifdef FG_CINT
    // the input fragments (upscale + add) once, FG_CINT channel blocks (= C / 16), instead of once per output tile
    uint4 sc[FG_CINT][FG_NT];
    [unroll] for (int ci = 0; ci < FG_CINT; ++ci)
        [unroll] for (int nti = 0; nti < FG_NT; ++nti) sc[ci][nti] = fg_pair_in(pc[nti], ci * 16 + 8 * hh, tx[nti], ty[nti]);
#endif
    const int c1t_loop = FG_C1T + (int)fg_opq0();
    for (int cb2 = 0; cb2 < c1t_loop; ++cb2) {
        v8f a1[FG_NT];
        uint4 hl[FG_NT];
        const uint4 bh1 = fg_bias8(b1, cb2, hh);
        [unroll] for (int nt7 = 0; nt7 < FG_NT; ++nt7) { hl[nt7] = bh1; a1[nt7] = fg_widen(bh1); }
#ifdef FG_CINT
        [unroll] for (int cb = 0; cb < FG_CINT; ++cb) {
            const uint4 fa = wg_ld128(wf1, (uint)(((cb2 * FG_CINT + cb) * 32 + (int)lane) * 16));
            [unroll] for (int nt9 = 0; nt9 < FG_NT; ++nt9) fg_chunk_b(fa, sc[cb][nt9], a1[nt9], hl[nt9]);
        }
#else
        const int nq_loop = nq + (int)fg_opq0();
        for (int cb = 0; cb < nq_loop; ++cb) {
            uint4 fb[FG_NT];
            [unroll] for (int nt8 = 0; nt8 < FG_NT; ++nt8) fb[nt8] = fg_pair_in(pc[nt8], cb * 16 + 8 * hh, tx[nt8], ty[nt8]);
            const uint4 fa = wg_ld128(wf1, (uint)(((cb2 * nq + cb) * 32 + (int)lane) * 16));
            [unroll] for (int nt9 = 0; nt9 < FG_NT; ++nt9) fg_chunk_b(fa, fb[nt9], a1[nt9], hl[nt9]);
        }
#endif
        uint4 eb[FG_NT];
        [unroll] for (int nt10 = 0; nt10 < FG_NT; ++nt10) eb[nt10] = fg_acc_out(hl[nt10], 1);
        uint4 fa2[FG_C2T];
        [unroll] for (int mt11 = 0; mt11 < FG_C2T; ++mt11)
            fa2[mt11] = wg_ld128(wf2, (uint)(((mt11 * FG_C1T + cb2) * 32 + (int)lane) * 16));
        [unroll] for (int mt12 = 0; mt12 < FG_C2T; ++mt12)
            [unroll] for (int nt12 = 0; nt12 < FG_NT; ++nt12) fg_chunk_b(fa2[mt12], eb[nt12], acc2[mt12][nt12], h2[mt12][nt12]);
    }
#endif
    const int C2 = 16 * FG_C2T;
    [unroll] for (int mt13 = 0; mt13 < FG_C2T; ++mt13)
        [unroll] for (int nt13 = 0; nt13 < FG_NT; ++nt13) {
            const uint4 r = fg_acc_out(h2[mt13][nt13], act2);
            if (p0 + 16 * nt13 + n < H * W) wg_st128(py, (uint)(pc[nt13] * C2 + 16 * mt13 + 8 * hh) * 2u, r);
        }
}
