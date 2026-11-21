// SPDX-License-Identifier: MIT
// Frame generation (issue #96): net2's 3 x 3 convolution on gfx12's matrix instructions (tier B of the fast path; the
// exact kernels fg_net2_conv.hlsl / fg_net2_up2.hlsl stay the validation path). Launch interface of fg_net2_conv (channel
// stride and offset of the input, channel range of the output, the post-ops) plus two fusions, with the arithmetic of
// net1's tier B (fg_mconv.hlsl): one v_wmma_f32_16x16x16_f16 per spec chunk through the driver channel (wg_wmma_f16),
// the float16 accumulator from the bias as C, the whole fragment rounded to float16 after every chunk (wg_to_f16_ags =
// v_cvt_f16_f32, nearest even). Chunk order of net2 (FG-PART3 3.1): taps (ky, kx) in row-major order, inside a tap
// ascending blocks of 16 input channels; channels at or past cin read 0 (L1's 10 channels as 16, L14's 18 as 16 + 16
// with zero weights). Not bit-exact to the CPU reference: the matrix unit's chunk sum is not the exact sum. Everything
// else is exact, the bits of fg_net2_conv / fg_net2_up2 computed 8 values at a time (fg_net2_fast.hlsli): leaky ReLU
// (leaky = 1), clamp to [-6, 6] for weight outputs o >= clamp_from, residual res 0 none, 1 h(v + 0), 2 fma(res_mul,
// skip, v), pool = 1: the 2 x 2 average h(h(h(h(a00 + a01) + a10) + a11) * 0.25).
//
// FG_UP = 1 (L12, L13, L24, L25): the conv grid H x W is the 2x upsample of the stored input (H / 2 x W / 2): the
// workgroup stages its conv-grid tile plus a one-pixel border in LDS, each value the exact half-pixel bilinear sample
// of fg_net2_up2 (top = fma(wx1, P01, h(wx0 P00)), bottom likewise, out = fma(wy1, bottom, h(wy0 top))), 0 outside
// the grid; the taps read the tile. heads = 1 (L13, L25): the three heads in one launch, head j = workgroup group mg
// (the weight tensor's one tile of 8 outputs for every head): input channels in_co + j cin, kept outputs 0-3 / 4 / 5-7
// (so clamp_from = 4 gives "none / clamp / clamp"), output y / y1 / y2 with out_cs = the kept count when out_cs = 0,
// else one shared output (out_cs, out_co + first kept output); residual skip / skip1 / skip2 (channel stride = kept
// count, multiplier res_mul for head 0 and 1 for heads 1 and 2).
//
// Work split: workgroup = 16 x FG_TH conv pixels (tile tx, ty; FG_TH = 2 rows per wave) and FG_MT output tiles of 16
// weight outputs (group mg); a wave owns two rows of 16 pixels (two B fragments per chunk), so a pooled 2 x 2 cell sits
// in one wave (lanes n, n ^ 1). A fragments: wf [cout / 16][9 taps][FG_NB blocks][32 lanes][8] float16
// (clean/fg/runner/fg_net2_mma.py swizzle); b: the bias padded to whole tiles. Driver rules (clean/dx12/README.md):
// every load of a tap before its first matrix call, branchless selects in the loads, the tap loop bound through an
// opaque zero (no forced unrolling), store pointers through wg_uniform(); the only branch before the matrix calls (a
// wave whose rows are all past the grid) is on a wave-uniform value, after the LDS barrier. Built only when --only
// names the kernels.
//
// @wg-kernel fg_net2_mconv_m1b1 co=fg_net2_mma.co entry=fg_net2_mconv block=256 wave=32 args=p,p,p,p,p,p,p,p,p,u,u,u,u,u,u,u,u,u,u,u,u,u,u,u,u,u,u,u defines=FG_MT=1,FG_NB=1
// @wg-kernel fg_net2_mconv_m1b2 co=fg_net2_mma.co entry=fg_net2_mconv block=256 wave=32 args=p,p,p,p,p,p,p,p,p,u,u,u,u,u,u,u,u,u,u,u,u,u,u,u,u,u,u,u defines=FG_MT=1,FG_NB=2
// @wg-kernel fg_net2_mconv_m2b1 co=fg_net2_mma.co entry=fg_net2_mconv block=256 wave=32 args=p,p,p,p,p,p,p,p,p,u,u,u,u,u,u,u,u,u,u,u,u,u,u,u,u,u,u,u defines=FG_MT=2,FG_NB=1
// @wg-kernel fg_net2_mconv_m2b2 co=fg_net2_mma.co entry=fg_net2_mconv block=256 wave=32 args=p,p,p,p,p,p,p,p,p,u,u,u,u,u,u,u,u,u,u,u,u,u,u,u,u,u,u,u defines=FG_MT=2,FG_NB=2
// @wg-kernel fg_net2_mconv_m2b4 co=fg_net2_mma.co entry=fg_net2_mconv block=256 wave=32 args=p,p,p,p,p,p,p,p,p,u,u,u,u,u,u,u,u,u,u,u,u,u,u,u,u,u,u,u defines=FG_MT=2,FG_NB=4
// @wg-kernel fg_net2_mconv_m1b4s co=fg_net2_mma.co entry=fg_net2_mconv block=128 wave=32 args=p,p,p,p,p,p,p,p,p,u,u,u,u,u,u,u,u,u,u,u,u,u,u,u,u,u,u,u defines=FG_MT=1,FG_NB=4
// @wg-kernel fg_net2_mconv_m1b2s co=fg_net2_mma.co entry=fg_net2_mconv block=128 wave=32 args=p,p,p,p,p,p,p,p,p,u,u,u,u,u,u,u,u,u,u,u,u,u,u,u,u,u,u,u defines=FG_MT=1,FG_NB=2
// @wg-kernel fg_net2_mconv_m1b1u co=fg_net2_mma.co entry=fg_net2_mconv block=256 wave=32 args=p,p,p,p,p,p,p,p,p,u,u,u,u,u,u,u,u,u,u,u,u,u,u,u,u,u,u,u defines=FG_MT=1,FG_NB=1,FG_UP=1
// @wg-kernel fg_net2_mconv_m1b2u co=fg_net2_mma.co entry=fg_net2_mconv block=256 wave=32 args=p,p,p,p,p,p,p,p,p,u,u,u,u,u,u,u,u,u,u,u,u,u,u,u,u,u,u,u defines=FG_MT=1,FG_NB=2,FG_UP=1
// @wg-kernel fg_net2_mconv_m2b4u co=fg_net2_mma.co entry=fg_net2_mconv block=128 wave=32 args=p,p,p,p,p,p,p,p,p,u,u,u,u,u,u,u,u,u,u,u,u,u,u,u,u,u,u,u defines=FG_MT=2,FG_NB=4,FG_UP=1
// @wg-kernel fg_net2_mconv_m3b2u co=fg_net2_mma.co entry=fg_net2_mconv block=256 wave=32 args=p,p,p,p,p,p,p,p,p,u,u,u,u,u,u,u,u,u,u,u,u,u,u,u,u,u,u,u defines=FG_MT=3,FG_NB=2,FG_UP=1
// @wg-kernel fg_net2_mconv_m3b4u co=fg_net2_mma.co entry=fg_net2_mconv block=128 wave=32 args=p,p,p,p,p,p,p,p,p,u,u,u,u,u,u,u,u,u,u,u,u,u,u,u,u,u,u,u defines=FG_MT=3,FG_NB=4,FG_UP=1
#include "wg_platform.hlsli"
#include "fg_net2.hlsli"

#ifndef FG_MT
#define FG_MT 1
#endif
#ifndef FG_NB
#define FG_NB 1
#endif
#ifndef FG_UP
#define FG_UP 0
#endif
#define FG_TH (2 * (WG_BLOCK / 32))          // conv rows per workgroup

WG_ARGS_BEGIN
    uint x, wf, b, y, skip, y1, y2, skip1, skip2;    // pointers (arena byte offsets)
    int in_cs, in_co, cin, H, W, co_lo, co_n, out_cs, out_co, pool, leaky, clamp_from, res;
    uint res_mul;
    int skip_cs, skip_co, tiles_x, tiles_y, heads;
WG_ARGS_END
#include "fg_net2_fast.hlsli"

#if FG_UP
#define FG_TG (FG_NB * 2)                    // 8-channel groups per staged pixel
groupshared uint4 fg_tile[(FG_TH + 2) * 18 * FG_TG];
#endif

// 8 packed float16 values -> the float32 accumulator fragment (exact)
v8f fg_widen(uint4 hp)
{
    v8f r;
    [unroll] for (uint v = 0; v < 8; ++v) r.v[v] = wg_h2f(fg_half(hp, v));
    return r;
}
// One chunk (tier B, as fg_mconv.hlsl): the matrix instruction, then the fragment rounded to float16 by the driver
// channel's conversion; hp keeps the packed float16 accumulator.
void fg_chunk_b(uint4 av, uint4 bv, inout v8f c, inout uint4 hp)
{
    const v8f d = wg_wmma_f16(av, bv, c);
    hp = wg_to_f16_ags(d);
    c = fg_widen(hp);
}
// the mask of 8 input channels c0 .. c0 + 7 (dword d: channels c0 + 2d, c0 + 2d + 1): channels >= cin read 0
uint4 fg_cmask(int c0)
{
    uint m[4];
    [unroll] for (int d = 0; d < 4; ++d) {
        const int c = c0 + 2 * d;
        m[d] = select(c + 1 < cin, 0xffffffffu, select(c < cin, 0x0000ffffu, 0u));
    }
    return uint4(m[0], m[1], m[2], m[3]);
}

// the launch's output parameters for this workgroup (heads: per head)
struct FgOut { int lo; int n; int cs; int co; uint yp; uint sp; int scs; uint rmul; };

// the exact post-ops of fg_net2_conv on the lane's 8 weight outputs o0 .. o0 + 7 at conv pixel (cy, cx): leaky,
// clamp from clamp_from, residual (res 1: h(v + 0); res 2: fma(rmul, skip, v), the skip channels of the kept outputs)
uint4 fg_post8(FgOut q, uint4 v, int o0, int cy, int cx, bool live)
{
    if (leaky != 0) v = fg_leaky8(v);
    uint r[8];
    [unroll] for (int i = 0; i < 8; ++i) {
        const uint hv = fg_half(v, (uint)i);
        r[i] = select(o0 + i >= clamp_from, fg2_clamp6(hv), hv);
        r[i] = select(res == 1, fg2_plus0(r[i]), r[i]);
    }
    v = fg_pack8(r);
    if (res == 2) {
        const int e0 = (cy * W + cx) * q.scs + skip_co - q.lo;
        uint4 sk;
        if (live && o0 >= q.lo && o0 + 8 <= q.lo + q.n && ((e0 + o0) & 1) == 0) {
            sk = wg_ld128(q.sp, (uint)(e0 + o0) * 2u);          // 8 contiguous skip channels
        } else {
            uint s8[8];
            [unroll] for (int j = 0; j < 8; ++j) {
                const bool kept = live && o0 + j >= q.lo && o0 + j < q.lo + q.n;
                s8[j] = wg_ld16(q.sp, kept ? (uint)(e0 + o0 + j) * 2u : 0u);
            }
            sk = fg_pack8(s8);
        }
        v = fg_fma8(wg_h2f(q.rmul), sk, v);
    }
    return v;
}

// stores the lane's 8 weight outputs o0 .. o0 + 7 of output pixel pix (those inside [lo, lo + n))
void fg_store8(FgOut q, int pix, int o0, uint4 p)
{
    const bool wide = (q.lo & 7) == 0 && (q.n & 7) == 0 && (q.cs & 7) == 0 && (q.co & 7) == 0;
    const int e0 = pix * q.cs + q.co - q.lo;
    if (wide) {
        if (o0 >= q.lo && o0 + 8 <= q.lo + q.n) wg_st128(q.yp, (uint)(e0 + o0) * 2u, p);
    } else {
        [unroll] for (int v = 0; v < 8; ++v)
            if (o0 + v >= q.lo && o0 + v < q.lo + q.n) wg_st16(q.yp, (uint)(e0 + o0 + v) * 2u, fg_half(p, (uint)v));
    }
}

#if FG_UP
// stages the 2x upsample (fg_up8) of conv-grid rows y0 - 1 .. y0 + FG_TH and columns x0 - 1 .. x0 + 16 (zero outside the grid)
void fg_stage(int x0, int y0, int ico)
{
    const int hi = H / 2, wi = W / 2;
    const int items = (FG_TH + 2) * 18 * FG_TG;
    const int rounds = (items + WG_BLOCK - 1) / WG_BLOCK + (int)fg_opq0();
    // uniform control flow around the driver-channel conversions: every lane computes a clamped position, the select
    // zeroes positions outside the grid and the store skips items past the tile
    for (int k = 0; k < rounds; ++k) {
        const int it = k * WG_BLOCK + (int)wg_lane_id();
        const int ic = min(it, items - 1);
        const int g = ic % FG_TG, pp = ic / FG_TG;
        const int cy = y0 - 1 + pp / 18, cx = x0 - 1 + pp % 18;
        const bool inside = cy >= 0 && cy < H && cx >= 0 && cx < W;
        const Fg2Tap tX = fg2_up_tap(clamp(cx, 0, W - 1), wi), tY = fg2_up_tap(clamp(cy, 0, H - 1), hi);
        const uint c0 = (uint)(ico + 8 * g);
        const uint4 q00 = wg_ld128(x, ((uint)((tY.i0 * wi + tX.i0) * in_cs) + c0) * 2u);
        const uint4 q01 = wg_ld128(x, ((uint)((tY.i0 * wi + tX.i1) * in_cs) + c0) * 2u);
        const uint4 q10 = wg_ld128(x, ((uint)((tY.i1 * wi + tX.i0) * in_cs) + c0) * 2u);
        const uint4 q11 = wg_ld128(x, ((uint)((tY.i1 * wi + tX.i1) * in_cs) + c0) * 2u);
        const uint4 v = fg_up8(tX, tY, q00, q01, q10, q11) & fg_cmask(8 * g);
        const uint z = select(inside, 0xffffffffu, 0u);
        if (it < items) fg_tile[it] = uint4(v.x & z, v.y & z, v.z & z, v.w & z);
    }
    GroupMemoryBarrierWithGroupSync();
}
#endif

WG_KERNEL(fg_net2_mconv)
{
    WG_SETUP();
    const uint lane = wg_lane_id() & 31u;
    const int blk = (int)wg_block_id();
    const int tx = blk % tiles_x, rest = blk / tiles_x;
    const int ty = rest % tiles_y, mg = rest / tiles_y;
    const int n = (int)(lane & 15u), hh = (int)(lane >> 4);
    const int px = tx * 16 + n;
    const int wrow = 2 * (int)(wg_lane_id() >> 5);        // the wave's first row in the tile
    const int py0 = wg_uniform(ty * FG_TH + wrow);

    // per-head launch parameters (uniform)
    FgOut q;
    const int wt = heads != 0 ? 0 : mg;                     // weight tile group
    const int hj = heads != 0 ? mg : 0;
    q.lo = heads != 0 ? (hj == 0 ? 0 : (hj == 1 ? 4 : 5)) : co_lo;
    q.n = heads != 0 ? (hj == 0 ? 4 : (hj == 1 ? 1 : 3)) : co_n;
    q.cs = heads != 0 && out_cs == 0 ? q.n : out_cs;
    q.co = heads != 0 ? (out_cs == 0 ? 0 : out_co + q.lo) : out_co;
    q.yp = (uint)wg_uniform((int)(hj == 0 || out_cs != 0 ? y : (hj == 1 ? y1 : y2)));
    q.sp = (uint)wg_uniform((int)(hj == 0 ? skip : (hj == 1 ? skip1 : skip2)));
    q.scs = heads != 0 ? q.n : skip_cs;
    q.rmul = hj == 0 ? res_mul : FG2_ONE16;
    const int ico = in_co + hj * cin;

#if FG_UP
    fg_stage(tx * 16, ty * FG_TH, ico);
#endif
    if (py0 >= H) return;                            // wave-uniform: both rows of this wave are past the grid

    v8f a32[FG_MT][2];
    uint4 h16[FG_MT][2];                             // the same accumulator as packed float16
    [unroll] for (int mt = 0; mt < FG_MT; ++mt) {
        const int o0 = (wt * FG_MT + mt) * 16 + 8 * hh;
        uint bb[8];
        [unroll] for (uint v = 0; v < 8; ++v) bb[v] = wg_ld16(b, (uint)(o0 + (int)v) * 2u);
        h16[mt][0] = fg_pack8(bb);
        h16[mt][1] = h16[mt][0];
        a32[mt][0] = fg_widen(h16[mt][0]);
        a32[mt][1] = a32[mt][0];
    }
#if !FG_UP
    uint4 cm[FG_NB];
    [unroll] for (int cb0 = 0; cb0 < FG_NB; ++cb0) cm[cb0] = fg_cmask(cb0 * 16 + 8 * hh);
#endif

    const int taps = 9 + (int)fg_opq0();
    for (int t = 0; t < taps; ++t) {
        const int ky = t / 3, kx = t - 3 * (t / 3);
        uint4 fb[2][FG_NB], fa[FG_MT][FG_NB];
        [unroll] for (int nt = 0; nt < 2; ++nt) {
#if FG_UP
            const int pp = (wrow + nt + ky) * 18 + n + kx;
            [unroll] for (int cb = 0; cb < FG_NB; ++cb) fb[nt][cb] = fg_tile[pp * FG_TG + 2 * cb + hh];
#else
            const int sx = px + kx - 1, sy = py0 + nt + ky - 1;
            const bool inb = and(and(sx >= 0, sx < W), and(sy >= 0, sy < H));
            const uint e = select(inb, (uint)((sy * W + sx) * in_cs + ico + 8 * hh), 0u);
            const uint z = select(inb, 0xffffffffu, 0u);
            [unroll] for (int cb = 0; cb < FG_NB; ++cb) {
                const uint4 u = wg_ld128(x, (e + (uint)(cb * 16)) * 2u);
                fb[nt][cb] = uint4(u.x & cm[cb].x & z, u.y & cm[cb].y & z, u.z & cm[cb].z & z, u.w & cm[cb].w & z);
            }
#endif
        }
        [unroll] for (int mt2 = 0; mt2 < FG_MT; ++mt2)
            [unroll] for (int cb2 = 0; cb2 < FG_NB; ++cb2)
                fa[mt2][cb2] = wg_ld128(wf, (uint)(((((wt * FG_MT + mt2) * 9 + t) * FG_NB + cb2) * 32 + (int)lane) * 16));
        [unroll] for (int cb3 = 0; cb3 < FG_NB; ++cb3)
            [unroll] for (int mt3 = 0; mt3 < FG_MT; ++mt3)
                [unroll] for (int nt3 = 0; nt3 < 2; ++nt3) fg_chunk_b(fa[mt3][cb3], fb[nt3][cb3], a32[mt3][nt3], h16[mt3][nt3]);
    }

    const bool live0 = px < W && py0 < H, live1 = px < W && py0 + 1 < H;
    [unroll] for (int mt4 = 0; mt4 < FG_MT; ++mt4) {
        const int o0 = (wt * FG_MT + mt4) * 16 + 8 * hh;
        const uint4 r0 = fg_post8(q, h16[mt4][0], o0, py0, px, live0);
        const uint4 r1 = fg_post8(q, h16[mt4][1], o0, py0 + 1, px, live1);
        if (pool == 0) {
            if (live0) fg_store8(q, py0 * W + px, o0, r0);
            if (live1) fg_store8(q, (py0 + 1) * W + px, o0, r1);
        } else {
            // h(h(h(h(a00 + a01) + a10) + a11) * 0.25), a01 / a11 from the neighbouring lane
            const uint4 x01 = uint4(wg_xor_u(r0.x, 1u), wg_xor_u(r0.y, 1u), wg_xor_u(r0.z, 1u), wg_xor_u(r0.w, 1u));
            const uint4 x11 = uint4(wg_xor_u(r1.x, 1u), wg_xor_u(r1.y, 1u), wg_xor_u(r1.z, 1u), wg_xor_u(r1.w, 1u));
            const uint4 m = fg_mul8(0.25f, fg_fma8(1.0f, x11, fg_fma8(1.0f, r1, fg_fma8(1.0f, x01, r0))));
            const int Ho = H / 2, Wo = W / 2, ox = px / 2, oy = py0 / 2;
            if ((n & 1) == 0 && ox < Wo && oy < Ho) fg_store8(q, oy * Wo + ox, o0, m);
        }
    }
}
