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
// workgroup builds its conv-grid tile and one-pixel border directly from the half-resolution input, in one LDS
// buffer. Adjacent rows share the exact horizontal samples of fg_net2_up2. Values outside the grid are zero.
// Selected plain variants stage the same halo directly from the input.
// heads = 1 (L13, L25): the three heads in one launch, head j = workgroup group mg
// (the weight tensor's one tile of 8 outputs for every head): input channels in_co + j cin, kept outputs 0-3 / 4 / 5-7
// (so clamp_from = 4 gives "none / clamp / clamp"), output y / y1 / y2 with out_cs = the kept count when out_cs = 0,
// else one shared output (out_cs, out_co + first kept output); residual skip / skip1 / skip2 (channel stride = kept
// count, multiplier res_mul for head 0 and 1 for heads 1 and 2). The h3 variants take the same arguments and use
// one workgroup per spatial tile. They reuse the LDS tile across heads and join a shared eight-channel output
// before the residual and store.
//
// FG_STAGE_ALL = 1 (fused heads only, fg_net2_mconv_h3b1u_s3): the three heads' tiles are staged in one pass (one loop over
// the three heads' items, one LDS barrier, a tile of three head blocks) instead of three passes with a barrier each, so
// the staging rounds are full (972 items over 256 lanes: 3.8 rounds instead of 3 x 1.27) and no barrier separates the heads.
// Every staged value and every convolution is the same computation on the same data as the three-pass kernel.
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
// @wg-kernel fg_net2_mconv_h3b1u co=fg_net2_mma.co entry=fg_net2_mconv block=256 wave=32 args=p,p,p,p,p,p,p,p,p,u,u,u,u,u,u,u,u,u,u,u,u,u,u,u,u,u,u,u defines=FG_MT=1,FG_NB=1,FG_UP=1,FG_FUSED_HEADS=1
// @wg-kernel fg_net2_mconv_h3b1u_s3 co=fg_net2_mma.co entry=fg_net2_mconv block=256 wave=32 args=p,p,p,p,p,p,p,p,p,u,u,u,u,u,u,u,u,u,u,u,u,u,u,u,u,u,u,u defines=FG_MT=1,FG_NB=1,FG_UP=1,FG_FUSED_HEADS=1,FG_STAGE_ALL=1
// @wg-kernel fg_net2_mconv_h3b2u co=fg_net2_mma.co entry=fg_net2_mconv block=256 wave=32 args=p,p,p,p,p,p,p,p,p,u,u,u,u,u,u,u,u,u,u,u,u,u,u,u,u,u,u,u defines=FG_MT=1,FG_NB=2,FG_UP=1,FG_FUSED_HEADS=1
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
#ifndef FG_FUSED_HEADS
#define FG_FUSED_HEADS 0
#endif
#ifndef FG_STAGE_ALL
#define FG_STAGE_ALL 0
#endif
#define FG_NATIVE_UP FG_UP
#define FG_NATIVE_POST ((!FG_UP && (FG_NB == 2 || FG_MT == 2 || FG_NB == 4)) || (FG_UP && FG_NB == 1))
#define FG_NATIVE_LEAKY 1
#define FG_JOINED_POST FG_FUSED_HEADS
#include "fg_net2_fast.hlsli"

// Preserve the unaligned path for packed 10- and 18-channel input and for channel slices.
uint4 fg_load_aligned8(uint base, uint off)
{
    return wg_mem.Load4((base + off) & ~15u);
}

uint4 fg_load_input8(uint off, int ico)
{
    if (((in_cs | ico) & 7) == 0) return fg_load_aligned8(x, off);
    return wg_ld128(x, off);
}

#define FG_STAGE (FG_UP || FG_NB == 2 || (FG_MT == 2 && FG_NB == 1) || (FG_MT == 1 && FG_NB == 4))
#if FG_STAGE
#define FG_TG (FG_NB * 2)
#define FG_TILE_PIXELS ((FG_TH + 2) * 18)
#if FG_STAGE_ALL
groupshared uint4 fg_tile[FG_TILE_PIXELS * FG_TG * 3];
static uint fg_tile_head;   // the head whose block of the tile the convolution reads (wave-uniform)

uint fg_tile_index(uint p, uint g) { return (fg_tile_head * FG_TG + g) * FG_TILE_PIXELS + p; }
#else
groupshared uint4 fg_tile[FG_TILE_PIXELS * FG_TG];

uint fg_tile_index(uint p, uint g) { return g * FG_TILE_PIXELS + p; }
#endif
#endif

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
// One chunk (tier B, as fg_mconv.hlsl): the matrix instruction, then the fragment rounded to float16 by the driver
// channel's conversion; hp keeps the packed float16 accumulator.
void fg_chunk_b(uint4 av, uint4 bv, inout uint4 hp)
{
    const v8f d = wg_wmma_f16(av, bv, fg_widen(hp));
    hp = wg_to_f16_ags(d);
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

// Load the packed 4-, 1-, or 3-channel residual for a head. The 3-channel stride alternates alignment.
uint4 fg_head_skip(FgOut q, int e, bool live)
{
    const uint at = live ? (uint)e * 2u : 0u;
    if (q.n == 4 && (e & 1) == 0) {
        const uint2 v = wg_ld64(q.sp, at);
        return uint4(v, 0u, 0u);
    }
    if (q.n == 1) return uint4(0u, 0u, wg_ld16(q.sp, at), 0u);
    if (q.n == 3) {
        uint a, bc;
        if ((e & 1) != 0) {
            a = wg_ld16(q.sp, at);
            bc = wg_ld32(q.sp, at + 2u * (uint)live);
        } else {
            const uint ab = wg_ld32(q.sp, at);
            a = ab & 0xffffu;
            bc = (ab >> 16) | (wg_ld16(q.sp, at + 4u * (uint)live) << 16);
        }
        return uint4(0u, 0u, a << 16, bc);
    }
    return uint4(wg_ld16(q.sp, at) | (wg_ld16(q.sp, at + 2u * (uint)live) << 16),
                 wg_ld16(q.sp, at + 4u * (uint)live) | (wg_ld16(q.sp, at + 6u * (uint)live) << 16), 0u, 0u);
}

// the exact post-ops of fg_net2_conv on the lane's 8 weight outputs o0 .. o0 + 7 at conv pixel (cy, cx): leaky,
// clamp from clamp_from, residual (res 1: h(v + 0); res 2: fma(rmul, skip, v), the skip channels of the kept outputs)
uint4 fg_adjust8(uint4 v, int o0)
{
    if (leaky != 0) v = fg_leaky8(v);
    uint r[8];
    [unroll] for (int i = 0; i < 8; ++i) {
        const uint hv = fg_half(v, (uint)i);
        const uint mag = hv & 0x7fffu;
        const bool clamp_value = o0 + i >= clamp_from && mag > FG2_SIX16 && mag <= 0x7c00u;
        r[i] = select(clamp_value, (hv & 0x8000u) | FG2_SIX16, hv);
        r[i] = select(res == 1, fg2_plus0(r[i]), r[i]);
    }
    return fg_pack8(r);
}

uint4 fg_post8(FgOut q, uint4 v, int o0, int cy, int cx, bool live)
{
    v = fg_adjust8(v, o0);
    if (res == 2) {
        const int e0 = (cy * W + cx) * q.scs + skip_co - q.lo;
        uint4 sk;
        if (heads != 0) {
            sk = fg_head_skip(q, (cy * W + cx) * q.scs + skip_co, live && o0 == 0);
        } else if (live && o0 >= q.lo && o0 + 8 <= q.lo + q.n && ((e0 + o0) & 1) == 0) {
            const uint off = (uint)(e0 + o0) * 2u;
            sk = ((q.scs | (skip_co - q.lo)) & 7) == 0 ? fg_load_aligned8(q.sp, off) : wg_ld128(q.sp, off);
        } else {
            uint s8[8];
            [unroll] for (int j = 0; j < 8; ++j) {
                const bool kept = live && o0 + j >= q.lo && o0 + j < q.lo + q.n;
                s8[j] = wg_ld16(q.sp, kept ? (uint)(e0 + o0 + j) * 2u : 0u);
            }
            sk = fg_pack8(s8);
        }
#if FG_NATIVE_POST
        v = fg_fma8_post(q.rmul, sk, v);
#else
        v = fg_fma8(wg_h2f(q.rmul), sk, v);
#endif
    }
    return v;
}

// stores the lane's 8 weight outputs o0 .. o0 + 7 of output pixel pix (those inside [lo, lo + n))
void fg_store8(FgOut q, int pix, int o0, uint4 p)
{
    const bool wide = (q.lo & 7) == 0 && (q.n & 7) == 0 && (q.cs & 7) == 0 && (q.co & 7) == 0;
    const int e0 = pix * q.cs + q.co - q.lo;
    if (heads != 0) {
        if (o0 != 0) return;
        const int e = e0 + q.lo;
        const uint at = (uint)e * 2u;
        if (q.n == 4 && (e & 1) == 0) {
            wg_st64(q.yp, at, p.xy);
        } else if (q.n == 1) {
            wg_st16(q.yp, at, p.z);
        } else if (q.n == 3) {
            if ((e & 1) == 0) {
                wg_st32(q.yp, at, (p.z >> 16) | (p.w << 16));
                wg_st16(q.yp, at + 4u, p.w >> 16);
            } else {
                wg_st16(q.yp, at, p.z >> 16);
                wg_st32(q.yp, at + 2u, p.w);
            }
        } else {
            [unroll] for (int i = 0; i < 4; ++i) wg_st16(q.yp, at + 2u * (uint)i, fg_half(p, (uint)i));
        }
    } else if (wide) {
        if (o0 >= q.lo && o0 + 8 <= q.lo + q.n) wg_st128(q.yp, (uint)(e0 + o0) * 2u, p);
    } else {
        [unroll] for (int v = 0; v < 8; ++v)
            if (o0 + v >= q.lo && o0 + v < q.lo + q.n) wg_st16(q.yp, (uint)(e0 + o0 + v) * 2u, fg_half(p, (uint)v));
    }
}

#if FG_UP
// Specialization of fg2_up_tap for integer output positions on a 2x grid.
Fg2Tap fg_stage_tap(int o, int n)
{
    const int i = (o - 1) >> 1;
    Fg2Tap t;
    t.i0 = max(i, 0);
    t.i1 = clamp(i + 1, 0, n - 1);
    t.w0 = o == 0 ? 0x3d00u : ((o & 1) ? 0x3a00u : 0x3400u);
    t.w1 = o == 0 ? 0xb400u : ((o & 1) ? 0x3400u : 0x3a00u);
    return t;
}

// Rows 2j-1 and 2j use the same source row pair. Reuse their rounded horizontal samples.
void fg_stage(int x0, int y0, int ico)
{
    const int hi = H / 2, wi = W / 2;
    const int items = (FG_TH / 2 + 1) * 18 * FG_TG;
    const int rounds = (items + WG_BLOCK - 1) / WG_BLOCK + (int)fg_opq0();
    for (int round = 0; round < rounds; ++round) {
        // Lanes past the end still participate in the wave's conversion.
        const int it = round * WG_BLOCK + (int)wg_lane_id(), elem = min(it, items - 1);
        const int g = elem % FG_TG, p = elem / FG_TG;
        const int cy = y0 - 1 + 2 * (p / 18), cx = x0 - 1 + p % 18;
        const Fg2Tap tx = fg_stage_tap(clamp(cx, 0, W - 1), wi);
        const Fg2Tap ty0 = fg_stage_tap(clamp(cy, 0, H - 1), hi);
        const Fg2Tap ty1 = fg_stage_tap(clamp(cy + 1, 0, H - 1), hi);
        const int row0 = ty0.i0 * wi, row1 = ty0.i1 * wi, c0 = ico + 8 * g;
        const uint4 q00 = fg_load_input8((uint)(((row0 + tx.i0) * in_cs + c0) * 2), ico);
        const uint4 q01 = fg_load_input8((uint)(((row0 + tx.i1) * in_cs + c0) * 2), ico);
        const uint4 q10 = fg_load_input8((uint)(((row1 + tx.i0) * in_cs + c0) * 2), ico);
        const uint4 q11 = fg_load_input8((uint)(((row1 + tx.i1) * in_cs + c0) * 2), ico);
        const uint4 top = fg_up_axis8(tx, q00, q01), bot = fg_up_axis8(tx, q10, q11);
        const uint4 mask = fg_cmask(8 * g);
        const bool inside_x = cx >= 0 && cx < W;
        const uint4 v0 = fg_up_axis8(ty0, top, bot) & mask & select(inside_x && cy >= 0 && cy < H, ~0u, 0u);
        const uint4 v1 = fg_up_axis8(ty1, top, bot) & mask & select(inside_x && cy + 1 >= 0 && cy + 1 < H, ~0u, 0u);
        if (it < items) {
            const int dst = p + p / 18 * 18;
            fg_tile[fg_tile_index(dst, g)] = v0;
            fg_tile[fg_tile_index(dst + 18, g)] = v1;
        }
    }
    wg_barrier_lds();
}
#if FG_STAGE_ALL
// fg_stage for the three heads (head hj reads the channels in_co + hj cin ..), one pass and one barrier
void fg_stage_heads(int x0, int y0)
{
    const int hi = H / 2, wi = W / 2;
    const int items1 = (FG_TH / 2 + 1) * 18 * FG_TG;
    const int items = 3 * items1;
    const int rounds = (items + WG_BLOCK - 1) / WG_BLOCK + (int)fg_opq0();
    for (int round = 0; round < rounds; ++round) {
        // Lanes past the end still participate in the wave's conversion.
        const int it = round * WG_BLOCK + (int)wg_lane_id(), elem = min(it, items - 1);
        const int hj = elem / items1, e1 = elem - hj * items1;
        const int ico = in_co + hj * cin;
        const int g = e1 % FG_TG, p = e1 / FG_TG;
        const int cy = y0 - 1 + 2 * (p / 18), cx = x0 - 1 + p % 18;
        const Fg2Tap tx = fg_stage_tap(clamp(cx, 0, W - 1), wi);
        const Fg2Tap ty0 = fg_stage_tap(clamp(cy, 0, H - 1), hi);
        const Fg2Tap ty1 = fg_stage_tap(clamp(cy + 1, 0, H - 1), hi);
        const int row0 = ty0.i0 * wi, row1 = ty0.i1 * wi, c0 = ico + 8 * g;
        const uint4 q00 = fg_load_input8((uint)(((row0 + tx.i0) * in_cs + c0) * 2), ico);
        const uint4 q01 = fg_load_input8((uint)(((row0 + tx.i1) * in_cs + c0) * 2), ico);
        const uint4 q10 = fg_load_input8((uint)(((row1 + tx.i0) * in_cs + c0) * 2), ico);
        const uint4 q11 = fg_load_input8((uint)(((row1 + tx.i1) * in_cs + c0) * 2), ico);
        const uint4 top = fg_up_axis8(tx, q00, q01), bot = fg_up_axis8(tx, q10, q11);
        const uint4 mask = fg_cmask(8 * g);
        const bool inside_x = cx >= 0 && cx < W;
        const uint4 v0 = fg_up_axis8(ty0, top, bot) & mask & select(inside_x && cy >= 0 && cy < H, ~0u, 0u);
        const uint4 v1 = fg_up_axis8(ty1, top, bot) & mask & select(inside_x && cy + 1 >= 0 && cy + 1 < H, ~0u, 0u);
        if (it < items) {
            const int dst = p + p / 18 * 18;
            fg_tile[(hj * FG_TG + g) * FG_TILE_PIXELS + dst] = v0;
            fg_tile[(hj * FG_TG + g) * FG_TILE_PIXELS + dst + 18] = v1;
        }
    }
    wg_barrier_lds();
}
#endif
#elif FG_STAGE
void fg_stage(int x0, int y0, int ico)
{
    const uint items = FG_TILE_PIXELS * FG_TG;
    for (uint i = wg_lane_id(); i < items; i += WG_BLOCK) {
        const int g = (int)(i % FG_TG), p = (int)(i / FG_TG);
        const int sx = x0 + p % 18 - 1, sy = y0 + p / 18 - 1;
        const bool inside = and(and(sx >= 0, sx < W), and(sy >= 0, sy < H));
        const uint e = select(inside, (uint)((sy * W + sx) * in_cs + ico), 0u);
        const uint mask = select(inside, 0xffffffffu, 0u);
        fg_tile[fg_tile_index(p, g)] = fg_load_input8((e + (uint)(8 * g)) * 2u, ico) & fg_cmask(8 * g) & mask;
    }
    wg_barrier_lds();
}
#endif

FgOut fg_output(int hj)
{
    FgOut q;
    q.lo = heads != 0 ? (hj == 0 ? 0 : (hj == 1 ? 4 : 5)) : co_lo;
    q.n = heads != 0 ? (hj == 0 ? 4 : (hj == 1 ? 1 : 3)) : co_n;
    q.cs = heads != 0 && out_cs == 0 ? q.n : out_cs;
    q.co = heads != 0 ? (out_cs == 0 ? 0 : out_co + q.lo) : out_co;
    q.yp = (uint)wg_uniform((int)(hj == 0 || out_cs != 0 ? y : (hj == 1 ? y1 : y2)));
    q.sp = (uint)wg_uniform((int)(hj == 0 ? skip : (hj == 1 ? skip1 : skip2)));
    q.scs = heads != 0 ? q.n : skip_cs;
    q.rmul = hj == 0 ? res_mul : FG2_ONE16;
    return q;
}

void fg_conv_tile(int px, int py0, int wrow, int n, int hh, uint lane, int wt, int ico, out uint4 h16[FG_MT][2])
{
    [unroll] for (int mt = 0; mt < FG_MT; ++mt) {
        const int o0 = (wt * FG_MT + mt) * 16 + 8 * hh;
        h16[mt][0] = fg_load_aligned8(b, (uint)o0 * 2u);
        h16[mt][1] = h16[mt][0];
    }
#if !FG_STAGE
    uint4 cm[FG_NB];
    [unroll] for (int cb = 0; cb < FG_NB; ++cb) cm[cb] = fg_cmask(cb * 16 + 8 * hh);
#endif
#if FG_STAGE && (FG_MT * FG_NB <= 2)
    // Software-pipeline tap loads with WMMAs (same win as fg_mconv.hlsl): prefetch tap t+1's weights and LDS tile
    // before tap t's WMMAs. Unrolled 9 taps eliminates dynamic tap-index bit math while keeping VGPRs low.
    uint4 fb[2][FG_NB], fa[FG_MT][FG_NB];
    {
        [unroll] for (int nt = 0; nt < 2; ++nt) {
            const int pp = (wrow + nt) * 18 + n;
            [unroll] for (int cb = 0; cb < FG_NB; ++cb) fb[nt][cb] = fg_tile[fg_tile_index(pp, 2 * cb + hh)];
        }
        [unroll] for (int mt = 0; mt < FG_MT; ++mt)
            [unroll] for (int cb = 0; cb < FG_NB; ++cb)
                fa[mt][cb] = fg_load_aligned8(wf, (uint)(((((wt * FG_MT + mt) * 9) * FG_NB + cb) * 32 + (int)lane) * 16));
    }
    [unroll] for (int t = 0; t < 9; ++t) {
        uint4 fb_n[2][FG_NB], fa_n[FG_MT][FG_NB];
        if (t + 1 < 9) {
            const int tn = t + 1;
            const int kxn = tn % 3, kyn = tn / 3;
            [unroll] for (int nt = 0; nt < 2; ++nt) {
                const int ppn = (wrow + nt + kyn) * 18 + n + kxn;
                [unroll] for (int cb = 0; cb < FG_NB; ++cb) fb_n[nt][cb] = fg_tile[fg_tile_index(ppn, 2 * cb + hh)];
            }
            [unroll] for (int mt = 0; mt < FG_MT; ++mt)
                [unroll] for (int cb = 0; cb < FG_NB; ++cb)
                    fa_n[mt][cb] = fg_load_aligned8(wf, (uint)(((((wt * FG_MT + mt) * 9 + tn) * FG_NB + cb) * 32 + (int)lane) * 16));
        }
        {
            const int N = FG_NB * FG_MT * 2;
            v8f w_next = fg_widen(h16[0][0]);
            v8f d_prev = w_next;
            [unroll] for (int s = 0; s < N; ++s) {
                const int cb = s / (FG_MT * 2);
                const int r = s - cb * (FG_MT * 2);
                const int mt = r >> 1;
                const int nt = r & 1;
                const v8f d_cur = wg_wmma_f16(fa[mt][cb], fb[nt][cb], w_next);
                if (s > 0) {
                    const int ps = s - 1;
                    const int pr = ps % (FG_MT * 2);
                    h16[pr >> 1][pr & 1] = wg_to_f16_ags(d_prev);
                }
                if (s + 1 < N) {
                    const int ns = s + 1;
                    const int nr = ns % (FG_MT * 2);
                    w_next = fg_widen(h16[nr >> 1][nr & 1]);
                }
                d_prev = d_cur;
            }
            const int lr = (N - 1) % (FG_MT * 2);
            h16[lr >> 1][lr & 1] = wg_to_f16_ags(d_prev);
        }
        if (t + 1 < 9) {
            [unroll] for (int nt2 = 0; nt2 < 2; ++nt2)
                [unroll] for (int cb2 = 0; cb2 < FG_NB; ++cb2) fb[nt2][cb2] = fb_n[nt2][cb2];
            [unroll] for (int mt2 = 0; mt2 < FG_MT; ++mt2)
                [unroll] for (int cb3 = 0; cb3 < FG_NB; ++cb3) fa[mt2][cb3] = fa_n[mt2][cb3];
        }
    }
#else
    const int taps = 9 + (int)fg_opq0();
    for (int t = 0; t < taps; ++t) {
        // Two-bit tap coordinates, low to high: x = 0,1,2,0,1,2,0,1,2 and y = 0,0,0,1,1,1,2,2,2.
        const uint shift = (uint)t * 2u;
        const int kx = (int)((0x24924u >> shift) & 3u), ky = (int)((0x2a540u >> shift) & 3u);
        uint4 fb[2][FG_NB], fa[FG_MT][FG_NB];
        [unroll] for (int nt = 0; nt < 2; ++nt) {
#if FG_STAGE
            const int pp = (wrow + nt + ky) * 18 + n + kx;
            [unroll] for (int cb = 0; cb < FG_NB; ++cb) fb[nt][cb] = fg_tile[fg_tile_index(pp, 2 * cb + hh)];
#else
            const int sx = px + kx - 1, sy = py0 + nt + ky - 1;
            const bool inb = and(and(sx >= 0, sx < W), and(sy >= 0, sy < H));
            const uint e = select(inb, (uint)((sy * W + sx) * in_cs + ico + 8 * hh), 0u);
            const uint z = select(inb, ~0u, 0u);
            [unroll] for (int cb = 0; cb < FG_NB; ++cb)
                fb[nt][cb] = fg_load_input8((e + (uint)(cb * 16)) * 2u, ico) & cm[cb] & z;
#endif
        }
        [unroll] for (int mt = 0; mt < FG_MT; ++mt)
            [unroll] for (int cb = 0; cb < FG_NB; ++cb)
                fa[mt][cb] = fg_load_aligned8(wf, (uint)(((((wt * FG_MT + mt) * 9 + t) * FG_NB + cb) * 32 + (int)lane) * 16));
        // Software-pipelined Tier B chunks across (cb, mt, nt): overlaps both fg_widen (8x v_cvt_f32_f16)
        // and wg_to_f16_ags (8x v_cvt_f16_f32) directly with independent wg_wmma_f16 ops (only 2 v8f live; bit-identical).
        {
            const int N = FG_NB * FG_MT * 2;
            v8f w_next = fg_widen(h16[0][0]);
            v8f d_prev = w_next;
            [unroll] for (int s = 0; s < N; ++s) {
                const int cb = s / (FG_MT * 2);
                const int r = s - cb * (FG_MT * 2);
                const int mt = r >> 1;
                const int nt = r & 1;
                const v8f d_cur = wg_wmma_f16(fa[mt][cb], fb[nt][cb], w_next);
                if (s > 0) {
                    const int ps = s - 1;
                    const int pr = ps % (FG_MT * 2);
                    h16[pr >> 1][pr & 1] = wg_to_f16_ags(d_prev);
                }
                if (s + 1 < N) {
                    const int ns = s + 1;
                    const int nr = ns % (FG_MT * 2);
                    w_next = fg_widen(h16[nr >> 1][nr & 1]);
                }
                d_prev = d_cur;
            }
            const int lr = (N - 1) % (FG_MT * 2);
            h16[lr >> 1][lr & 1] = wg_to_f16_ags(d_prev);
        }
    }
#endif
}

#if FG_FUSED_HEADS
void fg_fused_output(int hj, uint4 accum[2], int px, int py0, int hh, inout uint4 packed[2])
{
    const bool joined = out_cs == 8 && (out_co & 7) == 0;
    [unroll] for (int nt = 0; nt < 2; ++nt) {
        const int cy = py0 + nt, pix = cy * W + px;
        const bool live = px < W && cy < H;
        if (joined) {
            const uint4 r = accum[nt];
            if (hj == 0) packed[nt].xy = r.xy;
            if (hj == 1) packed[nt].z = r.z & 0xffffu;
            if (hj == 2) {
                packed[nt].z |= r.z & 0xffff0000u;
                packed[nt].w = r.w;
            }
        } else {
            const FgOut q = fg_output(hj);
            const uint4 r = fg_post8(q, accum[nt], 8 * hh, cy, px, live);
            if (live) fg_store8(q, pix, 8 * hh, r);
        }
    }
}

uint4 fg_joined_post(uint4 v, int px, int cy, int hh, bool live)
{
    v = fg_adjust8(v, 8 * hh);
    if (res == 2) {
        const int pix = cy * W + px;
        const bool kept = live && hh == 0;
        const uint4 sk0 = fg_head_skip(fg_output(0), pix * 4 + skip_co, kept);
        const uint4 sk1 = fg_head_skip(fg_output(1), pix + skip_co, kept);
        const uint4 sk2 = fg_head_skip(fg_output(2), pix * 3 + skip_co, kept);
        v = fg_fma8_heads(res_mul, uint4(sk0.xy, sk1.z | sk2.z, sk2.w), v);
    }
    return v;
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

#if FG_FUSED_HEADS
    uint4 packed[2] = {uint4(0u, 0u, 0u, 0u), uint4(0u, 0u, 0u, 0u)};
    // Each head consumes the same weights and bias on its own input channel slice.
#if FG_STAGE_ALL
    fg_stage_heads(tx * 16, ty * FG_TH);
#endif
    for (int hj = 0; hj < 3 + (int)fg_opq0(); ++hj) {
#if FG_STAGE_ALL
        const int ico = in_co + hj * cin;
        fg_tile_head = (uint)hj;
#else
        if (hj > 0) wg_barrier_lds();
        const int ico = in_co + hj * cin;
        fg_stage(tx * 16, ty * FG_TH, ico);
#endif
        if (py0 < H) {
            uint4 h16[1][2];
            fg_conv_tile(px, py0, wrow, n, hh, lane, 0, ico, h16);
            fg_fused_output(hj, h16[0], px, py0, hh, packed);
        }
    }
    if (out_cs == 8 && (out_co & 7) == 0) {
        const uint dst = (uint)wg_uniform((int)y);
        [unroll] for (int nt = 0; nt < 2; ++nt) {
            const int cy = py0 + nt;
            const bool live = px < W && cy < H;
            const uint4 v = fg_joined_post(packed[nt], px, cy, hh, live);
            if (hh == 0 && live) wg_st128(dst, (uint)(((cy * W + px) * 8 + out_co) * 2), v);
        }
    }
#else
    const int wt = heads != 0 ? 0 : mg;
    const int hj = heads != 0 ? mg : 0;
    const FgOut q = fg_output(hj);
    const int ico = in_co + hj * cin;

#if FG_STAGE
    fg_stage(tx * 16, ty * FG_TH, ico);
#endif
    if (py0 >= H) return;                            // wave-uniform: both rows of this wave are past the grid

    uint4 h16[FG_MT][2];
    fg_conv_tile(px, py0, wrow, n, hh, lane, wt, ico, h16);

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
#if FG_NATIVE_POST
            const uint4 s0 = fg_fma8_post(FG2_ONE16, x01, r0);
            const uint4 s1 = fg_fma8_post(FG2_ONE16, r1, s0);
            const uint4 m = fg_mul8(0.25f, fg_fma8_post(FG2_ONE16, x11, s1));
#else
            const uint4 m = fg_mul8(0.25f, fg_fma8(1.0f, x11, fg_fma8(1.0f, r1, fg_fma8(1.0f, x01, r0))));
#endif
            const int Ho = H / 2, Wo = W / 2, ox = px / 2, oy = py0 / 2;
            if ((n & 1) == 0 && ox < Wo && oy < Ho) fg_store8(q, oy * Wo + ox, o0, m);
        }
    }
#endif
}
