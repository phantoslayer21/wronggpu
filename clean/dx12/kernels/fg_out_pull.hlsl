// SPDX-License-Identifier: GPL-3.0-only
// Frame generation (issue #96): the Full path's output pull and middle pull, as clean/fg/reference/stages_full.py
// g6_output_pull and g6_output_pull_mid, bit for bit. Built only when --only names them.
//
// fg_out_pull: one workgroup per 16 x 16 frame tile (tiles_x = ceil(Wo / 16)), one lane per pixel in Morton order.
// Per frame pixel: the hole rule on Xw's 5 x 5 neighbourhood (outside the frame counts as need), net2's output N2
// (W2 x H2, 8 x float16) sampled at (x / k, y / k) with the logistic of channel 4 per tap from the table sigtab
// (65,536 float32, indexed by the float16 bits), X0 / X1 read bilinearly at the pixel plus k times net2's two vectors,
// c = sat(fma(sigma, A, (1 - sigma) B)), holes black. Writes out (4 x float16, alpha 1, row pitch out_pitch, offset
// (out_x, out_y)), the fine flag image (float16 0 / 1) and output pyramid levels 1..4 (4 x float16) through a lane tree.
// fg_out_pull_mid: one workgroup per 16 x 16 tile of level 4 (tiles_x = ceil(w4 / 16)): levels 5..8 from level 4's
// stored values (partials (r w, g w, b w, w)).
//
// fg_out_pull_pack (FGP_PACK): the same, but out is the presenter slot's packed row (fg_pack.hlsli): outp = the packed
// rows' offset, out_pitch = their pitch in bytes, out_x = the format code (1 R8G8B8A8, 2 B8G8R8A8, 3 R10G10B10A2),
// out_y = 0; each pixel is stored as the code fg_exec_pack makes of the float16 pixel fg_out_pull writes.
// FGP_X3 (fg_out_pull_x3, fg_out_pull_pack_x3): X0 / X1 are stored as 3 x float16 (6 bytes per pixel, fg_full_blend FGB_X3);
// a texel is read as the 8 bytes from its dword (the 6 bytes sit in them), the same float16 values as the 4-channel read.
//
// @wg-kernel fg_out_pull co=fg_full.co block=256 wave=32 args=p,p,p,p,p,p,p,p,p,p,p,u,u,u,u,u,u,u,u,u,u
// @wg-kernel fg_out_pull_pack co=- entry=fg_out_pull block=256 wave=32 args=p,p,p,p,p,p,p,p,p,p,p,u,u,u,u,u,u,u,u,u,u defines=FGP_PACK=1
// @wg-kernel fg_out_pull_b co=- entry=fg_out_pull block=256 wave=32 args=p,p,p,p,p,p,p,p,p,p,p,u,u,u,u,u,u,u,u,u,u defines=FG_BIL128=1
// @wg-kernel fg_out_pull_pack_b co=- entry=fg_out_pull block=256 wave=32 args=p,p,p,p,p,p,p,p,p,p,p,u,u,u,u,u,u,u,u,u,u defines=FGP_PACK=1,FG_BIL128=1
// @wg-kernel fg_out_pull_q co=- entry=fg_out_pull block=256 wave=32 args=p,p,p,p,p,p,p,p,p,p,p,u,u,u,u,u,u,u,u,u,u defines=FGP_POW2=1
// @wg-kernel fg_out_pull_pack_q co=- entry=fg_out_pull block=256 wave=32 args=p,p,p,p,p,p,p,p,p,p,p,u,u,u,u,u,u,u,u,u,u defines=FGP_PACK=1,FGP_POW2=1
// @wg-kernel fg_out_pull_bq co=- entry=fg_out_pull block=256 wave=32 args=p,p,p,p,p,p,p,p,p,p,p,u,u,u,u,u,u,u,u,u,u defines=FGP_POW2=1,FG_BIL128=1
// @wg-kernel fg_out_pull_pack_bq co=- entry=fg_out_pull block=256 wave=32 args=p,p,p,p,p,p,p,p,p,p,p,u,u,u,u,u,u,u,u,u,u defines=FGP_PACK=1,FGP_POW2=1,FG_BIL128=1
// @wg-kernel fg_out_pull_mid co=fg_full.co block=256 wave=32 args=p,p,p,p,p,u,u,u,u,u defines=FGP_MID=1
// @wg-kernel fg_out_pull_x3 co=- entry=fg_out_pull block=256 wave=32 args=p,p,p,p,p,p,p,p,p,p,p,u,u,u,u,u,u,u,u,u,u defines=FGP_X3=1
// @wg-kernel fg_out_pull_pack_x3 co=- entry=fg_out_pull block=256 wave=32 args=p,p,p,p,p,p,p,p,p,p,p,u,u,u,u,u,u,u,u,u,u defines=FGP_PACK=1,FGP_X3=1
#include "wg_platform.hlsli"
#include "fg_full.hlsli"
#include "fg_pack.hlsli"

#ifndef FGP_MID
#define FGP_MID 0
#endif
#ifndef FGP_PACK
#define FGP_PACK 0
#endif
#ifndef FGP_X3
#define FGP_X3 0
#endif
#ifndef FGP_POW2
#define FGP_POW2 0
#endif

groupshared float4 fgp_wave_sums[8];

int fgp_size(int n, uint lvl) { return (n + (1 << lvl) - 1) >> lvl; }

float4 fgp_level(float4 acc, uint m)
{
#if FGP_MID
    return fgf_level(acc, (float)(1u << (2u * m)));
#else
    // The first tile receives saturated finite RGB and weights of zero or one.
    // With no contributing pixels RGB is zero, so its denominator can be one.
    const float den = max(acc.w, 1.0f);
    const float scale = asfloat((127u - 2u * m) << 23u);
#if FGP_POW2
    // A17: a tile with no hole has acc.w = 4^m exactly, a power of two: den = acc.w, its reciprocal is exactly scale and
    // every refinement step of the division is exact (the residual is 0), so acc.xyz / den = acc.xyz * scale bit for
    // bit (zeros keep their sign through the select below, as before). Only tiles with a hole divide.
    float3 rgb = acc.xyz * scale;
    [branch] if (acc.w != (float)(1u << (2u * m)))
        rgb = fgf_div_normal3(acc.xyz, den, fgf_div_reciprocal(den));
#else
    const float3 rgb = fgf_div_normal3(acc.xyz, den, fgf_div_reciprocal(den));
#endif
    return float4(select(acc.xyz == 0.0f, acc.xyz, rgb), acc.w * scale);
#endif
}

void fgp_write(uint dst, int px, int py, int w, int h, uint lvl, uint m, float4 acc)
{
    const int X = px >> m, Y = py >> m;
    if (X < fgp_size(w, lvl) && Y < fgp_size(h, lvl))
        fgf_st4h(dst, Y * fgp_size(w, lvl) + X, fgp_level(acc, m));
}

float4 fgp_pair(float4 acc, uint lane, uint stride)
{
    return acc + WaveReadLaneAt(acc, lane ^ stride);
}

void fgp_reduce(float4 acc, int2 origin, int w, int h, uint lvl0, uint4 dst)
{
    const uint l = wg_lane_id(), lane = wg_wave_lane();
    const int px = origin.x + (int)fgf_mx(l), py = origin.y + (int)fgf_my(l);
#if FGP_MID
    acc = fgp_pair(fgp_pair(acc, lane, 1u), lane, 2u);
#else
    acc += QuadReadAcrossX(acc);
    acc += QuadReadAcrossY(acc);
#endif
    if ((l & 3u) == 0u) fgp_write(dst.x, px, py, w, h, lvl0 + 1u, 1u, acc);
    acc = fgp_pair(fgp_pair(acc, lane, 4u), lane, 8u);
    if ((l & 15u) == 0u) fgp_write(dst.y, px, py, w, h, lvl0 + 2u, 2u, acc);
    acc = fgp_pair(acc, lane, 16u);
    if (lane == 0u) fgp_wave_sums[l >> 5] = acc;
    wg_barrier_lds();

    // The first wave gathers the eight partial sums and continues the same tree.
    if (l < 32u) {
        const uint part = lane & 7u;
        acc = fgp_pair(fgp_wave_sums[part], lane, 1u);
        if (l < 8u && (part & 1u) == 0u) {
            const uint tile_lane = part * 32u;
            fgp_write(dst.z, origin.x + (int)fgf_mx(tile_lane), origin.y + (int)fgf_my(tile_lane),
                      w, h, lvl0 + 3u, 3u, acc);
        }
        acc = fgp_pair(fgp_pair(acc, lane, 2u), lane, 4u);
        if (l == 0u) fgp_write(dst.w, origin.x, origin.y, w, h, lvl0 + 4u, 4u, acc);
    }
}

WG_ARGS_BEGIN
#if FGP_MID
    uint lv4, lv5, lv6, lv7, lv8;
    int w4, h4, tiles_x, Wo, Ho;
#else
    uint x0, x1, xw, n2, sigtab, outp, fine, lv1, lv2, lv3, lv4;
    int Wo, Ho, W2, H2, k, tiles_x, out_pitch, out_x, out_y;
    uint rk_bits;   // 1 / k (float bits)
#endif
WG_ARGS_END

#if FGP_MID
WG_KERNEL(fg_out_pull_mid)
{
    FGF_SETUP();
    const uint l = wg_lane_id();
    const uint t = wg_block_id();
    const int2 origin = int2(t % (uint)tiles_x, t / (uint)tiles_x) * 16;
    const int ex = origin.x + (int)fgf_mx(l), ey = origin.y + (int)fgf_my(l);
    float4 acc = 0.0f;
    if (ex < w4 && ey < h4) {
        const float4 e = fgf_ld4h(lv4, ey * w4 + ex);
        acc = float4(e.xyz * e.w, e.w);
    }
    fgp_reduce(acc, origin, Wo, Ho, 4u, uint4(lv5, lv6, lv7, lv8));
}
#else
// net2's bilinear rule: fma(1 - fy, fma(1 - fx, q00, fx q10), fy fma(1 - fx, q01, fx q11))
float fgp_bil(float fx, float fy, float gx, float gy, float q00, float q10, float q01, float q11)
{
    return mad(gy, mad(gx, q00, fx * q10), fy * mad(gx, q01, fx * q11));
}

#if FGP_X3
// texel i of a 3 x float16 image: its 6 bytes are inside the 8 bytes from the dword at or below byte 6 i
float4 fgp_ld3h(uint p, int i)
{
    const uint a = (uint)i * 6u;
    const uint2 d = wg_ld64(p, a & ~3u);
    const bool hi = (a & 2u) != 0u;
    const uint rg = hi ? ((d.x >> 16) | (d.y << 16)) : d.x;
    const uint b = hi ? (d.y >> 16) : (d.y & 0xffffu);
    return float4(wg_h2f_lo(rg), wg_h2f_hi(rg), wg_h2f(b), 0.0f);
}
float4 fgp_bil3h(uint p, FgfTap t) { return fgf_mix4(t, fgp_ld3h(p, t.i00), fgp_ld3h(p, t.i10), fgp_ld3h(p, t.i01), fgp_ld3h(p, t.i11)); }
#define FGP_BIL_X(p, t) fgp_bil3h(p, t)
#else
#define FGP_BIL_X(p, t) fgf_bil4h(p, t)
#endif

// One 20-bit mask per row of the 16 x 16 tile with its two-pixel halo.
groupshared uint fgp_need_rows[20];
groupshared uint3 fgp_n2_tile[25];

uint3 fgp_n2_at(int x, int y)
{
    const uint4 d = wg_ld128(n2, (uint)(y * W2 + x) * 16u);
    return uint3(d.x, d.y, wg_ld32(sigtab, (d.z & 0xffffu) * 4u));
}

void fgp_stage(int2 origin, bool cache_n2)
{
    const uint l = wg_lane_id(), lane = wg_wave_lane();
    const int x = origin.x + (int)lane - 2;
    for (uint row = l >> 5; row < 20u; row += 8u) {
        const int y = origin.y + (int)row - 2;
        const bool inside = x >= 0 && x < Wo && y >= 0 && y < Ho;
        const bool need = lane < 20u && (!inside || fgf_ld1h(xw, y * Wo + x) > 0.0f);
        const uint mask = WaveActiveBallot(need).x;
        if (lane == 0u) fgp_need_rows[row] = mask;
    }

    // With k = 4, sixteen adjacent pixels sample the same four taps. The tile spans
    // four coarse cells in each dimension and needs one extra tap on the right and bottom.
    if (cache_n2 && l < 25u) {
        const int x0 = min((origin.x >> 2) + (int)(l % 5u), W2 - 1);
        const int y0 = min((origin.y >> 2) + (int)(l / 5u), H2 - 1);
        fgp_n2_tile[l] = fgp_n2_at(x0, y0);
    }
    wg_barrier_lds();
}

WG_KERNEL(fg_out_pull)
{
    FGF_SETUP();
    const uint l = wg_lane_id();
    const uint t = wg_block_id();
    const int2 origin = int2(t % (uint)tiles_x, t / (uint)tiles_x) * 16;
    const uint lx = fgf_mx(l), ly = fgf_my(l);
    const int px = origin.x + (int)lx, py = origin.y + (int)ly;
    const bool cache_n2 = k == 4 && rk_bits == 0x3e800000u;
    fgp_stage(origin, cache_n2);

    float4 acc = 0.0f;
    float3 out_c = 0.0f;
    bool hole = false;
    const bool in_bounds = px < Wo && py < Ho;
    if (in_bounds) {
        // Every 3 x 3 window containing this pixel must have at least one need bit.
        uint m = 0u;
        [unroll] for (uint dy = 0u; dy < 5u; ++dy)
            m |= ((fgp_need_rows[ly + dy] >> lx) & 31u) << (dy * 5u);
        // OR down three rows, then across three columns. Only the nine valid
        // window origins are tested, so shifts across packed row boundaries are harmless.
        const uint rows = m | (m >> 5u) | (m >> 10u);
        const uint windows = rows | (rows >> 1u) | (rows >> 2u);
        const uint origins = 7u | (7u << 5u) | (7u << 10u);
        hole = (windows & origins) == origins;
        float3 c = 0.0f;
        [branch] if (!hole) {
            // net2's output at (x / k, y / k)
            const float rk = asfloat(rk_bits);
            const float s = min((float)px * rk, (float)(W2 - 1)), tt = min((float)py * rk, (float)(H2 - 1));
            const float sf = floor(s), tf = floor(tt);
            const int i0 = (int)sf, j0 = (int)tf, i1 = min(i0 + 1, W2 - 1), j1 = min(j0 + 1, H2 - 1);
            const float fx = s - sf, fy = tt - tf, gx = 1.0f - fx, gy = 1.0f - fy;
            uint3 d00, d10, d01, d11;
            [branch] if (cache_n2) {
                const uint i = (ly >> 2) * 5u + (lx >> 2);
                d00 = fgp_n2_tile[i];
                d10 = fgp_n2_tile[i + 1u];
                d01 = fgp_n2_tile[i + 5u];
                d11 = fgp_n2_tile[i + 6u];
            } else {
                d00 = fgp_n2_at(i0, j0);
                d10 = fgp_n2_at(i1, j0);
                d01 = fgp_n2_at(i0, j1);
                d11 = fgp_n2_at(i1, j1);
            }
            const float sigma = fgp_bil(fx, fy, gx, gy, asfloat(d00.z), asfloat(d10.z), asfloat(d01.z), asfloat(d11.z));
            const float vax = fgp_bil(fx, fy, gx, gy, wg_h2f_lo(d00.x), wg_h2f_lo(d10.x), wg_h2f_lo(d01.x), wg_h2f_lo(d11.x));
            const float vay = fgp_bil(fx, fy, gx, gy, wg_h2f_hi(d00.x), wg_h2f_hi(d10.x), wg_h2f_hi(d01.x), wg_h2f_hi(d11.x));
            const float vbx = fgp_bil(fx, fy, gx, gy, wg_h2f_lo(d00.y), wg_h2f_lo(d10.y), wg_h2f_lo(d01.y), wg_h2f_lo(d11.y));
            const float vby = fgp_bil(fx, fy, gx, gy, wg_h2f_hi(d00.y), wg_h2f_hi(d10.y), wg_h2f_hi(d01.y), wg_h2f_hi(d11.y));
            // the two candidates re-sampled at the pixel plus k times the vectors (pixel units)
            const float kf = (float)k;
            const float3 A = FGP_BIL_X(x0, fgf_tap(vax * kf + (float)px, vay * kf + (float)py, Wo, Ho)).xyz;
            const float3 B = FGP_BIL_X(x1, fgf_tap(vbx * kf + (float)px, vby * kf + (float)py, Wo, Ho)).xyz;
            c = fgf_sat3(mad(sigma, A, (1.0f - sigma) * B));
            if (isnan(abs(c.z * (c.x * c.y)))) c = float3(0.0f, 1.0f, 1.0f);
        }
        out_c = c;
        acc = float4(c, hole ? 0.0f : 1.0f);
    }
    {
        v8f pv;
        pv.v[0] = fgf_opq(out_c.x); pv.v[1] = fgf_opq(out_c.y); pv.v[2] = fgf_opq(out_c.z);
        [unroll] for (uint ci = 3u; ci < 8u; ++ci) pv.v[ci] = 0.0f;
        const uint4 h = wg_to_f16_ags(pv);
#if FGP_PACK
        const uint word = fgx_pack_pixel(uint2(h.x, (h.y & 0xffffu) | 0x3c000000u), (uint)out_x);
        const uint pair = QuadReadAcrossX(word);
        // Adjacent Morton lanes are a horizontal pair (the origin is a multiple of 16); the rows are 8-byte aligned.
        if (in_bounds && (px & 1) == 0) {
            const uint at = (uint)(py + out_y) * (uint)out_pitch + (uint)px * 4u;
            if (px + 1 < Wo) wg_st64(outp, at, uint2(word, pair));
            else wg_st32(outp, at, word);
        }
#else
        if (in_bounds)
            wg_st64(outp, (uint)((py + out_y) * out_pitch + px + out_x) * 8u, uint2(h.x, (h.y & 0xffffu) | 0x3c000000u));
#endif
    }
    fgf_store_morton_flag(fine, px, py, Wo, Ho, hole);
    fgp_reduce(acc, origin, Wo, Ho, 0u, uint4(lv1, lv2, lv3, lv4));
}
#endif
