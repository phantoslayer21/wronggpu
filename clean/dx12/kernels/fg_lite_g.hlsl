// SPDX-License-Identifier: GPL-3.0-only
// Frame generation (issue #96), light path: G2 intermediate-time scatter and G3 blend weights (clean/fg/reference/
// stages_common.py interm_scatter, interm_inputs, blend_weights). D3D12 only.
//
// fg_lite_interm: one lane per scatter-grid pixel; pass 0 (A): displacement (1 - t) (Mv W1, Mv H1) and keys from Zc,
//   written unless both the truncated-key and the displacement test fail; pass 1 (B): displacement t F and keys from
//   Zp, written when the displacement test passes. Footprint 2 at a depth edge (key range over 3 x 3 > tau), else from
//   the spread of the warped 3 x 3 positions (2..5 per axis). Signed 64-bit atomic minimum (SM 6.6) of (key bits << 32
//   | float16 of the vector back to the source frame, normalised).
// fg_lite_interm_both: both passes in one launch (pass A into MA, then pass B into MB, the same tile geometry and the same
//   arithmetic per lane as fg_lite_interm_tiled), with the scatters of a tile binned in LDS first: a 32 x 32 window of
//   signed 64-bit cells (initialised to the empty element 0x7FFFFFFF7FFFFFFF) placed around the tile's centre
//   displacement takes the atomic minima of every target inside it, and each cell below the empty value is then
//   minimised into the global element once. A target outside the window goes to the global element directly. The minimum
//   of signed 64-bit values is associative and commutative: every global element ends as the minimum of its initial value and
//   all the values aimed at it, the same bits as one global atomic minimum per scatter, in any order.
// fg_lite_weights: W = (sigma(NA at the MB vector - NA co-located), sigma(NB at the MA vector - NB co-located)).
//
// @wg-kernel fg_lite_interm co=- block=256 wave=32 args=u,p,p,p,u,u,u,u,u,u,u defines=FG_K=1
// @wg-kernel fg_lite_interm_tiled co=- entry=fg_lite_interm block=256 wave=32 args=u,p,p,p,u,u,u,u,u,u,u defines=FG_K=1,FG_TILE=1
// @wg-kernel fg_lite_interm_both co=- block=256 wave=32 args=p,p,p,p,p,p,u,u,u,u,u,u,u defines=FG_K=3
// @wg-kernel fg_lite_weights co=- block=256 wave=32 args=p,p,p,p,p,u,u,u,u defines=FG_K=2
// @wg-kernel fg_lite_weights_b co=- entry=fg_lite_weights block=256 wave=32 args=p,p,p,p,p,u,u,u,u defines=FG_K=2,FG_BIL128=1
#include "wg_platform.hlsli"
#include "fg_lite_common.hlsli"

#if FG_K == 1
WG_ARGS_BEGIN
    uint pass_, src, zimg, target, w1, h1, tbits, s_k, inverted, t2, tau;
WG_ARGS_END

float2 fg_disp(int x, int y)
{
    const float t = asfloat(tbits);
    if (pass_ == 0u)
    {
        const float omt = 1.0f - t;
        const float4 m = fg_read_int(src, 5u, (int)w1, (int)h1, x, y);
        return float2(fg_mul(omt, fg_mul(m.x, (float)w1)), fg_mul(omt, fg_mul(m.y, (float)h1)));
    }
    const float4 f = fg_read_int(src, 2u, (int)w1, (int)h1, x, y);
    return float2(fg_mul(t, f.x), fg_mul(t, f.y));
}

float fg_key_at(int x, int y)
{
    const uint Z = wg_ld32(zimg, (uint)(clamp(y, 0, (int)h1 - 1) * (int)w1 + clamp(x, 0, (int)w1 - 1)) * 4u);
    return fg_depth_key(fg_unpack_depth(Z, 0.0f, 0.0f).x, asfloat(s_k), inverted);
}

int fg_radius(float qmax, float qmin, float qp)
{
    const float a = max(abs(qmax - qp), abs(qmin - qp));
    return clamp((int)fg_med3(ceil(a + 0.5f), 2.0f, 5.0f), 2, 5);
}

#if FG_TILE
// A four-pixel halo covers both the 3x3 depth test and the largest 5x5 scatter.
static const int fg_tile_width = 24;
static const int fg_halo = 4;
groupshared float fg_tile_key[fg_tile_width * fg_tile_width];
groupshared float fg_tile_dx[fg_tile_width * fg_tile_width];
groupshared float fg_tile_dy[fg_tile_width * fg_tile_width];

float2 fg_tile_disp(int i) { return float2(fg_tile_dx[i], fg_tile_dy[i]); }
#endif

WG_KERNEL(fg_lite_interm)
{
    FG_SETUP();
#if FG_TILE
    const uint tiles_x = (w1 + 15u) >> 4u, block = wg_block_id(), lane = wg_lane_id();
    const int2 origin = int2(block % tiles_x, block / tiles_x) * 16;
    for (uint i = lane; i < fg_tile_width * fg_tile_width; i += 256u) {
        const int2 p = origin + int2(i % fg_tile_width, i / fg_tile_width) - fg_halo;
        const float2 d = fg_disp(p.x, p.y);
        fg_tile_key[i] = fg_key_at(p.x, p.y);
        fg_tile_dx[i] = d.x;
        fg_tile_dy[i] = d.y;
    }
    wg_barrier_lds();
    const int2 local = int2(lane & 15u, lane >> 4u) + fg_halo;
    const int2 xy = origin + local - fg_halo;
    if (xy.x >= (int)w1 || xy.y >= (int)h1) return;
    const int center = local.y * fg_tile_width + local.x;
#else
    const uint idx = wg_block_id() * 256u + wg_lane_id();
    if (idx >= w1 * h1) return;
    const int2 xy = int2(fg_lite_tile_pixel(idx, w1, h1));
#endif
    const int x = xy.x, y = xy.y;
    const float tauf = asfloat(tau);
#if FG_TILE
    const float kp = fg_tile_key[center];
    const float2 dp = fg_tile_disp(center);
#else
    const float kp = fg_key_at(x, y);
    const float2 dp = fg_disp(x, y);
#endif
    float kmax = -FG_INF, kmin = FG_INF, qxmax = -10000.0f, qymax = -10000.0f, qxmin = 10000.0f, qymin = 10000.0f;
    [unroll] for (int j = -1; j <= 1; ++j)
        [unroll] for (int i = -1; i <= 1; ++i)
        {
#if FG_TILE
            const int neighbor = center + j * fg_tile_width + i;
            const float kn = fg_tile_key[neighbor];
            const float2 dn = fg_tile_disp(neighbor);
#else
            const bool centre = i == 0 && j == 0;
            const float kn = centre ? kp : fg_key_at(x + i, y + j);
            const float2 dn = centre ? dp : fg_disp(x + i, y + j);
#endif
            kmax = max(kmax, kn);
            kmin = min(kmin, kn);
            const float qx = fg_opq((float)(x + i) + 0.5f) + dn.x, qy = fg_opq((float)(y + j) + 0.5f) + dn.y;
            qxmax = max(qxmax, qx); qxmin = min(qxmin, qx);
            qymax = max(qymax, qy); qymin = min(qymin, qy);
        }
    const float qpx = fg_opq((float)x + 0.5f) + dp.x, qpy = fg_opq((float)y + 0.5f) + dp.y;
    const bool edge = (kmax - kp) > tauf || (kp - kmin) > tauf;
    const int rx = edge ? 2 : fg_radius(qxmax, qxmin, qpx);
    const int ry = edge ? 2 : fg_radius(qymax, qymin, qpy);
    const uint hi = asuint(kp);
    const uint lo = (fg_f2h(fg_adiv(-dp.x, (float)w1)) << 16) | fg_f2h(fg_adiv(-dp.y, (float)h1));
    const float thr = max(fg_adiv(fg_sq(dp.x, dp.y), asfloat(t2)), 1.0f);
    const int sx = fg_sgn(dp.x), sy = fg_sgn(dp.y);
    const int bx = x + (int)dp.x, by = y + (int)dp.y;
    [loop] for (int jj = 0; jj < ry; ++jj)
        [loop] for (int ii = 0; ii < rx; ++ii)
        {
            const int tx = bx + ii * sx, ty = by + jj * sy;
            if (tx < 0 || tx >= (int)w1 || ty < 0 || ty >= (int)h1) continue;
#if FG_TILE
            const int neighbor = center + jj * sy * fg_tile_width + ii * sx;
            const float2 nd = fg_tile_disp(neighbor);
#else
            const int nx = x + ii * sx, ny = y + jj * sy;
            const float2 nd = fg_disp(nx, ny);
#endif
            const float dd = fg_sq(nd.x - dp.x, nd.y - dp.y);
            bool ok;
            if (pass_ == 0u)
            {
#if FG_TILE
                const float kt = (float)(uint)fg_tile_key[neighbor];
#else
                const float kt = (float)(uint)fg_key_at(nx, ny);
#endif
                ok = !(!(abs(kt - kp) < 3.0f) && !(dd < thr));
            }
            else
                ok = dd < thr;
            if (ok) fg_atomic_min64(target, (uint)(ty * (int)w1 + tx), hi, lo);
        }
}
#endif


#if FG_K == 3
WG_ARGS_BEGIN
    uint src_a, z_a, tgt_a, src_b, z_b, tgt_b, w1, h1, tbits, s_k, inverted, t2, tau;
WG_ARGS_END

static const int fg_tile_width = 24;
static const int fg_halo = 4;
static const int64_t fg_empty64 = 0x7FFFFFFF7FFFFFFFll;
groupshared float fg_tile_key[fg_tile_width * fg_tile_width];
groupshared float fg_tile_dx[fg_tile_width * fg_tile_width];
groupshared float fg_tile_dy[fg_tile_width * fg_tile_width];
groupshared int64_t fg_bin[1024];

float2 fg_disp(uint pass_, uint src, int x, int y)
{
    const float t = asfloat(tbits);
    if (pass_ == 0u)
    {
        const float omt = 1.0f - t;
        const float4 m = fg_read_int(src, 5u, (int)w1, (int)h1, x, y);
        return float2(fg_mul(omt, fg_mul(m.x, (float)w1)), fg_mul(omt, fg_mul(m.y, (float)h1)));
    }
    const float4 f = fg_read_int(src, 2u, (int)w1, (int)h1, x, y);
    return float2(fg_mul(t, f.x), fg_mul(t, f.y));
}

float fg_key_at(uint zimg, int x, int y)
{
    const uint Z = wg_ld32(zimg, (uint)(clamp(y, 0, (int)h1 - 1) * (int)w1 + clamp(x, 0, (int)w1 - 1)) * 4u);
    return fg_depth_key(fg_unpack_depth(Z, 0.0f, 0.0f).x, asfloat(s_k), inverted);
}

int fg_radius(float qmax, float qmin, float qp)
{
    const float a = max(abs(qmax - qp), abs(qmin - qp));
    return clamp((int)fg_med3(ceil(a + 0.5f), 2.0f, 5.0f), 2, 5);
}

float2 fg_tile_disp(int i) { return float2(fg_tile_dx[i], fg_tile_dy[i]); }

void fg_min64_global(uint base, uint idx, int64_t v)
{
    wg_mem_gc.InterlockedMin64(base + idx * 8u, v);
}

void fg_interm_pass(uint pass_, uint src, uint zimg, uint target)
{
    const uint tiles_x = (w1 + 15u) >> 4u, block = wg_block_id(), lane = wg_lane_id();
    const int2 origin = int2(block % tiles_x, block / tiles_x) * 16;
    for (uint i = lane; i < fg_tile_width * fg_tile_width; i += 256u) {
        const int2 p = origin + int2(i % fg_tile_width, i / fg_tile_width) - fg_halo;
        const float2 d = fg_disp(pass_, src, p.x, p.y);
        fg_tile_key[i] = fg_key_at(zimg, p.x, p.y);
        fg_tile_dx[i] = d.x;
        fg_tile_dy[i] = d.y;
    }
    for (uint c = lane; c < 1024u; c += 256u) fg_bin[c] = fg_empty64;
    wg_barrier_lds();
    // the window follows the displacement of the tile's centre pixel (any window gives the same result)
    const int centre_tile = (fg_halo + 8) * fg_tile_width + fg_halo + 8;
    const int wx0 = origin.x + (int)clamp(fg_tile_dx[centre_tile], -16384.0f, 16384.0f) - 8;
    const int wy0 = origin.y + (int)clamp(fg_tile_dy[centre_tile], -16384.0f, 16384.0f) - 8;
    const int2 local = int2(lane & 15u, lane >> 4u) + fg_halo;
    const int2 xy = origin + local - fg_halo;
    if (xy.x < (int)w1 && xy.y < (int)h1)
    {
        const int center = local.y * fg_tile_width + local.x;
        const int x = xy.x, y = xy.y;
        const float tauf = asfloat(tau);
        const float kp = fg_tile_key[center];
        const float2 dp = fg_tile_disp(center);
        float kmax = -FG_INF, kmin = FG_INF, qxmax = -10000.0f, qymax = -10000.0f, qxmin = 10000.0f, qymin = 10000.0f;
        [unroll] for (int j = -1; j <= 1; ++j)
            [unroll] for (int i = -1; i <= 1; ++i)
            {
                const int neighbor = center + j * fg_tile_width + i;
                const float kn = fg_tile_key[neighbor];
                const float2 dn = fg_tile_disp(neighbor);
                kmax = max(kmax, kn);
                kmin = min(kmin, kn);
                const float qx = fg_opq((float)(x + i) + 0.5f) + dn.x, qy = fg_opq((float)(y + j) + 0.5f) + dn.y;
                qxmax = max(qxmax, qx); qxmin = min(qxmin, qx);
                qymax = max(qymax, qy); qymin = min(qymin, qy);
            }
        const float qpx = fg_opq((float)x + 0.5f) + dp.x, qpy = fg_opq((float)y + 0.5f) + dp.y;
        const bool edge = (kmax - kp) > tauf || (kp - kmin) > tauf;
        const int rx = edge ? 2 : fg_radius(qxmax, qxmin, qpx);
        const int ry = edge ? 2 : fg_radius(qymax, qymin, qpy);
        const uint hi = asuint(kp);
        const uint lo = (fg_f2h(fg_adiv(-dp.x, (float)w1)) << 16) | fg_f2h(fg_adiv(-dp.y, (float)h1));
        const int64_t value = (int64_t)(((uint64_t)hi << 32) | (uint64_t)lo);
        const float thr = max(fg_adiv(fg_sq(dp.x, dp.y), asfloat(t2)), 1.0f);
        const int sx = fg_sgn(dp.x), sy = fg_sgn(dp.y);
        const int bx = x + (int)dp.x, by = y + (int)dp.y;
        [loop] for (int jj = 0; jj < ry; ++jj)
            [loop] for (int ii = 0; ii < rx; ++ii)
            {
                const int tx = bx + ii * sx, ty = by + jj * sy;
                if (tx < 0 || tx >= (int)w1 || ty < 0 || ty >= (int)h1) continue;
                const int neighbor = center + jj * sy * fg_tile_width + ii * sx;
                const float2 nd = fg_tile_disp(neighbor);
                const float dd = fg_sq(nd.x - dp.x, nd.y - dp.y);
                bool ok;
                if (pass_ == 0u)
                {
                    const float kt = (float)(uint)fg_tile_key[neighbor];
                    ok = !(!(abs(kt - kp) < 3.0f) && !(dd < thr));
                }
                else
                    ok = dd < thr;
                if (ok)
                {
                    const int cx = tx - wx0, cy = ty - wy0;
                    if ((uint)cx < 32u && (uint)cy < 32u)
                    {
                        int64_t before;
                        InterlockedMin(fg_bin[cy * 32 + cx], value, before);
                    }
                    else
                        fg_min64_global(target, (uint)(ty * (int)w1 + tx), value);
                }
            }
    }
    wg_barrier_lds();
    for (uint c = lane; c < 1024u; c += 256u) {
        const int64_t v = fg_bin[c];
        if (v < fg_empty64) fg_min64_global(target, (uint)((wy0 + (int)(c >> 5u)) * (int)w1 + wx0 + (int)(c & 31u)), v);
    }
    wg_barrier_lds();
}

WG_KERNEL(fg_lite_interm_both)
{
    FG_SETUP();
    fg_interm_pass(0u, src_a, z_a, tgt_a);
    fg_interm_pass(1u, src_b, z_b, tgt_b);
}
#endif

#if FG_K == 2
WG_ARGS_BEGIN
    uint ma, mb, na, nb, wout, w1, h1, wd, hd;
WG_ARGS_END

WG_KERNEL(fg_lite_weights)
{
    FG_SETUP();
    const uint idx = wg_block_id() * 256u + wg_lane_id();
    if (idx >= w1 * h1) return;
    const uint x = idx % w1, y = idx / w1;
    const float u = fg_centre(x, w1), v = fg_centre(y, h1);
    const uint mi = fg_index_rint(v, h1) * w1 + fg_index_rint(u, w1);
    const float3 a = fg_vec64(ma, mi), b = fg_vec64(mb, mi);
    const uint ji = fg_index_rint(v, hd) * wd + fg_index_rint(u, wd);
    const float A1 = fg_read_linear(na, 2u, (int)wd, (int)hd, u + b.x, v + b.y).x;
    const float A0 = fg_texel(na, 2u, ji).y;
    const float B1 = fg_read_linear(nb, 2u, (int)wd, (int)hd, u + a.x, v + a.y).x;
    const float B0 = fg_texel(nb, 2u, ji).y;
    fg_st_h2(wout, idx, fg_sigmoid(A1, A0), fg_sigmoid(B1, B0));
}
#endif
