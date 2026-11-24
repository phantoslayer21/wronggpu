// SPDX-License-Identifier: MIT
// Frame generation (issue #96), light path: R6, the previous-to-current chain (clean/fg/reference/stages_common.py
// p2c_*). D3D12 only.
//
// fg_lite_p2c_scatter: one lane per current scatter-grid pixel: camera consistency (ClipToPrevClip), the camera mask
//   Kc, and the payload (depth key bits 31..1, consistent bit, float16 motion in pixels) by signed 64-bit atomic
//   minimum (SM 6.6) into S at up to four targets.
// fg_lite_p2c_unpack: one lane per previous scatter-grid pixel: F (float16 x 2), E, the flag word and the float32
//   vector Vf.
// fg_lite_p2c_inpaint: the hole rule over the 5 x 5 neighbourhood of bit 1 -> G = (w ? 2 : 0) | (inpaint ? 1 : 0).
// fg_lite_p2c_pull: one pull level: mode 0 from the base pixels (G, Vf), 1 from the previous level's float32 sums, 2
//   from a level's float16 elements; writes the level's sums (when sums_out != 0) and its float16 x 3 elements.
// fg_lite_p2c_push: fills one level in place from its filled parent (9/16, 3/16, 3/16, 1/16 taps).
// fg_lite_p2c_fine: F := v_up from level 1 at every pixel with G bit 0.
// fg_lite_p2c_fine_mask (the net1 grid is the scatter grid: d = dp): the fine pass and the hole mask in one launch. The lane
//   that finishes F at its pixel (v_up, or the unpack pass's value where G bit 0 is clear) continues with
//   fg_lite_hole_mask's work on that value: the mask reads F at the pixel centre of the same grid, which is its own pixel
//   (the nearest-index product of a centre and the grid size is within 1e-3 of x + 0.5), and writes zeros only. Hc := 1
//   must be filled before the launch (the hole mask's zeros land in it).
// fg_lite_p2c_inpaint_tiled: the same hole rule on 32 x 8 pixel tiles. Each wave gathers one row of the tile (and the two rows
//   above and below with the first four waves) as bit masks with ballots, so a pixel's 5 x 5 neighbourhood costs 1.7 loads
//   instead of 5; the rule (every 3 x 3 window of the neighbourhood holds a flagged pixel = erosion of the 3 x 3 dilation of
//   the flag-bit-1 image, zero outside the image) runs on the masks.
//
// @wg-kernel fg_lite_p2c_scatter co=- block=256 wave=32 args=p,p,p,p,u,u,u,u,u,u,u,u,u,u,u,u,u,u,u,u,u,u,u,u,u,u,u,u defines=FG_K=1
// @wg-kernel fg_lite_p2c_unpack co=- block=256 wave=32 args=p,p,p,p,p,p,p,p,u,u,u,u,u,u,u,u,u,u,u,u,u,u,u,u,u,u,u,u,u,u,u,u defines=FG_K=2
// @wg-kernel fg_lite_p2c_inpaint co=- block=256 wave=32 args=p,p,u,u defines=FG_K=3
// @wg-kernel fg_lite_p2c_inpaint_mut co=- entry=fg_lite_p2c_inpaint block=256 wave=32 args=p,p,u,u defines=FG_K=3,FG_MUT=1
// @wg-kernel fg_lite_p2c_pull co=- block=256 wave=32 args=u,p,p,p,p,u,u,u,u,u defines=FG_K=4
// @wg-kernel fg_lite_p2c_push co=- block=256 wave=32 args=p,u,u,p,u,u defines=FG_K=5
// @wg-kernel fg_lite_p2c_pull_tiled co=- entry=fg_lite_p2c_pull block=256 wave=32 args=u,p,p,p,p,u,u,u,u,u,p,p,p,u defines=FG_K=4,FG_PULL_TILE=1
// @wg-kernel fg_lite_p2c_push_tail co=- block=256 wave=32 args=u,p,p,p,p,p,p,p,p,u,u,u,u,u,u,u,u,u,u,u,u,u,u,u,u defines=FG_K=7
// @wg-kernel fg_lite_p2c_fine co=- block=256 wave=32 args=p,u,u,p,p,u,u defines=FG_K=6
// @wg-kernel fg_lite_p2c_fine_mask co=- entry=fg_lite_p2c_fine block=256 wave=32 args=p,u,u,p,p,u,u,p defines=FG_K=6,FG_MASK=1
// @wg-kernel fg_lite_p2c_inpaint_tiled co=- entry=fg_lite_p2c_inpaint block=256 wave=32 args=p,p,u,u defines=FG_K=3,FG_IP_TILE=1
#include "wg_platform.hlsli"
#include "fg_lite_common.hlsli"

#if FG_K == 1
WG_ARGS_BEGIN
    uint zc, mv, sbuf, kc, w1, h1;
    uint m0, m1, m2, m3, m4, m5, m6, m7, m8, m9, m10, m11, m12, m13, m14, m15;
    uint t1, t2, s_k, inverted, stepx, stepy;
WG_ARGS_END

WG_KERNEL(fg_lite_p2c_scatter)
{
    FG_SETUP();
    const uint idx = wg_block_id() * 256u + wg_lane_id();
    if (idx >= w1 * h1) return;
    const int x = (int)(idx % w1), y = (int)(idx / w1);
    const float u = fg_centre((uint)x, w1), v = fg_centre((uint)y, h1);
    const float3 zo = fg_unpack_depth(wg_ld32(zc, idx * 4u), asfloat(stepx), asfloat(stepy));
    const float kap = fg_depth_key(zo.x, asfloat(s_k), inverted);
    const float3 cam = fg_camera(u + zo.y, v + zo.z, zo.x, fg_f(m0), fg_f(m1), fg_f(m3), fg_f(m4), fg_f(m5), fg_f(m7),
                                 fg_f(m8), fg_f(m9), fg_f(m11), fg_f(m12), fg_f(m13), fg_f(m15), w1, h1);
    const float4 mvv = fg_texel(mv, 5u, idx);
    const float mx = fg_mul(mvv.x, (float)w1), my = fg_mul(mvv.y, (float)h1);
    const float msq = fg_sq(mx, my);
    const float dd = fg_sq(mx - cam.x, my - cam.y);
    const bool cons = !(cam.z <= 0.0f) && dd <= max(fg_adiv(msq, asfloat(t1)), 1.0f);
    fg_st_h1(kc, idx, cons ? 0.0f : 1.0f);
    const float key = cons ? kap + fg_opq(fg_mul(kap, cam.z) - kap) : kap;
    const uint hi = (asuint(key) & 0xFFFFFFFEu) | (cons ? 1u : 0u);
    const uint lo = (fg_f2h(mx) << 16) | fg_f2h(my);
    const float thr2 = max(fg_adiv(msq, asfloat(t2)), 1.0f);
    const int sx = fg_sgn(mx), sy = fg_sgn(my);
    const int bx = x + (int)mx, by = y + (int)my;
    [unroll] for (uint e = 0u; e < 4u; ++e)
    {
        const int ex = (int)(e >> 1), ey = (int)(e & 1u);
        const int tx = bx + ex * sx, ty = by + ey * sy;
        bool ok = tx >= 0 && tx < (int)w1 && ty >= 0 && ty < (int)h1;
        if (e != 0u)
        {
            const float4 nb = fg_read_int(mv, 5u, (int)w1, (int)h1, x + ex * sx, y + ey * sy);
            ok = ok && fg_sq(fg_mul(nb.x, (float)w1) - mx, fg_mul(nb.y, (float)h1) - my) < thr2;
        }
        if (ok) fg_atomic_min64(sbuf, (uint)(ty * (int)w1 + tx), hi, lo);
    }
}
#endif

#if FG_K == 2
WG_ARGS_BEGIN
    uint sbuf, zp, kp, mv, fo, eo, flo, vfo, w1, h1;
    uint m0, m1, m2, m3, m4, m5, m6, m7, m8, m9, m10, m11, m12, m13, m14, m15;
    uint s_k, inverted, stepx, stepy, kmax, tau;
WG_ARGS_END

WG_KERNEL(fg_lite_p2c_unpack)
{
    FG_SETUP();
    const uint idx = wg_block_id() * 256u + wg_lane_id();
    if (idx >= w1 * h1) return;
    const uint x = idx % w1, y = idx / w1;
    const float u = fg_centre(x, w1), v = fg_centre(y, h1);
    const uint2 e64 = wg_ld64(sbuf, idx * 8u);
    const bool empty = e64.x == 0x7FFFFFFFu && e64.y == 0x7FFFFFFFu;
    const float sxv = empty ? 0.0f : f16tof32(e64.x >> 16), syv = empty ? 0.0f : f16tof32(e64.x & 0xffffu);
    const float keyE = asfloat(e64.y & 0xFFFFFFFEu);
    const bool cbit = (e64.y & 1u) != 0u;
    const float3 zo = fg_unpack_depth(wg_ld32(zp, idx * 4u), asfloat(stepx), asfloat(stepy));
    const float kpv = fg_texel(kp, 1u, idx).x;
    const float3 cam = fg_camera(u + zo.y, v + zo.z, zo.x, fg_f(m0), fg_f(m1), fg_f(m3), fg_f(m4), fg_f(m5), fg_f(m7),
                                 fg_f(m8), fg_f(m9), fg_f(m11), fg_f(m12), fg_f(m13), fg_f(m15), w1, h1);
    float kap = fg_depth_key(zo.x, asfloat(s_k), inverted);
    kap = kpv == 0.0f ? fg_mul(kap, cam.z) : kap;
    const float km = asfloat(kmax);
    const float kE = keyE <= 0.0f ? 0.0f : (km <= keyE ? km : keyE);
    const float kP = kap <= 0.0f ? 0.0f : (kap >= km ? km : kap);
    const bool mismatch = empty || abs(kE - kP) > asfloat(tau);
    const float gu = u + fg_adiv(cam.x, (float)w1), gv = v + fg_adiv(cam.y, (float)h1);
    const float4 g = fg_read_nearest(mv, 5u, (int)w1, (int)h1, gu, gv);
    const float rx = mad(g.x, (float)w1, cam.x), ry = mad(g.y, (float)h1, cam.y);
    const bool dynamic = !(kpv == 0.0f);
    const bool use_cam = cbit || mismatch || fg_sq(rx, ry) < 0.25f;
    const float vx = dynamic ? -sxv : (use_cam ? cam.x : -sxv);
    const float vy = dynamic ? -syv : (use_cam ? cam.y : -syv);
    fg_st_h2(fo, idx, vx, vy);
    fg_st_h1(eo, idx, empty ? 1.0f : 0.0f);
    wg_st32(flo, idx * 4u, (dynamic ? 1u : 0u) | ((dynamic && mismatch) ? 2u : 0u) | (empty ? 4u : 0u));
    wg_st64(vfo, idx * 8u, uint2(asuint(vx), asuint(vy)));
}
#endif

#if FG_K == 3
WG_ARGS_BEGIN
    uint fl, gout, w1, h1;
WG_ARGS_END

#if FG_IP_TILE
// Tile row r (0 .. 11) is pixel row y0 - 2 + r, bit c (0 .. 35) is pixel column x0 - 2 + c: the flag-bit-1 image, 0 outside.
groupshared uint64_t fg_ip_row[12];
groupshared uint64_t fg_ip_dil[12];   // bit c (0 .. 33): OR of the 3 x 3 flags around centre (x0 - 1 + c, y0 - 1 + r)
groupshared uint64_t fg_ip_dv[10];    // the same ORed over three rows

bool fg_ip_flag(int x, int y)
{
    return x >= 0 && x < (int)w1 && y >= 0 && y < (int)h1 &&
           (wg_ld32(fl, ((uint)y * w1 + (uint)x) * 4u) & 2u) != 0u;
}

// the 36-bit mask of pixel row y of the tile (columns x0 - 2 .. x0 + 33); the wave's lanes are the 32 tile columns, lanes 0 .. 3
// also the four halo columns
uint64_t fg_ip_row_mask(int x0, int y, uint lane)
{
    const int x = x0 + (int)lane;
    const bool own = fg_ip_flag(x, y);
    const int xh = lane < 2u ? x0 - 2 + (int)lane : x0 + 32 + (int)lane - 2;
    const bool halo = lane < 4u && fg_ip_flag(xh, y);
    const uint m = WaveActiveBallot(own).x, hb = WaveActiveBallot(halo).x;
    return (uint64_t)(hb & 3u) | ((uint64_t)m << 2) | ((uint64_t)((hb >> 2) & 3u) << 34);
}

WG_KERNEL(fg_lite_p2c_inpaint)
{
    FG_SETUP();
    const uint l = wg_lane_id(), lane = l & 31u, wave = l >> 5u;
    const uint tiles_x = (w1 + 31u) >> 5u, tile = wg_block_id();
    const int x0 = (int)((tile % tiles_x) * 32u), y0 = (int)((tile / tiles_x) * 8u);
    const int x = x0 + (int)lane, y = y0 + (int)wave;
    const bool live = x < (int)w1 && y < (int)h1;
    const uint fw = wg_ld32(fl, select(live, ((uint)y * w1 + (uint)x) * 4u, 0u));
    const uint64_t main_row = fg_ip_row_mask(x0, y, lane);
    if (lane == 0u) fg_ip_row[2u + wave] = main_row;
    if (wave < 4u) {   // wave-uniform: the rows y0 - 2, y0 - 1, y0 + 8, y0 + 9
        const int ey = wave < 2u ? y0 - 2 + (int)wave : y0 + 8 + (int)wave - 2;
        const uint64_t extra = fg_ip_row_mask(x0, ey, lane);
        if (lane == 0u) fg_ip_row[wave < 2u ? wave : wave + 8u] = extra;
    }
    wg_barrier_lds();
    if (l < 12u) {
        const uint64_t b = fg_ip_row[l];
        fg_ip_dil[l] = b | (b >> 1) | (b >> 2);
    }
    wg_barrier_lds();
    if (l < 10u) fg_ip_dv[l] = fg_ip_dil[l] | fg_ip_dil[l + 1u] | fg_ip_dil[l + 2u];
    wg_barrier_lds();
    if (!live) return;
    const uint64_t e = fg_ip_dv[wave] & fg_ip_dv[wave + 1u] & fg_ip_dv[wave + 2u];
    const bool inp = ((e >> lane) & 7ull) == 7ull;
    const bool dyn = (fw & 1u) != 0u;
    wg_st32(gout, ((uint)y * w1 + (uint)x) * 4u, ((dyn && !inp) ? 2u : 0u) | (inp ? 1u : 0u));
}
#else
// Each row holds the 256 centers and the two horizontal halo pixels on either side.
// A row is addressed in the flat grid, then trimmed at the center pixel's image edges.
groupshared uint fg_inpaint_flags[5][9];

WG_KERNEL(fg_lite_p2c_inpaint)
{
    FG_SETUP();
    const uint l = wg_lane_id();
    const uint idx = wg_block_id() * 256u + l;
    const uint size = w1 * h1;
    const uint lane = wg_wave_lane();
    [unroll] for (int j = 0; j < 5; ++j)
    {
        const int p = (int)idx + (j - 2) * (int)w1 - 2;
        const bool flag = (uint)p < size && (wg_ld32(fl, (uint)p * 4u) & 2u) != 0u;
        const uint row = WaveActiveBallot(flag).x;
        if (lane == 0u) fg_inpaint_flags[j][l >> 5] = row;
        if (l < 32u)
        {
            const int q = p + 256;
            const bool halo = l < 4u && (uint)q < size && (wg_ld32(fl, (uint)q * 4u) & 2u) != 0u;
            const uint tail = WaveActiveBallot(halo).x;
            if (lane == 0u) fg_inpaint_flags[j][8] = tail;
        }
    }
    wg_barrier_lds();
    if (idx >= size) return;

    const int x = (int)(idx % w1);
    const uint xmask = ((1u << (uint)min((int)w1 - x + 2, 5)) - 1u) & (31u << (uint)max(2 - x, 0));
    uint bits = 0u;
    [unroll] for (int j = 0; j < 5; ++j)
    {
        const uint a = fg_inpaint_flags[j][l >> 5];
        const uint b = fg_inpaint_flags[j][(l >> 5) + 1u];
        const uint row = (a >> lane) | (lane > 27u ? (b << (32u - lane)) : 0u);
        bits |= (row & xmask) << (uint)(j * 5);
    }
    bool inp = true;
    [unroll] for (int j = 0; j < 3; ++j)
        [unroll] for (int i = 0; i < 3; ++i)
        {
            const uint mask = 0x1ce7u << (uint)(j * 5 + i); // three 3-bit rows on a 5-bit stride
            const bool w = (bits & mask) != 0u;
#if FG_MUT
            inp = inp && (w || (j == 2 && i == 2));   // mutant: the last window is not required
#else
            inp = inp && w;
#endif
        }
    const bool dyn = (wg_ld32(fl, idx * 4u) & 1u) != 0u;
    wg_st32(gout, idx * 4u, ((dyn && !inp) ? 2u : 0u) | (inp ? 1u : 0u));
}
#endif
#endif

#if FG_K == 4
WG_ARGS_BEGIN
    uint mode, srca, srcb, sums_out, elem_out, ws, hs, wl, hl, nbits;
#if FG_PULL_TILE
    uint elem2, elem3, elem4, levels;
#endif
WG_ARGS_END

float3 fg_child(int cx, int cy)
{
    if (cx >= (int)ws || cy >= (int)hs) return float3(0.0f, 0.0f, 0.0f);
    const uint i = (uint)cy * ws + (uint)cx;
    if (mode == 0u)
    {
        const float w = (float)((wg_ld32(srca, i * 4u) >> 1) & 1u);
        const float4 vf = fg_texel(srcb, 5u, i);
        return float3(fg_mul(w, vf.x), fg_mul(w, vf.y), w);
    }
    if (mode == 1u) return asfloat(uint3(wg_ld32(srca, i * 12u), wg_ld32(srca, i * 12u + 4u), wg_ld32(srca, i * 12u + 8u)));
    const float4 e = fg_texel(srca, 3u, i);
    return float3(fg_mul(e.x, e.z), fg_mul(e.y, e.z), e.z);
}

float3 fg_pull_sum(int X, int Y)
{
    const float3 c00 = fg_child(2 * X, 2 * Y), c10 = fg_child(2 * X + 1, 2 * Y);
    const float3 c01 = fg_child(2 * X, 2 * Y + 1), c11 = fg_child(2 * X + 1, 2 * Y + 1);
    float3 s;
    s.x = fg_opq(fg_opq(c00.x + c10.x) + fg_opq(c01.x + c11.x));
    s.y = fg_opq(fg_opq(c00.y + c10.y) + fg_opq(c01.y + c11.y));
    s.z = fg_opq(fg_opq(c00.z + c10.z) + fg_opq(c01.z + c11.z));
    return s;
}

void fg_pull_element(uint dst, uint idx, float3 s, float area)
{
    const float den = max(s.z, FG_WEIGHT_FLOOR);
    fg_st_h3(dst, idx, float3(fg_adiv(s.x, den), fg_adiv(s.y, den), fg_adiv(s.z, area)));
}

#if FG_PULL_TILE
groupshared float3 fg_pull_waves[8];

uint fg_pull_morton_x(uint l)
{
    return (l & 1u) | ((l >> 1u) & 2u) | ((l >> 2u) & 4u) | ((l >> 3u) & 8u);
}

float3 fg_pull_ordered(float3 left, float3 right)
{
    const float3 s = left + right;
    // A WaveReadLaneAt operand may be commuted by the compiler. When both
    // operands are NaNs, retain the left operand as the standalone pull does.
    return select(isnan(left), left, float3(fg_opq(s.x), fg_opq(s.y), fg_opq(s.z)));
}

float3 fg_pull_quad(float3 s, uint lane, uint stride)
{
    s = fg_pull_ordered(s, WaveReadLaneAt(s, lane ^ stride));
    return fg_pull_ordered(s, WaveReadLaneAt(s, lane ^ (2u * stride)));
}

void fg_pull_level(uint dst, uint x, uint y, uint shift, float3 s)
{
    const uint mask = (1u << shift) - 1u;
    const uint w = (wl + mask) >> shift, h = (hl + mask) >> shift;
    x >>= shift;
    y >>= shift;
    if (x < w && y < h) fg_pull_element(dst, y * w + x, s, (float)(4u << (2u * shift)));
}
#endif

WG_KERNEL(fg_lite_p2c_pull)
{
    FG_SETUP();
#if FG_PULL_TILE
    const uint l = wg_lane_id(), lane = wg_wave_lane();
    const uint tiles_x = (wl + 15u) >> 4u, tile = wg_block_id();
    const uint base_x = (tile % tiles_x) * 16u, base_y = (tile / tiles_x) * 16u;
    const uint x = base_x + fg_pull_morton_x(l), y = base_y + fg_pull_morton_x(l >> 1u);
    float3 s = 0.0f;
    if (x < wl && y < hl) {
        s = fg_pull_sum((int)x, (int)y);
        fg_pull_element(elem_out, y * wl + x, s, asfloat(nbits));
    }
    if (levels == 1u) return;
    s = fg_pull_quad(s, lane, 1u);
    if ((l & 3u) == 0u) fg_pull_level(elem2, x, y, 1u, s);
    if (levels == 2u) return;
    s = fg_pull_quad(s, lane, 4u);
    if ((l & 15u) == 0u) fg_pull_level(elem3, x, y, 2u, s);
    if (levels == 3u) return;
    s = fg_pull_ordered(s, WaveReadLaneAt(s, lane ^ 16u));
    if (lane == 0u) fg_pull_waves[l >> 5u] = s;
    wg_barrier_lds();
    if (l < 4u) {
        const uint wave = 2u * l;
        s = fg_pull_ordered(fg_pull_waves[wave], fg_pull_waves[wave + 1u]);
        fg_pull_level(elem4, base_x + (l & 1u) * 8u, base_y + (l >> 1u) * 8u, 3u, s);
    }
#else
    const uint idx = wg_block_id() * 256u + wg_lane_id();
    if (idx >= wl * hl) return;
    const float3 s = fg_pull_sum((int)(idx % wl), (int)(idx / wl));
    if (sums_out != 0u)
    {
        wg_st32(sums_out, idx * 12u, asuint(s.x));
        wg_st32(sums_out, idx * 12u + 4u, asuint(s.y));
        wg_st32(sums_out, idx * 12u + 8u, asuint(s.z));
    }
    fg_pull_element(elem_out, idx, s, asfloat(nbits));
#endif
}
#endif

#if FG_K == 5 || FG_K == 6 || FG_K == 7
// (v_up, w_up) of fine element (qx, qy) from the filled parent level [ph][pw] (float16 x 3)
float3 fg_upsample(uint parent, int pw, int ph, int qx, int qy)
{
    const int px = qx >> 1, py = qy >> 1;
    const int nx = clamp(px + ((qx & 1) != 0 ? 1 : -1), 0, pw - 1);
    const int ny = clamp(py + ((qy & 1) != 0 ? 1 : -1), 0, ph - 1);
    const float4 t[4] = { fg_texel(parent, 3u, (uint)(py * pw + px)), fg_texel(parent, 3u, (uint)(py * pw + nx)),
                          fg_texel(parent, 3u, (uint)(ny * pw + px)), fg_texel(parent, 3u, (uint)(ny * pw + nx)) };
    const float k[4] = { 0.5625f, 0.1875f, 0.1875f, 0.0625f };
    const float zero = asfloat(fg_z_);
    float ws = zero, v0 = zero, v1 = zero;
    [unroll] for (uint i = 0u; i < 4u; ++i)
    {
        ws = mad(t[i].z, k[i], ws);
        v0 = mad(fg_mul(t[i].x, t[i].z), k[i], v0);
        v1 = mad(fg_mul(t[i].y, t[i].z), k[i], v1);
    }
    const float den = fg_med3(ws, FG_WEIGHT_FLOOR, 1.0f);
    return float3(fg_adiv(v0, den), fg_adiv(v1, den), ws);
}
#endif

#if FG_K == 5 || FG_K == 7
void fg_push_pixel(uint parent, int pw, int ph, uint level, uint lw, uint lh, uint idx)
{
    if (idx >= lw * lh) return;
    const float4 own = fg_texel(level, 3u, idx);
    if (!(own.z <= 0.0f)) return;
    fg_st_h3(level, idx, fg_upsample(parent, pw, ph, (int)(idx % lw), (int)(idx / lw)));
}
#endif

#if FG_K == 5
WG_ARGS_BEGIN
    uint parent, pw, ph, level, lw, lh;
WG_ARGS_END

WG_KERNEL(fg_lite_p2c_push)
{
    FG_SETUP();
    const uint idx = wg_block_id() * 256u + wg_lane_id();
    fg_push_pixel(parent, (int)pw, (int)ph, level, lw, lh, idx);
}
#endif

#if FG_K == 7
WG_ARGS_BEGIN
    uint count, p0, p1, p2, p3, p4, p5, p6, p7;
    uint w0, w1, w2, w3, w4, w5, w6, w7;
    uint h0, h1, h2, h3, h4, h5, h6, h7;
WG_ARGS_END

// Every level fits in this one workgroup. The device barrier makes the stored FP16
// parent visible to the next level, with the same rounding as separate dispatches.
WG_KERNEL(fg_lite_p2c_push_tail)
{
    FG_SETUP();
    const uint p[8] = {p0, p1, p2, p3, p4, p5, p6, p7};
    const uint w[8] = {w0, w1, w2, w3, w4, w5, w6, w7};
    const uint h[8] = {h0, h1, h2, h3, h4, h5, h6, h7};
    [loop] for (uint i = 1u; i <= count; ++i) {
        fg_push_pixel(p[i - 1u], (int)w[i - 1u], (int)h[i - 1u], p[i], w[i], h[i], wg_lane_id());
        if (i < count) wg_barrier();
    }
}
#endif

#if FG_K == 6
WG_ARGS_BEGIN
    uint parent, pw, ph, gin, fio, w1, h1;
#if FG_MASK
    uint hc;
#endif
WG_ARGS_END

WG_KERNEL(fg_lite_p2c_fine)
{
    FG_SETUP();
    const uint idx = wg_block_id() * 256u + wg_lane_id();
    if (idx >= w1 * h1) return;
#if FG_MASK
    // F at this pixel as the next pass reads it (float16 x 2): v_up where G bit 0 is set, else the unpack pass's value
    uint fw;
    if ((wg_ld32(gin, idx * 4u) & 1u) != 0u) {
        const float3 up = fg_upsample(parent, (int)pw, (int)ph, (int)(idx % w1), (int)(idx / w1));
        fw = fg_pack_h2(up.x, up.y);
        wg_st32(fio, idx * 4u, fw);
    } else {
        fw = wg_ld32(fio, idx * 4u);
    }
    // fg_lite_hole_mask on the same grid (wd = w1, hd = h1): Hc := 0 at the four targets of this pixel's vector
    const int x = (int)(idx % w1), y = (int)(idx / w1);
    const float mx = fg_mul(fg_adiv(f16tof32(fw), (float)w1), (float)w1);
    const float my = fg_mul(fg_adiv(f16tof32(fw >> 16), (float)h1), (float)h1);
    const int bx = x + (int)mx, by = y + (int)my, sx = fg_sgn(mx), sy = fg_sgn(my);
    [unroll] for (uint e = 0u; e < 4u; ++e)
    {
        const int tx = bx + (int)(e >> 1) * sx, ty = by + (int)(e & 1u) * sy;
        if (tx >= 0 && tx < (int)w1 && ty >= 0 && ty < (int)h1) wg_st16(hc, (uint)(ty * (int)w1 + tx) * 2u, 0u);
    }
#else
    if ((wg_ld32(gin, idx * 4u) & 1u) == 0u) return;
    const float3 up = fg_upsample(parent, (int)pw, (int)ph, (int)(idx % w1), (int)(idx / w1));
    fg_st_h2(fio, idx, up.x, up.y);
#endif
}
#endif
