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
//
// @wg-kernel fg_lite_p2c_scatter co=- block=256 wave=32 args=p,p,p,p,u,u,u,u,u,u,u,u,u,u,u,u,u,u,u,u,u,u,u,u,u,u,u,u defines=FG_K=1
// @wg-kernel fg_lite_p2c_unpack co=- block=256 wave=32 args=p,p,p,p,p,p,p,p,u,u,u,u,u,u,u,u,u,u,u,u,u,u,u,u,u,u,u,u,u,u,u,u defines=FG_K=2
// @wg-kernel fg_lite_p2c_inpaint co=- block=256 wave=32 args=p,p,u,u defines=FG_K=3
// @wg-kernel fg_lite_p2c_inpaint_mut co=- entry=fg_lite_p2c_inpaint block=256 wave=32 args=p,p,u,u defines=FG_K=3,FG_MUT=1
// @wg-kernel fg_lite_p2c_pull co=- block=256 wave=32 args=u,p,p,p,p,u,u,u,u,u defines=FG_K=4
// @wg-kernel fg_lite_p2c_push co=- block=256 wave=32 args=p,u,u,p,u,u defines=FG_K=5
// @wg-kernel fg_lite_p2c_fine co=- block=256 wave=32 args=p,u,u,p,p,u,u defines=FG_K=6
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
    const uint lo = (wg_f2h(mx) << 16) | wg_f2h(my);
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

WG_KERNEL(fg_lite_p2c_inpaint)
{
    FG_SETUP();
    const uint idx = wg_block_id() * 256u + wg_lane_id();
    if (idx >= w1 * h1) return;
    const int x = (int)(idx % w1), y = (int)(idx / w1);
    uint bits = 0u;          // bit (j + 2) * 5 + (i + 2): flag bit 1 of pixel (x + i, y + j), 0 outside the grid
    [unroll] for (int j = -2; j <= 2; ++j)
        [unroll] for (int i = -2; i <= 2; ++i)
        {
            const int px = x + i, py = y + j;
            const bool in_ = px >= 0 && px < (int)w1 && py >= 0 && py < (int)h1;
            const uint b = in_ ? (wg_ld32(fl, (uint)(py * (int)w1 + px) * 4u) >> 1) & 1u : 0u;
            bits |= b << (uint)((j + 2) * 5 + (i + 2));
        }
    bool inp = true;
    [unroll] for (int cj = -1; cj <= 1; ++cj)
        [unroll] for (int ci = -1; ci <= 1; ++ci)
        {
            bool w = false;
            [unroll] for (int j2 = -1; j2 <= 1; ++j2)
                [unroll] for (int i2 = -1; i2 <= 1; ++i2)
                    w = w || ((bits >> (uint)((cj + j2 + 2) * 5 + (ci + i2 + 2))) & 1u) != 0u;
#if FG_MUT
            inp = inp && (w || (cj == 1 && ci == 1));   // mutant: the last window is not required
#else
            inp = inp && w;
#endif
        }
    const bool dyn = (wg_ld32(fl, idx * 4u) & 1u) != 0u;
    wg_st32(gout, idx * 4u, ((dyn && !inp) ? 2u : 0u) | (inp ? 1u : 0u));
}
#endif

#if FG_K == 4
WG_ARGS_BEGIN
    uint mode, srca, srcb, sums_out, elem_out, ws, hs, wl, hl, nbits;
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

WG_KERNEL(fg_lite_p2c_pull)
{
    FG_SETUP();
    const uint idx = wg_block_id() * 256u + wg_lane_id();
    if (idx >= wl * hl) return;
    const int X = (int)(idx % wl), Y = (int)(idx / wl);
    const float3 c00 = fg_child(2 * X, 2 * Y), c10 = fg_child(2 * X + 1, 2 * Y);
    const float3 c01 = fg_child(2 * X, 2 * Y + 1), c11 = fg_child(2 * X + 1, 2 * Y + 1);
    float3 s;
    s.x = fg_opq(fg_opq(c00.x + c10.x) + fg_opq(c01.x + c11.x));
    s.y = fg_opq(fg_opq(c00.y + c10.y) + fg_opq(c01.y + c11.y));
    s.z = fg_opq(fg_opq(c00.z + c10.z) + fg_opq(c01.z + c11.z));
    if (sums_out != 0u)
    {
        wg_st32(sums_out, idx * 12u, asuint(s.x));
        wg_st32(sums_out, idx * 12u + 4u, asuint(s.y));
        wg_st32(sums_out, idx * 12u + 8u, asuint(s.z));
    }
    const float den = max(s.z, FG_WEIGHT_FLOOR);
    fg_st_h3(elem_out, idx, float3(fg_adiv(s.x, den), fg_adiv(s.y, den), fg_adiv(s.z, asfloat(nbits))));
}
#endif

#if FG_K == 5 || FG_K == 6
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

#if FG_K == 5
WG_ARGS_BEGIN
    uint parent, pw, ph, level, lw, lh;
WG_ARGS_END

WG_KERNEL(fg_lite_p2c_push)
{
    FG_SETUP();
    const uint idx = wg_block_id() * 256u + wg_lane_id();
    if (idx >= lw * lh) return;
    const int qx = (int)(idx % lw), qy = (int)(idx / lw);
    const float4 own = fg_texel(level, 3u, idx);
    if (!(own.z <= 0.0f)) return;                 // the level's own value stays
    fg_st_h3(level, idx, fg_upsample(parent, (int)pw, (int)ph, qx, qy));
}
#endif

#if FG_K == 6
WG_ARGS_BEGIN
    uint parent, pw, ph, gin, fio, w1, h1;
WG_ARGS_END

WG_KERNEL(fg_lite_p2c_fine)
{
    FG_SETUP();
    const uint idx = wg_block_id() * 256u + wg_lane_id();
    if (idx >= w1 * h1) return;
    if ((wg_ld32(gin, idx * 4u) & 1u) == 0u) return;
    const float3 up = fg_upsample(parent, (int)pw, (int)ph, (int)(idx % w1), (int)(idx / w1));
    fg_st_h2(fio, idx, up.x, up.y);
}
#endif
