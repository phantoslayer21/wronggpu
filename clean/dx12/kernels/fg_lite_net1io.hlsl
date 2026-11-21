// SPDX-License-Identifier: MIT
// Frame generation (issue #96), light path: R7 hole mask, R8 quality image and R9 net1 input packer / output unpacker
// (clean/fg/reference/stages_common.py hole_mask, quality_init, net1_input, net1_output). D3D12 only.
//
// fg_lite_hole_mask: (after Hc := 1) one lane per net1-grid pixel: F at the pixel centre (nearest, scatter grid),
//   divided by the scatter grid size, times the net1 grid size; Hc := 0 at the four targets.
// fg_lite_quality_init: Qc := (sat(((r + g) + b) / 3), 0) of Dc.
// fg_lite_net1_input: one lane per pixel of the 2 x 2-fold grid (2 W0 x 2 H0): the ten values of its image pixel (zero
//   outside the image) at cell channels 4 v + 2 dy + dx (channels 40..47 zero, written by the dx = dy = 0 lane); the
//   cell layout is element (channel * cs + (Y * W0 + X) * ps) of float16 (HWC: cs 1, ps 48; CHW: cs H0 W0, ps 1);
//   image pixels also write Qc = (v6, v2) and Om = sat((E Hp) Hc).
// fg_lite_net1_output: one lane per net1 image pixel: the four values of its (clamped) cell, the occlusion push-down, the
//   sigmoid map m, the UI forcing (ui_forcing != 0) and the logit pairs NA, NB.
//
// @wg-kernel fg_lite_hole_mask co=- block=256 wave=32 args=p,u,u,p,u,u defines=FG_K=1
// @wg-kernel fg_lite_quality_init co=- block=256 wave=32 args=p,p,u defines=FG_K=2
// @wg-kernel fg_lite_net1_input co=- block=256 wave=32 args=p,p,p,p,p,p,p,p,p,p,p,p,p,u,u,u,u,u,u,u,u,u,u defines=FG_K=3
// @wg-kernel fg_lite_net1_output co=- block=256 wave=32 args=p,p,p,p,p,p,p,u,u,u,u,u,u,u,u,u defines=FG_K=4
#include "wg_platform.hlsli"
#include "fg_lite_common.hlsli"

#if FG_K == 1
WG_ARGS_BEGIN
    uint fin, w1, h1, hc, wd, hd;
WG_ARGS_END

WG_KERNEL(fg_lite_hole_mask)
{
    FG_SETUP();
    const uint idx = wg_block_id() * 256u + wg_lane_id();
    if (idx >= wd * hd) return;
    const int x = (int)(idx % wd), y = (int)(idx / wd);
    const float u = fg_centre((uint)x, wd), v = fg_centre((uint)y, hd);
    const float4 m = fg_read_nearest(fin, 2u, (int)w1, (int)h1, u, v);
    const float mx = fg_mul(fg_adiv(m.x, (float)w1), (float)wd), my = fg_mul(fg_adiv(m.y, (float)h1), (float)hd);
    const int bx = x + (int)mx, by = y + (int)my, sx = fg_sgn(mx), sy = fg_sgn(my);
    [unroll] for (uint e = 0u; e < 4u; ++e)
    {
        const int tx = bx + (int)(e >> 1) * sx, ty = by + (int)(e & 1u) * sy;
        if (tx >= 0 && tx < (int)wd && ty >= 0 && ty < (int)hd) wg_st16(hc, (uint)(ty * (int)wd + tx) * 2u, 0u);
    }
}
#endif

#if FG_K == 2
WG_ARGS_BEGIN
    uint dc, qc, n;
WG_ARGS_END

WG_KERNEL(fg_lite_quality_init)
{
    FG_SETUP();
    const uint idx = wg_block_id() * 256u + wg_lane_id();
    if (idx >= n) return;
    const float4 c = fg_texel(dc, 4u, idx);
    fg_st_h2(qc, idx, fg_sat(fg_adiv(fg_opq(c.x + c.y) + c.z, 3.0f)), 0.0f);
}
#endif

#if FG_K == 3
WG_ARGS_BEGIN
    uint ld, dc, dp, fin, mv, eimg, hc, hp, qp, mimg, cells, qc, om;
    uint wd, hd, w1, h1, W0, H0, pad_l, pad_t, cs, ps;
WG_ARGS_END

float fg_warp(float3 a, float3 b)
{
    const float3 d = abs(a - b);
    return fg_sat(fg_adiv(d.z + fg_opq(d.x + d.y), 3.0f));
}

void fg_cell_store(uint X, uint Y, uint ch, float val)
{
    wg_st16(cells, (ch * cs + (Y * W0 + X) * ps) * 2u, wg_f2h(val));
}

WG_KERNEL(fg_lite_net1_input)
{
    FG_SETUP();
    const uint idx = wg_block_id() * 256u + wg_lane_id();
    const uint gw = 2u * W0, gh = 2u * H0;
    if (idx >= gw * gh) return;
    const uint gx = idx % gw, gy = idx / gw;
    const int x = (int)gx - 2 * (int)pad_l, y = (int)gy - 2 * (int)pad_t;
    const uint X = gx >> 1, Y = gy >> 1, q = 2u * (gy & 1u) + (gx & 1u);
    float vals[10] = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };
    if (x >= 0 && x < (int)wd && y >= 0 && y < (int)hd)
    {
        const uint pi = (uint)y * wd + (uint)x;
        const float u = fg_centre((uint)x, wd), v = fg_centre((uint)y, hd);
        const float4 dcv = fg_texel(dc, 4u, pi), dpv = fg_texel(dp, 4u, pi);
        const float4 f = fg_read_nearest(fin, 2u, (int)w1, (int)h1, u, v);
        const float su = u + fg_adiv(f.x, (float)w1), sv = v + fg_adiv(f.y, (float)h1);
        const float v2 = fg_warp(fg_read_linear(dc, 4u, (int)wd, (int)hd, su, sv).xyz, dpv.xyz);
        const float4 m = fg_read_nearest(mv, 5u, (int)w1, (int)h1, u, v);
        const float v6 = fg_warp(fg_read_linear(dp, 4u, (int)wd, (int)hd, u + m.x, v + m.y).xyz, dcv.xyz);
        const float e = fg_read_nearest(eimg, 1u, (int)w1, (int)h1, u, v).x;
        const float hcv = fg_read_nearest(hc, 1u, (int)wd, (int)hd, u, v).x;
        const float hpv = fg_read_nearest(hp, 1u, (int)wd, (int)hd, u, v).x;
        const float mm = fg_texel(mimg, 1u, pi).x;
        vals[0] = fg_texel(ld, 1u, pi).x;
        vals[1] = fg_luma(dpv.xyz);
        vals[2] = v2;
        vals[3] = fg_texel(qp, 2u, pi).x;
        vals[4] = e;
        vals[5] = fg_luma(dcv.xyz);
        vals[6] = v6;
        vals[7] = hcv;
        vals[8] = mm;
        vals[9] = 1.0f - mm;
        fg_st_h2(qc, pi, v6, v2);
        fg_st_h1(om, pi, fg_sat(fg_mul(fg_mul(e, hpv), hcv)));
    }
    [unroll] for (uint k = 0u; k < 10u; ++k) fg_cell_store(X, Y, 4u * k + q, vals[k]);
    if (q == 0u)
        [unroll] for (uint c = 40u; c < 48u; ++c) fg_cell_store(X, Y, c, 0.0f);
}
#endif

#if FG_K == 4
WG_ARGS_BEGIN
    uint n1, om, um, ump, na, nb, mimg;
    uint wd, hd, pad_l, pad_t, W0, H0, cs, ps, ui_forcing;
WG_ARGS_END

WG_KERNEL(fg_lite_net1_output)
{
    FG_SETUP();
    const uint idx = wg_block_id() * 256u + wg_lane_id();
    if (idx >= wd * hd) return;
    const uint x = idx % wd, y = idx / wd;
    const uint ncx = max(wd / 2u, 1u), ncy = max(hd / 2u, 1u);
    const uint cx = pad_l + min(x >> 1, ncx - 1u), cy = pad_t + min(y >> 1, ncy - 1u);
    const uint q = 2u * (y & 1u) + (x & 1u), cell = cy * W0 + cx;
    float o[4];
    [unroll] for (uint i = 0u; i < 4u; ++i) o[i] = wg_h2f(wg_ld16(n1, ((4u * i + q) * cs + cell * ps) * 2u));
    const float omv = fg_texel(om, 1u, idx).x;
    o[0] = mad(FG_PUSH - o[0], omv, o[0]);
    o[2] = mad(FG_PUSH - o[2], omv, o[2]);
    const float mm = fg_sigmoid(o[2], o[3]);
    if (ui_forcing != 0u)
    {
        if (fg_texel(ump, 1u, idx).x > 0.99f) { o[0] = o[0] - 1000.0f; o[1] = o[1] + 1000.0f; }
        if (fg_texel(um, 1u, idx).x > 0.99f) { o[2] = o[2] - 1000.0f; o[3] = o[3] + 1000.0f; }
    }
    fg_st_h2(na, idx, o[0], o[1]);
    fg_st_h2(nb, idx, o[2], o[3]);
    fg_st_h1(mimg, idx, mm);
}
#endif
