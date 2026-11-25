// SPDX-License-Identifier: MIT
// Frame generation (issue #96): the ten cell values of one net1 image pixel, for fg_mconv2_mid_cells (which builds net1_01's
// staged cells in LDS, fg_mconv.hlsl FG_CELLS). It is fg_lite_net1_input's lane work (fg_lite_net1io.hlsl FG_K == 3, left as
// it was so that its DXIL stays byte-identical), line for line: clean/fg/tests/test_fg_turbo_v2.py compares the two
// texts. The including file defines the argument block
// (ld, dc, dp, fin, mv, eimg, hc, hp, qp, mimg, qc, om: arena offsets; wd, hd, w1, h1: the net1 grid and the scatter grid)
// and includes wg_platform.hlsli and fg_lite_common.hlsli first.
#pragma once

float fg_warp(float3 a, float3 b)
{
    const float3 d = abs(a - b);
    return fg_sat(fg_adiv(d.z + fg_opq(d.x + d.y), 3.0f));
}

// The values of image pixel (x, y), all zero when not active or outside the image (x in [0, wd), y in [0, hd)). With store_side the pixel
// also writes its Qc = (v6, v2) and Om = sat((E Hp) Hc) texels.
void fg_net1_pixel(int x, int y, bool active, bool store_side, out float vals[10])
{
    [unroll] for (int i = 0; i < 10; ++i) vals[i] = 0.0f;
    if (active && x >= 0 && x < (int)wd && y >= 0 && y < (int)hd)
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
        if (store_side)
        {
            fg_st_h2(qc, pi, v6, v2);
            fg_st_h1(om, pi, fg_sat(fg_mul(fg_mul(e, hpv), hcv)));
        }
    }
}
