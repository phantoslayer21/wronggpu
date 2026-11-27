// SPDX-License-Identifier: GPL-3.0-only
// Frame generation (issue #96): the blend weight of one scatter-grid texel, shared by fg_lite_weights (which stores it) and
// the blends that build their W tile in LDS (fg_lite_blend FG_WEIGHTS, fg_full_blend FGB_WEIGHTS): one definition, so
// both produce the same packed float16 x 2 word (clean/fg/reference/stages_common.py blend_weights).
// The body is a copy of fg_lite_weights' (fg_lite_g.hlsl FG_K == 2), which is left as it was so that its DXIL stays
// byte-identical; clean/fg/tests/test_fg_turbo_v2.py compares the two texts line by line.
// W(x, y) = (sigma(NA at the MB vector - NA co-located), sigma(NB at the MA vector - NB co-located)), each rounded to
// float16 (the driver conversion, fg_pack_h2). Include after fg_lite_common.hlsli.
#pragma once

// ma / mb: scatter-grid 64-bit words (w1 x h1); na / nb: net1 logit pairs (wd x hd, float16 x 2); (x, y): the texel.
uint fg_weight_texel(uint ma, uint mb, uint na, uint nb, uint x, uint y, uint w1, uint h1, uint wd, uint hd)
{
    const float u = fg_centre(x, w1), v = fg_centre(y, h1);
    const uint mi = fg_index_rint(v, h1) * w1 + fg_index_rint(u, w1);
    const float3 a = fg_vec64(ma, mi), b = fg_vec64(mb, mi);
    const uint ji = fg_index_rint(v, hd) * wd + fg_index_rint(u, wd);
    const float A1 = fg_read_linear(na, 2u, (int)wd, (int)hd, u + b.x, v + b.y).x;
    const float A0 = fg_texel(na, 2u, ji).y;
    const float B1 = fg_read_linear(nb, 2u, (int)wd, (int)hd, u + a.x, v + a.y).x;
    const float B0 = fg_texel(nb, 2u, ji).y;
    return fg_pack_h2(fg_sigmoid(A1, A0), fg_sigmoid(B1, B0));
}
