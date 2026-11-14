// SPDX-License-Identifier: MIT
// Model L's own post on the direct-resource route (runtime option l_post=own, native D3D12): the two D3D12-only steps
// around l_post_recon_p12 -> w64/post_filter_c1_s4, which read and write buffers (no direct forms). With them the direct
// frame gives the staged route's bytes: the staged route converts the game's textures into dense buffers (the
// Converter's input pass, color_divide when pi != 1), runs the same kernels, then scale_rgb when pi != 1 and the
// Converter's output pass into the game's texture.
// l_direct_color: dense texel i < npix of the game's colour texture -> color_net (RGBA f32, 16 bytes):
//   native_direct_normalized_color(i), the staged colour buffer's value (alpha +0; divided by pi as color_divide).
// l_direct_store: output pixel i < ow oh of the filter's FP32 RGB (out, 12 bytes a pixel) -> the game's output texture:
//   times pi as scale_rgb (native_direct_scale) when pi != 1, then the typed store with alpha 1 as the Converter's
//   output pass; with nd_output_rne16 (RGBA16F output-copy rows) the stored value is framebuffer_half's (+-65504 clamp,
//   round to nearest even), as post.hlsl's direct forms store it (pd_out16).
// Grids: cdiv(npix, 256) and cdiv(ow oh, 256) workgroups of 256 lanes.
//
// @wg-kernel l_direct_color co=- block=256 wave=32 args=p,u rs=direct defines=WG_DIRECT_TEXTURES=1,LPD_COLOR=1
// @wg-kernel l_direct_store co=- block=256 wave=32 args=p,u,u rs=direct defines=WG_DIRECT_TEXTURES=1,LPD_STORE=1
#include "wg_platform.hlsli"
#include "native_direct.hlsli"

#if defined(LPD_COLOR)
WG_ARGS_BEGIN
    uint color_net;   // pointer
    int npix;
WG_ARGS_END

WG_KERNEL(l_direct_color)
{
    WG_SETUP();
    const uint i = wg_block_id() * 256u + wg_lane_id();
    if ((int)i >= npix) return;
    wg_st128(color_net, i * 16u, asuint(native_direct_normalized_color(i)));
}

#elif defined(LPD_STORE)
WG_ARGS_BEGIN
    uint out_p;       // pointer
    int ow, oh;
WG_ARGS_END

// post.hip framebuffer_half's clamp (compares: a NaN passes), then wg_f2h (round to nearest even, NaN kept)
float lpd_half(float v)
{
    const float c = select(v > 65504.0f, 65504.0f, select(v < -65504.0f, -65504.0f, v));
    return f16tof32(wg_f2h(c));
}

WG_KERNEL(l_direct_store)
{
    WG_SETUP();
    const uint i = wg_block_id() * 256u + wg_lane_id();
    if ((int)i >= ow * oh) return;
    const uint y = i / (uint)ow, x = i - y * (uint)ow;
    float3 v = float3(wg_ldf(out_p, i * 12u), wg_ldf(out_p, i * 12u + 4u), wg_ldf(out_p, i * 12u + 8u));
    if (nd_color_pi != 1.0f)
        v = float3(native_direct_scale(v.x, nd_color_pi), native_direct_scale(v.y, nd_color_pi),
                   native_direct_scale(v.z, nd_color_pi));
    if (nd_output_rne16 != 0u) v = float3(lpd_half(v.x), lpd_half(v.y), lpd_half(v.z));
    native_direct_output[uint2(x, y)] = float4(v, 1.0f);
}
#endif
