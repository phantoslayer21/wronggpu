// SPDX-License-Identifier: MIT
// Model L's own post on the direct-resource route (runtime option l_post=own, native D3D12): the D3D12-only colour step
// before l_post_recon_p12, which reads its colour from a buffer (no direct form). With it the direct frame gives the
// staged route's bytes: the staged route converts the game's textures into dense buffers (the Converter's input pass,
// color_divide when pi != 1) and runs the same kernels. (The output: l_post_filter_direct, kernels/post_filter.hlsl,
// writes the game's texture as the staged route's scale_rgb and Converter output pass do.)
// l_direct_color: dense texel i < npix of the game's colour texture -> color_net (RGBA f32, 16 bytes):
//   native_direct_normalized_color(i), the staged colour buffer's value (alpha +0; divided by pi as color_divide).
// Grid: cdiv(npix, 256) workgroups of 256 lanes.
//
// @wg-kernel l_direct_color co=- block=256 wave=32 args=p,u rs=direct defines=WG_DIRECT_TEXTURES=1
#include "wg_platform.hlsli"
#include "native_direct.hlsli"

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
