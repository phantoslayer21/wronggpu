// SPDX-License-Identifier: GPL-3.0-only
// Frame generation (issue #96, F7b): the executor's two input / output kernels around the light and Full path kernels
// (clean/fg/runtime/fg_d3d12_executor.h). D3D12 only (no HIP counterpart). Built only when --only names them.
//
// fg_exec_capture: one lane per texel of a game texture (mip 0, read through the shader resource view at t0 in the
//   executor's descriptor table): dst[y * tw + x] := the texel as the arena image kind the stage kernels read
//   (clean/dx12/hlsl/fg_lite_common.hlsli): kind 4 = float16 x 4, kind 1 = float16 (channel 0), kind 5 = float32 x 2,
//   kind 6 = float32 (channel 0). Float16 values are rounded to nearest even (fg_f2h), so a float16 texture is copied
//   bit for bit and an 8 / 10-bit UNORM texture keeps every code (its float16 value converts back exactly).
// fg_exec_pack: one lane per pixel of a w x h float16 x 4 image (row stride sstride texels) into rows of dpitch bytes in
//   a presenter slot's format: fmt 0 = R16G16B16A16_FLOAT (the bits), 1 = R8G8B8A8_UNORM, 2 = B8G8R8A8_UNORM,
//   3 = R10G10B10A2_UNORM; UNORM codes are floor(sat(v) * max + 0.5) with sat(NaN) = 0.
//
// @wg-kernel fg_exec_capture co=- block=256 wave=32 args=p,u,u,u defines=FGX_K=1
// @wg-kernel fg_exec_pack co=- block=256 wave=32 args=p,u,p,u,u,u,u defines=FGX_K=2
// @wg-kernel fg_exec_pack_pairs co=- entry=fg_exec_pack block=256 wave=32 args=p,u,p,u,u,u,u defines=FGX_K=2,FGX_PAIRS=1
#include "wg_platform.hlsli"
#include "fg_fp16.hlsli"
#include "fg_pack.hlsli"

#if FGX_K == 1
WG_ARGS_BEGIN
    uint a0, a1, a2, a3;
WG_ARGS_END

Texture2D<float4> fgx_src : register(t0);

uint2 fgx_capture_tile(uint w, uint log_width)
{
    const uint tile = wg_block_id(), tiles_x = w >> log_width, lane = wg_lane_id();
    const uint ty = tile / tiles_x;
    const uint tile_w = 1u << log_width, tile_h = 256u >> log_width;
    return uint2((tile - ty * tiles_x) * tile_w + (lane & (tile_w - 1u)), ty * tile_h + (lane >> log_width));
}

uint2 fgx_capture_pixel(uint w, uint h, uint idx)
{
    // Both tile shapes cover exactly 256 texels. Partially tiled dimensions use the linear dispatch.
    if (((w | h) & 15u) == 0u) return fgx_capture_tile(w, 4u);
    if ((w & 31u) == 0u && (h & 7u) == 0u) return fgx_capture_tile(w, 5u);
    const uint y = idx / w;
    return uint2(idx - y * w, y);
}

// fg_exec_capture: a0 = dst, a1 = tw, a2 = th, a3 = kind
WG_KERNEL(fg_exec_capture)
{
    FG_FP16_SETUP();
    const uint linear_idx = wg_block_id() * 256u + wg_lane_id();
    const uint tw = a1, th = a2, kind = (uint)wg_uniform((int)a3);
    if (linear_idx >= tw * th) return;
    const uint2 xy = fgx_capture_pixel(tw, th, linear_idx);
    const uint idx = xy.y * tw + xy.x;
    const float4 c = fgx_src.Load(int3(xy, 0));
    if (kind == 4u)
        wg_st64(a0, idx * 8u, uint2(fg_pack_h2(c.x, c.y), fg_pack_h2(c.z, c.w)));
    else if (kind == 1u)
        wg_st16(a0, idx * 2u, fg_f2h(c.x));
    else if (kind == 5u)
        wg_st64(a0, idx * 8u, uint2(asuint(c.x), asuint(c.y)));
    else
        wg_st32(a0, idx * 4u, asuint(c.x));
}
#endif

#if FGX_K == 2
WG_ARGS_BEGIN
    uint a0, a1, a2, a3, a4, a5, a6;
WG_ARGS_END

// fg_exec_pack: a0 = src, a1 = sstride, a2 = dst, a3 = dpitch, a4 = w, a5 = h, a6 = fmt
WG_KERNEL(fg_exec_pack)
{
    WG_SETUP();
    const uint idx = wg_block_id() * 256u + wg_lane_id();
#if FGX_PAIRS
    const uint pairs = (a4 + 1u) >> 1u;
    if (idx >= pairs * a5) return;
    const uint x = (idx % pairs) * 2u, y = idx / pairs;
    const uint src = (y * a1 + x) * 8u, dst = y * a3 + x * 4u;
    if (x + 1u < a4) {
        const uint4 v = wg_ld128(a0, src);
        wg_st64(a2, dst, uint2(fgx_pack_pixel(v.xy, a6), fgx_pack_pixel(v.zw, a6)));
    } else {
        wg_st32(a2, dst, fgx_pack_pixel(wg_ld64(a0, src), a6));
    }
#else
    const uint w = a4, h = a5, fmt = a6;
    if (idx >= w * h) return;
    const uint x = idx % w, y = idx / w;
    const uint2 v = wg_ld64(a0, (y * a1 + x) * 8u);
    if (fmt == 0u) wg_st64(a2, y * a3 + x * 8u, v);
    else wg_st32(a2, y * a3 + x * 4u, fgx_pack_pixel(v, fmt));
#endif
}
#endif
