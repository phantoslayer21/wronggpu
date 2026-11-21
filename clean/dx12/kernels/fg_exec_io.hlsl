// SPDX-License-Identifier: MIT
// Frame generation (issue #96, F7b): the executor's two input / output kernels around the light and Full path kernels
// (clean/fg/runtime/fg_d3d12_executor.h). D3D12 only (no HIP counterpart). Built only when --only names them.
//
// fg_exec_capture: one lane per texel of a game texture (mip 0, read through the shader resource view at t0 in the
//   executor's descriptor table): dst[y * tw + x] := the texel as the arena image kind the stage kernels read
//   (clean/dx12/hlsl/fg_lite_common.hlsli): kind 4 = float16 x 4, kind 1 = float16 (channel 0), kind 5 = float32 x 2,
//   kind 6 = float32 (channel 0). Float16 values are rounded to nearest even (wg_f2h), so a float16 texture is copied
//   bit for bit and an 8 / 10-bit UNORM texture keeps every code (its float16 value converts back exactly).
// fg_exec_pack: one lane per pixel of a w x h float16 x 4 image (row stride sstride texels) into rows of dpitch bytes in
//   a presenter slot's format: fmt 0 = R16G16B16A16_FLOAT (the bits), 1 = R8G8B8A8_UNORM, 2 = B8G8R8A8_UNORM,
//   3 = R10G10B10A2_UNORM; UNORM codes are floor(sat(v) * max + 0.5) with sat(NaN) = 0.
//
// @wg-kernel fg_exec_capture co=- block=256 wave=32 args=p,u,u,u defines=FGX_K=1
// @wg-kernel fg_exec_pack co=- block=256 wave=32 args=p,u,p,u,u,u,u defines=FGX_K=2
#include "wg_platform.hlsli"

uint fgx_unorm(float v, float mx)
{
    const float s = min(max(v, 0.0f), 1.0f);
    return (uint)floor(mad(s, mx, 0.5f));
}

#if FGX_K == 1
WG_ARGS_BEGIN
    uint a0, a1, a2, a3;
WG_ARGS_END

Texture2D<float4> fgx_src : register(t0);

// fg_exec_capture: a0 = dst, a1 = tw, a2 = th, a3 = kind
WG_KERNEL(fg_exec_capture)
{
    WG_SETUP();
    const uint idx = wg_block_id() * 256u + wg_lane_id();
    const uint tw = a1, th = a2, kind = a3;
    if (idx >= tw * th) return;
    const uint x = idx % tw, y = idx / tw;
    const float4 c = fgx_src.Load(int3((int)x, (int)y, 0));
    if (kind == 4u)
        wg_st64(a0, idx * 8u, uint2(wg_f2h(c.x) | (wg_f2h(c.y) << 16), wg_f2h(c.z) | (wg_f2h(c.w) << 16)));
    else if (kind == 1u)
        wg_st16(a0, idx * 2u, wg_f2h(c.x));
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
    const uint w = a4, h = a5, fmt = a6;
    if (idx >= w * h) return;
    const uint x = idx % w, y = idx / w;
    const uint2 v = wg_ld64(a0, (y * a1 + x) * 8u);
    if (fmt == 0u) {
        wg_st64(a2, y * a3 + x * 8u, v);
        return;
    }
    const float r = f16tof32(v.x), g = f16tof32(v.x >> 16), b = f16tof32(v.y), a = f16tof32(v.y >> 16);
    uint word;
    if (fmt == 1u)
        word = fgx_unorm(r, 255.0f) | (fgx_unorm(g, 255.0f) << 8) | (fgx_unorm(b, 255.0f) << 16) |
               (fgx_unorm(a, 255.0f) << 24);
    else if (fmt == 2u)
        word = fgx_unorm(b, 255.0f) | (fgx_unorm(g, 255.0f) << 8) | (fgx_unorm(r, 255.0f) << 16) |
               (fgx_unorm(a, 255.0f) << 24);
    else
        word = fgx_unorm(r, 1023.0f) | (fgx_unorm(g, 1023.0f) << 10) | (fgx_unorm(b, 1023.0f) << 20) |
               (fgx_unorm(a, 3.0f) << 30);
    wg_st32(a2, y * a3 + x * 4u, word);
}
#endif
