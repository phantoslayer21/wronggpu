// SPDX-License-Identifier: GPL-3.0-only
// Frame generation (issue #96): the presenter-slot packing shared by fg_exec_pack and the output kernels that write the
// packed slot rows directly (fg_out_pull_pack, fg_out_push_fine_pack). fmt: 1 = R8G8B8A8_UNORM, 2 = B8G8R8A8_UNORM,
// 3 = R10G10B10A2_UNORM; UNORM codes are floor(sat(v) * max + 0.5) with sat(NaN) = 0. v = the pixel as 4 x float16.
#pragma once

uint fgx_unorm(float v, float mx)
{
    const float s = min(max(v, 0.0f), 1.0f);
    return (uint)floor(mad(s, mx, 0.5f));
}

uint fgx_pack_pixel(uint2 v, uint fmt)
{
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
    return word;
}
