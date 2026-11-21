// SPDX-License-Identifier: MIT
// Frame generation (issue #96), light path: buffer fills and 4-channel copies (clean/fg/reference/pipeline.py "clear",
// "fill64", "prefill", "final_copy"). D3D12 only (no HIP counterpart). Built only when --only names them.
//
// fg_lite_fill16: n16 float16-sized units of dst := a pattern of `period` (1, 2 or 4) units p0..p3, unit i taking
//   p[i % period]; each lane writes 8 units (one 16-byte store when the 8 units are inside the buffer).
// fg_lite_copy4: dst[(dy0 + y) * dstride + dx0 + x] := src[(sy0 + y) * sstride + sx0 + x], 8-byte texels, x < w, y < h.
//
// @wg-kernel fg_lite_fill16 co=- block=256 wave=32 args=p,u,u,u,u,u,u defines=FG_K=1
// @wg-kernel fg_lite_copy4 co=- block=256 wave=32 args=p,u,u,u,p,u,u,u,u,u defines=FG_K=2
#include "wg_platform.hlsli"
#include "fg_lite_common.hlsli"

#if FG_K == 1
WG_ARGS_BEGIN
    uint dst, n16, p0, p1, p2, p3, period;
WG_ARGS_END

uint fg_unit(uint i) { uint k = i % period; return k == 0u ? p0 : (k == 1u ? p1 : (k == 2u ? p2 : p3)); }

WG_KERNEL(fg_lite_fill16)
{
    FG_SETUP();
    const uint first = (wg_block_id() * 256u + wg_lane_id()) * 8u;
    if (first >= n16) return;
    if (first + 8u <= n16)
    {
        uint4 w;
        w.x = (fg_unit(first) & 0xffffu) | (fg_unit(first + 1u) << 16);
        w.y = (fg_unit(first + 2u) & 0xffffu) | (fg_unit(first + 3u) << 16);
        w.z = (fg_unit(first + 4u) & 0xffffu) | (fg_unit(first + 5u) << 16);
        w.w = (fg_unit(first + 6u) & 0xffffu) | (fg_unit(first + 7u) << 16);
        wg_st128(dst, first * 2u, w);
        return;
    }
    [loop] for (uint i = first; i < n16; ++i) wg_st16(dst, i * 2u, fg_unit(i) & 0xffffu);
}
#endif

#if FG_K == 2
WG_ARGS_BEGIN
    uint dst, dstride, dx0, dy0, src, sstride, sx0, sy0, cw, ch;
WG_ARGS_END

WG_KERNEL(fg_lite_copy4)
{
    FG_SETUP();
    const uint idx = wg_block_id() * 256u + wg_lane_id();
    if (idx >= cw * ch) return;
    const uint x = idx % cw, y = idx / cw;
    wg_st64(dst, ((dy0 + y) * dstride + dx0 + x) * 8u, wg_ld64(src, ((sy0 + y) * sstride + sx0 + x) * 8u));
}
#endif
