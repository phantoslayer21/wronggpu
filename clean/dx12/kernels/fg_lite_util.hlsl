// SPDX-License-Identifier: GPL-3.0-only
// Frame generation (issue #96), light path: buffer fills and 4-channel copies (clean/fg/reference/pipeline.py "clear",
// "fill64", "prefill", "final_copy"). D3D12 only (no HIP counterpart). Built only when --only names them.
//
// fg_lite_fill16: n16 float16-sized units of dst := a pattern of `period` (1, 2 or 4) units p0..p3, unit i taking
//   p[i % period]; each lane writes 8 units (one 16-byte store when the 8 units are inside the buffer).
// fg_lite_copy4: dst[(dy0 + y) * dstride + dx0 + x] := src[(sy0 + y) * sstride + sx0 + x], 8-byte texels, x < w, y < h.
//
// @wg-kernel fg_lite_fill16 co=- block=256 wave=32 args=p,u,u,u,u,u,u defines=FG_K=1
// @wg-kernel fg_lite_fill16_p2 co=- entry=fg_lite_fill16 block=256 wave=32 args=p,u,u,u,u,u,u defines=FG_K=1,FG_FILL_P2=1
// @wg-kernel fg_lite_copy4 co=- block=256 wave=32 args=p,u,u,u,p,u,u,u,u,u defines=FG_K=2
#include "wg_platform.hlsli"
#include "fg_lite_common.hlsli"

#ifndef FG_FILL_P2
#define FG_FILL_P2 0
#endif
#if FG_K == 1
WG_ARGS_BEGIN
    uint dst, n16, p0, p1, p2, p3, period;
WG_ARGS_END

#if FG_FILL_P2
// A15 (fg_lite_fill16_p2, the executor only launches it with period 1, 2 or 4): `first` is a multiple of 8 and the period
// divides 8, so unit j of every lane's eight is p[j & (period - 1)]: the 16-byte word is built once from the arguments and
// the tail unit k = i & (period - 1); no integer division anywhere. The same bytes as fg_lite_fill16.
uint fg_unit_p2(uint i) { uint k = i & (period - 1u); return k == 0u ? p0 : (k == 1u ? p1 : (k == 2u ? p2 : p3)); }

WG_KERNEL(fg_lite_fill16)
{
    FG_SETUP();
    const uint first = (wg_block_id() * 256u + wg_lane_id()) * 8u;
    if (first >= n16) return;
    if (first + 8u <= n16)
    {
        const uint u0 = p0 & 0xffffu;
        const uint u1 = (period == 1u ? p0 : p1) & 0xffffu;
        const uint u2 = (period == 4u ? p2 : p0) & 0xffffu;
        const uint u3 = (period == 1u ? p0 : (period == 4u ? p3 : p1)) & 0xffffu;
        const uint lo = u0 | (u1 << 16), hi = u2 | (u3 << 16);
        wg_st128(dst, first * 2u, uint4(lo, hi, lo, hi));
        return;
    }
    [loop] for (uint i = first; i < n16; ++i) wg_st16(dst, i * 2u, fg_unit_p2(i) & 0xffffu);
}
#else
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
