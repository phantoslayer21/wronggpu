// SPDX-License-Identifier: GPL-3.0-only
// D3D12 port of frame generation's net1 skip add (issue #96): fg_add_f16 of fg_net1.co (clean/fg/kernels/fg_net1.hip),
// byte for byte. y = h(a + b) per element of n float16 values: the single-precision sum (one IEEE operation), then
// float16 by wg_f2h (nearest even); single precision then float16 equals one rounding of the exact sum of two float16
// values. NaN results are written as 0x7E00. Built only when --only names it (fg_ kernels are in no model's set).
//
// @wg-kernel fg_add_f16 co=fg_net1.co block=256 wave=32 args=p,p,p,u
#include "wg_platform.hlsli"

#define FG_FN
#define FG_I64 int64_t
#define FG_FBH32(v) firstbithigh((uint)(v))
#include "../../fg/kernels/fg_exact16.h"

WG_ARGS_BEGIN
    uint a, b, y;        // pointers (arena byte offsets)
    int n;
WG_ARGS_END

WG_KERNEL(fg_add_f16)
{
    WG_SETUP();
    const int idx = (int)(wg_block_id() * 256u + wg_lane_id());
    if (idx >= n) return;
    uint outv = wg_f2h(wg_h2f(wg_ld16(a, (uint)idx * 2u)) + wg_h2f(wg_ld16(b, (uint)idx * 2u)));
    if (fg_is_nan16(outv) != 0u) outv = FG_NAN16;
    wg_st16(y, (uint)idx * 2u, outv);
}
