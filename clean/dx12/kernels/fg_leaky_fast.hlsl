// SPDX-License-Identifier: GPL-3.0-only
// D3D12 port of fg_leaky_fast_f16 of fg_net1_mma.co (clean/fg/kernels/fg_net1_mma.hip): fg_leaky_fast16 on n float16
// values, the exhaustive check of the fast path's leaky ReLU (runner/fg_net1_mma.py). Built only when --only names it.
//
// @wg-kernel fg_leaky_fast_f16 co=fg_net1_mma.co block=256 wave=32 args=p,p,u
#include "wg_platform.hlsli"

#define FG_FN
#define FG_I64 int64_t
#define FG_FBH32(v) firstbithigh((uint)(v))
#include "../../fg/kernels/fg_exact16.h"
#define FG_F2H(f) wg_f2h(f)
#define FG_H2F(h) wg_h2f(h)
#define FG_SEL(c, a, b) select(c, a, b)
#include "../../fg/kernels/fg_fast16.h"

WG_ARGS_BEGIN
    uint x, y;           // pointers (arena byte offsets)
    int n;
WG_ARGS_END

WG_KERNEL(fg_leaky_fast_f16)
{
    WG_SETUP();
    const int idx = (int)(wg_block_id() * 256u + wg_lane_id());
    if (idx >= n) return;
    wg_st16(y, (uint)idx * 2u, fg_leaky_fast16(wg_ld16(x, (uint)idx * 2u)));
}
