// SPDX-License-Identifier: GPL-3.0-only
// D3D12 port of frame generation's net1 max-pool (issue #96): fg_maxpool2_f16 of fg_net1.co
// (clean/fg/kernels/fg_net1.hip), byte for byte. 2 x 2 window, stride 2, floor sizes; from the first tap (0, 0), then
// (0, 1), (1, 0), (1, 1), the delivered step m <- (x > m) * x + (x <= m) * m in float16 (fg_pool_step16 of the shared
// integer code clean/fg/kernels/fg_exact16.h). x [C][H][W] float16, y [C][H / 2][W / 2] float16, one output element per
// lane. Built only when --only names it (fg_ kernels are in no model's kernel set).
//
// @wg-kernel fg_maxpool2_f16 co=fg_net1.co block=256 wave=32 args=p,p,u,u,u
#include "wg_platform.hlsli"

#define FG_FN
#define FG_I64 int64_t
#define FG_FBH32(v) firstbithigh((uint)(v))
#include "../../fg/kernels/fg_exact16.h"

WG_ARGS_BEGIN
    uint x, y;           // pointers (arena byte offsets)
    int C, H, W;
WG_ARGS_END

WG_KERNEL(fg_maxpool2_f16)
{
    WG_SETUP();
    const int Ho = H / 2;
    const int Wo = W / 2;
    const int idx = (int)(wg_block_id() * 256u + wg_lane_id());
    if (idx >= C * Ho * Wo) return;
    const int c = idx / (Ho * Wo);
    const int rem = idx - c * (Ho * Wo);
    const int oy = rem / Wo;
    const int ox = rem - oy * Wo;
    const uint p = (uint)((c * H + 2 * oy) * W + 2 * ox) * 2u;
    uint m = wg_ld16(x, p);
    m = fg_pool_step16(m, wg_ld16(x, p + 2u));
    m = fg_pool_step16(m, wg_ld16(x, p + (uint)W * 2u));
    m = fg_pool_step16(m, wg_ld16(x, p + (uint)W * 2u + 2u));
    wg_st16(y, (uint)idx * 2u, m);
}
