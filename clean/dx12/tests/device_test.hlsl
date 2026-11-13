// SPDX-License-Identifier: MIT
// Test kernels of the hybrid D3D12 device (device_test.cpp; not part of any runtime build).
// @wg-kernel wg_test_add co=- block=64 wave=32 args=p,u,u
// @wg-kernel tst/wg_test_shift co=- entry=wg_test_shift block=64 wave=32 args=p,p,b8,p
#include "wg_platform.hlsli"

WG_ARGS_BEGIN
    uint a0, a1, a2, a3, a4;   // per kernel: add (p x, u n, u k); shift (p src, p dst, {u n, u k}, p null)
WG_ARGS_END

// x[i] += k for i < n
WG_KERNEL(wg_test_add)
{
    WG_SETUP();
    uint i = wg_block_id() * WG_BLOCK + wg_lane_id();
    if (i < a1) wg_st32(a0, i * 4u, wg_ld32(a0, i * 4u) + a2);
}

// dst[i] = src[(i + 1) % n] + k (reads what other workgroups of the previous dispatch wrote); null must arrive as 0.
WG_KERNEL(wg_test_shift)
{
    WG_SETUP();
    uint i = wg_block_id() * WG_BLOCK + wg_lane_id();
    if (i < a2) wg_st32(a1, i * 4u, wg_ld32(a0, ((i + 1u) % a2) * 4u) + a3 + a4 * 1000u);
}
