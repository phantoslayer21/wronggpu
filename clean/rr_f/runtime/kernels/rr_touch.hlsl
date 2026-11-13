// SPDX-License-Identifier: MIT
// clean/rr-f-a14-frame E24 (WrongGPU's own code): rt.touch, a read-only cache warm-up pass. Every lane loads one
// dword of each of 4 consecutive 128-byte lines of [a0, a0 + a1 * 128) (a1 = lines) so later readers find them in the
// L2 / Infinity Cache; nothing is written unless a3 != 0 and the sum equals a2 (the runtime always passes a3 = 0; the
// opaque condition keeps the loads).
#include "wg_platform.hlsli"
WG_ARGS_BEGIN
    uint a0, a1, a2, a3;
WG_ARGS_END

WG_KERNEL(rr_rt_touch)
{
    WG_SETUP();
    const uint i = (wg_block_id() * WG_BLOCK + wg_lane_id()) * 4u;
    uint x = 0u;
    [unroll] for (uint k = 0u; k < 4u; ++k)
        if (i + k < a1) x += wg_ld32(a0, (i + k) * 128u);
    if (a3 != 0u && x == a2) wg_st32(a3, 0u, x);
}

// clean/rr-f-a14-frame E25 step 2: rt.blkcopy, the per-frame argument block (kBlock = 4096 bytes) from the upload ring
// (root SRV t0 space 9) into the arena: a0 = arena offset of the block, a1 = ring offset of this frame's slot. The arena
// then never leaves UNORDERED_ACCESS (no COPY_DEST transitions); one UAV barrier orders the block before its readers.
ByteAddressBuffer rr_blk_src : register(t0, space9);
WG_KERNEL(rr_rt_blkcopy)
{
    WG_SETUP();
    const uint i = wg_block_id() * WG_BLOCK + wg_lane_id();
    if (i < 1024u) wg_st32(a0, i * 4u, rr_blk_src.Load(a1 + i * 4u));
}
