// SPDX-License-Identifier: MIT
// Spec 51.5: regroup the four level-0 children for the E1 adapter, preserving every code bit.
#include "wg_platform.hlsli"
WG_ARGS_BEGIN
    uint source, target, width, height;
WG_ARGS_END
WG_KERNEL(rr_regroup_e1) {
    WG_SETUP();
    const uint q = wg_block_id() * WG_BLOCK + wg_lane_id();
    const uint count = width * height * 2u; // 32 bytes per source token, copied in 16-byte chunks.
    if (q >= count) return;
    const uint parent = q / 8u, within = q % 8u, child = within / 2u, half = within % 2u;
    const uint x = parent % (width / 2u), y = parent / (width / 2u);
    const uint token = (2u * y + child / 2u) * width + 2u * x + child % 2u;
    wg_st128(target, q * 16u, wg_ld128(source, token * 32u + half * 16u));
}
