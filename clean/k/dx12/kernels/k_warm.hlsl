// SPDX-License-Identifier: MIT
// k_warm: cache warm-up of model K's weights on the D3D12 executors (no HIP counterpart; runtime_k.cpp KRuntime::warm,
// runtime option k_warm). WrongGPU's own code. Launched unordered beside a stage whose grid leaves most of the GPU idle
// (dev_->unordered_next()), it reads one 4-byte word per `line` bytes of up to twelve byte ranges (the weight blobs of
// the latency-bound stages that follow), so their first weight loads hit the caches instead of memory.
// It changes nothing anyone reads: the XOR of the loaded words goes to the dummy word only when it equals key (the
// store keeps the loads alive; the dummy is a buffer of its own that no kernel reads).
// Args: KWarmParams by value (runtime_k.cpp): r[12], dummy (pointers), n[12] (bytes), line, key.
// @wg-kernel k_warm co=k_block_wide.co block=256 wave=32 args=s160:p13.b56
#include "wg_platform.hlsli"

WG_ARGS_BEGIN
    uint W_r0, W_r1, W_r2, W_r3, W_r4, W_r5, W_r6, W_r7, W_r8, W_r9, W_r10, W_r11, W_dummy;   // pointers (0 = null)
    uint W_n0, W_n1, W_n2, W_n3, W_n4, W_n5, W_n6, W_n7, W_n8, W_n9, W_n10, W_n11, W_line, W_key;
WG_ARGS_END

uint kwarm_range(uint r, uint n, uint first, uint stride)
{
    uint acc = 0u;
    [loop] for (uint o = first; o < n; o += stride)
        acc ^= wg_ld32(r, o);
    return acc;
}

WG_KERNEL(k_warm)
{
    WG_SETUP();
    const uint first = (wg_block_id() * WG_BLOCK + wg_lane_id()) * W_line;
    const uint stride = wg_grid_size() * WG_BLOCK * W_line;
    uint acc = kwarm_range(W_r0, W_n0, first, stride);
    acc ^= kwarm_range(W_r1, W_n1, first, stride);
    acc ^= kwarm_range(W_r2, W_n2, first, stride);
    acc ^= kwarm_range(W_r3, W_n3, first, stride);
    acc ^= kwarm_range(W_r4, W_n4, first, stride);
    acc ^= kwarm_range(W_r5, W_n5, first, stride);
    acc ^= kwarm_range(W_r6, W_n6, first, stride);
    acc ^= kwarm_range(W_r7, W_n7, first, stride);
    acc ^= kwarm_range(W_r8, W_n8, first, stride);
    acc ^= kwarm_range(W_r9, W_n9, first, stride);
    acc ^= kwarm_range(W_r10, W_n10, first, stride);
    acc ^= kwarm_range(W_r11, W_n11, first, stride);
    if (acc == W_key)
        wg_st32(W_dummy, 0u, acc);
}
