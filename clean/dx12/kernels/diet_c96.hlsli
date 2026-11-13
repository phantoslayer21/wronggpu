// SPDX-License-Identifier: MIT
// c96 Swin block instruction diet (track diet-c96): helpers of swin_block.hlsl's fused expand (swin_block_c96_add_ex_nc).
// Included by swin_block.hlsl after its wg_low0 (the kernel arguments H, W, ox, oy are in scope).
#pragma once

// wg_low0(p0, n) for p0 = 4 w - o (o = ox or oy, 0 or 2; w a window row / column of the map): the four mirrored
// coordinates p0 .. p0 + 3 are unimodal (rising, then falling past the border), so their minimum is at an end, and
// mirror(p) = min(p, 2 n - 2 - p) for p >= 0: min(max(p0, 0), 2 n - 5 - p0) >> 1. Equal to wg_low0 for every
// p0 = 4 w - o, w in [0, ceil((n + o) / 4)), o in {0, 1, 2}, n in [1, 4096] (exhaustive: 8.4 million cases, 0
// differences; a local proof script, not included).
int sw_low0(int p0, int n) { return min(max(p0, 0), 2 * n - 5 - p0) >> 1; }

// The low-resolution bases of window blk * 8 + (lane & 7) (clamped to the grid as the kernel clamps its windows):
// row base | column base << 16 (both in [0, 2^15)). Every lane of a wave holds the 8 windows of the workgroup 4 times
// over; fused_expand reads the 2 it needs from lane 4 n + (t >> 2), the block its own window's from lane wave_id.
uint sw_win_low0(int nwx, int nwin, uint lane, int blk)
{
    int win = blk * 8 + (int)(lane & 7u);
    win = win < nwin ? win : nwin - 1;
    const int wy = win / nwx, wx = win - wy * nwx;
    return (uint)sw_low0(4 * wy - oy, H) | ((uint)sw_low0(4 * wx - ox, W) << 16);
}
