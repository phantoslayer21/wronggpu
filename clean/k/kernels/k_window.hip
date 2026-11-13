// SPDX-License-Identifier: MIT
// Model K window partition for gfx1201 (spec 40, task K3): 8x8 windows with origin (ox, oy) and a margin of k tokens.
//
// Window (i, j) has its core columns [8 i - ox, 8 i - ox + 8) and rows [8 j - oy, 8 j - oy + 8); with a margin k it
// has S x S members, S = 8 + 2 k, starting at column 8 i - ox - k. Members outside the map follow the border rule
// (0 zero / invalid, 1 clamp, 2 mirror about the edge token without repeating it). Reference and the exact rules:
// clean/k/reference/layout.py (window_members, window_partition, window_reverse).
//
// k_win_gather:  map [H][W][C] -> windows [nwy nwx][S S][C] (window-major, members in raster order; zeros for
//                invalid members). One lane copies 16 bytes (8 channels).
// k_win_scatter: windows [nw][S S][C] -> map: each window's core members inside the map (every map token is the core
//                member of exactly one window).
// The device helpers wgk_border / wgk_win_member are what the real block kernels (spec 41) will use to address
// their windows directly, without a gathered copy.
#include "k_common.h"

enum { KB_ZERO = 0, KB_CLAMP = 1, KB_MIRROR = 2 };

// Coordinate p (any integer) mapped into [0, n) by the border rule; -1 = invalid (KB_ZERO).
WG_FN int wgk_border(int p, int n, int mode) {
  if (p >= 0 && p < n) return p;
  if (mode == KB_ZERO) return -1;
  if (mode == KB_CLAMP) return p < 0 ? 0 : n - 1;
  if (n == 1) return 0;
  const int period = 2 * (n - 1);
  int r = p % period;
  if (r < 0) r += period;
  return r < n ? r : period - r;
}

// Launch arguments (must match KWinParams in clean/k/runner/k_launch.py).
struct KWinParams {
  const h16* src;
  h16* dst;
  int w, h, c;       // map size and channels (c a multiple of 8)
  int ox, oy;        // window origin, 0 <= ox, oy < 8
  int margin, mode;  // margin k, border rule
  int nwx, nwy;      // window grid
  int count;         // work items (16-byte chunks)
};
static_assert(sizeof(KWinParams) == 56, "KWinParams must match clean/k/runner/k_launch.py");

// Flat map token of member (my, mx) of window (wx, wy) (margin included in the member coordinates), or -1.
WG_FN int wgk_win_member(const KWinParams& P, int wx, int wy, int mx, int my) {
  const int col = wgk_border(8 * wx - P.ox - P.margin + mx, P.w, P.mode);
  const int row = wgk_border(8 * wy - P.oy - P.margin + my, P.h, P.mode);
  return col < 0 || row < 0 ? -1 : row * P.w + col;
}

WG_KERNEL_N(k_win_gather, 128)(KWinParams P) {
  const int id = (int)(wg_block_id() * 128u + wg_lane_id());
  if (id >= P.count) return;
  const int cq = P.c >> 3, s = 8 + 2 * P.margin;
  const int q = id % cq, m = (id / cq) % (s * s), win = id / (cq * s * s);
  const int t = wgk_win_member(P, win % P.nwx, win / P.nwx, m % s, m / s);
  const v8h v = t >= 0 ? wgk_ld8(P.src + (size_t)t * P.c + 8 * q) : wgk_zero8();
  wgk_st8(P.dst + ((size_t)win * s * s + m) * P.c + 8 * q, v);
}

WG_KERNEL_N(k_win_scatter, 128)(KWinParams P) {
  const int id = (int)(wg_block_id() * 128u + wg_lane_id());
  if (id >= P.count) return;
  const int cq = P.c >> 3, s = 8 + 2 * P.margin;
  const int q = id % cq, m = (id / cq) % 64, win = id / (cq * 64);
  const int wx = win % P.nwx, wy = win / P.nwx;
  const int col = 8 * wx - P.ox + (m & 7), row = 8 * wy - P.oy + (m >> 3);
  if (col < 0 || col >= P.w || row < 0 || row >= P.h) return;
  const int member = ((m >> 3) + P.margin) * s + (m & 7) + P.margin;
  wgk_st8(P.dst + ((size_t)row * P.w + col) * P.c + 8 * q,
          wgk_ld8(P.src + ((size_t)win * s * s + member) * P.c + 8 * q));
}

WG_EMU_REGISTER(k_win_gather)
WG_EMU_REGISTER(k_win_scatter)
