// SPDX-License-Identifier: MIT
// Model L (issue #89): Swin block kernels at the widths Model L adds, from Model M's block template
// (clean/kernels/swin.hip, included for its templates only). Same arithmetic, packed weights
// (clean/runner/packing.py pack_block) and launch geometry as Model M's block kernels (clean/FORMATS.md, "Kernels and
// launch geometry"): 1-D grid of ceil(nwin / 8) workgroups of 256 lanes, one window per wave.
//
// Build: clang -x hip ... -I clean/kernels clean/l/kernels/l_swin.hip -o l_swin.co (clean/l/runner/l_build.py).
#define WG_SWIN_NO_EXPORTS 1
#include "swin.hip"

#ifndef WG_MINW_C32
#define WG_MINW_C32 1
#endif

// x, xadd (unused, may be null), out, w, H, W, ox, oy: as swin_block_c64. `_nc` when bit 0 of the blob's flags word is
// set (the weights prove every inner FP8 pack in range), with identical results.
WG_KERNEL_NW(l_swin_block_c32, 256, WG_MINW_C32)(const u8* x, const u8* xadd, u8* out, const u8* w, int H, int W,
                                                  int ox, int oy) {
  swin_block<32, 2, 128, 1, false, WG_MLP_GROUP, false, 256>(x, xadd, out, w, H, W, ox, oy, nullptr);
}
WG_KERNEL_NW(l_swin_block_c32_nc, 256, WG_MINW_C32)(const u8* x, const u8* xadd, u8* out, const u8* w, int H, int W,
                                                     int ox, int oy) {
  swin_block<32, 2, 128, 1, false, WG_MLP_GROUP, true, 256>(x, xadd, out, w, H, W, ox, oy, nullptr);
}

// The same template at Model M's C = 64 (as swin_block_c64[_nc] of swin.co): the GPU check runs it next to Model M's
// own kernel to show that this file builds the template exactly as swin.co does.
WG_KERNEL_NW(l_swin_block_c64, 256, WG_MINW_C64)(const u8* x, const u8* xadd, u8* out, const u8* w, int H, int W,
                                                  int ox, int oy) {
  swin_block<64, 2, 256, 1, false, WG_MLP_GROUP, false, 256>(x, xadd, out, w, H, W, ox, oy, nullptr);
}
