// SPDX-License-Identifier: MIT
// Motion vectors at output resolution (spec 03 draft 4 "Motion at output resolution", spec 06 section 3): the raw
// motion buffer both stages read. Written for WrongGPU from those two pages.
//
// The buffer holds the game's motion texels of the M = (Mw, Mh) region (its subrect origin already applied by whoever
// filled it) as rows of `pitch` bytes; a texel is two FP16 (x, y; 4 bytes; exact for the FP16 motion formats games
// use) or two f32 (8 bytes). It has at least 4 spare bytes after its last texel: wg_mo_texel loads 8 bytes in either
// format and selects by the (uniform) format, so there is no branch per pixel.
#ifndef WG_MOTION_HI_H
#define WG_MOTION_HI_H

#include "wg_platform.h"

// The two words of the raw motion texel at byte offset off (a multiple of 4), and their values. Apart, so that a
// kernel can issue the loads long before it needs the values.
WG_FN void wg_mo_load(const void* motion, u32 off, u32& w0, u32& w1) {
  w0 = *(const u32*)((const char*)motion + off);
  w1 = *(const u32*)((const char*)motion + off + 4u);
}
WG_FN void wg_mo_value(u32 w0, u32 w1, int wide, float& vx, float& vy) {
  const float hx = (float)__builtin_bit_cast(_Float16, (unsigned short)(w0 & 0xffffu));
  const float hy = (float)__builtin_bit_cast(_Float16, (unsigned short)(w0 >> 16));
  vx = wide ? __builtin_bit_cast(float, w0) : hx;
  vy = wide ? __builtin_bit_cast(float, w1) : hy;
}
WG_FN void wg_mo_texel(const void* motion, u32 off, int wide, float& vx, float& vy) {
  u32 w0, w1;
  wg_mo_load(motion, off, w0, w1);
  wg_mo_value(w0, w1, wide, vx, vy);
}

#endif  // WG_MOTION_HI_H
