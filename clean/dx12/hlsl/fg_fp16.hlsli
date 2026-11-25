// SPDX-License-Identifier: MIT
#pragma once

static uint fg_fp16_zero;
#define FG_FP16_SETUP() (WG_SETUP(), fg_fp16_zero = wg_grid_ >> 31)

// Driver-channel FP32 to FP16 RNE. The opaque zero keeps the driver from folding the producing
// FP32 operation into a mixed-precision instruction and losing its separate FP32 rounding.
uint fg_f2h(float value)
{
    return wg_ags_call(0x1cu, 0u, 1u, asuint(value) ^ fg_fp16_zero, 0u) & 0xffffu;
}

uint fg_pack_h2(float x, float y) { return fg_f2h(x) | (fg_f2h(y) << 16); }
