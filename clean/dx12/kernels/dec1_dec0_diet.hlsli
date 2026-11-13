// SPDX-License-Identifier: MIT
// Instruction-diet helpers of dec1_dec0.hlsl (d1p12/swin_block_c64_add_dec0_nc). Each helper gives the bytes of the code
// it replaces (HIP's bytes) for every input.
#pragma once
#include "wg_gelu_lut.hlsli"
#include "wg_gelu_t5.hlsli"

// The exact q8(GELU) arena-table lookup of hlsl/wg_gelu_lut.hlsli (same table, slots, entries, fma and conversion: the
// identical byte for every input) with fewer issued instructions per 8 values, in the bneck_gelu.hlsli form:
// - offsets: bits 2..4 of the pre-mask opaque (0 at run time: wg_bid_ < 2^31), so the driver keeps (d >> 2) & 0x7fff as
//   one v_bfe_u32 (v_and, v_bfe, v_lshrrev per pair instead of v_and, v_lshrrev, v_and, v_lshrrev); at run time the
//   pre-mask is 0xffe0ffe0 as in wg_gelu_lut_off2;
// - stage by stage over the 8 values (fmas, conversions, masks, offsets, then the 8 loads as one clause).
wg_gelu_lut_e dd_gelu_fetch8(v8f h)
{
    const uint z = wg_bid_ >> 31;
    const uint msk = 0xffe0ffe0u | (z << 16) | (z * 0x1cu);
    float w[8];
    [unroll] for (uint i = 0; i < 8; ++i) w[i] = mad(h.v[i], WG_GELU_LUT_S, WG_GELU_LUT_B);
#if WG_GELU_T5
    // T5 (hlsl/wg_gelu_t5.hlsli): entries indexed by the FP32 bits of w (one shift per value), the same entries
    wg_gelu_lut_e r;
    [unroll] for (uint i5 = 0; i5 < 8; ++i5) r.e[i5] = wg_gelu_t5[wg_gelu_t5_index(w[i5])];
#else
    uint d[4];
    [unroll] for (uint i1 = 0; i1 < 4; ++i1) d[i1] = f32tof16(w[2 * i1]) | (f32tof16(w[2 * i1 + 1]) << 16);
    [unroll] for (uint i2 = 0; i2 < 4; ++i2) d[i2] = d[i2] & msk;
    uint o[8];
    [unroll] for (uint i3 = 0; i3 < 4; ++i3) o[2 * i3] = WG_GELU_LUT_TAB + ((d[i3] >> 2) & 0x7fffu);
    [unroll] for (uint i4 = 0; i4 < 4; ++i4) o[2 * i4 + 1] = WG_GELU_LUT_TAB + (d[i4] >> 18);
    wg_gelu_lut_e r;
    [unroll] for (uint i5 = 0; i5 < 8; ++i5) r.e[i5] = wg_mem.Load2(o[i5]);
#endif
    return r;
}
