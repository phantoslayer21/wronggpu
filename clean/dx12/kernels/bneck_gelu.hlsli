// SPDX-License-Identifier: MIT
// The exact q8(GELU) table lookup of hlsl/wg_gelu_lut.hlsli (same table, same slots, same entries, same fma and
// conversion: the identical byte for every input) with fewer issued instructions per 8 values:
//
// - Offsets: wg_gelu_lut_off2 forms the low value's byte offset as (d >> 2) & 0x7fff with d = code pair & 0xffe0ffe0 (5 low
//   bits of both codes cleared). The driver's compiler knows bits 0..2 of (d >> 2) are zero, shrinks the mask to 0x7ff8
//   and can then no longer match a bit-field extract: v_lshrrev + v_and. Here bits 2..4 of the pre-mask are also opaque
//   (0 at run time: wg_bid_ < 2^31), so the mask keeps its low bits and (d >> 2) & 0x7fff is one v_bfe_u32: v_and, v_bfe,
//   v_lshrrev per pair instead of v_and, v_lshrrev, v_and, v_lshrrev. At run time the pre-mask is 0xffe0ffe0 as before.
// - Stage by stage over the 8 values (the 8 fmas, the 4 conversions, the masks, the offsets, then the 8 loads), so no
//   step waits on the one before it (no s_delay_alu) and the 8 loads form one clause instead of four.
// - The fmas from the last-loaded entry down: the first waits for all 8 loads and the other 7 need no wait (in load order
//   each fma waited for its own entry: 8 s_wait_loadcnt per 8 values).
#pragma once
#include "wg_gelu_lut.hlsli"
#include "wg_gelu_t5.hlsli"

wg_gelu_lut_e bn_gelu_fetch8(v8f h)
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
// wg_gelu_lut_finish8 with the fmas in reverse order (independent values: the same results)
uint2 bn_gelu_finish8(v8f h, wg_gelu_lut_e t)
{
    v8f y;
    [unroll] for (int i = 7; i >= 0; --i) y.v[i] = wg_gelu_lut_apply(h.v[i], t.e[i]);
    return wg_pack_q8_nc_ags(y);
}
uint2 bn_gelu_q8(v8f h) { return bn_gelu_finish8(h, bn_gelu_fetch8(h)); }
