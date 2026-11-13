// SPDX-License-Identifier: MIT
// Swin speed-pass helpers (additive; swin_common.hlsli itself is unchanged, so bneck / dec1_dec0 compile as before).
// Included after swin_common.hlsli by swin_block.hlsl and swin_gemm.hlsl.
#pragma once

// sw_lane_base masked to the largest LDS array (32 KB = 4096 elements; the identity for every valid index): with the
// index range known, the driver's compiler pairs fragment reads into ds_load_2addr_b64.
uint sw_lane_base_m(uint unit, uint lane) { return ((unit >> 3) + lane + sw_zero_bits()) & 4095u; }

// sw_pack_q8 without the per-value NaN select (whose compare / select pairs serialize on VCC): pc = the bytes of the
// clamp max(min(x, 448), -448) (NaN -> +448: the driver's min drops the NaN), px = the bytes of x unclamped (every NaN
// -> 0xff, see sw_pack_q8). A byte whose px sign is set and pc sign is clear comes from a NaN (for any other x the
// clamp keeps the sign, -0 included): that byte becomes 0x7e | 0x81 = 0xff, HIP's byte. Every other byte is pc, the
// clamped value's byte. Same bytes as sw_pack_q8 (swin_diff with full-range FP8 + NaN codes).
uint2 sw_pack_q8_v2(v8f d)
{
    v8f c;
    [unroll] for (uint i = 0; i < 8; ++i) c.v[i] = max(min(d.v[i], 448.0f), -448.0f);
    const uint2 pc = wg_pack_q8_nc_ags(c), px = wg_pack_q8_nc_ags(d);
    const uint2 m = px & ~pc & 0x80808080u;
    return pc | m | (m >> 7);
}

// Output FP8 pack with a wave-uniform skip of the clamp. HIP packs cvt_pk_fp8(maximumminimum(x, -448, 448)). When every
// value of the wave is NaN or |x| <= 448, the clamp is the identity (-0 included) and the plain conversion gives HIP's
// bytes (every NaN -> 0xff on both sides): sw_absmax8 accumulates max |x| (the driver's max drops NaN), sw_in448 tests
// the whole wave (a wave of NaNs only takes the clamped path, also exact). Otherwise the clamped pack (sw_pack_q8_v2).
float sw_absmax8(float m, v8f d)
{
    return max(max(max(m, abs(d.v[0])), max(abs(d.v[1]), abs(d.v[2]))),
               max(max(abs(d.v[3]), abs(d.v[4])), max(max(abs(d.v[5]), abs(d.v[6])), abs(d.v[7]))));
}
bool sw_in448(float m) { return WaveActiveAllTrue(m <= 448.0f); }
