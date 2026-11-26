// SPDX-License-Identifier: MIT
// The matrix helpers of wave64 kernels (WG_WAVE 64; wg_platform.hlsli defines its own only for wave32): "design D" of the
// wave64 ports (clean/rr_f: rr_block_c32.hlsl, rr_block_c64_w64.hlsl), one wave64 = the work of two wave32 waves, tile t (0, 1)
// = one wave32 wave's 16 tokens. Wave64 lane L = l + 32 hh holds, for each tile, half hh of wave32 lane l's fragment: FP8 A / B
// bytes 4 hh .. 4 hh + 3, accumulator elements 4 hh .. 4 hh + 3 (the v_wmma wave64 layout; a wave64 v_wmma is the wave32 one bit
// for bit and the conversions are per value, so every output keeps its bits). A v8f holds elements 0-3 of tile 0 and 0-3 of
// tile 1; a uint2 FP8 fragment: x = tile 0, y = tile 1 (one dword each); a uint4 of packed FP16: x, y = tile 0, z, w = tile 1.
// A shared operand (the weights) passes the same dword in both: uint2(w, w).
#pragma once
#if WG_WAVE != 64
#error "wg_w64.hlsli: wave64 kernels only"
#endif
#include "wg_ags.hlsli"
wg_frag wg_frag_v8f(v8f c) { wg_frag f; [unroll] for (uint i = 0; i < 8; ++i) f.v[i] = asuint(c.v[i]); return f; }
v8f wg_v8f_frag(wg_frag f) { v8f c; [unroll] for (uint i = 0; i < 8; ++i) c.v[i] = asfloat(f.v[i]); return c; }
wg_frag wg_w64f(uint a, uint b, uint c, uint d)   // a wave64 fragment: the first 4 / 2 / 1 dwords are used
{
    wg_frag f;
    f.v[0] = a; f.v[1] = b; f.v[2] = c; f.v[3] = d;
    [unroll] for (uint i = 4; i < 8; ++i) f.v[i] = 0u;
    return f;
}
v8f wg_wmma_fp8(uint2 a, uint2 b, v8f c)   // tile 0: (a.x, b.x, c 0-3), tile 1: (a.y, b.y, c 4-7)
{
    const wg_frag r0 = wg_ags_mma(WG_AGS_FP8_F32, wg_w64f(a.x, 0u, 0u, 0u), wg_w64f(b.x, 0u, 0u, 0u),
                                  wg_w64f(asuint(c.v[0]), asuint(c.v[1]), asuint(c.v[2]), asuint(c.v[3])));
    const wg_frag r1 = wg_ags_mma(WG_AGS_FP8_F32, wg_w64f(a.y, 0u, 0u, 0u), wg_w64f(b.y, 0u, 0u, 0u),
                                  wg_w64f(asuint(c.v[4]), asuint(c.v[5]), asuint(c.v[6]), asuint(c.v[7])));
    v8f d;
    [unroll] for (uint i = 0; i < 4; ++i) { d.v[i] = asfloat(r0.v[i]); d.v[4 + i] = asfloat(r1.v[i]); }
    return d;
}
uint2 wg_pack_q8_nc_ags(v8f d)   // v_cvt_pk_fp8_f32 per value pair, element i -> byte i of its tile's dword
{
    const wg_frag r0 = wg_ags_convert(wg_w64f(asuint(d.v[0]), asuint(d.v[1]), asuint(d.v[2]), asuint(d.v[3])),
                                      WG_AGS_T_F32_ACC, WG_AGS_T_FP8_B);
    const wg_frag r1 = wg_ags_convert(wg_w64f(asuint(d.v[4]), asuint(d.v[5]), asuint(d.v[6]), asuint(d.v[7])),
                                      WG_AGS_T_F32_ACC, WG_AGS_T_FP8_B);
    return uint2(r0.v[0], r1.v[0]);
}
uint2 wg_pack_q8_ags(v8f d)   // the clamp of wg_pack_q8 (wg_sat448, per value), then the conversion
{
    [unroll] for (uint i = 0; i < 8; ++i) d.v[i] = wg_sat448(d.v[i]);
    return wg_pack_q8_nc_ags(d);
}
uint4 wg_to_f16_ags(v8f d)       // v_cvt_f16_f32 per value (round to nearest even)
{
    const wg_frag r0 = wg_ags_convert(wg_w64f(asuint(d.v[0]), asuint(d.v[1]), asuint(d.v[2]), asuint(d.v[3])),
                                      WG_AGS_T_F32_ACC, WG_AGS_T_F16_ACC);
    const wg_frag r1 = wg_ags_convert(wg_w64f(asuint(d.v[4]), asuint(d.v[5]), asuint(d.v[6]), asuint(d.v[7])),
                                      WG_AGS_T_F32_ACC, WG_AGS_T_F16_ACC);
    return uint4(r0.v[0], r0.v[1], r1.v[0], r1.v[1]);
}
