// SPDX-License-Identifier: MIT
// Shared helpers of the Swin ports (swin_block.hlsl, swin_gemm.hlsl): the HLSL counterparts of the helpers at the top of
// clean/kernels/swin.hip (wg_mirror, load_frags / store_frags, hsum8 / hmax8, lfrag) and the workgroup's LDS.
//
// The including file defines SW_LDS_BYTES (LDS bytes, at most 32 KB) before including this header. A "byte offset" below is
// an offset into the LDS (a multiple of 8: fragments 8 bytes per lane, staged chunks 16).
// Besides the arithmetic (which follows the HIP ISA), the helpers carry the code-generation fixes the driver's compiler
// needed to come near HIP's instruction count: rolled loops (sw_zero), folded LDS offsets (sw_zero_bits), the driver's
// FP8 / FP16 conversions (sw_unpack_fp8, sw_pack_q8, wg_*_ags).
#pragma once

#ifndef SW_LDS_BYTES
#error "define SW_LDS_BYTES before including swin_common.hlsli"
#endif
#if SW_LDS_BYTES > 32768
#error "D3D12 groupshared memory is limited to 32 KB"
#endif

// HIP wg_mirror: reflect an out-of-range coordinate at the border (p in [-(n-1), 2(n-1)]).
int sw_mirror(int p, int n)
{
    p = p < 0 ? -p : p;
    return p >= n ? 2 * (n - 1) - p : p;
}

// ---- LDS ------------------------------------------------------------------------------------------------------------
// LDS as 8-byte elements (every access is 8-byte aligned): a fragment read is one ds_load.
// The words are 64-bit integers: the driver's compiler then reads and writes fragments as ds_load_2addr_b64 /
// ds_store_2addr_b64 (with uint2 elements it split every access into two 32-bit halves: twice the LDS instructions,
// and 28 more VGPRs in swin_block_c64).
groupshared uint64_t sw_lds_q[SW_LDS_BYTES / 8];
uint2 sw_q2(uint64_t q) { return uint2((uint)q, (uint)(q >> 32)); }
uint64_t sw_2q(uint2 v) { return (uint64_t)v.x | ((uint64_t)v.y << 32); }
#define SW_LDS_GET(a) sw_q2(sw_lds_q[a])
#define SW_LDS_PUT(a, v) (sw_lds_q[a] = sw_2q(v))

uint2 sw_lds_ld64(uint off) { return SW_LDS_GET(off >> 3); }
uint4 sw_lds_ld128(uint off) { uint a = off >> 3; return uint4(SW_LDS_GET(a), SW_LDS_GET(a + 1u)); }
void sw_lds_st64(uint off, uint2 v) { SW_LDS_PUT(off >> 3, v); }
void sw_lds_st128(uint off, uint4 v) { uint a = off >> 3; SW_LDS_PUT(a, v.xy); SW_LDS_PUT(a + 1u, v.zw); }
// A loop bound the driver's compiler cannot fold (wg_grid_, the launch's workgroup count, is below 2^31, so this is
// always 0): loops written with it stay loops instead of being unrolled, as HIP's "#pragma unroll 1" loops (the driver
// unrolled the head and MLP loops in spite of [loop]: 43 KB of code for the C = 96 block).
uint sw_zero() { return wg_grid_ >> 31; }
// The same 0 with no known bits (0 or 0xffffffff for the compiler): added to a lane index it hides the index's range.
uint sw_zero_bits() { return 0u - (wg_grid_ >> 31); }

// HIP lfrag: lane l reads bytes [8 l, 8 l + 8) of fragment `index` of the LDS unit at byte `unit`.
// sw_lane_base(unit, lane) = the lane's element of fragment 0, computed once per unit; sw_lfrag_b(base, i) = fragment i.
// The lane is offset by sw_zero_bits() (0): the driver's compiler then cannot prove that the unit and lane bits are disjoint,
// keeps an addition instead of an OR, and folds 256 i into the ds_load offset (with the plain lane every fragment read
// cost an OR and a shift).
uint sw_lane_base(uint unit, uint lane) { return (unit >> 3) + lane + sw_zero_bits(); }
uint2 sw_lfrag_b(uint base, uint index) { return SW_LDS_GET(base + index * 32u); }
uint2 sw_lfrag(uint unit, uint index, uint lane) { return sw_lfrag_b(sw_lane_base(unit, lane), index); }

// ---- FP32 sums (HIP's ISA order and fusion) -------------------------------------------------------------------------
// hsum8(a * a) as clang contracts it: each pair is fma(a_even, a_even, a_odd * a_odd), then
// ((p01 + p23) + (p45 + p67)).
float sw_sumsq8(v8f a)
{
    float p01 = mad(a.v[0], a.v[0], a.v[1] * a.v[1]), p23 = mad(a.v[2], a.v[2], a.v[3] * a.v[3]);
    float p45 = mad(a.v[4], a.v[4], a.v[5] * a.v[5]), p67 = mad(a.v[6], a.v[6], a.v[7] * a.v[7]);
    return (p01 + p23) + (p45 + p67);
}
// hsum8: ((a0 + a1) + (a2 + a3)) + ((a4 + a5) + (a6 + a7))
float sw_hsum8(v8f a) { return ((a.v[0] + a.v[1]) + (a.v[2] + a.v[3])) + ((a.v[4] + a.v[5]) + (a.v[6] + a.v[7])); }
// hmax8 (exact in any order)
float sw_hmax8(v8f a)
{
    return max(max(max(a.v[0], a.v[1]), max(a.v[2], a.v[3])), max(max(a.v[4], a.v[5]), max(a.v[6], a.v[7])));
}

// 8 FP8 bytes -> 8 floats (HIP wg_unpack_fp8, v_cvt_f32_fp8). SW_UNPACK_AGS 1: the driver's matrix conversion FP8 B
// fragment -> F32 accumulator (per lane, byte i -> float i, wave-uniform control flow only; the driver emits
// v_cvt_pk_f32_fp8); 0: software (about 8 instructions per value). Same bits as HIP, NaN codes included (checked by the
// whole-kernel differential test with random and full-range FP8 inputs; this direction is not in probe/arith3).
#ifndef SW_UNPACK_AGS
#define SW_UNPACK_AGS 1
#endif
v8f sw_unpack_fp8(uint2 q)
{
#if SW_UNPACK_AGS
    return wg_v8f_frag(wg_ags_convert(wg_frag2(q), WG_AGS_T_FP8_B, WG_AGS_T_F32_ACC));
#else
    return wg_unpack_fp8(q);
#endif
}

// HIP wg_pack_q8 (clamp to +-448 with NaN kept, then v_cvt_pk_fp8_f32) with the driver conversion (wave-uniform control
// flow only). The NaN is not quieted here as wg_sat448 does: the conversion encodes every NaN as 0xff, so the bytes are
// those of wg_pack_q8 / wg_pack_q8_ags with one instruction less per value (3: clamp, NaN test, select).
uint2 sw_pack_q8(v8f d)
{
    [unroll] for (uint i = 0; i < 8; ++i) d.v[i] = isnan(d.v[i]) ? d.v[i] : min(max(d.v[i], -448.0f), 448.0f);
    return wg_pack_q8_nc_ags(d);
}

// v8f helpers
v8f sw_add8(v8f a, v8f b) { [unroll] for (uint i = 0; i < 8; ++i) a.v[i] = a.v[i] + b.v[i]; return a; }
v8f sw_zero8() { return wg_splat(0.0f); }
