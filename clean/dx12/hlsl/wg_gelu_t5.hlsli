// SPDX-License-Identifier: MIT
// GELU table T5 of the MLP hidden path: wg_gelu_lut.hlsli's exact q8(GELU) table re-indexed by the FP32 encoding of
// w = fma(h, S, B) instead of its FP16 code. Index j = asuint(w) >> 18 (sign, 8 exponent bits, the top 5 mantissa bits):
// 16384 entries (a, b) as uint2 in a structured buffer (t0, space 1; the D3D12 devices fill it at creation from
// runtime/gelu_t5_table.h). Every FP32 w of bucket j has the same FP16 slot f16_rtz(w) >> 5 (RTZ keeps the top 5 mantissa
// bits of a normal FP16 code; FP16 denormal, zero, overflow and infinity codes depend only on bits above bit 18), and
// T5[j] = T16[that slot]: the same entry, the same fma and the same byte as the arena table for every non-NaN w. A NaN w
// only comes from a NaN h, whose fma(h, a, b) is NaN with any entry. Proof over all 2^32 w: gelu_lut/t5_check.py.
// Per value the index is one v_lshrrev (no FP16 conversion, mask or offset arithmetic); the loads are idxen buffer
// loads of the read-only table.
// Include after wg_gelu_lut.hlsli (not included here: a sibling include would reach that header under a second path,
// which #pragma once does not recognise).
#ifndef WG_GELU_T5_HLSLI
#define WG_GELU_T5_HLSLI
#ifndef WG_GELU_T5
#define WG_GELU_T5 1
#endif
#if WG_GELU_T5
StructuredBuffer<uint2> wg_gelu_t5 : register(t0, space1);
uint wg_gelu_t5_index(float w) { return asuint(w) >> 18; }
#endif
#endif  // WG_GELU_T5_HLSLI
