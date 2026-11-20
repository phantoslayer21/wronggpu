// SPDX-License-Identifier: MIT
// RDNA 3 (gfx11) layer over the platform layer of the D3D12 ports (WrongGPU's own code). A kernel that is built for gfx11
// (build_dx12.py --gfx 11, declaration gfx=11) includes THIS header first instead of "wg_platform.hlsli"; it includes that
// one itself. wg_platform.hlsli and wg_ags.hlsli stay as they are: they are part of model K's softmax self-check digest
// (clean/k/dx12/smx_selfcheck_digest.py), so every gfx11 definition lives here and in the gfx11 headers of the kernels.
//
// WG_GFX is 12 (RDNA 4, the default: this header then only includes the platform layer) or 11 (RDNA 3, defined by
// build_dx12.py --gfx 11). Under 11:
//  * the matrix instruction takes F16 only, with its own fragment layout (below); the FP8 matrix instruction, the FP8
//    conversion and the _ags packs of the platform layer do not exist on RDNA 3 and are hidden again (a shader that uses
//    one fails to compile, not to translate);
//  * the gfx12-shaped wg_wmma_f16 of the platform layer is renamed away while it is included and replaced by HIP's gfx11
//    form (operands in the gfx12 shape, one half-wave exchange per dword, then the K = 16 chain).
#pragma once

#ifndef WG_WAVE
#define WG_WAVE 32
#endif
#ifndef WG_GFX
#define WG_GFX 12
#endif
#if WG_GFX != 11 && WG_GFX != 12
#error "WG_GFX is 11 (RDNA 3) or 12 (RDNA 4)"
#endif
#if WG_GFX == 11 && WG_WAVE != 32
#error "WG_GFX 11 kernels run in wave32"
#endif

#if WG_GFX == 11
#define wg_wmma_f16 wg_wmma_f16_gfx12_shape
#endif
#include "wg_platform.hlsli"
#if WG_GFX == 11
#undef wg_wmma_f16

// Hidden on RDNA 3 (the identifiers are undeclared, so a use is a compile error).
#undef WG_AGS_FP8_F32
#undef WG_AGS_T_FP8_B
#define wg_wmma_fp8 wg_no_fp8_matrix_instruction_on_gfx11
#define wg_pack_q8_nc_ags wg_no_fp8_convert_on_gfx11
#define wg_pack_q8_ags wg_no_fp8_convert_on_gfx11
#define wg_pack_q8_inner_ags wg_no_fp8_convert_on_gfx11

// RDNA 3 (measured on gfx1100 through vkd3d-proton's raw fragments): the F16 instruction (format 0x5) takes 8 K values
// per call from the channel's fragment (K 0 .. 7; the K 8 .. 15 slots are zero) and, per lane l (half h = l / 16,
// c = l % 16):
//   A: accumulator row r reads its A values from lane 16 (r & 1) + r  -> hold row r's values in lanes r and r + 16;
//   B: column c reads lane l itself                                      -> lanes c and c + 16 hold column c;
//   C / D: element i of lane l is row 2 i + h, column c (wg_d_row).
// So an operand is replicated across the two halves of the wave, where gfx12 splits K between them. A K = 16 step is two
// chained instructions (the second takes K 8 .. 15 and the first's result as its C): the result equals HIP's one-
// instruction v_wmma_f32_16x16x16_f16 bit for bit.
bool wg_in_upper_half() { return (WaveGetLaneIndex() & 16u) != 0u; }
uint4 wg_xor16_4(uint4 v) { return uint4(wg_xor16_u(v.x), wg_xor16_u(v.y), wg_xor16_u(v.z), wg_xor16_u(v.w)); }
// One instruction: D = A x B + C over K 0 .. 7. a / b: the 8 halves of this lane's row / column, replicated across halves.
v8f wg_wmma_f16_k8(uint4 a, uint4 b, v8f c) { return wg_v8f_frag(wg_ags_mma(WG_AGS_F16_F32, wg_frag4(a), wg_frag4(b), wg_frag_v8f(c))); }
// K = 16: a0 / b0 hold K 0 .. 7, a1 / b1 K 8 .. 15 (replicated as above); two chained instructions.
v8f wg_wmma_f16_k16(uint4 a0, uint4 b0, uint4 a1, uint4 b1, v8f c)
{
    return wg_wmma_f16_k8(a1, b1, wg_wmma_f16_k8(a0, b0, c));
}
// Row of D held by accumulator element i of this lane (the gfx11 table: 2 i + h; packing.py D_ROW_LAYOUTS["gfx11"]).
uint wg_d_row(uint i) { return 2u * i + (WaveGetLaneIndex() >> 4); }
// HIP's wg_wmma_f16 on gfx11: operands in the gfx12 shape (lane l: row / column l % 16, half h holds K 8 h .. 8 h + 7);
// one half-wave exchange per dword, then the K = 16 chain.
v8f wg_wmma_f16(uint4 a, uint4 b, v8f c)
{
    const uint4 ya = wg_xor16_4(a), yb = wg_xor16_4(b);
    const bool hi = wg_in_upper_half();
    return wg_wmma_f16_k16(hi ? ya : a, hi ? yb : b, hi ? a : ya, hi ? b : yb, c);
}
#else
// Row of D held by accumulator element i of this lane (the gfx12 table: v + 8 h, h = lane / 16).
uint wg_d_row(uint i) { return i + 8u * (WaveGetLaneIndex() >> 4); }
#endif
