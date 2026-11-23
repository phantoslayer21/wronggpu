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

// ---- native K = 16 operands (developer-only; the K = 16 opcode needs the translator patch of the native RDNA 3 work) --
// A native operand is the 16 F16 values one lane hands to ONE K = 16 instruction: lo = K 0 .. 7, hi = K 8 .. 15 of its
// row (A) or column (B), the same in both half-waves, K in index order (the order of the shortcut's operands, so the
// products round as the shortcut's do). Built once and reused, it replaces the per-call exchange of wg_wmma_f16 above.
struct wg_h16 { uint4 lo; uint4 hi; };
// Opcode WG_AGS_F16_F32_K16: the translator hands all 8 dwords of the A and B fragments (16 halves, element 2 d + e = half
// e of dword d) to one cooperative-matrix multiply, with no exchange of its own. Not an AMD opcode (0x7e is past AMD's
// list): a stock translator does not know it, so a PSO that uses it fails to build, loudly.
#define WG_AGS_F16_F32_K16 0x7eu
wg_frag wg_frag8(uint4 lo, uint4 hi)
{
    wg_frag f;
    f.v[0] = lo.x; f.v[1] = lo.y; f.v[2] = lo.z; f.v[3] = lo.w;
    f.v[4] = hi.x; f.v[5] = hi.y; f.v[6] = hi.z; f.v[7] = hi.w;
    return f;
}
// D = A x B + C, K = 16, one instruction. The same arguments and the same result as wg_wmma_f16_k16 (two chained K = 8
// instructions) when the instruction sums in index order.
v8f wg_wmma_f16_k16n(uint4 a0, uint4 b0, uint4 a1, uint4 b1, v8f c)
{
    return wg_v8f_frag(wg_ags_mma(WG_AGS_F16_F32_K16, wg_frag8(a0, a1), wg_frag8(b0, b1), wg_frag_v8f(c)));
}
v8f wg_mma16(wg_h16 a, wg_h16 b, v8f c) { return wg_wmma_f16_k16n(a.lo, b.lo, a.hi, b.hi, c); }

// Row map of a native A operand. The instruction reads accumulator row p's A values from lane 16 (p & 1) + p, and
// element i of a lane of half h is row 2 i + h. A lane that holds the gfx12 row wg_a_row(lane % 16) = 8 (p & 1) + p / 2
// therefore leaves element i of half h holding gfx12 row 8 h + i: the gfx12 accumulator layout, so everything after the
// product (bias, rounding, stores, the next product's B operand) and the weight packing stay as they are on RDNA 4.
// WG_A_ROW_MUT (tests only, never in a build): a wrong table (rows 2 and 3 swapped), the negative control of the probes.
uint wg_a_row(uint p)
{
#if WG_A_ROW_MUT
    p = (p & 14u) == 2u ? p ^ 1u : p;
#endif
    return 8u * (p & 1u) + (p >> 1);
}
uint wg_a_row_lane() { return wg_a_row(WaveGetLaneIndex() & 15u); }
// A from a 16 x 16 tile stored in the gfx12 fragment order (256 halfs: lane 16 h + r holds row r, K 8 h .. 8 h + 7) in
// the arena: the two 16-byte slots of row wg_a_row_lane(). tile_h: the tile's offset in halfs from base (a multiple of 8).
wg_h16 wg_ldA16(uint base, uint tile_h)
{
    const uint r = wg_a_row_lane();
    wg_h16 a;
    a.lo = wg_ld128(base, (tile_h + 8u * r) * 2u);
    a.hi = wg_ld128(base, (tile_h + 8u * (16u + r)) * 2u);
    return a;
}
// B from a tile in the same order: the two slots of column lane % 16 (no exchange, no row map).
wg_h16 wg_ldB16(uint base, uint tile_h)
{
    const uint c = WaveGetLaneIndex() & 15u;
    wg_h16 b;
    b.lo = wg_ld128(base, (tile_h + 8u * c) * 2u);
    b.hi = wg_ld128(base, (tile_h + 8u * (16u + c)) * 2u);
    return b;
}
// B from this lane's own fragment in the gfx12 shape (column lane % 16, half h holds K 8 h .. 8 h + 7: an accumulator
// rounded to FP16, or a fragment loaded as on RDNA 4): one half-wave exchange of its 4 dwords. The same values serve as
// the A operand of a product whose rows are tokens (v^T = n^T Wv^T), with the instruction's own row order (no row map).
wg_h16 wg_b16(uint4 own)
{
    const uint4 y = wg_xor16_4(own);
    const bool hi = wg_in_upper_half();
    wg_h16 b;
    b.lo = hi ? y : own;
    b.hi = hi ? own : y;
    return b;
}
// A product whose A rows were NOT row-mapped leaves element i of half h holding row 2 i + h. This puts such a fragment
// (rounded to FP16) into the gfx12 shape (half h holds rows 8 h .. 8 h + 7 in order): each lane keeps its own elements
// 4 h .. 4 h + 3, fetches the partner's (one exchange of 2 dwords) and interleaves them (row 8 h + 2 m from half 0,
// row 8 h + 2 m + 1 from half 1).
uint4 wg_rows_to_gfx12(uint4 d)
{
    const bool hi = wg_in_upper_half();
    const uint2 own = hi ? uint2(d.z, d.w) : uint2(d.x, d.y);
    const uint2 snd = hi ? uint2(d.x, d.y) : uint2(d.z, d.w);
    const uint2 rcv = uint2(wg_xor16_u(snd.x), wg_xor16_u(snd.y));
    const uint2 e = hi ? rcv : own, o = hi ? own : rcv;   // rows of half 0 (even) / half 1 (odd)
    return uint4((e.x & 0xffffu) | (o.x << 16), (e.x >> 16) | (o.x & 0xffff0000u),
                 (e.y & 0xffffu) | (o.y << 16), (e.y >> 16) | (o.y & 0xffff0000u));
}
#else
// Row of D held by accumulator element i of this lane (the gfx12 table: v + 8 h, h = lane / 16).
uint wg_d_row(uint i) { return i + 8u * (WaveGetLaneIndex() >> 4); }
#endif
