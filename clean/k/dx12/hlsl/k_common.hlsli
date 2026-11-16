// SPDX-License-Identifier: MIT
// Model K network helpers for the D3D12 ports: the HLSL counterpart of clean/k/kernels/k_common.h (WrongGPU's own code),
// with the same names (wgk_*) and the same bits as the HIP kernels built from it (gfx1201, k_block.co: WGK_PAIR_CVT = 1,
// WGK_ASM_CLAMP = 1, FP32 accumulation; the FP16-accumulating KAcc<true> forms are not ported).
//
// Include AFTER the kernel's WG_ARGS_END: the opaque zero below reads the runtime constant wg_grid_.
//
// Types: a fragment of 8 FP16 values (HIP v8h) is a uint4, element i in dword i / 2, low half first; an FP32
// accumulator (HIP v8f) is the platform's v8f. WMMA layout and lane mapping as in k_common.h: weights = A operand
// (rows = output channels), tokens = B operand (columns = tokens); lane l holds column l % 16 and the 8 contiguous
// K elements 8 (l / 16) + 0..7; the gfx12 accumulator rows d_row(l / 16, 0..7) = 8 (l / 16) + 0..7 (the packer
// absorbs any other order, k_pack.py; the v^T product with swapped operands relies on this gfx12 layout as in HIP).
//
// Exactness on this driver (clean/dx12/hlsl/wg_math.hlsli, clean/dx12/README.md "Driver exceptions"):
//  * FP16 elementwise operations are native half2 operations (v_pk_mul_f16 / v_pk_add_f16 / v_pk_max_num_f16 ...,
//    each one IEEE FP16 operation rounded to nearest even, FP16 denormals kept), exactly the packed operations of the
//    HIP kernels (k_common.h: contraction off, one rounding per operation). The driver may fuse an FP16 multiply with a
//    following add (v_fma_mix) or regroup multiplication chains even under -Gis, so every FP16 result that feeds
//    another floating-point operation goes through an opaque zero (wgk_opq: x ^ (wg_grid_ >> 31), 0 at run time and
//    not provable by the compiler): the next operation sees an integer result and cannot fuse with or regroup it.
//  * FP32 -> FP16 of whole accumulator fragments (wgk_round8, HIP v_cvt_f16_f32 + v_fma_mixhi_f16 x * 1 + -0, round to
//    nearest even) is the driver's matrix conversion (op 0x32, RNE; the plain f32tof16 / (half) casts round toward
//    zero on this driver). Its input is always a WMMA result or an integer-made value (nothing to fold into it).
//    Scalars: wg_f2h (software RNE).
//  * Driver channel calls (WMMA, conversion) only in wave-uniform control flow, with branchless neighbours.
#pragma once

#include "k_smx.hlsli"

// ---- the opaque zero and half2 views ---------------------------------------------------------------------------------
// wgk_zero() is 0 at run time (grids < 2^31) and unknown to the compiler; wgk_opq xors it into BOTH halves of a packed
// pair (x 0x10001): with the plain 0 / 1 value the driver proved the high half's xor zero and dropped it (then fused
// that half's multiply and add into v_pk_fma_f16).
uint wgk_zero() { return wg_grid_ >> 31; }
uint wgk_opq(uint v) { return v ^ (wgk_zero() * 0x10001u); }
uint4 wgk_opq4(uint4 v) { return v ^ (wgk_zero() * 0x10001u); }
half2 wgk_h2(uint u) { return asfloat16(uint16_t2((uint16_t)(u & 0xffffu), (uint16_t)(u >> 16))); }
uint wgk_u(half2 h) { uint16_t2 v = asuint16(h); return (uint)v.x | ((uint)v.y << 16); }
half2 wgk_c2(uint h) { return wgk_h2((h & 0xffffu) * 0x10001u); }   // splatted FP16 constant (bits; no cast rounding)
// FP16 element i (0..7) of a fragment, as bits / as FP32 (exact)
uint wgk_el(uint4 v, uint i) { return (v[i >> 1] >> ((i & 1u) * 16u)) & 0xffffu; }
float wgk_elf(uint4 v, uint i) { return f16tof32(wgk_el(v, i)); }

// Pin: 0 at run time, a wave-wide reduction of v (WaveActiveBitOr of v & the opaque zero). Added to the base pointer of
// a predicated store, it keeps the matrix products that feed the stored values in wave-uniform control flow: without it
// the driver sank a whole GEMM (its v_wmma included) into the store's exec-masked branch, where lanes outside the map
// were off and the products of the other lanes came out wrong (k_stage_e2's merge at the map border). A reduction
// over the wave cannot be moved into divergent control flow, so everything it reads is computed before the branch.
uint wgk_pin(uint v) { return WaveActiveBitOr(v & wgk_zero()); }

// ---- 16-byte fragments -----------------------------------------------------------------------------------------------
// Global: base pointer (arena byte offset) + offset in halfs; 16-byte aligned. LDS: kb_lds (the kernel file declares
// groupshared uint4 kb_lds[]) indexed in halfs (multiples of 8).
uint4 wgk_ld8(uint base, uint off_h) { return wg_ld128(base, off_h * 2u); }
void wgk_st8(uint base, uint off_h, uint4 v) { wg_st128(base, off_h * 2u, v); }
uint4 wgk_zero8() { return uint4(0u, 0u, 0u, 0u); }
v8f wgk_zero8f() { return wg_splat(0.0f); }

// Value of v in lane (lane ^ 16): the other half of the wave (the other 8 channels of the same token).
float wgk_xor16(float v) { return wg_xor16(v); }
uint wgk_xor16_u(uint v) { return wg_xor16_u(v); }

// FP32 accumulators -> FP16, round to nearest even (the driver's matrix conversion; HIP wgk_round8).
uint4 wgk_round8(v8f d) { return wg_to_f16_ags(d); }

// FP16 fragment -> FP32 accumulator (exact), e.g. a bias that a matrix product starts from (HIP wgk_widen8).
v8f wgk_widen8(uint4 a)
{
    v8f r;
    [unroll] for (uint i = 0; i < 8u; ++i) r.v[i] = wgk_elf(a, i);
    return r;
}

// D = A x B + C, 16x16x16, FP16 A / B, FP32 accumulation (v_wmma_f32_16x16x16_f16; HIP KAcc<false>::mma).
v8f wgk_mma(uint4 a, uint4 b, v8f c) { return wg_wmma_f16(a, b, c); }

// ---- exact FP16 elementwise operations (one IEEE FP16 operation each; result through the opaque zero) ----------------
uint4 wgk_add8(uint4 a, uint4 b)
{
    uint4 r;
    [unroll] for (uint i = 0; i < 4u; ++i) r[i] = wgk_opq(wgk_u(wgk_h2(a[i]) + wgk_h2(b[i])));
    return r;
}
uint4 wgk_sub8(uint4 a, uint4 b)
{
    uint4 r;
    [unroll] for (uint i = 0; i < 4u; ++i) r[i] = wgk_opq(wgk_u(wgk_h2(a[i]) - wgk_h2(b[i])));
    return r;
}
uint4 wgk_mul8(uint4 a, uint4 b)
{
    uint4 r;
    [unroll] for (uint i = 0; i < 4u; ++i) r[i] = wgk_opq(wgk_u(wgk_h2(a[i]) * wgk_h2(b[i])));
    return r;
}
uint4 wgk_splat8h(uint h)   // h: FP16 bits
{
    const uint d = (h & 0xffffu) * 0x10001u;
    return uint4(d, d, d, d);
}
// min(max(x, lo), hi) and max(x, lo) per element with FP16 constants (v_pk_max_num_f16 / v_pk_min_num_f16: IEEE
// maxNum / minNum, a NaN operand gives the other; HIP wgk_clamp8c / wgk_maxc8 with WGK_ASM_CLAMP)
uint4 wgk_clamp8c(uint4 x, uint lo, uint hi)   // lo / hi: FP16 bits
{
    const half2 l = wgk_c2(lo), h = wgk_c2(hi);
    uint4 r;
    [unroll] for (uint i = 0; i < 4u; ++i) r[i] = wgk_opq(wgk_u(min(max(wgk_h2(x[i]), l), h)));
    return r;
}
uint4 wgk_maxc8(uint4 x, uint lo)   // lo: FP16 bits
{
    const half2 l = wgk_c2(lo);
    uint4 r;
    [unroll] for (uint i = 0; i < 4u; ++i) r[i] = wgk_opq(wgk_u(max(wgk_h2(x[i]), l)));
    return r;
}

// The same operations WITHOUT the opaque zero, for results whose only consumers cannot fuse with or regroup them: a
// matrix-product operand (WMMA), a store, a min / max. The FP16 operation itself is the same IEEE operation, so the
// bits are the same; the opaque zero only matters where a multiply feeds an add (fma contraction) or another multiply
// (regrouping). Every use is checked in the ISA: no v_pk_fma_f16 / v_fma_f16, every v_fma_mix_f32 is x * 1.0 + y.
uint4 wgk_add8n(uint4 a, uint4 b)
{
    uint4 r;
    [unroll] for (uint i = 0; i < 4u; ++i) r[i] = wgk_u(wgk_h2(a[i]) + wgk_h2(b[i]));
    return r;
}
uint4 wgk_sub8n(uint4 a, uint4 b)
{
    uint4 r;
    [unroll] for (uint i = 0; i < 4u; ++i) r[i] = wgk_u(wgk_h2(a[i]) - wgk_h2(b[i]));
    return r;
}
uint4 wgk_mul8n(uint4 a, uint4 b)
{
    uint4 r;
    [unroll] for (uint i = 0; i < 4u; ++i) r[i] = wgk_u(wgk_h2(a[i]) * wgk_h2(b[i]));
    return r;
}
uint4 wgk_clamp8cn(uint4 x, uint lo, uint hi)
{
    const half2 l = wgk_c2(lo), h = wgk_c2(hi);
    uint4 r;
    [unroll] for (uint i = 0; i < 4u; ++i) r[i] = wgk_u(min(max(wgk_h2(x[i]), l), h));
    return r;
}
uint4 wgk_maxc8n(uint4 x, uint lo)
{
    const half2 l = wgk_c2(lo);
    uint4 r;
    [unroll] for (uint i = 0; i < 4u; ++i) r[i] = wgk_u(max(wgk_h2(x[i]), l));
    return r;
}

// ---- the activation of spec 40 (exact; HIP wgk_act2 / wgk_act8) -----------------------------------------------------
//   t = min(max(x, -2), 2), p = f16(a - f16(b |t|)), q = f16(0.5 + f16(t p)), act(x) = f16(x q)
//   a = 0.412109375 (0x3698), b = 0.0810546875 (0x2d30); packed FP16, one rounding per operation.
#define WGK_ACT_A 0x3698u   // 0.412109375
#define WGK_ACT_B 0x2d30u   // 0.0810546875
#define WGK_H_M2 0xc000u    // -2
#define WGK_H_2 0x4000u     // 2
#define WGK_H_HALF 0x3800u  // 0.5
#define WGK_H_ONE 0x3c00u   // 1
uint wgk_act2(uint xu)
{
    const half2 x = wgk_h2(xu);
    const half2 t = min(max(x, wgk_c2(WGK_H_M2)), wgk_c2(WGK_H_2));
    const half2 at = abs(t);
    const half2 bt = wgk_h2(wgk_opq(wgk_u(wgk_c2(WGK_ACT_B) * at)));
    const half2 p = wgk_h2(wgk_opq(wgk_u(wgk_c2(WGK_ACT_A) - bt)));
    const half2 tp = wgk_h2(wgk_opq(wgk_u(t * p)));
    const half2 q = wgk_h2(wgk_opq(wgk_u(wgk_c2(WGK_H_HALF) + tp)));
    return wgk_opq(wgk_u(x * q));
}
uint4 wgk_act8(uint4 x)
{
    return uint4(wgk_act2(x.x), wgk_act2(x.y), wgk_act2(x.z), wgk_act2(x.w));
}
// act for a matrix-product operand (the MLP's hidden tile): the opaque zero only on the two products that feed an
// addition (b |t| -> a - b |t|, t p -> 0.5 + t p); a - b|t| feeds a multiply, 0.5 + t p the last multiply, whose result
// goes to the WMMA: nothing there to fuse or regroup.
uint wgk_act2n(uint xu)
{
    const half2 x = wgk_h2(xu);
    const half2 t = min(max(x, wgk_c2(WGK_H_M2)), wgk_c2(WGK_H_2));
    const half2 at = abs(t);
    const half2 bt = wgk_h2(wgk_opq(wgk_u(wgk_c2(WGK_ACT_B) * at)));
    const half2 p = wgk_c2(WGK_ACT_A) - bt;
    const half2 tp = wgk_h2(wgk_opq(wgk_u(t * p)));
    const half2 q = wgk_c2(WGK_H_HALF) + tp;
    return wgk_u(x * q);
}
uint4 wgk_act8n(uint4 x)
{
    return uint4(wgk_act2n(x.x), wgk_act2n(x.y), wgk_act2n(x.z), wgk_act2n(x.w));
}

// ---- softmax weights (HIP wgk_softmax_weights<false>: FP32 weights, the network's default) ---------------------------
// gg: one query's 4 fragments of 8 keys (lane = query, the clamped polynomial g of spec 41 applied). w = 2^g in FP32
// (v_exp_f32, every 2^g first, then the sum in fragment / element order), the sum over 64 keys = 32 per lane plus one
// lane exchange, inv = v_rcp_f32, A = f16(w inv) rounded ONCE from the exact product (HIP: v_fma_mixlo_f16 /
// v_fma_mixhi_f16 w * inv + 0): k_smx.hlsli (the mix form, or round to odd in the __rodd ports).
void wgk_softmax_weights(uint4 gg[4], out uint4 pa[4])
{
    float w[4][8];
    float sum = 0.0f;
    [unroll] for (uint st = 0; st < 4u; ++st)
        [unroll] for (uint j = 0; j < 8u; ++j) w[st][j] = wg_exp2(wgk_elf(gg[st], j));
    [unroll] for (uint s1 = 0; s1 < 4u; ++s1)
        [unroll] for (uint j = 0; j < 8u; ++j) sum = sum + w[s1][j];
    sum = sum + wgk_xor16(sum);
    const float inv = wg_rcp(sum);
    [unroll] for (uint s2 = 0; s2 < 4u; ++s2) pa[s2] = k_smx_weights8(w[s2], inv);
}

// ss + the sum of f16(x_c^2) over the 8 channels of x, in channel order, each addition in FP32 (HIP wgk_sumsq8: packed
// FP16 squares, then v_fma_mix_f32 x2_half * 1 + ss = the FP32 addition of the exact FP16 value).
float wgk_sumsq8(uint4 x, float ss)
{
    const uint4 q = wgk_mul8(x, x);
    [unroll] for (uint j = 0; j < 8u; ++j) ss = ss + wgk_elf(q, j);
    return ss;
}

// Mirror about the edge token, edge not repeated (spec 41): -1 -> 1, n -> n - 2; repeated for far positions (HIP
// wgk_mirror; branchless: for 0 <= p < n the general formula gives p, and n = 1 gives 0).
int wgk_mirror(int p, int n)
{
    const int per = max(2 * (n - 1), 1);
    int r = p % per;
    r = select(r < 0, r + per, r);
    return select(n == 1, 0, select(r < n, r, per - r));
}
