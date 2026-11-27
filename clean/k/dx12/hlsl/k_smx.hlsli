// SPDX-License-Identifier: GPL-3.0-only
// Model K's softmax weight rounding, shared by the narrow (k_common.hlsli wgk_softmax_weights) and wide
// (k_block_wide.hlsl) stage ports and by the probe kernel k_mix_probe (kernels/k_mix_probe.hlsl). WrongGPU's own code.
//
// HIP (k_block.hip / k_block_wide.hip) rounds each weight A = f16(w inv) ONCE from the exact product
// (v_fma_mixlo_f16 / v_fma_mixhi_f16 w, inv, 0). With p = RN32(w inv) and its exact residual e = fma(w, inv, -p),
// w inv = p + e exactly (w and inv are positive, finite and normal in the softmax: w = 2^g with |g| <= ~14.4, inv the
// reciprocal of a sum of such w, so p >= ~3e-11 and e is exact; no zero, sign or FTZ case arises).
//   K_SMX 2 (default, the mix form): the sum p + e goes into the driver's round-to-nearest-even fragment conversion
//     (wg_to_f16_ags); the gfx1201 D3D12 compiler folds that addition into v_fma_mixlo_f16 / v_fma_mixhi_f16 p, 1.0, e,
//     whose single rounding of the exact p + e is RN16(w inv), HIP's bits (3 VALU per weight instead of 8). The fold
//     is the driver's choice: without it the FP32 sum is p and the conversion a second rounding (one FP16 ulp off
//     where p is an FP16 midpoint). Guards: build_dx12.py --isa checks every mix-form port's v_fma_mix*_f16 count
//     (clean/dx12/isa_checks.json); the native D3D12 runtime runs k_mix_probe on FP16-midpoint products at create and
//     takes the K_SMX 1 ports (suffix __rodd) when it fails (clean/k/runtime/runtime_k.cpp, k_mix_probe.h).
//   K_SMX 1 (the __rodd ports): p and e give the round-to-odd FP32 value (13 bits more than FP16, so rounding it to
//     FP16 is the single rounding of the exact product), then the same conversion. Truncation of w inv is p, or p - 1
//     ulp when e < 0; the inexact bit e != 0 makes it odd (the same bits as wg_round_odd for every finite p > 0).
//   K_SMX_MUT 1: negative control of the checks (the residual dropped: a second rounding); never shipped.
//   K_SMX_MUT4 1: negative control of the offsets (a second, residual-dropped conversion whose result is taken only
//     when the window offset is 4; the exact folded conversion is unchanged elsewhere; needs the stage kernels' P_ox);
//     never shipped.
//   K_SMX_RODD_MUT 1: negative control of the __rodd ports (the inexact bit dropped: truncation, then the conversion's
//     rounding, wrong where the product is an FP16 midpoint); never shipped.
#pragma once

#ifndef K_SMX
#define K_SMX 2
#endif
#ifndef K_SMX_MUT
#define K_SMX_MUT 0
#endif
#ifndef K_SMX_MUT4
#define K_SMX_MUT4 0
#endif
#ifndef K_SMX_RODD_MUT
#define K_SMX_RODD_MUT 0
#endif

uint k_smx_round_odd_pos(float p, float e)
{
#if K_SMX_RODD_MUT
    return asuint(p) - (e < 0.0f ? 1u : 0u);
#else
    return (asuint(p) - (e < 0.0f ? 1u : 0u)) | (e != 0.0f ? 1u : 0u);
#endif
}

// A = f16(w[j] inv) for one fragment's eight weights, packed two per dword (element 2 i in the low half of dword i).
uint4 k_smx_weights8(float w[8], float inv)
{
    v8f ro;
#if K_SMX >= 2 && K_SMX_MUT4
    v8f rb;   // negative control only: the residual-dropped conversion, taken at window offset 4 alone
#endif
    [unroll] for (uint j = 0; j < 8u; ++j) {
        const float p = w[j] * inv;
        const float e = mad(w[j], inv, -p);   // exact: w inv = p + e
#if K_SMX >= 2 && K_SMX_MUT
        ro.v[j] = p;
#elif K_SMX >= 2
        ro.v[j] = p + e;
#if K_SMX_MUT4
        rb.v[j] = p;
#endif
#else
        ro.v[j] = asfloat(k_smx_round_odd_pos(p, e));
#endif
    }
#if K_SMX >= 2 && K_SMX_MUT4
    // two conversions, the choice made on their results: the folded exact one stays as it is at every other offset
    const uint4 good = wg_to_f16_ags(ro), bad = wg_to_f16_ags(rb);
    return P_ox == 4 ? bad : good;
#else
    return wg_to_f16_ags(ro);
#endif
}
