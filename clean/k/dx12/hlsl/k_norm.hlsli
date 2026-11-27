// SPDX-License-Identifier: GPL-3.0-only
// Model K block normalization reciprocal for the D3D12 ports: the HLSL counterpart of clean/k/kernels/k_norm.h
// (WrongGPU's own code). Needs k_common.hlsli.
//
// Spec 41: r = f16(1 / sqrt(s)) of the FP16 sum of squares s >= 0, as the HIP kernels compute it on gfx1201:
// kh_round(v_rsq_f32((float) s)) (round to nearest even), and r = 0 for s = 0 (s = inf gives rsq = 0 by itself).
// HIP's k_block.hip uses the SELECT form (a select after the rsq instead of a branch): the same bits; so does this one
// (branchless). rsqrt() is v_rsq_f32 on this driver (wg_math.hlsli); the FP32 -> FP16 rounding is wg_f2h (RNE).
#pragma once

uint wgk_norm_reciprocal(uint s16)   // FP16 bits in, FP16 bits out
{
    const uint r = wg_f2h(wg_rsq(f16tof32(s16)));
    return select((s16 & 0x7fffu) == 0u, 0u, r);
}
