// SPDX-License-Identifier: MIT
// Probe of Model K's softmax weight rounding on D3D12 (hlsl/k_smx.hlsli). WrongGPU's own code; D3D12 only (no HIP
// kernel: it is checked against the exact rounding of its inputs, clean/k/dx12/k_port_check.py family mixprobe, and the
// native D3D12 runtime runs it at create on the cases of clean/k/runtime/k_mix_probe.h).
// Lane i of the grid reads one record of 12 floats at byte 48 i: inv, w0 .. w7 (and 3 unused) and writes
// k_smx_weights8(w, inv) = f16(w_j inv) for j = 0..7, two per dword, at byte 16 i: the same helper, the same operations
// and the same driver conversion as every softmax port. The records are FP16-midpoint products with a nonzero residual
// of either sign, exact midpoints (ties to even), FP16-subnormal and ordinary products, so a second rounding through
// FP32 (the mix form without the driver's fold, or the K_SMX_MUT control) cannot pass.
// k_mix_probe: the default (mix) form; k_mix_probe__rodd: the round-to-odd form of the __rodd stage ports.
// Control flow is uniform (the driver channel's rule): every lane of every launched workgroup has a record.
// @wg-kernel k_mix_probe co=k_block.co block=32 wave=32 args=p,p
// @wg-kernel k_mix_probe__rodd co=k_block.co entry=k_mix_probe block=32 wave=32 args=p,p defines=K_SMX=1
#include "wg_platform.hlsli"

WG_ARGS_BEGIN
    uint P_in, P_out;
WG_ARGS_END

#include "k_smx.hlsli"

WG_KERNEL(k_mix_probe)
{
    WG_SETUP();
    const uint i = wg_block_id() * 32u + wg_lane_id();
    const uint4 a = wg_ld128(P_in, i * 48u);         // inv, w0, w1, w2
    const uint4 b = wg_ld128(P_in, i * 48u + 16u);   // w3 .. w6
    const uint4 c = wg_ld128(P_in, i * 48u + 32u);   // w7, unused
    float w[8];
    w[0] = asfloat(a.y);
    w[1] = asfloat(a.z);
    w[2] = asfloat(a.w);
    w[3] = asfloat(b.x);
    w[4] = asfloat(b.y);
    w[5] = asfloat(b.z);
    w[6] = asfloat(b.w);
    w[7] = asfloat(c.x);
    wg_st128(P_out, i * 16u, k_smx_weights8(w, asfloat(a.x)));
}
