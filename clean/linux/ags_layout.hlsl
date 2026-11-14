// SPDX-License-Identifier: MIT
// WrongGPU's driver-channel helpers (clean/dx12/hlsl/wg_ags.hlsli) on fixed register dwords, for the Linux test of
// the D3D12 translation layer (clean/linux/run_wine_tests.py, test ags-layout). Under Proton the channel is translated
// by vkd3d-proton's dxil-spirv; this checks that the results are the gfx12 register semantics the kernels rely on.
// Buffer io (u1): inputs at 0 (A16, B16, C, A8, B8, all FP8 codes: 1 KiB each, 32 lanes x 8 dwords), outputs from 6 KiB
// (one 1 KiB slot per case; slot 6 is the wave64 case).
#include "wg_ags.hlsli"
RWByteAddressBuffer io : register(u1);

wg_frag load_frag(uint base, uint lane, uint n)
{
    wg_frag f;
    [unroll] for (uint i = 0; i < 8; ++i) f.v[i] = i < n ? io.Load(base + (lane * 8 + i) * 4) : 0u;
    return f;
}

void store_frag(uint slot, uint lane, wg_frag f)
{
    [unroll] for (uint i = 0; i < 8; ++i) io.Store(6 * 1024 + slot * 1024 + (lane * 8 + i) * 4, f.v[i]);
}

[WaveSize(32)]
[numthreads(32, 1, 1)]
void main32(uint lane : SV_GroupThreadID)
{
    wg_frag a16 = load_frag(0, lane, 4), b16 = load_frag(1024, lane, 4), c = load_frag(2048, lane, 8);
    wg_frag a8 = load_frag(3072, lane, 2), b8 = load_frag(4096, lane, 2);
    store_frag(0, lane, wg_ags_mma(WG_AGS_F16_F32, a16, b16, c));
    store_frag(1, lane, wg_ags_mma(WG_AGS_FP8_F32, a8, b8, c));
    store_frag(2, lane, wg_ags_convert(c, WG_AGS_T_F32_ACC, WG_AGS_T_F16_ACC));
    store_frag(3, lane, wg_ags_convert(c, WG_AGS_T_F32_ACC, WG_AGS_T_FP8_B));
    wg_frag h;
    // FloatConversion, F16 near: opcode 0x1c with the arguments of f2h_hw in clean/rr_f/kernels/rr_fast.hlsli
    [unroll] for (uint i = 0; i < 8; ++i) h.v[i] = wg_ags_call(0x1cu, 0u, 1u, c.v[i], 0u);
    store_frag(4, lane, h);
    store_frag(5, lane, wg_ags_convert(load_frag(5 * 1024, lane, 2), WG_AGS_T_FP8_B, WG_AGS_T_F32_ACC));
}

// post.hlsl's cvt3_f16 in a wave64 kernel: lane l converts C values 3l .. 3l+2 (mod 256).
[WaveSize(64)]
[numthreads(64, 1, 1)]
void main64(uint lane : SV_GroupThreadID)
{
    wg_frag f;
    [unroll] for (uint i = 0; i < 3; ++i) f.v[i] = io.Load(2048 + ((3 * lane + i) % 256) * 4);
    [unroll] for (uint i = 3; i < 8; ++i) f.v[i] = 0u;
    wg_frag r = wg_ags_convert(f, WG_AGS_T_F32_ACC, WG_AGS_T_F16_ACC);
    io.Store(6 * 1024 + 6 * 1024 + (2 * lane) * 4, r.v[0]);
    io.Store(6 * 1024 + 6 * 1024 + (2 * lane + 1) * 4, r.v[1] & 0xffffu);
}
