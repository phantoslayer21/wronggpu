// SPDX-License-Identifier: MIT
// AMD D3D12 driver intrinsic channel: native matrix instructions (v_wmma_*) and matrix element conversions from an
// ordinary SM 6.6 compute shader, without any AMD library. Written for WrongGPU from the channel's interface facts:
//  - The driver watches a raw UAV bound at register space 0x7FFF0ADE (2147420894). An InterlockedCompareExchange on it
//    is not a memory operation but one instruction-encoding step: the "address" is a code word
//    (0x5 << 28) | (phase << 24) | (immediate << 8) | opcode, the compare and exchange values are two data dwords, and the
//    returned value is a result dword.
//  - Matrix multiply-accumulate (opcode 0x28): phase 0 loads the A, B and C fragments two dwords at a time
//    (immediate = channel | register << 4 | operand << 8; channel = dword pair & 1, register = dword pair >> 1;
//    operand A = 1, B = 2, C = 3), phase 1 executes (immediate = the format: 0x7 FP8 x FP8 -> FP32, 0x5 F16 -> FP32,
//    0x1 F16 -> F16 accumulate), phase 2 returns the 8 result dwords one per call (immediate = (i & 3) | (i >> 2) << 4).
//  - Matrix element conversion (opcode 0x32): phase 0 start (immediate 0, data 0), four phase-0 loads of the source
//    fragment (immediate = channel | register << 4), phase 1 (immediate = source type, data = destination type,
//    saturate flag), phase 2 returns the 8 dwords as above. Types: F32 accumulator 0x26, F16 accumulator 0x24, F16 B
//    0x14, FP8 (E4M3) B 0x1a. F32 -> F16 / FP8 round to nearest even (v_cvt_f16_f32 / v_cvt_pk_fp8_f32, the bits of HIP).
// The root signature binds a descriptor table with one raw UAV (any small buffer) at u0, space 0x7FFF0ADE
// (clean/runtime/device_dx12.cpp does it for every kernel).
// Rules on this driver (measured, clean/dx12/README.md): every call must be in wave-uniform control flow; keep these
// helpers and their neighbours branchless (branches next to the calls made the compiler spill to scratch); the compiler
// never moves loads across the calls, so issue all loads a group of multiplies needs before its first call; never pass
// the saturate flag (it adds per-value infinity checks, 40 % slower): clamp in the shader instead.
#pragma once

RWByteAddressBuffer wg_ags_uav : register(u0, space2147420894);

#define WG_AGS_MMA 0x28u
#define WG_AGS_CONVERT 0x32u
#define WG_AGS_FP8_F32 0x7u   // A, B FP8 E4M3 (2 dwords per lane), C / D FP32 (8 dwords)
#define WG_AGS_F16_F32 0x5u   // A, B F16 (4 dwords), C / D FP32
#define WG_AGS_F16_F16 0x1u   // A, B F16, C / D F16 (4 dwords)
#define WG_AGS_T_F32_ACC 0x26u
#define WG_AGS_T_F16_ACC 0x24u
#define WG_AGS_T_F16_B 0x14u
#define WG_AGS_T_FP8_B 0x1au

// Eight dwords of one lane's matrix fragment (unused trailing dwords are 0).
struct wg_frag { uint v[8]; };

uint wg_ags_call(uint op, uint phase, uint imm, uint d0, uint d1)
{
    uint code = 0x50000000u | ((phase & 3u) << 24) | ((imm & 0xffffu) << 8) | (op & 0xffu);
    uint r;
    wg_ags_uav.InterlockedCompareExchange(code, d0, d1, r);
    return r;
}

void wg_ags_load(uint op, uint operand, wg_frag f)
{
    [unroll] for (uint i = 0; i < 4; ++i)
        wg_ags_call(op, 0u, (i & 1u) | ((i >> 1) << 4) | (operand << 8), f.v[2 * i], f.v[2 * i + 1]);
}

wg_frag wg_ags_result(uint op)
{
    wg_frag r;
    [unroll] for (uint i = 0; i < 8; ++i) r.v[i] = wg_ags_call(op, 2u, (i & 3u) | ((i >> 2) << 4), 0u, 0u);
    return r;
}

// D = A x B + C (16 x 16 x 16, wave32 fragment layouts of gfx12's v_wmma_*_w32).
wg_frag wg_ags_mma(uint format, wg_frag a, wg_frag b, wg_frag c)
{
    wg_ags_load(WG_AGS_MMA, 1u, a);
    wg_ags_load(WG_AGS_MMA, 2u, b);
    wg_ags_load(WG_AGS_MMA, 3u, c);
    wg_ags_call(WG_AGS_MMA, 1u, format, 0u, 0u);
    return wg_ags_result(WG_AGS_MMA);
}

// Element conversion of one fragment (types WG_AGS_T_*; no saturation, see the rules above).
wg_frag wg_ags_convert(wg_frag m, uint from_type, uint to_type)
{
    wg_ags_call(WG_AGS_CONVERT, 0u, 0u, 0u, 0u);
    wg_ags_load(WG_AGS_CONVERT, 0u, m);
    wg_ags_call(WG_AGS_CONVERT, 1u, from_type, to_type, 0u);
    return wg_ags_result(WG_AGS_CONVERT);
}
