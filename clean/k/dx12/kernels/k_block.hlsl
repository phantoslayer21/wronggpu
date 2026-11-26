// SPDX-License-Identifier: MIT
// D3D12 ports of model K's narrow network stage kernels (clean/k/kernels/k_block.hip; HIP build: k_block.co). WrongGPU's
// own code. The k_block template (hlsl/k_block.hlsli) with the instantiations of k_block.hip, FP32 accumulation (the
// network's default "fused" plan, runtime_k.cpp KRuntime::network):
//   k_stage_e2          k_block<64, 2, softmax, MAP, 64, 96>                  G = 1, merge 64 -> 96
//   k_stage2_e0         k_block<32, 2, fixed, EMBED, 16, 64, G = 2>           embedding 16 -> 32 on load, merge
//   k_stage2_e1         k_block<64, 2, fixed, MAP, 64, 64, G = 2>             merge
//   k_stage_d2          k_block<64, 2, softmax, EXPAND, 96>                   G = 1, expand + skip
//   k_stage2_d1         k_block<64, 2, softmax, EXPAND, 64, G = 2, XG>        x through P.y
//   k_stage2_d0_hp      k_block<32, 2, fixed, EXPAND, 64, HEAD, G = 2, HIDDEN, PIX>   (k_head=compact, the default)
//   k_stage2_d0_hidden  k_block<32, 2, fixed, EXPAND, 64, HEAD, G = 2, HIDDEN>        (k_head=full)
// and the one-window (G = 1) forms of e0 / e1 / d1 that HIP's k_block.co also has (K_STAGES: k_stage_e0 / _e1 / _d1,
// the same per-window maths and outputs as the G = 2 forms; d1 without XG: x stays in LDS):
//   k_stage_e0          k_block<32, 2, fixed, EMBED, 16, 64>                  G = 1
//   k_stage_e1          k_block<64, 2, fixed, MAP, 64, 64>                    G = 1
//   k_stage_d1          k_block<64, 2, softmax, EXPAND, 64>                   G = 1
// The FP16-accumulating forms (k_stageh_*) and the FP8 forms are not ported: the exact path only.
// Basis: our HIP kernels (spec 41 block, spec 44 windows, spec 43 6.8 hidden history, spec 40 table); exactness and the
// forms that differ (bit-identical) are listed in hlsl/k_block.hlsli and hlsl/k_common.hlsli. Arguments as HIP:
// KBlkParams by value (token s184:p19.b32, 27 constants), the d0 stages also take the hidden history pointer.
// The narrow two-window ports and the one-window e2 / d2 / d1 take a FP32 bias plane (KB_BIAS32, with the lane-offset weight
// bases of KB_BASE_FORM), and d0_hp takes a separate compact head weight pointer.
// Launch as HIP: 128 lanes (4 waves of 32) per workgroup, one workgroup per window (G = 1) or per two windows (G = 2).
//
// @wg-kernel k_stage_e2 co=k_block.co block=128 wave=32 args=s184:p19.b32,p defines=KB_BIAS32=1,KB_BASE_FORM=1
// @wg-kernel k_stage2_e0 co=k_block.co block=128 wave=32 args=s184:p19.b32,p defines=KB_BIAS32=1,KB_BASE_FORM=1
// @wg-kernel k_stage2_e1 co=k_block.co block=128 wave=32 args=s184:p19.b32,p defines=KB_BIAS32=1,KB_BASE_FORM=1
// @wg-kernel k_stage_d2 co=k_block.co block=128 wave=32 args=s184:p19.b32,p defines=KB_BIAS32=1,KB_BASE_FORM=1
// KB_X1LDS / KB_W2LATE / KB_W2SEQ (hlsl/k_block.hlsli): x1 waits in LDS during the MLP and each W2 tile is loaded just before its
// own products: 231 -> 191 VGPRs (RGA gfx1201, no spill), under the 192 of four workgroups per CU; remove the three defines to undo.
// @wg-kernel k_stage2_d1 co=k_block.co block=128 wave=32 args=s184:p19.b32,p defines=KB_BIAS32=1,KB_BASE_FORM=1,KB_X1LDS=1,KB_W2LATE=1,KB_W2SEQ=1
// @wg-kernel k_stage_e0 co=k_block.co block=128 wave=32 args=s184:p19.b32 defines=KB_LDS_HALFS=6144
// @wg-kernel k_stage_e1 co=k_block.co block=128 wave=32 args=s184:p19.b32 defines=KB_LDS_HALFS=4096
// @wg-kernel k_stage_d1 co=k_block.co block=128 wave=32 args=s184:p19.b32,p defines=KB_BIAS32=1,KB_BASE_FORM=1
// @wg-kernel k_stage_d1__rodd co=k_block.co entry=k_stage_d1 block=128 wave=32 args=s184:p19.b32,p defines=K_SMX=1,KB_BIAS32=1,KB_BASE_FORM=1
// @wg-kernel k_stage2_d0_hp co=k_block.co block=128 wave=32 args=s184:p19.b32,p,p,p defines=KB_HIDDEN_ARG=1,KB_COMPACT_HEAD_ARG=1,KB_BIAS32=1,KB_BASE_FORM=1
// @wg-kernel k_stage2_d0_hidden co=k_block.co block=128 wave=32 args=s184:p19.b32,p defines=KB_HIDDEN_ARG=1
// The __rodd forms: the same stages with the round-to-odd softmax weights (hlsl/k_smx.hlsli K_SMX 1), launched by the
// native D3D12 runtime when the mix-form probe (kernels/k_mix_probe.hlsl) fails on the adapter's driver.
// @wg-kernel k_stage_e2__rodd co=k_block.co entry=k_stage_e2 block=128 wave=32 args=s184:p19.b32,p defines=K_SMX=1,KB_BIAS32=1,KB_BASE_FORM=1
// @wg-kernel k_stage_d2__rodd co=k_block.co entry=k_stage_d2 block=128 wave=32 args=s184:p19.b32,p defines=K_SMX=1,KB_BIAS32=1,KB_BASE_FORM=1
// @wg-kernel k_stage2_d1__rodd co=k_block.co entry=k_stage2_d1 block=128 wave=32 args=s184:p19.b32,p defines=K_SMX=1,KB_BIAS32=1,KB_BASE_FORM=1,KB_X1LDS=1,KB_W2LATE=1,KB_W2SEQ=1
#ifndef KB_LD_A16
#define KB_LD_A16 1   // 16-byte aligned loads: the whole address masked (hlsl/k_common.hlsli; 2 = scalar base only: loses RADV's b128 loads)
#endif
#include "wg_platform.hlsli"
#include "k_blkparams.hlsli"

WG_ARGS_BEGIN
    KBLK_PARAMS_DECL
#if KB_HIDDEN_ARG
    uint hidden_p;   // the hidden history Hh written by the d0 epilogue
#endif
#if KB_BIAS32
    uint P_b32;
#endif
#if KB_COMPACT_HEAD_ARG
    uint P_compact_head;
#endif
WG_ARGS_END
#include "k_block.hlsli"

#if KB_HIDDEN_ARG
WG_KERNEL(k_stage2_d0_hp)
{
    WG_SETUP();
#if KB_COMPACT_HEAD_ARG
    k_block<32, 2, KMIX_FIXED, KIN_EXPAND, 64, 0, true, 2, true, true, false>(hidden_p, wg_block_id(), P_b32,
                                                                           P_compact_head);
#else
    k_block<32, 2, KMIX_FIXED, KIN_EXPAND, 64, 0, true, 2, true, true, false>(hidden_p, wg_block_id());
#endif
}
#if !KB_COMPACT_HEAD_ARG
WG_KERNEL(k_stage2_d0_hidden)
{
    WG_SETUP();
    k_block<32, 2, KMIX_FIXED, KIN_EXPAND, 64, 0, true, 2, true, false, false>(hidden_p, wg_block_id());
}
#endif
#else
WG_KERNEL(k_stage_e2)
{
    WG_SETUP();
#if KB_BIAS32
    k_block<64, 2, KMIX_SOFTMAX, KIN_MAP, 64, 96, false, 1, false, false, false>(0u, wg_block_id(), P_b32);
#else
    k_block<64, 2, KMIX_SOFTMAX, KIN_MAP, 64, 96, false, 1, false, false, false>(0u, wg_block_id());
#endif
}
WG_KERNEL(k_stage2_e0)
{
    WG_SETUP();
#if KB_BIAS32
    k_block<32, 2, KMIX_FIXED, KIN_EMBED, 16, 64, false, 2, false, false, false>(0u, wg_block_id(), P_b32);
#else
    k_block<32, 2, KMIX_FIXED, KIN_EMBED, 16, 64, false, 2, false, false, false>(0u, wg_block_id());
#endif
}
WG_KERNEL(k_stage2_e1)
{
    WG_SETUP();
#if KB_BIAS32
    k_block<64, 2, KMIX_FIXED, KIN_MAP, 64, 64, false, 2, false, false, false>(0u, wg_block_id(), P_b32);
#else
    k_block<64, 2, KMIX_FIXED, KIN_MAP, 64, 64, false, 2, false, false, false>(0u, wg_block_id());
#endif
}
WG_KERNEL(k_stage_d2)
{
    WG_SETUP();
#if KB_BIAS32
    k_block<64, 2, KMIX_SOFTMAX, KIN_EXPAND, 96, 0, false, 1, false, false, false>(0u, wg_block_id(), P_b32);
#else
    k_block<64, 2, KMIX_SOFTMAX, KIN_EXPAND, 96, 0, false, 1, false, false, false>(0u, wg_block_id());
#endif
}
WG_KERNEL(k_stage2_d1)
{
    WG_SETUP();
#if KB_BIAS32
    k_block<64, 2, KMIX_SOFTMAX, KIN_EXPAND, 64, 0, false, 2, false, false, true>(0u, wg_block_id(), P_b32);
#else
    k_block<64, 2, KMIX_SOFTMAX, KIN_EXPAND, 64, 0, false, 2, false, false, true>(0u, wg_block_id());
#endif
}
WG_KERNEL(k_stage_e0)
{
    WG_SETUP();
    k_block<32, 2, KMIX_FIXED, KIN_EMBED, 16, 64, false, 1, false, false, false>(0u, wg_block_id());
}
WG_KERNEL(k_stage_e1)
{
    WG_SETUP();
    k_block<64, 2, KMIX_FIXED, KIN_MAP, 64, 64, false, 1, false, false, false>(0u, wg_block_id());
}
WG_KERNEL(k_stage_d1)
{
    WG_SETUP();
#if KB_BIAS32
    k_block<64, 2, KMIX_SOFTMAX, KIN_EXPAND, 64, 0, false, 1, false, false, false>(0u, wg_block_id(), P_b32);
#else
    k_block<64, 2, KMIX_SOFTMAX, KIN_EXPAND, 64, 0, false, 1, false, false, false>(0u, wg_block_id());
#endif
}
#endif
