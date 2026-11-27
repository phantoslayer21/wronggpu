// SPDX-License-Identifier: GPL-3.0-only
// Model K's stage kernel arguments for the D3D12 ports (WrongGPU's own code): KBlkParams of clean/k/kernels/k_block.hip
// as the manifest token s184:p19.b32 lays it out (clean/runtime/struct_args.h). Include before WG_ARGS_BEGIN and put
// KBLK_PARAMS_DECL inside the argument block; the kernel's other headers (k_block.hlsli, k_common.hlsli) come after
// WG_ARGS_END.
#pragma once

// KBlkParams (clean/k/kernels/k_block.hip, 184 bytes) as the manifest token s184:p19.b32 lays it out: 19 pointers (one
// root constant each: the arena byte offset, 0 = null), then 8 ints. Declare inside WG_ARGS_BEGIN / WG_ARGS_END.
#define KBLK_PARAMS_DECL                                                                                          \
    uint P_x, P_skip, P_y, P_merged, P_head, P_g1, P_g2, P_wqkv, P_mix, P_wo, P_bo, P_w1, P_b1, P_w2, P_b2, P_wa,  \
        P_ba, P_wz, P_bz;                                                                                         \
    int P_w, P_h, P_ox, P_oy, P_nwx, P_nwy, P_cw, P_ch;
