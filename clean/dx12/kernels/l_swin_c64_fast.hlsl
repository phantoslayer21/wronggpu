// SPDX-License-Identifier: MIT
// Model L's fast-mode middle Swin blocks at width 64 (runtime l_mode=fast, lever bit 8; native D3D12 only): the "_fast"
// twins of swin_block_c64_nc (enc1) and swin_block_c64_add_ex_nc (dec1, the fused expand), launched in place of them with
// the same arguments (runtime l_fast_alt). l_swin_c32_fast.hlsl's two-windows-per-wave body at C = 64 (the launch shape
// of the base kernels, swin_c64w2.hlsl: 128-lane workgroups, 8 windows per launched workgroup, 16 KB of staging) with
// its fast arithmetic: quick GELU (no table load) and the softmax without the maximum; everything else is the base
// kernels' M-form chain (the same packs, norms, soft cap, fused expand and output pack).
//
// @wg-kernel swin_block_c64_nc_fast co=- block=256 group=128 gscale=1 wave=32 args=p,p,p,p,u,u,u,u defines=SW_C=64,SW_HEADS=2,SW_MLP=256,SW_ADD=0,SW_SAFE=1,SW_ECL=0,WG_LD_A16=1,LF_NAME=swin_block_c64_nc_fast,LF_GELU=1
// @wg-kernel swin_block_c64_add_ex_nc_fast co=- block=256 group=128 gscale=1 wave=32 args=p,p,p,p,u,u,u,u,p defines=SW_C=64,SW_HEADS=2,SW_MLP=256,SW_ADD=1,SW_SAFE=1,SW_ECL=96,LF_NAME=swin_block_c64_add_ex_nc_fast,LF_GELU=1
#include "l_swin_c32_fast.hlsl"
