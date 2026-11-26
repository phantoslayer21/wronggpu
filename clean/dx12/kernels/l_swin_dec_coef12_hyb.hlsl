// SPDX-License-Identifier: MIT
// Dedicated compilation unit for Model L's hybrid C=32 decoder block with one-tile W2 (SW_COEF_ROWS=1),
// isolated from swin_block.hlsl so unrelated Swin/hybrid kernels keep identical source digests.
// @wg-kernel l_swin_block_c32_dec_coef12_hyb co=- block=256 wave=32 args=p,p,p,p,u,u,u,u,p,p,p,p,u,u defines=SW_C=32,SW_HEADS=2,SW_MLP=128,SW_ADD=1,SW_SAFE=1,SW_ECL=0,SW_HPU=8,SW_GELU=0,SW_OUT_F16=1,SW_OUT_COEF12=1,COEF_CVT_HW=1,SW_HYB=1,SW_HEAD_UNROLL=1,WG_LD_A16=1,SW_COEF_ROWS=1,SW_ENTRY_OVERRIDE=l_swin_block_c32_dec_coef12_hyb,WG_REV_ORDER=1
// @wg-kernel l_swin_block_c32_dec_coef12_hyb_scanmut co=- probe=1 block=256 wave=32 args=p,p,p,p,u,u,u,u,p,p,p,p,u,u defines=SW_C=32,SW_HEADS=2,SW_MLP=128,SW_ADD=1,SW_SAFE=1,SW_ECL=0,SW_HPU=8,SW_GELU=0,SW_OUT_F16=1,SW_OUT_COEF12=1,COEF_CVT_HW=1,SW_HYB=1,SW_HEAD_UNROLL=1,WG_LD_A16=1,HYB_SCAN_MUTANT=1,SW_COEF_ROWS=1,SW_ENTRY_OVERRIDE=l_swin_block_c32_dec_coef12_hyb_scanmut
#include "l_swin_coef_rows.hlsli"
