// SPDX-License-Identifier: MIT
// Model L's fast-mode decoder blocks (runtime option l_mode=fast, env WG_L_MODE=fast; native D3D12 only): the M-form
// chain (SW_HYB 0, the WMMA / FP8 kernels of Model M's Swin stages) with Model L's decoder skip S. l_swin.hlsl's
// l_swin_block_c32_dec_nc / _dec_coef12_nc read the decoder's projection map y and nothing else (SW_ADD 0): the M-form
// route of the earlier Model L releases had no skip. Model L's decoder block adds the encoder block's map x3 (FP8) to
// the block input, as the hybrid twins do (SW_ADD 1, l_swin_block_c32_dec_hyb: x by the token-grid mirror, S by the
// fine mirror, the members' input x + S); here in the M-form arithmetic (swin_block.hlsl: the sum of the two decoded FP8
// values in float32, no FP16 rounding step; the rest of the chain unchanged), so the fast route keeps the model's
// structure with Model M's cost per block.
//   l_swin_block_c32_dec_add_nc         _dec_nc with the skip (SW_ADD 1): FP16 map y2 out (the fallback route: l_coef /
//                                       l_coef_pack12 as their own launches)
//   l_swin_block_c32_dec_coef12_add_nc  _dec_coef12_nc with the skip: the pack12 coefficient store as the epilogue
// Arguments as the hybrid twins' (and _dec_nc's): y, S (the encoder block's map l_x3), out (unused with the store), the
// block's blob (Model M's packer on Model L's values, not the hybrid values), H, W, ox, oy, then the store's A, C,
// latent, dec0.coef, rw, rh. The defines are the _nc kernels' with the cheaper GELU table offsets for every tile (SW_GELU 3,
// pattern 13333; the same bytes as SW_GELU 1), the staged softmax and, for the store variant, the head loop in pairs
// (SW_HEAD_PAIRS: 111 -> 96 VGPRs, the same operations in the same order: the skip's second map needs the registers).
// WG_LD_A16 (16-byte aligned loads, the hybrid twins' form: every wg_ld128 base + offset here is a multiple of 16) is on.
//
// @wg-kernel l_swin_block_c32_dec_add_nc co=- block=256 wave=32 args=p,p,p,p,u,u,u,u defines=SW_C=32,SW_HEADS=2,SW_MLP=128,SW_ADD=1,SW_SAFE=1,SW_ECL=0,SW_HPU=8,SW_GELU=3,SW_GELU_PAT=13333,SW_SRC=2,SW_OUT_F16=1,WG_LD_A16=1,SW_ENTRY_OVERRIDE=l_swin_block_c32_dec_add_nc
// @wg-kernel l_swin_block_c32_dec_coef12_add_nc co=- block=256 wave=32 args=p,p,p,p,u,u,u,u,p,p,p,p,u,u defines=SW_C=32,SW_HEADS=2,SW_MLP=128,SW_ADD=1,SW_SAFE=1,SW_ECL=0,SW_HPU=8,SW_GELU=3,SW_GELU_PAT=13333,SW_SRC=2,SW_OUT_F16=1,SW_OUT_COEF12=1,COEF_CVT_HW=1,WG_LD_A16=1,SW_ENTRY_OVERRIDE=l_swin_block_c32_dec_coef12_add_nc,SW_SMAX_ADDST=1,SW_SMAX_STAGED=1,SW_HEAD_PAIRS=1
//
// The "_fast" twin of the encoder block with the linears (l_swin_block_c32_enc_lin_nc_fast; launched in place of
// l_swin_block_c32_enc_lin_nc with the same arguments, runtime l_fast_alt / lever bit 4; the base kernel's DXIL is not touched):
// the two input linears of its prologue (swin_block.hlsl SW_SRC 3, TT_LINF32) keep their results in FP32: x1 = FP8(relu(b + W q)),
// x2 = b2 + W2 x1 as the block input, without the two FP16 roundings and the widening back (16 conversions and 16 unpacks per
// lane; RGA: 997 -> 893 VALU instructions, 1732 -> 1626 in all, 96 VGPRs, no spill). The FP32 value differs from the
// original's FP16 rounding (which the hybrid route keeps) by at most half an FP16 ulp; WG_LD_A16 is not set (RGA: the 16-byte loads of this
// kernel are already single instructions, the mask only adds operations).
//
// @wg-kernel l_swin_block_c32_enc_lin_nc_fast co=- block=256 wave=32 args=p,p,p,p,u,u,u,u,p,p defines=SW_C=32,SW_HEADS=2,SW_MLP=128,SW_ADD=0,SW_SAFE=1,SW_ECL=0,SW_HPU=8,SW_GELU=3,SW_GELU_PAT=13333,SW_SRC=3,SW_ENTRY_OVERRIDE=l_swin_block_c32_enc_lin_nc_fast,WG_REV_ORDER=1,SW_HEAD_PAIRS=1,SW_SMAX_STAGED=1,SW_NORM1_EARLY=1,TT_LINF32=1
#include "swin_block.hlsl"
