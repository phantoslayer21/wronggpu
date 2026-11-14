// SPDX-License-Identifier: MIT
// D3D12 ports of Model L's C = 32 Swin blocks (clean/l/kernels/l_swin.hip, l_swin.co; issue #89): swin_block.hlsl
// instantiated at C = 32, 2 heads, MLP 128 (the same template as the HIP kernels'), with HIP's MLP unit cut (HPU 8 = all
// 8 hidden tiles in one unit). q8(GELU): the frame's blocks (_enc_nc, _dec_nc) read it from the 16 KB arena table
// (SW_GELU 1, bit-identical to HIP's transcendental chain + pack, swin_block.hlsl; measured -31 / -33 us per frame at
// 1080p Quality); the groupshared table (SW_GELU 2) takes them to 112 VGPRs, over their bound of 96. l_swin_block_c32_nc
// keeps HIP's transcendental chain (SW_GELU 0).
//   l_swin_block_c32_nc      FP8 map in, FP8 out, mirror border (#117)
//   l_swin_block_c32_enc_nc  the encoder block: FP16 halo map [H + 4][W + 4][32] in (SW_SRC 1), FP8 out
//   l_swin_block_c32_dec_nc  the decoder block: FP8 map in with the token-grid mirror (SW_SRC 2), FP16 out (SW_OUT_F16)
//
// @wg-kernel l_swin_block_c32_nc co=l_swin.co block=256 wave=32 args=p,p,p,p,u,u,u,u defines=SW_C=32,SW_HEADS=2,SW_MLP=128,SW_ADD=0,SW_SAFE=1,SW_ECL=0,SW_HPU=8,SW_GELU=0,SW_ENTRY_OVERRIDE=l_swin_block_c32_nc
// @wg-kernel l_swin_block_c32_enc_nc co=l_swin.co block=256 wave=32 args=p,p,p,p,u,u,u,u defines=SW_C=32,SW_HEADS=2,SW_MLP=128,SW_ADD=0,SW_SAFE=1,SW_ECL=0,SW_HPU=8,SW_GELU=1,SW_SRC=1,SW_ENTRY_OVERRIDE=l_swin_block_c32_enc_nc
// @wg-kernel l_swin_block_c32_dec_nc co=l_swin.co block=256 wave=32 args=p,p,p,p,u,u,u,u defines=SW_C=32,SW_HEADS=2,SW_MLP=128,SW_ADD=0,SW_SAFE=1,SW_ECL=0,SW_HPU=8,SW_GELU=1,SW_SRC=2,SW_OUT_F16=1,SW_ENTRY_OVERRIDE=l_swin_block_c32_dec_nc
//   l_swin_block_c32_dec_coef12_nc  D3D12 only: the decoder block with l_coef_pack12 as its epilogue (SW_OUT_COEF12;
//                                   byte for byte the pair _dec_nc -> l_coef_pack12, which stays the fallback route);
//                                   the FP16 map is not written. Arguments: _dec_nc's (out unused), then A, C, latent,
//                                   dec0.coef, rw, rh
// @wg-kernel l_swin_block_c32_dec_coef12_nc co=- block=256 wave=32 args=p,p,p,p,u,u,u,u,p,p,p,p,u,u defines=SW_C=32,SW_HEADS=2,SW_MLP=128,SW_ADD=0,SW_SAFE=1,SW_ECL=0,SW_HPU=8,SW_GELU=1,SW_SRC=2,SW_OUT_F16=1,SW_OUT_COEF12=1,SW_ENTRY_OVERRIDE=l_swin_block_c32_dec_coef12_nc
#include "swin_block.hlsl"
