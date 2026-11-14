// SPDX-License-Identifier: MIT
// D3D12 ports of Model L's C = 32 Swin blocks (clean/l/kernels/l_swin.hip, l_swin.co; issue #89): swin_block.hlsl
// instantiated at C = 32, 2 heads, MLP 128 (the same template as the HIP kernels'), with HIP's transcendental GELU chain
// (SW_GELU 0, no table) and HIP's MLP unit cut (HPU 8 = all 8 hidden tiles in one unit).
//   l_swin_block_c32_nc      FP8 map in, FP8 out, mirror border (#117)
//   l_swin_block_c32_enc_nc  the encoder block: FP16 halo map [H + 4][W + 4][32] in (SW_SRC 1), FP8 out
//   l_swin_block_c32_dec_nc  the decoder block: FP8 map in with the token-grid mirror (SW_SRC 2), FP16 out (SW_OUT_F16)
//
// @wg-kernel l_swin_block_c32_nc co=l_swin.co block=256 wave=32 args=p,p,p,p,u,u,u,u defines=SW_C=32,SW_HEADS=2,SW_MLP=128,SW_ADD=0,SW_SAFE=1,SW_ECL=0,SW_HPU=8,SW_GELU=0,SW_ENTRY_OVERRIDE=l_swin_block_c32_nc
// @wg-kernel l_swin_block_c32_enc_nc co=l_swin.co block=256 wave=32 args=p,p,p,p,u,u,u,u defines=SW_C=32,SW_HEADS=2,SW_MLP=128,SW_ADD=0,SW_SAFE=1,SW_ECL=0,SW_HPU=8,SW_GELU=0,SW_SRC=1,SW_ENTRY_OVERRIDE=l_swin_block_c32_enc_nc
// @wg-kernel l_swin_block_c32_dec_nc co=l_swin.co block=256 wave=32 args=p,p,p,p,u,u,u,u defines=SW_C=32,SW_HEADS=2,SW_MLP=128,SW_ADD=0,SW_SAFE=1,SW_ECL=0,SW_HPU=8,SW_GELU=0,SW_SRC=2,SW_OUT_F16=1,SW_ENTRY_OVERRIDE=l_swin_block_c32_dec_nc
#include "swin_block.hlsl"
