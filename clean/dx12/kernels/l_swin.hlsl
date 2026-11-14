// SPDX-License-Identifier: MIT
// D3D12 port of Model L's C = 32 Swin block (clean/l/kernels/l_swin.hip, l_swin_block_c32_nc of l_swin.co; issue #89):
// swin_block.hlsl instantiated at C = 32, 2 heads, MLP 128 (the same template as the HIP kernel's), with HIP's
// transcendental GELU chain (SW_GELU 0, no table) and HIP's MLP unit cut (HPU 8 = all 8 hidden tiles in one unit).
//
// @wg-kernel l_swin_block_c32_nc co=l_swin.co block=256 wave=32 args=p,p,p,p,u,u,u,u defines=SW_C=32,SW_HEADS=2,SW_MLP=128,SW_ADD=0,SW_SAFE=1,SW_ECL=0,SW_HPU=8,SW_GELU=0,SW_ENTRY_OVERRIDE=l_swin_block_c32_nc
#include "swin_block.hlsl"
