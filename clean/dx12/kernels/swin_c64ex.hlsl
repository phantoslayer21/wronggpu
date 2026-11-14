// SPDX-License-Identifier: MIT
// D3D12 port of swin_block_c64_add_ex_nc (clean/kernels/swin.hip WG_FUSED_KERNEL(swin_block_c64_add_ex, 64, 2, 256,
// true, 0, 96)): dec1's Swin block with its expand fused in, as a kernel of its own (dec1's map written out). Model M's
// D3D12 route runs dec1 fused with the pack12 decoder (d1p12/swin_block_c64_add_dec0*), so this kernel had no port;
// Model L's frame (issue #89) launches it for dec1 (default swin_fuse_stages dec1,dec2: the L decoder reads dec1's map).
// swin_block.hlsl instantiated at C = 64, 2 heads, MLP 256 (its C = 64 path, as swin_block_c64_nc) with the block input
// x + xadd (SW_ADD) and the fused expand from the 96-channel low-resolution map (SW_ECL 96), as
// swin_block_c96_add_ex_nc is at C = 96. 256 lanes, one window per wave; x = dec2's output (96 channels at half size),
// xadd = the skip map, aux unused.
//
// @wg-kernel swin_block_c64_add_ex_nc co=swin.co block=256 wave=32 args=p,p,p,p,u,u,u,u,p defines=SW_C=64,SW_HEADS=2,SW_MLP=256,SW_ADD=1,SW_SAFE=1,SW_ECL=96,SW_ENTRY_OVERRIDE=swin_block_c64_add_ex_nc
#include "swin_block.hlsl"
