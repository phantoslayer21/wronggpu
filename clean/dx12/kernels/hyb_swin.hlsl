// SPDX-License-Identifier: MIT
// The HYBRID form's Swin blocks of the shared stages (WG_L_FORM = hybrid; SW_HYB in swin_block.hlsl): swin_block.hlsl
// instantiated with SW_HYB=1 at Model M's stage shapes, one window per wave in 256-lane workgroups (HIP's shape; grid =
// ceil(windows / 8)), the plain schedule (swin_block.hlsl sets the hybrid knob set), unfused: the merges and expands are
// their own kernels (hyb_gemm.hlsl).
//   swin_block_c64_hyb      enc1          swin_block_c64_add_hyb   dec1 (x = E, the expanded map; xadd = enc1's output)
//   swin_block_c96_hyb      enc2          swin_block_c96_add_hyb   dec2 (x = E; xadd = enc2's output)
//   swin_block_c128_hyb     b0 .. b5 (one launch per block)
// Arguments as the M-form blocks: x, xadd (null for the encoder / bottleneck blocks), out, w (the stage's hybrid blob,
// clean/l/runner/l_hyb_pack.py and its C++ packer), H, W, ox, oy. The decoder blocks read E with the coarse mirror
// (the table's shared_dec_border) and the skip map with the fine mirror; the member input is RN16(E + S).
// HIP twins: the same names in swin_hyb.co (clean/kernels/swin_hyb.hip).
//
// @wg-kernel swin_block_c64_hyb co=- block=256 wave=32 args=p,p,p,p,u,u,u,u defines=SW_C=64,SW_HEADS=2,SW_MLP=256,SW_ADD=0,SW_SAFE=1,SW_ECL=0,SW_HYB=1,SW_ENTRY_OVERRIDE=swin_block_c64_hyb
// @wg-kernel swin_block_c64_add_hyb co=- block=256 wave=32 args=p,p,p,p,u,u,u,u defines=SW_C=64,SW_HEADS=2,SW_MLP=256,SW_ADD=1,SW_SAFE=1,SW_ECL=0,SW_HYB=1,SW_ENTRY_OVERRIDE=swin_block_c64_add_hyb
// @wg-kernel swin_block_c96_hyb co=- block=256 wave=32 args=p,p,p,p,u,u,u,u defines=SW_C=96,SW_HEADS=4,SW_MLP=384,SW_ADD=0,SW_SAFE=1,SW_ECL=0,SW_HYB=1,SW_ENTRY_OVERRIDE=swin_block_c96_hyb
// @wg-kernel swin_block_c96_add_hyb co=- block=256 wave=32 args=p,p,p,p,u,u,u,u defines=SW_C=96,SW_HEADS=4,SW_MLP=384,SW_ADD=1,SW_SAFE=1,SW_ECL=0,SW_HYB=1,SW_ENTRY_OVERRIDE=swin_block_c96_add_hyb
// @wg-kernel swin_block_c128_hyb co=- block=256 wave=32 args=p,p,p,p,u,u,u,u defines=SW_C=128,SW_HEADS=4,SW_MLP=512,SW_ADD=0,SW_SAFE=1,SW_ECL=0,SW_HYB=1,SW_ENTRY_OVERRIDE=swin_block_c128_hyb
//   swin_block_c64_hyb_scanmut / _c128_hyb_scanmut  the fused-square scan's mutants (probe=1, outside the manifest; see
//                          l_swin.hlsl l_swin_block_c32_enc_hyb_scanmut): the select-free norm trees KC 4 and 8
// @wg-kernel swin_block_c64_hyb_scanmut co=- probe=1 block=256 wave=32 args=p,p,p,p,u,u,u,u defines=SW_C=64,SW_HEADS=2,SW_MLP=256,SW_ADD=0,SW_SAFE=1,SW_ECL=0,SW_HYB=1,HYB_SCAN_MUTANT=1,SW_ENTRY_OVERRIDE=swin_block_c64_hyb_scanmut
// @wg-kernel swin_block_c64_add_hyb_scanmut co=- probe=1 block=256 wave=32 args=p,p,p,p,u,u,u,u defines=SW_C=64,SW_HEADS=2,SW_MLP=256,SW_ADD=1,SW_SAFE=1,SW_ECL=0,SW_HYB=1,HYB_SCAN_MUTANT=1,SW_ENTRY_OVERRIDE=swin_block_c64_add_hyb_scanmut
//   swin_block_c96_hyb_scanmut / _c96_add_hyb_scanmut  the same for KC 6, as SUBSTITUTE mutants: the KC 6 tree keeps its
//                          per-lane selects, so no square ever feeds an add (RGA: every square's first consumer is a
//                          v_cndmask_b16 or v_permlanex16_b32) and the plain barrier-free build forms no multiply-add;
//                          with HYB_SCAN_MUTANT, hyb_block.hlsli also bypasses the selects so that a square does feed
//                          an add, and the scan must find it (wrong values by design; never in a model build)
// @wg-kernel swin_block_c96_hyb_scanmut co=- probe=1 block=256 wave=32 args=p,p,p,p,u,u,u,u defines=SW_C=96,SW_HEADS=4,SW_MLP=384,SW_ADD=0,SW_SAFE=1,SW_ECL=0,SW_HYB=1,HYB_SCAN_MUTANT=1,SW_ENTRY_OVERRIDE=swin_block_c96_hyb_scanmut
// @wg-kernel swin_block_c96_add_hyb_scanmut co=- probe=1 block=256 wave=32 args=p,p,p,p,u,u,u,u defines=SW_C=96,SW_HEADS=4,SW_MLP=384,SW_ADD=1,SW_SAFE=1,SW_ECL=0,SW_HYB=1,HYB_SCAN_MUTANT=1,SW_ENTRY_OVERRIDE=swin_block_c96_add_hyb_scanmut
// @wg-kernel swin_block_c128_hyb_scanmut co=- probe=1 block=256 wave=32 args=p,p,p,p,u,u,u,u defines=SW_C=128,SW_HEADS=4,SW_MLP=512,SW_ADD=0,SW_SAFE=1,SW_ECL=0,SW_HYB=1,HYB_SCAN_MUTANT=1,SW_ENTRY_OVERRIDE=swin_block_c128_hyb_scanmut
#include "swin_block.hlsl"
