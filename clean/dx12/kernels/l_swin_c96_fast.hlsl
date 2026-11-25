// SPDX-License-Identifier: MIT
// Model L's fast-mode width-96 Swin blocks (runtime l_mode=fast, lever bit 8192; native D3D12 only): the "_fast" twins of
// swin_block_c96_pm_rev_nc (enc2 with enc1's patch merge as its prologue, fused_swin.hlsl SW_PM) and
// swin_block_c96_add_ex_nc (dec2 with the fused expand), launched in place of them with the same arguments (runtime
// l_fast_alt). l_swin_c32_fast.hlsl's body in the base kernel's launch shape (256-lane workgroups, one window per wave,
// 8 windows per workgroup) with its fast arithmetic: the clamped-cubic GELU (no table load, no groupshared table: LDS
// 28 672 -> 24 576 bytes; no exponential) and the softmax without the maximum; everything else is the base kernel's
// M-form chain (the same packs, norms, soft cap, fused expand / patch merge and output pack). The patch merge (LF_PM) is
// fused_swin.hlsl's prologue (FZ_PMT 1, the fragments staged in the idle staging buffers, bit-identical merged tokens);
// its two trailing pointers (aux = the merge weights, aux2 = the merged map, not written) are appended to swin_block.hlsl's
// argument block (WG_ARGS_END redefined before it is included: x, xadd, out_p, w, H, W, ox, oy, aux, aux2 as fused_swin.hlsl).
//
// @wg-kernel swin_block_c96_pm_rev_nc_fast co=- block=256 wave=32 args=p,p,p,p,u,u,u,u,p,p defines=SW_C=96,SW_HEADS=4,SW_MLP=384,SW_ADD=0,SW_SAFE=1,SW_ECL=0,LF_PM=1,LF_NAME=swin_block_c96_pm_rev_nc_fast,LF_GELU=1
// @wg-kernel swin_block_c96_add_ex_nc_fast co=- block=256 wave=32 args=p,p,p,p,u,u,u,u,p defines=SW_C=96,SW_HEADS=4,SW_MLP=384,SW_ADD=1,SW_SAFE=1,SW_ECL=128,LF_NAME=swin_block_c96_add_ex_nc_fast,LF_GELU=1
// swin_block.hlsl's width-96 expand kernels declare their groupshared table's offset helper (sw_gl_offs8s) whatever the
// pattern; without the table (no digit 2 here) it is never called: its constants only have to exist.
#if defined(LF_PM) && LF_PM
#include "wg_platform.hlsli"
#undef WG_ARGS_END
#define WG_ARGS_END uint aux, aux2; };   // fused_swin.hlsl's two trailing pointers of the patch-merge kernels
#define LF_PM_ARGS 1
#endif
#define WG_GL_A 0.0f
#define WG_GL_B 0.0f
#include "l_swin_c32_fast.hlsl"
