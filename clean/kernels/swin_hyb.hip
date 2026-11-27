// SPDX-License-Identifier: GPL-3.0-only
// The HYBRID form's shared-stage kernels (WG_L_FORM = hybrid): the HIP twins of the D3D12 ports in
// clean/dx12/kernels/hyb_swin.hlsl (blocks, SW_HYB=1) and hyb_gemm.hlsl (merges, expands), the same bytes. swin.hip's
// templates with HYB = true (swin_block: one window per wave, 256 lanes; patch_merge / expand: WG_GEMM_THREADS lanes,
// WG_NT token groups). Its own code object (swin_hyb.co), so swin.co stays as it is.
//   swin_block_c64_hyb, swin_block_c64_add_hyb, swin_block_c96_hyb, swin_block_c96_add_hyb, swin_block_c128_hyb
//     (x, xadd | null, out, w, H, W, ox, oy); grid ceil(windows / 8) x 256
//   patch_merge_c64_hyb, patch_merge_c96_hyb (s, out, w, H, W); expand_c96_hyb, expand_c128_hyb (y, e, w, Hl, Wl);
//     grid ceil(tokens / (8 x WG_NT x 16)) x 256
#define WG_SWIN_NO_EXPORTS 1
#include "swin.hip"

static_assert(WG_GEMM_THREADS == 256 && WG_NT == 2, "swin_hyb: the D3D12 twins' merge / expand geometry");

#define WG_HYB_BLOCK(name, C, HEADS, MLP, ADD, MINW)                                                                \
  WG_KERNEL_NW(name, 256, MINW)(const u8* x, const u8* xadd, u8* out, const u8* w, int H, int W, int ox, int oy) { \
    swin_block<C, HEADS, MLP, 1, ADD, WG_MLP_GROUP, true, 256, 0, 0, SwinStoreEpi, false, SwinFp8Src, true>(       \
        x, xadd, out, w, H, W, ox, oy, nullptr);                                                                    \
  }
WG_HYB_BLOCK(swin_block_c64_hyb, 64, 2, 256, false, WG_MINW_C64)
WG_HYB_BLOCK(swin_block_c64_add_hyb, 64, 2, 256, true, WG_MINW_C64)
WG_HYB_BLOCK(swin_block_c96_hyb, 96, 4, 384, false, WG_MINW_C96)
WG_HYB_BLOCK(swin_block_c96_add_hyb, 96, 4, 384, true, WG_MINW_C96)
WG_HYB_BLOCK(swin_block_c128_hyb, 128, 4, 512, false, WG_MINW_C128)

WG_KERNEL_N(patch_merge_c64_hyb, WG_GEMM_THREADS)(const u8* s, u8* out, const u8* w, int H, int W) {
  patch_merge<64, 96, WG_NT, true>(s, out, w, H, W);
}
WG_KERNEL_N(patch_merge_c96_hyb, WG_GEMM_THREADS)(const u8* s, u8* out, const u8* w, int H, int W) {
  patch_merge<96, 128, WG_NT, true>(s, out, w, H, W);
}
WG_KERNEL_N(expand_c96_hyb, WG_GEMM_THREADS)(const u8* y, u8* e, const u8* w, int Hl, int Wl) {
  expand<96, 64, WG_NT, true>(y, e, w, Hl, Wl);
}
WG_KERNEL_N(expand_c128_hyb, WG_GEMM_THREADS)(const u8* y, u8* e, const u8* w, int Hl, int Wl) {
  expand<128, 96, WG_NT, true>(y, e, w, Hl, Wl);
}
