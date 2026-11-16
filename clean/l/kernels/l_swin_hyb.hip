// SPDX-License-Identifier: MIT
// Model L's C = 32 blocks in the HYBRID form (WG_L_FORM=hybrid; swin_block<..., HYB = true>, clean/kernels/wg_hyb.h):
// the HIP twins of the D3D12 ports l_swin_block_c32_enc_hyb / _dec_hyb (clean/dx12/kernels/l_swin.hlsl, SW_HYB=1), the
// same bytes. Shapes of l_swin_block_c32_enc_nc / _dec_nc (clean/l/kernels/l_swin.hip). Its own code object
// (l_swin_hyb.co), so the default objects stay as they are.
#define L_SWIN_HELPERS_ONLY 1
#include "l_swin.hip"

#define L_HYB_KERNEL(name, EPI, SRC, OUTARG)                                                                          \
  WG_KERNEL_NW(name, 256, WG_MINW_C32)(const u8* x, const u8* xadd, u8* out, const u8* w, int H, int W, int ox,     \
                                       int oy) {                                                                      \
    swin_block<32, 2, 128, 1, false, WG_MLP_GROUP, true, 256, 0, 0, EPI, false, SRC, true>(x, xadd, OUTARG, w, H, W, \
                                                                                         ox, oy, nullptr, out);     \
  }
L_HYB_KERNEL(l_swin_block_c32_enc_hyb, SwinStoreEpi, LHaloF16Src, out)
// the decoder: ADD (xadd = the skip map S, fine mirror; x = E by the coarse mirror in swin_block's HYB && ADD gather),
// member input RN16(E + S) (oracle swin_block skip, l_q7_block_c32_dec)
WG_KERNEL_NW(l_swin_block_c32_dec_hyb, 256, WG_MINW_C32)(const u8* x, const u8* xadd, u8* out, const u8* w, int H, int W,
                                                        int ox, int oy) {
  swin_block<32, 2, 128, 1, true, WG_MLP_GROUP, true, 256, 0, 0, LStoreF16Epi, false, SwinFp8Src, true>(
      x, xadd, nullptr, w, H, W, ox, oy, nullptr, out);
}
