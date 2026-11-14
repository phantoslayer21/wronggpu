// SPDX-License-Identifier: MIT
// Model L (issue #89): Swin block kernels at the widths Model L adds, from Model M's block template
// (clean/kernels/swin.hip, included for its templates only). Same arithmetic, packed weights
// (clean/runner/packing.py pack_block) and launch geometry as Model M's block kernels (clean/FORMATS.md, "Kernels and
// launch geometry"): 1-D grid of ceil(nwin / 8) workgroups of 256 lanes, one window per wave.
//
// Build: clang -x hip ... -I clean/kernels clean/l/kernels/l_swin.hip -o l_swin.co (clean/l/runner/l_build.py).
#define WG_SWIN_NO_EXPORTS 1
#include "swin.hip"

#ifndef WG_MINW_C32
#define WG_MINW_C32 1
#endif

// x, xadd (unused, may be null), out, w, H, W, ox, oy: as swin_block_c64. `_nc` when bit 0 of the blob's flags word is
// set (the weights prove every inner FP8 pack in range), with identical results.
WG_KERNEL_NW(l_swin_block_c32, 256, WG_MINW_C32)(const u8* x, const u8* xadd, u8* out, const u8* w, int H, int W,
                                                  int ox, int oy) {
  swin_block<32, 2, 128, 1, false, WG_MLP_GROUP, false, 256>(x, xadd, out, w, H, W, ox, oy, nullptr);
}
WG_KERNEL_NW(l_swin_block_c32_nc, 256, WG_MINW_C32)(const u8* x, const u8* xadd, u8* out, const u8* w, int H, int W,
                                                     int ox, int oy) {
  swin_block<32, 2, 128, 1, false, WG_MLP_GROUP, true, 256>(x, xadd, out, w, H, W, ox, oy, nullptr);
}

// ---- Model L's two blocks (clean/l/FORMATS.md) ---------------------------------------------------------------------

// Encoder block input: the FP16 halo map [H + 4][W + 4][C] (grid position (pc, pr) at map pixel (pr + 2, pc + 2));
// window members outside the grid read the halo directly. Needs H, W multiples of 4 and offsets of 0 or 2.
struct LHaloF16Src {
  static constexpr bool kCustom = true;
  static constexpr int kHalo = 2;
  template <int C, int KC>
  static WG_FN void load(const u8* x, int pr, int pc, int H, int W, u32 hh, v8f* f) {
    (void)H;
    const _Float16* p = (const _Float16*)x +
                        ((size_t)((pr + kHalo) * (W + 2 * kHalo) + pc + kHalo) * C + hh * (C / 2));
#pragma unroll
    for (int k = 0; k < KC; ++k) {
      const v8h h = *(const v8h*)(p + 8 * k);
#pragma unroll
      for (int j = 0; j < 8; ++j) f[k][j] = (float)h[j];
    }
  }
};

// Decoder block input: the FP8 pixel map [H][W][C] of 2 x 2 sub-pixel groups; a member outside the grid takes the
// mirrored 2 x 2 group (mirror of the group index, edge group not repeated) and keeps its sub-pixel.
struct LCoarseFp8Src {
  static constexpr bool kCustom = true;
  template <int C, int KC>
  static WG_FN void load(const u8* x, int pr, int pc, int H, int W, u32 hh, v8f* f) {
    const int r = 2 * wg_mirror(pr >> 1, H >> 1) + (pr & 1);
    const int c = 2 * wg_mirror(pc >> 1, W >> 1) + (pc & 1);
    v2i q[KC];
    load_frags<KC>(x + (size_t)(r * W + c) * C + hh * (C / 2), q);
#pragma unroll
    for (int k = 0; k < KC; ++k) f[k] = wg_unpack_fp8(q[k]);
  }
};

// Decoder block output: RN16(x2) as an FP16 map [H][W][C] (epi = the output pointer), real members only.
struct LStoreF16Epi {
  static constexpr bool kFused = true;
  template <int C, int KC, int WPW, int THREADS>
  static WG_FN void run(const v8f (&acc)[WPW][KC], const int (&dst)[WPW], int W, u8* lds, int lds_bytes, u32 tid,
                        const void* epi) {
    (void)W, (void)lds, (void)lds_bytes;
    const u32 hh = (tid & 31u) >> 4;
#pragma unroll
    for (int i = 0; i < WPW; ++i) {
      if (dst[i] < 0) continue;
      _Float16* p = (_Float16*)epi + (size_t)dst[i] * C + hh * (C / 2);
#pragma unroll
      for (int k = 0; k < KC; ++k) *(v8h*)(p + 8 * k) = wg_to_f16(acc[i][k]);
    }
  }
};

// x, xadd (unused), out, w, H, W, ox, oy as above. Encoder: x = FP16 halo map, out = FP8 map. Decoder: x = FP8 map,
// out = FP16 map.
#define L_BLOCK_KERNEL(name, NC, EPI, SRC, OUTARG)                                                                    \
  WG_KERNEL_NW(name, 256, WG_MINW_C32)(const u8* x, const u8* xadd, u8* out, const u8* w, int H, int W, int ox,     \
                                       int oy) {                                                                      \
    swin_block<32, 2, 128, 1, false, WG_MLP_GROUP, NC, 256, 0, 0, EPI, false, SRC>(x, xadd, OUTARG, w, H, W, ox, oy, \
                                                                                   nullptr, out);                    \
  }
L_BLOCK_KERNEL(l_swin_block_c32_enc, false, SwinStoreEpi, LHaloF16Src, out)
L_BLOCK_KERNEL(l_swin_block_c32_enc_nc, true, SwinStoreEpi, LHaloF16Src, out)
L_BLOCK_KERNEL(l_swin_block_c32_dec, false, LStoreF16Epi, LCoarseFp8Src, nullptr)
L_BLOCK_KERNEL(l_swin_block_c32_dec_nc, true, LStoreF16Epi, LCoarseFp8Src, nullptr)

// The same template at Model M's C = 64 (as swin_block_c64[_nc] of swin.co): the GPU check runs it next to Model M's
// own kernel to show that this file builds the template exactly as swin.co does.
WG_KERNEL_NW(l_swin_block_c64, 256, WG_MINW_C64)(const u8* x, const u8* xadd, u8* out, const u8* w, int H, int W,
                                                  int ox, int oy) {
  swin_block<64, 2, 256, 1, false, WG_MLP_GROUP, false, 256>(x, xadd, out, w, H, W, ox, oy, nullptr);
}
