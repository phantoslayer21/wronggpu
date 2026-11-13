// SPDX-License-Identifier: MIT
// Fused dec1 + dec0 (performance milestone, prototype; clean/PERF.md "dec1 + dec0 fusion"): the last Swin block of
// stage dec1 (spec 02, C = 64, decoder block: input = expanded map + skip) and the coefficient decoder (spec 05) in one
// kernel. The block's output q8(x2), the 64-channel FP8 map at F/2 that dec0 reads, never goes to memory: each wave's
// window of 16 tokens is projected by dec0's WMMA projection straight from registers, and the window's real tokens are
// decoded into A, B, C and the new latent. Results are bit-identical to swin_block_c64_add[_nc] followed by
// coef_decoder (tests/test_dec1_dec0_emu.py).
//
// What this file takes from the other kernels (both included for their functions only):
//   swin.hip (clean/swin, 1d4babd lineage): swin_block<64, 2, 256, WPW = 1, ADD, GROUP, SAFE, THREADS = 256, MCO = 0,
//   ECL = 0, EPI> (MCO / ECL and the aux argument added when clean/integration merged the fused merge / expand)
//     with the output epilogue hook EPI (see SwinStoreEpi there): after the block, every lane calls
//     EPI::run(acc, dst, W, lds, lds_bytes, tid, epi) with acc[0][k] = x2 of its token in layout L (lane half h holds
//     channels 32h + 8k .. 32h + 8k + 7) and dst[0] = the token's index in the H x W map, or -1 for a mirrored member.
//   decoder.hip: dec_stage, dec_motion, dec_project, prev_fetch, dec_store (the steps of coef_decoder).
// The stored map and dec0's input are the same bytes (wg_pack_q8 of x2), so only the channel distribution between the
// lane halves differs: dec0's K order wants channels 16h .. 16h + 15 and 32 + 16h .. 32 + 16h + 15 in lane half h, the
// block's layout L has 32h .. 32h + 31. Each half sends two of its four 8-byte groups to the other (lane ^ 16).
//
// Kernel: swin_block_c64_add_dec0[_nc](x, xadd, w, H, W, ox, oy, motion, latent_prev, A, B, C, latent, wdec, P):
// x, xadd, w, H, W, ox, oy as for swin_block_c64_add (H x W = F/2), the rest as for coef_decoder. Launch geometry of
// the block kernel: ceil(windows / 8) workgroups of 256 lanes. The _nc kernel is chosen like the block's (FLAGS bit 0).
#define WG_SWIN_NO_EXPORTS 1
#include "swin.hip"
#define WG_DEC_NO_EXPORTS 1
#include "decoder.hip"

namespace {

struct Dec0Args {
  const u32* motion;
  const u32* lprev;
  Out o;
  const u8* w;
  WgDecParams P;
};

struct Dec0Epi {
  static constexpr bool kFused = true;

  template <int C, int KC, int WPW, int THREADS>
  static WG_FN void run(const v8f (&acc)[WPW][KC], const int (&dst)[WPW], int W, u8* lds, int lds_bytes, u32 tid,
                        const void* epi) {
    static_assert(C == 64 && KC == 4 && WPW == 1 && THREADS == 256, "dec1 + dec0: C = 64, one window per wave");
    const Dec0Args& a = *(const Dec0Args*)epi;
    const u32 lane = tid & 31u, hh = lane >> 4;

    // The token of this lane (WMMA column): its F/2 position and the lane's two render pixels.
    const bool real = dst[0] >= 0;
    const int ti = real ? dst[0] : 0;
    const int r = ti / W, cc = ti - r * W;
    const int x0 = 2 * cc, y = 2 * r + (int)hh;
    const bool row_ok = real && y < a.P.rh;
    u32 mv0, mv1;
    dec_motion(a.P, a.motion, x0, y, mv0, mv1);

    // dec0's B fragments from the block output, exactly the bytes dec1 would store.
    v2i o[KC];
#pragma unroll
    for (int k = 0; k < KC; ++k) o[k] = wg_pack_q8(acc[0][k]);
    u32 snd[4], rcv[4];
    snd[0] = (u32)(hh ? o[0][0] : o[2][0]);
    snd[1] = (u32)(hh ? o[0][1] : o[2][1]);
    snd[2] = (u32)(hh ? o[1][0] : o[3][0]);
    snd[3] = (u32)(hh ? o[1][1] : o[3][1]);
#pragma unroll
    for (int i = 0; i < 4; ++i) rcv[i] = wg_xor16_u(snd[i]);
    v2i bf[kKB];
    const v2i r01 = {(int)rcv[0], (int)rcv[1]}, r23 = {(int)rcv[2], (int)rcv[3]};
    bf[0] = hh ? r01 : o[0];
    bf[1] = hh ? r23 : o[1];
    bf[2] = hh ? o[2] : r01;
    bf[3] = hh ? o[3] : r23;

    // The block's weight buffers are free once every wave is past its last W2 product.
    (void)lds_bytes;  // >= kLdsStaged (32 KB for C = 64)
    wg_barrier_lds();
    dec_stage<THREADS>(lds, a.w, tid);
    wg_barrier_lds();
    u32 code[2 * kMT];
    dec_project(bf, lds, a.w, lane, hh, code);
    const Prev q0 = prev_fetch(a.P, a.lprev, x0, y, mv0), q1 = prev_fetch(a.P, a.lprev, x0 + 1, y, mv1);
    dec_store(code, lds, a.P, q0, q1, a.o, x0, y, row_ok);
  }
};

static_assert(BlockLayout<64, 2, 256>::MAXU * 2 >= kLdsStaged, "dec0's tables and fragments fit the block's LDS");

}  // namespace

// Not in the instrumented Swin builds (WG_PROFILE, build.py --profile): their block template takes a profile buffer.
#if !WG_PROFILE
#define WG_DEC1_DEC0_KERNEL(name, SAFE)                                                                               \
  WG_KERNEL_NW(name, 256, WG_MINW_C64)(const u8* x, const u8* xadd, const u8* w, int H, int W, int ox, int oy,         \
                                       const u32* motion, const u32* latent_prev, u32* A, u32* B, u32* C, u32* latent, \
                                       const u8* wdec, WgDecParams P) {                                               \
    const Dec0Args a = {motion, latent_prev, {A, B, C, latent}, wdec, P};                                             \
    swin_block<64, 2, 256, 1, true, WG_MLP_GROUP, SAFE, 256, 0, 0, Dec0Epi>(x, xadd, nullptr, w, H, W, ox, oy, nullptr, &a);        \
  }

WG_DEC1_DEC0_KERNEL(swin_block_c64_add_dec0, false)
WG_DEC1_DEC0_KERNEL(swin_block_c64_add_dec0_nc, true)

#if defined(WG_EMU)
WG_EMU_REGISTER(swin_block_c64_add_dec0)
WG_EMU_REGISTER(swin_block_c64_add_dec0_nc)
#endif
#endif  // !WG_PROFILE
