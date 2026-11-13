// SPDX-License-Identifier: MIT
// Micro-benchmark of the MLP tile loop of the Swin block (W1 chain -> GELU + FP8 pack -> W2), with every other part of
// the block removed: no global loads, no weight staging, no barriers, no attention. The weights are a fixed LDS unit
// (synthetic bytes), exactly as the real loop reads them (one 8-byte fragment per lane and WMMA). Driver:
// runner/mlp_probe.py. The point is to find out how far the loop is from the sum of its parts:
//   name mb<KC>_<W1><GELU><W2>: KC = 4 (C=64) or 8 (C=128); each digit 1 = part present, 0 = removed
//   (W1 absent: the accumulator starts from the bias; GELU absent: identity; W2 absent: the packed values are only
//   folded into a register so they stay alive). Device only.
#include "wg_platform.h"

#if defined(__AMDGCN__)

namespace {

WG_FN v2i lfrag(const u8* unit, int index, u32 lane) { return ((const v2i*)(unit + index * 256))[lane]; }

template <int KC, bool W1, bool GE, bool W2, bool PF>
WG_FN void mlp_bench(float* out, int tiles) {
  WG_LDS __attribute__((aligned(16))) u8 lds[16384];
  const u32 tid = wg_lane_id();
  for (u32 i = tid; i < 16384 / 4; i += 256) ((u32*)lds)[i] = 0x30303030u + ((i * 0x01010101u * 7u) & 0x0f0f0f0fu);
  wg_barrier();
  const u32 lane = tid & 31u;
  v2i mf[KC];
  v8f acc[KC];
#pragma unroll
  for (int k = 0; k < KC; ++k) {
    mf[k][0] = (int)(0x30303030u + ((tid * 0x01010101u + k) & 0x0f0f0f0fu));
    mf[k][1] = (int)(0x31313131u + ((tid * 0x03030303u + k) & 0x07070707u));
    acc[k] = wg_splat(0.0f);
  }
  const v8f bias = wg_splat(0.125f);
  v2i f1[KC];  // PF: this tile's W1 fragments, loaded during the previous tile
  if (PF) {
#pragma unroll
    for (int kc = 0; kc < KC; ++kc) f1[kc] = lfrag(lds, kc, lane);
  }
#pragma unroll 1
  for (int j = 0; j < tiles; ++j) {
    const u8* U = lds;  // the tiles cycle through 4 slots of the unit (the slot is part of the fragment index)
    v8f hacc = bias;
    if (W1) {
#pragma unroll
      for (int kc = 0; kc < KC; ++kc) hacc = wg_wmma_fp8(PF ? f1[kc] : lfrag(U, ((j & 3) * 2 * KC) + kc, lane), mf[kc], hacc);
    } else {
      hacc[0] += (float)mf[0][0];
    }
    v2i f2[KC];
    if (PF) {  // W2 fragments of this tile and W1 fragments of the next one, ahead of the GELU
#pragma unroll
      for (int mt = 0; mt < KC; ++mt) f2[mt] = lfrag(U, ((j & 3) * 2 * KC) + KC + mt, lane);
#pragma unroll
      for (int kc = 0; kc < KC; ++kc) f1[kc] = lfrag(U, (((j + 1) & 3) * 2 * KC) + kc, lane);
      __asm__ volatile("" ::: "memory");
      __builtin_amdgcn_sched_barrier(0);
    }
    v8f ge;
#pragma unroll
    for (int v = 0; v < 8; ++v) ge[v] = GE ? wg_gelu(hacc[v]) : hacc[v];
    const v2i hf = wg_pack_q8_nc(ge);
    if (W2) {
#pragma unroll
      for (int mt = 0; mt < KC; ++mt)
        acc[mt] = wg_wmma_fp8(PF ? f2[mt] : lfrag(U, ((j & 3) * 2 * KC) + KC + mt, lane), hf, acc[mt]);
    } else {
      acc[0][0] += (float)(hf[0] ^ hf[1]);
    }
  }
  float s = 0.0f;
#pragma unroll
  for (int k = 0; k < KC; ++k) s += acc[k][0] + acc[k][7];
  out[(size_t)wg_block_id() * 256 + tid] = s;
}

}  // namespace

#define MB(KC, W1, GE, W2) \
  WG_KERNEL_N(mb##KC##_##W1##GE##W2, 256)(float* out, int tiles) { mlp_bench<KC, W1 == 1, GE == 1, W2 == 1, false>(out, tiles); }
#define MB_ALL(KC) \
  MB(KC, 1, 1, 1) MB(KC, 0, 1, 1) MB(KC, 1, 0, 1) MB(KC, 1, 1, 0) MB(KC, 1, 0, 0) MB(KC, 0, 0, 1) MB(KC, 0, 1, 0) MB(KC, 0, 0, 0)
MB_ALL(4)
MB_ALL(8)
// the full loop with the fragment prefetch (the WG_MLP_PF idea of swin.hip)
WG_KERNEL_N(mb4_111p, 256)(float* out, int tiles) { mlp_bench<4, true, true, true, true>(out, tiles); }
WG_KERNEL_N(mb8_111p, 256)(float* out, int tiles) { mlp_bench<8, true, true, true, true>(out, tiles); }

#endif
