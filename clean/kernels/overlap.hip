// SPDX-License-Identifier: MIT
// Micro-benchmark kernels for runner/overlap_probe.py: can the matrix unit (WMMA) and the vector ALU make progress at
// the same time? Synthetic data only. Device-only (the host emulator has no use for it).
//
// Every kernel runs `iters` iterations per wave. One iteration issues 8 independent WMMAs (8 accumulators) and/or a
// number of independent vector instructions (16 independent dependency chains, so latency is hidden):
//   ov_mma       8 WMMA / iteration                                  (matrix only)
//   ov_mma_dep   8 WMMA / iteration, all on ONE accumulator           (dependent chain: WMMA latency)
//   ov_valu8     64 FMA / iteration                                   (vector only; 8 FMA per WMMA slot)
//   ov_valu32    256 FMA / iteration                                  (32 per WMMA slot)
//   ov_trans     64 v_exp_f32 / iteration                             (transcendentals only)
//   ov_mix8      8 WMMA + 64 FMA, interleaved 1 WMMA : 8 FMA          (same wave does both)
//   ov_mix32     8 WMMA + 256 FMA, interleaved 1 : 32
//   ov_mix_trans 8 WMMA + 64 v_exp_f32, interleaved 1 : 8
//   ov_split8    even waves: 8 WMMA / iteration; odd waves: 64 FMA / iteration       (different waves)
//   ov_split32   even waves: 8 WMMA; odd waves: 256 FMA
// Each wave stores one value so nothing is optimized away. See overlap_probe.py for how the times are read.
#include "wg_platform.h"

#if defined(__AMDGCN__)

namespace {

constexpr int kBlock = 256;

#define OV_SCHED_ALU 0x0002
#define OV_SCHED_WMMA 0x0008
#define OV_SCHED_TRANS 0x0400

struct Seed {
  v2i a, b;
  float c1, c2;
};

WG_FN Seed make_seed(float c1, float c2) {
  const u32 t = wg_lane_id();
  Seed s;
  s.a[0] = (int)(t * 0x01010101u + 0x38383838u);
  s.a[1] = (int)(t * 0x03030303u + 0x34343434u);
  s.b[0] = (int)(t * 0x05050505u + 0x30303030u);
  s.b[1] = (int)(t * 0x07070707u + 0x3a3a3a3au);
  s.c1 = c1;
  s.c2 = c2;
  return s;
}

WG_FN void finish(float* out, const v8f* acc, const float* x) {
  float s = 0.0f;
#pragma unroll
  for (int i = 0; i < 8; ++i) s += acc[i][0] + acc[i][7];
#pragma unroll
  for (int j = 0; j < 16; ++j) s += x[j];
  out[wg_block_id() * kBlock + wg_lane_id()] = s;
}

// NW WMMAs per iteration (0 = none), and KV vector instructions after each of them (or KV*8 total if NW == 0);
// TRANS: the vector instruction is v_exp_f32 instead of v_fma_f32; DEP: all WMMAs on one accumulator.
template <int NW, int KV, bool TRANS, bool DEP, bool VEC_ONLY>
WG_FN void body(float* out, int iters, float c1, float c2) {
  const Seed s = make_seed(c1, c2);
  v8f acc[8];
  float x[16];
#pragma unroll
  for (int i = 0; i < 8; ++i) acc[i] = wg_splat((float)i * 0.001f);
#pragma unroll
  for (int j = 0; j < 16; ++j) x[j] = 0.5f + (float)j * 0.01f + (float)wg_lane_id() * 0.0001f;
  for (int it = 0; it < iters; ++it) {
    if (!VEC_ONLY) {
#pragma unroll
      for (int i = 0; i < NW; ++i) {
        const int a = DEP ? 0 : i;
        acc[a] = wg_wmma_fp8(s.a, s.b, acc[a]);
        if (!VEC_ONLY) {
#pragma unroll
          for (int k = 0; k < KV; ++k) {
            const int j = (i * KV + k) & 15;
            x[j] = TRANS ? wg_exp2(x[j]) : __builtin_fmaf(x[j], s.c1, s.c2);
          }
        }
      }
      if (NW > 0 && KV > 0) {
#pragma unroll
        for (int i = 0; i < NW; ++i) {
          __builtin_amdgcn_sched_group_barrier(OV_SCHED_WMMA, 1, 0);
          __builtin_amdgcn_sched_group_barrier(TRANS ? OV_SCHED_TRANS : OV_SCHED_ALU, KV, 0);
        }
      }
    } else {
#pragma unroll
      for (int k = 0; k < KV; ++k) {
        const int j = k & 15;
        x[j] = TRANS ? wg_exp2(x[j]) : __builtin_fmaf(x[j], s.c1, s.c2);
      }
    }
  }
  finish(out, acc, x);
}

}  // namespace

#define OV_KERNEL(name, ...) \
  WG_KERNEL_N(name, kBlock)(float* out, int iters, float c1, float c2) { body<__VA_ARGS__>(out, iters, c1, c2); }

OV_KERNEL(ov_mma, 8, 0, false, false, false)
OV_KERNEL(ov_mma_dep, 8, 0, false, true, false)
OV_KERNEL(ov_valu8, 0, 64, false, false, true)
OV_KERNEL(ov_valu32, 0, 256, false, false, true)
OV_KERNEL(ov_trans, 0, 64, true, false, true)
OV_KERNEL(ov_mix8, 8, 8, false, false, false)
OV_KERNEL(ov_mix32, 8, 32, false, false, false)
OV_KERNEL(ov_mix_trans, 8, 8, true, false, false)

// Different waves of the same workgroup do the two kinds of work (even / odd wave id).
WG_KERNEL_N(ov_split8, kBlock)(float* out, int iters, float c1, float c2) {
  if (((wg_lane_id() >> 5) & 1u) == 0u) body<8, 0, false, false, false>(out, iters, c1, c2);
  else body<0, 64, false, false, true>(out, iters, c1, c2);
}
WG_KERNEL_N(ov_split32, kBlock)(float* out, int iters, float c1, float c2) {
  if (((wg_lane_id() >> 5) & 1u) == 0u) body<8, 0, false, false, false>(out, iters, c1, c2);
  else body<0, 256, false, false, true>(out, iters, c1, c2);
}

#endif  // __AMDGCN__
