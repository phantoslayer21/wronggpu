// SPDX-License-Identifier: MIT
// Auto-exposure of spec/07-auto-exposure.md (stage "exposure") for gfx1201 (RDNA4).
//
// exposure_partial: each 256-lane workgroup takes kRowsPerGroup rows of the n_x x n_y sample grid; lanes stride over
//   the columns. Sample positions use exact integer arithmetic (X_i = floor((2 i (rw - 1) + n_x - 1) /
//   (2 (n_x - 1)))). Each lane sums z = ln(L + 1e-6) (0 for non-finite or non-positive L) in FP32; the workgroup sum
//   is one partial. Version 2 (performance milestone, clean/PERF.md): a lane issues the loads of all its samples
//   (up to 4 columns x 4 rows) before using any of them, and the workgroup sum is a lane-exchange reduction inside
//   each wave plus one barrier (version 1: a loop with one load in flight per lane and an 8-barrier LDS tree).
// exposure_final: one workgroup sums the partials (FP32), m = sum / N, e = min(RN16(K exp(-m)), 65504).
// Spec 07 allows any summation order; the sums differ from version 1 only by FP32 rounding.
// Layouts: clean/FORMATS.md, "Auto-exposure (exposure)"; launch plan: clean/runner/exposure.py.
#include "wg_platform.h"
#include "wg_mem.h"

// Per-frame constants, passed by value (must match ExpParams in clean/runner/exposure.py).
struct WgExpParams {
  int rw, rh;    // render size
  int nx, ny;    // sample grid
  int nparts;    // number of partial sums (= exposure_partial workgroups)
  int pad0;
  float inv_pi;  // 1 / pre-exposure
  float K;       // alpha / (1 - alpha)
  float inv_n;   // 1 / (nx ny)
};
static_assert(sizeof(WgExpParams) == 36, "WgExpParams must match ExpParams in clean/runner/exposure.py");

namespace {

constexpr int kExpThreads = 256;
constexpr int kWaves = kExpThreads / WG_WAVE;
constexpr int kRowsPerGroup = 4;
constexpr int kColsPerPass = 4;  // columns per lane whose loads are in flight together
constexpr float kLn2 = 0.69314718055994531f;

// Sum over the 32 lanes of a wave (result in every lane), by lane exchanges.
WG_FN float wave_sum(float v) {
  for (int d = 16; d >= 1; d >>= 1) {  // unrolled by the compiler (constant trip count)
#if defined(__AMDGCN__)
    v += __builtin_bit_cast(float, __builtin_amdgcn_ds_bpermute((int)((wg_lane_id() ^ (u32)d) << 2),
                                                                 __builtin_bit_cast(int, v)));
#else
    v += wg_emu_xor(v, (u32)d);
#endif
  }
  return v;
}

// Sum over the workgroup (valid in every lane of wave 0). Every lane must call it.
WG_FN float block_sum(float v, float* red, u32 tid) {
  v = wave_sum(v);
  if ((tid & 31u) == 0u) red[tid >> 5] = v;
  wg_barrier();
  float s = 0.0f;
#pragma unroll
  for (int i = 0; i < kWaves; ++i) s += red[i];
  return s;
}

WG_FN float sample_z(v4f c, float inv_pi) {
  const float L = (0.25f * c[0] + 0.5f * c[1] + 0.25f * c[2]) * inv_pi;
  const bool ok = L > 0.0f && L <= 3.40282347e38f;  // finite and positive (false for NaN)
  return ok ? wg_log2(L + 1e-6f) * kLn2 : 0.0f;
}

// This workgroup's partial sum: rows j0 .. j0 + 3 of the sample grid. C16: RGBA16F texels in rows of `pitch` texels
// (model K's input copies, clean/k/runtime/runtime_k.cpp; the samples are the same values widened exactly), else
// dense RGBA float32 (pitch unused: rw).
template <bool C16 = false>
WG_FN float partial_sum(const void* __restrict__ color, const WgExpParams& P, u32 tid, float* red, int pitch = 0) {
  const int j0 = (int)wg_block_id() * kRowsPerGroup;
  u32 row[kRowsPerGroup];
  bool rok[kRowsPerGroup];
#pragma unroll
  for (int jj = 0; jj < kRowsPerGroup; ++jj) {
    const int j = j0 + jj;
    rok[jj] = j < P.ny;
    const int Y = (2 * j * (P.rh - 1) + (P.ny - 1)) / (2 * (P.ny - 1));
    row[jj] = (u32)(rok[jj] ? Y : 0) * (u32)(C16 ? pitch : P.rw);
  }
  float acc = 0.0f;
  for (int base = 0; base < P.nx; base += kColsPerPass * kExpThreads) {  // one pass at the in-game size
    v4f c[kColsPerPass][kRowsPerGroup];
    bool ok[kColsPerPass];
#pragma unroll
    for (int k = 0; k < kColsPerPass; ++k) {
      const int i = base + k * kExpThreads + (int)tid;
      ok[k] = i < P.nx;
      const int X = ok[k] ? (2 * i * (P.rw - 1) + (P.nx - 1)) / (2 * (P.nx - 1)) : 0;
#pragma unroll
      for (int jj = 0; jj < kRowsPerGroup; ++jj)
        if (C16) {
          typedef _Float16 h4 __attribute__((ext_vector_type(4)));
          const h4 h = *(const h4*)((const char*)color + (row[jj] + (u32)X) * 8u);
          c[k][jj] = v4f{(float)h[0], (float)h[1], (float)h[2], (float)h[3]};
        } else {
          c[k][jj] = __builtin_bit_cast(v4f, wg_ld128_a16((const float*)color, (row[jj] + (u32)X) * 16u));
        }
    }
#pragma unroll
    for (int k = 0; k < kColsPerPass; ++k)
#pragma unroll
      for (int jj = 0; jj < kRowsPerGroup; ++jj) acc += ok[k] && rok[jj] ? sample_z(c[k][jj], P.inv_pi) : 0.0f;
  }
  return block_sum(acc, red, tid);
}

// e = min(RN16(K exp(-m)), 65504), m = sum / N; exp(-m) = 2^n 2^(r log2 e), -m = n ln2 + r (two-part ln2).
WG_FN float exposure_value(float sum, const WgExpParams& P) {
  const float x = -(sum * P.inv_n);
  const float n = wg_rint(x * 1.44269504088896341f);
  float r = __builtin_fmaf(-n, 0.693145751953125f, x);
  r = __builtin_fmaf(-n, 1.42860682030941723e-06f, r);
  const float e = P.K * __builtin_ldexpf(wg_exp2(r * 1.44269504088896341f), (int)n);
  return wg_fmin((float)(_Float16)e, 65504.0f);  // RN16, saturated at the largest finite FP16 (spec 07)
}

// Sum of the partials in index order (lanes stride, then the fixed-order workgroup reduction).
WG_FN float sum_partials(const float* partials, int nparts, u32 tid, float* red) {
  float acc = 0.0f;
  for (int i = (int)tid; i < nparts; i += kExpThreads) acc += partials[i];
  return block_sum(acc, red, tid);
}

}  // namespace

WG_KERNEL_N(exposure_partial, 256)(const float* color, float* partials, WgExpParams P) {
  WG_LDS float red[kWaves];
  const u32 tid = wg_lane_id();
  const float s = partial_sum(color, P, tid, red);
  if (tid == 0) partials[wg_block_id()] = s;
}

WG_KERNEL_N(exposure_final, 256)(const float* partials, float* out, WgExpParams P) {
  WG_LDS float red[kWaves];
  const u32 tid = wg_lane_id();
  const float s = sum_partials(partials, P.nparts, tid, red);
  if (tid == 0) out[0] = exposure_value(s, P);
}

// Exposure pair of the per-frame runtime (spec 10 section 3), one lane: ex[0] = e of this frame (computed[0] when
// use_computed, else value), ex[1] = e_prev (e itself on the first frame, else ex[2]), ex[2] = e (kept for the next
// frame). Stages read [ex[0], ex[1]] as raw values; each replaces 0 by 1 itself (specs 03, 06).
// scale: 1, or 1 / pre-exposure for an exposure x the game supplies (runtime option pre_exposure=original): the pair
// is (x, x_prev) * scale, both with this frame's factor, so e / e_prev = x / x_prev as when the color is divided by
// the pre-exposure instead; ex[2] keeps the unscaled x. Times 1 leaves every value bit for bit.
WG_KERNEL_N(exposure_pair, 32)(float* ex, const float* computed, float value, int use_computed, int first,
                               float scale) {
  if (wg_lane_id() != 0) return;
  const float x = use_computed ? computed[0] : value;
  const float e = x * scale;
  const float e_prev = first ? e : ex[2] * scale;
  ex[0] = e;
  ex[1] = e_prev;
  ex[2] = x;
}

// exposure_pair plus the post kernels' exposure-derived uniforms (the _diet post kernels, runtime option post_diet=1):
// ex[3] = ratio = (float)(e / e_prev), ex[4] = dec_scale = (float)(0.01 / e), each with 0 replaced by 1 as the post
// kernels do (the same FP64 expressions, once instead of per wave). The pair buffer is 32 bytes.
WG_KERNEL_N(exposure_pair_diet, 32)(float* ex, const float* computed, float value, int use_computed, int first,
                                    float scale) {
  if (wg_lane_id() != 0) return;
  const float x = use_computed ? computed[0] : value;
  const float e = x * scale;
  const float e_prev = first ? e : ex[2] * scale;
  ex[0] = e;
  ex[1] = e_prev;
  ex[2] = x;
  const float e1 = e == 0.0f ? 1.0f : e, ep1 = e_prev == 0.0f ? 1.0f : e_prev;
  ex[3] = (float)((double)e1 / (double)ep1);
  ex[4] = (float)(0.01 / (double)e1);
}

// Auto exposure in one launch (runtime option post_diet=1): exposure_partial, then the workgroup that finishes last
// (a counter at `sync`, zero before the launch and zero again after it) does exposure_final and exposure_pair_diet:
// the same sums in the same order and the same expressions, so the same bytes as the three launches. Grid: nparts.
#if defined(__AMDGCN__)
WG_FN u32 exp_count(u32* p) { return __hip_atomic_fetch_add(p, 1u, __ATOMIC_ACQ_REL, __HIP_MEMORY_SCOPE_AGENT); }
WG_FN void exp_fence_release() { __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent"); }
WG_FN void exp_fence_acquire() { __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent"); }
#else
WG_FN u32 exp_count(u32* p) { return __atomic_fetch_add(p, 1u, __ATOMIC_SEQ_CST); }
WG_FN void exp_fence_release() {}
WG_FN void exp_fence_acquire() {}
#endif
// The body of exposure_auto_diet (C16: exposure_auto_diet_c16, RGBA16F texels in rows of cpitch texels).
template <bool C16>
WG_FN void auto_diet(const void* color, int cpitch, float* partials, u32* sync, float* computed, float* ex,
                     const WgExpParams& P, int first) {
  WG_LDS float red[kWaves];
  WG_LDS u32 is_last;
  const u32 tid = wg_lane_id();
  const float s = partial_sum<C16>(color, P, tid, red, cpitch);
  if (tid == 0) {
    partials[wg_block_id()] = s;
    exp_fence_release();
    is_last = exp_count(sync) == (u32)(P.nparts - 1) ? 1u : 0u;
  }
  wg_barrier();
  if (!is_last) return;
  exp_fence_acquire();
  const float t = sum_partials(partials, P.nparts, tid, red);
  if (tid != 0) return;
  const float e = exposure_value(t, P);
  computed[0] = e;
  const float e_prev = first ? e : ex[2];
  ex[0] = e;
  ex[1] = e_prev;
  ex[2] = e;
  const float e1 = e == 0.0f ? 1.0f : e, ep1 = e_prev == 0.0f ? 1.0f : e_prev;
  ex[3] = (float)((double)e1 / (double)ep1);
  ex[4] = (float)(0.01 / (double)e1);
  sync[0] = 0u;  // for the next launch (stream-ordered after this kernel)
}

WG_KERNEL_N(exposure_auto_diet, 256)(const float* color, float* partials, u32* sync, float* computed, float* ex,
                                     WgExpParams P, int first) {
  auto_diet<false>(color, 0, partials, sync, computed, ex, P, first);
}
// model K reading the interop's RGBA16F colour copy directly (same sums as exposure_auto_diet on the widened values)
WG_KERNEL_N(exposure_auto_diet_c16, 256)(const _Float16* color, int cpitch, float* partials, u32* sync,
                                         float* computed, float* ex, WgExpParams P, int first) {
  auto_diet<true>(color, cpitch, partials, sync, computed, ex, P, first);
}

#if defined(WG_EMU)
WG_EMU_REGISTER(exposure_auto_diet)
WG_EMU_REGISTER(exposure_auto_diet_c16)
WG_EMU_REGISTER(exposure_pair)
WG_EMU_REGISTER(exposure_pair_diet)
WG_EMU_REGISTER(exposure_partial)
WG_EMU_REGISTER(exposure_final)
#endif
