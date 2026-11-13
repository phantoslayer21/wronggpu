// SPDX-License-Identifier: MIT
// BASELINE COPY for timing comparisons only (perf_variants.py variant "v1"; not used by the runtime): clean/kernels/exposure.hip
// as of commit 2ecfc32 (branch clean/runtime-core), i.e. the version in use before the performance milestone (the
// spec team's timings 0.331 / 0.152 / 0.035 ms were taken on earlier commits of the same kernel). Unchanged except
// for the include path.
// Auto-exposure of spec/07-auto-exposure.md (stage "exposure") for gfx1201 (RDNA4).
//
// exposure_partial: each 256-lane workgroup takes kRowsPerGroup rows of the n_x x n_y sample grid; lanes stride over
//   the columns. Sample positions use exact integer arithmetic (X_i = floor((2 i (rw - 1) + n_x - 1) /
//   (2 (n_x - 1)))). Each lane sums z = ln(L + 1e-6) (0 for non-finite or non-positive L) in FP32; an LDS tree
//   reduces the workgroup to one partial sum.
// exposure_final: one workgroup sums the partials (FP32), m = sum / N, e = min(RN16(K exp(-m)), 65504).
// Layouts: clean/FORMATS.md, "Auto-exposure (exposure)"; launch plan: clean/runner/exposure.py.
#include "../wg_platform.h"

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
constexpr int kRowsPerGroup = 4;
constexpr float kLn2 = 0.69314718055994531f;

#if defined(__AMDGCN__)
WG_FN float exp_log2(float x) { return __builtin_amdgcn_logf(x); }
#else
WG_FN float exp_log2(float x) { return std::log2(x); }
#endif

// Sum of v over the workgroup (result valid in lane 0). Every lane must call it.
WG_FN float block_sum(float v, float* red, u32 tid) {
  red[tid] = v;
  wg_barrier();
  for (u32 s = kExpThreads / 2; s > 0; s >>= 1) {
    if (tid < s) red[tid] = red[tid] + red[tid + s];
    wg_barrier();
  }
  return red[0];
}

}  // namespace

WG_KERNEL_N(exposure_partial, 256)(const float* color, float* partials, WgExpParams P) {
  WG_LDS float red[kExpThreads];
  const u32 tid = wg_lane_id();
  const v4f* c4 = (const v4f*)color;  // [rh][rw] RGBA f32
  float acc = 0.0f;
  for (int jj = 0; jj < kRowsPerGroup; ++jj) {
    const int j = (int)wg_block_id() * kRowsPerGroup + jj;
    if (j >= P.ny) break;  // uniform over the workgroup
    const int Y = (2 * j * (P.rh - 1) + (P.ny - 1)) / (2 * (P.ny - 1));
    for (int i = (int)tid; i < P.nx; i += kExpThreads) {
      const int X = (2 * i * (P.rw - 1) + (P.nx - 1)) / (2 * (P.nx - 1));
      const v4f c = c4[(size_t)Y * P.rw + X];
      const float L = (0.25f * c[0] + 0.5f * c[1] + 0.25f * c[2]) * P.inv_pi;
      const bool ok = L > 0.0f && L <= 3.40282347e38f;  // finite and positive (false for NaN)
      acc += ok ? exp_log2(L + 1e-6f) * kLn2 : 0.0f;
    }
  }
  const float s = block_sum(acc, red, tid);
  if (tid == 0) partials[wg_block_id()] = s;
}

WG_KERNEL_N(exposure_final, 256)(const float* partials, float* out, WgExpParams P) {
  WG_LDS float red[kExpThreads];
  const u32 tid = wg_lane_id();
  float acc = 0.0f;
  for (int i = (int)tid; i < P.nparts; i += kExpThreads) acc += partials[i];
  const float s = block_sum(acc, red, tid);
  if (tid == 0) {
    const float m = s * P.inv_n;
    // exp(-m) = 2^n 2^(r log2 e), -m = n ln2 + r (two-part ln2), accurate to a few ulp
    const float x = -m;
    const float n = wg_rint(x * 1.44269504088896341f);
    float r = __builtin_fmaf(-n, 0.693145751953125f, x);
    r = __builtin_fmaf(-n, 1.42860682030941723e-06f, r);
    const float e = P.K * __builtin_ldexpf(wg_exp2(r * 1.44269504088896341f), (int)n);
    out[0] = wg_fmin((float)(_Float16)e, 65504.0f);  // RN16, saturated at the largest finite FP16 (spec 07)
  }
}

// Exposure pair of the per-frame runtime (spec 10 section 3), one lane: ex[0] = e of this frame (computed[0] when
// use_computed, else value), ex[1] = e_prev (e itself on the first frame, else ex[2]), ex[2] = e (kept for the next
// frame). Stages read [ex[0], ex[1]] as raw values; each replaces 0 by 1 itself (specs 03, 06).
WG_KERNEL_N(exposure_pair, 32)(float* ex, const float* computed, float value, int use_computed, int first) {
  if (wg_lane_id() != 0) return;
  const float e = use_computed ? computed[0] : value;
  const float e_prev = first ? e : ex[2];
  ex[0] = e;
  ex[1] = e_prev;
  ex[2] = e;
}

#if defined(WG_EMU)
WG_EMU_REGISTER(exposure_pair)
WG_EMU_REGISTER(exposure_partial)
WG_EMU_REGISTER(exposure_final)
#endif
