// SPDX-License-Identifier: MIT
// Test-only bridge to the actual native helper, including locally restored CPU modes.
#include "../runtime/k_input_uniforms.h"
#include <xmmintrin.h>
struct InputFrame {
  uintptr_t color, motion, depth, xs;
  int cpitch, mpitch, zpitch, cbx, cby, mbx, mby, zbx, zby, W, H, rw, rh, Bw, Bh, flags;
  float jx, jy, sx, sy, mox, moy, kappa, rho, lam, s;
  int mch;
  float rmsx, rmsy, rmox, rmoy;
  int pad;
};
static_assert(sizeof(InputFrame) == 160, "actual input argument layout");
#if defined(_WIN32)
#define TEST_EXPORT __declspec(dllexport)
#else
#define TEST_EXPORT __attribute__((visibility("default")))
#endif
extern "C" TEST_EXPORT int test_input_uniforms(const InputFrame* frames, wgrt::KiHost* out,
                                               unsigned* accepted, unsigned n, int mode) {
  std::fenv_t saved;
  std::fegetenv(&saved);
  const unsigned csr = _mm_getcsr();
  if (mode == 1) _mm_setcsr(csr | 0x8040u);  // this thread only: FTZ and DAZ
  if (mode == 2) std::fesetround(FE_UPWARD);
  if (mode == 3) std::fesetround(FE_DOWNWARD);
  if (mode == 4) std::fesetround(FE_TOWARDZERO);
  for (unsigned i = 0; i < n; ++i) {
    out[i] = wgrt::input_uniforms(frames[i]);
    wgrt::KiHost guarded;
    accepted[i] = wgrt::input_uniforms_safe(frames[i], &guarded);
  }
  std::fesetenv(&saved);
  _mm_setcsr(csr);
  return _mm_getcsr() == csr && std::fegetround() == FE_TONEAREST;
}
