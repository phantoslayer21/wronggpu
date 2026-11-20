// SPDX-License-Identifier: MIT
// The developer-only shortcut of the WG_DEV_GFX11 opening (runtime/rdna4.h, runtime/wmma_probe.h) on any host (no GPU): the
// choice by environment variable, the tolerance of the gfx12 probe under the FP8 emulation, and the probe of the F32 -> F16 ->
// F32 rounding. WrongGPU's own code. The same file is built twice and both builds must pass:
//   clang++ -std=c++17 test_shortcut_probe.cpp -o test_shortcut_probe.exe                  (a normal build)
//   clang++ -std=c++17 -DWG_DEV_GFX11 test_shortcut_probe.cpp -o test_shortcut_probe_dev.exe
// clean/tests/test_dev_gfx11_shortcut.py builds and runs both. Exit status 0 when every case passed.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <set>
#include <string>
#include <vector>

#include "../rdna4.h"
#include "../wmma_probe.h"

namespace {

namespace P = wgrt::wmma_probe;
namespace R = wgrt::wmma_probe::round16;

int failures = 0;

void expect(const std::string& name, bool ok) {
  std::printf("%s %s\n", ok ? "pass" : "FAIL", name.c_str());
  if (!ok) ++failures;
}

void set_env(const char* value) {
#ifdef _WIN32
  _putenv_s("WG_DEV_GFX11_SHORTCUT", value ? value : "");
#else
  if (value) setenv("WG_DEV_GFX11_SHORTCUT", value, 1); else unsetenv("WG_DEV_GFX11_SHORTCUT");
#endif
}

// (float)(_Float16) x by an independent route (round to nearest even through the default rounding mode of nearbyint)
float reference_round(float x) {
  int e;
  std::frexp(x, &e);   // |x| = m 2^e with m in [0.5, 1): the FP16 ulp is 2^(e - 11)
  const double ulp = std::ldexp(1.0, e - 11);
  return static_cast<float>(std::nearbyint(x / ulp) * ulp);
}

}  // namespace

int main() {
  // ---- the choice: an environment variable, read only in a build made with the macro ---------------------------------------
  const char* values[] = {"1", "0", "", "11", "yes", " 1"};
  set_env(nullptr);
  expect("unset: no shortcut in either build", !wgrt::dev_gfx11_shortcut());
  for (const char* v : values) {
    set_env(v);
    const bool on = std::string(v) == "1" && wgrt::kDevGfx11;
    expect(std::string("WG_DEV_GFX11_SHORTCUT='") + v + "': " + (on ? "on" : "off") + (wgrt::kDevGfx11 ? " (macro on)" : " (macro off)"),
           wgrt::dev_gfx11_shortcut() == on);
  }
  set_env("1");
  expect(wgrt::kDevGfx11 ? "macro on: =1 chooses the shortcut" : "macro off: =1 chooses nothing (a normal build never reads it)",
         wgrt::dev_gfx11_shortcut() == wgrt::kDevGfx11);
  set_env(nullptr);

  // ---- the gfx12 probe under the emulation: within the tolerance ---------------------------------------------------------
  {
    float r[P::kResults];
    for (unsigned i = 0; i < P::kResults; ++i) r[i] = P::expected(i);
    expect("shortcut probe: the exact results pass", P::first_mismatch_within(r, P::gfx11::kTolerance) < 0);
    for (unsigned i = 0; i < P::kResults; i += 3) r[i] += 3.0e-5f;   // the F16 instruction's last-bit accumulation
    expect("shortcut probe: last-bit differences (3e-5) pass", P::first_mismatch_within(r, P::gfx11::kTolerance) < 0);
    expect("... and the exact probe of RDNA 4 still rejects them (its rule is unchanged)", P::first_mismatch(r) >= 0);
    r[300] += 2.0e-3f;
    expect("shortcut probe: a difference of 2e-3 in float 300 is reported at 300", P::first_mismatch_within(r, P::gfx11::kTolerance) == 300);
    r[3] = std::nanf("");
    expect("shortcut probe: a NaN is reported", P::first_mismatch_within(r, P::gfx11::kTolerance) == 3);
    std::vector<float> e;
    for (unsigned i = 0; i < 256; ++i) e.push_back(P::expected(i));
    std::sort(e.begin(), e.end());
    float gap = 1e30f;
    for (size_t i = 1; i < e.size(); ++i) gap = std::min(gap, e[i] - e[i - 1]);
    expect("shortcut probe: the smallest gap between distinct expected results (" + std::to_string(gap) + ") is above 8 x the tolerance",
           gap > 8 * P::gfx11::kTolerance);
    // the results of the gfx11 accumulator rows 2 v + h: a layout the emulation must not have, and the probe must catch
    float w[P::kResults];
    for (unsigned i = 0; i < P::kResults; ++i) w[i] = P::gfx11::expected(i % 256);
    expect("shortcut probe: the gfx11 row table in place of gfx12's fails", P::first_mismatch_within(w, P::gfx11::kTolerance) >= 0);
  }

  // ---- the rounding probe -------------------------------------------------------------------------------------------------
  {
    std::set<uint32_t> bits;
    bool all_move = true, all_normal = true, matches_reference = true;
    unsigned ties_even = 0, ties_odd = 0;
    for (unsigned i = 0; i < R::kResults; ++i) {
      const float x = R::input(i), y = R::rounded(x);
      all_move = all_move && x != y;
      const float a = std::fabs(x);
      all_normal = all_normal && a >= std::ldexp(1.0f, -14) && a < 65504.0f;
      const float ref = reference_round(x);
      matches_reference = matches_reference && std::memcmp(&y, &ref, 4) == 0;
      uint32_t b;
      std::memcpy(&b, &x, 4);
      bits.insert(b);
      if ((b & 0x1fffu) == 0x1000u) ((b & 0x2000u) ? ties_odd : ties_even)++;
    }
    expect("rounding probe: all 256 inputs differ from their FP16 rounding (a folded pair fails every one)", all_move);
    expect("rounding probe: all inputs are FP16-normal and below the maximum", all_normal);
    expect("rounding probe: rounded() equals the independent nearest-even reference on all 256", matches_reference);
    expect("rounding probe: the 256 inputs are distinct", bits.size() == 256);
    expect("rounding probe: has exact ties that round up and ties that stay (" + std::to_string(ties_odd) + " / " + std::to_string(ties_even) + ")",
           ties_odd >= 8 && ties_even >= 8);

    float r[R::kResults];
    for (unsigned i = 0; i < R::kResults; ++i) r[i] = R::rounded(R::input(i));
    expect("rounding probe: the correctly rounded results pass", R::first_mismatch(r) < 0);
    for (unsigned i = 0; i < R::kResults; ++i) r[i] = R::input(i);
    expect("rounding probe: a folded pair (results = inputs) is reported at float 0", R::first_mismatch(r) == 0);
    for (unsigned i = 0; i < R::kResults; ++i) {   // round toward zero: the other rounding of the F16 conversion (f32tof16)
      uint32_t b;
      const float x = R::input(i);
      std::memcpy(&b, &x, 4);
      b &= ~0x1fffu;
      std::memcpy(&r[i], &b, 4);
    }
    expect("rounding probe: round-toward-zero results fail", R::first_mismatch(r) >= 0);
    for (unsigned i = 0; i < R::kResults; ++i) {   // ties away from zero
      uint32_t b;
      const float x = R::input(i);
      std::memcpy(&b, &x, 4);
      b = (b & 0x1fffu) >= 0x1000u ? (b & ~0x1fffu) + 0x2000u : (b & ~0x1fffu);
      std::memcpy(&r[i], &b, 4);
    }
    expect("rounding probe: ties-away-from-zero results fail (the tie inputs tell it from nearest-even)", R::first_mismatch(r) >= 0);
    for (unsigned i = 0; i < R::kResults; ++i) r[i] = R::rounded(R::input(i));
    r[77] = std::nextafterf(r[77], 1e9f);
    expect("rounding probe: a difference of one float ulp in float 77 is reported at 77", R::first_mismatch(r) == 77);
    r[77] = R::rounded(R::input(77));
    r[5] = std::nanf("");
    expect("rounding probe: a NaN is reported", R::first_mismatch(r) == 5);
    uint8_t buffer[R::kBytes];
    R::fill(buffer);
    float back[R::kResults], zeros[R::kResults] = {};
    std::memcpy(back, buffer + R::kIn, sizeof back);
    bool same = true;
    for (unsigned i = 0; i < R::kResults; ++i) same = same && back[i] == R::input(i);
    expect("rounding probe: fill() puts the inputs at byte 1024 and zeros at the results", same && std::memcmp(buffer, zeros, sizeof zeros) == 0);
  }

  std::printf("%s\n", failures ? "TESTS FAILED" : "ALL TESTS PASSED");
  return failures ? 1 : 0;
}
