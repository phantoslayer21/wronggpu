// SPDX-License-Identifier: MIT
// The developer-only native Model K name lookup of the WG_DEV_GFX11 opening (runtime/rdna4.h): which kernel runs for a requested
// name, and the environment variable that turns it on. WrongGPU's own code. Built twice, both builds must pass:
//   clang++ -std=c++17 test_native_k_name.cpp -o test_native_k_name.exe                  (a normal build: never a __g11 name)
//   clang++ -std=c++17 -DWG_DEV_GFX11 test_native_k_name.cpp -o test_native_k_name_dev.exe
// clean/tests/test_dev_gfx11_native_k.py builds and runs both. Exit status 0 when every case passed.
#include <cstdio>
#include <cstdlib>
#include <set>
#include <string>

#include "../rdna4.h"

namespace {

int failures = 0;

void expect(const std::string& name, bool ok) {
  std::printf("%s %s\n", ok ? "pass" : "FAIL", name.c_str());
  if (!ok) ++failures;
}

void set_env(const char* name, const char* value) {
#if defined(_WIN32)
  _putenv_s(name, value ? value : "");
#else
  if (value) setenv(name, value, 1); else unsetenv(name);
#endif
}

}  // namespace

int main() {
  using wgrt::kDevGfx11;
  // a manifest as the dx12-gfx11 folder has it: some K stages natively, helpers not
  const std::set<std::string> manifest = {"k_stage_e2__g11", "k_stage_e2__rodd__g11", "k_stage2_d0_hp__g11", "native_fill"};
  auto has = [&](const std::string& n) { return manifest.count(n) != 0; };
  auto pick = [&](bool on, const char* n) { return wgrt::dev_gfx11_native_kernel(on, n, has); };

  expect("off: the port runs (manifest has the native form)", pick(false, "k_stage_e2") == "k_stage_e2");
  // the same call with the switch on: the native form exactly in a build with the macro, never otherwise
  expect("on: native form only in a WG_DEV_GFX11 build", pick(true, "k_stage_e2") == (kDevGfx11 ? "k_stage_e2__g11" : "k_stage_e2"));
  expect("on: the round-to-odd form takes its own native form",
         pick(true, "k_stage_e2__rodd") == (kDevGfx11 ? "k_stage_e2__rodd__g11" : "k_stage_e2__rodd"));
  expect("on: a kernel the manifest has no __g11 for keeps its name", pick(true, "k_stage_d2") == "k_stage_d2");
  expect("on: a helper with a plain gfx11 entry keeps its name", pick(true, "native_fill") == "native_fill");
  expect("on: a name that is not a K stage is untouched", pick(true, "k_warm") == "k_warm");
  expect("on: two-window head form", pick(true, "k_stage2_d0_hp") == (kDevGfx11 ? "k_stage2_d0_hp__g11" : "k_stage2_d0_hp"));
  expect("on: a native name is not looked up again", pick(true, "k_stage_e2__g11") == "k_stage_e2__g11");
  expect("on: empty name", pick(true, "") == "");
  expect("negative control: an empty manifest never gives a native name",
         wgrt::dev_gfx11_native_kernel(true, std::string("k_stage_e2"), [](const std::string&) { return false; }) == "k_stage_e2");
  expect("a name with a prefix before __g11 only counts at the end", pick(true, "k__g11x") == "k__g11x");

  set_env("WG_DEV_GFX11_NATIVE_K", nullptr);
  expect("the variable unset: off", !wgrt::dev_gfx11_native_k());
  set_env("WG_DEV_GFX11_NATIVE_K", "1");
  expect("the variable 1: on exactly in a build with the macro", wgrt::dev_gfx11_native_k() == kDevGfx11);
  set_env("WG_DEV_GFX11_NATIVE_K", "0");
  expect("the variable 0: off", !wgrt::dev_gfx11_native_k());
  set_env("WG_DEV_GFX11_NATIVE_K", "11");
  expect("the variable 11: off", !wgrt::dev_gfx11_native_k());
  set_env("WG_DEV_GFX11_NATIVE_K", "true");
  expect("the variable true: off", !wgrt::dev_gfx11_native_k());
  set_env("WG_DEV_GFX11_NATIVE_K", nullptr);

  std::printf(failures ? "%d FAILED\n" : "ALL TESTS PASSED\n", failures);
  return failures ? 1 : 0;
}
