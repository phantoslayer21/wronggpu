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
  auto pick = [&](const char* value, const char* n) { return wgrt::dev_gfx11_native_kernel(std::string(value), n, has); };

  expect("off: the port runs (manifest has the native form)", pick("", "k_stage_e2") == "k_stage_e2");
  // the same call with the switch on: the native form exactly in a build with the macro, never otherwise
  expect("on: native form only in a WG_DEV_GFX11 build", pick("1", "k_stage_e2") == (kDevGfx11 ? "k_stage_e2__g11" : "k_stage_e2"));
  expect("on: the round-to-odd form takes its own native form",
         pick("1", "k_stage_e2__rodd") == (kDevGfx11 ? "k_stage_e2__rodd__g11" : "k_stage_e2__rodd"));
  expect("on: a kernel the manifest has no __g11 for keeps its name", pick("1", "k_stage_d2") == "k_stage_d2");
  expect("on: a helper with a plain gfx11 entry keeps its name", pick("1", "native_fill") == "native_fill");
  expect("on: a name that is not a K stage is untouched", pick("1", "k_warm") == "k_warm");
  expect("on: two-window head form", pick("1", "k_stage2_d0_hp") == (kDevGfx11 ? "k_stage2_d0_hp__g11" : "k_stage2_d0_hp"));
  expect("on: a native name is not looked up again", pick("1", "k_stage_e2__g11") == "k_stage_e2__g11");
  expect("on: empty name", pick("1", "") == "");
  expect("negative control: an empty manifest never gives a native name",
         wgrt::dev_gfx11_native_kernel(std::string("1"), std::string("k_stage_e2"), [](const std::string&) { return false; }) == "k_stage_e2");
  expect("a name with a prefix before __g11 only counts at the end", pick("1", "k__g11x") == "k__g11x");

  // the list form: only the named ports switch (whole names), in a build with the macro
  const char* const list = "k_stage2_d0_hp,k_stage2_d1,k_stage_d2,k_stage_e2";
  auto g = [&](const char* n) { return std::string(n) + "__g11"; };
  expect("list: a listed port switches", pick(list, "k_stage_e2") == (kDevGfx11 ? g("k_stage_e2") : "k_stage_e2"));
  expect("list: the first and the last entry switch",
         pick(list, "k_stage2_d0_hp") == (kDevGfx11 ? g("k_stage2_d0_hp") : "k_stage2_d0_hp") &&
             pick(list, "k_stage_e2") == (kDevGfx11 ? g("k_stage_e2") : "k_stage_e2"));
  expect("list negative control: a port not in the list does not switch (manifest has k_stage_e2__rodd__g11 only for e2 rodd)",
         pick(list, "k_stage_e2__rodd") == "k_stage_e2__rodd");
  expect("list negative control: a port the manifest has, not in the list, stays", pick("k_stage_d2", "k_stage_e2") == "k_stage_e2");
  expect("list: a prefix of a listed name does not match", pick("k_stage_e2__rodd", "k_stage_e2") == "k_stage_e2");
  expect("list: a listed name with no __g11 in the manifest keeps its name", pick(list, "k_stage_d2") == "k_stage_d2");
  expect("list: an empty entry or stray commas select nothing extra", pick(",,k_stage_d2,", "k_stage_e2") == "k_stage_e2");
  expect("list: a single listed name", pick("k_stage_e2", "k_stage_e2") == (kDevGfx11 ? g("k_stage_e2") : "k_stage_e2"));
  expect("selection: 0 selects nothing", pick("0", "k_stage_e2") == "k_stage_e2");
#ifdef WG_DEV_GFX11
  expect("selects: 1 selects every name", wgrt::dev_gfx11_native_k_selects("1", "anything"));
  expect("selects: a list selects only its names", wgrt::dev_gfx11_native_k_selects("a,b", "b") && !wgrt::dev_gfx11_native_k_selects("a,b", "c") &&
                                                      !wgrt::dev_gfx11_native_k_selects("a,b", "") && !wgrt::dev_gfx11_native_k_selects("a,b", "a,b"));
#endif

  set_env("WG_DEV_GFX11_NATIVE_K", nullptr);
  expect("the variable unset: off", !wgrt::dev_gfx11_native_k());
  set_env("WG_DEV_GFX11_NATIVE_K", "1");
  expect("the variable 1: on exactly in a build with the macro", wgrt::dev_gfx11_native_k() == kDevGfx11);
  set_env("WG_DEV_GFX11_NATIVE_K", "0");
  expect("the variable 0: off", !wgrt::dev_gfx11_native_k());
  set_env("WG_DEV_GFX11_NATIVE_K", "");
  expect("the variable empty: off", !wgrt::dev_gfx11_native_k());
  set_env("WG_DEV_GFX11_NATIVE_K", list);
  expect("the variable a list: on exactly in a build with the macro", wgrt::dev_gfx11_native_k() == kDevGfx11);
#ifdef WG_DEV_GFX11
  expect("the variable a list: its value is read back whole", wgrt::dev_gfx11_native_k_value() == list);
#endif
  set_env("WG_DEV_GFX11_NATIVE_K", nullptr);

  std::printf(failures ? "%d FAILED\n" : "ALL TESTS PASSED\n", failures);
  return failures ? 1 : 0;
}
