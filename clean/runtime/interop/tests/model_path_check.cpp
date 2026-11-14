// SPDX-License-Identifier: MIT
// The converted model file each model resolves to (clean/runtime/model_select.h model_wgcm_path, used by every
// executor's create check in interop/backend.cpp): Model M unchanged, Model L in its own folder. CPU only.
// Run by test_model_path.py. WrongGPU's own code.
#include <cstdio>
#include <string>

#include "../../model_select.h"

int main() {
  int failures = 0;
  auto expect = [&](const char* name, const std::string& got, const std::string& want) {
    const bool ok = got == want;
    std::printf("%s %s: %s\n", ok ? "pass" : "FAIL", name, got.c_str());
    if (!ok) ++failures;
  };
  using wgrt::ModelId;
  const std::string d = "C:\\Games\\X\\bin\\x64\\WrongGPU\\model";
  expect("model=M", wgrt::model_wgcm_path(d, ModelId::M), d + "\\model.wgcm");
  expect("model=L", wgrt::model_wgcm_path(d, ModelId::L), d + "\\L\\model.wgcm");
  expect("model=K", wgrt::model_wgcm_path(d, ModelId::K), d + "\\K\\model.wgcm");
  expect("model=L, '/'", wgrt::model_wgcm_path("/m", ModelId::L, '/'), "/m/L/model.wgcm");
  // the option the front end passes resolves to the same model
  ModelId id = ModelId::M;
  std::string bad;
  const bool parsed = wgrt::last_model_option("view=upscale;model=L;executor=dx12", &id, &bad);
  expect("model=L option", parsed && id == ModelId::L ? "L" : "other", "L");
  id = ModelId::M;
  wgrt::last_model_option("view=upscale;executor=dx12", &id, &bad);
  expect("no model option", id == ModelId::M ? "M" : "other", "M");
  return failures ? 1 : 0;
}
