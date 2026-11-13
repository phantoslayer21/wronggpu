// SPDX-License-Identifier: MIT
// Runtime library without model K (clean/runner/build.py --no-model-k, define WG_NO_MODEL_K): this file takes the place
// of clean/k/runtime in the build. Model selection stays (model=M and the default: model M, unchanged); model=K is
// refused at create, so the front end falls back as for any other failed create (spec 10 section 8).
#ifndef WG_NO_MODEL_K
#error "no_model_k.cpp belongs to builds without model K (build.py --no-model-k)"
#endif
#include <cstdlib>
#include <string>

#include "../k/runtime/model_k.h"
#include "wg_runtime.h"

namespace wgrt {

// As clean/k/runtime/model_k.cpp: the last model= of the options and then of WG_RUNTIME_OPTIONS decides.
bool model_from_options(const WgRtConfig& c, ModelId* model, std::string* err) {
  const char* env = std::getenv("WG_RUNTIME_OPTIONS");
  const std::string text = std::string(c.runtime_options ? c.runtime_options : "") + ";" + (env ? env : "");
  *model = ModelId::M;
  bool k = false;
  for (size_t a = 0; a <= text.size();) {
    size_t b = text.find_first_of(";,", a);
    if (b == std::string::npos) b = text.size();
    const std::string kv = text.substr(a, b - a);
    a = b + 1;
    if (kv.compare(0, 6, "model=") != 0) continue;
    const std::string v = kv.substr(6);
    if (v == "M" || v == "m") {
      k = false;
    } else if (v == "K" || v == "k") {
      k = true;
    } else {
      *err = "unknown model \"" + v + "\" in the runtime options (known: M)";
      return false;
    }
  }
  if (k) {
    *err = "Model K is not part of this build (runtime option model=K); remove the option to use Model M";
    return false;
  }
  return true;
}

bool model_k_create_check(const WgRtConfig& c, std::string* err) {
  ModelId m;
  return model_from_options(c, &m, err);
}

}  // namespace wgrt
