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
#include "model_select.h"
#include "wg_runtime.h"

namespace wgrt {

// As clean/k/runtime/model_k.cpp: the last model= of the options and then of WG_RUNTIME_OPTIONS decides.
bool model_from_options(const WgRtConfig& c, ModelId* model, std::string* err) {
  const char* env = std::getenv("WG_RUNTIME_OPTIONS");
  const std::string text = std::string(c.runtime_options ? c.runtime_options : "") + ";" + (env ? env : "");
  *model = ModelId::M;
  ModelId last = ModelId::M;
  std::string bad;
  if (!last_model_option(text, &last, &bad)) {
    *err = "unknown model \"" + bad + "\" in the runtime options (known: M" + (model_l_built() ? ", L)" : ")");
    return false;
  }
  if (last == ModelId::K) {
    *err = "Model K is not part of this build (runtime option model=K); remove the option to use Model M";
    return false;
  }
  if (last == ModelId::L && !model_l_built()) {  // #89: recognised; runs when its sources are compiled in
    *err = model_not_built_yet(ModelId::L);
    return false;
  }
  *model = last;
  return true;
}

bool model_k_create_check(const WgRtConfig& c, std::string* err) {
  ModelId m;
  return model_from_options(c, &m, err);
}

}  // namespace wgrt
