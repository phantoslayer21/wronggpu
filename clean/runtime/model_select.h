// SPDX-License-Identifier: MIT
// Model selection shared by the runtime, the D3D12 interop and the builds with and without model K (#89): the model
// table (name, converter map, folder under the model directory, native D3D12 support) and the parser of the runtime
// option model=<name>. Model M is the default and the first row; a model's row says what it needs from its
// surroundings, so the callers do not compare against one model by hand. Header only: every build that compiles
// clean/k/runtime/model_k.cpp or clean/runtime/no_model_k.cpp gets it without a new source file.
//
// Model L has a row so that model=L is recognised. The runtime runs it when its sources are compiled in (define
// WG_MODEL_L, model_l_built()); otherwise model=L is refused by name ("not part of this build yet").
#pragma once

#include <cstddef>
#include <string>

#include "../k/runtime/model_k.h"  // ModelId

namespace wgrt {

struct ModelInfo {
  ModelId id;
  const char* name;          // "M", "K", "L": the runtime option model=<name> (either case) and runtime_model()
  const char* map_file;      // the converter map of the model's preset, next to the runtime library
  const char* model_subdir;  // the model's folder under the model directory ("" = the model directory itself)
  bool native_d3d12;         // runs on the native D3D12 backend (backend 3)
};

constexpr ModelInfo kModelTable[] = {
    {ModelId::M, "M", "preset-M.v1.map.json", "", true},
    {ModelId::K, "K", "preset-K.v1.map.json", "K", false},
    {ModelId::L, "L", "preset-L.v1.map.json", "L", false},
};
constexpr size_t kModelCount = sizeof kModelTable / sizeof kModelTable[0];

inline const ModelInfo& model_info(ModelId id) {
  for (const ModelInfo& m : kModelTable)
    if (m.id == id) return m;
  return kModelTable[0];
}

// A model name in either case ("M", "m", "K", ...). False: no such model.
inline bool model_from_name(const std::string& v, ModelId* id) {
  for (const ModelInfo& m : kModelTable) {
    const char c = m.name[0];
    if (v.size() == 1 && (v[0] == c || v[0] == c - 'A' + 'a')) {
      *id = m.id;
      return true;
    }
  }
  return false;
}

// The last model=<value> of `text` (options separated by ';' or ','; no model= leaves *model as it is). False at the
// first unknown value (*bad = that value).
inline bool last_model_option(const std::string& text, ModelId* model, std::string* bad) {
  for (size_t a = 0; a <= text.size();) {
    size_t b = text.find_first_of(";,", a);
    if (b == std::string::npos) b = text.size();
    const std::string kv = text.substr(a, b - a);
    a = b + 1;
    if (kv.compare(0, 6, "model=") != 0) continue;
    const std::string v = kv.substr(6);
    if (!model_from_name(v, model)) {
      *bad = v;
      return false;
    }
  }
  return true;
}

// Model L is part of this runtime library: its sources (clean/l/runtime) compiled with the define WG_MODEL_L.
inline bool model_l_built() {
#ifdef WG_MODEL_L
  return true;
#else
  return false;
#endif
}

// The refusal of model=L when it is not part of the runtime library (one place for both builds).
inline std::string model_not_built_yet(ModelId id) {
  return std::string("Model ") + model_info(id).name + " is not part of this build yet (runtime option model=" +
         model_info(id).name + "); remove the option to use Model M";
}

// "Model M" (or "Models M, K" ...): the models the native D3D12 backend runs, for its messages.
inline std::string native_d3d12_models() {
  std::string names;
  int n = 0;
  for (const ModelInfo& m : kModelTable)
    if (m.native_d3d12) names += std::string(n++ ? ", " : "") + m.name;
  return std::string(n > 1 ? "Models " : "Model ") + names;
}

}  // namespace wgrt
