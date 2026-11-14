// SPDX-License-Identifier: MIT
// The runtime's model selection (clean/runtime/model_select.h, model_from_options of clean/k/runtime/model_k.cpp and
// clean/runtime/no_model_k.cpp): the model table, and the runtime option model=<name> with the exact messages the
// runtime gave before the table existed (#89). Built twice by clean/tests/test_model_select.py: with model K
// (model_k.cpp) and with -DWG_NO_MODEL_K (no_model_k.cpp). Prints one line per failed check; exit code = failures.
#include <cstdio>
#include <cstdlib>
#include <string>

#include "../k/runtime/model_k.h"
#include "../runtime/model_select.h"
#include "../runtime/wg_runtime.h"

namespace wgrt {
bool model_from_options(const WgRtConfig& c, ModelId* model, std::string* err);
}

static int g_fail = 0;
#define EXPECT(c)                                                     \
  do {                                                                \
    if (!(c)) {                                                       \
      ++g_fail;                                                       \
      std::printf("FAILED line %d: %s\n", __LINE__, #c);              \
    }                                                                 \
  } while (0)

// model_from_options for one options string: (ok, model, message)
static bool select(const char* opts, wgrt::ModelId* m, std::string* err) {
  WgRtConfig c = WgRtConfig();
  c.runtime_options = opts;
  err->clear();
  *m = wgrt::ModelId::K;  // overwritten by every call
  return wgrt::model_from_options(c, m, err);
}

int main() {
  using wgrt::ModelId;
#ifdef _WIN32
  _putenv("WG_RUNTIME_OPTIONS=");
#else
  unsetenv("WG_RUNTIME_OPTIONS");
#endif
  // ---- the table: Model M first, with the names the runtime and the interop used before
  EXPECT(wgrt::kModelTable[0].id == ModelId::M);
  const wgrt::ModelInfo& m = wgrt::model_info(ModelId::M);
  const wgrt::ModelInfo& k = wgrt::model_info(ModelId::K);
  const wgrt::ModelInfo& l = wgrt::model_info(ModelId::L);
  EXPECT(std::string(m.name) == "M" && std::string(m.map_file) == "preset-M.v1.map.json" &&
         std::string(m.model_subdir).empty() && m.native_d3d12);
  EXPECT(std::string(k.name) == "K" && std::string(k.map_file) == "preset-K.v1.map.json" &&
         std::string(k.model_subdir) == "K" && !k.native_d3d12);
  EXPECT(std::string(l.name) == "L" && std::string(l.map_file) == "preset-L.v1.map.json" &&
         std::string(l.model_subdir) == "L" && !l.native_d3d12);
  EXPECT(wgrt::native_d3d12_models() == "Model M");  // "native D3D12 supports Model M only", as before
  ModelId id = ModelId::K;
  EXPECT(wgrt::model_from_name("m", &id) && id == ModelId::M);
  EXPECT(wgrt::model_from_name("L", &id) && id == ModelId::L);
  EXPECT(!wgrt::model_from_name("MM", &id) && !wgrt::model_from_name("", &id) && !wgrt::model_from_name("x", &id));

  // ---- the option, Model M (both builds): the default and model=M select Model M
  std::string err;
  EXPECT(select(nullptr, &id, &err) && id == ModelId::M && err.empty());
  EXPECT(select("", &id, &err) && id == ModelId::M);
  EXPECT(select("model=M", &id, &err) && id == ModelId::M);
  EXPECT(select("model=m;network=fused", &id, &err) && id == ModelId::M);
  EXPECT(select("a=1,model=K;model=M", &id, &err) && id == ModelId::M);  // the last one decides
  EXPECT(select("modelx=K", &id, &err) && id == ModelId::M);              // not the model option
#ifndef WG_NO_MODEL_K
  EXPECT(select("model=K", &id, &err) && id == ModelId::K);
  EXPECT(select("model=k", &id, &err) && id == ModelId::K);
  EXPECT(!select("model=X", &id, &err) && err == "unknown model \"X\" in the runtime options (known: M, K)");
  EXPECT(!select("model=", &id, &err) && err == "unknown model \"\" in the runtime options (known: M, K)");
#else
  EXPECT(!select("model=K", &id, &err) &&
         err == "Model K is not part of this build (runtime option model=K); remove the option to use Model M");
  EXPECT(!select("model=X", &id, &err) && err == "unknown model \"X\" in the runtime options (known: M)");
  EXPECT(!select("model=", &id, &err) && err == "unknown model \"\" in the runtime options (known: M)");
#endif
  // ---- model=L: recognised and refused by name until its network exists (#89)
  const std::string not_yet = "Model L is not part of this build yet (runtime option model=L); remove the option to "
                              "use Model M";
  EXPECT(!select("model=L", &id, &err) && err == not_yet);
  EXPECT(!select("model=l", &id, &err) && err == not_yet);
  EXPECT(select("model=L;model=M", &id, &err) && id == ModelId::M);  // the last one decides, as for K

  std::printf("test_model_select (%s): %d failure(s)\n",
#ifdef WG_NO_MODEL_K
              "no model K",
#else
              "with model K",
#endif
              g_fail);
  return g_fail;
}
