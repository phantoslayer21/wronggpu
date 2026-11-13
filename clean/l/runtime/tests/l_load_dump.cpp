// SPDX-License-Identifier: MIT
// Model L's raw model loading (clean/l/runtime/l_model.cpp) on the CPU, for test_l_runtime_load.py:
//
//   l_load_dump specs
//       one line per tensor of wgrt::l::stage_specs(): "<stage> <name> <dims joined by x> <u8|f32>"
//   l_load_dump load <model dir>
//       wgrt::l::load_raw_model; prints "OK <stages>" and one line per loaded tensor, "<stage> <name> <values>
//       <sha256 of the values' bytes>", or "ERR <message>"
#include <cstdio>
#include <map>
#include <string>

#include "../../../converter/src/sha256.h"
#include "../l_model.h"

int main(int argc, char** argv) {
  const std::string mode = argc > 1 ? argv[1] : "";
  if (mode == "specs") {
    for (const auto& [stage, specs] : wgrt::l::stage_specs())
      for (const wgrt::TensorSpec& t : specs) {
        std::string dims;
        for (int d : t.shape) dims += (dims.empty() ? "" : "x") + std::to_string(d);
        std::printf("%s %s %s %s\n", stage.c_str(), t.name.c_str(), dims.c_str(), t.u8 ? "u8" : "f32");
      }
    return 0;
  }
  if (mode == "load" && argc > 2) {
    std::map<std::string, wgrt::StageTensors> model;
    std::string err;
    if (!wgrt::l::load_raw_model(argv[2], &model, &err)) return std::printf("ERR %s\n", err.c_str()), 0;
    std::printf("OK %zu\n", model.size());
    for (const auto& [stage, tensors] : model)
      for (const auto& [name, t] : tensors) {
        const wgc::Digest d = t.u.empty() ? wgc::Sha256::hash(t.f.data(), t.f.size() * 4)
                                          : wgc::Sha256::hash(t.u.data(), t.u.size());
        std::printf("%s %s %zu %s\n", stage.c_str(), name.c_str(), t.u.empty() ? t.f.size() : t.u.size(),
                    wgc::to_hex(d).c_str());
      }
    return 0;
  }
  std::printf("ERR usage\n");
  return 2;
}
