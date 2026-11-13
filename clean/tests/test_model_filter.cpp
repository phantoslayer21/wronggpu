// SPDX-License-Identifier: MIT
// Model loading of the optional group filter / output_taps (spec 08 draft 2), without a GPU: loads a model with
// clean/runtime/model.cpp and prints what the loader found. Driven by clean/tests/test_model_filter.py.
//
//   test_model_filter raw <model dir>
//   test_model_filter wgcm <model.wgcm> <map file>
//
// Output, one line: "OK none" (no group), "OK <five float32 bit patterns in hex>" or "ERR <the loader's message>";
// an accepted file that was converted with another map than the one given ends in " other-map".
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <map>
#include <string>

#include "../runtime/model.h"

int main(int argc, char** argv) {
  if (argc < 3) {
    std::printf("ERR usage\n");
    return 2;
  }
  const std::string mode = argv[1];
  std::map<std::string, wgrt::StageTensors> model;
  std::string err;
  bool ok = false, other_map = false;
  if (mode == "raw") {
    ok = wgrt::load_raw_model(argv[2], &model, &err);
  } else if (mode == "wgcm" && argc >= 4) {
    ok = wgrt::load_wgcm_model(argv[2], argv[3], &model, &err, &other_map);
  } else {
    err = "usage";
  }
  if (!ok) {
    std::printf("ERR %s\n", err.c_str());
    return 0;
  }
  for (const std::string& s : wgrt::model_stages())
    if (!model.count(s)) {
      std::printf("ERR stage %s not loaded\n", s.c_str());
      return 0;
    }
  float taps[5];
  if (!wgrt::model_output_taps(model, taps)) {
    std::printf("OK none%s\n", other_map ? " other-map" : "");
    return 0;
  }
  std::printf("OK");
  for (int k = 0; k < 5; ++k) {
    uint32_t b;
    std::memcpy(&b, &taps[k], 4);
    std::printf(" %08x", b);
  }
  std::printf("%s\n", other_map ? " other-map" : "");
  return 0;
}
