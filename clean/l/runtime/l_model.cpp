// SPDX-License-Identifier: MIT
// Model L's model data in the runtime (see l_model.h). Checked against the preset-L map and the structure table by
// clean/l/runtime/tests/test_l_runtime_load.py.
#include "l_model.h"

namespace wgrt {
namespace l {

namespace {

const int kHeadDim = 32, kWindowTokens = 16;

// One Swin block's tensors (table.block_shapes; the map's block record set).
std::vector<TensorSpec> block_specs(int C, int heads, int mlp) {
  return {{"gamma1", {C}, false},
          {"gamma2", {C}, false},
          {"Wqk", {heads, kHeadDim, C}, false},
          {"Wv", {heads, kHeadDim, C}, false},
          {"B", {heads, kWindowTokens, kWindowTokens}, false},
          {"Wp", {C, kHeadDim * heads}, false},
          {"bp", {C}, false},
          {"W1", {mlp, C}, false},
          {"b1", {mlp}, false},
          {"W2", {C, mlp}, false},
          {"b2", {C}, false}};
}

}  // namespace

StageSpecs stage_specs() {
  StageSpecs s;
  // enc0: the projection reads the 2 x 2 pixels of 32 channels of each token (128 inputs); the first input linear
  // reads the 20 values of a pixel, the second 32.
  s.emplace_back("enc0", std::vector<TensorSpec>{{"W", {64, 128}, false},
                                                 {"b", {64}, false},
                                                 {"T", {8, 16, 16}, true},
                                                 {"sigma1", {1}, false},
                                                 {"sigma2", {1}, false},
                                                 {"Win1", {32, 20}, false},
                                                 {"bin1", {32}, false},
                                                 {"Win2", {32, 32}, false},
                                                 {"bin2", {32}, false}});
  s.emplace_back("enc0_block", block_specs(32, 2, 128));
  for (const std::string& st : model_stages())
    if (st != "enc0" && st != "dec0" && st != "post") s.emplace_back(st, wgrt::stage_specs(st));
  s.emplace_back("dec0_block", block_specs(32, 2, 128));
  // dec0: 32 outputs per sub-pixel (Model M: 17), then Model M's scalars
  s.emplace_back("dec0", std::vector<TensorSpec>{
                             {"W", {4, 32, 64}, false}, {"b", {4, 32}, false}, {"S", {1}, false}, {"T", {1}, false}});
  s.emplace_back("post", wgrt::stage_specs("post"));
  return s;
}

bool load_raw_model(const std::string& model_dir, std::map<std::string, StageTensors>* out, std::string* err) {
  return load_raw_stages(model_dir, stage_specs(), out, err);
}

bool load_wgcm_model(const std::string& path, const std::string& map_path, std::map<std::string, StageTensors>* out,
                     std::string* err, bool* other_map) {
  return load_wgcm_stages(path, map_path, stage_specs(), out, err, other_map);
}

}  // namespace l
}  // namespace wgrt
