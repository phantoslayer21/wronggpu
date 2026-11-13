// SPDX-License-Identifier: MIT
// Model L's model data in the runtime (see l_model.h). Shapes mirror clean/l/reference/table.py for the default table
// (checked by clean/l/runtime/tests/test_l_runtime_load.py).
#include "l_model.h"

namespace wgrt {
namespace l {

namespace {

const int kHeadDim = 32, kWindowTokens = 16;

// One Swin block's tensors with a name prefix (table.block_shapes).
void add_block(std::vector<TensorSpec>* v, const std::string& prefix, int C, int heads, int mlp) {
  const std::vector<TensorSpec> b = {{prefix + "gamma1", {C}, false},
                                     {prefix + "gamma2", {C}, false},
                                     {prefix + "Wqk", {heads, kHeadDim, C}, false},
                                     {prefix + "Wv", {heads, kHeadDim, C}, false},
                                     {prefix + "B", {heads, kWindowTokens, kWindowTokens}, false},
                                     {prefix + "Wp", {C, kHeadDim * heads}, false},
                                     {prefix + "bp", {C}, false},
                                     {prefix + "W1", {mlp, C}, false},
                                     {prefix + "b1", {mlp}, false},
                                     {prefix + "W2", {C, mlp}, false},
                                     {prefix + "b2", {C}, false}};
  v->insert(v->end(), b.begin(), b.end());
}

}  // namespace

StageSpecs stage_specs() {
  StageSpecs s;
  // enc0: the projection reads [raw (80) | x (32) | 16 zeros] = 128 inputs; the first linear reads the 20 values of
  // one sub-pixel (table: select 0..20, linear 20 -> 32), the second 32 -> 32; then the width-32 block.
  std::vector<TensorSpec> enc0 = {{"W", {64, 128}, false}, {"b", {64}, false},      {"T", {8, 16, 16}, true},
                                  {"sigma1", {1}, false},  {"sigma2", {1}, false},  {"Wa", {32, 20}, false},
                                  {"ba", {32}, false},     {"Wb", {32, 32}, false}, {"bb", {32}, false}};
  add_block(&enc0, "blk.", 32, 2, 128);
  s.emplace_back("enc0", enc0);
  for (const std::string& st : model_stages())
    if (st != "enc0" && st != "dec0" && st != "post") s.emplace_back(st, wgrt::stage_specs(st));
  std::vector<TensorSpec> dec0 = wgrt::stage_specs("dec0");
  add_block(&dec0, "blk.", 32, 2, 128);
  s.emplace_back("dec0", dec0);
  s.emplace_back("post", wgrt::stage_specs("post"));
  return s;
}

bool load_raw_model(const std::string& model_dir, std::map<std::string, StageTensors>* out, std::string* err) {
  return load_raw_stages(model_dir, stage_specs(), out, err);
}

}  // namespace l
}  // namespace wgrt
