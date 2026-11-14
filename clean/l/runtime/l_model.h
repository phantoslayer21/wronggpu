// SPDX-License-Identifier: MIT
// Model L's model data in the runtime (#89): the stages and tensors of Model L in the converter's raw layout, and the
// raw loader for them (clean/runtime/model.h load_raw_stages). Stages, names and shapes are those of the preset-L map
// (data/preset-L.v1.map.json), which the structure table (clean/l/reference/table.py param_shapes) groups the same way:
//
//   enc0: W (the projection, 64 x 128), b, T, sigma1, sigma2, and the two input linears Win1 / bin1 (32 x 20),
//         Win2 / bin2 (32 x 32)
//   enc0_block: the encoder's width-32 Swin block
//   enc1 ... dec1: Model M's stages (clean/runtime/model.cpp stage_specs)
//   dec0_block: the decoder's width-32 Swin block
//   dec0: W (4 x 32 x 64), b (4 x 32), S, T
//   post: Model M's s
#pragma once

#include <map>
#include <string>

#include "../../runtime/model.h"

namespace wgrt {
namespace l {

// The stages in the map's order: enc0, enc0_block, enc1 ... dec1, dec0_block, dec0, post.
StageSpecs stage_specs();

// load_raw_stages with stage_specs(): <model_dir>[/raw]/<stage>/<name>.bin.
bool load_raw_model(const std::string& model_dir, std::map<std::string, StageTensors>* out, std::string* err);

// load_wgcm_stages with stage_specs(): the converter's model.wgcm, checked against the preset-L map at map_path
// (whole-file hash, every stage's keyed digest, names, types, shapes, values; clean/runtime/model.h).
bool load_wgcm_model(const std::string& path, const std::string& map_path, std::map<std::string, StageTensors>* out,
                     std::string* err, bool* other_map = nullptr);

}  // namespace l
}  // namespace wgrt
