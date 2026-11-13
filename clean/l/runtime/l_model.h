// SPDX-License-Identifier: MIT
// Model L's model data in the runtime (#89): the stages and tensors of Model L in the converter's raw layout, and the
// raw loader for them (clean/runtime/model.h load_raw_stages). Names are the logical names of the default structure
// table (clean/l/reference/table.py: encoder_shapes, decoder_shapes); the ten Swin stages and post are Model M's.
//
//   enc0: Model M's W, b, T, sigma1, sigma2 (W is the projection, 64 x 128) + the two linears Wa / ba, Wb / bb + the
//         width-32 block "blk.*"
//   enc1 ... dec1: Model M's stages (clean/runtime/model.cpp stage_specs)
//   dec0: Model M's W, b, S, T + the width-32 block "blk.*"
//   post: Model M's s
#pragma once

#include <map>
#include <string>

#include "../../runtime/model.h"

namespace wgrt {
namespace l {

// The stages in load order: enc0, enc1 ... dec1 (model_stages() order), dec0, post.
StageSpecs stage_specs();

// load_raw_stages with stage_specs(): <model_dir>[/raw]/<stage>/<name>.bin.
bool load_raw_model(const std::string& model_dir, std::map<std::string, StageTensors>* out, std::string* err);

}  // namespace l
}  // namespace wgrt
