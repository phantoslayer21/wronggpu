// SPDX-License-Identifier: MIT
// Loading the converted model K (spec 48 section 7) from the converter's model.wgcm (clean/converter/OUTPUT-FORMAT.md)
// for the runtime. Same checks as the model M loader (clean/runtime/model.cpp load_wgcm_model), for the K stage list:
// whole-file SHA-256, container version, SHA-256 of the map the runtime ships, every output group's keyed digest
// (spec 48 section 7.4, key stored in the file) against the stored digest and the map's, every tensor of spec 48
// sections 7.2 / 7.3 with its dtype and shape, finite float32 values, window offsets in 0..7.
//
// Needs clean/converter/src/sha256.cpp and json.cpp (the runtime links both already).
#pragma once

#include <map>
#include <string>
#include <vector>

#include "../../runtime/model.h"  // wgrt::Tensor, wgrt::StageTensors (header-only types)

namespace wgrt {

struct KTensorSpec {
  std::string name;
  std::vector<int> shape;
};

// The float32 tensors of one K stage (e0 ... d0), in map order (spec 48 section 7.2). Empty for an unknown stage.
std::vector<KTensorSpec> k_stage_tensor_specs(const std::string& stage);

// The output groups of a K model.wgcm, in map order: the eleven stages of spec 40 and "net" (spec 48 section 7.3).
const std::vector<std::string>& k_model_groups();

struct KModel {
  std::map<std::string, StageTensors> stages;  // "e0" ... "d0" -> tensor name -> Tensor (float32 in .f)
  int window_offsets[8][2] = {};               // net/window_offsets: row f mod 8, columns (ox, oy) of stage m
};

// Reads and checks path (model.wgcm) against map_path (the preset-K map shipped with the runtime). On failure returns
// false with a message in *err and leaves *out unchanged.
bool load_k_model(const std::string& path, const std::string& map_path, KModel* out, std::string* err);
// The same with the map's bytes already read (the runtime reads the map through embedded.h: on disk, else built in);
// map_path only names it in messages.
bool load_k_model_map(const std::string& path, const std::vector<uint8_t>& map_bytes, const std::string& map_path,
                      KModel* out, std::string* err);

}  // namespace wgrt
