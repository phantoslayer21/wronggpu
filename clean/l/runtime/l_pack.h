// SPDX-License-Identifier: MIT
// Model L weight packing for the runtime (#89): C++ port of clean/l/runner/l_pack.py (pack_model_l) for the default
// structure table (clean/l/reference/table.py TABLE), on top of the runtime's packing building blocks
// (clean/runtime/model.h namespace pack_parts). The blobs are byte-identical to the Python packer's
// (clean/l/runtime/tests/test_l_runtime_pack.py). Kernels, arguments and launch order: clean/l/FORMATS.md.
//
// The ten Swin stages Model L shares with Model M use Model M's packer (pack_stage_blob) unchanged; this file packs
// Model L's own launches: the encoder (values, two input linears, the width-32 block, the projection of each token's
// 2 x 2 pixels) and the decoder (projection to 32 values per pixel, the width-32 block, the coefficient functions).
// Stage and tensor names are the preset-L map's (data/preset-L.v1.map.json).
#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include "../../runtime/model.h"

namespace wgrt {
namespace l {

// One launch of Model L's encoder or decoder (l_pack.encoder_launches / decoder_launches). kind: "values", "linear",
// "swin", "project" (encoder); "dec_project", "swin", "coef" (decoder).
struct Launch {
  std::string kind, blob, src, dst;
  int C_in = 0, C_out = 0;                // linear: input / output channels
  std::string w, b, acc, out_round;       // linear, project, dec_project: tensor names; linear: roundings
  int C = 0, heads = 0, mlp = 0;          // values: channels of its pixel map; swin: width, heads, MLP
  std::string offsets, params;            // swin: window-offset kind ("first", "second"), the block's stage
  std::vector<int> groups;                // project, dec_project: input channel groups
  float clamp = 0.0f;                     // project: output clamp (0 = none)
  std::vector<int> values;                // coef: the value positions (table.L_DEC_VALUES)
  // Map formats ("fp8" / "fp16", l_pack.map_format; empty where a launch has no such map), the swin border
  // ("mirror"; "input": src is a halo map; "coarse": token-grid mirror) and the coefficient variant ("dec_pixel").
  // src and dst are distinct map identities: no kernel works in place (clean/l/FORMATS.md).
  std::string in_format, out_format, border, dec_pixel;
};

// The default table's launches, in order (mirrors of l_pack.encoder_launches(TABLE) / decoder_launches(TABLE); the
// test compares them field by field).
const std::vector<Launch>& encoder_launches();
const std::vector<Launch>& decoder_launches();

// K order of a projection: K[16 kb + 8 h + j] = 32 (kb / 2) + 16 h + 8 (kb % 2) + j (l_pack.kord_groups32).
std::vector<int> kord_groups32(int n);

// One linear blob: W [C_out][n] reading input channels lo .. lo + n of a C_in-wide map (zero weights elsewhere),
// bias [C_out] (l_pack.pack_linear).
bool pack_linear(const std::vector<float>& W, const std::vector<float>& b, int C_in, int C_out, int lo, const DRow& d,
                 std::vector<uint8_t>* out, std::string* err);
// The encoder's projection blob: W [n_out][sum(groups)], groups of 32 channels (l_pack.pack_project).
bool pack_project(const std::vector<float>& W, const std::vector<float>& b, const std::vector<int>& groups, int n_out,
                  const DRow& d, std::vector<uint8_t>* out, std::string* err);
// The decoder's projection blob: W [4][32][sum(groups)], b [4][32] (l_pack.pack_dec_project).
bool pack_dec_project(const std::vector<float>& W, const std::vector<float>& b, const std::vector<int>& groups,
                      const DRow& d, std::vector<uint8_t>* out, std::string* err);

// Every blob of Model L's own launches (l_pack.pack_model_l). `params` holds the map's stages by name: "enc0",
// "enc0_block", "dec0", "dec0_block" (float32 values in row-major order; enc0's dither table "T" as uint8 values
// 0..127 in Tensor::u, as the runtime's loader keeps it). Blob names as l_pack: "enc0.values", "enc0.Win1",
// "enc0.Win2", "enc0_block", "enc0.project", "dec0.project", "dec0_block", "dec0.coef".
bool pack_model(const std::map<std::string, StageTensors>& params, const DRow& d,
                std::map<std::string, std::vector<uint8_t>>* out, std::string* err);

}  // namespace l
}  // namespace wgrt
