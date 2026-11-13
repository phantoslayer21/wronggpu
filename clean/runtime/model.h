// SPDX-License-Identifier: MIT
// Model data of the runtime (spec 10 section 7): loads the converter's raw tensors (spec 08 sections 7 and 10:
// raw/<stage>/<name>.bin, row-major little-endian float32, uint8 for T) and packs every stage into the kernels'
// formats (clean/FORMATS.md). The packers are C++ ports of clean/runner/packing.py, enc0.py and dec0.py and must give
// byte-identical blobs (checked by clean/tests/test_runtime.py).
#pragma once

#include <array>
#include <cstdint>
#include <map>
#include <string>
#include <utility>
#include <vector>

namespace wgrt {

using DRow = std::array<std::array<int, 8>, 2>;  // WMMA accumulator layout: element v of lane half h holds row [h][v]

struct Tensor {
  std::vector<int> shape;
  std::vector<float> f;    // float32 tensors
  std::vector<uint8_t> u;  // uint8 tensors (enc0 T)
};

using StageTensors = std::map<std::string, Tensor>;

// The stages and tensors of spec 08 section 7.2, in map order: "enc0", "enc1" ... "dec1", "dec0", "post".
const std::vector<std::string>& model_stages();

// The optional output group of spec 08 draft 2 section 7.2: "filter" with the tensor output_taps [5], the output
// filter's weights w(-2) ... w(+2) of spec 06 for an even output index. Both loaders put it into *out under this name
// when the model data has it (raw: raw/filter/output_taps.bin; model.wgcm: a stage "filter" in the index) and leave
// it out otherwise (a file written before spec 08 draft 2, or converted from a DLL whose filter block the map does
// not know). When present it is checked like every other tensor (shape, type, finite values, digest).
extern const char* const kFilterStage;  // "filter"
extern const char* const kFilterTaps;   // "output_taps"
// The five weights when the model has the group; false when it has not.
bool model_output_taps(const std::map<std::string, StageTensors>& model, float taps[5]);

// One tensor of a stage: name, shape (raw layout), uint8 (else float32).
struct TensorSpec {
  std::string name;
  std::vector<int> shape;
  bool u8;
};
using StageSpecs = std::vector<std::pair<std::string, std::vector<TensorSpec>>>;  // (stage, its tensors), in order

// Model M's tensors of one stage of model_stages() (spec 08 section 7.2).
std::vector<TensorSpec> stage_specs(const std::string& stage);

// Reads and checks every tensor (sizes, finite values, T in 0..127). model_dir may hold raw/ or be raw/ itself.
// Test path (spec 10 answers: --raw is kept for tests). Model M: load_raw_stages with model_stages() / stage_specs.
bool load_raw_model(const std::string& model_dir, std::map<std::string, StageTensors>* out, std::string* err);

// The same for any list of stages (another model's): <root>/<stage>/<name>.bin per tensor, where root is
// model_dir/raw when that holds the first tensor of the first stage, else model_dir; uint8 tensors at most 127; plus
// the optional group "filter" as above.
bool load_raw_stages(const std::string& model_dir, const StageSpecs& stages,
                     std::map<std::string, StageTensors>* out, std::string* err);

// Normal path: the converter's model.wgcm (clean/converter/OUTPUT-FORMAT.md). Checks the whole-file SHA-256, every
// stage and tensor (names, dtypes, shapes), and recomputes each stage's keyed digest (spec 08 section 7.3, with the
// key stored in the file) against the stored digest and the digest of the map at map_path (for "filter": one entry
// of the map's list). Then the same value checks as load_raw_model.
// The file also names the map it was converted with (SHA-256). A file of another map is accepted when every stage
// this runtime needs is there and equals this map's digest: that is the case for a file written with an earlier
// version of the same map (before the group "filter" existed: it has thirteen stages and no weights; spec 08
// section 7.2), and *other_map then says so. A file whose stages this map does not describe is refused ("converted
// with a different map"). Groups in the file that this map does not list are skipped.
bool load_wgcm_model(const std::string& path, const std::string& map_path, std::map<std::string, StageTensors>* out,
                     std::string* err, bool* other_map = nullptr);

// Packed weight blob of one stage (enc0, enc1..dec1, dec0); post has no blob of its own (its scalar s is passed to
// the post kernels).
bool pack_stage_blob(const std::string& stage, const StageTensors& t, const DRow& drow, std::vector<uint8_t>* out,
                     std::string* err);

// Byte offset of the merge / expand section inside a Swin stage blob (packing.stage_sections).
size_t swin_tail_offset(const std::string& stage);

struct SwinStageInfo {
  char kind;  // 'e' (enc), 'b', 'd' (dec)
  int C, heads, mlp, C_out, C_low;
};
const SwinStageInfo* swin_stage_info(const std::string& stage);

DRow drow_gfx12();

}  // namespace wgrt
