// SPDX-License-Identifier: MIT
#pragma once
// Frame generation (issue #96): checks a local FG model file (the converter's model.wgcm for the preset-FG map,
// container version 1; clean/converter/OUTPUT-FORMAT.md) against data/fg.v1.map.json before anything is prepared from
// it: the whole-file SHA-256, the map hash in the header, exactly the map's 42 stages (17 net1 + 25 net2 layers) in
// natural order with their tensor names, types and shapes, every stage digest recomputed with the file's key and
// compared with the map's, and every value exact in float16. Any difference is a refusal with its reason; nothing is
// prepared from a file that did not pass. Plain C++.
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "fg_contract.h"
#include "fg_map_table.h"

namespace wgfg {

// What a model file must match. The shipped expectation is the generated map table; tests build their own from
// constructed models (same names and shapes, their own digests and map hash).
struct FgModelExpectation {
    const FgMapStage* stages = nullptr;
    uint32_t count = 0;
    const char* map_sha256 = nullptr;  // hex
};
FgModelExpectation FgShippedModelExpectation();

// A tensor of a validated file: float32 little-endian data inside the caller's bytes.
struct FgModelTensor {
    uint32_t rank = 0;
    uint32_t shape[4] = {};
    uint64_t offset = 0, bytes = 0;
};
struct FgModelStage {
    std::string name;
    FgModelTensor w, b;
};
struct FgModelReport {
    FgRefusal refusal = FgRefusal::ModelMissing;
    int stage = -1;                    // index of the failing stage, -1 when not stage-specific
    std::string detail;
    std::vector<FgModelStage> stages;  // filled when the file passed
};

// Checks `size` bytes of a model file. FgRefusal::None when every check passed (report->stages then indexes it).
FgRefusal FgValidateModelBytes(const uint8_t* data, size_t size, const FgModelExpectation& expect,
                               FgModelReport* report);
// The same for a file on disk (read whole; ModelMissing when it cannot be read).
FgRefusal FgValidateModelFile(const std::string& path, const FgModelExpectation& expect, FgModelReport* report,
                              std::vector<uint8_t>* bytes_out);

// float16 helpers (bit patterns).
float FgHalfToFloat(uint16_t h);
// True when f is exactly a float16 value (NaN: false; +-inf and +-0: true); *out gets its bits.
bool FgFloatToHalfExact(float f, uint16_t* out);

}  // namespace wgfg
