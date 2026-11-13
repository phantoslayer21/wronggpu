// SPDX-License-Identifier: MIT
// Writing the converter output (spec 08 sections 8 and 10). Written for this project.
// The container format is described in clean/converter/OUTPUT-FORMAT.md.
#pragma once

#include <filesystem>
#include <string>
#include <vector>

#include "convert.h"
#include "map.h"

namespace wgc {

constexpr const char* kModelFileName = "model.wgcm";

// Serialises the converted stages into the container format (whole file, trailer included).
std::vector<std::uint8_t> build_container(const Map& map, const std::vector<StageOut>& stages, const Digest& key);

// Writes everything to a staging directory inside out_dir, then moves it into place. On failure
// the staging directory is removed and any earlier output in out_dir is left as it was.
bool write_output(const std::filesystem::path& out_dir, const Map& map, const std::vector<StageOut>& stages,
                  const Digest& key, bool raw, std::string& error);

}  // namespace wgc
