// SPDX-License-Identifier: MIT
// Reads the file version of a Windows PE file for display and logs only (spec 08 section 3:
// the version must never be used to accept or reject a file). Written for this project from
// Microsoft's public documentation of VS_VERSIONINFO and VS_FIXEDFILEINFO.
#pragma once

#include <cstdint>
#include <string>

namespace wgc {

// Returns "a.b.c.d", or an empty string if no version resource is recognised.
std::string read_file_version(const std::uint8_t* file, std::uint64_t size);

}  // namespace wgc
