// SPDX-License-Identifier: MIT
// GPU family of a HIP target, by exact name: the C++ side of clean/runner/family.py (clean/tests/test_target_family.py
// keeps the two maps equal). LLVM files gfx1170-1172 under gfx11 although they carry the gfx12 matrix design, and
// gfx1102 / gfx1103 and the gfx115x parts are not tested, so each target is added here on purpose, never matched by prefix.
#pragma once

#include <string>

namespace wgrt {

// "gfx12" or "gfx11" for a supported target name, "" for any other.
inline std::string target_family(const std::string& arch) {
  if (arch == "gfx1200" || arch == "gfx1201") return "gfx12";
  if (arch == "gfx1100" || arch == "gfx1101") return "gfx11";
  return std::string();
}

}  // namespace wgrt
