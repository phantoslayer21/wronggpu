// SPDX-License-Identifier: MIT
#pragma once
// Frame generation limits shared by the presenter (clean/fg/presenter) and the runtime (clean/fg/runtime), so one
// translation unit can include both sides (the FG session, clean/fg/runtime/fg_session.h). Plain C++17.
#include <cstdint>

namespace wgfg {

inline constexpr std::uint32_t kFgMaxGenerated = 5;  // N = 1..5 generated images per game frame (2x .. 6x)

}  // namespace wgfg
