// SPDX-License-Identifier: MIT
#pragma once
// Frame generation limits shared by the presenter (clean/fg/presenter) and the runtime (clean/fg/runtime), so one
// translation unit can include both sides (the FG session, clean/fg/runtime/fg_session.h). Plain C++17.
#include <cstdint>

namespace wgfg {

inline constexpr std::uint32_t kFgMaxGenerated = 5;  // N = 1..5 generated images per game frame (2x .. 6x)

}  // namespace wgfg

// The release gate of the FG modes (wgfg::kFgModesQualified in fg_contract.h, wgfgp::kFgModesQualified in
// fg_proxy_core.h): closed until the end-to-end oracle gate (F5) and a qualified presenter pass. Developer test builds
// may open it with the CMake option WRONGGPU_FG_DEV_QUALIFIED (compile definition WRONGGPU_FG_DEV_QUALIFIED=1, the
// whole build tree); public and CI builds never set it.
#if defined(WRONGGPU_FG_DEV_QUALIFIED) && WRONGGPU_FG_DEV_QUALIFIED
#define WG_FG_MODES_QUALIFIED 1
#else
#define WG_FG_MODES_QUALIFIED 0
#endif
