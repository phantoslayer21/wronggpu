// SPDX-License-Identifier: MIT
#pragma once
// Frame generation limits shared by the presenter (clean/fg/presenter) and the runtime (clean/fg/runtime), so one
// translation unit can include both sides (the FG session, clean/fg/runtime/fg_session.h). Plain C++17.
#include <cstdint>

namespace wgfg {

inline constexpr std::uint32_t kFgMaxGenerated = 5;  // N = 1..5 generated images per game frame (2x .. 6x)

}  // namespace wgfg

// The release gate of the FG modes (wgfg::kFgModesQualified in fg_contract.h, wgfgp::kFgModesQualified in
// fg_proxy_core.h): open from 0.7.0, after the end-to-end checks and the live qualification of WrongGPU-FG.asi in
// Cyberpunk 2077 with OptiScaler on an RX 9070 XT (generated images displayed through the DXGI flip presenter). The
// per-game switch (fg=off by default) and every other route fact still decide whether a game gets frame generation;
// a test sets FgProxyConfig::modes_qualified / FgRouteFacts::modes_qualified = false to check the closed gate. The
// CMake option WRONGGPU_FG_DEV_QUALIFIED (compile definition WRONGGPU_FG_DEV_QUALIFIED=1) now only adds the developer
// diagnostics (the frame dumps under WRONGGPU_FG_DUMP_DIR, fg_asi_session.h); public and CI builds never set it.
#define WG_FG_MODES_QUALIFIED 1
