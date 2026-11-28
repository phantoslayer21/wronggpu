// SPDX-License-Identifier: GPL-3.0-only
#pragma once

#include "../fg_latency_pacer.h"

namespace wgfgp { class FgProxy; }

namespace wgfgp::win32 {

class IFgLatencyRoute {
public:
    virtual ~IFgLatencyRoute() = default;
    virtual bool LatencyActive() = 0;
    virtual bool LatencySleep(FgProxy& core, uint32_t frame, bool allow_local) = 0;
    virtual void LatencyOptions(bool enabled, uint32_t frame_limit_us) = 0;
    virtual void LatencyMarker(uint32_t frame, wgfg::FgLatencyMarker marker) = 0;
    virtual void LatencyPresent(uint64_t frame, std::optional<uint32_t> native_frame) = 0;
};

}  // namespace wgfgp::win32
