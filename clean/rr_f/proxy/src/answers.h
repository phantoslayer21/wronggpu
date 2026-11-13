// SPDX-License-Identifier: MIT
// The proxy's own answers for feature 1001 that the release build and the probe build share (spec 60.8, 61.5).
#pragma once
#include <stdint.h>

#include "../include/rr_proxy_abi.h"

namespace wgrr {

// <requirements>(1001) of 60.8: D3D12 only, 0 CPU threads, 1 viewport, the 5 unconditional buffer types, no version
// gate, Vulkan-only members 0. Writes only the members of the caller's structure version (v1 = the members before
// the Vulkan-only block, QUESTIONS-60-63.md Q-L1). The array pointer stays valid until unload.
WgResultCode answer_requirements(WgRequirements* out);

// <version>(1001) of 60.8: the game's Streamline version as passed to <init>, WrongGPU's release version.
WgResultCode answer_version(WgFeatureVersion* out, uint64_t api_version, const uint32_t release_version[3]);

// <rr-optimal> of 61.5 (WrongGPU's rule) with the zero-render-size answer of the Streamline variant `api_minor`
// (7: success with the zero sizes copied out; 14: 25). Modes > 6: 25.
WgResultCode answer_optimal(uint32_t api_minor, const WgRrOptions* options, WgRrOptimal* out);

// The header version a reader honours: 0 is read as 1 (QUESTIONS-60-63.md Q-L7).
inline uint64_t struct_version(const WgHeader& h) { return h.version < 1 ? 1 : h.version; }

}  // namespace wgrr
