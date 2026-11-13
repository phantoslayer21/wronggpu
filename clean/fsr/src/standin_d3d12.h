// SPDX-License-Identifier: MIT
#pragma once
// Stand-in upscaler (spec 20 "Boundary to the upscaler", "Stand-in (now)"): a D3D12 compute upscale of the color
// input to the output size, plus debug views for test 20-T2. Implements wronggpu_upscaler.h.
#include "../include/wronggpu_upscaler.h"

namespace wgfsr {
// splitTest: the variant for test 20-T5 (sets WG_BACKEND_NEEDS_SPLIT and upscales through split_command_list; the
// option split_test=1 in WgCreateInfo::options turns the split on).
const WgUpscalerBackend* GetStandInBackend(bool splitTest = false);
}
