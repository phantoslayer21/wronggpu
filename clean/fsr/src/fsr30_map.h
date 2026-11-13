// SPDX-License-Identifier: MIT
#pragma once
// FSR 3.0.x component structures (fsr30_types.h) -> the FFX API forms the front end already maps and logs. The FSR 3.0
// upscale inputs are the same quantities as FSR 3.1's (AMD's documentation describes both alike), so an FSR 3.0
// dispatch is converted to an ffxDispatchDescUpscale and goes through MapDispatch (clean/FORMATS.md).
#include <string>

#include "ffx_headers.h"
#include "fsr30_types.h"
#include "param_map.h"

namespace wgfsr {
namespace fsr30 {

uint32_t ApiSurfaceFormat(uint32_t fsr30Format);  // FfxSurfaceFormat (3.0.x) -> FfxApiSurfaceFormat
FfxApiResource ApiResource(const Resource& r);
bool IsNull(const Resource& r);

// Creation parameters shared by both entry points (flags bits 0-6, sizes).
UpscaleCreateParams CreateParams(const UpscalerContextDescription& d);
UpscaleCreateParams CreateParams(const Fsr3ContextDescription& d);

ffxDispatchDescUpscale ApiDispatch(const UpscalerDispatchDescription& d);
ffxDispatchDescUpscale ApiDispatch(const Fsr3DispatchUpscaleDescription& d);

// Log text.
std::string Describe(const Resource& r);
std::string Describe(const UpscalerContextDescription& d);
std::string Describe(const UpscalerDispatchDescription& d);
std::string Describe(const UpscalerGenerateReactiveDescription& d);
std::string Describe(const Fsr3ContextDescription& d);
std::string Describe(const Fsr3DispatchUpscaleDescription& d);
std::string Describe(const FrameGenerationConfig& c);
std::string Describe(const FrameGenerationDispatchDescription& d);
std::string DescribeFsr3Flags(uint32_t flags);
const char* ErrorName(ErrorCode code);

}  // namespace fsr30
}  // namespace wgfsr
