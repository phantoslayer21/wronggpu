// SPDX-License-Identifier: MIT
// Supported formats (spec 62.5, WrongGPU's choice). DXGI format numbers are the public Windows SDK enum values.
#pragma once
#include <stdint.h>

#include "../include/wg_rr_adapter.h"

namespace wgrr {

// Class of a DXGI format as read by the passes (62.5). Typeless formats resolve to the float member of their
// family when it has one, otherwise to the unsigned-normalised member. Depth formats (with or without stencil)
// and typeless depth families are accepted only when `depth_input` is set (buffer types 0 and 49) and are then
// read through their depth component. Returns WG_FMT_UNSUPPORTED for integer, block-compressed, video and every
// other format outside the four classes.
uint32_t format_class(uint32_t dxgi_format, bool depth_input);

}  // namespace wgrr
