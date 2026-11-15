// SPDX-License-Identifier: MIT
#pragma once
// The last resort of the front end's refused-frame contract (B8): the game's own colour copied into the output texture,
// unscaled. Records into frame.command_list (D3D12 CopyTextureRegion with the transitions the textures' states ask for, and
// back); false with the reason in *why when it cannot (no list, another format family, a multisampled texture, an empty region).
#include <string>

#include "../include/wronggpu_upscaler.h"

namespace wgfsr {
bool CopyColorToOutput(const WgFrame& frame, std::string* why);
}
