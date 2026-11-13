// SPDX-License-Identifier: MIT
// D3D12 side of the probe build (WG_RR_PROBE only; never compiled into the release build).
#pragma once
#include <stdint.h>

#include "probe.h"

namespace wgrr {

// Creates the probe's GPU side. The shaders come from probe_upscale.hlsl, compiled by tools/build.py.
ProbeGpu* create_d3d12_probe_gpu();

// DXGI view formats the probe uses for a resource of format `resource_format` tagged as `tag_format` (0 = the
// resource's): the shader-resource view for reading (0 = cannot be read; *decode_srgb: decode in the shader) and the
// unordered-access view for writing (0 = cannot be written; *encode_srgb: encode in the shader). Exposed for tests.
uint32_t probe_read_view_format(uint32_t resource_format, uint32_t tag_format, bool* decode_srgb);
uint32_t probe_write_view_format(uint32_t resource_format, uint32_t tag_format, bool* encode_srgb);

}  // namespace wgrr
