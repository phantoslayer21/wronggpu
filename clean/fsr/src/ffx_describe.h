// SPDX-License-Identifier: MIT
#pragma once
// Plain-text descriptions of FidelityFX API structures for the call log. No D3D12 calls; resources are printed
// from the FfxApiResource fields the game filled in.
#include <string>

#include "../include/wronggpu_upscaler.h"
#include "ffx_headers.h"

namespace wgfsr {

const char* StructTypeName(uint64_t type);  // "UPSCALE_DISPATCH", ..., or nullptr when unknown
const char* ReturnCodeName(uint32_t code);
std::string Hex(uint64_t value);
std::string Pointer(const void* p);

std::string DescribeResource(const FfxApiResource& r);
std::string DescribeUpscaleCreateFlags(uint32_t flags);

// Describes every struct in a pNext chain: "TYPE{field=value ...} -> TYPE{...}".
std::string DescribeChain(const ffxApiHeader* header);
// After a query: the values written through the query's output pointers (empty when the type has none).
std::string DescribeQueryOutputs(const ffxApiHeader* header);
// The converted per-frame values handed to the upscaler.
std::string DescribeFrame(const WgFrame& f);
std::string DescribeCreateInfo(const WgCreateInfo& c);

}  // namespace wgfsr
