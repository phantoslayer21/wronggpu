// SPDX-License-Identifier: MIT
#pragma once
// AMD FidelityFX API headers (SDK 1.1.4, MIT, vendored in clean/third_party) without any D3D12 dependency, plus
// D3D12/DXGI enum values the front end needs as plain numbers. The Windows build checks every mirrored value
// against the Windows SDK headers (see static_asserts in dllmain.cpp).
#if !defined(_WIN32) && !defined(__declspec)
#define __declspec(x)  // host unit tests: ffx_api.h marks its prototypes dllexport
#endif
#include "ffx_api/ffx_api.h"
#include "ffx_api/ffx_api_types.h"
#include "ffx_api/ffx_framegeneration.h"
#include "ffx_api/ffx_upscale.h"

#include <cstdint>

namespace wgfsr {

// Layout of ffxCreateBackendDX12Desc (ffx_api/dx12/ffx_api_dx12.h) with the device as void*.
struct BackendDx12DescView {
    ffxCreateContextDescHeader header;
    void* device;
};
constexpr uint64_t kCreateBackendDx12 = 0x0000002u;  // FFX_API_CREATE_CONTEXT_DESC_TYPE_BACKEND_DX12

// Effect ids of the FidelityFX API (ffx_api.h and the per-effect headers).
constexpr uint64_t kEffectMask = 0xffff0000u;
constexpr uint64_t kEffectUpscale = 0x00010000u;
constexpr uint64_t kEffectFrameGeneration = 0x00020000u;
constexpr uint64_t kEffectFrameGenerationSwapChainDx12 = 0x00030000u;

// DXGI_FORMAT values (dxgiformat.h).
namespace dxgi {
constexpr uint32_t UNKNOWN = 0;
constexpr uint32_t R32G32B32A32_TYPELESS = 1, R32G32B32A32_FLOAT = 2, R32G32B32A32_UINT = 3;
constexpr uint32_t R32G32B32_TYPELESS = 5, R32G32B32_FLOAT = 6;
constexpr uint32_t R16G16B16A16_TYPELESS = 9, R16G16B16A16_FLOAT = 10, R16G16B16A16_UNORM = 11;
constexpr uint32_t R32G32_TYPELESS = 15, R32G32_FLOAT = 16;
constexpr uint32_t R32G8X24_TYPELESS = 19, D32_FLOAT_S8X24_UINT = 20, R32_FLOAT_X8X24_TYPELESS = 21;
constexpr uint32_t R10G10B10A2_TYPELESS = 23, R10G10B10A2_UNORM = 24;
constexpr uint32_t R11G11B10_FLOAT = 26;
constexpr uint32_t R8G8B8A8_TYPELESS = 27, R8G8B8A8_UNORM = 28, R8G8B8A8_UNORM_SRGB = 29, R8G8B8A8_SNORM = 31;
constexpr uint32_t R16G16_TYPELESS = 33, R16G16_FLOAT = 34, R16G16_UNORM = 35, R16G16_UINT = 36, R16G16_SNORM = 37,
                   R16G16_SINT = 38;
constexpr uint32_t R32_TYPELESS = 39, D32_FLOAT = 40, R32_FLOAT = 41, R32_UINT = 42;
constexpr uint32_t R24G8_TYPELESS = 44, D24_UNORM_S8_UINT = 45, R24_UNORM_X8_TYPELESS = 46;
constexpr uint32_t R8G8_TYPELESS = 48, R8G8_UNORM = 49, R8G8_UINT = 50;
constexpr uint32_t R16_TYPELESS = 53, R16_FLOAT = 54, D16_UNORM = 55, R16_UNORM = 56, R16_UINT = 57, R16_SNORM = 58;
constexpr uint32_t R8_TYPELESS = 60, R8_UNORM = 61, R8_UINT = 62;
constexpr uint32_t R9G9B9E5_SHAREDEXP = 67;
constexpr uint32_t B8G8R8A8_UNORM = 87, B8G8R8A8_TYPELESS = 90, B8G8R8A8_UNORM_SRGB = 91;
}  // namespace dxgi

// D3D12_RESOURCE_STATES bits (d3d12.h).
namespace d3d12state {
constexpr uint32_t COMMON = 0;
constexpr uint32_t RENDER_TARGET = 0x4;
constexpr uint32_t UNORDERED_ACCESS = 0x8;
constexpr uint32_t DEPTH_WRITE = 0x10;
constexpr uint32_t DEPTH_READ = 0x20;
constexpr uint32_t NON_PIXEL_SHADER_RESOURCE = 0x40;
constexpr uint32_t PIXEL_SHADER_RESOURCE = 0x80;
constexpr uint32_t INDIRECT_ARGUMENT = 0x200;
constexpr uint32_t COPY_DEST = 0x400;
constexpr uint32_t COPY_SOURCE = 0x800;
constexpr uint32_t PRESENT = 0;
}  // namespace d3d12state

}  // namespace wgfsr
