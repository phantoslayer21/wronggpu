// SPDX-License-Identifier: MIT
#pragma once
// FidelityFX (FSR 3.1 API) parameters -> WrongGPU upscaler interface (wronggpu_upscaler.h). Pure functions, no
// D3D12 calls: everything the mapping needs to know about a D3D12 resource comes through ResourceInfoFn.
// Every conversion here is listed in clean/FORMATS.md, "FSR 3.1 to network conventions".
#include <string>

#include "../include/wronggpu_upscaler.h"
#include "config.h"
#include "ffx_headers.h"

namespace wgfsr {

// What the platform knows about a D3D12 resource (ID3D12Resource::GetDesc on Windows; a fake in tests).
struct ResourceInfo {
    uint32_t dxgiFormat = 0;  // the resource's own format (may be typeless)
    uint32_t width = 0;
    uint32_t height = 0;
};
using ResourceInfoFn = bool (*)(void* resource, ResourceInfo* out);

// FfxApiSurfaceFormat -> DXGI_FORMAT (0 when unknown).
uint32_t DxgiFromFfxSurfaceFormat(uint32_t ffxFormat);
// Format to create shader views with: the resource's own format when typed (depth formats become their
// readable view formats), else the game's FidelityFX format when it is a typed member of the resource's family,
// else the family's default typed format.
uint32_t ViewFormat(uint32_t resourceFormat, uint32_t ffxDxgiFormat);
bool IsTypeless(uint32_t dxgiFormat);
bool IsSrgb(uint32_t dxgiFormat);
// FfxApiResourceState bits -> D3D12_RESOURCE_STATES bits.
uint32_t D3D12StatesFromFfx(uint32_t ffxState);
// FfxApiCreateContextUpscaleFlags -> WG_CREATE_*.
uint32_t WgCreateFlagsFromFfx(uint32_t ffxFlags);

// Per-context state the mapping keeps between frames.
struct MapState {
    bool hasPreviousJitter = false;
    float previousJitter[2] = {0, 0};  // FidelityFX jitterOffset of the previous dispatch
};

struct UpscaleCreateParams {
    uint32_t flags = 0;  // FfxApiCreateContextUpscaleFlags
    FfxApiDimensions2D maxRenderSize{};
    FfxApiDimensions2D maxUpscaleSize{};
};

void MapCreate(const UpscaleCreateParams& create, void* device, WgCreateInfo& out);

// A WgResult in words for the log, e.g. "WG_ERROR_DEVICE (the D3D12 or HIP runtime returned an error)".
std::string WgResultText(int result);

// Interface version of a backend (1 or 2) and its WG_BACKEND_* flags (0 for version 1).
uint32_t BackendVersion(const WgUpscalerBackend* backend);
uint32_t BackendFlags(const WgUpscalerBackend* backend);
// Fits a mapped WgCreateInfo / WgFrame to the backend's interface version: version 1 backends get version 1 and the
// version 1 struct_size (the version 2 fields stay zero); version 2 backends get `services` (may be null).
void AdaptCreateInfo(const WgUpscalerBackend* backend, const WgFrontEndServices* services, WgCreateInfo& info);
void AdaptFrame(const WgUpscalerBackend* backend, WgFrame& frame);

// Dispatches `frame` on `backend`. When it answers WG_ERROR_NOT_READY (version 2: e.g. the command list cannot be
// split this frame), the stand-in upscales this one frame instead: it is created on first need from `info` (the
// context's create info; services are not passed on) and kept in *fallback (destroy it with the context). `note`
// then says so. Returns the backend's result, or the stand-in's after WG_ERROR_NOT_READY.
WgResult DispatchWithFallback(const WgUpscalerBackend* backend, WgUpscaler* upscaler, WgFrame& frame, const WgUpscalerBackend* standIn,
                              const WgCreateInfo& info, WgUpscaler** fallback, std::string& note);

// Fills `out` from one FSR upscale dispatch. Returns false (with the reason in `problem`) when a required input
// (command list, color, output, sizes) is missing or invalid; `notes` collects non-fatal observations.
bool MapDispatch(const ffxDispatchDescUpscale& d, const UpscaleCreateParams& create, const Config& config,
                 ResourceInfoFn resourceInfo, MapState& state, uint64_t frameIndex, WgFrame& out, std::string& problem,
                 std::string& notes);

// FidelityFX upscale helper queries answered for WrongGPU's own contexts (documented FSR behaviour; FORMATS.md).
float UpscaleRatioFromQualityMode(uint32_t qualityMode);  // 0 when the mode is unknown
int32_t JitterPhaseCount(uint32_t renderWidth, uint32_t displayWidth);
void JitterOffset(int32_t index, int32_t phaseCount, float* x, float* y);

}  // namespace wgfsr
