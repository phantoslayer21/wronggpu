// SPDX-License-Identifier: MIT
// Compile-time check (Windows builds) that the plain-type mirror in fsr30_types.h has exactly the layout of AMD's
// FidelityFX SDK fsr3-v3.0.4 host headers. Nothing in this file is executed, and it is linked into no DLL: it belongs to
// the wg_sdk_layout_check object library (AMD's ffx_types.h defines a static variable, s_FfxViewDescInit).
#define FFX_CPP 1
#include <FidelityFX/host/ffx_fsr3.h>

#include <cstddef>

#include "fsr30_types.h"

namespace m = wgfsr::fsr30;

#define SAME_SIZE(ours, amd) static_assert(sizeof(m::ours) == sizeof(amd), "size of " #amd)
#define SAME_OFFSET(ours, amd, field) static_assert(offsetof(m::ours, field) == offsetof(amd, field), #amd "::" #field)

static_assert(sizeof(wchar_t) == sizeof(char16_t), "wchar_t is 16-bit on Windows");
SAME_SIZE(ResourceDescription, FfxResourceDescription);
SAME_SIZE(Resource, FfxResource);
SAME_OFFSET(Resource, FfxResource, description);
SAME_OFFSET(Resource, FfxResource, state);
SAME_OFFSET(Resource, FfxResource, name);
SAME_SIZE(Interface, FfxInterface);
static_assert(offsetof(m::Interface, device) == offsetof(FfxInterface, device), "FfxInterface::device");

SAME_SIZE(UpscalerContextDescription, FfxFsr3UpscalerContextDescription);
SAME_OFFSET(UpscalerContextDescription, FfxFsr3UpscalerContextDescription, maxRenderSize);
SAME_OFFSET(UpscalerContextDescription, FfxFsr3UpscalerContextDescription, displaySize);
SAME_OFFSET(UpscalerContextDescription, FfxFsr3UpscalerContextDescription, backendInterface);
SAME_OFFSET(UpscalerContextDescription, FfxFsr3UpscalerContextDescription, fpMessage);

SAME_SIZE(UpscalerDispatchDescription, FfxFsr3UpscalerDispatchDescription);
SAME_OFFSET(UpscalerDispatchDescription, FfxFsr3UpscalerDispatchDescription, color);
SAME_OFFSET(UpscalerDispatchDescription, FfxFsr3UpscalerDispatchDescription, dilatedDepth);
SAME_OFFSET(UpscalerDispatchDescription, FfxFsr3UpscalerDispatchDescription, reconstructedPrevNearestDepth);
SAME_OFFSET(UpscalerDispatchDescription, FfxFsr3UpscalerDispatchDescription, output);
SAME_OFFSET(UpscalerDispatchDescription, FfxFsr3UpscalerDispatchDescription, jitterOffset);
SAME_OFFSET(UpscalerDispatchDescription, FfxFsr3UpscalerDispatchDescription, renderSize);
SAME_OFFSET(UpscalerDispatchDescription, FfxFsr3UpscalerDispatchDescription, enableSharpening);
SAME_OFFSET(UpscalerDispatchDescription, FfxFsr3UpscalerDispatchDescription, sharpness);
SAME_OFFSET(UpscalerDispatchDescription, FfxFsr3UpscalerDispatchDescription, preExposure);
SAME_OFFSET(UpscalerDispatchDescription, FfxFsr3UpscalerDispatchDescription, reset);
SAME_OFFSET(UpscalerDispatchDescription, FfxFsr3UpscalerDispatchDescription, cameraNear);
SAME_OFFSET(UpscalerDispatchDescription, FfxFsr3UpscalerDispatchDescription, viewSpaceToMetersFactor);

SAME_SIZE(UpscalerGenerateReactiveDescription, FfxFsr3UpscalerGenerateReactiveDescription);
SAME_OFFSET(UpscalerGenerateReactiveDescription, FfxFsr3UpscalerGenerateReactiveDescription, renderSize);
SAME_OFFSET(UpscalerGenerateReactiveDescription, FfxFsr3UpscalerGenerateReactiveDescription, flags);
SAME_SIZE(UpscalerGenerateReactiveDescription, FfxFsr3GenerateReactiveDescription);

SAME_SIZE(Fsr3ContextDescription, FfxFsr3ContextDescription);
SAME_OFFSET(Fsr3ContextDescription, FfxFsr3ContextDescription, upscaleOutputSize);
SAME_OFFSET(Fsr3ContextDescription, FfxFsr3ContextDescription, displaySize);
SAME_OFFSET(Fsr3ContextDescription, FfxFsr3ContextDescription, backendInterfaceSharedResources);
SAME_OFFSET(Fsr3ContextDescription, FfxFsr3ContextDescription, backendInterfaceUpscaling);
SAME_OFFSET(Fsr3ContextDescription, FfxFsr3ContextDescription, backendInterfaceFrameInterpolation);
SAME_OFFSET(Fsr3ContextDescription, FfxFsr3ContextDescription, fpMessage);
SAME_OFFSET(Fsr3ContextDescription, FfxFsr3ContextDescription, backBufferFormat);

SAME_SIZE(Fsr3DispatchUpscaleDescription, FfxFsr3DispatchUpscaleDescription);
SAME_OFFSET(Fsr3DispatchUpscaleDescription, FfxFsr3DispatchUpscaleDescription, upscaleOutput);
SAME_OFFSET(Fsr3DispatchUpscaleDescription, FfxFsr3DispatchUpscaleDescription, jitterOffset);
SAME_OFFSET(Fsr3DispatchUpscaleDescription, FfxFsr3DispatchUpscaleDescription, renderSize);
SAME_OFFSET(Fsr3DispatchUpscaleDescription, FfxFsr3DispatchUpscaleDescription, reset);
SAME_OFFSET(Fsr3DispatchUpscaleDescription, FfxFsr3DispatchUpscaleDescription, viewSpaceToMetersFactor);

SAME_SIZE(FrameGenerationConfig, FfxFrameGenerationConfig);
SAME_OFFSET(FrameGenerationConfig, FfxFrameGenerationConfig, HUDLessColor);
SAME_OFFSET(FrameGenerationConfig, FfxFrameGenerationConfig, flags);
SAME_OFFSET(FrameGenerationConfig, FfxFrameGenerationConfig, onlyPresentInterpolated);
SAME_SIZE(FrameGenerationDispatchDescription, FfxFrameGenerationDispatchDescription);
SAME_OFFSET(FrameGenerationDispatchDescription, FfxFrameGenerationDispatchDescription, numInterpolatedFrames);
SAME_OFFSET(FrameGenerationDispatchDescription, FfxFrameGenerationDispatchDescription, minMaxLuminance);

static_assert(sizeof(FfxFsr3UpscalerContext) == m::kUpscalerContextBytes, "FfxFsr3UpscalerContext size");
static_assert(sizeof(FfxFsr3Context) == m::kFsr3ContextBytes, "FfxFsr3Context size");
static_assert(FFX_SURFACE_FORMAT_R32_FLOAT == int(m::SF_R32_FLOAT) && FFX_SURFACE_FORMAT_R11G11B10_FLOAT == int(m::SF_R11G11B10_FLOAT) &&
                  FFX_SURFACE_FORMAT_R8G8B8A8_SRGB == int(m::SF_R8G8B8A8_SRGB) && FFX_SURFACE_FORMAT_R16G16_FLOAT == int(m::SF_R16G16_FLOAT),
              "FfxSurfaceFormat values");
static_assert(FFX_FSR3UPSCALER_ENABLE_DYNAMIC_RESOLUTION == m::FLAG_DYNAMIC_RESOLUTION &&
                  FFX_FSR3_ENABLE_DYNAMIC_RESOLUTION == m::FLAG_DYNAMIC_RESOLUTION &&
                  FFX_FSR3_ENABLE_UPSCALING_ONLY == m::FSR3_FLAG_UPSCALING_ONLY &&
                  FFX_FSR3_ENABLE_ASYNC_WORKLOAD_SUPPORT == m::FSR3_FLAG_ASYNC_WORKLOAD_SUPPORT,
              "initialization flags");
static_assert(FFX_RESOURCE_STATE_COMPUTE_READ == 4 && FFX_RESOURCE_STATE_RENDER_TARGET == 256, "resource states as in the FFX API");
static_assert(FFX_ERROR_INVALID_ARGUMENT == unsigned(m::ERROR_INVALID_ARGUMENT), "error code");
