// SPDX-License-Identifier: MIT
#pragma once
// The FidelityFX SDK FSR 3.0.x component ABI (ffx_fsr3_x64.dll, ffx_fsr3upscaler_x64.dll; spec 20 integration
// point 2), written out with plain types in the Windows x64 layout so that the front end and its unit tests compile on
// any host. Field order and meaning follow AMD's public headers of FidelityFX SDK tag fsr3-v3.0.4 (identical in
// fsr3-v3.0.3), MIT, vendored in clean/third_party/fidelityfx-sdk-fsr3-v3.0.4. Every size and offset is checked
// against those headers at compile time (fsr30_layout_check.cpp, Windows builds).
#include <cstddef>
#include <cstdint>

namespace wgfsr {
namespace fsr30 {

using ErrorCode = int32_t;  // FfxErrorCode
constexpr ErrorCode OK = 0;
constexpr ErrorCode ERROR_INVALID_POINTER = int32_t(0x80000000u);
constexpr ErrorCode ERROR_INVALID_ARGUMENT = int32_t(0x8000000au);
constexpr ErrorCode ERROR_BACKEND_API_ERROR = int32_t(0x8000000du);

// FfxSurfaceFormat (3.0.x order; not the same numbering as FfxApiSurfaceFormat).
enum SurfaceFormat : uint32_t {
    SF_UNKNOWN, SF_R32G32B32A32_TYPELESS, SF_R32G32B32A32_UINT, SF_R32G32B32A32_FLOAT, SF_R16G16B16A16_FLOAT,
    SF_R32G32_FLOAT, SF_R8_UINT, SF_R32_UINT, SF_R10G10B10A2_UNORM, SF_R8G8B8A8_TYPELESS, SF_R8G8B8A8_UNORM,
    SF_R8G8B8A8_SNORM, SF_R8G8B8A8_SRGB, SF_R11G11B10_FLOAT, SF_R16G16_FLOAT, SF_R16G16_UINT, SF_R16G16_SINT,
    SF_R16_FLOAT, SF_R16_UINT, SF_R16_UNORM, SF_R16_SNORM, SF_R8_UNORM, SF_R8G8_UNORM, SF_R8G8_UINT, SF_R32_FLOAT,
    SF_COUNT
};

// FfxFsr3UpscalerInitializationFlagBits / FfxFsr3InitializationFlagBits (bits 0-6 mean the same in both, and the
// same as FfxApiCreateContextUpscaleFlags bits 0-6).
constexpr uint32_t FLAG_HIGH_DYNAMIC_RANGE = 1u << 0;
constexpr uint32_t FLAG_DISPLAY_RESOLUTION_MOTION_VECTORS = 1u << 1;
constexpr uint32_t FLAG_MOTION_VECTORS_JITTER_CANCELLATION = 1u << 2;
constexpr uint32_t FLAG_DEPTH_INVERTED = 1u << 3;
constexpr uint32_t FLAG_DEPTH_INFINITE = 1u << 4;
constexpr uint32_t FLAG_AUTO_EXPOSURE = 1u << 5;
constexpr uint32_t FLAG_DYNAMIC_RESOLUTION = 1u << 6;
constexpr uint32_t FLAG_TEXTURE1D_USAGE = 1u << 7;
constexpr uint32_t FLAG_DEBUG_CHECKING = 1u << 8;
constexpr uint32_t FSR3_FLAG_UPSCALING_ONLY = 1u << 9;   // ffx_fsr3.h only
constexpr uint32_t FSR3_FLAG_HDR_UPSCALE_SDR_FINALOUTPUT = 1u << 10;
constexpr uint32_t FSR3_FLAG_SDR_UPSCALE_HDR_FINALOUTPUT = 1u << 11;
constexpr uint32_t FSR3_FLAG_ASYNC_WORKLOAD_SUPPORT = 1u << 12;
constexpr uint32_t kSharedUpscaleFlags = 0x7Fu;          // bits 0-6

struct Dimensions2D {
    uint32_t width, height;
};
struct FloatCoords2D {
    float x, y;
};

struct ResourceDescription {  // FfxResourceDescription
    uint32_t type;            // FfxResourceType
    uint32_t format;          // FfxSurfaceFormat (3.0.x)
    uint32_t width;           // or size
    uint32_t height;          // or stride
    uint32_t depth;           // or alignment
    uint32_t mipCount;
    uint32_t flags;           // FfxResourceFlags
    uint32_t usage;           // FfxResourceUsage
};

struct Resource {             // FfxResource
    void* resource;           // ID3D12Resource*
    ResourceDescription description;
    uint32_t state;           // FfxResourceStates (same bits as FfxApiResourceState)
    char16_t name[64];        // wchar_t on Windows
};

struct Interface {            // FfxInterface: 16 function pointers, scratch buffer, its size, device
    void* functions[16];
    void* scratchBuffer;
    size_t scratchBufferSize;
    void* device;             // FfxDevice
};

using MessageFn = void (*)(uint32_t type, const wchar_t* message);

struct UpscalerContextDescription {  // FfxFsr3UpscalerContextDescription
    uint32_t flags;
    Dimensions2D maxRenderSize;
    Dimensions2D displaySize;
    Interface backendInterface;
    MessageFn fpMessage;
};

struct UpscalerDispatchDescription {  // FfxFsr3UpscalerDispatchDescription
    void* commandList;
    Resource color;
    Resource depth;
    Resource motionVectors;
    Resource exposure;
    Resource reactive;
    Resource transparencyAndComposition;
    Resource dilatedDepth;                   // shared with frame interpolation (written by the upscaler)
    Resource dilatedMotionVectors;           // shared with frame interpolation
    Resource reconstructedPrevNearestDepth;  // shared with frame interpolation
    Resource output;
    FloatCoords2D jitterOffset;
    FloatCoords2D motionVectorScale;
    Dimensions2D renderSize;
    bool enableSharpening;
    float sharpness;
    float frameTimeDelta;
    float preExposure;
    bool reset;
    float cameraNear;
    float cameraFar;
    float cameraFovAngleVertical;
    float viewSpaceToMetersFactor;
};

struct UpscalerGenerateReactiveDescription {  // FfxFsr3UpscalerGenerateReactiveDescription (same as FfxFsr3GenerateReactiveDescription)
    void* commandList;
    Resource colorOpaqueOnly;
    Resource colorPreUpscale;
    Resource outReactive;
    Dimensions2D renderSize;
    float scale;
    float cutoffThreshold;
    float binaryValue;
    uint32_t flags;
};

struct Fsr3ContextDescription {  // FfxFsr3ContextDescription
    uint32_t flags;
    Dimensions2D maxRenderSize;
    Dimensions2D upscaleOutputSize;
    Dimensions2D displaySize;
    Interface backendInterfaceSharedResources;
    Interface backendInterfaceUpscaling;
    Interface backendInterfaceFrameInterpolation;
    MessageFn fpMessage;
    uint32_t backBufferFormat;  // FfxSurfaceFormat
};

struct Fsr3DispatchUpscaleDescription {  // FfxFsr3DispatchUpscaleDescription
    void* commandList;
    Resource color;
    Resource depth;
    Resource motionVectors;
    Resource exposure;
    Resource reactive;
    Resource transparencyAndComposition;
    Resource upscaleOutput;
    FloatCoords2D jitterOffset;
    FloatCoords2D motionVectorScale;
    Dimensions2D renderSize;
    bool enableSharpening;
    float sharpness;
    float frameTimeDelta;
    float preExposure;
    bool reset;
    float cameraNear;
    float cameraFar;
    float cameraFovAngleVertical;
    float viewSpaceToMetersFactor;
};

struct FrameGenerationConfig {  // FfxFrameGenerationConfig (ffx_interface.h)
    void* swapChain;
    void* presentCallback;
    void* frameGenerationCallback;
    bool frameGenerationEnabled;
    bool allowAsyncWorkloads;
    Resource HUDLessColor;
    uint32_t flags;
    bool onlyPresentInterpolated;
};

struct FrameGenerationDispatchDescription {  // FfxFrameGenerationDispatchDescription (ffx_types.h)
    void* commandList;
    Resource presentColor;
    Resource outputs[4];
    uint32_t numInterpolatedFrames;
    bool reset;
    uint32_t backBufferTransferFunction;
    float minMaxLuminance[2];
};

// Caller-allocated opaque contexts (uint32_t data[...]).
constexpr size_t kUpscalerContextBytes = 4u * 1024u * 128u;                         // FfxFsr3UpscalerContext
constexpr size_t kFsr3ContextBytes = 4u * (3u * 1024u * 128u + 16536u);             // FfxFsr3Context

}  // namespace fsr30
}  // namespace wgfsr
