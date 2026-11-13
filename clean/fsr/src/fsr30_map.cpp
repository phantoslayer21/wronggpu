// SPDX-License-Identifier: MIT
#include "fsr30_map.h"

#include <cstdarg>
#include <cstdio>

#include "ffx_describe.h"

namespace wgfsr {
namespace fsr30 {

namespace {

void Append(std::string& s, const char* fmt, ...) {
    char buffer[1024];
    va_list args;
    va_start(args, fmt);
    const int n = std::vsnprintf(buffer, sizeof buffer, fmt, args);
    va_end(args);
    if (n > 0) s.append(buffer, size_t(n) < sizeof buffer ? size_t(n) : sizeof buffer - 1);
}

const char* B(bool b) { return b ? "1" : "0"; }

std::string Name(const char16_t* name) {
    std::string s;
    for (int i = 0; i < 64 && name[i]; ++i) s += name[i] < 128 ? char(name[i]) : '?';
    return s;
}

template <class D>
void CopyCommon(const D& d, ffxDispatchDescUpscale& a) {
    a.header.type = FFX_API_DISPATCH_DESC_TYPE_UPSCALE;
    a.commandList = d.commandList;
    a.color = ApiResource(d.color);
    a.depth = ApiResource(d.depth);
    a.motionVectors = ApiResource(d.motionVectors);
    a.exposure = ApiResource(d.exposure);
    a.reactive = ApiResource(d.reactive);
    a.transparencyAndComposition = ApiResource(d.transparencyAndComposition);
    a.jitterOffset = {d.jitterOffset.x, d.jitterOffset.y};
    a.motionVectorScale = {d.motionVectorScale.x, d.motionVectorScale.y};
    a.renderSize = {d.renderSize.width, d.renderSize.height};
    a.upscaleSize = {0, 0};  // FSR 3.0 has no per-frame output size: the context's size is used
    a.enableSharpening = d.enableSharpening;
    a.sharpness = d.sharpness;
    a.frameTimeDelta = d.frameTimeDelta;
    a.preExposure = d.preExposure;
    a.reset = d.reset;
    a.cameraNear = d.cameraNear;
    a.cameraFar = d.cameraFar;
    a.cameraFovAngleVertical = d.cameraFovAngleVertical;
    a.viewSpaceToMetersFactor = d.viewSpaceToMetersFactor;
    a.flags = 0;
}

template <class D>
void DescribeCommonInputs(std::string& s, const D& d) {
    Append(s, "commandList=%s color=%s depth=%s motionVectors=%s exposure=%s reactive=%s transparencyAndComposition=%s",
           Pointer(d.commandList).c_str(), Describe(d.color).c_str(), Describe(d.depth).c_str(),
           Describe(d.motionVectors).c_str(), Describe(d.exposure).c_str(), Describe(d.reactive).c_str(),
           Describe(d.transparencyAndComposition).c_str());
}

template <class D>
void DescribeCommonScalars(std::string& s, const D& d) {
    Append(s, " jitterOffset=(%.6g,%.6g) motionVectorScale=(%.6g,%.6g) renderSize=%ux%u enableSharpening=%s sharpness=%.6g"
           " frameTimeDelta=%.6g preExposure=%.6g reset=%s cameraNear=%.6g cameraFar=%.6g cameraFovAngleVertical=%.6g"
           " viewSpaceToMetersFactor=%.6g",
           d.jitterOffset.x, d.jitterOffset.y, d.motionVectorScale.x, d.motionVectorScale.y, d.renderSize.width,
           d.renderSize.height, B(d.enableSharpening), d.sharpness, d.frameTimeDelta, d.preExposure, B(d.reset), d.cameraNear,
           d.cameraFar, d.cameraFovAngleVertical, d.viewSpaceToMetersFactor);
}

}  // namespace

uint32_t ApiSurfaceFormat(uint32_t f) {
    switch (f) {
    case SF_R32G32B32A32_TYPELESS: return FFX_API_SURFACE_FORMAT_R32G32B32A32_TYPELESS;
    case SF_R32G32B32A32_UINT: return FFX_API_SURFACE_FORMAT_R32G32B32A32_UINT;
    case SF_R32G32B32A32_FLOAT: return FFX_API_SURFACE_FORMAT_R32G32B32A32_FLOAT;
    case SF_R16G16B16A16_FLOAT: return FFX_API_SURFACE_FORMAT_R16G16B16A16_FLOAT;
    case SF_R32G32_FLOAT: return FFX_API_SURFACE_FORMAT_R32G32_FLOAT;
    case SF_R8_UINT: return FFX_API_SURFACE_FORMAT_R8_UINT;
    case SF_R32_UINT: return FFX_API_SURFACE_FORMAT_R32_UINT;
    case SF_R10G10B10A2_UNORM: return FFX_API_SURFACE_FORMAT_R10G10B10A2_UNORM;
    case SF_R8G8B8A8_TYPELESS: return FFX_API_SURFACE_FORMAT_R8G8B8A8_TYPELESS;
    case SF_R8G8B8A8_UNORM: return FFX_API_SURFACE_FORMAT_R8G8B8A8_UNORM;
    case SF_R8G8B8A8_SNORM: return FFX_API_SURFACE_FORMAT_R8G8B8A8_SNORM;
    case SF_R8G8B8A8_SRGB: return FFX_API_SURFACE_FORMAT_R8G8B8A8_SRGB;
    case SF_R11G11B10_FLOAT: return FFX_API_SURFACE_FORMAT_R11G11B10_FLOAT;
    case SF_R16G16_FLOAT: return FFX_API_SURFACE_FORMAT_R16G16_FLOAT;
    case SF_R16G16_UINT: return FFX_API_SURFACE_FORMAT_R16G16_UINT;
    case SF_R16G16_SINT: return FFX_API_SURFACE_FORMAT_R16G16_SINT;
    case SF_R16_FLOAT: return FFX_API_SURFACE_FORMAT_R16_FLOAT;
    case SF_R16_UINT: return FFX_API_SURFACE_FORMAT_R16_UINT;
    case SF_R16_UNORM: return FFX_API_SURFACE_FORMAT_R16_UNORM;
    case SF_R16_SNORM: return FFX_API_SURFACE_FORMAT_R16_SNORM;
    case SF_R8_UNORM: return FFX_API_SURFACE_FORMAT_R8_UNORM;
    case SF_R8G8_UNORM: return FFX_API_SURFACE_FORMAT_R8G8_UNORM;
    case SF_R8G8_UINT: return FFX_API_SURFACE_FORMAT_R8G8_UINT;
    case SF_R32_FLOAT: return FFX_API_SURFACE_FORMAT_R32_FLOAT;
    default: return FFX_API_SURFACE_FORMAT_UNKNOWN;
    }
}

FfxApiResource ApiResource(const Resource& r) {
    FfxApiResource a{};
    a.resource = r.resource;
    a.description.type = r.description.type;  // FfxResourceType has the same values in both
    a.description.format = ApiSurfaceFormat(r.description.format);
    a.description.width = r.description.width;
    a.description.height = r.description.height;
    a.description.depth = r.description.depth;
    a.description.mipCount = r.description.mipCount;
    a.description.flags = r.description.flags;
    a.description.usage = r.description.usage;
    a.state = r.state;  // FfxResourceStates has the same bits as FfxApiResourceState
    return a;
}

bool IsNull(const Resource& r) { return r.resource == nullptr; }

UpscaleCreateParams CreateParams(const UpscalerContextDescription& d) {
    UpscaleCreateParams p;
    p.flags = d.flags & kSharedUpscaleFlags;
    p.maxRenderSize = {d.maxRenderSize.width, d.maxRenderSize.height};
    p.maxUpscaleSize = {d.displaySize.width, d.displaySize.height};
    return p;
}

UpscaleCreateParams CreateParams(const Fsr3ContextDescription& d) {
    UpscaleCreateParams p;
    p.flags = d.flags & kSharedUpscaleFlags;
    p.maxRenderSize = {d.maxRenderSize.width, d.maxRenderSize.height};
    // The upscaler writes upscaleOutputSize; displaySize is the frame-generation (swap chain) size.
    const Dimensions2D out = d.upscaleOutputSize.width && d.upscaleOutputSize.height ? d.upscaleOutputSize : d.displaySize;
    p.maxUpscaleSize = {out.width, out.height};
    return p;
}

ffxDispatchDescUpscale ApiDispatch(const UpscalerDispatchDescription& d) {
    ffxDispatchDescUpscale a{};
    CopyCommon(d, a);
    a.output = ApiResource(d.output);
    return a;
}

ffxDispatchDescUpscale ApiDispatch(const Fsr3DispatchUpscaleDescription& d) {
    ffxDispatchDescUpscale a{};
    CopyCommon(d, a);
    a.output = ApiResource(d.upscaleOutput);
    return a;
}

std::string Describe(const Resource& r) {
    if (!r.resource) return "none";
    std::string s = DescribeResource(ApiResource(r));
    if (r.description.format >= SF_COUNT) Append(s, "(FSR3.0 format %u)", r.description.format);
    if (r.name[0]) s += "\"" + Name(r.name) + "\"";
    return s;
}

std::string DescribeFsr3Flags(uint32_t flags) {
    static const char* const names[] = {"HDR", "DISPLAY_RES_MV", "MV_JITTER_CANCELLATION", "DEPTH_INVERTED", "DEPTH_INFINITE",
                                        "AUTO_EXPOSURE", "DYNAMIC_RESOLUTION", "TEXTURE1D_USAGE", "DEBUG_CHECKING",
                                        "UPSCALING_ONLY", "HDR_UPSCALE_SDR_FINALOUTPUT", "SDR_UPSCALE_HDR_FINALOUTPUT",
                                        "ASYNC_WORKLOAD_SUPPORT"};
    if (!flags) return "0";
    std::string s;
    for (int i = 0; i < 13; ++i)
        if (flags & (1u << i)) {
            if (!s.empty()) s += '|';
            s += names[i];
        }
    if (flags >> 13) Append(s, "%s0x%x", s.empty() ? "" : "|", flags & ~0x1FFFu);
    return s;
}

std::string Describe(const UpscalerContextDescription& d) {
    std::string s;
    Append(s, "{flags=%s maxRenderSize=%ux%u displaySize=%ux%u backendInterface.device=%s scratch=%s/%zu fpMessage=%s}",
           DescribeFsr3Flags(d.flags).c_str(), d.maxRenderSize.width, d.maxRenderSize.height, d.displaySize.width,
           d.displaySize.height, Pointer(d.backendInterface.device).c_str(), Pointer(d.backendInterface.scratchBuffer).c_str(),
           d.backendInterface.scratchBufferSize, Pointer(reinterpret_cast<const void*>(d.fpMessage)).c_str());
    return s;
}

std::string Describe(const UpscalerDispatchDescription& d) {
    std::string s = "{";
    DescribeCommonInputs(s, d);
    Append(s, " dilatedDepth=%s dilatedMotionVectors=%s reconstructedPrevNearestDepth=%s output=%s", Describe(d.dilatedDepth).c_str(),
           Describe(d.dilatedMotionVectors).c_str(), Describe(d.reconstructedPrevNearestDepth).c_str(), Describe(d.output).c_str());
    DescribeCommonScalars(s, d);
    s += '}';
    return s;
}

std::string Describe(const UpscalerGenerateReactiveDescription& d) {
    std::string s;
    Append(s, "{commandList=%s colorOpaqueOnly=%s colorPreUpscale=%s outReactive=%s renderSize=%ux%u scale=%.6g"
           " cutoffThreshold=%.6g binaryValue=%.6g flags=0x%x}",
           Pointer(d.commandList).c_str(), Describe(d.colorOpaqueOnly).c_str(), Describe(d.colorPreUpscale).c_str(),
           Describe(d.outReactive).c_str(), d.renderSize.width, d.renderSize.height, d.scale, d.cutoffThreshold, d.binaryValue,
           d.flags);
    return s;
}

std::string Describe(const Fsr3ContextDescription& d) {
    std::string s;
    Append(s, "{flags=%s maxRenderSize=%ux%u upscaleOutputSize=%ux%u displaySize=%ux%u devices(shared/upscaling/interpolation)=%s/%s/%s"
           " fpMessage=%s backBufferFormat=%u}",
           DescribeFsr3Flags(d.flags).c_str(), d.maxRenderSize.width, d.maxRenderSize.height, d.upscaleOutputSize.width,
           d.upscaleOutputSize.height, d.displaySize.width, d.displaySize.height,
           Pointer(d.backendInterfaceSharedResources.device).c_str(), Pointer(d.backendInterfaceUpscaling.device).c_str(),
           Pointer(d.backendInterfaceFrameInterpolation.device).c_str(), Pointer(reinterpret_cast<const void*>(d.fpMessage)).c_str(),
           d.backBufferFormat);
    return s;
}

std::string Describe(const Fsr3DispatchUpscaleDescription& d) {
    std::string s = "{";
    DescribeCommonInputs(s, d);
    Append(s, " upscaleOutput=%s", Describe(d.upscaleOutput).c_str());
    DescribeCommonScalars(s, d);
    s += '}';
    return s;
}

std::string Describe(const FrameGenerationConfig& c) {
    std::string s;
    Append(s, "{swapChain=%s presentCallback=%s frameGenerationCallback=%s frameGenerationEnabled=%s allowAsyncWorkloads=%s"
           " HUDLessColor=%s flags=0x%x onlyPresentInterpolated=%s}",
           Pointer(c.swapChain).c_str(), Pointer(c.presentCallback).c_str(), Pointer(c.frameGenerationCallback).c_str(),
           B(c.frameGenerationEnabled), B(c.allowAsyncWorkloads), Describe(c.HUDLessColor).c_str(), c.flags,
           B(c.onlyPresentInterpolated));
    return s;
}

std::string Describe(const FrameGenerationDispatchDescription& d) {
    std::string s;
    Append(s, "{commandList=%s presentColor=%s outputs[0]=%s numInterpolatedFrames=%u reset=%s backBufferTransferFunction=%u"
           " minMaxLuminance=(%.6g,%.6g)}",
           Pointer(d.commandList).c_str(), Describe(d.presentColor).c_str(), Describe(d.outputs[0]).c_str(),
           d.numInterpolatedFrames, B(d.reset), d.backBufferTransferFunction, d.minMaxLuminance[0], d.minMaxLuminance[1]);
    return s;
}

const char* ErrorName(ErrorCode code) {
    switch (uint32_t(code)) {
    case 0: return "FFX_OK";
    case 0x80000000u: return "FFX_ERROR_INVALID_POINTER";
    case 0x80000001u: return "FFX_ERROR_INVALID_ALIGNMENT";
    case 0x80000002u: return "FFX_ERROR_INVALID_SIZE";
    case 0x80000003u: return "FFX_EOF";
    case 0x80000004u: return "FFX_ERROR_INVALID_PATH";
    case 0x80000005u: return "FFX_ERROR_EOF";
    case 0x80000006u: return "FFX_ERROR_MALFORMED_DATA";
    case 0x80000007u: return "FFX_ERROR_OUT_OF_MEMORY";
    case 0x80000008u: return "FFX_ERROR_INCOMPLETE_INTERFACE";
    case 0x80000009u: return "FFX_ERROR_INVALID_ENUM";
    case 0x8000000au: return "FFX_ERROR_INVALID_ARGUMENT";
    case 0x8000000bu: return "FFX_ERROR_OUT_OF_RANGE";
    case 0x8000000cu: return "FFX_ERROR_NULL_DEVICE";
    case 0x8000000du: return "FFX_ERROR_BACKEND_API_ERROR";
    case 0x8000000eu: return "FFX_ERROR_INSUFFICIENT_MEMORY";
    default: return "FFX_ERROR_(unknown)";
    }
}

}  // namespace fsr30
}  // namespace wgfsr
