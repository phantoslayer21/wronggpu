// SPDX-License-Identifier: MIT
#include "ffx_describe.h"

#include <cinttypes>
#include <cstdarg>
#include <cstdio>
#include <cstring>

namespace wgfsr {

namespace {

void Append(std::string& s, const char* fmt, ...) {
    char buffer[1024];
    va_list args;
    va_start(args, fmt);
    const int n = std::vsnprintf(buffer, sizeof buffer, fmt, args);
    va_end(args);
    if (n > 0) s.append(buffer, size_t(n) < sizeof buffer ? size_t(n) : sizeof buffer - 1);
}

const char* SurfaceFormatName(uint32_t f) {
    switch (f) {
    case FFX_API_SURFACE_FORMAT_UNKNOWN: return "UNKNOWN";
    case FFX_API_SURFACE_FORMAT_R32G32B32A32_TYPELESS: return "R32G32B32A32_TYPELESS";
    case FFX_API_SURFACE_FORMAT_R32G32B32A32_UINT: return "R32G32B32A32_UINT";
    case FFX_API_SURFACE_FORMAT_R32G32B32A32_FLOAT: return "R32G32B32A32_FLOAT";
    case FFX_API_SURFACE_FORMAT_R16G16B16A16_FLOAT: return "R16G16B16A16_FLOAT";
    case FFX_API_SURFACE_FORMAT_R32G32B32_FLOAT: return "R32G32B32_FLOAT";
    case FFX_API_SURFACE_FORMAT_R32G32_FLOAT: return "R32G32_FLOAT";
    case FFX_API_SURFACE_FORMAT_R8_UINT: return "R8_UINT";
    case FFX_API_SURFACE_FORMAT_R32_UINT: return "R32_UINT";
    case FFX_API_SURFACE_FORMAT_R8G8B8A8_TYPELESS: return "R8G8B8A8_TYPELESS";
    case FFX_API_SURFACE_FORMAT_R8G8B8A8_UNORM: return "R8G8B8A8_UNORM";
    case FFX_API_SURFACE_FORMAT_R8G8B8A8_SNORM: return "R8G8B8A8_SNORM";
    case FFX_API_SURFACE_FORMAT_R8G8B8A8_SRGB: return "R8G8B8A8_SRGB";
    case FFX_API_SURFACE_FORMAT_B8G8R8A8_TYPELESS: return "B8G8R8A8_TYPELESS";
    case FFX_API_SURFACE_FORMAT_B8G8R8A8_UNORM: return "B8G8R8A8_UNORM";
    case FFX_API_SURFACE_FORMAT_B8G8R8A8_SRGB: return "B8G8R8A8_SRGB";
    case FFX_API_SURFACE_FORMAT_R11G11B10_FLOAT: return "R11G11B10_FLOAT";
    case FFX_API_SURFACE_FORMAT_R10G10B10A2_UNORM: return "R10G10B10A2_UNORM";
    case FFX_API_SURFACE_FORMAT_R16G16_FLOAT: return "R16G16_FLOAT";
    case FFX_API_SURFACE_FORMAT_R16G16_UINT: return "R16G16_UINT";
    case FFX_API_SURFACE_FORMAT_R16G16_SINT: return "R16G16_SINT";
    case FFX_API_SURFACE_FORMAT_R16_FLOAT: return "R16_FLOAT";
    case FFX_API_SURFACE_FORMAT_R16_UINT: return "R16_UINT";
    case FFX_API_SURFACE_FORMAT_R16_UNORM: return "R16_UNORM";
    case FFX_API_SURFACE_FORMAT_R16_SNORM: return "R16_SNORM";
    case FFX_API_SURFACE_FORMAT_R8_UNORM: return "R8_UNORM";
    case FFX_API_SURFACE_FORMAT_R8G8_UNORM: return "R8G8_UNORM";
    case FFX_API_SURFACE_FORMAT_R8G8_UINT: return "R8G8_UINT";
    case FFX_API_SURFACE_FORMAT_R32_FLOAT: return "R32_FLOAT";
    case FFX_API_SURFACE_FORMAT_R9G9B9E5_SHAREDEXP: return "R9G9B9E5_SHAREDEXP";
    case FFX_API_SURFACE_FORMAT_R16G16B16A16_TYPELESS: return "R16G16B16A16_TYPELESS";
    case FFX_API_SURFACE_FORMAT_R32G32_TYPELESS: return "R32G32_TYPELESS";
    case FFX_API_SURFACE_FORMAT_R10G10B10A2_TYPELESS: return "R10G10B10A2_TYPELESS";
    case FFX_API_SURFACE_FORMAT_R16G16_TYPELESS: return "R16G16_TYPELESS";
    case FFX_API_SURFACE_FORMAT_R16_TYPELESS: return "R16_TYPELESS";
    case FFX_API_SURFACE_FORMAT_R8_TYPELESS: return "R8_TYPELESS";
    case FFX_API_SURFACE_FORMAT_R8G8_TYPELESS: return "R8G8_TYPELESS";
    case FFX_API_SURFACE_FORMAT_R32_TYPELESS: return "R32_TYPELESS";
    default: return nullptr;
    }
}

std::string Bits(uint32_t value, const char* const* names, int count) {
    if (!value) return "0";
    std::string s;
    for (int i = 0; i < count; ++i)
        if (value & (1u << i)) {
            if (!s.empty()) s += '|';
            s += names[i];
        }
    const uint32_t known = count >= 32 ? 0xffffffffu : ((1u << count) - 1u);
    if (value & ~known) {
        if (!s.empty()) s += '|';
        Append(s, "0x%x", value & ~known);
    }
    return s;
}

std::string StateName(uint32_t state) {
    static const char* const names[] = {"COMMON", "UAV", "COMPUTE_READ", "PIXEL_READ", "COPY_SRC",
                                        "COPY_DEST", "INDIRECT", "PRESENT", "RENDER_TARGET"};
    return Bits(state, names, 9);
}

std::string F2(const FfxApiFloatCoords2D& c) {
    std::string s;
    Append(s, "(%.6g,%.6g)", c.x, c.y);
    return s;
}

std::string D2(const FfxApiDimensions2D& d) {
    std::string s;
    Append(s, "%ux%u", d.width, d.height);
    return s;
}

const char* Bool(bool b) { return b ? "1" : "0"; }

void DescribeOne(std::string& s, const ffxApiHeader* h) {
    const char* name = StructTypeName(h->type);
    if (name) s += name;
    else Append(s, "UNKNOWN_TYPE_%s", Hex(h->type).c_str());
    s += '{';
    switch (h->type) {
    case FFX_API_CONFIGURE_DESC_TYPE_GLOBALDEBUG1: {
        auto* d = reinterpret_cast<const ffxConfigureDescGlobalDebug1*>(h);
        Append(s, "fpMessage=%s debugLevel=0x%x", Pointer(reinterpret_cast<const void*>(d->fpMessage)).c_str(), d->debugLevel);
        break;
    }
    case FFX_API_QUERY_DESC_TYPE_GET_VERSIONS: {
        auto* d = reinterpret_cast<const ffxQueryDescGetVersions*>(h);
        const char* created = StructTypeName(d->createDescType);
        Append(s, "createDescType=%s device=%s outputCount=%s", created ? created : Hex(d->createDescType).c_str(),
               Pointer(d->device).c_str(), d->outputCount ? std::to_string(*d->outputCount).c_str() : "null");
        Append(s, " versionIds=%s versionNames=%s", Pointer(d->versionIds).c_str(), Pointer(d->versionNames).c_str());
        break;
    }
    case FFX_API_DESC_TYPE_OVERRIDE_VERSION: {
        auto* d = reinterpret_cast<const ffxOverrideVersion*>(h);
        Append(s, "versionId=%s", Hex(d->versionId).c_str());
        break;
    }
    case FFX_API_QUERY_DESC_TYPE_GET_PROVIDER_VERSION:
        break;
    case kCreateBackendDx12: {
        auto* d = reinterpret_cast<const BackendDx12DescView*>(h);
        Append(s, "device=%s", Pointer(d->device).c_str());
        break;
    }
    case FFX_API_CREATE_CONTEXT_DESC_TYPE_UPSCALE: {
        auto* d = reinterpret_cast<const ffxCreateContextDescUpscale*>(h);
        Append(s, "flags=%s maxRenderSize=%s maxUpscaleSize=%s fpMessage=%s", DescribeUpscaleCreateFlags(d->flags).c_str(),
               D2(d->maxRenderSize).c_str(), D2(d->maxUpscaleSize).c_str(),
               Pointer(reinterpret_cast<const void*>(d->fpMessage)).c_str());
        break;
    }
    case FFX_API_DISPATCH_DESC_TYPE_UPSCALE: {
        auto* d = reinterpret_cast<const ffxDispatchDescUpscale*>(h);
        static const char* const flagNames[] = {"DRAW_DEBUG_VIEW", "NON_LINEAR_COLOR_SRGB", "NON_LINEAR_COLOR_PQ"};
        Append(s, "commandList=%s color=%s depth=%s motionVectors=%s exposure=%s reactive=%s transparencyAndComposition=%s output=%s",
               Pointer(d->commandList).c_str(), DescribeResource(d->color).c_str(), DescribeResource(d->depth).c_str(),
               DescribeResource(d->motionVectors).c_str(), DescribeResource(d->exposure).c_str(),
               DescribeResource(d->reactive).c_str(), DescribeResource(d->transparencyAndComposition).c_str(),
               DescribeResource(d->output).c_str());
        Append(s, " jitterOffset=%s motionVectorScale=%s renderSize=%s upscaleSize=%s enableSharpening=%s sharpness=%.6g",
               F2(d->jitterOffset).c_str(), F2(d->motionVectorScale).c_str(), D2(d->renderSize).c_str(),
               D2(d->upscaleSize).c_str(), Bool(d->enableSharpening), d->sharpness);
        Append(s, " frameTimeDelta=%.6g preExposure=%.6g reset=%s cameraNear=%.6g cameraFar=%.6g cameraFovAngleVertical=%.6g"
               " viewSpaceToMetersFactor=%.6g flags=%s",
               d->frameTimeDelta, d->preExposure, Bool(d->reset), d->cameraNear, d->cameraFar, d->cameraFovAngleVertical,
               d->viewSpaceToMetersFactor, Bits(d->flags, flagNames, 3).c_str());
        break;
    }
    case FFX_API_QUERY_DESC_TYPE_UPSCALE_GETUPSCALERATIOFROMQUALITYMODE: {
        auto* d = reinterpret_cast<const ffxQueryDescUpscaleGetUpscaleRatioFromQualityMode*>(h);
        Append(s, "qualityMode=%u", d->qualityMode);
        break;
    }
    case FFX_API_QUERY_DESC_TYPE_UPSCALE_GETRENDERRESOLUTIONFROMQUALITYMODE: {
        auto* d = reinterpret_cast<const ffxQueryDescUpscaleGetRenderResolutionFromQualityMode*>(h);
        Append(s, "display=%ux%u qualityMode=%u", d->displayWidth, d->displayHeight, d->qualityMode);
        break;
    }
    case FFX_API_QUERY_DESC_TYPE_UPSCALE_GETJITTERPHASECOUNT: {
        auto* d = reinterpret_cast<const ffxQueryDescUpscaleGetJitterPhaseCount*>(h);
        Append(s, "renderWidth=%u displayWidth=%u", d->renderWidth, d->displayWidth);
        break;
    }
    case FFX_API_QUERY_DESC_TYPE_UPSCALE_GETJITTEROFFSET: {
        auto* d = reinterpret_cast<const ffxQueryDescUpscaleGetJitterOffset*>(h);
        Append(s, "index=%d phaseCount=%d", d->index, d->phaseCount);
        break;
    }
    case FFX_API_DISPATCH_DESC_TYPE_UPSCALE_GENERATEREACTIVEMASK: {
        auto* d = reinterpret_cast<const ffxDispatchDescUpscaleGenerateReactiveMask*>(h);
        Append(s, "commandList=%s colorOpaqueOnly=%s colorPreUpscale=%s outReactive=%s renderSize=%s scale=%.6g"
               " cutoffThreshold=%.6g binaryValue=%.6g flags=0x%x",
               Pointer(d->commandList).c_str(), DescribeResource(d->colorOpaqueOnly).c_str(),
               DescribeResource(d->colorPreUpscale).c_str(), DescribeResource(d->outReactive).c_str(),
               D2(d->renderSize).c_str(), d->scale, d->cutoffThreshold, d->binaryValue, d->flags);
        break;
    }
    case FFX_API_CONFIGURE_DESC_TYPE_UPSCALE_KEYVALUE: {
        auto* d = reinterpret_cast<const ffxConfigureDescUpscaleKeyValue*>(h);
        static const char* const keys[] = {"FVELOCITYFACTOR", "FREACTIVENESSSCALE", "FSHADINGCHANGESCALE",
                                           "FACCUMULATIONADDEDPERFRAME", "FMINDISOCCLUSIONACCUMULATION"};
        if (d->key < 5) Append(s, "key=%s", keys[d->key]);
        else Append(s, "key=%" PRIu64, d->key);
        Append(s, " u64=%" PRIu64 " ptr=%s", d->u64, Pointer(d->ptr).c_str());
        if (d->key < 5) {
            // ffx_upscale.h: "The float value is casted from void * ptr"; shown as the low 32 bits read as a float.
            const uintptr_t bits = reinterpret_cast<uintptr_t>(d->ptr);
            const uint32_t low = uint32_t(bits);
            float f;
            static_assert(sizeof f == sizeof low, "float size");
            std::memcpy(&f, &low, sizeof f);
            Append(s, " (as float %.6g)", f);
        }
        break;
    }
    case FFX_API_QUERY_DESC_TYPE_UPSCALE_GPU_MEMORY_USAGE:
        break;
    case FFX_API_CREATE_CONTEXT_DESC_TYPE_FRAMEGENERATION: {
        auto* d = reinterpret_cast<const ffxCreateContextDescFrameGeneration*>(h);
        Append(s, "flags=0x%x displaySize=%s maxRenderSize=%s backBufferFormat=%s", d->flags, D2(d->displaySize).c_str(),
               D2(d->maxRenderSize).c_str(), SurfaceFormatName(d->backBufferFormat) ? SurfaceFormatName(d->backBufferFormat) : "?");
        break;
    }
    case FFX_API_CONFIGURE_DESC_TYPE_FRAMEGENERATION: {
        auto* d = reinterpret_cast<const ffxConfigureDescFrameGeneration*>(h);
        Append(s, "swapChain=%s presentCallback=%s frameGenerationCallback=%s frameGenerationEnabled=%s allowAsyncWorkloads=%s"
               " HUDLessColor=%s flags=0x%x onlyPresentGenerated=%s generationRect=(%d,%d,%d,%d) frameID=%" PRIu64,
               Pointer(d->swapChain).c_str(), Pointer(reinterpret_cast<const void*>(d->presentCallback)).c_str(),
               Pointer(reinterpret_cast<const void*>(d->frameGenerationCallback)).c_str(), Bool(d->frameGenerationEnabled),
               Bool(d->allowAsyncWorkloads), DescribeResource(d->HUDLessColor).c_str(), d->flags,
               Bool(d->onlyPresentGenerated), d->generationRect.left, d->generationRect.top, d->generationRect.width,
               d->generationRect.height, d->frameID);
        break;
    }
    case FFX_API_DISPATCH_DESC_TYPE_FRAMEGENERATION_PREPARE: {
        auto* d = reinterpret_cast<const ffxDispatchDescFrameGenerationPrepare*>(h);
        Append(s, "frameID=%" PRIu64 " flags=0x%x commandList=%s renderSize=%s jitterOffset=%s motionVectorScale=%s"
               " frameTimeDelta=%.6g cameraNear=%.6g cameraFar=%.6g cameraFovAngleVertical=%.6g depth=%s motionVectors=%s",
               d->frameID, d->flags, Pointer(d->commandList).c_str(), D2(d->renderSize).c_str(),
               F2(d->jitterOffset).c_str(), F2(d->motionVectorScale).c_str(), d->frameTimeDelta, d->cameraNear,
               d->cameraFar, d->cameraFovAngleVertical, DescribeResource(d->depth).c_str(),
               DescribeResource(d->motionVectors).c_str());
        break;
    }
    case FFX_API_DISPATCH_DESC_TYPE_FRAMEGENERATION: {
        auto* d = reinterpret_cast<const ffxDispatchDescFrameGeneration*>(h);
        Append(s, "commandList=%s presentColor=%s numGeneratedFrames=%u reset=%s frameID=%" PRIu64,
               Pointer(d->commandList).c_str(), DescribeResource(d->presentColor).c_str(), d->numGeneratedFrames,
               Bool(d->reset), d->frameID);
        break;
    }
    default:
        break;  // name only: forwarded unchanged
    }
    s += '}';
}

}  // namespace

std::string Hex(uint64_t value) {
    char buffer[32];
    std::snprintf(buffer, sizeof buffer, "0x%" PRIx64, value);
    return buffer;
}

std::string Pointer(const void* p) {
    if (!p) return "null";
    char buffer[32];
    std::snprintf(buffer, sizeof buffer, "0x%" PRIxPTR, reinterpret_cast<uintptr_t>(p));
    return buffer;
}

const char* StructTypeName(uint64_t type) {
    switch (type) {
    case FFX_API_CONFIGURE_DESC_TYPE_GLOBALDEBUG1: return "CONFIGURE_GLOBALDEBUG1";
    case kCreateBackendDx12: return "CREATE_BACKEND_DX12";
    case FFX_API_QUERY_DESC_TYPE_GET_VERSIONS: return "QUERY_GET_VERSIONS";
    case FFX_API_DESC_TYPE_OVERRIDE_VERSION: return "OVERRIDE_VERSION";
    case FFX_API_QUERY_DESC_TYPE_GET_PROVIDER_VERSION: return "QUERY_GET_PROVIDER_VERSION";
    case FFX_API_CREATE_CONTEXT_DESC_TYPE_UPSCALE: return "CREATE_UPSCALE";
    case FFX_API_DISPATCH_DESC_TYPE_UPSCALE: return "DISPATCH_UPSCALE";
    case FFX_API_QUERY_DESC_TYPE_UPSCALE_GETUPSCALERATIOFROMQUALITYMODE: return "QUERY_UPSCALE_RATIO_FROM_QUALITY_MODE";
    case FFX_API_QUERY_DESC_TYPE_UPSCALE_GETRENDERRESOLUTIONFROMQUALITYMODE: return "QUERY_UPSCALE_RENDER_RESOLUTION_FROM_QUALITY_MODE";
    case FFX_API_QUERY_DESC_TYPE_UPSCALE_GETJITTERPHASECOUNT: return "QUERY_UPSCALE_JITTER_PHASE_COUNT";
    case FFX_API_QUERY_DESC_TYPE_UPSCALE_GETJITTEROFFSET: return "QUERY_UPSCALE_JITTER_OFFSET";
    case FFX_API_DISPATCH_DESC_TYPE_UPSCALE_GENERATEREACTIVEMASK: return "DISPATCH_UPSCALE_GENERATE_REACTIVE_MASK";
    case FFX_API_CONFIGURE_DESC_TYPE_UPSCALE_KEYVALUE: return "CONFIGURE_UPSCALE_KEYVALUE";
    case FFX_API_QUERY_DESC_TYPE_UPSCALE_GPU_MEMORY_USAGE: return "QUERY_UPSCALE_GPU_MEMORY_USAGE";
    case FFX_API_CREATE_CONTEXT_DESC_TYPE_FRAMEGENERATION: return "CREATE_FRAMEGENERATION";
    case FFX_API_CONFIGURE_DESC_TYPE_FRAMEGENERATION: return "CONFIGURE_FRAMEGENERATION";
    case FFX_API_DISPATCH_DESC_TYPE_FRAMEGENERATION: return "DISPATCH_FRAMEGENERATION";
    case FFX_API_DISPATCH_DESC_TYPE_FRAMEGENERATION_PREPARE: return "DISPATCH_FRAMEGENERATION_PREPARE";
    case FFX_API_CALLBACK_DESC_TYPE_FRAMEGENERATION_PRESENT: return "CALLBACK_FRAMEGENERATION_PRESENT";
    case FFX_API_CONFIGURE_DESC_TYPE_FRAMEGENERATION_KEYVALUE: return "CONFIGURE_FRAMEGENERATION_KEYVALUE";
    case FFX_API_QUERY_DESC_TYPE_FRAMEGENERATION_GPU_MEMORY_USAGE: return "QUERY_FRAMEGENERATION_GPU_MEMORY_USAGE";
    case FFX_API_CONFIGURE_DESC_TYPE_FRAMEGENERATION_REGISTERDISTORTIONRESOURCE: return "CONFIGURE_FRAMEGENERATION_REGISTER_DISTORTION_RESOURCE";
    case FFX_API_CREATE_CONTEXT_DESC_TYPE_FRAMEGENERATION_HUDLESS: return "CREATE_FRAMEGENERATION_HUDLESS";
    case FFX_API_DISPATCH_DESC_TYPE_FRAMEGENERATION_PREPARE_CAMERAINFO: return "DISPATCH_FRAMEGENERATION_PREPARE_CAMERAINFO";
    case 0x30001u: return "CREATE_FGSWAPCHAIN_WRAP_DX12";
    case 0x30002u: return "CONFIGURE_FGSWAPCHAIN_REGISTER_UI_RESOURCE_DX12";
    case 0x30003u: return "QUERY_FGSWAPCHAIN_INTERPOLATION_COMMAND_LIST_DX12";
    case 0x30004u: return "QUERY_FGSWAPCHAIN_INTERPOLATION_TEXTURE_DX12";
    case 0x30005u: return "CREATE_FGSWAPCHAIN_NEW_DX12";
    case 0x30006u: return "CREATE_FGSWAPCHAIN_FOR_HWND_DX12";
    case 0x30007u: return "DISPATCH_FGSWAPCHAIN_WAIT_FOR_PRESENTS_DX12";
    case 0x30008u: return "CONFIGURE_FGSWAPCHAIN_KEYVALUE_DX12";
    case 0x30009u: return "QUERY_FGSWAPCHAIN_GPU_MEMORY_USAGE_DX12";
    default: return nullptr;
    }
}

const char* ReturnCodeName(uint32_t code) {
    switch (code) {
    case FFX_API_RETURN_OK: return "OK";
    case FFX_API_RETURN_ERROR: return "ERROR";
    case FFX_API_RETURN_ERROR_UNKNOWN_DESCTYPE: return "ERROR_UNKNOWN_DESCTYPE";
    case FFX_API_RETURN_ERROR_RUNTIME_ERROR: return "ERROR_RUNTIME_ERROR";
    case FFX_API_RETURN_NO_PROVIDER: return "NO_PROVIDER";
    case FFX_API_RETURN_ERROR_MEMORY: return "ERROR_MEMORY";
    case FFX_API_RETURN_ERROR_PARAMETER: return "ERROR_PARAMETER";
    default: return "UNKNOWN_RETURN_CODE";
    }
}

std::string DescribeResource(const FfxApiResource& r) {
    if (!r.resource) return "none";
    std::string s = Pointer(r.resource);
    static const char* const types[] = {"BUFFER", "TEX1D", "TEX2D", "TEXCUBE", "TEX3D"};
    static const char* const usages[] = {"RT", "UAV", "DEPTH", "INDIRECT", "ARRAYVIEW", "STENCIL"};
    const char* format = SurfaceFormatName(r.description.format);
    s += '[';
    if (r.description.type < 5) s += types[r.description.type];
    else Append(s, "type%u", r.description.type);
    if (format) Append(s, " %s", format);
    else Append(s, " format%u", r.description.format);
    Append(s, " %ux%ux%u mips=%u flags=0x%x usage=%s state=%s]", r.description.width, r.description.height,
           r.description.depth, r.description.mipCount, r.description.flags,
           Bits(r.description.usage, usages, 6).c_str(), StateName(r.state).c_str());
    return s;
}

std::string DescribeUpscaleCreateFlags(uint32_t flags) {
    static const char* const names[] = {"HDR", "DISPLAY_RES_MV", "MV_JITTER_CANCELLATION", "DEPTH_INVERTED",
                                        "DEPTH_INFINITE", "AUTO_EXPOSURE", "DYNAMIC_RESOLUTION", "DEBUG_CHECKING",
                                        "NON_LINEAR_COLORSPACE"};
    return Bits(flags, names, 9);
}

std::string DescribeChain(const ffxApiHeader* header) {
    std::string s;
    int count = 0;
    for (const ffxApiHeader* h = header; h; h = h->pNext) {
        if (++count > 16) {
            s += " -> (chain longer than 16, rest not shown)";
            break;
        }
        if (count > 1) s += " -> ";
        DescribeOne(s, h);
    }
    return s.empty() ? std::string("null") : s;
}

std::string DescribeQueryOutputs(const ffxApiHeader* h) {
    std::string s;
    if (!h) return s;
    switch (h->type) {
    case FFX_API_QUERY_DESC_TYPE_GET_VERSIONS: {
        auto* d = reinterpret_cast<const ffxQueryDescGetVersions*>(h);
        if (!d->outputCount) break;
        Append(s, "count=%" PRIu64, *d->outputCount);
        for (uint64_t i = 0; i < *d->outputCount && i < 16; ++i) {
            Append(s, " [%" PRIu64 "]", i);
            if (d->versionIds) Append(s, " id=%s", Hex(d->versionIds[i]).c_str());
            if (d->versionNames) Append(s, " name=\"%s\"", d->versionNames[i] ? d->versionNames[i] : "");
        }
        break;
    }
    case FFX_API_QUERY_DESC_TYPE_GET_PROVIDER_VERSION: {
        auto* d = reinterpret_cast<const ffxQueryGetProviderVersion*>(h);
        Append(s, "versionId=%s versionName=\"%s\"", Hex(d->versionId).c_str(), d->versionName ? d->versionName : "(null)");
        break;
    }
    case FFX_API_QUERY_DESC_TYPE_UPSCALE_GETUPSCALERATIOFROMQUALITYMODE: {
        auto* d = reinterpret_cast<const ffxQueryDescUpscaleGetUpscaleRatioFromQualityMode*>(h);
        if (d->pOutUpscaleRatio) Append(s, "ratio=%.6g", *d->pOutUpscaleRatio);
        break;
    }
    case FFX_API_QUERY_DESC_TYPE_UPSCALE_GETRENDERRESOLUTIONFROMQUALITYMODE: {
        auto* d = reinterpret_cast<const ffxQueryDescUpscaleGetRenderResolutionFromQualityMode*>(h);
        Append(s, "render=%sx%s", d->pOutRenderWidth ? std::to_string(*d->pOutRenderWidth).c_str() : "null",
               d->pOutRenderHeight ? std::to_string(*d->pOutRenderHeight).c_str() : "null");
        break;
    }
    case FFX_API_QUERY_DESC_TYPE_UPSCALE_GETJITTERPHASECOUNT: {
        auto* d = reinterpret_cast<const ffxQueryDescUpscaleGetJitterPhaseCount*>(h);
        if (d->pOutPhaseCount) Append(s, "phaseCount=%d", *d->pOutPhaseCount);
        break;
    }
    case FFX_API_QUERY_DESC_TYPE_UPSCALE_GETJITTEROFFSET: {
        auto* d = reinterpret_cast<const ffxQueryDescUpscaleGetJitterOffset*>(h);
        Append(s, "x=%s y=%s", d->pOutX ? std::to_string(*d->pOutX).c_str() : "null",
               d->pOutY ? std::to_string(*d->pOutY).c_str() : "null");
        break;
    }
    case FFX_API_QUERY_DESC_TYPE_UPSCALE_GPU_MEMORY_USAGE: {
        auto* d = reinterpret_cast<const ffxQueryDescUpscaleGetGPUMemoryUsage*>(h);
        if (d->gpuMemoryUsageUpscaler)
            Append(s, "total=%" PRIu64 " aliasable=%" PRIu64, d->gpuMemoryUsageUpscaler->totalUsageInBytes,
                   d->gpuMemoryUsageUpscaler->aliasableUsageInBytes);
        break;
    }
    case FFX_API_QUERY_DESC_TYPE_FRAMEGENERATION_GPU_MEMORY_USAGE: {
        auto* d = reinterpret_cast<const ffxQueryDescFrameGenerationGetGPUMemoryUsage*>(h);
        if (d->gpuMemoryUsageFrameGeneration)
            Append(s, "total=%" PRIu64 " aliasable=%" PRIu64, d->gpuMemoryUsageFrameGeneration->totalUsageInBytes,
                   d->gpuMemoryUsageFrameGeneration->aliasableUsageInBytes);
        break;
    }
    default:
        break;
    }
    return s;
}

std::string DescribeFrame(const WgFrame& f) {
    std::string s;
    auto tex = [&](const char* name, const WgTexture& t) {
        if (!t.resource) {
            Append(s, " %s=none", name);
            return;
        }
        Append(s, " %s=%s[fmt%u %ux%u state=0x%x]", name, Pointer(t.resource).c_str(), t.format, t.width, t.height, t.state);
    };
    Append(s, "frame=%" PRIu64 " R=%ux%u at (%u,%u) O=%ux%u", f.frame_index, f.render_width, f.render_height,
           f.render_x, f.render_y, f.output_width, f.output_height);
    Append(s, " jitter=(%.6g,%.6g) motion=raw*(%.6g,%.6g)+(%.6g,%.6g)", f.jitter[0], f.jitter[1], f.motion_scale[0],
           f.motion_scale[1], f.motion_offset[0], f.motion_offset[1]);
    static const char* const modes[] = {"value", "texture", "compute"};
    Append(s, " exposure=%s", f.exposure_mode < 3 ? modes[f.exposure_mode] : "?");
    if (f.exposure_mode == WG_EXPOSURE_VALUE) Append(s, "(%.6g)", f.exposure_value);
    Append(s, " pre_exposure=%.6g reset=%u", f.pre_exposure, f.reset);
    tex("color", f.color);
    tex("depth", f.depth);
    tex("motion", f.motion);
    tex("exposure", f.exposure);
    tex("output", f.output);
    return s;
}

std::string DescribeCreateInfo(const WgCreateInfo& c) {
    static const char* const names[] = {"HDR", "DEPTH_INVERTED", "AUTO_EXPOSURE", "DEPTH_INFINITE", "MOTION_OUTPUT_RES",
                                        "NONLINEAR_COLOR", "DYNAMIC_RESOLUTION"};
    std::string s;
    Append(s, "interface=%u device=%s maxRender=%ux%u O=%ux%u flags=%s options=\"%s\"", c.interface_version,
           Pointer(c.d3d12_device).c_str(), c.max_render_width, c.max_render_height, c.output_width, c.output_height,
           Bits(c.flags, names, 7).c_str(), c.options ? c.options : "");
    if (c.interface_version >= 2) Append(s, " split=%s", c.services ? "available" : "unavailable");
    return s;
}

}  // namespace wgfsr
