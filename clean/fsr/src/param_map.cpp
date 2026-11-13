// SPDX-License-Identifier: MIT
#include "param_map.h"

#include <cmath>
#include <cstddef>
#include <cstdio>

namespace wgfsr {

namespace {

struct FormatEntry {
    uint32_t format;
    uint32_t family;  // typeless family (the format itself for typeless formats, 0 for formats without one)
    uint32_t view;    // format to view it with in a shader (depth formats differ from themselves)
};

// Typeless families and the default typed member used when nothing more specific is known.
constexpr FormatEntry kFormats[] = {
    {dxgi::R32G32B32A32_TYPELESS, dxgi::R32G32B32A32_TYPELESS, dxgi::R32G32B32A32_FLOAT},
    {dxgi::R32G32B32A32_FLOAT, dxgi::R32G32B32A32_TYPELESS, dxgi::R32G32B32A32_FLOAT},
    {dxgi::R32G32B32A32_UINT, dxgi::R32G32B32A32_TYPELESS, dxgi::R32G32B32A32_UINT},
    {dxgi::R32G32B32_TYPELESS, dxgi::R32G32B32_TYPELESS, dxgi::R32G32B32_FLOAT},
    {dxgi::R32G32B32_FLOAT, dxgi::R32G32B32_TYPELESS, dxgi::R32G32B32_FLOAT},
    {dxgi::R16G16B16A16_TYPELESS, dxgi::R16G16B16A16_TYPELESS, dxgi::R16G16B16A16_FLOAT},
    {dxgi::R16G16B16A16_FLOAT, dxgi::R16G16B16A16_TYPELESS, dxgi::R16G16B16A16_FLOAT},
    {dxgi::R16G16B16A16_UNORM, dxgi::R16G16B16A16_TYPELESS, dxgi::R16G16B16A16_UNORM},
    {dxgi::R32G32_TYPELESS, dxgi::R32G32_TYPELESS, dxgi::R32G32_FLOAT},
    {dxgi::R32G32_FLOAT, dxgi::R32G32_TYPELESS, dxgi::R32G32_FLOAT},
    {dxgi::R32G8X24_TYPELESS, dxgi::R32G8X24_TYPELESS, dxgi::R32_FLOAT_X8X24_TYPELESS},
    {dxgi::D32_FLOAT_S8X24_UINT, dxgi::R32G8X24_TYPELESS, dxgi::R32_FLOAT_X8X24_TYPELESS},
    {dxgi::R32_FLOAT_X8X24_TYPELESS, dxgi::R32G8X24_TYPELESS, dxgi::R32_FLOAT_X8X24_TYPELESS},
    {dxgi::R10G10B10A2_TYPELESS, dxgi::R10G10B10A2_TYPELESS, dxgi::R10G10B10A2_UNORM},
    {dxgi::R10G10B10A2_UNORM, dxgi::R10G10B10A2_TYPELESS, dxgi::R10G10B10A2_UNORM},
    {dxgi::R11G11B10_FLOAT, 0, dxgi::R11G11B10_FLOAT},
    {dxgi::R8G8B8A8_TYPELESS, dxgi::R8G8B8A8_TYPELESS, dxgi::R8G8B8A8_UNORM},
    {dxgi::R8G8B8A8_UNORM, dxgi::R8G8B8A8_TYPELESS, dxgi::R8G8B8A8_UNORM},
    {dxgi::R8G8B8A8_UNORM_SRGB, dxgi::R8G8B8A8_TYPELESS, dxgi::R8G8B8A8_UNORM_SRGB},
    {dxgi::R8G8B8A8_SNORM, dxgi::R8G8B8A8_TYPELESS, dxgi::R8G8B8A8_SNORM},
    {dxgi::R16G16_TYPELESS, dxgi::R16G16_TYPELESS, dxgi::R16G16_FLOAT},
    {dxgi::R16G16_FLOAT, dxgi::R16G16_TYPELESS, dxgi::R16G16_FLOAT},
    {dxgi::R16G16_UNORM, dxgi::R16G16_TYPELESS, dxgi::R16G16_UNORM},
    {dxgi::R16G16_UINT, dxgi::R16G16_TYPELESS, dxgi::R16G16_UINT},
    {dxgi::R16G16_SNORM, dxgi::R16G16_TYPELESS, dxgi::R16G16_SNORM},
    {dxgi::R16G16_SINT, dxgi::R16G16_TYPELESS, dxgi::R16G16_SINT},
    {dxgi::R32_TYPELESS, dxgi::R32_TYPELESS, dxgi::R32_FLOAT},
    {dxgi::D32_FLOAT, dxgi::R32_TYPELESS, dxgi::R32_FLOAT},
    {dxgi::R32_FLOAT, dxgi::R32_TYPELESS, dxgi::R32_FLOAT},
    {dxgi::R32_UINT, dxgi::R32_TYPELESS, dxgi::R32_UINT},
    {dxgi::R24G8_TYPELESS, dxgi::R24G8_TYPELESS, dxgi::R24_UNORM_X8_TYPELESS},
    {dxgi::D24_UNORM_S8_UINT, dxgi::R24G8_TYPELESS, dxgi::R24_UNORM_X8_TYPELESS},
    {dxgi::R24_UNORM_X8_TYPELESS, dxgi::R24G8_TYPELESS, dxgi::R24_UNORM_X8_TYPELESS},
    {dxgi::R8G8_TYPELESS, dxgi::R8G8_TYPELESS, dxgi::R8G8_UNORM},
    {dxgi::R8G8_UNORM, dxgi::R8G8_TYPELESS, dxgi::R8G8_UNORM},
    {dxgi::R8G8_UINT, dxgi::R8G8_TYPELESS, dxgi::R8G8_UINT},
    {dxgi::R16_TYPELESS, dxgi::R16_TYPELESS, dxgi::R16_UNORM},  // usually a 16-bit depth buffer
    {dxgi::R16_FLOAT, dxgi::R16_TYPELESS, dxgi::R16_FLOAT},
    {dxgi::D16_UNORM, dxgi::R16_TYPELESS, dxgi::R16_UNORM},
    {dxgi::R16_UNORM, dxgi::R16_TYPELESS, dxgi::R16_UNORM},
    {dxgi::R16_UINT, dxgi::R16_TYPELESS, dxgi::R16_UINT},
    {dxgi::R16_SNORM, dxgi::R16_TYPELESS, dxgi::R16_SNORM},
    {dxgi::R8_TYPELESS, dxgi::R8_TYPELESS, dxgi::R8_UNORM},
    {dxgi::R8_UNORM, dxgi::R8_TYPELESS, dxgi::R8_UNORM},
    {dxgi::R8_UINT, dxgi::R8_TYPELESS, dxgi::R8_UINT},
    {dxgi::R9G9B9E5_SHAREDEXP, 0, dxgi::R9G9B9E5_SHAREDEXP},
    {dxgi::B8G8R8A8_TYPELESS, dxgi::B8G8R8A8_TYPELESS, dxgi::B8G8R8A8_UNORM},
    {dxgi::B8G8R8A8_UNORM, dxgi::B8G8R8A8_TYPELESS, dxgi::B8G8R8A8_UNORM},
    {dxgi::B8G8R8A8_UNORM_SRGB, dxgi::B8G8R8A8_TYPELESS, dxgi::B8G8R8A8_UNORM_SRGB},
};

const FormatEntry* Find(uint32_t format) {
    for (const auto& e : kFormats)
        if (e.format == format) return &e;
    return nullptr;
}

WgTexture MapTexture(const FfxApiResource& r, ResourceInfoFn resourceInfo) {
    WgTexture t{};
    t.resource = r.resource;
    if (!r.resource) return t;
    ResourceInfo info;
    const bool known = resourceInfo && resourceInfo(r.resource, &info);
    t.format = ViewFormat(known ? info.dxgiFormat : 0, DxgiFromFfxSurfaceFormat(r.description.format));
    t.width = known && info.width ? info.width : r.description.width;
    t.height = known && info.height ? info.height : r.description.height;
    t.state = D3D12StatesFromFfx(r.state);
    return t;
}

void Note(std::string& notes, const char* text) {
    if (!notes.empty()) notes += "; ";
    notes += text;
}

}  // namespace

uint32_t DxgiFromFfxSurfaceFormat(uint32_t f) {
    switch (f) {
    case FFX_API_SURFACE_FORMAT_R32G32B32A32_TYPELESS: return dxgi::R32G32B32A32_TYPELESS;
    case FFX_API_SURFACE_FORMAT_R32G32B32A32_UINT: return dxgi::R32G32B32A32_UINT;
    case FFX_API_SURFACE_FORMAT_R32G32B32A32_FLOAT: return dxgi::R32G32B32A32_FLOAT;
    case FFX_API_SURFACE_FORMAT_R16G16B16A16_FLOAT: return dxgi::R16G16B16A16_FLOAT;
    case FFX_API_SURFACE_FORMAT_R32G32B32_FLOAT: return dxgi::R32G32B32_FLOAT;
    case FFX_API_SURFACE_FORMAT_R32G32_FLOAT: return dxgi::R32G32_FLOAT;
    case FFX_API_SURFACE_FORMAT_R8_UINT: return dxgi::R8_UINT;
    case FFX_API_SURFACE_FORMAT_R32_UINT: return dxgi::R32_UINT;
    case FFX_API_SURFACE_FORMAT_R8G8B8A8_TYPELESS: return dxgi::R8G8B8A8_TYPELESS;
    case FFX_API_SURFACE_FORMAT_R8G8B8A8_UNORM: return dxgi::R8G8B8A8_UNORM;
    case FFX_API_SURFACE_FORMAT_R8G8B8A8_SNORM: return dxgi::R8G8B8A8_SNORM;
    case FFX_API_SURFACE_FORMAT_R8G8B8A8_SRGB: return dxgi::R8G8B8A8_UNORM_SRGB;
    case FFX_API_SURFACE_FORMAT_B8G8R8A8_TYPELESS: return dxgi::B8G8R8A8_TYPELESS;
    case FFX_API_SURFACE_FORMAT_B8G8R8A8_UNORM: return dxgi::B8G8R8A8_UNORM;
    case FFX_API_SURFACE_FORMAT_B8G8R8A8_SRGB: return dxgi::B8G8R8A8_UNORM_SRGB;
    case FFX_API_SURFACE_FORMAT_R11G11B10_FLOAT: return dxgi::R11G11B10_FLOAT;
    case FFX_API_SURFACE_FORMAT_R10G10B10A2_UNORM: return dxgi::R10G10B10A2_UNORM;
    case FFX_API_SURFACE_FORMAT_R16G16_FLOAT: return dxgi::R16G16_FLOAT;
    case FFX_API_SURFACE_FORMAT_R16G16_UINT: return dxgi::R16G16_UINT;
    case FFX_API_SURFACE_FORMAT_R16G16_SINT: return dxgi::R16G16_SINT;
    case FFX_API_SURFACE_FORMAT_R16_FLOAT: return dxgi::R16_FLOAT;
    case FFX_API_SURFACE_FORMAT_R16_UINT: return dxgi::R16_UINT;
    case FFX_API_SURFACE_FORMAT_R16_UNORM: return dxgi::R16_UNORM;
    case FFX_API_SURFACE_FORMAT_R16_SNORM: return dxgi::R16_SNORM;
    case FFX_API_SURFACE_FORMAT_R8_UNORM: return dxgi::R8_UNORM;
    case FFX_API_SURFACE_FORMAT_R8G8_UNORM: return dxgi::R8G8_UNORM;
    case FFX_API_SURFACE_FORMAT_R8G8_UINT: return dxgi::R8G8_UINT;
    case FFX_API_SURFACE_FORMAT_R32_FLOAT: return dxgi::R32_FLOAT;
    case FFX_API_SURFACE_FORMAT_R9G9B9E5_SHAREDEXP: return dxgi::R9G9B9E5_SHAREDEXP;
    case FFX_API_SURFACE_FORMAT_R16G16B16A16_TYPELESS: return dxgi::R16G16B16A16_TYPELESS;
    case FFX_API_SURFACE_FORMAT_R32G32_TYPELESS: return dxgi::R32G32_TYPELESS;
    case FFX_API_SURFACE_FORMAT_R10G10B10A2_TYPELESS: return dxgi::R10G10B10A2_TYPELESS;
    case FFX_API_SURFACE_FORMAT_R16G16_TYPELESS: return dxgi::R16G16_TYPELESS;
    case FFX_API_SURFACE_FORMAT_R16_TYPELESS: return dxgi::R16_TYPELESS;
    case FFX_API_SURFACE_FORMAT_R8_TYPELESS: return dxgi::R8_TYPELESS;
    case FFX_API_SURFACE_FORMAT_R8G8_TYPELESS: return dxgi::R8G8_TYPELESS;
    case FFX_API_SURFACE_FORMAT_R32_TYPELESS: return dxgi::R32_TYPELESS;
    default: return dxgi::UNKNOWN;
    }
}

bool IsTypeless(uint32_t format) {
    const FormatEntry* e = Find(format);
    return e && e->family == format;
}

bool IsSrgb(uint32_t format) {
    return format == dxgi::R8G8B8A8_UNORM_SRGB || format == dxgi::B8G8R8A8_UNORM_SRGB;
}

uint32_t ViewFormat(uint32_t resourceFormat, uint32_t ffxDxgiFormat) {
    if (resourceFormat == dxgi::UNKNOWN) resourceFormat = ffxDxgiFormat;
    const FormatEntry* r = Find(resourceFormat);
    if (!r) return resourceFormat;  // a format this table does not know: use it as it is
    if (!IsTypeless(resourceFormat)) return r->view;
    const FormatEntry* f = Find(ffxDxgiFormat);
    if (f && !IsTypeless(ffxDxgiFormat) && f->family == resourceFormat) return f->view;
    return r->view;
}

uint32_t D3D12StatesFromFfx(uint32_t s) {
    uint32_t d = d3d12state::COMMON;
    if (s & FFX_API_RESOURCE_STATE_UNORDERED_ACCESS) d |= d3d12state::UNORDERED_ACCESS;
    if (s & FFX_API_RESOURCE_STATE_COMPUTE_READ) d |= d3d12state::NON_PIXEL_SHADER_RESOURCE;
    if (s & FFX_API_RESOURCE_STATE_PIXEL_READ) d |= d3d12state::PIXEL_SHADER_RESOURCE;
    if (s & FFX_API_RESOURCE_STATE_COPY_SRC) d |= d3d12state::COPY_SOURCE;
    if (s & FFX_API_RESOURCE_STATE_COPY_DEST) d |= d3d12state::COPY_DEST;
    if (s & FFX_API_RESOURCE_STATE_INDIRECT_ARGUMENT) d |= d3d12state::INDIRECT_ARGUMENT;
    if (s & FFX_API_RESOURCE_STATE_RENDER_TARGET) d |= d3d12state::RENDER_TARGET;
    // FFX_API_RESOURCE_STATE_COMMON and _PRESENT are both D3D12 state 0.
    return d;
}

uint32_t WgCreateFlagsFromFfx(uint32_t f) {
    uint32_t w = 0;
    if (f & FFX_UPSCALE_ENABLE_HIGH_DYNAMIC_RANGE) w |= WG_CREATE_HDR;
    if (f & FFX_UPSCALE_ENABLE_DEPTH_INVERTED) w |= WG_CREATE_DEPTH_INVERTED;
    if (f & FFX_UPSCALE_ENABLE_AUTO_EXPOSURE) w |= WG_CREATE_AUTO_EXPOSURE;
    if (f & FFX_UPSCALE_ENABLE_DEPTH_INFINITE) w |= WG_CREATE_DEPTH_INFINITE;
    if (f & FFX_UPSCALE_ENABLE_DISPLAY_RESOLUTION_MOTION_VECTORS) w |= WG_CREATE_MOTION_OUTPUT_RES;
    if (f & FFX_UPSCALE_ENABLE_NON_LINEAR_COLORSPACE) w |= WG_CREATE_NONLINEAR_COLOR;
    if (f & FFX_UPSCALE_ENABLE_DYNAMIC_RESOLUTION) w |= WG_CREATE_DYNAMIC_RESOLUTION;
    return w;
}

void MapCreate(const UpscaleCreateParams& create, void* device, WgCreateInfo& out) {
    out.struct_size = sizeof(WgCreateInfo);
    out.interface_version = WG_UPSCALER_INTERFACE_VERSION;
    out.d3d12_device = device;
    out.max_render_width = create.maxRenderSize.width;
    out.max_render_height = create.maxRenderSize.height;
    out.output_width = create.maxUpscaleSize.width;
    out.output_height = create.maxUpscaleSize.height;
    out.flags = WgCreateFlagsFromFfx(create.flags);
}

std::string WgResultText(int result) {
    switch (result) {
    case WG_OK: return "WG_OK";
    case WG_ERROR: return "WG_ERROR (unspecified failure; the upscaler's own log lines above say more)";
    case WG_ERROR_UNSUPPORTED: return "WG_ERROR_UNSUPPORTED (this input combination is not supported: format, size or flag)";
    case WG_ERROR_DEVICE: return "WG_ERROR_DEVICE (the D3D12 or HIP runtime returned an error)";
    case WG_ERROR_MEMORY: return "WG_ERROR_MEMORY (out of memory)";
    case WG_ERROR_PARAMETER: return "WG_ERROR_PARAMETER (a required input is missing or invalid)";
    case WG_ERROR_NOT_READY: return "WG_ERROR_NOT_READY (not possible this frame)";
    case WG_ERROR_INPUT: return "WG_ERROR_INPUT (the game's inputs can never be taken: a texture's shape, format or flags)";
    default: return "unknown result code " + std::to_string(result);
    }
}

uint32_t BackendVersion(const WgUpscalerBackend* backend) {
    if (!backend) return 0;
    return backend->interface_version >= 2 && backend->struct_size >= offsetof(WgUpscalerBackend, backend_flags) + sizeof(uint32_t) ? 2u : 1u;
}

uint32_t BackendFlags(const WgUpscalerBackend* backend) { return BackendVersion(backend) >= 2 ? backend->backend_flags : 0u; }

void AdaptCreateInfo(const WgUpscalerBackend* backend, const WgFrontEndServices* services, WgCreateInfo& info) {
    if (BackendVersion(backend) >= 2) {
        info.struct_size = sizeof(WgCreateInfo);
        info.interface_version = 2;
        info.services = services;
    } else {
        info.struct_size = offsetof(WgCreateInfo, services);
        info.interface_version = 1;
        info.services = nullptr;
    }
}

void AdaptFrame(const WgUpscalerBackend* backend, WgFrame& frame) {
    frame.struct_size = BackendVersion(backend) >= 2 ? sizeof(WgFrame) : offsetof(WgFrame, command_queue);
}

WgResult DispatchWithFallback(const WgUpscalerBackend* backend, WgUpscaler* upscaler, WgFrame& frame, const WgUpscalerBackend* standIn,
                              const WgCreateInfo& info, WgUpscaler** fallback, std::string& note) {
    AdaptFrame(backend, frame);
    const WgResult r = backend->dispatch(upscaler, &frame);
    if (r != WG_ERROR_NOT_READY) return r;
    if (!standIn || standIn == backend) {
        note = "upscaler not ready this frame and no stand-in available";
        return r;
    }
    if (!*fallback) {
        WgCreateInfo fi = info;
        AdaptCreateInfo(standIn, nullptr, fi);
        const WgResult c = standIn->create(&fi, fallback);
        if (c != WG_OK || !*fallback) {
            *fallback = nullptr;
            note = "upscaler not ready this frame; the stand-in could not be created: " + WgResultText(c);
            return c == WG_OK ? WG_ERROR : c;
        }
    }
    AdaptFrame(standIn, frame);
    const WgResult s = standIn->dispatch(*fallback, &frame);
    AdaptFrame(backend, frame);
    note = s == WG_OK ? "upscaler not ready this frame (WG_ERROR_NOT_READY): stand-in for this frame"
                      : "upscaler not ready this frame; the stand-in failed too: " + WgResultText(s);
    return s;
}

bool MapDispatch(const ffxDispatchDescUpscale& d, const UpscaleCreateParams& create, const Config& config,
                 ResourceInfoFn resourceInfo, MapState& state, uint64_t frameIndex, WgFrame& out, std::string& problem,
                 std::string& notes) {
    out = WgFrame{};
    out.struct_size = sizeof(WgFrame);
    out.command_list = d.commandList;
    out.color = MapTexture(d.color, resourceInfo);
    out.depth = MapTexture(d.depth, resourceInfo);
    out.motion = MapTexture(d.motionVectors, resourceInfo);
    out.exposure = MapTexture(d.exposure, resourceInfo);
    out.output = MapTexture(d.output, resourceInfo);
    out.frame_index = frameIndex;
    out.frame_time_ms = d.frameTimeDelta;

    // Sizes: FSR's render region always starts at (0, 0) of the render textures; upscaleSize is optional.
    out.render_x = 0;
    out.render_y = 0;
    out.render_width = d.renderSize.width;
    out.render_height = d.renderSize.height;
    const bool explicitOutput = d.upscaleSize.width != 0 && d.upscaleSize.height != 0;
    out.output_width = explicitOutput ? d.upscaleSize.width : create.maxUpscaleSize.width;
    out.output_height = explicitOutput ? d.upscaleSize.height : create.maxUpscaleSize.height;

    // Jitter: FSR's jitterOffset is the offset the camera projection was shifted by, in render pixels, with
    // +y moving the image down (AMD's projection example uses -2*jy/height in NDC), so the image moves by
    // +jitterOffset. The network's jitter is that image shift (spec 20 draft 2, spec 03; measured on recorded game
    // frames, ANSWERS.md): j = +jitterOffset. The signs stay configurable for test 20-T2.
    out.jitter[0] = config.jitterSign[0] * d.jitterOffset.x;
    out.jitter[1] = config.jitterSign[1] * d.jitterOffset.y;

    // Motion: FSR wants motionVectorScale * raw = screen-space motion from the current pixel to its position in
    // the previous frame, in pixels of the motion texture, +y down (FSR documentation, "Providing motion vectors").
    // That is the network's convention already (previous = current + motion); only the unit changes when the
    // vectors are at display resolution (display pixels -> render pixels).
    const bool motionAtOutput = (create.flags & FFX_UPSCALE_ENABLE_DISPLAY_RESOLUTION_MOTION_VECTORS) != 0;
    float toRender[2] = {1.0f, 1.0f};
    if (motionAtOutput && out.output_width && out.output_height) {
        toRender[0] = float(out.render_width) / float(out.output_width);
        toRender[1] = float(out.render_height) / float(out.output_height);
    }
    out.motion_scale[0] = config.motionSign[0] * d.motionVectorScale.x * toRender[0];
    out.motion_scale[1] = config.motionSign[1] * d.motionVectorScale.y * toRender[1];
    // Motion vectors that include the jitter (FFX_UPSCALE_ENABLE_MOTION_VECTORS_JITTER_CANCELLATION) were measured
    // between jittered positions: raw motion = true motion + (previous jitterOffset - current jitterOffset).
    if ((create.flags & FFX_UPSCALE_ENABLE_MOTION_VECTORS_JITTER_CANCELLATION) && state.hasPreviousJitter) {
        out.motion_offset[0] = d.jitterOffset.x - state.previousJitter[0];
        out.motion_offset[1] = d.jitterOffset.y - state.previousJitter[1];
    }
    state.previousJitter[0] = d.jitterOffset.x;
    state.previousJitter[1] = d.jitterOffset.y;
    state.hasPreviousJitter = true;

    // Exposure (spec 10 section 5 step 1): a game texture wins, then auto-exposure, then 1.
    if (d.exposure.resource) {
        out.exposure_mode = WG_EXPOSURE_TEXTURE;
        if (create.flags & FFX_UPSCALE_ENABLE_AUTO_EXPOSURE) Note(notes, "exposure texture given with auto-exposure on: texture used");
    } else if (create.flags & FFX_UPSCALE_ENABLE_AUTO_EXPOSURE) {
        out.exposure_mode = WG_EXPOSURE_COMPUTE;
    } else {
        out.exposure_mode = WG_EXPOSURE_VALUE;
    }
    out.exposure_value = 1.0f;
    if (std::isfinite(d.preExposure) && d.preExposure > 0.0f) {
        out.pre_exposure = d.preExposure;
    } else {
        out.pre_exposure = 1.0f;
        Note(notes, "preExposure <= 0 or not finite: 1 used");
    }
    out.reset = d.reset ? 1u : 0u;

    if (d.flags & (FFX_UPSCALE_FLAG_NON_LINEAR_COLOR_SRGB | FFX_UPSCALE_FLAG_NON_LINEAR_COLOR_PQ))
        Note(notes, "color flagged non-linear (sRGB/PQ) by the game");
    if (!d.depth.resource) Note(notes, "no depth");
    if (!d.motionVectors.resource) Note(notes, "no motion vectors");

    // Required inputs.
    if (!d.commandList) { problem = "no command list"; return false; }
    if (!d.color.resource) { problem = "no color input"; return false; }
    if (!d.output.resource) { problem = "no output"; return false; }
    if (!out.render_width || !out.render_height) { problem = "render size is 0"; return false; }
    if (!out.output_width || !out.output_height) { problem = "output size is 0"; return false; }
    if (out.color.width && out.color.height && (out.render_width > out.color.width || out.render_height > out.color.height)) {
        char text[128];
        std::snprintf(text, sizeof text, "render size %ux%u exceeds the color texture %ux%u", out.render_width,
                      out.render_height, out.color.width, out.color.height);
        problem = text;
        return false;
    }
    if (out.output.width && out.output.height && (out.output_width > out.output.width || out.output_height > out.output.height)) {
        char text[128];
        std::snprintf(text, sizeof text, "output size %ux%u exceeds the output texture %ux%u", out.output_width,
                      out.output_height, out.output.width, out.output.height);
        problem = text;
        return false;
    }
    if (out.render_width > create.maxRenderSize.width || out.render_height > create.maxRenderSize.height)
        Note(notes, "render size exceeds maxRenderSize of create");
    return true;
}

float UpscaleRatioFromQualityMode(uint32_t mode) {
    switch (mode) {
    case FFX_UPSCALE_QUALITY_MODE_NATIVEAA: return 1.0f;
    case FFX_UPSCALE_QUALITY_MODE_QUALITY: return 1.5f;
    case FFX_UPSCALE_QUALITY_MODE_BALANCED: return 1.7f;
    case FFX_UPSCALE_QUALITY_MODE_PERFORMANCE: return 2.0f;
    case FFX_UPSCALE_QUALITY_MODE_ULTRA_PERFORMANCE: return 3.0f;
    default: return 0.0f;
    }
}

int32_t JitterPhaseCount(uint32_t renderWidth, uint32_t displayWidth) {
    if (!renderWidth || !displayWidth) return 1;
    // AMD's documentation gives ceil(8 * n^2) for custom ratios, but its table (Quality 18, Balanced 23, Performance
    // 32, Ultra performance 72) and AMD's own answer in Cyberpunk 2077 (17 for 1707 -> 2560, results/2026-10-05-fsr-20-
    // 3989fff.md) are both reproduced by rounding down (FORMATS.md).
    const double n = double(displayWidth) / double(renderWidth);
    const double count = std::floor(8.0 * n * n);
    return count < 1.0 ? 1 : count > 1024.0 ? 1024 : int32_t(count);
}

void JitterOffset(int32_t index, int32_t phaseCount, float* x, float* y) {
    if (phaseCount < 1) phaseCount = 1;
    const int32_t i = ((index % phaseCount) + phaseCount) % phaseCount + 1;  // Halton index from 1
    auto halton = [](int32_t n, int32_t base) {
        double f = 1.0, r = 0.0;
        while (n > 0) {
            f /= base;
            r += f * (n % base);
            n /= base;
        }
        return r;
    };
    if (x) *x = float(halton(i, 2) - 0.5);
    if (y) *y = float(halton(i, 3) - 0.5);
}

}  // namespace wgfsr
