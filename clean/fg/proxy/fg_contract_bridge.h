// SPDX-License-Identifier: MIT
// WrongGPU frame generation (issue #96, F8a): the adapter contract (fg_adapter_abi.h) to the FG runtime contract
// (clean/fg/runtime/fg_contract.h, namespace wgfg). Header-only; include it only where fg_contract.h is on the include
// path. The two contracts carry the same fields in the same order; this file converts field by field and checks at
// compile time that every shared number (presets, N, optional inputs, frame flags, refusal and real-only reasons) is the
// same on both sides. Written for this project.
#pragma once
#include <cstring>

#include "fg_adapter_abi.h"
#include "fg_contract.h"

namespace wgfgp {

static_assert(wgfg::kFgMaxGenerated == WG_FG_MAX_GENERATED, "N range");
static_assert(wgfg::kFgPresetCount == WG_FG_PRESET_COUNT, "preset count");
static_assert(static_cast<uint32_t>(wgfg::FgPreset::A) == WG_FG_PRESET_A &&
              static_cast<uint32_t>(wgfg::FgPreset::B) == WG_FG_PRESET_B &&
              static_cast<uint32_t>(wgfg::FgPreset::C) == WG_FG_PRESET_C, "presets");
static_assert(wgfg::kFgOptCount == WG_FG_OPTIONAL_COUNT &&
              wgfg::kFgOptHudless == WG_FG_ROLE_HUDLESS - WG_FG_OPTIONAL_FIRST_ROLE &&
              wgfg::kFgOptUiAlpha == WG_FG_ROLE_UI_ALPHA - WG_FG_OPTIONAL_FIRST_ROLE &&
              wgfg::kFgOptDistortion == WG_FG_ROLE_DISTORTION - WG_FG_OPTIONAL_FIRST_ROLE &&
              wgfg::kFgOptNoWarp == WG_FG_ROLE_NO_WARP - WG_FG_OPTIONAL_FIRST_ROLE, "optional inputs");
static_assert(static_cast<uint32_t>(wgfg::FgOptionalDecl::Never) == WG_FG_DECL_NEVER &&
              static_cast<uint32_t>(wgfg::FgOptionalDecl::May) == WG_FG_DECL_MAY &&
              static_cast<uint32_t>(wgfg::FgOptionalDecl::Always) == WG_FG_DECL_ALWAYS, "optional declarations");
static_assert(wgfg::kFgFlagReset == WG_FG_FLAG_RESET && wgfg::kFgFlagMenu == WG_FG_FLAG_MENU &&
              wgfg::kFgFlagDepthReversed == WG_FG_FLAG_DEPTH_REVERSED &&
              wgfg::kFgFlagMotionRenderRes == WG_FG_FLAG_MOTION_RENDER_RES && wgfg::kFgFlagHdr == WG_FG_FLAG_HDR &&
              wgfg::kFgFlagRealOutput == WG_FG_FLAG_REAL_OUTPUT, "frame flags");
#define WG_FG_SAME_REASON(n, v) static_assert(static_cast<uint32_t>(wgfg::FgRefusal::n) == (v), #n)
WG_FG_SAME_REASON(StructSize, WG_FG_REASON_STRUCT_SIZE);
WG_FG_SAME_REASON(AbiVersion, WG_FG_REASON_ABI_VERSION);
WG_FG_SAME_REASON(BadPreset, WG_FG_REASON_BAD_PRESET);
WG_FG_SAME_REASON(BadGeneratedCount, WG_FG_REASON_BAD_GENERATED_COUNT);
WG_FG_SAME_REASON(BadOptionalDecl, WG_FG_REASON_BAD_OPTIONAL_DECL);
WG_FG_SAME_REASON(TokenNotMonotonic, WG_FG_REASON_TOKEN_NOT_MONOTONIC);
WG_FG_SAME_REASON(ViewportMismatch, WG_FG_REASON_VIEWPORT_MISMATCH);
WG_FG_SAME_REASON(LuidMismatch, WG_FG_REASON_LUID_MISMATCH);
WG_FG_SAME_REASON(QueueEpochMismatch, WG_FG_REASON_QUEUE_EPOCH_MISMATCH);
WG_FG_SAME_REASON(SwapchainEpochMismatch, WG_FG_REASON_SWAPCHAIN_EPOCH_MISMATCH);
WG_FG_SAME_REASON(MissingColor, WG_FG_REASON_MISSING_COLOR);
WG_FG_SAME_REASON(MissingMotion, WG_FG_REASON_MISSING_MOTION);
WG_FG_SAME_REASON(MissingDepth, WG_FG_REASON_MISSING_DEPTH);
WG_FG_SAME_REASON(MissingPromisedOptional, WG_FG_REASON_MISSING_PROMISED_OPTIONAL);
WG_FG_SAME_REASON(UndeclaredOptional, WG_FG_REASON_UNDECLARED_OPTIONAL);
WG_FG_SAME_REASON(ConflictingTag, WG_FG_REASON_CONFLICTING_TAG);
WG_FG_SAME_REASON(StaleTag, WG_FG_REASON_STALE_TAG);
WG_FG_SAME_REASON(InstallOff, WG_FG_REASON_INSTALL_OFF);
WG_FG_SAME_REASON(UnsupportedAdapter, WG_FG_REASON_UNSUPPORTED_ADAPTER);
WG_FG_SAME_REASON(ModelNotValidated, WG_FG_REASON_MODEL_NOT_VALIDATED);
WG_FG_SAME_REASON(PresenterNotQualified, WG_FG_REASON_PRESENTER_NOT_QUALIFIED);
WG_FG_SAME_REASON(RouteNotQualified, WG_FG_REASON_ROUTE_NOT_QUALIFIED);
#undef WG_FG_SAME_REASON
static_assert(static_cast<uint32_t>(wgfg::FgRealOnlyReason::Menu) == WG_FG_REAL_ONLY_MENU &&
              static_cast<uint32_t>(wgfg::FgRealOnlyReason::EpochChange) == WG_FG_REAL_ONLY_EPOCH_CHANGE &&
              static_cast<uint32_t>(wgfg::FgRealOnlyReason::Disabled) == WG_FG_REAL_ONLY_DISABLED, "real-only reasons");

inline wgfg::FgRect ToRuntime(const WgFgRect& r) {
    wgfg::FgRect o;
    o.x = r.x;
    o.y = r.y;
    o.width = r.width;
    o.height = r.height;
    return o;
}

inline wgfg::FgLuid ToRuntime(const WgFgLuid& l) {
    wgfg::FgLuid o;
    o.low = l.low;
    o.high = l.high;
    return o;
}

inline wgfg::FgTextureRef ToRuntime(const WgFgTexture& t) {
    wgfg::FgTextureRef o;
    o.resource = t.resource;
    o.format = t.format;
    o.width = t.width;
    o.height = t.height;
    o.subresource = t.subresource;
    o.state = t.state;
    o.producer_epoch = t.producer_epoch;
    o.frame_token = t.frame_token;
    o.viewport_id = t.viewport_id;
    o.rect = ToRuntime(t.rect);
    return o;
}

inline wgfg::FgCreateDesc ToRuntime(const WgFgSessionDesc& s) {
    wgfg::FgCreateDesc o;  // struct_size / abi_version: the runtime's own
    o.device = s.device;
    o.queue = s.queue;
    o.adapter_luid = ToRuntime(s.adapter_luid);
    o.swapchain_id = s.swapchain_id;
    o.window_id = s.window_id;
    o.queue_epoch = s.queue_epoch;
    o.swapchain_epoch = s.swapchain_epoch;
    o.viewport_id = s.viewport_id;
    o.output_width = s.output_width;
    o.output_height = s.output_height;
    o.output_format = s.output_format;
    o.render_min_width = s.render_min_width;
    o.render_min_height = s.render_min_height;
    o.render_max_width = s.render_max_width;
    o.render_max_height = s.render_max_height;
    o.dynamic_resolution = s.dynamic_resolution;
    o.preset = static_cast<wgfg::FgPreset>(s.preset);
    o.max_generated = s.max_generated;
    for (uint32_t i = 0; i < WG_FG_OPTIONAL_COUNT; ++i) o.optional[i] = static_cast<wgfg::FgOptionalDecl>(s.optional_decl[i]);
    o.hdr = s.hdr;
    o.presenter_mode = s.presenter_mode;
    return o;
}

inline wgfg::FgFramePacket ToRuntime(const WgFgPacket& p) {
    wgfg::FgFramePacket o;  // struct_size / abi_version: the runtime's own
    o.frame_token = p.frame_token;
    o.viewport_id = p.viewport_id;
    o.adapter_luid = ToRuntime(p.adapter_luid);
    o.queue_epoch = p.queue_epoch;
    o.swapchain_epoch = p.swapchain_epoch;
    o.color = ToRuntime(p.color);
    o.motion = ToRuntime(p.motion);
    o.depth = ToRuntime(p.depth);
    for (uint32_t i = 0; i < WG_FG_OPTIONAL_COUNT; ++i) o.optional[i] = ToRuntime(p.optional[i]);
    o.optional_present = p.optional_present;
    o.render_rect = ToRuntime(p.render_rect);
    o.output_rect = ToRuntime(p.output_rect);
    o.jitter_x = p.jitter_x;
    o.jitter_y = p.jitter_y;
    o.motion_scale_x = p.motion_scale_x;
    o.motion_scale_y = p.motion_scale_y;
    std::memcpy(o.camera_current, p.camera_current, sizeof(o.camera_current));
    std::memcpy(o.camera_previous, p.camera_previous, sizeof(o.camera_previous));
    o.flags = p.flags;
    o.preset = static_cast<wgfg::FgPreset>(p.preset);
    o.generated_count = p.generated_count;
    o.generated_index = p.generated_index;
    return o;
}

}  // namespace wgfgp
