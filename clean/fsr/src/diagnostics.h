// SPDX-License-Identifier: MIT
#pragma once
// Original, metadata-only support schema. No images, resource bytes or learned data.
#include <array>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <map>
#include <string>
#include "../include/wronggpu_upscaler.h"

namespace wgdiag {
inline std::string Quote(const std::string& s) {
    std::string out = "\"";
    for (unsigned char c : s) {
        if (c == '"' || c == '\\') { out += '\\'; out += char(c); }
        else if (c < 32) { char b[7]; std::snprintf(b, sizeof b, "\\u%04x", unsigned(c)); out += b; }
        else out += char(c);
    }
    return out + '"';
}
inline std::string Number(float f) {
    if (!std::isfinite(f)) return "null";
    char b[48];
    const auto r = std::to_chars(b, b + sizeof b, f, std::chars_format::general, 9);
    return r.ec == std::errc() ? std::string(b, r.ptr) : "null";
}
struct Failure {
    const char* code = "WG-ERROR";
    const char* observedAt = "frontend";
    const char* domain = "WrongGPU";
    int numeric = -1;
};
inline Failure Classify(const std::string& message) {
    Failure f;
    const auto hipCode = message.find("HIP error ");
    if (hipCode != std::string::npos || message.find("hipStreamWaitValue32") != std::string::npos ||
        message.find("hipModuleLaunchKernel") != std::string::npos || message.find("hipInit failed") != std::string::npos) {
        f.domain = "HIP"; f.observedAt = "hip_api"; f.code = "WG-HIP-API";
        const auto p = hipCode == std::string::npos ? message.size() : hipCode + 10;
        if (p < message.size() && message[p] >= '0' && message[p] <= '9') {
            int n = 0;
            for (size_t i = p; i < message.size() && message[i] >= '0' && message[i] <= '9'; ++i) {
                if (n > 100000) { n = -1; break; }
                n = n * 10 + message[i] - '0';
            }
            f.numeric = n;
        }
        if (f.numeric == 719) f.code = "WG-HIP-719";
        if (message.find("hipStreamWaitValue32") != std::string::npos) f.observedAt = "input_wait";
        else if (message.find("hipModuleLaunchKernel") != std::string::npos) f.observedAt = "kernel_launch";
        else if (message.find("hipInit") != std::string::npos) f.observedAt = "hip_init";
    } else if (message.find("NO_PROVIDER") != std::string::npos) { f.code = "WG-NO-PROVIDER"; f.observedAt = "provider_resolution"; }
    else if (message.find("vtables") != std::string::npos) { f.code = "WG-HOOK-TABLE"; f.observedAt = "hook_registration"; }
    else if (message.find("split") != std::string::npos) { f.code = "WG-SPLIT"; f.observedAt = "command_list_split"; }
    else if (message.find("create") != std::string::npos) { f.code = "WG-CREATE"; f.observedAt = "context_create"; }
    return f;
}
// One compact CPU copy per dispatch. Pointer values identify objects within this
// session only; replay tooling must never treat them as usable addresses.
struct Frame {
    uint64_t context = 0, index = 0, commandList = 0, queue = 0;
    uint32_t rw = 0, rh = 0, ow = 0, oh = 0, x = 0, y = 0, reset = 0, queueType = 0, exposureMode = 0;
    float jitter[2]{}, motionScale[2]{}, motionOffset[2]{}, exposure = 0, preExposure = 0;
    struct Texture { uint32_t format, width, height, state; bool present; uint64_t resourceId; } textures[5]{};
};
inline Frame Capture(uint64_t context, const WgFrame& f) {
    Frame s;
    s.context = context; s.index = f.frame_index;
    s.commandList = uint64_t(uintptr_t(f.command_list)); s.queue = uint64_t(uintptr_t(f.command_queue));
    s.rw = f.render_width; s.rh = f.render_height; s.ow = f.output_width; s.oh = f.output_height;
    s.x = f.render_x; s.y = f.render_y; s.reset = f.reset; s.queueType = f.command_list_type;
    s.exposureMode = f.exposure_mode; s.exposure = f.exposure_value; s.preExposure = f.pre_exposure;
    for (int i = 0; i < 2; ++i) { s.jitter[i] = f.jitter[i]; s.motionScale[i] = f.motion_scale[i]; s.motionOffset[i] = f.motion_offset[i]; }
    const WgTexture* ts[] = {&f.color, &f.depth, &f.motion, &f.exposure, &f.output};
    for (int i = 0; i < 5; ++i) s.textures[i] = {ts[i]->format, ts[i]->width, ts[i]->height, ts[i]->state,
        ts[i]->resource != nullptr, uint64_t(uintptr_t(ts[i]->resource))};
    return s;
}
inline std::string Json(const Frame& f) {
    std::string out = "{\"context\":" + std::to_string(f.context) + ",\"frame\":" + std::to_string(f.index) +
        ",\"command_list_id\":" + Quote(std::to_string(f.commandList)) + ",\"queue_id\":" + Quote(std::to_string(f.queue)) +
        ",\"command_list_type\":" + std::to_string(f.queueType) + ",\"render\":[" + std::to_string(f.rw) + "," + std::to_string(f.rh) +
        "],\"output\":[" + std::to_string(f.ow) + "," + std::to_string(f.oh) + "],\"origin\":[" + std::to_string(f.x) + "," + std::to_string(f.y) +
        "],\"reset\":" + std::to_string(f.reset) + ",\"jitter\":[" + Number(f.jitter[0]) + "," + Number(f.jitter[1]) +
        "],\"motion_scale\":[" + Number(f.motionScale[0]) + "," + Number(f.motionScale[1]) + "],\"motion_offset\":[" +
        Number(f.motionOffset[0]) + "," + Number(f.motionOffset[1]) + "],\"exposure_mode\":" + std::to_string(f.exposureMode) +
        ",\"exposure\":" + Number(f.exposure) + ",\"pre_exposure\":" + Number(f.preExposure) + ",\"textures\":[";
    for (int i = 0; i < 5; ++i) {
        const auto& t = f.textures[i];
        if (i) out += ',';
        const char* roles[] = {"color", "depth", "motion", "exposure", "output"};
        out += "{\"role\":" + Quote(roles[i]) + ",\"resource_id\":" + Quote(std::to_string(t.resourceId)) +
            ",\"present\":" + std::string(t.present ? "true" : "false") + ",\"format\":" + std::to_string(t.format) +
            ",\"size\":[" + std::to_string(t.width) + "," + std::to_string(t.height) + "],\"state\":" + std::to_string(t.state) + "}";
    }
    return out + "]}";
}
struct Ring {
    std::array<Frame, 64> frames{};
    size_t next = 0, count = 0;
    void Push(const Frame& f) { frames[next] = f; next = (next + 1) % frames.size(); if (count < frames.size()) ++count; }
    template<class F> void Each(F emit) const {
        const size_t start = (next + frames.size() - count) % frames.size();
        for (size_t i = 0; i < count; ++i) emit(frames[(start + i) % frames.size()]);
    }
};
// Bug hunt F1: a failure repeated every frame wrote a full incident each time (the failure, its stack, the 64-frame ring
// and a module snapshot: ~100 KB per frame). One full incident per failure code per window; repeats in between are only
// counted, and the next full incident of that code says how many there were.
struct IncidentGate {
    static constexpr uint64_t kWindowMs = 30000;
    struct Entry { uint64_t lastMs = 0, repeats = 0; };
    std::map<std::string, Entry> codes;  // (the classifier's codes: a short list)
    // true: write the full incident now; *repeats = the ones only counted since the last full one
    bool Allow(const std::string& code, uint64_t nowMs, uint64_t* repeats) {
        auto it = codes.find(code);
        if (it != codes.end() && nowMs - it->second.lastMs < kWindowMs) { ++it->second.repeats; return false; }
        Entry& e = codes[code];
        *repeats = e.repeats;
        e.repeats = 0;
        e.lastMs = nowMs;
        return true;
    }
};
} // namespace wgdiag
