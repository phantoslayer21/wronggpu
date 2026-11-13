// SPDX-License-Identifier: MIT
// Test doubles for AMD's FSR 3.0 component DLLs, built twice:
//   FAKE_UPSCALER -> ffx_fsr3upscaler_x64.wronggpu-original.dll
//   FAKE_FSR3     -> ffx_fsr3_x64.wronggpu-original.dll
// They do no GPU work; they record every call (read back through the extra export fakeLog). The fake ffx_fsr3 imports
// ffx_fsr3upscaler_x64.dll by name (as AMD's dynamically linked FSR 3.0 build does) and, when the environment variable
// WG_FAKE_FSR3_CALLS_UPSCALER is 1, creates and dispatches its upscaler through it, passing non-null frame-generation
// resources (as AMD's does even with UPSCALING_ONLY); otherwise it behaves as if the upscaler were built into it.
#define WIN32_LEAN_AND_MEAN
#include <windows.h>

#include <cstdio>
#include <cstring>
#include <mutex>
#include <string>

#include "../src/fsr30_types.h"

using namespace wgfsr::fsr30;

namespace {
std::mutex g_mutex;
std::string g_log;

void Record(const char* fn, const void* context, const char* extra = "") {
    char line[256];
    std::snprintf(line, sizeof line, "%s ctx=%p %s\n", fn, context, extra);
    std::lock_guard<std::mutex> lock(g_mutex);
    g_log += line;
}
}  // namespace

extern "C" {

__declspec(dllexport) const char* fakeLog() {
    std::lock_guard<std::mutex> lock(g_mutex);
    static std::string copy;
    copy = g_log;
    return copy.c_str();
}

#if defined(FAKE_UPSCALER)

__declspec(dllexport) ErrorCode ffxFsr3UpscalerContextCreate(void* ctx, const UpscalerContextDescription* d) {
    char extra[96];
    std::snprintf(extra, sizeof extra, "flags=0x%x display=%ux%u", d->flags, d->displaySize.width, d->displaySize.height);
    Record("upscaler.create", ctx, extra);
    std::memset(ctx, 0xA5, 64);  // AMD keeps its state in the caller's context memory
    return OK;
}
__declspec(dllexport) ErrorCode ffxFsr3UpscalerGetSharedResourceDescriptions(void* ctx, void*) {
    Record("upscaler.shared", ctx);
    return OK;
}
__declspec(dllexport) ErrorCode ffxFsr3UpscalerContextDispatch(void* ctx, const UpscalerDispatchDescription* d) {
    Record("upscaler.dispatch", ctx, d->dilatedDepth.resource ? "shared=1" : "shared=0");
    return OK;
}
__declspec(dllexport) ErrorCode ffxFsr3UpscalerContextGenerateReactiveMask(void* ctx, const UpscalerGenerateReactiveDescription*) {
    Record("upscaler.reactive", ctx);
    return OK;
}
__declspec(dllexport) ErrorCode ffxFsr3UpscalerContextDestroy(void* ctx) {
    Record("upscaler.destroy", ctx);
    return OK;
}
__declspec(dllexport) float ffxFsr3UpscalerGetUpscaleRatioFromQualityMode(int32_t mode) {
    Record("upscaler.ratio", nullptr);
    return mode == 1 ? 1.5f : 2.0f;
}
__declspec(dllexport) ErrorCode ffxFsr3UpscalerGetRenderResolutionFromQualityMode(uint32_t* w, uint32_t* h, uint32_t dw, uint32_t dh, int32_t) {
    Record("upscaler.renderres", nullptr);
    *w = dw * 2 / 3;
    *h = dh * 2 / 3;
    return OK;
}
__declspec(dllexport) int32_t ffxFsr3UpscalerGetJitterPhaseCount(int32_t, int32_t) {
    Record("upscaler.phasecount", nullptr);
    return 77;
}
__declspec(dllexport) ErrorCode ffxFsr3UpscalerGetJitterOffset(float* x, float* y, int32_t, int32_t) {
    Record("upscaler.jitter", nullptr);
    *x = 0.125f;
    *y = -0.25f;
    return OK;
}
__declspec(dllexport) bool ffxFsr3UpscalerResourceIsNull(Resource r) {  // by value, as AMD declares it
    Record("upscaler.isnull", nullptr, r.resource ? "resource" : "null");
    return r.resource == nullptr;
}

#elif defined(FAKE_FSR3)

}  // extern "C"

// Imported by name from ffx_fsr3upscaler_x64.dll (WrongGPU's forwarder in the tests), as AMD's dynamically linked
// ffx_fsr3 does.
extern "C" {
__declspec(dllimport) ErrorCode ffxFsr3UpscalerContextCreate(void*, const UpscalerContextDescription*);
__declspec(dllimport) ErrorCode ffxFsr3UpscalerContextDispatch(void*, const UpscalerDispatchDescription*);
__declspec(dllimport) ErrorCode ffxFsr3UpscalerContextDestroy(void*);
}

namespace {
bool CallsUpscaler() {
    char value[8] = {};
    return GetEnvironmentVariableA("WG_FAKE_FSR3_CALLS_UPSCALER", value, sizeof value) && value[0] == '1';
}
// The upscaler context lives inside the FSR 3 context memory, after a small header.
void* InnerContext(void* ctx) { return static_cast<char*>(ctx) + 4096; }
char g_sharedDummy[3][16];  // stand-ins for the frame-generation resources (never touched by the GPU here)
}  // namespace

extern "C" {

__declspec(dllexport) ErrorCode ffxFsr3ContextCreate(void* ctx, Fsr3ContextDescription* d) {
    Record("fsr3.create", ctx);
    if (CallsUpscaler()) {
        UpscalerContextDescription u{};
        u.flags = d->flags & kSharedUpscaleFlags;
        u.maxRenderSize = d->maxRenderSize;
        u.displaySize = d->upscaleOutputSize;
        u.backendInterface = d->backendInterfaceUpscaling;
        u.fpMessage = d->fpMessage;
        return ffxFsr3UpscalerContextCreate(InnerContext(ctx), &u);
    }
    return OK;
}
__declspec(dllexport) ErrorCode ffxFsr3ContextDispatchUpscale(void* ctx, const Fsr3DispatchUpscaleDescription* d) {
    Record("fsr3.dispatchUpscale", ctx);
    if (CallsUpscaler()) {
        UpscalerDispatchDescription u{};
        u.commandList = d->commandList;
        u.color = d->color;
        u.depth = d->depth;
        u.motionVectors = d->motionVectors;
        u.exposure = d->exposure;
        u.reactive = d->reactive;
        u.transparencyAndComposition = d->transparencyAndComposition;
        u.dilatedDepth.resource = g_sharedDummy[0];
        u.dilatedMotionVectors.resource = g_sharedDummy[1];
        u.reconstructedPrevNearestDepth.resource = g_sharedDummy[2];
        u.output = d->upscaleOutput;
        u.jitterOffset = d->jitterOffset;
        u.motionVectorScale = d->motionVectorScale;
        u.renderSize = d->renderSize;
        u.preExposure = d->preExposure;
        u.reset = d->reset;
        u.frameTimeDelta = d->frameTimeDelta;
        return ffxFsr3UpscalerContextDispatch(InnerContext(ctx), &u);
    }
    return OK;
}
__declspec(dllexport) ErrorCode ffxFsr3ContextDestroy(void* ctx) {
    Record("fsr3.destroy", ctx);
    if (CallsUpscaler()) return ffxFsr3UpscalerContextDestroy(InnerContext(ctx));
    return OK;
}
__declspec(dllexport) ErrorCode ffxFsr3DispatchFrameGeneration(const FrameGenerationDispatchDescription*) {
    Record("fsr3.dispatchFrameGeneration", nullptr);
    return OK;
}
__declspec(dllexport) ErrorCode ffxFsr3SkipPresent(void* ctx) {
    Record("fsr3.skipPresent", ctx);
    return OK;
}
__declspec(dllexport) ErrorCode ffxFsr3ContextGenerateReactiveMask(void* ctx, const UpscalerGenerateReactiveDescription*) {
    Record("fsr3.reactive", ctx);
    return OK;
}
__declspec(dllexport) ErrorCode ffxFsr3ConfigureFrameGeneration(void* ctx, const FrameGenerationConfig* c) {
    Record("fsr3.configureFrameGeneration", ctx, c->frameGenerationEnabled ? "enabled=1" : "enabled=0");
    return OK;
}
__declspec(dllexport) float ffxFsr3GetUpscaleRatioFromQualityMode(int32_t) {
    Record("fsr3.ratio", nullptr);
    return 1.5f;
}
__declspec(dllexport) ErrorCode ffxFsr3GetRenderResolutionFromQualityMode(uint32_t* w, uint32_t* h, uint32_t dw, uint32_t dh, int32_t) {
    Record("fsr3.renderres", nullptr);
    *w = dw * 2 / 3;
    *h = dh * 2 / 3;
    return OK;
}
__declspec(dllexport) int32_t ffxFsr3GetJitterPhaseCount(int32_t, int32_t) {
    Record("fsr3.phasecount", nullptr);
    return 66;
}
__declspec(dllexport) ErrorCode ffxFsr3GetJitterOffset(float* x, float* y, int32_t, int32_t) {
    Record("fsr3.jitter", nullptr);
    *x = 0.25f;
    *y = 0.375f;
    return OK;
}
__declspec(dllexport) bool ffxFsr3ResourceIsNull(Resource r) {
    Record("fsr3.isnull", nullptr, r.resource ? "resource" : "null");
    return r.resource == nullptr;
}

#endif

}  // extern "C"
