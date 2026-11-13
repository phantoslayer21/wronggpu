// SPDX-License-Identifier: MIT
#pragma once
// FSR 3.0 component DLL forwarders (spec 20, integration point 2): ffx_fsr3upscaler_x64.dll and ffx_fsr3_x64.dll.
// Every export is forwarded to AMD's original DLL, kept next to ours as <name>.wronggpu-original.dll, and logged with
// all its parameters and the module that called it. With upscaler = standin the stand-in (or later the network)
// takes over the upscale output: see clean/fsr/README.md "FSR 3.0 component DLLs" and clean/FORMATS.md.
//
// No Windows or D3D12 code here: the DLL glue (dll_fsr3upscaler.cpp, dll_fsr3.cpp) and the unit tests provide the
// Platform.
#include <atomic>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <string>

#include "../include/wronggpu_upscaler.h"
#include "amd_compare.h"
#include "async_create.h"
#include "config.h"
#include "status.h"
#include "fsr30_types.h"
#include "param_map.h"

namespace wgfsr {
namespace fsr30 {

// AMD's ffx_fsr3upscaler_x64.dll. The last four functions exist only in SDK 1.1 (FSR 3.1) builds of the DLL.
struct UpscalerOriginal {
    ErrorCode (*contextCreate)(void* context, const UpscalerContextDescription* desc) = nullptr;
    ErrorCode (*getSharedResourceDescriptions)(void* context, void* out) = nullptr;
    ErrorCode (*contextDispatch)(void* context, const UpscalerDispatchDescription* desc) = nullptr;
    ErrorCode (*contextGenerateReactiveMask)(void* context, const UpscalerGenerateReactiveDescription* desc) = nullptr;
    ErrorCode (*contextDestroy)(void* context) = nullptr;
    float (*getUpscaleRatioFromQualityMode)(int32_t mode) = nullptr;
    ErrorCode (*getRenderResolutionFromQualityMode)(uint32_t* w, uint32_t* h, uint32_t dw, uint32_t dh, int32_t mode) = nullptr;
    int32_t (*getJitterPhaseCount)(int32_t renderWidth, int32_t displayWidth) = nullptr;
    ErrorCode (*getJitterOffset)(float* x, float* y, int32_t index, int32_t phaseCount) = nullptr;
    bool (*resourceIsNull)(const void* resourceCopy) = nullptr;  // FfxResource by value = pointer to the caller's copy (x64)
    ErrorCode (*contextGetGpuMemoryUsage)(void* context, void* usage) = nullptr;
    uint32_t (*getEffectVersion)() = nullptr;
    ErrorCode (*setConstant)(void* context, int32_t key, void* value) = nullptr;
    ErrorCode (*setGlobalDebugMessage)(void* fn, uint32_t level) = nullptr;
    bool sdk11 = false;  // the DLL exports SDK 1.1 functions: its structures are not the 3.0.x ones, decode nothing
};

// AMD's ffx_fsr3_x64.dll. The last three functions exist only in SDK 1.1 builds.
struct Fsr3Original {
    ErrorCode (*dispatchFrameGeneration)(const FrameGenerationDispatchDescription* desc) = nullptr;
    ErrorCode (*contextCreate)(void* context, Fsr3ContextDescription* desc) = nullptr;
    ErrorCode (*contextDispatchUpscale)(void* context, const Fsr3DispatchUpscaleDescription* desc) = nullptr;
    ErrorCode (*skipPresent)(void* context) = nullptr;
    ErrorCode (*contextGenerateReactiveMask)(void* context, const UpscalerGenerateReactiveDescription* desc) = nullptr;
    ErrorCode (*configureFrameGeneration)(void* context, const FrameGenerationConfig* config) = nullptr;
    ErrorCode (*contextDestroy)(void* context) = nullptr;
    float (*getUpscaleRatioFromQualityMode)(int32_t mode) = nullptr;
    ErrorCode (*getRenderResolutionFromQualityMode)(uint32_t* w, uint32_t* h, uint32_t dw, uint32_t dh, int32_t mode) = nullptr;
    int32_t (*getJitterPhaseCount)(int32_t renderWidth, int32_t displayWidth) = nullptr;
    ErrorCode (*getJitterOffset)(float* x, float* y, int32_t index, int32_t phaseCount) = nullptr;
    bool (*resourceIsNull)(const void* resourceCopy) = nullptr;
    ErrorCode (*contextGetGpuMemoryUsage)(void* context, void* upscaler, void* opticalFlow, void* frameGeneration) = nullptr;
    ErrorCode (*contextDispatchFrameGenerationPrepare)(void* context, const void* desc) = nullptr;
    uint32_t (*getEffectVersion)() = nullptr;
    bool sdk11 = false;
};

struct Platform {
    std::function<const UpscalerOriginal*()> upscaler;  // AMD's upscaler DLL (null when not loaded / not this DLL)
    std::function<const Fsr3Original*()> fsr3;           // AMD's ffx_fsr3 DLL
    std::function<const WgUpscalerBackend*(const Config&, std::string& why)> backend;
    std::function<void(const std::string&)> log;
    std::function<void()> flushLog;  // writes the queued log lines now (after an upscaler's "about to ..." line); may be empty
    std::function<uint64_t()> logBytes;
    ResourceInfoFn resourceInfo = nullptr;
    std::function<void*(void* commandList)> deviceFromCommandList;  // ID3D12Device* of an ID3D12GraphicsCommandList*
    // ffx_fsr3 side: how many ffxFsr3UpscalerContextDispatch calls WrongGPU's upscaler DLL has seen on this thread
    // (empty or 0 when WrongGPU's upscaler DLL is not loaded).
    std::function<uint64_t()> upscalerSeenOnThread;
    // ffx_fsr3 side: around AMD's ffxFsr3ContextCreate, tell WrongGPU's upscaler DLL whether the upscaler context
    // created inside it belongs to an UPSCALING_ONLY context (1), not (0), or nothing is pending (-1).
    std::function<void(int)> hintUpscalingOnly;
    // Interface version 2 (same-frame split). services: the split service for a backend (installs the D3D12 hooks
    // when the backend sets WG_BACKEND_NEEDS_SPLIT); null when unavailable. noteDispatch: every upscale dispatch with
    // its command list (also those forwarded to AMD): submission trace, hooks with d3d12_hooks = on, and the frame's
    // command_queue / command_list_type (frame may be null). standIn: the stand-in, used for one frame when a backend
    // answers WG_ERROR_NOT_READY.
    std::function<const WgFrontEndServices*(const WgUpscalerBackend*, void* device)> services;
    std::function<void(void* commandList, WgFrame* frame)> noteDispatch;
    std::function<void(uint64_t context, const WgFrame&)> diagnosticFrame;
    std::function<void(uint64_t context, uint64_t frame, void* device)> diagnosticFailure;
    std::function<void(uint64_t context, uint64_t frame, WgResult)> diagnosticResult;
    std::function<const WgUpscalerBackend*(const Config&)> standIn;
    // Every upscale dispatch WrongGPU's upscaler recorded (WG_OK), with its frame (frame capture, capture.h); standIn:
    // the one-frame stand-in upscaled it. May be empty.
    std::function<void(const WgFrame& frame, bool standIn)> afterDispatch;
    // Every upscale dispatch of a WrongGPU context: what upscaled it (for the in-game overlay); may be empty.
    std::function<void(const UpscaleStatus&)> status;
    GpuTimerFn gpuTimer;  // brackets every upscale dispatch (WrongGPU's and AMD's) for the overlay; may be empty
    CompareFn compareAmd;  // the player switched to AMD's FSR (comparison): WrongGPU's contexts dispatch to AMD;
                           // may be empty
    // AMD's FidelityFX API DLL for comparisons (FSR 4 / FSR 3.1); null or empty: AMD's FSR 3.0 calls compare.
    std::function<const OriginalApi*()> compareApi;
};

class Hook {
public:
    Hook(Platform platform, Config config, std::string tag);
    ~Hook();

    // ffx_fsr3upscaler_x64.dll exports.
    ErrorCode UpContextCreate(void* ctx, const UpscalerContextDescription* desc, const char* caller);
    ErrorCode UpGetSharedResourceDescriptions(void* ctx, void* out, const char* caller);
    ErrorCode UpContextDispatch(void* ctx, const UpscalerDispatchDescription* desc, const char* caller);
    ErrorCode UpContextGenerateReactiveMask(void* ctx, const UpscalerGenerateReactiveDescription* desc, const char* caller);
    ErrorCode UpContextDestroy(void* ctx, const char* caller);
    float UpGetUpscaleRatioFromQualityMode(int32_t mode, const char* caller);
    ErrorCode UpGetRenderResolutionFromQualityMode(uint32_t* w, uint32_t* h, uint32_t dw, uint32_t dh, int32_t mode, const char* caller);
    int32_t UpGetJitterPhaseCount(int32_t rw, int32_t dw, const char* caller);
    ErrorCode UpGetJitterOffset(float* x, float* y, int32_t index, int32_t count, const char* caller);
    bool UpResourceIsNull(const void* resourceCopy, const char* caller);
    ErrorCode UpContextGetGpuMemoryUsage(void* ctx, void* usage, const char* caller);
    uint32_t UpGetEffectVersion(const char* caller);
    ErrorCode UpSetConstant(void* ctx, int32_t key, void* value, const char* caller);
    ErrorCode UpSetGlobalDebugMessage(void* fn, uint32_t level, const char* caller);

    // ffx_fsr3_x64.dll exports.
    ErrorCode F3DispatchFrameGeneration(const FrameGenerationDispatchDescription* desc, const char* caller);
    ErrorCode F3ContextCreate(void* ctx, Fsr3ContextDescription* desc, const char* caller);
    ErrorCode F3ContextDispatchUpscale(void* ctx, const Fsr3DispatchUpscaleDescription* desc, const char* caller);
    ErrorCode F3SkipPresent(void* ctx, const char* caller);
    ErrorCode F3ContextGenerateReactiveMask(void* ctx, const UpscalerGenerateReactiveDescription* desc, const char* caller);
    ErrorCode F3ConfigureFrameGeneration(void* ctx, const FrameGenerationConfig* config, const char* caller);
    ErrorCode F3ContextDestroy(void* ctx, const char* caller);
    float F3GetUpscaleRatioFromQualityMode(int32_t mode, const char* caller);
    ErrorCode F3GetRenderResolutionFromQualityMode(uint32_t* w, uint32_t* h, uint32_t dw, uint32_t dh, int32_t mode, const char* caller);
    int32_t F3GetJitterPhaseCount(int32_t rw, int32_t dw, const char* caller);
    ErrorCode F3GetJitterOffset(float* x, float* y, int32_t index, int32_t count, const char* caller);
    bool F3ResourceIsNull(const void* resourceCopy, const char* caller);
    ErrorCode F3ContextGetGpuMemoryUsage(void* ctx, void* a, void* b, void* c, const char* caller);
    ErrorCode F3ContextDispatchFrameGenerationPrepare(void* ctx, const void* desc, const char* caller);
    uint32_t F3GetEffectVersion(const char* caller);

    // Upscaler-side dispatches seen on the calling thread (exported by WrongGPU's ffx_fsr3upscaler_x64.dll).
    static uint64_t SeenOnThisThread();
    // Upscaler side of Platform::hintUpscalingOnly (exported by WrongGPU's ffx_fsr3upscaler_x64.dll).
    static void HintUpscalingOnly(int hint);
    size_t ContextCount();  // for tests

    struct Context;

private:
    struct CallLog;
    bool ShouldLog(const char* function);
    std::string Begin(const char* function, const char* caller, uint64_t& call);
    void Log(const std::string& line);
    Context* Find(void* ctx);
    void Register(void* ctx, const UpscaleCreateParams& params, const char* kind);
    void Unregister(void* ctx);
    bool StandInActive() const;
    static std::string Label(const Context& c);
    // Runs the stand-in (or network) for one frame; false (logged) when it could not.
    // c.pending: takes the upscaler over when its create has finished; false while it runs or when it failed.
    bool TakePending(Context& c, const std::string& tag);
    bool RunStandIn(Context& c, const ffxDispatchDescUpscale& dispatch, const std::string& tag);
    // Platform::status for one dispatch of `c` (frame: the mapped frame of a WrongGPU-upscaled dispatch, else null).
    void Report(const Context& c, UpscaleState state, const WgFrame* frame, void* device);
    // The player compares (compare_key): `dispatch` on AMD's FidelityFX API upscaler (compareApi). False when there is
    // none (AMD's FSR 3.0 call compares instead).
    bool RunComparison(Context& c, ffxDispatchDescUpscale dispatch, const std::string& tag, bool log);

    Platform platform_;
    Config config_;
    std::string tag_;
    std::atomic<uint64_t> calls_{0};
    std::atomic<uint32_t> nextId_{1};
    std::mutex mutex_;
    std::map<void*, std::unique_ptr<Context>> contexts_;
    std::map<std::string, uint64_t> callCounts_;
    std::atomic<bool> thinning_{false};
};

}  // namespace fsr30
}  // namespace wgfsr
