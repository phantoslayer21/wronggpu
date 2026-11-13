// SPDX-License-Identifier: MIT
// ffx_fsr3_x64.dll replacement (spec 20, integration point 2): forwards every export to AMD's original,
// ffx_fsr3_x64.wronggpu-original.dll, logs each call with its parameters and caller, and (when WrongGPU's
// ffx_fsr3upscaler_x64.dll did not see the upscale) lets the stand-in overwrite the upscale output (fsr30_hook.h).
// Exports: the FSR 3.0.x functions (SDK fsr3-v3.0.4 ffx_fsr3.h) and the three SDK 1.1 additions.
#include <atomic>
#include <exception>
#include <mutex>
#include <string>
#include <vector>

#include "fsr30_hook.h"
#include "version.h"
#include "win_util.h"

using namespace wgfsr;
using namespace wgfsr::fsr30;
using namespace wgwin;

const char* const wronggpu::kLogTag = "ffx_fsr3";

namespace {

HMODULE g_self = nullptr;
std::once_flag g_hookOnce;
Hook* g_hook = nullptr;
std::once_flag g_originalOnce;
Fsr3Original g_original;
bool g_originalLoaded = false;

const std::vector<std::string> kExports = {
    "ffxFsr3DispatchFrameGeneration",
    "ffxFsr3ContextCreate",
    "ffxFsr3ContextDispatchUpscale",
    "ffxFsr3SkipPresent",
    "ffxFsr3ContextGenerateReactiveMask",
    "ffxFsr3ConfigureFrameGeneration",
    "ffxFsr3ContextDestroy",
    "ffxFsr3GetUpscaleRatioFromQualityMode",
    "ffxFsr3GetRenderResolutionFromQualityMode",
    "ffxFsr3GetJitterPhaseCount",
    "ffxFsr3GetJitterOffset",
    "ffxFsr3ResourceIsNull",
    "ffxFsr3ContextGetGpuMemoryUsage",
    "ffxFsr3ContextDispatchFrameGenerationPrepare",
    "ffxFsr3GetEffectVersion"};

const Fsr3Original* LoadOriginal() {
    std::call_once(g_originalOnce, [] {
        HMODULE m = LoadOriginalNextToUs(L"ffx_fsr3_x64.wronggpu-original.dll", g_self, kExports, "");
        if (!m) return;
        g_original.dispatchFrameGeneration = ProcAddress<decltype(g_original.dispatchFrameGeneration)>(m, "ffxFsr3DispatchFrameGeneration");
        g_original.contextCreate = ProcAddress<decltype(g_original.contextCreate)>(m, "ffxFsr3ContextCreate");
        g_original.contextDispatchUpscale = ProcAddress<decltype(g_original.contextDispatchUpscale)>(m, "ffxFsr3ContextDispatchUpscale");
        g_original.skipPresent = ProcAddress<decltype(g_original.skipPresent)>(m, "ffxFsr3SkipPresent");
        g_original.contextGenerateReactiveMask = ProcAddress<decltype(g_original.contextGenerateReactiveMask)>(m, "ffxFsr3ContextGenerateReactiveMask");
        g_original.configureFrameGeneration = ProcAddress<decltype(g_original.configureFrameGeneration)>(m, "ffxFsr3ConfigureFrameGeneration");
        g_original.contextDestroy = ProcAddress<decltype(g_original.contextDestroy)>(m, "ffxFsr3ContextDestroy");
        g_original.getUpscaleRatioFromQualityMode = ProcAddress<decltype(g_original.getUpscaleRatioFromQualityMode)>(m, "ffxFsr3GetUpscaleRatioFromQualityMode");
        g_original.getRenderResolutionFromQualityMode = ProcAddress<decltype(g_original.getRenderResolutionFromQualityMode)>(m, "ffxFsr3GetRenderResolutionFromQualityMode");
        g_original.getJitterPhaseCount = ProcAddress<decltype(g_original.getJitterPhaseCount)>(m, "ffxFsr3GetJitterPhaseCount");
        g_original.getJitterOffset = ProcAddress<decltype(g_original.getJitterOffset)>(m, "ffxFsr3GetJitterOffset");
        g_original.resourceIsNull = ProcAddress<decltype(g_original.resourceIsNull)>(m, "ffxFsr3ResourceIsNull");
        g_original.contextGetGpuMemoryUsage = ProcAddress<decltype(g_original.contextGetGpuMemoryUsage)>(m, "ffxFsr3ContextGetGpuMemoryUsage");
        g_original.contextDispatchFrameGenerationPrepare = ProcAddress<decltype(g_original.contextDispatchFrameGenerationPrepare)>(m, "ffxFsr3ContextDispatchFrameGenerationPrepare");
        g_original.getEffectVersion = ProcAddress<decltype(g_original.getEffectVersion)>(m, "ffxFsr3GetEffectVersion");
        g_original.sdk11 = g_original.getEffectVersion != nullptr;
        g_originalLoaded = true;
        wronggpu::Log::Get().WriteLine(g_original.sdk11
            ? "AMD's DLL is an SDK 1.1 (FSR 3.1) build: its structures differ from FSR 3.0.x, so calls are forwarded and logged without decoding, and the stand-in stays off"
            : "AMD's DLL is an FSR 3.0.x build (SDK fsr3-v3.0.3/3.0.4 layout): calls are decoded");
    });
    return g_originalLoaded ? &g_original : nullptr;
}

Hook& H() {
    std::call_once(g_hookOnce, [] {
        Config config = ReadSettings("");
        Platform p;
        p.fsr3 = &LoadOriginal;
        p.backend = &ResolveBackend;
        p.log = [](const std::string& line) { wronggpu::Log::Get().WriteLine(line); };
        p.flushLog = [] { wronggpu::Log::Get().Flush(); };
        p.logBytes = [] { return wronggpu::Log::Get().Bytes(); };
        p.resourceInfo = &D3D12ResourceInfo;
        p.deviceFromCommandList = &DeviceFromCommandList;
        AddSplitServices(p, config);
        // How many upscaler dispatches WrongGPU's ffx_fsr3upscaler_x64.dll has seen on this thread (0 if not ours).
        p.upscalerSeenOnThread = [] {
            using SeenFn = uint64_t (*)();
            static std::atomic<SeenFn> fn{nullptr};
            SeenFn f = fn.load();
            if (!f) {
                if (HMODULE up = GetModuleHandleW(L"ffx_fsr3upscaler_x64.dll")) f = ProcAddress<SeenFn>(up, "wgFsr3UpscalerSeenOnThread");
                if (f) fn.store(f);
            }
            return f ? f() : uint64_t(0);
        };
        // Whether the upscaler context AMD's ffx_fsr3 creates next (on this thread) belongs to an UPSCALING_ONLY context.
        p.hintUpscalingOnly = [](int hint) {
            using HintFn = void (*)(int);
            HMODULE up = GetModuleHandleW(L"ffx_fsr3upscaler_x64.dll");
            HintFn f = up ? ProcAddress<HintFn>(up, "wgFsr3UpscalerHintUpscalingOnly") : nullptr;
            if (f) f(hint);
        };
        g_hook = new Hook(p, config, "");
        LoadOriginal();
    });
    return *g_hook;
}

template <class R, class F>
R Guard(const char* name, R fail, F&& f) {
    try {
        return f();
    } catch (const std::exception& e) {
        wronggpu::Log::Get().Write("ERROR: exception in %s: %s", name, e.what());
    } catch (...) {
        wronggpu::Log::Get().Write("ERROR: unknown exception in %s", name);
    }
    return fail;
}

}  // namespace

extern "C" {

__declspec(dllexport) ErrorCode ffxFsr3DispatchFrameGeneration(const FrameGenerationDispatchDescription* desc) {
    const char* caller = ModuleNameOf(WG_RETURN_ADDRESS());
    return Guard<ErrorCode>("ffxFsr3DispatchFrameGeneration", ERROR_INVALID_POINTER, [&] { return H().F3DispatchFrameGeneration(desc, caller); });
}

__declspec(dllexport) ErrorCode ffxFsr3ContextCreate(void* context, Fsr3ContextDescription* desc) {
    const char* caller = ModuleNameOf(WG_RETURN_ADDRESS());
    return Guard<ErrorCode>("ffxFsr3ContextCreate", ERROR_INVALID_POINTER, [&] { return H().F3ContextCreate(context, desc, caller); });
}

__declspec(dllexport) ErrorCode ffxFsr3ContextDispatchUpscale(void* context, const Fsr3DispatchUpscaleDescription* desc) {
    const char* caller = ModuleNameOf(WG_RETURN_ADDRESS());
    return Guard<ErrorCode>("ffxFsr3ContextDispatchUpscale", ERROR_INVALID_POINTER, [&] { return H().F3ContextDispatchUpscale(context, desc, caller); });
}

__declspec(dllexport) ErrorCode ffxFsr3SkipPresent(void* context) {
    const char* caller = ModuleNameOf(WG_RETURN_ADDRESS());
    return Guard<ErrorCode>("ffxFsr3SkipPresent", ERROR_INVALID_POINTER, [&] { return H().F3SkipPresent(context, caller); });
}

__declspec(dllexport) ErrorCode ffxFsr3ContextGenerateReactiveMask(void* context, const UpscalerGenerateReactiveDescription* desc) {
    const char* caller = ModuleNameOf(WG_RETURN_ADDRESS());
    return Guard<ErrorCode>("ffxFsr3ContextGenerateReactiveMask", ERROR_INVALID_POINTER, [&] { return H().F3ContextGenerateReactiveMask(context, desc, caller); });
}

__declspec(dllexport) ErrorCode ffxFsr3ConfigureFrameGeneration(void* context, const FrameGenerationConfig* config) {
    const char* caller = ModuleNameOf(WG_RETURN_ADDRESS());
    return Guard<ErrorCode>("ffxFsr3ConfigureFrameGeneration", ERROR_INVALID_POINTER, [&] { return H().F3ConfigureFrameGeneration(context, config, caller); });
}

__declspec(dllexport) ErrorCode ffxFsr3ContextDestroy(void* context) {
    const char* caller = ModuleNameOf(WG_RETURN_ADDRESS());
    return Guard<ErrorCode>("ffxFsr3ContextDestroy", ERROR_INVALID_POINTER, [&] { return H().F3ContextDestroy(context, caller); });
}

__declspec(dllexport) float ffxFsr3GetUpscaleRatioFromQualityMode(int32_t mode) {
    const char* caller = ModuleNameOf(WG_RETURN_ADDRESS());
    return Guard<float>("ffxFsr3GetUpscaleRatioFromQualityMode", 0.0f, [&] { return H().F3GetUpscaleRatioFromQualityMode(mode, caller); });
}

__declspec(dllexport) ErrorCode ffxFsr3GetRenderResolutionFromQualityMode(uint32_t* w, uint32_t* h, uint32_t dw, uint32_t dh, int32_t mode) {
    const char* caller = ModuleNameOf(WG_RETURN_ADDRESS());
    return Guard<ErrorCode>("ffxFsr3GetRenderResolutionFromQualityMode", ERROR_INVALID_POINTER, [&] { return H().F3GetRenderResolutionFromQualityMode(w, h, dw, dh, mode, caller); });
}

__declspec(dllexport) int32_t ffxFsr3GetJitterPhaseCount(int32_t rw, int32_t dw) {
    const char* caller = ModuleNameOf(WG_RETURN_ADDRESS());
    return Guard<int32_t>("ffxFsr3GetJitterPhaseCount", 1, [&] { return H().F3GetJitterPhaseCount(rw, dw, caller); });
}

__declspec(dllexport) ErrorCode ffxFsr3GetJitterOffset(float* x, float* y, int32_t index, int32_t count) {
    const char* caller = ModuleNameOf(WG_RETURN_ADDRESS());
    return Guard<ErrorCode>("ffxFsr3GetJitterOffset", ERROR_INVALID_POINTER, [&] { return H().F3GetJitterOffset(x, y, index, count, caller); });
}

__declspec(dllexport) bool ffxFsr3ResourceIsNull(const void* resourceCopy) {
    const char* caller = ModuleNameOf(WG_RETURN_ADDRESS());
    return Guard<bool>("ffxFsr3ResourceIsNull", true, [&] { return H().F3ResourceIsNull(resourceCopy, caller); });
}

__declspec(dllexport) ErrorCode ffxFsr3ContextGetGpuMemoryUsage(void* context, void* upscaler, void* opticalFlow, void* frameGeneration) {
    const char* caller = ModuleNameOf(WG_RETURN_ADDRESS());
    return Guard<ErrorCode>("ffxFsr3ContextGetGpuMemoryUsage", ERROR_INVALID_POINTER, [&] { return H().F3ContextGetGpuMemoryUsage(context, upscaler, opticalFlow, frameGeneration, caller); });
}

__declspec(dllexport) ErrorCode ffxFsr3ContextDispatchFrameGenerationPrepare(void* context, const void* desc) {
    const char* caller = ModuleNameOf(WG_RETURN_ADDRESS());
    return Guard<ErrorCode>("ffxFsr3ContextDispatchFrameGenerationPrepare", ERROR_INVALID_POINTER, [&] { return H().F3ContextDispatchFrameGenerationPrepare(context, desc, caller); });
}

__declspec(dllexport) uint32_t ffxFsr3GetEffectVersion() {
    const char* caller = ModuleNameOf(WG_RETURN_ADDRESS());
    return Guard<uint32_t>("ffxFsr3GetEffectVersion", 0, [&] { return H().F3GetEffectVersion(caller); });
}

}  // extern "C"

BOOL WINAPI DllMain(HINSTANCE instance, DWORD reason, LPVOID reserved) {
    if (reason == DLL_PROCESS_ATTACH) {
        g_self = instance;
        DisableThreadLibraryCalls(instance);
        wronggpu::Log::Get().Write("WrongGPU FSR 3.0 forwarder %s (%s) loaded as %s", WRONGGPU_FSR_VERSION, WRONGGPU_FSR_BUILD,
                                   Narrow(wronggpu::ModuleDirectory()).c_str());
        wgwin::PrepareHipEnvironment();
    } else if (reason == DLL_PROCESS_DETACH) {
        if (!reserved) wronggpu::Log::Get().Write("WrongGPU FSR 3.0 forwarder unloaded");
    }
    return TRUE;
}
