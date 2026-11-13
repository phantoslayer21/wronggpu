// SPDX-License-Identifier: MIT
// ffx_fsr3upscaler_x64.dll replacement (spec 20, integration point 2): forwards every export to AMD's original,
// ffx_fsr3upscaler_x64.wronggpu-original.dll, logs each call with its parameters and caller, and lets the stand-in take
// over upscale dispatches (fsr30_hook.h). Exports: the FSR 3.0.x functions (SDK fsr3-v3.0.4 ffx_fsr3upscaler.h), the
// four SDK 1.1 additions, and wgFsr3UpscalerSeenOnThread / wgFsr3UpscalerHintUpscalingOnly for WrongGPU's
// ffx_fsr3_x64.dll.
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

const char* const wronggpu::kLogTag = "ffx_fsr3upscaler";

namespace {

HMODULE g_self = nullptr;
std::once_flag g_hookOnce;
Hook* g_hook = nullptr;
std::once_flag g_originalOnce;
UpscalerOriginal g_original;
bool g_originalLoaded = false;

const std::vector<std::string> kExports = {
    "ffxFsr3UpscalerContextCreate",
    "ffxFsr3UpscalerGetSharedResourceDescriptions",
    "ffxFsr3UpscalerContextDispatch",
    "ffxFsr3UpscalerContextGenerateReactiveMask",
    "ffxFsr3UpscalerContextDestroy",
    "ffxFsr3UpscalerGetUpscaleRatioFromQualityMode",
    "ffxFsr3UpscalerGetRenderResolutionFromQualityMode",
    "ffxFsr3UpscalerGetJitterPhaseCount",
    "ffxFsr3UpscalerGetJitterOffset",
    "ffxFsr3UpscalerResourceIsNull",
    "ffxFsr3UpscalerContextGetGpuMemoryUsage",
    "ffxFsr3UpscalerGetEffectVersion",
    "ffxFsr3UpscalerSetConstant",
    "ffxFsr3UpscalerSetGlobalDebugMessage"};

const UpscalerOriginal* LoadOriginal() {
    std::call_once(g_originalOnce, [] {
        HMODULE m = LoadOriginalNextToUs(L"ffx_fsr3upscaler_x64.wronggpu-original.dll", g_self, kExports, "");
        if (!m) return;
        g_original.contextCreate = ProcAddress<decltype(g_original.contextCreate)>(m, "ffxFsr3UpscalerContextCreate");
        g_original.getSharedResourceDescriptions = ProcAddress<decltype(g_original.getSharedResourceDescriptions)>(m, "ffxFsr3UpscalerGetSharedResourceDescriptions");
        g_original.contextDispatch = ProcAddress<decltype(g_original.contextDispatch)>(m, "ffxFsr3UpscalerContextDispatch");
        g_original.contextGenerateReactiveMask = ProcAddress<decltype(g_original.contextGenerateReactiveMask)>(m, "ffxFsr3UpscalerContextGenerateReactiveMask");
        g_original.contextDestroy = ProcAddress<decltype(g_original.contextDestroy)>(m, "ffxFsr3UpscalerContextDestroy");
        g_original.getUpscaleRatioFromQualityMode = ProcAddress<decltype(g_original.getUpscaleRatioFromQualityMode)>(m, "ffxFsr3UpscalerGetUpscaleRatioFromQualityMode");
        g_original.getRenderResolutionFromQualityMode = ProcAddress<decltype(g_original.getRenderResolutionFromQualityMode)>(m, "ffxFsr3UpscalerGetRenderResolutionFromQualityMode");
        g_original.getJitterPhaseCount = ProcAddress<decltype(g_original.getJitterPhaseCount)>(m, "ffxFsr3UpscalerGetJitterPhaseCount");
        g_original.getJitterOffset = ProcAddress<decltype(g_original.getJitterOffset)>(m, "ffxFsr3UpscalerGetJitterOffset");
        g_original.resourceIsNull = ProcAddress<decltype(g_original.resourceIsNull)>(m, "ffxFsr3UpscalerResourceIsNull");
        g_original.contextGetGpuMemoryUsage = ProcAddress<decltype(g_original.contextGetGpuMemoryUsage)>(m, "ffxFsr3UpscalerContextGetGpuMemoryUsage");
        g_original.getEffectVersion = ProcAddress<decltype(g_original.getEffectVersion)>(m, "ffxFsr3UpscalerGetEffectVersion");
        g_original.setConstant = ProcAddress<decltype(g_original.setConstant)>(m, "ffxFsr3UpscalerSetConstant");
        g_original.setGlobalDebugMessage = ProcAddress<decltype(g_original.setGlobalDebugMessage)>(m, "ffxFsr3UpscalerSetGlobalDebugMessage");
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
        p.upscaler = &LoadOriginal;
        p.backend = &ResolveBackend;
        p.log = [](const std::string& line) { wronggpu::Log::Get().WriteLine(line); };
        p.flushLog = [] { wronggpu::Log::Get().Flush(); };
        p.logBytes = [] { return wronggpu::Log::Get().Bytes(); };
        p.resourceInfo = &D3D12ResourceInfo;
        p.deviceFromCommandList = &DeviceFromCommandList;
        AddSplitServices(p, config);
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

__declspec(dllexport) ErrorCode ffxFsr3UpscalerContextCreate(void* context, const UpscalerContextDescription* desc) {
    const char* caller = ModuleNameOf(WG_RETURN_ADDRESS());
    return Guard<ErrorCode>("ffxFsr3UpscalerContextCreate", ERROR_INVALID_POINTER, [&] { return H().UpContextCreate(context, desc, caller); });
}

__declspec(dllexport) ErrorCode ffxFsr3UpscalerGetSharedResourceDescriptions(void* context, void* out) {
    const char* caller = ModuleNameOf(WG_RETURN_ADDRESS());
    return Guard<ErrorCode>("ffxFsr3UpscalerGetSharedResourceDescriptions", ERROR_INVALID_POINTER, [&] { return H().UpGetSharedResourceDescriptions(context, out, caller); });
}

__declspec(dllexport) ErrorCode ffxFsr3UpscalerContextDispatch(void* context, const UpscalerDispatchDescription* desc) {
    const char* caller = ModuleNameOf(WG_RETURN_ADDRESS());
    return Guard<ErrorCode>("ffxFsr3UpscalerContextDispatch", ERROR_INVALID_POINTER, [&] { return H().UpContextDispatch(context, desc, caller); });
}

__declspec(dllexport) ErrorCode ffxFsr3UpscalerContextGenerateReactiveMask(void* context, const UpscalerGenerateReactiveDescription* desc) {
    const char* caller = ModuleNameOf(WG_RETURN_ADDRESS());
    return Guard<ErrorCode>("ffxFsr3UpscalerContextGenerateReactiveMask", ERROR_INVALID_POINTER, [&] { return H().UpContextGenerateReactiveMask(context, desc, caller); });
}

__declspec(dllexport) ErrorCode ffxFsr3UpscalerContextDestroy(void* context) {
    const char* caller = ModuleNameOf(WG_RETURN_ADDRESS());
    return Guard<ErrorCode>("ffxFsr3UpscalerContextDestroy", ERROR_INVALID_POINTER, [&] { return H().UpContextDestroy(context, caller); });
}

__declspec(dllexport) float ffxFsr3UpscalerGetUpscaleRatioFromQualityMode(int32_t mode) {
    const char* caller = ModuleNameOf(WG_RETURN_ADDRESS());
    return Guard<float>("ffxFsr3UpscalerGetUpscaleRatioFromQualityMode", 0.0f, [&] { return H().UpGetUpscaleRatioFromQualityMode(mode, caller); });
}

__declspec(dllexport) ErrorCode ffxFsr3UpscalerGetRenderResolutionFromQualityMode(uint32_t* w, uint32_t* h, uint32_t dw, uint32_t dh, int32_t mode) {
    const char* caller = ModuleNameOf(WG_RETURN_ADDRESS());
    return Guard<ErrorCode>("ffxFsr3UpscalerGetRenderResolutionFromQualityMode", ERROR_INVALID_POINTER, [&] { return H().UpGetRenderResolutionFromQualityMode(w, h, dw, dh, mode, caller); });
}

__declspec(dllexport) int32_t ffxFsr3UpscalerGetJitterPhaseCount(int32_t rw, int32_t dw) {
    const char* caller = ModuleNameOf(WG_RETURN_ADDRESS());
    return Guard<int32_t>("ffxFsr3UpscalerGetJitterPhaseCount", 1, [&] { return H().UpGetJitterPhaseCount(rw, dw, caller); });
}

__declspec(dllexport) ErrorCode ffxFsr3UpscalerGetJitterOffset(float* x, float* y, int32_t index, int32_t count) {
    const char* caller = ModuleNameOf(WG_RETURN_ADDRESS());
    return Guard<ErrorCode>("ffxFsr3UpscalerGetJitterOffset", ERROR_INVALID_POINTER, [&] { return H().UpGetJitterOffset(x, y, index, count, caller); });
}

__declspec(dllexport) bool ffxFsr3UpscalerResourceIsNull(const void* resourceCopy) {
    const char* caller = ModuleNameOf(WG_RETURN_ADDRESS());
    return Guard<bool>("ffxFsr3UpscalerResourceIsNull", true, [&] { return H().UpResourceIsNull(resourceCopy, caller); });
}

__declspec(dllexport) ErrorCode ffxFsr3UpscalerContextGetGpuMemoryUsage(void* context, void* usage) {
    const char* caller = ModuleNameOf(WG_RETURN_ADDRESS());
    return Guard<ErrorCode>("ffxFsr3UpscalerContextGetGpuMemoryUsage", ERROR_INVALID_POINTER, [&] { return H().UpContextGetGpuMemoryUsage(context, usage, caller); });
}

__declspec(dllexport) uint32_t ffxFsr3UpscalerGetEffectVersion() {
    const char* caller = ModuleNameOf(WG_RETURN_ADDRESS());
    return Guard<uint32_t>("ffxFsr3UpscalerGetEffectVersion", 0, [&] { return H().UpGetEffectVersion(caller); });
}

__declspec(dllexport) ErrorCode ffxFsr3UpscalerSetConstant(void* context, int32_t key, void* value) {
    const char* caller = ModuleNameOf(WG_RETURN_ADDRESS());
    return Guard<ErrorCode>("ffxFsr3UpscalerSetConstant", ERROR_INVALID_POINTER, [&] { return H().UpSetConstant(context, key, value, caller); });
}

__declspec(dllexport) ErrorCode ffxFsr3UpscalerSetGlobalDebugMessage(void* fn, uint32_t level) {
    const char* caller = ModuleNameOf(WG_RETURN_ADDRESS());
    return Guard<ErrorCode>("ffxFsr3UpscalerSetGlobalDebugMessage", ERROR_INVALID_POINTER, [&] { return H().UpSetGlobalDebugMessage(fn, level, caller); });
}

__declspec(dllexport) uint64_t wgFsr3UpscalerSeenOnThread() { return Hook::SeenOnThisThread(); }

__declspec(dllexport) void wgFsr3UpscalerHintUpscalingOnly(int hint) { Hook::HintUpscalingOnly(hint); }

}  // extern "C"

BOOL WINAPI DllMain(HINSTANCE instance, DWORD reason, LPVOID reserved) {
    if (reason == DLL_PROCESS_ATTACH) {
        g_self = instance;
        DisableThreadLibraryCalls(instance);
        wronggpu::Log::Get().Write("WrongGPU FSR 3.0 upscaler forwarder %s (%s) loaded as %s", WRONGGPU_FSR_VERSION, WRONGGPU_FSR_BUILD,
                                   Narrow(wronggpu::ModuleDirectory()).c_str());
        wgwin::PrepareHipEnvironment();
    } else if (reason == DLL_PROCESS_DETACH) {
        if (!reserved) wronggpu::Log::Get().Write("WrongGPU FSR 3.0 upscaler forwarder unloaded");
    }
    return TRUE;
}
