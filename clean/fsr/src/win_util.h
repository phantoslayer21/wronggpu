// SPDX-License-Identifier: MIT
#pragma once
// Windows helpers shared by WrongGPU's FidelityFX DLLs (amd_fidelityfx_dx12.dll, ffx_fsr3_x64.dll,
// ffx_fsr3upscaler_x64.dll): loading AMD's renamed originals, settings, upscaler choice, D3D12 queries.
#include "log_win.h"

#include <string>
#include <vector>

#include "../include/wronggpu_upscaler.h"
#include "amd_compare.h"
#include "amd_provider.h"
#include "config.h"
#include "d3d12_hooks.h"
#include "overlay.h"
#include "param_map.h"

namespace wgwin {

template <class T>
T ProcAddress(HMODULE module, const char* name) {
    return reinterpret_cast<T>(reinterpret_cast<void*>(GetProcAddress(module, name)));
}

std::string Narrow(const std::wstring& w);
std::wstring Widen(const std::string& s);

// Names exported by a loaded module (from its PE export directory).
std::vector<std::string> ExportNames(HMODULE module);
// File version from the VERSIONINFO resource, or "unknown".
std::string FileVersion(const std::wstring& path);
// Product name and description from the VERSIONINFO resource (empty when absent).
std::string FileDescription(const std::wstring& path);

// Loads <this DLL's folder>\<fileName>; logs where it came from, its version and the exports it has beyond `known`.
// Refuses to load a copy of the calling module. Null (logged) when missing.
HMODULE LoadOriginalNextToUs(const wchar_t* fileName, HMODULE self, const std::vector<std::string>& known, const std::string& tag);

// Makes the game's process safe for the HIP runtime of AMD's driver, which the network runtime uses. Called by WrongGPU's
// DLLs when they load, before anything loads HIP:
//  - clears the ROCm SDK variables LLVM_PATH, ROCM_PATH, HIP_PATH, HIP_DEVICE_LIB_PATH. The driver's HIP does not need
//    them and games do not use them, but pointing at an SDK or LLVM of another version they make HIP's start-up compile
//    of its own copy kernels fail (LLVM_PATH: the first stream never starts);
//  - turns AMD's code-compiler cache off (AMD_COMGR_CACHE=0). A failed compile stays in the player's cache
//    (%LOCALAPPDATA%\comgr) and fails every later start too; without the cache HIP compiles its few kernels again at
//    every start (well under a second, inside the upscaler's create);
//  - turns HIP's resource cache off (GPU_RESOURCE_CACHE_SIZE=0, unless set already; 0.5.1). The cache keeps the device
//    memory hipFree gives back, for the rest of the process: every destroyed upscaler's memory stayed the process's,
//    up to the most that was ever in use at once (a host switching upscalers creates the new one before it destroys
//    the old one: Model M + Model K + the arenas, about 2.8 GB at 1440p, measured with test_dll_d3d12 switch_real and
//    test_interop_d3d12 hipleak, DXGI's process usage). The upscaler allocates at create, not per frame.
// When HIP's copy kernels fail, HIP's worker thread crashes (Borderlands 4: the game froze in its crash handler).
// Logs what it changed.
void PrepareHipEnvironment();
void DiagnosticProvider(const wgfsr::UpscaleStatus& status);
void DiagnosticFailure(uint64_t context, uint64_t frame, void* device);

// Settings (wronggpu.ini next to the DLL; written with defaults when absent). Logs warnings and the settings.
wgfsr::Config ReadSettings(const std::string& tag);

// Upscaler implementation for the settings: the stand-in, the network runtime DLL, or none (upscaler = amd).
const WgUpscalerBackend* ResolveBackend(const wgfsr::Config& config, std::string& why);

// The stand-in without split_test, for single frames a backend is not ready for (WG_ERROR_NOT_READY).
const WgUpscalerBackend* StandInForFallback(const wgfsr::Config& config);

// AMD's FidelityFX API DLL that gives a hosted WrongGPU AMD's upscaler versions and runs them (and the player's
// comparisons), chosen once per process: the candidates of amd_provider.h in order (compare_dll; the installer's
// private copy wronggpu-amd-compare.dll; AMD's FSR 4 DLL next to WrongGPU; the FSR 4 DLL the installer found on this
// PC; the renamed original; the game's own FidelityFX SDK and amd_fidelityfx_dx12.dll), each loaded by its full path
// and checked (CheckProviderModule). Logs the choice, or "FSR 4 / AMD versions not available: ..." when none is usable
// (null then).
const wgfsr::OriginalApi* CompareApi(const std::string& path);
// Whether `m`, got from LoadLibrary of `requested`, is that AMD FidelityFX API DLL: not WrongGPU's own module or a copy
// of it, the file that was asked for (a host's LoadLibrary hook can hand out OptiScaler's or WrongGPU's module for an
// AMD file name), the FidelityFX API exports, none of which jumps into a module that is not AMD's (a host's hook).
// detail: why not, for the log.
wgfsr::ProbeResult CheckProviderModule(HMODULE m, const std::wstring& requested, std::string& detail);

// Interface version 2 services of a front end's platform (FrontEnd or fsr30::Hook): the split service, the dispatch
// notes for the submission trace, the one-frame stand-in, the overlay's status. Also passes this DLL's settings to
// the D3D12 hooks and the overlay.
template <class PlatformT>
void AddSplitServices(PlatformT& p, const wgfsr::Config& config) {
    try {
    wronggpu::Log::Get().Event("effective_settings", "{\"hosted\":" + std::string(config.hosted ? "true" : "false") +
        ",\"runtime_dll\":" + wgdiag::Quote(config.runtimeDll) + ",\"backend_options\":" + wgdiag::Quote(config.BackendOptions()) +
        ",\"upscaler_choice\":" + std::to_string(int(config.upscaler)) + ",\"fallback_to_amd\":" + (config.fallbackToAmd ? "true" : "false") +
        ",\"create_wait_ms\":" + std::to_string(config.createWaitMs) + ",\"split_timeout_ms\":" + std::to_string(config.splitTimeoutMs) +
        ",\"d3d12_hooks\":" + wgdiag::Quote(config.d3d12Hooks) + ",\"overlay\":" + wgdiag::Quote(config.overlay) +
        ",\"compare_dll\":" + wgdiag::Quote(config.compareDll) + "}");
    } catch (...) {}
    ConfigureHooks(config);
    ConfigureOverlay(config);
    p.status = [](const wgfsr::UpscaleStatus& status) { DiagnosticProvider(status); OverlayStatus(status); };
    p.diagnosticFrame = [](uint64_t context, const WgFrame& frame) { wronggpu::Log::Get().RecordFrame(context, frame); };
    p.diagnosticFailure = &DiagnosticFailure;
    p.diagnosticResult = [](uint64_t context, uint64_t frame, WgResult result) {
        try {
        if (result != WG_OK || frame < 8 || frame % 300 == 0)
            wronggpu::Log::Get().Event("backend_return", "{\"context\":" + std::to_string(context) +
                ",\"frame\":" + std::to_string(frame) + ",\"wg_result\":" + std::to_string(int(result)) +
                ",\"gpu_completion_proven\":false}");
        } catch (...) {}
    };
    p.gpuTimer = &OverlayGpuTimer;
    p.compareAmd = &OverlayCompareAmd;
    p.compareApi = [dll = config.compareDll] { return CompareApi(dll); };
    p.services = &SplitServices;
    p.noteDispatch = &NoteDispatch;
    p.standIn = &StandInForFallback;
}

bool D3D12ResourceInfo(void* resource, wgfsr::ResourceInfo* out);
void* DeviceFromCommandList(void* commandList);  // ID3D12Device* (the game keeps it alive), or null

// "name.dll" of the module that contains `address` (cached per thread for repeated callers).
const char* ModuleNameOf(const void* address);

}  // namespace wgwin

#if defined(_MSC_VER)
#include <intrin.h>
#define WG_RETURN_ADDRESS() _ReturnAddress()
#else
#define WG_RETURN_ADDRESS() __builtin_return_address(0)
#endif
