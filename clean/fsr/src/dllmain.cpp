// SPDX-License-Identifier: GPL-3.0-only
// amd_fidelityfx_dx12.dll replacement (spec 20, integration point 1): the five public FidelityFX API exports,
// AMD's original DLL loaded from amd_fidelityfx_dx12.wronggpu-original.dll next to this one (spec 30), the
// upscaler implementation chosen by wronggpu.ini, and the call log wronggpu-fsr.log.
#include "log_win.h"

#include <d3d12.h>
#include <winver.h>

#include <cstddef>
#include <exception>
#include <mutex>
#include <string>
#include <vector>

#include "../include/wronggpu_host_event.h"
#include "ffx_headers.h"
#include "frontend.h"
#include "version.h"
#include "win_util.h"
#include "d3d12_hooks.h"
#include "asi_bridge.h"
#include "hosted_badge_win.h"
#include "settings_bridge.h"

using namespace wgfsr;
using namespace wgwin;

const char* const wronggpu::kLogTag = "amd_fidelityfx_dx12";

// The front end reads these values without the D3D12 headers; check them against the Windows SDK. (The checks against
// AMD's ffx_api_dx12.h are in ffx_api_layout_check.cpp, which is compiled but linked into no DLL.)
static_assert(dxgi::R16G16B16A16_FLOAT == DXGI_FORMAT_R16G16B16A16_FLOAT && dxgi::R11G11B10_FLOAT == DXGI_FORMAT_R11G11B10_FLOAT &&
                  dxgi::R10G10B10A2_UNORM == DXGI_FORMAT_R10G10B10A2_UNORM && dxgi::R8G8B8A8_UNORM_SRGB == DXGI_FORMAT_R8G8B8A8_UNORM_SRGB &&
                  dxgi::B8G8R8A8_UNORM_SRGB == DXGI_FORMAT_B8G8R8A8_UNORM_SRGB && dxgi::R32_FLOAT == DXGI_FORMAT_R32_FLOAT &&
                  dxgi::D32_FLOAT == DXGI_FORMAT_D32_FLOAT && dxgi::R24_UNORM_X8_TYPELESS == DXGI_FORMAT_R24_UNORM_X8_TYPELESS &&
                  dxgi::R32_FLOAT_X8X24_TYPELESS == DXGI_FORMAT_R32_FLOAT_X8X24_TYPELESS && dxgi::R16G16_FLOAT == DXGI_FORMAT_R16G16_FLOAT &&
                  dxgi::R16_UNORM == DXGI_FORMAT_R16_UNORM && dxgi::B8G8R8A8_TYPELESS == DXGI_FORMAT_B8G8R8A8_TYPELESS &&
                  dxgi::R9G9B9E5_SHAREDEXP == DXGI_FORMAT_R9G9B9E5_SHAREDEXP && dxgi::R16G16_SINT == DXGI_FORMAT_R16G16_SINT &&
                  dxgi::R8G8_UINT == DXGI_FORMAT_R8G8_UINT && dxgi::R32G32B32A32_TYPELESS == DXGI_FORMAT_R32G32B32A32_TYPELESS,
              "DXGI_FORMAT values");
static_assert(d3d12state::NON_PIXEL_SHADER_RESOURCE == D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE &&
                  d3d12state::PIXEL_SHADER_RESOURCE == D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE &&
                  d3d12state::UNORDERED_ACCESS == D3D12_RESOURCE_STATE_UNORDERED_ACCESS &&
                  d3d12state::RENDER_TARGET == D3D12_RESOURCE_STATE_RENDER_TARGET &&
                  d3d12state::COPY_SOURCE == D3D12_RESOURCE_STATE_COPY_SOURCE && d3d12state::COPY_DEST == D3D12_RESOURCE_STATE_COPY_DEST &&
                  d3d12state::INDIRECT_ARGUMENT == D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT &&
                  d3d12state::DEPTH_READ == D3D12_RESOURCE_STATE_DEPTH_READ && d3d12state::PRESENT == D3D12_RESOURCE_STATE_PRESENT,
              "D3D12_RESOURCE_STATES values");

namespace {

HMODULE g_self = nullptr;
std::once_flag g_initOnce;
FrontEnd* g_frontEnd = nullptr;
bool g_hosted = false;  // loaded under another name by an upscaler mod (OptiScaler) as its FidelityFX upscaler
// wronggpu.ini as it is on disk now (FrontEnd::Platform::readWronggpuIni). A named class, not a lambda that captures the
// path by copy: with MSVC's /Zc:lambda (on with /permissive-, as in CI) a lambda whose capture copy throws terminates the
// process instead of unwinding; the same copy in a class constructor unwinds normally (async_create.h DeviceRelease).
struct WronggpuIniReader {
    std::wstring path;
    explicit WronggpuIniReader(const std::wstring& p) : path(p) {}
    std::string operator()() const {
        std::string text;
        wgstate::ReadBytes(path, &text);
        return text;
    }
};

std::once_flag g_originalOnce;
OriginalApi g_original;
bool g_originalOk = false;

constexpr wchar_t kOriginalName[] = L"amd_fidelityfx_dx12.wronggpu-original.dll";

wronggpu::Log& L() { return wronggpu::Log::Get(); }

const OriginalApi* LoadOriginal() {
    std::call_once(g_originalOnce, [] {
        const std::wstring path = wronggpu::ModuleDirectory() + L"\\" + kOriginalName;
        HMODULE m = LoadLibraryExW(path.c_str(), nullptr, LOAD_WITH_ALTERED_SEARCH_PATH);
        if (!m) {
            if (g_hosted)  // the host loads AMD's other FidelityFX effects itself; WrongGPU only upscales for it
                L().Write("hosted: no %s next to this DLL (not needed: the host provides AMD's other FidelityFX effects)",
                          Narrow(path).c_str());
            else
                L().Write("AMD's original DLL not loaded: %s (error %lu). Only WrongGPU's own upscaler contexts can work;"
                          " frame generation and every other FidelityFX effect will fail.",
                          Narrow(path).c_str(), GetLastError());
            return;
        }
        if (m == g_self || FileDescription(path).find("WrongGPU") != std::string::npos) {
            L().Write("ERROR: %s is a copy of WrongGPU's own DLL, not AMD's; it is not used", Narrow(path).c_str());
            FreeLibrary(m);
            return;
        }
        g_original.createContext = ProcAddress<PfnFfxCreateContext>(m, "ffxCreateContext");
        g_original.destroyContext = ProcAddress<PfnFfxDestroyContext>(m, "ffxDestroyContext");
        g_original.configure = ProcAddress<PfnFfxConfigure>(m, "ffxConfigure");
        g_original.query = ProcAddress<PfnFfxQuery>(m, "ffxQuery");
        g_original.dispatch = ProcAddress<PfnFfxDispatch>(m, "ffxDispatch");
        g_originalOk = g_original.createContext && g_original.destroyContext && g_original.configure && g_original.query &&
                       g_original.dispatch;
        L().Write("AMD's original DLL loaded: %s, file version %s, %s", Narrow(path).c_str(), FileVersion(path).c_str(),
                  g_originalOk ? "all five FidelityFX exports found" : "ERROR: some FidelityFX exports are missing");
        std::string extra;
        for (const std::string& name : ExportNames(m))
            if (name != "ffxCreateContext" && name != "ffxDestroyContext" && name != "ffxConfigure" && name != "ffxQuery" &&
                name != "ffxDispatch")
                extra += " " + name;
        if (!extra.empty()) L().Write("note: AMD's DLL has exports WrongGPU does not provide:%s", extra.c_str());
    });
    return g_originalOk ? &g_original : nullptr;
}

// Platform::timer (parked upscalers, frontend.h): a thread that holds a reference on this DLL while it waits and
// runs, so that a FreeLibrary by the host cannot unmap it, and leaves through FreeLibraryAndExitThread. Nothing of it
// runs in DllMain; at process exit the system ends the thread wherever it is, and what is parked is left as it is.
struct TimerCall {
    uint32_t ms = 0;
    std::function<uint32_t()> fn;
    HMODULE self = nullptr;
};

DWORD WINAPI TimerThread(void* parameter) {
    auto* call = static_cast<TimerCall*>(parameter);
    const HMODULE self = call->self;
    uint32_t ms = call->ms;
    while (ms) {
        Sleep(ms);
        try {
            ms = call->fn();
        } catch (...) {
            ms = 0;
        }
    }
    delete call;
    FreeLibraryAndExitThread(self, 0);
}

bool StartTimer(uint32_t ms, std::function<uint32_t()> fn) {
    HMODULE self = nullptr;
    if (!GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS, reinterpret_cast<LPCWSTR>(&TimerThread), &self) || !self) return false;
    auto* call = new TimerCall{ms, std::move(fn), self};
    const HANDLE thread = CreateThread(nullptr, 0, &TimerThread, call, 0, nullptr);
    if (!thread) {
        delete call;
        FreeLibrary(self);
        return false;
    }
    CloseHandle(thread);
    return true;
}

// The callables Get() gives the front end that keep a string of the configuration. Named classes, not lambdas that
// capture the string by copy: with MSVC's /Zc:lambda (on with /permissive-, as in CI) a lambda whose capture copy throws
// (std::bad_alloc) terminates the process instead of unwinding; the same copy in a class constructor unwinds normally
// (async_create.h DeviceRelease).
//
// Where the converted model is looked for (config.h ModelPlace): wronggpu.ini [fsr] model_dir, the game's data folder,
// then the runtime's own default.
struct ModelPlacesOf {
    std::string modelDir;
    explicit ModelPlacesOf(const std::string& d) : modelDir(d) {}
    std::vector<ModelPlace> operator()() const {
        std::vector<ModelPlace> places;
        const std::wstring data = Place().dataDir;
        if (!modelDir.empty()) {
            std::wstring dir = Widen(modelDir);
            const bool absolute = dir.size() > 1 && (dir[1] == L':' || (dir[0] == L'\\' && dir[1] == L'\\'));
            if (!absolute) dir = data + L"\\" + dir;
            places.push_back({Narrow(dir), "wronggpu.ini model_dir", false});
        }
        std::wstring exeDir = wronggpu::ModuleFilePathWide(nullptr);
        const size_t sl = exeDir.find_last_of(L"\\/");
        exeDir = sl == std::wstring::npos ? std::wstring(L".") : exeDir.substr(0, sl);
        const std::wstring gameData = exeDir + L"\\WrongGPU";
        places.push_back({Narrow(gameData + L"\\model"), "the game's data folder", false});
        if (_wcsicmp(data.c_str(), gameData.c_str()) != 0) places.push_back({Narrow(data + L"\\model"), "the data folder (WG_DATA_DIR)", false});
        wchar_t local[MAX_PATH] = {};
        const DWORD m = GetEnvironmentVariableW(L"LOCALAPPDATA", local, MAX_PATH);
        if (m > 0 && m < MAX_PATH) places.push_back({Narrow(std::wstring(local, m) + L"\\WrongGPU\\model"), "%LOCALAPPDATA%", true});
        return places;
    }
};
// AMD's upscaler for fallbacks and the version list: the DLL next to this one, else the comparison DLL.
struct OriginalOrCompare {
    std::string dll;
    explicit OriginalOrCompare(const std::string& d) : dll(d) {}
    const OriginalApi* operator()() const {
        if (const OriginalApi* o = LoadOriginal()) return o;
        return CompareApi(dll);
    }
};

FrontEnd& Get() {
    std::call_once(g_initOnce, [] {
        // Shared settings first (clean/wgshared): data folder, wronggpu.ini, the shared block (this DLL writes it when
        // hosted: it is the voice of the host's menu), the folder sidecar and the OptiScaler.ini poll.
        InitSettings(g_hosted);
        Config config = ReadSettings("");
#if defined(WRONGGPU_HOSTED_DESKTOP_BADGE_QA)
        const Config hostedBadgeConfig = config;
#endif
        if (g_hosted) {
            config.hosted = true;
            // Hosted badge experiment is held after a Crimson cold-start GPU watchdog.
            // Keep dispatch-derived provider status, but do not enable drawing here
            // until queue/hook/split interaction is validated across restarts.
            config.overlay = "off";
            config.compareKey = "off";
            config.customSlot = Place().customSlot;
            if (Place().customSlot) {
                L().Write("hosted in OptiScaler's custom upscaler folder \"%s\": the entry \"%s\" of its main upscaler list; WrongGPU's"
                          " choices (SR models, RR modes: %d entries) are its versions; WrongGPU's overlay and compare key are off",
                          Narrow(Place().folder).c_str(), Narrow(Place().folder).c_str(), OwnEntryCount());
                L().Write("custom slot: when WrongGPU's upscaler cannot be created, the create fails (once; remembered for this session) and"
                          " OptiScaler's own upscaler takes over; AMD's FSR is used for that only from an AMD DLL next to this one"
                          " (amd_fidelityfx_dx12.wronggpu-original.dll), never through the game's FidelityFX DLLs (they lead back into"
                          " OptiScaler and to this DLL again)");
            } else
                L().Write("hosted by another upscaler mod (loaded under its own name): WrongGPU is one of its FidelityFX upscaler versions,"
                          " listed first; WrongGPU's overlay and compare key are off (hosted badge validation is pending;"
                          " the host's menu switches upscalers)");
        }
        Platform p;
        p.original = &LoadOriginal;
        p.backend = &ResolveBackend;
        p.log = [](const std::string& line) { L().WriteLine(line); };
        p.flushLog = [] { L().Flush(); };
        p.logBytes = [] { return L().Bytes(); };
        p.resourceInfo = &D3D12ResourceInfo;
        p.timer = &StartTimer;
        p.retainDevice = &RetainDevice;
        p.releaseDevice = &ReleaseDevice;
        p.deviceLost = [](void* device) {
            return device && static_cast<ID3D12Device*>(device)->GetDeviceRemovedReason() != S_OK;
        };
        AddSplitServices(p, config);
        // The FidelityFX front end installs the D3D12 hooks only after a WrongGPU upscaler was created (frontend.h).
        p.services = &SplitServicesDeferred;
        p.activateServices = &ActivateSplitServices;
        p.splitBlocked = &ListSplitBlocked;
        // Where the converted model is looked for (config.h ModelPlace): wronggpu.ini [fsr] model_dir, the game's data
        // folder, then the runtime's own default. Files a game reads must be in places the game's process sees: the
        // installer's %LOCALAPPDATA% copy can be private to another app's container.
        p.modelPlaces = ModelPlacesOf(config.modelDir);
        p.fileExists = [](const std::string& path) {
            const DWORD a = GetFileAttributesW(Widen(path).c_str());
            return a != INVALID_FILE_ATTRIBUTES && !(a & FILE_ATTRIBUTE_DIRECTORY);
        };
#if defined(WRONGGPU_HOSTED_DESKTOP_BADGE_QA)
        if (g_hosted) {
            ConfigureHostedBadge(hostedBadgeConfig);
            p.gpuTimer = {}; // no extra queries recorded into the game's command lists
            p.status = [](const UpscaleStatus& status) { OverlayStatus(status); ReportHostedBadge(); };
            L().Write("hosted badge: experimental Windows-composed provider status; no overlay graphics hooks or extra GPU timing");
        }
#endif
        if (g_hosted) {
            // The game's own amd_fidelityfx_dx12.dll is the host's (it hooks it): AMD's upscaler for fallbacks and for
            // the version list comes from the comparison DLL (AMD's FSR 4 next to us or the one the installer found).
            const std::string dll = config.compareDll;
            if (config.customSlot) {
                // OptiScaler's custom slot: only an AMD DLL placed next to this one; the game's FidelityFX DLLs are the
                // host's way in and would route the call back to this DLL (config.h Config::customSlot).
                p.original = []() -> const OriginalApi* { return LoadOriginal(); };
            } else {
                p.original = OriginalOrCompare(dll);
            }
            // If WrongGPU.asi has hooked OptiScaler's menu, its entry says WrongGPU or AMD, and the game's model choice
            // record (model_choice_record.h, next to wronggpu.ini) says which WrongGPU model. Absent or not hooked: hosted
            // mode as before.
            p.asiPlugin = [](char* model) { return QueryAsiPlugin(model); };
            p.asiShowModel = [](char model) { ShowAsiModel(model); };
            p.asiReportSr = [](const SrReport& r) {  // the card's status lines (rc3): what runs, a switch, a start failure
                ReportAsiSr(r.state, r.running, r.pending, r.runningMode, r.errorModel, r.errorKind, r.errorWidth, r.errorHeight, r.errorReason);
            };
        }
        // The game's model choice record (model_choice_record.h, in the game's WrongGPU state folder wronggpu-setup next to
        // wronggpu.ini) decides the WrongGPU model on every route; the plugin writes its menu's model clicks there too.
        // Always set: without the state folder the front end runs the session's last choice, wronggpu.ini's model item or
        // Model M, never the host's stored index (FrontEnd::ChooseModel).
        {
            const std::wstring ini = Place().ini;
            const size_t slash = ini.find_last_of(L"\\/");
            const std::filesystem::path record =
                wgchoice::RecordPathIn(slash == std::wstring::npos ? std::wstring(L".") : ini.substr(0, slash));
            p.choiceRecord = record;
            p.readWronggpuIni = WronggpuIniReader(ini);  // the upgrade migration reads wronggpu.ini's model item under the state lock
            if (g_hosted) {
                SetAsiChoiceRecord(record.wstring());
                SetHostIniPollModels(false);  // the host's stored index is no model choice (the record decides)
            }
        }
        p.menu = Menu();
        g_frontEnd = new FrontEnd(p, config);
        // The host's persisted choice (OptiScaler.ini [<custom folder>] UpscalerIndex): applied as if chosen in the menu,
        // also while the host never creates an upscaler (an RR-only session). The poll reads the file's time at most
        // once per second.
        if (HostIniPollActive()) {
            PollHostIni();
            if (!StartTimer(200, [] { return PollHostIni() ? 200u : 0u; })) L().Write("settings: the OptiScaler.ini poll could not start");
        }
    });
    return *g_frontEnd;
}

template <class F>
ffxReturnCode_t Guard(const char* name, F&& f) {
    try {
        return f();
    } catch (const std::exception& e) {
        L().Write("ERROR: exception in %s: %s", name, e.what());
    } catch (...) {
        L().Write("ERROR: unknown exception in %s", name);
    }
    return FFX_API_RETURN_ERROR;
}

}  // namespace

extern "C" {

FFX_API_ENTRY ffxReturnCode_t ffxCreateContext(ffxContext* context, ffxCreateContextDescHeader* desc, const ffxAllocationCallbacks* memCb) {
    return Guard("ffxCreateContext", [&] { return Get().CreateContext(context, desc, memCb); });
}

FFX_API_ENTRY ffxReturnCode_t ffxDestroyContext(ffxContext* context, const ffxAllocationCallbacks* memCb) {
    return Guard("ffxDestroyContext", [&] { return Get().DestroyContext(context, memCb); });
}

FFX_API_ENTRY ffxReturnCode_t ffxConfigure(ffxContext* context, const ffxConfigureDescHeader* desc) {
    return Guard("ffxConfigure", [&] { return Get().Configure(context, desc); });
}

FFX_API_ENTRY ffxReturnCode_t ffxQuery(ffxContext* context, ffxQueryDescHeader* desc) {
    return Guard("ffxQuery", [&] { return Get().Query(context, desc); });
}

FFX_API_ENTRY ffxReturnCode_t ffxDispatch(ffxContext* context, const ffxDispatchDescHeader* desc) {
    return Guard("ffxDispatch", [&] { return Get().Dispatch(context, desc); });
}

// The WrongGPU edition of OptiScaler: the player selected the listed version `versionId` in its upscaler list
// (include/wronggpu_host_event.h). Not hosted or not initialised: WG_HOST_SELECTED_NOT_READY.
__declspec(dllexport) int32_t WrongGpuHostSelectedV1(uint64_t versionId) {
    int32_t rc = WG_HOST_SELECTED_NOT_READY;
    Guard("WrongGpuHostSelectedV1", [&] {
        FrontEnd& fe = Get();
        if (fe.GetConfig().hosted) rc = fe.HostSelected(versionId);
        return FFX_API_RETURN_OK;
    });
    return rc;
}

}  // extern "C"

BOOL WINAPI DllMain(HINSTANCE instance, DWORD reason, LPVOID reserved) {
    if (reason == DLL_PROCESS_ATTACH) {
        g_self = instance;
        DisableThreadLibraryCalls(instance);
        const std::wstring file = wronggpu::ModuleFilePathWide(instance);
        const size_t sl = file.find_last_of(L"\\/");
        const wchar_t* base = sl == std::wstring::npos ? file.c_str() : file.c_str() + sl + 1;
        g_hosted = _wcsicmp(base, L"amd_fidelityfx_dx12.dll") != 0;
        L().Write("WrongGPU FSR front end %s (%s) loaded as %s", WRONGGPU_FSR_VERSION, WRONGGPU_FSR_BUILD, Narrow(file).c_str());
        wgwin::PrepareHipEnvironment();
    } else if (reason == DLL_PROCESS_DETACH) {
        if (!reserved) L().Write("WrongGPU FSR front end unloaded");  // FreeLibrary; at process exit, say nothing
    }
    return TRUE;
}
