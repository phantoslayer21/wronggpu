// SPDX-License-Identifier: MIT
// WrongGPU frame generation (issue #96, F8b): the ASI plugin entry (WrongGPU-FG.asi).
//
// OptiScaler's ASI convention, as the RR hook plugin and WrongGPU.asi use it: the host calls the exported
// InitializeASI once, early, before the game's first interface call. DllMain does nothing beyond turning thread
// notifications off (no loads, threads or file writes under the loader lock).
//
// InitializeASI reads the per-game FG settings (<game folder>\wronggpu-setup\settings.ini, fg_settings.h; the game
// folder is WG_STATE_GAME_DIR when set, else the exe's folder). fg=off (the default) returns at once: no transport is
// created and nothing is patched. With fg=on the transport installs only for a qualified title binding, selected by the
// exe's file name (fg_title_bindings.h: Cyberpunk 2077 on interface build 2.14.1); any other exe logs that this build
// carries no binding for it and patches nothing. A second InitializeASI call does nothing. WrongGPU_FgShutdown puts
// every redirected slot back;
// WrongGPU_FgQuery reports what happened (the launcher and the tests read it).
// F9: with a binding selected, the FG session host (fg_asi_session.h) checks the local FG model
// (%LOCALAPPDATA%\WrongGPU\model\FG\model.wgcm) and the kernel folder (wronggpu-setup\fg-dxil next to this plugin)
// before anything is patched; either failing: nothing is installed (kFgRuntimeMissing). Passing: the host is the
// transport's presenter and present route (session, D3D12 executor, Windows presenter). fg_net2=exact (developer key,
// not an FG settings key) selects net2's exact path for validation; the fast path is the default. The host probes the
// RDNA 4 adapter before the transport writes its first slot (FgAsiConfig::core_ready), so the game's first FG support
// query already sees the route; the transport logs each FG call kind (support query, function lookup, options, state)
// when its result changes. Developer builds: WRONGGPU_FG_DUMP_DIR turns on the frame dumps (fg_asi_session.h).
// Log: wronggpu-fg.log in the game folder, one append per line.
// Written for this project.
#include "fg_asi_transport.h"
#include "fg_asi_session.h"
#include "fg_title_bindings.h"

#include <memory>
#include <new>

#include "../../runtime/fg_d3d12_executor.h"
#include "wg_state.h"

namespace {

using namespace wgfgp;
using namespace wgfgp::win32;

std::atomic<int> g_init{0};           // 0: not called, 1: running, 2: done
std::atomic<uint32_t> g_init_calls{0};
std::atomic<uint32_t> g_result{kFgNullArgument};
std::atomic<uint32_t> g_settings_enabled{0};
FgAsiTransport* g_transport = nullptr;  // created only with fg=on; never freed while the plugin is loaded
std::wstring g_log_path;

std::wstring ExeDir() {
    wchar_t buf[MAX_PATH * 2];
    const DWORD n = GetModuleFileNameW(nullptr, buf, MAX_PATH * 2);
    if (!n || n >= MAX_PATH * 2) return L".";
    std::wstring w(buf, n);
    const size_t slash = w.find_last_of(L"\\/");
    return slash == std::wstring::npos ? L"." : w.substr(0, slash);
}

std::wstring StateGameDir() {
    wchar_t buf[MAX_PATH * 2];
    const DWORD n = GetEnvironmentVariableW(L"WG_STATE_GAME_DIR", buf, MAX_PATH * 2);
    if (n > 0 && n < MAX_PATH * 2) return std::wstring(buf, n);
    return ExeDir();
}

void LogLine(void*, const char* line) {
    if (g_log_path.empty() || !line) return;
    HANDLE h = CreateFileW(g_log_path.c_str(), FILE_APPEND_DATA, FILE_SHARE_READ | FILE_SHARE_WRITE, nullptr,
                           OPEN_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (h == INVALID_HANDLE_VALUE) return;
    SYSTEMTIME t;
    GetLocalTime(&t);
    char stamp[32];
    const int sn = wsprintfA(stamp, "%04u-%02u-%02u %02u:%02u:%02u ", t.wYear, t.wMonth, t.wDay, t.wHour, t.wMinute,
                             t.wSecond);
    std::string s(stamp, sn > 0 ? static_cast<size_t>(sn) : 0);
    s += line;
    s += "\r\n";
    DWORD written = 0;
    WriteFile(h, s.data(), static_cast<DWORD>(s.size()), &written, nullptr);  // one append per line
    CloseHandle(h);
}

std::wstring ExeLeaf() {
    wchar_t buf[MAX_PATH * 2];
    const DWORD n = GetModuleFileNameW(nullptr, buf, MAX_PATH * 2);
    if (!n || n >= MAX_PATH * 2) return L"";
    std::wstring w(buf, n);
    const size_t slash = w.find_last_of(L"\\/");
    return slash == std::wstring::npos ? w : w.substr(slash + 1);
}

FgSl214Codec* g_codec = nullptr;  // the selected binding's codec; never freed while the plugin is loaded
FgAsiSessionHost* g_host = nullptr;  // the FG session host; never freed while the plugin is loaded

std::string Utf8(const std::wstring& w) {
    if (w.empty()) return std::string();
    const int n = WideCharToMultiByte(CP_UTF8, 0, w.c_str(), static_cast<int>(w.size()), nullptr, 0, nullptr, nullptr);
    std::string s(n > 0 ? static_cast<size_t>(n) : 0, '\0');
    if (n > 0) WideCharToMultiByte(CP_UTF8, 0, w.c_str(), static_cast<int>(w.size()), &s[0], n, nullptr, nullptr);
    return s;
}

// This plugin's own folder (the kernel folder is next to it).
std::wstring ModuleDir() {
    HMODULE self = nullptr;
    if (!GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                            reinterpret_cast<LPCWSTR>(&ModuleDir), &self))
        return ExeDir();
    wchar_t buf[MAX_PATH * 2];
    const DWORD n = GetModuleFileNameW(self, buf, MAX_PATH * 2);
    if (!n || n >= MAX_PATH * 2) return ExeDir();
    std::wstring w(buf, n);
    const size_t slash = w.find_last_of(L"\\/");
    return slash == std::wstring::npos ? L"." : w.substr(0, slash);
}

std::wstring LocalAppData() {
    wchar_t buf[MAX_PATH * 2];
    const DWORD n = GetEnvironmentVariableW(L"LOCALAPPDATA", buf, MAX_PATH * 2);
    return (n > 0 && n < MAX_PATH * 2) ? std::wstring(buf, n) : std::wstring();
}

std::string RawValue(const wgstate::Settings& st, const char* key) {
    for (const auto& kv : st.kv)
        if (kv.first == key) return kv.second;
    return std::string();
}

void Initialize() {
    const std::wstring dir = StateGameDir();
    g_log_path = wgstate::JoinPath(dir, L"wronggpu-fg.log");
    const wgstate::Settings st = wgstate::ReadSettings(dir);
    std::vector<FgSettingsIssue> issues;
    const FgSettings settings = FgSettingsFromKeyValues(st.kv, &issues);
    for (const auto& i : issues)
        LogLine(nullptr, ("fg: setting " + i.key + "=" + i.value + " not used (" + i.why + "); its default stays").c_str());
    g_settings_enabled.store(settings.enabled ? 1u : 0u);
    if (!settings.enabled) {
        // INSTALL-OFF GATE: no transport, nothing read from the process, nothing patched
        g_result.store(kFgInstallOff);
        LogLine(nullptr, "fg: frame generation is off (fg=off or not set); no hooks installed");
        return;
    }
    FgAsiConfig cfg;
    cfg.settings = settings;
    // the qualified title binding for this exe, if this build carries one (else the transport patches nothing)
    FgTitleBinding binding;
    if (FgSelectShippedBinding(ExeLeaf(), &binding)) {
        LogLine(nullptr, ("fg: title binding " + binding.id + " (" + binding.title + ")").c_str());
        // the runtime the session needs, checked before anything is patched
        FgAsiSessionConfig hc;
        hc.settings = settings;
        const std::wstring lad = LocalAppData();
        hc.model_path = lad.empty() ? std::string() : Utf8(lad) + "\\WrongGPU\\model\\FG\\model.wgcm";
        hc.dxil_dir = Utf8(ModuleDir()) + "\\wronggpu-setup\\fg-dxil";
        const std::string net2 = RawValue(st, "fg_net2");
        hc.net2_exact = net2 == "exact";
        if (!net2.empty() && net2 != "exact" && net2 != "fast")
            LogLine(nullptr, ("fg: fg_net2=" + net2 + " not used (fast or exact); the fast path stays").c_str());
        hc.log = &LogLine;
        std::unique_ptr<wgfg::FgExecWeights> weights(new (std::nothrow) wgfg::FgExecWeights());
        std::string why;
        if (!weights || !FgAsiSessionPrecheck(hc, weights.get(), &why)) {
            g_result.store(kFgRuntimeMissing);
            LogLine(nullptr, ("fg: " + why + "; no hooks installed").c_str());
            return;
        }
        LogLine(nullptr, (std::string("fg: model and kernels checked (net2 ") + (hc.net2_exact ? "exact" : "fast") +
                          ")").c_str());
        g_host = new (std::nothrow) FgAsiSessionHost(hc, std::move(weights));
        if (!g_host) {
            g_result.store(kFgNullArgument);
            return;
        }
        FgApplyBinding(binding, &cfg);
        g_codec = new (std::nothrow) FgSl214Codec(binding.layout);
        cfg.codec = g_codec;
        cfg.presenter = g_host;
        cfg.present_route = &FgAsiSessionHost::Route;
        cfg.present_route_ctx = g_host;
        cfg.factory_note = &FgAsiSessionHost::FactoryNote;
        cfg.factory_note_ctx = g_host;
        cfg.core_ready = &FgAsiSessionHost::CoreReady;  // route facts and the adapter probe before any slot is live
        cfg.core_ready_ctx = g_host;
        cfg.tag_retain = &FgAsiSessionHost::RetainTag;  // transient D3D12 tags are copied by the host
        cfg.tag_retain_ctx = g_host;
#if defined(WRONGGPU_FG_DEV_QUALIFIED) && WRONGGPU_FG_DEV_QUALIFIED
        LogLine(nullptr, "fg: developer build: frame generation modes are taken without the release gate");
#endif
    } else {
        LogLine(nullptr, "fg: this build carries no title binding for this game");
    }
    cfg.core.settings = settings;
    cfg.log = &LogLine;
    g_transport = new (std::nothrow) FgAsiTransport();
    if (!g_transport) {
        g_result.store(kFgNullArgument);
        return;
    }
    const uint32_t r = g_transport->install(cfg);
    g_result.store(r);
    if (r == kFgInstalled && g_host) g_host->OnInstalled(g_transport->core());
}

}  // namespace

extern "C" {

// Status for the launcher and the tests. Size-tagged: a caller's smaller struct is refused.
struct WgFgAsiStatus {
    uint32_t struct_size;
    uint32_t result;            // FgInstallResult
    uint32_t settings_enabled;  // fg=on read from the settings file
    uint32_t transport_created;
    uint32_t patched_slots;
    uint32_t init_calls;
};

__declspec(dllexport) void InitializeASI() {
    g_init_calls.fetch_add(1);
    int expected = 0;
    if (!g_init.compare_exchange_strong(expected, 1)) return;  // a second call: nothing changes
    try {
        Initialize();
    } catch (...) {
        g_result.store(kFgNullArgument);
        LogLine(nullptr, "fg: initialisation failed; nothing stays patched");
        if (g_transport) g_transport->restore();
    }
    g_init.store(2);
}

__declspec(dllexport) void WrongGPU_FgShutdown() {
    if (g_init.load() != 2 || !g_transport) return;
    g_transport->restore();
    LogLine(nullptr, "fg: shut down; every redirected slot put back");
}

__declspec(dllexport) int WrongGPU_FgQuery(WgFgAsiStatus* out) {
    if (!out || out->struct_size < sizeof(WgFgAsiStatus)) return 0;
    out->result = g_result.load();
    out->settings_enabled = g_settings_enabled.load();
    out->transport_created = g_transport ? 1u : 0u;
    out->patched_slots = g_transport ? static_cast<uint32_t>(g_transport->patch_count()) : 0u;
    out->init_calls = g_init_calls.load();
    return 1;
}

}  // extern "C"

BOOL WINAPI DllMain(HINSTANCE inst, DWORD reason, LPVOID) {
    if (reason == DLL_PROCESS_ATTACH) DisableThreadLibraryCalls(inst);
    return TRUE;
}
