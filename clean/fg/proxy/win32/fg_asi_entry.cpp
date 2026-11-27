// SPDX-License-Identifier: GPL-3.0-only
// WrongGPU frame generation (issue #96, F8b): the ASI plugin entry (WrongGPU-FG.asi).
//
// OptiScaler's ASI convention, as the RR hook plugin and WrongGPU.asi use it: the host calls the exported
// InitializeASI once, early, before the game's first interface call. DllMain does nothing beyond turning thread
// notifications off (no loads, threads or file writes under the loader lock).
//
// InitializeASI reads the per-game FG settings (<game folder>\wronggpu-setup\settings.ini, fg_settings.h; the game
// folder is WG_STATE_GAME_DIR when set, else the exe's folder). fg=off (the default) returns at once: no transport is
// created and nothing is patched. With fg=on the transport installs for a qualified title binding, selected by the
// exe's file name (fg_title_bindings.h: Cyberpunk 2077 on interface build 2.14.1), or else for the generic interface 2.x
// binding when the game links the interface (already loaded, or named in the exe's import / delay-load table); the
// transport's version gate then admits 2.4 and later. Any other exe logs that this build carries no binding for it and
// patches nothing (with OptiScaler's plugin interface 3, the generic binding still starts when OptiScaler hooks an
// interface the game loads later). A second InitializeASI call does nothing. WrongGPU_FgShutdown puts every redirected
// slot back; WrongGPU_FgQuery reports what happened (the launcher and the tests read it).
//
// OptiPluginInit (OptiScaler's plugin interface, wg_host_menu_abi.h; called right after InitializeASI):
//  - v6: claim FG ownership before installing the transport and register controls in the Frame Gen tab.
//    Earlier hosts use the Custom card and require their native FG settings to be off at startup.
//  - the session host learns that OptiScaler hosts it: fg_presenter=auto then presents through OptiScaler's swap chain
//    (a DXGI flip presenter), so OptiScaler's menu and overlay stay visible on every image (fg_asi_session.h). The same
//    holds when InitializeASI was called from OptiScaler's proxy module (OptiScaler.ini next to it);
//  - plugin interface 3: OptiScaler's game-interface hook. When it hooks the interface, the FG handlers become routes in
//    front of the interface's exports (FgAsiTransport::install_routes): another plugin that redirects the same import
//    slots afterwards and forwards to the export (the RR plugin) no longer bypasses FG, and a game that loads the
//    interface at run time is served too.
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
#include "fg_panel.h"
#include "fg_title_bindings.h"

#include <intrin.h>

#include <atomic>

#include <memory>
#include <mutex>
#include <new>

#include "../../runtime/fg_d3d12_executor.h"
#include "wg_state.h"

namespace {

using namespace wgfgp;
using namespace wgfgp::win32;

std::atomic<int> g_init{0};           // 0: not called, 1: running, 2: done, 3: overlapping call
std::atomic<uint32_t> g_init_calls{0};
std::atomic_flag g_plugin_init_running = ATOMIC_FLAG_INIT;
std::atomic<uint32_t> g_result{kFgNullArgument};
std::atomic<uint32_t> g_settings_enabled{0};
std::atomic<bool> g_menu_enabled{false};
std::atomic<bool> g_known_multiplier{false};
std::atomic<bool> g_prepared_by_host{false};
std::wstring g_legacy_host_ini;
std::atomic<bool> g_host_conflict{false};
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

void LogToPath(const std::wstring& path, const char* line) {
    if (path.empty() || !line) return;
    HANDLE h = CreateFileW(path.c_str(), FILE_APPEND_DATA, FILE_SHARE_READ | FILE_SHARE_WRITE, nullptr,
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

void LogLine(void*, const char* line) { LogToPath(g_log_path, line); }

void LogEarly(const char* line) noexcept {
    try {
        LogToPath(wgstate::JoinPath(StateGameDir(), L"wronggpu-fg.log"), line);
    } catch (...) {
    }
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
std::atomic<FgAsiSessionHost*> g_host_seen{nullptr};  // g_host, for the menu card (any thread)
std::atomic<FgProxy*> g_core{nullptr};               // the installed transport's core, for the menu card
FgSettings g_settings;                               // as read at InitializeASI
FgTitleBinding g_title_binding;
FgAsiConfig g_cfg;                                   // the transport configuration (route installs use it again)
std::atomic<bool> g_generic_deferred{false};  // no binding at start-up: the generic one may start when OptiScaler hooks the interface
std::atomic<bool> g_host_overlay{false};  // OptiScaler hosts this plugin
const char* g_overlay_why = "OptiScaler's plugin interface";
std::recursive_mutex g_route_mu;          // OptiScaler's hook callbacks (any thread) and the setup they do
const WgHostMenuApi* g_host_api = nullptr;
WgHostMenuPlugin* g_host_plugin = nullptr;
bool g_host_init_result = false;

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

// The game links the interface module `leaf`: already loaded, or named in the exe's import or delay-load table.
bool InterfaceLinked(const FgImportBinding& im) {
    std::wstring w;
    for (char c : im.interposer_leaf) w.push_back(static_cast<wchar_t>(static_cast<unsigned char>(c)));
    if (!w.empty() && GetModuleHandleW(w.c_str())) return true;
    std::vector<std::string> names(im.role_names, im.role_names + kFgRoleCount);
    wgrr::guard::Budget budget(200000, 100);
    return !FgCollectSlots(GetModuleHandleW(nullptr), im.interposer_leaf.c_str(), names, budget).empty();
}

panel::Status PanelStatus();

void PublishCurrent(bool immediate) {
    const auto sequence = panel::NextReportSequence();
    panel::Publish(PanelStatus(), sequence, immediate);
}

// Keep the menu reporting outside the session's present-thread lock.
int32_t PresentRoute(void* ctx, FgProxy* core, const WgFgPresentDesc* desc, WgFgPresentResult* out) {
    const auto sequence = panel::NextReportSequence();
    const auto result = FgAsiSessionHost::Route(ctx, core, desc, out);
    panel::Publish(PanelStatus(), sequence);
    return result;
}

// A binding was taken: the runtime the session needs is checked before anything is patched, then the session host and
// the codec go into g_cfg. false: nothing is installed (g_result says why).
bool SetUpBinding(const FgTitleBinding& binding, const wgstate::Settings& st) {
    FgAsiSessionConfig hc;
    hc.settings = g_settings;
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
        return false;
    }
    LogLine(nullptr, (std::string("fg: model and kernels checked (net2 ") + (hc.net2_exact ? "exact" : "fast") +
                      ")").c_str());
    g_host = new (std::nothrow) FgAsiSessionHost(hc, std::move(weights));
    if (!g_host) {
        g_result.store(kFgNullArgument);
        return false;
    }
    if (g_host_overlay.load()) g_host->SetHostOverlay(g_overlay_why);
    g_host_seen.store(g_host);
    FgApplyBinding(binding, &g_cfg);
    g_cfg.core.panel_multipliers = true;
    g_codec = new (std::nothrow) FgSl214Codec(binding.layout);
    g_cfg.codec = g_codec;
    g_cfg.presenter = g_host;
    g_cfg.present_route = &PresentRoute;
    g_cfg.present_route_ctx = g_host;
    g_cfg.factory_note = &FgAsiSessionHost::FactoryNote;
    g_cfg.factory_note_ctx = g_host;
    g_cfg.core_ready = &FgAsiSessionHost::CoreReady;  // route facts and the adapter probe before any slot is live
    g_cfg.core_ready_ctx = g_host;
    g_cfg.tag_retain = &FgAsiSessionHost::RetainTag;  // transient D3D12 tags are copied by the host
    g_cfg.tag_retain_ctx = g_host;
#if defined(WRONGGPU_FG_DEV_QUALIFIED) && WRONGGPU_FG_DEV_QUALIFIED
    LogLine(nullptr, "fg: developer build: the developer diagnostics are compiled in (frame dumps)");
#endif
    return true;
}

void NoteInstalled(uint32_t r) {
    g_result.store(r);
    if (r == kFgInstalled && g_host && g_transport && g_transport->core()) {
        g_host->OnInstalled(g_transport->core());
        g_core.store(g_transport->core());
        g_transport->core()->enable_generation(g_menu_enabled.load());
    }
}

void Initialize() {
    const std::wstring dir = StateGameDir();
    g_log_path = wgstate::JoinPath(dir, L"wronggpu-fg.log");
    const wgstate::Settings st = wgstate::ReadSettings(dir);
    std::vector<FgSettingsIssue> issues;
    const FgSettings settings = FgSettingsFromKeyValues(st.kv, &issues);
    for (const auto& i : issues)
        LogLine(nullptr, ("fg: setting " + i.key + "=" + i.value + " not used (" + i.why + "); its default stays").c_str());
    g_settings = settings;
    g_settings_enabled.store(settings.enabled ? 1u : 0u);
    g_menu_enabled.store(settings.enabled);
    if (FgSelectShippedBinding(ExeLeaf(), &g_title_binding))
        g_known_multiplier.store(g_title_binding.core.has_mfg_selector);
    if (!settings.enabled) {
        // INSTALL-OFF GATE: no transport, no import scan, no hooks
        g_result.store(kFgInstallOff);
        LogLine(nullptr, "fg: frame generation is off (fg=off or not set); no hooks installed");
    }
}

void StartTransport() {
    if (!g_settings.enabled || g_transport) return;

    FgTitleBinding binding = g_title_binding;
    bool have = binding.valid;
    g_cfg.settings = g_settings;
    // the qualified title binding for this exe, else the generic interface 2.x binding when the game links the
    // interface (else the transport patches nothing)
    if (have) {
        LogLine(nullptr, ("fg: title binding " + binding.id + " (" + binding.title + ")").c_str());
    } else {
        const FgTitleBinding generic = FgSl2GenericBinding();
        if (generic.valid && generic.imports.qualified && InterfaceLinked(generic.imports)) {
            binding = generic;
            have = true;
            LogLine(nullptr, ("fg: generic binding " + binding.id + " (" + binding.title + "): no title binding for " +
                              Utf8(ExeLeaf()) + ", and the game links " + binding.imports.interposer_leaf).c_str());
        } else {
            g_generic_deferred = generic.valid && generic.imports.qualified;
            LogLine(nullptr, g_generic_deferred
                                 ? "fg: this build carries no title binding for this game, and the game interface is "
                                   "not linked yet; with OptiScaler's plugin interface 3 the generic binding starts "
                                   "when OptiScaler hooks the interface"
                                 : "fg: this build carries no title binding for this game");
        }
    }
    if (have && !SetUpBinding(binding, wgstate::ReadSettings(StateGameDir()))) return;
    g_cfg.core.settings = g_settings;
    g_cfg.log = &LogLine;
    g_transport = new (std::nothrow) FgAsiTransport();
    if (!g_transport) {
        g_result.store(kFgNullArgument);
        return;
    }
    NoteInstalled(g_transport->install(g_cfg));
}

// ---- OptiScaler (plugin interface) ----------------------------------------------------------------------------------

// The module InitializeASI returned into is OptiScaler's proxy (OptiScaler.ini next to it): OptiScaler hosts us.
void NoteCaller(void* return_address) {
    HMODULE m = nullptr;
    if (!return_address ||
        !GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                            static_cast<LPCWSTR>(return_address), &m) ||
        !m || m == GetModuleHandleW(nullptr))
        return;
    wchar_t buf[MAX_PATH * 2];
    const DWORD n = GetModuleFileNameW(m, buf, MAX_PATH * 2);
    if (!n || n >= MAX_PATH * 2) return;
    std::wstring path(buf, n);
    const size_t slash = path.find_last_of(L"\\/");
    if (slash == std::wstring::npos) return;
    const std::wstring ini = path.substr(0, slash + 1) + L"OptiScaler.ini";
    const DWORD a = GetFileAttributesW(ini.c_str());
    if (a == INVALID_FILE_ATTRIBUTES || (a & FILE_ATTRIBUTE_DIRECTORY)) return;
    g_legacy_host_ini = ini;
    g_overlay_why = "its proxy module loaded this plugin";
    g_host_overlay.store(true);
}

// The legacy host has no arbitration API. Its input and output are picked at game start.
bool LegacyNativeFgIsOff() {
    const auto is_default = [](const wchar_t* section, const wchar_t* key, const wchar_t* off) {
        wchar_t value[32];
        GetPrivateProfileStringW(section, key, L"auto", value, _countof(value), g_legacy_host_ini.c_str());
        return _wcsicmp(value, L"auto") == 0 || _wcsicmp(value, off) == 0;
    };
    return !g_legacy_host_ini.empty() && is_default(L"FrameGen", L"FGInput", L"nofg") &&
           is_default(L"FrameGen", L"FGOutput", L"nofg") && is_default(L"FrameGen", L"FGType", L"nofg") &&
           is_default(L"FrameGen", L"Enabled", L"false") && is_default(L"fakenvapi", L"ForceXeLL", L"false");
}

bool RegisterRoutes(void*, const FgRoute* routes, uint32_t count) {
    const WgHostMenuApi* api = g_host_api;
    if (!api || !WG_HOST_MENU_HAS(api, register_streamline_routes) || !routes) return false;
    static_assert(sizeof(FgRoute) == sizeof(WgHostSlRoute), "FgRoute is the host's route entry");
    std::vector<WgHostSlRoute> v(count);
    for (uint32_t i = 0; i < count; ++i) v[i] = WgHostSlRoute{routes[i].export_name, routes[i].handler, routes[i].next};
    return api->register_streamline_routes(g_host_plugin, v.data(), count);
}

std::string ModulePath(HMODULE m) {
    wchar_t buf[MAX_PATH * 2];
    const DWORD n = GetModuleFileNameW(m, buf, MAX_PATH * 2);
    return (n && n < MAX_PATH * 2) ? Utf8(std::wstring(buf, n)) : std::string("?");
}

// OptiScaler hooked a game interface module (plugin interface 3): the FG handlers become routes in front of its exports.
void OnHostHooked(void* module, uint32_t major, uint32_t minor, uint32_t patch, void*) {
    std::lock_guard<std::recursive_mutex> lock(g_route_mu);
    try {
        if (!g_transport || !module) return;
        const HMODULE m = static_cast<HMODULE>(module);
        LogLine(nullptr, ("fg: OptiScaler hooked the game interface " + ModulePath(m) + ", version " +
                          std::to_string(major) + "." + std::to_string(minor) + "." + std::to_string(patch))
                             .c_str());
        if (g_transport->active()) {
            g_transport->install_routes(g_cfg, m, major, minor, &RegisterRoutes, nullptr);
            return;
        }
        if (!g_generic_deferred || g_host) return;  // another refusal already decided this run
        const FgTitleBinding generic = FgSl2GenericBinding();
        const std::string path = ModulePath(m);
        const std::string& leaf = generic.imports.interposer_leaf;
        if (path.size() < leaf.size() || _stricmp(path.c_str() + path.size() - leaf.size(), leaf.c_str()) != 0) {
            LogLine(nullptr, "fg: not the game interface this build serves; nothing installed");
            return;
        }
        g_generic_deferred = false;
        if (panel::HasFrameGenApi(g_host_api) && !panel::ClaimOutput(g_host_api, g_host_plugin)) {
            g_host_conflict.store(true);
            LogLine(nullptr, "fg: OptiScaler refused the deferred FG claim; nothing installed");
            return;
        }
        LogLine(nullptr, ("fg: generic binding " + generic.id + " (" + generic.title + ")").c_str());
        if (SetUpBinding(generic, wgstate::ReadSettings(StateGameDir()))) {
            NoteInstalled(g_transport->install_routes(g_cfg, m, major, minor, &RegisterRoutes, nullptr));
            if (auto* core = g_core.load()) {
                const auto pending = panel::Current();
                core->update_live_settings(pending.preset, pending.multiplier);
            }
        }
        if (!g_transport->active()) panel::ReleaseOutput();
        PublishCurrent(true);
    } catch (...) {
        if (!g_transport || !g_transport->active()) panel::ReleaseOutput();
        LogLine(nullptr, "fg: OptiScaler's interface hook callback failed; nothing more is installed");
    }
}

panel::Status PanelStatus() {
    panel::Status s;
    s.enabled = g_menu_enabled.load();
    s.host_conflict = g_host_conflict.load();
    s.deferred = g_generic_deferred.load();
    s.install_result = g_result.load();
    s.installed = s.install_result == kFgInstalled;
    FgAsiSessionHost* h = g_host_seen.load();
    s.have_host = h != nullptr;
    if (h) s.live = h->LiveStatus();
    s.live.game_has_mfg_selector = g_known_multiplier.load();
    return s;
}

bool PanelApply(uint32_t preset, uint32_t multiplier) {
    std::unique_lock<std::recursive_mutex> lock(g_route_mu, std::try_to_lock);
    if (!lock) return false;
    FgProxy* core = g_core.load();
    return core && core->update_live_settings(preset, multiplier);
}

bool PanelEnable(bool enabled) {
    std::unique_lock<std::recursive_mutex> lock(g_route_mu, std::try_to_lock);
    if (!lock) return false;
    auto* core = g_core.load();
    if (!g_settings_enabled.load() || g_host_conflict.load() || (!core && !g_generic_deferred))
        return false;
    g_menu_enabled.store(enabled);
    if (core) core->enable_generation(enabled);
    const auto sequence = panel::NextReportSequence();
    auto status = PanelStatus();
    if (enabled) status.live.phase = kFgPhaseWaiting;
    panel::Publish(status, sequence, true);
    return true;
}

std::wstring PanelGameDir() { return StateGameDir(); }

void PanelLog(const char* line) { LogLine(nullptr, line); }

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

__declspec(dllexport) void OptiPluginPrepareASI() {
    g_prepared_by_host.store(true);
    if (g_init.load() != 0)
        LogEarly("fg: OptiPluginPrepareASI called after initialisation started; an earlier transport cannot become the hosted route");
}

// The return address also detects legacy OptiScaler hosts, which omit the prepare callback.
__declspec(dllexport) __declspec(noinline) void InitializeASI() {
    void* caller = _ReturnAddress();
    g_init_calls.fetch_add(1);
    int expected = 0;
    if (!g_init.compare_exchange_strong(expected, 1)) {
        if (expected == 1 && g_init.compare_exchange_strong(expected, 3))
            LogEarly("fg: overlapping InitializeASI calls; standing down for this run");
        return;
    }
    std::lock_guard<std::recursive_mutex> lock(g_route_mu);
    try {
        NoteCaller(caller);
        Initialize();
        if (g_init.load() == 1 && !g_prepared_by_host.load()) {
            if (!g_legacy_host_ini.empty() && !LegacyNativeFgIsOff()) {
                g_host_conflict.store(true);
                LogLine(nullptr, "fg: no exclusive legacy FG route; set OptiScaler's FG input and output to None, then restart");
            } else {
                StartTransport();
            }
        }
    } catch (...) {
        g_result.store(kFgNullArgument);
        LogLine(nullptr, "fg: initialisation failed; nothing stays patched");
        if (g_transport) g_transport->restore();
    }
    expected = 1;
    if (!g_init.compare_exchange_strong(expected, 2)) {
        g_core.store(nullptr);
        g_generic_deferred = false;
        if (g_transport) g_transport->restore();
        g_result.store(kFgAlreadyActive);
        g_init.store(2);
    }
}

// OptiScaler's plugin interface (see the top of this file). Called after InitializeASI; false: nothing was added.
__declspec(dllexport) bool OptiPluginInit(const WgHostMenuApi* api, WgHostMenuPlugin* plugin) {
    if (!api || !plugin) return false;
    if (g_plugin_init_running.test_and_set()) {
        LogEarly("fg: overlapping OptiPluginInit call declined");
        return false;
    }
    struct ClearInitFlag {
        ~ClearInitFlag() { g_plugin_init_running.clear(); }
    } clear_init_flag;
    if (g_init.load() != 2) {
        LogEarly("fg: OptiPluginInit overlapped or preceded InitializeASI; hosted FG declined");
        return false;
    }
    std::lock_guard<std::recursive_mutex> lock(g_route_mu);
    if (g_host_api) return api == g_host_api && plugin == g_host_plugin && g_host_init_result;
    g_host_api = api;
    g_host_plugin = plugin;
    try {
        g_host_overlay.store(true);
        if (g_host) g_host->SetHostOverlay("OptiPluginInit");
        panel::Hooks hooks;
        hooks.status = &PanelStatus;
        hooks.apply = &PanelApply;
        hooks.enable = &PanelEnable;
        hooks.game_dir = &PanelGameDir;
        hooks.log = &PanelLog;
        const bool card = panel::Init(api, plugin, hooks, g_settings);
        const bool v6 = panel::HasFrameGenApi(api);
        if (g_settings.enabled && g_result.load() != kFgAlreadyActive && !g_host_conflict.load()) {
            if (v6) {
                // In the v6 lifecycle only settings have been read. Never start without an accepted claim.
                if (g_transport && g_transport->active()) {
                    // A separate loader ran before this host's lifecycle. Stand down.
                    g_core.store(nullptr);
                    g_transport->restore();
                    g_host_conflict.store(true);
                    LogLine(nullptr, "fg: an earlier ASI loader started frame generation; routes removed");
                    return g_host_init_result = card;
                }
                if (!card || !panel::ClaimOutput(api, plugin)) {
                    g_host_conflict.store(true);
                    LogLine(nullptr, "fg: could not reserve OptiScaler's FG output; no hosted hooks installed");
                    return g_host_init_result = card;
                }
            } else if (g_prepared_by_host.load() || (!g_legacy_host_ini.empty() && !LegacyNativeFgIsOff())) {
                g_host_conflict.store(true);
                LogLine(nullptr, "fg: no exclusive legacy FG route; set OptiScaler's FG input and output to None, then restart");
                return g_host_init_result = card;
            }
            StartTransport();
        }

        bool routes = false;
        if (g_settings.enabled && g_transport && (g_transport->active() || g_generic_deferred)) {
            if (api->version >= 3 && WG_HOST_MENU_HAS(api, register_streamline_routes) &&
                WG_HOST_MENU_HAS(api, set_streamline_callback)) {
                LogLine(nullptr, "fg: asking OptiScaler for its game-interface hook (plugin interface 3)");
                routes = api->set_streamline_callback(plugin, &OnHostHooked, nullptr);
            } else {
                LogLine(nullptr, ("fg: OptiScaler's plugin interface " + std::to_string(api->version) +
                                  " has no game-interface routes (needs 3); the import slots stay the transport")
                                     .c_str());
            }
        }
        if (g_generic_deferred && !routes && (!g_transport || !g_transport->active()))
            g_generic_deferred = false;
        if (!g_transport || !g_transport->active()) {
            panel::ReleaseOutput();
            if (g_settings.enabled && !g_host_conflict.load())
                LogLine(nullptr, "fg: no transport routes installed; OptiScaler's FG output is available");
        }
        PublishCurrent(true);
        return g_host_init_result = card || routes;
    } catch (...) {
        g_core.store(nullptr);
        if (g_transport) g_transport->restore();
        panel::ReleaseOutput();
        LogLine(nullptr, "fg: OptiPluginInit failed; nothing was added to OptiScaler");
        return false;
    }
}

__declspec(dllexport) void WrongGPU_FgShutdown() {
    std::lock_guard<std::recursive_mutex> lock(g_route_mu);
    if (g_init.load() != 2) return;
    g_core.store(nullptr);
    if (g_transport) g_transport->restore();
    panel::ReleaseOutput();
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
