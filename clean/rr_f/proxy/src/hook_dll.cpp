// SPDX-License-Identifier: MIT
// Hook-mode delivery of the RR proxy (HOOK-MODE.md). Built as an OptiScaler ASI plugin
// (WrongGPU-RR.asi release, WrongGPU-RR-probe.asi probe), loaded by OptiScaler from
// bin\x64\plugins\. It does NOT replace the game's interface file. Instead it finds the already-loaded genuine
// interface module and redirects the import-address-table (IAT) slots of every module that imports it, for the RR
// path's functions, to our handlers (an exe that delay-loads the interface: the plugin loads the game's own file next to
// the exe and patches the exe's delay-load IAT); our handlers call through to the genuine export (the address the IAT held,
// which keeps OptiScaler's own inline patch on that export in the chain). Non-RR traffic is left untouched: those
// IAT slots are never patched, so the game reaches the genuine interface directly.
//
// The game interface itself is never inline-patched by us. The transport is the only thing that differs from the
// file-replacement proxy (src/proxy_dll.cpp): the proxy core (proxy_core.*), probe (probe.*), answers, settings
// contract and runtime adapter are reused unchanged.
//
// Entry: exported InitializeASI (OptiScaler's ASI convention, as WrongGPU.asi uses). Safe work only at DLL attach
// (records diagnostics into a fixed buffer, no loads/threads/file writes under the loader lock); the hooking runs
// from InitializeASI. Release: wronggpu-rr.log + wronggpu-rr.ini + wronggpu_runtime.dll (with the RR runtime) next to the game exe
// (bin\x64, the same place the file proxy used). Probe: wronggpu-rr-probe.log next to the game exe.
//
// Release 0.6.0 (owner 2026-11-02: one RR plugin, as few WrongGPU files as possible): ONE build serves every supported
// game interface family. It is generated from the 2.14.1 table (the complete role list; 2.8.0 has the same exports,
// roles, magic and GUIDs, 2.7.1 the same less the tag-for-frame role, spec annexes A9 / A10), and the route is chosen
// at run time from the game interface module's version (major.minor: 2.7 / 2.8 / 2.14), before the core is built:
// the core's api_minor follows it (the per-version answers of 61.5 and the frame-scoped paths of 2.8+), and on 2.7 the
// tag-for-frame role is neither redirected, routed nor bound (that interface has no such export). Any other version
// gets no game-interface route (nothing patched, no host routes); the NGX route (below) is registered all the same.
//
// clean/rr-ngx: games that call the NGX interface directly (Unreal Engine's plugin, Nixxes ports; spec page 66) get the
// same RR through the host's plugin interface version 4: OptiPluginInit registers the NGX RR provider (ngx_rr.h) when
// the host has it and the player's switch is on; it shares the runtime (rr_mode, model, wronggpu-rr.ini) with the
// game-interface route, and a game uses one route (the NGX route stays out while the game-interface route holds the RR
// feature and the other way round).
//
// clean/rr-sm2-shim: the SetupAPI identity shim (setupapi_identity.h), Spider-Man2.exe only: while DXGI reports the game
// another vendor / device pair than the kernel driver's (the host's GPU-name spoof), SetupAPI's instance ID and hardware /
// compatible IDs of the real adapter report the same pair as DXGI. The exe's import slots of setupapi.dll are redirected
// at InitializeASI; the decision is made at the game's first call (DXGI is not readable at InitializeASI; until it is,
// the host's own spoof settings in OptiScaler.ini give the pair). wronggpu-rr.ini setupapi_identity=auto|off.
//
// RR guard: the coexistence guard (rr_guard.h, HOOK-MODE.md "Another Ray Reconstruction plugin"). Before
// anything is patched, the plugin reads where the game's RR calls lead (the import slots it would patch, the first bytes
// of the interface's RR exports, in host-route mode the host's continuation pointers); after its handlers are in place
// it reads them again at OptiPluginInit and at the game's first call (decided once). WrongGPU RR switches itself off
// only on evidence that is positive by contract: a second copy of WrongGPU RR, or the host's plugin API refusing our
// registration (the NGX provider slot taken, our routes refused). A named foreign module on the path is inconclusive:
// RR stays active and one line names the module and the place. wronggpu-rr.ini rr_other_plugin=yield makes such a
// module switch WrongGPU RR off; take puts WrongGPU RR in front of it. rr_guard::decide holds the policy. Every read of
// memory we do not own goes through rr_guard's safe reads.
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <intrin.h>  // RR guard: _ReturnAddress (the module that calls InitializeASI is taken as the host)
#include <psapi.h>
#include <d3d12.h>  // NGX route: the device under a command list (a COM call only; no library)
#include <dxgi1_4.h>  // clean/rr-sl-diag: the adapter of a LUID in the diagnostics (system dxgi.dll loaded at run time)
#include <winternl.h>
#include <d3dkmthk.h>  // clean/rr-sm2-shim: the kernel driver's identity of an adapter (gdi32 exports, resolved at run time)

#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <wctype.h>

#include <mutex>
#include <string>
#include <vector>

#include <atomic>

#include "log_append.h"
#include "ngx_rr.h"
#include "proxy_core.h"
#include "rr_enabled.h"
#include "rr_guard.h"
#include "rr_panel.h"
#include "settings.h"
#include "setupapi_identity.h"
#include "../include/wg_host_menu_abi.h"
#include "wg_binding.inc"  // generated from the 2.14.1 table (hook variant: WG_BINDING_HOOK, kWgHookRoles, role names)

#ifndef WG_BINDING_HOOK
#error "hook_dll.cpp needs the hook binding (gen_forwarding --hook)"
#endif

#ifndef WG_RR_PROBE
#define WG_RR_PROBE 0
#endif
static_assert(WG_RR_PROBE == WG_BINDING_PROBE, "the probe hook build needs the probe binding (--probe --hook)");

#if WG_RR_PROBE
#include "probe.h"
#include "probe_gpu_d3d12.h"
#include "probe_log.h"
#endif

#pragma comment(lib, "version.lib")

namespace {

HMODULE g_self = nullptr;
HMODULE g_interposer = nullptr;   // the genuine game interface module (never patched)
// The route of this run (release 0.6.0: one build for 2.7 / 2.8 / 2.14): the game interface's minor version, set from its
// version before the core is built (do_hook / on_host_hooked); the binding table's (2.14) until then.
uint32_t g_api_minor = WG_BINDING_API_MINOR;
const uint32_t kWgRouteMinors[] = {7u, 8u, 14u};
const int kWgRoleTagForFrame = 13;  // gen_forwarding.ROLES order; absent from the 2.7 interface (A9.4)
// The interface version `major.minor` has a route in this build; *names: the families served, for the log.
bool route_for(uint32_t major, uint32_t minor, std::string* names) {
  bool ok = false;
  std::string n;
  for (uint32_t m : kWgRouteMinors) {
    if (!n.empty()) n += " / ";
    n += "2." + std::to_string(m);
    ok = ok || (major == 2u && minor == m);
  }
  if (names) *names = n;
  return ok;
}
// clean/060-rr-plugin-hunt: the words for a game interface version this build does not serve. Only the game-interface
// route is left out; the release build's NGX route (register_ngx_provider) does not depend on that version.
const char* const kNoGameInterfaceRoute = "the game-interface route is not used for this version";
#if WG_RR_PROBE
const char* const kNgxStillServed = "";
#else
const char* const kNgxStillServed = "; games that call NGX directly can still get WrongGPU RR (the NGX route)";
#endif
// role `role` exists in this run's interface family
bool role_in_route(int role) { return !(role == kWgRoleTagForFrame && g_api_minor < 8u); }
// Host routes (plugin interface 3): for a game that loads the interface at run time and resolves it by name, the host
// (OptiScaler) puts our handlers in front of the interface's exports when it hooks the module; each role's function
// to continue with is then the host's `next` pointer, not the export (calling the export would come back to us).
bool g_wait_for_host = false;     // do_hook found no interface module and no import of it
std::atomic<bool> g_route_mode{false};
WgHostMenuPlugin* g_host_plugin = nullptr;
void* g_route_next[32] = {};      // by role index (gen_forwarding.ROLES order)
HMODULE g_runtime_module = nullptr;
const WgRrRuntimeV1* g_runtime = nullptr;
wgrr::Settings g_settings;
std::once_flag g_init_once, g_settings_once, g_target_once;
std::mutex g_log_mutex;
// GetTickCount64 of the last evaluate the RR core accepted: the menu tab shows the RR box while RR runs (rr_panel.h)
std::atomic<uint64_t> g_last_rr_frame_ms{0};
// clean/rr-re9-pt: the host's command-list restore (plugin interface 5), see apply_host_restore()
std::atomic<const WgHostMenuApi*> g_host_api{nullptr};
std::atomic<WgHostMenuPlugin*> g_host_api_plugin{nullptr};
std::mutex g_host_restore_mutex;
// RR guard (rr_guard.h): the guard's state for this process, the override, and the pass-through targets
wgrr::guard::Decision g_guard;
// g_mode, g_host_module, g_host_api_module and g_fwd are written on the host's callback thread (OptiPluginInit, the
// host's hook callback) and read by the game's threads (the handlers, the late check): atomics, published with release
std::atomic<int> g_mode{0};             // wronggpu-rr.ini rr_other_plugin (wgrr::guard::Mode)
std::atomic<bool> g_hook_done{false};   // our handlers are installed: the late check may run
// The host as far as the plugin can tell: the module that called InitializeASI and the module holding the host's API
// table. Used to accept the host's own patches and for the log's wording only: a wrong guess can no longer switch RR
// off (a named module is inconclusive unless rr_other_plugin=yield).
std::atomic<HMODULE> g_host_module{nullptr};
std::atomic<HMODULE> g_host_api_module{nullptr};  // (when another module than the caller of InitializeASI)
std::atomic<bool> g_late_foreign_logged{false};   // the late check names a foreign module once per run
// g_prev is written once by guard_at_load on the thread that then builds the core (do_hook): no other thread reads it
// before the core exists (bind() runs inside the core's construction), so it stays a plain array
void* g_prev[32] = {};                  // take: what the exe's import slots held before us (the other plugin), by role
// where a passing handler sends the call (the genuine export or the host's next): an immutable snapshot, replaced whole
std::atomic<const wgrr::OriginalApi*> g_fwd{nullptr};
HANDLE g_marker = nullptr;              // the process-wide marker of the active WrongGPU RR copy (held for the run)
#if WG_RR_PROBE
wgrr::RrProbe* g_probe = nullptr;
wgrr::ProbeLog* g_probe_log = nullptr;
std::once_flag g_probe_log_once;
std::wstring g_probe_path;  // wronggpu-rr-probe.log (0.6.1: one append call per line, log_append.h)
#else
wgrr::RrProxy* g_proxy = nullptr;
// clean/rr-ngx: the NGX RR provider (plugin interface 4) and the runtime / switch both routes share
wgrr::NgxRrProvider* g_ngx = nullptr;
std::atomic<bool> g_ngx_registered{false}, g_ngx_decided{false};
std::once_flag g_runtime_once, g_enabled_once;
wgrr::RrEnabled g_enabled;
#endif

std::string narrow(const std::wstring& w) {
  int n = WideCharToMultiByte(CP_UTF8, 0, w.c_str(), (int)w.size(), nullptr, 0, nullptr, nullptr);
  std::string s(n, '\0');
  WideCharToMultiByte(CP_UTF8, 0, w.c_str(), (int)w.size(), &s[0], n, nullptr, nullptr);
  return s;
}

// Directory of the game executable (bin\x64 for the supported games): logs, ini and the runtime live here, the same
// place the file-replacement proxy used, not the plugins folder this ASI loads from.
std::wstring data_dir() {
  wchar_t buf[MAX_PATH * 2];
  DWORD n = GetModuleFileNameW(nullptr, buf, (DWORD)(sizeof buf / sizeof buf[0]));
  std::wstring p(buf, n);
  size_t slash = p.find_last_of(L"\\/");
  return slash == std::wstring::npos ? std::wstring() : p.substr(0, slash + 1);
}

std::wstring own_path() {
  wchar_t buf[MAX_PATH * 2];
  DWORD n = GetModuleFileNameW(g_self, buf, (DWORD)(sizeof buf / sizeof buf[0]));
  return std::wstring(buf, n);
}

// ---------------------------------------------------------------- logging (same format as the file proxy)
#if WG_RR_PROBE
wgrr::ProbeLog* probe_log() {
  std::call_once(g_probe_log_once, [] {
    std::wstring path = data_dir() + L"wronggpu-rr-probe.log";
    std::wstring prev = data_dir() + L"wronggpu-rr-probe.prev.log";
    {
      const wgrr::LogLock shared(path, wgrr::kLogLockWaitMs);
      MoveFileExW(path.c_str(), prev.c_str(), MOVEFILE_REPLACE_EXISTING);
      // a new, empty log (as proxy_dll.cpp does), also when the old one could not be moved aside: never append to it
      HANDLE fresh = CreateFileW(path.c_str(), GENERIC_WRITE, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
                                 nullptr, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
      if (fresh != INVALID_HANDLE_VALUE) CloseHandle(fresh);
    }
    // RR guard, 0.6.1: a second WrongGPU RR copy appends its line to this file too: every line is one append
    // call under the log's process-wide lock (log_append.h), so neither copy's line is torn or overwritten
    g_probe_path = path;
    g_probe_log = new wgrr::ProbeLog([](const std::string& line) {
      std::lock_guard<std::mutex> lock(g_log_mutex);
      const wgrr::LogLock shared(g_probe_path, wgrr::kLogLockWaitMs);
      wgrr::append_log_text(g_probe_path, wgrr::log_stamp() + line + "\r\n");  // each write reaches the file at once
    });
    g_probe_log->note(std::string("probe start: build Streamline 2.") + std::to_string(g_api_minor) +
                      " (hook mode), process " + std::to_string(GetCurrentProcessId()));
  });
  return g_probe_log;
}
#endif

// clean/rr-sl-diag: the log's path, kept for the summary at process exit (written there without the log's lock)
wchar_t g_log_path[MAX_PATH * 2] = {0};

#if !WG_RR_PROBE
// clean/060-rr-plugin-hunt: wronggpu-rr.log is appended to across game starts; past this size it is moved to
// wronggpu-rr.log.old (replacing an older one) and a new file starts, so it cannot grow without end.
constexpr uint64_t kLogCapBytes = 8ull << 20;
uint64_t g_log_rotate_blocked_below = 0;  // a rotation that failed (the file held open elsewhere): again 1 MiB later; g_log_mutex
#endif

// RR guard, 0.6.1: one line = one append call under the log's process-wide lock (log_append.h says why)
void log_line(const std::string& s) {
#if WG_RR_PROBE
  probe_log()->note(s);
#else
  std::lock_guard<std::mutex> lock(g_log_mutex);
  std::wstring path = data_dir() + L"wronggpu-rr.log";
  if (!g_log_path[0] && path.size() < sizeof g_log_path / sizeof g_log_path[0]) wcscpy_s(g_log_path, path.c_str());
  const wgrr::LogLock shared(path, wgrr::kLogLockWaitMs);
  const std::string ts = wgrr::log_stamp();
  std::string text;
  // the size as the file has it now (another copy may have written): checked and rotated only while holding the lock
  const uint64_t size = shared.owned() ? wgrr::log_file_size(path) : 0;
  if (size >= kLogCapBytes && size >= g_log_rotate_blocked_below) {
    if (MoveFileExW(path.c_str(), (path + L".old").c_str(), MOVEFILE_REPLACE_EXISTING)) {
      char b[160];
      snprintf(b, sizeof b, "log: wronggpu-rr.log reached %u MiB; the earlier lines are in wronggpu-rr.log.old\r\n",
               (unsigned)(kLogCapBytes >> 20));
      text = ts + b;
      g_log_rotate_blocked_below = 0;
    } else {
      g_log_rotate_blocked_below = size + (1u << 20);
    }
  }
  text += ts + s + "\r\n";
  wgrr::append_log_text(path, text);
#endif
}

// ---------------------------------------------------------------- attach diagnostics (recorded under the loader lock)
char g_attach_line[1200];
std::once_flag g_diag_once;

std::string ascii(const std::string& s) {
  std::string o;
  char buf[8];
  for (unsigned char c : s) {
    if (c >= 0x20 && c <= 0x7e) o.push_back((char)c);
    else { snprintf(buf, sizeof buf, "\\x%02x", c); o += buf; }
  }
  return o;
}

void record_attach(HINSTANCE inst) {
  wchar_t path[MAX_PATH * 2];
  DWORD n = GetModuleFileNameW(inst, path, MAX_PATH * 2);
  if (n >= MAX_PATH * 2) n = MAX_PATH * 2 - 1;
  path[n] = 0;
  static char p8[MAX_PATH * 8];
  if (WideCharToMultiByte(CP_UTF8, 0, path, -1, p8, (int)sizeof p8, nullptr, nullptr) <= 0) p8[0] = 0;
  snprintf(g_attach_line, sizeof g_attach_line, "load: process %lu attached %s (hook mode)",
           (unsigned long)GetCurrentProcessId(), p8);
}

// ---------------------------------------------------------------- PE helpers (import directory, version resource)
std::string module_path(HMODULE m) {
  wchar_t buf[MAX_PATH * 2];
  DWORD n = GetModuleFileNameW(m, buf, MAX_PATH * 2);
  return ascii(narrow(std::wstring(buf, n)));
}

const char* module_leaf(HMODULE m, std::wstring* out) {
  wchar_t buf[MAX_PATH * 2];
  DWORD n = GetModuleFileNameW(m, buf, MAX_PATH * 2);
  std::wstring p(buf, n);
  size_t slash = p.find_last_of(L"\\/");
  *out = slash == std::wstring::npos ? p : p.substr(slash + 1);
  return nullptr;
}

// The file version (FILEVERSION, HIWORD/LOWORD parts) of a module from its on-disk version resource.
bool module_file_version(HMODULE m, unsigned v[4]) {
  wchar_t path[MAX_PATH * 2];
  DWORD pn = GetModuleFileNameW(m, path, MAX_PATH * 2);
  if (!pn) return false;
  DWORD ignored = 0;
  DWORD size = GetFileVersionInfoSizeW(path, &ignored);
  if (!size) return false;
  std::vector<uint8_t> blk(size);
  if (!GetFileVersionInfoW(path, 0, size, blk.data())) return false;
  VS_FIXEDFILEINFO* f = nullptr;
  UINT n = 0;
  if (!VerQueryValueW(blk.data(), L"\\", (void**)&f, &n) || !f || n < sizeof(VS_FIXEDFILEINFO)) return false;
  if (f->dwSignature != 0xFEEF04BDu) return false;
  v[0] = HIWORD(f->dwFileVersionMS);
  v[1] = LOWORD(f->dwFileVersionMS);
  v[2] = HIWORD(f->dwFileVersionLS);
  v[3] = LOWORD(f->dwFileVersionLS);
  return true;
}

// ---------------------------------------------------------------- the core (reused unchanged)
uint32_t status_word();
void ensure_settings();
void apply_host_restore();

// clean/rr-re9-pt: per-frame copies keyed by the token object instead of its frame index (control runs only)
bool frame_by_token() {
  char ev[8] = {};
  if (GetEnvironmentVariableA("WG_RR_FRAME_BY_TOKEN", ev, sizeof ev)) return ev[0] == '1';
  return g_settings.text("rr_frame_by_token") == "1";
}

template <typename F>
void bind(F& slot, int role) {
  const char* name = role_in_route(role) ? kWgRoleNames[role] : "";
  if (g_route_mode.load() && g_route_next[role]) {
    slot = reinterpret_cast<F>(g_route_next[role]);  // host route: continue through the host's chain
    return;
  }
  if (g_guard.state() == wgrr::guard::State::kTaken && g_prev[role]) {  // RR guard, take: continue through the
                                                                         // other module (it sees no RR work)
    slot = reinterpret_cast<F>(g_prev[role]);
    return;
  }
  slot = name[0] ? reinterpret_cast<F>(GetProcAddress(g_interposer, name)) : nullptr;
}

extern "C" WgResultCode wg_rr_optimal_fn(const WgRrOptions* o, WgRrOptimal* s);
extern "C" WgResultCode wg_rr_state_fn(const WgViewport* v, WgRrState* s);
extern "C" WgResultCode wg_rr_options_fn(const WgViewport* v, const WgRrOptions* o);

// The forward targets of the core: the genuine interface exports, i.e. the addresses the IAT held before we patched
// (OptiScaler's inline patch on the export, if any, is reached through them).
wgrr::OriginalApi bind_original() {
  wgrr::OriginalApi o;
  bind(o.init, 0); bind(o.shutdown, 1); bind(o.is_supported, 2); bind(o.is_loaded, 3); bind(o.set_loaded, 4);
  bind(o.requirements, 5); bind(o.version, 6); bind(o.get_function, 7); bind(o.allocate, 8);
  bind(o.free_resources, 9); bind(o.set_device, 10); bind(o.set_vulkan_info, 11); bind(o.tag, 12);
  bind(o.tag_for_frame, 13); bind(o.constants, 14); bind(o.new_frame_token, 15); bind(o.evaluate, 16);
  bind(o.upgrade_interface, 17); bind(o.native_interface, 18);
  return o;
}

#if WG_RR_PROBE
wgrr::RrProbe* target() {
  std::call_once(g_target_once, [] {
    ensure_settings();
    probe_log();
    wgrr::ProbeConfig cfg;
    cfg.api_minor = g_api_minor;  // this run's route (release 0.6.0: one build)
    memcpy(cfg.ids.viewport.b, kWgGuid_viewport, 16);
    memcpy(cfg.ids.tag.b, kWgGuid_tag, 16);
    memcpy(cfg.ids.tag_conversion.b, kWgGuid_tag_conversion, 16);
    cfg.ids.has_tag_conversion = WG_BINDING_HAS_TAG_CONVERSION != 0;
    memcpy(cfg.frame_token.b, kWgGuid_frame_token, 16);
    cfg.has_frame_token = WG_BINDING_HAS_FRAME_TOKEN_GUID != 0;
    for (int i = 0; i < 3; ++i) cfg.release_version[i] = kWgReleaseVersion[i];
    cfg.fn_optimal = kWgFnOptimal;
    cfg.fn_state = kWgFnState;
    cfg.fn_options = kWgFnOptions;
    cfg.passes_need_restore_bit = true;  // hook mode: bind own list state only when the game restores its own (bit 0)
    for (const WgGuidName& g : kWgGuidNames) {
      WgGuid id;
      memcpy(id.b, g.b, 16);
      cfg.guid_names.push_back(std::make_pair(std::string(g.name), id));
    }
    cfg.frame_by_token = frame_by_token();
    log_line(cfg.frame_by_token ? "hook: per-frame copies keyed by the frame token object (rr_frame_by_token=1)"
                                : "hook: per-frame copies keyed by the frame token's index");
    g_probe = new wgrr::RrProbe(cfg, bind_original(), wgrr::create_d3d12_probe_gpu(), probe_log(),
                                [] { return (uint32_t)GetCurrentThreadId(); });
    g_probe->set_clock([] { return (uint64_t)GetTickCount64(); });
    g_probe->set_feature_function_pointers((void*)&wg_rr_optimal_fn, (void*)&wg_rr_state_fn,
                                           (void*)&wg_rr_options_fn);
    apply_host_restore();
  });
  return g_probe;
}
uint32_t status_word() { return g_probe ? g_probe->status_word() : 0u; }

// clean/rr-re9-pt: the probe's own passes (upscale, normal patches) bind a heap and a compute root signature on the
// game's list like the runtime does, so they get the same host restore (plugin interface 5) when the host has it.
void apply_host_restore() {
  wgrr::RrProbe* probe = g_probe;
  if (!probe) return;
  std::lock_guard<std::mutex> lock(g_host_restore_mutex);
  static int logged = -1;
  const WgHostMenuApi* api = g_host_api.load();
  WgHostMenuPlugin* plugin = g_host_api_plugin.load();
  const bool host_has = api && plugin && WG_HOST_MENU_HAS(api, begin_cmd_list_work) &&
                        WG_HOST_MENU_HAS(api, end_cmd_list_work);
  if (host_has) {
    wgrr::CmdListBracket b;
    b.begin = [api, plugin](void* cl) { api->begin_cmd_list_work(plugin, cl); };
    b.end = [api, plugin](void* cl) { api->end_cmd_list_work(plugin, cl); };
    probe->set_cmd_list_bracket(std::move(b));
  } else {
    probe->set_cmd_list_bracket(wgrr::CmdListBracket());
  }
  if ((int)host_has != logged) {
    logged = (int)host_has;
    log_line(host_has ? "hook: host restores the game's command-list state after the probe's passes (plugin interface 5)"
                      : "hook: the host has no command-list restore (plugin interface 5)");
  }
}
#else
// The player's switch (rr_enabled, rr_enabled.h), read once for both routes.
const wgrr::RrEnabled& rr_enabled() {
  std::call_once(g_enabled_once, [] {
    g_enabled = wgrr::read_rr_enabled(wgss::data_dir() + L"wronggpu.ini", data_dir() + L"wronggpu-rr.ini");
  });
  return g_enabled;
}

// The RR runtime of this process, loaded once for both routes (the game-interface core and the NGX provider): rr_mode
// (as the menu box resolves it, below), rr_model and diag_env_N from wronggpu-rr.ini, the player's switch. g_runtime stays null when RR is switched off or the
// runtime is missing / unusable.
void load_runtime() {
  std::call_once(g_runtime_once, [] {
    ensure_settings();
    // rr_mode (exact | accurate | fast, registry clean/wgshared). ONE runtime library holds the kernels of all three modes
    // and picks them at instance creation:
    //   exact:    everything exact (head, input, history / output / resolve and the body's arithmetic);
    //   accurate: relaxed head (gate-checked; owner decision 2026-11-01, confirmed 2026-11-04), exact input (history
    //             samples), exact output / resolve logic, relaxed body arithmetic;
    //   fast:     relaxed head, lite input, hardware reciprocal in the output stage, max-reprojection output.
    // clean/060-rr-plugin-hunt: resolved as the menu box resolves it (rr_panel.h resolved_mode: the shared settings block,
    // WG_RR_MODE, WrongGPU\wronggpu.ini where the launcher writes it, the registry default). wronggpu-rr.ini's rr_mode
    // (the place before 0.6.0) is a legacy fallback, used only when none of those sets the mode. The runtime reads the
    // block first, then WG_RR_MODE, never wronggpu.ini or wronggpu-rr.ini, so the mode is handed to it as WG_RR_MODE
    // (unless that variable is already set) only where it could not find it itself: a legacy wronggpu-rr.ini value, a
    // wronggpu.ini value while the block is unavailable, and an exact turned into accurate.
    // rr_model (optional): the model path handed to the runtime (WG_RR_MODEL) unless that variable is already set.
    const wgrr::panel::ModeSetting resolved = wgrr::panel::resolved_mode();
    int32_t mode_id = resolved.mode;
    std::string mode_source = resolved.source;
    bool hand_over = resolved.set && !resolved.shared;
    {
      const std::string legacy = wgss::lower(wgss::trim(g_settings.text("rr_mode")));
      int32_t v = 0;
      if (legacy.empty()) {
      } else if (!wgss::parse_value(kWgOptions[WG_OPT_RR_MODE], legacy, &v)) {
        log_line("hook: rr_mode=" + legacy + " in wronggpu-rr.ini not recognised; ignored");
      } else if (!resolved.set) {
        mode_id = v;
        mode_source = "legacy wronggpu-rr.ini";
        hand_over = true;
      } else if (v != mode_id) {
        log_line("hook: the legacy rr_mode line of wronggpu-rr.ini (" + legacy + ") is not used; " + mode_source +
                 " sets the mode");
      }
    }
    if (mode_id == WG_RR_MODE_EXACT && !wgrr::panel::exact_allowed()) {  // exact only when the launcher allowed it
      log_line("hook: rr_mode=exact (" + mode_source + ") without rr_exact_allowed=1; using accurate");
      mode_id = WG_RR_MODE_ACCURATE;
      hand_over = true;
    }
    const char* mode = kWgRrModeNames[mode_id];
    std::wstring rt_path = data_dir() + WG_RR_RUNTIME_FILE;
    char mv0[16];
    const bool env_mode = GetEnvironmentVariableA("WG_RR_MODE", mv0, sizeof mv0) != 0;
    hand_over = hand_over && !env_mode && SetEnvironmentVariableA("WG_RR_MODE", mode);
    const std::string model = g_settings.text("rr_model");
    wchar_t mv[8];
    if (!model.empty() && !GetEnvironmentVariableW(L"WG_RR_MODEL", mv, 8)) {
      const int n = MultiByteToWideChar(CP_UTF8, 0, model.c_str(), -1, nullptr, 0);
      std::wstring w((size_t)(n > 0 ? n : 1), L'\0');
      MultiByteToWideChar(CP_UTF8, 0, model.c_str(), -1, &w[0], n);
      SetEnvironmentVariableW(L"WG_RR_MODEL", w.c_str());
    }
    // diag_env_1 .. diag_env_4 (optional, diagnostics): "WG_RR_<NAME>=<value>" handed to the runtime as an environment
    // variable (its start-up knobs and WG_RR_DIAG_* switches) without setting it system-wide. Only WG_RR_ names; a
    // variable already set in the environment wins. Not set = nothing changes.
    for (int i = 1; i <= 4; ++i) {
      const std::string d = g_settings.text(("diag_env_" + std::to_string(i)).c_str());
      const size_t eq = d.find('=');
      if (d.empty()) continue;
      if (eq == std::string::npos || eq <= 6 || d.compare(0, 6, "WG_RR_") != 0) {
        log_line("hook: diag_env_" + std::to_string(i) + "=" + d + " ignored (needs WG_RR_<NAME>=<value>)");
        continue;
      }
      const std::string name = d.substr(0, eq), val = d.substr(eq + 1);
      char cur[8];
      if (GetEnvironmentVariableA(name.c_str(), cur, sizeof cur)) {
        log_line("hook: diag_env " + name + " already set in the environment; kept");
      } else if (SetEnvironmentVariableA(name.c_str(), val.c_str())) {
        log_line("hook: diag_env " + name + "=" + val);
      }
    }
    {
      char iv[16] = {};
      GetEnvironmentVariableA("WG_RR_INPUT", iv, sizeof iv);
      log_line(std::string("hook: rr_mode ") + mode + " (" + mode_source +
               (resolved.shared ? "" : ", shared settings unavailable") +
               (hand_over ? ", handed to the runtime as WG_RR_MODE" : "") + "; runtime " + narrow(rt_path) +
               ", WG_RR_INPUT=" + (iv[0] ? iv : "<by mode>") + ", model " + (model.empty() ? "<runtime default>" : model) +
               ")");
    }
    // The player's switch (rr_enabled, rr_enabled.h): off = WrongGPU does not take RR. The runtime is not loaded then
    // (no GPU memory, no HIP start-up); non-RR traffic and the menu box are unchanged.
    const wgrr::RrEnabled& enabled = rr_enabled();
    if (!enabled.on) {
      log_line("hook: rr_enabled=0 (" + enabled.source + "): WrongGPU does not take RR; RR is reported unsupported, so "
               "the game uses its own path. The RR runtime is not loaded.");
    } else {
      g_runtime_module = LoadLibraryW(rt_path.c_str());
    }
    if (g_runtime_module) {
      auto entry = reinterpret_cast<WgRrRuntimeEntryFn>(GetProcAddress(g_runtime_module, WG_RR_RUNTIME_EXPORT));
      const WgRrRuntimeV1* rt = entry ? entry() : nullptr;
      if (rt && rt->version == WG_RR_ADAPTER_VERSION && rt->size >= WG_RR_RUNTIME_V1_BASE_SIZE) g_runtime = rt;
      else log_line("hook: WrongGPU RR runtime has no usable adapter; RR reports unsupported");
    } else if (enabled.on) {
      log_line("hook: WrongGPU RR runtime not found; RR reports unsupported: " + narrow(rt_path));
    }
  });
}

// clean/rr-sl-diag: an adapter LUID in words for the diagnostics ("vendor 0x1002 device 0x7550, <name>"), from the system's
// own dxgi.dll (by its full path, so that no DXGI wrapper in the game folder is called); cached per LUID
std::string describe_luid(const void* luid, uint32_t size) {
  if (!luid || size != sizeof(LUID)) return "";
  static std::mutex mu;
  static std::map<uint64_t, std::string> seen;
  uint64_t key = 0;
  memcpy(&key, luid, sizeof key);
  std::lock_guard<std::mutex> lock(mu);
  auto it = seen.find(key);
  if (it != seen.end()) return it->second;
  std::string out;
  wchar_t sys[MAX_PATH];
  const UINT n = GetSystemDirectoryW(sys, MAX_PATH);
  HMODULE dx = n && n < MAX_PATH - 12 ? LoadLibraryW((std::wstring(sys, n) + L"\\dxgi.dll").c_str()) : nullptr;
  using CreateFactoryFn = HRESULT(WINAPI*)(REFIID, void**);
  auto create = dx ? reinterpret_cast<CreateFactoryFn>(GetProcAddress(dx, "CreateDXGIFactory1")) : nullptr;
  IDXGIFactory4* f = nullptr;
  if (create && SUCCEEDED(create(__uuidof(IDXGIFactory4), reinterpret_cast<void**>(&f))) && f) {
    LUID l;
    memcpy(&l, luid, sizeof l);
    IDXGIAdapter1* a = nullptr;
    DXGI_ADAPTER_DESC1 d;
    if (SUCCEEDED(f->EnumAdapterByLuid(l, __uuidof(IDXGIAdapter1), reinterpret_cast<void**>(&a))) && a &&
        SUCCEEDED(a->GetDesc1(&d))) {
      char b[200];
      snprintf(b, sizeof b, "vendor 0x%04x device 0x%04x, %s", d.VendorId, d.DeviceId, narrow(d.Description).c_str());
      out = b;
    } else {
      out = "no adapter of this LUID found by DXGI";
    }
    if (a) a->Release();
    f->Release();
  }
  if (dx) FreeLibrary(dx);
  seen[key] = out;
  return out;
}

wgrr::RrProxy* target() {
  std::call_once(g_target_once, [] {
    ensure_settings();
    load_runtime();
    const wgrr::RrEnabled& enabled = rr_enabled();
    wgrr::ProxyConfig cfg;
    cfg.api_minor = g_api_minor;  // this run's route (release 0.6.0: one build)
    memcpy(cfg.ids.viewport.b, kWgGuid_viewport, 16);
    memcpy(cfg.ids.tag.b, kWgGuid_tag, 16);
    memcpy(cfg.ids.tag_conversion.b, kWgGuid_tag_conversion, 16);
    cfg.ids.has_tag_conversion = WG_BINDING_HAS_TAG_CONVERSION != 0;
    cfg.version_magic = WG_BINDING_VERSION_MAGIC;
    for (int i = 0; i < 3; ++i) cfg.release_version[i] = kWgReleaseVersion[i];
    cfg.fn_optimal = kWgFnOptimal;
    cfg.fn_state = kWgFnState;
    cfg.fn_options = kWgFnOptions;
    cfg.frame_by_token = frame_by_token();
    log_line(cfg.frame_by_token ? "hook: per-frame copies keyed by the frame token object (rr_frame_by_token=1)"
                                : "hook: per-frame copies keyed by the frame token's index");
    g_proxy = new wgrr::RrProxy(cfg, bind_original(), g_runtime, &g_settings, log_line,
                                [] { return (uint64_t)GetTickCount64(); });
    g_proxy->set_feature_function_pointers((void*)&wg_rr_optimal_fn, (void*)&wg_rr_state_fn,
                                           (void*)&wg_rr_options_fn);
    g_proxy->set_adapter_describer(describe_luid);  // clean/rr-sl-diag
    if (!enabled.on) g_proxy->set_rr_off(true, wgrr::rr_off_reason(enabled));
    apply_host_restore();
  });
  return g_proxy;
}
uint32_t status_word() { return g_proxy ? g_proxy->status_word() : 0u; }

// clean/rr-re9-pt: our runtime binds its own descriptor heap, compute root signature and root arguments on the game's
// list. RE Engine games (RE Requiem) do not rebind their compute root signature after the call, so their next dispatch
// runs against our layout and the GPU hangs (games/re9/rgd/DIAGNOSIS.txt). OptiScaler plugin interface 5 brackets our
// work and puts the game's root state back. Off switch (control runs): rr_host_restore=0 in wronggpu-rr.ini or
// WG_RR_HOST_RESTORE=0. Without the host's restore, RR is reported unsupported for games known to need it, so the game
// keeps its RR options hidden instead of hanging.
bool exe_needs_host_restore() {
  wchar_t buf[MAX_PATH * 2];
  DWORD n = GetModuleFileNameW(nullptr, buf, (DWORD)(sizeof buf / sizeof buf[0]));
  std::wstring p(buf, n);
  size_t slash = p.find_last_of(L"\\/");
  std::wstring exe = slash == std::wstring::npos ? p : p.substr(slash + 1);
  for (wchar_t& c : exe) c = (wchar_t)towlower(c);
  return exe == L"re9.exe" || exe == L"re9demo.exe";
}

bool host_restore_switched_off() {
  char ev[8] = {};
  if (GetEnvironmentVariableA("WG_RR_HOST_RESTORE", ev, sizeof ev)) return ev[0] == '0';
  return g_settings.text("rr_host_restore") == "0";
}

void apply_host_restore() {
  wgrr::RrProxy* proxy = g_proxy;
  if (!proxy) return;
  std::lock_guard<std::mutex> lock(g_host_restore_mutex);
  static int logged = -1;  // the last state logged (0 off, 1 host restores, 2 no host restore)
  const WgHostMenuApi* api = g_host_api.load();
  WgHostMenuPlugin* plugin = g_host_api_plugin.load();
  const bool host_has = api && plugin && WG_HOST_MENU_HAS(api, begin_cmd_list_work) &&
                        WG_HOST_MENU_HAS(api, end_cmd_list_work);
  const bool off = host_restore_switched_off();
  int state = off ? 0 : host_has ? 1 : 2;
  if (state == 1) {
    wgrr::CmdListBracket b;
    b.begin = [api, plugin](void* cl) { api->begin_cmd_list_work(plugin, cl); };
    b.end = [api, plugin](void* cl) { api->end_cmd_list_work(plugin, cl); };
    proxy->set_cmd_list_bracket(std::move(b));
  } else {
    proxy->set_cmd_list_bracket(wgrr::CmdListBracket());
  }
  const bool refuse = state == 2 && exe_needs_host_restore();
  proxy->set_rr_refused(refuse, "this game needs the host to restore its command-list state after RR work and the "
                                "host has no restore (OptiScaler plugin interface 5)");
  if (state != logged) {
    logged = state;
    log_line(state == 1 ? "hook: host restores the game's command-list state after RR work (plugin interface 5)"
             : state == 0 ? "hook: host restore switched off (rr_host_restore=0 / WG_RR_HOST_RESTORE=0); RR leaves its "
                            "command-list state on the game's list"
                          : std::string("hook: the host has no command-list restore (plugin interface 5)") +
                                (refuse ? "; RR reported unsupported for this game" : ""));
  }
}
#endif

void ensure_settings() {
  std::call_once(g_settings_once, [] {
    g_settings.load(narrow(data_dir() + L"wronggpu-rr.ini"), log_line);
    wgrr::bind_contract(&g_settings, [] { return status_word(); });
#if !WG_RR_PROBE
    wgrr::panel::bind_settings(&g_settings);
#endif
  });
}

bool g_diagnostics_done = false;
void diagnostics(const char* first_call) {
  std::call_once(g_diag_once, [first_call] {
    log_line(ascii(g_attach_line));
    log_line(std::string("load: first call ") + first_call);
    g_diagnostics_done = true;
  });
}

// clean/rr-sl-diag 2: calls that reached each of our game-interface handlers (by role index; the RR feature functions
// after the roles), for the summary at process exit (lock-free)
constexpr int kDiagRoleSlots = 24;
std::atomic<uint64_t> g_role_calls[kDiagRoleSlots + 3];
std::atomic<uint64_t> g_ngx_creates{0}, g_ngx_created{0}, g_ngx_evaluates{0};
void slot_check_now(const char* why);  // below (the slot watch)

auto entered(const char* call) -> decltype(target()) {
  diagnostics(call);
  if (!strcmp(call, "rr-optimal")) ++g_role_calls[kDiagRoleSlots];
  else if (!strcmp(call, "rr-state")) ++g_role_calls[kDiagRoleSlots + 1];
  else if (!strcmp(call, "rr-options")) ++g_role_calls[kDiagRoleSlots + 2];
  return target();
}
auto entered(int role) -> decltype(target()) {
  if (role >= 0 && role < kDiagRoleSlots) ++g_role_calls[role];
  return entered(kWgRoleNames[role]);
}

// clean/rr-ngx: one route per game. While the NGX route holds an RR feature, the game-interface route refuses RR work
// (38), so a game can never get two evaluates per frame; logged once.
bool ngx_route_holds_rr(uint32_t feature) {
#if WG_RR_PROBE
  (void)feature;
  return false;
#else
  if (feature != WG_FEAT_RAY_RECONSTRUCTION || !g_ngx || g_ngx->live() == 0) return false;
  static std::atomic<bool> logged{false};
  if (!logged.exchange(true))
    log_line("hook: the NGX route holds WrongGPU RR in this game; game-interface RR work refused (38), one route per "
             "game");
  return true;
#endif
}

// RR guard: the late decision (below, with the IAT code), taken once at the first call after our handlers
// are installed; a handler passes the call on unchanged (to g_fwd) while WrongGPU RR is not active.
void guard_first_call();
bool guard_pass() {
  if (!g_guard.frozen() && g_hook_done.load(std::memory_order_acquire)) guard_first_call();
  return !g_guard.active();
}
// publishes the pass-through targets (an immutable snapshot; a replaced one is leaked: a handful per run at most)
void set_fwd(const wgrr::OriginalApi& o) { g_fwd.store(new wgrr::OriginalApi(o), std::memory_order_release); }
#define WG_PASS(field, ...)                                                                  \
  if (guard_pass()) {                                                                        \
    const wgrr::OriginalApi* fwd_ = g_fwd.load(std::memory_order_acquire);                   \
    return fwd_ && fwd_->field ? fwd_->field(__VA_ARGS__) : WG_E_MISSING_OR_INVALID_API;     \
  }                                                                                          \
  (void)0

}  // namespace

// ---------------------------------------------------------------- the handlers installed into the IAT
extern "C" {

WgResultCode wg_role_init(const WgPreferences* p, uint64_t v) {
  WG_PASS(init, p, v);
  auto* t = entered(0);
  char buf[200];
  snprintf(buf, sizeof buf, "init entered: thread %lu preferences=0x%llx sdk_version=0x%016llx (hook mode)",
           (unsigned long)GetCurrentThreadId(), (unsigned long long)(uintptr_t)p, (unsigned long long)v);
  log_line(buf);
  return t->init(p, v);
}
WgResultCode wg_role_shutdown(void) {
  WG_PASS(shutdown);
  return entered(1)->shutdown();
}
WgResultCode wg_role_is_supported(uint32_t f, const WgAdapterInfo* a) {
  WG_PASS(is_supported, f, a);
  return entered(2)->is_supported(f, a);
}
WgResultCode wg_role_is_loaded(uint32_t f, bool* l) {
  WG_PASS(is_loaded, f, l);
  return entered(3)->is_loaded(f, l);
}
WgResultCode wg_role_set_loaded(uint32_t f, bool l) {
  WG_PASS(set_loaded, f, l);
  return entered(4)->set_loaded(f, l);
}
WgResultCode wg_role_requirements(uint32_t f, WgRequirements* r) {
  WG_PASS(requirements, f, r);
  return entered(5)->requirements(f, r);
}
WgResultCode wg_role_version(uint32_t f, WgFeatureVersion* v) {
  WG_PASS(version, f, v);
  return entered(6)->version(f, v);
}
WgResultCode wg_role_get_function(uint32_t f, const char* n, void** o) {
  WG_PASS(get_function, f, n, o);
  return entered(7)->get_function(f, n, o);
}
WgResultCode wg_role_allocate(void* c, uint32_t f, const WgViewport* v) {
  WG_PASS(allocate, c, f, v);
  if (ngx_route_holds_rr(f)) return WG_E_INVALID_STATE;
  return entered(8)->allocate(c, f, v);
}
WgResultCode wg_role_free(uint32_t f, const WgViewport* v) {
  WG_PASS(free_resources, f, v);
  return entered(9)->free_resources(f, v);
}
WgResultCode wg_role_set_device(void* d) {
  WG_PASS(set_device, d);
  return entered(10)->set_device(d);
}
WgResultCode wg_role_tag(const WgViewport* v, const WgTag* t, uint32_t n, void* c) {
  WG_PASS(tag, v, t, n, c);
  return entered(12)->tag(v, t, n, c);
}
WgResultCode wg_role_tag_for_frame(const WgFrameToken* k, const WgViewport* v, const WgTag* t, uint32_t n, void* c) {
  WG_PASS(tag_for_frame, k, v, t, n, c);
  return entered(13)->tag_for_frame(k, v, t, n, c);
}
WgResultCode wg_role_constants(const WgConstants* c, const WgFrameToken* k, const WgViewport* v) {
  WG_PASS(constants, c, k, v);
  return entered(14)->constants(c, k, v);
}
WgResultCode wg_role_new_frame_token(WgFrameToken** o, const uint32_t* i) {
  WG_PASS(new_frame_token, o, i);
  return entered(15)->new_frame_token(o, i);
}
WgResultCode wg_role_evaluate(uint32_t f, const WgFrameToken* k, const WgHeader* const* in, uint32_t n, void* c) {
  WG_PASS(evaluate, f, k, in, n, c);
  if (ngx_route_holds_rr(f)) return WG_E_INVALID_STATE;
  const WgResultCode rc = entered(16)->evaluate(f, k, in, n, c);
  if (rc == WG_OK && f == WG_FEAT_RAY_RECONSTRUCTION) {
    g_last_rr_frame_ms.store(GetTickCount64(), std::memory_order_relaxed);
#if !WG_RR_PROBE
    wgrr::panel::frame_done();  // the RR name and GPU time in OptiScaler's overlay
#endif
  }
  return rc;
}
#if WG_RR_PROBE
WgResultCode wg_role_set_vulkan_info(const void* info) {
  WG_PASS(set_vulkan_info, info);
  return entered(11)->set_vulkan_info(info);
}
WgResultCode wg_role_upgrade_interface(void** iface) {
  WG_PASS(upgrade_interface, iface);
  return entered(17)->upgrade_interface(iface);
}
WgResultCode wg_role_native_interface(void* w, void** n) {
  WG_PASS(native_interface, w, n);
  return entered(18)->native_interface(w, n);
}
#endif

WgResultCode wg_rr_optimal_fn(const WgRrOptions* o, WgRrOptimal* s) { return entered("rr-optimal")->rr_optimal(o, s); }
WgResultCode wg_rr_state_fn(const WgViewport* v, WgRrState* s) { return entered("rr-state")->rr_state(v, s); }
WgResultCode wg_rr_options_fn(const WgViewport* v, const WgRrOptions* o) { return entered("rr-options")->rr_options(v, o); }

const WgRrSettingsContract* wg_get_settings_contract(void) {
  // RR guard: a second copy keeps its hands off the log (the active copy writes it)
  if (g_guard.state() != wgrr::guard::State::kOwnCopy) diagnostics(WG_RR_CONTRACT_EXPORT);
  ensure_settings();
  return wgrr::settings_contract();
}

}  // extern "C"

// ---------------------------------------------------------------- IAT redirection
namespace {

// Handler for a role index (gen_forwarding.ROLES order), or nullptr when the role is not hooked in this build.
void* handler_for(int role) {
  switch (role) {
    case 0: return (void*)&wg_role_init;
    case 1: return (void*)&wg_role_shutdown;
    case 2: return (void*)&wg_role_is_supported;
    case 3: return (void*)&wg_role_is_loaded;
    case 4: return (void*)&wg_role_set_loaded;
    case 5: return (void*)&wg_role_requirements;
    case 6: return (void*)&wg_role_version;
    case 7: return (void*)&wg_role_get_function;
    case 8: return (void*)&wg_role_allocate;
    case 9: return (void*)&wg_role_free;
    case 10: return (void*)&wg_role_set_device;
    case 12: return (void*)&wg_role_tag;
    case 13: return (void*)&wg_role_tag_for_frame;
    case 14: return (void*)&wg_role_constants;
    case 15: return (void*)&wg_role_new_frame_token;
    case 16: return (void*)&wg_role_evaluate;
#if WG_RR_PROBE
    case 11: return (void*)&wg_role_set_vulkan_info;
    case 17: return (void*)&wg_role_upgrade_interface;
    case 18: return (void*)&wg_role_native_interface;
#endif
    default: return nullptr;
  }
}

// name of a hooked role (one of kWgHookRoles) -> handler, or nullptr if `name` is not a hooked role.
void* handler_for_name(const char* name) {
  for (unsigned i = 0; i < WG_BINDING_HOOK_ROLE_COUNT; ++i) {
    const int role = kWgHookRoles[i];
    if (role_in_route(role) && kWgRoleNames[role][0] && strcmp(kWgRoleNames[role], name) == 0) return handler_for(role);
  }
  return nullptr;
}

// Overwrite one IAT slot with `value`, atomically, restoring the original page protection.
bool patch_slot(void** slot, void* value) {
  DWORD old = 0;
  if (!VirtualProtect(slot, sizeof(void*), PAGE_READWRITE, &old)) return false;
  InterlockedExchangePointer(slot, value);
  DWORD ignored = 0;
  VirtualProtect(slot, sizeof(void*), old, &ignored);
  return true;
}

struct PatchRecord { void** slot; void* original; };
std::vector<PatchRecord> g_patches;  // for an optional restore (the ASI does not unload; kept for the test)

// clean/rr-sl-diag 2: the slot watch. Every IAT slot we patched, and every other delay-load slot of the game interface's
// descriptor (left to the delay helper), is read again later (slot_check_now): a patched slot that no longer holds our
// handler was rewritten (the delay helper or another module), an unpatched delay-load slot that left its thunk was bound
// by the helper (the game called that function). Read only: nothing is patched again.
struct SlotWatch {
  void** slot;
  std::string name, module, kind;  // kind: "import" / "delay-load"
  void* ours;                      // our handler (null: a slot we left alone)
  void* original;                  // what the slot held when we read it first
  void* last;                      // what the last check saw
};
std::vector<SlotWatch> g_watch;   // filled while hooking (one thread), then read by the checks
std::mutex g_watch_mutex;
unsigned g_watch_checks = 0;

// "<module file>+0x<offset>", or the bare address when no module holds it
std::string where(const void* p) {
  if (!p) return "null";
  HMODULE m = nullptr;
  char b[64];
  if (GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                         static_cast<LPCWSTR>(p), &m) && m) {
    wchar_t name[MAX_PATH];
    const DWORD n = GetModuleFileNameW(m, name, MAX_PATH);
    std::wstring w(name, n);
    const size_t slash = w.find_last_of(L"\\/");
    snprintf(b, sizeof b, "+0x%llx", (unsigned long long)(static_cast<const uint8_t*>(p) - reinterpret_cast<const uint8_t*>(m)));
    return narrow(slash == std::wstring::npos ? w : w.substr(slash + 1)) + b;
  }
  snprintf(b, sizeof b, "0x%llx (no module)", (unsigned long long)(uintptr_t)p);
  return b;
}

// Patch, in `module`, every IAT slot that imports a hooked role name from the interposer DLL. Returns the number of
// slots patched. Non-RR imports and imports from other DLLs are left untouched.
unsigned patch_module_imports(HMODULE module, const char* interposer_leaf) {
  const uint8_t* base = reinterpret_cast<const uint8_t*>(module);
  const IMAGE_DOS_HEADER* dos = reinterpret_cast<const IMAGE_DOS_HEADER*>(base);
  if (dos->e_magic != IMAGE_DOS_SIGNATURE) return 0;
  const IMAGE_NT_HEADERS64* nt = reinterpret_cast<const IMAGE_NT_HEADERS64*>(base + dos->e_lfanew);
  if (nt->Signature != IMAGE_NT_SIGNATURE) return 0;
  const IMAGE_DATA_DIRECTORY& d = nt->OptionalHeader.DataDirectory[IMAGE_DIRECTORY_ENTRY_IMPORT];
  if (!d.VirtualAddress || !d.Size) return 0;
  unsigned patched = 0;
  for (const IMAGE_IMPORT_DESCRIPTOR* imp = reinterpret_cast<const IMAGE_IMPORT_DESCRIPTOR*>(base + d.VirtualAddress);
       imp->Name; ++imp) {
    const char* dll = reinterpret_cast<const char*>(base + imp->Name);
    if (_stricmp(dll, interposer_leaf) != 0) continue;  // only the game's interface DLL
    // OriginalFirstThunk names the imports; FirstThunk is the IAT we patch. Fall back to FirstThunk for names when
    // the import was already bound in place (no separate name table).
    const IMAGE_THUNK_DATA64* names = reinterpret_cast<const IMAGE_THUNK_DATA64*>(
        base + (imp->OriginalFirstThunk ? imp->OriginalFirstThunk : imp->FirstThunk));
    IMAGE_THUNK_DATA64* iat = reinterpret_cast<IMAGE_THUNK_DATA64*>(const_cast<uint8_t*>(base) + imp->FirstThunk);
    for (; names->u1.AddressOfData; ++names, ++iat) {
      if (names->u1.Ordinal & IMAGE_ORDINAL_FLAG64) continue;  // the games import by name only (A9.2)
      const IMAGE_IMPORT_BY_NAME* by = reinterpret_cast<const IMAGE_IMPORT_BY_NAME*>(base + names->u1.AddressOfData);
      void* h = handler_for_name(reinterpret_cast<const char*>(by->Name));
      if (!h) continue;  // non-RR import: leave it pointing at the genuine interface
      void** slot = reinterpret_cast<void**>(&iat->u1.Function);
      void* original = *slot;
      if (original == h) continue;
      if (patch_slot(slot, h)) {
        g_patches.push_back({slot, original});
        g_watch.push_back({slot, reinterpret_cast<const char*>(by->Name), module_path(module), "import", h, original, h});
        ++patched;
        log_line(std::string("hook: patched IAT slot ") + reinterpret_cast<const char*>(by->Name) + " in " +
                 module_path(module));
      } else {
        log_line(std::string("hook: FAILED to patch IAT slot ") + reinterpret_cast<const char*>(by->Name) + " in " +
                 module_path(module));
      }
    }
  }
  return patched;
}

// The delay-load descriptor in `module` that names the interposer DLL, or nullptr. Delay-load imports are bound on the
// first call through them (the linker's delay helper loads the DLL then), so a game that links the interface this way
// has not loaded it yet when the plugin runs.
const IMAGE_DELAYLOAD_DESCRIPTOR* find_delay_import(HMODULE module, const char* interposer_leaf) {
  const uint8_t* base = reinterpret_cast<const uint8_t*>(module);
  const IMAGE_DOS_HEADER* dos = reinterpret_cast<const IMAGE_DOS_HEADER*>(base);
  if (dos->e_magic != IMAGE_DOS_SIGNATURE) return nullptr;
  const IMAGE_NT_HEADERS64* nt = reinterpret_cast<const IMAGE_NT_HEADERS64*>(base + dos->e_lfanew);
  if (nt->Signature != IMAGE_NT_SIGNATURE) return nullptr;
  const IMAGE_DATA_DIRECTORY& d = nt->OptionalHeader.DataDirectory[IMAGE_DIRECTORY_ENTRY_DELAY_IMPORT];
  if (!d.VirtualAddress || !d.Size) return nullptr;
  for (const IMAGE_DELAYLOAD_DESCRIPTOR* dl =
           reinterpret_cast<const IMAGE_DELAYLOAD_DESCRIPTOR*>(base + d.VirtualAddress);
       dl->DllNameRVA; ++dl) {
    if (!dl->Attributes.RvaBased) continue;  // pre-VC7 VA-based descriptors: not produced by any supported game
    if (_stricmp(reinterpret_cast<const char*>(base + dl->DllNameRVA), interposer_leaf) == 0) return dl;
  }
  return nullptr;
}

// Patch the delay-load IAT slots of `dl` (in `module`) that name a hooked role. A slot still unbound holds the delay
// helper's thunk; once patched, the game calls our handler directly and the helper never runs for that function. The
// other slots keep their thunks and bind to the genuine interface through the helper as usual.
unsigned patch_module_delay_imports(HMODULE module, const IMAGE_DELAYLOAD_DESCRIPTOR* dl) {
  uint8_t* base = reinterpret_cast<uint8_t*>(module);
  const IMAGE_THUNK_DATA64* names = reinterpret_cast<const IMAGE_THUNK_DATA64*>(base + dl->ImportNameTableRVA);
  IMAGE_THUNK_DATA64* iat = reinterpret_cast<IMAGE_THUNK_DATA64*>(base + dl->ImportAddressTableRVA);
  unsigned patched = 0;
  for (; names->u1.AddressOfData; ++names, ++iat) {
    if (names->u1.Ordinal & IMAGE_ORDINAL_FLAG64) continue;  // by name only, as for the import table
    const IMAGE_IMPORT_BY_NAME* by = reinterpret_cast<const IMAGE_IMPORT_BY_NAME*>(base + names->u1.AddressOfData);
    void* h = handler_for_name(reinterpret_cast<const char*>(by->Name));
    if (!h) {  // clean/rr-sl-diag 2: watched only (bound by the helper when the game calls it)
      void** other = reinterpret_cast<void**>(&iat->u1.Function);
      g_watch.push_back({other, reinterpret_cast<const char*>(by->Name), module_path(module), "delay-load", nullptr, *other, *other});
      continue;
    }
    void** slot = reinterpret_cast<void**>(&iat->u1.Function);
    void* original = *slot;
    if (original == h) continue;
    if (patch_slot(slot, h)) {
      g_patches.push_back({slot, original});
      g_watch.push_back({slot, reinterpret_cast<const char*>(by->Name), module_path(module), "delay-load", h, original, h});
      ++patched;
      log_line(std::string("hook: patched delay-load IAT slot ") + reinterpret_cast<const char*>(by->Name) + " in " +
               module_path(module));
    } else {
      log_line(std::string("hook: FAILED to patch delay-load IAT slot ") + reinterpret_cast<const char*>(by->Name) +
               " in " + module_path(module));
    }
  }
  return patched;
}

// The game exe links the interface by delay-load import: load the game's own interface file (the one in the exe's
// folder, which is where the delay helper's search starts) so its exports can be bound now. The file is the
// game's genuine, signed one; nothing on disk changes, and any signature check the game runs on it is unaffected.
HMODULE load_delay_linked_interposer(const std::wstring& leaf_w) {
  wchar_t exe[MAX_PATH];
  DWORD n = GetModuleFileNameW(nullptr, exe, MAX_PATH);
  if (!n || n >= MAX_PATH) return nullptr;
  std::wstring path(exe, n);
  const size_t slash = path.find_last_of(L"\\/");
  path = (slash == std::wstring::npos ? std::wstring() : path.substr(0, slash + 1)) + leaf_w;
  if (GetFileAttributesW(path.c_str()) == INVALID_FILE_ATTRIBUTES) {
    log_line("hook: the exe delay-loads " + std::string(kWgInterposerFile) + " but it is not next to the exe (" +
             narrow(path) + "); nothing was patched");
    return nullptr;
  }
  HMODULE m = LoadLibraryW(path.c_str());
  if (!m) {
    log_line("hook: loading " + narrow(path) + " failed (error " + std::to_string(GetLastError()) +
             "); nothing was patched");
    return nullptr;
  }
  log_line("hook: game interface module loaded for the exe's delay-load import: " + module_path(m));
  return m;
}

// clean/rr-sl-diag 2: the loaded modules whose import or delay-import table names the game interface ("<module> (import,
// delay-load)", one entry per module), read from their headers; empty: none
std::string interface_importers() {
  std::string out;
  HMODULE mods[1024];
  DWORD need = 0;
  if (!K32EnumProcessModules(GetCurrentProcess(), mods, sizeof mods, &need)) return "(modules not listed)";
  const unsigned count = (unsigned)(need / sizeof(HMODULE)) < 1024 ? (unsigned)(need / sizeof(HMODULE)) : 1024;
  for (unsigned i = 0; i < count; ++i) {
    const uint8_t* base = reinterpret_cast<const uint8_t*>(mods[i]);
    const IMAGE_DOS_HEADER* dos = reinterpret_cast<const IMAGE_DOS_HEADER*>(base);
    if (dos->e_magic != IMAGE_DOS_SIGNATURE) continue;
    const IMAGE_NT_HEADERS64* nt = reinterpret_cast<const IMAGE_NT_HEADERS64*>(base + dos->e_lfanew);
    if (nt->Signature != IMAGE_NT_SIGNATURE || nt->OptionalHeader.Magic != IMAGE_NT_OPTIONAL_HDR64_MAGIC) continue;
    bool imp = false;
    const IMAGE_DATA_DIRECTORY& d = nt->OptionalHeader.DataDirectory[IMAGE_DIRECTORY_ENTRY_IMPORT];
    if (d.VirtualAddress && d.Size)
      for (const IMAGE_IMPORT_DESCRIPTOR* e = reinterpret_cast<const IMAGE_IMPORT_DESCRIPTOR*>(base + d.VirtualAddress);
           e->Name; ++e)
        imp = imp || _stricmp(reinterpret_cast<const char*>(base + e->Name), kWgInterposerFile) == 0;
    const bool delay = find_delay_import(mods[i], kWgInterposerFile) != nullptr;
    if (!imp && !delay) continue;
    if (!out.empty()) out += "; ";
    out += where(base).substr(0, where(base).find('+')) + (imp && delay ? " (import, delay-load)" : imp ? " (import)" : " (delay-load)");
  }
  return out;
}

// clean/rr-sl-diag 2: one check of the watched slots: on the first check every slot's state, later only the slots that
// changed since the check before; then the modules that link the game interface and the calls that reached us so far.
void slot_check_now(const char* why) {
  std::lock_guard<std::mutex> lock(g_watch_mutex);
  if (g_watch.empty()) return;
  const unsigned n = ++g_watch_checks;
  auto state = [](const SlotWatch& w, void* v) -> std::string {
    if (w.ours && v == w.ours) return "WrongGPU's handler (as patched)";
    if (v == w.original && !w.ours) return "unbound (the delay helper's thunk at " + where(v) + ")";
    return std::string(w.ours ? "REWRITTEN, now " : "bound to ") + where(v);
  };
  unsigned changed = 0, ours = 0, bound = 0;
  for (SlotWatch& w : g_watch) {
    void* v = *reinterpret_cast<void* volatile*>(w.slot);
    if (w.ours && v == w.ours) ++ours;
    if (!w.ours && v != w.original) ++bound;
    if (n == 1 || v != w.last) {
      ++changed;
      log_line("diag: slot check " + std::to_string(n) + ": " + w.kind + " slot " + w.name + " of " + w.module + ": " +
               state(w, v) + (n > 1 ? " (was " + state(w, w.last) + ")" : ""));
    }
    w.last = v;
  }
  uint64_t calls = 0;
  for (auto& c : g_role_calls) calls += c.load();
  const std::string importers = interface_importers();
  log_line("diag: slot check " + std::to_string(n) + " (" + why + "): " + std::to_string(ours) + " slot(s) hold our handler, " +
           std::to_string(bound) + " other delay-load slot(s) bound by the helper, " + std::to_string(changed) +
           " line(s) above; game interface module " + (GetModuleHandleA(kWgInterposerFile) ? "loaded" : "not loaded") +
           "; modules linking it: " + (importers.empty() ? "none" : importers) + "; game-interface calls that reached us: " +
           std::to_string(calls));
}

// The checks' times in seconds after the hook: off unless WG_RR_SLOT_CHECKS is "on" (the default times) or a comma list
// of seconds (clean/060-rr2: release default off, wgrr::slot_check_schedule)
std::string slot_schedule() {
  char env[256];
  const DWORD n = GetEnvironmentVariableA("WG_RR_SLOT_CHECKS", env, sizeof env);
  return wgrr::slot_check_schedule(n && n < sizeof env ? env : nullptr);
}

DWORD WINAPI slot_watch_thread(void*) {
  const std::string spec = slot_schedule();
  if (spec.empty()) return 0;
  const ULONGLONG t0 = GetTickCount64();
  size_t at = 0;
  while (at < spec.size()) {
    const size_t comma = spec.find(',', at);
    const unsigned s = (unsigned)strtoul(spec.substr(at, comma == std::string::npos ? std::string::npos : comma - at).c_str(), nullptr, 10);
    at = comma == std::string::npos ? spec.size() : comma + 1;
    const ULONGLONG due = t0 + 1000ull * s;
    const ULONGLONG now = GetTickCount64();
    if (due > now) Sleep((DWORD)(due - now));
    slot_check_now(("timer, " + std::to_string(s) + " s after the hook").c_str());
  }
  return 0;
}

// ---------------------------------------------------------------- RR guard: other modules on the RR path (rr_guard.h)
// The roles whose exports' first bytes are read (S2): the ones that decide where RR goes. The init export is not among
// them: the host patches it itself (HOOK-MODE.md).
const int kWgDecisiveRoles[] = {2, 7, 8, 16};  // is-supported, get-function, allocate, evaluate

double ms_since(const LARGE_INTEGER& t0) {
  LARGE_INTEGER t1, f;
  QueryPerformanceCounter(&t1);
  QueryPerformanceFrequency(&f);
  return f.QuadPart ? 1000.0 * (double)(t1.QuadPart - t0.QuadPart) / (double)f.QuadPart : 0.0;
}
std::string ms_text(double ms) {
  char b[32];
  snprintf(b, sizeof b, "%.3f ms", ms);
  return b;
}
std::string leaf_text(HMODULE m) {
  std::wstring w;
  module_leaf(m, &w);
  return ascii(narrow(w));
}

// The hooked role (gen_forwarding.ROLES index) of an import name in this run's route, or -1.
int role_for_name(const char* name) {
  for (unsigned i = 0; i < WG_BINDING_HOOK_ROLE_COUNT; ++i) {
    const int role = kWgHookRoles[i];
    if (role_in_route(role) && kWgRoleNames[role][0] && handler_for(role) && strcmp(kWgRoleNames[role], name) == 0)
      return role;
  }
  return -1;
}

// The by-name entries of one import (or delay-load) name table: f(role, import name, slot address) for each that names
// a hooked role. Every RVA (the name table, the address table, each name) is checked against the image's size (PeView)
// before an address is formed from it; an entry outside the image ends the walk (nothing found there).
template <typename F>
void guard_thunks(const wgrr::guard::PeView& pe, wgrr::guard::Reader& r, uint32_t names_rva, uint32_t iat_rva, F f) {
  for (unsigned i = 0; i < 4096; ++i) {
    const uint64_t off = (uint64_t)i * sizeof(IMAGE_THUNK_DATA64);
    const uint8_t* np = pe.at(names_rva + off, sizeof(IMAGE_THUNK_DATA64));
    const uint8_t* sp = pe.at(iat_rva + off, sizeof(IMAGE_THUNK_DATA64));
    IMAGE_THUNK_DATA64 t;
    if (!np || !sp || !r.get(np, &t) || !t.u1.AddressOfData) return;
    if (t.u1.Ordinal & IMAGE_ORDINAL_FLAG64) continue;
    const uint8_t* by = pe.at(t.u1.AddressOfData, offsetof(IMAGE_IMPORT_BY_NAME, Name) + 1);
    char nm[96];
    if (!by || !r.str(by + offsetof(IMAGE_IMPORT_BY_NAME, Name), nm, sizeof nm)) continue;
    const int role = role_for_name(nm);
    if (role >= 0) f(role, nm, const_cast<void**>(reinterpret_cast<void* const*>(sp)));
  }
}

// Every import slot of `module` that names a hooked role from the DLL `leaf`, read through the guard's reader: f(role,
// import name, slot address).
template <typename F>
void guard_import_slots(HMODULE module, const char* leaf, wgrr::guard::Reader& r, F f) {
  wgrr::guard::PeView pe;
  uint32_t rva = 0, size = 0;
  if (!pe.open(module, r) || !pe.dir(IMAGE_DIRECTORY_ENTRY_IMPORT, &rva, &size)) return;
  for (unsigned k = 0; k < 4096; ++k) {
    const uint8_t* dp = pe.at(rva + (uint64_t)k * sizeof(IMAGE_IMPORT_DESCRIPTOR), sizeof(IMAGE_IMPORT_DESCRIPTOR));
    IMAGE_IMPORT_DESCRIPTOR e;
    if (!dp || !r.get(dp, &e) || !e.Name) return;
    const uint8_t* np = pe.at(e.Name, 1);
    char dll[64];
    if (!np || !r.str(np, dll, sizeof dll) || _stricmp(dll, leaf) != 0) continue;
    guard_thunks(pe, r, e.OriginalFirstThunk ? e.OriginalFirstThunk : e.FirstThunk, e.FirstThunk, f);
  }
}

// S0: the process-wide marker of the active WrongGPU RR copy. True: this copy is the first (it holds the marker). False:
// another copy holds it; this copy stays off (one line).
bool guard_claim_marker() {
  wchar_t name[64];
  swprintf(name, 64, L"Local\\WrongGPU.rr-active.v1.%lu", (unsigned long)GetCurrentProcessId());
  HANDLE h = CreateFileMappingW(INVALID_HANDLE_VALUE, nullptr, PAGE_READWRITE, 0, 4096, name);
  if (!h) return true;  // cannot tell: the slot scan still sees a copy that patched
  const bool existed = GetLastError() == ERROR_ALREADY_EXISTS;
  wchar_t* view = static_cast<wchar_t*>(MapViewOfFile(h, FILE_MAP_ALL_ACCESS, 0, 0, 4096));
  if (!existed) {
    if (view) wcsncpy_s(view, 2047, own_path().c_str(), _TRUNCATE);
    if (view) UnmapViewOfFile(view);
    g_marker = h;
    return true;
  }
  const std::wstring active = view ? std::wstring(view, wcsnlen(view, 2047)) : std::wstring(L"another copy");
  if (view) UnmapViewOfFile(view);
  CloseHandle(h);
  g_guard.set(wgrr::guard::State::kOwnCopy);
  const std::string line = wgrr::guard::line_own_copy(ascii(narrow(active)), ascii(narrow(own_path())));
#if WG_RR_PROBE
  // the active copy writes the probe log (opening it here would start it again): append the one line to it
  {
    const std::wstring probe_path = data_dir() + L"wronggpu-rr-probe.log";
    const wgrr::LogLock shared(probe_path, wgrr::kLogLockWaitMs);  // the active copy's writes and this one, in one order
    wgrr::append_log_text(probe_path, wgrr::log_stamp() + line + "\r\n");
  }
#else
  log_line(line);
#endif
  return false;
}

void guard_read_override() {
  ensure_settings();
  std::string note;
  g_mode.store((int)wgrr::guard::parse_mode(g_settings.text(wgrr::guard::kOverrideKey), &note));
  if (!note.empty()) log_line("hook: " + note);
}
wgrr::guard::Mode guard_mode() { return static_cast<wgrr::guard::Mode>(g_mode.load()); }

// The guard's view of the process: the module that called InitializeASI and the one holding the host's API table are
// both taken as the host (a separate ASI loader may be the first, OptiScaler the second): their own patches are
// accepted and not named. A wrong guess only changes the wording (a named module is inconclusive).
wgrr::guard::Env guard_env(const void* iface) {
  return wgrr::guard::process_env(g_self, iface, g_host_module.load(), g_host_api_module.load());
}

// One diagnostic line when a scan could not explain some addresses (never a finding: RR stays active)
void guard_log_unclear(const char* when, const wgrr::guard::Verdict& v) {
  if (v.unclear) log_line(wgrr::guard::line_inconclusive(when, v.unclear, v.unclear_detail));
}

// The result words of a check's summary line
const char* guard_result(const wgrr::guard::Verdict& v) {
  return v.kind == wgrr::guard::Verdict::kClean     ? "no other Ray Reconstruction plugin found"
         : v.kind == wgrr::guard::Verdict::kOwnCopy ? "a second copy of WrongGPU RR found"
                                                    : "another module on the RR path (inconclusive)";
}

// Applies decide() to a verdict and logs what it means: the own-copy line, or the one line naming a foreign module
// (with the tail of the player's setting). Returns the new state (also stored).
wgrr::guard::State guard_apply(const wgrr::guard::Verdict& v, wgrr::guard::Stage stage) {
  namespace g = wgrr::guard;
  const g::State before = g_guard.state();
  const g::Mode mode = guard_mode();
  const g::State after = g::decide(v.kind, mode, stage, before);
  if (before != g::State::kActive && before != g::State::kTaken) return before;
  if (v.kind == g::Verdict::kOwnCopy) {
    log_line(g::line_own_copy(v.detail, ascii(narrow(own_path()))));
  } else if (v.kind == g::Verdict::kForeign) {
    const g::Tail tail = mode == g::Mode::kYield ? (stage == g::Stage::kLoad ? g::Tail::kYieldOff : g::Tail::kYieldPassing)
                         : (mode == g::Mode::kTake && stage != g::Stage::kLate) ? g::Tail::kTakeInFront
                                                                                : g::Tail::kStaysActive;
    log_line(g::line_foreign(v.module, v.detail, tail));
  }
  g_guard.set(after);
  return after;
}

// Where each RR export led at the load check (written by do_hook's thread before g_hook_done is published; read by the
// late check after it): an export that still leads to the same place was reported (or accepted) at load already and
// is not "taken after WrongGPU".
struct ExportSeen {
  bool set = false;
  wgrr::guard::Owner owner = wgrr::guard::Owner::kUnknown;
  const void* module = nullptr;
};
ExportSeen g_export_seen[32];

// S2: the first bytes of the interface's RR-deciding exports (module = the interface). anon_ok: code outside every module
// is accepted (host-route mode: the host's own trampolines). record: remember where each led (the load check);
// unchanged_only: count an export that leads where it led at load as read, without classifying it again (late check).
void guard_scan_exports(HMODULE module, const wgrr::guard::Env& env, bool anon_ok, wgrr::guard::Verdict* v,
                        bool record = false, bool unchanged_only = false) {
  const std::string leaf = leaf_text(module);
  for (int role : kWgDecisiveRoles) {
    if (!role_in_route(role) || !kWgRoleNames[role][0]) continue;
    const void* e = reinterpret_cast<const void*>(GetProcAddress(module, kWgRoleNames[role]));
    if (!e) continue;
    const wgrr::guard::Where w = wgrr::guard::classify(e, env, true);
    ExportSeen& seen = g_export_seen[role];
    if (record) seen = {true, w.owner, w.module_handle};
    if (unchanged_only && seen.set && seen.owner == w.owner && seen.module == w.module_handle) {
      ++v->reads;
      continue;
    }
    v->add(w, std::string("the export ") + kWgRoleNames[role] + " of " + leaf, anon_ok, false);
  }
}

// At load (do_hook, before the core is built and before anything is patched): S1 over every module's import slots of
// the interface (and the exe's delay-load slots), S2 over the RR exports. True: go on (active, or take). False: off (a
// second copy of WrongGPU RR, or rr_other_plugin=yield with a named module on the path).
bool guard_at_load(const IMAGE_DELAYLOAD_DESCRIPTOR* delay) {
  namespace g = wgrr::guard;
  guard_read_override();
  LARGE_INTEGER t0;
  QueryPerformanceCounter(&t0);
  const g::Env env = guard_env(g_interposer);
  g::Reader r;
  g::Verdict v;
  std::wstring leaf_w;
  module_leaf(g_interposer, &leaf_w);
  const std::string leaf = narrow(leaf_w);
  HMODULE exe = GetModuleHandleW(nullptr);
  void* prev[32] = {};
  HMODULE mods[1024];
  DWORD need = 0;
  if (K32EnumProcessModules(GetCurrentProcess(), mods, sizeof mods, &need)) {
    const unsigned count = (unsigned)(need / sizeof(HMODULE)) < 1024 ? (unsigned)(need / sizeof(HMODULE)) : 1024;
    for (unsigned i = 0; i < count; ++i) {
      if (mods[i] == g_self || mods[i] == g_interposer) continue;
      guard_import_slots(mods[i], leaf.c_str(), r, [&](int role, const char* nm, void** slot) {
        void* val = nullptr;
        const g::Where w = r.get(slot, &val) ? g::classify(val, env, false) : g::Where();
        v.add(w, std::string("the import slot ") + nm + " of " + leaf_text(mods[i]), false, false);
        if (mods[i] == exe && val && w.owner != g::Owner::kInterface) prev[role] = val;
      });
    }
  }
  g::PeView pe;
  if (delay && pe.open(exe, r)) {  // the exe's delay-load slots: unbound ones hold the delay helper's thunk in the exe
    guard_thunks(pe, r, delay->ImportNameTableRVA, delay->ImportAddressTableRVA, [&](int role, const char* nm, void** slot) {
      void* val = nullptr;
      const g::Where w = r.get(slot, &val) ? g::classify(val, env, false) : g::Where();
      v.add(w, std::string("the delay-load slot ") + nm + " of " + leaf_text(exe), false, true);
      if (val && w.owner != g::Owner::kInterface && w.owner != g::Owner::kExeThunk && w.owner != g::Owner::kHost)
        prev[role] = val;
    });
  }
  guard_scan_exports(g_interposer, env, false, &v, true);
  const double ms = ms_since(t0);
  log_line("hook: coexistence check at load: " + std::string(guard_result(v)) + " (" + std::to_string(v.reads) +
           " address(es) read in " + ms_text(ms) + ")");
  guard_log_unclear("at load", v);
  if (guard_mode() == g::Mode::kTake) memcpy(g_prev, prev, sizeof g_prev);  // used only when the state becomes kTaken
  const g::State s = guard_apply(v, g::Stage::kLoad);
  return s == g::State::kActive || s == g::State::kTaken;
}

// S4: after our handlers are installed. Our patched slots that no longer hold our handler lead somewhere else; while
// active (not taken) the RR exports are read again too. Returns the state to keep; logs a finding (a foreign module once
// per run) and, with log_clean, the clean result and its cost.
wgrr::guard::State guard_late(const char* why, bool log_clean) {
  namespace g = wgrr::guard;
  const g::State s = g_guard.state();
  if (s != g::State::kActive && s != g::State::kTaken) return s;
  LARGE_INTEGER t0;
  QueryPerformanceCounter(&t0);
  const g::Env env = guard_env(g_interposer);
  g::Verdict v;
  {
    std::lock_guard<std::mutex> lock(g_watch_mutex);
    for (const SlotWatch& w : g_watch) {
      if (!w.ours) continue;
      void* val = nullptr;
      if (g::safe_get(w.slot, &val) && val == w.ours) {
        ++v.reads;
        continue;
      }
      const size_t slash = w.module.find_last_of("\\/");  // the module's file name, as at load
      v.add(val ? g::classify(val, env, false) : g::Where(), "the " + w.kind + " slot " + w.name + " of " +
            (slash == std::string::npos ? w.module : w.module.substr(slash + 1)) + " (WrongGPU's hook)", false, false);
    }
  }
  if (s == g::State::kActive && g_interposer && !g_route_mode.load())
    guard_scan_exports(g_interposer, env, false, &v, false, true);  // what changed since the load check
  const double ms = ms_since(t0);
  const std::string cost = " (" + std::to_string(v.reads) + " address(es) read in " + ms_text(ms) + ")";
  const std::string head = std::string("hook: coexistence re-check at ") + why + ": ";
  // a foreign module that stays inconclusive (off, take) is named once per run: OptiPluginInit and the first call may
  // both see it
  const bool quiet_foreign = v.kind == g::Verdict::kForeign && guard_mode() != g::Mode::kYield &&
                             g_late_foreign_logged.exchange(true);
  if (v.kind == g::Verdict::kClean || quiet_foreign) {
    if (log_clean)
      log_line(head + (quiet_foreign ? "another module on the RR path, named above" : guard_result(v)) + cost);
    guard_log_unclear((std::string("re-check at ") + why).c_str(), v);
    return s;
  }
  log_line(head + (v.kind == g::Verdict::kOwnCopy ? "a second copy of WrongGPU RR" : "another module") +
           " took the RR calls after WrongGPU" + cost);
  guard_log_unclear((std::string("re-check at ") + why).c_str(), v);
  return guard_apply(v, g::Stage::kLate);
}

void guard_first_call() {
  g_guard.freeze_once([] { return guard_late("the game's first call", true); });
}

void do_hook() {
  diagnostics("InitializeASI");
  std::wstring interposer_w;
  for (const char* c = kWgInterposerFile; *c; ++c) interposer_w.push_back((wchar_t)(unsigned char)*c);
  g_interposer = GetModuleHandleW(interposer_w.c_str());
  const IMAGE_DELAYLOAD_DESCRIPTOR* delay = nullptr;
  if (!g_interposer) {
    delay = find_delay_import(GetModuleHandleW(nullptr), kWgInterposerFile);
    if (delay) g_interposer = load_delay_linked_interposer(interposer_w);
    if (!g_interposer) {
      if (!delay) {
        g_wait_for_host = true;
        log_line("hook: the game interface module " + std::string(kWgInterposerFile) +
                 " is not loaded at InitializeASI and the exe does not link it by import or delay-load import; "
                 "waiting for the host's game-interface routes (OptiScaler plugin interface 3).");
      }
      return;
    }
  } else {
    log_line("hook: game interface module found: " + module_path(g_interposer) + " (already loaded via import table)");
  }

  unsigned v[4] = {0, 0, 0, 0};
  if (!module_file_version(g_interposer, v)) {
    log_line("hook: could not read the game interface version resource; nothing was patched");
    return;
  }
  std::string families;
  const bool routed = route_for(v[0], v[1], &families);
  log_line("hook: game interface version " + std::to_string(v[0]) + "." + std::to_string(v[1]) + "." +
           std::to_string(v[2]) + "." + std::to_string(v[3]) + ", this build serves " + families + ".x");
  if (!routed) {
    log_line("hook: game interface version does not match this build's binding version (" + families + ".x): " +
             kNoGameInterfaceRoute + " (no import slot patched)" + kNgxStillServed);
    return;
  }
  g_api_minor = v[1];
  log_line("hook: route Streamline 2." + std::to_string(g_api_minor) + " chosen for this game interface");

  // RR guard: a second copy of WrongGPU RR (or rr_other_plugin=yield with a named module on the path) keeps WrongGPU RR
  // off (nothing built or patched); a named module alone is only logged
  if (!guard_at_load(delay)) return;
  set_fwd(bind_original());  // where a passing handler sends a call (the late decision, guard_first_call)

  // Build the core now, so our handlers forward to the genuine exports (the IAT's pre-patch addresses).
  (void)target();
  // We loaded the interface ourselves for a delay-load import: nothing has called it, so the game's <init> is still
  // to come and will reach our handler; there is nothing to adopt.
  if (!delay) {
#if WG_RR_PROBE
    // In hook mode the game's <init> may already have run (the interposer initialised before the plugin loaded); the
    // core adopts it with the safe default (no command-list state restore) rather than refusing RR.
    g_probe->adopt_unseen_init();
#else
    g_proxy->adopt_unseen_init();
    if (g_proxy->init_adopted())
      log_line("hook: patched after the game interface initialised; using safe defaults (see HOOK-MODE.md)");
#endif
  }

  std::wstring interposer_leaf_w;
  module_leaf(g_interposer, &interposer_leaf_w);
  const std::string interposer_leaf = narrow(interposer_leaf_w);

  HMODULE mods[1024];
  DWORD need = 0;
  unsigned total = 0, modules_patched = 0;
  if (K32EnumProcessModules(GetCurrentProcess(), mods, sizeof mods, &need)) {
    const unsigned count = (unsigned)(need / sizeof(HMODULE)) < 1024 ? (unsigned)(need / sizeof(HMODULE)) : 1024;
    for (unsigned i = 0; i < count; ++i) {
      if (mods[i] == g_self || mods[i] == g_interposer) continue;  // never patch ourselves or the interface module
      unsigned p = patch_module_imports(mods[i], interposer_leaf.c_str());
      total += p;
      modules_patched += (p != 0);
    }
  }
  if (delay) {
    const unsigned p = patch_module_delay_imports(GetModuleHandleW(nullptr), delay);
    total += p;
    modules_patched += (p != 0);
  }
  if (!g_watch.empty()) {  // clean/rr-sl-diag 2: the slot watch (read only)
    const bool checks = !slot_schedule().empty();
    log_line("diag: watching " + std::to_string(g_watch.size()) + " game-interface slot(s) (" + std::to_string(total) +
             " patched); modules linking the game interface now: " + interface_importers() +
             (checks ? "" : "; slot checks off (WG_RR_SLOT_CHECKS=on enables them)"));
    if (checks)
      if (HANDLE th = CreateThread(nullptr, 0, slot_watch_thread, nullptr, 0, nullptr)) CloseHandle(th);
  }
  log_line("hook: redirected " + std::to_string(total) + " IAT slot(s) across " + std::to_string(modules_patched) +
           " module(s) to WrongGPU; RR " +
           (
#if WG_RR_PROBE
               std::string("probe")
#else
               std::string(g_proxy && g_proxy->rr_off() ? "switched off (rr_enabled=0; RR reports unsupported)"
                           : g_runtime ? "runtime present" : "runtime absent (RR reports unsupported)")
#endif
               ) +
           ". Non-RR traffic reaches the game interface unchanged.");
  g_hook_done.store(true, std::memory_order_release);  // RR guard: the late check may run from now on
}

// The host hooked a game interface module (plugin interface 3): check its version against this build's binding,
// then register our handlers as routes in front of its exports; the host installs them at once.
void on_host_hooked(void* module, uint32_t major, uint32_t minor, uint32_t patch, void* user) {
  const WgHostMenuApi* api = static_cast<const WgHostMenuApi*>(user);
  const std::string version = std::to_string(major) + "." + std::to_string(minor) + "." + std::to_string(patch);
  if (g_route_mode.load()) {
    log_line("hook: the host hooked another game interface module (" + version + "); RR stays on the first one");
    return;
  }
  std::string families;
  const bool routed = route_for(major, minor, &families);
  log_line("hook: the host hooked the game interface " + module_path(static_cast<HMODULE>(module)) + ", version " +
           version + ", this build serves " + families + ".x");
  if (!routed) {
    log_line("hook: game interface version does not match this build's binding version (" + families + ".x): " +
             kNoGameInterfaceRoute + " (no host routes registered)" + kNgxStillServed);
    return;
  }
  g_api_minor = minor;  // before the core is built (target() below)
  log_line("hook: route Streamline 2." + std::to_string(g_api_minor) + " chosen for this game interface");
  // RR guard: the RR exports of the module the host hooked (its own trampolines are accepted), before our routes
  namespace g = wgrr::guard;
  if (!g_guard.active()) {
    log_line("hook: no game-interface routes registered: WrongGPU RR stays off in this process (see the line above)");
    return;
  }
  guard_read_override();
  {
    LARGE_INTEGER t0;
    QueryPerformanceCounter(&t0);
    g::Verdict v;
    guard_scan_exports(static_cast<HMODULE>(module), guard_env(module), true, &v);
    log_line(std::string("hook: coexistence check before the host routes: ") + guard_result(v) + " (" +
             std::to_string(v.reads) + " address(es) read in " + ms_text(ms_since(t0)) + ")");
    guard_log_unclear("before the host routes", v);
    const g::State s = guard_apply(v, g::Stage::kLoad);  // nothing of ours is in place yet
    if (s != g::State::kActive && s != g::State::kTaken) return;  // own copy, or yield: no routes
  }
  std::vector<WgHostSlRoute> routes;
  for (unsigned i = 0; i < WG_BINDING_HOOK_ROLE_COUNT; ++i) {
    const int role = kWgHookRoles[i];
    void* h = handler_for(role);
    if (!role_in_route(role) || !kWgRoleNames[role][0] || !h) continue;
    routes.push_back({kWgRoleNames[role], h, &g_route_next[role]});
  }
  g_interposer = static_cast<HMODULE>(module);
  g_route_mode.store(true);
  if (!api->register_streamline_routes(g_host_plugin, routes.data(), (uint32_t)routes.size())) {
    // the host's API refused our registration: positive by contract, the game-interface route stays off (the NGX route
    // is decided by the host's own answer there)
    g_route_mode.store(false);
    log_line("hook: the host refused our game-interface routes; nothing was patched: WrongGPU RR stays off on the "
             "game-interface route");
    return;
  }
  unsigned installed = 0;
  for (const WgHostSlRoute& r : routes) installed += (*r.next != nullptr);
  // RR guard: where each route continues (the host's next). Our routes are installed in front and cannot be taken back:
  // a named module below them is inconclusive (RR stays WrongGPU's; the other calls continue through it); a copy of
  // WrongGPU RR, or rr_other_plugin=yield with a named module, makes them pass every call on.
  {
    LARGE_INTEGER t0;
    QueryPerformanceCounter(&t0);
    const g::Env env = guard_env(module);
    g::Verdict v;
    for (const WgHostSlRoute& r : routes)
      if (*r.next) v.add(g::classify(*r.next, env, false), std::string("the host's next function of ") + r.export_name,
                         true, false);
    log_line(std::string("hook: coexistence check of the host routes: ") + guard_result(v) + " (" +
             std::to_string(v.reads) + " address(es) read in " + ms_text(ms_since(t0)) + ")");
    guard_log_unclear("of the host routes", v);
    set_fwd(bind_original());  // the host's next functions
    if (guard_apply(v, g::Stage::kRoutes) == g::State::kPassThrough) {
      g_hook_done.store(true, std::memory_order_release);
      return;
    }
  }
  g_hook_done.store(true, std::memory_order_release);
  (void)target();  // the core forwards to the host's next functions from now on
  log_line("hook: " + std::to_string(installed) + " of " + std::to_string(routes.size()) +
           " game-interface route(s) installed by the host; RR " +
#if WG_RR_PROBE
           std::string("probe")
#else
           std::string(g_proxy && g_proxy->rr_off() ? "switched off (rr_enabled=0; RR reports unsupported)"
                       : g_runtime ? "runtime present" : "runtime absent (RR reports unsupported)")
#endif
           + ". Non-RR traffic reaches the game interface unchanged.");
}

// ---------------------------------------------------------------- clean/rr-sm2-shim: the SetupAPI identity shim
std::string exe_leaf() {
  wchar_t buf[MAX_PATH * 2];
  const DWORD n = GetModuleFileNameW(nullptr, buf, (DWORD)(sizeof buf / sizeof buf[0]));
  std::wstring p(buf, n);
  const size_t slash = p.find_last_of(L"\\/");
  return ascii(narrow(slash == std::wstring::npos ? p : p.substr(slash + 1)));
}

// The first hardware adapter as the game sees it through DXGI: the dxgi.dll the game's own lookup finds (the one already
// loaded in the process, a host's wrapper included; describe_luid uses the system's file instead), and its LUID.
bool game_dxgi_adapter(LUID* luid, wgsi::PciPair* pair, std::string* name) {
  HMODULE dx = GetModuleHandleW(L"dxgi.dll");
  if (!dx) dx = LoadLibraryW(L"dxgi.dll");
  using CreateFactoryFn = HRESULT(WINAPI*)(REFIID, void**);
  auto create = dx ? reinterpret_cast<CreateFactoryFn>(GetProcAddress(dx, "CreateDXGIFactory1")) : nullptr;
  IDXGIFactory1* f = nullptr;
  if (!create || FAILED(create(__uuidof(IDXGIFactory1), reinterpret_cast<void**>(&f))) || !f) return false;
  bool found = false;
  IDXGIAdapter1* a = nullptr;
  for (UINT i = 0; !found && f->EnumAdapters1(i, &a) != DXGI_ERROR_NOT_FOUND; ++i) {
    DXGI_ADAPTER_DESC1 d;
    if (a && SUCCEEDED(a->GetDesc1(&d)) && !(d.Flags & DXGI_ADAPTER_FLAG_SOFTWARE)) {
      *luid = d.AdapterLuid;
      pair->ven = (uint16_t)d.VendorId;
      pair->dev = (uint16_t)d.DeviceId;
      *name = ascii(narrow(d.Description));
      found = true;
    }
    if (a) a->Release();
    a = nullptr;
  }
  f->Release();
  return found;
}

// The adapter's PCI vendor / device as the kernel-mode display driver reports it (the runtime's way, rr_runtime.cpp
// kmt_device_ids; not through DXGI). gdi32's exports are resolved at run time.
bool kmt_pair(LUID luid, wgsi::PciPair* pair) {
  HMODULE g = GetModuleHandleW(L"gdi32.dll");
  if (!g) g = LoadLibraryExW(L"gdi32.dll", nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32);
  if (!g) return false;
  auto open = reinterpret_cast<NTSTATUS(APIENTRY*)(D3DKMT_OPENADAPTERFROMLUID*)>(GetProcAddress(g, "D3DKMTOpenAdapterFromLuid"));
  auto query = reinterpret_cast<NTSTATUS(APIENTRY*)(const D3DKMT_QUERYADAPTERINFO*)>(GetProcAddress(g, "D3DKMTQueryAdapterInfo"));
  auto close = reinterpret_cast<NTSTATUS(APIENTRY*)(const D3DKMT_CLOSEADAPTER*)>(GetProcAddress(g, "D3DKMTCloseAdapter"));
  if (!open || !query || !close) return false;
  D3DKMT_OPENADAPTERFROMLUID o = {};
  o.AdapterLuid = luid;
  if (open(&o) != 0) return false;
  D3DKMT_QUERY_DEVICE_IDS ids = {};
  ids.PhysicalAdapterIndex = 0;
  D3DKMT_QUERYADAPTERINFO q = {};
  q.hAdapter = o.hAdapter;
  q.Type = KMTQAITYPE_PHYSICALADAPTERDEVICEIDS;
  q.pPrivateDriverData = &ids;
  q.PrivateDriverDataSize = sizeof ids;
  const NTSTATUS st = query(&q);
  D3DKMT_CLOSEADAPTER c = {};
  c.hAdapter = o.hAdapter;
  close(&c);
  if (st != 0) return false;
  pair->ven = (uint16_t)ids.DeviceIds.VendorID;
  pair->dev = (uint16_t)ids.DeviceIds.DeviceID;
  return true;
}

// The kernel driver's pair of the first hardware adapter it lists (vendor 1414, the software renderer, skipped): the real
// pair when DXGI cannot be read yet (no LUID from it).
bool kmt_first_adapter(wgsi::PciPair* pair) {
  HMODULE g = GetModuleHandleW(L"gdi32.dll");
  if (!g) g = LoadLibraryExW(L"gdi32.dll", nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32);
  if (!g) return false;
  auto enumerate = reinterpret_cast<NTSTATUS(APIENTRY*)(D3DKMT_ENUMADAPTERS2*)>(GetProcAddress(g, "D3DKMTEnumAdapters2"));
  auto query = reinterpret_cast<NTSTATUS(APIENTRY*)(const D3DKMT_QUERYADAPTERINFO*)>(GetProcAddress(g, "D3DKMTQueryAdapterInfo"));
  auto close = reinterpret_cast<NTSTATUS(APIENTRY*)(const D3DKMT_CLOSEADAPTER*)>(GetProcAddress(g, "D3DKMTCloseAdapter"));
  if (!enumerate || !query || !close) return false;
  D3DKMT_ENUMADAPTERS2 e = {};
  if (enumerate(&e) != 0 || !e.NumAdapters) return false;  // the count
  std::vector<D3DKMT_ADAPTERINFO> list(e.NumAdapters);
  e.pAdapters = list.data();
  if (enumerate(&e) != 0) return false;
  bool found = false;
  for (ULONG i = 0; i < e.NumAdapters; ++i) {
    D3DKMT_QUERY_DEVICE_IDS ids = {};
    D3DKMT_QUERYADAPTERINFO q = {};
    q.hAdapter = list[i].hAdapter;
    q.Type = KMTQAITYPE_PHYSICALADAPTERDEVICEIDS;
    q.pPrivateDriverData = &ids;
    q.PrivateDriverDataSize = sizeof ids;
    if (!found && query(&q) == 0 && ids.DeviceIds.VendorID && ids.DeviceIds.VendorID != 0x1414) {
      pair->ven = (uint16_t)ids.DeviceIds.VendorID;
      pair->dev = (uint16_t)ids.DeviceIds.DeviceID;
      found = true;
    }
    D3DKMT_CLOSEADAPTER c = {};
    c.hAdapter = list[i].hAdapter;
    close(&c);  // every adapter the enumeration opened
  }
  return found;
}

std::string read_text_file(const std::wstring& path) {
  std::string out;
  FILE* f = _wfopen(path.c_str(), L"rb");
  if (!f) return out;
  char b[4096];
  size_t n;
  while ((n = fread(b, 1, sizeof b, f)) > 0 && out.size() < (1u << 20)) out.append(b, n);
  fclose(f);
  return out;
}

// One try of the decision (the game's first SetupAPI calls): DXGI and its adapter's kernel pair; without DXGI, the host's
// spoof settings (OptiScaler.ini next to the game exe) and the kernel pair of the first hardware adapter.
wgsi::Resolution setupapi_identity_try(const std::string& setting, const std::string& exe) {
  wgsi::ResolveInputs in;
  in.setting = setting;
  in.exe = exe;
  LUID luid = {};
  std::string adapter;
  in.dxgi_ok = game_dxgi_adapter(&luid, &in.dxgi, &adapter);
  if (in.dxgi_ok) {
    in.kmt_ok = kmt_pair(luid, &in.kmt);
  } else {
    in.kmt_ok = kmt_first_adapter(&in.kmt);
    in.host = wgsi::parse_host_ini(read_text_file(data_dir() + L"OptiScaler.ini"));
  }
  wgsi::Resolution r = wgsi::resolve(in);
  if (r.final && in.dxgi_ok) r.d.why += " (DXGI adapter \"" + adapter + "\")";
  return r;
}

// At InitializeASI, before the game-interface hook. In Spider-Man2.exe with the switch not off: the two import slots are
// redirected now and the decision is deferred to the game's first call (one decision line then). Otherwise one line
// saying why the shim stays out, and nothing is redirected.
void setupapi_identity_start() {
  ensure_settings();
  const std::string setting = g_settings.text("setupapi_identity"), exe = exe_leaf();
  if (!wgsi::target_exe(exe) || wgsi::setting_off(setting, nullptr)) {
    wgsi::GateInputs in;
    in.setting = setting;
    in.exe = exe;
    log_line("setupapi identity: " + wgsi::decide(in).why);
    return;
  }
  std::string note, detail;
  wgsi::setting_off(setting, &note);
  const unsigned n = wgsi::install(GetModuleHandleW(nullptr), [setting, exe] { return setupapi_identity_try(setting, exe); },
                                   wgsi::Shim::kDecisionTries, log_line, &detail);
  log_line("setupapi identity: " + note + std::to_string(n) + " import slot(s) of " + exe + " redirected (" + detail +
           "); decision deferred to the first call");
}

// Plugin interface 3 present and do_hook found nothing to patch: ask the host for its game-interface hook.
void request_host_routes(const WgHostMenuApi* api, WgHostMenuPlugin* plugin) {
  if (!g_wait_for_host) return;
  if (!WG_HOST_MENU_HAS(api, register_streamline_routes) || !WG_HOST_MENU_HAS(api, set_streamline_callback)) {
    log_line("hook: the host has no game-interface routes (OptiScaler plugin interface 3); nothing was patched");
    return;
  }
  g_host_plugin = plugin;
  log_line("hook: asking the host for its game-interface hook (plugin interface 3)");
  api->set_streamline_callback(plugin, &on_host_hooked, const_cast<WgHostMenuApi*>(api));
}

#if !WG_RR_PROBE
// ---------------------------------------------------------------- clean/rr-ngx: the NGX RR provider (plugin interface 4)
// The provider's calls run inside the game's NGX calls. A fault inside them (an access violation in our code or the
// runtime) must not take the game down: the trampolines catch it, report a failure to the host (the game falls back)
// and switch the provider off for the rest of the run. After a structured exception the provider's lock may stay held
// by the faulting thread, so nothing takes it any more: the trampolines return at once and the route guard
// (ngx_route_holds_rr) reads the lock-free live count.
std::atomic<bool> g_ngx_faulted{false};

void ngx_fault(const char* call) {
  // a structured exception may leave the provider's bracket on the game's list open: close it, so the host's nesting
  // depth and state tracking stay balanced (no effect after a C++ exception: unwinding closed it already)
  wgrr::NgxRrProvider::close_open_bracket_after_fault();
  if (!g_ngx_faulted.exchange(true))
    log_line(std::string("NGX: a fault inside the RR provider's ") + call + "; the NGX route is off for this run (the "
             "game sees failures and keeps its own path)");
}
bool ngx_create_cpp(void* u, void* dev, void* cl, const WgHostNgxParams* p, void** ctx) {
  if (g_ngx_creates++ == 0 && !slot_schedule().empty())  // clean/rr-sl-diag 2; off by default (clean/060-rr2)
    slot_check_now("the host's first RR create of the NGX route");
  try {
    const bool ok = static_cast<wgrr::NgxRrProvider*>(u)->create(dev, cl, p, ctx);
    if (ok) ++g_ngx_created;
    return ok;
  } catch (...) {
    ngx_fault("create (exception)");
    return false;
  }
}
bool ngx_evaluate_cpp(void* u, void* ctx, void* cl, const WgHostNgxParams* p) {
  ++g_ngx_evaluates;  // clean/rr-sl-diag 2
  try {
    return static_cast<wgrr::NgxRrProvider*>(u)->evaluate(ctx, cl, p);
  } catch (...) {
    ngx_fault("evaluate (exception)");
    return false;
  }
}
void ngx_release_cpp(void* u, void* ctx) {
  try {
    static_cast<wgrr::NgxRrProvider*>(u)->release(ctx);
  } catch (...) {
    ngx_fault("release (exception)");
  }
}
// RR guard: WrongGPU RR went off after the provider was registered (a copy of WrongGPU RR, or yield): creates are
// refused (logged once)
bool ngx_guard_refuses() {
  if (g_guard.active()) return false;
  static std::atomic<bool> logged{false};
  if (!logged.exchange(true)) log_line("NGX: RR create refused: WrongGPU RR stays off in this process (see the line above)");
  return true;
}
bool ngx_create_seh(void* u, void* dev, void* cl, const WgHostNgxParams* p, void** ctx) {
  if (g_ngx_faulted.load() || ngx_guard_refuses()) return false;
  __try {
    return ngx_create_cpp(u, dev, cl, p, ctx);
  } __except (EXCEPTION_EXECUTE_HANDLER) {
    ngx_fault("create");
    return false;
  }
}
bool ngx_evaluate_seh(void* u, void* ctx, void* cl, const WgHostNgxParams* p) {
  if (g_ngx_faulted.load()) return false;
  __try {
    return ngx_evaluate_cpp(u, ctx, cl, p);
  } __except (EXCEPTION_EXECUTE_HANDLER) {
    ngx_fault("evaluate");
    return false;
  }
}
void ngx_release_seh(void* u, void* ctx) {
  if (g_ngx_faulted.load()) return;
  __try {
    ngx_release_cpp(u, ctx);
  } __except (EXCEPTION_EXECUTE_HANDLER) {
    ngx_fault("release");
  }
}

// The device under the game's command list (a host that hands no device at create).
void* device_of_cmd_list(void* cl) {
  if (!cl) return nullptr;
  ID3D12Device* dev = nullptr;
  if (FAILED(static_cast<ID3D12DeviceChild*>(cl)->GetDevice(__uuidof(ID3D12Device), reinterpret_cast<void**>(&dev))) ||
      !dev)
    return nullptr;
  dev->Release();  // the game keeps the device alive; the runtime takes its own references
  return dev;
}

// The host's command-list restore for the NGX route (plugin interface 5). The host already brackets each NGX evaluate;
// this pair nests inside it (only the outermost restores) and follows the same switch as the game-interface route.
void apply_ngx_restore() {
  if (!g_ngx) return;
  const WgHostMenuApi* api = g_host_api.load();
  WgHostMenuPlugin* plugin = g_host_api_plugin.load();
  const bool host_has = api && plugin && WG_HOST_MENU_HAS(api, begin_cmd_list_work) &&
                        WG_HOST_MENU_HAS(api, end_cmd_list_work);
  if (host_has && !host_restore_switched_off()) {
    wgrr::CmdListBracket b;
    b.begin = [api, plugin](void* cl) { api->begin_cmd_list_work(plugin, cl); };
    b.end = [api, plugin](void* cl) { api->end_cmd_list_work(plugin, cl); };
    g_ngx->set_cmd_list_bracket(std::move(b));
  } else {
    g_ngx->set_cmd_list_bracket(wgrr::CmdListBracket());
  }
}

// OptiPluginInit: register the NGX RR provider when the host has plugin interface 4 and the player's switch is on.
// Decided once per process (the host takes one provider); a host table without it is logged once and skipped.
void register_ngx_provider(const WgHostMenuApi* api, WgHostMenuPlugin* plugin) {
  if (g_ngx_decided.load() || !api || !plugin) return;
  static bool logged_skip = false;
  if (api->version < 4u || !WG_HOST_MENU_HAS(api, register_ngx_rr)) {
    if (!logged_skip) {
      logged_skip = true;
      log_line("NGX: the host has no RR provider registration (plugin interface " + std::to_string(api->version) +
               ", needs 4); games that call NGX directly keep their own path");
    }
    return;
  }
  if (!g_guard.active()) {  // RR guard: off in this process (a copy of WrongGPU RR, or rr_other_plugin=yield)
    log_line("NGX: RR provider not registered: WrongGPU RR stays off in this process (see the line above)");
    g_ngx_decided.store(true);
    return;
  }
  const wgrr::RrEnabled& enabled = rr_enabled();
  if (!enabled.on) {
    log_line("NGX: RR provider not registered: rr_enabled=0 (" + enabled.source + "); games that call NGX directly "
             "keep their own path");
    g_ngx_decided.store(true);  // decided for this run
    return;
  }
  const std::wstring rt_path = data_dir() + WG_RR_RUNTIME_FILE;
  if (GetFileAttributesW(rt_path.c_str()) == INVALID_FILE_ATTRIBUTES) {
    log_line("NGX: RR provider not registered: the RR runtime is not next to the game exe (" + narrow(rt_path) + ")");
    g_ngx_decided.store(true);
    return;
  }
  ensure_settings();
  if (!g_ngx) {
    wgrr::NgxRrDeps d;
    d.runtime = [] {
      load_runtime();
      return g_runtime;
    };
    d.settings = [] {
      ensure_settings();
      return &g_settings;
    };
    d.log = log_line;
    d.clock = [] { return (uint64_t)GetTickCount64(); };
    d.device_of = device_of_cmd_list;
    d.other_route_busy = [] { return g_proxy != nullptr && g_proxy->has_instance(); };
    d.frame_done = [] {
      g_last_rr_frame_ms.store(GetTickCount64(), std::memory_order_relaxed);
      wgrr::panel::frame_done();  // the RR name and GPU time in OptiScaler's overlay, as for the other route
    };
    g_ngx = new wgrr::NgxRrProvider(std::move(d));
  }
  apply_ngx_restore();
  static WgHostNgxRr table;  // the host copies it; kept anyway
  table = g_ngx->table();
  table.create = &ngx_create_seh;
  table.evaluate = &ngx_evaluate_seh;
  table.release = &ngx_release_seh;
  g_ngx_decided.store(true);
  if (api->register_ngx_rr(plugin, &table)) {
    g_ngx_registered.store(true);
    const bool host_has = WG_HOST_MENU_HAS(api, begin_cmd_list_work) && WG_HOST_MENU_HAS(api, end_cmd_list_work);
    log_line("NGX: RR provider registered with the host (plugin interface " + std::to_string(api->version) +
             "): games that call NGX directly get WrongGPU RR" +
             (host_has ? "; the host restores the game's command-list state after each evaluate"
                       : "; the host has no command-list restore (plugin interface 5)"));
  } else {  // RR guard: the host keeps one NGX RR provider; another one holds it (NGX route only)
    log_line(wgrr::guard::line_ngx_other());
  }
}
#endif

}  // namespace

// OptiScaler's ASI entry (same convention WrongGPU.asi uses). All real work runs here, never under the loader lock.
extern "C" __declspec(dllexport) void InitializeASI() {
  // RR guard: the module that calls us is taken as the host (its own patches are accepted; wording only)
  void* caller = _ReturnAddress();
  std::call_once(g_init_once, [caller] {
    HMODULE host = nullptr;
    if (GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                           static_cast<LPCWSTR>(caller), &host) && host != g_self)
      g_host_module.store(host);
    if (!guard_claim_marker()) return;  // a second copy of WrongGPU RR: nothing of this copy starts (one line)
#if !WG_RR_PROBE
    // RR runs while the core accepted an evaluate within the last second
    wgrr::panel::start(
        log_line, [] { return GetTickCount64() - g_last_rr_frame_ms.load(std::memory_order_relaxed) < 1000; },
        [] {  // the RR runtime's smoothed GPU time per frame (runtimes from before i20 have none: 0)
          const WgRrRuntimeV1* rt = g_runtime;
          return rt && rt->size >= WG_RR_RUNTIME_V1_TIME_SIZE && rt->frame_gpu_ms ? rt->frame_gpu_ms(rt->ctx) : 0.0f;
        },
        [] {  // the runtime still creates its frame kernels (first start, cold driver cache): frames pass through
          const WgRrRuntimeV1* rt = g_runtime;
          return rt && rt->status && (rt->status(rt->ctx) & WG_RR_RT_PREPARING) != 0;
        });
#endif
    diagnostics("InitializeASI");
    setupapi_identity_start();  // clean/rr-sm2-shim (Spider-Man2.exe only; one log line)
    do_hook();
  });
}

// OptiScaler's plugin interface (builds that have it call this right after InitializeASI): the WrongGPU tab, and for
// games that load the game interface at run time, the host's game-interface routes (interface 3).
extern "C" __declspec(dllexport) bool OptiPluginInit(const WgHostMenuApi* api, WgHostMenuPlugin* plugin) {
  // RR guard: a second copy of WrongGPU RR registers nothing (no second tab, routes or provider)
  if (g_guard.state() == wgrr::guard::State::kOwnCopy) return false;
  // The module holding the host's table is taken as the host as well (a separate ASI loader may have called
  // InitializeASI): later checks accept its own patches and do not name it. A guess, used for the wording only: a table
  // on the heap or in a helper image gives no module or another one, and nothing depends on it.
  if (api) {
    HMODULE host = nullptr;
    if (GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                           reinterpret_cast<LPCWSTR>(api), &host) && host && host != g_self &&
        host != g_host_module.load())
      g_host_api_module.store(host);
  }
  // a plugin that took the RR calls after our InitializeASI (a non-freezing check; the game's first call decides last)
  if (g_hook_done.load(std::memory_order_acquire))
    g_guard.check([] { return guard_late("OptiPluginInit", false); });
  if (api && plugin) {  // kept before the routes: the host may call back (and the core may start) at once
    g_host_api_plugin.store(plugin);
    g_host_api.store(api);
  }
  apply_host_restore();  // a core started before this call gets the host's restore now
  request_host_routes(api, plugin);
#if !WG_RR_PROBE
  register_ngx_provider(api, plugin);  // the NGX route: plugin interface 4
  apply_ngx_restore();
  return wgrr::panel::menu_init(api, plugin) || g_host_plugin != nullptr || g_ngx_registered.load();
#else
  return g_host_plugin != nullptr;
#endif
}

#if !WG_RR_PROBE
namespace {
// clean/rr-sl-diag: the call summary at process exit (or unload). This runs at DLL_PROCESS_DETACH, under the loader lock,
// and other threads may be gone while holding the log's in-process mutex or the proxy's lock: neither is taken, the
// counters are atomic, and the line is appended with one WriteFile. The log's process-wide named mutex is only tried
// (no wait, see the end): a copy stuck holding it must never hold up the process exit.
void exit_summary() {
  if (!g_log_path[0]) return;  // nothing was logged in this process
  // a second copy of WrongGPU RR (it stays off) has no calls to summarise: the "never reached the plugin" line would
  // mislead (the active copy writes its own summary)
  if (g_guard.state() == wgrr::guard::State::kOwnCopy) return;
  std::string line = g_proxy ? g_proxy->diag_summary("process exit")
                             : std::string("diag: summary (process exit): the game interface's calls never reached the plugin");
  // clean/rr-sl-diag 2: the calls that reached each handler, the NGX route's calls, the slot checks done
  {
    uint64_t total = 0;
    std::string per;
    char b[96];
    for (int r = 0; r < kDiagRoleSlots + 3; ++r) {
      const uint64_t c = g_role_calls[r].load();
      if (!c) continue;
      total += c;
      const char* nm = r < kDiagRoleSlots ? kWgRoleNames[r] : r == kDiagRoleSlots ? "rr-optimal" : r == kDiagRoleSlots + 1 ? "rr-state" : "rr-options";
      snprintf(b, sizeof b, "%s%s %llu", per.empty() ? "" : ", ", nm, (unsigned long long)c);
      per += b;
    }
    snprintf(b, sizeof b, "; game-interface calls that reached us %llu", (unsigned long long)total);
    line += b;
    if (!per.empty()) line += " (" + per + ")";
    snprintf(b, sizeof b, "; NGX route: RR creates %llu (created %llu), evaluates %llu; slot checks %u",
             (unsigned long long)g_ngx_creates.load(), (unsigned long long)g_ngx_created.load(),
             (unsigned long long)g_ngx_evaluates.load(), g_watch_checks);
    line += b;
  }
  SYSTEMTIME t;
  GetLocalTime(&t);
  char ts[24];
  snprintf(ts, sizeof ts, "%02u:%02u:%02u.%03u ", t.wHour, t.wMinute, t.wSecond, t.wMilliseconds);
  line = ts + line + "\r\n";
  // clean/060-rr2 (Brief 13): Marvel's Spider-Man 2 that never initialised its game interface or never asked whether RR is
  // supported: the likely cause is the Steam launch option that turns the game interface off
  {
    std::string exe = exe_leaf();
    for (char& c : exe) c = (char)tolower((unsigned char)c);
    const uint64_t inits = g_role_calls[0].load(), queries = g_proxy ? g_proxy->rr_support_queries() : 0;
    if (exe == "spider-man2.exe" && (!inits || !queries)) {
      char b[96];
      snprintf(b, sizeof b, "(game interface init calls %llu, RR support queries %llu)", (unsigned long long)inits,
               (unsigned long long)queries);
      line += std::string(ts) + "hint: Spider-Man 2 never asked for Ray Reconstruction " + b +
              ": the likely cause is the Steam launch option that turns the game interface off (the release notes name it); "
              "remove it for Ray Reconstruction\r\n";
    }
  }
  // 0.6.1: the log's process-wide lock is tried with no wait (loader lock held here: waiting could deadlock or stall the
  // exit); one append call either way (log_append.h), so the line is whole even when the lock was not ours
  const wgrr::LogLock shared(g_log_path, 0);
  wgrr::append_log_text(g_log_path, line, OPEN_EXISTING);
}
}  // namespace
#endif

BOOL WINAPI DllMain(HINSTANCE inst, DWORD reason, LPVOID) {
  if (reason == DLL_PROCESS_ATTACH) {
    g_self = inst;
    DisableThreadLibraryCalls(inst);
    record_attach(inst);  // fixed-buffer record only; the log is written at the first call
  }
#if !WG_RR_PROBE
  if (reason == DLL_PROCESS_DETACH) exit_summary();
#endif
  return TRUE;
}
