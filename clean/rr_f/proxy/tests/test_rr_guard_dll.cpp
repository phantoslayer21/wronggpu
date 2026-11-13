// SPDX-License-Identifier: MIT
// RR guard: end-to-end test of the coexistence guard in the hook-mode plugin (src/hook_dll.cpp + rr_guard.h).
// The executable is the fake game, as in test_hook_dll.cpp: it imports the RR path's functions by name from the stand-in
// interface (tests/gen_fake_original.py). The "other Ray Reconstruction plugin" is our own stand-in,
// wgtest_other_rr.dll (tests/fake_other_rr.cpp); no third-party file is used.
//   test_rr_guard_dll.exe <stage dir> <mode> [--take]
// Modes (import build):
//   clean        no other plugin: active; 16 threads make the first call at once: one re-check line; the scan cost
//   first        the other plugin redirected the exe's import slots before ours loaded (wrapping): off, nothing patched
//   inline       the other plugin put a jmp [rip] on the evaluate export: off
//   relay        the same through a jmp rel32 to an anonymous relay: off ("through 2 hook jumps")
//   anon         one import slot leads to code outside every module: off
//   noaccess     one import slot leads to a PAGE_NOACCESS page: "could not be read", off, no fault
//   guardpage    one import slot leads to a guard page: the same, and the page is still a guard page afterwards
//   late         the other plugin wrapped our hooks after InitializeASI: 16 threads make the first call at once: one
//                decision, one line, every call passed on unchanged
//   late-replace the other plugin replaced our hooks after InitializeASI (no call reaches us): found at OptiPluginInit
//   provider     the other plugin holds the host's NGX RR provider: NGX route only, the game-interface route stays
//   owncopy      two copies of WrongGPU RR: the second stays off and registers nothing
// --take (wronggpu-rr.ini rr_other_plugin=take): first / inline: WrongGPU RR in front; late: the can't-take line.
// Modes (built with WG_TEST_NOIMPORT: the exe does not link the interface, the host routes are used):
//   route        no other plugin: our routes continue to the interface
//   route-other  the other plugin's routes were registered first (under ours): passing (take: RR ours)
//   route-inline the other plugin patched the evaluate export before the host hooked the module: off, no routes
#define WIN32_LEAN_AND_MEAN
#include <windows.h>

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <atomic>
#include <map>
#include <string>
#include <thread>
#include <vector>

#include "wg_binding.inc"
#include "../include/rr_proxy_abi.h"
#include "../include/wg_host_menu_abi.h"
#include "test_util.h"

#ifndef WG_TEST_NOIMPORT
#define WG_TEST_NOIMPORT 0
#endif

#if !WG_TEST_NOIMPORT
extern "C" {
WgResultCode slIsFeatureSupported(uint32_t, const WgAdapterInfo*);
WgResultCode slEvaluateFeature(uint32_t, const WgFrameToken*, const WgHeader* const*, uint32_t, void*);
WgResultCode slGetFeatureFunction(uint32_t, const char*, void**);
WgResultCode slAllocateResources(void*, uint32_t, const WgViewport*);
WgResultCode slSetD3DDevice(void*);
WgResultCode slInit(const WgPreferences*, uint64_t);
}
// referenced, so that the linker keeps every one in the import table (the guard reads those slots)
void* volatile g_keep_imports[] = {(void*)&slIsFeatureSupported, (void*)&slEvaluateFeature, (void*)&slGetFeatureFunction,
                                   (void*)&slAllocateResources, (void*)&slSetD3DDevice, (void*)&slInit};
#endif

namespace {

struct OrState { long calls[32]; long rr_calls; long ngx_creates; };
struct FoState { int calls; int last_index; const char* last_role; uint32_t feature; };

const uint32_t kRR = WG_FEAT_RAY_RECONSTRUCTION;
const uint32_t kOther = 1;           // another feature: the stand-in interface answers 100 + 1 % 50
const WgResultCode kIfaceAnswer = (WgResultCode)101;
const int kIsSupported = 2, kGetFunction = 7, kAllocate = 8, kEvaluate = 16;

std::wstring widen(const std::string& s) { return std::wstring(s.begin(), s.end()); }
std::vector<std::string> read_lines(const std::wstring& path) {
  std::vector<std::string> out;
  FILE* f = _wfopen(path.c_str(), L"rb");
  if (!f) return out;
  char buf[8192];
  while (fgets(buf, sizeof buf, f)) {
    std::string s(buf);
    while (!s.empty() && (s.back() == '\n' || s.back() == '\r')) s.pop_back();
    out.push_back(s);
  }
  fclose(f);
  return out;
}
size_t count(const std::vector<std::string>& v, const std::string& needle) {
  size_t n = 0;
  for (const auto& s : v) n += s.find(needle) != std::string::npos;
  return n;
}
// 0.6.1: lines that do not start with the "hh:mm:ss.mmm " time stamp: a torn line (two copies' writes run into each
// other); every line the plugin writes starts with it
size_t torn_lines(const std::vector<std::string>& v) {
  size_t n = 0;
  for (const auto& s : v) {
    bool ok = s.size() > 13 && s[2] == ':' && s[5] == ':' && s[8] == '.' && s[12] == ' ';
    for (int i : {0, 1, 3, 4, 6, 7, 9, 10, 11}) ok = ok && s.size() > (size_t)i && s[i] >= '0' && s[i] <= '9';
    n += !ok;
  }
  return n;
}
// the message of a log line starts after the "hh:mm:ss.mmm " time stamp (the probe log adds "f=<frame> ")
size_t count_starting(const std::vector<std::string>& v, const std::string& start) {
  size_t n = 0;
  for (const auto& s : v) {
    size_t at = 13;
    if (s.compare(at, 2, "f=") == 0) at = s.find(' ', at) + 1;
    n += s.size() > at && s.compare(at, start.size(), start) == 0;
  }
  return n;
}
void write_text(const std::wstring& p, const char* s) {
  FILE* f = _wfopen(p.c_str(), L"wb");
  if (f) { fputs(s, f); fclose(f); }
}
// the cost of a coexistence check line ("... address(es) read in 0.123 ms)"), or -1
double cost_of(const std::vector<std::string>& v, const std::string& which) {
  for (const auto& s : v) {
    if (s.find(which) == std::string::npos) continue;
    const size_t at = s.rfind(" read in ");
    if (at != std::string::npos) return atof(s.c_str() + at + 9);
  }
  return -1.0;
}

// ---- a stand-in host (plugin interface 3 / 4 / 5 fields only; no menu): routes chain per export, one NGX provider
namespace host {
HMODULE iface = nullptr;
std::map<std::string, void*> head;                 // the current front of each export's chain
std::map<std::string, void*> handler_of_plugin[2];  // [0] WrongGPU, [1] the other plugin: the handler per export
WgHostMenuPlugin* const kOurs = reinterpret_cast<WgHostMenuPlugin*>(0x1000);
WgHostMenuPlugin* const kTheirs = reinterpret_cast<WgHostMenuPlugin*>(0x2000);
int route_regs[2] = {0, 0}, ngx_regs = 0;
WgHostMenuPlugin* ngx_owner = nullptr;
void (*callback)(void*, uint32_t, uint32_t, uint32_t, void*) = nullptr;
void* callback_user = nullptr;
bool register_routes(WgHostMenuPlugin* p, const WgHostSlRoute* r, uint32_t n) {
  const int who = p == kOurs ? 0 : 1;
  ++route_regs[who];
  for (uint32_t i = 0; i < n; ++i) {
    const std::string name = r[i].export_name;
    void* next = head.count(name) ? head[name] : reinterpret_cast<void*>(GetProcAddress(iface, r[i].export_name));
    *r[i].next = next;
    head[name] = r[i].handler;
    handler_of_plugin[who][name] = r[i].handler;
  }
  return true;
}
bool set_callback(WgHostMenuPlugin*, void (*cb)(void*, uint32_t, uint32_t, uint32_t, void*), void* user) {
  callback = cb;
  callback_user = user;
  return true;
}
bool register_ngx(WgHostMenuPlugin* p, const WgHostNgxRr* prov) {
  ++ngx_regs;
  if (ngx_owner || !prov || !prov->create) return false;
  ngx_owner = p;
  return true;
}
WgHostMenuApi table() {
  WgHostMenuApi a = {};
  a.size = sizeof a;
  a.version = 5;
  a.register_streamline_routes = register_routes;
  a.set_streamline_callback = set_callback;
  a.register_ngx_rr = register_ngx;
  return a;
}
}  // namespace host

using IsSupportedFn = WgResultCode (*)(uint32_t, const WgAdapterInfo*);

// n threads released at once, each calling f(); returns how many returned `want`
template <typename F>
int at_once(int n, F f, WgResultCode want) {
  std::atomic<int> go{0}, ok{0};
  std::vector<std::thread> th;
  for (int i = 0; i < n; ++i)
    th.emplace_back([&] {
      while (!go.load()) std::this_thread::yield();
      if (f() == want) ++ok;
    });
  go.store(1);
  for (auto& t : th) t.join();
  return ok.load();
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 3) return 2;
  std::string stage = argv[1];
  if (stage.back() != '\\') stage += '\\';
  const std::string mode = argv[2];
  bool take = false;
  for (int i = 3; i < argc; ++i) take = take || strcmp(argv[i], "--take") == 0;
  const std::wstring ws = widen(stage);
  const std::wstring log_path = ws + (WG_BINDING_PROBE ? L"wronggpu-rr-probe.log" : L"wronggpu-rr.log");
  DeleteFileW(log_path.c_str());
  DeleteFileW((ws + L"wronggpu-rr.ini").c_str());
  if (take) write_text(ws + L"wronggpu-rr.ini", "rr_other_plugin = take\r\n");
  CreateDirectoryW((ws + L"data").c_str(), nullptr);
  write_text(ws + L"data\\wronggpu.ini", "[wronggpu]\r\nrr_mode = accurate\r\n");
  SetEnvironmentVariableW(L"WG_DATA_DIR", (ws + L"data\\").c_str());
  SetEnvironmentVariableW(L"WG_RR_SLOT_CHECKS", L"0");
  const std::string label = std::string("test_rr_guard_dll (") + (WG_BINDING_PROBE ? "probe" : "release") + ", " + mode +
                            (take ? ", take" : "") + ")";

  HMODULE other = LoadLibraryW((ws + L"wgtest_other_rr.dll").c_str());
  CHECK(other != nullptr);
  if (!other) TEST_SUMMARY(label.c_str());
  auto or_state = reinterpret_cast<const OrState* (*)()>(GetProcAddress(other, "wgtest_or_state"));
  auto or_patch_iat = reinterpret_cast<int (*)(HMODULE, const char*, int)>(GetProcAddress(other, "wgtest_or_patch_iat"));
  auto or_set_slot =
      reinterpret_cast<int (*)(HMODULE, const char*, const char*, void*)>(GetProcAddress(other, "wgtest_or_set_slot"));
  auto or_patch_export = reinterpret_cast<int (*)(HMODULE, const char*, int)>(GetProcAddress(other, "wgtest_or_patch_export"));
  auto or_routes = reinterpret_cast<int (*)(const WgHostMenuApi*, WgHostMenuPlugin*)>(GetProcAddress(other, "wgtest_or_routes"));
  auto or_ngx = reinterpret_cast<bool (*)(const WgHostMenuApi*, WgHostMenuPlugin*)>(GetProcAddress(other, "wgtest_or_register_ngx"));
  const OrState* os = or_state();
  HMODULE exe = GetModuleHandleW(nullptr);
  const std::string exe_leaf = "test_rr_guard_dll.exe";
  const std::string iface_leaf = kWgInterposerFile;
  const std::string evaluate_name = kWgRoleNames[kEvaluate];
  const std::string stem = "another Ray Reconstruction plugin is active; WrongGPU RR stays off (";
  const WgHostMenuApi api = host::table();
  auto load_asi = [&](const std::wstring& file, void (**init)(), WgHostMenuInitFn* pinit) {
    HMODULE m = LoadLibraryW((ws + file).c_str());
    CHECK(m != nullptr);
    *init = m ? reinterpret_cast<void (*)()>(GetProcAddress(m, "InitializeASI")) : nullptr;
    *pinit = m ? reinterpret_cast<WgHostMenuInitFn>(GetProcAddress(m, WG_HOST_MENU_INIT_EXPORT)) : nullptr;
    CHECK(*init != nullptr && *pinit != nullptr);
    return m;
  };
  void (*init)() = nullptr;
  WgHostMenuInitFn plugin_init = nullptr;

#if !WG_TEST_NOIMPORT
  HMODULE iface = GetModuleHandleA(kWgInterposerFile);
  CHECK(iface != nullptr);
  host::iface = iface;
  FoState* fo = reinterpret_cast<FoState* (*)()>(GetProcAddress(iface, "wgtest_fo_state"))();
  const bool late = mode == "late" || mode == "late-replace";
  // ---- the other plugin before ours
  void* odd_page = nullptr;
  if (mode == "first") CHECK(or_patch_iat(exe, kWgInterposerFile, 1) >= 6);
  if (mode == "inline") CHECK(or_patch_export(iface, evaluate_name.c_str(), 0) == 1);
  if (mode == "relay") CHECK(or_patch_export(iface, evaluate_name.c_str(), 1) == 1);
  if (mode == "anon") {
    odd_page = VirtualAlloc(nullptr, 4096, MEM_RESERVE | MEM_COMMIT, PAGE_EXECUTE_READWRITE);
    memset(odd_page, 0xC3, 4096);  // ret: code outside every module that does not jump on
    CHECK(or_set_slot(exe, kWgInterposerFile, evaluate_name.c_str(), odd_page) == 1);
  }
  if (mode == "noaccess" || mode == "guardpage") {
    odd_page = VirtualAlloc(nullptr, 4096, MEM_RESERVE | MEM_COMMIT,
                            mode == "noaccess" ? PAGE_NOACCESS : (PAGE_EXECUTE_READ | PAGE_GUARD));
    CHECK(or_set_slot(exe, kWgInterposerFile, evaluate_name.c_str(), odd_page) == 1);
  }
  if (mode == "provider") CHECK(or_ngx(&api, host::kTheirs));

  // ---- WrongGPU RR (the host loads it and runs its entry)
  load_asi(widen(kWgHookFile), &init, &plugin_init);
  if (!init || !plugin_init) TEST_SUMMARY(label.c_str());
  init();
  if (mode == "owncopy") {  // a second copy under another file name (build.py stages it)
    void (*init2)() = nullptr;
    WgHostMenuInitFn plugin_init2 = nullptr;
    load_asi(L"WrongGPU-RR-copy.asi", &init2, &plugin_init2);
    if (init2) init2();
    CHECK(plugin_init2 && !plugin_init2(&api, host::kOurs));  // the copy registers nothing
  }
  // ---- the other plugin after ours
  if (mode == "late") CHECK(or_patch_iat(exe, kWgInterposerFile, 1) >= 6);          // wraps our handlers
  if (mode == "late-replace") CHECK(or_patch_iat(exe, kWgInterposerFile, 0) >= 6);  // replaces them
  // OptiPluginInit (late-replace: the check that finds it, as no call reaches our handlers any more); "late" makes the
  // first call before it, so that the first call decides
  if (mode != "late") plugin_init(&api, host::kOurs);

  // ---- the game's calls
  if (mode == "clean" || mode == "late" || mode == "owncopy" || mode == "provider") {
    // 16 threads make the game's first interface call at once
    const int ok = at_once(16, [] { return slIsFeatureSupported(kOther, nullptr); }, kIfaceAnswer);
    CHECK_EQ(ok, 16);
  }
  if (mode == "late") plugin_init(&api, host::kOurs);
  std::vector<std::string> L = read_lines(log_path);
  printf("log lines %zu, torn %zu\n", L.size(), torn_lines(L));
  CHECK(torn_lines(L) == 0);  // 0.6.1: one append call per line, also with two copies
  const double load_cost = cost_of(L, "hook: coexistence check at load: ");
  CHECK(load_cost >= 0.0 && load_cost < 5.0);
  printf("coexistence check at load: %.3f ms\n", load_cost);
  CHECK(count(L, "hook: coexistence check at load: ") == (mode == "owncopy" ? 1u : 1u));
  const long rr_before = os->rr_calls;
  const WgResultCode rr = slIsFeatureSupported(kRR, nullptr);
  const bool rr_by_other = os->rr_calls == rr_before + 1;
  const std::string self_slot = "the import slot ";

  if (mode == "clean" || mode == "owncopy") {
    CHECK(count(L, "hook: coexistence check at load: no other Ray Reconstruction plugin found (") == 1);
    CHECK(count(L, "hook: coexistence re-check at the game's first call: no other Ray Reconstruction plugin found (") == 1);
    CHECK(count(L, "another Ray Reconstruction plugin is active") == 0);
    CHECK(count(L, std::string("hook: patched IAT slot ") + kWgRoleNames[kIsSupported] + " in ") == 1);  // only one copy patched
    CHECK(!rr_by_other);
    if (mode == "owncopy") {
      CHECK(count_starting(L, "a second copy of WrongGPU RR is loaded (") == 1);
      CHECK(count(L, std::string(kWgHookFile) + " is active); this copy (") == 1);
      CHECK(count(L, "WrongGPU-RR-copy.asi) stays off") == 1);
    }
#if !WG_BINDING_PROBE
    CHECK(count(L, "NGX: RR provider registered with the host") == 1);
    CHECK(host::ngx_regs == 1 && host::ngx_owner == host::kOurs);
#endif
  } else if (mode == "first" || mode == "inline" || mode == "relay" || mode == "anon" || mode == "noaccess" ||
             mode == "guardpage") {
    const std::string detail =
        mode == "first"    ? self_slot  // the first redirected slot in import order; the module is checked below
        : mode == "inline" ? "the export " + evaluate_name + " of " + iface_leaf + " leads to wgtest_other_rr.dll (through 1 hook jump)"
        : mode == "relay"  ? "the export " + evaluate_name + " of " + iface_leaf + " leads to wgtest_other_rr.dll (through 2 hook jumps)"
        : mode == "anon"   ? self_slot + evaluate_name + " of " + exe_leaf + " leads to code outside every module"
                           : self_slot + evaluate_name + " of " + exe_leaf + " could not be read";
    CHECK(count(L, "hook: coexistence check at load: another Ray Reconstruction plugin found (") == 1);
    if (take && (mode == "first" || mode == "inline")) {
      CHECK(count_starting(L, "another Ray Reconstruction plugin is active; rr_other_plugin=take: WrongGPU RR goes in "
                              "front of it (" + detail) == 1);
      CHECK(count(L, stem) == 0);
      CHECK(count(L, std::string("hook: patched IAT slot ") + kWgRoleNames[kIsSupported] + " in ") == 1);
      CHECK(!rr_by_other);  // RR is WrongGPU's
      if (mode == "first") {  // the other calls go on through the other plugin (it sees them, no RR work)
        const long before = os->calls[kIsSupported];
        CHECK_EQ(slIsFeatureSupported(kOther, nullptr), kIfaceAnswer);
        CHECK(os->calls[kIsSupported] == before + 1);
      }
    } else {
      CHECK(count_starting(L, stem + detail) == 1);
      if (mode == "first") CHECK(count(L, " leads to wgtest_other_rr.dll; nothing patched; ") == 1);
      CHECK(count(L, "; nothing patched; rr_other_plugin=take in wronggpu-rr.ini uses WrongGPU's instead)") == 1);
      CHECK(count(L, "hook: patched IAT slot") == 0);
      CHECK(count(L, "hook: coexistence re-check") == 0);
      if (mode == "first" || mode == "inline" || mode == "relay") {
        CHECK(mode == "first" ? rr_by_other : !rr_by_other);  // inline: the game's import slot reaches the interface
        CHECK_EQ(slIsFeatureSupported(kOther, nullptr), kIfaceAnswer);
      } else {
        CHECK(fo->last_role && strcmp(fo->last_role, "is-supported") == 0 && fo->feature == kRR);  // untouched path
        (void)rr;
      }
#if !WG_BINDING_PROBE
      CHECK(count(L, "NGX: RR provider not registered: WrongGPU RR stays off in this process") == 1);
      CHECK(host::ngx_owner == nullptr);
#endif
    }
    if (mode == "guardpage") {
      MEMORY_BASIC_INFORMATION m;
      CHECK(VirtualQuery(odd_page, &m, sizeof m) && (m.Protect & PAGE_GUARD));  // never touched
    }
  } else if (late) {
    CHECK(count(L, "hook: coexistence check at load: no other Ray Reconstruction plugin found (") == 1);
    CHECK(count(L, std::string("hook: patched IAT slot ") + kWgRoleNames[kIsSupported] + " in ") == 1);
    const std::string why = mode == "late" ? "the game's first call" : "OptiPluginInit";
    CHECK(count(L, "hook: coexistence re-check at " + why + ": another Ray Reconstruction plugin took the RR calls "
                   "after WrongGPU (") == 1);
    CHECK(count(L, "hook: coexistence re-check") == 1);  // one decision, one line, even with 16 threads at once
    const std::string detail = "the import slot ";
    if (take) {
      CHECK(count_starting(L, "a plugin loaded after WrongGPU replaced its hooks; rr_other_plugin=take cannot be "
                              "applied: WrongGPU RR stays off and passes every call on unchanged (" + detail) == 1);
      CHECK(count(L, stem) == 0);
    } else {
      CHECK(count_starting(L, stem + detail) == 1);
      CHECK(count(L, " (WrongGPU's hook) leads to wgtest_other_rr.dll; it loaded after WrongGPU: WrongGPU passes every "
                     "call on unchanged; to use WrongGPU RR, move it aside)") == 1);
    }
    CHECK(rr_by_other);  // RR is the other plugin's
    // a call the other plugin passes on reaches our handler, which passes it on unchanged to the interface
    const int before = fo->calls;
    CHECK_EQ(slEvaluateFeature(kOther, nullptr, nullptr, 0, nullptr), kIfaceAnswer);
    CHECK(fo->calls == before + 1 && fo->last_role && strcmp(fo->last_role, "evaluate") == 0);
    CHECK_EQ(at_once(8, [] { return slIsFeatureSupported(kOther, nullptr); }, kIfaceAnswer), 8);  // consistent after
    CHECK(count(read_lines(log_path), "hook: coexistence re-check") == 1);
#if !WG_BINDING_PROBE
    CHECK(count(read_lines(log_path), "NGX: RR provider not registered: WrongGPU RR stays off in this process") == 1);
    CHECK(host::ngx_owner == nullptr);
#endif
  } else if (mode == "provider") {
#if !WG_BINDING_PROBE
    CHECK(count_starting(L, stem + "NGX route only: the host refused WrongGPU's RR provider, another provider holds it; "
                                   "the game-interface route is decided on its own)") == 1);
    CHECK(host::ngx_owner == host::kTheirs && host::ngx_regs == 2);
    CHECK(count(L, std::string("hook: patched IAT slot ") + kWgRoleNames[kIsSupported] + " in ") == 1);  // the route stays ours
    CHECK(count(L, stem) == 1);
    CHECK(!rr_by_other);
#endif
  } else {
    CHECK(!"unknown mode");
  }
  printf("%s: other plugin answered %ld RR call(s)\n", label.c_str(), os->rr_calls);
#else  // WG_TEST_NOIMPORT: host routes
  (void)or_patch_iat;
  (void)or_set_slot;
  (void)or_ngx;
  load_asi(widen(kWgHookFile), &init, &plugin_init);
  if (!init || !plugin_init) TEST_SUMMARY(label.c_str());
  init();  // the interface is not loaded and not linked: waits for the host
  plugin_init(&api, host::kOurs);
  CHECK(host::callback != nullptr);
  HMODULE iface = LoadLibraryW((ws + widen(kWgInterposerFile)).c_str());  // the game loads it at run time
  CHECK(iface != nullptr);
  host::iface = iface;
  FoState* fo = reinterpret_cast<FoState* (*)()>(GetProcAddress(iface, "wgtest_fo_state"))();
  if (mode == "route-other") CHECK(or_routes(&api, host::kTheirs) == 4);
  if (mode == "route-inline") CHECK(or_patch_export(iface, evaluate_name.c_str(), 0) == 1);
  if (host::callback) host::callback(iface, 2, 14, 1, host::callback_user);
  std::vector<std::string> L = read_lines(log_path);
  printf("log lines %zu, torn %zu\n", L.size(), torn_lines(L));
  CHECK(torn_lines(L) == 0);  // 0.6.1: one append call per line, also with two copies
  CHECK(count(L, "waiting for the host's game-interface routes") == 1);
  auto ours = reinterpret_cast<IsSupportedFn>(host::handler_of_plugin[0][kWgRoleNames[kIsSupported]]);
  if (mode == "route") {
    CHECK(count(L, "hook: coexistence check before the host routes: no other Ray Reconstruction plugin found (") == 1);
    CHECK(count(L, "hook: coexistence check of the host routes: no other Ray Reconstruction plugin found (") == 1);
    CHECK(count(L, "another Ray Reconstruction plugin is active") == 0);
    CHECK(ours != nullptr);
    if (ours) {
      CHECK_EQ(ours(kOther, nullptr), kIfaceAnswer);  // continues to the interface
      const long before = os->rr_calls;
      ours(kRR, nullptr);
      CHECK(os->rr_calls == before);
    }
  } else if (mode == "route-other") {
    const std::string detail = "the host's next function of " + std::string(kWgRoleNames[kIsSupported]) +
                               " leads to wgtest_other_rr.dll";
    CHECK(ours != nullptr);
    if (take) {
      CHECK(count_starting(L, "another Ray Reconstruction plugin is active; rr_other_plugin=take: WrongGPU RR goes in "
                              "front of it (" + detail) == 1);
      CHECK(count(L, stem) == 0);
      if (ours) {
        const long before = os->rr_calls, calls = os->calls[kIsSupported];
        ours(kRR, nullptr);
        CHECK(os->rr_calls == before);  // RR is WrongGPU's
        CHECK_EQ(ours(kOther, nullptr), kIfaceAnswer);  // the other calls continue through the other plugin
        CHECK(os->calls[kIsSupported] == calls + 1);
      }
    } else {
      CHECK(count_starting(L, stem + detail + "; WrongGPU's handlers pass every call on unchanged; rr_other_plugin=take "
                                              "in wronggpu-rr.ini uses WrongGPU's instead)") == 1);
      if (ours) {
        const long before = os->rr_calls;
        CHECK_EQ(ours(kRR, nullptr), (WgResultCode)0);  // passed on: the other plugin answers RR
        CHECK(os->rr_calls == before + 1);
        CHECK_EQ(ours(kOther, nullptr), kIfaceAnswer);
      }
    }
  } else if (mode == "route-inline") {
    CHECK(count_starting(L, stem + "the export " + evaluate_name + " of " + iface_leaf +
                                " leads to wgtest_other_rr.dll (through 1 hook jump); nothing patched; ") == 1);
    CHECK(host::route_regs[0] == 0);  // no routes registered
  } else {
    CHECK(!"unknown mode");
  }
  (void)fo;
#endif
  TEST_SUMMARY(label.c_str());
}
