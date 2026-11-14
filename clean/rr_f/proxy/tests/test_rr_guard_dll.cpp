// SPDX-License-Identifier: MIT
// RR guard: end-to-end test of the coexistence guard in the hook-mode plugin (src/hook_dll.cpp + rr_guard.h).
// The executable is the fake game, as in test_hook_dll.cpp: it imports the RR path's functions by name from the stand-in
// interface (tests/gen_fake_original.py). The "other module" is our own stand-in, wgtest_other_rr.dll
// (tests/fake_other_rr.cpp), and further copies of it under other file names; no third-party file is used.
//   test_rr_guard_dll.exe <stage dir> <mode> [--take | --yield]
// WrongGPU RR switches itself off only on evidence that is positive by contract: a second copy of WrongGPU RR, or the
// host's plugin API refusing our registration (the refused route stays off; the API does not say why, so the log
// claims no other plugin). A named module on the call path is inconclusive: it is named in one
// line and WrongGPU RR stays active; --yield (rr_other_plugin=yield) makes it switch WrongGPU RR off, --take
// (rr_other_plugin=take) puts WrongGPU RR in front of it.
// Modes (import build). Named module on the path, RR stays active (no second RR provider in the innocent ones):
//   first        the stand-in RR plugin redirected the exe's import slots before ours loaded (wrapping)
//   inline       the stand-in RR plugin put a jmp [rip] on the evaluate export
//   relay        the same through a jmp rel32 to an anonymous relay ("through 2 hook jumps")
//   execonly     one import slot leads to an execute-only relay page that jumps into the stand-in (the relay is read)
//   wrapper      a game-owned wrapper DLL (wgtest_gamewrap.dll, innocent) wraps the exe's import slots
//   overlay      an overlay-style DLL (wgtest_overlay.dll, innocent) hooks the evaluate export through a relay
//   forward      the exe's evaluate slot is bound to an implementation DLL (wgtest_impl.dll, innocent), as the loader
//                binds a forwarded export
//   helper       a separate loader DLL calls InitializeASI and holds the host table; a host helper in a third image
//                (wgtest_hosthelper.dll, innocent) patched the slots
//   hostsplit    a separate loader DLL calls InitializeASI; the host DLL (wgtest_other_rr.dll, innocent) holds the API
//                table and patched the slots
//   heaptable    the same with the host's API table on the heap (no module)
//   helpertable  the same with the host's API table in a helper image (wgtest_hosttable.dll)
// Nothing could be told, RR stays active (one inconclusive line):
//   anon         one import slot leads to code outside every module that does not jump on
//   noaccess     one import slot leads to a PAGE_NOACCESS page ("could not be read"), no fault
//   guardpage    one import slot leads to a guard page: the same, and the page is still a guard page afterwards
//   exe          one import slot leads into the exe's own code (a separate loader DLL calls InitializeASI)
//   hops         one import slot leads through five relay jumps (more than kMaxHops) to the stand-in
// Later modules:
//   late         the stand-in RR plugin wrapped our hooks after InitializeASI: 16 threads make the first call at once:
//                one decision, one line naming it, WrongGPU RR stays active behind it
//   late-replace the stand-in replaced our hooks after InitializeASI: named at OptiPluginInit, stays active
// Positive by contract, WrongGPU RR (or that route) switches off:
//   provider     the stand-in holds the host's NGX RR provider: the host refuses ours, the NGX route stays off (the
//                line says the host refused, not why), the game-interface route stays
//   provider-refused the host refuses our NGX provider with no provider installed (an incomplete structure gives the
//                same false): the same line, no other plugin claimed, the game-interface route stays
//   owncopy      WrongGPU RR and four more copies under other file names: the others stay off and register nothing; the
//                copies' InitializeASI / OptiPluginInit run from threads while 16 game threads make the first call (every
//                copy writes the one log at once: no torn line); freeing an inactive copy writes no exit summary
//   clean        no other module: active; 16 threads make the first call at once: one re-check line; the scan cost
// --take: first / inline: WrongGPU RR in front; late: as without (never off). --yield: wrapper: off, nothing patched.
// Modes (built with WG_TEST_NOIMPORT: the exe does not link the interface, the host routes are used):
//   route        no other module: our routes continue to the interface
//   route-other  the stand-in's routes were registered first (under ours): RR ours, the other calls continue through
//                it (take: the same, "in front" line; yield: our routes pass every call on)
//   route-inline the stand-in patched the evaluate export before the host hooked the module: named, routes registered
//   route-refused the host refuses our routes (positive by contract): the game-interface route stays off
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
// lines that start with `start` and end with `end`
size_t count_line(const std::vector<std::string>& v, const std::string& start, const std::string& end) {
  size_t n = 0;
  for (const auto& s : v) {
    size_t at = 13;
    if (s.compare(at, 2, "f=") == 0) at = s.find(' ', at) + 1;
    n += s.size() >= at + start.size() + end.size() && s.compare(at, start.size(), start) == 0 &&
         s.compare(s.size() - end.size(), end.size(), end) == 0;
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
bool refuse_routes = false;  // route-refused: the host's API says no
bool refuse_ngx = false;     // provider-refused: the host refuses the NGX provider though no provider holds the slot
WgHostMenuPlugin* ngx_owner = nullptr;
void (*callback)(void*, uint32_t, uint32_t, uint32_t, void*) = nullptr;
void* callback_user = nullptr;
bool register_routes(WgHostMenuPlugin* p, const WgHostSlRoute* r, uint32_t n) {
  const int who = p == kOurs ? 0 : 1;
  ++route_regs[who];
  if (refuse_routes) return false;
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
  if (refuse_ngx || ngx_owner || !prov || !prov->create) return false;
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

template <typename F>
F fn(HMODULE m, const char* name) {
  return m ? reinterpret_cast<F>(GetProcAddress(m, name)) : nullptr;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 3) return 2;
  std::string stage = argv[1];
  if (stage.back() != '\\') stage += '\\';
  const std::string mode = argv[2];
  bool take = false, yield = false;
  for (int i = 3; i < argc; ++i) {
    take = take || strcmp(argv[i], "--take") == 0;
    yield = yield || strcmp(argv[i], "--yield") == 0;
  }
  const std::wstring ws = widen(stage);
  const std::wstring log_path = ws + (WG_BINDING_PROBE ? L"wronggpu-rr-probe.log" : L"wronggpu-rr.log");
  DeleteFileW(log_path.c_str());
  DeleteFileW((ws + L"wronggpu-rr.ini").c_str());
  if (take) write_text(ws + L"wronggpu-rr.ini", "rr_other_plugin = take\r\n");
  if (yield) write_text(ws + L"wronggpu-rr.ini", "rr_other_plugin = yield\r\n");
  CreateDirectoryW((ws + L"data").c_str(), nullptr);
  write_text(ws + L"data\\wronggpu.ini", "[wronggpu]\r\nrr_mode = accurate\r\n");
  SetEnvironmentVariableW(L"WG_DATA_DIR", (ws + L"data\\").c_str());
  SetEnvironmentVariableW(L"WG_RR_SLOT_CHECKS", L"0");
  const std::string label = std::string("test_rr_guard_dll (") + (WG_BINDING_PROBE ? "probe" : "release") + ", " + mode +
                            (take ? ", take" : "") + (yield ? ", yield" : "") + ")";

  HMODULE other = LoadLibraryW((ws + L"wgtest_other_rr.dll").c_str());
  CHECK(other != nullptr);
  if (!other) TEST_SUMMARY(label.c_str());
  using PatchIat = int (*)(HMODULE, const char*, int);
  using PatchExport = int (*)(HMODULE, const char*, int);
  using Innocent = void (*)(int);
  using HostTable = const WgHostMenuApi* (*)(const WgHostMenuApi*);
  auto or_state = fn<const OrState* (*)()>(other, "wgtest_or_state");
  auto or_patch_iat = fn<PatchIat>(other, "wgtest_or_patch_iat");
  auto or_set_slot = fn<int (*)(HMODULE, const char*, const char*, void*)>(other, "wgtest_or_set_slot");
  auto or_patch_export = fn<PatchExport>(other, "wgtest_or_patch_export");
  auto or_routes = fn<int (*)(const WgHostMenuApi*, WgHostMenuPlugin*)>(other, "wgtest_or_routes");
  auto or_ngx = fn<bool (*)(const WgHostMenuApi*, WgHostMenuPlugin*)>(other, "wgtest_or_register_ngx");
  auto or_host_table = fn<HostTable>(other, "wgtest_or_host_table");
  auto or_innocent = fn<Innocent>(other, "wgtest_or_innocent");
  const OrState* os = or_state();
  // another copy of the stand-in under another file name (its own state): the innocent layouts
  auto load_copy = [&](const wchar_t* file) {
    HMODULE m = LoadLibraryW((ws + file).c_str());
    CHECK(m != nullptr);
    if (Innocent f = fn<Innocent>(m, "wgtest_or_innocent")) f(1);
    return m;
  };
  HMODULE exe = GetModuleHandleW(nullptr);
  const std::string exe_leaf = "test_rr_guard_dll.exe";
  const std::string iface_leaf = kWgInterposerFile;
  const std::string evaluate_name = kWgRoleNames[kEvaluate];
  const std::string stem = "the host refused WrongGPU's NGX RR provider registration; the NGX route stays off (";
  const std::string fstem = "another module hooks the game's RR calls: ";
  const std::string tail_active = "; it is not known to be a Ray Reconstruction plugin: WrongGPU RR stays active "
                                  "(rr_other_plugin=yield in wronggpu-rr.ini keeps WrongGPU RR off in this case)";
  const std::string tail_take = "; rr_other_plugin=take: WrongGPU RR goes in front of it (it gets the game's other calls, "
                                "no RR work)";
  const std::string tail_yield_off = "; rr_other_plugin=yield: WrongGPU RR stays off (nothing patched)";
  const std::string tail_yield_pass = "; rr_other_plugin=yield: WrongGPU RR stays off (WrongGPU's handlers pass every "
                                      "call on unchanged)";
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
  const bool unclear_mode = mode == "anon" || mode == "noaccess" || mode == "guardpage" || mode == "exe" || mode == "hops";
  const bool host_layout = mode == "hostsplit" || mode == "heaptable" || mode == "helpertable";
  const bool loader_mode = host_layout || mode == "helper" || mode == "exe";
  const std::string self_slot = "the import slot ";
  // a named module on the path at load: its file name and the beginning of the line's detail
  std::string fmod, fdetail;
  // ---- other modules before ours
  void* odd_page = nullptr;
  if (mode == "first" || host_layout) {
    if (host_layout) or_innocent(1);  // the host itself (not an RR plugin)
    CHECK(or_patch_iat(exe, kWgInterposerFile, 1) >= 6);
    fmod = "wgtest_other_rr.dll";
    fdetail = self_slot;  // the first redirected slot in import order
  }
  if (mode == "inline" || mode == "relay") {
    CHECK(or_patch_export(iface, evaluate_name.c_str(), mode == "relay") == 1);
    fmod = "wgtest_other_rr.dll";
    fdetail = "the export " + evaluate_name + " of " + iface_leaf + " leads to wgtest_other_rr.dll (through " +
              (mode == "relay" ? "2 hook jumps)" : "1 hook jump)");
  }
  if (mode == "wrapper") {
    CHECK(fn<PatchIat>(load_copy(L"wgtest_gamewrap.dll"), "wgtest_or_patch_iat")(exe, kWgInterposerFile, 1) >= 6);
    fmod = "wgtest_gamewrap.dll";
    fdetail = self_slot;
  }
  if (mode == "helper") {
    CHECK(fn<PatchIat>(load_copy(L"wgtest_hosthelper.dll"), "wgtest_or_patch_iat")(exe, kWgInterposerFile, 1) >= 6);
    fmod = "wgtest_hosthelper.dll";
    fdetail = self_slot;
  }
  if (mode == "overlay") {
    CHECK(fn<PatchExport>(load_copy(L"wgtest_overlay.dll"), "wgtest_or_patch_export")(iface, evaluate_name.c_str(), 1) == 1);
    fmod = "wgtest_overlay.dll";
    fdetail = "the export " + evaluate_name + " of " + iface_leaf + " leads to wgtest_overlay.dll (through 2 hook jumps)";
  }
  if (mode == "forward") {
    auto bind_one = fn<int (*)(HMODULE, const char*, const char*)>(load_copy(L"wgtest_impl.dll"), "wgtest_or_bind_one");
    CHECK(bind_one && bind_one(exe, kWgInterposerFile, evaluate_name.c_str()) == 1);
    fmod = "wgtest_impl.dll";
    fdetail = self_slot + evaluate_name + " of " + exe_leaf + " leads to wgtest_impl.dll";
  }
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
  if (mode == "exe") CHECK(or_set_slot(exe, kWgInterposerFile, evaluate_name.c_str(), reinterpret_cast<void*>(&read_lines)) == 1);
  if (mode == "hops" || mode == "execonly") {
    // relay jumps (jmp [rip+0]; dq target) in code outside every module that lead to a function of the stand-in
    uint8_t* p = static_cast<uint8_t*>(VirtualAlloc(nullptr, 4096, MEM_RESERVE | MEM_COMMIT, PAGE_EXECUTE_READWRITE));
    CHECK(p != nullptr);
    const void* to_other = reinterpret_cast<const void*>(GetProcAddress(other, "wgtest_or_state"));
    const int jumps = mode == "hops" ? 5 : 1;  // five is more than kMaxHops (4)
    for (int i = 0; i < jumps; ++i) {
      const void* t = i + 1 < jumps ? static_cast<const void*>(p + 32 * (i + 1)) : to_other;
      p[32 * i] = 0xFF;
      p[32 * i + 1] = 0x25;
      memset(p + 32 * i + 2, 0, 4);
      memcpy(p + 32 * i + 6, &t, sizeof t);
    }
    if (mode == "execonly") {
      DWORD old = 0;
      CHECK(VirtualProtect(p, 4096, PAGE_EXECUTE, &old));  // execute-only: the guard still reads the relay
      fmod = "wgtest_other_rr.dll";
      fdetail = self_slot + evaluate_name + " of " + exe_leaf + " leads to wgtest_other_rr.dll (through 1 hook jump)";
    }
    odd_page = p;
    CHECK(or_set_slot(exe, kWgInterposerFile, evaluate_name.c_str(), p) == 1);
  }
  if (mode == "provider") CHECK(or_ngx(&api, host::kTheirs));
  if (mode == "provider-refused") host::refuse_ngx = true;

  // ---- WrongGPU RR (the host loads it and runs its entry)
  load_asi(widen(kWgHookFile), &init, &plugin_init);
  if (!init || !plugin_init) TEST_SUMMARY(label.c_str());
  const WgHostMenuApi* host_api = &api;  // the table OptiPluginInit gets
  if (loader_mode) {
    // a separate ASI loader (another module) calls InitializeASI: a second instance of the stand-in DLL plays the loader
    // (in a game the host is a DLL; when this test exe called InitializeASI itself, the exe would be the host)
    HMODULE loader = LoadLibraryW((ws + L"wgtest_loader.dll").c_str());
    CHECK(loader != nullptr);
    auto call_init = fn<void (*)(void (*)())>(loader, "wgtest_or_call_init");
    CHECK(call_init != nullptr && or_host_table != nullptr);
    if (!call_init || !or_host_table) TEST_SUMMARY(label.c_str());
    call_init(init);
    if (mode == "hostsplit") host_api = or_host_table(&api);                // in the host DLL that patched
    if (mode == "heaptable") host_api = new WgHostMenuApi(api);             // on the heap: no module
    if (mode == "helpertable")                                               // in a helper image
      host_api = fn<HostTable>(load_copy(L"wgtest_hosttable.dll"), "wgtest_or_host_table")(&api);
    if (mode == "helper") host_api = fn<HostTable>(loader, "wgtest_or_host_table")(&api);  // in the loader itself
    CHECK(host_api != nullptr);
  } else {
    init();  // (owncopy: this copy is the active one, the others follow below)
  }
  struct Copy {
    HMODULE module = nullptr;
    void (*init)() = nullptr;
    WgHostMenuInitFn plugin_init = nullptr;
  };
  std::vector<Copy> copies;  // owncopy: more copies of WrongGPU RR under other file names (build.py stages them)
  if (mode == "owncopy")
    for (const wchar_t* f : {L"WrongGPU-RR-copy.asi", L"WrongGPU-RR-copy2.asi", L"WrongGPU-RR-copy3.asi", L"WrongGPU-RR-copy4.asi"}) {
      Copy c;
      c.module = load_asi(f, &c.init, &c.plugin_init);
      CHECK(c.module != nullptr && c.init != nullptr && c.plugin_init != nullptr);
      if (c.module) copies.push_back(c);
    }
  // ---- the stand-in RR plugin after ours
  if (mode == "late") CHECK(or_patch_iat(exe, kWgInterposerFile, 1) >= 6);          // wraps our handlers
  if (mode == "late-replace") CHECK(or_patch_iat(exe, kWgInterposerFile, 0) >= 6);  // replaces them
  // OptiPluginInit (late-replace: the check that finds it, as no call reaches our handlers any more); "late" makes the
  // first call before it, so that the first call decides
  if (mode == "owncopy") {
    // every copy at once: the active copy's OptiPluginInit and the game's first call (16 threads), and in each other
    // copy InitializeASI (two threads: the second waits for the first) followed by its OptiPluginInit. All of them
    // write wronggpu-rr.log at the same moment; no line may be torn, the other copies register nothing.
    std::atomic<int> go{0}, answered{0}, registered{0};
    std::vector<std::thread> th;
    auto spawn = [&](auto f) {
      th.emplace_back([&go, f] {
        while (!go.load()) std::this_thread::yield();
        f();
      });
    };
    spawn([&] { plugin_init(host_api, host::kOurs); });
    for (int i = 0; i < 16; ++i) spawn([&] { answered += slIsFeatureSupported(kOther, nullptr) == kIfaceAnswer; });
    for (const Copy& c : copies) {
      spawn([&c, &registered, host_api] {
        c.init();
        registered += c.plugin_init(host_api, host::kOurs) ? 1 : 0;
      });
      spawn([&c] { c.init(); });
    }
    go.store(1);
    for (auto& t : th) t.join();
    CHECK_EQ(answered.load(), 16);
    CHECK_EQ(registered.load(), 0);  // the other copies register nothing
  } else if (mode != "late") {
    plugin_init(host_api, host::kOurs);
  }

  // ---- the game's calls
  if (mode == "clean" || mode == "late" || mode == "provider" || mode == "provider-refused") {
    // 16 threads make the game's first interface call at once
    const int ok = at_once(16, [] { return slIsFeatureSupported(kOther, nullptr); }, kIfaceAnswer);
    CHECK_EQ(ok, 16);
  }
  if (mode == "late") plugin_init(host_api, host::kOurs);
  // the RR query: WrongGPU's core answers it itself (it never reaches the interface); the stand-in RR plugin counts the
  // ones it answered
  const int fo_before = fo->calls;
  const long rr_before = os->rr_calls;
  const WgResultCode rr = slIsFeatureSupported(kRR, nullptr);
  (void)rr;
  const bool rr_by_other = os->rr_calls == rr_before + 1;
  const bool rr_ours = fo->calls == fo_before && os->rr_calls == rr_before;
  std::vector<std::string> L = read_lines(log_path);
  printf("log lines %zu, torn %zu\n", L.size(), torn_lines(L));
  CHECK(torn_lines(L) == 0);  // 0.6.1: one append call per line, also with several copies
  const double load_cost = cost_of(L, "hook: coexistence check at load: ");
  CHECK(load_cost >= 0.0 && load_cost < 5.0);
  printf("coexistence check at load: %.3f ms\n", load_cost);
  // the check's result line (the stay-on line is a second line of the same check); the load check runs once
  CHECK(count(L, "hook: coexistence check at load: ") - count(L, "hook: coexistence check at load: inconclusive") == 1u);
  const std::string patched = std::string("hook: patched IAT slot ") + kWgRoleNames[kIsSupported] + " in ";
  const std::string clean_recheck = "hook: coexistence re-check at the game's first call: no other Ray Reconstruction "
                                    "plugin found (";

  if (mode == "clean" || mode == "owncopy") {
    CHECK(count(L, "hook: coexistence check at load: no other Ray Reconstruction plugin found (") == 1);
    CHECK(count(L, clean_recheck) == 1);
    CHECK(count(L, "another Ray Reconstruction plugin is active") == 0);
    CHECK(count(L, fstem) == 0);
    CHECK(count(L, "hook: coexistence check at load: inconclusive") == 0);
    CHECK(count(L, patched) == 1);  // only one copy patched
    CHECK(rr_ours);
    if (mode == "owncopy") {
      CHECK(count_starting(L, "a second copy of WrongGPU RR is loaded (") == copies.size());  // one line per other copy
      CHECK(count(L, std::string(kWgHookFile) + " is active); this copy (") == copies.size());
      CHECK(count(L, "WrongGPU-RR-copy.asi) stays off") == 1);
      CHECK(count(L, "WrongGPU-RR-copy2.asi) stays off") == 1);
      CHECK(count(L, "WrongGPU-RR-copy3.asi) stays off") == 1);
      CHECK(count(L, "WrongGPU-RR-copy4.asi) stays off") == 1);
      // an inactive copy has no calls to summarise: freeing it (DLL_PROCESS_DETACH under the loader lock) writes no
      // exit summary (the line would say the calls never reached the plugin); the log lock is only tried there
      for (const Copy& c : copies) CHECK(FreeLibrary(c.module));
      CHECK(count(read_lines(log_path), "diag: summary") == 0);
    }
#if !WG_BINDING_PROBE
    CHECK(count(L, "NGX: RR provider registered with the host") == 1);
    CHECK(host::ngx_regs == 1 && host::ngx_owner == host::kOurs);
    CHECK(GetModuleHandleW(L"wronggpu_runtime.dll") != nullptr);  // active: the core loaded the RR runtime
#endif
  } else if (!fmod.empty()) {
    // a named module on the path at load: one line names it and the place
    CHECK(count(L, "hook: coexistence check at load: another module on the RR path (inconclusive) (") == 1);
    CHECK(count_starting(L, fstem) == 1);
    CHECK(count(L, "another Ray Reconstruction plugin is active") == 0);  // nothing claims an RR plugin
    if (yield) {
      // rr_other_plugin=yield: WrongGPU RR stays off, nothing patched, the core and the runtime are not loaded
      CHECK(count_line(L, fstem + fmod + " (" + fdetail, tail_yield_off) == 1);
      CHECK(count(L, "hook: patched IAT slot") == 0);
      CHECK(count(L, "hook: coexistence re-check") == 0);
      CHECK(!rr_ours && fo->last_role && strcmp(fo->last_role, "is-supported") == 0 && fo->feature == kRR);
      CHECK_EQ(slIsFeatureSupported(kOther, nullptr), kIfaceAnswer);
#if !WG_BINDING_PROBE
      CHECK(count(L, "NGX: RR provider not registered: WrongGPU RR stays off in this process") == 1);
      CHECK(host::ngx_owner == nullptr);
      CHECK(GetModuleHandleW(L"wronggpu_runtime.dll") == nullptr);
#endif
    } else {
      // off (the default) and take: WrongGPU RR stays active and answers RR itself
      CHECK(count_line(L, fstem + fmod + " (" + fdetail, take ? tail_take : tail_active) == 1);
      CHECK(count(L, "stays off") == 0);
      CHECK(count(L, patched) == 1);
      CHECK(rr_ours);
      CHECK(count(L, clean_recheck) == 1);  // the export hooks were named at load: nothing new at the first call
      CHECK(count(L, "took the RR calls after WrongGPU") == 0);
      if (take && mode == "first") {  // the other calls go on through the other module (it sees them, no RR work)
        const long before = os->calls[kIsSupported];
        CHECK_EQ(slIsFeatureSupported(kOther, nullptr), kIfaceAnswer);
        CHECK(os->calls[kIsSupported] == before + 1);
      } else {
        CHECK_EQ(slIsFeatureSupported(kOther, nullptr), kIfaceAnswer);
      }
#if !WG_BINDING_PROBE
      CHECK(count(L, "NGX: RR provider registered with the host") == 1);
      CHECK(host::ngx_owner == host::kOurs);
      CHECK(GetModuleHandleW(L"wronggpu_runtime.dll") != nullptr);  // active: the core loaded the RR runtime
#endif
    }
  } else if (unclear_mode) {
    // nothing could be told: logged once as inconclusive, WrongGPU RR stays active (it patched, RR is its own)
    const std::string unclear =
        mode == "anon"       ? self_slot + evaluate_name + " of " + exe_leaf + " leads to code outside every module"
        : mode == "exe"      ? self_slot + evaluate_name + " of " + exe_leaf + " leads into the exe's own code"
        : mode == "hops"     ? self_slot + evaluate_name + " of " + exe_leaf + " leads through more than 4 hook jumps"
                             : self_slot + evaluate_name + " of " + exe_leaf + " could not be read";
    CHECK(count(L, "hook: coexistence check at load: no other Ray Reconstruction plugin found (") == 1);
    CHECK(count_starting(L, "hook: coexistence check at load: inconclusive for 1 address(es) (first: " + unclear +
                            "); no other plugin was positively found, WrongGPU RR stays active") == 1);
    CHECK(count(L, "another Ray Reconstruction plugin is active") == 0);
    CHECK(count(L, fstem) == 0);
    CHECK(count(L, "stays off") == 0);
    CHECK(count(L, patched) == 1);
    CHECK(count(L, clean_recheck) == 1);
    CHECK(rr_ours);
    CHECK_EQ(slIsFeatureSupported(kOther, nullptr), kIfaceAnswer);
    if (mode == "guardpage") {
      MEMORY_BASIC_INFORMATION m;
      CHECK(VirtualQuery(odd_page, &m, sizeof m) && (m.Protect & PAGE_GUARD));  // the check did not touch it
    }
#if !WG_BINDING_PROBE
    CHECK(count(L, "NGX: RR provider registered with the host") == 1);
    CHECK(host::ngx_owner == host::kOurs);
#endif
  } else if (late) {
    // a module that took the calls after us: named once, WrongGPU RR stays active behind it (take: the same, never off)
    CHECK(count(L, "hook: coexistence check at load: no other Ray Reconstruction plugin found (") == 1);
    CHECK(count(L, patched) == 1);
    const std::string why = mode == "late" ? "the game's first call" : "OptiPluginInit";
    CHECK(count(L, "hook: coexistence re-check at " + why + ": another module took the RR calls after WrongGPU (") == 1);
    CHECK(count(L, "hook: coexistence re-check") == 1);  // one decision, one line, even with 16 threads at once
    CHECK(count_line(L, fstem + "wgtest_other_rr.dll (the import slot ",
                     " (WrongGPU's hook) leads to wgtest_other_rr.dll)" + tail_active) == 1);
    CHECK(count_starting(L, fstem) == 1);
    CHECK(count(L, "stays off") == 0);
    CHECK(rr_by_other);  // the stand-in is in front and answers RR itself
    // a call the stand-in passes on reaches the interface unchanged (through our active handler in "late")
    const int before = fo->calls;
    CHECK_EQ(slEvaluateFeature(kOther, nullptr, nullptr, 0, nullptr), kIfaceAnswer);
    CHECK(fo->calls == before + 1 && fo->last_role && strcmp(fo->last_role, "evaluate") == 0);
    CHECK_EQ(at_once(8, [] { return slIsFeatureSupported(kOther, nullptr); }, kIfaceAnswer), 8);  // consistent after
    CHECK(count(read_lines(log_path), "hook: coexistence re-check") == 1);
#if !WG_BINDING_PROBE
    CHECK(count(read_lines(log_path), "NGX: RR provider registered with the host") == 1);
    CHECK(host::ngx_owner == host::kOurs);
#endif
  } else if (mode == "provider" || mode == "provider-refused") {
#if !WG_BINDING_PROBE
    // the host's API refuses our provider: the NGX route stays off (positive by contract). Its false does not say why
    // (an occupied slot, an incomplete structure), so the line says only that the host refused
    CHECK(count_starting(L, stem + "the game-interface route is decided on its own)") == 1);
    CHECK(count(L, "another Ray Reconstruction plugin is active") == 0 && count(L, "another provider") == 0);
    if (mode == "provider") CHECK(host::ngx_owner == host::kTheirs && host::ngx_regs == 2);
    else CHECK(host::ngx_owner == nullptr && host::ngx_regs == 1);  // no provider anywhere
    CHECK(count(L, patched) == 1);  // the game-interface route stays ours
    CHECK(count(L, stem) == 1);
    CHECK(rr_ours);
#endif
  } else {
    CHECK(!"unknown mode");
  }
  printf("%s: other plugin answered %ld RR call(s)\n", label.c_str(), os->rr_calls);
#else  // WG_TEST_NOIMPORT: host routes
  (void)or_patch_iat;
  (void)or_set_slot;
  (void)or_ngx;
  (void)or_host_table;
  (void)or_innocent;
  (void)load_copy;
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
  if (mode == "route-refused") host::refuse_routes = true;
  if (host::callback) host::callback(iface, 2, 14, 1, host::callback_user);
  std::vector<std::string> L = read_lines(log_path);
  printf("log lines %zu, torn %zu\n", L.size(), torn_lines(L));
  CHECK(torn_lines(L) == 0);  // 0.6.1: one append call per line, also with two copies
  CHECK(count(L, "waiting for the host's game-interface routes") == 1);
  CHECK(count(L, "another Ray Reconstruction plugin is active") == 0);
  auto ours = reinterpret_cast<IsSupportedFn>(host::handler_of_plugin[0][kWgRoleNames[kIsSupported]]);
  if (mode == "route") {
    CHECK(count(L, "hook: coexistence check before the host routes: no other Ray Reconstruction plugin found (") == 1);
    CHECK(count(L, "hook: coexistence check of the host routes: no other Ray Reconstruction plugin found (") == 1);
    CHECK(count(L, fstem) == 0);
    CHECK(ours != nullptr);
    if (ours) {
      CHECK_EQ(ours(kOther, nullptr), kIfaceAnswer);  // continues to the interface
      const long before = os->rr_calls;
      ours(kRR, nullptr);
      CHECK(os->rr_calls == before);
    }
  } else if (mode == "route-other") {
    const std::string start = fstem + "wgtest_other_rr.dll (the host's next function of " +
                              std::string(kWgRoleNames[kIsSupported]) + " leads to wgtest_other_rr.dll)";
    CHECK(count(L, "hook: coexistence check of the host routes: another module on the RR path (inconclusive) (") == 1);
    CHECK(ours != nullptr);
    if (yield) {
      CHECK(count_line(L, start, tail_yield_pass) == 1);
      if (ours) {
        const long before = os->rr_calls;
        CHECK_EQ(ours(kRR, nullptr), (WgResultCode)0);  // passed on: the stand-in answers RR
        CHECK(os->rr_calls == before + 1);
        CHECK_EQ(ours(kOther, nullptr), kIfaceAnswer);
      }
    } else {
      CHECK(count_line(L, start, take ? tail_take : tail_active) == 1);
      CHECK(count(L, "stays off") == 0);
      if (ours) {
        const long before = os->rr_calls, calls = os->calls[kIsSupported];
        ours(kRR, nullptr);
        CHECK(os->rr_calls == before);  // RR is WrongGPU's
        CHECK_EQ(ours(kOther, nullptr), kIfaceAnswer);  // the other calls continue through the stand-in
        CHECK(os->calls[kIsSupported] == calls + 1);
      }
    }
  } else if (mode == "route-inline") {
    // named before the routes: RR stays active, our routes are registered in front
    CHECK(count_line(L, fstem + "wgtest_other_rr.dll (the export " + evaluate_name + " of " + iface_leaf +
                            " leads to wgtest_other_rr.dll (through 1 hook jump))",
                     tail_active) == 1);
    CHECK(host::route_regs[0] == 1 && ours != nullptr);
    CHECK(count(L, "stays off") == 0);
    if (ours) CHECK_EQ(ours(kOther, nullptr), kIfaceAnswer);
  } else if (mode == "route-refused") {
    // the host's API refused our registration (positive by contract): the game-interface route stays off
    CHECK(count_starting(L, "hook: the host refused our game-interface routes; nothing was patched: WrongGPU RR stays "
                            "off on the game-interface route") == 1);
    CHECK(host::route_regs[0] == 1 && ours == nullptr);
    CHECK(count(L, "hook: coexistence check of the host routes") == 0);
  } else {
    CHECK(!"unknown mode");
  }
  (void)fo;
#endif
  TEST_SUMMARY(label.c_str());
}
