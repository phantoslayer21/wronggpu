// SPDX-License-Identifier: MIT
// End-to-end test of the built proxy library with the real export table of its Streamline version (annex A9.3):
// export table (every name at its ordinal, the own export above), every name a supported game imports (A9.2), the
// renamed original found from the proxy's own file name, forwarding of every export the proxy does not implement
// with its arguments unchanged, the version values of A9.1 forwarded unchanged, the RR call path through the
// role exports and the feature functions (requested with the real names of binding/rr-functions.json, taken from
// the generated binding) into a fake runtime library, other features forwarded, the A10.1 frame-token layout, the
// settings contract and the ASI side finding the proxy in the process; the version resource (the game interposer
// version, WrongGPU's strings) and the load diagnostics in wronggpu-rr.log (the init export hooked by a stand-in at
// the first call, which is a forwarded one here; the init entry logged). The proxy is staged as the install tool
// places it: under the game's interposer file name, next to the renamed fake original.
//   test_proxy_dll.exe <stage dir> <proxy file name in the stage> [--no-runtime] [--imports <file>]
// The imports file (written by tools/build.py from binding/game-imports.json): per game of this version a line
// "game <label>", a line "version <hex value the game passes to <init>>" and one imported name per line.
#define WIN32_LEAN_AND_MEAN
#include <windows.h>

#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include <ctype.h>

#include <fstream>
#include <string>
#include <vector>

#include "../asi/wg_rr_controls.h"
#include "../include/rr_proxy_abi.h"
#include "fake_runtime.h"
#include "test_util.h"
#include "wg_binding.inc"  // generated from the same binding table as the proxy under test
#include "dll_diag.h"

struct FoState { int calls; int last_index; const char* last_role; uint32_t feature; uint64_t a1, a3, a5, a6;
                 double a2, a4, a7; uint64_t u64; const void* ptr; void* cmd; uint32_t count; };

enum { R_INIT = 0, R_SHUTDOWN, R_IS_SUPPORTED, R_IS_LOADED, R_SET_LOADED, R_REQUIREMENTS, R_VERSION, R_GET_FUNCTION,
       R_ALLOCATE, R_FREE, R_SET_DEVICE, R_SET_VULKAN_INFO, R_TAG, R_TAG_FOR_FRAME, R_CONSTANTS, R_NEW_FRAME_TOKEN,
       R_EVALUATE, R_UPGRADE_INTERFACE, R_NATIVE_INTERFACE };

static std::wstring widen(const char* s) {
  std::wstring w;
  while (*s) w.push_back((wchar_t)*s++);
  return w;
}

// Read the export directory of a loaded module: name -> ordinal.
static bool read_exports(HMODULE m, std::vector<std::pair<std::string, unsigned>>* out) {
  const uint8_t* base = reinterpret_cast<const uint8_t*>(m);
  const IMAGE_DOS_HEADER* dos = reinterpret_cast<const IMAGE_DOS_HEADER*>(base);
  const IMAGE_NT_HEADERS64* nt = reinterpret_cast<const IMAGE_NT_HEADERS64*>(base + dos->e_lfanew);
  const IMAGE_DATA_DIRECTORY& d = nt->OptionalHeader.DataDirectory[IMAGE_DIRECTORY_ENTRY_EXPORT];
  if (!d.VirtualAddress) return false;
  const IMAGE_EXPORT_DIRECTORY* ed = reinterpret_cast<const IMAGE_EXPORT_DIRECTORY*>(base + d.VirtualAddress);
  const DWORD* names = reinterpret_cast<const DWORD*>(base + ed->AddressOfNames);
  const WORD* ords = reinterpret_cast<const WORD*>(base + ed->AddressOfNameOrdinals);
  for (DWORD i = 0; i < ed->NumberOfNames; ++i)
    out->push_back({reinterpret_cast<const char*>(base + names[i]), (unsigned)(ords[i] + ed->Base)});
  return true;
}

template <typename F>
static F role(HMODULE p, int r) {
  return reinterpret_cast<F>(GetProcAddress(p, kWgRoleNames[r]));
}

// The stand-in hook's target (never called: the hook is removed before <init>).
static WgResultCode hooked_init(const WgPreferences*, uint64_t) { return (WgResultCode)99; }

struct GameImports { std::string game; uint64_t version = 0; std::vector<std::string> names; };

static std::vector<GameImports> read_imports(const char* path) {
  std::vector<GameImports> out;
  std::ifstream f(path);
  std::string line;
  while (std::getline(f, line)) {
    while (!line.empty() && (line.back() == '\r' || line.back() == ' ')) line.pop_back();
    if (line.rfind("game ", 0) == 0) {
      out.push_back(GameImports());
      out.back().game = line.substr(5);
    } else if (line.rfind("version ", 0) == 0 && !out.empty()) {
      out.back().version = _strtoui64(line.c_str() + 8, nullptr, 16);
    } else if (!line.empty() && !out.empty()) {
      out.back().names.push_back(line);
    }
  }
  return out;
}

int main(int argc, char** argv) {
  if (argc < 3) return 2;
  bool no_runtime = false;
  const char* imports_path = nullptr;
  for (int i = 3; i < argc; ++i) {
    if (strcmp(argv[i], "--no-runtime") == 0) no_runtime = true;
    else if (strcmp(argv[i], "--imports") == 0 && i + 1 < argc) imports_path = argv[++i];
  }
  std::string stage = argv[1];
  if (stage.back() != '\\' && stage.back() != '/') stage += '\\';
  std::wstring wstage = widen(stage.c_str());
  DeleteFileW((wstage + L"wronggpu-rr.ini").c_str());
  DeleteFileW((wstage + L"wronggpu-rr.log").c_str());
  const std::wstring proxy_file = widen(argv[2]);
#if WG_BINDING_ORIGINAL_FROM_STEM
  const std::wstring original_file = proxy_file.substr(0, proxy_file.find_last_of(L'.')) + kWgOriginalFile;
#else
  const std::wstring original_file = kWgOriginalFile;
#endif

  CHECK(_stricmp(argv[2], kWgInterposerFile) == 0);  // staged under the game's interposer file name
  HMODULE proxy = LoadLibraryW((wstage + proxy_file).c_str());
  CHECK(proxy != nullptr);
  if (!proxy) TEST_SUMMARY("test_proxy_dll");
  // the original is loaded lazily, never under the loader lock
  CHECK(GetModuleHandleW(original_file.c_str()) == nullptr);
  // ---- version resource: the game interposer version, WrongGPU's own strings
  wgtest::check_version_resource(wstage + proxy_file, L"release", kWgInterposerFile);

  // ---- export table: every name of the original at its alphabetical ordinal, plus the own export above
  std::vector<std::pair<std::string, unsigned>> ex;
  CHECK(read_exports(proxy, &ex));
  CHECK_EQ(ex.size(), (size_t)WG_BINDING_EXPORT_COUNT + 1);
  size_t matched = 0;
  for (const WgExportInfo& e : kWgExports)
    for (auto& p : ex)
      if (p.first == e.name) { matched += p.second == e.ordinal; break; }
  CHECK_EQ(matched, (size_t)WG_BINDING_EXPORT_COUNT);
  bool own_ok = false;
  for (auto& p : ex) own_ok |= p.first == WG_RR_CONTRACT_EXPORT && p.second == WG_BINDING_OWN_ORDINAL;
  CHECK(own_ok);
  for (const WgExportInfo& e : kWgExports)
    CHECK(GetProcAddress(proxy, MAKEINTRESOURCEA(e.ordinal)) == GetProcAddress(proxy, e.name));
  // A9.2: ordinal = 1-based ASCII rank, base 1, no gaps; the own export directly above
  for (size_t i = 0; i < WG_BINDING_EXPORT_COUNT; ++i) {
    CHECK_EQ(kWgExports[i].ordinal, (unsigned)(i + 1));
    if (i) CHECK(strcmp(kWgExports[i - 1].name, kWgExports[i].name) < 0);
  }
  CHECK_EQ(WG_BINDING_OWN_ORDINAL, WG_BINDING_EXPORT_COUNT + 1);

  // ---- A9.2: every name each game of this interposer version imports is exported (by name)
  std::vector<GameImports> games = imports_path ? read_imports(imports_path) : std::vector<GameImports>();
  if (imports_path) CHECK(!games.empty());
  for (const GameImports& g : games) {
    CHECK(!g.names.empty() && g.version != 0);
    size_t found = 0;
    for (const std::string& name : g.names) {
      bool in_table = false;
      for (const WgExportInfo& e : kWgExports) in_table |= name == e.name;
      bool ok = in_table && GetProcAddress(proxy, name.c_str()) != nullptr;
      if (!ok) fprintf(stderr, "%s imports %s: not exported\n", g.game.c_str(), name.c_str());
      found += ok;
    }
    CHECK_EQ(found, g.names.size());
  }

  // ---- forwarding of every non-intercepted export, arguments unchanged (integer, float and stack arguments)
  typedef uint64_t (*Generic)(uint64_t, double, uint64_t, double, uint64_t, uint64_t, double);
  HMODULE orig = nullptr;
  FoState* st = nullptr;
  size_t forwarded_ok = 0, forwarded_total = 0;
  // the first call of all (a forwarded one) runs while a stand-in hooks the init export (load diagnostics)
  wgtest::Hook init_hook = wgtest::hook((void*)GetProcAddress(proxy, kWgRoleNames[R_INIT]), (const void*)&hooked_init,
                                        wgtest::HookForm::kMovRax);
  std::string first_call;
  for (int pass = 0; pass < 2; ++pass) {  // first call resolves the slot, the second takes the fast path
    for (size_t i = 0; i < WG_BINDING_EXPORT_COUNT; ++i) {
      const WgExportInfo& e = kWgExports[i];
      if (e.forward_slot < 0 || e.role == R_NATIVE_INTERFACE) continue;
      Generic f = reinterpret_cast<Generic>(GetProcAddress(proxy, e.name));
      uint64_t a1 = 0x1234567800000000ull + i, a3 = i * 7 + pass, a5 = ~(uint64_t)i, a6 = 42 + i;
      double a2 = 1.5 * (double)i, a4 = -0.25 - (double)i, a7 = 1e10 + (double)i;
      uint64_t r = f(a1, a2, a3, a4, a5, a6, a7);
      if (init_hook.at) {
        wgtest::unhook(init_hook);
        init_hook.at = nullptr;
        first_call = e.name;
      }
      if (!orig) {
        orig = GetModuleHandleW(original_file.c_str());
        CHECK(orig != nullptr);
        if (!orig) TEST_SUMMARY("test_proxy_dll");
        st = reinterpret_cast<FoState* (*)(void)>(GetProcAddress(orig, "wgtest_fo_state"))();
      }
      bool ok = st->last_index == (int)i && st->a1 == a1 && st->a2 == a2 && st->a3 == a3 && st->a4 == a4 &&
                st->a5 == a5 && st->a6 == a6 && st->a7 == a7 && r == a1 * 31u + i;
      forwarded_total++;
      forwarded_ok += ok;
      if (!ok) fprintf(stderr, "forwarding mismatch at %s\n", e.name);
    }
  }
  CHECK_EQ(forwarded_ok, forwarded_total);
  CHECK(forwarded_total == 2 * (WG_BINDING_FORWARD_COUNT - 1));
  wchar_t opath[MAX_PATH];
  GetModuleFileNameW(orig, opath, MAX_PATH);
  CHECK(_wcsicmp(opath, (wstage + original_file).c_str()) == 0);  // full path in the proxy's directory

  // ---- the RR path through the role exports
  FakeRuntime* rt = nullptr;
  auto init = role<WgResultCode (*)(const WgPreferences*, uint64_t)>(proxy, R_INIT);
  auto is_supported = role<WgResultCode (*)(uint32_t, const WgAdapterInfo*)>(proxy, R_IS_SUPPORTED);
  auto set_device = role<WgResultCode (*)(void*)>(proxy, R_SET_DEVICE);
  auto get_function = role<WgResultCode (*)(uint32_t, const char*, void**)>(proxy, R_GET_FUNCTION);
  auto tag = role<WgResultCode (*)(const WgViewport*, const WgTag*, uint32_t, void*)>(proxy, R_TAG);
  auto constants = role<WgResultCode (*)(const WgConstants*, const WgFrameToken*, const WgViewport*)>(proxy, R_CONSTANTS);
  auto new_token = role<WgResultCode (*)(WgFrameToken**, const uint32_t*)>(proxy, R_NEW_FRAME_TOKEN);
  auto evaluate = role<WgResultCode (*)(uint32_t, const WgFrameToken*, const WgHeader* const*, uint32_t, void*)>(proxy, R_EVALUATE);
  auto shutdown = role<WgResultCode (*)(void)>(proxy, R_SHUTDOWN);
  auto version = role<WgResultCode (*)(uint32_t, WgFeatureVersion*)>(proxy, R_VERSION);
  CHECK_EQ(is_supported(1001, nullptr), 23);  // before <init>
  CHECK_EQ(st->last_role != nullptr && strcmp(st->last_role, "is-supported") == 0, false);
  uint32_t feats[2] = {0, 1001};
  WgPreferences pr;
  memset(&pr, 0, sizeof pr);
  pr.features = feats;
  pr.feature_count = 2;
  // A9.1: the version value a game of this interposer version passes (game B: 2.11.1 with a 2.14.1 interposer),
  // forwarded unchanged; without an imports file the interposer's own version
  uint64_t ver = (2ull << 48) | ((uint64_t)WG_BINDING_API_MINOR << 32) | (1ull << 16) | WG_BINDING_VERSION_MAGIC;
  if (!games.empty()) ver = games.back().version;
  CHECK_EQ(WG_API_VERSION_MAGIC(ver), (uint32_t)WG_BINDING_VERSION_MAGIC);
  CHECK_EQ(init(&pr, ver), 0);
  HMODULE rtm = GetModuleHandleW(WG_RR_RUNTIME_FILE);  // loaded from the proxy's directory at first role call
  if (!no_runtime) {
    CHECK(rtm != nullptr);
    if (rtm) rt = reinterpret_cast<FakeRuntime* (*)(void)>(GetProcAddress(rtm, "wgtest_fr_state"))();
  } else {
    CHECK(rtm == nullptr);
  }
  CHECK(strcmp(st->last_role, "init") == 0 && st->ptr == &pr && st->u64 == ver);
  {
    WgFeatureVersion fv;
    memset(&fv, 0, sizeof fv);
    fv.h.version = 1;
    CHECK_EQ(version(1001, &fv), 0);  // the game's value as passed, WrongGPU's release version
    CHECK(fv.api_version[0] == WG_API_VERSION_MAJOR(ver) && fv.api_version[1] == WG_API_VERSION_MINOR(ver) &&
          fv.api_version[2] == WG_API_VERSION_PATCH(ver));
    CHECK(fv.runtime_version[0] == kWgReleaseVersion[0] && fv.runtime_version[1] == kWgReleaseVersion[1] &&
          fv.runtime_version[2] == kWgReleaseVersion[2]);
    CHECK_EQ(version(0, &fv), 100);  // other features forwarded
    CHECK(strcmp(st->last_role, "version") == 0);
  }
  CHECK_EQ(is_supported(1001, nullptr), no_runtime ? 7 : 0);
  CHECK_EQ(is_supported(0, nullptr), 100);  // other features forwarded
  CHECK(strcmp(st->last_role, "is-supported") == 0);
  static uint8_t pool[64];
  void* device = &pool[0];
  void* cmd = &pool[8];
  CHECK_EQ(set_device(device), 0);
  CHECK(st->ptr == device);
  if (rt) CHECK(rt->device == (uint8_t*)device + 1);  // unwrapped through the original's native-interface role
  void *fo = nullptr, *fs = nullptr, *fp = nullptr, *again = nullptr;
  const WgFrameToken* last_tok = nullptr;
  const WgHeader* last_in[1] = {nullptr};
  // the release build answers the game's requests with the real feature-function names (A9.5, binding/)
  CHECK(kWgFnOptimal[0] && kWgFnState[0] && kWgFnOptions[0]);
  CHECK_EQ(get_function(1001, kWgFnOptimal, &fo), 0);
  CHECK_EQ(get_function(1001, kWgFnState, &fs), 0);
  CHECK_EQ(get_function(1001, kWgFnOptions, &fp), 0);
  CHECK_EQ(get_function(1001, kWgFnOptimal, &again), 0);
  CHECK(fo && fs && fp && fo == again && fo != fs && fs != fp);
  {
    // any other name, a prefix or another case: 25, nothing returned
    std::string prefix(kWgFnOptimal, strlen(kWgFnOptimal) - 1), upper(kWgFnState);
    for (char& ch : upper) ch = (char)toupper((unsigned char)ch);
    void* none = nullptr;
    CHECK_EQ(get_function(1001, prefix.c_str(), &none), 25);
    CHECK_EQ(get_function(1001, upper.c_str(), &none), 25);
    CHECK_EQ(get_function(1001, "", &none), 25);
    CHECK(none == nullptr);
  }
  CHECK_EQ(get_function(0, kWgFnOptimal, &again), 100);  // other features forwarded
  CHECK(strcmp(st->last_role, "get-function") == 0 && st->ptr != nullptr &&
        strcmp((const char*)st->ptr, kWgFnOptimal) == 0);
  auto rr_optimal = reinterpret_cast<WgResultCode (*)(const WgRrOptions*, WgRrOptimal*)>(fo);
  auto rr_state = reinterpret_cast<WgResultCode (*)(const WgViewport*, WgRrState*)>(fs);
  auto rr_options = reinterpret_cast<WgResultCode (*)(const WgViewport*, const WgRrOptions*)>(fp);
  WgRrOptions op;
  memset(&op, 0, sizeof op);
  op.h.version = 3;
  op.mode = 3;
  op.output_width = 2560;
  op.output_height = 1440;
  op.colour_is_hdr = 1;
  WgRrOptimal os;
  memset(&os, 0, sizeof os);
  CHECK_EQ(rr_optimal(&op, &os), 0);
  CHECK(os.optimal_width == 1707 && os.optimal_height == 960 && os.min_width == 853 && os.max_width == 2560);
  static WgViewport vp;  // used again after this block
  memset(&vp, 0, sizeof vp);
  memcpy(vp.h.type.b, kWgGuid_viewport, 16);
  vp.h.version = 1;
  vp.id = 3;
  CHECK_EQ(rr_options(&vp, &op), 0);
  WgResource res[8];
  WgTag tags[8];
  const uint32_t types[8] = {3, 4, 1, 0, 7, 8, 5, 6};
  for (int i = 0; i < 8; ++i) {
    memset(&res[i], 0, sizeof res[i]);
    res[i].native = &pool[16 + i];
    res[i].width = i == 1 ? 2560 : 1707;
    res[i].height = i == 1 ? 1440 : 960;
    res[i].native_format = 10;  // R16G16B16A16_FLOAT
    res[i].state = 4;
    memset(&tags[i], 0, sizeof tags[i]);
    memcpy(tags[i].h.type.b, kWgGuid_tag, 16);
    tags[i].h.version = 1;
    tags[i].resource = &res[i];
    tags[i].buffer_type = types[i];
    tags[i].lifetime = 1;
  }
  CHECK_EQ(tag(&vp, tags, 8, cmd), 0);
  CHECK(strcmp(st->last_role, "tag") == 0 && st->ptr == tags && st->count == 8 && st->cmd == cmd);
  WgFrameToken* tok = nullptr;
  uint32_t idx = 1;
  CHECK_EQ(new_token(&tok, &idx), 0);
  CHECK(tok != nullptr && strcmp(st->last_role, "new-frame-token") == 0);
  // A10.1 token layout: table pointer at +0, the frame-token GUID at +16, index through the table's first function
  CHECK(tok && memcmp(reinterpret_cast<const uint8_t*>(tok) + 16, kWgGuid_frame_token, 16) == 0);
  CHECK(tok && tok->vtable && tok->vtable[0](tok) == 1u);
#if WG_BINDING_API_MINOR >= 8  // 2.8.0 and 2.14.1 have the frame-scoped tag function
  auto tag_for_frame = role<WgResultCode (*)(const WgFrameToken*, const WgViewport*, const WgTag*, uint32_t, void*)>(
      proxy, R_TAG_FOR_FRAME);
  CHECK(tag_for_frame != nullptr);
  if (tag_for_frame) {
    CHECK_EQ(tag_for_frame(tok, &vp, tags, 8, cmd), 0);  // copied keyed by frame, forwarded
    CHECK(strcmp(st->last_role, "tag-for-frame") == 0 && st->ptr == tags && st->cmd == cmd);
  }
#else
  CHECK(kWgRoleNames[R_TAG_FOR_FRAME][0] == 0);  // 2.7.x has no frame-scoped tag function
#endif
  WgConstants c;
  memset(&c, 0, sizeof c);
  c.h.version = 2;
  c.camera_motion_included = 1;
  c.mv_scale[0] = 1.0f / 1707.0f;
  c.mv_scale[1] = 1.0f / 960.0f;
  CHECK_EQ(constants(&c, tok, &vp), 0);
  CHECK(strcmp(st->last_role, "constants") == 0 && st->ptr == &c);
  const WgHeader* in[1] = {&vp.h};
  CHECK_EQ(evaluate(1001, tok, in, 1, cmd), no_runtime ? 7 : 0);
  CHECK(strcmp(st->last_role, "evaluate") != 0);  // RR never reaches the original's evaluate
  if (rt) {
    CHECK_EQ(rt->evaluates, 1);
    CHECK(rt->last().render_width == 1707 && rt->last().output_width == 2560 && rt->last().reset == 1);
    CHECK(rt->last().command_list == (uint8_t*)cmd + 1);
    WgRrState rs;
    CHECK_EQ(rr_state(&vp, &rs), 0);
    CHECK(rs.estimated_vram > 0);
  }
  CHECK_EQ(evaluate(0, tok, in, 1, cmd), 100);  // other features forwarded unchanged
  CHECK(strcmp(st->last_role, "evaluate") == 0 && st->feature == 0 && st->ptr == in && st->cmd == cmd);
  last_tok = tok;
  last_in[0] = &vp.h;

  // ---- settings contract and the ASI side
  auto get_contract = reinterpret_cast<WgRrGetSettingsContractFn>(GetProcAddress(proxy, WG_RR_CONTRACT_EXPORT));
  CHECK(get_contract != nullptr);
  const WgRrSettingsContract* ct = get_contract();
  CHECK(ct && ct->size == 40 && ct->version == 1 && ct->count >= 1);
  CHECK_EQ(ct->status(), no_runtime ? 0u : 7u);  // ready, loaded, active this frame
  {
    wgrr_asi::ProxyLink link;
    const WgRrSettingsContract* found = link.attach();
    CHECK(found == ct && link.module() == proxy);
    wgrr_asi::ControlsView v = wgrr_asi::build_view(found);
    if (no_runtime) {
      CHECK(v.mode == wgrr_asi::ControlsView::kNotActive && v.line == "WrongGPU RR not active");
    } else {
      CHECK(v.mode == wgrr_asi::ControlsView::kControls && v.items.size() == ct->count);
      CHECK_EQ(wgrr_asi::apply_change(found, 0, 0), 0u);  // "WrongGPU RR" off -> saved next to the proxy
      CHECK(GetFileAttributesW((wstage + L"wronggpu-rr.ini").c_str()) != INVALID_FILE_ATTRIBUTES);
      CHECK_EQ(ct->entries[0].get(), 0);
      CHECK_EQ(evaluate(1001, last_tok, last_in, 1, cmd), 0);  // the instance exists: keeps running
      CHECK_EQ(wgrr_asi::apply_change(found, 0, 1), 0u);
    }
    // the ASI holds a reference: the proxy stays loaded after the game's FreeLibrary until the ASI detaches
    CHECK_EQ(shutdown(), 0);
    FreeLibrary(proxy);
    CHECK(GetModuleHandleW(proxy_file.c_str()) == proxy);
    link.detach();
  }
  CHECK(GetModuleHandleW(proxy_file.c_str()) == nullptr);
  {
    wgrr_asi::ProxyLink link;
    CHECK(link.attach() == nullptr);  // proxy absent: WrongGPU.asi shows only the "not active" line
    CHECK(wgrr_asi::build_view(link.contract()).mode == wgrr_asi::ControlsView::kNotActive);
  }
  CHECK(GetFileAttributesW((wstage + L"wronggpu-rr.log").c_str()) != INVALID_FILE_ATTRIBUTES);
  // ---- load diagnostics in wronggpu-rr.log
  {
    std::vector<std::string> L = wgtest::log_lines(wstage + L"wronggpu-rr.log");
    wgtest::check_load_diagnostics(L, wstage + proxy_file, original_file, first_call.c_str(), "abs64 jmp (rax)",
                                   "test_proxy_dll.exe", ver, nullptr);
  }
  TEST_SUMMARY(no_runtime ? "test_proxy_dll (no runtime)" : "test_proxy_dll");
}
