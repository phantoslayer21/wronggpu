// SPDX-License-Identifier: MIT
// RR guard: the scene DLL of the Model M check (tools/guard_modelm.py, tests/guard_inject.cpp). Loaded into a
// suspended Model M test process before its main runs, it loads WrongGPU RR (the hook plugin) next to our stand-in game
// interface and, per WGTEST_GUARD_MODE, trips the guard:
//   none         no other plugin (control: WrongGPU RR as it is without the guard tripping)
//   off          our stand-in other plugin patched the interface's evaluate export first: WrongGPU RR stays off
//   take         the same with rr_other_plugin=take (the injector writes wronggpu-rr.ini next to the test exe)
//   owncopy      two copies of WrongGPU RR: the second stays off
//   passthrough  the stand-in patches the export after InitializeASI; OptiPluginInit finds it: WrongGPU RR passes on
// It checks that WrongGPU RR logged its line (wronggpu-rr.log next to the test exe, the part written by this process),
// writes "ok"/"fail <why>" to WGTEST_GUARD_RESULT and signals WGTEST_GUARD_EVENT; the injector then resumes the test.
// Files come from WGTEST_GUARD_STAGE (the stand-in interface, wgtest_other_rr.dll, the plugin and its copy).
#define WIN32_LEAN_AND_MEAN
#include <windows.h>

#include <stdio.h>
#include <string.h>

#include <string>

#include "wg_binding.inc"
#include "../include/wg_host_menu_abi.h"

namespace {

std::wstring env(const wchar_t* name) {
  wchar_t buf[2048];
  const DWORD n = GetEnvironmentVariableW(name, buf, 2048);
  return n && n < 2048 ? std::wstring(buf, n) : std::wstring();
}

std::string read_from(const std::wstring& path, long long from) {
  std::string s;
  FILE* f = _wfopen(path.c_str(), L"rb");
  if (!f) return s;
  _fseeki64(f, from, SEEK_SET);
  char b[4096];
  size_t n;
  while ((n = fread(b, 1, sizeof b, f)) > 0) s.append(b, n);
  fclose(f);
  return s;
}

long long size_of(const std::wstring& path) {
  WIN32_FILE_ATTRIBUTE_DATA a;
  return GetFileAttributesExW(path.c_str(), GetFileExInfoStandard, &a) ? ((long long)a.nFileSizeHigh << 32) | a.nFileSizeLow
                                                                       : 0;
}

size_t occurrences(const std::string& hay, const std::string& needle) {
  size_t n = 0;
  for (size_t at = hay.find(needle); at != std::string::npos; at = hay.find(needle, at + 1)) ++n;
  return n;
}

std::string run_scene() {
  const std::wstring stage = env(L"WGTEST_GUARD_STAGE");
  const std::wstring mode_w = env(L"WGTEST_GUARD_MODE");
  const std::string mode(mode_w.begin(), mode_w.end());
  wchar_t exe[MAX_PATH];
  const DWORD en = GetModuleFileNameW(nullptr, exe, MAX_PATH);
  std::wstring dir(exe, en);
  dir = dir.substr(0, dir.find_last_of(L"\\/") + 1);
  const std::wstring log = dir + L"wronggpu-rr.log";
  const long long from = size_of(log);

  HMODULE iface = LoadLibraryW((stage + L"\\" + std::wstring(kWgInterposerFile, kWgInterposerFile + strlen(kWgInterposerFile))).c_str());
  HMODULE other = LoadLibraryW((stage + L"\\wgtest_other_rr.dll").c_str());
  if (!iface || !other) return "fail: stand-ins not loaded";
  auto patch_export = reinterpret_cast<int (*)(HMODULE, const char*, int)>(GetProcAddress(other, "wgtest_or_patch_export"));
  const char* evaluate = kWgRoleNames[16];
  if ((mode == "off" || mode == "take") && patch_export(iface, evaluate, 0) != 1) return "fail: export not patched";
  const std::wstring hook(kWgHookFile, kWgHookFile + strlen(kWgHookFile));
  HMODULE asi = LoadLibraryW((stage + L"\\" + hook).c_str());
  if (!asi) return "fail: plugin not loaded";
  auto init = reinterpret_cast<void (*)()>(GetProcAddress(asi, "InitializeASI"));
  auto plugin_init = reinterpret_cast<WgHostMenuInitFn>(GetProcAddress(asi, WG_HOST_MENU_INIT_EXPORT));
  if (!init || !plugin_init) return "fail: plugin entries missing";
  init();
  if (mode == "owncopy") {
    HMODULE copy = LoadLibraryW((stage + L"\\WrongGPU-RR-copy.asi").c_str());
    auto init2 = copy ? reinterpret_cast<void (*)()>(GetProcAddress(copy, "InitializeASI")) : nullptr;
    if (!init2) return "fail: copy not loaded";
    init2();
  }
  if (mode == "passthrough" && patch_export(iface, evaluate, 0) != 1) return "fail: export not patched";
  static WgHostMenuApi api = {};  // a host table without menu, routes or NGX provider
  api.size = sizeof api;
  api.version = 5;
  plugin_init(&api, reinterpret_cast<WgHostMenuPlugin*>(0x1000));

  const std::string got = read_from(log, from);
  const std::string stem = "another Ray Reconstruction plugin is active; WrongGPU RR stays off (";
  struct Want { const char* text; size_t n; };
  Want want[3] = {};
  if (mode == "none") want[0] = {"hook: coexistence check at load: no other Ray Reconstruction plugin found (", 1};
  if (mode == "off") want[0] = {"; nothing patched; rr_other_plugin=take in wronggpu-rr.ini uses WrongGPU's instead)", 1};
  if (mode == "take") want[0] = {"another Ray Reconstruction plugin is active; rr_other_plugin=take: WrongGPU RR goes in front of it (", 1};
  if (mode == "owncopy") want[0] = {"a second copy of WrongGPU RR is loaded (", 1};
  if (mode == "passthrough") want[0] = {"; it loaded after WrongGPU: WrongGPU passes every call on unchanged; to use WrongGPU RR, move it aside)", 1};
  if (!want[0].text) return "fail: unknown mode " + mode;
  if (occurrences(got, want[0].text) != want[0].n) return std::string("fail: log lacks \"") + want[0].text + "\"";
  const size_t stems = occurrences(got, stem);
  if (stems != ((mode == "off" || mode == "passthrough") ? 1u : 0u)) return "fail: " + std::to_string(stems) + " stays-off line(s)";
  return "ok " + mode;
}

DWORD WINAPI scene_thread(void*) {
  std::string r;
  try {
    r = run_scene();
  } catch (...) {
    r = "fail: exception";
  }
  const std::wstring result = env(L"WGTEST_GUARD_RESULT");
  FILE* f = _wfopen(result.c_str(), L"wb");
  if (f) {
    fputs(r.c_str(), f);
    fclose(f);
  }
  HANDLE ev = OpenEventW(EVENT_MODIFY_STATE, FALSE, env(L"WGTEST_GUARD_EVENT").c_str());
  if (ev) {
    SetEvent(ev);
    CloseHandle(ev);
  }
  return 0;
}

}  // namespace

BOOL WINAPI DllMain(HINSTANCE inst, DWORD reason, LPVOID) {
  if (reason == DLL_PROCESS_ATTACH) {
    DisableThreadLibraryCalls(inst);
    if (HANDLE t = CreateThread(nullptr, 0, scene_thread, nullptr, 0, nullptr)) CloseHandle(t);  // runs after the lock
  }
  return TRUE;
}
