// SPDX-License-Identifier: MIT
// Shared by the DLL tests (test_proxy_dll, test_probe_dll): the built proxy's VERSIONINFO read back through
// GetFileVersionInfoW / VerQueryValueW, a stand-in hook on the proxy's init export (what another module in the game
// does to it) for the load diagnostics, and log helpers. Include after test_util.h and the generated wg_binding.inc.
#pragma once
#define WIN32_LEAN_AND_MEAN
#include <windows.h>

#include <stdint.h>
#include <string.h>
#include <wchar.h>

#include <fstream>
#include <string>
#include <vector>

#pragma comment(lib, "version.lib")

namespace wgtest {

// The game interposer version of each build (A9.1: 2.7.1 for the 2.7 build, 2.8.0 for the 2.8 build, 2.14.1 for the 2.14 build): what other
// mods read from the interposer file's version resource.
#if WG_BINDING_API_MINOR == 7
static const wchar_t kExpectedFileVersion[] = L"2.7.1.0";
#elif WG_BINDING_API_MINOR == 8
static const wchar_t kExpectedFileVersion[] = L"2.8.0.0";
#elif WG_BINDING_API_MINOR == 14
static const wchar_t kExpectedFileVersion[] = L"2.14.1.0";
#else
#error "unknown interposer version"
#endif

inline std::wstring ver_string(const std::vector<uint8_t>& blk, const wchar_t* key) {
  struct LangCp { WORD lang, cp; }* tr = nullptr;
  UINT n = 0;
  if (!VerQueryValueW(blk.data(), L"\\VarFileInfo\\Translation", (void**)&tr, &n) || n < sizeof(LangCp) || !tr)
    return std::wstring();
  wchar_t q[160];
  swprintf(q, 160, L"\\StringFileInfo\\%04x%04x\\%ls", tr->lang, tr->cp, key);
  wchar_t* v = nullptr;
  if (!VerQueryValueW(blk.data(), q, (void**)&v, &n) || !v) return std::wstring();
  return std::wstring(v);
}

// kind: "release" or "probe".
inline void check_version_resource(const std::wstring& dll, const wchar_t* kind, const char* interposer_file) {
  DWORD ignored = 0;
  const DWORD size = GetFileVersionInfoSizeW(dll.c_str(), &ignored);
  CHECK(size > 0);
  if (!size) return;
  std::vector<uint8_t> blk(size);
  CHECK(GetFileVersionInfoW(dll.c_str(), 0, size, blk.data()));
  VS_FIXEDFILEINFO* f = nullptr;
  UINT n = 0;
  CHECK(VerQueryValueW(blk.data(), L"\\", (void**)&f, &n) && f && n >= sizeof(VS_FIXEDFILEINFO));
  if (!f) return;
  wchar_t fixed[64], product[64];
  swprintf(fixed, 64, L"%u.%u.%u.%u", HIWORD(f->dwFileVersionMS), LOWORD(f->dwFileVersionMS),
           HIWORD(f->dwFileVersionLS), LOWORD(f->dwFileVersionLS));
  swprintf(product, 64, L"%u.%u.%u.%u", HIWORD(f->dwProductVersionMS), LOWORD(f->dwProductVersionMS),
           HIWORD(f->dwProductVersionLS), LOWORD(f->dwProductVersionLS));
  CHECK(f->dwSignature == 0xFEEF04BDu);
  CHECK(wcscmp(fixed, kExpectedFileVersion) == 0);    // FILEVERSION
  CHECK(wcscmp(product, kExpectedFileVersion) == 0);  // PRODUCTVERSION
  CHECK(f->dwFileType == VFT_DLL);
  CHECK(ver_string(blk, L"FileVersion") == kExpectedFileVersion);
  CHECK(ver_string(blk, L"ProductVersion") == kExpectedFileVersion);
  const std::wstring product_name = ver_string(blk, L"ProductName");
  const std::wstring description = ver_string(blk, L"FileDescription");
  // the install tool's double-install guard looks for "WrongGPU" in both
  CHECK(product_name.find(L"WrongGPU") != std::wstring::npos);
  CHECK(description.find(L"WrongGPU") != std::wstring::npos);
  CHECK(product_name == L"WrongGPU RR proxy");
  CHECK(description == std::wstring(L"WrongGPU ray reconstruction proxy (") + kind + L")");
  CHECK(ver_string(blk, L"CompanyName") == L"WrongGPU");
  CHECK(ver_string(blk, L"InternalName") == L"wg_rr_proxy");
  std::wstring original;
  for (const char* c = interposer_file; *c; ++c) original.push_back((wchar_t)*c);
  CHECK(_wcsicmp(ver_string(blk, L"OriginalFilename").c_str(), original.c_str()) == 0);
}

// A stand-in for another module hooking the proxy's init export: overwrites its first bytes with a jump to `to`
// (abs64: jmp [rip+0] + address; rax: mov rax, imm64 + jmp rax). Returns the saved bytes for unhook().
enum class HookForm { kAbs64, kMovRax };
struct Hook {
  uint8_t* at = nullptr;
  uint8_t saved[16] = {};
};
inline Hook hook(void* at, const void* to, HookForm form) {
  Hook h;
  h.at = static_cast<uint8_t*>(at);
  memcpy(h.saved, h.at, 16);
  uint8_t code[16];
  memcpy(code, h.saved, 16);
  const uint64_t t = (uint64_t)(uintptr_t)to;
  if (form == HookForm::kAbs64) {
    const uint8_t jmp[6] = {0xFF, 0x25, 0, 0, 0, 0};
    memcpy(code, jmp, 6);
    memcpy(code + 6, &t, 8);
  } else {
    code[0] = 0x48;
    code[1] = 0xB8;
    memcpy(code + 2, &t, 8);
    code[10] = 0xFF;
    code[11] = 0xE0;
  }
  DWORD old = 0;
  VirtualProtect(h.at, 16, PAGE_EXECUTE_READWRITE, &old);
  memcpy(h.at, code, 16);
  VirtualProtect(h.at, 16, old, &old);
  FlushInstructionCache(GetCurrentProcess(), h.at, 16);
  return h;
}
inline void unhook(const Hook& h) {
  DWORD old = 0;
  VirtualProtect(h.at, 16, PAGE_EXECUTE_READWRITE, &old);
  memcpy(h.at, h.saved, 16);
  VirtualProtect(h.at, 16, old, &old);
  FlushInstructionCache(GetCurrentProcess(), h.at, 16);
}

inline std::vector<std::string> log_lines(const std::wstring& path) {
  std::vector<std::string> out;
  std::ifstream f(path.c_str());
  std::string l;
  while (std::getline(f, l)) {
    while (!l.empty() && (l.back() == '\r' || l.back() == '\n')) l.pop_back();
    out.push_back(l);
  }
  return out;
}
// Index of the first line containing needle (-1: none) and the number of such lines.
inline long first_index(const std::vector<std::string>& v, const std::string& needle) {
  for (size_t i = 0; i < v.size(); ++i)
    if (v[i].find(needle) != std::string::npos) return (long)i;
  return -1;
}
inline size_t count_lines(const std::vector<std::string>& v, const std::string& needle) {
  size_t n = 0;
  for (const auto& l : v) n += l.find(needle) != std::string::npos;
  return n;
}
inline std::string narrow_ascii(const std::wstring& w) {
  std::string s;
  for (wchar_t c : w) s.push_back((char)c);
  return s;
}

// The load diagnostics every build writes (proxy_dll.cpp): process / module path, the renamed original present, the
// first call, the init export check (hooked at that moment by the test, with the expected jump form and the test
// executable as the module holding the jump target), and every <init> entry before it is forwarded (unhooked by then).
inline void check_load_diagnostics(const std::vector<std::string>& L, const std::wstring& proxy_path,
                                   const std::wstring& original_file, const char* first_call, const char* jump_kind,
                                   const char* test_exe, uint64_t init_version, const char* init_result_line) {
  const std::string pid = std::to_string(GetCurrentProcessId());
  CHECK_EQ(count_lines(L, "load: process "), (size_t)1);
  const long attach = first_index(L, "load: process " + pid + " attached " + narrow_ascii(proxy_path));
  CHECK(attach >= 0);
  CHECK_EQ(count_lines(L, "load: original "), (size_t)1);
  const long orig = first_index(L, "load: original ");
  CHECK(orig >= 0 && L[orig].find(narrow_ascii(original_file) + " present: yes") != std::string::npos);
  CHECK_EQ(count_lines(L, std::string("load: first call ") + first_call), (size_t)1);
  CHECK_EQ(count_lines(L, "load: first call "), (size_t)1);
  CHECK_EQ(count_lines(L, "load: init export patched by another module: "), (size_t)1);
  const long patch = first_index(L, "load: init export patched by another module: yes");
  CHECK(patch >= 0);
  if (patch >= 0) {
    CHECK(L[patch].find(std::string("; ") + jump_kind + " -> 0x") != std::string::npos);
    CHECK(L[patch].find(test_exe) != std::string::npos);  // the module holding the jump target
    CHECK(L[patch].find(std::string(" (") + kWgRoleNames[0] + " at 0x") != std::string::npos);
  }
  CHECK(attach >= 0 && orig > attach && patch > orig);  // logged together at the first call
  CHECK_EQ(count_lines(L, "init entered: "), (size_t)1);
  const long entered = first_index(L, "init entered: ");
  char ver[40];
  snprintf(ver, sizeof ver, "sdk_version=0x%016llx", (unsigned long long)init_version);
  CHECK(entered > patch && L[entered].find(ver) != std::string::npos &&
        L[entered].find("; init export patched by another module: no") != std::string::npos);
  if (init_result_line) {
    const long result = first_index(L, init_result_line);
    CHECK(result > entered);  // logged before the init was forwarded
  }
}

}  // namespace wgtest
