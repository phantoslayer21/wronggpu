// SPDX-License-Identifier: MIT
// The proxy library (spec 60.1, annexes A9 and A10): installed under the game's interposer file name; loads the
// renamed original (<own file stem>.wg-original.dll) from its own directory with a full path, forwards every export
// through generated argument-agnostic stubs, and implements the calls A9.4 marks for the proxy through RrProxy
// (declarations of 60.6, argument and return types of A10).
//
// Probe build (WG_RR_PROBE=1, tools/build.py, output wg_rr_proxy_probe.dll; never the release build): every API
// role goes to RrProbe (probe.h) instead, which logs to wronggpu-rr-probe.log next to the proxy; graphics and
// Vulkan entry points are logged by their stubs (wg_fwd_note) before the jump.
//
// Load diagnostics (both builds): DLL_PROCESS_ATTACH only records the process id, the own module path and whether
// the renamed original is present (no library loads, no threads, no file writes under the loader lock); the first
// call of any kind writes those lines and checks whether the first 16 bytes of the proxy's own init export in memory
// still equal its on-disk image (another module hooking it). Every init entry is logged before it is forwarded.
// Release: wronggpu-rr.log next to the proxy; probe: the probe log.
//
// Build: tools/build.py (the generated wg_binding.inc, wg_proxy.def and wg_forward_stubs.asm come from
// tools/gen_forwarding.py, a binding table, the A9.3 export list and the names files of binding/).
#define WIN32_LEAN_AND_MEAN
#include <windows.h>

#include <stdint.h>
#include <stdio.h>

#include <mutex>
#include <string>
#include <vector>

#include "log_append.h"
#include "proxy_core.h"
#include "rr_enabled.h"
#include "settings.h"
#include "wg_binding.inc"  // generated

#ifndef WG_RR_PROBE
#define WG_RR_PROBE 0
#endif
static_assert(WG_RR_PROBE == WG_BINDING_PROBE, "the probe build needs the probe binding (gen_forwarding --probe)");

#if WG_RR_PROBE
#include "probe.h"
#include "probe_gpu_d3d12.h"
#include "probe_log.h"
#endif

extern "C" {
void* wg_fwd_table[WG_BINDING_FORWARD_COUNT];  // filled on first use of each stub (wg_fwd_resolve)
}

namespace {

HMODULE g_self = nullptr;
HMODULE g_original = nullptr;
HMODULE g_runtime_module = nullptr;
const WgRrRuntimeV1* g_runtime = nullptr;
wgrr::Settings g_settings;
std::once_flag g_settings_once, g_target_once, g_original_once;
std::mutex g_log_mutex;
#if WG_RR_PROBE
wgrr::RrProbe* g_probe = nullptr;
wgrr::ProbeLog* g_probe_log = nullptr;
std::once_flag g_probe_log_once;
std::wstring g_probe_path;  // wronggpu-rr-probe.log (0.6.1: one append call per line, log_append.h)
#else
wgrr::RrProxy* g_proxy = nullptr;
#endif

std::wstring own_path() {
  wchar_t buf[MAX_PATH * 2];
  DWORD n = GetModuleFileNameW(g_self, buf, (DWORD)(sizeof buf / sizeof buf[0]));
  return std::wstring(buf, n);
}

std::wstring own_dir() {
  std::wstring p = own_path();
  size_t slash = p.find_last_of(L"\\/");
  return slash == std::wstring::npos ? std::wstring() : p.substr(0, slash + 1);
}

// The proxy's own file name without directory and last extension. The install tool gives the proxy the game's
// interposer file name, so the renamed original is found next to it whatever that name is.
std::wstring own_stem() {
  std::wstring p = own_path();
  size_t slash = p.find_last_of(L"\\/");
  std::wstring f = slash == std::wstring::npos ? p : p.substr(slash + 1);
  size_t dot = f.find_last_of(L'.');
  return dot == std::wstring::npos ? f : f.substr(0, dot);
}

std::wstring original_file_name() {
#if WG_BINDING_ORIGINAL_FROM_STEM
  return own_stem() + kWgOriginalFile;
#else
  return kWgOriginalFile;
#endif
}

std::string narrow(const std::wstring& w) {
  int n = WideCharToMultiByte(CP_UTF8, 0, w.c_str(), (int)w.size(), nullptr, 0, nullptr, nullptr);
  std::string s(n, '\0');
  WideCharToMultiByte(CP_UTF8, 0, w.c_str(), (int)w.size(), &s[0], n, nullptr, nullptr);
  return s;
}

#if WG_RR_PROBE
// wronggpu-rr-probe.log next to the proxy; a previous run's log is kept as wronggpu-rr-probe.prev.log.
wgrr::ProbeLog* probe_log() {
  std::call_once(g_probe_log_once, [] {
    std::wstring path = own_dir() + L"wronggpu-rr-probe.log";
    std::wstring prev = own_dir() + L"wronggpu-rr-probe.prev.log";
    {
      const wgrr::LogLock shared(path, wgrr::kLogLockWaitMs);
      MoveFileExW(path.c_str(), prev.c_str(), MOVEFILE_REPLACE_EXISTING);
      // a new, empty log (as the earlier "wb" open did, also when the old one could not be moved aside)
      HANDLE fresh = CreateFileW(path.c_str(), GENERIC_WRITE, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, nullptr,
                                 CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
      if (fresh != INVALID_HANDLE_VALUE) CloseHandle(fresh);
    }
    g_probe_path = path;  // 0.6.1: one append call per line under the log's process-wide lock (log_append.h)
    g_probe_log = new wgrr::ProbeLog([](const std::string& line) {
      std::lock_guard<std::mutex> lock(g_log_mutex);
      const wgrr::LogLock shared(g_probe_path, wgrr::kLogLockWaitMs);
      wgrr::append_log_text(g_probe_path, wgrr::log_stamp() + line + "\r\n");  // on disk at once: a crash keeps it
    });
    g_probe_log->note(std::string("probe start: build Streamline 2.") + std::to_string(WG_BINDING_API_MINOR) +
                      ", loaded as " + narrow(own_stem()) + ", process " + std::to_string(GetCurrentProcessId()));
  });
  return g_probe_log;
}
#endif

void log_line(const std::string& s) {
#if WG_RR_PROBE
  probe_log()->note(s);
#else
  std::lock_guard<std::mutex> lock(g_log_mutex);
  std::wstring path = own_dir() + L"wronggpu-rr.log";
  const wgrr::LogLock shared(path, wgrr::kLogLockWaitMs);  // 0.6.1: one append call under the process-wide lock
  wgrr::append_log_text(path, wgrr::log_stamp() + s + "\r\n");
#endif
}

// ---------------------------------------------------------------- load diagnostics
char g_attach_line[2][1200];  // written by DllMain (no loads, no threads), logged at the first call
std::once_flag g_diag_once, g_init_ref_once;
bool g_init_ref_ok = false;
uint32_t g_init_rva = 0;
uint8_t g_init_disk[16];  // the init export's first 16 bytes of the on-disk image, relocated to the load address
std::string g_init_ref_error;

// Printable ASCII only (the logs hold names and numbers); other bytes as \xNN.
std::string ascii(const std::string& s) {
  std::string o;
  char buf[8];
  for (unsigned char c : s) {
    if (c >= 0x20 && c <= 0x7e) {
      o.push_back((char)c);
    } else {
      snprintf(buf, sizeof buf, "\\x%02x", c);
      o += buf;
    }
  }
  return o;
}

std::string hex_bytes(const uint8_t* p, size_t n) {
  static const char d[] = "0123456789abcdef";
  std::string o;
  for (size_t i = 0; i < n; ++i) {
    o.push_back(d[p[i] >> 4]);
    o.push_back(d[p[i] & 15]);
  }
  return o;
}

std::string hex_addr(uintptr_t a) {
  char buf[32];
  snprintf(buf, sizeof buf, "0x%llx", (unsigned long long)a);
  return buf;
}

// Called from DllMain: kernel32 calls and fixed buffers only.
void record_attach(HINSTANCE inst) {
  const DWORD cap = MAX_PATH * 2;
  wchar_t path[MAX_PATH * 2];
  DWORD n = GetModuleFileNameW(inst, path, cap);
  if (n >= cap) n = cap - 1;
  path[n] = 0;
  wchar_t orig[MAX_PATH * 2 + 64];
  size_t cut = 0;
  for (DWORD i = 0; i < n; ++i)
    if (path[i] == L'\\' || path[i] == L'/') cut = i + 1;
#if WG_BINDING_ORIGINAL_FROM_STEM
  size_t dot = n;  // the own file stem: the file name up to its last dot
  for (size_t i = cut; i < n; ++i)
    if (path[i] == L'.') dot = i;
  cut = dot;
#endif
  memcpy(orig, path, cut * sizeof(wchar_t));
  size_t k = cut;
  for (const wchar_t* q = kWgOriginalFile; *q && k + 1 < sizeof orig / sizeof orig[0]; ++q) orig[k++] = *q;
  orig[k] = 0;
  DWORD attr = GetFileAttributesW(orig);
  bool present = attr != INVALID_FILE_ATTRIBUTES && !(attr & FILE_ATTRIBUTE_DIRECTORY);
  static char p8[MAX_PATH * 8], o8[MAX_PATH * 8 + 256];
  if (WideCharToMultiByte(CP_UTF8, 0, path, -1, p8, (int)sizeof p8, nullptr, nullptr) <= 0) p8[0] = 0;
  if (WideCharToMultiByte(CP_UTF8, 0, orig, -1, o8, (int)sizeof o8, nullptr, nullptr) <= 0) o8[0] = 0;
  snprintf(g_attach_line[0], sizeof g_attach_line[0], "load: process %lu attached %s",
           (unsigned long)GetCurrentProcessId(), p8);
  snprintf(g_attach_line[1], sizeof g_attach_line[1], "load: original %s present: %s", o8, present ? "yes" : "no");
}

// A PE32+ image, either mapped (rva = offset) or the file as on disk (rva through the section table).
struct PeImage {
  const uint8_t* p = nullptr;
  size_t size = 0;
  bool mapped = false;
  const IMAGE_NT_HEADERS64* nt = nullptr;

  bool open(const uint8_t* data, size_t n, bool is_mapped) {
    p = data;
    size = n;
    mapped = is_mapped;
    if (n < sizeof(IMAGE_DOS_HEADER)) return false;
    const IMAGE_DOS_HEADER* dos = reinterpret_cast<const IMAGE_DOS_HEADER*>(p);
    if (dos->e_magic != IMAGE_DOS_SIGNATURE || dos->e_lfanew <= 0 ||
        (size_t)dos->e_lfanew + sizeof(IMAGE_NT_HEADERS64) > n)
      return false;
    nt = reinterpret_cast<const IMAGE_NT_HEADERS64*>(p + dos->e_lfanew);
    if (mapped) size = nt->OptionalHeader.SizeOfImage;
    return nt->Signature == IMAGE_NT_SIGNATURE && nt->OptionalHeader.Magic == IMAGE_NT_OPTIONAL_HDR64_MAGIC;
  }
  // Pointer to [rva, rva + len) or nullptr.
  const uint8_t* at(uint32_t rva, size_t len) const {
    if (mapped) return (size_t)rva + len <= size ? p + rva : nullptr;
    const IMAGE_SECTION_HEADER* s = IMAGE_FIRST_SECTION(nt);
    for (unsigned i = 0; i < nt->FileHeader.NumberOfSections; ++i) {
      uint32_t va = s[i].VirtualAddress, raw = s[i].SizeOfRawData;
      if (rva >= va && (uint64_t)rva + len <= (uint64_t)va + raw) {
        size_t off = (size_t)s[i].PointerToRawData + (rva - va);
        return off + len <= size ? p + off : nullptr;
      }
    }
    return nullptr;
  }
  // The RVA of a named export as the export table says (0 = not found).
  uint32_t export_rva(const char* name) const {
    const IMAGE_DATA_DIRECTORY& d = nt->OptionalHeader.DataDirectory[IMAGE_DIRECTORY_ENTRY_EXPORT];
    if (!d.VirtualAddress) return 0;
    const IMAGE_EXPORT_DIRECTORY* ed =
        reinterpret_cast<const IMAGE_EXPORT_DIRECTORY*>(at(d.VirtualAddress, sizeof(IMAGE_EXPORT_DIRECTORY)));
    if (!ed) return 0;
    const DWORD* names = reinterpret_cast<const DWORD*>(at(ed->AddressOfNames, 4ull * ed->NumberOfNames));
    const WORD* ords = reinterpret_cast<const WORD*>(at(ed->AddressOfNameOrdinals, 2ull * ed->NumberOfNames));
    const DWORD* fns = reinterpret_cast<const DWORD*>(at(ed->AddressOfFunctions, 4ull * ed->NumberOfFunctions));
    if (!names || !ords || !fns) return 0;
    size_t len = strlen(name) + 1;
    for (DWORD i = 0; i < ed->NumberOfNames; ++i) {
      const char* nm = reinterpret_cast<const char*>(at(names[i], len));
      if (nm && memcmp(nm, name, len) == 0) return ords[i] < ed->NumberOfFunctions ? fns[ords[i]] : 0;
    }
    return 0;
  }
};

bool read_file(const std::wstring& path, std::vector<uint8_t>* out) {
  HANDLE h = CreateFileW(path.c_str(), GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, nullptr,
                         OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
  if (h == INVALID_HANDLE_VALUE) return false;
  LARGE_INTEGER sz;
  bool ok = GetFileSizeEx(h, &sz) && sz.QuadPart > 0 && sz.QuadPart < (256ll << 20);
  if (ok) {
    out->resize((size_t)sz.QuadPart);
    DWORD got = 0;
    ok = ReadFile(h, out->data(), (DWORD)out->size(), &got, nullptr) && got == out->size();
  }
  CloseHandle(h);
  return ok;
}

bool readable(uintptr_t a, size_t n) {
  MEMORY_BASIC_INFORMATION mb;
  if (!a || VirtualQuery(reinterpret_cast<const void*>(a), &mb, sizeof mb) != sizeof mb) return false;
  if (mb.State != MEM_COMMIT || mb.Protect == 0 || (mb.Protect & (PAGE_NOACCESS | PAGE_GUARD))) return false;
  return a + n <= (uintptr_t)mb.BaseAddress + mb.RegionSize;
}

// The init export's first 16 bytes as the on-disk image has them, with the base relocations applied for the
// address the module is loaded at. Done once (reads the own file; never under the loader lock).
void init_reference() {
  std::call_once(g_init_ref_once, [] {
    std::vector<uint8_t> file;
    PeImage disk;
    if (!read_file(own_path(), &file) || !disk.open(file.data(), file.size(), false)) {
      g_init_ref_error = "cannot read the own file";
      return;
    }
    uint32_t rva = disk.export_rva(kWgRoleNames[0]);
    const uint8_t* code = rva ? disk.at(rva, 16) : nullptr;
    if (!code) {
      g_init_ref_error = "init export not found in the own file";
      return;
    }
    memcpy(g_init_disk, code, 16);
    const uint64_t delta = (uint64_t)(uintptr_t)g_self - disk.nt->OptionalHeader.ImageBase;
    const IMAGE_DATA_DIRECTORY& rd = disk.nt->OptionalHeader.DataDirectory[IMAGE_DIRECTORY_ENTRY_BASERELOC];
    for (uint32_t off = 0; delta && rd.VirtualAddress && off + sizeof(IMAGE_BASE_RELOCATION) <= rd.Size;) {
      const IMAGE_BASE_RELOCATION* blk = reinterpret_cast<const IMAGE_BASE_RELOCATION*>(
          disk.at(rd.VirtualAddress + off, sizeof(IMAGE_BASE_RELOCATION)));
      if (!blk || blk->SizeOfBlock < sizeof(IMAGE_BASE_RELOCATION)) break;
      const uint32_t count = (blk->SizeOfBlock - (uint32_t)sizeof(IMAGE_BASE_RELOCATION)) / 2;
      const WORD* e = reinterpret_cast<const WORD*>(
          disk.at(rd.VirtualAddress + off + (uint32_t)sizeof(IMAGE_BASE_RELOCATION), 2ull * count));
      for (uint32_t i = 0; e && i < count; ++i) {
        const unsigned type = e[i] >> 12;
        const unsigned width = type == IMAGE_REL_BASED_DIR64 ? 8 : type == IMAGE_REL_BASED_HIGHLOW ? 4 : 0;
        const uint32_t r = blk->VirtualAddress + (e[i] & 0xfffu);
        if (!width || r >= rva + 16 || r + width <= rva) continue;
        const uint8_t* at = disk.at(r, width);
        if (!at) continue;
        uint64_t v = 0;
        memcpy(&v, at, width);
        v += delta;
        for (unsigned b = 0; b < width; ++b)
          if (r + b >= rva && r + b < rva + 16) g_init_disk[r + b - rva] = (uint8_t)(v >> (8 * b));
      }
      off += blk->SizeOfBlock;
    }
    g_init_rva = rva;
    g_init_ref_ok = true;
  });
}

std::string module_of(uintptr_t a) {
  HMODULE m = nullptr;
  if (!GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                          reinterpret_cast<LPCWSTR>(a), &m) || !m)
    return std::string();
  wchar_t buf[MAX_PATH * 2];
  DWORD n = GetModuleFileNameW(m, buf, MAX_PATH * 2);
  return ascii(narrow(std::wstring(buf, n)));
}

// A simple jump at a: rel8 / rel32 jmp, jmp [rip+disp32] (abs64 when disp32 = 0), mov rax|r11, imm64 + jmp reg.
// Returns the target, 0 if a holds no such jump.
uintptr_t jump_target(uintptr_t a, const char** kind) {
  if (!readable(a, 16)) return 0;
  const uint8_t* c = reinterpret_cast<const uint8_t*>(a);
  uintptr_t t = 0;
  int32_t d = 0;
  if (c[0] == 0xE9) {
    memcpy(&d, c + 1, 4);
    *kind = "rel32 jmp";
    return a + 5 + (intptr_t)d;
  }
  if (c[0] == 0xEB) {
    *kind = "rel8 jmp";
    return a + 2 + (intptr_t)(int8_t)c[1];
  }
  if (c[0] == 0xFF && c[1] == 0x25) {
    memcpy(&d, c + 2, 4);
    const uintptr_t slot = a + 6 + (intptr_t)d;
    if (!readable(slot, 8)) return 0;
    memcpy(&t, reinterpret_cast<const void*>(slot), 8);
    *kind = d == 0 ? "abs64 jmp" : "indirect jmp";
    return t;
  }
  if (c[0] == 0x48 && c[1] == 0xB8 && c[10] == 0xFF && c[11] == 0xE0) {
    memcpy(&t, c + 2, 8);
    *kind = "abs64 jmp (rax)";
    return t;
  }
  if (c[0] == 0x49 && c[1] == 0xBB && c[10] == 0x41 && c[11] == 0xFF && c[12] == 0xE3) {
    memcpy(&t, c + 2, 8);
    *kind = "abs64 jmp (r11)";
    return t;
  }
  return 0;
}

// Where a jump at a leads: follows up to 4 jumps through memory that belongs to no module (relay thunks).
std::string jump_report(uintptr_t a) {
  std::string out;
  for (int hop = 0; hop < 4; ++hop) {
    const char* kind = "";
    const uintptr_t t = jump_target(a, &kind);
    if (!t) return out.empty() ? std::string("; not a simple jmp") : out;
    const std::string mod = module_of(t);
    out += std::string("; ") + kind + " -> " + hex_addr(t) + (mod.empty() ? " (in no module)" : " in " + mod);
    if (!mod.empty()) return out;
    a = t;
  }
  return out;
}

// "yes": the init export's first 16 bytes in memory differ from the on-disk image or its export table entry was
// redirected; "no"; "unknown" when the own file could not be read.
const char* init_patched_state() {
  init_reference();
  if (!g_init_ref_ok) return "unknown";
  PeImage mem;
  const uint8_t* base = reinterpret_cast<const uint8_t*>(g_self);
  if (!mem.open(base, sizeof(IMAGE_DOS_HEADER) + 4096, true)) return "unknown";
  const uint32_t live = mem.export_rva(kWgRoleNames[0]);
  return live == g_init_rva && memcmp(base + g_init_rva, g_init_disk, 16) == 0 ? "no" : "yes";
}

std::string init_patch_report() {
  std::string line = std::string("load: init export patched by another module: ") + init_patched_state();
  if (!g_init_ref_ok) return line + " (" + g_init_ref_error + ")";
  const uint8_t* base = reinterpret_cast<const uint8_t*>(g_self);
  PeImage mem;
  mem.open(base, sizeof(IMAGE_DOS_HEADER) + 4096, true);
  const uint32_t live = mem.export_rva(kWgRoleNames[0]);
  line += std::string(" (") + kWgRoleNames[0] + " at " + hex_addr((uintptr_t)base + g_init_rva) + ", memory " +
          hex_bytes(base + g_init_rva, 16) + ", disk " + hex_bytes(g_init_disk, 16) + ")";
  if (live != g_init_rva) {
    const std::string mod = live ? module_of((uintptr_t)base + live) : std::string();
    line += "; export entry redirected to " + hex_addr((uintptr_t)base + live) +
            (mod.empty() ? " (in no module)" : " in " + mod);
  }
  if (memcmp(base + g_init_rva, g_init_disk, 16) != 0) line += jump_report((uintptr_t)base + g_init_rva);
  return line;
}

// The first call of any kind (forwarded or answered) writes the load lines and the init export check.
void diagnostics(const char* first_call) {
  std::call_once(g_diag_once, [first_call] {
    log_line(ascii(g_attach_line[0]));
    log_line(ascii(g_attach_line[1]));
    log_line(std::string("load: first call ") + first_call);
    log_line(init_patch_report());
  });
}

// Every <init> entry, before it is forwarded (a swallowed init stays visible).
void log_init_entered(const WgPreferences* p, uint64_t v) {
  char buf[200];
  snprintf(buf, sizeof buf, "init entered: thread %lu preferences=%s sdk_version=0x%016llx",
           (unsigned long)GetCurrentThreadId(), hex_addr((uintptr_t)p).c_str(), (unsigned long long)v);
  log_line(std::string(buf) + "; init export patched by another module: " + init_patched_state());
}

void load_original() {
  std::call_once(g_original_once, [] {
    std::wstring path = own_dir() + original_file_name();
    g_original = LoadLibraryW(path.c_str());  // full path, never the search order (60.1)
    if (!g_original) log_line("proxy: the renamed original could not be loaded: " + narrow(path));
  });
}

WgResultCode missing_export() { return WG_E_MISSING_OR_INVALID_API; }

void* resolve_original(const char* name) {
  load_original();
  void* p = g_original ? (void*)GetProcAddress(g_original, name) : nullptr;
  return p;
}

uint32_t status_word();

void ensure_settings() {
  std::call_once(g_settings_once, [] {
    g_settings.load(narrow(own_dir() + L"wronggpu-rr.ini"), log_line);  // read once (60.9)
    wgrr::bind_contract(&g_settings, [] { return status_word(); });
  });
}

template <typename F>
void bind(F& slot, int role) {
  const char* name = kWgRoleNames[role];
  slot = name[0] ? reinterpret_cast<F>(resolve_original(name)) : nullptr;
}

extern "C" WgResultCode wg_rr_optimal_fn(const WgRrOptions* o, WgRrOptimal* s);
extern "C" WgResultCode wg_rr_state_fn(const WgViewport* v, WgRrState* s);
extern "C" WgResultCode wg_rr_options_fn(const WgViewport* v, const WgRrOptions* o);

wgrr::OriginalApi bind_original() {
  wgrr::OriginalApi o;
  // role indices: gen_forwarding.ROLES
  bind(o.init, 0); bind(o.shutdown, 1); bind(o.is_supported, 2); bind(o.is_loaded, 3); bind(o.set_loaded, 4);
  bind(o.requirements, 5); bind(o.version, 6); bind(o.get_function, 7); bind(o.allocate, 8);
  bind(o.free_resources, 9); bind(o.set_device, 10); bind(o.set_vulkan_info, 11); bind(o.tag, 12);
  bind(o.tag_for_frame, 13); bind(o.constants, 14); bind(o.new_frame_token, 15); bind(o.evaluate, 16);
  bind(o.upgrade_interface, 17); bind(o.native_interface, 18);
  return o;
}

void own_name_check() {
  std::wstring self = own_stem() + L".dll";
  std::string expected = kWgInterposerFile;
  if (_stricmp(narrow(self).c_str(), expected.c_str()) != 0)
    log_line("proxy: loaded as " + narrow(self) + "; the install tool places it under the game's interposer name");
}

#if WG_RR_PROBE
wgrr::RrProbe* target() {
  std::call_once(g_target_once, [] {
    ensure_settings();
    probe_log();
    own_name_check();
    wgrr::ProbeConfig cfg;
    cfg.api_minor = WG_BINDING_API_MINOR;
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
    for (const WgGuidName& g : kWgGuidNames) {
      WgGuid id;
      memcpy(id.b, g.b, 16);
      cfg.guid_names.push_back(std::make_pair(std::string(g.name), id));
    }
    g_probe = new wgrr::RrProbe(cfg, bind_original(), wgrr::create_d3d12_probe_gpu(), probe_log(),
                                [] { return (uint32_t)GetCurrentThreadId(); });
    g_probe->set_clock([] { return (uint64_t)GetTickCount64(); });
    g_probe->set_feature_function_pointers((void*)&wg_rr_optimal_fn, (void*)&wg_rr_state_fn,
                                           (void*)&wg_rr_options_fn);
  });
  return g_probe;
}

uint32_t status_word() { return g_probe ? g_probe->status_word() : 0u; }
#else
wgrr::RrProxy* target() {
  std::call_once(g_target_once, [] {
    ensure_settings();
    own_name_check();
    wgrr::OriginalApi o = bind_original();
    std::wstring rt_path = own_dir() + WG_RR_RUNTIME_FILE;
    // The player's switch (rr_enabled.h): off = WrongGPU does not take RR (reported unsupported; runtime not loaded).
    const wgrr::RrEnabled enabled = wgrr::read_rr_enabled(wgss::data_dir() + L"wronggpu.ini",
                                                          own_dir() + L"wronggpu-rr.ini");
    if (!enabled.on)
      log_line("proxy: rr_enabled=0 (" + enabled.source + "): WrongGPU does not take RR; RR is reported unsupported, "
               "so the game uses its own path. The RR runtime is not loaded.");
    else
      g_runtime_module = LoadLibraryW(rt_path.c_str());
    if (g_runtime_module) {
      auto entry = reinterpret_cast<WgRrRuntimeEntryFn>(GetProcAddress(g_runtime_module, WG_RR_RUNTIME_EXPORT));
      const WgRrRuntimeV1* rt = entry ? entry() : nullptr;
      if (rt && rt->version == WG_RR_ADAPTER_VERSION && rt->size >= WG_RR_RUNTIME_V1_BASE_SIZE) g_runtime = rt;
      else log_line("proxy: WrongGPU RR runtime has no usable adapter; RR reports unsupported");
    } else if (enabled.on) {
      log_line("proxy: WrongGPU RR runtime not found; RR reports unsupported: " + narrow(rt_path));
    }
    wgrr::ProxyConfig cfg;
    cfg.api_minor = WG_BINDING_API_MINOR;
    memcpy(cfg.ids.viewport.b, kWgGuid_viewport, 16);
    memcpy(cfg.ids.tag.b, kWgGuid_tag, 16);
    memcpy(cfg.ids.tag_conversion.b, kWgGuid_tag_conversion, 16);
    cfg.ids.has_tag_conversion = WG_BINDING_HAS_TAG_CONVERSION != 0;
    cfg.version_magic = WG_BINDING_VERSION_MAGIC;
    for (int i = 0; i < 3; ++i) cfg.release_version[i] = kWgReleaseVersion[i];
    cfg.fn_optimal = kWgFnOptimal;  // A9.5 names (binding/rr-functions.json)
    cfg.fn_state = kWgFnState;
    cfg.fn_options = kWgFnOptions;
    g_proxy = new wgrr::RrProxy(cfg, o, g_runtime, &g_settings, log_line,
                                [] { return (uint64_t)GetTickCount64(); });
    g_proxy->set_feature_function_pointers((void*)&wg_rr_optimal_fn, (void*)&wg_rr_state_fn,
                                           (void*)&wg_rr_options_fn);
    if (!enabled.on) g_proxy->set_rr_off(true, wgrr::rr_off_reason(enabled));
  });
  return g_proxy;
}

uint32_t status_word() { return g_proxy ? g_proxy->status_word() : 0u; }
#endif

// Every export the proxy answers enters here: the load diagnostics on the very first call, then the target.
auto entered(const char* call) -> decltype(target()) {
  diagnostics(call);
  return target();
}
auto entered(int role) -> decltype(target()) { return entered(kWgRoleNames[role]); }

}  // namespace

extern "C" {

// Resolves one forwarding slot by name on first use (called from wg_fwd_slow with the argument registers saved).
void* wg_fwd_resolve(uint32_t slot) {
  diagnostics(slot < WG_BINDING_FORWARD_COUNT ? kWgForwardNames[slot] : "?");
  void* p = slot < WG_BINDING_FORWARD_COUNT ? resolve_original(kWgForwardNames[slot]) : nullptr;
  if (!p) {
    log_line(std::string("proxy: export missing in the original: ") +
             (slot < WG_BINDING_FORWARD_COUNT ? kWgForwardNames[slot] : "?"));
    p = (void*)&missing_export;
  }
  InterlockedExchangePointer(&wg_fwd_table[slot], p);
  return p;
}

#if WG_RR_PROBE
// Probe build: every graphics / Vulkan entry point passes here (argument registers saved by the stub): logged by
// name, then resolved like wg_fwd_resolve.
void* wg_fwd_note(uint32_t slot) {
  if (slot < WG_BINDING_FORWARD_COUNT) entered(kWgForwardNames[slot])->note_forward(kWgForwardNames[slot]);
  void* p = slot < WG_BINDING_FORWARD_COUNT ? wg_fwd_table[slot] : nullptr;
  return p ? p : wg_fwd_resolve(slot);
}
#endif

// ---- roles answered or recorded by the proxy (60.3.1; types of A10.2) ----
WgResultCode wg_role_init(const WgPreferences* p, uint64_t v) {
  auto* t = entered(0);
  log_init_entered(p, v);  // before the init is forwarded
  return t->init(p, v);
}
WgResultCode wg_role_shutdown(void) { return entered(1)->shutdown(); }
WgResultCode wg_role_is_supported(uint32_t f, const WgAdapterInfo* a) { return entered(2)->is_supported(f, a); }
WgResultCode wg_role_is_loaded(uint32_t f, bool* l) { return entered(3)->is_loaded(f, l); }
WgResultCode wg_role_set_loaded(uint32_t f, bool l) { return entered(4)->set_loaded(f, l); }
WgResultCode wg_role_requirements(uint32_t f, WgRequirements* r) { return entered(5)->requirements(f, r); }
WgResultCode wg_role_version(uint32_t f, WgFeatureVersion* v) { return entered(6)->version(f, v); }
WgResultCode wg_role_get_function(uint32_t f, const char* n, void** o) { return entered(7)->get_function(f, n, o); }
WgResultCode wg_role_allocate(void* c, uint32_t f, const WgViewport* v) { return entered(8)->allocate(c, f, v); }
WgResultCode wg_role_free(uint32_t f, const WgViewport* v) { return entered(9)->free_resources(f, v); }
WgResultCode wg_role_set_device(void* d) { return entered(10)->set_device(d); }
WgResultCode wg_role_tag(const WgViewport* v, const WgTag* t, uint32_t n, void* c) {
  return entered(12)->tag(v, t, n, c);
}
WgResultCode wg_role_tag_for_frame(const WgFrameToken* k, const WgViewport* v, const WgTag* t, uint32_t n, void* c) {
  return entered(13)->tag_for_frame(k, v, t, n, c);
}
WgResultCode wg_role_constants(const WgConstants* c, const WgFrameToken* k, const WgViewport* v) {
  return entered(14)->constants(c, k, v);
}
WgResultCode wg_role_new_frame_token(WgFrameToken** o, const uint32_t* i) { return entered(15)->new_frame_token(o, i); }
WgResultCode wg_role_evaluate(uint32_t f, const WgFrameToken* k, const WgHeader* const* in, uint32_t n, void* c) {
  return entered(16)->evaluate(f, k, in, n, c);
}
#if WG_RR_PROBE
// The three API calls the release build forwards through stubs; typed here (A10.2) so their results are logged.
WgResultCode wg_role_set_vulkan_info(const void* info) { return entered(11)->set_vulkan_info(info); }
WgResultCode wg_role_upgrade_interface(void** iface) { return entered(17)->upgrade_interface(iface); }
WgResultCode wg_role_native_interface(void* w, void** n) { return entered(18)->native_interface(w, n); }
#endif

// ---- RR feature functions (60.3.2, A10.3): valid until unload ----
WgResultCode wg_rr_optimal_fn(const WgRrOptions* o, WgRrOptimal* s) { return entered("rr-optimal")->rr_optimal(o, s); }
WgResultCode wg_rr_state_fn(const WgViewport* v, WgRrState* s) { return entered("rr-state")->rr_state(v, s); }
WgResultCode wg_rr_options_fn(const WgViewport* v, const WgRrOptions* o) {
  return entered("rr-options")->rr_options(v, o);
}

// ---- 60.9 the proxy's own export ----
const WgRrSettingsContract* wg_get_settings_contract(void) {
  diagnostics(WG_RR_CONTRACT_EXPORT);
  ensure_settings();
  return wgrr::settings_contract();
}

}  // extern "C"

BOOL WINAPI DllMain(HINSTANCE inst, DWORD reason, LPVOID) {
  if (reason == DLL_PROCESS_ATTACH) {
    g_self = inst;
    DisableThreadLibraryCalls(inst);
    record_attach(inst);  // recorded only; written at the first call
  }
  return TRUE;  // nothing is loaded or freed under the loader lock
}
