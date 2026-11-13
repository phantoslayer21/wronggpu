// SPDX-License-Identifier: MIT
// RR guard: see rr_guard.h.
#define WIN32_LEAN_AND_MEAN
#include <windows.h>

#include <string.h>

#include "rr_guard.h"

namespace wgrr {
namespace guard {

bool readable(const void* p, size_t n) {
  if (!p || !n) return false;
  uintptr_t a = reinterpret_cast<uintptr_t>(p);
  const uintptr_t end = a + n;
  if (end < a) return false;
  while (a < end) {
    MEMORY_BASIC_INFORMATION m;
    if (!VirtualQuery(reinterpret_cast<const void*>(a), &m, sizeof m)) return false;
    if (m.State != MEM_COMMIT) return false;
    const DWORD pr = m.Protect;
    if (pr & (PAGE_GUARD | PAGE_NOACCESS)) return false;
    // PAGE_EXECUTE counts: a hooking library's trampolines may be execute-only, and the copy that follows is guarded
    if (!(pr & (PAGE_READONLY | PAGE_READWRITE | PAGE_WRITECOPY | PAGE_EXECUTE | PAGE_EXECUTE_READ | PAGE_EXECUTE_READWRITE |
                PAGE_EXECUTE_WRITECOPY)))
      return false;
    const uintptr_t next = reinterpret_cast<uintptr_t>(m.BaseAddress) + m.RegionSize;
    if (next <= a) return false;
    a = next;
  }
  return true;
}

namespace {
// No C++ objects in this frame (structured exception handling).
bool copy_guarded(const void* p, void* out, size_t n) {
  __try {
    const volatile uint8_t* s = static_cast<const volatile uint8_t*>(p);
    uint8_t* d = static_cast<uint8_t*>(out);
    for (size_t i = 0; i < n; ++i) d[i] = s[i];
    return true;
  } __except (EXCEPTION_EXECUTE_HANDLER) {
    return false;
  }
}

// Up to `want` code bytes from p, never past the readable region p lies in. *got: bytes read.
bool read_code(const void* p, uint8_t* buf, size_t want, size_t* got) {
  *got = 0;
  MEMORY_BASIC_INFORMATION m;
  if (!p || !VirtualQuery(p, &m, sizeof m)) return false;
  const uintptr_t end = reinterpret_cast<uintptr_t>(m.BaseAddress) + m.RegionSize;
  const uintptr_t at = reinterpret_cast<uintptr_t>(p);
  if (end <= at) return false;
  const size_t n = (end - at) < want ? (size_t)(end - at) : want;
  if (!safe_read(p, buf, n)) return false;
  *got = n;
  return true;
}

int32_t rd32(const uint8_t* b) {
  int32_t v;
  memcpy(&v, b, 4);
  return v;
}
}  // namespace

bool safe_read(const void* p, void* out, size_t n) {
  if (!out || !readable(p, n)) return false;
  return copy_guarded(p, out, n);
}

bool Reader::read(const void* p, void* out, size_t n) {
  const uintptr_t a = reinterpret_cast<uintptr_t>(p);
  if (!p || !out || !n || a + n < a) return false;
  if (a < lo_ || a + n > hi_) {
    MEMORY_BASIC_INFORMATION m;
    ++queries;
    if (!VirtualQuery(p, &m, sizeof m)) return false;
    const uintptr_t lo = reinterpret_cast<uintptr_t>(m.BaseAddress), hi = lo + m.RegionSize;
    if (a + n > hi) return safe_read(p, out, n);  // spans regions: the uncached check
    if (!readable(p, n)) return false;
    lo_ = lo;
    hi_ = hi;
  }
  return copy_guarded(p, out, n);
}

bool Reader::str(const void* p, char* out, size_t max) {
  const char* s = static_cast<const char*>(p);
  for (size_t i = 0; i < max; ++i) {
    if (!get(s + i, &out[i])) return false;
    if (!out[i]) return true;
  }
  return false;
}

bool decode_jump(const uint8_t* c, size_t n, const uint8_t* at, const void** target, const void** cell) {
  if (cell) *cell = nullptr;
  if (n >= 5 && c[0] == 0xE9) {  // jmp rel32
    *target = at + 5 + rd32(c + 1);
    return true;
  }
  if (n >= 2 && c[0] == 0xEB) {  // jmp rel8
    *target = at + 2 + static_cast<int8_t>(c[1]);
    return true;
  }
  const size_t r = (n >= 1 && c[0] == 0x48) ? 1 : 0;  // REX.W jmp [rip+disp32] (same meaning)
  if (n >= r + 6 && c[r] == 0xFF && c[r + 1] == 0x25) {  // jmp [rip+disp32]
    const uint8_t* slot = at + r + 6 + rd32(c + r + 2);
    const void* t = nullptr;
    if (!safe_get(slot, &t)) return false;
    if (cell) *cell = slot;
    *target = t;
    return true;
  }
  if (n >= 12 && c[0] == 0x48 && c[1] == 0xB8 && ((c[10] == 0xFF && c[11] == 0xE0) || (c[10] == 0x50 && c[11] == 0xC3))) {
    const void* t;  // mov rax, imm64; jmp rax | push rax; ret
    memcpy(&t, c + 2, sizeof t);
    *target = t;
    return true;
  }
  if (n >= 13 && c[0] == 0x49 && c[1] == 0xBB && c[10] == 0x41 && c[11] == 0xFF && c[12] == 0xE3) {
    const void* t;  // mov r11, imm64; jmp r11
    memcpy(&t, c + 2, sizeof t);
    *target = t;
    return true;
  }
  return false;
}

const char* owner_word(Owner o) {
  switch (o) {
    case Owner::kUnknown: return "unknown (not readable, or too many hook jumps)";
    case Owner::kSelf: return "WrongGPU RR (this copy)";
    case Owner::kInterface: return "the game interface";
    case Owner::kHost: return "the host";
    case Owner::kOwnCopy: return "another copy of WrongGPU RR";
    case Owner::kWrongGpu: return "another WrongGPU module";
    case Owner::kExeThunk: return "the exe's own code";
    case Owner::kAnonymous: return "code outside every module";
    case Owner::kForeign: return "another module";
  }
  return "?";
}

Where classify(const void* p, const Env& env, bool follow_iface) {
  Where w;
  const void* cur = p;
  // follows one hook jump to `target`; false (kUnknown, too_many_hops) when kMaxHops jumps were followed already
  auto follow = [&](const void* target) {
    if (w.hops >= kMaxHops) {
      w.owner = Owner::kUnknown;
      w.too_many_hops = true;
      return false;
    }
    cur = target;
    ++w.hops;
    return true;
  };
  for (;;) {
    if (!cur) {
      w.owner = Owner::kUnknown;
      return w;
    }
    const void* m = env.module_of ? env.module_of(cur) : nullptr;
    w.module = m && env.name_of ? env.name_of(m) : std::string();
    w.module_handle = m;
    if (m && m == env.self) { w.owner = Owner::kSelf; return w; }
    if (m && (m == env.host || (env.host_alt && m == env.host_alt))) { w.owner = Owner::kHost; return w; }
    if (m && m == env.iface) {
      if (!follow_iface || w.hops != 0) { w.owner = Owner::kInterface; return w; }
      uint8_t code[16];
      size_t got = 0;
      if (!read_code(cur, code, sizeof code, &got)) { w.owner = Owner::kUnknown; return w; }
      const void* target = nullptr;
      const void* cell = nullptr;
      if (!decode_jump(code, got, static_cast<const uint8_t*>(cur), &target, &cell)) {
        // a jmp [rip+disp32] whose cell could not be read decodes as "no jump": check that case explicitly
        const size_t r = (got >= 1 && code[0] == 0x48) ? 1 : 0;
        if (got >= r + 6 && code[r] == 0xFF && code[r + 1] == 0x25) { w.owner = Owner::kUnknown; return w; }
        w.owner = Owner::kInterface;
        return w;
      }
      const uint8_t* c8 = static_cast<const uint8_t*>(cell);
      if (c8 && c8 >= env.iface_iat_begin && c8 < env.iface_iat_end) { w.owner = Owner::kInterface; return w; }
      if (env.module_of && env.module_of(target) == env.iface) { w.owner = Owner::kInterface; return w; }
      if (!follow(target)) return w;
      continue;
    }
    if (m) {
      if (env.exports && env.exports(m, kOwnRrExport)) w.owner = Owner::kOwnCopy;
      else if (env.exports && env.exports(m, kWrongGpuAsiExport)) w.owner = Owner::kWrongGpu;
      else if (m == env.exe) w.owner = Owner::kExeThunk;
      else w.owner = Owner::kForeign;
      return w;
    }
    // outside every module: a trampoline / relay; follow its jump, if any
    uint8_t code[16];
    size_t got = 0;
    if (!read_code(cur, code, sizeof code, &got)) { w.owner = Owner::kUnknown; return w; }
    const void* target = nullptr;
    if (!decode_jump(code, got, static_cast<const uint8_t*>(cur), &target, nullptr)) {
      const size_t r = (got >= 1 && code[0] == 0x48) ? 1 : 0;
      w.owner = (got >= r + 6 && code[r] == 0xFF && code[r + 1] == 0x25) ? Owner::kUnknown : Owner::kAnonymous;
      return w;
    }
    if (!follow(target)) return w;
  }
}

bool trusted(Owner o, bool anon_ok, bool exe_ok) {
  switch (o) {
    case Owner::kSelf: case Owner::kInterface: case Owner::kHost: case Owner::kWrongGpu: return true;
    case Owner::kAnonymous: return anon_ok;
    case Owner::kExeThunk: return exe_ok;
    default: return false;
  }
}

void Verdict::add(const Where& w, const std::string& what, bool anon_ok, bool exe_ok) {
  ++reads;
  if (kind == kOwnCopy) return;
  const std::string via = w.hops ? " (through " + std::to_string(w.hops) + " hook jump" + (w.hops > 1 ? "s" : "") + ")"
                                 : std::string();
  if (w.owner == Owner::kOwnCopy) {
    kind = kOwnCopy;
    detail = w.module;
    return;
  }
  if (trusted(w.owner, anon_ok, exe_ok)) return;
  if (w.owner == Owner::kForeign) {  // the one positive sign of another plugin: a named module that is not ours
    if (kind == kOther) return;
    kind = kOther;
    other_module = w.module_handle;
    detail = what + " leads to " + (w.module.empty() ? std::string("another module") : w.module) + via;
    return;
  }
  // not a sign of another plugin: counted, the first one named for the log
  ++unclear;
  if (unclear > 1) return;
  if (w.owner == Owner::kUnknown)
    unclear_detail = w.too_many_hops ? what + " leads through more than " + std::to_string(kMaxHops) + " hook jumps"
                                     : what + " could not be read" + via;
  else if (w.owner == Owner::kAnonymous) unclear_detail = what + " leads to code outside every module" + via;
  else unclear_detail = what + " leads into the exe's own code" + via;
}

bool override_take(const std::string& value, std::string* note) {
  std::string v;
  for (char c : value)
    if (c != ' ' && c != '\t' && c != '\r' && c != '\n') v.push_back((char)(c >= 'A' && c <= 'Z' ? c - 'A' + 'a' : c));
  if (note) note->clear();
  if (v == "take") return true;
  if (!v.empty() && v != "off" && note)
    *note = std::string(kOverrideKey) + "=" + value + " in wronggpu-rr.ini not recognised (off | take); off is used";
  return false;
}

// The lines start with the words the launcher's Last run page matches (no "hook: " prefix): kStem, kOwnCopyStem,
// kLaterStem. HOOK-MODE.md documents them; changing a beginning breaks that page.
static const char kStem[] = "another Ray Reconstruction plugin is active; WrongGPU RR stays off (";
static const char kOwnCopyStem[] = "a second copy of WrongGPU RR is loaded (";
static const char kLaterStem[] = "a plugin loaded after WrongGPU replaced its hooks; ";

std::string line_other(const std::string& d, bool passing) {
  return std::string(kStem) + d +
         (passing ? "; WrongGPU's handlers pass every call on unchanged; " : "; nothing patched; ") + kOverrideKey +
         "=take in wronggpu-rr.ini uses WrongGPU's instead)";
}
std::string line_other_later(const std::string& d) {
  return std::string(kStem) + d +
         "; it loaded after WrongGPU: WrongGPU passes every call on unchanged; to use WrongGPU RR, move it aside)";
}
std::string line_take(const std::string& d) {
  return std::string("another Ray Reconstruction plugin is active; ") + kOverrideKey +
         "=take: WrongGPU RR goes in front of it (" + d + "; it gets the game's other calls, no RR work)";
}
std::string line_cannot_take(const std::string& d) {
  return std::string(kLaterStem) + kOverrideKey +
         "=take cannot be applied: WrongGPU RR stays off and passes every call on unchanged (" + d +
         "); move it aside to use WrongGPU RR";
}
std::string line_own_copy(const std::string& active, const std::string& self) {
  return std::string(kOwnCopyStem) + active + " is active); this copy (" + self + ") stays off";
}
std::string line_ngx_other() {
  return std::string(kStem) +
         "NGX route only: the host refused WrongGPU's RR provider, another provider holds it; the game-interface "
         "route is decided on its own)";
}
std::string line_inconclusive(const std::string& when, unsigned n, const std::string& first) {
  return "hook: coexistence check " + when + ": inconclusive for " + std::to_string(n) + " address(es) (first: " + first +
         "); no other plugin was positively found, WrongGPU RR stays active";
}

void Decision::freeze_once(const std::function<State()>& decide) {
  if (frozen()) return;
  std::call_once(once_, [&] {
    std::lock_guard<std::mutex> lock(mu_);
    set(decide());
    frozen_.store(true, std::memory_order_release);
  });
}

void Decision::check(const std::function<State()>& decide) {
  std::lock_guard<std::mutex> lock(mu_);
  if (frozen()) return;
  set(decide());
}

// ---------------------------------------------------------------- the real process
namespace {
std::string leaf_utf8(const void* m) {
  wchar_t buf[MAX_PATH * 2];
  const DWORD n = GetModuleFileNameW(static_cast<HMODULE>(const_cast<void*>(m)), buf, MAX_PATH * 2);
  std::wstring p(buf, n);
  const size_t slash = p.find_last_of(L"\\/");
  if (slash != std::wstring::npos) p = p.substr(slash + 1);
  const int k = WideCharToMultiByte(CP_UTF8, 0, p.c_str(), (int)p.size(), nullptr, 0, nullptr, nullptr);
  std::string s(k > 0 ? k : 0, '\0');
  if (k > 0) WideCharToMultiByte(CP_UTF8, 0, p.c_str(), (int)p.size(), &s[0], k, nullptr, nullptr);
  return s;
}
}  // namespace

Env process_env(const void* self, const void* iface, const void* host, const void* host_alt) {
  Env e;
  e.self = self;
  e.iface = iface;
  e.host = host;
  e.host_alt = host_alt;
  e.exe = GetModuleHandleW(nullptr);
  e.module_of = [](const void* a) -> const void* {
    HMODULE m = nullptr;
    if (!a || !GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                                  static_cast<LPCWSTR>(a), &m))
      return nullptr;
    return m;
  };
  e.exports = [](const void* m, const char* name) {
    return GetProcAddress(static_cast<HMODULE>(const_cast<void*>(m)), name) != nullptr;
  };
  e.name_of = leaf_utf8;
  // the interface's own import address table (IMAGE_DIRECTORY_ENTRY_IAT), read through safe reads
  const uint8_t* base = static_cast<const uint8_t*>(iface);
  IMAGE_DOS_HEADER dos;
  IMAGE_NT_HEADERS64 nt;
  if (base && safe_get(base, &dos) && dos.e_magic == IMAGE_DOS_SIGNATURE && safe_get(base + dos.e_lfanew, &nt) &&
      nt.Signature == IMAGE_NT_SIGNATURE) {
    const IMAGE_DATA_DIRECTORY& d = nt.OptionalHeader.DataDirectory[IMAGE_DIRECTORY_ENTRY_IAT];
    if (d.VirtualAddress && d.Size) {
      e.iface_iat_begin = base + d.VirtualAddress;
      e.iface_iat_end = base + d.VirtualAddress + d.Size;
    }
  }
  return e;
}

}  // namespace guard
}  // namespace wgrr
