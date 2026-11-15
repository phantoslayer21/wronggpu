// SPDX-License-Identifier: MIT
// RR guard: see rr_guard.h.
#define WIN32_LEAN_AND_MEAN
#include <windows.h>

#include <stddef.h>
#include <stdio.h>
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

// at + d as an address (integer arithmetic: a jump target may lie anywhere, it is only ever read through safe reads)
const void* rel(const uint8_t* at, intptr_t d) {
  return reinterpret_cast<const void*>(reinterpret_cast<uintptr_t>(at) + (uintptr_t)d);
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

size_t mapped_extent(const void* p) {
  MEMORY_BASIC_INFORMATION m;
  if (!p || !VirtualQuery(p, &m, sizeof m) || m.State != MEM_COMMIT) return 0;
  const void* alloc = m.AllocationBase;
  const uintptr_t a = reinterpret_cast<uintptr_t>(p);
  uintptr_t end = reinterpret_cast<uintptr_t>(m.BaseAddress) + m.RegionSize;
  for (unsigned i = 0; i < 4096 && end > a; ++i) {
    if (!VirtualQuery(reinterpret_cast<const void*>(end), &m, sizeof m) || m.AllocationBase != alloc ||
        m.State != MEM_COMMIT)
      break;
    const uintptr_t next = reinterpret_cast<uintptr_t>(m.BaseAddress) + m.RegionSize;
    if (next <= end) break;
    end = next;
  }
  return end > a ? (size_t)(end - a) : 0;
}

bool PeView::open(const void* module, Reader& r) {
  base_ = nullptr;
  size_ = 0;
  ndirs_ = 0;
  const uint8_t* base = static_cast<const uint8_t*>(module);
  // what is really mapped at the module bounds every header read below (SizeOfImage is only a claim of the header)
  const size_t mapped = mapped_extent(base);
  IMAGE_DOS_HEADER dos;
  if (mapped < sizeof dos || !r.get(base, &dos) || dos.e_magic != IMAGE_DOS_SIGNATURE) return false;
  // e_lfanew is a signed 32-bit offset: checked as an integer (positive, past the DOS header, not absurd, the NT
  // headers inside the mapping) before the NT headers are read at all; against SizeOfImage once that is known
  const LONG lfanew = dos.e_lfanew;
  if (lfanew < (LONG)sizeof(IMAGE_DOS_HEADER) || lfanew > 0x100000) return false;
  if ((uint64_t)lfanew + sizeof(IMAGE_NT_HEADERS64) > mapped) return false;
  IMAGE_NT_HEADERS64 nt;
  if (!r.get(base + lfanew, &nt) || nt.Signature != IMAGE_NT_SIGNATURE ||
      nt.OptionalHeader.Magic != IMAGE_NT_OPTIONAL_HDR64_MAGIC)
    return false;
  const uint32_t size = nt.OptionalHeader.SizeOfImage;
  if ((uint64_t)lfanew + sizeof(IMAGE_NT_HEADERS64) > size) return false;  // the headers lie inside the image
  if ((uintptr_t)base + size < (uintptr_t)base) return false;              // the image does not wrap the address space
  base_ = base;
  size_ = (uint64_t)size < (uint64_t)mapped ? size : (uint32_t)mapped;  // a larger SizeOfImage does not widen the image
  ndirs_ = nt.OptionalHeader.NumberOfRvaAndSizes < 16 ? nt.OptionalHeader.NumberOfRvaAndSizes : 16;
  for (uint32_t i = 0; i < ndirs_; ++i) {
    dir_rva_[i] = nt.OptionalHeader.DataDirectory[i].VirtualAddress;
    dir_size_[i] = nt.OptionalHeader.DataDirectory[i].Size;
  }
  return true;
}

bool PeView::dir(unsigned index, uint32_t* rva, uint32_t* size) const {
  if (!base_ || index >= ndirs_ || !dir_rva_[index] || !dir_size_[index]) return false;
  if (!at(dir_rva_[index], dir_size_[index])) return false;  // the directory lies inside the image
  *rva = dir_rva_[index];
  *size = dir_size_[index];
  return true;
}

const uint8_t* PeView::at(uint64_t rva, uint64_t n) const {
  if (!base_ || rva > size_ || n > (uint64_t)size_ - rva) return nullptr;
  return base_ + rva;
}

bool PeView::str(uint64_t rva, char* out, size_t max, Reader& r) const {
  for (size_t i = 0; i < max; ++i) {
    const uint8_t* p = at(rva + i, 1);  // every byte inside the extent: a name cut off by the image's end is refused
    if (!p || !r.get(p, &out[i])) return false;
    if (!out[i]) return true;
  }
  return false;
}

Budget::Budget(unsigned max_entries, unsigned max_ms) : max_entries_(max_entries), max_ms_((double)max_ms) {
  LARGE_INTEGER f, t;
  QueryPerformanceFrequency(&f);
  QueryPerformanceCounter(&t);
  freq_ = f.QuadPart > 0 ? f.QuadPart : 1;
  t0_ = t.QuadPart;
}

double Budget::elapsed_ms() const {
  LARGE_INTEGER t;
  QueryPerformanceCounter(&t);
  return (double)(t.QuadPart - t0_) * 1000.0 / (double)freq_;
}

bool Budget::step() {
  if (over_) return false;
  ++entries_;
  if (entries_ > max_entries_ || ((entries_ & 63) == 0 && elapsed_ms() > max_ms_)) over_ = true;
  return !over_;
}

void thunks(const PeView& pe, Reader& r, Budget& b, uint32_t names_rva, uint32_t iat_rva,
            const std::function<void(const char* name, void** slot)>& f) {
  for (unsigned i = 0; i < 4096 && b.step(); ++i) {
    const uint64_t off = (uint64_t)i * sizeof(IMAGE_THUNK_DATA64);
    const uint8_t* np = pe.at(names_rva + off, sizeof(IMAGE_THUNK_DATA64));
    const uint8_t* sp = pe.at(iat_rva + off, sizeof(IMAGE_THUNK_DATA64));
    IMAGE_THUNK_DATA64 t;
    if (!np || !sp || !r.get(np, &t) || !t.u1.AddressOfData) return;
    if (t.u1.Ordinal & IMAGE_ORDINAL_FLAG64) continue;
    char nm[96];
    if (!pe.str(t.u1.AddressOfData + offsetof(IMAGE_IMPORT_BY_NAME, Name), nm, sizeof nm, r)) continue;
    f(nm, const_cast<void**>(reinterpret_cast<void* const*>(sp)));
  }
}

void import_slots(const void* module, const char* leaf, Reader& r, Budget& b,
                  const std::function<void(const char* name, void** slot)>& f) {
  PeView pe;
  uint32_t rva = 0, size = 0;
  if (!pe.open(module, r) || !pe.dir(IMAGE_DIRECTORY_ENTRY_IMPORT, &rva, &size)) return;
  // the descriptors the directory declares, no more (the null descriptor usually ends the walk earlier)
  const uint32_t count = size / sizeof(IMAGE_IMPORT_DESCRIPTOR) < 4096 ? size / sizeof(IMAGE_IMPORT_DESCRIPTOR) : 4096;
  for (uint32_t k = 0; k < count && b.step(); ++k) {
    const uint8_t* dp = pe.at(rva + (uint64_t)k * sizeof(IMAGE_IMPORT_DESCRIPTOR), sizeof(IMAGE_IMPORT_DESCRIPTOR));
    IMAGE_IMPORT_DESCRIPTOR e;
    if (!dp || !r.get(dp, &e) || !e.Name) return;
    char dll[64];
    if (!pe.str(e.Name, dll, sizeof dll, r) || _stricmp(dll, leaf) != 0) continue;
    thunks(pe, r, b, e.OriginalFirstThunk ? e.OriginalFirstThunk : e.FirstThunk, e.FirstThunk, f);
  }
}

bool decode_jump(const uint8_t* c, size_t n, const uint8_t* at, const void** target, const void** cell) {
  if (cell) *cell = nullptr;
  if (n >= 5 && c[0] == 0xE9) {  // jmp rel32
    *target = rel(at, 5 + (intptr_t)rd32(c + 1));
    return true;
  }
  if (n >= 2 && c[0] == 0xEB) {  // jmp rel8
    *target = rel(at, 2 + (intptr_t)static_cast<int8_t>(c[1]));
    return true;
  }
  const size_t r = (n >= 1 && c[0] == 0x48) ? 1 : 0;  // REX.W jmp [rip+disp32] (same meaning)
  if (n >= r + 6 && c[r] == 0xFF && c[r + 1] == 0x25) {  // jmp [rip+disp32]
    const void* slot = rel(at, (intptr_t)(r + 6) + rd32(c + r + 2));
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
  if (w.owner == Owner::kForeign) {  // a named module that is not ours: inconclusive, named (the first one) in the log
    ++foreign;
    if (kind == kForeign) return;
    kind = kForeign;
    other_module = w.module_handle;
    module = w.module.empty() ? std::string("another module") : w.module;
    detail = what + " leads to " + module + via;
    return;
  }
  // nothing could be told: counted, the first one named for the log
  ++unclear;
  if (unclear > 1) return;
  if (w.owner == Owner::kUnknown)
    unclear_detail = w.too_many_hops ? what + " leads through more than " + std::to_string(kMaxHops) + " hook jumps"
                                     : what + " could not be read" + via;
  else if (w.owner == Owner::kAnonymous) unclear_detail = what + " leads to code outside every module" + via;
  else unclear_detail = what + " leads into the exe's own code" + via;
}

Mode parse_mode(const std::string& value, std::string* note) {
  std::string v;
  for (char c : value)
    if (c != ' ' && c != '\t' && c != '\r' && c != '\n') v.push_back((char)(c >= 'A' && c <= 'Z' ? c - 'A' + 'a' : c));
  if (note) note->clear();
  if (v == "take") return Mode::kTake;
  if (v == "yield") return Mode::kYield;
  if (!v.empty() && v != "off" && note)
    *note = std::string(kOverrideKey) + "=" + value + " in wronggpu-rr.ini not recognised (off | take | yield); off is used";
  return Mode::kOff;
}

const char* mode_word(Mode m) {
  switch (m) {
    case Mode::kOff: return "off";
    case Mode::kTake: return "take";
    case Mode::kYield: return "yield";
  }
  return "?";
}

State decide(Verdict::Kind kind, Mode mode, Stage stage, State current) {
  if (current != State::kActive && current != State::kTaken) return current;  // off, own copy, passing: final
  switch (kind) {
    case Verdict::kClean:
      return current;
    case Verdict::kOwnCopy:  // positive by contract
      return stage == Stage::kLoad ? State::kOwnCopy : State::kPassThrough;
    case Verdict::kForeign:  // inconclusive: only the player's setting acts on it
      if (mode == Mode::kYield) return stage == Stage::kLoad ? State::kOff : State::kPassThrough;
      if (mode == Mode::kTake && stage != Stage::kLate) return State::kTaken;
      return current;
  }
  return current;
}

// The beginnings a reader can match (HOOK-MODE.md documents them; keep them exactly): kForeignStem, kRefusedStem (the
// NGX route's refusal), kOwnCopyStem. None of them has the "hook: " prefix of the diagnostics.
static const char kForeignStem[] = "another module hooks the game's RR calls: ";
static const char kRefusedStem[] =
    "the host refused WrongGPU's NGX RR provider registration; the NGX route stays off (";
static const char kOwnCopyStem[] = "a second copy of WrongGPU RR is loaded (";

std::string line_foreign(const std::string& module, const std::string& detail, Tail tail) {
  std::string s = std::string(kForeignStem) + module + " (" + detail + "); ";
  switch (tail) {
    case Tail::kStaysActive:
      return s + "it is not known to be a Ray Reconstruction plugin: WrongGPU RR stays active (" + kOverrideKey +
             "=yield in wronggpu-rr.ini keeps WrongGPU RR off in this case)";
    case Tail::kTakeInFront:
      return s + kOverrideKey + "=take: WrongGPU RR goes in front of it (it gets the game's other calls, no RR work)";
    case Tail::kYieldOff:
      return s + kOverrideKey + "=yield: WrongGPU RR stays off (nothing patched)";
    case Tail::kYieldPassing:
      return s + kOverrideKey + "=yield: WrongGPU RR stays off (WrongGPU's handlers pass every call on unchanged)";
  }
  return s;
}
std::string line_own_copy(const std::string& active, const std::string& self) {
  return std::string(kOwnCopyStem) + active + " is active); this copy (" + self + ") stays off";
}
std::string line_ngx_refused() {
  // the host API gives one false for an occupied slot and for an incomplete structure: only the refusal is known
  return std::string(kRefusedStem) + "the game-interface route is decided on its own)";
}
std::string line_inconclusive(const std::string& when, unsigned n, const std::string& first) {
  return "hook: coexistence check " + when + ": inconclusive for " + std::to_string(n) + " address(es) (first: " + first +
         "); no other plugin was positively found, WrongGPU RR stays active";
}

std::string line_over_budget(unsigned entries, double ms) {
  char t[32];
  snprintf(t, sizeof t, "%.1f ms", ms);
  return "hook: coexistence check at load: stopped after " + std::to_string(entries) + " entries (" + t +
         "), over its budget of " + std::to_string(kScanMaxEntries) + " entries / " + std::to_string(kScanMaxMs) +
         " ms; inconclusive, WrongGPU RR stays active";
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
ModuleRef::ModuleRef(const void* m) {
  HMODULE h = nullptr;
  if (m && GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS, static_cast<LPCWSTR>(m), &h)) h_ = h;
}
ModuleRef::~ModuleRef() {
  if (h_) FreeLibrary(static_cast<HMODULE>(h_));
}

namespace {
std::string leaf_utf8(const void* m) {
  const ModuleRef ref(m);  // the module stays loaded while its name is read
  if (!ref.held()) return std::string();
  wchar_t buf[MAX_PATH * 2];
  const DWORD n = GetModuleFileNameW(static_cast<HMODULE>(ref.handle()), buf, MAX_PATH * 2);
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
    const ModuleRef ref(m);  // the module stays loaded while its exports are looked up (unloaded: exports nothing)
    return ref.held() && GetProcAddress(static_cast<HMODULE>(ref.handle()), name) != nullptr;
  };
  e.name_of = leaf_utf8;
  // the interface's own import address table (IMAGE_DIRECTORY_ENTRY_IAT), read through safe reads and checked against
  // the image's size (PeView)
  Reader r;
  PeView pe;
  uint32_t rva = 0, size = 0;
  if (iface && pe.open(iface, r) && pe.dir(IMAGE_DIRECTORY_ENTRY_IAT, &rva, &size)) {
    e.iface_iat_begin = pe.at(rva, size);
    e.iface_iat_end = e.iface_iat_begin + size;
  }
  return e;
}

}  // namespace guard
}  // namespace wgrr
