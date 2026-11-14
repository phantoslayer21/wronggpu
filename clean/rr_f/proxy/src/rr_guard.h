// SPDX-License-Identifier: MIT
// RR guard: the coexistence guard of the hook-mode RR plugin (HOOK-MODE.md, "Another Ray Reconstruction
// plugin"). Generic and read-only: it looks at where the game's RR calls lead (import slots, the first bytes of the
// interface's RR exports, the host's continuation pointers), never at another plugin's files or internals.
//
// WrongGPU RR switches itself off automatically ONLY on evidence that is positive by contract:
//   - a second copy of WrongGPU RR (the process-wide marker, or an address that ends in a module exporting
//     kOwnRrExport);
//   - the host's plugin API refusing a registration (hook_dll.cpp): the route it refused stays off. The API does not
//     say why (an occupied slot and an incomplete structure give the same false), so the log says only that it
//     refused.
// Nothing else. A named foreign module on the call path is INCONCLUSIVE: it may be another RR plugin, but it may as well
// be a game's own wrapper DLL, an overlay, a security shim, a host helper in a third image or a forwarded export's
// implementation DLL. WrongGPU RR stays active (as in 0.6.0) and one line names the module and the place
// (line_foreign). wronggpu-rr.ini rr_other_plugin=yield makes such a module switch WrongGPU RR off; take puts WrongGPU RR
// in front of it. What the scan cannot explain at all (unreadable memory, too many hops, code outside every module that
// does not jump on, the exe's own code) is logged once as inconclusive (line_inconclusive) and changes nothing either.
//
// This header holds the parts without process state, so the CPU tests can drive them with synthetic memory:
//   - safe_read: every read of memory we do not own (VirtualQuery for committed, readable, non-guard pages, then a
//     structured-exception-guarded copy); a failed read is "unknown", never a fault. The check and the copy are not
//     atomic (HOOK-MODE.md says what that leaves);
//   - PeView: a module's PE64 headers read through the reader. The image's extent is the smaller of SizeOfImage and
//     what is really mapped there (mapped_extent); e_lfanew, every data directory, every RVA the scan follows and
//     every byte of a name are checked against that extent before an address is formed from them;
//   - thunks / import_slots: the import walk of the load scan (descriptors only within the import directory's
//     declared size), under a Budget (entries visited, elapsed time) that ends the scan as inconclusive when spent;
//   - decode_jump: the jump forms a hook puts at the start of a function (jmp rel32 / rel8, jmp [rip+disp32],
//     mov rax|r11, imm64 + jmp reg);
//   - classify: who an address belongs to (our module, the game interface, the host, another copy of WrongGPU RR,
//     another WrongGPU module, the exe's own delay-load thunk, code outside every module, another module), following
//     hook jumps out of the interface and out of anonymous memory, at most kMaxHops of them;
//   - Verdict: what several classifications found (a copy of WrongGPU RR, the first foreign module, the inconclusive
//     ones) and the words for the log;
//   - Mode / decide: what a verdict means for WrongGPU RR under rr_other_plugin (one table, tested on its own);
//   - Decision: the once-only late decision (several game threads may make the first call at the same time).
#pragma once
#include <stddef.h>
#include <stdint.h>

#include <atomic>
#include <functional>
#include <mutex>
#include <string>

namespace wgrr {
namespace guard {

// The export a module must have to count as a copy of WrongGPU RR (the 60.9 settings contract), and the one of the
// other WrongGPU plugin (WrongGPU.asi, Model M's menu plugin). Documented in HOOK-MODE.md (the launcher matches them).
constexpr const char* kOwnRrExport = "WrongGPU_GetSettingsContract";
constexpr const char* kWrongGpuAsiExport = "WrongGpuAsiQuery";
// wronggpu-rr.ini: rr_other_plugin = off (default) | take | yield
constexpr const char* kOverrideKey = "rr_other_plugin";
constexpr int kMaxHops = 4;  // hook jumps followed from one address (a fifth one is "too many hops": inconclusive)

// True when [p, p + n) lies in committed pages that are readable (execute-only pages count: code is read from them and
// the structured-exception guard of the copy covers a fault) and not guard pages, as VirtualQuery reports them now.
bool readable(const void* p, size_t n);
// Copies n bytes from p to out when readable() and the copy raises no structured exception. False: unknown.
bool safe_read(const void* p, void* out, size_t n);
template <typename T>
bool safe_get(const void* p, T* out) {
  return safe_read(p, out, sizeof(T));
}

// safe_read for walking a module's headers and tables: the last region VirtualQuery found readable is remembered, so
// reads inside it only take the structured-exception-guarded copy (the scan stays far below a millisecond).
class Reader {
 public:
  bool read(const void* p, void* out, size_t n);
  template <typename T>
  bool get(const void* p, T* out) {
    return read(p, out, sizeof(T));
  }
  // A NUL-terminated string of at most max - 1 characters (false: unreadable or longer).
  bool str(const void* p, char* out, size_t max);
  unsigned queries = 0;  // VirtualQuery calls made (the test checks the cache)

 private:
  uintptr_t lo_ = 0, hi_ = 0;
};

// The bytes from p to the end of the committed regions of the allocation p lies in (VirtualQuery, at most 4096
// regions). 0: p is not in committed memory.
size_t mapped_extent(const void* p);

// A loaded module's PE64 headers as the scan may use them. open() first measures what is mapped at the module
// (mapped_extent), reads the DOS header only inside that, checks e_lfanew as an integer against it before the NT
// headers are read, and accepts them only when they fit inside both SizeOfImage and the mapping, the optional header
// is PE32+ and SizeOfImage is plausible. The image's extent is then the smaller of SizeOfImage and the mapping (a
// forged larger SizeOfImage does not widen it). dir() hands out a data directory only when its RVA and size lie inside
// the extent.
// at() forms an address from an RVA only after checking [rva, rva + n) against the extent (integers first, no pointer
// arithmetic on unchecked values); str() reads a name byte by byte, each byte checked the same way, so a name without
// its NUL before the end of the image is refused, never read past. Anything that fails is "not readable": the caller
// treats it as nothing found.
class PeView {
 public:
  bool open(const void* module, Reader& r);
  // the directory `index` when present and inside the image (false: absent or out of bounds)
  bool dir(unsigned index, uint32_t* rva, uint32_t* size) const;
  // base + rva when [rva, rva + n) lies inside the image, else null
  const uint8_t* at(uint64_t rva, uint64_t n) const;
  // the NUL-terminated string at rva, at most max - 1 characters, every byte inside the extent (false otherwise)
  bool str(uint64_t rva, char* out, size_t max, Reader& r) const;
  uint32_t size_of_image() const { return size_; }  // the extent: min(SizeOfImage, the mapping)
  const uint8_t* base() const { return base_; }

 private:
  const uint8_t* base_ = nullptr;
  uint32_t size_ = 0;
  uint32_t ndirs_ = 0;
  uint32_t dir_rva_[16] = {};
  uint32_t dir_size_[16] = {};
};

// The work budget of one scan: entries visited (modules, import descriptors, name-table entries) and elapsed time.
// step() counts one entry; false once either limit is passed (the time is looked at every 64 entries), and it stays
// false. A scan over its budget is inconclusive: it decides nothing (RR stays active) and logs line_over_budget.
class Budget {
 public:
  Budget(unsigned max_entries, unsigned max_ms);
  bool step();
  bool over() const { return over_; }
  unsigned entries() const { return entries_; }
  double elapsed_ms() const;

 private:
  unsigned max_entries_, entries_ = 0;
  double max_ms_;
  int64_t t0_ = 0, freq_ = 1;
  bool over_ = false;
};
// The load scan's budget: far above a game's import tables (a few thousand entries, well under a millisecond), far
// below a stall a player would notice.
constexpr unsigned kScanMaxEntries = 200000;
constexpr unsigned kScanMaxMs = 100;

// The by-name entries of one import (or delay-load) name table at names_rva with its address table at iat_rva:
// f(import name, slot address) for each, in order. It stops at the table's null entry, at an entry outside the image,
// after 4096 entries, or when the budget is spent; ordinal entries and names that are unreadable or not terminated
// inside the image are skipped.
void thunks(const PeView& pe, Reader& r, Budget& b, uint32_t names_rva, uint32_t iat_rva,
            const std::function<void(const char* name, void** slot)>& f);
// Every by-name import slot of the image at `module` from the DLL named `leaf` (case-insensitive): the import
// descriptors are read only within the import directory's declared size (and up to the null descriptor), each named
// DLL's name table through thunks().
void import_slots(const void* module, const char* leaf, Reader& r, Budget& b,
                  const std::function<void(const char* name, void** slot)>& f);

// A hook jump at `at` (code holds the first bytes read from there, n of them). On success *target is where it goes;
// for jmp [rip+disp32] *cell is the pointer cell (read with safe_read; a failed read makes the decode fail).
bool decode_jump(const uint8_t* code, size_t n, const uint8_t* at, const void** target, const void** cell);

enum class Owner {
  kUnknown,     // the memory could not be read, a null pointer, or more than kMaxHops jumps: inconclusive
  kSelf,        // this WrongGPU RR module
  kInterface,   // the game interface module (its own code)
  kHost,        // the host that loaded us, or the module holding its API table (it patches the interface itself)
  kOwnCopy,     // another loaded copy of WrongGPU RR (exports kOwnRrExport): positive, it switches this copy off
  kWrongGpu,    // another WrongGPU module (exports kWrongGpuAsiExport)
  kExeThunk,    // the exe's own code (a delay-load slot not bound yet holds the delay helper's thunk there): inconclusive
  kAnonymous,   // readable code outside every module that does not jump on (a trampoline): inconclusive
  kForeign,     // another named module: inconclusive (named in the log; only rr_other_plugin=yield acts on it)
};
const char* owner_word(Owner o);

// The process as the classification sees it. The lookups are injectable (the unit test uses synthetic modules).
struct Env {
  const void* self = nullptr;
  const void* iface = nullptr;
  const void* host = nullptr;      // the module that called InitializeASI
  const void* host_alt = nullptr;  // the module holding the host's API table, when it is another module (a separate loader)
  const void* exe = nullptr;
  // the interface's import address table (its own imports): a jmp [rip+disp32] through a cell in it is the linker's
  // import thunk, not a hook
  const uint8_t* iface_iat_begin = nullptr;
  const uint8_t* iface_iat_end = nullptr;
  std::function<const void*(const void* addr)> module_of;        // the module holding addr, or null
  std::function<bool(const void* module, const char* name)> exports;  // the module exports `name`
  std::function<std::string(const void* module)> name_of;        // the module's file name (for the log)
};
// A reference to the module holding the address m (GetModuleHandleExW from address, refcount raised; released at the
// end of the scope), so the module cannot unload while a module API (GetProcAddress, GetModuleFileNameW) runs on it.
// held() false: no module holds m any more (unloaded meanwhile); the caller treats it as nothing found.
class ModuleRef {
 public:
  explicit ModuleRef(const void* m);
  ~ModuleRef();
  ModuleRef(const ModuleRef&) = delete;
  ModuleRef& operator=(const ModuleRef&) = delete;
  bool held() const { return h_ != nullptr; }
  void* handle() const { return h_; }

 private:
  void* h_ = nullptr;
};

// The real process: GetModuleHandleExW (from address, refcount unchanged) for module_of; GetProcAddress and
// GetModuleFileNameW under a ModuleRef.
Env process_env(const void* self, const void* iface, const void* host, const void* host_alt = nullptr);

struct Where {
  Owner owner = Owner::kUnknown;
  std::string module;  // file name of the module the address ends in ("" when none)
  const void* module_handle = nullptr;  // the module the address ends in (null when none)
  int hops = 0;        // hook jumps followed
  bool too_many_hops = false;  // kUnknown because the jump chain was longer than kMaxHops
};
// Where code at `p` leads. follow_iface: follow a hook jump at p when p is inside the interface (the first bytes of
// an export); a jump that stays inside the interface, or goes through the interface's own import table, is the
// interface's own code. Jumps are always followed out of anonymous memory.
Where classify(const void* p, const Env& env, bool follow_iface);

// What a scan found. kOwnCopy: an address ends in a second copy of WrongGPU RR (it wins over everything). kForeign: an
// address ends in another named module (the first one is kept for the log; inconclusive, see decide()). Whatever could
// not be explained at all (unreadable, too many hops, anonymous code that does not jump on, the exe's own code) is
// counted in `unclear` and leaves the kind as it is.
struct Verdict {
  enum Kind { kClean, kOwnCopy, kForeign } kind = kClean;
  std::string detail;  // e.g. "the import slot slEvaluateFeature of Game.exe leads to other.dll"
  std::string module;  // kForeign: the file name of the module the first foreign address ends in ("another module")
  const void* other_module = nullptr;  // kForeign: that module
  unsigned foreign = 0;  // addresses that end in another named module
  unsigned reads = 0;  // addresses looked at
  unsigned unclear = 0;          // addresses that were neither trusted nor a named module
  std::string unclear_detail;    // the first of them, e.g. "the import slot X of Y could not be read"
  // Takes one classification. `what` names the place ("the import slot X of Y"); anon_ok: code outside every
  // module is accepted here (the host's own trampolines); exe_ok: the exe's own code is accepted (delay thunks).
  void add(const Where& w, const std::string& what, bool anon_ok, bool exe_ok);
};
bool trusted(Owner o, bool anon_ok, bool exe_ok);

// rr_other_plugin from wronggpu-rr.ini. Unknown or empty values are off (*note says so for a nonempty unknown one).
enum class Mode { kOff, kTake, kYield };
Mode parse_mode(const std::string& value, std::string* note);
const char* mode_word(Mode m);

// The guard's state for the whole process run. Active until a decision says otherwise; the late decision is taken once.
enum class State : int { kActive = 0, kOff = 1, kOwnCopy = 2, kPassThrough = 3, kTaken = 4 };

// When a scan runs: before anything of ours is in place (kLoad: the exe's import slots and the exports at InitializeASI,
// the exports before the host routes), right after our host routes were installed in front (kRoutes: the host's next
// functions), or after our handlers were installed (kLate: OptiPluginInit, the game's first call).
enum class Stage { kLoad, kRoutes, kLate };

// What a scan's verdict means for WrongGPU RR (the whole policy, in one place):
//   - not active already (off, own copy, passing): stays as it is;
//   - clean: stays as it is;
//   - a copy of WrongGPU RR: off at load (nothing patched), passing once our handlers or routes are in place;
//   - a named foreign module: off -> stays as it is (active; the caller logs line_foreign); take -> kTaken while
//     WrongGPU can still go in front of it (load, routes), else as it is; yield -> off at load, passing later.
State decide(Verdict::Kind kind, Mode mode, Stage stage, State current);

// The log lines (stable; HOOK-MODE.md documents them; the beginnings are what a reader such as the launcher's Last run
// page can match: kForeignStem, the refusal stem of line_ngx_refused, the own-copy stem).
// A named foreign module on the RR path: "another module hooks the game's RR calls: <module> (<detail>); <tail>"
enum class Tail {
  kStaysActive,   // rr_other_plugin=off (also take after our handlers are in place): RR stays WrongGPU's
  kTakeInFront,   // take, at load or with our routes in front: WrongGPU RR in front, the module gets the other calls
  kYieldOff,      // yield at load: nothing patched
  kYieldPassing,  // yield with our handlers or routes in place: they pass every call on unchanged
};
std::string line_foreign(const std::string& module, const std::string& detail, Tail tail);
std::string line_own_copy(const std::string& active, const std::string& self);
std::string line_ngx_refused();                                 // the host refused our NGX provider
// The stay-on diagnostic (starts with "hook: "; never with a beginning above): the scan could not explain `n`
// address(es), `first` is the first one. RR stays active.
std::string line_inconclusive(const std::string& when, unsigned n, const std::string& first);
// The load scan ran out of its budget (entries visited, elapsed time): inconclusive, WrongGPU RR stays active
std::string line_over_budget(unsigned entries, double ms);

class Decision {
 public:
  State state() const { return static_cast<State>(state_.load(std::memory_order_acquire)); }
  // The calls go to WrongGPU RR's core (not off, not passing through).
  bool active() const {
    const State s = state();
    return s == State::kActive || s == State::kTaken;
  }
  void set(State s) { state_.store(static_cast<int>(s), std::memory_order_release); }
  bool frozen() const { return frozen_.load(std::memory_order_acquire); }
  // Runs `decide` exactly once (the first caller; concurrent callers wait until it returned), then freezes. Later
  // calls return at once. `decide` returns the state to keep.
  void freeze_once(const std::function<State()>& decide);
  // A non-freezing check (OptiPluginInit): runs `decide` under the same lock unless already frozen.
  void check(const std::function<State()>& decide);

 private:
  std::atomic<int> state_{0};
  std::atomic<bool> frozen_{false};
  std::once_flag once_;
  std::mutex mu_;
};

}  // namespace guard
}  // namespace wgrr
