// SPDX-License-Identifier: MIT
// RR guard: the coexistence guard of the hook-mode RR plugin (HOOK-MODE.md, "Another Ray Reconstruction
// plugin"). When another Ray Reconstruction plugin demonstrably holds the game's RR path in this process, WrongGPU RR
// stays off and logs one line; Model M is not touched. Generic and read-only: it looks at where the game's RR calls
// lead (import slots, the first bytes of the interface's RR exports, the host's continuation pointers), never at
// another plugin's files or internals.
//
// Only a POSITIVE finding stops the plugin: an address that ends in a named foreign module (also through decoded hook
// jumps), or in our own second copy. Everything else the scan cannot explain (unreadable memory, too many hops, code
// outside every module that does not jump on, the exe's own code) is logged once as inconclusive and the plugin stays
// active: a game's copy protection or a hooking library's trampolines must never switch RR off for a player who has no
// other plugin.
//
// This header holds the parts without process state, so the CPU tests can drive them with synthetic memory:
//   - safe_read: every read of memory we do not own (VirtualQuery for committed, readable, non-guard pages, then a
//     structured-exception-guarded copy); a failed read is "unknown", never a fault;
//   - decode_jump: the jump forms a hook puts at the start of a function (jmp rel32 / rel8, jmp [rip+disp32],
//     mov rax|r11, imm64 + jmp reg);
//   - classify: who an address belongs to (our module, the game interface, the host, another copy of WrongGPU RR,
//     another WrongGPU module, the exe's own delay-load thunk, code outside every module, another module), following
//     hook jumps out of the interface and out of anonymous memory, at most kMaxHops of them;
//   - Verdict: the positive finding (if any) of several classifications, the inconclusive ones, and the words for the
//     log;
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
// wronggpu-rr.ini: rr_other_plugin = off (default) | take
constexpr const char* kOverrideKey = "rr_other_plugin";
constexpr int kMaxHops = 4;  // hook jumps followed from one address (a fifth one is "too many hops": inconclusive)

// True when [p, p + n) lies in committed pages that are readable (execute-only pages count: code is read from them and
// the structured-exception guard of the copy covers a fault) and not guard pages (VirtualQuery only).
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

// A hook jump at `at` (code holds the first bytes read from there, n of them). On success *target is where it goes;
// for jmp [rip+disp32] *cell is the pointer cell (read with safe_read; a failed read makes the decode fail).
bool decode_jump(const uint8_t* code, size_t n, const uint8_t* at, const void** target, const void** cell);

enum class Owner {
  kUnknown,     // the memory could not be read, a null pointer, or more than kMaxHops jumps: inconclusive
  kSelf,        // this WrongGPU RR module
  kInterface,   // the game interface module (its own code)
  kHost,        // the host that loaded us, or the module holding its API table (it patches the interface itself)
  kOwnCopy,     // another loaded copy of WrongGPU RR (exports kOwnRrExport): positive
  kWrongGpu,    // another WrongGPU module (exports kWrongGpuAsiExport)
  kExeThunk,    // the exe's own code (a delay-load slot not bound yet holds the delay helper's thunk there): inconclusive
  kAnonymous,   // readable code outside every module that does not jump on (a trampoline): inconclusive
  kForeign,     // another named module: another plugin (positive)
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
// The real process: GetModuleHandleExW (from address, refcount unchanged), GetProcAddress, GetModuleFileNameW.
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

// What a scan found. kOther / kOwnCopy are positive findings only (an address that ends in another named module, or in
// a second copy of WrongGPU RR; the first foreign one decides, a copy of WrongGPU RR before anything else). Whatever
// could not be explained (unreadable, too many hops, anonymous code that does not jump on, the exe's own code) is
// counted in `unclear` and leaves the kind at kClean: the plugin stays active.
struct Verdict {
  enum Kind { kClean, kOwnCopy, kOther } kind = kClean;
  std::string detail;  // e.g. "the import slot slEvaluateFeature of Game.exe leads to other.asi"
  const void* other_module = nullptr;  // kOther: the module the first foreign finding ends in
  unsigned reads = 0;  // addresses looked at
  unsigned unclear = 0;          // addresses that were neither trusted nor a positive finding
  std::string unclear_detail;    // the first of them, e.g. "the import slot X of Y could not be read"
  // Takes one classification. `what` names the place ("the import slot X of Y"); anon_ok: code outside every
  // module is accepted here (the host's own trampolines); exe_ok: the exe's own code is accepted (delay thunks).
  void add(const Where& w, const std::string& what, bool anon_ok, bool exe_ok);
};
bool trusted(Owner o, bool anon_ok, bool exe_ok);

// rr_other_plugin from wronggpu-rr.ini: false = off (the default, also for an unknown value; *note says so).
bool override_take(const std::string& value, std::string* note);

// The log lines (stable; HOOK-MODE.md documents them, the launcher's Last run page matches them).
// stays off at load (passing: WrongGPU's handlers were already installed by the host and pass every call on)
std::string line_other(const std::string& detail, bool passing = false);
std::string line_other_later(const std::string& detail);        // stays off, found at the late check (pass-through)
std::string line_take(const std::string& detail);               // rr_other_plugin=take: WrongGPU RR in front
std::string line_cannot_take(const std::string& detail);        // take asked, the other plugin loaded later
std::string line_own_copy(const std::string& active, const std::string& self);
std::string line_ngx_other();                                   // the host refused our NGX provider
// The stay-on diagnostic (starts with "hook: "; never with the beginnings above): the scan could not explain `n`
// address(es), `first` is the first one. No finding, RR stays active.
std::string line_inconclusive(const std::string& when, unsigned n, const std::string& first);

// The guard's state for the whole process run. Active until a scan says otherwise; the late decision is taken once.
enum class State : int { kActive = 0, kOff = 1, kOwnCopy = 2, kPassThrough = 3, kTaken = 4 };

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
