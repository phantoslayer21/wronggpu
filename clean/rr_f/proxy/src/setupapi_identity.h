// SPDX-License-Identifier: MIT
// SetupAPI identity shim (clean/rr-sm2-shim, Brief 10; written for this project from the spec team's functional
// description and the public Win32 SetupAPI documentation).
//
// Marvel's Spider-Man 2 reads the display adapter's PCI identity through SetupAPI as well as through DXGI. When the host's
// GPU-name spoof is on, DXGI reports another vendor / device pair than the board's; the game then sees two different
// identities. While (and only while) that is so, in Spider-Man2.exe only, the plugin makes SetupAPI report the same
// pair as DXGI for the real active adapter (the decision is made at the game's first call of either function, when
// DXGI can be read; until DXGI exists, the spoofed pair from the host's own settings is used when its spoof is on):
//   - SetupDiGetDeviceInstanceIdW: a Display-class device whose instance ID starts with
//     PCI\VEN_<real>&DEV_<real>&SUBSYS_<8 hex>&REV_<2 hex> gets VEN / DEV replaced by the spoofed pair (SUBSYS, REV and
//     the instance suffix stay);
//   - SetupDiGetDeviceRegistryPropertyW, hardware IDs and compatible IDs only (multi-strings), for such a device: the
//     same replacement in every entry that starts with PCI\VEN_<real>&DEV_<real> (followed by '&' or the end).
// The replacement has the same length, so buffer sizes, returned sizes, the return value and the last error stay exactly
// what the real call gave. Nothing else is touched: other properties (the driver key in particular), other device
// classes, a failed call (buffer too small), other processes. Delivered by redirecting the game exe's import slots (the
// same in-process redirect the plugin uses for the game interface); no code is patched.
//
// This file: the rules and the two call wrappers, free of process state, so the unit tests can drive them with
// synthetic IDs (tests/test_setupapi_identity.cpp). The process side (which exe, the DXGI and kernel driver pairs, the
// import slots) is in hook_dll.cpp.
#pragma once
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <setupapi.h>
#include <cfgmgr32.h>  // MAX_DEVICE_ID_LEN

#include <stddef.h>
#include <stdint.h>

#include <atomic>
#include <functional>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

namespace wgsi {

struct PciPair {
  uint16_t ven = 0, dev = 0;
  bool operator==(const PciPair& o) const { return ven == o.ven && dev == o.dev; }
  bool operator!=(const PciPair& o) const { return !(*this == o); }
};
std::string pair_text(PciPair p);  // "1002:7550"

// The process and setting gate (at start), and the inputs of the decision (at the first calls).
struct GateInputs {
  std::string setting;      // setupapi_identity from wronggpu-rr.ini ("" = absent)
  std::string exe;          // the process's exe file name
  bool dxgi_ok = false;     // the adapter the game sees through DXGI could be read
  PciPair dxgi;             // its vendor / device as DXGI reports it to the game
  bool kmt_ok = false;      // the kernel driver's identity of that adapter (without DXGI: of the first hardware
  PciPair kmt;              // adapter the kernel lists) could be read; the real pair
};
struct Decision {
  bool active = false;
  PciPair real, spoof;
  std::string why;          // for the start line
};
// The exe is checked first; callers may skip reading DXGI / the driver when target_exe() is false or the setting is off.
bool target_exe(const std::string& exe);  // Spider-Man2.exe (any case)
bool setting_off(const std::string& setting, std::string* note);  // "off" -> true; "", "auto" -> false; other: note
Decision decide(const GateInputs& in);

// The spoofed pair the host's own settings give (its ini next to the game exe, section [Spoofing]): Dxgi=true turns its
// GPU-name spoof on; SpoofedDeviceId (0x-prefixed hex or decimal) is the device; the vendor is taken as 10de (the board
// the host's GPU-name spoof reports; decision (b) after test SH1). Keys and the section name in any case;
// ';' or '#' starts a comment.
struct HostSpoof {
  bool dxgi_on = false;     // [Spoofing] Dxgi=true
  bool pair_ok = false;     // a numeric SpoofedDeviceId was found (vendor 10de)
  PciPair pair;
  std::string note;         // why the pair is not known (for the log)
};
HostSpoof parse_host_ini(const std::string& text);

// One try of the lazy decision.
struct ResolveInputs : GateInputs {
  HostSpoof host;
};
struct Resolution {
  bool final = false;       // false: try again at a later call (DXGI not readable yet and nothing to fall back on)
  Decision d;               // when not final: d.why is the reason for the inactive line after the last try
};
// DXGI is preferred when readable (final, as decide()). Without DXGI: the host's settings when Dxgi=true and they give
// the pair, with the kernel driver's pair (final). Otherwise not final.
Resolution resolve(const ResolveInputs& in);

// The rules (exposed for the tests).
bool is_display_class(const GUID& g);
// `id` (n characters) starts with PCI\VEN_<real.ven>&DEV_<real.dev>&SUBSYS_<8 hex>&REV_<2 hex> (case-insensitive)
bool instance_id_matches(const wchar_t* id, size_t n, PciPair real);
// `s` (n characters) starts with PCI\VEN_<real.ven>&DEV_<real.dev> followed by '&' or its end
bool entry_matches(const wchar_t* s, size_t n, PciPair real);
// overwrites the VEN and DEV digits of a matching ID in place (upper-case hex; same length)
void write_pair(wchar_t* s, PciPair to);
// Rewrites, in a multi-string of `nchars` characters, every entry that entry_matches. Stops at the empty entry or at
// `nchars` (never reads or writes past it). Returns the number of entries rewritten; `changes` gets (old, new).
unsigned rewrite_multi_sz(wchar_t* buf, size_t nchars, PciPair real, PciPair spoof,
                          std::vector<std::pair<std::wstring, std::wstring>>* changes);

using InstanceIdFn = BOOL(WINAPI*)(HDEVINFO, PSP_DEVINFO_DATA, PWSTR, DWORD, PDWORD);
using RegistryPropertyFn = BOOL(WINAPI*)(HDEVINFO, PSP_DEVINFO_DATA, DWORD, PDWORD, PBYTE, DWORD, PDWORD);

class Shim {
 public:
  using LogFn = std::function<void(const std::string&)>;
  using Resolver = std::function<Resolution()>;
  static constexpr unsigned kLogLimit = 500;     // rewritten calls logged one line each, then one line saying so
  static constexpr unsigned kDecisionTries = 8;  // calls that may try the decision before it is taken as inactive

  // `instance_id` / `registry_property`: the functions the calls continue with (what the import slots held, or the
  // exports of setupapi.dll); `instance_id` is also used to read a device's instance ID for the property rule.
  // configure: decided already (no decision line). configure_lazy: the first calls run `resolver` (one at a time) until
  // it is final or `tries` calls have tried; then the decision is cached and logged once ("setupapi identity: <why>").
  // Calls before the decision pass through unchanged.
  void configure(const Decision& d, InstanceIdFn instance_id, RegistryPropertyFn registry_property, LogFn log);
  void configure_lazy(Resolver resolver, unsigned tries, InstanceIdFn instance_id, RegistryPropertyFn registry_property,
                      LogFn log);
  bool decided() const { return decided_.load(std::memory_order_acquire); }
  bool active() const { return decided() && d_.active; }
  const Decision& decision() const { return d_; }  // valid once decided()
  unsigned tries() const { return tries_; }

  BOOL instance_id(HDEVINFO set, PSP_DEVINFO_DATA dev, PWSTR buf, DWORD size, PDWORD required);
  BOOL registry_property(HDEVINFO set, PSP_DEVINFO_DATA dev, DWORD property, PDWORD type, PBYTE buf, DWORD size,
                         PDWORD required);
  unsigned rewritten_calls() const { return rewritten_.load(); }

 private:
  bool ready();  // decides when not decided yet (lazy); true when active
  bool device_matches(HDEVINFO set, PSP_DEVINFO_DATA dev);
  void note(const std::string& line);

  Decision d_;
  std::atomic<bool> decided_{false};
  std::mutex decide_mutex_;
  Resolver resolver_;
  unsigned max_tries_ = 0, tries_ = 0;
  InstanceIdFn instance_id_ = nullptr;
  RegistryPropertyFn registry_property_ = nullptr;
  LogFn log_;
  std::atomic<unsigned> rewritten_{0};
};

// ---- the process side of the redirect (the import slots)
// One import slot of `module` for a function of `dll`: the import table's (delay = false) or the delay-load table's.
struct ImportSlot {
  void** slot = nullptr;
  std::string name;
  bool delay = false;
  void* original = nullptr;  // what the slot held when read
};
// The import and delay-load slots of `module` that import one of `names` by name from `dll` (case-insensitive).
std::vector<ImportSlot> find_import_slots(HMODULE module, const char* dll, const std::vector<std::string>& names);

// The shim of this process and the handlers the import slots get.
Shim& process_shim();
BOOL WINAPI hooked_instance_id(HDEVINFO set, PSP_DEVINFO_DATA dev, PWSTR buf, DWORD size, PDWORD required);
BOOL WINAPI hooked_registry_property(HDEVINFO set, PSP_DEVINFO_DATA dev, DWORD property, PDWORD type, PBYTE buf,
                                     DWORD size, PDWORD required);
// Configures process_shim() lazily with `resolver` (Shim::configure_lazy) and redirects the import / delay-load slots of
// `module` for SetupDiGetDeviceInstanceIdW and SetupDiGetDeviceRegistryPropertyW to the handlers. The calls continue with
// what a bound slot held (another layer's hook stays in the chain), else with setupapi.dll's export (an unbound
// delay-load slot holds the delay helper's thunk, which would rebind the slot). Returns the slots redirected; *detail:
// which ones, for the start line.
unsigned install(HMODULE module, Shim::Resolver resolver, unsigned tries, Shim::LogFn log, std::string* detail);

}  // namespace wgsi
