// SPDX-License-Identifier: MIT
// SetupAPI identity shim: the rules and the call wrappers (setupapi_identity.h). Written for this project.
#include "setupapi_identity.h"

#include <stdio.h>
#include <string.h>
#include <wchar.h>

namespace wgsi {
namespace {

// {4d36e968-e325-11ce-bfc1-08002be10318}: the Display device setup class (public Win32 documentation, devguid.h)
const GUID kDisplayClass = {0x4d36e968, 0xe325, 0x11ce, {0xbf, 0xc1, 0x08, 0x00, 0x2b, 0xe1, 0x03, 0x18}};

const size_t kVenAt = 8;    // "PCI\VEN_" then 4 hex digits
const size_t kDevAt = 17;   // "&DEV_" then 4 hex digits
const size_t kPairEnd = 21;

wchar_t upper(wchar_t c) { return (c >= L'a' && c <= L'z') ? (wchar_t)(c - L'a' + L'A') : c; }
bool is_hex(wchar_t c) { return (c >= L'0' && c <= L'9') || (c >= L'a' && c <= L'f') || (c >= L'A' && c <= L'F'); }
int hex_value(wchar_t c) {
  return c <= L'9' ? c - L'0' : (upper(c) - L'A' + 10);
}

// `s` at `at` holds the literal `lit` (case-insensitive)
bool literal(const wchar_t* s, size_t n, size_t at, const wchar_t* lit) {
  for (size_t i = 0; lit[i]; ++i)
    if (at + i >= n || upper(s[at + i]) != upper(lit[i])) return false;
  return true;
}
// `s` at `at` holds `count` hex digits; *v gets their value (when v)
bool hex_digits(const wchar_t* s, size_t n, size_t at, size_t count, unsigned* v) {
  unsigned x = 0;
  for (size_t i = 0; i < count; ++i) {
    if (at + i >= n || !is_hex(s[at + i])) return false;
    x = x * 16u + (unsigned)hex_value(s[at + i]);
  }
  if (v) *v = x;
  return true;
}

bool pair_prefix(const wchar_t* s, size_t n, PciPair real) {
  unsigned ven = 0, dev = 0;
  return literal(s, n, 0, L"PCI\\VEN_") && hex_digits(s, n, kVenAt, 4, &ven) && literal(s, n, 12, L"&DEV_") &&
         hex_digits(s, n, kDevAt, 4, &dev) && ven == real.ven && dev == real.dev;
}

std::string ascii(const wchar_t* s, size_t n) {
  std::string o;
  o.reserve(n);
  for (size_t i = 0; i < n; ++i) o.push_back(s[i] >= 0x20 && s[i] < 0x7f ? (char)s[i] : '?');
  return o;
}
std::string ascii(const std::wstring& w) { return ascii(w.c_str(), w.size()); }

std::string lower(std::string s) {
  for (char& c : s) c = (c >= 'A' && c <= 'Z') ? (char)(c - 'A' + 'a') : c;
  return s;
}
std::string trim(const std::string& s) {
  size_t a = 0, b = s.size();
  while (a < b && (s[a] == ' ' || s[a] == '\t')) ++a;
  while (b > a && (s[b - 1] == ' ' || s[b - 1] == '\t' || s[b - 1] == '\r')) --b;
  return s.substr(a, b - a);
}

std::string id_prefix(PciPair p) {
  char b[40];
  snprintf(b, sizeof b, "PCI\\VEN_%04X&DEV_%04X", p.ven, p.dev);
  return b;
}

std::string scope(PciPair real, PciPair spoof) {
  return "for Display-class devices of the real adapter, SetupAPI's instance ID and hardware / compatible IDs " +
         id_prefix(real) + " are reported as " + id_prefix(spoof);
}

// "0x2206" (hex) or "8710" (decimal), at most 0xffff
bool parse_id(const std::string& v, uint16_t* out) {
  if (v.empty()) return false;
  unsigned long x = 0;
  size_t i = 0;
  int base = 10;
  if (v.size() > 2 && v[0] == '0' && (v[1] == 'x' || v[1] == 'X')) { base = 16; i = 2; }
  for (; i < v.size(); ++i) {
    const char c = v[i];
    int d = (c >= '0' && c <= '9') ? c - '0' : (base == 16 && c >= 'a' && c <= 'f') ? c - 'a' + 10
                                           : (base == 16 && c >= 'A' && c <= 'F') ? c - 'A' + 10 : -1;
    if (d < 0) return false;
    x = x * (unsigned long)base + (unsigned long)d;
    if (x > 0xffffUL) return false;
  }
  *out = (uint16_t)x;
  return true;
}

}  // namespace

std::string pair_text(PciPair p) {
  char b[16];
  snprintf(b, sizeof b, "%04x:%04x", p.ven, p.dev);
  return b;
}

bool target_exe(const std::string& exe) { return lower(exe) == "spider-man2.exe"; }

bool setting_off(const std::string& setting, std::string* note) {
  const std::string v = lower(trim(setting));
  if (v == "off") return true;
  if (!v.empty() && v != "auto" && note) *note = "setupapi_identity=" + trim(setting) + " not recognised, using auto; ";
  return false;
}

Decision decide(const GateInputs& in) {
  Decision d;
  std::string note;
  const bool off = setting_off(in.setting, &note);
  if (off) {
    d.why = "inactive: setupapi_identity=off in wronggpu-rr.ini";
    return d;
  }
  if (!target_exe(in.exe)) {
    d.why = note + "inactive: the process is " + in.exe + "; the shim serves Spider-Man2.exe only";
    return d;
  }
  if (!in.dxgi_ok) {
    d.why = note + "inactive: the adapter the game sees through DXGI could not be read";
    return d;
  }
  if (!in.kmt_ok) {
    d.why = note + "inactive: the kernel driver's identity of the adapter could not be read (DXGI reports " +
            pair_text(in.dxgi) + ")";
    return d;
  }
  if (in.dxgi == in.kmt) {
    d.why = note + "inactive: DXGI reports the real pair " + pair_text(in.kmt) + " to the game (the host's GPU-name "
            "spoof is off)";
    return d;
  }
  d.active = true;
  d.real = in.kmt;
  d.spoof = in.dxgi;
  d.why = note + "active: DXGI reports " + pair_text(in.dxgi) + " to the game, the kernel driver " + pair_text(in.kmt) +
          "; " + scope(in.kmt, in.dxgi);
  return d;
}

HostSpoof parse_host_ini(const std::string& text) {
  HostSpoof h;
  bool in_section = false, have_dev = false, dev_seen = false;
  std::string dev_value;
  size_t at = 0;
  while (at <= text.size()) {
    size_t end = text.find('\n', at);
    if (end == std::string::npos) end = text.size();
    std::string line = text.substr(at, end - at);
    at = end + 1;
    const size_t c = line.find_first_of(";#");
    if (c != std::string::npos) line = line.substr(0, c);
    line = trim(line);
    if (line.empty()) continue;
    if (line[0] == '[') {
      in_section = lower(line) == "[spoofing]";
      continue;
    }
    if (!in_section) continue;
    const size_t eq = line.find('=');
    if (eq == std::string::npos) continue;
    const std::string key = lower(trim(line.substr(0, eq))), value = trim(line.substr(eq + 1));
    if (key == "dxgi") h.dxgi_on = lower(value) == "true";
    if (key == "spoofeddeviceid") { dev_seen = true; dev_value = value; have_dev = parse_id(value, &h.pair.dev); }
  }
  h.pair_ok = have_dev;
  if (have_dev) h.pair.ven = 0x10de;  // the vendor the host's GPU-name spoof reports
  else {
    h.note = dev_seen ? "SpoofedDeviceId=" + dev_value + " is not a device ID" : "no SpoofedDeviceId";
    h.pair = PciPair();
  }
  return h;
}

Resolution resolve(const ResolveInputs& in) {
  Resolution r;
  std::string note;
  if (setting_off(in.setting, &note) || !target_exe(in.exe) || in.dxgi_ok) {  // DXGI preferred when readable
    r.final = true;
    r.d = decide(in);
    return r;
  }
  if (in.host.dxgi_on && in.host.pair_ok) {
    if (!in.kmt_ok) {
      r.d.why = note + "inactive: DXGI is not readable and the kernel driver's identity of the adapter could not be read "
                       "(the host's settings report " + pair_text(in.host.pair) + ")";
      return r;
    }
    r.final = true;
    if (in.host.pair == in.kmt) {
      r.d.why = note + "inactive: DXGI is not readable yet and the host's settings report the real pair " +
                pair_text(in.kmt);
      return r;
    }
    r.d.active = true;
    r.d.real = in.kmt;
    r.d.spoof = in.host.pair;
    r.d.why = note + "active: the host's settings report " + pair_text(in.host.pair) +
              " ([Spoofing] Dxgi=true; DXGI is not readable yet), the kernel driver " + pair_text(in.kmt) + "; " +
              scope(in.kmt, in.host.pair);
    return r;
  }
  r.d.why = note + "inactive: the adapter the game sees through DXGI could not be read" +
            (in.host.dxgi_on ? "; the host's settings give no spoofed pair (" + in.host.note + ")"
                             : std::string("; the host's GPU-name spoof is not on in its settings ([Spoofing] Dxgi)"));
  return r;
}

bool is_display_class(const GUID& g) { return memcmp(&g, &kDisplayClass, sizeof g) == 0; }

bool instance_id_matches(const wchar_t* id, size_t n, PciPair real) {
  return id && pair_prefix(id, n, real) && literal(id, n, kPairEnd, L"&SUBSYS_") && hex_digits(id, n, 29, 8, nullptr) &&
         literal(id, n, 37, L"&REV_") && hex_digits(id, n, 42, 2, nullptr);
}

bool entry_matches(const wchar_t* s, size_t n, PciPair real) {
  return s && pair_prefix(s, n, real) && (n == kPairEnd || s[kPairEnd] == L'&' || s[kPairEnd] == 0);
}

void write_pair(wchar_t* s, PciPair to) {
  static const wchar_t hex[] = L"0123456789ABCDEF";
  for (int i = 0; i < 4; ++i) {
    s[kVenAt + i] = hex[(to.ven >> (12 - 4 * i)) & 0xf];
    s[kDevAt + i] = hex[(to.dev >> (12 - 4 * i)) & 0xf];
  }
}

unsigned rewrite_multi_sz(wchar_t* buf, size_t nchars, PciPair real, PciPair spoof,
                          std::vector<std::pair<std::wstring, std::wstring>>* changes) {
  if (!buf) return 0;
  unsigned count = 0;
  size_t at = 0;
  while (at < nchars && buf[at]) {
    size_t len = 0;
    while (at + len < nchars && buf[at + len]) ++len;  // the entry, bounded by the buffer
    if (entry_matches(buf + at, len, real)) {
      std::wstring old(buf + at, len);
      write_pair(buf + at, spoof);
      if (changes) changes->push_back({old, std::wstring(buf + at, len)});
      ++count;
    }
    at += len + 1;
  }
  return count;
}

void Shim::configure(const Decision& d, InstanceIdFn instance_id, RegistryPropertyFn registry_property, LogFn log) {
  d_ = d;
  instance_id_ = instance_id;
  registry_property_ = registry_property;
  log_ = std::move(log);
  resolver_ = nullptr;
  decided_.store(true, std::memory_order_release);
}

void Shim::configure_lazy(Resolver resolver, unsigned tries, InstanceIdFn instance_id,
                          RegistryPropertyFn registry_property, LogFn log) {
  d_ = Decision();
  instance_id_ = instance_id;
  registry_property_ = registry_property;
  log_ = std::move(log);
  resolver_ = std::move(resolver);
  max_tries_ = tries ? tries : 1;
  tries_ = 0;
  decided_.store(false, std::memory_order_release);
}

bool Shim::ready() {
  if (decided_.load(std::memory_order_acquire)) return d_.active;
  static thread_local bool deciding = false;  // a SetupAPI call made while deciding (through these slots): pass through
  if (deciding) return false;
  std::lock_guard<std::mutex> lock(decide_mutex_);
  if (decided_.load(std::memory_order_acquire)) return d_.active;
  Resolution r;
  if (resolver_) {
    deciding = true;
    r = resolver_();
    deciding = false;
  } else {
    r.final = true;
  }
  ++tries_;
  if (!r.final && tries_ < max_tries_) return false;  // not yet: this call passes through
  d_ = r.d;
  if (!r.final) {
    d_.active = false;
    d_.why += " (after " + std::to_string(tries_) + (tries_ == 1 ? " try)" : " tries)");
  }
  decided_.store(true, std::memory_order_release);
  if (log_) log_("setupapi identity: " + d_.why);
  return d_.active;
}

void Shim::note(const std::string& line) {
  const unsigned n = ++rewritten_;
  if (!log_ || n > kLogLimit + 1) return;
  if (n == kLogLimit + 1)
    log_("setupapi identity: more rewritten calls; not logged any more (" + std::to_string(kLogLimit) + " logged)");
  else
    log_(line);
}

bool Shim::device_matches(HDEVINFO set, PSP_DEVINFO_DATA dev) {
  if (!dev || dev->cbSize < sizeof(SP_DEVINFO_DATA) || !is_display_class(dev->ClassGuid) || !instance_id_) return false;
  wchar_t id[MAX_DEVICE_ID_LEN + 1] = {};
  if (!instance_id_(set, dev, id, MAX_DEVICE_ID_LEN, nullptr)) return false;
  return instance_id_matches(id, wcsnlen(id, MAX_DEVICE_ID_LEN), d_.real);
}

BOOL Shim::instance_id(HDEVINFO set, PSP_DEVINFO_DATA dev, PWSTR buf, DWORD size, PDWORD required) {
  const bool act = ready();  // before the real call: its last error is what the caller gets
  const BOOL ok = instance_id_(set, dev, buf, size, required);
  const DWORD err = GetLastError();
  if (ok && act && buf && size && dev && dev->cbSize >= sizeof(SP_DEVINFO_DATA) &&
      is_display_class(dev->ClassGuid)) {
    const size_t n = wcsnlen(buf, size);
    if (instance_id_matches(buf, n, d_.real)) {
      const std::wstring old(buf, n);
      write_pair(buf, d_.spoof);
      note("setupapi identity: SetupDiGetDeviceInstanceIdW " + ascii(old) + " -> " + ascii(buf, n));
    }
  }
  SetLastError(err);
  return ok;
}

BOOL Shim::registry_property(HDEVINFO set, PSP_DEVINFO_DATA dev, DWORD property, PDWORD type, PBYTE buf, DWORD size,
                             PDWORD required) {
  const bool act = ready();
  const BOOL ok = registry_property_(set, dev, property, type, buf, size, required);
  const DWORD err = GetLastError();
  if (ok && act && buf && size >= sizeof(wchar_t) &&
      (property == SPDRP_HARDWAREID || property == SPDRP_COMPATIBLEIDS) && (!type || *type == REG_MULTI_SZ) &&
      device_matches(set, dev)) {
    DWORD bytes = size;  // what the call filled: the data's size when it said so
    if (required && *required < bytes) bytes = *required;
    std::vector<std::pair<std::wstring, std::wstring>> changes;
    if (rewrite_multi_sz(reinterpret_cast<wchar_t*>(buf), bytes / sizeof(wchar_t), d_.real, d_.spoof, &changes)) {
      std::string line = std::string("setupapi identity: SetupDiGetDeviceRegistryPropertyW ") +
                         (property == SPDRP_HARDWAREID ? "hardware IDs" : "compatible IDs") + ": ";
      for (size_t i = 0; i < changes.size(); ++i)
        line += (i ? "; " : "") + ascii(changes[i].first) + " -> " + ascii(changes[i].second);
      note(line);
    }
  }
  SetLastError(err);
  return ok;
}

// ---------------------------------------------------------------- import slots
std::vector<ImportSlot> find_import_slots(HMODULE module, const char* dll, const std::vector<std::string>& names) {
  std::vector<ImportSlot> out;
  uint8_t* base = reinterpret_cast<uint8_t*>(module);
  if (!base) return out;
  const IMAGE_DOS_HEADER* dos = reinterpret_cast<const IMAGE_DOS_HEADER*>(base);
  if (dos->e_magic != IMAGE_DOS_SIGNATURE) return out;
  const IMAGE_NT_HEADERS64* nt = reinterpret_cast<const IMAGE_NT_HEADERS64*>(base + dos->e_lfanew);
  if (nt->Signature != IMAGE_NT_SIGNATURE || nt->OptionalHeader.Magic != IMAGE_NT_OPTIONAL_HDR64_MAGIC) return out;
  auto wanted = [&names](const char* n) {
    for (auto& w : names)
      if (strcmp(w.c_str(), n) == 0) return true;
    return false;
  };
  auto walk = [&](const IMAGE_THUNK_DATA64* by_name, IMAGE_THUNK_DATA64* iat, bool delay) {
    for (; by_name->u1.AddressOfData; ++by_name, ++iat) {
      if (by_name->u1.Ordinal & IMAGE_ORDINAL_FLAG64) continue;  // by name only
      const char* n = reinterpret_cast<const char*>(
          reinterpret_cast<const IMAGE_IMPORT_BY_NAME*>(base + by_name->u1.AddressOfData)->Name);
      if (!wanted(n)) continue;
      ImportSlot s;
      s.slot = reinterpret_cast<void**>(&iat->u1.Function);
      s.name = n;
      s.delay = delay;
      s.original = *s.slot;
      out.push_back(s);
    }
  };
  const IMAGE_DATA_DIRECTORY& imp = nt->OptionalHeader.DataDirectory[IMAGE_DIRECTORY_ENTRY_IMPORT];
  if (imp.VirtualAddress && imp.Size)
    for (const IMAGE_IMPORT_DESCRIPTOR* e = reinterpret_cast<const IMAGE_IMPORT_DESCRIPTOR*>(base + imp.VirtualAddress);
         e->Name; ++e) {
      if (_stricmp(reinterpret_cast<const char*>(base + e->Name), dll) != 0) continue;
      walk(reinterpret_cast<const IMAGE_THUNK_DATA64*>(base + (e->OriginalFirstThunk ? e->OriginalFirstThunk : e->FirstThunk)),
           reinterpret_cast<IMAGE_THUNK_DATA64*>(base + e->FirstThunk), false);
    }
  const IMAGE_DATA_DIRECTORY& dl = nt->OptionalHeader.DataDirectory[IMAGE_DIRECTORY_ENTRY_DELAY_IMPORT];
  if (dl.VirtualAddress && dl.Size)
    for (const IMAGE_DELAYLOAD_DESCRIPTOR* e = reinterpret_cast<const IMAGE_DELAYLOAD_DESCRIPTOR*>(base + dl.VirtualAddress);
         e->DllNameRVA; ++e) {
      if (!e->Attributes.RvaBased || _stricmp(reinterpret_cast<const char*>(base + e->DllNameRVA), dll) != 0) continue;
      walk(reinterpret_cast<const IMAGE_THUNK_DATA64*>(base + e->ImportNameTableRVA),
           reinterpret_cast<IMAGE_THUNK_DATA64*>(base + e->ImportAddressTableRVA), true);
    }
  return out;
}

Shim& process_shim() {
  static Shim s;
  return s;
}

BOOL WINAPI hooked_instance_id(HDEVINFO set, PSP_DEVINFO_DATA dev, PWSTR buf, DWORD size, PDWORD required) {
  return process_shim().instance_id(set, dev, buf, size, required);
}

BOOL WINAPI hooked_registry_property(HDEVINFO set, PSP_DEVINFO_DATA dev, DWORD property, PDWORD type, PBYTE buf,
                                     DWORD size, PDWORD required) {
  return process_shim().registry_property(set, dev, property, type, buf, size, required);
}

namespace {
const char kInstanceId[] = "SetupDiGetDeviceInstanceIdW";
const char kRegistryProperty[] = "SetupDiGetDeviceRegistryPropertyW";

bool inside(HMODULE module, const void* p) {
  const uint8_t* base = reinterpret_cast<const uint8_t*>(module);
  const IMAGE_DOS_HEADER* dos = reinterpret_cast<const IMAGE_DOS_HEADER*>(base);
  const IMAGE_NT_HEADERS64* nt = reinterpret_cast<const IMAGE_NT_HEADERS64*>(base + dos->e_lfanew);
  const uint8_t* q = static_cast<const uint8_t*>(p);
  return q >= base && q < base + nt->OptionalHeader.SizeOfImage;
}

bool patch_slot(void** slot, void* value) {
  DWORD old = 0;
  if (!VirtualProtect(slot, sizeof(void*), PAGE_READWRITE, &old)) return false;
  InterlockedExchangePointer(slot, value);
  DWORD ignored = 0;
  VirtualProtect(slot, sizeof(void*), old, &ignored);
  return true;
}
}  // namespace

unsigned install(HMODULE module, Shim::Resolver resolver, unsigned tries, Shim::LogFn log, std::string* detail) {
  if (detail) detail->clear();
  const std::vector<ImportSlot> slots = find_import_slots(module, "setupapi.dll", {kInstanceId, kRegistryProperty});
  // the functions the calls continue with: a bound slot's target (another layer's hook stays in the chain), else the export
  HMODULE sa = GetModuleHandleW(L"setupapi.dll");
  if (!sa) sa = LoadLibraryExW(L"setupapi.dll", nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32);
  void* next_id = sa ? reinterpret_cast<void*>(GetProcAddress(sa, kInstanceId)) : nullptr;
  void* next_prop = sa ? reinterpret_cast<void*>(GetProcAddress(sa, kRegistryProperty)) : nullptr;
  bool id_from_slot = false, prop_from_slot = false;
  for (const ImportSlot& s : slots) {
    if (!s.original || inside(module, s.original)) continue;  // unbound delay-load slot: the helper's thunk
    if (s.name == kInstanceId && !id_from_slot) { next_id = s.original; id_from_slot = true; }
    if (s.name == kRegistryProperty && !prop_from_slot) { next_prop = s.original; prop_from_slot = true; }
  }
  if (!next_id || !next_prop) {
    if (detail) *detail = "setupapi.dll's functions could not be found; nothing was redirected";
    return 0;
  }
  process_shim().configure_lazy(std::move(resolver), tries, reinterpret_cast<InstanceIdFn>(next_id),
                                reinterpret_cast<RegistryPropertyFn>(next_prop), std::move(log));
  unsigned patched = 0;
  std::string list;
  for (const ImportSlot& s : slots) {
    void* h = s.name == kInstanceId ? reinterpret_cast<void*>(&hooked_instance_id)
                                    : reinterpret_cast<void*>(&hooked_registry_property);
    if (s.original == h) continue;
    const bool ok = patch_slot(s.slot, h);
    patched += ok;
    list += (list.empty() ? "" : ", ") + s.name + (s.delay ? " (delay-load)" : " (import)") + (ok ? "" : " FAILED");
  }
  if (detail) *detail = list.empty() ? "the exe imports neither function from setupapi.dll; nothing was redirected" : list;
  return patched;
}

}  // namespace wgsi
