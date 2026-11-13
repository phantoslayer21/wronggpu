// SPDX-License-Identifier: MIT
#include "settings.h"

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <utility>

namespace wgrr {

// clean/060-rr-plugin-hunt: the settings path is UTF-8 (the plugin builds it from the wide game folder); the file is
// opened by its wide name, so a game in a folder the ANSI code page cannot spell (Cyrillic, accented or CJK names) still
// reads and saves it. A path that is not valid UTF-8 is taken in the ANSI code page (a narrow caller's path).
static std::wstring wide_path(const std::string& p) {
  if (p.empty()) return std::wstring();
  UINT cp = CP_UTF8;
  int n = MultiByteToWideChar(cp, MB_ERR_INVALID_CHARS, p.c_str(), (int)p.size(), nullptr, 0);
  if (n <= 0) {
    cp = CP_ACP;
    n = MultiByteToWideChar(cp, 0, p.c_str(), (int)p.size(), nullptr, 0);
  }
  if (n <= 0) return std::wstring();
  std::wstring w((size_t)n, L'\0');
  MultiByteToWideChar(cp, 0, p.c_str(), (int)p.size(), &w[0], n);
  return w;
}

const SettingDef kSettingDefs[kSettingCount] = {
  {"wronggpu_rr", "WrongGPU RR", WG_RR_SETTING_KIND_ONOFF, WG_RR_SETTING_FLAG_AT_CREATION, 1, 1},
  {"specular_motion_from_v", "Specular motion from motion vectors (SV = V)", WG_RR_SETTING_KIND_ONOFF,
   WG_RR_SETTING_FLAG_OPT_IN, 0, 1},
  {"reset_after_failed_evaluate", "Reset history after a failed frame", WG_RR_SETTING_KIND_ONOFF,
   WG_RR_SETTING_FLAG_OPT_IN, 0, 1},
  {"change_frame_reset", "Reset history on the frame of a render-size change", WG_RR_SETTING_KIND_ONOFF,
   WG_RR_SETTING_FLAG_OPT_IN, 0, 1},
  {"bounded_stores", "Write only inside the output rectangle", WG_RR_SETTING_KIND_ONOFF,
   WG_RR_SETTING_FLAG_OPT_IN, 0, 1},
  {"hit_distance_filter", "Filter lone outliers of the reflection hit distance", WG_RR_SETTING_KIND_ONOFF,
   WG_RR_SETTING_FLAG_OPT_IN, 0, 1},
};

Settings::Settings() {
  for (uint32_t i = 0; i < kSettingCount; ++i) values_[i].store(kSettingDefs[i].default_value);
}

static bool parse_int(const char* s, int32_t* out) {
  while (*s == ' ' || *s == '\t') ++s;
  if (!*s) return false;
  char* end = nullptr;
  long v = strtol(s, &end, 10);
  while (*end == ' ' || *end == '\t' || *end == '\r' || *end == '\n') ++end;
  if (*end) return false;
  if (v < -2147483647L - 1 || v > 2147483647L) return false;
  *out = (int32_t)v;
  return true;
}

void Settings::load(const std::string& path, const LogFn& log) {
  path_ = path;
  log_ = log;
  extra_.clear();
  for (uint32_t i = 0; i < kSettingCount; ++i) values_[i].store(kSettingDefs[i].default_value);
  const std::wstring wpath = wide_path(path);
  FILE* f = wpath.empty() ? nullptr : _wfopen(wpath.c_str(), L"rb");
  if (!f) {
    // A missing file is the normal first run; an existing file that cannot be opened is unreadable (logged once).
    if (!wpath.empty() && GetFileAttributesW(wpath.c_str()) != INVALID_FILE_ATTRIBUTES && log_)
      log_("settings file unreadable, defaults used: " + path_);
    return;
  }
  char line[512];
  bool bad = false;
  while (fgets(line, sizeof line, f)) {
    if (strlen(line) == sizeof line - 1 && line[sizeof line - 2] != '\n') { bad = true; break; }
    char* p = line;
    while (*p == ' ' || *p == '\t') ++p;
    if (*p == '#' || *p == ';' || *p == '[' || *p == '\r' || *p == '\n' || !*p) continue;
    char* eq = strchr(p, '=');
    if (!eq) continue;
    *eq = 0;
    char* k_end = eq;
    while (k_end > p && (k_end[-1] == ' ' || k_end[-1] == '\t')) *--k_end = 0;
    int32_t v;
    bool known = false;
    for (uint32_t i = 0; i < kSettingCount; ++i) known = known || strcmp(p, kSettingDefs[i].key) == 0;
    if (!known && *p) {   // clean/rr-f-live2: a non-contract key (rr_mode, rr_model): kept as text, rewritten on save
      std::string val(eq + 1);
      const size_t cm = val.find_first_of(";#");
      if (cm != std::string::npos && (cm == 0 || val[cm - 1] == ' ' || val[cm - 1] == '\t')) val.resize(cm);
      while (!val.empty() && strchr(" \t\r\n", val.back())) val.pop_back();
      size_t a = 0;
      while (a < val.size() && (val[a] == ' ' || val[a] == '\t')) ++a;
      extra_.push_back({p, val.substr(a)});
    }
    for (uint32_t i = 0; i < kSettingCount; ++i) {
      if (strcmp(p, kSettingDefs[i].key) != 0) continue;
      if (parse_int(eq + 1, &v) && v >= 0 && (uint32_t)v <= kSettingDefs[i].max_value) values_[i].store(v);
      // unknown or invalid value: the default stays
    }
  }
  if (ferror(f)) bad = true;
  fclose(f);
  if (bad) {
    for (uint32_t i = 0; i < kSettingCount; ++i) values_[i].store(kSettingDefs[i].default_value);
    if (log_) log_("settings file unreadable, defaults used: " + path_);
  }
}

std::string Settings::text(const char* key) const {
  std::string v;
  for (const auto& kv : extra_)
    if (kv.first == key) v = kv.second;   // the last one wins
  return v;
}

int32_t Settings::get(uint32_t id) const { return id < kSettingCount ? values_[id].load() : 0; }

uint32_t Settings::set(uint32_t id, int32_t value) {
  if (id >= kSettingCount) return 25;
  if (value < 0 || (uint32_t)value > kSettingDefs[id].max_value) return 25;
  std::lock_guard<std::mutex> lock(save_mutex_);
  int32_t old = values_[id].exchange(value);
  if (old != value) save_locked();
  return 0;
}

SettingsSnapshot Settings::snapshot() const {
  SettingsSnapshot s;
  for (uint32_t i = 0; i < kSettingCount; ++i) s.v[i] = values_[i].load();
  return s;
}

void Settings::save_locked() {
  if (!save_enabled_ || path_.empty()) return;
  const std::wstring wpath = wide_path(path_), wtmp = wide_path(path_ + ".tmp");
  FILE* f = wpath.empty() || wtmp.empty() ? nullptr : _wfopen(wtmp.c_str(), L"wb");
  if (!f) {
    if (log_) log_("settings file could not be written: " + path_);
    return;
  }
  fputs("# WrongGPU RR settings (written by the proxy; change them in OptiScaler's overlay)\n", f);
  for (uint32_t i = 0; i < kSettingCount; ++i) fprintf(f, "%s=%d\n", kSettingDefs[i].key, (int)values_[i].load());
  for (const auto& kv : extra_) fprintf(f, "%s=%s\n", kv.first.c_str(), kv.second.c_str());
  bool ok = fclose(f) == 0;
  if (ok) {
    _wremove(wpath.c_str());
    ok = _wrename(wtmp.c_str(), wpath.c_str()) == 0;
  }
  if (!ok && log_) log_("settings file could not be replaced: " + path_);
}

// ---- contract table ----
namespace {
Settings* g_settings = nullptr;
std::function<uint32_t()> g_status;

template <uint32_t I>
int32_t get_thunk(void) { return g_settings ? g_settings->get(I) : kSettingDefs[I].default_value; }
template <uint32_t I>
uint32_t set_thunk(int32_t v) { return g_settings ? g_settings->set(I, v) : 25u; }
uint32_t status_thunk(void) { return g_status ? (g_status() & 7u) : 0u; }

#define WG_ENTRY(I) {kSettingDefs[I].key, kSettingDefs[I].label, kSettingDefs[I].kind, kSettingDefs[I].flags, \
                     kSettingDefs[I].default_value, kSettingDefs[I].max_value, &get_thunk<I>, &set_thunk<I>}
const WgRrSettingEntry* entries() {
  static const WgRrSettingEntry e[kSettingCount] = {WG_ENTRY(0), WG_ENTRY(1), WG_ENTRY(2), WG_ENTRY(3), WG_ENTRY(4), WG_ENTRY(5)};
  return e;
}
#undef WG_ENTRY
static_assert(kSettingCount == 6, "update the entry list");
}  // namespace

void bind_contract(Settings* settings, std::function<uint32_t()> status) {
  g_settings = settings;
  g_status = std::move(status);
}

const WgRrSettingsContract* settings_contract() {
  static const WgRrSettingsContract c = {WG_RR_CONTRACT_V1_SIZE, WG_RR_CONTRACT_VERSION, kSettingCount, 0,
                                         entries(), &status_thunk, nullptr};
  return &c;
}

}  // namespace wgrr
