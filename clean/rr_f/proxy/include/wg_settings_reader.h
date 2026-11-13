// SPDX-License-Identifier: MIT
// WrongGPU shared settings block: the RR side's header-only reader (RR plugin and RR runtime), round i15.
//
// Implements the normative rules of clean/wgshared/include/wg_shared_settings.h (the same rules the SR front end's
// writer-side wg_settings_io.h implements): open-or-create of the process's named mapping, seeding of a new block from
// <data dir>\wronggpu.ini (fallback: a legacy ini the caller names, with a log line), magic published last, seqlock
// snapshots by reader mask, the text area, version / size / slot-count refusal, registry-hash mismatch logged only,
// single-writer election and the writer's seqlock write (the RR plugin writes only under the 2-s-alone rule), and the
// per-option precedence block slot (set_mask bit) > process environment > the reader's own ini > registry default.
//
// Seeding rule (both sides): a slot whose key is present in the ini is written with its set_mask bit; a slot whose key
// is absent is written with the registry default and its bit CLEAR, so a reader's environment / own ini still apply
// (tests and harnesses select modes through the environment and never see a seeded default override them).
//
// Log text: describe(slot, value, source) -> "rr_mode=fast (menu)", the one place the log words come from (the registry
// names). No dependency on another WrongGPU DLL: only Windows, the two shared headers and the C++ standard library.
#pragma once
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>

#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include <string>
#include <utility>
#include <vector>

#include "../../../wgshared/include/wg_registry_gen.h"
#include "../../../wgshared/include/wg_shared_settings.h"

namespace wgss {

// ------------------------------------------------------------------------------------------------ paths
inline std::wstring dir_of(const std::wstring& p) {
  const size_t s = p.find_last_of(L"\\/");
  return s == std::wstring::npos ? std::wstring() : p.substr(0, s + 1);
}
inline std::wstring module_path(HMODULE m) {
  wchar_t buf[MAX_PATH * 2];
  const DWORD n = GetModuleFileNameW(m, buf, (DWORD)(sizeof buf / sizeof buf[0]));
  return std::wstring(buf, n);
}
// Directory of the process executable, with a trailing backslash.
inline std::wstring exe_dir() { return dir_of(module_path(nullptr)); }
// The WrongGPU data folder (DESIGN-i15 section 9): WG_DATA_DIR when set (tests, harnesses), else <exe dir>\WrongGPU\.
inline std::wstring data_dir(bool* from_env = nullptr) {
  wchar_t buf[MAX_PATH * 2];
  const DWORD n = GetEnvironmentVariableW(L"WG_DATA_DIR", buf, (DWORD)(sizeof buf / sizeof buf[0]));
  if (from_env) *from_env = n > 0 && n < sizeof buf / sizeof buf[0];
  if (n > 0 && n < sizeof buf / sizeof buf[0]) {
    std::wstring d(buf, n);
    if (d.back() != L'\\' && d.back() != L'/') d.push_back(L'\\');
    return d;
  }
  return exe_dir() + L"WrongGPU\\";
}
inline bool file_exists(const std::wstring& p) {
  const DWORD a = GetFileAttributesW(p.c_str());
  return a != INVALID_FILE_ATTRIBUTES && !(a & FILE_ATTRIBUTE_DIRECTORY);
}
inline bool dir_exists(const std::wstring& p) {
  const DWORD a = GetFileAttributesW(p.c_str());
  return a != INVALID_FILE_ATTRIBUTES && (a & FILE_ATTRIBUTE_DIRECTORY);
}
inline std::string utf8(const std::wstring& w) {
  if (w.empty()) return std::string();
  const int n = WideCharToMultiByte(CP_UTF8, 0, w.c_str(), (int)w.size(), nullptr, 0, nullptr, nullptr);
  std::string s((size_t)(n > 0 ? n : 0), '\0');
  if (n > 0) WideCharToMultiByte(CP_UTF8, 0, w.c_str(), (int)w.size(), &s[0], n, nullptr, nullptr);
  return s;
}
inline std::wstring wide(const std::string& s) {
  if (s.empty()) return std::wstring();
  const int n = MultiByteToWideChar(CP_UTF8, 0, s.c_str(), (int)s.size(), nullptr, 0);
  std::wstring w((size_t)(n > 0 ? n : 0), L'\0');
  if (n > 0) MultiByteToWideChar(CP_UTF8, 0, s.c_str(), (int)s.size(), &w[0], n);
  return w;
}
// Last-write time of a file (0 when absent).
inline uint64_t file_mtime(const std::wstring& p) {
  WIN32_FILE_ATTRIBUTE_DATA d;
  if (!GetFileAttributesExW(p.c_str(), GetFileExInfoStandard, &d)) return 0;
  return ((uint64_t)d.ftLastWriteTime.dwHighDateTime << 32) | d.ftLastWriteTime.dwLowDateTime;
}

// ------------------------------------------------------------------------------------------------ ini (read-only parser)
struct IniEntry {
  std::string section, key, value;  // section and key lower-case; value trimmed, inline " ; comment" removed
};
inline std::string lower(std::string s) {
  for (char& c : s) c = (char)((c >= 'A' && c <= 'Z') ? c - 'A' + 'a' : c);
  return s;
}
inline std::string trim(const std::string& s) {
  size_t a = 0, b = s.size();
  while (a < b && strchr(" \t\r\n", s[a])) ++a;
  while (b > a && strchr(" \t\r\n", s[b - 1])) --b;
  return s.substr(a, b - a);
}
// Reads key=value lines ([section] headers recorded, '#' / ';' comment lines, an inline comment after a blank).
// Returns false when the file cannot be opened.
inline bool read_ini(const std::wstring& path, std::vector<IniEntry>* out) {
  out->clear();
  HANDLE h = CreateFileW(path.c_str(), GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, nullptr,
                         OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
  if (h == INVALID_HANDLE_VALUE) return false;
  std::string text;
  char buf[4096];
  DWORD got = 0;
  while (ReadFile(h, buf, sizeof buf, &got, nullptr) && got) {
    text.append(buf, got);
    if (text.size() > (1u << 20)) break;  // an ini larger than 1 MiB is not ours
  }
  CloseHandle(h);
  if (text.size() >= 3 && (unsigned char)text[0] == 0xEF && (unsigned char)text[1] == 0xBB && (unsigned char)text[2] == 0xBF)
    text.erase(0, 3);
  std::string section;
  size_t i = 0;
  while (i < text.size()) {
    size_t j = text.find('\n', i);
    if (j == std::string::npos) j = text.size();
    std::string line = trim(text.substr(i, j - i));
    i = j + 1;
    if (line.empty() || line[0] == '#' || line[0] == ';') continue;
    if (line[0] == '[') {
      const size_t e = line.find(']');
      section = lower(trim(line.substr(1, e == std::string::npos ? std::string::npos : e - 1)));
      continue;
    }
    const size_t eq = line.find('=');
    if (eq == std::string::npos) continue;
    std::string val = line.substr(eq + 1);
    const size_t cm = val.find_first_of(";#");
    if (cm != std::string::npos && (cm == 0 || val[cm - 1] == ' ' || val[cm - 1] == '\t')) val.resize(cm);
    out->push_back({section, lower(trim(line.substr(0, eq))), trim(val)});
  }
  return true;
}
// The last value of `key` in `section` ("" = any section); nullptr when absent.
inline const std::string* ini_find(const std::vector<IniEntry>& ini, const char* key, const char* section = "") {
  const std::string* v = nullptr;
  const std::string k = lower(key), s = lower(section);
  for (const IniEntry& e : ini)
    if (e.key == k && (s.empty() || e.section == s || e.section.empty())) v = &e.value;
  return v;
}

// ------------------------------------------------------------------------------------------------ registry helpers
inline const WgOptionDef* option_by_key(const std::string& key) {
  for (const WgOptionDef& o : kWgOptions)
    if (key == o.key) return &o;
  return nullptr;
}
// Legacy keys of the RR plugin's old wronggpu-rr.ini (the 60.9 contract keys) -> registry slots.
struct LegacyKey { const char* key; uint32_t slot; };
static const LegacyKey kLegacyKeys[] = {
    {"wronggpu_rr", WG_OPT_RR_ENABLED},
    {"specular_motion_from_v", WG_OPT_RR_SPECULAR_MOTION_FROM_V},
    {"reset_after_failed_evaluate", WG_OPT_RR_RESET_AFTER_FAILED_EVALUATE},
    {"change_frame_reset", WG_OPT_RR_CHANGE_FRAME_RESET},
    {"bounded_stores", WG_OPT_RR_BOUNDED_STORES},
    {"hit_distance_filter", WG_OPT_RR_HIT_DISTANCE_FILTER},
};
// Text value -> slot value: a choice name (case-insensitive) or a decimal within [min, max].
inline bool parse_value(const WgOptionDef& o, const std::string& text, int32_t* out) {
  const std::string t = lower(trim(text));
  if (t.empty()) return false;
  if (o.kind == WG_KIND_CHOICE && o.value_names) {
    for (int32_t i = o.min_value; i <= o.max_value; ++i)
      if (t == lower(o.value_names[i])) { *out = i; return true; }
  }
  if (o.kind == WG_KIND_ONOFF) {
    if (t == "on" || t == "true" || t == "yes") { *out = 1; return true; }
    if (t == "off" || t == "false" || t == "no") { *out = 0; return true; }
  }
  char* end = nullptr;
  const long v = strtol(t.c_str(), &end, 10);
  if (!end || *end || v < o.min_value || v > o.max_value) return false;
  *out = (int32_t)v;
  return true;
}
// The value of an option in an ini: its registry key (in [wronggpu] or no section), else a legacy alias.
inline bool ini_value(const std::vector<IniEntry>& ini, const WgOptionDef& o, int32_t* out) {
  const std::string* v = ini_find(ini, o.key, o.ini_section);
  if (v && parse_value(o, *v, out)) return true;
  for (const LegacyKey& l : kLegacyKeys)
    if (l.slot == o.id) {
      const std::string* lv = ini_find(ini, l.key);
      if (lv && parse_value(o, *lv, out)) return true;
    }
  return false;
}
inline std::string value_name(const WgOptionDef& o, int32_t v) {
  if (o.kind == WG_KIND_CHOICE && o.value_names && v >= o.min_value && v <= o.max_value) return o.value_names[v];
  char b[16];
  snprintf(b, sizeof b, "%d", (int)v);
  return b;
}
// "rr_mode=fast (menu)": the one formatter of option log text (registry key and value names).
inline std::string describe(uint32_t slot, int32_t value, const char* source) {
  if (slot >= WG_OPT_COUNT) return "slot " + std::to_string(slot) + "=" + std::to_string(value);
  return std::string(kWgOptions[slot].key) + "=" + value_name(kWgOptions[slot], value) + " (" + source + ")";
}
inline std::string env_text(const char* name) {
  if (!name || !*name) return std::string();
  char buf[1024];
  const DWORD n = GetEnvironmentVariableA(name, buf, sizeof buf);
  return n && n < sizeof buf ? std::string(buf, n) : std::string();
}
inline bool env_set(const char* name) {
  if (!name || !*name) return false;
  char b[2];
  return GetEnvironmentVariableA(name, b, sizeof b) != 0 || GetLastError() != ERROR_ENVVAR_NOT_FOUND;
}

// ------------------------------------------------------------------------------------------------ snapshot
struct Snapshot {
  bool present = false;
  uint32_t generation = 0, writer = 0, seed_source = 0, flags = 0;
  int32_t v[WG_SS_SLOT_COUNT] = {};
  uint32_t set_mask[2] = {0, 0}, live_mask[2] = {0, 0};
  bool is_set(uint32_t slot) const { return present && slot < WG_SS_SLOT_COUNT && ((set_mask[slot >> 5] >> (slot & 31)) & 1u); }
  bool is_live(uint32_t slot) const { return present && slot < WG_SS_SLOT_COUNT && ((live_mask[slot >> 5] >> (slot & 31)) & 1u); }
  // where a set slot's value came from (log words)
  const char* source(uint32_t slot) const {
    if (is_live(slot))
      return writer == WG_SS_WRITER_SR_FRONTEND ? "menu" : writer == WG_SS_WRITER_RR_PLUGIN ? "host ini poll"
             : writer == WG_SS_WRITER_TEST ? "test block" : "block";
    return seed_source == 1 ? "wronggpu.ini" : seed_source == 3 ? "test block" : "block";
  }
};

using LogFn = void (*)(const std::string&);

// ------------------------------------------------------------------------------------------------ the block
class Block {
 public:
  Block() = default;
  Block(const Block&) = delete;
  Block& operator=(const Block&) = delete;
  ~Block() {
    if (view_) UnmapViewOfFile((void*)view_);
    if (map_) CloseHandle(map_);
  }

  // Open the process's block; create and seed it when it does not exist and `create` is set. `self` = WG_SS_WRITER_*
  // id of the calling module (the creator id). Seeding source: <data dir>\wronggpu.ini, else `legacy_ini` (logged),
  // else registry defaults; `seed_test` = no ini at all (seed_source 3, test harnesses). `log` gets every line.
  bool open(uint32_t self, bool create, const std::wstring& legacy_ini, std::vector<std::string>* log,
            bool seed_test = false) {
    if (view_) return true;
    self_ = self;
    char name[64];
    snprintf(name, sizeof name, WG_SS_NAME_FMT, (unsigned long)GetCurrentProcessId());
    const std::wstring wname = wide(name);
    HANDLE m = nullptr;
    bool created = false;
    if (create) {
      SetLastError(0);
      m = CreateFileMappingW(INVALID_HANDLE_VALUE, nullptr, PAGE_READWRITE, 0, WG_SS_SIZE, wname.c_str());
      created = m && GetLastError() != ERROR_ALREADY_EXISTS;
    } else {
      m = OpenFileMappingW(FILE_MAP_READ | FILE_MAP_WRITE, FALSE, wname.c_str());
    }
    if (!m) return false;
    void* v = MapViewOfFile(m, FILE_MAP_READ | FILE_MAP_WRITE, 0, 0, WG_SS_SIZE);
    if (!v) {
      CloseHandle(m);
      return false;
    }
    map_ = m;
    view_ = (WgSharedSettingsBlock*)v;
    if (created) {
      seed(legacy_ini, log, seed_test);
      created_ = true;
      return true;
    }
    for (int i = 0; i < 50 && view_->header.magic != WG_SS_MAGIC; ++i) Sleep(1);  // the creator publishes magic last
    MemoryBarrier();
    if (view_->header.magic != WG_SS_MAGIC) {
      if (log) log->push_back("settings: shared block found but never published (no magic after 50 ms); treated as absent");
      close();
      return false;
    }
    const WgSharedSettingsHeader& h = view_->header;
    if (h.version != WG_SS_VERSION || h.size < WG_SS_HEADER_SIZE + 4u * h.slot_count || h.size > WG_SS_SIZE) {
      if (log)
        log->push_back("settings: shared block version " + std::to_string(h.version) + " size " + std::to_string(h.size) +
                       " not understood by this build (version " + std::to_string(WG_SS_VERSION) + "); treated as absent");
      close();
      return false;
    }
    if (h.registry_hash != WG_REGISTRY_HASH && log) {
      char b[160];
      snprintf(b, sizeof b, "settings: shared block registry hash 0x%08x differs from this build's 0x%08x (slot ids are "
               "append-only: values used)", h.registry_hash, (unsigned)WG_REGISTRY_HASH);
      log->push_back(b);
    }
    return true;
  }
  bool present() const { return view_ != nullptr; }
  bool created() const { return created_; }
  const WgSharedSettingsBlock* view() const { return view_; }

  // Seqlock snapshot of the slots in `reader_mask` (WG_READER_* bits; 0 = every slot). Keeps `last` on a torn read.
  bool snapshot(uint32_t reader_mask, Snapshot* s) const {
    if (!view_) {
      s->present = false;
      return false;
    }
    volatile WgSharedSettingsBlock* b = view_;
    for (int tries = 0; tries < 16; ++tries) {
      const uint32_t g0 = b->header.generation;
      if (g0 & 1u) {
        YieldProcessor();
        continue;
      }
      MemoryBarrier();
      Snapshot t;
      t.present = true;
      t.generation = g0;
      t.writer = b->header.writer;
      t.seed_source = b->header.seed_source;
      t.flags = b->header.flags;
      const uint32_t n = b->header.slot_count < WG_SS_SLOT_COUNT ? b->header.slot_count : WG_SS_SLOT_COUNT;
      for (uint32_t i = 0; i < n; ++i) {
        if (reader_mask && (i >= WG_OPT_COUNT || !(kWgOptions[i].readers & reader_mask))) continue;
        t.v[i] = b->values[i];
        if ((b->set_mask[i >> 5] >> (i & 31)) & 1u) t.set_mask[i >> 5] |= 1u << (i & 31);
        if ((b->live_mask[i >> 5] >> (i & 31)) & 1u) t.live_mask[i >> 5] |= 1u << (i & 31);
      }
      MemoryBarrier();
      if (b->header.generation == g0) {
        *s = t;
        return true;
      }
    }
    return s->present;  // torn 16 times: the caller keeps its last good snapshot
  }
  // A text option from the block's text area ("" when absent).
  std::string text(const char* key) const {
    std::string out;
    if (!view_) return out;
    const uint32_t used = view_->text_used < WG_SS_TEXT_SIZE ? view_->text_used : WG_SS_TEXT_SIZE;
    const std::string k = std::string(key) + "=";
    uint32_t i = 0;
    while (i < used) {
      const char* s = view_->text + i;
      const size_t n = strnlen(s, used - i);
      if (n >= k.size() && memcmp(s, k.data(), k.size()) == 0) out.assign(s + k.size(), n - k.size());
      i += (uint32_t)n + 1;
    }
    return out;
  }
  bool text_present(const char* key) const {
    if (!view_) return false;
    const uint32_t used = view_->text_used < WG_SS_TEXT_SIZE ? view_->text_used : WG_SS_TEXT_SIZE;
    const std::string k = std::string(key) + "=";
    for (uint32_t i = 0; i < used;) {
      const char* s = view_->text + i;
      const size_t n = strnlen(s, used - i);
      if (n >= k.size() && memcmp(s, k.data(), k.size()) == 0) return true;
      i += (uint32_t)n + 1;
    }
    return false;
  }
  uint32_t writer() const { return view_ ? view_->header.writer : WG_SS_WRITER_NONE; }
  // Single-writer election: compare-exchange of `writer` from NONE. True when `id` holds it afterwards.
  bool claim_writer(uint32_t id) {
    if (!view_) return false;
    const LONG prev = InterlockedCompareExchange((volatile LONG*)&view_->header.writer, (LONG)id, (LONG)WG_SS_WRITER_NONE);
    if (prev == (LONG)WG_SS_WRITER_NONE) {
      view_->header.writer_pid = GetCurrentProcessId();
      return true;
    }
    return prev == (LONG)id;
  }
  // The writer's seqlock write of one slot (refused unless `id` is the elected writer).
  bool write_slot(uint32_t id, uint32_t slot, int32_t value) {
    if (!view_ || slot >= WG_SS_SLOT_COUNT || view_->header.writer != id) return false;
    locked_write(slot, value);
    return true;
  }
  // The RR plugin's menu panel owns rr_mode (wg_shared_settings.h, slot owners): its choice is written whether or not
  // the RR plugin is the elected writer. Any other slot is refused.
  bool write_panel_slot(uint32_t slot, int32_t value) {
    if (!view_ || slot != WG_OPT_RR_MODE) return false;
    locked_write(slot, value);
    return true;
  }
  void close() {
    if (view_) UnmapViewOfFile((void*)view_);
    if (map_) CloseHandle(map_);
    view_ = nullptr;
    map_ = nullptr;
  }

 private:
  // Enter the seqlock by compare-exchange even -> odd (two modules may write: the elected writer and the panel's owned
  // slot), write, publish the next even value.
  void locked_write(uint32_t slot, int32_t value) {
    volatile WgSharedSettingsBlock* b = view_;
    volatile LONG* gen = (volatile LONG*)&b->header.generation;
    LONG g;
    for (;;) {
      g = InterlockedCompareExchange(gen, 0, 0);
      if ((g & 1) == 0 && InterlockedCompareExchange(gen, g + 1, g) == g) break;
      YieldProcessor();
    }
    b->values[slot] = value;
    InterlockedOr((volatile LONG*)&b->set_mask[slot >> 5], (LONG)(1u << (slot & 31)));
    InterlockedOr((volatile LONG*)&b->live_mask[slot >> 5], (LONG)(1u << (slot & 31)));
    InterlockedExchange(gen, g + 2);  // even again: published
  }

  void seed(const std::wstring& legacy_ini, std::vector<std::string>* log, bool seed_test) {
    WgSharedSettingsBlock* b = view_;
    memset((void*)b, 0, WG_SS_SIZE);
    b->header.version = WG_SS_VERSION;
    b->header.size = WG_SS_SIZE;
    b->header.slot_count = WG_SS_SLOT_COUNT;
    b->header.registry_hash = WG_REGISTRY_HASH;
    b->header.creator = self_;
    std::vector<IniEntry> ini;
    std::wstring used;
    bool have = false;
    if (!seed_test) {
      const std::wstring main = data_dir() + L"wronggpu.ini";
      have = read_ini(main, &ini);
      used = main;
      if (!have && !legacy_ini.empty() && read_ini(legacy_ini, &ini)) {
        have = true;
        used = legacy_ini;
        if (log) log->push_back("settings: " + utf8(main) + " not found; seeded from the legacy " + utf8(legacy_ini));
      }
    }
    b->header.seed_source = seed_test ? 3u : (have ? 1u : 2u);
    for (const WgOptionDef& o : kWgOptions) {
      int32_t v = o.default_value;
      if (have && ini_value(ini, o, &v)) b->set_mask[o.id >> 5] |= 1u << (o.id & 31);
      b->values[o.id] = v;
    }
    uint32_t used_bytes = 0;
    for (const WgTextOptionDef& t : kWgTextOptions) {
      const std::string* v = have ? ini_find(ini, t.key) : nullptr;
      if (!v) continue;  // absent text options stay out of the text area (the reader's own fallback applies)
      const std::string kv = std::string(t.key) + "=" + *v;
      if (used_bytes + kv.size() + 1 > WG_SS_TEXT_SIZE) break;
      memcpy(b->text + used_bytes, kv.c_str(), kv.size() + 1);
      used_bytes += (uint32_t)kv.size() + 1;
    }
    b->text_used = used_bytes;
    if (log)
      log->push_back(std::string("settings: shared block created by this module, seeded from ") +
                     (seed_test ? std::string("nothing (test)") : have ? utf8(used) : std::string("registry defaults (no ini)")));
    MemoryBarrier();
    InterlockedExchange((volatile LONG*)&b->header.magic, (LONG)WG_SS_MAGIC);  // published last (release)
  }

  HANDLE map_ = nullptr;
  WgSharedSettingsBlock* view_ = nullptr;
  uint32_t self_ = WG_SS_WRITER_NONE;
  bool created_ = false;
};

// ------------------------------------------------------------------------------------------------ resolution
struct Resolved {
  int32_t value = 0;
  const char* source = "default";
};
// Precedence of one option: block slot (set_mask bit) > process environment (the option's env) > own ini > default.
inline Resolved resolve(const WgOptionDef& o, const Snapshot* snap, const std::vector<IniEntry>* own_ini,
                        const char* own_ini_name = "ini") {
  Resolved r;
  r.value = o.default_value;
  if (snap && snap->is_set(o.id)) {
    r.value = snap->v[o.id];
    if (r.value < o.min_value || r.value > o.max_value) r.value = o.default_value;
    r.source = snap->source(o.id);
    return r;
  }
  if (o.env && *o.env) {
    const std::string e = env_text(o.env);
    int32_t v;
    if (!e.empty() && parse_value(o, e, &v)) {
      r.value = v;
      r.source = o.env;
      return r;
    }
  }
  int32_t v;
  if (own_ini && ini_value(*own_ini, o, &v)) {
    r.value = v;
    r.source = own_ini_name;
  }
  return r;
}
// A text option: block text area > environment > own ini > registry default.
inline std::string resolve_text(const char* key, const Block* blk, const std::vector<IniEntry>* own_ini,
                                const char** source = nullptr) {
  const WgTextOptionDef* t = nullptr;
  for (const WgTextOptionDef& x : kWgTextOptions)
    if (strcmp(x.key, key) == 0) t = &x;
  if (blk && blk->text_present(key)) {
    if (source) *source = "block";
    return blk->text(key);
  }
  if (t && t->env && *t->env) {
    const std::string e = env_text(t->env);
    if (!e.empty()) {
      if (source) *source = t->env;
      return e;
    }
  }
  if (own_ini)
    if (const std::string* v = ini_find(*own_ini, key)) {
      if (source) *source = "ini";
      return *v;
    }
  if (source) *source = "default";
  return t && t->default_value ? t->default_value : "";
}

// Minimal ini writer for the elected writer's persistence: sets key=value in the file, keeping every other line
// (comments, unknown keys, other sections); appends under [section] (created when missing).
inline bool ini_set(const std::wstring& path, const char* section, const char* key, const std::string& value) {
  std::string text;
  HANDLE h = CreateFileW(path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, 0, nullptr);
  if (h != INVALID_HANDLE_VALUE) {
    char buf[4096];
    DWORD got = 0;
    while (ReadFile(h, buf, sizeof buf, &got, nullptr) && got) text.append(buf, got);
    CloseHandle(h);
  }
  std::vector<std::string> lines;
  for (size_t i = 0; i <= text.size();) {
    size_t j = text.find('\n', i);
    if (j == std::string::npos) j = text.size();
    std::string l = text.substr(i, j - i);
    if (!l.empty() && l.back() == '\r') l.pop_back();
    if (!(j == text.size() && l.empty())) lines.push_back(l);
    i = j + 1;
  }
  const std::string want_sec = lower(section ? section : ""), want_key = lower(key);
  std::string sec;
  int sec_end = -1, hit = -1;
  for (size_t i = 0; i < lines.size(); ++i) {
    const std::string t = trim(lines[i]);
    if (!t.empty() && t[0] == '[') {
      const size_t e = t.find(']');
      sec = lower(trim(t.substr(1, e == std::string::npos ? std::string::npos : e - 1)));
      continue;
    }
    if (sec == want_sec || (want_sec.empty() && sec.empty())) {
      sec_end = (int)i;
      const size_t eq = t.find('=');
      if (eq != std::string::npos && t[0] != ';' && t[0] != '#' && lower(trim(t.substr(0, eq))) == want_key) hit = (int)i;
    }
  }
  const std::string kv = std::string(key) + "=" + value;
  if (hit >= 0) lines[(size_t)hit] = kv;
  else if (sec_end >= 0) lines.insert(lines.begin() + sec_end + 1, kv);
  else {
    if (!want_sec.empty()) lines.push_back("[" + std::string(section) + "]");
    lines.push_back(kv);
  }
  std::string outtext;
  for (const std::string& l : lines) outtext += l + "\r\n";
  const std::wstring tmp = path + L".tmp";
  h = CreateFileW(tmp.c_str(), GENERIC_WRITE, 0, nullptr, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
  if (h == INVALID_HANDLE_VALUE) return false;
  DWORD put = 0;
  const bool ok = WriteFile(h, outtext.data(), (DWORD)outtext.size(), &put, nullptr) && put == outtext.size();
  CloseHandle(h);
  return ok && MoveFileExW(tmp.c_str(), path.c_str(), MOVEFILE_REPLACE_EXISTING);
}

}  // namespace wgss
