// SPDX-License-Identifier: MIT
/* WrongGPU settings I/O: the ONE implementation of the shared settings block, wronggpu.ini, the data folder rule and
 * the settings log words, used by every WrongGPU DLL (SR front end, RR plugin, RR runtime). clean/wgshared.
 *
 * Header-only C++17. Two parts:
 *   - always available, no Windows header: option tables from wg_registry_gen.h, value names, describe() log text,
 *     the ini reader / comment-preserving ini writer on text, the host-ini (OptiScaler.ini) parser, precedence;
 *   - Win32 part, compiled in exactly ONE translation unit per module that defines WG_SETTINGS_IO_IMPLEMENTATION
 *     before including this file: data folder, files, the named mapping (open / create / seed / seqlock / writer
 *     election / write + persist), OptiScaler.ini polling. <windows.h> is included only inside that part, so no
 *     Windows header leaks into the files that only use the declarations.
 *
 * Normative meaning of the block: wg_shared_settings.h. Reader precedence for a slot everywhere:
 *   block slot (set_mask bit) > process environment (the option's env) > own ini > registry default.
 * Values seeded from wronggpu.ini set their set_mask bit; registry defaults do not (an unset slot means "default").
 * A slot whose live_mask bit is set was written by the elected writer after seeding (the host's menu): source "menu".
 */
#ifndef WG_SETTINGS_IO_H
#define WG_SETTINGS_IO_H

#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cwctype>
#include <map>
#include <string>
#include <vector>

#include <wg_registry_gen.h>
#include <wg_registry_ini_gen.h>
#include "wg_shared_settings.h"

namespace wg {

// Where a value came from (the word in parentheses of every "settings: key=value (source)" log line).
enum class Source : int { Default = 0, Ini, Env, Menu, HostIni, Sidecar, Test };

inline const char* source_word(Source s) {
    switch (s) {
    case Source::Default: return "default";
    case Source::Ini: return "ini";
    case Source::Env: return "env";
    case Source::Menu: return "menu";
    case Source::HostIni: return "optiscaler.ini";
    case Source::Sidecar: return "folder";
    case Source::Test: return "test";
    }
    return "?";
}

struct Value {
    int32_t value = 0;
    Source source = Source::Default;
};

// ---- registry lookups --------------------------------------------------------------------------------------------

inline std::string lower(std::string s) {
    for (char& c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return s;
}

inline std::string trim(const std::string& s) {
    size_t b = 0, e = s.size();
    while (b < e && std::isspace(static_cast<unsigned char>(s[b]))) ++b;
    while (e > b && std::isspace(static_cast<unsigned char>(s[e - 1]))) --e;
    return s.substr(b, e - b);
}

inline int slot_of_key(const std::string& key) {
    const std::string k = lower(trim(key));
    for (uint32_t i = 0; i < WG_OPT_COUNT; ++i)
        if (k == kWgOptions[i].key) return static_cast<int>(i);
    return -1;
}

inline int text_option_of_key(const std::string& key) {
    const std::string k = lower(trim(key));
    for (uint32_t i = 0; i < WG_TEXT_OPT_COUNT; ++i)
        if (k == kWgTextOptions[i].key) return static_cast<int>(i);
    return -1;
}

// Any key of the generated [wronggpu] section (slot or text option).
inline bool is_registry_key(const std::string& key) { return slot_of_key(key) >= 0 || text_option_of_key(key) >= 0; }

// Bit i set: slot i is read by one of `readers` (WG_READER_* bits).
inline uint64_t reader_mask(uint32_t readers) {
    uint64_t m = 0;
    for (uint32_t i = 0; i < WG_OPT_COUNT; ++i)
        if (kWgOptions[i].readers & readers) m |= 1ull << i;
    return m;
}

inline bool value_in_range(uint32_t slot, int32_t v) {
    return slot < WG_OPT_COUNT && v >= kWgOptions[slot].min_value && v <= kWgOptions[slot].max_value;
}

// "fast" for rr_mode 2, "1" for an onoff, the number for an int.
inline std::string value_name(uint32_t slot, int32_t value) {
    if (slot >= WG_OPT_COUNT) return std::to_string(value);
    const WgOptionDef& o = kWgOptions[slot];
    if (o.kind == WG_KIND_CHOICE && o.value_names && value >= 0 && value <= o.max_value) return o.value_names[value];
    return std::to_string(value);
}

// Ini / environment text -> value: a choice by name (any case) or index, onoff as 0/1/on/off/true/false/yes/no, an
// int in range. False (out untouched) when the text is not a valid value of the slot.
inline bool parse_value(uint32_t slot, const std::string& text, int32_t* out) {
    if (slot >= WG_OPT_COUNT) return false;
    const WgOptionDef& o = kWgOptions[slot];
    const std::string t = lower(trim(text));
    if (t.empty()) return false;
    if (o.kind == WG_KIND_CHOICE && o.value_names) {
        for (int32_t i = 0; i <= o.max_value; ++i)
            if (t == lower(o.value_names[i])) { *out = i; return true; }
    }
    if (o.kind == WG_KIND_ONOFF) {
        if (t == "on" || t == "true" || t == "yes") { *out = 1; return true; }
        if (t == "off" || t == "false" || t == "no") { *out = 0; return true; }
    }
    char* end = nullptr;
    const long v = std::strtol(t.c_str(), &end, 10);
    if (end == t.c_str() || *end) return false;
    if (v < o.min_value || v > o.max_value) return false;
    *out = static_cast<int32_t>(v);
    return true;
}

// "rr_mode=fast (menu)": the one log text for an option value (every DLL logs every option it reads through this).
inline std::string describe(uint32_t slot, int32_t value, Source source) {
    const std::string key = slot < WG_OPT_COUNT ? kWgOptions[slot].key : "slot" + std::to_string(slot);
    return key + "=" + value_name(slot, value) + " (" + source_word(source) + ")";
}

inline std::string describe_text(const std::string& key, const std::string& value, Source source) {
    return key + "=" + value + " (" + source_word(source) + ")";
}

// The entry of a version-list id (kWgEntries), or null.
inline const WgEntryDef* entry_of_id(uint64_t id) {
    for (uint32_t i = 0; i < WG_ENTRY_COUNT; ++i)
        if (kWgEntries[i].id == id) return &kWgEntries[i];
    return nullptr;
}

inline int entry_index_of_id(uint64_t id) {
    for (uint32_t i = 0; i < WG_ENTRY_COUNT; ++i)
        if (kWgEntries[i].id == id) return static_cast<int>(i);
    return -1;
}

// ---- ini text ----------------------------------------------------------------------------------------------------

// Section/key lookup over ini text: "key = value" lines, sections in brackets, '#' or ';' comments at the start of a
// line or after a space or tab (so "a = x;y" keeps its ';'). Keys and sections compare without case.
struct IniValues {
    std::map<std::string, std::string> values;  // "section\nkey" (lower case) -> value
    bool present = false;                        // the text existed (a file was read)
    bool get(const std::string& section, const std::string& key, std::string* out) const {
        auto it = values.find(lower(section) + "\n" + lower(key));
        if (it == values.end()) return false;
        if (out) *out = it->second;
        return true;
    }
};

inline std::string strip_comment(const std::string& line) {
    for (size_t i = 0; i < line.size(); ++i)
        if ((line[i] == '#' || line[i] == ';') && (i == 0 || line[i - 1] == ' ' || line[i - 1] == '\t')) return line.substr(0, i);
    return line;
}

inline IniValues parse_ini(const std::string& text) {
    IniValues ini;
    ini.present = true;
    std::string section;
    size_t a = 0;
    while (a < text.size()) {
        size_t b = text.find('\n', a);
        if (b == std::string::npos) b = text.size();
        const std::string line = trim(strip_comment(text.substr(a, b - a)));
        a = b + 1;
        if (line.empty()) continue;
        if (line[0] == '[') {
            const size_t close = line.find(']');
            section = lower(trim(line.substr(1, close == std::string::npos ? std::string::npos : close - 1)));
            continue;
        }
        const size_t eq = line.find('=');
        if (eq == std::string::npos) continue;
        ini.values[section + "\n" + lower(trim(line.substr(0, eq)))] = trim(line.substr(eq + 1));
    }
    return ini;
}

// Sets `key = value` in [section] of ini text, keeping every other line (comments, unknown keys, other sections) and
// a trailing comment on the key's own line. A missing key goes after the last non-empty line of its section; a
// missing section is appended. Line ends follow the text (CRLF when it has any, else LF; CRLF for empty text).
inline std::string ini_set(const std::string& text, const std::string& section, const std::string& key, const std::string& value) {
    const std::string eol = text.empty() || text.find("\r\n") != std::string::npos ? "\r\n" : "\n";
    std::vector<std::string> lines;
    size_t a = 0;
    while (a < text.size()) {
        size_t b = text.find('\n', a);
        if (b == std::string::npos) b = text.size();
        std::string line = text.substr(a, b - a);
        if (!line.empty() && line.back() == '\r') line.pop_back();
        lines.push_back(line);
        a = b + 1;
    }
    const bool trailingNewline = !text.empty() && text.back() == '\n';
    const std::string wantSection = lower(section), wantKey = lower(key);
    std::string current;
    int sectionStart = -1, lastInSection = -1, found = -1;
    for (size_t i = 0; i < lines.size(); ++i) {
        const std::string t = trim(strip_comment(lines[i]));
        if (t.empty()) continue;
        if (t[0] == '[') {
            const size_t close = t.find(']');
            current = lower(trim(t.substr(1, close == std::string::npos ? std::string::npos : close - 1)));
            if (current == wantSection && sectionStart < 0) { sectionStart = static_cast<int>(i); lastInSection = static_cast<int>(i); }
            continue;
        }
        if (current != wantSection) continue;
        lastInSection = static_cast<int>(i);
        const size_t eq = t.find('=');
        if (eq != std::string::npos && lower(trim(t.substr(0, eq))) == wantKey && found < 0) found = static_cast<int>(i);
    }
    const std::string newLine = key + " = " + value;
    if (found >= 0) {
        const std::string& old = lines[size_t(found)];
        const std::string body = strip_comment(old);
        std::string comment = old.substr(body.size());
        size_t ws = body.size();
        while (ws > 0 && (body[ws - 1] == ' ' || body[ws - 1] == '\t')) --ws;
        lines[size_t(found)] = newLine + (comment.empty() ? std::string() : body.substr(ws) + comment);
    } else if (sectionStart >= 0) {
        lines.insert(lines.begin() + lastInSection + 1, newLine);
    } else {
        if (!lines.empty() && !trim(lines.back()).empty()) lines.push_back("");
        lines.push_back("[" + section + "]");
        lines.push_back(newLine);
    }
    std::string out;
    for (size_t i = 0; i < lines.size(); ++i) {
        out += lines[i];
        if (i + 1 < lines.size() || trailingNewline || found < 0) out += eol;
    }
    return out;
}

// ---- precedence without a block (env > ini > default) ------------------------------------------------------------

// The process environment (the C runtime's view, which _putenv_s and the game's own settings update).
inline bool get_env(const char* name, std::string* out) {
#if defined(_MSC_VER)
    char* buf = nullptr;
    size_t len = 0;
    if (_dupenv_s(&buf, &len, name) != 0 || !buf) return false;
    if (out) *out = buf;
    std::free(buf);
    return true;
#else
    const char* e = std::getenv(name);
    if (!e) return false;
    if (out) *out = e;
    return true;
#endif
}

inline Value resolve_unblocked(uint32_t slot, const IniValues* ini) {
    Value v;
    if (slot >= WG_OPT_COUNT) return v;
    const WgOptionDef& o = kWgOptions[slot];
    v.value = o.default_value;
    std::string e;
    if (o.env && *o.env && get_env(o.env, &e)) {
        int32_t x = 0;
        if (parse_value(slot, e, &x)) { v.value = x; v.source = Source::Env; return v; }
    }
    std::string text;
    if (ini && ini->get(o.ini_section, o.key, &text)) {
        int32_t x = 0;
        if (parse_value(slot, text, &x)) { v.value = x; v.source = Source::Ini; }
    }
    return v;
}

// A copy of the slots of the block (seqlock snapshot); valid = at least one consistent read happened.
struct Snapshot {
    int32_t values[WG_SS_SLOT_COUNT] = {};
    uint32_t set_mask[2] = {};
    uint32_t live_mask[2] = {};
    uint32_t generation = 0;
    uint32_t slot_count = 0;
    uint32_t writer = WG_SS_WRITER_NONE;
    uint32_t flags = 0;
    bool valid = false;
    bool is_set(uint32_t slot) const { return valid && slot < slot_count && slot < 64 && (set_mask[slot >> 5] >> (slot & 31)) & 1u; }
    bool is_live(uint32_t slot) const { return valid && slot < 64 && (live_mask[slot >> 5] >> (slot & 31)) & 1u; }
};

// block slot (set) > env > ini > default.
inline Value resolve(uint32_t slot, const Snapshot* snap, const IniValues* ini) {
    if (snap && snap->is_set(slot) && value_in_range(slot, snap->values[slot])) {
        Value v;
        v.value = snap->values[slot];
        v.source = snap->is_live(slot) ? Source::Menu : Source::Ini;
        return v;
    }
    return resolve_unblocked(slot, ini);
}

// ---- the host's ini (OptiScaler.ini): [<custom folder>] UpscalerIndex ----------------------------------------------

// The index from host-ini text, or false ("auto", missing, not a number). Read-only, our own reader.
inline bool parse_host_index(const std::string& text, const std::string& section, int* index) {
    std::string v;
    if (!parse_ini(text).get(section, "UpscalerIndex", &v)) return false;
    v = lower(trim(v));
    char* end = nullptr;
    const long n = std::strtol(v.c_str(), &end, 10);
    if (v.empty() || end == v.c_str() || *end || n < 0 || n > 1024) return false;
    *index = static_cast<int>(n);
    return true;
}

// Custom-slot detection from a module's own path: "...\custom\<folder>\ffx\<file>" -> <folder> (original case) and
// the host's ini next to the folder that holds "custom" (<that folder's parent>\OptiScaler.ini, i.e. OptiDllPath's
// parent). Empty when the path is not in a custom slot.
inline std::wstring custom_slot_folder(const std::wstring& modulePath, std::wstring* hostIni = nullptr) {
    std::wstring p = modulePath;
    for (wchar_t& c : p) if (c == L'/') c = L'\\';
    std::wstring l = p;
    for (wchar_t& c : l) c = static_cast<wchar_t>(std::towlower(c));
    const size_t file = l.find_last_of(L'\\');
    if (file == std::wstring::npos || file < 5) return std::wstring();
    const size_t slot = l.find_last_of(L'\\', file - 1);
    if (slot == std::wstring::npos || l.compare(slot + 1, file - slot - 1, L"ffx") != 0) return std::wstring();
    const size_t folder = l.find_last_of(L'\\', slot - 1);
    if (folder == std::wstring::npos || folder < 7) return std::wstring();
    const size_t custom = l.find_last_of(L'\\', folder - 1);
    if (custom == std::wstring::npos || l.compare(custom + 1, folder - custom - 1, L"custom") != 0) return std::wstring();
    if (hostIni) {
        const size_t opti = l.find_last_of(L'\\', custom - 1);
        *hostIni = opti == std::wstring::npos ? std::wstring(L"OptiScaler.ini") : p.substr(0, opti) + L"\\OptiScaler.ini";
    }
    return p.substr(folder + 1, slot - folder - 1);
}

// "entry=0x5747000000000103" from the one-folder-per-choice sidecar (wronggpu-entry.ini next to the DLL).
inline bool parse_sidecar_entry(const std::string& text, uint64_t* id) {
    const IniValues ini = parse_ini(text);
    std::string v;
    if (!ini.get("", "entry", &v) && !ini.get("wronggpu", "entry", &v)) return false;
    char* end = nullptr;
    const unsigned long long n = std::strtoull(trim(v).c_str(), &end, 0);
    if (end == v.c_str() || (end && *end)) return false;
    if (!entry_of_id(n)) return false;
    *id = n;
    return true;
}

// ---- Win32 part (declarations; implementation below under WG_SETTINGS_IO_IMPLEMENTATION) ---------------------------

std::wstring exe_path();
std::wstring exe_dir();
std::wstring data_dir();     // WG_DATA_DIR from the environment, else <exe dir>\WrongGPU
std::wstring ini_path();     // data_dir()\wronggpu.ini
std::wstring module_path(const void* addressInModule);
bool read_file(const std::wstring& path, std::string* out);
bool write_file_atomic(const std::wstring& path, const std::string& text);  // temp file + replace, same folder
bool file_exists(const std::wstring& path);
bool directory_exists(const std::wstring& path);
std::string narrow(const std::wstring& w);
std::wstring widen(const std::string& s);
uint64_t now_ms();
// Sets key = value in [section] of the ini file (created with the generated defaults when absent).
bool persist_ini_value(const std::wstring& path, const std::string& section, const std::string& key, const std::string& value);

class SharedSettings {
public:
    enum class Status { Closed, Absent, Created, Opened, Refused };
    SharedSettings() = default;
    ~SharedSettings();
    SharedSettings(const SharedSettings&) = delete;
    SharedSettings& operator=(const SharedSettings&) = delete;

    // Opens Local\WrongGPU.settings.v1.<pid> or creates and seeds it from `iniPath` (module = WG_SS_WRITER_* id of the
    // caller, recorded as creator). `name` overrides the mapping name (tests). Lines for the log go to *log.
    Status open(uint32_t module, const std::wstring& iniPath, std::string* log, const wchar_t* name = nullptr);
    void close();
    Status status() const { return status_; }
    bool present() const { return block_ != nullptr; }

    // Single writer: atomic compare-exchange of header.writer from NONE (or, with shareSameId, an equal id: the
    // one-folder-per-choice fallback, where several instances of the SR front end take turns). Sets `flags`.
    bool claim_writer(uint32_t id, uint32_t flags, bool shareSameId = false);
    bool is_writer() const { return writer_; }
    uint32_t writer() const;
    uint32_t flags() const;

    // Seqlock snapshot of the slots in `mask` (bounded retries); on failure the previous contents of *snap stay.
    bool snapshot(Snapshot* snap, uint64_t mask = ~0ull) const;
    // Writer only: one or several slots in one seqlock section, then (persist) wronggpu.ini. False: not the writer,
    // no block, or a value out of range.
    bool write_slot(uint32_t slot, int32_t value, bool persist = true);
    bool write_slots(const uint32_t* slots, const int32_t* values, uint32_t n, bool persist = true);
    // Text option from the block's text area (key=value, written at seeding).
    bool text(const std::string& key, std::string* out) const;
    const std::wstring& ini() const { return ini_; }
    std::string name() const;
    const WgSharedSettingsBlock* view() const { return block_; }
    WgSharedSettingsBlock* raw_for_tests() { return block_; }

private:
    void* mapping_ = nullptr;
    WgSharedSettingsBlock* block_ = nullptr;
    Status status_ = Status::Closed;
    bool writer_ = false;
    std::wstring ini_;
    std::wstring name_;
};

// Polls the host's ini for [<section>] UpscalerIndex: the file's modification time at most once per interval
// (default 1 s), re-read only when it changed. Read-only.
class HostIniPoll {
public:
    void configure(const std::wstring& path, const std::string& section, uint32_t intervalMs = 1000);
    // True when an index is present that differs from the last one returned (the first present one included).
    bool poll(int* index, uint64_t nowMs = 0);
    const std::wstring& path() const { return path_; }
    const std::string& section() const { return section_; }
    bool configured() const { return !path_.empty(); }

private:
    std::wstring path_;
    std::string section_;
    uint32_t interval_ = 1000;
    uint64_t lastCheck_ = 0;
    bool checked_ = false;
    uint64_t lastWrite_ = 0;
    int last_ = -1;
};

}  // namespace wg

// C-style names used in the design and the briefs (same functions).
inline std::string wg_describe(uint32_t slot, int32_t value, wg::Source source) { return wg::describe(slot, value, source); }
inline std::wstring wg_data_dir() { return wg::data_dir(); }

#endif  // WG_SETTINGS_IO_H

// =====================================================================================================================
#if defined(WG_SETTINGS_IO_IMPLEMENTATION) && !defined(WG_SETTINGS_IO_IMPLEMENTED)
#define WG_SETTINGS_IO_IMPLEMENTED
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

#include <cstdio>
#include <cwctype>

namespace wg {

std::string narrow(const std::wstring& w) {
    if (w.empty()) return std::string();
    const int n = WideCharToMultiByte(CP_UTF8, 0, w.c_str(), static_cast<int>(w.size()), nullptr, 0, nullptr, nullptr);
    std::string s(static_cast<size_t>(n > 0 ? n : 0), '\0');
    if (n > 0) WideCharToMultiByte(CP_UTF8, 0, w.c_str(), static_cast<int>(w.size()), &s[0], n, nullptr, nullptr);
    return s;
}

std::wstring widen(const std::string& s) {
    if (s.empty()) return std::wstring();
    const int n = MultiByteToWideChar(CP_UTF8, 0, s.c_str(), static_cast<int>(s.size()), nullptr, 0);
    std::wstring w(static_cast<size_t>(n > 0 ? n : 0), L'\0');
    if (n > 0) MultiByteToWideChar(CP_UTF8, 0, s.c_str(), static_cast<int>(s.size()), &w[0], n);
    return w;
}

static std::wstring wg_module_file(HMODULE m) {
    std::wstring buf(MAX_PATH, L'\0');
    for (;;) {
        const DWORD n = GetModuleFileNameW(m, &buf[0], static_cast<DWORD>(buf.size()));
        if (n == 0) return std::wstring();
        if (n < buf.size()) { buf.resize(n); return buf; }
        if (buf.size() >= 32768) return std::wstring();
        buf.resize(buf.size() * 2);
    }
}

std::wstring exe_path() { return wg_module_file(nullptr); }

std::wstring exe_dir() {
    const std::wstring p = exe_path();
    const size_t s = p.find_last_of(L"\\/");
    return s == std::wstring::npos ? std::wstring(L".") : p.substr(0, s);
}

std::wstring data_dir() {
    wchar_t buf[2048];
    const DWORD n = GetEnvironmentVariableW(L"WG_DATA_DIR", buf, 2048);
    if (n > 0 && n < 2048) {
        std::wstring d(buf, n);
        while (d.size() > 3 && (d.back() == L'\\' || d.back() == L'/')) d.pop_back();
        return d;
    }
    return exe_dir() + L"\\WrongGPU";
}

std::wstring ini_path() { return data_dir() + L"\\wronggpu.ini"; }

std::wstring module_path(const void* address) {
    HMODULE m = nullptr;
    if (!GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                            static_cast<LPCWSTR>(address), &m))
        return std::wstring();
    return wg_module_file(m);
}

bool file_exists(const std::wstring& path) {
    const DWORD a = GetFileAttributesW(path.c_str());
    return a != INVALID_FILE_ATTRIBUTES && !(a & FILE_ATTRIBUTE_DIRECTORY);
}

bool directory_exists(const std::wstring& path) {
    const DWORD a = GetFileAttributesW(path.c_str());
    return a != INVALID_FILE_ATTRIBUTES && (a & FILE_ATTRIBUTE_DIRECTORY);
}

bool read_file(const std::wstring& path, std::string* out) {
    HANDLE f = CreateFileW(path.c_str(), GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, nullptr,
                           OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (f == INVALID_HANDLE_VALUE) return false;
    LARGE_INTEGER size{};
    bool ok = GetFileSizeEx(f, &size) && size.QuadPart < (16 << 20);
    std::string text;
    if (ok) {
        text.resize(static_cast<size_t>(size.QuadPart));
        DWORD got = 0;
        ok = text.empty() || (ReadFile(f, &text[0], static_cast<DWORD>(text.size()), &got, nullptr) && got == text.size());
    }
    CloseHandle(f);
    if (ok && out) *out = text;
    return ok;
}

bool write_file_atomic(const std::wstring& path, const std::string& text) {
    const std::wstring tmp = path + L".tmp" + std::to_wstring(GetCurrentProcessId());
    HANDLE f = CreateFileW(tmp.c_str(), GENERIC_WRITE, 0, nullptr, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (f == INVALID_HANDLE_VALUE) return false;
    DWORD put = 0;
    const bool ok = text.empty() || (WriteFile(f, text.data(), static_cast<DWORD>(text.size()), &put, nullptr) && put == text.size());
    CloseHandle(f);
    if (!ok || !MoveFileExW(tmp.c_str(), path.c_str(), MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH)) {
        DeleteFileW(tmp.c_str());
        return false;
    }
    return true;
}

uint64_t now_ms() { return GetTickCount64(); }

bool persist_ini_value(const std::wstring& path, const std::string& section, const std::string& key, const std::string& value) {
    std::string text;
    if (!read_file(path, &text)) text = kWgDefaultIniText;
    return write_file_atomic(path, ini_set(text, section, key, value));
}

// ---- the block -----------------------------------------------------------------------------------------------------

static volatile LONG* wg_gen(WgSharedSettingsBlock* b) { return reinterpret_cast<volatile LONG*>(&b->header.generation); }

SharedSettings::~SharedSettings() { close(); }

void SharedSettings::close() {
    if (block_) UnmapViewOfFile(block_);
    if (mapping_) CloseHandle(mapping_);
    block_ = nullptr;
    mapping_ = nullptr;
    writer_ = false;
    status_ = Status::Closed;
}

std::string SharedSettings::name() const { return narrow(name_); }

SharedSettings::Status SharedSettings::open(uint32_t module, const std::wstring& iniPath, std::string* log, const wchar_t* name) {
    close();
    ini_ = iniPath;
    if (name) {
        name_ = name;
    } else {
        wchar_t buf[96];
        swprintf(buf, 96, L"Local\\WrongGPU.settings.v1.%lu", static_cast<unsigned long>(GetCurrentProcessId()));
        name_ = buf;
    }
    auto say = [log](const std::string& line) { if (log) *log += line + "\n"; };
    HANDLE m = CreateFileMappingW(INVALID_HANDLE_VALUE, nullptr, PAGE_READWRITE, 0, WG_SS_SIZE, name_.c_str());
    const DWORD err = GetLastError();
    if (!m) {
        say("settings: shared block " + narrow(name_) + " not available (error " + std::to_string(err) + "); own ini and environment only");
        status_ = Status::Absent;
        return status_;
    }
    auto* b = static_cast<WgSharedSettingsBlock*>(MapViewOfFile(m, FILE_MAP_READ | FILE_MAP_WRITE, 0, 0, WG_SS_SIZE));
    if (!b) {
        CloseHandle(m);
        say("settings: shared block " + narrow(name_) + " could not be mapped; own ini and environment only");
        status_ = Status::Absent;
        return status_;
    }
    if (err != ERROR_ALREADY_EXISTS) {
        // Creator: seed every slot from wronggpu.ini (registry defaults for absent keys, set_mask bit only for keys
        // that were in the ini), the text options, the header, then publish with magic last.
        std::string text;
        const bool haveIni = read_file(iniPath, &text);
        const IniValues ini = haveIni ? parse_ini(text) : IniValues{};
        std::string seeded;
        for (uint32_t i = 0; i < WG_OPT_COUNT; ++i) {
            const WgOptionDef& o = kWgOptions[i];
            b->values[i] = o.default_value;
            std::string v;
            int32_t x = 0;
            if (ini.get(o.ini_section, o.key, &v) && parse_value(i, v, &x)) {
                b->values[i] = x;
                b->set_mask[i >> 5] |= 1u << (i & 31);
                seeded += std::string(seeded.empty() ? "" : " ") + describe(i, x, Source::Ini);
            }
        }
        uint32_t used = 0;
        for (uint32_t i = 0; i < WG_TEXT_OPT_COUNT; ++i) {
            std::string v = kWgTextOptions[i].default_value;
            ini.get("wronggpu", kWgTextOptions[i].key, &v);
            const std::string item = std::string(kWgTextOptions[i].key) + "=" + v;
            if (used + item.size() + 1 >= WG_SS_TEXT_SIZE) break;
            std::memcpy(b->text + used, item.c_str(), item.size() + 1);
            used += static_cast<uint32_t>(item.size() + 1);
        }
        b->text_used = used;
        b->header.version = WG_SS_VERSION;
        b->header.size = WG_SS_SIZE;
        b->header.slot_count = WG_SS_SLOT_COUNT;
        b->header.generation = 0;
        b->header.writer = WG_SS_WRITER_NONE;
        b->header.registry_hash = WG_REGISTRY_HASH;
        b->header.creator = module;
        b->header.seed_source = haveIni ? 1u : 2u;
        MemoryBarrier();
        InterlockedExchange(reinterpret_cast<volatile LONG*>(&b->header.magic), static_cast<LONG>(WG_SS_MAGIC));
        say("settings: shared block " + narrow(name_) + " created, seeded from " +
            (haveIni ? narrow(iniPath) + (seeded.empty() ? std::string(" (no registry keys set)") : ": " + seeded)
                     : std::string("registry defaults (no ") + narrow(iniPath) + ")"));
        mapping_ = m;
        block_ = b;
        status_ = Status::Created;
        return status_;
    }
    // Opener: wait (bounded) for the creator to publish.
    bool published = false;
    for (int i = 0; i < 50 && !published; ++i) {
        published = static_cast<uint32_t>(InterlockedCompareExchange(reinterpret_cast<volatile LONG*>(&b->header.magic), 0, 0)) == WG_SS_MAGIC;
        if (!published) Sleep(1);
    }
    const WgSharedSettingsHeader& h = b->header;
    std::string why;
    if (!published) why = "not published by its creator within 50 ms";
    else if (h.version != WG_SS_VERSION) why = "version " + std::to_string(h.version) + " (this module knows " + std::to_string(WG_SS_VERSION) + ")";
    else if (h.size < WG_SS_HEADER_SIZE + 4u * h.slot_count || h.size > WG_SS_SIZE) why = "size " + std::to_string(h.size);
    else if (h.slot_count == 0 || h.slot_count > WG_SS_SLOT_COUNT) why = "slot_count " + std::to_string(h.slot_count);
    if (!why.empty()) {
        UnmapViewOfFile(b);
        CloseHandle(m);
        say("settings: shared block " + narrow(name_) + " ignored: " + why + "; own ini and environment only");
        status_ = Status::Refused;
        return status_;
    }
    if (h.registry_hash != WG_REGISTRY_HASH) {
        char line[160];
        std::snprintf(line, sizeof line, "settings: note: shared block registry hash 0x%08X, this module 0x%08X (ids are append-only; values used)",
                      h.registry_hash, WG_REGISTRY_HASH);
        say(line);
    }
    if (h.slot_count < WG_OPT_COUNT)
        say("settings: shared block has " + std::to_string(h.slot_count) + " slots; slots from there on use this module's defaults");
    mapping_ = m;
    block_ = b;
    status_ = Status::Opened;
    return status_;
}

uint32_t SharedSettings::writer() const {
    return block_ ? static_cast<uint32_t>(InterlockedCompareExchange(reinterpret_cast<volatile LONG*>(&block_->header.writer), 0, 0)) : 0u;
}

uint32_t SharedSettings::flags() const {
    return block_ ? static_cast<uint32_t>(InterlockedCompareExchange(reinterpret_cast<volatile LONG*>(&block_->header.flags), 0, 0)) : 0u;
}

bool SharedSettings::claim_writer(uint32_t id, uint32_t flags, bool shareSameId) {
    if (!block_ || id == WG_SS_WRITER_NONE) return false;
    const LONG was = InterlockedCompareExchange(reinterpret_cast<volatile LONG*>(&block_->header.writer), static_cast<LONG>(id),
                                                static_cast<LONG>(WG_SS_WRITER_NONE));
    if (static_cast<uint32_t>(was) == WG_SS_WRITER_NONE || (shareSameId && static_cast<uint32_t>(was) == id)) {
        InterlockedExchange(reinterpret_cast<volatile LONG*>(&block_->header.writer_pid), static_cast<LONG>(GetCurrentProcessId()));
        if (flags) InterlockedOr(reinterpret_cast<volatile LONG*>(&block_->header.flags), static_cast<LONG>(flags));
        writer_ = true;
    }
    return writer_;
}

bool SharedSettings::snapshot(Snapshot* snap, uint64_t mask) const {
    if (!block_ || !snap) return false;
    volatile LONG* gen = wg_gen(block_);
    const volatile int32_t* values = block_->values;
    for (int attempt = 0; attempt < 16; ++attempt) {
        const uint32_t g1 = static_cast<uint32_t>(InterlockedCompareExchange(gen, 0, 0));
        if (g1 & 1u) { YieldProcessor(); continue; }
        Snapshot s = *snap;
        s.slot_count = block_->header.slot_count;
        for (uint32_t i = 0; i < WG_SS_SLOT_COUNT; ++i)
            if ((mask >> i) & 1ull) s.values[i] = values[i];
        s.set_mask[0] = *reinterpret_cast<const volatile uint32_t*>(&block_->set_mask[0]);
        s.set_mask[1] = *reinterpret_cast<const volatile uint32_t*>(&block_->set_mask[1]);
        s.live_mask[0] = *reinterpret_cast<const volatile uint32_t*>(&block_->live_mask[0]);
        s.live_mask[1] = *reinterpret_cast<const volatile uint32_t*>(&block_->live_mask[1]);
        s.writer = *reinterpret_cast<const volatile uint32_t*>(&block_->header.writer);
        s.flags = *reinterpret_cast<const volatile uint32_t*>(&block_->header.flags);
        MemoryBarrier();
        const uint32_t g2 = static_cast<uint32_t>(InterlockedCompareExchange(gen, 0, 0));
        if (g1 != g2) continue;
        s.generation = g1;
        s.valid = true;
        *snap = s;
        return true;
    }
    return false;
}

bool SharedSettings::write_slots(const uint32_t* slots, const int32_t* vals, uint32_t n, bool persist) {
    if (!block_ || !writer_ || !n) return false;
    for (uint32_t i = 0; i < n; ++i)
        if (!value_in_range(slots[i], vals[i])) return false;
    volatile LONG* gen = wg_gen(block_);
    LONG g;
    for (;;) {  // enter the seqlock (CAS even -> odd: also safe for instances that share a writer id)
        g = InterlockedCompareExchange(gen, 0, 0);
        if ((g & 1) == 0 && InterlockedCompareExchange(gen, g + 1, g) == g) break;
        YieldProcessor();
    }
    volatile int32_t* values = block_->values;
    for (uint32_t i = 0; i < n; ++i) {
        values[slots[i]] = vals[i];
        InterlockedOr(reinterpret_cast<volatile LONG*>(&block_->set_mask[slots[i] >> 5]), static_cast<LONG>(1u << (slots[i] & 31)));
        InterlockedOr(reinterpret_cast<volatile LONG*>(&block_->live_mask[slots[i] >> 5]), static_cast<LONG>(1u << (slots[i] & 31)));
    }
    InterlockedExchange(gen, g + 2);
    if (persist && !ini_.empty()) {
        for (uint32_t i = 0; i < n; ++i)
            persist_ini_value(ini_, kWgOptions[slots[i]].ini_section, kWgOptions[slots[i]].key, value_name(slots[i], vals[i]));
    }
    return true;
}

bool SharedSettings::write_slot(uint32_t slot, int32_t value, bool persist) { return write_slots(&slot, &value, 1, persist); }

bool SharedSettings::text(const std::string& key, std::string* out) const {
    if (!block_) return false;
    const uint32_t used = block_->text_used < WG_SS_TEXT_SIZE ? block_->text_used : WG_SS_TEXT_SIZE;
    const std::string want = lower(key) + "=";
    for (uint32_t a = 0; a < used;) {
        const char* s = block_->text + a;
        const size_t len = strnlen(s, used - a);
        if (len >= want.size() && lower(std::string(s, want.size())) == want) {
            if (out) *out = std::string(s + want.size(), len - want.size());
            return true;
        }
        a += static_cast<uint32_t>(len + 1);
    }
    return false;
}

// ---- host ini poll ---------------------------------------------------------------------------------------------------

void HostIniPoll::configure(const std::wstring& path, const std::string& section, uint32_t intervalMs) {
    path_ = path;
    section_ = section;
    interval_ = intervalMs;
    checked_ = false;
    lastWrite_ = 0;
    last_ = -1;
}

bool HostIniPoll::poll(int* index, uint64_t nowMs) {
    if (path_.empty()) return false;
    const uint64_t now = nowMs ? nowMs : GetTickCount64();
    if (checked_ && now - lastCheck_ < interval_) return false;
    checked_ = true;
    lastCheck_ = now;
    WIN32_FILE_ATTRIBUTE_DATA a{};
    if (!GetFileAttributesExW(path_.c_str(), GetFileExInfoStandard, &a)) return false;
    const uint64_t write = (uint64_t(a.ftLastWriteTime.dwHighDateTime) << 32) | a.ftLastWriteTime.dwLowDateTime;
    const uint64_t stamp = write ^ (uint64_t(a.nFileSizeLow) << 1);
    if (stamp == lastWrite_) return false;
    lastWrite_ = stamp;
    std::string text;
    int n = -1;
    if (!read_file(path_, &text) || !parse_host_index(text, section_, &n) || n == last_) return false;
    last_ = n;
    if (index) *index = n;
    return true;
}

}  // namespace wg
#endif  // WG_SETTINGS_IO_IMPLEMENTATION
