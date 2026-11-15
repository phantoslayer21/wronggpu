// SPDX-License-Identifier: MIT
#pragma once
// WrongGPU's per-game state folder (0.6.5): ONE folder per game, <game folder>\wronggpu-setup (the launcher's state folder,
// beside the install record), holding everything WrongGPU's user interfaces change after the install:
//   settings.ini      settings chosen in the launcher or in a WrongGPU menu in the game (for example the RR plugin's
//                     rr_mode): they take precedence over the install's defaults in wronggpu.ini, which no WrongGPU
//                     program rewrites after the install (wronggpu.ini is the player's to edit);
//   model-choice.ini  the model choice record (clean/fsr/src/model_choice_record.h).
// The folder is always WrongGPU's: an uninstall removes these files, a repair or update keeps them.
//
// Every read-modify-write of a file in the folder, by any WrongGPU program (the launcher, the elevated worker, the front
// end, the menu plugin, the RR plugin), holds the game's state lock (StateLock): a named mutex per game folder,
//   Local\WrongGPU-state-<volume serial, 8 hex>-<file index, 16 hex>
// taken from the game folder's file identity, so every spelling of the path (case, 8.3 names, a mapped drive or a
// junction to the same folder) gives the same name. The mutex grants SYNCHRONIZE and MUTEX_MODIFY_STATE to everyone and
// carries a low integrity label, so an elevated launcher and the game (any integrity) share it. A holder that dies leaves
// the mutex abandoned: the next waiter gets it (the files are replaced atomically, so they are whole either way).
//
// Writes are atomic (WriteAtomic): a new temporary file in the same folder (<name>.wgtmp-<pid>-<n>, created only where
// nothing is), flushed, then MoveFileExW(REPLACE_EXISTING | WRITE_THROUGH) over the file; the temporary file is deleted when
// any step fails. A reader sees the old file or the new one, never part of one. Temporary files never count as state.
//
// settings.ini, strictly (anything else is CORRUPT: readers treat the file as empty and say so; the next write replaces it):
//   ; comment lines and blank lines
//   key=value         key: [a-z0-9_], 1..64 chars; value: [A-Za-z0-9_.+-], 0..256 chars; each key at most once;
//                     no sections, no spaces around '='; CRLF or LF line ends; UTF-8 without anything outside ASCII.
// Written for this project; Windows API and the C++ standard library only.
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <sddl.h>
#ifdef _MSC_VER
#pragma comment(lib, "advapi32.lib")
#endif

namespace wgstate {

constexpr const wchar_t* kStateFolder = L"wronggpu-setup";
constexpr const wchar_t* kSettingsFile = L"settings.ini";
constexpr const wchar_t* kModelChoiceFile = L"model-choice.ini";
constexpr const wchar_t* kTempMark = L".wgtmp-";  // <file>.wgtmp-<pid>-<n>: a write in progress (or one a crash left)
constexpr DWORD kDefaultWaitMs = 5000;

inline std::wstring JoinPath(std::wstring a, const std::wstring& b) {
    if (!a.empty() && a.back() != L'\\' && a.back() != L'/') a.push_back(L'\\');
    return a + b;
}
inline std::wstring StateDirOf(const std::wstring& gameDir) { return JoinPath(gameDir, kStateFolder); }
inline std::wstring SettingsPathOf(const std::wstring& gameDir) { return JoinPath(StateDirOf(gameDir), kSettingsFile); }
inline bool IsDirectory(const std::wstring& p) {
    const DWORD a = GetFileAttributesW(p.c_str());
    return a != INVALID_FILE_ATTRIBUTES && (a & FILE_ATTRIBUTE_DIRECTORY);
}
// A temporary file of WriteAtomic (its own name contains the mark; the state files' names never do).
inline bool IsTempName(const std::wstring& name) { return name.find(kTempMark) != std::wstring::npos; }

// The state lock's name for the game folder `gameDir`; false (and *err) when the folder cannot be opened.
inline bool LockNameFor(const std::wstring& gameDir, std::wstring* name, DWORD* err = nullptr) {
    HANDLE h = CreateFileW(gameDir.c_str(), FILE_READ_ATTRIBUTES, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, nullptr,
                           OPEN_EXISTING, FILE_FLAG_BACKUP_SEMANTICS, nullptr);
    if (h == INVALID_HANDLE_VALUE) {
        if (err) *err = GetLastError();
        return false;
    }
    BY_HANDLE_FILE_INFORMATION info{};
    const BOOL ok = GetFileInformationByHandle(h, &info);
    const DWORD e = GetLastError();
    CloseHandle(h);
    if (!ok) {
        if (err) *err = e;
        return false;
    }
    wchar_t buf[96];
    const unsigned long long index = (static_cast<unsigned long long>(info.nFileIndexHigh) << 32) | info.nFileIndexLow;
    swprintf(buf, sizeof buf / sizeof buf[0], L"Local\\WrongGPU-state-%08lX-%016llX", static_cast<unsigned long>(info.dwVolumeSerialNumber), index);
    *name = buf;
    return true;
}

// The game's state lock, held from construction (when held()) to destruction. Bounded wait: not held after waitMs.
class StateLock {
public:
    explicit StateLock(const std::wstring& gameDir, DWORD waitMs = kDefaultWaitMs) {
        std::wstring name;
        if (!LockNameFor(gameDir, &name, &err_)) return;
        PSECURITY_DESCRIPTOR sd = nullptr;
        SECURITY_ATTRIBUTES sa{sizeof(sa), nullptr, FALSE};
        // everyone: SYNCHRONIZE | MUTEX_MODIFY_STATE; SYSTEM and administrators: all; low integrity label (no write-up block)
        if (ConvertStringSecurityDescriptorToSecurityDescriptorW(L"D:(A;;0x00100001;;;WD)(A;;GA;;;SY)(A;;GA;;;BA)S:(ML;;NW;;;LW)",
                                                                 SDDL_REVISION_1, &sd, nullptr))
            sa.lpSecurityDescriptor = sd;
        mutex_ = CreateMutexW(sa.lpSecurityDescriptor ? &sa : nullptr, FALSE, name.c_str());
        if (!mutex_ && GetLastError() == ERROR_ACCESS_DENIED) mutex_ = OpenMutexW(SYNCHRONIZE | MUTEX_MODIFY_STATE, FALSE, name.c_str());
        if (!mutex_) err_ = GetLastError();
        if (sd) LocalFree(sd);
        if (!mutex_) return;
        const DWORD w = WaitForSingleObject(mutex_, waitMs);
        held_ = w == WAIT_OBJECT_0 || w == WAIT_ABANDONED;
        abandoned_ = w == WAIT_ABANDONED;
        if (!held_) err_ = w == WAIT_TIMEOUT ? WAIT_TIMEOUT : GetLastError();
    }
    ~StateLock() {
        if (held_) ReleaseMutex(mutex_);
        if (mutex_) CloseHandle(mutex_);
    }
    StateLock(const StateLock&) = delete;
    StateLock& operator=(const StateLock&) = delete;
    bool held() const { return held_; }
    bool abandoned() const { return abandoned_; }  // the previous holder ended without releasing it
    DWORD error() const { return err_; }           // why not held: WAIT_TIMEOUT, or the Win32 error

private:
    HANDLE mutex_ = nullptr;
    bool held_ = false, abandoned_ = false;
    DWORD err_ = 0;
};

// The whole file. false: missing (*missing = true) or unreadable (*missing = false; *err the Win32 error).
inline bool ReadBytes(const std::wstring& path, std::string* out, bool* missing = nullptr, DWORD* err = nullptr) {
    out->clear();
    if (missing) *missing = false;
    HANDLE h = CreateFileW(path.c_str(), GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, nullptr, OPEN_EXISTING,
                           FILE_ATTRIBUTE_NORMAL, nullptr);
    if (h == INVALID_HANDLE_VALUE) {
        const DWORD e = GetLastError();
        if (missing) *missing = e == ERROR_FILE_NOT_FOUND || e == ERROR_PATH_NOT_FOUND;
        if (err) *err = e;
        return false;
    }
    char buf[4096];
    DWORD got = 0;
    bool ok = true;
    for (;;) {
        if (!ReadFile(h, buf, sizeof buf, &got, nullptr)) { ok = false; if (err) *err = GetLastError(); break; }
        if (!got) break;
        out->append(buf, got);
        if (out->size() > (1u << 20)) { ok = false; if (err) *err = ERROR_FILE_TOO_LARGE; break; }  // no state file is that large
    }
    CloseHandle(h);
    if (!ok) out->clear();
    return ok;
}

// The atomic replace described above. false (and *err) when any step failed; the file is then as it was.
inline bool WriteAtomic(const std::wstring& path, const std::string& bytes, DWORD* err = nullptr) {
    static volatile LONG counter = 0;
    DWORD e = 0;
    for (int attempt = 0; attempt < 16; ++attempt) {
        const std::wstring tmp = path + kTempMark + std::to_wstring(GetCurrentProcessId()) + L"-" + std::to_wstring(InterlockedIncrement(&counter));
        HANDLE h = CreateFileW(tmp.c_str(), GENERIC_WRITE, 0, nullptr, CREATE_NEW, FILE_ATTRIBUTE_NORMAL, nullptr);
        if (h == INVALID_HANDLE_VALUE) {
            e = GetLastError();
            if (e == ERROR_FILE_EXISTS || e == ERROR_ALREADY_EXISTS) continue;
            break;
        }
        DWORD put = 0;
        bool ok = bytes.empty() || (WriteFile(h, bytes.data(), static_cast<DWORD>(bytes.size()), &put, nullptr) && put == bytes.size());
        if (!ok) e = GetLastError() ? GetLastError() : ERROR_WRITE_FAULT;
        if (ok && !FlushFileBuffers(h)) { ok = false; e = GetLastError(); }
        CloseHandle(h);
        if (ok && !MoveFileExW(tmp.c_str(), path.c_str(), MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH)) { ok = false; e = GetLastError(); }
        if (!ok) {
            DeleteFileW(tmp.c_str());
            break;
        }
        return true;
    }
    if (err) *err = e ? e : ERROR_FILE_EXISTS;
    return false;
}

// ---- settings.ini -------------------------------------------------------------------------------------------------
struct Settings {
    std::vector<std::pair<std::string, std::string>> kv;  // in file order
    bool present = false;                                 // the file was there and read
    bool corrupt = false;                                 // it was there and not in the format above (kv empty)
    std::string why;                                      // what was wrong (corrupt), or why it could not be read
};
inline bool KeyChar(char c) { return (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || c == '_'; }
inline bool ValueChar(char c) {
    return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '_' || c == '.' || c == '+' || c == '-';
}
inline bool ValidKey(const std::string& k) {
    if (k.empty() || k.size() > 64) return false;
    for (char c : k) if (!KeyChar(c)) return false;
    return true;
}
inline bool ValidValue(const std::string& v) {
    if (v.size() > 256) return false;
    for (char c : v) if (!ValueChar(c)) return false;
    return true;
}
inline const std::string* Get(const Settings& s, const std::string& key) {
    for (const auto& p : s.kv) if (p.first == key) return &p.second;
    return nullptr;
}
// Sets key=value (added at the end when new). false: not a valid key or value (nothing changed).
inline bool Set(Settings& s, const std::string& key, const std::string& value) {
    if (!ValidKey(key) || !ValidValue(value)) return false;
    for (auto& p : s.kv) if (p.first == key) { p.second = value; return true; }
    s.kv.emplace_back(key, value);
    return true;
}
// Parses settings.ini text strictly; a text not in the format gives corrupt = true and no keys.
inline Settings ParseSettings(const std::string& text) {
    Settings s;
    s.present = true;
    size_t a = 0;
    int lineNo = 0;
    while (a < text.size()) {
        size_t e = text.find('\n', a);
        if (e == std::string::npos) e = text.size();
        std::string line = text.substr(a, e - a);
        a = e + 1;
        ++lineNo;
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty() || line[0] == ';') {
            for (char c : line) if (static_cast<unsigned char>(c) >= 0x80 || (c < 0x20 && c != '\t')) { s.kv.clear(); s.corrupt = true; s.why = "line " + std::to_string(lineNo) + ": not plain text"; return s; }
            continue;
        }
        const size_t eq = line.find('=');
        const std::string k = eq == std::string::npos ? line : line.substr(0, eq), v = eq == std::string::npos ? std::string() : line.substr(eq + 1);
        if (eq == std::string::npos || !ValidKey(k) || !ValidValue(v) || Get(s, k)) {
            s.kv.clear();
            s.corrupt = true;
            s.why = "line " + std::to_string(lineNo) + (eq == std::string::npos ? ": no key=value" : !ValidKey(k) ? ": not a valid key" : !ValidValue(v) ? ": not a valid value" : ": the key appears twice");
            return s;
        }
        s.kv.emplace_back(k, v);
    }
    return s;
}
inline std::string RenderSettings(const Settings& s) {
    std::string out = "; WrongGPU: settings chosen in the launcher or in a WrongGPU menu in the game (they take precedence over wronggpu.ini)\r\n";
    for (const auto& p : s.kv) out += p.first + "=" + p.second + "\r\n";
    return out;
}
// settings.ini of the game in gameDir. Missing: present = false, no keys. Unreadable or corrupt: no keys, why says so.
inline Settings ReadSettings(const std::wstring& gameDir) {
    std::string text;
    bool missing = false;
    DWORD err = 0;
    if (!ReadBytes(SettingsPathOf(gameDir), &text, &missing, &err)) {
        Settings s;
        if (!missing) s.why = "unreadable (error " + std::to_string(err) + ")";
        return s;
    }
    return ParseSettings(text);
}

enum class SetResult { Ok, NoStateFolder, Busy, Unreadable, Invalid, WriteFailed };
inline const char* SetResultWords(SetResult r) {
    switch (r) {
        case SetResult::Ok: return "saved";
        case SetResult::NoStateFolder: return "no WrongGPU state folder for this game (not installed by the launcher)";
        case SetResult::Busy: return "another WrongGPU program kept the game's state lock";
        case SetResult::Unreadable: return "the settings file could not be read (left as it is)";
        case SetResult::Invalid: return "not a valid key or value";
        case SetResult::WriteFailed: return "the settings file could not be written";
    }
    return "?";
}
// The read-modify-write of settings.ini under the game's state lock: the changes applied to the file as it is now (a
// corrupt file starts empty; an unreadable one is never overwritten). The state folder is never created here: without it
// the game has no WrongGPU install of the launcher (NoStateFolder). *err: the Win32 error of a failure.
inline SetResult SetSettings(const std::wstring& gameDir, const std::vector<std::pair<std::string, std::string>>& changes,
                             DWORD waitMs = kDefaultWaitMs, DWORD* err = nullptr) {
    for (const auto& c : changes) if (!ValidKey(c.first) || !ValidValue(c.second)) return SetResult::Invalid;
    if (!IsDirectory(StateDirOf(gameDir))) return SetResult::NoStateFolder;
    StateLock lock(gameDir, waitMs);
    if (!lock.held()) { if (err) *err = lock.error(); return SetResult::Busy; }
    std::string text;
    bool missing = false;
    DWORD e = 0;
    Settings s;
    if (ReadBytes(SettingsPathOf(gameDir), &text, &missing, &e)) s = ParseSettings(text);
    else if (!missing) { if (err) *err = e; return SetResult::Unreadable; }
    if (s.corrupt) s = Settings();
    bool changed = !s.present;
    for (const auto& c : changes) {
        const std::string* now = Get(s, c.first);
        if (!now || *now != c.second) { Set(s, c.first, c.second); changed = true; }
    }
    if (!changed) return SetResult::Ok;
    if (!WriteAtomic(SettingsPathOf(gameDir), RenderSettings(s), &e)) { if (err) *err = e; return SetResult::WriteFailed; }
    return SetResult::Ok;
}

}  // namespace wgstate
