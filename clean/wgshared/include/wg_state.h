// SPDX-License-Identifier: MIT
#pragma once
// WrongGPU's per-game state folder (0.6.5): ONE folder per game, <game folder>\wronggpu-setup (the launcher's state folder,
// beside the install record), holding everything WrongGPU's user interfaces change after the install:
//   settings.ini      settings chosen in the launcher or in a WrongGPU menu in the game (for example the RR plugin's
//                     rr_mode): they take precedence over the install's defaults in wronggpu.ini, which no WrongGPU
//                     program rewrites after the install (wronggpu.ini is the player's to edit);
//   model-choice.ini  the model choice record (clean/fsr/src/model_choice_record.h).
// The folder is always WrongGPU's: an uninstall removes these files, a repair or update keeps them. The launcher creates the
// folder; nothing here creates it.
//
// THE STATE LOCK (StateLock) is the gate for everything below: a named mutex per game folder,
//   Global\WrongGPU-state-<volume serial, 8 hex>-<file index, 16 hex>
// taken from the game folder's file identity, so every spelling of the path (case, 8.3 names, a junction or a mapped drive
// to the same folder, the \\?\ form) gives the same name. Global\: one name for every session (another account's game in
// the same library is seen too). The mutex grants SYNCHRONIZE and MUTEX_MODIFY_STATE to everyone and carries a low integrity
// label, so an elevated launcher and the game (any integrity) share it. It is a Windows mutex: owned by a THREAD, recursive
// for that thread (a nested StateLock on the same thread is taken at once).
// - Every read-modify-write of a state file holds it (SetSettings does; the model choice record's writer must too).
// - Every protected operation RE-READS the state it changes from disk after taking the lock: StateLock caches nothing, so
//   a caller that re-reads never acts on a stale view. A holder that died leaves the mutex abandoned (abandoned() says so);
//   the next taker gets it, and because writes are atomic (below) the files are whole: the old version or the new one. A
//   temporary file such a holder left is never read as state (IsTempName) and goes with the uninstall.
//
// THE LOADED PROTOCOL (no check-then-act race between a game starting and the launcher changing the game's files):
// - The game: every WrongGPU module that is loaded in the game registers BEFORE it reads or writes any state
//   (LoadedRegistration): it takes the state lock (bounded wait, kRegisterWaitMs), opens its LoadedMarker
//   (Global\WrongGPU-loaded-<volume serial>-<file index>, held OPEN for as long as the module is loaded, never owned) and
//   releases the lock. When the lock is not had in time (a long uninstall, repair or update is running) or the marker cannot
//   be made, ok() is false: the module logs it and runs with the defaults, reading and writing no state that session. It
//   never blocks the game for longer than the bounded wait.
// - The launcher: an uninstall, repair or update holds an OperationGuard for its WHOLE run: it takes the state lock, then
//   asks QueryGameUse, and refuses unless the answer is NotRunning (Running and Unknown both refuse). The lock stays held
//   until the operation ends, so no game can register in between. The guard is taken by the process that changes the
//   files (the elevated worker when there is one), never by a parent waiting for it.
// The marker object exists exactly while some process has it open; the OS closes every handle of a process that ends, crashes
// or is killed, so a dead game leaves no stale marker and no PID is ever compared.
//
// Writes are atomic (WriteAtomic): a new temporary file in the same folder (<name>.wgtmp-<pid>-<n>, created only where
// nothing is), written in checked chunks, flushed and closed (both checked), then MoveFileExW(REPLACE_EXISTING |
// WRITE_THROUGH) over the file. On a failure the temporary file is deleted when possible; a temporary file that could not be
// deleted is reported (*tempLeft). A reader sees the old file or the new one, never part of one. State files are small:
// more than kMaxStateBytes is refused.
// Paths: every helper turns its path into the \\?\ form (full path; \\?\UNC\ for a share), so a game deeper than MAX_PATH
// works without the long-path setting.
//
// settings.ini, strictly (anything else is CORRUPT: readers treat the file as empty and say so; the next write replaces it):
//   ; comment lines and blank lines
//   key=value         key: [a-z0-9_], 1..64 chars; value: [A-Za-z0-9_.+-], 0..256 chars; each key at most once;
//                     no sections, no spaces around '='; CRLF or LF line ends (a CR anywhere else, also a bare final CR,
//                     is corrupt); ASCII only.
// Written for this project; Windows API and the C++ standard library only.
#include <cstdint>
#include <memory>
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
constexpr DWORD kDefaultWaitMs = 5000;            // the launcher's waits
constexpr DWORD kRegisterWaitMs = 2000;           // the game's registration: never longer
constexpr size_t kMaxStateBytes = 1u << 20;       // no state file is larger
constexpr DWORD kChunk = 64u * 1024u;

// Test seam (tests only): called before each step of WriteAtomic ("write", "flush", "close", "move") and of its cleanup
// ("delete"); returning true makes that step fail as a failed call would (error ERROR_WRITE_FAULT). nullptr: never.
using FaultHook = bool (*)(const char* step);
inline FaultHook& TestFault() { static FaultHook hook = nullptr; return hook; }
inline bool Faulted(const char* step) { return TestFault() && TestFault()(step); }

inline std::wstring JoinPath(std::wstring a, const std::wstring& b) {
    if (!a.empty() && a.back() != L'\\' && a.back() != L'/') a.push_back(L'\\');
    return a + b;
}
// The \\?\ form of a path (full path; \\?\UNC\server\share\... for a share; a path already in that form stays).
inline std::wstring ExtendedPath(const std::wstring& path) {
    std::wstring p = path;
    for (auto& c : p) if (c == L'/') c = L'\\';
    if (p.rfind(L"\\\\?\\", 0) == 0 || p.rfind(L"\\\\.\\", 0) == 0) return p;
    const DWORD n = GetFullPathNameW(p.c_str(), 0, nullptr, nullptr);
    if (n) {
        std::wstring full(n, L'\0');
        const DWORD m = GetFullPathNameW(p.c_str(), n, &full[0], nullptr);
        if (m && m < n) { full.resize(m); p = full; }
    }
    if (p.rfind(L"\\\\", 0) == 0) return L"\\\\?\\UNC\\" + p.substr(2);
    return L"\\\\?\\" + p;
}
inline std::wstring StateDirOf(const std::wstring& gameDir) { return JoinPath(gameDir, kStateFolder); }
inline std::wstring SettingsPathOf(const std::wstring& gameDir) { return JoinPath(StateDirOf(gameDir), kSettingsFile); }
// A temporary file of WriteAtomic (its own name contains the mark; the state files' names never do).
inline bool IsTempName(const std::wstring& name) { return name.find(kTempMark) != std::wstring::npos; }

enum class PathState { Present, Missing, Unknown };  // Unknown: it could not be told (access denied, a device error)
inline PathState DirectoryState(const std::wstring& path, DWORD* err = nullptr) {
    const DWORD a = GetFileAttributesW(ExtendedPath(path).c_str());
    if (a != INVALID_FILE_ATTRIBUTES) return (a & FILE_ATTRIBUTE_DIRECTORY) ? PathState::Present : PathState::Missing;
    const DWORD e = GetLastError();
    if (err) *err = e;
    return e == ERROR_FILE_NOT_FOUND || e == ERROR_PATH_NOT_FOUND ? PathState::Missing : PathState::Unknown;
}

// The state lock's name for the game folder `gameDir`; false (and *err) when the folder cannot be opened or identified.
inline bool LockNameFor(const std::wstring& gameDir, std::wstring* name, DWORD* err = nullptr) {
    HANDLE h = CreateFileW(ExtendedPath(gameDir).c_str(), FILE_READ_ATTRIBUTES, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, nullptr,
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
    swprintf(buf, sizeof buf / sizeof buf[0], L"Global\\WrongGPU-state-%08lX-%016llX", static_cast<unsigned long>(info.dwVolumeSerialNumber), index);
    *name = buf;
    return true;
}
// Another per-game name of the same identity: prefix + "<volume serial>-<file index>".
inline bool NameFor(const wchar_t* prefix, const std::wstring& gameDir, std::wstring* name, DWORD* err) {
    std::wstring lock;
    if (!LockNameFor(gameDir, &lock, err)) return false;
    *name = std::wstring(prefix) + lock.substr(lock.find(L"-state-") + 7);
    return true;
}
// A named mutex every integrity level may open (created, or opened when it exists); nullptr with GetLastError on failure.
inline HANDLE CreateSharedMutex(const std::wstring& name) {
    PSECURITY_DESCRIPTOR sd = nullptr;
    SECURITY_ATTRIBUTES sa{sizeof(sa), nullptr, FALSE};
    // everyone: SYNCHRONIZE | MUTEX_MODIFY_STATE; SYSTEM and administrators: all; low integrity label (no write-up block)
    if (ConvertStringSecurityDescriptorToSecurityDescriptorW(L"D:(A;;0x00100001;;;WD)(A;;GA;;;SY)(A;;GA;;;BA)S:(ML;;NW;;;LW)", SDDL_REVISION_1, &sd, nullptr))
        sa.lpSecurityDescriptor = sd;
    HANDLE m = CreateMutexW(sa.lpSecurityDescriptor ? &sa : nullptr, FALSE, name.c_str());
    if (!m && GetLastError() == ERROR_ACCESS_DENIED) m = OpenMutexW(SYNCHRONIZE | MUTEX_MODIFY_STATE, FALSE, name.c_str());
    const DWORD e = GetLastError();
    if (sd) LocalFree(sd);
    SetLastError(e);
    return m;
}

// The game's state lock, held from construction (when held()) to destruction. Bounded wait: not held after waitMs.
class StateLock {
public:
    explicit StateLock(const std::wstring& gameDir, DWORD waitMs = kDefaultWaitMs) {
        std::wstring name;
        if (!LockNameFor(gameDir, &name, &err_)) return;
        mutex_ = CreateSharedMutex(name);
        if (!mutex_) { err_ = GetLastError(); return; }
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
    // the previous holder ended without releasing it: informational (re-read the state from disk, as every holder must)
    bool abandoned() const { return abandoned_; }
    DWORD error() const { return err_; }  // why not held: WAIT_TIMEOUT, or the Win32 error

private:
    HANDLE mutex_ = nullptr;
    bool held_ = false, abandoned_ = false;
    DWORD err_ = 0;
};

// ---- the loaded protocol ------------------------------------------------------------------------------------------
constexpr const wchar_t* kLoadedPrefix = L"Global\\WrongGPU-loaded-";
// The marker alone (held open, never owned). Use it through LoadedRegistration, which takes the state lock around it.
class LoadedMarker {
public:
    explicit LoadedMarker(const std::wstring& gameDir) {
        std::wstring name;
        if (!NameFor(kLoadedPrefix, gameDir, &name, &err_)) return;
        handle_ = CreateSharedMutex(name);
        if (!handle_) err_ = GetLastError();
    }
    ~LoadedMarker() { if (handle_) CloseHandle(handle_); }
    LoadedMarker(const LoadedMarker&) = delete;
    LoadedMarker& operator=(const LoadedMarker&) = delete;
    bool held() const { return handle_ != nullptr; }
    DWORD error() const { return err_; }

private:
    HANDLE handle_ = nullptr;
    DWORD err_ = 0;
};

enum class GameUse { NotRunning, Running, Unknown };
inline const char* GameUseWords(GameUse u) {
    return u == GameUse::NotRunning ? "not running" : u == GameUse::Running ? "the game is running with WrongGPU loaded"
                                                                            : "WrongGPU could not tell whether the game is running";
}
// Whether a running process has WrongGPU loaded for the game folder gameDir. NotRunning ONLY when that is established: the
// folder is gone (nothing can run from it), or its marker does not exist. Any error on the way is Unknown (*err). Never
// creates the marker. Ask it under the state lock (OperationGuard does).
inline GameUse QueryGameUse(const std::wstring& gameDir, DWORD* err = nullptr) {
    std::wstring name;
    DWORD e = 0;
    if (!NameFor(kLoadedPrefix, gameDir, &name, &e)) {
        if (err) *err = e;
        return e == ERROR_FILE_NOT_FOUND || e == ERROR_PATH_NOT_FOUND ? GameUse::NotRunning : GameUse::Unknown;
    }
    HANDLE h = OpenMutexW(SYNCHRONIZE, FALSE, name.c_str());
    if (h) { CloseHandle(h); return GameUse::Running; }
    e = GetLastError();
    if (err) *err = e;
    if (e == ERROR_FILE_NOT_FOUND) return GameUse::NotRunning;
    return e == ERROR_ACCESS_DENIED ? GameUse::Running : GameUse::Unknown;  // there but not ours to open: in use
}

// The game's side: register under the state lock before any state is read. ok() false: run with the defaults and touch no
// state this session (why() says what to log). Keep the object for as long as the module is loaded.
class LoadedRegistration {
public:
    explicit LoadedRegistration(const std::wstring& gameDir, DWORD waitMs = kRegisterWaitMs) {
        StateLock lock(gameDir, waitMs > kRegisterWaitMs ? kRegisterWaitMs : waitMs);
        if (!lock.held()) {
            err_ = lock.error();
            why_ = err_ == WAIT_TIMEOUT ? "the game's state lock was not free in time (an uninstall, repair or update of WrongGPU is running)"
                                        : "the game's state lock could not be taken";
            return;
        }
        marker_.reset(new LoadedMarker(gameDir));
        if (!marker_->held()) {
            err_ = marker_->error();
            why_ = "the loaded marker could not be made";
            marker_.reset();
        }
    }
    bool ok() const { return marker_ != nullptr; }
    const char* why() const { return why_; }  // "" when ok
    DWORD error() const { return err_; }

private:
    std::unique_ptr<LoadedMarker> marker_;
    const char* why_ = "";
    DWORD err_ = 0;
};

// The launcher's side: hold for the whole uninstall, repair or update. Ok only when the lock is held and no game uses WrongGPU.
enum class GuardResult { Ok, Busy, GameRunning, Unknown };
class OperationGuard {
public:
    explicit OperationGuard(const std::wstring& gameDir, DWORD waitMs = kDefaultWaitMs) : lock_(gameDir, waitMs) {
        if (!lock_.held()) {
            err_ = lock_.error();
            const PathState s = DirectoryState(gameDir);
            result_ = s == PathState::Missing ? GuardResult::Ok : err_ == WAIT_TIMEOUT ? GuardResult::Busy : GuardResult::Unknown;  // (gone: nothing to guard)
            return;
        }
        const GameUse u = QueryGameUse(gameDir, &err_);
        result_ = u == GameUse::NotRunning ? GuardResult::Ok : u == GameUse::Running ? GuardResult::GameRunning : GuardResult::Unknown;
    }
    GuardResult result() const { return result_; }
    DWORD error() const { return err_; }
    const char* words() const {
        switch (result_) {
            case GuardResult::Ok: return "";
            case GuardResult::Busy: return "another WrongGPU program is changing this game's WrongGPU state right now; try again in a moment";
            case GuardResult::GameRunning: return "the game is running with WrongGPU loaded; close the game first";
            case GuardResult::Unknown: return "WrongGPU could not tell whether the game is running; close the game and try again";
        }
        return "?";
    }

private:
    StateLock lock_;
    GuardResult result_ = GuardResult::Unknown;
    DWORD err_ = 0;
};

// ---- files ---------------------------------------------------------------------------------------------------------
// The whole file. false: missing (*missing = true) or unreadable (*missing = false; *err the Win32 error).
inline bool ReadBytes(const std::wstring& path, std::string* out, bool* missing = nullptr, DWORD* err = nullptr) {
    out->clear();
    if (missing) *missing = false;
    HANDLE h = CreateFileW(ExtendedPath(path).c_str(), GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, nullptr, OPEN_EXISTING,
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
        if (out->size() > kMaxStateBytes) { ok = false; if (err) *err = ERROR_FILE_TOO_LARGE; break; }
    }
    CloseHandle(h);
    if (!ok) out->clear();
    return ok;
}

// The atomic replace described above. false: the file is as it was; *err the error of the step that failed
// (ERROR_FILE_TOO_LARGE: over kMaxStateBytes; ERROR_WRITE_FAULT: a short write). *tempLeft: a temporary file that could not
// be deleted after the failure ("" when none).
inline bool WriteAtomic(const std::wstring& path, const std::string& bytes, DWORD* err = nullptr, std::wstring* tempLeft = nullptr) {
    static volatile LONG counter = 0;
    if (tempLeft) tempLeft->clear();
    if (bytes.size() > kMaxStateBytes) { if (err) *err = ERROR_FILE_TOO_LARGE; return false; }
    const std::wstring target = ExtendedPath(path);
    DWORD e = 0;
    for (int attempt = 0; attempt < 16; ++attempt) {
        const std::wstring tmp = target + kTempMark + std::to_wstring(GetCurrentProcessId()) + L"-" + std::to_wstring(InterlockedIncrement(&counter));
        HANDLE h = CreateFileW(tmp.c_str(), GENERIC_WRITE, 0, nullptr, CREATE_NEW, FILE_ATTRIBUTE_NORMAL, nullptr);
        if (h == INVALID_HANDLE_VALUE) {
            e = GetLastError();
            if (e == ERROR_FILE_EXISTS || e == ERROR_ALREADY_EXISTS) continue;
            break;
        }
        bool ok = true;
        for (size_t done = 0; ok && done < bytes.size();) {
            const DWORD want = static_cast<DWORD>(bytes.size() - done < kChunk ? bytes.size() - done : kChunk);
            DWORD put = 0;
            if (Faulted("write")) { ok = false; e = ERROR_WRITE_FAULT; }
            else if (!WriteFile(h, bytes.data() + done, want, &put, nullptr)) { ok = false; e = GetLastError(); }
            else if (put != want) { ok = false; e = ERROR_WRITE_FAULT; }  // a short write: always this error
            else done += put;
        }
        if (ok) {
            if (Faulted("flush")) { ok = false; e = ERROR_WRITE_FAULT; }
            else if (!FlushFileBuffers(h)) { ok = false; e = GetLastError(); }
        }
        if (Faulted("close")) {
            CloseHandle(h);  // (the handle is closed all the same; the step reports the failure)
            if (ok) { ok = false; e = ERROR_WRITE_FAULT; }
        } else if (!CloseHandle(h) && ok) {
            ok = false;
            e = GetLastError();
        }
        if (ok) {
            if (Faulted("move")) { ok = false; e = ERROR_WRITE_FAULT; }
            else if (!MoveFileExW(tmp.c_str(), target.c_str(), MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH)) { ok = false; e = GetLastError(); }
        }
        if (ok) return true;
        if (Faulted("delete") || !DeleteFileW(tmp.c_str())) {
            if (tempLeft) *tempLeft = tmp;  // deleted when possible; reported otherwise
        }
        break;
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
    auto corrupt = [&](int lineNo, const char* what) {
        s.kv.clear();
        s.corrupt = true;
        s.why = "line " + std::to_string(lineNo) + ": " + what;
        return s;
    };
    size_t a = 0;
    int lineNo = 0;
    while (a < text.size()) {
        size_t e = text.find('\n', a);
        const bool lf = e != std::string::npos;
        if (!lf) e = text.size();
        std::string line = text.substr(a, e - a);
        a = e + 1;
        ++lineNo;
        if (lf && !line.empty() && line.back() == '\r') line.pop_back();  // CRLF; a CR anywhere else is not a line end
        for (char c : line)
            if (static_cast<unsigned char>(c) >= 0x80 || (c < 0x20 && c != '\t')) return corrupt(lineNo, c == '\r' ? "a CR that is not part of CRLF" : "not plain text");
        if (line.empty() || line[0] == ';') continue;
        const size_t eq = line.find('=');
        const std::string k = eq == std::string::npos ? line : line.substr(0, eq), v = eq == std::string::npos ? std::string() : line.substr(eq + 1);
        if (eq == std::string::npos) return corrupt(lineNo, "no key=value");
        if (!ValidKey(k)) return corrupt(lineNo, "not a valid key");
        if (!ValidValue(v)) return corrupt(lineNo, "not a valid value");
        if (Get(s, k)) return corrupt(lineNo, "the key appears twice");
        s.kv.emplace_back(k, v);
    }
    return s;
}
inline std::string RenderSettings(const Settings& s) {
    std::string out = "; WrongGPU: settings chosen in the launcher or in a WrongGPU menu in the game (they take precedence over wronggpu.ini)\r\n";
    for (const auto& p : s.kv) out += p.first + "=" + p.second + "\r\n";
    return out;
}
// settings.ini of the game in gameDir, read now from disk. Missing: present = false, no keys. Unreadable or corrupt: no keys,
// why says so.
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

enum class SetResult { Ok, NoStateFolder, StateFolderUnknown, Busy, Unreadable, Invalid, AccessDenied, WriteFailed };
inline const char* SetResultWords(SetResult r) {
    switch (r) {
        case SetResult::Ok: return "saved";
        case SetResult::NoStateFolder: return "no WrongGPU state folder for this game (not installed by the launcher)";
        case SetResult::StateFolderUnknown: return "WrongGPU's state folder of this game could not be checked";
        case SetResult::Busy: return "another WrongGPU program kept the game's state lock";
        case SetResult::Unreadable: return "the settings file could not be read (left as it is)";
        case SetResult::Invalid: return "not a valid key or value";
        case SetResult::AccessDenied: return "WrongGPU can't write to the game folder; run the launcher's repair as administrator";
        case SetResult::WriteFailed: return "the settings file could not be written";
    }
    return "?";
}
// The read-modify-write of settings.ini under the game's state lock: the file re-read from disk after the lock is taken, the
// changes applied (a corrupt file starts empty; an unreadable one is never overwritten), written atomically. The state folder
// is never created here. AccessDenied is its own result (the game folder is not writable for this process): consumers say
// so and offer the launcher's elevated repair. *err: the Win32 error of a failure; *tempLeft as in WriteAtomic.
inline SetResult SetSettings(const std::wstring& gameDir, const std::vector<std::pair<std::string, std::string>>& changes,
                             DWORD waitMs = kDefaultWaitMs, DWORD* err = nullptr, std::wstring* tempLeft = nullptr) {
    for (const auto& c : changes) if (!ValidKey(c.first) || !ValidValue(c.second)) return SetResult::Invalid;
    DWORD e = 0;
    switch (DirectoryState(StateDirOf(gameDir), &e)) {
        case PathState::Present: break;
        case PathState::Missing: return SetResult::NoStateFolder;
        case PathState::Unknown: if (err) *err = e; return e == ERROR_ACCESS_DENIED ? SetResult::AccessDenied : SetResult::StateFolderUnknown;
    }
    StateLock lock(gameDir, waitMs);
    if (!lock.held()) { if (err) *err = lock.error(); return SetResult::Busy; }
    std::string text;
    bool missing = false;
    Settings s;
    if (ReadBytes(SettingsPathOf(gameDir), &text, &missing, &e)) s = ParseSettings(text);
    else if (!missing) { if (err) *err = e; return e == ERROR_ACCESS_DENIED ? SetResult::AccessDenied : SetResult::Unreadable; }
    if (s.corrupt) s = Settings();
    bool changed = !s.present;
    for (const auto& c : changes) {
        const std::string* now = Get(s, c.first);
        if (!now || *now != c.second) { Set(s, c.first, c.second); changed = true; }
    }
    if (!changed) return SetResult::Ok;
    if (!WriteAtomic(SettingsPathOf(gameDir), RenderSettings(s), &e, tempLeft)) {
        if (err) *err = e;
        return e == ERROR_ACCESS_DENIED ? SetResult::AccessDenied : SetResult::WriteFailed;
    }
    return SetResult::Ok;
}

}  // namespace wgstate
