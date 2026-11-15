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
// THE OPERATION GUARD (a game that uses WrongGPU is never changed under it; Windows itself decides, nothing is registered):
// - The game registers nothing. A WrongGPU binary the game has loaded (the front end, the menu plugin, the RR plugin, the
//   runtime) is an image section of that process, and Windows refuses an exclusive open of it.
// - The launcher: an uninstall, repair or update holds an OperationGuard for its WHOLE run. It takes the state lock (a timeout
//   is Busy, access denied is LockDenied), then opens EVERY WrongGPU binary of the install with DELETE | GENERIC_READ |
//   FILE_WRITE_DATA and share mode 0 (a loaded image has no open handle, so share mode 0 alone does not see it; Windows refuses
//   an open for writing its data; nothing is ever written through the handle; a read-only binary is therefore Unknown):
//   ERROR_SHARING_VIOLATION = a process has it loaded (GameRunning: refuse); any other open error = Unknown (refuse); a
//   file that does not exist is skipped; all opened = nobody has WrongGPU loaded. The guard HOLDS those exclusive handles to
//   the end, so a game that starts meanwhile cannot load WrongGPU (its LoadLibrary fails; OptiScaler carries on without it).
//   The operation deletes a held binary through the guard's own handle (DeleteHeld), or releases that one handle right at
//   the step that replaces the file (Release). The guard is taken by the process that changes the files (the elevated worker
//   when there is one), never by a parent waiting for it.
//
// Writes are atomic (WriteAtomic): a new temporary file in the same folder (<name>.wgtmp-<pid>-<n>, created only where
// nothing is), written in checked chunks, flushed and closed (both checked), then MoveFileExW(REPLACE_EXISTING |
// WRITE_THROUGH) over the file. A reader sees the old file or the new one, never part of one. On a failure the temporary file
// is deleted when possible; the result (WriteResult, and SetOutcome for SetSettings) carries the first failure AND every
// secondary one: a close that failed after an earlier failure, a cleanup that failed (its error and the file left). words()
// says all of them. State files are small: more than kMaxStateBytes is refused.
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
/// A named mutex every integrity level may open (created, or opened when it exists); nullptr with GetLastError on failure.
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

// The game's state lock, held from construction (when held()) to destruction. Bounded wait: not held after waitMs; why:
// timedOut() (another holder), denied() (this process may not open the folder or the lock: ERROR_ACCESS_DENIED), else error().
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
    bool timedOut() const { return !held_ && err_ == WAIT_TIMEOUT; }
    bool denied() const { return !held_ && err_ == ERROR_ACCESS_DENIED; }
    DWORD error() const { return err_; }  // why not held: WAIT_TIMEOUT, ERROR_ACCESS_DENIED, or another Win32 error

private:
    HANDLE mutex_ = nullptr;
    bool held_ = false, abandoned_ = false;
    DWORD err_ = 0;
};

// ---- the operation guard -------------------------------------------------------------------------------------------
// The launcher's side of an uninstall, repair or update (see THE OPERATION GUARD above). Ok only when the state lock is held
// and every WrongGPU binary that exists is held exclusively. A timeout of the lock is Busy, access denied is LockDenied; a
// binary loaded in a process (ERROR_SHARING_VIOLATION) is GameRunning; any other open error is Unknown. Keep it for the
// whole operation, on the thread that took it.
enum class GuardResult { Ok, Busy, LockDenied, GameRunning, Unknown };
inline const char* GuardWords(GuardResult r) {
    switch (r) {
        case GuardResult::Ok: return "";
        case GuardResult::Busy: return "another WrongGPU program is changing this game's WrongGPU state right now; try again in a moment";
        case GuardResult::LockDenied: return "WrongGPU may not use this game's state lock (access denied); run the launcher's repair as administrator";
        case GuardResult::GameRunning: return "the game is running with WrongGPU loaded; close the game first";
        case GuardResult::Unknown: return "WrongGPU could not tell whether the game is using its files; close the game and try again";
    }
    return "?";
}
class OperationGuard {
public:
    // binaries: every WrongGPU binary of the install (the front end, the menu and RR plugins, the runtime), full paths; one that
    // does not exist is skipped (nothing can have it loaded).
    OperationGuard(const std::wstring& gameDir, const std::vector<std::wstring>& binaries, DWORD waitMs = kDefaultWaitMs) {
        lock_.reset(new StateLock(gameDir, waitMs));
        if (!lock_->held()) {
            err_ = lock_->error();
            const bool gone = err_ == ERROR_FILE_NOT_FOUND || err_ == ERROR_PATH_NOT_FOUND;  // the game folder is gone: nothing to guard
            result_ = gone ? GuardResult::Ok : lock_->timedOut() ? GuardResult::Busy : lock_->denied() ? GuardResult::LockDenied : GuardResult::Unknown;
            return;
        }
        for (const std::wstring& b : binaries) {
            const std::wstring p = ExtendedPath(b);
            // FILE_WRITE_DATA is what a loaded image refuses (the image section holds no open handle, so share mode 0 alone
            // would not conflict with it); nothing is ever written through this handle
            HANDLE h = CreateFileW(p.c_str(), DELETE | GENERIC_READ | FILE_WRITE_DATA, 0, nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
            if (h == INVALID_HANDLE_VALUE) {
                const DWORD e = GetLastError();
                if (e == ERROR_FILE_NOT_FOUND || e == ERROR_PATH_NOT_FOUND) continue;  // not there: nothing has it loaded
                err_ = e;
                which_ = b;
                result_ = e == ERROR_SHARING_VIOLATION ? GuardResult::GameRunning : GuardResult::Unknown;
                CloseAll();
                return;
            }
            held_.push_back({p, h});
        }
        result_ = GuardResult::Ok;
    }
    ~OperationGuard() { CloseAll(); }
    OperationGuard(const OperationGuard&) = delete;
    OperationGuard& operator=(const OperationGuard&) = delete;
    bool ok() const { return result_ == GuardResult::Ok; }
    GuardResult result() const { return result_; }
    DWORD error() const { return err_; }
    const std::wstring& which() const { return which_; }  // the binary that decided GameRunning / Unknown
    const char* words() const { return GuardWords(result_); }
    // Whether the guard holds `path` (a binary it was given).
    bool Holds(const std::wstring& path) const { return Find(path) >= 0; }
    // Closes the guard's handle of `path` right before the operation replaces that file itself (the file is then open to
    // others again: do it at the step, never earlier). True when it was held.
    bool Release(const std::wstring& path) {
        const int i = Find(path);
        if (i < 0) return false;
        CloseHandle(held_[i].h);
        held_.erase(held_.begin() + i);
        return true;
    }
    // Deletes `path` through the guard's own handle (no window in which another process can open or load it), then closes the
    // handle. ERROR_SUCCESS, ERROR_NOT_FOUND (not held), or the error of the delete (the handle is kept then).
    DWORD DeleteHeld(const std::wstring& path) {
        const int i = Find(path);
        if (i < 0) return ERROR_NOT_FOUND;
        struct DispositionEx { DWORD Flags; } ex{0x1u | 0x2u};  // FILE_DISPOSITION_FLAG_DELETE | FILE_DISPOSITION_FLAG_POSIX_SEMANTICS
        BOOL done = SetFileInformationByHandle(held_[i].h, static_cast<FILE_INFO_BY_HANDLE_CLASS>(21) /* FileDispositionInfoEx */, &ex, sizeof ex);
        if (!done) {  // (a file system without the POSIX form: the classic delete-on-close)
            FILE_DISPOSITION_INFO classic{TRUE};
            done = SetFileInformationByHandle(held_[i].h, FileDispositionInfo, &classic, sizeof classic);
        }
        if (!done) return GetLastError();
        CloseHandle(held_[i].h);
        held_.erase(held_.begin() + i);
        return ERROR_SUCCESS;
    }

private:
    struct Held { std::wstring path; HANDLE h; };
    int Find(const std::wstring& path) const {
        const std::wstring p = ExtendedPath(path);
        for (size_t i = 0; i < held_.size(); ++i)
            if (CompareStringOrdinal(held_[i].path.c_str(), -1, p.c_str(), -1, TRUE) == CSTR_EQUAL) return static_cast<int>(i);
        return -1;
    }
    void CloseAll() {
        for (auto& x : held_) CloseHandle(x.h);
        held_.clear();
    }
    std::unique_ptr<StateLock> lock_;
    std::vector<Held> held_;
    std::wstring which_;
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

/// The outcome of WriteAtomic: ok, or the first failure (error) AND every secondary failure: a close that failed after an
// earlier failure (closeError), a cleanup that failed (cleanupError, with the file left in tempLeft). words() says all of them.
struct WriteResult {
    bool ok = false;
    DWORD error = 0;         // the first failure (ERROR_FILE_TOO_LARGE: over kMaxStateBytes; ERROR_WRITE_FAULT: a short write)
    DWORD closeError = 0;    // the temporary file's close failed (also after an earlier failure)
    DWORD cleanupError = 0;  // the temporary file could not be deleted after the failure
    std::wstring tempLeft;   // ... and is still there under this name
    std::string words() const {
        if (ok) return "written";
        std::string w = "not written (error " + std::to_string(error) + ")";
        if (closeError && closeError != error) w += "; closing the temporary file also failed (error " + std::to_string(closeError) + ")";
        if (cleanupError) w += "; the temporary file could not be deleted (error " + std::to_string(cleanupError) + "), it is left next to the file";
        return w;
    }
};

// The atomic replace described above. On a failure the file is as it was.
inline WriteResult WriteAtomic(const std::wstring& path, const std::string& bytes) {
    static volatile LONG counter = 0;
    WriteResult r;
    if (bytes.size() > kMaxStateBytes) { r.error = ERROR_FILE_TOO_LARGE; return r; }
    const std::wstring target = ExtendedPath(path);
    for (int attempt = 0; attempt < 16; ++attempt) {
        const std::wstring tmp = target + kTempMark + std::to_wstring(GetCurrentProcessId()) + L"-" + std::to_wstring(InterlockedIncrement(&counter));
        HANDLE h = CreateFileW(tmp.c_str(), GENERIC_WRITE, 0, nullptr, CREATE_NEW, FILE_ATTRIBUTE_NORMAL, nullptr);
        if (h == INVALID_HANDLE_VALUE) {
            r.error = GetLastError();
            if (r.error == ERROR_FILE_EXISTS || r.error == ERROR_ALREADY_EXISTS) continue;
            return r;
        }
        bool ok = true;
        for (size_t done = 0; ok && done < bytes.size();) {
            const DWORD want = static_cast<DWORD>(bytes.size() - done < kChunk ? bytes.size() - done : kChunk);
            DWORD put = 0;
            if (Faulted("write")) { ok = false; r.error = ERROR_WRITE_FAULT; }
            else if (!WriteFile(h, bytes.data() + done, want, &put, nullptr)) { ok = false; r.error = GetLastError(); }
            else if (put != want) { ok = false; r.error = ERROR_WRITE_FAULT; }  // a short write: always this error
            else done += put;
        }
        if (ok) {
            if (Faulted("flush")) { ok = false; r.error = ERROR_WRITE_FAULT; }
            else if (!FlushFileBuffers(h)) { ok = false; r.error = GetLastError(); }
        }
        const bool faultClose = Faulted("close");
        const BOOL closed = CloseHandle(h);  // (closed in every case; a failure is reported, also after an earlier one)
        if (faultClose || !closed) {
            r.closeError = faultClose ? ERROR_WRITE_FAULT : GetLastError();
            if (ok) { ok = false; r.error = r.closeError; }
        }
        if (ok) {
            if (Faulted("move")) { ok = false; r.error = ERROR_WRITE_FAULT; }
            else if (!MoveFileExW(tmp.c_str(), target.c_str(), MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH)) { ok = false; r.error = GetLastError(); }
        }
        if (ok) { r.ok = true; r.error = 0; return r; }
        const bool faultDelete = Faulted("delete");
        if (faultDelete || !DeleteFileW(tmp.c_str())) {  // deleted when possible; reported otherwise
            r.cleanupError = faultDelete ? ERROR_WRITE_FAULT : GetLastError();
            r.tempLeft = tmp;
        }
        return r;
    }
    if (!r.error) r.error = ERROR_FILE_EXISTS;
    return r;
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

enum class SetResult { Ok, NoStateFolder, StateFolderUnknown, Busy, LockDenied, Unreadable, Invalid, AccessDenied, WriteFailed };
inline const char* SetResultWords(SetResult r) {
    switch (r) {
        case SetResult::Ok: return "saved";
        case SetResult::NoStateFolder: return "no WrongGPU state folder for this game (not installed by the launcher)";
        case SetResult::StateFolderUnknown: return "WrongGPU's state folder of this game could not be checked";
        case SetResult::Busy: return "another WrongGPU program kept the game's state lock";
        case SetResult::LockDenied: return "WrongGPU may not use this game's state lock (access denied); run the launcher's repair as administrator";
        case SetResult::Unreadable: return "the settings file could not be read (left as it is)";
        case SetResult::Invalid: return "not a valid key or value";
        case SetResult::AccessDenied: return "WrongGPU can't write to the game folder; run the launcher's repair as administrator";
        case SetResult::WriteFailed: return "the settings file could not be written";
    }
    return "?";
}
// The outcome of SetSettings: the result, its Win32 error, and (a failed write) the write's secondary failures. words() says
// everything a consumer must show, including a temporary file that could not be deleted.
struct SetOutcome {
    SetResult result = SetResult::Ok;
    DWORD error = 0;
    WriteResult write;  // the write's outcome when one was attempted (secondary close / cleanup failures)
    bool ok() const { return result == SetResult::Ok; }
    std::string words() const {
        std::string w = SetResultWords(result);
        if (result != SetResult::Ok && error && error != WAIT_TIMEOUT) w += " (error " + std::to_string(error) + ")";
        if (write.closeError && write.closeError != write.error) w += "; closing the temporary file also failed (error " + std::to_string(write.closeError) + ")";
        if (write.cleanupError) w += "; a temporary file could not be deleted (error " + std::to_string(write.cleanupError) + ") and is left in the state folder";
        return w;
    }
};
// The read-modify-write of settings.ini under the game's state lock: the file re-read from disk after the lock is taken, the
// changes applied (a corrupt file starts empty; an unreadable one is never overwritten), written atomically. The state folder
// is never created here. Distinct results for a consumer to show: AccessDenied (the game folder is not writable for this
// process) and LockDenied (the lock may not be opened) both point to the launcher's elevated repair; Busy is contention.
inline SetOutcome SetSettings(const std::wstring& gameDir, const std::vector<std::pair<std::string, std::string>>& changes, DWORD waitMs = kDefaultWaitMs) {
    SetOutcome o;
    auto fail = [&](SetResult r, DWORD e) { o.result = r; o.error = e; return o; };
    for (const auto& c : changes) if (!ValidKey(c.first) || !ValidValue(c.second)) return fail(SetResult::Invalid, 0);
    DWORD e = 0;
    switch (DirectoryState(StateDirOf(gameDir), &e)) {
        case PathState::Present: break;
        case PathState::Missing: return fail(SetResult::NoStateFolder, 0);
        case PathState::Unknown: return fail(e == ERROR_ACCESS_DENIED ? SetResult::AccessDenied : SetResult::StateFolderUnknown, e);
    }
    StateLock lock(gameDir, waitMs);
    if (!lock.held()) return fail(lock.timedOut() ? SetResult::Busy : lock.denied() ? SetResult::LockDenied : SetResult::StateFolderUnknown, lock.error());
    std::string text;
    bool missing = false;
    Settings s;
    if (ReadBytes(SettingsPathOf(gameDir), &text, &missing, &e)) s = ParseSettings(text);
    else if (!missing) return fail(e == ERROR_ACCESS_DENIED ? SetResult::AccessDenied : SetResult::Unreadable, e);
    if (s.corrupt) s = Settings();
    bool changed = !s.present;
    for (const auto& c : changes) {
        const std::string* now = Get(s, c.first);
        if (!now || *now != c.second) { Set(s, c.first, c.second); changed = true; }
    }
    if (!changed) return o;
    o.write = WriteAtomic(SettingsPathOf(gameDir), RenderSettings(s));
    if (!o.write.ok) return fail(o.write.error == ERROR_ACCESS_DENIED ? SetResult::AccessDenied : SetResult::WriteFailed, o.write.error);
    return o;
}

}  // namespace wgstate
