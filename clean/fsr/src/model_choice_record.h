// SPDX-License-Identifier: MIT
#pragma once
// The model choice record (0.6.5): ONE small file per game, wronggpu-setup\model-choice.ini in the folder of the front
// end's wronggpu.ini (the launcher's state folder: an uninstall removes it by that name, a reinstall keeps it with the
// game's settings; without that folder there is no record). It decides which WrongGPU model runs. Three places write it,
// each only for an EVENT, a choice the player made:
//  - the launcher, when it applies a model (`WrongGPU.exe set <game> model=<m>`, or a change of the model item); it then
//    sets OptiScaler's UpscalerIndex to that model's entry, for display only;
//  - the front end, when OptiScaler tells it that the player selected one of WrongGPU's entries in its upscaler list
//    during the session (the WrongGPU edition of OptiScaler calls WrongGpuHostSelectedV1 in the front-end DLL); the
//    stored index OptiScaler passes at a create is never taken for a choice;
//  - the WrongGPU menu plugin (clean/fsr/asi), when the player clicks one of its model entries.
// "Most recent" is the highest sequence number. Every read-modify-write (read, seq + 1, write) runs under ONE lock per
// game shared by all processes (a named mutex, Local\WrongGPU-choice-<hash of the record's full path>), so no update is
// lost and seq never goes back. No clock is involved.
//
// The file, strictly (anything else is CORRUPT: logged and treated as "no choice", never as a partial one):
//   ; comment lines and blank lines
//   model=<M|K|L>
//   seq=<1..18446744073709551615>          (decimal digits only, overflow refused)
//   source=<launcher|ffx|plugin>
// each key exactly once, no other key, no spaces around '='. Writes: a temporary file in the same folder, written in full
// and flushed (both checked), then MoveFileExW(REPLACE_EXISTING | WRITE_THROUGH) over the record (checked). Readers take
// the lock too, so a reader never sees part of a record. A temporary file a crash left behind is never read.
//
// Upgrade: a game without a record gets one from the launcher's existing choice (wronggpu.ini's model item, else Model M):
// seq 1, source=launcher (MigrateRecord). The model is never taken from OptiScaler's stored index.
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace wgchoice {

constexpr const wchar_t* kStateFolder = L"wronggpu-setup";  // the launcher's state folder (installer kStateDirName)
constexpr const wchar_t* kRecordFile = L"model-choice.ini";
// The record of the game whose wronggpu.ini is in `iniDir`.
inline std::filesystem::path RecordPathIn(const std::filesystem::path& iniDir) { return iniDir / kStateFolder / kRecordFile; }

enum class Source { Launcher, Ffx, Plugin };

struct Record {
    char model = 'M';  // 'M', 'K', 'L'
    uint64_t seq = 0;  // >= 1 in a valid record
    Source source = Source::Launcher;
};

inline const char* SourceName(Source s) { return s == Source::Launcher ? "launcher" : s == Source::Ffx ? "ffx" : "plugin"; }
inline const char* SourceWords(Source s) {
    return s == Source::Launcher ? "the launcher" : s == Source::Ffx ? "OptiScaler's upscaler list" : "the WrongGPU menu plugin";
}
inline bool IsModel(char c) { return c == 'M' || c == 'K' || c == 'L'; }

// Decimal digits only, 1..20 of them, no overflow, not zero.
inline bool ParseSeq(const std::string& v, uint64_t* out) {
    if (v.empty() || v.size() > 20) return false;
    uint64_t n = 0;
    for (char c : v) {
        if (c < '0' || c > '9') return false;
        const uint64_t d = uint64_t(c - '0');
        if (n > (UINT64_MAX - d) / 10) return false;  // n * 10 + d would overflow
        n = n * 10 + d;
    }
    if (n == 0) return false;
    *out = n;
    return true;
}

// false (with *why) for anything but one valid record.
inline bool ParseRecord(const std::string& text, Record* out, std::string* why) {
    Record r;
    bool hasModel = false, hasSeq = false, hasSource = false;
    auto fail = [&](const std::string& w) {
        if (why) *why = w;
        return false;
    };
    size_t at = 0;
    int lineNo = 0;
    while (at < text.size()) {
        size_t end = text.find('\n', at);
        if (end == std::string::npos) end = text.size();
        std::string line = text.substr(at, end - at);
        at = end + 1;
        ++lineNo;
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty() || line[0] == ';' || line[0] == '#') continue;
        const size_t eq = line.find('=');
        if (eq == std::string::npos) return fail("line " + std::to_string(lineNo) + " is not key=value");
        const std::string k = line.substr(0, eq), v = line.substr(eq + 1);
        if (k == "model") {
            if (hasModel) return fail("model given twice");
            if (v.size() != 1 || !IsModel(v[0])) return fail("model is not M, K or L");
            r.model = v[0];
            hasModel = true;
        } else if (k == "seq") {
            if (hasSeq) return fail("seq given twice");
            if (!ParseSeq(v, &r.seq)) return fail("seq is not a number from 1 to 18446744073709551615");
            hasSeq = true;
        } else if (k == "source") {
            if (hasSource) return fail("source given twice");
            if (v == "launcher") r.source = Source::Launcher;
            else if (v == "ffx") r.source = Source::Ffx;
            else if (v == "plugin") r.source = Source::Plugin;
            else return fail("source is not launcher, ffx or plugin");
            hasSource = true;
        } else {
            return fail("unknown key \"" + k + "\"");
        }
    }
    if (!hasModel || !hasSeq || !hasSource) return fail("model, seq and source are all needed");
    *out = r;
    return true;
}

inline std::string FormatRecord(const Record& r) {
    return std::string("; WrongGPU: the model chosen last for this game, by the launcher, OptiScaler's upscaler list (ffx) or the\r\n"
                       "; WrongGPU menu plugin. The highest seq is the most recent choice. Written by WrongGPU; do not edit.\r\n") +
           "model=" + r.model + "\r\nseq=" + std::to_string(r.seq) + "\r\nsource=" + SourceName(r.source) + "\r\n";
}

// ---- the lock -------------------------------------------------------------------------------------------------------

// The lock's name for a record: Local\WrongGPU-choice-<FNV-1a 64 of the record's full path, lower case, '\' separators>.
// The launcher, the front end and the plugin (different processes and modules) derive the same name from the same file.
inline std::wstring LockName(const std::filesystem::path& record) {
    std::wstring full = record.wstring();
#ifdef _WIN32
    wchar_t buf[1024];
    const DWORD n = GetFullPathNameW(full.c_str(), 1024, buf, nullptr);
    if (n > 0 && n < 1024) full.assign(buf, n);
#endif
    uint64_t h = 1469598103934665603ull;
    for (wchar_t c : full) {
        if (c == L'/') c = L'\\';
        if (c >= L'A' && c <= L'Z') c = wchar_t(c - L'A' + L'a');
        h ^= uint64_t(uint16_t(c));
        h *= 1099511628211ull;
    }
    wchar_t hex[17];
    for (int i = 15; i >= 0; --i, h >>= 4) hex[i] = L"0123456789abcdef"[h & 15];
    hex[16] = 0;
    return std::wstring(L"Local\\WrongGPU-choice-") + hex;
}

// Holds the record's lock for its lifetime. ok(): taken (within `timeoutMs`). An abandoned lock (a holder that died) is
// taken over: the record on disk is whole either way (atomic replacement).
class Lock {
public:
    explicit Lock(const std::filesystem::path& record, unsigned timeoutMs = 3000) {
#ifdef _WIN32
        handle_ = CreateMutexW(nullptr, FALSE, LockName(record).c_str());
        if (!handle_) return;
        const DWORD w = WaitForSingleObject(handle_, timeoutMs);
        ok_ = w == WAIT_OBJECT_0 || w == WAIT_ABANDONED;
#else
        (void)record;
        (void)timeoutMs;
        ok_ = true;
#endif
    }
    ~Lock() {
#ifdef _WIN32
        if (handle_) {
            if (ok_) ReleaseMutex(handle_);
            CloseHandle(handle_);
        }
#endif
    }
    bool ok() const { return ok_; }
    Lock(const Lock&) = delete;
    Lock& operator=(const Lock&) = delete;

private:
    bool ok_ = false;
#ifdef _WIN32
    HANDLE handle_ = nullptr;
#endif
};

// ---- files ----------------------------------------------------------------------------------------------------------

enum class ReadResult { Missing, Ok, Corrupt, Unreadable };

// Reads the record (call it under the record's Lock).
inline ReadResult ReadRecordFile(const std::filesystem::path& path, Record* out, std::string* why) {
    std::string text;
#ifdef _WIN32
    HANDLE h = CreateFileW(path.c_str(), GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, nullptr, OPEN_EXISTING,
                           FILE_ATTRIBUTE_NORMAL, nullptr);
    if (h == INVALID_HANDLE_VALUE) {
        const DWORD e = GetLastError();
        if (e == ERROR_FILE_NOT_FOUND || e == ERROR_PATH_NOT_FOUND) return ReadResult::Missing;
        if (why) *why = "cannot open (Windows error " + std::to_string(e) + ")";
        return ReadResult::Unreadable;
    }
    char buf[2048];
    DWORD got = 0;
    const BOOL ok = ReadFile(h, buf, sizeof buf, &got, nullptr);
    CloseHandle(h);
    if (!ok) {
        if (why) *why = "cannot read";
        return ReadResult::Unreadable;
    }
    if (got == sizeof buf) {
        if (why) *why = "larger than a record";
        return ReadResult::Corrupt;
    }
    text.assign(buf, got);
#else
    std::error_code ec;
    if (!std::filesystem::exists(path, ec)) return ReadResult::Missing;
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        if (why) *why = "cannot open";
        return ReadResult::Unreadable;
    }
    std::ostringstream s;
    s << f.rdbuf();
    text = s.str();
    if (text.size() >= 2048) {
        if (why) *why = "larger than a record";
        return ReadResult::Corrupt;
    }
#endif
    return ParseRecord(text, out, why) ? ReadResult::Ok : ReadResult::Corrupt;
}

// Atomic replace: a temporary file next to the record (its name ends in .tmp; never read), written in full and flushed
// (both checked), then MoveFileExW(REPLACE_EXISTING | WRITE_THROUGH) over the record (checked). Up to `attempts` tries,
// 20 ms apart. false with *why. Call it under the record's Lock.
inline bool WriteRecordFile(const std::filesystem::path& path, const Record& r, std::string* why, int attempts = 3) {
    const std::string text = FormatRecord(r);
    std::string last;
    for (int i = 0; i < attempts; ++i) {
#ifdef _WIN32
        const std::wstring tmp = path.wstring() + L"." + std::to_wstring(GetCurrentProcessId()) + L"." +
                                 std::to_wstring(GetTickCount64()) + L"." + std::to_wstring(i) + L".tmp";
        HANDLE h = CreateFileW(tmp.c_str(), GENERIC_WRITE, 0, nullptr, CREATE_NEW, FILE_ATTRIBUTE_NORMAL, nullptr);
        if (h == INVALID_HANDLE_VALUE) {
            last = "cannot create the temporary file (Windows error " + std::to_string(GetLastError()) + ")";
        } else {
            DWORD written = 0;
            const bool wrote = WriteFile(h, text.data(), DWORD(text.size()), &written, nullptr) && written == text.size();
            const bool flushed = wrote && FlushFileBuffers(h);
            const DWORD flushError = flushed ? 0 : GetLastError();
            const bool closed = CloseHandle(h) != 0;
            if (!wrote) {
                last = "cannot write the temporary file";
                DeleteFileW(tmp.c_str());
            } else if (!flushed) {
                last = "cannot flush the temporary file (Windows error " + std::to_string(flushError) + ")";
                DeleteFileW(tmp.c_str());
            } else if (!closed) {
                last = "cannot close the temporary file";
                DeleteFileW(tmp.c_str());
            } else if (MoveFileExW(tmp.c_str(), path.c_str(), MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH)) {
                return true;
            } else {
                last = "cannot replace the record (Windows error " + std::to_string(GetLastError()) + ")";
                DeleteFileW(tmp.c_str());
            }
        }
        if (i + 1 < attempts) Sleep(20);
#else
        const std::filesystem::path tmp = path.string() + "." + std::to_string(i) + ".tmp";
        {
            std::ofstream f(tmp, std::ios::binary | std::ios::trunc);
            f.write(text.data(), std::streamsize(text.size()));
            f.flush();
            if (!f) {
                last = "cannot write the temporary file";
                std::error_code ec;
                std::filesystem::remove(tmp, ec);
                continue;
            }
        }
        std::error_code ec;
        std::filesystem::rename(tmp, path, ec);
        if (!ec) return true;
        last = "cannot replace the record";
        std::filesystem::remove(tmp, ec);
#endif
    }
    if (why) *why = last;
    return false;
}

// ---- the operations, each one read-modify-write under the lock ------------------------------------------------------

enum class Outcome { Written, NoLock, Unreadable, SeqExhausted, WriteFailed, Superseded };

inline const char* OutcomeWords(Outcome o) {
    switch (o) {
    case Outcome::Written: return "written";
    case Outcome::Superseded: return "a newer choice was recorded meanwhile (this one is dropped)";
    case Outcome::NoLock: return "the record's lock could not be taken";
    case Outcome::Unreadable: return "the record cannot be read (not written over)";
    case Outcome::SeqExhausted: return "the record's sequence number is at its end (18446744073709551615): choice refused";
    case Outcome::WriteFailed: return "the record could not be written";
    }
    return "?";
}

// A choice that could not be written when it was made (NoLock, Unreadable, WriteFailed): kept by its writer, used for
// the session and written later (RetryPending). base: the seq the record had when the choice was made (0: no record or a
// corrupt one); baseKnown false: the record could not be read then.
struct Pending {
    char model = 0;  // 0: nothing pending
    Source source = Source::Plugin;
    bool baseKnown = false;
    uint64_t baseSeq = 0;
    bool active() const { return model != 0; }
};

// A choice of `model` by `source`: read the record and write seq + 1 (seq 1 without a record, or over a corrupt one,
// which *why then names). `retry`: a pending choice being written later: dropped (Superseded) when the record moved past
// its base meanwhile. *after: the record written. *pending (when given): filled on a failure that can be retried. Any
// failure leaves the record as it was. Call it under the record's Lock (RecordChoice / RetryPending take it).
inline Outcome RecordChoiceUnlocked(const std::filesystem::path& path, char model, Source source, Record* after, std::string* why,
                                    Pending* pending = nullptr, const Pending* retry = nullptr) {
    Record cur;
    std::string err;
    const ReadResult rr = ReadRecordFile(path, &cur, &err);
    if (rr == ReadResult::Unreadable) {
        if (why) *why = err;
        if (pending) *pending = Pending{model, source, false, 0};
        return Outcome::Unreadable;
    }
    const uint64_t seqNow = rr == ReadResult::Ok ? cur.seq : 0;
    if (retry && retry->baseKnown && seqNow != retry->baseSeq) return Outcome::Superseded;
    if (rr == ReadResult::Corrupt && why) *why = "a corrupt record was replaced (" + err + ")";
    if (rr == ReadResult::Ok && cur.seq == UINT64_MAX) return Outcome::SeqExhausted;
    const Record next{model, seqNow + 1, source};
    if (!WriteRecordFile(path, next, &err)) {
        if (why) *why = err;
        if (pending) *pending = Pending{model, source, true, seqNow};
        return Outcome::WriteFailed;
    }
    if (after) *after = next;
    return Outcome::Written;
}

inline Outcome RecordChoice(const std::filesystem::path& path, char model, Source source, Record* after, std::string* why,
                            Pending* pending = nullptr, unsigned lockMs = 3000) {
    Lock lock(path, lockMs);
    if (!lock.ok()) {
        if (pending) *pending = Pending{model, source, false, 0};
        return Outcome::NoLock;
    }
    return RecordChoiceUnlocked(path, model, source, after, why, pending);
}

// Writes a pending choice (see Pending). Written or Superseded: nothing is pending any more (*p cleared); SeqExhausted:
// cleared too (refused for good); otherwise *p stays for a later try.
inline Outcome RetryPending(const std::filesystem::path& path, Pending* p, Record* after, std::string* why, unsigned lockMs = 3000) {
    if (!p->active()) return Outcome::Superseded;
    Lock lock(path, lockMs);
    if (!lock.ok()) return Outcome::NoLock;
    const Pending retry = *p;
    const Outcome o = RecordChoiceUnlocked(path, retry.model, retry.source, after, why, nullptr, &retry);
    if (o == Outcome::Written || o == Outcome::Superseded || o == Outcome::SeqExhausted) *p = Pending{};
    return o;
}

// The record as it is now, read under the lock. A game without one (an install from before 0.6.5) gets one from the
// launcher's existing choice (`launcherModel`, 'M' / 'K' / 'L', or 0 for none: Model M), seq 1, source=launcher.
// Returns Ok (also after the migration; *out set, *migrated true then), Corrupt or Unreadable (*why; nothing written), or
// Missing when the lock or the migration's write failed (*why).
inline ReadResult LoadOrMigrateRecord(const std::filesystem::path& path, char launcherModel, Record* out, bool* migrated, std::string* why) {
    if (migrated) *migrated = false;
    Lock lock(path);
    if (!lock.ok()) {
        if (why) *why = "the record's lock could not be taken";
        return ReadResult::Unreadable;
    }
    const ReadResult rr = ReadRecordFile(path, out, why);
    if (rr != ReadResult::Missing) return rr;
    const Record first{IsModel(launcherModel) ? launcherModel : 'M', 1, Source::Launcher};
    if (!WriteRecordFile(path, first, why)) return ReadResult::Missing;
    *out = first;
    if (migrated) *migrated = true;
    return ReadResult::Ok;
}

}  // namespace wgchoice
