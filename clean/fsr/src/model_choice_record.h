// SPDX-License-Identifier: MIT
#pragma once
// The model choice record (0.6.5): ONE small file per game, model-choice.ini in the game's WrongGPU state folder
// (clean/wgshared/include/wg_state.h: <game folder>\wronggpu-setup, beside settings.ini; always WrongGPU's: an uninstall
// removes it, a repair, update or reinstall keeps it; without that folder there is no record). It decides which WrongGPU
// model runs, on every route; wronggpu.ini's model item is the install's default, which no WrongGPU program rewrites
// after the install and which only seeds the record of a game that has none (the upgrade below). Three places write the
// record, each only for an EVENT, a choice the player made:
//  - the launcher, when the player applies a model (the launcher's Apply, or `WrongGPU.exe set <game> model=<m>`: always,
//    also the same model again); it then sets OptiScaler's UpscalerIndex to that model's entry, for display only (and it
//    seeds the record of a game without one from its profile);
//  - the front end, when OptiScaler tells it that the player selected one of WrongGPU's entries in its upscaler list
//    (the WrongGPU edition of OptiScaler calls WrongGpuHostSelectedV1 in the front-end DLL); the stored index OptiScaler
//    passes at a create is never taken for a choice;
//  - the WrongGPU menu plugin (clean/fsr/asi), when the player clicks one of its model entries.
// "Most recent" is the highest sequence number. Every read-modify-write (read, seq + 1, write) runs under the game's
// state lock (wgstate::StateLock, one named mutex per game folder, shared by every WrongGPU program and by settings.ini's
// writers), so no update is lost and seq never goes back. No clock is involved.
//
// The file, strictly (anything else is CORRUPT: logged and treated as "no choice", never as a partial one):
//   ; comment lines and blank lines
//   model=<M|K|L>
//   seq=<1..18446744073709551615>          (decimal digits only, overflow refused)
//   source=<launcher|ffx|plugin>
// each key exactly once, no other key, no spaces around '='; the last line is WrongGPU's stamp (a comment,
// wg_state.h Stamped) when the file is WrongGPU's and unedited and the install has an id. Writes: in place through
// wgstate::UpdateStateFile (only the three keys' lines change; a corrupt record is never replaced; an edited or
// player-made record stays the player's, unstamped; RecordChoiceUnlocked), a temporary file in the same
// folder, written and flushed, then MoveFileExW(REPLACE_EXISTING | WRITE_THROUGH), every step checked), tried up to 3
// times. Readers take the lock too, so a reader never sees part of a record. A temporary file a crash left behind is
// never read.
//
// Upgrade: a game without a record gets one from the launcher's existing choice (wronggpu.ini's model item, else Model M):
// seq 1, source=launcher, read and written under the lock (LoadOrMigrateRecord). The model is never taken from OptiScaler's
// stored index. A choice that cannot be saved (the lock not taken in time, an unreadable record, a failed write) is not
// kept for later: its writer says "not saved, try again" and the model that runs does not change.
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <string>

#include "../../wgshared/include/wg_state.h"  // the state folder, its lock, atomic writes (shared with settings.ini)

namespace wgchoice {

constexpr const wchar_t* kStateFolder = wgstate::kStateFolder;  // wronggpu-setup
constexpr const wchar_t* kRecordFile = wgstate::kModelChoiceFile;
// The record of the game whose WrongGPU files (wronggpu.ini, the state folder) are in `gameDir`.
inline std::filesystem::path RecordPathIn(const std::filesystem::path& gameDir) { return gameDir / kStateFolder / kRecordFile; }
// The game folder of a record (the folder holding its state folder): the game's state lock is taken on it.
inline std::filesystem::path GameDirOf(const std::filesystem::path& record) { return record.parent_path().parent_path(); }

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

// The comment a new record starts with.
constexpr const char* kRecordHeader =
    "; WrongGPU: the model chosen last for this game, by the launcher, OptiScaler's upscaler list (ffx) or the\r\n"
    "; WrongGPU menu plugin. The highest seq is the most recent choice. Written by WrongGPU; do not edit.\r\n";

inline std::string FormatRecord(const Record& r) {
    return std::string(kRecordHeader) + "model=" + r.model + "\r\nseq=" + std::to_string(r.seq) + "\r\nsource=" + SourceName(r.source) + "\r\n";
}

// ---- the lock -------------------------------------------------------------------------------------------------------

// Holds the game's state lock (wgstate::StateLock on the record's game folder) for its lifetime. ok(): taken within
// `timeoutMs`. An abandoned lock (a holder that died) is taken over: the files are whole either way (atomic replacement).
class Lock {
public:
    explicit Lock(const std::filesystem::path& record, unsigned timeoutMs = 3000) : lock_(GameDirOf(record).wstring(), timeoutMs) {}
    bool ok() const { return lock_.held(); }
    Lock(const Lock&) = delete;
    Lock& operator=(const Lock&) = delete;

private:
    wgstate::StateLock lock_;
};

// ---- files ----------------------------------------------------------------------------------------------------------

enum class ReadResult { Missing, Ok, Corrupt, Unreadable };

// Reads the record (call it under the record's Lock).
inline ReadResult ReadRecordFile(const std::filesystem::path& path, Record* out, std::string* why) {
    std::string text;
    HANDLE h = CreateFileW(path.wstring().c_str(), GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, nullptr, OPEN_EXISTING,
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
    return ParseRecord(text, out, why) ? ReadResult::Ok : ReadResult::Corrupt;
}

// ---- the operations, each one read-modify-write under the lock ------------------------------------------------------

enum class Outcome { Written, NoLock, Unreadable, SeqExhausted, WriteFailed, Corrupt };

inline const char* OutcomeWords(Outcome o) {
    switch (o) {
    case Outcome::Written: return "written";
    case Outcome::NoLock: return "the game's state lock could not be taken in time";
    case Outcome::Unreadable: return "the record cannot be read (not written over)";
    case Outcome::SeqExhausted: return "the record's sequence number is at its end (18446744073709551615): choice refused";
    case Outcome::WriteFailed: return "the record could not be written";
    case Outcome::Corrupt:
        return "the record (wronggpu-setup\\model-choice.ini) is not a valid record, so it is left as it is and the choice is "
               "refused (fix the file or delete it)";
    }
    return "?";
}

// A choice of `model` by `source`: the record's model, seq + 1 (seq 1 without a record) and source, written IN PLACE through
// wgstate::UpdateStateFile (#224's rule for the game's state files): the file is held from the read to the replace, only
// the three keys' lines change, every other line (the player's comments, blank lines) and every line end stay, and the
// stamp is put back only when the file was WrongGPU's and unedited (its valid stamp) or new; an edited or player-made
// record stays unstamped, the player's, and an uninstall keeps it. A record that is not a valid record (corrupt, or text
// WrongGPU cannot keep) is never replaced: Corrupt (*why says what is wrong), and the caller keeps the model that runs.
// *after: the record written; *before: the record it replaced (model 0 when there was none). Any failure leaves the file
// as it was and nothing is kept for later: the caller says "not saved". A failed write is tried up to 3 times. Call it
// holding the game's state lock (RecordChoice takes it; the lock is recursive on one thread).
inline Outcome RecordChoiceUnlocked(const std::filesystem::path& path, char model, Source source, Record* after, std::string* why,
                                    Record* before = nullptr) {
    if (before) *before = Record{0, 0, Source::Launcher};
    Outcome result = Outcome::Written;
    Record next{model, 1, source};
    std::string err;
    wgstate::SetOutcome o;
    for (int attempt = 0; attempt < 3; ++attempt) {
        result = Outcome::Written;
        err.clear();
        o = wgstate::UpdateStateFile(
            GameDirOf(path).wstring(), kRecordFile,
            [&](const wgstate::Settings& now, std::vector<std::pair<std::string, std::string>>* changes) {
                Record cur;
                bool had = false;
                if (now.present) {  // (the strict record rules on the file as read, its valid stamp already off)
                    if (!ParseRecord(wgstate::RenderSettings(now), &cur, &err)) { result = Outcome::Corrupt; return; }
                    had = true;
                    if (before) *before = cur;
                    if (cur.seq == UINT64_MAX) { result = Outcome::SeqExhausted; return; }
                }
                next = Record{model, had ? cur.seq + 1 : 1, source};
                changes->push_back({"model", std::string(1, model)});
                changes->push_back({"seq", std::to_string(next.seq)});
                changes->push_back({"source", SourceName(source)});
            },
            1000, kRecordHeader);
        const bool writeFailed = o.result == wgstate::SetResult::WriteFailed || (o.result == wgstate::SetResult::AccessDenied && o.write.error);
        if (!writeFailed || result != Outcome::Written) break;
        Sleep(20);
    }
    switch (o.result) {
    case wgstate::SetResult::Ok:
        if (result != Outcome::Written) {
            if (why && !err.empty()) *why = err;
            return result;
        }
        if (after) *after = next;
        return Outcome::Written;
    case wgstate::SetResult::Unparseable:
        if (why) *why = o.why;
        return Outcome::Corrupt;
    case wgstate::SetResult::Busy:
    case wgstate::SetResult::LockDenied:
        if (why) *why = o.words();
        return Outcome::NoLock;
    case wgstate::SetResult::AccessDenied:
        if (why) *why = o.words();
        return o.write.error ? Outcome::WriteFailed : Outcome::Unreadable;  // (denied at the replace: a write that failed)
    case wgstate::SetResult::Unreadable:
    case wgstate::SetResult::NoStateFolder:
    case wgstate::SetResult::StateFolderUnknown:
        if (why) *why = o.words();
        return Outcome::Unreadable;
    default:
        if (why) *why = o.words();
        return Outcome::WriteFailed;
    }
}

inline Outcome RecordChoice(const std::filesystem::path& path, char model, Source source, Record* after, std::string* why,
                            Record* before = nullptr, unsigned lockMs = 3000) {
    Lock lock(path, lockMs);
    if (!lock.ok()) return Outcome::NoLock;
    return RecordChoiceUnlocked(path, model, source, after, why, before);
}

// The record as it is now, read under the game's state lock. A game without one (an install from before 0.6.5) gets one
// from the launcher's existing choice, read WHILE the lock is held (`launcherModel()`: wronggpu.ini's model item, 'M' /
// 'K' / 'L', or 0 for none: Model M), seq 1, source=launcher; so a launcher apply, which holds the same lock, is either
// wholly before or wholly after it. Returns Ok (also after the migration; *out set, *migrated true then), Corrupt or
// Unreadable (*why; nothing written), Missing when the migration's write failed (*why), or NoLock (the lock was not
// taken in `lockMs`; *why).
enum class LoadResult { Ok, Corrupt, Unreadable, Missing, NoLock };
template <class LauncherModelFn>
inline LoadResult LoadOrMigrateRecord(const std::filesystem::path& path, LauncherModelFn&& launcherModel, Record* out, bool* migrated,
                                      std::string* why, unsigned lockMs = 3000) {
    if (migrated) *migrated = false;
    Lock lock(path, lockMs);
    if (!lock.ok()) {
        if (why) *why = OutcomeWords(Outcome::NoLock);
        return LoadResult::NoLock;
    }
    const ReadResult rr = ReadRecordFile(path, out, why);
    if (rr == ReadResult::Ok) return LoadResult::Ok;
    if (rr == ReadResult::Corrupt) return LoadResult::Corrupt;
    if (rr == ReadResult::Unreadable) return LoadResult::Unreadable;
    const char m = launcherModel();
    Record first;
    const Outcome wo = RecordChoiceUnlocked(path, IsModel(m) ? m : 'M', Source::Launcher, &first, why);
    if (wo == Outcome::Corrupt) return LoadResult::Corrupt;  // (a file that appeared meanwhile is the player's: left as it is)
    if (wo != Outcome::Written) return LoadResult::Missing;
    *out = first;
    if (migrated) *migrated = true;
    return LoadResult::Ok;
}

}  // namespace wgchoice
