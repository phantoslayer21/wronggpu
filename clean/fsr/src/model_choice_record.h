// SPDX-License-Identifier: MIT
#pragma once
// The model choice record (0.6.5): ONE small file per game, wronggpu-setup\model-choice.ini in the folder of the front
// end's wronggpu.ini (the launcher's state folder: an uninstall removes it by that name, a reinstall keeps it with the
// game's settings; without that folder there is no record), that every place which can choose WrongGPU's model writes,
// and which decides the model:
//  - the launcher, when it applies a model (`WrongGPU.exe set <game> model=<m>`, or a change of the model item);
//  - the front end, when the WrongGPU entry picked in OptiScaler's upscaler list (the FFX box, or OptiScaler 10's WrongGPU
//    box) is not the one the record knows (`ffx`): the player picked it there;
//  - the WrongGPU menu plugin (clean/fsr/asi), when the player clicks one of its model entries.
// "Most recent" is the highest sequence number: a writer reads the record and writes seq + 1. No clock is involved, so a
// clock set back or forward changes nothing.
//
// The file, strictly (anything else is CORRUPT: logged and treated as "no choice", never as a partial one):
//   ; comment lines and blank lines
//   model=<M|K|L>
//   seq=<1..18446744073709551615>          (decimal digits only, overflow refused)
//   source=<launcher|ffx|plugin>
//   ffx=<M|K|L|->                          (the WrongGPU entry the writer knew OptiScaler's list to name; - = not known)
// each key exactly once, no other key, no spaces around '='. Writes are atomic: a temporary file in the same folder,
// flushed, then renamed over the record (on Windows through its handle with POSIX semantics, else MoveFileExW
// REPLACE_EXISTING | WRITE_THROUGH); a reader sees the old
// record or the new one, never part of one. A temporary file a crash left behind is never read.
//
// Why `ffx`: OptiScaler's list keeps naming the entry it last selected. The launcher sets that entry with its model (and
// writes ffx = the entry it set), the plugin keeps the record's ffx (it cannot move OptiScaler's list), so an unchanged
// list never looks like a new choice; only a list entry the record does not know is the player's choice there. With no
// record at all, the entry seen at the first create counts as a choice (the launcher writes the record whenever it sets a
// model and the entry together, so a game without a record has had no model choice yet).
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

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
constexpr char kUnknown = '-';

enum class Source { Launcher, Ffx, Plugin };

struct Record {
    char model = 'M';        // 'M', 'K', 'L'
    uint64_t seq = 0;        // >= 1 in a valid record
    Source source = Source::Launcher;
    char ffx = kUnknown;     // 'M', 'K', 'L' or kUnknown
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
    bool hasModel = false, hasSeq = false, hasSource = false, hasFfx = false;
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
        } else if (k == "ffx") {
            if (hasFfx) return fail("ffx given twice");
            if (v.size() != 1 || !(IsModel(v[0]) || v[0] == kUnknown)) return fail("ffx is not M, K, L or -");
            r.ffx = v[0];
            hasFfx = true;
        } else {
            return fail("unknown key \"" + k + "\"");
        }
    }
    if (!hasModel || !hasSeq || !hasSource || !hasFfx) return fail("model, seq, source and ffx are all needed");
    *out = r;
    return true;
}

inline std::string FormatRecord(const Record& r) {
    return std::string("; WrongGPU: the model chosen last for this game, by the launcher, OptiScaler's upscaler list (ffx) or the\r\n"
                       "; WrongGPU menu plugin. The highest seq is the most recent choice. Written by WrongGPU; do not edit.\r\n") +
           "model=" + r.model + "\r\nseq=" + std::to_string(r.seq) + "\r\nsource=" + SourceName(r.source) + "\r\nffx=" + r.ffx + "\r\n";
}

// The record after a new choice of `model` by `source`, given the current one (`cur`, or none). `ffx`: what that writer
// knows OptiScaler's list names (kUnknown to keep the current record's). false when seq cannot grow (2^64 - 1 reached).
inline bool NextRecord(const Record* cur, char model, Source source, char ffx, Record* out) {
    if (cur && cur->seq == UINT64_MAX) return false;
    out->model = model;
    out->seq = cur ? cur->seq + 1 : 1;
    out->source = source;
    out->ffx = ffx != kUnknown ? ffx : cur ? cur->ffx : kUnknown;
    return true;
}

enum class FfxStep {
    None,     // nothing to write: the list names the entry the record knows (or no WrongGPU entry)
    Adopt,    // the record did not know the list's entry (ffx = -): written with it, not a new choice
    Choice,   // the player picked another WrongGPU entry in OptiScaler's list (or the first create without a record)
};

// What a create that sees OptiScaler's list naming `seen` ('M', 'K', 'L', or kUnknown for none / an AMD version) does
// to the record `cur` (or none): *out is the record to write for Adopt and Choice.
inline FfxStep FfxObservation(const Record* cur, char seen, Record* out) {
    if (!IsModel(seen)) return FfxStep::None;
    if (!cur) {
        NextRecord(nullptr, seen, Source::Ffx, seen, out);
        return FfxStep::Choice;
    }
    if (cur->ffx == seen) return FfxStep::None;
    if (cur->ffx == kUnknown) {
        *out = *cur;
        out->ffx = seen;
        return FfxStep::Adopt;
    }
    if (!NextRecord(cur, seen, Source::Ffx, seen, out)) return FfxStep::None;
    return FfxStep::Choice;
}

// ---- files --------------------------------------------------------------------------------------------------------

enum class ReadResult { Missing, Ok, Corrupt, Unreadable };

inline ReadResult ReadRecordFile(const std::filesystem::path& path, Record* out, std::string* why) {
    std::string text;
#ifdef _WIN32
    // FILE_SHARE_DELETE: a writer may replace the record while it is read (the reader keeps the old one)
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

// Atomic replace: a temporary file next to the record (its name ends in .tmp; never read), written in full, flushed, then
// moved over the record. Up to `attempts` tries (a reader or scanner can hold the record for a moment). false with *why.
inline bool WriteRecordFile(const std::filesystem::path& path, const Record& r, std::string* why, int attempts = 3) {
    const std::string text = FormatRecord(r);
    std::string last;
    for (int i = 0; i < attempts; ++i) {
#ifdef _WIN32
        const std::wstring tmp = path.wstring() + L"." + std::to_wstring(GetCurrentProcessId()) + L"." +
                                 std::to_wstring(GetTickCount64()) + L"." + std::to_wstring(i) + L".tmp";
        HANDLE h = CreateFileW(tmp.c_str(), GENERIC_WRITE | DELETE, 0, nullptr, CREATE_NEW, FILE_ATTRIBUTE_NORMAL, nullptr);
        if (h == INVALID_HANDLE_VALUE) {
            last = "cannot create the temporary file (Windows error " + std::to_string(GetLastError()) + ")";
        } else {
            DWORD written = 0;
            const bool wrote = WriteFile(h, text.data(), DWORD(text.size()), &written, nullptr) && written == text.size();
            const bool flushed = wrote && FlushFileBuffers(h);
            bool moved = false;
            if (wrote && flushed) {
                // Rename through the handle with POSIX semantics (Windows 10 1809+, NTFS): the record is replaced also while
                // a reader holds it open (readers open it with FILE_SHARE_DELETE and keep the old one); then flushed.
                const std::wstring target = path.wstring();
                std::vector<unsigned char> buf(sizeof(FILE_RENAME_INFO) + target.size() * sizeof(wchar_t));
                auto* info = reinterpret_cast<FILE_RENAME_INFO*>(buf.data());
                const DWORD flags = 0x1 /* FILE_RENAME_FLAG_REPLACE_IF_EXISTS */ | 0x2 /* FILE_RENAME_FLAG_POSIX_SEMANTICS */;
                std::memcpy(buf.data(), &flags, sizeof flags);  // the union { ReplaceIfExists; Flags } at offset 0
                info->RootDirectory = nullptr;
                info->FileNameLength = DWORD(target.size() * sizeof(wchar_t));
                std::memcpy(info->FileName, target.c_str(), target.size() * sizeof(wchar_t));
                moved = SetFileInformationByHandle(h, static_cast<FILE_INFO_BY_HANDLE_CLASS>(22) /* FileRenameInfoEx */, info,
                                                   DWORD(buf.size())) != 0;
                if (moved) FlushFileBuffers(h);
            }
            CloseHandle(h);
            if (!wrote || !flushed) {
                last = wrote ? "cannot flush the temporary file" : "cannot write the temporary file";
                DeleteFileW(tmp.c_str());
            } else if (moved || MoveFileExW(tmp.c_str(), path.c_str(), MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH)) {
                return true;  // (MoveFileExW: older Windows or file systems without the POSIX rename)
            } else {
                last = "cannot replace the record (Windows error " + std::to_string(GetLastError()) + ")";
                DeleteFileW(tmp.c_str());
            }
        }
        Sleep(20);
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

}  // namespace wgchoice
