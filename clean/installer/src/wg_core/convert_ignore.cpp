// SPDX-License-Identifier: MIT
// WrongGPU launcher core: the ignore lists of the converter's file search (convert_ignore.h). Written for this project.
#include "convert_ignore.h"

#include <cstring>

namespace wg {

bool FileIdent::operator==(const FileIdent& o) const {
    return valid && o.valid && volume == o.volume && std::memcmp(id, o.id, sizeof(id)) == 0;
}

bool IgnoreLists::hasHashes() const {
    for (auto& e : entries) if (!e.sha256.empty()) return true;
    return false;
}

static wstring TrimW(const wstring& s) {
    size_t a = 0, b = s.size();
    while (a < b && (s[a] == L' ' || s[a] == L'\t' || s[a] == L'\r' || s[a] == L'\n' || s[a] == 0xFEFF)) ++a;
    while (b > a && (s[b - 1] == L' ' || s[b - 1] == L'\t' || s[b - 1] == L'\r' || s[b - 1] == L'\n')) --b;
    return s.substr(a, b - a);
}
static bool SameText(const wstring& a, const wstring& b) {  // ordinal, case-insensitive (file names)
    return CompareStringOrdinal(a.c_str(), (int)a.size(), b.c_str(), (int)b.size(), TRUE) == CSTR_EQUAL;
}
static bool IsHex64(const wstring& h) {
    if (h.size() != 64) return false;
    for (wchar_t c : h)
        if (!((c >= L'0' && c <= L'9') || (c >= L'a' && c <= L'f') || (c >= L'A' && c <= L'F'))) return false;
    return true;
}
// "\\?\C:\x" -> "C:\x", "\\?\UNC\server\share\x" -> "\\server\share\x"
static wstring StripLongPrefix(const wstring& p) {
    if (p.rfind(L"\\\\?\\UNC\\", 0) == 0) return L"\\\\" + p.substr(8);
    if (p.rfind(L"\\\\?\\", 0) == 0) return p.substr(4);
    return p;
}
// A full path: "X:\..." / "X:/..." or "\\server\share..." / "//server/share...", without wildcards or quotes inside.
static bool IsFullPath(const wstring& p) {
    if (p.find_first_of(L"*?\"<>|") != wstring::npos) return false;
    if (p.size() >= 3 && ((p[0] >= L'A' && p[0] <= L'Z') || (p[0] >= L'a' && p[0] <= L'z')) && p[1] == L':' && (p[2] == L'\\' || p[2] == L'/'))
        return p.find(L':', 2) == wstring::npos;
    if (p.size() >= 5 && (p[0] == L'\\' || p[0] == L'/') && (p[1] == L'\\' || p[1] == L'/') && p[2] != L'?' && p[2] != L'.')
        return p.find(L':') == wstring::npos;
    return false;
}

wstring FinalPath(const wstring& p, FileIdent* ident) {
    if (ident) *ident = FileIdent();
    // no access rights needed for the name and the id; links are followed (no FILE_FLAG_OPEN_REPARSE_POINT)
    HANDLE h = CreateFileW(p.c_str(), FILE_READ_ATTRIBUTES, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, nullptr, OPEN_EXISTING,
                           FILE_FLAG_BACKUP_SEMANTICS, nullptr);
    if (h == INVALID_HANDLE_VALUE) return L"";
    wstring out;
    DWORD n = GetFinalPathNameByHandleW(h, nullptr, 0, FILE_NAME_NORMALIZED | VOLUME_NAME_DOS);
    if (n) {
        wstring b(n, L'\0');
        const DWORD got = GetFinalPathNameByHandleW(h, &b[0], n, FILE_NAME_NORMALIZED | VOLUME_NAME_DOS);
        if (got && got < n) { b.resize(got); out = StripLongPrefix(b); }
    }
    if (ident) {
        FILE_ID_INFO fi{};
        if (GetFileInformationByHandleEx(h, FileIdInfo, &fi, sizeof(fi))) {
            ident->volume = fi.VolumeSerialNumber;
            std::memcpy(ident->id, fi.FileId.Identifier, sizeof(ident->id));
            ident->valid = true;
        } else {
            BY_HANDLE_FILE_INFORMATION bi{};
            if (GetFileInformationByHandle(h, &bi)) {
                ident->volume = bi.dwVolumeSerialNumber;
                const unsigned long long idx = ((unsigned long long)bi.nFileIndexHigh << 32) | bi.nFileIndexLow;
                std::memcpy(ident->id, &idx, sizeof(idx));
                ident->valid = true;
            }
        }
    }
    CloseHandle(h);
    return out;
}

bool ParseIgnoreLine(const wstring& raw, IgnoreEntry& e, wstring& why, int* version) {
    e = IgnoreEntry();
    const wstring line = TrimW(raw);
    if (line.empty() || line[0] == L'#') return true;
    if (StartsWithI(line, kIgnoreFormatTag)) {
        const wstring v = TrimW(line.substr(wcslen(kIgnoreFormatTag)));
        if (v.empty() || v.find_first_not_of(L"0123456789") != wstring::npos || v.size() > 6) {
            why = L"a format version line needs a number (\"" + wstring(kIgnoreFormatTag) + L" 1\")";
            return false;
        }
        if (version) *version = _wtoi(v.c_str());
        return true;
    }
    // "sha256=" at the start (a hash alone), or after whitespace that ends the path
    static const wstring kSha = L"sha256=";
    wstring path, hash;
    bool hasHash = false;
    if (StartsWithI(line, kSha)) {
        hash = TrimW(line.substr(kSha.size()));
        hasHash = true;
    } else {
        const wstring low = Lower(line);
        size_t at = wstring::npos;
        for (size_t p = low.find(kSha); p != wstring::npos; p = low.find(kSha, p + 1))
            if (p > 0 && (low[p - 1] == L' ' || low[p - 1] == L'\t')) at = p;
        if (at != wstring::npos) {
            path = TrimW(line.substr(0, at));
            hash = TrimW(line.substr(at + kSha.size()));
            hasHash = true;
        } else {
            path = line;
        }
    }
    if (hasHash && !IsHex64(hash)) { why = L"sha256= needs the file's SHA-256 as 64 hex digits"; return false; }
    if (path.size() >= 2 && path.front() == L'"' && path.back() == L'"') path = TrimW(path.substr(1, path.size() - 2));
    path = StripLongPrefix(path);  // "\\?\C:\..." and "\\?\UNC\server\share\..." are full paths too
    if (!hasHash || !path.empty()) {
        if (path.empty()) { why = L"no path"; return false; }
        if (!IsFullPath(path)) { why = L"not a full path (a drive letter or \\\\server\\share, no wildcards): " + path; return false; }
        e.path = StripLongPrefix(FullPath(path));  // '/', '.', '..' and repeated separators normalised
        e.resolved = FinalPath(e.path, &e.ident);
        if (e.resolved.empty()) e.resolved = e.path;
    }
    e.sha256 = Lower(hash);
    e.text = line;
    return true;
}

void ParseIgnoreText(const string& bytes, const wstring& source, IgnoreLists& into) {
    wstring text;
    if (bytes.size() >= 2 && (unsigned char)bytes[0] == 0xFF && (unsigned char)bytes[1] == 0xFE) {  // UTF-16 LE with its BOM
        text.resize((bytes.size() - 2) / 2);
        if (!text.empty()) std::memcpy(&text[0], bytes.data() + 2, text.size() * 2);
    } else {
        const size_t skip = bytes.size() >= 3 && (unsigned char)bytes[0] == 0xEF && (unsigned char)bytes[1] == 0xBB && (unsigned char)bytes[2] == 0xBF ? 3 : 0;
        text = FromUtf8(bytes.substr(skip));
    }
    int lineNo = 0, version = 0;
    bool newerSaid = false;
    for (size_t a = 0; a <= text.size();) {
        size_t e = text.find(L'\n', a);
        if (e == wstring::npos) e = text.size();
        const wstring line = text.substr(a, e - a);
        a = e + 1;
        ++lineNo;
        IgnoreEntry entry;
        wstring why;
        const wstring where = source + L", line " + std::to_wstring(lineNo);
        if (!ParseIgnoreLine(line, entry, why, &version)) {
            into.warnings.push_back(where + L": " + why + L"; the line is skipped");
            continue;
        }
        if (version > kIgnoreFormatVersion && !newerSaid) {
            newerSaid = true;
            into.warnings.push_back(where + L": format version " + std::to_wstring(version) + L" is newer than this launcher reads (" +
                                    std::to_wstring(kIgnoreFormatVersion) + L"); its lines are read as version " + std::to_wstring(kIgnoreFormatVersion));
        }
        if (entry.text.empty()) continue;
        entry.where = where;
        into.entries.push_back(entry);
    }
}

// The list's bytes. A sharing violation (an installer writing the list at this moment) is waited for: 5 tries 200 ms apart.
static bool ReadListBytes(const wstring& file, string& out, wstring& why) {
    const DWORD attrs = GetFileAttributesW(file.c_str());
    if (attrs == INVALID_FILE_ATTRIBUTES) { why = L"there is no such file"; return false; }
    if (attrs & FILE_ATTRIBUTE_DIRECTORY) { why = L"it is a folder, not a file"; return false; }
    HANDLE h = INVALID_HANDLE_VALUE;
    DWORD err = 0;
    for (int i = 0; i < kIgnoreReadTries; ++i) {
        if (i) Sleep(kIgnoreReadWaitMs);
        // FILE_SHARE_READ only: while another program has it open for writing, this waits instead of reading half a list
        h = CreateFileW(file.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
        if (h != INVALID_HANDLE_VALUE) break;
        err = GetLastError();
        if (err != ERROR_SHARING_VIOLATION && err != ERROR_LOCK_VIOLATION) break;
    }
    if (h == INVALID_HANDLE_VALUE) {
        why = (err == ERROR_SHARING_VIOLATION || err == ERROR_LOCK_VIOLATION
                   ? L"another program holds it open; tried " + std::to_wstring(kIgnoreReadTries) + L" times, " + std::to_wstring(kIgnoreReadWaitMs) + L" ms apart: "
                   : wstring()) + ErrText(err);
        while (!why.empty() && (why.back() == L'\n' || why.back() == L'\r' || why.back() == L' ' || why.back() == L'.')) why.pop_back();
        return false;
    }
    LARGE_INTEGER sz{};
    bool ok = GetFileSizeEx(h, &sz) && sz.QuadPart < (16ll << 20);
    if (!ok) why = L"it is larger than 16 MB";
    if (ok) {
        out.resize((size_t)sz.QuadPart);
        DWORD got = 0;
        ok = out.empty() || (ReadFile(h, &out[0], (DWORD)out.size(), &got, nullptr) && got == out.size());
        if (!ok) why = L"reading it failed: " + ErrText(GetLastError());
    }
    CloseHandle(h);
    return ok;
}

bool ReadIgnoreList(const wstring& file, IgnoreLists& into, wstring* why) {
    string bytes;
    wstring w;
    if (!ReadListBytes(file, bytes, w)) { if (why) *why = w; return false; }
    ParseIgnoreText(bytes, file, into);
    into.lists.push_back(file);
    return true;
}

std::vector<wstring> WellKnownIgnoreLists(const wstring& ownDataDir) {
    std::vector<wstring> r{Join(AppDataPath(), kIgnoreListName)};
    if (!ownDataDir.empty()) {
        const wstring own = Join(ownDataDir, kIgnoreListName);
        if (!SameText(StripLongPrefix(FullPath(own)), StripLongPrefix(FullPath(r[0])))) r.push_back(own);
    }
    return r;
}

// "on the ignore list <file>, line <n>" or "named by --ignore=<entry>": which line or argument matched
static wstring Source(const IgnoreEntry& e) {
    return StartsWithI(e.where, L"--ignore") ? L"named by " + e.where : L"on the ignore list " + e.where;
}
// The path entries only (no file is read).
static wstring PathReason(const IgnoreLists& l, const wstring& path) {
    FileIdent id;
    wstring fin = FinalPath(path, &id);
    if (fin.empty()) fin = StripLongPrefix(FullPath(path));
    for (auto& e : l.entries) {
        if (e.path.empty()) continue;
        if (SameText(fin, e.resolved) || SameText(fin, e.path)) return Source(e) + L" (path " + e.path + L")";
        if (id == e.ident) return Source(e) + L" (path " + e.path + L": the same file)";
    }
    return L"";
}
static wstring HashReason(const IgnoreLists& l, const wstring& sha) {
    if (sha.empty()) return L"";
    for (auto& e : l.entries)
        if (!e.sha256.empty() && e.sha256 == Lower(sha)) return Source(e) + L" (SHA-256 " + e.sha256 + L")";
    return L"";
}

wstring IgnoreReason(const IgnoreLists& l, const wstring& candidate) {
    if (l.empty()) return L"";
    const wstring byPath = PathReason(l, candidate);
    if (!byPath.empty() || !l.hasHashes()) return byPath;
    return HashReason(l, Sha256File(candidate));
}

wstring RecordedSourceReason(const IgnoreLists& l, const wstring& source, const wstring& sha) {
    if (l.empty()) return L"";
    const wstring byPath = source.empty() ? wstring() : PathReason(l, source);
    return byPath.empty() ? HashReason(l, sha) : byPath;
}

wstring SourceField(const wstring& file, const wchar_t* key) {
    string raw;
    if (!ReadAll(file, raw)) return L"";
    const wstring t = FromUtf8(raw), k = wstring(key) + L"\t";
    for (size_t a = 0; a < t.size();) {
        size_t e = t.find(L'\n', a);
        if (e == wstring::npos) e = t.size();
        wstring line = t.substr(a, e - a);
        a = e + 1;
        while (!line.empty() && (line.back() == L'\r' || line.back() == L' ')) line.pop_back();
        if (line.rfind(k, 0) == 0) return line.substr(k.size());
    }
    return L"";
}

IgnoreLists LoadConvertIgnore(Context& ctx, const wchar_t* what, IgnoreReport* rep) {
    IgnoreLists l;
    wstring why;
    auto unreadable = [&](const wstring& f) {
        // fail closed: a list that is there but can't be read might name the very file the search would take
        if (l.unreadable.empty()) l.unreadable = L"The ignore list " + f + L" can't be read (" + why + L"), so nothing was converted. Close the program "
                                                 L"that holds it, or fix the file, and try again.";
        LogLine(ctx, wstring(what) + L": the ignore list " + f + L" can't be read: " + why);
    };
    for (auto& f : WellKnownIgnoreLists(ctx.ownDataDir))
        if (GetFileAttributesW(f.c_str()) != INVALID_FILE_ATTRIBUTES && !ReadIgnoreList(f, l, &why)) unreadable(f);
    for (auto& f : ctx.ignoreListFiles)
        if (!ReadIgnoreList(f, l, &why)) unreadable(f);
    for (auto& s : ctx.ignoreEntries) {
        IgnoreEntry e;
        if (!ParseIgnoreLine(s, e, why)) { l.warnings.push_back(L"--ignore=" + s + L": " + why + L"; skipped"); continue; }
        if (e.text.empty()) continue;
        e.where = L"--ignore=" + s;
        l.entries.push_back(e);
    }
    for (auto& w : l.warnings) {
        LogLine(ctx, wstring(what) + L": warning: " + w);
        if (rep) rep->warnings.push_back(w);
    }
    if (rep) rep->unreadable = l.unreadable;
    if (!l.empty()) {
        wstring from;
        for (auto& f : l.lists) from += (from.empty() ? L"" : L", ") + f;
        if (!ctx.ignoreEntries.empty()) from += from.empty() ? L"--ignore" : L", --ignore";
        LogLine(ctx, wstring(what) + L": " + std::to_wstring(l.entries.size()) + L" ignore entr" + (l.entries.size() == 1 ? L"y" : L"ies") + L" (" + from + L")");
    }
    return l;
}

std::vector<bool> IgnoredMask(Context& ctx, const IgnoreLists& l, const std::vector<wstring>& paths, const wchar_t* what, IgnoreReport& rep) {
    std::vector<bool> skip(paths.size(), false);
    if (l.empty()) return skip;
    for (size_t i = 0; i < paths.size(); ++i) {
        const wstring why = IgnoreReason(l, paths[i]);
        if (why.empty()) continue;
        skip[i] = true;
        rep.skipped.push_back(paths[i] + L": " + why);
        LogLine(ctx, wstring(what) + L": skipped " + paths[i] + L": " + why);
    }
    return skip;
}

wstring PutIgnoreListBack(const wstring& moved, const wstring& shared) {
    const wstring from = Join(moved, kIgnoreListName), to = Join(shared, kIgnoreListName);
    const DWORD a = GetFileAttributesW(from.c_str());
    if (a == INVALID_FILE_ATTRIBUTES || (a & (FILE_ATTRIBUTE_DIRECTORY | FILE_ATTRIBUTE_REPARSE_POINT))) return L"";
    CreateDirectoryW(shared.c_str(), nullptr);
    if (!MoveFileExW(from.c_str(), to.c_str(), MOVEFILE_WRITE_THROUGH)) {  // another volume can't happen (a rename beside it); copy as a fallback
        if (!CopyFileW(from.c_str(), to.c_str(), TRUE)) return L"";
    }
    return wstring(kKeptIgnoreListText) + L": " + to;
}

}  // namespace wg
