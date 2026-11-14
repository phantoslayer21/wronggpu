// SPDX-License-Identifier: MIT
// WrongGPU launcher core: small helpers (strings, paths, files, SHA-256), moved unchanged from setup.cpp. Written for
// this project.
#include "util.h"

namespace wg {

[[noreturn]] void Fail(const wstring& m, int code) { throw SetupError{m + L" [WG-SETUP-" + std::to_wstring(code) + L"]", code}; }

wstring Lower(wstring s) { for (auto& c : s) c = (wchar_t)towlower(c); return s; }
bool IEq(const wstring& a, const wstring& b) { return _wcsicmp(a.c_str(), b.c_str()) == 0; }
bool Contains(const wstring& hay, const wchar_t* needle) { return Lower(hay).find(Lower(needle)) != wstring::npos; }
bool EndsWithI(const wstring& s, const wstring& suffix) { return s.size() >= suffix.size() && IEq(s.substr(s.size() - suffix.size()), suffix); }
bool StartsWithI(const wstring& s, const wstring& prefix) { return s.size() >= prefix.size() && IEq(s.substr(0, prefix.size()), prefix); }
wstring Join(const wstring& a, const wstring& b) {
    if (a.empty()) return b;
    wstring r = a;
    if (r.back() != L'\\' && r.back() != L'/') r += L'\\';
    return r + b;
}
wstring Parent(const wstring& p) { auto i = p.find_last_of(L"\\/"); return i == wstring::npos ? wstring() : p.substr(0, i); }
wstring BaseName(const wstring& p) { auto i = p.find_last_of(L"\\/"); return i == wstring::npos ? p : p.substr(i + 1); }
wstring Stem(const wstring& p) { wstring b = BaseName(p); auto i = b.find_last_of(L'.'); return i == wstring::npos ? b : b.substr(0, i); }
wstring ShortenMiddle(const wstring& s, size_t max) {
    if (s.size() <= max || max < 8) return s;
    size_t head = (max - 1) * 2 / 5, tail = max - 1 - head;
    const size_t hs = s.find_last_of(L"\\/", head - 1);
    if (hs != wstring::npos && hs + 1 >= head / 2) head = hs + 1;  // keep "G:\Games\"
    const size_t from = s.size() - tail, ts = s.find_first_of(L"\\/", from);
    if (ts != wstring::npos && ts - from <= tail / 2) tail = s.size() - ts;  // and "\Game\BattlEye"
    return s.substr(0, head) + L"\u2026" + s.substr(s.size() - tail);
}
bool Exists(const wstring& p) { return GetFileAttributesW(p.c_str()) != INVALID_FILE_ATTRIBUTES; }
bool IsDir(const wstring& p) { DWORD a = GetFileAttributesW(p.c_str()); return a != INVALID_FILE_ATTRIBUTES && (a & FILE_ATTRIBUTE_DIRECTORY); }
bool IsFile(const wstring& p) { DWORD a = GetFileAttributesW(p.c_str()); return a != INVALID_FILE_ATTRIBUTES && !(a & FILE_ATTRIBUTE_DIRECTORY); }
unsigned long long FileSize(const wstring& p) {
    WIN32_FILE_ATTRIBUTE_DATA d{};
    if (!GetFileAttributesExW(p.c_str(), GetFileExInfoStandard, &d)) return 0;
    return ((unsigned long long)d.nFileSizeHigh << 32) | d.nFileSizeLow;
}
wstring FullPath(const wstring& p) {
    wchar_t b[MAX_PATH * 2]; DWORD n = GetFullPathNameW(p.c_str(), (DWORD)std::size(b), b, nullptr);
    return n && n < std::size(b) ? wstring(b, n) : p;
}
wstring Hex32(DWORD v) { wchar_t b[16]; swprintf(b, 16, L"0x%08X", (unsigned)v); return b; }
wstring ErrText(DWORD e) {
    wchar_t* b = nullptr;
    FormatMessageW(FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS, nullptr, e, 0, (LPWSTR)&b, 0, nullptr);
    wstring s = b ? b : L"Unknown error";
    if (b) LocalFree(b);
    while (!s.empty() && (s.back() == L'\r' || s.back() == L'\n' || s.back() == L' ')) s.pop_back();
    return s + L" (" + Hex32(e) + L")";
}
string ToUtf8(const wstring& w) {
    if (w.empty()) return {};
    int n = WideCharToMultiByte(CP_UTF8, 0, w.data(), (int)w.size(), nullptr, 0, nullptr, nullptr);
    string s(n, 0);
    WideCharToMultiByte(CP_UTF8, 0, w.data(), (int)w.size(), s.data(), n, nullptr, nullptr);
    return s;
}
wstring FromUtf8(const string& s) {
    if (s.empty()) return {};
    int n = MultiByteToWideChar(CP_UTF8, 0, s.data(), (int)s.size(), nullptr, 0);
    wstring w(n, 0);
    MultiByteToWideChar(CP_UTF8, 0, s.data(), (int)s.size(), w.data(), n);
    return w;
}
bool ReadAll(const wstring& path, string& out) {
    // FILE_SHARE_DELETE: a read never stops another thread's move or delete of the file (the launcher's data check hashing
    // the model file while a conversion swaps it); the read goes on from the handle
    HANDLE h = CreateFileW(path.c_str(), GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (h == INVALID_HANDLE_VALUE) return false;
    LARGE_INTEGER sz{};
    GetFileSizeEx(h, &sz);
    out.resize((size_t)sz.QuadPart);
    size_t done = 0;
    while (done < out.size()) {
        DWORD got = 0;
        if (!ReadFile(h, out.data() + done, (DWORD)std::min<size_t>(out.size() - done, 1u << 24), &got, nullptr) || !got) break;
        done += got;
    }
    CloseHandle(h);
    return done == out.size();
}
// Case-insensitive search for an ASCII needle, as ASCII or as UTF-16LE text.
bool ContainsText(const uint8_t* d, size_t n, const char* needle) {
    const size_t k = strlen(needle);
    if (!d || !k || n < k) return false;
    auto low = [](uint8_t c) { return (uint8_t)(c >= 'A' && c <= 'Z' ? c + 32 : c); };
    const uint8_t first = low((uint8_t)needle[0]);
    for (size_t i = 0; i + k <= n; ++i) {
        if (low(d[i]) != first) continue;
        size_t j = 1;
        while (j < k && low(d[i + j]) == low((uint8_t)needle[j])) ++j;
        if (j == k) return true;
        if (i + 2 * k <= n && d[i + 1] == 0) {  // UTF-16LE candidate
            j = 1;
            while (j < k && low(d[i + 2 * j]) == low((uint8_t)needle[j]) && d[i + 2 * j + 1] == 0) ++j;
            if (j == k) return true;
        }
    }
    return false;
}
bool FileContainsText(const wstring& path, const char* needle, unsigned long long limit) {
    MappedFile f(path, limit);
    return ContainsText(f.data, f.size, needle);
}
void EnsureDir(const wstring& dir, std::vector<wstring>* created) {
    if (dir.empty() || IsDir(dir)) return;
    if (dir.size() <= 3 && dir.size() >= 2 && dir[1] == L':') return;
    EnsureDir(Parent(dir), created);
    if (!CreateDirectoryW(dir.c_str(), nullptr)) {
        DWORD e = GetLastError();
        if (e == ERROR_ALREADY_EXISTS && IsDir(dir)) return;
        if (e == ERROR_ALREADY_EXISTS) Fail(L"A file is in the way of the folder " + dir + L". Remove or rename it and run setup again.");
        Fail(L"Cannot create folder " + dir + L": " + ErrText(e));
    }
    if (created) created->push_back(dir);
}
void ForceDelete(const wstring& p) {
    SetFileAttributesW(p.c_str(), FILE_ATTRIBUTE_NORMAL);
    if (!DeleteFileW(p.c_str()) && GetLastError() != ERROR_FILE_NOT_FOUND) Fail(L"Cannot delete " + p + L": " + ErrText(GetLastError()));
}
void WriteFileAtomic(const wstring& path, const void* data, size_t size) {
    wstring tmp = path + kTmpSuffix;
    HANDLE h = CreateFileW(tmp.c_str(), GENERIC_WRITE, 0, nullptr, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (h == INVALID_HANDLE_VALUE) Fail(L"Cannot write " + path + L": " + ErrText(GetLastError()));
    const char* p = static_cast<const char*>(data);
    size_t done = 0;
    while (done < size) {
        DWORD wrote = 0;
        if (!WriteFile(h, p + done, (DWORD)std::min<size_t>(size - done, 1u << 24), &wrote, nullptr)) {
            DWORD e = GetLastError(); CloseHandle(h); DeleteFileW(tmp.c_str());
            Fail(L"Write failed for " + path + L": " + ErrText(e));
        }
        done += wrote;
    }
    FlushFileBuffers(h);
    CloseHandle(h);
    if (Exists(path)) SetFileAttributesW(path.c_str(), FILE_ATTRIBUTE_NORMAL);
    if (!MoveFileExW(tmp.c_str(), path.c_str(), MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH)) {
        DWORD e = GetLastError(); DeleteFileW(tmp.c_str());
        Fail(L"Cannot replace " + path + L": " + ErrText(e) + L". Is the file in use?");
    }
}
void CopyChecked(const wstring& src, const wstring& dst) {
    EnsureDir(Parent(dst));
    if (Exists(dst)) SetFileAttributesW(dst.c_str(), FILE_ATTRIBUTE_NORMAL);
    if (!CopyFileW(src.c_str(), dst.c_str(), FALSE)) Fail(L"Cannot copy " + src + L" to " + dst + L": " + ErrText(GetLastError()));
}
bool CopyNew(const wstring& src, const wstring& dst, DWORD* err) {
    if (CopyFileW(src.c_str(), dst.c_str(), TRUE)) return true;  // TRUE: fail when dst exists (creation, not a check)
    if (err) *err = GetLastError();
    return false;
}

// ---- PinnedFile (util.h) ----------------------------------------------------------------------------------------------
namespace {
// FILE_DISPOSITION_INFO_EX and FileDispositionInfoEx (Windows 10 1709+) are declared here so older SDK and MinGW headers build.
struct DispositionInfoEx { DWORD Flags; };
constexpr DWORD kDispositionDelete = 0x1, kDispositionPosix = 0x2, kDispositionIgnoreReadOnly = 0x10;
constexpr int kFileDispositionInfoEx = 21;
wstring HashHandle(HANDLE h) {
    LARGE_INTEGER zero{};
    if (!SetFilePointerEx(h, zero, nullptr, FILE_BEGIN)) return L"";
    wgc::Sha256 sha;
    std::vector<char> buf(1u << 20);
    for (;;) {
        DWORD n = 0;
        if (!ReadFile(h, buf.data(), (DWORD)buf.size(), &n, nullptr)) return L"";
        if (n == 0) break;
        sha.update(buf.data(), n);
    }
    return HexW(sha.finish());
}
}  // namespace

namespace {
// The attributes a person sets on a file; what is put back when a read-only bit that was cleared for a write is given up.
constexpr DWORD kUserAttrs = FILE_ATTRIBUTE_READONLY | FILE_ATTRIBUTE_HIDDEN | FILE_ATTRIBUTE_SYSTEM | FILE_ATTRIBUTE_ARCHIVE;
bool SetAttrsByHandle(HANDLE h, DWORD attrs) {
    FILE_BASIC_INFO b{};  // (times 0: unchanged)
    b.FileAttributes = (attrs & kUserAttrs) ? (attrs & kUserAttrs) : FILE_ATTRIBUTE_NORMAL;
    return SetFileInformationByHandle(h, FileBasicInfo, &b, sizeof(b)) != 0;
}
}  // namespace

bool PinnedFile::Open(const wstring& path, bool forWrite, DWORD* err) {
    Close();
    auto fail = [&](DWORD e) { if (err) *err = e; return false; };
    // A probe first: attribute access only, every sharing mode (it neither blocks nor is blocked), the link itself and not its
    // target. What the file is (a folder, a link, hard-linked) is known before anything about it is changed.
    HANDLE probe = CreateFileW(path.c_str(), FILE_READ_ATTRIBUTES | FILE_WRITE_ATTRIBUTES, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, nullptr,
                               OPEN_EXISTING, FILE_FLAG_OPEN_REPARSE_POINT | FILE_FLAG_BACKUP_SEMANTICS, nullptr);
    if (probe == INVALID_HANDLE_VALUE) return fail(GetLastError());
    struct Closer { HANDLE h; ~Closer() { CloseHandle(h); } } closeProbe{probe};
    BY_HANDLE_FILE_INFORMATION pi{};
    if (!GetFileInformationByHandle(probe, &pi)) return fail(GetLastError());
    if (pi.dwFileAttributes & (FILE_ATTRIBUTE_DIRECTORY | FILE_ATTRIBUTE_REPARSE_POINT)) return fail(ERROR_NOT_SUPPORTED);
    if (forWrite && pi.nNumberOfLinks > 1) return fail(ERROR_TOO_MANY_LINKS);  // another name for these bytes: never written through
    // A read-only file to be written: the bit is cleared through the probe, and put back if the file is not written after all
    // (the open fails, it is another file, or the caller leaves it: Close).
    const bool readOnly = forWrite && (pi.dwFileAttributes & FILE_ATTRIBUTE_READONLY);
    if (readOnly && !SetAttrsByHandle(probe, pi.dwFileAttributes & ~FILE_ATTRIBUTE_READONLY)) return fail(GetLastError());
    auto giveUp = [&](DWORD e) { if (readOnly) SetAttrsByHandle(probe, pi.dwFileAttributes); return fail(e); };
    const DWORD access = GENERIC_READ | DELETE | FILE_WRITE_ATTRIBUTES | (forWrite ? GENERIC_WRITE : 0);
    h_ = CreateFileW(path.c_str(), access, FILE_SHARE_READ, nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL | FILE_FLAG_OPEN_REPARSE_POINT, nullptr);
    if (h_ == INVALID_HANDLE_VALUE) return giveUp(GetLastError());
    BY_HANDLE_FILE_INFORMATION bi{};
    if (!GetFileInformationByHandle(h_, &bi)) { const DWORD e = GetLastError(); Close(); return giveUp(e); }
    const bool same = bi.dwVolumeSerialNumber == pi.dwVolumeSerialNumber && bi.nFileIndexHigh == pi.nFileIndexHigh && bi.nFileIndexLow == pi.nFileIndexLow;
    if (!same || (bi.dwFileAttributes & (FILE_ATTRIBUTE_DIRECTORY | FILE_ATTRIBUTE_REPARSE_POINT)) || (forWrite && bi.nNumberOfLinks > 1)) {
        Close();
        return giveUp(ERROR_NOT_SUPPORTED);  // another file than the probe saw (swapped in between), or now a link
    }
    restoreAttrs_ = readOnly ? pi.dwFileAttributes : 0;
    return true;
}
void PinnedFile::Close() {
    if (h_ != INVALID_HANDLE_VALUE) {
        if (restoreAttrs_) SetAttrsByHandle(h_, restoreAttrs_);  // the file was not changed: its read-only bit is as it was
        CloseHandle(h_);
    }
    h_ = INVALID_HANDLE_VALUE;
    restoreAttrs_ = 0;
}
wstring PinnedFile::Sha256() { return h_ == INVALID_HANDLE_VALUE ? wstring() : HashHandle(h_); }
bool PinnedFile::Delete(DWORD* err) {
    if (h_ == INVALID_HANDLE_VALUE) { if (err) *err = ERROR_INVALID_HANDLE; return false; }
    DispositionInfoEx ex{kDispositionDelete | kDispositionPosix | kDispositionIgnoreReadOnly};
    bool ok = SetFileInformationByHandle(h_, static_cast<FILE_INFO_BY_HANDLE_CLASS>(kFileDispositionInfoEx), &ex, sizeof(ex)) != 0;
    DWORD e = ok ? 0 : GetLastError();
    if (!ok) {  // no POSIX delete here (an older Windows, another file system, Wine): the classic disposition, read-only cleared first
        FILE_BASIC_INFO basic{};
        bool cleared = false;
        if (GetFileInformationByHandleEx(h_, FileBasicInfo, &basic, sizeof(basic)) && (basic.FileAttributes & FILE_ATTRIBUTE_READONLY))
            cleared = SetAttrsByHandle(h_, basic.FileAttributes & ~FILE_ATTRIBUTE_READONLY);
        FILE_DISPOSITION_INFO d{TRUE};
        ok = SetFileInformationByHandle(h_, FileDispositionInfo, &d, sizeof(d)) != 0;
        e = ok ? 0 : GetLastError();
        if (!ok && cleared) SetAttrsByHandle(h_, basic.FileAttributes);  // not deleted: its read-only bit is as it was
    }
    if (ok) restoreAttrs_ = 0;
    Close();  // (the classic disposition takes effect when the handle closes; nobody else can hold the file open for delete)
    if (!ok && err) *err = e;
    return ok;
}
bool PinnedFile::Overwrite(const wstring& src, const wstring& srcSha, DWORD* err) {
    auto fail = [&](DWORD e) { if (err) *err = e; return false; };
    if (h_ == INVALID_HANDLE_VALUE) return fail(ERROR_INVALID_HANDLE);
    HANDLE s = CreateFileW(src.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (s == INVALID_HANDLE_VALUE) return fail(GetLastError());
    struct Closer { HANDLE h; ~Closer() { CloseHandle(h); } } closer{s};
    if (!IEq(HashHandle(s), srcSha)) return fail(ERROR_FILE_CORRUPT);
    BY_HANDLE_FILE_INFORMATION si{};
    GetFileInformationByHandle(s, &si);
    LARGE_INTEGER zero{};
    if (!SetFilePointerEx(s, zero, nullptr, FILE_BEGIN) || !SetFilePointerEx(h_, zero, nullptr, FILE_BEGIN)) return fail(GetLastError());
    std::vector<char> buf(1u << 20);
    for (;;) {
        DWORD n = 0, w = 0;
        if (!ReadFile(s, buf.data(), (DWORD)buf.size(), &n, nullptr)) return fail(GetLastError());
        if (n == 0) break;
        if (!WriteFile(h_, buf.data(), n, &w, nullptr) || w != n) return fail(GetLastError());
    }
    if (!SetEndOfFile(h_)) return fail(GetLastError());
    FlushFileBuffers(h_);
    FILE_BASIC_INFO b{};  // the original's attributes and times (a restored file is the file it was)
    b.CreationTime.u.LowPart = si.ftCreationTime.dwLowDateTime; b.CreationTime.u.HighPart = (LONG)si.ftCreationTime.dwHighDateTime;
    b.LastWriteTime.u.LowPart = si.ftLastWriteTime.dwLowDateTime; b.LastWriteTime.u.HighPart = (LONG)si.ftLastWriteTime.dwHighDateTime;
    b.FileAttributes = si.dwFileAttributes & (FILE_ATTRIBUTE_READONLY | FILE_ATTRIBUTE_HIDDEN | FILE_ATTRIBUTE_SYSTEM | FILE_ATTRIBUTE_ARCHIVE);
    if (!b.FileAttributes) b.FileAttributes = FILE_ATTRIBUTE_NORMAL;
    SetFileInformationByHandle(h_, FileBasicInfo, &b, sizeof(b));  // (best effort: the content is what matters)
    restoreAttrs_ = 0;  // written: the source's attributes are on the file now
    return true;
}
bool PinnedFile::OverwriteBytes(const void* data, size_t size, DWORD* err) {
    if (h_ == INVALID_HANDLE_VALUE) { if (err) *err = ERROR_INVALID_HANDLE; return false; }
    LARGE_INTEGER zero{};
    if (!SetFilePointerEx(h_, zero, nullptr, FILE_BEGIN)) { if (err) *err = GetLastError(); return false; }
    const char* p = static_cast<const char*>(data);
    for (size_t done = 0; done < size;) {
        DWORD w = 0;
        if (!WriteFile(h_, p + done, (DWORD)std::min<size_t>(size - done, 1u << 24), &w, nullptr) || w == 0) { if (err) *err = GetLastError(); return false; }
        done += w;
    }
    if (!SetEndOfFile(h_)) { if (err) *err = GetLastError(); return false; }
    FlushFileBuffers(h_);
    restoreAttrs_ = 0;
    return true;
}
bool PinnedFile::Create(const wstring& path, DWORD* err) {
    Close();
    // CREATE_NEW fails on anything at the name (a file, a folder, a link: FILE_FLAG_OPEN_REPARSE_POINT, never its target)
    h_ = CreateFileW(path.c_str(), GENERIC_READ | GENERIC_WRITE | DELETE | FILE_WRITE_ATTRIBUTES, FILE_SHARE_READ, nullptr, CREATE_NEW,
                     FILE_ATTRIBUTE_NORMAL | FILE_FLAG_OPEN_REPARSE_POINT, nullptr);
    if (h_ == INVALID_HANDLE_VALUE) { if (err) *err = GetLastError(); return false; }
    restoreAttrs_ = 0;
    return true;
}
bool PinnedFile::ReadBytes(string& out) {
    out.clear();
    if (h_ == INVALID_HANDLE_VALUE) return false;
    LARGE_INTEGER zero{};
    if (!SetFilePointerEx(h_, zero, nullptr, FILE_BEGIN)) return false;
    std::vector<char> buf(1u << 16);
    for (;;) {
        DWORD n = 0;
        if (!ReadFile(h_, buf.data(), (DWORD)buf.size(), &n, nullptr)) return false;
        if (n == 0) return true;
        out.append(buf.data(), n);
    }
}
bool PinnedFile::CopyOut(const wstring& dst, wstring* sha, DWORD* err) {
    auto fail = [&](DWORD e) { if (err) *err = e; return false; };
    if (h_ == INVALID_HANDLE_VALUE) return fail(ERROR_INVALID_HANDLE);
    PinnedFile out;
    if (!out.Create(dst, err)) return false;
    LARGE_INTEGER zero{};
    if (!SetFilePointerEx(h_, zero, nullptr, FILE_BEGIN)) { const DWORD e = GetLastError(); out.Delete(); return fail(e); }
    wgc::Sha256 hash;
    std::vector<char> buf(1u << 20);
    for (;;) {
        DWORD n = 0, w = 0;
        if (!ReadFile(h_, buf.data(), (DWORD)buf.size(), &n, nullptr)) { const DWORD e = GetLastError(); out.Delete(); return fail(e); }
        if (n == 0) break;
        if (!WriteFile(out.h_, buf.data(), n, &w, nullptr) || w != n) { const DWORD e = GetLastError(); out.Delete(); return fail(e ? e : ERROR_WRITE_FAULT); }
        hash.update(buf.data(), n);
    }
    FlushFileBuffers(out.h_);
    if (sha) *sha = HexW(hash.finish());
    return true;
}
bool PinnedFile::SetReadOnly(bool on, DWORD* err) {
    if (h_ == INVALID_HANDLE_VALUE) { if (err) *err = ERROR_INVALID_HANDLE; return false; }
    FILE_BASIC_INFO b{};
    if (!GetFileInformationByHandleEx(h_, FileBasicInfo, &b, sizeof(b))) { if (err) *err = GetLastError(); return false; }
    const DWORD attrs = on ? (b.FileAttributes | FILE_ATTRIBUTE_READONLY) : (b.FileAttributes & ~FILE_ATTRIBUTE_READONLY);
    if (!SetAttrsByHandle(h_, attrs)) { if (err) *err = GetLastError(); return false; }
    restoreAttrs_ = 0;  // set on purpose: Close leaves it so
    return true;
}
bool WriteFileNew(const wstring& path, const void* data, size_t size, bool readOnly, DWORD* err) {
    PinnedFile f;
    if (!f.Create(path, err)) return false;
    if (!f.OverwriteBytes(data, size, err) || (readOnly && !f.SetReadOnly(true, err))) {
        f.Delete();  // (ours: created just now through this handle)
        return false;
    }
    return true;
}
DeleteIf DeleteIfSha(const wstring& path, const std::vector<wstring>& shas, DWORD* err) {
    PinnedFile f;
    DWORD e = 0;
    if (!f.Open(path, false, &e)) {
        if (e == ERROR_FILE_NOT_FOUND || e == ERROR_PATH_NOT_FOUND) return DeleteIf::Gone;
        if (err) *err = e;
        return e == ERROR_NOT_SUPPORTED ? DeleteIf::Changed : DeleteIf::Failed;
    }
    const wstring h = f.Sha256();
    if (h.empty()) { if (err) *err = ERROR_READ_FAULT; return DeleteIf::Failed; }
    if (std::none_of(shas.begin(), shas.end(), [&](const wstring& s) { return IEq(s, h); })) return DeleteIf::Changed;
    return f.Delete(err) ? DeleteIf::Deleted : DeleteIf::Failed;
}

namespace {
void WalkOwned(const wstring& root, const wstring& rel, const std::function<bool(const wstring&, PinnedFile&)>& ours, const wstring& keepFile, std::vector<wstring>& left) {
    const wstring dir = rel.empty() ? root : Join(root, rel);
    struct Item { wstring name; DWORD attrs; };
    std::vector<Item> items;
    WIN32_FIND_DATAW fd;
    HANDLE h = FindFirstFileW(Join(dir, L"*").c_str(), &fd);
    if (h != INVALID_HANDLE_VALUE) {
        do {
            const wstring n = fd.cFileName;
            if (n != L"." && n != L"..") items.push_back({n, fd.dwFileAttributes});
        } while (FindNextFileW(h, &fd));
        FindClose(h);
    }
    for (auto& it : items) {
        const wstring relN = rel.empty() ? it.name : rel + L"\\" + it.name, p = Join(root, relN);
        if (it.attrs & FILE_ATTRIBUTE_REPARSE_POINT) { left.push_back(relN); continue; }  // a link: not followed, not removed
        if (it.attrs & FILE_ATTRIBUTE_DIRECTORY) {
            WalkOwned(root, relN, ours, keepFile, left);
            RemoveDirectoryW(p.c_str());  // only when empty
            continue;
        }
        if (!keepFile.empty() && IEq(FullPath(p), FullPath(keepFile))) continue;
        PinnedFile f;
        DWORD e = 0;
        if (!f.Open(p, false, &e) || !ours(relN, f) || !f.Delete(&e)) left.push_back(relN);  // (ours: by name, and by the bytes where a hash is known)
    }
}
}  // namespace
std::vector<wstring> RemoveOwned(const wstring& root, const std::function<bool(const wstring&, PinnedFile&)>& ours, const wstring& keepFile) {
    std::vector<wstring> left;
    const DWORD a = GetFileAttributesW(root.c_str());
    if (a == INVALID_FILE_ATTRIBUTES || (a & FILE_ATTRIBUTE_REPARSE_POINT) || !(a & FILE_ATTRIBUTE_DIRECTORY)) return left;
    WalkOwned(root, L"", ours, keepFile, left);
    RemoveDirectoryW(root.c_str());  // only when empty
    return left;
}

void RemoveTree(const wstring& dir, const wstring& keepFile) {
    // Check the root too: enumerating dir\* would otherwise follow a root junction into its target.
    const DWORD attrs = GetFileAttributesW(dir.c_str());
    if (attrs == INVALID_FILE_ATTRIBUTES) return;
    if (attrs & FILE_ATTRIBUTE_REPARSE_POINT) { RemoveDirectoryW(dir.c_str()); return; }
    WIN32_FIND_DATAW fd;
    HANDLE h = FindFirstFileW(Join(dir, L"*").c_str(), &fd);
    if (h != INVALID_HANDLE_VALUE) {
        do {
            wstring n = fd.cFileName;
            if (n == L"." || n == L"..") continue;
            wstring p = Join(dir, n);
            if (fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) {
                if (fd.dwFileAttributes & FILE_ATTRIBUTE_REPARSE_POINT) RemoveDirectoryW(p.c_str());
                else RemoveTree(p, keepFile);
            } else if (keepFile.empty() || !IEq(p, keepFile)) {
                SetFileAttributesW(p.c_str(), FILE_ATTRIBUTE_NORMAL);
                DeleteFileW(p.c_str());
            }
        } while (FindNextFileW(h, &fd));
        FindClose(h);
    }
    RemoveDirectoryW(dir.c_str());
}
// Files (not folders) in dir whose name matches the wildcard pattern.
std::vector<wstring> ListFiles(const wstring& dir, const wchar_t* pattern) {
    std::vector<wstring> r;
    WIN32_FIND_DATAW fd;
    HANDLE h = FindFirstFileW(Join(dir, pattern).c_str(), &fd);
    if (h == INVALID_HANDLE_VALUE) return r;
    do { if (!(fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY)) r.push_back(Join(dir, fd.cFileName)); } while (FindNextFileW(h, &fd));
    FindClose(h);
    std::sort(r.begin(), r.end(), [](const wstring& a, const wstring& b) { return Lower(a) < Lower(b); });
    return r;
}
std::vector<wstring> ListDirs(const wstring& dir) {
    std::vector<wstring> r;
    WIN32_FIND_DATAW fd;
    HANDLE h = FindFirstFileW(Join(dir, L"*").c_str(), &fd);
    if (h == INVALID_HANDLE_VALUE) return r;
    do {
        wstring n = fd.cFileName;
        if (n == L"." || n == L"..") continue;
        if ((fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) && !(fd.dwFileAttributes & FILE_ATTRIBUTE_REPARSE_POINT)) r.push_back(Join(dir, n));
    } while (FindNextFileW(h, &fd));
    FindClose(h);
    std::sort(r.begin(), r.end(), [](const wstring& a, const wstring& b) { return Lower(a) < Lower(b); });
    return r;
}
// Every file named `name` below root (depth-limited, budgeted, no links), sorted by path.
std::vector<wstring> FindFilesNamed(const wstring& root, const wchar_t* name, size_t budget, int maxDepth) {
    std::vector<wstring> out;
    std::vector<std::pair<wstring, int>> stack{{root, 0}};
    size_t visited = 0;
    while (!stack.empty() && visited < budget) {
        auto [dir, depth] = stack.back(); stack.pop_back();
        WIN32_FIND_DATAW fd;
        HANDLE h = FindFirstFileW(Join(dir, L"*").c_str(), &fd);
        if (h == INVALID_HANDLE_VALUE) continue;
        do {
            ++visited;
            wstring n = fd.cFileName;
            if (n == L"." || n == L"..") continue;
            if (fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) {
                if (depth < maxDepth && !(fd.dwFileAttributes & FILE_ATTRIBUTE_REPARSE_POINT) && !IEq(n, kStateDirName) && !IEq(n, kLegacyStateDir) && !IEq(n, kKeptDirName))
                    stack.push_back({Join(dir, n), depth + 1});
            } else if (IEq(n, name)) out.push_back(Join(dir, n));
        } while (FindNextFileW(h, &fd));
        FindClose(h);
    }
    std::sort(out.begin(), out.end(), [](const wstring& a, const wstring& b) { return Lower(a) < Lower(b); });
    return out;
}

// SHA-256 with the converter's FIPS 180-4 implementation (no bcrypt dependency).
wstring HexW(const wgc::Digest& d) { return FromUtf8(wgc::to_hex(d)); }
wstring Sha256Bytes(const void* data, size_t size) { return HexW(wgc::Sha256::hash(data, size)); }
wstring Sha256File(const wstring& path) {
    MappedFile f(path, ~0ull);
    if (f.data) return Sha256Bytes(f.data, f.size);
    if (IsFile(path) && FileSize(path) == 0) return Sha256Bytes("", 0);
    return L"";
}
wstring NowStamp(bool forFileName) {
    SYSTEMTIME t; GetLocalTime(&t);
    wchar_t b[40];
    if (forFileName) swprintf(b, 40, L"%04d%02d%02d-%02d%02d", t.wYear, t.wMonth, t.wDay, t.wHour, t.wMinute);
    else swprintf(b, 40, L"%04d-%02d-%02d %02d:%02d:%02d", t.wYear, t.wMonth, t.wDay, t.wHour, t.wMinute, t.wSecond);
    return b;
}
wstring EnvVar(const wchar_t* name) {
    wchar_t b[MAX_PATH * 2];
    DWORD n = GetEnvironmentVariableW(name, b, (DWORD)std::size(b));
    return n && n < std::size(b) ? wstring(b, n) : wstring();
}
// System DLLs this exe uses only sometimes are loaded from System32 only, never from the download folder.
HMODULE LoadSystemDll(const wchar_t* name) { return LoadLibraryExW(name, nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32); }
bool UnderWine() { return GetProcAddress(GetModuleHandleW(L"ntdll.dll"), "wine_get_version") != nullptr; }
wstring PathForGamePrefix(const wstring& path) {
    if (path.empty() || !UnderWine()) return path;
    // Wine's kernel32: the Unix name of a DOS path, allocated on the process heap (UTF-8). It fails when more than the last
    // part is missing (the RR data's F\ folder before its first conversion), so the deepest folder that exists is converted
    // and the missing rest appended.
    using UnixNameFn = char*(CDECL*)(const wchar_t*);
    auto unixName = reinterpret_cast<UnixNameFn>(reinterpret_cast<void*>(GetProcAddress(GetModuleHandleW(L"kernel32.dll"), "wine_get_unix_file_name")));
    if (!unixName) return path;
    wstring head = path, tail;
    char* u = nullptr;
    while (!(u = unixName(head.c_str()))) {
        if (head.size() <= 3) return path;
        const size_t cut = head.find_last_of(L"\\/");
        if (cut == wstring::npos || cut < 2) return path;
        tail = head.substr(cut) + tail;
        head.resize(cut == 2 ? 3 : cut);  // keep the separator on a drive root: "C:\" converts, "C:" alone is not tried
    }
    string unixPath = u;
    if (!tail.empty() && unixPath.size() > 1 && unixPath.back() == '/') unixPath.pop_back();  // a root's Unix name ends in '/'; the tail brings its own separator
    wstring z = L"Z:" + FromUtf8(unixPath.c_str()) + tail;
    HeapFree(GetProcessHeap(), 0, u);
    for (wchar_t& c : z)
        if (c == L'/') c = L'\\';
    return z;
}
wstring SelfPath() { wchar_t b[MAX_PATH * 2]; DWORD n = GetModuleFileNameW(nullptr, b, (DWORD)std::size(b)); return wstring(b, n); }

}  // namespace wg
