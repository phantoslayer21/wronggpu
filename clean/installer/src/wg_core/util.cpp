// SPDX-License-Identifier: GPL-3.0-only
// WrongGPU launcher core: small helpers (strings, paths, files, SHA-256), moved unchanged from setup.cpp. Written for
// this project.
#include "util.h"
#include <aclapi.h>
#include "context.h"
#include "../../../wgshared/include/wg_state.h"

#include <atomic>

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
    auto giveUp = [&](DWORD e) {  // (a read-only bit that cannot be put back is noted: ReadOnlyLost)
        if (readOnly && !SetAttrsByHandle(probe, pi.dwFileAttributes)) readOnlyLost_ = true;
        return fail(e);
    };
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
bool PinnedFile::Close() {
    bool ok = true;
    if (h_ != INVALID_HANDLE_VALUE) {
        if (restoreAttrs_) ok = SetAttrsByHandle(h_, restoreAttrs_);  // the file was not changed: its read-only bit is as it was
        if (!ok) readOnlyLost_ = true;
        CloseHandle(h_);
    }
    h_ = INVALID_HANDLE_VALUE;
    restoreAttrs_ = 0;
    return ok;
}
HANDLE PinnedFile::Release() {
    const HANDLE h = h_;
    h_ = INVALID_HANDLE_VALUE;
    restoreAttrs_ = 0;
    return h;
}

static wstring NtRenamePath(const wstring& path) {
    const wstring extended = wgstate::ExtendedPath(FullPath(path));
    return extended.rfind(L"\\\\?\\", 0) == 0 ? L"\\??\\" + extended.substr(4) : extended;
}
bool PinnedFile::RenameNew(const wstring& path, DWORD* err) {
    if (h_ == INVALID_HANDLE_VALUE) { if (err) *err = ERROR_INVALID_HANDLE; return false; }
    const wstring name = NtRenamePath(path);
    const size_t size = sizeof(FILE_RENAME_INFO) + name.size() * sizeof(wchar_t);
    std::vector<unsigned char> storage(size, 0);
    auto* info = reinterpret_cast<FILE_RENAME_INFO*>(storage.data());
    info->FileNameLength = static_cast<DWORD>(name.size() * sizeof(wchar_t));
    memcpy(info->FileName, name.data(), info->FileNameLength);
    if (SetFileInformationByHandle(h_, FileRenameInfo, info, static_cast<DWORD>(size))) return true;
    if (err) *err = GetLastError();
    return false;
}

// Only clean up staging siblings made by a process that has exited.
void CleanStaleStages(const wstring& path) {
    const wstring prefix = BaseName(path) + L".wronggpu-tmp-";
    for (const auto& temp : ListFiles(Parent(path), (prefix + L"*").c_str())) {
        const wstring name = BaseName(temp);
        if (!StartsWithI(name, prefix)) continue;
        const wstring suffix = name.substr(prefix.size());
        const size_t dash = suffix.find(L'-');
        if (dash == 0 || dash == wstring::npos || dash + 1 == suffix.size() ||
            suffix.find_first_not_of(L"0123456789") != dash ||
            suffix.find_first_not_of(L"0123456789", dash + 1) != wstring::npos) continue;
        const auto number = _wcstoui64(suffix.c_str(), nullptr, 10);
        if (!number || number > MAXDWORD || number == GetCurrentProcessId()) continue;
        HANDLE process = OpenProcess(SYNCHRONIZE, FALSE, static_cast<DWORD>(number));
        bool exited = process == nullptr && GetLastError() == ERROR_INVALID_PARAMETER;
        if (process) {
            exited = WaitForSingleObject(process, 0) == WAIT_OBJECT_0;
            CloseHandle(process);
        }
        PinnedFile file;
        if (exited && file.Open(temp, true)) file.Delete();
    }
}

bool PinnedFile::ReplaceBytes(const wstring& path, const void* data, size_t size, DWORD* err, bool failWrite,
                              PinnedFile* metadata, BY_HANDLE_FILE_INFORMATION* published) {
    if (!metadata && h_ == INVALID_HANDLE_VALUE) { if (err) *err = ERROR_INVALID_HANDLE; return false; }
    CleanStaleStages(path);
    static volatile LONG serial = 0;
    PinnedFile stage;
    const wstring temp = path + L".wronggpu-tmp-" + std::to_wstring(GetCurrentProcessId()) + L"-" +
                         std::to_wstring(InterlockedIncrement(&serial));
    if (!stage.Create(temp, err)) return false;
    auto fail = [&](DWORD e) { stage.Delete(); if (err) *err = e; return false; };
    DWORD error = 0;
    if (!stage.OverwriteBytes(data, failWrite ? size / 2 : size, &error) || !stage.Flush(&error)) return fail(error);
    if (failWrite) return fail(ERROR_DISK_FULL);
    PinnedFile& source = metadata ? *metadata : *this;
    if ((error = source.CopyMetadataTo(stage, temp, metadata != nullptr)) != ERROR_SUCCESS) return fail(error);
    BY_HANDLE_FILE_INFORMATION identity{};
    if (published && !GetFileInformationByHandle(stage.h_, &identity)) return fail(GetLastError());
    wstring originalLeft;
    if (!(IsOpen() ? ReplaceWith(stage, path, &error, &originalLeft) : stage.RenameNew(path, &error))) {
        fail(error);
        if (!originalLeft.empty())
            Fail(L"Cannot replace " + path + L": " + ErrText(error) + L". The original is kept at " + originalLeft);
        return false;
    }
    if (published) *published = identity;
    return true;
}
bool PinnedFile::WriteChunk(const void* data, DWORD size, DWORD* err) {
    DWORD written = 0, error = ERROR_SUCCESS;
    if (h_ == INVALID_HANDLE_VALUE) error = ERROR_INVALID_HANDLE;
    else if (!WriteFile(h_, data, size, &written, nullptr)) error = GetLastError();
    else if (written != size) error = ERROR_WRITE_FAULT;
    if (err) *err = error;
    return error == ERROR_SUCCESS;
}
bool PinnedFile::ReplaceWith(PinnedFile& stage, const wstring& path, DWORD* err, wstring* originalLeft) {
    if (h_ == INVALID_HANDLE_VALUE || !stage.IsOpen()) { if (err) *err = ERROR_INVALID_HANDLE; return false; }
    DWORD error = ERROR_SUCCESS;
#ifdef WRONGGPU_TESTING
    const bool simulateRace = IEq(EnvVar(L"WRONGGPU_TEST_REPLACE_RACE"), path) && GamePathAllowed(path);
    if (simulateRace) error = ERROR_SHARING_VIOLATION;
#endif
    if (!error && stage.RenameReplace(path, &error)) return true;
    if (error != ERROR_SHARING_VIOLATION && error != ERROR_NOT_SUPPORTED &&
        error != ERROR_INVALID_PARAMETER && error != ERROR_ACCESS_DENIED) {
        if (err) *err = error;
        return false;
    }
    // Some filesystems reject a POSIX replace over a delete-denying pin. Move the verified
    // file by handle, publish the complete sibling with no replace, and retain the old name
    // if a competing creator prevents recovery.
    static volatile LONG serial = 0;
    const wstring saved = path + L".wronggpu-old-" + std::to_wstring(GetCurrentProcessId()) + L"-" +
                          std::to_wstring(InterlockedIncrement(&serial));
    if (!RenameNew(saved, &error)) { if (err) *err = error; return false; }
#ifdef WRONGGPU_TESTING
    if (simulateRace) {
        PinnedFile competing;
        if (competing.Create(path)) {
            const string bytes = "; competing fixture file\r\n";
            if (!competing.OverwriteBytes(bytes.data(), bytes.size())) competing.Delete();
        }
    }
#endif
    if (!stage.RenameNew(path, &error)) {
        DWORD restoreError = 0;
        if (!RenameNew(path, &restoreError)) {
            error = restoreError;
            if (originalLeft) *originalLeft = saved;
        }
        if (err) *err = error;
        return false;
    }
    Delete();
    return true;
}
bool PinnedFile::RenameReplace(const wstring& path, DWORD* err) {
    if (h_ == INVALID_HANDLE_VALUE) { if (err) *err = ERROR_INVALID_HANDLE; return false; }
    const wstring name = NtRenamePath(path);
    const size_t bytes = sizeof(FILE_RENAME_INFO) + name.size() * sizeof(wchar_t);
    std::vector<unsigned char> storage(bytes, 0);
    auto* info = reinterpret_cast<FILE_RENAME_INFO*>(storage.data());
    // FileRenameInfoEx, REPLACE_IF_EXISTS | POSIX_SEMANTICS. The old handle remains valid through the swap.
    const DWORD flags = 0x1 | 0x2;
    memcpy(storage.data(), &flags, sizeof flags);
    info->FileNameLength = static_cast<DWORD>(name.size() * sizeof(wchar_t));
    memcpy(info->FileName, name.data(), info->FileNameLength);
    if (!SetFileInformationByHandle(h_, static_cast<FILE_INFO_BY_HANDLE_CLASS>(22), info, static_cast<DWORD>(bytes))) {
        if (err) *err = GetLastError();
        return false;
    }
    return true;
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
    if (h_ == INVALID_HANDLE_VALUE) { if (err) *err = ERROR_INVALID_HANDLE; return false; }
    HANDLE s = CreateFileW(src.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (s == INVALID_HANDLE_VALUE) { if (err) *err = GetLastError(); return false; }
    struct Closer { HANDLE h; ~Closer() { CloseHandle(h); } } closer{s};
    return CopyFrom(s, srcSha, err);
}
bool PinnedFile::OverwriteFrom(PinnedFile& src, const wstring& srcSha, DWORD* err) {
    if (h_ == INVALID_HANDLE_VALUE || src.h_ == INVALID_HANDLE_VALUE) { if (err) *err = ERROR_INVALID_HANDLE; return false; }
    return CopyFrom(src.h_, srcSha, err);
}
bool PinnedFile::CopyFrom(HANDLE s, const wstring& srcSha, DWORD* err) {
    auto fail = [&](DWORD e) { if (err) *err = e; return false; };
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
    h_ = CreateFileW(path.c_str(), GENERIC_READ | GENERIC_WRITE | DELETE | FILE_WRITE_ATTRIBUTES | WRITE_DAC, FILE_SHARE_READ, nullptr, CREATE_NEW,
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
static DWORD CopyNamedStream(const wstring& from, const wstring& to) {
    struct File {
        HANDLE h;
        ~File() { if (h != INVALID_HANDLE_VALUE) CloseHandle(h); }
    };
    const DWORD share = FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE;
    File input{CreateFileW(from.c_str(), GENERIC_READ, share, nullptr, OPEN_EXISTING, FILE_FLAG_OPEN_REPARSE_POINT, nullptr)};
    if (input.h == INVALID_HANDLE_VALUE) return GetLastError();
    File output{CreateFileW(to.c_str(), GENERIC_WRITE, share, nullptr, CREATE_NEW, FILE_FLAG_OPEN_REPARSE_POINT, nullptr)};
    if (output.h == INVALID_HANDLE_VALUE) return GetLastError();
    std::vector<char> buffer(1u << 16);
    for (;;) {
        DWORD read = 0, written = 0;
        if (!ReadFile(input.h, buffer.data(), static_cast<DWORD>(buffer.size()), &read, nullptr)) return GetLastError();
        if (!read) break;
        if (!WriteFile(output.h, buffer.data(), read, &written, nullptr)) return GetLastError();
        if (written != read) return ERROR_WRITE_FAULT;
    }
    return FlushFileBuffers(output.h) ? ERROR_SUCCESS : GetLastError();
}

DWORD PinnedFile::CopyMetadataTo(PinnedFile& target, const wstring& targetPath, bool restoreTimes) {
    FILE_BASIC_INFO basic{};
    if (!GetFileInformationByHandleEx(h_, FileBasicInfo, &basic, sizeof(basic))) return GetLastError();
    DWORD volumeFlags = 0;
    if (!GetVolumeInformationByHandleW(h_, nullptr, 0, nullptr, nullptr, &volumeFlags, nullptr, 0)) return GetLastError();
    if (volumeFlags & FILE_NAMED_STREAMS) {
        const wstring path = ResolvedHandlePath(h_);
        if (path.empty()) return ERROR_INVALID_NAME;
        WIN32_FIND_STREAM_DATA stream{};
        HANDLE found = FindFirstStreamW(path.c_str(), FindStreamInfoStandard, &stream, 0);
        if (found == INVALID_HANDLE_VALUE && GetLastError() != ERROR_HANDLE_EOF) return GetLastError();
        if (found != INVALID_HANDLE_VALUE) {
            struct FindCloser { HANDLE h; ~FindCloser() { FindClose(h); } } closer{found};
            do {
                if (wcscmp(stream.cStreamName, L"::$DATA") == 0) continue;
                const DWORD error = CopyNamedStream(path + stream.cStreamName, targetPath + stream.cStreamName);
                if (error != ERROR_SUCCESS) return error;
            } while (FindNextStreamW(found, &stream));
            if (GetLastError() != ERROR_HANDLE_EOF) return GetLastError();
        }
    }
    if (volumeFlags & FILE_PERSISTENT_ACLS) {
        PACL dacl = nullptr;
        PSECURITY_DESCRIPTOR descriptor = nullptr;
        const DWORD got = GetSecurityInfo(h_, SE_FILE_OBJECT, DACL_SECURITY_INFORMATION, nullptr, nullptr, &dacl, nullptr, &descriptor);
        if (got != ERROR_SUCCESS) return got;
        SECURITY_DESCRIPTOR_CONTROL control{};
        DWORD revision = 0;
        DWORD error = ERROR_SUCCESS;
        if (!GetSecurityDescriptorControl(descriptor, &control, &revision)) error = GetLastError();
        else {
            const SECURITY_INFORMATION protection = control & SE_DACL_PROTECTED
                ? PROTECTED_DACL_SECURITY_INFORMATION : UNPROTECTED_DACL_SECURITY_INFORMATION;
            error = SetSecurityInfo(target.h_, SE_FILE_OBJECT, DACL_SECURITY_INFORMATION | protection, nullptr, nullptr, dacl, nullptr);
        }
        LocalFree(descriptor);
        if (error != ERROR_SUCCESS) return error;
    }
    basic.ChangeTime.QuadPart = 0;
    if (!restoreTimes) basic.LastWriteTime.QuadPart = 0;
    basic.FileAttributes = (restoreAttrs_ ? restoreAttrs_ : basic.FileAttributes) & kUserAttrs;
    if (!basic.FileAttributes) basic.FileAttributes = FILE_ATTRIBUTE_NORMAL;
    return SetFileInformationByHandle(target.h_, FileBasicInfo, &basic, sizeof(basic)) ? ERROR_SUCCESS : GetLastError();
}

bool PinnedFile::CopyOut(const wstring& dst, wstring* sha, DWORD* err, PinnedFile* created) {
    auto fail = [&](DWORD e) { if (err) *err = e; return false; };
    if (h_ == INVALID_HANDLE_VALUE) return fail(ERROR_INVALID_HANDLE);
    PinnedFile local;
    PinnedFile& out = created ? *created : local;
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
    if (!FlushFileBuffers(out.h_)) { const DWORD e = GetLastError(); out.Delete(); return fail(e); }
    const DWORD metadata = CopyMetadataTo(out, dst, true);
    if (metadata != ERROR_SUCCESS) { out.Delete(); return fail(metadata); }
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
bool PinnedFile::Id(DWORD* volume, ULONGLONG* index) const {
    BY_HANDLE_FILE_INFORMATION i{};
    if (h_ == INVALID_HANDLE_VALUE || !GetFileInformationByHandle(h_, &i)) return false;
    *volume = i.dwVolumeSerialNumber;
    *index = (static_cast<ULONGLONG>(i.nFileIndexHigh) << 32) | i.nFileIndexLow;
    return true;
}
bool PinnedFile::Flush(DWORD* err) {
    if (h_ == INVALID_HANDLE_VALUE) { if (err) *err = ERROR_INVALID_HANDLE; return false; }
    if (FlushFileBuffers(h_)) return true;
    if (err) *err = GetLastError();
    return false;
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
// ---- store access (#243): the one way to the store libraries' metadata and folders --------------------------------------
static std::atomic<size_t> g_libraryAccesses{0};

bool TestLibraries(wstring* folder) {
#ifdef WRONGGPU_TESTING
    wchar_t buf[MAX_PATH * 2] = {};
    SetLastError(ERROR_SUCCESS);
    const DWORD n = GetEnvironmentVariableW(L"WRONGGPU_TEST_LIBRARIES", buf, (DWORD)(sizeof buf / sizeof buf[0]));
    if (n == 0 && GetLastError() == ERROR_ENVVAR_NOT_FOUND) return false;
    if (folder) *folder = n > 0 && n < sizeof buf / sizeof buf[0] ? wstring(buf) : wstring();
    return true;
#else
    (void)folder;
    return false;
#endif
}

#ifdef WRONGGPU_TESTING
// A path as one form for the scope test: \\?\C:\x and \??\C:\x as C:\x, \\?\UNC\srv\share as \\srv\share, then full.
#endif

static wstring ScopeForm(const wstring& p) {
    wstring s = p;
    if (StartsWithI(s, L"\\\\?\\UNC\\")) s = L"\\\\" + s.substr(8);
    else if (StartsWithI(s, L"\\\\?\\") || StartsWithI(s, L"\\??\\")) s = s.substr(4);
    std::replace(s.begin(), s.end(), L'/', L'\\');
    s = FullPath(s);
    while (s.size() > 3 && s.back() == L'\\') s.pop_back();
    return s;
}

#ifdef WRONGGPU_TESTING
// Test builds: the access log (WRONGGPU_TEST_ACCESS_LOG). A new log starts with "run <WRONGGPU_TEST_RUN_ID>" (the tests' guard
// checks it is this run's); every line is appended before the read it records. false: not written (the read is refused).
static bool TestAccessLine(const wstring& line) {
    const wstring log = EnvVar(L"WRONGGPU_TEST_ACCESS_LOG");
    if (log.empty() || !GamePathAllowed(log)) return false;
    HANDLE h = CreateFileW(log.c_str(), FILE_APPEND_DATA, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, nullptr, CREATE_NEW,
                           FILE_ATTRIBUTE_NORMAL, nullptr);
    string text;
    if (h != INVALID_HANDLE_VALUE) {
        text = ToUtf8(L"run " + EnvVar(L"WRONGGPU_TEST_RUN_ID") + L"\r\n");  // (this process made the log)
    } else {
        if (GetLastError() != ERROR_FILE_EXISTS) return false;
        h = CreateFileW(log.c_str(), FILE_APPEND_DATA, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, nullptr, OPEN_EXISTING,
                        FILE_ATTRIBUTE_NORMAL, nullptr);
        if (h == INVALID_HANDLE_VALUE) return false;
    }
    text += ToUtf8(line + L"\r\n");
    DWORD w = 0;
    const bool ok = WriteFile(h, text.data(), (DWORD)text.size(), &w, nullptr) && w == text.size();
    CloseHandle(h);
    return ok;
}

#endif

wstring ResolvedHandlePath(HANDLE file) {
    if (file == INVALID_HANDLE_VALUE) return L"";
    wchar_t buf[MAX_PATH * 4] = {};
    const DWORD n = GetFinalPathNameByHandleW(file, buf, (DWORD)(sizeof buf / sizeof buf[0]), FILE_NAME_NORMALIZED | VOLUME_NAME_DOS);
    return n > 0 && n < sizeof buf / sizeof buf[0] ? ScopeForm(buf) : L"";
}
// The final form of a path that exists: the name Windows resolves it to (every junction and link above it followed), as
// C:\... or \\server\...; empty when it cannot be opened.
static wstring FinalForm(const wstring& p) {
    const HANDLE h = CreateFileW(p.c_str(), FILE_READ_ATTRIBUTES, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, nullptr, OPEN_EXISTING,
                                 FILE_FLAG_BACKUP_SEMANTICS, nullptr);
    if (h == INVALID_HANDLE_VALUE) return L"";
    const wstring resolved = ResolvedHandlePath(h);
    CloseHandle(h);
    return resolved;
}
// Resolve the existing ancestor of a new path before adding the still-missing names.
wstring ResolvedPath(const wstring& path) {
    wstring p = ScopeForm(path), suffix;
    while (!p.empty()) {
        if (GetFileAttributesW(p.c_str()) != INVALID_FILE_ATTRIBUTES) {
            const wstring final = FinalForm(p);
            return final.empty() || suffix.empty() ? final : ScopeForm(Join(final, suffix));
        }
        const DWORD error = GetLastError();
        if (error != ERROR_FILE_NOT_FOUND && error != ERROR_PATH_NOT_FOUND) return L"";
        const wstring up = Parent(p);
        if (up.empty() || up == p) break;
        suffix = suffix.empty() ? BaseName(p) : Join(BaseName(p), suffix);
        p = up;
    }
    return L"";
}
// The long lexical form of a path that exists (8.3 names expanded), for comparing with its final form.
static wstring LongForm(const wstring& p) {
    wchar_t buf[MAX_PATH * 4] = {};
    const DWORD n = GetLongPathNameW(p.c_str(), buf, (DWORD)(sizeof buf / sizeof buf[0]));
    return n > 0 && n < sizeof buf / sizeof buf[0] ? ScopeForm(buf) : p;
}
// Test builds, the scope test on what a path really is (not only its spelling): the nearest part of it that exists must
// resolve to its own long name (no junction or link above it), the path itself must not be a reparse point, and a file
// must have no second name (a hard link inside could name an outside file's data). why: the refusal's reason.
static bool RealPathOk(const wstring& lexical, const wchar_t** why) {
    wstring probe = lexical;
    while (!probe.empty() && GetFileAttributesW(probe.c_str()) == INVALID_FILE_ATTRIBUTES) {
        const wstring up = Parent(probe);
        if (up == probe || up.empty()) { *why = L"cannot be resolved"; return false; }
        probe = up;
    }
    if (probe.empty()) return true;
    const wstring finalForm = FinalForm(probe);
    if (finalForm.empty() || !IEq(finalForm, LongForm(probe))) { *why = L"resolves elsewhere (a junction or link)"; return false; }
    if (IEq(probe, lexical)) {
        const HANDLE h = CreateFileW(lexical.c_str(), FILE_READ_ATTRIBUTES, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, nullptr, OPEN_EXISTING,
                                     FILE_FLAG_BACKUP_SEMANTICS | FILE_FLAG_OPEN_REPARSE_POINT, nullptr);
        if (h == INVALID_HANDLE_VALUE) { *why = L"cannot be checked"; return false; }
        BY_HANDLE_FILE_INFORMATION fi{};
        const bool got = GetFileInformationByHandle(h, &fi) != 0;
        CloseHandle(h);
        if (!got) { *why = L"cannot be checked"; return false; }
        if (fi.dwFileAttributes & FILE_ATTRIBUTE_REPARSE_POINT) { *why = L"is a reparse point (a link or junction)"; return false; }
        if (!(fi.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) && fi.nNumberOfLinks > 1) { *why = L"has another name (a hard link)"; return false; }
    }
    return true;
}

// Expand 8.3 names without resolving junctions. Keep the names of any still-missing descendants.
static wstring LongAncestors(const wstring& lexical) {
    wstring probe = lexical, suffix;
    while (GetFileAttributesW(probe.c_str()) == INVALID_FILE_ATTRIBUTES) {
        const wstring up = Parent(probe);
        if (up.empty() || up == probe) return lexical;
        suffix = suffix.empty() ? BaseName(probe) : Join(BaseName(probe), suffix);
        probe = up;
    }
    return suffix.empty() ? LongForm(probe) : Join(LongForm(probe), suffix);
}

bool PathWithin(const wstring& path, const wstring& root, bool requirePlain) {
    if (path.empty() || root.empty()) return false;
    const wstring lexical = ScopeForm(path), rootLexical = ScopeForm(root);
    auto absolute = [](const wstring& s) {
        if (s.size() >= 3 && s[1] == L':' && s[2] == L'\\') return s.find(L':', 2) == wstring::npos;
        const auto serverEnd = s.find(L'\\', 2);
        return StartsWithI(s, L"\\\\") && !StartsWithI(s, L"\\\\.\\") &&
               serverEnd != wstring::npos && serverEnd > 2 && serverEnd + 1 < s.size() &&
               s.find(L':') == wstring::npos;
    };
    if (!absolute(lexical) || !absolute(rootLexical)) return false;
    auto under = [](const wstring& p, const wstring& r) {
        return IEq(p, r) || (p.size() > r.size() && StartsWithI(p, r) && (p[r.size()] == L'\\' || r.back() == L'\\'));
    };
    const bool inside = under(lexical, rootLexical) || under(LongAncestors(lexical), LongAncestors(rootLexical));
    const wchar_t* why = nullptr;
    return inside && (!requirePlain || (RealPathOk(rootLexical, &why) && RealPathOk(lexical, &why)));
}

bool LibraryAccess(const wstring& path) {
    ++g_libraryAccesses;
#ifdef WRONGGPU_TESTING
    wstring folder;
    if (!TestLibraries(&folder)) {
        // Tests linked directly against the core use only their own fixture arguments.
        if (DataOverride().empty()) return true;
        TestAccessLine(L"library access refused: " + path);
        return false;
    }
    // the tests' scope: the folder that holds WRONGGPU_TEST_LIBRARIES (the test run's own folder), as it really is; nothing
    // else is read, and nothing is read whose access line could not be written (fail closed)
    const wstring root = folder.empty() ? wstring() : Parent(ScopeForm(folder));
    const wstring p = ScopeForm(path);
    const wchar_t* why = nullptr;
    bool inside = !root.empty() && (IEq(p, root) || (p.size() > root.size() && StartsWithI(p, root) && (p[root.size()] == L'\\' || root.back() == L'\\')));
    if (inside && !GamePathAllowed(path)) { inside = false; why = L"outside the test build's --data folder"; }
    if (inside && !RealPathOk(p, &why)) inside = false;
    if (!inside) {
        TestAccessLine(L"library access refused: " + path + (why ? wstring(L" (") + why + L")" : wstring()));
        return false;
    }
    return TestAccessLine(L"library access: " + path);
#else
    return GamePathAllowed(path);
#endif
}

bool LibraryRegistryAccess(const wchar_t* what) {
    ++g_libraryAccesses;
#ifdef WRONGGPU_TESTING
    TestAccessLine(wstring(L"library access refused: registry ") + what);
    return false;
#else
    (void)what;
    return true;
#endif
}

void TestAccessStart(const wchar_t* program) {
#ifdef WRONGGPU_TESTING
    if (TestLibraries()) TestAccessLine(L"process " + std::to_wstring(GetCurrentProcessId()) + L" " + program + L" start");
#else
    (void)program;
#endif
}

size_t LibraryAccessCount() { return g_libraryAccesses; }

// Every file named `name` below root (depth-limited, budgeted, no links), sorted by path.
std::vector<wstring> FindFilesNamed(const wstring& root, const wchar_t* name, size_t budget, int maxDepth) {
    std::vector<wstring> out;
    std::vector<std::pair<wstring, int>> stack{{root, 0}};
    size_t visited = 0;
    while (!stack.empty() && visited < budget) {
        auto [dir, depth] = stack.back(); stack.pop_back();
        if (!LibraryAccess(dir)) continue;  // every folder listed is asked first (a test build: never one outside its own)
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
            } else if (IEq(n, name) && LibraryAccess(Join(dir, n))) {
                out.push_back(Join(dir, n));  // (each file found is asked too: what its caller opens next)
            }
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
