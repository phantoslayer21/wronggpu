// SPDX-License-Identifier: MIT
// WrongGPU launcher core: small helpers (strings, paths, files, SHA-256), moved unchanged from setup.cpp. Written for
// this project.
#pragma once
#include "constants.h"
#include "sha256.h"

namespace wg {

struct SetupError { wstring msg; int code = EXIT_FAILED; };
// CreateFileW(path, GENERIC_READ, share, OPEN_EXISTING), through the elevated worker's guard while one is active
// (secure_fs.h). INVALID_HANDLE_VALUE with the last error set.
HANDLE OpenForRead(const wstring& path, DWORD share);
// Read-only view of a whole file (large files are scanned without copying them).
struct MappedFile {
    HANDLE file = INVALID_HANDLE_VALUE, map = nullptr;
    const uint8_t* data = nullptr; size_t size = 0;
    explicit MappedFile(const wstring& path, unsigned long long limit = 1ull << 30) {
        file = OpenForRead(path, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE);
        if (file == INVALID_HANDLE_VALUE) return;
        LARGE_INTEGER sz{};
        if (!GetFileSizeEx(file, &sz) || sz.QuadPart == 0 || (unsigned long long)sz.QuadPart > limit) return;
        map = CreateFileMappingW(file, nullptr, PAGE_READONLY, 0, 0, nullptr);
        if (!map) return;
        data = static_cast<const uint8_t*>(MapViewOfFile(map, FILE_MAP_READ, 0, 0, 0));
        if (data) size = (size_t)sz.QuadPart;
    }
    ~MappedFile() { if (data) UnmapViewOfFile(data); if (map) CloseHandle(map); if (file != INVALID_HANDLE_VALUE) CloseHandle(file); }
    MappedFile(const MappedFile&) = delete;
    MappedFile& operator=(const MappedFile&) = delete;
};

[[noreturn]] void Fail(const wstring& m, int code = EXIT_FAILED);
wstring Lower(wstring s);
bool IEq(const wstring& a, const wstring& b);
bool Contains(const wstring& hay, const wchar_t* needle);
bool EndsWithI(const wstring& s, const wstring& suffix);
bool StartsWithI(const wstring& s, const wstring& prefix);
wstring Join(const wstring& a, const wstring& b);
wstring Parent(const wstring& p);
wstring BaseName(const wstring& p);
wstring Stem(const wstring& p);
// s shortened in the middle to at most max characters (an ellipsis; cut at a folder separator where one is near).
wstring ShortenMiddle(const wstring& s, size_t max);
bool Exists(const wstring& p);
bool IsDir(const wstring& p);
bool IsFile(const wstring& p);
unsigned long long FileSize(const wstring& p);
wstring FullPath(const wstring& p);
wstring Hex32(DWORD v);
wstring ErrText(DWORD e);
string ToUtf8(const wstring& w);
wstring FromUtf8(const string& s);
bool ReadAll(const wstring& path, string& out);
bool ContainsText(const uint8_t* d, size_t n, const char* needle);
bool FileContainsText(const wstring& path, const char* needle, unsigned long long limit = 1ull << 30);
void EnsureDir(const wstring& dir, std::vector<wstring>* created = nullptr);
void ForceDelete(const wstring& p);
void WriteFileAtomic(const wstring& path, const void* data, size_t size);
void CopyChecked(const wstring& src, const wstring& dst);
void RemoveTree(const wstring& dir, const wstring& keepFile = L"");
// The modules' other path-based Win32 calls, through the elevated worker's guard while one is active (secure_fs.h; its
// refusals throw SetupError). Each returns what the Win32 call returns and sets the last error as it does.
DWORD GetAttrs(const wstring& p);                                  // GetFileAttributesW
bool SetAttrs(const wstring& p, DWORD attrs);                      // SetFileAttributesW
bool DeletePath(const wstring& p);                                 // DeleteFileW
bool RemoveDirPath(const wstring& p);                              // RemoveDirectoryW (an empty folder)
bool MakeDir(const wstring& p);                                    // CreateDirectoryW(p, nullptr)
bool MovePath(const wstring& from, const wstring& to, DWORD flags); // MoveFileExW (MOVEFILE_REPLACE_EXISTING, _COPY_ALLOWED; written through)
bool CopyPath(const wstring& from, const wstring& to);             // CopyFileW(from, to, FALSE)
HANDLE CreateNewFile(const wstring& p);                            // CreateFileW(GENERIC_WRITE, no sharing, CREATE_NEW)
// The entries of `dir` (no "." or ".."); false (last error set) when it cannot be listed. Links are marked
// FILE_ATTRIBUTE_REPARSE_POINT; reparse points that only store a file differently (secure_fs.h) are not. Through the
// elevated worker's guard while one is active.
struct DirEntry { wstring name; DWORD attrs; unsigned long long size; };
bool ListEntries(const wstring& dir, std::vector<DirEntry>& out);
std::vector<wstring> ListFiles(const wstring& dir, const wchar_t* pattern);
std::vector<wstring> ListDirs(const wstring& dir);
std::vector<wstring> FindFilesNamed(const wstring& root, const wchar_t* name, size_t budget = 60000, int maxDepth = 10);
wstring HexW(const wgc::Digest& d);
wstring Sha256Bytes(const void* data, size_t size);
wstring Sha256File(const wstring& path);
wstring NowStamp(bool forFileName);
wstring EnvVar(const wchar_t* name);
HMODULE LoadSystemDll(const wchar_t* name);
// The launcher under Wine (Linux): Wine's ntdll exports wine_get_version.
bool UnderWine();
// A path for a game in another Wine prefix (Proton gives every game its own): under Wine the Z: path of the same file
// (Z: maps the Linux root in every prefix, C: does not), otherwise the path unchanged.
wstring PathForGamePrefix(const wstring& path);
wstring SelfPath();
}  // namespace wg
