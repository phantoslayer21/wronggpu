// SPDX-License-Identifier: MIT
// WrongGPU launcher core: small helpers (strings, paths, files, SHA-256), moved unchanged from setup.cpp. Written for
// this project.
#pragma once
#include <functional>

#include "constants.h"
#include "sha256.h"

namespace wg {

struct SetupError { wstring msg; int code = EXIT_FAILED; };
// Read-only view of a whole file (large files are scanned without copying them).
struct MappedFile {
    HANDLE file = INVALID_HANDLE_VALUE, map = nullptr;
    const uint8_t* data = nullptr; size_t size = 0;
    explicit MappedFile(const wstring& path, unsigned long long limit = 1ull << 30) {
        file = CreateFileW(path.c_str(), GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
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
// Copies src to dst only when nothing is at dst (the creation itself fails when something is: ERROR_FILE_EXISTS, so a file
// that appeared a moment ago is never overwritten). false: not copied, `err` holds why.
bool CopyNew(const wstring& src, const wstring& dst, DWORD* err = nullptr);
// Writes `data` into a new file at path, created only where nothing is (CREATE_NEW: ERROR_FILE_EXISTS when anything is there,
// so a file another program made a moment ago is never replaced); readOnly: the new file is made read-only. Not atomic for a
// reader: the file exists, partly written, while it is written (a write that fails removes it again through its handle).
bool WriteFileNew(const wstring& path, const void* data, size_t size, bool readOnly, DWORD* err = nullptr);

// A plain file held while it is checked and then deleted or overwritten (record.h, "the replace race"). It is opened once with
// DELETE and read access (and write access when forWrite) and FILE_SHARE_READ only: while this object holds it, no other
// program can write to it, rename it, replace it or delete it, so the bytes hashed through the handle are the bytes the delete
// or the overwrite through the same handle acts on. Links and folders are refused, and so is a file with more than one name
// (a hard link) when it is opened for writing (ERROR_TOO_MANY_LINKS): bytes written through one name change the others. A
// read-only file opened for writing is made writable only after a probe has shown what it is, and is put back as it was when
// it is not written. What it does not cover: the folders
// above the file are resolved by name when it is opened (a program that can rename or replace one of them can make the open
// reach another file; the open then hashes that file, and the hash is compared with the one the plan was made on).
class PinnedFile {
public:
    PinnedFile() = default;
    PinnedFile(const PinnedFile&) = delete;
    PinnedFile& operator=(const PinnedFile&) = delete;
    ~PinnedFile() { Close(); }
    // false: nothing is held; err is ERROR_FILE_NOT_FOUND / ERROR_PATH_NOT_FOUND (nothing there), a sharing violation or access
    // denied (another program holds it), or ERROR_NOT_SUPPORTED (a folder or a link, not a plain file).
    bool Open(const wstring& path, bool forWrite, DWORD* err = nullptr);
    // Creates path only where nothing is (CREATE_NEW: ERROR_FILE_EXISTS when a file, folder or link is there, so a file that
    // appeared a moment ago is never replaced) and holds the new, empty file as Open(path, true) holds one. The caller fills
    // it (Overwrite, OverwriteBytes) or, when it gives up, removes it again through the handle (Delete).
    bool Create(const wstring& path, DWORD* err = nullptr);
    bool IsOpen() const { return h_ != INVALID_HANDLE_VALUE; }
    wstring Sha256();  // of the bytes through the handle (empty: cannot be read)
    bool ReadBytes(string& out);  // all of the bytes through the handle (false: cannot be read)
    // Copies the bytes through the handle into a new file dst (created only where nothing is, as Create); *sha: the SHA-256 of
    // the bytes copied. false: err holds why (a partly written dst is removed).
    bool CopyOut(const wstring& dst, wstring* sha, DWORD* err = nullptr);
    // Sets or clears the read-only attribute through the handle (the other attributes stay); what Open cleared for a write is
    // then not put back on Close.
    bool SetReadOnly(bool on, DWORD* err = nullptr);
    // The file was read-only when Open made it writable (the bit is put back on Close unless it was written or set on purpose).
    bool WasReadOnly() const { return (restoreAttrs_ & FILE_ATTRIBUTE_READONLY) != 0; }
    bool Flush(DWORD* err = nullptr);  // FlushFileBuffers through the handle, its result checked
    // The held file's identity (volume serial number and file index): two names with the same identity are one file, a file
    // with the same bytes but another identity is another file. false: nothing held, or it cannot be read.
    bool Id(DWORD* volume, ULONGLONG* index) const;
    // Deletes the file through the handle (FileDispositionInfoEx with POSIX semantics, the read-only attribute ignored; the
    // older FileDispositionInfo where that is not supported) and closes it. The name is free when this returns true.
    bool Delete(DWORD* err = nullptr);
    // Replaces the bytes through the handle (forWrite) with src's, which are hashed first and must be `srcSha` (nothing is
    // touched otherwise); the attributes and times of src are put on the file. false: err holds why (ERROR_FILE_CORRUPT when
    // src is not srcSha). Not atomic against a stop: the record and the backup are kept until the restore is complete.
    bool Overwrite(const wstring& src, const wstring& srcSha, DWORD* err = nullptr);
    // The same from a source this process already holds (Open(src, false)): hashed through its handle, which must give srcSha,
    // and copied through it, so the bytes checked before the first write of a multi-file step are the bytes written.
    bool OverwriteFrom(PinnedFile& src, const wstring& srcSha, DWORD* err = nullptr);
    // Replaces the bytes through the handle (forWrite) with `data`: the conditional write of a file that was hashed through this
    // same handle (a settings migration). The file's attributes stay.
    bool OverwriteBytes(const void* data, size_t size, DWORD* err = nullptr);
    // A read-only bit that Open cleared for a write that did not happen is put back. false: that failed (the file stays
    // writable; ReadOnlyLost then says so too).
    bool Close();
    // A read-only bit this object cleared could not be put back (by a failed Open or by Close): the file is writable now.
    bool ReadOnlyLost() const { return readOnlyLost_; }

private:
    bool CopyFrom(HANDLE src, const wstring& srcSha, DWORD* err);
    HANDLE h_ = INVALID_HANDLE_VALUE;
    DWORD restoreAttrs_ = 0;
    bool readOnlyLost_ = false;  // the attributes to put back on Close (a read-only file that Open made writable); 0: none
};
enum class DeleteIf { Deleted, Gone, Changed, Failed };
// Deletes path when its bytes are one of `shas` (pinned: hashed and deleted through one handle). Gone: nothing was there.
// Changed: other bytes (or a link or folder): nothing was done. Failed: held by another program or not deletable (err).
DeleteIf DeleteIfSha(const wstring& path, const std::vector<wstring>& shas, DWORD* err = nullptr);
// Deletes below `root` only the files `ours` accepts (their path below root, backslashes, and the file held: it can hash it) and
// the folders left empty by that (RemoveDirectoryW: a folder with anything in it stays). A file is deleted through the handle
// it was checked through (PinnedFile); a link, a file
// another program holds and every file that is not ours stay. keepFile (a full path, the running uninstaller) is skipped
// without being listed. Returns what stays, as paths below root. root itself goes only when it is empty at the end.
std::vector<wstring> RemoveOwned(const wstring& root, const std::function<bool(const wstring&, PinnedFile&)>& ours, const wstring& keepFile = L"");
void RemoveTree(const wstring& dir, const wstring& keepFile = L"");
std::vector<wstring> ListFiles(const wstring& dir, const wchar_t* pattern);
std::vector<wstring> ListDirs(const wstring& dir);
// Walks root (asked of LibraryAccess first: a test build never walks a folder outside the test run's own).
std::vector<wstring> FindFilesNamed(const wstring& root, const wchar_t* name, size_t budget = 60000, int maxDepth = 10);
// Store access (#243): every read of a store's metadata (Steam's library list and app manifests, Epic's manifests, the GOG
// registry, Steam's art cache) and every walk of a library game's folder asks here first.
// - LibraryAccess(path): true = may be read. In a WRONGGPU_TESTING build with WRONGGPU_TEST_LIBRARIES set, only paths below
//   the folder that holds it (the test run's own folder; \\?\ and UNC forms compared as such); anything else is refused.
// - LibraryRegistryAccess(what): a store's registry keys; refused in such a test build (the PC's own stores).
// - Such a test build checks what a path really is: its final form (GetFinalPathNameByHandle) must be its own long name (no
//   junction or link above it), it must not be a reparse point, a file must have no second name (hard link), and the test
//   folder itself must resolve to itself; anything else is refused.
// - Test builds append "library access: <path>" or "library access refused: <path> (<why>)" to WRONGGPU_TEST_ACCESS_LOG
//   before the read (the tests' guard parses it). A new log starts with "run <WRONGGPU_TEST_RUN_ID>". With the tests' list
//   set, a read whose line cannot be written is refused (fail closed). Nothing of this in other builds.
// - TestAccessStart(program): a test build with the tests' list writes "process <pid> <program> start" (every program run
//   leaves a line, so the guard can tell a run that logged nothing). LibraryAccessCount counts the questions (every build).
// - TestLibraries(folder): such a test build (WRONGGPU_TEST_LIBRARIES set, its value in folder); false in other builds.
bool LibraryAccess(const wstring& path);
bool LibraryRegistryAccess(const wchar_t* what);
void TestAccessStart(const wchar_t* program);
size_t LibraryAccessCount();
bool TestLibraries(wstring* folder = nullptr);
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
