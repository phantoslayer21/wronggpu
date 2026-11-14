// SPDX-License-Identifier: MIT
// The elevated worker's file operations through held handles (elevated_job.h). Written for this project.
//
// While an FsGuard is active (ScopedFsGuard), the file helpers of util.h (Exists, ReadAll, EnsureDir, WriteFileAtomic,
// CopyChecked, MovePath, DeletePath, RemoveTree, ListFiles, ...) act through it instead of by path:
//   - the folders the job may change (its roots: the game, the player's local app data and Documents folders, the worker's
//     private folder) are opened when the job is checked, from the volume root one component at a time, never following a
//     link, and stay open until the job ends: every component is held without delete sharing, so none of them can be
//     renamed, replaced or deleted meanwhile;
//   - every path is opened the same way, below the nearest root's held handle (a path outside every root, which is only
//     ever read, below the volume root);
//   - a component that is a link (a junction, a symbolic link or any other reparse point that redirects a name) is refused,
//     the leaf too. Reparse points that only store a file differently and lead nowhere else (Windows' file compression,
//     OneDrive's files on demand, deduplication) are opened as the file itself and checked to be the same file;
//   - after each open, the path Windows reports for the handle (GetFinalPathNameByHandle) must be the path asked for;
//   - files are created next to their target under a new name and renamed over it through the handle, renamed and deleted
//     through their handle (SetFileInformationByHandle), and created relative to their folder's handle (NtCreateFile with a
//     root directory);
//   - nothing outside the roots is written.
// A refusal throws SetupError (Fail) with the path and the reason; an ordinary failure (not there, in use) is returned as
// with the path-based helpers.
#pragma once

#include <memory>

#include "util.h"

namespace wg {

class FsGuard {
public:
    FsGuard();
    ~FsGuard();
    FsGuard(const FsGuard&) = delete;
    FsGuard& operator=(const FsGuard&) = delete;

    // A folder this job may change: opened and held as described above. False with `why` when a component is a link, the
    // folder is not there or is not where `path` says (its final path is another).
    bool AddRoot(const wstring& path, wstring& why);
    std::vector<wstring> Roots() const;

    // ---- what util.h's helpers do while the guard is active -------------------------------------------------------------
    // Attributes of `path` (INVALID_FILE_ATTRIBUTES with `err` set: not there); `size`: the file's size.
    DWORD Attributes(const wstring& path, unsigned long long* size, DWORD& err);
    // An open handle (INVALID_HANDLE_VALUE with `err`): to read, or a new file to write (fails when the name exists).
    HANDLE OpenRead(const wstring& path, DWORD share, DWORD& err);
    HANDLE CreateNew(const wstring& path, DWORD& err);
    // A new folder; `sd`: its security descriptor (null: inherited). Fails when the name exists.
    bool CreateNewDir(const wstring& path, PSECURITY_DESCRIPTOR sd, DWORD& err);
    void EnsureDir(const wstring& dir, std::vector<wstring>* created);
    void WriteAtomic(const wstring& path, const void* data, size_t size);
    bool Copy(const wstring& src, const wstring& dst, DWORD& err);
    bool Unlink(const wstring& path, DWORD& err);  // a file
    bool RemoveDir(const wstring& path, DWORD& err);  // an empty folder
    void RemoveTree(const wstring& dir, const wstring& keepFile);
    bool Move(const wstring& from, const wstring& to, bool replace, bool copyAllowed, DWORD& err);
    bool SetAttributes(const wstring& path, DWORD attrs, DWORD& err);
    struct Entry { wstring name; DWORD attrs; unsigned long long size; };  // attrs: FILE_ATTRIBUTE_REPARSE_POINT only for a link
    bool List(const wstring& dir, std::vector<Entry>& out, DWORD& err);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// The guard util.h's helpers go through (null: by path, as everywhere but in the elevated worker's jobs).
FsGuard* ActiveFsGuard();
class ScopedFsGuard {
public:
    explicit ScopedFsGuard(FsGuard* g);
    ~ScopedFsGuard();
    ScopedFsGuard(const ScopedFsGuard&) = delete;
    ScopedFsGuard& operator=(const ScopedFsGuard&) = delete;
private:
    FsGuard* prev_;
};

// The converter library (wgc) opens its files by name. While a guard is active, a file it reads is first copied through
// the guard into the worker's private folder (ConverterFile), and a folder it writes into is a new folder there whose files
// are then copied through the guard into place (ConverterDir::Publish); so the converter itself only ever opens names in
// the private folder, which only administrators can write. Without a guard both are the paths themselves.
class ConverterFile {
public:
    ConverterFile(const wstring& file, const wstring& privateDir);
    ~ConverterFile();
    ConverterFile(const ConverterFile&) = delete;
    ConverterFile& operator=(const ConverterFile&) = delete;
    const wstring& path() const { return path_; }
private:
    wstring path_, dir_;
};
class ConverterDir {
public:
    ConverterDir(const wstring& dir, const wstring& privateDir);
    ~ConverterDir();
    ConverterDir(const ConverterDir&) = delete;
    ConverterDir& operator=(const ConverterDir&) = delete;
    const wstring& path() const { return path_; }  // where the converter writes
    void Publish();                                // what it wrote, copied into the folder named (no guard: nothing to do)
private:
    wstring dir_, path_;
};

// Whether a reparse point with this tag is a link (leads elsewhere), as the guard decides; the others only store a file
// differently (compression, files on demand, deduplication).
bool ReparseTagIsLink(DWORD tag);

// `p` as an absolute path in its plain form (C:\a\b or \\server\share\a, no \\?\ prefix, no trailing separator except a
// drive's root), or empty when it is not one.
wstring NormalPath(const wstring& p);
// The path Windows reports for the open handle `h`, in NormalPath's form (empty when it cannot say).
wstring FinalPathOfHandle(HANDLE h);
// The launcher's side: the path of the existing folder or file `p` as the worker will find it (links and short names
// resolved by Windows, as the player reaches it); a part that does not exist yet is kept as named. Empty stays empty.
wstring BoundPath(const wstring& p);

}  // namespace wg
