// SPDX-License-Identifier: MIT
// Linux layer internals shared by win32_posix*.cpp: the handle table, path and string conversion, error mapping.
#pragma once
#include <sys/stat.h>
#include <sys/types.h>

#include <mutex>
#include <string>
#include <vector>

#include "win32_posix.h"

namespace wgposix {

// One object type behind every HANDLE. Kind says which fields mean something.
enum class Kind { File, Pipe, Map, Process, Thread, Find, Job, Snapshot, Token, Mutex };
struct Handle {
    Kind kind;
    int fd = -1;                    // File, Pipe, Map (dup of the file)
    std::string path;               // File: for FILE_FLAG_DELETE_ON_CLOSE
    dev_t dev = 0; ino_t ino = 0;   // File: identity for the sharing table
    bool read = false, write = false, del = false;  // File: what it was opened for (sharing rules)
    DWORD share = 0;                // File: FILE_SHARE_* it allows others
    bool tracked = false;           // File: is in the sharing table
    bool deleteOnClose = false;
    bool inheritable = false;       // Pipe: SECURITY_ATTRIBUTES said inherit
    bool borrowed = false;          // Pipe: standard handle, the fd is not ours to close
    uint64_t mapSize = 0; DWORD mapProtect = 0;  // Map
    int pid = 0;                    // Process
    std::vector<int> pids;          // Job: processes assigned to it
    bool killOnClose = false;       // Job
    std::vector<WIN32_FIND_DATAW> found; size_t next = 0;  // Find
    std::vector<PROCESSENTRY32W> procs;                    // Snapshot
    explicit Handle(Kind k) : kind(k) {}
};
Handle* NewHandle(Kind k);
Handle* Lookup(HANDLE h, Kind k);   // null (and ERROR_INVALID_HANDLE) when h is not a live handle of that kind
void Forget(Handle* h);

// UTF-8 <-> wchar_t. Bytes that are not UTF-8 travel as U+DC80..U+DCFF and come back as the same bytes.
std::string ToUtf8(const wchar_t* s, size_t n);
std::wstring FromUtf8(const char* s, size_t n);
inline std::string ToUtf8(const std::wstring& s) { return ToUtf8(s.data(), s.size()); }
inline std::wstring FromUtf8(const std::string& s) { return FromUtf8(s.data(), s.size()); }

// A Windows path as a POSIX path ("\" to "/", "\\?\" gone, "Z:\" is "/"); false with `err` set for one that cannot exist here.
// Names that differ from the real ones only in case are matched to the real ones (NTFS ignores case; ext4 does not).
bool ToPosix(const wchar_t* w, std::string& out, DWORD& err);
bool ToPosix(const wchar_t* w, std::string& out);   // sets the last error
std::string FixCase(const std::string& p);
wint_t FoldChar(wint_t c);   // simple Unicode case folding, for names
std::wstring WinPath(const std::string& posix);   // an absolute POSIX path as "Z:\..."; a relative one as it is

DWORD ErrorFromErrno(int e, const std::string& path);
void SetErrorFromErrno(int e, const std::string& path);
inline bool Fail(DWORD e) { SetLastError(e); return false; }

DWORD AttributesOf(const struct stat& lst, const std::string& path);   // lst: from lstat
FILETIME FileTimeOf(const struct timespec& t);
struct timespec TimespecOf(const FILETIME& ft);
void FillFind(WIN32_FIND_DATAW& d, const struct stat& lst, const std::string& path, const std::string& name);

// sharing: does a delete, move or replace of the file with this identity clash with another open handle?
bool ShareAllowsDelete(const struct stat& st);
// a child process, reaped once however many handles name it
struct ChildState { bool exited = false; int code = 0; int forcedCode = -1; int refs = 0; /* open process handles */ bool suspended = false;
                  std::string image; /* the program a suspended start will exec, until it has */ };
bool PollChild(int pid, ChildState& out);
void KillChild(int pid, int code);

// CloseHandle of a job or a process handle
void ReleaseJob(Handle* h);
void ReleaseProcess(Handle* h);

// registry stores: default HKCU, loaded app hives
std::string HomeDir();
std::string XdgDataHome();
std::string XdgConfigHome();
std::string SteamRoot();   // "" when no Steam install is found

}  // namespace wgposix
