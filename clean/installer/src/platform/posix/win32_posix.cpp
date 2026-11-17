// SPDX-License-Identifier: MIT
// Linux: Windows file, string, time, environment and shell calls on POSIX (see win32_posix.h). Registry and processes are in
// win32_posix_registry.cpp and win32_posix_process.cpp.
#include <filesystem>
#include <dirent.h>
#include <locale.h>
#include <fcntl.h>
#include <pwd.h>
#include <sys/mman.h>
#include <sys/statvfs.h>
#include <sys/xattr.h>
#include <sys/time.h>

#include <algorithm>
#include <cerrno>
#include <cstdarg>
#include <map>
#include <unordered_set>

#include "win32_posix_internal.h"

extern char** environ;

namespace wgposix {

// ---- last error --------------------------------------------------------------------------------------------------------
static thread_local DWORD g_lastError = 0;

// ---- handle table ------------------------------------------------------------------------------------------------------
static std::mutex g_handleLock;
// The tables here are never destroyed: the program's own globals (wg::app::App's Context cleans up in its destructor, which runs
// during exit) still call into this layer, and a table already destroyed by then is a crash at exit.
static auto& g_handles = *new std::unordered_set<Handle*>;
Handle* NewHandle(Kind k) {
    Handle* h = new Handle(k);
    std::lock_guard<std::mutex> g(g_handleLock);
    g_handles.insert(h);
    return h;
}
Handle* Lookup(HANDLE hv, Kind k) {
    Handle* h = static_cast<Handle*>(hv);
    std::lock_guard<std::mutex> g(g_handleLock);
    if (!hv || hv == INVALID_HANDLE_VALUE || !g_handles.count(h) || h->kind != k) { g_lastError = ERROR_INVALID_HANDLE; return nullptr; }
    return h;
}
static Handle* LookupAny(HANDLE hv) {
    Handle* h = static_cast<Handle*>(hv);
    std::lock_guard<std::mutex> g(g_handleLock);
    if (!hv || hv == INVALID_HANDLE_VALUE || !g_handles.count(h)) { g_lastError = ERROR_INVALID_HANDLE; return nullptr; }
    return h;
}
void Forget(Handle* h) {
    {
        std::lock_guard<std::mutex> g(g_handleLock);
        g_handles.erase(h);
    }
    delete h;
}

// ---- UTF-8 ---------------------------------------------------------------------------------------------------------------
std::string ToUtf8(const wchar_t* s, size_t n) {
    std::string o;
    o.reserve(n);
    for (size_t i = 0; i < n; ++i) {
        const uint32_t c = (uint32_t)s[i];
        if (c >= 0xDC80 && c <= 0xDCFF) o += (char)(c - 0xDC00);
        else if (c < 0x80) o += (char)c;
        else if (c < 0x800) { o += (char)(0xC0 | (c >> 6)); o += (char)(0x80 | (c & 0x3F)); }
        else if ((c >= 0xD800 && c <= 0xDFFF) || c > 0x10FFFF) o += "\xEF\xBF\xBD";
        else if (c < 0x10000) { o += (char)(0xE0 | (c >> 12)); o += (char)(0x80 | ((c >> 6) & 0x3F)); o += (char)(0x80 | (c & 0x3F)); }
        else { o += (char)(0xF0 | (c >> 18)); o += (char)(0x80 | ((c >> 12) & 0x3F)); o += (char)(0x80 | ((c >> 6) & 0x3F)); o += (char)(0x80 | (c & 0x3F)); }
    }
    return o;
}
// One sequence at s[i]: its length and code point, or 0 when the bytes there are not valid UTF-8.
static size_t DecodeOne(const unsigned char* s, size_t n, size_t i, uint32_t& cp) {
    const unsigned char b = s[i];
    if (b < 0x80) { cp = b; return 1; }
    size_t len = b >= 0xF0 && b <= 0xF4 ? 4 : b >= 0xE0 ? 3 : b >= 0xC2 && b < 0xE0 ? 2 : 0;
    if (b >= 0xF5 || !len || i + len > n) return 0;
    cp = b & (0xFF >> (len + 1));
    for (size_t k = 1; k < len; ++k) {
        if ((s[i + k] & 0xC0) != 0x80) return 0;
        cp = (cp << 6) | (s[i + k] & 0x3F);
    }
    if ((len == 3 && (cp < 0x800 || (cp >= 0xD800 && cp <= 0xDFFF))) || (len == 4 && (cp < 0x10000 || cp > 0x10FFFF))) return 0;
    return len;
}
std::wstring FromUtf8(const char* s, size_t n) {
    std::wstring w;
    w.reserve(n);
    const unsigned char* u = reinterpret_cast<const unsigned char*>(s);
    for (size_t i = 0; i < n;) {
        uint32_t cp;
        const size_t len = DecodeOne(u, n, i, cp);
        if (!len) { w += (wchar_t)(0xDC00 + u[i]); ++i; }
        else { w += (wchar_t)cp; i += len; }
    }
    return w;
}

// Case folding for names: Unicode simple case mapping through a C.UTF-8 locale object of our own, so the process locale stays as
// it is; without that locale, whatever towlower does in the current one.
wint_t FoldChar(wint_t c) {
    static const locale_t loc = newlocale(LC_CTYPE_MASK, "C.UTF-8", (locale_t)0);
    return loc ? towlower_l(towupper_l(c, loc), loc) : towlower(towupper(c));
}

// ---- paths -------------------------------------------------------------------------------------------------------------
std::string FixCase(const std::string& p) {
    struct stat st;
    if (p.empty() || lstat(p.c_str(), &st) == 0) return p;
    std::string out;
    size_t pos = 0;
    bool broken = false;
    if (p[0] == '/') { out = "/"; pos = 1; }
    for (;;) {
        size_t e = p.find('/', pos);
        const bool last = e == std::string::npos;
        if (last) e = p.size();
        std::string comp = p.substr(pos, e - pos);
        if (!comp.empty() && comp != "." && comp != ".." && !broken && lstat((out + comp).c_str(), &st) != 0) {
            bool found = false;
            const std::wstring wcomp = FromUtf8(comp);
            if (DIR* d = opendir(out.empty() ? "." : out.c_str())) {
                while (dirent* ent = readdir(d)) {
                    if (_wcsicmp(FromUtf8(ent->d_name, std::strlen(ent->d_name)).c_str(), wcomp.c_str()) == 0) { comp = ent->d_name; found = true; break; }
                }
                closedir(d);
            }
            if (!found) broken = true;
        }
        out += comp;
        if (last) break;
        out += '/';
        pos = e + 1;
    }
    return out;
}
bool ToPosix(const wchar_t* w, std::string& out, DWORD& err) {
    if (!w || !*w) { err = ERROR_PATH_NOT_FOUND; return false; }
    std::wstring s(w);
    if (s.rfind(L"\\\\?\\", 0) == 0 || s.rfind(L"//?/", 0) == 0) {
        s.erase(0, 4);
        if (s.size() >= 4 && _wcsnicmp(s.c_str(), L"UNC\\", 4) == 0) { err = ERROR_PATH_NOT_FOUND; return false; }
    } else if (s.rfind(L"\\\\.\\", 0) == 0 || s.rfind(L"//./", 0) == 0) {
        err = ERROR_PATH_NOT_FOUND;
        return false;
    }
    if (s.size() >= 2 && s[1] == L':' && ((s[0] >= L'A' && s[0] <= L'Z') || (s[0] >= L'a' && s[0] <= L'z'))) {
        if ((s[0] == L'Z' || s[0] == L'z') && (s.size() == 2 || s[2] == L'\\' || s[2] == L'/')) { s.erase(0, 2); if (s.empty()) s = L"/"; }
        else { err = ERROR_PATH_NOT_FOUND; return false; }
    }
    // names Windows refuses (ERROR_INVALID_NAME), so the core's handling of them is the one it has there
    if (s.find_first_of(L"<>\"|?*") != std::wstring::npos || std::any_of(s.begin(), s.end(), [](wchar_t c) { return c < 0x20; })) { err = ERROR_INVALID_NAME; return false; }
    for (auto& c : s) if (c == L'\\') c = L'/';
    out = FixCase(ToUtf8(s));
    return true;
}
bool ToPosix(const wchar_t* w, std::string& out) {
    DWORD e = 0;
    if (ToPosix(w, out, e)) return true;
    g_lastError = e;
    return false;
}

DWORD ErrorFromErrno(int e, const std::string& path) {
    switch (e) {
        case ENOENT: {
            const size_t cut = path.find_last_of('/');
            struct stat st;
            if (cut != std::string::npos && cut > 0 && stat(path.substr(0, cut).c_str(), &st) != 0) return ERROR_PATH_NOT_FOUND;
            return ERROR_FILE_NOT_FOUND;
        }
        case ENOTDIR: return ERROR_PATH_NOT_FOUND;
        case EACCES: case EPERM: case EROFS: case EISDIR: case EBADF: return ERROR_ACCESS_DENIED;
        case EEXIST: return ERROR_ALREADY_EXISTS;
        case ENOTEMPTY: return ERROR_DIR_NOT_EMPTY;
        case ENOSPC: case EDQUOT: return ERROR_DISK_FULL;
        case EXDEV: return ERROR_NOT_SAME_DEVICE;
        case EBUSY: case ETXTBSY: return ERROR_SHARING_VIOLATION;
        case EMFILE: case ENFILE: return ERROR_TOO_MANY_OPEN_FILES;
        case ENAMETOOLONG: return ERROR_FILENAME_EXCED_RANGE;
        case EFBIG: return ERROR_FILE_TOO_LARGE;
        case EPIPE: return ERROR_BROKEN_PIPE;
        case ENOMEM: return ERROR_NOT_ENOUGH_MEMORY;
        case EINVAL: return ERROR_INVALID_PARAMETER;
        case ENOEXEC: return ERROR_BAD_EXE_FORMAT;
        default: return ERROR_GEN_FAILURE;
    }
}
void SetErrorFromErrno(int e, const std::string& path) { g_lastError = ErrorFromErrno(e, path); }

// The hidden attribute has no mode bit; it is kept as a user attribute of the file (a file system without them just never shows it).
static const char* const kHiddenXattr = "user.wronggpu.hidden";
static bool IsHiddenPath(const std::string& path) { return lgetxattr(path.c_str(), kHiddenXattr, nullptr, 0) >= 0; }
static void SetHiddenPath(const std::string& path, bool hidden) {
    if (hidden) lsetxattr(path.c_str(), kHiddenXattr, "1", 1, 0);
    else lremovexattr(path.c_str(), kHiddenXattr);
}
DWORD AttributesOf(const struct stat& lst, const std::string& path) {
    DWORD a = 0;
    if (!S_ISLNK(lst.st_mode) && IsHiddenPath(path)) a |= FILE_ATTRIBUTE_HIDDEN;
    if (S_ISLNK(lst.st_mode)) {
        a |= FILE_ATTRIBUTE_REPARSE_POINT;
        struct stat t;
        if (stat(path.c_str(), &t) == 0 && S_ISDIR(t.st_mode)) a |= FILE_ATTRIBUTE_DIRECTORY;
    } else if (S_ISDIR(lst.st_mode)) {
        a |= FILE_ATTRIBUTE_DIRECTORY;
    } else if (!(lst.st_mode & S_IWUSR)) {
        a |= FILE_ATTRIBUTE_READONLY;
    }
    return a ? a : FILE_ATTRIBUTE_NORMAL;
}
// The read-only attribute as a mode: no write bit at all, or the owner's back. SetFileAttributesW and SetFileInformationByHandle share it.
static mode_t ModeWithReadOnly(mode_t mode, bool readOnly) {
    const mode_t m = mode & 07777;
    return readOnly ? (m & ~(mode_t)(S_IWUSR | S_IWGRP | S_IWOTH)) : (m | S_IWUSR);
}
FILETIME FileTimeOf(const struct timespec& t) {
    const uint64_t v = ((uint64_t)((int64_t)t.tv_sec + 11644473600LL)) * 10000000ull + (uint64_t)t.tv_nsec / 100;
    return FILETIME{(DWORD)(v & 0xFFFFFFFFu), (DWORD)(v >> 32)};
}
struct timespec TimespecOf(const FILETIME& ft) {
    const uint64_t v = ((uint64_t)ft.dwHighDateTime << 32) | ft.dwLowDateTime;
    struct timespec t;
    t.tv_sec = (time_t)((int64_t)(v / 10000000ull) - 11644473600LL);
    t.tv_nsec = (long)((v % 10000000ull) * 100);
    return t;
}
void FillFind(WIN32_FIND_DATAW& d, const struct stat& lst, const std::string& path, const std::string& name) {
    std::memset(&d, 0, sizeof(d));
    d.dwFileAttributes = AttributesOf(lst, path);
    d.ftCreationTime = FileTimeOf(lst.st_ctim);
    d.ftLastAccessTime = FileTimeOf(lst.st_atim);
    d.ftLastWriteTime = FileTimeOf(lst.st_mtim);
    const uint64_t size = (d.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) ? 0 : (uint64_t)lst.st_size;
    d.nFileSizeHigh = (DWORD)(size >> 32);
    d.nFileSizeLow = (DWORD)(size & 0xFFFFFFFFu);
    const std::wstring w = FromUtf8(name);
    wcsncpy(d.cFileName, w.c_str(), MAX_PATH - 1);
}

// ---- sharing (what Windows' share modes do between handles of this process) ------------------------------------------------
static std::mutex g_shareLock;
static auto& g_shared = *new std::vector<Handle*>;
static bool Conflicts(dev_t dev, ino_t ino, bool wantR, bool wantW, bool wantD, DWORD share) {
    for (Handle* e : g_shared) {
        if (e->dev != dev || e->ino != ino) continue;
        if ((wantR && !(e->share & FILE_SHARE_READ)) || (wantW && !(e->share & FILE_SHARE_WRITE)) || (wantD && !(e->share & FILE_SHARE_DELETE)) ||
            (e->read && !(share & FILE_SHARE_READ)) || (e->write && !(share & FILE_SHARE_WRITE)) || (e->del && !(share & FILE_SHARE_DELETE)))
            return true;
    }
    return false;
}
bool ShareAllowsDelete(const struct stat& st) {
    std::lock_guard<std::mutex> g(g_shareLock);
    return !Conflicts(st.st_dev, st.st_ino, false, false, true, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE);
}

}  // namespace wgposix

using namespace wgposix;

// The Windows spelling of an absolute POSIX path: drive Z:, backslashes (the core's path code, and FullPath's equality tests,
// are written for that form). Relative paths come back as they are.
std::wstring wgposix::WinPath(const std::string& posix) {
    std::wstring w = FromUtf8(posix);
    if (w.empty() || w[0] != L'/') return w;
    for (auto& c : w) if (c == L'/') c = L'\\';
    return L"Z:" + w;
}

DWORD GetLastError() { return g_lastError; }
void SetLastError(DWORD e) { g_lastError = e; }

// ---- files -------------------------------------------------------------------------------------------------------------
HANDLE CreateFileW(LPCWSTR name, DWORD access, DWORD share, LPSECURITY_ATTRIBUTES, DWORD disposition, DWORD flags, HANDLE) {
    std::string p;
    if (!ToPosix(name, p)) return INVALID_HANDLE_VALUE;
    const bool wantR = access & (GENERIC_READ | GENERIC_ALL | FILE_READ_DATA);
    const bool wantW = access & (GENERIC_WRITE | GENERIC_ALL | FILE_WRITE_DATA | FILE_APPEND_DATA);
    const bool deleteOnClose = flags & FILE_FLAG_DELETE_ON_CLOSE;
    const bool wantD = (access & (DELETE | GENERIC_ALL)) || deleteOnClose;
    struct stat lst, st;
    const bool exists = lstat(p.c_str(), &lst) == 0, linked = exists && S_ISLNK(lst.st_mode);
    const bool target = exists && (!linked || stat(p.c_str(), &st) == 0);
    if (target && !linked) st = lst;
    if (target && S_ISDIR(st.st_mode)) {
        if (!(flags & FILE_FLAG_BACKUP_SEMANTICS) || wantW || disposition == CREATE_ALWAYS || disposition == CREATE_NEW || disposition == TRUNCATE_EXISTING) {
            SetLastError(disposition == CREATE_NEW ? ERROR_FILE_EXISTS : ERROR_ACCESS_DENIED);
            return INVALID_HANDLE_VALUE;
        }
    }
    if (disposition == CREATE_NEW && exists) { SetLastError(ERROR_FILE_EXISTS); return INVALID_HANDLE_VALUE; }
    if ((disposition == OPEN_EXISTING || disposition == TRUNCATE_EXISTING) && !target) {
        SetErrorFromErrno(ENOENT, p);
        return INVALID_HANDLE_VALUE;
    }
    const bool truncate = disposition == CREATE_ALWAYS || disposition == TRUNCATE_EXISTING;
    if (target && !S_ISDIR(st.st_mode) && (wantW || truncate) && !(st.st_mode & S_IWUSR)) { SetLastError(ERROR_ACCESS_DENIED); return INVALID_HANDLE_VALUE; }
    int oflags = O_CLOEXEC | (wantR && wantW ? O_RDWR : wantW ? O_WRONLY : O_RDONLY);
    if ((access & FILE_APPEND_DATA) && !(access & (GENERIC_WRITE | GENERIC_ALL | FILE_WRITE_DATA))) oflags |= O_APPEND;   // append-only access
    if (disposition == CREATE_NEW) oflags |= O_CREAT | O_EXCL;
    else if (disposition == CREATE_ALWAYS || disposition == OPEN_ALWAYS) oflags |= O_CREAT;
    if (linked && (flags & FILE_FLAG_OPEN_REPARSE_POINT)) oflags = O_CLOEXEC | O_PATH | O_NOFOLLOW;
    const int fd = open(p.c_str(), oflags, (flags & FILE_ATTRIBUTE_READONLY) ? 0444 : 0666);
    if (fd < 0) {
        if (errno == EEXIST && disposition == CREATE_NEW) SetLastError(ERROR_FILE_EXISTS);
        else SetErrorFromErrno(errno, p);
        return INVALID_HANDLE_VALUE;
    }
    struct stat fs;
    if (fstat(fd, &fs) != 0) { SetErrorFromErrno(errno, p); close(fd); return INVALID_HANDLE_VALUE; }
    Handle* h = NewHandle(Kind::File);
    h->fd = fd; h->path = p; h->dev = fs.st_dev; h->ino = fs.st_ino;
    h->read = wantR; h->write = wantW; h->del = wantD; h->share = share; h->deleteOnClose = deleteOnClose;
    if ((wantR || wantW || wantD) && !S_ISDIR(fs.st_mode)) {
        std::lock_guard<std::mutex> g(g_shareLock);
        if (Conflicts(fs.st_dev, fs.st_ino, wantR, wantW, wantD, share)) {
            close(fd);
            Forget(h);
            SetLastError(ERROR_SHARING_VIOLATION);
            return INVALID_HANDLE_VALUE;
        }
        g_shared.push_back(h);
        h->tracked = true;
    }
    if (truncate && exists && S_ISREG(fs.st_mode) && fs.st_size > 0 && ftruncate(fd, 0) != 0) {
        SetErrorFromErrno(errno, p);
        CloseHandle(h);
        return INVALID_HANDLE_VALUE;
    }
    SetLastError(exists && (disposition == CREATE_ALWAYS || disposition == OPEN_ALWAYS) ? (DWORD)ERROR_ALREADY_EXISTS : 0u);
    return h;
}

BOOL ReadFile(HANDLE hv, LPVOID buf, DWORD n, LPDWORD got, OVERLAPPED*) {
    Handle* h = LookupAny(hv);
    if (!h || (h->kind != Kind::File && h->kind != Kind::Pipe)) return Fail(ERROR_INVALID_HANDLE);
    ssize_t r;
    do r = read(h->fd, buf, n); while (r < 0 && errno == EINTR);
    if (r < 0) { SetErrorFromErrno(errno, ""); return FALSE; }
    if (got) *got = (DWORD)r;
    if (r == 0 && h->kind == Kind::Pipe && n) return Fail(ERROR_BROKEN_PIPE);
    return TRUE;
}
BOOL WriteFile(HANDLE hv, LPCVOID buf, DWORD n, LPDWORD wrote, OVERLAPPED*) {
    Handle* h = LookupAny(hv);
    if (!h || (h->kind != Kind::File && h->kind != Kind::Pipe)) return Fail(ERROR_INVALID_HANDLE);
    DWORD done = 0;
    while (done < n) {
        const ssize_t r = write(h->fd, static_cast<const char*>(buf) + done, n - done);
        if (r < 0) {
            if (errno == EINTR) continue;
            if (wrote) *wrote = done;
            SetErrorFromErrno(errno, "");
            return FALSE;
        }
        done += (DWORD)r;
    }
    if (wrote) *wrote = done;
    return TRUE;
}
BOOL CloseHandle(HANDLE hv) {
    Handle* h = LookupAny(hv);
    if (!h) return FALSE;
    switch (h->kind) {
        case Kind::File:
            if (h->tracked) {
                std::lock_guard<std::mutex> g(g_shareLock);
                g_shared.erase(std::remove(g_shared.begin(), g_shared.end(), h), g_shared.end());
            }
            if (h->fd >= 0) close(h->fd);
            if (h->deleteOnClose) {   // the file this handle opened, not whatever was put under its name since
                struct stat now;
                if (lstat(h->path.c_str(), &now) == 0 && now.st_dev == h->dev && now.st_ino == h->ino) unlink(h->path.c_str());
            }
            break;
        case Kind::Mutex: CloseMutex(h); if (h->fd >= 0) close(h->fd); break;
        case Kind::Pipe: case Kind::Map: if (h->fd >= 0 && !h->borrowed) close(h->fd); break;
        case Kind::Job: ReleaseJob(h); break;
        case Kind::Process: ReleaseProcess(h); break;
        default: break;
    }
    Forget(h);
    return TRUE;
}
BOOL FlushFileBuffers(HANDLE hv) {
    Handle* h = Lookup(hv, Kind::File);
    if (!h) return FALSE;
    if (fsync(h->fd) != 0 && errno != EINVAL && errno != EROFS) { SetErrorFromErrno(errno, h->path); return FALSE; }
    return TRUE;
}
BOOL GetFileSizeEx(HANDLE hv, LARGE_INTEGER* size) {
    Handle* h = Lookup(hv, Kind::File);
    struct stat st;
    if (!h) return FALSE;
    if (fstat(h->fd, &st) != 0) { SetErrorFromErrno(errno, h->path); return FALSE; }
    size->QuadPart = S_ISDIR(st.st_mode) ? 0 : st.st_size;
    return TRUE;
}
BOOL SetFilePointerEx(HANDLE hv, LARGE_INTEGER dist, LARGE_INTEGER* newPos, DWORD method) {
    Handle* h = Lookup(hv, Kind::File);
    if (!h) return FALSE;
    const off_t r = lseek(h->fd, (off_t)dist.QuadPart, method == FILE_BEGIN ? SEEK_SET : method == FILE_CURRENT ? SEEK_CUR : SEEK_END);
    if (r < 0) { SetErrorFromErrno(errno, h->path); return FALSE; }
    if (newPos) newPos->QuadPart = r;
    return TRUE;
}
BOOL GetFileTime(HANDLE hv, FILETIME* created, FILETIME* accessed, FILETIME* written) {
    Handle* h = Lookup(hv, Kind::File);
    struct stat st;
    if (!h) return FALSE;
    if (fstat(h->fd, &st) != 0) { SetErrorFromErrno(errno, h->path); return FALSE; }
    if (created) *created = FileTimeOf(st.st_ctim);
    if (accessed) *accessed = FileTimeOf(st.st_atim);
    if (written) *written = FileTimeOf(st.st_mtim);
    return TRUE;
}
BOOL SetFileTime(HANDLE hv, const FILETIME*, const FILETIME* accessed, const FILETIME* written) {
    Handle* h = Lookup(hv, Kind::File);
    if (!h) return FALSE;
    struct timespec ts[2] = {{0, UTIME_OMIT}, {0, UTIME_OMIT}};
    if (accessed) ts[0] = TimespecOf(*accessed);
    if (written) ts[1] = TimespecOf(*written);
    if (futimens(h->fd, ts) != 0) { SetErrorFromErrno(errno, h->path); return FALSE; }
    return TRUE;
}
BOOL GetFileInformationByHandle(HANDLE hv, BY_HANDLE_FILE_INFORMATION* info) {
    Handle* h = Lookup(hv, Kind::File);
    struct stat st;
    if (!h) return FALSE;
    if (fstat(h->fd, &st) != 0) { SetErrorFromErrno(errno, h->path); return FALSE; }
    std::memset(info, 0, sizeof(*info));
    struct stat lst;
    info->dwFileAttributes = lstat(h->path.c_str(), &lst) == 0 && lst.st_ino == st.st_ino ? AttributesOf(lst, h->path) : FILE_ATTRIBUTE_NORMAL;
    info->ftCreationTime = FileTimeOf(st.st_ctim);
    info->ftLastAccessTime = FileTimeOf(st.st_atim);
    info->ftLastWriteTime = FileTimeOf(st.st_mtim);
    info->dwVolumeSerialNumber = (DWORD)st.st_dev;
    info->nFileSizeHigh = (DWORD)((uint64_t)st.st_size >> 32);
    info->nFileSizeLow = (DWORD)((uint64_t)st.st_size & 0xFFFFFFFFu);
    info->nNumberOfLinks = (DWORD)st.st_nlink;
    info->nFileIndexHigh = (DWORD)((uint64_t)st.st_ino >> 32);
    info->nFileIndexLow = (DWORD)((uint64_t)st.st_ino & 0xFFFFFFFFu);
    return TRUE;
}
BOOL GetFileInformationByHandleEx(HANDLE hv, FILE_INFO_BY_HANDLE_CLASS cls, LPVOID info, DWORD size) {
    Handle* h = Lookup(hv, Kind::File);
    struct stat st;
    if (!h) return FALSE;
    if (cls != FileIdInfo && cls != FileBasicInfo) return Fail(ERROR_INVALID_PARAMETER);
    if (size < (cls == FileIdInfo ? sizeof(FILE_ID_INFO) : sizeof(FILE_BASIC_INFO))) return Fail(ERROR_INSUFFICIENT_BUFFER);
    if (fstat(h->fd, &st) != 0) { SetErrorFromErrno(errno, h->path); return FALSE; }
    if (cls == FileBasicInfo) {
        FILE_BASIC_INFO* bi = static_cast<FILE_BASIC_INFO*>(info);
        std::memset(bi, 0, sizeof(*bi));
        struct stat lst;
        bi->FileAttributes = lstat(h->path.c_str(), &lst) == 0 && lst.st_ino == st.st_ino ? AttributesOf(lst, h->path) : FILE_ATTRIBUTE_NORMAL;
        const FILETIME c = FileTimeOf(st.st_ctim), a = FileTimeOf(st.st_atim), w = FileTimeOf(st.st_mtim);
        bi->CreationTime.QuadPart = (LONGLONG)(((uint64_t)c.dwHighDateTime << 32) | c.dwLowDateTime);   // as GetFileTime: the status change time
        bi->ChangeTime = bi->CreationTime;
        bi->LastAccessTime.QuadPart = (LONGLONG)(((uint64_t)a.dwHighDateTime << 32) | a.dwLowDateTime);
        bi->LastWriteTime.QuadPart = (LONGLONG)(((uint64_t)w.dwHighDateTime << 32) | w.dwLowDateTime);
        return TRUE;
    }
    FILE_ID_INFO* fi = static_cast<FILE_ID_INFO*>(info);
    std::memset(fi, 0, sizeof(*fi));
    fi->VolumeSerialNumber = (ULONGLONG)st.st_dev;
    const uint64_t ino = (uint64_t)st.st_ino;
    std::memcpy(fi->FileId.Identifier, &ino, sizeof(ino));
    return TRUE;
}
// FileRenameInfo (win32_posix.h has what it guarantees). The handle's name now is the one /proc/self/fd gives (renames since
// the open included), used only while it still names the handle's file.
static BOOL RenameByHandle(Handle* h, const struct stat& st, const FILE_RENAME_INFO* ri, DWORD size) {
    const size_t head = offsetof(FILE_RENAME_INFO, FileName);
    if (size < head || ri->FileNameLength % sizeof(WCHAR) || size < head + ri->FileNameLength || !ri->FileNameLength) return Fail(ERROR_INVALID_PARAMETER);
    if (ri->RootDirectory) return Fail(ERROR_INVALID_PARAMETER);   // a name relative to a folder handle: no caller needs it
    if (!h->del) return Fail(ERROR_ACCESS_DENIED);                 // renaming needs DELETE access
    std::wstring name(ri->FileName, ri->FileNameLength / sizeof(WCHAR));
    if (name.rfind(L"\\??\\", 0) == 0) name.replace(0, 4, L"\\\\?\\");   // the NT form of a full path
    std::string to, from = h->path;
    if (!ToPosix(name.c_str(), to)) return FALSE;
    char cur[4096];
    const ssize_t n = readlink(("/proc/self/fd/" + std::to_string(h->fd)).c_str(), cur, sizeof(cur) - 1);
    if (n > 0) from.assign(cur, (size_t)n);
    struct stat now, dest;
    if (from.empty() || lstat(from.c_str(), &now) != 0 || now.st_dev != st.st_dev || now.st_ino != st.st_ino) return Fail(ERROR_FILE_NOT_FOUND);
    if (lstat(to.c_str(), &dest) == 0) {
        if (dest.st_dev == st.st_dev && dest.st_ino == st.st_ino) { h->path = to; return TRUE; }   // it has that name already
        if (!ri->ReplaceIfExists) return Fail(ERROR_ALREADY_EXISTS);
        if (S_ISDIR(dest.st_mode) || (!S_ISLNK(dest.st_mode) && !(dest.st_mode & S_IWUSR))) return Fail(ERROR_ACCESS_DENIED);   // a folder, a read-only file
        if (!ShareAllowsDelete(dest)) return Fail(ERROR_ACCESS_DENIED);   // a file open here without delete sharing (Windows: access denied)
    }
    int r;
    if (ri->ReplaceIfExists) {
        r = rename(from.c_str(), to.c_str());
    } else {
        r = renameat2(AT_FDCWD, from.c_str(), AT_FDCWD, to.c_str(), RENAME_NOREPLACE);
        if (r != 0 && (errno == EINVAL || errno == ENOSYS)) {   // a file system without RENAME_NOREPLACE: a new link never replaces either
            r = link(from.c_str(), to.c_str());
            if (r == 0) {
                unlink(from.c_str());
            } else if (errno == EPERM || errno == EOPNOTSUPP) {   // nor hard links (FAT, exFAT, a folder): checked, then renamed
                struct stat x;
                if (lstat(to.c_str(), &x) == 0) { errno = EEXIST; r = -1; }
                else r = rename(from.c_str(), to.c_str());
            }
        }
    }
    if (r != 0) {
        const int e = errno;
        if (e == EEXIST || e == ENOTEMPTY) return Fail(ERROR_ALREADY_EXISTS);
        SetErrorFromErrno(e, to);
        return FALSE;
    }
    h->path = to;   // a later delete through the handle, or its delete on close, goes to the new name
    return TRUE;
}
// Attributes and times through a handle (a time of 0 and attributes of 0 stay as they are), a rename, and the two ways to delete through one.
BOOL SetFileInformationByHandle(HANDLE hv, FILE_INFO_BY_HANDLE_CLASS cls, LPVOID info, DWORD size) {
    Handle* h = Lookup(hv, Kind::File);
    if (!h) return FALSE;
    struct stat st;
    if (fstat(h->fd, &st) != 0) { SetErrorFromErrno(errno, h->path); return FALSE; }
    if (cls == FileBasicInfo) {
        if (size < sizeof(FILE_BASIC_INFO)) return Fail(ERROR_BAD_LENGTH);
        const FILE_BASIC_INFO* b = static_cast<const FILE_BASIC_INFO*>(info);
        // Only the access and write times can be set here: the creation time has no place on this file system, and the change time is the kernel's.
        struct timespec ts[2] = {{0, UTIME_OMIT}, {0, UTIME_OMIT}};
        bool times = false;
        if (b->LastAccessTime.QuadPart > 0) { ts[0] = TimespecOf(FILETIME{(DWORD)(b->LastAccessTime.QuadPart & 0xFFFFFFFFu), (DWORD)((uint64_t)b->LastAccessTime.QuadPart >> 32)}); times = true; }
        if (b->LastWriteTime.QuadPart > 0) { ts[1] = TimespecOf(FILETIME{(DWORD)(b->LastWriteTime.QuadPart & 0xFFFFFFFFu), (DWORD)((uint64_t)b->LastWriteTime.QuadPart >> 32)}); times = true; }
        if (b->FileAttributes && !h->path.empty()) SetHiddenPath(h->path, b->FileAttributes & FILE_ATTRIBUTE_HIDDEN);
        if (b->FileAttributes && !S_ISDIR(st.st_mode)) {   // (as SetFileAttributesW: a folder has no read-only bit here)
            const mode_t m = ModeWithReadOnly(st.st_mode, b->FileAttributes & FILE_ATTRIBUTE_READONLY);
            if (m != (st.st_mode & 07777) && fchmod(h->fd, m) != 0) { SetErrorFromErrno(errno, h->path); return FALSE; }
        }
        if (times && futimens(h->fd, ts) != 0) { SetErrorFromErrno(errno, h->path); return FALSE; }
        return TRUE;
    }
    if (cls == FileRenameInfo) return RenameByHandle(h, st, static_cast<const FILE_RENAME_INFO*>(info), size);
    if (cls != FileDispositionInfo && cls != FileDispositionInfoEx) return Fail(ERROR_INVALID_PARAMETER);
    if (size < (cls == FileDispositionInfo ? sizeof(FILE_DISPOSITION_INFO) : sizeof(FILE_DISPOSITION_INFO_EX))) return Fail(ERROR_BAD_LENGTH);
    bool del, posix = false, ignoreReadOnly = false;
    if (cls == FileDispositionInfo) {
        del = static_cast<const FILE_DISPOSITION_INFO*>(info)->DeleteFile != 0;
    } else {
        const DWORD f = static_cast<const FILE_DISPOSITION_INFO_EX*>(info)->Flags;
        del = f & FILE_DISPOSITION_FLAG_DELETE;
        posix = f & FILE_DISPOSITION_FLAG_POSIX_SEMANTICS;
        ignoreReadOnly = f & FILE_DISPOSITION_FLAG_IGNORE_READONLY_ATTRIBUTE;
    }
    if (!del) { h->deleteOnClose = false; return TRUE; }   // "do not delete" takes back an earlier request
    if (!h->del) return Fail(ERROR_ACCESS_DENIED);          // deleting needs DELETE access
    if (S_ISDIR(st.st_mode)) return Fail(ERROR_NOT_SUPPORTED);   // (folders are removed by name here)
    if (!ignoreReadOnly && !(st.st_mode & S_IWUSR)) return Fail(ERROR_ACCESS_DENIED);   // a read-only file is not deleted unless told to
    if (!posix) { h->deleteOnClose = true; return TRUE; }   // classic: the name goes when the handle closes
    struct stat now;
    if (h->path.empty() || lstat(h->path.c_str(), &now) != 0 || now.st_dev != st.st_dev || now.st_ino != st.st_ino) return Fail(ERROR_FILE_NOT_FOUND);
    if (unlink(h->path.c_str()) != 0) { SetErrorFromErrno(errno, h->path); return FALSE; }
    h->deleteOnClose = false;
    h->path.clear();   // the name is free; this handle no longer has one
    return TRUE;
}
BOOL SetEndOfFile(HANDLE hv) {
    Handle* h = Lookup(hv, Kind::File);
    if (!h) return FALSE;
    const int fl = fcntl(h->fd, F_GETFL);
    if (fl < 0 || (fl & O_PATH) || (fl & O_ACCMODE) == O_RDONLY) return Fail(ERROR_ACCESS_DENIED);   // truncating needs write access
    const off_t pos = lseek(h->fd, 0, SEEK_CUR);
    if (pos < 0) { SetErrorFromErrno(errno, h->path); return FALSE; }
    if (ftruncate(h->fd, pos) != 0) { SetErrorFromErrno(errno, h->path); return FALSE; }
    return TRUE;
}
DWORD GetFinalPathNameByHandleW(HANDLE hv, LPWSTR out, DWORD size, DWORD) {
    Handle* h = Lookup(hv, Kind::File);
    if (!h) return 0;
    char link[4096];
    const ssize_t n = readlink(("/proc/self/fd/" + std::to_string(h->fd)).c_str(), link, sizeof(link) - 1);
    if (n < 0) { SetErrorFromErrno(errno, h->path); return 0; }
    const std::wstring w = WinPath(std::string(link, (size_t)n));
    if (!out || size < w.size() + 1) return (DWORD)w.size() + 1;
    std::memcpy(out, w.c_str(), (w.size() + 1) * sizeof(wchar_t));
    return (DWORD)w.size();
}
DWORD GetFileAttributesW(LPCWSTR name) {
    std::string p;
    struct stat st;
    if (!ToPosix(name, p)) return INVALID_FILE_ATTRIBUTES;
    if (lstat(p.c_str(), &st) != 0) { SetErrorFromErrno(errno, p); return INVALID_FILE_ATTRIBUTES; }
    return AttributesOf(st, p);
}
BOOL GetFileAttributesExW(LPCWSTR name, GET_FILEEX_INFO_LEVELS, LPVOID data) {
    std::string p;
    struct stat st;
    if (!ToPosix(name, p)) return FALSE;
    if (lstat(p.c_str(), &st) != 0) { SetErrorFromErrno(errno, p); return FALSE; }
    WIN32_FIND_DATAW f;
    FillFind(f, st, p, "x");
    WIN32_FILE_ATTRIBUTE_DATA* d = static_cast<WIN32_FILE_ATTRIBUTE_DATA*>(data);
    d->dwFileAttributes = f.dwFileAttributes;
    d->ftCreationTime = f.ftCreationTime; d->ftLastAccessTime = f.ftLastAccessTime; d->ftLastWriteTime = f.ftLastWriteTime;
    d->nFileSizeHigh = f.nFileSizeHigh; d->nFileSizeLow = f.nFileSizeLow;
    return TRUE;
}
BOOL SetFileAttributesW(LPCWSTR name, DWORD attrs) {
    std::string p;
    struct stat st;
    if (!ToPosix(name, p)) return FALSE;
    if (stat(p.c_str(), &st) != 0) { SetErrorFromErrno(errno, p); return FALSE; }
    SetHiddenPath(p, attrs & FILE_ATTRIBUTE_HIDDEN);
    if (S_ISDIR(st.st_mode)) return TRUE;   // a read-only folder on Windows still takes files; chmod would not mean that
    const mode_t m = ModeWithReadOnly(st.st_mode, attrs & FILE_ATTRIBUTE_READONLY);
    if (m != (st.st_mode & 07777) && chmod(p.c_str(), m) != 0) { SetErrorFromErrno(errno, p); return FALSE; }
    return TRUE;
}
BOOL DeleteFileW(LPCWSTR name) {
    std::string p;
    struct stat st;
    if (!ToPosix(name, p)) return FALSE;
    if (lstat(p.c_str(), &st) != 0) { SetErrorFromErrno(errno, p); return FALSE; }
    if (S_ISDIR(st.st_mode)) return Fail(ERROR_ACCESS_DENIED);
    if (!S_ISLNK(st.st_mode) && !(st.st_mode & S_IWUSR)) return Fail(ERROR_ACCESS_DENIED);
    if (!ShareAllowsDelete(st)) return Fail(ERROR_SHARING_VIOLATION);
    if (unlink(p.c_str()) != 0) { SetErrorFromErrno(errno, p); return FALSE; }
    return TRUE;
}
static bool CopyData(const std::string& from, const std::string& to, mode_t keepMode, bool keepTimes, DWORD& err);
BOOL MoveFileExW(LPCWSTR from, LPCWSTR to, DWORD flags) {
    std::string a, b;
    struct stat sa, sb;
    if (!ToPosix(from, a)) return FALSE;
    if (flags & MOVEFILE_DELAY_UNTIL_REBOOT) return Fail(ERROR_CALL_NOT_IMPLEMENTED);
    if (!to) return Fail(ERROR_CALL_NOT_IMPLEMENTED);
    if (!ToPosix(to, b)) return FALSE;
    if (lstat(a.c_str(), &sa) != 0) { SetErrorFromErrno(errno, a); return FALSE; }
    const bool destExists = lstat(b.c_str(), &sb) == 0;
    if (!ShareAllowsDelete(sa)) return Fail(ERROR_SHARING_VIOLATION);
    if (destExists) {
        if (sa.st_dev == sb.st_dev && sa.st_ino == sb.st_ino && !S_ISLNK(sa.st_mode)) return TRUE;
        if (!(flags & MOVEFILE_REPLACE_EXISTING)) return Fail(ERROR_ALREADY_EXISTS);
        if (S_ISDIR(sb.st_mode) || S_ISDIR(sa.st_mode)) return Fail(ERROR_ACCESS_DENIED);
        if (!S_ISLNK(sb.st_mode) && !(sb.st_mode & S_IWUSR)) return Fail(ERROR_ACCESS_DENIED);
        if (!ShareAllowsDelete(sb)) return Fail(ERROR_SHARING_VIOLATION);
    }
    if (rename(a.c_str(), b.c_str()) != 0) {
        const int e = errno;
        if (e == EXDEV && (flags & MOVEFILE_COPY_ALLOWED) && S_ISREG(sa.st_mode)) {
            DWORD err = 0;
            if (!CopyData(a, b, sa.st_mode, true, err)) return Fail(err);
            unlink(a.c_str());
            return TRUE;
        }
        SetErrorFromErrno(e, e == ENOENT ? b : a);
        return FALSE;
    }
    if (flags & MOVEFILE_WRITE_THROUGH) {
        const int fd = open(b.c_str(), O_RDONLY | O_CLOEXEC);
        if (fd >= 0) { fsync(fd); close(fd); }
        const size_t cut = b.find_last_of('/');
        const std::string dir = cut == std::string::npos ? "." : cut == 0 ? "/" : b.substr(0, cut);
        const int dfd = open(dir.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC);
        if (dfd >= 0) { fsync(dfd); close(dfd); }
    }
    return TRUE;
}
// The replacement takes the replaced file's name in one rename, and the replaced file (if a backup name is given) takes the backup name.
// The failure codes say what is left: 1175, nothing moved; 1177, the replaced file is at the backup name and the replacement is
// still under its own. (Windows' 1176, the replaced file gone and the replacement not moved, cannot arise here.)
BOOL ReplaceFileW(LPCWSTR replaced, LPCWSTR replacement, LPCWSTR backup, DWORD flags, LPVOID exclude, LPVOID reserved) {
    std::string a, b, bak;
    struct stat sa, sb;
    if (exclude || reserved) return Fail(ERROR_INVALID_PARAMETER);
    if (!ToPosix(replaced, a) || !ToPosix(replacement, b)) return FALSE;
    if (backup && !ToPosix(backup, bak)) return FALSE;
    if (lstat(a.c_str(), &sa) != 0) { SetErrorFromErrno(errno, a); return FALSE; }
    if (lstat(b.c_str(), &sb) != 0) { SetErrorFromErrno(errno, b); return FALSE; }
    if (S_ISDIR(sa.st_mode) || S_ISDIR(sb.st_mode)) return Fail(ERROR_ACCESS_DENIED);
    if (!S_ISLNK(sa.st_mode) && !(sa.st_mode & S_IWUSR)) return Fail(ERROR_ACCESS_DENIED);   // a read-only file is not replaced
    if (!ShareAllowsDelete(sb)) return Fail(ERROR_SHARING_VIOLATION);
    if (!ShareAllowsDelete(sa)) return Fail(ERROR_UNABLE_TO_REMOVE_REPLACED);
    if (sa.st_dev == sb.st_dev && sa.st_ino == sb.st_ino) return Fail(ERROR_UNABLE_TO_REMOVE_REPLACED);
    if (!S_ISLNK(sa.st_mode)) {   // the replaced file's attributes stay (as Windows keeps them)
        chmod(b.c_str(), sa.st_mode & 07777);
        SetHiddenPath(b, IsHiddenPath(a));
    }
    // without a backup name one rename does it all (the name never goes missing); when it fails nothing has moved
    if (backup && rename(a.c_str(), bak.c_str()) != 0) return Fail(ERROR_UNABLE_TO_REMOVE_REPLACED);
    if (rename(b.c_str(), a.c_str()) != 0) return Fail(backup ? ERROR_UNABLE_TO_MOVE_REPLACEMENT_2 : ERROR_UNABLE_TO_REMOVE_REPLACED);
    if (flags & REPLACEFILE_WRITE_THROUGH) {
        const size_t cut = a.find_last_of('/');
        const std::string dir = cut == std::string::npos ? "." : cut == 0 ? "/" : a.substr(0, cut);
        const int dfd = open(dir.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC);
        if (dfd >= 0) { fsync(dfd); close(dfd); }
    }
    return TRUE;
}
static bool CopyData(const std::string& from, const std::string& to, mode_t keepMode, bool keepTimes, DWORD& err) {
    const int in = open(from.c_str(), O_RDONLY | O_CLOEXEC);
    if (in < 0) { err = ErrorFromErrno(errno, from); return false; }
    struct stat st;
    fstat(in, &st);
    const int out = open(to.c_str(), O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0666);
    if (out < 0) { err = ErrorFromErrno(errno, to); close(in); return false; }
    char buf[1 << 16];
    bool ok = true;
    for (;;) {
        const ssize_t r = read(in, buf, sizeof(buf));
        if (r < 0) { if (errno == EINTR) continue; err = ErrorFromErrno(errno, from); ok = false; break; }
        if (r == 0) break;
        for (ssize_t done = 0; done < r;) {
            const ssize_t w = write(out, buf + done, (size_t)(r - done));
            if (w < 0) { if (errno == EINTR) continue; err = ErrorFromErrno(errno, to); ok = false; break; }
            done += w;
        }
        if (!ok) break;
    }
    if (ok && keepTimes) { struct timespec ts[2] = {st.st_atim, st.st_mtim}; futimens(out, ts); }
    if (ok) fchmod(out, keepMode & 07777);
    close(in);
    if (close(out) != 0 && ok) { err = ErrorFromErrno(errno, to); ok = false; }
    if (!ok) unlink(to.c_str());
    return ok;
}
BOOL CopyFileW(LPCWSTR from, LPCWSTR to, BOOL failIfExists) {
    std::string a, b;
    struct stat sa, sb;
    if (!ToPosix(from, a) || !ToPosix(to, b)) return FALSE;
    if (stat(a.c_str(), &sa) != 0) { SetErrorFromErrno(errno, a); return FALSE; }
    if (S_ISDIR(sa.st_mode)) return Fail(ERROR_ACCESS_DENIED);
    if (stat(b.c_str(), &sb) == 0) {
        if (failIfExists) return Fail(ERROR_FILE_EXISTS);
        if (S_ISDIR(sb.st_mode) || !(sb.st_mode & S_IWUSR)) return Fail(ERROR_ACCESS_DENIED);
        if (sa.st_dev == sb.st_dev && sa.st_ino == sb.st_ino) return Fail(ERROR_SHARING_VIOLATION);
        std::lock_guard<std::mutex> g(g_shareLock);
        if (Conflicts(sb.st_dev, sb.st_ino, false, true, false, 0)) return Fail(ERROR_SHARING_VIOLATION);
    }
    {
        std::lock_guard<std::mutex> g(g_shareLock);
        for (Handle* e : g_shared)   // the source is read with FILE_SHARE_READ: a writer that denies readers blocks the copy
            if (e->dev == sa.st_dev && e->ino == sa.st_ino && !(e->share & FILE_SHARE_READ)) return Fail(ERROR_SHARING_VIOLATION);
    }
    DWORD err = 0;
    if (!CopyData(a, b, sa.st_mode, true, err)) return Fail(err);
    return TRUE;
}
BOOL CreateDirectoryW(LPCWSTR name, LPSECURITY_ATTRIBUTES) {
    std::string p;
    if (!ToPosix(name, p)) return FALSE;
    if (mkdir(p.c_str(), 0777) != 0) { SetErrorFromErrno(errno, p); return FALSE; }
    return TRUE;
}
BOOL RemoveDirectoryW(LPCWSTR name) {
    std::string p;
    struct stat st;
    if (!ToPosix(name, p)) return FALSE;
    if (lstat(p.c_str(), &st) != 0) { SetErrorFromErrno(errno, p); return FALSE; }
    if (S_ISLNK(st.st_mode)) {   // a junction or directory symbolic link goes without touching its target
        struct stat t;
        if (stat(p.c_str(), &t) == 0 && !S_ISDIR(t.st_mode)) return Fail(ERROR_DIRECTORY);
        if (unlink(p.c_str()) != 0) { SetErrorFromErrno(errno, p); return FALSE; }
        return TRUE;
    }
    if (!S_ISDIR(st.st_mode)) return Fail(ERROR_DIRECTORY);
    if (rmdir(p.c_str()) != 0) { SetErrorFromErrno(errno, p); return FALSE; }
    return TRUE;
}
BOOL CreateHardLinkW(LPCWSTR link, LPCWSTR existing, LPSECURITY_ATTRIBUTES) {
    std::string a, b;
    if (!ToPosix(link, a) || !ToPosix(existing, b)) return FALSE;
    if (::link(b.c_str(), a.c_str()) != 0) { SetErrorFromErrno(errno, errno == ENOENT ? b : a); return FALSE; }
    return TRUE;
}
BOOL CreateSymbolicLinkW(LPCWSTR link, LPCWSTR target, DWORD) {
    std::string a, b;
    if (!ToPosix(link, a)) return FALSE;
    if (!target || !*target) return Fail(ERROR_INVALID_PARAMETER);
    std::wstring t = target;
    for (auto& c : t) if (c == L'\\') c = L'/';
    if (t.size() >= 2 && t[1] == L':') { if (!ToPosix(target, b)) return FALSE; }   // a drive-lettered target maps the same way
    else b = ToUtf8(t);
    if (::symlink(b.c_str(), a.c_str()) != 0) { SetErrorFromErrno(errno, a); return FALSE; }
    return TRUE;
}
// There is no cmd.exe: every command "fails" (1), except `mklink /J "link" "folder"`, which the tests use to make a junction. A
// junction becomes a symbolic link to the folder: the nearest thing here (a reparse point that RemoveTree must not follow).
int _wsystem(const wchar_t* cmd) {
    const std::wstring c = cmd ? cmd : L"";
    const size_t at = c.find(L"mklink /J ");
    if (at == std::wstring::npos) return 1;
    std::vector<std::wstring> q;   // the quoted arguments
    for (size_t i = at; i < c.size();) {
        const size_t a = c.find(L'"', i), b = a == std::wstring::npos ? a : c.find(L'"', a + 1);
        if (b == std::wstring::npos) break;
        q.push_back(c.substr(a + 1, b - a - 1));
        i = b + 1;
    }
    if (q.size() != 2 || (GetFileAttributesW(q[1].c_str()) & FILE_ATTRIBUTE_DIRECTORY) == 0) return 1;
    return CreateSymbolicLinkW(q[0].c_str(), q[1].c_str(), SYMBOLIC_LINK_FLAG_DIRECTORY) ? 0 : 1;
}
BOOL CopyFileExW(LPCWSTR from, LPCWSTR to, void*, LPVOID, BOOL*, DWORD flags) {
    std::string a, b;
    struct stat sa, sb;
    if ((flags & COPY_FILE_COPY_SYMLINK) && ToPosix(from, a) && lstat(a.c_str(), &sa) == 0 && S_ISLNK(sa.st_mode)) {
        if (!ToPosix(to, b)) return FALSE;
        char target[4096];
        const ssize_t n = readlink(a.c_str(), target, sizeof(target) - 1);
        if (n < 0) { SetErrorFromErrno(errno, a); return FALSE; }
        target[n] = 0;
        if (lstat(b.c_str(), &sb) == 0) {
            if (flags & COPY_FILE_FAIL_IF_EXISTS) return Fail(ERROR_FILE_EXISTS);
            if (S_ISDIR(sb.st_mode) || unlink(b.c_str()) != 0) return Fail(ERROR_ACCESS_DENIED);
        }
        if (::symlink(target, b.c_str()) != 0) { SetErrorFromErrno(errno, b); return FALSE; }
        return TRUE;
    }
    return CopyFileW(from, to, (flags & COPY_FILE_FAIL_IF_EXISTS) != 0);
}
DWORD GetFileType(HANDLE hv) {
    Handle* h = LookupAny(hv);
    struct stat st;
    if (!h || (h->kind != Kind::File && h->kind != Kind::Pipe) || fstat(h->fd, &st) != 0) return FILE_TYPE_UNKNOWN;
    return S_ISREG(st.st_mode) || S_ISDIR(st.st_mode) || S_ISBLK(st.st_mode) ? FILE_TYPE_DISK : S_ISCHR(st.st_mode) ? FILE_TYPE_CHAR : FILE_TYPE_PIPE;
}
BOOL SetThreadErrorMode(DWORD, LPDWORD old) { if (old) *old = 0; return TRUE; }   // no error dialogs exist
DWORD GetLogicalDrives() { return 1u << ('Z' - 'A'); }
FILE* _wfopen(const wchar_t* path, const wchar_t* mode) {
    std::string p;
    if (!ToPosix(path, p)) { errno = ENOENT; return nullptr; }
    return std::fopen(p.c_str(), ToUtf8(mode, wcslen(mode)).c_str());
}
// stdin, stdout and stderr as handles ReadFile and WriteFile take; their descriptors stay open
HANDLE GetStdHandle(DWORD which) {
    static Handle* std3[3] = {};
    static std::once_flag once;
    std::call_once(once, [] { for (int i = 0; i < 3; ++i) { std3[i] = NewHandle(Kind::Pipe); std3[i]->fd = i; std3[i]->borrowed = true; } });
    const int n = which == STD_INPUT_HANDLE ? 0 : which == STD_OUTPUT_HANDLE ? 1 : which == STD_ERROR_HANDLE ? 2 : -1;
    if (n < 0) { SetLastError(ERROR_INVALID_HANDLE); return INVALID_HANDLE_VALUE; }
    return std3[n];
}

// FindFirstFileW: the folder's entries are read at once, matched against the wildcard and sorted as NTFS lists them.
static bool WildMatch(const wchar_t* p, const wchar_t* s) {
    if (!*p) return !*s;
    if (*p == L'*') {
        while (*p == L'*') ++p;
        if (!*p) return true;
        for (; *s; ++s) if (WildMatch(p, s)) return true;
        return WildMatch(p, s);
    }
    if (!*s) return false;
    return (*p == L'?' || FoldChar((wint_t)*p) == FoldChar((wint_t)*s)) && WildMatch(p + 1, s + 1);
}
HANDLE FindFirstFileW(LPCWSTR pattern, WIN32_FIND_DATAW* data) {
    if (!pattern || !*pattern) { SetLastError(ERROR_PATH_NOT_FOUND); return INVALID_HANDLE_VALUE; }
    std::wstring w = pattern;
    for (auto& c : w) if (c == L'\\') c = L'/';
    const size_t cut = w.find_last_of(L'/');
    const std::wstring dirW = cut == std::wstring::npos ? L"." : cut == 0 ? L"/" : w.substr(0, cut), pat = cut == std::wstring::npos ? w : w.substr(cut + 1);
    std::string dir;
    if (!ToPosix(dirW.c_str(), dir)) return INVALID_HANDLE_VALUE;
    struct stat st;
    if (stat(dir.c_str(), &st) != 0 || !S_ISDIR(st.st_mode)) { SetLastError(ERROR_PATH_NOT_FOUND); return INVALID_HANDLE_VALUE; }
    DIR* d = opendir(dir.c_str());
    if (!d) { SetErrorFromErrno(errno, dir); return INVALID_HANDLE_VALUE; }
    std::vector<WIN32_FIND_DATAW> found;
    const bool wild = pat.find_first_of(L"*?") != std::wstring::npos;
    while (dirent* ent = readdir(d)) {
        const std::wstring name = FromUtf8(ent->d_name, std::strlen(ent->d_name));
        if (!WildMatch(pat == L"*.*" ? L"*" : pat.c_str(), name.c_str())) continue;
        if (!wild && std::strcmp(ent->d_name, ".") != 0 && std::strcmp(ent->d_name, "..") != 0 && name.size() != pat.size()) continue;
        struct stat lst;
        const std::string full = (dir == "/" ? std::string() : dir) + "/" + ent->d_name;
        if (lstat(full.c_str(), &lst) != 0) continue;
        WIN32_FIND_DATAW f;
        FillFind(f, lst, full, ent->d_name);
        found.push_back(f);
    }
    closedir(d);
    std::sort(found.begin(), found.end(), [](const WIN32_FIND_DATAW& a, const WIN32_FIND_DATAW& b) {
        const bool da = !wcscmp(a.cFileName, L".") || !wcscmp(a.cFileName, L".."), db = !wcscmp(b.cFileName, L".") || !wcscmp(b.cFileName, L"..");
        if (da != db) return da;
        return _wcsicmp(a.cFileName, b.cFileName) < 0;
    });
    if (found.empty()) { SetLastError(ERROR_FILE_NOT_FOUND); return INVALID_HANDLE_VALUE; }
    Handle* h = NewHandle(Kind::Find);
    h->found = std::move(found);
    *data = h->found[0];
    h->next = 1;
    return h;
}
HANDLE FindFirstFileExW(LPCWSTR pattern, FINDEX_INFO_LEVELS, LPVOID data, FINDEX_SEARCH_OPS, LPVOID, DWORD) {
    return FindFirstFileW(pattern, static_cast<WIN32_FIND_DATAW*>(data));
}
BOOL FindNextFileW(HANDLE hv, WIN32_FIND_DATAW* data) {
    Handle* h = Lookup(hv, Kind::Find);
    if (!h) return FALSE;
    if (h->next >= h->found.size()) return Fail(ERROR_NO_MORE_FILES);
    *data = h->found[h->next++];
    return TRUE;
}
BOOL FindClose(HANDLE hv) {
    Handle* h = Lookup(hv, Kind::Find);
    if (!h) return FALSE;
    Forget(h);
    return TRUE;
}

// "C:\a\.\b\..\c" is "C:\a\c": "." and ".." fold, separators become backslashes, a relative path gets the current folder (Z:) in
// front. A drive other than Z is only spelled, never looked at. Nothing needs to exist.
DWORD GetLongPathNameW(LPCWSTR shortPath, LPWSTR out, DWORD size) {
    std::string p;
    struct stat st;
    if (!ToPosix(shortPath, p)) return 0;
    if (lstat(p.c_str(), &st) != 0) { SetErrorFromErrno(errno, p); return 0; }
    const std::wstring w = (std::wstring(shortPath).rfind(L"\\\\?\\", 0) == 0 ? L"\\\\?\\" : L"") + WinPath(p);
    if (!out || size < w.size() + 1) return (DWORD)w.size() + 1;
    std::memcpy(out, w.c_str(), (w.size() + 1) * sizeof(wchar_t));
    return (DWORD)w.size();
}
DWORD GetFullPathNameW(LPCWSTR name, DWORD size, LPWSTR out, LPWSTR* filePart) {
    if (!name || !*name) { SetLastError(ERROR_PATH_NOT_FOUND); return 0; }
    std::wstring raw = name, drive;
    if (raw.rfind(L"\\\\?\\", 0) == 0) raw.erase(0, 4);
    if (raw.size() >= 2 && raw[1] == L':' && iswalpha((wint_t)raw[0])) { drive = raw.substr(0, 2); raw.erase(0, 2); }
    const bool rooted = !raw.empty() && (raw[0] == L'/' || raw[0] == L'\\');
    if (!rooted && !drive.empty() && towupper((wint_t)drive[0]) != L'Z') { SetLastError(ERROR_PATH_NOT_FOUND); return 0; }
    if (!rooted) {
        char cwd[4096];
        if (!getcwd(cwd, sizeof(cwd))) { SetErrorFromErrno(errno, ""); return 0; }
        raw = FromUtf8(cwd, std::strlen(cwd)) + L"/" + raw;
        if (drive.empty()) drive = L"Z:";
    }
    if (drive.empty()) drive = L"Z:";
    std::vector<std::wstring> parts;
    size_t i = 0;
    while (i <= raw.size()) {
        size_t j = raw.find_first_of(L"/\\", i);
        if (j == std::wstring::npos) j = raw.size();
        const std::wstring c = raw.substr(i, j - i);
        if (c == L"..") { if (!parts.empty()) parts.pop_back(); }
        else if (!c.empty() && c != L".") parts.push_back(c);
        i = j + 1;
    }
    std::wstring w = drive;
    for (auto& c : parts) { w += L'\\'; w += c; }
    if (parts.empty() || raw.back() == L'/' || raw.back() == L'\\') w += L'\\';
    if (size < w.size() + 1) return (DWORD)w.size() + 1;
    std::memcpy(out, w.c_str(), (w.size() + 1) * sizeof(wchar_t));
    if (filePart) {
        const size_t slash = w.find_last_of(L'\\');
        *filePart = (slash == std::wstring::npos || slash + 1 >= w.size()) ? nullptr : out + slash + 1;
    }
    return (DWORD)w.size();
}
static DWORD CopyOut(const std::wstring& w, LPWSTR out, DWORD size) {
    if (!out || size < w.size() + 1) return (DWORD)w.size() + 1;
    std::memcpy(out, w.c_str(), (w.size() + 1) * sizeof(wchar_t));
    return (DWORD)w.size();
}
DWORD GetTempPathW(DWORD size, LPWSTR out) {
    const char* t = getenv("TMPDIR");
    std::string d = t && *t ? t : "/tmp";
    if (d.back() != '/') d += '/';
    return CopyOut(WinPath(d), out, size);
}
DWORD GetCurrentDirectoryW(DWORD size, LPWSTR out) {
    char cwd[4096];
    if (!getcwd(cwd, sizeof(cwd))) { SetErrorFromErrno(errno, ""); return 0; }
    return CopyOut(WinPath(cwd), out, size);
}
BOOL SetCurrentDirectoryW(LPCWSTR dir) {
    std::string p;
    if (!ToPosix(dir, p)) return FALSE;
    if (chdir(p.c_str()) != 0) { SetErrorFromErrno(errno, p); return FALSE; }
    return TRUE;
}
// There is no System32. This folder exists and holds no cmd.exe, no PING.EXE and no Windows DLL, so what the core
// builds from it fails to start or to load (cleanly).
DWORD GetSystemDirectoryW(LPWSTR out, DWORD size) { return CopyOut(L"Z:\\usr\\bin", out, size); }
DWORD GetModuleFileNameW(HMODULE m, LPWSTR out, DWORD size) {
    if (m) { SetLastError(ERROR_MOD_NOT_FOUND); return 0; }
    char b[4096];
    const ssize_t n = readlink("/proc/self/exe", b, sizeof(b) - 1);
    if (n < 0) { SetErrorFromErrno(errno, ""); return 0; }
    const std::wstring w = WinPath(std::string(b, (size_t)n));
    if (size == 0) return 0;
    const size_t k = std::min<size_t>(w.size(), size - 1);
    std::memcpy(out, w.c_str(), k * sizeof(wchar_t));
    out[k] = 0;
    if (k < w.size()) { SetLastError(ERROR_INSUFFICIENT_BUFFER); return size; }
    return (DWORD)k;
}
BOOL GetDiskFreeSpaceExW(LPCWSTR dir, ULARGE_INTEGER* available, ULARGE_INTEGER* total, ULARGE_INTEGER* freeBytes) {
    std::string p;
    struct statvfs vs;
    if (!ToPosix(dir, p)) return FALSE;
    if (statvfs(p.c_str(), &vs) != 0) { SetErrorFromErrno(errno, p); return FALSE; }
    if (available) available->QuadPart = (ULONGLONG)vs.f_bavail * vs.f_frsize;
    if (total) total->QuadPart = (ULONGLONG)vs.f_blocks * vs.f_frsize;
    if (freeBytes) freeBytes->QuadPart = (ULONGLONG)vs.f_bfree * vs.f_frsize;
    return TRUE;
}

// memory-mapped files
static std::mutex g_viewLock;
static auto& g_views = *new std::map<const void*, size_t>;
HANDLE CreateFileMappingW(HANDLE file, LPSECURITY_ATTRIBUTES, DWORD protect, DWORD sizeHigh, DWORD sizeLow, LPCWSTR) {
    Handle* f = Lookup(file, Kind::File);
    struct stat st;
    if (!f) return nullptr;
    if (fstat(f->fd, &st) != 0) { SetErrorFromErrno(errno, f->path); return nullptr; }
    const uint64_t asked = ((uint64_t)sizeHigh << 32) | sizeLow, size = asked ? asked : (uint64_t)st.st_size;
    if (!size) { SetLastError(ERROR_FILE_INVALID); return nullptr; }
    if ((!f->read && !f->write) || (protect != PAGE_READONLY && !f->write)) { SetLastError(ERROR_ACCESS_DENIED); return nullptr; }
    Handle* m = NewHandle(Kind::Map);
    m->fd = fcntl(f->fd, F_DUPFD_CLOEXEC, 0);
    m->mapSize = size;
    m->mapProtect = protect;
    if (m->fd < 0) { SetErrorFromErrno(errno, f->path); Forget(m); return nullptr; }
    return m;
}
LPVOID MapViewOfFile(HANDLE map, DWORD access, DWORD offsetHigh, DWORD offsetLow, SIZE_T bytes) {
    Handle* m = Lookup(map, Kind::Map);
    if (!m) return nullptr;
    const uint64_t off = ((uint64_t)offsetHigh << 32) | offsetLow;
    if (off >= m->mapSize) { SetLastError(ERROR_INVALID_PARAMETER); return nullptr; }
    const size_t len = bytes ? bytes : (size_t)(m->mapSize - off);
    int prot = PROT_READ, mode = MAP_SHARED;
    if (access & FILE_MAP_WRITE) { if (m->mapProtect != PAGE_READWRITE) { SetLastError(ERROR_ACCESS_DENIED); return nullptr; } prot |= PROT_WRITE; }
    if (access & FILE_MAP_COPY) { mode = MAP_PRIVATE; prot |= PROT_WRITE; }
    void* v = mmap(nullptr, len, prot, mode, m->fd, (off_t)off);
    if (v == MAP_FAILED) { SetErrorFromErrno(errno, ""); return nullptr; }
    std::lock_guard<std::mutex> g(g_viewLock);
    g_views[v] = len;
    return v;
}
BOOL UnmapViewOfFile(LPCVOID base) {
    size_t len = 0;
    {
        std::lock_guard<std::mutex> g(g_viewLock);
        auto it = g_views.find(base);
        if (it == g_views.end()) return Fail(ERROR_INVALID_PARAMETER);
        len = it->second;
        g_views.erase(it);
    }
    munmap(const_cast<void*>(base), len);
    return TRUE;
}

// ---- strings -----------------------------------------------------------------------------------------------------------
int MultiByteToWideChar(UINT, DWORD flags, const char* s, int n, wchar_t* out, int outN) {
    if (!s || n == 0) { SetLastError(ERROR_INVALID_PARAMETER); return 0; }
    const size_t len = n < 0 ? std::strlen(s) + 1 : (size_t)n;
    if (flags & MB_ERR_INVALID_CHARS) {
        const unsigned char* u = reinterpret_cast<const unsigned char*>(s);
        for (size_t i = 0; i < len;) { uint32_t cp; const size_t k = DecodeOne(u, len, i, cp); if (!k) { SetLastError(ERROR_NO_UNICODE_TRANSLATION); return 0; } i += k; }
    }
    std::wstring w = FromUtf8(s, len);
    for (auto& c : w) if ((uint32_t)c >= 0xDC80 && (uint32_t)c <= 0xDCFF) c = 0xFFFD;
    if (outN == 0) return (int)w.size();
    if ((size_t)outN < w.size()) { SetLastError(ERROR_INSUFFICIENT_BUFFER); return 0; }
    std::memcpy(out, w.data(), w.size() * sizeof(wchar_t));
    return (int)w.size();
}
int WideCharToMultiByte(UINT, DWORD, const wchar_t* s, int n, char* out, int outN, const char*, BOOL* used) {
    if (used) *used = FALSE;
    if (!s || n == 0) { SetLastError(ERROR_INVALID_PARAMETER); return 0; }
    const size_t len = n < 0 ? wcslen(s) + 1 : (size_t)n;
    const std::string u = ToUtf8(s, len);
    if (outN == 0) return (int)u.size();
    if ((size_t)outN < u.size()) { SetLastError(ERROR_INSUFFICIENT_BUFFER); return 0; }
    std::memcpy(out, u.data(), u.size());
    return (int)u.size();
}
int _wcsicmp(const wchar_t* a, const wchar_t* b) {
    for (;; ++a, ++b) {
        const wint_t x = FoldChar((wint_t)*a), y = FoldChar((wint_t)*b);
        if (x != y) return x < y ? -1 : 1;
        if (!x) return 0;
    }
}
int _wcsnicmp(const wchar_t* a, const wchar_t* b, size_t n) {
    for (; n; --n, ++a, ++b) {
        const wint_t x = FoldChar((wint_t)*a), y = FoldChar((wint_t)*b);
        if (x != y) return x < y ? -1 : 1;
        if (!x) return 0;
    }
    return 0;
}
int CompareStringOrdinal(const wchar_t* a, int an, const wchar_t* b, int bn, BOOL ignoreCase) {
    const size_t x = an < 0 ? wcslen(a) : (size_t)an, y = bn < 0 ? wcslen(b) : (size_t)bn;
    for (size_t i = 0; i < x && i < y; ++i) {
        const wint_t p = ignoreCase ? FoldChar((wint_t)a[i]) : (wint_t)a[i], q = ignoreCase ? FoldChar((wint_t)b[i]) : (wint_t)b[i];
        if (p != q) return p < q ? CSTR_LESS_THAN : CSTR_GREATER_THAN;
    }
    return x == y ? CSTR_EQUAL : x < y ? CSTR_LESS_THAN : CSTR_GREATER_THAN;
}
// Positions the conversions of fmt take pointers for; the size after %s, %c or %[ is not one.
int sscanf_s(const char* s, const char* fmt, ...) {
    void* ptr[8] = {};
    int n = 0;
    va_list ap;
    va_start(ap, fmt);
    for (const char* f = fmt; *f; ++f) {
        if (*f != '%') continue;
        ++f;
        if (*f == '%') continue;
        const bool skip = *f == '*';
        while (*f && std::strchr("*0123456789hlLzjtI", *f)) ++f;
        if (!*f) break;
        if (skip) continue;
        if (n < 8) ptr[n++] = va_arg(ap, void*);
        if (*f == 's' || *f == 'c' || *f == '[') (void)va_arg(ap, unsigned);
    }
    va_end(ap);
    return sscanf(s, fmt, ptr[0], ptr[1], ptr[2], ptr[3], ptr[4], ptr[5], ptr[6], ptr[7]);
}
int _stricmp(const char* a, const char* b) { return strcasecmp(a, b); }
int _wtoi(const wchar_t* s) { return (int)wcstol(s, nullptr, 10); }
long long _wtoi64(const wchar_t* s) { return wcstoll(s, nullptr, 10); }

DWORD FormatMessageW(DWORD flags, LPCVOID, DWORD id, DWORD, LPWSTR buf, DWORD size, void*) {
    static const struct { DWORD id; const wchar_t* text; } kText[] = {
        {ERROR_FILE_NOT_FOUND, L"The system cannot find the file specified."}, {ERROR_PATH_NOT_FOUND, L"The system cannot find the path specified."},
        {ERROR_ACCESS_DENIED, L"Access is denied."}, {ERROR_INVALID_HANDLE, L"The handle is invalid."}, {ERROR_NOT_ENOUGH_MEMORY, L"Not enough memory resources are available to process this command."},
        {ERROR_SHARING_VIOLATION, L"The process cannot access the file because it is being used by another process."},
        {ERROR_LOCK_VIOLATION, L"The process cannot access the file because another process has locked a portion of the file."},
        {ERROR_NOT_SUPPORTED, L"The request is not supported."}, {ERROR_FILE_EXISTS, L"The file exists."}, {ERROR_INVALID_PARAMETER, L"The parameter is incorrect."},
        {ERROR_BROKEN_PIPE, L"The pipe has been ended."}, {ERROR_DISK_FULL, L"There is not enough space on the disk."},
        {ERROR_CALL_NOT_IMPLEMENTED, L"This function is not supported on this system."}, {ERROR_DIR_NOT_EMPTY, L"The directory is not empty."},
        {ERROR_ALREADY_EXISTS, L"Cannot create a file when that file already exists."}, {ERROR_NOT_SAME_DEVICE, L"The system cannot move the file to a different disk drive."},
        {ERROR_MOD_NOT_FOUND, L"The specified module could not be found."}, {ERROR_PROC_NOT_FOUND, L"The specified procedure could not be found."},
        {ERROR_DIRECTORY, L"The directory name is invalid."}, {ERROR_INVALID_NAME, L"The filename, directory name, or volume label syntax is incorrect."}, {ERROR_CANCELLED, L"The operation was canceled by the user."},
    };
    const wchar_t* text = nullptr;
    for (auto& t : kText) if (t.id == id) text = t.text;
    if (!text || !(flags & FORMAT_MESSAGE_FROM_SYSTEM)) { SetLastError(ERROR_RESOURCE_LANG_NOT_FOUND); return 0; }
    const size_t n = wcslen(text);
    if (flags & FORMAT_MESSAGE_ALLOCATE_BUFFER) {
        wchar_t* m = static_cast<wchar_t*>(std::malloc((n + 3) * sizeof(wchar_t)));
        if (!m) return 0;
        std::memcpy(m, text, n * sizeof(wchar_t));
        m[n] = L'\r'; m[n + 1] = L'\n'; m[n + 2] = 0;
        *reinterpret_cast<wchar_t**>(buf) = m;
        return (DWORD)n + 2;
    }
    if (size < n + 3) { SetLastError(ERROR_INSUFFICIENT_BUFFER); return 0; }
    std::memcpy(buf, text, n * sizeof(wchar_t));
    buf[n] = L'\r'; buf[n + 1] = L'\n'; buf[n + 2] = 0;
    return (DWORD)n + 2;
}
HLOCAL LocalFree(HLOCAL p) { std::free(p); return nullptr; }
BOOL ConvertStringSecurityDescriptorToSecurityDescriptorW(LPCWSTR, DWORD, PSECURITY_DESCRIPTOR* sd, ULONG* size) {
    if (sd) *sd = nullptr;
    if (size) *size = 0;
    return Fail(ERROR_NOT_SUPPORTED);
}
// One heap here: the C one. What Windows code frees with HeapFree(GetProcessHeap(), ...) came from malloc.
HANDLE GetProcessHeap() { static char heap; return &heap; }
BOOL HeapFree(HANDLE, DWORD, LPVOID p) { std::free(p); return TRUE; }
// Like a kill, not like exit(): no static destructors run while other threads still use the layer's tables.
void ExitProcess(UINT code) { std::fflush(nullptr); _exit((int)code); }

// ---- environment -------------------------------------------------------------------------------------------------------
// Windows names ignore case; a name the environment lacks but Windows always has is filled in from the user's folders.
static bool EnvLookup(const std::string& name, std::string& value) {
    if (const char* v = getenv(name.c_str())) { value = v; return true; }
    for (char** e = environ; e && *e; ++e) {
        const char* eq = std::strchr(*e, '=');
        if (eq && (size_t)(eq - *e) == name.size() && strncasecmp(*e, name.c_str(), name.size()) == 0) { value = eq + 1; return true; }
    }
    if (strcasecmp(name.c_str(), "LOCALAPPDATA") == 0) { value = ToUtf8(WinPath(XdgDataHome())); return true; }
    if (strcasecmp(name.c_str(), "APPDATA") == 0) { value = ToUtf8(WinPath(XdgConfigHome())); return true; }
    if (strcasecmp(name.c_str(), "USERPROFILE") == 0) { value = ToUtf8(WinPath(HomeDir())); return true; }
    if (strcasecmp(name.c_str(), "TEMP") == 0 || strcasecmp(name.c_str(), "TMP") == 0) {
        const char* t = getenv("TMPDIR");
        value = ToUtf8(WinPath(t && *t ? t : "/tmp"));
        return true;
    }
    return false;
}
DWORD GetEnvironmentVariableW(LPCWSTR name, LPWSTR out, DWORD size) {
    std::string v;
    if (!name || !*name || !EnvLookup(ToUtf8(name, wcslen(name)), v)) { SetLastError(ERROR_ENVVAR_NOT_FOUND); return 0; }
    return CopyOut(FromUtf8(v), out, size);
}
BOOL SetEnvironmentVariableW(LPCWSTR name, LPCWSTR value) {
    if (!name || !*name || wcschr(name, L'=')) return Fail(ERROR_INVALID_PARAMETER);
    const std::string n = ToUtf8(name, wcslen(name));
    for (char** e = environ; e && *e;) {   // a variant of the name in another case is the same variable on Windows
        const char* eq = std::strchr(*e, '=');
        if (eq && (size_t)(eq - *e) == n.size() && strncasecmp(*e, n.c_str(), n.size()) == 0 && strncmp(*e, n.c_str(), n.size()) != 0) {
            unsetenv(std::string(*e, (size_t)(eq - *e)).c_str());
            e = environ;
        } else ++e;
    }
    if (!value) unsetenv(n.c_str());
    else setenv(n.c_str(), ToUtf8(value, wcslen(value)).c_str(), 1);
    return TRUE;
}
LPWCH GetEnvironmentStringsW() {
    std::wstring block;
    for (char** e = environ; e && *e; ++e) { block += FromUtf8(*e, std::strlen(*e)); block += L'\0'; }
    block += L'\0';
    wchar_t* m = static_cast<wchar_t*>(std::malloc(block.size() * sizeof(wchar_t)));
    if (m) std::memcpy(m, block.data(), block.size() * sizeof(wchar_t));
    return m;
}
BOOL FreeEnvironmentStringsW(LPWCH block) { std::free(block); return TRUE; }

// ---- time --------------------------------------------------------------------------------------------------------------
ULONGLONG GetTickCount64() {
    struct timespec t;
    clock_gettime(CLOCK_BOOTTIME, &t);
    return (ULONGLONG)t.tv_sec * 1000 + (ULONGLONG)t.tv_nsec / 1000000;
}
static void ToSystemTime(const struct tm& tm, long ms, SYSTEMTIME* t) {
    t->wYear = (WORD)(tm.tm_year + 1900); t->wMonth = (WORD)(tm.tm_mon + 1); t->wDayOfWeek = (WORD)tm.tm_wday; t->wDay = (WORD)tm.tm_mday;
    t->wHour = (WORD)tm.tm_hour; t->wMinute = (WORD)tm.tm_min; t->wSecond = (WORD)tm.tm_sec; t->wMilliseconds = (WORD)ms;
}
void GetLocalTime(SYSTEMTIME* t) {
    struct timespec ts; struct tm tm;
    clock_gettime(CLOCK_REALTIME, &ts);
    localtime_r(&ts.tv_sec, &tm);
    ToSystemTime(tm, ts.tv_nsec / 1000000, t);
}
BOOL FileTimeToSystemTime(const FILETIME* ft, SYSTEMTIME* st) {
    const uint64_t v = ((uint64_t)ft->dwHighDateTime << 32) | ft->dwLowDateTime;
    const struct timespec ts = TimespecOf(*ft);
    struct tm tm;
    if (!gmtime_r(&ts.tv_sec, &tm)) return Fail(ERROR_INVALID_PARAMETER);
    ToSystemTime(tm, (long)((v / 10000) % 1000), st);
    return TRUE;
}
BOOL SystemTimeToFileTime(const SYSTEMTIME* st, FILETIME* ft) {
    struct tm tm{};
    tm.tm_year = st->wYear - 1900; tm.tm_mon = st->wMonth - 1; tm.tm_mday = st->wDay;
    tm.tm_hour = st->wHour; tm.tm_min = st->wMinute; tm.tm_sec = st->wSecond;
    if (st->wMonth < 1 || st->wMonth > 12 || st->wDay < 1 || st->wDay > 31) return Fail(ERROR_INVALID_PARAMETER);
    struct timespec ts{timegm(&tm), (long)st->wMilliseconds * 1000000};
    *ft = FileTimeOf(ts);
    return TRUE;
}
BOOL FileTimeToLocalFileTime(const FILETIME* ft, FILETIME* local) {
    const struct timespec ts = TimespecOf(*ft);
    struct tm tm;
    if (!localtime_r(&ts.tv_sec, &tm)) return Fail(ERROR_INVALID_PARAMETER);
    const uint64_t v = (((uint64_t)ft->dwHighDateTime << 32) | ft->dwLowDateTime) + (uint64_t)((int64_t)tm.tm_gmtoff * 10000000);
    local->dwLowDateTime = (DWORD)(v & 0xFFFFFFFFu);
    local->dwHighDateTime = (DWORD)(v >> 32);
    return TRUE;
}
BOOL LocalFileTimeToFileTime(const FILETIME* local, FILETIME* ft) {
    const struct timespec ts = TimespecOf(*local);
    struct tm tm;
    gmtime_r(&ts.tv_sec, &tm);
    tm.tm_isdst = -1;
    const time_t asLocal = mktime(&tm);   // the local clock reading `local` shows, as an instant
    const int64_t shift = (int64_t)asLocal - (int64_t)ts.tv_sec;
    const uint64_t v = (((uint64_t)local->dwHighDateTime << 32) | local->dwLowDateTime) + (uint64_t)(shift * 10000000);
    ft->dwLowDateTime = (DWORD)(v & 0xFFFFFFFFu);
    ft->dwHighDateTime = (DWORD)(v >> 32);
    return TRUE;
}
LONG CompareFileTime(const FILETIME* a, const FILETIME* b) {
    const uint64_t x = ((uint64_t)a->dwHighDateTime << 32) | a->dwLowDateTime, y = ((uint64_t)b->dwHighDateTime << 32) | b->dwLowDateTime;
    return x < y ? -1 : x > y ? 1 : 0;
}
void Sleep(DWORD ms) {
    struct timespec t{(time_t)(ms / 1000), (long)(ms % 1000) * 1000000};
    while (nanosleep(&t, &t) != 0 && errno == EINTR) {}
}
DWORD GetCurrentProcessId() { return (DWORD)getpid(); }
BOOL QueryPerformanceCounter(LARGE_INTEGER* v) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    v->QuadPart = (LONGLONG)t.tv_sec * 1000000000LL + t.tv_nsec;
    return TRUE;
}
BOOL QueryPerformanceFrequency(LARGE_INTEGER* v) { v->QuadPart = 1000000000LL; return TRUE; }

// ---- user folders ------------------------------------------------------------------------------------------------------
namespace wgposix {
std::string HomeDir() {
    if (const char* h = getenv("HOME")) if (*h) return h;
    if (passwd* pw = getpwuid(getuid())) return pw->pw_dir;
    return "/";
}
std::string XdgDataHome() {
    const char* x = getenv("XDG_DATA_HOME");
    return x && *x ? x : HomeDir() + "/.local/share";
}
std::string XdgConfigHome() {
    const char* x = getenv("XDG_CONFIG_HOME");
    return x && *x ? x : HomeDir() + "/.config";
}
std::string SteamRoot() {
    for (const std::string& c : {HomeDir() + "/.local/share/Steam", HomeDir() + "/.steam/steam", HomeDir() + "/.steam/root"}) {
        struct stat st;
        if (stat(c.c_str(), &st) == 0 && S_ISDIR(st.st_mode)) return c;
    }
    return "";
}
}  // namespace wgposix
// user-dirs.dirs holds lines like XDG_DESKTOP_DIR="$HOME/Desktop"
static std::string XdgUserDir(const char* key, const char* fallback) {
    const std::string home = HomeDir();
    std::string found;
    if (FILE* f = std::fopen((XdgConfigHome() + "/user-dirs.dirs").c_str(), "r")) {
        char line[2048];
        const std::string want = std::string(key) + "=";
        while (std::fgets(line, sizeof(line), f)) {
            std::string l = line;
            if (l.compare(0, want.size(), want) != 0) continue;
            l = l.substr(want.size());
            while (!l.empty() && (l.back() == '\n' || l.back() == '\r')) l.pop_back();
            if (l.size() >= 2 && l.front() == '"' && l.back() == '"') l = l.substr(1, l.size() - 2);
            if (l.compare(0, 5, "$HOME") == 0) l = home + l.substr(5);
            found = l;
        }
        std::fclose(f);
    }
    return found.empty() ? home + "/" + fallback : found;
}

#define WG_GUID(name, a, b, c, d0, d1, d2, d3, d4, d5, d6, d7) const GUID name = {a, b, c, {d0, d1, d2, d3, d4, d5, d6, d7}}
WG_GUID(FOLDERID_LocalAppData, 0xF1B32785, 0x6FBA, 0x4FCF, 0x9D, 0x55, 0x7B, 0x8E, 0x7F, 0x15, 0x70, 0x91);
WG_GUID(FOLDERID_RoamingAppData, 0x3EB685DB, 0x65F9, 0x4CF6, 0xA0, 0x3A, 0xE3, 0xEF, 0x65, 0x72, 0x9F, 0x3D);
WG_GUID(FOLDERID_Desktop, 0xB4BFCC3A, 0xDB2C, 0x424C, 0xB0, 0x29, 0x7F, 0xE9, 0x9A, 0x87, 0xC6, 0x41);
WG_GUID(FOLDERID_Documents, 0xFDD39AD0, 0x238F, 0x46AF, 0xAD, 0xB4, 0x6C, 0x85, 0x48, 0x03, 0x69, 0xC7);
WG_GUID(FOLDERID_Programs, 0xA77F5D77, 0x2E2B, 0x44C3, 0xA6, 0xA2, 0xAB, 0xA6, 0x01, 0x05, 0x4A, 0x51);
WG_GUID(FOLDERID_ProgramData, 0x62AB5D82, 0xFDC1, 0x4DC3, 0xA9, 0xDD, 0x07, 0x0D, 0x1D, 0x49, 0x5D, 0x97);
WG_GUID(FOLDERID_Profile, 0x5E6C858F, 0x0E22, 0x4760, 0x9A, 0xFE, 0xEA, 0x33, 0x17, 0xB6, 0x71, 0x73);
WG_GUID(FOLDERID_StartMenu, 0x625B53C3, 0xAB48, 0x4EC1, 0xBA, 0x1F, 0xA1, 0xEF, 0x41, 0x46, 0xFC, 0x19);
WG_GUID(FOLDERID_ProgramFiles, 0x905E63B6, 0xC1BF, 0x494E, 0xB2, 0x9C, 0x65, 0xB7, 0x32, 0xD3, 0xD2, 0x1A);
WG_GUID(FOLDERID_ProgramFilesX86, 0x7C5A40EF, 0xA0FB, 0x4BFC, 0x87, 0x4A, 0xC0, 0xF2, 0xE0, 0xB9, 0xFA, 0x8E);
WG_GUID(FOLDERID_Downloads, 0x374DE290, 0x123F, 0x4565, 0x91, 0x64, 0x39, 0xC4, 0x92, 0x5E, 0x46, 0x7B);
WG_GUID(CLSID_ShellLink, 0x00021401, 0x0000, 0x0000, 0xC0, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x46);
WG_GUID(IID_IShellLinkW, 0x000214F9, 0x0000, 0x0000, 0xC0, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x46);
WG_GUID(IID_IPersistFile, 0x0000010B, 0x0000, 0x0000, 0xC0, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x46);
static const GUID kDxgiFactory1 = {0x770aae78, 0xf26f, 0x4dba, {0xa8, 0x29, 0x25, 0x3c, 0x83, 0xd1, 0xb3, 0x87}};
static const GUID kDxgiAdapter1 = {0x29038f61, 0x3839, 0x4626, {0x91, 0xfd, 0x08, 0x68, 0x79, 0x01, 0x1a, 0x05}};
static const GUID kDxgiDevice = {0x54ec77fa, 0x1377, 0x44e6, {0x8c, 0x32, 0x88, 0xfd, 0x5f, 0x44, 0xc8, 0x4c}};
template <> const GUID& wg_posix_uuidof<IDXGIFactory1>() { return kDxgiFactory1; }
template <> const GUID& wg_posix_uuidof<IDXGIAdapter1>() { return kDxgiAdapter1; }
template <> const GUID& wg_posix_uuidof<IDXGIDevice>() { return kDxgiDevice; }

HRESULT SHGetKnownFolderPath(REFKNOWNFOLDERID id, DWORD, HANDLE, PWSTR* out) {
    *out = nullptr;
    std::string p;
    auto is = [&](const GUID& g) { return std::memcmp(&id, &g, sizeof(GUID)) == 0; };
    if (is(FOLDERID_LocalAppData)) p = XdgDataHome();
    else if (is(FOLDERID_RoamingAppData)) p = XdgConfigHome();
    else if (is(FOLDERID_Desktop)) p = XdgUserDir("XDG_DESKTOP_DIR", "Desktop");
    else if (is(FOLDERID_Documents)) p = XdgUserDir("XDG_DOCUMENTS_DIR", "Documents");
    else if (is(FOLDERID_Downloads)) p = XdgUserDir("XDG_DOWNLOAD_DIR", "Downloads");
    else if (is(FOLDERID_Programs) || is(FOLDERID_StartMenu)) p = XdgDataHome() + "/applications";
    else if (is(FOLDERID_Profile)) p = HomeDir();
    else if (is(FOLDERID_ProgramData)) p = "/var/lib";
    else return HRESULT_ERROR_FILE_NOT_FOUND;   // Program Files and the like have no counterpart
    const std::wstring w = WinPath(p);
    wchar_t* m = static_cast<wchar_t*>(CoTaskMemAlloc((w.size() + 1) * sizeof(wchar_t)));
    if (!m) return E_OUTOFMEMORY;
    std::memcpy(m, w.c_str(), (w.size() + 1) * sizeof(wchar_t));
    *out = m;
    return S_OK;
}
void CoTaskMemFree(LPVOID p) { std::free(p); }
LPVOID CoTaskMemAlloc(SIZE_T n) { return std::malloc(n); }
HRESULT CoInitializeEx(LPVOID, DWORD) { return S_OK; }
void CoUninitialize() {}
HRESULT CoCreateInstance(REFCLSID, IUnknown*, DWORD, REFIID, LPVOID* out) {
    if (out) *out = nullptr;
    return REGDB_E_CLASSNOTREG;   // no shell link object, so no shortcut
}

// ---- modules: no Windows DLL loads, so every GetProcAddress user takes its "not available" branch ------------------------
HMODULE LoadLibraryExW(LPCWSTR, HANDLE, DWORD) { SetLastError(ERROR_MOD_NOT_FOUND); return nullptr; }
// ntdll's RtlGetVersion is the one export that answers: Proton prefixes report Windows 10, and the launcher's Windows check
// would refuse every Linux install on "unknown" otherwise. A stand-in module handle names it.
static char g_ntdll;
static LONG WINAPI RtlGetVersionStandIn(OSVERSIONINFOW* v) {
    if (!v || v->dwOSVersionInfoSize < sizeof(DWORD) * 5) return -1;
    v->dwMajorVersion = 10; v->dwMinorVersion = 0; v->dwBuildNumber = 19045; v->dwPlatformId = 2;
    return 0;
}
HMODULE GetModuleHandleW(LPCWSTR name) {
    if (name && _wcsicmp(name, L"ntdll.dll") == 0) return &g_ntdll;
    SetLastError(ERROR_MOD_NOT_FOUND);
    return nullptr;
}
FARPROC GetProcAddress(HMODULE m, LPCSTR name) {
    if (m == &g_ntdll && name && std::strcmp(name, "RtlGetVersion") == 0) return reinterpret_cast<FARPROC>(&RtlGetVersionStandIn);
    SetLastError(ERROR_PROC_NOT_FOUND);
    return nullptr;
}

// wg::FsPath (wg_core/util.h): the converter takes host paths, the core holds Windows-style ones
namespace wg {
std::filesystem::path FsPath(const std::wstring& p) {
    std::string host;
    return ToPosix(p.c_str(), host) ? std::filesystem::path(host) : std::filesystem::path(ToUtf8(p));
}
}  // namespace wg

// ---- resources: the table tools/make_payload.py writes into resources.S (RCDATA only; a build without it embeds none) ---------
struct WgResource { unsigned id, type; const unsigned char* data; unsigned long long size; };
extern "C" const WgResource wg_resource_table[] __attribute__((weak));  // weak: the unit-test programs link no payload

HRSRC FindResourceW(HMODULE, LPCWSTR name, LPCWSTR type) {
    // MAKEINTRESOURCE values are small integers; anything above 16 bits is a name, which no resource here has
    const uintptr_t id = reinterpret_cast<uintptr_t>(name), ty = reinterpret_cast<uintptr_t>(type);
    if (wg_resource_table && id <= 0xFFFF && ty <= 0xFFFF)
        for (const WgResource* r = wg_resource_table; r->id; ++r)
            if (r->id == id && r->type == ty) return const_cast<WgResource*>(r);
    SetLastError(ERROR_RESOURCE_DATA_NOT_FOUND);
    return nullptr;
}
HGLOBAL LoadResource(HMODULE, HRSRC r) {
    if (!r) { SetLastError(ERROR_INVALID_HANDLE); return nullptr; }
    return const_cast<unsigned char*>(static_cast<const WgResource*>(r)->data);
}
DWORD SizeofResource(HMODULE, HRSRC r) {
    if (!r) { SetLastError(ERROR_INVALID_HANDLE); return 0; }
    return static_cast<DWORD>(static_cast<const WgResource*>(r)->size);
}
LPVOID LockResource(HGLOBAL h) { return h; }

// ---- WinHTTP -----------------------------------------------------------------------------------------------------------
HINTERNET WinHttpOpen(LPCWSTR, DWORD, LPCWSTR, LPCWSTR, DWORD) { SetLastError(ERROR_NOT_SUPPORTED); return nullptr; }
BOOL WinHttpSetTimeouts(HINTERNET, int, int, int, int) { return Fail(ERROR_INVALID_HANDLE); }
BOOL WinHttpSetOption(HINTERNET, DWORD, LPVOID, DWORD) { return Fail(ERROR_INVALID_HANDLE); }
HINTERNET WinHttpConnect(HINTERNET, LPCWSTR, INTERNET_PORT, DWORD) { SetLastError(ERROR_INVALID_HANDLE); return nullptr; }
HINTERNET WinHttpOpenRequest(HINTERNET, LPCWSTR, LPCWSTR, LPCWSTR, LPCWSTR, LPCWSTR*, DWORD) { SetLastError(ERROR_INVALID_HANDLE); return nullptr; }
BOOL WinHttpSendRequest(HINTERNET, LPCWSTR, DWORD, LPVOID, DWORD, DWORD, DWORD_PTR) { return Fail(ERROR_INVALID_HANDLE); }
BOOL WinHttpReceiveResponse(HINTERNET, LPVOID) { return Fail(ERROR_INVALID_HANDLE); }
BOOL WinHttpQueryHeaders(HINTERNET, DWORD, LPCWSTR, LPVOID, LPDWORD, LPDWORD) { return Fail(ERROR_INVALID_HANDLE); }
BOOL WinHttpReadData(HINTERNET, LPVOID, DWORD, LPDWORD) { return Fail(ERROR_INVALID_HANDLE); }
BOOL WinHttpCloseHandle(HINTERNET) { return Fail(ERROR_INVALID_HANDLE); }
// Parsing needs no network: the core validates an address with it before it opens a session.
BOOL WinHttpCrackUrl(LPCWSTR url, DWORD length, DWORD, LPURL_COMPONENTS uc) {
    if (!url || !uc || uc->dwStructSize != sizeof(URL_COMPONENTS)) return Fail(ERROR_INVALID_PARAMETER);
    const std::wstring u = length ? std::wstring(url, length) : std::wstring(url);
    const size_t sep = u.find(L"://");
    if (sep == std::wstring::npos) return Fail(ERROR_WINHTTP_UNRECOGNIZED_SCHEME);
    const std::wstring scheme = u.substr(0, sep);
    if (_wcsicmp(scheme.c_str(), L"http") == 0) uc->nScheme = INTERNET_SCHEME_HTTP;
    else if (_wcsicmp(scheme.c_str(), L"https") == 0) uc->nScheme = INTERNET_SCHEME_HTTPS;
    else return Fail(ERROR_WINHTTP_UNRECOGNIZED_SCHEME);
    const size_t hostFrom = sep + 3, pathAt = u.find_first_of(L"/?#", hostFrom);
    std::wstring auth = u.substr(hostFrom, pathAt == std::wstring::npos ? std::wstring::npos : pathAt - hostFrom);
    const size_t at = auth.rfind(L'@');
    std::wstring user, pass;
    if (at != std::wstring::npos) {
        const std::wstring cred = auth.substr(0, at);
        const size_t colon = cred.find(L':');
        user = cred.substr(0, colon);
        if (colon != std::wstring::npos) pass = cred.substr(colon + 1);
        auth.erase(0, at + 1);
    }
    std::wstring host = auth;
    unsigned port = uc->nScheme == INTERNET_SCHEME_HTTPS ? 443 : 80;
    const size_t colon = auth.rfind(L':');
    if (colon != std::wstring::npos && auth.find(L']', colon) == std::wstring::npos) {
        host = auth.substr(0, colon);
        port = (unsigned)wcstoul(auth.c_str() + colon + 1, nullptr, 10);
    }
    if (host.empty() || port > 65535) return Fail(ERROR_WINHTTP_INVALID_URL);
    uc->nPort = (INTERNET_PORT)port;
    const std::wstring rest = pathAt == std::wstring::npos ? L"" : u.substr(pathAt);
    const size_t extraAt = rest.find_first_of(L"?#");
    const std::wstring path = rest.substr(0, extraAt), extra = extraAt == std::wstring::npos ? L"" : rest.substr(extraAt);
    // a requested component goes into the caller's buffer, or (no buffer, a length) is pointed at inside the address
    auto give = [&](LPWSTR& ptr, DWORD& len, const std::wstring& v, size_t at) -> bool {
        if (!ptr && !len) return true;
        if (ptr) {
            if (len < v.size() + 1) { len = (DWORD)v.size() + 1; return false; }
            std::memcpy(ptr, v.c_str(), (v.size() + 1) * sizeof(wchar_t));
        } else {
            ptr = const_cast<LPWSTR>(url) + at;
        }
        len = (DWORD)v.size();
        return true;
    };
    bool ok = give(uc->lpszScheme, uc->dwSchemeLength, scheme, 0) && give(uc->lpszHostName, uc->dwHostNameLength, host, hostFrom + (at == std::wstring::npos ? 0 : u.find(L'@', hostFrom) + 1 - hostFrom)) &&
              give(uc->lpszUserName, uc->dwUserNameLength, user, hostFrom) && give(uc->lpszPassword, uc->dwPasswordLength, pass, hostFrom) &&
              give(uc->lpszUrlPath, uc->dwUrlPathLength, path, pathAt == std::wstring::npos ? u.size() : pathAt) &&
              give(uc->lpszExtraInfo, uc->dwExtraInfoLength, extra, pathAt == std::wstring::npos ? u.size() : pathAt + path.size());
    if (!ok) return Fail(ERROR_INSUFFICIENT_BUFFER);
    return TRUE;
}
