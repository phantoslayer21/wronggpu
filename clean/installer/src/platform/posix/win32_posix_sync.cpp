// SPDX-License-Identifier: MIT
// Linux: named mutexes and unnamed events (the design is in win32_posix.h, "synchronization").
//
// Why open file description locks: they are dropped by the kernel when the last descriptor of the open file goes, a
// process's end included (a crashed owner never keeps the lock), they lock byte ranges (one file carries "exists" and
// "owned"), and turning the creator's write lock into a read lock is atomic (flock's conversion is not). A POSIX named
// semaphore has a timed wait but stays taken when its owner dies, so a crash would leave the game's state locked.
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <map>
#include <thread>
#include <utility>

#include "win32_posix_internal.h"

using namespace wgposix;

namespace {
constexpr off_t kExistsByte = 0, kOwnerByte = 1;
// The file's first byte: kOwned from the wait that takes the mutex to the ReleaseMutex that gives it back, so whoever takes
// it after an owner that ended without releasing it sees that (WAIT_ABANDONED). Locks are advisory: the locked bytes and the
// content are independent.
constexpr char kOwned = 'O', kFree = '-';

// 0, or the errno (EAGAIN: someone else holds a conflicting lock)
int SetLock(int fd, short type, off_t byte, bool wait) {
    struct flock l{};
    l.l_type = type;
    l.l_whence = SEEK_SET;
    l.l_start = byte;
    l.l_len = 1;
    int r;
    do r = fcntl(fd, wait ? F_OFD_SETLKW : F_OFD_SETLK, &l); while (r != 0 && errno == EINTR);
    return r == 0 ? 0 : (errno == EACCES ? EAGAIN : errno);
}

// What this process knows of one mutex object (its lock file's identity): the owning thread and its count, a thread busy
// taking it, the descriptor that holds the owner byte (a dup, so the lock outlives the handle it was taken through, as
// ownership does on Windows), and how many handles of it are open here.
struct State {
    std::thread::id thread;
    unsigned count = 0;
    bool taking = false;
    int fd = -1;
    unsigned handles = 0;
};
std::mutex g_lock;
std::condition_variable g_changed;   // an owner released, a take ended, an event was set
auto& g_mutexes = *new std::map<std::pair<dev_t, ino_t>, State>;   // never destroyed (handles may close during exit)

// The lock file of a name: Local\X is X (as in a desktop session); Global\X keeps its prefix (another object, as on Windows).
bool LockPath(const wchar_t* name, std::string& out, DWORD& err) {
    std::wstring n = name;
    if (n.rfind(L"Local\\", 0) == 0) n.erase(0, 6);
    std::string enc;
    for (unsigned char c : ToUtf8(n)) {
        if (isalnum(c) || c == '.' || c == '-' || c == '_') {
            enc += (char)c;
        } else {
            char b[4];
            std::snprintf(b, sizeof b, "%%%02X", c);
            enc += b;
        }
    }
    if (enc.empty() || enc.size() > 200) { err = ERROR_INVALID_NAME; return false; }
    std::string dir;
    if (const char* r = getenv("XDG_RUNTIME_DIR"); r && *r && access(r, W_OK) == 0) dir = r;
    else dir = XdgDataHome();
    mkdir(dir.c_str(), 0700);
    out = dir + "/wronggpu-mutex-" + enc + ".lock";
    return true;
}

HANDLE OpenNamed(LPCWSTR name, bool create, bool initialOwner) {
    if (!name || !*name) { SetLastError(ERROR_NOT_SUPPORTED); return nullptr; }
    std::string path;
    DWORD err = 0;
    if (!LockPath(name, path, err)) { SetLastError(err); return nullptr; }
    const int fd = open(path.c_str(), O_RDWR | O_CLOEXEC | (create ? O_CREAT : 0), 0600);
    if (fd < 0) {
        SetLastError(errno == ENOENT ? ERROR_FILE_NOT_FOUND : ErrorFromErrno(errno, path));
        return nullptr;
    }
    struct stat st;
    if (fstat(fd, &st) != 0) { SetErrorFromErrno(errno, path); close(fd); return nullptr; }
    bool existed = true;
    int e = SetLock(fd, F_WRLCK, kExistsByte, false);
    if (e == 0) {   // no other handle anywhere: a new object, not owned whatever an earlier one left in the file
        if (!create) { close(fd); SetLastError(ERROR_FILE_NOT_FOUND); return nullptr; }
        existed = false;
        if (pwrite(fd, &kFree, 1, 0) != 1) e = errno;
        if (!e) e = SetLock(fd, F_RDLCK, kExistsByte, false);   // the write lock becomes a read lock at once
    } else if (e == EAGAIN) {
        e = SetLock(fd, F_RDLCK, kExistsByte, true);   // waits only while a creator holds its write lock for a moment
    }
    if (e) {   // a file system without these locks (ENOLCK, EINVAL) or another failure
        close(fd);
        SetLastError(e == ENOLCK || e == EINVAL ? ERROR_NOT_SUPPORTED : ErrorFromErrno(e, path));
        return nullptr;
    }
    Handle* h = NewHandle(Kind::Mutex);
    h->fd = fd;
    h->dev = st.st_dev;
    h->ino = st.st_ino;
    {
        std::lock_guard<std::mutex> g(g_lock);
        ++g_mutexes[{st.st_dev, st.st_ino}].handles;
    }
    if (initialOwner && !existed) WaitMutex(h, 0);   // Windows: the creator owns it; whoever opened it meanwhile could own it first
    SetLastError(existed && create ? ERROR_ALREADY_EXISTS : 0);
    return h;
}
}  // namespace

DWORD wgposix::WaitMutex(Handle* h, DWORD ms) {
    using clock = std::chrono::steady_clock;
    const clock::time_point deadline = ms == INFINITE ? clock::time_point::max() : clock::now() + std::chrono::milliseconds(ms);
    const std::thread::id me = std::this_thread::get_id();
    std::unique_lock<std::mutex> g(g_lock);
    State& s = g_mutexes[{h->dev, h->ino}];
    if (s.count && s.thread == me) { ++s.count; return WAIT_OBJECT_0; }   // recursive for its owner
    while (s.count || s.taking) {   // another thread of this process owns it or is taking it
        if (ms == INFINITE) g_changed.wait(g);
        else if (g_changed.wait_until(g, deadline) == std::cv_status::timeout && (s.count || s.taking)) return WAIT_TIMEOUT;
    }
    s.taking = true;
    g.unlock();
    const int fd = fcntl(h->fd, F_DUPFD_CLOEXEC, 0);
    int err = fd < 0 ? errno : 0;
    bool got = false;
    for (unsigned pause = 1; !err;) {   // between processes: polled, so the timeout holds
        const int e = SetLock(fd, F_WRLCK, kOwnerByte, false);
        if (e == 0) { got = true; break; }
        if (e != EAGAIN) { err = e; break; }
        const clock::time_point now = clock::now();
        if (now >= deadline) break;
        const auto left = std::chrono::duration_cast<std::chrono::milliseconds>(deadline - now).count();
        std::this_thread::sleep_for(std::chrono::milliseconds(std::min<long long>(pause, std::max<long long>(left, 1))));
        pause = std::min(pause * 2, 20u);
    }
    bool abandoned = false;
    if (got) {
        char mark = kFree;
        abandoned = pread(fd, &mark, 1, 0) == 1 && mark == kOwned;
        if (pwrite(fd, &kOwned, 1, 0) != 1) { err = errno; got = false; SetLock(fd, F_UNLCK, kOwnerByte, false); }
    }
    g.lock();
    s.taking = false;
    if (!got) {
        if (fd >= 0) close(fd);
        g_changed.notify_all();
        if (err) { SetLastError(ErrorFromErrno(err, "")); return WAIT_FAILED; }
        return WAIT_TIMEOUT;
    }
    s.thread = me;
    s.count = 1;
    s.fd = fd;
    return abandoned ? WAIT_ABANDONED : WAIT_OBJECT_0;
}

void wgposix::CloseMutex(Handle* h) {
    std::lock_guard<std::mutex> g(g_lock);
    auto it = g_mutexes.find({h->dev, h->ino});
    if (it == g_mutexes.end() || --it->second.handles) return;
    // the last handle here: the object ends for this process (on Windows it is destroyed); an owner's lock goes with it, and
    // the file still says "owned", so a taker elsewhere that kept it alive sees it abandoned
    if (it->second.fd >= 0) close(it->second.fd);
    g_mutexes.erase(it);
    g_changed.notify_all();
}

HANDLE CreateMutexW(LPSECURITY_ATTRIBUTES, BOOL initialOwner, LPCWSTR name) { return OpenNamed(name, true, initialOwner != FALSE); }
HANDLE OpenMutexW(DWORD, BOOL, LPCWSTR name) { return OpenNamed(name, false, false); }

BOOL ReleaseMutex(HANDLE hv) {
    Handle* h = Lookup(hv, Kind::Mutex);
    if (!h) return FALSE;
    std::lock_guard<std::mutex> g(g_lock);
    auto it = g_mutexes.find({h->dev, h->ino});
    if (it == g_mutexes.end() || !it->second.count || it->second.thread != std::this_thread::get_id()) return Fail(ERROR_NOT_OWNER);
    State& s = it->second;
    if (--s.count) return TRUE;
    // "free" before the lock goes, so the next owner never reads a released mutex as abandoned
    if (pwrite(s.fd, &kFree, 1, 0) != 1) { ++s.count; SetErrorFromErrno(errno, ""); return FALSE; }
    SetLock(s.fd, F_UNLCK, kOwnerByte, false);
    close(s.fd);
    s.fd = -1;
    g_changed.notify_all();
    return TRUE;
}

// ---- events -------------------------------------------------------------------------------------------------------------
HANDLE CreateEventW(LPSECURITY_ATTRIBUTES, BOOL manualReset, BOOL initialState, LPCWSTR name) {
    if (name && *name) { SetLastError(ERROR_NOT_SUPPORTED); return nullptr; }
    Handle* h = NewHandle(Kind::Event);
    h->manualReset = manualReset != FALSE;
    h->signaled = initialState != FALSE;
    SetLastError(0);
    return h;
}
BOOL SetEvent(HANDLE hv) {
    Handle* h = Lookup(hv, Kind::Event);
    if (!h) return FALSE;
    std::lock_guard<std::mutex> g(g_lock);
    h->signaled = true;
    g_changed.notify_all();
    return TRUE;
}
DWORD wgposix::WaitEvent(Handle* h, DWORD ms) {
    using clock = std::chrono::steady_clock;
    const clock::time_point deadline = ms == INFINITE ? clock::time_point::max() : clock::now() + std::chrono::milliseconds(ms);
    std::unique_lock<std::mutex> g(g_lock);
    while (!h->signaled) {
        if (ms == INFINITE) g_changed.wait(g);
        else if (g_changed.wait_until(g, deadline) == std::cv_status::timeout && !h->signaled) return WAIT_TIMEOUT;
    }
    if (!h->manualReset) h->signaled = false;   // an auto-reset event lets one waiter through
    return WAIT_OBJECT_0;
}
