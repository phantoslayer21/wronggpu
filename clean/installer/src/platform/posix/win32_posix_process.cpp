// SPDX-License-Identifier: MIT
// Linux: Windows process, pipe, job object, token and ShellExecute calls on POSIX (see win32_posix.h).
//
// A process is a child spawned with posix_spawn, reaped once however many handles name it, and forgotten once it is reaped and
// no handle names it. CREATE_SUSPENDED forks instead: the program is found and checked first, so a missing or bad one fails the
// call as it does without the flag; the child then drops the parent's signal handlers and close-on-exec files, stops itself
// (SIGSTOP) before it execs, and ResumeThread sends SIGCONT. Until it has exec'd, its image is the one recorded for it. A job object here is a
// list of process ids, killed one by one, so a child of a child is not in it. No window or priority class exists; those flags
// are ignored.
#include <dirent.h>
#include <fcntl.h>
#include <poll.h>
#include <signal.h>
#include <sys/ioctl.h>
#include <spawn.h>
#include <sys/resource.h>
#include <sys/syscall.h>
#include <sys/wait.h>

#include <algorithm>
#include <cerrno>
#include <climits>
#include <map>
#include <pthread.h>

#include "win32_posix_internal.h"

extern char** environ;

namespace wgposix {

// ---- children ----------------------------------------------------------------------------------------------------------
static std::mutex g_childLock;
static auto& g_children = *new std::map<int, ChildState>;  // never destroyed (see g_handles in win32_posix.cpp)

static void Reap(int pid, ChildState& c) {
    if (c.exited) return;
    int status = 0;
    const pid_t r = waitpid(pid, &status, WNOHANG);
    if (r == pid) {
        c.exited = true;
        c.code = WIFEXITED(status) ? WEXITSTATUS(status) : c.forcedCode >= 0 ? c.forcedCode : 128 + WTERMSIG(status);
    } else if (r < 0 && errno == ECHILD) {   // not ours any more (reaped elsewhere): gone
        c.exited = true;
        c.code = c.forcedCode >= 0 ? c.forcedCode : 0;
    }
}
bool PollChild(int pid, ChildState& out) {
    std::lock_guard<std::mutex> g(g_childLock);
    auto it = g_children.find(pid);
    if (it == g_children.end()) return false;
    Reap(pid, it->second);
    out = it->second;
    return true;
}
void KillChild(int pid, int code) {
    std::lock_guard<std::mutex> g(g_childLock);
    auto it = g_children.find(pid);
    if (it == g_children.end()) return;
    Reap(pid, it->second);
    if (it->second.exited) return;
    it->second.forcedCode = code;
    kill(pid, SIGKILL);
}
static void SweepChildren() {   // children nobody waits for are reaped here, on the next start; finished ones no handle names go
    std::lock_guard<std::mutex> g(g_childLock);
    for (auto it = g_children.begin(); it != g_children.end();) {
        Reap(it->first, it->second);
        it = it->second.exited && it->second.refs == 0 ? g_children.erase(it) : std::next(it);
    }
}
static void AddChild(int pid, int handles, bool suspended, const std::string& image = std::string()) {
    std::lock_guard<std::mutex> g(g_childLock);
    ChildState c;
    c.refs = handles;
    c.suspended = suspended;
    c.image = image;
    g_children[pid] = c;
}

// ---- jobs --------------------------------------------------------------------------------------------------------------
static std::mutex g_jobLock;
static auto& g_jobs = *new std::vector<Handle*>;
static void KillJobMembers(Handle* job, int code) {
    for (int pid : job->pids) KillChild(pid, code);
}
static void KillJobsAtExit() {
    std::lock_guard<std::mutex> g(g_jobLock);
    for (Handle* j : g_jobs) if (j->killOnClose) KillJobMembers(j, 1);
}
void ReleaseJob(Handle* h) {
    std::lock_guard<std::mutex> g(g_jobLock);
    if (h->killOnClose) KillJobMembers(h, 1);
    g_jobs.erase(std::remove(g_jobs.begin(), g_jobs.end(), h), g_jobs.end());
}
void ReleaseProcess(Handle* h) {
    std::lock_guard<std::mutex> g(g_childLock);
    auto it = g_children.find(h->pid);
    if (it == g_children.end()) return;
    if (it->second.refs > 0) --it->second.refs;
    if (it->second.refs > 0) return;
    Reap(h->pid, it->second);
    if (it->second.exited) g_children.erase(it);
}

}  // namespace wgposix

using namespace wgposix;

// The command line the way CommandLineToArgvW reads it: the first word has no escapes; the rest follow the backslash-quote rules.
static std::vector<std::wstring> SplitCommandLine(const wchar_t* c) {
    std::vector<std::wstring> out;
    const size_t n = wcslen(c);
    size_t i = 0;
    while (i < n && (c[i] == L' ' || c[i] == L'\t')) ++i;
    if (i < n) {
        const bool quotedName = c[i] == L'"';
        const size_t from = quotedName ? ++i : i;
        while (i < n && (quotedName ? c[i] != L'"' : (c[i] != L' ' && c[i] != L'\t'))) ++i;
        out.emplace_back(c + from, i - from);
        if (quotedName && i < n) ++i;
    }
    for (;;) {
        while (i < n && (c[i] == L' ' || c[i] == L'\t')) ++i;
        if (i >= n) break;
        std::wstring a;
        bool quoted = false;
        while (i < n) {
            size_t bs = 0;
            while (i < n && c[i] == L'\\') { ++bs; ++i; }
            if (i < n && c[i] == L'"') {
                a.append(bs / 2, L'\\');
                if (bs % 2) { a += L'"'; ++i; }
                else if (quoted && i + 1 < n && c[i + 1] == L'"') { a += L'"'; i += 2; }
                else { quoted = !quoted; ++i; }
            } else {
                a.append(bs, L'\\');
                if (i >= n || (!quoted && (c[i] == L' ' || c[i] == L'\t'))) break;
                a += c[i++];
            }
        }
        out.push_back(a);
    }
    return out;
}

// ---- the part of a suspended start that happens before fork --------------------------------------------------------------
// Does the file start like a program the kernel runs: an ELF image or a #! script. Anything else is what execve calls ENOEXEC.
static bool LooksRunnable(const std::string& path) {
    const int fd = open(path.c_str(), O_RDONLY | O_CLOEXEC);
    if (fd < 0) return true;   // unreadable but executable (a setuid program): leave it to exec
    char b[4];
    const ssize_t n = read(fd, b, sizeof(b));
    close(fd);
    return n >= 2 && ((n == 4 && std::memcmp(b, "\x7f" "ELF", 4) == 0) || (b[0] == '#' && b[1] == '!'));
}
// One candidate for the program: 0 when it can be run, else the errno execve would give for it. `seen` is where the file is as
// the child will see it (after chdir to `cwd`).
static int CheckCandidate(const std::string& cand, const std::string& cwd, std::string& seen) {
    seen = cand[0] == '/' || cwd.empty() ? cand : cwd + "/" + cand;
    struct stat st;
    if (stat(seen.c_str(), &st) != 0) return errno;
    if (!S_ISREG(st.st_mode) || access(seen.c_str(), X_OK) != 0) return EACCES;
    return LooksRunnable(seen) ? 0 : ENOEXEC;
}
// Finds the program the way posix_spawn(p) does and reports the error it would: `exec` is the name to hand to execve, `image` the
// file's real path. Returns 0, or a Windows error code.
static DWORD ResolveProgram(const std::string& exe, bool search, const std::string& cwd, std::string& exec, std::string& image) {
    std::string seen;
    int failure = 0;
    std::vector<std::string> candidates;
    if (!search) {
        candidates.push_back(exe);
    } else {
        const char* pathEnv = std::getenv("PATH");
        std::string path = pathEnv ? pathEnv : "/usr/local/bin:/usr/bin:/bin";
        for (size_t from = 0;;) {
            const size_t colon = path.find(':', from);
            std::string dir = path.substr(from, colon == std::string::npos ? std::string::npos : colon - from);
            candidates.push_back((dir.empty() ? std::string(".") : dir) + "/" + exe);
            if (colon == std::string::npos) break;
            from = colon + 1;
        }
    }
    for (const std::string& cand : candidates) {
        const int e = CheckCandidate(cand, cwd, seen);
        if (e == 0) {
            exec = cand;
            char real[PATH_MAX];
            image = realpath(seen.c_str(), real) ? real : seen;
            return 0;
        }
        if (!search) return ErrorFromErrno(e, cand);
        if (e != ENOENT && e != ENOTDIR && !failure) failure = e;
    }
    return failure ? ErrorFromErrno(failure, exe) : ERROR_FILE_NOT_FOUND;
}

// Between fork and exec in the child: only calls that are safe there. Every signal is blocked by now (the parent blocked them
// before it forked), so no handler of the parent's can run; the handlers go, and so does every close-on-exec file the child
// would hold until it execs.
static void ResetSignals() {
    for (int sig = 1; sig < NSIG; ++sig) {
        struct sigaction sa;
        if (sigaction(sig, nullptr, &sa) != 0 || sa.sa_handler == SIG_IGN) continue;   // an ignored signal stays ignored through exec
        struct sigaction dfl;
        std::memset(&dfl, 0, sizeof(dfl));
        dfl.sa_handler = SIG_DFL;
        sigaction(sig, &dfl, nullptr);
    }
}
static void CloseCloexecFiles(int maxFd) {
    struct Dirent64 { uint64_t ino; int64_t off; unsigned short len; unsigned char type; char name[1]; };
    const int dirFd = open("/proc/self/fd", O_RDONLY | O_DIRECTORY | O_CLOEXEC);
    if (dirFd >= 0) {
        alignas(8) char buf[2048];
        long got;
        while ((got = syscall(SYS_getdents64, dirFd, buf, sizeof(buf))) > 0) {
            for (long at = 0; at < got;) {
                const Dirent64* d = reinterpret_cast<const Dirent64*>(buf + at);
                at += d->len;
                int fd = 0;
                const char* c = d->name;
                for (; *c >= '0' && *c <= '9'; ++c) fd = fd * 10 + (*c - '0');
                if (*c || c == d->name || fd < 3 || fd == dirFd) continue;
                const int fl = fcntl(fd, F_GETFD);
                if (fl >= 0 && (fl & FD_CLOEXEC)) close(fd);
            }
        }
        close(dirFd);
        return;
    }
    for (int fd = 3; fd < maxFd; ++fd) {   // no /proc: every number the process may use
        const int fl = fcntl(fd, F_GETFD);
        if (fl >= 0 && (fl & FD_CLOEXEC)) close(fd);
    }
}

BOOL CreateProcessW(LPCWSTR app, LPWSTR cmdline, LPSECURITY_ATTRIBUTES, LPSECURITY_ATTRIBUTES, BOOL, DWORD flags, LPVOID env, LPCWSTR dir,
                    STARTUPINFOW* si, PROCESS_INFORMATION* pi) {
    std::vector<std::wstring> args = SplitCommandLine(cmdline ? cmdline : L"");
    const std::wstring exeW = app && *app ? app : args.empty() ? std::wstring() : args[0];
    if (exeW.empty()) return Fail(ERROR_FILE_NOT_FOUND);
    if (args.empty()) args.push_back(exeW);
    const bool search = !(app && *app) && exeW.find_first_of(L"\\/") == std::wstring::npos;
    std::string exe;
    if (search) exe = ToUtf8(exeW);
    else if (!ToPosix(exeW.c_str(), exe)) return FALSE;
    struct stat st;
    if (!search && stat(exe.c_str(), &st) != 0) { SetErrorFromErrno(errno, exe); return FALSE; }
    if (!search && S_ISDIR(st.st_mode)) return Fail(ERROR_ACCESS_DENIED);
    std::string cwd;
    if (dir && *dir) {
        if (!ToPosix(dir, cwd)) return FALSE;
        if (stat(cwd.c_str(), &st) != 0 || !S_ISDIR(st.st_mode)) return Fail(ERROR_DIRECTORY);
    }

    std::vector<std::string> argStore;
    for (auto& a : args) argStore.push_back(ToUtf8(a));
    std::vector<char*> argv;
    for (auto& a : argStore) argv.push_back(a.data());
    argv.push_back(nullptr);

    std::vector<std::string> envStore;
    std::vector<char*> envp;
    if (env) {
        if (flags & CREATE_UNICODE_ENVIRONMENT) {
            for (const wchar_t* p = static_cast<const wchar_t*>(env); *p; p += wcslen(p) + 1) envStore.push_back(ToUtf8(p, wcslen(p)));
        } else {
            for (const char* p = static_cast<const char*>(env); *p; p += std::strlen(p) + 1) envStore.push_back(p);
        }
        for (auto& e : envStore) envp.push_back(e.data());
        envp.push_back(nullptr);
    }

    int stdFd[3] = {-2, -2, -2};   // what the child gets as 0, 1, 2: a descriptor, -1 for /dev/null, -2 for the parent's own
    if (si && (si->dwFlags & STARTF_USESTDHANDLES)) {
        const HANDLE std3[3] = {si->hStdInput, si->hStdOutput, si->hStdError};
        for (int n = 0; n < 3; ++n) {
            Handle* h = std3[n] && std3[n] != INVALID_HANDLE_VALUE ? Lookup(std3[n], Kind::File) : nullptr;
            if (!h && std3[n] && std3[n] != INVALID_HANDLE_VALUE) h = Lookup(std3[n], Kind::Pipe);
            stdFd[n] = h ? h->fd : -1;
        }
    }
    char** const childEnv = env ? envp.data() : environ;
    SweepChildren();
    pid_t pid = 0;
    std::string image;   // a suspended start: the program the child will exec
    if (flags & CREATE_SUSPENDED) {
        // fork, not posix_spawn: the child must stop before it runs a line of the program, and the parent waits to see that.
        // Everything the child needs is made here, so that it only has to call what is safe between fork and exec.
        std::string execPath;
        const DWORD found = ResolveProgram(exe, search, cwd, execPath, image);
        if (found) return Fail(found);
        struct rlimit rl;
        const int maxFd = getrlimit(RLIMIT_NOFILE, &rl) == 0 && rl.rlim_cur < (1u << 20) ? (int)rl.rlim_cur : (1 << 20);
        sigset_t all, before;
        sigfillset(&all);
        pthread_sigmask(SIG_SETMASK, &all, &before);   // the child starts with every signal blocked, so no handler runs in it
        pid = fork();
        const int forkErr = errno;
        if (pid == 0) {
            ResetSignals();
            for (int n = 0; n < 3; ++n) {
                if (stdFd[n] == n) fcntl(n, F_SETFD, 0);
                else if (stdFd[n] >= 0) dup2(stdFd[n], n);
                else if (stdFd[n] == -1) { const int f = open("/dev/null", n == 0 ? O_RDONLY : O_WRONLY); if (f >= 0) { dup2(f, n); if (f != n) close(f); } }
            }
            CloseCloexecFiles(maxFd);
            if (!cwd.empty() && chdir(cwd.c_str()) != 0) _exit(126);
            raise(SIGSTOP);
            sigprocmask(SIG_SETMASK, &before, nullptr);
            execve(execPath.c_str(), argv.data(), childEnv);
            _exit(127);
        }
        pthread_sigmask(SIG_SETMASK, &before, nullptr);
        if (pid < 0) { SetErrorFromErrno(forkErr, exe); return FALSE; }
        int status = 0;
        pid_t w;
        while ((w = waitpid(pid, &status, WUNTRACED)) < 0 && errno == EINTR) {}
        if (w != pid || !WIFSTOPPED(status)) { SetLastError(ERROR_GEN_FAILURE); return FALSE; }   // the child died before it stopped
    } else {
        posix_spawn_file_actions_t fa;
        posix_spawn_file_actions_init(&fa);
        if (!cwd.empty()) posix_spawn_file_actions_addchdir_np(&fa, cwd.c_str());
        for (int n = 0; n < 3; ++n) {
            if (stdFd[n] >= 0) posix_spawn_file_actions_adddup2(&fa, stdFd[n], n);
            else if (stdFd[n] == -1) posix_spawn_file_actions_addopen(&fa, n, "/dev/null", n == 0 ? O_RDONLY : O_WRONLY, 0);
        }
        const int rc = search ? posix_spawnp(&pid, exe.c_str(), &fa, nullptr, argv.data(), childEnv)
                              : posix_spawn(&pid, exe.c_str(), &fa, nullptr, argv.data(), childEnv);
        posix_spawn_file_actions_destroy(&fa);
        if (rc != 0) {
            SetErrorFromErrno(rc, exe);   // ENOEXEC: a file that is no program here (a Windows .exe) is a bad format
            return FALSE;
        }
    }
    AddChild(pid, 1, (flags & CREATE_SUSPENDED) != 0, image);
    Handle* p = NewHandle(Kind::Process);
    p->pid = pid;
    Handle* t = NewHandle(Kind::Thread);
    t->pid = pid;
    if (pi) { pi->hProcess = p; pi->hThread = t; pi->dwProcessId = (DWORD)pid; pi->dwThreadId = (DWORD)pid; }
    else { CloseHandle(p); CloseHandle(t); }
    return TRUE;
}
DWORD WaitForSingleObject(HANDLE hv, DWORD ms) {
    if (Handle* m = Lookup(hv, Kind::Mutex)) return WaitMutex(m, ms);
    if (Handle* e = Lookup(hv, Kind::Event)) return WaitEvent(e, ms);
    Handle* h = Lookup(hv, Kind::Process);
    if (!h) return WAIT_FAILED;
    struct timespec start;
    clock_gettime(CLOCK_MONOTONIC, &start);
    for (;;) {
        ChildState c;
        if (!PollChild(h->pid, c) || c.exited) return WAIT_OBJECT_0;
        if (ms == 0) return WAIT_TIMEOUT;
        struct timespec now;
        clock_gettime(CLOCK_MONOTONIC, &now);
        const long long elapsed = (now.tv_sec - start.tv_sec) * 1000LL + (now.tv_nsec - start.tv_nsec) / 1000000;
        if (ms != INFINITE && elapsed >= (long long)ms) return WAIT_TIMEOUT;
        Sleep(2);
    }
}
BOOL GetExitCodeProcess(HANDLE hv, LPDWORD code) {
    Handle* h = Lookup(hv, Kind::Process);
    if (!h) return FALSE;
    ChildState c;
    *code = PollChild(h->pid, c) && c.exited ? (DWORD)c.code : STILL_ACTIVE;
    return TRUE;
}
BOOL TerminateProcess(HANDLE hv, UINT code) {
    Handle* h = Lookup(hv, Kind::Process);
    if (!h) return FALSE;
    ChildState c;
    if (!PollChild(h->pid, c) || c.exited) return Fail(ERROR_ACCESS_DENIED);
    KillChild(h->pid, (int)code);
    return TRUE;
}
// The suspend count is 1 for a process started suspended and not yet resumed, else 0.
DWORD ResumeThread(HANDLE hv) {
    Handle* t = Lookup(hv, Kind::Thread);
    if (!t) return (DWORD)-1;
    std::lock_guard<std::mutex> g(g_childLock);
    auto it = g_children.find(t->pid);
    if (it == g_children.end() || !it->second.suspended) return 0;
    it->second.suspended = false;
    kill(t->pid, SIGCONT);
    return 1;
}
HANDLE GetCurrentProcess() {
    static Handle* self = [] { Handle* h = NewHandle(Kind::Process); h->pid = getpid(); return h; }();
    return self;
}
HANDLE OpenProcess(DWORD, BOOL, DWORD pid) {
    if (kill((pid_t)pid, 0) != 0 && errno != EPERM) { SetErrorFromErrno(errno, ""); return nullptr; }
    {
        std::lock_guard<std::mutex> g(g_childLock);
        auto it = g_children.find((int)pid);
        if (it != g_children.end()) ++it->second.refs;
    }
    Handle* h = NewHandle(Kind::Process);
    h->pid = (int)pid;
    return h;
}

BOOL CreatePipe(HANDLE* read, HANDLE* write, LPSECURITY_ATTRIBUTES sa, DWORD) {
    int fds[2];
    if (pipe2(fds, O_CLOEXEC) != 0) { SetErrorFromErrno(errno, ""); return FALSE; }
    Handle* r = NewHandle(Kind::Pipe);
    Handle* w = NewHandle(Kind::Pipe);
    r->fd = fds[0]; w->fd = fds[1];
    r->inheritable = w->inheritable = sa && sa->bInheritHandle;
    *read = r; *write = w;
    return TRUE;
}
BOOL PeekNamedPipe(HANDLE hv, LPVOID, DWORD, LPDWORD, LPDWORD available, LPDWORD left) {
    Handle* h = Lookup(hv, Kind::Pipe);
    if (!h) return FALSE;
    int n = 0;
    if (ioctl(h->fd, FIONREAD, &n) != 0) { SetErrorFromErrno(errno, ""); return FALSE; }
    if (n == 0) {   // nothing waiting: a closed write end is an error, as on Windows
        struct pollfd p = {h->fd, POLLIN, 0};
        if (poll(&p, 1, 0) > 0 && (p.revents & POLLHUP) && !(p.revents & POLLIN)) return Fail(ERROR_BROKEN_PIPE);
    }
    if (available) *available = (DWORD)n;
    if (left) *left = (DWORD)n;
    return TRUE;
}
BOOL SetHandleInformation(HANDLE hv, DWORD mask, DWORD flags) {
    Handle* h = Lookup(hv, Kind::Pipe);
    if (!h) h = Lookup(hv, Kind::File);
    if (!h) return FALSE;
    if (mask & HANDLE_FLAG_INHERIT) h->inheritable = flags & HANDLE_FLAG_INHERIT;
    return TRUE;
}

HANDLE CreateJobObjectW(LPSECURITY_ATTRIBUTES, LPCWSTR) {
    static std::once_flag once;
    std::call_once(once, [] { std::atexit(KillJobsAtExit); });
    Handle* j = NewHandle(Kind::Job);
    std::lock_guard<std::mutex> g(g_jobLock);
    g_jobs.push_back(j);
    return j;
}
BOOL SetInformationJobObject(HANDLE hv, JOBOBJECTINFOCLASS cls, LPVOID info, DWORD size) {
    Handle* j = Lookup(hv, Kind::Job);
    if (!j) return FALSE;
    if (cls != JobObjectExtendedLimitInformation || size < sizeof(JOBOBJECT_EXTENDED_LIMIT_INFORMATION)) return Fail(ERROR_INVALID_PARAMETER);
    std::lock_guard<std::mutex> g(g_jobLock);
    j->killOnClose = static_cast<JOBOBJECT_EXTENDED_LIMIT_INFORMATION*>(info)->BasicLimitInformation.LimitFlags & JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
    return TRUE;
}
BOOL QueryInformationJobObject(HANDLE hv, JOBOBJECTINFOCLASS cls, LPVOID info, DWORD size, LPDWORD returned) {
    Handle* j = Lookup(hv, Kind::Job);
    if (!j) return FALSE;
    if (cls != JobObjectBasicAccountingInformation || size < sizeof(JOBOBJECT_BASIC_ACCOUNTING_INFORMATION)) return Fail(ERROR_INVALID_PARAMETER);
    auto* ai = static_cast<JOBOBJECT_BASIC_ACCOUNTING_INFORMATION*>(info);
    std::memset(ai, 0, sizeof(*ai));
    std::lock_guard<std::mutex> g(g_jobLock);
    for (int pid : j->pids) {
        ChildState c;
        ++ai->TotalProcesses;
        if (PollChild(pid, c) && !c.exited) ++ai->ActiveProcesses;
    }
    if (returned) *returned = sizeof(*ai);
    return TRUE;
}
BOOL AssignProcessToJobObject(HANDLE job, HANDLE process) {
    Handle* j = Lookup(job, Kind::Job);
    Handle* p = Lookup(process, Kind::Process);
    if (!j || !p) return FALSE;
    ChildState c;
    if (!PollChild(p->pid, c) || c.exited) return Fail(ERROR_ACCESS_DENIED);
    std::lock_guard<std::mutex> g(g_jobLock);
    j->pids.push_back(p->pid);
    return TRUE;
}
BOOL TerminateJobObject(HANDLE hv, UINT code) {
    Handle* j = Lookup(hv, Kind::Job);
    if (!j) return FALSE;
    std::lock_guard<std::mutex> g(g_jobLock);
    KillJobMembers(j, (int)code);
    return TRUE;
}

// What /proc says about a process. The name the kernel keeps (comm) stops at 15 characters, so names come from the command line.
static std::string Slurp(const std::string& f) {
    std::string r;
    if (FILE* fp = std::fopen(f.c_str(), "r")) { char b[4096]; size_t n; while ((n = std::fread(b, 1, sizeof(b), fp)) > 0) r.append(b, n); std::fclose(fp); }
    return r;
}
static std::vector<std::string> CommandLine(int pid) {
    const std::string raw = Slurp("/proc/" + std::to_string(pid) + "/cmdline");
    std::vector<std::string> out;
    for (size_t i = 0; i < raw.size();) {
        const size_t e = raw.find('\0', i);
        out.push_back(raw.substr(i, e == std::string::npos ? std::string::npos : e - i));
        if (e == std::string::npos) break;
        i = e + 1;
    }
    return out;
}
static std::string BaseOf(const std::string& p) {
    const size_t cut = p.find_last_of("\\/");
    return cut == std::string::npos ? p : p.substr(cut + 1);
}
static bool EndsExe(const std::string& s) { return s.size() > 4 && strcasecmp(s.c_str() + s.size() - 4, ".exe") == 0; }
static std::string Lower(std::string s) { for (auto& c : s) c = (char)std::tolower((unsigned char)c); return s; }
// Wine's own program: the Windows program it runs is named in the command line, not in the image.
static bool IsWineLoader(const std::string& image) {
    const std::string base = Lower(BaseOf(image));
    if (base == "wine" || base == "wine64" || base == "wine-preloader" || base == "wine64-preloader") return true;
    const std::string low = Lower(image);
    return low.find("/bin/") != std::string::npos && (low.find("proton") != std::string::npos || low.find("wine") != std::string::npos);
}
// The Windows program a Wine process runs: its own name when that is one (Wine sets it), else the first .exe argument.
static bool WineProgram(const std::vector<std::string>& args, std::string& out) {
    for (const std::string& a : args) if (EndsExe(a)) { out = a; return true; }
    return false;
}
static std::string ImageOf(int pid) {
    char link[4096];
    const ssize_t n = readlink(("/proc/" + std::to_string(pid) + "/exe").c_str(), link, sizeof(link) - 1);
    return n < 0 ? std::string() : std::string(link, (size_t)n);
}
// The program of one of our own children that was started suspended and has not exec'd yet: until then /proc shows this process's
// image, so the one recorded at the start stands in. False for any other process, and once the child runs its own image.
static bool PendingImage(int pid, std::string& image) {
    {
        std::lock_guard<std::mutex> g(g_childLock);
        auto it = g_children.find(pid);
        if (it == g_children.end() || it->second.image.empty()) return false;
        image = it->second.image;
    }
    const std::string now = ImageOf(pid);
    return !now.empty() && now == ImageOf(getpid());
}

// Processes by name: the Windows name of a Wine process is in its command line; any other process has its own name.
HANDLE CreateToolhelp32Snapshot(DWORD, DWORD) {
    Handle* s = NewHandle(Kind::Snapshot);
    DIR* d = opendir("/proc");
    if (!d) { Forget(s); SetErrorFromErrno(errno, ""); return INVALID_HANDLE_VALUE; }
    while (dirent* e = readdir(d)) {
        char* end = nullptr;
        const long pid = std::strtol(e->d_name, &end, 10);
        if (*end || pid <= 0) continue;
        const std::string base = std::string("/proc/") + e->d_name;
        const std::vector<std::string> cmd = CommandLine((int)pid);
        std::string name;
        std::string prog;
        if (PendingImage((int)pid, prog)) name = BaseOf(prog);
        if (name.empty() && IsWineLoader(ImageOf((int)pid)) && WineProgram(cmd, prog)) name = BaseOf(prog);
        if (name.empty() && !cmd.empty()) name = BaseOf(cmd[0]);
        if (name.empty()) {   // a process with no command line (a kernel thread): the kernel's own name
            name = Slurp(base + "/comm");
            while (!name.empty() && name.back() == '\n') name.pop_back();
        }
        if (name.empty()) continue;
        PROCESSENTRY32W pe{};
        pe.dwSize = sizeof(pe);
        pe.th32ProcessID = (DWORD)pid;
        const std::string stat = Slurp(base + "/stat");
        const size_t paren = stat.rfind(')');
        if (paren != std::string::npos) { int ppid = 0; std::sscanf(stat.c_str() + paren + 2, "%*c %d", &ppid); pe.th32ParentProcessID = (DWORD)ppid; }
        const std::wstring w = FromUtf8(name);
        wcsncpy(pe.szExeFile, w.c_str(), MAX_PATH - 1);
        s->procs.push_back(pe);
    }
    closedir(d);
    return s;
}
BOOL Process32FirstW(HANDLE snap, PROCESSENTRY32W* pe) {
    Handle* s = Lookup(snap, Kind::Snapshot);
    if (!s) return FALSE;
    s->next = 0;
    if (s->procs.empty()) return Fail(ERROR_NO_MORE_FILES);
    *pe = s->procs[s->next++];
    return TRUE;
}
BOOL Process32NextW(HANDLE snap, PROCESSENTRY32W* pe) {
    Handle* s = Lookup(snap, Kind::Snapshot);
    if (!s) return FALSE;
    if (s->next >= s->procs.size()) return Fail(ERROR_NO_MORE_FILES);
    *pe = s->procs[s->next++];
    return TRUE;
}

// The Windows spelling of the program a process runs. A Wine process runs the loader; its program is the .exe in the command line, and
// with none there the call fails, so a caller that cannot tell takes the process for the one it looks for, as it does for a
// process it cannot open.
BOOL QueryFullProcessImageNameW(HANDLE process, DWORD, LPWSTR out, LPDWORD size) {
    Handle* p = Lookup(process, Kind::Process);
    if (!p) return FALSE;
    const std::string image = ImageOf(p->pid);
    if (image.empty()) { SetLastError(ERROR_ACCESS_DENIED); return FALSE; }
    std::wstring w;
    std::string prog;
    if (PendingImage(p->pid, prog)) {
        w = WinPath(prog);
    } else if (IsWineLoader(image)) {
        if (!WineProgram(CommandLine(p->pid), prog)) return Fail(ERROR_NOT_FOUND);
        const bool windows = prog.size() >= 2 && ((prog[1] == ':') || prog.compare(0, 2, "\\\\") == 0);
        if (windows) w = FromUtf8(prog);
        else {
            if (prog[0] != '/') {   // relative to the process's own folder
                char cwd[4096];
                const ssize_t n = readlink(("/proc/" + std::to_string(p->pid) + "/cwd").c_str(), cwd, sizeof(cwd) - 1);
                if (n > 0) prog = std::string(cwd, (size_t)n) + "/" + prog;
            }
            w = WinPath(prog);
        }
    } else {
        w = WinPath(image);
    }
    if (!size || *size < w.size() + 1) { if (size) *size = (DWORD)w.size() + 1; return Fail(ERROR_INSUFFICIENT_BUFFER); }
    std::memcpy(out, w.c_str(), (w.size() + 1) * sizeof(wchar_t));
    *size = (DWORD)w.size();
    return TRUE;
}

// A token is this process's: its user is the real uid and "elevated" means root.
BOOL OpenProcessToken(HANDLE process, DWORD, HANDLE* token) {
    if (!Lookup(process, Kind::Process)) return FALSE;
    *token = NewHandle(Kind::Token);
    return TRUE;
}
BOOL GetTokenInformation(HANDLE token, TOKEN_INFORMATION_CLASS cls, LPVOID info, DWORD size, LPDWORD needed) {
    if (!Lookup(token, Kind::Token)) return FALSE;
    if (cls == TokenElevation) {
        if (needed) *needed = sizeof(TOKEN_ELEVATION);
        if (size < sizeof(TOKEN_ELEVATION)) return Fail(ERROR_INSUFFICIENT_BUFFER);
        static_cast<TOKEN_ELEVATION*>(info)->TokenIsElevated = geteuid() == 0;
        return TRUE;
    }
    if (cls == TokenUser) {
        const DWORD need = sizeof(TOKEN_USER) + sizeof(SID);
        if (needed) *needed = need;
        if (size < need) return Fail(ERROR_INSUFFICIENT_BUFFER);
        SID* sid = reinterpret_cast<SID*>(static_cast<char*>(info) + sizeof(TOKEN_USER));
        sid->uid = (uint32_t)geteuid();
        static_cast<TOKEN_USER*>(info)->User = SID_AND_ATTRIBUTES{sid, 0};
        return TRUE;
    }
    return Fail(ERROR_INVALID_PARAMETER);
}
BOOL ConvertSidToStringSidW(PSID sid, LPWSTR* out) {
    if (!sid) return Fail(ERROR_INVALID_PARAMETER);
    const std::wstring s = L"S-1-22-1-" + std::to_wstring(sid->uid);   // the Unix user authority
    wchar_t* m = static_cast<wchar_t*>(std::malloc((s.size() + 1) * sizeof(wchar_t)));
    if (!m) return Fail(ERROR_NOT_ENOUGH_MEMORY);
    std::memcpy(m, s.c_str(), (s.size() + 1) * sizeof(wchar_t));
    *out = m;
    return TRUE;
}

// ShellExecuteExW: documents and addresses go to xdg-open; a Windows program has no way to start here.
BOOL ShellExecuteExW(SHELLEXECUTEINFOW* sei) {
    if (!sei || !sei->lpFile || !*sei->lpFile) return Fail(ERROR_FILE_NOT_FOUND);
    const std::wstring file = sei->lpFile;
    auto ends = [&](const wchar_t* ext) { const size_t n = wcslen(ext); return file.size() >= n && _wcsicmp(file.c_str() + file.size() - n, ext) == 0; };
    if (ends(L".exe") || ends(L".com") || ends(L".bat") || ends(L".cmd") || ends(L".msi")) return Fail(ERROR_NOT_SUPPORTED);
    const size_t colon = file.find(L':');
    const bool address = colon != std::wstring::npos && colon > 1 && file.find_first_of(L"\\/") > colon;
    std::string target;
    if (address) target = ToUtf8(file);
    else if (!ToPosix(file.c_str(), target)) return FALSE;
    std::vector<char*> argv = {const_cast<char*>("xdg-open"), target.data(), nullptr};
    posix_spawn_file_actions_t fa;
    posix_spawn_file_actions_init(&fa);
    for (int n = 0; n < 3; ++n) posix_spawn_file_actions_addopen(&fa, n, "/dev/null", n == 0 ? O_RDONLY : O_WRONLY, 0);
    pid_t pid = 0;
    SweepChildren();
    const int rc = posix_spawnp(&pid, "xdg-open", &fa, nullptr, argv.data(), environ);
    posix_spawn_file_actions_destroy(&fa);
    if (rc != 0) return Fail(rc == ENOENT ? ERROR_NO_ASSOCIATION : ErrorFromErrno(rc, ""));
    AddChild(pid, (sei->fMask & SEE_MASK_NOCLOSEPROCESS) ? 1 : 0, false);
    sei->hInstApp = reinterpret_cast<HINSTANCE>(static_cast<uintptr_t>(33));
    if (sei->fMask & SEE_MASK_NOCLOSEPROCESS) { Handle* p = NewHandle(Kind::Process); p->pid = pid; sei->hProcess = p; }
    return TRUE;
}
