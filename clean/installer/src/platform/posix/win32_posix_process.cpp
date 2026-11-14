// SPDX-License-Identifier: MIT
// Linux: Windows process, pipe, job object, token and ShellExecute calls on POSIX (see win32_posix.h).
//
// A process is a child spawned with posix_spawn, reaped once however many handles name it. CREATE_SUSPENDED is not honoured
// (ResumeThread does nothing): a job object here is a list of process ids, killed one by one, so a child of a child is not in
// it. No window or priority class exists; those flags are ignored.
#include <dirent.h>
#include <fcntl.h>
#include <poll.h>
#include <signal.h>
#include <sys/ioctl.h>
#include <spawn.h>
#include <sys/wait.h>

#include <algorithm>
#include <cerrno>
#include <map>

#include "win32_posix_internal.h"

extern char** environ;

namespace wgposix {

// ---- children ----------------------------------------------------------------------------------------------------------
static std::mutex g_childLock;
static std::map<int, ChildState> g_children;

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
static void SweepChildren() {   // children nobody waits for are reaped here, on the next start
    std::lock_guard<std::mutex> g(g_childLock);
    for (auto& [pid, c] : g_children) Reap(pid, c);
}

// ---- jobs --------------------------------------------------------------------------------------------------------------
static std::mutex g_jobLock;
static std::vector<Handle*> g_jobs;
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
void ReleaseProcess(Handle*) {}

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

    posix_spawn_file_actions_t fa;
    posix_spawn_file_actions_init(&fa);
    if (!cwd.empty()) posix_spawn_file_actions_addchdir_np(&fa, cwd.c_str());
    if (si && (si->dwFlags & STARTF_USESTDHANDLES)) {
        const HANDLE std3[3] = {si->hStdInput, si->hStdOutput, si->hStdError};
        for (int n = 0; n < 3; ++n) {
            Handle* h = std3[n] && std3[n] != INVALID_HANDLE_VALUE ? Lookup(std3[n], Kind::File) : nullptr;
            if (!h && std3[n] && std3[n] != INVALID_HANDLE_VALUE) h = Lookup(std3[n], Kind::Pipe);
            if (h) posix_spawn_file_actions_adddup2(&fa, h->fd, n);
            else posix_spawn_file_actions_addopen(&fa, n, "/dev/null", n == 0 ? O_RDONLY : O_WRONLY, 0);
        }
    }
    SweepChildren();
    pid_t pid = 0;
    const int rc = search ? posix_spawnp(&pid, exe.c_str(), &fa, nullptr, argv.data(), env ? envp.data() : environ)
                          : posix_spawn(&pid, exe.c_str(), &fa, nullptr, argv.data(), env ? envp.data() : environ);
    posix_spawn_file_actions_destroy(&fa);
    if (rc != 0) {
        SetErrorFromErrno(rc, exe);   // ENOEXEC: a file that is no program here (a Windows .exe) is a bad format
        return FALSE;
    }
    {
        std::lock_guard<std::mutex> g(g_childLock);
        g_children[pid] = ChildState{};
    }
    Handle* p = NewHandle(Kind::Process);
    p->pid = pid;
    Handle* t = NewHandle(Kind::Thread);
    t->pid = pid;
    if (pi) { pi->hProcess = p; pi->hThread = t; pi->dwProcessId = (DWORD)pid; pi->dwThreadId = (DWORD)pid; }
    else { CloseHandle(p); CloseHandle(t); }
    return TRUE;
}
DWORD WaitForSingleObject(HANDLE hv, DWORD ms) {
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
DWORD ResumeThread(HANDLE hv) { return Lookup(hv, Kind::Thread) ? 1 : (DWORD)-1; }
HANDLE GetCurrentProcess() {
    static Handle* self = [] { Handle* h = NewHandle(Kind::Process); h->pid = getpid(); return h; }();
    return self;
}
HANDLE OpenProcess(DWORD, BOOL, DWORD pid) {
    if (kill((pid_t)pid, 0) != 0 && errno != EPERM) { SetErrorFromErrno(errno, ""); return nullptr; }
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
        auto slurp = [](const std::string& f) {
            std::string r;
            if (FILE* fp = std::fopen(f.c_str(), "r")) { char b[4096]; size_t n; while ((n = std::fread(b, 1, sizeof(b), fp)) > 0) r.append(b, n); std::fclose(fp); }
            return r;
        };
        std::string name = slurp(base + "/comm");
        while (!name.empty() && name.back() == '\n') name.pop_back();
        const std::string cmd = slurp(base + "/cmdline").c_str();   // up to the first NUL: argv[0]
        const size_t cut = cmd.find_last_of("\\/");
        const std::string first = cut == std::string::npos ? cmd : cmd.substr(cut + 1);
        if (first.size() > 4 && strcasecmp(first.c_str() + first.size() - 4, ".exe") == 0) name = first;
        if (name.empty()) continue;
        PROCESSENTRY32W pe{};
        pe.dwSize = sizeof(pe);
        pe.th32ProcessID = (DWORD)pid;
        const std::string stat = slurp(base + "/stat");
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

// The Windows path of a Wine process is not one this layer can give; a caller that cannot tell takes the process for the one it
// looks for, and so does a process it cannot open.
BOOL QueryFullProcessImageNameW(HANDLE process, DWORD, LPWSTR out, LPDWORD size) {
    Handle* p = Lookup(process, Kind::Process);
    if (!p) return FALSE;
    char link[4096];
    const ssize_t n = readlink(("/proc/" + std::to_string(p->pid) + "/exe").c_str(), link, sizeof(link) - 1);
    if (n < 0) { SetErrorFromErrno(errno, ""); return FALSE; }
    const std::wstring w = FromUtf8(link, (size_t)n);
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
    {
        std::lock_guard<std::mutex> g(g_childLock);
        g_children[pid] = ChildState{};
    }
    sei->hInstApp = reinterpret_cast<HINSTANCE>(static_cast<uintptr_t>(33));
    if (sei->fMask & SEE_MASK_NOCLOSEPROCESS) { Handle* p = NewHandle(Kind::Process); p->pid = pid; sei->hProcess = p; }
    return TRUE;
}
