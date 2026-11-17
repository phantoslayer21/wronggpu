// SPDX-License-Identifier: MIT
// Linux: the Restart Manager's list of processes that hold files (restartmanager.h has what it sees and what it cannot).
#include <dirent.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cctype>
#include <cerrno>
#include <cstdio>
#include <fstream>
#include <map>
#include <set>
#include <sstream>

#include <restartmanager.h>

#include "win32_posix_internal.h"

using namespace wgposix;

namespace {
std::mutex g_rmLock;
DWORD g_nextSession = 1;
auto& g_sessions = *new std::map<DWORD, std::vector<std::string>>;   // session: the files registered (POSIX paths)

bool Numeric(const char* s) {
    if (!*s) return false;
    for (; *s; ++s) if (*s < '0' || *s > '9') return false;
    return true;
}
std::string ReadSmall(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    std::ostringstream o;
    o << f.rdbuf();
    return o.str();
}
// The program's file name: the first word of its command line (a Wine process shows its Windows program there), else comm.
std::wstring AppName(const std::string& pid) {
    std::string cmd = ReadSmall("/proc/" + pid + "/cmdline");
    cmd = cmd.substr(0, cmd.find('\0'));
    const size_t cut = cmd.find_last_of("/\\");
    if (cut != std::string::npos) cmd.erase(0, cut + 1);
    if (cmd.empty()) {
        cmd = ReadSmall("/proc/" + pid + "/comm");
        while (!cmd.empty() && cmd.back() == '\n') cmd.pop_back();
    }
    return FromUtf8(cmd);
}
// The process's start as a FILETIME: boot time (/proc/stat btime) plus its start in clock ticks (/proc/<pid>/stat field 22).
FILETIME StartTime(const std::string& pid) {
    FILETIME ft{};
    const std::string st = ReadSmall("/proc/" + pid + "/stat");
    const size_t close = st.rfind(')');   // the name in parentheses may hold spaces
    if (close == std::string::npos) return ft;
    std::istringstream rest(st.substr(close + 2));
    std::string field;
    unsigned long long ticks = 0;
    for (int i = 3; i <= 22 && rest >> field; ++i) if (i == 22) ticks = std::strtoull(field.c_str(), nullptr, 10);
    std::istringstream sys(ReadSmall("/proc/stat"));
    std::string line;
    unsigned long long boot = 0;
    while (std::getline(sys, line)) if (line.rfind("btime ", 0) == 0) boot = std::strtoull(line.c_str() + 6, nullptr, 10);
    const long hz = sysconf(_SC_CLK_TCK);
    if (!boot || hz <= 0) return ft;
    struct timespec t;
    t.tv_sec = (time_t)(boot + ticks / (unsigned long long)hz);
    t.tv_nsec = (long)((ticks % (unsigned long long)hz) * (1000000000ull / (unsigned long long)hz));
    return FileTimeOf(t);
}
typedef std::set<std::pair<dev_t, ino_t>> Ids;
bool Wanted(const Ids& ids, const struct stat& st) { return ids.count({st.st_dev, st.st_ino}) != 0; }
// The process's real user (status "Uid:"; /proc/<pid> itself shows root for a process that is not dumpable); -1: gone.
long long RealUid(const std::string& base) {
    std::istringstream st(ReadSmall(base + "/status"));
    std::string line;
    while (std::getline(st, line)) if (line.rfind("Uid:", 0) == 0) return std::strtoll(line.c_str() + 4, nullptr, 10);
    return -1;
}
// Does the process have one of the files mapped? 1, 0; -1 its maps cannot be read; -2 it ended. A maps line is "address perms
// offset major:minor inode path"; the device there is the superblock's (on btrfs not what stat() says), so a line whose inode
// matches is confirmed by a stat() of its path as the process sees it (/proc/<pid>/root: a game in a container sees other paths).
int ScanMaps(const std::string& base, const Ids& ids) {
    FILE* maps = std::fopen((base + "/maps").c_str(), "re");
    if (!maps) return errno == ENOENT ? -2 : -1;
    std::set<ino_t> inodes;
    for (auto& id : ids) inodes.insert(id.second);
    char* buf = nullptr;
    size_t cap = 0;
    bool found = false;
    while (!found && getline(&buf, &cap, maps) > 0) {
        std::istringstream l(buf);
        std::string addr, perms, offset, dev, path;
        unsigned long long inode = 0;
        if (!(l >> addr >> perms >> offset >> dev >> inode) || !inode || !inodes.count((ino_t)inode)) continue;
        std::getline(l >> std::ws, path);
        struct stat st;
        found = !path.empty() && path[0] == '/' &&
                ((stat((base + "/root" + path).c_str(), &st) == 0 && Wanted(ids, st)) || (stat(path.c_str(), &st) == 0 && Wanted(ids, st)));
    }
    std::free(buf);
    std::fclose(maps);
    return found ? 1 : 0;
}
// A Windows program under Wine, or Wine itself: Wine sets the first word of the command line to the Windows path of the
// program (C:\..., Z:\...); wineserver and the preloader are named wine*. Anything else does not load Windows images.
bool WineProgram(const std::string& base) {
    std::string cmd = ReadSmall(base + "/cmdline");
    cmd = cmd.substr(0, cmd.find('\0'));
    if (cmd.find('\\') != std::string::npos) return true;
    const size_t cut = cmd.find_last_of('/');
    std::string name = cut == std::string::npos ? cmd : cmd.substr(cut + 1);
    for (auto& c : name) c = (char)std::tolower((unsigned char)c);
    return (name.size() > 4 && name.compare(name.size() - 4, 4, ".exe") == 0) || name.rfind("wine", 0) == 0;
}
}  // namespace

int wgposix::ImageInUse(dev_t dev, ino_t ino, int* who) {
    const Ids ids{{dev, ino}};
    DIR* proc = opendir("/proc");
    if (!proc) return -1;
    int result = 0;
    while (dirent* e = readdir(proc)) {
        if (!Numeric(e->d_name)) continue;
        const int pid = std::atoi(e->d_name);
        const std::string base = std::string("/proc/") + e->d_name;
        if (pid == (int)getpid() || RealUid(base) != (long long)getuid() || !WineProgram(base)) continue;
        const int r = ScanMaps(base, ids);
        if (r == 1) {
            result = 1;
            if (who) *who = pid;
            break;
        }
        if (r == -1 && result == 0) {
            result = -1;
            if (who) *who = pid;
        }
    }
    closedir(proc);
    return result;
}

// Does process `pid` hold one of the files (1, 0)? -1: a process of this user that could not be read (its fd or maps
// closed: not dumpable, or holding capabilities); -2: another user's (not looked at: closed to this user on any Linux) or gone.
int wgposix::RmScanProcess(int pidNum, const std::vector<std::pair<dev_t, ino_t>>& wanted) {
    const Ids ids(wanted.begin(), wanted.end());
    const std::string pid = std::to_string(pidNum), base = "/proc/" + pid;
    if (RealUid(base) != (long long)getuid()) return -2;
    DIR* d = opendir((base + "/fd").c_str());
    if (!d) return errno == ENOENT ? -2 : -1;   // ENOENT: it ended meanwhile
    bool found = false;
    while (dirent* e = readdir(d)) {
        if (!Numeric(e->d_name)) continue;
        struct stat st;
        if (stat((base + "/fd/" + e->d_name).c_str(), &st) == 0 && Wanted(ids, st)) { found = true; break; }
    }
    closedir(d);
    return found ? 1 : ScanMaps(base, ids);
}

DWORD RmStartSession(DWORD* session, DWORD flags, WCHAR key[]) {
    if (!session || flags) return ERROR_INVALID_PARAMETER;
    std::lock_guard<std::mutex> g(g_rmLock);
    *session = g_nextSession++;
    g_sessions[*session];
    if (key) {
        const std::string k = std::to_string(getpid()) + "-" + std::to_string(*session);
        size_t i = 0;
        for (; i < k.size() && i < CCH_RM_SESSION_KEY; ++i) key[i] = (WCHAR)k[i];
        key[i] = 0;
    }
    return ERROR_SUCCESS;
}

DWORD RmRegisterResources(DWORD session, UINT nFiles, LPCWSTR files[], UINT nApplications, RM_UNIQUE_PROCESS[], UINT nServices, LPCWSTR[]) {
    if (nApplications || nServices) return ERROR_NOT_SUPPORTED;
    std::vector<std::string> paths;
    for (UINT i = 0; i < nFiles; ++i) {
        std::string p;
        if (!files || !files[i] || !ToPosix(files[i], p)) return ERROR_INVALID_PARAMETER;
        paths.push_back(p);
    }
    std::lock_guard<std::mutex> g(g_rmLock);
    auto it = g_sessions.find(session);
    if (it == g_sessions.end()) return ERROR_INVALID_HANDLE;
    it->second.insert(it->second.end(), paths.begin(), paths.end());
    return ERROR_SUCCESS;
}

DWORD RmGetList(DWORD session, UINT* needed, UINT* count, RM_PROCESS_INFO info[], LPDWORD reasons) {
    if (!needed || !count || !reasons || (*count && !info)) return ERROR_INVALID_PARAMETER;
    std::vector<std::string> paths;
    {
        std::lock_guard<std::mutex> g(g_rmLock);
        auto it = g_sessions.find(session);
        if (it == g_sessions.end()) return ERROR_INVALID_HANDLE;
        paths = it->second;
    }
    Ids ids;
    for (auto& p : paths) {
        struct stat st;
        if (stat(p.c_str(), &st) == 0) ids.insert({st.st_dev, st.st_ino});   // a file that is not there: nobody holds it
    }
    *reasons = RmRebootReasonNone;
    std::vector<std::string> holders;
    if (!ids.empty()) {
        DIR* proc = opendir("/proc");
        if (!proc) return ERROR_ACCESS_DENIED;
        const std::vector<std::pair<dev_t, ino_t>> wanted(ids.begin(), ids.end());
        bool denied = false;
        while (dirent* e = readdir(proc)) {
            if (!Numeric(e->d_name)) continue;
            const int r = RmScanProcess(std::atoi(e->d_name), wanted);
            if (r == 1) holders.push_back(e->d_name);
            denied = denied || r == -1;
        }
        closedir(proc);
        if (denied) *reasons |= RmRebootReasonPermissionDenied;
    }
    *needed = (UINT)holders.size();
    if (*count < holders.size()) return ERROR_MORE_DATA;
    *count = (UINT)holders.size();
    for (size_t i = 0; i < holders.size(); ++i) {
        RM_PROCESS_INFO& r = info[i];
        std::memset(&r, 0, sizeof r);
        r.Process.dwProcessId = (DWORD)std::strtoul(holders[i].c_str(), nullptr, 10);
        r.Process.ProcessStartTime = StartTime(holders[i]);
        const std::wstring app = AppName(holders[i]);
        wcsncpy(r.strAppName, app.c_str(), CCH_RM_MAX_APP_NAME);
        r.ApplicationType = RmUnknownApp;
        r.bRestartable = FALSE;
    }
    return ERROR_SUCCESS;
}

DWORD RmEndSession(DWORD session) {
    std::lock_guard<std::mutex> g(g_rmLock);
    return g_sessions.erase(session) ? ERROR_SUCCESS : ERROR_INVALID_HANDLE;
}
