// SPDX-License-Identifier: MIT
// WrongGPU launcher core: the native Linux launcher's own removal (appimage.h). Written for this project.
#include "appimage.h"

#include <cstdio>
#include <filesystem>

#include "launcher.h"

namespace wg {

// A Linux path from the environment ("/home/x/WrongGPU.AppImage") as the core spells it (Z:\home\x\WrongGPU.AppImage).
static wstring CoreSpelling(wstring p) {
    if (!p.empty() && p[0] == L'/') p = L"Z:" + p;
    for (auto& c : p) if (c == L'/') c = L'\\';
    return p;
}

// The system's process folder. A WRONGGPU_TESTING build takes WRONGGPU_TEST_PROC_ROOT instead (a made-up tree the tests fill with
// the entries the real one would have); a release build never reads it.
static wstring ProcRoot() {
#ifdef WRONGGPU_TESTING
    const wstring env = EnvVar(L"WRONGGPU_TEST_PROC_ROOT");
    if (!env.empty()) return CoreSpelling(env);
#endif
    return L"Z:\\proc";
}

// A file's identity on its disk (st_dev and st_ino through the layer) and, when asked, the first 16 bytes it holds right now.
struct DiskIdent {
    DWORD volume = 0, high = 0, low = 0;
    bool operator==(const DiskIdent& o) const { return volume == o.volume && high == o.high && low == o.low; }
};
static bool IdentOf(const wstring& p, DiskIdent& id, unsigned char* head16 = nullptr) {
    HANDLE h = CreateFileW(p.c_str(), GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (h == INVALID_HANDLE_VALUE) return false;
    BY_HANDLE_FILE_INFORMATION bi{};
    bool ok = GetFileInformationByHandle(h, &bi) != FALSE;
    if (ok) { id.volume = bi.dwVolumeSerialNumber; id.high = bi.nFileIndexHigh; id.low = bi.nFileIndexLow; }
    DWORD got = 0;
    if (ok && head16) ok = ReadFile(h, head16, 16, &got, nullptr) && got == 16;
    CloseHandle(h);
    return ok;
}
// an AppImage is an ELF file with "AI" and the type number 2 in bytes 8 to 10 of its header
static bool IsAppImageHead(const unsigned char* head) {
    return head[0] == 0x7f && head[1] == 'E' && head[2] == 'L' && head[3] == 'F' && head[8] == 'A' && head[9] == 'I' && head[10] == 2;
}

// What a file read gives back as text, up to a limit (the process folder's files report size 0 and are read to their end).
static bool ReadText(const wstring& p, string& out, size_t limit = 1u << 20) {
    out.clear();
    HANDLE h = CreateFileW(p.c_str(), GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (h == INVALID_HANDLE_VALUE) return false;
    char buf[4096];
    DWORD got = 0;
    while (out.size() < limit && ReadFile(h, buf, sizeof buf, &got, nullptr) && got) out.append(buf, got);
    CloseHandle(h);
    return true;
}

// The real path of an existing file or folder in the host's own spelling: links followed, "." and ".." folded, and the case the
// disk has (a Linux disk tells "App" from "app").
// (Both spellings are host ones, taken as they are: the layer's own path code would fold the letter case.)
static bool Canonical(const string& hostPath, string& host) {
    std::error_code ec;
    const auto c = std::filesystem::canonical(hostPath, ec);
    if (ec) return false;
    host = c.string();
    return !host.empty() && host[0] == '/';
}

// The device number the way a file's st_dev reaches the core (the layer keeps its low 32 bits): Linux's own packing.
static DWORD DevNumber(unsigned major, unsigned minor) {
    return (DWORD)(((major & 0xfffu) << 8) | (minor & 0xffu) | ((minor & ~0xffu) << 12));
}

// The line of the system's mount table (mountinfo) for a mount point, as "major:minor fstype". Octal escapes (\040) are undone.
// The last line wins: it is the one on top when a folder is mounted over more than once.
static bool MountOf(const string& table, const string& point, unsigned& major, unsigned& minor, string& fstype) {
    bool found = false;
    size_t pos = 0;
    while (pos < table.size()) {
        size_t end = table.find('\n', pos);
        if (end == string::npos) end = table.size();
        const string line = table.substr(pos, end - pos);
        pos = end + 1;
        // id parent major:minor root mountpoint options [optional fields] - fstype source superoptions
        size_t f[6];
        size_t at = 0;
        bool bad = false;
        for (int i = 0; i < 6 && !bad; ++i) {
            f[i] = at;
            const size_t sp = line.find(' ', at);
            if (sp == string::npos) bad = true; else at = sp + 1;
        }
        if (bad) continue;
        auto field = [&](int i) { const size_t e = line.find(' ', f[i]); return line.substr(f[i], e == string::npos ? string::npos : e - f[i]); };
        string mp;
        const string raw = field(4);
        for (size_t i = 0; i < raw.size(); ++i) {
            if (raw[i] == '\\' && i + 3 < raw.size() && raw[i + 1] >= '0' && raw[i + 1] <= '7' && raw[i + 2] >= '0' && raw[i + 2] <= '7' && raw[i + 3] >= '0' && raw[i + 3] <= '7') {
                mp += (char)((raw[i + 1] - '0') * 64 + (raw[i + 2] - '0') * 8 + (raw[i + 3] - '0'));
                i += 3;
            } else mp += raw[i];
        }
        if (mp != point) continue;
        unsigned a = 0, b = 0;
        if (std::sscanf(field(2).c_str(), "%u:%u", &a, &b) != 2) continue;
        const size_t dash = line.find(" - ");
        if (dash == string::npos) continue;
        const size_t ts = dash + 3, te = line.find(' ', ts);
        major = a; minor = b;
        fstype = line.substr(ts, te == string::npos ? string::npos : te - ts);
        found = true;
    }
    return found;
}

// Does a live process serve this FUSE mount, and is it the program that file holds? The AppImage runtime mounts its own file
// through its own copy of FUSE: the process that serves the mount is the runtime, started from that file, and the kernel
// names the mount's connection (fuse_connection: <minor of the mount's device>) in the fdinfo of the process's /dev/fuse
// descriptor. So the link from mount to file is the kernel's: connection number -> server -> its executable's identity.
static bool ServedFromFile(const wstring& procRoot, unsigned minor, const DiskIdent& image) {
    std::error_code ec;
    const string want = "fuse_connection:\t" + std::to_string(minor);
    for (const auto& pid : std::filesystem::directory_iterator(FsPath(procRoot), ec)) {
        const string name = pid.path().filename().string();
        if (name.empty() || name.find_first_not_of("0123456789") != string::npos) continue;
        const wstring base = procRoot + L"\\" + FromUtf8(name);
        DiskIdent exe;
        if (!IdentOf(base + L"\\exe", exe) || !(exe == image)) continue;
        std::error_code ec2;
        for (const auto& fd : std::filesystem::directory_iterator(FsPath(base + L"\\fdinfo"), ec2)) {
            string info;
            if (!ReadText(base + L"\\fdinfo\\" + FromUtf8(fd.path().filename().string()), info, 4096)) continue;
            size_t at = 0;
            while ((at = info.find(want, at)) != string::npos) {
                const size_t after = at + want.size();
                if (at == 0 || info[at - 1] == '\n') {
                    if (after >= info.size() || info[after] == '\n') return true;  // (the whole number: 12 is not 120)
                }
                at = after;
            }
        }
    }
    return false;
}

namespace {
// The file that was proven to be this program's own AppImage, as it was when it was checked.
struct Checked { wstring file; DiskIdent id; };
}

static bool CheckOwn(Checked& out, wstring* why) {
    auto no = [&](const wstring& w) { if (why) *why = w; return false; };
    const wstring env = EnvVar(L"APPIMAGE");
    if (env.empty()) return no(L"WrongGPU is not running from an AppImage here, so there is no file for it to delete. Delete the program file yourself.");
    wstring p = CoreSpelling(env);
    // the environment's own spelling is looked up as it is (the layer's path code would fold letter case, a Linux disk does not)
    std::error_code ec;
    const string envHost = ToUtf8(env);
    const auto status = std::filesystem::symlink_status(envHost, ec);  // (a link is not followed: it is not the file)
    const wstring notPlain = p + L" is not a plain file (it is missing, a folder or a link), so WrongGPU will not delete it.";
    if (ec || !std::filesystem::is_regular_file(status)) return no(notPlain);
    // the file as the disk spells it (a link in a folder above it is followed once, here, and the answer used from now on)
    string imageHost;
    if (!Canonical(envHost, imageHost)) return no(notPlain);
    p = CoreSpelling(FromUtf8(imageHost));
    const wstring name = BaseName(p);
    if (!StartsWithI(name, L"WrongGPU") || !EndsWithI(name, L".AppImage"))
        return no(L"The file is called " + name + L", not WrongGPU...AppImage, so WrongGPU cannot tell it is its own and will not delete it.");
    unsigned char head[16] = {};
    DiskIdent id;
    if (!IdentOf(p, id, head) || !IsAppImageHead(head))
        return no(name + L" is not an AppImage file, so WrongGPU will not delete it.");
    const wstring notStarted = L"WrongGPU cannot prove that " + name + L" is the AppImage this program was started from, so it will not delete it.";
    string appdirHost, selfHost;
    if (EnvVar(L"APPDIR").empty() || !Canonical(ToUtf8(EnvVar(L"APPDIR")), appdirHost) || !Canonical(FsPath(SelfPath()).string(), selfHost) ||
        selfHost.compare(0, appdirHost.size() + 1, appdirHost + "/") != 0)
        return no(L"This running program is not inside " + name + L", so WrongGPU cannot tell the file is the one it was started from and will not delete it.");
    // $APPDIR must be a FUSE mount that the system itself says is served from this very file
    const wstring procRoot = ProcRoot();
    string table;
    unsigned major = 0, minor = 0;
    string fstype;
    DiskIdent selfId;
    if (!ReadText(procRoot + L"\\self\\mountinfo", table) || !MountOf(table, appdirHost, major, minor, fstype) ||
        fstype.compare(0, 5, "fuse.") != 0)
        return no(notStarted + L" (" + FromUtf8(appdirHost) + L" is not a mounted AppImage.)");
    if (!IdentOf(SelfPath(), selfId) || selfId.volume != DevNumber(major, minor))
        return no(notStarted + L" (This program is not on that mount.)");
    if (!ServedFromFile(procRoot, minor, id))
        return no(notStarted + L" (No process started from the file serves the mount.)");
    out.file = p;
    out.id = id;
    return true;
}

// Just before the delete: the path is still a plain file, still the same one that was proven, and still an AppImage.
static bool StillTheSame(const Checked& c, wstring* why) {
    std::error_code ec;
    const auto status = std::filesystem::symlink_status(FsPath(c.file), ec);
    unsigned char head[16] = {};
    DiskIdent now;
    if (ec || !std::filesystem::is_regular_file(status) || !IdentOf(c.file, now, head) || !(now == c.id) || !IsAppImageHead(head)) {
        if (why) *why = BaseName(c.file) + L" changed while WrongGPU was checking it (another file is there now), so WrongGPU will not delete it. Nothing was removed.";
        return false;
    }
    return true;
}

bool OwnAppImage(wstring& file, wstring* why) {
    Checked c;
    if (!CheckOwn(c, why)) return false;
    file = c.file;
    return true;
}

bool UninstallAppImage(Context& ctx, wstring* why, wstring* removedFile, wstring* kept) {
    if (removedFile) removedFile->clear();
    Checked c;
    if (!CheckOwn(c, why)) return false;
    // what can stop the data from going is looked at first: the file goes before the data, so nothing may be left to refuse after it
    const SharedDataLook look = LookAtSharedData(ctx);
    if (look.there && look.anyGame) { if (why) *why = L"a game still has WrongGPU and uses this data. Uninstall WrongGPU from it first. Nothing was removed."; return false; }
    TestFault("appimage-before-delete");  // tests: a file is put in the AppImage's place here
    if (!StillTheSame(c, why)) return false;
    if (!DeleteFileW(c.file.c_str())) {
        if (why) *why = c.file + L" could not be deleted (" + ErrText(GetLastError()) + L"). Nothing was removed.";
        return false;
    }
    if (removedFile) *removedFile = c.file;
    wstring dataWhy;
    if (!RemoveSharedData(ctx, &dataWhy, kept, true)) {
        if (why) *why = L"The AppImage file " + c.file + L" is deleted, but WrongGPU's data was not removed: " + dataWhy;
        return false;
    }
    return true;
}

}  // namespace wg
